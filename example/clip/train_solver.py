import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)


import argparse
from easydict import EasyDict
from tensorboardX import SummaryWriter
import pprint
import time
import datetime
import torch
import prototype.linklink as link
import json
import wandb


from prototype.data.imagenet_dataloader import build_common_augmentation
from tqdm import tqdm
from PIL import Image
import subprocess
from data_process.classification_data import CustomImageNet
from torch.utils.data import DataLoader
import numpy as np

from prototype.solver.base_solver import BaseSolver
from prototype.utils.torch_ddp_dist import get_rank, get_world_size, get_local_rank, set_random_seed, init_ddp, convert_to_ddp_model
from prototype.utils.misc import makedir, create_logger, get_logger, count_params, \
    param_group_all, AverageMeter, accuracy, load_state_optimizer,parse_config
from prototype.utils.ema import EMA
from prototype.model import model_entry
from prototype.optimizer import optim_entry
from prototype.lr_scheduler import scheduler_entry
from prototype.loss_functions import ClipInfoCELoss
from prototype.utils.grad_clip import clip_grad_norm_, clip_grad_value_, clip_param_grad_value_

from prototype.data.datasets.clip_dataset_wsd import get_wds_dataset
from prototype.data.img_cls_dataloader import build_imagenet_test_dataloader


def create_logits(x1, x2, logit_scale=1):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 =  logit_scale * x1 @ x2.t()
    logits_per_x2 =  logit_scale * x2 @ x1.t()
    # print(logits_per_x1.mean().tolist())
    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2

def get_one2many_metrics(preds, name='image_to_text', arg=None):
    metrics = {}
    for k in [1, 5,]:
        metrics[f"{name}_R@{k}"] = np.mean(preds < k)
        metrics[f"{name}_R@{k}_std"] = np.std(preds < k)
    return metrics

class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        ctx.rank = link.get_rank()
        ctx.world_size = link.get_world_size()

        #         y = tensor.new(ctx.world_size, *tensor.size())

        y = [tensor.new(*tensor.size()) for _ in range(ctx.world_size)]

        link.allgather(y, tensor)  # call pytorch all togherer

        y = torch.cat(y, 0).view(-1, *tensor.size())

        return y

    @staticmethod
    def backward(ctx, grad_output):
        in_grad = torch.zeros_like(grad_output)
        in_grad.copy_(grad_output)
        # sum grad for gathered tensor
        link.allreduce(in_grad)
        # split
        return in_grad[ctx.rank]


class EMA_logit_scale():
    def __init__(self, param, threshold):
        self.param = param
        self.buffer = 3.125
        self.momentum = 0.9
        self.threshold = threshold
        self.clip_number = 0

    def update(self):
        self.buffer = self.momentum*self.buffer + \
            (1-self.momentum)*self.param.data

    def clamp(self):
        if (self.param-self.buffer) > self.threshold:
            self.param.data = torch.as_tensor(
                self.buffer+self.threshold, dtype=self.param.dtype, device=self.param.device)
            self.clip_number += 1
        elif (self.buffer-self.param) > self.threshold:
            self.param.data = torch.as_tensor(
                self.buffer-self.threshold, dtype=self.param.dtype, device=self.param.device)
            self.clip_number += 1
        # self.param.data = torch.as_tensor(
        #     3.125, dtype=self.param.dtype, device=self.param.device)




class LipRegManager():
    def __init__(self) -> None:
        self.stored_v = {}
        
    def compute_spectral_norm(self, weight, num_iterations, v=None):
        if v is None:
            v = torch.randn(weight.shape[1])
            v = v / torch.norm(v, 2)  # Normalize the vector

        v = v.to(weight.device)
        u = torch.mv(weight, v)
        u = u / torch.norm(u, 2)  # Normalize u for the first iteration
        for _ in range(num_iterations):
            # Recompute v based on the updated u
            v = torch.mv(weight.t(), u)  # Now, weight.t() matches dimensions with u
            v = v / torch.norm(v, 2)
            # Update u based on the new v
            u = torch.mv(weight, v)
            u = u / torch.norm(u, 2)
        sigma = torch.dot(u, torch.mv(weight, v))  # Approximation of the largest singular value
        return sigma, v

    def layerwise_regularization_with_spectral_norm(self, model, lambda_reg, num_iterations=1, prefix=''):
        regularization_loss = 0.0
        for name, layer in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name  # Construct the full name for nested layers
            if hasattr(layer, 'weight') and layer.weight.dim() == 2:  # Ensure the layer has a 2D weight matrix
                v = self.stored_v.get(full_name, None)
                spectral_norm, updated_v = self.compute_spectral_norm(layer.weight.data, num_iterations, v)
                regularization_loss += lambda_reg * spectral_norm
                self.stored_v[full_name] = updated_v.cpu()
            else:
                # Recursively apply to children modules
                child_loss = self.layerwise_regularization_with_spectral_norm(layer, lambda_reg, num_iterations, full_name)
                regularization_loss += child_loss

        return regularization_loss


class ClsSolver(BaseSolver):
    def __init__(self, args):
        self.args = args
        self.config_file = args.config
        self.prototype_info = EasyDict() #a dict
        self.config = parse_config(self.config_file)
        self.config.data.batch_size = args.batch_size
        #output path
        self.config.output_path = args.output_path


        #---training dataset
        self.config.data.train.batch_size = args.batch_size

        self.setup_env()
        self.build_model()
        self.build_optimizer()
        self.build_data()
        self.build_lr_scheduler()

    def setup_env(self):
        #set random seed

        #set random seed
        set_random_seed()
        self.dist = EasyDict()

        #----get local rank, global rank and world size
        self.dist.local_rank, self.dist.rank, self.dist.world_size = get_local_rank(), get_rank(), get_world_size()
        self.prototype_info.world_size = self.dist.world_size

        #-----set output directories
        self.path = EasyDict()
        self.path.output_path = self.config.output_path

        exp_name = self.args.exp_name if self.args.exp_name else ''
        if self.args.debug: exp_name += "_debug"
        self.path.output_path = os.path.join(self.path.output_path, exp_name)

        self.path.save_path = os.path.join(self.path.output_path, 'checkpoints')
        # self.path.event_path = os.path.join(self.path.output_path, 'events')
        self.path.result_path = os.path.join(self.path.output_path, 'results') #save location as the config_file

        #make local dir
        makedir(self.path.output_path)
        makedir(self.path.save_path)
        # makedir(self.path.event_path)
        makedir(self.path.result_path)
        
        # create logger
        self.path.log_path = os.path.join(self.path.output_path, 'log.txt')
        create_logger(self.path.log_path) #local
        self.logger = get_logger(__name__)

        #--------------
        # add host names
        self.logger.critical(f'config: {pprint.pformat(self.config)}')
        #----------------------

        # create wandb_logger for the main process
        if self.dist.rank == 0 and not self.args.debug:
            self.wandb_logger =  wandb.init(
                project="FDT-Go", 
                name=exp_name,
            )
            #save config file
            config_pth = self.path.output_path + '/config.json'
            with open(config_pth, 'w') as file:
                json.dump(self.config, file)

            self.logger.critical('saved configs to {}'.format(self.config.output_path + '/config.json'))


        ckpt_pth = self.args.ckpt_path
        if ckpt_pth:
            self.state = torch.load(ckpt_pth, map_location='cpu')
            self.logger.info(f"load ckpt from {ckpt_pth}")
        else:
            self.state = {}
            self.state['last_iter'] = 0

        torch.backends.cudnn.benchmark = True

    def build_model(self):


        self.model = model_entry(self.config.model)
        self.prototype_info.model = self.config.model.type
        self.model.cuda()

        count_params(self.model)


        self.model = convert_to_ddp_model(self.model, self.dist.local_rank)



    def build_optimizer(self):
        opt_config = self.config.optimizer
        opt_config.kwargs.lr = self.config.lr_scheduler.kwargs.base_lr
        self.prototype_info.optimizer = self.config.optimizer.type

        # set non weight-decay parameter
        pconfig = {}

        if opt_config.get('no_wd', False):
            pconfig['conv_b'] = {'weight_decay': 0.0}
            pconfig['linear_b'] = {'weight_decay': 0.0}
            pconfig['bn_w'] = {'weight_decay': 0.0}
            pconfig['bn_b'] = {'weight_decay': 0.0}
            pconfig['ln_w'] = {'weight_decay': 0.0}
            pconfig['ln_b'] = {'weight_decay': 0.0}


        if 'pconfig' in opt_config:
            pconfig.update(opt_config['pconfig'])

        param_group, type2num = param_group_all(self.model, pconfig)

        #----optimizer
        opt_config.kwargs.params = param_group
        self.optimizer = optim_entry(opt_config)
        if 'optimizer' in self.state:
            load_state_optimizer(self.optimizer, self.state['optimizer'])


    def all_gather(self, input):
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output

    def build_lr_scheduler(self):
        self.prototype_info.lr_scheduler = self.config.lr_scheduler.type
        if not getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.lr_scheduler.kwargs.max_iter = self.config.data.max_iter
        self.config.lr_scheduler.kwargs.optimizer = self.optimizer
        self.config.lr_scheduler.kwargs.last_iter = self.state['last_iter']
        self.lr_scheduler = scheduler_entry(self.config.lr_scheduler)

    def build_data(self):
        test_config = {}
        self.config.data.last_iter = self.state['last_iter']
        test_config['last_iter'] = self.state['last_iter']
        if getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.data.max_iter = self.config.lr_scheduler.kwargs.max_iter
            test_config['max_iter'] = self.config.lr_scheduler.kwargs.max_iter
        else:
            self.config.data.max_epoch = self.config.lr_scheduler.kwargs.max_epoch
            test_config['max_epoch'] = self.config.lr_scheduler.kwargs.max_epoch

        self.train_data = get_wds_dataset(self.config.data.train, world_size=get_world_size())

        
        self.logger.info('loading test dataset sugar-crepe')
        self.sugar_crepe_image_root = self.config.data.test.sc_image_root
        data_root = self.config.data.test.sc_data_root
        data_dict = {
            'add_obj'    : f'{data_root}/add_obj.json',
            'add_att'    : f'{data_root}/add_att.json',
            'replace_obj': f'{data_root}/replace_obj.json',
            'replace_att': f'{data_root}/replace_att.json',
            'replace_rel': f'{data_root}/replace_rel.json',
            'swap_obj'   : f'{data_root}/swap_obj.json',
            'swap_att'   : f'{data_root}/swap_att.json',
        }
        self.sugar_crepe_dataset = {}
        for c, data_path in data_dict.items():
            self.sugar_crepe_dataset[c] = json.load(open(data_path, 'r', encoding='utf-8'))


    def pre_train(self):
        self.meters = EasyDict()
        self.meters.batch_time = AverageMeter(self.config.saver.print_freq)
        self.meters.step_time = AverageMeter(self.config.saver.print_freq)
        self.meters.data_time = AverageMeter(self.config.saver.print_freq)
        self.meters.losses = AverageMeter(self.config.saver.print_freq)
        self.meters.liploss = AverageMeter(self.config.saver.print_freq)
        self.meters.top1 = AverageMeter(self.config.saver.print_freq)
        self.meters.top5 = AverageMeter(self.config.saver.print_freq)

        self.model.train()

        self.topk = 5
        self.criterion = ClipInfoCELoss()
        if self.args.lipreg > 1e-5:  self.lipregModule = LipRegManager()


    def train(self):
        self.pre_train() #set up for setting
        

        #set training steps by epcoh
        dataloader = self.train_data.dataloader
        each_epoch_step = dataloader.num_batches
        epoch = self.config.data.train.epoch
        total_step = epoch * each_epoch_step

        if self.dist.rank == '0':
            self.logger.critical(
                'total_step: {}'.format(total_step))

        #step
        start_step = self.state['last_iter']
        curr_step = start_step


        end = time.time()
        last_logit_scale = 0
        logit_scale = EMA_logit_scale(self.model.module.logit_scale,
                          self.config.grad_clip.value)
        train_loss = 1e9

        for epoch_id in range(epoch):
            self.train_data.set_epoch(epoch_id) #set epoch for dataloader
            for i, (image, text) in enumerate(dataloader):

                #learing rate scheduler, by steps
                curr_step += 1
                self.lr_scheduler.step(curr_step) #learning rate is calculated based on step
                current_lr = self.lr_scheduler.get_lr()[0]

                # measure data loading time
                self.meters.data_time.update(time.time() - end)

                image = image.cuda()

                # forward
                logits_per_image, logits_per_text = self.model(image, text) #did it fuse multi-gpu>

                # loss
                loss, target = self.criterion(logits_per_image, logits_per_text)
                if self.args.lipreg > 1e-5: 
                    lipLoss = self.lipregModule.layerwise_regularization_with_spectral_norm(self.model, num_iterations=3, lambda_reg=self.args.lipreg)
                    loss += lipLoss
                    
                
                loss /= self.dist.world_size

                # measure accuracy and record loss
                prec1, prec5 = accuracy(
                    logits_per_image, target, topk=(1, self.topk))

                reduced_loss = loss.clone()
                reduced_prec1 = prec1.clone() / self.dist.world_size
                reduced_prec5 = prec5.clone() / self.dist.world_size
                reduced_liploss = torch.tensor(lipLoss) if self.args.lipreg > 1e-5 else torch.zeros(1)

                #update meter
                self.meters.losses.reduce_update(reduced_loss)
                self.meters.top1.reduce_update(reduced_prec1)
                self.meters.top5.reduce_update(reduced_prec5)
                self.meters.liploss.reduce_update(reduced_liploss.cuda())

                #
                if curr_step % 50 == 0:
                    self.logger.info(f'Epoch[{epoch_id+1}] Iter[{curr_step}]: losses.avg:{self.meters.losses.avg:.5f},  liploss.avg:{self.meters.liploss.avg:.5f}, current_lr:{current_lr},  previous_loss:{train_loss:.5f}')

                #loss increase, report crash
                if curr_step > 100 and self.meters.losses.avg > train_loss + 0.5:
                    self.logger.info(f'[ERROR] Training Loss Crashed,lr:{current_lr},prec1:{prec1},curr_step:{curr_step}, meters.losses.avg:{self.meters.losses.avg}')
                else:
                    train_loss = self.meters.losses.avg


                self.optimizer.zero_grad()

                def param_clip_before():
                    if self.config.grad_clip.type == 'constant':
                        self.model.module.logit_scale.requires_grad = False
                    elif self.config.grad_clip.type == 'logit_scale_param':
                        before = self.model.module.logit_scale.data.item()
                    elif self.config.grad_clip.type == 'logit_scale_param_abs_min':
                        self.model.module.logit_scale.data.clamp_(min=self.config.grad_clip.value)
                    elif self.config.grad_clip.type == 'logit_scale_param_value':  # clip param
                        self.model.module.logit_scale.data.clamp_(min=self.config.grad_clip.value, max=self.config.grad_clip.max_value)
                def param_clip_after():
                    if self.config.grad_clip.type == 'logit_scale_param':
                        after = self.model.module.logit_scale.data.item()
                        tem = self.model.module.logit_scale.data
                        if (after-before) > self.config.grad_clip.value:
                            self.model.module.logit_scale.data = torch.as_tensor(
                                before+self.config.grad_clip.value, dtype=tem.dtype, device=tem.device)
                        elif (before-after) > self.config.grad_clip.value:
                            self.model.module.logit_scale.data = torch.as_tensor(
                                before-self.config.grad_clip.value, dtype=tem.dtype, device=tem.device)
                    elif self.config.grad_clip.type == 'logit_scale_param_abs_min':
                        self.model.module.logit_scale.data.clamp_(min=self.config.grad_clip.value)
                    elif self.config.grad_clip.type == 'logit_scale_param_value':  # clip param
                        self.model.module.logit_scale.data.clamp_(min=self.config.grad_clip.value, max=self.config.grad_clip.max_value)

                def grad_clip_before():  # before update(optimizer.step)
                    if self.config.grad_clip.type == 'norm':
                        clip_grad_norm_(self.model.parameters(),
                                        self.config.grad_clip.value)
                    elif self.config.grad_clip.type == 'value':
                        clip_grad_value_(self.model.parameters(),
                                         self.config.grad_clip.value)
                    elif self.config.grad_clip.type == 'logit_scale_grad':
                        clip_param_grad_value_(
                            self.model.module.logit_scale, self.config.grad_clip.value)

                param_clip_before()
                link.barrier()

                loss.backward()
                #self.model.sync_gradients()
                grad_clip_before()
                if self.config.get('check_grad', False):
                    self.check_model_and_grad(10)
                self.optimizer.step()
                link.barrier()
                param_clip_after()

                # clamp
                if self.config.grad_clip.type == 'logit_scale_param_ema':
                    logit_scale.clamp()
                    logit_scale.update()

                # measure elapsed time
                self.meters.batch_time.update(time.time() - end)
                # training logger

                #self.logger.info('self.dist.rank', self.dist.rank, 'save log')
                if curr_step % self.config.saver.print_freq == 0 and self.dist.rank == 0:

                    #---------------
                    if not self.args.debug:
                        wandb.log({'loss_all': self.meters.losses.avg})
                        
                        wandb.log({'acc1_train': self.meters.top1.avg})
                        wandb.log({'acc5_train': self.meters.top5.avg})
                        wandb.log({'lr': current_lr})
                        wandb.log({'logit_scale_exp':self.model.module.logit_scale.exp()})
                        wandb.log({'logit_scale':self.model.module.logit_scale.item()})
                        # ---------------
                        wandb.log({'delta_logit_scale':self.model.module.logit_scale.item()-last_logit_scale})
                        if self.model.module.logit_scale.grad is not None:
                            wandb.log({'logit_scale_grad':self.model.module.logit_scale.grad.item()})

                        wandb.log({'clip_number': logit_scale.clip_number})

                    remain_secs = (total_step - curr_step) * \
                        self.meters.batch_time.avg
                    remain_time = datetime.timedelta(seconds=round(remain_secs))
                    finish_time = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(time.time()+remain_secs))
                    log_msg = f'Iter: [{curr_step}/{total_step}]\t' \
                        f'Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t' \
                        f'Data {self.meters.data_time.val:.3f} ({self.meters.data_time.avg:.3f})\t' \
                        f'Loss_all {self.meters.losses.val:.4f} ({self.meters.losses.avg:.4f})\t' \
                        f'Prec@1 {self.meters.top1.val:.3f} ({self.meters.top1.avg:.3f})\t' \
                        f'Prec@5 {self.meters.top5.val:.3f} ({self.meters.top5.avg:.3f})\t' \
                        f'LR {current_lr:.4f}\t' \
                        f'logit_scale_exp {float(self.model.module.logit_scale.exp()):.4f}\t' \
                        f'logit_scale {float(self.model.module.logit_scale):.4f}\t' \
                        f'delta_logit_scale {float(self.model.module.logit_scale-last_logit_scale):.4f}\t' \
                        f'clip_number {logit_scale.clip_number:.1f}\t' \
                        f'Remaining Time {remain_time} ({finish_time})'
                    self.logger.critical(log_msg)
                last_logit_scale = self.model.module.logit_scale.clone()


                if curr_step > 0 and curr_step % 6000 == 0:
                    self.sugar_crepe_evaluate()
                    # self.imagenet_evaluate()

                #save ckpt---
                if curr_step > 0 and (curr_step % self.config.saver.save_freq == 0 or curr_step == total_step):
                    # save ckpt when at save_freq or the last step !!!
                    if self.dist.rank == 0:
                        if self.config.saver.save_many:
                            ckpt_name = f'{self.path.save_path}/ckpt_{curr_step}.pth.tar'
                        else:
                            ckpt_name = f'{self.path.save_path}/ckpt.pth.tar'
                        self.state['model'] = self.model.state_dict()
                        self.state['optimizer'] = self.optimizer.state_dict()
                        self.state['last_iter'] = curr_step
                        torch.save(self.state, ckpt_name)

                        if curr_step % (self.config.saver.save_freq*10) == 0:
                            self.logger.info('save model kth')
                            k_times_save_path = f'{self.path.save_path}_k_times'
                            if not os.path.exists(k_times_save_path):
                                os.makedirs(k_times_save_path)
                            ckpt_name = f'{k_times_save_path}/ckpt_{curr_step}.pth.tar'
                            torch.save(self.state, ckpt_name)

                end = time.time()
                if curr_step > total_step:
                    break


    @torch.no_grad()
    def sugar_crepe_evaluate(self):
        transform = build_common_augmentation(self.config.data.train.transforms)
        metrics = {}
        if self.dist.rank == 0:
            self.model.eval()
            for c, data_dict in self.sugar_crepe_dataset.items():
                correct_cnt = 0
                for i, data in tqdm(data_dict.items(), desc=f'evaluating {c}'):
                    image_path = os.path.join(self.sugar_crepe_image_root, data['filename'])
                    image = Image.open(image_path).convert('RGB')
                    
                    
                    pos_text_embedding = self.model.module.encode_text(data['caption'])
                    neg_text_embedding = self.model.module.encode_text(data['negative_caption'])
                    cuda_image = transform(image).unsqueeze(dim=0).cuda()
                    image_embedding = self.model.module.encode_image(cuda_image)
                    
                    pos_text_embedding /= pos_text_embedding.norm(dim=-1, keepdim=True)
                    neg_text_embedding /= neg_text_embedding.norm(dim=-1, keepdim=True)
                    image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
                    
                    pos_score = pos_text_embedding @ image_embedding.t()
                    neg_score = neg_text_embedding @ image_embedding.t()
                    correct =  1 if pos_score.item() > neg_score.item() else 0
                    correct_cnt += correct
                
                if self.dist.rank == 0:
                    count = len(data_dict)
                    metrics[f"sugar-crepe-{c}"] = correct_cnt / count

            mean_score = torch.mean(torch.tensor(list(metrics.values())))
            self.model.train()
            if not self.args.debug: wandb.log({f'eval/sugar-crepe-mean-score': mean_score})
            for k, v in metrics.items():
                self.logger.critical(f'{k}: {v}')
                if not self.args.debug:
                    wandb.log({f'eval/sugar-crepe-{k}': v})
        
        
     
        
    @torch.no_grad()
    def imagenet_evaluate(self):
        self.model.eval()
        if self.dist.rank == 0:
            if not os.path.exists("/tmp/ILSVRC2012_img_val.tar"):
                print("download ILSVRC2012_img_val.tar")
                cmd = "wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar -P /tmp/"
                arch = subprocess.check_output(cmd,shell=True)
            if not os.path.exists("/tmp/ILSVRC2012_devkit_t12.tar.gz"):
                print("download /tmp/ILSVRC2012_devkit_t12.tar.gz")
                cmd = "wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz -P /tmp/"
                arch = subprocess.check_output(cmd,shell=True)
            transform = build_common_augmentation(self.config.data.train.transforms)
            dataset = CustomImageNet(root="/tmp", split="val", transform=transform)
            all_class_prompts = dataset.get_all_class_prompts()
            
            class_embeddings = self.model.module.encode_text(all_class_prompts)
            eval_dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
            
            all_rank = []
            for image, labels in eval_dataloader:
                image, labels = image.cuda(), labels.cuda()
                image_embedding = self.model.module.encode_image(image)
                image_logits, _ = create_logits(image_embedding,  class_embeddings)
                sorted_idx = torch.argsort(image_logits, dim=-1, descending=True)
                correct_class_indices = (sorted_idx == labels.view(-1, 1)).nonzero(as_tuple=False)[:, 1]
                all_rank += correct_class_indices.tolist()
            metrics = get_one2many_metrics(preds=np.array(all_rank), name="ImageNet")
            for k, v in metrics.items():
                self.logger.critical(f'{k}: {v}')
                if not self.args.debug:
                    wandb.log({f'eval/{k}': v})     

        self.model.train()        
        
        

def main():
    parser = argparse.ArgumentParser(description='Classification Solver')
    parser.add_argument('--config', required=True, type=str)
    #output pt
    parser.add_argument('--output_path', required=True, type=str)

    parser.add_argument('--batch_size', default=128, type=int)
    
    parser.add_argument("--debug", default=False, action='store_true')
    
    parser.add_argument("--lipreg", default=0, type=float)
    
    parser.add_argument("--exp_name", default=None)
    
    parser.add_argument('--ckpt_path', default=None)

    args = parser.parse_args()


    init_ddp()

    solver = ClsSolver(args)
    # evaluate or train
    solver.train()


if __name__ == '__main__':
    main()
