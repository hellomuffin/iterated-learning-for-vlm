import sys
import os
import inspect
from datetime import datetime

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
import os
import argparse
from easydict import EasyDict
import pprint
import time
import datetime
import torch
import json
import prototype.linklink as link
import wandb
from copy import deepcopy
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
    param_group_all, AverageMeter, accuracy, load_state_optimizer, load_state_model,\
    parse_config
from prototype.model import model_entry
from prototype.optimizer import optim_entry
from prototype.lr_scheduler import scheduler_entry
from prototype.data.datasets.clip_dataset_wsd import get_wds_dataset, sample_shard_paths
from prototype.loss_functions import LabelSmoothCELoss, ClipInfoCELoss
from prototype.utils.grad_clip import clip_grad_norm_, clip_grad_value_, clip_param_grad_value_


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








class ClsSolver(BaseSolver):
    def __init__(self, args):
        self.args = args
        self.prototype_info = EasyDict() #a dict
        self.config = parse_config(args.config)

        #output path
        self.config.output_path = args.output_path


        #---training dataset
        self.config.data.train.batch_size = args.batch_size

        #set env
        self.setup_env()
        self.build_model()
        self.build_optimizer()
        self.build_data()
        self.build_lr_scheduler()

    def setup_env(self):
        #---- (1)set ddp env (2) set random seed (3)  set output directories


        #set random seed
        set_random_seed()
        self.dist = EasyDict()

        #----get local rank, global rank and world size
        self.dist.local_rank, self.dist.rank, self.dist.world_size = get_local_rank(), get_rank(), get_world_size()
        self.prototype_info.world_size = self.dist.world_size

        #-----set output directories
        self.path = EasyDict()
        self.path.output_path = self.config.output_path

        ft_cfg = self.config.model.kwargs.fdt
        decay_cfg = self.config.t_decay
        # self.path.output_path = self.path.output_path + \
        #                                  '/sd-num-{}_sd-dim-{}_warmup-lr-{}_pool-{}_sd-T-{}_T-decay-w-{}_T-min-{}_T-iter-{}'.format( \
        #                                      ft_cfg.sd_num, ft_cfg.sd_dim,
        #                                      self.config.lr_scheduler.kwargs.warmup_lr,
        #                                      ft_cfg.pool_type,
        #                                      ft_cfg.sd_temperature, decay_cfg.sd_T_decay_w, decay_cfg.sd_T_min, decay_cfg.sd_T_decay_iter
        #                                  )
        

        # now = datetime.now()
        # date_time_string = now.strftime("_%Y-%m-%d-%H-%M-%S")
        exp_name = self.args.exp_name if self.args.exp_name else ''
        exp_name += f'_Reset_{self.config.reset.enable}_steps_{self.config.reset.reset_steps}_smooth_{self.config.reset.smooth_steps}'
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
                config={
                    "reset_enable": self.config.reset.enable,
                    "reset_steps": self.config.reset.reset_steps,
                })

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
        
        if 'model' in self.state:
            self.logger.info(f"load model from checkpoint")
            load_state_model(self.model, self.state['model'])
    
    def store_codebook_value(self):
        self.stored_codebook = self.model.module.space_dict.data.clone()
    
    def keep_codebook_value(self):
        self.model.module.space_dict.data = self.stored_codebook

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

        # split parameters to different group, and for different groups, using different paramters
        if 'pconfig' in opt_config:
            pconfig.update(opt_config['pconfig'])

        # vision optimizer
        param_group = param_group_all(self.model, pconfig)[0]
        opt_config.kwargs.params = param_group
        self.optimizer = optim_entry(opt_config)
        
        
    def build_lr_scheduler(self):
        
        self.prototype_info.lr_scheduler = self.config.lr_scheduler.type
        if not getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.lr_scheduler.kwargs.max_iter = self.config.data.max_iter
        self.config.lr_scheduler.kwargs.last_iter = self.state['last_iter']
        self.config.lr_scheduler.kwargs.reset_steps = self.config.reset.reset_steps
        
        self.config.lr_scheduler.kwargs.optimizer = self.optimizer
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

        self.big_train_data = get_wds_dataset(self.config.data.train, world_size=get_world_size())
        self.train_data = self.big_train_data
        
        
        self.logger.info('loading test dataset sugar-crepe')
        
        # TODO: load from config
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


        
    def get_random_subset_data(self):
        subset_shard_list = sample_shard_paths(
            total_shards=self.config.data.train.num_shards,
            sample_factor=10
        )
        subset_train_config = deepcopy(self.config.data.train)
        subset_train_config.data_path = subset_shard_list
        subset_train_config.num_shards = len(subset_shard_list)
        subset_train_config.num_samples = len(subset_shard_list) * 1000
        subset_train_data = get_wds_dataset(subset_train_config, world_size=get_world_size())
        return subset_train_data
        

    def pre_train(self):
        self.meters = EasyDict()
        self.meters.batch_time = AverageMeter(self.config.saver.print_freq)
        self.meters.step_time = AverageMeter(self.config.saver.print_freq)
        self.meters.data_time = AverageMeter(self.config.saver.print_freq)
        self.meters.losses = AverageMeter(self.config.saver.print_freq)
        self.meters.top1 = AverageMeter(self.config.saver.print_freq)
        self.meters.top5 = AverageMeter(self.config.saver.print_freq)

        self.model.train()

        self.num_classes = self.config.model.kwargs.get('num_classes', 1000)
        self.topk = 5 if self.num_classes >= 5 else self.num_classes
        self.criterion = ClipInfoCELoss()
        
        self.model.module.find_always_freeze_weight()



    def train(self):
        self.pre_train() #set up for setting

        each_epoch_step = self.train_data.dataloader.num_batches
        epoch = self.config.data.train.epoch
        total_step = epoch * each_epoch_step


        self.logger.info('each_epoch_step: {} total_step: {}'.format(each_epoch_step, total_step))

        start_step = self.state['last_iter']
        curr_step = all_step = start_step
        end = time.time()
        last_logit_scale = 0
        logit_scale = EMA_logit_scale(self.model.module.logit_scale,
                          self.config.grad_clip.value)
        train_loss = 1e9

        print("begin training")
        self.best_composition_score = {}
        for epoch_id in range(epoch):
            self.train_data.set_epoch(epoch_id) #set epoch

            for i, (image, text) in enumerate(self.train_data.dataloader):
                curr_step += 1
                all_step += 1
                self.lr_scheduler.step(curr_step) #learning rate is calculated based on step           
                
                if curr_step % self.config.t_decay.sd_T_decay_iter == 0:

                    sd_T = self.config.t_decay.org_t #get orginal temperature
                    sd_T_decay_w = self.config.t_decay.sd_T_decay_w #get decay weight
                    sd_T_decay_iter = self.config.t_decay.sd_T_decay_iter #get decay iter
                    sd_T_min = self.config.t_decay.sd_T_min #get min temperature

                    temperature = sd_T * (sd_T_decay_w ** (curr_step / sd_T_decay_iter))
                    temperature = max(temperature, sd_T_min)

                    self.model.module.img_query_model.temperature = temperature
                    self.model.module.txt_query_model.temperature = temperature


                current_lr = self.lr_scheduler.get_lr()[0]
                # measure data loading time
                self.meters.data_time.update(time.time() - end)



                #self.logger.info('self.dist.rank', self.dist.rank, 'forward done')
                def param_clip_before():
                    if self.config.grad_clip.type == 'constant':
                        self.model.module.logit_scale.requires_grad = False
                    elif self.config.grad_clip.type == 'logit_scale_param':
                        before = self.model.module.logit_scale.data.item()
                    elif self.config.grad_clip.type == 'logit_scale_param_abs_min':
                        self.model.module.logit_scale.data.clamp_(min=self.config.grad_clip.value)
                    elif self.config.grad_clip.type == 'logit_scale_param_value':  # clip param
                        self.model.module.logit_scale.data.clamp_(min=self.config.grad_clip.value, max=self.config.grad_clip.max_value)
                        #for sd
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
                        #for sd


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
                def grad_clip_after():
                    pass
                
                image = image.cuda()
                
                logit_sd, _ = self.model(image, text) #did it fuse multi-gpu>

                loss, target = self.criterion(logit_sd[0], logit_sd[1])
                loss = loss / self.dist.world_size
                prec1, prec5 = accuracy(logit_sd[0], target, topk=(1, self.topk))

                #self.logger.info('self.dist.rank', self.dist.rank, 'clip grad begin')
                self.optimizer.zero_grad()
                param_clip_before()
                link.barrier()


                loss.backward()
                #self.model.sync_gradients()
                grad_clip_before()
                if self.config.get('check_grad', False):
                    self.check_model_and_grad(10)
                self.optimizer.step()
                
                
                grad_clip_after()
                link.barrier()
                param_clip_after()
                
                #-----update meter
                reduced_loss = loss.clone()
                reduced_prec1 = prec1.clone() / self.dist.world_size
                reduced_prec5 = prec5.clone() / self.dist.world_size


                self.meters.losses.reduce_update(reduced_loss)
                self.meters.top1.reduce_update(reduced_prec1)
                self.meters.top5.reduce_update(reduced_prec5)


                if curr_step % 50 == 0:
                    self.logger.info(f'Epoch[{epoch_id+1}] Iter[{curr_step}]: '
                                     f'losses.avg:{self.meters.losses.avg:.5f},  '
                                     f'current_lr:{current_lr},  previous_loss:{train_loss:.5f} '
                                     f'temperature:{self.model.module.img_query_model.temperature:.5f}'
                                     )

                #loss increase, report crash
                # if curr_step > 100 and self.meters.losses.avg > train_loss + 0.5:
                #     resume = True
                #     self.logger.info(f'[ERROR] Training Loss Crashed,lr:{current_lr},prec1:{prec1},curr_step:{curr_step}, meters.losses.avg:{self.meters.losses.avg}')
                # else:
                #     train_loss = self.meters.losses.avg
                

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

                #self.logger.info('self.dist.rank', self.dist.rank, 'save done')


                if curr_step > 0 and curr_step % 6000 == 0:
                    self.sugar_crepe_evaluate()
                    # self.imagenet_evaluate()
                    
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

                    
                link.barrier()
                
                if self.config.reset.enable and curr_step > self.config.reset.reset_steps and curr_step < self.config.reset.reset_steps * self.config.reset.reset_nums:
                    if curr_step % self.config.reset.reset_steps == 0: self.model.module.reset_text_encoder()
                    if curr_step % self.config.reset.reset_steps < self.config.reset.smooth_steps:
                        if curr_step == start_step + 1:  # just in case the model is just resumed from a checkpoint 
                            self.store_codebook_value()
                            self.model.module.reset_text_encoder()
                            if self.dist.rank == 0: self.logger.info(f"step {curr_step}: reset text encoder")
                        else:
                            self.keep_codebook_value()
                            if self.dist.rank == 0: self.logger.info(f"step {curr_step}: keep codebook value")
                    if curr_step % self.config.reset.reset_steps == self.config.reset.smooth_steps:
                        self.model.module.freeze_unfreeze_vision_weights(unfreeze=True, freeze_codebook=False)
                        if self.dist.rank == 0: self.logger.info(f"step {curr_step}:unfreeze vision encoder")
                end = time.time()
                # if curr_step > total_step:
                #     break
                
                
    

    def vision_distillation(self, train_data, total_distil_steps):
        # freeze text part
        teacher_model = model_entry(self.config.model).cuda()
        teacher_model.load_state_dict(self.model.module.state_dict())
        
        self.model.module.swap_vision_encoder()
        # self.model.module.reset_codebook()  # TODO: if we want to reset the codebook
        self.model.module.freeze_unfreeze_text_weights(unfreeze=False, freeze_codebook=True) # TODO: if we want to freeze it?
        
        # build optimizer
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.module.parameters()), lr=self.config.distil_lr_scheduler.kwargs.base_lr)
        
        # set up learning rate scheduler
        lr_scheduler_config = self.config.distil_lr_scheduler
        lr_scheduler_config.kwargs.optimizer = optimizer
        lr_scheduler = scheduler_entry(lr_scheduler_config)

        distil_step = 0
        epoch_id = 0
        finish_distil = False
        while not finish_distil:
            epoch_id += 1
            train_data.set_epoch(epoch_id) #set epoch
            for i, (image, _) in enumerate(train_data.dataloader):
                if distil_step > total_distil_steps: 
                    finish_distil = True
                    break
                distil_step += 1
                lr_scheduler.step(distil_step) #learning rate is calculated based on step
                current_lr = self.lr_scheduler.get_lr()[0]

                
                image = image.cuda()
                _, sd_img_ft, _ =  self.model.module.extract_img_sd_ft(image)
                with torch.no_grad():  _, sd_img_ft_t, _ =  teacher_model.extract_img_sd_ft(image)
                
                # Compute norms of A and B
                norm_sd_img_ft = sd_img_ft / (sd_img_ft.norm(dim=-1, keepdim=True) + 1e-10)
                norm_sd_img_ft_t = sd_img_ft_t / (sd_img_ft_t.norm(dim=-1, keepdim=True) + 1e-10)
                cosine_similarity = torch.sum(norm_sd_img_ft * norm_sd_img_ft_t, dim=1)
                
                loss = - torch.mean(cosine_similarity)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #self.logger.info('self.dist.rank', self.dist.rank, 'save log')
                if distil_step % self.config.saver.print_freq == 0 and self.dist.rank == 0 and not self.args.debug:
                    wandb.log({'loss_distil': loss})
                if self.args.debug:
                    print("distillation step", distil_step, "loss_distil", loss.item())
                    

        self.model.module.freeze_unfreeze_text_weights(unfreeze=True, freeze_codebook=False) 


    @torch.no_grad()
    def sugar_crepe_evaluate(self):
        prev_best_score = deepcopy(self.best_composition_score)
        transform = build_common_augmentation(self.config.data.train.transforms)
        metrics = {}
        self.model.eval()
        cnt = 0
        for c, data_dict in self.sugar_crepe_dataset.items():
            correct_cnt = 0
            cnt += 1
            for i, data in tqdm(data_dict.items(), desc=f'evaluating {c}'):
                image_path = os.path.join(self.sugar_crepe_image_root, data['filename'])
                image = Image.open(image_path).convert('RGB')
                
                _, pos_text_embedding, _ = self.model.module.extract_txt_sd_ft(data['caption'])
                _, neg_text_embedding, _ = self.model.module.extract_txt_sd_ft(data['negative_caption'])
                cuda_image = transform(image).unsqueeze(dim=0).cuda()
                _, image_embedding, _ = self.model.module.extract_img_sd_ft(cuda_image)
                
                pos_text_embedding /= pos_text_embedding.norm(dim=-1, keepdim=True)
                neg_text_embedding /= neg_text_embedding.norm(dim=-1, keepdim=True)
                image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
                
                pos_score = pos_text_embedding @ image_embedding.t()
                neg_score = neg_text_embedding @ image_embedding.t()
                correct =  1 if pos_score.item() > neg_score.item() else 0
                correct_cnt += correct
            
            count = len(data_dict)
            metrics[f"sugar-crepe-{c}"] = correct_cnt / count
            
        self.model.train()
        
        
        curr_mean_score = torch.mean(torch.tensor(list(metrics.values())))
    
        if len(prev_best_score) == 0: self.best_composition_score = metrics
        else:
            prev_mean_score = torch.mean(torch.tensor(list(prev_best_score.values())))
            print("rank", self.dist.rank, "says: curr mean score", curr_mean_score, "prev mean score", prev_mean_score)
            if curr_mean_score + 0.003 < prev_mean_score: return False
            # for k, v in metrics.items():
            #     prev_v = prev_best_score[k]
            #     if v + 0.03 < prev_v: return False
            self.best_composition_score = metrics
            
        
        if self.dist.rank == 0:
            # if not self.args.debug: wandb.log({f'eval/sugar-crepe-mean-score': curr_mean_score})
            if not self.args.debug:
                wandb.log({f'eval/sugar-crepe-mean-score': curr_mean_score})
            for k, v in metrics.items():
                self.logger.critical(f'{k}: {v}')
                if not self.args.debug:
                    wandb.log({f'eval/sugar-crepe-{k}': v})
        return True
    
        
     
        
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
            
            _, class_embeddings, _ = self.model.module.extract_txt_sd_ft(all_class_prompts)
            eval_dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
            
            all_rank = []
            for image, labels in eval_dataloader:
                image, labels = image.cuda(), labels.cuda()
                _, image_embedding, _ = self.model.module.extract_img_sd_ft(image)
                image_logits, _ = create_logits(image_embedding,  class_embeddings)
                sorted_idx = torch.argsort(image_logits, dim=-1, descending=True)
                correct_class_indices = (sorted_idx == labels.view(-1, 1)).nonzero(as_tuple=False)[:, 1]
                all_rank += correct_class_indices.tolist()
            metrics = get_one2many_metrics(preds=np.array(all_rank), name="ImageNet")
            for k, v in metrics.items():
                self.logger.critical(f'{k}: {v}')
                if not self.args.debug:
                    wandb.log({f'eval/{k}': v})     
        link.barrier()
        self.model.train()        
        
        
        



def main():
    parser = argparse.ArgumentParser(description='Classification Solver')
    parser.add_argument('--config', required=True, type=str)
    #output pt
    parser.add_argument('--output_path', required=True, type=str)

    parser.add_argument('--batch_size', default=128, type=int)
    
    parser.add_argument("--debug", default=False, action='store_true')
    
    parser.add_argument("--exp_name", default=None)
    
    parser.add_argument('--ckpt_path', default=None)

    args = parser.parse_args()

    # set up pytorch ddp
    init_ddp()

    solver = ClsSolver(args)
    # evaluate or train
    if solver.config.data.last_iter < solver.config.data.max_iter:
        solver.train()
    else:
        solver.logger.info('Training has been completed to max_iter!')


if __name__ == '__main__':
    main()
