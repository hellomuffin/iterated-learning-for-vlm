import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

import os
import argparse
from easydict import EasyDict
from tensorboardX import SummaryWriter
import pprint
import time
import seaborn as sns
import datetime
from torchvision import transforms
import numpy as np
import torch
import json
import wandb
import prototype.linklink as link
from torchvision.transforms import functional as TF
from PIL import Image


from prototype.solver.base_solver import BaseSolver
from prototype.utils.torch_ddp_dist import get_rank, get_world_size, get_local_rank, set_random_seed, init_ddp, convert_to_ddp_model
from prototype.utils.misc import makedir, create_logger, get_logger, count_params, \
    param_group_all, AverageMeter, accuracy, load_state_optimizer, load_state_model,\
    parse_config
from prototype.model import model_entry
from prototype.optimizer import optim_entry
from prototype.lr_scheduler import scheduler_entry
from prototype.data.datasets.clip_dataset_wsd import get_unshuffled_wds_dataset
from prototype.loss_functions import LabelSmoothCELoss, ClipInfoCELoss
from prototype.utils.grad_clip import clip_grad_norm_, clip_grad_value_, clip_param_grad_value_
import matplotlib.pyplot as plt

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



# Function to reverse the normalization
def reverse_normalize(tensor):
    # First clone the tensor to not do changes in-place
    tensor = tensor.clone()
    # The mean and std are for each of the image's channels
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    tensor.mul_(std).add_(mean)  # Reverse the normalization
    # Clip the values to be between 0 and 1
    tensor = tensor.clamp(0, 1)
    return tensor

class ClsSolver(BaseSolver):
    def __init__(self, args, exp_path, ckpt_iter):
        self.args = args
        self.prototype_info = EasyDict() #a dict
        self.config = parse_config(args.config)
        self.exp_path = exp_path
        self.ckpt_iter = ckpt_iter
        #update config from command lines

        #learning rate
        #self.config.lr_scheduler.kwargs.base_lr = self.args.base_lr
        #self.config.lr_scheduler.kwargs.warmup_lr = self.args.warmup_lr

        #model parameters
        # self.config.model.kwargs.fdt.sd_temperature = self.args.sd_T
        # self.config.model.kwargs.fdt.att_func_type = self.args.att_func_type
        # self.config.model.kwargs.fdt.pool_type = self.args.pool_type
        # self.config.model.kwargs.fdt.sd_num = self.args.sd_num
        # self.config.model.kwargs.fdt.sd_dim = self.args.sd_dim

        #temperature decay schedule
        # self.config.t_decay.org_t = self.args.sd_T
        # self.config.t_decay.sd_T_decay_iter = self.args.sd_T_decay_iter #iteration
        # self.config.t_decay.sd_T_decay_w = self.args.sd_T_decay_w #decay weight
        # self.config.t_decay.sd_T_min = self.args.sd_T_min #min

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
        self.path.output_path = self.exp_path

        self.path.save_path = os.path.join(self.path.output_path, 'continue_checkpoints')
        # self.path.event_path = os.path.join(self.path.output_path, 'events')
        self.path.result_path = os.path.join(self.path.output_path, 'continue_results') #save location as the config_file

        #make local dir
        # makedir(self.path.output_path)
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



        # create tb_logger for the main process
        if self.dist.rank == 0 and not self.args.debug:
            self.wandb_logger =  wandb.init(
                project="FDT", 
                name=os.path.basename(self.exp_path),
                config={
                    "reset_enable": self.config.reset.enable,
                    "reset_steps": self.config.reset.reset_steps,
                })

            #save config file
            config_pth = self.path.output_path + '/config.json'
            with open(config_pth, 'w') as file:
                json.dump(self.config, file)

            self.logger.critical('saved configs to {}'.format(self.config.output_path + '/config.json'))

        ckpt_path = os.path.join(self.exp_path, "checkpoints", f"ckpt_{self.ckpt_iter}.pth.tar")
        self.state = torch.load(ckpt_path, map_location='cpu')
        self.logger.info(f"load ckpt from {ckpt_path}")
        self.last_iter = self.state['last_iter']
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
            self.model.module.unfreeze_all_parameters()


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

        param_group = param_group_all(self.model, pconfig)[0]
        #----optimizer
        opt_config.kwargs.params = param_group
        self.optimizer = optim_entry(opt_config)
        
        
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

        self.train_data = get_unshuffled_wds_dataset(self.config.data.train, world_size=get_world_size())




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



    def train(self):
        self.pre_train() #set up for setting

        dataloader = self.train_data.dataloader
        each_epoch_step = dataloader.num_batches
        epoch = self.config.data.train.epoch
        total_step = epoch * each_epoch_step


        self.logger.info('each_epoch_step: {} total_step: {}'.format(each_epoch_step, total_step))

        start_step = self.state['last_iter']
        curr_step = start_step
        top_scores = torch.full((200, 20), float('-inf')).cuda()
        top_images = torch.zeros((200, 20, 3, 224, 224))
        top_inner_dots = torch.zeros((200, 20, 7, 7))
        
        print("start right now!")
        
        self.model.eval()
        with torch.no_grad():
            starting_num = 0
            for i, (image, text) in enumerate(dataloader):
                t1 = time.time()
                if i < 5: print("text", i, ":", text[i][:20])
                curr_step += 1
                image = image.cuda()
                att_weight, sd_img_ft, full_inner_dot = self.model.module.extract_img_sd_ft(image)  # att_weight: [batch_size, 100]
                
                t2 = time.time()
                for code_idx in range(200):
                    tt1 = time.time()
                    image_weights = att_weight[:, code_idx]
                    combined_scores, indices = torch.topk(torch.cat((top_scores[code_idx], image_weights)), 20)
                    all_images = torch.cat((top_images[code_idx].cuda(), image))
                    all_inner_dots = torch.cat((top_inner_dots[code_idx].cuda(), full_inner_dot[:, :, code_idx].reshape(image.shape[0], 7,7)))
                    tt2 = time.time()
                    top_scores[code_idx] = combined_scores
                    top_images[code_idx] = all_images[indices].cpu()
                    top_inner_dots[code_idx] = all_inner_dots[indices].cpu()
                    tt3 = time.time()
                t3 = time.time()
                starting_num += image.shape[0]
                # if starting_num > 10000000: break

            os.makedirs(f"results/code_visualization_formal/{self.ckpt_iter}", exist_ok=True)
            for code_idx in range(200):
                starting_num = 0
                print("=====================")
                
                
                fig, axes = plt.subplots(2, 20, figsize=(4 * 20, 8))  # Adjust the size as needed
                axes = axes.flatten()
                for num in range(20):
                    target_image = reverse_normalize(top_images[code_idx, num])
                    pil_image = transforms.ToPILImage()(target_image)
                    axes[20 + num].imshow(np.array(pil_image))
                    axes[20 + num].axis('off')  # Hide the axes
                    
                    score_map = top_inner_dots[code_idx, num]
                    heatmap_array_resized = torch.nn.functional.interpolate(score_map.unsqueeze(0).unsqueeze(0),
                                                        size=pil_image.size[::-1],  # PIL and torch have different conventions for image size
                                                        mode='bilinear',
                                                        align_corners=False).squeeze()
                    heatmap_normalized = (heatmap_array_resized - heatmap_array_resized.min()) / (heatmap_array_resized.max() - heatmap_array_resized.min())
                    heatmap_colormap = plt.get_cmap('coolwarm')
                    heatmap_image = heatmap_colormap(heatmap_normalized.numpy())  # This returns RGBA image
                    heatmap_image_pil = TF.to_pil_image(torch.tensor(heatmap_image).permute(2, 0, 1).float())
                    blended_image = Image.blend(pil_image.convert("RGBA"), heatmap_image_pil.convert("RGBA"), alpha=0.6)
                    axes[num].imshow(np.array(blended_image))

                    # sns.heatmap(score_map.cpu().numpy(), cmap="coolwarm",
                    #     vmin=score_map.min(), vmax=score_map.max(),
                    #     xticklabels=False, yticklabels=False, cbar=False,
                    #     ax=axes[num])
                    axes[num].axis('off')
                plt.tight_layout()  # Adjust the layout
                plt.savefig(f"results/code_visualization_formal/{self.ckpt_iter}/code_{code_idx}.png")
                plt.clf()
                print("finish")
        
                        
        

    def txt_train(self):
        self.pre_train() #set up for setting

        dataloader = self.train_data.dataloader
        each_epoch_step = dataloader.num_batches
        epoch = self.config.data.train.epoch
        total_step = epoch * each_epoch_step


        self.logger.info('each_epoch_step: {} total_step: {}'.format(each_epoch_step, total_step))

        start_step = self.state['last_iter']
        curr_step = start_step
        top_scores = torch.full((1000, 20), float('-inf')).cuda()
        top_txts = [["" for _ in range(20)] for _ in range(1000)]
        # top_inner_dots = torch.zeros((1000, 20, 7, 7))
        
        self.model.eval()
        with torch.no_grad():
            starting_num = 0
            for i, (image, text) in enumerate(dataloader):
                if i < 5: print("text", i, ":", text[i][:20])
                curr_step += 1
                att_weight, sd_img_ft, full_inner_dot = self.model.module.extract_txt_sd_ft(text)  # att_weight: [batch_size, 1000]
                
                
                for code_idx in range(1000):
                    image_weights = att_weight[:, code_idx]
                    combined_scores, indices = torch.topk(torch.cat((top_scores[code_idx], image_weights)), 20)
                    all_texts = top_txts[code_idx] + text
                    top_scores[code_idx] = combined_scores
                    for x, idx in enumerate(indices):
                        top_txts[code_idx][x] = all_texts[idx]
                    
                starting_num += image.shape[0]
                # if starting_num > 1000000: break
                
            

            os.makedirs(f"results/code_visualization_formal/{self.ckpt_iter}", exist_ok=True)
            for code_idx in range(1000):
                starting_num = 0
                print("=====================")
                filePath = f"results/code_visualization_formal/{self.ckpt_iter}/code_{code_idx}.txt"
                with open(filePath, "w") as outfile: 
                    for txt in top_txts[code_idx] : outfile.write(f"{txt}\n")
                print("finish")
        
            
def int_list(values):
    """Convert a [-]-separated string into a list of integers."""
    return [int(value) for value in values.split('-')]

def main():
    parser = argparse.ArgumentParser(description='Classification Solver')
    parser.add_argument('--config', required=True, type=str)
    
    parser.add_argument('--output_path', required=True, type=str)

    parser.add_argument('--batch_size', default=128, type=int)

    parser.add_argument('--exp_path', required=True, type=str)
    
    parser.add_argument('--iters', type=int_list, help="A - list of integers. Example: 1,2,3,4")
    
    parser.add_argument("--debug", default=False, action='store_true')

    # parser.add_argument('--sd_num', default=100, type=int) #fdt nums
    # parser.add_argument('--sd_dim', default=512, type=int) #fdt dims
    # parser.add_argument('--att_func_type', default='sparsemax', type=str) #normalization function of attention weights ['sparsemax', 'softmax']
    # parser.add_argument('--pool_type', default='max', type=str) #pooing type attention weights ['max', 'mean']
    #
    #
    # parser.add_argument('--sd_T', default= 1000, type=float) #the tempture parameters of the attention weights
    # parser.add_argument('--sd_T_decay_w', default= 0.3, type=float) #decay ratio of parameters
    # parser.add_argument('--sd_T_decay_iter', default= 2700, type=float) #decay at every sd_T_decay_iter iterations
    # parser.add_argument('--sd_T_min', default= 0.01, type=float) #min value of sd_T
    #
    # parser.add_argument('--base_lr', default= 0.0001, type=float) #inital lr
    # parser.add_argument('--warmup_lr', default= 0.001, type=float) #warmup lr

    args = parser.parse_args()
    print("iters:", args.iters)

    # set up pytorch ddp
    init_ddp()

    for iter in args.iters:
        solver = ClsSolver(args, exp_path=args.exp_path, ckpt_iter=iter)
        # evaluate or train
        solver.txt_train()
        solver.train()
        
    


if __name__ == '__main__':
    main()
