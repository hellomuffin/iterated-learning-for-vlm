import torch
from torch import nn
import numpy as np
import math
from .image_encoder.visual_transformer import visual_transformer_B32, visual_transformer_B16
from .text_encoder.text_transformer import text_transformers
from .sparsemax import Sparsemax
import prototype.linklink as link
from .swin.models import build_swin_model
import yaml
from easydict import EasyDict
from copy import deepcopy


import torch.nn.utils.spectral_norm as spectral_norm


def apply_spectral_norm_to_linear_layers(model):
    """
    Recursively apply spectral normalization to all linear layers in the given model.

    Parameters:
    - model: The PyTorch model to modify.
    """
    for name, module in model.named_children():
        # Check if the module is a linear layer
        if isinstance(module, nn.Linear) and name != 'out_proj':
            # Wrap the linear layer with spectral normalization
            print("set spectrum", name, module)
            setattr(model, name, spectral_norm(module))
        else:
            # Recursively apply to children modules
            apply_spectral_norm_to_linear_layers(module)
            
            




def weight_reset(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.ConvTranspose2d)):
        m.reset_parameters()
    elif isinstance(m, torch.nn.BatchNorm2d):
        m.reset_parameters() 
        m.running_mean.zero_()
        m.running_var.fill_(1)
    elif isinstance(m, torch.nn.LayerNorm):
        m.reset_parameters()




#---- attention models for FDT
class Query_model(nn.Module):
    def __init__(self, ft_dim, sd_dim, temperature=1, att_func_type='softmax', pool_type='sum'):
        '''
        ft_dim: feature dim of image patch or text token
        sd_dim: dim of FDT
        temperature: temperature for softmax or sparsemax
        att_func_type: attention normlization function type
        pool_type: pooling type for attention weights
        '''

        super().__init__()

        #activation 
        assert att_func_type in ['softmax', 'sigmoid', 'sparsemax']
        self.att_func_type = att_func_type

        assert pool_type in ['mean', 'max', 'sum']
        self.pool_type = pool_type

        if self.att_func_type == 'softmax':
            self.att_activation = nn.Softmax(dim=-1)
        elif self.att_func_type == 'sparsemax':
            self.att_activation = Sparsemax(dim=-1)
        else:
            self.att_activation = nn.Sigmoid()

        self.att_dim = sd_dim
        self.temperature = temperature
        
        #map patch/text tokens to codebook (query) spaces
        #---note that we donot use mapping for FDT

        self.q_map = nn.Sequential(
            nn.LayerNorm(ft_dim),
            nn.Linear(ft_dim, sd_dim),
            nn.GELU(),
            nn.LayerNorm(sd_dim),
            nn.Linear(sd_dim, sd_dim)
        )



    def forward(self, ft, sd, mask=None, return_token_att=False):


        '''
        Args:
            ft: [batch, token_num, ft_dim]
            sd: [FDT_num, sd_dim]
            mask: [batch, token_num]: mask for padded tokens.
            return_token_att: flag for returning attention weights before nomalization.
            used for visualizing FDT.
        Returns:

        '''

        #map image/text token to query space
        q = self.q_map(ft) #bacth, token_num, dim

        k = sd #code_num, sd_dim
        k = k.unsqueeze(0) #[1, code_num, sd_dim]
        k = k.transpose(2, 1) #[1,sd_dim, sd_num]
        
        #-----calculate inner dot
        inner_dot = torch.matmul(q, k) #[bacth, token_num, code_num]

        if return_token_att: #cosine sim
            token_att = inner_dot

        inner_dot = inner_dot / math.sqrt(self.att_dim) #scale dot norm

        if mask is not None: # mask paded tokens
            
            assert mask.shape == q.shape[:2]
            mask = (mask == 0) * 1 #0 --> 1, inf --> 0

            inner_dot = inner_dot * mask.unsqueeze(-1) #sigmod(-inf) = 0, softmax(-inf) = 0

            if return_token_att: #if has pad, return maksed
                token_att = inner_dot


        # temptural norm
        inner_dot = inner_dot / self.temperature #[bacth, token_num, code_num]

        #pooling
        if self.pool_type == 'sum':
            inner_dot = inner_dot.sum(1) #mean poolings
        elif self.pool_type == 'mean':
            inner_dot = inner_dot.mean(1)
        else:
            inner_dot = inner_dot.max(1)[0]

        
        #----get attention weights
        att_weight = self.att_activation(inner_dot) #normaliztion

        #----calculate weighted sum of v
        #v = self.ln_v(ft) #map to v_space
        
        att_ft = att_weight @ sd  #[bacth, dictory_size] * [dictory_size, dim]  ---> [bacth, sd_num, dim]

        if self.att_func_type == 'sigmoid':
            att_ft = att_ft / att_weight.sum(dim=-1, keepdim=True)
        
        if return_token_att:
            return token_att, att_ft, sd
        return att_weight, att_ft, sd
        # return full_inner_dot, att_ft, att_weight

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




class Clip_FDT(nn.Module):
    def __init__(self, image_encode, text_encode, use_allgather, sd_num, sd_dim, raw_img_ft_dim, raw_txt_ft_dim, att_func_type, pool_type, sd_temperature):
        super().__init__()
        '''
        Args:
            image_encode: image encoder
            text_encode: text encoder
            use_allgather: flag for using allgather for calculating infoNCE loss
            sd_num: number of FDT
            sd_dim: dimension of FDT
            raw_img_ft_dim: dimension of patch features
            raw_txt_ft_dim: dimension of text token features
            att_func_type: attention function type
            pool_type: pooling type of FDT attention weights
            sd_temperature: temperature for FDT attention
        '''
        self.use_allgather = use_allgather
        self.visual = image_encode
        self.encode_text = text_encode


        #learnable FDT
        self.space_dict = nn.Parameter(torch.randn(sd_num, sd_dim))

        #query mapping
        self.img_query_model = Query_model(ft_dim=raw_img_ft_dim, sd_dim=sd_dim, temperature=sd_temperature, att_func_type=att_func_type, pool_type=pool_type)
        self.txt_query_model = Query_model(ft_dim=raw_txt_ft_dim, sd_dim=sd_dim, temperature=sd_temperature, att_func_type=att_func_type, pool_type=pool_type)

        # # storing the state dict for text encoder
        # self.encode_text_init_weight = self.encode_text.state_dict()
        # self.txt_query_model_init_weight = self.txt_query_model.state_dict()
        
        #learnable temperature for infoNCE loss
        self.logit_scale = nn.Parameter(torch.ones([1]))
        self.logit_scale_sd = nn.Parameter(torch.ones([1]))
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        nn.init.constant_(self.logit_scale_sd, np.log(1 / 0.07))
        # nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
        
        self.stored_vision_encoder_weight = None

    

    @property
    def dtype(self):
        try:
            return self.visual.conv1.weight.dtype
        except:
            try:
                return self.visual.head.weight.dtype
            except:
                try:
                    return self.visual.stem[0].weight.dtype
                except:
                    return self.encode_text.text_projection.weight.dtype
             
            
    def find_always_freeze_weight(self):
        self.weight_always_freeze = []
        for name, param in self.named_parameters():
            if param.requires_grad == False: self.weight_always_freeze.append(name)
        print("always freeze weight", self.weight_always_freeze)
    
    def reset_text_encoder(self, seed):
        # self.encode_text.load_state_dict(self.encode_text_init_weight)
        # self.txt_query_model.load_state_dict(self.txt_query_model_init_weight)
        torch.manual_seed(seed) 
        self.encode_text.apply(weight_reset)
        self.txt_query_model.apply(weight_reset)
        

    def reset_vision_encoder(self):
        self.visual.apply(weight_reset)
        self.img_query_model.apply(weight_reset)   
        
    def swap_vision_encoder(self):
        current_vision_weights = (self.visual.state_dict().copy(), self.img_query_model.state_dict().copy())
        
        if self.stored_vision_encoder_weight == None:
            self.visual.apply(weight_reset)
            self.img_query_model.apply(weight_reset) 
        else:
            self.visual.load_state_dict(self.stored_vision_encoder_weight[0])
            self.img_query_model.load_state_dict(self.stored_vision_encoder_weight[1])
        
        self.stored_vision_encoder_weight = current_vision_weights
    
    
    def reset_codebook(self):
        sd_num, sd_dim = self.space_dict.shape
        self.space_dict.data = torch.randn(sd_num, sd_dim)
        
    def unfreeze_weights(self, module_names, freeze_codebook=False):
        for encoder_name in module_names:
            encoder = getattr(self, encoder_name)
            for name, param in encoder.named_parameters():
                if name not in self.weight_always_freeze: param.requires_grad = True
                else: print("not unfreezing parameter named", name )
        self.logit_scale.requires_grad = True
        self.logit_scale_sd.requires_grad = True
        if freeze_codebook: self.space_dict.requires_grad = False
        else: self.space_dict.requires_grad = True
        
    def unfreeze_all_parameters(self):
        for name, param in self.named_parameters():
            param.requires_grad = True

        
    def freeze_weights(self, module_names, freeze_codebook=False):
        for encoder_name in module_names:
            encoder = getattr(self, encoder_name)
            for name, param in encoder.named_parameters():
                param.requires_grad = False
        self.logit_scale.requires_grad = False
        self.logit_scale_sd.requires_grad = False
        if freeze_codebook: self.space_dict.requires_grad = False
        else: self.space_dict.requires_grad = True
        
    
    def freeze_unfreeze_vision_weights(self, unfreeze, freeze_codebook):
        module_names = ["visual", "img_query_model"]
        if unfreeze: self.unfreeze_weights(module_names, freeze_codebook)
        else: self.freeze_weights(module_names, freeze_codebook)
        
        
    def freeze_unfreeze_text_weights(self, unfreeze, freeze_codebook):
        module_names = ["encode_text", "txt_query_model"]
        if unfreeze: self.unfreeze_weights(module_names, freeze_codebook)
        else: self.freeze_weights(module_names, freeze_codebook)
    

    def extract_img_sd_ft(self, images, return_token_att=False):

        #etract image represenation
        img_info = self.encode_image(images) #[bacth, dim
        if len(img_info) == 3:
            img_ft, patch_ft, raw_img_ft = img_info
        else:
            img_ft, patch_ft = img_info #for swin, only return 2 features

        #print(patch_ft.shape)

        full_inner_dot, sd_img_ft, att_weight = self.img_query_model(patch_ft, self.space_dict, return_token_att=return_token_att)

        #print(img_ft.shape, sd_img_ft.shape, att_weight.shape)
        return full_inner_dot, sd_img_ft, att_weight


    def extract_patch_ft(self, images):


        #etract image represenation
        img_info = self.encode_image(images) #[bacth, dim
        if len(img_info) == 3:
            img_ft, patch_ft, raw_img_ft = img_info
        else:
            img_ft, patch_ft = img_info #for swin, only return 2 features

        patch_ft = self.img_query_model.q_map(patch_ft)


        return patch_ft


    def extract_word_ft(self, texts):

        #extract word embedingd
        txt_ft, word_ft, raw_txt_ft, pad_mask = self.encode_text(texts, return_dense=True, 
        return_padmask=True, return_raw_feature=True) #[bacth, dim]

        word_ft = self.txt_query_model.q_map(word_ft) #project

        return word_ft, pad_mask

    def extract_txt_sd_ft(self, texts, return_token_att=False, raw_text=True):

        #extract word embedingd
        txt_ft, word_ft, raw_txt_ft, pad_mask = self.encode_text(texts, return_dense=True, 
        return_padmask=True, return_raw_feature=True, raw_text=raw_text) #[bacth, dim]

        full_inner_dot, sd_img_ft, att_weight = self.txt_query_model(word_ft, self.space_dict, mask=pad_mask, return_token_att=return_token_att)

        #print(txt_ft.shape, sd_txt_ft.shape, att_weight.shape)
        return full_inner_dot, sd_img_ft, att_weight


    def encode_image(self, image):
        return self.visual(image.type(self.dtype), return_dense=True, return_raw_feature=True)


    def all_gather(self, input):
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output

    

    def forward(self, images, texts):
        # input

        #etract image represenation
        
        img_output = self.encode_image(images) #[bacth, dim]
        if len(img_output) == 3:
            img_ft, patch_ft, raw_img_ft = img_output
        else:
            img_ft, patch_ft = img_output #for swin, only return 2 features
            

        #extract text represenation
        txt_ft, word_ft, raw_txt_ft, pad_mask = self.encode_text(texts, return_dense=True, 
        return_padmask=True, return_raw_feature=True) #[bacth, dim]

        #calculate FDT-based features
        sd_img_att_weight, sd_img_ft, img_k = self.img_query_model(patch_ft, self.space_dict)
        sd_txt_att_weight , sd_txt_ft, txt_k = self.txt_query_model(word_ft, self.space_dict, mask=pad_mask)

        #l2 normalization
        sd_img_ft = sd_img_ft / (sd_img_ft.norm(dim=-1, keepdim=True) + 1e-10)
        sd_txt_ft = sd_txt_ft / (sd_txt_ft.norm(dim=-1, keepdim=True) + 1e-10)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)

        gathered_sd_img_ft = self.all_gather(sd_img_ft) #[gather_bs, v_dim]
        gathered_sd_txt_ft = self.all_gather(sd_txt_ft) #[gather_bs, v_dim]

        logits_per_image_sd = sd_img_ft @ gathered_sd_txt_ft.t() * logit_scale
        logits_per_text_sd = sd_txt_ft @ gathered_sd_img_ft.t() * logit_scale

        # assert logits_per_image_sd.shape == logits_per_text_sd.shape


        return (logits_per_image_sd, logits_per_text_sd), \
                (img_k, txt_k)







class Clip_FDT_SP(Clip_FDT):
    def __init__(self, image_encode, text_encode, use_allgather, sd_num, sd_dim, raw_img_ft_dim, raw_txt_ft_dim, att_func_type, pool_type, sd_temperature):
        super().__init__(image_encode, text_encode, use_allgather, sd_num, sd_dim, raw_img_ft_dim, raw_txt_ft_dim, att_func_type, pool_type, sd_temperature)
        apply_spectral_norm_to_linear_layers(self)
        print("Spectral normalization version of the model")
        

def clip_fdt_vitb16(**kwargs):
    """'
    Constructs a clip_ViT_B32 model.
    """
    image_encode = visual_transformer_B16(**kwargs['image_encode'])
    text_encode = text_transformers(**kwargs['text_encode'])
    model = Clip_FDT(image_encode, text_encode, **kwargs['fdt'])
    return model


def clip_fdt_vitb32(**kwargs):
    """'
    Constructs a clip_ViT_B32 model.
    """
    image_encode = visual_transformer_B32(**kwargs['image_encode'])
    text_encode = text_transformers(**kwargs['text_encode'])
    model = Clip_FDT(image_encode, text_encode, **kwargs['fdt'])
    return model




def clip_fdt_sp_vitb32(**kwargs):
    """'
    Constructs a clip_ViT_B32 model.
    """
    image_encode = visual_transformer_B32(**kwargs['image_encode'])
    text_encode = text_transformers(**kwargs['text_encode'])
    model = Clip_FDT_SP(image_encode, text_encode, **kwargs['fdt'])
    return model



def clip_fdt_swinB_v2(**kwargs):
    """'
    Constructs a clip_ViT_L14 model.
    """
    #load cfg_pth
    swin_cfg_pth = './prototype/model/swin/configs/swinv2/swinv2_base_patch4_window7_224.yaml'
    with open(swin_cfg_pth, 'r') as f:
        swin_cfg = yaml.load(f, Loader=yaml.FullLoader)
    swin_cfg = EasyDict(swin_cfg)

    image_encode = build_swin_model(swin_cfg)
    text_encode = text_transformers(**kwargs['text_encode'])
    model = Clip_FDT(image_encode,text_encode,**kwargs['fdt'])
    return model