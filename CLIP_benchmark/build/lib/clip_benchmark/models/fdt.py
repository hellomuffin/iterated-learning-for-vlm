
import sys
sys.path.append(".")
from prototype.utils.misc import makedir, create_logger, get_logger, count_params, \
    param_group_all, AverageMeter, accuracy, load_state_optimizer, load_state_model,\
    parse_config
from prototype.model import model_entry
from prototype.data.imagenet_dataloader import build_common_augmentation
from easydict import EasyDict
import torch
config = None
ckpt_pth = None



class MyModelZoo(torch.nn.Module):
    def __init__(self, config, ckpt_pth=None) -> None:
        super().__init__()

        prototype_info = EasyDict() #a dict
        config = parse_config(config)
        self.model = model_entry(config.model)
        prototype_info.model = config.model.type
        print(prototype_info.model)
        self.model.cuda()
        count_params(self.model)
        
        if isinstance(ckpt_pth, list):
            
            state = [torch.load(k, map_location='cpu') for k in ckpt_pth]
            print(f"load ckpt from {ckpt_pth}")
            new_model_state = {}
            for per_state in state:
                for k, v in per_state["model"].items():
                    if k[:6] == 'module': k_ = k[7:]
                    else: k_ = k
                    if k_ not in new_model_state.keys(): new_model_state[k_] = [v]
                    else: new_model_state[k_].append(v)
            complete_model_state = {k: sum(v)/len(v) for k, v in new_model_state.items()}
            load_state_model(self.model, complete_model_state)
        elif ckpt_pth:
            state = torch.load(ckpt_pth, map_location='cpu')
            print(f"load ckpt from {ckpt_pth}")
            print(f"load model from checkpoint")
            new_model_state = {}
            for k, v in state["model"].items():
                if k[:6] == 'module': k_ = k[7:]
                else: k_ = k
                new_model_state[k_] = v
            load_state_model(self.model, new_model_state)
        else:
            state = {}
        self.model.eval()
    
    def encode_image(self, image):
        with torch.no_grad():
            cuda_image = image.cuda()
            try:
                _, image_embedding, _ = self.model.extract_img_sd_ft(cuda_image)
            except:
                print("use encode image")
                image_embedding = self.model.encode_image(cuda_image)
            return image_embedding
    
    def encode_text(self, text_tokenize):
        with torch.no_grad():
            try: _, text_embedding, _ = self.model.extract_txt_sd_ft(text_tokenize, raw_text=True)
            except: 
                text_embedding = self.model.encode_text(text_tokenize, raw_text=True)
                print("use encode text")
            return text_embedding
    
    def get_full_image_embedding_info(self, image):
        with torch.no_grad():
            return self.model.extract_img_sd_ft(image)
        
    def get_tokenize_function(self):
        return self.model.encode_text.wrap_tokenize
    
    def get_test_transform(self):
        return build_common_augmentation('ONECROP')
    
    
            
def load_fdt(model_name, pretrained, cache_dir, device):
    if model_name == 'clip': config = 'example/clip/config_cc3m.yaml'
    elif model_name == 'clip_sp': config = 'example/clip_fdt/config_cc3m_sp.yaml'
    else: config = 'example/clip_fdt/config_cc3m.yaml'
    model = MyModelZoo(config=config, ckpt_pth=pretrained).to(device)
    transform = model.get_test_transform()
    return model, transform, None

            
        


