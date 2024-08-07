from .clip_fdt import clip_fdt_vitb16, clip_fdt_vitb32, clip_fdt_swinB_v2, clip_fdt_sp_vitb32
from .declip_fdt import declip_fdt_vitb32
from .clip import clip_vitb32, clip_vitb32_sp

def model_entry(config):
    return globals()[config['type']](**config['kwargs'])
