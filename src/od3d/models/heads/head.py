from torch import nn
from omegaconf import DictConfig
from typing import List

class OD3D_Head(nn.Module):

    subclasses = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls
    def __init__(self, in_dims: List, in_upsample_scales: List, config: DictConfig):
        super().__init__()
        self.config = config
        self.in_dims = in_dims
        self.in_upsample_scales = in_upsample_scales
        self.out_dim = config.fully_connected.out_dim if config.fully_connected.out_dim is not None else config.conv_blocks.out_dims[-1] #.get('out_dim', None) if config.get('out_dim', None) is not None else 3


