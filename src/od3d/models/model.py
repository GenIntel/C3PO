import logging
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
from omegaconf import DictConfig, open_dict
from od3d.models.backbones.backbone import OD3D_Backbone
from od3d.models.heads.head import OD3D_Head

class OD3D_Model(nn.Module):

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.backbone: OD3D_Backbone = OD3D_Backbone.subclasses[self.config.backbone.class_name](config=self.config.backbone)
        with open_dict(self.config):
            self.config.head.in_dims = self.backbone.out_dims

        self.head: OD3D_Head = OD3D_Head.subclasses[self.config.head.class_name](config=self.config.head, in_dims=self.backbone.out_dims, in_upsample_scales=self.backbone.out_downsample_scales)
        self.transform = self.backbone.transform

    def forward(self, x: torch.Tensor):
        feats_maps = self.backbone(x)
        return self.head(feats_maps)
