import logging

import od3d.io

logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
from omegaconf import DictConfig, open_dict
from od3d.models.backbones.backbone import OD3D_Backbone
from od3d.models.heads.head import OD3D_Head
from pathlib import Path


class OD3D_Model(nn.Module):

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.backbone: OD3D_Backbone = OD3D_Backbone.subclasses[self.config.backbone.class_name](config=self.config.backbone)
        self.transform = self.backbone.transform

        if self.config.head is not None:
            with open_dict(self.config):
                self.config.head.in_dims = self.backbone.out_dims
            self.head: OD3D_Head = OD3D_Head.subclasses[self.config.head.class_name](config=self.config.head, in_dims=self.backbone.out_dims, in_upsample_scales=self.backbone.out_downsample_scales)
            self.out_dim = self.head.out_dim
            self.downsample_rate = self.backbone.downsample_rate * self.head.downsample_rate
        else:
            self.out_dim = self.backbone.out_dims[-1]
            self.downsample_rate = self.backbone.downsample_rate

    @staticmethod
    def create_by_name(name: str):
        config = od3d.io.read_config_intern(rfpath=Path("methods/model").joinpath(f"{name}.yaml"))
        return OD3D_Model(config)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        feats_maps = self.backbone(x, *args, **kwargs)

        if self.config.head is not None:
            return self.head(feats_maps)
        else:
            return feats_maps
