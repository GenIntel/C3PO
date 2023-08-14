import logging
logger = logging.getLogger(__name__)
from torch import nn
from omegaconf import DictConfig
import torch
from od3d.cv.transforms import RGB_UInt8ToFloat, RGB_Normalize # , CenterZoom3D, RGB_Random
import torchvision
from od3d.models.backbones.backbone import OD3D_Backbone
from od3d.data.ext_enum import ExtEnum
from od3d.cv.visual.resize import resize

class DINOv2_WEIGHTS(str, ExtEnum):
    DEFAULT = 'default'
    NONE = 'none'


class DINOv2(OD3D_Backbone):
    def __init__(
        self,
        config: DictConfig
    ):

        super().__init__(config=config)

        self.transform = torchvision.transforms.Compose([
                RGB_UInt8ToFloat(),
                RGB_Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


        # dinov2_vits14 dinov2_vitb14 dinov2_vitl14 dinov2_vitg14
        self.extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=self.config.weights=='default')

        self.layers_returned = config.layers_returned # choose from [1, 2, 3, 4]
        self.layers_count = len(self.layers_returned)

        self.out_dims = [384 ]
        self.out_downsample_scales = []

        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = resize(x, H_out=64 * 14, W_out=64 * 14)
        x = self.extractor.forward_features(x)["x_norm_patchtokens"]  # # 'x_norm_patchtokens', 'x_prenorm'
        x = x.reshape(-1, 64, 64, 384).permute(0, 3, 1, 2)

        x_layers = [x]
        return x_layers


