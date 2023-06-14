from torch import nn
from enum import Enum
from omegaconf import DictConfig
from od3d.methods.nemo.keypoint_representation_net import NetE2E
import torch
from od3d.cv.visual.resize import resize
from od3d.cv.transforms import RGB_UInt8ToFloat, RGB_Normalize, CenterZoom3D, RGB_Random
import torchvision

class OD3D_Backbone(nn.Module):

    subclasses = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.transform = None

    pass

class DINOv2(OD3D_Backbone):
    def __init__(self, config: DictConfig):
        super().__init__(config=config)
        # dinov2_vits14 dinov2_vitb14 dinov2_vitl14 dinov2_vitg14
        self.net = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        self.transform = torchvision.transforms.Compose([
                CenterZoom3D(H=config.transform.height, W=config.transform.width, dist=config.transform.distance),
                RGB_UInt8ToFloat(),
                RGB_Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.feat_dim = 384
    # 'x_norm_patchtokens', 'x_prenorm'
    def forward(self, x):
        return resize(self.net.forward_features(resize(x, H_out=518, W_out=518))['x_norm_patchtokens'].reshape(-1, 37, 37, 384).permute(0, 3, 1, 2), H_out=64, W_out=64)

class ResNetExt(OD3D_Backbone):
    def __init__(
        self,
        config: DictConfig
    ):

        super().__init__(config=config)

        self.transform = torchvision.transforms.Compose([
                CenterZoom3D(H=config.transform.height, W=config.transform.width, dist=config.transform.distance),
                RGB_Random(),
                RGB_UInt8ToFloat(),
                RGB_Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.feat_dim = 128

        self.net = NetE2E(
            net_type=config.net_type,
            local_size=[config.local_size[0], config.local_size[1]],
            output_dimension=config.output_dimension,
            reduce_function=None,
            n_noise_points=5, #config.num_noise,
            pretrain=True,
        )

    def forward(self, rgb):

        return self.net.forward_test(rgb)