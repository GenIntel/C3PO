import logging
logger = logging.getLogger(__name__)
from torch import nn
from omegaconf import DictConfig
import torch
from od3d.cv.visual.resize import resize
from od3d.cv.transforms import RGB_UInt8ToFloat, RGB_Normalize # , CenterZoom3D, RGB_Random
import torchvision
from od3d.models.backbones.backbone import OD3D_Backbone

class DINOv2(OD3D_Backbone):
    def __init__(self, config: DictConfig):
        super().__init__(config=config)
        # dinov2_vits14 dinov2_vitb14 dinov2_vitl14 dinov2_vitg14
        self.net = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        for param in self.net.parameters():
            param.requires_grad = False

        self.transform = torchvision.transforms.Compose([
                RGB_UInt8ToFloat(),
                RGB_Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.dino_feat_dim = 384
        self.feat_dim = config.channels # 128

        self.doubleconv = nn.Sequential(
            nn.Conv2d(self.dino_feat_dim, self.feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_dim, self.feat_dim , kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True),
        )

    # 'x_norm_patchtokens', 'x_prenorm'
    def forward(self, x):
        # b, _, h, w = x.size()
        #patches_w = w // 14 + 1
        #patches_h = h // 14 + 1
        #  torch.nn.functional.interpolate(x, (patches_h * 14, patches_w * 14))
        inter = resize(x, H_out=64 * 14, W_out=64 * 14)

        patch_tokens = self.extractor.forward_features(inter)["x_norm_patchtokens"]  # # 'x_norm_patchtokens', 'x_prenorm'

        feat = patch_tokens.reshape(-1, 64, 64, 384).permute(0, 3, 1, 2)

        return torch.nn.functional.normalize(self.doubleconv(feat), p=2, dim=1)
