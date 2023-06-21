from torch import nn
from enum import Enum
from omegaconf import DictConfig
import od3d.methods.nemo.keypoint_representation_net #  import NetE2E, ResNetExt
import torch
from od3d.cv.visual.resize import resize
from od3d.cv.transforms import RGB_UInt8ToFloat, RGB_Normalize # , CenterZoom3D, RGB_Random
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
        dinov2_out = self.net.forward_features(resize(x, H_out=518, W_out=518))
        patch_tokens = dinov2_out['x_prenorm'][:, 1:]
        #patch_tokens = dinov2_out['x_norm_patchtokens']

        feat = resize(patch_tokens.reshape(-1, 37, 37, 384).permute(0, 3, 1, 2), H_out=64, W_out=64, mode='bilinear')

        return torch.nn.functional.normalize(self.doubleconv(feat), p=2, dim=1)

class ResNet(OD3D_Backbone):
    def __init__(
        self,
        config: DictConfig
    ):

        super().__init__(config=config)

        self.transform = torchvision.transforms.Compose([
                RGB_UInt8ToFloat(),
                RGB_Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.feat_dim = config.channels # 128

        net = torchvision.models.resnet50(pretrained=True)
        self.extractor1 = nn.Sequential()
        self.extractor1.add_module("0", net.conv1)
        self.extractor1.add_module("1", net.bn1)
        self.extractor1.add_module("2", net.relu)
        self.extractor1.add_module("3", net.maxpool)
        self.extractor1.add_module("4", net.layer1) # 256
        self.extractor2 = net.layer2                # 512
        self.extractor3 = net.layer3                # 1024
        self.extractor4 = net.layer4                # 2048

        self.layer_channels = [256, 512, 1024, 2048]
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.down = nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=True)

        self.doubleconv1 = nn.Sequential(
            nn.Conv2d(self.layer_channels[0],  self.feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_dim, self.feat_dim , kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True),
        )
        self.doubleconv2 = nn.Sequential(
            nn.Conv2d(self.layer_channels[1],  self.feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True),
        )
        self.doubleconv3 = nn.Sequential(
            nn.Conv2d(self.layer_channels[2],  self.feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True),
        )
        self.doubleconv4 = nn.Sequential(
            nn.Conv2d(self.layer_channels[3],  self.feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True),
        )

        self.doubleconv_out = nn.Sequential(
            nn.Conv2d(self.feat_dim * len(self.layer_channels),  self.feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True),
        )
        #self.extractor.add_module("5", net.layer2)
        #self.extractor.add_module("6", net.layer3)
        #self.extractor.add_module("7", net.layer4)

    def forward(self, rgb):
        feat1 = self.extractor1(rgb)
        feat2 = self.extractor2(feat1)
        feat3 = self.extractor3(feat2)
        feat4 = self.extractor4(feat3)

        feat1 = self.doubleconv1(feat1)
        feat2 = self.up(self.doubleconv2(feat2))
        feat3 = self.up(self.up(self.doubleconv3(feat3)))
        feat4 = self.up(self.up(self.up(self.doubleconv4(feat4))))

        return torch.nn.functional.normalize(self.down(self.doubleconv_out(torch.cat([feat1, feat2, feat3, feat4], dim=1))), p=2, dim=1)
        # return self.extractor(rgb)


class ResNetExt(OD3D_Backbone):
    def __init__(
        self,
        config: DictConfig
    ):

        super().__init__(config=config)

        self.transform = torchvision.transforms.Compose([
                RGB_UInt8ToFloat(),
                RGB_Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.feat_dim = config.channels # 128

        # self.net = od3d.methods.nemo.keypoint_representation_net.ResNetExt(pretrained=True)

        self.net = od3d.methods.nemo.keypoint_representation_net.NetE2E(
            net_type=config.net_type,
            local_size=[config.local_size[0], config.local_size[1]],
            output_dimension=config.output_dimension,
            reduce_function=None,
            n_noise_points=5, #config.num_noise,
            pretrain=True,
        )

    def forward(self, rgb):
        return self.net.forward_test(resize(rgb, H_out=512, W_out=512))
        #return torch.nn.functional.normalize(self.net.net(rgb), p=2, dim=1)