import logging
logger = logging.getLogger(__name__)
from od3d.models.heads.head import OD3D_Head
from omegaconf import DictConfig
import torch
import torchvision.models.resnet
from torchvision.models.resnet import Bottleneck
import torch.nn as nn
from typing import List

class ResNet(OD3D_Head):
    def __init__(
        self,
        in_dims: List,
        in_upsample_scales: List,
        config: DictConfig,
    ):
        super().__init__(in_dims=in_dims, in_upsample_scales=in_upsample_scales, config=config)

        self.upsample_conv_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()

        assert len(self.in_upsample_scales) == len(self.in_dims) - 1

        for i in range(len(self.in_dims) - 1):
            downsample_channels = nn.Sequential(
                nn.Conv2d(self.in_dims[i] + self.in_dims[i + 1], self.in_dims[i + 1], kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.in_dims[i + 1]))
            self.upsample_conv_blocks.append(Bottleneck(inplanes=self.in_dims[i] + self.in_dims[i + 1], planes=self.in_dims[i + 1] // 4, downsample=downsample_channels))
            self.upsample.append(nn.Upsample(scale_factor=self.in_upsample_scales[i]))

        self.conv_blocks = nn.ModuleList()
        self.conv_blocks_out_dims = config.conv_blocks.out_dims
        self.conv_blocks_count = len(self.conv_blocks_out_dims)
        self.conv_blocks_strides = config.conv_blocks.strides
        self.conv_blocks_in_dims = [self.in_dims[-1]] + [config.conv_blocks.out_dims[i] for i in range(self.conv_blocks_count - 1)]

        assert len(self.conv_blocks_in_dims) == len(self.conv_blocks_out_dims)
        assert len(self.conv_blocks_out_dims) == len(self.conv_blocks_strides)
        assert len(self.conv_blocks_in_dims) == 0 or self.conv_blocks_in_dims[0] == self.in_dims[-1]

        self.conv_blocks = nn.Sequential(*[Bottleneck(inplanes=self.conv_blocks_in_dims[i],
                                                      planes=self.conv_blocks_out_dims[i] // 4,
                                                      stride=self.conv_blocks_strides[i],
                                                      downsample=nn.Sequential(
                                                            nn.Conv2d(self.conv_blocks_in_dims[i], self.conv_blocks_out_dims[i], kernel_size=1, stride=1, bias=False),
                                                            nn.BatchNorm2d(self.conv_blocks_out_dims[i])))
                                           for i in range(self.conv_blocks_count)])

        if config.fully_connected.out_dim is not None:
            self.fc_enabled = True
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear_in_dim = self.conv_blocks_out_dims[-1]
            self.fc = nn.Linear(self.linear_in_dim, self.out_dim)
        else:
            self.fc_enabled = False

    def forward(self, x):
        if len(self.in_dims) == 1:
            x_res = x[0]
        else:
            x_res = None
            for i in range(len(self.in_dims) - 1):
                if i == 0:
                    x_low = x[i]
                else:
                    x_low = x_res
                x_res = self.upsample_conv_blocks[i](torch.cat([x[i+1], self.upsample[i](x_low)], dim=1))

        x_res = self.conv_blocks(x_res)

        if self.fc_enabled:
            x_res = self.avgpool(x_res)
            x_res = torch.flatten(x_res, 1)
            x_res = self.fc(x_res)
        return x_res