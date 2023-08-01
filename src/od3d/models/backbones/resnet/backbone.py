import logging
logger = logging.getLogger(__name__)
from torch import nn
from omegaconf import DictConfig
import torch
from od3d.cv.transforms import RGB_UInt8ToFloat, RGB_Normalize # , CenterZoom3D, RGB_Random
import torchvision
from od3d.models.backbones.backbone import OD3D_Backbone

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

        self.layers_returned = config.layers_returned
        self.layers_count = len(self.layers_returned) # choose from [1, 2, 3, 4]

        resnet = torchvision.models.resnet50(weights=torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V2)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layers = nn.ModuleList()
        self.layers.append(resnet.layer1) # 256
        self.layers.append(resnet.layer2) # 512
        self.layers.append(resnet.layer3) # 1024
        self.layers.append(resnet.layer4) # 2048

        self.out_dims = [self.layers[layer_id - 1][-1].conv3.out_channels for layer_id in self.layers_returned]
        self.out_downsample_scales = [2**(self.layers_returned[i]-self.layers_returned[i+1]) for i in range(self.layers_count - 1)]
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_layers = []
        x_layers.append(self.layers[0](x))
        x_layers.append(self.layers[1](x_layers[-1]))
        x_layers.append(self.layers[2](x_layers[-1]))
        x_layers.append(self.layers[3](x_layers[-1]))

        x_layers = [x_layers[layer_id - 1] for layer_id in self.layers_returned]

        return x_layers


