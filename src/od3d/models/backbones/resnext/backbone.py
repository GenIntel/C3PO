import logging
logger = logging.getLogger(__name__)
from omegaconf import DictConfig
from od3d.cv.transforms import RGB_UInt8ToFloat, RGB_Normalize # , CenterZoom3D, RGB_Random
import torchvision
from od3d.models.backbones.backbone import OD3D_Backbone
from od3d.models.backbones.resnext.backbone_old import NetE2E

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

        # self.net = od3d.methods.nemo.backbone_old.ResNetExt(pretrained=True)

        self.net = NetE2E(
            net_type=config.net_type,
            local_size=[config.local_size[0], config.local_size[1]],
            output_dimension=config.output_dimension,
            reduce_function=None,
            n_noise_points=5, #config.num_noise,
            pretrain=True,
        )

    def forward(self, rgb):
        #return self.net.forward_test(resize(rgb, H_out=512, W_out=512))
        return self.net.forward_test(rgb)
        #return torch.nn.functional.normalize(self.net.net(rgb), p=2, dim=1)