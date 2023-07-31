from omegaconf import DictConfig
import torchvision
from od3d.methods.method import OD3D_Dataset
from typing import List
import torch
from od3d.methods.method import OD3D_Method
from od3d.methods.nemo.backbone import OD3D_Backbone
from od3d.cv.transforms.center_and_zoom3d import RandomCenterZoom3D, CenterZoom3D
from od3d.cv.transforms.rgb import RGB_Random
from pathlib import Path
from torch import nn as nn

class RegressionNet(nn.Module):
    def __init__(self, backbone: nn.Module, dim: int):
        super().__init__()
        self.backbone = backbone
        self.dim = dim

    def forward(self, img):
        feats = self.backbone(img)
        pred = feats

        return pred

class Regression(OD3D_Method):
    def __init__(
        self,
        config: DictConfig,
        logging_dir,
    ):
        super().__init__(config=config, logging_dir=logging_dir)

        self.device = 'cuda:0'


        # init Network
        backbone = OD3D_Backbone.subclasses[config.backbone.class_name](config.backbone)
        self.net = RegressionNet(backbone=backbone, dim=3)

        if config.train.transform.random_color:
            self.transform_train = torchvision.transforms.Compose([
                RandomCenterZoom3D(**config.train.transform.random_center_zoom3d),
                RGB_Random(),
                self.net.transform,
            ])
        else:
            self.transform_train = torchvision.transforms.Compose([
                RandomCenterZoom3D(**config.train.transform.random_center_zoom3d),
                self.net.transform,
            ])

        self.transform_test = torchvision.transforms.Compose([
            CenterZoom3D(**config.test.transform),
            self.net.transform
        ])

        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        # self.net = torch.nn.DataParallel(self.net).cuda()
        self.net.cuda()
        self.net.eval()

        self.optim = torch.optim.Adam(list(self.net.parameters()),
                                      lr=self.config.train.optimizer.lr,  #
                                      weight_decay=self.config.train.optimizer.weight_decay)  #
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, gamma=self.config.train.scheduler.gamma,
                                                              milestones=self.config.train.scheduler.milestones)

    def save_checkpoint(self, path_checkpoint: Path):
        torch.save({
            'net_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path_checkpoint)

    def load_checkpoint(self, path_checkpoint):
        checkpoint = torch.load(path_checkpoint)
        self.net.load_state_dict(checkpoint['net_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    @property
    def path_checkpoint(self):
        return self.logging_dir.joinpath('nemo.ckpt')
    def train(self, dataset: OD3D_Dataset, datasets_val: List[OD3D_Dataset]):
        score_metric_name = 'pose/acc_pi6'
        score_ckpt_val = 0.
        score_latest = 0.

        train_dataset_sub, val_dataset_sub = dataset.get_split(fraction1=1.-self.config.train.val_fraction,
                                                               fraction2=self.config.train.val_fraction)

        for epoch in range(self.config.train.epochs):
            if self.config.train.val and self.config.train.epochs_to_next_test > 0 and epoch % self.config.train.epochs_to_next_test == 0:
                for dataset_val in datasets_val + [val_dataset_sub]:
                    results_val = self.test(dataset_val)
                    results_val.log_with_prefix(prefix=f'val/{dataset_val.name}')
                    score_latest = results_val[score_metric_name]

                if score_latest > score_ckpt_val:
                    score_ckpt_val = score_latest
                    self.save_checkpoint(path_checkpoint=self.path_checkpoint)

            results_epoch = self.train_epoch(dataset=train_dataset_sub)
            results_epoch.log_with_prefix('train')
        self.load_checkpoint(path_checkpoint=self.path_checkpoint)

