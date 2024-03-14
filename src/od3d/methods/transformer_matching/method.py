import time
from typing import List
import od3d.io
from od3d.cv.statistics.standard_devation import mean_avg_std_with_id
from od3d.methods.method import OD3D_Method
from od3d.datasets.dataset import OD3D_Dataset
from od3d.benchmark.results import OD3D_Results
from omegaconf import DictConfig
import pytorch3d.transforms
import pandas as pd
import numpy as np
from torch.utils.data import RandomSampler
import logging

from od3d.cv.metric.pose import get_pose_diff_in_rad
logger = logging.getLogger(__name__)
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from od3d.cv.geometry.transform import se3_exp_map
from od3d.cv.visual.show import imgs_to_img
from od3d.cv.geometry.mesh import Meshes
from pathlib import Path
from od3d.cv.geometry.transform import transf4x4_from_spherical, tform4x4, rot3x3, inv_tform4x4, tform4x4_broadcast
from od3d.cv.visual.show import show_img
import torchvision
from od3d.cv.visual.blend import blend_rgb
from od3d.cv.visual.sample import sample_pxl2d_pts
from tqdm import tqdm
from od3d.cv.geometry.mesh import MESH_RENDER_MODALITIES
# note: math is actually used by config
import math
from od3d.datasets.co3d import CO3D

from od3d.cv.io import image_as_wandb_image
from od3d.cv.visual.resize import resize
from od3d.models.model import OD3D_Model

from od3d.cv.geometry.grid import get_pxl2d_like
from od3d.cv.geometry.fit3d2d import batchwise_fit_se3_to_corresp_3d_2d_and_masks
from od3d.cv.transforms.transform import OD3D_Transform
from od3d.cv.transforms.sequential import SequentialTransform

from typing import Dict
from od3d.data.ext_enum import ExtEnum

import matplotlib.pyplot as plt

class TransformerMatching(OD3D_Method):
    def setup(self):
        pass


    def __init__(
            self,
            config: DictConfig,
            logging_dir,
    ):
        super().__init__(config=config, logging_dir=logging_dir)


    def save_checkpoint(self, path_checkpoint: Path):
        pass

    def load_checkpoint(self, path_checkpoint):
        pass
    
    @property
    def path_checkpoint(self):
        return self.logging_dir.joinpath('nemo.ckpt')

    def train(self, datasets_train: Dict[str, OD3D_Dataset], datasets_val: Dict[str, OD3D_Dataset]):
        pass


    def test(self, dataset: OD3D_Dataset, config_inference: DictConfig = None):
        pass


    def train_batch(self, batch) -> OD3D_Results:
       pass
