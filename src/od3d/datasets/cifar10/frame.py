import logging
logger = logging.getLogger(__name__)
from omegaconf import OmegaConf, DictConfig
import torch
from typing import List
from pathlib import Path
from dataclasses import dataclass
from od3d.datasets.frame import OD3D_FrameMeta, OD3D_Frame
import scipy.io
import math
import numpy as np
from od3d.cv.geometry.transform import transf4x4_from_spherical
from od3d.datasets.dataset import OD3D_Frames, OD3D_FRAME_MODALITIES
from od3d.datasets.pascal3d.enum import PASCAL3D_SCALE_NORMALIZE_TO_REAL, PASCAL3D_CATEGORIES
from od3d.cv.io import read_image, write_mask_image, write_depth_image, read_depth_image
from od3d.cv.geometry.mesh import Mesh, Meshes
from od3d.cv.geometry.mesh import Meshes, MESH_RENDER_MODALITIES

from od3d.datasets.frame_meta import OD3D_FrameMeta, \
    OD3D_FrameMetaMeshMixin, OD3D_FrameMetaCategoryMixin, OD3D_FrameMetaRGBMixin, \
    OD3D_FrameMetaSizeMixin, OD3D_FrameMetaKpts2D3DMixin, OD3D_FrameMetaBBoxMixin, OD3D_FrameMetaSubsetMixin, \
    OD3D_FrameMetaCamTform4x4ObjMixin, OD3D_FrameMetaCamIntr4x4Mixin

from od3d.datasets.pascal3d.enum import MAP_CATEGORIES_PASCAL3D_TO_OD3D
from od3d.datasets.frame import OD3D_FrameMeshMixin, OD3D_FrameTformObjMixin, OD3D_CamProj4x4ObjMixin, \
    OD3D_FrameRGBMaskMixin, OD3D_FrameMaskMixin, OD3D_FrameRGBMixin, OD3D_FrameDepthMixin, OD3D_FrameDepthMaskMixin, \
    OD3D_FrameCategoryMixin, OD3D_FrameSizeMixin, OD3D_Frame, OD3D_FrameBBoxMixin, OD3D_FrameKpts2d3dMixin


@dataclass
class CIFAR10_FrameMeta(OD3D_FrameMetaSubsetMixin, OD3D_FrameMetaCategoryMixin,
                        OD3D_FrameMetaRGBMixin, OD3D_FrameMetaSizeMixin, OD3D_FrameMeta):

    @property
    def name_unique(self):
        return f'{self.subset}/{self.category}/{self.name}'

    @staticmethod
    def get_name_unique_from_category_subset_name(category, subset, name):
        return f'{subset}/{category}/{name}'

@dataclass
class CIFAR10_Frame(OD3D_FrameRGBMaskMixin, OD3D_FrameMaskMixin, OD3D_FrameRGBMixin,
                    OD3D_FrameCategoryMixin, OD3D_FrameSizeMixin, OD3D_Frame):
    meta_type = CIFAR10_FrameMeta
    map_categories_to_od3d = MAP_CATEGORIES_PASCAL3D_TO_OD3D

