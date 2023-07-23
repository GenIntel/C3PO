import logging
logger = logging.getLogger(__name__)

from pathlib import Path
import torch
from omegaconf import OmegaConf
from od3d.cv.geometry.transform import tform4x4
from od3d.cv.io import read_image
import torchvision
from dataclasses import dataclass
from typing import List
from enum import Enum

class OD3D_FRAME_MODALITIES(str, Enum):
    RGB = 'rgb'
    MASK = 'mask'
    DEPTH = 'depth'
    DEPTH_MASK = 'depth_mask'
    MESH = 'mesh'
    KPTS = 'kpts'
    BBOX = 'bbox'
    CUBOID_FRONT_TFORM4X4_OBJ = 'cuboid_front_tform4x4_obj'
    SEQUENCE_NAME = 'sequence_name'
    SEQUENCE = 'sequence'

@dataclass
class OD3D_FrameMeta:
    category: str
    name: str
    rfpath_rgb: Path
    rfpath_mask: Path
    rfpath_depth: Path
    rfpath_depth_mask: Path
    # rfpath_pcl: Path
    l_cam_tform4x4_obj: List[List[float]] #  torch.Tensor
    l_cam_intr4x4: List[List[float]] # torch.Tensor
    l_size: List[float] # torch.Tensor
    H: int
    W: int

    @staticmethod
    def load_from_meta_with_rfpath(path_meta: Path, rfpath: Path):
        fpath_meta = path_meta.joinpath(rfpath)
        if not fpath_meta.exists():
            logger.error(f'Missing meta fpath {fpath_meta}. Preprocess meta before.')
        return OD3D_FrameMeta(**OmegaConf.load(fpath_meta))

    @staticmethod
    def get_rfpath_frames():
        return Path("frames")

    @staticmethod
    def get_path_frames(path_meta: Path):
        return path_meta.joinpath(OD3D_FrameMeta.get_rfpath_frames())

    @property
    def name_unique(self):
        raise NotImplementedError

    @property
    def cam_tform4x4_obj(self):
        return torch.Tensor(self.l_cam_tform4x4_obj)

    def get_fpath(self, path_meta):
        raise NotImplementedError

    def save(self, path_meta):
        frame_meta_fpath = self.get_fpath(path_meta=path_meta)
        frame_meta_config = OmegaConf.structured(self)
        if not frame_meta_fpath.parent.exists():
            frame_meta_fpath.parent.mkdir(parents=True)
        OmegaConf.save(frame_meta_config, frame_meta_fpath, resolve=True)

class OD3D_Frame():

    def __init__(self, path_raw: Path, path_preprocess: Path, path_meta: Path, meta: OD3D_FrameMeta, modalities: List[OD3D_FRAME_MODALITIES], categories: List[str]):
        self.meta: OD3D_FrameMeta = meta
        self.path_raw: Path = path_raw
        self.path_preprocess: Path = path_preprocess
        self.path_meta: Path = path_meta
        self._cam_tform4x4_obj = None
        self._cam_intr4x4 = None
        self._size = None
        self._rgb = None
        self._mask = None
        self._depth = None
        self._depth_mask = None
        self._kpts2d_orient = None
        self.categories = categories
        self.category_id = categories.index(self.category)
        self.modalities = modalities
        self.item_id = None

    @property
    def name_unique(self):
        return self.meta.name_unique

    @property
    def size(self):
        if self._size is None:
            self._size = torch.Tensor(self.meta.l_size)
        return self._size

    @property
    def cam_intr4x4(self):
        if self._cam_intr4x4 is None:
            self._cam_intr4x4 = torch.Tensor(self.meta.l_cam_intr4x4)
        return self._cam_intr4x4

    @property
    def cam_tform4x4_obj(self):
        if self._cam_tform4x4_obj is None:
            self._cam_tform4x4_obj = torch.Tensor(self.meta.l_cam_tform4x4_obj)
        return self._cam_tform4x4_obj

    @property
    def cam_proj4x4_obj(self):
        return tform4x4(self.cam_intr4x4, self.cam_tform4x4_obj)

    @property
    def fpath_kpts2d_orient(self):
        return self.path_preprocess.joinpath("labels", "kpts2d_orient", f"{self.name_unique}.pt")

    @property
    def kpts2d_orient_labeled(self):
        return self.fpath_kpts2d_orient.exists()
    @property
    def kpts2d_orient(self):
        kpts2d_orient = torch.load(self.fpath_kpts2d_orient)
        return kpts2d_orient

    @property
    def path_mask(self):
        return self.path_raw.joinpath(self.meta.rfpath_mask)

    @property
    def mask(self):
        if self._mask is None:
            self._mask = read_image(self.path_mask) / 255.
        return self._mask

    @property
    def path_rgb(self):
        return self.path_raw.joinpath(self.meta.rfpath_rgb)
    @property
    def rgb(self):
        if self._rgb is None:
            self._rgb = torchvision.io.read_image(str(self.path_rgb), mode=torchvision.io.ImageReadMode.RGB)
        return self._rgb

    @property
    def depth(self):
        raise NotImplementedError

    @property
    def depth_mask(self):
        if self._depth_mask is None:
            self._depth_mask = read_image(self.path_raw.joinpath(self.meta.rfpath_depth_mask))
        return self._depth_mask

    @property
    def category(self):
        return self.meta.category

    @property
    def name(self):
        return self.meta.name
