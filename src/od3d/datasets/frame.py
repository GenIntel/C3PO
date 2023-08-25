import logging
logger = logging.getLogger(__name__)

from pathlib import Path
import torch
from omegaconf import OmegaConf, DictConfig
from od3d.cv.geometry.transform import tform4x4
from od3d.cv.io import read_image
import torchvision
from dataclasses import dataclass
from typing import List, Union, Dict
from enum import Enum
from abc import ABC, abstractmethod
from od3d.cv.geometry.mesh import Mesh
from od3d.data.ext_dicts import unroll_nested_dict, rollup_flattened_dict
import re

class OD3D_FRAME_MODALITIES(str, Enum):
    NAME = 'name'
    CAM_INTR4X4 = 'cam_intr4x4'
    CAM_TFORM4X4_OBJ = 'cam_tform4x4_obj'
    CAM_TFORM4X4_OBJS = 'cam_tform4x4_objs'
    CATEGORY = 'category'
    CATEGORIES = 'categories'
    RGB = 'rgb'
    MASK = 'mask'
    DEPTH = 'depth'
    DEPTH_MASK = 'depth_mask'
    MESH = 'mesh'
    MESHS = 'meshs'
    KPTS = 'kpts'
    BBOX = 'bbox'
    BBOXS = 'bboxs'
    CUBOID_FRONT_TFORM4x4_OBJ = 'cuboid_front_tform4x4_obj'
    SEQUENCE_NAME = 'sequence_name'
    SEQUENCE = 'sequence'


@dataclass
class OD3D_FrameMetaBBoxMixin():
    # x0, y0, x1, y1
    l_bbox: List[float]
    @property
    def bbox(self):
        return torch.Tensor(self.l_bbox)

@dataclass
class OD3D_FrameMetaBBoxsMixin():
    l_bboxs: List[List[float]]
    @property
    def bboxs(self):
        return torch.Tensor(self.l_bboxs)

@dataclass
class OD3D_FrameKPTS2D3DMixin():
    l_kpts2d_annot: List[List[float]]
    l_kpts2d_annot_vsbl: List[bool]
    l_kpts3d: List[List[float]]
    kpts_names: List[str]

    @property
    def kpts2d_annot(self):
        return torch.Tensor(self.l_kpts2d_annot)
    @property
    def kpts2d_annot_vsbl(self):
        return torch.Tensor(self.l_kpts2d_annot_vsbl).to(dtype=bool)
    @property
    def kpts3d(self):
        return torch.Tensor(self.l_kpts3d)

@dataclass
class OD3D_FrameMetaSubsetMixin:
    subset: str

@dataclass
class OD3D_FrameMetaMeshMixin():
    rfpath_mesh: Path

@dataclass
class OD3D_FrameMetaMeshsMixin():
    rfpaths_meshs: List[Path]

@dataclass
class OD3D_FrameMetaSequenceMixin():
    sequence_name: str

@dataclass
class OD3D_FrameMetaCategoryMixin():
    category: str

@dataclass
class OD3D_FrameMetaCategoriesMixin:
    categories: List[str]

@dataclass
class OD3D_FrameMetaCamIntr4x4Mixin:
    l_cam_intr4x4: List[List[float]]  # torch.Tensor

    @property
    def cam_intr4x4(self):
        return torch.Tensor(self.l_cam_intr4x4)

@dataclass
class OD3D_FrameMetaCamTform4x4ObjMixin:
    l_cam_tform4x4_obj: List[List[float]]  # torch.Tensor

    @property
    def cam_tform4x4_obj(self):
        return torch.Tensor(self.l_cam_tform4x4_obj)

@dataclass
class OD3D_FrameMetaCamTform4x4ObjsMixin:
    l_cam_tform4x4_objs: List[List[List[float]]]  # torch.Tensor

    @property
    def cam_tform4x4_objs(self):
        return torch.Tensor(self.l_cam_tform4x4_objs)

@dataclass
class OD3D_FrameMetaSizeMixin:
    l_size: List[float] # torch.Tensor
    @property
    def size(self):
        return torch.Tensor(self.l_size)
    @property
    def H(self):
        return self.size[0]
    @property
    def W(self):
        return self.size[1]

@dataclass
class OD3D_FrameMetaRGBMixin:
    rfpath_rgb: Path

@dataclass
class OD3D_FrameMetaMaskMixin:
    rfpath_mask: Path
@dataclass
class OD3D_FrameMetaDepthMixin:
    rfpath_depth: Path

@dataclass
class OD3D_FrameMetaDepthMaskMixin:
    rfpath_depth_mask: Path

@dataclass
class OD3D_FrameMetaPCLMixin:
    rfpath_pcl: Path

@dataclass
class OD3D_Meta(ABC):
    name: str

    # @staticmethod
    # @abstractmethod
    @classmethod
    def load_from_meta_with_rfpath(cls, path_meta: Path, rfpath: Path):
        return cls(**cls.load_omega_conf_with_rfpath(path_meta=path_meta, rfpath=rfpath))
        # raise NotImplementedError
        # each subclass must implement this function with
        #   SUBCLASS(**SUBCLASS.load_omega_conf_with_rfpath(path_meta=path_meta, rfpath=rfpath))

    # @staticmethod
    @classmethod
    def load_from_meta_with_name_unique(cls, path_meta: Path, name_unique: str):
        rfpath = cls.get_rfpath_from_name_unique(name_unique=name_unique)
        return cls.load_from_meta_with_rfpath(path_meta=path_meta, rfpath=rfpath)

    @staticmethod
    @abstractmethod
    def load_from_raw(**kwargs):
        raise NotImplementedError

    @property
    def name_unique(self):
        return self.name

    @classmethod
    def get_rfpath_from_name_unique(cls, name_unique: str):
        return cls.get_rfpath_metas().joinpath(Path(f'{name_unique}.yaml'))

    @classmethod
    def get_rfpath_metas(cls):
        raise NotImplementedError

    @classmethod
    def get_path_metas(cls, path_meta: Path):
        return path_meta.joinpath(cls.get_rfpath_metas())

    @staticmethod
    def atoi(text):
        return int(text) if text.isdigit() else text

    @classmethod
    def complete_nested_metas(cls, path_meta: Path, dict_nested_metas: Union[Dict, DictConfig, None], parent_key='', separator='/'):
        if dict_nested_metas is None:
            dict_nested_metas = {'': None}
        dict_nested_frames_completed = {}

        for key, value in dict_nested_metas.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key
            if value is None:
                dir_fpaths = [fpath for fpath in list(
                    cls.get_path_metas(path_meta=path_meta).joinpath(new_key).iterdir())]
                if dir_fpaths[0].is_dir():
                    dict_nested_frames_completed[key] = \
                        cls.complete_nested_metas(path_meta=path_meta, parent_key=new_key,
                                                  dict_nested_metas={f'{dir_fpath.stem}': None
                                                                      for dir_fpath in dir_fpaths})
                else:

                    dict_nested_frames_completed[key] = [dir_fpath.stem for dir_fpath in sorted(dir_fpaths, key=lambda f: [OD3D_Meta.atoi(val) for val in re.split(r'(\d+)', f.stem)])]

            elif isinstance(value,  Union[Dict, DictConfig]):
                dict_nested_frames_completed[key] = cls.complete_nested_metas(path_meta=path_meta,
                                                                              parent_key=new_key,
                                                                              dict_nested_metas=value)
            else:
                dict_nested_frames_completed[key] = value

        if len(dict_nested_frames_completed.keys()) == 1 and '' in dict_nested_frames_completed.keys():
            dict_nested_frames_completed = dict_nested_frames_completed['']
        return dict_nested_frames_completed

    @property
    def rfpath(self):
        return self.get_rfpath_from_name_unique(self.name_unique)

    @staticmethod
    def unroll_nested_metas(dict_nested_meta: Dict, separator='/'):
        dict_frames = unroll_nested_dict(dict_nested_meta, separator=separator)
        list_frames_names_unique = []
        for key, frames_names in dict_frames.items():
            for frame_name in frames_names:
                frame_name_unique = f"{key}{separator}{frame_name}" if key else frame_name
                list_frames_names_unique.append(frame_name_unique)
        return list_frames_names_unique

    @staticmethod
    def rollup_flattened_frames(list_meta_names_unique: List):
        dict_frames = {}
        for frame_name_unique in list_meta_names_unique:
            frame_name_unique_split = frame_name_unique.split('/')
            key = '/'.join(frame_name_unique_split[:-1])
            if key not in dict_frames.keys():
                dict_frames[key] = []
            frame_name = frame_name_unique_split[-1]
            dict_frames[key].append(frame_name)
        return rollup_flattened_dict(flattened_dict=dict_frames)

    def get_fpath(self, path_meta: Path):
        return path_meta.joinpath(self.rfpath)

    @staticmethod
    def load_omega_conf_with_rfpath(path_meta: Path, rfpath: Path):
        fpath_meta = path_meta.joinpath(rfpath)
        if not fpath_meta.exists():
            logger.error(f'Missing meta fpath {fpath_meta}. Preprocess meta before.')
        return OmegaConf.load(fpath_meta)

    def save(self, path_meta):
        frame_meta_fpath = self.get_fpath(path_meta=path_meta)
        frame_meta_config = OmegaConf.structured(self)
        if not frame_meta_fpath.parent.exists():
            frame_meta_fpath.parent.mkdir(parents=True)
        OmegaConf.save(frame_meta_config, frame_meta_fpath, resolve=True)
@dataclass
class OD3D_FrameMeta(OD3D_Meta):
    @classmethod
    def get_rfpath_metas(cls):
        return Path("frames")

@dataclass
class OD3D_SequenceMeta(OD3D_Meta):
    @classmethod
    def get_rfpath_metas(cls):
        return Path("sequences")

OD3D_FrameMetaClasses = Union[OD3D_FrameMeta, OD3D_FrameMetaCategoryMixin, OD3D_FrameMetaCategoriesMixin,
                              OD3D_FrameMetaCamTform4x4ObjMixin, OD3D_FrameMetaCamIntr4x4Mixin,
                              OD3D_FrameMetaMaskMixin, OD3D_FrameMetaDepthMaskMixin, OD3D_FrameMetaDepthMixin,
                              OD3D_FrameMetaPCLMixin, OD3D_FrameMetaSizeMixin, OD3D_FrameMetaRGBMixin,
                              OD3D_FrameMetaMeshMixin, OD3D_FrameMetaSequenceMixin, OD3D_FrameMetaSubsetMixin,
                              OD3D_FrameMetaBBoxMixin, OD3D_FrameKPTS2D3DMixin]

class OD3D_Frame():

    def __init__(self, path_raw: Path, path_preprocess: Path, path_meta: Path, meta: OD3D_FrameMetaClasses, modalities: List[OD3D_FRAME_MODALITIES], categories: List[str]):
        self.meta: OD3D_FrameMetaClasses = meta
        self.path_raw: Path = path_raw
        self.path_preprocess: Path = path_preprocess
        self.path_meta: Path = path_meta
        self.all_categories = categories
        #self.category_id = categories.index(self.category)
        self.modalities = modalities
        self.item_id = None
        self._cam_tform4x4_obj = None
        self._cam_intr4x4 = None
        self._mask_rgb = None
        self._size = None
        self._rgb = None
        self._mask = None
        self._depth = None
        self._depth_mask = None
        self._kpts2d_orient = None
        self._bbox = None
        self._mesh = None
        self._kpts2d_annot_vsbl = None
        self._kpts2d_annot = None
        self._kpts3d = None

    @property
    def category_id(self):
        return self.all_categories.index(self.category)

    @property
    def name_unique(self):
        return self.meta.name_unique

    @property
    def mask_rgb(self):
        if self._mask_rgb is None:
            self._mask_rgb = torch.ones(size=(1, self.H, self.W), dtype=torch.bool)
        return self._mask_rgb

    @mask_rgb.setter
    def mask_rgb(self, value: torch.Tensor):
        self._mask_rgb = value

    @property
    def size(self):
        if self._size is None:
            self._size = self.meta.size
        return self._size

    @size.setter
    def size(self, value: torch.Tensor):
        self._size = value

    @property
    def H(self):
        return int(self.size[0].item())

    @property
    def W(self):
        return int(self.size[1].item())

    @property
    def cam_intr4x4(self):
        if self._cam_intr4x4 is None:
            self._cam_intr4x4 = self.meta.cam_intr4x4
        return self._cam_intr4x4

    @cam_intr4x4.setter
    def cam_intr4x4(self, value: torch.Tensor):
            self._cam_intr4x4 = value

    @property
    def cam_tform4x4_obj(self):
        if self._cam_tform4x4_obj is None:
            self._cam_tform4x4_obj = self.meta.cam_tform4x4_obj
        return self._cam_tform4x4_obj

    @cam_tform4x4_obj.setter
    def cam_tform4x4_obj(self, value: torch.Tensor):
            self._cam_tform4x4_obj = value

    @property
    def cam_proj4x4_obj(self):
        return tform4x4(self.cam_intr4x4, self.cam_tform4x4_obj)

    @property
    def bbox(self):
        if self._bbox is None:
            self._bbox = self.meta.bbox
        return self._bbox

    @bbox.setter
    def bbox(self, value: torch.Tensor):
            self._bbox = value

    @property
    def kpts_names(self):
        return self.meta.kpts_names

    @property
    def kpts2d_annot(self):
        if self._kpts2d_annot is None:
            self._kpts2d_annot = self.meta.kpts2d_annot
        return self._kpts2d_annot

    @kpts2d_annot.setter
    def kpts2d_annot(self, value: torch.Tensor):
            self._kpts2d_annot = value

    @property
    def kpts2d_annot_vsbl(self):
        if self._kpts2d_annot_vsbl is None:
            self._kpts2d_annot_vsbl = self.meta.kpts2d_annot_vsbl
        return self._kpts2d_annot_vsbl

    @kpts2d_annot_vsbl.setter
    def kpts2d_annot_vsbl(self, value: torch.Tensor):
            self._kpts2d_annot_vsbl = value

    @property
    def kpts3d(self):
        if self._kpts3d is None:
            self._kpts3d = self.meta.kpts3d
        return self._kpts3d

    @kpts3d.setter
    def kpts3d(self, value: torch.Tensor):
            self._kpts3d = value

    @property
    def fpath_mesh(self):
        return self.path_raw.joinpath(self.meta.rfpath_mesh)

    @property
    def mesh(self):
        if self._mesh is None:
            self._mesh = Mesh.load_from_file(fpath=self.fpath_mesh)
        return self._mesh

    @mesh.setter
    def mesh(self, value: torch.Tensor):
            self._mesh = value

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
    def fpath_mask(self):
        return self.path_raw.joinpath(self.meta.rfpath_mask)

    @property
    def mask(self):
        if self._mask is None:
            self._mask = read_image(self.fpath_mask) / 255.
        return self._mask

    @mask.setter
    def mask(self, value: torch.Tensor):
            self._mask = value

    @property
    def fpath_rgb(self):
        return self.path_raw.joinpath(self.meta.rfpath_rgb)

    @property
    def rgb(self):
        if self._rgb is None:
            self._rgb = torchvision.io.read_image(str(self.fpath_rgb), mode=torchvision.io.ImageReadMode.RGB)
        return self._rgb

    @rgb.setter
    def rgb(self, value: torch.Tensor):
            self._rgb = value

    @property
    def fpath_depth(self):
        raise self.path_raw.joinpath(self.meta.rfpath_depth)

    @property
    def depth(self):
        if self._depth is None:
            self._depth = torchvision.io.read_image(str(self.fpath_depth), mode=torchvision.io.ImageReadMode.UNCHANGED)
        return self._depth

    @depth.setter
    def depth(self, value: torch.Tensor):
            self._depth = value

    @property
    def depth_mask(self):
        if self._depth_mask is None:
            self._depth_mask = read_image(self.path_raw.joinpath(self.meta.rfpath_depth_mask))
        return self._depth_mask

    @depth_mask.setter
    def depth_mask(self, value: torch.Tensor):
            self._depth_mask = value

    @property
    def category(self):
        return self.meta.category

    @property
    def name(self):
        return self.meta.name

    @property
    def categories(self):
        return self.meta.categories