import logging
logger = logging.getLogger(__name__)

import torch
from od3d.cv.geometry.transform import tform4x4
from od3d.cv.io import read_image
import torchvision
from dataclasses import dataclass
from typing import List, Union
from enum import Enum
from od3d.cv.geometry.mesh import Mesh
from od3d.datasets.object import OD3D_Object, OD3D_CamTform4x4ObjTypeMixin, OD3D_MaskTypeMixin, OD3D_MeshTypeMixin, \
    OD3D_CAM_TFORM_OBJ_TYPES, OD3D_MESH_TYPES, OD3D_FRAME_MASK_TYPES
from od3d.datasets.frame_meta import OD3D_FrameMeta
from pathlib import Path

# from od3d.datasets.sequence import OD3D_Sequence

class OD3D_FRAME_MODALITIES(str, Enum):
    NAME = 'name'
    CAM_INTR4X4 = 'cam_intr4x4'
    CAM_TFORM4X4_OBJ = 'cam_tform4x4_obj'
    CAM_TFORM4X4_OBJS = 'cam_tform4x4_objs'
    CATEGORY = 'category'
    CATEGORIES = 'categories'
    PCL = 'pcl'
    RGB = 'rgb'
    MASK = 'mask'
    MASKS = 'masks'
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
    RAYS_CENTER3D = 'rays_center3d'

class OD3D_FRAME_KPTS2D_ANNOT_TYPES(str, Enum):
    META = 'meta'
    LABEL = 'label'

@dataclass
class OD3D_Frame(OD3D_Object):
    meta_type = OD3D_FrameMeta

    #@property
    #def meta(self):
    #    return OD3D_FrameMeta.load_from_meta_with_name_unique(path_meta=self.path_meta, name_unique=self.name_unique)


    # pass
    # def __init__(self, path_raw: Path, path_preprocess: Path, name_unique: Path, modalities: List[OD3D_FRAME_MODALITIES], categories: List[str]):
    #     self.path_raw: Path = path_raw
    #     self.path_preprocess: Path = path_preprocess
    #     self.all_categories = categories
    #     self.name_unique = name_unique
        # self.modalities = modalities
        #self.meta: OD3D_FrameMetaClasses = meta
        # self.path_meta: Path = path_meta
        # #self.category_id = categories.index(self.category)
        # self.item_id = None
        # self._rgb = None
        # self._depth = None
        # self._depth_mask = None
        # self._kpts2d_orient = None
        # self._mesh = None
        # self._kpts3d = None

@dataclass
class OD3D_FrameMaskMixin(OD3D_MaskTypeMixin):
    _mask = None

    @property
    def fpath_mask(self):
        if self.mask_type == OD3D_FRAME_MASK_TYPES.META:
            return self.path_raw.joinpath(self.meta.rfpath_mask)
        else:
            return self.path_preprocess.joinpath("mask", f"{self.mask_type}", f"{self.name_unique}.png")

    def write_mask(self, value: torch.Tensor):
        if self.fpath_mask.parent.exists() is False:
            self.fpath_mask.parent.mkdir(parents=True, exist_ok=True)
        if value.dtype == torch.bool:
            value_write = value.to(torch.uint8).detach().cpu() * 255
        elif value.dtype == torch.uint8:
            value_write = value.detach().cpu()
        else:
            value_write = (value * 255).to(torch.uint8).detach().cpu()
        torchvision.io.write_png(input=value_write, filename=str(self.fpath_mask))
        self._mask = value

    def get_mask(self):
        if self._mask is None:
            self._mask = read_image(self.fpath_mask) / 255.
        return self._mask

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value: torch.Tensor):
            self._mask = value

@dataclass
class OD3D_FrameSizeMixin(OD3D_Object):
    _size = None

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

@dataclass
class OD3D_FrameMaskRGBMixin(OD3D_FrameSizeMixin):
    _mask_rgb = None

    @property
    def mask_rgb(self):
        if self._mask_rgb is None:
            self._mask_rgb = torch.ones(size=(1, self.H, self.W), dtype=torch.bool)
        return self._mask_rgb

    @mask_rgb.setter
    def mask_rgb(self, value: torch.Tensor):
        self._mask_rgb = value



@dataclass
class OD3D_FrameCamTform4x4ObjMixin(OD3D_CamTform4x4ObjTypeMixin):
    _cam_tform4x4_obj = None

    @property
    def cam_tform4x4_obj(self):
        if self._cam_tform4x4_obj is None:
            if self.cam_tform4x4_obj_type == OD3D_CAM_TFORM_OBJ_TYPES.META:
                self._cam_tform4x4_obj = self.meta.cam_tform4x4_obj
            elif self.cam_tform4x4_obj_type == OD3D_CAM_TFORM_OBJ_TYPES.SFM:
                self._cam_tform4x4_obj = self.sequence.get_cam_tform4x4_obj(f"{Path(self.name_unique).stem}")
            else:
                raise ValueError(f"cam_tform4x4_obj_type {self.cam_tform4x4_obj_type} not supported")
        return self._cam_tform4x4_obj

    @cam_tform4x4_obj.setter
    def cam_tform4x4_obj(self, value: torch.Tensor):
            self._cam_tform4x4_obj = value

@dataclass
class OD3D_FrameMeshMixin(OD3D_MeshTypeMixin):
    _mesh = None

    @property
    def fpath_mesh(self):
        if self.mesh_type == OD3D_MESH_TYPES.META:
            return self.path_raw.joinpath(self.meta.rfpath_mesh)
        else:
            return self.path_preprocess.joinpath("mesh", f"{self.mesh_type}", f"{self.name_unique}.ply")

    @property
    def mesh(self):
        if self._mesh is None:
            self._mesh = Mesh.load_from_file(fpath=self.fpath_mesh)
        return self._mesh

    @mesh.setter
    def mesh(self, value: torch.Tensor):
            self._mesh = value


@dataclass
class OD3D_FrameCamIntr4x4Mixin(OD3D_Frame):
    _cam_intr4x4 = None

    @property
    def cam_intr4x4(self):
        if self._cam_intr4x4 is None:
            self._cam_intr4x4 = self.meta.cam_intr4x4
        return self._cam_intr4x4

    @cam_intr4x4.setter
    def cam_intr4x4(self, value: torch.Tensor):
            self._cam_intr4x4 = value

@dataclass
class OD3D_CamProj4x4ObjMixin(OD3D_FrameCamTform4x4ObjMixin, OD3D_FrameCamIntr4x4Mixin):
    @property
    def cam_proj4x4_obj(self):
        return tform4x4(self.cam_intr4x4, self.cam_tform4x4_obj)

@dataclass
class OD3D_FrameCategoryMixin(OD3D_Object):
    all_categories: List[str]

    @property
    def category(self):
        return self.meta.category
    @property
    def category_id(self):
        return self.all_categories.index(self.category)

@dataclass
class OD3D_FrameCategoriesMixin(OD3D_Object):
    all_categories: List[str]

    @property
    def categories(self):
        return self.meta.categories

    @property
    def categories_ids(self):
        return torch.LongTensor([self.all_categories.index(cat) for cat in self.categories])

@dataclass
class OD3D_FrameBBoxMixin(OD3D_Object):
    _bbox = None
    @property
    def bbox(self):
        if self._bbox is None:
            self._bbox = self.meta.bbox
        return self._bbox

    @bbox.setter
    def bbox(self, value: torch.Tensor):
            self._bbox = value


@dataclass
class OD3D_FrameKpts2d3dMixin(OD3D_Object):
    kpts2d_annot_type: OD3D_FRAME_KPTS2D_ANNOT_TYPES
    _kpts3d = None
    _kpts2d_annot = None
    _kpts2d_annot_vsbl = None

    @property
    def kpts_names(self):
        return self.meta.kpts_names

    @property
    def kpts2d_annot(self):
        if self._kpts2d_annot is None:
            if self.kpts2d_annot_type == OD3D_FRAME_KPTS2D_ANNOT_TYPES.META:
                self._kpts2d_annot = self.meta.kpts2d_annot
            else:
                self._kpts2d_annot = torch.load(self.fpath_kpts2d_annot)
        return self._kpts2d_annot

    @property
    def fpath_kpts2d_annot(self):
        if self.kpts2d_annot_type == OD3D_FRAME_KPTS2D_ANNOT_TYPES.META:
            raise ValueError("Meta kpts2d_annot is not saved in a file")
        else:
            return self.path_preprocess.joinpath("kpts2d_annot", self.kpts2d_annot_type, f"{self.name_unique}.pt")
    @property
    def kpts2d_annot_labeled(self):
        if self.kpts2d_annot_type == OD3D_FRAME_KPTS2D_ANNOT_TYPES.META:
            return True
        else:
            return

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

    # @property
    # def fpath_kpts2d_orient(self):
    #     return self.path_preprocess.joinpath("labels", "kpts2d_orient", f"{self.name_unique}.pt")
    #
    # @property
    # def kpts2d_orient_labeled(self):
    #     return self.fpath_kpts2d_orient.exists()
    # @property
    # def kpts2d_orient(self):
    #     kpts2d_orient = torch.load(self.fpath_kpts2d_orient)
    #     return kpts2d_orient


#class OD3D_Kpts3dMixin(OD3D_Object):

class OD3D_FrameRGBMixin(OD3D_Object):
    _rgb = None

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

class OD3D_FrameDepthMixin(OD3D_Object):
    _depth = None

    @property
    def fpath_depth(self):
        return self.path_raw.joinpath(self.meta.rfpath_depth)

    @property
    def depth(self):
        if self._depth is None:
            self._depth = torchvision.io.read_image(str(self.fpath_depth), mode=torchvision.io.ImageReadMode.UNCHANGED)
        return self._depth

    @depth.setter
    def depth(self, value: torch.Tensor):
            self._depth = value

class OD3D_FrameDepthMaskMixin(OD3D_Object):
    _depth_mask = None
    @property
    def depth_mask(self):
        if self._depth_mask is None:
            self._depth_mask = read_image(self.path_raw.joinpath(self.meta.rfpath_depth_mask))
        return self._depth_mask

    @depth_mask.setter
    def depth_mask(self, value: torch.Tensor):
            self._depth_mask = value

@dataclass
class OD3D_FrameSequenceMixin(OD3D_Object):
    sequence_type = None #  OD3D_Sequence

    @property
    def sequence_name(self):
        return self.meta.sequence_name

    @property
    def sequence_name_unique(self):
        return self.meta.sequence_name_unique

    @property
    def sequence(self):
        from dataclasses import fields
        frame_fields = fields(self)
        sequence_fields_names = [field.name for field in fields(self.sequence_type)]
        all_attrs_except_name_unique = {field.name: getattr(self, field.name) for field in frame_fields
                                        if field.name != 'name_unique' and field.name in sequence_fields_names}
        return self.sequence_type(name_unique=self.sequence_name_unique, **all_attrs_except_name_unique)

@dataclass
class OD3D_FrameRaysCenter3dMixin(OD3D_FrameSequenceMixin):
    _rays_center3d = None

    @property
    def rays_center3d(self):
        if self._rays_center3d is None:
            self._rays_center3d = self.sequence.get_sfm_rays_center3d()
        return self._rays_center3d

@dataclass
class OD3D_FrameSubsetMixin(OD3D_Object):
    @property
    def subset(self):
        return self.meta.subset

from od3d.datasets.frame_meta import OD3D_FrameMeta



@dataclass
class OD3D_FrameCamIntr4x4Mixin(OD3D_Frame):
    _cam_intr4x4 = None

    @property
    def cam_intr4x4(self):
        if self._cam_intr4x4 is None:
            self._cam_intr4x4 = self.meta.cam_intr4x4
        return self._cam_intr4x4

    @cam_intr4x4.setter
    def cam_intr4x4(self, value: torch.Tensor):
            self._cam_intr4x4 = value


OD3D_FrameClasses = Union[OD3D_Object, OD3D_FrameCategoryMixin, OD3D_FrameCategoriesMixin,
                          OD3D_FrameCamTform4x4ObjMixin, OD3D_CamProj4x4ObjMixin, OD3D_FrameCamIntr4x4Mixin,
                          OD3D_FrameMaskMixin, OD3D_FrameDepthMixin, OD3D_FrameDepthMaskMixin,
                          OD3D_FrameSizeMixin, OD3D_FrameRGBMixin,
                          OD3D_FrameMeshMixin, OD3D_FrameSequenceMixin, OD3D_FrameSubsetMixin,
                          OD3D_FrameBBoxMixin, OD3D_FrameKpts2d3dMixin, OD3D_FrameMaskRGBMixin]