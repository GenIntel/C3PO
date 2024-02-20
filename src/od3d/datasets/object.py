import logging
logger = logging.getLogger(__name__)
from od3d.datasets.meta import OD3D_Meta
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import torch
from enum import Enum
from typing import List
# from od3d.datasets.frame import OD3D_FRAME_MODALITIES

@dataclass
class OD3D_Object(ABC):
    name_unique: str
    path_raw: Path
    path_preprocess: Path
    meta_type = OD3D_Meta
    @property
    def meta(self):
        return self.meta_type.load_from_meta_with_name_unique(path_meta=self.path_meta, name_unique=self.name_unique)

    @property
    def name(self):
        return self.meta.name

    @property
    def path_meta(self):
        return self.path_preprocess.joinpath("meta")

class OD3D_CAM_TFORM_OBJ_TYPES(str, Enum):
    META = 'meta'
    SFM = 'sfm'

class OD3D_FRAME_MASK_TYPES(str, Enum):
    META = 'meta'
    SAM = 'sam'
    SAM_SFM_RAYS_CENTER3D = 'sam_sfm_rays_center3d'

class OD3D_MESH_TYPES(str, Enum):
    META = 'meta'
    CONVEX500 = 'convex500'
    ALPHA500 = 'alpha500'
    CUBOID500 = 'cuboid500'

class OD3D_MESH_FEATS_TYPES(str, Enum):
    M_DINOV2_VITB14_FROZEN_BASE_NO_NORM_T_CENTERZOOM512_R_ACC = 'M_dinov2_vitb14_frozen_base_T_centerzoom512_R_acc'
    M_DINOV2_FROZEN_BASE_NO_NORM_T_CENTERZOOM512_R_ACC = 'M_dinov2_frozen_base_no_norm_T_centerzoom512_R_acc'
    M_DINOV2_FROZEN_BASE_T_CENTERZOOM512_R_ACC = 'M_dinov2_frozen_base_T_centerzoom512_R_acc'
    M_DINO_VITS8_FROZEN_BASE_NO_NORM_T_CENTERZOOM512_R_ACC = 'M_dino_vits8_frozen_base_no_norm_T_centerzoom512_R_acc'
    M_RESNET50_FROZEN_BASE_NO_NORM_T_CENTERZOOM512_R_ACC = 'M_resnet50_frozen_base_no_norm_T_centerzoom512_R_acc'

class OD3D_MESH_FEATS_DIST_REDUCE_TYPES(str, Enum):
    AVG = 'avg'
    AVG50 = 'avg50'
    MIN = 'min'
    MIN_AVG = 'min_avg'

class OD3D_PCL_TYPES(str, Enum):
    META = 'meta'
    META_MASK = 'meta_mask'
    SFM = 'sfm'
    SFM_MASK = 'sfm_mask'

class OD3D_TFROM_OBJ_TYPES(str, Enum):
    RAW = 'raw'
    CENTER3D = 'center3d'
    LABEL3D_ZSP = 'label3d_zsp'
    LABEL3D_ZSP_CUBOID = 'label3d_zsp_cuboid'
    LABEL3D = 'label3d'
    LABEL3D_CUBOID = 'label3d_cuboid'
    # ALIGNED7D = 'aligned7d'

class OD3D_SEQUENCE_SFM_TYPES(str, Enum):
    META = 'meta'
    DROID = 'droid'
    COLMAP = 'colmap'

@dataclass
class OD3D_TformObjMixin():
    tform_obj_type: OD3D_TFROM_OBJ_TYPES

@dataclass
class OD3D_FrameModalitiesMixin():
    modalities: List

# note these classes can be inherited by frame and sequence classes
@dataclass
class OD3D_CamTform4x4ObjTypeMixin(OD3D_Object):
    cam_tform4x4_obj_type: OD3D_CAM_TFORM_OBJ_TYPES #  = OD3D_CAM_TFORM_OBJ_TYPES.META

@dataclass
class OD3D_MaskTypeMixin(OD3D_Object):
    mask_type: OD3D_FRAME_MASK_TYPES #  = OD3D_FRAME_MASK_TYPES.META

@dataclass
class OD3D_MeshTypeMixin(OD3D_Object):
    mesh_type: OD3D_MESH_TYPES

@dataclass
class OD3D_MeshFeatsTypeMixin(OD3D_Object):
    mesh_feats_type: OD3D_MESH_FEATS_TYPES
    mesh_feats_dist_reduce_type: OD3D_MESH_FEATS_DIST_REDUCE_TYPES


@dataclass
class OD3D_PCLTypeMixin(OD3D_Object):
    pcl_type: OD3D_PCL_TYPES


@dataclass
class OD3D_SequenceSfMTypeMixin(OD3D_Object):
    sfm_type: OD3D_SEQUENCE_SFM_TYPES # = OD3D_SEQUENCE_SFM_TYPES.DROID

