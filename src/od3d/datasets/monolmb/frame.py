import logging
logger = logging.getLogger(__name__)
from typing import List
import torch
from pathlib import Path

from od3d.datasets.object import OD3D_CAM_TFORM_OBJ_TYPES
from od3d.datasets.frame_meta import OD3D_FrameMeta, OD3D_FrameMetaRGBMixin,  \
    OD3D_FrameMetaCategoryMixin, OD3D_FrameMetaMaskMixin, OD3D_FrameMetaSizeMixin, OD3D_FrameMetaSequenceMixin

from od3d.datasets.frame import OD3D_Frame, OD3D_FrameSizeMixin, OD3D_FrameRGBMixin, OD3D_FrameMaskMixin, \
    OD3D_FrameMaskRGBMixin, OD3D_FrameRaysCenter3dMixin, \
    OD3D_FrameSequenceMixin, OD3D_FrameCategoryMixin, OD3D_FrameCamIntr4x4Mixin, OD3D_FrameCamTform4x4ObjMixin, \
    OD3D_CamProj4x4ObjMixin, OD3D_FRAME_MASK_TYPES
from dataclasses import dataclass
import numpy as np


@dataclass
class MonoLMB_FrameMeta(OD3D_FrameMetaRGBMixin, OD3D_FrameMetaSizeMixin, OD3D_FrameMetaCategoryMixin,
                        OD3D_FrameMetaSequenceMixin, OD3D_FrameMeta):
    #
    # @property
    # def name_unique(self):
    #     return f'{self.category}/{self.sequence_name}/{self.name}'
    @staticmethod
    def load_from_raw(name: str, category: str, sequence_name: str, rfpath_rgb: Path, l_size: List):
        return MonoLMB_FrameMeta(rfpath_rgb=rfpath_rgb, category=category, sequence_name=sequence_name, l_size=l_size, name=name)

@dataclass
class MonoLMB_Frame(OD3D_FrameRaysCenter3dMixin, OD3D_CamProj4x4ObjMixin, OD3D_FrameMaskRGBMixin,
                    OD3D_FrameMaskMixin, OD3D_FrameRGBMixin, OD3D_FrameCategoryMixin,
                    OD3D_FrameSequenceMixin, OD3D_FrameSizeMixin, OD3D_Frame):
    meta_type = MonoLMB_FrameMeta
    mask_type = OD3D_FRAME_MASK_TYPES.SAM
    cam_tform4x4_obj_type = OD3D_CAM_TFORM_OBJ_TYPES.SFM

    def __post_init__(self):
        from od3d.datasets.monolmb.sequence import MonoLMB_Sequence
        self.sequence_type = MonoLMB_Sequence


    #@property
    #def meta(self):
    #    return MonoLMB_FrameMeta.load_from_meta_with_name_unique(path_meta=self.path_meta, name_unique=self.name_unique)

    @property
    def cam_intr4x4(self):
        if self._cam_intr4x4 is None:
            self._cam_intr4x4 = torch.eye(4)
            px = self.meta.l_size[1] / 2
            py = self.meta.l_size[0] / 2
            fov = 1.7 # fov in radians
            fx = max(self.meta.l_size[0], self.meta.l_size[1]) / np.tan(fov / 2)
            fy = fx
            self._cam_intr4x4[0, 0] = fx
            self._cam_intr4x4[1, 1] = fy
            self._cam_intr4x4[0, 2] = px
            self._cam_intr4x4[1, 2] = py
        return self._cam_intr4x4

    # @cam_intr4x4.setter
    #def cam_intr4x4(self, value: torch.Tensor):
    #    self._cam_intr4x4 = value

    # @property
    # def cam_tform4x4_obj(self):
    #     if self._cam_tform4x4_obj is None:
    #         self._cam_tform4x4_obj = torch.eye(4)
    #         logger.warning('cam_tform4x4_obj is not set')
    #     return self._cam_tform4x4_obj
    #
    # @cam_tform4x4_obj.setter
    # def cam_tform4x4_obj(self, value: torch.Tensor):
    #     self._cam_tform4x4_obj = value

    #self.meta = meta
    #self.path_meta: Path = path_meta
    #self._sequence = None
    #self.cam_tform_obj_source = cam_tform_obj_source
    #self.aligned_name = aligned_name
    #self.pcl_source = pcl_source
    #self.cuboid_source = cuboid_source
    #self.mesh_name = mesh_name
    # the following variables can be configured dynamically
    # self._config = None

    # @staticmethod
    # def create_with_config(config, **kwargs):
    #     co3d_seq = MonoLMB_Frame(**kwargs)
    #     co3d_seq._config = config
    #     return co3d_seq

