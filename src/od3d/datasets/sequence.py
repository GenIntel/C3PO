import logging
logger = logging.getLogger(__name__)

from od3d.datasets.frame import OD3D_Frame
from od3d.datasets.frame_meta import OD3D_FrameMeta

#from od3d.datasets.frame_meta import OD3D_FrameMeta
# from od3d.datasets.frame import OD3D_Frame, OD3D_FrameCamIntr4x4Mixin, OD3D_FrameCategoryMixin
from od3d.datasets.object import OD3D_Object, OD3D_SequenceSfMTypeMixin, OD3D_SEQUENCE_SFM_TYPES
from od3d.datasets.sequence_meta import OD3D_SequenceMeta
from od3d.data.ext_dicts import unroll_nested_dict, rollup_flattened_dict
from dataclasses import dataclass
from typing import List
import numpy as np
from pathlib import Path
from enum import Enum
import torch

@dataclass
class OD3D_Sequence(OD3D_Object):
    frame_type = OD3D_Frame
    _frames_names = None
    _frames_names_unique = None

    @property
    def meta(self):
        return OD3D_SequenceMeta.load_from_meta_with_name_unique(path_meta=self.path_meta, name_unique=self.name_unique)

    @property
    def first_frame(self):
        return self.get_frame_by_index(index=0)

    @property
    def frames_names_unique(self):
        if self._frames_names_unique is None:
            dict_nested_frames = rollup_flattened_dict({self.name_unique: None})
            dict_nested_frames = OD3D_FrameMeta.complete_nested_metas(path_meta=self.path_meta,
                                                                      dict_nested_metas=dict_nested_frames)
            self._frames_names_unique = OD3D_FrameMeta.unroll_nested_metas(dict_nested_meta=dict_nested_frames)
        return self._frames_names_unique
    @property
    def frames_names(self):
        if self._frames_names is None:
            self._frames_names = [frame_name.split('/')[-1] for frame_name in self._frames_names_unique]
        return self._frames_names

    @staticmethod
    def get_subset_frames_names_uniform(frames_names, count_max_per_sequence=None):
        if count_max_per_sequence is not None:
            frames_names = [frames_names[fid] for fid in
                            np.linspace(0, len(frames_names) - 1, count_max_per_sequence).astype(int).tolist()]
        return frames_names

    @property
    def frames_count(self):
        return len(self.frames_names)

    def get_frames(self, frames_ids=None):
        if frames_ids is None:
            frames_ids = list(range(self.frames_count))
        frames = [self.get_frame_by_index(frame_id) for frame_id in frames_ids]
        return frames

    def get_frame_by_index(self, index: int):
        return self.get_frame_by_name_unique(self.frames_names_unique[index])

    def get_frame_by_name_unique(self, frame_name_unique: str):
        from dataclasses import fields
        frame_fields_names = [field.name for field in fields(self.frame_type)]
        sequence_fields = fields(self)
        all_attrs_except_name_unique = {field.name: getattr(self, field.name) for field in sequence_fields
                                        if field.name != 'name_unique' and field.name in frame_fields_names}
        return self.frame_type(name_unique=frame_name_unique, **all_attrs_except_name_unique)


@dataclass
class OD3D_SequenceCategoryMixin(OD3D_Sequence):
    #frame_type = OD3D_FrameCategoryMixin
    all_categories: List[str]

    @property
    def category(self):
        return self.meta.category
    @property
    def category_id(self):
        return self.all_categories.index(self.category)

@dataclass
class OD3D_SequenceSfMMixin(OD3D_SequenceSfMTypeMixin, OD3D_Sequence):
    #frame_type = OD3D_FrameCamIntr4x4Mixin

    @property
    def path_sfm(self):
        if self.sfm_type == OD3D_SEQUENCE_SFM_TYPES.META:
            return self.path_raw.joinpath(self.meta.rfpath_sfm)
        else:
            return self.path_sfm_root.joinpath(self.name_unique)

    @property
    def path_sfm_root(self):
        return self.path_preprocess.joinpath("sfm", f'{self.sfm_type}')
    @property
    def path_sfm_cams_tform4x4_obj(self):
        return self.path_sfm.joinpath('cam_tform4x4_obj')

    @property
    def fpath_sfm_pcl(self):
        return self.path_sfm.joinpath('pcl.ply')

    @property
    def fpath_sfm_pcl_obj(self):
        return self.path_sfm.joinpath('pcl_obj.ply')

    @property
    def fpath_sfm_rays_center3d(self):
        return self.path_sfm.joinpath('rays_center3d.pt')

    def get_cam_tform4x4_obj(self, frame_name):
        return torch.load(self.path_sfm_cams_tform4x4_obj.joinpath(f'{frame_name}.pt'))

    def get_sfm_rays_center3d(self):
        return torch.load(self.fpath_sfm_rays_center3d)

    def preprocess_sfm(self, override=False):
        if not override and self.path_sfm.exists():
            logger.info(f'path sfm already exists at {self.path_sfm}')
            return

        if self.sfm_type == OD3D_SEQUENCE_SFM_TYPES.DROID:
            path_in = self.path_raw.joinpath('frames', self.name_unique)
            path_out_root = self.path_sfm_root #  self.path_preprocess.joinpath('droid_slam')
            rpath_out = Path(self.name_unique)

            #from od3d.models.model import OD3D_Model
            #from od3d.cv.transforms.transform import OD3D_Transform
            #from od3d.cv.transforms.sequential import SequentialTransform
            #model = OD3D_Model.create_by_name('sam')
            #model.cuda()
            #model.eval()
            #transform = SequentialTransform([OD3D_Transform.create_by_name(''), model.transform])


            from od3d.cv.reconstruction.droid_slam import run_droid_slam
            run_droid_slam(path_rgbs=path_in, path_out_root=path_out_root, rpath_out=rpath_out, cam_intr4x4=self.first_frame.cam_intr4x4)
        else:
            raise NotImplementedError(f'sfm_type {self.sfm_type} not implemented')

    # @classmethod
    # def get_rfpath_droid_slam(cls):
    #     return Path("droid_slam")
    #
    # @property
    # def path_droid_slam(self):
    #     return self.path_preprocess.joinpath(OD3D_SequenceDroidSlamMixin.get_rfpath_droid_slam(), self.name_unique)

