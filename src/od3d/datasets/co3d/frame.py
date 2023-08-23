from od3d.datasets.co3d.enum import CAM_TFORM_OBJ_SOURCES, CUBOID_SOURCES
from od3d.datasets.frame import OD3D_FrameMeta, OD3D_Frame
from od3d.datasets.dataset import OD3D_Dataset, OD3D_FRAME_MODALITIES
from omegaconf import DictConfig, OmegaConf
from co3d.dataset.data_types import (
    load_dataclass_jgzip, FrameAnnotation, SequenceAnnotation)

import torch
from od3d.cv.geometry.transform import transf4x4_from_rot3x3_and_transl3
from pathlib import Path
from od3d.cv.io import read_image, read_co3d_depth_image

from od3d.cv.geometry.transform import inv_tform4x4

from od3d.cv.geometry.transform import tform4x4

import logging
logger = logging.getLogger(__name__)
from dataclasses import dataclass
import torch.utils.data
from typing import List
import numpy as np

from od3d.datasets.frame import OD3D_FrameMeta, \
    OD3D_FrameMetaSequenceMixin, OD3D_FrameMetaCategoryMixin, OD3D_FrameMetaRGBMixin, \
    OD3D_FrameMetaSizeMixin, OD3D_FrameMetaMaskMixin, OD3D_FrameMetaDepthMixin, OD3D_FrameMetaDepthMaskMixin, \
    OD3D_FrameMetaCamTform4x4ObjMixin, OD3D_FrameMetaCamIntr4x4Mixin

@dataclass
class CO3D_FrameMeta(OD3D_FrameMetaCamTform4x4ObjMixin, OD3D_FrameMetaCamIntr4x4Mixin,
                     OD3D_FrameMetaCategoryMixin, OD3D_FrameMetaSequenceMixin, OD3D_FrameMetaDepthMaskMixin,
                     OD3D_FrameMetaDepthMixin, OD3D_FrameMetaMaskMixin, OD3D_FrameMetaRGBMixin,
                     OD3D_FrameMetaSizeMixin, OD3D_FrameMeta):
    depth_scale: float
    frame_type: str

    @property
    def name_unique(self):
        return f'{self.category}/{self.sequence_name}/{self.name}'

    @staticmethod
    def get_name_unique_with_category_sequence_and_name(category: str, sequence_name: str, name: str):
        return f'{category}/{sequence_name}/{name}'

    @staticmethod
    def load_from_raw(frame_annotation: FrameAnnotation):
        category = frame_annotation.image.path.split('/')[0]
        sequence_name = frame_annotation.sequence_name
        frame_type = frame_annotation.meta['frame_type']
        name = f'{frame_annotation.frame_number}'

        rfpath_mask = Path(frame_annotation.mask.path)

        rfpath_rgb = Path(frame_annotation.image.path)

        depth_scale = frame_annotation.depth.scale_adjustment
        rfpath_depth = Path(frame_annotation.depth.path)

        rfpath_depth_mask = Path(frame_annotation.depth.mask_path)

        cam_tform4x4_obj = transf4x4_from_rot3x3_and_transl3(rot3x3=torch.Tensor(frame_annotation.viewpoint.R).T, transl3=torch.Tensor(frame_annotation.viewpoint.T))
        default_tform_t3d = torch.Tensor([[-1., 0., 0., 0.],
                                         [0., -1., 0., 0.],
                                         [0., 0., 1., 0.],
                                         [0., 0., 0., 1.]])
        cam_tform4x4_obj = torch.bmm(default_tform_t3d[None,], cam_tform4x4_obj[None,])[0]

        H, W = frame_annotation.image.size
        size = torch.Tensor([H, W])

        s = min(H, W)
        focal_length = torch.Tensor(frame_annotation.viewpoint.focal_length) * s / 2.
        principal_point = -torch.Tensor(frame_annotation.viewpoint.principal_point) * s / 2. + size.flip(dims=(0,)) / 2.
        cam_intr4x4 = torch.Tensor([[focal_length[0], 0., principal_point[0], 0.],
                           [0., focal_length[1], principal_point[1], 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]])

        l_size = size.tolist()
        l_cam_intr4x4 = cam_intr4x4.tolist()
        l_cam_tform4x4_obj = cam_tform4x4_obj.tolist()
        return CO3D_FrameMeta(category=category, frame_type=frame_type,
                   name=name, rfpath_mask=rfpath_mask, rfpath_depth=rfpath_depth, rfpath_depth_mask=rfpath_depth_mask,
                   rfpath_rgb=rfpath_rgb, l_size=l_size, l_cam_intr4x4=l_cam_intr4x4,
                   sequence_name=sequence_name,
                   l_cam_tform4x4_obj=l_cam_tform4x4_obj, depth_scale=depth_scale)

    @staticmethod
    def get_subset_frames_names_uniform(frames_names, count_max_per_sequence=None):
        if count_max_per_sequence is not None:
            frames_names = [frames_names[fid] for fid in
                            np.linspace(0, len(frames_names) - 1, count_max_per_sequence).astype(int).tolist()]
        return frames_names

    @staticmethod
    def load_from_meta_with_category_sequence_and_frame_name(path_meta: Path, category: str, sequence_name:str, frame_name: str):
        return CO3D_FrameMeta.load_from_meta_with_rfpath(path_meta=path_meta,
                                                         rfpath=CO3D_FrameMeta.get_rfpath_frame_meta_with_category_sequence_and_frame_name(category=category, sequence_name=sequence_name, name=frame_name))

    @staticmethod
    def get_rfpath_frame_meta_with_category_sequence_and_frame_name(category: str, sequence_name: str, name: str):
        return CO3D_FrameMeta.get_rfpath_metas().joinpath(category, sequence_name, name + '.yaml')

    @staticmethod
    def get_path_frames_meta_with_category_sequence(path_meta: Path, category: str, sequence: str):
        return path_meta.joinpath(CO3D_FrameMeta.get_rpath_frames_meta_with_category_sequence_name(category=category, sequence=sequence))

    @staticmethod
    def get_rpath_frames_meta_with_category_sequence_name(category: str, sequence: str):
        return CO3D_FrameMeta.get_rfpath_metas().joinpath(category, sequence)

    @staticmethod
    def get_fpath_frame_meta_with_category_sequence_and_frame_name(path_meta: Path, category: str, sequence_name: str, name: str):
        return path_meta.joinpath(CO3D_FrameMeta.get_rfpath_frame_meta_with_category_sequence_and_frame_name(category=category, sequence_name=sequence_name, name=name))

    """
    @staticmethod
    def load_from_meta_with_rfpath(path_meta: Path, rfpath: Path):
        return CO3D_FrameMeta(**CO3D_FrameMeta.load_omega_conf_with_rfpath(path_meta=path_meta, rfpath=rfpath))

    
    @staticmethod
    def load_from_meta_with_name_unique(path_meta: Path, name_unique: str):

        return CO3D_FrameMeta.load_from_meta_with_rfpath(path_meta=path_meta, rfpath=CO3D_FrameMeta.get_rfpath_frame_meta_with_category_sequence_and_frame_name(category=category, sequence_name=sequence_name, name=name))
    """


    """"
    @staticmethod
    def meta_rfpath_to_sequence_name(rfpath: Path):
        return rfpath.parent.stem
    @staticmethod
    def meta_rfpath_to_category(rfpath: Path):
        return rfpath.parent.parent.stem

    @staticmethod
    def meta_fpath_to_name(fpath: Path):
        return fpath.stem
    @staticmethod
    def meta_rfpath_to_name(rfpath: Path):
        return rfpath.stem





    """

    """
    @staticmethod
    def get_frames_names_with_category_sequence_name(path_meta, category, sequence_name, count_max_per_sequence=None):
        frames_fpath = list(
            CO3D_FrameMeta.get_path_frames_meta_with_category_sequence(path_meta=path_meta, category=category,
                                                                       sequence=sequence_name).iterdir())
        frames_names = [CO3D_FrameMeta.meta_fpath_to_name(fpath) for fpath in frames_fpath]

        frames_names = CO3D_FrameMeta.get_subset_frames_names_uniform(frames_names,
                                                                      count_max_per_sequence=count_max_per_sequence)

        frames_names = sorted(frames_names, key=lambda frame_name: int(frame_name))
        return frames_names
    @staticmethod
    def get_dict_category_sequence_name_frames_names(path_meta, categories, dict_category_sequences_names,
                                                     count_max_per_sequence=None, dict_category_sequence_name_frames_names=None):
        new_dict_category_sequence_name_frames_names = {}
        for category in dict_category_sequences_names.keys():
            if dict_category_sequence_name_frames_names is None or (category in dict_category_sequence_name_frames_names.keys()):
                new_dict_category_sequence_name_frames_names[category] = {}
                for sequence_name in dict_category_sequences_names[category]:
                    if dict_category_sequence_name_frames_names is None or \
                            dict_category_sequence_name_frames_names[category] is None or \
                            dict_category_sequence_name_frames_names[category][sequence_name] is None:
                        frames_names = CO3D_FrameMeta.get_frames_names_with_category_sequence_name(path_meta=path_meta,
                                                                                                   category=category,
                                                                                                   sequence_name=sequence_name,
                                                                                                   count_max_per_sequence=
                                                                                                   count_max_per_sequence)
                    else:
                        frames_names = dict_category_sequence_name_frames_names[category][sequence_name]
                        frames_names = CO3D_FrameMeta.get_subset_frames_names_uniform(frames_names,
                                                                                      count_max_per_sequence=count_max_per_sequence)
                    new_dict_category_sequence_name_frames_names[category][sequence_name] = frames_names
        return new_dict_category_sequence_name_frames_names

    @staticmethod
    def get_dict_category_sequence_name_frames_names_with_names_unique(names_unique: List[str]):
        dict_category_sequence_name_frames_names = {}
        for name_unique in names_unique:
            category, sequence_name, name = name_unique.split('/')
            if category not in dict_category_sequence_name_frames_names:
                dict_category_sequence_name_frames_names[category] = {}
            if sequence_name not in dict_category_sequence_name_frames_names[category]:
                dict_category_sequence_name_frames_names[category][sequence_name] = []
            dict_category_sequence_name_frames_names[category][sequence_name].append(name_unique)
        return dict_category_sequence_name_frames_names
    """
    """
    # legacy code
    @staticmethod
    def get_rfpaths_frames_meta(path_meta, categories, map_category_sequences_names, count_max_per_sequence=None, return_sequences_lengths=False):
        sequences_lengths = []
        frames_rfpaths = []
        for category in categories:
            for sequence_name in map_category_sequences_names[category]:
                frames_fpaths_partial = list(CO3D_FrameMeta.get_path_frames_meta_with_category_sequence(path_meta=path_meta, category=category, sequence=sequence_name).iterdir())
                frames_rfpaths_partial = [CO3D_FrameMeta.get_rfpath_frame_meta_with_category_sequence_name(category=category, sequence=sequence_name, name=fpath.stem) for fpath in frames_fpaths_partial]
                if count_max_per_sequence is not None:
                    frames_rfpaths_partial = [frames_rfpaths_partial[fid] for fid in np.linspace(0, len(frames_rfpaths_partial)-1, count_max_per_sequence).astype(int).tolist()]
                frames_rfpaths_partial = sorted(frames_rfpaths_partial, key=lambda rfpath: int(rfpath.stem))
                frames_rfpaths += frames_rfpaths_partial
                sequences_lengths.append(len(frames_rfpaths_partial))
        if return_sequences_lengths:
            return frames_rfpaths, sequences_lengths
        else:
            return frames_rfpaths
    """

class CO3D_Frame(OD3D_Frame):
    def __init__(self, path_raw: Path, path_preprocess: Path, meta: CO3D_FrameMeta, path_meta: Path,
                 modalities: List[OD3D_FRAME_MODALITIES], categories: List[str],
                 cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.KPTS2D_ORIENT_AND_PCL.value,
                 cuboid_source=CUBOID_SOURCES.KPTS2D_ORIENT_AND_PCL.value):
        super().__init__(path_raw=path_raw, path_preprocess=path_preprocess, path_meta=path_meta, meta=meta, modalities=modalities, categories=categories)

        self.meta = meta
        self.path_meta: Path = path_meta
        self._sequence = None
        self.cam_tform_obj_source = cam_tform_obj_source
        self.cuboid_source = cuboid_source
        # the following variables can be configured dynamically
        # self._config = None

    @staticmethod
    def create_with_config(config, **kwargs):
        co3d_seq = CO3D_Frame(**kwargs)
        co3d_seq._config = config
        return co3d_seq

    #@property
    #def config(self):
    #    if self._config is None:
    #        self._config = OmegaConf.create()
    #        self._config.cam_tform_obj_source = CAM_TFORM_OBJ_SOURCES.KPTS2D_ORIENT_AND_PCL.value
    #       self._config.cuboid_source = CUBOID_SOURCES.KPTS2D_ORIENT_AND_PCL.value
    #    return self._config


    @property
    def sequence(self):
        if self._sequence is None:
            from od3d.datasets.co3d.sequence import CO3D_Sequence, CO3D_SequenceMeta
            sequence_meta = CO3D_SequenceMeta.load_from_meta_with_category_and_name(path_meta=self.path_meta,
                                                                                    category=self.category,
                                                                                    name=self.meta.sequence_name)
            self._sequence = CO3D_Sequence(path_raw=self.path_raw, path_preprocess=self.path_preprocess,
                                           path_meta=self.path_meta, meta=sequence_meta, modalities=self.modalities,
                                           categories=self.all_categories, cam_tform_obj_source=self.cam_tform_obj_source,
                                           cuboid_source=self.cuboid_source)
        return self._sequence

    @property
    def depth(self):
        if self._depth is None:
            self._depth = read_co3d_depth_image(self.fpath_depth) * self.meta.depth_scale
        return self._depth

    @property
    def cam_tform4x4_obj(self):
        if self._cam_tform4x4_obj is None:
            if self.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.CO3D:
                self._cam_tform4x4_obj = torch.Tensor(self.meta.l_cam_tform4x4_obj)
            elif self.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.FRONT_FRAME_AND_PCL:
                self._cam_tform4x4_obj = tform4x4(torch.Tensor(self.meta.l_cam_tform4x4_obj),
                                                  inv_tform4x4(self.sequence.cuboid_front_tform4x4_obj))
            elif self.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.KPTS2D_ORIENT_AND_PCL:
                self._cam_tform4x4_obj = tform4x4(torch.Tensor(self.meta.l_cam_tform4x4_obj),
                                                  inv_tform4x4(self.sequence.cuboid_front_tform4x4_obj))
            elif self.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.LIMITS3D:
                self._cam_tform4x4_obj = tform4x4(torch.Tensor(self.meta.l_cam_tform4x4_obj),
                                                  inv_tform4x4(self.sequence.cuboid_front_tform4x4_obj))

        return self._cam_tform4x4_obj


