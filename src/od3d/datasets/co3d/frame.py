from od3d.datasets.co3d.enum import CAM_TFORM_OBJ_SOURCES, CUBOID_SOURCES
from od3d.datasets.dataset import OD3D_Dataset, OD3D_FRAME_MODALITIES, OD3D_Frame
from omegaconf import DictConfig, OmegaConf
from co3d.dataset.data_types import (
    load_dataclass_jgzip, FrameAnnotation, SequenceAnnotation)

import torch
from od3d.cv.geometry.transform import transf4x4_from_rot3x3_and_transl3, transf3d_broadcast
from pathlib import Path
from od3d.cv.io import read_image, read_co3d_depth_image

from od3d.cv.geometry.transform import inv_tform4x4

from od3d.cv.geometry.transform import transf3d_broadcast, tform4x4

import logging
logger = logging.getLogger(__name__)
from dataclasses import dataclass
import torch.utils.data


@dataclass
class CO3D_Frame(OD3D_Frame):
    sequence_name: str
    path_meta: Path
    frame_number: int
    depth_scale: float
    frame_type: str
    _sequence = None
    return_cam_tform4x4_cuboid_front = False
    # the following variables can be configured dynamically
    _config = None

    @staticmethod
    def create_with_config(config, **kwargs):
        co3d_seq = CO3D_Frame(**kwargs)
        co3d_seq._config = config
        return co3d_seq
    @property
    def config(self):
        if self._config is None:
            self._config = OmegaConf.create()
            self._config.cam_tform_obj_source = CAM_TFORM_OBJ_SOURCES.KPTS2D_ORIENT_AND_PCL.value
            self._config.cuboid_source = CUBOID_SOURCES.KPTS2D_ORIENT_AND_PCL.value
        return self._config

    @property
    def name_unique(self):
        return f'{self.sequence_name}_{self.name}'
    @property
    def sequence(self):
        if self._sequence is None:
            from od3d.datasets.co3d.sequence import CO3D_Sequence
            sequence_config = OmegaConf.load(self.path_meta.joinpath(self.sequence_name + '.yaml'))
            self._sequence = CO3D_Sequence.create_with_config(**sequence_config, config=self.config)
        return self._sequence

    @property
    def depth(self):
        if self._depth is None:
            self._depth = read_co3d_depth_image(self.path_dataset.joinpath(self.rfpath_depth)) * self.depth_scale
        return self._depth

    @property
    def cam_tform4x4_obj(self):
        if self._cam_tform4x4_obj is None:
            if self.config.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.FIRST_FRAME:
                self._cam_tform4x4_obj = torch.Tensor(self.l_cam_tform4x4_obj)
            elif self.config.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.FRONT_FRAME_AND_PCL:
                self._cam_tform4x4_obj = tform4x4(torch.Tensor(self.l_cam_tform4x4_obj),
                                                  inv_tform4x4(self.sequence.cuboid_front_tform4x4_obj))
            elif self.config.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.KPTS2D_ORIENT_AND_PCL:
                self._cam_tform4x4_obj = tform4x4(torch.Tensor(self.l_cam_tform4x4_obj),
                                                  inv_tform4x4(self.sequence.cuboid_front_tform4x4_obj))

        return self._cam_tform4x4_obj

    @staticmethod
    def load_from_raw(path_co3d: Path, path_preprocess: Path, path_meta: Path, frame_annotation: FrameAnnotation):
        path_dataset = path_co3d
        category = frame_annotation.image.path.split('/')[0]
        sequence_name = frame_annotation.sequence_name
        frame_number = frame_annotation.frame_number
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
        return CO3D_Frame(path_dataset=path_dataset, path_preprocess=path_preprocess, path_meta=path_meta, category=category, frame_number=frame_number, frame_type=frame_type,
                   name=name, rfpath_mask=rfpath_mask, rfpath_depth=rfpath_depth, rfpath_depth_mask=rfpath_depth_mask,
                   rfpath_rgb=rfpath_rgb, H=H, W=W, l_size=l_size, l_cam_intr4x4=l_cam_intr4x4,
                   sequence_name=sequence_name,
                   l_cam_tform4x4_obj=l_cam_tform4x4_obj, depth_scale=depth_scale)
