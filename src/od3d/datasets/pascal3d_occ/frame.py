import logging

logger = logging.getLogger(__name__)
import torch
from typing import List
from pathlib import Path
from dataclasses import dataclass
from od3d.cv.geometry.transform import transf4x4_from_spherical
from od3d.datasets.pascal3d.frame import Pascal3DFrame, Pascal3DFrameMeta

from od3d.datasets.frame_meta import OD3D_FrameMetaMaskMixin
from od3d.datasets.frame import (
    OD3D_FRAME_MASK_TYPES,
    OD3D_FRAME_DEPTH_TYPES,
    OD3D_FRAME_KPTS2D_ANNOT_TYPES,
)
from od3d.datasets.object import (
    OD3D_CAM_TFORM_OBJ_TYPES,
    OD3D_FRAME_MASK_TYPES,
    OD3D_MESH_TYPES,
    OD3D_MESH_FEATS_TYPES,
    OD3D_MESH_FEATS_DIST_REDUCE_TYPES,
    OD3D_TFROM_OBJ_TYPES,
)


@dataclass
class Pascal3D_OccFrameMeta(OD3D_FrameMetaMaskMixin, Pascal3DFrameMeta):
    pass


class Pascal3D_OccFrame(Pascal3DFrame):
    def __init__(
        self,
        path_raw: Path,
        path_preprocess: Path,
        modalities,
        name_unique: str,
        path_meshes: Path,
        all_categories: List[str],
        depth_type: OD3D_FRAME_DEPTH_TYPES,
        mask_type: OD3D_FRAME_MASK_TYPES,
        cam_tform4x4_obj_type: OD3D_CAM_TFORM_OBJ_TYPES,
        kpts2d_annot_type: OD3D_FRAME_KPTS2D_ANNOT_TYPES,
        tform_obj_type: OD3D_TFROM_OBJ_TYPES,
        mesh_type: OD3D_MESH_TYPES,
        mesh_feats_type: OD3D_MESH_FEATS_TYPES,
        mesh_feats_dist_reduce_type: OD3D_MESH_FEATS_DIST_REDUCE_TYPES,
    ):
        super().__init__(
            path_raw=path_raw,
            path_preprocess=path_preprocess,
            modalities=modalities,
            name_unique=name_unique,
            all_categories=all_categories,
            depth_type=depth_type,
            mask_type=mask_type,
            cam_tform4x4_obj_type=cam_tform4x4_obj_type,
            kpts2d_annot_type=kpts2d_annot_type,
            tform_obj_type=tform_obj_type,
            mesh_type=mesh_type,
            mesh_feats_type=mesh_feats_type,
            mesh_feats_dist_reduce_type=mesh_feats_dist_reduce_type,
        )
        self.path_meshes = path_meshes

    meta_type = Pascal3D_OccFrameMeta
    # @property
    # def mask(self):
    #     if self._mask is None:
    #         fpath = self.fpath_mask
    #         annot = np.load(fpath)
    #         self._mask = (torch.from_numpy(annot['mask'])[None,] / 255. - torch.from_numpy(annot['occluder_mask'])[None,] * 1.).clamp(0., 1.) # occluder_mask, mask
    #     return self._mask
    # @mask.setter
    # def mask(self, value: torch.Tensor):
    #         self._mask = value

    # @property
    # def cam_tform4x4_obj(self):
    #     if self._cam_tform4x4_obj is None:
    #         self._cam_tform4x4_obj = torch.Tensor(self.meta.l_cam_tform4x4_obj)
    #         self._cam_tform4x4_obj[2, 3] *= PASCAL3D_SCALE_NORMALIZE_TO_REAL[self.category]
    #     return self._cam_tform4x4_obj

    # @property
    # def kpts3d(self):
    #     if self._kpts3d is None:
    #         self._kpts3d = self.meta.kpts3d * PASCAL3D_SCALE_NORMALIZE_TO_REAL[self.category]
    #     return self._kpts3d

    # @property
    # def fpath_mesh(self):
    #     return self.path_meshes.parent.joinpath(self.meta.rfpath_mesh)

    # @property
    # def mesh(self):
    #     if self._mesh is None:
    #         self._mesh = Mesh.load_from_file(fpath=self.fpath_mesh, scale=PASCAL3D_SCALE_NORMALIZE_TO_REAL[self.category])
    #     return self._mesh
    def get_fpath_mesh(self, mesh_type=None):
        if mesh_type is None:
            mesh_type = self.mesh_type
        if mesh_type == OD3D_MESH_TYPES.META:
            return self.path_meshes.joinpath(self.meta.rfpath_mesh)
        else:
            return self.path_preprocess.joinpath(
                self.get_rfpath_pp_categorical_mesh(
                    mesh_type=mesh_type,
                    category=self.category,
                ),
            )

    @staticmethod
    def calc_cam_tform_obj(azimuth, elevation, theta, distance):
        cam_tform4x4_obj = transf4x4_from_spherical(
            azim=torch.Tensor([azimuth]),
            elev=torch.Tensor([elevation]),
            theta=torch.Tensor([theta]),
            dist=torch.Tensor([distance]),
        )[0]
        # cam_tform4x4_obj[0, :] = cam_tform4x4_obj[0, :]
        # cam_tform4x4_obj[1, :] = -cam_tform4x4_obj[1, :]
        # cam_tform4x4_obj[2, :] = -cam_tform4x4_obj[2, :]
        # cam_tform4x4_obj[2, 2:3] = -cam_tform4x4_obj[2, 2:3]
        return cam_tform4x4_obj
