import logging
logger = logging.getLogger(__name__)

from od3d.datasets.frame import OD3D_Frame
from od3d.datasets.frame_meta import OD3D_FrameMeta

#from od3d.datasets.frame_meta import OD3D_FrameMeta
# from od3d.datasets.frame import OD3D_Frame, OD3D_FrameCamIntr4x4Mixin, OD3D_FrameCategoryMixin
from od3d.datasets.object import OD3D_Object, OD3D_SequenceSfMTypeMixin, OD3D_SEQUENCE_SFM_TYPES, \
    OD3D_PCLTypeMixin, OD3D_PCL_TYPES, OD3D_MeshTypeMixin, OD3D_MESH_TYPES, OD3D_FrameModalitiesMixin, \
    OD3D_TformObjMixin, OD3D_TFROM_OBJ_TYPES
from od3d.datasets.sequence_meta import OD3D_SequenceMeta
from od3d.data.ext_dicts import unroll_nested_dict, rollup_flattened_dict
from dataclasses import dataclass
from typing import List
import numpy as np
from pathlib import Path
from enum import Enum
import torch
from od3d.cv.reconstruction.clean import get_pcl_clean_with_masks
from od3d.cv.io import write_pts3d_with_colors_and_normals
from od3d.datasets.object import OD3D_CAM_TFORM_OBJ_TYPES


@dataclass
class OD3D_Sequence(OD3D_FrameModalitiesMixin, OD3D_Object):
    frame_type = OD3D_Frame
    _frames_names = None
    _frames_names_unique = None

    # @property
    # def meta(self):
    #     return sel.frame_type.load_from_meta_with_name_unique(path_meta=self.path_meta, name_unique=self.name_unique)

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
            self._frames_names = [frame_name.split('/')[-1] for frame_name in self.frames_names_unique]
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


    def get_cams(self, cam_tform4x4_obj_type: OD3D_CAM_TFORM_OBJ_TYPES=None, cams_count=5, show_imgs=True):
        cams_tform4x4_world = []
        cams_intr4x4 = []
        cams_imgs = []
        frames_count = len(self.frames_names)
        if cams_count == -1:
            step_size = 1
        else:
            step_size = (frames_count // cams_count) + 1
        for c in range(0, frames_count, step_size):
            frame = self.get_frame_by_index(c)
            cams_tform4x4_world.append(
                frame.get_cam_tform4x4_obj(cam_tform4x4_obj_type=cam_tform4x4_obj_type))

            cams_intr4x4.append(frame.cam_intr4x4)
            if show_imgs:
                cams_imgs.append(frame.rgb)
        return cams_tform4x4_world, cams_intr4x4, cams_imgs

@dataclass
class OD3D_SequenceCategoryMixin(OD3D_Sequence):
    #frame_type = OD3D_FrameCategoryMixin
    all_categories: List[str]
    map_categories_to_od3d = None

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
        return self.path_sfm.joinpath(self.dname_sfm_cams_tform4x4_obj)

    @property
    def dname_sfm_cams_tform4x4_obj(self):
        return 'cam_tform4x4_obj'

    @property
    def fpath_sfm_pcl(self):
        return self.path_sfm.joinpath(self.fname_sfm_pcl)

    @property
    def fname_sfm_pcl(self):
        return 'pcl.ply'
    @property
    def fpath_sfm_rays_center3d(self):
        return self.path_sfm.joinpath(self.fname_sfm_rays_center3d)

    @property
    def fname_sfm_rays_center3d(self):
        return 'rays_center3d.pt'

    def get_sfm_cam_tform4x4_obj(self, frame_name):
        return torch.load(self.path_sfm_cams_tform4x4_obj.joinpath(f'{frame_name}.pt'))

    def get_sfm_rays_center3d(self):
        return torch.load(self.fpath_sfm_rays_center3d)

    def preprocess_sfm(self, override=False):
        if not override and self.path_sfm.exists():
            logger.info(f'path sfm already exists at {self.path_sfm}')
            return
        else:
            logger.info(f'preprocessing sfm for {self.name_unique} with type {self.sfm_type}')

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
            run_droid_slam(path_rgbs=path_in, path_out_root=path_out_root, rpath_out=rpath_out,
                           cam_intr4x4=self.first_frame.cam_intr4x4, pcl_fname=self.fname_sfm_pcl,
                           rays_center3d_fname=self.fname_sfm_rays_center3d,
                           cam_tform_obj_dname=self.dname_sfm_cams_tform4x4_obj )
        else:
            raise NotImplementedError(f'sfm_type {self.sfm_type} not implemented')

    # @classmethod
    # def get_rfpath_droid_slam(cls):
    #     return Path("droid_slam")
    #
    # @property
    # def path_droid_slam(self):
    #     return self.path_preprocess.joinpath(OD3D_SequenceDroidSlamMixin.get_rfpath_droid_slam(), self.name_unique)

import re
from od3d.cv.io import read_pts3d_with_colors_and_normals
import open3d
from od3d.cv.geometry.mesh import Mesh
from od3d.cv.geometry.downsample import random_sampling, voxel_downsampling

from od3d.cv.io import get_default_device


@dataclass
class OD3D_SequencePCLMixin(OD3D_PCLTypeMixin, OD3D_SequenceSfMMixin):
    pts3d = None
    pts3d_colors = None
    pts3d_normals = None

    @property
    def fpath_pcl(self):
        return self.get_fpath_pcl()

    @property
    def fname_sfm_mask_pcl(self):
        return 'pcl_mask.ply'

    @property
    def fpath_sfm_mask_pcl(self):
        return self.path_sfm.joinpath(self.fname_sfm_mask_pcl)

    def get_fpath_pcl(self, pcl_type=None):
        if pcl_type is None:
            pcl_type = self.pcl_type

        if pcl_type == OD3D_PCL_TYPES.META:
            return self.path_raw.joinpath(self.meta.rfpath_pcl)
        elif pcl_type == OD3D_PCL_TYPES.SFM:
            return self.fpath_sfm_pcl
        elif pcl_type == OD3D_PCL_TYPES.SFM_MASK:
            return self.fpath_sfm_mask_pcl
        else:
            raise NotImplementedError(f'pcl_source {pcl_type} not implemented')

    def read_pcl(self, pcl_type=None, device='cpu'):
        pts3d, pts3d_colors, pts3d_normals = read_pts3d_with_colors_and_normals(fpath=self.get_fpath_pcl(pcl_type=pcl_type), device=device)
        if pcl_type is None or pcl_type == self.pcl_type:
            self.pts3d, self.pts3d_colors, self.pts3d_normals = pts3d, pts3d_colors, pts3d_normals
        return pts3d, pts3d_colors, pts3d_normals

    def get_pcl(self, pcl_type=None, clone=False, device='cpu'):
        if self.pts3d is not None and (pcl_type is None or pcl_type == self.pcl_type):
            pts3d, pts3d_colors, pts3d_normals = self.pts3d, self.pts3d_colors, self.pts3d_normals
        else:
            pts3d, pts3d_colors, pts3d_normals = self.read_pcl(pcl_type=pcl_type, device=device)
        if not clone:
            return pts3d, pts3d_colors, pts3d_normals
        else:
            return pts3d.clone(), pts3d_colors.clone(), pts3d_normals.clone()

    def preprocess_pcl(self, override=False):
        if self.pcl_type == OD3D_PCL_TYPES.META:
            logger.info('no need to preprocess pcl for meta pcl type')
            return
        elif self.pcl_type == OD3D_PCL_TYPES.SFM:
            logger.info('no need to preprocess pcl for sfm pcl type')
            return
        elif self.pcl_type == OD3D_PCL_TYPES.SFM_MASK:

            if not override and self.fpath_sfm_mask_pcl.exists():
                logger.info(f'fpath sfm mask pcl already exists at {self.fpath_sfm_mask_pcl}')
                return

            frames = self.get_frames()
            device = get_default_device()

            masks = torch.stack([frame.get_mask() for frame in frames], dim=0).to(device=device)
            cams_intr4x4 = torch.stack([frame.cam_intr4x4 for frame in frames], dim=0).to(device=device)
            cams_tform4x4_obj = torch.stack([frame.cam_tform4x4_obj for frame in frames], dim=0).to(device=device)

            pts3d, pts3d_colors, pts3d_normals = self.get_pcl(pcl_type=OD3D_PCL_TYPES.SFM, clone=True, device=device)
            pts3d, pts3d_mask = get_pcl_clean_with_masks(pcl=pts3d, masks=masks,
                                                         cams_intr4x4=cams_intr4x4,
                                                         cams_tform4x4_obj=cams_tform4x4_obj,
                                                         pts3d_prob_thresh=0.6,
                                                         pts3d_max_count=20000,
                                                         pts3d_count_min=10,
                                                         return_mask=True)
            pts3d_colors = pts3d_colors[pts3d_mask]
            pts3d_normals = pts3d_normals[pts3d_mask]

            write_pts3d_with_colors_and_normals(fpath=self.fpath_sfm_mask_pcl,
                                                pts3d=pts3d.detach().cpu(),
                                                pts3d_colors=pts3d_colors.detach().cpu(),
                                                pts3d_normals=pts3d_normals.detach().cpu())


from od3d.cv.geometry.transform import transf3d_broadcast, transf3d_normal_broadcast
from od3d.datasets.object import OD3D_TFROM_OBJ_TYPES


@dataclass
class OD3D_SequenceTformObjMixin(OD3D_TformObjMixin, OD3D_SequencePCLMixin):

    def read_pcl(self, pcl_type=None, device='cpu'):
        pts3d, pts3d_colors, pts3d_normals = super().read_pcl(pcl_type=pcl_type, device=device)

        tform_obj = self.get_tform_obj()
        if tform_obj is not None:
            pts3d = transf3d_broadcast(pts3d=pts3d, transf4x4=tform_obj)
            pts3d_normals = transf3d_normal_broadcast(normals3d=pts3d_normals, transf4x4=tform_obj)

        if pcl_type is None or pcl_type == self.pcl_type:
            self.pts3d, self.pts3d_colors, self.pts3d_normals = pts3d, pts3d_colors, pts3d_normals
        return pts3d, pts3d_colors, pts3d_normals

    # note: all meshes are saved in the labeled format
    # def read_mesh(self, mesh_type=None):
    #     mesh = super().read_mesh(mesh_type=mesh_type)
    #     tform_obj = self.get_tform_obj()
    #     if tform_obj is not None:
    #         mesh.verts = transf3d_broadcast(pts3d=mesh.verts, transf4x4=tform_obj)
    #     if mesh_type is None or mesh_type == self.mesh_type:
    #         self.mesh = mesh
    #     return mesh

    def get_tform_obj(self, tform_obj_type: OD3D_TFROM_OBJ_TYPES=None):
        if tform_obj_type is None:
            tform_obj_type = self.tform_obj_type

        if tform_obj_type == OD3D_TFROM_OBJ_TYPES.RAW:
            return None
        else:
            fpath_tform_obj = self.get_fpath_tform_obj(tform_obj_type=tform_obj_type)
            if fpath_tform_obj.exists():
                return torch.load(self.get_fpath_tform_obj(tform_obj_type=tform_obj_type))
            else:
                logger.warning(f'tform_obj_type {tform_obj_type} does not exists at {fpath_tform_obj}')
                return None

    def get_fpath_tform_obj(self, tform_obj_type=None):
        if tform_obj_type is None:
            tform_obj_type = self.tform_obj_type
        return self.path_preprocess.joinpath('tform_obj', f'{tform_obj_type}', f'{self.sfm_type}', f'{self.pcl_type}',
                                             self.name_unique, 'tform_obj.pt')

    def write_tform_obj(self, tform_obj: torch.Tensor, fpath_tform_obj=None):
        if fpath_tform_obj is None:
            fpath_tform_obj = self.get_fpath_tform_obj()


        if fpath_tform_obj.parent.exists() is False:
            fpath_tform_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tform_obj.detach().cpu(), f=fpath_tform_obj)

    def preprocess_tform_obj(self, override=False, tform_obj_type=None):

        if tform_obj_type is None:
            tform_obj_type = self.tform_obj_type

        from od3d.cv.label.axis import label_axis_in_pcl

        fpath_tform_obj = self.get_fpath_tform_obj(tform_obj_type=tform_obj_type)
        if fpath_tform_obj.exists() and not override:
            logger.info(f'Label axis already exists {fpath_tform_obj}, override disabled.')
            return

        if tform_obj_type == OD3D_TFROM_OBJ_TYPES.RAW:
            logger.info(f'No need to preprocess tform_obj for raw tform_obj type')
            return
        elif tform_obj_type == OD3D_TFROM_OBJ_TYPES.LABEL3D:
            fpath_tform_obj.parent.mkdir(parents=True, exist_ok=True)
            cams_tform4x4_world, cams_intr4x4, cams_imgs = self.get_cams(cams_count=4, show_imgs=False, cam_tform4x4_obj_type=OD3D_CAM_TFORM_OBJ_TYPES.SFM)
            pts3d, pts3d_colors, pts3d_normals = super().read_pcl()
            while True:
                prev_tform_obj = self.get_tform_obj(tform_obj_type=tform_obj_type)
                axis_pts3d = label_axis_in_pcl(pts3d=pts3d,
                                               pts3d_colors=pts3d_colors,
                                               prev_labeled_pcl_tform_pcl=prev_tform_obj,
                                               cams_tform4x4_world=cams_tform4x4_world,
                                               cams_intr4x4=cams_intr4x4,
                                               cams_imgs=cams_imgs)

                from od3d.cv.geometry.fit.axis_tform_from_pts3d import axis_tform4x4_obj_from_pts3d


                if axis_pts3d is None or axis_pts3d.shape != (3, 2, 3):
                    if prev_tform_obj is not None:
                        logger.warning('not overriding previous tform_obj')
                        break

                    logger.warning(f'axis_pts3d is None or axis_pts3d.shape != (3, 2, 3) {axis_pts3d.shape if axis_pts3d is not None else None}')
                    continue

                tform_obj = axis_tform4x4_obj_from_pts3d(axis_pts3d=axis_pts3d)
                tform_obj[:3, 3] = -pts3d.mean(dim=0)

                if not (torch.linalg.det(tform_obj[:3, :3]) - 1.).abs() <= 1e-5:
                    logger.warning(f'determinant is not close to 1. {torch.linalg.det(tform_obj[:3, :3])}')
                    continue

                self.write_tform_obj(tform_obj=tform_obj, fpath_tform_obj=fpath_tform_obj)
                break
        elif tform_obj_type == OD3D_TFROM_OBJ_TYPES.LABEL3D_CUBOID:

            from od3d.datasets.enum import OD3D_CATEGORIES_SIZES_IN_M
            from od3d.cv.geometry.fit.cuboid import fit_cuboid_to_pts3d
            from od3d.cv.geometry.transform import tform4x4

            size = OD3D_CATEGORIES_SIZES_IN_M[self.map_categories_to_od3d[self.category]]

            self.preprocess_tform_obj(override=override, tform_obj_type=OD3D_TFROM_OBJ_TYPES.LABEL3D)

            tform_obj = self.get_tform_obj(tform_obj_type=OD3D_TFROM_OBJ_TYPES.LABEL3D)
            pts3d, pts3d_colors, pts3d_normals = super().read_pcl()
            pts3d_label3d = transf3d_broadcast(pts3d=pts3d, transf4x4=tform_obj)
            _, obj_cuboid_tform_obj = fit_cuboid_to_pts3d(pts3d=pts3d_label3d, size=size,
                                                          optimize_rot=False,
                                                          optimize_transl=True)
            tform_obj = tform4x4(obj_cuboid_tform_obj, tform_obj)

            self.write_tform_obj(tform_obj=tform_obj, fpath_tform_obj=fpath_tform_obj)
        else:
            raise NotImplementedError(f'tform_obj_type {tform_obj_type} not implemented')

        # if axis_pcl is not None and axis_pcl.shape == (3, 2, 3):
        #
        #     logger.info(f'storing axis labeled, cuboid tform, and cuboid ')
        #     torch.save(axis_pcl, f=fpath_axis_droid_slam)
        #
        #     self.fpath_labeled_obj_tform_obj.parent.mkdir(parents=True, exist_ok=True)
        #     torch.save(pcl_labeled_tform_pcl, self.fpath_labeled_obj_tform_obj)
        #
        #     size = OD3D_CATEGORIES_SIZES_IN_M[MAP_CATEGORIES_CO3D_TO_OD3D[self.category]]
        #     pcl_labeled_tform_pts3d = transf3d_broadcast(
        #         pts3d=self.get_pcl(),
        #         transf4x4=pcl_labeled_tform_pcl)
        #     pcl_labeled_cuboid, pcl_labeled_cuboid_tform_pcl_labeled = \
        #         fit_cuboid_to_pts3d(pts3d=pcl_labeled_tform_pts3d, size=size, optimize_rot=False,
        #                             optimize_transl=True)
        #
        #     self.fpath_obj_labeled_cuboid.parent.mkdir(parents=True, exist_ok=True)
        #     pcl_labeled_cuboid.write_to_file(fpath=self.fpath_obj_labeled_cuboid)
        #
        #     self.fpath_labeled_cuboid_obj_tform_labeled_obj.parent.mkdir(parents=True, exist_ok=True)
        #     torch.save(pcl_labeled_cuboid_tform_pcl_labeled.detach().cpu(),
        #                f=self.fpath_labeled_cuboid_obj_tform_labeled_obj)
        # else:
        #     logger.info(f'not storing labeled axis.')


@dataclass
class OD3D_SequenceMeshMixin(OD3D_MeshTypeMixin, OD3D_SequenceTformObjMixin):
    mesh = None

    def get_fpath_mesh(self, mesh_type=None):
        if mesh_type is None:
            mesh_type = self.mesh_type
        if mesh_type == OD3D_MESH_TYPES.META:
            return self.path_raw.joinpath(self.meta.rfpath_mesh)
        else:
            return self.path_preprocess.joinpath("mesh", f'{self.mesh_type}', f'{self.tform_obj_type}', f'{self.pcl_type}', f'{self.sfm_type}', self.name_unique, 'mesh.ply')

    @property
    def fpath_mesh(self):
        return self.get_fpath_mesh()

    def read_mesh(self, mesh_type=None):
        mesh = Mesh.load_from_file(fpath=self.get_fpath_mesh(mesh_type=mesh_type))
        if mesh_type is None or mesh_type == self.mesh_type:
            self.mesh = mesh
        return mesh

    def get_mesh(self, mesh_type=None, clone=False):
        if (mesh_type is None or  mesh_type == self.mesh_type) and self.mesh is not None:
            mesh = self.mesh
        else:
            mesh = self.read_mesh(mesh_type=mesh_type)

        if not clone:
            return mesh
        else:
            return mesh.clone()

    def preprocess_mesh(self, override=False):
        if self.fpath_mesh.exists() and not override:
            logger.warning(f'mesh already exists {self.fpath_mesh}')
            return
        else:
            logger.info(f'preprocessing mesh for {self.name_unique} with type {self.mesh_type}')


        match = re.match(r"([a-z]+)([0-9]+)", self.mesh_type, re.I)
        if match and len(match.groups()) == 2:
            mesh_type, mesh_vertices_count = match.groups()
            mesh_vertices_count = int(mesh_vertices_count)
        else:
            msg = f'could not retrieve mesh type and vertices count from mesh name {self.mesh_type}'
            raise Exception(msg)

        # fpath_pcl = self.get_fpath_pcl(pcl_source=self.pcl_source)
        # if not fpath_pcl.exists():
        #     self.preprocess_pcl(override=True)
        #
        # if fpath_pcl.exists():
        #     pts3d, pts3d_colors, pts3d_normals = read_pts3d_with_colors_and_normals(fpath_pcl)
        # else:
        #     logger.warning(f'could not preprocess mesh, due to fpath to pcl {fpath_pcl} does not exists.')
        #     return None

        pts3d, pts3d_colors, pts3d_normals = self.get_pcl(clone=True)
        N = pts3d.shape[0]
        if N < 4:
            logger.warning(f'Could not estimate mesh for sequence {self.name_unique} due to too few points in raw pcl {N}')
            return

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        o3d_pcl = open3d.geometry.PointCloud()
        o3d_pcl.points = open3d.utility.Vector3dVector(pts3d.detach().cpu().numpy())
        o3d_pcl.normals = open3d.utility.Vector3dVector(pts3d_normals.detach().cpu().numpy())  # invalidate existing normals
        o3d_pcl.colors = open3d.utility.Vector3dVector(pts3d_colors.detach().cpu().numpy())

        ## DEBUG BLOCK START
        #open3d.visualization.draw_geometries([o3d_pcl])
        # scams = 30
        # frames = self.get_frames()
        # H, W = frames[0].H, frames[0].W
        # rgb = torch.stack([frame.rgb[:, :int(H * 0.9), :int(W * 0.9)] for frame in frames], dim=0).to(device=self.device)
        # cams_intr4x4 = torch.stack([frame.cam_intr4x4 for frame in frames], dim=0).to(device=self.device)
        # cams_tform4x4_obj = torch.stack([frame.get_cam_tform4x4_obj(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.PCL) for frame in frames], dim=0).to(device=self.device)
        # show_scene(pts3d=[pts3d], pts3d_colors=[pts3d_colors], cams_tform4x4_world=cams_tform4x4_obj[::scams], cams_intr4x4=cams_intr4x4[::scams], cams_imgs=rgb[::scams])
        ## DEBUG BLOCK END

        # #### OPTION 1: CONVEX HULL
        if mesh_type == 'convex':
            o3d_obj_mesh, _ = o3d_pcl.compute_convex_hull()
            o3d_obj_mesh.compute_vertex_normals()
            logger.info(o3d_obj_mesh)
            o3d_obj_mesh = o3d_obj_mesh.remove_unreferenced_vertices()
            logger.info(o3d_obj_mesh)
            o3d_obj_mesh = o3d_obj_mesh.simplify_quadric_decimation(mesh_vertices_count)
            logger.info(o3d_obj_mesh)
            obj_mesh = Mesh.from_o3d(o3d_obj_mesh, device=device)

        elif mesh_type == 'poisson':
            # #### OPTION 2: POISSON (requires normals)

            o3d_obj_mesh, densities = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(o3d_pcl, depth=9,
                                                                                                   linear_fit=False)
            vertices_to_remove = densities < np.quantile(densities, 0.05)
            o3d_obj_mesh.remove_vertices_by_mask(vertices_to_remove)
            logger.info(o3d_obj_mesh)
            o3d_obj_mesh = o3d_obj_mesh.remove_unreferenced_vertices()
            logger.info(o3d_obj_mesh)
            o3d_obj_mesh = o3d_obj_mesh.simplify_quadric_decimation(mesh_vertices_count)
            logger.info(o3d_obj_mesh)
            obj_mesh = Mesh.from_o3d(o3d_obj_mesh, device=device)

        elif mesh_type == 'alpha':
            # #### OPTION 3: ALPHA_SHAPE
            pts3d = random_sampling(pts3d, pts3d_max_count=20000)
            particle_size = torch.cdist(pts3d[None,], pts3d[None,]).quantile(dim=-1, q=5. / len(pts3d)).mean()
            alpha = 10 * particle_size
            o3d_obj_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(o3d_pcl, alpha)
            logger.info(o3d_obj_mesh)
            o3d_obj_mesh = o3d_obj_mesh.remove_unreferenced_vertices()
            logger.info(o3d_obj_mesh)
            o3d_obj_mesh = o3d_obj_mesh.simplify_quadric_decimation(mesh_vertices_count)
            logger.info(o3d_obj_mesh)
            obj_mesh = Mesh.from_o3d(o3d_obj_mesh, device=device)

        elif mesh_type == 'voxel':
            #### OPTION 4: VOXEL GRID
            from pytorch3d.ops.marching_cubes import marching_cubes
            voxel_grid, voxel_grid_range, voxel_grid_offset = voxel_downsampling(pts3d_cls=pts3d, K=mesh_vertices_count * 2,
                                                                                 return_voxel_grid=True, min_steps=2)
            # vol_batch(N, D, H, W) ->  (X, Y, Z).permute(2, 0, 1)
            verts, faces = marching_cubes(vol_batch=voxel_grid.permute(2, 0, 1)[None,] * 1., return_local_coords=True)
            faces = faces[0].to(device=device)
            verts = (verts[0].to(device=device) + 1) / 2.

            obj_mesh = Mesh(verts=voxel_grid_offset[None,] + voxel_grid_range[None,] * verts, faces=faces)

        elif mesh_type == 'cuboid':
            from od3d.cv.geometry.fit.cuboid import fit_cuboid_to_pts3d
            cuboids, _ = fit_cuboid_to_pts3d(pts3d=pts3d,
                                              optimize_rot=False,
                                              optimize_transl=False)

            obj_mesh = cuboids.get_mesh_with_id(0)

        else:
            msg = f'Unknown mesh type {mesh_type}'
            raise Exception(msg)

        # visualization...
        ## DEBUG BLOCK START
        # open3d.visualization.draw(o3d_pcl)
        # open3d.visualization.draw(o3d_obj_mesh)
        ## DEBUG BLOCK END



        obj_mesh.write_to_file(fpath=self.fpath_mesh)

        ## DEBUG BLOCK START
        # scams = 30
        # frames = self.get_frames()
        # H, W = frames[0].H, frames[0].W
        # rgb = torch.stack([frame.rgb[:, :int(H * 0.9), :int(W * 0.9)] for frame in frames], dim=0).to(device=self.device)
        # cams_intr4x4 = torch.stack([frame.cam_intr4x4 for frame in frames], dim=0).to(device=self.device)
        # cams_tform4x4_obj = torch.stack([frame.cam_tform4x4_obj for frame in frames], dim=0).to(device=self.device)
        # show_scene(meshes=[obj_mesh], pts3d=[pts3d], pts3d_colors=[pts3d_colors], cams_tform4x4_world=cams_tform4x4_obj[::scams], cams_intr4x4=cams_intr4x4[::scams], cams_imgs=rgb[::scams])
        # ## DEBUG BLOCK END
