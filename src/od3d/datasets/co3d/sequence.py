from od3d.datasets.co3d.enum import CUBOID_SOURCES, CAM_TFORM_OBJ_SOURCES, CO3D_CATEGORIES
from od3d.datasets.co3d.frame import CO3D_Frame, CO3D_FrameMeta

from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
import torch.utils.data

import torch.utils.data
import od3d.io

import subprocess

from od3d.datasets.dataset import OD3D_Dataset, OD3D_FRAME_MODALITIES, OD3D_Frame
from omegaconf import DictConfig, OmegaConf
from co3d.dataset.data_types import (
    load_dataclass_jgzip, FrameAnnotation, SequenceAnnotation
)
from od3d.cv.geometry.transform import se3_exp_map, tform4x4
from od3d.cv.geometry.transform import rot3x3

from od3d.cv.geometry.mesh import Meshes
from typing import List
import torchvision
import torch
from pathlib import Path
from od3d.cv.visual.show import show_img
from od3d.cv.io import load_ply, save_ply


from od3d.cv.geometry.transform import proj3d2d_broadcast
from od3d.cv.visual.sample import sample_pxl2d_pts

from od3d.cv.geometry.points_alignment import get_pca_tform_world
from od3d.cv.geometry.transform import transf3d_broadcast, tform4x4, inv_tform4x4
from od3d.cv.geometry.primitives import Cuboids
from od3d.cv.geometry.points_alignment import icp

from dataclasses import dataclass
import torch.utils.data
from od3d.cv.geometry.downsample import voxel_downsampling, random_sampling
import od3d.io
import cv2

@dataclass
class CO3D_SequenceMeta():
    name: str
    category: str
    pcl_pts_count: int
    pcl_quality_score: float
    rfpath_pcl: Path
    viewpoint_quality_score: float

    @staticmethod
    def load_from_raw(sequence_annotation: SequenceAnnotation):
        name = sequence_annotation.sequence_name
        category = sequence_annotation.category
        if sequence_annotation.point_cloud is not None:
            rfpath_pcl = sequence_annotation.point_cloud.path
            pcl_pts_count = sequence_annotation.point_cloud.n_points
            pcl_quality_score = sequence_annotation.point_cloud.quality_score
        else:
            rfpath_pcl = Path('None')
            pcl_pts_count = 0
            pcl_quality_score = float('nan')

        viewpoint_quality_score = sequence_annotation.viewpoint_quality_score

        return CO3D_SequenceMeta(name=name, category=category, rfpath_pcl=rfpath_pcl,
                                 pcl_pts_count=pcl_pts_count, pcl_quality_score=pcl_quality_score,
                                 viewpoint_quality_score=viewpoint_quality_score)

    @staticmethod
    def load_from_meta_with_category_and_name(path_meta: Path, category: str, name: str):
        fpath_meta = CO3D_SequenceMeta.get_fpath_sequence_meta_with_category_and_name(path_meta=path_meta, category=category, name=name)
        if not fpath_meta.exists():
            logger.error(f'Missing meta fpath {fpath_meta}. Preprocess meta before.')
        return CO3D_SequenceMeta(**OmegaConf.load(fpath_meta))

    @staticmethod
    def load_from_meta_with_rfpath(path_meta: Path, rfpath: Path):
        fpath_meta = path_meta.joinpath(rfpath)
        if not fpath_meta.exists():
            logger.error(f'Missing meta fpath {fpath_meta}. Preprocess meta before.')
        return CO3D_SequenceMeta(**OmegaConf.load(fpath_meta))
    @staticmethod
    def get_fpath_sequence_meta_with_rfpath(path_meta: Path, rfpath_meta: Path):
        return path_meta.joinpath(rfpath_meta)

    @staticmethod
    def get_fpath_sequence_meta_with_category_and_name(path_meta: Path, category: str, name: str):
        return path_meta.joinpath(CO3D_SequenceMeta.get_rfpath_sequence_meta_with_category_and_name(category=category, name=name))

    @staticmethod
    def get_rfpath_sequences():
        return Path("sequences")

    @staticmethod
    def get_rfpath_sequences_meta_with_category(category: str):
        return CO3D_SequenceMeta.get_rfpath_sequences().joinpath(category)

    @staticmethod
    def get_rfpath_sequence_meta_with_category_and_name(category: str, name: str):
        return CO3D_SequenceMeta.get_rfpath_sequences_meta_with_category(category=category).joinpath(name + '.yaml')

    @staticmethod
    def get_path_sequences_meta(path_meta: Path):
        return path_meta.joinpath(CO3D_SequenceMeta.get_rfpath_sequences())

    @staticmethod
    def get_path_sequences_meta_with_category(path_meta: Path, category: str):
        return path_meta.joinpath(
            CO3D_SequenceMeta.get_rfpath_sequences_meta_with_category(category=category))


    @staticmethod
    def get_map_category_sequences_names(path_meta, categories):
        map_category_sequences_names = {}
        for category in categories:
            map_category_sequences_names[category] = list(CO3D_SequenceMeta.get_path_sequences_meta_with_category(path_meta=path_meta, category=category).iterdir())
        return map_category_sequences_names

    """ 
    # legacy code
    @staticmethod
    def get_rfpaths_sequences_meta(categories, path_meta=None, map_category_sequences_names=None):
        sequences_rfpaths = []
        #if categories is None:
        #    if path_meta is None:
        #        categories = CO3D_CATEGORIES.list()
        #    else:
        #        categories = list(CO3D_SequenceMeta.get_path_sequences_meta(path_meta=path_meta).iterdir())
        for category in categories:
            if map_category_sequences_names is None or category not in map_category_sequences_names.keys():
                if path_meta is None:
                    logger.error(f'cannot load rfpaths_sequences_meta for category `{category}` as neither map_category_sequences_names nor path_meta are provided')
                    sequences_names = []
                else:
                    sequences_names = list(CO3D_SequenceMeta.get_path_sequences_meta_with_category(path_meta=path_meta, category=category).iterdir())
            else:
                sequences_names = map_category_sequences_names[category]
            for sequence_name in sequences_names:
                sequences_rfpaths.append(CO3D_SequenceMeta.get_rfpath_sequence_meta_with_category_and_name(category=category, name=sequence_name))
        return sequences_rfpaths
    """

    @staticmethod
    def meta_rfpath_to_category(rfpath: Path):
        return rfpath.parent.stem
    @staticmethod
    def meta_rfpath_to_name(rfpath: Path):
        return rfpath.stem

    def get_fpath(self, path_meta):
        return CO3D_SequenceMeta.get_fpath_sequence_meta_with_category_and_name(path_meta=path_meta,
                                                                                category=self.category,
                                                                                name=self.name)
    @property
    def rfpath(self):
        return CO3D_SequenceMeta.get_rfpath_sequence_meta_with_category_and_name(category=self.category, name=self.name)

    def save(self, path_meta):
        sequence_meta_fpath = self.get_fpath(path_meta=path_meta)
        sequence_meta_config = OmegaConf.structured(self)
        if not sequence_meta_fpath.parent.exists():
            sequence_meta_fpath.parent.mkdir(parents=True)
        OmegaConf.save(sequence_meta_config, sequence_meta_fpath, resolve=True)


class CO3D_Sequence():

    def __init__(self, path_raw: Path, path_preprocess: Path, path_meta: Path, meta: CO3D_SequenceMeta,
                 modalities: List[OD3D_FRAME_MODALITIES], categories: List[str],
                 cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.KPTS2D_ORIENT_AND_PCL.value,
                 cuboid_source=CUBOID_SOURCES.KPTS2D_ORIENT_AND_PCL.value
                 ):
        self.path_raw: Path = path_raw
        self.path_preprocess: Path = path_preprocess
        self.path_meta: Path = path_meta
        self.meta = meta
        self.cam_tform_obj_source = cam_tform_obj_source
        self.cuboid_source = cuboid_source
        self.modalities = modalities
        self.categories = categories
        self.category_id = categories.index(self.category)
        self._pcl = None
        self._pcl_clean = None
        self._front_name = None
        self._cuboid_front_tform4x4_obj = None
        self._cuboid = None


    @property
    def name(self):
        return self.meta.name

    @property
    def category(self):
        return self.meta.category

    def preprocess_front_name(self, override=False):
        fpath_front_name = self.fpath_front_name

        if override or not fpath_front_name.exists():
            from od3d.datasets.co3d.dataset import CO3D

            config = OmegaConf.create()
            config.sequences = [self.name]
            config.path = []
            config.path_meta = self.path_meta
            config.classes = [self.category]
            config.preprocess = False
            config.setup = False
            dataset = CO3D(config=config)

            if fpath_front_name.exists():
                self._front_name = od3d.io.read_str_from_file(fpath_front_name)
                front_item_id = dataset.get_item_id_by_name(self.name, self._front_name)
            else:
                front_item_id = 0

            k = ord('a')  # 2424832
            while (k == ord('a') or k == ord('d')):  # k == 2424832 or k == 2555904:
                show_img(dataset.get_item(front_item_id).rgb, duration=1)
                k = cv2.waitKey(0)
                if k == ord('a'):  # 2424832: # :
                    # left key:
                    front_item_id -= 1
                elif k == ord('d'):  # 2555904:
                    # right key:
                    front_item_id += 1
                front_item_id %= len(dataset)
            self._front_name = dataset.get_item(front_item_id).name
            if not fpath_front_name.parent.exists():
                fpath_front_name.parent.mkdir(parents=True)
            od3d.io.write_str_to_file(fpath_front_name, text=self._front_name)

    def align_cuboid_tform_obj(self, cuboid_tform_obj):
        if self.cuboid_source == CUBOID_SOURCES.FRONT_FRAME_AND_PCL:

            cam_front_rot3x3_cuboid = rot3x3(self.cam_front_tform4x4_obj[:3, :3], cuboid_tform_obj[:3, :3].T)
            cam_front_rot3x3_max_ids = cam_front_rot3x3_cuboid.abs().max(dim=-1)[1]
            assert (
                    0 in cam_front_rot3x3_max_ids.unique() and 1 in cam_front_rot3x3_max_ids.unique() and 2 in cam_front_rot3x3_max_ids.unique())
            cuboid_front_tform4x4_cuboid = torch.eye(n=4).to(device=cam_front_rot3x3_cuboid.device)
            cuboid_front_tform4x4_cuboid[:3, :3] = cuboid_front_tform4x4_cuboid[cam_front_rot3x3_max_ids,
                                                   :3] * cam_front_rot3x3_cuboid.sign()

            # x y z camera should be mapped to x z -y in object coordinate system
            cuboid_front_tform4x4_cuboid[:3] = cuboid_front_tform4x4_cuboid[[0, 2, 1]]
            cuboid_front_tform4x4_cuboid[2] = -cuboid_front_tform4x4_cuboid[2]

            cuboid_tform_obj = tform4x4(cuboid_front_tform4x4_cuboid, cuboid_tform_obj)

        elif self.cuboid_source == CUBOID_SOURCES.KPTS2D_ORIENT_AND_PCL:
            from od3d.cv.geometry.fit.axis3d_from_pxl2d import axis3d_from_pxl2d
            from od3d.cv.visual.show import show_img
            cam_rot3x3_cuboid_front = axis3d_from_pxl2d(kpts2d_orient=self.first_frame.kpts2d_orient, cam_intr4x4=self.first_frame.cam_intr4x4) #  orients
            # cam_rot3x3_cuboid = rot3x3(self.cam_first_tform4x4_obj[:3, :3], cuboid_tform_obj[:3, :3].T)
            #
            # cuboid_front_rot3x3_cuboid = rot3x3(cam_rot3x3_cuboid_front.T, cam_rot3x3_cuboid)
            # cuboid_front_rot3x3_cuboid_alignment =
            # cam_front_rot3x3_cuboid = rot3x3(self.cam_front_tform4x4_obj[:3, :3], cuboid_tform_obj[:3, :3].T)
            # cam_front_rot3x3_max_ids = cam_front_rot3x3_cuboid.abs().max(dim=-1)[1]
            # assert (
            #         0 in cam_front_rot3x3_max_ids.unique() and 1 in cam_front_rot3x3_max_ids.unique() and 2 in cam_front_rot3x3_max_ids.unique())
            # cuboid_front_tform4x4_cuboid = torch.eye(n=4).to(device=cam_front_rot3x3_cuboid.device)
            # cuboid_front_tform4x4_cuboid[:3, :3] = cuboid_front_tform4x4_cuboid[cam_front_rot3x3_max_ids,
            #                                        :3] * cam_front_rot3x3_cuboid.sign()
            #
            #
            #
            #cuboid_front_rot3x3_cuboid = rot3x3(rot3x3(cam_rot3x3_cuboid_front.T, self.cam_first_tform4x4_obj[:3, :3]), cuboid_tform_obj[:3, :3].T)
            #cuboid_front_tform4x4_cuboid = torch.eye(4)
            #cuboid_front_tform4x4_cuboid[:3, :3] = cuboid_front_rot3x3_cuboid
            #cuboid_tform_obj = tform4x4(cuboid_front_tform4x4_cuboid, cuboid_tform_obj)
            #
            #
            # cuboid_tform_obj[:3, :3] = rot3x3(cuboid_front_rot3x3_cuboid, cuboid_tform_obj[:3, :3])

            cuboid_tform_obj[:3, :3] = rot3x3(cam_rot3x3_cuboid_front.T, self.cam_first_tform4x4_obj[:3, :3])
            #cuboid_tform_obj[:3, 3] = rot3d(cam_rot3x3_cuboid_front.T, self.cam_first_tform4x4_obj[:3, :3])
        return cuboid_tform_obj

    def preprocess_cuboid(self, override=False):
        fpath_cuboid = self.fpath_cuboid
        if override or not fpath_cuboid.exists():
            fpath_cuboid.parent.mkdir(parents=True, exist_ok=True)

            cuboid_pts3d_max_count = 1000
            percentile_noise = 0.01

            pts3d_clean = self.pcl_clean

            pca_tform_obj = get_pca_tform_world(pts3d_clean)
            pca_pts3d_clean = transf3d_broadcast(pts3d_clean, pca_tform_obj)

            cuboids_limits = torch.stack([pca_pts3d_clean.min(dim=-2)[0], pca_pts3d_clean.max(dim=-2)[0]], dim=-2)[
                None,]

            cuboids = Cuboids.create_dense_from_limits(limits=cuboids_limits, verts_count=cuboid_pts3d_max_count)
            icp_tform_pca = inv_tform4x4(icp(cuboids.verts, pca_pts3d_clean))
            icp_tform_obj = tform4x4(icp_tform_pca, pca_tform_obj)

            # from od3d.cv.visual.show import show_pcl
            # obj_verts = transf3d_broadcast(pts3d=cuboids.verts, transf4x4=icp_tform_obj.inverse())
            # show_pcl([pts3d_clean, obj_verts]) #, cframe.pts3d_axis[0], cframe.pts3d_axis[1], cframe.pts3d_axis[2]])

            icp_tform_obj = self.align_cuboid_tform_obj(icp_tform_obj)

            # from od3d.cv.visual.show import show_pcl
            # obj_verts = transf3d_broadcast(pts3d=cuboids.verts, transf4x4=icp_tform_obj.inverse())
            # show_pcl([pts3d_clean, obj_verts]) #, cframe.pts3d_axis[0], cframe.pts3d_axis[1], cframe.pts3d_axis[2]])

            tmp_tform6_cuboid = torch.zeros(6).to(device=pts3d_clean.device)
            cuboid_tform4x4_obj = icp_tform_obj  # torch.eye(4).to(device=pts3d_clean.device)


            """
            cuboid_pts3d = transf3d_broadcast(pts3d=pts3d_clean, transf4x4=cuboid_tform4x4_obj)
            #cuboids_limits = torch.stack([cuboid_pts3d.min(dim=-2)[0], cuboid_pts3d.max(dim=-2)[0]], dim=-2)[None,]
            cuboids_limits = torch.stack(
                [cuboid_pts3d.quantile(dim=-2, q=percentile_noise), cuboid_pts3d.quantile(dim=-2, q=1. - percentile_noise)], dim=-2)[None,]
            cuboids = Cuboids.create_dense_from_limits(limits=cuboids_limits, verts_count=cuboid_pts3d_max_count)
            icp_tform_cuboid = icp(cuboids.verts, cuboid_pts3d).inverse()
            cuboid_tform4x4_obj = tform4x4(icp_tform_cuboid, cuboid_tform4x4_obj)
            """

            for i in range(100):
                if self.cuboid_source == CUBOID_SOURCES.KPTS2D_ORIENT_AND_PCL:
                   tmp_tform6_cuboid.data[3:] = 0.
                cuboid_tform4x4_obj = tform4x4(se3_exp_map(tmp_tform6_cuboid.detach()), cuboid_tform4x4_obj.detach())

                tmp_tform6_cuboid = torch.nn.Parameter(torch.zeros(6).to(device=pts3d_clean.device),
                                                       requires_grad=True)
                optimizer = torch.optim.SGD(params=[tmp_tform6_cuboid], lr=0.001)

                cuboid_tform4x4_obj = tform4x4(se3_exp_map(tmp_tform6_cuboid), cuboid_tform4x4_obj)

                cuboid_pts3d = transf3d_broadcast(pts3d=pts3d_clean, transf4x4=cuboid_tform4x4_obj)
                _, icp_pts3d_ids_min = cuboid_pts3d.min(dim=0)
                _, icp_pts3d_ids_max = cuboid_pts3d.max(dim=0)
                cuboid_pts3d_limits = cuboid_pts3d[torch.cat([icp_pts3d_ids_min, icp_pts3d_ids_max], dim=0)]
                #cuboid_pts3d_limits = torch.cat([cuboid_pts3d.quantile(dim=0, q=percentile_noise), cuboid_pts3d.quantile(dim=0, q=1. - percentile_noise)], dim=0)
                # icp_cuboids_vol = (cuboid_pts3d_limits[3, 0] - cuboid_pts3d_limits[0, 0]) * (cuboid_pts3d_limits[4, 1] - cuboid_pts3d_limits[1, 1]) * (cuboid_pts3d_limits[5, 2] - cuboid_pts3d_limits[2, 2])
                # using maximum ensures centering.
                icp_cuboids_vol = (max(abs(cuboid_pts3d_limits[3, 0]), abs(cuboid_pts3d_limits[0, 0])) * 2) * \
                                  (max(abs(cuboid_pts3d_limits[4, 1]), abs(cuboid_pts3d_limits[1, 1])) * 2) * \
                                  (max(abs(cuboid_pts3d_limits[5, 2]), abs(cuboid_pts3d_limits[2, 2])) * 2)

                loss = torch.norm(icp_cuboids_vol, p=2)
                # loss = cuboid_pts3d_limits.norm(dim=-1).prod() + (cuboid_pts3d_limits[0:3] - cuboid_pts3d_limits[3:6]).norm()
                loss.backward()
                logger.info(f'Volume {loss}')
                optimizer.step()

            cuboid_pts3d = transf3d_broadcast(pts3d=pts3d_clean, transf4x4=cuboid_tform4x4_obj)
            cuboids_limits = torch.stack(
                [cuboid_pts3d.quantile(dim=-2, q=percentile_noise), cuboid_pts3d.quantile(dim=-2, q=1. - percentile_noise)], dim=-2)[None,]
            cuboid_center_tform4x4_cuboid = torch.eye(4, device=cuboid_tform4x4_obj.device)
            cuboid_center_tform4x4_cuboid[:3, 3] = - (cuboids_limits[0, 1] + cuboids_limits[0, 0]) / 2.
            cuboid_tform4x4_obj = tform4x4(cuboid_center_tform4x4_cuboid, cuboid_tform4x4_obj)

            cuboid_front_tform_obj = cuboid_tform4x4_obj.detach()

            if not self.fpath_cuboid_front_tform4x4_obj.parent.exists():
                self.fpath_cuboid_front_tform4x4_obj.parent.mkdir(parents=True, exist_ok=True)
            torch.save(cuboid_front_tform_obj, f=str(self.fpath_cuboid_front_tform4x4_obj))

            # obj_tform_cuboid_front = cuboid_front_tform_obj.inverse()

            cuboid_pts3d = transf3d_broadcast(pts3d=pts3d_clean, transf4x4=cuboid_front_tform_obj)
            #cuboids_limits = torch.stack([cuboid_pts3d.min(dim=-2)[0], cuboid_pts3d.max(dim=-2)[0]], dim=-2)[None,]
            cuboids_limits = torch.stack(
                [cuboid_pts3d.quantile(dim=-2, q=percentile_noise), cuboid_pts3d.quantile(dim=-2, q=1. - percentile_noise)], dim=-2)[None,]

            cuboids = Cuboids.create_dense_from_limits(limits=cuboids_limits, verts_count=cuboid_pts3d_max_count)

            #from od3d.cv.visual.show import show_pcl
            #show_pcl([cuboid_pts3d, cuboids.verts]) #, cframe.pts3d_axis[0], cframe.pts3d_axis[1], cframe.pts3d_axis[2]])

            # from od3d.cv.geometry.primitives import CoordinateFrame
            # from od3d.cv.geometry.transform import transf4x4_from_spherical, transf4x4_from_pos_and_theta
            # cframe = CoordinateFrame(origin=obj_tform_cuboid_front[:3, 3], axes=obj_tform_cuboid_front[:3, :3])

            save_ply(fpath_cuboid, verts=cuboids.verts, faces=cuboids.faces)



    @property
    def pcl(self):
        if self._pcl is None:
            fpath_pcl = self.path_raw.joinpath(self.meta.rfpath_pcl)
            verts, _ = load_ply(str(fpath_pcl))
            self._pcl = verts
        return self._pcl

    @property
    def cuboid_labeled(self):
        return self.fpath_cuboid.exists()

    @property
    def cuboid_front_tform4x4_obj_labeled(self):
        return self.fpath_cuboid_front_tform4x4_obj.exists()

    @property
    def fpath_cuboid_front_tform4x4_obj(self):
        return self.path_preprocess.joinpath('cuboid_front_tform4x4_obj', self.cuboid_source, self.category, self.name, 'tform4x4.pt')

    # def get_cuboid_front_tform4x4_obj(self, cuboid_source: CUBOID_SOURCES):#
        #
        # if cuboid_source == CUBOID_SOURCES.FRONT_FRAME_AND_PCL:
        #     pass
        #
        # elif cuboid_source == CUBOID_SOURCES.KPTS2D_ORIENT_AND_PCL:
        #
        #     from od3d.cv.geometry.transform import reproj2d3d_broadcast, so3_log_map, so3_exp_map
        #     from od3d.cv.visual.draw import draw_pixels
        #     from od3d.cv.geometry.fit.axis3d_from_pxl2d import axis3d_from_pxl2d
        #     from od3d.cv.visual.show import show_img
        #     # show_img(draw_pixels(self.rgb, pxls=self.kpts2d_orient['back-front'][1][:]))
        #     self.cuboid_source
        #     self._cam_tform4x4_obj[:3, :3] = axis3d_from_pxl2d(kpts2d_orient=self.kpts2d_orient, cam_intr4x4=self.cam_intr4x4) #  orients
        #

    @property
    def front_name(self):
        if self._front_name is None:
            fpath_front_name = self.fpath_front_name
            if not fpath_front_name.exists():
                self.preprocess_front_name()
            self._front_name = od3d.io.read_str_from_file(fpath_front_name)
        return self._front_name

    @property
    def cam_front_tform4x4_obj(self):
        return torch.Tensor(self.front_frame.meta.l_cam_tform4x4_obj)

    @property
    def cam_first_tform4x4_obj(self):
        return torch.Tensor(self.first_frame.meta.l_cam_tform4x4_obj)

    def get_frame_by_name(self, frame_name: str):
        frame_meta = CO3D_FrameMeta.load_from_meta_with_rfpath(path_meta=self.path_meta,
                                                               rfpath=CO3D_FrameMeta.
                                                               get_rfpath_frame_meta_with_category_sequence_and_frame_name(
                                                                   category=self.category, sequence_name=self.name,
                                                                   name=frame_name))
        frame = CO3D_Frame(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
                           meta=frame_meta, modalities=self.modalities, categories=self.categories,
                           cam_tform_obj_source=self.cam_tform_obj_source, cuboid_source=self.cuboid_source)
        return frame

    @property
    def front_frame(self):
        return self.get_frame_by_name(frame_name=self.front_name)

    @property
    def first_frame(self):
        first_frame_fpath = sorted(CO3D_FrameMeta.get_path_frames_meta_with_category_sequence(path_meta=self.path_meta, category=self.category, sequence=self.name).iterdir(), key=lambda p: int(p.stem))[0]
        return self.get_frame_by_name(frame_name=first_frame_fpath.stem)

    @property
    def cuboid_front_tform4x4_obj(self):
        if self._cuboid_front_tform4x4_obj is None:
            if not self.fpath_cuboid_front_tform4x4_obj.exists():
                self.preprocess_cuboid()
            self._cuboid_front_tform4x4_obj = torch.load(f=str(self.fpath_cuboid_front_tform4x4_obj))
        return self._cuboid_front_tform4x4_obj

    def preprocess_pcl_clean(self, override=False):
        fpath_pcl_clean = self.fpath_pcl_clean
        if override or not fpath_pcl_clean.exists():
            from od3d.datasets.co3d.dataset import CO3D
            fpath_pcl_clean.parent.mkdir(parents=True, exist_ok=True)

            pts3d_max_count = 20000
            pts3d_prob_thresh = 0.6

            dataset = CO3D(name='co3d', modalities=[OD3D_FRAME_MODALITIES.RGB, OD3D_FRAME_MODALITIES.MASK],
                           path_raw=self.path_raw, path_preprocess=self.path_preprocess,
                           categories=[CO3D_CATEGORIES(self.category).value], sequences_names=[self.name], cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.CO3D.value)
            dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=10, shuffle=False,
                                                     collate_fn=dataset.collate_fn,
                                                     num_workers=0)

            pts3d = self.pcl

            pts3d = random_sampling(pts3d, pts3d_max_count=pts3d_max_count * 3)
            pts3d = voxel_downsampling(pts3d, K=pts3d_max_count)
            pts3d_prob = torch.ones(size=(pts3d.shape[0], 1), device=pts3d.device, dtype=pts3d.dtype)

            if torch.cuda.is_available():
                pts3d = pts3d.to(device='cuda:0')
                pts3d_prob = pts3d_prob.to(device='cuda:0')
            for frames in tqdm(iter(dataloader)):
                if torch.cuda.is_available():
                    frames.to(device='cuda:0')
                pxl2d = proj3d2d_broadcast(pts3d=pts3d, proj4x4=frames.cam_proj4x4_obj[:, None])
                pts3d_prob += sample_pxl2d_pts(frames.mask, pxl2d=pxl2d, padding_mode='zeros').sum(dim=0)

                # pxl2d = proj3d2d_broadcast(pts3d=pts3d_co3d, proj4x4=frame.cam_proj4x4_obj)
                # gb_with_mask_with_pts3d = draw_pixels(img=blend_rgb(frame.rgb, frame.mask*255.), pxls=pxl2d)
                # show_img(rgb_with_mask_with_pts3d)

            pts3d_prob = pts3d_prob / len(dataset)
            # pts3d_co3ds = []
            # for prob in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:  #0.5, 0.6, 0.7, 0.8, 0.9, 0.95
            #    pts3d = torch.zeros_like(pts3d_co3d)
            #    pts3d[pts3d_co3d_prob[..., 0] > prob] = pts3d[pts3d_prob[..., 0] > prob]
            #    pts3d_co3ds.append(pts3d)
            # show_pcl(torch.stack(pts3d_co3ds, dim=0))
            # show_pcl(pts3d[pts3d_prob[..., 0] > 0.6])
            pts3d_clean = pts3d[pts3d_prob[..., 0] > pts3d_prob_thresh]
            # frame0: CO3D_Frame = seq_dataset[0]
            # show_img(frame0.rgb)
            # show_pcl(pts3d_clean, cam_tform4x4_obj=frame0.cam_tform4x4_obj, cam_intr4x4=frame0.cam_intr4x4, img_size=frame0.size)

            save_ply(fpath_pcl_clean, pts3d_clean)

    @property
    def fpath_front_name(self):
        return self.path_preprocess.joinpath('front_names', self.category, self.name, 'front_name.yaml')
    @property
    def fpath_cuboid(self):
        return self.path_preprocess.joinpath('cuboids', self.cuboid_source, self.category, self.name + '.ply')
    @property
    def fpath_pcl_clean(self):
        return self.path_preprocess.joinpath('pcls', self.category, self.name, 'pcl_clean.ply') #  f'co3d_probthresh_{str(pts3d_prob_thresh).replace(".", "_")}_max_{pts3d_max_count}' + '.ply')
    @property
    def pcl_clean(self):
        if self._pcl_clean is None:
            fpath_pcl_clean = self.fpath_pcl_clean
            if not fpath_pcl_clean.exists():
                self.preprocess_pcl_clean()
            self._pcl_clean, _ = load_ply(fpath_pcl_clean)
        return self._pcl_clean

    @property
    def cuboid(self):
        if self._cuboid is None:
            fpath_cuboid = self.fpath_cuboid
            if not fpath_cuboid.exists():
                self.preprocess_cuboid()
            self._cuboid = Cuboids.load_from_files(fpaths_meshes=[fpath_cuboid])
        return self._cuboid

