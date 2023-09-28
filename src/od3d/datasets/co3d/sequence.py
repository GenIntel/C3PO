import open3d.geometry
from od3d.datasets.co3d.enum import CUBOID_SOURCES, CAM_TFORM_OBJ_SOURCES, CO3D_CATEGORIES
from od3d.datasets.co3d.frame import CO3D_Frame, CO3D_FrameMeta
from od3d.datasets.frame import OD3D_SequenceMeta
from od3d.cv.geometry.mesh import Mesh

from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
import torch.utils.data

import torch.utils.data
import od3d.io

import subprocess

from od3d.datasets.dataset import OD3D_Dataset, OD3D_FRAME_MODALITIES, OD3D_Frame

from co3d.dataset.data_types import (
    load_dataclass_jgzip, FrameAnnotation, SequenceAnnotation
)
from od3d.cv.geometry.transform import se3_exp_map, tform4x4, transf4x4_from_rot3x3_and_transl3
from od3d.cv.geometry.transform import rot3x3

from typing import List
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
class CO3D_SequenceMeta(OD3D_SequenceMeta):
    category: str
    pcl_pts_count: int
    pcl_quality_score: float
    rfpath_pcl: Path
    viewpoint_quality_score: float

    @property
    def name_unique(self):
        return f'{self.category}/{self.name}'

    @staticmethod
    def load_from_meta_with_category_and_name(path_meta: Path, category: str, name: str):
        name_unique = f'{category}/{name}'
        return CO3D_SequenceMeta.load_from_meta_with_name_unique(path_meta=path_meta, name_unique=name_unique)

    @staticmethod
    def get_name_unique_with_category_and_name(category: str, name: str):
        return f'{category}/{name}'

    @staticmethod
    def get_fpath_sequence_meta_with_category_and_name(path_meta: Path, category: str, name: str):
        return path_meta.joinpath(CO3D_SequenceMeta.get_rfpath_metas(), CO3D_SequenceMeta.get_name_unique_with_category_and_name(category=category, name=name))

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


    """
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
            map_category_sequences_names[category] = [fpath.stem for fpath in CO3D_SequenceMeta.get_path_sequences_meta_with_category(path_meta=path_meta, category=category).iterdir()]
        return map_category_sequences_names

    """
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
    """

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
        self._mesh = None
        self._front_name = None
        self._cuboid_front_tform4x4_obj = None
        self._cuboid = None

    @property
    def name(self):
        return self.meta.name

    @property
    def name_unique(self):
        return self.meta.name_unique

    @property
    def category(self):
        return self.meta.category

    @property
    def fpath_cuboid_limits3d(self):
        return self.path_preprocess.joinpath("labels", "cuboid_limits3d", f"{self.name_unique}.pt")

    def preprocess_cuboid_limits3d(self, override=False):
        import open3d as o3d
        import numpy as np

        if override or not self.fpath_cuboid_limits3d.exists():
            pcl = self.pcl_clean

            #if self.fpath_cuboid.exists() and self.fpath_cuboid_front_tform4x4_obj.exists():
            #    logger.info(f'press "s"  to skip this lable')
            #    k = self.cuboid.visualize(pcl=transf3d_broadcast(pts3d=pcl, transf4x4=self.cuboid_front_tform4x4_obj))
            #    if k == ord('s'):
            #        return

            # Create an Open3D PointCloud object
            #pcd = o3d.geometry.PointCloud()

            # Set the point cloud data
            #pcd.points = o3d.utility.Vector3dVector(pcl.numpy())

            logger.info("")
            logger.info(
                "1) Please pick left, right, back, front, top, bottom [shift + left click]"
            )
            logger.info("   Press [shift + right click] to undo point picking")
            logger.info("2) Afther picking points, press q for close the window")
            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window()
            #vis.add_geometry(pcd)
            pcd = o3d.io.read_point_cloud(str(self.fpath_pcl))

            # vis.add_geometry(pcd)

            if self.fpath_cuboid.exists() and self.fpath_cuboid_front_tform4x4_obj.exists():
                verts3d = self.cuboid.get_verts_with_mesh_id(mesh_id=0)
                ncds = self.cuboid.get_verts_ncds_with_mesh_id(mesh_id=0)
                cuboid_pcd = o3d.geometry.PointCloud()
                # from od3d.cv.geometry.transform import inv_tform4x4
                cuboid_pcd.points = o3d.utility.Vector3dVector(transf3d_broadcast(pts3d=verts3d, transf4x4=inv_tform4x4(self.cuboid_front_tform4x4_obj)).numpy())
                cuboid_pcd.colors = o3d.utility.Vector3dVector(ncds.numpy())
                vis.add_geometry(pcd + cuboid_pcd)

                #logger.info(f'press "s"  to skip this lable')
                #k = self.cuboid.visualize(pcl=transf3d_broadcast(pts3d=pcl, transf4x4=self.cuboid_front_tform4x4_obj))
                #if k == ord('s'):
                #    return
            else:
                vis.add_geometry(pcd)


            vis.run()  # user picks points
            vis.destroy_window()
            logger.info("")
            limits3d_ids = vis.get_picked_points()

            if len(limits3d_ids) < 6:
                logger.warning("No labels saved due to less than 6 points selected.")
                return

            if (torch.Tensor(limits3d_ids) > len(pcd.points)).any():
                logger.warning("No labels saved due to point on prev. cuboid selected")
                return


            limits3d = torch.from_numpy(np.asarray(pcd.points)).to(dtype=torch.float32)[limits3d_ids]
            self.fpath_cuboid_limits3d.parent.mkdir(parents=True, exist_ok=True)
            torch.save(limits3d, self.fpath_cuboid_limits3d)


    @property
    def cuboid_limits3d(self):
        if not self.fpath_cuboid_limits3d.exists():
            self.preprocess_cuboid_limits3d()
        cuboid_limits3d = torch.load(self.fpath_cuboid_limits3d).to(dtype=torch.float32)
        return cuboid_limits3d


    def preprocess_front_name(self, override=False):
        fpath_front_name = self.fpath_front_name

        if override or not fpath_front_name.exists():
            from od3d.datasets.co3d.dataset import CO3D

            dict_nested_frames = {self.category: {self.name: None}}

            dataset = CO3D(name='sequence', path_raw=self.path_raw, path_preprocess=self.path_preprocess,
                           modalities=self.modalities, dict_nested_frames=dict_nested_frames,
                           categories=[CO3D_CATEGORIES[self.category]])

            if fpath_front_name.exists():
                self._front_name = od3d.io.read_str_from_file(fpath_front_name)
                front_item_id = dataset.get_item_id_by_name_unique(CO3D_FrameMeta.get_name_unique_with_category_sequence_and_name(category=self.category, sequence_name=self.name, name=self._front_name))
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

        if self.cuboid_source == CUBOID_SOURCES.LIMITS3D:
            self.preprocess_cuboid_limits3d(override=override)

        if override or not fpath_cuboid.exists():
            fpath_cuboid.parent.mkdir(parents=True, exist_ok=True)

            cuboid_pts3d_max_count = 1000
            percentile_noise = 0.01

            pts3d_clean = self.pcl_clean

            if self.cuboid_source == CUBOID_SOURCES.LIMITS3D:
                if not self.fpath_cuboid_limits3d.exists():
                    logger.warning("No cuboid saved due to no limits3d available")
                    return
                import open3d as o3d
                from od3d.cv.visual.draw import get_colors
                from od3d.cv.geometry.transform import rot3d, rot3d_broadcast

                # Load the point cloud from a file (replace 'path_to_point_cloud.ply' with the actual path)
                #point_cloud = o3d.io.read_point_cloud(
                #    '/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/pcls/car/106_12650_23736/pcl_clean.ply')
                def visualize_points(l_pts3d: List):
                    colors = get_colors(K=len(l_pts3d))

                    logger.info("")
                    logger.info(
                        "1) Please pick left, right, back, front, top, bottom [shift + left click]"
                    )
                    logger.info("   Press [shift + right click] to undo point picking")
                    logger.info("2) Afther picking points, press q for close the window")
                    vis = o3d.visualization.Visualizer()
                    vis.create_window()

                    for i, pts3d in enumerate(l_pts3d):
                        # Create an Open3D PointCloud object
                        pcd = o3d.geometry.PointCloud()
                        # Set the point cloud data
                        pcd.points = o3d.utility.Vector3dVector(pts3d.numpy())
                        pcd.paint_uniform_color(colors[i].numpy())
                        vis.add_geometry(pcd)
                    vis.run()  # user picks points

                limits3d = self.cuboid_limits3d
                obj_axis3d = torch.nn.functional.normalize(limits3d[:6].reshape(3, 2, 3)[:, 0] - limits3d[:6].reshape(3, 2, 3)[:, 1], dim=-1)
                U, S, V = torch.linalg.svd(obj_axis3d)
                cuboid_rot3x3_obj = rot3x3(U, rot3x3(torch.diag(S.sign()), V))
                obj_center3d = limits3d[:6].mean(dim=0)
                cuboid_tform4x4_obj = transf4x4_from_rot3x3_and_transl3(rot3x3=cuboid_rot3x3_obj, transl3=-rot3d_broadcast(pts3d=obj_center3d, rot3x3=cuboid_rot3x3_obj))

                # visualize_points([transf3d_broadcast(pts3d=pts3d_clean, transf4x4=cuboid_tform4x4_obj), transf3d_broadcast(pts3d=limits3d, transf4x4=cuboid_tform4x4_obj)])

                tmp_tform6_cuboid = torch.zeros(6).to(device=pts3d_clean.device)

                for i in range(200):
                    tmp_tform6_cuboid.data[3:] = 0.

                    cuboid_tform4x4_obj = tform4x4(se3_exp_map(tmp_tform6_cuboid.detach()),
                                                   cuboid_tform4x4_obj.detach())

                    tmp_tform6_cuboid = torch.nn.Parameter(torch.zeros(6).to(device=pts3d_clean.device),
                                                           requires_grad=True)
                    optimizer = torch.optim.SGD(params=[tmp_tform6_cuboid], lr=0.001)

                    cuboid_tform4x4_obj = tform4x4(se3_exp_map(tmp_tform6_cuboid), cuboid_tform4x4_obj)

                    cuboid_pts3d = transf3d_broadcast(pts3d=limits3d, transf4x4=cuboid_tform4x4_obj)

                    _, cuboid_pts3d_ids_min = cuboid_pts3d.min(dim=0)
                    _, cuboid_pts3d_ids_max = cuboid_pts3d.max(dim=0)
                    cuboid_pts3d_limits = cuboid_pts3d[torch.cat([cuboid_pts3d_ids_min, cuboid_pts3d_ids_max], dim=0)]

                    # using maximum ensures centering.
                    cuboids_vol = (max(abs(cuboid_pts3d_limits[3, 0]), abs(cuboid_pts3d_limits[0, 0])) * 2) * \
                                      (max(abs(cuboid_pts3d_limits[4, 1]), abs(cuboid_pts3d_limits[1, 1])) * 2) * \
                                      (max(abs(cuboid_pts3d_limits[5, 2]), abs(cuboid_pts3d_limits[2, 2])) * 2)

                    loss = torch.norm(cuboids_vol, p=2)

                    loss.backward()
                    logger.info(f'Volume {loss}')
                    optimizer.step()

                cuboid_tform4x4_obj = cuboid_tform4x4_obj.detach()

                if not self.fpath_cuboid_front_tform4x4_obj.parent.exists():
                    self.fpath_cuboid_front_tform4x4_obj.parent.mkdir(parents=True, exist_ok=True)
                torch.save(cuboid_tform4x4_obj, f=str(self.fpath_cuboid_front_tform4x4_obj))

                cuboid_pts3d = transf3d_broadcast(pts3d=limits3d, transf4x4=cuboid_tform4x4_obj)
                _, cuboid_pts3d_ids_min = cuboid_pts3d.min(dim=0)
                _, cuboid_pts3d_ids_max = cuboid_pts3d.max(dim=0)
                cuboid_pts3d_limits = cuboid_pts3d[torch.cat([cuboid_pts3d_ids_min, cuboid_pts3d_ids_max], dim=0)]

                cuboid_pts3d_limits = torch.stack([cuboid_pts3d_limits[0, 0], cuboid_pts3d_limits[3, 0], cuboid_pts3d_limits[1, 1], cuboid_pts3d_limits[4, 1], cuboid_pts3d_limits[2, 2], cuboid_pts3d_limits[5, 2]]).reshape(3, 2).T.reshape(1, 2, 3)
                # cuboid_pts3d_limits = cuboid_pts3d_limits.flip(dims=[1,])

                logger.info(cuboid_pts3d_limits)

                cuboids = Cuboids.create_dense_from_limits(limits=cuboid_pts3d_limits, verts_count=cuboid_pts3d_max_count)


                # visualize_points([transf3d_broadcast(pts3d=pts3d_clean, transf4x4=cuboid_tform4x4_obj), transf3d_broadcast(pts3d=limits3d, transf4x4=cuboid_tform4x4_obj), cuboids.verts])

                save_ply(fpath_cuboid, verts=cuboids.verts, faces=cuboids.faces)


            else:
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
            fpath_pcl = self.fpath_pcl
            verts, _ = load_ply(str(fpath_pcl))
            self._pcl = verts
        return self._pcl

    @property
    def fpath_pcl(self):
        return self.path_raw.joinpath(self.meta.rfpath_pcl)

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


    def get_frame_by_index(self, index: int):
        return self.get_frame_by_name(self.frames_names[index])

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
    def frames_names(self):
        frames_names = CO3D_FrameMeta.get_frames_names_of_category_sequence(path_meta=self.path_meta, category=self.category, sequence_name=self.name)
        return frames_names

    @property
    def first_frame(self):
        #first_frame_fpath = sorted(CO3D_FrameMeta.get_path_frames_meta_with_category_sequence(path_meta=self.path_meta,
        #                                                                                      category=self.category,
        #                                                                                      sequence=self.name).iterdir(),
        #                           key=lambda p: int(p.stem))[0]
        return self.get_frame_by_index(index=0)

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

            dataset = CO3D(name='co3d', modalities=[OD3D_FRAME_MODALITIES.RGB, OD3D_FRAME_MODALITIES.MASK, OD3D_FRAME_MODALITIES.CAM_TFORM4X4_OBJ, OD3D_FRAME_MODALITIES.CAM_INTR4X4],
                           path_raw=self.path_raw, path_preprocess=self.path_preprocess,
                           categories=[CO3D_CATEGORIES(self.category).value], dict_nested_frames={self.category: {self.name: None}},
                           cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.CO3D.value)

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

    def preprocess_mesh(self):
        # N x 3
        pts3d, _ = load_ply(self.path_preprocess.joinpath('droid_slam', self.category, self.name, 'pcl.ply'))
        # F x 4 x 4
        cams_tform4x4_obj = torch.load(self.path_preprocess.joinpath('droid_slam', self.category, self.name, 'traj_est.pt'))
        import numpy as np

        #a = torch.from_numpy(np.load(str(self.path_preprocess.joinpath('droid_slam', self.category, self.name, 'poses.npy'))))
        cam_intrinsics_pts3d = torch.from_numpy(np.load(str(self.path_preprocess.joinpath('droid_slam', self.category, self.name, 'intrinsics.npy'))))
        scale = self.first_frame.cam_intr4x4[0, 0] / cam_intrinsics_pts3d[0, 0]
        #pts3d *= scale

        geometries = []

        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'

        pts3d = pts3d.to(device=device)
        cams_tform4x4_obj = cams_tform4x4_obj.to(device=device)

        scene_particles = 2000
        particle_quantile_dist = 0.05

        pts3d = random_sampling(pts3d_cls=pts3d, pts3d_max_count=scene_particles)
        #pts3d = voxel_downsampling(pts3d_cls=pts3d, K=scene_particles)

        #pts3d = pts3d[torch.randperm(pts3d.shape[0])[:scene_particles]]
        pts3d_range = max(pts3d.max(dim=0)[0] - pts3d.min(dim=0)[0]).item()
        scene_size = pts3d_range
        particle_size = torch.cdist(pts3d, pts3d).quantile(q=particle_quantile_dist)


        obj_cams_rays6d = torch.cat([inv_tform4x4(cams_tform4x4_obj)[:, :3, 3], cams_tform4x4_obj[:, 2, :3]], dim=-1).to(device=device)
        obj_cams_rays_start = obj_cams_rays6d[:, :3]
        obj_cams_rays_dir = obj_cams_rays6d[:, 3:]
        obj_cams_rays_end = obj_cams_rays_start + obj_cams_rays_dir * scene_size

        # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
        S = (torch.eye(3).to(device=device)[None, ] - obj_cams_rays_dir[:, None, :] * obj_cams_rays_dir[:, :, None]).sum(dim=0)
        C = torch.einsum('BXY,BY->BX', (torch.eye(3).to(device=device)[None, ] - (obj_cams_rays_dir[:, None, :] * obj_cams_rays_dir[:, :, None])), obj_cams_rays_start).sum(dim=0)
        center3d = torch.linalg.solve(A=S, B=C)

        center3d_tform4x4_obj = torch.eye(4).to(device=device)
        center3d_tform4x4_obj[:3, 3] = -center3d

        #cams_tform4x4_obj[:, :3, 3] = cams_tform4x4_obj[:, :3, 3] - center3d[None,]
        #center3d[:] = 0.
        from od3d.cv.geometry.transform import tform4x4_broadcast
        cams_tform4x4_obj = tform4x4_broadcast(cams_tform4x4_obj, inv_tform4x4(center3d_tform4x4_obj))
        pts3d = transf3d_broadcast(pts3d, transf4x4=center3d_tform4x4_obj)
        center3d = transf3d_broadcast(center3d, transf4x4=center3d_tform4x4_obj)


        cam_intr4x4 = self.first_frame.cam_intr4x4

        #pts3d_3 = pts3d[:3]
        #pts3d_3 = [cams_tform4x4_obj[0, :3, 3], cams_tform4x4_obj[50, :3, 3], cams_tform4x4_obj[100, :3, 3]]
        #axis1 = pts3d_3[1] - pts3d_3[0]
        #axis2 = pts3d_3[2] - pts3d_3[0]
        #plane3d = torch.cross(axis1, axis2)
        #plane3d = plane3d / (plane3d.norm()) * 5.  # + 5.

        N = pts3d.shape[0]
        # 1. proposals


        from od3d.cv.optimization.ransac import ransac
        from od3d.cv.select import batched_index_select
        def fit_plane(pts: torch.Tensor, pts_ids: torch.Tensor):
            """
            Args:
                pts (torch.Tensor): ...xNxF
                pts_ids (torch.Tensor): ...xPxS
            Returns:
                planes (torch.Tensor): ...xPxM
            """

            # ...xPxSxF
            pts_sampled = batched_index_select(index=pts_ids.flatten(-2), input=pts).view(pts_ids.shape + (-1,))

            proposed_shape = pts_sampled.shape[:-2]
            proposed_planes_axis_1 = pts_sampled[..., 1, :] - pts_sampled[..., 0, :]
            proposed_planes_axis_2 = pts_sampled[..., 2, :] - pts_sampled[..., 0, :]
            proposed_planes_axis_z = torch.cross(proposed_planes_axis_1, proposed_planes_axis_2, dim=-1)
            proposed_planes_axis_z = proposed_planes_axis_z / proposed_planes_axis_z.norm(dim=-1, keepdim=True)
            proposed_planes_signed_dist = torch.einsum('pf,pf->p', pts_sampled[..., 0, :].view(-1, 3), proposed_planes_axis_z.view(-1, 3)).view(proposed_shape + (1,))
            proposed_planes4d = torch.cat([proposed_planes_axis_z, proposed_planes_signed_dist], dim=-1)
            return proposed_planes4d

        def score_plane4d_fit(pts: torch.Tensor, plane4d: torch.Tensor, plane_dist_thresh: float, pts_on_plane_weight: float=1.3):
            """
            Args:
                pts (torch.Tensor): ...xNxF
                planes (torch.Tensor): ...xPxM
            Returns:
                scores (torch.Tensor): ...xP
            """
            N, F = pts.shape[-2:]
            batch_shape = pts.shape[:-2]
            batch_count = batch_shape.numel()

            proposed_planes_axis_z = plane4d[..., :3]
            proposed_planes_signed_dist = plane4d[..., 3:]
            signed_dists_pts3d_to_proposed_planes = torch.einsum('bnc,bpc->bpn', pts.view(batch_count, -1, F), proposed_planes_axis_z.view(batch_count, -1, F)) - proposed_planes_signed_dist.view(batch_count, -1, 1)
            signed_dists_pos_perc = (signed_dists_pts3d_to_proposed_planes > plane_dist_thresh).sum(dim=-1) / N
            signed_dists_thresh_perc = (signed_dists_pts3d_to_proposed_planes.abs() < plane_dist_thresh).sum(dim=-1) / N
            scores = signed_dists_pos_perc + signed_dists_thresh_perc * pts_on_plane_weight
            scores = scores.view(batch_shape + (-1, ))
            return scores

        mask_plane_thresh = particle_size / 5.
        from functools import partial
        plane4d = ransac(pts=pts3d, fit_func=fit_plane, score_func=partial(score_plane4d_fit, plane_dist_thresh=mask_plane_thresh), fits_count=1000, fit_pts_count=3)


        plane3d_tform4x4_obj = torch.eye(4).to(device=device)
        top_axis = plane4d[:3] / plane4d[:3].norm()
        x = top_axis[0]
        y = top_axis[1]
        z = top_axis[2]
        if x != 0 or y !=0:
            left_axis = torch.Tensor([-y, x, 0.]).to(device=device)
            left_axis = left_axis / left_axis.norm()
            back_axis = torch.Tensor([-x * z, -y * z, x * x + y * y]).to(device=device)
            back_axis = back_axis / back_axis.norm()
        else:
            left_axis = torch.Tensor([1., 0., 0.], device=device)
            back_axis = torch.Tensor([0., 1., 0.], device=device)
        plane3d_tform4x4_obj[0, :3] = left_axis
        plane3d_tform4x4_obj[1, :3] = back_axis
        plane3d_tform4x4_obj[2, :3] = top_axis
        plane3d_tform4x4_obj[2, 3] = -plane4d[3]
        obj_tform4x4_plane = inv_tform4x4(plane3d_tform4x4_obj)

        plane_z = -plane3d_tform4x4_obj[2, 3]
        plane3d_tform4x4_obj[:3, 3] = 0.
        cams_tform4x4_obj = tform4x4_broadcast(cams_tform4x4_obj, inv_tform4x4(plane3d_tform4x4_obj))
        pts3d = transf3d_broadcast(pts3d, transf4x4=plane3d_tform4x4_obj)
        center3d = transf3d_broadcast(center3d, transf4x4=plane3d_tform4x4_obj)

        height = scene_size / 5.
        radius = scene_size * 0.5
        # plane3d_open3d = open3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth).translate((-width / 2., -height / 2., -depth / 2.))
        R = open3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz((np.pi, 0., 0.))

        # note: height becomes larger with lower resolution
        plane3d_open3d = open3d.geometry.TriangleMesh.create_cone(radius=radius, height=height, resolution=100, split=1, create_uv_map=False).rotate(R=R, center=(0, 0, 0)) # .translate((0., 0., -depth / 2.))
        plane3d_open3d.translate((0., 0., plane_z.item()))
        plane3d_open3d.paint_uniform_color([0.2, 0.2, 0.4])

        mat_box = open3d.visualization.rendering.MaterialRecord()
        mat_box.shader = 'defaultLitTransparency'
        #mat_box.shader = 'defaultLitSSR'
        mat_box.base_color = [0.467, 0.467, 0.467, 0.02]
        mat_box.base_roughness = 0.0
        mat_box.base_reflectance = 0.0
        mat_box.base_clearcoat = 1.0
        mat_box.thickness = 1.0
        mat_box.transmission = 1.0
        mat_box.absorption_distance = 10
        mat_box.absorption_color = [0.5, 0.5, 0.5]

        #mat_box.base_color = np.array([1, 1, 1, .5])
        # open3d.visualization.draw_geometries(geoms_to_draw)

        center3d_open3d = open3d.geometry.TriangleMesh.create_sphere(radius=scene_size / 50.).translate(center3d.detach().cpu().numpy())
        geometries.append({'name': f'center', 'geometry': center3d_open3d})

        geometries.append({'name': 'plane3d', 'geometry': plane3d_open3d, 'material': mat_box})

        #plane_tform4x4_pts3d = transf3d_broadcast(pts3d=pts3d, transf4x4=plane3d_tform4x4_obj)
        mask_pts3d_on_plane = pts3d[:, 2] - plane_z < + mask_plane_thresh
        # starting with 10 percentage of points
        mask_center_thresh = (pts3d[~mask_pts3d_on_plane] - center3d).norm(dim=-1).quantile(0.1)
        mask_pts3d_on_center = (pts3d[~mask_pts3d_on_plane] - center3d).norm(dim=-1) < mask_center_thresh
        pts3d_not_on_plane = pts3d[~mask_pts3d_on_plane]

        mask_pts3d_obj = mask_pts3d_on_center.clone()
        dists_pts3d_not_on_plane_plane = pts3d[~mask_pts3d_on_plane, 2] - plane_z
        dists_pts3d_not_on_plane_obj_dists = (pts3d_not_on_plane[mask_pts3d_obj][:, None] - pts3d_not_on_plane[None, :]).norm(dim=-1).min(dim=0)[0]

        while ((dists_pts3d_not_on_plane_obj_dists < particle_size) * (dists_pts3d_not_on_plane_obj_dists < dists_pts3d_not_on_plane_plane)).sum() > mask_pts3d_obj.sum():
            logger.info(mask_pts3d_obj.sum())
            mask_pts3d_obj += (dists_pts3d_not_on_plane_obj_dists < particle_size) * (dists_pts3d_not_on_plane_obj_dists < dists_pts3d_not_on_plane_plane)
            #dists_pts3d_not_on_plane_obj_dists = \
            #(pts3d_not_on_plane[mask_pts3d_obj][:, None] - pts3d_not_on_plane[None, :]).norm(dim=-1).min(dim=0)[0]
            dists_pts3d_not_on_plane_obj_dists = torch.cdist(pts3d_not_on_plane[mask_pts3d_obj], pts3d_not_on_plane).min(dim=0)[0]

        pts3d_obj = pts3d_not_on_plane[mask_pts3d_obj]

        o3d_pcl = open3d.geometry.PointCloud()
        # from od3d.cv.geometry.transform import inv_tform4x4
        #open3d.visualization.draw_geometries([o3d_pcl], point_show_normal=True)
        o3d_pcl.points = open3d.utility.Vector3dVector(pts3d_obj[:].detach().cpu().numpy())
        o3d_pcl.normals = open3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
        #o3d_pcl.estimate_normals()
        geometries.append({'name': 'pcl', 'geometry': o3d_pcl})

        alpha = mask_center_thresh
        #mesh, densities = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(o3d_pcl, depth=9)
        mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(o3d_pcl, alpha)
        mesh.compute_vertex_normals()
        geometries.append({'name': 'mesh', 'geometry': mesh})

        for i, cam_tform4x4_obj in enumerate(cams_tform4x4_obj):
            width = int(cam_intr4x4[0, 2] * 2)
            height = int(cam_intr4x4[1, 2] * 2)

            cam = open3d.geometry.LineSet.create_camera_visualization(view_width_px=width, view_height_px=height,
                                                                      intrinsic=cam_intr4x4[:3, :3].detach().numpy(),
                                                                      extrinsic=cam_tform4x4_obj.detach().cpu().numpy(),
                                                                      scale=0.01)
            geometries.append({'name': f'cam{i}', 'geometry': cam})

            ray_range = scene_size
            ray = open3d.geometry.TriangleMesh.create_arrow(cylinder_radius=1.0*particle_size, cone_radius=1.5*particle_size, cylinder_height=ray_range, cone_height=4.0*particle_size)
            ray.transform(inv_tform4x4(cam_tform4x4_obj).detach().cpu().numpy())

            #cam_start_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=particle_size)
            #cam_start_sphere.translate(obj_cams_rays_start[i].detach().cpu().numpy())
            #cam_end_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=particle_size)
            #cam_end_sphere.translate(obj_cams_rays_end[i].detach().cpu().numpy())
            #geometries.append({'name': f'ray{i}_start', 'geometry': cam_start_sphere})
            #geometries.append({'name': f'ray{i}_end', 'geometry': cam_end_sphere})

        geometries.append({'name': f'obj', 'geometry': o3d_pcl})


        o3d_pcl_on_plane = open3d.geometry.PointCloud()
        o3d_pcl_on_plane.points = open3d.utility.Vector3dVector(pts3d[mask_pts3d_on_plane].detach().cpu().numpy())
        o3d_pcl_on_plane.paint_uniform_color((0.1, 0.1, 0.5))
        geometries.append({'name': 'pcl_on_plane', 'geometry': o3d_pcl_on_plane})

        o3d_pcl_on_center = open3d.geometry.PointCloud()
        o3d_pcl_on_center.points = open3d.utility.Vector3dVector(pts3d[~mask_pts3d_on_plane][mask_pts3d_on_center].detach().cpu().numpy())
        o3d_pcl_on_center.paint_uniform_color((0.1, 0.5, 0.1))
        geometries.append({'name': 'pcl_on_center', 'geometry': o3d_pcl_on_center})

        o3d_pcl_noise = open3d.geometry.PointCloud()
        o3d_pcl_noise.points = open3d.utility.Vector3dVector(pts3d[~mask_pts3d_on_plane][~mask_pts3d_obj].detach().cpu().numpy())
        o3d_pcl_noise.paint_uniform_color((0.5, 0.1, 0.1))
        geometries.append({'name': 'pcl_noise', 'geometry': o3d_pcl_noise})

        open3d.visualization.draw(geometries)

        save_ply(f=self.fpath_mesh, verts=torch.from_numpy(np.asarray(mesh.vertices)), faces=torch.LongTensor(np.asarray(mesh.triangles)))

        for i, cam_tform4x4_obj in enumerate(cams_tform4x4_obj):
            frame = self.get_frame_by_index(i)
            frame.fpath_cam_tform4x4_obj_droid_slam.parent.mkdir(parents=True, exist_ok=True)
            torch.save(obj=cam_tform4x4_obj.detach().cpu(), f=frame.fpath_cam_tform4x4_obj_droid_slam)

    @property
    def fpath_mesh(self):
        return self.path_preprocess.joinpath('droid_slam', self.category, self.name, 'mesh.ply')

    @property
    def mesh(self):
        if self._mesh is None:
            fpath_mesh = self.fpath_mesh
            if not fpath_mesh.exists():
                self.preprocess_mesh()
            self._mesh = Mesh.load_from_file(fpath=self.fpath_mesh)
        return self._mesh

    @mesh.setter
    def mesh(self, value: torch.Tensor):
            self._mesh = value

    @property
    def com(self):
        return self.pcl_clean.mean(dim=0)

    @property
    def cuboid(self):
        if self._cuboid is None:
            fpath_cuboid = self.fpath_cuboid
            if not fpath_cuboid.exists():
                self.preprocess_cuboid()
            self._cuboid = Cuboids.load_from_files(fpaths_meshes=[fpath_cuboid])
        return self._cuboid

