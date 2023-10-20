import open3d.geometry
from od3d.datasets.co3d.enum import CUBOID_SOURCES, CAM_TFORM_OBJ_SOURCES, CO3D_CATEGORIES, FEATURE_TYPES, REDUCE_TYPES, PCL_SOURCES
from od3d.datasets.enum import OD3D_CATEGORIES_SIZES_IN_M
from od3d.datasets.co3d.enum import MAP_CATEGORIES_CO3D_TO_OD3D
from od3d.datasets.co3d.frame import CO3D_Frame, CO3D_FrameMeta
from od3d.datasets.frame import OD3D_SequenceMeta
from od3d.cv.geometry.mesh import Mesh, Meshes
from od3d.cv.io import read_pts3d_colors, read_pts3d, write_pts3d_with_colors
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
from od3d.cv.geometry.fit.cuboid import fit_cuboid_to_pts3d

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



class CO3D_Sequence():

    def __init__(self, path_raw: Path, path_preprocess: Path, path_meta: Path, meta: CO3D_SequenceMeta,
                 modalities: List[OD3D_FRAME_MODALITIES], categories: List[str],
                 cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.KPTS2D_ORIENT_AND_PCL.value,
                 cuboid_source=CUBOID_SOURCES.KPTS2D_ORIENT_AND_PCL.value,
                 mesh_feats_type=FEATURE_TYPES.DINOV2_AVG.value,
                 dist_verts_mesh_feats_reduce_type=REDUCE_TYPES.MIN.value,
                 pcl_source=PCL_SOURCES.CO3D.value,
                 aligned_name:str =None,
                 mesh_name: str='default',
                 ):
        self.path_raw: Path = path_raw
        self.path_preprocess: Path = path_preprocess
        self.path_meta: Path = path_meta
        self.meta = meta
        self.cam_tform_obj_source = cam_tform_obj_source
        self.aligned_name = aligned_name
        self.mesh_name = mesh_name
        self.mesh_feats_type = mesh_feats_type
        self.pcl_source = pcl_source
        self.dist_verts_mesh_feats_reduce_type = dist_verts_mesh_feats_reduce_type
        self.cuboid_source = cuboid_source
        self.modalities = modalities
        self.categories = categories
        self.category_id = categories.index(self.category)
        self._mesh_feats = None
        self._pcl = None
        self._pcl_colors = None
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
            pcd = o3d.io.read_point_cloud(str(self.fpath_pcl_co3d))

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

    """
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
    """


    @property
    def pcl(self):
        if self._pcl is None:
            fpath_pcl = self.fpath_pcl_co3d
            verts, _ = load_ply(str(fpath_pcl))
            self._pcl = verts

        return self._pcl

    def get_pcl(self, pcl_source: PCL_SOURCES):
        if pcl_source == PCL_SOURCES.CO3D:
            fpath = self.fpath_pcl_co3d
            return read_pts3d(fpath)
        elif pcl_source == PCL_SOURCES.DROID_SLAM:
            fpath = self.fpath_pcl_droid_slam
            if not fpath.exists():
                self.preprocess_mesh(override=True)
            return read_pts3d(fpath)
        elif pcl_source == PCL_SOURCES.DROID_SLAM_CLEAN:
            fpath = self.fpath_pcl_droid_slam_clean
            if not fpath.exists():
                self.preprocess_mesh(override=True)
            return read_pts3d(fpath)
        else:
            logger.warning(f'Unknown pcl source {pcl_source}')
            return None

    def get_pcl_colors(self, pcl_source: PCL_SOURCES):
        if pcl_source == PCL_SOURCES.CO3D:
            fpath = self.fpath_pcl_co3d
            return read_pts3d_colors(fpath)
        elif pcl_source == PCL_SOURCES.DROID_SLAM:
            fpath = self.fpath_pcl_droid_slam
            return read_pts3d_colors(fpath)
        elif pcl_source == PCL_SOURCES.DROID_SLAM_CLEAN:
            fpath = self.fpath_pcl_droid_slam_clean
            return read_pts3d_colors(fpath)
        else:
            logger.warning(f'Unknown pcl source {pcl_source}')
            return None


    @property
    def pcl_colors_co3d(self):
        if self._pcl_colors is None:
            self._pcl_colors = read_pts3d_colors(self.fpath_pcl_co3d)
        return self._pcl_colors

    @property
    def fpath_pcl_co3d(self):
        return self.path_raw.joinpath(self.meta.rfpath_pcl)

    @property
    def fpath_pcl_droid_slam(self):
        return self.path_preprocess.joinpath('pcl', 'droid_slam', self.category, self.name, 'pcl.ply')

    @property
    def fpath_pcl_droid_slam_clean(self):
        return self.path_preprocess.joinpath('pcl', 'droid_slam_clean', self.category, self.name, 'pcl.ply')

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

    """
    @property
    def front_name(self):
        if self._front_name is None:
            fpath_front_name = self.fpath_front_name
            if not fpath_front_name.exists():
                self.preprocess_front_name()
            self._front_name = od3d.io.read_str_from_file(fpath_front_name)
        return self._front_name
    """

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
                           cam_tform_obj_source=self.cam_tform_obj_source, cuboid_source=self.cuboid_source,
                           aligned_name=self.aligned_name, mesh_name=self.mesh_name)
        return frame

    def get_sequence_by_category_and_name(self, category: str, name: str):
        sequence_meta = CO3D_SequenceMeta.load_from_meta_with_category_and_name(path_meta=self.path_meta,
                                                                                category=category, name=name)
        return CO3D_Sequence(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
                             meta=sequence_meta, modalities=self.modalities, categories=self.categories,
                             aligned_name=self.aligned_name,
                             mesh_feats_type=self.mesh_feats_type,
                             dist_verts_mesh_feats_reduce_type=self.dist_verts_mesh_feats_reduce_type,
                             cuboid_source=self.cuboid_source,
                             cam_tform_obj_source=self.cam_tform_obj_source, pcl_source=self.pcl_source,
                             mesh_name=self.mesh_name)

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
                           cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.CO3D.value,
                           mesh_name=self.mesh_name)

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
    def fpath_mesh_feats(self):
        return self.path_preprocess.joinpath('mesh_feats', self.mesh_name, self.mesh_feats_type, self.name_unique, 'mesh_feats.pt')

    def preprocess_mesh_feats(self):
        # fpath_mesh_feats
        from od3d.datasets.co3d import CO3D
        dataset = CO3D(name='co3d', modalities=[OD3D_FRAME_MODALITIES.RGB, OD3D_FRAME_MODALITIES.CAM_TFORM4X4_OBJ,
                                                OD3D_FRAME_MODALITIES.CAM_INTR4X4],
                       path_raw=self.path_raw, path_preprocess=self.path_preprocess,
                       categories=[CO3D_CATEGORIES(self.category).value],
                       dict_nested_frames={self.category: {self.name: None}},
                       cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.DROID_SLAM.value,
                       cuboid_source=CUBOID_SOURCES.DROID_SLAM.value,
                       mesh_feats_type=self.mesh_feats_type,
                       mesh_name=self.mesh_name)

        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=10, shuffle=False,
                                                 collate_fn=dataset.collate_fn,
                                                 num_workers=0)

        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'

        from od3d.models.model import OD3D_Model
        from od3d.cv.transforms.transform import OD3D_Transform
        from od3d.cv.transforms.sequential import SequentialTransform


        model = OD3D_Model.create_by_name('dinov2_frozen_base')
        model.cuda()
        model.eval()
        transform = SequentialTransform([OD3D_Transform.create_by_name('centerzoom512'), model.transform])

        down_sample_rate = 16
        feature_dim = 384
        meshes = Meshes.load_from_meshes([self.mesh], device=device)
        dataset.transform = transform

        meshes_verts_aggregated_features = [torch.zeros((0, feature_dim), device=device)] * meshes.verts.shape[0]
        vertices_count = len(meshes_verts_aggregated_features)


        for batch in tqdm(iter(dataloader)):
            B = len(batch)
            batch.to(device=device)

            batch.cam_tform4x4_obj = batch.cam_tform4x4_obj.detach()

            vts2d, vts2d_mask = meshes.verts2d(cams_intr4x4=batch.cam_intr4x4,
                                               cams_tform4x4_obj=batch.cam_tform4x4_obj,
                                               imgs_sizes=batch.size, mesh_ids=[0,] * B,
                                               down_sample_rate=down_sample_rate)

            N = vts2d.shape[1]

            # B x C x H x W
            feats2d_net = model(batch.rgb)
            H, W = feats2d_net.shape[-2:]
            xy = torch.stack(
                torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device),
                               indexing='xy'), dim=0)  # HxW
            noise2d = torch.ones(size=(vts2d.shape[0], 0, 2), device=device)

            # B x F+N x C
            net_feats = sample_pxl2d_pts(feats2d_net, pxl2d=torch.cat([vts2d, noise2d], dim=1))

            C = net_feats.shape[2]
            # args: X: Bx3xHxW, keypoint_positions: BxNx2, obj_mask: BxHxW ensures that noise is sampled outside of object mask
            # returns: BxF+NxC

            # net_feats = net_feats[:, :].reshape(-1, net_feats.shape[-1])
            batch_vts_ids = meshes.get_verts_and_noise_ids_stacked([0,] * B, count_noise_ids=0)

            # N,
            batch_vts_ids = torch.cat([batch_vts_ids[:, :N][vts2d_mask], batch_vts_ids[:, N:].reshape(-1)],
                                      dim=0)

            # N x C
            net_feats = torch.cat([net_feats[:, :N][vts2d_mask], net_feats[:, N:].reshape(-1, C)], dim=0)

            for b, vertex_id in enumerate(batch_vts_ids):
                meshes_verts_aggregated_features[vertex_id] = torch.cat([net_feats[b:b + 1], meshes_verts_aggregated_features[vertex_id]], dim=0)


        if self.mesh_feats_type == FEATURE_TYPES.DINOV2_ACC:
            if not self.fpath_mesh_feats.parent.exists():
                self.fpath_mesh_feats.parent.mkdir(parents=True, exist_ok=True)
            torch.save(meshes_verts_aggregated_features, f=self.fpath_mesh_feats)
        elif self.mesh_feats_type == FEATURE_TYPES.DINOV2_AVG:
            if not self.fpath_mesh_feats.parent.exists():
                self.fpath_mesh_feats.parent.mkdir(parents=True, exist_ok=True)
            meshes_verts_aggregated_features_avg = torch.stack([agg_feats.mean(dim=0) for agg_feats in meshes_verts_aggregated_features], dim=0)
            torch.save(meshes_verts_aggregated_features_avg, f=self.fpath_mesh_feats)
        elif self.mesh_feats_type == FEATURE_TYPES.DINOV2_AVG_NORM:
            if not self.fpath_mesh_feats.parent.exists():
                self.fpath_mesh_feats.parent.mkdir(parents=True, exist_ok=True)
            meshes_verts_aggregated_features_avg_norm = torch.nn.functional.normalize(torch.stack([agg_feats.mean(dim=0) for agg_feats in meshes_verts_aggregated_features], dim=0), dim=-1)
            torch.save(meshes_verts_aggregated_features_avg_norm, f=self.fpath_mesh_feats)
        else:
            logger.warning(f'Unknown mesh feature type {self.mesh_feats_type}.')

    def get_dist_verts_mesh_feats_to_other_sequence(self, sequence: 'CO3D_Sequence'):
        fpath_dist_verts_mesh_feats = self.path_preprocess.joinpath('dist_verts_mesh_feats', self.mesh_name, self.mesh_feats_type, self.dist_verts_mesh_feats_reduce_type, self.name_unique, sequence.name_unique, 'dist_verts_mesh_feats.pt')
        if fpath_dist_verts_mesh_feats.exists():
            return torch.load(fpath_dist_verts_mesh_feats)
        else:
            device = self.feats[0].device
            seq1_feats = self.feats
            seq2_feats = sequence.feats


            if isinstance(seq1_feats, list):
                seq1_verts = len(seq1_feats)
                seq2_verts = len(seq2_feats)
                dist_verts_seq1_seq2 = torch.zeros((seq1_verts, seq2_verts)).to(device=device)

                for i in tqdm(range(seq1_verts)):
                    for j in range(seq2_verts):
                        #if i == j:
                        #    dist_verts_seq1_seq2[i, j] = torch.inf
                        #else:
                        dists = torch.cdist(self.feats[i], sequence.feats[j])
                        if dists.numel() == 0:
                            dist_verts_seq1_seq2[i, j] = torch.inf
                        else:
                            if self.dist_verts_mesh_feats_reduce_type == REDUCE_TYPES.MIN:
                                dist_verts_seq1_seq2[i, j] = dists.min()
                            elif self.dist_verts_mesh_feats_reduce_type == REDUCE_TYPES.AVG:
                                dist_verts_seq1_seq2[i, j] = dists.mean()
                            else:
                                logger.warning(f'Unknown reduce type {self.dist_verts_mesh_feats_reduce_type}.')

            else:
                dist_verts_seq1_seq2 = torch.cdist(seq1_feats, seq2_feats)
            if not fpath_dist_verts_mesh_feats.parent.exists():
                fpath_dist_verts_mesh_feats.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dist_verts_seq1_seq2, fpath_dist_verts_mesh_feats)
            return dist_verts_seq1_seq2


    @property
    def feats(self):
        if self._mesh_feats is None:
            if not self.fpath_mesh_feats.exists():
                self.preprocess_mesh_feats()
            self._mesh_feats = torch.load(self.fpath_mesh_feats)
        return self._mesh_feats

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

    def get_trajectory(self, cam_tform_obj_source: CAM_TFORM_OBJ_SOURCES):

        cams_tform_obj = []
        for f in range(len(self.frames_names)):
            cams_tform_obj.append(self.get_frame_by_index(f).get_cam_tform4x4_obj(cam_tform_obj_source=cam_tform_obj_source))
        cams_tform_obj = torch.stack(cams_tform_obj, dim=0)
        obj_cams_traj = inv_tform4x4(cams_tform_obj)[:, :3, 3]
        return obj_cams_traj

    def get_a_src_tform_b_src(self, src_a: CAM_TFORM_OBJ_SOURCES, src_b: CAM_TFORM_OBJ_SOURCES, estimate_scale=True, device='cpu'):
        fpath = self.path_preprocess.joinpath('a_src_tform_b_src', f'{src_a}_tform_{src_b}', self.name_unique, f'{src_a}_tform_{src_b}.pt')
        if fpath.exists():
            a_src_tform_b_src = torch.load(fpath).to(device=device)
        else:
            obj_cams_traj_a = self.get_trajectory(cam_tform_obj_source=src_a).to(device=device)
            obj_cams_traj_b = self.get_trajectory(cam_tform_obj_source=src_b).to(device=device)

            #from od3d.cv.visual.show import show_scene
            #show_scene(pts3d=[obj_cams_traj_a[50:], obj_cams_traj_b[50:]])

            from od3d.cv.geometry.fit.tform4x4 import fit_tform4x4_with_matches
            a_src_tform_b_src = fit_tform4x4_with_matches(pts=obj_cams_traj_b, pts_ref=obj_cams_traj_a, estimate_scale=estimate_scale)

            fpath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(a_src_tform_b_src, fpath)
        return a_src_tform_b_src
    #    co3d_src_tform_src = tform4x4(
    #        inv_tform4x4(src_frame.get_cam_tform4x4_obj(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.CO3D)),
    #        src_frame.get_cam_tform4x4_obj(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.DROID_SLAM))

    def run_droid_slam(self):
        from od3d.io import run_cmd
        stride = "1"
        image_tag = "limpbot/droid-slam:v1"
        path_in = self.path_raw.joinpath(self.name_unique, 'images')
        path_out_root = self.path_preprocess.joinpath('droid_slam')
        rpath_out = self.name_unique
        path_out = path_out_root.joinpath(rpath_out)

        fx = self.first_frame.meta.l_cam_intr4x4[0][0]
        fy = self.first_frame.meta.l_cam_intr4x4[1][1]
        cx = self.first_frame.meta.l_cam_intr4x4[0][2]
        cy = self.first_frame.meta.l_cam_intr4x4[1][2]
        if not path_out.exists():
            path_out.mkdir(parents=True, exist_ok=True)
        run_cmd(cmd=f'echo "{fx} {fy} {cx} {cy}" > {path_out_root}/{rpath_out}/calib.txt', logger=logger)
        run_cmd(cmd=f'docker run --user=$(id -u):$(id -g) --gpus all -e RPATH_OUT={rpath_out} -e STRIDE={stride} -v {path_in}:/home/appuser/in -v {path_out_root}:/home/appuser/DROID-SLAM/reconstructions/out -t {image_tag}', logger=logger, live=True)

    def preprocess_mesh(self, override=False):

        if self.fpath_mesh.exists() and not override:
            logger.warning(f'mesh already exists {self.fpath_mesh}')
            return
        else:
            logger.info(f'preprocessing mesh for {self.name_unique} with type {self.mesh_name}')

        import numpy as np
        from od3d.cv.visual.show import show_scene
        from od3d.cv.geometry.fit.rays_center3d import fit_rays_center3d
        from od3d.cv.geometry.fit.plane4d import fit_plane, score_plane4d_fit
        from od3d.cv.geometry.transform import plane4d_to_tform4x4, tform4x4_from_transl3d
        from od3d.cv.optimization.ransac import ransac
        from od3d.cv.geometry.transform import tform4x4_broadcast
        from od3d.cv.geometry.mesh import Mesh
        from functools import partial
        from od3d.cv.geometry.downsample import random_sampling, voxel_downsampling

        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'


        fpath_droid_slam_pcl = self.path_preprocess.joinpath('droid_slam', self.category, self.name, 'pcl.ply')
        fpath_droid_slam_traj = self.path_preprocess.joinpath('droid_slam', self.category, self.name, 'traj_est.pt')

        if not fpath_droid_slam_pcl.exists() or not fpath_droid_slam_traj.exists():
            self.run_droid_slam()


        # N x 3
        pts3d = read_pts3d(fpath_droid_slam_pcl).to(device=device)
        pts3d_colors = read_pts3d_colors(fpath_droid_slam_pcl).to(device=device)

        # F x 4 x 4
        cams_tform4x4_obj = torch.load(fpath_droid_slam_traj).to(device=device)

        #show_scene(pts3d=[pts3d], pts3d_colors=[pts3d_colors], cams_tform4x4_world=cams_tform4x4_obj, cams_intr4x4=[self.first_frame.cam_intr4x4])

        pts3d_range = max(pts3d.max(dim=0)[0] - pts3d.min(dim=0)[0]).item()
        scene_size = pts3d_range
        import re
        match = re.match(r"([a-z]+)([0-9]+)", self.mesh_name, re.I)
        if match and len(match.groups()) == 2:
            mesh_type, mesh_vertices_count = match.groups()
            mesh_vertices_count = int(mesh_vertices_count)
        else:
            msg = f'could not retrieve mesh type and vertices count from mesh name {self.mesh_name}'
            raise Exception(msg)
        scene_particles = 2000
        _, mask_pts3d_sampled_rand = random_sampling(pts3d, pts3d_max_count=10000, return_mask=True)
        mask_pts3d_sampled = mask_pts3d_sampled_rand.clone()
        # _, mask_pts3d_sampled_voxel = voxel_downsampling(pts3d_cls=pts3d[mask_pts3d_sampled_rand], K=100000, top_bins_perc=0.90, return_mask=True)
        #mask_pts3d_sampled[mask_pts3d_sampled_rand] = mask_pts3d_sampled_voxel
        # particle_size = torch.cdist(pts3d[mask_pts3d_sampled], pts3d[mask_pts3d_sampled]).fill_diagonal_(torch.inf).min(dim=-1).values.mean()
        # average Kth nearest neighbor distance
        particle_size = torch.cdist(pts3d[mask_pts3d_sampled], pts3d[mask_pts3d_sampled]).quantile(dim=-1, q=3. / mask_pts3d_sampled.sum()).mean()

        plane_dist_thresh = particle_size / 1.
        plane_not_aggregating_dist_thresh = particle_size * 2.
        obj_agg_dist_thresh = particle_size * 10.

        center3d = fit_rays_center3d(cams_tform4x4_obj=cams_tform4x4_obj)
        center3d_tform4x4_obj = tform4x4_from_transl3d(-center3d)

        cams_tform4x4_obj = tform4x4_broadcast(cams_tform4x4_obj, inv_tform4x4(center3d_tform4x4_obj))
        pts3d = transf3d_broadcast(pts3d, transf4x4=center3d_tform4x4_obj)
        center3d = transf3d_broadcast(center3d, transf4x4=center3d_tform4x4_obj)

        center3d_mesh = Mesh.create_sphere(center3d=center3d, radius=scene_size / 30., device=device)
        #show_scene(meshes=[center3d_mesh], pts3d=[pts3d], pts3d_colors=[pts3d_colors], cams_tform4x4_world=cams_tform4x4_obj, cams_intr4x4=[self.first_frame.cam_intr4x4])


        cams_traj = inv_tform4x4(cams_tform4x4_obj)[:, :3, 3]
        plane4d = ransac(pts=pts3d[mask_pts3d_sampled], fit_func=fit_plane,
                         score_func=partial(score_plane4d_fit, plane_dist_thresh=plane_dist_thresh,
                                            cams_traj=cams_traj, pts_on_plane_weight=2.), fits_count=1000, fit_pts_count=3)

        plane3d_tform4x4_obj = plane4d_to_tform4x4(plane4d)

        #plane_z = -plane3d_tform4x4_obj[2, 3]
        #plane3d_tform4x4_obj[:3, 3] = 0.
        cams_tform4x4_obj = tform4x4_broadcast(cams_tform4x4_obj, inv_tform4x4(plane3d_tform4x4_obj))

        pts3d = transf3d_broadcast(pts3d, transf4x4=plane3d_tform4x4_obj)
        center3d = transf3d_broadcast(center3d, transf4x4=plane3d_tform4x4_obj)

        height = scene_size / 5.
        radius = scene_size * 0.5
        plan3d_mesh = Mesh.create_plane_as_cone(radius=radius, height=height, device=device)

        # show_scene(meshes=[center3d_mesh, plan3d_mesh], pts3d=[pts3d], pts3d_colors=[pts3d_colors], cams_tform4x4_world=cams_tform4x4_obj, cams_intr4x4=[self.first_frame.cam_intr4x4])

        #plane_tform4x4_pts3d = transf3d_broadcast(pts3d=pts3d, transf4x4=plane3d_tform4x4_obj)
        mask_pts3d_on_plane = mask_pts3d_sampled * (pts3d[:, 2] < plane_dist_thresh)
        mask_pts3d_on_plane_not_aggregating = mask_pts3d_sampled * (pts3d[:, 2] < plane_not_aggregating_dist_thresh)
        # starting with 10 percentage of points
        if (mask_pts3d_sampled * (~mask_pts3d_on_plane)).sum() < 10:
            logger.warning(f'Could not estimate mesh for sequence {self.name_unique} due to too few points after removing plane {(mask_pts3d_sampled * (~mask_pts3d_on_plane)).sum()}')
            return

        mask_center_thresh = (pts3d[mask_pts3d_sampled * (~mask_pts3d_on_plane)] - center3d).norm(dim=-1).quantile(0.1)
        mask_pts3d_obj = mask_pts3d_sampled * (~mask_pts3d_on_plane) * ((pts3d - center3d).norm(dim=-1) < mask_center_thresh)

        pts3d_not_on_plane = pts3d[mask_pts3d_sampled * (~mask_pts3d_on_plane)]
        dists_pts3d_not_on_plane_plane = pts3d[mask_pts3d_sampled * (~mask_pts3d_on_plane), 2]
        if ((~mask_pts3d_on_plane_not_aggregating) * mask_pts3d_obj).sum().item() > 0:
            dists_pts3d_not_on_plane_obj_dists = (pts3d[(~mask_pts3d_on_plane_not_aggregating) * mask_pts3d_obj][:, None] - pts3d[mask_pts3d_sampled * (~mask_pts3d_on_plane)][None, :]).norm(dim=-1).min(dim=0)[0]
            while ((dists_pts3d_not_on_plane_obj_dists < obj_agg_dist_thresh) * (dists_pts3d_not_on_plane_obj_dists < dists_pts3d_not_on_plane_plane)).sum() > mask_pts3d_obj.sum():
                logger.info(mask_pts3d_obj.sum())
                mask_pts3d_obj[mask_pts3d_sampled * (~mask_pts3d_on_plane)] += (dists_pts3d_not_on_plane_obj_dists < obj_agg_dist_thresh) * (dists_pts3d_not_on_plane_obj_dists < dists_pts3d_not_on_plane_plane)
                if ((~mask_pts3d_on_plane_not_aggregating) * mask_pts3d_obj).sum().item() > 0:
                    dists_pts3d_not_on_plane_obj_dists = torch.cdist(pts3d[(~mask_pts3d_on_plane_not_aggregating) * mask_pts3d_obj], pts3d_not_on_plane).min(dim=0)[0]
                else:
                    break

        pts3d_obj = pts3d[mask_pts3d_obj]

        if len(pts3d_obj) < 10:
            logger.warning(f'Could not estimate mesh for sequence {self.name_unique} due to too few points after removing plane {len(pts3d_obj)}')
            return

        pts3d_colors[mask_pts3d_on_plane] = torch.Tensor([0., 0., 1.]).to(device=device)
        pts3d_colors[mask_pts3d_obj] = torch.Tensor([0., 1., 0.]).to(device=device)
        pts3d_colors[mask_pts3d_sampled * (~mask_pts3d_on_plane) * (~mask_pts3d_obj)] = torch.Tensor([1., 0., 0.]).to(device=device)

        # show_scene(meshes=[center3d_mesh, plan3d_mesh], pts3d=[pts3d], pts3d_colors=[pts3d_colors], cams_tform4x4_world=cams_tform4x4_obj, cams_intr4x4=[self.first_frame.cam_intr4x4])

        o3d_pcl = open3d.geometry.PointCloud()
        o3d_pcl.points = open3d.utility.Vector3dVector(pts3d_obj[:].detach().cpu().numpy())
        o3d_pcl.normals = open3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals

        # #### OPTION 1: CONVEX HULL
        if mesh_type == 'convex':
            o3d_obj_mesh, _ = o3d_pcl.compute_convex_hull()
            o3d_obj_mesh.compute_vertex_normals()

            logger.info(o3d_obj_mesh)
            while np.asarray(o3d_obj_mesh.vertices).shape[0] < mesh_vertices_count:
                o3d_obj_mesh = o3d_obj_mesh.subdivide_loop(number_of_iterations=1)
                logger.info(o3d_obj_mesh)

            logger.info(o3d_obj_mesh)
            voxel_size = o3d_obj_mesh.get_volume() / 10.
            while np.asarray(o3d_obj_mesh.vertices).shape[0] > mesh_vertices_count:
                o3d_obj_mesh = o3d_obj_mesh.simplify_vertex_clustering(voxel_size=voxel_size,
                                                                       contraction=open3d.geometry.SimplificationContraction.Average)
                logger.info(o3d_obj_mesh)
                voxel_size = voxel_size * 2.

            obj_mesh = Mesh.from_o3d(o3d_obj_mesh, device=device)

        elif mesh_type == 'poisson':
            # #### OPTION 2: POISSON (requires normals)
            o3d_pcl.estimate_normals()
            o3d_obj_mesh, densities = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(o3d_pcl, depth=9)
            logger.info(o3d_obj_mesh)
            while np.asarray(o3d_obj_mesh.vertices).shape[0] < mesh_vertices_count:
                o3d_obj_mesh = o3d_obj_mesh.subdivide_loop(number_of_iterations=1)
                logger.info(o3d_obj_mesh)

            logger.info(o3d_obj_mesh)
            voxel_size = o3d_obj_mesh.get_volume() / 10.
            while np.asarray(o3d_obj_mesh.vertices).shape[0] > mesh_vertices_count:
                o3d_obj_mesh = o3d_obj_mesh.simplify_vertex_clustering(voxel_size=voxel_size,
                                                                       contraction=open3d.geometry.SimplificationContraction.Average)
                logger.info(o3d_obj_mesh)
                voxel_size = voxel_size * 2.

            obj_mesh = Mesh.from_o3d(o3d_obj_mesh, device=device)
        elif mesh_type == 'alpha':
            # #### OPTION 3: ALPHA_SHAPE
            alpha = 10 * particle_size
            o3d_obj_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(o3d_pcl, alpha)
            logger.info(o3d_obj_mesh)
            while np.asarray(o3d_obj_mesh.vertices).shape[0] < mesh_vertices_count:
                o3d_obj_mesh = o3d_obj_mesh.subdivide_loop(number_of_iterations=1)
                logger.info(o3d_obj_mesh)
            logger.info(o3d_obj_mesh)
            voxel_size = o3d_obj_mesh.get_surface_area() / 10.
            while np.asarray(o3d_obj_mesh.vertices).shape[0] > mesh_vertices_count:
                o3d_obj_mesh = o3d_obj_mesh.simplify_vertex_clustering(voxel_size=voxel_size,
                                                                       contraction=open3d.geometry.SimplificationContraction.Average)
                logger.info(o3d_obj_mesh)
                voxel_size = voxel_size * 2.

            obj_mesh = Mesh.from_o3d(o3d_obj_mesh, device=device)
        elif mesh_type == 'voxel':
            #### OPTION 4: VOXEL GRID
            from pytorch3d.ops.marching_cubes import marching_cubes
            voxel_grid, voxel_grid_range, voxel_grid_offset = voxel_downsampling(pts3d_cls=pts3d_obj, K=mesh_vertices_count * 2,
                                                                                 return_voxel_grid=True, min_steps=2)
            # vol_batch(N, D, H, W) ->  (X, Y, Z).permute(2, 0, 1)
            verts, faces = marching_cubes(vol_batch=voxel_grid.permute(2, 0, 1)[None,] * 1., return_local_coords=True)
            faces = faces[0].to(device=device)
            verts = (verts[0].to(device=device) + 1) / 2.

            obj_mesh = Mesh(verts=voxel_grid_offset[None,] + voxel_grid_range[None,] * verts, faces=faces)
        else:
            msg = f'Unknown mesh type {mesh_type}'
            raise Exception(msg)

        logger.info(f'vertices count = {obj_mesh.verts_count()}')
        #show_scene(meshes=[obj_mesh, center3d_mesh, plan3d_mesh], pts3d=[pts3d], pts3d_colors=[pts3d_colors],
        #           cams_tform4x4_world=cams_tform4x4_obj, cams_intr4x4=[self.first_frame.cam_intr4x4], meshes_as_wireframe=True)

        logger.info(self.name_unique)
        # open3d.visualization.draw(geometries)
        obj_tform_droid_slam = tform4x4(plane3d_tform4x4_obj, center3d_tform4x4_obj)
        # save point cloud clean
        pts3d = read_pts3d(fpath_droid_slam_pcl).to(device=device)
        pts3d = transf3d_broadcast(pts3d, transf4x4=obj_tform_droid_slam)
        pts3d_colors = read_pts3d_colors(fpath_droid_slam_pcl).to(device=device)

        # note: only if mask is not available due to downsampling consider this
        #vertices_dist_median = torch.cdist(pts3d_obj[None,], pts3d_obj[None,])[0].median()
        #mask_pts3d_obj = torch.cdist(pts3d[None,], pts3d_obj[None,])[0].min(dim=-1).values < vertices_dist_median / 5.

        mask_pts3d_clean = torch.cdist(pts3d[None,], pts3d[mask_pts3d_obj][None,])[0].min(dim=-1).values < particle_size
        pts3d_clean = pts3d[mask_pts3d_clean]
        pts3d_colors_clean = pts3d_colors[mask_pts3d_clean]

        #show_scene(meshes=[obj_mesh, center3d_mesh, plan3d_mesh], pts3d=[pts3d_clean], pts3d_colors=[pts3d_colors_clean],
        #           cams_tform4x4_world=cams_tform4x4_obj, cams_intr4x4=[self.first_frame.cam_intr4x4], meshes_as_wireframe=True)


        self.fpath_pcl_droid_slam_clean.parent.mkdir(parents=True, exist_ok=True)
        self.fpath_pcl_droid_slam.parent.mkdir(parents=True, exist_ok=True)
        write_pts3d_with_colors(fpath=self.fpath_pcl_droid_slam_clean, pts3d=pts3d_clean,
                                pts3d_colors=pts3d_colors_clean)
        write_pts3d_with_colors(fpath=self.fpath_pcl_droid_slam, pts3d=pts3d, pts3d_colors=pts3d_colors)

        obj_mesh.write_to_file(fpath=self.fpath_mesh)

        # save transformation
        for i, cam_tform4x4_obj in enumerate(cams_tform4x4_obj):
            frame = self.get_frame_by_index(i)
            frame.fpath_cam_tform4x4_obj_droid_slam.parent.mkdir(parents=True, exist_ok=True)
            torch.save(obj=cam_tform4x4_obj.detach().cpu(), f=frame.fpath_cam_tform4x4_obj_droid_slam)

    @property
    def fpath_droid_slam_axis_labeled(self):
        return self.path_preprocess.joinpath('axis', 'droid_slam', self.category, self.name, 'axis.pt')


    def get_cams(self, cam_tform_obj_source: CAM_TFORM_OBJ_SOURCES=CAM_TFORM_OBJ_SOURCES.DROID_SLAM, cams_count=5, show_imgs=True):
        cams_tform4x4_world = []
        cams_intr4x4 = []
        cams_imgs = []
        frames_count = len(self.frames_names)
        for c in range(0, frames_count, (frames_count // cams_count) + 1):
            frame = self.get_frame_by_index(c)
            if cam_tform_obj_source is not CAM_TFORM_OBJ_SOURCES.CO3D or cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM:
                cams_tform4x4_world.append(
                    frame.get_cam_tform4x4_obj(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.DROID_SLAM))
            else:
                cams_tform4x4_world.append(
                    frame.get_cam_tform4x4_obj(cam_tform_obj_source=cam_tform_obj_source))

            cams_intr4x4.append(frame.cam_intr4x4)
            if show_imgs:
                cams_imgs.append(frame.rgb)
        return cams_tform4x4_world, cams_intr4x4, cams_imgs

    def show(self, cam_tform_obj_source: CAM_TFORM_OBJ_SOURCES=CAM_TFORM_OBJ_SOURCES.DROID_SLAM, pcl_source: PCL_SOURCES=PCL_SOURCES.DROID_SLAM, cams_count=5, show_imgs=False):
        cams_tform4x4_world, cams_intr4x4, cams_imgs = self.get_cams(cam_tform_obj_source=cam_tform_obj_source, cams_count=cams_count, show_imgs=show_imgs)
        if not show_imgs:
            cams_imgs = None
        pts3d = self.get_pcl(pcl_source=pcl_source)
        pts3d_colors = self.get_pcl_colors(pcl_source=pcl_source)
        from od3d.cv.visual.show import show_scene
        show_scene(cams_tform4x4_world=cams_tform4x4_world, cams_intr4x4=cams_intr4x4, cams_imgs=cams_imgs, pts3d=[pts3d], pts3d_colors=[pts3d_colors])

    @property
    def droid_slam_axis_labeled(self):
        fpath_axis_droid_slam = self.fpath_droid_slam_axis_labeled
        if not fpath_axis_droid_slam.exists():
            fpath_axis_droid_slam.parent.mkdir(parents=True, exist_ok=True)
            from od3d.cv.label.axis import label_axis_in_pcl

            cams_tform4x4_world = []
            cams_intr4x4 = []
            cams_imgs = []
            frames_count = len(self.frames_names)
            for c in range(0, frames_count, frames_count // 4):
                frame = self.get_frame_by_index(c)
                if self.cam_tform_obj_source is not CAM_TFORM_OBJ_SOURCES.CO3D or self.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM:
                    cams_tform4x4_world.append(frame.get_cam_tform4x4_obj(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.DROID_SLAM))
                else:
                    cams_tform4x4_world.append(
                        frame.get_cam_tform4x4_obj(cam_tform_obj_source=self.cam_tform_obj_source))

                cams_intr4x4.append(frame.cam_intr4x4)
                cams_imgs.append(frame.rgb)

            axis_droid_slam = label_axis_in_pcl(pts3d=self.get_pcl(pcl_source=PCL_SOURCES.DROID_SLAM_CLEAN), pts3d_colors=self.get_pcl_colors(pcl_source=PCL_SOURCES.DROID_SLAM_CLEAN),
                                                cams_tform4x4_world=cams_tform4x4_world,
                                                cams_intr4x4=cams_intr4x4,
                                                cams_imgs=cams_imgs)
            logger.info(f'{axis_droid_slam}')

            if axis_droid_slam is not None and axis_droid_slam.shape == (3, 2, 3):
                torch.save(axis_droid_slam, f=fpath_axis_droid_slam)

        else:
            axis_droid_slam = torch.load(fpath_axis_droid_slam)
        return axis_droid_slam

    @property
    def fpath_droid_slam_labeled_tform_droid_slam(self):
        return self.path_droid_slam_labeled_tform_droid_slam.joinpath(self.name_unique, 'tform.pt')

    @property
    def path_droid_slam_labeled_tform_droid_slam(self):
        return self.path_preprocess.joinpath('a_src_tform_b_src', 'droid_slam_labeled_tform_droid_slam')

    @property
    def droid_slam_labeled_tform_droid_slam(self):
        from od3d.cv.geometry.fit.axis_tform_from_pts3d import axis_tform4x4_obj_from_pts3d
        fpath_droid_slam_labeled_tform_droid_slam = self.fpath_droid_slam_labeled_tform_droid_slam
        if fpath_droid_slam_labeled_tform_droid_slam.exists():
            droid_slam_labeled_tform_droid_slam = torch.load(fpath_droid_slam_labeled_tform_droid_slam)
        else:
            logger.info(f'labeling axis for sequence {self.name_unique}')
            droid_slam_labeled_tform_droid_slam = axis_tform4x4_obj_from_pts3d(axis_pts3d=self.droid_slam_axis_labeled)
            logger.info(f'blub')

        while (torch.linalg.det(droid_slam_labeled_tform_droid_slam[:3, :3]) - 1.).abs() > 1e-5:
            logger.warning(f'labeled determinant is not ~1, but {torch.linalg.det(droid_slam_labeled_tform_droid_slam[:3, :3])}')
            if self.fpath_droid_slam_axis_labeled.exists():
                self.fpath_droid_slam_axis_labeled.unlink()
            if self.fpath_droid_slam_labeled_tform_droid_slam.exists():
                self.fpath_droid_slam_labeled_tform_droid_slam.unlink()
            droid_slam_labeled_tform_droid_slam = axis_tform4x4_obj_from_pts3d(axis_pts3d=self.droid_slam_axis_labeled)

        fpath_droid_slam_labeled_tform_droid_slam.parent.mkdir(parents=True, exist_ok=True)
        torch.save(droid_slam_labeled_tform_droid_slam, fpath_droid_slam_labeled_tform_droid_slam)
        return droid_slam_labeled_tform_droid_slam

    @property
    def path_zsp_labels(self):
        return Path('third_party/zero-shot-pose/data/class_labels')
    @property
    def co3dv1_zsp_obj_tform_co3dv1_obj(self):
        import numpy as np
        from od3d.io import read_json
        path_zsp = self.path_zsp_labels
        fpath_src_gt = path_zsp.joinpath(self.name_unique + '.json')
        #if fpath_src_gt.exists():
        gt_co3d_global_tform_co3d_src = inv_tform4x4(torch.from_numpy(np.array(read_json(fpath_src_gt)['trans'])))
        gt_co3d_global_tform_co3d_src = gt_co3d_global_tform_co3d_src.to(torch.float)
        return gt_co3d_global_tform_co3d_src


    @property
    def co3dv1_zsp_obj_tform_droid_slam_obj(self):
        droid_slam_obj_tform_co3dv1_obj = self.get_a_src_tform_b_src(src_a=CAM_TFORM_OBJ_SOURCES.DROID_SLAM, src_b=CAM_TFORM_OBJ_SOURCES.CO3DV1)
        co3dv1_zsp_obj_tform_droid_slam_obj = tform4x4(self.co3dv1_zsp_obj_tform_co3dv1_obj, inv_tform4x4(droid_slam_obj_tform_co3dv1_obj))
        return co3dv1_zsp_obj_tform_droid_slam_obj

    @property
    def zsp_labeled_cuboid_ref_tform_droid_slam_obj(self):
        ref_seq_name = list(self.path_zsp_labels.joinpath(self.category).iterdir())[0].stem
        ref_seq = self.get_sequence_by_category_and_name(category=self.category, name=ref_seq_name)

        droid_slam_labeled_ref_tform_droid_slam_obj = tform4x4(ref_seq.droid_slam_labeled_cuboid_tform_droid_slam_labeled, tform4x4(ref_seq.droid_slam_labeled_tform_droid_slam, tform4x4(inv_tform4x4(ref_seq.co3dv1_zsp_obj_tform_droid_slam_obj), self.co3dv1_zsp_obj_tform_droid_slam_obj)))
        # note: as zsp does not offer scale, we cannot retrieve actual translation, therefore we use this pcl center
        droid_slam_labeled_ref_tform_droid_slam_obj[:3, 3] = 0
        droid_slam_labeled_ref_tform_droid_slam_obj[:3, 3] = -transf3d_broadcast(self.get_pcl(pcl_source=PCL_SOURCES.DROID_SLAM_CLEAN).mean(dim=0), transf4x4=droid_slam_labeled_ref_tform_droid_slam_obj)

        return droid_slam_labeled_ref_tform_droid_slam_obj

    @property
    def fpath_droid_slam_labeled_cuboid(self):
        return self.path_preprocess.joinpath('mesh', 'cuboid', self.name_unique, 'mesh.ply')

    @property
    def droid_slam_labeled_cuboid(self):
        fpath_droid_slam_labeled_cuboid = self.fpath_droid_slam_labeled_cuboid
        if fpath_droid_slam_labeled_cuboid.exists():
            droid_slam_labeled_cuboid = Meshes.load_from_files([fpath_droid_slam_labeled_cuboid])
        else:
            size = OD3D_CATEGORIES_SIZES_IN_M[MAP_CATEGORIES_CO3D_TO_OD3D[self.category]]
            droid_slam_labeled_tform_pts3d = transf3d_broadcast(
                pts3d=self.get_pcl(pcl_source=PCL_SOURCES.DROID_SLAM_CLEAN),
                transf4x4=self.droid_slam_labeled_tform_droid_slam)
            droid_slam_labeled_cuboid, droid_slam_labeled_cuboid_tform_droid_slam_labeled = \
                fit_cuboid_to_pts3d(pts3d=droid_slam_labeled_tform_pts3d, size=size, optimize_rot=False,
                                    optimize_transl=True)

            fpath_droid_slam_labeled_cuboid.parent.mkdir(parents=True, exist_ok=True)
            droid_slam_labeled_cuboid.write_to_file(fpath=fpath_droid_slam_labeled_cuboid)

            self.fpath_droid_slam_labeled_cuboid_tform_droid_slam_labeled.parent.mkdir(parents=True, exist_ok=True)
            torch.save(droid_slam_labeled_cuboid_tform_droid_slam_labeled.detach().cpu(),
                       f=self.fpath_droid_slam_labeled_cuboid_tform_droid_slam_labeled)

        return droid_slam_labeled_cuboid

    @property
    def fpath_droid_slam_labeled_cuboid_tform_droid_slam_labeled(self):
        return self.path_preprocess.joinpath('a_src_tform_b_src', 'droid_slam_labeled_cuboid_tform_droid_slam_labeled', self.name_unique, 'tform.pt')
    @property
    def droid_slam_labeled_cuboid_tform_droid_slam_labeled(self):
        fpath_droid_slam_labeled_cuboid_tform_droid_slam_labeled = self.fpath_droid_slam_labeled_cuboid_tform_droid_slam_labeled
        if fpath_droid_slam_labeled_cuboid_tform_droid_slam_labeled.exists():
            droid_slam_labeled_cuboid_tform_droid_slam_labeled = torch.load(fpath_droid_slam_labeled_cuboid_tform_droid_slam_labeled)
        else:
            size = OD3D_CATEGORIES_SIZES_IN_M[MAP_CATEGORIES_CO3D_TO_OD3D[self.category]]
            droid_slam_labeled_tform_pts3d = transf3d_broadcast(
                pts3d=self.get_pcl(pcl_source=PCL_SOURCES.DROID_SLAM_CLEAN),
                transf4x4=self.droid_slam_labeled_tform_droid_slam)
            droid_slam_labeled_cuboid, droid_slam_labeled_cuboid_tform_droid_slam_labeled = \
                fit_cuboid_to_pts3d(pts3d=droid_slam_labeled_tform_pts3d, size=size, optimize_rot=False,
                                    optimize_transl=True)

            fpath_droid_slam_labeled_cuboid_tform_droid_slam_labeled.parent.mkdir(parents=True, exist_ok=True)
            torch.save(droid_slam_labeled_cuboid_tform_droid_slam_labeled.detach().cpu(), f=fpath_droid_slam_labeled_cuboid_tform_droid_slam_labeled)

            self.fpath_droid_slam_labeled_cuboid.parent.mkdir(parents=True, exist_ok=True)
            droid_slam_labeled_cuboid.write_to_file(fpath=self.fpath_droid_slam_labeled_cuboid)
        return droid_slam_labeled_cuboid_tform_droid_slam_labeled

    def write_aligned_droid_slam_tform_droid_slam(self, aligned_droid_slam_tform_droid_slam: torch.Tensor, aligned_name: str):
        fpath_tform_aligned = self.get_fpath_droid_slam_aligned_tform_droid_slam_with_aligned_name(aligned_name=aligned_name)
        fpath_tform_aligned.parent.mkdir(parents=True, exist_ok=True)
        torch.save(aligned_droid_slam_tform_droid_slam.detach().cpu(), f=fpath_tform_aligned)

    def write_aligned_cuboid(self, cuboid: Meshes, aligned_name: str):
        fpath_mesh_aligned = self.path_preprocess.joinpath('aligned', aligned_name, 'mesh', self.category, 'mesh.ply')
        fpath_mesh_aligned.parent.mkdir(parents=True, exist_ok=True)
        cuboid.write_to_file(fpath=fpath_mesh_aligned)

    @property
    def droid_slam_aligned_tform_droid_slam(self):
        fpath_tform_aligned = self.fpath_droid_slam_aligned_tform_droid_slam
        droid_slam_aligned_tform_droid_slam = torch.load(fpath_tform_aligned).to('cpu')
        return droid_slam_aligned_tform_droid_slam

    @property
    def gt_pose_available(self):
        if self.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM_ALIGNED:
            return self.fpath_droid_slam_aligned_tform_droid_slam.exists()
        elif self.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM_LABELED:
            return self.fpath_droid_slam_labeled_cuboid_tform_droid_slam_labeled.exists()
        else:
            return True
    @property
    def fpath_droid_slam_aligned_tform_droid_slam(self):
        return self.get_fpath_droid_slam_aligned_tform_droid_slam_with_aligned_name(aligned_name=self.aligned_name)

    def get_fpath_droid_slam_aligned_tform_droid_slam_with_aligned_name(self, aligned_name: str):
        return self.path_preprocess.joinpath('aligned', aligned_name, 'aligned_droid_slam_tform_droid_slam', self.name_unique, 'tform.pt')

    @property
    def fpath_mesh(self):
        if self.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM_ALIGNED:
            return self.path_preprocess.joinpath('aligned', self.aligned_name, 'mesh', self.category, 'mesh.ply')
        elif self.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM_ZSP_LABELED:
            ref_seq_name = list(self.path_zsp_labels.joinpath(self.category).iterdir())[0].stem
            ref_seq = self.get_sequence_by_category_and_name(category=self.category, name=ref_seq_name)
            #_ = ref_seq.droid_slam_labeled_cuboid # leads to infinity loop
            return ref_seq.fpath_droid_slam_labeled_cuboid
        else:
            return self.path_preprocess.joinpath('mesh', f'{self.mesh_name}', self.category, self.name, f'mesh.ply')

    def get_fpath_mesh(self, mesh_source: CUBOID_SOURCES ):
        if mesh_source == CUBOID_SOURCES.ALIGNED:
            return self.path_preprocess.joinpath('aligned', self.aligned_name, 'mesh', self.category, 'mesh.ply')
        elif self.cam_tform_obj_source == CUBOID_SOURCES.ZSP_REF_CUBOID:
            ref_seq_name = list(self.path_zsp_labels.joinpath(self.category).iterdir())[0].stem
            ref_seq = self.get_sequence_by_category_and_name(category=self.category, name=ref_seq_name)
            #_ = ref_seq.droid_slam_labeled_cuboid # leads to infinity loop
            return ref_seq.fpath_droid_slam_labeled_cuboid
        else:
            return self.path_preprocess.joinpath('mesh', f'{self.mesh_name}', self.category, self.name, f'mesh.ply')

    def get_mesh(self, mesh_source: CUBOID_SOURCES):
        fpath_mesh = self.get_fpath_mesh(mesh_source=mesh_source)
        if not fpath_mesh.exists():
            self.preprocess_mesh()
        return Mesh.load_from_file(fpath=fpath_mesh)

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
    def fpath_categorical_pca_V(self):
        return self.path_preprocess.joinpath('pca', 'categorical_pca_V', self.mesh_feats_type, self.category)

    @property
    def categorical_pca_V(self):
        if self.fpath_categorical_pca_V.exists():
            return torch.load(self.fpath_categorical_pca_V)
        else:
            logger.warning(f'Categorical pca V does not exist for sequence {self.name_unique}')
            return None

    @categorical_pca_V.setter
    def categorical_pca_V(self, value: torch.Tensor):
        torch.save(value.detach().cpu(), f=self.fpath_categorical_pca_V)

    @property
    def cuboid(self):
        if self._cuboid is None:
            fpath_cuboid = self.fpath_cuboid
            if not fpath_cuboid.exists():
                self.preprocess_cuboid()
            self._cuboid = Cuboids.load_from_files(fpaths_meshes=[fpath_cuboid])
        return self._cuboid

