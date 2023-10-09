import logging

import torchvision.io.image

logger = logging.getLogger(__name__)
import shutil
from od3d.datasets.co3d import CO3D
from od3d.datasets.co3d.enum import CO3D_CATEGORIES, ALLOW_LIST_FRAME_TYPES
from od3d.datasets.dataset import OD3D_Dataset, OD3D_FRAME_MODALITIES, OD3D_DATASET_SPLITS
from omegaconf import DictConfig
from pathlib import Path
from od3d.io import run_cmd
from co3d.dataset.data_types import (
    load_dataclass_jgzip, FrameAnnotation, SequenceAnnotation
)
from od3d.datasets.co3d.frame import CO3D_Frame, CO3D_FrameMeta
from od3d.datasets.co3d.sequence import CO3D_Sequence, CO3D_SequenceMeta
from tqdm import tqdm
from typing import List


from od3d.cv.geometry.points_alignment import get_pca_tform_world
from od3d.cv.geometry.transform import transf3d_broadcast, tform4x4, inv_tform4x4
from od3d.cv.geometry.primitives import Cuboids
from od3d.cv.geometry.points_alignment import icp

from dataclasses import dataclass
import torch.utils.data
from od3d.cv.geometry.downsample import voxel_downsampling, random_sampling
import od3d.io
import open3d
import torch
from pytorch3d.io import load_ply, save_ply

class CO3Dv1(CO3D):

    @staticmethod
    def setup(config: DictConfig):

        # logger.info(OmegaConf.to_yaml(config))
        path_raw = Path(config.path_raw)
        if path_raw.exists() and config.setup.remove_previous:
            logger.info(f"Removing previous CO3Dv1")
            shutil.rmtree(path_raw)

        if path_raw.exists() and not config.setup.override:
            logger.info(f"Found CO3D dataset at {path_raw}")
        else:
            path_co3d_repo = path_raw.joinpath('co3d')
            path_co3d_repo.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cloning CO3D github repository to {path_co3d_repo}")
            run_cmd(cmd=f'cd {path_raw} && git clone git@github.com:facebookresearch/co3d.git', live=True, logger=logger)
            run_cmd(cmd=f'cd {path_raw}/co3d && git fetch', live=True, logger=logger)
            run_cmd(cmd=f'cd {path_raw}/co3d && git checkout v1', live=True, logger=logger)

            logger.info(f"Downloading CO3D dataset at {path_raw}")
            run_cmd(cmd=f'python {path_co3d_repo.joinpath("download_dataset.py")} --download_folder {path_raw}', live=True, logger=logger)

    @staticmethod
    def extract_meta(config: DictConfig):
        path = Path(config.path_raw)
        path_meta = CO3D.get_path_meta(config=config)

        dict_nested_frames = config.get('dict_nested_frames', None)
        dict_nested_frames_banned = config.get('dict_nested_frames_ban', None)
        preprocess_meta_override = config.get('extract_meta', False).get('override', False)
        preprocess_meta_remove_previous = config.get('extract_meta', False).get('remove_previous', False)

        categories = list(dict_nested_frames.keys()) if dict_nested_frames is not None else CO3D_CATEGORIES.list()
        sequences_count_max_per_class = config.get("sequences_count_max_per_class", None)

        if preprocess_meta_remove_previous:
            if path_meta.exists():
                shutil.rmtree(path_meta)

        for category in categories:
            sequences_names = list(dict_nested_frames[category].keys()) if dict_nested_frames is not None and dict_nested_frames[category] is not None else None
            if dict_nested_frames_banned is not None and category in dict_nested_frames_banned.keys() and dict_nested_frames_banned[category] is not None:
                sequences_names = list(filter(lambda seq: seq not in dict_nested_frames_banned[category].keys(), sequences_names))
            logger.info(f'preprocess meta for class {category}')
            sequence_annotations = load_dataclass_jgzip(
                f"{path}/{category}/sequence_annotations.jgz", List[SequenceAnnotation]
            )
            logger.info('reading sequence annotations...')
            seq_count_per_class = 0
            read_sequences = []
            for sequence_annoation in tqdm(sequence_annotations):
                if sequences_names is not None and sequence_annoation.sequence_name not in sequences_names:
                    continue

                read_sequences.append(sequence_annoation.sequence_name)
                seq_count_per_class += 1
                if sequences_count_max_per_class is not None:
                    if seq_count_per_class > sequences_count_max_per_class:
                        break

                sequence_name = str(sequence_annoation.sequence_name)
                sequence_meta_fpath = CO3D_SequenceMeta.get_fpath_sequence_meta_with_category_and_name(path_meta=path_meta,
                                                                                                       category=category,
                                                                                                       name=sequence_name)

                if sequence_meta_fpath.exists() and not preprocess_meta_override:
                    continue

                sequence_meta = CO3D_SequenceMeta.load_from_raw(sequence_annotation=sequence_annoation)
                sequence_meta.save(path_meta=path_meta)

            cls_frame_annotations = load_dataclass_jgzip(
                f"{path}/{category}/frame_annotations.jgz", List[FrameAnnotation]
            )
            #cls_frame_annotations = [fa for fa in cls_frame_annotations if fa.meta[
            #    'frame_type'] in ALLOW_LIST_FRAME_TYPES]

            logger.info('reading frame annotations...')
            for frame_annotation in tqdm(cls_frame_annotations):
                if sequences_names is not None and frame_annotation.sequence_name not in sequences_names:
                    continue

                if frame_annotation.sequence_name not in read_sequences:
                    continue

                frame_name = str(frame_annotation.frame_number)
                sequence_name = str(frame_annotation.sequence_name)
                frame_meta_fpath = CO3D_FrameMeta.get_fpath_frame_meta_with_category_sequence_and_frame_name(path_meta=path_meta,
                                                                                                             category=category,
                                                                                                             sequence_name=sequence_name,
                                                                                                             name=frame_name)
                if frame_meta_fpath.exists() and not preprocess_meta_override:
                    continue


                frame_meta = CO3D_FrameMeta.load_from_raw(frame_annotation=frame_annotation)
                frame_meta.save(path_meta=path_meta)

    def get_sequence_by_category_and_name(self, category, name):
        sequence_meta = CO3D_SequenceMeta.load_from_meta_with_category_and_name(path_meta=self.path_meta, category=category, name=name)
        return CO3Dv1_Sequence(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
                               meta=sequence_meta, modalities=self.modalities, categories=self.categories,
                               mesh_feats_type=self.mesh_feats_type, dist_verts_mesh_feats_reduce_type=self.dist_verts_mesh_feats_reduce_type, cuboid_source=self.cuboid_source,
                               cam_tform_obj_source=self.cam_tform_obj_source)


class CO3Dv1_Sequence(CO3D_Sequence):
    def preprocess_mesh(self):
        fpath_droid_slam_pcl = self.path_preprocess.joinpath('droid_slam', self.category, self.name, 'pcl.ply')
        fpath_droid_slam_traj = self.path_preprocess.joinpath('droid_slam', self.category, self.name, 'traj_est.pt')

        if not fpath_droid_slam_pcl.exists() or not fpath_droid_slam_traj.exists():
            from od3d.io import run_cmd
            stride = "1"
            image_tag = "limpbot/droid-slam:v1"
            path_co3d_in = self.path_raw.joinpath(self.name_unique, 'images')
            path_out_root = self.path_preprocess.joinpath('droid_slam')
            rpath_out = self.name_unique
            path_out = path_out_root.joinpath(rpath_out)
            path_in = path_out.joinpath('images')
            if not path_in.exists():
                path_in.mkdir(parents=True, exist_ok=True)

            H = 999999999
            W = 999999999
            for f_id in range(len(self.frames_names)):
                frame = self.get_frame_by_index(f_id)
                frame_H = frame.H - frame.H % 8 # 8 required fore droid slam to work
                frame_W = frame.W - frame.W % 8 # 8 required fore droid slam to work
                if frame_H < H:
                    H = frame_H
                if frame_W < W:
                    W = frame_W

            for f_id in range(len(self.frames_names)):
                frame = self.get_frame_by_index(f_id)
                rgb = frame.rgb[:, :H, :W].clone()
                torchvision.io.image.write_jpeg(rgb, filename=str(path_in.joinpath(f'{f_id:05d}' + '.jpg')))

            fx = self.first_frame.meta.l_cam_intr4x4[0][0]
            fy = self.first_frame.meta.l_cam_intr4x4[1][1]
            cx = self.first_frame.meta.l_cam_intr4x4[0][2]
            cy = self.first_frame.meta.l_cam_intr4x4[1][2]
            if not path_out.exists():
                path_out.mkdir(parents=True, exist_ok=True)
            run_cmd(cmd=f'echo "{fx} {fy} {cx} {cy}" > {path_out_root}/{rpath_out}/calib.txt', logger=logger)
            run_cmd(cmd=f'docker run --user=$(id -u):$(id -g) --gpus all -e RPATH_OUT={rpath_out} -e STRIDE={stride} -v {path_in}:/home/appuser/in -v {path_out_root}:/home/appuser/DROID-SLAM/reconstructions/out -t {image_tag}', logger=logger, live=True)

        # N x 3
        pts3d, _ = load_ply(fpath_droid_slam_pcl)
        # F x 4 x 4
        cams_tform4x4_obj = torch.load(fpath_droid_slam_traj)
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

        #o3d_pcl = open3d.geometry.PointCloud()
        #o3d_pcl.points = open3d.utility.Vector3dVector(pts3d[:].detach().cpu().numpy())
        #o3d_pcl.normals = open3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
        #o3d_pcl.estimate_normals()
        #open3d.visualization.draw({'name': 'pcl', 'geometry': o3d_pcl})

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

        logger.info(self.name_unique)
        # open3d.visualization.draw(geometries)

        save_ply(f=self.fpath_mesh, verts=torch.from_numpy(np.asarray(mesh.vertices)), faces=torch.LongTensor(np.asarray(mesh.triangles)))

        for i, cam_tform4x4_obj in enumerate(cams_tform4x4_obj):
            frame = self.get_frame_by_index(i)
            frame.fpath_cam_tform4x4_obj_droid_slam.parent.mkdir(parents=True, exist_ok=True)
            torch.save(obj=cam_tform4x4_obj.detach().cpu(), f=frame.fpath_cam_tform4x4_obj_droid_slam)