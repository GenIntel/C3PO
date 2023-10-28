import logging
logger = logging.getLogger(__name__)
import time
from typing import List
import od3d.io
from od3d.methods.method import OD3D_Method
from od3d.datasets.dataset import OD3D_Dataset
from od3d.benchmark.results import OD3D_Results
from od3d.datasets.co3d import CO3D

from omegaconf import DictConfig
import pytorch3d.transforms
import pandas as pd
import numpy as np
from torch.utils.data import RandomSampler
from od3d.cv.visual.show import show_scene
import math
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from od3d.cv.geometry.transform import se3_exp_map
from od3d.cv.visual.show import imgs_to_img
from od3d.cv.geometry.mesh import Meshes
from pathlib import Path
from od3d.cv.geometry.transform import transf4x4_from_spherical, tform4x4, rot3x3
from od3d.cv.visual.show import show_img
import torchvision
from od3d.cv.visual.blend import blend_rgb
from od3d.cv.visual.sample import sample_pxl2d_pts
from tqdm import tqdm
from od3d.cv.geometry.mesh import MESH_RENDER_MODALITIES
from od3d.cv.metric.pose import get_pose_diff_in_rad

from od3d.cv.io import image_as_wandb_image
from od3d.cv.visual.resize import resize
from od3d.models.model import OD3D_Model

from od3d.cv.geometry.grid import get_pxl2d_like
from od3d.cv.geometry.fit3d2d import batchwise_fit_se3_to_corresp_3d_2d_and_masks
from od3d.cv.transforms.transform import OD3D_Transform
from od3d.cv.transforms.sequential import SequentialTransform

from typing import Dict
from od3d.data.ext_enum import ExtEnum

import matplotlib.pyplot as plt

plt.switch_backend('Agg')
from od3d.cv.visual.show import get_img_from_plot
from od3d.cv.visual.draw import draw_text_in_rgb
from od3d.cv.optimization.ransac import ransac
from od3d.cv.geometry.fit.tform4x4 import fit_tform4x4, score_tform4x4_fit

from functools import partial
from pathlib import Path
from od3d.io import read_json
from od3d.cv.geometry.transform import inv_tform4x4, tform4x4
from od3d.datasets.co3d.enum import CAM_TFORM_OBJ_SOURCES

class VISUAL_MODALITIES(str, ExtEnum):
    PRED_VERTS_NCDS_IN_RGB = 'pred_verts_ncds_in_rgb'
    GT_VERTS_NCDS_IN_RGB = 'gt_verts_ncds_in_rgb'
    PRED_VS_GT_VERTS_NCDS_IN_RGB = 'pred_vs_gt_verts_ncds_in_rgb'
    NET_FEATS_NEAREST_VERTS = 'net_feats_nearest_verts'
    SIM_PXL = 'sim_pxl'
    SAMPLES = 'samples'

class SIM_FEATS_MESH_WITH_IMAGE(str, ExtEnum):
    VERTS2D = 'verts2d'
    RENDERED = 'rendered'

class NeMo_Align3D(OD3D_Method):
    def setup(self):
        pass


    def __init__(
            self,
            config: DictConfig,
            logging_dir,
    ):
        super().__init__(config=config, logging_dir=logging_dir)

        self.device = 'cuda:0'

        # init Network
        self.net = OD3D_Model(config.model)

        self.transform_train = SequentialTransform([
            OD3D_Transform.subclasses[config.train.transform.class_name].create_from_config(config=config.train.transform),
            self.net.transform,
        ])
        self.transform_test = SequentialTransform([
            OD3D_Transform.subclasses[config.test.transform.class_name].create_from_config(config=config.test.transform),
            self.net.transform
        ])
        # if config.train.transform.random_color:
        #     self.transform_train = torchvision.transforms.Compose([
        #         RandomCenterZoom3D(**config.train.transform.random_center_zoom3d),
        #         #RGB_Random(),
        #         self.net.transform,
        #     ])
        # else:
        #     self.transform_train = torchvision.transforms.Compose([
        #         RandomCenterZoom3D(**config.train.transform.random_center_zoom3d),
        #         self.net.transform,
        #     ])

        #self.transform_test = torchvision.transforms.Compose([
        #    CenterZoom3D(**config.test.transform),
        #    self.net.transform
        #])

        self.meshes = None
        self.sequences_unique_names = None
        self.meshes = None

        # init Meshes / Features
        self.total_params = sum(p.numel() for p in self.net.parameters())
        self.net.eval()
        self.net.cpu()

        self.config.down_sample_rate = 16
        self.down_sample_rate = self.config.down_sample_rate

        self.dir_tmp = Path('/tmp').joinpath(self.__class__.__name__)
        self.dir_tmp.mkdir(parents=True, exist_ok=True)
    def train(self, datasets_train: Dict[str, CO3D], datasets_val: Dict[str, OD3D_Dataset]):
        score_metric_name = 'pose/acc_pi18'  # 'pose/acc_pi18' 'pose/acc_pi6'
        score_ckpt_val = 0.
        score_latest = 0.

        dataset_src: CO3D = datasets_train['src']
        dataset_ref: CO3D = datasets_train['labeled']
        from od3d.cv.geometry.mesh import Meshes

        categories = dataset_src.categories

        src_sequences = dataset_src.get_sequences()
        ref_sequences = dataset_ref.get_sequences()

        #sequences = src_sequences + ref_sequences

        # tform4x4(inv_tform4x4(src_frame.get_cam_tform4x4_obj(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.CO3D)), src_frame.get_cam_tform4x4_obj(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.DROID_SLAM))
        logger.info('loading mesh feats...')
        #self.sequences_mesh_feats = [seq.feats for seq in self.sequences]
        src_sequences_unique_names = [seq.name_unique for seq in src_sequences]
        ref_sequences_unique_names = [seq.name_unique for seq in ref_sequences]
        #sequences_unique_names = [seq.name_unique for seq in sequences]

        src_map_seq_to_cat = torch.LongTensor([categories.index(name.split('/')[0]) for name in src_sequences_unique_names])
        ref_map_seq_to_cat = torch.LongTensor([categories.index(name.split('/')[0]) for name in ref_sequences_unique_names])

        categories_count = len(categories)

        src_instances_count_per_category = [(src_map_seq_to_cat == c).sum().item() for c in range(categories_count)]
        ref_instances_count_per_category = [(ref_map_seq_to_cat == c).sum().item() for c in range(categories_count)]

        logger.info('loading meshes...')
        src_meshes = Meshes.load_from_meshes([seq.mesh for seq in src_sequences], device=self.device)
        ref_meshes = Meshes.load_from_meshes([seq.mesh for seq in ref_sequences], device=self.device)

        # note: first get mesh to get DROID_SLAM tforms
        if self.config.use_gt_src:
            logger.info('getting co3d_tform_droid_slam for each instance...')
            if dataset_src.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM_ZSP_LABELED:
                for seq in src_sequences:
                    _ = seq.get_a_src_tform_b_src(CAM_TFORM_OBJ_SOURCES.DROID_SLAM, CAM_TFORM_OBJ_SOURCES.CO3DV1, device=self.device)
                for seq in ref_sequences:
                    _ = seq.get_a_src_tform_b_src(CAM_TFORM_OBJ_SOURCES.DROID_SLAM, CAM_TFORM_OBJ_SOURCES.CO3DV1, device=self.device)

        src_instances_count = len(src_meshes)
        ref_instances_count = len(ref_meshes)

        #self.meshes_verts_aggregated_features = [vert_feats for mesh_feats in self.sequences_mesh_feats for vert_feats in mesh_feats]
        #dtype = self.meshes_verts_aggregated_features[0].dtype
        #vertices_count = len(self.meshes_verts_aggregated_features)
        dtype = src_meshes.verts.dtype
        src_vertices_count = len(src_meshes.verts)
        ref_vertices_count = len(ref_meshes.verts)
        #self.sequences_meshes.rgb = self.sequences_meshes.get_verts_ncds_cat_with_mesh_ids()
        #self.sequences_meshes.show()

        src_sequences_mesh_ids_for_verts = src_meshes.get_mesh_ids_for_verts()
        ref_sequences_mesh_ids_for_verts = ref_meshes.get_mesh_ids_for_verts()

        results_diff_log_rot = {}
        all_pred_ref_tform_src = {}
        all_pred_pose_dist_geo = {}
        all_pred_pose_dist_appear = {}
        for cat_id, category in enumerate(categories):
            logger.info(f'category id {cat_id} name {category}')
            src_instance_ids = torch.LongTensor(list(range(src_instances_count)))
            ref_instance_ids = torch.LongTensor(list(range(ref_instances_count)))

            src_category_instance_ids = src_instance_ids[src_map_seq_to_cat == cat_id]
            ref_category_instance_ids = ref_instance_ids[ref_map_seq_to_cat == cat_id]

            # this ensures that we first label the axis, which are required later to fit the cuboid
            droid_slam_labeled_tform_droid_slam = ref_sequences[
                ref_category_instance_ids[0]].droid_slam_labeled_tform_droid_slam.to(dtype=dtype, device=self.device)
            droid_slam_labeled_cuboid_tform_droid_slam_labeled = ref_sequences[
                ref_category_instance_ids[0]].droid_slam_labeled_cuboid_tform_droid_slam_labeled.to(dtype=dtype,
                                                                                                    device=self.device)
            if self.config.use_gt_src:
                droid_slam_labeled_tform_droid_slam = src_sequences[
                    src_category_instance_ids[0]].droid_slam_labeled_tform_droid_slam.to(dtype=dtype, device=self.device)
                droid_slam_labeled_cuboid_tform_droid_slam_labeled = src_sequences[
                    src_category_instance_ids[0]].droid_slam_labeled_cuboid_tform_droid_slam_labeled.to(dtype=dtype, device=self.device)


            results_diff_log_rot[category] = torch.zeros(
                size=(ref_instances_count_per_category[cat_id], src_instances_count_per_category[cat_id])).to(
                device=self.device, dtype=dtype)
            all_pred_ref_tform_src[category] = torch.zeros(
                size=(ref_instances_count_per_category[cat_id], src_instances_count_per_category[cat_id], 4, 4)).to(
                device=self.device, dtype=dtype)
            all_pred_pose_dist_geo[category] = torch.zeros(
                size=(ref_instances_count_per_category[cat_id], src_instances_count_per_category[cat_id])).to(
                device=self.device, dtype=dtype)
            all_pred_pose_dist_appear[category] = torch.zeros(
                size=(ref_instances_count_per_category[cat_id], src_instances_count_per_category[cat_id])).to(
                device=self.device, dtype=dtype)

        for i in range(self.config.global_optimization_steps):
            for cat_id, category in enumerate(categories):
                src_instance_ids = torch.LongTensor(list(range(src_instances_count)))
                ref_instance_ids = torch.LongTensor(list(range(ref_instances_count)))

                src_mesh_ids = src_instance_ids[src_map_seq_to_cat == cat_id]
                ref_mesh_ids = ref_instance_ids[ref_map_seq_to_cat == cat_id]

                for r, ref_mesh_id in enumerate(ref_mesh_ids):
                    for s, src_mesh_id in enumerate(src_mesh_ids):
                        src_vertices_mask = src_sequences_mesh_ids_for_verts == src_mesh_id
                        #src_vertices = torch.arange(src_vertices_count).to(device=self.device)[src_vertices_mask]
                        pts_src = src_meshes.verts[src_vertices_mask].clone()
                        if r > 0 and (self.config.use_only_first_reference or self.config.global_optimization_steps > 1):
                                pred_ref_tform_src = tform4x4(inv_tform4x4(all_pred_ref_tform_src[category][0, r]), all_pred_ref_tform_src[category][0, s])
                                all_pred_ref_tform_src[category][r, s] = pred_ref_tform_src
                                all_pred_pose_dist_geo[category][r, s] = all_pred_pose_dist_geo[category][0, r] + all_pred_pose_dist_geo[category][0, s]
                                all_pred_pose_dist_appear[category][r, s] = all_pred_pose_dist_appear[category][0, r] + all_pred_pose_dist_appear[category][0, s]
                                #logger.info(pred_ref_tform_src)
                        else:
                            if self.config.global_optimization_steps > 1 and i > 0:
                                #ref_vertices_mask = self.sequences_mesh_ids_for_verts == ref_mesh_id
                                #pts = self.meshes.verts.clone().detach()
                                #pts_ref = pts[ref_vertices_mask].clone()
                                from od3d.cv.geometry.transform import transf3d_broadcast, transf3d
                                # TODO: reference points from multiple point clouds have different scale and therefore problematic to fit with correspondences over multiple points
                                pts_ref = torch.cat([transf3d_broadcast(pts3d=ref_meshes.verts[ref_sequences_mesh_ids_for_verts == _ref_mesh_id].clone(), transf4x4=all_pred_ref_tform_src[category][0, _r]) for _r, _ref_mesh_id in enumerate(ref_mesh_ids)], dim=0)

                                dist_src_ref = torch.cat([src_sequences[src_mesh_id].get_dist_verts_mesh_feats_to_other_sequence(
                                    ref_sequences[_ref_mesh_id]).to(device=self.device, dtype=dtype) for _ref_mesh_id in ref_mesh_ids], dim=-1)

                                # division by two to normalize to 0. - 1.
                                dist_src_ref = dist_src_ref / 2.

                                if s != 0:
                                    # four points required, otherwise rotation yields an ambiguity. like planes without normals
                                    ref_tform4x4_src = ransac(pts=pts_src, fit_func=partial(fit_tform4x4, pts_ref=pts_ref,
                                                                                            dist_ref=dist_src_ref),
                                                              score_func=partial(score_tform4x4_fit, pts_ref=pts_ref,
                                                                                 dist_ref=dist_src_ref,
                                                                                 use_appear_argmin=self.config.use_appear_argmin,
                                                                                 dist_appear_weight=self.config.dist_appear_weight,
                                                                                 score_perc=self.config.ransac.score_perc),
                                                              fits_count=self.config.ransac.samples,
                                                              fit_pts_count=4)
                                else:
                                    ref_tform4x4_src = torch.eye(4).to(device=self.device, dtype=dtype)

                                pose_dist_geo, pose_dist_appear = score_tform4x4_fit(pts=pts_src,
                                                                                     tform4x4=ref_tform4x4_src[None,],
                                                                                     pts_ref=pts_ref,
                                                                                     dist_ref=dist_src_ref,
                                                                                     return_dists=True,
                                                                                     use_appear_argmin=self.config.use_appear_argmin,
                                                                                     dist_appear_weight=self.config.dist_appear_weight,
                                                                                     score_perc=self.config.ransac.score_perc)
                                all_pred_pose_dist_geo[category][r, s] = pose_dist_geo
                                all_pred_pose_dist_appear[category][r, s] = pose_dist_appear
                                pred_ref_tform_src = ref_tform4x4_src.clone()
                                #pred_ref_tform_src[:3, :3] /= torch.linalg.norm(pred_ref_tform_src[:3, :3], dim=-1,
                                #                                                keepdim=True)

                                all_pred_ref_tform_src[category][r, s] = pred_ref_tform_src
                            else:
                                ref_vertices_mask = ref_sequences_mesh_ids_for_verts == ref_mesh_id
                                pts_ref = ref_meshes.verts[ref_vertices_mask].clone()

                                logger.info(f'category: {category}, pts-src: {pts_src.shape}, pts-ref: {pts_ref.shape}')

                                dist_src_ref = src_sequences[src_mesh_id].get_dist_verts_mesh_feats_to_other_sequence(
                                    ref_sequences[ref_mesh_id]).to(device=self.device, dtype=dtype)

                                # division by two to normalize to 0. - 1.
                                dist_src_ref = dist_src_ref / 2.

                                if s != 0:
                                    # four points required, otherwise rotation yields an ambiguity. like planes without normals
                                    ref_tform4x4_src = ransac(pts=pts_src,
                                                              fit_func=partial(fit_tform4x4, pts_ref=pts_ref,
                                                                               dist_ref=dist_src_ref),
                                                              score_func=partial(score_tform4x4_fit, pts_ref=pts_ref,
                                                                                 dist_ref=dist_src_ref,
                                                                                 use_appear_argmin=self.config.use_appear_argmin,
                                                                                 dist_appear_weight=self.config.dist_appear_weight,
                                                                                 score_perc=self.config.ransac.score_perc),
                                                              fits_count=self.config.ransac.samples, fit_pts_count=4)
                                else:
                                    ref_tform4x4_src = torch.eye(4).to(device=self.device, dtype=dtype)
                                pose_dist_geo, pose_dist_appear = score_tform4x4_fit(pts=pts_src, tform4x4=ref_tform4x4_src[None,], pts_ref=pts_ref,
                                                                                     dist_ref=dist_src_ref, return_dists=True,
                                                                                     use_appear_argmin=self.config.use_appear_argmin,
                                                                                     dist_appear_weight=self.config.dist_appear_weight,
                                                                                     score_perc=self.config.ransac.score_perc)
                                all_pred_pose_dist_geo[category][r, s] = pose_dist_geo
                                all_pred_pose_dist_appear[category][r, s] = pose_dist_appear

                                pred_ref_tform_src = ref_tform4x4_src.clone()
                                #pred_ref_tform_src[:3, :3] /= torch.linalg.norm(pred_ref_tform_src[:3, :3], dim=-1, keepdim=True)
                                all_pred_ref_tform_src[category][r, s] = pred_ref_tform_src

                        if self.config.use_gt_src:
                            if dataset_src.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM_ZSP_LABELED:
                                gt_ref_tform_src = tform4x4(
                                    inv_tform4x4(ref_sequences[ref_mesh_id].co3dv1_zsp_obj_tform_droid_slam_obj.to(device=self.device, dtype=pred_ref_tform_src.dtype)),
                                   src_sequences[src_mesh_id].co3dv1_zsp_obj_tform_droid_slam_obj.to(device=self.device, dtype=pred_ref_tform_src.dtype))
                            elif dataset_src.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM_LABELED:
                                gt_ref_tform_src = tform4x4(
                                    inv_tform4x4(ref_sequences[ref_mesh_id].droid_slam_labeled_tform_droid_slam.to(
                                    device=self.device, dtype=pred_ref_tform_src.dtype)),
                                    src_sequences[src_mesh_id].droid_slam_labeled_tform_droid_slam.to(device=self.device, dtype=pred_ref_tform_src.dtype))
                            else:
                                gt_ref_tform_src = torch.eye(4).to(device=self.device)
                                logger.warning('No gt available ')
                            diff_rot_angle_rad = get_pose_diff_in_rad(pred_tform4x4=pred_ref_tform_src, gt_tform4x4=gt_ref_tform_src)
                            results_diff_log_rot[category][r, s] = diff_rot_angle_rad


                        #from od3d.cv.geometry.transform import transf3d_broadcast
                        #if r == 0:
                        #    verts = transf3d_broadcast(pts3d=self.meshes.get_verts_with_mesh_id(src_mesh_id), transf4x4=pred_ref_tform_src)
                        #    self.meshes.verts[src_vertices] = verts


        results = OD3D_Results()
        #results_ref = OD3D_Results()
        for cat_id, category in enumerate(categories):
            category_results = OD3D_Results()
            exclude_diagonal = dataset_src.name == dataset_ref.name

            if exclude_diagonal:
                # excluding diagonal entries as these are predicted transformation between same instance
                if self.config.use_gt_src:
                    category_results[f'rot_diff_rad'] = results_diff_log_rot[category][
                        torch.eye(src_instances_count_per_category[cat_id]).to(device=self.device) == 0].reshape(ref_instances_count_per_category[cat_id], src_instances_count_per_category[cat_id]-1)
                category_results[f'pose_sim_geo'] = 1.0 - all_pred_pose_dist_geo[category][
                    torch.eye(src_instances_count_per_category[cat_id]).to(device=self.device) == 0].reshape(ref_instances_count_per_category[cat_id], src_instances_count_per_category[cat_id]-1)
                category_results[f'pose_sim_appear'] = 1.0 - all_pred_pose_dist_appear[category][
                    torch.eye(src_instances_count_per_category[cat_id]).to(device=self.device) == 0].reshape(ref_instances_count_per_category[cat_id], src_instances_count_per_category[cat_id]-1)
            else:
                if self.config.use_gt_src:
                    category_results[f'rot_diff_rad'] = results_diff_log_rot[category]
                category_results[f'pose_sim_geo'] = 1.0 - all_pred_pose_dist_geo[category]
                category_results[f'pose_sim_appear'] = 1.0 - all_pred_pose_dist_appear[category]

            results += category_results.mean()
            category_results_mean = category_results.add_prefix(category)
            category_results_mean = category_results_mean.mean()
            category_results_mean.log()
            logger.info(category_results_mean)
            #if self.config.use_gt_src:
            #    category_results[f'rot_diff_rad'] = category_results[f'rot_diff_rad'][None,]
            #category_results[f'pose_sim_geo'] = category_results[f'pose_sim_geo'][None,]
            #category_results[f'pose_sim_appear'] = category_results[f'pose_sim_appear'][None,]
            #results += category_results_mean # category_results
            #
            # category_results_ref = OD3D_Results()
            # # excluding diagonal entries as these are predicted transformation between same instance
            # if self.config.use_gt_src:
            #     category_results_ref[f'rot_diff_rad'] = results_diff_log_rot[category][0, 1:]
            # category_results_ref[f'pose_sim_geo'] = 1.0 - all_pred_pose_dist_geo[category][0, 1:]
            # category_results_ref[f'pose_sim_appear'] = 1.0 - all_pred_pose_dist_appear[category][0, 1:]
            #
            # results_ref += category_results_ref
            # category_results_ref = category_results_ref.add_prefix(category)
            # category_results_ref_mean = category_results_ref.mean()
            # category_results_ref_mean.log_with_prefix(prefix=f'only_to_ref')

        results_mean = results.mean()
        #results_ref_mean = results_ref.mean()
        results_mean.log()
        logger.info(results_mean)
        #results_ref_mean.log_with_prefix(prefix='only_to_ref')

        from od3d.cv.geometry.transform import transf3d_broadcast
        from od3d.datasets.co3d.enum import PCL_SOURCES
        # if r == 0:
        #    verts = transf3d_broadcast(pts3d=self.meshes.get_verts_with_mesh_id(src_mesh_id), transf4x4=pred_ref_tform_src)
        #    self.meshes.verts[src_vertices] = verts

        ref_meshes.rgb = ref_meshes.get_verts_ncds_cat_with_mesh_ids()
        src_meshes.rgb = src_meshes.get_verts_ncds_cat_with_mesh_ids()

        ### VISUALIZATIONS
        ref_instance_ids = torch.LongTensor(list(range(ref_instances_count)))
        src_instance_ids = torch.LongTensor(list(range(src_instances_count)))

        for ref_instance_id_in_category in range(max(ref_instances_count_per_category)):
            aligned_name = f'{self.config.aligned_name}_r{ref_instance_id_in_category}'
            aligned_path = dataset_src.path_preprocess.joinpath('aligned', aligned_name)
            od3d.io.rm_dir(aligned_path)

        for cat_id, category in enumerate(categories):
            logger.info(f'category {category}')



            ref_category_instance_ids = ref_instance_ids[ref_map_seq_to_cat == cat_id]
            if self.config.use_only_first_reference:
                ref_category_instance_ids = ref_category_instance_ids[:1]

            src_category_instance_ids = src_instance_ids[src_map_seq_to_cat == cat_id]

            for ref_instance_id_in_category, ref_instance_id in enumerate(ref_category_instance_ids):
                aligned_name = f'{self.config.aligned_name}_r{ref_instance_id_in_category}'

                # geometry/appearance: 0.81/0.18 | 0.59/0.29 | 0.89/0.29 | 0.9/0.65 (best qualit.) | 0.9/0.55 | 0.9 / 0.6
                if self.config.use_gt_src:
                    rot_diff_rad = results_diff_log_rot[category][ref_instance_id_in_category, :]
                    accurate_pi6 = rot_diff_rad < (math.pi / 6.)
                    accurate_pi18 = rot_diff_rad < (math.pi / 18.)
                accurate_sim_geo = (1.0 - all_pred_pose_dist_geo[category][ref_instance_id_in_category, :]) > 0.90
                accurate_sim_appear = (1.0 - all_pred_pose_dist_appear[category][ref_instance_id_in_category, :]) > 0.60
                accurate_sim = accurate_sim_geo * accurate_sim_appear

                #if self.config.use_gt_src:
                if dataset_ref.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM_ZSP_LABELED:
                    droid_slam_labeled_cuboid_tform_droid_slam = inv_tform4x4(ref_sequences[ref_category_instance_ids[0]].co3dv1_zsp_obj_tform_droid_slam_obj.to(device=self.device, dtype=dtype))
                    droid_slam_labeled_cuboid = ref_sequences[ref_category_instance_ids[0]].droid_slam_labeled_cuboid
                else:
                    droid_slam_labeled_tform_droid_slam = ref_sequences[ref_instance_id].droid_slam_labeled_tform_droid_slam.to(dtype=dtype, device=self.device)
                    droid_slam_labeled_cuboid_tform_droid_slam_labeled = ref_sequences[ref_instance_id].droid_slam_labeled_cuboid_tform_droid_slam_labeled.to(dtype=dtype, device=self.device)
                    droid_slam_labeled_cuboid_tform_droid_slam = tform4x4(droid_slam_labeled_cuboid_tform_droid_slam_labeled, droid_slam_labeled_tform_droid_slam)
                    droid_slam_labeled_cuboid = ref_sequences[ref_instance_id].droid_slam_labeled_cuboid
                #else:
                #    droid_slam_labeled_cuboid_tform_droid_slam = torch.eye(4).to(device=self.device)

                ref_sequences[ref_instance_id].write_aligned_cuboid(aligned_name=aligned_name, cuboid=droid_slam_labeled_cuboid)

                pts3d = []
                pts3d_colors = []

                # pts3d.append(transf3d_broadcast(
                #     pts3d=ref_sequences[ref_category_instance_ids[0]].get_pcl(pcl_source=PCL_SOURCES.DROID_SLAM_CLEAN).to(
                #         device=self.device, dtype=dtype), transf4x4=droid_slam_labeled_cuboid_tform_droid_slam))
                #
                # pts3d_colors.append(
                #     ref_sequences[ref_category_instance_ids[0]].get_pcl_colors(pcl_source=PCL_SOURCES.DROID_SLAM_CLEAN).to(
                #         device=self.device, dtype=dtype))
                #
                # ref_vertices_mask = ref_sequences_mesh_ids_for_verts == ref_category_instance_ids[0]
                # ref_meshes.verts[ref_vertices_mask] = transf3d_broadcast(
                #     pts3d=ref_meshes.get_verts_with_mesh_id(ref_category_instance_ids[0]),
                #     transf4x4=droid_slam_labeled_cuboid_tform_droid_slam)

                src_meshes_cloned = src_meshes.get_meshes_with_ids(clone=True)
                for src_instance_id_in_category, src_instance_id in enumerate(src_category_instance_ids):
                    # prediction
                    droid_slam_labeled_cuboid_tform_droid_slam_instance = tform4x4(droid_slam_labeled_cuboid_tform_droid_slam, all_pred_ref_tform_src[category][ref_instance_id_in_category, src_instance_id_in_category]) # droid_slam_labeled_cuboid_tform_droid_slam
                    # ground truth
                    # droid_slam_labeled_cuboid_tform_droid_slam_instance = self.sequences[instance_id].zsp_labeled_cuboid_ref_tform_droid_slam_obj.to(device=self.device)

                    if accurate_sim[src_instance_id_in_category] or not self.config.aligned_store_only_similar:
                        src_sequences[src_instance_id].write_aligned_droid_slam_tform_droid_slam(aligned_name=aligned_name, aligned_droid_slam_tform_droid_slam=droid_slam_labeled_cuboid_tform_droid_slam_instance)

                    src_vertices_mask = src_sequences_mesh_ids_for_verts == src_instance_id
                    ref_vertices_mask = ref_sequences_mesh_ids_for_verts == ref_instance_id

                    dist_verts_ref = src_sequences[src_instance_id].get_dist_verts_mesh_feats_to_other_sequence(ref_sequences[ref_instance_id]).to(device=self.device, dtype=dtype)
                    dists_verts_min_ref_vertices = dist_verts_ref.min(dim=-1)[1]

                    src_meshes_cloned.verts[src_vertices_mask] = transf3d_broadcast(pts3d=src_meshes_cloned.get_verts_with_mesh_id(src_instance_id), transf4x4=droid_slam_labeled_cuboid_tform_droid_slam_instance)
                    src_meshes_cloned.rgb[src_vertices_mask] = ref_meshes.rgb[ref_vertices_mask][dists_verts_min_ref_vertices]

                    #co3d_src_tform_src = self.sequences_co3d_tform_droid_slam[instance_id]
                    #pts3d.append(transf3d_broadcast(pts3d=self.sequences[instance_id].pcl.to(device=self.device, dtype=dtype), transf4x4=tform4x4(all_pred_ref_tform_src[category][ref_instance_id_in_category, instance_id_in_category], inv_tform4x4(co3d_src_tform_src))))

                    pts3d.append(transf3d_broadcast(pts3d=src_sequences[src_instance_id].get_pcl(pcl_source=PCL_SOURCES.DROID_SLAM_CLEAN).to(device=self.device, dtype=dtype), transf4x4=droid_slam_labeled_cuboid_tform_droid_slam_instance))

                    pts3d_colors.append(src_sequences[src_instance_id].get_pcl_colors(pcl_source=PCL_SOURCES.DROID_SLAM_CLEAN).to(device=self.device, dtype=dtype))

                viewpoints_count = 2
                category_meshes = src_meshes_cloned.get_meshes_with_ids(meshes_ids=src_category_instance_ids)

                imgs = show_scene(pts3d=pts3d, pts3d_colors=pts3d_colors, return_visualization=True, viewpoints_count=viewpoints_count, meshes=category_meshes, device=self.device, meshes_add_translation=True, pts3d_add_translation=True)
                from od3d.cv.visual.draw import add_boolean_table
                if self.config.use_gt_src:
                    accurate_table = torch.stack([accurate_pi6, accurate_pi18, accurate_sim, accurate_sim_geo, accurate_sim_appear], dim=0)
                else:
                    accurate_table = torch.stack(
                        [accurate_sim, accurate_sim_geo, accurate_sim_appear], dim=0)
                from od3d.cv.visual.crop import crop_white_border_from_img
                for v in range(viewpoints_count):
                    img = crop_white_border_from_img(imgs[v])
                    img = add_boolean_table(img, table=accurate_table, text=['Label (PI/6)', 'Label (PI/18)', 'Sim.', 'Sim. Geo.', 'Sim. Appear.'])
                    results_visual = OD3D_Results()
                    results_visual[f'{category}'] = image_as_wandb_image(img, caption='blub')
                    results_visual.log_with_prefix('aligned')



    def test(self, dataset: OD3D_Dataset, config_inference: DictConfig = None):
        return OD3D_Results()

