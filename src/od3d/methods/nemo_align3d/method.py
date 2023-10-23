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

        dataset_train: CO3D = datasets_train['labeled']

        from od3d.cv.geometry.mesh import Meshes

        self.categories = dataset_train.categories
        self.sequences = dataset_train.get_sequences()


        # tform4x4(inv_tform4x4(src_frame.get_cam_tform4x4_obj(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.CO3D)), src_frame.get_cam_tform4x4_obj(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.DROID_SLAM))
        logger.info('loading mesh feats...')
        #self.sequences_mesh_feats = [seq.feats for seq in self.sequences]
        self.sequences_unique_names = [seq.name_unique for seq in self.sequences]
        self.map_seq_to_cat = torch.LongTensor([self.categories.index(name.split('/')[0]) for name in self.sequences_unique_names])
        self.categories_count = len(self.categories)
        self.instances_count_per_category = [(self.map_seq_to_cat == c).sum().item() for c in range(self.categories_count)]

        logger.info('loading meshes...')
        self.meshes = Meshes.load_from_meshes([seq.mesh for seq in self.sequences], device=self.device)

        # note: first get mesh to get DROID_SLAM tforms
        logger.info('getting co3d_tform_droid_slam for each instance...')
        self.sequences_co3d_tform_droid_slam = []
        for seq in self.sequences:
            self.sequences_co3d_tform_droid_slam.append(seq.get_a_src_tform_b_src(CAM_TFORM_OBJ_SOURCES.DROID_SLAM,
                                                                                  CAM_TFORM_OBJ_SOURCES.CO3DV1,
                                                                                  device=self.device))

        self.instances_count = len(self.meshes)
        #self.meshes_verts_aggregated_features = [vert_feats for mesh_feats in self.sequences_mesh_feats for vert_feats in mesh_feats]
        #dtype = self.meshes_verts_aggregated_features[0].dtype
        #vertices_count = len(self.meshes_verts_aggregated_features)
        dtype = self.meshes.verts.dtype
        vertices_count = len(self.meshes.verts)
        #self.sequences_meshes.rgb = self.sequences_meshes.get_verts_ncds_cat_with_mesh_ids()
        #self.sequences_meshes.show()

        self.sequences_mesh_ids_for_verts = self.meshes.get_mesh_ids_for_verts()

        results_diff_log_rot = {}
        all_pred_ref_tform_src = {}
        all_pred_pose_dist_geo = {}
        all_pred_pose_dist_appear = {}
        for cat_id, category in enumerate(self.categories):
            logger.info(f'category id {cat_id} name {category}')
            instance_ids = torch.LongTensor(list(range(self.instances_count)))
            category_instance_ids = instance_ids[self.map_seq_to_cat == cat_id]
            # this ensures that we first label the axis, which are required later to fit the cuboid
            droid_slam_labeled_tform_droid_slam = self.sequences[
                category_instance_ids[0]].droid_slam_labeled_tform_droid_slam.to(dtype=dtype, device=self.device)
            droid_slam_labeled_cuboid_tform_droid_slam_labeled = self.sequences[
                category_instance_ids[0]].droid_slam_labeled_cuboid_tform_droid_slam_labeled.to(dtype=dtype, device=self.device)

            results_diff_log_rot[category] = torch.zeros(
                size=(self.instances_count_per_category[cat_id], self.instances_count_per_category[cat_id])).to(
                device=self.device, dtype=dtype)
            all_pred_ref_tform_src[category] = torch.zeros(
                size=(self.instances_count_per_category[cat_id], self.instances_count_per_category[cat_id], 4, 4)).to(
                device=self.device, dtype=dtype)
            all_pred_pose_dist_geo[category] = torch.zeros(
                size=(self.instances_count_per_category[cat_id], self.instances_count_per_category[cat_id])).to(
                device=self.device, dtype=dtype)
            all_pred_pose_dist_appear[category] = torch.zeros(
                size=(self.instances_count_per_category[cat_id], self.instances_count_per_category[cat_id])).to(
                device=self.device, dtype=dtype)

        for i in range(self.config.global_optimization_steps):
            for cat_id, category in enumerate(self.categories):
                instance_ids = torch.LongTensor(list(range(self.instances_count)))

                ref_mesh_ids = instance_ids[self.map_seq_to_cat == cat_id]
                src_mesh_ids = instance_ids[self.map_seq_to_cat == cat_id]

                for r, ref_mesh_id in enumerate(ref_mesh_ids):
                    #ref_mesh_id = 3 # 0, 1, 2, 3, 4
                    # calculate reference vertex/feature given dists: nearest-neighbor, k-nearest-neighbor, average feature -
                    ref_vertices_mask = self.sequences_mesh_ids_for_verts == ref_mesh_id

                    ref_vertices = torch.arange(vertices_count).to(device=self.device)[ref_vertices_mask]

                    pts = self.meshes.verts.clone().detach()

                    #dist_ref = dists_appearance_verts[:, ref_vertices]
                    pts_ref = pts[ref_vertices_mask].clone()

                    rgbs_all = self.meshes.get_verts_ncds_cat_with_mesh_ids()
                    #rgbs_all[~ref_vertices_mask] = rgbs_all[ref_vertices_mask][dists_verts_min_ref_vertices[~ref_vertices_mask]]
                    self.meshes.rgb = rgbs_all

                    for s, src_mesh_id in enumerate(src_mesh_ids):
                        src_vertices_mask = self.sequences_mesh_ids_for_verts == src_mesh_id
                        src_vertices = torch.arange(vertices_count).to(device=self.device)[src_vertices_mask]
                        pts_src = pts[src_vertices]


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
                                pts_ref = torch.cat([transf3d_broadcast(pts3d=pts[self.sequences_mesh_ids_for_verts == _ref_mesh_id].clone(), transf4x4=all_pred_ref_tform_src[category][0, _r]) for _r, _ref_mesh_id in enumerate(ref_mesh_ids)], dim=0)

                                dist_src_ref = torch.cat([self.sequences[src_mesh_id].get_dist_verts_mesh_feats_to_other_sequence(
                                    self.sequences[_ref_mesh_id]) for _ref_mesh_id in ref_mesh_ids], dim=-1)

                                # remove infinities is highly important
                                # dist_src_ref[~dist_src_ref.isfinite()] = dist_src_ref[dist_src_ref.isfinite()].max()
                                # division by two to normalize to 0. - 1.
                                dist_src_ref = dist_src_ref / 2.

                                if s != 0:
                                    # four points required, otherwise rotation yields an ambiguity. like planes without normals
                                    ref_tform4x4_src = ransac(pts=pts_src, fit_func=partial(fit_tform4x4, pts_ref=pts_ref,
                                                                                            dist_ref=dist_src_ref),
                                                              score_func=partial(score_tform4x4_fit, pts_ref=pts_ref,
                                                                                 dist_ref=dist_src_ref, use_appear_argmin=self.config.use_appear_argmin,
                                                                                 dist_appear_weight=self.config.dist_appear_weight), fits_count=500,
                                                              fit_pts_count=4)
                                else:
                                    ref_tform4x4_src = torch.eye(4).to(device=self.device, dtype=dtype)

                                pose_dist_geo, pose_dist_appear = score_tform4x4_fit(pts=pts_src,
                                                                                     tform4x4=ref_tform4x4_src[None,],
                                                                                     pts_ref=pts_ref,
                                                                                     dist_ref=dist_src_ref,
                                                                                     return_dists=True,
                                                                                     use_appear_argmin=self.config.use_appear_argmin,
                                                                                     dist_appear_weight=self.config.dist_appear_weight)
                                all_pred_pose_dist_geo[category][r, s] = pose_dist_geo
                                all_pred_pose_dist_appear[category][r, s] = pose_dist_appear
                                pred_ref_tform_src = ref_tform4x4_src.clone()
                                #pred_ref_tform_src[:3, :3] /= torch.linalg.norm(pred_ref_tform_src[:3, :3], dim=-1,
                                #                                                keepdim=True)

                                all_pred_ref_tform_src[category][r, s] = pred_ref_tform_src
                            else:
                                logger.info(f'category: {category}, pts-src: {pts_src.shape}, pts-ref: {pts_ref.shape}')

                                dist_src_ref = self.sequences[src_mesh_id].get_dist_verts_mesh_feats_to_other_sequence(
                                    self.sequences[ref_mesh_id])
                                # remove infinities is highly important
                                #dist_src_ref[~dist_src_ref.isfinite()] = dist_src_ref[dist_src_ref.isfinite()].max()
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
                                                                                 dist_appear_weight=self.config.dist_appear_weight),
                                                              fits_count=2000, fit_pts_count=4)
                                else:
                                    ref_tform4x4_src = torch.eye(4).to(device=self.device, dtype=dtype)
                                pose_dist_geo, pose_dist_appear = score_tform4x4_fit(pts=pts_src, tform4x4=ref_tform4x4_src[None,], pts_ref=pts_ref,
                                                                                     dist_ref=dist_src_ref, return_dists=True,
                                                                                     use_appear_argmin=self.config.use_appear_argmin,
                                                                                     dist_appear_weight=self.config.dist_appear_weight)
                                all_pred_pose_dist_geo[category][r, s] = pose_dist_geo
                                all_pred_pose_dist_appear[category][r, s] = pose_dist_appear

                                pred_ref_tform_src = ref_tform4x4_src.clone()
                                #pred_ref_tform_src[:3, :3] /= torch.linalg.norm(pred_ref_tform_src[:3, :3], dim=-1, keepdim=True)
                                all_pred_ref_tform_src[category][r, s] = pred_ref_tform_src

                        if self.config.use_gt:
                            gt_ref_tform_src = tform4x4(
                               inv_tform4x4(self.sequences[ref_mesh_id].co3dv1_zsp_obj_tform_droid_slam_obj.to(device=self.device, dtype=pred_ref_tform_src.dtype)),
                               self.sequences[src_mesh_id].co3dv1_zsp_obj_tform_droid_slam_obj.to(device=self.device, dtype=pred_ref_tform_src.dtype))


                            from od3d.cv.metric.pose import get_pose_diff_in_rad
                            diff_rot_angle_rad = get_pose_diff_in_rad(pred_tform4x4=pred_ref_tform_src, gt_tform4x4=gt_ref_tform_src)
                            results_diff_log_rot[category][r, s] = diff_rot_angle_rad


                        #from od3d.cv.geometry.transform import transf3d_broadcast
                        #if r == 0:
                        #    verts = transf3d_broadcast(pts3d=self.meshes.get_verts_with_mesh_id(src_mesh_id), transf4x4=pred_ref_tform_src)
                        #    self.meshes.verts[src_vertices] = verts

        results = OD3D_Results()
        results_ref = OD3D_Results()
        for cat_id, category in enumerate(self.categories):
            category_results = OD3D_Results()
            # excluding diagonal entries as these are predicted transformation between same instance
            if self.config.use_gt:
                category_results[f'rot_diff_rad'] = results_diff_log_rot[category][
                    torch.eye(self.instances_count_per_category[cat_id]).to(device=self.device) == 0]
            category_results[f'pose_sim_geo'] = 1.0 - all_pred_pose_dist_geo[category][
                torch.eye(self.instances_count_per_category[cat_id]).to(device=self.device) == 0]
            category_results[f'pose_sim_appear'] = 1.0 - all_pred_pose_dist_appear[category][
                torch.eye(self.instances_count_per_category[cat_id]).to(device=self.device) == 0]

            results += category_results
            category_results = category_results.add_prefix(category)
            category_results_mean = category_results.mean()
            category_results_mean.log()


            category_results_ref = OD3D_Results()
            # excluding diagonal entries as these are predicted transformation between same instance
            if self.config.use_gt:
                category_results_ref[f'rot_diff_rad'] = results_diff_log_rot[category][0, 1:]
            category_results_ref[f'pose_sim_geo'] = 1.0 - all_pred_pose_dist_geo[category][0, 1:]
            category_results_ref[f'pose_sim_appear'] = 1.0 - all_pred_pose_dist_appear[category][0, 1:]

            results_ref += category_results_ref
            category_results_ref = category_results_ref.add_prefix(category)
            category_results_ref_mean = category_results_ref.mean()
            category_results_ref_mean.log_with_prefix(prefix=f'only_to_ref')

        results_mean = results.mean()
        results_ref_mean = results_ref.mean()
        results_mean.log()
        results_ref_mean.log_with_prefix(prefix='only_to_ref')

        from od3d.cv.geometry.transform import transf3d_broadcast
        from od3d.datasets.co3d.enum import PCL_SOURCES
        # if r == 0:
        #    verts = transf3d_broadcast(pts3d=self.meshes.get_verts_with_mesh_id(src_mesh_id), transf4x4=pred_ref_tform_src)
        #    self.meshes.verts[src_vertices] = verts
        for cat_id, category in enumerate(self.categories):
            logger.info(f'category {category}')

            # geometry/appearance: 0.81/0.18 | 0.59/0.29 | 0.89/0.29 | 0.9/0.65 (best qualit.) | 0.9/0.55 | 0.9 / 0.6
            rot_diff_rad = results_diff_log_rot[category][0, :]
            accurate_pi6 = rot_diff_rad < (math.pi / 6.)
            accurate_pi18 = rot_diff_rad < (math.pi / 18.)
            accurate_sim_geo = (1.0 - all_pred_pose_dist_geo[category][0, :]) > 0.90
            accurate_sim_appear = (1.0 - all_pred_pose_dist_appear[category][0, :]) > 0.60
            accurate_sim = accurate_sim_geo * accurate_sim_appear

            ### VISUALIZATIONS
            instance_ids = torch.LongTensor(list(range(self.instances_count)))
            category_instance_ids = instance_ids[self.map_seq_to_cat == cat_id]
            droid_slam_labeled_tform_droid_slam = self.sequences[category_instance_ids[0]].droid_slam_labeled_tform_droid_slam.to(dtype=dtype, device=self.device)
            droid_slam_labeled_cuboid_tform_droid_slam_labeled = self.sequences[category_instance_ids[0]].droid_slam_labeled_cuboid_tform_droid_slam_labeled.to(dtype=dtype, device=self.device)
            droid_slam_labeled_cuboid_tform_droid_slam = tform4x4(droid_slam_labeled_cuboid_tform_droid_slam_labeled, droid_slam_labeled_tform_droid_slam)
            self.sequences[category_instance_ids[0]].write_aligned_cuboid(aligned_name=self.config.aligned_name, cuboid=self.sequences[category_instance_ids[0]].droid_slam_labeled_cuboid)

            pts3d = []
            pts3d_colors = []
            for instance_id_in_category, instance_id in enumerate(category_instance_ids):
                # prediction
                droid_slam_labeled_cuboid_tform_droid_slam_instance = tform4x4(droid_slam_labeled_cuboid_tform_droid_slam, all_pred_ref_tform_src[category][0, instance_id_in_category]) # droid_slam_labeled_cuboid_tform_droid_slam
                # ground truth
                # droid_slam_labeled_cuboid_tform_droid_slam_instance = self.sequences[instance_id].zsp_labeled_cuboid_ref_tform_droid_slam_obj.to(device=self.device)

                if accurate_sim[instance_id_in_category] or not self.config.aligned_store_only_similar:
                    self.sequences[instance_id].write_aligned_droid_slam_tform_droid_slam(aligned_name=self.config.aligned_name, aligned_droid_slam_tform_droid_slam=droid_slam_labeled_cuboid_tform_droid_slam_instance)

                vertices_mask = self.sequences_mesh_ids_for_verts == instance_id
                ref_vertices_mask = self.sequences_mesh_ids_for_verts == category_instance_ids[0]

                dist_verts_ref = self.sequences[instance_id].get_dist_verts_mesh_feats_to_other_sequence(self.sequences[category_instance_ids[0]])
                dists_verts_min_ref_vertices = dist_verts_ref.min(dim=-1)[1]
                self.meshes.verts[vertices_mask] = transf3d_broadcast(pts3d=self.meshes.get_verts_with_mesh_id(instance_id), transf4x4=droid_slam_labeled_cuboid_tform_droid_slam_instance)
                self.meshes.rgb[vertices_mask] = self.meshes.rgb[ref_vertices_mask][dists_verts_min_ref_vertices]

                #co3d_src_tform_src = self.sequences_co3d_tform_droid_slam[instance_id]
                #pts3d.append(transf3d_broadcast(pts3d=self.sequences[instance_id].pcl.to(device=self.device, dtype=dtype), transf4x4=tform4x4(all_pred_ref_tform_src[category][0, instance_id_in_category], inv_tform4x4(co3d_src_tform_src))))

                pts3d.append(transf3d_broadcast(pts3d=self.sequences[instance_id].get_pcl(pcl_source=PCL_SOURCES.DROID_SLAM_CLEAN).to(device=self.device, dtype=dtype), transf4x4=droid_slam_labeled_cuboid_tform_droid_slam_instance))

                pts3d_colors.append(self.sequences[instance_id].get_pcl_colors(pcl_source=PCL_SOURCES.DROID_SLAM_CLEAN).to(device=self.device, dtype=dtype))

            viewpoints_count = 2
            category_meshes = self.meshes.get_meshes_with_ids(meshes_ids=category_instance_ids)

            imgs = show_scene(pts3d=pts3d, pts3d_colors=pts3d_colors, return_visualization=True, viewpoints_count=viewpoints_count, meshes=category_meshes, device=self.device, meshes_add_translation=True, pts3d_add_translation=True)
            from od3d.cv.visual.draw import add_boolean_table
            accurate_table = torch.stack([accurate_pi6, accurate_pi18, accurate_sim, accurate_sim_geo, accurate_sim_appear], dim=0)
            from od3d.cv.visual.crop import crop_white_border_from_img
            for v in range(viewpoints_count):
                img = crop_white_border_from_img(imgs[v])
                img = add_boolean_table(img, table=accurate_table, text=['Label (PI/6)', 'Label (PI/18)', 'Sim.', 'Sim. Geo.', 'Sim. Appear.'])
                results_visual = OD3D_Results()
                results_visual[f'{category}'] = image_as_wandb_image(img, caption='blub')
                results_visual.log_with_prefix('aligned')

            # #rgbs_all[~ref_vertices_mask] = rgbs_all[ref_vertices_mask][dists_verts_min_ref_vertices[~ref_vertices_mask]]
            # rgbs_all = self.meshes.get_verts_ncds_cat_with_mesh_ids()
            # from od3d.cv.visual.show import show_scene
            # show_scene(meshes=Meshes.load_from_meshes([self.sequences[ref_mesh_id].mesh], device=self.device))
            # # rgbs_all[~ref_vertices_mask] = rgbs_all[ref_vertices_mask][dists_verts_min_ref_vertices[~ref_vertices_mask]]
            # self.meshes.rgb = rgbs_all
            #self.meshes.show()

            """
            logger.info('tnse plot')
            from sklearn.manifold import TSNE
            all_features_tsne = TSNE().fit_transform(all_features.detach().cpu().numpy())
    
            from od3d.cv.visual.show import get_colors
            import matplotlib.pyplot as plt
            plt.switch_backend('Agg')
            plt.switch_backend('TKAgg')
    
            colors = get_colors(self.instances_count)
            pts_colors = torch.stack([colors[all_features_instances_ids[i].item()] for i in range(len(all_features_instances_ids))], dim=0)
            fig, axs = plt.subplots(nrows=2, figsize=(3, 3), facecolor="white", constrained_layout=True)
            axs[0].scatter(all_features_tsne[:, 0], all_features_tsne[:, 1], c=pts_colors.detach().cpu().numpy(), s=50, alpha=0.8)
    
            #first_instance_features_count =
            #all_features_vertices_ids[]
            colors = get_colors(vertices_count)
            pts_colors = torch.stack([colors[all_features_vertices_ids[i].item()] for i in range(len(all_features_vertices_ids))], dim=0)
            #fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
            axs[1].scatter(all_features_tsne[:, 0], all_features_tsne[:, 1], c=pts_colors.detach().cpu().numpy(), s=50, alpha=0.8)
            plt.show()
            """
            # fix dinov2 features with correct transformation



    def test(self, dataset: OD3D_Dataset, config_inference: DictConfig = None):
        return OD3D_Results()
        #
        # logger.info(f'test dataset {dataset.name}')
        # if config_inference is None:
        #     config_inference = self.config.inference
        # self.net.eval()
        # dataset.transform = self.transform_test
        #
        # dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.config.test.dataloader.batch_size,
        #                                          shuffle=False,
        #                                          collate_fn=dataset.collate_fn,
        #                                          num_workers=self.config.test.dataloader.num_workers,
        #                                          pin_memory=self.config.test.dataloader.pin_memory)
        #
        # logger.info(f"Dataset contains {len(dataset)} frames.")
        #
        # results_epoch = OD3D_Results()
        # for i, batch in tqdm(enumerate(iter(dataloader))):
        #     batch.to(device=self.device)
        #
        #     results_batch = self.inference_batch(batch=batch)
        #     results_epoch += results_batch
        #
        # count_pred_frames = len(results_epoch['item_id'])
        # logger.info(f'Predicted {count_pred_frames} frames.')
        #
        # results_visual = self.get_results_visual(results_epoch=results_epoch, dataset=dataset,
        #                                          config_visualize=self.config.test.visualize)
        # results_epoch = results_epoch.mean()
        # results_epoch += results_visual
        # return results_epoch


    def train_epoch(self, dataset: OD3D_Dataset) -> OD3D_Results:
        self.net.train()
        dataset.transform = self.transform_train
        dataloader_train = torch.utils.data.DataLoader(dataset=dataset,
                                                       batch_size=self.config.train.dataloader.batch_size,
                                                       shuffle=False,
                                                       collate_fn=dataset.collate_fn,
                                                       num_workers=self.config.train.dataloader.num_workers,
                                                       pin_memory=self.config.train.dataloader.pin_memory)

        results_epoch = OD3D_Results()
        accumulate_steps = 0
        for i, batch in tqdm(enumerate(iter(dataloader_train))):
            results_batch: OD3D_Results = self.train_batch(batch=batch)
            results_batch.log_with_prefix('train')
            results_epoch += results_batch

        results_visual = self.get_results_visual(results_epoch=results_epoch, dataset=dataset,
                                                 config_visualize=self.config.train.visualize)
        results_epoch = results_epoch.mean()
        results_epoch += results_visual
        return results_epoch


    def train_batch(self, batch) -> OD3D_Results:
        results_batch = OD3D_Results()

        batch.to(device=self.device)

        batch.cam_tform4x4_obj = batch.cam_tform4x4_obj.detach()

        # logger.info(f'batch.label {batch.label}')
        # B x x N x 2

        batch_sequences_ids = [ self.sequences_unique_names.index('/'.join(name_unique.split('/')[:-1])) for name_unique in batch.name_unique]

        vts2d, vts2d_mask = self.meshes.verts2d(cams_intr4x4=batch.cam_intr4x4,
                                                cams_tform4x4_obj=batch.cam_tform4x4_obj,
                                                imgs_sizes=batch.size, mesh_ids=batch_sequences_ids,
                                                down_sample_rate=self.down_sample_rate)

        N = vts2d.shape[1]

        # B x C x H x W
        feats2d_net = self.net(batch.rgb)
        feats2d_net_mask = resize(batch.mask_rgb, H_out=feats2d_net.shape[2], W_out=feats2d_net.shape[3])
        if self.config.train.use_mask_object:
            feats2d_net_mask = feats2d_net_mask * 1. * resize(batch.mask, H_out=feats2d_net.shape[2],
                                                              W_out=feats2d_net.shape[3])

        H, W = feats2d_net.shape[-2:]
        xy = torch.stack(
            torch.meshgrid(torch.arange(W, device=self.device), torch.arange(H, device=self.device),
                           indexing='xy'), dim=0)  # HxW
        prob_noise = (1. - 1. * resize(feats2d_net_mask, scale_factor=1. / self.down_sample_rate)).flatten(1)
        prob_noise[prob_noise.sum(dim=-1) <= 0.] = 1.
        if self.config.num_noise > 0:
            noise2d = xy.flatten(1)[:, torch.multinomial(prob_noise, self.config.num_noise)].permute(1, 2, 0)
        else:
            noise2d = torch.ones(size=(vts2d.shape[0], 0, 2), device=self.device)
        vts2d_feats2d_net_mask = sample_pxl2d_pts(feats2d_net_mask, pxl2d=torch.cat([vts2d], dim=1))
        vts2d_mask = vts2d_mask * (vts2d_feats2d_net_mask[:, :, 0] > 0.5)

        # B x F+N x C
        net_feats = sample_pxl2d_pts(feats2d_net, pxl2d=torch.cat([vts2d, noise2d], dim=1))

        C = net_feats.shape[2]
        # args: X: Bx3xHxW, keypoint_positions: BxNx2, obj_mask: BxHxW ensures that noise is sampled outside of object mask
        # returns: BxF+NxC


        # net_feats = net_feats[:, :].reshape(-1, net_feats.shape[-1])
        batch_vts_ids = self.meshes.get_verts_and_noise_ids_stacked(batch_sequences_ids,
                                                                    count_noise_ids=self.config.num_noise)

        # weighting with similarity score
        # net_feats = net_feats * (batch.cam_tform4x4_obj_sim[:, None, None] ** 4)

        # sim_weight = batch.cam_tform4x4_obj_sim[:, None].expand(*net_feats.shape[:2])
        # sim_weight = torch.cat([sim_weight[:, :N][mask_vts2d_vsbl], sim_weight[:, N:].reshape(-1)], dim=0)

        # N,
        batch_vts_ids = torch.cat([batch_vts_ids[:, :N][vts2d_mask], batch_vts_ids[:, N:].reshape(-1)],
                                  dim=0)

        # N x C
        net_feats = torch.cat([net_feats[:, :N][vts2d_mask], net_feats[:, N:].reshape(-1, C)], dim=0)

        for b, vertex_id in enumerate(batch_vts_ids):
            self.meshes_verts_aggregated_features[vertex_id] = torch.cat([net_feats[b:b+1], self.meshes_verts_aggregated_features[vertex_id]], dim=0)
        # 1. add tsne and pca plot ( for all instances with coloring instances, for one instance with coloring vertices, for one instance with vertices coloring with pca/tsne color (avg) )
        # 2. use dino (without head)
        # 3. accumulate features

        # batch_vts_ids = self.meshes.get_feats_ids_stacked(batch.label.tolist())

        #bank_feats = torch.cat([self.meshes.feats, self.clutter_feats], dim=0)
        # if self.config.train.bank_feats_update == 'loss_gradient':
        #     sim = torch.einsum('nc,vc->nv', net_feats, bank_feats)
        # elif self.config.train.bank_feats_update == 'normalize_loss_gradient':
        #     sim = torch.einsum('nc,vc->nv', net_feats, torch.nn.functional.normalize(bank_feats, dim=1))
        # elif self.config.train.bank_feats_update == 'moving_average':
        #     sim = torch.einsum('nc,vc->nv', net_feats, bank_feats.detach())
        #     bank_feats_new = self.config.train.alpha * bank_feats[batch_vts_ids].detach() + (1. - self.config.train.alpha) * net_feats.detach()
        #     batch_vts_ids_unique, batch_vts_ids_unique_inverse, batch_vts_ids_unique_counts = batch_vts_ids.unique(return_inverse=True, return_counts=True)
        #     bank_feats_new = torch.einsum('nk,nc->kc', torch.nn.functional.one_hot(batch_vts_ids_unique_inverse).to(dtype= bank_feats_new.dtype, device= bank_feats_new.device), bank_feats_new) / batch_vts_ids_unique_counts[:, None]
        #     bank_feats[batch_vts_ids_unique].data = bank_feats_new
        # else:
        #     logger.error(f'unknown bank_feats_update: {self.config.train.bank_feats_update}')
        #     sim = None
        #
        # sim_batchwise_borders = torch.cat([torch.LongTensor([0]).to(device=vts2d_mask.device), vts2d_mask.sum(dim=1).cumsum(dim=0)], dim=0)
        # sim_batchwise = torch.stack([sim[sim_batchwise_borders[b]:sim_batchwise_borders[b+1]].max(dim=-1)[0].mean() for b in range(len(sim_batchwise_borders)-1)], dim=0)
        # # in case there are 0 vertices inside one image
        # sim_batchwise[sim_batchwise.isnan()] = 0.
        results_batch['sim'] = torch.zeros_like(batch.cam_tform4x4_obj[:, 0, 0])
        #
        # # loss: cross_entropy  # cross_entropy, nll_softmax, nll_clip, nll_affine_to_prob
        # # bank_feats_update: loss_gradient  # loss_gradient, normalize_loss_gradient, moving_average, loss
        # loss = self.criterion(sim / self.config.train.T, batch_vts_ids)
        #
        # loss.backward()
        # logger.info(f'loss {loss.item()}')
        #
        results_batch['loss'] = torch.zeros_like(batch.cam_tform4x4_obj[:, 0, 0])
        results_batch['item_id'] = batch.item_id
        results_batch['name_unique'] = batch.name_unique
        results_batch['gt_cam_tform4x4_obj'] = batch.cam_tform4x4_obj

        return results_batch

    """
    def get_sim_cam_tform4x4_obj(self, batch, cam_intr4x4, cam_tform4x4_obj, broadcast_batch_and_cams=False):
        with torch.no_grad():
            net_feats2d = self.net(batch.rgb)
            mesh_feats2d_rendered = self.meshes.render_feats(cams_tform4x4_obj=cam_tform4x4_obj,
                                                             cams_intr4x4=cam_intr4x4,
                                                             imgs_sizes=batch.size, meshes_ids=batch.label,
                                                             down_sample_rate=self.down_sample_rate,
                                                             broadcast_batch_and_cams=broadcast_batch_and_cams)

            sim = self.get_sim_feats2d_net_and_rendered(feats2d_net=net_feats2d, feats2d_rendered=mesh_feats2d_rendered)
        return sim
    """


    def inference_batch(self, batch, return_samples_with_sim=True):
        results = OD3D_Results()
        B = len(batch)

        """
        # these parameters are used in prev. version
        batch.cam_tform4x4_obj[:, 2, 3] = 5. * 6. # 5. * 6.
        batch.cam_tform4x4_obj[:, 0, 3] = 0.
        batch.cam_tform4x4_obj[:, 1, 3] = 0.
        batch.cam_intr4x4[:, 0, 0] = 3000.
        batch.cam_intr4x4[:, 1, 1] = 3000.
        batch.cam_intr4x4[:, 0, 2] = batch.size[1] / 2.
        batch.cam_intr4x4[:, 1, 2] = batch.size[0] / 2.
        """


        time_loaded = time.time()
        with torch.no_grad():
            feats2d_net = self.net(batch.rgb)
            feats2d_net_mask = resize(batch.mask_rgb, H_out=feats2d_net.shape[2], W_out=feats2d_net.shape[3])
            if self.config.inference.use_mask_object:
                feats2d_net_mask = feats2d_net_mask * 1. * resize(batch.mask, H_out=feats2d_net.shape[2], W_out=feats2d_net.shape[3])

            time_pred_net_feats2d = time.time()
            # logger.info(
            #    f"predicted net feats2d, took {(time_pred_net_feats2d - time_loaded):.3f}")
            results['time_feats2d'] = torch.Tensor([time_pred_net_feats2d - time_loaded,]) / B

            meshes_scores = []
            for mesh_id in range(len(self.meshes)):
                # logger.info(f'calc score for mesh {self.config.classes[mesh_id]}')
                bank_feats = torch.cat([self.meshes.get_feats_with_mesh_id(mesh_id), self.clutter_feats.detach()],
                                       dim=0)
                # inner_feats2d_net_bank_vts_max_vals = torch.sum(net_feats2d[:, None] * bank_feats[None, :, :, None, None], dim=2, keepdim=True).max(dim=1).values
                out_shape = feats2d_net.shape[:1] + torch.Size([1]) + feats2d_net.shape[2:]
                inner_feats2d_net_bank_vts_max_vals = torch.einsum('bchw,kc->bkhw', feats2d_net, bank_feats).max(dim=1,
                                                                                                                 keepdim=True).values
                # inner_feats2d_net_bank_vts_max_vals, inner_feats2d_net_bank_vts_max_ids = inner_feats2d.max(dim=1)
                # show_img(self.meshes.get_verts_with_mesh_id[mesh_id][inner_feats2d_net_bank_vts_max_ids[0, 0]].permute(2, 0, 1), normalize=True)
                # show_img(inner_feats2d_net_bank_vts_max_vals[0])
                mesh_score = inner_feats2d_net_bank_vts_max_vals.flatten(1).mean(dim=1)
                # clutter_score = inner_feats2d[:, -clutter_feats.shape[0]:].mean(dim=1).flatten(1).mean(dim=1)
                # mesh_score -= clutter_score
                meshes_scores.append(mesh_score)
            meshes_scores = torch.stack(meshes_scores, dim=-1)
            pred_class_scores, pred_class_ids = meshes_scores.max(dim=1)

            # logger.info(f'pred class ids {pred_class_ids}')
            time_pred_class = time.time()
            # logger.info(f"predicted class: {self.config.classes[int(pred_class_ids[0])]}, took {(time_pred_class - time_pred_net_feats2d):.3f}")

            results['time_class'] = torch.Tensor([time_pred_class - time_pred_net_feats2d,]) / B

            b_cams_multiview_tform4x4_obj, b_cams_multiview_intr4x4 = self.get_samples(config_sample=self.config.inference.sample,
                                                                                       cam_intr4x4=batch.cam_intr4x4,
                                                                                       cam_tform4x4_obj=batch.cam_tform4x4_obj,
                                                                                       feats2d_net=feats2d_net,
                                                                                       categories_ids=batch.label,
                                                                                       feats2d_net_mask=feats2d_net_mask)
            #  OPTION A: Use 2d gradient of rendered features
            sim = self.get_sim_feats2d_net_with_cams(
                feats2d_net=feats2d_net,
                feats2d_net_mask=feats2d_net_mask,
                cam_tform4x4_obj=b_cams_multiview_tform4x4_obj,
                cam_intr4x4=b_cams_multiview_intr4x4,
                categories_ids=batch.label,
                broadcast_batch_and_cams=True
            )

            if return_samples_with_sim:
                results['samples_cam_tform4x4_obj'] = b_cams_multiview_tform4x4_obj
                results['samples_cam_intr4x4'] = b_cams_multiview_intr4x4
                results['samples_sim'] = sim

            mesh_multiple_cams_loss = -sim


            mesh_cam_loss_min_val, mesh_cam_loss_min_id = mesh_multiple_cams_loss.min(dim=1)

            cam_tform4x4_obj = b_cams_multiview_tform4x4_obj[:, mesh_cam_loss_min_id].permute(2, 3, 0, 1).diagonal(
                dim1=-2, dim2=-1).permute(2, 0, 1)

        if self.config.inference.refine.enabled:
            obj_tform6_tmp = torch.nn.Parameter(torch.zeros(size=(B, 6)).to(device=cam_tform4x4_obj.device),
                                                requires_grad=True)

            optim_inference = torch.optim.Adam(
                params=[obj_tform6_tmp],
                lr=self.config.inference.optimizer.lr,
                betas=(self.config.inference.optimizer.beta0, self.config.inference.optimizer.beta1),
            )

            time_before_pose_iterative = time.time()
            cam_tform4x4_obj = tform4x4(cam_tform4x4_obj.detach(), se3_exp_map(obj_tform6_tmp))

            for epoch in range(self.config.inference.optimizer.epochs):
                obj_tform6_tmp.data[:, self.config.inference.refine.dims_detached] = 0.
                cam_tform4x4_obj = tform4x4(cam_tform4x4_obj.detach(), se3_exp_map(obj_tform6_tmp.detach()))
                obj_tform6_tmp.data[:, :] = 0.
                cam_tform4x4_obj = tform4x4(cam_tform4x4_obj.detach(), se3_exp_map(obj_tform6_tmp))

                sim, sim_pxl = self.get_sim_feats2d_net_with_cams(feats2d_net=feats2d_net,
                                                                  feats2d_net_mask=feats2d_net_mask,
                                                                  cam_tform4x4_obj=cam_tform4x4_obj,
                                                                  cam_intr4x4=batch.cam_intr4x4,
                                                                  categories_ids=batch.label, return_sim_pxl=True,
                                                                  broadcast_batch_and_cams=False,
                                                                  pre_rendered=False)
                mesh_cam_loss = -sim

                if self.config.inference.live:
                    show_img(
                        blend_rgb(batch.rgb[0], (self.meshes.render_feats(cams_tform4x4_obj=cam_tform4x4_obj[0:0 + 1],
                                                                          cams_intr4x4=batch.cam_intr4x4[0:0 + 1],
                                                                          imgs_sizes=batch.size,
                                                                          meshes_ids=batch.label[0:0 + 1],
                                                                          modality=MESH_RENDER_MODALITIES.VERTS_NCDS)[
                            0]).to(dtype=batch.rgb.dtype)), duration=1)

                loss = mesh_cam_loss.mean()
                loss.backward()
                optim_inference.step()
                optim_inference.zero_grad()

            cam_tform4x4_obj = tform4x4(cam_tform4x4_obj.detach(), se3_exp_map(obj_tform6_tmp.detach()))

            results['time_pose_iterative'] = torch.Tensor([time.time() - time_before_pose_iterative,]) / B

        cam_tform4x4_obj = cam_tform4x4_obj.clone().detach()

        results['time_pose'] = torch.Tensor([time.time() - time_pred_class,]) / B

        diff_rot3x3 = rot3x3(batch.cam_tform4x4_obj[:, :3, :3].permute(0, 2, 1), cam_tform4x4_obj[:, :3, :3])

        try:
            diff_so3_log = pytorch3d.transforms.so3_log_map(diff_rot3x3.permute(0, 2, 1))
            diff_rot_angle_rad = torch.norm(diff_so3_log, dim=-1)
        except ValueError:
            logger.warning(
                f'Cannot calculate deviation in rotation angle due to rot3x3 trace being too small, setting deviation to 0.')
            diff_rot_angle_rad = torch.zeros_like(diff_rot3x3[:, 0, 0])
        results['rot_diff_rad'] = diff_rot_angle_rad
        results['label_gt'] = batch.label
        results['label_pred'] = pred_class_ids
        results['label_names'] = self.config.categories
        results['sim'] = sim
        results['cam_tform4x4_obj'] = cam_tform4x4_obj
        results['item_id'] = batch.item_id
        results['name_unique'] = batch.name_unique

        return results

    def get_results_visual(self, results_epoch, dataset: OD3D_Dataset, config_visualize: DictConfig):
        results = OD3D_Results()

        count_best = config_visualize.count_best
        count_worst = config_visualize.count_worst
        count_rand = config_visualize.count_rand
        modalities = config_visualize.modalities
        live = config_visualize.live

        if len(modalities) == 0:
            return results

        caption_metrics = ['sim', 'rot_diff_rad']

        if 'rot_diff_rad' in results_epoch.keys():

            rank_metric_name = 'rot_diff_rad'
            # sorts values ascending
            epoch_ranked_ids = results_epoch[rank_metric_name].sort(dim=0)[1]
        elif 'sim' in results_epoch.keys():
            rank_metric_name = 'sim'
            # sorts values descending
            epoch_ranked_ids = results_epoch[rank_metric_name].sort(dim=0, descending=True)[1]
        else:
            logger.warning(f'Could not find a suitable rank metric in results {results_epoch.keys()}')
            return results

        if 'name_unique' in results_epoch.keys() and len(results_epoch['name_unique']) > 0:
            # this only groups the ranked elements depending on their category / sequence etc.
            # https://stackoverflow.com/questions/51408344/pandas-dataframe-interleaved-reordering
            group_names = list(set(['/'.join(name_unique.split('/')[:-1]) for name_unique in results_epoch['name_unique']]))
            group_ids = [group_id for result_id in range(len(results_epoch['name_unique'])) for group_id, group_name in enumerate(group_names) if results_epoch['name_unique'][epoch_ranked_ids[result_id]].startswith(group_name)]
            df = pd.DataFrame(np.stack([epoch_ranked_ids, np.array(group_ids)], axis=-1), columns=['rank', 'group'])
            epoch_ranked_ids = torch.from_numpy(df.loc[df.groupby("group").cumcount().sort_values(kind='mergesort').index]['rank'].values)

            df = df[::-1]
            epoch_ranked_ids_worst = torch.from_numpy(df.loc[df.groupby("group").cumcount().sort_values(kind='mergesort').index]['rank'].values)
        else:
            epoch_ranked_ids_worst = epoch_ranked_ids.flip(dims=(0,))

        epoch_best_ids = epoch_ranked_ids[:count_best]
        epoch_best_names = [f'best/{i+1}' for i in range(len(epoch_best_ids))]
        epoch_worst_ids = epoch_ranked_ids_worst[:count_worst]
        epoch_worst_names = [f'worst/{len(epoch_worst_ids) - i}' for i in range(len(epoch_worst_ids))]
        epoch_rand_ids = epoch_ranked_ids[torch.randperm(len(epoch_ranked_ids))[:count_rand]]
        epoch_rand_names = [f'rand/{i+1}' for i in range(len(epoch_rand_ids))]

        sel_rank_ids = torch.cat([epoch_best_ids, epoch_worst_ids, epoch_rand_ids], dim=0)
        sel_item_ids = results_epoch['item_id'][sel_rank_ids]
        sel_names = epoch_best_names + epoch_worst_names + epoch_rand_names
        sel_name_unique = [results_epoch['name_unique'][id] for id in sel_rank_ids]
        dict_name_unique_to_result_id = dict(zip(sel_name_unique, sel_rank_ids))
        dict_name_unique_to_sel_name = dict(zip(sel_name_unique, sel_names))
        logger.info('create dataset ...')
        dataset_visualize = dataset.get_subset_with_item_ids(item_ids=sel_item_ids)


        logger.info('create dataloader ...')
        dataloader = torch.utils.data.DataLoader(dataset=dataset_visualize, batch_size=self.config.test.dataloader.batch_size,
                                                 shuffle=False,
                                                 collate_fn=dataset.collate_fn,
                                                 num_workers=self.config.test.dataloader.num_workers,
                                                 pin_memory=self.config.test.dataloader.pin_memory)
        for i, batch in tqdm(enumerate(iter(dataloader))):
            with torch.no_grad():
                batch.to(device=self.device)
                B = len(batch)
                batch_result_ids = torch.LongTensor([dict_name_unique_to_result_id[batch.name_unique[b]] for b in range(B)]).to(device=self.device)
                if 'gt_cam_tform4x4_obj' in results_epoch.keys():
                    batch.cam_tform4x4_obj = results_epoch['gt_cam_tform4x4_obj'].to(device=self.device)[batch_result_ids]

                batch_sequences_ids = torch.LongTensor([self.sequences_unique_names.index('/'.join(name_unique.split('/')[:-1])) for
                                       name_unique in batch.name_unique])

                batch_sel_names = [dict_name_unique_to_sel_name[batch.name_unique[b]] for b in range(B)]
                batch_names = [batch.name_unique[b] for b in range(B)]
                batch_sel_scores = []
                for b in range(B):
                    batch_sel_scores.append('\n'.join([f'{metric}={results_epoch[metric].to(device=self.device)[batch_result_ids[b]].cpu().detach().item():.3f}' for metric in caption_metrics if metric in results_epoch.keys()]))

                feats2d_net = self.net(batch.rgb)
                #feats2d_net = resize(feats2d_net,
                #                     scale_factor=self.down_sample_rate / config_visualize.down_sample_rate)

                if VISUAL_MODALITIES.NET_FEATS_NEAREST_VERTS in modalities:

                    logger.info('create net_feats_nearest_verts ...')
                    verts3d = self.get_nearest_verts3d_to_feats2d_net(feats2d_net=feats2d_net, categories_ids=batch_sequences_ids,
                                                                      zero_if_sim_clutter_larger=True)
                    verts3d = resize(verts3d, scale_factor=self.down_sample_rate / config_visualize.down_sample_rate)
                    for b in range(len(batch)):
                        img = blend_rgb(resize(batch.rgb[b], scale_factor=1. / config_visualize.down_sample_rate), verts3d[b])
                        results[f'visual/{batch_sel_names[b]}_{VISUAL_MODALITIES.NET_FEATS_NEAREST_VERTS}'] = image_as_wandb_image(img, caption=f'{batch_sel_names[b]}, {batch_names[b]}, {batch_sel_scores[b]}')
                        if live:
                            show_img(img)

                if VISUAL_MODALITIES.SAMPLES in modalities:
                    logger.info('create samples ...')
                    s_cam_tform4x4_obj = results_epoch['samples_cam_tform4x4_obj'].to(device=self.device)[batch_result_ids]
                    s_cam_intr4x4 = results_epoch['samples_cam_intr4x4'].to(device=self.device)[batch_result_ids]
                    sim = results_epoch['samples_sim'].to(device=self.device)[batch_result_ids]

                    """        
                    s_cam_tform4x4_obj, s_cam_intr4x4 = self.get_samples(config_sample=self.config.inference.sample,
                                                                                               cam_intr4x4=batch.cam_intr4x4,
                                                                                               cam_tform4x4_obj=batch.cam_tform4x4_obj,
                                                                                               feats2d_net=feats2d_net,
                                                                                               categories_ids=batch.label)
                    sim = self.get_sim_feats2d_net_with_cams(feats2d_net=feats2d_net,
                                                             cam_tform4x4_obj=s_cam_tform4x4_obj,
                                                             cam_intr4x4=s_cam_intr4x4,
                                                             categories_ids=batch.label,
                                                             broadcast_batch_and_cams=True)
                    """

                    ncds = self.get_ncds_with_cam(cam_tform4x4_obj=s_cam_tform4x4_obj,
                                                  cam_intr4x4=s_cam_intr4x4, size=batch.size,
                                                  categories_ids=batch_sequences_ids, down_sample_rate=config_visualize.down_sample_rate, broadcast_batch_and_cams=True, pre_rendered=False)


                    for b in range(len(batch)):
                        imgs = ncds[b]
                        imgs_sim = sim[b][:].expand(*sim[b].shape)  # , *mesh_feats2d_rendered.shape[-2:]
                        if self.config.inference.sample.method == 'uniform':
                            imgs = imgs.reshape(self.config.inference.sample.uniform.azim.steps, self.config.inference.sample.uniform.elev.steps, self.config.inference.sample.uniform.theta.steps,
                                                *imgs.shape[-3:])[:, :, :]
                            imgs_sim = imgs_sim.reshape(self.config.inference.sample.uniform.azim.steps, self.config.inference.sample.uniform.elev.steps, self.config.inference.sample.uniform.theta.steps)[:, :, :]
                        imgs = blend_rgb(resize(batch.rgb[b], scale_factor=1. / config_visualize.down_sample_rate), imgs)

                        if config_visualize.samples_sorted:
                            imgs_sim = imgs_sim.flatten(0)
                            imgs = imgs.reshape(-1, *imgs.shape[-3:])
                            imgs_sim_ids = imgs_sim.sort(descending=True)[1]
                            imgs_sim_ids = imgs_sim_ids[:49]
                            imgs = imgs[imgs_sim_ids]
                            imgs_sim = imgs_sim[imgs_sim_ids]

                        if config_visualize.samples_scores:
                            logger.info('create plot samples scores...')

                            plt.ioff()
                            fig, ax = plt.subplots()
                            ax.plot(imgs_sim.detach().cpu().numpy(), label='sim')  # density=False would make counts
                            #ax.set_ylim(0., 1.)
                            #ax.ylabel('sim')
                            #ax.xlabel('samples')

                            img = get_img_from_plot(ax=ax, fig=fig)
                            plt.close(fig)
                            img = resize(img, H_out=imgs.shape[-2], W_out=imgs.shape[-1])
                            img = draw_text_in_rgb(img, fontScale=0.4, lineThickness=2, fontColor=(0, 0, 0), text=f'{batch_sel_scores[b]}\nmin={imgs_sim.min().item():.3f}\nmax={imgs_sim.max().item():.3f}')
                            imgs = torch.cat([imgs, img[None,].to(device=imgs.device)], dim=0)
                            #resize(img, )
                            """
                            samples_score_size = imgs.shape[-1] // 5
                            imgs[..., -samples_score_size:, -samples_score_size:] = (
                                    255 * imgs_sim.reshape(*imgs_sim.shape, 1, 1, 1).expand(*imgs.shape[:-2],
                                                                                            samples_score_size,
                                                                                            samples_score_size)).to(
                                torch.uint8)
                            """

                        img = imgs_to_img(imgs)
                        results[f'visual/{batch_sel_names[b]}_{VISUAL_MODALITIES.SAMPLES}'] = image_as_wandb_image(img, caption=f'{batch_sel_names[b]}, {batch_names[b]}, {batch_sel_scores[b]}, min={imgs_sim.min().item():.3f}, max={imgs_sim.max().item():.3f}')
                        if live:
                            show_img(img)


                if VISUAL_MODALITIES.SIM_PXL in modalities:
                    logger.info('create sim pxl...')
                    batch_pred_label = results_epoch['label_pred'].to(device=self.device)[batch_result_ids]
                    batch_pred_cam_tform4x4 = results_epoch['cam_tform4x4_obj'].to(device=self.device)[batch_result_ids]
                    sim, sim_pxl = self.get_sim_feats2d_net_with_cams(feats2d_net=feats2d_net,
                                                                      cam_intr4x4=batch.cam_intr4x4,
                                                                      cam_tform4x4_obj=batch_pred_cam_tform4x4,
                                                                      categories_ids=batch_sequences_ids, return_sim_pxl=True,
                                                                      broadcast_batch_and_cams=False,
                                                                      sim_feats_mesh_with_image=SIM_FEATS_MESH_WITH_IMAGE.RENDERED,
                                                                      pre_rendered=False)

                    sim_pxl = resize(sim_pxl, scale_factor=self.down_sample_rate / config_visualize.down_sample_rate)
                    for b in range(len(batch)):
                        img = blend_rgb(resize(batch.rgb[b], scale_factor=1. / config_visualize.down_sample_rate),
                                        sim_pxl[b])
                        results[f'visual/{batch_sel_names[b]}_{VISUAL_MODALITIES.SIM_PXL}'] = image_as_wandb_image(img, caption=f'{batch_sel_names[b]}, {batch_names[b]}, mean sim={sim[b].item()}')
                        if live:
                            show_img(img)


                if VISUAL_MODALITIES.PRED_VERTS_NCDS_IN_RGB in modalities or VISUAL_MODALITIES.PRED_VS_GT_VERTS_NCDS_IN_RGB in modalities:
                    logger.info('create pred verts ncds...')
                    batch_pred_label = results_epoch['label_pred'].to(device=self.device)[batch_result_ids]
                    batch_pred_cam_tform4x4 = results_epoch['cam_tform4x4_obj'].to(device=self.device)[batch_result_ids]
                    pred_verts_ncds = self.get_ncds_with_cam(cam_intr4x4=batch.cam_intr4x4, cam_tform4x4_obj=batch_pred_cam_tform4x4, categories_ids=batch_sequences_ids, size=batch.size, down_sample_rate=config_visualize.down_sample_rate, pre_rendered=False)
                    if VISUAL_MODALITIES.PRED_VERTS_NCDS_IN_RGB in modalities:
                        for b in range(len(batch)):
                            img = blend_rgb(resize(batch.rgb[b], scale_factor=1. / config_visualize.down_sample_rate),
                                            pred_verts_ncds[b])
                            results[f'visual/{batch_sel_names[b]}_{VISUAL_MODALITIES.PRED_VERTS_NCDS_IN_RGB}'] = image_as_wandb_image(img, caption=f'{batch_sel_names[b]}, {batch_names[b]}, {batch_sel_scores[b]}')
                            if live:
                                show_img(img)

                if VISUAL_MODALITIES.GT_VERTS_NCDS_IN_RGB in modalities or VISUAL_MODALITIES.PRED_VS_GT_VERTS_NCDS_IN_RGB in modalities:
                    logger.info('create gt verts ncds...')
                    gt_verts_ncds = self.get_ncds_with_cam(cam_intr4x4=batch.cam_intr4x4,
                                                           cam_tform4x4_obj=batch.cam_tform4x4_obj,
                                                           categories_ids=batch_sequences_ids, size=batch.size,
                                                           down_sample_rate=config_visualize.down_sample_rate, pre_rendered=False)
                    if VISUAL_MODALITIES.GT_VERTS_NCDS_IN_RGB in modalities:
                        for b in range(len(batch)):
                            img = blend_rgb(resize(batch.rgb[b], scale_factor=1. / config_visualize.down_sample_rate),
                                            gt_verts_ncds[b])
                            results[
                                f'visual/{batch_sel_names[b]}_{VISUAL_MODALITIES.GT_VERTS_NCDS_IN_RGB}'] = image_as_wandb_image(
                                img, caption=f'{batch_sel_names[b]}, {batch_names[b]}, {batch_sel_scores[b]}')
                            if live:
                                show_img(img)
                if VISUAL_MODALITIES.PRED_VS_GT_VERTS_NCDS_IN_RGB in modalities:
                    logger.info('create pred vs gt verts ncds...')
                    for b in range(len(batch)):
                        img1 = blend_rgb(resize(batch.rgb[b], scale_factor=1. / config_visualize.down_sample_rate),
                                         pred_verts_ncds[b])
                        img2 = blend_rgb(resize(batch.rgb[b], scale_factor=1. / config_visualize.down_sample_rate),
                                         gt_verts_ncds[b])
                        img = imgs_to_img(torch.stack([img1, img2], dim=0)[None,])

                        results[
                            f'visual/{batch_sel_names[b]}_{VISUAL_MODALITIES.PRED_VS_GT_VERTS_NCDS_IN_RGB}'] = image_as_wandb_image(
                            img, caption=f'{batch_sel_names[b]}, {batch_names[b]}, {batch_sel_scores[b]}')
                        if live:
                            show_img(img)
        return results


    def get_nearest_verts3d_to_feats2d_net(self, feats2d_net, categories_ids, zero_if_sim_clutter_larger=False,
                                           return_sim_texture_and_clutter=False):
        B = len(feats2d_net)
        sim_nearest_texture_vals, sim_nearest_texture_ids = torch.einsum('bchw,bvc->bvhw', feats2d_net,
                                                                         self.meshes.
                                                                         get_feats_stacked_with_mesh_ids(categories_ids)
                                                                         .detach()).max(dim=1, keepdim=True)

        sim_nearest_texture_verts3d = torch.stack([self.meshes.get_verts_stacked_with_mesh_ids(
            categories_ids[b: b + 1])[0, sim_nearest_texture_ids[b, 0]] for b in range(B)], dim=0)

        if zero_if_sim_clutter_larger or return_sim_texture_and_clutter:
            sim_clutter = \
                torch.einsum('bchw,nc->bnhw', feats2d_net, self.clutter_feats.detach()).max(dim=1, keepdim=True)[0]
            # sim_clutter = torch.einsum('bchw,nc->bnhw', net_feats2d, self.clutter_feats.detach()).mean(dim=1, keepdim=True)
            if zero_if_sim_clutter_larger:
                sim_nearest_texture_verts3d[sim_clutter[:, 0] > sim_nearest_texture_vals[:, 0]] = 0.

        sim_nearest_texture_verts3d = sim_nearest_texture_verts3d.permute(0, 3, 1, 2)
        if return_sim_texture_and_clutter:
            return sim_nearest_texture_verts3d, sim_nearest_texture_vals, sim_clutter

        return sim_nearest_texture_verts3d

    def get_ncds_with_cam(self, cam_intr4x4: torch.Tensor, cam_tform4x4_obj: torch.Tensor, size: torch.Tensor, categories_ids: torch.Tensor, down_sample_rate=1., broadcast_batch_and_cams=False, pre_rendered: bool= None):
        if pre_rendered is None:
            pre_rendered = self.config.inference.pre_rendered

        if pre_rendered:
            return self.meshes.get_pre_rendered_feats(cams_tform4x4_obj=cam_tform4x4_obj,
                                                      cams_intr4x4=cam_intr4x4,
                                                      imgs_sizes=size, meshes_ids=categories_ids,
                                                      modality=MESH_RENDER_MODALITIES.VERTS_NCDS,
                                                      down_sample_rate=down_sample_rate,
                                                      broadcast_batch_and_cams=broadcast_batch_and_cams
                                                      )
        else:
            return self.meshes.render_feats(cams_tform4x4_obj=cam_tform4x4_obj,
                                        cams_intr4x4=cam_intr4x4,
                                        imgs_sizes=size, meshes_ids=categories_ids,
                                        modality=MESH_RENDER_MODALITIES.VERTS_NCDS,
                                        down_sample_rate=down_sample_rate,
                                        broadcast_batch_and_cams=broadcast_batch_and_cams)

    def get_sim_feats2d_net_with_cams(self, feats2d_net, cam_tform4x4_obj, cam_intr4x4, categories_ids, return_sim_pxl=False,
                                      broadcast_batch_and_cams=False, feats2d_net_mask=None, sim_feats_mesh_with_image: SIM_FEATS_MESH_WITH_IMAGE=None, pre_rendered: bool=None):
        if sim_feats_mesh_with_image is None:
            sim_feats_mesh_with_image = self.config.inference.sim_feats_mesh_with_image
        if pre_rendered is None:
            pre_rendered = self.config.inference.pre_rendered
        size = torch.Tensor([feats2d_net.shape[2] * self.down_sample_rate, feats2d_net.shape[3] * self.down_sample_rate]).to(device=feats2d_net.device)
        if sim_feats_mesh_with_image == SIM_FEATS_MESH_WITH_IMAGE.RENDERED:
            if pre_rendered:
                mesh_feats2d_rendered = self.meshes.get_pre_rendered_feats(cams_tform4x4_obj=cam_tform4x4_obj,
                                                                           cams_intr4x4=cam_intr4x4,
                                                                           imgs_sizes=size, meshes_ids=categories_ids,
                                                                           modality=MESH_RENDER_MODALITIES.FEATS,
                                                                           down_sample_rate=self.down_sample_rate,
                                                                           broadcast_batch_and_cams=broadcast_batch_and_cams
                                                                           )
            else:
                mesh_feats2d_rendered = self.meshes.render_feats(cams_tform4x4_obj=cam_tform4x4_obj,
                                                                 cams_intr4x4=cam_intr4x4,
                                                                 imgs_sizes=size, meshes_ids=categories_ids,
                                                                 down_sample_rate=self.down_sample_rate,
                                                                 broadcast_batch_and_cams=broadcast_batch_and_cams)

            return self.get_sim_feats2d_net_and_rendered(feats2d_net=feats2d_net, feats2d_rendered=mesh_feats2d_rendered, return_sim_pxl=return_sim_pxl, feats2d_net_mask=feats2d_net_mask)

        elif sim_feats_mesh_with_image == SIM_FEATS_MESH_WITH_IMAGE.VERTS2D:
            verts2d_mesh, verts2d_mesh_mask = self.meshes.verts2d(cams_intr4x4=cam_intr4x4,
                                                                  cams_tform4x4_obj=cam_tform4x4_obj,
                                                                  imgs_sizes=size, mesh_ids=categories_ids,
                                                                  down_sample_rate=self.down_sample_rate,
                                                                  broadcast_batch_and_cams=broadcast_batch_and_cams)
            return self.get_sim_feats2d_net_and_verts(categories_ids=categories_ids, feats2d_net=feats2d_net, verts2d_mesh=verts2d_mesh, verts2d_mesh_mask=verts2d_mesh_mask, return_sim_pxl=return_sim_pxl, feats2d_net_mask=feats2d_net_mask)


    def get_nearest_corresp2d3d(self, feats2d_net, meshes_ids, feats2d_net_mask: torch.Tensor=None):
        nearest_verts3d, sim_texture, sim_clutter = self.get_nearest_verts3d_to_feats2d_net(feats2d_net, meshes_ids, return_sim_texture_and_clutter=True)
        nearest_verts2d = get_pxl2d_like(nearest_verts3d.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # H=sim_clutter.shape[1], W=sim_clutter.shape[2], dtype=sim_nearest_texture_verts.dtype, device=sim_nearest_texture_verts.device)[None,].expand()
        prob_corresp2d3d = (sim_clutter < sim_texture) * sim_texture
        prob_corresp2d3d *= feats2d_net_mask

        return nearest_verts3d, nearest_verts2d, prob_corresp2d3d

    def get_sim_feats2d_net_and_verts(self, categories_ids, feats2d_net, verts2d_mesh, verts2d_mesh_mask, return_sim_pxl=False, feats2d_net_mask=None):
        """
        Args:
            categories_ids (torch.Tensor): B, ids of categories=meshes to which the similarty should be calculated
            feats2d_net (torch.Tensor): BxCxHxW
            verts2d_mesh (torch.Tensor): Bx(T)xVx2
            verts2d_mesh_mask (torch.Tensor): Bx(T)xV
            return_sim_pxl (bool): Indicates whether pixelwise similarity should be returned or not.
            feats2d_net_mask (torch.Tensor): BxCxHxW

        Returns:
            sim (torch.Tensor): BxV, or Bx1 if rendered features is 4-dimensional.
            sim_pxl (torch.Tensor, optional): BxTxHxW, or Bx1xHxW if rendered features is 4-dimensional.

        """
        B, C, H, W = feats2d_net.shape
        V = verts2d_mesh.shape[-2]

        verts2d_mesh_ids = self.meshes.get_verts_and_noise_ids_stacked(categories_ids, count_noise_ids=0)  # B x V
        feats2d_mesh = self.meshes.feats[verts2d_mesh_ids]  # B x V x C

        if verts2d_mesh.dim() == 4:
            T = verts2d_mesh.shape[1]
            feats2d_net_sampled = sample_pxl2d_pts(feats2d_net, pxl2d=verts2d_mesh.reshape(B, -1, 2)).reshape(B, -1, V, C)
            if feats2d_net_mask is not None:
                feats2d_net_mask_sampled = sample_pxl2d_pts(feats2d_net_mask, pxl2d=verts2d_mesh.reshape(B, -1, 2)).reshape(B, -1, V)
        elif verts2d_mesh.dim() == 3:
            # B x T x V x C
            T = 1
            feats2d_net_sampled = sample_pxl2d_pts(feats2d_net, pxl2d=verts2d_mesh)[:, None]
            if feats2d_net_mask is not None:
                # B x T x V
                feats2d_net_mask_sampled = sample_pxl2d_pts(feats2d_net_mask, pxl2d=verts2d_mesh).reshape(B, -1, V)

        else:
            T = 1
            feats2d_net_mask_sampled = None
            feats2d_net_sampled = None
            logger.error(f'Unexpected dimension of verts2d_mesh {verts2d_mesh.shape}')
        # [verts2d_mesh_mask]
        sim_clutter = torch.einsum('btvc,nc->bntv', feats2d_net_sampled, self.clutter_feats.detach()).max(dim=1, keepdim=False)[0]
        sim_texture_multiple_cams = torch.einsum('btvc,bvc->btv', feats2d_net_sampled, feats2d_mesh)
        sim_pxl = torch.max(sim_texture_multiple_cams, sim_clutter)

        if verts2d_mesh_mask.dim() == 3:
            sim_pxl_mask = verts2d_mesh_mask
        elif verts2d_mesh_mask.dim() == 2:
            sim_pxl_mask = verts2d_mesh_mask[:, None]
        else:
            sim_pxl_mask = None
            logger.error(f'Unexpected dimension of verts2d_mesh_mask {verts2d_mesh_mask.shape}')

        if feats2d_net_mask is not None:
            sim_pxl_mask = sim_pxl_mask * feats2d_net_mask_sampled

        sim_pxl *= sim_pxl_mask
        sim = sim_pxl.flatten(2).sum(dim=-1) / (sim_pxl_mask.flatten(2).sum(dim=-1) + 1e-10)

        sim_pxl2d = torch.zeros(size=(B, T, H, W), dtype=feats2d_net.dtype, device=feats2d_net.device)
        #torch.gather(input=sim_pxl2d, dim=-1, index=verts2d_mesh[:, :, :, 0][..., None, :].long())
        #sim_pxl2d_x =
        #sim_pxl2d_y = verts2d_mesh.reshape(-1, V, 2)[:, :, 1].long()

        #sim_pxl2d[]
        #sim_pxl2d_sampled = sample_pxl2d_pts(sim_pxl2d.reshape(-1, 1, H, W), pxl2d=verts2d_mesh.reshape(-1, V, 2)).reshape(B, T, V)
        #sim_pxl2d_sampled[:, :] = sim_pxl
        sim_pxl = sim_pxl2d

        if return_sim_pxl:
            return sim, sim_pxl
        else:
            return sim

    @staticmethod
    def batched_index_select(input, dim, index):
        for ii in range(1, len(input.shape)):
            if ii != dim:
                index = index.unsqueeze(ii)
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.expand(expanse)
        return torch.gather(input, dim, index)

    def get_sim_feats2d_net_and_rendered(self, feats2d_net, feats2d_rendered, return_sim_pxl=False, feats2d_net_mask=None):
        """

        Args:
            feats2d_net (torch.Tensor): BxCxHxW
            feats2d_rendered (torch.Tensor): Bx(T)xCxHxW
            return_sim_pxl (bool): Indicates whether pixelwise similarity should be returned or not.
            feats2d_net_mask (torch.Tensor): Bx1xHxW

        Returns:
            sim (torch.Tensor): BxT, or Bx1 if rendered features is 4-dimensional.
            sim_pxl (torch.Tensor, optional): BxTxHxW, or Bx1xHxW if rendered features is 4-dimensional.
        """

        sim_clutter = torch.einsum('bchw,nc->bnhw', feats2d_net, self.clutter_feats.detach()).max(dim=1, keepdim=True)[
            0]
        # sim_clutter = torch.einsum('bchw,nc->bnhw', feats2d_net, self.clutter_feats.detach()).mean(dim=1, keepdim=True)

        if feats2d_rendered.dim() == 5:
            feats2d_rendered_clutter_mask = feats2d_rendered.norm(dim=2) == 0.
            sim_texture_multiple_cams = torch.einsum('bchw,bvchw->bvhw', feats2d_net, feats2d_rendered)
        else:
            feats2d_rendered_clutter_mask = (feats2d_rendered.norm(dim=1) == 0.)[:, None]
            sim_texture_multiple_cams = torch.einsum('bchw,bchw->bhw', feats2d_net, feats2d_rendered)[:, None]

        sim_pxl = torch.max(sim_texture_multiple_cams, sim_clutter)
        sim_pxl[feats2d_rendered_clutter_mask] = sim_clutter.expand(*sim_pxl.shape)[feats2d_rendered_clutter_mask]

        if feats2d_net_mask is None:
            sim = sim_pxl.flatten(2).mean(dim=-1)
        else:
            sim_pxl *= feats2d_net_mask
            sim = sim_pxl.flatten(2).sum(dim=-1) / (feats2d_net_mask.flatten(2).sum(dim=-1) + 1e-10)

        if return_sim_pxl:
            return sim, sim_pxl
        else:
            return sim

    def get_samples(self, config_sample: DictConfig, cam_intr4x4: torch.Tensor, cam_tform4x4_obj: torch.Tensor,
                    feats2d_net: torch.Tensor, categories_ids: torch.Tensor, feats2d_net_mask: torch.Tensor=None):
        B = len(feats2d_net)
        if config_sample.method == 'uniform':
            azim = torch.linspace(start=eval(config_sample.uniform.azim.min), end=eval(config_sample.uniform.azim.max), steps=config_sample.uniform.azim.steps).to(
                device=self.device)  # 12
            elev = torch.linspace(start=eval(config_sample.uniform.elev.min), end=eval(config_sample.uniform.elev.max), steps=config_sample.uniform.elev.steps).to(
                device=self.device)  # start=-torch.pi / 6, end=torch.pi / 3, steps=4
            theta = torch.linspace(start=eval(config_sample.uniform.theta.min), end=eval(config_sample.uniform.theta.max),
                                   steps=config_sample.uniform.theta.steps).to(
                device=self.device)  # -torch.pi / 6, end=torch.pi / 6, steps=3

            #dist = torch.linspace(start=eval(config_sample.uniform.dist.min), end=eval(config_sample.uniform.dist.max), steps=config_sample.uniform.dist.steps).to(
            #    device=self.device)
            dist = torch.linspace(start=1., end=1., steps=1).to(device=self.device)


            azim_shape = azim.shape
            elev_shape = elev.shape
            theta_shape = theta.shape
            dist_shape = dist.shape
            in_shape = azim_shape + elev_shape + theta_shape + dist_shape
            azim = azim[:, None, None, None].expand(in_shape).reshape(-1)
            elev = elev[None, :, None, None].expand(in_shape).reshape(-1)
            theta = theta[None, None, :, None].expand(in_shape).reshape(-1)
            dist = dist[None, None, None, :].expand(in_shape).reshape(-1)
            cams_multiview_tform4x4_cuboid = transf4x4_from_spherical(azim=azim, elev=elev, theta=theta, dist=dist)

            C = len(cams_multiview_tform4x4_cuboid)

            b_cams_multiview_tform4x4_obj = cams_multiview_tform4x4_cuboid[None,].repeat(B, 1, 1, 1)

            # assumption 1: distance translation to object is known
            #b_cams_multiview_tform4x4_obj[:, :, 2, 3] = cam_tform4x4_obj[:, None].repeat(1, C, 1, 1)[:, :, 2, 3]
            # logger.info(f'dist {batch.cam_tform4x4_obj[:, 2, 3]}')
            # assumption 2: translation to object is known
            b_cams_multiview_tform4x4_obj[:, :, :3, 3] = cam_tform4x4_obj[:, None].repeat(1, C, 1, 1)[:, :, :3, 3]

            b_cams_multiview_intr4x4 = cam_intr4x4[:, None].repeat(1, C, 1, 1)

        elif config_sample.method == 'epnp3d2d':

            nearest_verts3d, nearest_verts2d, prob_well_corresp = self.get_nearest_corresp2d3d(feats2d_net=feats2d_net,
                                                                                               meshes_ids=categories_ids,
                                                                                               feats2d_net_mask=feats2d_net_mask)
            H, W = feats2d_net.shape[2:]
            prob_well_corresp = prob_well_corresp.flatten(1)
            if (prob_well_corresp.sum(dim=-1) == 0).any():
                logger.warning(f"No texture similarity is larger than the clutter similarity")
            prob_well_corresp[prob_well_corresp.sum(dim=-1) == 0] = 1.

            K = config_sample.epnp3d2d.count_cams
            N = config_sample.epnp3d2d.count_pts

            masks_in_ids = torch.multinomial(prob_well_corresp, num_samples=K * N).reshape(-1, K, N)

            masks_in = torch.zeros(size=(B, K, H * W),
                                   device=self.device, dtype=torch.bool)
            for b in range(B):
                for k in range(K):
                    masks_in[b, k, masks_in_ids[b, k]] = True
            masks_in = masks_in.reshape(B, K, H, W)
            b_cams_multiview_tform4x4_obj = batchwise_fit_se3_to_corresp_3d_2d_and_masks(masks_in=masks_in,
                                                                                         pts1=nearest_verts3d,
                                                                                         pxl2=nearest_verts2d,
                                                                                         proj_mat=cam_intr4x4[
                                                                                                  :, :2,
                                                                                                  :3] / self.down_sample_rate,
                                                                                         method="cpu-epnp")
            b_cams_multiview_intr4x4 = cam_intr4x4[:, None].repeat(1, K, 1, 1)
            b_cams_multiview_tform4x4_obj[b_cams_multiview_tform4x4_obj.flatten(2).isinf().any(dim=2), :,
            :] = torch.eye(4, device=b_cams_multiview_tform4x4_obj.device)
            b_cams_multiview_tform4x4_obj[(b_cams_multiview_tform4x4_obj[:, :, 3, :3] != 0.).any(dim=-1), :,
            :] = torch.eye(4, device=b_cams_multiview_tform4x4_obj.device)

        return b_cams_multiview_tform4x4_obj, b_cams_multiview_intr4x4


    """
    
            if self.config.train.visualize.verts_ncds_in_rgb:
                for b in range(len(batch)):
                    if batch.name_unique[b] in visual_names_unique:
                        verts_ncds_in_rgb = blend_rgb(batch.rgb[b], (
                            self.meshes.render_feats(cams_tform4x4_obj=batch.cam_tform4x4_obj[b:b + 1],
                                                     cams_intr4x4=batch.cam_intr4x4[b:b + 1],
                                                     imgs_sizes=batch.size, meshes_ids=batch.label[b:b + 1],
                                                     modality=MESH_RENDER_MODALITIES.VERTS_NCDS)[0]).to(
                            dtype=batch.rgb.dtype))
                        from od3d.cv.geometry.transform import proj3d2d_origin
                        verts_ncds_in_rgb = draw_pixels(verts_ncds_in_rgb, pxls=proj3d2d_origin(
                            torch.bmm(batch.cam_intr4x4, batch.cam_tform4x4_obj)[b:b + 1]), colors=[1., 0., 0.])
                        verts_ncds_in_rgb = draw_pixels(verts_ncds_in_rgb,
                                                        pxls=proj3d2d_origin(batch.cam_proj4x4_obj[b:b + 1]),
                                                        colors=[1., 0., 0.])
                        img = draw_pixels(verts_ncds_in_rgb, self.down_sample_rate * vts2d[b, mask_vts2d_vsbl[b]],
                                          colors=self.meshes.get_verts_ncds_with_mesh_id(batch.label[b])[
                                              mask_vts2d_vsbl[b]])
                        results_train['verts_ncds_in_rgb'] = image_as_wandb_image(img,
                                                                                  caption=f'Frame Name {batch.name_unique[b]}')
                        if self.config.train.visualize.live:
                            show_img(img)
    
            if self.config.train.visualize.net_feats_nearest_verts:
                for b in range(len(batch)):
                    if batch.name_unique[b] in visual_names_unique:
                        clutter_sim, clutter_sim_ids = torch.einsum('bcn,vc->bnv', net_feats2d[b:b + 1].flatten(-2),
                                                                    self.clutter_feats).max(dim=-1)
                        net_mesh_nearest_feats_sim, net_mesh_nearest_feats_ids = torch.einsum('bcn,vc->bnv',
                                                                                              net_feats2d[
                                                                                              b:b + 1].flatten(-2),
                                                                                              self.meshes.get_feats_with_mesh_id(
                                                                                                  batch.label[
                                                                                                      b])).max(
                            dim=-1)
                        net_mesh_nearest_feats_verts_ncds = self.meshes.get_verts_ncds_with_mesh_id(batch.label[b])[
                            net_mesh_nearest_feats_ids]
                        net_mesh_nearest_feats_verts_ncds[clutter_sim > net_mesh_nearest_feats_sim] = 0.
                        net_mesh_nearest_feats_verts_ncds = net_mesh_nearest_feats_verts_ncds.reshape(-1,
                                                                                                      *net_feats2d.shape[
                                                                                                       -2:],
                                                                                                      3).permute(0,
                                                                                                                 3,
                                                                                                                 1,
                                                                                                                 2)
                        img = blend_rgb(resize(batch.rgb[b], scale_factor=1. / self.down_sample_rate),
                                        net_mesh_nearest_feats_verts_ncds[0])
                        results_train['net_feats_nearest_verts'] = image_as_wandb_image(img,
                                                                                        caption=f'Frame Name {batch.name[b]}')
                        if self.config.train.visualize.live:
                            show_img(img)
    """

    # inference()

    # train()
    # 1. net_feats2d = get_net_feats2d()
    # 1. net_feats2d = get_net_feats2d()
    # 2. corresp2d3d = get_corresp2d3d()
    #   -> visualize NET_FEATS_NEAREST_VERTS
    # 2. samples = get_samples(corresp2d3d, method)
    #   -> visualize SAMPLES
    # 3. sample = select_sample()
    # 4. sample = pose_iterative(sample)
    #   -> visualize SIM_PXL, VERTS_NCDS_IN_RGB

    def visualize_test(self):
        pass