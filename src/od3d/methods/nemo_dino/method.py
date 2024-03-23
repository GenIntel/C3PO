import time
from typing import List
from od3d.methods.method import OD3D_Method
from od3d.datasets.dataset import OD3D_Dataset, OD3D_Frame, OD3D_Frames
from od3d.datasets.co3d.dataset import CO3D
from omegaconf import DictConfig
import pytorch3d.transforms
from od3d.cv.geometry.transform import se3_log_map


import logging
logger = logging.getLogger(__name__)
import torch
import numpy as np
import wandb
import math
from od3d.cv.visual.draw import draw_pixels
from od3d.cv.geometry.transform import se3_exp_map
from od3d.cv.visual.show import imgs_to_img
from od3d.cv.geometry.mesh import Meshes
from pathlib import Path
from od3d.cv.geometry.transform import transf4x4_from_spherical, tform4x4_broadcast, tform4x4, rot3x3
from od3d.cv.metric.pose import get_pose_diff_in_rad

from od3d.cv.geometry.transform import inv_tform4x4
from od3d.methods.nemo import NeMo
from od3d.benchmark.results import OD3D_Results
from dataclasses import dataclass
from typing import Dict
import random
from od3d.cv.visual.resize import resize

from tqdm import tqdm

from od3d.cv.visual.show import show_img
import torchvision
from od3d.cv.visual.blend import blend_rgb
from od3d.cv.visual.sample import sample_pxl2d_pts
from tqdm import tqdm
from od3d.cv.geometry.mesh import MESH_RENDER_MODALITIES

from od3d.datasets.meta import OD3D_Meta


@dataclass
class SequencePseudoLabel():
    obj_tform4x4_cuboid_front: torch.Tensor
    sim: float

from torch.utils.data import Dataset




class NeMo_DINO(NeMo):
    def __init__(
        self,
        config: DictConfig,
        logging_dir,
    ):
        super().__init__(config=config, logging_dir=logging_dir)
        self.mesh_update_count = torch.zeros(size=(self.meshes.feats.shape[0] + self.clutter_feats.shape[0],), device=self.device)
        self.mesh_feats_total = None



    def train_batch(self, batch) -> OD3D_Results:
        results_batch = OD3D_Results()

        batch.to(device=self.device)

        batch.cam_tform4x4_obj = batch.cam_tform4x4_obj.detach()

        logger.info(f'batch.category_id {batch.category_id}')
        logger.info(f'batch.size {batch.size}')

        # B x x N x 2
        vts2d, vts2d_mask = self.meshes.verts2d(cams_intr4x4=batch.cam_intr4x4,
                                                cams_tform4x4_obj=batch.cam_tform4x4_obj,
                                                imgs_sizes=batch.size, mesh_ids=batch.category_id,
                                                down_sample_rate=self.down_sample_rate)

        N = vts2d.shape[1]
        # B x F+N x C
        logger.info(f'batch.size {batch.size}')
        feats2d_net = self.net(batch.rgb)

        logger.info(f'batch.size {batch.size}')
        feats2d_net_mask = torch.ones(size=(feats2d_net.shape[0], 1, feats2d_net.shape[2], feats2d_net.shape[3])).to(device=self.device)
        if self.config.train.use_mask_object:
            feats2d_net_mask = feats2d_net_mask * 1. * resize(batch.mask, H_out=feats2d_net.shape[2],
                                                              W_out=feats2d_net.shape[3])
        if self.config.train.use_mask_rendered_object:
            logger.info(f'batch.size {batch.size}')
            feats2d_net_mask = feats2d_net_mask * self.meshes.render_feats(
                cams_intr4x4=batch.cam_intr4x4,
                cams_tform4x4_obj=batch.cam_tform4x4_obj,
                imgs_sizes=batch.size, meshes_ids=batch.category_id,
                down_sample_rate=self.down_sample_rate,
                modality=MESH_RENDER_MODALITIES.MASK)

        H, W = feats2d_net.shape[-2:]
        xy = torch.stack(
            torch.meshgrid(torch.arange(W, device=self.device), torch.arange(H, device=self.device),
                           indexing='xy'), dim=0)  # HxW
        prob_noise = (1. - 1. * feats2d_net_mask).clamp(0, 1).flatten(1)
        prob_noise[prob_noise.sum(dim=-1) <= 0.] = 1.
        noise2d = xy.flatten(1)[:, torch.multinomial(prob_noise, self.config.num_noise, replacement=True)].permute(1, 2, 0)

        #from od3d.cv.visual.show import show_imgs
        #show_imgs(prob_noise.reshape(-1, 1, H, W))

        vts2d_feats2d_net_mask = sample_pxl2d_pts(feats2d_net_mask, pxl2d=torch.cat([vts2d], dim=1))
        vts2d_mask = vts2d_mask * (vts2d_feats2d_net_mask[:, :, 0] > 0.5)
        net_feats = sample_pxl2d_pts(feats2d_net, pxl2d=torch.cat([vts2d, noise2d], dim=1))

        C = net_feats.shape[2]
        # args: X: Bx3xHxW, keypoint_positions: BxNx2, obj_mask: BxHxW ensures that noise is sampled outside of object mask
        # returns: BxF+NxC


        # net_feats = net_feats[:, :].reshape(-1, net_feats.shape[-1])
        logger.info(batch.category_id)
        batch_vts_ids = self.meshes.get_verts_and_noise_ids_stacked(batch.category_id.tolist(),
                                                                    count_noise_ids=self.config.num_noise)

        # weighting with similarity score
        # net_feats = net_feats * (batch.cam_tform4x4_obj_sim[:, None, None] ** 4)

        # sim_weight = batch.cam_tform4x4_obj_sim[:, None].expand(*net_feats.shape[:2])
        # sim_weight = torch.cat([sim_weight[:, :N][mask_vts2d_vsbl], sim_weight[:, N:].reshape(-1)], dim=0)

        batch_vts_ids = torch.cat([batch_vts_ids[:, :N][vts2d_mask], batch_vts_ids[:, N:].reshape(-1)],
                                  dim=0)
        net_feats = torch.cat([net_feats[:, :N][vts2d_mask], net_feats[:, N:].reshape(-1, C)], dim=0)

        # batch_vts_ids = self.meshes.get_feats_ids_stacked(batch.category_id.tolist())

        bank_feats = torch.cat([self.meshes.feats, self.clutter_feats], dim=0)
        if self.mesh_feats_total is None:
            self.mesh_feats_total = torch.zeros_like(bank_feats)
        if self.config.train.bank_feats_update == 'loss_gradient':
            sim = self.calc_sim('nc,vc->nv', net_feats, bank_feats)
        elif self.config.train.bank_feats_update == 'normalize_loss_gradient':
            sim = self.calc_sim('nc,vc->nv', net_feats, torch.nn.functional.normalize(bank_feats, dim=1))
        elif self.config.train.bank_feats_update == 'moving_average':
            sim = self.calc_sim('nc,vc->nv', net_feats, bank_feats.detach())
            bank_feats_new = self.config.train.alpha * bank_feats[batch_vts_ids].detach() + (1. - self.config.train.alpha) * net_feats.detach()
            batch_vts_ids_unique, batch_vts_ids_unique_inverse, batch_vts_ids_unique_counts = batch_vts_ids.unique(return_inverse=True, return_counts=True)
            bank_feats_new = torch.einsum('nk,nc->kc', torch.nn.functional.one_hot(batch_vts_ids_unique_inverse).to(dtype= bank_feats_new.dtype, device= bank_feats_new.device), bank_feats_new) / batch_vts_ids_unique_counts[:, None]
            bank_feats[batch_vts_ids_unique].data = bank_feats_new
            self.normalize_feats()
        elif self.config.train.bank_feats_update == 'average':
            batch_vts_ids_unique, batch_vts_ids_unique_inverse, batch_vts_ids_unique_counts = batch_vts_ids.unique(return_inverse=True, return_counts=True)
            sim = self.calc_sim('nc,vc->nv', net_feats, bank_feats.detach())  
            bank_feats_new =  self.mesh_feats_total[batch_vts_ids]  + net_feats
            bank_feats_new = torch.einsum('nk,nc->kc', torch.nn.functional.one_hot(batch_vts_ids_unique_inverse).to(dtype= bank_feats_new.dtype, device= bank_feats_new.device), bank_feats_new) / batch_vts_ids_unique_counts[:, None]
            self.mesh_update_count[batch_vts_ids_unique] += 1
            self.mesh_feats_total[batch_vts_ids_unique] = bank_feats_new
            bank_feats[batch_vts_ids_unique].data = self.mesh_feats_total[batch_vts_ids_unique]/ self.mesh_update_count[batch_vts_ids_unique, None]
            self.normalize_feats()

        else:
            logger.error(f'unknown bank_feats_update: {self.config.train.bank_feats_update}')
            sim = None

        sim_batchwise_borders = torch.cat([torch.LongTensor([0]).to(device=vts2d_mask.device), vts2d_mask.sum(dim=1).cumsum(dim=0)], dim=0)
        sim_batchwise = torch.stack([sim[sim_batchwise_borders[b]:sim_batchwise_borders[b+1]].max(dim=-1)[0].mean() for b in range(len(sim_batchwise_borders)-1)], dim=0)
        # in case there are 0 vertices inside one image
        sim_batchwise[sim_batchwise.isnan()] = 0.
        results_batch['sim'] = sim_batchwise

        # loss: cross_entropy  # cross_entropy, nll_softmax, nll_clip, nll_affine_to_prob
        # bank_feats_update: loss_gradient  # loss_gradient, normalize_loss_gradient, moving_average, loss
        loss = self.criterion(sim / self.config.train.T, batch_vts_ids)
        if self.back_propagate:
            loss.backward()
        logger.info(f'loss {loss.item()}')
        results_batch['noise2d'] = noise2d
        results_batch['loss'] = loss[None,]
        results_batch['item_id'] = batch.item_id
        results_batch['name_unique'] = batch.name_unique
        results_batch['gt_cam_tform4x4_obj'] = batch.cam_tform4x4_obj

        return results_batch
    
    def inference_batch_single_view(self, batch, return_samples_with_sim=True):
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
            feats2d_net_mask = resize(batch.rgb_mask, H_out=feats2d_net.shape[2], W_out=feats2d_net.shape[3])
            if self.config.inference.use_mask_object:
                feats2d_net_mask = feats2d_net_mask * 1. * resize(batch.mask, H_out=feats2d_net.shape[2], W_out=feats2d_net.shape[3])

            time_pred_net_feats2d = time.time()
            # logger.info(
            #    f"predicted net feats2d, took {(time_pred_net_feats2d - time_loaded):.3f}")
            results['time_feats2d'] = torch.Tensor([time_pred_net_feats2d - time_loaded,]) / B

            meshes_scores = []
            for mesh_id in range(len(self.meshes)):
                # logger.info(f'calc score for mesh {self.config.categories[mesh_id]}')
                bank_feats = torch.cat([self.meshes.get_feats_with_mesh_id(mesh_id), self.clutter_feats.detach()],
                                       dim=0)
                # inner_feats2d_net_bank_vts_max_vals = torch.sum(net_feats2d[:, None] * bank_feats[None, :, :, None, None], dim=2, keepdim=True).max(dim=1).values
                out_shape = feats2d_net.shape[:1] + torch.Size([1]) + feats2d_net.shape[2:]
                inner_feats2d_net_bank_vts_max_vals = self.calc_sim('bchw,kc->bkhw', feats2d_net, bank_feats).max(dim=1,
                                                                                                                 keepdim=True).values
                sim, sim_pxl = self.get_sim_feats2d_net_with_cams(feats2d_net=feats2d_net,
                                                                  feats2d_net_mask=feats2d_net_mask,
                                                                  cam_tform4x4_obj=batch.cam_tform4x4_obj,
                                                                  cam_intr4x4=batch.cam_intr4x4,
                                                                  categories_ids= torch.tensor([mesh_id] * B), return_sim_pxl=True,
                                                                  broadcast_batch_and_cams=False,
                                                                  pre_rendered=False,
                                                                  only_use_rendered_inliers=self.config.inference.only_use_rendered_inliers,
                                                                  allow_clutter=self.config.inference.allow_clutter,
                                                                  use_sigmoid=self.config.inference.use_sigmoid)
                # sim (torch.Tensor): BxT, or Bx1 if rendered features is 4-dimensional.
                
                # inner_feats2d_net_bank_vts_max_vals, inner_feats2d_net_bank_vts_max_ids = inner_feats2d.max(dim=1)
                # show_img(self.meshes.get_verts_with_mesh_id[mesh_id][inner_feats2d_net_bank_vts_max_ids[0, 0]].permute(2, 0, 1), normalize=True)
                # show_img(inner_feats2d_net_bank_vts_max_vals[0])
                mesh_score = inner_feats2d_net_bank_vts_max_vals.flatten(1).mean(dim=1)
                # clutter_score = inner_feats2d[:, -clutter_feats.shape[0]:].mean(dim=1).flatten(1).mean(dim=1)
                # mesh_score -= clutter_score
                meshes_scores.append(sim.squeeze(1))
            meshes_scores = torch.stack(meshes_scores, dim=-1)
            pred_class_scores, pred_class_ids = meshes_scores.max(dim=1)

            # logger.info(f'pred class ids {pred_class_ids}')
            time_pred_class = time.time()
            # logger.info(f"predicted class: {self.config.categories[int(pred_class_ids[0])]}, took {(time_pred_class - time_pred_net_feats2d):.3f}")

            results['time_class'] = torch.Tensor([time_pred_class - time_pred_net_feats2d,]) / B

            b_cams_multiview_tform4x4_obj, b_cams_multiview_intr4x4 = self.get_samples(config_sample=self.config.inference.sample,
                                                                                       cam_intr4x4=batch.cam_intr4x4,
                                                                                       cam_tform4x4_obj=batch.cam_tform4x4_obj,
                                                                                       feats2d_net=feats2d_net,
                                                                                       categories_ids=batch.category_id,
                                                                                       feats2d_net_mask=feats2d_net_mask)

            # if self.config.inference.live:
            #     from od3d.cv.visual.show import show_imgs
            #     show_imgs(
            #         blend_rgb(batch.rgb[:1], (self.meshes.render_feats(cams_tform4x4_obj=b_cams_multiview_tform4x4_obj[0],
            #                                                           cams_intr4x4=b_cams_multiview_intr4x4[0],
            #                                                           imgs_sizes=batch.size,
            #                                                           meshes_ids=batch.category_id[:1],
            #                                                           modality=MESH_RENDER_MODALITIES.VERTS_NCDS,
            #                                                           broadcast_batch_and_cams=True)[0]).to(dtype=batch.rgb.dtype)), duration=-1)

            #  OPTION A: Use 2d gradient of rendered features
            sim = self.get_sim_feats2d_net_with_cams(
                feats2d_net=feats2d_net,
                feats2d_net_mask=feats2d_net_mask,
                cam_tform4x4_obj=b_cams_multiview_tform4x4_obj,
                cam_intr4x4=b_cams_multiview_intr4x4,
                categories_ids=batch.category_id,
                broadcast_batch_and_cams=True,
                only_use_rendered_inliers=self.config.inference.only_use_rendered_inliers,
                allow_clutter=self.config.inference.allow_clutter,
                use_sigmoid=self.config.inference.use_sigmoid
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
            # transl: 0, 1, 2 rot: 3, 4, 5
            optim_inference = torch.optim.Adam(
                params=[obj_tform6_tmp],
                lr=self.config.inference.optimizer.lr,
                betas=(self.config.inference.optimizer.beta0, self.config.inference.optimizer.beta1),
            )

            time_before_pose_iterative = time.time()
            cam_tform4x4_obj = tform4x4(cam_tform4x4_obj.detach(), se3_exp_map(obj_tform6_tmp))

            refine_update_max = self.refine_update_max[batch.category_id].clone()
            for epoch in range(self.config.inference.optimizer.epochs):

                cam_tform4x4_obj = tform4x4(cam_tform4x4_obj.detach(), se3_exp_map(obj_tform6_tmp.detach()))
                obj_tform6_tmp.data[:, :] = 0.
                cam_tform4x4_obj = tform4x4(cam_tform4x4_obj.detach(), se3_exp_map(obj_tform6_tmp))

                sim, sim_pxl = self.get_sim_feats2d_net_with_cams(feats2d_net=feats2d_net,
                                                                  feats2d_net_mask=feats2d_net_mask,
                                                                  cam_tform4x4_obj=cam_tform4x4_obj,
                                                                  cam_intr4x4=batch.cam_intr4x4,
                                                                  categories_ids=batch.category_id, return_sim_pxl=True,
                                                                  broadcast_batch_and_cams=False,
                                                                  pre_rendered=False,
                                                                  only_use_rendered_inliers=self.config.inference.only_use_rendered_inliers,
                                                                  allow_clutter=self.config.inference.allow_clutter,
                                                                  use_sigmoid=self.config.inference.use_sigmoid)
                mesh_cam_loss = -sim

                if self.config.inference.live:
                    show_img(
                        blend_rgb(batch.rgb[0], (self.meshes.render_feats(cams_tform4x4_obj=cam_tform4x4_obj[0:0 + 1],
                                                                          cams_intr4x4=batch.cam_intr4x4[0:0 + 1],
                                                                          imgs_sizes=batch.size,
                                                                          meshes_ids=batch.category_id[0:0 + 1],
                                                                          modality=MESH_RENDER_MODALITIES.VERTS_NCDS)[
                            0]).to(dtype=batch.rgb.dtype)), duration=1)

                loss = mesh_cam_loss.sum()
                loss.backward()
                optim_inference.step()
                optim_inference.zero_grad()

                # detach update
                obj_tform6_tmp.data[:, self.config.inference.refine.dims_detached] = 0.
                # clip update
                refine_update_mask = obj_tform6_tmp.data.abs() > refine_update_max
                obj_tform6_tmp.data[refine_update_mask] = obj_tform6_tmp.data[refine_update_mask].sign() * refine_update_max[refine_update_mask]


            cam_tform4x4_obj = tform4x4(cam_tform4x4_obj.detach(), se3_exp_map(obj_tform6_tmp.detach()))

            results['time_pose_iterative'] = torch.Tensor([time.time() - time_before_pose_iterative,]) / B

        cam_tform4x4_obj = cam_tform4x4_obj.clone().detach()

        results['time_pose'] = torch.Tensor([time.time() - time_pred_class,]) / B
        batch_rot_diff_rad = get_pose_diff_in_rad(pred_tform4x4=cam_tform4x4_obj, gt_tform4x4=batch.cam_tform4x4_obj)
        results['rot_diff_rad'] = batch_rot_diff_rad
        for cat_id, cat in enumerate(self.config.categories):
            results[f'{cat}_rot_diff_rad'] = batch_rot_diff_rad[batch.category_id == cat_id]
        results['label_gt'] = batch.category_id
        results['label_pred'] = pred_class_ids
        results['label_names'] = self.config.categories
        results['sim'] = sim
        results['cam_tform4x4_obj'] = cam_tform4x4_obj
        results['item_id'] = batch.item_id
        results['name_unique'] = batch.name_unique

        return results

        

   

