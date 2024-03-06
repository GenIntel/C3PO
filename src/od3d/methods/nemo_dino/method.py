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
        self.mesh_update_count = torch.ones(size=(self.meshes.feats.shape[0] + self.clutter_feats.shape[0],), device=self.device)
        # hard coded for now
        self.mesh_feats_total = torch.zeros(size=(self.meshes.feats.shape[0] + self.clutter_feats.shape[0],384), device=self.device)

    @staticmethod
    def count_parameters(model):
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            logger.info(f"{name}: {params}")
            total_params += params
        logger.info(f"Total Trainable Params: {total_params}")

    def train(self, datasets_train: Dict[str, OD3D_Dataset], datasets_val: Dict[str, OD3D_Dataset]):
        score_metric_name = 'pose/acc_pi18'  # 'pose/acc_pi18' 'pose/acc_pi6'
        score_ckpt_val = 0.
        score_latest = 0.
        self.save_checkpoint(path_checkpoint=self.path_checkpoint)
        logger.info(f"net parameters")
        self.count_parameters(self.net)
        logger.info(f"mesh parameters")
        self.count_parameters(self.meshes)
        if 'main' in datasets_val.keys():
            dataset_train_sub = datasets_train['labeled']
        else:
            dataset_train_sub, dataset_val_sub = datasets_train['labeled'].get_split(fraction1=1. - self.config.train.val_fraction,
                                                                                     fraction2=self.config.train.val_fraction,
                                                                                     split=self.config.train.split)
            datasets_val['main'] = dataset_val_sub

        for epoch in range(self.config.train.epochs):
            if self.config.train.val and self.config.train.epochs_to_next_test > 0 and epoch % self.config.train.epochs_to_next_test == 0:
                for dataset_val_key, dataset_val in datasets_val.items():
                    results_val = self.test(dataset_val)
                    results_val.log_with_prefix(prefix=f'val/{dataset_val.name}')
                    if dataset_val_key == 'main':
                        score_latest = results_val[score_metric_name]

                if not self.config.train.early_stopping or score_latest > score_ckpt_val:
                    score_ckpt_val = score_latest
                    self.save_checkpoint(path_checkpoint=self.path_checkpoint)

            results_epoch = self.train_epoch(dataset=dataset_train_sub)
            results_epoch.log_with_prefix('train')
        self.load_checkpoint(path_checkpoint=self.path_checkpoint)
        

    def train_epoch(self, dataset: OD3D_Dataset) -> OD3D_Results:
        # change to train mode later
        self.net.train()
        self.meshes.del_pre_rendered()
        self.meshes.feats.requires_grad = True
        dataset.transform = self.transform_train
        dataloader_train = torch.utils.data.DataLoader(dataset=dataset,
                                                       batch_size=self.config.train.dataloader.batch_size,
                                                       shuffle=True,
                                                       collate_fn=dataset.collate_fn,
                                                       num_workers=self.config.train.dataloader.num_workers,
                                                       pin_memory=self.config.train.dataloader.pin_memory)

        results_epoch = OD3D_Results()
        accumulate_steps = 0
        for i, batch in enumerate(iter(dataloader_train)):
            results_batch: OD3D_Results = self.train_batch(batch=batch)
            results_batch.log_with_prefix('train')
            accumulate_steps += 1
            if (accumulate_steps % self.config.train.batch_accumulate_to_next_step) == 0:
                if (self.config.train.bank_feats_update != 'average'):
                    self.optim.step()
                    self.normalize_feats()
                    self.optim.zero_grad()

            results_epoch += results_batch
        if (self.config.train.bank_feats_update != 'average'):
            self.scheduler.step()
            self.optim.zero_grad()

        results_visual = self.get_results_visual(results_epoch=results_epoch, dataset=dataset,
                                                 config_visualize=self.config.train.visualize)
        results_epoch = results_epoch.mean()
        results_epoch += results_visual
        return results_epoch
    

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
            bank_feats = self.mesh_feats_total/ self.mesh_update_count[:, None]
            self.meshes.feats.data = bank_feats[:self.meshes.feats.shape[0]]
            self.clutter_feats.data = bank_feats[self.meshes.feats.shape[0]:]
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
        if self.config.train.bank_feats_update != 'average':
            loss.backward()
        logger.info(f'loss {loss.item()}')
        results_batch['noise2d'] = noise2d
        results_batch['loss'] = loss[None,]
        results_batch['item_id'] = batch.item_id
        results_batch['name_unique'] = batch.name_unique
        results_batch['gt_cam_tform4x4_obj'] = batch.cam_tform4x4_obj

        return results_batch

        

   

