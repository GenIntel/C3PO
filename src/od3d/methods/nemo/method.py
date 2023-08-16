import time
from typing import List

import od3d.io
from od3d.methods.method import OD3D_Method
from od3d.datasets.dataset import OD3D_Dataset
from od3d.benchmark.results import OD3D_Results
from omegaconf import DictConfig
import pytorch3d.transforms
import pandas as pd
import numpy as np
from torch.utils.data import RandomSampler
import logging

logger = logging.getLogger(__name__)
import torch
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
from od3d.cv.geometry.fit3d2d import batchwise_fit_se3_to_corresp_3d_2d_and_masks  # fit_se3_to_corresp_3d_2d_and_masks
from od3d.cv.transforms import RandomCenterZoom3D, RGB_Random, CenterZoom3D
import math

from typing import Dict
from od3d.data.ext_enum import ExtEnum
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

class NeMo(OD3D_Method):
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


        if config.train.transform.random_color:
            self.transform_train = torchvision.transforms.Compose([
                RandomCenterZoom3D(**config.train.transform.random_center_zoom3d),
                RGB_Random(),
                self.net.transform,
            ])
        else:
            self.transform_train = torchvision.transforms.Compose([
                RandomCenterZoom3D(**config.train.transform.random_center_zoom3d),
                self.net.transform,
            ])

        self.transform_test = torchvision.transforms.Compose([
            CenterZoom3D(**config.test.transform),
            self.net.transform
        ])

        # init Meshes / Features
        self.total_params = sum(p.numel() for p in self.net.parameters())
        # self.path_shapenemo = Path(config.path_shapenemo)
        # self.fpaths_meshes_shapenemo = [self.path_shapenemo.joinpath(cls, '01.off') for cls in config.classes]
        self.fpaths_meshes = [self.config.fpaths_meshes[cls] for cls in config.classes]
        self.meshes = Meshes.load_from_files(fpaths_meshes=self.fpaths_meshes)
        # self.meshes.show()
        self.verts_count_max = self.meshes.verts_counts_max
        self.mem_verts_feats_count = len(config.classes) * self.verts_count_max
        self.mem_clutter_feats_count = config.num_noise * config.max_group
        self.mem_count = self.mem_verts_feats_count + self.mem_clutter_feats_count

        self.clutter_feats = torch.nn.Parameter(torch.randn(size=(1, self.net.out_dim), device=self.device),
                                                requires_grad=True)
        self.meshes.set_feats_cat_with_pad(torch.nn.Parameter(
            torch.randn(size=(self.verts_count_max * len(self.meshes), self.net.out_dim), device=self.device),
            requires_grad=True))
        # self.meshes.set_feats_cat_with_pad(torch.nn.Parameter(torch.randn(size=(self.verts_count_max * len(self.meshes), self.net.feat_dim), device=self.device), requires_grad=True))

        # dict to save estimated tforms, sequence : tform,
        self.seq_obj_tform4x4_est_obj = {}
        self.seq_obj_tform4x4_est_obj_sim = {}

        self.normalize_feats()

        if self.config.train.loss == 'cross_entropy':
            self.criterion = torch.nn.CrossEntropyLoss().cuda()
        elif self.config.train.loss == 'nll_softmax':
            self.softmax = torch.nn.LogSoftmax(dim=1)
            self.criterion = torch.nn.NLLLoss().cuda()
        elif self.config.train.loss == 'nll_clip':
            self.criterion = torch.nn.NLLLoss().cuda()
        elif self.config.train.loss == 'nll_affine_to_prob':
            self.criterion = torch.nn.NLLLoss().cuda()
        elif self.config.train.loss == 'l2':
            self.criterion = torch.nn.MSELoss().cuda()
        elif self.config.train.loss == 'l2_squared':
            self.criterion = torch.nn.MSELoss().cuda()


        # self.net = torch.nn.DataParallel(self.net).cuda()
        self.net.cuda()
        self.meshes.cuda()
        self.net.eval()

        self.optim = od3d.io.get_obj_from_config(config=self.config.train.optimizer, params=list(self.net.parameters()) + [self.meshes.feats] + [self.clutter_feats])
        self.scheduler = od3d.io.get_obj_from_config(self.optim, config=self.config.train.scheduler)

        # load checkpoint
        if config.get("checkpoint", None) is not None:
            self.load_checkpoint(config.checkpoint)
        elif config.get("checkpoint_old", None) is not None:
            self.load_checkpoint_old(config.checkpoint_old)
        # load_mesh(config.path_shapenemo)

        # self.meshes.show()

        # self.verts_feats = checkpoint["memory"][:self.mem_verts_feats_count].clone().detach().cpu()
        # note: somehow vertices are stored in wrong order of classes (starting with last class tvmonitor until first class aeroplane
        # self.verts_feats = self.verts_feats.reshape(len(self.meshes), self.verts_count_max, -1).flip(dims=(0,)).reshape(len(self.meshes) * self.verts_count_max, -1)
        self.down_sample_rate = self.config.down_sample_rate

    def normalize_feats(self):
        self.clutter_feats.data = self.clutter_feats.detach() / self.clutter_feats.detach().norm(dim=-1, keepdim=True)
        self.meshes.feats.data = self.meshes.feats.detach() / self.meshes.feats.detach().norm(dim=-1, keepdim=True)
        # self.meshes.set_feats_cat(self.meshes.feats.detach() / self.meshes.feats.detach().norm(dim=-1, keepdim=True))
        # logger.info(self.clutter_feats[:1])
        # logger.info(self.meshes.feats[:1])

    def load_checkpoint_old(self, path_checkpoint):
        fpaths_meshes_old = list(self.config.fpaths_meshes.values())
        meshes_old = Meshes.load_from_files(fpaths_meshes=fpaths_meshes_old)
        verts_count_max = meshes_old.verts_counts_max
        mem_verts_feats_count = len(fpaths_meshes_old) * verts_count_max
        checkpoint = torch.load(path_checkpoint, map_location="cuda:0")
        self.net.net = torch.nn.DataParallel(self.net.net).cuda()
        self.net.net.load_state_dict(checkpoint["state"], strict=False)
        self.net.net = self.net.net.module
        self.clutter_feats = checkpoint["memory"][mem_verts_feats_count:].clone().detach().cpu()
        # self.clutter_feats = self.clutter_feats.mean(dim=0, keepdim=True)
        self.clutter_feats = torch.nn.Parameter(self.clutter_feats.to(device=self.device), requires_grad=True)

        verts_feats = []
        map_mesh_id_to_old_id = [fpaths_meshes_old.index(fpath_mesh) for fpath_mesh in self.fpaths_meshes]
        for i in range(len(self.fpaths_meshes)):
            mesh_old_id = map_mesh_id_to_old_id[i]
            verts_feats.append(checkpoint["memory"][mesh_old_id * verts_count_max: (mesh_old_id + 1) * verts_count_max].clone().detach().cpu())
        self.meshes.set_feats_cat_with_pad(torch.cat(verts_feats, dim=0))

    def save_checkpoint(self, path_checkpoint: Path):
        torch.save({
            'net_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'meshes_feats': self.meshes.feats,
            'clutter_feats': self.clutter_feats
        }, path_checkpoint)

    def load_checkpoint(self, path_checkpoint):
        checkpoint = torch.load(path_checkpoint)
        self.net.load_state_dict(checkpoint['net_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.meshes.set_feats_cat(checkpoint['meshes_feats'])
        self.clutter_feats = checkpoint['clutter_feats']

    @property
    def path_checkpoint(self):
        return self.logging_dir.joinpath('nemo.ckpt')

    def train(self, datasets_train: Dict[str, OD3D_Dataset], datasets_val: Dict[str, OD3D_Dataset]):
        score_metric_name = 'pose/acc_pi18'  # 'pose/acc_pi18' 'pose/acc_pi6'
        score_ckpt_val = 0.
        score_latest = 0.

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

                if score_latest > score_ckpt_val:
                    score_ckpt_val = score_latest
                    self.save_checkpoint(path_checkpoint=self.path_checkpoint)

            results_epoch = self.train_epoch(dataset=dataset_train_sub)
            results_epoch.log_with_prefix('train')
        self.load_checkpoint(path_checkpoint=self.path_checkpoint)


    def test(self, dataset: OD3D_Dataset, config_inference: DictConfig = None, pose_iterative_refine=True):
        logger.info(f'test dataset {dataset.name}')
        if config_inference is None:
            config_inference = self.config.inference
        self.net.eval()
        self.meshes.feats.requires_grad = False
        clutter_feats = self.clutter_feats.detach()
        dataset.transform = self.transform_test

        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.config.test.dataloader.batch_size,
                                                 shuffle=False,
                                                 collate_fn=dataset.collate_fn,
                                                 num_workers=self.config.test.dataloader.num_workers,
                                                 pin_memory=self.config.test.dataloader.pin_memory)

        logger.info(f"Dataset contains {len(dataset)} frames.")

        results_epoch = OD3D_Results()
        for i, batch in tqdm(enumerate(iter(dataloader))):
            batch.to(device=self.device)

            results_batch = self.inference_batch(batch=batch)
            results_epoch += results_batch

        count_pred_frames = len(results_epoch['item_id'])
        logger.info(f'Predicted {count_pred_frames} frames.')

        results_visual = self.get_results_visual(results_epoch=results_epoch, dataset=dataset,
                                                 config_visualize=self.config.test.visualize)
        results_epoch = results_epoch.mean()
        results_epoch += results_visual
        return results_epoch


    def train_epoch(self, dataset: OD3D_Dataset) -> OD3D_Results:
        self.net.train()
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
            if accumulate_steps % self.config.train.batch_accumulate_to_next_step == 0:
                self.optim.step()
                self.normalize_feats()
                self.optim.zero_grad()

            results_epoch += results_batch

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

        # logger.info(f'batch.label {batch.label}')
        # B x x N x 2
        vts2d, vts2d_mask = self.meshes.verts2d(cams_intr4x4=batch.cam_intr4x4,
                                                cams_tform4x4_obj=batch.cam_tform4x4_obj,
                                                imgs_sizes=batch.size, mesh_ids=batch.label,
                                                down_sample_rate=self.down_sample_rate)

        N = vts2d.shape[1]
        # B x F+N x C
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
        prob_noise[prob_noise.sum(dim=-1) == 0.] = 1.
        noise2d = xy.flatten(1)[:, torch.multinomial(prob_noise, self.config.num_noise)].permute(1, 2, 0)
        vts2d_feats2d_net_mask = sample_pxl2d_pts(feats2d_net_mask, pxl2d=torch.cat([vts2d], dim=1))
        vts2d_mask = vts2d_mask * (vts2d_feats2d_net_mask[:, :, 0] > 0.5)
        net_feats = sample_pxl2d_pts(feats2d_net, pxl2d=torch.cat([vts2d, noise2d], dim=1))

        C = net_feats.shape[2]
        # args: X: Bx3xHxW, keypoint_positions: BxNx2, obj_mask: BxHxW ensures that noise is sampled outside of object mask
        # returns: BxF+NxC


        # net_feats = net_feats[:, :].reshape(-1, net_feats.shape[-1])
        batch_vts_ids = self.meshes.get_verts_and_noise_ids_stacked(batch.label.tolist(),
                                                                    count_noise_ids=self.config.num_noise)

        # weighting with similarity score
        # net_feats = net_feats * (batch.cam_tform4x4_obj_sim[:, None, None] ** 4)

        # sim_weight = batch.cam_tform4x4_obj_sim[:, None].expand(*net_feats.shape[:2])
        # sim_weight = torch.cat([sim_weight[:, :N][mask_vts2d_vsbl], sim_weight[:, N:].reshape(-1)], dim=0)

        batch_vts_ids = torch.cat([batch_vts_ids[:, :N][vts2d_mask], batch_vts_ids[:, N:].reshape(-1)],
                                  dim=0)
        net_feats = torch.cat([net_feats[:, :N][vts2d_mask], net_feats[:, N:].reshape(-1, C)], dim=0)

        # batch_vts_ids = self.meshes.get_feats_ids_stacked(batch.label.tolist())

        bank_feats = torch.cat([self.meshes.feats, self.clutter_feats], dim=0)


        if self.config.train.bank_feats_update == 'loss_gradient':
            sim = torch.einsum('nc,vc->nv', net_feats, bank_feats)
        elif self.config.train.bank_feats_update == 'normalize_loss_gradient':
            sim = torch.einsum('nc,vc->nv', net_feats, torch.nn.functional.normalize(bank_feats, dim=1))
        elif self.config.train.bank_feats_update == 'moving_average':
            sim = torch.einsum('nc,vc->nv', net_feats, bank_feats.detach())
            bank_feats_new = self.config.train.alpha * bank_feats[batch_vts_ids].detach() + (1. - self.config.train.alpha) * net_feats.detach()
            batch_vts_ids_unique, batch_vts_ids_unique_inverse, batch_vts_ids_unique_counts = batch_vts_ids.unique(return_inverse=True, return_counts=True)
            bank_feats_new = torch.einsum('nk,nc->kc', torch.nn.functional.one_hot(batch_vts_ids_unique_inverse).to(dtype= bank_feats_new.dtype, device= bank_feats_new.device), bank_feats_new) / batch_vts_ids_unique_counts[:, None]
            bank_feats[batch_vts_ids_unique].data = bank_feats_new
        else:
            logger.error(f'unknown bank_feats_update: {self.config.train.bank_feats_update}')
            sim = None

        sim_batchwise_borders = torch.cat([torch.LongTensor([0]).to(device=vts2d_mask.device), vts2d_mask.sum(dim=1).cumsum(dim=0)], dim=0)
        sim_batchwise = torch.stack([sim[sim_batchwise_borders[b]:sim_batchwise_borders[b+1]].max(dim=-1)[0].mean() for b in range(len(sim_batchwise_borders)-1)], dim=0)
        results_batch['sim'] = sim_batchwise

        # loss: cross_entropy  # cross_entropy, nll_softmax, nll_clip, nll_affine_to_prob
        # bank_feats_update: loss_gradient  # loss_gradient, normalize_loss_gradient, moving_average, loss
        loss = self.criterion(sim / self.config.train.T, batch_vts_ids)

        loss.backward()
        logger.info(f'loss {loss.item()}')

        results_batch['loss'] = loss[None,]

        results_batch['item_id'] = batch.item_id
        results_batch['name_unique'] = batch.name_unique

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
            results['time_feats2d'] = torch.Tensor([time_pred_net_feats2d - time_loaded,])

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

            results['time_class'] = torch.Tensor([time_pred_class - time_pred_net_feats2d,])

            b_cams_multiview_tform4x4_obj, b_cams_multiview_intr4x4 = self.get_samples(config_sample=self.config.inference.sample,
                                                                                       cam_intr4x4=batch.cam_intr4x4,
                                                                                       cam_tform4x4_obj=batch.cam_tform4x4_obj,
                                                                                       feats2d_net=feats2d_net,
                                                                                       categories_ids=pred_class_ids,
                                                                                       feats2d_net_mask=feats2d_net_mask)
            #  OPTION A: Use 2d gradient of rendered features
            sim = self.get_sim_feats2d_net_with_cams(
                feats2d_net=feats2d_net,
                feats2d_net_mask=feats2d_net_mask,
                cam_tform4x4_obj=b_cams_multiview_tform4x4_obj,
                cam_intr4x4=b_cams_multiview_intr4x4,
                categories_ids=pred_class_ids,
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

        if self.config.inference.pose_iterative_refine:
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
                if self.config.inference.sample.method == 'uniform':
                    obj_tform6_tmp.data[:, :3] = 0.
                cam_tform4x4_obj = tform4x4(cam_tform4x4_obj.detach(), se3_exp_map(obj_tform6_tmp.detach()))
                obj_tform6_tmp.data[:, :] = 0.
                cam_tform4x4_obj = tform4x4(cam_tform4x4_obj.detach(), se3_exp_map(obj_tform6_tmp))

                sim, sim_pxl = self.get_sim_feats2d_net_with_cams(feats2d_net=feats2d_net,
                                                                  feats2d_net_mask=feats2d_net_mask,
                                                                  cam_tform4x4_obj=cam_tform4x4_obj,
                                                                  cam_intr4x4=batch.cam_intr4x4,
                                                                  categories_ids=pred_class_ids, return_sim_pxl=True,
                                                                  broadcast_batch_and_cams=False,
                                                                  pre_rendered=False)
                mesh_cam_loss = -sim

                if self.config.inference.live:
                    show_img(
                        blend_rgb(batch.rgb[0], (self.meshes.render_feats(cams_tform4x4_obj=cam_tform4x4_obj[0:0 + 1],
                                                                          cams_intr4x4=batch.cam_intr4x4[0:0 + 1],
                                                                          imgs_sizes=batch.size,
                                                                          meshes_ids=pred_class_ids[0:0 + 1],
                                                                          modality=MESH_RENDER_MODALITIES.VERTS_NCDS)[
                            0]).to(dtype=batch.rgb.dtype)), duration=1)

                loss = mesh_cam_loss.mean()
                loss.backward()
                optim_inference.step()
                optim_inference.zero_grad()

            cam_tform4x4_obj = tform4x4(cam_tform4x4_obj.detach(), se3_exp_map(obj_tform6_tmp.detach()))

            results['time_pose_iterative'] = torch.Tensor([time.time() - time_before_pose_iterative,])

        cam_tform4x4_obj = cam_tform4x4_obj.clone().detach()

        results['time_pose'] = torch.Tensor([time.time() - time_pred_class,])

        diff_rot3x3 = rot3x3(batch.cam_tform4x4_obj[:, :3, :3].permute(0, 2, 1), cam_tform4x4_obj[:, :3, :3])

        try:
            diff_so3_log = pytorch3d.transforms.so3_log_map(diff_rot3x3.permute(0, 2, 1))
            diff_rot_angle_rad = torch.norm(diff_so3_log, dim=-1)
        except ValueError:
            logger.warning(
                f'Cannot calculate deviation in rotation angle due to rot3x3 trace being too small, setting deviation to 0.')
            diff_rot_angle_rad = 0.
        results['rot_diff_rad'] = diff_rot_angle_rad
        results['label_gt'] = batch.label
        results['label_pred'] = pred_class_ids
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
            group_ids = [ group_id for group_id, group_name in enumerate(group_names) for name_unique in results_epoch['name_unique'] if name_unique.startswith(group_name)]
            df = pd.DataFrame(np.stack([epoch_ranked_ids.detach().cpu().numpy(), np.array(group_ids)], axis=-1), columns=['rank', 'group'])
            epoch_ranked_ids = torch.from_numpy(df.loc[df.groupby("group").cumcount().sort_values(kind='mergesort').index]['rank'].values)


        epoch_best_ids = epoch_ranked_ids[:count_best]
        epoch_best_names = [f'best/{i+1}' for i in range(len(epoch_best_ids))]
        epoch_worst_ids = epoch_ranked_ids[-count_worst:]
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
                batch_sel_names = [dict_name_unique_to_sel_name[batch.name_unique[b]] for b in range(B)]
                batch_sel_scores = []
                for b in range(B):
                    batch_sel_scores.append('\n'.join([f'{metric}={results_epoch[metric].to(device=self.device)[batch_result_ids[b]].cpu().detach().item():.3f}' for metric in caption_metrics if metric in results_epoch.keys()]))

                feats2d_net = self.net(batch.rgb)
                #feats2d_net = resize(feats2d_net,
                #                     scale_factor=self.down_sample_rate / config_visualize.down_sample_rate)

                if VISUAL_MODALITIES.NET_FEATS_NEAREST_VERTS in modalities:

                    logger.info('create net_feats_nearest_verts ...')
                    verts3d = self.get_nearest_verts3d_to_feats2d_net(feats2d_net=feats2d_net, categories_ids=batch.label,
                                                                      zero_if_sim_clutter_larger=True)
                    verts3d = resize(verts3d, scale_factor=self.down_sample_rate / config_visualize.down_sample_rate)
                    for b in range(len(batch)):
                        img = blend_rgb(resize(batch.rgb[b], scale_factor=1. / config_visualize.down_sample_rate), verts3d[b])
                        results[f'visual/{batch_sel_names[b]}_{VISUAL_MODALITIES.NET_FEATS_NEAREST_VERTS}'] = image_as_wandb_image(img, caption=f'{batch_sel_names[b]}, {batch_sel_scores[b]}')
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
                                                  categories_ids=batch.label, down_sample_rate=config_visualize.down_sample_rate, broadcast_batch_and_cams=True)




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
                            import matplotlib.pyplot as plt
                            plt.ioff()
                            fig, ax = plt.subplots()
                            ax.plot(imgs_sim.detach().cpu().numpy(), label='sim')  # density=False would make counts
                            #ax.set_ylim(0., 1.)
                            #ax.ylabel('sim')
                            #ax.xlabel('samples')
                            from od3d.cv.visual.show import get_img_from_plot
                            from od3d.cv.visual.draw import draw_text_in_rgb
                            img = get_img_from_plot(ax=ax, fig=fig)
                            plt.close(fig)
                            img = resize(img, H_out=imgs.shape[-2], W_out=imgs.shape[-2])
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
                        results[f'visual/{batch_sel_names[b]}_{VISUAL_MODALITIES.SAMPLES}'] = image_as_wandb_image(img, caption=f'{batch_sel_names[b]}, {batch_sel_scores[b]}, min={imgs_sim.min().item():.3f}, max={imgs_sim.max().item():.3f}')
                        if live:
                            show_img(img)


                if VISUAL_MODALITIES.SIM_PXL in modalities:
                    logger.info('create sim pxl...')
                    batch_pred_label = results_epoch['label_pred'].to(device=self.device)[batch_result_ids]
                    batch_pred_cam_tform4x4 = results_epoch['cam_tform4x4_obj'].to(device=self.device)[batch_result_ids]
                    sim, sim_pxl = self.get_sim_feats2d_net_with_cams(feats2d_net=feats2d_net,
                                                                      cam_intr4x4=batch.cam_intr4x4,
                                                                      cam_tform4x4_obj=batch_pred_cam_tform4x4,
                                                                      categories_ids=batch_pred_label, return_sim_pxl=True,
                                                                      broadcast_batch_and_cams=False,
                                                                      sim_feats_mesh_with_image=SIM_FEATS_MESH_WITH_IMAGE.RENDERED,
                                                                      pre_rendered=False)

                    sim_pxl = resize(sim_pxl, scale_factor=self.down_sample_rate / config_visualize.down_sample_rate)
                    for b in range(len(batch)):
                        img = blend_rgb(resize(batch.rgb[b], scale_factor=1. / config_visualize.down_sample_rate),
                                        sim_pxl[b])
                        results[f'visual/{batch_sel_names[b]}_{VISUAL_MODALITIES.SIM_PXL}'] = image_as_wandb_image(img, caption=f'{batch_sel_names[b]}, mean sim={sim[b].item()}')
                        if live:
                            show_img(img)


                if VISUAL_MODALITIES.PRED_VERTS_NCDS_IN_RGB in modalities or VISUAL_MODALITIES.PRED_VS_GT_VERTS_NCDS_IN_RGB in modalities:
                    logger.info('create pred verts ncds...')
                    batch_pred_label = results_epoch['label_pred'].to(device=self.device)[batch_result_ids]
                    batch_pred_cam_tform4x4 = results_epoch['cam_tform4x4_obj'].to(device=self.device)[batch_result_ids]
                    pred_verts_ncds = self.get_ncds_with_cam(cam_intr4x4=batch.cam_intr4x4, cam_tform4x4_obj=batch_pred_cam_tform4x4, categories_ids=batch_pred_label, size=batch.size, down_sample_rate=config_visualize.down_sample_rate, pre_rendered=False)
                    if VISUAL_MODALITIES.PRED_VERTS_NCDS_IN_RGB in modalities:
                        for b in range(len(batch)):
                            img = blend_rgb(resize(batch.rgb[b], scale_factor=1. / config_visualize.down_sample_rate),
                                            pred_verts_ncds[b])
                            results[f'visual/{batch_sel_names[b]}_{VISUAL_MODALITIES.PRED_VERTS_NCDS_IN_RGB}'] = image_as_wandb_image(img, caption=f'{batch_sel_names[b]}, {batch_sel_scores[b]}')
                            if live:
                                show_img(img)

                if VISUAL_MODALITIES.GT_VERTS_NCDS_IN_RGB in modalities or VISUAL_MODALITIES.PRED_VS_GT_VERTS_NCDS_IN_RGB in modalities:
                    logger.info('create gt verts ncds...')
                    gt_verts_ncds = self.get_ncds_with_cam(cam_intr4x4=batch.cam_intr4x4,
                                                  cam_tform4x4_obj=batch.cam_tform4x4_obj,
                                                  categories_ids=batch.label, size=batch.size,
                                                  down_sample_rate=config_visualize.down_sample_rate, pre_rendered=False)
                    if VISUAL_MODALITIES.GT_VERTS_NCDS_IN_RGB in modalities:
                        for b in range(len(batch)):
                            img = blend_rgb(resize(batch.rgb[b], scale_factor=1. / config_visualize.down_sample_rate),
                                            gt_verts_ncds[b])
                            results[
                                f'visual/{batch_sel_names[b]}_{VISUAL_MODALITIES.GT_VERTS_NCDS_IN_RGB}'] = image_as_wandb_image(
                                img, caption=f'{batch_sel_names[b]}, {batch_sel_scores[b]}')
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
                            img, caption=f'{batch_sel_names[b]}, {batch_sel_scores[b]}')
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