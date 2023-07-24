import time
from typing import List
from od3d.methods.method import OD3D_Method
from od3d.datasets.dataset import OD3D_Dataset
from od3d.datasets.frames import OD3D_Frames
from od3d.benchmark.results import OD3D_Results
from omegaconf import DictConfig
import pytorch3d.transforms
from od3d.cv.geometry.transform import se3_log_map

from torch.utils.data import RandomSampler
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
from od3d.cv.visual.show import show_imgs, show_img
import torchvision
from od3d.cv.visual.blend import blend_rgb
from od3d.cv.geometry.transform import transf4x4_from_pos_and_theta
from sklearn.metrics import confusion_matrix
from od3d.cv.differentiation.gradient import calc_batch_gradients
from od3d.cv.visual.sample import sample_pxl2d_pts
from tqdm import tqdm
from od3d.cv.geometry.mesh import MESH_RENDER_MODALITIES

from od3d.cv.io import image_as_wandb_image
from od3d.cv.visual.resize import resize
from od3d.methods.nemo.backbone import OD3D_Backbone
from functools import partial

from od3d.cv.geometry.grid import get_pxl2d_like
from od3d.cv.geometry.fit3d2d import batchwise_fit_se3_to_corresp_3d_2d_and_masks  # fit_se3_to_corresp_3d_2d_and_masks
from od3d.cv.geometry.transform import inv_tform4x4
from od3d.cv.transforms import RandomCenterZoom3D, RGB_Random, CenterZoom3D

from od3d.data.ext_enum import ExtEnum
class VISUAL_MODALITIES(str, ExtEnum):
    VERTS_NCDS_IN_RGB = 'verts_ncds_in_rgb'
    GT_VERTS_NCDS_IN_RGB = 'gt_verts_ncds_in_rgb'
    NET_FEATS_NEAREST_VERTS = 'net_feats_nearest_verts'
    SIM_PXL = 'sim_pxl'
    SAMPLES = 'samples'

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
        self.net = OD3D_Backbone.subclasses[config.backbone.class_name](config.backbone)

        self.transform_train = torchvision.transforms.Compose([
            RandomCenterZoom3D(**config.train.transform),
            RGB_Random() if config.train.transform.color_random else None,
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

        self.clutter_feats = torch.nn.Parameter(torch.randn(size=(1, self.net.feat_dim), device=self.device),
                                                requires_grad=True)
        self.meshes.set_feats_cat_with_pad(torch.nn.Parameter(
            torch.randn(size=(self.verts_count_max * len(self.meshes), self.net.feat_dim), device=self.device),
            requires_grad=True))
        # self.meshes.set_feats_cat_with_pad(torch.nn.Parameter(torch.randn(size=(self.verts_count_max * len(self.meshes), self.net.feat_dim), device=self.device), requires_grad=True))

        # dict to save estimated tforms, sequence : tform,
        self.seq_obj_tform4x4_est_obj = {}
        self.seq_obj_tform4x4_est_obj_sim = {}

        self.normalize_feats()
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        # self.net = torch.nn.DataParallel(self.net).cuda()
        self.net.cuda()
        self.meshes.cuda()
        self.net.eval()

        self.optim = torch.optim.Adam(list(self.net.parameters()) + [self.meshes.feats] + [self.clutter_feats],
                                      lr=self.config.train.optimizer.lr,  #
                                      weight_decay=self.config.train.optimizer.weight_decay)  #
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, gamma=self.config.train.scheduler.gamma,
                                                              milestones=self.config.train.scheduler.milestones)

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

    def save_checkpoint(self, path_checkpoint):
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

    def train(self, dataset: OD3D_Dataset, datasets_val: List[OD3D_Dataset]):
        score_metric_name = 'pose/acc_pi6'
        score_ckpt_val = 0.
        score_latest = 0.

        train_dataset_sub, val_dataset_sub = dataset.get_split(fraction1=1.-self.config.train.val_fraction,
                                                               fraction2=self.config.train.val_fraction)

        for epoch in range(self.config.train.epochs):
            if self.config.train.val and self.config.train.epochs_to_next_test > 0 and epoch % self.config.train.epochs_to_next_test == 0:
                for dataset_val in datasets_val + [val_dataset_sub]:
                    results_val = self.test(dataset_val)
                    results_val.log_with_prefix(prefix=f'val/{dataset_val.name}')
                    score_latest = results_val[score_metric_name]

                if score_latest > score_ckpt_val:
                    score_ckpt_val = score_latest
                    self.save_checkpoint(path_checkpoint=self.path_checkpoint)

            self.net.train()
            self.meshes.feats.requires_grad = True
            results_epoch = self.train_epoch(dataset=train_dataset_sub)
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
                                                 rank_metric_name='rot_diff_rad',
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
            accumulate_steps += 1
            if accumulate_steps % self.config.train.batch_accumulate_to_next_step == 0:
                self.optim.step()
                self.normalize_feats()
                self.optim.zero_grad()

            results_epoch += results_batch
            results_batch.log_with_prefix('train')

        self.scheduler.step()
        self.optim.zero_grad()

        results_visual = self.get_results_visual(results_epoch=results_epoch, dataset=dataset,
                                                 rank_metric_name='sim',
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
        vts2d, mask_vts2d_vsbl = self.meshes.verts2d(cams_intr4x4=batch.cam_intr4x4,
                                                     cams_tform4x4_obj=batch.cam_tform4x4_obj,
                                                     imgs_sizes=batch.size, mesh_ids=batch.label,
                                                     down_sample_rate=self.down_sample_rate)
        N = vts2d.shape[1]
        # B x F+N x C
        net_feats2d = self.net(batch.rgb)
        H, W = net_feats2d.shape[-2:]
        xy = torch.stack(
            torch.meshgrid(torch.arange(W, device=self.device), torch.arange(H, device=self.device),
                           indexing='xy'), dim=0)  # HxW
        prob_noise = (1. - 1. * resize(batch.mask, scale_factor=1. / self.down_sample_rate)).flatten(1)
        prob_noise[prob_noise.sum(dim=-1) == 0.] = 1.
        noise2d = xy.flatten(1)[:, torch.multinomial(prob_noise, self.config.num_noise)].permute(1, 2, 0)
        net_feats = sample_pxl2d_pts(net_feats2d, pxl2d=torch.cat([vts2d, noise2d], dim=1))

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

        batch_vts_ids = torch.cat([batch_vts_ids[:, :N][mask_vts2d_vsbl], batch_vts_ids[:, N:].reshape(-1)],
                                  dim=0)
        net_feats = torch.cat([net_feats[:, :N][mask_vts2d_vsbl], net_feats[:, N:].reshape(-1, C)], dim=0)

        # batch_vts_ids = self.meshes.get_feats_ids_stacked(batch.label.tolist())

        bank_feats = torch.cat([self.meshes.feats, self.clutter_feats], dim=0)

        sim = torch.einsum('nc,vc->nv', net_feats, bank_feats)

        sim_batchwise_borders = torch.cat([torch.LongTensor([0]).to(device=mask_vts2d_vsbl.device), mask_vts2d_vsbl.sum(dim=1).cumsum(dim=0)], dim=0)
        sim_batchwise = torch.stack([sim[sim_batchwise_borders[b]:sim_batchwise_borders[b+1]].max(dim=-1)[0].mean() for b in range(len(sim_batchwise_borders)-1)], dim=0)
        results_batch['sim'] = sim_batchwise

        sim = sim / self.config.train.T

        # subsample_ids = torch.multinomial(sim_weight, num_samples = sim_weight.shape[0], replacement=True)
        # loss = criterion(sim[subsample_ids], batch_vts_ids[subsample_ids])
        lossCLS = self.criterion(sim, batch_vts_ids)
        loss = lossCLS

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


    def inference_batch(self, batch):
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
            net_feats2d = self.net(batch.rgb)

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
                out_shape = net_feats2d.shape[:1] + torch.Size([1]) + net_feats2d.shape[2:]
                inner_feats2d_net_bank_vts_max_vals = torch.einsum('bchw,kc->bkhw', net_feats2d, bank_feats).max(dim=1,
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
                                                                                       feats2d_net=net_feats2d,
                                                                                       categories_ids=pred_class_ids)
            #  OPTION A: Use 2d gradient of rendered features
            sim = self.get_sim_feats2d_net_with_cams(feats2d_net=net_feats2d,
                                                     cam_tform4x4_obj=b_cams_multiview_tform4x4_obj,
                                                     cam_intr4x4=b_cams_multiview_intr4x4,
                                                     categories_ids=pred_class_ids,
                                                     broadcast_batch_and_cams=True)
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

                sim, sim_pxl = self.get_sim_feats2d_net_with_cams(feats2d_net=net_feats2d,
                                                                  cam_tform4x4_obj=cam_tform4x4_obj,
                                                                  cam_intr4x4=batch.cam_intr4x4,
                                                                  categories_ids=pred_class_ids, return_sim_pxl=True,
                                                                  broadcast_batch_and_cams=False)
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
            diff_so3_log = pytorch3d.transforms.so3_log_map(diff_rot3x3)
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

    def get_results_visual(self, results_epoch, dataset: OD3D_Dataset, config_visualize: DictConfig, rank_metric_name='rot_diff_rad'):
        results = OD3D_Results()

        count_best = config_visualize.count_best
        count_worst = config_visualize.count_worst
        count_rand = config_visualize.count_rand
        modalities = config_visualize.modalities
        live = config_visualize.live

        if len(modalities) == 0:
            return results

        # sorts values ascending
        epoch_ranked_ids = results_epoch[rank_metric_name].sort(dim=0)[1]
        epoch_best_ids = epoch_ranked_ids[:count_best]
        epoch_best_names = [f'best_{i+1}' for i in range(len(epoch_best_ids))]
        epoch_worst_ids = epoch_ranked_ids[-count_worst:]
        epoch_worst_names = [f'worst_{len(epoch_worst_ids) - i}' for i in range(len(epoch_worst_ids))]
        epoch_rand_ids = epoch_ranked_ids[torch.randperm(len(epoch_ranked_ids))[:count_rand]]
        epoch_rand_names = [f'rand_{i+1}' for i in range(len(epoch_rand_ids))]
        sel_rank_ids = torch.cat([epoch_best_ids, epoch_worst_ids, epoch_rand_ids], dim=0)
        sel_item_ids = results_epoch['item_id'][sel_rank_ids]
        sel_names = epoch_best_names + epoch_worst_names + epoch_rand_names
        sel_name_unique = [results_epoch['name_unique'][id] for id in sel_rank_ids]
        dict_name_unique_to_result_id = dict(zip(sel_name_unique, sel_rank_ids))
        dict_name_unique_to_sel_name = dict(zip(sel_name_unique, sel_names))
        dataset_visualize = dataset.get_subset_with_item_ids(item_ids=sel_item_ids)
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



                feats2d_net = self.net(batch.rgb)

                if VISUAL_MODALITIES.NET_FEATS_NEAREST_VERTS in modalities:
                    verts3d = self.get_nearest_verts3d_to_feats2d_net(feats2d_net=feats2d_net, categories_ids=batch.label,
                                                                      zero_if_sim_clutter_larger=True)
                    for b in range(len(batch)):
                        img = blend_rgb(resize(batch.rgb[b], scale_factor=1. / self.down_sample_rate), verts3d[b])
                        results[f'visual/{batch_sel_names[b]}_{VISUAL_MODALITIES.NET_FEATS_NEAREST_VERTS}'] = image_as_wandb_image(img, caption=batch_sel_names[b])
                        if live:
                            show_img(img)

                if VISUAL_MODALITIES.SAMPLES in modalities:
                    s_cam_tform4x4_obj, s_cam_intr4x4 = self.get_samples(config_sample=self.config.inference.sample,
                                                                                               cam_intr4x4=batch.cam_intr4x4,
                                                                                               cam_tform4x4_obj=batch.cam_tform4x4_obj,
                                                                                               feats2d_net=feats2d_net,
                                                                                               categories_ids=batch.label)

                    ncds = self.get_ncds_with_cam(cam_tform4x4_obj=s_cam_tform4x4_obj,
                                                  cam_intr4x4=s_cam_intr4x4, size=batch.size,
                                                  categories_ids=batch.label, down_sample_rate=self.down_sample_rate, broadcast_batch_and_cams=True)

                    sim = self.get_sim_feats2d_net_with_cams(feats2d_net=feats2d_net,
                                                             cam_tform4x4_obj=s_cam_tform4x4_obj,
                                                             cam_intr4x4=s_cam_intr4x4,
                                                             categories_ids=batch.label,
                                                             broadcast_batch_and_cams=True)

                    for b in range(len(batch)):
                        imgs = ncds[b]
                        imgs_sim = sim[b][:].expand(*sim[b].shape)  # , *mesh_feats2d_rendered.shape[-2:]
                        if self.config.inference.sample.method == 'uniform':
                            imgs = imgs.reshape(self.config.inference.sample.uniform.azim.steps, self.config.inference.sample.uniform.elev.steps, self.config.inference.sample.uniform.theta.steps,
                                                *imgs.shape[-3:])[:, :, 0]
                            imgs_sim = imgs_sim.reshape(self.config.inference.sample.uniform.azim.steps, self.config.inference.sample.uniform.elev.steps, self.config.inference.sample.uniform.theta.steps)[:, :, 0]
                        imgs = blend_rgb(resize(batch.rgb[b], scale_factor=1. / self.down_sample_rate), imgs)

                        if config_visualize.samples_sorted:
                            imgs_sim = imgs_sim.flatten(0)
                            imgs = imgs.reshape(-1, *imgs.shape[-3:])
                            imgs_sim_ids = imgs_sim.sort(descending=True)[1]
                            imgs = imgs[imgs_sim_ids]
                            imgs_sim = imgs_sim[imgs_sim_ids]

                        if config_visualize.samples_scores:
                            samples_score_size = imgs.shape[-1] // 5
                            imgs[..., -samples_score_size:, -samples_score_size:] = (
                                    255 * imgs_sim.reshape(*imgs_sim.shape, 1, 1, 1).expand(*imgs.shape[:-2],
                                                                                            samples_score_size,
                                                                                            samples_score_size)).to(
                                torch.uint8)

                        img = imgs_to_img(imgs)
                        results[f'visual/{batch_sel_names[b]}_{VISUAL_MODALITIES.SAMPLES}'] = image_as_wandb_image(img, caption=batch_sel_names[b])
                        if live:
                            show_img(img)


                if VISUAL_MODALITIES.SIM_PXL in modalities:
                    batch_pred_label = results_epoch['label_pred'].to(device=self.device)[batch_result_ids]
                    batch_pred_cam_tform4x4 = results_epoch['cam_tform4x4_obj'].to(device=self.device)[batch_result_ids]
                    sim, sim_pxl = self.get_sim_feats2d_net_with_cams(feats2d_net=feats2d_net,
                                                                      cam_intr4x4=batch.cam_intr4x4,
                                                                      cam_tform4x4_obj=batch_pred_cam_tform4x4,
                                                                      categories_ids=batch_pred_label, return_sim_pxl=True,
                                                                      broadcast_batch_and_cams=False)

                    for b in range(len(batch)):
                        img = blend_rgb(resize(batch.rgb[b], scale_factor=1. / self.down_sample_rate),
                                        sim_pxl[b])
                        results[f'visual/{batch_sel_names[b]}_{VISUAL_MODALITIES.SIM_PXL}'] = image_as_wandb_image(img, caption=f'{batch_sel_names[b]}, mean sim={sim[b].item()}')
                        if live:
                            show_img(img)


                if VISUAL_MODALITIES.VERTS_NCDS_IN_RGB in modalities:
                    batch_pred_label = results_epoch['label_pred'].to(device=self.device)[batch_result_ids]
                    batch_pred_cam_tform4x4 = results_epoch['cam_tform4x4_obj'].to(device=self.device)[batch_result_ids]
                    ncds = self.get_ncds_with_cam(cam_intr4x4=batch.cam_intr4x4, cam_tform4x4_obj=batch_pred_cam_tform4x4, categories_ids=batch_pred_label, size=batch.size, down_sample_rate=self.down_sample_rate)
                    for b in range(len(batch)):
                        img = blend_rgb(resize(batch.rgb[b], scale_factor=1. / self.down_sample_rate),
                                        ncds[b])
                        results[f'visual/{batch_sel_names[b]}_{VISUAL_MODALITIES.VERTS_NCDS_IN_RGB}'] = image_as_wandb_image(img, caption=batch_sel_names[b])
                        if live:
                            show_img(img)

                if VISUAL_MODALITIES.GT_VERTS_NCDS_IN_RGB in modalities:
                    ncds = self.get_ncds_with_cam(cam_intr4x4=batch.cam_intr4x4,
                                                  cam_tform4x4_obj=batch.cam_tform4x4_obj,
                                                  categories_ids=batch.label, size=batch.size,
                                                  down_sample_rate=self.down_sample_rate)
                    for b in range(len(batch)):
                        img = blend_rgb(resize(batch.rgb[b], scale_factor=1. / self.down_sample_rate),
                                        ncds[b])
                        results[
                            f'visual/{batch_sel_names[b]}_{VISUAL_MODALITIES.VERTS_NCDS_IN_RGB}'] = image_as_wandb_image(
                            img, caption=batch_sel_names[b])
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

    def get_ncds_with_cam(self, cam_intr4x4: torch.Tensor, cam_tform4x4_obj: torch.Tensor, size: torch.Tensor, categories_ids: torch.Tensor, down_sample_rate=1., broadcast_batch_and_cams=False):
        return self.meshes.render_feats(cams_tform4x4_obj=cam_tform4x4_obj,
                                        cams_intr4x4=cam_intr4x4,
                                        imgs_sizes=size, meshes_ids=categories_ids,
                                        modality=MESH_RENDER_MODALITIES.VERTS_NCDS,
                                        down_sample_rate=down_sample_rate,
                                        broadcast_batch_and_cams=broadcast_batch_and_cams)

    def get_sim_feats2d_net_with_cams(self, feats2d_net, cam_tform4x4_obj, cam_intr4x4, categories_ids, return_sim_pxl=False,
                                      broadcast_batch_and_cams=False):
        size = torch.Tensor([feats2d_net.shape[2] * self.down_sample_rate, feats2d_net.shape[3] * self.down_sample_rate])
        mesh_feats2d_rendered = self.meshes.render_feats(cams_tform4x4_obj=cam_tform4x4_obj,
                                                         cams_intr4x4=cam_intr4x4,
                                                         imgs_sizes=size, meshes_ids=categories_ids,
                                                         down_sample_rate=self.down_sample_rate,
                                                         broadcast_batch_and_cams=broadcast_batch_and_cams)

        return self.get_sim_feats2d_net_and_rendered(feats2d_net=feats2d_net, feats2d_rendered=mesh_feats2d_rendered, return_sim_pxl=return_sim_pxl)

    def get_sim_feats2d_net_with_samples(self, config_sample: DictConfig, cam_intr4x4: torch.Tensor,
                                         cam_tform4x4_obj: torch.Tensor, feats2d_net: torch.Tensor,
                                         categories_ids: torch.Tensor, return_sim_pxl=False):
        b_cams_multiview_tform4x4_obj, b_cams_multiview_intr4x4 = self.get_samples(config_sample=config_sample,
                                                                                   cam_intr4x4=cam_intr4x4,
                                                                                   cam_tform4x4_obj=cam_tform4x4_obj,
                                                                                   feats2d_net=feats2d_net,
                                                                                   categories_ids=categories_ids)
        return self.get_sim_feats2d_net_with_cams(feats2d_net=feats2d_net,
                                                  cam_tform4x4_obj=b_cams_multiview_tform4x4_obj,
                                                  cam_intr4x4=b_cams_multiview_intr4x4,
                                                  categories_ids=categories_ids, return_sim_pxl=return_sim_pxl,
                                                  broadcast_batch_and_cams=True)

    def get_nearest_corresp2d3d(self, feats2d_net, meshes_ids):
        nearest_verts3d, sim_texture, sim_clutter = self.get_nearest_verts3d_to_feats2d_net(feats2d_net, meshes_ids, return_sim_texture_and_clutter=True)
        nearest_verts2d = get_pxl2d_like(nearest_verts3d.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # H=sim_clutter.shape[1], W=sim_clutter.shape[2], dtype=sim_nearest_texture_verts.dtype, device=sim_nearest_texture_verts.device)[None,].expand()
        prob_corresp2d3d = (sim_clutter < sim_texture) * sim_texture

        return nearest_verts3d, nearest_verts2d, prob_corresp2d3d


    def get_sim_feats2d_net_and_rendered(self, feats2d_net, feats2d_rendered, return_sim_pxl=False):
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

        sim = sim_pxl.flatten(2).mean(dim=-1)

        if return_sim_pxl:
            return sim, sim_pxl
        else:
            return sim

    def get_samples(self, config_sample: DictConfig, cam_intr4x4: torch.Tensor, cam_tform4x4_obj: torch.Tensor,
                    feats2d_net: torch.Tensor, categories_ids: torch.Tensor):
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
            # b_cams_multiview_tform4x4_obj[:, :, 2, 3] = cam_tform4x4_obj[:, None].repeat(1, C, 1, 1)[:, :, 2, 3]
            # logger.info(f'dist {batch.cam_tform4x4_obj[:, 2, 3]}')
            # assumption 2: translation to object is known
            b_cams_multiview_tform4x4_obj[:, :, :3, 3] = cam_tform4x4_obj[:, None].repeat(1, C, 1, 1)[:, :, :3, 3]

            b_cams_multiview_intr4x4 = cam_intr4x4[:, None].repeat(1, C, 1, 1)

        elif config_sample.method == 'epnp3d2d':

            nearest_verts3d, nearest_verts2d, prob_well_corresp = self.get_nearest_corresp2d3d(feats2d_net=feats2d_net,
                                                                                               meshes_ids=categories_ids)
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