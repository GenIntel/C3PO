import time

from od3d.methods.method import OD3DMethod
from od3d.datasets.dataset import OD3D_Dataset
from omegaconf import DictConfig
import pytorch3d.transforms

from torch.utils.data import RandomSampler
import logging
logger = logging.getLogger(__name__)
import torch
import numpy as np
import wandb
import math
from od3d.cv.visual.draw import draw_pixels

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


class NeMo(OD3DMethod):
    def __init__(
        self,
        config: DictConfig,
        logging_dir,
    ):
        super().__init__(config=config, logging_dir=logging_dir)

        self.device = 'cuda:0'

        # init Network
        self.net = OD3D_Backbone.subclasses[config.backbone.class_name](config.backbone)
        from od3d.cv.transforms import RandomCenterZoom3D, RGB_Random, CenterZoom3D

        self.transform_train = torchvision.transforms.Compose([
            RandomCenterZoom3D(**config.train.transform),
            RGB_Random(),
            self.net.transform,
        ])
        self.transform_test = torchvision.transforms.Compose([
            CenterZoom3D(**config.test.transform),
            self.net.transform
        ])

        # init Meshes / Features
        self.total_params = sum(p.numel() for p in self.net.parameters())
        #self.path_shapenemo = Path(config.path_shapenemo)
        #self.fpaths_meshes_shapenemo = [self.path_shapenemo.joinpath(cls, '01.off') for cls in config.classes]
        self.fpaths_meshes = [self.config.fpaths_meshes[cls] for cls in config.classes]
        self.meshes = Meshes.load_from_files(fpaths_meshes=self.fpaths_meshes)
        # self.meshes.show()
        self.verts_count_max = self.meshes.verts_counts_max
        self.mem_verts_feats_count = len(config.classes) * self.verts_count_max
        self.mem_clutter_feats_count = config.num_noise * config.max_group
        self.mem_count = self.mem_verts_feats_count + self.mem_clutter_feats_count

        self.clutter_feats = torch.nn.Parameter(torch.randn(size=(1, self.net.feat_dim), device=self.device), requires_grad=True)
        self.meshes.set_feats_cat_with_pad(torch.nn.Parameter(torch.randn(size=(self.verts_count_max * len(self.meshes), self.net.feat_dim), device=self.device), requires_grad=True))

        self.normalize_feats()

        #self.net = torch.nn.DataParallel(self.net).cuda()
        self.net.cuda()
        self.meshes.cuda()
        self.net.eval()

        self.optim = torch.optim.Adam(list(self.net.parameters()) + [self.meshes.feats] + [self.clutter_feats], lr=self.config.train.optimizer.lr,
                                 weight_decay=self.config.train.optimizer.weight_decay)  #
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, gamma=self.config.train.scheduler.gamma,
                                                         milestones=self.config.train.scheduler.milestones)


        # load checkpoint
        if config.get("checkpoint", None) is not None:
            self.load_checkpoint(config.checkpoint)
        elif config.get("checkpoint_old", None) is not None:
            self.load_checkpoint_old(config.checkpoint_old)
        #load_mesh(config.path_shapenemo)

        # self.meshes.show()



        #self.verts_feats = checkpoint["memory"][:self.mem_verts_feats_count].clone().detach().cpu()
        # note: somehow vertices are stored in wrong order of classes (starting with last class tvmonitor until first class aeroplane
        # self.verts_feats = self.verts_feats.reshape(len(self.meshes), self.verts_count_max, -1).flip(dims=(0,)).reshape(len(self.meshes) * self.verts_count_max, -1)


        self.down_sample_rate = self.config.down_sample_rate
        # self.set_distance = 5.0
        self.classification_size = [512, 512]
        self.render_image_size = max(self.classification_size) // self.down_sample_rate
        self.map_shape = (
            self.classification_size[0] // self.down_sample_rate,
            self.classification_size[1] // self.down_sample_rate,
        )

    def normalize_feats(self):
        self.clutter_feats.data = self.clutter_feats.detach() / self.clutter_feats.detach().norm(dim=-1, keepdim=True)
        self.meshes.feats.data = self.meshes.feats.detach() / self.meshes.feats.detach().norm(dim=-1, keepdim=True)
        #logger.info(self.clutter_feats[:1])
        #logger.info(self.meshes.feats[:1])

    def load_checkpoint_old(self, path_checkpoint):
        checkpoint = torch.load(path_checkpoint, map_location="cuda:0")
        self.net.net = torch.nn.DataParallel(self.net.net).cuda()
        self.net.net.load_state_dict(checkpoint["state"], strict=False)
        self.net.net = self.net.net.module
        self.clutter_feats = checkpoint["memory"][self.mem_verts_feats_count:].clone().detach().cpu()
        self.clutter_feats = self.clutter_feats.mean(dim=0, keepdim=True)
        self.clutter_feats = torch.nn.Parameter(self.clutter_feats.to(device=self.device), requires_grad=True)

        verts_feats = checkpoint["memory"][:self.mem_verts_feats_count].clone().detach().cpu()
        self.meshes.set_feats_cat_with_pad(verts_feats)


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
    def setup(self):
        pass

    def train(self, dataset: OD3D_Dataset, dataset_test: OD3D_Dataset):
        dataset.transform = self.transform_train

        self.net.train()
        self.meshes.feats.requires_grad = True

        accumulate_steps = 0

        generator = torch.Generator().manual_seed(42)
        # self.meshes.show(pts3d=dataset.get_sequence_by_id(0).pcl[None,])
        dataset_sub, _ = torch.utils.data.random_split(dataset, [dataset.config.subset_fraction, 1. - dataset.config.subset_fraction], generator=generator)


        dataset_train, dataset_val = torch.utils.data.random_split(dataset_sub, [1. - self.config.train.val_fraction, self.config.train.val_fraction], generator=generator)

        criterion = torch.nn.CrossEntropyLoss().cuda() # (reduction="none").cuda()
        dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=self.config.train.dataloader.batch_size, shuffle=True,
                                                       collate_fn=dataset.collate_fn, num_workers=self.config.train.dataloader.num_workers, pin_memory=self.config.train.dataloader.pin_memory)



        logger.info(f"Dataset contains {len(dataset_sub)} frames.")

        for e in range(self.config.train.epochs):
            if e % self.config.train.epochs_to_next_val == 0:
                results_val = self.test(dataset, dataset_sub=dataset_val)
                wandb.log({'val_' + k: v for k, v in results_val.items()})

            if self.config.train.epochs_to_next_test > 0 and e % self.config.train.epochs_to_next_test == 0:
                results_test = self.test(dataset_test)
                wandb.log({'test_' + k: v for k, v in results_test.items()})

            self.net.train()
            self.meshes.feats.requires_grad = True

            results_train = {}
            for i, batch in enumerate(iter(dataloader_train)):
                batch.to(device=self.device)
                # B x x N x 2
                vts2d, mask_vts2d_vsbl = self.meshes.verts2d(cams_intr4x4=batch.cam_intr4x4, cams_tform4x4_obj=batch.cam_tform4x4_obj, imgs_sizes=batch.size, mesh_ids=batch.label, down_sample_rate=self.down_sample_rate)
                N = vts2d.shape[1]

                if self.config.train.visualize.verts_ncds_in_rgb:
                    verts_ncds_in_rgb = blend_rgb(batch.rgb[0], (self.meshes.render_feats(cams_tform4x4_obj=batch.cam_tform4x4_obj[:1], cams_intr4x4=batch.cam_intr4x4[:1],
                                                    imgs_sizes=batch.size, meshes_ids=batch.label[:1],
                                                    modality=MESH_RENDER_MODALITIES.VERTS_NCDS)[0]).to(dtype=batch.rgb.dtype))
                    from od3d.cv.geometry.transform import proj3d2d_origin
                    verts_ncds_in_rgb = draw_pixels(verts_ncds_in_rgb, pxls=proj3d2d_origin(torch.bmm(batch.cam_intr4x4, batch.cam_tform4x4_obj)[:1]), colors=[1., 0., 0.])
                    verts_ncds_in_rgb = draw_pixels(verts_ncds_in_rgb, pxls=proj3d2d_origin(batch.cam_proj4x4_obj[:1]), colors=[1., 0., 0.])
                    img = draw_pixels(verts_ncds_in_rgb, self.down_sample_rate * vts2d[0, mask_vts2d_vsbl[0]], colors=self.meshes.get_verts_ncds_with_mesh_id(batch.label[0])[mask_vts2d_vsbl[0]])
                    results_train['verts_ncds_in_rgb'] = image_as_wandb_image(img, caption=f'Frame Name {batch.name[0]}')
                    if self.config.train.visualize.live:
                        show_img(img)


                # B x F+N x C
                net_feats2d = self.net(batch.rgb)
                H, W = net_feats2d.shape[-2:]
                xy = torch.stack(torch.meshgrid(torch.arange(W,device=self.device), torch.arange(H, device=self.device), indexing='xy'), dim=0) # HxW
                noise2d = xy.flatten(1)[:, torch.multinomial((1. - 1. * resize(batch.mask, scale_factor=1. / self.down_sample_rate)).flatten(1), self.config.num_noise)].permute(1, 2, 0)
                net_feats = sample_pxl2d_pts(net_feats2d, pxl2d=torch.cat([vts2d, noise2d], dim=1))

                C = net_feats.shape[2]
                # args: X: Bx3xHxW, keypoint_positions: BxNx2, obj_mask: BxHxW ensures that noise is sampled outside of object mask
                # returns: BxF+NxC
                if self.config.train.visualize.net_feats_nearest_verts:
                    clutter_sim, clutter_sim_ids = torch.einsum('bcn,vc->bnv', net_feats2d[:1].flatten(-2), self.clutter_feats).max(dim=-1)
                    net_mesh_nearest_feats_sim, net_mesh_nearest_feats_ids = torch.einsum('bcn,vc->bnv', net_feats2d[:1].flatten(-2), self.meshes.get_feats_with_mesh_id(batch.label[0])).max(dim=-1)
                    net_mesh_nearest_feats_verts_ncds = self.meshes.get_verts_ncds_with_mesh_id(batch.label[0])[net_mesh_nearest_feats_ids]
                    net_mesh_nearest_feats_verts_ncds[clutter_sim > net_mesh_nearest_feats_sim] = 0.
                    net_mesh_nearest_feats_verts_ncds = net_mesh_nearest_feats_verts_ncds.reshape(-1, *net_feats2d.shape[-2:], 3).permute(0, 3, 1, 2)
                    img = blend_rgb(resize(batch.rgb[0], scale_factor=1./self.down_sample_rate), net_mesh_nearest_feats_verts_ncds[0])
                    results_train['net_feats_nearest_verts'] = image_as_wandb_image(img, caption=f'Frame Name {batch.name[0]}')
                    if self.config.train.visualize.live:
                        show_img(img)


                # net_feats = net_feats[:, :].reshape(-1, net_feats.shape[-1])
                batch_vts_ids = self.meshes.get_verts_and_noise_ids_stacked(batch.label.tolist(), count_noise_ids=self.config.num_noise)

                batch_vts_ids = torch.cat([batch_vts_ids[:, :N][mask_vts2d_vsbl], batch_vts_ids[:, N:].reshape(-1)], dim=0)
                net_feats = torch.cat([net_feats[:, :N][mask_vts2d_vsbl], net_feats[:, N:].reshape(-1, C)], dim=0)

                # batch_vts_ids = self.meshes.get_feats_ids_stacked(batch.label.tolist())

                bank_feats = torch.cat([self.meshes.feats, self.clutter_feats], dim=0)

                sim = torch.einsum('nc,vc->nv', net_feats, bank_feats)

                sim = sim / self.config.train.T

                loss = criterion(sim, batch_vts_ids).mean()
                loss.backward()
                logger.info(f'loss {loss.item()}')
                results_train['loss'] = loss
                wandb.log({'train_' + k: v for k, v in results_train.items()})

                accumulate_steps += 1
                if accumulate_steps % self.config.train.batch_accumulate_to_next_step == 0:
                    self.optim.step()
                    self.normalize_feats()
                    self.optim.zero_grad()


            self.scheduler.step()
            if (e + 1) % self.config.train.epochs_to_next_ckpt == 0:
                self.save_checkpoint(path_checkpoint=self.logging_dir.joinpath('nemo.ckpt'))

            if not accumulate_steps % self.config.train.batch_accumulate_to_next_step == 0:
                self.optim.step()
                self.normalize_feats()
                self.optim.zero_grad()

    def calc_loss_feat2d_net_bank(self, feats2d_net, feats2d_bank):
        pass
    def calc_loss_feat2d_net_rendered(self, feats2d_net, feats2d_rendered):
        pass
    def test(self, dataset: OD3D_Dataset, pose_iterative_refine=False, pose_iterative_xy_shift=False, dataset_sub=None):
        self.net.eval()
        self.meshes.feats.requires_grad = False
        clutter_feats = self.clutter_feats.detach()
        dataset.transform = self.transform_test

        if dataset_sub is None:
            generator = torch.Generator().manual_seed(42) # dataset.config.subset_fraction
            dataset_sub, _ = torch.utils.data.random_split(dataset, [dataset.config.subset_fraction, 1. - dataset.config.subset_fraction], generator=generator)
        dataloader = torch.utils.data.DataLoader(dataset=dataset_sub, batch_size=self.config.test.dataloader.batch_size, shuffle=False,
                                                 collate_fn=dataset.collate_fn, num_workers=self.config.test.dataloader.num_workers, pin_memory=self.config.test.dataloader.pin_memory)

        logger.info(f"Dataset contains {len(dataset_sub)} frames.")

        results = {
            'time_feats2d': [],
            'time_class': [],
            'time_pose_iterative': [],
            'time_pose': [],
            'rot_diff_rad': [],
            'label_gt': [],
            'label_pred': [],
        }

        azim = torch.linspace(start=eval(self.config.test.azim.min), end=eval(self.config.test.azim.max), steps=self.config.test.azim.steps).to(device=self.device)  # 12
        elev = torch.linspace(start=eval(self.config.test.elev.min), end=eval(self.config.test.elev.max), steps=self.config.test.elev.steps).to(
            device=self.device)  # start=-torch.pi / 6, end=torch.pi / 3, steps=4
        theta = torch.linspace(start=eval(self.config.test.theta.min), end=eval(self.config.test.theta.max), steps=self.config.test.theta.steps).to(
            device=self.device)  # -torch.pi / 6, end=torch.pi / 6, steps=3
        dist = torch.linspace(start=eval(self.config.test.dist.min), end=eval(self.config.test.dist.max), steps=self.config.test.dist.steps).to(device=self.device)

        azim_shape = azim.shape
        elev_shape = elev.shape
        theta_shape = theta.shape
        dist_shape = dist.shape
        in_shape = azim_shape + elev_shape + theta_shape + dist_shape
        azim = azim[:, None, None, None].expand(in_shape).reshape(-1)
        elev = elev[None, :, None, None].expand(in_shape).reshape(-1)
        theta = theta[None, None, :, None].expand(in_shape).reshape(-1)
        dist = dist[None, None, None, :].expand(in_shape).reshape(-1)
        cams_count = in_shape.numel()
        classes_count = len(self.config.classes)
        cams_multiview_tform4x4_cuboid = transf4x4_from_spherical(azim=azim, elev=elev, theta=theta, dist=dist)

        C = len(cams_multiview_tform4x4_cuboid)

        diffs_so3d_log = []

        for i, batch in tqdm(enumerate(iter(dataloader))):
            batch.to(device=self.device)
            B = len(batch)

            b_cams_multiview_tform4x4_obj = cams_multiview_tform4x4_cuboid[None,].repeat(B, 1, 1, 1)

            # assumption 1: distance to object is known
            b_cams_multiview_tform4x4_obj[:, :, 2, 3] = batch.cam_tform4x4_obj[:, None].repeat(1, C, 1, 1)[:, :, 2, 3]

            b_cams_multiview_intr4x4 = batch.cam_intr4x4[:, None].repeat(1, C, 1, 1)

            time_loaded = time.time()
            with torch.no_grad():
                net_feats2d = self.net(batch.rgb)
                time_pred_net_feats2d = time.time()
                #logger.info(
                #    f"predicted net feats2d, took {(time_pred_net_feats2d - time_loaded):.3f}")
                results['time_feats2d'].append(time_pred_net_feats2d - time_loaded)

                meshes_scores = []
                for mesh_id in range(len(self.meshes)):
                    # logger.info(f'calc score for mesh {self.config.classes[mesh_id]}')
                    bank_feats = torch.cat([self.meshes.get_feats_with_mesh_id(mesh_id), clutter_feats], dim=0)
                    # inner_feats2d_net_bank_vts_max_vals = torch.sum(net_feats2d[:, None] * bank_feats[None, :, :, None, None], dim=2, keepdim=True).max(dim=1).values
                    out_shape = net_feats2d.shape[:1] + torch.Size([1]) + net_feats2d.shape[2:]
                    inner_feats2d_net_bank_vts_max_vals = torch.einsum('bchw,kc->bkhw', net_feats2d, bank_feats).max(dim=1, keepdim=True).values
                    # inner_feats2d_net_bank_vts_max_vals, inner_feats2d_net_bank_vts_max_ids = inner_feats2d.max(dim=1)
                    # show_img(self.meshes.get_verts_with_mesh_id[mesh_id][inner_feats2d_net_bank_vts_max_ids[0, 0]].permute(2, 0, 1), normalize=True)
                    # show_img(inner_feats2d_net_bank_vts_max_vals[0])
                    mesh_score = inner_feats2d_net_bank_vts_max_vals.flatten(1).mean(dim=1)
                    #clutter_score = inner_feats2d[:, -clutter_feats.shape[0]:].mean(dim=1).flatten(1).mean(dim=1)
                    #mesh_score -= clutter_score
                    meshes_scores.append(mesh_score)
                meshes_scores = torch.stack(meshes_scores, dim=-1)
                pred_class_scores, pred_class_ids = meshes_scores.max(dim=1)

                logger.info(f'pred class ids {pred_class_ids}')
                time_pred_class = time.time()
                # logger.info(f"predicted class: {self.config.classes[int(pred_class_ids[0])]}, took {(time_pred_class - time_pred_net_feats2d):.3f}")

                results['time_class'].append(time_pred_class - time_pred_net_feats2d)


                #  OPTION A: Use 2d gradient of rendered features
                mesh_feats2d_rendered = self.meshes.render_feats(cams_tform4x4_obj=b_cams_multiview_tform4x4_obj, cams_intr4x4=b_cams_multiview_intr4x4, imgs_sizes=batch.size, meshes_ids=pred_class_ids, down_sample_rate=self.down_sample_rate, broadcast_batch_and_cams=True)
                inner_feats2d_net_mesh_multiple_cams = torch.einsum('bchw,bvchw->bvhw', net_feats2d, mesh_feats2d_rendered)[:, :, None]
                inner_feats2d_net_clutter = torch.einsum('bchw,nc->bnhw', net_feats2d, clutter_feats)[:, None,].mean(dim=1, keepdim=True)
                inner_feats2d_net_bank_multiple_cams = torch.max(inner_feats2d_net_mesh_multiple_cams, inner_feats2d_net_clutter)
                mesh_multiple_cams_loss = 1 - (inner_feats2d_net_bank_multiple_cams.flatten(-3).mean(dim=-1)) #  - inner_feats2d_net_clutter.flatten(1).mean())


                # OPTION B: Use 2d gradient of net features
                #vts2d, mask_vts2d_vsbl = self.meshes.verts2d(cams_tform4x4_obj=b_cams_multiview_tform4x4_obj, cams_intr4x4=b_cams_multiview_intr4x4, imgs_sizes=batch.size, mesh_ids=pred_class_ids, down_sample_rate=self.down_sample_rate, broadcast_batch_and_cams=True)
                #net_feats = sample_pxl2d_pts(net_feats2d, pxl2d=vts2d.reshape(len(batch),-1 , 2)).reshape(*vts2d.shape[:3], -1)
                #sim = torch.einsum('bvfc,bfc->bvf', net_feats, self.meshes.get_feats_stacked_with_mesh_ids(pred_class_ids)) * mask_vts2d_vsbl
                #mesh_multiple_cams_loss = -sim.mean(dim=-1)

                if self.config.test.visualize.samples and self.config.test.visualize.live:
                    show_imgs(self.meshes.render_feats(cams_tform4x4_obj=b_cams_multiview_tform4x4_obj, cams_intr4x4=b_cams_multiview_intr4x4, imgs_sizes=batch.size, meshes_ids=pred_class_ids, down_sample_rate=self.down_sample_rate, broadcast_batch_and_cams=True, modality='rgb'))

                mesh_cam_loss_min_val, mesh_cam_loss_min_id = mesh_multiple_cams_loss.min(dim=1)

                cam_transf4x4_obj = b_cams_multiview_tform4x4_obj[:, mesh_cam_loss_min_id].permute(2, 3, 0, 1).diagonal(dim1=-2, dim2=-1).permute(2, 0, 1)
                cam_theta = theta[mesh_cam_loss_min_id]

            #show_img(self.meshes.render_feats(cams_tform4x4_obj=init_cams_tform4x4_obj, cams_intr4x4=batch.cam_intr4x4 / 2,
            #                                imgs_sizes=batch.size // 2, meshes_ids=[int(pred_class_ids[0])],
            #                                replace_feats_with_verts3d=True)[0, 0])
            #show_img(self.meshes.render_feats(cams_tform4x4_obj=cam_transf4x4_obj, cams_intr4x4=batch.cam_intr4x4 / 2,
            #                                  imgs_sizes=batch.size //2, meshes_ids=[int(pred_class_ids[0])],
            #                                  replace_feats_with_verts3d=True)[0, 0])

            if self.config.test.visualize.net_feats_nearest_verts:
                clutter_sim, clutter_sim_ids = torch.einsum('bcn,vc->bnv', net_feats2d[:1].flatten(-2), self.clutter_feats).max(dim=-1)
                net_mesh_nearest_feats_sim, net_mesh_nearest_feats_ids = torch.einsum('bcn,vc->bnv', net_feats2d[:1].flatten(-2), self.meshes.get_feats_with_mesh_id(batch.label[0])).max(dim=-1)
                net_mesh_nearest_feats_verts_ncds = self.meshes.get_verts_ncds_with_mesh_id(batch.label[0])[net_mesh_nearest_feats_ids]
                net_mesh_nearest_feats_verts_ncds[clutter_sim > net_mesh_nearest_feats_sim] = 0.
                net_mesh_nearest_feats_verts_ncds = net_mesh_nearest_feats_verts_ncds.reshape(-1, *net_feats2d.shape[-2:], 3).permute(0, 3, 1, 2)
                img = blend_rgb(resize(batch.rgb[0], scale_factor=1./self.down_sample_rate), net_mesh_nearest_feats_verts_ncds[0])
                results['net_feats_nearest_verts_' + batch.name[0]] = image_as_wandb_image(img)
                if self.config.test.visualize.live:
                    show_img(img)

            if pose_iterative_refine:
                cam_pos = torch.nn.Parameter(cam_transf4x4_obj.inverse()[:, :3, 3].clone(), requires_grad=True)
                cam_theta = torch.nn.Parameter(cam_theta.clone(), requires_grad=True)
                cam_transf4x4_obj = transf4x4_from_pos_and_theta(pos=cam_pos, theta=cam_theta)
                cam_shift_xyz = torch.nn.Parameter(torch.zeros_like(cam_pos), requires_grad=True)

                if pose_iterative_xy_shift:
                    cam_transf4x4_obj[:, :2, 3] += cam_shift_xyz[:, :2]

                optim_inference = torch.optim.Adam(
                    params=[cam_pos, cam_theta, cam_shift_xyz],
                    lr=self.config.test.optimizer.lr,
                    betas=(self.config.test.optimizer.beta0, self.config.test.optimizer.beta1),
                )

                time_before_pose_iterative = time.time()

                for epoch in range(self.config.test.optimizer.epochs):

                    # OPTION A: Use 2d gradient of rendered features
                    mesh_feats2d_rendered = self.meshes.render_feats(cams_tform4x4_obj=cam_transf4x4_obj, cams_intr4x4=batch.cam_intr4x4,
                                              imgs_sizes=batch.size, meshes_ids=pred_class_ids, down_sample_rate=self.down_sample_rate) # [:, 0]
                    inner_feats2d_net_mesh = torch.einsum('bchw,bchw->bhw', net_feats2d, mesh_feats2d_rendered)[:, None]
                    inner_feats2d_net_clutter = torch.einsum('bchw,nc->bnhw', net_feats2d, clutter_feats).mean(dim=1, keepdim=True)
                    inner_feats2d_net_bank = torch.max(inner_feats2d_net_mesh, inner_feats2d_net_clutter)
                    mesh_cam_loss = 1. - (inner_feats2d_net_bank.flatten(-3).mean(dim=-1)) # - inner_feats2d_net_clutter.flatten(1).mean())


                    # OPTION A: Use 2d gradient of net features
                    #vts2d, mask_vts2d_vsbl = self.meshes.verts2d(cams_intr4x4=batch.cam_intr4x4,
                    #                                             cams_tform4x4_obj=cam_transf4x4_obj,
                    #                                             imgs_sizes=batch.size, mesh_ids=pred_class_ids,
                    #                                             down_sample_rate=self.down_sample_rate)
                    #net_feats = sample_pxl2d_pts(net_feats2d, pxl2d=vts2d.reshape(len(batch), -1, 2))
                    #sim = torch.einsum('bfc,bfc->bf', net_feats,
                    #                   self.meshes.get_feats_stacked_with_mesh_ids(pred_class_ids)) * mask_vts2d_vsbl
                    #mesh_cam_loss = -sim.mean(dim=-1)

                    if self.config.test.visualize.live:
                        # show_img(inner_feats2d_net_bank[0])
                        show_img(blend_rgb(batch.rgb[0], (self.meshes.render_feats(cams_tform4x4_obj=cam_transf4x4_obj[:1], cams_intr4x4=batch.cam_intr4x4[:1],
                                                          imgs_sizes=batch.size, meshes_ids=pred_class_ids[:1],
                                                          modality=MESH_RENDER_MODALITIES.VERTS_NCDS)[0]).to(dtype=batch.rgb.dtype)))

                    loss = mesh_cam_loss.mean()
                    loss.backward()
                    optim_inference.step()
                    optim_inference.zero_grad()
                    cam_transf4x4_obj = transf4x4_from_pos_and_theta(pos=cam_pos, theta=cam_theta)
                    if pose_iterative_xy_shift:
                        cam_transf4x4_obj[:, :2, 3] += cam_shift_xyz[:, :2]

                results['time_pose_iterative'].append(time.time() - time_before_pose_iterative)
                # logger.info(f"predicted pose iterative took {(time.time() - time_before_pose_iterative):.3f}s")

            if self.config.test.visualize.verts_ncds_in_rgb:
                img = blend_rgb(batch.rgb[0], (self.meshes.render_feats(cams_tform4x4_obj=cam_transf4x4_obj[:1], cams_intr4x4=batch.cam_intr4x4[:1],
                                         imgs_sizes=batch.size, meshes_ids=pred_class_ids[:1],
                                         modality=MESH_RENDER_MODALITIES.VERTS_NCDS)[0]).to(dtype=batch.rgb.dtype))
                results['verts_ncds_in_rgb_' + batch.name[0]] = image_as_wandb_image(img)
                if self.config.test.visualize.live:
                    show_img(img)




            results['time_pose'].append(time.time() - time_pred_class)
            #logger.info(
            #    f"predicted pose, took {(time_pred_pose - time_pred_class):.3f}")

            #cam_tform4x4_cuboid = tform4x4(cam_transf4x4_obj, batch.cuboid_front_tform4x4_obj.inverse())
            #gt_cam_tform4x4_cuboid = tform4x4(batch.cam_tform4x4_obj, batch.cuboid_front_tform4x4_obj.inverse())
            #diff_rot3x3 = rot3x3(gt_cam_tform4x4_cuboid[:, :3, :3].permute(0, 2, 1), cam_tform4x4_cuboid[:, :3, :3])

            diff_rot3x3 = rot3x3(batch.cam_tform4x4_obj[:, :3, :3].permute(0, 2, 1), cam_transf4x4_obj[:, :3, :3])


            try:
                diff_so3_log = pytorch3d.transforms.so3_log_map(diff_rot3x3)
                diffs_so3d_log.append(diff_so3_log)
                diff_rot_angle_rad = torch.norm(diff_so3_log, dim=-1)
            except ValueError:
                logger.warning(f'Cannot calculate deviation in rotation angle due to rot3x3 trace being too small, setting deviation to 0.')
                diff_rot_angle_rad = 0.
            results['rot_diff_rad'].append(diff_rot_angle_rad)
            results['label_gt'].append(batch.label)
            results['label_pred'].append(pred_class_ids)

        for key, val in results.items():
            if key.startswith('time_'):
                results[key] = np.sum(results[key]) / len(dataset_sub)

        logger.info(f'Predicted {len(dataset_sub)} frames.')
        if len(dataloader) > 0:
            results['rot_diff_rad'] = torch.cat(results['rot_diff_rad'], dim=0)
            results['label_gt'] = torch.cat(results['label_gt'], dim=0)
            results['label_pred'] = torch.cat(results['label_pred'], dim=0)

            results['label_acc'] = (results['label_gt'] == results['label_pred']).to(dtype=float).mean()
            results['pose_acc_pi6'] = (results['rot_diff_rad'] < math.pi /6).to(dtype=float).mean()
            results['pose_acc_pi18'] = (results['rot_diff_rad'] < math.pi / 18).to(dtype=float).mean()
            results['pose_err_median'] = 180 / math.pi * results['rot_diff_rad'].median()
            #results['pose_err_mean'] = 180 / math.pi * results['rot_diff_rad'].mean()

            diffs_so3d_log = torch.cat(diffs_so3d_log, dim=0)

            from od3d.cv.geometry.transform import rot3x3_broadcast, so3_exp_map
            diffs_rot3x3_mean = so3_exp_map(diffs_so3d_log.mean(dim=0, keepdim=False))
            diffs_rot3x3 = rot3x3_broadcast(diffs_rot3x3_mean.T, pytorch3d.transforms.so3_exp_map(diffs_so3d_log))
            diffs_rot3 = pytorch3d.transforms.so3_log_map(diffs_rot3x3)
            results['consist_rot_diff_rad'] = torch.norm(diffs_rot3, dim=-1)
            results['consist_pose_err_median'] = 180 / math.pi * results['consist_rot_diff_rad'].median()
            # results['consist_pose_err_mean'] = 180 / math.pi * results['consist_rot_diff_rad'].mean()

            # cmatrix = confusion_matrix(results['label_gt'].detach().cpu().numpy(), results['label_pred'].detach().cpu().numpy())
            # logger.info(f'Confusion matrix:\n {cmatrix} ')

            del results['label_gt']
            del results['label_pred']
            del results['rot_diff_rad']
            del results['consist_rot_diff_rad']
            for key, val in results.items():
                logger.info(f'{key} : {val}')


        return results