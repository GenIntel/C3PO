
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

from nemo.models.KeypointRepresentationNet import NetE2E
from od3d.cv.geometry.mesh import Meshes
from pathlib import Path
from od3d.cv.geometry.transform import transf4x4_from_spherical
from od3d.cv.visual.show import show_imgs, show_img
import torchvision
from od3d.cv.visual.blend import blend_rgb
from od3d.cv.geometry.transform import transf4x4_from_pos_and_theta
from sklearn.metrics import confusion_matrix

class NeMo(OD3DMethod):
    def __init__(
        self,
        config: DictConfig
    ):
        super().__init__(config=config)
        self.net = NetE2E(
            net_type=config.backbone.net_type,
            local_size=[config.backbone.local_size[0], config.backbone.local_size[1]],
            output_dimension=config.backbone.output_dimension,
            reduce_function=None,
            n_noise_points=config.num_noise,
            pretrain=True,
        )
        self.device = 'cuda:0'
        self.net = torch.nn.DataParallel(self.net).cuda()
        self.net.eval()
        checkpoint = torch.load(config.checkpoint, map_location="cuda:0")
        self.net.load_state_dict(checkpoint["state"], strict=False)
        self.total_params = sum(p.numel() for p in self.net.parameters())
        self.path_shapenemo = Path(config.path_shapenemo)
        #meshes = (Mesh()
        #test_dataset
        self.fpaths_meshes_shapenemo = [self.path_shapenemo.joinpath(cls, '01.off') for cls in config.classes]
        self.meshes = Meshes(fpaths_meshes=self.fpaths_meshes_shapenemo, device=self.device)
        #load_mesh(config.path_shapenemo)
        self.verts_count_max = self.meshes.verts_count_max()
        self.mem_verts_feats_count = len(config.classes) * self.verts_count_max
        self.mem_clutter_feats_count = config.num_noise * config.max_group
        self.mem_count = self.mem_verts_feats_count + self.mem_clutter_feats_count
        self.verts_feats = checkpoint["memory"][:self.mem_verts_feats_count].clone().detach().cpu()
        # note: somehow vertices are stored in wrong order of classes (starting with last class tvmonitor until first class aeroplane
        self.verts_feats = self.verts_feats.reshape(len(self.meshes), self.verts_count_max, -1).flip(dims=(0,)).reshape(len(self.meshes) * self.verts_count_max, -1)
        self.meshes.add_feats(self.verts_feats)
        self.clutter_feats = checkpoint["memory"][self.mem_verts_feats_count:].clone().detach().cpu()
        self.clutter_feats_mean = self.clutter_feats.mean(dim=0)

        self.down_sample_rate = self.config.down_sample_rate
        self.set_distance = 5.0
        self.classification_size = [512, 512]
        self.render_image_size = max(self.classification_size) // self.down_sample_rate
        self.map_shape = (
            self.classification_size[0] // self.down_sample_rate,
            self.classification_size[1] // self.down_sample_rate,
        )

    def setup(self):
        pass

    def train(self, train_dataset: OD3D_Dataset):
        cfg = self.config
        # dataset_kwargs = {"data_type": "train", "category": cfg.args.cate}
        # train_dataset = construct_class_by_name(**cfg.dataset, **dataset_kwargs)
        if train_dataset.config.sampler is not None:
            train_dataset_sampler = RandomSampler(train_dataset, replacement=True, num_samples=int(1e10))
            shuffle = False
        else:
            train_dataset_sampler = None
            shuffle = True
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=shuffle,
            num_workers=cfg.training.workers,
            sampler=train_dataset_sampler
        )
        logging.info(f"Number of training images: {len(train_dataset)}")

        # Debug dataset
        if cfg.training.visualize_training_data:
            for i in range(10):
                train_dataset.debug(
                    np.random.randint(0, len(train_dataset)), save_dir=cfg.args.save_dir
                )

        if cfg.args.dry_run:
            exit()

        model = construct_class_by_name(
            **cfg.model, cfg=cfg, cate=cfg.args.cate, mode='train',
            image_sizes=cfg.dataset.image_sizes)

        logging.info("Start training")
        for epo in range(cfg.training.total_epochs):
            num_iterations = int(cfg.training.scale_iterations_per_epoch * len(train_dataloader))
            for i, sample in enumerate(train_dataloader):
                if i >= num_iterations:
                    break
                loss_dict = model.train(sample)
                if cfg.use_wandb:
                    wandb.log(loss_dict)

            if (epo + 1) % cfg.training.log_interval == 0:
                logging.info(
                    f"[Epoch {epo+1}/{cfg.training.total_epochs}] {model.get_training_state()}"
                )

            if (epo + 1) % cfg.training.ckpt_interval == 0:
                torch.save(model.get_ckpt(epoch=epo+1, cfg=cfg.asdict()), os.path.join(cfg.args.save_dir, "ckpts", f"model_{epo+1}.pth"))
            model.step_scheduler()

    def calc_loss_feat2d_net_bank(self, feats2d_net, feats2d_bank):
        pass
    def calc_loss_feat2d_net_rendered(self, feats2d_net, feats2d_rendered):
        pass
    def test(self, dataset: OD3D_Dataset):

        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False,
                                                 collate_fn=dataset.collate_fn)

        logging.info(f"Dataset contains {len(dataset)} frames.")

        results = {
            'rot_diff_rad': [],
            'label_gt': [],
            'label_pred': [],
        }
        for i, batch in enumerate(iter(dataloader)):
            # batch.visualize()

            self.trans = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            rgb = self.trans(batch.rgb / 255.)
            with torch.no_grad():
                net_feats2d = self.net.module.forward_test(rgb)
            self.meshes.to(batch.device)
            self.clutter_feats = self.clutter_feats.to(batch.device)
            meshes_scores = []
            for mesh_id in range(len(self.meshes)):
                # logger.info(f'calc score for mesh {self.config.classes[mesh_id]}')
                bank_feats = torch.cat([self.meshes.feats[mesh_id], self.clutter_feats], dim=0)
                inner_feats2d_net_bank_vts_max_vals = torch.sum(net_feats2d[:, None] * bank_feats[None, :, :, None, None], dim=2, keepdim=True).max(dim=1).values
                # inner_feats2d_net_bank_vts_max_vals, inner_feats2d_net_bank_vts_max_ids = inner_feats2d.max(dim=1)
                # show_img(self.meshes.verts[mesh_id][inner_feats2d_net_bank_vts_max_ids[0, 0]].permute(2, 0, 1), normalize=True)
                # show_img(inner_feats2d_net_bank_vts_max_vals[0])
                mesh_score = inner_feats2d_net_bank_vts_max_vals.flatten(1).mean(dim=1)
                #clutter_score = inner_feats2d[:, -self.clutter_feats.shape[0]:].mean(dim=1).flatten(1).mean(dim=1)
                #mesh_score -= clutter_score
                meshes_scores.append(mesh_score)
            meshes_scores = torch.stack(meshes_scores, dim=-1)
            pred_class_scores, pred_class_ids = meshes_scores.max(dim=1)
            logger.info(f"predicted {self.config.classes[int(pred_class_ids[0])]}")

            # show_img(inner_feats2d_net_bank_vts_max[0])
            azim = torch.linspace(start=0, end=torch.pi * 2 - (torch.pi * 2) / 13, steps=12).to(device=batch.device) # 12
            elev = torch.linspace(start=-torch.pi / 6, end=torch.pi / 3, steps=4).to(device=batch.device) # start=-torch.pi / 6, end=torch.pi / 3, steps=4
            theta = torch.linspace(start=-torch.pi / 6, end=torch.pi / 6, steps=3).to(device=batch.device) # -torch.pi / 6, end=torch.pi / 6, steps=3
            dist = torch.linspace(start=5., end=5., steps=1).to(device=batch.device)

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
            cams_intr4x4 = batch.cam_intr4x4[0][None,]
            imgs_sizes = batch.size[0][None,]

            cams_tform4x4_obj = transf4x4_from_spherical(azim=azim, elev=elev, theta=theta, dist=dist)
            cams_intr4x4 = cams_intr4x4.expand(cams_tform4x4_obj.shape) / self.down_sample_rate
            imgs_sizes = imgs_sizes.expand(cams_tform4x4_obj.shape[:-2] + torch.Size([2])) // self.down_sample_rate

            # TODO: from here on no batch works...
            mesh_feats2d_rendered = self.meshes.render_feats(cams_tform4x4_obj=cams_tform4x4_obj, cams_intr4x4=cams_intr4x4, imgs_sizes=imgs_sizes, meshes_ids=[int(pred_class_ids[0])], replace_feats_with_verts3d=False)[0]
            # show_imgs(meshes_feats2d_rendered)

            # self.calc_loss_feat2d_net_rendered(feats2d_net=mesh_feats2d_rendered, feats2d_rendered=net_feats2d)
            inner_feats2d_net_mesh_multiple_cams = torch.sum(net_feats2d[:, None] * mesh_feats2d_rendered[None, :], dim=-3, keepdim=True)
            inner_feats2d_net_clutter = torch.sum(net_feats2d[:, None] * self.clutter_feats[None, :, :, None, None], dim=2, keepdim=True).mean(dim=1)

            inner_feats2d_net_bank_multiple_cams =inner_feats2d_net_mesh_multiple_cams # torch.max(inner_feats2d_net_mesh_multiple_cams, inner_feats2d_net_clutter[:, None,])
            mesh_multiple_cams_loss = 1 - (inner_feats2d_net_bank_multiple_cams.flatten(-3).mean(dim=2)) #  - inner_feats2d_net_clutter.flatten(1).mean())
            mesh_cam_loss_min_val, mesh_cam_loss_min_id = mesh_multiple_cams_loss.min(dim=1)

            init_cams_tform4x4_obj = cams_tform4x4_obj[mesh_cam_loss_min_id]
            init_theta = theta[mesh_cam_loss_min_id]

            init_cams_tform4x4_obj_neg = init_cams_tform4x4_obj.clone()
            init_cams_tform4x4_obj_neg[:, 1:3] = -init_cams_tform4x4_obj_neg[:, 1:3]
            # cam_pos = init_cams_tform4x4_obj.inverse()[:, :3, 3].clone()
            cam_pos = torch.nn.Parameter(init_cams_tform4x4_obj.inverse()[:, :3, 3].clone(), requires_grad=True)
            cam_theta = torch.nn.Parameter(init_theta.clone(), requires_grad=True)
            cam_transf4x4_obj = transf4x4_from_pos_and_theta(pos=cam_pos, theta=cam_theta)

            #show_img(self.meshes.render_feats(cams_tform4x4_obj=init_cams_tform4x4_obj, cams_intr4x4=batch.cam_intr4x4 / 2,
            #                                imgs_sizes=batch.size // 2, meshes_ids=[int(pred_class_ids[0])],
            #                                replace_feats_with_verts3d=True)[0, 0])
            #show_img(self.meshes.render_feats(cams_tform4x4_obj=cam_transf4x4_obj, cams_intr4x4=batch.cam_intr4x4 / 2,
            #                                  imgs_sizes=batch.size //2, meshes_ids=[int(pred_class_ids[0])],
            #                                  replace_feats_with_verts3d=True)[0, 0])


            optim = torch.optim.Adam(
                params=[cam_pos, cam_theta],
                lr=self.config.inference.optimizer.lr,
                betas=(self.config.inference.optimizer.beta0, self.config.inference.optimizer.beta1),
            )

            for epoch in range(self.config.inference.optimizer.epochs):

                mesh_feats2d_rendered = self.meshes.render_feats(cams_tform4x4_obj=cam_transf4x4_obj, cams_intr4x4=batch.cam_intr4x4 / self.down_sample_rate,
                                         imgs_sizes=batch.size // self.down_sample_rate, meshes_ids=[int(pred_class_ids[0])],
                                         replace_feats_with_verts3d=False)[:, 0]

                inner_feats2d_net_mesh = torch.sum(net_feats2d * mesh_feats2d_rendered, dim=-3, keepdim=True)
                inner_feats2d_net_clutter = torch.sum(net_feats2d[:, None] * self.clutter_feats[None, :, :, None, None], dim=2, keepdim=True).mean(dim=1)
                inner_feats2d_net_bank = torch.max(inner_feats2d_net_mesh, inner_feats2d_net_clutter)
                mesh_cam_loss = 1 - (inner_feats2d_net_bank.flatten(-3).mean(dim=-1) - inner_feats2d_net_clutter.flatten(1).mean())
                #show_img(blend_rgb(batch.rgb[0], (self.meshes.render_feats(cams_tform4x4_obj=cam_transf4x4_obj, cams_intr4x4=batch.cam_intr4x4,
                #                                  imgs_sizes=batch.size, meshes_ids=[int(pred_class_ids[0])],
                #                                  replace_feats_with_verts3d=True)[0, 0] * 255).to(dtype=batch.rgb.dtype)))

                loss = mesh_cam_loss.mean()
                loss.backward()
                optim.step()
                optim.zero_grad()
                cam_transf4x4_obj = transf4x4_from_pos_and_theta(pos=cam_pos, theta=cam_theta)


            diff_rot3x3 = torch.matmul(batch.cam_tform4x4_obj[:, :3, :3].permute(0, 2, 1), cam_transf4x4_obj[:, :3, :3])
            diff_so3_log = pytorch3d.transforms.so3_log_map(diff_rot3x3)
            diff_rot_angle_rad = torch.norm(diff_so3_log, dim=-1)

            results['rot_diff_rad'].append(diff_rot_angle_rad)
            results['label_gt'].append(batch.label)
            results['label_pred'].append(pred_class_ids)


        results['rot_diff_rad'] = torch.cat(results['rot_diff_rad'], dim=0)
        results['label_gt'] = torch.cat(results['label_gt'], dim=0)
        results['label_pred'] = torch.cat(results['label_pred'], dim=0)

        results['label_acc'] = (results['label_gt'] == results['label_pred']).to(dtype=float).mean()
        results['pose_acc_pi6'] = (results['rot_diff_rad'] < math.pi /6).to(dtype=float).mean()
        results['pose_acc_pi6'] = (results['rot_diff_rad'] < math.pi / 18).to(dtype=float).mean()
        results['pose_err_median'] = 180 / math.pi * results['rot_diff_rad'].median()

        for key, val in results.items():
            logger.info(f'{key} : {val}')
        cmatrix = confusion_matrix(results['label_gt'].detach().cpu().numpy(), results['label_pred'].detach().cpu().numpy())
        logger.info(f'Confusion matrix:\n {cmatrix} ')