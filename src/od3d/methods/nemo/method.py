import logging
import time

import numpy as np
import od3d.io
import pandas as pd
from od3d.benchmark.results import OD3D_Results
from od3d.cv.metric.pose import get_pose_diff_in_rad
from od3d.datasets.dataset import OD3D_Dataset
from od3d.methods.method import OD3D_Method
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
import torch

torch.multiprocessing.set_sharing_strategy("file_system")
from od3d.cv.geometry.transform import se3_exp_map
from od3d.cv.visual.show import imgs_to_img, show_scene2d
from od3d.cv.geometry.objects3d.meshes import Meshes
from pathlib import Path
from od3d.cv.geometry.transform import (
    transf4x4_from_spherical,
    tform4x4,
    inv_tform4x4,
    tform4x4_broadcast,
)
from od3d.cv.visual.show import show_img
from od3d.cv.visual.show import show_bar_chart
from od3d.cv.visual.blend import blend_rgb
from tqdm import tqdm
from od3d.cv.geometry.objects3d.objects3d import PROJECT_MODALITIES

# note: math is actually used by config
import math  # noqa
from od3d.datasets.co3d import CO3D

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

plt.switch_backend("Agg")
from od3d.cv.visual.show import get_img_from_plot
from od3d.cv.visual.draw import draw_text_in_rgb


class VISUAL_MODALITIES(str, ExtEnum):
    PRED_VERTS_NCDS_IN_RGB = "pred_verts_ncds_in_rgb"
    GT_VERTS_NCDS_IN_RGB = "gt_verts_ncds_in_rgb"
    PRED_VS_GT_VERTS_NCDS_IN_RGB = "pred_vs_gt_verts_ncds_in_rgb"
    NET_FEATS_NEAREST_VERTS = "net_feats_nearest_verts"
    SIM_PXL = "sim_pxl"
    SAMPLES = "samples"
    TSNE = "tsne"
    PCA = "pca"
    RECONSTRUCTION_MAP = "reconstruction_map"
    TSNE_PER_IMAGE = "tsne_per_image"


class SIM_FEATS_MESH_WITH_IMAGE(str, ExtEnum):
    VERTS2D = "verts2d"
    RENDERED = "rendered"


class NeMo(OD3D_Method):
    def setup(self):
        pass

    def __init__(
        self,
        config: DictConfig,
        logging_dir,
    ):
        super().__init__(config=config, logging_dir=logging_dir)

        self.device = "cuda:0"

        # init Network
        self.net = OD3D_Model(config.model)

        self.transform_train = SequentialTransform(
            [
                OD3D_Transform.subclasses[
                    config.train.transform.class_name
                ].create_from_config(config=config.train.transform),
                self.net.transform,
            ],
        )
        self.transform_test = SequentialTransform(
            [
                OD3D_Transform.subclasses[
                    config.test.transform.class_name
                ].create_from_config(config=config.test.transform),
                self.net.transform,
            ],
        )

        # if config.train.transform.random_color:
        #     self.transform_train = SequentialTransform([
        #         RandomCenterZoom3D.create_from_config(config.train.transform.random_center_zoom3d),
        #         RGB_Random(),
        #         self.net.transform
        #     ])
        # else:
        #     self.transform_train = SequentialTransform([
        #         RandomCenterZoom3D.create_from_config(config.train.transform.random_center_zoom3d),
        #         self.net.transform
        #     ])
        #
        # self.transform_test = SequentialTransform([
        #         CenterZoom3D.create_from_config(config.test.transform),
        #         self.net.transform
        # ])

        # init Meshes / Features
        self.total_params = sum(p.numel() for p in self.net.parameters())
        self.trainable_params = sum(
            p.numel() for p in self.net.parameters() if p.requires_grad
        )

        # self.path_shapenemo = Path(config.path_shapenemo)
        # self.fpaths_meshes_shapenemo = [self.path_shapenemo.joinpath(cls, '01.off') for cls in config.categories]
        self.fpaths_meshes = [
            self.config.fpaths_meshes[cls] for cls in config.categories
        ]
        fpaths_meshes_tform_obj = self.config.get("fpaths_meshes_tform_obj", None)
        if fpaths_meshes_tform_obj is not None:
            self.fpaths_meshes_tform_obj = [
                fpaths_meshes_tform_obj[cls] for cls in config.categories
            ]
        else:
            self.fpaths_meshes_tform_obj = [None for _ in config.categories]

        self.meshes = Meshes.read_from_ply_files(
            fpaths_meshes=self.fpaths_meshes,
            fpaths_meshes_tforms=self.fpaths_meshes_tform_obj,
            gaussian_splat_enabled=self.config.meshes_gaussian_splat_enabled,
            gaussian_splat_opacity=self.config.meshes_gaussian_splat_opacity,
            geodesic_prob_sigma=self.config.train.geodesic_prob_sigma,
            pt3d_raster_perspective_correct=self.config.meshes_pt3d_raster_perspective_correct,
            gaussian_splat_pts3d_size_rel_to_neighbor_dist=self.config.meshes_gaussian_splat_pts3d_size_rel_to_neighbor_dist,
            feats_objects=True,
            feat_clutter=True,
            feat_dim=self.net.out_dim,
            feats_requires_grad=self.config.train.bank_feats_update != "moving_average"
            and self.config.train.bank_feats_update != "average",
        )

        self.meshes_ranges = self.meshes.get_ranges().detach().cuda()
        self.refine_update_max = (
            torch.Tensor(self.config.inference.refine.dims_grad_max)
            .cuda()[None,]
            .expand(self.meshes_ranges.shape[0], 6)
            .clone()
        )
        self.refine_update_max[:, :3] = (
            self.refine_update_max[:, :3] * self.meshes_ranges
        )

        # self.meshes.rgb = (self.meshes.geodesic_prob[3, :, None].repeat(1, 3)).clamp(0, 1)
        # self.meshes.show()
        # watch_model_in_wandb(self.net, log="all")

        logger.info(f"loading meshes from following fpaths: {self.fpaths_meshes}...")
        # self.meshes.show()
        self.verts_count_max = self.meshes.verts_counts_max
        self.mem_verts_feats_count = len(config.categories) * self.verts_count_max
        self.mem_clutter_feats_count = config.num_noise * config.max_group
        self.mem_count = self.mem_verts_feats_count + self.mem_clutter_feats_count

        self.feats_bank_count = self.verts_count_max * len(self.meshes) + 1

        # self.meshes.set_feats_cat_with_pad(torch.nn.Parameter(torch.randn(size=(self.verts_count_max * len(self.meshes), self.net.feat_dim), device=self.device), requires_grad=True))

        # dict to save estimated tforms, sequence : tform,
        self.seq_obj_tform4x4_est_obj = {}
        self.seq_obj_tform4x4_est_obj_sim = {}

        self.total_params_mesh_clutter = sum(
            p.numel() for p in self.meshes.parameters()
        )
        self.trainable_params_mesh_clutter = sum(
            p.numel() for p in self.meshes.parameters() if p.requires_grad
        )

        if (
            self.config.train.loss == "cross_entropy"
            or self.config.train.loss == "cross_entropy_smooth"
        ):
            self.criterion = torch.nn.CrossEntropyLoss().cuda()
        elif self.config.train.loss == "nll_softmax":
            self.softmax = torch.nn.LogSoftmax(dim=1)
            self.criterion = torch.nn.NLLLoss().cuda()
        elif self.config.train.loss == "nll_clip":
            self.criterion = torch.nn.NLLLoss().cuda()
        elif self.config.train.loss == "nll_affine_to_prob":
            self.criterion = torch.nn.NLLLoss().cuda()
        elif self.config.train.loss == "l2":
            self.criterion = torch.nn.MSELoss().cuda()
        elif self.config.train.loss == "l2_squared":
            self.criterion = torch.nn.MSELoss().cuda()

        # self.net = torch.nn.DataParallel(self.net).cuda()
        self.net.cuda()
        self.meshes.cuda()
        self.net.eval()
        self.meshes.eval()
        self.back_propagate = True

        logger.info(
            f"total params: {self.total_params}, trainable params: {self.trainable_params}",
        )
        logger.info(
            f"total params mesh and clutter: {self.total_params_mesh_clutter}, trainable params mesh and clutter: {self.trainable_params_mesh_clutter}",
        )

        if (
            self.config.train.bank_feats_update == "moving_average"
            or self.config.train.bank_feats_update == "average"
        ):
            if self.trainable_params == 0:
                logger.info("no trainable params, no optimizer needed.")
                self.optim = od3d.io.get_obj_from_config(
                    config=self.config.train.optimizer,
                    params=list(self.net.parameters()),
                )
                self.back_propagate = False
            else:
                self.optim = od3d.io.get_obj_from_config(
                    config=self.config.train.optimizer,
                    params=list(self.net.parameters()),
                )
        else:
            if self.trainable_params == 0:
                self.optim = od3d.io.get_obj_from_config(
                    config=self.config.train.optimizer,
                    params=list(self.meshes.parameters()),
                )
            else:
                self.optim = od3d.io.get_obj_from_config(
                    config=self.config.train.optimizer,
                    params=list(self.meshes.parameters()) + list(self.net.parameters()),
                )

        self.scheduler = od3d.io.get_obj_from_config(
            self.optim,
            config=self.config.train.scheduler,
        )

        # load checkpoint
        if config.get("checkpoint", None) is not None:
            self.load_checkpoint(Path(config.checkpoint))
        elif config.get("checkpoint_old", None) is not None:
            self.load_checkpoint_old(Path(config.checkpoint_old))
        # load_mesh(config.path_shapenemo)

        # self.meshes.show()

        # self.verts_feats = checkpoint["memory"][:self.mem_verts_feats_count].clone().detach().cpu()
        # note: somehow vertices are stored in wrong order of classes (starting with last class tvmonitor until first class aeroplane
        # self.verts_feats = self.verts_feats.reshape(len(self.meshes), self.verts_count_max, -1).flip(dims=(0,)).reshape(len(self.meshes) * self.verts_count_max, -1)
        self.down_sample_rate = self.net.downsample_rate

        color_ = plt.get_cmap("tab20", len(config.categories))
        self.feats_all_colors = []
        for i, cat in enumerate(config.categories):
            self.feats_all_colors.extend([color_(i)] * self.meshes.verts_counts[i])

        # import wandb
        # wandb.watch(self.meshes, log="all", log_freq=1)
        # wandb.watch(self.net, log="all", log_freq=1)

    def calc_sim(self, comb, featsA, featsB):
        """
        Expand and permute a tensor based on the einsum equation.

        Parameters:
            tensor (torch.Tensor): Input tensor.
            equation (str): Einsum equation specifying the dimensions.

        Returns:
            torch.Tensor: Expanded and permuted tensor.
        """
        if self.config.bank_feats_distribution == "von-mises-fisher":
            return torch.einsum(comb, featsA, featsB)
        elif self.config.bank_feats_distribution == "gaussian":
            from od3d.cv.geometry.dist import einsum_cdist

            return -einsum_cdist(comb, featsA, featsB)
        else:
            msg = f"Unknown distribution {self.config.distribution}"
            raise NotImplementedError(msg)

    def load_checkpoint_old(self, path_checkpoint):
        fpaths_meshes_old = list(self.config.fpaths_meshes.values())
        meshes_old = Meshes.load_from_files(fpaths_meshes=fpaths_meshes_old)
        verts_count_max = meshes_old.verts_counts_max
        mem_verts_feats_count = len(fpaths_meshes_old) * verts_count_max
        checkpoint = torch.load(path_checkpoint, map_location="cuda:0")
        self.net.backbone.net = torch.nn.DataParallel(self.net.backbone.net).cuda()
        self.net.backbone.net.load_state_dict(checkpoint["state"], strict=False)
        self.net.backbone.net = self.net.backbone.net.module
        self.clutter_feats = (
            checkpoint["memory"][mem_verts_feats_count:].clone().detach().cpu()
        )
        # self.clutter_feats = self.clutter_feats.mean(dim=0, keepdim=True)
        self.clutter_feats = torch.nn.Parameter(
            self.clutter_feats.to(device=self.device),
            requires_grad=True,
        )

        verts_feats = []
        map_mesh_id_to_old_id = [
            fpaths_meshes_old.index(fpath_mesh) for fpath_mesh in self.fpaths_meshes
        ]
        for i in range(len(self.fpaths_meshes)):
            mesh_old_id = map_mesh_id_to_old_id[i]
            verts_feats.append(
                checkpoint["memory"][
                    mesh_old_id * verts_count_max : (mesh_old_id + 1) * verts_count_max
                ]
                .clone()
                .detach()
                .cpu(),
            )
        self.meshes.set_feats_cat_with_pad(torch.cat(verts_feats, dim=0))

    def save_checkpoint(self, path_checkpoint: Path):
        torch.save(
            {
                "net_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "meshes_feats": self.meshes.state_dict(),
            },
            path_checkpoint,
        )

    def load_checkpoint(self, path_checkpoint):
        checkpoint = torch.load(path_checkpoint)
        self.net.load_state_dict(checkpoint["net_state_dict"])
        self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.meshes.load_state_dict(checkpoint["meshes_feats"])

    @property
    def fpath_checkpoint(self):
        return self.logging_dir.joinpath(self.rfpath_checkpoint)

    @property
    def rfpath_checkpoint(self):
        return Path("nemo.ckpt")

    def extract_features(self, dataset: OD3D_Dataset) -> torch.Tensor:
        self.net.eval()
        dataset.transform = self.transform_train
        dataloader_train = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config.train.dataloader.batch_size,
            shuffle=True,
            collate_fn=dataset.collate_fn,
            num_workers=self.config.train.dataloader.num_workers,
            pin_memory=self.config.train.dataloader.pin_memory,
        )

        net_feats_all = []
        for i, batch in tqdm(enumerate(iter(dataloader_train))):
            batch.to(device=self.device)
            batch.cam_tform4x4_obj = batch.cam_tform4x4_obj.detach()

            # B x F+N x C
            feats2d_net = self.net(batch.rgb)
            feats2d_net_mask = 1.0 * resize(
                batch.rgb_mask,
                H_out=feats2d_net.shape[2],
                W_out=feats2d_net.shape[3],
            )

            mods1d_sampled = self.meshes.sample_with_img2d(
                img2d=feats2d_net,
                img2d_mask=feats2d_net_mask,
                modalities=[PROJECT_MODALITIES.IMG, PROJECT_MODALITIES.MASK],
                cams_tform4x4_obj=batch.cam_tform4x4_obj,
                cams_intr4x4=batch.cam_intr4x4,
                imgs_sizes=batch.size,
                objects_ids=batch.category_id,
                broadcast_batch_and_cams=False,
                down_sample_rate=self.down_sample_rate,
                sample_clutter_count=self.config.num_noise,
                dtype=feats2d_net.dtype,
                device=feats2d_net.device,
            )

            feats1d_sampled = mods1d_sampled[PROJECT_MODALITIES.IMG]
            feats1d_sampled_mask = mods1d_sampled[PROJECT_MODALITIES.MASK]
            net_feats_all.append(feats1d_sampled[feats1d_sampled_mask])

        net_feats_all = torch.cat(net_feats_all, dim=0)

        return net_feats_all

    def train(
        self,
        datasets_train: Dict[str, OD3D_Dataset],
        datasets_val: Dict[str, OD3D_Dataset],
    ):
        score_metric_name = "pose/acc_pi18"  # 'pose/acc_pi18' 'pose/acc_pi6'
        score_ckpt_val = 0.0
        score_latest = 0.0
        self.save_checkpoint(path_checkpoint=self.fpath_checkpoint)

        if "main" in datasets_val.keys():
            dataset_train_sub = datasets_train["labeled"]
        else:
            dataset_train_sub, dataset_val_sub = datasets_train["labeled"].get_split(
                fraction1=1.0 - self.config.train.val_fraction,
                fraction2=self.config.train.val_fraction,
                split=self.config.train.split,
            )
            datasets_val["main"] = dataset_val_sub
        if self.config.model.head.get(
            "pca",
            None,
        ) is not None and self.config.model.head.pca.get("enable", False):
            logger.info("calc pca ...")
            from od3d.cv.cluster.embed import pca
            from od3d.datasets.dataset import OD3D_DATASET_SPLITS

            dataset_pca, _ = dataset_train_sub.get_split(
                fraction1=self.config.model.head.pca.get("subset_fraction", 1.0),
                fraction2=1.0 - self.config.model.head.pca.get("subset_fraction", 1.0),
                split=OD3D_DATASET_SPLITS.RANDOM,
            )
            batch_feature_vectors = self.extract_features(dataset=dataset_pca)
            feature_vector_mean = batch_feature_vectors.mean(dim=0)
            logger.info(f"shape of mean feature vectors:{feature_vector_mean.shape}")
            self.net.head.mean_features = feature_vector_mean
            logger.info(
                f"shape of accumulated feature vectors:{batch_feature_vectors.shape}",
            )
            pca_dim = self.config.model.head.pca.out_dim
            pca_V = pca(batch_feature_vectors, C=pca_dim, return_V=True)

            self.net.head.pca_layer.weight.copy_(pca_V.T)
            self.net.head.pca_enabled = True
            del batch_feature_vectors

        # first validation
        if self.config.train.val:
            for dataset_val_key, dataset_val in datasets_val.items():
                results_val = self.test(dataset_val, val=True)
                results_val.log_with_prefix(prefix=f"val/{dataset_val.name}")
                if dataset_val_key == "main":
                    score_latest = results_val[score_metric_name]
            if not self.config.train.early_stopping or score_latest > score_ckpt_val:
                score_ckpt_val = score_latest
                self.save_checkpoint(path_checkpoint=self.fpath_checkpoint)

        for epoch in range(self.config.train.epochs):
            results_epoch = self.train_epoch(dataset=dataset_train_sub)
            results_epoch.log_with_prefix("train")
            if (
                self.config.train.val
                and self.config.train.epochs_to_next_test > 0
                and epoch % self.config.train.epochs_to_next_test == 0
            ):
                for dataset_val_key, dataset_val in datasets_val.items():
                    results_val = self.test(dataset_val, val=True)
                    results_val.log_with_prefix(prefix=f"val/{dataset_val.name}")
                    if dataset_val_key == "main":
                        score_latest = results_val[score_metric_name]

                if (
                    not self.config.train.early_stopping
                    or score_latest > score_ckpt_val
                ):
                    score_ckpt_val = score_latest
                    self.save_checkpoint(path_checkpoint=self.fpath_checkpoint)

        self.load_checkpoint(path_checkpoint=self.fpath_checkpoint)

    def test(self, dataset: OD3D_Dataset, val=False):
        # note: ensure that checkpoint is saved for checkpointed runs
        if not self.fpath_checkpoint.exists():
            self.save_checkpoint(path_checkpoint=self.fpath_checkpoint)

        logger.info(f"test dataset {dataset.name}")
        self.net.eval()
        self.meshes.eval()

        dataset.transform = self.transform_test
        if not isinstance(dataset, CO3D):
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.config.test.dataloader.batch_size,
                shuffle=False,
                collate_fn=dataset.collate_fn,
                num_workers=self.config.test.dataloader.num_workers,
                pin_memory=self.config.test.dataloader.pin_memory,
            )
            logger.info(f"Dataset contains {len(dataset)} frames.")

        else:
            dict_category_sequences = {
                category: list(sequence_dict.keys())
                for category, sequence_dict in dataset.dict_nested_frames.items()
            }
            dataset_sub = dataset.get_subset_by_sequences(
                dict_category_sequences=dict_category_sequences,
                frames_count_max_per_sequence=self.config.multiview.batch_size,
            )

            dataloader = torch.utils.data.DataLoader(
                dataset=dataset_sub,
                batch_size=self.config.multiview.batch_size,
                shuffle=False,
                collate_fn=dataset_sub.collate_fn,
                num_workers=self.config.test.dataloader.num_workers,
                pin_memory=self.config.test.dataloader.pin_memory,
            )
            logger.info(f"Dataset contains {len(dataset_sub)} frames.")

        results_epoch = OD3D_Results(logging_dir=self.logging_dir)
        for i, batch in tqdm(enumerate(iter(dataloader))):
            batch.to(device=self.device)

            if not isinstance(dataset, CO3D):
                results_batch = self.inference_batch_single_view(batch=batch)
            else:
                results_batch = self.inference_batch_multiview(
                    batch=batch,
                    return_samples_with_sim=True,
                )
            results_epoch += results_batch

            if not val and self.config.test.save_results:
                results_visual_batch = self.get_results_visual_batch(
                    batch=batch,
                    results_batch=results_batch,
                    config_visualize=self.config.test.visualize,
                )
                results_visual_batch.save_visual(prefix=f"test/{dataset.name}")

        count_pred_frames = len(results_epoch["item_id"])
        logger.info(f"Predicted {count_pred_frames} frames.")
        if not val and self.config.test.save_results:
            results_epoch.save_with_dataset(prefix="test", dataset=dataset)

        if not isinstance(dataset, CO3D):
            results_visual = self.get_results_visual(
                results_epoch=results_epoch,
                dataset=dataset,
                config_visualize=self.config.test.visualize,
            )
        else:
            results_visual = self.get_results_visual(
                results_epoch=results_epoch,
                dataset=dataset_sub,
                config_visualize=self.config.test.visualize,
            )

        results_epoch = results_epoch.mean()
        results_epoch += results_visual
        return results_epoch

    def train_epoch(self, dataset: OD3D_Dataset) -> OD3D_Results:
        self.net.train()
        self.meshes.del_pre_rendered()
        self.meshes.train()
        self.optim.zero_grad()
        dataset.transform = self.transform_train
        dataloader_train = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config.train.dataloader.batch_size,
            shuffle=True,
            collate_fn=dataset.collate_fn,
            num_workers=self.config.train.dataloader.num_workers,
            pin_memory=self.config.train.dataloader.pin_memory,
        )

        results_epoch = OD3D_Results(logging_dir=self.logging_dir)
        accumulate_steps = 0
        for i, batch in enumerate(iter(dataloader_train)):
            results_batch: OD3D_Results = self.train_batch(batch=batch)
            results_batch.log_with_prefix("train")
            accumulate_steps += 1
            if accumulate_steps % self.config.train.batch_accumulate_to_next_step == 0:
                if self.back_propagate:
                    self.optim.step()
                    self.meshes.normalize_feats()
                    self.optim.zero_grad()

            results_epoch += results_batch

        self.scheduler.step()
        self.optim.zero_grad()

        # results_epoch.log_dict_to_dir(name=f'train_frames/{dataset.name}')

        results_visual = self.get_results_visual(
            results_epoch=results_epoch,
            dataset=dataset,
            config_visualize=self.config.train.visualize,
        )
        results_epoch = results_epoch.mean()
        results_epoch += results_visual
        return results_epoch

    def train_batch(self, batch) -> OD3D_Results:
        results_batch = OD3D_Results(logging_dir=self.logging_dir)
        B = len(batch)

        batch.to(device=self.device)

        batch.cam_tform4x4_obj = batch.cam_tform4x4_obj.detach()

        # logger.info(f"batch.category_id {batch.category_id}")
        # logger.info(f"batch.size {batch.size}")

        # B x F+N x C
        # logger.info(f"batch.size {batch.size}")
        feats2d_img = self.net(batch.rgb)

        # logger.info(f"batch.size {batch.size}")
        feats2d_img_mask = torch.ones(
            size=(feats2d_img.shape[0], 1, feats2d_img.shape[2], feats2d_img.shape[3]),
        ).to(device=self.device)

        if self.config.train.use_mask_rgb:
            feats2d_img_mask = 1.0 * resize(
                batch.rgb_mask,
                H_out=feats2d_img.shape[2],
                W_out=feats2d_img.shape[3],
            )

        add_clutter = True

        if "cross_entropy" in self.config.train.loss:
            add_other_objects = self.config.train.get("inter_class_loss", True)
            (
                labels,
                labels_mask,
                noise_pxl2d,
                sim,
                feats,
            ) = self.meshes.get_label_and_sim_feats2d_img_to_all(
                feats2d_img=feats2d_img,
                imgs_sizes=batch.size,
                cams_tform4x4_obj=batch.cam_tform4x4_obj,
                cams_intr4x4=batch.cam_intr4x4,
                objects_ids=batch.category_id,
                broadcast_batch_and_cams=False,
                feats2d_img_mask=feats2d_img_mask,
                down_sample_rate=self.down_sample_rate,
                add_clutter=add_clutter,
                add_other_objects=add_other_objects,
                sample_clutter_count=self.config.num_noise,
                dense=self.config.train.dense_loss,
                smooth_labels="smooth" in self.config.train.loss,
                sim_temp=self.config.train.T,
                return_feats=True,
            )

            if labels.isnan().any():
                logger.error("labels contains nan")
            if labels_mask is not None and labels_mask.isnan().any():
                logger.error("labels_mask contains nan")
            if noise_pxl2d is not None and noise_pxl2d.isnan().any():
                logger.error("noise_pxl2d contains nan")
            if sim.isnan().any():
                logger.error("sim contains nan")
            if feats.isnan().any():
                logger.error("feats contains nan")

            if self.config.train.bank_feats_update == "moving_average":
                assert (
                    not self.config.train.dense_loss
                    and not "smooth" in self.config.train.loss
                )
                self.meshes.update_feats_moving_average(
                    labels=labels,
                    labels_mask=labels_mask,
                    feats=feats,
                    objects_ids=batch.category_id,
                    alpha=self.config.train.alpha,
                    add_clutter=add_clutter,
                    add_other_objects=add_other_objects,
                )

            if self.config.train.bank_feats_update == "average":
                assert (
                    not self.config.train.dense_loss
                    and not "smooth" in self.config.train.loss
                )
                self.meshes.update_feats_total_average(
                    labels=labels,
                    labels_mask=labels_mask,
                    feats=feats,
                    objects_ids=batch.category_id,
                    add_clutter=add_clutter,
                    add_other_objects=add_other_objects,
                )

            if labels.dim() == 2:
                labels = labels[labels_mask]
            elif labels.dim() == 3:
                labels = labels.permute(0, 2, 1)
                if labels_mask is not None:
                    labels = labels[labels_mask]

            if sim.dim() == 3:
                sim_batchwise = (sim.max(dim=1).values * labels_mask).flatten(1).sum(
                    dim=-1,
                ) / (labels_mask.flatten(1).sum(dim=-1) + 1e-6).detach()
                sim = sim.permute(0, 2, 1)[labels_mask]
            else:
                sim_batchwise = sim.max(dim=1).values.flatten(1).mean(dim=-1)
            loss = self.criterion(sim, labels)

        elif "sim_max" in self.config.train.loss:
            sim_batchwise = self.meshes.get_sim_render(
                feats2d_img=feats2d_img,
                cams_tform4x4_obj=batch.cam_tform4x4_obj,
                cams_intr4x4=batch.cam_intr4x4,
                objects_ids=batch.category_id,
                broadcast_batch_and_cams=False,
                down_sample_rate=self.down_sample_rate,
                feats2d_img_mask=feats2d_img_mask,
                allow_clutter=False,
                return_sim_pxl=False,
                add_clutter=True,
                add_other_objects=False,
                temp=self.config.train.T,
            )
            sim = sim_batchwise.mean()
            loss = -sim * 10.0
            noise_pxl2d = None
        else:
            raise ValueError(f"Unknown loss {self.config.train.loss}")

        if self.back_propagate:
            loss.backward()

        logger.info(f"loss {loss.item()}")
        results_batch["sim"] = sim_batchwise
        results_batch["noise2d"] = noise_pxl2d
        results_batch["loss"] = loss[None,]
        results_batch["item_id"] = batch.item_id
        results_batch["name_unique"] = batch.name_unique
        results_batch["gt_cam_tform4x4_obj"] = batch.cam_tform4x4_obj
        if self.config.train.bank_feats_update == "average":
            result_visual = OD3D_Results(logging_dir=self.logging_dir)
            bar_image = show_bar_chart(
                int(self.meshes.feats_objects.shape[0]),
                self.meshes.feats_total_count[: self.meshes.feats_objects.shape[0]],
                pts2d_colors=self.feats_all_colors,
                return_visualization=True,
            )
            bar_image_wandb = image_as_wandb_image(
                bar_image,
                caption=f"number of vertices seen in epoch",
            )
            result_visual["vertices_count"] = bar_image_wandb
            result_visual.log_with_prefix(prefix=f"train/visual")

        return results_batch

    """
    def get_sim_cam_tform4x4_obj(self, batch, cam_intr4x4, cam_tform4x4_obj, broadcast_batch_and_cams=False):
        with torch.no_grad():
            net_feats2d = self.net(batch.rgb)
            mesh_feats2d_rendered = self.meshes.render_feats(cams_tform4x4_obj=cam_tform4x4_obj,
                                                             cams_intr4x4=cam_intr4x4,
                                                             imgs_sizes=batch.size, meshes_ids=batch.category_id,
                                                             down_sample_rate=self.down_sample_rate,
                                                             broadcast_batch_and_cams=broadcast_batch_and_cams)

            sim = self.get_sim_feats2d_net_and_rendered(feats2d_net=net_feats2d, feats2d_rendered=mesh_feats2d_rendered)
        return sim
    """

    def inference_batch_single_view(self, batch, return_samples_with_sim=True):
        results = OD3D_Results(logging_dir=self.logging_dir)
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
            feats2d_net_mask = resize(
                batch.rgb_mask,
                H_out=feats2d_net.shape[2],
                W_out=feats2d_net.shape[3],
            )
            if self.config.inference.use_mask_object:
                feats2d_net_mask = (
                    feats2d_net_mask
                    * 1.0
                    * resize(
                        batch.mask,
                        H_out=feats2d_net.shape[2],
                        W_out=feats2d_net.shape[3],
                    )
                )

            time_pred_net_feats2d = time.time()
            # logger.info(
            #    f"predicted net feats2d, took {(time_pred_net_feats2d - time_loaded):.3f}")
            results["time_feats2d"] = (
                torch.Tensor([time_pred_net_feats2d - time_loaded]) / B
            )

            meshes_scores = []
            for mesh_id in range(len(self.meshes)):
                if self.config.inference.get("render_classify", False):
                    sim = self.meshes.get_sim_render(
                        feats2d_img=feats2d_net,
                        cams_tform4x4_obj=batch.cam_tform4x4_obj,
                        cams_intr4x4=batch.cam_intr4x4,
                        objects_ids=torch.LongTensor([mesh_id] * B).to(
                            device=batch.cam_tform4x4_obj.device,
                        ),
                        broadcast_batch_and_cams=False,
                        down_sample_rate=self.down_sample_rate,
                        feats2d_img_mask=feats2d_net_mask,
                        allow_clutter=self.config.inference.allow_clutter,
                        return_sim_pxl=False,
                        add_clutter=self.config.inference.allow_clutter,
                        temp=self.config.T,
                    )
                    sim = sim.squeeze(1)
                else:
                    # logger.info(f'calc score for mesh {self.config.categories[mesh_id]}')
                    sim_feats2d = self.meshes.get_sim_feats2d_img_to_all(
                        feats2d_img=feats2d_net,
                        imgs_sizes=batch.size,
                        cams_tform4x4_obj=None,
                        cams_intr4x4=None,
                        objects_ids=mesh_id,
                        broadcast_batch_and_cams=False,
                        down_sample_rate=self.down_sample_rate,
                        add_clutter=True,
                        add_other_objects=False,
                        dense=True,
                        sim_temp=self.config.train.T,
                    )
                    sim = sim_feats2d.max(dim=1).values.flatten(1).mean(dim=-1)

                meshes_scores.append(sim)
            meshes_scores = torch.stack(meshes_scores, dim=-1)
            pred_class_scores, pred_class_ids = meshes_scores.max(dim=1)

            # logger.info(f'pred class ids {pred_class_ids}')
            time_pred_class = time.time()
            # logger.info(f"predicted class: {self.config.categories[int(pred_class_ids[0])]}, took {(time_pred_class - time_pred_net_feats2d):.3f}")

            results["time_class"] = (
                torch.Tensor([time_pred_class - time_pred_net_feats2d]) / B
            )

            b_cams_multiview_tform4x4_obj, b_cams_multiview_intr4x4 = self.get_samples(
                config_sample=self.config.inference.sample,
                cam_intr4x4=batch.cam_intr4x4,
                cam_tform4x4_obj=batch.cam_tform4x4_obj,
                feats2d_net=feats2d_net,
                categories_ids=batch.category_id,
                feats2d_net_mask=feats2d_net_mask,
            )

            # if self.config.inference.live:
            #     from od3d.cv.visual.show import show_imgs
            #     show_imgs(
            #         blend_rgb(batch.rgb[:1], (self.meshes.render_feats(cams_tform4x4_obj=b_cams_multiview_tform4x4_obj[0],
            #                                                           cams_intr4x4=b_cams_multiview_intr4x4[0],
            #                                                           imgs_sizes=batch.size,
            #                                                           meshes_ids=batch.category_id[:1],
            #                                                           modality=MESH_RENDER_MODALITIES.VERTS_NCDS,
            #                                                           brocacadcast_batch_and_cams=True)[0]).to(dtype=batch.rgb.dtype)), duration=-1)

            logger.info(batch.category_id)
            #  OPTION A: Use 2d gradient of rendered features
            sim = self.meshes.get_sim_render(
                feats2d_img=feats2d_net,
                cams_tform4x4_obj=b_cams_multiview_tform4x4_obj,
                cams_intr4x4=b_cams_multiview_intr4x4,
                objects_ids=batch.category_id,
                broadcast_batch_and_cams=True,
                down_sample_rate=self.down_sample_rate,
                feats2d_img_mask=feats2d_net_mask,
                allow_clutter=self.config.inference.allow_clutter,
                return_sim_pxl=False,
                add_clutter=self.config.inference.allow_clutter,
                temp=self.config.train.T,
            )
            # sim = self.get_sim_feats2d_net_with_cams(
            #     feats2d_net=feats2d_net,
            #     feats2d_net_mask=feats2d_net_mask,
            #     cam_tform4x4_obj=b_cams_multiview_tform4x4_obj,
            #     cam_intr4x4=b_cams_multiview_intr4x4,
            #     categories_ids=batch.category_id,
            #     broadcast_batch_and_cams=True,
            #     only_use_rendered_inliers=self.config.inference.only_use_rendered_inliers,
            #     allow_clutter=self.config.inference.allow_clutter,
            #     use_sigmoid=self.config.inference.use_sigmoid,
            # )

            if return_samples_with_sim:
                results["samples_cam_tform4x4_obj"] = b_cams_multiview_tform4x4_obj
                results["samples_cam_intr4x4"] = b_cams_multiview_intr4x4
                results["samples_sim"] = sim

            mesh_multiple_cams_loss = -sim

            mesh_cam_loss_min_val, mesh_cam_loss_min_id = mesh_multiple_cams_loss.min(
                dim=1,
            )

            cam_tform4x4_obj = (
                b_cams_multiview_tform4x4_obj[:, mesh_cam_loss_min_id]
                .permute(2, 3, 0, 1)
                .diagonal(
                    dim1=-2,
                    dim2=-1,
                )
                .permute(2, 0, 1)
            )

        if self.config.inference.refine.enabled:
            obj_tform6_tmp = torch.nn.Parameter(
                torch.zeros(size=(B, 6)).to(device=cam_tform4x4_obj.device),
                requires_grad=True,
            )
            # transl: 0, 1, 2 rot: 3, 4, 5
            optim_inference = torch.optim.Adam(
                params=[obj_tform6_tmp],
                lr=self.config.inference.optimizer.lr,
                betas=(
                    self.config.inference.optimizer.beta0,
                    self.config.inference.optimizer.beta1,
                ),
            )

            time_before_pose_iterative = time.time()
            cam_tform4x4_obj = tform4x4(
                cam_tform4x4_obj.detach(),
                se3_exp_map(obj_tform6_tmp),
            )

            refine_update_max = self.refine_update_max[batch.category_id].clone()
            for epoch in range(self.config.inference.optimizer.epochs):
                cam_tform4x4_obj = tform4x4(
                    cam_tform4x4_obj.detach(),
                    se3_exp_map(obj_tform6_tmp.detach()),
                )
                obj_tform6_tmp.data[:, :] = 0.0
                cam_tform4x4_obj = tform4x4(
                    cam_tform4x4_obj.detach(),
                    se3_exp_map(obj_tform6_tmp),
                )

                sim, sim_pxl = self.meshes.get_sim_render(
                    feats2d_img=feats2d_net,
                    cams_tform4x4_obj=cam_tform4x4_obj,
                    cams_intr4x4=batch.cam_intr4x4,
                    objects_ids=batch.category_id,
                    broadcast_batch_and_cams=False,
                    down_sample_rate=self.down_sample_rate,
                    feats2d_img_mask=feats2d_net_mask,
                    allow_clutter=self.config.inference.allow_clutter,
                    return_sim_pxl=True,
                    add_clutter=self.config.inference.allow_clutter,
                    temp=self.config.train.T,
                )

                # sim, sim_pxl = self.get_sim_feats2d_net_with_cams(
                #     feats2d_net=feats2d_net,
                #     feats2d_net_mask=feats2d_net_mask,
                #     cam_tform4x4_obj=cam_tform4x4_obj,
                #     cam_intr4x4=batch.cam_intr4x4,
                #     categories_ids=batch.category_id,
                #     return_sim_pxl=True,
                #     broadcast_batch_and_cams=False,
                #     pre_rendered=False,
                #     only_use_rendered_inliers=self.config.inference.only_use_rendered_inliers,
                #     allow_clutter=self.config.inference.allow_clutter,
                #     use_sigmoid=self.config.inference.use_sigmoid,
                # )
                mesh_cam_loss = -sim

                if self.config.inference.live:
                    show_img(
                        blend_rgb(
                            batch.rgb[0],
                            (
                                self.meshes.render(
                                    cams_tform4x4_obj=cam_tform4x4_obj[0 : 0 + 1],
                                    cams_intr4x4=batch.cam_intr4x4[0 : 0 + 1],
                                    imgs_sizes=batch.size,
                                    meshes_ids=batch.category_id[0 : 0 + 1],
                                    modality=PROJECT_MODALITIES.PT3D_NCDS,
                                )[0]
                            ).to(dtype=batch.rgb.dtype),
                        ),
                        duration=1,
                    )

                loss = mesh_cam_loss.sum()
                loss.backward()
                optim_inference.step()
                optim_inference.zero_grad()

                # detach update
                obj_tform6_tmp.data[:, self.config.inference.refine.dims_detached] = 0.0
                # clip update
                refine_update_mask = obj_tform6_tmp.data.abs() > refine_update_max
                obj_tform6_tmp.data[refine_update_mask] = (
                    obj_tform6_tmp.data[refine_update_mask].sign()
                    * refine_update_max[refine_update_mask]
                )

            cam_tform4x4_obj = tform4x4(
                cam_tform4x4_obj.detach(),
                se3_exp_map(obj_tform6_tmp.detach()),
            )

            results["time_pose_iterative"] = (
                torch.Tensor([time.time() - time_before_pose_iterative]) / B
            )

        cam_tform4x4_obj = cam_tform4x4_obj.clone().detach()

        results["time_pose"] = torch.Tensor([time.time() - time_pred_class]) / B
        batch_rot_diff_rad = get_pose_diff_in_rad(
            pred_tform4x4=cam_tform4x4_obj,
            gt_tform4x4=batch.cam_tform4x4_obj,
        )
        results["rot_diff_rad"] = batch_rot_diff_rad
        for cat_id, cat in enumerate(self.config.categories):
            results[f"{cat}_rot_diff_rad"] = batch_rot_diff_rad[
                batch.category_id == cat_id
            ]
        results["label_gt"] = batch.category_id
        results["label_pred"] = pred_class_ids
        results["label_names"] = self.config.categories
        results["sim"] = sim
        results["cam_tform4x4_obj"] = cam_tform4x4_obj
        results["item_id"] = batch.item_id
        results["name_unique"] = batch.name_unique

        return results

    def inference_batch_multiview(self, batch, return_samples_with_sim=True):
        results = OD3D_Results(logging_dir=self.logging_dir)
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
            feats2d_net_mask = resize(
                batch.rgb_mask,
                H_out=feats2d_net.shape[2],
                W_out=feats2d_net.shape[3],
            )
            if self.config.inference.use_mask_object:
                feats2d_net_mask = (
                    feats2d_net_mask
                    * 1.0
                    * resize(
                        batch.mask,
                        H_out=feats2d_net.shape[2],
                        W_out=feats2d_net.shape[3],
                    )
                )

            time_pred_net_feats2d = time.time()
            # logger.info(
            #    f"predicted net feats2d, took {(time_pred_net_feats2d - time_loaded):.3f}")
            results["time_feats2d"] = (
                torch.Tensor([time_pred_net_feats2d - time_loaded]) / B
            )

            meshes_scores = []
            for mesh_id in range(len(self.meshes)):
                # logger.info(f'calc score for mesh {self.config.categories[mesh_id]}')
                bank_feats = torch.cat(
                    [
                        self.meshes.get_feats_with_mesh_id(mesh_id),
                        self.clutter_feats.detach(),
                    ],
                    dim=0,
                )
                # inner_feats2d_net_bank_vts_max_vals = torch.sum(net_feats2d[:, None] * bank_feats[None, :, :, None, None], dim=2, keepdim=True).max(dim=1).values
                out_shape = (
                    feats2d_net.shape[:1] + torch.Size([1]) + feats2d_net.shape[2:]
                )
                inner_feats2d_net_bank_vts_max_vals = (
                    self.calc_sim("bchw,kc->bkhw", feats2d_net, bank_feats)
                    .max(
                        dim=1,
                        keepdim=True,
                    )
                    .values
                )
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
            # logger.info(f"predicted class: {self.config.categories[int(pred_class_ids[0])]}, took {(time_pred_class - time_pred_net_feats2d):.3f}")

            results["time_class"] = (
                torch.Tensor([time_pred_class - time_pred_net_feats2d]) / B
            )

            b_cams_multiview_tform4x4_obj, b_cams_multiview_intr4x4 = self.get_samples(
                config_sample=self.config.inference.sample,
                cam_intr4x4=batch.cam_intr4x4,  # [:1],
                cam_tform4x4_obj=batch.cam_tform4x4_obj,  # [:1],
                feats2d_net=feats2d_net,  # [:1],
                categories_ids=batch.category_id,  # [:1],
                feats2d_net_mask=feats2d_net_mask,  # [:1],
                multiview=True,
            )

            #  OPTION A: Use 2d gradient of rendered features
            sim = self.get_sim_feats2d_net_with_cams(
                feats2d_net=feats2d_net,
                cam_tform4x4_obj=b_cams_multiview_tform4x4_obj,
                cam_intr4x4=b_cams_multiview_intr4x4,
                categories_ids=batch.category_id,
                broadcast_batch_and_cams=True,
                feats2d_net_mask=feats2d_net_mask,
                pre_rendered=False,
                only_use_rendered_inliers=self.config.inference.only_use_rendered_inliers,
                allow_clutter=self.config.inference.allow_clutter,
                use_sigmoid=self.config.inference.use_sigmoid,
            )

            sim = sim.mean(dim=0, keepdim=True).expand(*sim.shape)

            if return_samples_with_sim:
                results["samples_cam_tform4x4_obj"] = b_cams_multiview_tform4x4_obj
                results["samples_cam_intr4x4"] = b_cams_multiview_intr4x4
                results["samples_sim"] = sim

            mesh_multiple_cams_loss = -sim

            mesh_cam_loss_min_val, mesh_cam_loss_min_id = mesh_multiple_cams_loss.min(
                dim=1,
            )

            objs_multiview_tform4x4_cuboid_front = tform4x4_broadcast(
                inv_tform4x4(batch.cam_tform4x4_obj[:1])[:, None],
                b_cams_multiview_tform4x4_obj,
            )
            obj_tform4x4_cuboid_front = objs_multiview_tform4x4_cuboid_front[
                0,
                mesh_cam_loss_min_id[0],
            ]

            # cam_tform4x4_obj = b_cams_multiview_tform4x4_obj[:, mesh_cam_loss_min_id].permute(2, 3, 0, 1).diagonal(
            #    dim1=-2, dim2=-1).permute(2, 0, 1)

        if self.config.inference.refine.enabled:
            obj_tform6_tmp = torch.nn.Parameter(
                torch.zeros(size=(1, 6)).to(device=obj_tform4x4_cuboid_front.device),
                requires_grad=True,
            )

            optim_inference = torch.optim.Adam(
                params=[obj_tform6_tmp],
                lr=self.config.inference.optimizer.lr,
                betas=(
                    self.config.inference.optimizer.beta0,
                    self.config.inference.optimizer.beta1,
                ),
            )

            time_before_pose_iterative = time.time()
            obj_tform4x4_cuboid_front = tform4x4(
                obj_tform4x4_cuboid_front.detach(),
                se3_exp_map(obj_tform6_tmp),
            )

            refine_update_max = self.refine_update_max[batch.category_id[:1]].clone()

            for epoch in range(self.config.inference.optimizer.epochs):
                obj_tform4x4_cuboid_front = tform4x4(
                    obj_tform4x4_cuboid_front.detach(),
                    se3_exp_map(obj_tform6_tmp.detach()),
                )
                obj_tform6_tmp.data[:, :] = 0.0
                obj_tform4x4_cuboid_front = tform4x4(
                    obj_tform4x4_cuboid_front.detach(),
                    se3_exp_map(obj_tform6_tmp),
                )

                sim, sim_pxl = self.get_sim_feats2d_net_with_cams(
                    feats2d_net=feats2d_net,
                    cam_tform4x4_obj=tform4x4_broadcast(
                        batch.cam_tform4x4_obj,
                        obj_tform4x4_cuboid_front,
                    ),
                    cam_intr4x4=batch.cam_intr4x4,
                    categories_ids=batch.category_id,
                    return_sim_pxl=True,
                    broadcast_batch_and_cams=False,
                    feats2d_net_mask=feats2d_net_mask,
                    pre_rendered=False,
                    only_use_rendered_inliers=self.config.inference.only_use_rendered_inliers,
                    allow_clutter=self.config.inference.allow_clutter,
                    use_sigmoid=self.config.inference.use_sigmoid,
                )
                mesh_cam_loss = -sim

                if self.config.inference.live:
                    show_img(
                        blend_rgb(
                            batch.rgb[0],
                            (
                                self.meshes.render(
                                    cams_tform4x4_obj=tform4x4_broadcast(
                                        batch.cam_tform4x4_obj,
                                        obj_tform4x4_cuboid_front,
                                    )[0 : 0 + 1],
                                    cams_intr4x4=batch.cam_intr4x4[0 : 0 + 1],
                                    imgs_sizes=batch.size,
                                    meshes_ids=batch.category_id[0 : 0 + 1],
                                    modality=PROJECT_MODALITIES.VERTS_NCDS,
                                )[0]
                            ).to(dtype=batch.rgb.dtype),
                        ),
                        duration=1,
                    )

                loss = mesh_cam_loss.mean()
                loss.backward()
                optim_inference.step()
                optim_inference.zero_grad()

                # detach update
                obj_tform6_tmp.data[:, self.config.inference.refine.dims_detached] = 0.0
                # clip update
                refine_update_mask = obj_tform6_tmp.data.abs() > refine_update_max
                obj_tform6_tmp.data[refine_update_mask] = (
                    obj_tform6_tmp.data[refine_update_mask].sign()
                    * refine_update_max[refine_update_mask]
                )

            obj_tform4x4_cuboid_front = tform4x4(
                obj_tform4x4_cuboid_front.detach(),
                se3_exp_map(obj_tform6_tmp.detach()),
            )

            results["time_pose_iterative"] = (
                torch.Tensor([time.time() - time_before_pose_iterative]) / B
            )

        obj_tform4x4_cuboid_front = obj_tform4x4_cuboid_front.clone().detach()
        cam_tform4x4_obj = tform4x4_broadcast(
            batch.cam_tform4x4_obj,
            obj_tform4x4_cuboid_front,
        )

        results["time_pose"] = torch.Tensor([time.time() - time_pred_class]) / B
        batch_rot_diff_rad = get_pose_diff_in_rad(
            pred_tform4x4=cam_tform4x4_obj,
            gt_tform4x4=batch.cam_tform4x4_obj,
        )
        results["rot_diff_rad"] = batch_rot_diff_rad
        for cat_id, cat in enumerate(self.config.categories):
            results[f"{cat}_rot_diff_rad"] = batch_rot_diff_rad[
                batch.category_id == cat_id
            ]
        results["label_gt"] = batch.category_id
        results["label_pred"] = pred_class_ids
        results["sim"] = sim.mean(dim=0, keepdim=True).expand(*sim.shape)
        results["cam_tform4x4_obj"] = cam_tform4x4_obj
        results["obj_tform4x4_cuboid_front"] = obj_tform4x4_cuboid_front
        results["item_id"] = batch.item_id
        results["name_unique"] = batch.name_unique

        return results

    def get_results_visual_batch(
        self,
        batch,
        results_batch: OD3D_Results,
        config_visualize: DictConfig,
        dict_name_unique_to_sel_name=None,
        dict_name_unique_to_result_id=None,
        caption_metrics=["sim", "rot_diff_rad"],
    ):
        results_batch_visual = OD3D_Results(logging_dir=self.logging_dir)
        modalities = config_visualize.modalities
        if len(modalities) == 0:
            return results_batch_visual

        down_sample_rate = config_visualize.down_sample_rate
        samples_sorted = config_visualize.samples_sorted
        samples_scores = config_visualize.samples_scores
        samples_scores = config_visualize.samples_scores
        live = config_visualize.live

        with torch.no_grad():
            batch.to(device=self.device)
            B = len(batch)
            if dict_name_unique_to_result_id is not None:
                batch_result_ids = torch.LongTensor(
                    [
                        dict_name_unique_to_result_id[batch.name_unique[b]]
                        for b in range(B)
                    ],
                ).to(device=self.device)
            else:
                batch_result_ids = torch.LongTensor(range(B)).to(device=self.device)

            if "gt_cam_tform4x4_obj" in results_batch.keys():
                batch.cam_tform4x4_obj = results_batch["gt_cam_tform4x4_obj"].to(
                    device=self.device,
                )[batch_result_ids]
            if (
                "noise2d" in results_batch.keys()
                and results_batch["noise2d"] is not None
            ):
                batch.noise2d = results_batch["noise2d"].to(device=self.device)[
                    batch_result_ids
                ]

            if dict_name_unique_to_sel_name is not None:
                batch_sel_names = [
                    dict_name_unique_to_sel_name[batch.name_unique[b]] for b in range(B)
                ]
            else:
                batch_sel_names = [batch.name_unique[b] for b in range(B)]

            batch_names = [batch.name_unique[b] for b in range(B)]
            batch_sel_scores = []
            for b in range(B):
                batch_sel_scores.append(
                    "\n".join(
                        [
                            f"{metric}={results_batch[metric].to(device=self.device)[batch_result_ids[b]].cpu().detach().item():.3f}"
                            for metric in caption_metrics
                            if metric in results_batch.keys()
                        ],
                    ),
                )

            feats2d_net = self.net(batch.rgb)
            feats2d_net_mask = 1.0 * resize(
                batch.rgb_mask,
                H_out=feats2d_net.shape[2],
                W_out=feats2d_net.shape[3],
            )
            sample1d_mods = self.meshes.sample_with_img2d(
                img2d=feats2d_net,
                img2d_mask=feats2d_net_mask,
                cams_intr4x4=batch.cam_intr4x4,
                cams_tform4x4_obj=batch.cam_tform4x4_obj,
                imgs_sizes=batch.size,
                objects_ids=batch.category_id,
                down_sample_rate=self.down_sample_rate,
                modalities=[
                    PROJECT_MODALITIES.PXL2D,
                    PROJECT_MODALITIES.MASK,
                    PROJECT_MODALITIES.IMG,
                ],
            )
            vts2d, vts2d_mask = (
                sample1d_mods[PROJECT_MODALITIES.PXL2D],
                sample1d_mods[PROJECT_MODALITIES.MASK],
            )
            net_feats = sample1d_mods[PROJECT_MODALITIES.IMG]
            N = vts2d.shape[1]
            C = net_feats.shape[2]

            #
            # vts2d_feats2d_net_mask = sample_pxl2d_pts(
            #     feats2d_net_mask,
            #     pxl2d=torch.cat([vts2d], dim=1),
            # )
            # vts2d_mask = vts2d_mask * (vts2d_feats2d_net_mask[:, :, 0] > 0.5)
            # net_feats = sample_pxl2d_pts(
            #     feats2d_net,
            #     pxl2d=torch.cat([vts2d, noise2d], dim=1),
            # )
            #

            if VISUAL_MODALITIES.NET_FEATS_NEAREST_VERTS in modalities:
                logger.info("create net_feats_nearest_verts ...")
                nearest_pt3d_ncds = self.meshes.sample_nearest_to_feats2d_img(
                    feats2d_img=feats2d_net,
                    objects_ids=batch.category_id,
                    modalities=PROJECT_MODALITIES.PT3D_NCDS,
                    add_clutter=True,
                )

                # verts3d = self.get_nearest_verts3d_to_feats2d_net(
                #    feats2d_net=feats2d_net,
                #    categories_ids=batch.category_id,
                #    zero_if_sim_clutter_larger=True,
                # )
                nearest_pt3d_ncds = resize(
                    nearest_pt3d_ncds,
                    scale_factor=self.down_sample_rate / down_sample_rate,
                )
                for b in range(len(batch)):
                    img = blend_rgb(
                        resize(batch.rgb[b], scale_factor=1.0 / down_sample_rate),
                        nearest_pt3d_ncds[b],
                    )
                    results_batch_visual[
                        f"visual/{VISUAL_MODALITIES.NET_FEATS_NEAREST_VERTS}/{batch_sel_names[b]}"
                    ] = image_as_wandb_image(
                        img,
                        caption=f"{batch_sel_names[b]}, {batch_names[b]}, {batch_sel_scores[b]}",
                    )
                    if live:
                        show_img(img)

            if VISUAL_MODALITIES.SAMPLES in modalities:
                logger.info("create samples ...")
                s_cam_tform4x4_obj = results_batch["samples_cam_tform4x4_obj"].to(
                    device=self.device,
                )[batch_result_ids]
                s_cam_intr4x4 = results_batch["samples_cam_intr4x4"].to(
                    device=self.device,
                )[batch_result_ids]
                sim = results_batch["samples_sim"].to(device=self.device)[
                    batch_result_ids
                ]

                """
                s_cam_tform4x4_obj, s_cam_intr4x4 = self.get_samples(config_sample=self.config.inference.sample,
                                                                                           cam_intr4x4=batch.cam_intr4x4,
                                                                                           cam_tform4x4_obj=batch.cam_tform4x4_obj,
                                                                                           feats2d_net=feats2d_net,
                                                                                           categories_ids=batch.category_id)
                sim = self.get_sim_feats2d_net_with_cams(feats2d_net=feats2d_net,
                                                         cam_tform4x4_obj=s_cam_tform4x4_obj,
                                                         cam_intr4x4=s_cam_intr4x4,
                                                         categories_ids=batch.category_id,
                                                         broadcast_batch_and_cams=True)
                """

                ncds = self.meshes.render(
                    cams_tform4x4_obj=s_cam_tform4x4_obj,
                    cams_intr4x4=s_cam_intr4x4,
                    imgs_sizes=batch.size,
                    objects_ids=batch.category_id,
                    down_sample_rate=down_sample_rate,
                    broadcast_batch_and_cams=True,
                    modalities=PROJECT_MODALITIES.PT3D_NCDS,
                )

                for b in range(len(batch)):
                    imgs = ncds[b]
                    imgs_sim = sim[b][:].expand(
                        *sim[b].shape,
                    )  # , *mesh_feats2d_rendered.shape[-2:]
                    if self.config.inference.sample.method == "uniform":
                        imgs = imgs.reshape(
                            self.config.inference.sample.uniform.azim.steps,
                            self.config.inference.sample.uniform.elev.steps,
                            self.config.inference.sample.uniform.theta.steps,
                            *imgs.shape[-3:],
                        )[:, :, :]
                        imgs_sim = imgs_sim.reshape(
                            self.config.inference.sample.uniform.azim.steps,
                            self.config.inference.sample.uniform.elev.steps,
                            self.config.inference.sample.uniform.theta.steps,
                        )[:, :, :]
                    imgs = blend_rgb(
                        resize(batch.rgb[b], scale_factor=1.0 / down_sample_rate),
                        imgs,
                    )

                    if samples_sorted:
                        imgs_sim = imgs_sim.flatten(0)
                        imgs = imgs.reshape(-1, *imgs.shape[-3:])
                        imgs_sim_ids = imgs_sim.sort(descending=True)[1]
                        imgs_sim_ids = imgs_sim_ids[:49]
                        imgs = imgs[imgs_sim_ids]
                        imgs_sim = imgs_sim[imgs_sim_ids]

                    if samples_scores:
                        logger.info("create plot samples scores...")

                        plt.ioff()
                        fig, ax = plt.subplots()
                        ax.plot(
                            imgs_sim.detach().cpu().numpy(),
                            label="sim",
                        )  # density=False would make counts
                        # ax.set_ylim(0., 1.)
                        # ax.ylabel('sim')
                        # ax.xlabel('samples')

                        img = get_img_from_plot(ax=ax, fig=fig)
                        plt.close(fig)
                        img = resize(img, H_out=imgs.shape[-2], W_out=imgs.shape[-1])
                        img = draw_text_in_rgb(
                            img,
                            fontScale=0.4,
                            lineThickness=2,
                            fontColor=(0, 0, 0),
                            text=f"{batch_sel_scores[b]}\nmin={imgs_sim.min().item():.3f}\nmax={imgs_sim.max().item():.3f}",
                        )
                        imgs = torch.cat(
                            [imgs, img[None,].to(device=imgs.device)],
                            dim=0,
                        )
                        # resize(img, )
                        """
                        samples_score_size = imgs.shape[-1] // 5
                        imgs[..., -samples_score_size:, -samples_score_size:] = (
                                255 * imgs_sim.reshape(*imgs_sim.shape, 1, 1, 1).expand(*imgs.shape[:-2],
                                                                                        samples_score_size,
                                                                                        samples_score_size)).to(
                            torch.uint8)
                        """

                    img = imgs_to_img(imgs)
                    results_batch_visual[
                        f"visual/{VISUAL_MODALITIES.SAMPLES}/{batch_sel_names[b]}"
                    ] = image_as_wandb_image(
                        img,
                        caption=f"{batch_sel_names[b]}, {batch_names[b]}, {batch_sel_scores[b]}, min={imgs_sim.min().item():.3f}, max={imgs_sim.max().item():.3f}",
                    )
                    if live:
                        show_img(img)

            if VISUAL_MODALITIES.SIM_PXL in modalities:
                logger.info("create sim pxl...")
                batch_pred_label = results_batch["label_pred"].to(device=self.device)[
                    batch_result_ids
                ]
                batch_pred_cam_tform4x4 = results_batch["cam_tform4x4_obj"].to(
                    device=self.device,
                )[batch_result_ids]

                sim, sim_pxl = self.meshes.get_sim_render(
                    feats2d_img=feats2d_net,
                    cams_intr4x4=batch.cam_intr4x4,
                    cams_tform4x4_obj=batch_pred_cam_tform4x4,
                    objects_ids=batch.category_id,
                    return_sim_pxl=True,
                    broadcast_batch_and_cams=False,
                    allow_clutter=self.config.inference.allow_clutter,
                    add_clutter=self.config.inference.allow_clutter,
                )

                sim_pxl = resize(
                    sim_pxl,
                    scale_factor=self.down_sample_rate / down_sample_rate,
                )
                for b in range(len(batch)):
                    img = blend_rgb(
                        resize(batch.rgb[b], scale_factor=1.0 / down_sample_rate),
                        sim_pxl[b],
                    )
                    results_batch_visual[
                        f"visual/{VISUAL_MODALITIES.SIM_PXL}/{batch_sel_names[b]}"
                    ] = image_as_wandb_image(
                        img,
                        caption=f"{batch_sel_names[b]}, {batch_names[b]}, mean sim={sim[b].item()}",
                    )
                    if live:
                        show_img(img)

            if (
                VISUAL_MODALITIES.PRED_VERTS_NCDS_IN_RGB in modalities
                or VISUAL_MODALITIES.PRED_VS_GT_VERTS_NCDS_IN_RGB in modalities
            ):
                logger.info("create pred verts ncds...")
                batch_pred_label = results_batch["label_pred"].to(device=self.device)[
                    batch_result_ids
                ]
                batch_pred_cam_tform4x4 = results_batch["cam_tform4x4_obj"].to(
                    device=self.device,
                )[batch_result_ids]

                pred_verts_ncds = self.meshes.render(
                    cams_tform4x4_obj=batch_pred_cam_tform4x4,
                    cams_intr4x4=batch.cam_intr4x4,
                    imgs_sizes=batch.size,
                    objects_ids=batch.category_id,
                    down_sample_rate=down_sample_rate,
                    broadcast_batch_and_cams=False,
                    modalities=PROJECT_MODALITIES.PT3D_NCDS,
                )

                if VISUAL_MODALITIES.PRED_VERTS_NCDS_IN_RGB in modalities:
                    for b in range(len(batch)):
                        img = blend_rgb(
                            resize(batch.rgb[b], scale_factor=1.0 / down_sample_rate),
                            pred_verts_ncds[b],
                        )
                        results_batch_visual[
                            f"visual/{VISUAL_MODALITIES.PRED_VERTS_NCDS_IN_RGB}/{batch_sel_names[b]}"
                        ] = image_as_wandb_image(
                            img,
                            caption=f"{batch_sel_names[b]}, {batch_names[b]}, {batch_sel_scores[b]}",
                        )
                        if live:
                            show_img(img)

            if (
                VISUAL_MODALITIES.GT_VERTS_NCDS_IN_RGB in modalities
                or VISUAL_MODALITIES.PRED_VS_GT_VERTS_NCDS_IN_RGB in modalities
            ):
                logger.info("create gt verts ncds...")

                gt_verts_ncds = self.meshes.render(
                    cams_tform4x4_obj=batch.cam_tform4x4_obj,
                    cams_intr4x4=batch.cam_intr4x4,
                    imgs_sizes=batch.size,
                    objects_ids=batch.category_id,
                    down_sample_rate=down_sample_rate,
                    broadcast_batch_and_cams=False,
                    modalities=PROJECT_MODALITIES.PT3D_NCDS,
                )

                if VISUAL_MODALITIES.GT_VERTS_NCDS_IN_RGB in modalities:
                    for b in range(len(batch)):
                        img = blend_rgb(
                            resize(batch.rgb[b], scale_factor=1.0 / down_sample_rate),
                            gt_verts_ncds[b],
                        )
                        if (
                            "noise2d" in results_batch.keys()
                            and results_batch["noise2d"] is not None
                        ):
                            from od3d.cv.visual.draw import draw_pixels

                            img = draw_pixels(
                                img,
                                batch.noise2d[b]
                                * (self.down_sample_rate / down_sample_rate),
                            )

                        results_batch_visual[
                            f"visual/{VISUAL_MODALITIES.GT_VERTS_NCDS_IN_RGB}/{batch_sel_names[b]}"
                        ] = image_as_wandb_image(
                            img,
                            caption=f"{batch_sel_names[b]}, {batch_names[b]}, {batch_sel_scores[b]}",
                        )
                        if live:
                            show_img(img)
            if VISUAL_MODALITIES.PRED_VS_GT_VERTS_NCDS_IN_RGB in modalities:
                logger.info("create pred vs gt verts ncds...")
                for b in range(len(batch)):
                    # if 'noise2d' in results
                    img1 = blend_rgb(
                        resize(batch.rgb[b], scale_factor=1.0 / down_sample_rate),
                        pred_verts_ncds[b],
                    )
                    img2 = blend_rgb(
                        resize(batch.rgb[b], scale_factor=1.0 / down_sample_rate),
                        gt_verts_ncds[b],
                    )
                    img = imgs_to_img(torch.stack([img1, img2], dim=0)[None,])

                    results_batch_visual[
                        f"visual/{VISUAL_MODALITIES.PRED_VS_GT_VERTS_NCDS_IN_RGB}/{batch_sel_names[b]}"
                    ] = image_as_wandb_image(
                        img,
                        caption=f"{batch_sel_names[b]}, {batch_names[b]}, {batch_sel_scores[b]}",
                    )
                    if live:
                        show_img(img)
            if VISUAL_MODALITIES.TSNE_PER_IMAGE in modalities:
                logger.info("create tsne plots for the mesh and image features...")
                from od3d.cv.cluster.embed import tsne

                # normalize feats2d_net

                # fg_feats = torch.masked_select(feats2d_net, feats2d_net_mask >= 0.5).view(feats2d_net.shape[0],feats2d_net.shape[1],-1).permute(0,2,1)
                color_ = plt.get_cmap("tab20", len(self.meshes))

                # bg_feats = torch.masked_select(feats2d_net, feats2d_net_mask < 0.5).view(feats2d_net.shape[0],feats2d_net.shape[1],-1).permute(0,2,1)

                for b in range(len(batch)):
                    mesh_and_image_feats_colors = self.feats_all_colors.copy()
                    mesh_and_image_feats_length = [self.meshes.feats_objects.shape[0]]
                    fg_feats = net_feats[b, :N][vts2d_mask[b]]
                    bg_feats = net_feats[b, N:].reshape(-1, C)
                    print(fg_feats.shape, bg_feats.shape)
                    mesh_and_image_feats_colors.extend(
                        [color_(batch.category_id[b].cpu().numpy())]
                        * fg_feats.shape[0],
                    )
                    mesh_and_image_feats_colors.extend(
                        [(0, 0, 0, 1)] * bg_feats.shape[0],
                    )
                    mesh_and_image_feats_length.extend(
                        [fg_feats.shape[0], bg_feats.shape[0]],
                    )
                    feats_tsne_all = tsne(
                        torch.cat(
                            [self.meshes.feats_objects, fg_feats, bg_feats],
                            dim=0,
                        ),
                        C=2,
                    )
                    print(len(mesh_and_image_feats_colors))
                    print(mesh_and_image_feats_length)

                    img = show_scene2d(
                        [feats_tsne_all],
                        pts2d_colors=[mesh_and_image_feats_colors],
                        pts2d_lengths=mesh_and_image_feats_length,
                        return_visualization=True,
                    )

                    results_batch_visual[
                        f"visual/{VISUAL_MODALITIES.TSNE_PER_IMAGE}/{batch_sel_names[b]}"
                    ] = image_as_wandb_image(
                        img,
                        caption=f"{batch_sel_names[b]}, {batch_names[b]}, {batch_sel_scores[b]}",
                    )
        return results_batch_visual

    def get_results_visual(
        self,
        results_epoch,
        dataset: OD3D_Dataset,
        config_visualize: DictConfig,
        filter_name_unique=True,
        caption_metrics=["sim", "rot_diff_rad"],
    ):
        results = OD3D_Results(logging_dir=self.logging_dir)
        count_best = config_visualize.count_best
        count_worst = config_visualize.count_worst
        count_rand = config_visualize.count_rand
        modalities = config_visualize.modalities
        if len(modalities) == 0:
            return results

        if "rot_diff_rad" in results_epoch.keys():
            rank_metric_name = "rot_diff_rad"
            # sorts values ascending
            epoch_ranked_ids = results_epoch[rank_metric_name].sort(dim=0)[1]
        elif "sim" in results_epoch.keys():
            rank_metric_name = "sim"
            # sorts values descending
            epoch_ranked_ids = results_epoch[rank_metric_name].sort(
                dim=0,
                descending=True,
            )[1]
        else:
            logger.warning(
                f"Could not find a suitable rank metric in results {results_epoch.keys()}",
            )
            return results

        if (
            filter_name_unique
            and "name_unique" in results_epoch.keys()
            and len(results_epoch["name_unique"]) > 0
        ):
            # this only groups the ranked elements depending on their category / sequence etc.
            # https://stackoverflow.com/questions/51408344/pandas-dataframe-interleaved-reordering
            group_names = list(
                {
                    "/".join(name_unique.split("/")[:-1])
                    for name_unique in results_epoch["name_unique"]
                },
            )
            group_ids = [
                group_id
                for result_id in range(len(results_epoch["name_unique"]))
                for group_id, group_name in enumerate(group_names)
                if results_epoch["name_unique"][epoch_ranked_ids[result_id]].startswith(
                    group_name,
                )
            ]
            df = pd.DataFrame(
                np.stack([epoch_ranked_ids, np.array(group_ids)], axis=-1),
                columns=["rank", "group"],
            )
            epoch_ranked_ids = torch.from_numpy(
                df.loc[
                    df.groupby("group").cumcount().sort_values(kind="mergesort").index
                ]["rank"].values,
            )

            df = df[::-1]
            epoch_ranked_ids_worst = torch.from_numpy(
                df.loc[
                    df.groupby("group").cumcount().sort_values(kind="mergesort").index
                ]["rank"].values,
            )
        else:
            epoch_ranked_ids_worst = epoch_ranked_ids.flip(dims=(0,))

        epoch_best_ids = epoch_ranked_ids[:count_best]
        epoch_best_names = [f"best/{i+1}" for i in range(len(epoch_best_ids))]
        epoch_worst_ids = epoch_ranked_ids_worst[:count_worst]
        epoch_worst_names = [
            f"worst/{len(epoch_worst_ids) - i}" for i in range(len(epoch_worst_ids))
        ]
        epoch_rand_ids = epoch_ranked_ids[
            torch.randperm(len(epoch_ranked_ids))[:count_rand]
        ]
        epoch_rand_names = [f"rand/{i+1}" for i in range(len(epoch_rand_ids))]

        sel_rank_ids = torch.cat(
            [epoch_best_ids, epoch_worst_ids, epoch_rand_ids],
            dim=0,
        )
        sel_item_ids = results_epoch["item_id"][sel_rank_ids]
        sel_names = epoch_best_names + epoch_worst_names + epoch_rand_names
        sel_name_unique = [results_epoch["name_unique"][id] for id in sel_rank_ids]
        dict_name_unique_to_result_id = dict(zip(sel_name_unique, sel_rank_ids))
        dict_name_unique_to_sel_name = dict(zip(sel_name_unique, sel_names))
        logger.info("create dataset ...")
        dataset_visualize = dataset.get_subset_with_item_ids(item_ids=sel_item_ids)

        logger.info("create dataloader ...")
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset_visualize,
            batch_size=self.config.test.dataloader.batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            num_workers=self.config.test.dataloader.num_workers,
            pin_memory=self.config.test.dataloader.pin_memory,
        )

        if VISUAL_MODALITIES.TSNE in modalities:
            logger.info("create tsne plots for the mesh...")
            from od3d.cv.cluster.embed import tsne

            feats_tsne = tsne(self.meshes.feats_objects, C=2)

            img = show_scene2d(
                [feats_tsne],
                pts2d_colors=[self.feats_all_colors],
                return_visualization=True,
            )

            results[f"visual/{VISUAL_MODALITIES.TSNE}"] = image_as_wandb_image(
                img,
                caption=f"tsne of mesh feats",
            )

        if VISUAL_MODALITIES.PCA in modalities:
            logger.info("create pca plots for the mesh...")
            from od3d.cv.cluster.embed import pca

            feats_pca = pca(self.meshes.feats_objects, C=2)

            img = show_scene2d(
                [feats_pca],
                pts2d_colors=[self.feats_all_colors],
                return_visualization=True,
            )

            results[f"visual/{VISUAL_MODALITIES.PCA}"] = image_as_wandb_image(
                img,
                caption=f"PCA of mesh feats",
            )
        for i, batch in tqdm(enumerate(iter(dataloader))):
            results += self.get_results_visual_batch(
                batch,
                results_epoch,
                config_visualize=config_visualize,
                dict_name_unique_to_sel_name=dict_name_unique_to_sel_name,
                caption_metrics=caption_metrics,
                dict_name_unique_to_result_id=dict_name_unique_to_result_id,
            )

        return results

    def get_nearest_corresp2d3d(
        self,
        feats2d_net,
        meshes_ids,
        feats2d_net_mask: torch.Tensor = None,
    ):
        (
            nearest_verts3d,
            sim_texture,
            sim_clutter,
        ) = self.get_nearest_verts3d_to_feats2d_net(
            feats2d_net,
            meshes_ids,
            return_sim_texture_and_clutter=True,
        )
        nearest_verts2d = get_pxl2d_like(nearest_verts3d.permute(0, 2, 3, 1)).permute(
            0,
            3,
            1,
            2,
        )
        # H=sim_clutter.shape[1], W=sim_clutter.shape[2], dtype=sim_nearest_texture_verts.dtype, device=sim_nearest_texture_verts.device)[None,].expand()
        # prob_corresp2d3d = ((sim_texture + 1) / 2) * (1-((sim_clutter+1)/ 2)) #  (sim_clutter < sim_texture) * sim_texture
        prob_corresp2d3d = torch.exp(sim_texture) / (
            torch.exp(sim_texture) + torch.exp(sim_clutter)
        )

        prob_corresp2d3d *= feats2d_net_mask

        return nearest_verts3d, nearest_verts2d, prob_corresp2d3d

    def get_samples(
        self,
        config_sample: DictConfig,
        cam_intr4x4: torch.Tensor,
        cam_tform4x4_obj: torch.Tensor,
        feats2d_net: torch.Tensor,
        categories_ids: torch.LongTensor,
        feats2d_net_mask: torch.Tensor = None,
        multiview=False,
    ):
        if multiview:
            b_cams_multiview_tform4x4_obj, b_cams_multiview_intr4x4 = self.get_samples(
                config_sample=config_sample,
                cam_intr4x4=cam_intr4x4[:1],
                cam_tform4x4_obj=cam_tform4x4_obj[:1],
                feats2d_net=feats2d_net[:1],
                categories_ids=categories_ids[:1],
                feats2d_net_mask=feats2d_net_mask[:1],
            )
            scale = (
                b_cams_multiview_tform4x4_obj[..., 2, 3]
                / cam_tform4x4_obj[:, None, 2, 3]
            )
            cam_tform4x4_obj_scaled = (
                cam_tform4x4_obj[:, None]
                .clone()
                .expand(
                    cam_tform4x4_obj.shape[0],
                    *b_cams_multiview_tform4x4_obj.shape[1:],
                )
                .clone()
            )

            # use rot. scaling
            # cam_tform4x4_obj_scaled[:, :, :3, :3] = cam_tform4x4_obj_scaled[:, :, :3, :3] * scale[:, :, None, None]
            # use transl. scaling
            # cam_tform4x4_obj_scaled[:, :, :3, 3] = cam_tform4x4_obj_scaled[:, :, :3, 3] * scale[:, :, None]
            # use depth scaling
            cam_tform4x4_obj_scaled[:, :, 2, 3] = (
                cam_tform4x4_obj_scaled[:, :, 2, 3] * scale[:, :]
            )

            objs_multiview_tform4x4_cuboid_front = tform4x4_broadcast(
                inv_tform4x4(cam_tform4x4_obj_scaled[:1]),
                b_cams_multiview_tform4x4_obj,
            )
            b_cams_multiview_tform4x4_obj = tform4x4_broadcast(
                cam_tform4x4_obj_scaled,
                objs_multiview_tform4x4_cuboid_front,
            )
            b_cams_multiview_intr4x4 = b_cams_multiview_intr4x4.expand(
                *b_cams_multiview_tform4x4_obj.shape,
            )

        else:
            B = len(feats2d_net)
            if config_sample.method == "uniform":
                azim = torch.linspace(
                    start=eval(config_sample.uniform.azim.min),
                    end=eval(config_sample.uniform.azim.max),
                    steps=config_sample.uniform.azim.steps,
                ).to(
                    device=self.device,
                )  # 12
                elev = torch.linspace(
                    start=eval(config_sample.uniform.elev.min),
                    end=eval(config_sample.uniform.elev.max),
                    steps=config_sample.uniform.elev.steps,
                ).to(
                    device=self.device,
                )  # start=-torch.pi / 6, end=torch.pi / 3, steps=4
                theta = torch.linspace(
                    start=eval(config_sample.uniform.theta.min),
                    end=eval(config_sample.uniform.theta.max),
                    steps=config_sample.uniform.theta.steps,
                ).to(
                    device=self.device,
                )  # -torch.pi / 6, end=torch.pi / 6, steps=3

                # dist = torch.linspace(start=eval(config_sample.uniform.dist.min), end=eval(config_sample.uniform.dist.max), steps=config_sample.uniform.dist.steps).to(
                #    device=self.device)
                dist = torch.linspace(start=1.0, end=1.0, steps=1).to(
                    device=self.device,
                )

                azim_shape = azim.shape
                elev_shape = elev.shape
                theta_shape = theta.shape
                dist_shape = dist.shape
                in_shape = azim_shape + elev_shape + theta_shape + dist_shape
                azim = azim[:, None, None, None].expand(in_shape).reshape(-1)
                elev = elev[None, :, None, None].expand(in_shape).reshape(-1)
                theta = theta[None, None, :, None].expand(in_shape).reshape(-1)
                dist = dist[None, None, None, :].expand(in_shape).reshape(-1)
                cams_multiview_tform4x4_cuboid = transf4x4_from_spherical(
                    azim=azim,
                    elev=elev,
                    theta=theta,
                    dist=dist,
                )

                C = len(cams_multiview_tform4x4_cuboid)

                b_cams_multiview_tform4x4_obj = cams_multiview_tform4x4_cuboid[
                    None,
                ].repeat(B, 1, 1, 1)

                # assumption 1: distance translation to object is known
                # b_cams_multiview_tform4x4_obj[:, :, 2, 3] = cam_tform4x4_obj[:, None].repeat(1, C, 1, 1)[:, :, 2, 3]
                # logger.info(f'dist {batch.cam_tform4x4_obj[:, 2, 3]}')
                # assumption 2: translation to object is known
                b_cams_multiview_tform4x4_obj[:, :, :3, 3] = cam_tform4x4_obj[
                    :,
                    None,
                ].repeat(1, C, 1, 1)[:, :, :3, 3]

                b_cams_multiview_intr4x4 = cam_intr4x4[:, None].repeat(1, C, 1, 1)

            elif config_sample.method == "epnp3d2d":
                (
                    nearest_verts3d,
                    nearest_verts2d,
                    prob_well_corresp,
                ) = self.get_nearest_corresp2d3d(
                    feats2d_net=feats2d_net,
                    meshes_ids=categories_ids,
                    feats2d_net_mask=feats2d_net_mask,
                )
                H, W = feats2d_net.shape[2:]
                prob_well_corresp = prob_well_corresp.flatten(1)
                if (prob_well_corresp.sum(dim=-1) == 0).any():
                    logger.warning(
                        f"No texture similarity is larger than the clutter similarity",
                    )
                prob_well_corresp[prob_well_corresp.sum(dim=-1) == 0] = 1.0

                K = config_sample.epnp3d2d.count_cams
                N = config_sample.epnp3d2d.count_pts

                masks_in_ids = torch.multinomial(
                    prob_well_corresp,
                    num_samples=K * N,
                ).reshape(-1, K, N)

                masks_in = torch.zeros(
                    size=(B, K, H * W),
                    device=self.device,
                    dtype=torch.bool,
                )
                for b in range(B):
                    for k in range(K):
                        masks_in[b, k, masks_in_ids[b, k]] = True
                masks_in = masks_in.reshape(B, K, H, W)
                b_cams_multiview_tform4x4_obj = (
                    batchwise_fit_se3_to_corresp_3d_2d_and_masks(
                        masks_in=masks_in,
                        pts1=nearest_verts3d,
                        pxl2=nearest_verts2d,
                        proj_mat=cam_intr4x4[
                            :,
                            :2,
                            :3,
                        ]
                        / self.down_sample_rate,
                        method="cpu-epnp",
                    )
                )
                b_cams_multiview_intr4x4 = cam_intr4x4[:, None].repeat(1, K, 1, 1)
                b_cams_multiview_tform4x4_obj[
                    b_cams_multiview_tform4x4_obj.flatten(2).isinf().any(dim=2),
                    :,
                    :,
                ] = torch.eye(4, device=b_cams_multiview_tform4x4_obj.device)
                b_cams_multiview_tform4x4_obj[
                    (b_cams_multiview_tform4x4_obj[:, :, 3, :3] != 0.0).any(dim=-1),
                    :,
                    :,
                ] = torch.eye(4, device=b_cams_multiview_tform4x4_obj.device)

            else:
                raise NotImplementedError

        if config_sample.depth_from_box:
            from od3d.cv.geometry.fit.depth_from_mesh_and_box import (
                depth_from_mesh_and_box,
            )

            b_cams_multiview_tform4x4_obj[..., 2, 3] = depth_from_mesh_and_box(
                b_cams_multiview_intr4x4=cam_intr4x4,
                b_cams_multiview_tform4x4_obj=b_cams_multiview_tform4x4_obj,
                meshes=self.meshes,
                labels=categories_ids,
                mask=feats2d_net_mask,
                downsample_rate=self.down_sample_rate,
                multiview=multiview,
            )
        return b_cams_multiview_tform4x4_obj, b_cams_multiview_intr4x4

    def visualize_test(self):
        pass
