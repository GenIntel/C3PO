import time
from typing import List
from od3d.methods.method import OD3DMethod
from od3d.datasets.dataset import OD3D_Dataset
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
from od3d.methods.nemo_incremental.backbone import OD3D_Backbone
from functools import partial

from od3d.cv.geometry.grid import get_pxl2d_like
from od3d.cv.geometry.fit3d2d import batchwise_fit_se3_to_corresp_3d_2d_and_masks #  fit_se3_to_corresp_3d_2d_and_masks
from od3d.cv.geometry.transform import inv_tform4x4
from od3d.methods.nemo_incremental.nemo import NeMo


class NeMo_Incremental(NeMo):
    def __init__(
        self,
        config: DictConfig,
        logging_dir,
    ):
        super().__init__(config=config, logging_dir=logging_dir)

    def get_dataset_sub(self, dataset):
        if self.config.train.epochs_to_next_forget_est_tforms4x4 > 0 and e % self.config.train.epochs_to_next_forget_est_tforms4x4 == 0:
            previous_sequences_names = self.seq_labeled + self.seq_filtered
            proposals_sequences_names = self.seq_filtered + list(
                set(dataset.sequences_names) - set(previous_sequences_names))[
                                                            :self.config.train.sequences_new_proposals_count]

            self.seq_obj_tform4x4_est_obj = {}
            self.seq_obj_tform4x4_est_obj_sim = {}
            self.seq_obj_tform4x4_est_obj_transl_consist = {}
            self.seq_obj_tform4x4_est_obj_rot_consist = {}
            self.seq_filtered = []

            for s, seq in enumerate(proposals_sequences_names):

                seq_dataset = dataset.get_subset_by_sequences([seq],
                                                              frames_count_max_per_sequence=self.config.train.sequences_tform4x4_estimated_frames_count)
                seq_dataset.transform = self.transform_test
                dataloader_train_seq = torch.utils.data.DataLoader(dataset=seq_dataset,
                                                                   batch_size=self.config.test.dataloader.batch_size,
                                                                   shuffle=False,
                                                                   collate_fn=dataset.collate_fn,
                                                                   num_workers=self.config.test.dataloader.num_workers,
                                                                   pin_memory=self.config.test.dataloader.pin_memory)
                logger.info(f'estimating obj_tform4x4_obj_est for {seq}')
                seq_obj_tform4x4_est_obj = []
                seq_obj_tform4x4_est_obj_sim = []
                for i, batch in enumerate(iter(dataloader_train_seq)):
                    batch.to(device=self.device)
                    cam_tform4x4_obj_est, est_sim, results_batch = self.inference_batch(batch,
                                                                                        config=self.config.inference,
                                                                                        visual_names_unique=[
                                                                                            batch.name_unique[j]
                                                                                            for j in
                                                                                            range(len(batch))])
                    seq_obj_tform4x4_est_obj.append(
                        tform4x4(inv_tform4x4(batch.cam_tform4x4_obj), cam_tform4x4_obj_est))
                    seq_obj_tform4x4_est_obj_sim.append(est_sim)

                seq_obj_tform4x4_est_obj_sim = torch.cat(seq_obj_tform4x4_est_obj_sim, dim=0)
                seq_obj_tform4x4_est_obj = torch.cat(seq_obj_tform4x4_est_obj, dim=0)

                seq_obj_tform4x4_est_obj = seq_obj_tform4x4_est_obj[
                                           :self.config.train.sequences_tform4x4_estimated_frames_count]
                seq_obj_tform4x4_est_obj_sim = seq_obj_tform4x4_est_obj_sim[
                                               :self.config.train.sequences_tform4x4_estimated_frames_count]

                seq_obj_tform6_est_obj = se3_log_map(seq_obj_tform4x4_est_obj)
                seq_obj_transl3_est_obj_dist_mat = (
                        seq_obj_tform6_est_obj[None, :, :3] - seq_obj_tform6_est_obj[:, None, :3]).norm(
                    dim=-1)
                seq_obj_rot3_est_obj_dist_mat = (
                        seq_obj_tform6_est_obj[None, :, 3:] - seq_obj_tform6_est_obj[:, None, 3:]).norm(
                    dim=-1)
                seq_obj_tform4x4_est_obj_transl_consist = seq_obj_transl3_est_obj_dist_mat.triu(
                    diagonal=1).sum() / torch.ones_like(seq_obj_transl3_est_obj_dist_mat).triu(diagonal=1).sum()
                seq_obj_tform4x4_est_obj_rot_consist = seq_obj_rot3_est_obj_dist_mat.triu(
                    diagonal=1).sum() / torch.ones_like(seq_obj_rot3_est_obj_dist_mat).triu(diagonal=1).sum()
                seq_obj_tform4x4_est_obj_sim = seq_obj_tform4x4_est_obj_sim.mean()
                seq_obj_tform4x4_est_obj = se3_exp_map(seq_obj_tform6_est_obj.mean(dim=0))

                self.seq_obj_tform4x4_est_obj[seq] = seq_obj_tform4x4_est_obj
                self.seq_obj_tform4x4_est_obj_sim[seq] = seq_obj_tform4x4_est_obj_sim
                self.seq_obj_tform4x4_est_obj_transl_consist[seq] = seq_obj_tform4x4_est_obj_transl_consist
                self.seq_obj_tform4x4_est_obj_rot_consist[seq] = seq_obj_tform4x4_est_obj_rot_consist

                if seq_obj_tform4x4_est_obj_sim > self.config.train.sequences_tform4x4_est_sim_min \
                        and seq_obj_tform4x4_est_obj_transl_consist < self.config.train.sequences_tform4x4_est_transl_consist_max \
                        and seq_obj_tform4x4_est_obj_rot_consist < self.config.train.sequences_tform4x4_est_rot_consist_max:

                    self.seq_filtered.append(seq)

                    if self.config.train.visualize.seq_added_tform:
                        for key, val in results_batch.items():
                            if type(val) == wandb.Image:
                                results_train['add_' + key] = val

            results_train['seq_obj_tform4x4_est_obj_sim'] = torch.stack(
                list(self.seq_obj_tform4x4_est_obj_sim.values())).mean()
            results_train['seq_obj_tform4x4_est_obj_transl_consist'] = torch.stack(
                list(self.seq_obj_tform4x4_est_obj_transl_consist.values())).mean()
            results_train['seq_obj_tform4x4_est_obj_rot_consist'] = torch.stack(
                list(self.seq_obj_tform4x4_est_obj_rot_consist.values())).mean()

            logger.info(
                f'seq_obj_tform4x4_est_obj_transl_consist {self.seq_obj_tform4x4_est_obj_transl_consist.values()}')
            # logger.info(f'estimating obj_tform4x4_obj_est_sims of {self.seq_obj_tform4x4_est_obj_sim}')

            sequences_filtered = list(self.seq_filtered)
            results_train["count_sequences"] = len(sequences_filtered + self.seq_labeled)
            dataset_sub = dataset.get_subset_by_sequences(sequences_filtered + self.seq_labeled)

            logger.info(f"Dataset contains {len(dataset_sub)} frames.")

            return dataset_sub

