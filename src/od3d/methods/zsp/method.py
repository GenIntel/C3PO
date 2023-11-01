import logging
logger = logging.getLogger(__name__)
from od3d.methods.method import OD3D_Method
from od3d.datasets.dataset import OD3D_Dataset
from od3d.benchmark.results import OD3D_Results
from omegaconf import DictConfig
import pytorch3d.transforms
import pandas as pd
import numpy as np
from torch.utils.data import RandomSampler
import torch
from od3d.cv.transforms.transform import OD3D_Transform
from od3d.cv.transforms.sequential import SequentialTransform
from typing import Dict
from tqdm import tqdm
from od3d.datasets.frames import OD3D_Frames
from od3d.datasets.co3d import CO3D
import torch
import random
import requests
import pickle
import io
from od3d.cv.geometry.transform import inv_tform4x4, tform4x4
from od3d.datasets.co3d.enum import CAM_TFORM_OBJ_SOURCES
from od3d.cv.metric.pose import get_pose_diff_in_rad

class ZSP(OD3D_Method):
    def setup(self):
        pass

    def __init__(
            self,
            config: DictConfig,
            logging_dir,
    ):
        super().__init__(config=config, logging_dir=logging_dir)

        self.device = 'cpu' #'cuda:0'

        # init Network
        # self.net = OD3D_Model(config.docker)

        self.transform_train = OD3D_Transform.subclasses[config.train.transform.class_name].create_from_config(config=config.train.transform)
        self.transform_test = OD3D_Transform.subclasses[config.test.transform.class_name].create_from_config(config=config.test.transform)
        self.target_data = None

    def train(self, datasets_train: Dict[str, OD3D_Dataset], datasets_val: Dict[str, OD3D_Dataset]):

        dataset_src: CO3D = datasets_train['src']
        dataset_ref: CO3D = datasets_train['labeled']

        categories = dataset_src.categories

        src_sequences = dataset_src.get_sequences()
        ref_sequences = dataset_ref.get_sequences()

        src_sequences_unique_names = [seq.name_unique for seq in src_sequences]
        ref_sequences_unique_names = [seq.name_unique for seq in ref_sequences]

        src_map_seq_to_cat = torch.LongTensor([categories.index(name.split('/')[0]) for name in src_sequences_unique_names])
        ref_map_seq_to_cat = torch.LongTensor([categories.index(name.split('/')[0]) for name in ref_sequences_unique_names])

        categories_count = len(categories)

        src_instances_count_per_category = [(src_map_seq_to_cat == c).sum().item() for c in range(categories_count)]
        ref_instances_count_per_category = [(ref_map_seq_to_cat == c).sum().item() for c in range(categories_count)]

        src_instances_count = len(src_sequences)
        ref_instances_count = len(ref_sequences)
        dtype= torch.float

        results_diff_log_rot = {}
        all_pred_ref_tform_src = {}
        all_pred_pose_dist_geo = {}
        all_pred_pose_dist_appear = {}
        if self.config.use_train_only_to_collect_target_data:
            self.target_data = {}
        for cat_id, category in enumerate(categories):
            if self.config.use_train_only_to_collect_target_data:
                self.target_data[category] = []
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

            results_diff_log_rot[category] = torch.zeros(
                size=(ref_instances_count_per_category[cat_id], src_instances_count_per_category[cat_id])).to(
                device=self.device, dtype=dtype)
            all_pred_ref_tform_src[category] = torch.zeros(
                size=(ref_instances_count_per_category[cat_id], src_instances_count_per_category[cat_id], 4, 4)).to(
                device=self.device, dtype=dtype)

        for cat_id, category in enumerate(categories):
            src_instance_ids = torch.LongTensor(list(range(src_instances_count)))
            ref_instance_ids = torch.LongTensor(list(range(ref_instances_count)))

            src_mesh_ids = src_instance_ids[src_map_seq_to_cat == cat_id]
            ref_mesh_ids = ref_instance_ids[ref_map_seq_to_cat == cat_id]

            for r, ref_mesh_id in enumerate(ref_mesh_ids):
                from od3d.datasets.co3d.enum import PCL_SOURCES
                logger.info(f'ref {r+1} out of {len(ref_mesh_ids)}')
                ref_frames_N = len(ref_sequences[ref_mesh_id].frames_names)
                ref_frames_S = 10
                ref_frames_indices = np.linspace(0, ref_frames_N-1, ref_frames_S).astype(int)
                ref_frames = [self.transform_train(ref_sequences[ref_mesh_id].get_frame_by_index(ref_frame_index)) for ref_frame_index in ref_frames_indices]

                #ref_sequences[ref_mesh_id].show(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.DROID_SLAM, pcl_source=PCL_SOURCES.DROID_SLAM_CLEAN, show_imgs=True)


                """
                       Args:
                           ref_image (torch.Tensor): Bx3xSxS
                           all_target_images (torch.Tensor): N_TGTx3xSxS
                           ref_scalings (torch.Tensor): Bx2 (H,W) rescaling from original resolution
                           target_scalings (torch.Tensor): N_TGTx2 (H,W) rescaling from original resolution
                           ref_depth_map (torch.Tensor): Bx1xHxW
                           target_depth_map (torch.Tensor): N_TGTx1xHxW
                           ref_cam_extr (torch.Tensor): Bx4x4
                           target_cam_extr (torch.Tensor): N_TGTx4x4
                           ref_cam_intr (torch.Tensor): Bx4x4
                           target_cam_intr (torch.Tensor): N_TGTx4x4
                       Returns:
                           target_tform_ref (torch.Tensor): Bx4x4
                   """

                srcs_frames_N = [len(src_sequences[src_mesh_id].frames_names) for src_mesh_id in src_mesh_ids]
                srcs_frames_ids = [src_frames_N //2  for src_frames_N in srcs_frames_N]



                #src_frames_N = len(src_sequences[src_mesh_id].frames_names)
                # source_frame_index = random.choice(np.arange(src_frames_N))
                #source_frame_index = src_frames_N // 2
                src_frames = [self.transform_train(src_sequences[src_mesh_id].get_frame_by_index(srcs_frames_ids[s]))  for s, src_mesh_id in enumerate(src_mesh_ids)]
                #src_frame = self.transform_train(src_sequences[src_mesh_id].get_frame_by_index(source_frame_index))

                if dataset_ref.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM_ZSP_LABELED or \
                        dataset_ref.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM or \
                        dataset_ref.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM_LABELED:
                    ref_sequence_scale = ref_sequences[ref_mesh_id].get_a_src_scale_b_src(CAM_TFORM_OBJ_SOURCES.DROID_SLAM, CAM_TFORM_OBJ_SOURCES.CO3D)
                    ref_cam_source = CAM_TFORM_OBJ_SOURCES.DROID_SLAM

                else:
                    ref_cam_source = CAM_TFORM_OBJ_SOURCES.CO3D
                    ref_sequence_scale = 1.
                    logger.warning('No gt available ')

                if dataset_src.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM_ZSP_LABELED or \
                        dataset_src.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM or \
                        dataset_src.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM_LABELED:
                    src_sequences_scales = torch.Tensor([src_sequences[src_mesh_id].get_a_src_scale_b_src(CAM_TFORM_OBJ_SOURCES.DROID_SLAM, CAM_TFORM_OBJ_SOURCES.CO3D) for s, src_mesh_id in enumerate(src_mesh_ids)])
                    src_cam_source = CAM_TFORM_OBJ_SOURCES.DROID_SLAM
                else:
                    src_sequences_scales = torch.Tensor([1. for s, src_mesh_id in enumerate(src_mesh_ids)])
                    src_cam_source = CAM_TFORM_OBJ_SOURCES.CO3D
                    logger.warning('No gt available ')
                from od3d.cv.geometry.transform import depth2pts3d_grid, transf3d_broadcast, inv_tform4x4, proj3d2d_broadcast
                from od3d.cv.visual.show import show_scene

                # data = {'img': batch.rgb, 'cam_tform4x4_obj': batch.cam_tform4x4_obj}
                ref_image = torch.stack([src_frame.rgb for src_frame in src_frames], dim=0)
                B = len(ref_image)
                ref_scalings = src_frames[0].size.clone()  # torch.stack([src_frame.size], dim=0)
                ref_scalings[:] = 1. # * src_frame.depth_mask
                ref_depth_map = torch.stack([src_frame.depth * src_sequences_scales[s] for s, src_frame in enumerate(src_frames)], dim=0)
                ref_cam_intr = torch.stack([src_frame.cam_intr4x4 for src_frame in src_frames], dim=0)
                ref_cam_extr = torch.stack([src_frame.get_cam_tform4x4_obj(src_cam_source) for src_frame in src_frames], dim=0)

                all_target_images = torch.stack([ref_frame.rgb for ref_frame in ref_frames], dim=0)[None,].repeat(B, 1, 1, 1, 1)
                N_TGT = len(ref_frames)
                target_scalings = ref_frames[
                    0].size.clone()  # torch.stack([ref_frame.size for ref_frame in ref_frames], dim=0)[None,]
                target_scalings[:] = 1. # * ref_frame.depth_mask
                target_depth_map = torch.stack([ref_frame.depth * ref_sequence_scale for r, ref_frame in enumerate(ref_frames)], dim=0)[None,].repeat(B, 1, 1, 1, 1)
                target_cam_intr = torch.stack([ref_frame.cam_intr4x4 for ref_frame in ref_frames], dim=0)[None,].repeat(B, 1, 1, 1)
                target_cam_extr = torch.stack([ref_frame.get_cam_tform4x4_obj(ref_cam_source) for ref_frame in ref_frames], dim=0)[None,].repeat(B, 1, 1, 1)

                # if ref_cam_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM:
                #
                #     pts3d = ref_sequences[ref_mesh_id].get_pcl(PCL_SOURCES.DROID_SLAM_CLEAN)
                #     H, W = src_frames[0].size.to(int)
                #
                #     pxl2d = proj3d2d_broadcast(pts3d=pts3d[:, None, None], proj4x4=tform4x4(target_cam_intr, target_cam_extr)[None,]).permute(1, 2, 0, 3)
                #     pxl2d_z = transf3d_broadcast(pts3d[:, None, None], target_cam_extr[None,]).permute(1, 2, 0, 3)[:, :, :, 2]# [None, None].expand(*pxl2d.shape[:-1])
                #     depth_map = torch.zeros(size=target_depth_map.shape)
                #     from od3d.cv.select import index_MD_to_1D
                #     pxl2d_out_of_bounds = (pxl2d[:, :, :, 0] < 0) + (pxl2d[:, :, :, 0] > (W-1)) + (pxl2d[:, :, :, 1] < 0) + (pxl2d[:, :, :, 1] > (H-1))
                #     pxl2d[pxl2d_out_of_bounds] = 0
                #     pxl1d = index_MD_to_1D(indexMD=pxl2d.to(int), inputMD=depth_map, dims=[-1, -2])
                #     # target_depth_map = torch.scatter_reduce(depth_map.reshape(B, N_TGT, -1), src=pxl2d_z, index=pxl1d, reduce='amin', dim=-1, include_self=False).reshape(B, N_TGT, 1, H, W)
                #     target_depth_map = torch.scatter_reduce(depth_map.permute(0, 1, 2, 4, 3).reshape(B, N_TGT, -1), src=pxl2d_z, index=pxl1d, reduce='amin', dim=-1, include_self=False).reshape(B, N_TGT, 1, W, H).permute(0, 1, 2, 4, 3)
                #     target_depth_map[:, :, :, 0, 0] = 0.
                #
                #     srcs_pts3d = [src_sequences[src_mesh_id].get_pcl(PCL_SOURCES.DROID_SLAM_CLEAN) for src_mesh_id in src_mesh_ids]
                #     pts3d_counts_max = max([len(p) for p in srcs_pts3d])
                #     pts3d = torch.zeros(B, pts3d_counts_max, 3)
                #     srcs_pts3d_first = torch.stack([p[0] for p in srcs_pts3d], dim=0)
                #     pts3d = srcs_pts3d_first[:, None,].expand(*pts3d.shape)
                #     pxl2d = proj3d2d_broadcast(pts3d=pts3d, proj4x4=tform4x4(ref_cam_intr, ref_cam_extr)[:, None,])
                #     pxl2d_z = transf3d_broadcast(pts3d, ref_cam_extr[:, None,])[:, :, 2]# [None, None].expand(*pxl2d.shape[:-1])
                #     depth_map = torch.zeros(size=ref_depth_map.shape)
                #     from od3d.cv.select import index_MD_to_1D
                #     pxl2d_out_of_bounds = (pxl2d[:, :, 0] < 0) + (pxl2d[:, :, 0] > (W-1)) + (pxl2d[:, :, 1] < 0) + (pxl2d[:, :, 1] > (H-1))
                #     pxl2d[pxl2d_out_of_bounds] = 0
                #     pxl1d = index_MD_to_1D(indexMD=pxl2d.to(int), inputMD=depth_map, dims=[-1, -2])
                #     # target_depth_map = torch.scatter_reduce(depth_map.reshape(B, N_TGT, -1), src=pxl2d_z, index=pxl1d, reduce='amin', dim=-1, include_self=False).reshape(B, N_TGT, 1, H, W)
                #     ref_depth_map = torch.scatter_reduce(depth_map.permute(0, 1, 3, 2).reshape(B, -1), src=pxl2d_z, index=pxl1d, reduce='amin', dim=-1, include_self=False).reshape(B, 1, W, H).permute(0, 1, 3, 2)
                #     ref_depth_map[:, :, 0, 0] = 0.

                if self.config.use_train_only_to_collect_target_data:
                    self.target_data[category].append({
                        "all_target_images": all_target_images[0],
                        "target_scalings": target_scalings,
                        "target_depth_map": target_depth_map[0],
                        "target_cam_extr": target_cam_extr[0],
                        "target_cam_intr": target_cam_intr[0],
                    })
                    continue


                # pts3d = [depth2pts3d_grid(target_depth_map[0, n], cam_intr4x4=target_cam_intr[0, n]) for n in range(N_TGT)]
                # pts3d = torch.cat(pts3d, dim=0)
                # pts3d = transf3d_broadcast(pts3d=pts3d.permute(0, 2, 3, 1).reshape(N_TGT, -1, 3), transf4x4=inv_tform4x4(target_cam_extr[0, :, None]))
                # show_scene(pts3d=pts3d)

                #from od3d.cv.visual.show import show_imgs
                #show_imgs(rgbs=torch.cat([all_target_images[0], (target_depth_map[0].repeat(1, 3, 1, 1) * 100).clamp(0, 255)]  , dim=0))
                #show_imgs(rgbs=ref_image)
                # !!! dist scale of droid slam, does not fit to depth scale of
                # show_imgs(rgbs=ref_depth_map)
                #show_imgs(rgbs=target_depth_map[0])
                data = {
                    "ref_image": ref_image,
                    "all_target_images": all_target_images,
                    "ref_scalings": ref_scalings,
                    "target_scalings": target_scalings,
                    "ref_depth_map": ref_depth_map,
                    "target_depth_map": target_depth_map,
                    "ref_cam_extr": ref_cam_extr,
                    "target_cam_extr": target_cam_extr,
                    "ref_cam_intr": ref_cam_intr,
                    "target_cam_intr": target_cam_intr,
                }
                bytes_io = io.BytesIO()
                pickle.dump(data, bytes_io, pickle.HIGHEST_PROTOCOL)
                bytes_io.seek(0)

                try:
                    resp = requests.post("http://127.0.0.1:5000/predict", files={"file": bytes_io})
                    # all_pred_ref_tform_src = resp.json()['obj2_tform_obj1']
                    pred_ref_tform_srcs = resp.json()['obj2_tform_obj1']
                    pred_ref_tform_srcs = torch.Tensor(pred_ref_tform_srcs).to(device=self.device)
                except:
                    pred_ref_tform_srcs = torch.eye(4)[None,].repeat(B, 4, 4).to(device=self.device)

                for s, src_mesh_id in enumerate(src_mesh_ids):
                    pred_ref_tform_src = pred_ref_tform_srcs[s]
                    all_pred_ref_tform_src[category][r, s] = pred_ref_tform_src

                    if self.config.use_gt_src:
                        if dataset_src.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM_ZSP_LABELED:
                            gt_ref_tform_src = tform4x4(
                                inv_tform4x4(ref_sequences[ref_mesh_id].co3dv1_zsp_obj_tform_droid_slam_obj.to(device=self.device,
                                                                                                               dtype=pred_ref_tform_src.dtype)),
                                src_sequences[src_mesh_id].co3dv1_zsp_obj_tform_droid_slam_obj.to(device=self.device,
                                                                                                  dtype=pred_ref_tform_src.dtype))
                        elif dataset_src.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM_LABELED:
                            gt_ref_tform_src = tform4x4(
                                inv_tform4x4(ref_sequences[ref_mesh_id].droid_slam_labeled_tform_droid_slam.to(
                                    device=self.device, dtype=pred_ref_tform_src.dtype)),
                                src_sequences[src_mesh_id].droid_slam_labeled_tform_droid_slam.to(device=self.device,
                                                                                                  dtype=pred_ref_tform_src.dtype))
                        elif dataset_src.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.CO3D:
                            gt_ref_tform_src = tform4x4(
                                inv_tform4x4(
                                    ref_sequences[ref_mesh_id].co3dv1_zsp_obj_tform_co3dv1_obj.to(device=self.device,
                                                                                                  dtype=pred_ref_tform_src.dtype)),
                                src_sequences[src_mesh_id].co3dv1_zsp_obj_tform_co3dv1_obj.to(device=self.device,
                                                                                              dtype=pred_ref_tform_src.dtype))
                        else:
                            gt_ref_tform_src = torch.eye(4).to(device=self.device)
                            logger.warning('No gt available ')
                        diff_rot_angle_rad = get_pose_diff_in_rad(pred_tform4x4=pred_ref_tform_src, gt_tform4x4=gt_ref_tform_src)
                        logger.info(diff_rot_angle_rad)
                        results_diff_log_rot[category][r, s] = diff_rot_angle_rad

        results = OD3D_Results()
        for cat_id, category in enumerate(categories):
            category_results = OD3D_Results()
            exclude_diagonal = dataset_src.name == dataset_ref.name

            if exclude_diagonal:
                # excluding diagonal entries as these are predicted transformation between same instance
                if self.config.use_gt_src:
                    category_results[f'rot_diff_rad'] = results_diff_log_rot[category][
                        torch.eye(src_instances_count_per_category[cat_id]).to(device=self.device) == 0].reshape(ref_instances_count_per_category[cat_id], src_instances_count_per_category[cat_id]-1)
            else:
                if self.config.use_gt_src:
                    category_results[f'rot_diff_rad'] = results_diff_log_rot[category]

            results += category_results.mean()
            category_results_mean = category_results.add_prefix(category)
            category_results_mean = category_results_mean.mean()
            category_results_mean.log()
            logger.info(category_results_mean)

        results_mean = results.mean()
        results_mean.log()
        logger.info(results_mean)


    def test(self, dataset: OD3D_Dataset, config_inference: DictConfig = None):
        if self.target_data is not None:
            logger.info(f'test dataset {dataset.name}')

            score_metric_name = 'pose/acc_pi18'  # 'pose/acc_pi18' 'pose/acc_pi6'
            score_ckpt_val = 0.
            score_latest = 0.

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

            #results_visual = self.get_results_visual(results_epoch=results_epoch, dataset=dataset,
            #                                         config_visualize=self.config.test.visualize)
            results_epoch = results_epoch.mean()
            #results_epoch += results_visual
            return results_epoch
        else:
            return OD3D_Results()

    def inference_batch(self, batch: OD3D_Frames):
        results = OD3D_Results()

        import io
        import pickle
        import torch

        import requests



        #resp = requests.post("http://localhost:5000/predict",
        #                             files={"file": open('<PATH/TO/.jpg/FILE>/cat.jpg', 'rb')})

        bytes_io = io.BytesIO()
        pickle.dump({'img': batch.rgb, 'cam_tform4x4_obj': batch.cam_tform4x4_obj}, bytes_io, pickle.HIGHEST_PROTOCOL)

        resp = requests.post("http://localhost:5000/predict",files={"file": bytes_io})

        """
        s1 = "foo"
        s1 = {"foo": torch.zeros(3,), "blub": torch.ones(4,)}
        bytes_io = io.BytesIO()
        pickle.dump(s1, bytes_io, pickle.HIGHEST_PROTOCOL)
        bytes_io.seek(0)
        s2 = pickle.load(bytes_io)
        """



        cam_tform4x4_obj = None

        from od3d.cv.geometry.transform import rot3x3

        diff_rot3x3 = rot3x3(batch.cam_tform4x4_obj[:, :3, :3].permute(0, 2, 1), cam_tform4x4_obj[:, :3, :3])

        try:
            diff_so3_log = pytorch3d.transforms.so3_log_map(diff_rot3x3.permute(0, 2, 1))
            diff_rot_angle_rad = torch.norm(diff_so3_log, dim=-1)
        except ValueError:
            logger.warning(
                f'Cannot calculate deviation in rotation angle due to rot3x3 trace being too small, setting deviation to 0.')
            diff_rot_angle_rad = torch.zeros_like(diff_rot3x3[:, 0, 0])
        results['rot_diff_rad'] = diff_rot_angle_rad
        #results['label_gt'] = batch.label
        #results['label_pred'] = pred_class_ids
        results['label_names'] = self.config.classes
        #results['sim'] = sim
        results['cam_tform4x4_obj'] = cam_tform4x4_obj
        results['item_id'] = batch.item_id
        results['name_unique'] = batch.name_unique

        B = len(batch)

        return results
