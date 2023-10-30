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
        for cat_id, category in enumerate(categories):
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
            all_pred_pose_dist_geo[category] = torch.zeros(
                size=(ref_instances_count_per_category[cat_id], src_instances_count_per_category[cat_id])).to(
                device=self.device, dtype=dtype)
            all_pred_pose_dist_appear[category] = torch.zeros(
                size=(ref_instances_count_per_category[cat_id], src_instances_count_per_category[cat_id])).to(
                device=self.device, dtype=dtype)

        for cat_id, category in enumerate(categories):
            src_instance_ids = torch.LongTensor(list(range(src_instances_count)))
            ref_instance_ids = torch.LongTensor(list(range(ref_instances_count)))

            src_mesh_ids = src_instance_ids[src_map_seq_to_cat == cat_id]
            ref_mesh_ids = ref_instance_ids[ref_map_seq_to_cat == cat_id]

            for r, ref_mesh_id in enumerate(ref_mesh_ids):
                ref_frames_N = len(ref_sequences[ref_mesh_id].frames_names)
                ref_frames_S = 10
                ref_frames_indices = np.linspace(0, ref_frames_N-1, ref_frames_S).astype(int)
                ref_frames = [self.transform_train(ref_sequences[ref_mesh_id].get_frame_by_index(ref_frame_index)) for ref_frame_index in ref_frames_indices]
                for s, src_mesh_id in enumerate(src_mesh_ids):
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

                    src_frames_N = len(src_sequences[src_mesh_id].frames_names)

                    source_frame_index = random.choice(np.arange(src_frames_N))
                    src_frame = self.transform_train(src_sequences[src_mesh_id].get_frame_by_index(source_frame_index))



                    bytes_io = io.BytesIO()
                    # data = {'img': batch.rgb, 'cam_tform4x4_obj': batch.cam_tform4x4_obj}
                    ref_image = torch.stack([src_frame.rgb], dim=0)
                    B = len(ref_image)
                    ref_scalings = src_frame.size # torch.stack([src_frame.size], dim=0)
                    ref_depth_map = torch.stack([src_frame.depth], dim=0)
                    ref_cam_intr = torch.stack([src_frame.cam_intr4x4], dim=0)
                    ref_cam_extr = torch.stack([src_frame.cam_tform4x4_obj], dim=0)

                    all_target_images = torch.stack([ref_frame.rgb for ref_frame in ref_frames], dim=0)[None,]
                    N_TGT = len(all_target_images)
                    target_scalings = ref_frames[0].size # torch.stack([ref_frame.size for ref_frame in ref_frames], dim=0)[None,]
                    target_depth_map = torch.stack([ref_frame.depth for ref_frame in ref_frames], dim=0)[None,]
                    target_cam_intr = torch.stack([ref_frame.cam_intr4x4 for ref_frame in ref_frames], dim=0)[None,]
                    target_cam_extr = torch.stack([ref_frame.cam_tform4x4_obj for ref_frame in ref_frames], dim=0)[None,]
                    data = {
                        "ref_image": ref_image * 1.,
                        "all_target_images": all_target_images * 1.,
                        "ref_scalings": ref_scalings,
                        "target_scalings": target_scalings,
                        "ref_depth_map": ref_depth_map,
                        "target_depth_map": target_depth_map,
                        "ref_cam_extr": ref_cam_extr,
                        "target_cam_extr": target_cam_extr,
                        "ref_cam_intr": ref_cam_intr,
                        "target_cam_intr": target_cam_intr,
                    }
                    pickle.dump(data, bytes_io, pickle.HIGHEST_PROTOCOL)
                    bytes_io.seek(0)
                    resp = requests.post("http://127.0.0.1:5000/predict", files={"file": bytes_io})

                    resp = requests.post("http://localhost:5000/predict", files={"file": bytes_io})
                    print(resp.json())
                    # resp: flask.Response =



                    ref_image = data['ref_image'],
                    all_target_images = data['all_target_images'],
                    ref_scalings = data['ref_scalings'],
                    target_scalings = data['target_scalings'],
                    ref_depth_map = data['ref_depth_map'],
                    target_depth_map = data['target_depth_map'],
                    ref_cam_extr = data['ref_cam_extr'],
                    target_cam_extr = data['target_cam_extr'],
                    ref_cam_intr = data['ref_cam_intr'],
                    target_cam_intr = data['target_cam_intr'],

                    # all_pred_ref_tform_src

        logger.info('loading mesh feats...')
        src_sequences_unique_names = [seq.name_unique for seq in src_sequences]
        ref_sequences_unique_names = [seq.name_unique for seq in ref_sequences]
        #sequences_unique_names = [seq.name_unique for seq in sequences]

        src_map_seq_to_cat = torch.LongTensor([categories.index(name.split('/')[0]) for name in src_sequences_unique_names])
        ref_map_seq_to_cat = torch.LongTensor([categories.index(name.split('/')[0]) for name in ref_sequences_unique_names])

        categories_count = len(categories)

        src_instances_count_per_category = [(src_map_seq_to_cat == c).sum().item() for c in range(categories_count)]
        ref_instances_count_per_category = [(ref_map_seq_to_cat == c).sum().item() for c in range(categories_count)]


    def test(self, dataset: OD3D_Dataset, config_inference: DictConfig = None):
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
