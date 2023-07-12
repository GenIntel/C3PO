from od3d.methods.method import OD3DMethod
from omegaconf import DictConfig
import torchvision
from od3d.methods.method import OD3D_Dataset
from typing import List
import torch
import math
import wandb
"""
class CNN(OD3DMethod):
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


def train(self, dataset: OD3D_Dataset, datasets_test: List[OD3D_Dataset]):
    dataset.transform = self.transform_train
    results_train = {}
    self.net.train()
    self.meshes.feats.requires_grad = True


    accumulate_steps = 0

    generator = torch.Generator().manual_seed(42)
    # self.meshes.show(pts3d=dataset.get_sequence_by_id(0).pcl[None,])
    # dataset_sub, _ = torch.utils.data.random_split(dataset, [dataset.config.subset_fraction, 1. - dataset.config.subset_fraction], generator=generator)

    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [1. - self.config.train.val_fraction, self.config.train.val_fraction], generator=generator)

    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                                   batch_size=self.config.train.dataloader.batch_size,
                                                   shuffle=True,
                                                   collate_fn=dataset.collate_fn,
                                                   num_workers=self.config.train.dataloader.num_workers,
                                                   pin_memory=self.config.train.dataloader.pin_memory)

    criterion = torch.nn.CrossEntropyLoss().cuda() # (reduction="none").cuda()
    # dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=self.config.train.dataloader.batch_size, shuffle=True,
    #                                               collate_fn=dataset.collate_fn, num_workers=self.config.train.dataloader.num_workers, pin_memory=self.config.train.dataloader.pin_memory)


    for e in range(self.config.train.epochs):
        if self.config.train.epochs_to_next_test > 0 and e % self.config.train.epochs_to_next_test == 0:
            for dataset_test in datasets_test:
                results_test = self.test(dataset_test)
                wandb.log({f'test_{dataset_test.config.name}_{k}': v for k, v in results_test.items()})



            if e % self.config.train.epochs_to_next_val == 0:
                results_val = self.test(dataset, dataset_sub=dataset_val)
                wandb.log({'val_' + k: v for k, v in results_val.items()})



            self.net.train()
            self.meshes.feats.requires_grad = True


            for i, batch in enumerate(iter(dataloader_train)):

                batch.to(device=self.device)

                pred = self.net()


    def test(self, dataset: OD3D_Dataset, config_inference: DictConfig=None, pose_iterative_refine=True, dataset_sub=None):
        if config_inference is None:
            config_inference = self.config.inference
        self.net.eval()
        self.meshes.feats.requires_grad = False
        clutter_feats = self.clutter_feats.detach()
        dataset.transform = self.transform_test

        if dataset_sub is None:
            generator = torch.Generator().manual_seed(42) # dataset.config.subset_fraction
            dataset_sub, _ = torch.utils.data.random_split(dataset, [dataset.config.subset_fraction, 1. - dataset.config.subset_fraction], generator=generator)
        dataloader = torch.utils.data.DataLoader(dataset=dataset_sub, batch_size=self.config.test.dataloader.batch_size, shuffle=False,
                                                 collate_fn=dataset.collate_fn, num_workers=self.config.test.dataloader.num_workers, pin_memory=self.config.test.dataloader.pin_memory)

        visual_names_unique = [dataset_sub[i].name_unique for i in range(config_inference.visualize.num_samples)]
        logger.info(f"Dataset contains {len(dataset_sub)} frames.")

        results = {
            'time_feats2d': [],
            'time_class': [],
            'time_pose_iterative': [],
            'time_pose': [],
            'rot_diff_rad': [],
            'label_gt': [],
            'label_pred': [],
            'sim': [],
        }

        for i, batch in tqdm(enumerate(iter(dataloader))):
            batch.to(device=self.device)

            _, _ , results_batch = self.inference_batch(batch=batch, config=config_inference, visual_names_unique=visual_names_unique)

            for key, val in results_batch.items():
                if key in results.keys() and isinstance(results[key], list):
                    results[key].append(results_batch[key])
                else:
                    results[key] = results_batch[key]

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

            #diffs_so3d_log = torch.cat(diffs_so3d_log, dim=0)
            #from od3d.cv.geometry.transform import rot3x3_broadcast, so3_exp_map
            #diffs_rot3x3_mean = so3_exp_map(diffs_so3d_log.mean(dim=0, keepdim=False))
            #diffs_rot3x3 = rot3x3_broadcast(diffs_rot3x3_mean.T, pytorch3d.transforms.so3_exp_map(diffs_so3d_log))
            #diffs_rot3 = pytorch3d.transforms.so3_log_map(diffs_rot3x3)
            #results['consist_rot_diff_rad'] = torch.norm(diffs_rot3, dim=-1)
            #results['consist_pose_err_median'] = 180 / math.pi * results['consist_rot_diff_rad'].median()
            # results['consist_pose_err_mean'] = 180 / math.pi * results['consist_rot_diff_rad'].mean()

            # cmatrix = confusion_matrix(results['label_gt'].detach().cpu().numpy(), results['label_pred'].detach().cpu().numpy())
            # logger.info(f'Confusion matrix:\n {cmatrix} ')

            results['sim'] = torch.cat(results['sim'], dim=0).mean()


            del results['label_gt']
            del results['label_pred']
            del results['rot_diff_rad']
            # del results['consist_rot_diff_rad']
            for key, val in results.items():
                logger.info(f'{key} : {val}')


        return results

"""