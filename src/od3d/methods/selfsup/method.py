import logging
logger = logging.getLogger(__name__)

from omegaconf import DictConfig
from od3d.benchmark.results import OD3D_Results
from od3d.datasets.dataset import OD3D_Dataset
from od3d.methods.method import OD3D_Method
from od3d.models.model import OD3D_Model
from od3d.models.backbones.backbone import OD3D_Backbone
from od3d.models.heads.head import OD3D_Head
from od3d.cv.transforms.transform import OD3D_Transform
from od3d.cv.transforms.sequential import SequentialTransform
import torch
from od3d.datasets.co3d.dataset import CO3D

from typing import Dict
from pathlib import Path
from tqdm import tqdm
import torch.utils.data
import od3d.io


class SelfSup(OD3D_Method):

    def __init__(
            self,
            config: DictConfig,
            logging_dir,
    ):
        super().__init__(config=config, logging_dir=logging_dir)

        self.device = 'cuda:0'

        # init Network
        self.backbone = OD3D_Backbone.subclasses[self.config.model.backbone.class_name](config=self.config.model.backbone)
        self.head_selfsup = OD3D_Head.subclasses[self.config.model.head.selfsup.class_name](config=self.config.model.head.selfsup,
                                                                               in_dims=self.backbone.out_dims,
                                                                               in_upsample_scales=
                                                                               self.backbone.out_downsample_scales)



        self.backbone: OD3D_Backbone = OD3D_Backbone.subclasses[self.config.backbone.class_name](config=self.config.backbone)
        self.transform = self.backbone.transform

        self.to_device()
        self.optim_selfsup = od3d.io.get_obj_from_config(config=self.config.train.selfsup.optimizer, params=self.get_params_selfsup())
        self.scheduler_selfsup = od3d.io.get_obj_from_config(self.optim_selfsup, config=self.config.train.selfsup.scheduler)
        self.loss_selfsup = od3d.io.get_obj_from_config(config=self.config.train.selfsup.loss)


        self.loss_sup = od3d.io.get_obj_from_config(config=self.config.train.sup.loss)

        #self.out_dim = self.head.out_dim
        #self.downsample_rate = self.backbone.downsample_rate * self.head.downsample_rate

        self.transform_train_selfsup = SequentialTransform([
            OD3D_Transform.subclasses[config.train.selfsup.transform.class_name].create_from_config(config=config.train.selfsup.transform),
            self.backbone.transform,
        ])

        self.transform_train_sup = SequentialTransform([
            OD3D_Transform.subclasses[config.train.sup.transform.class_name].create_from_config(
                config=config.train.sup.transform),
            self.backbone.transform,
        ])

        self.transform_test = SequentialTransform([
            OD3D_Transform.subclasses[config.test.transform.class_name].create_from_config(config=config.test.transform),
            self.backbone.transform
        ])

    def init_sup(self):
        self.head_sup = OD3D_Head.subclasses[self.config.model.head.sup.class_name](config=self.config.model.head.sup,
                                                                               in_dims=self.backbone.out_dims,
                                                                               in_upsample_scales=
                                                                               self.backbone.out_downsample_scales)
        self.optim_sup = od3d.io.get_obj_from_config(config=self.config.train.sup.optimizer,
                                                         params=self.get_params_sup())
        self.scheduler_sup = od3d.io.get_obj_from_config(self.optim_sup,
                                                             config=self.config.train.sup.scheduler)

    def get_params_selfsup(self):
        return list(self.backbone.parameters()) + list(self.head_selfsup.parameters())

    def get_params_sup(self):
        return list(self.head_sup.parameters())

    def to_device(self, device = None):
        if device is None:
            device = self.device
        self.backbone.to(device)
        self.head_selfsup.to(device)
        self.head_sup.to(device)

    def switch_mode_test(self):
        self.backbone.eval()
        self.head_selfsup.eval()
        self.head_sup.eval()
    def switch_mode_train_selfsup(self):
        self.backbone.train()
        self.head_selfsup.train()
        self.head_sup.eval()

    def switch_mode_train_sup(self):
        self.backbone.eval()
        self.head_selfsup.eval()
        self.head_sup.train()

    def forward_selfsup(self, batch):
        return self.head_selfsup(self.backbone(batch.rgb))

    def forward_sup(self, batch):
        return self.head_sup(self.backbone(batch.rgb))

    @property
    def fpath_checkpoint(self):
        return self.logging_dir.joinpath(self.rfpath_checkpoint)

    @property
    def rfpath_checkpoint(self):
        return Path('selfsup.ckpt')

    def write_checkpoint(self, fpath_checkpoint: Path=None):
        if fpath_checkpoint is None:
            fpath_checkpoint = self.fpath_checkpoint
        torch.save({
            'backbone_state_dict': self.backbone.state_dict(),
            'head_selfsup': self.head_selfsup.state_dict(),
            'head_sup': self.head_sup.state_dict(),
        }, fpath_checkpoint)

    def read_checkpoint(self, fpath_checkpoint=None):
        if fpath_checkpoint is None:
            fpath_checkpoint = self.fpath_checkpoint
        checkpoint = torch.load(fpath_checkpoint)
        self.backbone.load_state_dict(checkpoint['backbone_state_dict'])
        self.head_selfsup.load_state_dict(checkpoint['head_selfsup'])
        self.head_sup.load_state_dict(checkpoint['head_sup'])

    def train_sup(self, dataset_train):
        self.init_sup()
        for epoch in range(self.config.train.sup.epochs):
            results_epoch = self.train_epoch_sup(dataset=dataset_train)
            results_epoch.log_with_prefix('train_sup')
            self.write_checkpoint()

    def train_epoch_sup(self, dataset: OD3D_Dataset) -> OD3D_Results:
        self.switch_mode_train_sup()
        dataset.transform = self.transform_train_sup
        dataloader_train = torch.utils.data.DataLoader(dataset=dataset,
                                                       batch_size=self.config.train.sup.dataloader.batch_size,
                                                       shuffle=True,
                                                       collate_fn=dataset.collate_fn,
                                                       num_workers=self.config.train.sup.dataloader.num_workers,
                                                       pin_memory=self.config.train.sup.dataloader.pin_memory)

        results_epoch = OD3D_Results(logging_dir=self.logging_dir)
        accumulate_steps = 0
        for i, batch in enumerate(iter(dataloader_train)):
            results_batch: OD3D_Results = self.train_batch_sup(batch=batch)
            results_batch.log_with_prefix('train_sup')
            accumulate_steps += 1
            if accumulate_steps % self.config.train.sup.batch_accumulate_to_next_step == 0:
                self.optim_sup.step()
                self.optim_sup.zero_grad()

            results_epoch += results_batch

        self.scheduler_sup.step()
        self.optim_sup.zero_grad()


        #results_visual = self.get_results_visual(results_epoch=results_epoch, dataset=dataset,
        #                                         config_visualize=self.config.train.sup.visualize)
        #results_epoch = results_epoch.mean()
        #results_epoch += results_visual
        return results_epoch

    def train_batch_sup(self, batch):
        results_batch = OD3D_Results(logging_dir=self.logging_dir)

        batch.to(device=self.device)
        batch_pred = self.forward_sup(batch)

        self.loss_sup(batch_pred, batch.labels)
        self.loss_sup.backward()
        logger.info(f'loss {self.loss_sup.item()}')

        return results_batch

    def train(self, datasets_train: Dict[str, OD3D_Dataset], datasets_val: Dict[str, OD3D_Dataset]):
        if 'main' in datasets_val.keys():
            dataset_train_sub = datasets_train['labeled']
        else:
            dataset_train_sub, dataset_val_sub = datasets_train['labeled'].get_split(fraction1=1. - self.config.train.selfsup.val_fraction,
                                                                                     fraction2=self.config.train.selfsup.val_fraction,
                                                                                     split=self.config.train.selfsup.split)
            datasets_val['main'] = dataset_val_sub

        for epoch in range(self.config.train.selfsup.epochs):
            if self.config.train.selfsup.val and self.config.train.selfsup.epochs_to_next_test > 0 and epoch % self.config.train.selfsup.epochs_to_next_test == 0:
                self.train_sup(dataset_train_sub)
                for dataset_val_key, dataset_val in datasets_val.items():
                    results_val = self.test(dataset_val, val=True)
                    results_val.log_with_prefix(prefix=f'val/{dataset_val.name}')

            results_epoch = self.train_epoch_selfsup(dataset=dataset_train_sub)
            results_epoch.log_with_prefix('train')
            self.write_checkpoint()

        self.read_checkpoint()

    def train_epoch_selfsup(self, dataset: OD3D_Dataset) -> OD3D_Results:
        self.switch_mode_train_selfsup()
        dataset.transform = self.transform_train_selfsup
        dataloader_train = torch.utils.data.DataLoader(dataset=dataset,
                                                       batch_size=self.config.train.selfsup.dataloader.batch_size,
                                                       shuffle=True,
                                                       collate_fn=dataset.collate_fn,
                                                       num_workers=self.config.train.selfsup.dataloader.num_workers,
                                                       pin_memory=self.config.train.selfsup.dataloader.pin_memory)

        results_epoch = OD3D_Results(logging_dir=self.logging_dir)
        accumulate_steps = 0
        for i, batch in enumerate(iter(dataloader_train)):
            results_batch: OD3D_Results = self.train_batch_selfsup(batch=batch)
            results_batch.log_with_prefix('train')
            accumulate_steps += 1
            if accumulate_steps % self.config.train.selfsup.batch_accumulate_to_next_step == 0:
                self.optim_selfsup.step()
                self.optim_selfsup.zero_grad()

            results_epoch += results_batch

        self.scheduler_selfsup.step()
        self.optim_selfsup.zero_grad()

        # results_epoch.log_dict_to_dir(name=f'train_frames/{dataset.name}')

        results_visual = self.get_results_visual(results_epoch=results_epoch, dataset=dataset,
                                                 config_visualize=self.config.train.selfsup.visualize)
        results_epoch = results_epoch.mean()
        results_epoch += results_visual
        return results_epoch

    def train_batch_selfsup(self, batch) -> OD3D_Results:
        results_batch = OD3D_Results(logging_dir=self.logging_dir)

        batch.to(device=self.device)
        batch_pred = self.forward_selfsup(batch)

        B = batch_pred.shape[0]
        batch_vts_ids = torch.arange(B, device=self.device)
        loss = self.loss_selfsup(batch_pred, batch_vts_ids)

        loss.backward()
        results_batch['loss'] = loss.item()

        return results_batch

    def test(self, dataset: OD3D_Dataset, val=False):
        # note: ensure that checkpoint is saved for checkpointed runs
        if not self.fpath_checkpoint.exists():
            self.write_checkpoint()

        logger.info(f'test dataset {dataset.name}')
        self.switch_mode_test()
        dataset.transform = self.transform_test
        if not isinstance(dataset, CO3D):
            dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.config.test.dataloader.batch_size,
                                                     shuffle=False,
                                                     collate_fn=dataset.collate_fn,
                                                     num_workers=self.config.test.dataloader.num_workers,
                                                     pin_memory=self.config.test.dataloader.pin_memory)
            logger.info(f"Dataset contains {len(dataset)} frames.")

        else:
            dict_category_sequences = {category: list(sequence_dict.keys()) for category, sequence_dict in dataset.dict_nested_frames.items()}
            dataset_sub = dataset.get_subset_by_sequences(dict_category_sequences=dict_category_sequences,
                                                          frames_count_max_per_sequence=self.config.multiview.batch_size)


            dataloader = torch.utils.data.DataLoader(dataset=dataset_sub, batch_size=self.config.multiview.batch_size,
                                                     shuffle=False,
                                                     collate_fn=dataset_sub.collate_fn,
                                                     num_workers=self.config.test.dataloader.num_workers,
                                                     pin_memory=self.config.test.dataloader.pin_memory)
            logger.info(f"Dataset contains {len(dataset_sub)} frames.")

        results_epoch = OD3D_Results(logging_dir=self.logging_dir)
        for i, batch in tqdm(enumerate(iter(dataloader))):
            batch.to(device=self.device)

            if not isinstance(dataset, CO3D):
                results_batch = self.inference_batch_single_view(batch=batch)
            else:
                results_batch = self.inference_batch_multiview(batch=batch, return_samples_with_sim=True)
            results_epoch += results_batch

            if not val and self.config.test.save_results:
                results_visual_batch = self.get_results_visual_batch(batch=batch, results_batch=results_batch,
                                                                         config_visualize=self.config.test.visualize)
                results_visual_batch.save_visual(prefix=f'test/{dataset.name}')

        count_pred_frames = len(results_epoch['item_id'])
        logger.info(f'Predicted {count_pred_frames} frames.')
        if not val and self.config.test.save_results:
            results_epoch.save_with_dataset(prefix='test', dataset=dataset)

        if not isinstance(dataset, CO3D):
            results_visual = self.get_results_visual(results_epoch=results_epoch, dataset=dataset,
                                                     config_visualize=self.config.test.visualize)
        else:
            results_visual = self.get_results_visual(results_epoch=results_epoch, dataset=dataset_sub,
                                                     config_visualize=self.config.test.visualize)


        results_epoch = results_epoch.mean()
        results_epoch += results_visual
        return results_epoch

    def setup(self):
        pass

    def get_results_visual_batch(self, batch, results_batch, config_visualize):
        results_batch_visual = OD3D_Results(logging_dir=self.logging_dir)
        return results_batch_visual

    def get_results_visual(self, results_epoch, dataset, config_visualize):
        results_visual = OD3D_Results(logging_dir=self.logging_dir)
        logger.info('create dataloader ...')
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=self.config.test.dataloader.batch_size,
                                                 shuffle=False,
                                                 collate_fn=dataset.collate_fn,
                                                 num_workers=self.config.test.dataloader.num_workers,
                                                 pin_memory=self.config.test.dataloader.pin_memory)

        for i, batch in tqdm(enumerate(iter(dataloader))):
            results_visual += self.get_results_visual_batch(batch, results_epoch, config_visualize=config_visualize)
        return results_visual