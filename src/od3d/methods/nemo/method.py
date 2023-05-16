
from od3d.methods.method import OD3DMethod
from od3d.datasets.dataset import OD3D_Dataset
from omegaconf import DictConfig

from torch.utils.data import RandomSampler
import logging
import torch
import numpy as np
import wandb
class NeMo(OD3DMethod):
    def __init__(
        self,
        config: DictConfig
    ):
        super().__init__(config=config)

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



    def test(self):
        pass


    """
    # single_mesh: true
    
        mesh_d = "single" if config.single_mesh else "multi"
    save_mesh_path = path_pascal3d_raw.joinpath(f"CAD_{mesh_d}")
    if os.path.isdir(save_mesh_path):
        print(f"Found {mesh_d} meshes at {save_mesh_path}")
    else:
        print(f"Generating {mesh_d} meshes at {save_mesh_path}")
        create_meshes(
            mesh_d,
            path_pascal3d_raw.joinpath("CAD"),
            path_pascal3d_raw.joinpath(f"CAD_{mesh_d}"),
            number_vertices=1000,
            linear_coverage=0.99,
        )
    
    """