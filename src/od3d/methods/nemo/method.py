
from od3d.methods.method import OD3DMethod
from omegaconf import DictConfig

class NeMo(OD3DMethod):
    def __init__(
        self,
        config: DictConfig
    ):
        super().__init__(config=config)

    def setup(self):
        pass

    def train(self):
        cfg = self.config
        dataset_kwargs = {"data_type": "train", "category": cfg.args.cate}
        train_dataset = construct_class_by_name(**cfg.dataset, **dataset_kwargs)
        if cfg.dataset.sampler is not None:
            train_dataset_sampler = construct_class_by_name(
                **cfg.dataset.sampler, dataset=train_dataset, rank=0, num_replicas=1,
                seed=cfg.training.random_seed)
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