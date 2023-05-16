from omegaconf import DictConfig
import wandb
from od3d.datasets.dataset import OD3D_Dataset
from od3d.methods.method import OD3DMethod
from pathlib import Path
import datetime

print(dir(datetime))

def bench_single_method_local(config: DictConfig):

    # 1. setup logger
    now = datetime.datetime.now()
    run_name = now.strftime("%m-%d_%H-%M-%S")
    logging_dir = Path(config.logger.local_dir).joinpath(run_name)
    logging_dir.mkdir(parents=True)
    if config.logger.use_wandb:
        wandb.init(project=config.logger.wandb_project_name, config=config, dir=Path(config.logger.local_dir), name=run_name)

    # 2. setup datasets
    dataset_test = OD3D_Dataset.subclasses[config.test_dataset.class_name](config.test_dataset)
    dataset_train = OD3D_Dataset.subclasses[config.train_dataset.class_name](config.train_dataset)

    # 3. setup method
    method = OD3DMethod.subclasses[config.method.class_name](config.method)

    # 4. train method
    method.train(dataset_train)

    # 5. bench method (logs results inside class)
    method.test(dataset_test)

def bench_single_method_local_separate_venv(cfg: DictConfig):
    # 1. save config
    # 2. setup od3d in separate virtual environment
    # 3. from virtual environment: run od3d bench single -f `path-to-config`
    # TODO
    raise NotImplementedError

def bench_single_method_local_docker(cfg: DictConfig):
    # 1. save config
    # 2. setup od3d in docker image
    # 3. from inside docker: run od3d bench single -f `path-to-config`
    # TODO
    raise NotImplementedError
def bench_single_method_torque(cfg: DictConfig):
    # 1. save config
    # 2. setup od3d on torque
    # 3. execute script with command: run od3d bench single -f `path-to-config`
    # TODO
    raise NotImplementedError

def bench_single_method_slurm(cfg: DictConfig):
    # 1. save config
    # 2. setup od3d on slurm
    # 3. execute script with command: run od3d bench single -f `path-to-config`
    # TODO
    raise NotImplementedError