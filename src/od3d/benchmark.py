import subprocess

from omegaconf import DictConfig, OmegaConf
import wandb
from od3d.datasets.dataset import OD3D_Dataset
from od3d.methods.method import OD3DMethod
from pathlib import Path
import datetime

print(dir(datetime))

def get_timestamp_as_string():
    now = datetime.datetime.now()
    timestamp = now.strftime("%m-%d_%H-%M-%S")
    return timestamp
def bench_single_method_local(config: DictConfig):

    # 1. setup logger
    run_name = get_timestamp_as_string()
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
    from pathlib import Path
    tmp_config_fpath = Path('/tmp/config.yaml').resolve()
    if not tmp_config_fpath.parent.exists():
        tmp_config_fpath.parent.mkdir(parents=True)
    with open(tmp_config_fpath, 'w') as fp:
        OmegaConf.save(config=cfg, f=fp.name)
    tmp_script_fpath = Path('/tmp/run.sh').resolve()
    if not tmp_script_fpath.parent.exists():
        tmp_script_fpath.parent.mkdir(parents=True)

    with open(tmp_script_fpath, 'w') as rsh:

        gpu_count = 0
        node_count = 1
        cpu_count = 1
        ram = "10gb"
        walltime = "24:00:00"

        timestamp = get_timestamp_as_string()
        gpu_cfg_str = f':gpus={gpu_count}' if gpu_count > 0 else ""
        cuda_cfg_str = f':nvidiaMinCC75' if gpu_count > 0 else ""

        script_as_string = f'''#!/bin/bash
#PBS -N {timestamp}_{cfg.test_dataset.name}_{cfg.method.name}
#PBS -S /bin/bash
#PBS -l nodes={node_count}:ppn={cpu_count}{gpu_cfg_str}{cuda_cfg_str},mem={ram},walltime={walltime}
#PBS -q default-cpu
#PBS -m a
#PBS -M sommerl@informatik.uni-freiburg.de
#PBS -j oe

# For interactive jobs: #PBS -I
# For array jobs: #PBS -t START-END[%SIMULTANEOUS]

PATH=${{PATH}}:{cfg.platform.path_cuda}/bin
LD_LIBRARY_PATH=${{LD_LIBRARY_PATH}}:{cfg.platform.path_cuda}/lib64
CUDA_HOME={cfg.platform.path_cuda}
export PATH
export LD_LIBRARY_PATH
export CUDA_HOME

echo PATH=${{PATH}}
echo LD_LIBRARY_PATH=${{LD_LIBRARY_PATH}}
echo CUDA_HOME=${{CUDA_HOME}}

# Setup Repository
if [[ -d "{cfg.platform.path_od3d}" ]]; then
    echo "OD3D is already cloned to {cfg.platform.path_od3d}."
else
    git clone {cfg.platform.url_od3d} {cfg.platform.path_od3d}
fi

cd {cfg.platform.path_od3d}
git pull {cfg.platform.url_od3d}

# Install OD3D in venv
if [[ -d "venv" ]]; then
    echo "Venv already exists at {cfg.platform.path_od3d}/venv."
    source venv/bin/activate
else
    echo "Creating venv at {cfg.platform.path_od3d}/venv."
    python3 -m venv venv
    source venv/bin/activate
    pip install pip --upgrade
    FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
    pip install -e .
fi

od3d debug hello-world

PYTHONUNBUFFERED=1 
CUDA_VISIBLE_DEVICES=1

exit 0
        '''
        rsh.write(script_as_string)
    subprocess.run(f'scp {tmp_script_fpath} torque:{tmp_script_fpath}', capture_output=True, shell=True)
    subprocess.run(f'ssh torque "qsub {tmp_script_fpath}"', capture_output=True, shell=True)

    #f'scp {tmp_script_fpath} torque:{Path(cfg.platform.path_exps).joinpath(tmp_script_fpath)}'
    # FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
    # raise NotImplementedError

def bench_single_method_slurm(cfg: DictConfig):
    # 1. save config
    # 2. setup od3d on slurm
    # 3. execute script with command: run od3d bench single -f `path-to-config`
    # TODO
    raise NotImplementedError