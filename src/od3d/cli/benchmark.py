import typer
import od3d.io
from omegaconf import OmegaConf
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
from od3d.benchmark import bench_single_method_local, bench_single_method_local_separate_venv, bench_single_method_local_docker, bench_single_method_torque, bench_single_method_slurm
app = typer.Typer()
import subprocess
from omegaconf import open_dict

from datetime import datetime
def get_timestamp_as_string():
    now = datetime.now()
    timestamp = now.strftime("%m-%d_%H-%M-%S")
    return timestamp

@app.command()
def multiple(benchmark: str = typer.Option('co3d_nemo', '-b', '--benchmark'),
        ablation: str = typer.Option(None, '-a', '--ablation'),
        platform: str = typer.Option('local', '-p', '--platform')):
    logging.basicConfig(level=logging.INFO)

    file_fpath = Path(__file__).parent.resolve()
    config_dir_rel = "../../../config"
    config_dir_abs = file_fpath.joinpath(config_dir_rel)
    ablations_root_dir = config_dir_abs.joinpath("ablations")

    if ablation is None:
        cfgs = [od3d.io.load_hierarchical_config(benchmark=benchmark, platform=platform)]
    else:
        cfgs = []
        # create one config per ablation
        ablation_dir = ablations_root_dir.joinpath(ablation)
        for ablation_file_fpath in ablation_dir.iterdir():
            ablation_fpath_rel = ablation_file_fpath.relative_to(ablations_root_dir).with_suffix('')
            if not ablation_fpath_rel.name.startswith("_"):
                cfgs.append(od3d.io.load_hierarchical_config(benchmark=benchmark, platform=platform, ablation=str(ablation_fpath_rel)))

    # create one config per method
    methods_cfgs = []
    for cfg in cfgs:
        methods_keys = cfg.method.keys()
        for key in methods_keys:
            method_cfg = cfg.copy()
            method_cfg.method = cfg.method[key]
            method_cfg_exists = False
            for prev_method_cfg in methods_cfgs:
                if method_cfg == prev_method_cfg:
                    method_cfg_exists = True
            if not method_cfg_exists:
                methods_cfgs.append(method_cfg)

    print(f"{len(methods_cfgs)} configs with single method.")



    for method_cfg in methods_cfgs:
        with open_dict(method_cfg):
            ablation_name = method_cfg.get("ablation_name", None)
            if ablation_name is not None:
                method_cfg.run_name = f'{get_timestamp_as_string()}_{method_cfg.test_dataset.class_name}_{method_cfg.method.class_name}_{ablation_name}_{method_cfg.platform.link}'
            else:
                method_cfg.run_name = f'{get_timestamp_as_string()}_{method_cfg.test_dataset.class_name}_{method_cfg.method.class_name}_{method_cfg.platform.link}'

        if method_cfg.platform.link == 'local':
            bench_single_method_local(method_cfg)
        elif method_cfg.platform.link == 'local-separate-venv':
            bench_single_method_local_separate_venv(method_cfg)
        elif method_cfg.platform.link == 'local-docker':
            bench_single_method_local_docker(method_cfg)
        elif method_cfg.platform.link == 'torque':
            bench_single_method_torque(method_cfg)
        elif method_cfg.platform.link == 'slurm':
            bench_single_method_slurm(method_cfg)



@app.command()
def single_local(config_fpath: str = typer.Option(None, '-c', '--config')):
    logging.basicConfig(level=logging.INFO)
    method_cfg = OmegaConf.load(config_fpath)
    bench_single_method_local(method_cfg)


@app.command()
def test(benchmark: str = typer.Option('timeseries_internal', '-b', '--benchmark'),
         mode: str = typer.Option('local', '-m', '--mode'),
         tasks: str = typer.Option(None, '-t', '--tasks'),
         constraint: str = typer.Option('4h8c', '-c', '--constraint'),
         frameworks: str = typer.Option(None, '-f', '--frameworks'),
         localcode: str = typer.Option(None, '-l', '--localcode')):
    logging.basicConfig(level=logging.DEBUG)
    logger.info("test")
@app.command()
def info_slurm():
    'scontrol show job'

    'srun -p lmb_gpu-rtx2080 -w dagobert --pty bash'
    pass


@app.command()
def status_slurm():
    logging.basicConfig(level=logging.INFO)

    slurm_result = subprocess.run(f'ssh slurm "squeue --me"', capture_output=True, shell=True)
    slurm_jobs = slurm_result.stdout.decode("utf-8").split("\n")
    for slurm_job in slurm_jobs:
        logger.info(slurm_job)
@app.command()
def status_torque():
    logging.basicConfig(level=logging.INFO)

    torque_result = subprocess.run(f'ssh torque "qstat -a -u $(whoami)"', capture_output=True, shell=True)
    torque_jobs = torque_result.stdout.decode("utf-8").split("\n")
    for torque_job in torque_jobs:
        logger.info(torque_job)
@app.command()
def stop_torque(job: str = typer.Option(None, '-j', '--job')):
    logging.basicConfig(level=logging.INFO)

    torque_result = subprocess.run(f'ssh torque "qdel {job}"', capture_output=True, shell=True)
    for line in torque_result.stdout.decode("utf-8").split("\n"):
        logger.info(line)

@app.command()
def stop_slurm(job: str = typer.Option(None, '-j', '--job')):
    logging.basicConfig(level=logging.INFO)

    slurm_result = subprocess.run(f'ssh slurm "scancel {job}"', capture_output=True, shell=True)
    for line in slurm_result.stdout.decode("utf-8").split("\n"):
        logger.info(line)