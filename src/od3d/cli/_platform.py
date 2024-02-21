
import logging
from pathlib import Path
logger = logging.getLogger(__name__)
import typer
import od3d.io
import re
from pygit2 import Repository

app = typer.Typer()
from od3d.benchmark.run import torque_run_method_or_cmd, slurm_run_method_or_cmd
from omegaconf import open_dict
from od3d.io import run_cmd
import subprocess

@app.command()
def slurm():
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform='slurm')

    logger.info(f'ssh slurm')
    logger.info(f'cd {config.platform.path_od3d} && source venv310/bin/activate')

@app.command()
def slurm_run(cmd: str = typer.Option('od3d debug hello-world', '-c', '--command'),):
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform='slurm')
    with open_dict(config):
        config.branch = Repository('.').head.shorthand  # 'master'
    slurm_run_method_or_cmd(config, cmd)

@app.command()
def rsync_configs(platform: str = typer.Option(None, '-p', '--platform')):
    logging.basicConfig(level=logging.INFO)
    config_target = od3d.io.load_hierarchical_config(platform=platform)
    config_source = od3d.io.load_hierarchical_config(platform='local')

    source_link = f'{config_source.platform.link}:' if config_source.platform.link != 'local' else ''
    target_link = f'{config_target.platform.link}:' if config_target.platform.link != 'local' else ''

    config_rfpaths_source = ['credentials/default.yaml',
                             f'platform/{platform}.yaml']
    config_rfpaths_target = ['credentials/default.yaml',
                             f'platform/local.yaml']
    for config_rfpath_source, config_rfpath_target in zip(config_rfpaths_source, config_rfpaths_target):
        path_source = Path(config_source.platform.path_od3d).joinpath('config', config_rfpath_source)
        path_target = Path(config_target.platform.path_od3d).joinpath('config', config_rfpath_target)
        od3d.io.run_cmd(cmd=f'rsync -avrzP {source_link}{path_source} {target_link}{path_target}', live=True, logger=logger)

@app.command()
def run(platform: str = typer.Option(None, '-p', '--platform'),
        cmd: str = typer.Option('od3d debug hello-world', '-c', '--command')):
    logging.basicConfig(level=logging.INFO)

    run_cmd(f'od3d platform rsync-configs -p {platform}', logger=logger)

    config = od3d.io.load_hierarchical_config(platform=platform)

    with open_dict(config):
        config.branch = Repository('.').head.shorthand  # 'master'

    if platform == 'slurm':
        slurm_run_method_or_cmd(config, cmd)
    elif platform == 'torque':
        torque_run_method_or_cmd(config, cmd)
    else:
        raise NotImplementedError

@app.command()
def status(platform: str = typer.Option(None, '-p', '--platform')):
    logging.basicConfig(level=logging.INFO)
    if platform == 'slurm':
        # 60j = 60 characters
        format = '"%.18i %.9P %.60j %.8u %.8T %.10M %.9l %.6D %R"'
        slurm_result = subprocess.run(f"ssh slurm 'squeue --me --format={format}'", capture_output=True, shell=True)
        slurm_jobs = slurm_result.stdout.decode("utf-8").split("\n")
        for slurm_job in slurm_jobs:
            logger.info(slurm_job)
    elif platform == 'torque':
        torque_result = subprocess.run(f"ssh torque 'qstat -a -u $(whoami)'", capture_output=True, shell=True)
        torque_jobs = torque_result.stdout.decode("utf-8").split("\n")
        for torque_job in torque_jobs:
            logger.info(torque_job)
    else:
        raise NotImplementedError


# sinfo
@app.command()
def torque():
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform='torque')

    logger.info(f'ssh torque')
    logger.info(f'cd {config.platform.path_od3d} && source venv310/bin/activate')

