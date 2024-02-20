
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
def torque_run(cmd: str = typer.Option('od3d debug hello-world', '-c', '--command'), ):
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform='torque')
    with open_dict(config):
        config.branch = Repository('.').head.shorthand  # 'master'
    torque_run_method_or_cmd(config, cmd)


# sinfo
@app.command()
def torque():
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform='torque')

    logger.info(f'ssh torque')
    logger.info(f'cd {config.platform.path_od3d} && source venv310/bin/activate')

