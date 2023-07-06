
import logging
from pathlib import Path
logger = logging.getLogger(__name__)
import typer
import od3d.io
import re

app = typer.Typer()


@app.command()
def slurm():
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform='slurm')

    logger.info(f'ssh slurm && source {config.platform.path_od3d}/venv310/bin/activate')

@app.command()
def torque():
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform='torque')

    logger.info(f'ssh torque && source {config.platform.path_od3d}/venv310/bin/activate')
