
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

    logger.info(f'ssh slurm')
    logger.info(f'cd {config.platform.path_od3d} && source venv310/bin/activate')

# sinfo
@app.command()
def torque():
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform='torque')

    logger.info(f'ssh torque')
    logger.info(f'cd {config.platform.path_od3d} && source venv310/bin/activate')

