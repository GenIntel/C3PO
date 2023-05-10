import typer

app = typer.Typer()

from omegaconf import OmegaConf
from od3d.datasets.dataset import OD3DDataset

@app.command()
def classes():
    print(list(OD3DDataset.subclasses.keys()))

@app.command()
def setup(config_fpath: str = typer.Option(None, '-c', '--config')):
    print(config_fpath)
    config = OmegaConf.load(config_fpath)
    od3ddataset = OD3DDataset.subclasses[config.class_name](config)
    od3ddataset.setup()
