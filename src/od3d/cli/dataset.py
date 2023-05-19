import logging

import torch.utils.data
import typer
from hydra import compose, initialize
from od3d.datasets.dataset import OD3D_Dataset
from omegaconf import OmegaConf

app = typer.Typer()

@app.command()
def classes():
    print(list(OD3D_Dataset.subclasses.keys()))

@app.command()
def setup(config_fpath: str = typer.Option(None, '-c', '--config')):
    print(config_fpath)
    config = OmegaConf.load(config_fpath)
    od3ddataset = OD3D_Dataset.subclasses[config.class_name](config)
    od3ddataset.setup()

@app.command()
def visualize(dataset: str = typer.Option('pascal3d', '-d', '--dataset'),
              platform: str = typer.Option('roycoffee', '-p', '--platform')):
    config_dir_rel = "../../../config"
    initialize(version_base=None, config_path=config_dir_rel, job_name="test_app")
    config = compose(config_name="dummy", overrides=["+datasets@dataset=" + dataset, "+platform=" + platform])
    dataset = OD3D_Dataset.subclasses[config.dataset.class_name](config.dataset)

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=5, shuffle=False, collate_fn=dataset.collate_fn)
    logging.info(f"Dataset contains {len(dataset)} frames.")
    for batch in iter(dataloader):
        batch[0].visualize()
        # dataset.visualize(i)


@app.command()
def hello_world():
    logging.info("hello world")