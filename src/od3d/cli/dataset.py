import logging

import torch.utils.data
import typer
import od3d.io
from od3d.datasets.dataset import OD3D_Dataset, OD3D_FRAME_MODALITIES
from omegaconf import OmegaConf
from functools import partial

app = typer.Typer()

@app.command()
def classes():
    print(list(OD3D_Dataset.subclasses.keys()))

@app.command()
def setup(config_fpath: str = typer.Option(None, '-c', '--config')):
    logging.basicConfig(level=logging.DEBUG)
    print(config_fpath)
    config = OmegaConf.load(config_fpath)
    od3ddataset = OD3D_Dataset.subclasses[config.class_name](config)
    od3ddataset.setup()

@app.command()
def visualize(dataset: str = typer.Option('co3d', '-d', '--dataset'),
              platform: str = typer.Option('local', '-p', '--platform')):
    logging.basicConfig(level=logging.DEBUG)
    config = od3d.io.load_hierarchical_config(platform=platform, overrides=["+datasets@dataset=" + dataset])
    dataset = OD3D_Dataset.subclasses[config.dataset.class_name](config.dataset)

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=partial(dataset.collate_fn, modalities=[OD3D_FRAME_MODALITIES.RGB, OD3D_FRAME_MODALITIES.MASK]))
    logging.info(f"Dataset contains {len(dataset)} frames.")
    for batch in iter(dataloader):
        batch.visualize()
        # dataset.visualize(i)
