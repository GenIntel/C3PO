import logging
logger = logging.getLogger(__name__)
import torch.utils.data
import typer
import od3d.io
from od3d.datasets.dataset import OD3D_Dataset, OD3D_FRAME_MODALITIES
from omegaconf import OmegaConf
from pathlib import Path

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
def sequences(dataset: str = typer.Option('co3d', '-d', '--dataset'),
              platform: str = typer.Option('local', '-p', '--platform')):
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform=platform, overrides=["+datasets@dataset=" + dataset])
    dataset = OD3D_Dataset.subclasses[config.dataset.class_name](config.dataset)
    sequences_names_as_str = '\n'.join(dataset.sequences_names)
    logger.info(f"Dataset sequences names: \n {sequences_names_as_str}")


@app.command()
def setup(dataset: str = typer.Option('pascal3d', '-d', '--dataset'),
          platform: str = typer.Option('local', '-p', '--platform'),
          override: bool = typer.Option(False, '-o', '--override'),
          remove_previous: bool = typer.Option(False, '-r', '--remove-previous')):
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform=platform, overrides=["+datasets@dataset=" + dataset])
    config.dataset.setup_remove_previous = remove_previous
    config.dataset.setup_override = override
    OD3D_Dataset.subclasses[config.dataset.class_name].setup(config.dataset)

@app.command()
def preprocess(dataset: str = typer.Option('pascal3d', '-d', '--dataset'),
          platform: str = typer.Option('local', '-p', '--platform'),
          override: bool = typer.Option(False, '-o', '--override'),
          remove_previous: bool = typer.Option(False, '-r', '--remove-previous')):
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform=platform, overrides=["+datasets@dataset=" + dataset])
    config.dataset.preprocess_meta_remove_previous = remove_previous
    config.dataset.preprocess_meta_override = override
    OD3D_Dataset.subclasses[config.dataset.class_name].preprocess(config.dataset)

@app.command()
def rsync(directory: str = typer.Option('CO3D_Preprocess', '-d', '--directory'),
          platform: str = typer.Option('local', '-p', '--platform'),):
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform=platform)

    path_datasets_local = Path(config.platform_local.path_datasets).joinpath(directory)
    path_datasets_remote = Path(config.platform.path_datasets).joinpath(directory)
    od3d.io.run_cmd(cmd=f'rsync -avrzP {path_datasets_local} {config.platform.link}:{path_datasets_remote}', live=True, logger=logger)

@app.command()
def visualize(dataset: str = typer.Option('pascal3d', '-d', '--dataset'),
              platform: str = typer.Option('local', '-p', '--platform')):
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform=platform, overrides=["+datasets@dataset=" + dataset])
    dataset = OD3D_Dataset.subclasses[config.dataset.class_name](config.dataset)
    import torchvision
    from od3d.cv.transforms import CenterZoom3D, RandomCenterZoom3D
    # modalities = [OD3D_FRAME_MODALITIES(mod) for mod in config.dataset.modalities]
    dataset.transform = torchvision.transforms.Compose([
        RandomCenterZoom3D(H=512, W=512, dist=50., center3d_min=[-1., -1., -1.], center3d_max=[1., 1., 1.], apply_mask=True, apply_kpts2d_annot=False, apply_bbox_annot=False, apply_txtr=False, config=config.dataset),
        dataset.transform,
    ]
    )

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)
    logging.info(f"Dataset contains {len(dataset)} frames.")
    for batch in iter(dataloader):
        batch.visualize(cuboids=dataset.cuboids)
        # batch[0].sequence_name
        # dataset.visualize(i)
