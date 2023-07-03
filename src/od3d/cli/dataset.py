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
    logger.info(f'There are {len(dataset.sequences_names)} sequences in the dataset.')


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
    path_datasets_remote = Path(config.platform.path_datasets).joinpath(directory).parent
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
        RandomCenterZoom3D(H=512, W=512, dist=20., center3d_min=[0., 0., 0.], center3d_max=[0., 0., 0.], apply_mask=True, apply_kpts2d_annot=False, apply_bbox_annot=False, apply_txtr=False, config=config.dataset),
        dataset.transform,
    ]
    )

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)
    logging.info(f"Dataset contains {len(dataset)} frames.")
    for batch in iter(dataloader):
        logger.info(f'{batch.sequence_name[0]}')
        batch.visualize()

        # batch[0].sequence_name
        # dataset.visualize(i)


import http.server
import socketserver
import torchvision
@app.command()
def serve(dataset: str = typer.Option('co3d', '-d', '--dataset'),
              platform: str = typer.Option('local', '-p', '--platform')):
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform=platform, overrides=["+datasets@dataset=" + dataset])
    dataset = OD3D_Dataset.subclasses[config.dataset.class_name](config.dataset)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)
    logging.info(f"Dataset contains {len(dataset)} frames.")
    iterator = iter(dataloader)


    # Define the request handler class
    class MyRequestHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            batch = next(iterator)
            if batch is not None:
                # Set the appropriate content type for image files
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg' if self.path.endswith(".jpg") else 'image/png')
                self.end_headers()

                # Open the image file and send its contents as the response body
                self.wfile.write(bytes(torchvision.io.encode_png(input=(batch.rgb[0].detach().cpu() * 255).to(torch.uint8)).byte()))
            else:
                # If the requested file is not an image, fall back to the default behavior
                super().do_GET()

            #self.send_response(200)  # Set the response status code
            #self.send_header('Content-type', 'text/html')  # Set the content type header
            #self.end_headers()
            #self.wfile.write(b"Hello, World!")  # Send the response body

    # Set up the server
    port = 8081  # Choose any available port number
    handler = MyRequestHandler
    httpd = socketserver.TCPServer(("", port), handler)

    # Start the server
    print(f"Server running on port {port}")
    httpd.serve_forever()