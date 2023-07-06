import logging
import shutil

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
        # RandomCenterZoom3D(H=512, W=512, dist=20., center3d_min=[0., 0., 0.], center3d_max=[0., 0., 0.], apply_mask=True, apply_kpts2d_annot=False, apply_bbox_annot=False, apply_txtr=False, config=config.dataset),
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
from od3d.io import run_cmd

@app.command()
def label_export(dataset: str = typer.Option('co3d', '-d', '--dataset'),
                 platform: str = typer.Option('local', '-p', '--platform'),
                 name: str = typer.Option('co3d_only_first_car', '-n', '--name'),
                 restart: bool = typer.Option(True, '-r', '--restart')):
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform=platform, overrides=["+datasets@dataset=" + dataset])
    dataset = OD3D_Dataset.subclasses[config.dataset.class_name](config.dataset)
    #logging.info(f"Dataset contains {len(dataset)} frames.")

    #path_labelstudio_in = dataset.path_preprocess.joinpath('labelstudio', 'in')
    #path_labelstudio_out = dataset.path_preprocess.joinpath('labelstudio', 'out')

    path_labelstudio_labels = dataset.path_preprocess.joinpath('labels', 'kpts2d_orient')

    if restart and path_labelstudio_labels.exists():
        shutil.rmtree(path_labelstudio_labels)

    if not path_labelstudio_labels.exists():
        path_labelstudio_labels.mkdir(parents=True, exist_ok=True)

    import json

    tmp_fpath = 'tmp.json'
    cmd = f"curl -X GET http://localhost:8080/api/projects/ -H 'Authorization: Token 458fc5a491443a009e210be0261dd6b2da8585be' > {tmp_fpath}"
    run_cmd(cmd, live=True, logger=logger)

    with open(tmp_fpath, mode='r') as f:
        projects = json.load(f)
    logger.info(projects)

    count = projects['count']
    results = projects['results']
    map_id_to_name = {}
    map_name_to_id = {}
    for i in range(count):
        map_id_to_name[results[i]['id']] = results[i]['title']
        map_name_to_id[results[i]['title']] = results[i]['id']

    cmd = f"curl -X GET http://localhost:8080/api/projects/{map_name_to_id[name]}/export?exportType=JSON -H 'Authorization: Token 458fc5a491443a009e210be0261dd6b2da8585be' --output {tmp_fpath}"
    # cmd = f'label-studio export {map_name_to_id[name]} json --data-dir out --export-path {tmp_fpath}'
    run_cmd(cmd, live=True, logger=logger)

    with open(tmp_fpath, mode='r') as f:
        images = json.load(f)

    kpoints_pairs = {
        'left': 'right',
        'right': 'left',
        'back': 'front',
        'front': 'back',
        'top': 'bottom',
        'bottom': 'top',
    }
    for frame in images:
        # logger.info(label['data']['image'])
        unique_name = frame['data']['image'].split('-')[-1].split('.')[0]
        logger.info(unique_name)
        fpath_label = path_labelstudio_labels.joinpath(unique_name + '.pt')
        l_orients_kpoints_pairs = {'left-right': [], 'back-front' : [], 'top-bottom': []}
        for annots in frame['annotations']:
            count_kpts = 0
            for annot in annots["result"]:
                count_kpts += 1
                if count_kpts % 2 == 1:
                    kpoint1_name = annot["value"]["keypointlabels"][0]
                    kpoint_pair = []
                    kpoint_pair.append(torch.Tensor([annot["value"]["x"]  / 100. * annot["original_width"],  annot["value"]["y"] / 100. * annot["original_height"]]))
                elif count_kpts % 2 == 0:
                    kpoint2_name = annot["value"]["keypointlabels"][0]
                    if kpoint2_name != kpoints_pairs[kpoint1_name]:
                        logger.warning(f"Expected kpoint {kpoints_pairs[kpoint1_name]}, got {kpoint2_name}. Skipping kpoint...")
                        continue
                    kpoint_pair.append(torch.Tensor([annot["value"]["x"] / 100. * annot["original_width"], annot["value"]["y"] / 100. * annot["original_height"]]))
                    if kpoint1_name in ['left', 'back', 'top']:
                        kpoint_pair_name = f'{kpoint1_name}-{kpoint2_name}'
                    else:
                        kpoint_pair_name = f'{kpoint2_name}-{kpoint1_name}'
                        kpoint_pair = [kpoint_pair[-i] for i in range(len(kpoint_pair))]
                    kpoint_pair = torch.stack(kpoint_pair, dim=0)
                    l_orients_kpoints_pairs[kpoint_pair_name].append(kpoint_pair)
        for orient in l_orients_kpoints_pairs.keys():
            if len(l_orients_kpoints_pairs[orient]) > 0:
                l_orients_kpoints_pairs[orient] = torch.stack(l_orients_kpoints_pairs[orient], dim=0)
            else:
                l_orients_kpoints_pairs[orient] = torch.Tensor(l_orients_kpoints_pairs[orient])
        logger.info(l_orients_kpoints_pairs)
        torch.save(l_orients_kpoints_pairs, fpath_label)
        #with open(fpath_label, mode='w') as f:
        #    json.dump(label_clean, fp=f)

@app.command()
def label_start(dataset: str = typer.Option('co3d_only_first', '-d', '--dataset'),
                platform: str = typer.Option('local', '-p', '--platform'),
                category: str = typer.Option('car', '-c', '--category'),
                restart: bool = typer.Option(False, '-r', '--restart')):
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform=platform, overrides=["+datasets@dataset=" + dataset])
    config.dataset.classes = [category]
    dataset = OD3D_Dataset.subclasses[config.dataset.class_name](config.dataset)
    logging.info(f"Dataset contains {len(dataset)} frames.")

    path_labelstudio_in = dataset.path_preprocess.joinpath('labelstudio', 'in')
    path_labelstudio_out = dataset.path_preprocess.joinpath('labelstudio', 'out')
    if restart:
        if path_labelstudio_in.exists():
            shutil.rmtree(path_labelstudio_in)
        if path_labelstudio_out.exists():
            shutil.rmtree(path_labelstudio_out)

    path_labelstudio_labelconfig = dataset.path_preprocess.joinpath('labelstudio', f'label_{category}_config.xml')
    keypoints = dataset.config.keypoints[category]

    labelconfig_str="""
<View>
<Header value="Select label and click the image to start"/>
  <Image name="image" value="$image" zoom="true"/>
  <KeyPointLabels name="keypoints" toName="image">
    <Label value="left" background="red"/>
    <Label value="right" background="yellow"/>
    <Label value="back" background="green"/>
    <Label value="front" background="orange"/>
    <Label value="top" background="blue"/>
    <Label value="bottom" background="purple"/>
  </KeyPointLabels>
</View>

    """


#     labelconfig_str = """
#         <View>
#         <KeyPointLabels name="kp-1" toName="img-1">
#     """
#     for keypoint in keypoints:
#         labelconfig_str += f'<Label value="{keypoint}"/>' #  background="red"
#     labelconfig_str += """
#   </KeyPointLabels>
#   <Image name="img-1" value="$img" />
# </View>
#     """


    with open(file=path_labelstudio_labelconfig, mode='w') as f:
        f.write(labelconfig_str)

    if not path_labelstudio_in.exists():
        path_labelstudio_in.mkdir(parents=True, exist_ok=True)

    if not path_labelstudio_out.exists():
        path_labelstudio_out.mkdir(parents=True, exist_ok=True)


    for i in range(len(dataset)):
        frame = dataset.__getitem__(i)
        fname = f'{frame.name_unique}{frame.path_rgb.suffix}'
        if not path_labelstudio_in.joinpath(fname).exists():
            cmd = f'cp "{frame.path_rgb}" "{path_labelstudio_in.joinpath(fname)}"'
            run_cmd(cmd, live=True, logger=logger)
    cmd = f'label-studio init {dataset.config.name}_{category} --username abc@def.com --password abcdefghj --data-dir {path_labelstudio_out} --label-config {path_labelstudio_labelconfig}'
    run_cmd(cmd, live=True, logger=logger)

    cmd = f'label-studio start --data-dir {path_labelstudio_out}'
    run_cmd(cmd, live=True, logger=logger)


    """
    # "curl -X GET http://localhost:8080/api/projects/ -H 'Authorization: Token 964c47bc573f19aca5067f7bc8901b97524cfedb'"
        cmd = f"curl -X GET http://localhost:8080/api/import/file-upload/{i+1} {frame.path_rgb} -H 'Authorization: Token 964c47bc573f19aca5067f7bc8901b97524cfedb'"
        cmd = f"curl -X POST http://localhost:8080/api/import/file-upload/{i+1} -F {frame.path_rgb} -H 'Authorization: Token 964c47bc573f19aca5067f7bc8901b97524cfedb'"
        cmd = f"curl -X POST http://localhost:8080/api/projects/1/import/{i+1} -F {frame.path_rgb} -H 'Authorization: Token 964c47bc573f19aca5067f7bc8901b97524cfedb'"


    # Define the request handler class
    class MyRequestHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            # Set the appropriate content type for image files
            self.send_response(200)
            frame_id = int(self.path.split('.')[-2].split('/')[-1])
            frame = dataset.__getitem__(frame_id)
            message = bytes(torchvision.io.encode_png(input=(frame.rgb.detach().cpu() * 255).to(torch.uint8)).byte())
            self.send_header('Content-type', 'image/png')
            self.send_header('Content-length', str(len(message)))
            self.end_headers()

            # Open the image file and send its contents as the response body
            #self.wfile.write(bytes(torchvision.io.encode_png(input=(batch.rgb[0].detach().cpu() * 255).to(torch.uint8)).byte()))
            self.wfile.write(message)

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
    """