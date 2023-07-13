import logging
logger = logging.getLogger(__name__)
from od3d.datasets.dataset import OD3D_Dataset
from omegaconf import DictConfig
from pathlib import Path
from od3d.datasets.pascal3d import Pascal3D
# from od3d.datasets.pascal3d.dataset import CATEGORIES, SUBSETS
# import wget
import shutil
import od3d.io

class Pascal3D_Occ(Pascal3D):

    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__(config=config)
        Pascal3D_Occ.setup(config=self.config)

    @staticmethod
    def setup(config: DictConfig):
        path_pascal3d_occ_raw = Path(config.path_pascal3d_occ_raw)


        if path_pascal3d_occ_raw.exists() and config.setup_remove_previous:
            logger.info(f"Removing previous Pascal3D_Occ")
            shutil.rmtree(path_pascal3d_occ_raw)

        if path_pascal3d_occ_raw.exists():
            logger.info(f"Found Pascal3D_Occ dataset at {path_pascal3d_occ_raw}")
        else:
            logger.info(f"Downloading Pascal3D_Occ dataset at {path_pascal3d_occ_raw}")
            path_pascal3d_occ_raw.mkdir(parents=True, exist_ok=True)
            fpath = path_pascal3d_occ_raw.joinpath("pascal3d_occ.sh")

            od3d.io.download(url=config.url_pascal3d_occ_script, fpath=fpath)
            od3d.io.run_cmd(cmd=f'chmod +x {fpath}', logger=logger, live=True)
            od3d.io.run_cmd(cmd=f'cd {path_pascal3d_occ_raw} && {fpath}', logger=logger, live=True)

    @staticmethod
    def get_frames_names_from_subsets_and_cateogories_from_raw(path_pascal3d_raw, subsets, categories):
        pass
    @staticmethod
    def preprocess(config: DictConfig):
        Pascal3D_Occ.preprocess_meta(config=config)
    @staticmethod
    def preprocess_meta(config: DictConfig):
        path = Path(config.path_pascal3d_raw)
        path_pascal3d_occ_raw = Path(config.path_pascal3d_occ_raw)
        path_meshes = path.joinpath("CAD")
        path_preprocess = Path(config.path_pascal3d_preprocess)
        categories = config.get("classes", CATEGORIES)
        subsets = config.get("subsets", SUBSETS)

        path_meta = path_preprocess.joinpath('meta')
        # remove_previous = False, override = False
        if not config.preprocess_meta_override and path_meta.exists():
            return

        if config.preprocess_meta_remove_previous:
            if path_meta.exists():
                shutil.rmtree(path_meta)

        frames_names, frames_rfpaths = Pascal3D_occ.get_frames_names_from_subsets_and_cateogories_from_raw(path_pascal3d_raw=path,
                                                                                                           subsets=subsets,
                                                                                                           categories=categories)

        if config.get('frames', None) is not None:
            frames_names = list(filter(lambda f: f in config.frames, frames_names))

        for i in tqdm(range(len(frames_names))):
            rfpath_annotation = Path("Annotations").joinpath(f"{frames_rfpaths[i]}.mat")
            rfpath_rgb = Path("Images").joinpath(f"{frames_rfpaths[i]}.JPEG")
            frame = Pascal3DFrame.load_from_raw(path_dataset=path, path_preprocess=path_preprocess, rfpath_rgb=rfpath_rgb,
                                                rfpath_annotation=rfpath_annotation, path_meshes=path_meshes)
            if frame.complete:
                conf = OmegaConf.structured(frame)
                fpath = path_meta.joinpath("frames", frame.name + '.yaml')
                if not fpath.parent.exists():
                    fpath.parent.mkdir(parents=True)
                OmegaConf.save(conf, fpath, resolve=True)
                _ = frame.mask