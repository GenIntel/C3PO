import logging
logger = logging.getLogger(__name__)
from od3d.datasets.dataset import OD3D_Dataset
from omegaconf import DictConfig
from pathlib import Path
from od3d.datasets.pascal3d import Pascal3D

from od3d.datasets.pascal3d.enum import PASCAL3D_CATEGORIES
from od3d.datasets.frame import OD3D_FRAME_MODALITIES
from typing import Dict, List
import shutil
import od3d.io

from od3d.datasets.pascal3d.frame import Pascal3DFrameMeta
from od3d.datasets.ood_cv.frame import OOD_CV_FrameMeta, OOD_CV_Frame

from tqdm import tqdm

from od3d.data.ext_enum import ExtEnum


class OOD_CV(OD3D_Dataset):

    def __init__(
            self,
            name: str,
            modalities: List[OD3D_FRAME_MODALITIES],
            path_raw: Path,
            path_preprocess: Path,
            path_pascal3d_raw: Path,
            categories: List[PASCAL3D_CATEGORIES] = None,
            dict_nested_frames: Dict[str, Dict[str, List[str]]] = None,
            transform=None,
            subset_fraction=1.,
            index_shift=0,
    ):
        categories = categories if categories is not None else PASCAL3D_CATEGORIES.list()
        super().__init__(categories=categories, name=name,
                         modalities=modalities, path_raw=path_raw,
                         path_preprocess=path_preprocess, transform=transform,
                         subset_fraction=subset_fraction, index_shift=index_shift,
                         dict_nested_frames=dict_nested_frames)

        self.path_pascal3d_raw = Path(path_pascal3d_raw)

    def get_subset_with_dict_nested_frames(self, dict_nested_frames):
        return OOD_CV(name=self.name, modalities=self.modalities, path_raw=self.path_raw,
                      path_preprocess=self.path_preprocess, categories=self.categories,
                      dict_nested_frames=dict_nested_frames, transform=self.transform,
                      index_shift=self.index_shift, path_pascal3d_raw=self.path_pascal3d_raw)

    def get_item(self, item):
        frame_meta = OOD_CV_FrameMeta.load_from_meta_with_name_unique(path_meta=self.path_meta,
                                                                      name_unique=self.list_frames_unique[item])
        return OOD_CV_Frame(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
                            path_meshes=self.path_meshes, meta=frame_meta, modalities=self.modalities,
                            categories=self.categories)
    @staticmethod
    def setup(config: DictConfig):
        path_raw = Path(config.path_raw)

        url = 'https://drive.google.com/file/d/1djm2ugmk98__9jgL8Sqb_QUIIpC7e1ir/view'

        if path_raw.exists() and config.setup_remove_previous:
            logger.info(f"Removing previous Pascal3D_Occ")
            shutil.rmtree(path_raw)

        if path_raw.exists():
            logger.info(f"Found OOD-CV dataset at {path_raw}")
        else:
            logger.info(f"Downloading OOD-CV dataset at {path_raw}")
            path_raw.mkdir(parents=True, exist_ok=True)

            fpath = path_raw.joinpath(f'pose.zip')

            #gdown.download(url, output=fpath, fuzzy=True)
            #os.system(f"unzip pose.zip")
            #os.system("rm pose.zip")

            od3d.io.download(url=url, fpath=fpath)
            od3d.io.unzip(fpath, dst=path_raw.joinpath('tmp'))
            od3d.io.move_dir(src=path_raw.joinpath('tmp', 'ood_cv'), dst=path_raw)

    #### PREPROCESS META
    @staticmethod
    def preprocess_meta(config: DictConfig):
        subsets = config.get("subsets", None)
        #if subsets is None:
        #    subsets = PASCAL3D_SUBSETS.list()
        subsets =[]
        categories = config.get("categories", None)
        if categories is None:
            categories = PASCAL3D_CATEGORIES.list()

        path_raw = OD3D_Dataset.get_path_raw(config=config)
        path_meta = OD3D_Dataset.get_path_meta(config=config)
        path_pascal3d_raw = Path(config.path_pascal3d_raw)
        rpath_meshes = Pascal3D.get_rpath_meshes()

        if config.preprocess_meta_remove_previous:
            if path_meta.exists():
                shutil.rmtree(path_meta)

    @property
    def path_meshes(self):
        return Pascal3D.get_path_meshes(path_raw=self.path_pascal3d_raw)