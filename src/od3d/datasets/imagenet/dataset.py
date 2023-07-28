import logging

import od3d.io

logger = logging.getLogger(__name__)
from od3d.datasets.dataset import OD3D_Dataset
from od3d.datasets.frame import OD3D_FRAME_MODALITIES, OD3D_Frame, OD3D_FrameMeta, OD3D_FrameMetaRGBMixin, \
    OD3D_FrameMetaBBoxMixin, OD3D_FrameMetaCategoryMixin, OD3D_FrameMetaSubsetMixin, OD3D_FrameMetaSizeMixin
#from od3d.datasets.objectnet3d.enum import OBJECTNET3D_CATEOGORIES
from pathlib import Path
from typing import List, Dict
from omegaconf import DictConfig
import shutil

class ImageNetFrameMeta(OD3D_FrameMetaSubsetMixin, OD3D_FrameMetaRGBMixin, OD3D_FrameMetaSizeMixin,
                        OD3D_FrameMetaCategoryMixin, OD3D_FrameMetaBBoxMixin, OD3D_FrameMeta):

    @property
    def name_unique(self):
        return f'{self.subset}/{self.category}/{self.name}'

    @staticmethod
    def load_from_raw():
        pass

class ImageNet(OD3D_Dataset):
    def __init__(self, name: str, modalities: List[OD3D_FRAME_MODALITIES], path_raw: Path, path_preprocess: Path,
                 categories: List=None,
                 dict_nested_frames: Dict=None,
                 transform=None, index_shift=0, subset_fraction=1.):

        categories = categories if categories is not None else  [] # TODO: OBJECTNET3D_CATEOGORIES.list()
        super().__init__(categories=categories, dict_nested_frames=dict_nested_frames, name=name, modalities=modalities, path_raw=path_raw,
                         path_preprocess=path_preprocess, transform=transform, index_shift=index_shift,
                         subset_fraction=subset_fraction)

    def get_item(self, item):
        frame_meta = ImageNetFrameMeta.load_from_meta_with_name_unique(path_meta=self.path_meta, name_unique=self.list_frames_unique[item])
        return OD3D_Frame(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
                          meta=frame_meta, modalities=self.modalities, categories=self.categories)

    @staticmethod
    def setup(config: DictConfig):
        # logger.info(OmegaConf.to_yaml(config))
        path_raw = Path(config.path_raw)
        if path_raw.exists() and config.setup_remove_previous:
            logger.info(f"Removing previous ImageNet")
            shutil.rmtree(path_raw)

        path_raw.mkdir(parents=True, exist_ok=True)
        od3d.io.run_cmd('pip install kaggle', logger=logger, live=True)
        od3d.io.write_config_to_json_file(config=config.credentials.kaggle, fpath=Path('~/.kaggle/kaggle.json'))
        od3d.io.run_cmd(f"cd {path_raw} && kaggle competitions download -c imagenet-object-localization-challenge",
                        logger=logger, live=True)
        logger.info('extracting files...')
        od3d.io.unzip(path_raw.joinpath("imagenet-object-localization-challenge.zip"), dst=path_raw)

    @staticmethod
    def preprocess_meta(config: DictConfig):
        pass

    def preprocess(self, override: False):
        pass