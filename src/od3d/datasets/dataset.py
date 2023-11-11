import logging
logger = logging.getLogger(__name__)
from torch.utils.data import Dataset
from omegaconf import OmegaConf, DictConfig
from enum import Enum
from pathlib import Path
import torch
from typing import List, Dict
from od3d.datasets.frame import OD3D_FRAME_MODALITIES, OD3D_Frame, OD3D_FrameMeta
from od3d.datasets.frames import OD3D_Frames
from od3d.data import ExtEnum
import inspect
from tqdm import tqdm
import numpy as np
import od3d.io

class OD3D_SEQ_MODALITIES(str, Enum):
    PCL = 'pcl'

class OD3D_PREPROCESS_MODALITIES(str, Enum):
    FRONT_FRAME = 'front_frame'
    CUBOID = 'cuboid'
    CUBOID_AVG = 'cuboid_avg'
    PCL = 'pcl'
    MASK = 'mask'

class OD3D_DATASET_SPLITS(str, ExtEnum):
    SEQUENCES_SEPARATED = 'sequences_separated'
    RANDOM = 'random'
    SEQUENCES_SHARED = 'sequences_shared'


class OD3D_Dataset(Dataset):
    subclasses = {}

    @classmethod
    def create_from_config(cls, config: DictConfig, transform=None):
        if config.get("setup", False).get("enabled", False):
            cls.setup(config=config)
        if config.get("extract_meta", False).get("enabled", False):
            cls.extract_meta(config=config)

        keys = inspect.getfullargspec(cls.__init__)[0][1:]
        od3d_dataset = cls(**dict((key, config.get(key)) for key in keys if config.get(key, None) is not None), transform=transform)

        if config.get("preprocess", False):
            od3d_dataset.preprocess(config_preprocess=config.preprocess)

        return od3d_dataset

    @classmethod
    def create_by_name(cls, name: str, config: dict = None):
        config_loaded = od3d.io.read_config_intern(rfpath=Path("datasets").joinpath(f"{name}.yaml"))
        if config is not None:
            config_loaded = OmegaConf.merge(config_loaded, config)
        return cls.create_from_config(config_loaded)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls
    def __init__(self, name: str, modalities: List[OD3D_FRAME_MODALITIES], path_raw: Path, path_preprocess: Path,
                 categories: List[str]=None, transform=None, index_shift=0, subset_fraction=1.,
                 dict_nested_frames: Dict=None, dict_nested_frames_ban: Dict=None):

        logger.info(f'init dataset {name}...')

        self.name = name
        self.path_raw: Path = Path(path_raw)
        self.path_preprocess: Path = Path(path_preprocess)
        self.subset_fraction: float = subset_fraction

        if transform is None:
            from od3d.cv.transforms.rgb_uint8_to_float import RGB_UInt8ToFloat
            transform = RGB_UInt8ToFloat

        self.transform = transform
        self.index_shift = index_shift
        self.modalities = modalities
        self.splits_featured = [OD3D_DATASET_SPLITS.RANDOM]
        self.categories = categories if categories is not None else []

        logger.info('completing nested frames..., can take up to 500 seconds...')
        dict_nested_frames = OD3D_FrameMeta.complete_nested_metas(path_meta=self.path_meta,
                                                                  dict_nested_metas=dict_nested_frames, dict_nested_metas_ban=dict_nested_frames_ban)


        dict_nested_frames = self.filter_dict_nested_frames(dict_nested_frames)

        logger.info('unrolling nested frames...')
        list_frames_unique = OD3D_FrameMeta.unroll_nested_metas(dict_nested_meta=dict_nested_frames)

        logger.info('filtering frames...')
        list_frames_unique = self.filter_list_frames_unique(list_frames_unique)

        self.frames_count = len(list_frames_unique)
        if self.subset_fraction is not None and self.subset_fraction != 1.:
            logger.info('filtering with subset fraction...')
            frames_ids_subset = self.get_subset_item_ids(subset_fraction=subset_fraction)
            list_frames_unique = [list_frames_unique[id] for id in frames_ids_subset]

        self.set_list_frames_unique(list_frames_unique=list_frames_unique)
        logger.info(f"found {self.frames_count} frames.")

    def filter_dict_nested_frames(self, dict_nested_frames):
        return dict_nested_frames

    def filter_list_frames_unique(self, list_frames_unique):
        return list_frames_unique

    def __len__(self):
        return self.frames_count

    def get_subset_with_dict_nested_frames(self, dict_nested_frames: Dict):
        raise NotImplementedError

    def set_list_frames_unique(self, list_frames_unique):
        self.list_frames_unique = list_frames_unique
        self.dict_nested_frames = OD3D_FrameMeta.rollup_flattened_frames(
            list_meta_names_unique=self.list_frames_unique)
        self.frames_count = len(self.list_frames_unique)

    def get_subset_with_item_ids(self, item_ids):
        list_frames_unique = [self.list_frames_unique[id] for id in item_ids]
        dict_nested_frames = OD3D_FrameMeta.rollup_flattened_frames(list_meta_names_unique=list_frames_unique)
        return self.get_subset_with_dict_nested_frames(dict_nested_frames)

    def get_subset_with_names_unique(self, list_frames_unique: List[str]):
        dict_nested_frames = OD3D_FrameMeta.rollup_flattened_frames(list_meta_names_unique=list_frames_unique)
        return self.get_subset_with_dict_nested_frames(dict_nested_frames)

    def get_subset_item_ids(self, subset_fraction):
        return torch.multinomial(torch.ones(size=(len(self),)),
                                 num_samples=int(subset_fraction * len(self)),
                                 replacement=False)

    def get_split_item_ids(self, fraction1: float):
        item_ids_subsetA = self.get_subset_item_ids(subset_fraction=fraction1)
        item_ids = torch.arange(len(self))
        item_ids_maskA = torch.zeros(size=(len(self),), dtype=torch.bool, device=item_ids_subsetA.device)
        item_ids_maskA[item_ids_subsetA] = True
        item_ids_subsetA = item_ids[item_ids_maskA]
        item_ids_subsetB = item_ids[~item_ids_maskA]

        return item_ids_subsetA, item_ids_subsetB
    def get_fractionA_from_fractionA_and_fraction_B(self, fraction1: float, fraction2: float=None):
        if fraction2 is None:
            assert fraction1 > 0. and fraction1 < 1.
        else:
            fraction12 = fraction1 + fraction2
            if fraction12 != 1.:
                fraction1 = fraction1 / fraction12
                fraction2 = fraction2 / fraction12
                logger.warning(f'Subset fractions dont sum up to 1. Setting 1.={fraction1}, 2.={fraction2}')
        return fraction1

    def get_split(self, fraction1: float, fraction2: float, split: OD3D_DATASET_SPLITS = OD3D_DATASET_SPLITS.RANDOM):
        fraction1 = self.get_fractionA_from_fractionA_and_fraction_B(fraction1, fraction2)
        if split == OD3D_DATASET_SPLITS.SEQUENCES_SHARED:
            return self.get_split_sequences_shared(fraction1=fraction1)
        elif split == OD3D_DATASET_SPLITS.RANDOM:
            return self.get_split_random(fraction1=fraction1)
        elif split == OD3D_DATASET_SPLITS.SEQUENCES_SEPARATED:
            return self.get_split_sequences_separated(fraction1=fraction1)
        else:
            logger.warning(f'Unknown split {split}.')


    def get_split_random(self, fraction1: float):
        item_ids_subsetA, item_ids_subsetB = self.get_split_item_ids(fraction1=fraction1)
        co3d_subsetA = self.get_subset_with_item_ids(item_ids=item_ids_subsetA)
        co3d_subsetB = self.get_subset_with_item_ids(item_ids=item_ids_subsetB)
        return co3d_subsetA, co3d_subsetB

    def get_split_sequences_shared(self, fraction1: float):
        raise NotImplementedError

    def get_split_sequences_separated(self, fraction1: float):
        raise NotImplementedError

    def __getitem__(self, item):
        item_id_shift = (item + self.index_shift) % len(self)
        frame = self.transform(self.get_item(item_id_shift))
        frame.item_id = item_id_shift
        return frame

    def get_item(self, item):
        raise NotImplementedError

    def get_random_item(self):
        return self.__getitem__(np.random.choice(self.__len__()))

    def get_item_id_by_name_unique(self, name_unique: str):
        for i in range(len(self)):
            if self.list_frames_unique[i] == name_unique:
                return i
        return -1

    def collate_fn(self, frames: List[OD3D_Frame], device='cpu', dtype=torch.float32):
        frames = OD3D_Frames.get_frames_from_list(frames, modalities=self.modalities, dtype=dtype, device=device)
        return frames

    def get_dataloader(self, batch_size=1, shuffle=False):
        dataloader = torch.utils.data.DataLoader(dataset=self, batch_size=1, shuffle=False, collate_fn=self.collate_fn)
        return dataloader

    @staticmethod
    def setup(config: DictConfig):
        raise NotImplementedError

    @staticmethod
    def extract_meta(config: DictConfig):
        raise NotImplementedError
    def preprocess(self, config_preprocess: DictConfig):
        raise NotImplementedError

    def visualize(self, item: int):
        raise NotImplementedError


    @property
    def path_meta(self):
        return self.path_preprocess.joinpath('meta')

    @staticmethod
    def get_path_meta(config):
        return OD3D_Dataset.get_path_meta_from_path_preprocess(OD3D_Dataset.get_path_preprocess(config=config))

    @staticmethod
    def get_path_meta_from_path_preprocess(path_preprocess: Path):
        return path_preprocess.joinpath('meta')

    @staticmethod
    def get_path_preprocess(config):
        return Path(config.path_preprocess)

    @staticmethod
    def get_path_raw(config):
        return Path(config.path_raw)

    def get_frames_categories(self, max_frames_count_per_category=1, filter_single_category=False):
        dict_frames = {category: [] for category in self.categories}
        categories_filled = []
        dataloader = torch.utils.data.DataLoader(dataset=self, batch_size=10, shuffle=True,
                                                 collate_fn=self.collate_fn)
        for i, batch in enumerate(tqdm(dataloader)):
            for b in range(len(batch)):
                if batch.category is not None:
                    categories = [batch.category[b]]
                elif batch.categories is not None and (not filter_single_category or len(batch.categories[b]) == 1):
                    categories = list(set(batch.categories[b]))
                else:
                    categories = []
                for category in categories:
                    if category not in self.categories:
                        continue
                    # logger.info(batch.categories[b])
                    if len(dict_frames[category]) < max_frames_count_per_category:
                        dict_frames[category].append(batch.rgb[b])
                    else:
                        categories_filled.append(category)
                        categories_filled = list(set(categories_filled))
            if len(categories_filled) == len(self.categories):
                break

        dict_frames_stacked = {}
        for category in dict_frames.keys():
            if len(dict_frames[category]) > 0:
                dict_frames_stacked[category] = torch.stack(list(dict_frames[category]), dim=0)

        return dict_frames_stacked