import logging
logger = logging.getLogger(__name__)
from torch.utils.data import Dataset
from omegaconf import OmegaConf, DictConfig
from enum import Enum
from pathlib import Path
import torch
from typing import List
from od3d.datasets.frame import OD3D_FRAME_MODALITIES, OD3D_Frame
from od3d.datasets.frames import OD3D_Frames
from od3d.data import ExtEnum
class OD3D_SEQ_MODALITIES(str, Enum):
    PCL = 'pcl'

class OD3D_DATASET_SPLITS(str, ExtEnum):
    SEQUENCES_SEPARATED = 'sequences_separated'
    RANDOM = 'random'
    SEQUENCES_SHARED = 'seqences_shared'


class OD3D_Dataset(Dataset):
    subclasses = {}

    @staticmethod
    def create_from_config(config: DictConfig, transform=None):
        raise NotImplementedError
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls
    def __init__(self, name: str, modalities: List[OD3D_FRAME_MODALITIES], path_raw: Path, path_preprocess: Path, transform=None, index_shift=0, subset_fraction=1.):
        self.name = name
        self.path_raw: Path = Path(path_raw)
        self.path_preprocess: Path = Path(path_preprocess)
        self.subset_fraction: float = subset_fraction

        if transform is None:
            import torchvision
            from od3d.cv.transforms.rgb import RGB_UInt8ToFloat, RGB_Normalize, RGB_Random
            transform = torchvision.transforms.Compose([
                # RGB_Random(),
                RGB_UInt8ToFloat(),
            ])

        self.transform = transform
        self.index_shift = index_shift
        self.modalities = modalities
        self.splits_featured = [OD3D_DATASET_SPLITS.RANDOM]

    def get_subset_with_names_unique(self, names_unique: List[str]):
        raise NotImplementedError

    def get_subset_with_item_ids(self, item_ids):
        raise NotImplementedError

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

    def get_split(self, fraction1: float, fraction2: float):
        fraction1 = self.get_fractionA_from_fractionA_and_fraction_B(fraction1, fraction2)
        if OD3D_DATASET_SPLITS.SEQUENCES_SEPARATED in self.splits_featured:
            return self.get_split_sequences_separated(fraction1=fraction1)
        else:
            return self.get_split_random(fraction1=fraction1)


    def get_split_random(self, fraction1: float):
        item_ids_subsetA, item_ids_subsetB = self.get_split_item_ids(fraction1=fraction1)
        co3d_subsetA = self.get_subset_with_item_ids(item_ids=item_ids_subsetA)
        co3d_subsetB = self.get_subset_with_item_ids(item_ids=item_ids_subsetB)
        return co3d_subsetA, co3d_subsetB

    def get_split_sequences_shared(self, fraction1: float):
        raise NotImplementedError

    def get_split_sequences_separated(self, fraction1: float):
        raise NotImplementedError


    def __len__(self):
        raise NotImplementedError
    def __getitem__(self, item):
        item_id_shift = (item + self.index_shift) % len(self)
        frame = self.transform(self.get_item(item_id_shift))
        frame.item_id = item_id_shift
        return frame

    def get_item(self, item):
        raise NotImplementedError

    def collate_fn(self, frames: List[OD3D_Frame], device='cpu', dtype=torch.float32):
        frames = OD3D_Frames.get_frames_from_list(frames, modalities=self.modalities, dtype=dtype, device=device)
        return frames

    def get_dataloader(self, batch_size=1, shuffle=False):
        dataloader = torch.utils.data.DataLoader(dataset=self, batch_size=1, shuffle=False, collate_fn=self.collate_fn)
        return dataloader

    @staticmethod
    def setup(config: DictConfig):
        raise NotImplementedError
    def visualize(self, item: int):
        raise NotImplementedError

    @property
    def path_meta(self):
        return self.path_preprocess.joinpath('meta')

    @staticmethod
    def get_path_meta(config):
        return OD3D_Dataset.get_path_preprocess(config=config).joinpath('meta')

    @staticmethod
    def get_path_preprocess(config):
        return Path(config.path_preprocess)

    @staticmethod
    def get_path_raw(config):
        return Path(config.path_raw)
