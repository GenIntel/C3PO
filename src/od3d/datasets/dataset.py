
from torch.utils.data import Dataset
from omegaconf import OmegaConf, DictConfig
from enum import Enum

class OD3D_FRAME_MODALITIES(str, Enum):
    RGB = 'rgb'
    MASK = 'mask'
    DEPTH = 'depth'
    DEPTH_MASK = 'depth_mask'

class OD3D_SEQ_MODALITIES(str, Enum):
    PCL = 'pcl'

class OD3D_Dataset(Dataset):
    subclasses = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls
    def __init__(self, config: DictConfig):
        self.config = config
        self.index_shift = self.config.get('index_shift', 0)

    def __len__(self):
        raise NotImplementedError
    def __getitem__(self, item):
        return self.get_item((item + self.index_shift) % len(self))
    def get_item(self, item):
        raise NotImplementedError
    @staticmethod
    def collate_fn(frames: list, modalities: list[OD3D_FRAME_MODALITIES]):
        return frames
    @staticmethod
    def setup(config: DictConfig):
        raise NotImplementedError
    def visualize(self, item: int):
        raise NotImplementedError
