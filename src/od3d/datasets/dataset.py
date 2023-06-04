
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
    def __init__(self, config: DictConfig, transform=None):
        self.config = config
        if transform is None:
            import torchvision
            from od3d.cv.transforms.center_and_zoom3d import CenterZoom3D
            from od3d.cv.transforms.rgb import RGB_UInt8ToFloat, RGB_Normalize
            from od3d.cv.transforms.center_and_zoom3d import CenterZoom3D
            transform = torchvision.transforms.Compose([
                RGB_UInt8ToFloat(),
                RGB_Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                CenterZoom3D(H=512, W=512, dist=12.),
            ])
        self.transform = transform
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
