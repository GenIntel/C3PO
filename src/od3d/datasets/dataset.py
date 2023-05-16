
from torch.utils.data import Dataset
from omegaconf import OmegaConf, DictConfig
class OD3D_Dataset(Dataset):
    subclasses = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls
    def __init__(self, config: DictConfig):
        self.config = config

    @staticmethod
    def setup(config: DictConfig):
        raise NotImplementedError
    def visualize(self, item: int):
        raise NotImplementedError