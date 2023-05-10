
from torch.utils.data import Dataset
from omegaconf import OmegaConf
class OD3DDataset(Dataset):
    subclasses = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls
    def __init__(self, config):
        self.config = config
    def setup(self):
        raise NotImplementedError

