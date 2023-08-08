import logging
logger = logging.getLogger(__name__)
import abc
import subprocess
import sys
from od3d.benchmark.results import OD3D_Results
from od3d.datasets.dataset import OD3D_Dataset
from omegaconf import DictConfig
from typing import List, Dict

class OD3D_Method(abc.ABC):
    subclasses = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls
    def __init__(self, config: DictConfig, logging_dir):
        self.config = config
        self.logging_dir = logging_dir
    #@abc.abstractmethod
    #def setup(self):
    #    raise NotImplementedError
    @abc.abstractmethod
    def train(self, dataset: OD3D_Dataset, datasets_val: Dict[str, OD3D_Dataset]):
        raise NotImplementedError

    @abc.abstractmethod
    def test(self, datasets_test: List[OD3D_Dataset]) -> OD3D_Results:
        raise NotImplementedError


    def install(self, package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])