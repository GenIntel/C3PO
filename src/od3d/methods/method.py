import abc
import subprocess
import sys
class OD3DMethod(abc.ABC):
    subclasses = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls
    def __init__(self, config):
        self.config = config
    @abc.abstractmethod
    def setup(self):
        raise NotImplementedError
    @abc.abstractmethod
    def train(self):
        raise NotImplementedError

    @abc.abstractmethod
    def test(self):
        raise NotImplementedError
    def install(self, package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])