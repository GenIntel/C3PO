import inspect
from pathlib import Path
import od3d.io
from omegaconf import DictConfig
import abc


class OD3D_Transform(abc.ABC):

    subclasses = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def __call__(self, frame):
        pass

    @classmethod
    def create_by_name(cls, name: str):
        config = od3d.io.read_config_intern(rfpath=Path("methods/transform").joinpath(f"{name}.yaml"))
        return cls.create_from_config(config)

    @classmethod
    def create_from_config(cls, config: DictConfig):

        keys = inspect.getfullargspec(cls.subclasses[config.class_name].__init__)[0][1:]
        return cls.subclasses[config.class_name](**dict((key, config.get(key)) for key in keys if config.get(key, None) is not None))


