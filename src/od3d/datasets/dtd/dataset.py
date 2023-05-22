import logging
from od3d.datasets.dataset import OD3D_Dataset
import scipy.io as sio
import numpy as np
from omegaconf import DictConfig
import od3d.io
from pathlib import Path
import scipy.io
import torchvision.io


SUBSETS = [
    "train",
    "val"
]

class DTD(OD3D_Dataset):
    def __init__(self, config):
        super().__init__(config=config)
        self.setup(self.config)

        self.subsets = self.config.get("subsets", SUBSETS)
        self.path = Path(config.path_dtd_raw)
        txtrs_meta = scipy.io.loadmat(self.path.joinpath("imdb", "imdb.mat"))["images"]
        txtrs_ids = txtrs_meta[0, 0][0][0]
        txtrs_fnames = txtrs_meta[0, 0][1][0]
        txtrs_fnames = np.array([f[0] for f in txtrs_fnames])
        txtrs_subsets = txtrs_meta[0, 0][2][0]
        self.rfpaths = []
        for j, f, s in zip(txtrs_ids, txtrs_fnames, txtrs_subsets):
            if s == 1 and "train" in self.subsets:
                self.rfpaths.append(Path("images").joinpath(f))
            elif s == 2 and "val" in self.subsets:
                self.rfpaths.append(Path("images").joinpath(f))

    def __len__(self):
        return len(self.rfpaths)
    def __getitem__(self, item):
        fpath = self.path.joinpath(self.rfpaths[item])
        return torchvision.io.read_image(str(fpath))
    def get_random_item(self):
        return self.__getitem__(np.random.choice(self.__len__()))
    @staticmethod
    def setup(config: DictConfig):
        if Path(config.path_dtd_raw).exists():
            logging.info(f"Found Describable Textures Dataset at {config.path_dtd_raw}")
        else:
            logging.info(f"Download Describable Textures Dataset at {config.path_dtd_raw}")
            fpath = Path(config.path_dtd_raw).joinpath("dtd.tar.gz")
            od3d.io.download("https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz", fpath=fpath)
            od3d.io.untar(fpath=fpath, dst=fpath.parent)
            od3d.io.move_dir(src=fpath.parent.joinpath('dtd'), dst=fpath.parent)

