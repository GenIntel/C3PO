import logging
from od3d.datasets.dataset import OD3D_Dataset
import scipy.io as sio
import numpy as np
from omegaconf import DictConfig
import od3d.io
from pathlib import Path
class DTD(OD3D_Dataset):
    def __init__(self, config):
        super().__init__(config=config)
        self.setup(self.config)
        dtd_raw_path = Path(self.config.dtd_raw_path)
        dtd_mat = sio.loadmat(dtd_raw_path.joinpath("imdb", "imdb.mat"))
        images = dtd_mat["images"]
        ids = images[0, 0][0][0]
        rfpaths = np.array([f[0] for f in images[0, 0][1][0]])
        splits = images[0, 0][2][0]
        classes = images[0, 0][3][0]
        train_rfpaths = []
        val_rfpaths = []
        for j, f, s in zip(ids, rfpaths, splits):
            if s == 1:
                train_rfpaths.append(f)
            elif s == 2:
                val_rfpaths.append(f)
            # note: s = 1 2 3
        self.rfpaths = {"train": train_rfpaths, "val": val_rfpaths}
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

