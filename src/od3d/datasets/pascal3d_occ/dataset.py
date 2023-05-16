import os
from od3d.datasets.dataset import OD3D_Dataset
from omegaconf import DictConfig
from pathlib import Path
import wget

class Pascal3D_Occ(OD3D_Dataset):

    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__(config=config)
        self.setup()

    def setup(self):
        self.download()

    def download(self):
        if max(self.config.occ_levels.train) == 0 and max(self.config.occ_levels.val) == 0:
            print("Skipping OccludedPASCAL3D+")
        elif os.path.isdir(self.path_pascal3d_occ_raw):
            print(f"Found OccludedPascal3D+ dataset at {self.path_pascal3d_occ_raw}")
        else:
            os.makedirs(self.config.path_pascal3d_occ_raw, exist_ok=True)
            os.chdir(self.config.path_pascal3d_occ_raw)
            print(f"Downloading OccludedPascal3D+ dataset at {self.config.path_pascal3d_occ_raw}")
            wget.download(self.config.url_pascal3d_occ_script)
            os.system("chmod +x download_FG.sh")
            os.system("sh download_FG.sh")
            os.chdir("..")