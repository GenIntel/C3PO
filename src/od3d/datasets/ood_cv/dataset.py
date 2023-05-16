
from od3d.datasets.dataset import OD3D_Dataset
from od3d.datasets.ood_cv.setup import download_ood_cv, prepare_ood_cv
from omegaconf import DictConfig

class OOD_CV(OD3D_Dataset):
    def __init__(
        self,
        config: DictConfig
    ):
        super().__init__(config=config)
        self.setup()
    def setup(self):
        download_ood_cv(self.config)
        prepare_ood_cv(self.config)