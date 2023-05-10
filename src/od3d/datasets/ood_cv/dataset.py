
from od3d.datasets.dataset import OD3DDataset
from od3d.datasets.ood_cv.setup import download_ood_cv, prepare_ood_cv
from omegaconf import DictConfig

class OOD_CV(OD3DDataset):
    def __init__(
        self,
        config: DictConfig
    ):
        super().__init__(config=config)

    def setup(self):
        download_ood_cv(self.config)
        prepare_ood_cv(self.config)