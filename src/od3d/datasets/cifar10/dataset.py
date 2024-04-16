import logging

logger = logging.getLogger(__name__)

from od3d.datasets.dataset import OD3D_Dataset
from pathlib import Path
from od3d.datasets.cifar10.enum import (
    MAP_CATEGORIES_OD3D_TO_CIFAR10,
    CIFAR10_CATEGORIES,
)
from od3d.datasets.cifar10.frame import CIFAR10_FrameMeta, CIFAR10_Frame
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm


class CIFAR10(OD3D_Dataset):
    map_od3d_categories = MAP_CATEGORIES_OD3D_TO_CIFAR10
    all_categories = list(CIFAR10_CATEGORIES)
    frame_type = CIFAR10_Frame

    def get_frame_by_name_unique(self, name_unique):
        from od3d.datasets.object import OD3D_FRAME_MASK_TYPES

        return self.frame_type(
            path_raw=self.path_raw,
            path_preprocess=self.path_preprocess,
            name_unique=name_unique,
            all_categories=self.categories,
            mask_type=OD3D_FRAME_MASK_TYPES.SAM,
            modalities=self.modalities,
        )

    @staticmethod
    def setup(config):
        path_pascal3d_raw = Path(config.path_raw)

        import torchvision

        pt_cifar10_train = torchvision.datasets.CIFAR10(
            root=str(path_pascal3d_raw),
            train=True,
            transform=None,
            target_transform=None,
            download=True,
        )
        pt_cifar10_test = torchvision.datasets.CIFAR10(
            root=str(path_pascal3d_raw),
            train=False,
            transform=None,
            target_transform=None,
            download=True,
        )

    @staticmethod
    def extract_meta(config: DictConfig):
        path_pascal3d_raw = Path(config.path_raw)
        path_meta = CIFAR10.get_path_meta(config=config)

        from torch.utils.data import DataLoader
        import torchvision

        pt_cifar10_train = torchvision.datasets.CIFAR10(
            root=str(path_pascal3d_raw),
            train=True,
            transform=torchvision.transforms.ToTensor(),
            target_transform=None,
            download=True,
        )
        pt_cifar10_test = torchvision.datasets.CIFAR10(
            root=str(path_pascal3d_raw),
            train=False,
            transform=torchvision.transforms.ToTensor(),
            target_transform=None,
            download=True,
        )

        subsets = {"train": pt_cifar10_train, "test": pt_cifar10_test}
        for subset_name, subset in subsets.items():
            dataloader = DataLoader(subset, batch_size=1, shuffle=False)

            for i, batch in tqdm(enumerate(iter(dataloader))):
                batch_img = batch[0]

                batch_class_id = batch[1]

                category = CIFAR10_CATEGORIES.list()[batch_class_id[0].item()]
                name = f"{i}"
                name_unique = f"{subset_name}/{category}/{name}"
                rfpath = Path("rgb").joinpath(f"{name_unique}.png")
                logger.info(name_unique)

                from od3d.cv.io import write_image

                write_image(img=batch_img[0], path=path_pascal3d_raw.joinpath(rfpath))

                frame_meta = CIFAR10_FrameMeta(
                    subset=subset_name,
                    category=category,
                    name=name,
                    rfpath_rgb=rfpath,
                    l_size=list(batch_img.shape[2:]),
                )
                frame_meta.save(path_meta=path_meta)

                # logger.info(i)
                # logger.info(batch_img.shape)
                # logger.info(batch_class_id.shape)
