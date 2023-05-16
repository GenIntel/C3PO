import os
import cv2
import numpy as np
from od3d.datasets.dataset import OD3D_Dataset
from od3d.datasets.synthetic_shapenet.setup import download_shapenet, prepare_shapenet
from omegaconf import DictConfig

class SyntheticShapeNet(OD3D_Dataset):
    def __init__(self,
                 config: DictConfig,
                 # data_type,
                 # category,
                 # root_path,
                 # data_camera_mode='shapnet_car',
                 # transforms=[],
                 # **kwargs
                 ):
        super().__init__(config=config)
        self.data_type = self.config.get("data_type", None)
        if self.data_type is None:
            return

        self.root_path = self.config.root_path
        self.category = self.config.get("category", "all")
        self.subtypes = {}
        self.occ_level = self.config.occ_level
        self.enable_cache = self.config.enable_cache
        self.weighted = self.config.weighted
        self.remove_no_bg = self.config.remove_no_bg
        self.skip_kp = self.config.get("skip_kp", False)
        self.segmentation_masks = self.config.get("segmentation_masks", [])
        self.mesh_path = self.config.mesh_path
        self.transforms = []
        self.data_camera_mode = self.config.get("data_camera_mode", 'shapnet_car')

        self.img_path = os.path.join(self.root_path, self.data_type, self.category, 'img')
        self.anno_path = os.path.join(self.root_path, self.data_type, self.category, 'anno')

        self.file_list = [x[:-4] for x in os.listdir(self.img_path) if x.endswith('.png')]

    def setup(self):
        download_shapenet(self.config)
        prepare_shapenet(self.config)
    def __getitem__(self, item):
        name = self.file_list[item]

        ori_img = cv2.imread(os.path.join(self.img_path, f'{name}.png'), cv2.IMREAD_UNCHANGED)
        anno = np.load(os.path.join(self.anno_path, f'{name}.npy'), allow_pickle=True)[()]

        img = ori_img[:, :, :3][:, :, ::-1]
        mask = ori_img[:, :, 3:4]

        condinfo = np.array([
            anno['rotation'],
            np.pi / 2.0 - anno['elevation']
        ], dtype=np.float32)

        sample = {}
        sample['img'] = np.ascontiguousarray(img)
        sample['mask'] = np.ascontiguousarray(mask)
        sample['condinfo'] = condinfo
        return sample

    def __len__(self):
        return len(self.file_list)
