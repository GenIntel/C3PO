import copy
import os

import BboxTools as bbt
import numpy as np
import torch
import torchvision
from PIL import Image
import skimage

from od3d.utils import construct_class_by_name
from od3d.utils import get_abs_path
from od3d.utils import load_off
from od3d.utils.pascal3d_utils import CATEGORIES
from od3d.datasets.dataset import OD3DDataset
from od3d.datasets.pascal3d.setup import download_pascal3d, prepare_pascal3d
from omegaconf import DictConfig

class Pascal3DPlus(OD3DDataset):
    def __init__(
        self,
        config: DictConfig,
        #data_type,
        #category,
        #root_path,
        #transforms,
        #mesh_path,
        #subtypes=None,
        #occ_level=0,
        #enable_cache=True,
        #weighted=True,
        #remove_no_bg=None,
        #skip_kp=False,
        #segmentation_masks=[],
        # **kwargs,
    ):
        super().__init__(config=config)
        self.data_type = self.config.get("data_type", None)
        if self.data_type is None:
            return

        self.root_path = self.config.root_path
        self.category = self.config.get("category", "all")
        self.subtypes = self.config.subtypes if self.config.subtypes is not None else {}
        self.occ_level = self.config.occ_level
        self.enable_cache = self.config.enable_cache
        self.weighted = self.config.weighted
        self.remove_no_bg = self.config.remove_no_bg
        self.skip_kp = self.config.get("skip_kp", False)
        self.segmentation_masks = self.config.get("segmentation_masks", [])
        self.mesh_path = self.config.mesh_path
        self.transforms = torchvision.transforms.Compose(
            [construct_class_by_name(**t) for t in self.config.transforms]
        )

        if self.category == 'all':
            self.category = CATEGORIES
        if not isinstance(self.category, list):
            self.category = [self.category]
        self.multi_cate = len(self.category) > 1

        self.image_path = os.path.join(self.root_path, self.data_type, "images")
        self.annotation_path = os.path.join(self.root_path, self.data_type, "annotations")
        self.list_path = os.path.join(self.root_path, self.data_type, "lists")


        num_verts = []
        for cate in self.category:
            num_verts.append(load_off(os.path.join(self.mesh_path, cate, '01.off'))[0].shape[0])
        self.max_n = max(num_verts)

        file_list = []
        for cate in self.category:
            if self.occ_level == 0:
                _list_path = os.path.join(self.list_path, cate)
            else:
                _list_path = os.path.join(self.list_path, f"{cate}FGL{self.occ_level}_BGL{self.occ_level}")

            if cate not in self.subtypes:
                self.subtypes[cate] = [t.split(".")[0] for t in os.listdir(_list_path)]

            _file_list = sum(
                (
                    [
                        os.path.join(cate if self.occ_level == 0 else f"{cate}FGL{self.occ_level}_BGL{self.occ_level}", l.strip())
                        for l in open(
                            os.path.join(_list_path, subtype_ + ".txt")
                        ).readlines()
                    ]
                    for subtype_ in self.subtypes[cate]
                ),
                [],
            )
            file_list += [(f, cate) for f in _file_list]
        # OOD-CV seems to have duplicate samples -- remove duplicates from file list
        self.file_list = list(set(file_list))
        self.cache = {}

        self.filter()

    def setup(self):
        download_pascal3d(self.config)
        prepare_pascal3d(self.config)

    def filter(self):
        if self.remove_no_bg is not None:
            filtered_file_list = []
            for i in range(len(self.file_list)):
                sample = self.__getitem__(i)
                obj_mask = skimage.measure.block_reduce(sample['obj_mask'], (self.remove_no_bg, self.remove_no_bg), np.max)
                if np.sum(1-obj_mask) >= 5:
                    filtered_file_list.append(self.file_list[i])
            self.file_list = filtered_file_list

        if self.segmentation_masks is not None:
            filtered_file_list = []
            for i in range(len(self.file_list)):
                sample = self.__getitem__(i)
                if 'inmodal' in self.segmentation_masks and len(sample['inmodal_mask'].shape) < 2:
                    continue
                if 'amodal' in self.segmentation_masks and len(sample['amodal_mask'].shape) < 2:
                    continue
                filtered_file_list.append(self.file_list[i])
            self.file_list = filtered_file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        name_img, cate = self.file_list[item]

        if self.enable_cache and name_img in self.cache.keys():
            sample = copy.deepcopy(self.cache[name_img])
        else:
            img = Image.open(os.path.join(self.image_path, f"{name_img}.JPEG"))
            if img.mode != "RGB":
                img = img.convert("RGB")
            annotation_file = np.load(
                os.path.join(self.annotation_path, name_img.split(".")[0] + ".npz"),
                allow_pickle=True,
            )

            if "cropped_kp_list" in annotation_file and "visible" in annotation_file:
                kp = annotation_file["cropped_kp_list"]
                iskpvisible = annotation_file["visible"] == 1

                if self.weighted:
                    iskpvisible = iskpvisible * annotation_file["kp_weights"]

                iskpvisible = np.logical_and(
                    iskpvisible, np.all(kp >= np.zeros_like(kp), axis=1)
                )
                iskpvisible = np.logical_and(
                    iskpvisible, np.all(kp < np.array([img.size[::-1]]), axis=1)
                )

                kp = np.max([np.zeros_like(kp), kp], axis=0)
                kp = np.min(
                    [np.ones_like(kp) * (np.array([img.size[::-1]]) - 1), kp], axis=0
                )
            else:
                kp = np.zeros((100, 2), dtype=np.float32)
                iskpvisible = np.zeros((100,), dtype=np.int32)

            this_name = name_img.split(".")[0]

            try:
                box_obj = bbt.from_numpy(annotation_file["box_obj"])
                obj_mask = np.zeros(box_obj.boundary, dtype=np.float32)
                box_obj.assign(obj_mask, 1)
            except KeyboardInterrupt:
                obj_mask = np.zeros((img.size[1], img.size[0]))

            label = 0 if len(self.category) == 0 else self.category.index(cate)
            pad_size = self.max_n - kp.shape[0]
            kp = np.pad(kp, pad_width=((0, pad_size), (0, 0)), mode='constant', constant_values=0)
            iskpvisible = np.pad(iskpvisible, pad_width=(0, pad_size), mode='constant', constant_values=False)
            index = np.array([self.max_n * label + k for k in range(self.max_n)])

            sample = {
                "this_name": this_name,
                "cad_index": int(annotation_file["cad_index"]),
                "azimuth": float(annotation_file["azimuth"]),
                "elevation": float(annotation_file["elevation"]),
                "theta": float(annotation_file["theta"]),
                "distance": 5.0,
                "bbox": annotation_file["box_obj"],
                "obj_mask": obj_mask,
                "img": img,
                "original_img": np.array(img),
                "label": label,
                "index": index,
            }
            if 'amodal' in self.segmentation_masks:
                sample['amodal_mask'] = annotation_file['amodal_mask']
            if 'inmodal' in self.segmentation_masks:
                sample['inmodal_mask'] = annotation_file['inmodal_mask']
            if not self.skip_kp:
                sample['kp'] = kp.astype(np.float32)
                sample['kpvis'] = iskpvisible.astype(bool)

            if self.enable_cache:
                self.cache[name_img] = copy.deepcopy(sample)

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def debug(self, item, save_dir=""):
        sample = self.__getitem__(item)
        img = sample["original_img"]
        kp, kpvis = sample["kp"], sample["kpvis"]
        y0, y1, x0, x1, _, _ = sample["bbox"]
        obj_mask = sample["obj_mask"]

        import cv2

        for i in range(len(kp)):
            if kpvis[i]:
                img = cv2.circle(
                    img, (int(kp[i, 1]), int(kp[i, 0])), 2, (255, 0, 0), -1
                )
        img = cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)

        gray_img = (img * 0.3).astype(np.uint8)
        gray_img[obj_mask == 1] = img[obj_mask == 1]

        Image.fromarray(gray_img).save(
            os.path.join(save_dir, f'debug_{sample["this_name"].replace("/", "_")}.png')
        )
