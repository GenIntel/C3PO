import logging
logger = logging.getLogger(__name__)
import od3d.io
import shutil
# from od3d.datasets.dataset import OD3D_Dataset
from od3d.datasets.pascal3d.frame import Pascal3DFrameMeta
from od3d.datasets.dataset import OD3D_Dataset
from od3d.datasets.frame import OD3D_FRAME_MODALITIES, OD3D_Frame
from od3d.datasets.objectnet3d.enum import OBJECTNET3D_CATEOGORIES
from pathlib import Path
from typing import List, Dict
from omegaconf import DictConfig
from od3d.datasets.objectnet3d.frame import ObjectNet3D_FrameMeta, ObjectNet3D_Frame # , OD3D_Frame
from tqdm import tqdm


class ObjectNet3D(OD3D_Dataset):
    def __init__(self, name: str, modalities: List[OD3D_FRAME_MODALITIES], path_raw: Path, path_preprocess: Path,
                 categories: List[OBJECTNET3D_CATEOGORIES]=None,
                 dict_nested_frames: Dict=None,
                 transform=None, index_shift=0, subset_fraction=1.):

        categories = categories if categories is not None else OBJECTNET3D_CATEOGORIES.list()
        super().__init__(categories=categories, dict_nested_frames=dict_nested_frames, name=name, modalities=modalities, path_raw=path_raw,
                         path_preprocess=path_preprocess, transform=transform, index_shift=index_shift,
                         subset_fraction=subset_fraction)

    def get_item(self, item):
        frame_meta = ObjectNet3D_FrameMeta.load_from_meta_with_name_unique(path_meta=self.path_meta, name_unique=self.list_frames_unique[item])
        return OD3D_Frame(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta, meta=frame_meta, modalities=self.modalities, categories=self.categories)


    @staticmethod
    def setup(config: DictConfig):
        # logger.info(OmegaConf.to_yaml(config))
        path_raw = Path(config.path_raw)
        if path_raw.exists() and config.setup_remove_previous:
            logger.info(f"Removing previous ObjectNet3D")
            shutil.rmtree(path_raw)

        path_raw.mkdir(parents=True, exist_ok=True)

        dict_name_to_url = {
            "Images": "ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_images.zip",
            "Annotations": "ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_annotations.zip",
            "CAD": "ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_cads.zip",
            "Image_sets": "ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_image_sets.zip",
        }
        for name, url in dict_name_to_url.items():
            fpath=path_raw.joinpath(f'{name}.zip')
            path_dir = path_raw.joinpath(name)

            if path_dir.exists() and not config.setup_override:
                logger.info(f"Found {name} of ObjectNet3D at {path_dir}")
            else:
                od3d.io.download(url=url, fpath=fpath)
                od3d.io.unzip(path_raw.joinpath(f'{name}.zip'), dst=path_raw.joinpath('tmp'))
                od3d.io.move_dir(src=path_raw.joinpath('tmp', 'ObjectNet3D'), dst=path_raw)

    @staticmethod
    def preprocess_meta(config: DictConfig):
        path = Path(config.path_raw)
        path_meta = ObjectNet3D.get_path_meta(config=config)
        path_raw = Path(config.path_raw)
        rfpath_meshes = Path('CAD')
        rfpath_annotations = Path('Annotations')
        rfpath_images = Path('Images')

        subsets = ['test'] #  ['train', 'test', 'val']
        for subset in subsets:
            path_frames_subset = ObjectNet3D_FrameMeta.get_path_frames_meta_with_subset(path_meta=path_meta, subset=subset)
            if not path_frames_subset.exists() or config.preprocess_meta_override:
                fpath_image_set_subset = path_raw.joinpath('Image_sets', subset + '.txt')
                frames_str = od3d.io.read_str_from_file(fpath=fpath_image_set_subset)
                logger.info(f'preprocess subset {subset}')
                frames_names = frames_str.split()
                for i in tqdm(range(len(frames_names))):
                    frame_name = frames_names[i]
                    # logger.info(f'preprocess {frame_name}')
                    frame_meta = ObjectNet3D_FrameMeta.load_from_raw(path_raw=path_raw,
                                                                     rfpath_annotations=rfpath_annotations,
                                                                     rfpath_images=rfpath_images,
                                                                     rfpath_meshes=rfpath_meshes, subset=subset,
                                                                     name=frame_name)

                    if frame_meta is not None:
                        frame_meta.save(path_meta=path_meta)
            else:
                logger.info(f'found subset frames at {path_frames_subset}')

        def preprocess(self, override: False):
            pass
"""
class ObjectNet3D(Pascal3D):

    def __init__(self, name: str, modalities: List[OD3D_FRAME_MODALITIES], path_raw: Path, path_preprocess: Path,
                 path_cuboids: Path, categories: List[OBJECTNET3D_CATEOGORIES]=None,
                 dict_subset_category_frames_names: Dict[str, Dict[str, List[str]]]=None,
                 transform=None, index_shift=0, subset_fraction=1.):

        super().__init__(name=name, modalities=modalities, path_raw=path_raw, path_preprocess=path_preprocess,
                         path_cuboids=path_cuboids, transform=transform, index_shift=index_shift,
                         subset_fraction=subset_fraction, categories=categories,
                         dict_subset_category_frames_names=dict_subset_category_frames_names)


    @staticmethod
    def setup(config: DictConfig):
        # logger.info(OmegaConf.to_yaml(config))
        path_raw = Path(config.path_raw)
        if path_raw.exists() and config.setup_remove_previous:
            logger.info(f"Removing previous ObjectNet3D")
            shutil.rmtree(path_raw)

        path_raw.mkdir(parents=True, exist_ok=True)

        dict_name_to_url = {
            "Images": "ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_images.zip",
            "Annotations": "ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_annotations.zip",
            "CAD": "ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_cads.zip",
            "Image_sets": "ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_image_sets.zip",
        }
        for name, url in dict_name_to_url.items():
            fpath=path_raw.joinpath(f'{name}.zip')
            path_dir = path_raw.joinpath(name)

            if path_dir.exists() and not config.setup_override:
                logger.info(f"Found {name} of ObjectNet3D at {path_dir}")
            else:
                od3d.io.download(url=url, fpath=fpath)
                od3d.io.unzip(path_raw.joinpath(f'{name}.zip'), dst=path_raw.joinpath('tmp'))
                od3d.io.move_dir(src=path_raw.joinpath('tmp', 'ObjectNet3D'), dst=path_raw)

    @staticmethod
    def preprocess_meta(config: DictConfig):
        path = Path(config.path_raw)
        path_meta = ObjectNet3D.get_path_meta(config=config)
        path_raw = Path(config.path_raw)
        rfpath_meshes = Path('CAD')
        rfpath_annotations = Path('Annotations')
        rfpath_images = Path('Images')

        subsets = ['train', 'test', 'val']
        for subset in subsets:
            path_frames_subset = ObjectNet3D_FrameMeta.get_path_frames_meta_with_subset(path_meta=path_meta, subset=subset)
            if not path_frames_subset.exists() or config.preprocess_meta_override:
                fpath_image_set_subset = path_raw.joinpath('Image_sets', subset + '.txt')
                frames_str = od3d.io.read_str_from_file(fpath=fpath_image_set_subset)
                logger.info(f'preprocess subset {subset}')
                frames_names = frames_str.split()
                for i in tqdm(range(len(frames_names))):
                    frame_name = frames_names[i]
                    # logger.info(f'preprocess {frame_name}')
                    frame_meta = ObjectNet3D_FrameMeta.load_meta_from_raw(path_raw=path_raw,
                                                                          rfpath_annotations=rfpath_annotations,
                                                                          rfpath_images=rfpath_images,
                                                                          rfpath_meshes=rfpath_meshes, subset=subset,
                                                                          name=frame_name)

                    if frame_meta.complete:
                        frame_meta.save(path_meta=path_meta)
            else:
                logger.info(f'found subset frames at {path_frames_subset}')

    @staticmethod
    def print_classes(config: DictConfig):
        path_raw = Path(config.path_raw)
        raw_classes_fpath = path_raw.joinpath('Image_sets', 'classes.txt')
        categories_str = od3d.io.read_str_from_file(fpath=raw_classes_fpath)
        for c in categories_str.split():
            print(f'{c.upper()}="{c}"')

    def get_frame_by_subset_category_name(self, subset: str, category: str, name: str):
        frame_meta = ObjectNet3D_FrameMeta.load_from_meta_with_subset_category_name(path_meta=self.path_meta, subset=subset,
                                                                                category=category, name=name)
        return self.get_frame_by_meta(frame_meta=frame_meta)

    def get_frame_by_name_unique(self, name_unique: str):
        frame_meta = ObjectNet3D_FrameMeta.load_from_meta_with_name_unique(path_meta=self.path_meta, name_unique=name_unique)
        return self.get_frame_by_meta(frame_meta=frame_meta)

    def get_frame_by_meta(self, frame_meta: ObjectNet3D_FrameMeta):
        return ObjectNet3D_Frame(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
                                 path_meshes=self.path_meshes, meta=frame_meta, modalities=self.modalities,
                                 categories=self.categories)
"""