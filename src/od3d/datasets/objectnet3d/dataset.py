import logging
logger = logging.getLogger(__name__)
import od3d.io
import shutil
# from od3d.datasets.dataset import OD3D_Dataset
from od3d.datasets.pascal3d.frame import Pascal3DFrameMeta
from od3d.datasets.dataset import OD3D_Dataset
from od3d.datasets.frame import OD3D_FRAME_MODALITIES, OD3D_Frame
from od3d.datasets.objectnet3d.enum import OBJECTNET3D_CATEGORIES, MAP_CATEGORIES_OD3D_TO_OBJECTNET3D
from pathlib import Path
from typing import List, Dict
from omegaconf import DictConfig
from od3d.datasets.objectnet3d.frame import ObjectNet3D_FrameMeta, ObjectNet3D_Frame # , OD3D_Frame
from tqdm import tqdm


class ObjectNet3D(OD3D_Dataset):

    CATEGORIES = OBJECTNET3D_CATEGORIES
    MAP_OD3D_CATEGORIES = MAP_CATEGORIES_OD3D_TO_OBJECTNET3D

    def __init__(self, name: str, modalities: List[OD3D_FRAME_MODALITIES], path_raw: Path, path_preprocess: Path,
                 categories: List[OBJECTNET3D_CATEGORIES]=None,
                 dict_nested_frames: Dict=None, dict_nested_frames_ban: Dict=None,
                 transform=None, index_shift=0, subset_fraction=1., filter_frames_categorical=False):

        if categories is not None:
            categories = [self.MAP_OD3D_CATEGORIES.get(category, category) if category not in self.CATEGORIES.list() else category for category in categories]
        else:
            categories = self.CATEGORIES.list()

        self.filter_frames_categorical = filter_frames_categorical
        super().__init__(categories=categories, dict_nested_frames=dict_nested_frames, dict_nested_frames_ban=dict_nested_frames_ban, name=name, modalities=modalities, path_raw=path_raw,
                         path_preprocess=path_preprocess, transform=transform, index_shift=index_shift,
                         subset_fraction=subset_fraction)


        self.path_meshes = self.path_raw.joinpath('CAD')
        # logger.info(self.list_frames_unique)

    def get_item(self, item):
        frame_meta = ObjectNet3D_FrameMeta.load_from_meta_with_name_unique(path_meta=self.path_meta, name_unique=self.list_frames_unique[item])
        return ObjectNet3D_Frame(path_meshes=self.path_meshes, path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta, meta=frame_meta, modalities=self.modalities, categories=self.categories)

    def filter_list_frames_unique(self, list_frames_unique):
        list_frames_unique = super().filter_list_frames_unique(list_frames_unique)

        if self.filter_frames_categorical:
            logger.info('filtering frames categorical...')

            allowed_frames_unique = []
            allowed_subsets = set([f_unique.split('/')[0] for f_unique in list_frames_unique])
            for subset in allowed_subsets:
                for category in self.categories:
                    logger.info(f'{subset}, {category}')
                    allowed_frames_unique += self.get_subset_category_names_unique(subset=subset, category=category)
            list_frames_unique_filtered = list(set.intersection(set(allowed_frames_unique), set(list_frames_unique)))
            list_frames_unique_filtered = sorted(list_frames_unique_filtered)
            # for frame_name_unique in tqdm(list_frames_unique):
            #     meta = ObjectNet3D_FrameMeta.load_from_meta_with_name_unique(path_meta=self.path_meta, name_unique=frame_name_unique)
            #     if meta.category in self.categories: # filter with only first object's category
            #         list_frames_unique_filtered.append(frame_name_unique)
            #     #if len(set(self.categories).intersection(set(meta.categories))) > 0:
            #     #    list_frames_unique_filtered.append(frame_name_unique)
            list_frames_unique = list_frames_unique_filtered
        return list_frames_unique

    @staticmethod
    def setup(config: DictConfig):
        # logger.info(OmegaConf.to_yaml(config))
        path_raw = Path(config.path_raw)
        if path_raw.exists() and config.setup.remove_previous:
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

            if path_dir.exists() and not config.setup.override:
                logger.info(f"Found {name} of ObjectNet3D at {path_dir}")
            else:
                od3d.io.download(url=url, fpath=fpath)
                od3d.io.unzip(path_raw.joinpath(f'{name}.zip'), dst=path_raw.joinpath('tmp'))
                od3d.io.move_dir(src=path_raw.joinpath('tmp', 'ObjectNet3D'), dst=path_raw)

    @staticmethod
    def extract_meta(config: DictConfig):
        path = Path(config.path_raw)
        path_meta = ObjectNet3D.get_path_meta(config=config)
        path_raw = Path(config.path_raw)
        rfpath_meshes = Path('CAD')
        rfpath_annotations = Path('Annotations')
        rfpath_images = Path('Images')

        subsets = ['train', 'test', 'val']
        for subset in subsets:
            path_frames_subset = ObjectNet3D_FrameMeta.get_path_frames_meta_with_subset(path_meta=path_meta, subset=subset)
            if not path_frames_subset.exists() or config.extract_meta.override:
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

    ##### PREPROCESS
    def preprocess(self, config_preprocess: DictConfig):
        logger.info("preprocess")
        for key in config_preprocess.keys():
            # if key == 'cuboid' and config_preprocess.cuboid.get('enabled', False):
            #     override = config_preprocess.cuboid.get('override', False)
            #     remove_previous = config_preprocess.cuboid.get('remove_previous', False)
            #     self.preprocess_cuboids(override=override, remove_previous=remove_previous)
            if key == 'mask' and config_preprocess.mask.get('enabled', False):
                override = config_preprocess.mask.get('override', False)
                remove_previous = config_preprocess.mask.get('remove_previous', False)
                self.preprocess_masks(override=override, remove_previous=remove_previous)
            if key == 'depth' and config_preprocess.depth.get('enabled', False):
                override = config_preprocess.depth.get('override', False)
                remove_previous = config_preprocess.mask.get('remove_previous', False)
                self.preprocess_depths(override=override, remove_previous=remove_previous)
            if key == 'subset_category_names_unique' and config_preprocess.subset_category_names_unique.get('enabled', False):
                override = config_preprocess.subset_category_names_unique.get('override', False)
                remove_previous = config_preprocess.subset_category_names_unique.get('remove_previous', False)
                self.preprocess_subset_category_names_unique(override=override, remove_previous=remove_previous)

    # def preprocess_cuboids(self, override=False, remove_previous=False):
    #     logger.info('preprocess cuboids...')
    #     perc_axis_coverage = 0.99
    #     verts_count = 1000
    #
    #     for path_meshes_category in tqdm(self.path_meshes.iterdir()):
    #         if not path_meshes_category.is_dir():
    #             continue
    #
    #         fpath = self.path_cuboids.joinpath(f'{path_meshes_category.name}.ply')
    #         if not fpath.exists() or override:
    #             paths_meshes_category = []
    #             for path_mesh_category in path_meshes_category.iterdir():
    #                 print(path_mesh_category)
    #                 paths_meshes_category.append(path_mesh_category)
    #
    #             meshes = Meshes.load_from_files(paths_meshes_category)
    #
    #             verts_count_axis_coverage = int(meshes.verts.shape[0] * perc_axis_coverage)
    #
    #             verts_sorted = meshes.verts.sort(dim=0)[0]
    #             verts_group = verts_sorted[verts_count_axis_coverage::] - verts_sorted[0:-verts_count_axis_coverage]
    #             min_ids = verts_group.min(dim=0)[1]
    #             cuboid_limits = verts_sorted[
    #                 torch.stack([min_ids, min_ids + verts_count_axis_coverage], dim=0)].diagonal(dim1=-2, dim2=-1)
    #
    #             category = path_meshes_category.name
    #             cuboid_limits = cuboid_limits * PASCAL3D_SCALE_NORMALIZE_TO_REAL[category]
    #             meshes = Cuboids.create_dense_from_limits(limits=cuboid_limits[None,], verts_count=verts_count)
    #
    #             fpath.parent.mkdir(parents=True, exist_ok=True)
    #             save_ply(fpath, verts=meshes.verts, faces=meshes.faces)

    def get_subset_with_dict_nested_frames(self, dict_nested_frames):
        return ObjectNet3D(name=self.name, modalities=self.modalities, path_raw=self.path_raw,
                           path_preprocess=self.path_preprocess, categories=self.categories,
                           dict_nested_frames=dict_nested_frames, transform=self.transform,
                           index_shift=self.index_shift, filter_frames_categorical=self.filter_frames_categorical)

    def preprocess_subset_category_names_unique(self, override=False, remove_previous=False):
        logger.info('preprocess subset_category_names_unique...')
        dict_subset_category_names_unique = {}
        if self.filter_frames_categorical:
            msg = f'Preprocessing requires to not filter categorically. Set `filter_frames_categorical` to `False`'
            raise Exception(msg)
        for frame_id in tqdm(range(len(self))):
            frame_meta = self.get_item(frame_id).meta
            if frame_meta.subset not in dict_subset_category_names_unique.keys():
                dict_subset_category_names_unique[frame_meta.subset] = {}
            if frame_meta.category not in dict_subset_category_names_unique[frame_meta.subset]:
                dict_subset_category_names_unique[frame_meta.subset][frame_meta.category] = []
            dict_subset_category_names_unique[frame_meta.subset][frame_meta.category].append(frame_meta.name_unique)

        for subset in dict_subset_category_names_unique.keys():
            for category in dict_subset_category_names_unique[subset]:
                fpath = self.path_preprocess.joinpath('subset_category_names_unique', f'{subset}_{category}.yaml')
                if not fpath.exists() or override:
                    od3d.io.write_list_as_yaml(fpath=fpath, _list=dict_subset_category_names_unique[subset][category])
                else:
                    logger.warning(f'not overriding {fpath}, set override flag if desired.')
    def get_subset_category_names_unique(self, subset: str, category: str):
        fpath = self.path_preprocess.joinpath('subset_category_names_unique', f'{subset}_{category}.yaml')
        if not fpath.exists():
            logger.warning('preprocess subset_category_names_unique first ...')
            return []
        names_unique = od3d.io.read_list_from_yaml(fpath)
        return names_unique

    def preprocess_masks(self, override=False, remove_previous=False):
        logger.info('preprocess masks...')
        for frame_id in tqdm(range(len(self))):
            frame = self.get_item(frame_id)
            frame.preprocess_mask(override=override)
    def preprocess_depths(self, override=False, remove_previous=False):
        logger.info('preprocess depths...')
        for frame_id in tqdm(range(len(self))):
            frame = self.get_item(frame_id)
            frame.preprocess_depth(override=override)
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