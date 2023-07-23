import logging
logger = logging.getLogger(__name__)
import torch.nn
from typing import List, Tuple

from od3d.datasets.dataset import OD3D_Dataset, OD3D_FRAME_MODALITIES
from omegaconf import DictConfig
from pathlib import Path
import od3d.io
import shutil
from tqdm import tqdm
from od3d.cv.geometry.mesh import Meshes
from od3d.cv.geometry.primitives import Cuboids
from od3d.cv.io import save_ply
from od3d.datasets.pascal3d.frame import Pascal3DFrame, Pascal3DFrameMeta, Pascal3DFrames
from od3d.datasets.pascal3d.enum import PASCAL3D_CATEGORIES, PASCAL3D_SUBSETS, PASCAL3D_SCALE_NORMALIZE_TO_REAL
from typing import Dict
import inspect


class Pascal3D(OD3D_Dataset):


    @staticmethod
    def create_from_config(config: DictConfig, transform=None):
        if config.get("setup", False):
            Pascal3D.setup(config=config)
        if config.get("preprocess_meta", False):
            Pascal3D.preprocess_meta(config=config)

        keys = inspect.getfullargspec(Pascal3D.__init__)[0][1:]
        pascal3d = Pascal3D(**dict((key, config.get(key)) for key in keys if config.get(key, None) is not None), transform=transform)

        if config.get("preprocess", False):
            pascal3d.preprocess(override=config.get("preprocess_override", False),
                                preprocess_masks=config.get("preprocess_masks", True),
                                preprocess_cuboids=config.get("preprocess_cuboids", True))

        return pascal3d

    def __init__(
        self,
        name: str,
        modalities: List[OD3D_FRAME_MODALITIES],
        path_raw: Path,
        path_preprocess: Path,
        path_cuboids: Path,
        categories: List[PASCAL3D_CATEGORIES] = None,
        dict_subset_category_frames_names: Dict[str, Dict[str, List[str]]] = None,
        transform=None,
        subset_fraction=1.
    ):
        super().__init__(name=name, modalities=modalities, path_raw=path_raw, path_preprocess=path_preprocess, transform=transform, subset_fraction=subset_fraction)

        self.path_cuboids = Path(path_cuboids)
        self.categories = categories if categories is not None else PASCAL3D_CATEGORIES.list()

        # get frames
        self.dict_subset_category_frames_names = Pascal3DFrameMeta.get_dict_subset_category_frames_names(
            categories=self.categories, path_meta=self.path_meta,
            dict_subset_category_frames_names=dict_subset_category_frames_names)

        self.list_subsets_categories_frames_names = self.dict_subset_category_frames_names_to_list(self.dict_subset_category_frames_names)

        self.frames_count = len(self.list_subsets_categories_frames_names)
        #self.frames_meta_rfpaths = self.get_rfpaths_frames_meta()
        #self.frames_names = [rfpath.stem for rfpath in self.frames_meta_rfpaths]
        #self.frames_count = len(self.frames_meta_rfpaths)
        #
        #self.map_frame_name_to_frame_rfpath = dict(zip(self.frames_names, self.frames_meta_rfpaths))

        if self.subset_fraction is not None and self.subset_fraction != 1.:
            frames_ids_subset = self.get_subset_item_ids(subset_fraction=subset_fraction)
            self.list_subsets_categories_frames_names = [self.list_subsets_categories_frames_names[id] for id in frames_ids_subset]
            self.dict_subset_category_frames_names = self.list_subsets_categories_frames_names_to_dict(self.list_subsets_categories_frames_names)
            #self.frames_meta_rfpaths = [self.frames_meta_rfpaths[id] for id in frames_ids_subset]
            #self.frames_names = [self.frames_names[id] for id in frames_ids_subset]
            #self.map_frame_name_to_frame_rfpath = dict(zip(self.frames_names, self.frames_meta_rfpaths))
            #self.frames_count = len(self.frames_meta_rfpaths)
            self.frames_count = len(self.list_subsets_categories_frames_names)
        logger.info(f"found {self.frames_count} frames.")

    def dict_subset_category_frames_names_to_list(self, dict_subset_category_frames_names):
        list_subsets_categories_frames_names: List[Tuple[str, str, str]] = []
        for subset, dict_category_frames_names in dict_subset_category_frames_names.items():
            for category, frames_names in dict_category_frames_names.items():
                for frame_name in frames_names:
                    list_subsets_categories_frames_names.append((subset, category, frame_name))
        return list_subsets_categories_frames_names

    def list_subsets_categories_frames_names_to_dict(self, list_subsets_categories_frames_names):
        dict_subset_category_frames_names = {}
        for subset, category, frame_name in list_subsets_categories_frames_names:
            if subset not in dict_subset_category_frames_names.keys():
                dict_subset_category_frames_names[subset] = {}
            if category not in dict_subset_category_frames_names[subset].keys():
                dict_subset_category_frames_names[subset][category] = []
            dict_subset_category_frames_names[subset][category].append(frame_name)
        return dict_subset_category_frames_names

    def get_subset_with_names_unique(self, names_unique: List[str]):
        dict_subset_category_frames_names = Pascal3DFrameMeta.get_dict_subset_category_frames_names_with_names_unique(names_unique=names_unique)
        return self.get_subset_with_dict_subset_category_frames_names(dict_subset_category_frames_names=dict_subset_category_frames_names)

    def get_subset_with_dict_subset_category_frames_names(self, dict_subset_category_frames_names:  Dict[str, Dict[str, List[str]]]):
        return Pascal3D(name=self.name, modalities=self.modalities, path_raw=self.path_raw,
                        path_preprocess=self.path_preprocess, path_cuboids=self.path_cuboids,
                        categories=self.categories,
                        dict_subset_category_frames_names=dict_subset_category_frames_names,
                        transform=self.transform)
    def get_subset_with_item_ids(self, item_ids):
        list_subsets_categories_frames_names = [self.list_subsets_categories_frames_names[id] for id in item_ids]
        dict_subset_category_frames_names = self.list_subsets_categories_frames_names_to_dict(list_subsets_categories_frames_names)

        return self.get_subset_with_dict_subset_category_frames_names(dict_subset_category_frames_names=dict_subset_category_frames_names)

    ##### SETUP
    @staticmethod
    def setup(config):
        path_pascal3d_raw = Path(config.path_raw)

        if path_pascal3d_raw.exists() and config.setup_remove_previous:
            logger.info(f"Removing previous Pascal3D+")
            shutil.rmtree(path_pascal3d_raw)

        if path_pascal3d_raw.exists():
            logger.info(f"Found Pascal3D+ dataset at {path_pascal3d_raw}")
        else:
            logger.info(f"Downloading Pascal3D+ dataset at {path_pascal3d_raw}")
            fpath = path_pascal3d_raw.joinpath("pascal3d.zip")
            od3d.io.download(url=config.url_pascal3d_raw, fpath=fpath)
            od3d.io.unzip(fpath=fpath, dst=fpath.parent)
            od3d.io.move_dir(src=fpath.parent.joinpath(Path(config.url_pascal3d_raw).with_suffix("").name),
                             dst=fpath.parent)


    #### PREPROCESS META
    @staticmethod
    def preprocess_meta(config: DictConfig):
        subsets = config.get("subsets", PASCAL3D_SUBSETS.list())
        categories = config.get("categories", PASCAL3D_CATEGORIES.list())

        path_raw = Pascal3D.get_path_raw(config=config)
        path_meta = Pascal3D.get_path_meta(config=config)
        path_meshes = Pascal3D.get_path_meshes(path_raw=path_raw)

        if config.preprocess_meta_remove_previous:
            if path_meta.exists():
                shutil.rmtree(path_meta)

        frames_subsets, frames_categories, frames_names = Pascal3DFrameMeta.get_frames_names_from_subsets_and_cateogories_from_raw(path_pascal3d_raw=path_raw, subsets=subsets, categories=categories)

        if config.get('frames', None) is not None:
            frames_names = list(filter(lambda f: f in config.frames, frames_names))

        for i in tqdm(range(len(frames_names))):
            fpath = Pascal3DFrameMeta.get_fpath_frame_meta_with_category_name(path_meta=path_meta, subset=frames_subsets[i], category=frames_categories[i], name=frames_names[i])
            if not fpath.exists() or config.preprocess_meta_override:

                frame_meta = Pascal3DFrameMeta.load_from_raw(frame_name=frames_names[i], subset=frames_subsets[i], category=frames_categories[i],
                                                             path_raw=path_raw, path_meshes=path_meshes)

                if frame_meta.complete:
                    frame_meta.save(path_meta=path_meta)

    @staticmethod
    def get_path_meshes(path_raw: Path):
        return path_raw.joinpath("CAD")



    ##### PREPROCESS
    def preprocess(self, preprocess_cuboids=True, preprocess_masks=True, override=False):
        if preprocess_cuboids:
            self.preprocess_cuboids(override=override)
        if preprocess_masks:
            self.preprocess_masks(override=override)
    def preprocess_cuboids(self, override=False):
        logger.info('preprocess cuboids...')
        perc_axis_coverage = 0.99
        verts_count = 1000

        for path_meshes_category in tqdm(self.path_meshes.iterdir()):
            if not path_meshes_category.is_dir():
                continue

            fpath = self.path_cuboids.joinpath(f'{path_meshes_category.name}.ply')
            if not fpath.exists() or override:
                paths_meshes_category = []
                for path_mesh_category in path_meshes_category.iterdir():
                    print(path_mesh_category)
                    paths_meshes_category.append(path_mesh_category)

                meshes = Meshes.load_from_files(paths_meshes_category)

                verts_count_axis_coverage = int(meshes.verts.shape[0] * perc_axis_coverage)

                verts_sorted = meshes.verts.sort(dim=0)[0]
                verts_group = verts_sorted[verts_count_axis_coverage::] - verts_sorted[0:-verts_count_axis_coverage]
                min_ids = verts_group.min(dim=0)[1]
                cuboid_limits = verts_sorted[torch.stack([min_ids, min_ids + verts_count_axis_coverage], dim=0)].diagonal(dim1=-2, dim2=-1)

                category = path_meshes_category.name
                cuboid_limits = cuboid_limits * PASCAL3D_SCALE_NORMALIZE_TO_REAL[category]
                meshes = Cuboids.create_dense_from_limits(limits=cuboid_limits[None,], verts_count=verts_count)


                fpath.parent.mkdir(parents=True, exist_ok=True)
                save_ply(fpath, verts=meshes.verts, faces=meshes.faces)

    def preprocess_masks(self, override=False):
        logger.info('preprocess masks...')
        for frame_id in tqdm(range(len(self))):
            frame = self.get_item(frame_id)
            frame.preprocess_mask(override=override)

    ##### DATASET PROPERTIES
    def __len__(self):
        return self.frames_count
    def get_frame_by_subset_category_name(self, subset: str, category: str, name: str):
        frame_meta = Pascal3DFrameMeta.load_from_meta_with_subset_category_name(path_meta=self.path_meta, subset=subset,
                                                                                category=category, name=name)
        return self.get_frame_by_meta(frame_meta=frame_meta)

    def get_frame_by_name_unique(self, name_unique: str):
        frame_meta = Pascal3DFrameMeta.load_from_meta_with_name_unique(path_meta=self.path_meta, name_unique=name_unique)
        return self.get_frame_by_meta(frame_meta=frame_meta)

    def get_frame_by_meta(self, frame_meta: Pascal3DFrameMeta):
        return Pascal3DFrame(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
                             path_meshes=self.path_meshes, meta=frame_meta, modalities=self.modalities,
                             categories=self.categories)

    def get_item(self, item):
        subset, category, frame_name = self.list_subsets_categories_frames_names[item]
        return self.get_frame_by_subset_category_name(subset=subset, category=category, name=frame_name)

    @property
    def path_meshes(self):
        return Pascal3D.get_path_meshes(path_raw=self.path_raw)



    """
    def get_rfpaths_frames_meta(self):
        frames_rfpaths = []
        for subset in self.subsets:
            for category in self.categories:
                frames_fpaths_partial = list(Pascal3DFrameMeta.get_path_frames_meta_with_subset_category(path_meta=self.path_meta, subset=subset, category=category).iterdir())
                frames_rfpaths_partial = [Pascal3DFrameMeta.get_rfpath_frame_meta_with_subset_category_name(subset=subset, category=category, name=fpath.stem) for fpath in frames_fpaths_partial]
                frames_rfpaths += frames_rfpaths_partial
        return frames_rfpaths
    """

