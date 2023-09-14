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
from od3d.datasets.pascal3d.frame import Pascal3DFrame, Pascal3DFrameMeta
from od3d.datasets.pascal3d.enum import PASCAL3D_CATEGORIES, PASCAL3D_SUBSETS, PASCAL3D_SCALE_NORMALIZE_TO_REAL, MAP_CATEGORIES_OD3D_TO_PASCAL3D
from typing import Dict
import inspect

class Pascal3D(OD3D_Dataset):

    CATEGORIES = PASCAL3D_CATEGORIES
    MAP_OD3D_CATEGORIES = MAP_CATEGORIES_OD3D_TO_PASCAL3D

    def __init__(
        self,
        name: str,
        modalities: List[OD3D_FRAME_MODALITIES],
        path_raw: Path,
        path_preprocess: Path,
        path_cuboids: Path,
        categories: List[PASCAL3D_CATEGORIES] = None,
        dict_nested_frames: Dict[str, Dict[str, List[str]]] = None,
        dict_nested_frames_ban: Dict[str, Dict[str, List[str]]] = None,
        transform=None,
        subset_fraction=1.,
        index_shift=0,
    ):
        if categories is not None:
            categories = [self.MAP_OD3D_CATEGORIES[category] if category not in self.CATEGORIES.list() else category for category in categories]
        else:
            categories = self.CATEGORIES.list()

        super().__init__(categories=categories, name=name, modalities=modalities, path_raw=path_raw, path_preprocess=path_preprocess, transform=transform, subset_fraction=subset_fraction, index_shift=index_shift, dict_nested_frames=dict_nested_frames, dict_nested_frames_ban=dict_nested_frames_ban)

        self.path_cuboids = Path(path_cuboids)

    def get_subset_with_dict_nested_frames(self, dict_nested_frames):
        return Pascal3D(name=self.name, modalities=self.modalities, path_raw=self.path_raw,
                        path_preprocess=self.path_preprocess, categories=self.categories,
                        dict_nested_frames=dict_nested_frames, transform=self.transform, index_shift=self.index_shift,
                        path_cuboids=self.path_cuboids)

    @staticmethod
    def setup(config):
        path_pascal3d_raw = Path(config.path_raw)

        if path_pascal3d_raw.exists() and config.setup.remove_previous:
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

    ##### SETUP
    def filter_dict_nested_frames(self, dict_nested_frames):
        dict_nested_frames = super().filter_dict_nested_frames(dict_nested_frames)
        logger.info('filtering frames categorical...')
        dict_nested_frames_filtered = {}
        for subset, dict_category_frames in dict_nested_frames.items():
            dict_nested_frames_filtered[subset] = {}
            for category, list_frames in dict_category_frames.items():
                if category in self.categories:
                    dict_nested_frames_filtered[subset][category] = list_frames

        dict_nested_frames = dict_nested_frames_filtered
        return dict_nested_frames


    #### PREPROCESS META
    @staticmethod
    def extract_meta(config: DictConfig):
        subsets = config.get("subsets", None)
        if subsets is None:
            subsets = PASCAL3D_SUBSETS.list()
        categories = config.get("categories", None)
        if categories is None:
            categories = PASCAL3D_CATEGORIES.list()

        path_raw = Pascal3D.get_path_raw(config=config)
        path_meta = Pascal3D.get_path_meta(config=config)
        rpath_meshes = Pascal3D.get_rpath_meshes()

        if config.extract_meta.remove_previous:
            if path_meta.exists():
                shutil.rmtree(path_meta)

        frames_names = []
        frames_categories = []
        frames_subsets = []
        for subset in subsets:
            for category in categories:
                fpath_frame_names_partial = path_raw.joinpath("Image_sets", f"{category}_imagenet_{subset}.txt")
                with fpath_frame_names_partial.open() as f:
                    frame_names_partial = f.read().splitlines()
                frames_names += frame_names_partial
                frames_categories += [category] * len(frame_names_partial)
                frames_subsets += [subset] * len(frame_names_partial)

        #frames_subsets, frames_categories, frames_names = Pascal3DFrameMeta.get_frames_names_from_subsets_and_cateogories_from_raw(path_pascal3d_raw=path_raw, subsets=subsets, categories=categories)

        if config.get('frames', None) is not None:
            frames_names = list(filter(lambda f: f in config.frames, frames_names))

        for i in tqdm(range(len(frames_names))):
            fpath = path_meta.joinpath(Pascal3DFrameMeta.get_rfpath_from_name_unique(name_unique=Pascal3DFrameMeta.get_name_unique_from_category_subset_name(subset=frames_subsets[i],
                                                                                                                                                             category=frames_categories[i], name=frames_names[i])))
            if not fpath.exists() or config.extract_meta.override:

                frame_meta = Pascal3DFrameMeta.load_from_raw(frame_name=frames_names[i], subset=frames_subsets[i],
                                                             category=frames_categories[i],
                                                             path_raw=path_raw, rpath_meshes=rpath_meshes)

                if frame_meta is not None:
                    frame_meta.save(path_meta=path_meta)

    @staticmethod
    def get_rpath_meshes():
        return Path("CAD")
    @staticmethod
    def get_path_meshes(path_raw: Path):
        return path_raw.joinpath(Pascal3D.get_rpath_meshes())

    ##### PREPROCESS
    def preprocess(self, config_preprocess: DictConfig):
        logger.info("preprocess")
        for key in config_preprocess.keys():
            if key == 'cuboid' and config_preprocess.cuboid.get('enabled', False):
                override = config_preprocess.cuboid.get('override', False)
                remove_previous = config_preprocess.cuboid.get('remove_previous', False)
                self.preprocess_cuboids(override=override, remove_previous=remove_previous)
            elif key == 'mask' and config_preprocess.mask.get('enabled', False):
                override = config_preprocess.mask.get('override', False)
                remove_previous = config_preprocess.mask.get('remove_previous', False)
                self.preprocess_masks(override=override, remove_previous=remove_previous)

    def preprocess_cuboids(self, override=False, remove_previous=False):
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

    def preprocess_masks(self, override=False, remove_previous=False):
        logger.info('preprocess masks...')
        for frame_id in tqdm(range(len(self))):
            frame = self.get_item(frame_id)
            frame.preprocess_mask(override=override)

    ##### DATASET PROPERTIES

    def get_item(self, item):
        frame_meta = Pascal3DFrameMeta.load_from_meta_with_name_unique(path_meta=self.path_meta, name_unique=self.list_frames_unique[item])
        return Pascal3DFrame(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
                             path_meshes=self.path_meshes, meta=frame_meta, modalities=self.modalities,
                             categories=self.categories)

    @property
    def path_meshes(self):
        return Pascal3D.get_path_meshes(path_raw=self.path_raw)

