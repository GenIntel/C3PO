from od3d.datasets.dataset import OD3D_Dataset, OD3D_FRAME_MODALITIES, OD3D_DATASET_SPLITS
from omegaconf import DictConfig
from co3d.dataset.data_types import (
    load_dataclass_jgzip, FrameAnnotation, SequenceAnnotation
)
from od3d.datasets.co3d.frame import CO3D_Frame, CO3D_FrameMeta
from od3d.datasets.co3d.sequence import CO3D_Sequence, CO3D_SequenceMeta
import random

from typing import List, Dict, Tuple
import torch
from pathlib import Path

from enum import Enum
from od3d.io import run_cmd
from copy import copy

import logging
logger = logging.getLogger(__name__)
import shutil
from tqdm import tqdm
import torch.utils.data
from od3d.cv.io import load_ply, save_ply

from od3d.datasets.co3d.enum import CAM_TFORM_OBJ_SOURCES, CUBOID_SOURCES, CO3D_FRAME_TYPES, CO3D_FRAME_SPLITS, CO3D_CATEGORIES, MAP_CATEGORIES_OD3D_TO_CO3D

ALLOW_LIST_FRAME_TYPES = [CO3D_FRAME_TYPES.DEV_KNOWN, CO3D_FRAME_TYPES.DEV_UNSEEN,
                         CO3D_FRAME_TYPES.TRAIN_KNOWN, CO3D_FRAME_TYPES.TRAIN_UNSEEN,
                         CO3D_FRAME_TYPES.TEST_KNOWN]

class CO3D(OD3D_Dataset):

    CATEGORIES = CO3D_CATEGORIES
    MAP_OD3D_CATEGORIES = MAP_CATEGORIES_OD3D_TO_CO3D

    def __init__(self, name: str, modalities: List[OD3D_FRAME_MODALITIES], path_raw: Path, path_preprocess: Path,
                 categories: List[CO3D_CATEGORIES]=None,
                 dict_nested_frames: Dict[str, Dict[str, List[str]]]=None,
                 dict_nested_frames_ban: Dict[str, Dict[str, List[str]]]=None,
                 frames_block_negative_depth=False,
                 frames_count_max_per_sequence=None,
                 sequences_require_pcl=False,
                 sequences_sort_pcl_score=False,
                 sequences_require_pcl_score=-1000.1,
                 sequences_count_max_per_category=None,
                 cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.KPTS2D_ORIENT_AND_PCL.value,
                 cuboid_source=CUBOID_SOURCES.KPTS2D_ORIENT_AND_PCL.value,
                 transform=None, index_shift=0, subset_fraction=1.):

        if categories is not None:
            categories = [self.MAP_OD3D_CATEGORIES[category] if category not in self.CATEGORIES.list() else category for category in categories]
        else:
            categories = self.CATEGORIES.list()

        self.categories = categories
        self.path_raw = Path(path_raw)
        self.path_preprocess = Path(path_preprocess)
        self.modalities = modalities
        self.cam_tform_obj_source = cam_tform_obj_source # required for block negative depth
        self.frames_count_max_per_sequence = frames_count_max_per_sequence
        self.frames_block_negative_depth = frames_block_negative_depth
        self.cuboid_source = cuboid_source

        logger.info("filtering sequences...")
        self.dict_category_sequences_names = self.filter_dict_nested_sequences(dict_nested_frames=
                                                                               dict_nested_frames,
                                                                               require_pcl=sequences_require_pcl,
                                                                               sort_pcl_score=sequences_sort_pcl_score,
                                                                               require_pcl_score=sequences_require_pcl_score,
                                                                               count_max_per_category=sequences_count_max_per_category,
                                                                               dict_nested_frames_ban=dict_nested_frames_ban)

        logger.info(f'sequences filtered {self.dict_category_sequences_names}')

        super().__init__(categories=categories, name=name, modalities=modalities, path_raw=path_raw,
                         path_preprocess=path_preprocess, transform=transform, subset_fraction=subset_fraction,
                         index_shift=index_shift, dict_nested_frames=dict_nested_frames, dict_nested_frames_ban=dict_nested_frames_ban)

        self.splits_featured = [OD3D_DATASET_SPLITS.RANDOM, OD3D_DATASET_SPLITS.SEQUENCES_SEPARATED, OD3D_DATASET_SPLITS.SEQUENCES_SHARED]


    def get_subset_by_sequences(self, dict_category_sequences: Dict[str, List[str]], frames_count_max_per_sequence=None):
        dict_nested_frames = {}
        for cat, seqs in dict_category_sequences.items():
            dict_nested_frames[cat] = {}
            for seq in seqs:
                 dict_nested_frames[cat][seq] = None
        return CO3D(name=self.name, modalities=self.modalities, path_raw=self.path_raw,
                    path_preprocess=self.path_preprocess, categories=self.categories,
                    frames_count_max_per_sequence=frames_count_max_per_sequence,
                    dict_nested_frames=dict_nested_frames,
                    cam_tform_obj_source=self.cam_tform_obj_source,
                    cuboid_source=self.cuboid_source, transform=self.transform, index_shift=self.index_shift)

    def get_split_sequences_shared(self, fraction1: float):
        dict_category_sequence_name_frames_names_subsetA = {}
        dict_category_sequence_name_frames_names_subsetB = {}
        dict_category_sequence_name_frames_names = CO3D_FrameMeta.rollup_flattened_frames(self.list_frames_unique)
        #dict_category_sequence_name_frames_names = self.list_categories_sequences_names_frames_names_to_dict(self.list_frames_unique)
        for category, dict_sequence_name_frames_names in dict_category_sequence_name_frames_names.items():
            dict_category_sequence_name_frames_names_subsetA[category] = {}
            dict_category_sequence_name_frames_names_subsetB[category] = {}
            for sequence_name, frames_names in dict_sequence_name_frames_names.items():
                frames_names = sorted(frames_names, key=lambda fn: int(fn))
                cutoff = int(len(frames_names) * fraction1)
                dict_category_sequence_name_frames_names_subsetA[category][sequence_name] = frames_names[:cutoff]
                dict_category_sequence_name_frames_names_subsetB[category][sequence_name] = frames_names[cutoff:]

        return self.get_split_from_dicts(dict_category_sequence_name_frames_names_subsetA, dict_category_sequence_name_frames_names_subsetB)

    def get_split_sequences_separated(self, fraction1: float):
        dict_category_sequence_name_frames_names_subsetA = {}
        dict_category_sequence_name_frames_names_subsetB = {}
        dict_category_sequence_name_frames_names = CO3D_FrameMeta.rollup_flattened_frames(self.list_frames_unique)
        #dict_category_sequence_name_frames_names = self.list_categories_sequences_names_frames_names_to_dict(self.list_frames_unique)
        for category, dict_sequence_name_frames_names in dict_category_sequence_name_frames_names.items():
            seqs_names = list(dict_sequence_name_frames_names.keys())
            cutoff = int(len(seqs_names) * fraction1)
            dict_category_sequence_name_frames_names_subsetA[category] = {s: dict_sequence_name_frames_names[s] for s in seqs_names[:cutoff]}
            dict_category_sequence_name_frames_names_subsetB[category] = {s: dict_sequence_name_frames_names[s] for s in seqs_names[cutoff:]}

        return self.get_split_from_dicts(dict_category_sequence_name_frames_names_subsetA, dict_category_sequence_name_frames_names_subsetB)

    """
    def get_subset_with_item_ids(self, item_ids):
        list_categories_sequences_names_frames_names = [self.list_frames_unique[id] for id in item_ids]
        dict_category_sequence_name_frames_names = self.list_categories_sequences_names_frames_names_to_dict(list_categories_sequences_names_frames_names)
        return self.get_subset_with_dict_category_sequence_name_frames_names(dict_category_sequence_name_frames_names)

    def get_subset_with_names_unique(self, names_unique: List[str]):
        dict_category_sequence_name_frames_names = CO3D_FrameMeta.get_dict_category_sequence_name_frames_names_with_names_unique(names_unique=names_unique)
        return self.get_subset_with_dict_category_sequence_name_frames_names(
            dict_nested_frames=dict_category_sequence_name_frames_names)
    """

    def get_subset_with_dict_nested_frames(self, dict_nested_frames):
        return CO3D(name=self.name, modalities=self.modalities, path_raw=self.path_raw,
                    path_preprocess=self.path_preprocess, categories=self.categories,
                    dict_nested_frames=dict_nested_frames,
                    cam_tform_obj_source=self.cam_tform_obj_source,
                    cuboid_source=self.cuboid_source, transform=self.transform, index_shift=self.index_shift)

    def get_split_from_dicts(self, dict_nested_frames_subsetA, dict_nested_frames_subsetB):
        co3d_subsetA = CO3D(name=self.name, modalities=self.modalities, path_raw=self.path_raw,
                            path_preprocess=self.path_preprocess, categories=self.categories,
                            dict_nested_frames=dict_nested_frames_subsetA,
                            cam_tform_obj_source=self.cam_tform_obj_source,
                            cuboid_source=self.cuboid_source, transform=self.transform, index_shift=self.index_shift)

        co3d_subsetB = CO3D(name=self.name, modalities=self.modalities, path_raw=self.path_raw,
                            path_preprocess=self.path_preprocess, categories=self.categories,
                            dict_nested_frames=dict_nested_frames_subsetB,
                            cam_tform_obj_source=self.cam_tform_obj_source,
                            cuboid_source=self.cuboid_source, transform=self.transform, index_shift=self.index_shift)

        return co3d_subsetA, co3d_subsetB

    def get_item(self, item):
        frame_meta = CO3D_FrameMeta.load_from_meta_with_name_unique(path_meta=self.path_meta, name_unique=self.list_frames_unique[item])
        return CO3D_Frame(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta, meta=frame_meta, modalities=self.modalities, categories=self.categories, cam_tform_obj_source=self.cam_tform_obj_source, cuboid_source=self.cuboid_source)


    """
    def get_item(self, item):
        category, sequence_name, frame_name = self.list_categories_sequences_names_frames_names[item]
        return self.get_frame_by_category_sequence_and_frame_name(category=category,
                                                                  sequence_name=sequence_name,
                                                                  frame_name=frame_name)
    """

    def filter_dict_nested_frames(self, dict_nested_frames: Dict[str, Dict[str, List[str]]]):
        dict_nested_frames = super().filter_dict_nested_frames(dict_nested_frames=dict_nested_frames)

        frames_count_max_per_sequence = self.frames_count_max_per_sequence
        block_negative_depth = self.frames_block_negative_depth

        # filter frames to exist in sequences:
        dict_nested_frames_filtered = {}
        for category, sequences in self.dict_category_sequences_names.items():
            dict_nested_frames_filtered[category] = {}
            for sequence in sequences:
                dict_nested_frames_filtered[category][sequence] = dict_nested_frames[category][sequence]
        dict_nested_frames = dict_nested_frames_filtered

        #if frames_count_max_per_sequence is not None or block_negative_depth:
        #    dict_nested_frames = CO3D_FrameMeta.complete_nested_metas(path_meta=self.path_meta, dict_nested_metas=dict_nested_frames)

        if frames_count_max_per_sequence is not None:
            dict_nested_frames_filtered = {}
            for category, dict_sequence_name_frames_names in dict_nested_frames.items():
                for sequence_name, frames_names in dict_sequence_name_frames_names.items():
                    frames_names_filtered = CO3D_FrameMeta.get_subset_frames_names_uniform(frames_names, count_max_per_sequence=frames_count_max_per_sequence)
                    if category not in dict_nested_frames_filtered.keys():
                        dict_nested_frames_filtered[category] = {}
                    dict_nested_frames_filtered[category][sequence_name] = frames_names_filtered
            dict_nested_frames = dict_nested_frames_filtered

        if block_negative_depth:
            dict_nested_frames_filtered = {}
            for category, dict_sequence_name_frames_names in dict_nested_frames.items():
                for sequence_name, frames_names in dict_sequence_name_frames_names.items():
                    frames: List[CO3D_Frame] = [
                        self.get_frame_by_category_sequence_and_frame_name(category=category,
                                                                           sequence_name=sequence_name,
                                                                           frame_name=frame_name)
                        for frame_name in frames_names]
                    frames = list(filter(lambda frame: frame.cam_tform4x4_obj[2, 3] >= 0.01, frames))
                    dict_nested_frames_filtered[category][sequence_name] = [frame.name for frame in frames]
        return dict_nested_frames

    def visualize(self, item: int):
        pass

    def get_frame_by_category_sequence_and_frame_name(self, category, sequence_name, frame_name):
        frame_meta = CO3D_FrameMeta.load_from_meta_with_category_sequence_and_frame_name(
            path_meta=self.path_meta,
            category=category,
            sequence_name=sequence_name,
            frame_name=frame_name)
        return self.get_frame_by_meta(frame_meta=frame_meta)

    def get_frame_by_name_unique(self, name_unique: str):
        frame_meta = CO3D_FrameMeta.load_from_meta_with_name_unique(
            path_meta=self.path_meta,
            name_unique=name_unique)
        return self.get_frame_by_meta(frame_meta=frame_meta)

    def get_frame_by_meta(self, frame_meta: CO3D_FrameMeta):
        return CO3D_Frame(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
                          meta=frame_meta, modalities=self.modalities, categories=self.categories,
                          cuboid_source=self.cuboid_source, cam_tform_obj_source=self.cam_tform_obj_source)

    def get_sequence_by_category_and_name(self, category, name):
        sequence_meta = CO3D_SequenceMeta.load_from_meta_with_category_and_name(path_meta=self.path_meta, category=category, name=name)
        return CO3D_Sequence(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
                             meta=sequence_meta, modalities=self.modalities, categories=self.categories,
                              cuboid_source=self.cuboid_source, cam_tform_obj_source=self.cam_tform_obj_source)

    def filter_dict_nested_sequences(self, dict_nested_frames: Dict[str, Dict[str, List[str]]], require_pcl, sort_pcl_score, require_pcl_score, count_max_per_category, dict_nested_frames_ban: Dict[str, Dict[str, List[str]]]=None):
        logger.info("filtering frames...")
        if dict_nested_frames is not None:
            dict_nested_sequences = {}
            for category, dict_sequence_frames in dict_nested_frames.items():
                if category not in self.categories:
                    continue
                dict_nested_sequences[category] = []

                if dict_sequence_frames is not None:
                    for sequence, frames in dict_sequence_frames.items():
                        dict_nested_sequences[category].append(sequence)
                else:
                    dict_nested_sequences[category] = None
        else:
            dict_nested_sequences = None

        dict_nested_sequences_ban = None
        if dict_nested_frames_ban is not None:
            for category, dict_sequence_frames in dict_nested_frames_ban.items():
                if dict_sequence_frames is not None:
                    for sequence, frames in dict_sequence_frames.items():
                        if frames is None:
                            if dict_nested_sequences_ban is None:
                                dict_nested_sequences_ban = {}
                            if category not in dict_nested_sequences_ban.keys():
                                dict_nested_sequences_ban[category] = []
                            dict_nested_sequences_ban[category].append(sequence)
                else:
                    if dict_nested_sequences_ban is None:
                        dict_nested_sequences_ban = {}
                    dict_nested_sequences_ban[category] = None

        # get sequences
        dict_nested_sequences = CO3D_SequenceMeta.complete_nested_metas(path_meta=self.path_meta, dict_nested_metas=dict_nested_sequences, dict_nested_metas_ban=dict_nested_sequences_ban)

        # filter dict_nested_sequences
        for i, category in tqdm(enumerate(dict_nested_sequences.keys())):
            if category not in self.categories:
                dict_nested_sequences[category] = []
                continue
            if require_pcl or count_max_per_category is not None:
                sequences = [self.get_sequence_by_category_and_name(category=category, name=sequence_name) for sequence_name
                             in dict_nested_sequences[category]]
                #if dict_nested_sequences_ban is not None and category in dict_nested_sequences_ban.keys():
                #    sequences = [seq for seq in sequences if seq not in dict_nested_sequences_ban[category]]
                if require_pcl:
                    sequences = list(filter(lambda sequence: sequence.meta.rfpath_pcl != Path('None'), sequences))
                    if require_pcl_score is not None:
                        sequences = list(
                            filter(lambda sequence: sequence.meta.pcl_quality_score > require_pcl_score, sequences))
                    if sort_pcl_score:
                        sequences = sorted(sequences, key=lambda sequence: -sequence.meta.pcl_quality_score)
                if count_max_per_category is not None:
                    sequences = sequences[:count_max_per_category]
                dict_nested_sequences[category] = [sequence.name for sequence in sequences]
        return dict_nested_sequences

    @staticmethod
    def setup(config: DictConfig):

        # logger.info(OmegaConf.to_yaml(config))
        path_raw = Path(config.path_raw)
        if path_raw.exists() and config.setup.remove_previous:
            logger.info(f"Removing previous CO3D")
            shutil.rmtree(path_raw)

        if path_raw.exists() and not config.setup.override:
            logger.info(f"Found CO3D dataset at {path_raw}")
        else:
            path_co3d_repo = path_raw.joinpath('co3d')
            path_co3d_repo.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cloning CO3D github repository to {path_co3d_repo}")
            run_cmd(cmd=f'cd {path_raw} && git clone git@github.com:facebookresearch/co3d.git', live=True, logger=logger)
            logger.info(f"Downloading CO3D dataset at {path_raw}")
            run_cmd(cmd=f'python {path_co3d_repo.joinpath("co3d/download_dataset.py")} --download_folder {path_raw}', live=True, logger=logger)
            # --n_download_workers 1 --n_extract_workers 1

    @staticmethod
    def extract_meta(config: DictConfig):
        path = Path(config.path_raw)
        path_meta = CO3D.get_path_meta(config=config)

        dict_nested_frames = config.get('dict_nested_frames', None)
        dict_nested_frames_banned = config.get('dict_nested_frames_ban', None)
        preprocess_meta_override = config.get('extract_meta', False).get('override', False)
        preprocess_meta_remove_previous = config.get('extract_meta', False).get('remove_previous', False)

        categories = list(dict_nested_frames.keys()) if dict_nested_frames is not None else CO3D_CATEGORIES.list()
        sequences_count_max_per_class = config.get("sequences_count_max_per_class", None)

        if preprocess_meta_remove_previous:
            if path_meta.exists():
                shutil.rmtree(path_meta)

        for category in categories:
            sequences_names = list(dict_nested_frames[category].keys()) if dict_nested_frames is not None and dict_nested_frames[category] is not None else None
            if dict_nested_frames_banned is not None and category in dict_nested_frames_banned.keys() and dict_nested_frames_banned[category] is not None:
                sequences_names = list(filter(lambda seq: seq not in dict_nested_frames_banned[category].keys(), sequences_names))
            logger.info(f'preprocess meta for class {category}')
            sequence_annotations = load_dataclass_jgzip(
                f"{path}/{category}/sequence_annotations.jgz", List[SequenceAnnotation]
            )
            logger.info('reading sequence annotations...')
            seq_count_per_class = 0
            read_sequences = []
            for sequence_annoation in tqdm(sequence_annotations):
                if sequences_names is not None and sequence_annoation.sequence_name not in sequences_names:
                    continue

                read_sequences.append(sequence_annoation.sequence_name)
                seq_count_per_class += 1
                if sequences_count_max_per_class is not None:
                    if seq_count_per_class > sequences_count_max_per_class:
                        break

                sequence_name = str(sequence_annoation.sequence_name)
                sequence_meta_fpath = CO3D_SequenceMeta.get_fpath_sequence_meta_with_category_and_name(path_meta=path_meta,
                                                                                                       category=category,
                                                                                                       name=sequence_name)

                if sequence_meta_fpath.exists() and not preprocess_meta_override:
                    continue

                sequence_meta = CO3D_SequenceMeta.load_from_raw(sequence_annotation=sequence_annoation)
                sequence_meta.save(path_meta=path_meta)

            cls_frame_annotations = load_dataclass_jgzip(
                f"{path}/{category}/frame_annotations.jgz", List[FrameAnnotation]
            )
            cls_frame_annotations = [fa for fa in cls_frame_annotations if fa.meta[
                'frame_type'] in ALLOW_LIST_FRAME_TYPES]

            logger.info('reading frame annotations...')
            for frame_annotation in tqdm(cls_frame_annotations):
                if sequences_names is not None and frame_annotation.sequence_name not in sequences_names:
                    continue

                if frame_annotation.sequence_name not in read_sequences:
                    continue

                frame_name = str(frame_annotation.frame_number)
                sequence_name = str(frame_annotation.sequence_name)
                frame_meta_fpath = CO3D_FrameMeta.get_fpath_frame_meta_with_category_sequence_and_frame_name(path_meta=path_meta,
                                                                                                             category=category,
                                                                                                             sequence_name=sequence_name,
                                                                                                             name=frame_name)
                if frame_meta_fpath.exists() and not preprocess_meta_override:
                    continue


                frame_meta = CO3D_FrameMeta.load_from_raw(frame_annotation=frame_annotation)
                frame_meta.save(path_meta=path_meta)



    def preprocess(self, config_preprocess: DictConfig):
        logger.info("preprocess")
        for key in config_preprocess.keys():
            if key == 'pcl' and config_preprocess.pcl.get('enabled', False):
                override = config_preprocess.pcl.get('override', False)
                remove_previous = config_preprocess.pcl.get('remove_previous', False)
                self.preprocess_pcls(override=override, remove_previous=remove_previous)
            elif key == 'cuboid' and config_preprocess.cuboid.get('enabled', False):
                override = config_preprocess.cuboid.get('override', False)
                remove_previous = config_preprocess.cuboid.get('remove_previous', False)
                self.preprocess_cuboids(override=override, remove_previous=remove_previous)
            elif key == 'cuboid_avg' and config_preprocess.cuboid_avg.get('enabled', False):
                override = config_preprocess.cuboid_avg.get('override', False)
                remove_previous = config_preprocess.cuboid_avg.get('remove_previous', False)
                self.preprocess_cuboid_avg(override=override, remove_previous=remove_previous)

        # CO3D.preprocess_cam_tform4x4_obj_canonic(config=config)
        # CO3D.preprocess_front_names(config=config)

    def preprocess_pcls(self, override=False, remove_previous=False):
        logger.info("preprocess pcls...")

        for category, sequences_names in self.dict_category_sequences_names.items():
            for sequence_name in sequences_names:
                logger.info(f"preprocess pcls, sequence {sequence_name}")
                sequence = self.get_sequence_by_category_and_name(category=category, name=sequence_name)
                sequence.preprocess_pcl_clean(override=override)

    def preprocess_cuboids(self, override=False, remove_previous=False):
        logger.info("preprocess cuboids...")

        for category, sequences_names in self.dict_category_sequences_names.items():
            for sequence_name in sequences_names:
                logger.info(f"preprocess cuboids, sequence {sequence_name}")
                sequence = self.get_sequence_by_category_and_name(category=category, name=sequence_name)
                sequence.preprocess_cuboid(override=override)

    def preprocess_cuboid_avg(self, override=False, remove_previous=False):
        logger.info("preprocess cuboids avg...")
        for category in self.categories:
            fpath = Path(self.path_preprocess).joinpath('cuboids', 'avg', self.name, f'{category}.ply')

            if not fpath.exists() or override:
                logger.info(f"preprocessing average cuboid and saving it to {fpath}")
                percentile_noise = 0.03
                cuboid_pts3d_max_count = 1000
                device = 'cuda:0'
                from od3d.cv.visual.show import show_pcl
                from od3d.cv.geometry.downsample import voxel_downsampling
                from od3d.cv.geometry.transform import transf3d_broadcast
                from od3d.cv.geometry.primitives import Cuboids

                cuboids_limits =  []
                pcls = []
                for sequence_name in self.dict_category_sequences_names[category]:

                    sequence = self.get_sequence_by_category_and_name(category=category, name=sequence_name)
                    cuboids_limits.append(sequence.cuboid.get_limits())

                    #pcls.append(transf3d_broadcast(voxel_downsampling(sequence.pcl_clean, K=cuboid_pts3d_max_count).to(device=device),
                    #                               transf4x4=sequence.cuboid_front_tform4x4_obj.to(device=device)))
                #pcl_max_pts_id = torch.Tensor([pcl.shape[0] for pcl in pcls]).max(dim=0)[1]
                #pcls.append(pcls[0])
                #pcls[0] = pcls[pcl_max_pts_id]

                #cuboid_pts3d = torch.cat(pcls, dim=0)
                #cuboids_limits = torch.stack(
                #    [cuboid_pts3d.quantile(dim=-2, q=percentile_noise), cuboid_pts3d.quantile(dim=-2, q=1. - percentile_noise)],
                #    dim=-2)[None,]

                cuboids_limits = torch.cat(cuboids_limits, dim=0)

                #cuboids_limits = torch.stack([cuboids_limits[:, 0].quantile(dim=0, q=percentile_noise), cuboids_limits[:, 1].quantile(dim=0, q=1. - percentile_noise)], dim=0)

                cuboids_limits = torch.stack([cuboids_limits[:, 0].mean(dim=0), cuboids_limits[:, 1].mean(dim=0)], dim=0)

                cuboids = Cuboids.create_dense_from_limits(limits=cuboids_limits[None,], verts_count=cuboid_pts3d_max_count)

                fpath.parent.mkdir(parents=True, exist_ok=True)
                save_ply(fpath, verts=cuboids.verts, faces=cuboids.faces)
                # show_pcl([cuboids.verts.to(device=device)] + pcls)

    """
    @staticmethod
    def get_rfpath_sequence_meta_with_category(category: str, name: str):
        return Path(category).joinpath(name + '.yaml')
    @staticmethod
    def get_path_meta_sequences(path_meta: Path):
        return path_meta.joinpath("sequences")

    def get_fpath_sequence_meta(self, rfpath_sequence_meta: Path):
        return CO3D.get_path_meta_sequences(path_meta=self.path_meta).joinpath(rfpath_sequence_meta)

    """

    """
    def get_item_id_by_name(self, sequence_name, frame_name):
        for i in range(len(self)):
            seq_id = self.map_item_id_to_seq_id[i]
            frame_id = self.map_item_id_to_frame_id[i]
            if self.sequences_names[seq_id] == sequence_name and self.frames_names[seq_id][frame_id] == frame_name:
                return i
        return -1
    """

    """
    sequences = [self.get_sequence_by_rfpath(sequence_meta_rfpath=rfpath) for rfpath in sequences_meta_rfpaths]
    if require_pcl:
        # quality score lies in range [-2.35x, 1.04x]
        sequences = list(filter(lambda sequence: sequence.meta.rfpath_pcl != Path('None'), sequences))
        if require_pcl_score is not None:
            sequences = list(
                filter(lambda sequence: sequence.meta.pcl_quality_score > require_pcl_score, sequences))
        if sort_pcl_score:
            sequences = sorted(sequences, key=lambda sequence: -sequence.meta.pcl_quality_score)


    sequences_meta_rfpaths = [seq.meta.rfpath for seq in sequences]
    return sequences_meta_rfpaths
    """

    """
    def get_sequences_names_meta(self):
        sequences_names = []
        for category in self.categories:
            sequences_fpaths_partial = list(CO3D_SequenceMeta.get_path_sequences_meta_with_category(path_meta=self.path_meta, category=category).iterdir())
            sequences_names_partial = [fpath.stem for fpath in sequences_fpaths_partial]
            sequences_names += sequences_names_partial
        return sequences_names


    def get_subset_by_sequences(self, map_category_sequences_names: DictConfig, frames_count_max_per_sequence=None):
        sequences_meta_rfpaths = CO3D_SequenceMeta.get_rfpaths_sequences_meta(categories=self.categories, map_category_sequences_names=map_category_sequences_names)
        frames_meta_rfpaths = CO3D_FrameMeta.get_rfpaths_frames_meta(path_meta=self.path_meta, )
        if frames_count_max_per_sequence is None:
            sequences_meta_rfpaths = list(filter(lambda rfpath:  CO3D_SequenceMeta.meta_rfpath_to_name(rfpath) in map_category_sequences_names[CO3D_SequenceMeta.meta_rfpath_to_category(rfpath)] ))
            self.frames_meta_rfpaths =
            frames_count_max_per_sequence = self.frames_count_max_per_sequence
        return CO3D(name=self.name, modalities=self.modalities, path_raw=self.path_raw, path_preprocess=self.path_preprocess,
                 categories=self.categories,
                 frames_meta_rfpaths=self.frames_meta_rfpaths,
                 map_category_sequences_names=s
                 cam_tform_obj_source=self.cam_tform_obj_source,
                 cuboid_source=self.cuboid_source,
                 transform=self.transform, index_shift=self.index_shift)
    """

    """
    def get_frame_meta_by_rfpath(self, frame_meta_rfpath):
        return CO3D_FrameMeta.load_from_meta_with_rfpath(path_meta=self.path_meta, rfpath=frame_meta_rfpath)

    def get_frame_by_rfpath(self, frame_meta_rfpath):
        frame_meta = self.get_frame_meta_by_rfpath(frame_meta_rfpath)
        return CO3D_Frame(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
                          meta=frame_meta, modalities=self.modalities, categories=self.categories,
                          cuboid_source=self.cuboid_source, cam_tform_obj_source=self.cam_tform_obj_source)
    """


    #def get_sequence_meta_by_rfpath(self, sequence_meta_rfpath):
    #    return CO3D_SequenceMeta.load_from_meta_with_rfpath(path_meta=self.path_meta, rfpath=sequence_meta_rfpath)


    """
    def get_sequence_by_rfpath(self, sequence_meta_rfpath):
        sequence_meta = self.get_sequence_meta_by_rfpath(sequence_meta_rfpath)
        return CO3D_Sequence(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
                             meta=sequence_meta, modalities=self.modalities, categories=self.categories,
                              cuboid_source=self.cuboid_source, cam_tform_obj_source=self.cam_tform_obj_source)

    def get_sequence_by_name(self, sequence_name):
        rfpath = self.map_sequence_name_to_sequence_rfpath[sequence_name]
        return self.get_sequence_by_rfpath(sequence_meta_rfpath=rfpath)
    """



    """
    def preprocess_front_names(config: DictConfig):
        logger.info("preprocess front_names")
        config = copy(config)
        config.fpaths_cuboids = None
        config.setup = False
        config.preprocess = False
        dataset = CO3D(config=config)
        for sequence_name in dataset.sequences_names:
            logger.info(f"preprocess front_names, sequence {sequence_name}")
            sequence = dataset.get_sequence_by_name(sequence_name=sequence_name)
            sequence.preprocess_front_name(override=config.preprocess_front_names_override)


    def preprocess_cam_tform4x4_obj_canonic(config: DictConfig):
        logger.info("preprocess cam_tform4x4_obj_canonic")
        config = copy(config)
        config.fpaths_cuboids = None
        config.setup = False
        config.preprocess = False
        dataset = CO3D(config=config)
        for sequence_name in dataset.sequences_names:
            logger.info(f"preprocess cam_tform4x4_obj_canonic, sequence {sequence_name}")
            sequence = dataset.get_sequence_by_name(sequence_name=sequence_name)
            sequence.preprocess_cam_tform4x4_obj_canonic(override=config.preprocess_cam_tform4x4_obj_canonic_override)
    """

    """
    def preprocess_blacklist_negative_depth(self, override=False):
        logger.info("preprocess blacklist negative depth")
        config = copy(config)
        config.fpaths_cuboids = None
        config.setup = False
        config.preprocess = False
        dataset = CO3D(config=config)
        blacklist_negative_depth_fpath = CO3D.get_fpath_blacklist_negative_depth(config=config)
        if not blacklist_negative_depth_fpath.exists():
            blacklist_negative_depth = OmegaConf.create()
        else:
            blacklist_negative_depth = OmegaConf.load(blacklist_negative_depth_fpath)

        for sequence_name in dataset.sequences_names:
            logger.info(f"preprocess blacklist negative depth, sequence {sequence_name}")
            dataset_seq = dataset.get_subset_by_sequences([sequence_name])
            dataloader = torch.utils.data.DataLoader(dataset=dataset_seq, batch_size=1, shuffle=False,
                                                     collate_fn=dataset.collate_fn)
            for batch in iter(dataloader):
                if batch.cam_tform4x4_obj[0, 2, 3] < 0.01:
                    if sequence_name not in blacklist_negative_depth.keys():
                        blacklist_negative_depth[sequence_name] = []
                    blacklist_negative_depth[sequence_name].append(batch.name[0])
        OmegaConf.save(blacklist_negative_depth, blacklist_negative_depth_fpath, resolve=True)
    """

