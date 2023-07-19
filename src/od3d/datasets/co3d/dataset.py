from od3d.datasets.dataset import OD3D_Dataset, OD3D_FRAME_MODALITIES, OD3D_Frame
from omegaconf import DictConfig, OmegaConf
from co3d.dataset.data_types import (
    load_dataclass_jgzip, FrameAnnotation, SequenceAnnotation
)
from od3d.datasets.co3d.frame import CO3D_Frame, CO3D_FrameMeta
from od3d.datasets.co3d.sequence import CO3D_Sequence, CO3D_SequenceMeta

from typing import List
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
import numpy as np
from od3d.cv.io import load_ply, save_ply

from od3d.datasets.co3d.enum import CAM_TFORM_OBJ_SOURCES, CUBOID_SOURCES, CO3D_FRAME_TYPES, CO3D_FRAME_SPLITS, CO3D_CATEGORIES

ALLOW_LIST_FRAME_TYPES = [CO3D_FRAME_TYPES.DEV_KNOWN, CO3D_FRAME_TYPES.DEV_UNSEEN,
                         CO3D_FRAME_TYPES.TRAIN_KNOWN, CO3D_FRAME_TYPES.TRAIN_UNSEEN,
                         CO3D_FRAME_TYPES.TEST_KNOWN]

class CO3D(OD3D_Dataset):

    @staticmethod
    def create_from_config(config: DictConfig, transform=None):
        if config.get("setup", False):
            CO3D.setup(config=config)
        if config.get("preprocess_meta", False):
            CO3D.preprocess_meta(config=config)

        import inspect
        keys = inspect.getfullargspec(CO3D.__init__)[0][1:]
        co3d = CO3D(**dict((key, config.get(key)) for key in keys if config.get(key, None) is not None), transform=transform)

        if config.get("preprocess", False):
            co3d.preprocess(override=config.get("preprocess_override", False),
                            preprocess_pcls=config.get("preprocess_pcls", True),
                            preprocess_cuboids=config.get("preprocess_cuboids", True),
                            preprocess_cuboid_avg=config.get("preprocess_cuboid_avg", True))

        return co3d
    def __init__(self, name: str, modalities: List[OD3D_FRAME_MODALITIES], path_raw: Path, path_preprocess: Path,
                 categories: List[CO3D_CATEGORIES]=None,
                 sequences_names: List[str]=None,
                 frames_block_negative_depth=False,
                 frames_count_max_per_sequence=None,
                 sequences_require_pcl=False,
                 sequences_sort_pcl_score=False,
                 sequences_require_pcl_score=-1000.1,
                 sequences_count_max_per_class=None,
                 cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.KPTS2D_ORIENT_AND_PCL.value,
                 cuboid_source=CUBOID_SOURCES.KPTS2D_ORIENT_AND_PCL.value,
                 transform=None, index_shift=0, subset_fraction=1.):
        super().__init__(name=name, modalities=modalities, path_raw=path_raw, path_preprocess=path_preprocess, transform=transform, index_shift=index_shift, subset_fraction=subset_fraction)

        self.device = "cpu"
        self.dtype = torch.float32
        self.categories = categories if categories is not None else CO3D_CATEGORIES.list()
        self.frames_block_negative_depth = frames_block_negative_depth
        self.frames_count_max_per_sequence = frames_count_max_per_sequence
        self.sequences_count_max_per_class = sequences_count_max_per_class
        self.sequences_require_pcl = sequences_require_pcl
        self.sequences_sort_pcl_score = sequences_sort_pcl_score
        self.sequences_require_pcl_score = sequences_require_pcl_score
        self.cam_tform_obj_source = cam_tform_obj_source
        self.cuboid_source = cuboid_source

        logger.info("reading sequences meta...")

        self.sequences_names = sequences_names if sequences_names is not None else self.get_sequences_names_meta()

        # filter sequences
        sequences_meta_rfpaths = self.get_rfpaths_sequences_meta()
        self.sequences = [self.get_sequence_by_rfpath(sequence_meta_rfpath=rfpath) for rfpath in sequences_meta_rfpaths]
        if self.sequences_require_pcl:
            # quality score lies in range [-2.35x, 1.04x]
            self.sequences = list(filter(lambda sequence: sequence.meta.rfpath_pcl != Path('None'), self.sequences))
            if self.sequences_require_pcl_score is not None:
                self.sequences = list(filter(lambda sequence: sequence.meta.pcl_quality_score > self.sequences_require_pcl_score, self.sequences))
            if self.sequences_sort_pcl_score:
                self.sequences = sorted(self.sequences, key=lambda sequence: -sequence.meta.pcl_quality_score)

        if self.sequences_count_max_per_class is not None:
            self.sequences_count_per_class = {category: 0 for category in self.categories}
            self.sequences_names = []
            for seq in self.sequences:
                if self.sequences_count_per_class[seq.category] < self.sequences_count_max_per_class:
                    self.sequences_count_per_class[seq.category] += 1
                    self.sequences_names.append(seq.name)

        self.sequences_meta_rfpaths = [seq.meta.rfpath for seq in self.sequences]
        self.sequences_names = [rfpath.stem for rfpath in self.sequences_meta_rfpaths]
        self.map_sequence_name_to_sequence_rfpath = dict(zip(self.sequences_names, self.sequences_meta_rfpaths))
        self.sequences_count = len(self.sequences_meta_rfpaths)
        logger.info(f"found {self.sequences_count} sequences.")
        logger.info("reading frames rfpaths of sequences...")

        self.frames_meta_rfpaths, self.sequences_lengths = self.get_rfpaths_frames_meta(return_sequences_lengths=True)
        self.frames_names = [rfpath.stem for rfpath in self.frames_meta_rfpaths]
        self.frames_count = len(self.frames_meta_rfpaths)

        self.map_item_id_to_seq_id = []
        self.map_item_id_to_frame_id = []
        for sequence_name in self.sequences_names:
            self.map_item_id_to_frame_id += list(range(self.sequences_lengths[-1]))
            self.map_item_id_to_seq_id += list((len(self.sequences_lengths)-1, ) * self.sequences_lengths[-1])

        if self.subset_fraction is not None and self.subset_fraction != 1.:
            frames_ids_subset = torch.multinomial(torch.ones(size=(self.frames_count,)),
                                                  num_samples=int(self.subset_fraction * self.frames_count),
                                                  replacement=False)
            self.map_item_id_to_seq_id = [self.map_item_id_to_seq_id[id] for id in frames_ids_subset]
            self.map_item_id_to_frame_id = [self.map_item_id_to_frame_id[id] for id in frames_ids_subset]
            self.frames_meta_rfpaths = [self.frames_meta_rfpaths[id] for id in frames_ids_subset]
            self.frames_names = [self.frames_names[id] for id in frames_ids_subset]
            self.frames_count = len(self.frames_meta_rfpaths)

        logger.info(f"found {self.frames_count} frames.")


    def get_rfpaths_frames_meta(self, return_sequences_lengths=False):
        sequences_lengths = []
        frames_rfpaths = []
        for category in self.categories:
            for sequence_name in self.sequences_names:
                frames_fpaths_partial = list(CO3D_FrameMeta.get_path_frames_meta_with_category_sequence(path_meta=self.path_meta, category=category, sequence=sequence_name).iterdir())
                frames_rfpaths_partial = [CO3D_FrameMeta.get_rfpath_frame_meta_with_category_sequence_name(category=category, sequence=sequence_name, name=fpath.stem) for fpath in frames_fpaths_partial]
                frames_rfpaths_partial = sorted(frames_rfpaths_partial, key=lambda rfpath: int(rfpath.stem))
                if self.frames_count_max_per_sequence is not None:
                    frames_rfpaths_partial = [frames_rfpaths_partial[fid] for fid in np.linspace(0, len(frames_rfpaths_partial)-1, self.frames_count_max_per_sequence).astype(int).tolist()]

                if self.frames_block_negative_depth:
                    frames_rfpaths_partial = list(filter(lambda rfpath: self.get_frame_by_rfpath(rfpath).cam_tform4x4_obj[0, 2, 3] >= 0.01, frames_rfpaths_partial))
                frames_rfpaths += frames_rfpaths_partial
                sequences_lengths.append(len(frames_rfpaths_partial))
        if return_sequences_lengths:
            return frames_rfpaths, sequences_lengths
        else:
            return frames_rfpaths

    def get_sequences_names_meta(self):
        sequences_names = []
        for category in self.categories:
            sequences_fpaths_partial = list(CO3D_SequenceMeta.get_path_sequences_meta_with_category(path_meta=self.path_meta, category=category).iterdir())
            sequences_names_partial = [fpath.stem for fpath in sequences_fpaths_partial]
            sequences_names += sequences_names_partial
        return sequences_names

    def get_rfpaths_sequences_meta(self):
        sequences_rfpaths = []
        for category in self.categories:
            for sequence_name in self.sequences_names:
                sequences_rfpaths.append(CO3D_SequenceMeta.get_rfpath_sequence_meta_with_category_and_name(category=category,name=sequence_name))
        return sequences_rfpaths

    def get_subset_by_sequences(self, sequences_names: List[str], frames_count_max_per_sequence=None):
        if frames_count_max_per_sequence is None:
            frames_count_max_per_sequence = self.frames_count_max_per_sequence
        return CO3D(name=self.name, modalities=self.modalities, path_raw=self.path_raw, path_preprocess=self.path_preprocess,
                 categories=self.categories,
                 sequences_names=sequences_names,
                 frames_block_negative_depth=self.frames_block_negative_depth,
                 frames_count_max_per_sequence=frames_count_max_per_sequence,
                 sequences_require_pcl=self.sequences_require_pcl,
                 sequences_sort_pcl_score=self.sequences_sort_pcl_score,
                 sequences_require_pcl_score=self.sequences_require_pcl_score,
                 sequences_count_max_per_class=self.sequences_count_max_per_class,
                 cam_tform_obj_source=self.cam_tform_obj_source,
                 cuboid_source=self.cuboid_source,
                 transform=self.transform, index_shift=self.index_shift)


    @staticmethod
    def setup(config: DictConfig):

        # logger.info(OmegaConf.to_yaml(config))
        path_co3d_raw = Path(config.path_co3d_raw)
        if path_co3d_raw.exists() and config.setup_remove_previous:
            logger.info(f"Removing previous CO3D")
            shutil.rmtree(path_co3d_raw)

        if path_co3d_raw.exists() and not config.setup_override:
            logger.info(f"Found CO3D dataset at {path_co3d_raw}")
        else:
            path_co3d_repo = path_co3d_raw.joinpath('co3d')
            path_co3d_repo.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cloning CO3D github repository to {path_co3d_repo}")
            run_cmd(cmd=f'cd {path_co3d_raw} && git clone git@github.com:facebookresearch/co3d.git', live=True, logger=logger)
            logger.info(f"Downloading CO3D dataset at {path_co3d_raw}")
            run_cmd(cmd=f'python {path_co3d_repo.joinpath("co3d/download_dataset.py")} --download_folder {path_co3d_raw}', live=True, logger=logger)
            # --n_download_workers 1 --n_extract_workers 1

    @staticmethod
    def preprocess_meta(config: DictConfig):
        path = Path(config.path_raw)
        path_meta = CO3D.get_path_meta(config=config)

        sequences_names = config.get("sequences", None)
        sequences_count_max_per_class = config.get("sequences_count_max_per_class", None)

        if config.preprocess_meta_remove_previous:
            if path_meta.exists():
                shutil.rmtree(path_meta)

        for category in config.categories:
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

                if sequence_meta_fpath.exists() and not config.preprocess_meta_override:
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
                frame_meta_fpath = CO3D_FrameMeta.get_fpath_frame_meta_with_category_sequence_name(path_meta=path_meta,
                                                                                                   category=category,
                                                                                                   sequence=sequence_name,
                                                                                                   name=frame_name)
                if frame_meta_fpath.exists() and not config.preprocess_meta_override:
                    continue


                frame_meta = CO3D_FrameMeta.load_from_raw(frame_annotation=frame_annotation)
                frame_meta.save(path_meta=path_meta)

    @staticmethod
    def get_rfpath_sequence_meta_with_category(category: str, name: str):
        return Path(category).joinpath(name + '.yaml')
    @staticmethod
    def get_path_meta_sequences(path_meta: Path):
        return path_meta.joinpath("sequences")

    def get_fpath_sequence_meta(self, rfpath_sequence_meta: Path):
        return CO3D.get_path_meta_sequences(path_meta=self.path_meta).joinpath(rfpath_sequence_meta)

    def get_item_id_by_name(self, sequence_name, frame_name):
        for i in range(len(self)):
            seq_id = self.map_item_id_to_seq_id[i]
            frame_id = self.map_item_id_to_frame_id[i]
            if self.sequences_names[seq_id] == sequence_name and self.frames_names[seq_id][frame_id] == frame_name:
                return i
        return -1

    def preprocess(self, preprocess_pcls=True, preprocess_cuboids=True, preprocess_cuboid_avg=True, override=False):
        logger.info("preprocess")
        if preprocess_pcls:
            self.preprocess_pcls(override=override)
        if preprocess_cuboids:
            self.preprocess_cuboids(override=override)
        if preprocess_cuboid_avg:
            self.preprocess_cuboid_avg(override=override)
        # CO3D.preprocess_cam_tform4x4_obj_canonic(config=config)
        # CO3D.preprocess_front_names(config=config)

    def preprocess_pcls(self, override=False):
        logger.info("preprocess pcls...")

        for sequence_name in self.sequences_names:
            logger.info(f"preprocess pcls, sequence {sequence_name}")
            sequence = self.get_sequence_by_name(sequence_name=sequence_name)
            sequence.preprocess_pcl_clean(override=override)

    def preprocess_cuboids(self, override=False):
        logger.info("preprocess cuboids...")

        for sequence_name in self.sequences_names:
            logger.info(f"preprocess cuboids, sequence {sequence_name}")
            sequence = self.get_sequence_by_name(sequence_name=sequence_name)
            sequence.preprocess_cuboid(override=override)

    def preprocess_cuboid_avg(self, override=False):
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
                pcls = []
                for sequence_name in self.sequences_names:
                    sequence = self.get_sequence_by_name(sequence_name)
                    pcls.append(transf3d_broadcast(voxel_downsampling(sequence.pcl_clean, K=cuboid_pts3d_max_count).to(device=device),
                                                   transf4x4=sequence.cuboid_front_tform4x4_obj.to(device=device)))
                    # batch[0].sequence_name
                    # dataset.visualize(i)
                pcl_max_pts_id = torch.Tensor([pcl.shape[0] for pcl in pcls]).max(dim=0)[1]
                pcls.append(pcls[0])
                pcls[0] = pcls[pcl_max_pts_id]

                cuboid_pts3d = torch.cat(pcls, dim = 0)

                cuboids_limits = torch.stack(
                    [cuboid_pts3d.quantile(dim=-2, q=percentile_noise), cuboid_pts3d.quantile(dim=-2, q=1. - percentile_noise)],
                    dim=-2)[None,]
                cuboids = Cuboids.create_dense_from_limits(limits=cuboids_limits, verts_count=cuboid_pts3d_max_count)

                fpath.parent.mkdir(parents=True, exist_ok=True)
                save_ply(fpath, verts=cuboids.verts, faces=cuboids.faces)
                # show_pcl([cuboids.verts.to(device=device)] + pcls)

    def __len__(self):
        return len(self.frames_meta_rfpaths)

    def get_frame_meta_by_rfpath(self, frame_meta_rfpath):
        return CO3D_FrameMeta.load_from_meta_with_rfpath(path_meta=self.path_meta, rfpath=frame_meta_rfpath)

    def get_frame_by_rfpath(self, frame_meta_rfpath):
        frame_meta = self.get_frame_meta_by_rfpath(frame_meta_rfpath)
        return CO3D_Frame(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
                          meta=frame_meta, modalities=self.modalities, categories=self.categories,
                          cuboid_source=self.cuboid_source, cam_tform_obj_source=self.cam_tform_obj_source)

    def get_sequence_meta_by_rfpath(self, sequence_meta_rfpath):
        return CO3D_SequenceMeta.load_from_meta_with_rfpath(path_meta=self.path_meta, rfpath=sequence_meta_rfpath)

    def get_sequence_by_rfpath(self, sequence_meta_rfpath):
        sequence_meta = self.get_sequence_meta_by_rfpath(sequence_meta_rfpath)
        return CO3D_Sequence(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
                             meta=sequence_meta, modalities=self.modalities, categories=self.categories,
                              cuboid_source=self.cuboid_source, cam_tform_obj_source=self.cam_tform_obj_source)

    def get_sequence_by_name(self, sequence_name):
        rfpath = self.map_sequence_name_to_sequence_rfpath[sequence_name]
        return self.get_sequence_by_rfpath(sequence_meta_rfpath=rfpath)

    def get_item(self, item):
        return self.get_frame_by_rfpath(frame_meta_rfpath=self.frames_meta_rfpaths[item])

    def visualize(self, item: int):
        pass


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

