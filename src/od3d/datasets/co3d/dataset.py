from od3d.datasets.dataset import OD3D_Dataset, OD3D_FRAME_MODALITIES, OD3D_Frame
from omegaconf import DictConfig, OmegaConf
from co3d.dataset.data_types import (
    load_dataclass_jgzip, FrameAnnotation, SequenceAnnotation
)
from od3d.datasets.co3d.frame import CO3D_Frame
from od3d.datasets.co3d.sequence import CO3D_Sequence

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

from od3d.datasets.co3d.enum import CAM_TFORM_OBJ_SOURCES, CUBOID_SOURCES, CO3D_FRAME_TYPES, CO3D_FRAME_SPLITS, CO3D_CLASSES

class CO3D(OD3D_Dataset):
    def __init__(
        self,
        config: DictConfig,
        transform=None
    ):
        super().__init__(config=config, transform=transform)

        if config.get("setup", False):
            CO3D.setup(config=self.config)
        if config.get("preprocess", False):
            CO3D.preprocess(config=self.config)

        self.device = "cpu"
        self.dtype = torch.float32
        self.whitelist_frame_types = [CO3D_FRAME_TYPES.DEV_KNOWN, CO3D_FRAME_TYPES.DEV_UNSEEN,
                                      CO3D_FRAME_TYPES.TRAIN_KNOWN, CO3D_FRAME_TYPES.TRAIN_UNSEEN,
                                      CO3D_FRAME_TYPES.TEST_KNOWN]

        self.classes = self.config.get("classes", [])
        self.sequences = None

        self.config.sequence = self.config.get("sequence", {})
        self.config.frame = self.config.get("frame", {})

        self.config.blacklist_negative_depth = self.config.get("blacklist_negative_depth", False)
        self.config.frames_count_max_per_sequence = self.config.get("frames_count_max_per_sequence", -1)

        #if self.config.get("fpaths_cuboids", None) is not None:
        #    self.cuboids = Meshes.load_from_files(fpaths_meshes=[self.config.fpaths_cuboids[cls] for cls in self.classes])

        # [
        #    'car',
        #    #'carrot'
        #]

        # CO3D.preprocess(config=self.config)

        # self.preprocess_meta(remove_previous=self.config.preprocess_meta_remove_previous, override=self.config.preprocess_meta_override, sequences=self.config.get("sequences"))

        logger.info("reading sequences meta...")

        self.sequences_names = list(sorted([fpath.name.split('.')[0] for fpath in self.path_meta.iterdir() if fpath.is_file()], key= lambda n: int(n)))
        if self.config.get('sequences', None) is not None:
            self.sequences_names = list(filter(lambda sequence_name: sequence_name in self.config.sequences, self.sequences_names))

        self.sequences = []
        for sequence_name in self.sequences_names:
            sequence_config = OmegaConf.load(self.path_meta.joinpath(sequence_name + '.yaml'))
            sequence = CO3D_Sequence.create_with_config(**sequence_config, config=self.config.sequence)
            self.sequences.append(sequence)

        self.sequences_names = [seq.name for seq in self.sequences if seq.category in self.classes]
        self.sequences = list(filter(lambda seq: seq.name in self.sequences_names, self.sequences))

        if self.config.get("sequences_require_pcl", False):
            # quality score lies in range [-2.35x, 1.04x]
            self.sequences = list(filter(lambda sequence: sequence.rfpath_pcl != Path('None'), self.sequences))
            self.sequences = list(filter(lambda sequence: sequence.pcl_quality_score > self.config.get("sequences_require_pcl_score", 0.8), self.sequences))

            if self.config.get("sequences_sort_pcl_score", False):
                self.sequences = sorted(self.sequences, key=lambda sequence: -sequence.pcl_quality_score)

            self.sequences_names = [seq.name for seq in self.sequences]

        if self.config.get("sequences_count_max_per_class", None) is not None:
            self.sequences_count_per_class = {_cls: 0 for _cls in self.classes}

            self.sequences_names = []
            for seq in self.sequences:
                if self.sequences_count_per_class[seq.category] < self.config.sequences_count_max_per_class:
                    self.sequences_count_per_class[seq.category] += 1
                    self.sequences_names.append(seq.name)
            self.sequences = list(filter(lambda seq: seq.name in self.sequences_names, self.sequences))

        self.sequences_count = len(self.sequences_names)

        logger.info("reading frames names of sequences...")
        self.frames = []
        self.sequences_lengths = []
        self.frames_count = 0
        self.frames_names = []
        self.sequences_item_ids = []
        self.map_item_id_to_seq_id = []
        self.map_item_id_to_frame_id = []

        if self.config.blacklist_negative_depth:
            blacklist_negative_depth = OmegaConf.load(self.get_fpath_blacklist_negative_depth(config=self.config))
        for sequence_name in tqdm(self.sequences_names):
            seq_frames_names = sorted([fpath.name.split('.')[0] for fpath in list(self.path_meta.joinpath(sequence_name).iterdir())], key=lambda n: int(n))
            if self.config.blacklist_negative_depth:
                logger.info("filtering frames with blocklist negative depth...")
                seq_frames_names = list(filter(lambda frame_name: frame_name not in blacklist_negative_depth, seq_frames_names))
            if self.config.get("frames_count_max_per_sequence", -1) > 0:
                #seq_frames_names = seq_frames_names[:self.config.frames_count_max_per_sequence]
                seq_frames_names = [seq_frames_names[fid] for fid in np.linspace(0, len(seq_frames_names)-1, self.config.frames_count_max_per_sequence).astype(np.int).tolist()]
            self.frames_names.append(seq_frames_names)
            self.sequences_lengths.append(len(self.frames_names[-1]))
            self.sequences_item_ids.append(list(range(self.frames_count, self.frames_count + self.sequences_lengths[-1])))
            self.frames_count += self.sequences_lengths[-1]
            self.map_item_id_to_frame_id += list(range(self.sequences_lengths[-1]))
            self.map_item_id_to_seq_id += list((len(self.sequences_lengths)-1, ) * self.sequences_lengths[-1])
            #self.sequences_map_name_to_id = {}
        #self.frames_map_name_to_id = {}

        # sequence_names
    def get_subset_by_sequences(self, sequences: List[str]):
        config = self.config.copy()
        config.setup = False
        config.preprocess = False
        config.sequences = sequences
        dataset = CO3D(config=config, transform=self.transform)
        return dataset
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

    def get_item_id_by_name(self, sequence_name, frame_name):
        for i in range(len(self)):
            seq_id = self.map_item_id_to_seq_id[i]
            frame_id = self.map_item_id_to_frame_id[i]
            if self.sequences_names[seq_id] == sequence_name and self.frames_names[seq_id][frame_id] == frame_name:
                return i
        return -1
    def get_sequence_by_name(self, sequence_name):

        sequence_config = OmegaConf.load(self.path_meta.joinpath(sequence_name + '.yaml'))
        return CO3D_Sequence.create_with_config(**sequence_config, config=self.config.sequence)

    def get_frame_by_name(self, sequence_name, frame_name):
        frame_config = OmegaConf.load(self.path_meta.joinpath(sequence_name, frame_name + '.yaml'))
        frame = CO3D_Frame.create_with_config(**frame_config, config=self.config.frame)
        # frame.return_cam_tform4x4_cuboid_front = self.config.get("return_cam_tform4x4_cuboid_front", False)
        return frame

    def get_frame_by_id(self, frame_id, seq_id):
        return self.get_frame_by_name(sequence_name=self.sequences_names[seq_id], frame_name=self.frames_names[seq_id][frame_id])

    def get_sequence_by_id(self, seq_id):
        return self.get_sequence_by_name(sequence_name=self.sequences_names[seq_id])

    @staticmethod
    def preprocess(config: DictConfig):
        logger.info("preprocess")
        config = config.copy()
        config.return_cam_tform4x4_cuboid_front = False
        if config.preprocess_meta:
            CO3D.preprocess_meta(config=config)
        if config.preprocess_cuboids:
            CO3D.preprocess_cuboids(config=config)
        if config.preprocess_blacklist_negative_depth:
            CO3D.preprocess_blacklist_negative_depth(config=config)
        if config.preprocess_cuboid_avg:
            CO3D.preprocess_cuboid_avg(config=config)
        # CO3D.preprocess_cam_tform4x4_obj_canonic(config=config)
        # CO3D.preprocess_front_names(config=config)

    @staticmethod
    def get_fpath_blacklist_negative_depth(config):
        return OD3D_Dataset.get_path_preprocess(config=config).joinpath('blacklist_negative_depth.yaml')

    @staticmethod
    def preprocess_meta(config: DictConfig):
        path = Path(config.path_raw)
        path_preprocess = Path(config.path_preprocess)
        path_meta = path_preprocess.joinpath('meta')
        sequences = config.get("sequences", None)
        whitelist_frame_types = [CO3D_FRAME_TYPES.DEV_KNOWN, CO3D_FRAME_TYPES.DEV_UNSEEN,
                                 CO3D_FRAME_TYPES.TRAIN_KNOWN, CO3D_FRAME_TYPES.TRAIN_UNSEEN,
                                 CO3D_FRAME_TYPES.TEST_KNOWN]

        #if not config.preprocess_meta_override and path_meta.exists():
        #    return

        if config.preprocess_meta_remove_previous:
            if path_meta.exists():
                shutil.rmtree(path_meta)

        for cls in config.classes:
            logger.info(f'preprocess meta for class {cls}')
            sequence_annotations = load_dataclass_jgzip(
                f"{path}/{cls}/sequence_annotations.jgz", List[SequenceAnnotation]
            )
            logger.info('reading sequence annotations...')
            seq_count_per_class = 0
            read_sequences = []
            for sequence_annoation in tqdm(sequence_annotations):

                if sequences is not None and sequence_annoation.sequence_name not in sequences:
                    continue

                read_sequences.append(sequence_annoation.sequence_name)
                seq_count_per_class += 1
                if config.get("sequences_count_max_per_class", None) is not None:
                    if seq_count_per_class > config.get("sequences_count_max_per_class"):
                        break

                fpath = path_meta.joinpath(sequence_annoation.sequence_name + '.yaml')

                if fpath.exists() and not config.preprocess_meta_override:
                    continue

                sequence = CO3D_Sequence.load_from_raw(path_co3d=path, path_meta=path_meta, path_preprocess=path_preprocess, sequence_annotation=sequence_annoation)

                seq_config = OmegaConf.structured(sequence)
                if not fpath.parent.exists():
                    fpath.parent.mkdir(parents=True)
                OmegaConf.save(seq_config, fpath, resolve=True)

            cls_frame_annotations = load_dataclass_jgzip(
                f"{path}/{cls}/frame_annotations.jgz", List[FrameAnnotation]
            )
            cls_frame_annotations = [fa for fa in cls_frame_annotations if fa.meta[
                'frame_type'] in whitelist_frame_types]

            logger.info('reading frame annotations...')
            for frame_annotation in tqdm(cls_frame_annotations):
                if sequences is not None and frame_annotation.sequence_name not in sequences:
                    continue

                if frame_annotation.sequence_name not in read_sequences:
                    continue

                fpath = path_meta.joinpath(frame_annotation.sequence_name, f'{frame_annotation.frame_number}.yaml')
                if fpath.exists() and not config.preprocess_meta_override:
                    continue

                frame = CO3D_Frame.load_from_raw(path_co3d=path, path_preprocess=path_preprocess, path_meta=path_meta, frame_annotation=frame_annotation)
                frame_conf = OmegaConf.structured(frame)
                if not fpath.parent.exists():
                    fpath.parent.mkdir(parents=True)
                OmegaConf.save(frame_conf, fpath, resolve=True)

    @staticmethod
    def preprocess_cuboids(config: DictConfig):
        logger.info("preprocess cuboids")
        config = copy(config)
        config.fpaths_cuboids = None
        config.setup = False
        config.preprocess = False
        dataset = CO3D(config=config)
        for sequence_name in dataset.sequences_names:
            logger.info(f"preprocess cuboids, sequence {sequence_name}")

            sequence = dataset.get_sequence_by_name(sequence_name=sequence_name)
            sequence.preprocess_cuboid(override=config.preprocess_cuboids_override)

    @staticmethod
    def preprocess_cuboid_avg(config: DictConfig):
        logger.info("preprocess cuboids")


        for cls in config.classes:
            fpath = Path(config.path_preprocess).joinpath('cuboids', 'avg', config.name, f'{cls}.ply')

            if not fpath.exists() or config.preprocess_cuboid_avg_override:
                logger.info(f"preprocessing average cuboid and saving it to {fpath}")
                config = copy(config)
                config.fpaths_cuboids = None
                config.setup = False
                config.preprocess = False
                dataset = CO3D(config=config)
                percentile_noise = 0.03
                cuboid_pts3d_max_count = 1000
                device = 'cuda:0'
                from od3d.cv.visual.show import show_pcl
                from od3d.cv.geometry.downsample import voxel_downsampling
                from od3d.cv.geometry.transform import transf3d_broadcast
                from od3d.cv.geometry.primitives import Cuboids
                pcls = []
                for sequence_name in dataset.sequences_names:
                    sequence = dataset.get_sequence_by_name(sequence_name)
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
    @staticmethod
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

    @staticmethod
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


    @staticmethod
    def preprocess_blacklist_negative_depth(config: DictConfig):
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

    def __len__(self):
        return self.frames_count

    def get_item(self, item):
        seq_id = self.map_item_id_to_seq_id[item]
        frame_id = self.map_item_id_to_frame_id[item]
        frame = self.transform(self.get_frame_by_id(seq_id=seq_id, frame_id=frame_id))
        frame.label = self.config.classes.index(frame.category)
        return frame
    def visualize(self, item: int):
        pass


