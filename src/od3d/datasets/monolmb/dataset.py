import logging
logger = logging.getLogger(__name__)

import od3d.datasets.monolmb.frame # import MonoLMB_Frame, MonoLMB_FrameMeta
import od3d.datasets.monolmb.sequence # import MonoLMB_Sequence

from od3d.datasets.dataset import OD3D_Dataset, OD3D_SequenceDataset
from od3d.datasets.object import OD3D_FRAME_MASK_TYPES, OD3D_CAM_TFORM_OBJ_TYPES, OD3D_MESH_TYPES, OD3D_PCL_TYPES, \
    OD3D_SEQUENCE_SFM_TYPES, OD3D_TFROM_OBJ_TYPES
from od3d.datasets.sequence_meta import OD3D_SequenceMetaCategoryMixin

from pathlib import Path
from omegaconf import DictConfig
import shutil

class MonoLMB(OD3D_SequenceDataset):
    from od3d.datasets.monolmb.enum import MONOLMB_CATEGORIES
    all_categories = list(MONOLMB_CATEGORIES)

    def path_videos(self):
        return MonoLMB.get_path_videos(self.path_raw)

    @staticmethod
    def get_path_videos(path_raw: Path):
        return path_raw.joinpath('videos')

    def path_frames_rgb(self):
        return MonoLMB.get_path_frames_rgb(path_raw=self.path_raw)

    @staticmethod
    def get_path_frames_rgb(path_raw: Path):
        return path_raw.joinpath('frames')

    def get_frame_by_name_unique(self, name_unique):
        return od3d.datasets.monolmb.frame.MonoLMB_Frame(path_raw=self.path_raw, path_preprocess=self.path_preprocess,
                                                         name_unique=name_unique, all_categories=self.categories,
                                                         mask_type=OD3D_FRAME_MASK_TYPES.SAM_SFM_RAYS_CENTER3D,
                                                         cam_tform4x4_obj_type=OD3D_CAM_TFORM_OBJ_TYPES.SFM,
                                                         mesh_type=OD3D_MESH_TYPES.CUBOID500,
                                                         pcl_type=OD3D_PCL_TYPES.SFM_MASK,
                                                         sfm_type=OD3D_SEQUENCE_SFM_TYPES.DROID,
                                                         modalities=self.modalities,
                                                         tform_obj_type=OD3D_TFROM_OBJ_TYPES.LABEL3D,)

    def get_sequence_by_name_unique(self, name_unique):
        return od3d.datasets.monolmb.sequence.MonoLMB_Sequence(name_unique=name_unique,
                                                        path_raw=self.path_raw,
                                                        path_preprocess=self.path_preprocess,
                                                        all_categories=self.categories,
                                                        mask_type=OD3D_FRAME_MASK_TYPES.SAM_SFM_RAYS_CENTER3D,
                                                        cam_tform4x4_obj_type=OD3D_CAM_TFORM_OBJ_TYPES.SFM,
                                                        mesh_type=OD3D_MESH_TYPES.CUBOID500,
                                                        pcl_type=OD3D_PCL_TYPES.SFM_MASK,
                                                        sfm_type=OD3D_SEQUENCE_SFM_TYPES.DROID,
                                                        modalities=self.modalities,
                                                        tform_obj_type=OD3D_TFROM_OBJ_TYPES.LABEL3D,)

    @staticmethod
    def setup(config: DictConfig):
        logger.info('recording video...')

        # /misc/lmbraid19/sommerl/datasets/MonoLMB/videos/elephant/24_01_29__18_10.mp4
        #
        from od3d.cv.io import extract_frames_from_video
        fpath_video = Path('/misc/lmbraid19/sommerl/datasets/MonoLMB/videos/elephant/24_01_29__18_10.mp4')
        path_frames = Path('/misc/lmbraid19/sommerl/datasets/MonoLMB/frames/elephant/24_01_29__18_10')
        extract_frames_from_video(fpath_video, path_frames, fps=5)

    @staticmethod
    def extract_meta(config: DictConfig):
        path_meta = MonoLMB.get_path_meta(config=config)
        path_raw = Path(config.path_raw)
        path_sequences = MonoLMB.get_path_frames_rgb(path_raw=path_raw)
        config.setup.remove_previous = True
        if path_meta.exists() and config.setup.remove_previous:
            logger.info(f"Removing previous Objectron")
            shutil.rmtree(path_meta)

        path_meta.mkdir(parents=True, exist_ok=True)

        dict_nested_frames = config.get('dict_nested_frames', None)
        dict_nested_frames_banned = config.get('dict_nested_frames_ban', None)
        preprocess_meta_override = config.get('extract_meta', False).get('override', False)


        from od3d.datasets.monolmb.enum import MONOLMB_CATEGORIES
        categories = list(dict_nested_frames.keys()) if dict_nested_frames is not None else MONOLMB_CATEGORIES.list()
        sequences_count_max_per_category = config.get("sequences_count_max_per_category", None)

        for category in categories:
            logger.info(f'preprocess meta for class {category}')

            sequences_names = list(dict_nested_frames[category].keys()) if dict_nested_frames is not None and category in dict_nested_frames.keys() and dict_nested_frames[category] is not None else None
            if sequences_names is None and (dict_nested_frames is None or (category in dict_nested_frames.keys() and dict_nested_frames[category] is None)):
                if not path_sequences.joinpath(category).exists():
                    continue

                sequences_names = [fpath.stem for fpath in path_sequences.joinpath(category).iterdir()]
            if dict_nested_frames_banned is not None and category in dict_nested_frames_banned.keys() and dict_nested_frames_banned[category] is not None:
                sequences_names = list(filter(lambda seq: seq not in dict_nested_frames_banned[category].keys(), sequences_names))

            for sequence_name in sequences_names:
                fpath_sequence_meta = OD3D_SequenceMetaCategoryMixin.get_fpath_sequence_meta_with_category_and_name(
                    path_meta=path_meta, category=category, name=sequence_name)

                if fpath_sequence_meta.exists() and not preprocess_meta_override:
                    continue

                sequence_meta = OD3D_SequenceMetaCategoryMixin.load_from_raw(category=category, name=sequence_name)

                if len(list(path_sequences.joinpath(sequence_meta.name_unique).iterdir())) > 0:
                    # filtering sequences with no rgb images
                    sequence_meta.save(path_meta=path_meta)

                for fpath_frame in path_sequences.joinpath(sequence_meta.name_unique).iterdir():
                    import torch
                    from od3d.cv.io import read_image
                    H, W = read_image(fpath_frame).shape[1:]
                    l_size = torch.LongTensor([H, W]).tolist()
                    frame_meta = od3d.datasets.monolmb.frame.MonoLMB_FrameMeta.load_from_raw(
                        name=fpath_frame.stem, category=category, sequence_name=sequence_name,
                        rfpath_rgb=Path(fpath_frame.relative_to(path_raw)), l_size=l_size)


                    frame_meta.save(path_meta=path_meta)

