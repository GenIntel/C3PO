import logging
logger = logging.getLogger(__name__)

import od3d.datasets.monolmb.frame # import MonoLMB_Frame, MonoLMB_FrameMeta
import od3d.datasets.monolmb.sequence # import MonoLMB_Sequence

from od3d.datasets.dataset import OD3D_Dataset
from od3d.datasets.object import OD3D_FRAME_MASK_TYPES, OD3D_CAM_TFORM_OBJ_TYPES
from od3d.datasets.frame import OD3D_FRAME_MODALITIES, OD3D_Frame
from od3d.datasets.sequence_meta import OD3D_SequenceMetaCategoryMixin

#from od3d.datasets.objectnet3d.enum import OBJECTNET3D_CATEOGORIES
from pathlib import Path
from typing import List, Dict
from omegaconf import DictConfig
import shutil
import numpy as np
import torch


#OD3D_FrameMetaCamTform4x4ObjMixin, OD3D_FrameMetaCamIntr4x4Mixin,
#                     OD3D_FrameMetaCategoryMixin, OD3D_FrameMetaSequenceMixin, OD3D_FrameMetaDepthMaskMixin,
#                     OD3D_FrameMetaDepthMixin, OD3D_FrameMetaMaskMixin, OD3D_FrameMetaRGBMixin,
#                     OD3D_FrameMetaSizeMixin, OD3D_FrameMeta

from dataclasses import dataclass


class MonoLMB(OD3D_Dataset):
    def __init__(self, name: str, modalities: List[OD3D_FRAME_MODALITIES], path_raw: Path, path_preprocess: Path,
                 categories: List=None,
                 dict_nested_frames: Dict=None,
                 dict_nested_frames_ban: Dict=None,
                 transform=None, index_shift=0, subset_fraction=1.):

        from od3d.datasets.monolmb.enum import MONOLMB_CATEGORIES
        categories = categories if categories is not None else MONOLMB_CATEGORIES.list() # TODO: OBJECTNET3D_CATEOGORIES.list()
        self.categories = categories
        self.path_raw = Path(path_raw)
        self.path_preprocess = Path(path_preprocess)
        self.modalities = modalities


        logger.info("filtering sequences...")
        self.dict_category_sequences_names = self.filter_dict_nested_sequences(dict_nested_frames=dict_nested_frames,
                                                                               dict_nested_frames_ban=dict_nested_frames_ban)

        logger.info(f'sequences filtered')
        sequences_filtered_str = '\n'
        for category in self.dict_category_sequences_names.keys():
            if len(self.dict_category_sequences_names[category]) > 0:
                sequences_filtered_str += category + ': \n'
                for sequence_name in self.dict_category_sequences_names[category]:
                    sequences_filtered_str += f"  '{sequence_name}':\n"
        logger.info(sequences_filtered_str)

        logger.info(f'Found sequences per category')

        dict_nested_frames_seqs_filtered = {}
        for category in self.dict_category_sequences_names.keys(): #.keys():
            logger.info(f'{category}: {len(self.dict_category_sequences_names[category])}')
            if len(self.dict_category_sequences_names[category]) == 0:
                continue
            if dict_nested_frames is not None and category in dict_nested_frames.keys():
                dict_nested_frames_seqs_filtered[category] = dict_nested_frames[category]
            else:
                if dict_nested_frames is None:
                    dict_nested_frames_seqs_filtered[category] = None
                else:
                    # category not in dict_nested_frames
                    dict_nested_frames_seqs_filtered[category] = {}

            for sequence_name in self.dict_category_sequences_names[category]:
                if dict_nested_frames is not None and category in dict_nested_frames.keys() and dict_nested_frames[category] is not None and sequence_name in dict_nested_frames[category]:
                    dict_nested_frames_seqs_filtered[category][sequence_name] = dict_nested_frames[category][sequence_name]
                else:
                    if dict_nested_frames is None or (category in dict_nested_frames and dict_nested_frames_seqs_filtered[category] is None):
                        if not isinstance(dict_nested_frames_seqs_filtered[category], dict):
                            dict_nested_frames_seqs_filtered[category] = {}
                        dict_nested_frames_seqs_filtered[category][sequence_name] = None
                    else:
                        # category / sequence not in dict_nested_frames
                        dict_nested_frames_seqs_filtered[category][sequence_name] = []
        dict_nested_frames = dict_nested_frames_seqs_filtered
        super().__init__(categories=categories, dict_nested_frames=dict_nested_frames, dict_nested_frames_ban=dict_nested_frames_ban, name=name, modalities=modalities, path_raw=path_raw,
                         path_preprocess=path_preprocess, transform=transform, index_shift=index_shift,
                         subset_fraction=subset_fraction)

    def filter_dict_nested_sequences(self, dict_nested_frames: Dict[str, Dict[str, List[str]]], dict_nested_frames_ban: Dict[str, Dict[str, List[str]]]=None):
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
        dict_nested_sequences = OD3D_SequenceMetaCategoryMixin.complete_nested_metas(path_meta=self.path_meta, dict_nested_metas=dict_nested_sequences, dict_nested_metas_ban=dict_nested_sequences_ban)

        return dict_nested_sequences

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

    def get_subset_with_dict_nested_frames(self, dict_nested_frames):
        return MonoLMB(name=self.name, modalities=self.modalities, path_raw=self.path_raw,
                       path_preprocess=self.path_preprocess, categories=self.categories,
                       dict_nested_frames=dict_nested_frames, transform=self.transform, index_shift=self.index_shift)

    def preprocess(self, config_preprocess: DictConfig):
        logger.info("preprocess")
        for key in config_preprocess.keys():
            if key == 'sfm' and config_preprocess.sfm.get('enabled', False):
                override = config_preprocess.sfm.get('override', False)
                self.preprocess_sfm(override=override)
            if key == 'mask' and config_preprocess.mask.get('enabled', False):
                override = config_preprocess.mask.get('override', False)
                self.preprocess_mask(override=override)

    # def get_sequence_by_category_and_name(self, category: str, name: str):
    #     sequence_meta = OD3D_CategoricalSequenceMeta.load_from_meta_with_category_and_name(path_meta=self.path_meta,
    #                                                                                        category=category, name=name)
    #     return MonoLMB_Sequence(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
    #                             meta=sequence_meta, modalities=self.modalities, categories=self.categories)


    # def get_frame_by_category_sequence_and_frame_name(self, category, sequence_name, frame_name):
    #     frame_meta = MonoLMB_FrameMeta.load_from_meta_with_category_sequence_and_frame_name(
    #         path_meta=self.path_meta,
    #         category=category,
    #         sequence_name=sequence_name,
    #         frame_name=frame_name)
    #     return self.get_frame_by_meta(frame_meta=frame_meta)
    #
    # def get_frame_by_meta(self, frame_meta: MonoLMB_FrameMeta):
    #     return MonoLMB_Frame(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
    #                       meta=frame_meta, modalities=self.modalities,
    #                       categories=self.categories)


    def preprocess_mask(self, override=False, remove_previous=False):
        logger.info("preprocess masks...")

        dataloader = torch.utils.data.DataLoader(dataset=self, batch_size=1, shuffle=False,
                                                 collate_fn=self.collate_fn)
        logging.info(f"Dataset contains {len(self)} frames.")

        from od3d.models.model import OD3D_Model
        model = OD3D_Model.create_by_name('sam')
        model.cuda()
        model.eval()
        self.transform = model.transform

        for batch in iter(dataloader):
            logger.info(f'{batch.name_unique[0]}')  # sequence_name[0]}')
            if torch.cuda.is_available():
                batch.to(device='cuda:0')
                # batch.cam_proj4x4_obj batch.rays_center3d
                from od3d.cv.geometry.transform import proj3d2d_broadcast
                center_pxl2d = proj3d2d_broadcast(proj4x4=batch.cam_proj4x4_obj, pts3d=batch.rays_center3d)
                frames = [self.get_frame_by_name_unique(name_unique=name_unique) for name_unique in batch.name_unique]
                masks, scores, logits = model(batch.rgb, center_pxl2d)

                for b in range(len(batch.name_unique)):
                    masks_b = masks[b]
                    frame = frames[b]
                    sam_lvl = scores[b].argmax()
                    mask = masks[b, sam_lvl:sam_lvl+1]
                    # from od3d.cv.visual.draw import draw_pixels
                    # mask = draw_pixels(mask, pxls=center_pxl2d[b:b+1])
                    frame.write_mask(mask)

    def get_frame_by_name_unique(self, name_unique):
        return od3d.datasets.monolmb.frame.MonoLMB_Frame(path_raw=self.path_raw, path_preprocess=self.path_preprocess,
                                                         name_unique=name_unique, all_categories=self.categories,
                                                         mask_type=OD3D_FRAME_MASK_TYPES.SAM,
                                                         cam_tform4x4_obj_type=OD3D_CAM_TFORM_OBJ_TYPES.SFM)

    def preprocess_sfm(self, override=False, remove_previous=False):
        logger.info("preprocess sfm...")
        from od3d.datasets.sequence_meta import OD3D_SequenceMeta
        for sequence_name_unique in OD3D_SequenceMeta.unroll_nested_metas(self.dict_category_sequences_names):
            sequence = od3d.datasets.monolmb.sequence.MonoLMB_Sequence(name_unique=sequence_name_unique, path_raw=self.path_raw,
                                        path_preprocess=self.path_preprocess, all_categories=self.categories)
            sequence.preprocess_sfm(override=override)
    def get_item(self, item):
        return self.get_frame_by_name_unique(name_unique=self.list_frames_unique[item])


    @staticmethod
    def setup(config: DictConfig):
        logger.info('recording video...')

        # /misc/lmbraid19/sommerl/datasets/MonoLMB/videos/elephant/24_01_29__18_10.mp4
        #
        from od3d.cv.io import extract_frames_from_video
        fpath_video = Path('/misc/lmbraid19/sommerl/datasets/MonoLMB/videos/elephant/24_01_29__18_10.mp4')
        path_frames = Path('/misc/lmbraid19/sommerl/datasets/MonoLMB/frames/elephant/24_01_29__18_10')
        extract_frames_from_video(fpath_video, path_frames, fps=5)


        # import cv2
        #
        # # define a video capture object
        # vid = cv2.VideoCapture(0)
        #
        # while (True):
        #
        #     # Capture the video frame
        #     # by frame
        #     ret, frame = vid.read()
        #
        #     # Display the resulting frame
        #     cv2.imshow('frame', frame)
        #
        #     # the 'q' button is set as the
        #     # quitting button you may use any
        #     # desired button of your choice
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        #
        # # After the loop release the cap object
        # vid.release()
        # # Destroy all the windows
        # cv2.destroyAllWindows()
        # pass

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

