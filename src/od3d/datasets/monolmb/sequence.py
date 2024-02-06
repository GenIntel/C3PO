import logging
logger = logging.getLogger(__name__)
from dataclasses import dataclass
from od3d.datasets.monolmb.frame import MonoLMB_Frame

from od3d.datasets.sequence import OD3D_Sequence, OD3D_SequenceCategoryMixin, OD3D_SequenceSfMMixin, OD3D_SEQUENCE_SFM_TYPES
from od3d.datasets.sequence_meta import OD3D_SequenceMeta, OD3D_SequenceMetaCategoryMixin

@dataclass
class MonoLMB_SequenceMeta(OD3D_SequenceMetaCategoryMixin, OD3D_SequenceMeta):
    @staticmethod
    def load_from_raw(category: str, name: str):
        return MonoLMB_SequenceMeta(category=category, name=name)

@dataclass
class MonoLMB_Sequence(OD3D_SequenceSfMMixin, OD3D_SequenceCategoryMixin, OD3D_Sequence):
    sfm_type: OD3D_SEQUENCE_SFM_TYPES = OD3D_SEQUENCE_SFM_TYPES.DROID
    frame_type = MonoLMB_Frame

    #def __init__(self, path_raw: Path, path_preprocess: Path, path_meta: Path, meta: OD3D_CategoricalSequenceMeta,
    #             modalities: List[OD3D_FRAME_MODALITIES], categories: List[str]):
        # self.path_raw: Path = path_raw
        # self.path_preprocess: Path = path_preprocess
        # self.path_meta: Path = path_meta
        # self.meta = meta
        # self.modalities = modalities
        # self.categories = categories
        # self.category_id = categories.index(self.category)
        # self._mesh_feats = None
        # self._mesh_feats_viewpoints = None
        # self._pcl = None
        # self._pcl_colors = None
        # self._pcl_clean = None
        # self._mesh = None
        # self._front_name = None
        # self._cuboid_front_tform4x4_obj = None
        # self._cuboid = None
        # self._meta_ext = None

    # @property
    # def name(self):
    #     return self.meta.name
    #
    # @property
    # def name_unique(self):
    #     return self.meta.name_unique
    #
    # @property
    # def category(self):
    #     return self.meta.category
    #
    # @property
    # def frames_names(self):
    #     frames_names = MonoLMB_FrameMeta.get_frames_names_of_category_sequence(path_meta=self.path_meta, category=self.category, sequence_name=self.name)
    #     return frames_names
    #
    # @property
    # def frames_count(self):
    #     return len(self.frames_names)
    #
    # @property
    # def first_frame(self):
    #     #first_frame_fpath = sorted(CO3D_FrameMeta.get_path_frames_meta_with_category_sequence(path_meta=self.path_meta,
    #     #                                                                                      category=self.category,
    #     #                                                                                      sequence=self.name).iterdir(),
    #     #                           key=lambda p: int(p.stem))[0]
    #     return self.get_frame_by_index(index=0)
    #
    # def get_frames(self, frames_ids=None):
    #     if frames_ids is None:
    #         frames_ids = list(range(self.frames_count))
    #     frames = [self.get_frame_by_index(frame_id) for frame_id in frames_ids]
    #     return frames
    #
    # def get_frame_by_index(self, index: int):
    #     return self.get_frame_by_name(self.frames_names[index])
    #
    # def get_frame_by_name(self, frame_name: str):
    #     try:
    #         frame_meta = MonoLMB_FrameMeta.load_from_meta_with_rfpath(path_meta=self.path_meta,
    #                                                                rfpath=MonoLMB_FrameMeta.
    #                                                                get_rfpath_frame_meta_with_category_sequence_and_frame_name(
    #                                                                    category=self.category, sequence_name=self.name,
    #                                                                    name=frame_name))
    #         frame = MonoLMB_Frame(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
    #                            meta=frame_meta, modalities=self.modalities, categories=self.categories)
    #     except Exception as e:
    #         logger.warning(f'could not retrieve frame {self.path_meta.joinpath(MonoLMB_FrameMeta.get_rfpath_frame_meta_with_category_sequence_and_frame_name(category=self.category, sequence_name=self.name, name=frame_name))}')
    #         frame = None
    #     return frame