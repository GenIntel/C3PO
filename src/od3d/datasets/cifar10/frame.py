import logging

logger = logging.getLogger(__name__)
from dataclasses import dataclass
from od3d.datasets.frame import OD3D_FrameMeta, OD3D_Frame

from od3d.datasets.frame_meta import (
    OD3D_FrameMeta,
    OD3D_FrameMetaCategoryMixin,
    OD3D_FrameMetaRGBMixin,
    OD3D_FrameMetaSizeMixin,
    OD3D_FrameMetaSubsetMixin,
)

from od3d.datasets.pascal3d.enum import MAP_CATEGORIES_PASCAL3D_TO_OD3D
from od3d.datasets.frame import (
    OD3D_FrameRGBMaskMixin,
    OD3D_FrameMaskMixin,
    OD3D_FrameRGBMixin,
    OD3D_FrameCategoryMixin,
    OD3D_FrameSizeMixin,
    OD3D_Frame,
)


@dataclass
class CIFAR10_FrameMeta(
    OD3D_FrameMetaSubsetMixin,
    OD3D_FrameMetaCategoryMixin,
    OD3D_FrameMetaRGBMixin,
    OD3D_FrameMetaSizeMixin,
    OD3D_FrameMeta,
):
    @property
    def name_unique(self):
        return f"{self.subset}/{self.category}/{self.name}"

    @staticmethod
    def get_name_unique_from_category_subset_name(category, subset, name):
        return f"{subset}/{category}/{name}"


@dataclass
class CIFAR10_Frame(
    OD3D_FrameRGBMaskMixin,
    OD3D_FrameMaskMixin,
    OD3D_FrameRGBMixin,
    OD3D_FrameCategoryMixin,
    OD3D_FrameSizeMixin,
    OD3D_Frame,
):
    meta_type = CIFAR10_FrameMeta
    map_categories_to_od3d = MAP_CATEGORIES_PASCAL3D_TO_OD3D
