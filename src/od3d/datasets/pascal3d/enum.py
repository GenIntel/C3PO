from od3d.data import ExtEnum
from od3d.datasets.enum import OD3D_CATEGORIES

class PASCAL3D_SUBSETS(str, ExtEnum):
    VAL = "val"
    TRAIN = "train"

class PASCAL3D_CATEGORIES(str, ExtEnum):
    AEROPLANE = "aeroplane"
    BICYCLE = "bicycle"
    BOAT = "boat"
    BOTTLE = "bottle"
    BUS = "bus"
    CAR = "car"
    CHAIR = "chair"
    DININGTABLE = "diningtable"
    MOTORBIKE = "motorbike"
    SOFA = "sofa"
    TRAIN = "train"
    TVMONITOR = "tvmonitor"

MAP_CATEGORIES_OD3D_TO_PASCAL3D = {
    OD3D_CATEGORIES.AIRPLANE: PASCAL3D_CATEGORIES.AEROPLANE,
    OD3D_CATEGORIES.BICYCLE: PASCAL3D_CATEGORIES.BICYCLE,
    OD3D_CATEGORIES.BOAT: PASCAL3D_CATEGORIES.BOAT,
    OD3D_CATEGORIES.BOTTLE: PASCAL3D_CATEGORIES.BOTTLE,
    OD3D_CATEGORIES.BUS: PASCAL3D_CATEGORIES.BUS,
    OD3D_CATEGORIES.CAR: PASCAL3D_CATEGORIES.CAR,
    OD3D_CATEGORIES.CHAIR: PASCAL3D_CATEGORIES.CHAIR,
    OD3D_CATEGORIES.DINING_TABLE: PASCAL3D_CATEGORIES.DININGTABLE,
    OD3D_CATEGORIES.MOTORCYCLE: PASCAL3D_CATEGORIES.MOTORBIKE,
    OD3D_CATEGORIES.COUCH: PASCAL3D_CATEGORIES.SOFA,
    OD3D_CATEGORIES.TRAIN: PASCAL3D_CATEGORIES.TRAIN,
    OD3D_CATEGORIES.TV: PASCAL3D_CATEGORIES.TVMONITOR
}

# note: changing these parameters, requires to recompute cuboids.
PASCAL3D_SCALE_NORMALIZE_TO_REAL = {
    'aeroplane': 1.,
    'bicycle': 0.22, # 0.18m / 0.81 ~= 0.22
    'boat': 1.,
    'bottle': 1.,
    'bus': 1.,
    'car': 5., # 4m / 0.88 ~= 5.
    'chair': 1.,
    'diningtable': 1.,
    'motorbike': 2.65, # 2.2m / 0.83 ~= 2.65
    'sofa': 1.,
    'train': 1.,
    'tvmonitor': 1.
}
