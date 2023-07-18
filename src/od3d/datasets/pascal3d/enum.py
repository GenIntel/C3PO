from od3d.data import ExtEnum
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

# note: changing these parameters, requires to recompute cuboids.
PASCAL3D_SCALE_NORMALIZE_TO_REAL = {
    'aeroplane': 1.,
    'bicycle': 1.,
    'boat': 1.,
    'bottle': 1.,
    'bus': 1.,
    'car': 6.,
    'chair': 1.,
    'diningtable': 1.,
    'motorbike': 1.,
    'sofa': 1.,
    'train': 1.,
    'tvmonitor': 1.
}
