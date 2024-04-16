import logging

logger = logging.getLogger(__name__)
from od3d.cv.transforms.transform import OD3D_Transform


class RGB_UInt8ToFloat(OD3D_Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, frame):
        frame.rgb = frame.get_rgb() / 255.0
        return frame
