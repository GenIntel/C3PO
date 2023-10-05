import torch
from od3d.cv.transforms.transform import OD3D_Transform

class RGB_FloatToUInt8(OD3D_Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, frame):
        frame._rgb = (frame.rgb * 255.).to(dtype=torch.uint8)
        return frame