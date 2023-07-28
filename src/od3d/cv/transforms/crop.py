import logging
logger = logging.getLogger(__name__)
import torch
from od3d.cv.geometry.transform import proj3d2d_origin, rot3x3_from_two_vectors, proj3d2d
from od3d.datasets.frame import OD3D_Frame, OD3D_FRAME_MODALITIES
from od3d.cv.visual.crop import crop
from omegaconf import DictConfig
class Crop():

    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, frame: OD3D_Frame):
        # logger.info(f"Frame name {self.name}")

        #_ = frame.size

        scale = min(self.H / frame.H, self.W / frame.W)
        frame.size[0:1] = self.H
        frame.size[1:2] = self.W

        frame.rgb, cam_crop_tform_cam = crop(img=frame.rgb, H_out=self.H, W_out=self.W, scale=scale)

        return frame