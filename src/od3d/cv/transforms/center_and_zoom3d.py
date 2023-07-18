
import logging
logger = logging.getLogger(__name__)
import torch
from od3d.cv.geometry.transform import proj3d2d_origin, rot3x3_from_two_vectors, proj3d2d
from od3d.datasets.dataset import OD3D_FRAME_MODALITIES
from od3d.cv.visual.crop import crop
from omegaconf import DictConfig
from od3d.datasets.dtd import DTD

class CenterZoom3D():

    def __init__(self, H, W, dist, center3d=[0., 0., 0.], apply_txtr=False, apply_kpts2d_annot=False, apply_bbox_annot=False, apply_mask=True, config:DictConfig = None):
        self.center3d = torch.Tensor(center3d) if center3d is not None else None
        self.H = H
        self.W = W
        self.dist = dist
        self.apply_txtr = apply_txtr
        self.apply_kpts2d_annot = apply_kpts2d_annot
        self.apply_bbox_annot=apply_bbox_annot
        self.apply_mask = apply_mask
        if self.apply_txtr:
            self.dtd = DTD(config=config)

    def __call__(self, frame):
        # logger.info(f"Frame name {self.name}")
        _, _, _, _ = frame.size, frame.cam_intr4x4, frame.cam_tform4x4_obj, frame.cam_proj4x4_obj
        if self.dist is not None:
            scale = frame.cam_tform4x4_obj[2, 3] / self.dist
            if scale < 0.01:
                logger.warning(f'Scale is < 0.01. Setting scale to 1.')
                scale = 1.
        else:
            scale = 1.

        if frame.cam_tform4x4_obj[2, 3] <= 0.:
            logger.warning(f"dist <= 0")

        if self.center3d is not None:
            center = proj3d2d(self.center3d, proj4x4=frame.cam_proj4x4_obj)
        else:
            center = frame.size.flip(dims=[0]) / 2

        if OD3D_FRAME_MODALITIES.MASK in frame.modalities:
            frame._mask, _ = crop(img=frame.mask, center=center, H_out=self.H, W_out=self.W, scale=scale, ctx=None)

        frame._size[0:1] = self.H
        frame._size[1:2] = self.W

        #mix_real_with_synthetic, cam_crop_tform_cam = crop(img=mix_real_with_synthetic, center=center, H_out=H_out, W_out=W_out, scale=scale, ctx=self.txtr)
        if self.apply_txtr:
            frame._rgb, cam_crop_tform_cam = crop(img=frame.rgb, center=center, H_out=self.H, W_out=self.W, scale=scale,
                                                  ctx=self.dtd.get_random_item())
        else:
            frame._rgb, cam_crop_tform_cam = crop(img=frame.rgb, center=center, H_out=self.H, W_out=self.W, scale=scale,
                                                 ctx=None)

        frame._cam_intr4x4 = torch.bmm(cam_crop_tform_cam[None,], frame._cam_intr4x4[None,])[0]

        if OD3D_FRAME_MODALITIES.BBOX in frame.modalities:
            frame._bbox = frame.bbox * scale
            frame._bbox[[0, 2]] = frame.bbox[[0, 2]] + cam_crop_tform_cam[0, 2]
            frame._bbox[[1, 3]] = frame.bbox[[1, 3]] + cam_crop_tform_cam[1, 2]

        if OD3D_FRAME_MODALITIES.KPTS in frame.modalities:
            frame._kpts2d_annot = frame.kpts2d_annot * scale
            frame._kpts2d_annot = frame.kpts2d_annot + cam_crop_tform_cam[:2, 2]

        return frame

        """
        if frame.fpath_shapenemo is not None:
            shapenemo_mesh = Mesh(fpath_mesh=frame.fpath_shapenemo)
            frame.shapenemo_vts3d = shapenemo_mesh.verts # load_mesh_vertices(fpath_mesh=self.fpath_shapenemo, device=self.device)
            frame.shapenemo_mask, frame.shapenemo_depth, frame.vts2d, frame.vts3d_vsbl = frame.calc_mesh_proj(fpath_mesh=self.fpath_shapenemo, pts3d=self.shapenemo_vts3d)

        frame.mask, frame.depth, frame.kpts2d, frame.kpts3d_vsbl = frame.calc_mesh_proj(fpath_mesh=frame.fpath_mesh, pts3d=frame.kpts3d)
        """


class RandomCenterZoom3D():
    def __init__(self, H, W, dist, center3d=[0., 0., 0.], apply_txtr=False, apply_kpts2d_annot=False, apply_bbox_annot=False, apply_mask=True, config:DictConfig = None, center3d_min=[0., 0., 0.], center3d_max=[0., 0., 0.]):
        self.centerzoom3d = CenterZoom3D(H=H, W=W, dist=dist, center3d=center3d, apply_txtr=apply_txtr,
                                    apply_kpts2d_annot=apply_kpts2d_annot, apply_bbox_annot=apply_bbox_annot,
                                    apply_mask=apply_mask, config=config)
        self.center3d_min = torch.Tensor(center3d_min)
        self.center3d_max = torch.Tensor(center3d_max)

    def __call__(self, frame):
        self.centerzoom3d.center3d = self.center3d_min + torch.rand(3) * (self.center3d_max - self.center3d_min)
        return self.centerzoom3d(frame)
