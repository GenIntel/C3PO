
import logging
logger = logging.getLogger(__name__)
import torch
from od3d.cv.geometry.transform import proj3d2d_origin, rot3x3_from_two_vectors, proj3d2d
from od3d.datasets.frame import OD3D_FRAME_MODALITIES, OD3D_Frame
from od3d.cv.visual.crop import crop
from omegaconf import DictConfig
from od3d.datasets.dtd import DTD
import torchvision
class CenterZoom3D():
    # resize types: fit to
    def __init__(self, H, W, scale=None, center_rel_shift_xy=[0., 0.], apply_txtr=False, config: DictConfig = None):
        self.center_rel_shift_xy = torch.Tensor(center_rel_shift_xy) if center_rel_shift_xy is not None else None
        self.H = H
        self.W = W
        self.scale = scale
        self.apply_txtr = apply_txtr
        if self.apply_txtr:
            self.dtd = DTD.create_from_config(config=config, transform=torchvision.transforms.Compose([]))

    def __call__(self, frame: OD3D_Frame):
        # logger.info(f"Frame name {self.name}")
        # _, _, _, _ = frame.size, frame.cam_intr4x4, frame.cam_tform4x4_obj, frame.cam_proj4x4_obj

        if frame.cam_tform4x4_obj[2, 3] <= 0.:
            logger.warning(f"dist <= 0")

        if self.center_rel_shift_xy is not None:
            center2d = proj3d2d(torch.Tensor([0., 0., 0.]), proj4x4=frame.cam_proj4x4_obj)
            if center2d.isnan().any():
                center2d = frame.size.flip(dims=[0]) / 2

        else:
            center2d = frame.size.flip(dims=[0]) / 2

        # this automatic scales to fit the cropped image
        centered_frame_H = int(max(abs(frame.H - center2d[1]), abs(center2d[1])) * 2)
        centered_frame_W = int(max(abs(frame.W - center2d[0]), abs(center2d[0])) * 2)
        scale = min(self.H / centered_frame_H, self.W / centered_frame_W)

        if self.scale is not None:
            # scale = frame.cam_tform4x4_obj[2, 3] / self.dist
            scale *= self.scale
            if scale < 0.01:
                logger.warning(f'Scale is < 0.01. Setting scale to 1.')
                scale = 1.

        center2d_shifted = center2d.clone()
        if self.center_rel_shift_xy is not None:
            center2d_shifted[0] += frame.W * self.center_rel_shift_xy[0]
            center2d_shifted[1] += frame.H * self.center_rel_shift_xy[1]

        frame.mask_rgb, _ = crop(frame.mask_rgb, center=center2d_shifted, H_out=self.H, W_out=self.W, scale=scale, ctx=None)

        if OD3D_FRAME_MODALITIES.MASK in frame.modalities:
            frame.mask, _ = crop(img=frame.mask, center=center2d_shifted, H_out=self.H, W_out=self.W, scale=scale, ctx=None)


        #mix_real_with_synthetic, cam_crop_tform_cam = crop(img=mix_real_with_synthetic, center=center, H_out=H_out, W_out=W_out, scale=scale, ctx=self.txtr)
        if self.apply_txtr:
            frame.rgb, cam_crop_tform_cam = crop(img=frame.rgb, center=center2d_shifted, H_out=self.H, W_out=self.W, scale=scale,
                                                  ctx=self.dtd.get_random_item().rgb)
        else:
            frame.rgb, cam_crop_tform_cam = crop(img=frame.rgb, center=center2d_shifted, H_out=self.H, W_out=self.W, scale=scale,
                                                 ctx=None)

        frame.size[0:1] = self.H
        frame.size[1:2] = self.W

        frame.cam_intr4x4 = torch.bmm(cam_crop_tform_cam[None,], frame.cam_intr4x4[None,])[0]

        if OD3D_FRAME_MODALITIES.BBOX in frame.modalities:
            frame.bbox = frame.bbox * scale
            frame.bbox[[0, 2]] = frame.bbox[[0, 2]] + cam_crop_tform_cam[0, 2]
            frame.bbox[[1, 3]] = frame.bbox[[1, 3]] + cam_crop_tform_cam[1, 2]

        if OD3D_FRAME_MODALITIES.KPTS in frame.modalities:
            frame.kpts2d_annot = frame.kpts2d_annot * scale
            frame.kpts2d_annot = frame.kpts2d_annot + cam_crop_tform_cam[:2, 2]

        # assumption: depth of all image points is the same (which of course does only approximately holds)
        #frame.cam_tform4x4_obj[:2, 3] = 0.
        #frame.cam_intr4x4[0, 2] = self.W / 2.
        #frame.cam_intr4x4[1, 2] = self.H / 2.

        return frame

        """
        if frame.fpath_shapenemo is not None:
            shapenemo_mesh = Mesh(fpath_mesh=frame.fpath_shapenemo)
            frame.shapenemo_vts3d = shapenemo_mesh.verts # load_mesh_vertices(fpath_mesh=self.fpath_shapenemo, device=self.device)
            frame.shapenemo_mask, frame.shapenemo_depth, frame.vts2d, frame.vts3d_vsbl = frame.calc_mesh_proj(fpath_mesh=self.fpath_shapenemo, pts3d=self.shapenemo_vts3d)

        frame.mask, frame.depth, frame.kpts2d, frame.kpts3d_vsbl = frame.calc_mesh_proj(fpath_mesh=frame.fpath_mesh, pts3d=frame.kpts3d)
        """


class RandomCenterZoom3D():
    def __init__(self, H, W, apply_txtr=False, config:DictConfig = None, scale_min=None, scale_max=None, center_rel_shift_xy_min=[0., 0.], center_rel_shift_xy_max=[0., 0.]):
        self.centerzoom3d = CenterZoom3D(H=H, W=W, scale=None, apply_txtr=apply_txtr, config=config)
        self.center_rel_shift_xy_min = torch.Tensor(center_rel_shift_xy_min)
        self.center_rel_shift_xy_max = torch.Tensor(center_rel_shift_xy_max)
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, frame):
        if self.scale_min is not None and self.scale_max is not None:
            self.centerzoom3d.scale = self.scale_min + torch.rand(1)[0] * (self.scale_max - self.scale_min)
        self.centerzoom3d.center_rel_shift_xy = self.center_rel_shift_xy_min + torch.rand(2) * (self.center_rel_shift_xy_max - self.center_rel_shift_xy_min)
        return self.centerzoom3d(frame)
