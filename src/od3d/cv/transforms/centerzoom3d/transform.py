
import logging
logger = logging.getLogger(__name__)
import torch
from od3d.cv.geometry.transform import proj3d2d_origin, rot3x3_from_two_vectors, proj3d2d
from od3d.datasets.frame import OD3D_FRAME_MODALITIES, OD3D_Frame
from od3d.cv.visual.crop import crop
from omegaconf import DictConfig
from od3d.datasets.dtd import DTD
import torchvision
from od3d.cv.transforms.transform import OD3D_Transform

class CenterZoom3D(OD3D_Transform):
    # resize types: fit to
    def __init__(self, H, W, scale=None, center_rel_shift_xy=[0., 0.], apply_txtr=False, config: DictConfig = None, scale_with_mask=None, scale_with_dist=None):
        super().__init__()
        self.center_rel_shift_xy = torch.Tensor(center_rel_shift_xy) if center_rel_shift_xy is not None else None
        self.H = H
        self.W = W
        self.scale = scale
        self.scale_with_mask = scale_with_mask
        self.scale_with_dist = scale_with_dist
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

        if self.scale_with_dist is not None:
            # note: this usage should become deprecated in the future.
            from od3d.datasets.pascal3d.enum import PASCAL3D_SCALE_NORMALIZE_TO_REAL
            dist = self.scale_with_dist
            if frame.category is not None and frame.category in PASCAL3D_SCALE_NORMALIZE_TO_REAL.keys():
                dist *= PASCAL3D_SCALE_NORMALIZE_TO_REAL[frame.category]
            scale = frame.cam_tform4x4_obj[2, 3] / dist

        elif self.scale_with_mask is not None and frame.mask is not None:
            # note: this usage should become deprecated in the future.
            if self.scale is not None:
                logger.warning('For CenterZoom3D `scale` and `scale_with_mask` are not None. Only using `scale_with_mask`.')

            if frame.mask.sum() > 0.:
                from od3d.cv.geometry.grid import get_pxl2d
                mask_pxl2d = get_pxl2d(H=frame.mask.shape[1], W=frame.mask.shape[2], dtype=float, device=frame.mask.device)
                mask_H = mask_pxl2d[:, 1].max() - mask_pxl2d[:, 1].min()
                mask_W = mask_pxl2d[:, 0].max() - mask_pxl2d[:, 0].min()
            else:
                logger.warning('For CenterZoom3D using `scale_with_mask` despite mask has only zeros. Setting mask width and height to image width and height.')
                mask_H = self.H
                mask_W = self.W

            scale = min((self.scale_with_mask * self.H) / mask_H, (self.W * self.scale_with_mask) / mask_W)
        else:
            if self.scale_with_mask is not None:
                logger.warning('For CenterZoom3D `scale_with_mask` is not None, but frame.mask is None. Ignoring `scale_with_mask`')
            # this automatic scales to fit the cropped image
            centered_frame_H = int(max(abs(frame.H - center2d[1]), abs(center2d[1])) * 2)
            centered_frame_W = int(max(abs(frame.W - center2d[0]), abs(center2d[0])) * 2)
            scale = min(self.H / centered_frame_H, self.W / centered_frame_W)

            if self.scale is not None:
                # scale = frame.cam_tform4x4_obj[2, 3] / self.dist
                scale *= self.scale

        # logger.info(f'scale = {scale}')
        if scale < 0.01:
            logger.warning(f'Scale is < 0.01. Setting scale to 1.')
            scale = 1.

        if scale > 100.:
            logger.warning(f'Scale is > 100. Setting scale to 1.')
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

        if self.scale_with_dist is not None:
            # note: this usage should become deprecated in the future.
            frame.cam_intr4x4[:2, :2] *= 1./scale
            frame.cam_tform4x4_obj[2, 3] *= 1./scale

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

