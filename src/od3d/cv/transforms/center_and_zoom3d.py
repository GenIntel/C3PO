

import torch
from od3d.cv.geometry.transform import proj3d2d
from od3d.cv.visual.crop import crop

class CenterZoom3D():

    def __init__(self, H, W, dist, apply_txtr=False, apply_kpts2d_annot=False, apply_bbox_annot=False):
        self.H = H
        self.W = W
        self.dist = dist
        self.apply_txtr = apply_txtr
        self.apply_kpts2d_annot = apply_kpts2d_annot
        self.apply_bbox_annot= apply_bbox_annot

    def __call__(self, frame):

        # logger.info(f"Frame name {self.name}")

        #center = torch.LongTensor([500, 200]).to(device=self.device)
        #center = torch.Tensor([(self.bbox[0] + self.bbox[2]) / 2., (self.bbox[1] + self.bbox[3]) / 2.]).to(
        #    device=self.device, dtype=self.dtype)
        center = proj3d2d(pts3d=torch.zeros(size=(1, 3)), proj4x4=frame.cam_proj4x4_obj)[0]
        scale = frame.cam_tform4x4_obj[2, 3] / self.dist
        frame.cam_tform4x4_obj[2, 3] = frame.cam_tform4x4_obj[2, 3] / scale
        frame.cam_intr4x4[:2, 2] = frame.cam_intr4x4[:2, 2] * scale

        frame.size[0:1] = self.H
        frame.size[1:2] = self.W
        # cam_proj4x4_obj = torch.bmm(frame.cam_intr4x4[None,], frame.cam_tform4x4_obj[None,])[0]

        #mix_real_with_synthetic, cam_crop_tform_cam = crop(img=mix_real_with_synthetic, center=center, H_out=H_out, W_out=W_out, scale=scale, ctx=self.txtr)
        if self.apply_txtr:
            frame._rgb, cam_crop_tform_cam = crop(img=frame.rgb, center=center, H_out=self.H, W_out=self.W, scale=scale,
                                                 ctx=frame.txtr)
        else:
            frame._rgb, cam_crop_tform_cam = crop(img=frame.rgb, center=center, H_out=self.H, W_out=self.W, scale=scale,
                                                 ctx=None)

        # we already account for the scale with the transformation, but we cannot do that for the padding
        cam_crop_tform_cam[0, 0] = 1.
        cam_crop_tform_cam[1, 1] = 1.
        frame.cam_intr4x4[:, :] = torch.bmm(cam_crop_tform_cam[None,], frame.cam_intr4x4[None,])[0]
        frame.cam_proj4x4_obj[:, :] = torch.bmm(frame.cam_intr4x4[None,], frame.cam_tform4x4_obj[None,])[0]

        if self.apply_bbox_annot:
            frame.bbox = frame.bbox * scale
            frame.bbox[[0, 2]] = frame.bbox[[0, 2]] + cam_crop_tform_cam[0, 2]
            frame.bbox[[1, 3]] = frame.bbox[[1, 3]] + cam_crop_tform_cam[1, 2]

        if self.apply_kpts2d_annot:
            frame.kpts2d_annot = frame.kpts2d_annot * scale
            frame.kpts2d_annot = frame.kpts2d_annot + cam_crop_tform_cam[:2, 2]

        return frame

        """
        if frame.fpath_shapenemo is not None:
            shapenemo_mesh = Mesh(fpath_mesh=frame.fpath_shapenemo)
            frame.shapenemo_vts3d = shapenemo_mesh.verts # load_mesh_vertices(fpath_mesh=self.fpath_shapenemo, device=self.device)
            frame.shapenemo_mask, frame.shapenemo_depth, frame.vts2d, frame.vts3d_vsbl = frame.calc_mesh_proj(fpath_mesh=self.fpath_shapenemo, pts3d=self.shapenemo_vts3d)

        frame.mask, frame.depth, frame.kpts2d, frame.kpts3d_vsbl = frame.calc_mesh_proj(fpath_mesh=frame.fpath_mesh, pts3d=frame.kpts3d)
        """