import torch
from pytorch3d.renderer.cameras import look_at_view_transform
import math

def transf4x4_from_pos_and_theta(pos, theta):
    dist = pos.norm(dim=-1)
    azim = torch.atan(pos[..., 0] / -pos[..., 1]) % math.pi + math.pi * (pos[..., 0] < 0) # torch.atan(pos[..., 0] / pos[..., 2])  % math.pi + math.pi * (pos[..., 0] < 0)
    elev = torch.asin(pos[..., 2] / dist)
    return transf4x4_from_spherical(azim=azim, elev=elev, theta=theta, dist=dist)

def transf4x4_from_spherical(azim, elev, theta, dist):
    in_shape = azim.shape
    dtype = azim.dtype
    device = azim.device
    zeros = torch.zeros(size=in_shape, dtype=dtype, device=device).reshape(-1)
    ones = torch.ones(size=in_shape, dtype=dtype, device=device).reshape(-1)

    # camera center
    obj_transl3_cam = torch.stack([
        dist * torch.cos(elev) * torch.sin(azim),
        -dist * torch.cos(elev) * torch.cos(azim),
        dist * torch.sin(elev)
    ], dim=-1)

    cam_transl3_obj = -obj_transl3_cam

    azim = -azim
    elev = -(torch.pi / 2 - elev)

    # rotation matrix
    camazim_tform_cam = torch.stack([
        torch.cos(azim), -torch.sin(azim), zeros, torch.sin(azim), torch.cos(azim), zeros, zeros, zeros, ones
    ], dim=-1).reshape(-1, 3, 3) # .permute(2, 0, 1)

    camelev_tform_camazim = torch.stack([
        ones, zeros, zeros, zeros, torch.cos(elev), -torch.sin(elev), zeros, torch.sin(elev), torch.cos(elev)
    ], dim=-1).reshape(-1, 3, 3) # .permute(2, 0, 1)

    camtheta_tform_camelev = torch.stack([
        torch.cos(theta), -torch.sin(theta), zeros, torch.sin(theta), torch.cos(theta), zeros, zeros, zeros, ones
    ], dim=-1).reshape(-1, 3, 3) # .permute(2, 0, 1)

    camrot_rot3x3_cam = torch.matmul(camtheta_tform_camelev, torch.matmul(camelev_tform_camazim, camazim_tform_cam))

    camrot_transl3_obj = rot3d(pts3d=cam_transl3_obj, rot3x3=camrot_rot3x3_cam)

    camrot_tform4x4_obj = transf4x4_from_rot3x3_and_transl3(camrot_rot3x3_cam, camrot_transl3_obj)

    camrot_tform4x4_obj[:, 1, :] = -camrot_tform4x4_obj[:, 1, :]
    camrot_tform4x4_obj[:, 2, :] = -camrot_tform4x4_obj[:, 2, :]

    return camrot_tform4x4_obj

    #camrot_tform_obj = np.hstack((camrot_tform_cam, np.dot(camrot_tform_cam, cam_transl3_obj)))
    #camrot_tform_obj = np.vstack((camrot_tform_obj, [0, 0, 0, 1]))

    # T =
    # dist, elev, azim
    #R, t = look_at_view_transform(dist, elev=elev_s, azim=azim_s, degrees=False)

    # return 0.

def transf4x4_from_rot3x3_and_transl3(rot3x3, transl3):
    transf4x4 = transf4x4_from_rot3x3(rot3x3)
    transf4x4[..., :3, 3] = transl3
    return transf4x4
def transf4x4_from_rot3x3(rot3x3):
    transf4x4 = torch.eye(n=4, dtype=rot3x3.dtype, device=rot3x3.device).repeat(*rot3x3.shape[:-2], 1, 1)
    transf4x4[..., :3, :3] = rot3x3
    return transf4x4
def rot2d(pts2d, rot2x2):
    pts2d = pts2d.reshape(list(pts2d.shape) + [1])
    pts2d_rot = torch.matmul(rot2x2, pts2d)
    pts2d_rot = pts2d_rot.squeeze(dim=-1)
    return pts2d_rot

def rot3d(pts3d, rot3x3):
    pts3d = pts3d.reshape(list(pts3d.shape) + [1])
    pts3d_rot = torch.matmul(rot3x3, pts3d)
    pts3d_rot = pts3d_rot.squeeze(dim=-1)
    return pts3d_rot

def proj3d2d(pts3d, proj4x4):
    device = pts3d.device
    dtype = pts3d.dtype
    ones1d = torch.ones(size=list(pts3d.shape[:-1]) + [1]).to(device=device, dtype=dtype)
    pts4d = torch.concatenate([pts3d, ones1d], dim=-1)
    pts4d = pts4d.reshape(list(pts4d.shape) + [1])
    pts4d_transf = torch.matmul(proj4x4, pts4d)
    dim_coords3d = pts4d_transf.dim() - 2
    pts3d_transf = pts4d_transf.index_select(dim=dim_coords3d, index=torch.LongTensor([0, 1, 2]).to(device=device))
    pts2d_transf = pts3d_transf.index_select(dim=dim_coords3d, index=torch.LongTensor([0, 1]).to(device=device))
    ptsZ_transf = pts3d_transf.index_select(dim=dim_coords3d, index=torch.LongTensor([2]).to(device=device))
    pts2d_transf_proj = pts2d_transf / ptsZ_transf
    pts2d_transf_proj = pts2d_transf_proj.squeeze(dim=-1)
    return pts2d_transf_proj

def reproj2d3d(pxl2d, proj4x4):
    device = pxl2d.device
    dtype = pxl2d.dtype
    ones2d = torch.ones(size=list(pxl2d.shape[:-1]) + [2]).to(device=device, dtype=dtype)
    pts4d = torch.concatenate([pxl2d, ones2d], dim=-1)
    pts4d = pts4d.reshape(list(pts4d.shape) + [1])
    pts4d_reproj = torch.matmul(proj4x4.inverse(), pts4d)
    dim_coords2d = pts4d_reproj.dim() - 2
    pts2d_reproj = pts4d_reproj.index_select(dim=dim_coords2d, index=torch.LongTensor([0, 1]).to(device=device))
    pts2d_reproj = pts2d_reproj.squeeze(dim=-1)
    return pts2d_reproj

def transf3d(pts3d, transf4x4):
    """
    Args:
        pts3d (torch.Tensor): ...x3
        transf4x4 (torch.Tensor): ...x4x4

    Returns:
        pts3d_transf (torch.Tensor): ...x3
    """
    device = pts3d.device
    dtype = pts3d.dtype
    ones1d = torch.ones(size=list(pts3d.shape[:-1]) + [1]).to(device=device, dtype=dtype)
    pts4d = torch.concatenate([pts3d, ones1d], dim=-1)
    pts4d = pts4d.reshape(list(pts4d.shape) + [1])
    pts4d_transf = torch.matmul(transf4x4, pts4d)
    dim_coords3d = pts4d_transf.dim() - 2
    pts3d_transf = pts4d_transf.index_select(dim=dim_coords3d, index=torch.LongTensor([0, 1, 2]).to(device=device))
    pts3d_transf = pts3d_transf.squeeze(dim=-1)
    return pts3d_transf


def transf2d(pts2d, transf3x3):
    """
    Args:
        pts2d (torch.Tensor): ...x2
        transf3x3 (torch.Tensor): ...x3x3

    Returns:
        pts2d_transf (torch.Tensor): ...x2
    """
    device = pts2d.device
    dtype = pts2d.dtype
    ones1d = torch.ones(size=list(pts2d.shape[:-1]) + [1]).to(device=device, dtype=dtype)
    pts3d = torch.concatenate([pts2d, ones1d], dim=-1)
    pts3d = pts3d.reshape(list(pts3d.shape) + [1])
    pts2d_transf = torch.matmul(transf3x3, pts3d)
    dim_coords2d = pts2d_transf.dim() - 2
    pts2d_transf = pts2d_transf.index_select(dim=dim_coords2d, index=torch.LongTensor([0, 1]).to(device=device))
    pts2d_transf = pts2d_transf.squeeze(dim=-1)
    return pts2d_transf