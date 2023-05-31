import torch
from pytorch3d.renderer.cameras import look_at_view_transform, look_at_rotation
import math
from pytorch3d.transforms import axis_angle_to_matrix

def transf4x4_from_pos_and_theta(pos, theta):
    in_shape = theta.shape
    dtype = theta.dtype
    device = theta.device
    zeros = torch.zeros(size=in_shape, dtype=dtype, device=device).reshape(-1)
    ones = torch.ones(size=in_shape, dtype=dtype, device=device).reshape(-1)
    camrot_pos_rot3x3_obj = look_at_rotation(pos, up=((0, 0, 1),), device=pos.device).transpose(-1, -2) # pytorch3d convention to store rotation matrix transposed
    camrot_pos_rot3x3_obj[..., 0:2, :] = -camrot_pos_rot3x3_obj[..., 0:2, :] # pytorch3d convention to have negative x,y axis

    camrot_theta_rot3x3_camrot_pos = torch.stack([
        torch.cos(theta), -torch.sin(theta), zeros, torch.sin(theta), torch.cos(theta), zeros, zeros, zeros, ones
    ], dim=-1).reshape(-1, 3, 3).transpose(-1, -2) # transpose is required because theta is usually given in -z axis instead of +z

    camrot_rot3x3_obj = torch.bmm(camrot_theta_rot3x3_camrot_pos, camrot_pos_rot3x3_obj)
    camrot_transl3_obj = rot3d(pts3d=-pos, rot3x3=camrot_rot3x3_obj)
    camrot_tform4x4_obj = transf4x4_from_rot3x3_and_transl3(camrot_rot3x3_obj, camrot_transl3_obj)

    return camrot_tform4x4_obj

    #dist = pos.norm(dim=-1)
    #azim = torch.atan(pos[..., 0] / -pos[..., 1]) % math.pi + math.pi * (pos[..., 0] < 0) # torch.atan(pos[..., 0] / pos[..., 2])  % math.pi + math.pi * (pos[..., 0] < 0)
    #elev = torch.asin(pos[..., 2] / dist)
    #return transf4x4_from_spherical(azim=azim, elev=elev, theta=theta, dist=dist)

def transf4x4_from_spherical(azim, elev, theta, dist):
    # camera center
    obj_transl3_cam = torch.stack([
        dist * torch.cos(elev) * torch.sin(azim),
        -dist * torch.cos(elev) * torch.cos(azim),
        dist * torch.sin(elev)
    ], dim=-1)

    return transf4x4_from_pos_and_theta(obj_transl3_cam, theta)
    """

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

    camrot_rot3x3_cam = torch.bmm(camtheta_tform_camelev, torch.bmm(camelev_tform_camazim, camazim_tform_cam))

    camrot_transl3_obj = rot3d(pts3d=cam_transl3_obj, rot3x3=camrot_rot3x3_cam)

    camrot_tform4x4_obj = transf4x4_from_rot3x3_and_transl3(camrot_rot3x3_cam, camrot_transl3_obj)

    camrot_tform4x4_obj[:, 1:3, :] = -camrot_tform4x4_obj[:, 1:3, :]

    return camrot_tform4x4_obj
    
    """

    #camrot_tform_obj = np.hstack((camrot_tform_cam, np.dot(camrot_tform_cam, cam_transl3_obj)))
    #camrot_tform_obj = np.vstack((camrot_tform_obj, [0, 0, 0, 1]))

    # T =
    # dist, elev, azim
    #R, t = look_at_view_transform(dist, elev=elev_s, azim=azim_s, degrees=False)

    # return 0.


@torch.jit.script
def transf4x4_from_rot3x3(rot3x3):
    transf4x4 = torch.zeros(rot3x3.shape[:-2] + torch.Size([4, 4]), device=rot3x3.device, dtype=rot3x3.dtype)
    transf4x4[..., :3, :3] = rot3x3
    transf4x4[..., 3, 3] = 1.
    return transf4x4

@torch.jit.script
def transf4x4_from_rot3x3_and_transl3(rot3x3, transl3):
    transf4x4 = transf4x4_from_rot3x3(rot3x3)
    transf4x4[..., :3, 3] = transl3
    return transf4x4


def rot2d(pts2d, rot2x2):
    pts2d_shape_in = pts2d.shape
    pts2d_rot = torch.bmm(rot2x2.reshape(-1, 2, 2), pts2d.reshape(-1, 2, 1)).reshape(pts2d_shape_in)
    return pts2d_rot

@torch.jit.script
def rot3d(pts3d, rot3x3):
    pts3d_shape_in = pts3d.shape
    pts3d_rot = torch.bmm(rot3x3.reshape(-1, 3, 3), pts3d.reshape(-1, 3, 1)).reshape(pts3d_shape_in)
    return pts3d_rot

def proj3d2d_broadcast(pts3d, proj4x4):
    shape_first_dims = torch.broadcast_shapes(pts3d.shape[:-1], proj4x4.shape[:-2])
    return proj3d2d(pts3d.expand(*shape_first_dims, 3), proj4x4.expand(*shape_first_dims, 4, 4))


def pts3d_to_pts4d(pts3d):
    device = pts3d.device
    dtype = pts3d.dtype
    ones1d = torch.ones(size=list(pts3d.shape[:-1]) + [1]).to(device=device, dtype=dtype)
    pts4d = torch.concatenate([pts3d, ones1d], dim=-1)
    return pts4d

def pts2d_to_pts3d(pts2d):
    device = pts2d.device
    dtype = pts2d.dtype
    ones1d = torch.ones(size=list(pts2d.shape[:-1]) + [1]).to(device=device, dtype=dtype)
    pts3d = torch.concatenate([pts2d, ones1d], dim=-1)
    return pts3d

def pts2d_to_pts4d(pts2d):
    device = pts2d.device
    dtype = pts2d.dtype
    ones2d = torch.ones(size=list(pts2d.shape[:-1]) + [2]).to(device=device, dtype=dtype)
    pts4d = torch.concatenate([pts2d, ones2d], dim=-1)
    pts4d = pts4d.reshape(list(pts4d.shape) + [1])
    return pts4d

def proj3d2d(pts3d, proj4x4):
    device = pts3d.device
    dtype = pts3d.dtype
    ones1d = torch.ones(size=list(pts3d.shape[:-1]) + [1]).to(device=device, dtype=dtype)
    pts4d = torch.concatenate([pts3d, ones1d], dim=-1)
    pts4d = pts4d.reshape(list(pts4d.shape) + [1])
    pts4d_transf = torch.bmm(proj4x4.reshape(-1, 4, 4), pts4d.reshape(-1, 4, 1)).reshape(pts4d.shape)
    dim_coords3d = pts4d_transf.dim() - 2
    pts3d_transf = pts4d_transf.index_select(dim=dim_coords3d, index=torch.LongTensor([0, 1, 2]).to(device=device))
    pts2d_transf = pts3d_transf.index_select(dim=dim_coords3d, index=torch.LongTensor([0, 1]).to(device=device))
    ptsZ_transf = pts3d_transf.index_select(dim=dim_coords3d, index=torch.LongTensor([2]).to(device=device))
    pts2d_transf_proj = pts2d_transf / ptsZ_transf
    pts2d_transf_proj = pts2d_transf_proj.squeeze(dim=-1)
    return pts2d_transf_proj

def reproj2d3d_broadcast(pxl2d, proj4x4_inv):
    shape_first_dims = torch.broadcast_shapes(pxl2d.shape[:-1], proj4x4_inv.shape[:-2])
    return reproj2d3d(pxl2d.expand(*shape_first_dims, 2), proj4x4_inv.expand(*shape_first_dims, 4, 4))

def reproj2d3d(pxl2d, proj4x4_inv):
    device = pxl2d.device
    pts4d = pts2d_to_pts4d(pxl2d)
    pts4d_reproj = torch.bmm(proj4x4_inv.reshape(-1, 4, 4), pts4d.reshape(-1, 4, 1)).reshape(pts4d.shape)
    dim_coords2d = pts4d_reproj.dim() - 2
    pts3d_reproj = pts4d_reproj.index_select(dim=dim_coords2d, index=torch.LongTensor([0, 1, 2]).to(device=device))
    pts3d_reproj = pts3d_reproj.squeeze(dim=-1)
    return pts3d_reproj

def depth2pts3d(depth, cam_intr4x4):
    device = cam_intr4x4.device
    dtype = cam_intr4x4.dtype
    H, W = depth.shape[-2:]
    pxl2d = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy'), dim=-1).to(device=device, dtype=dtype)
    pts3d_homog = reproj2d3d_broadcast(pxl2d, proj4x4_inv=cam_intr4x4.inverse()).transpose(-2, -1).transpose(-3, -2)
    pts3d = pts3d_homog[(None, ) * (depth.dim() - 3)] * depth
    return pts3d

def transf3d_broadcast(pts3d, transf4x4):
    shape_first_dims = torch.broadcast_shapes(pts3d.shape[:-1], transf4x4.shape[:-2])
    return transf3d(pts3d.expand(*shape_first_dims, 3), transf4x4.expand(*shape_first_dims, 4, 4))

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
    pts4d_transf = torch.bmm(transf4x4.reshape(-1, 4, 4), pts4d.reshape(-1, 4, 1)).reshape(pts4d.shape)
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
    pts2d_transf = torch.bmm(transf3x3.reshape(-1, 3, 3), pts3d.reshape(-1, 3, 1)).reshape(pts3d.shape)
    dim_coords2d = pts2d_transf.dim() - 2
    pts2d_transf = pts2d_transf.index_select(dim=dim_coords2d, index=torch.LongTensor([0, 1]).to(device=device))
    pts2d_transf = pts2d_transf.squeeze(dim=-1)
    return pts2d_transf