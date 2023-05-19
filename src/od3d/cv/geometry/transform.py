import torch

def rot2d(pts2d, rot2x2):
    pts2d = pts2d.reshape(list(pts2d.shape) + [1])
    pts2d_rot = torch.matmul(rot2x2, pts2d)
    pts2d_rot = pts2d_rot.squeeze(dim=-1)
    return pts2d_rot
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