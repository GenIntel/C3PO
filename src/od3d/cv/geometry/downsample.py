
import torch
def voxel_downsampling(pts3d_cls, K):
    """
    Args:
        pts3d_cls (torch.Tensor): Nx3

    Returns:
        pts3d_cls (torch.Tensor): (<=K)x3
    """
    device = pts3d_cls.device
    bounds_max, _ = pts3d_cls.max(dim=-2)
    bounds_min, _ = pts3d_cls.min(dim=-2)
    bounds = bounds_max - bounds_min
    voxel_size = (bounds.prod(dim=-1) / K) ** (1 / 3)
    steps = (bounds / voxel_size).int()
    pts3d_voxel = torch.stack(
        torch.meshgrid(torch.linspace(start=bounds_min[0], end=bounds_max[0], steps=steps[0], device=device),
                       torch.linspace(start=bounds_min[1], end=bounds_max[1], steps=steps[1], device=device),
                       torch.linspace(start=bounds_min[2], end=bounds_max[2], steps=steps[2], device=device),
                       indexing='xy'), dim=-1)
    dist = (pts3d_voxel.reshape(-1, 3)[None, :] - pts3d_cls[:, None, ]).norm(dim=-1)
    _, dist_min_ids = dist.min(dim=0)
    pts3d_cls = pts3d_cls[dist_min_ids]
    del pts3d_voxel
    del dist
    del dist_min_ids

    return pts3d_cls