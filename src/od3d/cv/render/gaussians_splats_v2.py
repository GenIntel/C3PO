
import torch
from od3d.cv.geometry.transform import proj3d2d_broadcast, tform4x4, transf3d_broadcast


def render_gaussians(
        cams_tform4x4_obj: torch.Tensor,
        cams_intr4x4: torch.Tensor,
        imgs_size: torch.Tensor,
        pts3d: torch.Tensor,
        pts3d_mask: torch.Tensor,
        feats: torch.Tensor,
        pts3d_size_rel_to_neighbor_dist: float=0.5,
        opacity: float=1.,
        z_far: float=10000.,
        z_near: float=0.01,
):
    """
    Args:
        cams_tform4x4_obj (torch.Tensor): B x 4 x 4,
        cams_intr4x4 (torch.Tensor): B x 4 x 4,
        imgs_size (torch.Tensor): 2, (height, width)
        pts3d (torch.Tensor): B x N x 3
        pts3d_mask (torch.Tensor): B x N
        feats (torch.Tensor): B x N x F
        opacity (float)
        z_far (float)
        z_near (float)
        feats_dim_base (int)
    Returns:
        img (torch.Tensor): B x F x H x W
    """
    # logger.info(pts3d.shape)
    device = pts3d.device
    dtype = pts3d.dtype
    B = pts3d.shape[0]
    F = feats.shape[-1]
    N_max = pts3d.shape[1]

    image_height, image_width = imgs_size
    image_height = int(image_height)
    image_width = int(image_width)
    rendered_imgs = []

    for b in range(B):

        N = int(pts3d_mask[b].sum())
        cam_tform4x4_obj_b = cams_tform4x4_obj[b]
        cam_intr4x4_b = cams_intr4x4[b]
        pts3d_b = pts3d[b, pts3d_mask[b]].clone()
        pts3d_cam = transf3d_broadcast(pts3d=pts3d_b, transf4x4=cam_tform4x4_obj_b)

        pts3d_dists = torch.cdist(pts3d_cam.clone().detach(), pts3d_cam.clone().detach())
        pts3d_dists.fill_diagonal_(torch.inf)
        pts3d_size_b = pts3d_dists.min(dim=-1).values[:, None].mean(dim=0, keepdim=True).expand(N, 3) * pts3d_size_rel_to_neighbor_dist
        pts3d_size_b = pts3d_size_b.clamp(1e-5, 1e+5) # otherwise illegal access memory
        feats_b = feats[b, pts3d_mask[b]]


        means3d_cam_z = pts3d_cam[:, 2]

        # gaussians_sorted_id = means3d_cam_z.argsort()
        # means3d_cam = means3d_cam[gaussians_sorted_id]
        # means3d_cam_z = means3d_cam[:, 2]
        # rgb = rgb[gaussians_sorted_id]

        means2d = proj3d2d_broadcast(pts3d=pts3d_cam, proj4x4=cam_intr4x4_b)

        # print(means2d[:10])

        # 2 x H x W
        grid_pxl2d = torch.stack(torch.meshgrid(torch.arange(image_height), torch.arange(image_width), indexing='xy'), dim=0)

        # N x 2 x 3
        cov3d_var = pts3d_size_b
        cov3d = (torch.eye(3).to(device, dtype))[None,].repeat(N, 1, 1) * cov3d_var[:, :, None]
        jacobian3d2d = torch.zeros((N, 2, 3)).to(device, dtype)
        jacobian3d2d[:, 0, 0] = cam_intr4x4_b[0, 0] / means3d_cam_z
        jacobian3d2d[:, 0, 2] = -cam_intr4x4_b[0, 2] * pts3d_cam[:, 0] / (means3d_cam_z**2)
        jacobian3d2d[:, 1, 1] = cam_intr4x4_b[1, 1] / means3d_cam_z
        jacobian3d2d[:, 1, 2] = -cam_intr4x4_b[1, 2] * pts3d_cam[:, 1] / (means3d_cam_z**2)
        cov2d = jacobian3d2d @ cov3d @ jacobian3d2d.permute(0, 2, 1)
        # note: constant 2d covariance for debug
        # cov2d = (torch.eye(2).to(device, dtype))[None,].repeat(N, 1, 1) * 5

        inv_cov2d = torch.inverse(cov2d)

        # note: visualization for debug
        # from od3d.cv.visual.show import show_scene2d, show_img, show_imgs
        # show_scene2d(pts2d =[means2d[gaussians_z_fov][:100], grid_pxl2d.flatten(1).permute(1,0)[:]])

        grid_pxl2d = grid_pxl2d.to(dtype=dtype, device=device)
        grid_pxl2d_dist2d = (grid_pxl2d[:, :, :, None] - means2d.permute(1, 0)[:, None, None]).abs()

        grid_pxl2d_cov_dist = grid_pxl2d_dist2d[0] ** 2 * inv_cov2d[None, None, :, 0, 0] + grid_pxl2d_dist2d[1] ** 2 * inv_cov2d[None, None, :, 1, 1] + \
                          2 * grid_pxl2d_dist2d[0] * grid_pxl2d_dist2d[1] * inv_cov2d[None, None, :, 0, 1]

        grid_pxl2d_cov_opacity = opacity * torch.exp(-0.5 * grid_pxl2d_cov_dist)

        gaussians_z_fov = (means3d_cam_z > z_near) * (means3d_cam_z < z_far)

        # original
        gaussians_z_occlusion = (means3d_cam_z[:, None] > means3d_cam_z[None,]) * 1.
        gaussians_z_occlusion *= gaussians_z_fov[None,] * 1.

        # modified: note: does not work with additive model
        #gaussians_z_occlusion = torch.sigmoid(0.5 * (means3d_cam_z[:, None] - means3d_cam_z[None,]) / cov3d_var_z)
        #gaussians_z_occlusion.fill_diagonal_(0)
        #gaussians_z_occlusion *= gaussians_z_fov[None,] * 1.

        # hwo, no -> hwno: note: does take too much gpu memory
        #grid_pxl2d_z_opacity = 1. - torch.einsum('hwo,no->hwno', grid_pxl2d_cov_opacity, gaussians_z_occlusion).clamp(0, 1)

        # hwo, no -> hwn
        grid_pxl2d_z_opacity = 1. - torch.einsum('hwo,no->hwn', grid_pxl2d_cov_opacity, gaussians_z_occlusion).clamp(0, 1)

        grid_pxl2d_opacity = gaussians_z_fov[None, None] * grid_pxl2d_cov_opacity * grid_pxl2d_z_opacity

        img = torch.einsum('hwn,nc->chw', grid_pxl2d_opacity, feats_b)

        rendered_imgs.append(img)

    rendered_imgs = torch.stack(rendered_imgs, dim=0)

    return rendered_imgs
