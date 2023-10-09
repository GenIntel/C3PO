import logging
logger = logging.getLogger(__name__)
from od3d.cv.select import batched_index_select
import torch
from od3d.cv.select import batched_indexMD_select
from pytorch3d.ops.points_alignment import corresponding_points_alignment


def fit_tform4x4(pts: torch.Tensor, pts_ids: torch.LongTensor, pts_ref: torch.Tensor, dist_ref: torch.Tensor):
    """
    Args:
        pts (torch.Tensor): ...xNxF
        pts_ids (torch.Tensor): ...xPxS
        pts_ref (torch.Tensor): ...xRxF
        dist_ref (torch.Tensor): ...xNxR
    Returns:
        tform4x4 (torch.Tensor): ...xPx4x4
    """
    N, F = pts.shape[-2:]
    P, S = pts_ids.shape[-2:]
    device=pts.device

    # ...xPxSxF
    pts_sampled = batched_index_select(index=pts_ids.flatten(-2), input=pts).view(pts_ids.shape + (-1,))
    # ...xPxSxR
    dist_sampled = batched_index_select(index=pts_ids.flatten(-2), input=dist_ref).view(pts_ids.shape + (-1,))

    pts_ref_ids = dist_sampled.argmin(dim=-1)
    pts_ref_sampled = batched_index_select(index=pts_ref_ids.flatten(-2), input=pts_ref).view(pts_ref_ids.shape + (-1,))

    ## start single batch dimension (= flatten proposals)

    S = corresponding_points_alignment(pts_sampled.view(-1, S, F), pts_ref_sampled.view(-1, S, F), weights=None,
                                       estimate_scale=True)

    pts_ref_tform4x4_pts = torch.zeros(size=(pts_sampled.shape[:-2].numel(), 4, 4)).to(
        device=device)
    pts_ref_tform4x4_pts[..., 3, 3] = 1.
    pts_ref_tform4x4_pts[..., :3, :3] = S.R.permute(0, 2, 1) * S.s[..., None, None]
    pts_ref_tform4x4_pts[..., :3, 3] = S.T

    from od3d.cv.geometry.transform import transf3d_broadcast
    pts_ref_tform_pts_sampled = transf3d_broadcast(pts3d=pts_sampled, transf4x4=pts_ref_tform4x4_pts[:, None])

    logger.info(
        f'sampled geometrical error before transform: \n{(pts_sampled - pts_ref_sampled).norm(dim=-1).mean(dim=1).mean()}')
    logger.info(
        f'sampled geometrical error after transform: \n{(pts_ref_tform_pts_sampled - pts_ref_sampled).norm(dim=-1).mean(dim=1).mean()}')

    # problem of zeros is ill posed problem if from all 4 sampled points the nearest neighbor is the same.
    mask_pts_ref_tform4x4_pts_zeros = pts_ref_tform4x4_pts[:, :3, :3].flatten(1).sum(dim=-1) == 0.
    pts_ref_tform4x4_pts[mask_pts_ref_tform4x4_pts_zeros, :3, :3] = torch.eye(3)[None,].expand(
        mask_pts_ref_tform4x4_pts_zeros.sum(), 3, 3).to(device=device)

    ## end single batch dimension
    pts_ref_tform4x4_pts = pts_ref_tform4x4_pts.view(pts_sampled.shape[:-2] + (4, 4))

    return pts_ref_tform4x4_pts


def score_tform4x4_fit(pts: torch.Tensor, tform4x4: torch.Tensor, pts_ref: torch.Tensor, dist_ref: torch.Tensor):
    """
    Args:
        pts (torch.Tensor): ...xNxF
        tform4x4 (torch.Tensor): ...xPx4x4
        pts_ref (torch.Tensor): ...xRxF
        dist_ref (torch.Tensor): ...xNxR
    Returns:
        scores (torch.Tensor): ...xP
    """
    N, F = pts.shape[-2:]
    P = tform4x4.shape[-3]
    device = pts.device

    from od3d.cv.geometry.transform import transf3d_broadcast
    proposal_tform_pts = transf3d_broadcast(pts3d=pts[None,], transf4x4=tform4x4[:, None])

    # PxNxR
    # dist_ref_geometry = (proposal_tform_pts[:, :, None] - pts_ref[None, None,]).norm(dim=-1)
    dist_ref_geometry = torch.cdist(proposal_tform_pts, pts_ref[None,], p=1)  #

    #dist_ref_finite_mask = dist_ref_geometry.isfinite()
    #dist_ref_geometry[~dist_ref_finite_mask] = dist_ref_geometry[dist_ref_finite_mask].max()
    if (~dist_ref_geometry.isfinite()).any():
        logger.warning(f'There are some infinite vlaues in dist geometry. WHY?')
    dist_ref_geometry = dist_ref_geometry / (pts_ref.flatten(-2).max() - pts_ref.flatten(-2).min())

    # PxN
    proposal_tform_pts_nn_ref_id = dist_ref_geometry.argmin(dim=-1)
    # PxR
    proposal_tform_pts_ref_nn_pts_id = dist_ref_geometry.argmin(dim=-2)

    # PxNx2
    proposal_tform_pts_nn_ref_id_2D = torch.stack(
        [torch.arange(N).view(1, -1).expand(proposal_tform_pts_nn_ref_id.shape).to(device=device),
         proposal_tform_pts_nn_ref_id], dim=-1)
    proposal_tform_pts_nn_ref_id_2D = proposal_tform_pts_nn_ref_id_2D.clone()
    # PxRx2
    proposal_tform_pts_ref_nn_pts_id_2D = torch.stack([torch.arange(proposal_tform_pts_ref_nn_pts_id.shape[-1]).view(1,
                                                                                                                     -1).expand(
        proposal_tform_pts_ref_nn_pts_id.shape).to(device=device), proposal_tform_pts_ref_nn_pts_id], dim=-1)
    proposal_tform_pts_ref_nn_pts_id_2D = proposal_tform_pts_ref_nn_pts_id_2D.flip(dims=[-1])
    proposal_tform_pts_ref_nn_pts_id_2D = proposal_tform_pts_ref_nn_pts_id_2D.clone()

    # PxNxR
    dist_ref_total = dist_ref_geometry + dist_ref[None,]  # + dist_ref_geometry.mean(dim=-1).mean(dim=-1)[:, None, None]

    proposal_dist_ref = batched_indexMD_select(
        indexMD=torch.cat([proposal_tform_pts_nn_ref_id_2D, proposal_tform_pts_ref_nn_pts_id_2D], dim=1),
        inputMD=dist_ref_total)
    # proposal_dist_ref = batched_indexMD_select(indexMD=proposal_tform_pts_ref_nn_pts_id_2D, inputMD=dist_ref_total)
    # proposal_dist_ref = batched_indexMD_select(indexMD=proposal_tform_pts_nn_ref_id_2D, inputMD=dist_ref_total)

    propsoal_dist_ref_isfinite = proposal_dist_ref.isfinite()
    proposal_dist_ref[~propsoal_dist_ref_isfinite] = proposal_dist_ref.max()

    # scores = -proposal_dist_ref.quantile(q=0.9, dim=-1) #  (proposal_dist_ref * propsoal_dist_ref_isfinite).sum(dim=-1) / (propsoal_dist_ref_isfinite.sum(dim=-1) + 1e-10)
    scores = -proposal_dist_ref.mean(
        dim=-1)  # (proposal_dist_ref * propsoal_dist_ref_isfinite).sum(dim=-1) / (propsoal_dist_ref_isfinite.sum(dim=-1) + 1e-10)

    return scores