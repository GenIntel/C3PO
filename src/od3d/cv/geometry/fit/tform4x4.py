import logging
logger = logging.getLogger(__name__)
from od3d.cv.select import batched_index_select, batched_index_fill
import torch
from od3d.cv.select import batched_indexMD_select
from pytorch3d.ops.points_alignment import corresponding_points_alignment
from od3d.cv.geometry.transform import transf3d_broadcast
from od3d.cv.geometry.transform import transf3d_broadcast


def fit_tform4x4_with_matches3d3d(pts: torch.Tensor, pts_ref: torch.Tensor, estimate_scale=True):
    """
    Args:
        pts (torch.Tensor): ...xNxF
        pts_ref (torch.Tensor): ...xNxF
    Returns:
        ref_tform4x4 (torch.Tensor): ...xNx4x4
    """
    N, F = pts.shape[-2:]
    device = pts.device
    S = corresponding_points_alignment(pts.view(-1, N, F), pts_ref.view(-1, N, F), weights=None,
                                       estimate_scale=estimate_scale)

    pts_ref_tform4x4_pts = torch.zeros(size=(pts.shape[:-2].numel(), 4, 4)).to(device=device)
    pts_ref_tform4x4_pts[..., 3, 3] = 1.
    pts_ref_tform4x4_pts[..., :3, :3] = S.R.permute(0, 2, 1) * S.s[..., None, None]
    pts_ref_tform4x4_pts[..., :3, 3] = S.T

    pts_ref_tform4x4_pts = pts_ref_tform4x4_pts.reshape(*pts.shape[:-2], 4, 4)
    return pts_ref_tform4x4_pts

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
    batch_dims = pts.shape[:-2]
    N, F = pts.shape[-2:]
    R = pts_ref.shape[-2]
    P, S = pts_ids.shape[-2:]
    device=pts.device

    # ...xPxSxF
    pts_sampled = batched_index_select(index=pts_ids.flatten(-2), input=pts).view(pts_ids.shape + (-1,))

    # ...xPxSxR
    dist_sampled = batched_index_select(index=pts_ids.flatten(-2), input=dist_ref).view(pts_ids.shape + (-1,))
    # nearest neighbor correspondences
    pts_ref_ids = dist_sampled.argmin(dim=-1)
    dist_sampled_ref, pts_ref_ids = dist_sampled.min(dim=-1)

    dist_sampled_ref_inf_mask = dist_sampled_ref == torch.inf
    pts_ref_ids[dist_sampled_ref_inf_mask] = (torch.rand(size=(dist_sampled_ref_inf_mask.sum(),)) * R).to(dtype=int, device=device)

    # random correspondences, worse results
    #pts_ref_sample_probs = torch.ones(size=batch_dims + (P, R)).to(device=device)
    #pts_ref_ids = torch.multinomial(pts_ref_sample_probs.view(-1, R), num_samples=S).view(batch_dims + (P, S))

    pts_ref_sampled = batched_index_select(index=pts_ref_ids.flatten(-2), input=pts_ref).view(pts_ref_ids.shape + (-1,))

    ## start single batch dimension (= flatten proposals)
    #
    # S = corresponding_points_alignment(pts_sampled.view(-1, S, F), pts_ref_sampled.view(-1, S, F), weights=None,
    #                                    estimate_scale=True)
    #
    # pts_ref_tform4x4_pts = torch.zeros(size=(pts_sampled.shape[:-2].numel(), 4, 4)).to(
    #     device=device)
    # pts_ref_tform4x4_pts[..., 3, 3] = 1.
    # pts_ref_tform4x4_pts[..., :3, :3] = S.R.permute(0, 2, 1) * S.s[..., None, None]
    # pts_ref_tform4x4_pts[..., :3, 3] = S.T

    pts_ref_tform4x4_pts = fit_tform4x4_with_matches3d3d(pts=pts_sampled, pts_ref=pts_ref_sampled, estimate_scale=True)

    pts_ref_tform_pts_sampled = transf3d_broadcast(pts3d=pts_sampled, transf4x4=pts_ref_tform4x4_pts[:, None])

    ##logger.info(
    #    f'sampled geometrical error before transform: \n{(pts_sampled - pts_ref_sampled).norm(dim=-1).mean(dim=1).mean()}')
    #logger.info(
    #    f'sampled geometrical error after transform: \n{(pts_ref_tform_pts_sampled - pts_ref_sampled).norm(dim=-1).mean(dim=1).mean()}')

    # problem of zeros is ill posed problem if from all 4 sampled points the nearest neighbor is the same.
    mask_pts_ref_tform4x4_pts_zeros = pts_ref_tform4x4_pts[:, :3, :3].flatten(1).sum(dim=-1) == 0.
    pts_ref_tform4x4_pts[mask_pts_ref_tform4x4_pts_zeros, :3, :3] = torch.eye(3)[None,].expand(
        mask_pts_ref_tform4x4_pts_zeros.sum(), 3, 3).to(device=device)

    ## end single batch dimension
    pts_ref_tform4x4_pts = pts_ref_tform4x4_pts.view(pts_sampled.shape[:-2] + (4, 4))

    return pts_ref_tform4x4_pts


def score_tform4x4_fit(pts: torch.Tensor, tform4x4: torch.Tensor, pts_ref: torch.Tensor, dist_ref: torch.Tensor, return_dists=False, use_appear_argmin=False, dist_appear_weight=0.5, score_perc=1.):
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
    R = dist_ref.shape[-1]
    P = tform4x4.shape[-3]
    device = pts.device

    proposal_tform_pts = transf3d_broadcast(pts3d=pts[None,], transf4x4=tform4x4[:, None])
    norm_p = 1

    # PxNxR
    # dist_ref_geometry = (proposal_tform_pts[:, :, None] - pts_ref[None, None,]).norm(dim=-1)
    dist_ref_geometry = torch.cdist(proposal_tform_pts, pts_ref[None,], p=norm_p)  #
    dist_ref_geo_max = torch.cdist(pts_ref[None,], pts_ref[None,], p=norm_p).max()  #

    #if (~dist_ref_geometry.isfinite()).any():
    #    logger.warning(f'There are some infinite vlaues in dist geometry. WHY?')


    if not use_appear_argmin:
        # PxN
        proposal_tform_pts_nn_ref_id = dist_ref_geometry.argmin(dim=-1)
        proposal_tform_pts_id = torch.arange(proposal_tform_pts_nn_ref_id.shape[-1]).view(1, -1).\
            expand(proposal_tform_pts_nn_ref_id.shape).to(device=device)

        # PxR
        proposal_tform_pts_ref_nn_pts_id = dist_ref_geometry.argmin(dim=-2)
        proposal_tform_pts_ref_id = torch.arange(proposal_tform_pts_ref_nn_pts_id.shape[-1]).view(1, -1).\
            expand(proposal_tform_pts_ref_nn_pts_id.shape).to(device=device)

    else:
        argmin_ref_from_src = dist_ref.argmin(dim=-1) # N,
        argmin_src_from_ref = dist_ref.argmin(dim=-2) # R,

        src_cyclic_dist = (pts - pts[argmin_src_from_ref[argmin_ref_from_src]]).norm(dim=-1)
        ref_cyclic_dist = (pts_ref - pts_ref[argmin_ref_from_src[argmin_src_from_ref]]).norm(dim=-1)

        # PxN?
        proposal_tform_pts_nn_ref_id = argmin_ref_from_src
        proposal_tform_pts_nn_ref_id = proposal_tform_pts_nn_ref_id[None].expand(P, proposal_tform_pts_nn_ref_id.shape[-1]).contiguous()
        proposal_tform_pts_id = torch.arange(proposal_tform_pts_nn_ref_id.shape[-1]).view(1, -1). \
            expand(proposal_tform_pts_nn_ref_id.shape).to(device=device)

        src_cyclic_mask = src_cyclic_dist <= src_cyclic_dist.quantile(q=0.1)
        proposal_tform_pts_nn_ref_id = proposal_tform_pts_nn_ref_id[:, src_cyclic_mask]
        proposal_tform_pts_id = proposal_tform_pts_id[:, src_cyclic_mask]

        # PxR?
        proposal_tform_pts_ref_nn_pts_id = argmin_src_from_ref
        proposal_tform_pts_ref_nn_pts_id = proposal_tform_pts_ref_nn_pts_id[None].expand(P, proposal_tform_pts_ref_nn_pts_id.shape[-1]).contiguous()
        proposal_tform_pts_ref_id = torch.arange(proposal_tform_pts_ref_nn_pts_id.shape[-1]).view(1, -1). \
            expand(proposal_tform_pts_ref_nn_pts_id.shape).to(device=device)

        ref_cyclic_mask = ref_cyclic_dist <= ref_cyclic_dist.quantile(q=0.1)
        proposal_tform_pts_ref_nn_pts_id = proposal_tform_pts_ref_nn_pts_id[:, ref_cyclic_mask]
        proposal_tform_pts_ref_id = proposal_tform_pts_ref_id[:, ref_cyclic_mask]

    # PxNx2
    proposal_tform_pts_nn_ref_id_2D = torch.stack([proposal_tform_pts_id, proposal_tform_pts_nn_ref_id], dim=-1)
    proposal_tform_pts_nn_ref_id_2D = proposal_tform_pts_nn_ref_id_2D.clone()
    # PxRx2
    proposal_tform_pts_ref_nn_pts_id_2D = torch.stack([proposal_tform_pts_ref_id, proposal_tform_pts_ref_nn_pts_id], dim=-1)
    proposal_tform_pts_ref_nn_pts_id_2D = proposal_tform_pts_ref_nn_pts_id_2D.flip(dims=[-1])
    proposal_tform_pts_ref_nn_pts_id_2D = proposal_tform_pts_ref_nn_pts_id_2D.clone()

    # PxNxR
    # dist_ref_appearance = dist_ref[None,].expand(*dist_ref_geometry.shape)

    argmin_ref_from_src = dist_ref.argmin(dim=-1)  # N,
    argmin_src_from_ref = dist_ref.argmin(dim=-2)  # R,
    src_cyclic_dist = (pts - pts[argmin_src_from_ref[argmin_ref_from_src]]).norm(dim=-1, p=norm_p)
    ref_cyclic_dist = (pts_ref - pts_ref[argmin_ref_from_src[argmin_src_from_ref]]).norm(dim=-1, p=norm_p)
    cyclic_dist_avg = (src_cyclic_dist[:, None] + ref_cyclic_dist[None,]) / 2.
    cyclic_dist_avg = cyclic_dist_avg[None,].expand(*dist_ref_geometry.shape)
    alpha = 1
    dist_ref_appearance = (dist_ref_geometry.clone() / (dist_ref_geo_max))
    dist_ref_appearance_weight = torch.exp(-alpha * cyclic_dist_avg / dist_ref_geo_max)
    dist_ref_appearance_weight = dist_ref_appearance_weight / dist_ref_appearance_weight.flatten(-2).mean(dim=-1)[..., None, None]
    dist_ref_appearance = dist_ref_appearance_weight * dist_ref_appearance

    # cyclic_dist_avg = cyclic_dist_avg / dist_ref_geo_max
    # dist_ref_geometry = (dist_ref_geometry / dist_ref_geo_max).clamp(0, 1)
    # dist_ref_appearance = ((dist_ref_geometry.clone() / (cyclic_dist_avg + 0.1)) / 10.).clamp(0, 1)
    dist_ref_geometry = (dist_ref_geometry.clone() / (dist_ref_geo_max))
    #dist_ref_geometry = dist_ref_appearance

    # forward+backward nn
    proposal_pts_nn_id_2D = torch.cat([proposal_tform_pts_nn_ref_id_2D, proposal_tform_pts_ref_nn_pts_id_2D], dim=1)
    # backward nn
    #proposal_pts_nn_id_2D = proposal_tform_pts_ref_nn_pts_id_2D
    # forward nn
    #proposal_pts_nn_id_2D = proposal_tform_pts_nn_ref_id_2D


    proposal_dist_ref_appear = batched_indexMD_select(indexMD=proposal_pts_nn_id_2D, inputMD=dist_ref_appearance)
    proposal_dist_ref_geometry = batched_indexMD_select(indexMD=proposal_pts_nn_id_2D, inputMD=dist_ref_geometry)

    proposal_dist_ref_appear = proposal_dist_ref_appear.nan_to_num(1., posinf=1., neginf=1.)

    proposal_dist_ref_appear_pointwise_vals, proposal_dist_ref_appear_pointwise_ids = proposal_dist_ref_appear.sort(dim=-1, descending=False)
    N_score = int(proposal_dist_ref_appear.shape[-1] * score_perc)
    proposal_dist_ref_appear = batched_index_fill(input=proposal_dist_ref_appear, value=0.,  index=proposal_dist_ref_appear_pointwise_ids[..., N_score:])

    proposal_scores_pointwise = -((1.-dist_appear_weight) * proposal_dist_ref_geometry + dist_appear_weight * proposal_dist_ref_appear)

    proposal_scores = proposal_scores_pointwise.mean(dim=-1)
    #proposal_scores = proposal_scores_pointwise_vals[..., :N_score].mean(dim=-1)

    proposal_dist_ref_geo_avg = proposal_dist_ref_geometry.mean(dim=-1)
    proposal_dist_ref_appear_avg = proposal_dist_ref_appear.mean(dim=-1)
    #proposal_dist_ref_geo_avg = batched_index_select(input=proposal_dist_ref_geometry, index=proposal_scores_pointwise_ids)[..., :].mean(dim=-1)
    #proposal_dist_ref_appear_avg = batched_index_select(input=proposal_dist_ref_appear, index=proposal_scores_pointwise_ids)[..., :].mean(dim=-1)

    # mask_finite = proposal_dist_ref_appear.isfinite()
    # proposal_dist_ref_appear_avg = (proposal_dist_ref_appear.nan_to_num(1., posinf=1., neginf=1.).mean(dim=-1)) # * mask_finite).sum(dim=-1) / ((mask_finite).sum(dim=-1))
    # proposal_dist_ref_appear_avg = proposal_dist_ref_appear_avg.nan_to_num(1., posinf=1., neginf=1.)
    #
    # proposal_dist_ref_geo_avg = proposal_dist_ref_geometry.mean(dim=-1)

    # did not show improvements
    #proposal_dist_ref_geo_avg = (proposal_dist_ref_geometry.nan_to_num(0., posinf=0., neginf=0.) * mask_finite).sum(dim=-1) / ((mask_finite).sum(dim=-1))
    #proposal_dist_ref_geo_avg = proposal_dist_ref_geo_avg.nan_to_num(1., posinf=1., neginf=1.)

    # did not show improvements
    # scores = -proposal_dist_ref.quantile(q=0.9, dim=-1) #  (proposal_dist_ref * propsoal_dist_ref_isfinite).sum(dim=-1) / (propsoal_dist_ref_isfinite.sum(dim=-1) + 1e-10)
    #scores = -proposal_dist_ref.mean(dim=-1)  # (proposal_dist_ref * propsoal_dist_ref_isfinite).sum(dim=-1) / (propsoal_dist_ref_isfinite.sum(dim=-1) + 1e-10)

    #proposal_scores = -((1.-dist_appear_weight) * proposal_dist_ref_geo_avg + dist_appear_weight * proposal_dist_ref_appear_avg)
    if return_dists:
        return proposal_dist_ref_geo_avg, proposal_dist_ref_appear_avg
    else:
        return proposal_scores