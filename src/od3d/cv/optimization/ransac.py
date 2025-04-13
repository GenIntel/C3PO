import logging

logger = logging.getLogger(__name__)
import torch
from od3d.cv.select import batched_index_select
from od3d.cv.geometry.transform import inv_tform4x4, tform4x4
from od3d.cv.metric.pose import get_pose_diff_in_rad

def sample_models(
    pts,
    fit_func,
    fit_pts_count: int,
    fits_count: int,
    pts_dist=None,
    pts_affinity=None,
):
    """
    Args:
        pts (torch.Tensor): ...xNxF
        fit_func: returns multiple fitted models for points and points ids, in: ...xNxF ...xPxS -> ...xPxM
        fit_pts_count (int): number of points required to fit a model (P)
        fits_count (int): number of proposed model fits (S)
        pts_affinity: ...xNxN
        pts_dist: ...xNxN
    Returns:
        models: ...xSx...
    """
    device = pts.device
    N = pts.shape[-2]
    batch_dims = pts.shape[:-2]

    if pts_dist is not None:
        if pts_affinity is None:
            pts_affinity = 1.0 / pts_dist
        else:
            logger.warning("ignoring `pts_dist` as `pts_affinity` is defined as well")

    if pts_affinity is None:
        pts_affinity = torch.ones(size=batch_dims + (N, N)).to(device=device)
    else:
        logger.warning("ignoring `pts_affinity` as it is not implemented yet")

    pts_sample_probs = torch.ones(size=batch_dims + (fits_count, N)).to(device=device)

    # ...xPxS
    pts_ids = torch.multinomial(
        pts_sample_probs.view(-1, N),
        num_samples=fit_pts_count,
    ).view(batch_dims + (fits_count, fit_pts_count))

    # ...xPxM
    models = fit_func(pts, pts_ids)

    return models


def ransac(
    pts,
    fit_func,
    score_func,
    fit_pts_count: int,
    fits_count: int,
    pts_dist=None,
    pts_affinity=None,
    return_score=False,
    ref_sph_feat = None,
    src_sph_feat = None,
    return_pts_id = False,
):
    """
    Args:
        pts (torch.Tensor): ...xNxF
        fit_func: returns multiple fitted models for points and points ids, in: ...xNxF ...xPxS -> ...xPxM
        score_func: return scores for multiple fitted models, in: ...xNxF, ...xPxM -> ...xP
        fit_pts_count (int): number of points required to fit a model (P)
        fits_count (int): number of proposed model fits (S)
        pts_affinity: ...xNxN
        pts_dist: ...xNxN
    Returns:
        model: returns best fit model
    """
    batch_dims = pts.shape[:-2]
    batch_dims_count = len(batch_dims)

    models, pts_ids, pts_ref_ids = sample_models(
        pts=pts,
        fit_func=fit_func,
        fit_pts_count=fit_pts_count,
        fits_count=fits_count,
        pts_dist=pts_dist,
        pts_affinity=pts_affinity,
    )
    # ...xP
    # dot_products = torch.einsum('mij,mij->m', src_sph_feat[pts_ids.cpu()], ref_sph_feat[pts_ref_ids.cpu()])
    # best_sph_id = dot_products.argmax(dim = -1)

    scores,proposal_dist_ref_geo_avg,proposal_dist_ref_appear_avg, proposal_dist_ref_geometry_weight,proposal_dist_ref_appear_weight = score_func(pts, models,  return_dists=True, return_weights=True,)
    
    # B...
    best_id = scores.argmax(dim=-1)
    # best_id = best_model_id

    best_model = batched_index_select(input=models, index=best_id[..., None]).squeeze(
        dim=batch_dims_count,
    )
    best_score = batched_index_select(input=scores, index=best_id[..., None]).squeeze(
        dim=batch_dims_count,
    )
    best_correspondence = pts_ids[best_id]
    best_ref_correspondence = pts_ref_ids[best_id]
    
    best_geo_dist = proposal_dist_ref_geo_avg[best_id]
    best_appear_dist = proposal_dist_ref_appear_avg[best_id]
    # print('src_sph_feat[best_correspondence.cpu()] ', src_sph_feat[best_correspondence.cpu()])
    # print('ref_sph_feat[best_ref_correspondence.cpu()] ', ref_sph_feat[best_ref_correspondence.cpu()])
    # if return_score:
    #     return best_model, best_score
    #else:
        #return best_model, models, scores, best_correspondence, best_ref_correspondence, best_geo_dist, best_appear_dist, best_score, models[best_sph_id]
    if return_pts_id:
        return best_model, models, scores, best_correspondence, best_ref_correspondence, best_geo_dist, best_appear_dist, best_score, pts_ids, pts_ref_ids, proposal_dist_ref_geo_avg, proposal_dist_ref_appear_avg
    else:
        return best_model, models, scores, best_correspondence, best_ref_correspondence, best_geo_dist, best_appear_dist, best_score