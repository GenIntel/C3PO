import logging
logger = logging.getLogger(__name__)
import open3d
import torch
from typing import Union, List
from od3d.cv.visual.show import get_o3d_geometries_for_cams


def label_axis_in_pcl(pts3d, pts3d_colors=None, prev_axis=None,
                      cams_tform4x4_world: Union[torch.Tensor, List[torch.Tensor]]=None,
                      cams_intr4x4: Union[torch.Tensor, List[torch.Tensor]]=None,
                      cams_imgs: Union[torch.Tensor, List[torch.Tensor]]=None,
                      cams_names: List[str]=None,):
    """
    Args:
        pts3d (torch.Tensor): Nx3
        pts3d_colors (torch.Tensor): Nx3
        prev_axis (torch.Tensor): 3x2x3
        cams_tform4x4_world (Union[torch.Tensor, List[torch.Tensor]]): (Cx4x4) or List(4x4)
        cams_intr4x4 (Union[torch.Tensor, List[torch.Tensor]]): Cx4x4 or List(4x4)
        cams_imgs (Union[torch.Tensor, List[torch.Tensor]]): Cx3xHxW or List(3xHxW)
        cams_names: (List[str]): (P,)
    Returns:
        axis (torch.Tensor): 3x2x3
    """

    logger.info("")
    logger.info(
        "1) Please pick left, right, back, front, top, bottom [shift + left click]"
    )
    logger.info("   Press [shift + right click] to undo point picking")
    logger.info("2) Afther picking points, press q for close the window")
    vis = open3d.visualization.VisualizerWithEditing()
    vis.create_window()

    pcd = open3d.geometry.PointCloud()

    # Set the point cloud data
    pcd.points = open3d.utility.Vector3dVector(pts3d.detach().cpu().numpy())

    if pts3d_colors is not None:
        pcd.colors = open3d.utility.Vector3dVector(pts3d_colors.detach().cpu().numpy())

    if prev_axis is not None:
        pcd_prev_axis = open3d.geometry.PointCloud()
        pcd_prev_axis.points = open3d.utility.Vector3dVector(prev_axis.reshape(-1, 3).detach().cpu().numpy())
        prev_axis_colors = torch.Tensor([[[1., 0., 0.], [1., 0., 0.]], [[0., 1., 0.], [0., 1., 0.]], [[0., 0., 1.], [0., 0., 1.]]])
        pcd_prev_axis.colors = open3d.utility.Vector3dVector(prev_axis_colors.reshape(-1, 3).detach().cpu().numpy())
        pcd = pcd + pcd_prev_axis
        pts3d_selectable = torch.cat([pts3d.reshape(-1, 3), prev_axis.reshape(-1, 3)], dim=0)
    else:
        pts3d_selectable = pts3d

    for geometry_dict in get_o3d_geometries_for_cams(cams_tform4x4_world=cams_tform4x4_world,
                                                     cams_intr4x4=cams_intr4x4,
                                                     cams_imgs=cams_imgs,
                                                     cams_names=cams_names):
        if isinstance(geometry_dict['geometry'], open3d.geometry.PointCloud):
            pcd += geometry_dict['geometry']

    vis.add_geometry(pcd)


    vis.run()  # user picks points
    vis.destroy_window()
    logger.info("")
    pts3d_picked_ids = vis.get_picked_points()

    if len(pts3d_picked_ids) != 6:
        if prev_axis is not None:
            logger.warning("Return prev axis, because not exactly 6 points were selected")
            return prev_axis
        else:
            logger.warning("Return none, because not exactly 6 points were selected")
            return None
    else:
        pts3d_picked = pts3d_selectable[pts3d_picked_ids]
        return pts3d_picked.reshape(3, 2, 3)
