import logging
logger = logging.getLogger(__name__)
import open3d as o3d
import torch

def label_axis_in_pcl(pts3d, pts3d_colors=None, prev_axis=None):
    """
    Args:
        pts3d (torch.Tensor): Nx3
        pts3d_colors (torch.Tensor): Nx3
        prev_axis (torch.Tensor): 3x2x3
    Returns:
        axis (torch.Tensor): 3x2x3
    """

    logger.info("")
    logger.info(
        "1) Please pick left, right, back, front, top, bottom [shift + left click]"
    )
    logger.info("   Press [shift + right click] to undo point picking")
    logger.info("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()

    pcd = o3d.geometry.PointCloud()

    # Set the point cloud data
    pcd.points = o3d.utility.Vector3dVector(pts3d.detach().cpu().numpy())

    if pts3d_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(pts3d_colors.detach().cpu().numpy())

    if prev_axis is not None:
        pcd_prev_axis = o3d.geometry.PointCloud()
        pcd_prev_axis.points = o3d.utility.Vector3dVector(prev_axis.reshape(-1, 3).detach().cpu().numpy())
        prev_axis_colors = torch.Tensor([[[1., 0., 0.], [1., 0., 0.]], [[0., 1., 0.], [0., 1., 0.]], [[0., 0., 1.], [0., 0., 1.]]])
        pcd_prev_axis.colors = o3d.utility.Vector3dVector(prev_axis_colors.reshape(-1, 3).detach().cpu().numpy())
        vis.add_geometry(pcd + pcd_prev_axis)
        pts3d_selectable = torch.cat([pts3d.reshape(-1, 3), prev_axis.reshape(-1, 3)], dim=0)
    else:
        vis.add_geometry(pcd)
        pts3d_selectable = pts3d

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
