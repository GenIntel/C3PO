import logging
logger = logging.getLogger(__name__)
import os
import cv2
from od3d.cv.visual.draw import tensor_to_cv_img
from od3d.cv.visual.resize import resize
import torch
import math
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene, AxisArgs
from pytorch3d.renderer.cameras import PerspectiveCameras
from od3d.cv.geometry.transform import transf3d_broadcast
from od3d.cv.visual.draw import get_colors

def pt3d_camera_from_tform4x4_intr4x4_imgs_size(cam_tform4x4_obj: torch.Tensor, cam_intr4x4: torch.Tensor, img_size: torch.Tensor):
    if cam_tform4x4_obj.dim() == 2:
        cam_tform4x4_obj = cam_tform4x4_obj[None,]
        cam_intr4x4 = cam_intr4x4[None, ]
        img_size = img_size[None, ]
    t3d_tform_default = torch.Tensor([[-1., 0., 0., 0.],
                                      [0., -1., 0., 0.],
                                      [0., 0., 1., 0.],
                                      [0., 0., 0., 1.]]).to(device=cam_tform4x4_obj.device,
                                                            dtype=cam_tform4x4_obj.dtype)

    cam_tform4x4_obj = torch.bmm(t3d_tform_default[None,], cam_tform4x4_obj)
    focal_length = torch.stack([cam_intr4x4[:, 0, 0], cam_intr4x4[:, 1, 1]], dim=1)
    principal_point = torch.stack([cam_intr4x4[:, 0, 2], cam_intr4x4[:, 1, 2]], dim=1)

    R = cam_tform4x4_obj[:, :3, :3]
    t = cam_tform4x4_obj[:, :3, 3]
    cameras = PerspectiveCameras(R=R.transpose(-2, -1), T=t, focal_length=focal_length,
                                 principal_point=principal_point, in_ndc=False,
                                 image_size=img_size, device=cam_tform4x4_obj.device)

    return cameras



def show_mesh():
    raise NotImplementedError
    """
    from pytorch3d.structures.meshes import Meshes as PT3DMeshes
    meshes = PT3DMeshes(verts=[verts], faces=[faces])
    from pytorch3d.vis.plotly_vis import plot_scene, AxisArgs
    fig = plot_scene({
        "Meshes": {
            f"mesh{i+1}": meshes[i] for i in range(len(meshes))
        }}, axis_args=AxisArgs(backgroundcolor="rgb(200, 200, 230)", showgrid=True, zeroline=True, showline=True,
                          showaxeslabels=True, showticklabels=True))
    fig.show()
    input('bla')
    """


def show_pcl(verts, cam_tform4x4_obj: torch.Tensor=None, cam_intr4x4: torch.Tensor=None, img_size: torch.Tensor=None):
    """

    Args:
        verts: Nx3 / BxNx3 / list(torch.Tensor Nx3)

    """

    if cam_tform4x4_obj is not None:

        pt3d_cameras = pt3d_camera_from_tform4x4_intr4x4_imgs_size(cam_tform4x4_obj=cam_tform4x4_obj, cam_intr4x4=cam_intr4x4, img_size=img_size)
    else:
        pt3d_cameras = []


    if isinstance(verts, list) or verts.dim() == 3:
        if isinstance(verts, list):
            B = len(verts)
            N, _ = verts[0].shape
            device = verts[0].device
        else:
            B, N, _ = verts.shape
            device = verts.device
        colors = get_colors(B, device=device)
        rgb = colors[:, None].repeat(1, N, 1)
    else:
        N, _ = verts.shape
        colors = get_colors(1, device=verts.device)
        rgb = colors.repeat(N, 1)
        rgb = rgb[None,]
        verts = verts[None,]
        #cls = cls[None,]

    """
    # o3d.camera.PinholeCameraIntrinsic(640, 480, 525, 525, 320, 240)
    rgb = rgb.reshape(-1, 3)
    verts = verts.reshape(-1, 3)
    cls = cls.reshape(-1, 3)
    import open3d as o3d
    import numpy as np
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    pcd.write_property('class', cls)
    geometries = [pcd]
    #o3d.visualization.draw_geometries([pcd],
    #                                  zoom=0.3412,
    #                                  front=[0.4257, -0.2125, -0.8795],
    #                                  lookat=[2.6172, 2.0475, 1.532],
    #                                  up=[-0.0694, -0.9768, 0.2024])
    viewer = o3d.visualization.Visualizer()
    o3d.visualization.gui.Label3D(color=[1., 0., 0.], position=[0., 0., 0.], scale=1., text='blub')
    viewer.create_window()
    for geometry in geometries:
        viewer.add_geometry(geometry)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.8, 0.8, 0.9])
    viewer.run()
    viewer.destroy_window()
    """

    point_cloud = Pointclouds(points=verts, features=rgb)
    fig = plot_scene({
        "Pointcloud": {**{
            f"pcl{i+1}": point_cloud[i] for i in range(len(point_cloud))
        },
        **{
            f"cam{i+1}": pt3d_cameras[i] for i in range(len(pt3d_cameras))
        },
        }
    }, viewpoint_cameras=pt3d_cameras, axis_args=AxisArgs(backgroundcolor="rgb(200, 200, 230)", showgrid=True, zeroline=True, showline=True,
                          showaxeslabels=True, showticklabels=True))
    fig.show()
    input('bla')


def imgs_to_img(rgbs):
    # rgb: K x 3 x H x W / GH x GW x 3 x H x W

    # , masks_mulitply=None, masks_overlay=None
    #rgbs = (rgbs * 1.0).clamp(0, 1)
    #if masks is not None:
    #    rgb = (rgbs + masks) / 2.0

    rgbs = torch.nn.functional.pad(rgbs, (1, 1, 1, 1), "constant", 1.0)
    #margin = 2
    #torch.nn.functional.pad(rgbs, (1, 1), "constant", 0)

    rgbs_shape = rgbs.shape

    if len(rgbs_shape) == 5:
        GH, GW, C, H, W = rgbs_shape
        rgb = rgbs.clone()
    elif len(rgbs_shape) == 4:
        K, C, H, W = rgbs.shape
        prop_w = 4
        prop_h = 3
        GW = math.ceil(math.sqrt((K * prop_w ** 2) / prop_h ** 2))
        GH = math.ceil(K / GW)
        GTOTAL = GH * GW

        img_placeholder = torch.zeros_like(rgbs[:1]).repeat(GTOTAL - K, 1, 1, 1)

        rgb = torch.cat((rgbs, img_placeholder), dim=0)

        rgb = rgb.reshape(GH, GW, C, H, W)
    else:
        logger.error('Visualize imgs requires the input rgb tensor to have 4 (KxCxHxW) or 5 (GHxGWxCxHxW) dimensions')
        raise NotImplementedError

    rgb = rgb.permute(2, 0, 3, 1, 4)

    rgb = rgb.reshape(C, GH * H, GW * W)

    return rgb

def show_imgs(rgbs, duration=0, vwriter=None, fpath=None, height=None, width=None):
    rgb = imgs_to_img(rgbs)
    show_img(rgb, duration, vwriter, fpath, height, width)

def show_img(rgb, duration=0, vwriter=None, fpath=None, height=None, width=None, normalize=False):
    # img: 3xHxW
    rgb = rgb.clone()

    if normalize:
        rgb = (rgb -rgb.min()) / (rgb.max() - rgb.min())

    if width is not None:
        orig_width = rgb.size(2)
        scale_factor = width / orig_width

    elif height is not None:
        orig_height = rgb.size(1)
        scale_factor = height / orig_height

    if width or height is not None:
        rgb = resize(
            rgb[
                None,
            ],
            scale_factor=scale_factor,
        )[0]

    img = tensor_to_cv_img(rgb)

    if vwriter is not None:
        vwriter.write(img)

    if fpath is not None:
        if not os.path.exists(os.path.dirname(fpath)):
            os.makedirs(os.path.dirname(fpath))
        cv2.imwrite(str(fpath), img)
    else:
        cv2.imshow("img", img)
        cv2.waitKey(duration)