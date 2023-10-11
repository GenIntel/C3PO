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
from pathlib import Path
from typing import List
import torchvision
import open3d as o3d
import numpy as np

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

from typing import Union
from od3d.cv.geometry.mesh import Meshes
from od3d.cv.visual.draw import get_colors
import open3d

def show_scene(cams_tform4x4_world: Union[torch.Tensor, List[torch.Tensor]]=None,
               cams_intr4x4: Union[torch.Tensor, List[torch.Tensor]]=None,
               cams_names: List[str]=None,
               pts3d: Union[torch.Tensor, List[torch.Tensor]]=None,
               pts3d_names: List[str]=None,
               pts3d_colors: Union[torch.Tensor, List]=None,
               meshes: Meshes=None,
               meshes_names: List[str]=None,
               meshes_colors: Union[torch.Tensor, List]=None,
               meshes_add_translation: bool=True,
               fpath: Path=None,
               return_visualization=False,
               viewpoints_count=1):
    """
    Args:
        cams_tform4x4_world (Union[torch.Tensor, List[torch.Tensor]]): (Cx4x4) or List(4x4)
        cams_intr4x4 (Union[torch.Tensor, List[torch.Tensor]]): Cx4x4 or List(4x4)
        pts3d (Union[torch.Tensor, List[torch.Tensor]]): PxNx3 or List(Npx3)
        pts3d_names (List[str]): (P,)
        pts3d_colors (Union[torch.Tensor, List]): Px3 or List(3)
        meshes (Meshes)
        meshes_names (List[str]): (M,)
        meshes_colors (Union[torch.Tensor, List]): Mx3 or List(3)

    Returns:
        -
    """

    geometries = []

    if meshes is not None:

        x_offset = 0.
        for i in range(len(meshes)):
            vertices = meshes.get_verts_with_mesh_id(mesh_id=i).clone()
            if meshes_add_translation:
                x_offset_delta_current = 1.1 * (-vertices[:, 0].min()).clamp(min=0.)
                x_offset += x_offset_delta_current
                x_offset_delta_next = 1.1 * (vertices[:, 0].max())
                vertices[:, 0] += x_offset
                x_offset += x_offset_delta_next

            vertices = open3d.utility.Vector3dVector(vertices.detach().cpu().numpy())
            triangles = open3d.utility.Vector3iVector(meshes.get_faces_with_mesh_id(mesh_id=i).detach().cpu().numpy())

            mesh_o3d = open3d.geometry.TriangleMesh(vertices=vertices, triangles=triangles)
            if meshes.rgb is not None:
                vertex_colors = open3d.utility.Vector3dVector(meshes.get_rgb_with_mesh_id(mesh_id=i).detach().cpu().numpy())
                mesh_o3d.vertex_colors = vertex_colors

            else:
                vertex_colors = None
            #vertex_colors
            #vertex_normals

            if meshes_colors is not None and len(meshes_colors) >= i+1 and meshes_colors[i] is not None:
                mesh_color = meshes_colors[i]
            else:
                mesh_color = get_colors(len(meshes))[i]

            if meshes_names is not None and len(meshes_names) >= i+1 and meshes_names[i] is not None:
                mesh_name = meshes_names[i]
            else:
                mesh_name = f'mesh{i}'

            if isinstance(mesh_color, torch.Tensor):
                mesh_color =mesh_color.detach().cpu().numpy()
            mat_box = open3d.visualization.rendering.MaterialRecord()
            mat_box.shader = 'defaultLitTransparency'
            #mat_box.shader = 'defaultLitSSR'

            if vertex_colors is None:
                if len(mesh_color) == 4:
                    mat_box.base_color = [mesh_color[0], mesh_color[1], mesh_color[2], mesh_color[3]]
                else:
                    mat_box.base_color = [mesh_color[0], mesh_color[1], mesh_color[2], 0.9] # [0.467, 0.467, 0.467, 0.02]
            else:
                mat_box.base_color = [0.5, 0.5, 0.5, 0.9]  # [0.467, 0.467, 0.467, 0.02]

            #mat_box.base_roughness = 0.0
            #mat_box.base_reflectance = 0.0
            #mat_box.base_clearcoat = 1.0
            #mat_box.thickness = 1.0
            #mat_box.transmission = 1.0
            #mat_box.absorption_distance = 10
            #mat_box.absorption_color = [0.5, 0.5, 0.5]

            geometries.append({'name': mesh_name, 'geometry': mesh_o3d, 'material': mat_box})
            #vertices: open3d.cpu.pybind.utility.Vector3dVector,
            #triangles: open3d.cpu.pybind.utility.Vector3iVector


    if pts3d is not None:
        for i, pts3d_i in enumerate(pts3d):
            pts3d_i_o3d = open3d.geometry.PointCloud()
            pts3d_i_o3d.points = open3d.utility.Vector3dVector(pts3d_i.detach().cpu().numpy())

            if pts3d_colors is not None and len(pts3d_colors) >= i+1 and pts3d_colors[i] is not None:
                pts3d_i_color = pts3d_colors[i]
            else:
                pts3d_i_color = get_colors(len(pts3d))[i]

            if isinstance(pts3d_i_color, list) or pts3d_i_color.dim() == 1:
                pts3d_i_o3d.paint_uniform_color((pts3d_i_color[0], pts3d_i_color[1], pts3d_i_color[2]))
            else:
                pts3d_i_o3d.colors = o3d.utility.Vector3dVector(pts3d_i_color.detach().cpu().numpy())
            if pts3d_names is not None and len(pts3d_names) >= i+1 and pts3d_names[i] is not None:
                pts3d_i_name = pts3d_names[i]
            else:
                pts3d_i_name = f'pts3d_{i}'

            geometries.append({'name': pts3d_i_name, 'geometry': pts3d_i_o3d})

    if cams_tform4x4_world is not None and cams_intr4x4 is not None:

        for i, cam_tform4x4_obj in enumerate(cams_tform4x4_world):
            width = int(cams_intr4x4[0, 2] * 2)
            height = int(cams_intr4x4[1, 2] * 2)

            if len(cams_intr4x4) == 1:
                cam_intr4x4 = cams_intr4x4[0]
            else:
                cam_intr4x4 = cams_intr4x4[i]

            cam = open3d.geometry.LineSet.create_camera_visualization(view_width_px=width, view_height_px=height,
                                                                      intrinsic=cam_intr4x4[i][:3, :3].detach().numpy(),
                                                                      extrinsic=cam_tform4x4_obj.detach().cpu().numpy(),
                                                                      scale=0.01)
            if cams_names is not None and len(cams_names) >= i+1 and cams_names[i] is not None:
                cam_name = cams_names[i]
            else:
                cam_name = f'cam{i}'

            geometries.append({'name': cam_name, 'geometry': cam})

    if return_visualization is False and fpath is None:
        open3d.visualization.draw(geometries)
    else:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False) #, height=720, width=1280)

        geometries_vertices_orig = []

        for geometry in geometries:
            vis.add_geometry(geometry['geometry'])
            if isinstance(geometry['geometry'], open3d.geometry.PointCloud):
                geometries_vertices_orig.append(torch.from_numpy(np.asarray(geometry['geometry'].points)).clone())
            elif isinstance(geometry['geometry'], open3d.geometry.TriangleMesh):
                geometries_vertices_orig.append(torch.from_numpy(np.asarray(geometry['geometry'].vertices)).clone())
            else:
                geometries_vertices_orig.append(None)
            #vis.update_geometry(geometry['geometry'])
        vis.poll_events()
        vis.update_renderer()
        view_control = vis.get_view_control()
        if return_visualization:
            imgs = []

            from od3d.cv.geometry.transform import get_cam_tform4x4_obj_for_viewpoints_count, transf3d, tform4x4_broadcast
            objs_new_tform4x4_obj = get_cam_tform4x4_obj_for_viewpoints_count(viewpoints_count=viewpoints_count, dist=0.)
            # open3d version 0.17.0 bug, view control does not work
            #camera_orig = view_control.convert_to_pinhole_camera_parameters()
            #cam_tform4x4_obj = torch.from_numpy(camera_orig.extrinsic).to(dtype=objs_new_tform4x4_obj.dtype, device=objs_new_tform4x4_obj.device)

            for v in range(viewpoints_count):
                vis.update_renderer()
                img = vis.capture_screen_float_buffer(do_render=True)
                img = torch.from_numpy(np.array(img)).permute(2, 0, 1)
                imgs.append(img)

                #camera_orig.extrinsic = tform4x4_broadcast(cam_tform4x4_obj,
                #                                           objs_new_tform4x4_obj[v]).detach().cpu().numpy()
                #view_control.convert_from_pinhole_camera_parameters(camera_orig)

                for g, geometry in enumerate(geometries):
                    if geometries_vertices_orig[g] is not None:
                        vertices = geometries_vertices_orig[g].to(dtype=objs_new_tform4x4_obj.dtype, device=objs_new_tform4x4_obj.device)
                        vertices = transf3d_broadcast(pts3d=vertices, transf4x4=objs_new_tform4x4_obj[v])

                        if isinstance(geometry['geometry'], open3d.geometry.PointCloud):
                            geometry['geometry'].points = open3d.utility.Vector3dVector(vertices.detach().cpu().numpy())
                        elif isinstance(geometry['geometry'], open3d.geometry.TriangleMesh):
                            geometry['geometry'].vertices = open3d.utility.Vector3dVector(vertices.detach().cpu().numpy())
                        vis.update_geometry(geometry['geometry'])

                vis.update_renderer()
            if viewpoints_count == 1:
                imgs = imgs[0]
            return imgs
        else:
            vis.capture_screen_image(str(fpath))


        vis.update_renderer()
        vis.destroy_window()

    # open3d.visualization.draw_geometries(geometries)

def show_pcl_via_open3d(pts3d):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    # vis.add_geometry(pcd)

    pcd = o3d.geometry.PointCloud()
    # from od3d.cv.geometry.transform import inv_tform4x4
    pcd.points = o3d.utility.Vector3dVector(pts3d.numpy())
    # pcd.colors = o3d.utility.Vector3dVector(ncds.numpy())
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()

def show_open3d_pcl(pcd):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()

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


def fpaths_to_rgb(fpaths: List[Path], H: int, W: int):

    rgbs = torch.stack([resize(torchvision.io.read_image(path=str(fpath)), H_out=H, W_out=W) for fpath in fpaths], dim=0)
    rgb = imgs_to_img(rgbs)
    return rgb

def show_imgs(rgbs, duration=0, vwriter=None, fpath=None, height=None, width=None):
    rgb = imgs_to_img(rgbs)
    return show_img(rgb, duration, vwriter, fpath, height, width)

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
        Path(fpath).parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(fpath), img)
    else:
        logging.basicConfig(level=logging.DEBUG)
        cv2.imshow("img", img)
        return cv2.waitKey(duration)


def get_img_from_plot(ax, fig, axis_off=True):

    # Image from plot
    if axis_off:
        ax.axis('off')
        # To remove the huge white borders
        ax.margins(0)
        fig.tight_layout(pad=0)
    else:
        fig.tight_layout(pad=1)
        ax.margins(1)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot: np.ndarray = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return torch.from_numpy(image_from_plot.copy()).permute(2, 0, 1)