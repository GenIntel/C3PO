import logging

logger = logging.getLogger(__name__)
import os
import cv2
from od3d.cv.visual.draw import tensor_to_cv_img
from od3d.cv.visual.resize import resize
import torch
import math
from pytorch3d.structures import Pointclouds as PT3D_Pointclouds
from pytorch3d.structures import Meshes as PT3D_Meshes

from pytorch3d.vis.plotly_vis import plot_scene, AxisArgs
from pytorch3d.renderer.cameras import PerspectiveCameras
from od3d.cv.geometry.transform import transf3d_broadcast
from od3d.cv.visual.draw import get_colors
from pathlib import Path
from typing import List
import torchvision
import PIL

import open3d as o3d
import numpy as np
from od3d.cv.geometry.transform import tform4x4

DEFAULT_CAM_TFORM_OBJ = torch.Tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
)

OPEN3D_DEFAULT_CAM_TFORM_OBJ = torch.Tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
)

OBJ_TFORM_OPEN3D_DEFAULT_CAM = torch.Tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
)

OPEN3D_DEFAULT_TFORM_DEFAULT = torch.Tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
)

DEFAULT_TFORM_OPEN3D_DEFAULT = torch.Tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
)

# OPEN3D_DEFAULT_TFORM_DEFAULT = tform4x4(OPEN3D_DEFAULT_CAM_TFORM_OBJ, inv_tform4x4(DEFAULT_CAM_TFORM_OBJ))
# DEFAULT_TFORM_OPEN3D_DEFAULT = inv_tform4x4(OPEN3D_DEFAULT_TFORM_DEFAULT)

def get_default_camera_intrinsics_from_img_size(W, H, dtype=None, device=None):
    cam_intr4x4 = torch.eye(4).to(dtype=dtype, device=device) * (W + H) / 2
    cam_intr4x4[0, 2] = W / 2
    cam_intr4x4[1, 2] = H / 2
    return cam_intr4x4

def pt3d_camera_from_tform4x4_intr4x4_imgs_size(
    cam_tform4x4_obj: torch.Tensor,
    cam_intr4x4: torch.Tensor,
    img_size: torch.Tensor,
):
    if cam_tform4x4_obj.dim() == 2:
        cam_tform4x4_obj = cam_tform4x4_obj[None,]
    if cam_intr4x4.dim() == 2:
        cam_intr4x4 = cam_intr4x4[None,].expand(cam_tform4x4_obj.shape[0], 4, 4)
    if img_size.dim() == 1:
        img_size = img_size[None,].expand(cam_tform4x4_obj.shape[0], 2)

    t3d_tform_default = torch.Tensor(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    ).to(
        device=cam_tform4x4_obj.device,
        dtype=cam_tform4x4_obj.dtype,
    )

    from od3d.cv.geometry.transform import tform4x4_broadcast
    cam_tform4x4_obj = tform4x4_broadcast(t3d_tform_default[None,], cam_tform4x4_obj)
    #cam_tform4x4_obj = torch.bmm(t3d_tform_default[None,], cam_tform4x4_obj)
    focal_length = torch.stack([cam_intr4x4[:, 0, 0], cam_intr4x4[:, 1, 1]], dim=1)
    principal_point = torch.stack([cam_intr4x4[:, 0, 2], cam_intr4x4[:, 1, 2]], dim=1)
    #principal_point = torch.stack([img_size[:, 1] - cam_intr4x4[:, 0, 2], img_size[:, 0] - cam_intr4x4[:, 1, 2]], dim=1)

    R = cam_tform4x4_obj[:, :3, :3]
    t = cam_tform4x4_obj[:, :3, 3]
    cameras = PerspectiveCameras(
        R=R.transpose(-2, -1),
        T=t,
        focal_length=focal_length,
        principal_point=principal_point,
        in_ndc=False,
        image_size=img_size,
        device=cam_tform4x4_obj.device,
    )

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
from od3d.cv.geometry.mesh import Meshes, Mesh
from od3d.cv.visual.draw import get_colors
import open3d


def show_scene2d(
    pts2d: Union[torch.Tensor, List[torch.Tensor]] = None,
    pts2d_names: List[str] = None,
    pts2d_colors: Union[torch.Tensor, List] = None,
    pts2d_lengths: List[int] = None,
    return_visualization=False,
):
    """

    Args:
        pts2d (Union[torch.Tensor, List[torch.Tensor]]): PxNx2 or List(Npx2)
        pts2d_names (List[str]): (P,)
        pts2d_colors (Union[torch.Tensor, List]): Px2x3 or List(3)
    Returns:
        img: torch.Tensor

    """
    import matplotlib.pyplot as plt
    import itertools

    # matplotlib.use("TkAgg")
    if pts2d_lengths is not None:
        pts2d_lengths_sum = np.cumsum([0, *pts2d_lengths])

    # scatter plot 2d with legend and colors
    fig, ax = plt.subplots(len(pts2d), 1)
    # to make ax iterable
    ax = [ax] if len(pts2d) == 1 else ax
    if pts2d is not None:
        for i, pts2d_i in enumerate(pts2d):
            if pts2d_colors is not None:
                pts2d_colors_i = pts2d_colors[i]
            else:
                pts2d_colors_i = get_colors(len(pts2d))[i]
            if isinstance(pts2d_colors_i, torch.Tensor):
                c = pts2d_colors_i.detach().cpu().numpy()
            else:
                c = pts2d_colors_i
            if pts2d_lengths is not None:
                marker = itertools.cycle((".", "+", "v", "o", "*"))
                for j in range(len(pts2d_lengths)):
                    ax[i].scatter(
                        pts2d_i[pts2d_lengths_sum[j] : pts2d_lengths_sum[j + 1], 0]
                        .detach()
                        .cpu()
                        .numpy(),
                        pts2d_i[pts2d_lengths_sum[j] : pts2d_lengths_sum[j + 1], 1]
                        .detach()
                        .cpu()
                        .numpy(),
                        marker=next(marker),
                        c=c[pts2d_lengths_sum[j] : pts2d_lengths_sum[j + 1]],
                    )
            else:
                ax[i].scatter(
                    pts2d_i[:, 0].detach().cpu().numpy(),
                    pts2d_i[:, 1].detach().cpu().numpy(),
                    c=c,
                )
            if pts2d_names is not None:
                ax[i].set_title(pts2d_names[i])
            ax[i].set_axis_off()
    if return_visualization:
        fig.canvas.draw()
        img = PIL.Image.frombytes(
            "RGB",
            fig.canvas.get_width_height(),
            fig.canvas.tostring_rgb(),
        )
        img_tensor = torchvision.transforms.ToTensor()(img)

        return img_tensor
    plt.show()

from od3d.data import ExtEnum

class OD3D_RENDERER(str, ExtEnum):
    OPEN3D = "open3d"
    PYTORCH3D = "pytorch3d"

def plotly_fig_2_tensor(fig, width=None, height=None):
    from PIL import Image
    import io

    #convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="png", width=width, height=height)
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    img = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)[:3] / 255.
    return img

def show_scene(
    cams_tform4x4_world: Union[torch.Tensor, List[torch.Tensor]] = None,
    cams_intr4x4: Union[torch.Tensor, List[torch.Tensor]] = None,
    cams_imgs: Union[torch.Tensor, List[torch.Tensor]] = None,
    cams_names: List[str] = None,
    cams_imgs_resize: bool = True,
    cams_imgs_depth_scale: float = 0.2,
    cams_show_wireframe: bool = True,
    pts3d: Union[torch.Tensor, List[torch.Tensor]] = None,
    pts3d_names: List[str] = None,
    pts3d_colors: Union[torch.Tensor, List] = None,
    pts3d_normals: Union[torch.Tensor, List] = None,
    lines3d: Union[torch.Tensor, List[torch.Tensor]] = None,
    lines3d_names: List[str] = None,
    lines3d_colors: Union[torch.Tensor, List] = None,
    meshes: Union[Meshes, List[Mesh]] = None,
    meshes_names: List[str] = None,
    meshes_colors: Union[torch.Tensor, List] = None,
    meshes_add_translation: bool = False,
    pts3d_add_translation: bool = False,
    fpath: Path = None,
    return_visualization=False,
    viewpoints_count=1,
    viewpoint_init_dist=10.,
    dtype=torch.float,
    H=1080,
    W=1980,
    fps=10,
    pts3d_size=10.0,
    background_color=(1., 1., 1.),
    device='cpu',
    meshes_as_wireframe=False,
    crop_white_border=False,
    renderer=OD3D_RENDERER.OPEN3D,
):
    """
    Args:
        cams_tform4x4_world (Union[torch.Tensor, List[torch.Tensor]]): (Cx4x4) or List(4x4)
        cams_intr4x4 (Union[torch.Tensor, List[torch.Tensor]]): Cx4x4 or List(4x4)
        cams_imgs (Union[torch.Tensor, List[torch.Tensor]]): Cx3xHxW or List(3xHxW)
        cams_names: (List[str]): (P,)
        pts3d (Union[torch.Tensor, List[torch.Tensor]]): PxNx3 or List(Npx3)
        pts3d_names (List[str]): (P,)
        pts3d_colors (Union[torch.Tensor, List]): Px2x3 or List(3)
        lines3d (Union[torch.Tensor, List[torch.Tensor]]): PxNx2x3 or List(Npx2x3)
        lines3d_names (List[str]): (P,)
        lines3d_colors (Union[torch.Tensor, List]): Px2x3 or List(3)
        meshes (Meshes)
        meshes_names (List[str]): (M,)
        meshes_colors (Union[torch.Tensor, List]): Mx3 or List(3)
        renderer (OD3D_RENDERER): renderer to use
    Returns:
        -
    """

    geometries = []
    meshes_x_offsets = []
    meshes_z_offset = 0.0
    meshes_y_offset = 0.0
    if meshes is not None:
        if isinstance(meshes, List):
            meshes = Meshes.load_from_meshes(meshes)

        x_offset = 0.0
        for i in range(len(meshes)):
            vertices = meshes.get_verts_with_mesh_id(mesh_id=i).clone()
            if meshes_add_translation:
                x_offset_delta_current = 1.1 * (-vertices[:, 0].min()).clamp(min=0.0)
                if (
                    vertices[:, 2].max() - vertices[:, 2].min()
                ) * 1.1 > meshes_z_offset:
                    meshes_z_offset = (
                        vertices[:, 2].max() - vertices[:, 2].min()
                    ) * 1.1
                if (
                    vertices[:, 1].max() - vertices[:, 1].min()
                ) * 1.1 > meshes_y_offset:
                    meshes_y_offset = (
                        vertices[:, 1].max() - vertices[:, 1].min()
                    ) * 1.1
                x_offset += x_offset_delta_current
                x_offset_delta_next = 1.1 * (vertices[:, 0].max())
                vertices[:, 0] += x_offset
                meshes_x_offsets.append(x_offset.item())

                x_offset += x_offset_delta_next

            faces = meshes.get_faces_with_mesh_id(mesh_id=i)


            if (
                meshes_colors is not None
                and len(meshes_colors) >= i + 1
                and meshes_colors[i] is not None
            ):

                mesh_color = meshes_colors[i]
                #if isinstance(mesh_color, torch.Tensor):
                #    mesh_color = mesh_color.detach().cpu().numpy()

                mesh_rgb = meshes.get_verts_ncds_with_mesh_id(mesh_id=i)
                mesh_rgb[:, 0] = mesh_color[0]
                mesh_rgb[:, 1] = mesh_color[1]
                mesh_rgb[:, 2] = mesh_color[2]
            else:
                if meshes.rgb is not None:
                    mesh_rgb = meshes.get_rgb_with_mesh_id(mesh_id=i)
                else:
                    mesh_rgb = meshes.get_verts_ncds_with_mesh_id(mesh_id=i)

            if (
                meshes_names is not None
                and len(meshes_names) >= i + 1
                and meshes_names[i] is not None
            ):
                mesh_name = meshes_names[i]
            else:
                mesh_name = f"mesh{i}"

            if renderer == OD3D_RENDERER.OPEN3D:
                mesh_engine = open3d.geometry.TriangleMesh(
                    vertices=open3d.utility.Vector3dVector(vertices.detach().cpu().numpy()),
                    triangles=open3d.utility.Vector3iVector(faces.detach().cpu().numpy(),),
                )

                mesh_material = open3d.visualization.rendering.MaterialRecord()
                # mat_box.shader = 'defaultUnlit'
                mesh_material.shader = "defaultLitTransparency"
                # mat_box.shader = 'defaultLitSSR'
                # alpha = 0.5
                # mat_box.base_reflectance = 0.
                # mat_box.base_roughness = 0.0
                # mat_box.base_reflectance = 0.0
                # mat_box.base_clearcoat = 1.0
                # mat_box.thickness = 1.0
                # mat_box.transmission = 1.0
                # mat_box.absorption_distance = 10
                # mat_box.absorption_color = [0.5, 0.5, 0.5]
                vertex_colors = open3d.utility.Vector3dVector(mesh_rgb.detach().cpu().numpy(),)
                mesh_engine.vertex_colors = vertex_colors
                #mesh_engine.paint_uniform_color([0., 0., 0.])
                if meshes_as_wireframe:
                    mesh_engine = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_engine)

            elif renderer == OD3D_RENDERER.PYTORCH3D:
                from pytorch3d.renderer import TexturesVertex as PT3D_TexturesVertex
                textures = PT3D_TexturesVertex(verts_features=mesh_rgb.to(device)[None,])
                mesh_engine = PT3D_Meshes(verts=[vertices.to(device=device)], faces=[faces.to(device=device)], textures=textures)

                mesh_engine.rgb = mesh_rgb
                mesh_material = None
            else:
                raise NotImplementedError

            geometries.append(
                {"name": mesh_name, "geometry": mesh_engine, "material": mesh_material},
            )
            # vertices: open3d.cpu.pybind.utility.Vector3dVector,
            # triangles: open3d.cpu.pybind.utility.Vector3iVector

    if lines3d is not None:
        for i, lines3d_i in enumerate(lines3d):

            # N x 2 x 3
            _lines3d_i = lines3d_i.clone()
            _lines3d_i_pts3d = _lines3d_i.reshape(-1, 3)
            _lines3d_i_pts3d_ids = torch.arange(len(_lines3d_i_pts3d)).reshape(-1, 2)

            if (
                lines3d_colors is not None
                and len(lines3d_colors) >= i + 1
                and lines3d_colors[i] is not None
            ):
                lines3d_i_color = lines3d_colors[i]
            else:
                lines3d_i_color = get_colors(len(lines3d))[i]

            if (
                lines3d_names is not None
                and len(lines3d_names) >= i + 1
                and lines3d_names[i] is not None
            ):
                line3d_name = lines3d_names[i]
            else:
                line3d_name = f"lines3d_{i}"

            if renderer == OD3D_RENDERER.OPEN3D:
                line3d_engine = open3d.geometry.LineSet()
                line3d_engine.points = o3d.utility.Vector3dVector(
                    _lines3d_i_pts3d.detach().cpu().numpy(),
                )
                line3d_engine.lines = o3d.utility.Vector2iVector(
                    _lines3d_i_pts3d_ids.detach().cpu().numpy(),
                )
                if isinstance(lines3d_i_color, list) or lines3d_i_color.dim() == 1:
                    line3d_engine.paint_uniform_color(
                        (lines3d_i_color[0], lines3d_i_color[1], lines3d_i_color[2]),
                    )
                else:
                    line3d_engine.colors = o3d.utility.Vector3dVector(
                        lines3d_i_color.detach().cpu().numpy(),
                    )

            elif renderer == OD3D_RENDERER.PYTORCH3D:
                line3d_engine = None
            else:
                raise NotImplementedError

            geometries.append({"name": line3d_name, "geometry": line3d_engine})

    if pts3d is not None:
        x_offset = 0.0
        for i, pts3d_i in enumerate(pts3d):
            _pts3d_i = pts3d_i.clone()
            if pts3d_add_translation:
                _pts3d_i[:, 1] += meshes_y_offset
                # _pts3d_i[:, 2] += meshes_z_offset
                if len(meshes_x_offsets) > i:
                    x_offset = meshes_x_offsets[i]
                else:
                    x_offset_delta_current = 1.1 * (-_pts3d_i[:, 0].min()).clamp(
                        min=0.0,
                    )
                    x_offset += x_offset_delta_current
                if _pts3d_i[:, 0].numel() == 0:
                    x_offset_delta_next = 0.0
                else:
                    x_offset_delta_next = 1.1 * (_pts3d_i[:, 0].max())
                _pts3d_i[:, 0] += x_offset
                x_offset += x_offset_delta_next

            if (
                pts3d_colors is not None
                and len(pts3d_colors) >= i + 1
                and pts3d_colors[i] is not None
            ):
                pts3d_i_color = pts3d_colors[i]
            else:
                pts3d_i_color = get_colors(len(pts3d))[i]

            if (
                pts3d_names is not None
                and len(pts3d_names) >= i + 1
                and pts3d_names[i] is not None
            ):
                pts3d_name = pts3d_names[i]
            else:
                pts3d_name = f"pts3d_{i}"

            if renderer == OD3D_RENDERER.OPEN3D:
                pts3d_engine = open3d.geometry.PointCloud()
                pts3d_engine.points = open3d.utility.Vector3dVector(
                    _pts3d_i.detach().cpu().numpy(),
                )
                if (
                        pts3d_normals is not None
                        and len(pts3d_normals) >= i + 1
                        and pts3d_normals[i] is not None
                ):
                    pts3d_engine.normals = open3d.utility.Vector3dVector(
                        pts3d_normals[i].detach().cpu().numpy(),
                    )
                if isinstance(pts3d_i_color, list) or pts3d_i_color.dim() == 1:
                    pts3d_engine.paint_uniform_color(
                        (pts3d_i_color[0], pts3d_i_color[1], pts3d_i_color[2]),
                    )
                else:
                    pts3d_engine.colors = o3d.utility.Vector3dVector(
                        pts3d_i_color.detach().cpu().numpy(),
                    )

            elif renderer == OD3D_RENDERER.PYTORCH3D:
                if isinstance(pts3d_i_color, list) or pts3d_i_color.dim() == 1:
                    _pts3d_i_rgb = _pts3d_i.clone().to(device=device)
                    _pts3d_i_rgb[:, 0] = pts3d_i_color[0]
                    _pts3d_i_rgb[:, 1] = pts3d_i_color[1]
                    _pts3d_i_rgb[:, 2] = pts3d_i_color[2]
                else:
                    _pts3d_i_rgb = pts3d_i_color

                _pts3d_i_normals = [pts3d_normals[i]] if pts3d_normals is not None else None
                pts3d_engine = PT3D_Pointclouds(points=[_pts3d_i], normals=_pts3d_i_normals, features=[_pts3d_i_rgb])
            else:
                raise NotImplementedError

            geometries.append({"name": pts3d_name, "geometry": pts3d_engine})

    engine_geometries_for_cams = get_engine_geometries_for_cams(
        cams_tform4x4_world=cams_tform4x4_world,
        cams_intr4x4=cams_intr4x4,
        cams_imgs=cams_imgs,
        cams_names=cams_names,
        cams_imgs_resize=cams_imgs_resize,
        cams_imgs_depth_scale=cams_imgs_depth_scale,
        cams_show_wireframe=cams_show_wireframe,
        renderer=renderer,
        device=device
    )

    for engine_geometry_for_cam in engine_geometries_for_cams:
        geometries.append(engine_geometry_for_cam)

    if return_visualization is False and fpath is None:
        if os.environ.get("DISPLAY"):

            if renderer == OD3D_RENDERER.OPEN3D:
                # advantage: transparent
                open3d.visualization.draw(
                    geometries,
                    show_skybox=False,
                    bg_color=[1.0, 1.0, 1.0, 1.0],
                    raw_mode=True,
                )

                # advantage: normals shown
                # open3d.visualization.draw_geometries( [geometry['geometry'] for geometry in geometries])
            elif renderer == OD3D_RENDERER.PYTORCH3D:
                cam_tform4x4_obj = DEFAULT_CAM_TFORM_OBJ.clone().to(dtype=dtype, device=device)
                cam_intr4x4 = get_default_camera_intrinsics_from_img_size(H=H, W=W, dtype=dtype, device=device)
                img_size = torch.Tensor([H, W]).to(dtype=dtype, device=device)

                pt3d_cameras = pt3d_camera_from_tform4x4_intr4x4_imgs_size(
                    cam_tform4x4_obj=cam_tform4x4_obj,
                    cam_intr4x4=cam_intr4x4,
                    img_size=img_size,
                )

                plotly_background_color = f'rgb({int(255*background_color[0])}, {int(255*background_color[1])}, {int(255*background_color[2])})'

                fig = plot_scene(
                    {
                        "default": {geometry['name']: geometry['geometry'] for geometry in geometries},
                    },
                    viewpoint_cameras=pt3d_cameras,
                    axis_args=AxisArgs(
                        backgroundcolor=plotly_background_color,
                        showgrid=True,
                        zeroline=True,
                        showline=True,
                        showaxeslabels=True,
                        showticklabels=True,
                    ),
                )

                fig.show()
        else:
            logger.warning(
                "could not visualize with open3d, because env DISPLAY not set, try `export DISPLAY=:0.0;`",
            )
            return
    else:
        imgs = []
        from od3d.io import is_fpath_video
        from od3d.cv.geometry.transform import (
            get_cam_tform4x4_obj_for_viewpoints_count, inv_tform4x4
        )

        cams_new_tform4x4_obj = get_cam_tform4x4_obj_for_viewpoints_count(
            viewpoints_count=viewpoints_count,
            dist=viewpoint_init_dist,
            spiral=is_fpath_video(fpath),
        ).to(dtype=dtype, device=device)

        #if (
        #    os.environ.get("DISPLAY")
        #    or open3d._build_config["ENABLE_HEADLESS_RENDERING"]
        #):
        if renderer == OD3D_RENDERER.OPEN3D and os.environ.get("DISPLAY"):
            cams_new_tform4x4_obj[:, 2, 3] /= viewpoint_init_dist
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, height=H, width=W)
            opt = vis.get_render_option()
            opt.point_size = pts3d_size
            opt.mesh_show_back_face = False
            opt.background_color = np.asarray(background_color)
            # opt.background_color = np.asarray([0, 0, 0])
            # opt.mesh_show_wireframe = mesh_show_wireframe

            geometries_vertices_orig = []
            for geometry in geometries:
                vis.add_geometry(geometry["geometry"])
                if isinstance(geometry["geometry"], open3d.geometry.PointCloud):
                    geometries_vertices_orig.append(
                        torch.from_numpy(np.asarray(geometry["geometry"].points))
                        .clone()
                        .to(device=device, dtype=dtype),
                    )
                elif isinstance(geometry["geometry"], open3d.geometry.TriangleMesh):
                    geometries_vertices_orig.append(
                        torch.from_numpy(np.asarray(geometry["geometry"].vertices))
                        .clone()
                        .to(device=device, dtype=dtype),
                    )
                elif isinstance(geometry["geometry"], open3d.geometry.LineSet):
                    geometries_vertices_orig.append(
                        torch.from_numpy(np.asarray(geometry["geometry"].points))
                        .clone()
                        .to(device=device, dtype=dtype),
                    )
                else:
                    geometries_vertices_orig.append(None)
                # vis.update_geometry(geometry['geometry'])
            vis.poll_events()
            vis.update_renderer()
            # view_control = vis.get_view_control()

            # open3d version 0.17.0 bug, view control does not work
            # camera_orig = view_control.convert_to_pinhole_camera_parameters()
            # cam_tform4x4_obj = torch.from_numpy(camera_orig.extrinsic).to(dtype=objs_new_tform4x4_obj.dtype, device=objs_new_tform4x4_obj.device)
            from tqdm import tqdm

            for v in tqdm(range(viewpoints_count)):
                # camera_orig.extrinsic = tform4x4_broadcast(cam_tform4x4_obj,
                #                                           objs_new_tform4x4_obj[v]).detach().cpu().numpy()
                # view_control.convert_from_pinhole_camera_parameters(camera_orig)


                for g, geometry in enumerate(geometries):
                    if geometries_vertices_orig[g] is not None:
                        vertices = geometries_vertices_orig[
                            g
                        ]  # .to(dtype=objs_new_tform4x4_obj.dtype, device=objs_new_tform4x4_obj.device)
                        vertices = transf3d_broadcast(
                            pts3d=vertices,
                            transf4x4=tform4x4(
                                OBJ_TFORM_OPEN3D_DEFAULT_CAM.to(
                                    dtype=dtype,
                                    device=device,
                                ),
                                cams_new_tform4x4_obj[v],
                            ),
                        )

                        if isinstance(
                                geometry["geometry"],
                                open3d.geometry.PointCloud,
                        ):
                            geometry[
                                "geometry"
                            ].points = open3d.utility.Vector3dVector(
                                vertices.detach().cpu().numpy(),
                            )
                        elif isinstance(
                                geometry["geometry"],
                                open3d.geometry.TriangleMesh,
                        ):
                            geometry[
                                "geometry"
                            ].vertices = open3d.utility.Vector3dVector(
                                vertices.detach().cpu().numpy(),
                            )
                        elif isinstance(
                                geometry["geometry"],
                                open3d.geometry.LineSet,
                        ):
                            geometry[
                                "geometry"
                            ].points = open3d.utility.Vector3dVector(
                                vertices.detach().cpu().numpy(),
                            )
                        vis.update_geometry(geometry["geometry"])

                vis.update_renderer()
                img = vis.capture_screen_float_buffer(do_render=True)
                img = torch.from_numpy(np.array(img)).permute(2, 0, 1)
                if crop_white_border:
                    from od3d.cv.visual.crop import crop_white_border_from_img
                    img = crop_white_border_from_img(img, resize_to_orig=True)
                imgs.append(img)

            vis.update_renderer()
            vis.destroy_window()

        elif renderer == OD3D_RENDERER.PYTORCH3D:
            from od3d.cv.geometry.fit.depth_from_mesh_and_box import get_scale_bbox_pts3d_to_image

            pts3d = [] # [for g, geometry in enumerate(geometries)]
            for g, geometry in enumerate(geometries):
                if isinstance(geometry['geometry'], PT3D_Pointclouds):
                    pts3d.append(geometry['geometry'].points_packed())
                elif isinstance(geometry['geometry'], PT3D_Meshes):
                    pts3d.append(geometry['geometry'].verts_packed())
                elif isinstance(geometry['geometry'], PerspectiveCameras):
                    pts3d.append(geometries[1]['geometry'].get_camera_center())
                else:
                    logger.info(f'geometry not supported {type(geometry["geometry"])}')
            pts3d = torch.cat(pts3d, dim=0)
            cam_intr4x4 = get_default_camera_intrinsics_from_img_size(H=H, W=W, dtype=dtype, device=device)

            scale = get_scale_bbox_pts3d_to_image(cam_intr4x4=cam_intr4x4,
                                                  cam_tform4x4_obj=cams_new_tform4x4_obj[0],
                                                  pts3d=pts3d, img_width=W, img_height=H)
            cams_new_tform4x4_obj[:, 2, 3] *= scale.max(dim=-1).values / 2. # plot_scene inside pytorch3d does scale otherwise

            img_size = torch.Tensor([H, W]).to(dtype=dtype, device=device)

            plotly_background_color = f'rgb({int(255 * background_color[0])}, {int(255 * background_color[1])}, {int(255 * background_color[2])})'

            plotly_geometries = {geometry['name']: geometry['geometry'] for geometry in geometries}

            from tqdm import tqdm
            for v in tqdm(range(viewpoints_count)):
                pt3d_cameras = pt3d_camera_from_tform4x4_intr4x4_imgs_size(
                    cam_tform4x4_obj=cams_new_tform4x4_obj[v],
                    cam_intr4x4=cam_intr4x4,
                    img_size=img_size,
                )

                fig = plot_scene(
                    {
                        f'': plotly_geometries # view{v}
                    },
                    viewpoint_cameras=pt3d_cameras,
                    axis_args=AxisArgs(
                        backgroundcolor=plotly_background_color,
                        showgrid=False,
                        zeroline=False,
                        showline=False,
                        showaxeslabels=False,
                        showticklabels=False,
                    ),
                )
                fig.update_layout(showlegend=False)
                fig.update_layout(scene = dict(
                    xaxis = dict(visible=False), yaxis = dict(visible=False), zaxis = dict(visible=False)
                ))

                for a in range(len(fig.layout.annotations)):
                    fig.layout.annotations[a].update(text="") #a=1, remove text from subplot
                img = plotly_fig_2_tensor(fig, width=W, height=H)
                if crop_white_border:
                    from od3d.cv.visual.crop import crop_white_border_from_img
                    img = crop_white_border_from_img(img, resize_to_orig=True)
                imgs.append(img)
        else:
            logger.warning(
                "could not visualize with open3d, most likely env DISPLAY not set, try `export DISPLAY=:0.0;` (maybe without ;)",
            )

            return [
                torch.zeros(size=(3, 480, 640)).to(device=device),
            ] * viewpoints_count

        if viewpoints_count == 1:
            imgs = imgs[0]
        else:
            if viewpoints_count > 4 and not is_fpath_video(fpath):
                viewpoints_count_sqrt = math.ceil(math.sqrt(viewpoints_count))
                imgs_placeholder = torch.zeros(
                    size=(viewpoints_count_sqrt**2, 3, H, W),
                    dtype=dtype,
                    device=device,
                )
                imgs_placeholder[:viewpoints_count] = torch.stack(imgs, dim=0)
                imgs = imgs_placeholder
                imgs = imgs.reshape(
                    viewpoints_count_sqrt,
                    viewpoints_count_sqrt,
                    3,
                    H,
                    W,
                )

                logger.info(imgs.shape)
            else:
                imgs = torch.stack(imgs, dim=0)

        if fpath is not None:
            if is_fpath_video(fpath):
                from od3d.cv.visual.video import save_video

                save_video(imgs=imgs, fpath=fpath, fps=fps)
            else:
                show_imgs(rgbs=imgs, fpath=fpath, pad=0)

        if return_visualization:
            return imgs
        else:
            return 0



def get_engine_geometries_for_cams(
    cams_tform4x4_world: Union[torch.Tensor, List[torch.Tensor]] = None,
    cams_intr4x4: Union[torch.Tensor, List[torch.Tensor]] = None,
    cams_imgs: Union[torch.Tensor, List[torch.Tensor]] = None,
    cams_names: List[str] = None,
    cams_imgs_resize: bool = True,
    cams_imgs_depth_scale: float = 0.2,
    cams_show_wireframe: bool = True,
    renderer = OD3D_RENDERER.OPEN3D,
    device='cpu',
):
    """
    Args:
        cams_tform4x4_world (Union[torch.Tensor, List[torch.Tensor]]): (Cx4x4) or List(4x4)
        cams_intr4x4 (Union[torch.Tensor, List[torch.Tensor]]): Cx4x4 or List(4x4)
        cams_imgs (Union[torch.Tensor, List[torch.Tensor]]): Cx3xHxW or List(3xHxW)
        cams_names: (List[str]): (P,)
    Returns:
        od3d_geometries (List): list with geometries as dict
    """

    geometries = []

    if cams_tform4x4_world is not None and cams_intr4x4 is not None:
        for i in range(len(cams_tform4x4_world)):
            if isinstance(cams_intr4x4, torch.Tensor) and cams_intr4x4.dim() == 2:
                cam_intr4x4 = cams_intr4x4
            elif len(cams_intr4x4) == 1:
                cam_intr4x4 = cams_intr4x4[0]
            else:
                cam_intr4x4 = cams_intr4x4[i]

            width = int(cam_intr4x4[0, 2] * 2)
            height = int(cam_intr4x4[1, 2] * 2)


            if (
                cams_names is not None
                and len(cams_names) >= i + 1
                and cams_names[i] is not None
            ):
                cam_name = cams_names[i]
            else:
                cam_name = f"cam{i}"

            if cams_imgs is not None and len(cams_imgs) > i:
                h, w = cams_imgs[i].shape[1:]
                if cams_imgs_resize:
                    cam_img = resize(cams_imgs[i], H_out=256, W_out=256)
                    h_resize = 256. / h
                    w_resize = 256. / w
                else:
                    cam_img = cams_imgs[i]
                    h_resize = 1.0
                    w_resize = 1.0

                if cam_img.dtype == torch.float:
                    cam_img = cam_img.clone() * 255

                cam_intr4x4_res = cam_intr4x4.clone()
                cam_intr4x4_res[0, 0] = cam_intr4x4_res[0, 0].item() * w_resize
                cam_intr4x4_res[1, 1] = cam_intr4x4_res[1, 1].item() * h_resize
                cam_intr4x4_res[0, 2] = cam_intr4x4_res[0, 2].item() * w_resize
                cam_intr4x4_res[1, 2] = cam_intr4x4_res[1, 2].item() * h_resize

                if renderer == OD3D_RENDERER.OPEN3D:
                    cam_tform4x4_obj = cams_tform4x4_world[i].detach().cpu().numpy()
                    depth_scale = (
                        cams_imgs_depth_scale * cam_tform4x4_obj[2, 3]
                    )  #  * 10 * cam_tform4x4_obj[2, 3]
                    depth = open3d.geometry.Image(
                        (
                            (torch.ones(size=cam_img.shape[1:])).cpu().detach().numpy()
                            * 255
                        ).astype(np.uint8),
                    )
                    img = open3d.geometry.Image(
                        (
                            cam_img.permute(1, 2, 0).contiguous().cpu().detach().numpy()
                        ).astype(np.uint8),
                    )
                    fx = cam_intr4x4_res[0, 0].item()
                    fy = cam_intr4x4_res[1, 1].item()
                    cx = cam_intr4x4_res[0, 2].item()
                    cy = cam_intr4x4_res[1, 2].item()
                    rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
                        color=img,
                        depth=depth,
                        depth_scale=1 / depth_scale,
                        depth_trunc=3 * depth_scale,
                        convert_rgb_to_intensity=False,
                    )

                    intrinsic = open3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
                    intrinsic.intrinsic_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                    cam = open3d.camera.PinholeCameraParameters()
                    cam.intrinsic = intrinsic
                    cam.extrinsic = cam_tform4x4_obj  # cams_tform4x4_world[i].detach().cpu().numpy()

                    pts3d_engine = open3d.geometry.PointCloud.create_from_rgbd_image(
                        rgbd,
                        cam.intrinsic,
                        cam.extrinsic,
                    )


                    if cams_show_wireframe:
                        cam_wireframe_engine = open3d.geometry.LineSet.create_camera_visualization(
                            view_width_px=width,
                            view_height_px=height,
                            intrinsic=cam_intr4x4[:3, :3].detach().cpu().numpy(),
                            extrinsic=cam_tform4x4_obj,
                            scale=cams_imgs_depth_scale * cam_tform4x4_obj[2, 3],
                        )

                elif renderer == OD3D_RENDERER.PYTORCH3D:
                    dtype = cams_tform4x4_world[i].dtype

                    from od3d.cv.geometry.transform import depth2pts3d_grid, inv_tform4x4
                    depth = cams_imgs_depth_scale* 2.5 * torch.ones(cam_img.shape[1:]).to(device=device, dtype=dtype)
                    pts3d_i = depth2pts3d_grid(depth, cam_intr4x4_res.to(device=device, dtype=dtype)).flatten(1).permute(1, 0)
                    pts3d_i = transf3d_broadcast(pts3d_i, inv_tform4x4(cams_tform4x4_world[i].to(device=device, dtype=dtype)))
                    pts3d_i_rgb = cam_img.to(device=device, dtype=dtype).flatten(1).permute(1, 0) / 255.
                    # cam_img
                    #pts3d_i =  torch.from_numpy(np.asarray(pts3d_engine.points)).to(device=device, dtype=dtype)
                    #pts3d_i_rgb = torch.from_numpy(np.asarray(pts3d_engine.colors)).to(device=device, dtype=dtype)
                    pts3d_engine = PT3D_Pointclouds(points=[pts3d_i], features=[pts3d_i_rgb])

                    if cams_show_wireframe:
                        h, w = cam_img.shape[1:]

                        cam_wireframe_engine = pt3d_camera_from_tform4x4_intr4x4_imgs_size(cam_tform4x4_obj=cams_tform4x4_world[i],
                                                                                           cam_intr4x4=cam_intr4x4,
                                                                                           img_size=torch.tensor([height, width]))
                else:
                    raise NotImplementedError

                if cams_show_wireframe:
                    geometries.append({"name": cam_name, "geometry": cam_wireframe_engine})

                geometries.append(
                    {"name": f"{cam_name}_img", "geometry": pts3d_engine},
                )

    return geometries


def show_pcl_via_open3d(pts3d, pts3d_colors=None):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    # vis.add_geometry(pcd)

    pcd = o3d.geometry.PointCloud()
    # from od3d.cv.geometry.transform import inv_tform4x4
    pcd.points = o3d.utility.Vector3dVector(pts3d.numpy())
    if pts3d_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(pts3d_colors)
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()


def show_open3d_pcl(pcd):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()


def show_pcl(
    verts,
    cam_tform4x4_obj: torch.Tensor = None,
    cam_intr4x4: torch.Tensor = None,
    img_size: torch.Tensor = None,
):
    """

    Args:
        verts: Nx3 / BxNx3 / list(torch.Tensor Nx3)

    """

    if cam_tform4x4_obj is not None:
        pt3d_cameras = pt3d_camera_from_tform4x4_intr4x4_imgs_size(
            cam_tform4x4_obj=cam_tform4x4_obj,
            cam_intr4x4=cam_intr4x4,
            img_size=img_size,
        )
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
        # cls = cls[None,]

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

    point_cloud = PT3D_Pointclouds(points=verts, features=rgb)
    fig = plot_scene(
        {
            "Pointcloud": {
                **{f"pcl{i+1}": point_cloud[i] for i in range(len(point_cloud))},
                **{f"cam{i+1}": pt3d_cameras[i] for i in range(len(pt3d_cameras))},
            },
        },
        viewpoint_cameras=pt3d_cameras,
        axis_args=AxisArgs(
            backgroundcolor="rgb(200, 200, 230)",
            showgrid=True,
            zeroline=True,
            showline=True,
            showaxeslabels=True,
            showticklabels=True,
        ),
    )
    fig.show()
    input("bla")


def imgs_to_img(rgbs, pad=1, pad_value=0, H_out=None, W_out=None):
    # rgb: K x 3 x H x W / GH x GW x 3 x H x W

    if isinstance(rgbs, List) and isinstance(rgbs[0], List):
        H_in, W_in = rgbs[0][0].shape[-2:]
        rgbs = torch.stack(
            [
                torch.stack(
                    [resize(rgb, H_out=H_in, W_out=W_in) for rgb in rgbs_i],
                    dim=0,
                )
                for rgbs_i in rgbs
            ],
            dim=0,
        )
    elif isinstance(rgbs, List):
        H_in, W_in = rgbs[0].shape[-2:]
        rgbs = torch.stack([resize(rgb, H_out=H_in, W_out=W_in) for rgb in rgbs], dim=0)

    rgbs = torch.nn.functional.pad(rgbs, (pad, pad, pad, pad), "constant", pad_value)
    # margin = 2
    # torch.nn.functional.pad(rgbs, (1, 1), "constant", 0)

    rgbs_shape = rgbs.shape

    if len(rgbs_shape) == 5:
        GH, GW, C, H, W = rgbs_shape
        rgb = rgbs.clone()
    elif len(rgbs_shape) == 4:
        K, C, H, W = rgbs.shape
        prop_w = 4
        prop_h = 3
        GW = math.ceil(math.sqrt((K * prop_w**2) / prop_h**2))
        GH = math.ceil(K / GW)
        GTOTAL = GH * GW

        img_placeholder = torch.zeros_like(rgbs[:1]).repeat(GTOTAL - K, 1, 1, 1)

        rgb = torch.cat((rgbs, img_placeholder), dim=0)

        rgb = rgb.reshape(GH, GW, C, H, W)
    elif len(rgbs_shape) == 3:
        GH = 1
        GW = 1
        C, H, W = rgbs_shape
        rgb = rgbs.clone()[None, None]
    else:
        logger.error(
            "Visualize imgs requires the input rgb tensor to have 4 (KxCxHxW) or 5 (GHxGWxCxHxW) dimensions",
        )
        raise NotImplementedError

    rgb = rgb.permute(2, 0, 3, 1, 4)

    rgb = rgb.reshape(C, GH * H, GW * W)

    if H_out is not None and W_out is not None:
        rgb = resize(rgb, H_out=H_out, W_out=W_out)
    return rgb


def fpaths_to_rgb(fpaths: List[Path], H: int, W: int, pad=1):
    rgbs = torch.stack(
        [
            resize(torchvision.io.read_image(path=str(fpath)), H_out=H, W_out=W)
            for fpath in fpaths
        ],
        dim=0,
    )
    rgb = imgs_to_img(rgbs, pad=pad)
    return rgb


def show_imgs(
    rgbs,
    duration=0,
    vwriter=None,
    fpath=None,
    height=None,
    width=None,
    pad=1,
):
    rgb = imgs_to_img(rgbs, pad=pad)
    return show_img(rgb, duration, vwriter, fpath, height, width)


def show_img(
    rgb,
    duration=0,
    vwriter=None,
    fpath=None,
    height=None,
    width=None,
    normalize=False,
):
    # img: 3xHxW
    rgb = rgb.clone()

    if normalize:
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

    if width is not None and height is not None:
        orig_width = rgb.size(2)
        orig_height = rgb.size(1)
        scale_factor = min(width / orig_width, height / orig_height)
    elif width is not None:
        orig_width = rgb.size(2)
        scale_factor = width / orig_width
    elif height is not None:
        orig_height = rgb.size(1)
        scale_factor = height / orig_height

    if width or height is not None:
        rgb = resize(
            rgb[None,],
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


def get_img_from_plot(ax, fig, axis_off=True, margins=1, pad=1):
    try:
        count_axes = len(ax)
        single_axes = False
    except TypeError:
        count_axes = 1
        single_axes = True
    # Image from plot
    if axis_off:
        if single_axes:
            ax.axis("off")
            # To remove the huge white borders
            ax.margins(margins)
        else:
            for ax_single in ax:
                ax_single.axis("off")
                # To remove the huge white borders
                ax_single.margins(margins)
        fig.tight_layout(pad=pad)
    else:
        fig.tight_layout(pad=pad)
        if single_axes:
            ax.margins(margins)
        else:
            for ax_single in ax:
                ax_single.margins(margins)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot: np.ndarray = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (3,),
    )

    return torch.from_numpy(image_from_plot.copy()).permute(2, 0, 1)
