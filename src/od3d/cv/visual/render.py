import torch
import logging
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    # SoftPhongShader,
    HardPhongShader,
    TexturesVertex
)
from pytorch3d.structures.meshes import Meshes
import matplotlib.pyplot as plt
from pytorch3d.io import IO
from od3d.cv.geometry.transform import proj3d2d

def render_mesh(fpath_mesh, cam_tform_obj, cam_intr, img_size):

    dtype = cam_tform_obj.dtype
    device = cam_tform_obj.device

    focal_length = torch.Tensor([cam_intr[0, 0], cam_intr[1, 1]]).to(device=device, dtype=dtype)
    principal_point = torch.Tensor([cam_intr[0, 2], cam_intr[1, 2]]).to(device=device, dtype=dtype)
    io = IO()
    mesh = io.load_mesh(fpath_mesh, device=device)
    verts = mesh[0].verts_list()[0]
    faces = mesh[0].faces_list()[0]
    verts_shape = verts.shape
    verts_rgb = torch.ones(size=verts_shape)[None,] * 0.5  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
    )
    logging.info("we can visualize the Pascald3D frame here.")



    pscl3d_tform_t3d = torch.Tensor([[-1., 0., 0., 0.],
                                     [0., -1., 0., 0.],
                                     [0., 0., 1., 0.],
                                     [0., 0., 0., 1.]]).to(device=device, dtype=dtype)
    t3d_cam_tform_obj = torch.matmul(pscl3d_tform_t3d, cam_tform_obj)

    R = t3d_cam_tform_obj[:3, :3].T[None,]
    t = t3d_cam_tform_obj[:3, 3][None,]

    cameras = PerspectiveCameras(device=device, R=R, T=t, focal_length=focal_length[None,],
                                 principal_point=principal_point[None,], in_ndc=False,
                                 image_size=img_size[None,])  # K=self.K_4x4[None,]) #, K=K) # , K=K , znear=0.001, zfar=100000,
    #  znear=0.001, zfar=100000, fov=10
    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
    # the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=[int(img_size[0]), int(img_size[1])],
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
    # -z direction.
    lights = PointLights(device=device, location=[[0.0, 0.0, 10.0]])

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )

    rgba_synthetic_batch = renderer(mesh)

    rgb_synthetic = (rgba_synthetic_batch[0, ..., :3] * 255).to(torch.uint8).permute(2, 0, 1)
    return rgb_synthetic