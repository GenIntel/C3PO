
import typer
from od3d.cv.io import load_ply
from od3d.cv.geometry.mesh import Meshes
from od3d.cv.visual.show import show_pcl, show_open3d_pcl
import od3d.cv.visual.show as show
app = typer.Typer()

@app.command()
def show_shapenet():
    from pytorch3d.renderer import (
        OpenGLPerspectiveCameras,
        PointLights,
        RasterizationSettings,
        TexturesVertex,
        look_at_view_transform,
    )
    import torch

    from pytorch3d.structures import Meshes

    device = 'cuda'
    # Rendering settings.
    R, T = look_at_view_transform(1.0, 1.0, 90)
    cameras = OpenGLPerspectiveCameras(R=R, T=T, device=device)
    raster_settings = RasterizationSettings(image_size=512)
    lights = PointLights(location=torch.tensor([0.0, 1.0, -2.0], device=device)[None], device=device)

    Meshes()

@app.command()
def show_scene():
    fpath = '/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/aligned/latest_50s_to_5s_mesh/r0/mesh/car/mesh.ply'
    fpath = '/misc/lmbraid19/sommerl/datasets/ShapeNetCore.v2/02691156/885b7ba506c9d68ab6ecbbb2e4d05900/models/model_normalized.obj'
    fpath = '/misc/lmbraid19/sommerl/datasets/ShapeNetCore.v2/03325088/ad51249f5960d841c36fb70296e45483/models/model_normalized.obj'
    fpath = '/misc/lmbraid19/sommerl/datasets/ShapeNetCore.v2/02958343/adc6f0f3274cf92cd4f6529a209c5dc0/models/model_normalized.obj'
    fpath = '/misc/lmbraid19/sommerl/datasets/ShapeNetCore.v2/02958343/bac6953b6866ec02300856337cd5b2e/models/model_normalized.obj'

    # 02958343 : car
    meshes = Meshes.load_from_files([fpath])
    meshes.feats = meshes.rgb
    show.show_scene(meshes=meshes, meshes_colors=meshes.rgb)

    # import open3d as o3d
    #
    # def visualize(mesh):
    #     vis = o3d.visualization.Visualizer()
    #     vis.create_window()
    #     vis.add_geometry(mesh)
    #     vis.run()
    #     vis.destroy_window()
    #
    # def main():
    #     mesh = o3d.io.read_triangle_model(fpath) # read_triangle_mesh read_triangle_model
    #
    #     o3d.visualization.draw_geometries([mesh])
    #     # visualize(mesh)
    #
    # main()



# @app.command()
# def show(fpath: str = typer.Option(None, '-f', '--fpath'), device: str = typer.Option('cpu', '-d', '--device')):
#     fpath = "/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/pcls/206_21810_45890/cuboid_max_1000.ply"
#     verts, faces = load_ply(fpath)
#
#     show_pcl(verts.to(device=device))
#
@app.command()
def show_open3d():
    import open3d as o3d
    fpath = '/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/aligned/latest_50s_to_5s_mesh/r0/mesh/car/mesh.ply'
    fpath = '/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/aligned/all_20s_to_5s_mesh/r0/mesh/car/mesh.ply'

    # fpath = "/scratch/sommerl/repos/NeMo/third_party/AutoReconoutputs/neusfacto-wbg-reg_sep-plane-nerf_60k_plane-h-ratio-0.3_co3d-scan1_cvpr/neus-facto-wbg-reg_sep-plane-nerf/2023-09-20_211422/extracted_mesh_res-512_max-component.ply"
    #pcd = o3d.io.read_point_cloud(fpath)
    mesh = o3d.io.read_triangle_mesh(fpath)

    #print(pcd)
    print(mesh.is_watertight())
    print(mesh)

    o3d.visualization.draw_geometries([mesh])