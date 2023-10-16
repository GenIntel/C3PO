
import typer
from od3d.cv.io import load_ply
from od3d.cv.visual.show import show_pcl, show_open3d_pcl
app = typer.Typer()

@app.command()
def show(fpath: str = typer.Option(None, '-f', '--fpath'), device: str = typer.Option('cpu', '-d', '--device')):
    fpath = "/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/pcls/206_21810_45890/cuboid_max_1000.ply"
    verts, faces = load_ply(fpath)

    show_pcl(verts.to(device=device))

@app.command()
def show_open3d():
    import open3d as o3d
    fpath = "/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/droid_slam/car/340_35306_64677/pcl.ply"
    # fpath = "/scratch/sommerl/repos/NeMo/third_party/AutoReconoutputs/neusfacto-wbg-reg_sep-plane-nerf_60k_plane-h-ratio-0.3_co3d-scan1_cvpr/neus-facto-wbg-reg_sep-plane-nerf/2023-09-20_211422/extracted_mesh_res-512_max-component.ply"
    pcd = o3d.io.read_point_cloud(fpath)
    print(pcd)
    o3d.visualization.draw_geometries([pcd])