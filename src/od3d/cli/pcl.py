
import typer
from od3d.cv.io import load_ply
from od3d.cv.visual.show import show_pcl
app = typer.Typer()

@app.command()
def show(fpath: str = typer.Option(None, '-f', '--fpath'), device: str = typer.Option('cpu', '-d', '--device')):
    fpath = "/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/pcls/206_21810_45890/cuboid_max_1000.ply"
    verts, faces = load_ply(fpath)

    show_pcl(verts.to(device=device))