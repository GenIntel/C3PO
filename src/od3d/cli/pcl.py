
import typer
from od3d.cv.io import load_ply
from od3d.cv.visual.show import show_pcl
app = typer.Typer()

@app.command()
def show(fpath: str = typer.Option(None, '-f', '--fpath'), device: str = typer.Option('cpu', '-d', '--device')):
    pcl, _ = load_ply(fpath)

    show_pcl(pcl.to(device=device))