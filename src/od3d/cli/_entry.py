import logging
import typer
app = typer.Typer()
from od3d.cli.benchmark import app as app_run
from od3d.cli.dataset import app as app_setup

app.add_typer(app_run, name='bench')
app.add_typer(app_setup, name='dataset')
def main():
    app()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()