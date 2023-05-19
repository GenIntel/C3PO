from pathlib import Path
import logging
logging.basicConfig(level=logging.DEBUG)
import typer
app = typer.Typer()
from od3d.cli.benchmark import app as app_run
from od3d.cli.dataset import app as app_setup
from od3d.cli.debug import app as app_debug

app.add_typer(app_run, name='bench')
app.add_typer(app_setup, name='dataset')
app.add_typer(app_debug, name='debug')

def main():
    app()

if __name__ == "__main__":
    main()



