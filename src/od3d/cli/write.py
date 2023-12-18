import logging
logger = logging.getLogger(__name__)
import typer
app = typer.Typer()

import torch
from od3d.datasets.co3d import CO3D
from od3d.cv.visual.show import show_imgs

@app.command()
def sequence():
    sequence_name_unique = 'bicycle/397_49943_98337'
    co3d = CO3D.create_by_name('co3d_10s_zsp_unlabeled', config={'categories': ['bicycle']}) #co3d_5s_no_zsp_labeled 'co3d_50s_no_zsp_aligned' 'co3dv1_10s_zsp_aligned' 'co3d_10s_zsp_aligned' 'co3dv1_10s_zsp_unlabeled'
    categories = co3d.categories
    sequence = co3d.get_sequence_by_category_and_name(category='bicycle', name='397_49943_98337')
    cams_tform4x4_world, cams_intr4x4, cams_imgs = sequence.get_cams(show_imgs=True, cams_count=6)
    show_imgs(rgbs=torch.stack(cams_imgs, dim=0), fpath='sequence.png')
