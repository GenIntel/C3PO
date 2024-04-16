import matplotlib.pyplot as plt
import od3d.io
import torch
import torchvision.transforms as T
from od3d.cv.cluster.embed import tsne
from od3d.cv.geometry.mesh import Meshes
from od3d.cv.io import image_as_wandb_image
from od3d.cv.visual.crop import crop_white_border_from_img
from od3d.cv.visual.draw import get_colors
from od3d.cv.visual.show import show_scene
from od3d.cv.visual.show import show_scene2d

benchmark = "pascal3d_nemo_dino"

ckpt_path = "/home/heisenberg/Thesis_repo/exps_remote/03-11_10-47-25_Pascal3D_NeMo_torque/nemo.ckpt"  # moving average with head with mesh feats updated
ckpt_path = "/home/heisenberg/Thesis_repo/exps_remote/03-09_18-59-01_Pascal3D_NeMo_torque/nemo.ckpt"  # loss gradient
# ckpt_path = "/home/heisenberg/Thesis_repo/exps_remote/03-08_19-34-46_Pascal3D_NeMo_DINO_torque/nemo.ckpt" # average

ckpt_path_old = "/home/heisenberg/Thesis_repo/exps_remote/NeMo-Classification/classification_saved_model_199.pth"
checkpoint = torch.load(ckpt_path)

colors = get_colors(K=20, K_rel=10)


cfgs = [od3d.io.load_hierarchical_config(benchmark=benchmark, platform="local")]


methods_cfgs = []
for i, cfg in enumerate(cfgs):
    # logger.info(f'{i} of {len(cfgs)}')
    methods_keys = cfg.method.keys()
    for key in methods_keys:
        method_cfg = cfg.copy()
        method_cfg.method = cfg.method[key]
        method_cfg_exists = False
        for prev_method_cfg in methods_cfgs:
            if method_cfg == prev_method_cfg:
                method_cfg_exists = True
        if not method_cfg_exists:
            methods_cfgs.append(method_cfg)

print(f"{len(methods_cfgs)} configs with single method.")

methods_cfgs
method_config = methods_cfgs[0].method
color_ = plt.get_cmap("nipy_spectral", len(method_config.categories))
fpaths_meshes = [method_config.fpaths_meshes[cls] for cls in method_config.categories]
meshes = Meshes.load_from_files(fpaths_meshes=fpaths_meshes)

# load_checkpoint_old(meshes, method_config, ckpt_path_old)
meshes.set_feats_cat(checkpoint["meshes_feats"])

cat_borders = torch.cat(
    [
        torch.LongTensor([0]).to(device=meshes.device),
        torch.cumsum(torch.tensor(meshes.verts_counts), dim=0),
    ],
    dim=0,
)
feats_all_cats = []
feats_all_colors = []
for i, cat in enumerate(method_config.categories):
    feats_all_colors.extend([list(color_(i))] * meshes.verts_counts[i])


feats_all_cats_tsne = tsne(checkpoint["meshes_feats"], C=3)

img = show_scene2d(
    [feats_all_cats_tsne],
    pts2d_colors=[feats_all_colors],
    return_visualization=True,
)
img_wandb = image_as_wandb_image(img)
img_cropped = crop_white_border_from_img(img)
transform = T.ToPILImage()
img = transform(img_cropped)
img.show()
print(img)
show_scene(
    pts3d=[feats_all_cats_tsne[:]],
    pts3d_colors=[torch.tensor([sublist[:3] for sublist in feats_all_colors])],
)
