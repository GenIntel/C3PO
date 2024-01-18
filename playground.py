# from huggingface_hub import login
#
# login('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', add_to_git_credential=True)
#
# from datasets import get_dataset_split_names
# from datasets import load_dataset_builder
# from datasets import load_dataset
# from datasets import get_dataset_config_names
# #"ShapeNet/ShapeNetCore"  "rotten_tomatoes"
# ds_builder = load_dataset_builder("ShapeNet/ShapeNetCore")
# # dataset = load_dataset("ShapeNet/ShapeNetCore", Token=True)
# print(ds_builder.info.description)
# print(ds_builder.info.features)
# #get_dataset_config_names("ShapeNet/ShapeNetCore")
# # load_dataset('LOADING_SCRIPT', cache_dir="PATH/TO/MY/CACHE/DIR")
# from datasets import load_dataset
# dataset = load_dataset("/scratch/sommerl/repos/NeMo/ShapeNetCore")
# # git clone git@hf.co:datasets/ShapeNet/ShapeNetCore

from od3d.cv.render.gaussian_splats import render_gaussians

import torch

import open3d as o3d
import numpy as np

fpath = '/misc/lmbraid19/sommerl/datasets/CO3D/car/469_66192_130586/pointcloud.ply'
fpath = '/misc/lmbraid19/sommerl/datasets/CO3D/car/185_19982_37678/pointcloud.ply'

#from od3d.cv.visual.show import show_scene
#from od3d.cv.io import read_pts3d
#pts3d = read_pts3d(fpath)
#show_scene(pts3d=[pts3d])
#
pcd = o3d.io.read_point_cloud(fpath)
size_pts = 0.03
opacity = 10.1
pcd = pcd.voxel_down_sample(voxel_size=size_pts)
N = len(pcd.points)

device = 'cuda'
dtype = torch.float
image_width = 256
image_height = 256
zfar = 100.0
znear = 0.01
fx = 500. # x/z = fx * x'
fy = 500.
z_dist = 10.

cam_tform_obj = torch.eye(4).to(device)
cam_tform_obj[0, 3] = 1.5
cam_tform_obj[1, 3] = 0.
cam_tform_obj[2, 3] = z_dist
cam_intr = torch.eye(4).to(device)
cam_intr[0, 0] = fx
cam_intr[1, 1] = fy
cam_intr[0, 2] = -image_width / 2.
cam_intr[1, 2] = -image_height / 2.

# campos = torch.Tensor([0.0, 0.0, -z_dist]).to(device)
# viewmatrix = torch.eye(4).to(device)
# viewmatrix[2, 3] = z_dist
# viewmatrix = viewmatrix.T
# bg = torch.zeros((num_channels,)).to(device)
# bg[:3] = 0.


rgb = torch.zeros((N, 3)).to(device)
rgb[:, :3] = torch.from_numpy(np.asarray(pcd.colors)).to(device, dtype) # N x 3
means3d = torch.from_numpy(np.asarray(pcd.points)).to(device, dtype)  # N x 3
means3d = means3d - means3d.mean(dim=0, keepdim=True)
pts3d_mask = torch.ones((N,)).to(device, bool)

img = render_gaussians(
        cams_tform4x4_obj=cam_tform_obj[None,],
        cams_intr4x4=cam_intr[None,],
        imgs_size=torch.Tensor([image_height, image_width]).to(device),
        pts3d=means3d[None,],
        pts3d_mask=pts3d_mask[None,],
        feats=rgb[None,],
        opacity=opacity,
        pts3d_size=size_pts,
        z_far=zfar,
        z_near=znear,
        feats_dim_base=32,
)[0]

from od3d.cv.visual.show import show_img
show_img(img)

a = torch.Tensor([[ 8.6603e-01, -2.5000e-01,  4.3301e-01, -1.4901e-08],
         [ 5.0000e-01,  4.3301e-01, -7.5000e-01,  2.9802e-08],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  8.7890e-01],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])