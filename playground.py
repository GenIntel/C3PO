# from CGAL.CGAL_Kernel import Point_3
# from CGAL.CGAL_Alpha_wrap_3 import alpha_wrap_3
#
# points = [ for ]
# a = alpha_wrap_3(points)


import torch
from od3d.cv.geometry.fit.axis_tform_from_pts3d import axis_tform4x4_obj_from_pts3d
from pathlib import Path
path = Path('/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/tform_obj/label3d/meta/sfm_mask')
path = Path('/misc/lmbraid19/sommerl/datasets/CO3Dv1_Preprocess/tform_obj/label3d/meta_mask/meta')
for category in path.iterdir():
    for sequence in category.iterdir():
        #fpath_in = '/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/tform_obj/label3d/meta/sfm_mask/bottle/38_1661_5028/axis.pt'
        fpath_in = sequence.joinpath('axis.pt')
        if fpath_in.exists():
            fpath_out = Path(fpath_in).parent.joinpath('tform_obj.pt')
            axis_pts3d = torch.load(fpath_in)
            tform_obj = axis_tform4x4_obj_from_pts3d(axis_pts3d=axis_pts3d)
            #tform_obj[:3, 3] = -pts3d.mean(dim=0)
            torch.save(obj=tform_obj, f=fpath_out)

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

# import torch
# def calc_cdist(featsA=torch.zeros(30, 10), featsB=torch.zeros(10, 20), comb='ab,bc->ac'):
#     """
#     Expand and permute a tensor based on the einsum equation.
#
#     Parameters:
#         featsA (torch.Tensor): Input tensor.
#         featsB (torch.Tensor): Input tensor.
#         comb (str): Einsum equation specifying the dimensions.
#
#     Returns:
#         dist (torch.Tensor): Distance tensor.
#     """
#
#     # Parse the einsum equation to get input and output subscripts
#     input_subscripts, output_subscripts = comb.split('->')
#     input_subscripts = input_subscripts.split(',')
#     featsA_input_subscripts = [*input_subscripts[0]]
#     featsB_input_subscripts = [*input_subscripts[1]]
#     removed_subscripts = list(set(featsA_input_subscripts + featsB_input_subscripts) - set([*output_subscripts]))
#     featsA_input_subscripts = ''.join(featsA_input_subscripts)
#     featsB_input_subscripts = ''.join(featsB_input_subscripts)
#     if len(removed_subscripts) > 0:
#         print(f'calc_sim: removed_subscripts={removed_subscripts}')
#         # logger.warning(f'calc_sim: removed_subscripts={removed_subscripts}')
#     removed_subscript = removed_subscripts[0]
#     batch_subscripts = list(
#         set([*input_subscripts[0]]).intersection(set([*input_subscripts[1]])) - set([removed_subscript]))
#     featsA_specific_subscripts = list(
#         set([*input_subscripts[0]]) - set([*batch_subscripts]) - set([removed_subscript]))
#     featsB_specific_subscripts = list(
#         set([*input_subscripts[1]]) - set([*batch_subscripts]) - set([removed_subscript]))
#     print('batch subscripts: ', batch_subscripts)
#     print('feats A specific subscripts: ', featsA_specific_subscripts)
#     print('feats B specific subscripts: ', featsB_specific_subscripts)
#     print('removed subscripts: ', removed_subscript)
#     featsA_intermed_subscripts = ''.join(batch_subscripts + featsA_specific_subscripts + [removed_subscript])
#     featsB_intermed_subscripts = ''.join(batch_subscripts + featsB_specific_subscripts + [removed_subscript])
#     print('feats A subscripts change: ', featsA_input_subscripts + '->' + featsA_intermed_subscripts)
#     print('feats B subscripts change: ', featsB_input_subscripts + '->' + featsB_intermed_subscripts)
#     featsA_bAd = torch.einsum(featsA_input_subscripts + '->' + featsA_intermed_subscripts, featsA)
#     featsB_bBd = torch.einsum(featsB_input_subscripts + '->' + featsB_intermed_subscripts, featsB)
#     batch_shapes = featsA_bAd.shape[:len(batch_subscripts)]
#     featsA_specific_shapes = featsA_bAd.shape[len(batch_subscripts):len(batch_subscripts)+len(featsA_specific_subscripts)]
#     featsB_specific_shapes = featsB_bBd.shape[len(batch_subscripts):len(batch_subscripts)+len(featsB_specific_subscripts)]
#     feats_dim = featsA_bAd.shape[-1]
#     print('batch shapes: ', batch_shapes)
#     print('feats A specific shapes: ', featsA_specific_shapes)
#     print('feats B specific shapes: ', featsB_specific_shapes)
#     print('feats dim: ', feats_dim)
#     dist = torch.cdist(featsA_bAd.reshape(batch_shapes.numel(), featsA_specific_shapes, feats_dim),
#                        featsB_bBd.reshape(batch_shapes.numel(), featsB_specific_shapes, feats_dim))
#     dist = dist.reshape(batch_shapes + featsA_specific_shapes + featsB_specific_shapes)
#     return dist
#
# from od3d.cv.render.gaussian_splats import render_gaussians
#
# import torch
#
# import open3d as o3d
# import numpy as np
#
# fpath = '/misc/lmbraid19/sommerl/datasets/CO3D/car/469_66192_130586/pointcloud.ply'
# fpath = '/misc/lmbraid19/sommerl/datasets/CO3D/car/185_19982_37678/pointcloud.ply'
#
# #from od3d.cv.visual.show import show_scene
# #from od3d.cv.io import read_pts3d
# #pts3d = read_pts3d(fpath)
# #show_scene(pts3d=[pts3d])
# #
# pcd = o3d.io.read_point_cloud(fpath)
# size_pts = 0.03
# opacity = 10.1
# pcd = pcd.voxel_down_sample(voxel_size=size_pts)
# N = len(pcd.points)
#
# device = 'cuda'
# dtype = torch.float
# image_width = 256
# image_height = 256
# zfar = 100.0
# znear = 0.01
# fx = 500. # x/z = fx * x'
# fy = 500.
# z_dist = 10.
#
# cam_tform_obj = torch.eye(4).to(device)
# cam_tform_obj[0, 3] = 1.5
# cam_tform_obj[1, 3] = 0.
# cam_tform_obj[2, 3] = z_dist
# cam_intr = torch.eye(4).to(device)
# cam_intr[0, 0] = fx
# cam_intr[1, 1] = fy
# cam_intr[0, 2] = -image_width / 2.
# cam_intr[1, 2] = -image_height / 2.
#
# # campos = torch.Tensor([0.0, 0.0, -z_dist]).to(device)
# # viewmatrix = torch.eye(4).to(device)
# # viewmatrix[2, 3] = z_dist
# # viewmatrix = viewmatrix.T
# # bg = torch.zeros((num_channels,)).to(device)
# # bg[:3] = 0.
#
#
# rgb = torch.zeros((N, 3)).to(device)
# rgb[:, :3] = torch.from_numpy(np.asarray(pcd.colors)).to(device, dtype) # N x 3
# means3d = torch.from_numpy(np.asarray(pcd.points)).to(device, dtype)  # N x 3
# means3d = means3d - means3d.mean(dim=0, keepdim=True)
# pts3d_mask = torch.ones((N,)).to(device, bool)
#
# img = render_gaussians(
#         cams_tform4x4_obj=cam_tform_obj[None,],
#         cams_intr4x4=cam_intr[None,],
#         imgs_size=torch.Tensor([image_height, image_width]).to(device),
#         pts3d=means3d[None,],
#         pts3d_mask=pts3d_mask[None,],
#         feats=rgb[None,],
#         opacity=opacity,
#         pts3d_size=size_pts,
#         z_far=zfar,
#         z_near=znear,
#         feats_dim_base=32,
# )[0]
#
# from od3d.cv.visual.show import show_img
# show_img(img)
#
# a = torch.Tensor([[ 8.6603e-01, -2.5000e-01,  4.3301e-01, -1.4901e-08],
#          [ 5.0000e-01,  4.3301e-01, -7.5000e-01,  2.9802e-08],
#          [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  8.7890e-01],
#          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])