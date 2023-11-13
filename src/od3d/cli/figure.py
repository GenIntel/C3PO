import logging
import random

logger = logging.getLogger(__name__)
import typer
import od3d.io

app = typer.Typer()


from od3d.datasets.co3d import CO3D
from od3d.cv.visual.show import show_scene
import torch

from od3d.cv.geometry.transform import transf3d_broadcast, tform4x4_from_transl3d, tform4x4, get_spherical_uniform_tform4x4
from od3d.datasets.co3d.enum import PCL_SOURCES, CUBOID_SOURCES, CAM_TFORM_OBJ_SOURCES
from od3d.cv.geometry.mesh import Meshes, Mesh
from typing import List

from od3d.cv.visual.show import show_imgs, get_img_from_plot, show_img
from od3d.cv.visual.crop import crop_white_border_from_img
import matplotlib.pyplot as plt
from od3d.cv.visual.resize import resize
from od3d.datasets.pascal3d.dataset import Pascal3D
from od3d.datasets.objectnet3d.dataset import ObjectNet3D

from od3d.methods.nemo import NeMo
from pathlib import Path

from omegaconf.omegaconf import OmegaConf  # bottle, suitcase, cup, airplane, microwave, tv, train, hairdryer, remote
from od3d.cv.transforms.transform import OD3D_Transform
import torch.utils.data
from od3d.cv.geometry.transform import inv_tform4x4
from tqdm import tqdm
import open3d

@app.command()
def viewpoints():
    logging.basicConfig(level=logging.INFO)



    bunny_data = open3d.data.BunnyMesh()
    bunny_mesh_open3d = open3d.io.read_triangle_mesh(bunny_data.path)
    bunny_mesh = Meshes.load_from_meshes([Mesh.from_o3d(bunny_mesh_open3d)])
    bunny_rot = torch.eye(4)
    bunny_rot = torch.Tensor(
        [[0., 0., 1., 0.,],
         [1., 0., 0., 0.,],
         [0., 1., 0., 0.,],
         [0., 0., 0., 1.,]])


    bunny_mesh.verts.data = transf3d_broadcast(pts3d=bunny_mesh.verts, transf4x4=bunny_rot)
    bunny_mesh.rgb = bunny_mesh.get_verts_ncds_cat_with_mesh_ids()
    # 'bottle', 'train'
    # 'bottle', 'suitcase', 'cup', 'microwave', 'tv', 'train', 'hairdryer', 'remote'
    categories = ['bottle', 'train', 'airplane']
    for category in categories:
        logger.info(f'category: {category}')
        config = {'categories': [category]}
        #dataset = ObjectNet3D.create_by_name('objectnet3d_test', config=config)
        dataset = Pascal3D.create_by_name('pascal3d_test', config=config)

        dataset.transform = OD3D_Transform.create_by_name('scale_mask_separate_centerzoom512')
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=10, shuffle=False, collate_fn=dataset.collate_fn, num_workers=4)
        logging.info(f"Dataset contains {len(dataset)} frames.")
        all_vpts = []
        all_cams = []
        cam_intr4x4 = None
        for i, batch in tqdm(enumerate(dataloader)):
            if torch.cuda.is_available():
                batch.to(device='cuda:0')

            cam_intr4x4 = batch.cam_intr4x4[0]
            all_cams.append(batch.cam_tform4x4_obj)

            vpts = inv_tform4x4(batch.cam_tform4x4_obj)[:, :3, 3]
            all_vpts.append(vpts)

        all_vpts = torch.cat(all_vpts, dim=0)
        all_cams = torch.cat(all_cams, dim=0)
        all_vpts = torch.nn.functional.normalize(all_vpts, dim=-1)
        all_cams[:, :3, 3] = torch.nn.functional.normalize(all_cams[:, :3, 3], dim=-1)

        #img = show_scene(pts3d=[all_vpts], return_visualization=True, viewpoints_count=1, crop_white_border=True)
        imgs = show_scene(meshes=[bunny_mesh], cams_intr4x4=cam_intr4x4, cams_tform4x4_world=all_cams, return_visualization=True, viewpoints_count=3, crop_white_border=True, cams_imgs_depth_scale=0.05)
        img = imgs[1] #  img.permute(1, 2, 0, 3).flatten(2)
        #img = torch.cat(img, dim=-1)
        img = crop_white_border_from_img(img, white_pad=20)
        show_img(img, height=1080, width=1980, fpath=f'{dataset.name}_azimuth_{category}.png')


@app.command()
def align3d():
    logging.basicConfig(level=logging.INFO)
    device = 'cuda'
    dtype = torch.float
    co3d = CO3D.create_by_name('co3d_5s_no_zsp_labeled') #co3d_5s_no_zsp_labeled 'co3d_50s_no_zsp_aligned' 'co3dv1_10s_zsp_aligned' 'co3d_10s_zsp_aligned' 'co3dv1_10s_zsp_unlabeled'
    categories = co3d.categories
    sequences = co3d.get_sequences()
    sequences_unique_names = [seq.name_unique for seq in sequences]
    instances_count = len(sequences)
    map_seq_to_cat = torch.LongTensor([categories.index(name.split('/')[0]) for name in sequences_unique_names])
    instance_ids = torch.LongTensor(list(range(instances_count)))

    category = 'chair'
    category = 'bicycle'

    rand_category_id = categories.index(category) # 'car', 'chair',
    rand_category_instance_ids = instance_ids[map_seq_to_cat == rand_category_id]

    # # CALCULATING CATEGORICAL PCA
    # category_instance_ids = instance_ids[map_seq_to_cat == rand_category_id]
    # categorical_features = []
    # for instance_id_in_category, instance_id in enumerate(category_instance_ids):
    #     instance_feats = sequences[instance_id].feats
    #
    #     if isinstance(instance_feats, List):
    #         categorical_features += torch.cat([vert_feats for vert_feats in instance_feats], dim=0)
    #     else:
    #         categorical_features.append(instance_feats)
    # categorical_features = torch.stack(categorical_features, dim=0)
    # _, _, categorical_pca_V = torch.pca_lowrank(categorical_features)
    # sequences[category_instance_ids[0]].categorical_pca_V = categorical_pca_V

    while True:
        rand_category_rand_instance_ids = random.sample(rand_category_instance_ids.tolist(), k=2)

        rand_category_rand_instance_ids = [3, 4]#  344 349 337 334 322 324
        logger.info(f'chosen category is {category}')
        logger.info(f'chosen ids are {rand_category_rand_instance_ids}')

        mesh_source = CUBOID_SOURCES.DEFAULT
        import math
        #uniform_objs_tform_obj = get_spherical_uniform_tform4x4(azim_min = -math.pi / 2 , azim_max= + math.pi / 2, azim_steps=5,
        #                                                        elev_min=-math.pi / 2, elev_max=-math.pi / 2,
        #                                                        elev_steps=1, theta_min=0., theta_max=0., theta_steps=1).to(device=device)
        P = 3

        seq1 = sequences[rand_category_rand_instance_ids[0]]
        seq2 = sequences[rand_category_rand_instance_ids[1]]
        mesh1 = seq1.get_mesh(mesh_source=mesh_source, add_rgb_from_pca=True, device=device)
        mesh2 = seq2.get_mesh(mesh_source=mesh_source, add_rgb_from_pca=True, device=device)

        pts1 = mesh1.verts.to(device=device)
        pts2 = mesh2.verts.to(device=device)
        dist_2_1 = seq2.get_dist_verts_mesh_feats_to_other_sequence(seq1).to(device=device)
        dist_2_1 = dist_2_1 / 2.
        from od3d.cv.geometry.fit.tform4x4 import score_tform4x4_fit, fit_tform4x4


        N = len(dist_2_1)
        fits_count = P
        fit_pts_count = 4
        pts_sample_probs = torch.ones(size=(fits_count, N)).to(device=device)
        pts_ids = torch.multinomial(pts_sample_probs.view(-1, N), num_samples=fit_pts_count).view(
            (fits_count, fit_pts_count))
        pts1_ids = dist_2_1.argmin(dim=-1)[pts_ids]

        # sequences[rand_category_rand_instance_ids[0]]
        uniform_objs_tform_obj = fit_tform4x4(pts=pts2, pts_ref=pts1, pts_ids=pts_ids, dist_ref=dist_2_1)

        dist_appear_weight = 0.1
        _, proposal_dist_ref_geo_avg, proposal_dist_ref_appear_avg, proposal_dist_src_ref_2d_ids, proposal_dist_src_ref_weights = \
            score_tform4x4_fit(pts=pts2, pts_ref=pts1, tform4x4=uniform_objs_tform_obj, dist_ref=dist_2_1, return_dists=True, return_weights=True, cyclic_weight_temp=0.7, dist_appear_weight=dist_appear_weight)


        proposal_dist_ref_geo_avg = proposal_dist_ref_geo_avg * (1. - dist_appear_weight)
        proposal_dist_ref_appear_avg = proposal_dist_ref_appear_avg * dist_appear_weight

        from od3d.cv.select import batched_index_fill
        #mesh1_weights = torch.zeros_like(mesh1.rgb[:, :1])
        #mesh1_weights = batched_index_fill(input=mesh1_weights.permute(1, 0).repeat(3, 1), value=proposal_dist_src_ref_weights,  index=proposal_dist_src_ref_2d_ids[:, :, 1]).permute(1, 0)
        #mesh1_weights = mesh1_weights.mean(dim=1, keepdim=True)
        #mesh1_weights = mesh1_weights / mesh1_weights.max(dim=0, keepdim=True).values
        #mesh1.rgb *= mesh1_weights

        mesh2_weights = torch.zeros_like(mesh2.rgb[:, :1])
        mesh2_weights = batched_index_fill(input=mesh2_weights.permute(1, 0).repeat(P, 1), value=proposal_dist_src_ref_weights,  index=proposal_dist_src_ref_2d_ids[:, :, 0]).permute(1, 0)
        mesh2_weights = mesh2_weights / mesh2_weights.max(dim=0, keepdim=True).values

        meshes = [mesh1]
        lines = []
        for t in range(P):
            uniform_obj_tform_obj = uniform_objs_tform_obj[t]
            uniform_obj_tform_obj[:3, 3] = 0.
            uniform_obj_tform_obj[0, 3] = 1. # shift in x
            uniform_obj_tform_obj[2, 3] = -1. * (t - P // 2) # shift in z
            mesh2_verts = transf3d_broadcast(pts3d=mesh2.verts.to(device=device, dtype=dtype), transf4x4=uniform_obj_tform_obj.to(device=device))

            #uniform_obj_tform_obj = torch.eye(4).to(device=device)
            uniform_obj_tform_obj[:3, 3] = 0.
            uniform_obj_tform_obj[0, 3] = 1.5  # shift in x
            uniform_obj_tform_obj[2, 3] = -1. * (t - P // 2)  # shift in z
            mesh2_weights_verts = transf3d_broadcast(pts3d=mesh2.verts.to(device=device, dtype=dtype), transf4x4=uniform_obj_tform_obj.to(device=device))
            mesh2_weights_rgb = torch.ones_like(mesh2.rgb)
            mesh2_weights_rgb *= mesh2_weights[:, t:t+1].clamp(0, 1)
            mesh = Mesh(verts=mesh2_verts,
                        faces=mesh2.faces, rgb=mesh2.rgb)
            mesh_weights = Mesh(verts=mesh2_weights_verts,
                        faces=mesh2.faces, rgb=mesh2_weights_rgb)
            meshes.append(mesh)
            meshes.append(mesh_weights)
            t_lines = torch.stack([mesh2_verts[pts_ids[t]], pts1[pts1_ids[t]]], dim=-1).permute(0, 2, 1)
            lines.append(t_lines)

        imgs= show_scene(pts3d=[], pts3d_colors=[], device=device, lines3d=lines,
                         meshes=meshes,
                         meshes_add_translation=False, pts3d_add_translation=False,
                         return_visualization=True, viewpoints_count=1, crop_white_border=True)

        imgs = crop_white_border_from_img(imgs, white_pad=30)
        H, W = imgs.shape[-2:]
        # show_imgs(imgs, height=640, width=1280)

        max_y = (proposal_dist_ref_geo_avg + proposal_dist_ref_appear_avg).max().item()
        fig, ax = plt.subplots(P, 1, figsize=(2, 6))
        for p in range(P):
            dist_geo = proposal_dist_ref_geo_avg[p].item()
            dist_appear = proposal_dist_ref_appear_avg[p].item()
            dist_total = dist_geo + dist_appear
            #if p == P-1:
            #    x = ['total', 'geometry', 'appearance']
            #else:
            x = [0, 1, 2] #, 'appearance']
            y = [dist_total, dist_geo, dist_appear]
            bar_labels = ['total', 'geometry', 'appearance']
            bar_colors = ['cornflowerblue', 'slategrey', 'lightgreen'] #''tab:green']
            ax[p].bar(x, y, label=bar_labels, color=bar_colors,  width=0.8)
            if p == 0:
                ax[p].legend(title='')
            # ax[p].set_ylabel(None) # 'distance'
            ax[p].set_title('')
            ax[p].set_xticks([], minor=False)
            ax[p].set_yticks([], minor=False)
            ax[p].set_ylim([0., max_y * 1.1])
            ax[p].set_xlim([-0.55, 2.55])
            # ax[p].legend(title='Distance')

        img = get_img_from_plot(ax=ax, fig=fig, axis_off=False)
        img = resize(img, scale_factor=H/img.shape[-2])
        total_imgs = torch.cat([imgs* 255, img], dim=-1)
        show_img(total_imgs, height=1080, width=1980, fpath='method_align3d.png')
        show_img(total_imgs, height=1080, width=1980)

@app.command()
def mv_pose_inference():
    logging.basicConfig(level=logging.INFO)
    device = 'cuda'
    dtype = torch.float

    run_name = '11-11_20-30-50_CO3D_NeMo_cat1_bicycle_ref4_filtered_mesh_slurm'
    mesh_fpath = Path('/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/aligned/all_20s_to_5s_mesh/r4/mesh/bicycle/mesh.ply')
    aligned_name = 'all_20s_to_5s_mesh/r4'
    # od3d bench rsync -r 11-11_20-30-50_CO3D_NeMo_cat1_bicycle_ref4_filtered_mesh_slurm

    # /misc/lmbraid19/sommerl/exps/11-11_20-30-27_CO3D_NeMo_cat1_bicycle_ref0_filtered_mesh_slurm/nemo.ckpt

    #config_loaded.train.transform.transforms[0].config = None
    #config_loaded.categories = ['bicycle']
    #config_loaded.fpaths_meshes = {'bicycle': ''}
    config_transform = od3d.io.read_config_intern(rfpath=Path("methods").joinpath('transform', f"scale_mask_shorter_1_centerzoom512.yaml"))
    nemo = NeMo.create_by_name('nemo',
                               logging_dir=Path('nemo_out'),
                               config={'texture_dataset': None,
                                        'train': {'transform': {'transforms': [config_transform]}},
                                       'categories': ['bicycle'],
                                       'fpaths_meshes': {'bicycle': str(mesh_fpath)},
                                       'checkpoint': f'/misc/lmbraid19/sommerl/exps/{run_name}/nemo.ckpt',
                                       'multiview': {'batch_size': 3}
                                       })
    #config_dataset = {'categories': ['bicycle'], 'dict_nested_frames': {'val': ['n03792782_687']}} # n03792782_6218, n03792782_687

    sequences_count = 10
    samples_count = 3
    samples_max_position = 2
    mv_final_count = 2
    category = 'bicycle'

    co3d = CO3D.create_by_name('co3d_no_zsp_20s', config={'categories': [category], 'aligned_name': aligned_name, 'sequences_count_max_per_category': sequences_count}) # co3d_no_zsp_20s_aligned #co3d_5s_no_zsp_labeled 'co3d_50s_no_zsp_aligned' 'co3dv1_10s_zsp_aligned' 'co3d_10s_zsp_aligned' 'co3dv1_10s_zsp_unlabeled'
    categories = co3d.categories
    sequences = co3d.get_sequences()
    sequences_unique_names = [seq.name_unique for seq in sequences]
    instances_count = len(sequences)
    map_seq_to_cat = torch.LongTensor([categories.index(name.split('/')[0]) for name in sequences_unique_names])
    instance_ids = torch.LongTensor(list(range(instances_count)))

    dict_category_sequences = {category: list(sequence_dict.keys()) for category, sequence_dict in co3d.dict_nested_frames.items()}
    co3d.transform = nemo.transform_train
    # co3d.transform = nemo.transform_train

    dataset_sub = co3d.get_subset_by_sequences(dict_category_sequences=dict_category_sequences,
                                                  frames_count_max_per_sequence=nemo.config.multiview.batch_size)
    dataset_sub.transform = nemo.transform_train

    dataloader = torch.utils.data.DataLoader(dataset=dataset_sub, batch_size=nemo.config.multiview.batch_size,
                                             shuffle=False,
                                             collate_fn=dataset_sub.collate_fn,
                                             num_workers=nemo.config.test.dataloader.num_workers,
                                             pin_memory=nemo.config.test.dataloader.pin_memory)

    nemo.meshes.to(device=device)
    nemo.net.to(device=device)
    nemo.net.eval()
    from od3d.cv.geometry.mesh import MESH_RENDER_MODALITIES
    # next(nemo.net.parameters()).is_cuda
    for i, batch in tqdm(enumerate(dataloader)):
        if torch.cuda.is_available():
            batch.to(device=device)

        # show_imgs(rgbs=batch.rgb[:, :5])
        batch_res = nemo.inference_batch_multiview(batch)
        samples_cam_tform4x4_obj = batch_res['samples_cam_tform4x4_obj']
        pred_cam_tform4x4_obj = batch_res['cam_tform4x4_obj']
        C = samples_cam_tform4x4_obj.shape[1]
        sim = batch_res['samples_sim'].mean(dim=0)  #  batch_res['samples_sim']
        samples_ids = random.sample(torch.arange(C).tolist(), samples_count)
        samples_ids[samples_max_position] = sim.argmax().item()
        samples_ids = torch.LongTensor(samples_ids).to(device=device)
        samples_sim = sim[samples_ids]

        sims = torch.cat([torch.Tensor([samples_sim.min(), samples_sim.min()]).to(device=device), samples_sim, batch_res['sim'].mean(dim=0)], dim=0).detach()
        sims = (sims - sims.min()) / (sims.max() - sims.min())
        sims[2:] += 0.1

        S = len(sims)
        fig, ax = plt.subplots(1, 1, figsize=(6, 1))
        x = torch.arange(S).tolist()
        y = (sims).tolist()
        #bar_labels = ['total', 'geometry', 'appearance']
        #label = bar_labels
        #bar_colors = ['cornflowerblue', 'slategrey', 'lightgreen'] #''tab:green']
        # color=bar_colors
        ax.bar(x, y,  color='slategrey', width=0.4)
        # ax.legend(title='log probability')
        # ax[p].set_ylabel(None) # 'distance'
        ax.set_title('')
        ax.set_xticks([], minor=False)
        ax.set_yticks([], minor=False)
        ax.set_ylim([0., 1.2])
        ax.set_xlim([-0.55, S - 1 + 0.55])
        img_log_prob = get_img_from_plot(ax=ax, fig=fig, axis_off=True)
        #img_log_prob = resize(img_log_prob, )
        # show_img(img_log_prob)

        cam_intr4x4 = batch.cam_intr4x4
        img_feats = nemo.net(batch.rgb)
        size = torch.Tensor([img_feats.shape[2] * nemo.down_sample_rate, img_feats.shape[3] * nemo.down_sample_rate]).to(device=device)
        B, F, H, W = img_feats.shape
        mesh_feats = nemo.meshes.render_feats(cams_tform4x4_obj=samples_cam_tform4x4_obj,
                                              cams_intr4x4=cam_intr4x4[:, None],
                                              imgs_sizes=size, meshes_ids=batch.label,
                                              modality=MESH_RENDER_MODALITIES.FEATS,
                                              down_sample_rate=nemo.down_sample_rate,
                                              broadcast_batch_and_cams=True)
        C = mesh_feats.shape[1]
        _, _, pca_V = torch.pca_lowrank(torch.cat([img_feats.permute(0, 2, 3, 1).reshape(-1, F), mesh_feats.permute(0, 1, 3, 4, 2).reshape(-1, F)], dim=0))
        img_feats_pca = torch.matmul(img_feats.permute(0, 2, 3, 1).reshape(-1, F), pca_V[:, 1:4]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        mesh_feats_pca = torch.matmul(mesh_feats.permute(0, 1, 3, 4, 2).reshape(-1, F), pca_V[:, 1:4]).reshape(B, C, H, W, -1).permute(0, 1, 4, 2, 3)
        mesh_feats_pca_mask_bg = (mesh_feats_pca == 0.).all(dim=2, keepdim=True)




        mesh_feats_optimal = nemo.meshes.render_feats(cams_tform4x4_obj=pred_cam_tform4x4_obj,
                                                      cams_intr4x4=cam_intr4x4,
                                                      imgs_sizes=size, meshes_ids=batch.label,
                                                      modality=MESH_RENDER_MODALITIES.FEATS,
                                                      down_sample_rate=nemo.down_sample_rate,
                                                      broadcast_batch_and_cams=False)
        mesh_feats_optimal_pca = torch.matmul(mesh_feats_optimal.permute(0, 2, 3, 1).reshape(-1, F), pca_V[:, 1:4]).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        mesh_feats_optimal_pca_mask_bg = (mesh_feats_optimal_pca == 0.).all(dim=1, keepdim=True)

        feats_pca = torch.cat([img_feats_pca[:, None], mesh_feats_pca[:, samples_ids], mesh_feats_optimal_pca[:, None]], dim=1)
        feats_pca = (feats_pca - feats_pca.min()) / (feats_pca.max()- feats_pca.min())
        feats_pca[:, 1:-1][mesh_feats_pca_mask_bg[:, samples_ids].expand(*feats_pca[:, 1:-1].shape)] = 1.
        feats_pca[:, -1][mesh_feats_optimal_pca_mask_bg.expand(*feats_pca[:, -1].shape)] = 1.

        rgb = (batch.rgb - batch.rgb.min()) / (batch.rgb.max() - batch.rgb.min())
        rgb_no_rgb_mask = (rgb < 0.1).all(dim=1, keepdim=True)
        rgb[rgb_no_rgb_mask.expand(*rgb.shape)] = 1.
        _, _, H_rgb, W_rgb = batch.rgb.shape
        scale_factor = H_rgb / H
        feats_pca = resize(feats_pca.reshape(-1, 3, H, W), scale_factor=H_rgb / H, mode='nearest_v2').reshape(B, -1, 3, int(H * scale_factor), int(W* scale_factor))
        img = torch.cat([rgb[:, None], feats_pca], dim=1)
        from od3d.cv.visual.show import imgs_to_img
        img = imgs_to_img(img[:mv_final_count], pad = 10)
        img_log_prob = (resize(img_log_prob, scale_factor=img.shape[-1] / img_log_prob.shape[-1])).to(device=device)
        img_mv_pose_inference = torch.cat([img * 255, img_log_prob], dim=-2)
        show_img(rgb=img_mv_pose_inference, height=1080, width=1980, fpath='mv_pose_inference.png')
        show_img(rgb=img_mv_pose_inference, height=1080, width=1980)


    rand_category_id = categories.index(category) # 'car', 'chair',
    rand_category_instance_ids = instance_ids[map_seq_to_cat == rand_category_id]

    # # CALCULATING CATEGORICAL PCA
    # category_instance_ids = instance_ids[map_seq_to_cat == rand_category_id]
    # categorical_features = []
    # for instance_id_in_category, instance_id in enumerate(category_instance_ids):
    #     instance_feats = sequences[instance_id].feats
    #
    #     if isinstance(instance_feats, List):
    #         categorical_features += torch.cat([vert_feats for vert_feats in instance_feats], dim=0)
    #     else:
    #         categorical_features.append(instance_feats)
    # categorical_features = torch.stack(categorical_features, dim=0)
    # _, _, categorical_pca_V = torch.pca_lowrank(categorical_features)
    # sequences[category_instance_ids[0]].categorical_pca_V = categorical_pca_V


    rand_category_rand_instance_ids = random.sample(rand_category_instance_ids.tolist(), k=1)
    seq1 = sequences[rand_category_rand_instance_ids[0]]


@app.command()
def mv_pose_train():
    logging.basicConfig(level=logging.INFO)
    device = 'cuda'
    dtype = torch.float

    run_name = '11-11_20-30-50_CO3D_NeMo_cat1_bicycle_ref4_filtered_mesh_slurm'
    mesh_fpath = Path('/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/aligned/all_20s_to_5s_mesh/r4/mesh/bicycle/mesh.ply')
    aligned_name = 'all_20s_to_5s_mesh/r4'
    # od3d bench rsync -r 11-11_20-30-50_CO3D_NeMo_cat1_bicycle_ref4_filtered_mesh_slurm

    # /misc/lmbraid19/sommerl/exps/11-11_20-30-27_CO3D_NeMo_cat1_bicycle_ref0_filtered_mesh_slurm/nemo.ckpt

    #config_loaded.train.transform.transforms[0].config = None
    #config_loaded.categories = ['bicycle']
    #config_loaded.fpaths_meshes = {'bicycle': ''}
    config_transform = od3d.io.read_config_intern(rfpath=Path("methods").joinpath('transform', f"scale_mask_shorter_1_centerzoom512.yaml"))
    nemo = NeMo.create_by_name('nemo',
                               logging_dir=Path('nemo_out'),
                               config={'texture_dataset': None,
                                        'train': {'transform': {'transforms': [config_transform]}},
                                       'categories': ['bicycle'],
                                       'fpaths_meshes': {'bicycle': str(mesh_fpath)},
                                       'checkpoint': f'/misc/lmbraid19/sommerl/exps/{run_name}/nemo.ckpt'})
    config_dataset = {'categories': ['bicycle'], 'dict_nested_frames': {'val': ['n03792782_687']}} # n03792782_6218, n03792782_687
    dataset = ObjectNet3D.create_by_name('objectnet3d', config=config_dataset)
    dataset.transform = nemo.transform_train  # OD3D_Transform.create_by_name('scale_mask_separate_centerzoom512')
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False,
                                             collate_fn=dataset.collate_fn, num_workers=4)
    nemo.net.to(device=device)
    nemo.net.eval()
    # next(nemo.net.parameters()).is_cuda
    for i, batch in tqdm(enumerate(dataloader)):
        if torch.cuda.is_available():
            batch.to(device=device)

        from od3d.cv.visual.blend import blend_rgb
        batch_res = nemo.inference_batch_single_view(batch)

        #batch_res.keys()
        pred_cam_tform4x4_obj = batch_res['cam_tform4x4_obj']
        pred_verts_ncds = nemo.get_ncds_with_cam(cam_intr4x4=batch.cam_intr4x4, cam_tform4x4_obj=pred_cam_tform4x4_obj,
                                             categories_ids=batch.label, size=batch.size,
                                             down_sample_rate=1., pre_rendered=False)
        for b in range(len(batch)):
            img_rgb = batch.rgb[b].clone()
            img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min())
            img_in_the_wild = blend_rgb(resize(img_rgb, scale_factor=1.), pred_verts_ncds[b], alpha1=0.7, alpha2=0.7)
            show_img(img_in_the_wild, fpath='mv_pose_in_the_wild.png')

    samples_count = 3
    sequences_count = 3
    co3d = CO3D.create_by_name('co3d_no_zsp_20s_aligned', config={'aligned_name': aligned_name, 'sequences_count_max_per_category': sequences_count}) # co3d_no_zsp_20s_aligned #co3d_5s_no_zsp_labeled 'co3d_50s_no_zsp_aligned' 'co3dv1_10s_zsp_aligned' 'co3d_10s_zsp_aligned' 'co3dv1_10s_zsp_unlabeled'
    categories = co3d.categories
    sequences = co3d.get_sequences()
    sequences_unique_names = [seq.name_unique for seq in sequences]
    instances_count = len(sequences)
    map_seq_to_cat = torch.LongTensor([categories.index(name.split('/')[0]) for name in sequences_unique_names])
    instance_ids = torch.LongTensor(list(range(instances_count)))

    category = 'bicycle'
    rand_category_id = categories.index(category) # 'car', 'chair',
    rand_category_instance_ids = instance_ids[map_seq_to_cat == rand_category_id]

    # # CALCULATING CATEGORICAL PCA
    # category_instance_ids = instance_ids[map_seq_to_cat == rand_category_id]
    # categorical_features = []
    # for instance_id_in_category, instance_id in enumerate(category_instance_ids):
    #     instance_feats = sequences[instance_id].feats
    #
    #     if isinstance(instance_feats, List):
    #         categorical_features += torch.cat([vert_feats for vert_feats in instance_feats], dim=0)
    #     else:
    #         categorical_features.append(instance_feats)
    # categorical_features = torch.stack(categorical_features, dim=0)
    # _, _, categorical_pca_V = torch.pca_lowrank(categorical_features)
    # sequences[category_instance_ids[0]].categorical_pca_V = categorical_pca_V

    while True:

        rand_category_rand_instance_ids = random.sample(rand_category_instance_ids.tolist(), k=samples_count)
        seq1 = sequences[rand_category_rand_instance_ids[0]]
        cams_imgs = []
        cams_intr4x4 = []
        cams_tform4x4_obj = []
        for s in range(samples_count):
            seq = sequences[rand_category_rand_instance_ids[s]]
            frames_ids = torch.arange(seq.frames_count)[::50]
            frames = seq.get_frames(frames_ids=frames_ids)
            _cams_imgs = torch.stack([frame.rgb for frame in frames], dim=0).to(device=device)
            _cams_intr4x4 = torch.stack([frame.cam_intr4x4 for frame in frames], dim=0).to(device=device)
            _cams_tform4x4_obj = torch.stack([frame.cam_tform4x4_obj for frame in frames], dim=0).to(device=device)
            cams_imgs += [cam_img.to(device=device) for cam_img in _cams_imgs]
            cams_intr4x4.append(_cams_intr4x4)
            cams_tform4x4_obj.append(_cams_tform4x4_obj)

        cams_intr4x4 = torch.cat(cams_intr4x4, dim=0).to(device=device)
        cams_tform4x4_obj = torch.cat(cams_tform4x4_obj, dim=0).to(device=device)

        #pts3d = seq1.get_pcl(pcl_source=seq1.pcl_source)
        #pts3d_colors = seq1.get_pcl_colors(pcl_source=seq1.pcl_source)
        from od3d.cv.visual.show import show_scene
        mesh = Meshes.load_from_meshes([seq1.get_mesh(mesh_source=CUBOID_SOURCES.ALIGNED)], device=device)
        #mesh = Meshes.load_from_files(fpaths_meshes=[mesh_fpath], device=device)
        mesh.rgb = mesh.get_verts_ncds_cat_with_mesh_ids()
        #pts3d = transf3d_broadcast(pts3d=pts3d.to(device=device, dtype=dtype),
        #                           transf4x4=seq1.droid_slam_aligned_tform_droid_slam.to(device=device))
        # meshes=mesh,

        cams_imgs_depth_scale=0.3
        viewpoints_count = 5
        logger.info('rendering real images...')
        imgs_real = show_scene(meshes=mesh, cams_tform4x4_world=cams_tform4x4_obj, cams_intr4x4=cams_intr4x4, cams_imgs=cams_imgs, viewpoints_count=viewpoints_count, return_visualization=True, crop_white_border=True, cams_imgs_depth_scale=cams_imgs_depth_scale, cams_show_wireframe=False)
        # frames_ids = torch.arange(seq1.frames_count)[::50]
        # frames = seq1.get_frames(frames_ids=frames_ids)
        # cams_imgs = torch.stack([frame.rgb for frame in frames], dim=0).to(device=device)
        # cams_intr4x4 = torch.stack([frame.cam_intr4x4 for frame in frames], dim=0).to(device=device)
        # cams_tform4x4_obj = torch.stack([frame.cam_tform4x4_obj for frame in frames], dim=0).to(device=device)
        from od3d.cv.geometry.mesh import MESH_RENDER_MODALITIES

        logger.info('rendering rendered images...')
        cams_imgs_rendered = []
        for c, cam_img in enumerate(cams_imgs):
            img_size = torch.Tensor(list(cams_imgs[c].shape[-2:]))
            cams_imgs_rendered.append(255 * mesh.render_feats(cams_tform4x4_obj=cams_tform4x4_obj[c:c+1], cams_intr4x4=cams_intr4x4[c:c+1], imgs_sizes=img_size, broadcast_batch_and_cams=False, modality=MESH_RENDER_MODALITIES.RGB)[0])

        imgs_rendered = show_scene(meshes=mesh, cams_tform4x4_world=cams_tform4x4_obj, cams_intr4x4=cams_intr4x4, cams_imgs=cams_imgs_rendered, viewpoints_count=viewpoints_count, return_visualization=True, crop_white_border=True, cams_imgs_depth_scale=cams_imgs_depth_scale, cams_show_wireframe=False)

        for i in range(viewpoints_count):
            img_real = crop_white_border_from_img(imgs_real[i], white_pad=30)
            img_rendered = crop_white_border_from_img(imgs_rendered[i], white_pad=30)
            img_rendered = resize(img_rendered, scale_factor=img_real.shape[-2]/img_rendered.shape[-2])
            img_in_the_wild = resize(img_in_the_wild.detach().cpu(), scale_factor=img_real.shape[-2]/img_in_the_wild.shape[-2])
            logger.info('writing img...')
            show_imgs(torch.cat([img_real * 255, img_rendered* 255, img_in_the_wild], dim=-1), height=640, width=1280, fpath=f'mv_pose_train_{i}.png')

@app.command()
def teaser():
    logging.basicConfig(level=logging.INFO)
    viewpoints_count = 2
    # category_meshes = self.meshes.get_meshes_with_ids(meshes_ids=category_instance_ids)
    device = 'cuda:0'
    dtype = torch.float

    co3d = CO3D.create_by_name('co3d_50s_no_zsp_aligned') # 'co3dv1_10s_zsp_aligned' 'co3d_10s_zsp_aligned' 'co3dv1_10s_zsp_unlabeled'
    categories = co3d.categories
    sequences = co3d.get_sequences()


    sequences_unique_names = [seq.name_unique for seq in sequences]
    instances_count = len(sequences)
    map_seq_to_cat = torch.LongTensor([categories.index(name.split('/')[0]) for name in sequences_unique_names])
    categories_count = len(categories)
    instances_count_per_category = [(map_seq_to_cat == c).sum().item() for c in range(categories_count)]
    instance_ids = torch.LongTensor(list(range(instances_count)))

    mesh_source = CUBOID_SOURCES.DEFAULT
    meshes = Meshes.load_from_meshes([seq.get_mesh(mesh_source=mesh_source, add_rgb_from_pca=True, device=device) for seq in sequences], device=device)
    sequences_mesh_ids_for_verts = meshes.get_mesh_ids_for_verts()


    # CALCULATING CATEGORICAL PCA
    logger.info('calculating categorical pca...')
    categorical_features = {}
    categorical_pca_V = {}
    for cat_id, category in enumerate(categories):
        logger.info(category)
        category_instance_ids = instance_ids[map_seq_to_cat == cat_id]
        categorical_features[category] = []
        for instance_id_in_category, instance_id in enumerate(category_instance_ids):
            instance_feats = sequences[instance_id].feats

            if isinstance(instance_feats, List):
                categorical_features[category] += torch.cat([vert_feats for vert_feats in instance_feats], dim=0)
            else:
                categorical_features[category].append(instance_feats)
        categorical_features[category] = torch.stack(categorical_features[category], dim=0)
        _, _, categorical_pca_V[category] = torch.pca_lowrank(categorical_features[category])
        sequences[category_instance_ids[0]].categorical_pca_V = categorical_pca_V[category]

    ## SHOW SINGLE SEQUENCES
    # for sequence in sequences:
    #     logger.info(sequence.name_unique)
    #     mesh = sequence.mesh
    #     instance_feats = sequence.feats
    #     pca_V = sequence.categorical_pca_V.to(device=device)
    #     if isinstance(instance_feats, List):
    #         verts_feats_pca = torch.stack([torch.matmul(vert_feats, pca_V[:, :3]).mean(dim=0) for vert_feats in instance_feats], dim=0)
    #     else:
    #         verts_feats_pca = torch.matmul(instance_feats, pca_V[:, :3])
    #     mesh.rgb = (verts_feats_pca.nan_to_num() + 1.) / 2.
    #     show_scene(meshes=[mesh])
    #     sequence.show(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.DROID_SLAM, pcl_source=PCL_SOURCES.DROID_SLAM_CLEAN, cams_count=200, show_imgs=False)
    #     #sequence.show(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.CO3D, pcl_source=PCL_SOURCES.CO3D, cams_count=200, show_imgs=False)
    #

    offset_x = 3.
    offset_y = 0.
    offset_z = -5.
    offset_y_prev = 0.
    pts3d = []
    pts3d_colors = []

    for cat_id, category in enumerate(categories):
        logger.info(category)
        category_instance_ids = instance_ids[map_seq_to_cat == cat_id]

        for instance_id_in_category, instance_id in enumerate(category_instance_ids):

            if instance_id_in_category == 0:
                droid_slam_aligned_tform_droid_slam = sequences[instance_id].aligned_obj_tform_obj.to(
                    device=device, dtype=dtype)
                pts3d_first = transf3d_broadcast(pts3d=sequences[instance_id].get_pcl(pcl_source=PCL_SOURCES.DROID_SLAM_CLEAN).to(device=device, dtype=dtype),transf4x4=droid_slam_aligned_tform_droid_slam)

                offset_y = offset_y_prev - pts3d_first[:, 1].min().item() * 1.2
                offset_y_prev = offset_y + pts3d_first[:, 1].max().item() * 1.2
                logger.info(f'category {category} offset {offset_y}')
            droid_slam_aligned_tform_droid_slam = sequences[instance_id].aligned_obj_tform_obj.to(device=device, dtype=dtype)

            offset_aligned_tform_aligned = tform4x4_from_transl3d(torch.Tensor([offset_x * instance_id_in_category, offset_y, 0.]).to(device=device, dtype=dtype))

            droid_slam_aligned_tform_droid_slam = tform4x4(offset_aligned_tform_aligned, droid_slam_aligned_tform_droid_slam)

            #droid_slam_labeled_cuboid_tform_droid_slam_instance = tform4x4(droid_slam_labeled_cuboid_tform_droid_slam,
            #                                                               all_pred_ref_tform_src[category][0, instance_id_in_category])

            pts3d.append(transf3d_broadcast(
                pts3d=sequences[instance_id].get_pcl(pcl_source=PCL_SOURCES.DROID_SLAM_CLEAN).to(device=device, dtype=dtype),
                transf4x4=droid_slam_aligned_tform_droid_slam))

            pts3d_colors.append(
                sequences[instance_id].get_pcl_colors(pcl_source=PCL_SOURCES.DROID_SLAM_CLEAN).to(device=device, dtype=dtype))

            if mesh_source == CUBOID_SOURCES.DEFAULT:
                mesh_verts_mask = sequences_mesh_ids_for_verts == instance_id
                offset_aligned_tform_aligned = tform4x4_from_transl3d(
                    torch.Tensor([0., 0., offset_z]).to(device=device, dtype=dtype))
                droid_slam_aligned_tform_droid_slam = tform4x4(offset_aligned_tform_aligned,
                                                               droid_slam_aligned_tform_droid_slam)

                #mesh_verts = meshes.get_verts_with_mesh_id(mesh_id=instance_id)
                meshes.verts[mesh_verts_mask] = transf3d_broadcast(pts3d=meshes.verts[mesh_verts_mask] .to(device=device, dtype=dtype),
                                                                   transf4x4=droid_slam_aligned_tform_droid_slam)

        # category_meshes = meshes.get_meshes_with_ids(meshes_ids=category_instance_ids)

    viewpoints_count = 2
    # show_scene(pts3d=pts3d, pts3d_colors=pts3d_colors, device=device,
    #            meshes=None, # category_meshes,
    #            meshes_add_translation=False, pts3d_add_translation=False,
    #            return_visualization=False, viewpoints_count=viewpoints_count)

    show_scene(pts3d=pts3d, pts3d_colors=pts3d_colors, device=device,
               meshes=meshes,
               meshes_add_translation=False, pts3d_add_translation=False,
               return_visualization=False, viewpoints_count=viewpoints_count)