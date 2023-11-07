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

        # rand_category_rand_instance_ids = [22, 23]#  344 349 337 334 322 324
        logger.info(f'chosen category is {category}')
        logger.info(f'chosen ids are {rand_category_rand_instance_ids}')

        mesh_source = CUBOID_SOURCES.DEFAULT
        import math
        uniform_objs_tform_obj = get_spherical_uniform_tform4x4(azim_min = -math.pi / 2 , azim_max= + math.pi / 2, azim_steps=3,
                                                                elev_min=-math.pi / 2, elev_max=-math.pi / 2,
                                                                elev_steps=1, theta_min=0., theta_max=0., theta_steps=1).to(device=device)
        P = len(uniform_objs_tform_obj)

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

        proposal_dist_ref_geo_avg, proposal_dist_ref_appear_avg = score_tform4x4_fit(pts=pts2, pts_ref=pts1, tform4x4=uniform_objs_tform_obj, dist_ref=dist_2_1, return_dists=True, score_perc=0.7)
        meshes = [mesh1]
        lines = []
        for t in range(P):
            uniform_obj_tform_obj = uniform_objs_tform_obj[t]
            uniform_obj_tform_obj[:3, 3] = 0.
            uniform_obj_tform_obj[0, 3] = 1. # shift in x
            uniform_obj_tform_obj[2, 3] = -1. * (t - P // 2) # shift in z
            mesh2_verts = transf3d_broadcast(pts3d=mesh2.verts.to(device=device, dtype=dtype), transf4x4=uniform_obj_tform_obj.to(device=device))
            mesh = Mesh(verts=mesh2_verts,
                        faces=mesh2.faces, rgb=mesh2.rgb)
            meshes.append(mesh)
            t_lines = torch.stack([mesh2_verts[pts_ids[t]], pts1[pts1_ids[t]]], dim=-1).permute(0, 2, 1)
            lines.append(t_lines)

        imgs= show_scene(pts3d=[], pts3d_colors=[], device=device, lines3d=lines,
                         meshes=meshes,
                         meshes_add_translation=False, pts3d_add_translation=False,
                         return_visualization=True, viewpoints_count=1, crop_white_border=True)

        imgs = crop_white_border_from_img(imgs, white_pad=30)
        H, W = imgs.shape[-2:]
        # show_imgs(imgs, height=640, width=1280)

        fig, ax = plt.subplots(P, 1)
        for p in range(P):
            x = ['geometry', 'appearance']
            y = [proposal_dist_ref_geo_avg[p].item(), proposal_dist_ref_appear_avg[p].item()]
            bar_labels = ['geometry', 'appearance']
            bar_colors = ['tab:blue', 'tab:green']
            ax[p].bar(x, y, label=bar_labels, color=bar_colors,  width=0.8)
            ax[p].set_ylabel(None) # 'distance'
            ax[p].set_title('')
            ax[p].set_ylim([0., 1.])
            ax[p].set_xlim([-0.5, 1.5])

            # ax[p].legend(title='Distance')

        img = get_img_from_plot(ax=ax, fig=fig, axis_off=False)
        img = resize(img, scale_factor=H/img.shape[-2])
        total_imgs = torch.cat([imgs* 255, img], dim=-1)
        show_img(total_imgs, height=640, width=1280, fpath='method_align3d.png')
        show_img(total_imgs, height=640, width=1280)

@app.command()
def mv_pose():
    logging.basicConfig(level=logging.INFO)
    device = 'cuda'
    dtype = torch.float
    co3d = CO3D.create_by_name('co3d_5s_no_zsp_aligned') #co3d_5s_no_zsp_labeled 'co3d_50s_no_zsp_aligned' 'co3dv1_10s_zsp_aligned' 'co3d_10s_zsp_aligned' 'co3dv1_10s_zsp_unlabeled'
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
        samples_count = 5
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
        mesh.rgb = mesh.get_verts_ncds_cat_with_mesh_ids()
        #pts3d = transf3d_broadcast(pts3d=pts3d.to(device=device, dtype=dtype),
        #                           transf4x4=seq1.droid_slam_aligned_tform_droid_slam.to(device=device))
        # meshes=mesh,

        cams_imgs_depth_scale=0.2
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
            logger.info('writing img...')
            show_imgs(torch.cat([img_real, img_rendered], dim=-1), height=640, width=1280, fpath=f'mv_pose_{i}.png')

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
                droid_slam_aligned_tform_droid_slam = sequences[instance_id].droid_slam_aligned_tform_droid_slam.to(
                    device=device, dtype=dtype)
                pts3d_first = transf3d_broadcast(pts3d=sequences[instance_id].get_pcl(pcl_source=PCL_SOURCES.DROID_SLAM_CLEAN).to(device=device, dtype=dtype),transf4x4=droid_slam_aligned_tform_droid_slam)

                offset_y = offset_y_prev - pts3d_first[:, 1].min().item() * 1.2
                offset_y_prev = offset_y + pts3d_first[:, 1].max().item() * 1.2
                logger.info(f'category {category} offset {offset_y}')
            droid_slam_aligned_tform_droid_slam = sequences[instance_id].droid_slam_aligned_tform_droid_slam.to(device=device, dtype=dtype)

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