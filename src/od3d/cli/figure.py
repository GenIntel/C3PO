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

@app.command()
def align3d():
    logging.basicConfig(level=logging.INFO)
    device = 'cuda:0'
    dtype = torch.float
    co3d = CO3D.create_by_name('co3d_50s_no_zsp_aligned') # 'co3d_50s_no_zsp_aligned' 'co3dv1_10s_zsp_aligned' 'co3d_10s_zsp_aligned' 'co3dv1_10s_zsp_unlabeled'
    categories = co3d.categories
    sequences = co3d.get_sequences()
    sequences_unique_names = [seq.name_unique for seq in sequences]
    instances_count = len(sequences)
    map_seq_to_cat = torch.LongTensor([categories.index(name.split('/')[0]) for name in sequences_unique_names])
    instance_ids = torch.LongTensor(list(range(instances_count)))

    category = 'car'
    rand_category_id = categories.index(category) # 'car', 'chair',
    rand_category_instance_ids = instance_ids[map_seq_to_cat == rand_category_id]
    rand_category_rand_instance_ids = random.sample(rand_category_instance_ids.tolist(), k=2)

    rand_category_rand_instance_ids = [45, 42]
    logger.info(f'chosen category is {category}')
    logger.info(f'chosen ids are {rand_category_rand_instance_ids}')

    mesh_source = CUBOID_SOURCES.DEFAULT
    import math
    uniform_objs_tform_obj = get_spherical_uniform_tform4x4(azim_steps=3, azim_max=math.pi - math.pi / 3, elev_min=-math.pi / 2, elev_max=-math.pi / 2, elev_steps=1, theta_min=0., theta_max=0., theta_steps=1)
    mesh1 = sequences[rand_category_rand_instance_ids[0]].get_mesh(mesh_source=mesh_source, add_rgb_from_pca=True, device=device)
    mesh2 = sequences[rand_category_rand_instance_ids[1]].get_mesh(mesh_source=mesh_source, add_rgb_from_pca=True, device=device)
    meshes = [mesh1]
    tform_count = len(uniform_objs_tform_obj)
    for t in range(tform_count):
        uniform_obj_tform_obj = uniform_objs_tform_obj[t]
        uniform_obj_tform_obj[:3, 3] = 0.
        uniform_obj_tform_obj[0, 3] = 1.
        uniform_obj_tform_obj[2, 3] = 1. * (t - tform_count // 2)

        mesh = Mesh(verts=transf3d_broadcast(pts3d=mesh2.verts.to(device=device, dtype=dtype), transf4x4=uniform_obj_tform_obj.to(device=device)),
                    faces=mesh2.faces, rgb=mesh2.rgb)
        meshes.append(mesh)

    viewpoints_count = 2
    show_scene(pts3d=[], pts3d_colors=[], device=device,
               meshes=meshes,
               meshes_add_translation=False, pts3d_add_translation=False,
               return_visualization=False, viewpoints_count=viewpoints_count)


    mesh1 = None
    mesh2 = None

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


    ## CALCULATING CATEGORICAL PCA
    # categorical_features = {}
    # categorical_pca_V = {}
    # for cat_id, category in enumerate(categories):
    #     logger.info(category)
    #     category_instance_ids = instance_ids[map_seq_to_cat == cat_id]
    #     categorical_features[category] = []
    #     for instance_id_in_category, instance_id in enumerate(category_instance_ids):
    #         instance_feats = sequences[instance_id].feats
    #
    #         if isinstance(instance_feats, List):
    #             categorical_features[category] += torch.cat([vert_feats for vert_feats in instance_feats], dim=0)
    #         else:
    #             categorical_features[category].append(instance_feats)
    #     categorical_features[category] = torch.stack(categorical_features[category], dim=0)
    #     _, _, categorical_pca_V[category] = torch.pca_lowrank(categorical_features[category])
    #     sequences[category_instance_ids[0]].categorical_pca_V = categorical_pca_V[category]

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