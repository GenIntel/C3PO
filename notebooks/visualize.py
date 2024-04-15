from typing import List

import numpy as np
import torch
from od3d.cv.cluster.embed import pca
from od3d.cv.cluster.embed import tsne
from od3d.cv.visual.show import show_scene2d
from od3d.datasets.co3dv1 import CO3Dv1
from od3d.datasets.co3dv1.dataset import CO3Dv1_Sequence


def get_sequence_data(sequences: List[CO3Dv1_Sequence], index: int):
    sequence = sequences[index]
    mesh = sequence.get_mesh()
    mesh_verts = mesh.verts_ncds.clone()
    cams_tform4x4_world, cams_intr4x4, cams_imgs = sequence.get_cams(cams_count=-1)
    # mesh.verts = transf3d_broadcast(mesh.verts),#sequence.zsp_labeled_cuboid_ref_tform_obj)
    sequence_dict = {
        "feats": sequence.get_mesh_feats(),
        "viewpoints": sequence.get_mesh_feats_viewpoint(),
        "cams_intr4x4": cams_intr4x4,
        "cams_tform4x4_world": cams_tform4x4_world,
        "pts3d": mesh_verts,
        "imgs": cams_imgs,
        "mesh": mesh,
    }
    return sequence_dict


Type = "tsne"
sequence_list = [
    0,
]  # M_dinov2_vitb14_frozen_base_no_norm_T_centerzoom512_R_acc ,M_nemo_old_T_centerzoom512_R_acc
co3d = CO3Dv1.create_by_name(
    "co3dv1_zsp_10s",
    config={
        "mesh_feats_type": "M_dinov2_vitb14_frozen_base_no_norm_T_centerzoom512_R_acc",
        "mesh_feats": {"enabled": True, "override": False},
    },
)  # , config={'preprocess' : {'tform_obj': {'enabled': True}},'mesh_feats' :{'enabled':False},'mesh_feats_dist' :{'enabled':False}}) #, config={'categories': [category], 'aligned_name': aligned_name, 'sequences_count_max_per_category': sequences_count}) # co3d_no_zsp_20s_aligned #co3d_5s_no_zsp_labeled 'co3d_50s_no_zsp_aligned' 'co3dv1_10s_zsp_aligned' 'co3d_10s_zsp_aligned' 'co3dv1_10s_zsp_unlabeled'
categories = co3d.categories
sequences = co3d.get_sequences()


seq_feats = []
seq_dict = []
feats_length = []
feats_length.append(0)
for i, idx in enumerate(sequence_list):
    seq_dict.append(get_sequence_data(sequences, idx))
    seq_feats.extend(seq_dict[i]["feats"])
    feats_length.append(len(seq_dict[i]["feats"]))
    # imgs = show_scene(meshes=[seq_dict[i]["mesh"]], cams_tform4x4_world=seq_dict[i]["cams_tform4x4_world"], cams_intr4x4=seq_dict[i]["cams_intr4x4"],
    #                     cams_imgs=seq_dict[i]["imgs"], return_visualization=True,viewpoints_count=2, pts3d=[torch.cat(seq_dict[i]["viewpoints"], dim=0)])
    # show_imgs(rgbs=seq_dict[i]["imgs"], fpath=f'image_{i}.png')

print("categories:", categories)

feats_length = np.cumsum(feats_length)


seq_feats_padded = torch.nn.utils.rnn.pad_sequence(
    seq_feats,
    batch_first=True,
    padding_value=torch.nan,
)
seq_feats_padded_mask = ~seq_feats_padded.isnan().all(dim=-1)
if Type == "tsne":
    seq_feats_embed = tsne(seq_feats_padded[seq_feats_padded_mask], C=2)
    seq_feats_embed = seq_feats_embed.detach().cpu()
    seq_feats_embed_3d = tsne(seq_feats_padded[seq_feats_padded_mask], C=3)
    seq_feats_embed_3d = seq_feats_embed_3d.detach().cpu()
elif Type == "pca":
    seq_feats_embed = pca(seq_feats_padded[seq_feats_padded_mask], C=2)
    seq_feats_embed = seq_feats_embed.detach().cpu()
    seq_feats_embed_3d = pca(seq_feats_padded[seq_feats_padded_mask], C=3)
    seq_feats_embed_3d = seq_feats_embed_3d.detach().cpu()


seq_colors_padded = torch.zeros(
    size=seq_feats_padded.shape[:-1] + (4,),
    dtype=torch.float32,
)
seq_colors_padded[:, :, 3] = 0.2

seq_colors_padded_verts = seq_colors_padded.clone()

seq_feats_viewpoints = []
seq_length = []
for i, idx in enumerate(sequence_list):
    seq_colors_padded_verts[feats_length[i] : feats_length[i + 1], :, :3] = seq_dict[i][
        "pts3d"
    ][:, None, :3]
    seq_feats_viewpoints.extend(seq_dict[i]["viewpoints"])
    seq_length.append(
        sum(
            [
                seq_dict[i]["feats"][j].shape[0]
                for j in range(len(seq_dict[i]["feats"]))
            ],
        ),
    )

print("seq_length:", seq_length)

seq_feats_viewpoints_padded = torch.nn.utils.rnn.pad_sequence(
    seq_feats_viewpoints,
    batch_first=True,
    padding_value=torch.nan,
)
seq_feats_viewpoints_padded = (
    seq_feats_viewpoints_padded
    - seq_feats_viewpoints_padded[seq_feats_padded_mask].min(dim=0).values[None, None]
) / (
    seq_feats_viewpoints_padded[seq_feats_padded_mask].max(dim=0).values[None, None]
    - seq_feats_viewpoints_padded[seq_feats_padded_mask].min(dim=0).values[None, None]
)

seq_colors_padded_viewpoint = seq_colors_padded.clone()
seq_colors_padded_viewpoint[:, :, :3] = seq_feats_viewpoints_padded[:, :, :3]

show_scene2d(
    pts2d=[seq_feats_embed[:], seq_feats_embed[:]],
    pts2d_colors=[
        seq_colors_padded_verts[seq_feats_padded_mask],
        seq_colors_padded_viewpoint[seq_feats_padded_mask],
    ],
    pts2d_names=["Vertex Position", "Viewpoints"],
    pts2d_lengths=seq_length,
)


# show_scene(pts3d=[seq_feats_embed_3d[:]],pts3d_colors=[seq_colors_padded_viewpoint[seq_feats_padded_mask][:,:3]])
