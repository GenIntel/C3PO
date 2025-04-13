import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def filter(tensor):
    isnan = torch.isnan(tensor)
    row_with_all_nan = isnan.all(dim=1)
    valid_rows = ~row_with_all_nan
    filtered_tensor = tensor[valid_rows]
    original_indices = torch.arange(tensor.size(0))[valid_rows]  # Keep original indices
    return filtered_tensor, original_indices

class Neural_Mesh_Vertices_Cluster():
    def __init__(self, 
                 tensor_list):
        self.tensor_list = tensor_list
        self.n_cluster = 3
        self.random_state = 42
        
    def filter_vertices(self):
        shape_list = []
        for tensor in self.tensor_list:
            shape_list.append(tensor.shape[0])
        print('shape list ', shape_list)
        filtered_tensor_list = np.vstack(self.tensor_list)
        kmeans = KMeans(n_clusters=self.n_cluster, random_state= self.random_state)
        labels = kmeans.fit_predict(filtered_tensor_list)
        label_list = []
        start = 0
        for i in range(len(self.tensor_list)):
            label_list.append(labels[start: start+ shape_list[i]])
            start = shape_list[i]
            
        return label_list
    
if __name__ == '__main__':
    
    tensor1_original = torch.load('/CT/3D_DST_Scene/work/od3d/datasets/CO3D_V2_Preprocess/feats/M_dinov2_vitb14_frozen_base_T_centerzoom512_R_avg/alpha500/sfm_mask/colmap50/toybus/152_16878_31659/Partial_Ratio_25.0_Percent_start_frame_0_use_sph_sph_excludes_co3d_use_flipped_feature_False_use_sd_False_flip_sfm_False/mesh_feats.pt')
    tensor2_original = torch.load('/CT/3D_DST_Scene/work/od3d/datasets/CO3D_V2_Preprocess/feats/M_dinov2_vitb14_frozen_base_T_centerzoom512_R_avg/alpha500/sfm_mask/colmap50/toybus/199_21374_39988/Partial_Ratio_25.0_Percent_start_frame_0_use_sph_sph_excludes_co3d_use_flipped_feature_False_use_sd_False_flip_sfm_False/mesh_feats.pt')
    tensor3_original = torch.load('/CT/3D_DST_Scene/work/od3d/datasets/CO3D_V2_Preprocess/feats/M_dinov2_vitb14_frozen_base_T_centerzoom512_R_avg/alpha500/sfm_mask/colmap50/toybus/375_42441_85051/Partial_Ratio_25.0_Percent_start_frame_0_use_sph_sph_excludes_co3d_use_flipped_feature_False_use_sd_False_flip_sfm_False/mesh_feats.pt')
    tensor4_original = torch.load('/CT/3D_DST_Scene/work/od3d/datasets/CO3D_V2_Preprocess/feats/M_dinov2_vitb14_frozen_base_T_centerzoom512_R_avg/alpha500/sfm_mask/colmap50/toybus/396_49702_97897/Partial_Ratio_25.0_Percent_start_frame_0_use_sph_sph_excludes_co3d_use_flipped_feature_False_use_sd_False_flip_sfm_False/mesh_feats.pt')
    tensor5_original = torch.load('/CT/3D_DST_Scene/work/od3d/datasets/CO3D_V2_Preprocess/feats/M_dinov2_vitb14_frozen_base_T_centerzoom512_R_avg/alpha500/sfm_mask/colmap50/toybus/404_53714_104744/Partial_Ratio_25.0_Percent_start_frame_0_use_sph_sph_excludes_co3d_use_flipped_feature_False_use_sd_False_flip_sfm_False/mesh_feats.pt')
    tensor1, index1 = filter(tensor1_original)
    tensor2, index2 = filter(tensor2_original)
    tensor3, index3 = filter(tensor3_original)
    tensor4, index4 = filter(tensor4_original)
    tensor5, index5 = filter(tensor5_original)
    
    tensor_list = [tensor1.numpy(), tensor2.numpy(), tensor3.numpy(), tensor4.numpy(), tensor5.numpy()]
    shape_list = []
    for i, tensor in enumerate(tensor_list):
        print(f"Tensor {i+1} shape after filtering: {tensor.shape}")
        shape_list.append(tensor.shape[0])
    try:
        cluster = Neural_Mesh_Vertices_Cluster(tensor_list=tensor_list)
        label_list = cluster.filter_vertices()
        # Print labels for each tensor
        for i, labels in enumerate(label_list):
            print(f"Labels for tensor {i+1}: {labels[:10]}")  # Print first 10 labels for brevity
    except Exception as e:
        print(f"Error occurred: {e}")
        
    label1 = np.ones(tensor1_original.shape[0]) * 100
    label2 = np.ones(tensor2_original.shape[0]) * 100
    label3 = np.ones(tensor3_original.shape[0]) * 100
    label4 = np.ones(tensor4_original.shape[0]) * 100
    label5 = np.ones(tensor5_original.shape[0]) * 100
    
    label1[index1] = label_list[0]
    label2[index2] = label_list[1]
    label3[index3] = label_list[2]
    label4[index4] = label_list[3]
    label5[index5] = label_list[4]
    # label2[index2] = label_list[tensor1.shape[0] : tensor1.shape[0] + tensor2.shape[0]]
    
    
    
    