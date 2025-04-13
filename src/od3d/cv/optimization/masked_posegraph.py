import json
import math

import gtsam
import numpy as np
import torch
from od3d.cv.metric.pose import get_pose_diff_in_rad


def vector6(sigma):
    """Utility to create a vector6 for variances."""
    return np.array([sigma, sigma, sigma, sigma, sigma, sigma])


def random_pose3():
    """Generates a random Pose3 for initial transformation matrices."""
    return gtsam.Pose3(
        gtsam.Rot3.Rodrigues(
            -0.1 + 0.2 * np.random.rand(),
            -0.1 + 0.2 * np.random.rand(),
            -0.1 + 0.2 * np.random.rand(),
        ),
        gtsam.Point3(
            -0.5 + np.random.rand(),
            -0.5 + np.random.rand(),
            -0.5 + np.random.rand(),
        ),
    )


def extract_rotation_from_sim3(sim3_matrix):
    # Step 1: Extract the top-left 3x3 part, this is sR
    sR = sim3_matrix[:3, :3]

    # Step 2: Calculate the scale s using the Frobenius norm
    # Frobenius norm of the scaled rotation matrix sR is s * sqrt(3)
    s = torch.norm(sR) / torch.sqrt(torch.tensor(3.0))

    # Step 3: Divide by s to get the rotation matrix R
    R = sR / s

    return R, s


class WeightedPoseGraphOptimizationRotationMatrixMasking:
    def __init__(self, rotation_matrices, weighted_matrices, seq_list=None) -> None:
        self.n = rotation_matrices.size()[0]
        self.rotation_matrices = rotation_matrices
        self.weight_matrices = weighted_matrices
        self.graph = gtsam.NonlinearFactorGraph()
        self.masked_graph = gtsam.NonlinearFactorGraph()
        self.symbols = [gtsam.symbol("x", i) for i in range(1, self.n + 1)]
        self.gt_ref_tform_src = torch.eye(
            4
        )  # Keeping 4x4 for comparison with a generic identity matrix
        self.seq_list = seq_list

    def convert_rotation_to_gtsam_pose3(self, rotation_matrix):
        rotation_matrix_numpy = rotation_matrix.cpu().numpy()
        rotation = gtsam.Rot3(rotation_matrix_numpy[:3, :3])
        translation = gtsam.Point3(0, 0, 0)  # Assuming no translation
        pose = gtsam.Pose3(rotation, translation)
        return pose

    def random_rotation_se3(self):
        random_rotation = gtsam.Rot3.Rodrigues(
            np.random.randn(), np.random.randn(), np.random.randn()
        )
        # Generate a random translation
        zero_translation = gtsam.Point3(0, 0, 0)
        return gtsam.Pose3(random_rotation, zero_translation)

    def pose3_to_tensor(self, pose):
        # Extract rotation matrix and create a homogeneous transformation matrix
        R = pose.rotation().matrix()
        T = np.eye(4)
        T[:3, :3] = R
        return torch.tensor(T, dtype=torch.float32)

    def save_edges_to_file(self, filename):
        with open(filename, "w") as f:
            json.dump(self.edges, f, indent=4)
        print(f"Edges saved to {filename}")

    def optimization(self):
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    Rij = self.convert_rotation_to_gtsam_pose3(
                        self.rotation_matrices[i][j]
                    )
                    weight = self.weight_matrices[i][j].cpu().numpy()
                    self.graph.add(
                        gtsam.BetweenFactorPose3(
                            self.symbols[i],
                            self.symbols[j],
                            Rij,
                            gtsam.noiseModel.Diagonal.Variances(
                                np.ones(6) * (1 / weight)
                            ),
                        ),
                    )
        initial_estimate = gtsam.Values()
        initializer = PGOInitializer(
            self.rotation_matrices, self.weight_matrices, self.seq_list
        )
        pose_node = initializer._iterate()
        self.pose_node = pose_node
        self.edges = initializer.get_edges()
        # print('After initialization, pose node looks like ', pose_node )
        node_order = initializer.get_node_oder()
        self.node_order = node_order
        self.weights_pgo_init = initializer.get_weights_list()
        # threshold = sum(self.weights_pgo_init)/ len(self.weights_pgo_init)
        # threshold = 0
        threshold = min(self.weights_pgo_init)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    Rij = self.convert_rotation_to_gtsam_pose3(
                        self.rotation_matrices[i][j]
                    )
                    weight = self.weight_matrices[i][j].cpu().numpy()
                    if weight < threshold:
                        weight = weight * 1e-5
                        # weight = 1e-3
                        self.masked_graph.add(
                            gtsam.BetweenFactorPose3(
                                self.symbols[i],
                                self.symbols[j],
                                Rij,
                                gtsam.noiseModel.Diagonal.Variances(
                                    np.ones(6) * (1 / weight)
                                ),
                            ),
                        )
                    else:
                        # weight = self.weight_matrices[i][j]
                        self.masked_graph.add(
                            gtsam.BetweenFactorPose3(
                                self.symbols[i],
                                self.symbols[j],
                                Rij,
                                gtsam.noiseModel.Diagonal.Variances(
                                    np.ones(6) * (1 / weight)
                                ),
                            ),
                        )
        # for i in range(self.n):
        #     initial_estimate.insert(self.symbols[i], gtsam.Pose3())
        for i in range(self.n):
            # pose = self.random_rotation_se3()
            trafo = self.convert_rotation_to_gtsam_pose3(pose_node[str(i)])
            initial_estimate.insert(self.symbols[i], trafo)

        print("Initial error:", self.masked_graph.error(initial_estimate))
        # optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, initial_estimate)
        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self.masked_graph, initial_estimate
        )
        result = optimizer.optimize()
        self.result = result
        print("Final error:", self.masked_graph.error(result))
        updated_rad = torch.zeros(self.n, self.n)
        # for i in range(self.n):
        #     pose = result.atPose3(self.symbols[i])
        #     pose = self.pose3_to_tensor(pose)
        #     print(f'pose {i}', pose)

        for i in range(self.n):
            for j in range(self.n):
                pose_i = result.atPose3(self.symbols[i])
                pose_j = result.atPose3(self.symbols[j])
                pose_i_j = pose_i.between(pose_j)
                # print('pose_i_j shape ', pose_i_j.shape)
                diff_rot_angle_rad = get_pose_diff_in_rad(
                    pred_tform4x4=self.pose3_to_tensor(pose_i_j),
                    gt_tform4x4=self.gt_ref_tform_src,
                )
                updated_rad[i][j] = diff_rot_angle_rad

        # print('updated rad ', updated_rad)
        return updated_rad.cuda()

    def return_node_order(self):
        return self.node_order

    def return_pose_graph_init(self):
        return self.edges

    def return_pgo_init_weights(self):
        return self.weights_pgo_init

    def return_pose_node_init(self):
        return self.pose_node

    def return_pose_node_optimized(self):
        pose_node_optimized = {}
        for i in range(self.n):
            pose_i = self.result.atPose3(self.symbols[i])
            # Extracting the rotation matrix from the Pose3 object
            rot_matrix_np = pose_i.rotation().matrix()
            trafo = np.eye(4)
            trafo[:3, :3] = rot_matrix_np
            # Converting the Numpy array to a PyTorch tensor, assuming the tensor should be on the same device as your input tensors
            rot_matrix_tensor = torch.tensor(trafo, dtype=torch.float32)
            if torch.cuda.is_available():
                rot_matrix_tensor = (
                    rot_matrix_tensor.cuda()
                )  # Ensuring the tensor is on GPU if available
            # Storing the tensor in the dictionary with the key as a string of the index
            pose_node_optimized[f"{i}"] = rot_matrix_tensor
        return pose_node_optimized

    # def return_pose_node_optimized(self):
    #     pose_node_optimized = {}
    #     for i in range(self.n):
    #         pose_i = self.result.atPose3(self.symbols[i])
    #         pose_node_optimized[f'{i}'] = pose_i.cuda()[:3,:3]

    #     return pose_node_optimized

    def return_pose_graph_init_transformations(self):
        init_rad = torch.zeros(self.n, self.n).cuda()
        for i in range(self.n):
            for j in range(self.n):
                pose_i = self.pose_node[f"{i}"].cuda()
                pose_j = self.pose_node[f"{j}"].cuda()
                transformation_i_j = pose_i @ torch.linalg.inv(pose_j)
                init_rad[i][j] = get_pose_diff_in_rad(
                    pred_tform4x4=transformation_i_j.cuda(),
                    gt_tform4x4=self.gt_ref_tform_src.cuda(),
                )

        return init_rad


class WeightedPoseGraphOptimizationRotation:
    def __init__(self, rotation_matrices, weighted_matrices, seq_list=None) -> None:
        self.n = rotation_matrices.size()[0]
        self.rotation_matrices = rotation_matrices
        self.weight_matrices = weighted_matrices
        self.graph = gtsam.NonlinearFactorGraph()
        self.masked_graph = gtsam.NonlinearFactorGraph()
        self.symbols = [gtsam.symbol("x", i) for i in range(1, self.n + 1)]
        self.gt_ref_tform_src = torch.eye(
            4
        )  # Keeping 4x4 for comparison with a generic identity matrix
        self.seq_list = seq_list

    def convert_rotation_to_gtsam_pose3(self, rotation_matrix):
        rotation_matrix_numpy = rotation_matrix.cpu().numpy()
        rotation = gtsam.Rot3(rotation_matrix_numpy[:3, :3])
        translation = gtsam.Point3(0, 0, 0)  # Assuming no translation
        pose = gtsam.Pose3(rotation, translation)
        return pose

    def random_rotation_se3(self):
        random_rotation = gtsam.Rot3.Rodrigues(
            np.random.randn(), np.random.randn(), np.random.randn()
        )
        # Generate a random translation
        zero_translation = gtsam.Point3(0, 0, 0)
        return gtsam.Pose3(random_rotation, zero_translation)

    def pose3_to_tensor(self, pose):
        # Extract rotation matrix and create a homogeneous transformation matrix
        R = pose.rotation().matrix()
        T = np.eye(4)
        T[:3, :3] = R
        return torch.tensor(T, dtype=torch.float32)

    def save_edges_to_file(self, filename):
        with open(filename, "w") as f:
            json.dump(self.edges, f, indent=4)
        print(f"Edges saved to {filename}")

    def optimization(self):
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    Rij = self.convert_rotation_to_gtsam_pose3(
                        self.rotation_matrices[i][j]
                    )
                    weight = self.weight_matrices[i][j].cpu().numpy()
                    self.graph.add(
                        gtsam.BetweenFactorPose3(
                            self.symbols[i],
                            self.symbols[j],
                            Rij,
                            gtsam.noiseModel.Diagonal.Variances(
                                np.ones(6) * (1 / weight)
                            ),
                        ),
                    )
        initial_estimate = gtsam.Values()
        initializer = PGOInitializer(
            self.rotation_matrices, self.weight_matrices, self.seq_list
        )
        pose_node = initializer._iterate()
        self.pose_node = pose_node
        self.edges = initializer.get_edges()
        # print('After initialization, pose node looks like ', pose_node )
        node_order = initializer.get_node_oder()
        self.node_order = node_order
        self.weights_pgo_init = initializer.get_weights_list()
        self.edge_nodes = initializer.get_edge_nodes()
        threshold = min(self.weights_pgo_init)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    Rij = self.convert_rotation_to_gtsam_pose3(
                        self.rotation_matrices[i][j]
                    )
                    if [i, j] in self.edge_nodes:
                        weight = 1.0
                        self.masked_graph.add(
                            gtsam.BetweenFactorPose3(
                                self.symbols[i],
                                self.symbols[j],
                                Rij,
                                gtsam.noiseModel.Diagonal.Variances(
                                    np.ones(6) * (1 / weight)
                                ),
                            ),
                        )
                    # weight = self.weight_matrices[i][j].cpu().numpy()
                    # if weight < threshold:
                    #     #weight = weight * 1e-5
                    #     weight = 1e-2
                    #     self.masked_graph.add(gtsam.BetweenFactorPose3(self.symbols[i], self.symbols[j],
                    #                                    Rij, gtsam.noiseModel.Diagonal.Variances(np.ones(6) * (1/weight))))
                    # else:
                    #     weight = 1.0
                    #     self.masked_graph.add(gtsam.BetweenFactorPose3(self.symbols[i], self.symbols[j],
                    #                                         Rij, gtsam.noiseModel.Diagonal.Variances(np.ones(6) * (1/weight))))
        # for i in range(self.n):
        #     initial_estimate.insert(self.symbols[i], gtsam.Pose3())
        for i in range(self.n):
            # pose = self.random_rotation_se3()
            trafo = self.convert_rotation_to_gtsam_pose3(pose_node[str(i)])
            initial_estimate.insert(self.symbols[i], trafo)

        print("Initial error:", self.masked_graph.error(initial_estimate))
        # optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, initial_estimate)
        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self.masked_graph, initial_estimate
        )
        result = optimizer.optimize()
        self.result = result
        print("Final error:", self.masked_graph.error(result))
        updated_rad = torch.zeros(self.n, self.n)
        # for i in range(self.n):
        #     pose = result.atPose3(self.symbols[i])
        #     pose = self.pose3_to_tensor(pose)
        #     print(f'pose {i}', pose)

        for i in range(self.n):
            for j in range(self.n):
                pose_i = result.atPose3(self.symbols[i])
                pose_j = result.atPose3(self.symbols[j])
                pose_i_j = pose_i.between(pose_j)
                # print('pose_i_j shape ', pose_i_j.shape)
                diff_rot_angle_rad = get_pose_diff_in_rad(
                    pred_tform4x4=self.pose3_to_tensor(pose_i_j),
                    gt_tform4x4=self.gt_ref_tform_src,
                )
                updated_rad[i][j] = diff_rot_angle_rad

        # print('updated rad ', updated_rad)
        return updated_rad.cuda()

    def return_node_order(self):
        return self.node_order

    def return_pose_graph_init(self):
        return self.edges

    def return_pgo_init_weights(self):
        return self.weights_pgo_init

    def return_pose_node_init(self):
        return self.pose_node

    def return_pose_node_optimized(self):
        pose_node_optimized = {}
        for i in range(self.n):
            pose_i = self.result.atPose3(self.symbols[i])
            # Extracting the rotation matrix from the Pose3 object
            rot_matrix_np = pose_i.rotation().matrix()
            trafo = np.eye(4)
            trafo[:3, :3] = rot_matrix_np
            # Converting the Numpy array to a PyTorch tensor, assuming the tensor should be on the same device as your input tensors
            rot_matrix_tensor = torch.tensor(trafo, dtype=torch.float32)
            if torch.cuda.is_available():
                rot_matrix_tensor = (
                    rot_matrix_tensor.cuda()
                )  # Ensuring the tensor is on GPU if available
            # Storing the tensor in the dictionary with the key as a string of the index
            pose_node_optimized[f"{i}"] = rot_matrix_tensor
        return pose_node_optimized

    # def return_pose_node_optimized(self):
    #     pose_node_optimized = {}
    #     for i in range(self.n):
    #         pose_i = self.result.atPose3(self.symbols[i])
    #         pose_node_optimized[f'{i}'] = pose_i.cuda()[:3,:3]

    #     return pose_node_optimized

    def return_pose_graph_init_transformations(self):
        init_rad = torch.zeros(self.n, self.n).cuda()
        for i in range(self.n):
            for j in range(self.n):
                pose_i = self.pose_node[f"{i}"].cuda()
                pose_j = self.pose_node[f"{j}"].cuda()
                transformation_i_j = pose_i @ torch.linalg.inv(pose_j)
                init_rad[i][j] = get_pose_diff_in_rad(
                    pred_tform4x4=transformation_i_j.cuda(),
                    gt_tform4x4=self.gt_ref_tform_src.cuda(),
                )

        return init_rad


class PGOInitializer:
    def __init__(self, transformation_matrices, weight_matrix, seq_list):
        self.transformation_matrices = transformation_matrices
        self.weight_matrix = weight_matrix
        self.selected = []
        self.remaining = list(
            range(transformation_matrices.shape[0])
        )  # Assuming transformation_matrices is a numpy array
        self.pose_node = {}
        self.idx = 0
        self.node_order = []
        self.edges = []
        self.edge_nodes = []
        self.weights = []
        self.seq_list = seq_list

    def start(self):
        max_value = float("-inf")
        max_pos = None
        for i in range(self.transformation_matrices.shape[0]):
            for j in range(self.transformation_matrices.shape[1]):
                if i != j and self.weight_matrix[i, j] > max_value:
                    max_value = self.weight_matrix[i, j]
                    max_pos = (i, j)

        if max_pos is not None:
            self.selected.extend(max_pos)
            self.remaining.remove(max_pos[0])
            self.remaining.remove(max_pos[1])
            self.pose_node[str(max_pos[0])] = torch.eye(4)
            self.pose_node[str(max_pos[1])] = torch.linalg.inv(
                self.transformation_matrices[max_pos[0], max_pos[1]]
            )
            self.node_order.append(str(max_pos[0]))
            self.node_order.append(str(max_pos[1]))
            self.edges.append(
                {
                    "source": self.seq_list[(max_pos[0])],
                    "target": self.seq_list[(max_pos[1])],
                    "weight": float(self.weight_matrix[max_pos[0], max_pos[1]]),
                    "transformation": self.transformation_matrices[
                        max_pos[0], max_pos[1]
                    ].tolist(),
                },
            )
            self.weights.append(float(self.weight_matrix[max_pos[0], max_pos[1]]))
            self.edge_nodes.append([max_pos[0], max_pos[1]])

    def _iterate(self):
        self.start()
        while self.remaining:
            max_value = float("-inf")
            added_new_node = None
            used_node = None
            for i in self.selected:
                for k in self.remaining:
                    if self.weight_matrix[i, k] > max_value:
                        max_value = self.weight_matrix[i, k]
                        added_new_node = k
                        used_node = i
            self.node_order.append(str(added_new_node))

            if added_new_node is not None:
                # inverse_transform = torch.linalg.inv(self.pose_node[str(used_node)]).cuda()
                # self.pose_node[str(added_new_node)] = inverse_transform @ self.transformation_matrices[used_node, added_new_node]
                inverse_transform = torch.linalg.inv(
                    self.transformation_matrices[used_node, added_new_node]
                )
                self.pose_node[str(added_new_node)] = (
                    inverse_transform @ self.pose_node[str(used_node)].cuda()
                )
                self.selected.append(added_new_node)
                self.remaining.remove(added_new_node)
                self.edges.append(
                    {
                        "source": self.seq_list[(used_node)],
                        "target": self.seq_list[(added_new_node)],
                        "weight": float(self.weight_matrix[used_node, added_new_node]),
                        "transformation": self.transformation_matrices[
                            used_node, added_new_node
                        ].tolist(),
                    },
                )
                self.edge_nodes.append([used_node, added_new_node])
                self.weights.append(
                    float(self.weight_matrix[used_node, added_new_node])
                )
        return self.pose_node

    def get_node_oder(self):
        return self.node_order

    def get_edges(self):
        return self.edges

    def get_edge_nodes(self):
        return self.edge_nodes

    def get_weights_list(self):
        return self.weights


if __name__ == "__main__":
    initializer = PGOInitializer(transformation_matrices, weight_matrix)
    initializer._iterate()
