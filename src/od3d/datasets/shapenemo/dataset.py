import logging
from od3d.datasets.dataset import OD3D_Dataset
import scipy.io as sio
import numpy as np
from omegaconf import DictConfig
import od3d.io
from pathlib import Path
import scipy.io
import torchvision.io
from od3d.utils import load_off, save_off
import os


class ShapeNemo(OD3D_Dataset):
    def __init__(self, config, categories):
        super().__init__(config=config)
        self.setup(self.config)

        self.categories = categories
        self.path = Path(self.config.path_shapenemo)
        self.path_CAD = Path(self.config.path_meshes)
        self.create_meshes(CAD_path=self.path_CAD, save_path=self.path, number_vertices=1000, linear_coverage=0.99)

        self.map_cat2id = {c : i for i, c in enumerate(self.categories)}
        self.fpaths = [self.path.joinpath(c, '01.off') for c in self.categories]

    @staticmethod
    def setup(config: DictConfig):
        pass

    def __len__(self):
        return len(self.fpaths)
    def __getitem__(self, item):
        return self.fpaths[item]

    def get_cat(self, cat):
        return self[self.map_cat2id[cat]]
    def meshelize(self, x_range, y_range, z_range, number_vertices):
        w, h, d = x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]
        total_area = (w * h + h * d + w * d) * 2

        # On average, every vertice attarch 6 edges. Each triangle has 3 edges
        mesh_size = total_area / (number_vertices * 2)

        edge_length = (mesh_size * 2) ** .5

        x_samples = x_range[0] + np.linspace(0, w, int(w / edge_length + 1))
        y_samples = y_range[0] + np.linspace(0, h, int(h / edge_length + 1))
        z_samples = z_range[0] + np.linspace(0, d, int(d / edge_length + 1))

        xn = x_samples.size
        yn = y_samples.size
        zn = z_samples.size

        out_vertices = []
        out_faces = []
        base_idx = 0

        for n in range(yn):
            for m in range(xn):
                out_vertices.append((x_samples[m], y_samples[n], z_samples[0]))
        for m in range(yn - 1):
            for n in range(xn - 1):
                out_faces.append((base_idx + m * xn + n, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
                out_faces.append((base_idx + (m + 1) * xn + n + 1, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
        base_idx += yn * xn

        for n in range(yn):
            for m in range(xn):
                out_vertices.append((x_samples[m], y_samples[n], z_samples[-1]))
        for m in range(yn - 1):
            for n in range(xn - 1):
                out_faces.append((base_idx + m * xn + n, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
                out_faces.append((base_idx + (m + 1) * xn + n + 1, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
        base_idx += yn * xn

        for n in range(zn):
            for m in range(xn):
                out_vertices.append((x_samples[m], y_samples[0], z_samples[n]))
        for m in range(zn - 1):
            for n in range(xn - 1):
                out_faces.append((base_idx + m * xn + n, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
                out_faces.append((base_idx + (m + 1) * xn + n + 1, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
        base_idx += zn * xn

        for n in range(zn):
            for m in range(xn):
                out_vertices.append((x_samples[m], y_samples[-1], z_samples[n]))
        for m in range(zn - 1):
            for n in range(xn - 1):
                out_faces.append((base_idx + m * xn + n, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
                out_faces.append((base_idx + (m + 1) * xn + n + 1, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
        base_idx += zn * xn

        for n in range(zn):
            for m in range(yn):
                out_vertices.append((x_samples[0], y_samples[m], z_samples[n]))
        for m in range(zn - 1):
            for n in range(yn - 1):
                out_faces.append((base_idx + m * yn + n, base_idx + m * yn + n + 1, base_idx + (m + 1) * yn + n))
                out_faces.append((base_idx + (m + 1) * yn + n + 1, base_idx + m * yn + n + 1, base_idx + (m + 1) * yn + n))
        base_idx += zn * yn

        for n in range(zn):
            for m in range(yn):
                out_vertices.append((x_samples[-1], y_samples[m], z_samples[n]))
        for m in range(zn - 1):
            for n in range(yn - 1):
                out_faces.append((base_idx + m * yn + n, base_idx + m * yn + n + 1, base_idx + (m + 1) * yn + n))
                out_faces.append((base_idx + (m + 1) * yn + n + 1, base_idx + m * yn + n + 1, base_idx + (m + 1) * yn + n))
        base_idx += zn * yn

        return np.array(out_vertices), np.array(out_faces)


    def create_meshes(self, CAD_path, save_path, number_vertices, linear_coverage):
        for cate in self.categories:
            os.makedirs(os.path.join(save_path, cate), exist_ok=True)
            fnames = [x for x in os.listdir(os.path.join(CAD_path, cate)) if x.endswith('.off')]

            vertices = []
            for f in fnames:
                vertices_, faces_ = load_off(os.path.join(CAD_path, cate, f))
                vertices.append(vertices_)

            vertices = np.concatenate(vertices, axis=0)
            selected_shape = int(vertices.shape[0] * linear_coverage)
            out_pos = []

            for i in range(vertices.shape[1]):
                v_sorted = np.sort(vertices[:, i])
                v_group = v_sorted[selected_shape::] - v_sorted[0:-selected_shape]
                min_idx = np.argmin(v_group)
                # print(min_idx, min_idx + selected_shape)
                out_pos.append((v_sorted[min_idx], v_sorted[min_idx + selected_shape]))

            xvert, xface = self.meshelize(*out_pos, number_vertices=number_vertices)
            save_off(os.path.join(save_path, cate, '01.off'), xvert, xface)

