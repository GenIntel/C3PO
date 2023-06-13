from typing import List
import torch
from pytorch3d.structures import Pointclouds
from pytorch3d.structures import Meshes as PT3DMeshes
from od3d.cv.geometry.mesh import Meshes
import numpy as np
from od3d.cv.geometry.transform import transf3d_broadcast, transf4x4_from_rot3x3_and_transl3

_cuboid_corner_verts_limit_ids = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
]

_cuboid_rays = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7]
]

_cuboid_planes = [
    [0, 1, 2, 3],
    [3, 2, 6, 7],
    [0, 1, 5, 4],
    [0, 3, 7, 4],
    [1, 2, 6, 5],
    [4, 5, 6, 7],
]

_cuboid_triangles = [
    [0, 1, 2],
    [0, 3, 2],
    [4, 5, 6],
    [4, 6, 7],
    [1, 5, 6],
    [1, 6, 2],
    [0, 4, 7],
    [0, 7, 3],
    [3, 2, 6],
    [3, 6, 7],
    [0, 1, 5],
    [0, 4, 5],
]

class CoordinateFrame():

    def __init__(self, origin=torch.Tensor([0., 0., 0.,]), axes=torch.Tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), pts_count_axis=100):
        """
        Args:
            origin (torch.Tensor): 3
            axes (torch.Tensor): 3x3
        """

        # origin = world_transl_frame
        # axes = world_rot_frame

        linspaceXYZ = torch.linspace(start=0., end=1., steps=pts_count_axis)
        pts3d = torch.stack(torch.meshgrid(linspaceXYZ, linspaceXYZ, linspaceXYZ, indexing='xy'), dim=-1)
        # self.pts3d = pts3d[((pts3d == 0).sum(dim=-1)) >= 2]
        pts3d_axisX = pts3d[(pts3d[..., 1] == 0) * (pts3d[..., 2] == 0)]
        pts3d_axisY = pts3d[(pts3d[..., 0] == 0) * (pts3d[..., 2] == 0)]
        pts3d_axisZ = pts3d[(pts3d[..., 0] == 0) * (pts3d[..., 1] == 0)]
        self.pts3d_axis = torch.stack([pts3d_axisX, pts3d_axisY, pts3d_axisZ], dim=0)

        world_tform_frame = transf4x4_from_rot3x3_and_transl3(rot3x3=axes, transl3=origin)
        self.pts3d_axis = transf3d_broadcast(pts3d=self.pts3d_axis, transf4x4=world_tform_frame)

class Cuboids(Meshes):

    def __int__(self, verts: List[torch.Tensor], faces: List[torch.Tensor], rgb: List[torch.Tensor]= None, feats: List[torch.Tensor]=None):
        super().__init__(verts=verts, faces=faces, rgb=rgb, feats=feats)


    """
    def __init__(self, cuboids_limits: torch.Tensor, max_pts_count=1000):


        self.cuboid_limits = cuboids_limits
        self.cuboid_size = self.cuboid_limits[..., 1, :] - self.cuboid_limits[..., 0, :]
        self.cuboid_area = 2 * (self.cuboid_size[:, 0] * self.cuboid_size[:, 1] + self.cuboid_size[:, 1] * self.cuboid_size[:, 2] + self.cuboid_size[:, 0] * self.cuboid_size[:, 2])
        self.max_pts_count = max_pts_count
        self.step_size = (self.cuboid_area / self.max_pts_count) ** 0.5

        self.B = cuboids_limits.shape[0]
        self.corner_verts_ids = torch.LongTensor(_box_corner_verts_limit_ids)
        self.corner_verts = cuboids_limits[:, self.corner_verts_ids].diagonal(dim1=-2, dim2=-1)

        self.faces = torch.tensor(_box_triangles, dtype=torch.int64, device=cuboids_limits.device)[None,].expand(self.B, 12, 3)

        self.pts3d_edges = []
        self.pts3d_surface = []
        for b in range(self.B):
            linspaceX = torch.linspace(self.cuboid_limits[b, 0, 0], self.cuboid_limits[b, 1, 0], steps=(self.cuboid_size[b, 0] / self.step_size[b]).int())
            linspaceY = torch.linspace(self.cuboid_limits[b, 0, 1], self.cuboid_limits[b, 1, 1], steps=(self.cuboid_size[b, 1] / self.step_size[b]).int())
            linspaceZ = torch.linspace(self.cuboid_limits[b, 0, 2], self.cuboid_limits[b, 1, 2], steps=(self.cuboid_size[b, 2] / self.step_size[b]).int())
            _pts3d = torch.stack(torch.meshgrid(linspaceX, linspaceY, linspaceZ, indexing='xy'), dim=-1)
            _pts3d_surface = _pts3d[(_pts3d[:, :, :, None] == self.cuboid_limits[b][None, None, None]).any(dim=-2).any(dim=-1)]
            _pts3d_edges = _pts3d[((_pts3d[:, :, :, None] == self.cuboid_limits[b][None, None, None]).any(dim=-2).sum(dim=-1)) >= 2]
            #_pts3d_planes = _pts3d[((_pts3d[:, :, :, None] == self.cuboid_limits[b][None, None, None]).any(dim=-2).sum(dim=-1)) == 1]
            #_pts3d_corners = _pts3d[((_pts3d[:, :, :, None] == self.cuboid_limits[b][None, None, None]).any(dim=-2).sum(dim=-1)) == 3]

            self.pts3d_edges.append(_pts3d_edges)
            self.pts3d_surface.append(_pts3d_surface)

        #self.pt3dpcl = Pointclouds(points=self.pts3d)

        #self.pt3dmeshes = PT3DMeshes(
        #    verts=[v for v in self.corner_verts], faces=[f for f in self.faces]
        #)
        """

    @staticmethod
    def create_from_limits(limits):
        """
            Args:
                limits (torch.Tensor): Bx2x3
            Returns:
                verts (torch.Tensor): Bx8x3
                faces (torch.LongTensor): Bx12x3
        """
        B = limits.shape[0]

        # size = limits[..., 1, :] - limits[..., 0, :]
        # area = 2 * (size[:, 0] * size[:, 1] + size[:, 1] * size[:, 2] + size[:, 0] * size[:, 2])

        corner_verts_ids = torch.LongTensor(_cuboid_corner_verts_limit_ids)
        corner_verts = limits[:, corner_verts_ids].diagonal(dim1=-2, dim2=-1)

        faces = torch.tensor(_cuboid_triangles, dtype=torch.int64, device=limits.device)[None,].expand(B, 12, 3)

        return Cuboids(verts=corner_verts, faces=faces)

    @staticmethod
    def create_dense_from_limits(limits, verts_count=1000):
        """
            Args:
                limits (torch.Tensor): Bx2x3
            Returns:
                verts (list[torch.Tensor]): Bx[Vx3]
                faces (list[torch.LongTensor]): Bx[Fx3]

        """
        verts = []
        faces = []

        for sample_limits in limits:
            sample_limits = sample_limits.detach().cpu().numpy()
            x_range = sample_limits[:, 0]
            y_range = sample_limits[:, 1]
            z_range = sample_limits[:, 2]
            w, h, d = x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]
            total_area = (w * h + h * d + w * d) * 2

            # On average, every vertice attarch 6 edges. Each triangle has 3 edges
            mesh_size = total_area / (verts_count * 2)

            edge_length = (mesh_size * 2) ** 0.5

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
                    out_faces.append(
                        (
                            base_idx + m * xn + n,
                            base_idx + m * xn + n + 1,
                            base_idx + (m + 1) * xn + n,
                        ),
                    )
                    out_faces.append(
                        (
                            base_idx + (m + 1) * xn + n + 1,
                            base_idx + m * xn + n + 1,
                            base_idx + (m + 1) * xn + n,
                        ),
                    )
            base_idx += yn * xn

            for n in range(yn):
                for m in range(xn):
                    out_vertices.append((x_samples[m], y_samples[n], z_samples[-1]))
            for m in range(yn - 1):
                for n in range(xn - 1):
                    out_faces.append(
                        (
                            base_idx + m * xn + n,
                            base_idx + m * xn + n + 1,
                            base_idx + (m + 1) * xn + n,
                        ),
                    )
                    out_faces.append(
                        (
                            base_idx + (m + 1) * xn + n + 1,
                            base_idx + m * xn + n + 1,
                            base_idx + (m + 1) * xn + n,
                        ),
                    )
            base_idx += yn * xn

            for n in range(zn):
                for m in range(xn):
                    out_vertices.append((x_samples[m], y_samples[0], z_samples[n]))
            for m in range(zn - 1):
                for n in range(xn - 1):
                    out_faces.append(
                        (
                            base_idx + m * xn + n,
                            base_idx + m * xn + n + 1,
                            base_idx + (m + 1) * xn + n,
                        ),
                    )
                    out_faces.append(
                        (
                            base_idx + (m + 1) * xn + n + 1,
                            base_idx + m * xn + n + 1,
                            base_idx + (m + 1) * xn + n,
                        ),
                    )
            base_idx += zn * xn

            for n in range(zn):
                for m in range(xn):
                    out_vertices.append((x_samples[m], y_samples[-1], z_samples[n]))
            for m in range(zn - 1):
                for n in range(xn - 1):
                    out_faces.append(
                        (
                            base_idx + m * xn + n,
                            base_idx + m * xn + n + 1,
                            base_idx + (m + 1) * xn + n,
                        ),
                    )
                    out_faces.append(
                        (
                            base_idx + (m + 1) * xn + n + 1,
                            base_idx + m * xn + n + 1,
                            base_idx + (m + 1) * xn + n,
                        ),
                    )
            base_idx += zn * xn

            for n in range(zn):
                for m in range(yn):
                    out_vertices.append((x_samples[0], y_samples[m], z_samples[n]))
            for m in range(zn - 1):
                for n in range(yn - 1):
                    out_faces.append(
                        (
                            base_idx + m * yn + n,
                            base_idx + m * yn + n + 1,
                            base_idx + (m + 1) * yn + n,
                        ),
                    )
                    out_faces.append(
                        (
                            base_idx + (m + 1) * yn + n + 1,
                            base_idx + m * yn + n + 1,
                            base_idx + (m + 1) * yn + n,
                        ),
                    )
            base_idx += zn * yn

            for n in range(zn):
                for m in range(yn):
                    out_vertices.append((x_samples[-1], y_samples[m], z_samples[n]))
            for m in range(zn - 1):
                for n in range(yn - 1):
                    out_faces.append(
                        (
                            base_idx + m * yn + n,
                            base_idx + m * yn + n + 1,
                            base_idx + (m + 1) * yn + n,
                        ),
                    )
                    out_faces.append(
                        (
                            base_idx + (m + 1) * yn + n + 1,
                            base_idx + m * yn + n + 1,
                            base_idx + (m + 1) * yn + n,
                        ),
                    )
            base_idx += zn * yn

            out_vertices = np.array(out_vertices)
            out_faces = np.array(out_faces)

            verts.append(torch.from_numpy(out_vertices))
            faces.append(torch.from_numpy(out_faces))

        return Cuboids(verts=verts, faces=faces)
