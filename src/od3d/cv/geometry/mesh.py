import logging
from pytorch3d.io import IO
import torch
from pathlib import Path
from pytorch3d.structures.meshes import Meshes as PT3DMeshes
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.renderer.mesh.utils import interpolate_face_attributes
from od3d.cv.geometry.transform import proj3d2d, proj3d2d_broadcast
from od3d.cv.visual.draw import draw_pixels
from od3d.cv.visual.show import show_img
from od3d.cv.io import load_ply
from enum import Enum
from typing import List
logger = logging.getLogger(__name__)

class MESH_RENDER_MODALITIES(str, Enum):
    DEPTH = 'depth'
    MASK = 'mask'
    RGB = 'rgb'
    RGBA = 'rgba'
    FEATS = 'feats'
    MASK_VERTS_VSBL = 'mask_verts_vsbl'
    VERTS_NCDS = 'verts_ncds'

class Mesh:
    def __init__(self, verts, faces, rgb=None, feats=None):
        self.verts = verts
        self.faces = faces
        self.rgb = rgb
        self.feats = feats

    @staticmethod
    def load_from_file(fpath: Path, device='cpu'):
        io = IO()
        mesh = io.load_mesh(fpath, device=device)
        verts = mesh[0].verts_list()[0]
        faces = mesh[0].faces_list()[0]
        return Mesh(verts=verts, faces=faces)

    @staticmethod
    def load_from_file_ply(fpath: Path):
        verts, faces = load_ply(fpath)
        return Mesh(verts=verts, faces=faces)

    def verts_count(self):
        return self.verts.shape[0]
class Meshes(torch.nn.Module):
    def __init__(self, verts: List[torch.Tensor], faces: List[torch.Tensor], rgb: List[torch.Tensor]= None, feats: List[torch.Tensor]=None):
        super().__init__()

        self.meshes_count = len(verts)
        self.verts = torch.nn.Parameter(torch.cat([_verts for _verts in verts], dim=0), requires_grad=False)
        self.faces = torch.nn.Parameter(torch.cat([_faces for _faces in faces], dim=0), requires_grad=False)

        self.verts_counts = [_verts.shape[0] for _verts in verts]
        self.faces_counts = [_faces.shape[0] for _faces in faces]
        self.verts_counts_acc_from_0 = [0] + [sum(self.verts_counts[:i+1]) for i in range(self.meshes_count)]
        self.faces_counts_acc_from_0 = [0] + [sum(self.faces_counts[:i+1]) for i in range(self.meshes_count)]
        self.verts_counts_max = max(self.verts_counts)
        self.faces_counts_max = max(self.faces_counts)

        self.mask_verts_not_padded = torch.ones(size=[len(self), self.verts_counts_max], dtype=torch.bool, device=self.verts.device)
        for i in range(len(self)):
            self.mask_verts_not_padded[i, self.verts_counts[i]:] = False

        if rgb is not None:
            self.rgb = torch.nn.Parameter(torch.cat([_rgb for _rgb in rgb], dim=0), requires_grad=False)
        else:
            self.rgb = None

        if feats is not None:
            self.feats = torch.nn.Parameter(torch.cat([_feats for _feats in feats], dim=0), requires_grad=True)
            self.feats_from_faces = torch.nn.Parameter(torch.cat([self.get_feats_with_mesh_id(mesh_id)[self.get_faces_with_mesh_id(mesh_id)] for mesh_id in range(len(self))], dim=0))
        else:
            self.feats = None
            self.feats_from_faces = None

        self.init_pt3d()

    def init_pt3d(self):
        self.pt3dmeshes = PT3DMeshes(
            verts=[self.get_verts_with_mesh_id(i) for i in range(self.meshes_count)],
            faces=[self.get_faces_with_mesh_id(i) for i in range(self.meshes_count)]
        )

    @staticmethod
    def load_from_files(fpaths_meshes: List[Path], device='cpu'):
        meshes = []
        for fpath_mesh in fpaths_meshes:
            meshes.append(Mesh.load_from_file(fpath=fpath_mesh, device=device))

        verts = [mesh.verts for mesh in meshes]
        faces = [mesh.faces for mesh in meshes]

        return Meshes(verts=verts, faces=faces)

    @staticmethod
    def get_faces_from_verts(verts, ball_radius=0.3):
        import open3d
        import numpy as np
        from od3d.cv.geometry.transform import transf3d_broadcast, transf4x4_from_spherical

        verts_rot = transf3d_broadcast(pts3d=verts, transf4x4=transf4x4_from_spherical(azim=torch.Tensor([0.05]), elev=torch.Tensor([0.05]), theta=torch.Tensor([0.05]), dist=torch.Tensor([1.])))
        verts_centered = verts_rot - verts_rot.mean(dim=-2, keepdim=True)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(verts_centered)
        pcd.normals = open3d.utility.Vector3dVector(verts_centered)
        pcd.estimate_normals()
        # mesh, densities = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=pcd)
        mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd=pcd, radii=open3d.utility.DoubleVector([ball_radius]))
        faces = torch.from_numpy(np.asarray(mesh.triangles))
        return faces

    def __len__(self):
        return self.meshes_count
    def _apply(self, fn):
        super()._apply(fn)
        self.init_pt3d()

    def get_verts_ncds_with_mesh_id(self, mesh_id):
        verts3d = self.get_verts_with_mesh_id(mesh_id)
        verts3d_ncds = (verts3d - verts3d.min(dim=0).values[None,]) / (
                verts3d.max(dim=0).values[None,] - verts3d.min(dim=0).values[None,])
        return verts3d_ncds
    def get_verts_ncds_from_faces_with_mesh_id(self, mesh_id):
        verts3d_ncds = self.get_verts_ncds_with_mesh_id(mesh_id)
        feats_from_faces = verts3d_ncds[self.get_faces_with_mesh_id(mesh_id)]
        return feats_from_faces

    def get_feats_from_faces_with_mesh_id(self, mesh_id):
        return self.feats_from_faces[self.faces_counts_acc_from_0[mesh_id]: self.faces_counts_acc_from_0[mesh_id+1]]
    def get_rgb_with_mesh_id(self, mesh_id):
        return self.rgb[self.verts_counts_acc_from_0[mesh_id]: self.verts_counts_acc_from_0[mesh_id+1]]
    def get_verts_with_mesh_id(self, mesh_id):
        return self.verts[self.verts_counts_acc_from_0[mesh_id]: self.verts_counts_acc_from_0[mesh_id+1]]

    def get_feats_with_mesh_id(self, mesh_id):
        return self.feats[self.verts_counts_acc_from_0[mesh_id]: self.verts_counts_acc_from_0[mesh_id+1]]

    def get_faces_with_mesh_id(self, mesh_id):
        return self.faces[self.faces_counts_acc_from_0[mesh_id]: self.faces_counts_acc_from_0[mesh_id+1]]
    def get_faces_padded_with_mesh_id(self, mesh_id):
        return self.get_tensor_faces_with_pad(tensor=self.get_faces_with_mesh_id(mesh_id), mesh_id=mesh_id)
    def get_tensor_verts_with_pad(self, tensor, mesh_id):
        pad = torch.Size([self.verts_counts_max - self.verts_counts[mesh_id]])
        return torch.cat([tensor, torch.zeros(size=pad + tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)], dim=0)

    def get_tensor_faces_with_pad(self, tensor, mesh_id):
        pad = torch.Size([self.faces_counts_max - self.faces_counts[mesh_id]])
        return torch.cat([tensor, torch.zeros(size=pad + tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)], dim=0)

    def get_feats_padded_with_mesh_id(self, mesh_id):
        return self.get_tensor_verts_with_pad(tensor=self.get_feats_with_mesh_id(mesh_id), mesh_id=mesh_id)
    def get_verts_padded_with_mesh_id(self, mesh_id):
        return self.get_tensor_verts_with_pad(tensor=self.get_verts_with_mesh_id(mesh_id), mesh_id=mesh_id)
    #def to(self, device):
    #    if self.device != device:
    #        self.verts = [v.to(device=device) for v in self.verts]
    #        self.faces = [f.to(device=device) for f in self.faces]
    #        if self.rgb is not None:
    #            self.rgb = [i.to(device=device) for i in self.rgb]
    #        if self.feats is not None:
    #            self.feats = [f.to(device=device) for f in self.feats]
    #            self.feats_from_faces = [self.feats[mesh_id][self.faces[mesh_id]] for mesh_id in range(len(self))]

    #           self.device = device
    def set_feats_cat_with_pad(self, feats):
        vts_ct_max = self.verts_counts_max
        device = self.verts.device
        self.feats = torch.nn.Parameter(torch.cat([feats[i*vts_ct_max: i*vts_ct_max + self.verts_counts[i]].to(device=device) for i in range(len(self))], dim=0), requires_grad=True)

        self.feats_from_faces = torch.nn.Parameter(torch.cat([self.get_feats_with_mesh_id(mesh_id)[self.get_faces_with_mesh_id(mesh_id)] for mesh_id in range(len(self))], dim=0))

    def set_feats_cat(self, feats):
        self.feats = torch.nn.Parameter(feats, requires_grad=True)
        self.feats_from_faces = torch.nn.Parameter(torch.cat([self.get_feats_with_mesh_id(mesh_id)[self.get_faces_with_mesh_id(mesh_id)] for mesh_id in range(len(self))], dim=0))

    def get_verts_stacked_with_mesh_ids(self, mesh_ids):
        if mesh_ids == None:
            mesh_ids = list(range(len(self)))
        return torch.stack([self.get_verts_padded_with_mesh_id(mesh_id) for mesh_id in mesh_ids], dim=0)

    def get_feats_stacked_with_mesh_ids(self, mesh_ids):
        if mesh_ids == None:
            mesh_ids = list(range(len(self)))
        return torch.stack([self.get_feats_with_mesh_id(mesh_id) for mesh_id in mesh_ids], dim=0)

    def get_faces_stacked_with_mesh_ids(self, mesh_ids):
        if mesh_ids == None:
            mesh_ids = list(range(len(self)))
        return torch.stack([self.get_faces_padded_with_mesh_id(mesh_id) for mesh_id in mesh_ids], dim=0)
    #def add_feats_cat(self, feats):
    #    raise Not I
    #    self.feats = [feats[self.verts_counts_acc_from_0[i] : self.verts_counts_acc_from_0[i+1]].to(device=self.device) for i in range(len(self))]
    #    self.feats_from_faces = [self.feats[mesh_id][self.faces[mesh_id]] for mesh_id in range(len(self))]

    # def add_feats
    #def verts_stacked(self, mesh_ids: list=None):
    #    if mesh_ids == None:
    #        mesh_ids = list(range(len(self)))

        #verts_stacked = torch.zeros(size=(len(mesh_ids), self.verts_counts_max(), 3), device=self.device)
        #for mesh_id in mesh_ids:
        #    verts_stacked[mesh_id, : len(self.verts[mesh_id])] = self.verts[mesh_id]

    #    return torch.stack([self.verts[mesh_id] for mesh_id in mesh_ids], dim=0)

    #def feats_cat(self, mesh_ids: list=None):
    #    if mesh_ids == None:
    #        mesh_ids = list(range(len(self)))
    #
    #    return torch.cat([self.feats[mesh_id] for mesh_id in mesh_ids], dim=0)

    #def get_feats_ids_stacked(self, mesh_ids: list=None):
    #    if mesh_ids == None:
    #        mesh_ids = list(range(len(self)))

    #    device = self.verts.device
    #    return torch.stack([torch.arange(mesh_id*self.verts_counts_max, (mesh_id+1)*self.verts_counts_max, device=device) for mesh_id in mesh_ids], dim=0)

    #def get_verts_and_noise_ids_cat(self, mesh_ids: list=None, count_noise_ids=5):
    #   if mesh_ids == None:
    #        mesh_ids = list(range(len(self)))

    #    device = self.verts.device
    #    noise_ids = torch.ones(size=(count_noise_ids,), dtype=torch.long, device=device) * self.verts_counts_acc_from_0[-1]
    #    verts_ids = [torch.arange(self.verts_counts_acc_from_0[mesh_id], self.verts_counts_acc_from_0[mesh_id+1], device=device) for mesh_id in mesh_ids]
    #    return torch.cat([torch.cat([verts_ids[i], noise_ids], dim=0) for i in range(len(mesh_ids))], dim=0)

    def get_verts_and_noise_ids_stacked(self, mesh_ids: list=None, count_noise_ids=5):
        if mesh_ids == None:
            mesh_ids = list(range(len(self)))

        device = self.verts.device
        noise_ids = torch.ones(size=(count_noise_ids,), dtype=torch.long, device=device) * self.verts_counts_acc_from_0[-1]
        verts_ids = [torch.arange(self.verts_counts_acc_from_0[mesh_id], self.verts_counts_acc_from_0[mesh_id] + self.verts_counts_max, device=device) for mesh_id in mesh_ids]
        return torch.stack([torch.cat([verts_ids[i], noise_ids], dim=0) for i in range(len(mesh_ids))], dim=0)


    def verts2d(self, cams_tform4x4_obj, cams_intr4x4, imgs_sizes, mesh_ids: torch.LongTensor, down_sample_rate=1., broadcast_batch_and_cams=False):
        """
            Args:
                cams_tform4x4_obj (torch.Tensor): Bx4x4
                cams_intr4x4 (torch.Tensor): Bx4x4
                imgs_sizes (torch.Tensor): Bx2 / 2
                mesh_ids (list): len(mesh_ids) == B

            Returns:
                verts2d (torch.Tensor): BxNx2

        """
        meshes_count = mesh_ids.shape[0]
        cams_count = cams_tform4x4_obj.shape[0]

        if broadcast_batch_and_cams:
            mesh_ids = mesh_ids
            cams_tform4x4_obj = cams_tform4x4_obj[None, :].expand(meshes_count, cams_count, 4, 4).reshape(-1, 4, 4)
            if cams_intr4x4.dim() == 3:
                cams_intr4x4 = cams_intr4x4[None, :]
            cams_intr4x4 = cams_intr4x4.expand(meshes_count, cams_count, 4, 4).reshape(-1, 4, 4)
            mesh_ids = mesh_ids[:, None].expand(meshes_count, cams_count).reshape(-1)

        B = cams_tform4x4_obj.shape[0]
        #if imgs_sizes.dim() == 2:
        #    imgs_sizes = imgs_sizes[None,].expand(B, imgs_sizes.shape[0], imgs_sizes.shape[1])
        cams_proj4x4_obj = torch.bmm(cams_intr4x4, cams_tform4x4_obj)
        verts3d = self.get_verts_stacked_with_mesh_ids(mesh_ids=mesh_ids)
        verts2d = proj3d2d_broadcast(verts3d, proj4x4=cams_proj4x4_obj[:, None])

        mask_verts_vsbl = self.render_feats(cams_tform4x4_obj=cams_tform4x4_obj, cams_intr4x4=cams_intr4x4, imgs_sizes=imgs_sizes, meshes_ids=mesh_ids, modality=MESH_RENDER_MODALITIES.MASK_VERTS_VSBL, down_sample_rate=down_sample_rate)
        mask_verts_vsbl *= (verts2d <= (imgs_sizes[None, None] - 1)).all(dim=-1)
        mask_verts_vsbl *= (verts2d >= 0).all(dim=-1)

        verts2d[~mask_verts_vsbl] = 0
        # verts2d.clamp()

        if broadcast_batch_and_cams:
            verts2d = verts2d.reshape(meshes_count, cams_count, *verts2d.shape[1:])
            mask_verts_vsbl = mask_verts_vsbl.reshape(meshes_count, cams_count, *mask_verts_vsbl.shape[1:])

        verts2d /= down_sample_rate

        return verts2d, mask_verts_vsbl

    def show(self, pts3d=[], meshes_ids=None):
        from pytorch3d.vis.plotly_vis import plot_scene, AxisArgs
        from pytorch3d.structures import Pointclouds
        if meshes_ids is None:
            meshes_ids = list(range(len(self.pt3dmeshes)))
        pcls = Pointclouds(points=pts3d)
        verts = Pointclouds(points=self.get_verts_stacked_with_mesh_ids(mesh_ids=meshes_ids))
        fig = plot_scene({
            "Meshes":
                {
                    # **{f"mesh{i + 1}": self.pt3dmeshes[i] for i in meshes_ids},
                    **{f"verts{i + 1}": verts[i] for i in meshes_ids},
                    **{f"pcl{i + 1}": pcls[i] for i in range(len(pcls))}
                }
        }, axis_args=AxisArgs(backgroundcolor="rgb(200, 200, 230)", showgrid=True, zeroline=True, showline=True,
                                   showaxeslabels=True, showticklabels=True))
        fig.show()
        input('bla')
    def render_feats(self, cams_tform4x4_obj, cams_intr4x4, imgs_sizes, meshes_ids=None, modality=MESH_RENDER_MODALITIES.FEATS, broadcast_batch_and_cams=False, down_sample_rate=1.):
        dtype = cams_tform4x4_obj.dtype
        device = cams_tform4x4_obj.device

        if down_sample_rate != 1.:
            cams_intr4x4 = cams_intr4x4 / down_sample_rate
            imgs_sizes = imgs_sizes // down_sample_rate

        if meshes_ids is None:
            meshes_ids = torch.LongTensor(list(range(len(self)))).to(device=device)

        meshes_count = meshes_ids.shape[0]
        cams_count = cams_tform4x4_obj.shape[0]

        if broadcast_batch_and_cams:
            meshes_ids = meshes_ids
            cams_tform4x4_obj = cams_tform4x4_obj[None, :].expand(meshes_count, cams_count, 4, 4).reshape(-1, 4, 4)
            if cams_intr4x4.dim() == 3:
                cams_intr4x4 = cams_intr4x4[None, :]
            cams_intr4x4 = cams_intr4x4.expand(meshes_count, cams_count, 4, 4).reshape(-1, 4, 4)
            meshes_ids = meshes_ids[:, None].expand(meshes_count, cams_count).reshape(-1)
            render_count = meshes_count * cams_count
        else:
            if meshes_count != cams_count:
                raise ValueError(f'Set `broadcast_batch_and_cams=True` to allow different number of cameras and meshes')
            render_count = meshes_count


        # self.to(device)

        #num_cams = cams_tform4x4_obj.shape[0]

        #cams_tform4x4_obj = cams_tform4x4_obj.repeat_interleave(num_meshes, dim=0)
        #cams_intr4x4 = cams_intr4x4.repeat_interleave(num_meshes, dim=0)
        #imgs_sizes = imgs_sizes.repeat_interleave(num_meshes, dim=0)
        t3d_tform_pscl3d = torch.Tensor([[-1., 0., 0., 0.],
                                         [0., -1., 0., 0.],
                                         [0., 0., 1., 0.],
                                         [0., 0., 0., 1.]]).to(device=device, dtype=dtype)
        # t3d_cam_tform_obj = torch.matmul(t3d_tform_pscl3d, cams_tform4x4_obj)
        t3d_cam_tform_obj = t3d_tform_pscl3d[None].expand(cams_tform4x4_obj.shape).bmm(cams_tform4x4_obj)
        R = t3d_cam_tform_obj[..., :3, :3].permute(0, 2, 1)
        t = t3d_cam_tform_obj[..., :3, 3]
        focal_length = torch.stack([cams_intr4x4[..., 0, 0], cams_intr4x4[..., 1, 1]], dim=-1)
        principal_point = torch.stack([cams_intr4x4[..., 0, 2], cams_intr4x4[..., 1, 2]], dim=-1)

        cameras = PerspectiveCameras(device=device, R=R, T=t, focal_length=focal_length,
                                     principal_point=principal_point, in_ndc=False,
                                     image_size=imgs_sizes[None, ].expand(render_count, 2))

          # K=self.K_4x4[None,]) #, K=K) # , K=K , znear=0.001, zfar=100000,
        #  znear=0.001, zfar=100000, fov=10
        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
        # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
        # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
        # the difference between naive and coarse-to-fine rasterization.

        raster_settings = RasterizationSettings(
            image_size=[int(imgs_sizes[0]), int(imgs_sizes[1])],
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=None,
        )

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )

        pt3dmeshes = self.pt3dmeshes[meshes_ids]
        fragments = rasterizer(pt3dmeshes)
        # pix_to_face: BxHxWx1, zbuf: BxHxWx1, bary_coords: BxHxWx1x3, dists: BxHxWx1
        if modality == MESH_RENDER_MODALITIES.MASK:
            mask = fragments.zbuf.permute(0, 3, 1, 2) > 0.
            if broadcast_batch_and_cams:
                mask = mask.reshape(meshes_count, cams_count, *mask.shape[-3:])
            return mask

        if modality == MESH_RENDER_MODALITIES.DEPTH:
            depth = fragments.zbuf.permute(0, 3, 1, 2)
            if broadcast_batch_and_cams:
                depth = depth.reshape(meshes_count, cams_count, *depth.shape[-3:])
            return depth

        if modality == MESH_RENDER_MODALITIES.MASK_VERTS_VSBL:
            B = fragments.pix_to_face.shape[0]
            faces_ids = torch.cat([self.get_faces_with_mesh_id(mesh_id) for mesh_id in meshes_ids], dim=0)
            # verts_ids_vsbl = torch.cat([self.get_faces_with_mesh_id(mesh_id) for mesh_id in meshes_ids], dim=0) [fragments.pix_to_face.reshape(B, -1)].reshape(B, -1)  # .unique(dim=1)
            verts_vsbl_mask = torch.zeros(size=(B, self.verts_counts_max), dtype=torch.bool, device=device)
            for b in range(B):
                # logger.info(f'meshes_ids {meshes_ids}')
                faces_ids_vsbl = fragments.pix_to_face[b]
                faces_ids_vsbl = faces_ids_vsbl.unique()
                faces_ids_vsbl = faces_ids_vsbl[faces_ids_vsbl >= 0]
                verts_ids_vsbl = faces_ids[faces_ids_vsbl].unique()
                verts_vsbl_mask[b, verts_ids_vsbl] = 1
            return verts_vsbl_mask

        if modality == MESH_RENDER_MODALITIES.FEATS:
            feats_from_faces = torch.cat([self.get_feats_from_faces_with_mesh_id(mesh_id) for mesh_id in meshes_ids], dim=0)
        else:
            feats_from_faces = torch.cat([self.get_verts_ncds_from_faces_with_mesh_id(mesh_id) for mesh_id in meshes_ids], dim=0)

        mesh_feats2d_rendered = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, feats_from_faces)[:, ..., 0,:].permute(0, 3, 1, 2)

        if broadcast_batch_and_cams:
            mesh_feats2d_rendered = mesh_feats2d_rendered.reshape(meshes_count, cams_count, *mesh_feats2d_rendered.shape[-3:])

        return mesh_feats2d_rendered
