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

logger = logging.getLogger(__name__)

class Mesh:
    def __init__(self, fpath_mesh, device='cpu', rgb=None, feats=None):
        io = IO()
        mesh = io.load_mesh(fpath_mesh, device=device)
        self.verts = mesh[0].verts_list()[0]
        self.faces = mesh[0].faces_list()[0]
        self.rgb = rgb
        self.feats = feats
    def verts_count(self):
        return self.verts.shape[0]
class Meshes(torch.nn.Module):
    def __init__(self, meshes: list[Mesh] = None, fpaths_meshes: list[Path] = None):
        super().__init__()
        if meshes is None:
            meshes = []
            for fpath_mesh in fpaths_meshes:
                meshes.append(Mesh(fpath_mesh=fpath_mesh, device='cpu'))
        mesh0 = meshes[0]
        self.meshes_count = len(meshes)
        self.verts = torch.nn.Parameter(torch.cat([mesh.verts for mesh in meshes], dim=0), requires_grad=False)
        self.faces = torch.nn.Parameter(torch.cat([mesh.faces for mesh in meshes], dim=0), requires_grad=False)
        self.verts_counts = [mesh.verts.shape[0] for mesh in meshes]
        self.faces_counts = [mesh.faces.shape[0] for mesh in meshes]
        self.verts_counts_acc_from_0 = [0] + [sum(self.verts_counts[:i+1]) for i in range(self.meshes_count)]
        self.faces_counts_acc_from_0 = [0] + [sum(self.faces_counts[:i+1]) for i in range(self.meshes_count)]
        self.verts_counts_max = max(self.verts_counts)
        self.faces_counts_max = max(self.faces_counts)

        self.mask_verts_not_padded = torch.ones(size=[len(self), self.verts_counts_max], dtype=torch.bool, device=self.verts.device)
        for i in range(len(self)):
            self.mask_verts_not_padded[i, self.verts_counts[i]:] = False

        self.initialize_pytorch3d_meshes()

        if mesh0.rgb is not None:
            self.rgb = torch.nn.Parameter(torch.cat([mesh.rgb for mesh in meshes], dim=0), requires_grad=False)
        else:
            self.rgb = None

        if mesh0.feats is not None:
            self.feats = torch.nn.Parameter(torch.cat([mesh.feats for mesh in meshes], dim=0), requires_grad=True)
            self.feats_from_faces = torch.nn.Parameter(torch.cat([self.get_feats_with_mesh_id(mesh_id)[self.get_faces_with_mesh_id(mesh_id)] for mesh_id in range(len(self))], dim=0))

        else:
            self.feats = None
            self.feats_from_faces = None

    def __len__(self):
        return self.meshes_count

    def initialize_pytorch3d_meshes(self):
        self.pt3dmeshes = PT3DMeshes(
            verts=[self.get_verts_with_mesh_id(i) for i in range(self.meshes_count)],
            faces=[self.get_faces_with_mesh_id(i) for i in range(self.meshes_count)]
        )
    def _apply(self, fn):
        super()._apply(fn)
        self.initialize_pytorch3d_meshes()
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


    def get_verts_stacked_with_mesh_ids(self, mesh_ids):
        if mesh_ids == None:
            mesh_ids = list(range(len(self)))
        return torch.stack([self.get_verts_padded_with_mesh_id(mesh_id) for mesh_id in mesh_ids], dim=0)

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


    def verts2d(self, cams_tform4x4_obj, cams_intr4x4, imgs_sizes, mesh_ids: list=None, down_sample_rate=8.):
        """
            Args:
                cams_tform4x4_obj (torch.Tensor): Bx4x4
                cams_intr4x4 (torch.Tensor): Bx4x4
                imgs_sizes (torch.Tensor): Bx2
                mesh_ids (list): len(mesh_ids) == B

            Returns:
                verts2d (torch.Tensor): BxNx2

        """
        cams_proj4x4_obj = torch.bmm(cams_intr4x4, cams_tform4x4_obj)
        verts3d = self.get_verts_stacked_with_mesh_ids(mesh_ids=mesh_ids)
        mask_verts_vsbl = self.render_feats(cams_tform4x4_obj=cams_tform4x4_obj, cams_intr4x4=cams_intr4x4 / down_sample_rate, imgs_sizes=imgs_sizes // down_sample_rate, meshes_ids=mesh_ids, replace_feats_with_mask_vsbl=True)
        verts2d = proj3d2d_broadcast(verts3d, proj4x4=cams_proj4x4_obj[:, None])

        mask_verts_vsbl *= (verts2d <= (imgs_sizes[:, None] - 1)).all(dim=-1)
        mask_verts_vsbl *= (verts2d >= 0).all(dim=-1)

        verts2d[~mask_verts_vsbl] = 0
        # verts2d.clamp()
        return verts2d, mask_verts_vsbl
    def render_feats(self, cams_tform4x4_obj, cams_intr4x4, imgs_sizes, meshes_ids=None, replace_feats_with_verts3d=False, replace_feats_with_mask_vsbl=False):

        if meshes_ids is None:
            meshes_ids = list(range(len(self)))
        dtype = cams_tform4x4_obj.dtype
        device = cams_tform4x4_obj.device

        # self.to(device)

        num_cams = cams_tform4x4_obj.shape[0]

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
                                     image_size=imgs_sizes)

          # K=self.K_4x4[None,]) #, K=K) # , K=K , znear=0.001, zfar=100000,
        #  znear=0.001, zfar=100000, fov=10
        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
        # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
        # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
        # the difference between naive and coarse-to-fine rasterization.

        raster_settings = RasterizationSettings(
            image_size=[int(imgs_sizes[0, 0]), int(imgs_sizes[0, 1])],
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=None,
        )

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )

        # this is not parallizable currently, because different meshes have different number of vertices
        meshes_feats2d_rendered = []
        for mesh_id in meshes_ids:
            mesh_feats2d_rendered = []
            # for cam_id in range(num_cams):
            pt3dmeshes = self.pt3dmeshes[mesh_id]

            # pix_to_face: BxHxWx1, zbuf, bary_coord: BxHxWx1x3, dists
            fragments = rasterizer(pt3dmeshes.extend(num_cams)) # , R=R, t=t, focal_length=focal_length, principal_point=principal_point, imgs_sizes=imgs_sizes)

            if replace_feats_with_mask_vsbl:
                B = fragments.pix_to_face.shape[0]
                verts_ids_vsbl = self.get_faces_with_mesh_id(mesh_id).repeat(num_cams, 1)[fragments.pix_to_face.reshape(B, -1)].reshape(B, -1) # .unique(dim=1)
                verts_vsbl_mask = torch.zeros(size=(B, self.verts_counts_max), dtype=torch.bool, device=device)
                for b in range(B):
                    verts_vsbl_mask[b, verts_ids_vsbl[b]] = 1
                return verts_vsbl_mask

            if replace_feats_with_verts3d is False:
                feats_from_faces = self.get_feats_from_faces_with_mesh_id(mesh_id) #  self.feats[mesh_id][self.faces[mesh_id]]
            else:
                verts3d = self.get_verts_with_mesh_id(mesh_id)
                verts3d_ncds = (verts3d - verts3d.min(dim=0).values[None,]) / (verts3d.max(dim=0).values[None,] - verts3d.min(dim=0).values[None,])
                feats_from_faces = verts3d_ncds[self.get_faces_with_mesh_id(mesh_id)]

            mesh_feats2d_rendered = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, feats_from_faces.repeat(num_cams, 1, 1))[:, ..., 0, :].permute(0, 3, 1, 2)
            meshes_feats2d_rendered.append(mesh_feats2d_rendered)
        return torch.stack(meshes_feats2d_rendered, dim=0)

        """
        if modality == "depth":
            fragments = rasterizer(meshes)
            return (fragments.zbuf[0]).permute(2, 0, 1)
        elif modality == "mask_verts_vsbl":
            fragments = rasterizer(mesh)
            B = fragments.pix_to_face.shape[0]
            verts_ids_vsbl = faces[fragments.pix_to_face.reshape(B, -1)].reshape(B, -1).unique(dim=1)
            verts_vsbl_mask = torch.zeros(size=(B, verts_shape[0]), dtype=torch.bool, device=device)
            verts_vsbl_mask[verts_ids_vsbl] = 1
        elif modality == "interpolate":
            # pix_to_face, zbuf, bary_coord, dists
            fragments = rasterizer(mesh)
            if feats is None:
                verts_rgb_ncds = verts.clone()
                verts_rgb_ncds = (verts_rgb_ncds - verts_rgb_ncds.min(dim=0).values[None,]) / (verts_rgb_ncds.max(dim=0).values[None,] - verts_rgb_ncds.min(dim=0).values[None,])
                feats = verts_rgb_ncds
            return interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, feats[faces])[0, ..., 0, :].permute(2, 0, 1)
        elif modality == "nearest":
            pix_to_face, zbuf, bary_coord, dists = rasterizer(mesh)
            # TODO:
            raise NotImplementedError
            #ori_shape = bary_coord.shape
            #exr = bary_coord * (bary_coord < 0)
            #bary_coords_ = bary_coord.view(-1, bary_coord.shape[-1])
            #arg_max_idx = bary_coords_.argmax(1)
            #bary_coord = (
            #        torch.zeros_like(bary_coords_)
            #        .scatter(1, arg_max_idx.unsqueeze(1), 1.0)
            #        .view(*ori_shape)
            #        + exr
            #)
            return interpolate_face_attributes(pix_to_face, bary_coord, verts_rgb).squeeze()
        else:
            # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
            # -z direction.
            lights = PointLights(device=device, location=[[0.0, 0.0, 10.0]])

            # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
            # interpolate the texture uv coordinates for each vertex, sample from a texture image and
            # apply the Phong lighting model
            renderer = MeshRenderer(
                rasterizer=rasterizer,
                shader=HardPhongShader(
                    device=device,
                    cameras=cameras,
                    lights=lights
                )
            )

            rgba_synthetic_batch = renderer(mesh)
            rgba_synthetic = (rgba_synthetic_batch[0, ..., :] * 255).to(torch.uint8).permute(2, 0, 1)

            if modality == "rgba" or modality == "all":
                return rgba_synthetic
            else:
                return rgba_synthetic[:3]
        """