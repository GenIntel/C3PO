import logging
from pytorch3d.io import IO
import torch
from pathlib import Path
from pytorch3d.structures.meshes import Meshes as PT3DMeshes
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.renderer.mesh.utils import interpolate_face_attributes
logger = logging.getLogger(__name__)

class Mesh:
    def __init__(self, fpath_mesh, device, rgb=None, feats=None):
        io = IO()
        mesh = io.load_mesh(fpath_mesh, device=device)
        self.verts = mesh[0].verts_list()[0]
        self.faces = mesh[0].faces_list()[0]
        self.rgb = rgb
        self.feats = feats
    def verts_count(self):
        return self.verts.shape[0]
class Meshes:
    def __init__(self, meshes: list[Mesh] = None, fpaths_meshes: list[Path] = None, device=None):
        self.device = device
        if meshes is None:
            meshes = []
            for fpath_mesh in fpaths_meshes:
                meshes.append(Mesh(fpath_mesh=fpath_mesh, device=device))
        mesh0 = meshes[0]
        self.verts = [mesh.verts for mesh in meshes]
        self.faces = [mesh.faces for mesh in meshes]
        if mesh0.rgb is not None:
            self.rgb = [mesh.rgb for mesh in meshes]
        else:
            self.rgb = None
        if mesh0.feats is not None:
            self.feats = [mesh.feats for mesh in meshes]
        else:
            self.feats = None

    def __len__(self):
        return len(self.verts)
    def verts_count_max(self):
        return max([verts.shape[0] for verts in self.verts])

    def to(self, device):
        if self.device != device:
            self.verts = [v.to(device=device) for v in self.verts]
            self.faces = [f.to(device=device) for f in self.faces]
            if self.rgb is not None:
                self.rgb = [i.to(device=device) for i in self.rgb]
            if self.feats is not None:
                self.feats = [f.to(device=device) for f in self.feats]
            self.device = device
    def add_feats(self, feats):
        verts_count_max = self.verts_count_max()
        self.feats = [feats[i*verts_count_max : i*verts_count_max + self.verts[i].shape[0]].to(device=self.device) for i in range(len(self))]

    def render_feats(self, cams_tform4x4_obj, cams_intr4x4, imgs_sizes, meshes_ids=None, replace_feats_with_verts3d=False):

        if meshes_ids is None:
            meshes_ids = list(range(len(self)))
        dtype = cams_tform4x4_obj.dtype
        device = cams_tform4x4_obj.device

        self.to(device)

        num_cams = cams_tform4x4_obj.shape[0]

        #cams_tform4x4_obj = cams_tform4x4_obj.repeat_interleave(num_meshes, dim=0)
        #cams_intr4x4 = cams_intr4x4.repeat_interleave(num_meshes, dim=0)
        #imgs_sizes = imgs_sizes.repeat_interleave(num_meshes, dim=0)
        t3d_tform_pscl3d = torch.Tensor([[-1., 0., 0., 0.],
                                         [0., -1., 0., 0.],
                                         [0., 0., 1., 0.],
                                         [0., 0., 0., 1.]]).to(device=device, dtype=dtype)
        t3d_cam_tform_obj = torch.matmul(t3d_tform_pscl3d, cams_tform4x4_obj)
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
        )

        # this is not parallizable currently, because different meshes have different number of vertices
        meshes_feats2d_rendered = []
        for mesh_id in meshes_ids:
            mesh_feats2d_rendered = []
            for cam_id in range(num_cams):
                rasterizer = MeshRasterizer(
                    cameras=cameras[cam_id],
                    raster_settings=raster_settings
                )
                pt3dmeshes = PT3DMeshes(
                    verts=[self.verts[mesh_id].clone()] * 1,
                    faces=[self.faces[mesh_id].clone()] * 1,
                )

                # pix_to_face: BxHxWx1, zbuf, bary_coord: BxHxWx1x3, dists
                fragments = rasterizer(pt3dmeshes)
                if replace_feats_with_verts3d is False:
                    feats_from_faces = self.feats[mesh_id][self.faces[mesh_id]]
                else:
                    verts3d = self.verts[mesh_id]
                    verts3d_ncds = (verts3d - verts3d.min(dim=0).values[None,]) / (verts3d.max(dim=0).values[None,] - verts3d.min(dim=0).values[None,])
                    feats_from_faces = verts3d_ncds[self.faces[mesh_id]]

                mesh_feats2d_rendered_cam = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, feats_from_faces)[:, ..., 0, :].permute(0, 3, 1, 2)
                mesh_feats2d_rendered.append(mesh_feats2d_rendered_cam)
            mesh_feats2d_rendered = torch.cat(mesh_feats2d_rendered, dim=0)
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