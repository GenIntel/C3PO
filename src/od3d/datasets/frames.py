import logging
logger = logging.getLogger(__name__)

from pathlib import Path
import torch
from od3d.cv.geometry.transform import tform4x4
from od3d.datasets.frame import OD3D_FRAME_MODALITIES, OD3D_Frame
from typing import List
from od3d.cv.geometry.mesh import Meshes, MESH_RENDER_MODALITIES
from dataclasses import dataclass

@dataclass
class OD3D_Frames():
    modalities: List[OD3D_FRAME_MODALITIES]
    length: int
    name: List[str]
    name_unique: List[str]
    item_id: torch.LongTensor
    path_co3d: Path
    size: torch.Tensor
    cam_intr4x4: torch.Tensor
    cam_tform4x4_obj: torch.Tensor
    category: List[str]
    categories: List[List[str]]
    label: torch.LongTensor
    dtype: None
    device: None
    sequence_name: None
    sequence: None
    rgb: None
    mask_rgb: None
    depth: None
    mask: None
    depth_mask: None
    kpts2d_annot: None
    kpts2d_annot_vsbl: None
    kpts_names: None
    kpts3d: None
    bbox: None
    mesh: None

    @staticmethod
    def get_frames_from_list(frames: List[OD3D_Frame], modalities: List[OD3D_FRAME_MODALITIES], dtype, device):
        frame0 = frames[0]

        length = len(frames)
        name = [frame.name for frame in frames]
        name_unique = [frame.name_unique for frame in frames]
        dtype = dtype
        device = device
        item_id = torch.cat([torch.LongTensor([frame.item_id, ]) for frame in frames], dim=0)
        path_co3d = frame0.path_raw
        size = frame0.size # .to(device=device)


        if OD3D_FRAME_MODALITIES.CAM_INTR4X4 in modalities:
            cam_intr4x4 = torch.stack([frame.cam_intr4x4 for frame in frames], dim=0) # .to(device=device)
        else:
            cam_intr4x4 = None
        if OD3D_FRAME_MODALITIES.CAM_TFORM4X4_OBJ in modalities:
            cam_tform4x4_obj = torch.stack([frame.cam_tform4x4_obj for frame in frames], dim=0) #.to(device=device)
        else:
            cam_tform4x4_obj = None

        if OD3D_FRAME_MODALITIES.CATEGORIES in modalities:
            categories = [frame.categories for frame in frames]
        else:
            categories = None

        if OD3D_FRAME_MODALITIES.CATEGORY in modalities:
            category = [frame.category for frame in frames]
            label = torch.LongTensor([frame.category_id for frame in frames]) # .to(device=device)
        else:
            category = None
            label = None

        if OD3D_FRAME_MODALITIES.SEQUENCE_NAME in modalities:
            sequence_name = [frame.sequence.name for frame in frames]
        else:
            sequence_name = None

        if OD3D_FRAME_MODALITIES.SEQUENCE in modalities:
            sequence = [frame.sequence for frame in frames]
        else:
            sequence = None
        # if OD3D_FRAME_MODALITIES.CUBOID_FRONT_TFORM4X4_OBJ in modalities:
        #
        #    cuboid_front_tform4x4_obj = torch.stack([frame.sequence.cuboid_front_tform4x4_obj for frame in frames],
        #                                                  dim=0)
        #
        #    self.cam_tform4x4_obj = tform4x4(self.cam_tform4x4_obj, cuboid_front_tform4x4_obj.inverse())
        #    self.cam_proj4x4_obj = tform4x4(self.cam_intr4x4, self.cam_tform4x4_obj)

        if OD3D_FRAME_MODALITIES.RGB in modalities:
            rgb = torch.stack([frame.rgb for frame in frames], dim=0) #.to(device=device)
            mask_rgb = torch.stack([frame.mask_rgb for frame in frames], dim=0) #.to(device=device)
        else:
            rgb = None
            mask_rgb = None

        if OD3D_FRAME_MODALITIES.MASK in modalities:
            mask = torch.stack([frame.mask for frame in frames], dim=0) #.to(device=device)
        else:
            mask = None

        if OD3D_FRAME_MODALITIES.DEPTH in modalities:
            depth = torch.stack([frame.depth for frame in frames], dim=0) # .to(device=device)
        else:
            depth = None

        if OD3D_FRAME_MODALITIES.DEPTH_MASK in modalities:
            depth_mask = torch.stack([frame.depth_mask for frame in frames], dim=0) # .to(device=device)
        else:
            depth_mask = None

        if OD3D_FRAME_MODALITIES.KPTS in modalities:
            kpts2d_annot = [frame.kpts2d_annot for frame in frames]
            kpts2d_annot_vsbl = [frame.kpts2d_annot_vsbl for frame in frames]
            kpts_names = [frame.kpts_names for frame in frames]
            kpts3d = [frame.kpts3d for frame in frames]
        else:
            kpts2d_annot = None
            kpts2d_annot_vsbl = None
            kpts_names = None
            kpts3d = None

        if OD3D_FRAME_MODALITIES.BBOX in modalities:
            bbox = torch.stack([frame.bbox for frame in frames])
        else:
            bbox = None

        if OD3D_FRAME_MODALITIES.MESH in modalities:
            mesh = Meshes.load_from_meshes(meshes=[frame.mesh for frame in frames], device=device)
        else:
            mesh = None

        return OD3D_Frames(modalities=modalities, length=length, name=name, name_unique=name_unique, dtype=dtype, device=device, item_id=item_id,
                           path_co3d=path_co3d, size=size, cam_intr4x4=cam_intr4x4, cam_tform4x4_obj=cam_tform4x4_obj,
                           category=category, categories=categories, label=label, sequence_name=sequence_name,
                           rgb=rgb, depth=depth, mesh=mesh,
                           mask=mask, depth_mask=depth_mask, kpts2d_annot=kpts2d_annot,
                           kpts2d_annot_vsbl=kpts2d_annot_vsbl, kpts_names=kpts_names, kpts3d=kpts3d, bbox = bbox, sequence=sequence, mask_rgb=mask_rgb)


    def get_items(self, items):
        return OD3D_Frames(modalities=self.modalities, length=len(items), name=[self.name[item] for item in items], name_unique=[self.name_unique[item] for item in items], dtype=self.dtype, device=self.device,
                           item_id=self.item_id[items],
                           path_co3d=self.path_co3d, size=self.size, cam_intr4x4=self.cam_intr4x4[items], cam_tform4x4_obj=self.cam_tform4x4_obj[items],
                           categories=[self.categories[item] for item in items],
                           category=[self.category[item] for item in items], label=self.label[items],
                           sequence_name=[self.sequence_name[item] for item in items] if self.sequence_name is not None else None,
                           rgb=self.rgb[items] if self.rgb is not None else None,
                           mask_rgb=self.mask_rgb[items] if self.mask_rgb is not None else None,
                           depth=self.depth[items] if self.depth is not None else None,
                           mask=self.mask[items] if self.mask is not None else None,
                           depth_mask=self.depth_mask[items] if self.depth_mask is not None else None,
                           kpts2d_annot=self.kpts2d_annot[items] if self.kpts2d_annot is not None else None,
                           kpts2d_annot_vsbl=self.kpts2d_annot_vsbl[items] if self.kpts2d_annot_vsbl is not None else None,
                           kpts_names=self.kpts_names[items] if self.kpts_names is not None else None,
                           kpts3d=self.kpts3d[items] if self.kpts3d is not None else None,
                           bbox =self.bbox[items] if self.bbox is not None else None,
                           sequence=[self.sequence[item] for item in items] if self.sequence is not None else None)

    @property
    def cam_proj4x4_obj(self):
        return tform4x4(self.cam_intr4x4, self.cam_tform4x4_obj)
    def __len__(self):
        return self.length

    def visualize(self, cuboids: Meshes = None):
        from od3d.cv.visual.show import show_img
        from od3d.cv.visual.blend import blend_rgb
        from od3d.cv.visual.draw import draw_pixels, draw_bbox
        from od3d.cv.geometry.transform import proj3d2d_broadcast
        # show_pcl
        # print(self.rfpath_pcl[0])
        # show_img(self.rgb[0])

        #verts, _ = load_ply(str(self.path_co3d.joinpath(self.rfpath_pcl[0])))
        #verts = verts.to(self.device)

        #how_pcl(verts, cam_tform4x4_obj=self.cam_tform4x4_obj[0], cam_intr4x4=self.cam_intr4x4[0], img_size=self.size)


        # verts, faces = load_ply(filename)

        if OD3D_FRAME_MODALITIES.MASK in self.modalities:
            img = blend_rgb(self.rgb[0], self.mask[0] * 255)
        else:
            img = self.rgb[0]
        if OD3D_FRAME_MODALITIES.KPTS in self.modalities:
            img = draw_pixels(pxls=self.kpts2d_annot[0][self.kpts2d_annot_vsbl[0]], img=img, colors=[0., 0., 255.])
            #img = draw_pixels(pxls=self.kpts2d_annot[0], img=img, colors=[0., 0., 255.])

            kpts3d_inf_mask = torch.isinf(self.kpts3d[0]).any(dim=-1)
            if kpts3d_inf_mask.sum() > 0:
                logger.warning(f'There are {kpts3d_inf_mask.sum()} kpts with infinity for label {self.category[0]}')
            kpts3d = self.kpts3d[0][~kpts3d_inf_mask].to(self.device)
            kpts3d = torch.cat([kpts3d, torch.zeros(size=(1, 3,), device=kpts3d.device)])

            kpts3d2d = proj3d2d_broadcast(proj4x4=self.cam_proj4x4_obj[0], pts3d=kpts3d)
            img = draw_pixels(pxls=kpts3d2d, img=img, colors=[0., 255., 0.])

        if OD3D_FRAME_MODALITIES.BBOX in self.modalities:
            img = draw_bbox(img=img, bbox=self.bbox[0])

        if OD3D_FRAME_MODALITIES.PCL in self.modalities:
            from od3d.datasets.co3d.enum import PCL_SOURCES
            from od3d.cv.visual.show import show_scene
            pcl = self.sequence[0].get_pcl(pcl_source=PCL_SOURCES.DROID_SLAM_CLEAN)
            pts3d_colors = self.sequence[0].get_pcl_colors(pcl_source=PCL_SOURCES.DROID_SLAM_CLEAN)

            show_scene(pts3d=[pcl], pts3d_colors=[pts3d_colors])

        if OD3D_FRAME_MODALITIES.DEPTH in self.modalities:
            img = blend_rgb(img, self.depth[0])
            # from od3d.cv.visual.show import show_scene
            # from od3d.cv.geometry.transform import depth2pts3d_grid, transf3d_broadcast, inv_tform4x4
            # pts3d_obj = depth2pts3d_grid(self.depth[0], cam_intr4x4=self.cam_intr4x4[0])[0].reshape(3, -1).permute(1, 0).to(torch.float)
            # pts3d_obj = transf3d_broadcast(pts3d=pts3d_obj, transf4x4=inv_tform4x4(self.cam_tform4x4_obj[0]))
            # show_scene(pts3d=[pts3d_obj],
            #            cams_tform4x4_world=self.cam_tform4x4_obj[:1], cams_intr4x4=self.cam_intr4x4[:1],
            #            cams_imgs=self.rgb[:1])

        if OD3D_FRAME_MODALITIES.MESH in self.modalities:
            # from od3d.cv.geometry.fit3d2d import fit_se3_to_corresp_3d_2d_and_masks
            cam_tform4x4_obj = self.cam_tform4x4_obj[:1]
            #cam_tform4x4_obj = fit_se3_to_corresp_3d_2d_and_masks(masks_in=self.kpts2d_annot_vsbl[0][None,] * (~torch.isinf(self.kpts3d[0]).any(dim=-1))[None,], #
            #                                                  pts1=self.kpts3d[0].T, pxl2=self.kpts2d_annot[0].T,
            #                                                  proj_mat=self.cam_intr4x4[0][:2, :3].to(device='cpu'))
            #cam_tform4x4_obj = cam_tform4x4_obj.to(device='cuda:0')
            #self.mesh.verts *= 5. # this is ionly for pascal3d required currentlay

            #cam_tform4x4_obj[:, :3, :4] = cam_tform4x4_obj[:, :3, :4] / cam_tform4x4_obj[0, :3, :3].norm(dim=-1, keepdim=True)
            # cam_tform4x4_obj[:, :3, :3] *= 5
            #from od3d.cv.visual.show import show_scene
            #show_scene(cams_tform4x4_world=self.cam_tform4x4_obj, cams_intr4x4=self.cam_intr4x4, meshes=self.mesh)

            img = blend_rgb(img, (self.mesh.render_feats(
                cams_tform4x4_obj=cam_tform4x4_obj,
                cams_intr4x4=self.cam_intr4x4[:1],
                imgs_sizes=self.size, meshes_ids=torch.LongTensor([0]),
                modality=MESH_RENDER_MODALITIES.VERTS_NCDS)[0] * 255).to(dtype=self.rgb.dtype, device=img.device))

        elif self.sequence is not None and self.sequence[0].cuboid_labeled:
            # if self.sequence_name
            img = blend_rgb(img, (self.sequence[0].cuboid.render_feats(
                                    cams_tform4x4_obj=self.cam_tform4x4_obj[:1],
                                    cams_intr4x4=self.cam_intr4x4[:1],
                                    imgs_sizes=self.size, meshes_ids=torch.LongTensor([0]),
                                    modality=MESH_RENDER_MODALITIES.VERTS_NCDS)[0]).to(dtype=self.rgb.dtype, device=img.device))


        #mix_real_with_synthetic = draw_pixels(mix_real_with_synthetic,
        #                                      proj3d2d_broadcast(pts3d=torch.cat((pts3d, self.kpts3d[0, self.kpts3d_vsbl[0]])),
        #                                               proj4x4=self.cam_proj4x4_obj[0]), colors=(0, 255, 0))
        #mix_real_with_synthetic = draw_pixels(mix_real_with_synthetic, self.kpts2d_annot[0, self.kpts2d_annot_vsbl[0]],
        #                                     colors=(0, 0, 255), radius_in=2, radius_out=4)

        show_img(img)
        return img

    def to(self, device: torch.device):
        if self.device != device:
            for k, a in self.__dict__.items():
                if isinstance(a, torch.Tensor):
                    setattr(self, k, a.to(device))
                    # self.__dict__[k] = a.to(device)
            self.device = device
