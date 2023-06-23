import logging

from od3d.cv.geometry.transform import tform4x4
from torch.utils.data import Dataset
from omegaconf import OmegaConf, DictConfig
from enum import Enum
from od3d.cv.geometry.mesh import Meshes, MESH_RENDER_MODALITIES

from dataclasses import dataclass
from pathlib import Path
import torch
from od3d.cv.io import read_image, read_co3d_depth_image
import torchvision
from dataclasses import dataclass, field
from typing import List
import logging
logger = logging.getLogger(__name__)


class OD3D_FRAME_MODALITIES(str, Enum):
    RGB = 'rgb'
    MASK = 'mask'
    DEPTH = 'depth'
    DEPTH_MASK = 'depth_mask'
    MESH = 'mesh'
    KPTS = 'kpts'
    BBOX = 'bbox'
    CUBOID_FRONT_TFORM4X4_OBJ = 'cuboid_front_tform4x4_obj'
    SEQUENCE_NAME = 'sequence_name'
    SEQUENCE = 'sequence'

class OD3D_SEQ_MODALITIES(str, Enum):
    PCL = 'pcl'


@dataclass
class OD3D_Frame:
    path_dataset: Path
    category: str
    name: str
    rfpath_rgb: Path
    rfpath_mask: Path
    rfpath_depth: Path
    rfpath_depth_mask: Path
    # rfpath_pcl: Path
    l_cam_tform4x4_obj: List[List[float]] #  torch.Tensor
    l_cam_intr4x4: List[List[float]] # torch.Tensor
    l_size: List[float] # torch.Tensor
    H: int
    W: int
    label= None
    _cam_tform4x4_obj = None
    _cam_intr4x4 = None
    _size = None
    _rgb = None
    _mask = None
    _depth = None
    _depth_mask = None

    @property
    def size(self):
        if self._size is None:
            self._size = torch.Tensor(self.l_size)
        return self._size

    @property
    def cam_intr4x4(self):
        if self._cam_intr4x4 is None:
            self._cam_intr4x4 = torch.Tensor(self.l_cam_intr4x4)
        return self._cam_intr4x4

    @property
    def cam_tform4x4_obj(self):
        if self._cam_tform4x4_obj is None:
            self._cam_tform4x4_obj = torch.Tensor(self.l_cam_tform4x4_obj)
        return self._cam_tform4x4_obj

    @property
    def cam_proj4x4_obj(self):
        return tform4x4(self.cam_intr4x4, self.cam_tform4x4_obj)
    @property
    def mask(self):
        if self._mask is None:
            self._mask = read_image(self.path_dataset.joinpath(self.rfpath_mask)) / 255.
        return self._mask

    @property
    def rgb(self):
        if self._rgb is None:
            self._rgb = torchvision.io.read_image(str(self.path_dataset.joinpath(self.rfpath_rgb)), mode=torchvision.io.ImageReadMode.RGB)
        return self._rgb

    @property
    def depth(self):
        raise NotImplementedError

    @property
    def depth_mask(self):
        if self._depth_mask is None:
            self._depth_mask = read_image(self.path_dataset.joinpath(self.rfpath_depth_mask))
        return self._depth_mask

@dataclass
class OD3D_Frames():
    modalities: List[OD3D_FRAME_MODALITIES]
    length: int
    name: List[str]
    path_co3d: Path
    size: torch.Tensor
    cam_intr4x4: torch.Tensor
    cam_tform4x4_obj: torch.Tensor
    category: List[str]
    label: torch.LongTensor
    dtype: None
    device: None
    sequence_name: None
    sequence: None
    rgb: None
    depth: None
    mask: None
    depth_mask: None
    kpts2d_annot: None
    kpts2d_annot_vsbl: None
    kpts_names: None
    kpts3d: None
    bbox: None

    @staticmethod
    def get_frames_from_list(frames: List[OD3D_Frame], modalities: List[OD3D_FRAME_MODALITIES], dtype, device):
        frame0 = frames[0]

        length = len(frames)
        name = [frame.name for frame in frames]
        dtype = dtype
        device = device
        path_co3d = frame0.path_dataset
        size = frame0.size # .to(device=device)
        cam_intr4x4 = torch.stack([frame.cam_intr4x4 for frame in frames], dim=0) # .to(device=device)
        cam_tform4x4_obj = torch.stack([frame.cam_tform4x4_obj for frame in frames], dim=0) #.to(device=device)
        category = [frame.category for frame in frames]
        label = torch.LongTensor([frame.label for frame in frames]) # .to(device=device)

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
        else:
            rgb = None

        if OD3D_FRAME_MODALITIES.MASK in modalities:
            mask = torch.stack([frame.mask for frame in frames], dim=0) #.to(device=device)
        else:
            mask = None

        if OD3D_FRAME_MODALITIES.DEPTH in modalities:
            depth = torch.stack([frame.depth for frame in frames], dim=0)# .to(device=device)
        else:
            depth = None

        if OD3D_FRAME_MODALITIES.DEPTH_MASK in modalities:
            depth_mask = torch.stack([frame.depth_mask for frame in frames], dim=0)# .to(device=device)
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

        return OD3D_Frames(modalities=modalities, length=length,name=name,  dtype=dtype, device=device,
                           path_co3d=path_co3d, size=size, cam_intr4x4=cam_intr4x4, cam_tform4x4_obj=cam_tform4x4_obj,
                           category=category, label=label, sequence_name=sequence_name,
                           rgb=rgb, depth = depth,
                           mask=mask, depth_mask=depth_mask, kpts2d_annot=kpts2d_annot,
                           kpts2d_annot_vsbl=kpts2d_annot_vsbl, kpts_names=kpts_names, kpts3d=kpts3d, bbox = bbox, sequence=sequence)


    def get_items(self, items):
        return OD3D_Frames(modalities=self.modalities, length=len(items), name=[self.name[item] for item in items], dtype=self.dtype, device=self.device,
                           path_co3d=self.path_co3d, size=self.size, cam_intr4x4=self.cam_intr4x4[items], cam_tform4x4_obj=self.cam_tform4x4_obj[items],
                           category=[self.category[item] for item in items], label=self.label[items],
                           sequence_name=[self.sequence_name[item] for item in items] if self.sequence_name is not None else None,
                           rgb=self.rgb[items] if self.rgb is not None else None,
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
        img = blend_rgb(self.rgb[0], self.mask[0] * 255)

        if OD3D_FRAME_MODALITIES.KPTS in self.modalities:
            img = draw_pixels(pxls=self.kpts2d_annot[0][self.kpts2d_annot_vsbl[0]], img=img, colors=[0., 0., 255.])

            kpts3d_inf_mask = torch.isinf(self.kpts3d[0]).any(dim=-1)
            if kpts3d_inf_mask.sum() > 0:
                logger.warn(f'There are {kpts3d_inf_mask.sum()} kpts with infinity for label {self.category[0]}')
            kpts3d = self.kpts3d[0][~kpts3d_inf_mask]
            kpts3d = torch.cat([kpts3d, torch.zeros(size=(1, 3,), device=self.device)])
            kpts3d2d = proj3d2d_broadcast(proj4x4=self.cam_proj4x4_obj[0], pts3d=kpts3d)
            img = draw_pixels(pxls=kpts3d2d, img=img, colors=[0., 255., 0.])

        if OD3D_FRAME_MODALITIES.BBOX in self.modalities:
            img = draw_bbox(img=img, bbox=self.bbox[0])

        if self.sequence is not None:
            # if self.sequence_name
            img = blend_rgb(img, (self.sequence[0].cuboid.render_feats(
                                    cams_tform4x4_obj=self.cam_tform4x4_obj[:1],
                                    cams_intr4x4=self.cam_intr4x4[:1],
                                    imgs_sizes=self.size, meshes_ids=torch.LongTensor([0]),
                                    modality=MESH_RENDER_MODALITIES.VERTS_NCDS)[0]).to(dtype=self.rgb.dtype))

        #mix_real_with_synthetic = draw_pixels(mix_real_with_synthetic,
        #                                      proj3d2d_broadcast(pts3d=torch.cat((pts3d, self.kpts3d[0, self.kpts3d_vsbl[0]])),
        #                                               proj4x4=self.cam_proj4x4_obj[0]), colors=(0, 255, 0))
        #mix_real_with_synthetic = draw_pixels(mix_real_with_synthetic, self.kpts2d_annot[0, self.kpts2d_annot_vsbl[0]],
        #                                     colors=(0, 0, 255), radius_in=2, radius_out=4)

        show_img(img)

    def to(self, device: torch.device):
        if self.device != device:
            for k, a in self.__dict__.items():
                if isinstance(a, torch.Tensor):
                    setattr(self, k, a.to(device))
                    # self.__dict__[k] = a.to(device)
            self.device = device

class OD3D_Dataset(Dataset):
    subclasses = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls
    def __init__(self, config: DictConfig, transform=None):
        self.config = config
        if transform is None:
            import torchvision
            from od3d.cv.transforms.center_and_zoom3d import CenterZoom3D
            from od3d.cv.transforms.rgb import RGB_UInt8ToFloat, RGB_Normalize, RGB_Random
            from od3d.cv.transforms.center_and_zoom3d import CenterZoom3D
            transform = torchvision.transforms.Compose([
                # RGB_Random(),
                RGB_UInt8ToFloat(),
            ])
        self.transform = transform
        self.index_shift = self.config.get('index_shift', 0)

    def __len__(self):
        raise NotImplementedError
    def __getitem__(self, item):
        return self.get_item((item + self.index_shift) % len(self))
    def get_item(self, item):
        raise NotImplementedError

    def collate_fn(self, frames: List[OD3D_Frame], device='cpu', dtype=torch.float32):
        frames = OD3D_Frames.get_frames_from_list(frames, modalities=self.config.modalities, dtype=dtype, device=device)
        return frames

    @staticmethod
    def setup(config: DictConfig):
        raise NotImplementedError
    def visualize(self, item: int):
        raise NotImplementedError
