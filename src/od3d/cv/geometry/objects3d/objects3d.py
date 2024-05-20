import logging

logger = logging.getLogger(__name__)
import abc

from od3d.cv.visual.sample import sample_pxl2d_pts

from enum import Enum
from typing import List, Union, Optional

from omegaconf import DictConfig
import inspect
import torch
import torch.nn as nn


class PROJECT_MODALITIES(str, Enum):
    DEPTH = "depth"
    MASK = "mask"
    MASK_VERTS_VSBL = "mask_verts_vsbl"
    RGB = "rgb"
    RGBA = "rgba"
    FEATS = "feats"
    IMG = "img"
    CLUTTER_PXL2D = "clutter_pxl2d"
    ID = "id"
    ONEHOT = "onehot"
    ONEHOT_SMOOTH = "onehot_smooth"
    PT3D = "pt3d"
    PT3D_NCDS = "pt3d_ncds"
    PXL2D = "pxl2d"


class FEATS_DISTR(str, Enum):
    VON_MISES_FISHER = "von-mises-fisher"
    GAUSSIAN = "gaussian"


# method = OD3D_Objects3D.subclasses[self.config.method.class_name]()


class OD3D_Objects3D(abc.ABC, nn.Module):
    subclasses = {}
    feat_clutter: Optional[torch.Tensor]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def create_from_config(cls, config: DictConfig):
        keys = inspect.getfullargspec(cls.__init__)[0][1:]
        od3d_objects3d = cls(
            **{
                key: config.get(key)
                for key in keys
                if config.get(key, None) is not None
            },
        )
        return od3d_objects3d

    def read_from_files(self):
        raise NotImplementedError

    def read_from_file(self):
        raise NotImplementedError

    def write_to_files(self):
        raise NotImplementedError

    def __init__(
        self,
        feat_dim=128,
        objects_count=0,
        feat_clutter=False,
        feats_objects=False,
        feats_requires_grad=True,
        feats_distribution=FEATS_DISTR.VON_MISES_FISHER,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        self.feat_dim = feat_dim
        self.objects_count = objects_count

        if feat_clutter:
            self.feat_clutter = torch.nn.Parameter(
                torch.empty(self.feat_dim, **factory_kwargs),
                requires_grad=feats_requires_grad,
            )  # F,
        else:
            self.register_parameter("feat_clutter", None)  # None

        self.feats_distribution = feats_distribution

    def reset_parameters(self) -> None:
        if self.feat_clutter is not None:
            import math

            # equals kaiming uniform
            bound = 1 / math.sqrt(self.feat_dim) if self.feat_dim > 0 else 0
            #torch.nn.init.uniform_(self.feat_clutter, a=-bound, b=bound)
            torch.nn.init.uniform_(self.feat_clutter, a=0., b=1.) # note: somehow better at least without head

        self.normalize_feats()

    def normalize_feats(self):
        if self.feat_clutter is not None:
            self.feat_clutter.data = (
                self.feat_clutter.detach()
                / (self.feat_clutter.detach().norm(dim=-1, keepdim=True) + 1e-10)
            )

    def __len__(self):
        return self.objects_count

    def cams_downsample(self, cams_intr4x4=None, imgs_sizes=None, down_sample_rate=1.0):
        """
        Render the objects in the scene with the given camera parameters.
        Args:
            cams_tform4x4_obj: (B, C, 4, 4) or (B, 4, 4) tensor of camera poses in the object frame.
            cams_intr4x4: (4, 4), or (B, 4, 4) or (B, C, 4, 4) tensor of camera intrinsics.
            imgs_sizes: (2,) tensor of (H, W)
        Returns:
            cams_intr4x4: (4, 4), or (B, 4, 4) or (B, C, 4, 4) tensor of camera intrinsics.
            imgs_sizes: (2,) tensor of (H, W)
        """
        if down_sample_rate != 1.0:
            if cams_intr4x4 is not None:
                cams_intr4x4 = cams_intr4x4.clone()
                if cams_intr4x4.dim() == 2:
                    cams_intr4x4[:2] /= down_sample_rate
                elif cams_intr4x4.dim() == 3:
                    cams_intr4x4[:, :2] /= down_sample_rate
                elif cams_intr4x4.dim() == 4:
                    cams_intr4x4[:, :, :2] /= down_sample_rate
                else:
                    raise NotImplementedError

            if imgs_sizes is not None:
                if isinstance(imgs_sizes, torch.Size):
                    imgs_sizes = torch.LongTensor(list(imgs_sizes))
                imgs_sizes = imgs_sizes.clone() // down_sample_rate
        else:
            if cams_intr4x4 is not None:
                cams_intr4x4 = cams_intr4x4.clone()
            if imgs_sizes is not None:
                if isinstance(imgs_sizes, torch.Size):
                    imgs_sizes = torch.LongTensor(list(imgs_sizes))
                imgs_sizes = imgs_sizes.clone()
        return cams_intr4x4, imgs_sizes

    def cams_and_objects_broadcast(self, cams_tform4x4_obj, cams_intr4x4, objects_ids):
        objects_count = objects_ids.shape[0]
        if cams_tform4x4_obj.dim() == 4:
            cams_count = cams_tform4x4_obj.shape[1]
        elif cams_tform4x4_obj.dim() == 3:
            cams_count = cams_tform4x4_obj.shape[0]
        else:
            raise ValueError(f"Set `cams_tform4x4_obj.dim()` must be 3 or 4")

        objects_ids = objects_ids
        if cams_tform4x4_obj.dim() == 3:
            cams_tform4x4_obj = cams_tform4x4_obj[None, :]
        if cams_intr4x4.dim() == 3:
            cams_intr4x4 = cams_intr4x4[None, :]
        cams_tform4x4_obj = cams_tform4x4_obj.expand(
            objects_count,
            cams_count,
            4,
            4,
        ).reshape(-1, 4, 4)
        cams_intr4x4 = cams_intr4x4.expand(objects_count, cams_count, 4, 4).reshape(
            -1,
            4,
            4,
        )
        objects_ids = objects_ids[:, None].expand(objects_count, cams_count).reshape(-1)

        return cams_tform4x4_obj, cams_intr4x4, objects_ids

    def forward(self):
        pass

    def render(
        self,
        cams_tform4x4_obj,
        cams_intr4x4,
        imgs_sizes,
        objects_ids=None,
        modalities: Union[
            PROJECT_MODALITIES, List[PROJECT_MODALITIES]
        ] = PROJECT_MODALITIES.FEATS,
        broadcast_batch_and_cams=False,
        down_sample_rate=1.0,
        add_clutter=False,
        add_other_objects=False,
    ):
        """
        Render the objects in the scene with the given camera parameters.
        Args:
            cams_tform4x4_obj: (B, C, 4, 4) or (B, 4, 4) tensor of camera poses in the object frame.
            cams_intr4x4: (4, 4), or (B, 4, 4) or (B, C, 4, 4) tensor of camera intrinsics.
            imgs_sizes: (2,) tensor of (H, W)
            objects_ids: (B, ) tensor of objects ids to render.
            modalities: (Union[PROJECT_MODALITIES, List(PROJECT_MODALITIES)]) the modalities to render.
            broadcast_batch_and_cams: bool, whether to broadcast the batch and camera dimensions.
            down_sample_rate: float, the downsample rate for the rendered image.
            add_clutter: bool, determines wether to add clutter features to the rendered features.
            add_other_objects: bool, determines wether to add other objects' features to the rendered features.
        Returns:
            mods2d_rendered (Dict[PROJECT_MODALITIES, torch.Tensor]): (B, F, H, W) dict of rendered modalities.
        """

        self()

        if not isinstance(modalities, List):
            _modalities = [modalities]
        else:
            _modalities = modalities

        # imgs_size: (height, width)
        dtype = cams_tform4x4_obj.dtype
        device = cams_tform4x4_obj.device

        self.to(device)

        cams_intr4x4, imgs_sizes = self.cams_downsample(
            cams_intr4x4, imgs_sizes, down_sample_rate
        )

        if objects_ids is None:
            objects_ids = torch.LongTensor(list(range(len(self)))).to(device=device)
        elif isinstance(objects_ids, int):
            objects_ids = torch.LongTensor([objects_ids]).to(device=device)
        elif isinstance(objects_ids, List):
            objects_ids = torch.LongTensor(objects_ids).to(device=device)
        elif isinstance(objects_ids, torch.LongTensor):
            objects_ids = objects_ids.clone().to(device=device)

        if broadcast_batch_and_cams:
            objects_count = objects_ids.shape[0]

            (
                cams_tform4x4_obj,
                cams_intr4x4,
                objects_ids,
            ) = self.cams_and_objects_broadcast(
                cams_tform4x4_obj=cams_tform4x4_obj,
                cams_intr4x4=cams_intr4x4,
                objects_ids=objects_ids,
            )

            cams_times_objects_count = cams_tform4x4_obj.shape[0]
            cams_count = int(cams_times_objects_count / objects_count)
        else:
            objects_count = objects_ids.shape[0]
            if cams_tform4x4_obj is not None:
                cams_count = cams_tform4x4_obj.shape[0]
            else:
                cams_count = objects_count

            if objects_count != cams_count:
                raise ValueError(
                    f"Set `broadcast_batch_and_cams=True` to allow different number of cameras and objects",
                )

        mods2d_rendered = self.render_batch(
            cams_tform4x4_obj=cams_tform4x4_obj,
            cams_intr4x4=cams_intr4x4,
            imgs_sizes=imgs_sizes,
            objects_ids=objects_ids,
            modalities=_modalities,
            add_clutter=add_clutter,
            add_other_objects=add_other_objects,
        )

        if broadcast_batch_and_cams:
            for key, val in mods2d_rendered.items():
                mods2d_rendered[key] = val.reshape(
                    objects_count,
                    cams_count,
                    *val.shape[1:],
                )

        if not isinstance(modalities, List):
            return mods2d_rendered[_modalities[0]]
        else:
            return mods2d_rendered

    def render_batch(
        self,
        cams_tform4x4_obj,
        cams_intr4x4,
        imgs_sizes,
        objects_ids=None,
        modalities: Union[
            PROJECT_MODALITIES, List[PROJECT_MODALITIES]
        ] = PROJECT_MODALITIES.FEATS,
        add_clutter=False,
        add_other_objects=False,
    ):
        """
        Render the objects in the scene with the given camera parameters.
        Args:
            cams_tform4x4_obj: (B, 4, 4) tensor of camera poses in the object frame.
            cams_intr4x4: (B, 4, 4) tensor of camera intrinsics.
            imgs_sizes: (2,) tensor of (H, W)
            objects_ids: (B, ) tensor of objects ids to render.
            modalities: (Union[PROJECT_MODALITIES, List(PROJECT_MODALITIES)]) the modalities to render.
            add_clutter: bool, determines whether to add clutter features to the rendered features.
            add_other_objects: bool, determines whether to add other objects' features to the rendered features.
        Returns:
            mods2d_rendered (Dict[PROJECT_MODALITIES, torch.Tensor]): (B, F, H, W) dict of rendered modalities.
        """

        raise NotImplementedError

    def sample_with_img2d(
        self,
        img2d=None,
        img2d_mask=None,
        cams_tform4x4_obj=None,
        cams_intr4x4=None,
        imgs_sizes=None,
        objects_ids=None,
        modalities: Union[
            PROJECT_MODALITIES, List[PROJECT_MODALITIES]
        ] = PROJECT_MODALITIES.FEATS,
        broadcast_batch_and_cams=False,
        down_sample_rate=1.0,
        add_clutter=False,
        add_other_objects=False,
        dtype=None,
        device=None,
        sample_clutter_count=0,
        clutter_pxl2d=None,
    ):
        """
        Sample the objects' projection.
        Args:
            img2d: (B, C, H, W) tensor of the image.
            img2d_mask: (B, 1, H, W) tensor of the image mask.
            cams_tform4x4_obj: (B, C, 4, 4) or (B, 4, 4) tensor of camera poses in the object frame.
            cams_intr4x4: (4, 4), or (B, 4, 4) or (B, C, 4, 4) tensor of camera intrinsics.
            imgs_sizes: (2,) tensor of (H, W)
            objects_ids: (B, ) tensor of objects ids to render.
            modalities: (Union[PROJECT_MODALITIES, List(PROJECT_MODALITIES)]) the modalities to render.
            broadcast_batch_and_cams: bool, whether to broadcast the batch and camera dimensions.
            down_sample_rate: float, the downsample rate for the rendered image.
            add_clutter: bool, determines wether to add clutter features to the sampled features.
            add_other_objects: bool, determines wether to add other objects' features to the sampled features.
        Returns:
            mods1d_sampled (Dict[PROJECT_MODALITIES, torch.Tensor]): (B, C, V, F), or (B, V, F) dict of projected modalities.
        """

        if dtype is None and cams_tform4x4_obj is not None:
            dtype = cams_tform4x4_obj.dtype
        if device is None and cams_tform4x4_obj is not None:
            device = cams_tform4x4_obj.device

        if not isinstance(modalities, List):
            _modalities = [modalities]
        else:
            _modalities = modalities

        sample_modalities = _modalities.copy()
        if PROJECT_MODALITIES.IMG in sample_modalities:
            sample_modalities.remove(PROJECT_MODALITIES.IMG)
        if PROJECT_MODALITIES.CLUTTER_PXL2D in sample_modalities:
            sample_modalities.remove(PROJECT_MODALITIES.CLUTTER_PXL2D)
        if PROJECT_MODALITIES.MASK not in sample_modalities:
            sample_modalities.append(PROJECT_MODALITIES.MASK)
        if PROJECT_MODALITIES.PXL2D not in sample_modalities:
            sample_modalities.append(PROJECT_MODALITIES.PXL2D)

        mods1d_sampled = self.sample(
            modalities=sample_modalities,
            cams_tform4x4_obj=cams_tform4x4_obj,
            cams_intr4x4=cams_intr4x4,
            imgs_sizes=imgs_sizes,
            objects_ids=objects_ids,
            broadcast_batch_and_cams=broadcast_batch_and_cams,
            down_sample_rate=down_sample_rate,
            add_clutter=add_clutter,
            add_other_objects=add_other_objects,
            dtype=dtype,
            device=device,
        )

        feats1d_obj_mask, feats1d_obj_pxl2d = (
            mods1d_sampled[PROJECT_MODALITIES.MASK],
            mods1d_sampled[PROJECT_MODALITIES.PXL2D],
        )

        if sample_clutter_count > 0:
            feats2d_obj_mask = self.render(
                modalities=PROJECT_MODALITIES.MASK,
                cams_tform4x4_obj=cams_tform4x4_obj,
                cams_intr4x4=cams_intr4x4,
                imgs_sizes=imgs_sizes,
                objects_ids=objects_ids,
                broadcast_batch_and_cams=broadcast_batch_and_cams,
                down_sample_rate=down_sample_rate,
                add_clutter=False,
                add_other_objects=False,
            )
            if img2d_mask is not None:
                feats2d_clutter_mask = 1.0 - (feats2d_obj_mask * img2d_mask)
            else:
                feats2d_clutter_mask = 1.0 - feats2d_obj_mask

            H, W = img2d_mask.shape[-2:]
            xy = torch.stack(
                torch.meshgrid(
                    torch.arange(W, device=device),
                    torch.arange(H, device=device),
                    indexing="xy",
                ),
                dim=0,
            )  # HxW
            prob_noise = feats2d_clutter_mask.clamp(0, 1).flatten(1)
            prob_noise[prob_noise.sum(dim=-1) <= 0.0] = 1.0
            clutter_pxl2d = xy.flatten(1)[
                :,
                torch.multinomial(prob_noise, sample_clutter_count, replacement=True),
            ].permute(1, 2, 0)

        if clutter_pxl2d is not None:
            feats1d_obj_pxl2d = torch.cat([feats1d_obj_pxl2d, clutter_pxl2d], dim=1)
            B = feats1d_obj_mask.shape[0]
            sample_clutter_count = clutter_pxl2d.shape[1]
            feats1d_clutter_mask = torch.ones(
                (B, sample_clutter_count),
                dtype=torch.bool,
                device=feats1d_obj_mask.device,
            )
            feats1d_obj_mask = torch.cat(
                [feats1d_obj_mask, feats1d_clutter_mask], dim=-1
            )

        # mods1d_sampled = {}
        for modality in _modalities:
            # feats, mask, labels
            if modality == PROJECT_MODALITIES.IMG:
                mods1d_sampled[modality] = sample_pxl2d_pts(img2d, feats1d_obj_pxl2d)
            elif modality == PROJECT_MODALITIES.PXL2D:
                mods1d_sampled[modality] = feats1d_obj_pxl2d
            elif modality == PROJECT_MODALITIES.CLUTTER_PXL2D:
                mods1d_sampled[modality] = clutter_pxl2d
            elif modality == PROJECT_MODALITIES.MASK:
                mods1d_sampled[modality] = feats1d_obj_mask
            elif (
                modality == PROJECT_MODALITIES.ONEHOT
                or modality == PROJECT_MODALITIES.ONEHOT_SMOOTH
            ):
                B = mods1d_sampled[modality].shape[0]
                V = mods1d_sampled[modality].shape[1]
                label1d_clutter = self.get_clutter_label(
                    add_other_objects=add_other_objects,
                    one_hot=True,
                    device=device,
                )[None, :, None].expand(
                    B,
                    V,
                    sample_clutter_count,
                )
                mods1d_sampled[modality] = torch.cat(
                    [mods1d_sampled[modality], label1d_clutter], dim=-1
                )
            elif modality == PROJECT_MODALITIES.ID:
                B = mods1d_sampled[modality].shape[0]
                label1d_clutter = self.get_clutter_label(
                    add_other_objects=add_other_objects, one_hot=False, device=device
                )[
                    None,
                ].expand(
                    B, sample_clutter_count
                )
                mods1d_sampled[modality] = torch.cat(
                    [mods1d_sampled[modality], label1d_clutter], dim=-1
                )
            elif modality == PROJECT_MODALITIES.FEATS:
                mods1d_sampled[modality] = mods1d_sampled[modality]
                # mods1d_sampled[modality] = torch.cat([mods1d_sampled[modality], self.feat_clutter[None, :].expand(
                #    mods1d_sampled[modality].shape[0], sample_clutter_count, self.feat_dim)], dim=-2)

            else:
                raise ValueError(f"Unknown modality {modality}")

        if not isinstance(modalities, List):
            return mods1d_sampled[_modalities[0]]
        else:
            return mods1d_sampled

    def sample(
        self,
        cams_tform4x4_obj=None,
        cams_intr4x4=None,
        imgs_sizes=None,
        objects_ids=None,
        modalities: Union[
            PROJECT_MODALITIES, List[PROJECT_MODALITIES]
        ] = PROJECT_MODALITIES.FEATS,
        broadcast_batch_and_cams=False,
        down_sample_rate=1.0,
        add_clutter=False,
        add_other_objects=False,
        dtype=None,
        device=None,
        sample_clutter=False,
        sample_other_objects=False,
    ):
        """
        Sample the objects' projection.
        Args:
            cams_tform4x4_obj: (B, C, 4, 4) or (B, 4, 4) tensor of camera poses in the object frame.
            cams_intr4x4: (4, 4), or (B, 4, 4) or (B, C, 4, 4) tensor of camera intrinsics.
            imgs_sizes: (2,) tensor of (H, W)
            objects_ids: (B, ) tensor of objects ids to render.
            modalities: (Union[PROJECT_MODALITIES, List(PROJECT_MODALITIES)]) the modalities to render.
            broadcast_batch_and_cams: bool, whether to broadcast the batch and camera dimensions.
            down_sample_rate: float, the downsample rate for the rendered image.
            add_clutter: bool, determines wether to add clutter features to the sampled features.
            add_other_objects: bool, determines wether to add other objects' features to the sampled features.
        Returns:
            mods1d_sampled (Dict[PROJECT_MODALITIES, torch.Tensor]): (B, C, V, F), or (B, V, F) dict of projected modalities.
        """

        self()

        if not isinstance(modalities, List):
            _modalities = [modalities]
        else:
            _modalities = modalities

        # imgs_size: (height, width)
        if dtype is None and cams_tform4x4_obj is not None:
            dtype = cams_tform4x4_obj.dtype
        if device is None and cams_tform4x4_obj is not None:
            device = cams_tform4x4_obj.device

        self.to(device)

        cams_intr4x4, imgs_sizes = self.cams_downsample(
            cams_intr4x4, imgs_sizes, down_sample_rate
        )

        if objects_ids is None:
            objects_ids = torch.LongTensor(list(range(len(self)))).to(device=device)
        elif isinstance(objects_ids, int):
            objects_ids = torch.LongTensor([objects_ids]).to(device=device)
        elif isinstance(objects_ids, List):
            objects_ids = torch.LongTensor(objects_ids).to(device=device)
        elif isinstance(objects_ids, torch.LongTensor):
            objects_ids = objects_ids.clone().to(device=device)

        if broadcast_batch_and_cams:
            objects_count = objects_ids.shape[0]

            (
                cams_tform4x4_obj,
                cams_intr4x4,
                objects_ids,
            ) = self.cams_and_objects_broadcast(
                cams_tform4x4_obj=cams_tform4x4_obj,
                cams_intr4x4=cams_intr4x4,
                objects_ids=objects_ids,
            )

            cams_times_objects_count = cams_tform4x4_obj.shape[0]
            cams_count = int(cams_times_objects_count / objects_count)
        else:
            objects_count = objects_ids.shape[0]
            if cams_tform4x4_obj is not None:
                cams_count = cams_tform4x4_obj.shape[0]
            else:
                cams_count = objects_count

            if objects_count != cams_count:
                raise ValueError(
                    f"Set `broadcast_batch_and_cams=True` to allow different number of cameras and objects",
                )

        mods1d_rendered = self.sample_batch(
            cams_tform4x4_obj=cams_tform4x4_obj,
            cams_intr4x4=cams_intr4x4,
            imgs_sizes=imgs_sizes,
            objects_ids=objects_ids,
            modalities=_modalities,
            add_clutter=add_clutter,
            add_other_objects=add_other_objects,
            device=device,
            dtype=dtype,
            sample_other_objects=sample_other_objects,
            sample_clutter=sample_clutter,
        )

        if broadcast_batch_and_cams:
            for key, val in mods1d_rendered.items():
                mods1d_rendered[key] = val.reshape(
                    objects_count,
                    cams_count,
                    *val.shape[1:],
                )

        if not isinstance(modalities, List):
            return mods1d_rendered[_modalities[0]]
        else:
            return mods1d_rendered

    def sample_batch(
        self,
        cams_tform4x4_obj,
        cams_intr4x4,
        imgs_sizes,
        objects_ids=None,
        modalities: Union[
            PROJECT_MODALITIES, List[PROJECT_MODALITIES]
        ] = PROJECT_MODALITIES.FEATS,
        add_clutter=False,
        add_other_objects=False,
        device=None,
        dtype=None,
        sample_clutter=False,
        sample_other_objects=False,
    ):
        """
        Sample the objects' projection.
        Args:
            cams_tform4x4_obj: (B, 4, 4) tensor of camera poses in the object frame.
            cams_intr4x4: (B, 4, 4) tensor of camera intrinsics.
            imgs_sizes: (2,) tensor of (H, W)
            objects_ids: (B, ) tensor of objects ids to render.
            modalities: (List(PROJECT_MODALITIES)) the modalities to render.
            broadcast_batch_and_cams: bool, whether to broadcast the batch and camera dimensions.
            down_sample_rate: float, the downsample rate for the rendered image.
            add_clutter: bool, determines wether to add clutter features to the sampled features.
            add_other_objects: bool, determines wether to add other objects' features to the sampled features.

        Returns:
            mods1d_sampled (Dict[PROJECT_MODALITIES, torch.Tensor]): (B, V, F) dict of projected modalities.
        """
        raise NotImplementedError

    # def get_sim_project_sample(self, ):

    def get_sim_render(
        self,
        feats2d_img,
        cams_tform4x4_obj,
        cams_intr4x4,
        objects_ids=None,
        broadcast_batch_and_cams=False,
        down_sample_rate=1.0,
        feats2d_img_mask=None,
        allow_clutter=True,
        return_sim_pxl=False,
        add_clutter=False,
        add_other_objects=False,
        temp=1.0,
    ):
        """
        Args:
            feats2d_img (torch.Tensor): BxCxHxW
            feats2d_img_mask (torch.Tensor): Bx1xHxW
            cams_tform4x4_obj: (B, C, 4, 4) or (B, 4, 4) tensor of camera poses in the object frame.
            cams_intr4x4: (4, 4), or (B, 4, 4) or (B, C, 4, 4) tensor of camera intrinsics.
            imgs_sizes: (2,) tensor of (H, W)
            objects_ids: (B, ) tensor of objects ids to render.
            modalities: (Union[str, List(str)]) the modalities to render.
            broadcast_batch_and_cams: bool, whether to broadcast the batch and camera dimensions.
            down_sample_rate: float, the downsample rate for the rendered image.
            return_sim_pxl (bool): Indicates whether pixelwise similarity should be returned or not.
            add_clutter: bool, determines wether to add clutter features to the sampled features.
            add_other_objects: bool, determines wether to add other objects' features to the sampled features.

        Returns:
            sim (torch.Tensor): Bx(C)
            sim_feats2d (torch.Tensor, optional): Bx(C)xHxW
        """
        imgs_sizes = torch.LongTensor(list(feats2d_img.shape[-2:])) * down_sample_rate
        feats2d_rendered = self.render(
            modalities=PROJECT_MODALITIES.FEATS,
            cams_tform4x4_obj=cams_tform4x4_obj,
            cams_intr4x4=cams_intr4x4,
            imgs_sizes=imgs_sizes,
            objects_ids=objects_ids,
            broadcast_batch_and_cams=broadcast_batch_and_cams,
            down_sample_rate=down_sample_rate,
            add_clutter=add_clutter,
            add_other_objects=add_other_objects,
        )

        return self.get_sim_feats2d_img_and_rendered(
            feats2d_img,
            feats2d_rendered,
            return_sim_pxl=return_sim_pxl,
            feats2d_img_mask=feats2d_img_mask,
            allow_clutter=allow_clutter,
            temp=temp,
        )

    def sample_nearest_to_feats2d_img(
        self,
        feats2d_img,
        cams_tform4x4_obj=None,
        cams_intr4x4=None,
        imgs_sizes=None,
        objects_ids=None,
        modalities: Union[
            PROJECT_MODALITIES, List[PROJECT_MODALITIES]
        ] = PROJECT_MODALITIES.FEATS,
        broadcast_batch_and_cams=False,
        down_sample_rate=1.0,
        add_clutter=False,
        add_other_objects=False,
        dtype=None,
        device=None,
        smooth_labels=False,
        sim_temp=1.0,
    ):
        """
        Args:
            feats2d_img (torch.Tensor): BxFxHxW
            feats1d_obj (torch.Tensor): BxVxF
        Returns:
            sim_feats (torch.Tensor): (B, V(+1), V+N) or (B, V(+1), H, W) if dense=True
        """

        sim_feats2d = self.get_sim_feats2d_img_to_all(
            feats2d_img=feats2d_img,
            imgs_sizes=imgs_sizes,
            cams_tform4x4_obj=cams_tform4x4_obj,
            cams_intr4x4=cams_intr4x4,
            objects_ids=objects_ids,
            broadcast_batch_and_cams=broadcast_batch_and_cams,
            down_sample_rate=down_sample_rate,
            add_clutter=add_clutter,
            add_other_objects=add_other_objects,
            dense=True,
            sim_temp=sim_temp,
            clutter_pxl2d=None,
        )

        label_feats2d_nearest = sim_feats2d.argmax(
            dim=1, keepdim=True
        )  # (B, V(+1), V+N) or (B, V(+1), H, W) if dense=True

        nearest_mods2d = self.sample(
            modalities=modalities,
            cams_tform4x4_obj=cams_tform4x4_obj,
            cams_intr4x4=cams_intr4x4,
            imgs_sizes=imgs_sizes,
            objects_ids=objects_ids,
            broadcast_batch_and_cams=broadcast_batch_and_cams,
            down_sample_rate=down_sample_rate,
            add_clutter=add_clutter,
            add_other_objects=add_other_objects,
            sample_clutter=add_clutter,
            sample_other_objects=add_other_objects,
        )  # BxV(+1)x3

        from od3d.cv.select import batched_index_select

        B, _, H, W = label_feats2d_nearest.shape
        nearest_mods2d = (
            batched_index_select(
                input=nearest_mods2d, index=label_feats2d_nearest.flatten(1), dim=1
            )
            .permute(0, 2, 1)
            .reshape(B, 3, H, W)
        )  # Bx3xHxW
        return nearest_mods2d

    def get_sim_feats2d_img_to_all(
        self,
        feats2d_img,
        imgs_sizes,
        cams_tform4x4_obj=None,
        cams_intr4x4=None,
        objects_ids=None,
        broadcast_batch_and_cams=False,
        down_sample_rate=1.0,
        add_clutter=True,
        add_other_objects=True,
        dense=False,
        sim_temp=1.0,
        clutter_pxl2d=None,
        return_feats=False,
    ):
        """
        Args:
            feats2d_img (torch.Tensor): BxFxHxW
            feats1d_obj (torch.Tensor): BxVxF
        Returns:
            sim_feats (torch.Tensor): (B, V(+1), V+N) or (B, V(+1), H, W) if dense=True
        """

        if dense:
            feats1d_sampled = self.sample(
                modalities=PROJECT_MODALITIES.FEATS,
                cams_tform4x4_obj=cams_tform4x4_obj,
                cams_intr4x4=cams_intr4x4,
                imgs_sizes=imgs_sizes,
                objects_ids=objects_ids,
                broadcast_batch_and_cams=broadcast_batch_and_cams,
                down_sample_rate=down_sample_rate,
                add_clutter=add_clutter,
                add_other_objects=add_other_objects,
            )
            sim_feats2d = self.get_sim_feats2d_img_and_feats1d_obj(
                feats2d_img,
                feats1d_sampled,
                add_clutter=False,
                temp=sim_temp,
            )
            if return_feats:
                return sim_feats2d, feats2d_img
            else:
                return sim_feats2d

        else:
            feat_modality = PROJECT_MODALITIES.FEATS

            mods1d_sampled = self.sample_with_img2d(
                img2d=feats2d_img,
                modalities=[feat_modality, PROJECT_MODALITIES.IMG],
                cams_tform4x4_obj=cams_tform4x4_obj,
                cams_intr4x4=cams_intr4x4,
                imgs_sizes=imgs_sizes,
                objects_ids=objects_ids,
                broadcast_batch_and_cams=broadcast_batch_and_cams,
                down_sample_rate=down_sample_rate,
                add_clutter=add_clutter,
                add_other_objects=add_other_objects,
                clutter_pxl2d=clutter_pxl2d,
                dtype=feats2d_img.dtype,
                device=feats2d_img.device,
            )

            sim_feats1d = self.get_sim_feats1d_img_and_feats1d_obj(
                mods1d_sampled[PROJECT_MODALITIES.IMG],
                mods1d_sampled[feat_modality],
                add_clutter=False,
                temp=sim_temp,
            )
            # sim_feats1d[feats1d_obj_mask[:, None,]] = -1

            if return_feats:
                return sim_feats1d, mods1d_sampled[PROJECT_MODALITIES.IMG]
            else:
                return sim_feats1d

    def get_label_feats2d_img(
        self,
        feats2d_img,
        imgs_sizes,
        cams_tform4x4_obj,
        cams_intr4x4,
        objects_ids=None,
        broadcast_batch_and_cams=False,
        down_sample_rate=1.0,
        feats2d_img_mask=None,
        add_clutter=True,
        add_other_objects=True,
        sample_clutter_count=5,
        dense=False,
        smooth_labels=False,
    ):
        """
        Args:
            feats2d_img (torch.Tensor): BxFxHxW
            feats1d_obj (torch.Tensor): BxVxF
        Returns:
            label (torch.Tensor): (B, V+N) or (B, V(+1), H, W) if dense=True or (B, V(+1),V+N) if smooth_labels=True
            label_mask (torch.Tensor): (B, V+N) or (B, V(+1), H, W) if dense=True
            noise_pxl2d (torch.Tensor): (B, V, 2), or None if dense=True
        """

        if dense:
            if smooth_labels:
                label_modality = PROJECT_MODALITIES.ONEHOT_SMOOTH
            else:
                label_modality = PROJECT_MODALITIES.ONEHOT

            label2d = self.render(
                modalities=label_modality,
                cams_tform4x4_obj=cams_tform4x4_obj,
                cams_intr4x4=cams_intr4x4,
                imgs_sizes=imgs_sizes,
                objects_ids=objects_ids,
                broadcast_batch_and_cams=broadcast_batch_and_cams,
                down_sample_rate=down_sample_rate,
                add_clutter=add_clutter,
                add_other_objects=add_other_objects,
            )
            return label2d, None, None

        else:
            mask_modality = PROJECT_MODALITIES.MASK
            if smooth_labels:
                label_modality = PROJECT_MODALITIES.ONEHOT_SMOOTH
            else:
                label_modality = PROJECT_MODALITIES.ID

            mods1d_sampled = self.sample_with_img2d(
                img2d=feats2d_img,
                img2d_mask=feats2d_img_mask,
                modalities=[
                    mask_modality,
                    label_modality,
                    PROJECT_MODALITIES.CLUTTER_PXL2D,
                ],
                cams_tform4x4_obj=cams_tform4x4_obj,
                cams_intr4x4=cams_intr4x4,
                imgs_sizes=imgs_sizes,
                objects_ids=objects_ids,
                broadcast_batch_and_cams=broadcast_batch_and_cams,
                down_sample_rate=down_sample_rate,
                add_clutter=add_clutter,
                add_other_objects=add_other_objects,
                sample_clutter_count=sample_clutter_count,
            )

            return (
                mods1d_sampled[label_modality],
                mods1d_sampled[mask_modality],
                mods1d_sampled[PROJECT_MODALITIES.CLUTTER_PXL2D],
            )

    def get_clutter_label(self, add_other_objects=False, one_hot=False, device=None):
        raise NotImplementedError

    def get_label_and_sim_feats2d_img_to_all(
        self,
        feats2d_img,
        imgs_sizes,
        cams_tform4x4_obj,
        cams_intr4x4,
        objects_ids=None,
        broadcast_batch_and_cams=False,
        down_sample_rate=1.0,
        feats2d_img_mask=None,
        add_clutter=True,
        add_other_objects=True,
        sample_clutter_count=5,
        dense=False,
        smooth_labels=False,
        sim_temp=1.0,
        return_feats=False,
    ):
        """
        Args:
            feats2d_img (torch.Tensor): BxFxHxW
            feats1d_obj (torch.Tensor): BxVxF
        Returns:
            label (torch.Tensor): (B, V+N) or (B, V(+1), H, W) if dense=True or (B, V(+1),V+N) if smooth_labels=True
            label_mask (torch.Tensor): (B, V+N) or (B, V(+1), H, W) if dense=True
            noise_pxl2d (torch.Tensor): (B, V, 2)
            sim_feats (torch.Tensor): (B, V(+1), V+N) or (B, V(+1), H, W) if dense=True

        """
        label_feats, label_feats_mask, feat_clutter_pxl2d = self.get_label_feats2d_img(
            feats2d_img=feats2d_img,
            imgs_sizes=imgs_sizes,
            cams_tform4x4_obj=cams_tform4x4_obj,
            cams_intr4x4=cams_intr4x4,
            objects_ids=objects_ids,
            broadcast_batch_and_cams=broadcast_batch_and_cams,
            down_sample_rate=down_sample_rate,
            feats2d_img_mask=feats2d_img_mask,
            add_clutter=add_clutter,
            add_other_objects=add_other_objects,
            sample_clutter_count=sample_clutter_count,
            dense=dense,
            smooth_labels=smooth_labels,
        )

        sim_feats = self.get_sim_feats2d_img_to_all(
            feats2d_img=feats2d_img,
            imgs_sizes=imgs_sizes,
            cams_tform4x4_obj=cams_tform4x4_obj,
            cams_intr4x4=cams_intr4x4,
            objects_ids=objects_ids,
            broadcast_batch_and_cams=broadcast_batch_and_cams,
            down_sample_rate=down_sample_rate,
            add_clutter=add_clutter,
            add_other_objects=add_other_objects,
            dense=dense,
            sim_temp=sim_temp,
            clutter_pxl2d=feat_clutter_pxl2d,
            return_feats=return_feats,
        )

        if return_feats:
            return (
                label_feats,
                label_feats_mask,
                feat_clutter_pxl2d,
                sim_feats[0],
                sim_feats[1],
            )
        else:
            return label_feats, label_feats_mask, feat_clutter_pxl2d, sim_feats

    def update_feats_moving_average(
        self,
        labels,
        labels_mask,
        feats,
        alpha,
        objects_ids=None,
        add_clutter=True,
        add_other_objects=True,
    ):
        """
        Args:
            labels (torch.Tensor): BxN (or BxVxN)
            labels_mask (torch.Tensor): BxN
            feats (torch.Tensor): BxNxF
            alpha (float): the moving average factor.
        """
        raise NotImplementedError

    def update_feats_total_average(
        self,
        labels,
        labels_mask,
        feats,
        objects_ids=None,
        add_clutter=True,
        add_other_objects=True,
    ):
        """
        Args:
            labels (torch.Tensor): BxN (or BxVxN)
            labels_mask (torch.Tensor): BxN
            feats (torch.Tensor): BxNxF
            alpha (float): the moving average factor.
        """
        raise NotImplementedError

    def get_sim_cams_verts(
        self,
        categories_ids,
        feats2d_net,
        verts2d_mesh,
        verts2d_mesh_mask,
        return_sim_pxl=False,
        feats2d_net_mask=None,
    ):
        pass

    def render_with_clutter(self):
        pass

    def get_sim_feats1d_img_and_feats1d_obj(
        self, feats1d_img, feats1d_obj, add_clutter=False, temp=1.0
    ):
        """
        Args:
            feats1d_img (torch.Tensor): BxNxC
            feats1d_obj (torch.Tensor): BxVxC

        Returns:
            sim_feats1d (torch.Tensor): BxVxN or BxV+1xN if add_clutter=True
        """
        if add_clutter:
            feats1d_obj = torch.cat([feats1d_obj, self.feat_clutter[None, None]], dim=1)

        sim_feats1d = self.get_sim(
            "bnc,bvc->bvn",
            feats1d_img,
            feats1d_obj,
            temp=temp,
        )
        return sim_feats1d

    def get_sim_feats2d_img_and_feats1d_obj(
        self, feats2d_img, feats1d_obj, add_clutter=False, temp=1.0
    ):
        """
        Args:
            feats2d_img (torch.Tensor): BxCxHxW
            feats1d_obj (torch.Tensor): BxVxC

        Returns:
            sim_feats2d (torch.Tensor): BxVxHxW or BxV+1xHxW if add_clutter=True
        """
        if add_clutter:
            feats1d_obj = torch.cat([feats1d_obj, self.feat_clutter[None, None]], dim=1)

        sim_feats2d = self.get_sim(
            "bchw,bvc->bvhw",
            feats2d_img,
            feats1d_obj,
            temp=temp,
        )
        return sim_feats2d

    def get_sim_feats2d_img_and_rendered(
        self,
        feats2d_img,
        feats2d_rendered,
        return_sim_pxl=False,
        feats2d_img_mask=None,
        allow_clutter=True,
        temp=1.0,
    ):
        """
        Args:
            feats2d_img (torch.Tensor): BxCxHxW
            feats2d_rendered (torch.Tensor): Bx(T)xCxHxW
            return_sim_pxl (bool): Indicates whether pixelwise similarity should be returned or not.
            feats2d_img_mask (torch.Tensor): Bx1xHxW

        Returns:
            sim (torch.Tensor): Bx(T)
            sim_feats2d (torch.Tensor, optional): Bx(T)xHxW
        """
        if feats2d_rendered.dim() == 5:
            sim_feats2d = self.get_sim(
                "bchw,bvchw->bvhw",
                feats2d_img,
                feats2d_rendered,
                temp=temp,
            )
        else:
            sim_feats2d = self.get_sim(
                "bchw,bchw->bhw",
                feats2d_img,
                feats2d_rendered,
                temp=temp,
            )[:, None]

        # shape (B, V, H, W)
        sim_clutter2d = self.get_sim(
            "bchw,c->bhw",
            feats2d_img,
            self.feat_clutter,
            temp=temp,
        )[:, None].expand(sim_feats2d.shape)

        if feats2d_img_mask is not None:
            clutter_mask = (feats2d_img_mask < 0.5).expand(sim_feats2d.shape)
        else:
            clutter_mask = torch.zeros_like(sim_feats2d, dtype=torch.bool)

        if allow_clutter:
            clutter_mask = clutter_mask | (sim_feats2d < sim_clutter2d)

        sim_feats2d[clutter_mask] = sim_clutter2d[clutter_mask]

        sim = sim_feats2d.flatten(2).mean(dim=-1)

        if feats2d_rendered.dim() == 4:
            sim_feats2d = sim_feats2d.squeeze(1)
            sim = sim.squeeze(1)

        if return_sim_pxl:
            return sim, sim_feats2d
        else:
            return sim

    def get_sim(self, comb, featsA, featsB, temp=1.0):
        """
        Args:
            comb (str): e.g. hwf,hwf->hw
            featsA (torch.Tensor): e.g. shape (H, W, F)
            featsB (torch.Tensor): e.g. shape (H, W, F)

        Returns:
            sim (torch.Tensor): e.g. shape (H, W)
        """
        if self.feats_distribution == FEATS_DISTR.VON_MISES_FISHER:
            return torch.einsum(comb, featsA, featsB) / temp
        elif self.feats_distribution == FEATS_DISTR.GAUSSIAN:
            from od3d.cv.geometry.dist import einsum_cdist

            return -einsum_cdist(comb, featsA, featsB) / temp
        else:
            msg = f"Unknown distribution {self.feats_distribution}"
            raise NotImplementedError(msg)
