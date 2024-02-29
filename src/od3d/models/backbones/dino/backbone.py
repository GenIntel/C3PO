import logging
logger = logging.getLogger(__name__)
from torch import nn
from omegaconf import DictConfig
import torch
from od3d.cv.transforms.sequential import SequentialTransform
from od3d.cv.transforms.rgb_uint8_to_float import RGB_UInt8ToFloat
from od3d.cv.transforms.rgb_normalize import RGB_Normalize
import torchvision
from od3d.models.backbones.backbone import OD3D_Backbone
from od3d.data.ext_enum import ExtEnum
from od3d.cv.visual.resize import resize
from typing import Tuple
import math
import types

from od3d.models.backbones.dino.dinov1 import ViTExtractor # for selecting keys, querys, values

class DINOv2_WEIGHTS(str, ExtEnum):
    DEFAULT = 'default'
    NONE = 'none'

class DINOv2(OD3D_Backbone):
    def __init__(
            self,
            config: DictConfig
    ):

        super().__init__(config=config)

        self.transform = SequentialTransform([
            RGB_UInt8ToFloat(),
            RGB_Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.layers_returned = config.layers_returned # choose from [1, 2, 3, 4]
        self.layers_count = len(self.layers_returned)

        # dino_vits8, dino_vitb8, dino_vits16, dino_vitb16, dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
        self.dinov2 = 'dinov2' in self.config.hub_model
        self.stride = self.config.get('stride', 14)
        if self.dinov2:
            self.extractor = torch.hub.load(self.config.hub_repo, self.config.hub_model,
                                        pretrained=self.config.weights == 'default')
            self.extractor = self.patch_vit_resolution(self.extractor, self.stride)
            self.out_dims = [self.extractor.embed_dim]
        else: # using keys did not show any improvement
            self.extractor = ViTExtractor(model_type=self.config.hub_model)
            self.out_dims = [self.extractor.model.embed_dim]

        self.out_downsample_scales = []
        self.downsample_rate = self.config.downsample_rate
        import re
        match = re.match(r"dino[v2]*_vit[a-z]*([0-9]+)", self.config.hub_model, re.I)
        if match and len(match.groups()) == 1:
            self.dino_patch_size = int(match.groups()[0])
            self.downsample_rate_dino = int(match.groups()[0]) // (int(match.groups()[0]) // self.stride)
        else:
            msg = f'could not retrieve down sample rate dino from model name {self.config.hub_model}'
            raise Exception(msg)

        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False

            # if not self.dinov2:
            #     for param in self.extractor.model.parameters():
            #         param.requires_grad = False

    def forward(self, x):
        if x.dim() == 3:
            C, H, W = x.shape
        elif x.dim() == 4:
            B, C, H, W = x.shape
        else:
            raise NotImplementedError

        H_out = (H // self.downsample_rate)
        W_out = (W // self.downsample_rate) 
        H_out_expected =(((H_out * self.downsample_rate_dino) - self.dino_patch_size)// self.stride)+1
        W_out_expected =(((W_out * self.downsample_rate_dino) - self.dino_patch_size)// self.stride)+1
        offset_H = H_out - H_out_expected
        offset_H = self.round_up_to_even(offset_H) 
        
        offset_W = W_out - W_out_expected
        offset_W = self.round_up_to_even(offset_W)
        H_in = H_out * self.downsample_rate_dino + offset_H * self.downsample_rate_dino
        W_in = W_out * self.downsample_rate_dino + offset_W * self.downsample_rate_dino

        x = resize(x, H_out= H_in, W_out=W_in)
        
        if self.dinov2:
            x = self.extractor.forward_features(x)["x_norm_patchtokens"]  # # 'x_norm_patchtokens', 'x_prenorm'
        
        else:
            #x = self.extractor.get_intermediate_layers(x, n=12)[9]  # maximum 12 layers, zsp uses 9
            #x = x[:, 1:] # remove cls token

            x = self.extractor.extract_descriptors(batch=x, layer=9, facet='key', bin=False, include_cls=False)
            # note: key layer 9 outperforms layer 9

        x = x.reshape(-1, H_out_expected +offset_H , W_out_expected+offset_W, self.out_dims[-1]).permute(0, 3, 1, 2)
        x = x[:, :, :H_out, :W_out]
        x_layers = [x]
        return x_layers
    
    @staticmethod
    def round_up_to_even(num: int) -> int:
        return num if num % 2 == 0 else num + 1
    
    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):

        def interpolate_pos_encoding(self, x, w, h):
            previous_dtype = x.dtype
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            pos_embed = self.pos_embed.float()
            class_pos_embed = pos_embed[:, 0]
            patch_pos_embed = pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            M = int(math.sqrt(N))  # Recover the number of patches in each dimension
            assert N == M * M
            kwargs = {}
            if self.interpolate_offset:
                # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
                # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
                sx = float(w0 + self.interpolate_offset) / M
                sy = float(h0 + self.interpolate_offset) / M
                kwargs["scale_factor"] = (sx, sy)
            else:
                # Simply specify an output size instead of a scale factor
                kwargs["size"] = (w0, h0)
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
                mode="bicubic",
                antialias=self.interpolate_antialias,
                **kwargs,
            )
            assert (w0, h0) == patch_pos_embed.shape[-2:]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)
        
        return interpolate_pos_encoding
    
    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        patch_size = model.patch_embed.patch_size[0]
        if stride == patch_size:  # nothing to do
            return model

        stride = nn.modules.utils._pair(stride)
        assert all([(patch_size // s_) * s_ == patch_size for s_ in
                    stride]), f'stride {stride} should divide patch_size {patch_size}'

        # fix the stride
        
        model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(ViTExtractor._fix_pos_enc(patch_size, stride), model)
        return model