from od3d.methods.method import OD3D_Method
from od3d.datasets.dataset import OD3D_Dataset
from od3d.benchmark.results import OD3D_Results
from od3d.models.model import OD3D_Model
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import warnings
import logging

logger = logging.getLogger(__name__)
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from pathlib import Path

# note: math is actually used by config
import math

from typing import Dict
# import sys
# sys.path.append(str(Path(__file__).parents[4] / 'third_party/LightGlue'))
from od3d.models.feature_extractors import Extractor, rbd
from od3d.cv.utils.dnnlib import construct_class_by_name
from .utils import filter_matches, pad_to_length, normalize_keypoints, TokenConfidence, MatchAssignment, LearnableFourierPositionalEncoding, TransformerLayer
import torch.nn.functional as F
from torch import nn

try:
    from flash_attn.modules.mha import FlashCrossAttention
except ModuleNotFoundError:
    FlashCrossAttention = None

if FlashCrossAttention or hasattr(F, "scaled_dot_product_attention"):
    FLASH_AVAILABLE = True
else:
    FLASH_AVAILABLE = False

torch.backends.cudnn.deterministic = True

class FeatureExtractor(nn.Module):
    def __init__(self, config: DictConfig):
        super(FeatureExtractor, self).__init__()
        self.config = config
        self.net: Extractor = construct_class_by_name(**self.config.features.backbone)

    def forward(self, x: torch.Tensor):
        """Forward pass of the feature extractor.

        Args:
            x (torch.Tensor): input image with shape (B, C, H, W). B = 1 supported only

        Returns:
            Dict: dictionary with the following keys:
                - keypoints (torch.Tensor): detected keypoints with shape (B, K, 2)
                - keypoint_scores (torch.Tensor): keypoint scores with shape (B, K)
                - descriptors (torch.Tensor): descriptors with shape (B, K, D)
        """
        return self.net.extract(x, **self.config.features.preprocess)

class Matcher(nn.Module):
    def __init__(self, config: DictConfig, device="cpu"):
        super(Matcher, self).__init__()
        self.config = config
        self.device = device
        h, n, d = self.config.lightglue.num_heads, self.config.lightglue.n_layers, self.config.lightglue.descriptor_dim
        if self.config.features.backbone.output_dim != self.config.lightglue.descriptor_dim:
            self.input_proj = nn.Linear(
                self.config.features.backbone.output_dim,
                d,
                bias=True
            )
        else:
            self.input_proj = nn.Identity()
        head_dim = d // h
        self.posenc = LearnableFourierPositionalEncoding(2 + 2 * self.config.features.backbone.add_scale_ori, head_dim, head_dim)
        self.transformers = nn.ModuleList([TransformerLayer(d, h, self.config.lightglue.flash) for _ in range(n)])
        self.output_proj = nn.Linear(d, 1, bias=True)
        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = nn.ModuleList([TokenConfidence(d) for _ in range(n - 1)])
        self.register_buffer("confidence_thresholds", torch.Tensor([self.confidence_threshold(i) for i in range(self.config.lightglue.n_layers)]))

        # static lengths LightGlue is compiled for (only used with torch.compile)
        self.static_lengths = None
        self.to(device)
        
    def forward(self, data: Dict):
        with torch.autocast(enabled=self.config.lightglue.mp, device_type="cuda"):
            return self._compute_matches(data)

    def load_checkpoint(self, path_checkpoint: Path = None):
        state_dict = None
        if path_checkpoint is not None:
            state_dict = torch.load(path_checkpoint, map_location=self.device)
        if self.config.features is not None:
            fname = f"{self.config.features.backbone.weights}_{self.config.lightglue.version.replace('.', '-')}.pth"
            state_dict = torch.hub.load_state_dict_from_url(
                self.config.lightglue.url.format(
                    self.config.lightglue.version,
                    self.config.features.backbone.name
                ),
                file_name=fname,
                map_location=self.device
            )
        if state_dict:
            # rename old state dict entries
            for i in range(self.config.lightglue.n_layers):
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            incompatible_keys = self.load_state_dict(state_dict, strict=False)
            if incompatible_keys.missing_keys:
                logger.warning(f"Missing keys: {incompatible_keys.missing_keys}")
            if incompatible_keys.unexpected_keys:
                logger.warning(f"Unexpected keys: {incompatible_keys.unexpected_keys}")
        else:
            logger.warning("No checkpoint found.")
     
    def compile(self, static_lengths, mode="reduce-overhead"):
        if self.config.lightglue.width_confidence != -1:
            warnings.warn(
                "Point pruning is partially disabled for compiled forward.",
                stacklevel=2,
            )

        for i in range(self.config.lightglue.n_layers):
            self.transformers[i].masked_forward = torch.compile(
                self.transformers[i].masked_forward, mode=mode, fullgraph=True
            )
        self.static_lengths = static_lengths

    def confidence_threshold(self, layer_index: int) -> float:
        """scaled confidence threshold"""
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.config.lightglue.n_layers)
        return np.clip(threshold, 0, 1)

    def get_pruning_mask(
        self, confidences: torch.Tensor, scores: torch.Tensor, layer_index: int
    ) -> torch.Tensor:
        """mask points which should be removed"""
        keep = scores > (1 - self.config.lightglue.width_confidence)
        if confidences is not None:  # Low-confidence points are never pruned.
            keep |= confidences <= self.confidence_thresholds[layer_index]
        return keep

    def check_if_stop(
        self,
        confidences0: torch.Tensor,
        confidences1: torch.Tensor,
        layer_index: int,
        num_points: int,
    ) -> torch.Tensor:
        """evaluate stopping condition"""
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = 1.0 - (confidences < threshold).float().sum() / num_points
        return ratio_confident > self.config.lightglue.depth_confidence

    def pruning_min_kpts(self, device: torch.device):
        if self.config.lightglue.flash and FLASH_AVAILABLE and device.type == "cuda":
            return self.config.lightglue.pruning_keypoint_thresholds.flash
        else:
            return self.config.lightglue.pruning_keypoint_thresholds[device.type]

    def _compute_matches(self, data: Dict[str, torch.Tensor]):
        data0, data1 = data["image0"], data["image1"]
        kpts0, kpts1 = data0["keypoints"], data1["keypoints"]
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape
        device = kpts0.device
        size0, size1 = data0.get("image_size"), data1.get("image_size")
        kpts0 = normalize_keypoints(kpts0, size0).clone()
        kpts1 = normalize_keypoints(kpts1, size1).clone()

        if self.config.features.backbone.add_scale_ori:
            kpts0 = torch.cat(
                [kpts0] + [data0[k].unsqueeze(-1) for k in ("scales", "oris")], -1
            )
            kpts1 = torch.cat(
                [kpts1] + [data1[k].unsqueeze(-1) for k in ("scales", "oris")], -1
            )
        desc0 = data0["descriptors"].detach().contiguous()
        desc1 = data1["descriptors"].detach().contiguous()

        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()

        mask0, mask1 = None, None
        c = max(m, n)
        do_compile = self.static_lengths and c <= max(self.static_lengths)
        if do_compile:
            kn = min([k for k in self.static_lengths if k >= c])
            desc0, mask0 = pad_to_length(desc0, kn)
            desc1, mask1 = pad_to_length(desc1, kn)
            kpts0, _ = pad_to_length(kpts0, kn)
            kpts1, _ = pad_to_length(kpts1, kn)
        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        # cache positional embeddings
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        # GNN + final_proj + assignment
        do_early_stop = self.config.lightglue.depth_confidence > 0
        do_point_pruning = self.config.lightglue.width_confidence > 0 and not do_compile
        pruning_th = self.pruning_min_kpts(device)
        if do_point_pruning:
            ind0 = torch.arange(0, m, device=device)[None]
            ind1 = torch.arange(0, n, device=device)[None]
            # We store the index of the layer at which pruning is detected.
            prune0 = torch.ones_like(ind0)
            prune1 = torch.ones_like(ind1)
        token0, token1 = None, None
        for i in range(self.config.lightglue.n_layers):
            if desc0.shape[1] == 0 or desc1.shape[1] == 0:  # no keypoints
                break
            desc0, desc1 = self.transformers[i](
                desc0, desc1, encoding0, encoding1, mask0=mask0, mask1=mask1
            )
            if i == self.config.lightglue.n_layers - 1:
                continue  # no early stopping or adaptive width at last layer

            if do_early_stop:
                token0, token1 = self.token_confidence[i](desc0, desc1)
                if self.check_if_stop(token0[..., :m], token1[..., :n], i, m + n):
                    break
            if do_point_pruning and desc0.shape[-2] > pruning_th:
                scores0 = self.log_assignment[i].get_matchability(desc0)
                prunemask0 = self.get_pruning_mask(token0, scores0, i)
                keep0 = torch.where(prunemask0)[1]
                ind0 = ind0.index_select(1, keep0)
                desc0 = desc0.index_select(1, keep0)
                encoding0 = encoding0.index_select(-2, keep0)
                prune0[:, ind0] += 1
            if do_point_pruning and desc1.shape[-2] > pruning_th:
                scores1 = self.log_assignment[i].get_matchability(desc1)
                prunemask1 = self.get_pruning_mask(token1, scores1, i)
                keep1 = torch.where(prunemask1)[1]
                ind1 = ind1.index_select(1, keep1)
                desc1 = desc1.index_select(1, keep1)
                encoding1 = encoding1.index_select(-2, keep1)
                prune1[:, ind1] += 1

        if desc0.shape[1] == 0 or desc1.shape[1] == 0:  # no keypoints
            m0 = desc0.new_full((b, m), -1, dtype=torch.long)
            m1 = desc1.new_full((b, n), -1, dtype=torch.long)
            mscores0 = desc0.new_zeros((b, m))
            mscores1 = desc1.new_zeros((b, n))
            matches = desc0.new_empty((b, 0, 2), dtype=torch.long)
            mscores = desc0.new_empty((b, 0))
            if not do_point_pruning:
                prune0 = torch.ones_like(mscores0) * self.config.lightglue.n_layers
                prune1 = torch.ones_like(mscores1) * self.config.lightglue.n_layers
            return {
                "matches0": m0,
                "matches1": m1,
                "matching_scores0": mscores0,
                "matching_scores1": mscores1,
                "stop": i + 1,
                "matches": matches,
                "scores": mscores,
                "prune0": prune0,
                "prune1": prune1,
            }

        desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]  # remove padding
        scores, _ = self.log_assignment[i](desc0, desc1)
        m0, m1, mscores0, mscores1 = filter_matches(scores, self.config.lightglue.filter_threshold)
        matches, mscores = [], []
        for k in range(b):
            valid = m0[k] > -1
            m_indices_0 = torch.where(valid)[0]
            m_indices_1 = m0[k][valid]
            if do_point_pruning:
                m_indices_0 = ind0[k, m_indices_0]
                m_indices_1 = ind1[k, m_indices_1]
            matches.append(torch.stack([m_indices_0, m_indices_1], -1))
            mscores.append(mscores0[k][valid])

        # TODO: Remove when hloc switches to the compact format.
        if do_point_pruning:
            m0_ = torch.full((b, m), -1, device=m0.device, dtype=m0.dtype)
            m1_ = torch.full((b, n), -1, device=m1.device, dtype=m1.dtype)
            m0_[:, ind0] = torch.where(m0 == -1, -1, ind1.gather(1, m0.clamp(min=0)))
            m1_[:, ind1] = torch.where(m1 == -1, -1, ind0.gather(1, m1.clamp(min=0)))
            mscores0_ = torch.zeros((b, m), device=mscores0.device)
            mscores1_ = torch.zeros((b, n), device=mscores1.device)
            mscores0_[:, ind0] = mscores0
            mscores1_[:, ind1] = mscores1
            m0, m1, mscores0, mscores1 = m0_, m1_, mscores0_, mscores1_
        else:
            prune0 = torch.ones_like(mscores0) * self.config.lightglue.n_layers
            prune1 = torch.ones_like(mscores1) * self.config.lightglue.n_layers

        return {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "stop": i + 1,
            "matches": matches,
            "scores": mscores,
            "prune0": prune0,
            "prune1": prune1,
        }

class TransformerMatching(OD3D_Method):
    def __init__(
            self,
            config: DictConfig,
            logging_dir,
            device="cpu"
    ):
        super().__init__(config=config, logging_dir=logging_dir)
        self.extractor = FeatureExtractor(config)
        self.matcher = Matcher(config)
        self.to(device)


    def save_checkpoint(self, path_checkpoint: Path):
        pass

    def load_checkpoint(self, path_checkpoint: Path = None):
        self.matcher.load_checkpoint(path_checkpoint)

    def to(self, device):
        self.extractor.to(device)
        self.matcher.to(device)
        self.device = device

    def cuda(self):
        self.to("cuda")

    def compile(self, mode="reduce-overhead", static_lengths=[256, 512, 768, 1024, 1280, 1536]):
        self.extractor = torch.compile(self.extractor, mode=mode, fullgraph=True)
        self.matcher.compile(mode=mode, static_lengths=static_lengths)

    def match_pair(self, image0: torch.Tensor, image1: torch.Tensor):
        """
        Match keypoints and descriptors between two images

        Input (dict):
            image0 (torch.Tensor): [B x C x H x W]
            image1 (torch.Tensor): [B x C x H x W]
        Output (tuple):
            feats0 (dict):
                keypoints: [B x M x 2]
                keypoint_scores: [B x M]
                descriptors: [B x M x D]
            feats1 (dict):
                keypoints: [B x N x 2]
                keypoint_scores: [B x N]
                descriptors: [B x N x D]
            matches (dict):
                matches0: [B x M]
                matching_scores0: [B x M]
                matches1: [B x N]
                matching_scores1: [B x N]
                matches: List[[Si x 2]]
                scores: List[[Si]]
                stop: int
                prune0: [B x M]
                prune1: [B x N]
        """
        feats0 = self.extractor(image0)
        feats1 = self.extractor(image1)
        matches = self.matcher({'image0': feats0, 'image1': feats1})
        return feats0, feats1, matches

    @property
    def path_checkpoint(self):
        return self.logging_dir.joinpath('nemo.ckpt')

    def train(self, datasets_train: Dict[str, OD3D_Dataset], datasets_val: Dict[str, OD3D_Dataset]):
        self.extractor.train()
        self.matcher.train()
        pass


    def test(self, dataset: OD3D_Dataset, config_inference: DictConfig = None):
        self.extractor.eval()
        self.matcher.eval()
        pass

    def _compute_matches_images(self, image0, image1):
        """Compute matches between two images.

        Args:
            image0 (torch.Tensor): image with shape (3,H,W), normalized in [0,1]
            image1 (torch.Tensor): image with shape (3,H,W), normalized in [0,1]

        Returns:
            points0 (torch.Tensor): coordinates in image #0, shape (K,2)
            points1 (torch.Tensor): coordinates in image #1, shape (K,2)
        """
        # extract local features
        feats0 = self.extractor(image0)  # auto-resize the image, disable with resize=None
        feats1 = self.extractor(image1)

        # match the features
        matches01 = self.matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
        matches = matches01['matches']  # indices with shape (K,2)
        points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
        points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
        return points0, points1

    def train_batch(self, batch) -> OD3D_Results:
        pass
