import warnings
from typing import Dict

import torch
from omegaconf import DictConfig
from torch import nn

from ..backbones.resnet_old.backbone_old import ResNetExt
from .extractor import Extractor


class NeMoBackbone(Extractor):
    required_keywords = ["image", "keypoints", "visibility"]

    def __init__(self, config: DictConfig):
        super().__init__(config)  # Update with default configuration.
        self.net = ResNetExt(config)

    def load_checkpoint(self, checkpoint_path: str = None):
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.net.load_state_dict(
                state_dict,
                strict=self.config.get("strict_loading", False),
            )
        elif self.config.get("checkpoint", None) is not None:
            state_dict = torch.load(self.config.checkpoint, map_location=self.device)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            assert (
                "strict_loading" in self.config
            ), "If checkpoint is given, strict_loading must be specified"
            self.net.load_state_dict(state_dict, strict=self.config.strict_loading)
        else:
            warnings.warn("No checkpoint provided, using random initialization.")

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert all(
            [kw in data for kw in self.required_keywords],
        ), f"Required keywords: {self.required_keywords}. Found keywords: {list(data.keys())}."
        image = data["image"]  # (B, C, H, W)
        keypoints = data["keypoints"]  # (B, N, 2)
        keypoints = keypoints[..., [1, 0]]  # (B, N, 2) -> (B, N, 2) (y, x)
        visibility = data["visibility"]  # (B, N)
        features = self.net(image)  # (B, C, H//R, W//R)
        # Extract features at keypoints.
        features_kp = (
            nn.functional.grid_sample(
                features,
                keypoints.unsqueeze(1),
            )
            .squeeze(2)
            .permute(0, 2, 1)
        )  # (B, N, C)

        result = {
            "keypoints": keypoints,
            "keypoint_scores": visibility,
            "descriptors": features_kp,
        }
        return result

    def forward2(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert all(
            [kw in data for kw in self.required_keywords],
        ), f"Required keywords {self.required_keywords} not found in data."
        image = data["image"]  # (B, C, H, W)
        features = self.net(image)  # (B, C, H//R, W//R)
        keypoints = None  # TODO: compute keypoints coordinates from features (pixels where similarity with mesh features is high)
        similarity = None  # TODO: compute similarity between features and mesh features
        # Extract features at keypoints.
        features_kp = nn.functional.grid_sample(
            features,
            keypoints.unsqueeze(1),
        )  # check that this works
        result = {
            "keypoints": keypoints,
            "keypoint_scores": similarity,
            "descriptors": features_kp,
        }
        return result
