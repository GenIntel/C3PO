import logging

logger = logging.getLogger(__name__)
from od3d.methods.method import OD3D_Method
from od3d.datasets.dataset import OD3D_Dataset
from od3d.benchmark.results import OD3D_Results
from od3d.io import get_obj_from_config
from omegaconf import DictConfig
import numpy as np
import warnings
import logging
from typing import Dict, Tuple, Union

logger = logging.getLogger(__name__)
import torch

torch.multiprocessing.set_sharing_strategy("file_system")
from pathlib import Path

# note: math is actually used by config

# import sys
# sys.path.append(str(Path(__file__).parents[4] / 'third_party/LightGlue'))
from od3d.cv.geometry.mesh import Meshes
from od3d.models.feature_extractors import Extractor, rbd
from od3d.cv.utils.dnnlib import construct_class_by_name
from .utils import (
    filter_matches,
    pad_to_length,
    normalize_keypoints,
    TokenConfidence,
    MatchAssignment,
    LearnableFourierPositionalEncoding,
    TransformerLayer,
    matcher_metrics,
)
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
    def __init__(self, config: DictConfig, device="cpu"):
        super().__init__()
        self.config = config
        self.device = device
        self.net: Extractor = construct_class_by_name(self.config.features.backbone)

    def to(self, device):
        super().to(device)
        self.net.to(device)
        self.device = device

    def forward(self, data: Dict[str, torch.Tensor]):
        """Forward pass of the feature extractor.

        Args:
            Dict: dictionary with the following keys:
                - image (torch.Tensor): input image with shape (B, C, H, W). B = 1 supported only

        Returns:
            Dict: dictionary with the following keys:
                - keypoints (torch.Tensor): detected keypoints with shape (B, K, 2)
                - keypoint_scores (torch.Tensor): keypoint scores with shape (B, K)
                - descriptors (torch.Tensor): descriptors with shape (B, K, D)
        """
        return self.net.extract(data, self.config.features.preprocess)

    def load_checkpoint(self, path_checkpoint: Path = None):
        self.net.load_checkpoint(path_checkpoint)

    def save_checkpoint(self, path_checkpoint: Path = None):
        pass

    @property
    def evaluation(self):
        return not self.training


class Matcher(nn.Module):
    def __init__(self, config: DictConfig, device="cpu"):
        super().__init__()
        self.config = config
        self.device = device
        h, n, d = (
            self.config.lightglue.num_heads,
            self.config.lightglue.n_layers,
            self.config.lightglue.descriptor_dim,
        )
        if (
            self.config.features.backbone.output_dim
            != self.config.lightglue.descriptor_dim
        ):
            self.input_proj = nn.Linear(
                self.config.features.backbone.output_dim,
                d,
                bias=True,
            )
        else:
            self.input_proj = nn.Identity()
        head_dim = d // h
        self.posenc = LearnableFourierPositionalEncoding(
            2 + 2 * self.config.features.backbone.add_scale_ori,
            head_dim,
            head_dim,
        )
        self.transformers = nn.ModuleList(
            [TransformerLayer(d, h, self.config.lightglue.flash) for _ in range(n)],
        )
        self.output_proj = nn.Linear(d, 1, bias=True)
        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = nn.ModuleList(
            [TokenConfidence(d) for _ in range(n - 1)],
        )
        self.register_buffer(
            "confidence_thresholds",
            torch.Tensor(
                [
                    self.confidence_threshold(i)
                    for i in range(self.config.lightglue.n_layers)
                ],
            ),
        )

        # training
        self.loss_fn = construct_class_by_name(self.config.train.loss)

        # static lengths LightGlue is compiled for (only used with torch.compile)
        self.static_lengths = None
        self.to(device)

    @property
    def evaluation(self):
        return not self.training

    def forward(self, data: Dict) -> Dict[str, Union[torch.Tensor, int]]:
        with torch.autocast(enabled=self.config.lightglue.mp, device_type="cuda"):
            return self._compute_matches(data)

    def load_checkpoint(self, path_checkpoint: Path = None):
        state_dict = None
        if path_checkpoint is not None:
            state_dict = torch.load(path_checkpoint, map_location=self.device)
        elif self.config.get("features", None) and self.config.lightglue.load_from_url:
            fname = f"{self.config.features.backbone.weights}_{self.config.lightglue.version.replace('.', '-')}.pth"
            print(
                f"Loading checkpoint from {self.config.lightglue.url.format(self.config.lightglue.version,self.config.features.backbone.name)}",
            )
            state_dict = torch.hub.load_state_dict_from_url(
                self.config.lightglue.url.format(
                    self.config.lightglue.version,
                    self.config.features.backbone.name,
                ),
                file_name=fname,
                map_location=self.device,
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
            logger.warning("No checkpoint loaded for Matcher.")

    def compile(self, static_lengths, mode="reduce-overhead"):
        if self.config.lightglue.width_confidence != -1:
            warnings.warn(
                "Point pruning is partially disabled for compiled forward.",
                stacklevel=2,
            )

        for i in range(self.config.lightglue.n_layers):
            self.transformers[i].masked_forward = torch.compile(
                self.transformers[i].masked_forward,
                mode=mode,
                fullgraph=True,
            )
        self.static_lengths = static_lengths

    def confidence_threshold(self, layer_index: int) -> float:
        """scaled confidence threshold"""
        threshold = 0.8 + 0.1 * np.exp(
            -4.0 * layer_index / self.config.lightglue.n_layers,
        )
        return np.clip(threshold, 0, 1)

    def get_pruning_mask(
        self,
        confidences: torch.Tensor,
        scores: torch.Tensor,
        layer_index: int,
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

    def _compute_matches(self, data: Dict[str, Dict[str, torch.Tensor]]):
        assert (
            "image0" in data and "image1" in data
        ), "Missing image0 or image1 in data."
        data0, data1 = data["image0"], data["image1"]
        assert (
            "keypoints" in data0 and "keypoints" in data1
        ), "Missing keypoints in data."
        assert (
            "descriptors" in data0 and "descriptors" in data1
        ), "Missing descriptors in data."
        kpts0, kpts1 = data0["keypoints"], data1["keypoints"]
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape
        device = kpts0.device
        size0, size1 = data0.get("image_size"), data1.get("image_size")
        kpts0 = normalize_keypoints(kpts0, size0).clone()
        kpts1 = normalize_keypoints(kpts1, size1).clone()

        if self.config.features.backbone.add_scale_ori:
            kpts0 = torch.cat(
                [kpts0] + [data0[k].unsqueeze(-1) for k in ("scales", "oris")],
                -1,
            )
            kpts1 = torch.cat(
                [kpts1] + [data1[k].unsqueeze(-1) for k in ("scales", "oris")],
                -1,
            )
        desc0 = data0["descriptors"].contiguous()
        desc1 = data1["descriptors"].contiguous()
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

        all_desc0, all_desc1 = [], []

        # GNN + final_proj + assignment
        do_early_stop = self.config.lightglue.depth_confidence > 0 and self.evaluation
        do_point_pruning = (
            self.config.lightglue.width_confidence > 0
            and not do_compile
            and self.evaluation
        )
        pruning_th = self.pruning_min_kpts(device)
        if do_point_pruning:
            ind0 = torch.arange(0, m, device=device)[None]
            ind1 = torch.arange(0, n, device=device)[None]
            # We store the index of the layer at which pruning is detected.
            prune0 = torch.ones_like(ind0)
            prune1 = torch.ones_like(ind1)
        # TODO: What are these tokens ?
        token0, token1 = None, None
        for i in range(self.config.lightglue.n_layers):
            no_keypoints = desc0.shape[1] == 0 or desc1.shape[1] == 0
            if no_keypoints:
                assert self.evaluation, "Keypoints should not be empty when training."
                break
            desc0, desc1 = self.transformers[i](
                desc0,
                desc1,
                encoding0,
                encoding1,
                mask0=mask0,
                mask1=mask1,
            )
            if self.training:
                all_desc0.append(desc0)
                all_desc1.append(desc1)

            if i == self.config.lightglue.n_layers - 1:
                continue  # no early stopping or adaptive width at last layer

            if do_early_stop:
                token0, token1 = self.token_confidence[i](desc0, desc1)
                if self.check_if_stop(token0[..., :m], token1[..., :n], i, m + n):
                    break

            if do_point_pruning:
                if desc0.shape[-2] < pruning_th:
                    scores0 = self.log_assignment[i].get_matchability(desc0)
                    prunemask0 = self.get_pruning_mask(token0, scores0, i)
                    keep0 = torch.where(prunemask0)[1]
                    ind0 = ind0.index_select(1, keep0)
                    desc0 = desc0.index_select(1, keep0)
                    encoding0 = encoding0.index_select(-2, keep0)
                    prune0[:, ind0] += 1
                if desc0.shape[-2] < pruning_th:
                    scores1 = self.log_assignment[i].get_matchability(desc1)
                    prunemask1 = self.get_pruning_mask(token1, scores1, i)
                    keep1 = torch.where(prunemask1)[1]
                    ind1 = ind1.index_select(1, keep1)
                    desc1 = desc1.index_select(1, keep1)
                    encoding1 = encoding1.index_select(-2, keep1)
                    prune1[:, ind1] += 1

        if no_keypoints:
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
        m0, m1, mscores0, mscores1 = filter_matches(
            scores,
            self.config.lightglue.filter_threshold,
        )

        if self.evaluation:
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

        eval_results = (
            {
                "stop": i + 1,
                "matches": matches,
                "scores": mscores,
            }
            if self.evaluation
            else {}
        )
        train_results = (
            {
                "ref_descriptors0": torch.stack(all_desc0, 1),
                "ref_descriptors1": torch.stack(all_desc1, 1),
                "log_assignment": scores,
            }
            if self.training
            else {}
        )
        return (
            {
                "matches0": m0,
                "matches1": m1,
                "matching_scores0": mscores0,
                "matching_scores1": mscores1,
                "prune0": prune0,
                "prune1": prune1,
            }
            | eval_results
            | train_results
        )

    def loss(self, pred, data):
        def loss_params(pred, i):
            la, _ = self.log_assignment[i](
                pred["ref_descriptors0"][:, i],
                pred["ref_descriptors1"][:, i],
            )
            return {
                "log_assignment": la,
            }

        sum_weights = 1.0
        nll, gt_weights, loss_metrics = self.loss_fn(loss_params(pred, -1), data)
        N = pred["ref_descriptors0"].shape[1]
        losses = {"total": nll, "last": nll.clone().detach(), **loss_metrics}

        if self.training:
            losses["confidence"] = 0.0

        # B = pred['log_assignment'].shape[0]
        losses["row_norm"] = pred["log_assignment"].exp()[:, :-1].sum(2).mean(1)
        for i in range(N - 1):
            params_i = loss_params(pred, i)
            nll, _, _ = self.loss_fn(params_i, data, weights=gt_weights)

            if self.conf.loss.gamma > 0.0:
                weight = self.conf.loss.gamma ** (N - i - 1)
            else:
                weight = i + 1
            sum_weights += weight
            losses["total"] = losses["total"] + nll * weight

            losses["confidence"] += self.token_confidence[i].loss(
                pred["ref_descriptors0"][:, i],
                pred["ref_descriptors1"][:, i],
                params_i["log_assignment"],
                pred["log_assignment"],
            ) / (N - 1)

            del params_i
        losses["total"] /= sum_weights

        # confidences
        if self.training:
            losses["total"] = losses["total"] + losses["confidence"]

        metrics = matcher_metrics(pred, data) if self.evaluation else {}
        return losses, metrics


class TransformerMatching(OD3D_Method):
    def __init__(
        self,
        config: DictConfig,
        logging_dir,
        device="cpu",
    ):
        super().__init__(config=config, logging_dir=logging_dir)
        self.extractor = FeatureExtractor(config, device)
        self.matcher = Matcher(config, device)

        # init neural meshes
        self.total_params = sum(p.numel() for p in self.extractor.parameters()) + sum(
            p.numel() for p in self.matcher.parameters()
        )
        self.fpaths_meshes = [
            self.config.fpaths_meshes[cls] for cls in config.categories
        ]
        fpaths_meshes_tform_obj = self.config.get("fpaths_meshes_tform_obj", None)
        if fpaths_meshes_tform_obj is not None:
            self.fpaths_meshes_tform_obj = [
                fpaths_meshes_tform_obj[cls] for cls in config.categories
            ]
        else:
            self.fpaths_meshes_tform_obj = [None for _ in config.categories]

        self.meshes = Meshes.load_from_files(
            fpaths_meshes=self.fpaths_meshes,
            fpaths_meshes_tforms=self.fpaths_meshes_tform_obj,
        )
        self.meshes_ranges = self.meshes.get_ranges().detach().cuda()
        logger.info(f"loading meshes from following fpaths: {self.fpaths_meshes}...")

        self.verts_count_max = self.meshes.verts_counts_max
        self.mem_verts_feats_count = len(config.categories) * self.verts_count_max
        self.mem_clutter_feats_count = (
            config.neural_mesh.num_noise * config.neural_mesh.max_group
        )
        self.mem_count = self.mem_verts_feats_count + self.mem_clutter_feats_count

        self.feats_bank_count = self.verts_count_max * len(self.meshes) + 1
        self.clutter_feats = torch.nn.Parameter(
            torch.randn(size=(1, config.features.backbone.output_dim), device=device),
            requires_grad=True,
        )
        self.meshes.set_feats_cat_with_pad(
            torch.nn.Parameter(
                torch.randn(
                    size=(
                        self.verts_count_max * len(self.meshes),
                        config.features.backbone.output_dim,
                    ),
                    device=device,
                ),
                requires_grad=True,
            ),
        )

        self.seq_obj_tform4x4_est_obj = {}
        self.seq_obj_tform4x4_est_obj_sim = {}

        crit_kwargs = {}
        if (
            config.train.loss.class_name
            == "od3d.cv.metric.cross_entropy_smooth.CrossEntropyLabelsSmoothed"
        ):
            crit_kwargs[
                "labels_smoothed"
            ] = self.meshes.get_geodesic_prob_with_noise().to(device=device)
        self.criterion = construct_class_by_name(
            self.config.train.loss,
            **crit_kwargs,
        ).cuda()

        self.to(device)

    def setup_optimizers(self):
        params = (
            [p for p in self.extractor.parameters() if p.requires_grad]
            + [p for p in self.matcher.parameters() if p.requires_grad]
            + [self.meshes.feats]
            + [self.clutter_feats]
        )
        self.optim = get_obj_from_config(
            config=self.config.train.optimizer,
            params=params,
        )
        self.scheduler = get_obj_from_config(
            self.optim,
            config=self.config.train.scheduler,
        )

    def set_requires_grad(self, extractor_grad=True, matcher_grad=True):
        for param in self.extractor.parameters():
            param.requires_grad = extractor_grad
        for param in self.matcher.parameters():
            param.requires_grad = matcher_grad

    def save_checkpoint(self, path_checkpoint: Path):
        pass

    def load_checkpoint(
        self,
        extractor_checkpoint: Path = None,
        matcher_checkpoint: Path = None,
        mesh_checkpoint: Path = None,
    ):
        self.extractor.load_checkpoint(extractor_checkpoint)
        self.matcher.load_checkpoint(matcher_checkpoint)
        if mesh_checkpoint is not None:
            raise NotImplementedError("Loading mesh checkpoint is not implemented yet.")
            self.meshes.load_state_dict(
                torch.load(mesh_checkpoint, map_location=self.device),
            )

    def to(self, device):
        self.extractor.to(device)
        self.matcher.to(device)
        self.meshes.to(device)
        self.device = device

    def cuda(self):
        self.to("cuda")

    @property
    def training(self):
        return self.matcher.training and self.matcher.training

    @property
    def evaluation(self):
        return not self.training

    def compile(
        self,
        mode="reduce-overhead",
        static_lengths=[256, 512, 768, 1024, 1280, 1536],
    ):
        self.extractor = torch.compile(self.extractor, mode=mode, fullgraph=True)
        self.matcher.compile(mode=mode, static_lengths=static_lengths)

    def match_pair(
        self,
        image0: Dict[str, torch.Tensor],
        image1: Dict[str, torch.Tensor],
    ):
        """
        Match keypoints and descriptors between two images

        Input:
            image0
                image (torch.Tensor): [B x C x H x W]
                ...
            NEMO_ONLY:
                keypoints: [B x N x 2]
                keypoint_scores: [B x N]
            image1
                image (torch.Tensor): [B x C x H x W]
                ...
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
        matches = self.matcher({"image0": feats0, "image1": feats1})
        return feats0, feats1, matches

    @property
    def path_checkpoint(self):
        return self.logging_dir.joinpath("nemo.ckpt")

    def train(
        self,
        datasets_train: Dict[str, OD3D_Dataset],
        datasets_val: Dict[str, OD3D_Dataset],
    ):
        if self.config.train.profile:
            prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    str(self.logging_dir),
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            prof.__enter__()
        self.set_requires_grad(
            extractor_grad=self.config.features.backbone.freeze,
            matcher_grad=True,
        )
        self.extractor.train()
        self.matcher.train()
        self.load_checkpoint(
            extractor_checkpoint=self.config.features.backbone.checkpoint,
            matcher_checkpoint=self.config.lightglue.checkpoint,
        )
        train_dataset: OD3D_Dataset = datasets_train["main"]
        val = "main" in datasets_val
        if val:
            eval_dataset: OD3D_Dataset = datasets_val["main"]

        # setup optimizer and scheduler
        params = (
            self.matcher.parameters()
            if self.config.features.backbone.freeze
            else list(self.extractor.parameters()) + list(self.matcher.parameters())
        )
        optimizer: torch.optim.Optimizer = construct_class_by_name(
            self.config.train.optimizer,
            params=params,
        )
        scheduler: torch.optim.lr_scheduler._LRScheduler = construct_class_by_name(
            self.config.train.scheduler,
            optimizer=optimizer,
        )

        for epoch in range(self.config.train.epochs):
            for it, batch in enumerate(train_dataset):
                print(batch)
                optimizer.zero_grad()
                losses = self._train_step(batch)
                loss = torch.mean(losses["total"])
                if torch.isnan(loss).any():
                    logger.warn(f"Detected NAN, skipping epoch {epoch}, iteration {it}")
                    del data, loss, losses
                    continue
                loss.backward()
                optimizer.step()
                # wandb.log(loss)
            scheduler.step(epoch)
            if val:
                self.test(eval_dataset)
            if self.config.train.profile:
                prof.step()
        self.save_checkpoint(self.path_checkpoint)

    def test(self, dataset: OD3D_Dataset):
        self.extractor.eval()
        self.matcher.eval()

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        data0 = {"image": batch["image0"]}
        data1 = {"image": batch["image1"]}
        if self.config.features.backbone.requires_kpts:
            data0["keypoints"] = batch["keypoints0"]
            data0["visibility"] = batch["visibility0"]
            data1["keypoints"] = batch["keypoints1"]
            data1["visibility"] = batch["visibility1"]
        feats0 = self.extractor(data0)
        feats1 = self.extractor(data1)
        matches = self.matcher({"image0": feats0, "image1": feats1})
        loss, _ = self.matcher.loss(matches, batch)
        return loss

    def _test_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        data0 = {"image": batch["image0"]}
        data1 = {"image": batch["image1"]}
        if self.config.features.backbone.requires_kpts:
            data0["keypoints"] = batch["keypoints0"]
            data0["visibility"] = batch["visibility0"]
            data1["keypoints"] = batch["keypoints1"]
            data1["visibility"] = batch["visibility1"]
        feats0 = self.extractor(data0)
        feats1 = self.extractor(data1)
        matches = self.matcher({"image0": feats0, "image1": feats1})
        loss, metrics = self.matcher.loss(matches, batch)
        return loss, metrics

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
        feats0 = self.extractor(
            image0,
        )  # auto-resize the image, disable with resize=None
        feats1 = self.extractor(image1)

        # match the features
        matches01 = self.matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = (
            rbd(x) for x in [feats0, feats1, matches01]
        )  # remove batch dimension
        matches = matches01["matches"]  # indices with shape (K,2)
        points0 = feats0["keypoints"][
            matches[..., 0]
        ]  # coordinates in image #0, shape (K,2)
        points1 = feats1["keypoints"][
            matches[..., 1]
        ]  # coordinates in image #1, shape (K,2)
        return points0, points1

    def train_batch(self, batch) -> OD3D_Results:
        pass
