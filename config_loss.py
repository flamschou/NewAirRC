# -*- coding: utf-8 -*-
"""
config_loss.py

Training losses: the deep-supervision-weighted Dice+CE loss used since the
start, and an optional clDice continuity term layered on top of it.

Dice and cross-entropy are both voxel-wise: a two-voxel break in a vessel
costs them almost nothing, yet it splits the tree in two and, downstream,
shifts the Strahler order of everything past the break (see centerline.py).
clDice (Shit et al., CVPR 2021) is the cheapest term that sees such a break,
because it scores the *skeletons* of the masks rather than the masks.
"""
from contextlib import contextmanager
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from monai.losses import DiceCELoss
from monai.networks.utils import one_hot


class DeepSupervisionLossBase(nn.Module):
    def __init__(
        self,
        deep_supr_num: int,
        weights: Optional[np.ndarray] = None,
        **kwargs_base_loss,
    ):
        super().__init__()
        self.num_levels = deep_supr_num + 1

        # Create weights
        if weights is not None:
            self.weights = torch.from_numpy(weights.astype(np.float32))
        else:
            weights_array = np.array(
                [1 / (2**i) for i in range(self.num_levels)], dtype=np.float32
            )
            weights_array[-1] = 0.0
            total_weight = weights_array.sum()
            if total_weight > 0:
                weights_array = weights_array / total_weight
            self.weights = torch.from_numpy(weights_array)

    def forward(
        self, outputs: torch.Tensor, labels: torch.Tensor, **kwargs_loss: Any
    ) -> torch.Tensor:
        # If not deep supervision format, calculate single loss
        if outputs.ndim != 6:
            return self.base_loss(outputs, labels, **kwargs_loss)

        # Calculate weighted sum of losses across all levels
        total_loss = torch.tensor(0.0, device=outputs.device)
        for level_index in range(self.num_levels):
            level_weight = self.weights[level_index]
            if level_weight < 1e-8:
                continue
            level_output = outputs[:, level_index, ...]
            level_loss = self.base_loss(level_output, labels, **kwargs_loss)
            total_loss += level_weight * level_loss

        return total_loss


class DeepSupervisionDiceCELoss(DeepSupervisionLossBase):
    def __init__(self, deep_supr_num: int, **kwargs):
        super().__init__(deep_supr_num, **kwargs)
        self.base_loss = DiceCELoss(to_onehot_y=True, softmax=True, **kwargs)


@contextmanager
def _force_fp32(device_type: str):
    """
    Disables autocast for the enclosed block, so the clDice term is computed
    in fp32 even under `precision="16-mixed"`.

    The soft skeleton is a chain of ~2 x num_iter pooling passes ending in
    differences of nearly-equal numbers (`img - opened`), and the two ratios
    it feeds divide sums that can be small; in fp16 that chain loses enough
    precision to make the gradient noisy. Only cuda/cpu are guarded --
    torch.autocast has no meaningful state to override on the other
    backends this pipeline can land on.
    """
    if device_type in ("cuda", "cpu"):
        with torch.autocast(device_type=device_type, enabled=False):
            yield
    else:
        yield


class SoftSkeletonize(nn.Module):
    """
    Differentiable ("soft") 3D skeletonization, as defined by clDice.

    Morphological thinning written with pooling so it stays differentiable:
    erosion is a min-filter -- a max-pool on the negated input -- over the
    three 1D 3-voxel kernels (6-connectivity), dilation is a 3x3x3 max-pool.
    Each iteration peels one voxel off the surface and keeps whatever the
    opening removed, i.e. the ridge; the ridges accumulate into the
    skeleton.

    `num_iter` must be at least the largest vessel *radius* in voxels, or
    the thickest vessels never thin down to a curve and their core never
    enters the skeleton. At TARGET_SPACING = 1 mm and vessels up to ~10 mm
    across, that is 5; the cost is linear in num_iter, so there is no
    reason to go far beyond it.

    Memory, not compute, is the binding constraint here: the thinning is a
    chain of ~20 elementwise/pooling ops per iteration, each of which
    autograd would keep a full-resolution copy of. `use_checkpoint` trades
    that for one recomputation of each iteration during the backward pass
    -- see the comment on the loop below.
    """

    def __init__(self, num_iter: int = 6, use_checkpoint: bool = True):
        super().__init__()
        self.num_iter = num_iter
        self.use_checkpoint = use_checkpoint

    @staticmethod
    def soft_erode(img: torch.Tensor) -> torch.Tensor:
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)

    @staticmethod
    def soft_dilate(img: torch.Tensor) -> torch.Tensor:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def _thin(self, img: torch.Tensor, skeleton: torch.Tensor):
        """
        One thinning iteration: erode once, add to the skeleton whatever the
        opening of `img` removed, and hand the eroded volume to the next
        iteration.

        The single erosion is not a shortcut. The textbook formulation
        erodes twice per iteration -- once to shrink the volume, once more
        inside `open = dilate(erode(.))` -- but the second erosion of
        iteration j computes exactly the first erosion of iteration j+1, so
        keeping the eroded volume around makes the two the same tensor.
        """
        eroded = self.soft_erode(img)
        delta = F.relu(img - self.soft_dilate(eroded))
        # soft union of skeleton and delta: a + b - a*b
        return eroded, skeleton + F.relu(delta - skeleton * delta)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): (B, 1, D, H, W) values in [0, 1] -- one
                probability channel, or one channel of a one-hot label.
        Returns:
            torch.Tensor: same shape, the soft skeleton.
        """
        skeleton = torch.zeros_like(img)
        for _ in range(self.num_iter + 1):
            if self.use_checkpoint and torch.is_grad_enabled() and img.requires_grad:
                # Checkpointing per iteration, not around the whole loop:
                # autograd then holds two full-resolution tensors per
                # iteration instead of ~20, and the backward pass only ever
                # rebuilds one iteration's graph at a time. Around the whole
                # loop it would rebuild all of them at once, which is the
                # peak we are trying to avoid in the first place.
                img, skeleton = checkpoint(
                    self._thin, img, skeleton, use_reentrant=False
                )
            else:
                img, skeleton = self._thin(img, skeleton)
        return skeleton


class SoftClDiceLoss(nn.Module):
    """
    1 - clDice, averaged over the foreground classes.

    clDice is the harmonic mean of two quantities:
      - topology precision: how much of the *predicted* skeleton falls
        inside the true mask -- penalizes skeletons drawn through
        non-vessel;
      - topology sensitivity: how much of the *true* skeleton falls inside
        the predicted mask -- this is the one that sees breaks, since a
        gap removes a whole run of true skeleton from the intersection
        while costing Dice only the handful of voxels of the gap itself.

    Background is excluded: its skeleton is the medial surface of
    everything that is not a vessel, which says nothing about vessel
    continuity and costs as much to compute as the vessel classes.

    The sums run over the whole batch rather than per patch. With
    POS_NEG_SAMPLE_RATIO = (1, 1) roughly half the patches hold little or
    no foreground, and a per-patch mean would be dominated by their
    degenerate (smooth / smooth) ratios.

    `max_patches` caps how many patches of the batch the term looks at.
    RandCropByPosNegLabeld returns num_samples patches per item and
    list_data_collate flattens them, so the batch reaching the loss is
    BATCH_SIZE x num_samples patches -- 24 of 128^3 in the default setup,
    which is more full-resolution volume than the skeletonization can hold
    a graph for. Since the term is a batch-level statistic anyway, scoring
    a random subset of the patches is a noisier estimate of the same
    quantity, not a different one. The subset is random while training and
    the first `max_patches` in eval, so val_loss stays reproducible.
    """

    def __init__(
        self,
        num_iter: int = 6,
        smooth: float = 1.0,
        include_background: bool = False,
        max_patches: int = 0,
    ):
        super().__init__()
        self.skeletonize = SoftSkeletonize(num_iter=num_iter)
        self.smooth = smooth
        self.include_background = include_background
        self.max_patches = max_patches

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): (B, C, D, H, W) raw network output for a
                single resolution level (no deep-supervision axis).
            labels (torch.Tensor): (B, 1, D, H, W) integer class indices.
        Returns:
            torch.Tensor: scalar loss in [0, 1].
        """
        num_classes = logits.shape[1]
        first_class = 0 if self.include_background else 1

        if 0 < self.max_patches < logits.shape[0]:
            if self.training:
                keep = torch.randperm(logits.shape[0], device=logits.device)
                keep = keep[: self.max_patches]
            else:
                keep = torch.arange(self.max_patches, device=logits.device)
            logits, labels = logits[keep], labels[keep]

        with _force_fp32(logits.device.type):
            probabilities = torch.softmax(logits.float(), dim=1)
            target = one_hot(labels, num_classes=num_classes, dim=1).float()

            class_losses = []
            for class_index in range(first_class, num_classes):
                prediction = probabilities[:, class_index : class_index + 1]
                truth = target[:, class_index : class_index + 1]

                skeleton_prediction = self.skeletonize(prediction)
                # the label is a constant: no graph, no saved activations
                with torch.no_grad():
                    skeleton_truth = self.skeletonize(truth)

                topology_precision = (
                    torch.sum(skeleton_prediction * truth) + self.smooth
                ) / (torch.sum(skeleton_prediction) + self.smooth)
                topology_sensitivity = (
                    torch.sum(skeleton_truth * prediction) + self.smooth
                ) / (torch.sum(skeleton_truth) + self.smooth)

                cl_dice = (
                    2.0
                    * topology_precision
                    * topology_sensitivity
                    / (topology_precision + topology_sensitivity)
                )
                class_losses.append(1.0 - cl_dice)

            return torch.stack(class_losses).mean()


class DeepSupervisionDiceCEClDiceLoss(DeepSupervisionDiceCELoss):
    """
    Dice+CE on every deep-supervision level, plus clDice on the
    full-resolution level only.

    Not on the lower levels, on purpose: at 1/2 and 1/4 resolution the
    thinnest vessels (~1.3 mm across at TARGET_SPACING = 1 mm) are thinner
    than a voxel, so their "skeleton" is an artifact of the downsampling --
    and the skeletonization is the expensive part of the term.

    `cldice_weight` is read on every forward rather than baked in, so
    train.py can ramp it up from 0 during the warm-up. A weight of 0 skips
    the term entirely, which is what makes this class safe to use as the
    only loss in the pipeline.
    """

    def __init__(
        self,
        deep_supr_num: int,
        cldice_weight: float = 0.0,
        cldice_iterations: int = 6,
        cldice_smooth: float = 1.0,
        cldice_max_patches: int = 0,
        **kwargs,
    ):
        super().__init__(deep_supr_num, **kwargs)
        self.cldice_loss = SoftClDiceLoss(
            num_iter=cldice_iterations,
            smooth=cldice_smooth,
            max_patches=cldice_max_patches,
        )
        self.cldice_weight = float(cldice_weight)

    def forward(
        self, outputs: torch.Tensor, labels: torch.Tensor, **kwargs_loss: Any
    ) -> torch.Tensor:
        loss = super().forward(outputs, labels, **kwargs_loss)
        if self.cldice_weight <= 0.0:
            return loss

        main_output = outputs[:, 0, ...] if outputs.ndim == 6 else outputs
        return loss + self.cldice_weight * self.cldice_loss(main_output, labels)


def cldice_warmup_weight(
    epoch: int, target_weight: float, warmup_epochs: int
) -> float:
    """
    Linear ramp of the clDice weight from 0 to `target_weight` over the
    first `warmup_epochs` epochs.

    The skeleton of a near-random probability map is noise, and clDice on
    noise pulls the prediction towards thin, fragmented masks it then has
    to climb back out of. Ramping in keeps the term quiet until Dice+CE
    has produced something roughly vessel-shaped for it to fix.

    Args:
        epoch (int): 0-based epoch index.
        target_weight (float): config.CLDICE_WEIGHT.
        warmup_epochs (int): config.CLDICE_WARMUP_EPOCHS; 0 disables the
            ramp (full weight from the first step).
    Returns:
        float
    """
    if target_weight <= 0.0:
        return 0.0
    if warmup_epochs <= 0:
        return target_weight
    return target_weight * min(1.0, (epoch + 1) / warmup_epochs)


def build_loss(config):
    """
    Args:
        config: config module (see config.py). Uses DEEP_SUPERVISION_LEVELS
            and the CLDICE_* settings.
    Returns:
        DeepSupervisionDiceCEClDiceLoss: with the clDice weight at 0 if
            config.CLDICE_WEIGHT is 0, or at its warm-up starting point
            otherwise (train.py updates it every epoch).
    """
    return DeepSupervisionDiceCEClDiceLoss(
        deep_supr_num=config.DEEP_SUPERVISION_LEVELS,
        cldice_weight=cldice_warmup_weight(
            0, config.CLDICE_WEIGHT, config.CLDICE_WARMUP_EPOCHS
        ),
        cldice_iterations=config.CLDICE_ITERATIONS,
        cldice_smooth=config.CLDICE_SMOOTH,
        cldice_max_patches=config.CLDICE_MAX_PATCHES,
    )
