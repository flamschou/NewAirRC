# -*- coding: utf-8 -*-
"""
transforms.py

Modality-agnostic MONAI preprocessing/augmentation pipelines. No CT-specific
logic (e.g. Hounsfield unit clipping) -- this pipeline targets already
roughly-normalized volumes (originally written for z-scored synthetic MRI).
"""
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
)

KEYS = ["image", "label"]


def _make_label_filter(keep_classes):
    """
    Args:
        keep_classes (Sequence[int]): raw label values to keep as
            foreground (remapped to 1); every other value becomes 0.
    Returns:
        Callable[[torch.Tensor], torch.Tensor]
    """
    keep_classes = list(keep_classes)

    def _filter(label):
        keep_tensor = torch.as_tensor(keep_classes, dtype=label.dtype, device=label.device)
        mask = torch.isin(label, keep_tensor)
        out = label.clone()
        out[mask] = 1
        out[~mask] = 0
        return out

    return _filter


def _base_transforms(config):
    transforms = [
        LoadImaged(keys=KEYS),
        EnsureChannelFirstd(keys=KEYS),
        Lambdad(keys=["label"], func=_make_label_filter(config.KEEP_LABEL_CLASSES)),
        Orientationd(keys=KEYS, axcodes="RAS"),
        Spacingd(
            keys=KEYS,
            pixdim=config.TARGET_SPACING,
            mode=("bilinear", "nearest"),
        ),
    ]
    if config.NORMALIZE_INTENSITY:
        transforms.append(
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True)
        )
    return transforms


def get_train_transforms(config, num_samples):
    """
    Args:
        config: config module (see config.py).
        num_samples (int): number of random patches to extract per volume
            per access (see config.TRAIN_PATCH_BUDGET).
    Returns:
        monai.transforms.Compose
    """
    pos, neg = config.POS_NEG_SAMPLE_RATIO
    transforms = _base_transforms(config) + [
        RandCropByPosNegLabeld(
            keys=KEYS,
            label_key="label",
            image_key="image",
            spatial_size=config.PATCH_SIZE,
            pos=pos,
            neg=neg,
            num_samples=num_samples,
        ),
        RandFlipd(keys=KEYS, prob=0.5, spatial_axis=0),
        RandFlipd(keys=KEYS, prob=0.5, spatial_axis=1),
        RandFlipd(keys=KEYS, prob=0.5, spatial_axis=2),
        RandRotate90d(keys=KEYS, prob=0.5, max_k=3, spatial_axes=(0, 1)),
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.05),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.15),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.15),
        EnsureTyped(keys=KEYS),
    ]
    return Compose(transforms)


def get_val_transforms(config, num_samples):
    """
    Args:
        config: config module (see config.py).
        num_samples (int): number of random patches to extract per volume
            per access (see config.VAL_PATCH_BUDGET).
    Returns:
        monai.transforms.Compose
    """
    pos, neg = config.POS_NEG_SAMPLE_RATIO
    transforms = _base_transforms(config) + [
        RandCropByPosNegLabeld(
            keys=KEYS,
            label_key="label",
            image_key="image",
            spatial_size=config.PATCH_SIZE,
            pos=pos,
            neg=neg,
            num_samples=num_samples,
        ),
        EnsureTyped(keys=KEYS),
    ]
    return Compose(transforms)
