# -*- coding: utf-8 -*-
"""
dataset.py

Builds MONAI PersistentDataset/DataLoader objects from a manifest and the
transform pipelines in transforms.py. PersistentDataset caches the
deterministic part of the pipeline (loading, resampling, normalization) to
disk; the random patch cropping and augmentations in transforms.py are
re-applied on every access.
"""
import numpy as np
from monai.data import DataLoader, PersistentDataset, list_data_collate

import transforms as transforms_module


def build_datasets(config, train_entries, val_entries):
    """
    Args:
        config: config module (see config.py).
        train_entries (list[dict]): "image"/"label" pairs for training.
        val_entries (list[dict]): "image"/"label" pairs for validation.
    Returns:
        tuple[PersistentDataset, PersistentDataset]: (train_ds, val_ds)
    """
    train_num_samples = int(np.ceil(config.TRAIN_PATCH_BUDGET / len(train_entries)))
    val_num_samples = int(np.ceil(config.VAL_PATCH_BUDGET / len(val_entries)))

    train_ds = PersistentDataset(
        data=train_entries,
        transform=transforms_module.get_train_transforms(config, train_num_samples),
        cache_dir=config.CACHE_DIR,
    )
    val_ds = PersistentDataset(
        data=val_entries,
        transform=transforms_module.get_val_transforms(config, val_num_samples),
        cache_dir=config.CACHE_DIR,
    )
    return train_ds, val_ds


def build_dataloaders(config, train_ds, val_ds):
    """
    Args:
        config: config module (see config.py).
        train_ds, val_ds: datasets as returned by build_datasets.
    Returns:
        tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=list_data_collate,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=list_data_collate,
        shuffle=False,
    )
    return train_loader, val_loader
