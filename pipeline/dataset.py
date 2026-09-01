# -*- coding: utf-8 -*-
"""
dataset.py

Builds MONAI PersistentDataset/DataLoader objects from a manifest and the
transform pipelines in transforms.py. PersistentDataset caches the
deterministic part of the pipeline (loading, resampling, normalization) to
disk; the random patch cropping and augmentations in transforms.py are
re-applied on every access.
"""
import json
import os

import numpy as np
from monai.data import DataLoader, PersistentDataset, list_data_collate

from . import transforms as transforms_module


def _write_cache_settings(config):
    """
    Records what preprocessing produced this cache, beside it.

    CACHE_DIR is named by a hash so that changing the preprocessing gets a
    fresh cache without anyone having to rename anything -- but a directory
    called `c6226b28fe9d` tells whoever finds it nothing at all. This is the
    same sidecar `truncate.cut_settings` writes next to every cut, for the
    same reason: a derived artefact should say what produced it.

    Informational only. Nothing reads it back -- the hash in the directory
    name IS the key, and it is computed from these values.
    """
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    path = os.path.join(config.CACHE_DIR, "preprocessing.json")
    if os.path.exists(path):
        return
    with open(path, "w") as handle:
        json.dump({"fingerprint": config.PREPROCESSING_FINGERPRINT,
                   **config.PREPROCESSING_SETTINGS}, handle, indent=2, sort_keys=True)


def build_datasets(config, train_entries, val_entries):
    """
    Args:
        config: config module (see config.py).
        train_entries (list[dict]): "image"/"label" pairs for training.
        val_entries (list[dict]): "image"/"label" pairs for validation.
    Returns:
        tuple[PersistentDataset, PersistentDataset]: (train_ds, val_ds)
    """
    _write_cache_settings(config)

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
    # persistent_workers because an epoch is only 28 steps: without it the
    # 16 worker processes are torn down and respawned every ~35 seconds, and
    # each generation leaves shared-memory segments to be reclaimed. Under
    # the "file_system" sharing strategy train.py selects, that churn is what
    # eventually exhausts /dev/shm mid-run.
    common = dict(
        batch_size=config.BATCH_SIZE,
        pin_memory=True,
        collate_fn=list_data_collate,
    )
    if config.NUM_WORKERS > 0:
        common.update(
            prefetch_factor=config.PREFETCH_FACTOR, persistent_workers=True
        )

    train_loader = DataLoader(
        train_ds, num_workers=config.NUM_WORKERS, shuffle=True, **common
    )

    val_common = dict(common)
    if config.VAL_NUM_WORKERS == 0:
        val_common.pop("prefetch_factor", None)
        val_common.pop("persistent_workers", None)
    val_loader = DataLoader(
        val_ds, num_workers=config.VAL_NUM_WORKERS, shuffle=False, **val_common
    )
    return train_loader, val_loader
