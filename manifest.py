# -*- coding: utf-8 -*-
"""
manifest.py

Loads the JSON manifest that pairs each image with its label. A manifest
entry looks like:

    {"image": "data/images/case001.nii.gz",
     "label": "data/labels/case001.nii.gz",
     "split": "train"}

A manifest (rather than a filename convention) is used because production
file naming is not expected to match any fixed pattern.
"""
import json


def load_manifest(manifest_path):
    """
    Reads the manifest JSON file.
    Args:
        manifest_path (str): Path to the manifest JSON file.
    Returns:
        list[dict]: Each entry has at least "image", "label", "split" keys.
    Raises:
        FileNotFoundError: If the manifest file does not exist.
        ValueError: If an entry is missing a required key or has an
            unrecognized "split" value.
    """
    with open(manifest_path, "r") as f:
        entries = json.load(f)

    valid_splits = {"train", "val"}
    for entry in entries:
        missing = {"image", "label", "split"} - entry.keys()
        if missing:
            raise ValueError(f"Manifest entry {entry} is missing keys: {missing}")
        if entry["split"] not in valid_splits:
            raise ValueError(
                f"Manifest entry {entry} has unrecognized split "
                f"'{entry['split']}' (expected one of {valid_splits})"
            )
    return entries


def split_manifest(entries):
    """
    Splits manifest entries into train/val lists based on their "split" key.
    Args:
        entries (list[dict]): Entries as returned by load_manifest.
    Returns:
        tuple[list[dict], list[dict]]: (train_entries, val_entries), with the
            "split" key stripped out (only "image"/"label" kept).
    """
    train_entries = [
        {"image": e["image"], "label": e["label"]}
        for e in entries
        if e["split"] == "train"
    ]
    val_entries = [
        {"image": e["image"], "label": e["label"]}
        for e in entries
        if e["split"] == "val"
    ]
    if not train_entries:
        raise ValueError("Manifest contains no entries with split == 'train'")
    if not val_entries:
        raise ValueError("Manifest contains no entries with split == 'val'")
    return train_entries, val_entries
