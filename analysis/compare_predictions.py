# -*- coding: utf-8 -*-
"""
compare_predictions.py

Volume-by-volume comparison of two prediction sets produced by
inference.py (e.g. two different checkpoints/models run on the same
input images). Predictions live side by side in one directory per
patient and are told apart by filename suffix
(<stem><suffix_a>.nii.gz vs <stem><suffix_b>.nii.gz).

No ground truth is assumed -- this measures agreement between the two
predictions, not accuracy against a reference. For each class (plus a
merged "vessel" class = all foreground) it computes, per case:

- Dice: voxel overlap.
- NSD: Normalized Surface Dice at a configurable tolerance (mm) -- the
  fraction of each surface within tolerance of the other. More robust
  than centerline (clDice) metrics here: clDice assumes a thin tubular
  structure whose skeleton is a compact 1D curve, which breaks down
  (skeleton degenerates into an unstable surface) on the thicker/more
  fragmented regions present in these predictions.
- HD95 / ASD: 95th-percentile Hausdorff distance and average symmetric
  surface distance (mm), i.e. how far the two segmented surfaces are
  from each other.
- Volumes (mm^3) and their relative difference.
- Number of connected components, as a rough fragmentation signal.

Predictions from different checkpoints/configs may not agree on which
raw class index means "artery" vs "vein" (e.g. a model trained with a
swapped LABEL_CLASS_MAP) -- use --swap-av-a/--swap-av-b to correct for
a known inversion in one of the two inputs before comparing.

Writes one row per (case, class) to a long-format CSV.
"""
import argparse
import glob
import logging
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.metrics import (
    compute_average_surface_distance,
    compute_hausdorff_distance,
    compute_surface_dice,
)
from scipy.ndimage import label as connected_components

from pipeline import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def find_pairs(root_dir, suffix_a, suffix_b):
    """
    Recursively scans root_dir for "*.nii.gz" files and pairs them up by
    stripping suffix_a/suffix_b from the filename stem: two files in the
    same directory that share the resulting stem are a pair.

    Returns:
        list[tuple[str, str, str, str]]: (case_id, path_a, path_b, folder)
            sorted by case_id. case_id combines the patient folder name
            and the shared stem, since the same stem could in principle
            recur across patients.
    """
    files = glob.glob(os.path.join(root_dir, "**", "*.nii.gz"), recursive=True)
    by_stem = {}
    for f in files:
        folder, base = os.path.split(f)
        stem = base[: -len(".nii.gz")]
        for suffix, key in ((suffix_a, "a"), (suffix_b, "b")):
            if stem.endswith(suffix):
                case_stem = stem[: -len(suffix)]
                by_stem.setdefault((folder, case_stem), {})[key] = f
                break

    pairs = []
    for (folder, case_stem), found in sorted(by_stem.items()):
        if "a" in found and "b" in found:
            case_id = f"{os.path.basename(folder)}/{case_stem}"
            pairs.append((case_id, found["a"], found["b"], folder))
        else:
            logging.warning(
                f"Incomplete pair for '{case_stem}' in {folder}: found {sorted(found)}, skipping"
            )
    return pairs


def load_label_volume(path):
    img = nib.load(path)
    data = np.asarray(img.dataobj).astype(np.uint8)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    return data, spacing


def dice(a, b):
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def surface_metrics(a, b, spacing, nsd_tolerance_mm):
    """
    Returns (hd95_mm, asd_mm, nsd). NaN (and nsd=0.0) when one mask is
    empty and the other isn't (distance to an empty set is undefined);
    (0.0, 0.0, 1.0) when both are empty (perfect, trivial agreement).
    """
    if a.sum() == 0 and b.sum() == 0:
        return 0.0, 0.0, 1.0
    if a.sum() == 0 or b.sum() == 0:
        return float("nan"), float("nan"), 0.0
    at = torch.from_numpy(a[None, None]).float()
    bt = torch.from_numpy(b[None, None]).float()
    hd95 = compute_hausdorff_distance(
        at, bt, include_background=True, percentile=95, spacing=spacing
    ).item()
    asd = compute_average_surface_distance(
        at, bt, include_background=True, symmetric=True, spacing=spacing
    ).item()
    nsd = compute_surface_dice(
        at, bt, class_thresholds=[nsd_tolerance_mm], include_background=True, spacing=spacing
    ).item()
    return hd95, asd, nsd


def num_components(a):
    _, n = connected_components(a)
    return int(n)


def swap_artery_vein(data, class_names):
    """Swaps the raw label values of the "artery" and "vein" classes in place, to correct for a known inversion between the two models' class-index conventions."""
    artery_idx = class_names.index("artery")
    vein_idx = class_names.index("vein")
    swapped = data.copy()
    swapped[data == artery_idx] = vein_idx
    swapped[data == vein_idx] = artery_idx
    return swapped


def compare_pair(case_id, path_a, path_b, class_names, swap_av_a, swap_av_b, skip_surface, skip_topology, nsd_tolerance_mm):
    data_a, spacing_a = load_label_volume(path_a)
    data_b, spacing_b = load_label_volume(path_b)

    if data_a.shape != data_b.shape:
        logging.warning(
            f"{case_id}: shape mismatch {data_a.shape} vs {data_b.shape}, skipping"
        )
        return []
    if spacing_a != spacing_b:
        logging.warning(f"{case_id}: spacing mismatch {spacing_a} vs {spacing_b}, using {spacing_a}")
    voxel_volume_mm3 = float(np.prod(spacing_a))

    if swap_av_a:
        data_a = swap_artery_vein(data_a, class_names)
    if swap_av_b:
        data_b = swap_artery_vein(data_b, class_names)

    classes = [(idx, name) for idx, name in enumerate(class_names) if idx != 0]
    classes.append((None, "vessel"))  # merged foreground, all classes combined

    rows = []
    for class_idx, class_name in classes:
        if class_idx is None:
            mask_a, mask_b = data_a > 0, data_b > 0
        else:
            mask_a, mask_b = data_a == class_idx, data_b == class_idx

        row = {
            "case_id": case_id,
            "class": class_name,
            "dice": dice(mask_a, mask_b),
            "volume_a_mm3": float(mask_a.sum() * voxel_volume_mm3),
            "volume_b_mm3": float(mask_b.sum() * voxel_volume_mm3),
        }
        max_vol = max(row["volume_a_mm3"], row["volume_b_mm3"])
        row["volume_diff_pct"] = (
            100.0 * abs(row["volume_a_mm3"] - row["volume_b_mm3"]) / max_vol if max_vol > 0 else 0.0
        )

        if not skip_topology:
            row["n_components_a"] = num_components(mask_a)
            row["n_components_b"] = num_components(mask_b)

        if not skip_surface:
            row["hd95_mm"], row["asd_mm"], row["nsd"] = surface_metrics(
                mask_a, mask_b, spacing_a, nsd_tolerance_mm
            )

        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root-dir", required=True, help="Directory tree containing both prediction sets (searched recursively for **/*.nii.gz)")
    parser.add_argument("--suffix-a", default="_vascular_pred_ct", help="Filename suffix (before .nii.gz) identifying model A's predictions")
    parser.add_argument("--suffix-b", default="_vascular_pred2", help="Filename suffix (before .nii.gz) identifying model B's predictions")
    parser.add_argument("--swap-av-a", action="store_true", help="Swap artery/vein class indices in model A's predictions before comparing (use if model A is known to invert them)")
    parser.add_argument("--swap-av-b", action="store_true", help="Swap artery/vein class indices in model B's predictions before comparing")
    parser.add_argument("--class-names", nargs="+", default=None, help="Override class list (index 0 = background). Defaults to config.CLASS_NAMES")
    parser.add_argument("--output", default="prediction_comparison.csv", help="Output CSV path (long format: one row per case x class)")
    parser.add_argument("--nsd-tolerance-mm", type=float, default=2.0, help="Tolerance (mm) for Normalized Surface Dice (default: 2.0)")
    parser.add_argument("--skip-surface", action="store_true", help="Skip HD95/ASD/NSD (slow on large volumes)")
    parser.add_argument("--skip-topology", action="store_true", help="Skip connected-component counts")
    args = parser.parse_args()

    class_names = args.class_names or cfg.CLASS_NAMES

    pairs = find_pairs(args.root_dir, args.suffix_a, args.suffix_b)
    if not pairs:
        parser.error(
            f"No matching pairs found under {args.root_dir} for suffixes "
            f"'{args.suffix_a}' / '{args.suffix_b}'"
        )
    logging.info(f"Found {len(pairs)} matched pairs")

    all_rows = []
    for i, (case_id, path_a, path_b, _folder) in enumerate(pairs, 1):
        logging.info(f"[{i}/{len(pairs)}] Comparing {case_id}")
        all_rows.extend(
            compare_pair(
                case_id, path_a, path_b, class_names,
                args.swap_av_a, args.swap_av_b,
                args.skip_surface, args.skip_topology, args.nsd_tolerance_mm,
            )
        )

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    logging.info(f"Wrote {len(df)} rows ({len(pairs)} cases x {df['class'].nunique()} classes) to {args.output}")

    summary_cols = [c for c in ["dice", "nsd", "volume_diff_pct", "hd95_mm", "asd_mm"] if c in df.columns]
    summary = df.groupby("class")[summary_cols].agg(["mean", "std"])
    logging.info(f"Summary by class:\n{summary}")


if __name__ == "__main__":
    main()
