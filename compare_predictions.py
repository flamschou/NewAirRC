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
- clDice: centerline Dice (Shit et al., 2021), symmetrized between the
  two masks -- sensitive to tree continuity/connectivity rather than
  raw volume overlap, which matters for thin tubular vessels.
- HD95 / ASD: 95th-percentile Hausdorff distance and average symmetric
  surface distance (mm), i.e. how far the two segmented surfaces are
  from each other.
- Volumes (mm^3) and their relative difference.
- Number of connected components, as a rough fragmentation signal.

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
from monai.metrics import compute_average_surface_distance, compute_hausdorff_distance
from scipy.ndimage import label as connected_components
from skimage.morphology import skeletonize

import config as cfg

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


def cl_dice(a, b):
    """
    Symmetric centerline Dice between two binary masks (Shit et al.,
    2021). Used here pred-vs-pred (no ground truth available): each mask
    stands in as the "reference" for the other's skeleton in turn, then
    the two directional scores are harmonic-averaged.
    """
    if a.sum() == 0 and b.sum() == 0:
        return 1.0
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    skel_a = skeletonize(a)
    skel_b = skeletonize(b)
    t_prec = np.logical_and(skel_a, b).sum() / max(skel_a.sum(), 1)
    t_sens = np.logical_and(skel_b, a).sum() / max(skel_b.sum(), 1)
    if t_prec + t_sens == 0:
        return 0.0
    return float(2 * t_prec * t_sens / (t_prec + t_sens))


def surface_metrics(a, b, spacing):
    """
    Returns (hd95_mm, asd_mm). NaN when one mask is empty and the other
    isn't (distance to an empty set is undefined); (0.0, 0.0) when both
    are empty (perfect, trivial agreement).
    """
    if a.sum() == 0 and b.sum() == 0:
        return 0.0, 0.0
    if a.sum() == 0 or b.sum() == 0:
        return float("nan"), float("nan")
    at = torch.from_numpy(a[None, None]).float()
    bt = torch.from_numpy(b[None, None]).float()
    hd95 = compute_hausdorff_distance(
        at, bt, include_background=True, percentile=95, spacing=spacing
    ).item()
    asd = compute_average_surface_distance(
        at, bt, include_background=True, symmetric=True, spacing=spacing
    ).item()
    return hd95, asd


def num_components(a):
    _, n = connected_components(a)
    return int(n)


def compare_pair(case_id, path_a, path_b, class_names, skip_surface, skip_topology):
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
            row["cldice"] = cl_dice(mask_a, mask_b)
            row["n_components_a"] = num_components(mask_a)
            row["n_components_b"] = num_components(mask_b)

        if not skip_surface:
            row["hd95_mm"], row["asd_mm"] = surface_metrics(mask_a, mask_b, spacing_a)

        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root-dir", required=True, help="Directory tree containing both prediction sets (searched recursively for **/*.nii.gz)")
    parser.add_argument("--suffix-a", default="_vascular_pred_ct", help="Filename suffix (before .nii.gz) identifying model A's predictions")
    parser.add_argument("--suffix-b", default="_vascular_pred2", help="Filename suffix (before .nii.gz) identifying model B's predictions")
    parser.add_argument("--class-names", nargs="+", default=None, help="Override class list (index 0 = background). Defaults to config.CLASS_NAMES")
    parser.add_argument("--output", default="prediction_comparison.csv", help="Output CSV path (long format: one row per case x class)")
    parser.add_argument("--skip-surface", action="store_true", help="Skip HD95/ASD (slow on large volumes)")
    parser.add_argument("--skip-topology", action="store_true", help="Skip clDice / connected-component counts")
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
            compare_pair(case_id, path_a, path_b, class_names, args.skip_surface, args.skip_topology)
        )

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    logging.info(f"Wrote {len(df)} rows ({len(pairs)} cases x {df['class'].nunique()} classes) to {args.output}")

    summary_cols = [c for c in ["dice", "cldice", "volume_diff_pct", "hd95_mm", "asd_mm"] if c in df.columns]
    summary = df.groupby("class")[summary_cols].agg(["mean", "std"])
    logging.info(f"Summary by class:\n{summary}")


if __name__ == "__main__":
    main()
