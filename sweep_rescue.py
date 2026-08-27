# -*- coding: utf-8 -*-
"""
sweep_rescue.py

What --rescue-margin and --rescue-distance are worth on a cohort, read off
the same trees.

`compute_dice.py` cuts each side twice and rescues the branches the floor
only just removed (see `truncate.py: truncate_pair`). Two knobs decide how
far that goes -- how far under the floor a branch may measure, and how far
the other side's axis may sit from its own -- and neither has a value that
can be argued from first principles. This runs the pair of cuts over a grid
of both and prints what each combination does to the Dice, so the value gets
picked on the cohort instead of on a plausible story about partial volume.

It is cheap because it reuses what does not change. The skeletonization is
the expensive half by a wide margin and depends on neither knob, so every
case is planned ONCE (`truncate.plan_cut`); the correspondence between the
two centerlines depends on the distance alone, so it is computed once per
distance and every margin reads it; and the reference is moved onto the
prediction's grid as an owner map rather than as a mask, once, which makes
each combination's Dice a couple of array operations instead of a resample.

Read the table for the knee, not the maximum. The Dice rises with both knobs
by construction -- a wider band admits more agreement -- so the largest
number is always the loosest setting, and it is not the answer. What to look
for is where the curve flattens: past that point the rescue stops recovering
vessels the floor split between the two sides and starts admitting whatever
runs nearby. `n_rescued` climbing while the Dice no longer moves is that
point.

Usage:
    python sweep_rescue.py --manifest manifest_vibe.json --split val --limit 4 \
        --rewrite /data/flamant/data/ct=/biomaps/.../data/ct --csv sweep.csv

    python sweep_rescue.py --manifest manifest_ct.json --split val \
        --margins 0 0.5 1 1.5 2 3 --distances 1.0 1.5 2.0 --csv sweep.csv
"""
import argparse
import csv
import json
import os

import nibabel as nib
import numpy as np

import config as cfg
from centerline import DIRECTION_OFFSET
from compute_dice import (PRED_SUFFIX, class_pairs, dice, parse_rewrite, prediction_path_for,
                          relocate, rewrite_paths)
from truncate import (add_cut_arguments, add_skeleton_arguments, branch_coverage,
                      centerline_support, class_mask, plan_cut, read_volume, retable,
                      select_branches, subdivide, to_grid)


def owner_volumes(plan):
    """
    The plan's voxel bookkeeping as volumes, so it can be resampled.

    A cut is `is_kept[owner] & within` read on the mask's voxels, and only
    `is_kept` changes as the knobs move. Carrying the owner index and the
    sleeve test onto the other grid ONCE therefore gives every combination's
    cut there for free, and gives it exactly: nearest neighbour picks one
    source voxel, and reading both volumes at it is the same as reading the
    mask it would have produced. Background is -1, which no node index is.
    """
    owner = np.full(plan["shape"], -1, dtype=np.int32)
    within = np.zeros(plan["shape"], dtype=bool)
    if len(plan["voxels"]):
        index = tuple(plan["voxels"].T)
        owner[index] = plan["owner"]
        within[index] = plan["within"]
    return owner, within


def resample(volume, affine, target_shape, target_affine, fill):
    """Nearest-neighbour onto another grid, keeping the dtype."""
    from nibabel.processing import resample_from_to

    image = nib.Nifti1Image(volume.astype(np.int32), affine)
    moved = resample_from_to(image, (tuple(target_shape), target_affine), order=0, cval=fill)
    return np.asarray(moved.dataobj).astype(volume.dtype)


def kept_nodes(tree, keep):
    """The nodes a selection of branches covers."""
    marked = np.zeros(len(tree["world"]), dtype=bool)
    if keep:
        marked[np.concatenate([tree["table"][b]["nodes"] for b in keep])] = True
    return marked


def cut_from(owner, within, marked):
    """A selection's mask, on whatever grid `owner` was resampled to."""
    return (owner >= 0) & within & marked[np.maximum(owner, 0)]


def sweep_case(reference_path, reference_values, prediction_path, prediction_values, args):
    """Every (support, distance, margin) of one case and one class, on one pair of trees."""
    reference_data, reference_affine, reference_spacing = read_volume(reference_path)
    prediction_data, prediction_affine, prediction_spacing = read_volume(prediction_path)

    plans, raw = [], []
    for data, affine, spacing, values in (
            (reference_data, reference_affine, reference_spacing, reference_values),
            (prediction_data, prediction_affine, prediction_spacing, prediction_values)):
        mask = class_mask(data, values)
        if not mask.any():
            return None, "no voxel of the class"
        # planned with the branches as they come: the subdivision is swept
        # below, and it does not touch the skeleton the plan is built on
        args.cut_step = 0.0
        plan, _ = plan_cut(mask, affine, spacing, args, verbose=False)
        if plan is None:
            return None, "nothing left after pruning"
        plans.append(plan)
        raw.append(plan["tree"]["table"])

    owners = [owner_volumes(plan) for plan in plans]
    voxel_ml = float(np.prod(prediction_spacing)) / 1000.0

    # the reference's bookkeeping, moved onto the prediction's grid once, so
    # every combination's Dice is read there without a resample of its own
    regrid = not (reference_data.shape == prediction_data.shape
                  and np.allclose(reference_affine, prediction_affine, atol=1e-3))
    if regrid:
        scoring = (resample(owners[0][0], reference_affine, prediction_data.shape,
                            prediction_affine, -1),
                   resample(owners[0][1], reference_affine, prediction_data.shape,
                            prediction_affine, 0))
    else:
        scoring = owners[0]

    rows = []
    for step in args.steps:
        for index, plan in enumerate(plans):
            retable(plan, subdivide(raw[index], plan["tree"]["world"], plan["tree"]["radii"],
                                    plan["tree"]["voxel_size"], step))
        # the plain rule, on both sides: the baseline, and what every rescue is
        # judged against whatever the other knobs are
        plain = [select_branches(plan["tree"]["table"], args.min_diameter, args.max_generation,
                                 args.ordering, args.min_strahler) for plan in plans]
        support_masks = [None, None]
        if "mask" in args.supports:
            onto_prediction = cut_from(*scoring, kept_nodes(plans[0]["tree"], plain[0]))
            onto_reference = cut_from(*owners[1], kept_nodes(plans[1]["tree"], plain[1]))
            if regrid:
                onto_reference = to_grid(onto_reference, prediction_affine, plans[0]["shape"],
                                         reference_affine)
            support_masks = [onto_reference, onto_prediction]
        rows.extend(sweep_knobs(plans, plain, support_masks, owners, scoring, step, voxel_ml, args))
    return rows, None


def sweep_knobs(plans, plain, support_masks, owners, scoring, step, voxel_ml, args):
    """Every (support, distance, margin) at one --cut-step, on one pair of trees."""
    rows = []
    for support in args.supports:
        # the mask overlap has no distance to sweep; the correspondence is
        # computed once per distance and every margin reads it
        for distance in ([None] if support == "mask" else list(args.distances)):
            predicates = []
            for index, plan in enumerate(plans):
                branches = plan["tree"]["table"]
                if support == "mask":
                    coverage = branch_coverage(plan, support_masks[index])
                    predicates.append(
                        lambda branch, coverage=coverage: coverage[branch] >= args.rescue_coverage)
                else:
                    matched = centerline_support(plan, plans[1 - index], plain[1 - index], distance)
                    predicates.append(
                        lambda branch, matched=matched, branches=branches:
                        float(matched[branches[branch]["nodes"]].mean()) >= args.rescue_coverage)
            for margin in args.margins:
                keep = []
                for index, plan in enumerate(plans):
                    if margin <= 0 or not plain[1 - index]:
                        keep.append(plain[index])
                        continue
                    keep.append(select_branches(
                        plan["tree"]["table"], args.min_diameter, args.max_generation,
                        args.ordering, args.min_strahler, margin=margin,
                        supported=predicates[index]))
                reference_cut = cut_from(*scoring, kept_nodes(plans[0]["tree"], keep[0]))
                prediction_cut = cut_from(*owners[1], kept_nodes(plans[1]["tree"], keep[1]))
                rows.append({
                    "cut_step_mm": step, "support": support, "margin_mm": margin,
                    "distance_radii": "" if distance is None else distance,
                    "n_kept_reference": len(keep[0]), "n_kept_prediction": len(keep[1]),
                    "n_rescued_reference": len(keep[0] - plain[0]),
                    "n_rescued_prediction": len(keep[1] - plain[1]),
                    "reference_kept_mm": sum(plans[0]["tree"]["table"][b]["length_mm"]
                                             for b in keep[0]),
                    "prediction_kept_mm": sum(plans[1]["tree"]["table"][b]["length_mm"]
                                              for b in keep[1]),
                    "reference_large_ml": float(reference_cut.sum() * voxel_ml),
                    "prediction_large_ml": float(prediction_cut.sum() * voxel_ml),
                    "dice_large": dice(reference_cut, prediction_cut),
                })
    return rows


def summarize(rows, args):
    """The grid, averaged over the cases that produced every combination."""
    print(f"\n{'step':>6}{'support':>11}{'margin':>8}{'dist':>6}{'dice_large':>13}"
          f"{'rescued ref/pred':>20}{'kept ref/pred':>18}{'centerline ref/pred':>22}"
          f"{'volume gap mL':>15}")
    baseline = None
    for step in args.steps:
        for support in args.supports:
            for distance in ([""] if support == "mask" else list(args.distances)):
                for margin in args.margins:
                    subset = [r for r in rows
                              if r["cut_step_mm"] == step and r["support"] == support
                              and r["margin_mm"] == margin and r["distance_radii"] == distance]
                    if not subset:
                        continue
                    mean = lambda key: float(np.mean([r[key] for r in subset]))
                    gap = float(np.mean([abs(r["reference_large_ml"] - r["prediction_large_ml"])
                                         for r in subset]))
                    rescued = (f"{mean('n_rescued_reference'):.1f}"
                               f" / {mean('n_rescued_prediction'):.1f}")
                    kept = f"{mean('n_kept_reference'):.0f} / {mean('n_kept_prediction'):.0f}"
                    centerline = (f"{mean('reference_kept_mm'):.0f}"
                                  f" / {mean('prediction_kept_mm'):.0f}")
                    if baseline is None:
                        baseline = mean("dice_large")
                    print(f"{step:>6.1f}{support:>11}{margin:>8.2f}{str(distance):>6}"
                          f"{mean('dice_large'):>13.4f}{rescued:>20}{kept:>18}"
                          f"{centerline:>22}{gap:>15.2f}")
    print("\nRead the knee, not the maximum: the Dice rises with every knob by construction.\n"
          "'centerline ref/pred' is the one to watch for --cut-step: it is how many mm of\n"
          "vessel each side calls large, and the two disagreeing is the granularity problem\n"
          "-- a long branch that one side cuts whole and the other keeps in part. It should\n"
          "close as the step shrinks, and then stop closing. The volume gap says the same in mL.")
    print(f"\nthe first row is the cut with no rescue and no subdivision: "
          f"dice_large {baseline:.4f}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=cfg.MANIFEST_PATH)
    parser.add_argument("--split", default="val")
    parser.add_argument("--classes", nargs="+", default=["artery"], metavar="NAME")
    parser.add_argument("--pred-dir", default=None)
    parser.add_argument("--pred-suffix", default=PRED_SUFFIX)
    parser.add_argument("--rewrite", action="append", metavar="OLD=NEW")
    parser.add_argument("--data-dir")
    parser.add_argument("--swap-av", action="store_true")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Only the first N cases of the split. Start here: the trees cost "
                             "the same as a compute_dice run, the grid on top of them is free")
    parser.add_argument("--margins", type=float, nargs="+",
                        default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0], metavar="MM")
    parser.add_argument("--distances", type=float, nargs="+", default=[1.0, 1.5, 2.0],
                        metavar="RADII")
    parser.add_argument("--rescue-coverage", type=float, default=0.5, metavar="FRACTION")
    parser.add_argument("--steps", type=float, nargs="+", default=[0.0, 3.0, 5.0, 10.0],
                        metavar="MM",
                        help="--cut-step values to put in the grid: how long a piece of branch the "
                             "floor is applied to. 0 is one decision per branch. Default: 0 3 5 10")
    parser.add_argument("--supports", nargs="+", choices=("mask", "centerline"),
                        default=["mask", "centerline"],
                        help="Which definition(s) of supported to put in the grid. Default: both")
    parser.add_argument("--csv", help="One row per case, class and combination")
    add_cut_arguments(parser)
    add_skeleton_arguments(parser)
    return parser


def main():
    args = build_parser().parse_args()
    args.max_shift = None
    args.angle_offset = DIRECTION_OFFSET
    args.quiet = True

    with open(args.manifest) as handle:
        entries = [e for e in json.load(handle) if e.get("split") == args.split]
    if not entries:
        raise SystemExit(f"no entry with split={args.split} in {args.manifest}")
    entries = rewrite_paths(entries, parse_rewrite(args.rewrite))
    if args.data_dir:
        entries = relocate(entries, args.data_dir)
    if args.limit:
        entries = entries[: args.limit]
    classes = class_pairs(args.classes, args.swap_av)
    print(f"{len(entries)} case(s), {len(args.margins)} margin(s) x {len(args.distances)} "
          f"distance(s), floor {args.min_diameter} mm")

    rows = []
    for entry in entries:
        case = os.path.basename(entry["label"])
        prediction_file = prediction_path_for(entry["image"], args.pred_dir, args.pred_suffix)
        if not os.path.exists(prediction_file):
            print(f"{case}: no prediction at {prediction_file}, skipping")
            continue
        for name, raw, predicted in classes:
            print(f"{case} [{name}]")
            produced, why = sweep_case(entry["label"], raw, prediction_file, predicted, args)
            if produced is None:
                print(f"  skipped: {why}")
                continue
            for row in produced:
                row.update(case=case, **{"class": name})
            rows.append(produced)
            best = max(produced, key=lambda r: r["dice_large"])
            plain = min(produced, key=lambda r: (r["margin_mm"], r["cut_step_mm"]))
            print(f"  dice_large {plain['dice_large']:.4f} brut, {best['dice_large']:.4f} au mieux "
                  f"(step {best['cut_step_mm']}, {best['support']}, margin {best['margin_mm']})")

    if not rows:
        raise SystemExit("no case scored")
    flat = [row for produced in rows for row in produced]
    summarize(flat, args)
    if args.csv:
        columns = ["case", "class"] + [k for k in flat[0] if k not in ("case", "class")]
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(flat)
        print(f"\nwrote {args.csv}  ({len(flat)} rows)")


if __name__ == "__main__":
    main()
