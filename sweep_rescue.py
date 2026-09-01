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

Every row is read on trees cut by the rule the sweep is NOT sweeping, and one
part of it is worth knowing before reading any number: --peel-terminals
defaults to `1 0` here, as in `compute_dice.py` -- the reference loses its
last layer of tips and the prediction does not. Its tips are where a hand and
a model disagree for reasons that are not the model's, and the asymmetry is
because a model draws a vessel thinner than a hand does, so the floor has
already stopped its tree earlier; see `compute_dice.PEEL_TERMINALS`. It is
deliberately not in the grid -- every row is then a rescue read on the same
tips -- and `--peel-terminals 1 1` or `0` are the other two settings.

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
from compute_dice import (PEEL_TERMINALS, PRED_SUFFIX, class_pairs, dice, parse_rewrite,
                          prediction_path_for, relocate, rewrite_paths)
from truncate import (add_cut_arguments, add_skeleton_arguments, branch_coverage,
                      centerline_support, class_mask, limit_terminal_length, peel_layers,
                      peel_terminals, plan_cut, read_volume, retable, select_branches, subdivide,
                      to_grid)


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

    # the reference's whole tree on the scoring grid, to say what share of it
    # each floor leaves standing
    reference_ml = float((scoring[0] >= 0).sum() * voxel_ml)

    rows = []
    for step in args.steps:
        for index, plan in enumerate(plans):
            retable(plan, subdivide(raw[index], plan["tree"]["world"], plan["tree"]["radii"],
                                    plan["tree"]["voxel_size"], step))
        for floor in args.floors:
            # the plain rule, on both sides: the baseline, and what every
            # rescue is judged against whatever the other knobs are
            plain = [trim(plan["tree"]["table"],
                          select_branches(plan["tree"]["table"], floor, args.max_generation,
                                          args.ordering, args.min_strahler), side, args)
                     for side, plan in enumerate(plans)]
            support_masks = [None, None]
            # the support is only ever read by a rescue: with no margin in the
            # grid, nothing asks for it, and moving it between the two grids is
            # the one resample per floor this loop would otherwise pay
            if "mask" in args.supports and any(m > 0 for m in args.margins):
                onto_prediction = cut_from(*scoring, kept_nodes(plans[0]["tree"], plain[0]))
                onto_reference = cut_from(*owners[1], kept_nodes(plans[1]["tree"], plain[1]))
                if regrid:
                    onto_reference = to_grid(onto_reference, prediction_affine, plans[0]["shape"],
                                             reference_affine)
                support_masks = [onto_reference, onto_prediction]
            rows.extend(sweep_knobs(plans, plain, support_masks, owners, scoring, step, floor,
                                    reference_ml, voxel_ml, args))
    return rows, None


def trim(table, keep, side, args):
    """
    What `truncate.py` does to a selection after `select_branches`: the
    terminal layers peeled, then what is left of the terminal runs capped.

    `side` is 0 for the reference and 1 for the prediction, because the peel
    is not the same on the two (`compute_dice.PEEL_TERMINALS`). Neither knob
    is swept -- they are part of the rule the sweep is run under, so every row
    of the sweep is a rescue read on the same tips.
    """
    keep = peel_terminals(table, keep, peel_layers(args)[side])
    return limit_terminal_length(table, keep, args.max_terminal_length)


def sweep_knobs(plans, plain, support_masks, owners, scoring, step, floor, reference_ml, voxel_ml,
                args):
    """Every (support, distance, margin) at one --cut-step and one floor."""
    rescuing = any(margin > 0 for margin in args.margins)
    rows = []
    for support in args.supports:
        # the mask overlap has no distance to sweep; the correspondence is
        # computed once per distance and every margin reads it
        for distance in ([None] if support == "mask" else list(args.distances)):
            # nothing to build when no margin will ask for it: with --margins 0
            # the selection never calls a predicate, and the mask support was
            # not resampled either
            predicates = [None, None]
            for index, plan in enumerate(plans if rescuing else ()):
                branches = plan["tree"]["table"]
                if support == "mask":
                    coverage = branch_coverage(plan, support_masks[index])
                    predicates[index] = (
                        lambda branch, coverage=coverage: coverage[branch] >= args.rescue_coverage)
                else:
                    matched = centerline_support(plan, plans[1 - index], plain[1 - index], distance)
                    predicates[index] = (
                        lambda branch, matched=matched, branches=branches:
                        float(matched[branches[branch]["nodes"]].mean()) >= args.rescue_coverage)
            for margin in args.margins:
                keep = []
                for index, plan in enumerate(plans):
                    if margin <= 0 or not plain[1 - index]:
                        keep.append(plain[index])
                        continue
                    keep.append(trim(plan["tree"]["table"], select_branches(
                        plan["tree"]["table"], floor, args.max_generation,
                        args.ordering, args.min_strahler, margin=margin,
                        supported=predicates[index]), index, args))
                reference_cut = cut_from(*scoring, kept_nodes(plans[0]["tree"], keep[0]))
                prediction_cut = cut_from(*owners[1], kept_nodes(plans[1]["tree"], keep[1]))
                reference_large_ml = float(reference_cut.sum() * voxel_ml)
                rows.append({
                    "min_diameter_mm": floor, "cut_step_mm": step, "support": support,
                    "margin_mm": margin,
                    "peel_terminals_reference": peel_layers(args)[0],
                    "peel_terminals_prediction": peel_layers(args)[1],
                    "kept_fraction_reference": (reference_large_ml / reference_ml
                                                if reference_ml else float("nan")),
                    "distance_radii": "" if distance is None else distance,
                    "n_kept_reference": len(keep[0]), "n_kept_prediction": len(keep[1]),
                    "n_rescued_reference": len(keep[0] - plain[0]),
                    "n_rescued_prediction": len(keep[1] - plain[1]),
                    "reference_kept_mm": sum(plans[0]["tree"]["table"][b]["length_mm"]
                                             for b in keep[0]),
                    "prediction_kept_mm": sum(plans[1]["tree"]["table"][b]["length_mm"]
                                              for b in keep[1]),
                    "reference_large_ml": reference_large_ml,
                    "prediction_large_ml": float(prediction_cut.sum() * voxel_ml),
                    "dice_large": dice(reference_cut, prediction_cut),
                })
    return rows


def summarize(rows, args):
    """The grid, averaged over the cases that produced every combination."""
    print(f"\n{'floor':>7}{'step':>6}{'support':>11}{'margin':>8}{'dist':>6}{'dice_large':>13}"
          f"{'kept vol':>10}{'centerline ref/pred':>22}{'volume gap mL':>15}")
    for floor in args.floors:
        for step in args.steps:
            for support in args.supports:
                for distance in ([""] if support == "mask" else list(args.distances)):
                    for margin in args.margins:
                        subset = [r for r in rows
                                  if r["min_diameter_mm"] == floor and r["cut_step_mm"] == step
                                  and r["support"] == support and r["margin_mm"] == margin
                                  and r["distance_radii"] == distance]
                        if not subset:
                            continue
                        mean = lambda key: float(np.mean([r[key] for r in subset]))
                        gap = float(np.mean([abs(r["reference_large_ml"] - r["prediction_large_ml"])
                                             for r in subset]))
                        centerline = (f"{mean('reference_kept_mm'):.0f}"
                                      f" / {mean('prediction_kept_mm'):.0f}")
                        print(f"{floor:>7.1f}{step:>6.1f}{support:>11}{margin:>8.2f}"
                              f"{str(distance):>6}{mean('dice_large'):>13.4f}"
                              f"{mean('kept_fraction_reference'):>9.1%}"
                              f"{centerline:>22}{gap:>15.2f}")
    reference_peel, prediction_peel = peel_layers(args)
    if reference_peel or prediction_peel:
        print(f"\nEvery row is read on trees whose tips are gone: --peel-terminals "
              f"{reference_peel} {prediction_peel}, the\ndefault here as in compute_dice.py"
              + (" -- the reference peeled and the prediction not,\nbecause a model draws a "
                 "vessel thinner than the hand that annotated it and the floor\nhas already "
                 "stopped its tree earlier." if reference_peel != prediction_peel else ".")
              + "\nIt is not swept: it is part of the rule the sweep is run under, so the rescue "
                "is read\non the same tips throughout, and 'centerline ref/pred' says whether it "
                "left the two\ntrees the same length. Report --peel-terminals with whatever the "
                "table settles.")
    print("\nThe three knobs do not behave alike, so do not read the table the same way down\n"
          "each column.\n"
          "\n"
          "  floor  changes WHICH REGION is scored, so its Dice moves in no guaranteed\n"
          "         direction -- a higher floor is an easier region but it is also a smaller\n"
          "         one, with its own errors, and the number can fall. What is monotone is\n"
          "         'kept vol': a floor that leaves 20% of the volume standing is a Dice on\n"
          "         the hilum, whatever it reads. And no single model settles this -- the\n"
          "         right floor is the one where two models still SEPARATE, so run this on\n"
          "         both checkpoints and take the floor where the gap between them is widest.\n"
          "  step   watch 'centerline ref/pred'. The two disagreeing is the granularity\n"
          "         problem -- a long branch one side cuts whole and the other keeps in part.\n"
          "         It should close as the step shrinks, and then stop closing.\n"
          "  margin an on/off more than a distance: the rescue reaches at most where the other\n"
          "         side's plain cut stopped, so 0 against non-0 is the only real comparison.")


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
    parser.add_argument("--floors", type=float, nargs="+", default=None, metavar="MM",
                        help="--min-diameter values to put in the grid: what \"large vessel\" is "
                             "taken to mean. Default: the single value of --min-diameter")
    parser.add_argument("--steps", type=float, nargs="+", default=[0.0, 3.0, 5.0, 10.0],
                        metavar="MM",
                        help="--cut-step values to put in the grid: how long a piece of branch the "
                             "floor is applied to. 0 is one decision per branch. Default: 0 3 5 10")
    parser.add_argument("--supports", nargs="+", choices=("mask", "centerline"),
                        default=["mask", "centerline"],
                        help="Which definition(s) of supported to put in the grid. Default: both")
    parser.add_argument("--label", default="",
                        help="Written into every row, to say which model produced it. Two "
                             "checkpoints swept into two CSVs then concatenate into the one "
                             "table the floor is actually chosen on")
    parser.add_argument("--csv", help="One row per case, class and combination")
    add_cut_arguments(parser, peel_terminals=PEEL_TERMINALS)
    add_skeleton_arguments(parser)
    return parser


def main():
    args = build_parser().parse_args()
    args.peel_terminals = list(peel_layers(args))
    args.max_shift = None
    args.angle_offset = DIRECTION_OFFSET
    args.quiet = True
    args.floors = args.floors or [args.min_diameter]

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
          f"distance(s), floor {args.min_diameter} mm"
          + f", --peel-terminals {' '.join(str(n) for n in peel_layers(args))}")

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
                row.update(case=case, model=args.label, **{"class": name})
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
        head = ("model", "case", "class")
        columns = list(head) + [k for k in flat[0] if k not in head]
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(flat)
        print(f"\nwrote {args.csv}  ({len(flat)} rows)")


if __name__ == "__main__":
    main()
