# -*- coding: utf-8 -*-
"""
truncate.py

Cuts a vascular segmentation back to its proximal tree -- the trunk, the
lobar and the segmental vessels, nothing beyond -- and writes the truncated
mask next to the input.

Why cut at all. A whole-tree Dice is dominated by the periphery: the
subsegmental vessels carry most of the surface and almost none of the
volume, they are where the reference itself is least certain, and a model
that draws the hilum perfectly but fades out two generations early scores
about the same as one that does the opposite. Truncating both sides first
gives a Dice read on the vessels the acquisition actually resolves.

Nothing here computes a metric: it writes masks, and the masks go to
whatever already scores them (`compare_predictions.py`, or the validation
loop). Run it on the references AND on the predictions, with the same
--min-diameter, so the two sides are cut by the same rule -- a truncated
reference scored against a whole prediction would count every peripheral
vessel of the prediction as a false positive.

How the cut is made. The mask is skeletonized and ordered exactly as
`centerline.py` does it -- same pruning, same cycle breaking, same ordering
-- and the tree is then truncated: walking from the root, a branch is kept
only if it clears the calibre floor (`--min-diameter`, read on
`calibre_mm`, the junction-trimmed median radius) AND its parent was kept.
The closure is what makes this a truncation rather than a calibre filter: a
wide distal blob sitting behind a thin branch is a leak, not a large
vessel, and it goes with its parent.

How the cut is turned back into voxels. Every voxel is assigned to the
nearest centerline node -- a Voronoi partition of the mask by its own
skeleton -- and kept when that node belongs to a kept branch and lies
within `--sleeve` local radii. Two consequences worth knowing:

  - the cut surface falls where the tree was truncated, roughly normal to
    the vessel, instead of on an arbitrary plane;
  - dropped branches keep competing for voxels, so a subsegmental vessel
    running along the trunk is claimed by its own centerline and does not
    survive because it happens to touch a large vessel.

What is written. The input's own label values, on the input grid: a mask
truncated with --label 3 holds 3 where it was kept and 0 elsewhere, so the
result is a drop-in replacement for the file it came from. Every class is
truncated on its own tree -- the union of the arterial and the venous tree
is not a tree, and skeletonizing it would order the two against each other.

Usage:
    # one mask, one class -> vascular_case001_large.nii.gz beside it
    python truncate.py --input vascular_case001.nii.gz --label 3

    # every class of config.LABEL_CLASS_MAP, one file per case
    python truncate.py --input vascular_case001.nii.gz --classes

    # the validation split of a manifest, references truncated in place
    python truncate.py --manifest manifest_ct.json --split val

    # the predictions that go with them, cut by the same rule
    python truncate.py --input-dir /data/flamant/data/ct/lidc_idri \
                       --pattern '*_vascular_pred.nii.gz' --classes
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict, deque

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.spatial import cKDTree

from centerline import DIRECTION_OFFSET, build_tree, resample_isotropic, skeletonize_graph

# A segmental pulmonary artery leaves its lobar parent at roughly 4-6 mm and
# the subsegmentals under it at 2-3, so a 4 mm floor keeps the segmentals and
# cuts the generation below them. It is the definition of "large vessel" here
# and it is a knob, not a fact: report the value used with whatever the
# truncated masks end up scoring.
MIN_DIAMETER_MM = 4.0
SUFFIX = "_large"


# --------------------------------------------------------------------------- #
# masks
# --------------------------------------------------------------------------- #
def read_volume(path):
    """Loads a nifti and returns (label array, affine, voxel spacing in mm)."""
    img = nib.load(path)
    data = np.asarray(img.dataobj)
    spacing = np.linalg.norm(img.affine[:3, :3], axis=0)
    return data, img.affine, spacing


def class_mask(data, values=None):
    """
    The boolean mask of one class.

    `values` is the list of raw label values that make up it; None takes
    every nonzero voxel. A list rather than the single label
    `centerline.load_mask` takes, because one training class can come from
    several raw values.
    """
    mask = data > 0 if values is None else np.isin(data, np.asarray(values))
    return np.ascontiguousarray(mask, dtype=bool)


# --------------------------------------------------------------------------- #
# tree
# --------------------------------------------------------------------------- #
def extract_tree(mask, affine, spacing, args, verbose=True):
    """
    Mask -> ordered branch table, through the same stages as `centerline.py`.

    Returns the `build_tree` result with the node arrays the truncation
    needs added to it: `world` (node coordinates in mm, on the input
    volume's own physical frame, whatever grid the skeleton was built on)
    and `radii` (local half-width in mm). Returns None when nothing
    survives the pruning.
    """
    if not args.no_fill_holes:
        mask = binary_fill_holes(mask)

    factors = np.ones(3)
    work_mask, work_affine, work_spacing = mask, affine, spacing
    if not args.no_resample:
        work_mask, work_affine, factors, work_spacing = resample_isotropic(
            mask, affine, spacing, args.spacing)
        if verbose and not np.allclose(factors, 1.0):
            print(f"  resampled to {np.round(work_spacing, 3).tolist()} mm, shape={work_mask.shape}")
    args.factors = factors

    _, _, base_graph, positions, radii, voxel_size, world = skeletonize_graph(
        work_mask, work_affine, work_spacing, verbose=verbose)
    tree = build_tree(base_graph, positions, radii, world, voxel_size, args, args.radius_factor)
    if tree is None:
        return None
    tree.update(world=world, radii=radii, voxel_size=voxel_size)
    return tree


def select_branches(table, min_diameter, max_generation=None, ordering="generation",
                    min_strahler=None):
    """
    The branches that are still a large vessel: the trunk, and everything
    reachable from it through vessels that are themselves large.

    A branch qualifies when

      - its diameter -- twice `calibre_mm`, the median radius with the
        junction blobs trimmed off, not the mean, which the blobs inflate --
        clears `min_diameter`;
      - it sits no deeper than `max_generation` in `ordering`, which counts
        away from the root;
      - and its diameter-defined Strahler order is at least `min_strahler`,
        which counts up from the tips.

    Qualifying is necessary and not sufficient: what comes back is the
    connected component of the qualifying branches that contains the widest
    branch of the tree. That is what makes this a truncation rather than a
    calibre filter -- a wide distal blob behind a thin branch (a leak, a
    fused vein, an aneurysm) does not reach the trunk through large vessels
    and goes with the branch that carries it -- and it is why the connection
    is grown from the WIDEST BRANCH rather than down from the root.

    The root would be the obvious seed, and it is the wrong one. It is
    chosen upstream as the widest free END of the skeleton
    (`centerline.order_branches`), which is the trunk in a clean mask and
    anywhere at all in a degraded one: on an eroded venous prediction here,
    the root landed on a 5 mm peripheral stump whose daughters measured
    3.83 mm, the real 37 mm trunk sat six generations "below" it, and a cut
    at 4 mm kept 1 segment out of 203. Growing from the widest branch cannot
    fail that way, and it agrees with the root-first traversal whenever the
    root is right.
    """
    def qualifies(entry):
        if 2.0 * entry["calibre_mm"] < min_diameter:
            return False
        if max_generation is not None and not 0 <= entry[ordering] <= max_generation:
            return False
        if min_strahler is not None and entry["strahler_dd"] < min_strahler:
            return False
        return True

    seed = max(table, key=lambda entry: entry["calibre_mm"])
    if not qualifies(seed):
        return set()

    # two branches are neighbours when they share an end node; the ordering
    # of the tree plays no part in this, which is the point
    at_node = defaultdict(list)
    for entry in table:
        at_node[entry["nodes"][0]].append(entry["branch_id"])
        at_node[entry["nodes"][-1]].append(entry["branch_id"])

    keep = {seed["branch_id"]}
    queue = deque(keep)
    while queue:
        entry = table[queue.popleft()]
        for end in (entry["nodes"][0], entry["nodes"][-1]):
            for neighbour in at_node[end]:
                if neighbour not in keep and qualifies(table[neighbour]):
                    keep.add(neighbour)
                    queue.append(neighbour)
    return keep


def keep_voxels(mask, affine, tree, keep, sleeve):
    """
    The voxels the truncated tree owns, as a boolean volume on the input grid.

    A voxel is kept when the nearest centerline node of the WHOLE tree --
    kept branches and dropped ones alike -- is one of the kept ones, and
    when it lies within `sleeve` times that node's own radius. Both halves
    matter: the first is the Voronoi partition that puts the cut where the
    tree was truncated and gives every dropped vessel a chance to claim its
    own voxels back, the second drops what the skeleton never covered -- a
    blob the thinning walked past, and the components the pruning removed.

    The sleeve is proportional rather than a fixed millimetre tolerance: a
    voxel of a wide vessel sits further from its axis than a voxel of a thin
    one, so a fixed distance would either cut into the trunk or reach past
    the segmentals.

    Only the mask's own voxels are queried -- a nearest-neighbour lookup
    over a whole 512^3 volume costs minutes and answers nothing, since a
    background voxel is not written either way.
    """
    kept = np.zeros(mask.shape, dtype=bool)
    voxels = np.argwhere(mask)
    if not keep or not len(voxels):
        return kept

    nodes = np.unique(np.concatenate([entry["nodes"] for entry in tree["table"]]))
    is_kept = np.zeros(len(tree["world"]), dtype=bool)
    is_kept[np.concatenate([tree["table"][b]["nodes"] for b in keep])] = True

    # The node coordinates are the raw (unsmoothed) skeleton ones: smoothing
    # is for measuring lengths and angles, and here the axis has to stay
    # where the voxels are.
    points = voxels @ affine[:3, :3].T + affine[:3, 3]
    distance, index = cKDTree(tree["world"][nodes]).query(points)
    owner = nodes[index]
    kept[tuple(voxels.T)] = is_kept[owner] & (distance <= sleeve * tree["radii"][owner])
    return kept


def truncate_class(mask, affine, spacing, args, verbose=True):
    """
    Truncates one class. Returns (kept mask, row) or (None, row) when the
    tree could not be built -- the caller decides whether one unusable class
    is worth losing the file over.
    """
    voxel_ml = float(np.prod(spacing)) / 1000.0
    row = {"min_diameter_mm": args.min_diameter, "sleeve": args.sleeve,
           "volume_ml": float(mask.sum() * voxel_ml)}
    if verbose:
        print(f"  {int(mask.sum())} voxels ({row['volume_ml']:.2f} mL), "
              f"spacing={np.round(spacing, 3).tolist()} mm")

    tree = extract_tree(mask, affine, spacing, args, verbose=verbose)
    if tree is None:
        print("  WARNING: nothing left after pruning, the class is dropped")
        return None, row

    table = tree["table"]
    keep = select_branches(table, args.min_diameter, args.max_generation, args.ordering,
                           args.min_strahler)
    if not keep:
        # the trunk itself is under the floor: the floor is wrong for this
        # volume, or the root landed on a fragment
        widest = max(2.0 * entry["calibre_mm"] for entry in table)
        print(f"  WARNING: no branch clears {args.min_diameter} mm (widest is {widest:.2f} mm), "
              f"the cut is empty")

    kept = keep_voxels(mask, affine, tree, keep, args.sleeve)
    diameters = [2.0 * table[b]["calibre_mm"] for b in keep]
    row.update({
        "large_ml": float(kept.sum() * voxel_ml),
        "large_volume_fraction": float(kept.sum()) / float(mask.sum()) if mask.any() else 0.0,
        "n_segments": len(table),
        "n_segments_kept": len(keep),
        "centerline_mm": sum(entry["length_mm"] for entry in table),
        "centerline_kept_mm": sum(table[b]["length_mm"] for b in keep),
        "min_kept_diameter_mm": min(diameters) if diameters else float("nan"),
        "max_kept_diameter_mm": max(diameters) if diameters else float("nan"),
    })
    if verbose:
        print(f"  kept {row['n_segments_kept']}/{row['n_segments']} segments, "
              f"{row['centerline_kept_mm']:.0f}/{row['centerline_mm']:.0f} mm of centerline, "
              f"{row['large_ml']:.2f}/{row['volume_ml']:.2f} mL "
              f"({row['large_volume_fraction']:.1%} of the volume)")
    return kept, row


# --------------------------------------------------------------------------- #
# files
# --------------------------------------------------------------------------- #
def cut_settings(classes, args):
    """
    The rule a cut was made by, written beside the mask it produced.

    A truncated mask is not readable on its own: two files called
    `..._large.nii.gz` can hold a 4 mm cut of the artery and a 6 mm cut of
    both trees, and nothing in the name says which. The sidecar makes the
    rule part of the output -- so a later run can tell whether the cut on
    disk is the cut it is asking for, instead of silently reusing another
    one, and so the floor that has to be reported with any number read off
    these masks can be found next to them.
    """
    return {
        "classes": {name: (list(values) if values else None) for name, values in classes},
        "min_diameter_mm": args.min_diameter,
        "max_generation": args.max_generation,
        "ordering": args.ordering,
        "min_strahler": args.min_strahler,
        "sleeve": args.sleeve,
        "min_branch_length": args.min_branch_length,
        "radius_factor": args.radius_factor,
        "min_component_length": args.min_component_length,
        "smoothing": args.smoothing,
        "spacing": args.spacing,
        "no_resample": bool(args.no_resample),
        "no_fill_holes": bool(args.no_fill_holes),
        "all_components": bool(args.all_components),
        "keep_cycles": bool(args.keep_cycles),
        "root": list(args.root) if args.root else None,
    }


def settings_path(destination):
    """"case_large.nii.gz" -> "case_large.json", the sidecar of `cut_settings`."""
    if destination.endswith(".nii.gz"):
        return destination[: -len(".nii.gz")] + ".json"
    return os.path.splitext(destination)[0] + ".json"


def output_path(input_path, suffix, output_dir=None):
    """"vascular_case001.nii.gz" -> "vascular_case001_large.nii.gz"."""
    directory, filename = os.path.split(input_path)
    if filename.endswith(".nii.gz"):
        stem, ext = filename[: -len(".nii.gz")], ".nii.gz"
    else:
        stem, ext = os.path.splitext(filename)
    return os.path.join(output_dir if output_dir else directory, f"{stem}{suffix}{ext}")


def truncate_file(path, classes, args, destination=None):
    """
    Truncates every class of one file and writes the result as one volume.

    The label values are the input's own, so the file that comes out is the
    file that went in with its periphery removed, and anything downstream
    that read the original -- a manifest, a comparison script -- reads this
    one unchanged. Raw values outside `classes` are not carried over: they
    were not truncated, and writing them back would produce a file that is
    part cut and part not.

    Returns (rows, kept) with `kept` the boolean mask of each class, so a
    caller that has to score the truncation does not read the file it just
    watched being written.
    """
    data, affine, spacing = read_volume(path)
    output = np.zeros_like(data)
    rows, kept_by_class = [], {}
    for name, values in classes:
        mask = class_mask(data, values)
        print(f"{os.path.basename(path)} [{name}]")
        if not mask.any():
            print(f"  WARNING: no voxel with label {values}, skipping")
            kept_by_class[name] = np.zeros(data.shape, dtype=bool)
            continue
        kept, row = truncate_class(mask, affine, spacing, args, verbose=not args.quiet)
        row.update(file=path, **{"class": name})
        rows.append(row)
        kept_by_class[name] = np.zeros(data.shape, dtype=bool) if kept is None else kept
        if kept is not None:
            output[kept] = data[kept]

    destination = destination or output_path(path, args.suffix, args.output_dir)
    directory = os.path.dirname(destination)
    if directory:
        os.makedirs(directory, exist_ok=True)
    nib.save(nib.Nifti1Image(output, affine), destination)
    with open(settings_path(destination), "w") as handle:
        json.dump(cut_settings(classes, args), handle, indent=1, sort_keys=True)
    print(f"  wrote {destination}")
    for row in rows:
        row["output"] = destination
    return rows, kept_by_class


def resolve_classes(args):
    """
    What to truncate, as (name, raw values) pairs.

    --label names the values outright; --classes reads them off
    config.LABEL_CLASS_MAP, the same raw-value-to-class pairing
    transforms.py makes, so a reference file is cut class by class without
    having to spell its convention out again. Neither: everything nonzero,
    as one tree, which is only right for a single-class mask.
    """
    if args.label:
        return [("label_" + "_".join(str(v) for v in args.label), list(args.label))]
    if not args.classes:
        return [("foreground", None)]

    import config as cfg  # only this branch needs the class map

    classes = []
    for index, name in enumerate(cfg.CLASS_NAMES):
        if index == 0:
            continue
        values = sorted(raw for raw, mapped in cfg.LABEL_CLASS_MAP.items() if mapped == index)
        classes.append((name, values or [index]))
    return classes


def write_csv(path, rows):
    """One row per (file, class): what was cut, and by which rule."""
    columns = [key for key in ("file", "class", "output") if any(key in row for row in rows)]
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}  ({len(rows)} rows)")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def add_cut_arguments(parser):
    """Where the tree is cut. Shared with compute_dice.py, so the two agree by construction."""
    cut = parser.add_argument_group("where to cut")
    cut.add_argument("--min-diameter", type=float, default=MIN_DIAMETER_MM, metavar="MM",
                     help="A branch thinner than this is cut off, with everything under it. This "
                          f"is what \"large vessel\" means here -- report it. Default: {MIN_DIAMETER_MM} "
                          "mm, which keeps the segmental vessels and drops the generation below")
    cut.add_argument("--max-generation", type=int, default=None, metavar="N",
                     help="Also cut below this order of --ordering. Off by default: the depth of a "
                          "branch is a property of the skeleton's topology, which one missed "
                          "junction changes, whereas its calibre is a measurement")
    cut.add_argument("--ordering", choices=("generation", "bfs_generation"), default="generation",
                     help="Ordering --max-generation counts in, both of them counting away from "
                          "the root. generation: the widest daughter continues the parent, so the "
                          "number tracks the vessel and a trunk giving off a collateral is not "
                          "renumbered. bfs_generation: raw junction count. Default: generation")
    cut.add_argument("--min-strahler", type=int, default=None, metavar="N",
                     help="Also cut below this diameter-defined Strahler order, which counts up "
                          "from the tips instead of down from the root. Off by default: it is "
                          "read off the leaves, and in vivo the leaves are wherever the "
                          "segmentation ran out of contrast, not where the tree ends")
    cut.add_argument("--sleeve", type=float, default=1.5, metavar="FACTOR",
                     help="How far from the axis, in local radii, a voxel may sit and still belong "
                          "to its branch. Under 1 it cuts into the vessel itself; well over it, a "
                          "vessel the skeleton missed is kept whole. Default: 1.5")

    return parser


def add_skeleton_arguments(parser):
    """The knobs `centerline.py` exposes, with its defaults. Shared with compute_dice.py."""
    skeleton = parser.add_argument_group("skeleton (same meaning as in centerline.py)")
    skeleton.add_argument("--min-branch-length", type=float, default=3.0,
                          help="Terminal branches shorter than this (mm) are pruned. Default: 3")
    skeleton.add_argument("--radius-factor", type=float, default=1.0,
                          help="Also prune a terminal branch shorter than this many local radii. Default: 1")
    skeleton.add_argument("--min-component-length", type=float, default=10.0,
                          help="Skeleton components shorter than this (mm) are dropped. Default: 10")
    skeleton.add_argument("--smoothing", type=int, default=20,
                          help="Laplacian smoothing iterations on the centerline. Default: 20")
    skeleton.add_argument("--spacing", type=float, default=None,
                          help="Isotropic voxel size (mm) used for skeletonization. Default: smallest "
                               "input spacing")
    skeleton.add_argument("--no-resample", action="store_true", help="Skeletonize on the input grid")
    skeleton.add_argument("--no-fill-holes", action="store_true", help="Do not fill internal cavities")
    skeleton.add_argument("--all-components", action="store_true",
                          help="Do not restrict the mask to its largest component before ordering")
    skeleton.add_argument("--keep-cycles", action="store_true",
                          help="Do not cut the loops of the skeleton. Inspection only: the orders "
                               "downstream of a loop are undefined")
    skeleton.add_argument("--root", type=int, nargs=3, metavar=("I", "J", "K"),
                          help="Voxel (input grid) closest to the trunk. Default: the widest free end")
    return parser


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_argument_group("what to truncate")
    source.add_argument("--input", help="One segmentation to cut (nifti)")
    source.add_argument("--input-dir", help="Directory of segmentations to cut, searched "
                                            "recursively for --pattern")
    source.add_argument("--pattern", default="*.nii.gz",
                        help="Filename pattern under --input-dir. Files already carrying --suffix "
                             "are skipped, so a directory can be re-run. Default: *.nii.gz")
    source.add_argument("--manifest", help="Cut the labels of a manifest split instead, in place "
                                           "(implies --classes)")
    source.add_argument("--split", default="val", help="Manifest split to cut. Default: val")
    source.add_argument("--label", type=int, nargs="+", default=None,
                        help="Raw label value(s) making up the class to cut. Default: every "
                             "nonzero voxel, as one tree")
    source.add_argument("--classes", action="store_true",
                        help="Cut every foreground class of config.LABEL_CLASS_MAP separately, "
                             "each on its own tree, into one output file. Use it on a multi-class "
                             "reference (raw 3 = artery, 4 = vein)")

    add_cut_arguments(parser)

    output = parser.add_argument_group("where to write")
    output.add_argument("--output", help="Output path for a single --input. Default: "
                                         "<input><suffix>.nii.gz next to the input")
    output.add_argument("--output-dir", help="Write the results here instead of next to each "
                                             "input, keeping the same filenames")
    output.add_argument("--suffix", default=SUFFIX,
                        help="Appended to the input stem. Change it to keep two floors side by "
                             f"side, e.g. --min-diameter 6 --suffix _large6mm. Default: {SUFFIX}")
    output.add_argument("--csv", help="What was cut out of each file, one row per class")
    output.add_argument("--overwrite", action="store_true",
                        help="Cut a file again even when its output already exists. Default: skip "
                             "it, so an interrupted run resumes")
    output.add_argument("--quiet", action="store_true", help="Only the warnings and the paths written")

    add_skeleton_arguments(parser)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    given = [name for name in ("input", "input_dir", "manifest") if getattr(args, name)]
    if len(given) != 1:
        parser.error("pass exactly one of --input, --input-dir and --manifest")
    if args.label and args.classes:
        parser.error("--label names the values to cut, --classes reads them from config.py; "
                     "pass one or the other")
    if args.output and (args.input_dir or args.manifest):
        parser.error("--output names a single file; use --output-dir for a batch")
    # build_tree reads these off args; they are not worth a flag here
    args.max_shift = None
    args.angle_offset = DIRECTION_OFFSET

    if args.manifest:
        args.classes = True
        with open(args.manifest) as handle:
            entries = [e for e in json.load(handle) if e.get("split") == args.split]
        if not entries:
            raise SystemExit(f"no entry with split={args.split} in {args.manifest}")
        paths = [e["label"] for e in entries]
    elif args.input_dir:
        paths = sorted(p for p in glob.glob(os.path.join(args.input_dir, "**", args.pattern),
                                            recursive=True)
                       if args.suffix not in os.path.basename(p))
        if not paths:
            raise SystemExit(f"no file matching {args.pattern} under {args.input_dir}")
    else:
        paths = [args.input]

    classes = resolve_classes(args)
    print(f"{len(paths)} file(s), classes: "
          + ", ".join(f"{name} ({'any nonzero' if values is None else values})"
                      for name, values in classes)
          + f", cut at {args.min_diameter} mm")

    rows = []
    for path in paths:
        destination = args.output or output_path(path, args.suffix, args.output_dir)
        if os.path.exists(destination) and not args.overwrite:
            print(f"{os.path.basename(path)}: {destination} exists, skipping (--overwrite to redo)")
            continue
        rows.extend(truncate_file(path, classes, args, destination)[0])

    if args.csv and rows:
        write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
