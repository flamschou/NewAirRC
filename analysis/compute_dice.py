# -*- coding: utf-8 -*-
"""
compute_dice.py

Scores a checkpoint's predictions against the references of a manifest
split, twice: on the whole tree, and on the large vessels alone.

The large-vessel pass truncates BOTH sides with the same rule (see
`truncate.py`: a branch is kept only if it clears --min-diameter and its
parent was kept, and the last layer of tips of what is left goes too) and
writes the truncated masks out, so the number can be traced back to the
volumes it was read on. Cutting only the reference would
count every peripheral vessel of the prediction as a false positive, and
the Dice would then measure the truncation rather than the model.

Cutting the prediction with its own tree does mean a prediction whose trunk
is broken gets a different truncation from the reference's. That is not a
defect of the metric -- a broken trunk IS the failure, and it shows up as
the missing branches it causes -- but it is why the per-case row keeps the
two truncations side by side (`kept_ml_reference` against `kept_ml_prediction`,
`n_segments_kept_*`): a case where they disagree wildly is a case to look at
before quoting its Dice.

The two cuts are therefore made against each other rather than one after the
other, because --min-diameter is a floor on a MEASUREMENT and not on the
vessel: half a voxel of partial volume and the same segmental artery measures
4.1 mm in the reference and 3.9 mm in the prediction, which cut independently
leaves it in one tree and drops it -- with its whole subtree -- from the
other. Each side is cut twice: the plain rule first, then again with a
tolerance band of --rescue-margin under the floor, in which a branch is kept
when --rescue-coverage of it is supported by the other side's first cut.
--rescue-support says what supported means: by default the share of the
branch's voxels lying inside that cut, or, with `centerline`, the share of
its axis running along an axis the other side kept -- stricter near a trunk,
whose cut is a wide sleeve, and more easily missed when the two skeletons sit
a little apart (see `truncate.py: truncate_pair`, and `sweep_rescue.py` to
put the two side by side on a cohort). It runs both ways, though in practice it is
mostly the prediction -- smoother and a touch thinner than the reference it
was trained on -- that loses branches to the floor. The rescue adds where the
two already agree, so it mostly raises dice_large: `n_segments_rescued_*` and
`rescued_ml_*` say how much of the number rests on it, and --rescue-margin 0
is the number without it.

The cut then loses its tips, and NOT SYMMETRICALLY: --peel-terminals defaults
to `1 0` here -- one terminal layer off the reference, none off the prediction
-- where `truncate.py` peels neither side. What is being scored is a model
against a HAND-DRAWN annotation, and the last layer of tips is where the two
disagree for reasons that are not the model's: an annotator stops a vessel
where the contrast goes rather than where the vessel does, the calibre that
ended a terminal run was measured where partial volume weighs most, and
whether the run ended there at all depends on the skeleton having found the
junction above it.

The asymmetry is the point. The model draws a vessel thinner than the hand
that annotated it, so the same floor already stops the prediction's tree
earlier than the reference's, and peeling both sides would take that one
asymmetry out twice -- leaving the prediction shorter than the reference it is
being compared with, which is a truncation difference scored as a model error.
Peeling the reference alone is what brings the two to the same extent.

That is a heuristic about these two objects and not a property of the metric,
so it is checked rather than believed: the summary prints the centerline each
side kept (`centerline_kept_mm_reference` against `centerline_kept_mm_prediction`
in the CSV), and near 0 for the difference is two trees of the same extent.
`--peel-terminals 1 1` peels both sides, `--peel-terminals 0` neither. The
peel drops real vessel whichever way it is set, so both values are printed
with the summary and written into every row.

Two volumes are written per pass, to say WHERE the two masks differ:

    --errors  a label map, 1 = agreement, 2 = predicted only (false
              positive), 3 = reference only (false negative). Exact, no
              parameter, and it opens straight into Slicer as a
              segmentation.
    --heat    the local Dice: at every voxel, the Dice of the two masks
              inside a --heat-window cube around it. 1 where they agree, 0
              where they do not, NaN where there is not enough vessel in
              the window to divide by (--heat-min-voxels). It is the global
              Dice decomposed in space: LOW IS BAD, which is the opposite
              convention from an error map, so do not read the two with the
              same colour scale.

`local_dice_p10` and `local_dice_median`, in the CSV, are that heat map read
at the vessels themselves -- how bad the bad regions are, next to the single
average the Dice reports. They are taken on the mask voxels and not over the
whole field: the map extends a window's reach around the masks, so most of
the voxels carrying a value sit in the fringe, where the window catches a
vessel by its edge and the local Dice is near 0 whatever the model did.

Predictions are read off the disk, where `inference.py` writes them
(`<image stem>_vascular_pred.nii.gz`). Pass --checkpoint and the missing
ones are produced here first, through inference.py's own sliding window, so
the split can be scored in one command from a checkpoint alone. A prediction
already on disk is never overwritten: it is the thing being measured, and
regenerating it silently would change the measurement halfway through a
cohort.

Usage:
    # predictions already produced by inference.py
    python -m analysis.compute_dice --manifest manifests/manifest_ct.json --split val --csv dice.csv

    # from the checkpoint, predicting whatever is missing on the way
    python -m analysis.compute_dice --manifest manifests/manifest_ct.json --split val \
                           --checkpoint "$DATASET_ROOT/checkpoints/.../last.ckpt" \
                           --csv dice.csv

    python -m analysis.compute_dice --manifest manifests/manifest_ct.json --classes artery vein \
                           --output-dir results/ --min-diameter 6

    # the cut alone, each side judged strictly on its own tree
    python -m analysis.compute_dice --manifest manifests/manifest_ct.json --rescue-margin 0

    # the tips kept, for a reference that is not a hand-drawn annotation
    python -m analysis.compute_dice --manifest manifests/manifest_ct.json --peel-terminals 0

    # both sides peeled alike, to check the asymmetric default against
    python -m analysis.compute_dice --manifest manifests/manifest_ct.json --peel-terminals 1 1
"""
import argparse
import collections
import csv
import glob
import json
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import uniform_filter

from pipeline import config as cfg
from .truncate import (SUFFIX, add_cut_arguments, add_skeleton_arguments, class_mask,
                      cut_settings, output_path, peel_layers, read_volume, settings_path, to_grid,
                      truncate_pair)
from .centerline import DIRECTION_OFFSET

# What inference.py appends to the image stem. Kept as a literal rather than
# imported, so this file does not drag torch and monai in behind it.
PRED_SUFFIX = "_vascular_pred"

# (reference, prediction): one terminal layer off the hand-drawn annotation,
# none off the model's mask.
#
# The tips are where a hand and a model disagree for reasons that are not the
# model's -- an annotator stops a vessel where the contrast goes, the calibre
# that ended a terminal run was measured where partial volume weighs most, and
# whether the run ended there at all depends on the skeleton having found the
# junction above it -- so the last layer of the reference's cut is the least
# comparable thing in it.
#
# The peel is asymmetric because the two sides are not: the model draws a
# vessel THINNER than the hand that annotated it, so the same floor already
# stops the prediction's tree earlier, and a symmetric peel would take the
# same asymmetry out twice -- once through the calibre, once through the tips
# -- leaving the prediction shorter than the reference it is compared with.
# Peeling the reference alone is what brings the two to the same extent.
#
# It is a heuristic about these two objects, not a property of the metric, and
# it is worth checking rather than believing: `--peel-terminals 1 1` peels both
# sides, `--peel-terminals 0` neither, and `centerline_kept_mm_reference`
# against `centerline_kept_mm_prediction` in the CSV -- printed with the
# summary -- says which setting actually leaves the two trees the same length. `truncate.py` peels nothing by default, since cutting
# a mask to look at it is not scoring it against a hand.
PEEL_TERMINALS = (1, 0)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def dice(reference, prediction):
    """Overlap of two boolean volumes. Two empty masks agree perfectly."""
    denominator = int(reference.sum()) + int(prediction.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(reference, prediction).sum() / denominator)


def error_map(reference, prediction):
    """1 where both agree, 2 predicted only, 3 reference only, 0 background."""
    volume = np.zeros(reference.shape, dtype=np.uint8)
    volume[prediction & ~reference] = 2
    volume[reference & ~prediction] = 3
    volume[reference & prediction] = 1
    return volume


def bounding_box(mask, margin):
    """The slices around everything true in `mask`, grown by `margin` voxels."""
    coords = np.argwhere(mask)
    low = np.maximum(coords.min(axis=0) - margin, 0)
    high = np.minimum(coords.max(axis=0) + margin + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(low, high))


def local_dice(reference, prediction, spacing, window_mm, min_voxels):
    """
    The Dice read in a sliding cube: the global one, decomposed in space.

    Every voxel carries the Dice of the two masks inside a `window_mm` cube
    centred on it, so a region where the prediction drifts shows as a cold
    patch whatever its volume -- unlike the global Dice, in which a whole
    lost lobe and a thin systematic wall error can produce the same drop.

    Box filters give it for the price of three convolutions: the window
    count cancels between numerator and denominator, so the sums can stay
    normalized. Where the window holds less than `min_voxels` of mask
    altogether the ratio is 0/0 and the voxel is left NaN rather than
    reported as a disagreement -- most of a thorax is background, and
    painting it 0 would put the whole volume at the bottom of the scale.

    Only the bounding box of the union is filtered: the rest is NaN by
    construction and filtering it would cost the volume three more copies.
    """
    size = np.maximum(np.rint(np.asarray(window_mm) / np.asarray(spacing)).astype(int), 1)
    size = size + (size % 2 == 0)  # odd, so the window is centred on its voxel
    heat = np.full(reference.shape, np.nan, dtype=np.float32)
    union = reference | prediction
    if not union.any():
        return heat

    box = bounding_box(union, size)
    a = reference[box].astype(np.float32)
    b = prediction[box].astype(np.float32)
    window = tuple(int(s) for s in size)
    overlap = uniform_filter(a * b, size=window, mode="constant")
    total = uniform_filter(a, size=window, mode="constant") + uniform_filter(b, size=window, mode="constant")

    enough = total >= float(min_voxels) / float(np.prod(size))
    local = np.full(a.shape, np.nan, dtype=np.float32)
    local[enough] = 2.0 * overlap[enough] / total[enough]
    heat[box] = local
    return heat


def score(reference, prediction, spacing, args):
    """Dice, volumes and the two difference maps of one pair of masks."""
    voxel_ml = float(np.prod(spacing)) / 1000.0
    row = {
        "dice": dice(reference, prediction),
        "reference_ml": float(reference.sum() * voxel_ml),
        "prediction_ml": float(prediction.sum() * voxel_ml),
        "true_positive_ml": float(np.logical_and(reference, prediction).sum() * voxel_ml),
        "false_positive_ml": float(np.logical_and(prediction, ~reference).sum() * voxel_ml),
        "false_negative_ml": float(np.logical_and(reference, ~prediction).sum() * voxel_ml),
    }
    errors = error_map(reference, prediction) if not args.no_errors else None
    heat = None
    if not args.no_heat:
        heat = local_dice(reference, prediction, spacing, args.heat_window, args.heat_min_voxels)
        # summarized ON the vessels, not over the whole map: the map is
        # defined in a neighbourhood of the masks, so most of the voxels
        # carrying a number sit in the fringe where the window catches a
        # vessel by its edge and the local Dice is near 0 whatever the model
        # did. Read over the field, the median would say 0.0 next to a
        # global Dice of 0.8.
        at_vessel = heat[(reference | prediction) & np.isfinite(heat)]
        row["local_dice_p10"] = float(np.percentile(at_vessel, 10)) if at_vessel.size else float("nan")
        row["local_dice_median"] = float(np.median(at_vessel)) if at_vessel.size else float("nan")
    return row, errors, heat


# --------------------------------------------------------------------------- #
# files
# --------------------------------------------------------------------------- #
def outside_fraction(mask, affine, target_shape, target_affine):
    """
    The share of a mask that falls outside another grid's field of view.

    Asked of the geometry -- where each foreground voxel centre lands in the
    target's voxel coordinates -- and not of the volume before and after the
    resampling, which folds in the loss of moving to a bigger voxel and
    would report a field-of-view problem where there is only a coarser grid.

    Anything outside is dropped rather than scored, so a Dice that comes
    with a large fraction here is a Dice on part of the tree.
    """
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        return 0.0
    world = coordinates @ affine[:3, :3].T + affine[:3, 3]
    inverse = np.linalg.inv(target_affine)
    voxels = world @ inverse[:3, :3].T + inverse[:3, 3]
    inside = np.all((voxels >= -0.5) & (voxels <= np.asarray(target_shape) - 0.5), axis=1)
    return float(1.0 - inside.mean())


def derived_path(base, tag, output_dir=None):
    """"vascular_case001.nii.gz" + "artery_errors_full" -> "vascular_case001_artery_errors_full.nii.gz"."""
    directory, filename = os.path.split(base)
    stem = filename[: -len(".nii.gz")] if filename.endswith(".nii.gz") else os.path.splitext(filename)[0]
    return os.path.join(output_dir if output_dir else directory, f"{stem}_{tag}.nii.gz")


def save(path, data, affine):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), path)
    print(f"  wrote {path}")


def prediction_path_for(image_path, pred_dir, suffix):
    """Where inference.py put the prediction of an image, per its naming rule."""
    directory, filename = os.path.split(image_path)
    stem = filename[: -len(".nii.gz")] if filename.endswith(".nii.gz") else os.path.splitext(filename)[0]
    return os.path.join(pred_dir if pred_dir else directory, f"{stem}{suffix}.nii.gz")


class Predictor:
    """
    inference.py's sliding window, loaded once and only if it is needed.

    torch and monai come in with `inference`, which costs seconds and a GPU
    context, so the import sits here rather than at the top of the file: a
    run over predictions that already exist must not pay for a model it
    never loads.

    The out-of-memory fallback is inference.py's -- the model moves to the
    CPU and stays there for the rest of the run. A cohort half predicted on
    the GPU and half on the CPU is the same arithmetic either way, and
    finishing slowly beats losing thirty cases to the one volume that did
    not fit.
    """

    def __init__(self, checkpoint, cpu=False):
        self.checkpoint = checkpoint
        self.cpu = cpu
        self.model = None
        self.device = None

    def load(self):
        import torch

        from pipeline.inference import load_model

        if self.cpu:
            self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"loading {self.checkpoint} on {self.device}")
        self.model = load_model(self.checkpoint, cfg, self.device)

    def run(self, image_path, destination):
        """Predicts one image and writes it where inference.py would."""
        import torch

        from pipeline.inference import predict_volume, save_prediction

        if self.model is None:
            self.load()
        print(f"  predicting {os.path.basename(image_path)}")
        try:
            labels, affine = predict_volume(image_path, self.model, cfg, self.device)
        except torch.cuda.OutOfMemoryError:
            print(f"  CUDA out of memory on {image_path}, retrying on CPU")
            torch.cuda.empty_cache()
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
            labels, affine = predict_volume(image_path, self.model, cfg, self.device)
        save_prediction(labels, affine, destination)
        print(f"  wrote {destination}")


def rewrite_paths(entries, rules):
    """
    Moves the manifest's absolute paths onto the root they are mounted at here.

    A manifest records where the data was when it was written -- these ones
    say /data/flamant/data/ct -- and the volumes outlive that path: a cluster
    mount, a copy on another filer, a scratch directory. Rewriting the prefix
    at read time keeps one manifest, and therefore one definition of the
    split, across all of them. Editing the file per machine instead is how
    two runs end up disagreeing about which cases are validation.

    This is the right tool when the tree MOVED: everything under the old root
    is under the new one, laid out the same way, so one prefix substitution
    is exact and every case either resolves or visibly does not. When the
    layout itself changed, there is no prefix to substitute and `relocate`
    matches on filenames instead.

    The first rule that matches a path wins; nothing is written back to the
    manifest.
    """
    for entry in entries:
        for key in ("image", "label"):
            for old, new in rules:
                if entry[key].startswith(old):
                    entry[key] = new + entry[key][len(old):]
                    break
    return entries


def parse_rewrite(values):
    """--rewrite OLD=NEW, repeatable, into (old, new) pairs."""
    rules = []
    for value in values or ():
        if "=" not in value:
            raise SystemExit(f"--rewrite wants OLD=NEW, got '{value}'")
        old, new = value.split("=", 1)
        rules.append((old, new))
    return rules


def relocate(entries, root):
    """
    Finds the manifest's files under `root`, by filename.

    The fallback to `rewrite_paths` for when the layout changed and not just
    the root, so no prefix substitution exists: the manifest still says the
    thing only it knows, which case is validation, and `root` says where the
    files are now -- the same way inference.py takes --input-dir rather than
    trusting a path written elsewhere. Prefer --rewrite when the tree simply
    moved; a prefix is exact, whereas a filename match is a guess that
    happens to be safe here because these names carry their patient id.

    Matching is on the filename, over one recursive walk. A name found twice
    is a hard error rather than a guess: two patients holding a
    `vascular.nii.gz` each would otherwise be silently collapsed onto
    whichever the walk reached first, and the split would quietly stop
    meaning what it says.
    """
    found = {}
    duplicates = collections.defaultdict(list)
    for path in glob.glob(os.path.join(root, "**", "*.nii.gz"), recursive=True):
        name = os.path.basename(path)
        if name in found:
            duplicates[name].append(path)
        else:
            found[name] = path

    wanted = {os.path.basename(entry[key]) for entry in entries for key in ("image", "label")}
    clashing = sorted(name for name in duplicates if name in wanted)
    if clashing:
        raise SystemExit(
            f"--data-dir {root} holds several files called " + ", ".join(clashing[:3])
            + (f" (and {len(clashing) - 3} more)" if len(clashing) > 3 else "")
            + ";\nthe manifest names files, not directories, so which one is meant "
              "cannot be decided here")

    missing = []
    for entry in entries:
        for key in ("image", "label"):
            name = os.path.basename(entry[key])
            if name in found:
                entry[key] = found[name]
            else:
                missing.append(name)
    if missing:
        print(f"{len(missing)} file(s) of the split are not under {root}, "
              f"e.g. {', '.join(missing[:3])}")
    return entries


def class_pairs(names, swap_av=False):
    """
    (class name, raw values in a reference, values in a prediction).

    The two are not the same: a reference carries the raw values of the
    generator (config.LABEL_CLASS_MAP: 3 = artery, 4 = vein) and a
    prediction the training class indices (1 = artery, 2 = vein). Pairing
    them here is the same pairing transforms.py makes at training time.

    `swap_av` corrects a checkpoint trained with the inverted convention by
    reading the artery out of the vein index and back, which is a change of
    which VALUE is read -- nothing is rewritten, so the prediction file that
    gets truncated is still the one on disk.
    """
    pairs = []
    for name in names:
        if name not in cfg.CLASS_NAMES:
            raise SystemExit(f"unknown class '{name}'; config.CLASS_NAMES has "
                             f"{cfg.CLASS_NAMES[1:]}")
        index = cfg.CLASS_NAMES.index(name)
        raw = sorted(value for value, mapped in cfg.LABEL_CLASS_MAP.items() if mapped == index)
        predicted = index
        if swap_av and name in ("artery", "vein"):
            predicted = cfg.CLASS_NAMES.index("vein" if name == "artery" else "artery")
        pairs.append((name, raw or [index], [predicted]))
    return pairs


def reusable(destination, classes, args, source, counterpart=None, peel=None):
    """
    Whether the cut already on disk was made by the rule being asked for.

    The filename cannot answer this: `..._large.nii.gz` is the name of a 4 mm
    cut of the artery and of a 6 mm cut of both trees alike, and reusing one
    for the other scores a class that was never cut -- two empty masks, which
    Dice reports as a perfect 1.0. The sidecar `truncate.py` writes next to
    every mask carries the rule AND a fingerprint of the file it was cut
    from, so a mask rewritten under the same name -- a reference resampled
    onto the image grid, a prediction from another checkpoint -- invalidates
    its cut instead of quietly outliving it. No sidecar, no reuse.

    `counterpart` is the other side of the pair the cut was made against,
    which the sidecar fingerprints too: a rescued cut belongs to the file it
    was rescued against as much as to the file it was cut from.

    `peel` is this side's --peel-terminals, which the two sides of a pair do
    not share (PEEL_TERMINALS): a cut of the reference made when the peel was
    symmetric is not the cut being asked for now, and the sidecar is what says
    so instead of it being reused under the same name.
    """
    sidecar = settings_path(destination)
    if not (os.path.exists(destination) and os.path.exists(sidecar)):
        return False, "not cut yet"
    with open(sidecar) as handle:
        stored = json.load(handle)
    wanted = cut_settings(classes, args, source, counterpart, peel)
    if stored == wanted:
        return True, ""
    differing = sorted(key for key in set(stored) | set(wanted)
                       if stored.get(key) != wanted.get(key))
    if differing == ["source"]:
        return False, "cut from an older version of the file"
    if differing == ["rescue"] and stored.get("rescue") and wanted.get("rescue") and all(
            stored["rescue"].get(key) == value
            for key, value in wanted["rescue"].items() if key != "against"):
        return False, "cut against an older version of the other side"
    if differing:
        return False, "cut by another rule (" + ", ".join(differing) + ")"
    # The dicts are unequal yet no key's value differs: one side carries keys
    # the other does not, and `.get` reads both as None. That is a change to
    # the SCHEMA -- a setting added to or dropped from cut_settings -- not to
    # this cut, and saying "cut by another rule ()" would send someone looking
    # for a rule that did not move. Name the keys instead.
    added = sorted(set(wanted) - set(stored))
    dropped = sorted(set(stored) - set(wanted))
    return False, ("cut before the settings schema changed ("
                   + ", ".join(["+" + key for key in added] + ["-" + key for key in dropped])
                   + ")")


def truncated_pair(reference_path, reference_classes, prediction_path, prediction_classes,
                   args, verbose=True):
    """
    The truncated masks of both sides, computing them if the cuts on disk
    are not the ones being asked for.

    Reusing an existing cut is not just a speed-up: it is what lets a run be
    repeated, or a Dice be recomputed with different maps, without the
    skeletonization -- and therefore without any chance of the second run
    cutting somewhere slightly different from the first.

    The pair is reused, or recut, as a unit. With a rescue margin each cut
    depends on the other side, so half a pair reused next to a half just
    remade would be a cut against a file that is no longer there; and recutting
    one side needs the other side's tree anyway, which is the expensive part.
    """
    destinations = (output_path(reference_path, args.suffix, args.output_dir),
                    output_path(prediction_path, args.suffix, args.output_dir))
    peel = peel_layers(args)
    states = (reusable(destinations[0], reference_classes, args, reference_path, prediction_path,
                       peel[0]),
              reusable(destinations[1], prediction_classes, args, prediction_path, reference_path,
                       peel[1]))
    if all(reuse for reuse, _ in states) and not args.overwrite:
        kept = []
        for destination, classes in zip(destinations, (reference_classes, prediction_classes)):
            if verbose:
                print(f"  reusing {destination}")
            data = read_volume(destination)[0]
            kept.append({name: class_mask(data, values) for name, values in classes})
        return destinations, kept[0], kept[1], []
    for destination, (reuse, reason) in zip(destinations, states):
        if not reuse and reason != "not cut yet":
            print(f"  {os.path.basename(destination)}: {reason}, cutting the pair again")
    rows, reference_kept, prediction_kept = truncate_pair(
        reference_path, reference_classes, prediction_path, prediction_classes, args,
        destinations, verbose=verbose)
    return destinations, reference_kept, prediction_kept, rows


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def write_csv(path, rows):
    columns = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}  ({len(rows)} rows)")


def describe(values):
    """Mean, SD and count of the values that are not NaN."""
    values = [v for v in values if v == v]
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0, len(array)


def peeling(args):
    """--peel-terminals as a phrase, or "" when nothing is peeled."""
    reference, prediction = peel_layers(args)
    if not (reference or prediction):
        return ""
    if reference == prediction:
        return f"{reference} terminal layer(s) peeled off both sides"
    return f"{reference} terminal layer(s) peeled off the reference and {prediction} off the " \
           f"prediction"


def print_summary(rows, args):
    """The two Dices over the split, per class."""
    classes = []
    for row in rows:
        if row["class"] not in classes:
            classes.append(row["class"])

    cases = len({row["case"] for row in rows})
    rescuing = args.rescue_margin > 0
    print(f"\n{cases} case(s) scored on {len(classes)} class(es), "
          f"large vessels cut at {args.min_diameter} mm"
          + (f", {peeling(args)}" if peeling(args) else "")
          + (f", branches down to {args.min_diameter - args.rescue_margin} mm kept when the other "
             f"side held them" if rescuing else ""))
    print(f"{'class':<10}{'n':>4}{'dice (whole)':>18}{'dice (large)':>18}{'kept volume':>14}"
          + (f"{'rescued ref/pred':>20}" if rescuing else ""))
    for name in classes:
        subset = [row for row in rows if row["class"] == name]
        full = describe([row["dice_full"] for row in subset])
        large = describe([row["dice_large"] for row in subset])
        kept = describe([row["kept_fraction_reference"] for row in subset])
        line = (f"{name:<10}{full[2]:>4}"
                f"{f'{full[0]:.4f}+-{full[1]:.3f}':>18}"
                f"{f'{large[0]:.4f}+-{large[1]:.3f}':>18}"
                f"{kept[0]:>13.1%}")
        if rescuing:
            reference = describe([row.get("n_segments_rescued_reference") or 0 for row in subset])
            prediction = describe([row.get("n_segments_rescued_prediction") or 0 for row in subset])
            line += f"{f'{reference[0]:.1f} / {prediction[0]:.1f}':>20}"
        print(line)
    print("\nThe two columns are read on different regions -- the large-vessel one is the\n"
          "easier region by construction -- so they compare models, not each other.\n"
          "Report --min-diameter with the second one.")
    reference_peel, prediction_peel = peel_layers(args)
    if reference_peel or prediction_peel:
        lengths = describe([row["centerline_kept_mm_reference"]
                            - row["centerline_kept_mm_prediction"] for row in rows
                            if row.get("centerline_kept_mm_reference") is not None
                            and row.get("centerline_kept_mm_prediction") is not None])
        print(f"\nThe cuts lost their tips: --peel-terminals {reference_peel} {prediction_peel}. "
              f"A tip is where the calibre\nwas measured worst and where a hand and a model agree "
              f"least -- an annotator stops a\nvessel where the contrast goes, not where the "
              f"vessel does.")
        if reference_peel != prediction_peel:
            print("The two sides are peeled differently because they are not the same kind of\n"
                  "object: a model draws a vessel thinner than a hand does, so the floor has\n"
                  "already stopped its tree earlier, and peeling it again would take one\n"
                  "asymmetry out twice.")
        if lengths:
            others = [f"--peel-terminals {a} {b}" for a, b in
                      ((reference_peel, 0), (reference_peel, reference_peel), (0, 0))
                      if (a, b) != (reference_peel, prediction_peel)]
            print(f"Whether it worked is the centerline left standing: reference minus prediction "
                  f"is\n{lengths[0]:+.0f} mm on average ({lengths[1]:.0f} SD). Near 0 is two trees "
                  f"of the same extent, which is\nwhat the peel is for; far from it, "
                  + " or ".join(others) + "\nis the same cohort read the other ways.")
        print("It drops real vessel either way: report --peel-terminals with the number.")
    if rescuing:
        print(f"The rescued segments are branches under the floor kept because the other side\n"
              f"kept them, so they mostly add agreement: --rescue-margin 0 for the large-vessel\n"
              f"Dice without them. Report --rescue-margin ({args.rescue_margin} mm) with it.")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    data = parser.add_argument_group("what to score")
    data.add_argument("--manifest", default=cfg.MANIFEST_PATH,
                      help=f"Manifest train.py was run on. Default: config.MANIFEST_PATH "
                           f"({cfg.MANIFEST_PATH})")
    data.add_argument("--split", default="val", help="Split to score. Default: val")
    data.add_argument("--classes", nargs="+", default=["artery"], metavar="NAME",
                      help="Classes to score, each truncated on its own tree -- the union of the "
                           "arterial and the venous tree is not a tree. Default: artery")
    data.add_argument("--pred-dir", default=None,
                      help="Directory holding the predictions. Default: next to each image, where "
                           "inference.py writes them")
    data.add_argument("--pred-suffix", default=PRED_SUFFIX,
                      help=f"Suffix inference.py appended to the image stem. Default: {PRED_SUFFIX}")
    data.add_argument("--rewrite", action="append", metavar="OLD=NEW",
                      help="Replace the prefix OLD with NEW in every path of the manifest, for a "
                           "cohort read from another mount than the one it was written on. "
                           "Repeatable; the first matching rule wins; the manifest is not "
                           "modified. e.g. --rewrite /data/flamant/data/ct=/biomaps/spiro3d/ct")
    data.add_argument("--data-dir", help="Fallback for when the layout changed and not just the "
                                         "root, so no prefix substitution exists: where the "
                                         "manifest's volumes actually are, searched "
                                         "recursively and matched by filename, for a cohort read "
                                         "from another mount than the one the manifest was written "
                                         "on. The manifest still decides which cases are in the "
                                         "split; this only says where they live, as inference.py's "
                                         "--input-dir does. The manifest is not modified")
    data.add_argument("--checkpoint", help="Run inference for every case whose prediction is "
                                           "missing, with this .ckpt, and write it where "
                                           "inference.py would. Predictions already on disk are "
                                           "reused, never overwritten")
    data.add_argument("--cpu", action="store_true",
                      help="Force --checkpoint inference onto the CPU, even with a GPU available")
    data.add_argument("--swap-av", action="store_true",
                      help="Swap the artery and vein indices of the predictions before scoring, "
                           "for a checkpoint trained with the inverted convention (see "
                           "compare_predictions.py --swap-av-a)")

    add_cut_arguments(parser, peel_terminals=PEEL_TERMINALS)

    rescue = parser.add_argument_group("the threshold effect on the cut")
    rescue.add_argument("--rescue-margin", type=float, default=2.0, metavar="MM",
                        help="Tolerance band under --min-diameter, in which a branch is kept "
                             "anyway when the other side's cut holds the same vessel -- so a "
                             "vessel measuring 4.1 mm in the reference and 3.9 mm in the "
                             "prediction is not counted as a miss, with its whole subtree, on a "
                             "rounding difference. Both ways, and only through the closure: a "
                             "rescued branch still hangs off one that cleared the floor. 0 cuts "
                             "each side strictly on its own tree. Default: 2.0")
    rescue.add_argument("--rescue-coverage", type=float, default=0.5, metavar="FRACTION",
                        help="How much of a branch in that band must be supported by the other "
                             "side's cut for the two to be the same vessel. Default: 0.5")
    rescue.add_argument("--rescue-support", choices=("mask", "centerline"), default="mask",
                        help="What supported means. mask: the share of the branch's voxels lying "
                             "inside the other side's cut -- generous near a trunk, whose cut is a "
                             "sleeve of --sleeve local radii wide enough to hold a thin vessel "
                             "that merely runs beside it. centerline: the share of its axis "
                             "running along an axis the other side kept, which separates a vessel "
                             "from its large neighbour but misses matches when the two skeletons "
                             "sit apart, as two different grids are enough to cause. Default: mask")
    rescue.add_argument("--rescue-distance", type=float, default=1.5, metavar="RADII",
                        help="--rescue-support centerline only: how far the other side's axis may "
                             "sit from a branch's own, in that branch's local radii. The nearest "
                             "centerline decides, dropped branches included, so a vessel the other "
                             "side has and cut answers for itself instead of handing the question "
                             "to the nearest trunk. Default: 1.5")

    maps = parser.add_argument_group("where the two masks differ")
    maps.add_argument("--no-errors", action="store_true",
                      help="Do not write the 1/2/3 agreement/false-positive/false-negative map")
    maps.add_argument("--no-heat", action="store_true", help="Do not write the local-Dice heat map")
    maps.add_argument("--heat-window", type=float, default=20.0, metavar="MM",
                      help="Side of the cube the local Dice is read in. Too small and it is binary "
                           "noise, too large and it is the global Dice again. Default: 20")
    maps.add_argument("--heat-min-voxels", type=int, default=20, metavar="N",
                      help="A window holding fewer voxels of mask than this is left NaN instead of "
                           "being scored on a handful of voxels. Default: 20")

    output = parser.add_argument_group("where to write")
    output.add_argument("--output-dir", help="Write every volume here. Default: next to the file "
                                             "each one derives from")
    output.add_argument("--suffix", default=SUFFIX,
                        help=f"Appended to the stem of the truncated masks. Default: {SUFFIX}")
    output.add_argument("--csv", help="One row per case and class")
    output.add_argument("--overwrite", action="store_true",
                        help="Truncate again even when the cut mask already exists. Default: reuse it")
    output.add_argument("--quiet", action="store_true", help="Only the warnings and the paths written")

    add_skeleton_arguments(parser)
    return parser


def main():
    args = build_parser().parse_args()
    # one value or two, normalized once so nothing downstream has to ask
    args.peel_terminals = list(peel_layers(args))
    # build_tree reads these off args; they are not worth a flag here
    args.max_shift = None
    args.angle_offset = DIRECTION_OFFSET

    with open(args.manifest) as handle:
        entries = [e for e in json.load(handle) if e.get("split") == args.split]
    if not entries:
        raise SystemExit(f"no entry with split={args.split} in {args.manifest}")
    entries = rewrite_paths(entries, parse_rewrite(args.rewrite))
    if args.data_dir:
        entries = relocate(entries, args.data_dir)
    classes = class_pairs(args.classes, args.swap_av)
    print(f"{len(entries)} case(s) in split '{args.split}', "
          + ", ".join(f"{name}: raw {raw} against predicted {predicted}"
                      for name, raw, predicted in classes)
          + f", large vessels cut at {args.min_diameter} mm"
          + (f", {peeling(args)}" if peeling(args) else ""))

    predictor = Predictor(args.checkpoint, args.cpu) if args.checkpoint else None
    skipped = collections.Counter()
    rows = []
    for entry in entries:
        case = os.path.basename(entry["label"])
        prediction_file = prediction_path_for(entry["image"], args.pred_dir, args.pred_suffix)
        if not os.path.exists(prediction_file):
            if predictor is None:
                print(f"{case}: no prediction at {prediction_file}, skipping "
                      f"(--checkpoint to produce it)")
                skipped["no prediction, and no --checkpoint to produce it"] += 1
                continue
            if not os.path.exists(entry["image"]):
                print(f"{case}: no image at {entry['image']}, skipping")
                skipped["the manifest's image path does not exist here"] += 1
                continue
            print(f"{case}")
            predictor.run(entry["image"], prediction_file)

        reference_data, affine, spacing = read_volume(entry["label"])
        prediction_data, prediction_affine, prediction_spacing = read_volume(prediction_file)

        # The two need not be sampled the same way: the labels come out of
        # the generator, the synthetic images out of a second pipeline, and
        # `inference.py` maps its output back onto the IMAGE. A viewer hides
        # this -- it overlays in world coordinates and resamples for display,
        # so two volumes describing the same anatomy superimpose perfectly
        # while their arrays have different shapes -- but a Dice is read
        # element by element and needs one grid, and `truncate.to_grid` moves
        # a mask onto the other's.
        #
        # The reference is the side that moves, onto the prediction's grid,
        # for two reasons. It is what training compares against
        # (`transforms.py`: Spacingd on the image, then
        # `ResampleToMatchd(keys=["label"], key_dst="image",
        # mode="nearest")`), so the metric is read in the space the model was
        # asked to answer in. And it is the direction that cannot invent
        # errors: were the prediction moved onto a reference covering more
        # ground, everything the model was never shown would come back as a
        # false negative.
        #
        # Each side is still CUT on its own grid: the skeleton deserves the
        # resolution its file was written at, and a `_large` mask sitting
        # next to its source should have its source's geometry. Nothing is
        # resampled to cut them against each other either -- `truncate_pair`
        # matches the two centerlines in world millimetres.
        regrid = not (reference_data.shape == prediction_data.shape
                      and np.allclose(affine, prediction_affine, atol=1e-3))
        scoring_affine = prediction_affine if regrid else affine
        scoring_spacing = prediction_spacing if regrid else spacing
        if regrid and not args.quiet:
            print(f"{case}: reference {reference_data.shape} and prediction "
                  f"{prediction_data.shape} are sampled differently; the reference is "
                  f"resampled onto the prediction's grid, as training does")

        # both sides truncated by the same rule, against each other, and both
        # written out
        reference_classes = [(name, raw) for name, raw, _ in classes]
        prediction_classes = [(name, predicted) for name, _, predicted in classes]
        cut_files, reference_cut, prediction_cut, cut_rows = truncated_pair(
            entry["label"], reference_classes, prediction_file, prediction_classes, args,
            verbose=not args.quiet)
        reference_cut_file, prediction_cut_file = cut_files
        cut_rows = {(r["class"], r["file"]): r for r in cut_rows}

        for name, raw, predicted in classes:
            print(f"{case} [{name}]")
            row = {"case": case, "class": name, "reference": entry["label"],
                   "prediction": prediction_file, "predicted_value": predicted[0],
                   "min_diameter_mm": args.min_diameter,
                   "peel_terminals_reference": peel_layers(args)[0],
                   "peel_terminals_prediction": peel_layers(args)[1],
                   "rescue_margin_mm": args.rescue_margin}
            prediction_whole = class_mask(prediction_data, predicted)
            pairs = (("full", class_mask(reference_data, raw), prediction_whole),
                     ("large", reference_cut[name], prediction_cut[name]))
            moved = {}
            for tag, reference, prediction in pairs:
                if regrid:
                    outside = outside_fraction(reference, affine, prediction_data.shape,
                                               prediction_affine)
                    if outside > 0.01:
                        print(f"  WARNING: {outside:.1%} of the {tag} reference lies outside the "
                              f"prediction's field of view; it is dropped, not scored, so the "
                              f"Dice below covers {1 - outside:.1%} of the tree")
                    reference = to_grid(reference, affine, prediction_data.shape, prediction_affine)
                moved[tag] = (reference, prediction)
                scored, errors, heat = score(reference, prediction, scoring_spacing, args)
                row.update({f"{key}_{tag}": value for key, value in scored.items()})
                if errors is not None:
                    save(derived_path(entry["label"], f"{name}_errors_{tag}", args.output_dir),
                         errors, scoring_affine)
                if heat is not None:
                    save(derived_path(entry["label"], f"{name}_localdice_{tag}", args.output_dir),
                         heat, scoring_affine)

            # Of the large-vessel false negatives, the share the model did
            # segment and the truncation then removed: a vessel the prediction
            # calls half a voxel thinner than the reference falls under the
            # floor, leaves with its whole subtree by the closure, and shows up
            # as a miss it is not. High here means dice_large is reading the
            # cut as much as the model -- see --min-diameter and the gap
            # between n_segments_kept_reference and _prediction.
            #
            # This is what --rescue-margin exists to remove, so it should now
            # be small; what is left of it is the vessels the rescue could not
            # reach -- further than a margin under the floor, or hanging off a
            # branch that was itself cut.
            reference_large, prediction_large = moved["large"]
            missed = reference_large & ~prediction_large
            row["fn_large_segmented_but_cut"] = (
                float((missed & prediction_whole).sum()) / float(missed.sum())
                if missed.any() else 0.0)

            reference_row = cut_rows.get((name, entry["label"]), {})
            prediction_row = cut_rows.get((name, prediction_file), {})
            row.update({
                "reference_cut": reference_cut_file,
                "prediction_cut": prediction_cut_file,
                "kept_fraction_reference": row["reference_ml_large"] / row["reference_ml_full"]
                if row["reference_ml_full"] else float("nan"),
                "kept_fraction_prediction": row["prediction_ml_large"] / row["prediction_ml_full"]
                if row["prediction_ml_full"] else float("nan"),
                "n_segments_kept_reference": reference_row.get("n_segments_kept"),
                "n_segments_kept_prediction": prediction_row.get("n_segments_kept"),
                # the length of what each side kept, in world mm, so it is
                # comparable across the two grids: this is what says whether
                # --peel-terminals brought the two trees to the same extent
                "centerline_kept_mm_reference": reference_row.get("centerline_kept_mm"),
                "centerline_kept_mm_prediction": prediction_row.get("centerline_kept_mm"),
                "n_segments_rescued_reference": reference_row.get("n_segments_rescued"),
                "n_segments_rescued_prediction": prediction_row.get("n_segments_rescued"),
                # of those, the ones the tolerance band admitted; the rest
                # rejoined behind them by the ordinary rule
                "n_segments_in_band_reference": reference_row.get("n_segments_in_band"),
                "n_segments_in_band_prediction": prediction_row.get("n_segments_in_band"),
                "rescued_ml_reference": reference_row.get("rescued_ml"),
                "rescued_ml_prediction": prediction_row.get("rescued_ml"),
            })
            print(f"  dice: {row['dice_full']:.4f} (whole tree)   "
                  f"{row['dice_large']:.4f} (large vessels)")
            rows.append(row)

    if not rows:
        why = "; ".join(f"{count} case(s): {reason}" for reason, count in skipped.most_common())
        hint = ""
        if skipped["the manifest's image path does not exist here"]:
            root = os.path.commonpath([os.path.dirname(e["image"]) for e in entries])
            hint = (f"\n{args.manifest} points its data at {root}, which is not readable from "
                    f"here.\nMap it onto the root they are under now:\n"
                    f"    --rewrite {root}=/the/root/they/are/under\n"
                    f"or, if the layout changed too and no prefix fits, --data-dir DIR to find "
                    f"them by name.")
        raise SystemExit(f"no case scored -- {why}{hint}")
    print_summary(rows, args)
    if args.csv:
        write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
