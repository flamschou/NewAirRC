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

Two knobs cut the result back further, both acting on the tips of the KEPT
tree -- the runs past its last bifurcation -- because that is where a cut is
least reproducible: the calibre that ended a run was measured where partial
volume weighs most, and whether it ended there at all depends on the
skeleton having found the junction above it. `--peel-terminals` drops those
runs whole, one layer at a time, leaving a tree whose every tip is a
bifurcation both sides saw; `--max-terminal-length` only shortens them.
Both are off by default HERE: this file cuts a mask so it can be looked at,
and the tips are vessel. The scoring tools that share these arguments
(`compute_dice.py`, `sweep_rescue.py`) peel the REFERENCE by one layer
instead, and only the reference, because they read a model against a
hand-drawn annotation: the tips are where the two disagree for reasons that
are not the model's, and the model's own tree has already been shortened by
the floor, its vessels being drawn thinner. --peel-terminals therefore takes
one value for both sides of a pair or two, reference then prediction. Either
way it drops vessel the two sides may well have agreed on, and like the floor
it has to be reported.

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
    python -m analysis.truncate --input vascular_case001.nii.gz --label 3

    # every class of config.LABEL_CLASS_MAP, one file per case
    python -m analysis.truncate --input vascular_case001.nii.gz --classes

    # the validation split of a manifest, references truncated in place
    python -m analysis.truncate --manifest manifests/manifest_ct.json --split val

    # the predictions that go with them, cut by the same rule
    python -m analysis.truncate --input-dir /data/flamant/data/ct/lidc_idri \
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

from .centerline import (DIRECTION_OFFSET, build_tree, resample_isotropic, skeletonize_graph,
                        trunk_calibre)

# A segmental pulmonary artery leaves its lobar parent at roughly 4-6 mm and
# the subsegmentals under it at 2-3. A 4 mm floor sits at the BOTTOM of the
# segmental range, which puts the truncation boundary inside that generation --
# exactly where two segmentations of one tree disagree most, since half a
# millimetre of wall drops a branch there to either side of the floor. 5 mm
# sits inside the range rather than on its edge, so the boundary falls on
# vessels both sides agree are large.
#
# It is the definition of "large vessel" here and it is a knob, not a fact:
# report the value used with whatever the truncated masks end up scoring.
# `sweep_rescue.py --floors` is what moving it does to a cohort, and the only
# operational test is two models -- the right floor is the one where they
# still separate.
MIN_DIAMETER_MM = 5.0
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


def to_grid(mask, affine, target_shape, target_affine):
    """
    Nearest-neighbour resample of a boolean mask onto another grid, by affine.

    Two masks of the same case need not be sampled the same way -- a
    reference comes out of the generator, a prediction out of
    `inference.py`, which maps its output back onto the IMAGE -- and a
    viewer hides it, overlaying in world coordinates so two volumes
    describing the same anatomy superimpose perfectly while their arrays
    have different shapes. Anything read element by element needs one grid.

    Nearest neighbour, never an interpolation: an averaged label value is
    not a label.
    """
    from nibabel.processing import resample_from_to

    image = nib.Nifti1Image(mask.astype(np.uint8), affine)
    resampled = resample_from_to(image, (tuple(target_shape), target_affine), order=0, cval=0)
    return np.asarray(resampled.dataobj) > 0


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


def subdivide(table, world, radii, voxel_size, step):
    """
    Cuts every branch into pieces of at most `step` mm, so the truncation can
    fall INSIDE a branch instead of only between branches.

    Why the branch is the wrong object to decide on. `calibre_mm` is one
    median over a whole branch (`centerline.trunk_calibre`), and a branch
    runs from one bifurcation to the next -- which is a property of the
    SKELETON, not of the anatomy. Two segmentations of the same tree do not
    bifurcate in the same places: wherever one has a small side branch the
    other missed, it has a junction the other has not, and the vessel that
    the first describes with three segments the second describes with one
    long one. That long branch gets a single calibre averaging its 5 mm
    proximal end with its 2.5 mm distal one, lands under the floor, and
    leaves with its whole subtree, while the other side keeps the proximal
    two thirds of the same vessel because they happen to be their own
    segments. The disagreement is then read as a false negative at every
    branch tip of the tree, and no rescue can undo it: the rescue argues
    about which branches to keep, and the branch is what is wrong.

    Splitting at a fixed arc length gives both sides the same granularity in
    millimetres whatever their bifurcations do. A piece keeps the boundary
    node it shares with the next one, so consecutive pieces are neighbours
    by the rule `select_branches` already uses and the closure walks a
    branch from its proximal end to wherever the calibre drops.

    The calibre of a piece is `trunk_calibre` over that piece alone. The
    junction trim is applied only where there is a junction -- the two ends
    of the ORIGINAL branch -- since a split point invented here carries no
    bifurcation blob to cut away.

    `step` of 0 returns the table untouched, which is the old behaviour: one
    decision per branch.
    """
    if step <= 0:
        return table

    ends = defaultdict(int)
    for entry in table:
        ends[entry["nodes"][0]] += 1
        ends[entry["nodes"][-1]] += 1

    pieces = []
    for entry in table:
        nodes = np.asarray(entry["nodes"])
        points = world[nodes]
        walked = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
        total = float(walked[-1])
        count = max(1, int(np.ceil(total / step))) if total > 0 else 1

        cuts = [0]
        for k in range(1, count):
            index = int(np.searchsorted(walked, k * total / count))
            if cuts[-1] < index < len(nodes) - 1:
                cuts.append(index)
        cuts.append(len(nodes) - 1)

        # a free end carries no junction blob, and neither does a split point
        head_junction = float(radii[nodes[0]]) if ends[entry["nodes"][0]] > 1 else 0.0
        tail_junction = float(radii[nodes[-1]]) if ends[entry["nodes"][-1]] > 1 else 0.0
        for k in range(len(cuts) - 1):
            start, stop = cuts[k], cuts[k + 1]
            piece = nodes[start:stop + 1]
            pieces.append({
                "branch_id": len(pieces),
                "source_branch_id": entry["branch_id"],
                "nodes": piece,
                "n_points": len(piece),
                "length_mm": float(walked[stop] - walked[start]),
                "calibre_mm": trunk_calibre(
                    piece, world, radii,
                    head_junction if k == 0 else 0.0,
                    tail_junction if k == len(cuts) - 2 else 0.0, voxel_size),
                "generation": entry["generation"],
                "bfs_generation": entry["bfs_generation"],
                "strahler_dd": entry["strahler_dd"],
            })
    return pieces


def select_branches(table, min_diameter, max_generation=None, ordering="generation",
                    min_strahler=None, margin=0.0, supported=None):
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

    `margin` opens a tolerance band under the floor: a branch whose diameter
    falls within `margin` mm of `min_diameter` is admitted anyway when
    `supported(branch_id)` says so -- in `truncate_pair`, when the other
    side of a comparison kept that same vessel. `min_diameter` is a floor on
    a MEASUREMENT of a segmentation and not on the vessel, and half a voxel
    of partial volume moves a segmental artery either side of it. The band
    is only ever entered through the closure, so a rescued branch still has
    to hang off something that cleared the floor on its own, and the
    branches beyond it rejoin by the ordinary rule -- what comes back is the
    subtree, not the one borderline segment.

    The seed is never rescued: a tree whose widest branch is under the floor
    has nothing to grow from, and no support makes that a truncation.
    """
    def in_limits(entry):
        if max_generation is not None and not 0 <= entry[ordering] <= max_generation:
            return False
        if min_strahler is not None and entry["strahler_dd"] < min_strahler:
            return False
        return True

    def qualifies(entry):
        return 2.0 * entry["calibre_mm"] >= min_diameter and in_limits(entry)

    def admissible(entry):
        if qualifies(entry):
            return True
        # fails on the calibre alone, and only just
        if supported is None or margin <= 0.0:
            return False
        if 2.0 * entry["calibre_mm"] < min_diameter - margin:
            return False
        return in_limits(entry) and supported(entry["branch_id"])

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
                if neighbour not in keep and admissible(table[neighbour]):
                    keep.add(neighbour)
                    queue.append(neighbour)
    return keep


def kept_tree(table, keep):
    """
    A selection, oriented: (seed, breadth-first order, parent, children).

    The topology the two trimming rules below are read on is the topology of
    the CUT, not of the tree it was cut from. A branch that divides in the
    full tree is a tip once everything under it has gone, and it is as a tip
    that it is unreliable -- so the parent-child relation has to be rebuilt
    on `keep` alone, every time `keep` changes.

    Rebuilt from the same seed the closure was grown from -- the widest
    branch, not the root, for the reason `select_branches` gives -- and
    through the same neighbour rule, two pieces being adjacent when they
    share an end node.
    """
    at_node = defaultdict(list)
    for branch in keep:
        entry = table[branch]
        at_node[entry["nodes"][0]].append(branch)
        at_node[entry["nodes"][-1]].append(branch)

    seed = max(keep, key=lambda branch: table[branch]["calibre_mm"])
    order, entered, parent, children = [seed], {seed: None}, {seed: None}, defaultdict(list)
    queue = deque(order)
    while queue:
        branch = queue.popleft()
        entry = table[branch]
        for end in (entry["nodes"][0], entry["nodes"][-1]):
            if end == entered[branch]:
                continue
            for neighbour in at_node[end]:
                if neighbour not in entered:
                    entered[neighbour], parent[neighbour] = end, branch
                    children[branch].append(neighbour)
                    order.append(neighbour)
                    queue.append(neighbour)
    return seed, order, parent, children


def terminal_runs(table, keep):
    """
    The terminal runs of a selection: every chain of pieces that leaves the
    last bifurcation of the KEPT tree and reaches a tip without dividing
    again, listed head first.

    A run, not a piece, is the object both trimming rules act on. `--cut-step`
    cuts branches into pieces so that the CALIBRE is judged per millimetre,
    and a piece is the right unit for that because a calibre is a local
    measurement. Being terminal is not: it is a property of the whole branch
    between two bifurcations, and the last piece of a run is terminal only in
    the sense that the run is.

    The run carrying the seed is not returned. On a healthy tree it is not
    terminal in the first place, since the trunk divides; when it is -- a cut
    that came back as a single chain, a tree that fell apart -- the trunk is
    the last thing worth removing, and it is what keeps a peel from ever
    emptying a selection.
    """
    if not keep:
        return []

    seed, order, parent, children = kept_tree(table, keep)

    # nothing below a terminal run divides
    terminal = {}
    for branch in reversed(order):
        below = children[branch]
        terminal[branch] = not below or (len(below) == 1 and terminal[below[0]])

    runs = []
    for branch in order:
        # the head of a run is the first piece past the last bifurcation
        if branch == seed or not terminal[branch] or terminal[parent[branch]]:
            continue
        run, piece = [branch], branch
        while children[piece]:
            piece = children[piece][0]
            run.append(piece)
        runs.append(run)
    return runs


def peel_layers_of(peel):
    """
    --peel-terminals as (reference side, prediction side).

    One value peels both sides of a pair alike, two peel them separately.
    The asymmetric form exists because the two sides of a scoring pair are
    not the same kind of object -- see `compute_dice.PEEL_TERMINALS` -- and
    a cut made on ONE mask, which has no other side, reads the first value.
    """
    layers = (peel,) if isinstance(peel, int) else tuple(peel)
    if len(layers) == 1:
        layers = layers * 2
    if len(layers) != 2 or any(n < 0 for n in layers):
        raise SystemExit("--peel-terminals takes one value for both sides of a pair, or two "
                         "(reference then prediction), none of them negative")
    return layers


def peel_layers(args):
    """`peel_layers_of` read off a parsed command line."""
    return peel_layers_of(args.peel_terminals)


def peel_terminals(table, keep, layers, verbose=False):
    """
    Removes the terminal branches of a selection, `layers` times over.

    Why a whole layer. A tip of a cut is where the two things this file
    warns about meet: the calibre that ended the run was measured on the
    pieces where partial volume weighs most, and whether the run ended there
    at all depends on the skeleton having found the junction above it. So
    the tips are what two segmentations of one tree disagree about even when
    they agree about every vessel that divides, and a Dice, driven by
    surface, is read mostly on them. Peeling removes that generation
    outright, on both sides, and leaves a tree whose every tip is a
    bifurcation both sides did see.

    What leaves is one ORDER and not one generation: the runs returned by
    `terminal_runs` are exactly the order-1 class of the kept tree counted
    from the tips -- Strahler's order, recomputed here on the cut -- so the
    branches that go are all of the same order however deep they sit. A
    short collateral coming straight off the trunk is a tip, is order 1, and
    leaves with the rest; the generation it sits at, which one junction the
    skeleton missed would change, plays no part. That is what makes the peel
    reproducible between two segmentations: they need not have found the
    same number of generations, only the same bifurcations.

    Not to be confused with `--min-strahler`, which reads `strahler_dd` off
    the FULL tree, before the truncation: there, no kept branch is order 1,
    since every one of them had a periphery under it.

    Peeling twice is peeling once, twice: what became a tip in the first
    layer is one of the second. The run carrying the seed is never peeled,
    so a selection cannot be emptied however many layers are asked for --
    a tree that is down to its trunk simply stops shrinking.

    `layers` of 0 or None returns the selection untouched.
    """
    if not keep or not layers or layers <= 0:
        return keep

    peeled, gone = keep, []
    for _ in range(int(layers)):
        runs = terminal_runs(table, peeled)
        if not runs:
            break
        gone.extend(runs)
        peeled = peeled - {piece for run in runs for piece in run}

    if verbose and gone:
        length = sum(table[piece]["length_mm"] for run in gone for piece in run)
        print(f"  peeled {len(gone)} terminal branch(es) off the cut, "
              f"{length:.0f} mm of centerline")
    return peeled


def limit_terminal_length(table, keep, max_length, verbose=False):
    """
    Cuts the terminal runs of a selection back to `max_length` mm.

    The gentler half of `peel_terminals`: the same object -- what is left of
    a branch of the KEPT tree past its last bifurcation -- and the same
    reason, but the run is shortened instead of dropped. Where a run ends is
    the least reproducible thing about a cut, one median radius against the
    floor, and a single missed side branch stretches a run by a whole
    generation because the junction that should have ended it is not in the
    skeleton. Capping bounds how far that disagreement can run: what is kept
    is the same tree minus its last centimetre or so of periphery, the part
    the two sides agree on least and the part a surface-driven metric is
    most sensitive to.

    What is dropped is real vessel, so the cap belongs with the floor in
    whatever is reported, and it has to be applied to BOTH sides of a
    comparison, like `--min-diameter`.

    A run is trimmed piece by piece and never emptied: a piece is kept while
    the run's length up to its PROXIMAL end is under `max_length`, so a run
    keeps at least its first piece and overshoots the cap by at most one.
    The resolution of the trim is therefore `--cut-step`, and under
    `--cut-step 0` -- one piece per whole branch -- it can only drop
    terminal branches entire, which is a peel by another name.

    `max_length` of 0 or None returns the selection untouched.
    """
    if not keep or not max_length or max_length <= 0:
        return keep

    drop = set()
    for run in terminal_runs(table, keep):
        walked = 0.0
        for piece in run:
            if walked >= max_length:
                drop.add(piece)
            walked += table[piece]["length_mm"]

    if verbose and drop:
        print(f"  trimmed {len(drop)} piece(s), "
              f"{sum(table[b]['length_mm'] for b in drop):.0f} mm of centerline, off the terminal "
              f"runs past {max_length} mm")
    return keep - drop


def voxel_owners(mask, affine, tree, sleeve):
    """
    Which centerline node each voxel of the mask belongs to, and whether it
    is close enough to it to be kept by anything at all.

    A Voronoi partition of the mask by its own skeleton, over the WHOLE tree
    -- kept branches and dropped ones alike. That is what puts the cut where
    the tree was truncated, roughly normal to the vessel instead of on an
    arbitrary plane, and what lets every dropped branch compete for its own
    voxels, so a subsegmental running along the trunk does not survive
    because it happens to touch a large vessel.

    `within` is the other half of the rule: a voxel sitting further than
    `sleeve` times its node's own radius is kept by no branch. It drops what
    the skeleton never covered -- a blob the thinning walked past, the
    components the pruning removed. The sleeve is proportional rather than a
    fixed millimetre tolerance: a voxel of a wide vessel sits further from
    its axis than a voxel of a thin one, so a fixed distance would either
    cut into the trunk or reach past the segmentals.

    Only the mask's own voxels are queried -- a nearest-neighbour lookup
    over a whole 512^3 volume costs minutes and answers nothing, since a
    background voxel is not written either way. The node coordinates are the
    raw (unsmoothed) skeleton ones: smoothing is for measuring lengths and
    angles, and here the axis has to stay where the voxels are.
    """
    voxels = np.argwhere(mask)
    if not len(voxels):
        return voxels, np.zeros(0, dtype=int), np.zeros(0, dtype=bool)
    nodes = np.unique(np.concatenate([entry["nodes"] for entry in tree["table"]]))
    points = voxels @ affine[:3, :3].T + affine[:3, 3]
    distance, index = cKDTree(tree["world"][nodes]).query(points)
    owner = nodes[index]
    return voxels, owner, distance <= sleeve * tree["radii"][owner]


def plan_cut(mask, affine, spacing, args, verbose=True):
    """
    Everything about one class that does not depend on where the floor
    falls: the ordered tree, and the voxel each of its nodes owns.

    Split off from the selection because a pair of masks cut against each
    other (`truncate_pair`) selects twice on the same tree, and the
    skeletonization is the expensive half by a wide margin.

    Returns (plan, row), the plan being None when nothing survives the
    pruning -- the caller decides whether one unusable class is worth losing
    the file over.
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

    # from here on the "branches" of the tree are pieces of bounded length,
    # so that what is decided per entry is decided per millimetre of vessel
    # and not per bifurcation of one side's skeleton
    row["cut_step_mm"] = args.cut_step
    branches = len(tree["table"])
    tree["table"] = subdivide(tree["table"], tree["world"], tree["radii"],
                              tree["voxel_size"], args.cut_step)
    if verbose and len(tree["table"]) != branches:
        print(f"  {branches} branches cut into {len(tree['table'])} pieces "
              f"of at most {args.cut_step} mm")

    voxels, owner, within = voxel_owners(mask, affine, tree, args.sleeve)
    plan = {"tree": tree, "shape": mask.shape, "voxel_ml": voxel_ml, "voxels": voxels,
            "owner": owner, "within": within}
    return retable(plan, tree["table"]), row


def retable(plan, table):
    """
    Puts a branch table on a plan, and refreshes what is derived from it.

    The Voronoi ownership of the voxels depends on the skeleton's NODES and
    not on how they are grouped into branches, so a plan outlives a change
    of table -- which is what lets `sweep_rescue.py` try several --cut-step
    on one skeletonization.

    `branch` is which branch each owned voxel belongs to, for
    `branch_coverage` alone. A junction node is shared by several branches
    and lands on whichever writes last: it is a statistic over a branch's
    voxels, not a decision about them -- `keep_voxels` marks nodes, and
    keeps a shared one as soon as any kept branch contains it.
    """
    plan["tree"]["table"] = table
    branch_of_node = np.full(len(plan["tree"]["world"]), -1, dtype=int)
    for entry in table:
        branch_of_node[entry["nodes"]] = entry["branch_id"]
    plan["branch"] = (branch_of_node[plan["owner"]] if len(plan["voxels"])
                      else np.zeros(0, dtype=int))
    return plan


def keep_voxels(plan, keep):
    """The voxels a selection of branches owns, as a boolean volume on the input grid."""
    kept = np.zeros(plan["shape"], dtype=bool)
    if not keep or not len(plan["voxels"]):
        return kept
    table = plan["tree"]["table"]
    is_kept = np.zeros(len(plan["tree"]["world"]), dtype=bool)
    is_kept[np.concatenate([table[b]["nodes"] for b in keep])] = True
    kept[tuple(plan["voxels"].T)] = is_kept[plan["owner"]] & plan["within"]
    return kept


def branch_coverage(plan, other):
    """
    Per branch, the share of the voxels it owns that `other` holds too --
    the support `--rescue-support mask` grants a rescue on.

    `other` is the other side's first-pass cut, already resampled onto this
    plan's grid. It is asked of the branch's OWN voxels rather than of the
    two masks at large, because the question is about one vessel: whether
    the other side kept the thing this branch runs through, not whether the
    two segmentations agree around there. A branch owning no voxel --
    everything it might have claimed lies outside its sleeve -- scores 0.

    What it cannot see, and it is worth knowing before reading a cut made
    with it: what a side keeps is a tube of `--sleeve` local radii, and the
    tube around a trunk is wide. A thin vessel that merely runs beside a
    large one, inside its sleeve and inside its mask, is covered 100% by
    this test while having nothing to do with it. `centerline_support` is
    the same question asked of the axes, which does separate the two; it is
    stricter, and strictness has its own cost in matches missed.
    """
    table = plan["tree"]["table"]
    coverage = np.zeros(len(table), dtype=float)
    take = plan["within"]
    if not take.any():
        return coverage
    branch = plan["branch"][take]
    voxels = plan["voxels"][take]
    inside = other[tuple(voxels.T)].astype(float)
    total = np.bincount(branch, minlength=len(table)).astype(float)
    held = np.bincount(branch, weights=inside, minlength=len(table))
    np.divide(held, total, out=coverage, where=total > 0)
    return coverage


def centerline_support(plan, other_plan, other_keep, tolerance):
    """
    Per node of this tree: does the other side have a vessel here, and did
    it KEEP it? The support a rescue is granted on.

    The question is about one vessel -- did the other side keep the thing
    this branch runs through -- and the object that answers it is the
    centerline, not the mask. Two segmentations of one vessel have axes that
    follow each other to within about a radius, whatever their walls do,
    while a thin vessel running beside a large one has its axis millimetres
    away however deeply it sits inside the large one's sleeve. Asking the
    masks instead -- are my voxels inside what you kept -- cannot separate
    those two, because what a side keeps is a tube of `--sleeve` local radii
    and the tube around a trunk is wide.

    Every node of the other tree is queried, DROPPED BRANCHES INCLUDED, and
    the nearest one is then required to be a kept one. That is the whole
    guard: when the other side has this vessel and cut it, its own
    centerline is the nearest thing there and the match fails, honestly.
    Were only the kept nodes queried, the answer would be the nearest large
    vessel however far away -- which, near a hilum, is always the trunk.

    `tolerance` is in local radii of the branch being asked about, not of
    the match, so how far a match may sit scales with the vessel in
    question: matching a 3 mm branch to a 12 mm trunk requires the trunk to
    pass within 1.5 mm of its axis, not within its own 6.

    Both trees are read in world millimetres -- `extract_tree` puts `world`
    on the input volume's own physical frame whatever grid it skeletonized
    on -- so two sides sampled differently are compared where they both
    live, and nothing is resampled to ask this.
    """
    tree, other = plan["tree"], other_plan["tree"]
    nodes = np.unique(np.concatenate([entry["nodes"] for entry in other["table"]]))
    is_kept = np.zeros(len(other["world"]), dtype=bool)
    if other_keep:
        is_kept[np.concatenate([other["table"][b]["nodes"] for b in other_keep])] = True
    distance, index = cKDTree(other["world"][nodes]).query(tree["world"])
    return is_kept[nodes[index]] & (distance <= tolerance * tree["radii"])


def warn_if_empty(table, keep, min_diameter):
    """The trunk itself is under the floor: the floor is wrong for this
    volume, or the root landed on a fragment."""
    if not keep:
        widest = max(2.0 * entry["calibre_mm"] for entry in table)
        print(f"  WARNING: no branch clears {min_diameter} mm (widest is {widest:.2f} mm), "
              f"the cut is empty")


def cut_plan(plan, keep, row, verbose=True):
    """The mask of one selection, with what it kept written into `row`."""
    table = plan["tree"]["table"]
    kept = keep_voxels(plan, keep)
    diameters = [2.0 * table[b]["calibre_mm"] for b in keep]
    row.update({
        "large_ml": float(kept.sum() * plan["voxel_ml"]),
        "large_volume_fraction": (float(kept.sum()) / float(len(plan["voxels"]))
                                  if len(plan["voxels"]) else 0.0),
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
    return kept


def truncate_class(mask, affine, spacing, args, verbose=True):
    """
    Truncates one class on its own tree alone. Returns (kept mask, row) or
    (None, row) when the tree could not be built -- the caller decides
    whether one unusable class is worth losing the file over.
    """
    plan, row = plan_cut(mask, affine, spacing, args, verbose=verbose)
    if plan is None:
        return None, row
    keep = select_branches(plan["tree"]["table"], args.min_diameter, args.max_generation,
                           args.ordering, args.min_strahler)
    warn_if_empty(plan["tree"]["table"], keep, args.min_diameter)
    keep = peel_terminals(plan["tree"]["table"], keep, peel_layers(args)[0], verbose=verbose)
    keep = limit_terminal_length(plan["tree"]["table"], keep, args.max_terminal_length,
                                 verbose=verbose)
    return cut_plan(plan, keep, row, verbose=verbose), row


# --------------------------------------------------------------------------- #
# files
# --------------------------------------------------------------------------- #
def fingerprint(path):
    """A file's identity for a sidecar: name, size and modification time."""
    stat = os.stat(path)
    return {"path": os.path.basename(path), "bytes": stat.st_size, "mtime": round(stat.st_mtime, 3)}


def cut_settings(classes, args, source=None, counterpart=None, peel=None):
    """
    The rule a cut was made by, written beside the mask it produced.

    A truncated mask is not readable on its own: two files called
    `..._large.nii.gz` can hold a 4 mm cut of the artery and a 6 mm cut of
    both trees, and nothing in the name says which. The sidecar makes the
    rule part of the output -- so a later run can tell whether the cut on
    disk is the cut it is asking for, instead of silently reusing another
    one, and so the floor that has to be reported with any number read off
    these masks can be found next to them.

    `source` fingerprints the file that was cut, by size and modification
    time. The rule alone is not enough to decide that a cut can be reused:
    the mask it was made from can be rewritten under the same name -- a
    reference resampled onto the image grid, a prediction regenerated from
    another checkpoint -- and everything about the rule still matches while
    the cut on disk no longer belongs to the file next to it.

    `counterpart` is the file this one was cut AGAINST, for a cut made by
    `truncate_pair`: with a rescue margin the cut depends on the other side
    as much as on the rule, so the other side gets fingerprinted too and a
    cut outlives neither. A margin of 0 makes no use of the counterpart and
    records none, which is what makes such a cut interchangeable with a
    standalone one.
    """
    settings = {
        "classes": {name: (list(values) if values else None) for name, values in classes},
        "min_diameter_mm": args.min_diameter,
        "cut_step": args.cut_step,
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
    # written only when it is on: an absent key and a null one describe the
    # same cut, and always writing it would invalidate every sidecar already
    # on disk -- a cohort's worth of skeletonizations -- for a knob nobody set
    peel = peel_layers(args)[0] if peel is None else peel
    if peel:
        settings["peel_terminals"] = peel
    if args.max_terminal_length:
        settings["max_terminal_length"] = args.max_terminal_length
    if source is not None and os.path.exists(source):
        settings["source"] = fingerprint(source)
    margin = getattr(args, "rescue_margin", 0.0) if counterpart else 0.0
    settings["rescue"] = None if not margin else {
        "margin_mm": margin,
        "coverage": args.rescue_coverage,
        "support": args.rescue_support,
        "distance": args.rescue_distance if args.rescue_support == "centerline" else None,
        "against": fingerprint(counterpart) if os.path.exists(counterpart)
        else os.path.basename(counterpart),
    }
    return settings


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
    write_cut(path, output, affine, classes, destination, args)
    for row in rows:
        row["output"] = destination
    return rows, kept_by_class


def write_cut(path, output, affine, classes, destination, args, counterpart=None, peel=None):
    """The truncated volume and the sidecar saying by which rule it was cut."""
    directory = os.path.dirname(destination)
    if directory:
        os.makedirs(directory, exist_ok=True)
    nib.save(nib.Nifti1Image(output, affine), destination)
    with open(settings_path(destination), "w") as handle:
        json.dump(cut_settings(classes, args, path, counterpart, peel), handle, indent=1,
                  sort_keys=True)
    print(f"  wrote {destination}")


def truncate_pair(reference_path, reference_classes, prediction_path, prediction_classes, args,
                  destinations=(None, None), verbose=True):
    """
    Cuts two masks of the same case, each on its own tree and by the same
    rule, letting each side keep the branches the floor only just cut off
    the other one has.

    Why the two cuts are made together. `--min-diameter` is a floor on a
    MEASUREMENT of a segmentation, not on the vessel: half a voxel of
    partial volume, a wall the model drew one voxel thin, and the same
    segmental artery measures 5.1 mm in the reference and 4.9 mm in the
    prediction. Cut independently it stays in one tree and leaves the other
    with its whole subtree behind it, and a Dice read on the two then
    reports as a miss what is a rounding difference on the cut. The effect
    is not symmetric in practice -- a prediction is smoother and a touch
    thinner than the reference it was trained on, so it is mostly the
    prediction that loses branches -- but nothing here assumes that, and the
    rescue runs both ways.

    So each side is cut twice. The first pass is the plain rule, and it is
    what the other side is judged against; the second re-selects with a
    tolerance band of `--rescue-margin` under the floor, admitting a branch
    that falls in it when `--rescue-coverage` of it is supported by the
    OTHER side's first pass. The support is always that first pass and never
    the rescued one: two sides feeding each other would walk a chain of
    sub-threshold vessels arbitrarily far into the periphery, one margin at
    a time, and the floor would stop meaning anything.

    `--rescue-support` picks what "supported" means, and the two answers are
    not interchangeable:

      mask        (default) the share of the branch's own voxels that lie
                  inside the other side's cut -- `branch_coverage`. Generous
                  near large vessels, where the other side's cut is a wide
                  sleeve, so a thin vessel running beside a trunk can be
                  fully covered by it without being it.
      centerline  the share of the branch's axis that runs along an axis the
                  other side kept -- `centerline_support`. It separates a
                  vessel from its large neighbour, at the price of the
                  matches it misses when the two skeletons sit a little
                  apart, which two different grids are enough to cause.

    Which one is right is a question about a cohort, not about the method;
    `sweep_rescue.py` puts them side by side on the same trees.

    A rescued branch is still only reached through the closure -- it has to
    hang off something that cleared the floor on its own -- and the branches
    beyond it rejoin by the ordinary rule, which is the point: what comes
    back is the subtree, not the one borderline segment.

    Reading what this produces: the rescue can only ADD to a cut, and it
    adds where the two sides already agree, so it mostly raises the Dice.
    `n_segments_rescued` and `rescued_ml`, on every row, say how much of the
    number rests on it; --rescue-margin 0 is the cut without it.

    Returns (rows, reference masks, prediction masks), the masks per class so
    a caller that has to score the truncation does not read the files it
    just watched being written.
    """
    if [name for name, _ in reference_classes] != [name for name, _ in prediction_classes]:
        raise ValueError("the two sides of a pair must name the same classes, in the same order")

    # indexed like `sides`: the reference's peel, then the prediction's
    peel = peel_layers(args)
    sides = []
    for role, path, classes, destination in (
            ("reference", reference_path, reference_classes, destinations[0]),
            ("prediction", prediction_path, prediction_classes, destinations[1])):
        data, affine, spacing = read_volume(path)
        sides.append({"role": role, "path": path, "classes": classes, "data": data,
                      "affine": affine, "spacing": spacing, "kept": {}, "rows": [],
                      "output": np.zeros_like(data),
                      "destination": destination or output_path(path, args.suffix, args.output_dir)})

    # Each side is cut on its own grid -- the skeleton deserves the resolution
    # its file was written at, and a cut mask sitting next to its source should
    # have its source's geometry. Only --rescue-support mask has to cross, and
    # it moves the other side's first-pass cut onto the grid of the side being
    # rescued; the centerlines are compared in world millimetres, where the two
    # sides already live, and are never resampled.
    same_grid = (sides[0]["data"].shape == sides[1]["data"].shape
                 and np.allclose(sides[0]["affine"], sides[1]["affine"], atol=1e-3))

    for name, _ in reference_classes:
        plans, rows = [], []
        for side in sides:
            values = dict(side["classes"])[name]
            mask = class_mask(side["data"], values)
            print(f"{os.path.basename(side['path'])} [{name}]")
            side["kept"][name] = np.zeros(side["data"].shape, dtype=bool)
            if not mask.any():
                print(f"  WARNING: no voxel with label {values}, skipping")
                plans.append(None)
                rows.append(None)
                continue
            plan, row = plan_cut(mask, side["affine"], side["spacing"], args, verbose=verbose)
            row.update(file=side["path"], **{"class": name})
            side["rows"].append(row)
            plans.append(plan)
            rows.append(row)

        # first pass, the plain rule, on both sides: what the rescue is judged against
        plain, first = [], []
        for index, plan in enumerate(plans):
            if plan is None:
                plain.append(set())
                first.append(None)
                continue
            keep = select_branches(plan["tree"]["table"], args.min_diameter, args.max_generation,
                                   args.ordering, args.min_strahler)
            warn_if_empty(plan["tree"]["table"], keep, args.min_diameter)
            keep = peel_terminals(plan["tree"]["table"], keep, peel[index])
            keep = limit_terminal_length(plan["tree"]["table"], keep, args.max_terminal_length)
            plain.append(keep)
            first.append(keep_voxels(plan, keep))

        # second pass, each side against the other's first
        for index, (side, plan) in enumerate(zip(sides, plans)):
            if plan is None:
                continue
            other_plan, other_side = plans[1 - index], sides[1 - index]
            keep, rescued = plain[index], set()
            if args.rescue_margin > 0 and other_plan is not None and plain[1 - index]:
                branches = plan["tree"]["table"]
                if args.rescue_support == "centerline":
                    matched = centerline_support(plan, other_plan, plain[1 - index],
                                                 args.rescue_distance)

                    def supported(branch, matched=matched, branches=branches):
                        return (float(matched[branches[branch]["nodes"]].mean())
                                >= args.rescue_coverage)
                else:
                    other = first[1 - index]
                    coverage = branch_coverage(plan, other if same_grid else to_grid(
                        other, other_side["affine"], plan["shape"], side["affine"]))

                    def supported(branch, coverage=coverage):
                        return coverage[branch] >= args.rescue_coverage

                keep = select_branches(
                    branches, args.min_diameter, args.max_generation, args.ordering,
                    args.min_strahler, margin=args.rescue_margin, supported=supported)
                # both are applied to the rescued selection and not to what
                # the rescue added: a rescued branch can turn a run that was
                # terminal into an ordinary one, and the run is the object
                keep = peel_terminals(branches, keep, peel[index])
                keep = limit_terminal_length(branches, keep, args.max_terminal_length)
                rescued = keep - plain[index]

            row = rows[index]
            kept = cut_plan(plan, keep, row, verbose=verbose)
            row["rescue_margin_mm"] = args.rescue_margin
            row["rescue_coverage"] = args.rescue_coverage
            row["n_segments_rescued"] = len(rescued)
            row["rescued_ml"] = float(kept.sum() - first[index].sum()) * plan["voxel_ml"]
            # what the band admitted, and what rejoined behind it by the
            # ordinary rule -- the second is the point of the first, and
            # counting them together would read as a much wider tolerance
            table = plan["tree"]["table"]
            in_band = sorted(2.0 * table[b]["calibre_mm"] for b in rescued
                             if 2.0 * table[b]["calibre_mm"] < args.min_diameter)
            row["n_segments_in_band"] = len(in_band)
            if rescued and verbose:
                behind = len(rescued) - len(in_band)
                print(f"  rescued {len(in_band)} branch(es) in the "
                      f"{args.min_diameter - args.rescue_margin}-{args.min_diameter} mm band the "
                      f"{other_side['role']} kept too"
                      + (f" ({in_band[0]:.2f}-{in_band[-1]:.2f} mm)" if in_band else "")
                      + (f", and {behind} behind them" if behind else "")
                      + f": +{row['rescued_ml']:.2f} mL")
            side["kept"][name] = kept
            side["output"][kept] = side["data"][kept]

    rows = []
    for index, side in enumerate(sides):
        write_cut(side["path"], side["output"], side["affine"], side["classes"],
                  side["destination"], args, sides[1 - index]["path"], peel=peel[index])
        for row in side["rows"]:
            row["output"] = side["destination"]
        rows.extend(side["rows"])
    return rows, sides[0]["kept"], sides[1]["kept"]


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

    from pipeline import config as cfg  # only this branch needs the class map

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
def add_cut_arguments(parser, peel_terminals=0):
    """
    Where the tree is cut. Shared with compute_dice.py, so the two agree by
    construction.

    `peel_terminals` is the only default a caller sets, because the right
    value depends on what is being cut and not on the rule. It is one number
    for both sides of a pair, or two -- reference then prediction.

    Cutting one segmentation to look at it (this file) peels nothing: the tips
    are vessel, and dropping them throws away what was asked for. SCORING one
    against a hand-drawn reference (compute_dice.py, sweep_rescue.py) peels the
    REFERENCE alone, `1 0`: its tips are where the two disagree for reasons
    that are not the model's -- an annotator stops a vessel where the contrast
    goes, and one voxel of that decision moves a whole terminal run -- while
    the prediction's tree has already been shortened by the floor, a model
    drawing its vessels thinner. See `compute_dice.PEEL_TERMINALS`.
    """
    cut = parser.add_argument_group("where to cut")
    cut.add_argument("--min-diameter", type=float, default=MIN_DIAMETER_MM, metavar="MM",
                     help="A branch thinner than this is cut off, with everything under it. This "
                          f"is what \"large vessel\" means here -- report it. Default: {MIN_DIAMETER_MM} "
                          "mm, which sits inside the segmental range rather than at its lower "
                          "edge, where the two sides disagree most")
    cut.add_argument("--cut-step", type=float, default=5.0, metavar="MM",
                     help="Cut every branch into pieces of at most this length before deciding, so "
                          "the floor is applied to a LOCAL calibre and the cut can fall inside a "
                          "branch. Without it the decision is one median over a whole branch, and "
                          "a branch runs between bifurcations -- a property of the skeleton, so "
                          "two segmentations that do not bifurcate alike get different cuts of the "
                          "same anatomy. 0 restores one decision per branch. Default: 5")
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
    cut.add_argument("--peel-terminals", type=int, nargs="+", default=list(peel_layers_of(
                         peel_terminals)), metavar="N",
                     help="Also drop the terminal branches of the KEPT tree, N layers of them. A "
                          "layer is every run past the last bifurcation OF THE CUT -- the order-1 "
                          "class counted from the tips, so everything that leaves is of the same "
                          "order however deep it sits, and a tip is removed for being a tip and "
                          "not for its generation, which one junction the skeleton missed would "
                          "change. Peeled on both sides, what is left is a tree whose every tip "
                          "is a bifurcation both saw; peeled on one, it is that side brought back "
                          "to the extent of the other. Either way a tip is where the calibre was "
                          "measured worst and where two segmentations of one tree agree least. "
                          "Not "
                          "--min-strahler, which reads the order off the full tree, before the "
                          "truncation. Drops real vessel: report it. One value peels both sides "
                          "of a pair alike; two, REFERENCE then PREDICTION, peel them differently "
                          "-- which is the point when the two sides are not the same kind of "
                          "object (see compute_dice.py: a model draws a vessel thinner than a "
                          "hand does, so the floor has already shortened its tree and peeling it "
                          "again cuts twice for one asymmetry). "
                          f"Default here: {' '.join(str(n) for n in peel_layers_of(peel_terminals))}")
    cut.add_argument("--max-terminal-length", type=float, default=None, metavar="MM",
                     help="Also cut every terminal run of the KEPT tree -- the chain of pieces "
                          "past its last bifurcation -- back to this length. Where a run ends is "
                          "the least reproducible part of a cut: one median radius against the "
                          "floor, and one side branch the skeleton missed stretches a run by a "
                          "whole generation, so two segmentations stop theirs in different places "
                          "even where they agree about every vessel that divides. The cap trades "
                          "those tips away; it drops real vessel, so report it and pass the same "
                          "value on both sides. Trimmed in steps of --cut-step, so a run overshoots "
                          "by at most one piece. Off by default; 10 is a reasonable value")
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
    args.peel_terminals = list(peel_layers(args))
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
          + f", cut at {args.min_diameter} mm"
          + (f", {peel_layers(args)[0]} terminal layer(s) peeled" if peel_layers(args)[0] else "")
          + (f", terminal runs capped at {args.max_terminal_length} mm"
             if args.max_terminal_length else ""))

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
