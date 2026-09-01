# analysis/ — evaluating a segmentation

Everything here starts from a segmentation that already exists, whether
`pipeline/` produced it or not. The [main README](../README.md) has the two
commands most people want; this is the rest, and the reasoning behind the
defaults.

## Which tool

| | |
| --- | --- |
| **`centerline`** | mask → skeleton → branch graph → branching ratios, with quality metrics and an orphan-component report. The one everything else is built on |
| **`truncate`** | cut a tree back to its large vessels. Writes masks, computes no metric |
| **`compute_dice`** | score predictions against references, on the whole tree and on the large vessels |
| **`sweep_rescue`** | run `compute_dice` over a grid of cut parameters, to choose the floor on the cohort rather than on one case |
| **`compare_predictions`** | two prediction sets against each other, no ground truth needed |
| **`cohort`** | assemble per-subject ratio files into one table, with the checks that decide whether the subjects are comparable |
| **`connectivity`** | how fragmented a mask is, in one number |
| **`phantom`** | build a synthetic tree of known geometry |
| **`calibrate`** | what the chain reports against what the phantom imposed, swept over spacings |
| **`radius_audit`** | the same, for the radius alone, stratified by the vessel's angle with z |

They form a stack rather than a set — each imports the one below, and nothing is
duplicated between them:

```
sweep_rescue  ->  compute_dice  ->  truncate  ->  centerline
```

The method behind `centerline` — how branches are numbered, what the ratios mean,
what has to hold before they can be believed, and how `phantom`, `calibrate` and
`radius_audit` validate it — is documented on its own in
[**docs/centerline.md**](../docs/centerline.md).

## Cutting a segmentation back to its large vessels

A whole-tree Dice is dominated by the periphery: the subsegmental vessels carry
most of the surface and almost none of the volume, they are where the reference
itself is least certain, and a model that draws the hilum perfectly but fades out
two generations early scores about the same as one that does the opposite.

`truncate` cuts a segmentation back to its proximal tree — trunk, lobar and
segmental vessels — and writes the truncated mask. It computes no metric: it
writes masks, and the masks go to whatever already scores them. Run it on the
references **and** on the predictions with the same `--min-diameter`, so both
sides are cut by the same rule — a truncated reference scored against a whole
prediction would count every peripheral vessel of the prediction as a false
positive.

The mask is skeletonized and measured exactly as `centerline` does it, then the
tree is truncated: a branch has to clear `--min-diameter` (default 5 mm, which
sits inside the segmental range rather than at its lower edge, where two
segmentations of one tree disagree most), and what is kept is the connected group
of such branches containing the **widest branch of the tree**. Clearing the floor
is necessary and not sufficient, which is what makes this a truncation rather
than a calibre filter: a wide distal blob sitting behind a thin branch — a leak,
a fused vein — cannot reach the trunk through large vessels, so it goes with the
branch that carries it.

Growing from the widest branch rather than down from the root is deliberate. The
root is chosen upstream as the widest free *end* of the skeleton, which is the
trunk in a clean mask and anywhere at all in a degraded one: on an eroded venous
prediction it landed on a 5 mm peripheral stump whose daughters measured 3.83 mm,
the real 37 mm trunk sat six generations "below" it, and a root-first traversal
kept 1 segment out of 203. Seeding from the widest branch cannot fail that way,
and returns exactly the same cut whenever the root is right.

The truncated tree becomes voxels again through a Voronoi partition of the mask
by its own skeleton: a voxel is kept when its nearest centerline node belongs to
a kept branch and sits within `--sleeve` local radii (1.5 by default). The cut
surface therefore falls where the tree was truncated, roughly normal to the
vessel, and a subsegmental vessel running alongside the trunk is claimed by its
own centerline instead of surviving because it happens to touch a large one.

What comes out carries the input's own label values on the input grid, so it is a
drop-in replacement for the file it came from:

```bash
# one class of one mask -> vascular_case001_large.nii.gz beside it
python -m analysis.truncate --input vascular_case001.nii.gz --label 3

# every foreground class of config.LABEL_CLASS_MAP, each on its own tree,
# into one file (raw 3 = artery, 4 = vein, airway classes 1-2 dropped)
python -m analysis.truncate --input vascular_case001.nii.gz --all-classes

# the labels of a manifest split
python -m analysis.truncate --manifest manifests/manifest_ct.json --split val \
    --output-dir cut/
```

## Dice on the whole tree and on the large vessels

`compute_dice` scores a checkpoint's predictions against the references of a
manifest split, twice: on the whole tree, and on the large vessels alone.

```bash
# predictions already produced by inference
python -m analysis.compute_dice --manifest manifests/manifest_ct.json --split val \
    --csv dice.csv

# or straight from the checkpoint, predicting whatever is missing on the way
python -m analysis.compute_dice --manifest manifests/manifest_ct.json --split val \
    --checkpoint "$DATASET_ROOT/checkpoints/.../last.ckpt" --csv dice.csv
```

Predictions are read off the disk where `pipeline.inference` writes them
(`<image stem>_vascular_pred.nii.gz`, or under `--pred-dir`). With
`--checkpoint`, the missing ones are produced here first, through inference's own
sliding window and preprocessing — and **only** the missing ones: a prediction
already on disk is reused and never overwritten, since it is the thing being
measured and regenerating it halfway through a cohort would change what the
cohort means. The checkpoint is loaded lazily, so a run over predictions that all
exist never pays for torch at all.

Going through `compute_dice --checkpoint` rather than `inference --input-dir`
also avoids a trap: `--input-dir` globs every `*.nii.gz` under the directory and
only excludes files already carrying the prediction suffix, so it will happily
run the model on your label volumes. The manifest says which files are images.

### When the data moved

A manifest records where the data was when it was written, and the volumes
outlive that path: a cluster mount, a copy on another filer, a scratch directory.
`--rewrite OLD=NEW` maps the prefix at read time — repeatable, first match wins,
the file is not modified — so one manifest holds one definition of the split
across all of them:

```bash
python -m analysis.compute_dice --manifest manifests/manifest_vibe.json --split val \
    --rewrite /data/flamant/data/ct=/biomaps/spiro3d/.../ct \
    --checkpoint .../last.ckpt --csv dice.csv
```

Editing the manifest per machine instead is how two runs end up disagreeing about
which cases are validation. When every case is skipped, the exit message names
the root the manifest asked for and writes the `--rewrite` line for you.

`--rewrite` is exact when the tree simply *moved*. When the layout changed too and
no prefix fits, `--data-dir DIR` is the fallback: it walks the directory and
matches the manifest's files by name. A name found in two places under it is an
error rather than a guess.

### The terminal peel, and why it is asymmetric

The large-vessel pass truncates **both sides** with the same calibre floor, then
peels the last layer of tips off the **reference alone** (`--peel-terminals 1 0`),
and writes the truncated masks out so the number can be traced back to the
volumes it was read on.

The reference scored here is a **hand-drawn annotation**, and its last layer of
tips is where a hand and a model disagree for reasons that are not the model's —
an annotator stops a vessel where the contrast goes rather than where the vessel
does, and the calibre that ended a terminal run was measured where partial volume
weighs most.

The peel is asymmetric because the two sides are not the same kind of object: the
model draws a vessel *thinner* than the hand that annotated it, so the same floor
already stops the prediction's tree earlier, and peeling both sides would take
that one asymmetry out twice — leaving the prediction shorter than the reference
it is compared with, a truncation difference scored as a model error.
`--peel-terminals` takes one value for both sides or two, reference then
prediction: `1 0` is the default, `1 1` peels symmetrically, `0` peels nothing.
`truncate` on its own still peels nothing.

That is a heuristic about these two objects, not a property of the metric, and
the tools print what it is worth: `centerline_kept_mm_reference` against
`centerline_kept_mm_prediction` is the length each side kept, in world
millimetres, and near-equal means two trees of the same extent. Each row also
keeps `n_segments_kept_reference` next to `n_segments_kept_prediction` — a case
where those disagree wildly is a case to look at before quoting its Dice.

### Choosing the floor on the cohort

`sweep_rescue` settles it on the cohort rather than on the story, by putting the
peel in the same grid as the rescue knobs:

```bash
python -m analysis.sweep_rescue --manifest manifests/manifest_ct.json --split val \
    --peels 0 "1 0" "1 1" --margins 0 2 --csv sweep_peel.csv
```

Each value is one or two numbers in **one** shell argument. Read the result on
the `length gap` column — reference minus prediction, signed, in mm of centerline
— and take the peel nearest 0, **not** the best `dice_large`: peeling always
raises the Dice, since what is left is a smaller and easier region, so reading
the table that way picks the heaviest peel every time.

The peel shares the grid rather than being three separate runs because it is not
independent of the margin: each side's plain cut is what the *other* side's
rescue is judged against, so peeling the reference alone leaves the prediction
more to be rescued by, and the knee of the margin curve can move with it. Compare
margins within one peel.

`--model` names the checkpoint in every row it writes, so two sweeps concatenate
into the one table the floor is actually chosen on.
[`figures.plot_sweep`](../figures) turns that CSV into the two figures the
decision is read off.

### Where the two masks differ

Two volumes are written per pass:

- `<case>_<class>_errors_{full,large}.nii.gz` — a label map, 1 = agreement,
  2 = predicted only (false positive), 3 = reference only (false negative).
  Exact, no parameter, opens straight into Slicer as a segmentation.
- `<case>_<class>_localdice_{full,large}.nii.gz` — the local Dice: at every
  voxel, the Dice of the two masks inside a `--heat-window` cube (20 mm by
  default) around it, computed with box filters. It is the global Dice decomposed
  in space, so **low is bad** — the opposite convention from the error map, do
  not read the two with the same colour scale. NaN where there is not enough
  vessel in the window to divide by.

`local_dice_p10` and `local_dice_median` in the CSV are that map read at the mask
voxels themselves — how bad the bad regions are, next to the single average the
Dice reports. They are deliberately not taken over the whole field: the map
extends a window's reach around the masks, so most voxels carrying a value sit in
the fringe where the window catches a vessel by its edge, and the median over the
field reads 0.0 next to a global Dice of 0.8.

`--swap-av` corrects a checkpoint trained with the inverted artery/vein
convention, by reading the artery out of the vein index rather than by rewriting
anything. Truncated masks already on disk are reused when their sidecar matches
the current settings, so a second run — different maps, a different window —
costs seconds instead of re-skeletonizing.

## Comparing predictions from two models

`compare_predictions` does a volume-by-volume comparison of two prediction sets —
two checkpoints run over the same images via `inference --output-suffix ...` —
with **no ground truth required**. It pairs files by filename suffix within each
patient folder and, per class plus a merged "vessel" class, computes Dice,
Normalized Surface Dice, HD95, average surface distance, volumes and
connected-component counts, one row per case × class:

```bash
python -m analysis.compare_predictions \
    --input-dir /data/flamant/data/ct/lidc_idri \
    --suffix-a _vascular_pred_ct \
    --suffix-b _vascular_pred2 \
    --swap-av-a \
    --csv prediction_comparison.csv
```

`--swap-av-a` corrects for a known artery/vein inversion in model A's predictions
before computing per-class metrics — without it, artery/vein Dice looks near-zero
even though the two models agree on vessel location (`--swap-av-b` is the
equivalent for model B). `--no-surface` and `--no-topology` drop the slower
metrics (NSD/HD95/ASD, and component counts) for a quick Dice + volume pass
first.
