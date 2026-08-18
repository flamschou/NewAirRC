# AirRC

Deep learning pipeline for 3D vascular tree segmentation, built on a
[MONAI](https://monai.io/) `DynUNet` (nnU-Net-style 3D U-Net with deep
supervision) trained with PyTorch Lightning.

## Pipeline overview

```
manifest.json --> manifest.py --> transforms.py --> dataset.py --> train.py --> model.py (DynUNet)
```

- `config.py` — single source of truth for paths, patch geometry, class
  names, and training hyperparameters. Edit this file to change the setup;
  nothing else in the codebase hardcodes these values.
- `manifest.py` — loads the JSON manifest that pairs each image with its
  label and splits it into train/val.
- `transforms.py` — MONAI preprocessing (resampling, intensity
  normalization) and augmentation (random patch cropping, flips, noise)
  pipelines. Modality-agnostic (no CT-specific Hounsfield unit logic).
- `dataset.py` — builds the cached `PersistentDataset`/`DataLoader` objects.
- `model.py` — `DynUNet` construction.
- `config_loss.py` — deep-supervision-weighted Dice+CE loss.
- `train.py` — training entrypoint (single stage).
- `inference.py` — sliding-window prediction on full volumes from a trained
  checkpoint.

## Data format

Images and labels are 3D NIfTI (`.nii.gz`) volumes. Labels must use
contiguous integer class indices starting at 0 (0 = background). The number
and meaning of classes is defined by `config.CLASS_NAMES`, e.g.:

```python
CLASS_NAMES = ["background", "vessel"]                  # binary
CLASS_NAMES = ["background", "vein", "artery"]           # multi-class
```

## Manifest

Pairs of image/label files are declared explicitly in a JSON manifest
(file naming conventions are not assumed, since production filenames won't
match any fixed pattern):

```json
[
  {"image": "data/images/case001.nii.gz", "label": "data/labels/case001.nii.gz", "split": "train"},
  {"image": "data/images/case002.nii.gz", "label": "data/labels/case002.nii.gz", "split": "val"}
]
```

Point `config.MANIFEST_PATH` (or the `MANIFEST_PATH` env var) at your
manifest file.

## Running training

```bash
pip install -r requirements.txt
MANIFEST_PATH=/path/to/manifest.json python train.py
```

Set `DEBUG=1` for a fast, tiny run (few patches, one epoch) to sanity-check
the pipeline before a full training run:

```bash
DEBUG=1 MANIFEST_PATH=manifest.example.json python train.py
```

`manifest.example.json` points at `example_data/` (not tracked in git) and
is only meant for smoke-testing the pipeline mechanics — it reuses a single
volume for both train and val, so it is not a real training run.

## Running on the cluster (SLURM)

```bash
sbatch train.slurm
```

This queues the job and prints its `<jobid>`. `train.slurm` writes stdout to
`slurm-<jobid>.out` and stderr to `slurm-<jobid>.err` in the submission
directory, and sets `DATASET_ROOT` (checkpoints/logs/cache) and
`MANIFEST_PATH` for `train.py`.

## Monitoring training

**Job status**

```bash
squeue -u $USER                 # is it queued / running, on which node
scontrol show job <jobid>       # full job detail (node, time limit, reason if pending)
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode  # after it finishes
```

**Live logs**

```bash
tail -f slurm-<jobid>.out       # training progress (epoch/step, loss, lr)
tail -f slurm-<jobid>.err       # tracebacks / warnings
```

**GPU usage** (run on the compute node, e.g. via `srun --jobid=<jobid> --pty nvidia-smi`,
or `ssh` to the node shown by `squeue` first):

```bash
srun --jobid=<jobid> --pty watch -n 2 nvidia-smi
```

**TensorBoard** (loss, `val_dice_metric`, learning rate curves)

Metrics are logged under `$DATASET_ROOT/logs/<EXPERIMENT_NAME>/run`
(`vessel_segmentation` is the experiment name set in `config.py`). From a
machine with access to that path:

```bash
tensorboard --logdir "$DATASET_ROOT/logs/vessel_segmentation" --port 6006
```

If `$DATASET_ROOT` is only reachable on the cluster, forward the port over SSH
instead of running TensorBoard locally:

```bash
ssh -L 6006:localhost:6006 <cluster-host>
# then, on the cluster:
tensorboard --logdir "$DATASET_ROOT/logs/vessel_segmentation" --port 6006 --bind_all
```

Then open `http://localhost:6006` locally.

**Checkpoints**

Saved every 10 epochs plus a rolling `last.ckpt` (used to auto-resume) under
`$DATASET_ROOT/checkpoints/<EXPERIMENT_NAME>/run/`.

```bash
ls -lh "$DATASET_ROOT/checkpoints/vessel_segmentation/run"
```

## Running inference

`inference.py` runs sliding-window prediction (Gaussian-blended, 50%
overlap, patch size `config.PATCH_SIZE`) on a full volume using a trained
checkpoint. Preprocessing matches training (reorient, resample to
`config.TARGET_SPACING`, normalize); the predicted label is then mapped back
onto the original image's orientation/spacing before being saved, so its
shape and affine match the input file.

`--output`/`--output-dir` are optional. If omitted, each prediction is
written next to its input image as `<name>_vascular_pred.nii.gz` (e.g.
`case001.nii.gz` -> `case001_vascular_pred.nii.gz`).

Single volume:

```bash
python inference.py \
  --checkpoint "$DATASET_ROOT/checkpoints/vessel_segmentation/run/last.ckpt" \
  --input path/to/image.nii.gz
```

Batch over a directory (one prediction per `*.nii.gz` input):

```bash
python inference.py \
  --checkpoint "$DATASET_ROOT/checkpoints/vessel_segmentation/run/last.ckpt" \
  --input-dir path/to/images/
```

## Comparing predictions from two models

`compare_predictions.py` does a volume-by-volume comparison of two
prediction sets (e.g. two checkpoints run over the same real-world
images via `inference.py --output-suffix ...`), with no ground truth
required. It pairs up files by filename suffix within each patient
folder and, per class (plus a merged "vessel" class), computes Dice,
Normalized Surface Dice + HD95 + average surface distance (mm),
volumes, and connected-component counts, writing one row per case x
class to a CSV:

```bash
python compare_predictions.py \
  --root-dir /data/flamant/data/ct/lidc_idri \
  --suffix-a _vascular_pred_ct \
  --suffix-b _vascular_pred2 \
  --swap-av-a \
  --output prediction_comparison.csv
```

`--swap-av-a` corrects for the known artery/vein inversion in the CT
model's predictions (`_vascular_pred_ct`) before computing per-class
metrics -- without it, artery/vein Dice will look near-zero even
though the two models agree on vessel location (`--swap-av-b` is the
equivalent flag for model B). `--skip-surface` / `--skip-topology`
drop the slower metrics (NSD/HD95/ASD and component counts,
respectively) if you just want a quick Dice + volume pass first.

## Extracting a centerline

`centerline.py` extracts the centerline (curve-skeleton) of a pulmonary
artery mask: it cleans the mask up, resamples it to isotropic voxels so
the skeleton is not biased by the slice thickness, thins it with Lee's 3D
algorithm, then turns the skeleton into a branch graph -- junction clusters
merged, short spurs pruned, loops cut, a local radius from the distance
transform, and an order counted from the trunk (the widest free end, or
`--root i j k`). From that graph it fits the three branching ratios R_b,
R_d and R_l with their confidence intervals.

```bash
python centerline.py --input artery.nii.gz
```

Like `inference.py`, `--output` is optional: the centerline mask is written
next to the input as `<name>_centerline.nii.gz`, on the input grid (so it
overlays directly on the mask). `--paint generation|branch` colors it by
generation or branch id instead of a binary mask. The other outputs are
opt-in: `--csv` has one row per point (voxel index, world mm, radius),
`--branches-csv` one row per segment (length, chord, tortuosity, radii, all
four orderings), `--elements-csv` one row per element, `--orders-csv`,
`--ratios-csv`, `--bifurcations-csv` and `--sweep-csv` the analysis tables
below, and `--vtk` a legacy polydata for Slicer/ParaView.

```bash
python centerline.py \
  --input artery.nii.gz \
  --output artery_centerline.nii.gz \
  --csv centerline_points.csv \
  --branches-csv centerline_branches.csv \
  --orders-csv centerline_orders.csv \
  --ratios-csv centerline_ratios.csv \
  --bifurcations-csv centerline_bifurcations.csv \
  --vtk centerline.vtk
```

### Anatomical report

Every run prints a morphometric report (`--no-report` to skip it):

- **per order** — count, how many end there, total length, mean length and
  its SD, mean diameter and its SD, tortuosity, and the mean radius at the
  tips of that order. Followed by the calibre monotonicity check (see
  below).
- **terminal branches** — the distribution of the tip radius, i.e. the
  calibre at which the segmentation stops resolving vessels, plus the
  length and depth of the leaves.
- **bifurcations** — parent radius, asymmetry (smallest daughter over
  largest) and the angle between daughters, then the area ratio (sum of
  the daughter sections over the parent section) and the exponent of
  Murray's law (3 = optimal for laminar flow, 2 = cross-section conserved)
  restricted to the junctions where the parent and both daughters are
  wider than `--murray-min-voxels` voxels. Those two are ratios raised to
  a power, so they are meaningless where the radius saturates on the voxel
  size.
- **tree** — tortuosity and length distributions.

### Quality metrics

Printed at the end of every run, even with `--no-report`, and exportable as
a single CSV row with `--quality-csv` so cases can be concatenated and
compared. They are independent on purpose: a defect in one implies nothing
about the others.

```
=== quality metrics ===
1. centerline length in the largest component :  94.2%   (687 mm outside, 31 components, 10 of them over 10 mm)
   the same fraction measured on mask volume  :  97.5%   (optimistic: the trunks weigh more than the twigs)
2. leaves at the resolution floor (1.50 mm)   :  86.8%   (336/387 leaves; ...)
3. Murray exponent, vessels over 3 voxels     :   2.29   (IQR 1.62-3.72, n=17; ...)
4. cycles in the skeleton                     :      2   (an artery tree has no anastomosis, expected 0; 2 cut to make the tree orderable)
5. leaves ending early (order <= 1, r > 2.00 mm):     1   (0.3% of leaves)
     r= 2.50 mm  order 0  voxel (144, 101, 213)  world (16.0, -19.5, 85.0) mm
```

1. **Continuity.** An artery tree is one connected object; everything
   outside its main component is a cut branch or a false positive. Measured
   on centerline length, where every branch counts the same, and not on
   volume, where a fat trunk hides hundreds of broken twigs -- both are
   printed so the gap is visible. This is why the mask is skeletonized
   whole and only then restricted to its main component.
2. **Depth.** How far down the segmentation goes. The median tip radius
   cannot answer that: it saturates at 1.5 voxels, which is the grid, not
   the model. The share of leaves *sitting* on that floor can -- near 100%
   the image is the limit and the model uses all the resolution available,
   well below it the model stops early and is itself the limit.
3. **Calibre.** The only indicator here that catches over-segmentation, a
   leak into a neighbouring structure or two vessels fused: topology alone
   would accept a correctly connected but systematically too thick tree.
   Median and IQR only, never the mean, and only over vessels wider than
   `--murray-min-voxels` voxels (below that the radius is quantized and the
   exponent is noise).
4. **Cycles (beta 1).** Zero is expected: arteries do not anastomose. Any
   cycle is two branches touching in the mask -- artery to artery, or worse
   a bridge to a vein -- and it corrupts every order downstream, so it
   contaminates the other metrics too. Report the raw count per case. This
   is also the `edges = nodes - 1` check on the tree: the count is
   `E - N + components`, so zero means the graph really is a forest. The
   loops are then cut so the tree can be ordered at all, and how many were
   cut is reported next to the count.
5. **Breakpoints.** Terminal branches still close to the main path
   (`--breakpoint-order`) yet several millimetres wide
   (`--breakpoint-radius`): a vessel that size does not simply end. This is
   the only metric that localizes -- it prints the worst offenders and
   `--breakpoints-csv` exports all of them with voxel and world
   coordinates, ready to open in a viewer. They are the likely attachment
   points of the fragments counted by metric 1.

### How branches are numbered

Counting junctions from the root is not anatomical: a trunk giving off
collaterals gets renumbered at every one of them, so the interlobar artery
ends up labelled "generation 16" while still being the same vessel, and the
radius stops decreasing with the number. `--ordering` picks between:

- `generation` (default) — the main path: at a junction the widest daughter
  keeps its parent's number, only the others are incremented.
- `strahler` — counted up from the tips: a leaf is 1, and two branches of
  equal order n meeting yield n+1.
- `strahler_dd` — **the one to fit ratios on.** Diameter-defined Strahler
  (Jiang, Kassab & Fung 1994): classic Strahler is only as good as its
  leaves, and in an in-vivo mask the leaves are not real terminals, they are
  wherever the segmentation ran out of contrast — two vessels of the same
  calibre land several orders apart because one was truncated earlier. This
  variant iterates: per-order diameter mean and SD, a boundary between
  orders n and n+1 at `(Dn + SDn + Dn+1 - SDn+1) / 2`, re-order the tree
  against those boundaries, repeat until nothing moves. It runs on segments
  rather than elements and falls back to the classic rule at the top order,
  both flagged in the docstring; check them against the paper before
  quoting anything.
- `bfs_generation` — the raw junction count, kept for reference.

All four are in `--branches-csv`. Under all but the last the mean calibre
must vary monotonically, which the report checks and reports explicitly: an
inversion means a leak into a vein, two vessels fused by partial volume, or
a wrong root, so it doubles as an automatic quality check.

### Segments or elements

A **segment** is the piece of vessel between two bifurcations. An **element**
is the whole run of segments that keep the same order — which is what a
trunk becomes when a small lateral branch leaves it without raising its
order. The distinction changes both N_n and L_n, so it changes R_b and R_l;
Horsfield-ordered lengths in the literature are normally elemental, and
comparing segmental lengths against them understates L_n at every order.

Both are always computed and both ratio sets are always printed, so the
choice is never implicit. `--count segment|element` only picks which one the
per-order table and `--orders-csv` describe.

### Branching ratios

```
=== branching ratios (strahler_dd, elements) ===
  fit over orders 1..6 (pre-specified), 6 points
  ratio                              value   95% CI            R2
  R_b  branching  (10^-slope N)      2.065   [1.962, 2.173]   0.997
  R_d  diameter   (10^slope D)       1.457   [1.441, 1.472]   1.000
  R_l  length     (10^slope L)       1.364   [1.308, 1.422]   0.991
  order      :       1       2       3       4       5       6
  N          :      40      16       8       4       2       1
  D mean (mm):    3.01    4.57    6.47    9.37   13.81   20.03
  L mean (mm):    8.74   10.34   15.00   20.37   30.30   37.89
```

All three are slopes of a semi-log plot against the order, fitted per step
*towards the trunk* whichever ordering is selected. Reported with the fit
range, R², the 95% t interval on the slope carried through the same
exponential, and N per order so it is visible when a fit rests on two
branches. With five or six usable orders those intervals are wide; that
width is the result, not a failure.

The three are fitted over **one** range of orders, not three, so they stay
comparable. Fix that range with `--fit-orders MIN MAX` before looking at the
numbers — chosen afterwards it is a knob, and it is the one that most easily
turns any tree into a published value. Without it the range defaults to
every order whose mean diameter clears `--fit-min-voxels` (3 voxels, where
the distance transform stops having any dynamic range) and the output says
so.

### Pruning sensitivity

Pruning is the one free parameter no measurement constrains, and it acts on
the terminal branches — the lowest orders, the steepest end of every fit.
`--sweep-k` re-runs the whole post-skeleton stage for each value of
`--radius-factor` and tabulates the ratios against it:

```bash
python centerline.py --input artery.nii.gz --ordering strahler_dd \
  --min-branch-length 0 --sweep-k 0.5 1 1.5 2 2.5 3 --sweep-csv sweep.csv
```

Set `--min-branch-length 0`, otherwise the absolute floor does the pruning
and k does nothing (the report says so if that happens). Flat ratios across
the sweep are a result; ratios that move are a measurement of the pruning.

### Calibrating the chain

`phantom.py` builds a symmetric binary tree with R_b = 2 exactly and R_d,
R_l imposed, rasterizes it as anti-aliased capsules at a chosen voxel size,
blurs it and adds noise. Running the whole chain on it gives the bias of the
chain alone, which is the only way to tell "the segmentation misses small
vessels" from "the measuring chain compresses the dynamic range".

```bash
python phantom.py --orders 7 --spacing 1.0 --rd 1.5 --rl 1.4 \
  --output tree.nii.gz --segments-csv truth.csv
python centerline.py --input tree.nii.gz --ordering strahler_dd \
  --min-branch-length 0 --fit-orders 1 6
```

`phantom.py` prints the exact answers, including the Murray exponent
(`ln 2 / ln R_d`, closed-form because every bifurcation is symmetric), and
which orders fall under three voxels of diameter and therefore cannot be
measured at that spacing at all. Re-run it at the voxel size of the study
before quoting anything.

At 1.0 mm isotropic with a two-thirds-voxel blur, the chain returns
R_b = 2.07, R_d = 1.457 and R_l = 1.364 on elements against a truth of
2.000 / 1.500 / 1.400 — the diameter ratio comes back ~3% low because the
inscribed radius is overestimated more on thin vessels than on thick ones,
and the finest order is lost to the blur entirely.

### Measurement caveats

Lengths, tortuosities and angles are read on a smoothed centerline
(`--smoothing`, `--max-shift`), never on the raw voxel path: a digitized
path wobbles half a voxel around the true axis, which on an oblique tube
adds ~12% to its length. Directions are fitted over 2.5 local radii by
PCA -- over a couple of voxels they snap to the axes of the grid and pile
the bifurcation angles up at 90 degrees. Radii come from the distance
transform and are accurate to about half a voxel, which is enough for the
calibre but propagates hard into Murray's exponent, hence the filter.

The mask is resampled to isotropic voxels by linear interpolation of the
occupancy re-thresholded at 0.5, not by nearest neighbour: nearest neighbour
replicates the coarse voxels, so the surface keeps the steps of the input
grid and the thinning follows them.

Loops in the skeleton are cut before ordering (`--keep-cycles` to leave them
in, for inspection only). A branch caught in a loop has no well-defined
depth, so Strahler would be quietly wrong downstream of it rather than
visibly missing; the edge removed is the one whose thinner endpoint has the
smallest radius, and the count *before* cutting stays in quality metric 4.
This fixes the ordering, not the mask.

Per-branch calibre is the median radius with one local junction radius
trimmed off each end (`calibre_mm`). The plain mean over a branch
(`mean_radius_mm`, still exported) averages in the bifurcation cavities,
which inflates short branches more than long ones — i.e. high orders more
than low ones, which is exactly the direction that biases the slope of
log D against the order.

Use `--label 2` to
isolate one class of a multi-class segmentation. Pruning is controlled by
`--min-branch-length` (mm) and `--radius-factor` (a terminal branch shorter
than that many local radii is a thinning artifact); raise them if the tree
looks hairy, lower them to keep small distal vessels.

## Not included yet

The two-stage training scheme (pretraining + hard-case fine-tuning) has been
set aside in favor of a single training stage; it can be reintroduced later
if needed.
