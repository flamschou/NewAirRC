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
merged, short spurs pruned, a local radius from the distance transform, and
a generation index counted from the trunk (the widest free end, or
`--root i j k`).

```bash
python centerline.py --input artery.nii.gz
```

Like `inference.py`, `--output` is optional: the centerline mask is written
next to the input as `<name>_centerline.nii.gz`, on the input grid (so it
overlays directly on the mask). `--paint generation|branch` colors it by
generation or branch id instead of a binary mask. The other outputs are
opt-in: `--csv` has one row per point (voxel index, world mm, radius),
`--branches-csv` one row per branch (length, chord, tortuosity, radii,
generation, Strahler order), `--orders-csv` and `--bifurcations-csv` the two
analysis tables below, and `--vtk` a legacy polydata for Slicer/ParaView.

```bash
python centerline.py \
  --input artery.nii.gz \
  --output artery_centerline.nii.gz \
  --csv centerline_points.csv \
  --branches-csv centerline_branches.csv \
  --orders-csv centerline_orders.csv \
  --bifurcations-csv centerline_bifurcations.csv \
  --vtk centerline.vtk
```

### Anatomical report

Every run prints a morphometric report (`--no-report` to skip it):

- **per order** — branch count, how many end there, total and mean length,
  mean radius at the proximal end / distal end / over the branch,
  tortuosity, and the mean radius at the tips of that order. Followed by
  the calibre monotonicity check (see below).
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
- **tree** — tortuosity and length distributions, growth ratio up to the
  widest generation, and the number of loops in the skeleton (two vessels
  touching in the mask; they also shift the generations downstream, so a
  high count means the segmentation should be checked).

### How branches are numbered

Counting junctions from the root is not anatomical: a trunk giving off
collaterals gets renumbered at every one of them, so the interlobar artery
ends up labelled "generation 16" while still being the same vessel, and the
radius stops decreasing with the number. `--ordering` picks between:

- `generation` (default) — the main path: at a junction the widest daughter
  keeps its parent's number, only the others are incremented.
- `strahler` — counted up from the tips: a leaf is 1, and two branches of
  equal order n meeting yield n+1.
- `bfs_generation` — the raw junction count, kept for reference.

All three are in `--branches-csv`. Under the first two the mean calibre must
vary monotonically, which the report checks and reports explicitly: an
inversion means a leak into a vein, two vessels fused by partial volume, or
a wrong root, so it doubles as an automatic quality check.

### Measurement caveats

Lengths, tortuosities and angles are read on a smoothed centerline
(`--smoothing`, `--max-shift`), never on the raw voxel path: a digitized
path wobbles half a voxel around the true axis, which on an oblique tube
adds ~12% to its length. Directions are fitted over 2.5 local radii by
PCA -- over a couple of voxels they snap to the axes of the grid and pile
the bifurcation angles up at 90 degrees. Radii come from the distance
transform and are accurate to about half a voxel, which is enough for the
calibre but propagates hard into Murray's exponent, hence the filter.

Use `--label 2` to
isolate one class of a multi-class segmentation. Pruning is controlled by
`--min-branch-length` (mm) and `--radius-factor` (a terminal branch shorter
than that many local radii is a thinning artifact); raise them if the tree
looks hairy, lower them to keep small distal vessels.

## Not included yet

The two-stage training scheme (pretraining + hard-case fine-tuning) has been
set aside in favor of a single training stage; it can be reintroduced later
if needed.
