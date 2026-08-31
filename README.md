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
- `config_loss.py` — deep-supervision-weighted Dice+CE loss, plus the
  optional clDice continuity term (see *Adding the continuity term*).
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

`finetune_cldice.slurm` is the same job with the continuity term switched on
— see the next section.

### Dataloader workers and `/dev/shm`

A training batch is not `BATCH_SIZE` volumes: `RandCropByPosNegLabeld` returns
`num_samples` patches per item and `list_data_collate` flattens them, so the
loader hands out `BATCH_SIZE x num_samples` = **24 patches of 128³**, image and
label, ~0.4 GiB per batch. Every prefetched batch sits in `/dev/shm` until the
main process consumes it, and `train.py` selects the `file_system` sharing
strategy, under which those segments are files.

At torch's default `prefetch_factor=2` with 8 workers on each of the two
loaders, that reserves ~10 GiB of `/dev/shm` at steady state, and the 28-step
epochs respawn all 16 workers every ~35 seconds, each generation leaving
segments to be reclaimed. That combination exhausts `/dev/shm` mid-run on a
node whose shared memory is sized from the job's memory cgroup; it surfaces as
a `RuntimeError: unable to open shared memory object` from the pin-memory
thread, not as anything resembling a data problem.

Three settings control it, all in `config.py`:

| Setting | Default | |
| --- | --- | --- |
| `NUM_WORKERS` | `8` | training loader workers |
| `VAL_NUM_WORKERS` | `NUM_WORKERS // 4` | the validation loader runs 6 batches an epoch against the training loader's 28; it does not need the same fleet |
| `PREFETCH_FACTOR` | `1` | batches held ready per worker. The main `/dev/shm` knob |

plus `persistent_workers=True` on both loaders, which stops the respawn churn.
Together these bring steady-state shared memory to ~3.6 GiB. If a node still
runs out, `PREFETCH_FACTOR` and `VAL_NUM_WORKERS` are the two to turn down
first; check what the node actually offers with `df -h /dev/shm` on it.

Note that persistent workers keep all `NUM_WORKERS + VAL_NUM_WORKERS`
processes alive for the whole run rather than only during their own loader's
epoch, so host RAM use becomes constant instead of intermittent — each worker
carries a full torch/monai import.

## Adding the continuity term (clDice)

Dice and cross-entropy are voxel-wise. A two-voxel break in a vessel costs
them almost nothing, and yet it splits the tree in two: downstream, every
branch past the break is reassigned to a different component, which shifts
its Strahler order and corrupts the branching ratios `centerline.py`
measures. Whole-tree Dice is simply not the metric that sees this.

`config_loss.py` implements **clDice** (Shit et al., CVPR 2021) as an
optional extra term:

```
loss = DiceCE + CLDICE_WEIGHT * clDice
```

clDice compares the *skeletons* of the masks instead of the masks. It is the
harmonic mean of a topology precision (how much of the predicted skeleton
falls inside the true mask) and a topology sensitivity (how much of the true
skeleton falls inside the predicted mask); a gap removes a whole run of true
skeleton from the second term, so the term sees the break that Dice missed.
The skeletonization is *soft* — iterated min/max pooling — so the whole thing
stays differentiable and needs no post-processing.

Settings, all in `config.py` and all overridable from the environment:

| Setting | Default | What it does |
| --- | --- | --- |
| `CLDICE_WEIGHT` | `0.0` | lambda above. **0 disables the term** and reproduces the original loss exactly. `0.5` is the clDice paper's value for tubular structures. |
| `CLDICE_ITERATIONS` | `6` | soft-skeletonization iterations. Must be >= the largest vessel radius in voxels, or thick vessels never thin down to a curve: ~10 mm across at `TARGET_SPACING = 1 mm` is a radius of 5, plus one for margin. |
| `CLDICE_WARMUP_EPOCHS` | `20` | epochs over which the weight ramps linearly from 0. The skeleton of a not-yet-vessel-shaped prediction is noise, and clDice on noise pulls towards thin fragmented masks; the ramp keeps the term quiet until there is something for it to fix. |
| `CLDICE_MAX_PATCHES` | `8` | how many patches of the batch the term scores (0 = all). See *Why this term is memory-hungry* below. |

Two deliberate restrictions in the implementation: the term is applied to the
**full-resolution deep-supervision level only** (at 1/2 and 1/4 resolution the
thinnest vessels are sub-voxel, so their "skeleton" is a downsampling
artifact), and it is computed in **fp32** even under `precision="16-mixed"`
(the pooling chain ends in differences of nearly-equal numbers).

### Why this term is memory-hungry

Worth knowing before turning any of the knobs up, because it OOMed a 93 GiB
H100 on the first try. The thinning is a chain of ~20 pooling and elementwise
ops per iteration, and autograd keeps a full-resolution copy of nearly every
intermediate. The batch that reaches the loss is not `BATCH_SIZE`:
`RandCropByPosNegLabeld` returns `num_samples` patches per item and
`list_data_collate` flattens them, so the default setup hands the loss **24
patches of 128³**. A naive implementation keeps **79 GiB** of graph for that,
on top of the network's own activations.

Three things bring it down to ~2 GiB, none of which changes the value of the
loss:

| | graph size |
| --- | --- |
| naive implementation | 79.3 GiB |
| + erosion reused across iterations (the textbook loop erodes twice per iteration, but the second erosion of iteration *j* is the first of *j+1*) | 51.2 GiB |
| + gradient checkpointing per thinning iteration | 6.6 GiB |
| + `CLDICE_MAX_PATCHES=8` | 2.2 GiB |

The patch cap is the only one that changes the loss *statistically*: clDice
is a batch-level ratio, so scoring 8 of the 24 patches is a noisier estimate
of the same quantity, not a different quantity. The subset is drawn at random
while training and is the first 8 patches in eval, so `val_loss` stays
reproducible. The other two are exact — the skeleton is bit-identical to the
textbook formulation and the gradients match to 0.

With all three, the term costs about **+4%** per training step.

### Fine-tuning rather than retraining

The term is meant to be added to an already-trained model, not trained from
scratch with: clDice only starts saying something useful once the
segmentation is roughly right, and a fine-tuning run is short enough to sweep
`CLDICE_WEIGHT` and `CLDICE_ITERATIONS` over. Set `FINETUNE_FROM` to a
checkpoint and `train.py` loads its **weights only**:

```bash
sbatch finetune_cldice.slurm
```

which is `train.slurm` plus, in the environment:

```bash
EXPERIMENT_NAME="${BASE_EXPERIMENT}_cldice_ft"   # own logs/checkpoints
FINETUNE_FROM=".../checkpoints/$BASE_EXPERIMENT/run/last.ckpt"
CACHE_NAME="$BASE_EXPERIMENT"                     # share the dataset cache
CLDICE_WEIGHT=0.5
CLDICE_MAX_PATCHES=8
LEARNING_RATE=1e-4                                # a tenth of the baseline
MAX_EPOCHS=150
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Three things that would otherwise be easy to get wrong, and that the code now
handles:

- **weights only, not a resume.** `FINETUNE_FROM` is deliberately not passed
  as `Trainer.fit(ckpt_path=...)`, which would also restore the source run's
  optimizer state and epoch counter — and with the epoch counter its PolyLR
  schedule, already decayed to ~0 at the end of that run. The fine-tuning
  would train at `lr = 0` and nothing would move. Rebuilding the schedule
  from `LEARNING_RATE`/`MAX_EPOCHS` restarts the decay at 1e-4 instead.
- **auto-resume still wins.** If the fine-tuning run's own `last.ckpt` exists,
  it is resumed normally, so a preempted job picks up where it stopped rather
  than restarting from the base weights.
- **the dataset cache is shared.** `CACHE_DIR` is scoped to `EXPERIMENT_NAME`
  so that a change in preprocessing cannot silently reuse stale cached
  tensors. A fine-tuning changes no preprocessing, so `CACHE_NAME` points it
  back at the base experiment's cache instead of paying for a full re-cache.

Keep one from-scratch run (`CLDICE_WEIGHT=0.5` in `train.slurm`, no
`FINETUNE_FROM`) for the final comparison if the numbers are going in a
paper: "DiceCE vs DiceCE+clDice" is only an honest comparison at equal budget
from the same initialization.

### Judging whether it worked

`val_dice_metric` will barely move — it is not the metric this term targets,
and `val_loss` is not comparable across the warm-up epochs either (the
`cldice_weight` scalar is logged to TensorBoard so you can see where the ramp
was). Judge it on full volumes, via `inference.py`, with:

- `connectivity.py` — number of connected components and the fraction of the
  mask held by the largest one;
- `compare_predictions.py` — component counts and surface metrics of the
  fine-tuned predictions against the baseline's, no ground truth needed;
- `centerline.py --orphans-csv --orders-csv` — the one that actually matters
  here: how much of the tree falls outside the main component, and how many
  Strahler orders survive.

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

## Cutting a segmentation back to its large vessels

A whole-tree Dice is dominated by the periphery: the subsegmental vessels
carry most of the surface and almost none of the volume, they are where the
reference itself is least certain, and a model that draws the hilum
perfectly but fades out two generations early scores about the same as one
that does the opposite.

`truncate.py` cuts a segmentation back to its proximal tree -- trunk, lobar
and segmental vessels -- and writes the truncated mask. It computes no
metric: it writes masks, and the masks go to whatever already scores them
(`compare_predictions.py`, or the validation loop). Run it on the
references **and** on the predictions, with the same `--min-diameter`, so
both sides are cut by the same rule -- a truncated reference scored against
a whole prediction would count every peripheral vessel of the prediction as
a false positive.

The mask is skeletonized and measured exactly as `centerline.py` does it,
then the tree is truncated: a branch has to clear `--min-diameter` (default
5 mm, which sits inside the segmental range rather than at its lower edge,
where two segmentations of one tree disagree most), and
what is kept is the connected group of such branches that contains the
**widest branch of the tree**. Clearing the floor is necessary and not
sufficient, which is what makes this a truncation rather than a calibre
filter: a wide distal blob sitting behind a thin branch -- a leak, a fused
vein -- cannot reach the trunk through large vessels, so it goes with the
branch that carries it.

Growing from the widest branch rather than down from the root is deliberate.
The root is chosen upstream as the widest free *end* of the skeleton
(`centerline.order_branches`), which is the trunk in a clean mask and
anywhere at all in a degraded one: on an eroded venous prediction here it
landed on a 5 mm peripheral stump whose daughters measured 3.83 mm, the real
37 mm trunk sat six generations "below" it, and a root-first traversal kept
1 segment out of 203. Seeding from the widest branch cannot fail that way,
and it returns exactly the same cut whenever the root is right.

The truncated tree becomes voxels again through a Voronoi partition of the
mask by its own skeleton: a voxel is kept when its nearest centerline node
belongs to a kept branch and sits within `--sleeve` local radii (1.5 by
default). The cut surface therefore falls where the tree was truncated,
roughly normal to the vessel, and a subsegmental vessel running along the
trunk is claimed by its own centerline instead of surviving because it
happens to touch a large one.

What comes out carries the input's own label values on the input grid, so
it is a drop-in replacement for the file it came from:

```bash
# one class of one mask -> vascular_case001_large.nii.gz beside it
python truncate.py --input vascular_case001.nii.gz --label 3

# every foreground class of config.LABEL_CLASS_MAP, each on its own tree,
# into one file (raw 3 = artery, 4 = vein, airway classes 1-2 dropped)
python truncate.py --input vascular_case001.nii.gz --classes

# the labels of a manifest split
python truncate.py --manifest manifest_ct.json --split val --output-dir cut/

# the predictions that go with them, cut by the same rule
python truncate.py --input-dir /data/flamant/data/ct/lidc_idri \
                   --pattern '*_vascular_pred.nii.gz' --classes
```

Each class is truncated on its own tree -- the union of the arterial and the
venous tree is not a tree, and skeletonizing it would order the two against
each other. Files already carrying `--suffix` are skipped by `--input-dir`,
and a file whose output exists is skipped unless `--overwrite` is given, so
an interrupted run resumes. `--csv` records what was cut out of each file:
segments and centerline length kept, volume kept, and the floor that was
applied.

Every mask is written with a `<stem>_large.json` sidecar holding the rule it
was cut by -- the floor, the sleeve, the classes, the skeleton settings. A
filename cannot carry that (`..._large.nii.gz` is the name of a 4 mm cut of
the artery and of a 6 mm cut of both trees alike), and `compute_dice.py`
reads the sidecar to decide whether a cut already on disk is the one being
asked for.

The floor is what "large vessel" means here, so report it with whatever the
truncated masks end up scoring, and use `--suffix` to keep two floors side
by side (`--min-diameter 6 --suffix _large6mm`). `--max-generation` (depth
from the root) and `--min-strahler` (order counted up from the tips) are
available as extra cuts, both off by default -- the depth of a branch is a
property of the skeleton's topology, which one missed junction changes,
whereas its calibre is a measurement.

## Dice on the whole tree and on the large vessels

`compute_dice.py` scores a checkpoint's predictions against the references
of a manifest split, twice: on the whole tree, and on the large vessels
alone.

```bash
# predictions already produced by inference.py
python compute_dice.py --manifest manifest_ct.json --split val --csv dice.csv

# or straight from the checkpoint, predicting whatever is missing on the way
python compute_dice.py --manifest manifest_ct.json --split val \
                       --checkpoint "$DATASET_ROOT/checkpoints/.../last.ckpt" \
                       --csv dice.csv

python compute_dice.py --manifest manifest_ct.json --classes artery vein \
                       --output-dir results/ --min-diameter 6
```

Predictions are read off the disk, where `inference.py` writes them
(`<image stem>_vascular_pred.nii.gz`, or under `--pred-dir`). With
`--checkpoint`, the missing ones are produced here first, through
inference.py's own sliding window and preprocessing, so the split can be
scored in one command -- and only the missing ones: a prediction already on
disk is reused and never overwritten, since it is the thing being measured
and regenerating it halfway through a cohort would change what the cohort
means. The checkpoint is loaded lazily, so a run over predictions that all
exist never pays for torch at all. `--cpu` forces the inference off the GPU;
a CUDA out-of-memory falls back to the CPU by itself, as in `inference.py`.

A manifest records where the data was when it was written -- these ones say
`/data/flamant/data/ct` -- and the volumes outlive that path: a cluster
mount, a copy on another filer, a scratch directory. `--rewrite OLD=NEW`
maps the prefix at read time (repeatable, first match wins, the file is not
modified), so one manifest holds one definition of the split across all of
them:

```bash
python compute_dice.py --manifest manifest_vibe.json --split val \
                       --rewrite /data/flamant/data/ct=/biomaps/spiro3d/.../ct \
                       --checkpoint .../last.ckpt --csv dice.csv
```

Editing the manifest per machine instead is how two runs end up disagreeing
about which cases are validation. When every case is skipped, the exit
message names the root the manifest asked for and writes the `--rewrite`
line for you.

`--rewrite` is exact when the tree simply *moved*: one prefix substitution,
and every case either resolves or visibly does not. When the layout changed
too and no prefix fits, `--data-dir DIR` is the fallback -- it walks the
directory and matches the manifest's files by name, the way `inference.py`
takes `--input-dir`. A name found in two places under it is an error rather
than a guess.

Going through `compute_dice.py --checkpoint` rather than
`inference.py --input-dir` also avoids a trap: `--input-dir` globs every
`*.nii.gz` under the directory and only excludes the files already carrying
the prediction suffix, so it will happily run the model on your label
volumes. The manifest says which files are images.

The large-vessel pass truncates **both sides** with the same rule and writes
the truncated masks out (`truncate.py` does the cutting, sidecar included),
so the number can be traced back to the volumes it was read on. Cutting only
the reference would count every peripheral vessel of the prediction as a
false positive, and the Dice would measure the truncation rather than the
model. Cutting the prediction with its own tree does mean a prediction whose
trunk is broken gets a different truncation from the reference's -- that is
the failure showing up, not a defect of the metric, but it is why each row
keeps `n_segments_kept_reference` next to `n_segments_kept_prediction`: a
case where those two disagree wildly is a case to look at before quoting its
Dice.

Two volumes are written per pass, to say **where** the two masks differ:

- `<case>_<class>_errors_{full,large}.nii.gz` -- a label map, 1 = agreement,
  2 = predicted only (false positive), 3 = reference only (false negative).
  Exact, no parameter, opens straight into Slicer as a segmentation.
- `<case>_<class>_localdice_{full,large}.nii.gz` -- the local Dice: at every
  voxel, the Dice of the two masks inside a `--heat-window` cube (20 mm by
  default) around it, computed with box filters. It is the global Dice
  decomposed in space, so **low is bad** -- the opposite convention from the
  error map, do not read the two with the same colour scale. NaN where there
  is not enough vessel in the window to divide by.

`local_dice_p10` and `local_dice_median` in the CSV are that map read at the
mask voxels themselves -- how bad the bad regions are, next to the single
average the Dice reports. They are not taken over the whole field: the map
extends a window's reach around the masks, so most voxels carrying a value
sit in the fringe where the window catches a vessel by its edge, and the
median over the field reads 0.0 next to a global Dice of 0.8.

`--swap-av` corrects a checkpoint trained with the inverted artery/vein
convention, by reading the artery out of the vein index rather than by
rewriting anything. Truncated masks already on disk are reused when their
sidecar matches the current settings, so a second run -- different maps, a
different window -- costs seconds instead of re-skeletonizing.

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
`--ratios-csv`, `--bifurcations-csv`, `--orphans-csv` and `--sweep-csv` the
analysis tables below, and `--vtk` a legacy polydata for Slicer/ParaView.

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
### Orphan components

Metric 1 says how much of the tree is not analysed. It does not say why, and
"52% connectivity" aggregates three unrelated defects with three different
corrections and three different costs. They are reported separately:

```
=== orphan components (39 outside the main tree, 463 mm) ===
  severed      8 comps     309.3 mm  66.8%  median r 1.91 mm    A/V classification cut the pedicle
      rejoins the trunk as soon as the class boundary is ignored, so the vessel is there and
      correctly classified -- repairable in post-processing, no retraining
  hole         5 comps      73.4 mm  15.8%  median r 1.50 mm    a real hole, in neither class
      the vessel is missing over several millimetres: a frank false negative, from low contrast or
      motion. Retraining, or geometric bridging if you accept it
  dust        26 comps      80.7 mm  17.4%  median r 1.50 mm    speckle under the resolution floor
      too short to be vessel and sitting on the quantization floor -- a size filter removes it
      with no argument
```

`--dust-length` (10 mm) draws the speckle line, and size is judged first: a
two-millimetre speck that happens to touch the other class is speckle, not a
severed pedicle, and letting it in would inflate the very number the repair
is judged on. `severed` versus `hole` is decided by the union test below, so
without a comparison mask both collapse into `detached`.

Underneath, the per-component measurements. The distance to the main tree is
wall to wall rather than centreline to centreline — two vessels whose axes are
4 mm apart are touching if both are 2 mm across.

```
=== orphan components (39 outside the main tree) ===
broken off (wall gap <= 3.0 mm):   11 components,     55.2 mm, median radius 1.50 mm
isolated   (wall gap >  3.0 mm):   28 components,    408.2 mm, median radius 1.50 mm
  of the 13 over 10 mm: 1 broken off (19 mm), 12 isolated (364 mm)
  len_mm  n_pts  med_r  max_r  gap_wall  gap_axis  fragment voxel      nearest on trunk
    80.2     59   1.91   2.74      9.22     13.64  (  73,  68, 176)  (  72,  72, 189)
    78.3     57   1.91   2.50      3.81      7.81  ( 184, 156, 128)  ( 178, 160, 125)
```

`--orphan-gap` sets the threshold (3 mm). Both ends of each gap are printed
as input-grid voxels so they open directly in a viewer, which is the only way
to settle the ambiguous ones; `--orphans-csv` exports all of them, and the
two totals go into `--quality-csv`.

**That threshold alone does not settle anything** and the report says so: an
artery that dropped out over a centimetre of poor contrast — routine in VIBE
near the diaphragm — lands on the "isolated" side of any reasonable gap. Two
measurements do settle it.

#### The bridging curve

`--bridge-sweep [MM ...]` dilates the mask by each radius (one voxel to six
if no value is given) and reports how much centerline ends up in the largest
component. Dilating by r closes any gap up to 2r, since both walls advance.

```
  dilation  closes gaps  mask comps  centerline comps  merged  in largest  of missing
    0.00 mm      0.00 mm          43                40       1       95.3%        0.0%
    1.00 mm      2.00 mm          30                28      13       95.9%       12.6%
    2.00 mm      4.00 mm          17                17      24       98.2%       60.7%
    3.00 mm      6.00 mm          14                14      27       98.7%       72.2%
  knee at 2.00 mm of dilation (4.00 mm of gap closed): 48% of the missing
  centerline reattaches in that single step.
```

The last column is the share of the *missing* length recovered, not the raw
fraction — a tree already at 95% cannot gain ten points however cleanly its
fragments reattach, so the raw fraction would call every well-connected tree
knee-less. A sharp knee means the fragments were separated vessel and gives
the bridging radius the data supports; a smooth climb with no knee means they
are genuinely elsewhere and dilating merely glues unrelated objects together.
`--bridge-csv` exports it.

#### Cross-check against the venous prediction

Two forms. With a multi-class segmentation, name the two classes and the
second tree is read out of the same file:

```bash
python centerline.py --input av_seg.nii.gz --label 4 --compare-label 3 \
  --bridge-sweep --orphans-csv orphans.csv
```

With two files, `--compare-mask veins.nii.gz` (plus `--compare-label` if that
file is itself multi-class); it has to be on the input grid. Either way, and
with `--compare-dilate`, this adds three columns per orphan: `compare_gap_mm` (wall to wall), 
`compare_overlap` (share of the fragment's centerline inside the other mask
once dilated) and `nearer`. A long, coherent, well-calibred fragment that is
not attached to the arterial trunk but *does* lie inside the venous one is
not a false positive at all — it is an A/V labelling error, and the fix is the
classification head rather than the sensitivity.

Read the two columns differently. High overlap is evidence. Mere proximity is
not: arteries and veins run alongside each other everywhere in the lung, so
`nearer = compare` is nearly free, and a fragment touching the venous mask
without lying inside it is a kissing-vessel geometry rather than a swap. The
report makes that distinction explicitly.

#### The severed pedicle

Neither test above sees the most common case. A fragment can be correctly
classified, and the trunk too, while the few millimetres of vessel joining
them were given to the *other* class: at a crossing the two trees run so
close that the separation is a coin toss, and losing it cuts the vessel in
two. The fragment is then an orphan not because it is wrong and not because
there is a hole, but because its pedicle was taken.

Whenever a comparison mask is available this is tested with no threshold at
all, by labelling the union of the two classes: if a fragment and the trunk
are separate in one class and joined in the union, the cut ran through the
other class. The `union` column says `joins` or `hole` per fragment and the
totals go into `--quality-csv`.

```
  through label 3 of the same file: 318 mm in 11 components rejoin the trunk once the
  two classes are taken together, 145 mm in 28 do not
  of the ones over 10 mm: 309 mm rejoin, 73 mm do not
  the dominant defect is therefore a severed pedicle, not a missing vessel [...]
  the fix is the A/V classification head, not the sensitivity
```

#### The symmetric control

Run it both ways and concatenate the two `--quality-csv` rows:

```bash
python centerline.py --input av.nii.gz --label 1 --compare-label 2 --quality-csv qual_A.csv ...
python centerline.py --input av.nii.gz --label 2 --compare-label 1 --quality-csv qual_V.csv ...
```

`--quality-csv` opens with a structural block — `mask_volume_ml`,
`n_segments`, `n_elements`, `n_leaves`, `total_length_mm`, `max_order`,
`ordering` — precisely so those two rows are a readable pair:

```
                                         label 4     label 3
mask_volume_ml                          194.8770    215.0660
n_segments                                   620         754
n_leaves                                     325         387
max_order                                      8           7
largest_component_length_fraction         0.9530      0.9424
orphan_length_severed_mm                309.3434    529.4932
orphan_length_holed_mm                   73.3833    107.4283
```

One tree severed and the other clean means the classifier hands crossings
preferentially to one class; both severed means the exchanges go both ways;
neither severed means the continuity problem is not an A/V confusion at all.
And a volume ratio far from 1 between the two classes is itself a finding —
the two trees drain the same parenchyma, so a 2:1 imbalance is a class bias
or a leak, not anatomy, and it would explain a one-sided severing directly.

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

One more warning to expect: the report flags the lowest fitted order when its
mean diameter sits on the censoring boundary. Under 1.5 voxels of radius the
distance transform *cannot* return less — the distribution is truncated from
below, not merely imprecise — and that order anchors the steep end of every
fit. Pre-specify a range stopping one order earlier and check the ratios do
not move.

**These numbers are not comparable to the literature unless the ordering is
Strahler.** A generation step is one bifurcation; a Strahler step is several,
because the order only rises where two branches of equal order meet. On one
and the same tree, the three ratios counted in generations are therefore
mechanically smaller than their Strahler counterparts, and an R_d of 1,3
against a published 1,56 may be entirely the ordering. The report warns
loudly when a peripheral ordering is selected.

The other warning to watch for: counted in generations, a truncated tree
gives N rising and then falling — a hump, not a line — because the deep
generations are cut off before the tree would naturally thin out. Fitting a
slope through that returns an R_b, an interval and an R², all describing the
parabola. The report detects the interior peak and says so; either order by
Strahler or start the range past the peak.

The three are fitted over **one** range of orders, not three, so they stay
comparable. Fix that range with `--fit-orders MIN MAX` before looking at the
numbers — chosen afterwards it is a knob, and it is the one that most easily
turns any tree into a published value. Without it the range defaults to
every order whose mean diameter clears `--fit-min-voxels` (3 voxels, where
the distance transform stops having any dynamic range) and the output says
so.

Those three voxels are counted **on the coarse axis of the acquired grid**,
not on the isotropic grid the mask is resampled onto. Upsampling adds samples,
not information: a vessel two slices across does not become resolved by being
interpolated onto four, and a floor counted after resampling is the anisotropy
ratio too permissive. The order it lets in is censored from below — its
diameter is the grid, not the vessel — and it anchors the steep end of every
slope. On a 1.19/0.80/1.31 mm acquisition that single admitted order moves
R_d by 5.6 %. On an isotropic input the three axes agree and the criterion is
the one it has always been. `--fit-min-diameter` overrides the rule in
millimetres outright.

### The other end of the range: orders carried by one element

The diameter floor censors the thin end of the fit. The trunk end needs
censoring too, and for a different reason.

At the top of a Strahler tree an order holds one or two elements — the
topology guarantees it, the top order holds exactly one. That is not a
sampling accident that a better acquisition would fix, and its consequences
are mechanical: such an order's SD is zero by construction, and its mean
diameter is the median calibre of a single row.

Which would be fine if that row were a vessel. Often it is not. An element is
the run of segments that keep the same order, and near the hilum a chain can
keep its order across the central junction: four segments, aggregated into
one element, entering at r = 3.8 mm and leaving at r = 11.2 mm. Its
`calibre_mm` — a median — reports 7.1 for a run whose radius triples. Nothing
in the per-order table shows this; the order simply comes out with a mean
diameter that violates monotonicity against the order below it, and the
temptation is to blame the skeletonization, which is innocent.

`--fit-min-branches N` (default 3) drops any order carried by fewer than N
segments or elements. It is mechanical, declarable in advance, and symmetric
with the diameter floor: unreliable orders are removed from both ends of the
range rather than from the thin end only. `--fit-min-branches 1` disables it.

It costs R_b something, and the cost is real: N is an exact count, not an
estimate, and the trunk-ward orders this removes are the ones anchoring the
log N line. The three ratios are fitted over one single range by
construction, so the choice is between paying that and quoting ratios that do
not rest on the same orders. Dropping the top two orders also leaves fewer
points, so the *confidence interval widens* even as R² improves — a fit over
four orders carries two degrees of freedom and a t multiplier of 4.30.
Both numbers are the result.

`cohort.py` refuses a set of files that mixes two values of the floor, the
same way it refuses a set that mixes orderings.

### The range must stay contiguous

Censoring orders out of the middle leaves a hole, and a hole is worse than
the order it removed. The slope is fitted against the order **number**, so a
surviving island above the gap sits far from the mean of what is left and
carries leverage in proportion to that distance squared.

Concretely: drop order 5 from a range of 1..6 and the lone point at 6 is 2.8
orders off a mean of 3.2. It supplies 7.84 of the 14.8 total scatter — more
than half the weight deciding the slope — and it is by construction the least
reliable point in the set, since it survived only because the filter that
removed its neighbour did not quite reach it. On one subject that produced
R_b = 1.924, below the theoretical floor of 2 for a binary tree, with
R² = 0.803. Cutting back to 1..4 gave R_d = 1.500, R² = 0.987.

So whatever the two floors remove, the range is then cut back to the orders
consecutive with the **lowest** survivor. Lowest and not longest: for a
Strahler ordering the low end is the periphery, where an order holds hundreds
of branches, and the high end is the trunk, where it holds one or two.
Anchoring low keeps the reliable block; taking whichever run happens to be
longer would make the range depend on where the holes fell, which is the
data-dependence the pre-specification exists to rule out.

The cost is that a hole low in the range can leave fewer than three points,
and then there is no fit at all. That is the correct failure: it is reported
as `at least 3 are needed`, rather than a slope quietly drawn across a gap.

### Which elements are not vessels

The direct check is `--element-flare` (default 1.5): every element whose
distal calibre exceeds its proximal one by more than that factor is listed
before the ratios, worst first, with its order, its segment count and its
tortuosity. `flare` is also a column in `--elements-csv`, and `max_flare` /
`n_flared` in `--orders-csv`.

A vessel tapers. It never widens by half over its own length, so a flared
element is a heterogeneous aggregation and nothing else. This is more
specific than tortuosity, which catches the same elements sometimes and for
the wrong reason — a real vessel is allowed to wander around a junction; the
1.72 tortuosity of a spliced chain is the detour around the carrefour, not a
serpentine artery.

Only elements of two segments or more are eligible. A one-segment element was
aggregated from nothing, so whatever its end calibres do, the answer cannot
be that the grouping put two vessels in one row. Left in, those rows swamp
the list: on the bundled LIDC example, 46 of 52 hits are single-segment
order-1 twigs and the eight that mean something disappear among them.

**A flared element leaves the statistics; the order that held it does not.**
The tree keeps it — removing it from the graph would change the ordering —
but it comes out of the averages and out of the fit, one row at a time, on
the same footing as `--max-synthetic`.

Excluding the whole order instead scales the wrong way. The larger an order,
the likelier it holds at least one flared element, so a presence-based rule
preferentially destroys the orders worth fitting. Measured on the bundled
example:

| order | elements | flared | share |
|------:|---------:|-------:|------:|
| 1 | 1084 | 4 | 0.37 % |
| 3 | 72 | 3 | 4.2 % |
| 4 | 28 | 1 | 3.6 % |
| 5–8 | 25 | 0 | 0 % |

Dropping orders 1, 3 and 4 for eight bad rows out of 1184 left the fit a
single point. Removing the eight rows moved R_d from 1.347 to 1.346.

### Orders that are not one population

`calibre_spread` (in `--orders-csv`) is the 90th percentile of an order's
member calibres over the 10th. It is the *other* way an order fails to be a
summary, and it is not flare: flare is measured **along** a row that turns
out to be two vessels, spread **across** rows that are not the same size.
Neither implies the other, and segments — which can never flare, having been
aggregated from nothing — can still spread badly. That is the case the flare
check cannot see.

An order wider than 2× across that band is reported under the ratio table and
left in, since unlike flare there is no single culprit row to point at.
Percentiles and not max-over-min: an order of 1274 segments has two extreme
rows whose ratio is 8 or 11 and says nothing about the other 1272. Measured
that way every order of the example looks broken; on the 10–90 band they run
1.24 to 1.82 and the statistic starts discriminating.

Since `strahler_dd` promotes on calibre, an order that spreads is telling you
the promotion failed there.

### Ratios below their floor

R_b < 2, R_d < 1 and R_l < 1 are each flagged under the table. A binary tree
cannot branch less than 2:1, and vessels neither narrow nor shorten towards
the trunk, so a ratio under its floor did not measure a tree.

**The diagnosis is per ratio.** The three share a proximate cause — branches
that were never segmented — and nothing else:

- **R_b < 2** — missed lateral branches collapsing the counts at the low
  orders. The periphery is where a segmentation loses the most, so N there is
  undercounted, the log N line flattens, and the count ratio per step comes
  out under a floor no tree can cross.
- **R_d < 1** — an ordering that failed, not a tree that narrows. Either the
  diameter-defined promotion put wide vessels in low orders (check
  `calibre_spread`) or an order near the trunk is a heterogeneous aggregate
  whose median calibre understates it (check the flare table).
- **R_l < 1** — order-1 elements running long because the bifurcations that
  should have interrupted them were never segmented, while every order above
  them is bounded by junctions that were. The trunk end is shortened at the
  same time by the field of view, since the top element starts at the edge of
  the mask rather than at an anatomical origin.

**The R² decides whether there is anything to diagnose at all**, against a
hard `TREND_R2 = 0.7` and not the reader's judgement. Above it the trend is
real and the ratio is a quantified measurement of how much of the tree went
missing — L̄ of 23.32, 22.04, 17.59, 14.63 on one subject gives R_l = 0.850
with R² = 0.954, and that should be quoted as the symptom it is. Under it the
points do not lie on a line: the floor breach is scatter, the ratio is not
interpretable either way, and what deserves attention is the scatter rather
than the value.

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

`phantom.py` builds a binary tree whose ratios are known, rasterizes it as
anti-aliased capsules at a chosen voxel size, blurs it and adds noise.
Running the whole chain on it gives the bias of the chain alone, which is the
only way to tell "the segmentation misses small vessels" from "the measuring
chain compresses the dynamic range".

```bash
python phantom.py --orders 7 --spacing 1.0 --rd 1.5 --rl 1.4 \
  --output tree.nii.gz --segments-csv truth.csv
python centerline.py --input tree.nii.gz --ordering strahler_dd \
  --min-branch-length 0 --fit-orders 1 6
```

`phantom.py` prints the exact answers, including the Murray exponent
(`ln 2 / ln R_d`, closed-form at the symmetric junctions), and which orders
fall under three voxels of diameter and therefore cannot be measured at that
spacing at all. Re-run it at the voxel size of the study before quoting
anything.

At 1.0 mm isotropic with a two-thirds-voxel blur, the chain returns
R_b = 2.07, R_d = 1.457 and R_l = 1.364 on elements against a truth of
2.000 / 1.500 / 1.400 — the diameter ratio comes back ~3% low because the
inscribed radius is overestimated more on thin vessels than on thick ones,
and the finest order is lost to the blur entirely.

Three things decide whether a calibration run describes the study or a
different experiment.

**Give the acquired voxel, not the resampled one.** `--spacing` takes three
values as readily as one. The chain upsamples an anisotropic acquisition to
its finest axis, and an isotropic phantom rasterized at that finest axis has
information the study never had: its boundary is known to 0.8 mm in all three
directions where the study's is known to 0.8 mm along one axis and to the
slice thickness along the others. Same working grid, different information —
and it is the difference that leaves a measured R_d in a range instead of at
a value.

```bash
python phantom.py --spacing 1.25 0.799 1.25 --orders 7 --output tree.nii.gz
```

The point spread function follows the grid: `--blur` defaults to two thirds
of the voxel *along each axis*, because a thick slice is integrated over its
thickness and not merely sampled coarsely. The diameter floor of the fit
follows it too, on both sides of the comparison: `calibrate.py` hands
`centerline.py` a plain voxel count and `centerline.py` applies it to the
coarse axis of whatever it was given, so the phantom and the real data are
floored by one rule with one implementation. Fixing a ratio instead — passing
the 4.93 working voxels that one anisotropy happens to need — would cut too
high on a rounder volume and too low on a flatter one, which across a cohort
is the same trap the other way round.

**A symmetric phantom cannot calibrate R_b.** It has R_b = 2 by
construction, the theoretical floor, while what is worth measuring in a real
tree is the excess above it. `--side-branches` imposes that excess: an
element of order n then carries side branches of order n − `--side-drop`
along its length, which is the monopodial pattern of a lung. Strahler is
unharmed — the two children of such a joint have different orders, so the
element stays order n end to end — and the counts follow
`N_n = 2 N_{n+1} + s N_{n+k}`, so R_b is the root of `x^k = 2x^(k-1) + s`:
2.414 for one side branch two orders down, 2.732 for two, 3.000 for one
order down. R_d and R_l are untouched, being functions of the order alone.

The truth quoted against a measurement is the *fitted* R_b, not that
asymptote: a finite tree counts 1, 2, 4 elements at the top whatever its
rule, and over seven orders the fit lands near 2.35 where the asymptote is
2.414. Both sides are fitted by the same estimator over the same orders, so
what is left between them is the chain. `phantom.py` prints both.

**One phantom cannot settle whether a bias is real.** The interval
`centerline.py` reports is the confidence interval of a regression through
five or six orders. It is 4 to 5 % wide however many times the case is run —
wider than the bias being measured — so a single case can only ever say the
bias is not distinguishable from zero. `calibrate.py --repeats` runs each
case again under a fresh draw and reports the interval on the *mean bias*,
which narrows as 1/√n and is the right uncertainty for a systematic effect.
Run it with `--jitter` as well, or the repeats vary the noise, hold the tree
fixed, and come out too narrow.

With `--jitter` the comparison has to be **paired**: the imposed tree is
redrawn every repeat and its own fitted ratio moves with it — unbiased over
ten draws, but about a percent off on any one, since its per-order means rest
on 16, 8, 4, 2, 1 segments. Dividing a mean over ten chain runs by the truth
of one of the ten reports that draw's offset as a bias of the chain, and
reports the *same* offset at every point of an arm, because every case is
jittered from the same seed sequence — which dresses a single unlucky tree up
as a consistent trend. Each repeat is therefore compared with the tree it was
run on, which also takes the tree-to-tree variance out of the interval.

`--pin-smallest` holds the bottom of the tree at a fixed number of voxels of
the **coarse** axis, so every case in a sweep offers the chain the same
measurable span. The trunk is pinned with it, but against the R_l the arm
holds *fixed*, never against the R_l being swept: both ends of a geometric
series cannot be held while its ratio varies, and pinning the small end walks
the trunk from L/D 0.8 at R_l 1.15 to 4.6 at 1.80 — the stubby end is a disc
its daughters weld into, and the fit on it measures the weld.

```bash
python calibrate.py --spacing 1.25,0.799,1.25 --side-branches 0 1 2 3 \
  --repeats 5 --jitter 0.1 --measured-rb 2.31 --out calibration.csv
```

**Nothing is kept out of the curve for being imprecise.** Both criteria that
were once used for it are off by default.

R² is the clearer error: it is the share of the variance a straight line
explains, so an arm that sweeps its ratio towards 1 sweeps the variance to be
explained towards zero with it, and a flat truth measured perfectly scores
near zero. On the R_l arm the case imposed at 1.05 scores R² 0.57 and carries
the tightest interval of the arm, while cases scoring 0.97 carry per-fit
intervals of ±40 % and worse — so the R² floor removed the low end of that
arm systematically, the end a measured R_l near 1.2 has to be bracketed by,
and kept the loose end.

The width of the per-case fit interval is scale-free where R² is not, and it
fails a second objection. It describes how uncertain *one* realization's
regression was, which is exactly what `--repeats` averages away; what remains
after the repeats is the paired interval, the precision of the quantity being
estimated. On the R_b arm at 30 repeats the case at 2.865 carries a paired
interval 5.2 points wide against 5.2 for its neighbour at 3.080 and 5.6 for
the one at 2.624, while their per-fit spreads read 38 %, 29 % and 24 %: a
threshold on the spread separates cases the repeats have made
indistinguishable.

And excluding an imprecise point does not make the reading more precise. The
band already carries every point's uncertainty through the envelope; dropping
one only widens the gap the interpolation has to cross. Dropping 2.865 took
the bracket around a measured R_b of 2.782 from 0.19 to 0.33 in recovered
units and widened the band on the answer by a fifth.

Wrongness is caught where it shows instead: `keep_consistent` for a fit
resting on a different number of orders, and the inversion itself for a curve
or an envelope that stops increasing. `--max-fit-spread` remains for a sweep
run *without* repeats, where the per-case interval is the only precision
there is; `--min-r2` for an arm whose slope is nowhere near zero.

Each arm ends with what it supports, in the register it supports it in: a
per-point correction, or a sign and an order of magnitude with the drift
across the arm as the evidence, or nothing. The inversion answers with a
band rather than a value — the inverse of a curve known to a few percent is
an interval of imposed ratios, and quoting its centre turns that interval
into a point value the phantom cannot support.

Each sweep is written to `--out` and can be read again with `--from-csv`,
which runs no phantom:

```bash
python calibrate.py --from-csv calibration.csv --measured-rd 1.432
```

Use it whenever the measured ratio is revised — after a fit range is
corrected, say. Re-sweeping to read a new value would move the curve as well
as the value, leaving nothing to attribute the difference to.

`calibration.csv` in this repository is one such sweep, on a
1.188/0.799/1.313 mm acquisition over five orders, the smallest pinned at 3.5
coarse voxels, jitter 0.1, `--fit-min-voxels` counted on the coarse axis:

| ratio | what the chain does to it |
|---|---|
| R_d | unbiased: +1.0 / −0.2 / +0.2 / +0.5 / +0.1 % over imposed 1.30→1.85, one point of five excluding zero |
| R_l | +0.8 to +9.6 %, two points of five excluding zero |
| R_b | compressed, and increasingly so with the asymmetry: −0.1 % at R_b 2.00, −2.2 % at 2.62, −5.6 % at 2.87, −6.1 % at 3.08, −12.2 % at 3.46 |

R_b is the one that needs correcting, which is the reverse of what a
symmetric phantom would have suggested — it cannot see the effect at all,
since it sits at the point where the bias is zero. The mechanism is visible
in the arm: side branches are the thinnest vessels in the tree, they are lost
first, and what goes with them is exactly the excess of R_b over 2.

Both isotropic controls disagree with the anisotropic grid on R_d, giving
−2.3 to −4.7 %. At equal sampling *on the coarse axis*, the study's grid has
a second axis 1.64× finer, and that is what keeps the thin end measurable. An
isotropic phantom would have prescribed a 2 to 5 % correction that does not
apply.

### Running a cohort

The floor is mechanical but it lands on a different order at every spacing,
so a finer acquisition keeps more orders than a coarser one. That is the
shape of the data rather than a defect of the method, and the one thing it
requires is that it be declared: **the rule is pre-specified, the range is
not.** `--prespecified` says the rule was fixed before the per-order table
was visible; it does not say the range was the same for everyone, because it
cannot be.

So the acquisition travels with the number. `--ratios-csv` carries the
spacing, the anisotropy, the floor applied and the orders it left, and
`--subject` names the row:

```bash
python centerline.py --input sub-01.nii.gz --subject sub-01 \
  --ordering strahler_dd --prespecified --ratios-csv results/sub-01_ratios.csv
python cohort.py 'results/*_ratios.csv' --out cohort.csv
```

`cohort.py` assembles those files and runs the two checks that decide whether
the subjects are comparable at all.

**Is it one protocol?** The anisotropy should be near-constant across a
cohort acquired the same way. A subject that departs from the cohort median
was acquired differently, or has a header that misreports it; either way its
floor is elsewhere, its fit rests on different orders of the same tree, and
its ratios are not comparable to the rest.

**Does every subject support a fit?** Three points still admit a regression,
if barely. Two make the slope an arithmetic identity. A subject under the bar
is dropped with its reason recorded, not fitted anyway.

Both matter more than they look. Run on five phantoms built with the *same*
imposed R_d of 1.500, the three sharing one protocol return 1.481, 1.495 and
1.499 — within 1.2 % of each other — while a fourth on a near-isotropic grid
keeps one order more and returns 1.453. Three percent apart, on identical
trees. Without the columns a reader attributes that to anatomy.

### Auditing the radius directly

Every ratio rests on the distance transform, and the distance transform is
the one measurement in the chain with no redundancy: a length is averaged
over hundreds of points, a radius is one inscribed sphere in one place.
`radius_audit.py` compares it to the imposed radius point by point and
stratifies the error by the angle the vessel makes with z.

```bash
python radius_audit.py --spacing 1.188,0.799,1.313 --orders 7 \
  --side-branches 1 --control --points-csv audit.csv
```

The reason to stratify by angle is that an anisotropic acquisition degrades
a vessel by its orientation, not by its size: a radius is measured in the
plane across the vessel, so one running along the fine axis has its
cross-section sampled by the two coarse ones and is the worst case. The
`cross-plane voxel` column is that sampling. Angle and calibre are
confounded in any tree grown from a trunk, so the two-way table against the
imposed order is the one to read, and the slope is fitted within order.
Statistics are over segments, not points: hundreds of points along one
vessel are one draw of what the grid did to that vessel.

Two cuts are applied and both matter. Points within one local radius of a
junction are dropped, where the inscribed sphere is the bifurcation cavity;
and orders under three coarse voxels are excluded from every average,
because there the transform returns its own floor — the same number for
every such vessel — which pools in as the largest term in the result while
being an artefact of where the tree was cut.

Give all three axes even when two of them are close. The partial-volume
ramp is now the voxel width along the wall normal, so two coarse axes that
differ — 1.188 and 1.313 — do not resolve a vessel the same way, and the fit
floor is set by the coarsest of the three alone.

`--control` re-runs the same tree isotropic at the finest axis, held to the
same orders, so the comparison is the anisotropy and not a different set of
surviving vessels. On a 1.188/0.799/1.313 grid over seven orders, the radius
bias on the resolved orders is +1.6 % [+0.0, +3.2] against +3.8 % [+2.1,
+5.4] for the isotropic control, and the within-order orientation slope
covers zero (−16.5 % per mm of cross-plane voxel, [−33.5, +0.6]; 3.7 % of
the radius between the best- and worst-oriented vessel of an order). At this
spacing the anisotropy costs *orders* — the two finest are lost — not
accuracy within an order.

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
