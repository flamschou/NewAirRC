# AirRC

Deep learning pipeline for 3D vascular tree segmentation, built on a
[MONAI](https://monai.io/) `DynUNet` (nnU-Net-style 3D U-Net with deep
supervision) trained with PyTorch Lightning.

## Repository layout

```
pipeline/     training and inference
analysis/     everything downstream of a segmentation
figures/      the plots those analyses are read from
manifests/    the image/label manifests
slurm/        cluster job scripts
reference/    the calibration sweep, replayable with `calibrate --from-csv`
docs/         the centerline method, in detail
```

Everything is run as a module from the repository root, so no installation
step is needed:

```bash
python -m pipeline.train
python -m analysis.centerline --input mask.nii.gz
```

## Pipeline overview

```
manifest.json --> manifest.py --> transforms.py --> dataset.py --> train.py --> model.py (DynUNet)
```

The modules below are in `pipeline/`.

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
MANIFEST_PATH=/path/to/manifest.json python -m pipeline.train
```

Set `DEBUG=1` for a fast, tiny run (few patches, one epoch) to sanity-check
the pipeline before a full training run:

```bash
DEBUG=1 MANIFEST_PATH=manifests/manifest.example.json python -m pipeline.train
```

`manifests/manifest.example.json` points at `example_data/` (not tracked in git) and
is only meant for smoke-testing the pipeline mechanics — it reuses a single
volume for both train and val, so it is not a real training run.

## Running on the cluster (SLURM)

```bash
sbatch slurm/train.slurm
```

Submit **from the repository root**, as above: the job runs from
`$SLURM_SUBMIT_DIR`, which is where `sbatch` was invoked, and
`python -m pipeline.train` only resolves from there. The script checks
this and exits with a message rather than failing later on an import.

This queues the job and prints its `<jobid>`. `slurm/train.slurm` writes stdout to
`slurm-<jobid>.out` and stderr to `slurm-<jobid>.err` in the submission
directory, and sets `DATASET_ROOT` (checkpoints/logs/cache) and
`MANIFEST_PATH` for `train.py`.

`slurm/finetune_cldice.slurm` is the same job with the continuity term switched on
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
sbatch slurm/finetune_cldice.slurm
```

which is `slurm/train.slurm` plus, in the environment:

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

Keep one from-scratch run (`CLDICE_WEIGHT=0.5` in `slurm/train.slurm`, no
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
python -m pipeline.inference \
  --checkpoint "$DATASET_ROOT/checkpoints/vessel_segmentation/run/last.ckpt" \
  --input path/to/image.nii.gz
```

Batch over a directory (one prediction per `*.nii.gz` input):

```bash
python -m pipeline.inference \
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
python -m analysis.compare_predictions \
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
python -m analysis.truncate --input vascular_case001.nii.gz --label 3

# every foreground class of config.LABEL_CLASS_MAP, each on its own tree,
# into one file (raw 3 = artery, 4 = vein, airway classes 1-2 dropped)
python -m analysis.truncate --input vascular_case001.nii.gz --classes

# the labels of a manifest split
python -m analysis.truncate --manifest manifests/manifest_ct.json --split val --output-dir cut/

# the predictions that go with them, cut by the same rule
python -m analysis.truncate --input-dir /data/flamant/data/ct/lidc_idri \
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
python -m analysis.compute_dice --manifest manifests/manifest_ct.json --split val --csv dice.csv

# or straight from the checkpoint, predicting whatever is missing on the way
python -m analysis.compute_dice --manifest manifests/manifest_ct.json --split val \
                       --checkpoint "$DATASET_ROOT/checkpoints/.../last.ckpt" \
                       --csv dice.csv

python -m analysis.compute_dice --manifest manifests/manifest_ct.json --classes artery vein \
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
python -m analysis.compute_dice --manifest manifests/manifest_vibe.json --split val \
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

The large-vessel pass truncates **both sides** with the same calibre floor,
then peels the last layer of tips off the **reference alone**
(`--peel-terminals 1 0`), and writes the truncated masks out (`truncate.py`
does the cutting, sidecar included),
so the number can be traced back to the volumes it was read on. Cutting only
the reference would count every peripheral vessel of the prediction as a
false positive, and the Dice would measure the truncation rather than the
model. Cutting the prediction with its own tree does mean a prediction whose
trunk is broken gets a different truncation from the reference's -- that is
the failure showing up, not a defect of the metric, but it is why each row
keeps `n_segments_kept_reference` next to `n_segments_kept_prediction`: a
case where those two disagree wildly is a case to look at before quoting its
Dice.

The terminal peel is the default in `compute_dice.py` and `sweep_rescue.py`
only, and `truncate.py` still peels nothing. The reference scored here is a
**hand-drawn annotation**, and its last layer of tips is where a hand and a
model disagree for reasons that are not the model's -- an annotator stops a
vessel where the contrast goes rather than where the vessel does, and the
calibre that ended a terminal run was measured where partial volume weighs
most.

The peel is **asymmetric** because the two sides are not the same kind of
object: the model draws a vessel *thinner* than the hand that annotated it,
so the same floor already stops the prediction's tree earlier, and peeling
both sides would take that one asymmetry out twice -- leaving the prediction
shorter than the reference it is compared with, a truncation difference
scored as a model error. `--peel-terminals` therefore takes one value for
both sides or two, reference then prediction: `1 0` is the default, `1 1`
peels symmetrically, `0` peels nothing.

That is a heuristic about these two objects, not a property of the metric,
and the tools print what it is worth: `centerline_kept_mm_reference` against
`centerline_kept_mm_prediction` (`centerline ref/pred` in `sweep_rescue.py`'s
table) is the length each side kept, in world millimetres, and near-equal is
two trees of the same extent. The peel drops real vessel whichever way it is
set, so both values are printed with the summary and written into every CSV
row.

`sweep_rescue.py --peels` settles it on the cohort rather than on the story,
by putting the peel in the same grid as the rescue knobs:

```bash
python -m analysis.sweep_rescue --manifest manifests/manifest_ct.json --split val     --peels 0 "1 0" "1 1" --margins 0 2 --csv sweep_peel.csv
```

Each value is one or two numbers in **one** shell argument. Read the result
on the `length gap` column -- reference minus prediction, signed, in mm of
centerline -- and take the peel nearest 0, **not** the best `dice_large`:
peeling always raises the Dice, since what is left is a smaller and easier
region, so reading the table that way picks the heaviest peel every time.

The peel shares the grid rather than being three separate runs because it is
not independent of the margin: each side's plain cut is what the *other*
side's rescue is judged against, so peeling the reference alone leaves the
prediction more to be rescued by, and the knee of the margin curve can move
with it. Compare margins within one peel.

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

The centerline chain — skeletonization, branch numbering, branching ratios,
the quality metrics and orphan-component report, and the phantom calibration
the whole thing rests on — is documented on its own in
**[docs/centerline.md](docs/centerline.md)**. It is 800-odd lines of method,
which is why it does not live here.

```bash
python -m analysis.centerline --input artery.nii.gz
```
