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

## Not included yet

The two-stage training scheme (pretraining + hard-case fine-tuning) has been
set aside in favor of a single training stage; it can be reintroduced later
if needed.
