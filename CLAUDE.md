# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running things

Everything runs as a module **from the repository root**. There is no install step, no
`pyproject.toml`, no `PYTHONPATH` to set — `python -m` puts the root on `sys.path` and the three
packages resolve from there.

```bash
python -m pipeline.train
python -m analysis.centerline --input mask.nii.gz
python -m figures.plot_sweep --from-csv results/sweep.csv
```

Smoke-test the whole training pipeline in about ten seconds (tiny patches, one epoch, one volume
used for both train and val):

```bash
python -m pipeline.train --debug --manifest manifests/manifest.example.json
```

The debug mode is handled at the bottom of `pipeline/config.py`, which shrinks the patch to 64³,
drops the batch to 1, sets `NUM_WORKERS=0` and disables the clDice warm-up so the term is actually
exercised in the single epoch. Because `config.py` acts on it *at import time*, `--debug` cannot
be applied after the fact — `main()` sets `DEBUG=1` and re-execs itself. `DEBUG=1 python -m
pipeline.train` still works and skips the re-exec.

On the cluster, submit **from the repository root** — `slurm/train.slurm` runs from
`$SLURM_SUBMIT_DIR`, which is where `sbatch` was invoked, not where the script lives. It checks
this and exits with a message rather than failing later on an import.

```bash
sbatch slurm/train.slurm
```

## Checks

There are **no tests and no linter config**. The only tool installed in `.venv` is pyflakes, and
the repository is currently clean under it — keep it that way:

```bash
.venv/bin/python -m pyflakes pipeline analysis figures
```

Verification is done by running the CLIs on `example_data/` and comparing output. The two that
exercise most of the analysis chain:

```bash
python -m analysis.truncate --input example_data/vascular_gen_LIDC-IDRI-0015.nii.gz \
    --output-dir /tmp/out --min-diameter 6      # expect 616/5761 segments, 47.8% of the volume
python -m analysis.centerline --input example_data/vascular_gen_LIDC-IDRI-0015.nii.gz \
    --output /tmp/cl.nii.gz                     # expect 18079 centerline voxels
```

Those numbers are a regression check: a refactor that changes them changed behaviour.

## Layout

```
pipeline/     training and inference (MONAI DynUNet + PyTorch Lightning)
analysis/     everything downstream of a segmentation
figures/      plots read off the analysis CSVs
manifests/  slurm/  reference/  docs/
```

`data/`, `example_data/` and `results/` are gitignored — run outputs, smoke-test volumes and
figures respectively.

## Architecture

### The analysis chain is layered, not parallel

The four big modules in `analysis/` form a strict stack, each importing the one below:

```
sweep_rescue  ->  compute_dice  ->  truncate  ->  centerline
```

`centerline` turns a mask into a branch graph. `truncate` cuts that graph back to its large
vessels. `compute_dice` scores a reference against a prediction, both cut the same way.
`sweep_rescue` runs `compute_dice` over a grid of cut parameters. **Nothing is duplicated between
them** — there are no copy-pasted helpers, and a change belongs at the lowest layer that can
carry it. `phantom` (synthetic trees of known geometry) sits beside the stack and feeds
`calibrate` and `radius_audit`, which validate what `centerline` measures.

### Shared CLI arguments live in truncate.py

`truncate.add_cut_arguments()` (7 flags: `--min-diameter`, `--cut-step`, `--max-generation`,
`--ordering`, `--min-strahler`, `--peel-terminals`, `--sleeve`) and
`truncate.add_skeleton_arguments()` (10 flags) are called by `truncate`, `compute_dice` and
`sweep_rescue` so the three cut identically. **A new cut or skeleton flag goes in those helpers,
not in an individual parser** — otherwise the tools stop agreeing about what "the same cut" means.

### The cut sidecar is a cache key — treat its schema as load-bearing

`truncate.cut_settings()` writes a JSON sidecar next to every cut it produces, recording the
settings that produced it. `compute_dice.reusable()` decides whether an existing cut can be reused
by comparing `stored == wanted` — **exact dict equality**.

Consequence: adding or removing a key in `cut_settings` makes every sidecar already on disk
compare unequal, and a cohort's worth of skeletonizations gets recomputed. This is why some keys
(`peel_terminals`) are written conditionally, with a comment saying so — an absent key and a null
one describe the same cut. Follow that pattern for any new optional setting.

### config.py is the single source of truth, and it is env-driven

`pipeline/config.py` holds every path, patch geometry, class name and hyperparameter. Nothing else
hardcodes them. Settings resolve in three layers:

```
command-line flag  >  environment variable  >  config.py default
```

`train.py`'s parser gets this for free by declaring `default=cfg.X` on each flag, since `cfg.X`
has already resolved the environment variable — the value is never written down twice.
`apply_overrides()` then writes the parsed values back onto the config module, which is what lets
`dataset`, `model` and `config_loss` keep taking the module and reading attributes off it.

The split is deliberate: **a flag describes the run** (`--learning-rate`, `--cldice-weight`,
`--experiment-name`) because a `.env` is gitignored and invisible, so a learning rate living there
makes two people running the same command get different results with nothing in the log to say
why. **An environment variable describes the machine** (`DATASET_ROOT`, `NUM_WORKERS`) — see
`.env.example`. The SLURM scripts follow it: machine settings exported, run settings on the
command line, where `slurm-<jobid>.out` records them.

Two subtleties that are not visible from the file alone:

- **`MANIFEST_PATH` is anchored on `REPO_ROOT`**, not on `DATASET_ROOT`. `DATASET_ROOT` is where a
  run *writes* (cache, checkpoints, logs); the manifest is an input that lives with the source.
- **The dataset cache is keyed on the preprocessing, not on the run.** `PersistentDataset`'s own
  cache key is item identity only — the image/label paths — never the transform pipeline, so
  changing `TARGET_SPACING` would silently reuse tensors resampled at the old one. `config.py`
  supplies the missing half: `PREPROCESSING_FINGERPRINT` hashes the three values `_base_transforms`
  reads (`LABEL_CLASS_MAP`, `TARGET_SPACING`, `NORMALIZE_INTENSITY`) together with the AST of
  `transforms.py`, docstrings stripped — so rewording a comment does not invalidate a cohort's
  preprocessing while an edit to `mode=` does. `CACHE_DIR` is that hash. Nothing has to be renamed
  to get a fresh cache, and two runs that preprocess identically share one without being told to.
  `dataset._write_cache_settings` drops a `preprocessing.json` beside the cache so the hash is
  legible, mirroring the cut sidecar in the analysis half.

### Fine-tuning is not a resume

`FINETUNE_FROM` loads **weights only** via `model.load_checkpoint_weights()`. A resume goes through
`Trainer.fit(ckpt_path=...)` and restores the optimizer, the epoch counter and with it the PolyLR
schedule — which at the end of a run has already decayed to ~0, so a "fine-tuning" done that way
would not move. Auto-resuming this run's own `last.ckpt` still takes precedence, so a preempted
fine-tuning picks up where it stopped.

Note that `ModelCheckpoint` is configured with `monitor=None`: checkpoints are periodic snapshots
of the *latest* state, not the best. `val_dice_metric` is informational.

### Raw labels are remapped in memory

Label files carry more classes than are trained on (the generator labels airways as 1-2 alongside
vessels 3-4). `config.LABEL_CLASS_MAP` maps raw values to training indices and everything else
collapses to background. This happens once, in `transforms.py` — label files are never
pre-processed on disk.

## CLI conventions

The flags were uniformized across all modules; new ones should follow:

| | flag |
| --- | --- |
| read a CSV | `--from-csv` |
| write a CSV | `--csv` |
| write a NIfTI | `--output` |
| write an image | `--out` |
| one input file / a directory of them | `--input` / `--input-dir` |
| negate a default behaviour | `--no-X` (never `--skip-X`) |

`--label` always means a raw voxel value. `--classes NAME...` always means a list of class names.
A spacing in the phantom family (`phantom`, `calibrate`, `radius_audit`) is parsed by the shared
`phantom.parse_spacing()` and accepts `1.25 0.799 1.25` or `1.25,0.799,1.25` interchangeably;
`--spacing` in the skeleton chain is a different quantity (the isotropic size to resample to) and
is a scalar float.

Every flag in the repository has help text. Keep it that way — the help is where the method is
documented, and several flags carry a paragraph explaining why the default is what it is.

## Documentation

Five documents, each with an exclusive job — they link to each other rather than restating:

| | |
| --- | --- |
| `README.md` | getting from zero to a first result. Deliberately lean |
| `pipeline/README.md` | running trainings in anger: clDice, `/dev/shm`, fine-tuning, the cache |
| `analysis/README.md` | the ten tools, which to use when, the cut and the scoring in detail |
| `docs/centerline.md` | the centerline method, 800-odd lines: branch numbering, ratios, quality metrics, orphan components, phantom calibration |
| `CLAUDE.md` | this file |

When adding documentation, put it in the one whose job it is and link — do not restate. Three
documents describing the same architecture is how they start to disagree.
Prose in this repository argues for its choices rather than restating the code — match that
register when editing it.
