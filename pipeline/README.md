# pipeline/ — training and inference

This is the second half of the [main README](../README.md), for someone already
running trainings rather than starting one. Installation, the manifest format and
the essential commands are there; what follows is what the defaults are hiding.

```
manifest.json -> manifest.py -> transforms.py -> dataset.py -> train.py -> model.py
```

| module | |
| --- | --- |
| `config.py` | single source of truth for paths, geometry, classes and hyperparameters |
| `manifest.py` | reads the JSON manifest and splits it into train/val |
| `transforms.py` | MONAI preprocessing and augmentation; modality-agnostic, no Hounsfield logic |
| `dataset.py` | the cached `PersistentDataset` and the `DataLoader`s |
| `model.py` | `DynUNet` construction, and weight-only checkpoint loading |
| `config_loss.py` | deep-supervision-weighted Dice+CE, plus the optional clDice term |
| `train.py` | training entrypoint |
| `inference.py` | sliding-window prediction on full volumes |

## Where a setting lives

Three layers, each overriding the one below:

```
command-line flag  >  environment variable  >  config.py default
```

The split is deliberate. **A flag describes the run** — the learning rate, the
number of epochs, the continuity term. A `.env` is gitignored and invisible, so a
learning rate living there makes two people running the same command get
different results with nothing in the log to say why; a flag lands in the shell
history and in `slurm-<jobid>.out`. **An environment variable describes the
machine** — where the data is, how many workers the node can feed. Nobody wants
to retype those, and they mean nothing scientifically. See `.env.example`.

Anything not exposed as either — the patch size, the class list, the patch
budgets — is edited in `config.py`, which nothing else in the codebase
hardcodes.

## Dataloader workers and `/dev/shm`

A training batch is not `BATCH_SIZE` volumes: `RandCropByPosNegLabeld` returns
`num_samples` patches per item and `list_data_collate` flattens them, so the
loader hands out `BATCH_SIZE x num_samples` = **24 patches of 128³**, image and
label, ~0.4 GiB per batch. Every prefetched batch sits in `/dev/shm` until the
main process consumes it, and `train.py` selects the `file_system` sharing
strategy, under which those segments are files.

At torch's default `prefetch_factor=2` with 8 workers on each of the two loaders,
that reserves ~10 GiB of `/dev/shm` at steady state, and the 28-step epochs
respawn all 16 workers every ~35 seconds, each generation leaving segments to be
reclaimed. That combination exhausts `/dev/shm` mid-run on a node whose shared
memory is sized from the job's memory cgroup. It surfaces as a `RuntimeError:
unable to open shared memory object` from the pin-memory thread — not as
anything resembling a data problem, which is why it is worth knowing about
before it happens.

Three settings control it, all environment variables (see `.env.example`):

| | default | |
| --- | --- | --- |
| `NUM_WORKERS` | `8` | training loader workers |
| `VAL_NUM_WORKERS` | `NUM_WORKERS // 4` | the validation loader runs 6 batches an epoch against the training loader's 28; it does not need the same fleet |
| `PREFETCH_FACTOR` | `1` | batches held ready per worker. The main `/dev/shm` knob |

plus `persistent_workers=True` on both loaders, which stops the respawn churn.
Together these bring steady-state shared memory to ~3.6 GiB. If a node still runs
out, `PREFETCH_FACTOR` and `VAL_NUM_WORKERS` are the two to turn down first;
check what the node actually offers with `df -h /dev/shm` on it.

Persistent workers keep all `NUM_WORKERS + VAL_NUM_WORKERS` processes alive for
the whole run rather than only during their own loader's epoch, so host RAM use
becomes constant instead of intermittent — each worker carries a full
torch/monai import.

## The dataset cache

`PersistentDataset` caches the deterministic half of the pipeline — load,
relabel, reorient, resample, normalize — and re-applies only the random cropping
and augmentation on each access. It is what makes an epoch 35 seconds instead of
minutes, and it is worth understanding because MONAI's own cache key is
incomplete.

That key is **item identity alone**: the image and label paths. The transform
pipeline does not enter it. Change `TARGET_SPACING` and MONAI will happily
reload tensors resampled at the old one, without a word.

`config.py` supplies the missing half. `PREPROCESSING_FINGERPRINT` hashes the
three values the cached transforms read — `LABEL_CLASS_MAP`, `TARGET_SPACING`,
`NORMALIZE_INTENSITY` — together with the AST of `transforms.py`, docstrings
stripped. `CACHE_DIR` is that hash:

```
$DATASET_ROOT/cache/c6226b28fe9d/
```

So nothing has to be renamed to get a fresh cache: change the preprocessing and
you get one, change nothing and you reuse it. Two runs that preprocess
identically share a cache without being told to, which is what a fine-tuning
wants. Hashing the AST rather than the file text means rewording a comment does
not invalidate a cohort's worth of preprocessing, while an edit to `mode=` or to
the label filter does.

The directory is named by a hash, so a `preprocessing.json` is written beside it
recording what produced it. Nothing reads that file back — the hash *is* the key
— it is there so that whoever finds the directory knows what it holds.

## Adding the continuity term (clDice)

Dice and cross-entropy are voxel-wise. A two-voxel break in a vessel costs them
almost nothing, and yet it splits the tree in two: downstream, every branch past
the break is reassigned to a different component, which shifts its Strahler order
and corrupts the branching ratios `analysis/centerline.py` measures. Whole-tree
Dice is simply not the metric that sees this.

`config_loss.py` implements **clDice** (Shit et al., CVPR 2021) as an optional
extra term:

```
loss = DiceCE + cldice_weight * clDice
```

clDice compares the *skeletons* of the masks instead of the masks. It is the
harmonic mean of a topology precision (how much of the predicted skeleton falls
inside the true mask) and a topology sensitivity (how much of the true skeleton
falls inside the predicted mask); a gap removes a whole run of true skeleton from
the second term, so the term sees the break that Dice missed. The
skeletonization is *soft* — iterated min/max pooling — so the whole thing stays
differentiable and needs no post-processing.

| flag | default | |
| --- | --- | --- |
| `--cldice-weight` | `0.0` | the lambda above. **0 disables the term** and reproduces the original loss exactly. `0.5` is the paper's value for tubular structures |
| `--cldice-iterations` | `6` | soft-skeletonization iterations. Must be >= the largest vessel radius in voxels, or thick vessels never thin down to a curve: ~10 mm across at 1 mm spacing is a radius of 5, plus one for margin |
| `--cldice-warmup-epochs` | `20` | epochs over which the weight ramps linearly from 0. The skeleton of a not-yet-vessel-shaped prediction is noise, and clDice on noise pulls towards thin fragmented masks; the ramp keeps the term quiet until there is something for it to fix |
| `--cldice-max-patches` | `8` | how many patches of the batch the term scores (0 = all). See below |

Two deliberate restrictions in the implementation: the term is applied to the
**full-resolution deep-supervision level only** (at 1/2 and 1/4 resolution the
thinnest vessels are sub-voxel, so their "skeleton" is a downsampling artifact),
and it is computed in **fp32** even under `precision="16-mixed"` (the pooling
chain ends in differences of nearly-equal numbers).

### Why this term is memory-hungry

Worth knowing before turning any of the knobs up, because it OOMed a 93 GiB H100
on the first try. The thinning is a chain of ~20 pooling and elementwise ops per
iteration, and autograd keeps a full-resolution copy of nearly every
intermediate. The batch that reaches the loss is not `BATCH_SIZE` but **24
patches of 128³**, as above. A naive implementation keeps **79 GiB** of graph for
that, on top of the network's own activations.

Three things bring it down to ~2 GiB:

| | graph size |
| --- | --- |
| naive implementation | 79.3 GiB |
| + erosion reused across iterations (the textbook loop erodes twice per iteration, but the second erosion of iteration *j* is the first of *j+1*) | 51.2 GiB |
| + gradient checkpointing per thinning iteration | 6.6 GiB |
| + `--cldice-max-patches 8` | 2.2 GiB |

Only the patch cap changes the loss, and only *statistically*: clDice is a
batch-level ratio, so scoring 8 of the 24 patches is a noisier estimate of the
same quantity, not a different quantity. The subset is drawn at random while
training and is the first 8 patches in eval, so `val_loss` stays reproducible.
The other two are exact — the skeleton is bit-identical to the textbook
formulation and the gradients match to 0.

With all three, the term costs about **+4%** per training step.

### Fine-tuning rather than retraining

The term is meant to be added to an already-trained model, not trained from
scratch with: clDice only starts saying something useful once the segmentation is
roughly right, and a fine-tuning run is short enough to sweep the weight and the
iteration count over.

```bash
sbatch slurm/finetune_cldice.slurm
```

which is `slurm/train.slurm` with the run described on the command line:

```bash
python -m pipeline.train \
    --experiment-name "${BASE_EXPERIMENT}_cldice_ft" \
    --finetune-from ".../checkpoints/$BASE_EXPERIMENT/run/last.ckpt" \
    --learning-rate 1e-4 \
    --max-epochs 150 \
    --cldice-weight 0.5 \
    --cldice-max-patches 8
```

Three things that would otherwise be easy to get wrong, and that the code
handles:

- **Weights only, not a resume.** `--finetune-from` is deliberately not passed as
  `Trainer.fit(ckpt_path=...)`, which would also restore the source run's
  optimizer state and epoch counter — and with the epoch counter its PolyLR
  schedule, already decayed to ~0 at the end of that run. The fine-tuning would
  train at `lr = 0` and nothing would move. Rebuilding the schedule from
  `--learning-rate` and `--max-epochs` restarts the decay at 1e-4 instead.
- **Auto-resume still wins.** If the fine-tuning run's own `last.ckpt` exists, it
  is resumed normally, so a preempted job picks up where it stopped rather than
  restarting from the base weights.
- **The dataset cache is shared, for free.** A fine-tuning changes no
  preprocessing, so its fingerprint is the base run's and the two use one cache.
  Nothing has to be said for that to happen.

Keep one from-scratch run (`--cldice-weight 0.5`, no `--finetune-from`) for the
final comparison if the numbers are going in a paper: "DiceCE vs DiceCE+clDice"
is only an honest comparison at equal budget from the same initialization.

### Judging whether it worked

`val_dice_metric` will barely move — it is not the metric this term targets — and
`val_loss` is not comparable across the warm-up epochs either, though the
`cldice_weight` scalar is logged to TensorBoard so you can see where the ramp
was. Judge it on full volumes, via `pipeline.inference`, with the tools in
[`analysis/`](../analysis/README.md):

- `analysis.connectivity` — number of connected components and the fraction of
  the mask held by the largest one;
- `analysis.compare_predictions` — component counts and surface metrics of the
  fine-tuned predictions against the baseline's, no ground truth needed;
- `analysis.centerline --orphans-csv --orders-csv` — the one that actually
  matters here: how much of the tree falls outside the main component, and how
  many Strahler orders survive.

## Reproducibility

`train.py` calls `seed_everything(cfg.SEED, workers=True)`. The `workers=True`
matters as much as the seed: the patch sampling and every augmentation run inside
the dataloader workers, and the validation loader crops randomly too. Without it,
`val_dice_metric` is not comparable between two runs of the same command, let
alone between a baseline and its fine-tuning.

Note that `ModelCheckpoint` runs with `monitor=None`: checkpoints are periodic
snapshots of the *latest* state every 10 epochs, plus `last.ckpt`. The best
epoch is not selected for you.
