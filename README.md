# Vascular segmentation and evaluation

The segmentation part of this code is based on [b-niu/AirRC](https://github.com/b-niu/AirRC).
It fills the holes left in it and adds the specifics of this training.

It also adds a whole evaluation part, based on morphometric measurements computed
from an extracted centerline.

Everything works on NIfTI (`.nii.gz`) volumes.

Two halves, which can be used independently:

| | | |
| --- | --- | --- |
| [`pipeline/`](pipeline/README.md) | segmentation | train a model, run it on new volumes |
| [`analysis/`](analysis/README.md) | evaluation | cut a tree back to its large vessels, score it, measure its branching ratios |

Each has its own README for what this one leaves out — the continuity term and
the dataloader's shared memory on one side, the cut and the scoring in detail on
the other.

Commands are run **from the repository root**, as modules. There is no installation
step beyond the environment below — `python -m` finds the packages on its own.

## Initialisation

To run this code you first need a Python environment (3.11 or later):

```bash
python -m venv .venv
```

Then activate it and install the packages inside it:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Finally, tell the code where your data is and what your machine can take:

```bash
cp .env.example .env
$EDITOR .env
set -a; source .env; set +a
```

`.env` holds only what describes **your machine** — where the data is
(`MANIFEST_PATH`, `DATASET_ROOT`) and how many dataloader workers the node can
feed. It is gitignored, so your paths stay yours. Everything that describes a
**run** — the learning rate, the number of epochs, the continuity term — is a
command-line flag instead, so the command that produced a result also records it.
Each setting is commented in `.env.example`.

You can check the install with:

```bash
python -m pipeline.train --help
python -m analysis.centerline --help
```

## Segmentation

### Data

#### Load and process data from the AirRC dataset

<!-- TODO: where the volumes and their vascular annotations are downloaded from,
     and the commands to fetch them. -->

#### Data format

We advise you to first process your data to extract a crop around the lung for
training. To do so, see our processing part.

<!-- TODO: the preprocessing (lung crop + z-score) currently lives in a separate
     repository and will be merged into this one. -->

Images and labels are 3D NIfTI volumes. Labels carry raw integer values; which of
them are trained on, and as what class, is set by `LABEL_CLASS_MAP` in
`pipeline/config.py`:

```python
CLASS_NAMES     = ["background", "artery", "vein"]
LABEL_CLASS_MAP = {3: 1, 4: 2}    # raw 3 becomes artery, raw 4 becomes vein
```

Any raw value not listed there — the airway classes 1 and 2, for instance —
collapses to background. This happens in memory at load time, so your label files
never need to be rewritten on disk.

Once the data is formatted, split it into train/val. The manifests in this
repository use **83% train / 17% val** (163 and 34 cases).

The paths are declared explicitly in a JSON manifest rather than inferred from a
naming convention, because production filenames match no fixed pattern. See
`manifests/manifest.example.json`:

```json
[
  {"image": "data/images/case001.nii.gz", "label": "data/labels/case001.nii.gz", "split": "train"},
  {"image": "data/images/case002.nii.gz", "label": "data/labels/case002.nii.gz", "split": "val"}
]
```

Point `MANIFEST_PATH` at your file in `.env`, or pass `--manifest` per run.

### Training

#### Launch a training

```bash
python -m pipeline.train --manifest manifests/manifest_ct.json
```

That is the whole essential command: everything else has a default. The flags
worth knowing:

| flag | | default |
| --- | --- | --- |
| `--manifest` | image/label pairs to train on | `MANIFEST_PATH` |
| `--experiment-name` | names this run's checkpoints and logs | `vessel_segmentation_vein_artery_vibe_v1` |
| `--learning-rate` | initial LR of the PolyLR schedule | `1e-3` |
| `--max-epochs` | | `2000` |
| `--finetune-from` | checkpoint whose **weights only** seed this run | train from scratch |
| `--debug` | tiny run that exercises every code path in seconds | off |

Four more flags control the clDice continuity term, which is off by default;
`python -m pipeline.train --help` lists them with what each costs.

Checkpoints land in `$DATASET_ROOT/checkpoints/<experiment-name>/run/`, logs in
`$DATASET_ROOT/logs/<experiment-name>/run/`. Relaunching the same experiment name
resumes from its `last.ckpt` automatically.

On a SLURM cluster, submit **from the repository root**:

```bash
sbatch slurm/train.slurm
```

#### Monitoring

Metrics — training and validation loss, `val_dice_metric`, the learning rate —
are written as TensorBoard events:

```bash
tensorboard --logdir "$DATASET_ROOT/logs" --port 6006
```

On a cluster, the job's own output is the other half:

```bash
squeue -u $USER              # queued or running, and on which node
tail -f slurm-<jobid>.out    # epoch, step, loss, lr
tail -f slurm-<jobid>.err    # tracebacks and warnings
```

### Inference

```bash
python -m pipeline.inference \
    --checkpoint $DATASET_ROOT/checkpoints/<experiment-name>/run/last.ckpt \
    --input volume.nii.gz
```

| flag | |
| --- | --- |
| `--checkpoint` | required |
| `--input` / `--input-dir` | one volume, or a directory of them |
| `--output` / `--output-dir` | where predictions go; by default beside each input |
| `--output-suffix` | appended to the image stem |
| `--cpu` | force CPU even when a GPU is available |

Prediction runs by sliding window over the full volume and is written back on the
input's own grid and orientation.

Beyond this: the clDice continuity term, the `/dev/shm` settings a cluster run
needs, and how the dataset cache decides what to reuse are in
[`pipeline/README.md`](pipeline/README.md).

## Evaluation

Both tools below need a segmentation, whether it came from this pipeline or from
somewhere else.

### Compute the dice on a restricted area (large branch)

A Dice over the whole tree is dominated by the thin distal vessels, where two
segmentations of the same lung agree least. Cutting both sides back to their
large vessels first scores the part that is actually comparable.

```bash
python -m analysis.compute_dice \
    --manifest manifests/manifest_ct.json --split val \
    --min-diameter 5 --csv dice.csv
```

| flag | | default |
| --- | --- | --- |
| `--manifest` / `--split` | which cases to score | `MANIFEST_PATH`, `val` |
| `--min-diameter` | a branch thinner than this is cut off, with everything under it | `5` mm |
| `--classes` | classes to score, each on its own tree | `artery` |
| `--checkpoint` | predict the missing volumes on the way, instead of reading them | — |
| `--csv` | one row per case and class | — |

To cut a segmentation without scoring it, `analysis.truncate` does the cut alone:

```bash
python -m analysis.truncate --input segmentation.nii.gz --label 3 --min-diameter 5
```

### Compute the morphometrical ratios

`analysis.centerline` skeletonizes a mask, turns the skeleton into a branch graph,
and fits the branching ratios R_b, R_d and R_l with their confidence intervals:

```bash
python -m analysis.centerline --input artery.nii.gz \
    --ordering strahler_dd --fit-orders 1 6 \
    --subject sub-01 --ratios-csv results/sub-01_ratios.csv
```

`--fit-orders` is the range the ratios are fitted over and has to be fixed
**before** looking at the result; `--subject` writes an identifier into the CSV so
the per-subject files of a cohort concatenate. Assembling them, with the two
checks that decide whether the subjects are comparable at all:

```bash
python -m analysis.cohort 'results/*_ratios.csv' --csv cohort.csv
```

`analysis/` holds six more tools than the two above — sweeping the cut over a
cohort, comparing two models without a ground truth, auditing the chain against
synthetic trees. [`analysis/README.md`](analysis/README.md) says which does what,
and why the defaults are what they are.

## Methodology

The centerline chain is documented on its own, at length, in
**[docs/centerline.md](docs/centerline.md)**: how branches are numbered, what the
ratios mean, the quality metrics and orphan-component report, and the synthetic
phantoms the whole chain is calibrated and audited against.

Those phantoms are trees of known geometry, so they are also the way to test the
measurement chain without any data:

```bash
python -m analysis.phantom --orders 7 --spacing 1.0 --rd 1.5 --rl 1.4 \
    --output tree.nii.gz
python -m analysis.centerline --input tree.nii.gz --ordering strahler_dd \
    --min-branch-length 0 --fit-orders 1 6
```

The confidence intervals that come out should contain the ratios that went in —
here `R_d = 1.456 [1.394, 1.519]` against the 1.5 imposed, and
`R_l = 1.413 [1.357, 1.471]` against 1.4. Watch the fit range: at 1 mm the
resolution floor is three voxels, so the thinnest orders of a 7-order tree are
not measurable and the chain says so rather than fitting them anyway.
