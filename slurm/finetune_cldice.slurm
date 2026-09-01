#!/bin/bash
#SBATCH --job-name=vessel_cldice_ft
#SBATCH --partition=gpuh100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=86G
# 150 epochs where the baseline does 2000 in a day, plus ~40% for the
# skeletonization passes: ~3h. 6 asks for enough slack to survive a slow
# node without sitting in the queue for a full day's worth of allocation.
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Fine-tunes the baseline checkpoint with the clDice continuity term added
# to the Dice+CE loss. Everything below is read by config.py from the
# environment; no source file needs to change between this and train.slurm.

# Submit from the repository root (`sbatch slurm/train.slurm`): SLURM_SUBMIT_DIR
# is the directory sbatch was invoked from, and `python -m pipeline.train`
# only resolves from there.
REPO_DIR="$SLURM_SUBMIT_DIR"
cd "$REPO_DIR"
if [ ! -d "$REPO_DIR/pipeline" ]; then
    echo "ERROR: submit from the repository root, not from $REPO_DIR" >&2
    exit 1
fi

source /etc/profile.d/modules.sh
module use --append /share/sharedmodules
module add conda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flamant_segmentation

# Machine-level only: where the run writes, and what the node can feed.
# Everything that describes the RUN is a flag on the command at the bottom,
# so slurm-<jobid>.out records the parameters that produced the checkpoint.
export DATASET_ROOT="/data/flamant/data/AirRC_runs/vessel_segmentation"

# The run this one starts from. Its checkpoints and logs are left untouched:
# the fine-tuning writes under its own --experiment-name. The dataset cache
# is shared automatically -- it is keyed on the preprocessing, which is
# identical here, not on the name of the run.
BASE_EXPERIMENT="vessel_segmentation_vein_artery_vibe_v1"
FINETUNE_FROM="$DATASET_ROOT/checkpoints/$BASE_EXPERIMENT/run/last.ckpt"

# The clDice backward recomputes each thinning iteration (gradient
# checkpointing), so it allocates and frees full-resolution buffers in a
# tight loop. Expandable segments keep that from fragmenting the pool.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Dataloader shared memory: a training batch is 24 patches of 128^3 (image
# and label), ~0.4 GiB, and every prefetched batch sits in /dev/shm until the
# main process takes it. The config defaults (PREFETCH_FACTOR=1,
# VAL_NUM_WORKERS=NUM_WORKERS/4, persistent workers) hold that to ~3.6 GiB.
# Turn PREFETCH_FACTOR and VAL_NUM_WORKERS down first if a node still runs
# out -- check what it offers with `df -h /dev/shm`.
export NUM_WORKERS="8"

echo "Node: $(hostname)"
echo "Conda base: $(conda info --base)"
echo "Python: $(which python)"
echo "Fine-tuning from: $FINETUNE_FROM"
if [ ! -f "$FINETUNE_FROM" ]; then
    echo "ERROR: base checkpoint not found" >&2
    exit 1
fi
python -c "import pytorch_lightning; print('pytorch_lightning OK:', pytorch_lightning.__version__)"

# --learning-rate is a tenth of the baseline's, over a short schedule: the
# point is to reconnect broken branches, not to relearn the segmentation --
# at 1e-3 the fine-tuning would walk away from the voxel-level accuracy
# already in the checkpoint.
#
# --cldice-weight 0.5 is the clDice paper's value for tubular structures;
# 6 skeletonization iterations covers the ~5-voxel radius of the largest
# vessels at 1 mm spacing; the weight ramps in over 20 epochs. The batch
# reaching the loss is 24 patches of 128^3 and the term scores 8 of them per
# step -- raising that raises peak GPU memory steeply, the run OOMed at 24
# on a 93 GiB H100.
python -m pipeline.train \
    --manifest "$REPO_DIR/manifests/manifest_vibe.json" \
    --experiment-name "${BASE_EXPERIMENT}_cldice_ft" \
    --finetune-from "$FINETUNE_FROM" \
    --learning-rate 1e-4 \
    --max-epochs 150 \
    --cldice-weight 0.5 \
    --cldice-iterations 6 \
    --cldice-warmup-epochs 20 \
    --cldice-max-patches 8
