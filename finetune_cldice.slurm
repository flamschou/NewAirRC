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

cd "$SLURM_SUBMIT_DIR"

source /etc/profile.d/modules.sh
module use --append /share/sharedmodules
module add conda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flamant_segmentation

export MANIFEST_PATH="$SLURM_SUBMIT_DIR/manifest_vibe.json"
export DATASET_ROOT="/data/flamant/data/AirRC_runs/vessel_segmentation"

# The run this one starts from. Its checkpoints, logs and dataset cache are
# left untouched: the fine-tuning writes under its own EXPERIMENT_NAME.
BASE_EXPERIMENT="vessel_segmentation_vein_artery_vibe_v1"

export EXPERIMENT_NAME="${BASE_EXPERIMENT}_cldice_ft"
export FINETUNE_FROM="$DATASET_ROOT/checkpoints/$BASE_EXPERIMENT/run/last.ckpt"
# Preprocessing is identical to the baseline, so share its PersistentDataset
# cache instead of re-caching the whole dataset under the new name.
export CACHE_NAME="$BASE_EXPERIMENT"

# The continuity term. 0.5 is the clDice paper's value for tubular
# structures; 6 skeletonization iterations covers the ~5-voxel radius of the
# largest vessels at 1 mm spacing; the weight ramps in over 20 epochs.
export CLDICE_WEIGHT="0.5"
export CLDICE_ITERATIONS="6"
export CLDICE_WARMUP_EPOCHS="20"
# The batch reaching the loss is BATCH_SIZE x num_samples = 24 patches of
# 128^3; the term scores 8 of them per step. Raising this raises peak GPU
# memory steeply -- the run OOMed at 24 on a 93 GiB H100.
export CLDICE_MAX_PATCHES="8"

# The clDice backward recomputes each thinning iteration (gradient
# checkpointing), so it allocates and frees full-resolution buffers in a
# tight loop. Expandable segments keep that from fragmenting the pool.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# A tenth of the baseline LR, and a short schedule. The point is to reconnect
# broken branches, not to relearn the segmentation -- at 1e-3 the fine-tuning
# would walk away from the voxel-level accuracy already in the checkpoint.
export LEARNING_RATE="1e-4"
export MAX_EPOCHS="150"

echo "Node: $(hostname)"
echo "Conda base: $(conda info --base)"
echo "Python: $(which python)"
echo "Fine-tuning from: $FINETUNE_FROM"
if [ ! -f "$FINETUNE_FROM" ]; then
    echo "ERROR: base checkpoint not found" >&2
    exit 1
fi
python -c "import pytorch_lightning; print('pytorch_lightning OK:', pytorch_lightning.__version__)"

python train.py
