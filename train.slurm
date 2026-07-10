#!/bin/bash
#SBATCH --job-name=vessel_segmentation
#SBATCH --partition=gpuh100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=86G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

cd "$SLURM_SUBMIT_DIR"

source /etc/profile.d/modules.sh
module use --append /share/sharedmodules
module add conda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flamant_segmentation

# config.py reads these from the environment; nothing else needs to change.
export MANIFEST_PATH="$SLURM_SUBMIT_DIR/manifest.json"
export DATASET_ROOT="/data/flamant/data/AirRC_runs/vessel_segmentation"  # TODO: persistent path for cache/checkpoints/logs

python train.py
