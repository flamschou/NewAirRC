#!/bin/bash
#SBATCH --job-name=vessel_segmentation
#SBATCH --partition=gpuh100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=86G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

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

# Machine-level only: where the run writes. Everything that describes the RUN
# is a flag on the command at the bottom, so slurm-<jobid>.out records the
# parameters that produced the checkpoint.
export DATASET_ROOT="/data/flamant/data/AirRC_runs/vessel_segmentation"  # TODO: persistent path for cache/checkpoints/logs

echo "Node: $(hostname)"
echo "Conda base: $(conda info --base)"
echo "Python: $(which python)"
python -c "import pytorch_lightning; print('pytorch_lightning OK:', pytorch_lightning.__version__)"

python -m pipeline.train --manifest "$REPO_DIR/manifests/manifest_vibe.json"
