# -*- coding: utf-8 -*-
"""
config.py

Single source of truth for the vascular tree segmentation training pipeline.
Nothing else in the codebase should hardcode paths, patch geometry, or class
counts -- change values here instead.
"""
import os

# --- Paths ---
ROOT_DIR = os.environ.get("DATASET_ROOT", "./data")
MANIFEST_PATH = os.environ.get("MANIFEST_PATH", os.path.join(ROOT_DIR, "manifest.json"))
CACHE_DIR = os.path.join(ROOT_DIR, "cache")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")

EXPERIMENT_NAME = "vessel_segmentation_vein_artery"

# --- Classes ---
# Index 0 must be the background. Add/remove foreground class names here --
# NUM_CLASSES and the model's out_channels follow automatically, nothing
# else in the codebase hardcodes a class count.
# Example for a vein/artery split: ["background", "vein", "artery"]
CLASS_NAMES = ["background", "vein", "artery"]
NUM_CLASSES = len(CLASS_NAMES)

# Raw label files may carry more classes than we train on (e.g. the
# vascular_gen generator also labels airway structures as classes 1-2
# alongside vessel classes 3-4: raw 3 = artery, raw 4 = vein). LABEL_CLASS_MAP
# maps each raw integer value to the training class index it should become;
# any raw value not listed here (including airway classes 1-2) collapses to
# background=0. This remapping happens once, in transforms.py, so raw label
# files don't need to be pre-processed on disk. Keys must match indices into
# CLASS_NAMES.
LABEL_CLASS_MAP = {
    4: CLASS_NAMES.index("vein"),
    3: CLASS_NAMES.index("artery"),
}

# --- Geometry ---
PATCH_SIZE = (128, 128, 128)
TARGET_SPACING = (1.0, 1.0, 1.0)

# --- Preprocessing ---
# Data is expected to already be roughly z-scored, but per-volume
# normalization is kept on by default since it is a safe no-op on data
# that is already normalized and a useful safety net otherwise.
NORMALIZE_INTENSITY = True

# --- Patch sampling ---
# Number of patches drawn is a total *budget* per epoch, split evenly across
# available volumes: num_samples = ceil(budget / num_volumes).
TRAIN_PATCH_BUDGET = 500
VAL_PATCH_BUDGET = 100
# Ratio of patches centered on a foreground voxel vs a background voxel.
POS_NEG_SAMPLE_RATIO = (1, 1)

# --- Training ---
BATCH_SIZE = 6
NUM_WORKERS = 8
LEARNING_RATE = 1e-3
MAX_EPOCHS = 2000
DEEP_SUPERVISION_LEVELS = 4
SEED = 42
DEVICE_INDEX = 0

# --- Debug mode: fast, tiny run to sanity-check the pipeline ---
DEBUG = os.environ.get("DEBUG", "0") == "1"
if DEBUG:
    TRAIN_PATCH_BUDGET = 4
    VAL_PATCH_BUDGET = 2
    MAX_EPOCHS = 1
    # A handful of patches from a single smoke-test volume don't need 8
    # worker processes per loader (16 total) -- each spawns a fresh
    # torch/monai import, which is what actually eats the RAM.
    NUM_WORKERS = 0
    # The smoke test only needs to exercise the pipeline mechanics, not
    # produce a useful model -- the real memory/compute hog is the forward
    # pass itself (128^3 patches through a 6-level, up-to-320-filter
    # DynUNet). Shrink the patch (must stay a multiple of 32: 5 stride-2
    # downsamples) and drop to batch size 1.
    PATCH_SIZE = (64, 64, 64)
    BATCH_SIZE = 1
