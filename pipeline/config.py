# -*- coding: utf-8 -*-
"""
config.py

Single source of truth for the vascular tree segmentation training pipeline.
Nothing else in the codebase should hardcode paths, patch geometry, or class
counts -- change values here instead.
"""
import ast
import hashlib
import json
import os

# --- Paths ---
# DATASET_ROOT is where the run *writes* (cache, checkpoints, logs); the
# manifest is an input that lives with the source, so it is resolved from
# the repository rather than from DATASET_ROOT. Anchoring it on this file's
# location keeps it correct whatever directory the job is launched from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROOT_DIR = os.environ.get("DATASET_ROOT", "./data")
MANIFEST_PATH = os.environ.get(
    "MANIFEST_PATH", os.path.join(REPO_ROOT, "manifests", "manifest_ct.json")
)
LOG_DIR = os.path.join(ROOT_DIR, "logs")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")

# Overridable from the environment so a variant run (a different loss, a
# fine-tuning) gets its own logs/checkpoints without editing this file --
# see finetune_cldice.slurm.
EXPERIMENT_NAME = os.environ.get(
    "EXPERIMENT_NAME", "vessel_segmentation_vein_artery_vibe_v1"
)

# CACHE_DIR is further down: it is keyed on the preprocessing, not on the
# name of the run, and the values it depends on are defined below.

# --- Classes ---
# Index 0 must be the background. Add/remove foreground class names here --
# NUM_CLASSES and the model's out_channels follow automatically, nothing
# else in the codebase hardcodes a class count.
CLASS_NAMES = ["background", "artery", "vein"]
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
    3: CLASS_NAMES.index("artery"),
    4: CLASS_NAMES.index("vein"),
}

# --- Geometry ---
PATCH_SIZE = (128, 128, 128)
TARGET_SPACING = (1.0, 1.0, 1.0)

# --- Preprocessing ---
# Data is expected to already be roughly z-scored, but per-volume
# normalization is kept on by default since it is a safe no-op on data
# that is already normalized and a useful safety net otherwise.
NORMALIZE_INTENSITY = True

# --- Dataset cache ---
# PersistentDataset keys its cache on item identity alone -- the image/label
# paths -- never on the transform pipeline. Change TARGET_SPACING and it
# happily reloads tensors resampled at the old one, without a word.
#
# So the missing half of the key is computed here: everything that decides
# what lands in the cache, which is exactly the three values `_base_transforms`
# reads plus the code of transforms.py itself. PATCH_SIZE and
# POS_NEG_SAMPLE_RATIO are deliberately absent -- they act after the first
# random transform, downstream of what is cached.
#
# transforms.py goes in as its AST with docstrings stripped, not as text, so
# rewording a comment does not invalidate a cohort's worth of preprocessing
# while an edit to `mode=` or to the label filter does.
#
# The consequence is that nothing has to be renamed to get a fresh cache, and
# two runs that preprocess identically share one without being told to.
def _preprocessing_fingerprint():
    source = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transforms.py")
    with open(source) as handle:
        tree = ast.parse(handle.read())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:]
    blob = json.dumps(PREPROCESSING_SETTINGS, sort_keys=True) + "\n" + ast.dump(tree)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


PREPROCESSING_SETTINGS = {
    "label_class_map": {str(k): v for k, v in LABEL_CLASS_MAP.items()},
    "target_spacing": list(TARGET_SPACING),
    "normalize_intensity": NORMALIZE_INTENSITY,
}
PREPROCESSING_FINGERPRINT = _preprocessing_fingerprint()
CACHE_DIR = os.path.join(ROOT_DIR, "cache", PREPROCESSING_FINGERPRINT)

# --- Patch sampling ---
# Number of patches drawn is a total *budget* per epoch, split evenly across
# available volumes: num_samples = ceil(budget / num_volumes).
TRAIN_PATCH_BUDGET = 500
VAL_PATCH_BUDGET = 100
# Ratio of patches centered on a foreground voxel vs a background voxel.
POS_NEG_SAMPLE_RATIO = (1, 1)

# --- Training ---
BATCH_SIZE = 6
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 8))
# The validation loader gets fewer: it runs 6 batches per epoch against the
# training loader's 28, and every worker holds its prefetched batches in
# /dev/shm (see PREFETCH_FACTOR).
VAL_NUM_WORKERS = int(os.environ.get("VAL_NUM_WORKERS", max(1, NUM_WORKERS // 4)))
# Batches per worker held ready. This is the main /dev/shm knob: a training
# batch is BATCH_SIZE x num_samples = 24 patches of 128^3, image and label,
# i.e. ~0.4 GiB, and every prefetched batch sits in shared memory until the
# main process consumes it. At the torch default of 2 the two loaders reserve
# ~10 GiB of /dev/shm between them, which is enough to exhaust it on a node
# whose /dev/shm is sized from the job's memory cgroup. 28 steps per epoch
# does not need that much read-ahead.
PREFETCH_FACTOR = int(os.environ.get("PREFETCH_FACTOR", 1))
# LEARNING_RATE and MAX_EPOCHS are read from the environment because a
# fine-tuning run is exactly the same setup with those two turned down; the
# PolyLR schedule is rebuilt from them, so a fine-tuning restarts the decay
# at the lower LR instead of resuming a schedule that has already reached 0.
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 1e-3))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", 2000))
DEEP_SUPERVISION_LEVELS = 4
SEED = 42
DEVICE_INDEX = 0

# Path to a checkpoint whose *weights only* seed this run (optimizer state,
# LR schedule and epoch counter are not restored -- that is what separates a
# fine-tuning from a resume). Empty means train from scratch. Auto-resuming
# this run's own last.ckpt still takes precedence, so a preempted
# fine-tuning picks up where it stopped rather than restarting from the
# original weights.
FINETUNE_FROM = os.environ.get("FINETUNE_FROM", "") or None

# --- Continuity term (clDice) ---
# Dice and CE are voxel-wise and barely notice a two-voxel break in a
# vessel, which nonetheless splits the tree and corrupts every branching
# statistic computed downstream. clDice scores the skeletons of the masks
# instead, so a break costs it a whole run of centerline. See config_loss.py.
#
# CLDICE_WEIGHT is the lambda in `DiceCE + lambda * clDice`; 0 disables the
# term entirely and reproduces the original loss exactly. 0.5 is the value
# the clDice paper uses for tubular structures.
CLDICE_WEIGHT = float(os.environ.get("CLDICE_WEIGHT", 0.0))
# Soft-skeletonization iterations. Must be >= the largest vessel radius in
# voxels or thick vessels never thin to a curve: ~10 mm across at
# TARGET_SPACING = 1 mm gives a radius of 5, plus one for margin.
CLDICE_ITERATIONS = int(os.environ.get("CLDICE_ITERATIONS", 6))
# Epochs over which the weight ramps linearly from 0. clDice on a
# not-yet-vessel-shaped prediction skeletonizes noise; the ramp keeps it
# quiet until there is something for it to fix. 0 disables the ramp.
CLDICE_WARMUP_EPOCHS = int(os.environ.get("CLDICE_WARMUP_EPOCHS", 20))
# How many patches of the batch the term scores. The batch that reaches the
# loss is BATCH_SIZE x num_samples patches (RandCropByPosNegLabeld returns
# num_samples per item and list_data_collate flattens them): 24 patches of
# 128^3 in the default setup, whose skeletonization graph does not fit in
# 93 GiB next to the network's own activations. clDice is a batch-level
# statistic, so scoring a random subset is a noisier estimate of the same
# quantity. 0 means no cap.
CLDICE_MAX_PATCHES = int(os.environ.get("CLDICE_MAX_PATCHES", 8))
CLDICE_SMOOTH = 1.0

# --- Debug mode: fast, tiny run to sanity-check the pipeline ---
DEBUG = os.environ.get("DEBUG", "0") == "1"
if DEBUG:
    TRAIN_PATCH_BUDGET = 4
    VAL_PATCH_BUDGET = 2
    MAX_EPOCHS = 1
    # The point of the smoke test is to exercise every code path, and a
    # warm-up that leaves the weight at ~0 for the single debug epoch would
    # skip the clDice branch entirely.
    CLDICE_WARMUP_EPOCHS = 0
    # A handful of patches from a single smoke-test volume don't need 8
    # worker processes per loader (16 total) -- each spawns a fresh
    # torch/monai import, which is what actually eats the RAM.
    NUM_WORKERS = 0
    VAL_NUM_WORKERS = 0
    # The smoke test only needs to exercise the pipeline mechanics, not
    # produce a useful model -- the real memory/compute hog is the forward
    # pass itself (128^3 patches through a 6-level, up-to-320-filter
    # DynUNet). Shrink the patch (must stay a multiple of 32: 5 stride-2
    # downsamples) and drop to batch size 1.
    PATCH_SIZE = (64, 64, 64)
    BATCH_SIZE = 1
