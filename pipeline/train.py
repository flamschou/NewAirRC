# -*- coding: utf-8 -*-
"""
train.py

Single-stage training entrypoint for the DynUNet vascular segmentation model.

Settings resolve in three layers, each overriding the one below:

    command-line flag  >  environment variable  >  config.py default

What describes the RUN is a flag (--learning-rate, --cldice-weight, ...);
what describes the MACHINE stays in the environment (DATASET_ROOT,
NUM_WORKERS, ...) -- see .env.example.

Usage:
    python -m pipeline.train --manifest manifests/manifest_vibe.json
    python -m pipeline.train --debug --manifest manifests/manifest.example.json
"""
import argparse
import logging
import os
import sys

import pytorch_lightning
import torch
import torch.multiprocessing
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import _LRScheduler

from . import config as cfg
from . import dataset
from . import manifest as manifest_module
from .config_loss import build_loss, cldice_warmup_weight
from .model import build_dynunet, load_checkpoint_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

torch.multiprocessing.set_sharing_strategy("file_system")


class PolyLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        initial_lr: float,
        max_steps: int,
        exponent: float = 0.9,
        current_step: int = None,
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        return self._last_lr

    def state_dict(self):
        return {"ctr": self.ctr, "_last_lr": self._last_lr}

    def load_state_dict(self, state_dict):
        self.ctr = state_dict["ctr"]
        self._last_lr = state_dict["_last_lr"]


class Net(pytorch_lightning.LightningModule):
    def __init__(self, train_ds, val_ds):
        super().__init__()
        self.model = build_dynunet(cfg)
        self.loss_function = build_loss(cfg)
        self.post_pred = Compose(
            [
                EnsureType("tensor"),
                AsDiscrete(argmax=True, to_onehot=cfg.NUM_CLASSES),
            ]
        )
        self.post_label = Compose(
            [EnsureType("tensor"), AsDiscrete(to_onehot=cfg.NUM_CLASSES)]
        )
        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )
        self.train_loader, self.val_loader = dataset.build_dataloaders(
            cfg, train_ds, val_ds
        )

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        # The clDice weight is ramped in rather than fixed (see
        # config_loss.cldice_warmup_weight). Logged because val_loss is
        # computed with the same weight, so it is not comparable across the
        # warm-up epochs unless you can see where the ramp was.
        self.loss_function.cldice_weight = cldice_warmup_weight(
            self.current_epoch, cfg.CLDICE_WEIGHT, cfg.CLDICE_WARMUP_EPOCHS
        )
        self.log(
            "cldice_weight",
            self.loss_function.cldice_weight,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        if outputs.ndim == 6:
            main_output = outputs[:, 0, ...]
        else:
            main_output = outputs

        outputs_post = [self.post_pred(i) for i in decollate_batch(main_output)]
        labels_post = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs_post, y=labels_post)

    def on_validation_epoch_end(self):
        dice_metric = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        self.log(
            "val_dice_metric", dice_metric, on_step=False, on_epoch=True, prog_bar=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=cfg.LEARNING_RATE,
            momentum=0.99,
            nesterov=True,
            weight_decay=3e-5,
        )
        scheduler = PolyLRScheduler(
            optimizer=optimizer,
            initial_lr=cfg.LEARNING_RATE,
            max_steps=self.trainer.max_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


def build_parser():
    """
    Every default is the value config.py already resolved, and config.py
    resolves each from the environment first. That is what gives, for free
    and without the value being written down twice:

        flag  >  environment variable  >  config.py default

    So `--learning-rate 1e-4` and `LEARNING_RATE=1e-4` still mean the same
    thing, and slurm/*.slurm keep working unchanged by exporting variables.

    What belongs here rather than in the environment is anything that
    describes the RUN: a .env is gitignored and invisible, so a learning rate
    living there makes two people running the same command get different
    results with nothing in the log to say why. A flag lands in the shell
    history and in slurm-<jobid>.out. Machine-level settings -- where the
    data is, how many workers the node can feed -- stay in the environment,
    where nobody wants to retype them.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    run = parser.add_argument_group("the run")
    run.add_argument("--manifest", default=cfg.MANIFEST_PATH,
                     help=f"Image/label pairs to train on. Default: {cfg.MANIFEST_PATH}")
    run.add_argument("--experiment-name", default=cfg.EXPERIMENT_NAME, metavar="NAME",
                     help="Names this run's checkpoints and TensorBoard logs. It no longer "
                          "affects the dataset cache, which is keyed on the preprocessing "
                          f"itself. Default: {cfg.EXPERIMENT_NAME}")
    run.add_argument("--learning-rate", type=float, default=cfg.LEARNING_RATE, metavar="LR",
                     help=f"Initial LR of the PolyLR schedule. Default: {cfg.LEARNING_RATE}")
    run.add_argument("--max-epochs", type=int, default=cfg.MAX_EPOCHS, metavar="N",
                     help=f"Default: {cfg.MAX_EPOCHS}")
    run.add_argument("--finetune-from", default=cfg.FINETUNE_FROM, metavar="CKPT",
                     help="Checkpoint whose WEIGHTS ONLY seed this run -- not its optimizer "
                          "state or epoch counter, which is what separates a fine-tuning from "
                          "a resume. Auto-resuming this run's own last.ckpt still takes "
                          "precedence. Default: train from scratch")
    run.add_argument("--debug", action="store_true", default=cfg.DEBUG,
                     help="Fast, tiny run (64^3 patches, one epoch, no workers) that exercises "
                          "every code path without producing a useful model. Use it to check "
                          "an install before starting anything real")

    cldice = parser.add_argument_group(
        "continuity term (clDice)",
        "Dice and cross-entropy barely notice a two-voxel break in a vessel, which "
        "nonetheless splits the tree and corrupts every branching statistic measured "
        "downstream. clDice scores the skeletons instead.")
    cldice.add_argument("--cldice-weight", type=float, default=cfg.CLDICE_WEIGHT,
                        metavar="LAMBDA",
                        help="The lambda in `DiceCE + lambda * clDice`. 0 disables the term "
                             "entirely and reproduces the original loss exactly; 0.5 is the "
                             f"clDice paper's value for tubular structures. Default: "
                             f"{cfg.CLDICE_WEIGHT}")
    cldice.add_argument("--cldice-iterations", type=int, default=cfg.CLDICE_ITERATIONS,
                        metavar="N",
                        help="Soft-skeletonization iterations. Must be >= the largest vessel "
                             "radius in voxels, or thick vessels never thin to a curve. "
                             f"Default: {cfg.CLDICE_ITERATIONS}")
    cldice.add_argument("--cldice-warmup-epochs", type=int, default=cfg.CLDICE_WARMUP_EPOCHS,
                        metavar="N",
                        help="Epochs over which the weight ramps linearly from 0; on a "
                             "not-yet-vessel-shaped prediction the term skeletonizes noise. "
                             f"0 disables the ramp. Default: {cfg.CLDICE_WARMUP_EPOCHS}")
    cldice.add_argument("--cldice-max-patches", type=int, default=cfg.CLDICE_MAX_PATCHES,
                        metavar="N",
                        help="How many patches of the batch the term scores. Raising this "
                             "raises peak GPU memory steeply -- the run OOMed at 24 on a "
                             f"93 GiB H100. 0 means no cap. Default: {cfg.CLDICE_MAX_PATCHES}")
    return parser


def apply_overrides(args):
    """
    Writes the parsed values back onto the config module.

    Unusual, and deliberate: `dataset.build_datasets(config, ...)`,
    `model.build_dynunet(config)` and `config_loss.build_loss(config)` all
    take the config module and read attributes off it, and train.py reads
    `cfg.X` in fifteen places. Setting the attributes here means none of that
    has to change, and there is exactly one place where a flag becomes a
    setting. Threading a settings object through the whole chain instead
    would touch four modules for no benefit this pipeline can use.

    DEBUG is not overridable this way: config.py acts on it at import time,
    shrinking the patch and the batch, so it has to be set before the import
    -- which is why --debug re-execs rather than assigning.
    """
    cfg.MANIFEST_PATH = args.manifest
    cfg.EXPERIMENT_NAME = args.experiment_name
    cfg.LEARNING_RATE = args.learning_rate
    cfg.MAX_EPOCHS = args.max_epochs
    cfg.FINETUNE_FROM = args.finetune_from or None
    cfg.CLDICE_WEIGHT = args.cldice_weight
    cfg.CLDICE_ITERATIONS = args.cldice_iterations
    cfg.CLDICE_WARMUP_EPOCHS = args.cldice_warmup_epochs
    cfg.CLDICE_MAX_PATCHES = args.cldice_max_patches


def main():
    args = build_parser().parse_args()
    if args.debug and not cfg.DEBUG:
        # config.py resolves DEBUG at import time -- it is what shrinks
        # PATCH_SIZE and BATCH_SIZE -- so the flag cannot be applied after the
        # fact. Re-exec with the variable set and let the import do its job.
        os.environ["DEBUG"] = "1"
        os.execv(sys.executable, [sys.executable, "-m", "pipeline.train"] + sys.argv[1:])
    apply_overrides(args)

    torch.set_float32_matmul_precision("medium")
    # Seeds python/numpy/torch, and with workers=True the dataloader workers
    # too -- which is what actually matters here, since the patch sampling
    # and every augmentation run inside them. Without this the validation
    # crops differ from run to run, so val_dice_metric cannot be compared
    # across runs and a fine-tuning cannot be told apart from its baseline.
    pytorch_lightning.seed_everything(cfg.SEED, workers=True)

    entries = manifest_module.load_manifest(cfg.MANIFEST_PATH)
    train_entries, val_entries = manifest_module.split_manifest(entries)

    if cfg.DEBUG:
        train_entries = train_entries[:4]
        val_entries = val_entries[:4]

    train_ds, val_ds = dataset.build_datasets(cfg, train_entries, val_entries)

    version_name = "run"
    tb_logger = pytorch_lightning.loggers.TensorBoardLogger(
        save_dir=cfg.LOG_DIR, name=cfg.EXPERIMENT_NAME, version=version_name
    )
    check_point_dir = os.path.join(cfg.CHECKPOINT_DIR, cfg.EXPERIMENT_NAME, version_name)

    net = Net(train_ds=train_ds, val_ds=val_ds)

    checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(
        dirpath=check_point_dir,
        filename=None,
        monitor=None,
        save_weights_only=False,
        every_n_epochs=10,
        save_on_train_epoch_end=False,
        enable_version_counter=False,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    last_ckpt_path = os.path.join(check_point_dir, "last.ckpt")
    if os.path.exists(last_ckpt_path):
        # A resume: Lightning restores weights, optimizer, LR schedule and
        # epoch counter, so this also covers a fine-tuning that got
        # preempted partway through.
        resume_from_checkpoint = last_ckpt_path
        logging.info(f"Resume training from checkpoint: {resume_from_checkpoint}")
    elif cfg.FINETUNE_FROM:
        # A fine-tuning: weights only. Deliberately *not* passed as
        # Trainer.fit(ckpt_path=...), which would also restore the source
        # run's optimizer state and epoch counter -- and with the epoch
        # counter comes its PolyLR schedule, already decayed to ~0 at the
        # end of that run, so nothing would move.
        resume_from_checkpoint = None
        load_checkpoint_weights(net.model, cfg.FINETUNE_FROM, map_location="cpu")
        logging.info(f"Fine-tuning from weights: {cfg.FINETUNE_FROM}")
        logging.info(
            f"  lr={cfg.LEARNING_RATE}, max_epochs={cfg.MAX_EPOCHS}, "
            f"cldice_weight={cfg.CLDICE_WEIGHT} "
            f"(warmup {cfg.CLDICE_WARMUP_EPOCHS} epochs, "
            f"{cfg.CLDICE_ITERATIONS} skeletonization iterations)"
        )
    else:
        resume_from_checkpoint = None
        logging.info("Start training from scratch")

    if torch.cuda.is_available():
        accelerator, devices, precision = "gpu", [cfg.DEVICE_INDEX], "16-mixed"
    elif torch.backends.mps.is_available():
        accelerator, devices, precision = "mps", 1, 32
    else:
        accelerator, devices, precision = "cpu", 1, 32

    trainer = pytorch_lightning.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=cfg.MAX_EPOCHS,
        logger=tb_logger,
        enable_checkpointing=True,
        num_sanity_val_steps=1,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=precision,
    )
    trainer.fit(net, ckpt_path=resume_from_checkpoint)


if __name__ == "__main__":
    main()
