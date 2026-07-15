# -*- coding: utf-8 -*-
"""
train.py

Single-stage training entrypoint for the DynUNet vascular segmentation
model. Reads all hyperparameters from config.py and the image/label pairs
from the manifest pointed to by config.MANIFEST_PATH.
"""
import logging
import os

import pytorch_lightning
import torch
import torch.multiprocessing
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import _LRScheduler

import config as cfg
import dataset
import manifest as manifest_module
from config_loss import DeepSupervisionDiceCELoss
from model import build_dynunet

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


class Net(pytorch_lightning.LightningModule):
    def __init__(self, train_ds, val_ds):
        super().__init__()
        self.model = build_dynunet(cfg)
        self.loss_function = DeepSupervisionDiceCELoss(
            deep_supr_num=cfg.DEEP_SUPERVISION_LEVELS
        )
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


def main():
    torch.set_float32_matmul_precision("medium")

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
        resume_from_checkpoint = last_ckpt_path
        logging.info(f"Resume training from checkpoint: {resume_from_checkpoint}")
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
