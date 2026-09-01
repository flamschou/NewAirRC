# -*- coding: utf-8 -*-
"""
model.py

DynUNet (nnU-Net-style 3D U-Net) construction, kept separate from the
training loop so train.py and inference.py build the same network.
"""
import torch
from monai.networks.nets import DynUNet


def build_dynunet(config):
    """
    Args:
        config: config module (see config.py). Uses NUM_CLASSES and
            DEEP_SUPERVISION_LEVELS.
    Returns:
        torch.nn.Module: the DynUNet, with explicit biases added to any
            Conv3d layer that MONAI left without one (instance norm
            defaults to no conv bias).
    """
    model = DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=config.NUM_CLASSES,
        kernel_size=[3, 3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2, 2],
        upsample_kernel_size=[2, 2, 2, 2, 2],
        filters=[32, 64, 128, 256, 320, 320],
        norm_name=("INSTANCE", {"eps": 1e-05, "affine": True}),
        act_name=("leakyrelu", {"negative_slope": 0.01, "inplace": True}),
        deep_supervision=True,
        deep_supr_num=config.DEEP_SUPERVISION_LEVELS,
        res_block=True,
        trans_bias=True,
    )
    _ensure_conv_bias(model)
    return model


def load_checkpoint_weights(model, checkpoint_path, map_location=None):
    """
    Loads the DynUNet weights out of a PyTorch Lightning checkpoint written
    by train.py, where the network sits under `self.model` and every key is
    therefore prefixed with "model.".

    Only the weights: the optimizer state, LR schedule and epoch counter in
    the checkpoint are ignored. That is what a fine-tuning wants (and what
    inference wants); a *resume* goes through Trainer.fit(ckpt_path=...)
    instead.

    Args:
        model (torch.nn.Module): as returned by build_dynunet.
        checkpoint_path (str): path to a `.ckpt` file.
        map_location: forwarded to torch.load.
    Returns:
        torch.nn.Module: the same model, weights loaded in place.
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint)
    prefix = "model."
    weights = {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    model.load_state_dict(weights)
    return model


def _ensure_conv_bias(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv3d) and module.bias is None:
            module.bias = torch.nn.Parameter(torch.zeros(module.out_channels))
