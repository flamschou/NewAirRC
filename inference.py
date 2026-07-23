# -*- coding: utf-8 -*-
"""
inference.py

Sliding-window prediction on full volumes using a trained DynUNet checkpoint.
Applies the same deterministic preprocessing as training
(transforms.get_inference_transforms), runs the model patch-by-patch over
the whole volume, then maps the predicted label back onto the original
image's orientation/spacing before saving.
"""
import argparse
import glob
import logging
import os

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Invertd

import config as cfg
from model import build_dynunet
from transforms import get_inference_transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_model(checkpoint_path, config, device):
    """
    Args:
        checkpoint_path (str): path to a PyTorch Lightning `.ckpt` file
            written by train.py (Net wraps the DynUNet as `self.model`).
        config: config module (see config.py).
        device (torch.device)
    Returns:
        torch.nn.Module: DynUNet in eval mode, weights loaded, on `device`.
    """
    model = build_dynunet(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    prefix = "model."
    state_dict = {
        key[len(prefix):]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith(prefix)
    }
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_volume(image_path, model, config, device):
    """
    Runs sliding-window inference on a single image.

    Args:
        image_path (str): path to a NIfTI image.
        model (torch.nn.Module): eval-mode DynUNet (see load_model).
        config: config module (see config.py).
        device (torch.device)
    Returns:
        tuple[np.ndarray, np.ndarray]: (label_array, affine) in the
            *original* image's grid, ready to write out with nibabel.
    """
    pre_transforms = get_inference_transforms(config)
    data = pre_transforms({"image": image_path})
    image = data["image"].unsqueeze(0).to(device)  # add batch dim

    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=image,
            roi_size=config.PATCH_SIZE,
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
            mode="gaussian",
        )

    data["pred"] = AsDiscrete(argmax=True)(logits[0].cpu())

    invert = Invertd(
        keys="pred",
        transform=pre_transforms,
        orig_keys="image",
        nearest_interp=True,
        to_tensor=True,
    )
    pred = invert(data)["pred"]

    label_array = pred[0].numpy().astype(np.uint8)  # drop channel dim
    affine = pred.affine.numpy()
    return label_array, affine


def save_prediction(label_array, affine, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    nib.save(nib.Nifti1Image(label_array, affine), output_path)


PRED_SUFFIX = "_vascular_pred"


def _default_output_path(image_path, output_dir=None, suffix=PRED_SUFFIX):
    """
    Args:
        image_path (str): input image path.
        output_dir (str, optional): directory to place the output in;
            defaults to the input image's own directory.
        suffix (str): inserted before the extension, e.g.
            "case001.nii.gz" -> "case001_vascular_pred.nii.gz".
    Returns:
        str: output path.
    """
    directory, filename = os.path.split(image_path)
    if filename.endswith(".nii.gz"):
        stem, ext = filename[: -len(".nii.gz")], ".nii.gz"
    else:
        stem, ext = os.path.splitext(filename)
    return os.path.join(output_dir if output_dir is not None else directory, f"{stem}{suffix}{ext}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a .ckpt file")
    parser.add_argument("--input", help="Path to a single input image (.nii.gz)")
    parser.add_argument(
        "--input-dir",
        help="Directory of images to run inference on, searched recursively "
        "(**/*.nii.gz)",
    )
    parser.add_argument(
        "--output",
        help="Output path for a single --input prediction. Defaults to "
        "<input>_vascular_pred.nii.gz next to the input image.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for --input-dir. Defaults to writing "
        "<name>_vascular_pred.nii.gz next to each input image.",
    )
    parser.add_argument(
        "--output-suffix",
        default=PRED_SUFFIX,
        help="Suffix inserted before the extension of generated files, e.g. "
        f"'{PRED_SUFFIX}2' -> <name>{PRED_SUFFIX}2.nii.gz. "
        f"Defaults to '{PRED_SUFFIX}'.",
    )
    args = parser.parse_args()

    if bool(args.input) == bool(args.input_dir):
        parser.error("Pass exactly one of --input or --input-dir")

    if args.input:
        image_paths = [args.input]
        output_paths = [
            args.output or _default_output_path(args.input, suffix=args.output_suffix)
        ]
    else:
        image_paths = sorted(
            p
            for p in glob.glob(os.path.join(args.input_dir, "**", "*.nii.gz"), recursive=True)
            if PRED_SUFFIX not in os.path.basename(p)
        )
        if not image_paths:
            parser.error(f"No .nii.gz files found under {args.input_dir}")
        output_paths = [
            _default_output_path(p, args.output_dir, suffix=args.output_suffix)
            for p in image_paths
        ]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    model = load_model(args.checkpoint, cfg, device)

    for image_path, output_path in zip(image_paths, output_paths):
        logging.info(f"Running inference on {image_path}")
        label_array, affine = predict_volume(image_path, model, cfg, device)
        save_prediction(label_array, affine, output_path)
        logging.info(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
