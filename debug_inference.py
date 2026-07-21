# -*- coding: utf-8 -*-
"""
debug_inference.py

Ad-hoc diagnostic: runs the same steps as inference.predict_volume but
prints intermediate stats to find where class information is lost
(model output, argmax, or the inverse-transform step).

Usage:
    python debug_inference.py --checkpoint <ckpt> --input <image.nii.gz>
"""
import argparse

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Invertd

import config as cfg
from inference import load_model, get_inference_transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, cfg, device)

    pre_transforms = get_inference_transforms(cfg)
    data = pre_transforms({"image": args.input})
    image = data["image"].unsqueeze(0).to(device)
    print("input image shape:", image.shape, "min/max/mean:",
          image.min().item(), image.max().item(), image.mean().item())

    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=image,
            roi_size=cfg.PATCH_SIZE,
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
            mode="gaussian",
        )

    print("logits shape:", logits.shape)
    for c in range(logits.shape[1]):
        ch = logits[0, c]
        print(f"  channel {c} ({cfg.CLASS_NAMES[c]}): "
              f"min={ch.min().item():.3f} max={ch.max().item():.3f} "
              f"mean={ch.mean().item():.3f}")

    # Per-voxel argmax channel counts, straight off the logits (pre-AsDiscrete)
    argmax_raw = logits[0].argmax(dim=0)
    vals, counts = torch.unique(argmax_raw, return_counts=True)
    print("raw argmax value counts (pre-AsDiscrete, in resampled space):")
    for v, c in zip(vals.tolist(), counts.tolist()):
        print(f"  class {v} ({cfg.CLASS_NAMES[v]}): {c} voxels")

    data["pred"] = AsDiscrete(argmax=True)(logits[0].cpu())
    vals, counts = np.unique(data["pred"].numpy(), return_counts=True)
    print("AsDiscrete(argmax=True) value counts (in resampled space):")
    for v, c in zip(vals.tolist(), counts.tolist()):
        print(f"  class {int(v)}: {c} voxels")

    invert = Invertd(
        keys="pred",
        transform=pre_transforms,
        orig_keys="image",
        nearest_interp=True,
        to_tensor=True,
    )
    pred = invert(data)["pred"]
    vals, counts = np.unique(pred.numpy(), return_counts=True)
    print("Final inverted-back label value counts (original grid):")
    for v, c in zip(vals.tolist(), counts.tolist()):
        print(f"  class {int(v)}: {c} voxels")


if __name__ == "__main__":
    main()
