# -*- coding: utf-8 -*-
"""
debug_val_crop.py

Reproduces a single training-style validation step (one 128^3 crop
centered on foreground, direct forward pass -- no sliding window) and
prints per-channel logit stats, in both fp32 and fp16 autocast, to
compare against the full-volume sliding-window path in inference.py.
"""
import argparse

import numpy as np
import torch

import config as cfg
import manifest as manifest_module
from inference import load_model
from transforms import get_val_transforms


def report(logits, label, tag):
    print(f"--- {tag} ---")
    for c in range(logits.shape[0]):
        ch = logits[c]
        print(f"  channel {c} ({cfg.CLASS_NAMES[c]}): "
              f"min={ch.min().item():.3f} max={ch.max().item():.3f} "
              f"mean={ch.mean().item():.3f}")
    vals, counts = np.unique(label.numpy(), return_counts=True)
    print("  GT voxel counts in this crop:",
          {cfg.CLASS_NAMES[int(v)]: int(c) for v, c in zip(vals, counts)})
    pred = logits.argmax(dim=0)
    vals, counts = torch.unique(pred, return_counts=True)
    print("  pred voxel counts in this crop:",
          {cfg.CLASS_NAMES[int(v)]: int(c) for v, c in zip(vals.tolist(), counts.tolist())})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tries", type=int, default=20,
                         help="crops to sample looking for one with real artery GT voxels")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, cfg, device)

    entries = manifest_module.load_manifest(cfg.MANIFEST_PATH)
    val_entries = [e for e in entries if e["split"] == "val"]
    entry = {"image": val_entries[0]["image"], "label": val_entries[0]["label"]}

    transforms = get_val_transforms(cfg, num_samples=1)

    artery_class = cfg.CLASS_NAMES.index("artery")
    picked = None
    for _ in range(args.tries):
        out = transforms(entry)[0]
        if (out["label"] == artery_class).sum() > 50:
            picked = out
            break
    if picked is None:
        print("No crop with >50 artery voxels found in "
              f"{args.tries} tries -- using the last one anyway.")
        picked = out

    image = picked["image"].unsqueeze(0).to(device)
    label = picked["label"][0]  # drop channel dim

    with torch.no_grad():
        logits_fp32 = model(image)[0].cpu()
    report(logits_fp32, label, "fp32, direct forward, single 128^3 crop")

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16):
        logits_fp16 = model(image)[0].float().cpu()
    report(logits_fp16, label, "fp16 autocast, direct forward, single 128^3 crop")


if __name__ == "__main__":
    main()
