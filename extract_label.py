# -*- coding: utf-8 -*-
"""
extract_label.py

Ad-hoc utility: keeps a single label value from a nifti volume and zeroes
out everything else (e.g. isolating the artery class from a multi-class
segmentation).

Usage:
    python extract_label.py --input <in.nii.gz> --output <out.nii.gz> --label 2
"""
import argparse

import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", type=int, default=2)
    args = parser.parse_args()

    img = nib.load(args.input)
    data = np.asarray(img.dataobj)

    vals, counts = np.unique(data, return_counts=True)
    print("input value counts:")
    for v, c in zip(vals.tolist(), counts.tolist()):
        print(f"  {v}: {c} voxels")

    out = np.where(data == args.label, data, 0).astype(data.dtype)

    nib.save(nib.Nifti1Image(out, img.affine, img.header), args.output)
    print(f"kept only label {args.label}, wrote {args.output}")


if __name__ == "__main__":
    main()

