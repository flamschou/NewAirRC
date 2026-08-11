# -*- coding: utf-8 -*-
"""
connectivity.py

Ad-hoc utility: computes the number of connected components in a mask
(26-connectivity) and the fraction of the mask's volume covered by its
largest component -- a quick way to check how fragmented a
segmentation is.

Usage:
    python connectivity.py --path <mask.nii.gz> --label 2
    python connectivity.py --path <mask.nii.gz>          # any nonzero voxel
"""
import argparse

import nibabel as nib
import numpy as np
from scipy.ndimage import label as connected_components


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", required=True, help="Path to the mask (nifti)")
    parser.add_argument("--label", type=int, default=None, help="Raw label value to isolate before analysis. Default: any nonzero voxel")
    args = parser.parse_args()

    img = nib.load(args.path)
    data = np.asarray(img.dataobj)

    mask = (data == args.label) if args.label is not None else (data > 0)
    total_voxels = int(mask.sum())
    if total_voxels == 0:
        print("mask is empty, nothing to analyze")
        return

    structure = np.ones((3, 3, 3), dtype=int)  # 26-connectivity
    components, n_components = connected_components(mask, structure=structure)

    sizes = np.bincount(components.ravel())
    sizes[0] = 0  # background
    largest_size = int(sizes.max())
    largest_fraction = largest_size / total_voxels

    print(f"mask: {args.path}")
    print(f"label: {'nonzero' if args.label is None else args.label}")
    print(f"total voxels in mask: {total_voxels}")
    print(f"number of connected components: {n_components}")
    print(f"largest component: {largest_size} voxels ({100.0 * largest_fraction:.2f}% of mask volume)")


if __name__ == "__main__":
    main()
