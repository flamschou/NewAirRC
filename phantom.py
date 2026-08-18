# -*- coding: utf-8 -*-
"""
phantom.py

Builds a synthetic vascular tree whose branching ratios are known exactly,
rasterizes it at a chosen voxel size, and blurs and corrupts it like a real
segmentation. It exists to calibrate `centerline.py`, not to look pretty.

The point: a measured R_d of 1.45 has two indistinguishable explanations --
a segmentation that misses the small vessels, or a measuring chain that
compresses the dynamic range at this voxel size. Nothing in an in-vivo case
separates them, because the truth is unknown there. Here it is imposed, so
running the whole chain on the output and comparing gives the bias of the
chain alone. Do that before believing any number the chain produces, and
redo it at the voxel size of the study.

The tree is a symmetric binary tree in Strahler order: every order-n segment
splits into two of order n-1, so

    R_b = 2 exactly
    R_d = --rd exactly       D_n = D_1 * R_d^(n-1)
    R_l = --rl exactly       L_n = L_1 * R_l^(n-1)

and, being symmetric, elements and segments coincide: every element is one
segment. That is a simplification and a real tree is not like this -- what it
buys is that any departure the chain reports is entirely the chain's.

Usage:
    python phantom.py --orders 8 --spacing 1.5 --output tree.nii.gz
    python centerline.py --input tree.nii.gz --ordering strahler_dd --fit-orders 1 8
"""
import argparse

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter


def unit(vector):
    """Normalizes a vector, leaving the zero vector alone."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def build_tree(orders, root_diameter, root_length, rd, rl, half_angle, taper_jitter, rng):
    """
    Lays out the segments of the tree, from the trunk down.

    Each bifurcation is planar and symmetric, and its plane is turned by 90
    degrees at the next generation -- the alternation is what keeps the two
    halves of the tree from growing into each other, which would weld them
    together on the raster and give the skeleton the loops it is meant to be
    tested against.

    `taper_jitter` multiplies every diameter and length by a lognormal draw.
    With 0 the tree is exact and the ratios come back to the third decimal;
    the interesting question is how much scatter the fit tolerates, so run it
    at 0 first, then at the dispersion of a real tree (0.1 to 0.2).

    Returns a list of segment dicts, trunk first.
    """
    segments = []

    def grow(start, direction, up, order, parent):
        diameter = root_diameter * rd ** (order - orders)
        length = root_length * rl ** (order - orders)
        if taper_jitter > 0:
            diameter *= float(rng.lognormal(0.0, taper_jitter))
            length *= float(rng.lognormal(0.0, taper_jitter))
        end = start + direction * length
        index = len(segments)
        segments.append({
            "segment_id": index, "parent_id": parent, "strahler": order,
            "length_mm": float(length), "diameter_mm": float(diameter),
            "x0_mm": float(start[0]), "y0_mm": float(start[1]), "z0_mm": float(start[2]),
            "x1_mm": float(end[0]), "y1_mm": float(end[1]), "z1_mm": float(end[2]),
        })
        if order <= 1:
            return
        # the bifurcation plane is spanned by the parent direction and `up`;
        # the children take the old plane normal as their own `up`, which
        # rotates the next bifurcation a quarter turn
        in_plane = unit(up - np.dot(up, direction) * direction)
        normal = unit(np.cross(direction, in_plane))
        for sign in (1.0, -1.0):
            child = unit(np.cos(half_angle) * direction + sign * np.sin(half_angle) * in_plane)
            grow(end, child, normal, order - 1, index)

    grow(np.zeros(3), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), orders, -1)
    return segments


def rasterize(segments, spacing, margin):
    """
    Draws the segments as capsules on a grid, with an anti-aliased surface.

    A voxel gets the fraction of it that the vessel covers, approximated from
    its distance to the axis. Drawing hard binary capsules instead would put
    every wall exactly on a voxel boundary, which is the one thing a real
    acquisition never does and which would flatter the radius estimates by
    removing the partial volume the chain has to cope with.

    Returns the occupancy volume in [0, 1] and the world origin of voxel 0.
    """
    ends = np.array([[[s["x0_mm"], s["y0_mm"], s["z0_mm"]], [s["x1_mm"], s["y1_mm"], s["z1_mm"]]]
                     for s in segments]).reshape(-1, 3)
    radius = 0.5 * max(s["diameter_mm"] for s in segments)
    origin = ends.min(axis=0) - (radius + margin)
    shape = np.ceil((ends.max(axis=0) + radius + margin - origin) / spacing).astype(int) + 1

    volume = np.zeros(shape, dtype=np.float32)
    for segment in segments:
        p0 = np.array([segment["x0_mm"], segment["y0_mm"], segment["z0_mm"]])
        p1 = np.array([segment["x1_mm"], segment["y1_mm"], segment["z1_mm"]])
        r = 0.5 * segment["diameter_mm"]
        low = np.maximum(np.floor((np.minimum(p0, p1) - r - spacing - origin) / spacing).astype(int), 0)
        high = np.minimum(np.ceil((np.maximum(p0, p1) + r + spacing - origin) / spacing).astype(int) + 1, shape)
        if np.any(high <= low):
            continue

        grid = np.meshgrid(*(np.arange(a, b) for a, b in zip(low, high)), indexing="ij")
        points = np.stack(grid, axis=-1) * spacing + origin
        axis = p1 - p0
        squared = float(axis @ axis)
        t = np.clip(((points - p0) @ axis) / squared, 0.0, 1.0)[..., None] if squared > 0 else 0.0
        distance = np.linalg.norm(points - (p0 + t * axis), axis=-1)
        coverage = np.clip(0.5 + (r - distance) / spacing, 0.0, 1.0)
        window = volume[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
        np.maximum(window, coverage.astype(np.float32), out=window)

    return volume, origin


def degrade(volume, spacing, blur_mm, noise, threshold, rng):
    """
    Blurs the occupancy, adds noise to it and re-thresholds.

    This is the acquisition and the segmentation lumped into one crude step.
    The blur is the point spread function, and it is what actually destroys
    the thin vessels -- a tube one voxel across survives a threshold but not
    a blur wider than itself. The noise roughens the surface, which is what
    thinning turns into spurs, so it is also what makes the pruning sweep
    mean anything.
    """
    if blur_mm > 0:
        volume = gaussian_filter(volume, blur_mm / spacing)
    if noise > 0:
        volume = volume + rng.normal(0.0, noise, volume.shape).astype(np.float32)
    return volume >= threshold


def write_segments_csv(path, segments):
    """The ground truth, one row per segment."""
    columns = ("segment_id", "parent_id", "strahler", "length_mm", "diameter_mm",
               "x0_mm", "y0_mm", "z0_mm", "x1_mm", "y1_mm", "z1_mm")
    rows = [",".join(columns)]
    for segment in segments:
        rows.append(",".join(f"{segment[c]:.4f}" if isinstance(segment[c], float) else str(segment[c])
                             for c in columns))
    with open(path, "w") as handle:
        handle.write("\n".join(rows) + "\n")


def print_truth(segments, orders, spacing, rd, rl):
    """
    Prints what the chain should find, and where the grid stops allowing it.

    The last column is the whole reason for this file: an order whose
    diameter is under three voxels cannot be measured, only guessed, so the
    fit range that `centerline.py` will honestly support is known here in
    advance rather than discovered afterwards.
    """
    # every bifurcation is symmetric with d_parent = R_d * d_child, so
    # d_p^n = 2 d_c^n solves in closed form and centerline.py's histogram of
    # Murray exponents has a single value to be compared against
    murray = np.log(2.0) / np.log(rd) if rd > 1.0 else float("inf")
    print(f"\n=== ground truth (R_b = 2.000, R_d = {rd:.3f}, R_l = {rl:.3f}, "
          f"Murray exponent = {murray:.3f}) ===")
    print("ord     n   diameter_mm   length_mm   diameter in voxels")
    usable = []
    for order in range(1, orders + 1):
        group = [s for s in segments if s["strahler"] == order]
        diameter = float(np.mean([s["diameter_mm"] for s in group]))
        length = float(np.mean([s["length_mm"] for s in group]))
        voxels = diameter / spacing
        flag = "" if voxels >= 3.0 else "   <- under 3 voxels, not measurable"
        if voxels >= 3.0:
            usable.append(order)
        print(f"{order:3d} {len(group):5d} {diameter:13.3f} {length:11.3f} {voxels:20.2f}{flag}")
    if usable:
        print(f"the chain can only be held to orders {min(usable)}..{max(usable)} at {spacing} mm; "
              f"run centerline.py with --fit-orders {min(usable)} {max(usable)}")
    else:
        print(f"no order resolves at {spacing} mm -- lower --spacing or raise --root-diameter")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="phantom.nii.gz", help="Mask to write (nifti)")
    parser.add_argument("--segments-csv", help="Ground truth segment table to write")
    parser.add_argument("--orders", type=int, default=8, help="Number of Strahler orders. Default: 8")
    parser.add_argument("--root-diameter", type=float, default=20.0,
                        help="Diameter (mm) of the trunk. Default: 20")
    parser.add_argument("--root-length", type=float, default=40.0,
                        help="Length (mm) of the trunk. Default: 40")
    parser.add_argument("--rd", type=float, default=1.50, help="Imposed diameter ratio. Default: 1.50")
    parser.add_argument("--rl", type=float, default=1.40, help="Imposed length ratio. Default: 1.40")
    parser.add_argument("--angle", type=float, default=70.0,
                        help="Full bifurcation angle in degrees. Default: 70")
    parser.add_argument("--jitter", type=float, default=0.0,
                        help="Lognormal spread applied to every diameter and length. 0 gives an "
                             "exact tree. Default: 0")
    parser.add_argument("--spacing", type=float, default=1.5,
                        help="Isotropic voxel size (mm) of the raster. Default: 1.5")
    parser.add_argument("--blur", type=float, default=None,
                        help="Gaussian blur sigma (mm) applied before thresholding. "
                             "Default: two thirds of a voxel")
    parser.add_argument("--noise", type=float, default=0.05,
                        help="SD of the noise added to the blurred occupancy. Default: 0.05")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Occupancy at which a voxel becomes vessel. Default: 0.5")
    parser.add_argument("--margin", type=float, default=4.0, help="Empty margin (mm). Default: 4")
    parser.add_argument("--seed", type=int, default=0, help="Random seed. Default: 0")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    blur = (2.0 / 3.0) * args.spacing if args.blur is None else args.blur

    segments = build_tree(args.orders, args.root_diameter, args.root_length, args.rd, args.rl,
                          np.radians(0.5 * args.angle), args.jitter, rng)
    print(f"tree: {len(segments)} segments, {sum(1 for s in segments if s['strahler'] == 1)} terminals, "
          f"{args.orders} orders")

    volume, origin = rasterize(segments, args.spacing, args.margin)
    print(f"raster: shape={volume.shape} at {args.spacing} mm")
    mask = degrade(volume, args.spacing, blur, args.noise, args.threshold, rng)
    print(f"mask: {int(mask.sum())} voxels ({mask.sum() * args.spacing ** 3 / 1000.0:.2f} mL), "
          f"blur sigma {blur:.2f} mm, noise {args.noise}")

    affine = np.eye(4)
    affine[:3, :3] = np.diag([args.spacing] * 3)
    affine[:3, 3] = origin
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), args.output)
    print(f"wrote {args.output}")
    if args.segments_csv:
        write_segments_csv(args.segments_csv, segments)
        print(f"wrote {args.segments_csv}")

    print_truth(segments, args.orders, args.spacing, args.rd, args.rl)


if __name__ == "__main__":
    main()
