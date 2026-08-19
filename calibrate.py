# -*- coding: utf-8 -*-
"""
calibrate.py

Runs the whole chain on phantoms whose ratios are known, over a grid of
imposed R_d and voxel sizes, and reports what comes back.

The question it answers is the only one an in-vivo measurement cannot: of a
measured R_d of 1.52, how much is the tree and how much is the chain? Two
runs on the same subject that differ by a few percent of length give two
different R_d, and nothing in either of them says which is closer to the
truth. Here the truth is imposed, so the difference is the bias, and the bias
is a function of the voxel size -- which is why the sweep is two-dimensional
and not a single number.

Read it backwards. The forward table gives recovered(imposed); what is wanted
is imposed(recovered), and --measured does that inversion by linear
interpolation on the forward curve at each spacing. That is the reading which
turns a measured 1.52 into a statement about the tree.

R_b is deliberately absent. The phantom is a symmetric binary tree, so its
R_b is 2 by construction and calibrating against it would measure nothing.

Usage:
    python calibrate.py --rd 1.30 1.45 1.56 1.70 1.85 --spacing 0.80 1.05 1.31 \
        --measured-rd 1.517 1.430 --out calibration.csv
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np

import phantom

RATIOS = ("R_d", "R_l")


def run_case(rd, rl, spacing, args, workdir):
    """
    Builds one phantom, runs centerline.py on it, returns the fitted ratios.

    The fit range is left to centerline.py's own three-voxel diameter floor
    rather than passed as --fit-orders, and that is not a shortcut. The two
    sides do not number their orders alike: the phantom knows its imposed
    Strahler order, the chain numbers what it can actually see, and when the
    thinnest orders are blurred away the chain's order 1 is the phantom's
    order 2 or 3. Passing imposed numbers through --fit-orders therefore
    selects the wrong subset, silently, and worse as the voxels grow -- which
    is precisely the axis being measured. The diameter floor is the same
    mechanical criterion expressed in the numbering that survives, so it is
    fixed in advance in the sense --prespecified means, and it is comparable
    across spacings. `usable` is still computed, to report what the phantom
    expected the chain to reach and to skip cases too coarse to fit at all.
    """
    rng = np.random.default_rng(args.seed)
    root_diameter = args.root_diameter
    if args.pin_smallest:
        # hold the bottom of the tree at a fixed number of voxels so every
        # phantom in the sweep offers the chain the same measurable span
        root_diameter = args.pin_smallest * spacing * rd ** (args.orders - 1)
    segments = phantom.build_tree(args.orders, root_diameter, args.root_length,
                                  rd, rl, args.angle, args.jitter, rng)
    volume, origin = phantom.rasterize(segments, spacing, args.margin)
    mask = phantom.degrade(volume, spacing, phantom.default_blur(spacing, args.blur),
                           args.noise, args.threshold, rng)
    usable = phantom.usable_orders(segments, args.orders, spacing)
    if len(usable) < 3:
        return None, usable

    mask_path = os.path.join(workdir, f"ph_{rd:.3f}_{spacing:.3f}.nii.gz")
    ratios_path = os.path.join(workdir, f"ra_{rd:.3f}_{spacing:.3f}.csv")
    phantom.write_mask(mask, spacing, origin, mask_path)

    command = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "centerline.py"),
               "--input", mask_path, "--ordering", "strahler_dd", "--no-report",
               "--prespecified",
               "--ratios-csv", ratios_path, "--output", os.path.join(workdir, "cl.nii.gz")]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0 or not os.path.exists(ratios_path):
        print(f"  centerline.py failed at R_d={rd}, spacing={spacing}:\n{done.stderr[-500:]}")
        return None, usable

    with open(ratios_path) as handle:
        rows = [row for row in csv.DictReader(handle) if row["counting"] == args.counting]
    return {row["ratio"]: row for row in rows}, usable


def relative(value, truth):
    """Relative bias, or None when either side is missing."""
    if value in (None, "") or truth <= 0:
        return None
    return float(value) / truth - 1.0


def invert(curve, measured):
    """
    Reads the forward curve backwards: which imposed value yields `measured`.

    Linear interpolation between the two bracketing points, and None outside
    the range covered -- extrapolating a bias curve past the values it was
    computed at is exactly the move this file exists to avoid.
    """
    usable = [(imposed, got) for imposed, got in curve if got is not None]
    if len(usable) < 2:
        return None
    usable.sort()
    recovered = [got for _, got in usable]
    if any(b <= a for a, b in zip(recovered, recovered[1:])):
        return "not monotonic"
    if measured < recovered[0] or measured > recovered[-1]:
        return None
    return float(np.interp(measured, recovered, [imposed for imposed, _ in usable]))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rd", type=float, nargs="+", default=[1.30, 1.45, 1.56, 1.70, 1.85],
                        help="Imposed diameter ratios to sweep")
    parser.add_argument("--rl", type=float, nargs="+", default=[1.15, 1.30, 1.49, 1.65, 1.80],
                        help="Imposed length ratios to sweep. This one has to be a sweep too: a "
                             "single value gives a degenerate curve that cannot be inverted")
    parser.add_argument("--rd-ref", type=float, default=None,
                        help="R_d held fixed while R_l is swept. Default: the median of --rd")
    parser.add_argument("--rl-ref", type=float, default=None,
                        help="R_l held fixed while R_d is swept. Default: the median of --rl")
    parser.add_argument("--pin-smallest", type=float, default=None, metavar="VOXELS",
                        help="Scale the trunk so the SMALLEST order sits at this many voxels, "
                             "instead of fixing --root-diameter. Without it a larger imposed R_d "
                             "spans a wider diameter range over the same orders, so fewer of them "
                             "clear the resolution floor -- and the bias then varies with the "
                             "number of usable orders as much as with the ratio. Costs volume: "
                             "the trunk grows as R_d^(orders-1)")
    parser.add_argument("--spacing", type=float, nargs="+", default=[0.80, 1.05, 1.31],
                        help="Isotropic voxel sizes to sweep. Bracket the anisotropy of the study: "
                             "the in-plane size and the slice thickness")
    parser.add_argument("--measured-rd", type=float, nargs="*", default=[],
                        help="Measured R_d values to read back off the curve")
    parser.add_argument("--measured-rl", type=float, nargs="*", default=[],
                        help="Measured R_l values to read back off the curve")
    parser.add_argument("--orders", type=int, default=7, help="Strahler orders. Default: 7")
    parser.add_argument("--root-diameter", type=float, default=20.0,
                        help="Trunk diameter (mm). Match the study's trunk. Default: 20")
    parser.add_argument("--root-length", type=float, default=40.0, help="Trunk length (mm). Default: 40")
    parser.add_argument("--angle", type=float, default=70.0, help="Bifurcation half-angle. Default: 70")
    parser.add_argument("--jitter", type=float, default=0.0,
                        help="Lognormal scatter on diameters and lengths. Run at 0 first: on an "
                             "exact tree the ratios must come back to the third decimal, and if "
                             "they do not, nothing downstream is worth reading. Default: 0")
    parser.add_argument("--blur", type=float, default=None, help="Blur sigma (mm). Default: 2/3 voxel")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise level. Default: 0.05")
    parser.add_argument("--threshold", type=float, default=0.5, help="Rebinarization. Default: 0.5")
    parser.add_argument("--margin", type=float, default=4.0, help="Empty margin (mm). Default: 4")
    parser.add_argument("--counting", choices=("segment", "element"), default="element",
                        help="Which counting the recovered ratios are read from. On a symmetric "
                             "tree the two coincide by construction. Default: element")
    parser.add_argument("--seed", type=int, default=0, help="Random seed. Default: 0")
    parser.add_argument("--out", help="CSV to write, one row per (imposed R_d, spacing, ratio)")
    parser.add_argument("--keep", help="Directory to keep the phantoms and per-case CSVs in")
    args = parser.parse_args()

    workdir = args.keep or tempfile.mkdtemp(prefix="calibrate_")
    os.makedirs(workdir, exist_ok=True)
    print(f"working in {workdir}")
    print(f"{len(args.rd)} imposed R_d x {len(args.spacing)} spacings = "
          f"{len(args.rd) * len(args.spacing)} chain runs, jitter {args.jitter}")

    # One arm per ratio. A curve can only be read backwards along the axis
    # that was actually varied: sweeping R_d while R_l stays at one value
    # gives an R_l "curve" whose x is constant, which inverts to nothing. So
    # R_d is swept at a fixed R_l, R_l at a fixed R_d, and each ratio is
    # inverted on its own arm only. The cross product would answer both at
    # once and costs the product of the two sweeps; two arms cost the sum.
    rd_ref = args.rd_ref if args.rd_ref is not None else sorted(args.rd)[len(args.rd) // 2]
    rl_ref = args.rl_ref if args.rl_ref is not None else sorted(args.rl)[len(args.rl) // 2]
    plan = ([("R_d", rd, rl_ref) for rd in args.rd] +
            [("R_l", rd_ref, rl) for rl in args.rl])
    print(f"  R_d arm: {len(args.rd)} value(s) at R_l = {rl_ref}")
    print(f"  R_l arm: {len(args.rl)} value(s) at R_d = {rd_ref}")

    results = []
    for spacing in args.spacing:
        for arm, rd, rl in plan:
            fits, usable = run_case(rd, rl, spacing, args, workdir)
            truth = {"R_d": rd, "R_l": rl}
            for name in [arm]:
                row = fits.get(name) if fits else None
                value = float(row["value"]) if row and row["value"] else None
                results.append({
                    "ratio": name, "spacing_mm": spacing, "imposed": truth[name],
                    "held_rd": rd, "held_rl": rl, "recovered": value,
                    "ci_low": float(row["ci_low"]) if row and row["ci_low"] else None,
                    "ci_high": float(row["ci_high"]) if row and row["ci_high"] else None,
                    "r2": float(row["r2"]) if row and row["r2"] else None,
                    "bias": relative(value, truth[name]),
                    "bias_low": relative(row["ci_low"] if row else None, truth[name]),
                    "bias_high": relative(row["ci_high"] if row else None, truth[name]),
                    "order_min": int(row["order_min"]) if row and row["order_min"] else None,
                    "order_max": int(row["order_max"]) if row and row["order_max"] else None,
                    "n_orders": int(row["n_orders"]) if row and row["n_orders"] else 0,
                    "orders_expected": len(usable),
                })
            print(f"  spacing {spacing:.2f} mm, {arm} arm, R_d {rd:.2f} R_l {rl:.2f}: "
                  f"orders {usable} -> {arm}={results[-1]['recovered']}")

    for name in RATIOS:
        print(f"\n=== {name}: imposed vs recovered ===")
        print("  spacing   imposed   recovered   relative bias   95% CI on the bias      R2   orders")
        for row in [r for r in results if r["ratio"] == name]:
            if row["recovered"] is None:
                print(f"  {row['spacing_mm']:7.2f} {row['imposed']:9.3f}   not measurable "
                      f"({row['orders_expected']} order(s) expected to resolve)")
                continue
            print(f"  {row['spacing_mm']:7.2f} {row['imposed']:9.3f} {row['recovered']:11.3f} "
                  f"{row['bias']:+15.1%}   [{row['bias_low']:+6.1%}, {row['bias_high']:+6.1%}] "
                  f"{row['r2']:7.3f}   {row['order_min']}..{row['order_max']}")
        for spacing in args.spacing:
            counts = {r["n_orders"] for r in results
                      if r["ratio"] == name and r["spacing_mm"] == spacing and r["recovered"]}
            if len(counts) > 1:
                print(f"  WARNING at {spacing:.2f} mm: the fit rests on {sorted(counts)} orders "
                      f"across this sweep, not the same number every time. Part of what the bias "
                      f"column shows is that changing count, not the ratio -- a larger imposed "
                      f"ratio pushes more orders under the resolution floor. Re-run with "
                      f"--pin-smallest 3 to hold the measurable span fixed and separate the two")

    measured = {"R_d": args.measured_rd, "R_l": args.measured_rl}
    for name in RATIOS:
        if not measured[name]:
            continue
        print(f"\n=== {name}: reading the curve backwards ===")
        for spacing in args.spacing:
            curve = [(r["imposed"], r["recovered"]) for r in results
                     if r["ratio"] == name and r["spacing_mm"] == spacing]
            for value in measured[name]:
                back = invert(curve, value)
                if back == "not monotonic":
                    print(f"  spacing {spacing:.2f} mm: the forward curve does not increase with "
                          f"the imposed value, so it cannot be read backwards. That is itself the "
                          f"finding -- at this voxel size the chain does not order these trees "
                          f"correctly, and no measured {name} can be attributed")
                    break
                if back is None:
                    print(f"  spacing {spacing:.2f} mm: measured {value:.3f} falls outside the "
                          f"range these phantoms cover -- widen --rd")
                else:
                    print(f"  spacing {spacing:.2f} mm: a measured {value:.3f} is what an imposed "
                          f"{back:.3f} produces through this chain")
        print("  the spread across spacings is the resolution dependence: read the study's own "
              "voxel size off it, and remember an anisotropic grid sits between its in-plane "
              "size and its slice thickness rather than at either")

    if args.out:
        columns = ("ratio", "spacing_mm", "imposed", "held_rd", "held_rl", "recovered",
                   "ci_low", "ci_high", "r2", "bias", "bias_low", "bias_high",
                   "order_min", "order_max", "n_orders", "orders_expected")
        with open(args.out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
