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

Run it with the blur left at its default. Turning the blur off looks like the
cleaner experiment and is the opposite: the raster carries partial coverage,
and thresholding that with no blur leaves a staircase surface harder than any
acquisition produces. The EDT reads the staircase, and one voxel of it is a
larger share of a thin vessel than of a thick one -- which is a bias on R_d,
not a neutral simplification.

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
    root_diameter, root_length = args.root_diameter, args.root_length
    if args.pin_smallest:
        # Hold the bottom of the tree at a fixed number of voxels so every
        # phantom in the sweep offers the chain the same measurable span.
        # The length has to be pinned with it: pinning the diameter alone
        # scales the trunk as R_d^(orders-1) while its length stays put, and
        # past R_d ~ 1.7 the trunk comes out wider than it is long. That is a
        # disc, its daughters weld into it, and the fit that follows measures
        # the weld. Both ends of every segment are pinned to the grid here,
        # which is the only way the sweep varies the ratio and nothing else.
        root_diameter = args.pin_smallest * spacing * rd ** (args.orders - 1)
        root_length = args.pin_length * spacing * rl ** (args.orders - 1)
    segments = phantom.build_tree(args.orders, root_diameter, root_length,
                                  rd, rl, args.angle, args.jitter, rng)
    volume, origin = phantom.rasterize(segments, spacing, args.margin)
    mask = phantom.degrade(volume, spacing, phantom.default_blur(spacing, args.blur),
                           args.noise, args.threshold, rng)
    usable = phantom.usable_orders(segments, args.orders, spacing)
    if len(usable) < 3:
        return None, usable

    # every varied parameter goes in the name. Keying on (R_d, spacing) alone
    # collides across the two arms -- the R_l arm holds R_d fixed, so all of
    # its cases share one file -- and nothing downstream catches a collision:
    # it simply reads a neighbour's numbers as if they were this case's
    stem = f"{rd:.3f}_{rl:.3f}_{spacing:.3f}"
    mask_path = os.path.join(workdir, f"ph_{stem}.nii.gz")
    ratios_path = os.path.join(workdir, f"ra_{stem}.csv")
    # and a failed run must never be able to read the previous run's file
    if os.path.exists(ratios_path):
        os.remove(ratios_path)
    phantom.write_mask(mask, spacing, origin, mask_path)

    command = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "centerline.py"),
               "--input", mask_path, "--ordering", "strahler_dd", "--no-report",
               "--prespecified",
               "--ratios-csv", ratios_path,
               "--output", os.path.join(workdir, f"cl_{stem}.nii.gz")]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0 or not os.path.exists(ratios_path):
        print(f"  centerline.py failed at R_d={rd}, R_l={rl}, spacing={spacing}:"
              f"\n{done.stderr[-500:]}")
        return None, usable

    with open(ratios_path) as handle:
        rows = [row for row in csv.DictReader(handle) if row["counting"] == args.counting]
    return {row["ratio"]: row for row in rows}, usable


def keep_consistent(rows):
    """
    Keeps only the cases whose fit rests on the same number of orders.

    A curve is read point against point, so its points have to be the same
    kind of measurement. A fit on four orders and a fit on five differ by
    more than precision -- the four-order one is missing the thin end, which
    is where the slope is anchored -- and mixing them puts a step in the
    curve that has nothing to do with the ratio.

    What matters is that the kept cases agree with EACH OTHER, not that they
    agree with what the phantom predicted would resolve. Requiring the
    latter is too strict: adding realistic scatter shifts every case down by
    one order at once, which leaves the curve perfectly usable and internally
    consistent while failing a comparison against the noiseless prediction.
    So the modal count wins, and the minority is dropped.
    """
    counts = [row["n_orders"] for row in rows if row["reliable"]]
    if not counts:
        return
    modal = max(set(counts), key=counts.count)
    for row in rows:
        if row["reliable"] and row["n_orders"] != modal:
            row["reliable"] = False
            row["dropped_for"] = f"{row['n_orders']} orders against {modal} for the arm"


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
                        help="R_d held fixed while R_l is swept. Pick a value that makes a "
                             "well-formed tree: the arm is measured on it. Default: 1.56")
    parser.add_argument("--rl-ref", type=float, default=None,
                        help="R_l held fixed while R_d is swept. A low value makes a stubby tree "
                             "whose trunk is barely longer than it is wide, and the fit on it "
                             "loses an order. Default: 1.49")
    parser.add_argument("--pin-smallest", type=float, default=None, metavar="VOXELS",
                        help="Scale the trunk so the SMALLEST order sits at this many voxels, "
                             "instead of fixing --root-diameter. Without it a larger imposed R_d "
                             "spans a wider diameter range over the same orders, so fewer of them "
                             "clear the resolution floor -- and the bias then varies with the "
                             "number of usable orders as much as with the ratio. Costs volume: "
                             "the trunk grows as R_d^(orders-1)")
    parser.add_argument("--pin-length", type=float, default=None, metavar="VOXELS",
                        help="Smallest segment LENGTH, in voxels, when --pin-smallest is used. "
                             "Pinning the diameter alone lets the trunk outgrow its own length "
                             "and turns it into a disc. Default: 4x --pin-smallest, which puts "
                             "the tip at the length-to-diameter ratio of a real distal vessel")
    parser.add_argument("--min-r2", type=float, default=0.90,
                        help="A fit below this R2 is reported but kept out of the curve. A "
                             "straight line through a phantom whose branches welded is not a "
                             "measurement, and letting it into the inversion breaks the curve "
                             "with a number that describes the weld. Default: 0.90")
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

    if args.blur == 0:
        print("  NOTE: --blur 0 does not isolate the chain, it hardens the phantom. The raster "
              "carries partial coverage; thresholding it with no blur leaves a staircase surface "
              "harder than any acquisition produces, the EDT reads that staircase, and the error "
              "weighs more on a thin vessel than on a thick one -- which compresses R_d. Measured "
              "here it costs about a point of bias at R_d 1.56 and two at 1.75. Keep the default "
              "blur for any number meant to describe a real image")
    if args.pin_smallest and args.pin_length is None:
        args.pin_length = 4.0 * args.pin_smallest
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
    # The held-fixed value is not a detail of bookkeeping: each arm measures
    # its ratio ON a tree whose other ratio is that value, so a poor choice
    # calibrates the degeneracy instead of the ratio. Taking the median of
    # whatever list was swept is the wrong default -- sweeping R_l down to
    # 1.15 drags the held R_l to 1.30, whose trunk is only twice as long as
    # it is wide, and the fit on those trees loses an order and swings by
    # ten percent. Both defaults are therefore anatomical reference values,
    # independent of what is being swept.
    rd_ref = args.rd_ref if args.rd_ref is not None else 1.56
    rl_ref = args.rl_ref if args.rl_ref is not None else 1.49
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
                    # R2 gates the individual fit; the order count is settled
                    # afterwards, across the arm (see `keep_consistent`)
                    "reliable": bool(row and row["r2"] and float(row["r2"]) >= args.min_r2),
                    "dropped_for": "",
                })
            print(f"  spacing {spacing:.2f} mm, {arm} arm, R_d {rd:.2f} R_l {rl:.2f}: "
                  f"orders {usable} -> {arm}={results[-1]['recovered']}")

    for name in RATIOS:
        for spacing in args.spacing:
            keep_consistent([r for r in results
                             if r["ratio"] == name and r["spacing_mm"] == spacing])
        print(f"\n=== {name}: imposed vs recovered ===")
        print("  spacing   imposed   recovered   relative bias   95% CI on the bias      R2   orders")
        for row in [r for r in results if r["ratio"] == name]:
            if row["recovered"] is None:
                print(f"  {row['spacing_mm']:7.2f} {row['imposed']:9.3f}   not measurable "
                      f"({row['orders_expected']} order(s) expected to resolve)")
                continue
            if row["reliable"]:
                flag = ""
            elif row["r2"] < args.min_r2:
                flag = f"   <- R2 under {args.min_r2}, kept out of the curve"
            else:
                flag = f"   <- {row.get('dropped_for', 'inconsistent')}, kept out of the curve"
            print(f"  {row['spacing_mm']:7.2f} {row['imposed']:9.3f} {row['recovered']:11.3f} "
                  f"{row['bias']:+15.1%}   [{row['bias_low']:+6.1%}, {row['bias_high']:+6.1%}] "
                  f"{row['r2']:7.3f}   {row['order_min']}..{row['order_max']}{flag}")
        for spacing in args.spacing:
            here = [r for r in results if r["ratio"] == name and r["spacing_mm"] == spacing]
            kept = [r for r in here if r["reliable"]]
            counts = {r["n_orders"] for r in kept}
            print(f"  at {spacing:.2f} mm: {len(kept)}/{len(here)} case(s) kept, all on "
                  f"{sorted(counts) if counts else 'no'} order(s)")
            if len(counts) > 1:
                print(f"  WARNING: the kept cases still do not rest on the same number of orders. "
                      f"Part of what the bias column shows is that changing count rather than the "
                      f"ratio, and the curve is not comparable point to point")

    measured = {"R_d": args.measured_rd, "R_l": args.measured_rl}
    for name in RATIOS:
        if not measured[name]:
            continue
        print(f"\n=== {name}: reading the curve backwards ===")
        for spacing in args.spacing:
            curve = [(r["imposed"], r["recovered"]) for r in results
                     if r["ratio"] == name and r["spacing_mm"] == spacing and r["reliable"]]
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
                   "order_min", "order_max", "n_orders", "orders_expected", "reliable",
                   "dropped_for")
        with open(args.out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
