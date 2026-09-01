# -*- coding: utf-8 -*-
"""
calibrate.py

Runs the whole chain on phantoms whose ratios are known, over a grid of
imposed ratios and voxel sizes, and reports what comes back.

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

Three things decide whether that reading means anything.

  The GRID. `--spacing` takes triples: "1.25,0.799,1.25" is one spacing, not
  three. Sweeping isotropic sizes around the study's anisotropy is not the
  same experiment as reproducing it -- the chain upsamples an anisotropic
  acquisition to its finest axis, and an isotropic phantom rasterized at that
  finest axis has information the study never had. Give the acquired size.

  The ASYMMETRY. A symmetric binary tree has R_b = 2 by construction, so
  running it measures nothing about R_b, and 2 is the floor: what is worth
  knowing in a real tree is the excess above it. `--side-branches` sweeps
  that excess, and it is the R_b arm's x axis the way `--rd` is the R_d
  arm's. Without it R_b is absent from the output, as it was before.

  The REPEATS. One phantom gives one fit, and the 95% interval of a
  regression on five orders is 4 or 5 percent wide -- wider than the bias
  being measured, so a single case can only ever say the bias is not
  distinguishable from zero. `--repeats` runs each case again under a fresh
  draw, and the interval then reported is the one on the mean bias, which
  narrows as 1/sqrt(n) and is the right uncertainty for a systematic effect.
  Run it with --jitter as well: repeats over noise alone hold the tree fixed
  and understate the spread. Each repeat is then compared with the truth of
  the tree IT was run on, never with a truth borrowed from another draw --
  see `paired_bias`, which is what makes the jittered arm readable at all.

Run it with the blur left at its default. Turning the blur off looks like the
cleaner experiment and is the opposite: the raster carries partial coverage,
and thresholding that with no blur leaves a staircase surface harder than any
acquisition produces. The EDT reads the staircase, and one voxel of it is a
larger share of a thin vessel than of a thick one -- which is a bias on R_d,
not a neutral simplification.

Usage:
    python -m analysis.calibrate --rd 1.30 1.45 1.56 1.70 1.85 --spacing 0.80 1.05 1.31 \
        --measured-rd 1.517 1.430 --out calibration.csv
    python -m analysis.calibrate --spacing 1.25,0.799,1.25 --side-branches 0 1 2 3 \
        --repeats 5 --jitter 0.1 --measured-rb 2.31 --out calibration.csv
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.stats import t as student_t

from . import phantom

RATIOS = ("R_b", "R_d", "R_l")


def parse_spacing(text):
    """
    One --spacing entry to a 3-vector: "0.8" or "1.25,0.799,1.25".

    The comma form is the point of the option. A study acquires anisotropic
    voxels; bracketing them with two isotropic runs answers a question about
    two grids that do not exist rather than the one that does.
    """
    return phantom.as_triple([float(v) for v in str(text).replace("x", ",").split(",")])


def label(spacing):
    """A spacing printed the same way everywhere: "0.80" or "1.25/0.80/1.25"."""
    return (f"{spacing[0]:.2f}" if np.allclose(spacing, spacing[0])
            else "/".join(f"{v:.2f}" for v in spacing))


def run_case(rd, rl, side, spacing, seed, args, workdir, rl_pin=None):
    """
    Builds one phantom, runs centerline.py on it, returns fits and truth.

    The fit range is left to centerline.py's own diameter floor rather than
    passed as --fit-orders, and that is not a shortcut. The two sides do not
    number their orders alike: the phantom knows its imposed Strahler order,
    the chain numbers what it can actually see, and when the thinnest orders
    are blurred away the chain's order 1 is the phantom's order 2 or 3.
    Passing imposed numbers through --fit-orders therefore selects the wrong
    subset, silently, and worse as the voxels grow -- which is precisely the
    axis being measured. The diameter floor is the same mechanical criterion
    expressed in the numbering that survives, so it is fixed in advance in
    the sense --prespecified means, and it is comparable across spacings.

    That floor is handed over in millimetres of the ACQUIRED grid, not left
    to be recomputed in voxels of the resampled one. Upsampling an
    anisotropic mask to its finest axis multiplies the voxels without adding
    information, and a floor counted there admits an order the acquisition
    never resolved -- an order censored from below, sitting at the smallest
    radius the transform can return, anchoring the steep end of every slope.
    On a 1.6:1 grid that one admitted order moves R_d by several percent,
    which would be charged here to the voxel size rather than to the range.

    The truth is refitted per case rather than read off `rd` and `rl`: on an
    asymmetric tree R_b is a regression slope over finitely many orders and
    has no closed form, and segment-counted R_l is not `rl` either. Both
    sides are fitted by the same estimator over the same orders, so what is
    left between them is the chain.
    """
    rng = np.random.default_rng(seed)
    root_diameter, root_length = args.root_diameter, args.root_length
    if args.pin_smallest:
        # Hold the bottom of the tree at a fixed number of voxels so every
        # phantom in the sweep offers the chain the same measurable span.
        # Voxels of the COARSE axis: that is the one that decides whether an
        # order is resolved, and pinning against the fine axis on a 1.6:1
        # grid pins the tree an order and a half above where it is readable.
        # The length has to be pinned with it: pinning the diameter alone
        # scales the trunk as R_d^(orders-1) while its length stays put, and
        # past R_d ~ 1.7 the trunk comes out wider than it is long. That is a
        # disc, its daughters weld into it, and the fit that follows measures
        # the weld.
        #
        # But the length is pinned against the R_l the ARM holds fixed, never
        # against the R_l being swept. Both ends of a geometric series cannot
        # be held while its ratio varies, and pinning the small end is the
        # wrong choice of the two: it scales the trunk as R_l^(orders-1), so
        # the R_l arm walks the trunk from L/D 0.8 at R_l 1.15 to 4.6 at 1.80
        # and welds the stubby end into the same disc. Pinning the trunk
        # instead leaves the smallest segment between 4 and 25 voxels over
        # that range, all of them measurable, and the trunk identical in
        # every case -- which is what "the sweep varies the ratio and nothing
        # else" has to mean.
        coarse = float(spacing.max())
        root_diameter = args.pin_smallest * coarse * rd ** (args.orders - 1)
        root_length = args.pin_length * coarse * (rl if rl_pin is None else rl_pin) ** (args.orders - 1)
    segments = phantom.build_tree(args.orders, root_diameter, root_length, rd, rl,
                                  np.radians(0.5 * args.angle), args.jitter, rng,
                                  side, args.side_drop)
    volume, origin = phantom.rasterize(segments, spacing, args.margin)
    mask = phantom.degrade(volume, spacing, phantom.default_blur(spacing, args.blur),
                           args.noise, args.threshold, rng)
    usable = phantom.usable_orders(segments, args.orders, spacing)
    if len(usable) < 3:
        return None, usable, {}
    truth = phantom.truth_ratios(segments, args.counting, (min(usable), max(usable)))

    # every varied parameter goes in the name. Keying on (R_d, spacing) alone
    # collides across the arms -- the R_l arm holds R_d fixed, so all of its
    # cases share one file -- and nothing downstream catches a collision: it
    # simply reads a neighbour's numbers as if they were this case's
    stem = f"{rd:.3f}_{rl:.3f}_s{side}_{'-'.join(f'{v:.3f}' for v in spacing)}_{seed}"
    mask_path = os.path.join(workdir, f"ph_{stem}.nii.gz")
    ratios_path = os.path.join(workdir, f"ra_{stem}.csv")
    # and a failed run must never be able to read the previous run's file
    if os.path.exists(ratios_path):
        os.remove(ratios_path)
    phantom.write_mask(mask, spacing, origin, mask_path)

    # handed over as a plain count: centerline.py counts it on the coarse axis
    # of the acquired grid, which is the rule this file was converting by hand
    # before. One rule, one implementation -- the phantom and the real data
    # have to be floored the same way or they stop being comparable
    command = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "centerline.py"),
               "--input", mask_path, "--ordering", "strahler_dd", "--no-report",
               "--prespecified", "--fit-min-voxels", f"{args.fit_min_voxels:.4f}",
               "--ratios-csv", ratios_path,
               "--output", os.path.join(workdir, f"cl_{stem}.nii.gz")]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0 or not os.path.exists(ratios_path):
        print(f"  centerline.py failed at R_d={rd}, R_l={rl}, side={side}, "
              f"spacing={label(spacing)}:\n{done.stderr[-500:]}")
        return None, usable, truth

    with open(ratios_path) as handle:
        rows = [row for row in csv.DictReader(handle) if row["counting"] == args.counting]
    return {row["ratio"]: row for row in rows}, usable, truth


def aggregate(repeats):
    """
    Pools a per-repeat quantity into a mean and an interval on that mean.

    This is the interval that answers "is the correction real". The one
    centerline.py reports is the confidence interval of a regression through
    five or six points, which is 4 or 5 percent wide however many times the
    experiment is run: it describes the scatter of the orders about the line,
    not the reproducibility of the chain. Repeating the case under fresh
    noise gives independent draws of the same systematic quantity, and the
    standard error of their mean is the uncertainty of the bias itself. With
    one repeat there is no such interval and the field stays empty -- an
    honest blank rather than the regression interval wearing the wrong hat.
    """
    values = [v for v in repeats if v is not None]
    if not values:
        return {"mean": None, "sd": None, "se": None, "low": None, "high": None, "n": 0}
    mean = float(np.mean(values))
    if len(values) < 2:
        return {"mean": mean, "sd": None, "se": None, "low": None, "high": None, "n": 1}
    sd = float(np.std(values, ddof=1))
    se = sd / np.sqrt(len(values))
    half = float(student_t.ppf(0.975, len(values) - 1)) * se
    return {"mean": mean, "sd": sd, "se": se, "low": mean - half, "high": mean + half,
            "n": len(values)}


def paired_bias(recovered, truths):
    """
    The bias of each repeat against the truth of the tree THAT repeat was run
    on, pooled.

    With --jitter the imposed tree is redrawn every repeat, and its own
    fitted ratio moves with it: over ten draws that ratio is unbiased, but a
    single draw sits about a percent off, and its per-order means rest on 16,
    8, 4, 2, 1 segments so the scatter is not small. Dividing a mean over ten
    chain runs by the truth of one of those ten therefore reports that draw's
    offset as a bias of the chain -- and reports the SAME offset at every
    point of an arm, because every case is jittered from the same seed
    sequence, which dresses it up as a consistent trend.

    Pairing removes it exactly: each measurement is compared with the tree it
    measured, and what is left between them is the chain and nothing else.
    It also removes the tree-to-tree variance from the interval, which is the
    larger of the two terms here.
    """
    ratios = [r / t - 1.0 for r, t in zip(recovered, truths)
              if r is not None and t not in (None, 0)]
    return aggregate(ratios)


def fit_spread(row):
    """
    Relative half-width of the per-case fit interval, or None.

    The 95% interval centerline.py returns on a ratio, expressed as a factor
    either side of it: sqrt(high/low) - 1. Scale-free, and unlike R2 it does
    not depend on how steep the truth is.
    """
    low, high = row.get("fit_ci_low"), row.get("fit_ci_high")
    if not low or not high or low <= 0:
        return None
    return float(np.sqrt(high / low)) - 1.0


def gate(row, args):
    """
    Whether a case may enter the curve the inversion is read off.

    Nothing is excluded for being IMPRECISE, and both of the criteria that
    were once used for it are off by default.

    R2 first, because it is the clearer error. R2 is the share of the
    variance a straight line explains, and an arm that sweeps its ratio
    towards 1 sweeps the variance to be explained towards zero with it: a
    flat truth measured perfectly scores near zero. Measured here, the R_l
    case imposed at 1.05 scores R2 0.57 and carries the tightest interval of
    its whole arm, while the case at 1.80 scores 0.97 and carries the widest.
    Gating on R2 removes the low end of that arm systematically -- the end a
    measured R_l near 1.2 has to be bracketed by -- and keeps the loose end.

    The width of the per-case fit interval is scale-free where R2 is not, so
    it survives that objection, and it fails a second one. It describes how
    uncertain ONE realization's regression was, and `--repeats` exists to
    average exactly that away: what is left after the repeats is the paired
    interval, which is the precision of the quantity actually being
    estimated. Measured on the R_b arm at 30 repeats, the case at 2.865
    carries a paired interval 5.2 points wide against 5.2 for its neighbour
    at 3.080 and 5.6 for the one at 2.624, while their per-fit spreads read
    38%, 29% and 24%. A threshold on the spread separates cases the repeats
    have made indistinguishable.

    And excluding an imprecise point does not make the reading more precise.
    The band already carries every point's uncertainty through the envelope;
    dropping one only forces the interpolation across a wider gap. Dropping
    2.865 widened the bracket around a measured R_b of 2.782 from 0.19 to
    0.33 in recovered units -- the exclusion doubled the uncertainty of the
    reading at the one place it mattered.

    What is left here is the case that is not a measurement at all, and
    wrongness is caught downstream where it shows: `keep_consistent` for a
    fit resting on a different number of orders, and `invert` for a curve or
    an envelope that stops increasing. `--max-fit-spread` remains for a sweep
    run without repeats, where the per-case interval is the only precision
    there is, and `--min-r2` for an arm whose slope is nowhere near zero.

    Recomputed from the stored columns rather than read back from `reliable`,
    so a saved sweep can be re-gated with --from-csv instead of re-run: the
    threshold is a reading decision, and re-running to change it would move
    the curve as well.
    """
    if row["recovered"] is None or not row["truth"]:
        return False, "no fit"
    spread = fit_spread(row)
    if args.max_fit_spread and spread is not None and spread > args.max_fit_spread:
        return False, f"fit interval +-{spread:.0%}, over {args.max_fit_spread:.0%}"
    if args.min_r2 > 0 and (row["r2"] is None or row["r2"] < args.min_r2):
        return False, f"R2 under {args.min_r2}"
    return True, ""


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


def trend(points):
    """
    Least squares of the bias on the imposed value, with a t interval.

    The monotone drift of the bias across an arm is a second finding and a
    stronger one than any single point: each point's own interval may cover
    zero while the drift between them does not, because the drift is a
    comparison at fixed everything-else. It is also the property the
    inversion actually uses -- a curve that does not tilt cannot be read
    backwards at all. Returns None below three points.
    """
    usable = [(x, y) for x, y in points if y is not None]
    if len(usable) < 3:
        return None
    x = np.array([p[0] for p in usable], float)
    y = np.array([p[1] for p in usable], float)
    scatter = float(((x - x.mean()) ** 2).sum())
    if scatter <= 0:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    se = np.sqrt(float((residual ** 2).sum()) / (len(x) - 2) / scatter) if len(x) > 2 else None
    if se is None or not np.isfinite(se) or se <= 0:
        return {"slope": float(slope), "low": None, "high": None}
    half = float(student_t.ppf(0.975, len(x) - 2)) * se
    return {"slope": float(slope), "low": float(slope - half), "high": float(slope + half)}


def invert(curve, measured):
    """
    Reads the forward curve backwards: which imposed value yields `measured`.

    Linear interpolation between the two bracketing points, and None outside
    the range covered -- extrapolating a bias curve past the values it was
    computed at is exactly the move this file exists to avoid.

    `curve` is (imposed, recovered) pairs; with a third and fourth field it
    is (imposed, recovered, low, high) and the answer comes back as a band.
    The band is the whole point of the reading: the inverse of a curve known
    to a few percent is an interval of imposed values, and quoting its centre
    alone turns an interval into the point value the phantom cannot support.
    A curve read against its upper envelope gives the LOW end of the band --
    the same measurement, produced by a smaller tree.
    """
    usable = [row for row in curve if row[0] is not None and row[1] is not None]
    if len(usable) < 2:
        return None
    usable.sort(key=lambda row: row[0])
    imposed = [row[0] for row in usable]
    recovered = [row[1] for row in usable]
    if any(b <= a for a, b in zip(recovered, recovered[1:])):
        return "not monotonic"
    if measured < recovered[0] or measured > recovered[-1]:
        return None
    answer = {"value": float(np.interp(measured, recovered, imposed)),
              "low": None, "high": None, "why": {}}
    for end, index in (("low", 3), ("high", 2)):
        # one point without an envelope is enough to lose the band. Filling
        # it in from the mean curve would collapse that end onto the point
        # value and report a zero-width interval, which is the false
        # precision this whole reading exists to refuse
        if any(len(row) <= index or row[index] is None for row in usable):
            answer["why"][end] = "no interval on the mean; run with --repeats"
            continue
        bound = [row[index] for row in usable]
        # the two ways an envelope fails are not the same finding and must
        # not print the same advice: one is answered by widening the sweep,
        # the other only by narrowing the interval that crosses its neighbour
        if any(b <= a for a, b in zip(bound, bound[1:])):
            answer["why"][end] = ("that envelope of the curve does not increase point to point -- "
                                  "one case's interval overlaps its neighbour's, so add repeats "
                                  "there rather than widening the sweep")
            continue
        if measured < bound[0] or measured > bound[-1]:
            answer["why"][end] = ("that envelope does not reach this value over the imposed range "
                                  "swept; widen the sweep to close it")
            continue
        answer[end] = float(np.interp(measured, bound, imposed))
    return answer


def verdict(rows):
    """
    What the arm supports, in the register it supports it in.

    Three outcomes, and the middle one is the usual one. If no point's
    interval clears zero and the trend does not either, the arm has measured
    nothing. If the trend clears zero but the individual points do not, the
    arm has established a direction and a size without a point value, which
    is a real result and has to be written as one. Only when the points
    themselves clear zero is a per-point correction quotable.
    """
    biased = [row for row in rows if row["bias_low"] is not None
              and (row["bias_low"] > 0) == (row["bias_high"] > 0)]
    repeats = min(row["n_repeats"] for row in rows)
    drift = trend([(row["imposed"], row["bias"]) for row in rows])
    tilted = drift is not None and drift["low"] is not None and (drift["low"] > 0) == (drift["high"] > 0)
    if len(biased) == len(rows):
        return (f"every point's interval on the mean excludes zero (on {repeats} repeat(s)): the "
                f"correction is resolved point by point and can be applied as such")
    if biased:
        return (f"{len(biased)}/{len(rows)} point(s) have an interval on the mean that excludes "
                f"zero, on {repeats} repeat(s). The correction is resolved where it is and not "
                f"elsewhere; apply it only at those points, or add repeats until the arm is whole")
    if tilted:
        return ("no single point's interval excludes zero, but the bias drifts across the arm by "
                f"{drift['slope']:+.1%} per unit of imposed ratio [{drift['low']:+.1%}, "
                f"{drift['high']:+.1%}], which does. The phantom establishes the SIGN and the "
                f"ORDER OF MAGNITUDE of the correction, not a point value -- quote it that way, "
                f"and quote the band from the inversion rather than its centre")
    return ("neither the individual biases nor their drift across the arm exclude zero. At this "
            "resolution the phantom does not resolve a correction: report that the chain is "
            "unbiased to within the width of the interval, and give the width")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rd", type=float, nargs="+", default=[1.30, 1.45, 1.56, 1.70, 1.85],
                        help="Imposed diameter ratios to sweep")
    parser.add_argument("--rl", type=float, nargs="+", default=[1.15, 1.30, 1.49, 1.65, 1.80],
                        help="Imposed length ratios to sweep. This one has to be a sweep too: a "
                             "single value gives a degenerate curve that cannot be inverted")
    parser.add_argument("--side-branches", type=int, nargs="+", default=[], metavar="N",
                        help="Side branches per element to sweep, which is the R_b arm: 0 is the "
                             "symmetric tree (R_b = 2 exactly, the floor), 1 and 2 impose a known "
                             "excess above it. Without this argument R_b is not calibrated at all, "
                             "because a symmetric phantom cannot calibrate it. Try 0 1 2 3")
    parser.add_argument("--side-drop", type=int, default=2, metavar="K",
                        help="How many orders below its parent a side branch is, held fixed across "
                             "the R_b arm. Default: 2, the monopodial pattern of a lung")
    parser.add_argument("--rd-ref", type=float, default=None,
                        help="R_d held fixed while R_l and the asymmetry are swept. Pick a value "
                             "that makes a well-formed tree: the arms are measured on it. "
                             "Default: 1.56")
    parser.add_argument("--rl-ref", type=float, default=None,
                        help="R_l held fixed while R_d and the asymmetry are swept. A low value "
                             "makes a stubby tree whose trunk is barely longer than it is wide, "
                             "and the fit on it loses an order. Default: 1.49")
    parser.add_argument("--side-ref", type=int, default=0, metavar="N",
                        help="Side branches carried while R_d and R_l are swept. 0 keeps those two "
                             "arms on the symmetric tree they have always been measured on; a "
                             "non-zero value asks whether their bias survives asymmetry, which is "
                             "a different and also worthwhile question. Default: 0")
    parser.add_argument("--repeats", type=int, default=1, metavar="N",
                        help="Independent noise realizations per case. The reported interval is "
                             "then the one on the MEAN recovered value, which is the uncertainty "
                             "of a systematic bias; with 1 repeat there is no such interval and "
                             "nothing in the output can settle whether a correction is real. "
                             "Costs N chain runs per case. Default: 1")
    parser.add_argument("--pin-smallest", type=float, default=None, metavar="VOXELS",
                        help="Scale the trunk so the SMALLEST order sits at this many voxels of the "
                             "coarse axis, instead of fixing --root-diameter. Without it a larger "
                             "imposed R_d spans a wider diameter range over the same orders, so "
                             "fewer of them clear the resolution floor -- and the bias then varies "
                             "with the number of usable orders as much as with the ratio. Costs "
                             "volume: the trunk grows as R_d^(orders-1)")
    parser.add_argument("--pin-length", type=float, default=None, metavar="VOXELS",
                        help="Smallest segment LENGTH, in voxels, when --pin-smallest is used. "
                             "Pinning the diameter alone lets the trunk outgrow its own length "
                             "and turns it into a disc. Default: 4x --pin-smallest, which puts "
                             "the tip at the length-to-diameter ratio of a real distal vessel")
    parser.add_argument("--max-fit-spread", type=float, default=None, metavar="FRACTION",
                        help="Keep out of the curve any case whose per-fit 95%% interval is wider "
                             "than this, as a factor either side of the ratio. OFF by default: it "
                             "describes how uncertain one realization\'s regression was, which is "
                             "what --repeats averages away, and excluding an imprecise point does "
                             "not make the reading more precise -- the band already carries its "
                             "uncertainty, and dropping it only widens the gap the inversion has "
                             "to interpolate across. Worth setting for a sweep run WITHOUT "
                             "repeats, where it is the only precision there is")
    parser.add_argument("--min-r2", type=float, default=0.0,
                        help="R2 floor, off by default. R2 is the share of the variance a line "
                             "explains, so an arm sweeping its ratio towards 1 drives it to zero "
                             "however well the fit is done -- it cannot gate the low end of the "
                             "R_l arm, and it kept that arm\'s loosest cases while dropping its "
                             "tightest. Default: 0 (off)")
    parser.add_argument("--fit-min-voxels", type=float, default=3.0,
                        help="Diameter floor of the fit, in voxels of the coarse axis -- the axis "
                             "that decides whether an order was resolved. Passed to centerline.py "
                             "unchanged, which applies the same rule to real data. Default: 3")
    parser.add_argument("--spacing", nargs="+", default=["0.80", "1.05", "1.31"], metavar="MM",
                        help="Voxel sizes to sweep. An entry is one number for an isotropic grid "
                             "or three comma-separated for an anisotropic one, e.g. "
                             "1.25,0.799,1.25. Give the ACQUIRED size of the study, not the size "
                             "the chain resamples it to")
    parser.add_argument("--measured-rd", type=float, nargs="*", default=[],
                        help="Measured R_d values to read back off the curve")
    parser.add_argument("--measured-rl", type=float, nargs="*", default=[],
                        help="Measured R_l values to read back off the curve")
    parser.add_argument("--measured-rb", type=float, nargs="*", default=[],
                        help="Measured R_b values to read back off the curve. Needs --side-branches")
    parser.add_argument("--orders", type=int, default=7, help="Strahler orders. Default: 7")
    parser.add_argument("--root-diameter", type=float, default=20.0,
                        help="Trunk diameter (mm). Match the study's trunk. Default: 20")
    parser.add_argument("--root-length", type=float, default=40.0, help="Trunk length (mm). Default: 40")
    parser.add_argument("--angle", type=float, default=70.0, help="Full bifurcation angle. Default: 70")
    parser.add_argument("--jitter", type=float, default=0.0,
                        help="Lognormal scatter on diameters and lengths. Run at 0 first: on an "
                             "exact tree the ratios must come back to the third decimal, and if "
                             "they do not, nothing downstream is worth reading. Then run the "
                             "repeats at 0.1 to 0.2, or they vary the noise and hold the tree "
                             "fixed and the interval comes out too narrow. Default: 0")
    parser.add_argument("--blur", type=float, nargs="+", default=None, metavar="MM",
                        help="Blur sigma (mm), one value or three. Default: 2/3 voxel per axis")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise level. Default: 0.05")
    parser.add_argument("--threshold", type=float, default=0.5, help="Rebinarization. Default: 0.5")
    parser.add_argument("--margin", type=float, default=4.0, help="Empty margin (mm). Default: 4")
    parser.add_argument("--counting", choices=("segment", "element"), default="element",
                        help="Which counting the recovered ratios are read from. On a symmetric "
                             "tree the two coincide by construction; with --side-branches they do "
                             "not, and the truth is refitted for whichever is asked for. "
                             "Default: element")
    parser.add_argument("--seed", type=int, default=0, help="First random seed. Default: 0")
    parser.add_argument("--from-csv", metavar="PATH",
                        help="Read a sweep written by --out and go straight to the tables and the "
                             "backward reading, running no phantom. This is how a revised measured "
                             "ratio is put through the curve: re-sweeping to read a new value "
                             "would move the curve as well, leaving nothing to attribute the "
                             "difference to. Only --measured-*, --max-fit-spread and --min-r2 still\n"
                             "apply, and re-gating with them is exactly what it is for")
    parser.add_argument("--out", help="CSV to write, one row per (imposed value, spacing, ratio)")
    parser.add_argument("--keep", help="Directory to keep the phantoms and per-case CSVs in")
    args = parser.parse_args()

    if args.from_csv:
        results = load_results(args.from_csv)
        # the order the spacings were swept in, kept as written
        labels, seen = [], set()
        for row in results:
            if row["spacing_mm"] not in seen:
                seen.add(row["spacing_mm"])
                labels.append(row["spacing_mm"])
        print(f"read {len(results)} row(s) over {len(labels)} spacing(s) from {args.from_csv}; "
              f"no phantom is built and the curve is exactly the one that file records")
        report(results, labels, args)
        return

    if args.blur is not None and not np.any(np.asarray(args.blur) > 0):
        print("  NOTE: --blur 0 does not isolate the chain, it hardens the phantom. The raster "
              "carries partial coverage; thresholding it with no blur leaves a staircase surface "
              "harder than any acquisition produces, the EDT reads that staircase, and the error "
              "weighs more on a thin vessel than on a thick one -- which compresses R_d. Measured "
              "here it costs about a point of bias at R_d 1.56 and two at 1.75. Keep the default "
              "blur for any number meant to describe a real image")
    if args.repeats < 2:
        print("  NOTE: --repeats 1 gives no interval on the mean, so the output can show a bias "
              "but cannot show it is one. The intervals centerline.py reports are regression "
              "intervals on five or six orders and stay 4 to 5 percent wide however often the "
              "case is run -- wider than the effect. Use --repeats 5 or more, with --jitter")
    if args.pin_smallest and args.pin_length is None:
        args.pin_length = 4.0 * args.pin_smallest
    spacings = [parse_spacing(text) for text in args.spacing]
    workdir = args.keep or tempfile.mkdtemp(prefix="calibrate_")
    os.makedirs(workdir, exist_ok=True)
    print(f"working in {workdir}")

    # One arm per ratio. A curve can only be read backwards along the axis
    # that was actually varied: sweeping R_d while R_l stays at one value
    # gives an R_l "curve" whose x is constant, which inverts to nothing. So
    # R_d is swept at a fixed R_l, R_l at a fixed R_d, R_b at fixed both, and
    # each ratio is inverted on its own arm only. The cross product would
    # answer all three at once and costs the product of the sweeps; the arms
    # cost the sum. The held-fixed values are not bookkeeping: each arm
    # measures its ratio ON a tree whose other ratios are those values, so a
    # poor choice calibrates the degeneracy instead of the ratio. Taking the
    # median of whatever list was swept is the wrong default -- sweeping R_l
    # down to 1.15 drags the held R_l to 1.30, whose trunk is only twice as
    # long as it is wide, and the fit on those trees loses an order and
    # swings by ten percent. The defaults are therefore anatomical reference
    # values, independent of what is being swept.
    rd_ref = args.rd_ref if args.rd_ref is not None else 1.56
    rl_ref = args.rl_ref if args.rl_ref is not None else 1.49
    plan = ([("R_d", rd, rl_ref, args.side_ref) for rd in args.rd] +
            [("R_l", rd_ref, rl, args.side_ref) for rl in args.rl] +
            [("R_b", rd_ref, rl_ref, side) for side in args.side_branches])
    print(f"{len(plan)} case(s) x {len(spacings)} spacing(s) x {args.repeats} repeat(s) = "
          f"{len(plan) * len(spacings) * args.repeats} chain runs, jitter {args.jitter}")
    print(f"  R_d arm: {len(args.rd)} value(s) at R_l = {rl_ref}, {args.side_ref} side branch(es)")
    print(f"  R_l arm: {len(args.rl)} value(s) at R_d = {rd_ref}, {args.side_ref} side branch(es)")
    if args.side_branches:
        print(f"  R_b arm: side branches {args.side_branches} at order -{args.side_drop}, "
              f"R_d = {rd_ref}, R_l = {rl_ref}")
    else:
        print("  R_b arm: absent. A symmetric phantom has R_b = 2 by construction and calibrates "
              "nothing about it -- pass --side-branches 0 1 2 3 to impose an excess above 2")

    results = []
    for spacing in spacings:
        for arm, rd, rl, side in plan:
            values, truths, usable = [], [], []
            scores, spans, fit_ci = [], [], []
            for repeat in range(args.repeats):
                fits, usable, truth = run_case(rd, rl, side, spacing, args.seed + repeat,
                                               args, workdir, rl_pin=rl_ref)
                row = fits.get(arm) if fits else None
                values.append(float(row["value"]) if row and row["value"] else None)
                # kept per repeat, not overwritten: with --jitter the tree is
                # redrawn every time and each measurement belongs with the
                # truth of the tree it was made on (see `paired_bias`)
                truths.append(truth.get(arm))
                if not row or not row["value"]:
                    continue
                scores.append(float(row["r2"]) if row["r2"] else 0.0)
                spans.append((int(row["n_orders"]), int(row["order_min"]), int(row["order_max"])))
                fit_ci.append((float(row["ci_low"]), float(row["ci_high"])))
            pooled = aggregate(values)
            target = aggregate(truths)["mean"]
            bias = paired_bias(values, truths)
            # the repeats are one case, so their fit metadata is summarized
            # rather than taken from whichever ran last: an arm is kept or
            # dropped on what the case did typically, not on its final draw
            span = max(set(spans), key=spans.count) if spans else (0, None, None)
            # the imposed x of the R_b arm is not a knob but the ratio the
            # construction actually produces, fitted over the same orders
            imposed = target if arm == "R_b" else {"R_d": rd, "R_l": rl}[arm]
            # the envelope of the curve on the recovered axis, carried from
            # the paired interval so the band the inversion reads is the one
            # the bias was actually established with
            envelope = [None if bias[end] is None or target is None else target * (1.0 + bias[end])
                        for end in ("low", "high")]
            results.append({
                "ratio": arm, "spacing_mm": label(spacing), "imposed": imposed,
                "truth": target, "held_rd": rd, "held_rl": rl, "side_branches": side,
                "recovered": pooled["mean"], "sd": pooled["sd"], "se": pooled["se"],
                "mean_low": envelope[0], "mean_high": envelope[1], "n_repeats": pooled["n"],
                "fit_ci_low": float(np.mean([c[0] for c in fit_ci])) if fit_ci else None,
                "fit_ci_high": float(np.mean([c[1] for c in fit_ci])) if fit_ci else None,
                "r2": float(np.mean(scores)) if scores else None,
                "bias": bias["mean"], "bias_low": bias["low"], "bias_high": bias["high"],
                "order_min": span[1], "order_max": span[2], "n_orders": span[0],
                "orders_expected": len(usable),
                # the per-case fit gates itself (see `gate`, applied just
                # below); the order count is settled afterwards, across the
                # arm (see `keep_consistent`)
                "reliable": True,
                "dropped_for": "",
            })
            got = results[-1]["recovered"]
            print(f"  spacing {label(spacing)} mm, {arm} arm, R_d {rd:.2f} R_l {rl:.2f} "
                  f"side {side}: orders {usable} -> imposed "
                  f"{imposed if imposed is None else round(imposed, 3)}, "
                  f"{arm}={got if got is None else round(got, 3)}")

    # written before the reading, so a sweep whose reading fails is not lost
    if args.out:
        write_results(args.out, results)
        print(f"\nwrote {args.out}")
    report(results, [label(spacing) for spacing in spacings], args)


def report(results, labels, args):
    """
    Prints the forward tables and the backward reading, from results alone.

    Separated from the sweep so a saved run can be read again without
    being run again. Re-reading is not a convenience: a measured ratio
    that was itself obtained with the wrong fit range has to be put back
    through the same curve once it is corrected, and re-sweeping to do
    that would change the curve as well as the value, leaving nothing to
    attribute the difference to.
    """
    # gate first, on each case's own fit, then settle the order count across
    # each arm. Both are recomputed here rather than trusted from the file,
    # so --from-csv can re-gate a saved sweep instead of re-running it
    for row in results:
        ok, reason = gate(row, args)
        row["reliable"] = ok
        row["dropped_for"] = "" if ok else reason

    for name in RATIOS:
        present = [r for r in results if r["ratio"] == name]
        if not present:
            continue
        for grid in labels:
            keep_consistent([r for r in present if r["spacing_mm"] == grid])
        print(f"\n=== {name}: imposed vs recovered ===")
        print("  spacing        imposed   recovered   relative bias   95% CI on the mean bias"
              "      R2   orders")
        for row in present:
            if row["recovered"] is None or row["truth"] is None:
                print(f"  {row['spacing_mm']:>10} {row['imposed'] or float('nan'):9.3f}   "
                      f"not measurable ({row['orders_expected']} order(s) expected to resolve)")
                continue
            flag = ("" if row["reliable"]
                    else f"   <- {row.get('dropped_for') or 'inconsistent'}, kept out of the curve")
            interval = ("        (needs --repeats)" if row["bias_low"] is None
                        else f"[{row['bias_low']:+6.1%}, {row['bias_high']:+6.1%}]")
            print(f"  {row['spacing_mm']:>10} {row['imposed']:9.3f} {row['recovered']:11.3f} "
                  f"{row['bias']:+15.1%}   {interval:<22} {row['r2']:7.3f}   "
                  f"{row['order_min']}..{row['order_max']}{flag}")
        for grid in labels:
            here = [r for r in present if r["spacing_mm"] == grid]
            kept = [r for r in here if r["reliable"]]
            counts = {r["n_orders"] for r in kept}
            print(f"  at {grid} mm: {len(kept)}/{len(here)} case(s) kept, all on "
                  f"{sorted(counts) if counts else 'no'} order(s)")
            if len(counts) > 1:
                print("  WARNING: the kept cases still do not rest on the same number of orders. "
                      "Part of what the bias column shows is that changing count rather than the "
                      "ratio, and the curve is not comparable point to point")
            if kept:
                print(f"    {verdict(kept)}")

    measured = {"R_b": args.measured_rb, "R_d": args.measured_rd, "R_l": args.measured_rl}
    for name in RATIOS:
        if not measured[name]:
            continue
        print(f"\n=== {name}: reading the curve backwards ===")
        for grid in labels:
            curve = [(r["imposed"], r["recovered"], r["mean_low"], r["mean_high"])
                     for r in results
                     if r["ratio"] == name and r["spacing_mm"] == grid and r["reliable"]]
            for value in measured[name]:
                back = invert(curve, value)
                if back == "not monotonic":
                    print(f"  spacing {grid} mm: the forward curve does not increase with "
                          f"the imposed value, so it cannot be read backwards. That is itself the "
                          f"finding -- at this voxel size the chain does not order these trees "
                          f"correctly, and no measured {name} can be attributed")
                    break
                if back is None:
                    print(f"  spacing {grid} mm: measured {value:.3f} falls outside the "
                          f"range these phantoms cover -- widen the sweep")
                else:
                    # a band with one end missing is not a failure to report:
                    # that end's envelope simply does not reach this measured
                    # value over the imposed range swept, and saying so is
                    # more use than a symmetric interval that was not measured
                    band = {"low": f"{back['low']:.3f}" if back["low"] is not None else None,
                            "high": f"{back['high']:.3f}" if back["high"] is not None else None}
                    span, note = "", ""
                    if band["low"] and band["high"]:
                        span = f" [{band['low']}, {band['high']}]"
                    elif band["low"] or band["high"]:
                        open_end = "low" if band["low"] is None else "high"
                        span = f" [{band['low'] or '?'}, {band['high'] or '?'}]"
                        note = f"      the {open_end} end of that band is open: " \
                               f"{back['why'].get(open_end, 'unavailable')}"
                    else:
                        why = "; ".join(sorted(set(back["why"].values()))) or "unavailable"
                        note = f"      no band: {why}"
                    print(f"  spacing {grid} mm: a measured {value:.3f} is what an "
                          f"imposed {back['value']:.3f}{span} produces through this chain")
                    if note:
                        print(note)
        print("  the band is the answer, not its centre. The spread across spacings on top of it "
              "is the resolution dependence: read the study's own voxel size off it, and give that "
              "size as it was acquired -- an anisotropic grid is not equivalent to any isotropic "
              "one, and the chain resampling it does not make it so")


COLUMNS = ("ratio", "spacing_mm", "imposed", "truth", "held_rd", "held_rl", "side_branches",
           "recovered", "sd", "se", "mean_low", "mean_high", "n_repeats",
           "fit_ci_low", "fit_ci_high", "r2", "bias", "bias_low", "bias_high",
           "order_min", "order_max", "n_orders", "orders_expected", "reliable", "dropped_for")

TEXT_COLUMNS = ("ratio", "spacing_mm", "dropped_for")
INT_COLUMNS = ("side_branches", "n_repeats", "order_min", "order_max", "n_orders",
               "orders_expected")


def load_results(path):
    """
    Reads back a sweep written by --out, for --from-csv.

    Everything the reading needs is in that file -- the curve, its envelope
    and what was kept -- so a measured ratio can be put back through the same
    curve as often as it is revised. `reliable` is read as written rather
    than recomputed: it records what the run decided with the R2 threshold
    and the order counts it saw, and recomputing it later against different
    arguments would silently re-select the points the curve rests on.
    """
    rows = []
    with open(path, newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for column in COLUMNS:
                value = raw.get(column, "")
                if column in TEXT_COLUMNS:
                    row[column] = value
                elif column == "reliable":
                    row[column] = value.strip().lower() in ("1", "true", "yes")
                elif value in ("", "None"):
                    row[column] = None
                elif column in INT_COLUMNS:
                    row[column] = int(float(value))
                else:
                    row[column] = float(value)
            rows.append(row)
    return rows


def write_results(path, results):
    """One row per (imposed value, spacing, ratio)."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
