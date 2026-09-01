# -*- coding: utf-8 -*-
"""
radius_audit.py

Measures the radius the chain reports against the radius the phantom was
built with, one centerline point at a time, and stratifies the error by the
angle the vessel makes with z.

Every ratio in `centerline.py` rests on the distance transform, and the
distance transform is the one measurement in the chain with no redundancy
in it: a length is averaged over hundreds of points and an angle over a
window, but a radius is one inscribed sphere in one place. `calibrate.py`
tests it only through the slope it produces. This tests it directly.

The reason to stratify by angle is that an anisotropic acquisition does not
degrade a vessel by its size but by its ORIENTATION. A radius is measured in
the plane across the vessel, so a vessel running along the fine axis has its
cross-section sampled by the two coarse ones and is the worst case, while a
vessel running along a coarse axis has a coarse and a fine axis across it and
is the best. On a 1.6:1 grid those two cases differ by a factor of 1.6 in
the sampling that actually matters, and the chain resamples both onto the
same isotropic grid, where the difference is invisible and unrecorded. The
column `cross-plane voxel` is that sampling: the root-mean-square voxel width
over the directions perpendicular to the vessel, which is 1.25 mm for a
vessel along the fine axis of a 1.25/0.80/1.25 grid and 1.05 mm for one
along a coarse axis.

Angle and calibre are confounded in any tree grown from a trunk -- the trunk
runs along z and is thick, the periphery is oblique and thin -- so the
one-way table is reported for reference and the two-way table against the
imposed order is the one to read. Within a single order the angles still
spread, because the bifurcation planes turn a quarter turn at each
generation, and that spread is the controlled comparison.

The statistics are computed over SEGMENTS, not over points. Hundreds of
points along one vessel are hundreds of samples of the same inscribed
sphere and one draw of whatever the grid did to that vessel; pooling them as
independent would divide the interval by twenty and quote a precision the
experiment does not have.

Usage:
    python -m analysis.radius_audit --spacing 1.25,0.799,1.25 --control
    python -m analysis.radius_audit --spacing 1.25,0.799,1.25 --orders 7 --side-branches 1 \
        --points-csv audit.csv
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import t as student_t

from . import phantom

ANGLE_EDGES = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)


def cross_plane_voxel(direction, spacing):
    """
    RMS voxel width over the directions across the vessel, in mm.

    The radius is a measurement in the plane perpendicular to the axis, so
    what limits it is the voxel as seen in that plane, not the voxel. For a
    unit axis u the mean of n_i^2 over the perpendicular circle is
    (1 - u_i^2) / 2, so the mean square width is sum s_i^2 (1 - u_i^2) / 2.
    It returns the voxel size unchanged on an isotropic grid, which is the
    check that it is measuring anisotropy and not size.
    """
    u = np.asarray(direction, float)
    u = u / np.linalg.norm(u)
    return float(np.sqrt(np.sum(np.asarray(spacing, float) ** 2 * (1.0 - u ** 2)) / 2.0))


def sample_axes(segments, step):
    """
    Dense samples along every segment axis, for the nearest-axis lookup.

    Sampling and a KD-tree rather than a projection onto each capsule: the
    projection is exact but quadratic in the number of segments, and the
    error it would avoid is a fraction of `step`, which is set well under a
    voxel. Returns the points, the segment index each belongs to, and the
    arclength from the segment start.
    """
    points, owners, offsets = [], [], []
    for index, segment in enumerate(segments):
        p0 = np.array([segment["x0_mm"], segment["y0_mm"], segment["z0_mm"]])
        p1 = np.array([segment["x1_mm"], segment["y1_mm"], segment["z1_mm"]])
        length = float(np.linalg.norm(p1 - p0))
        count = max(int(np.ceil(length / step)) + 1, 2)
        t = np.linspace(0.0, 1.0, count)
        points.append(p0 + t[:, None] * (p1 - p0))
        owners.append(np.full(count, index))
        offsets.append(t * length)
    return np.concatenate(points), np.concatenate(owners), np.concatenate(offsets)


def match(points, radii, segments, spacing, junction_margin, max_offset):
    """
    Attaches every measured centerline point to the segment it is inside.

    Two rejections, and both are the point of the exercise rather than
    housekeeping:

    - a point further from the nearest axis than `max_offset` local radii is
      not on that vessel. It is a spur thrown off by the thinning, or a
      skeleton branch that crossed into a weld, and its radius describes
      neither vessel.
    - a point within `junction_margin` local radii of either END of a
      segment sits in the bifurcation blob, where the maximal inscribed
      sphere is the cavity of the junction and not the vessel. That is a
      known and separately quantified effect (`trunk_calibre` trims exactly
      this), and leaving it in would swamp the orientation effect being
      measured, because junctions are not distributed evenly over angle.

    The margin is clamped to 40% of the segment so a short high-order vessel
    keeps a middle.
    """
    axis_points, owners, offsets = sample_axes(segments, 0.25 * float(np.min(spacing)))
    distance, nearest = cKDTree(axis_points).query(points)
    owner = owners[nearest]
    offset = offsets[nearest]

    truth_radius = np.array([0.5 * segments[i]["diameter_mm"] for i in owner])
    length = np.array([segments[i]["length_mm"] for i in owner])
    keep = distance <= max_offset * truth_radius
    margin = np.minimum(junction_margin * truth_radius, 0.4 * length)
    keep &= (offset >= margin) & (offset <= length - margin)

    direction = np.array([[segments[i]["x1_mm"] - segments[i]["x0_mm"],
                           segments[i]["y1_mm"] - segments[i]["y0_mm"],
                           segments[i]["z1_mm"] - segments[i]["z0_mm"]] for i in owner])
    norm = np.linalg.norm(direction, axis=1)
    angle = np.degrees(np.arccos(np.clip(np.abs(direction[:, 2]) / np.maximum(norm, 1e-12), 0.0, 1.0)))
    return {
        "segment": owner[keep], "measured": radii[keep], "truth": truth_radius[keep],
        "angle_deg": angle[keep], "order": np.array([segments[i]["strahler"] for i in owner])[keep],
        "cross_mm": np.array([cross_plane_voxel(d, spacing) for d in direction[keep]]),
        "n_points": int(len(points)), "n_kept": int(keep.sum()),
        "n_off_axis": int((distance > max_offset * truth_radius).sum()),
    }


def per_segment(matched, resolved_orders):
    """
    One row per segment: its angle, its truth, and the median error on it.

    The median over the points of a segment, because a single point that
    strayed into a neighbouring vessel would drag a mean and there is no
    reason to let it.

    `resolved_orders` marks the rows that are a measurement at all. Below
    three coarse voxels of diameter the distance transform returns its own
    floor -- 1.5 voxels, the same number for every such vessel -- so those
    segments all report the identical relative error, which is the ratio of
    the floor to their true radius and says nothing about the chain. They
    are kept in the table, flagged, and left out of every average: pooled in,
    they are the largest term in the result and they are an artefact of
    where the tree was cut, not of how it was measured.
    """
    rows = []
    for index in np.unique(matched["segment"]):
        mask = matched["segment"] == index
        truth = float(matched["truth"][mask][0])
        rows.append({
            "segment": int(index), "order": int(matched["order"][mask][0]),
            "angle_deg": float(matched["angle_deg"][mask][0]),
            "cross_mm": float(matched["cross_mm"][mask][0]),
            "truth_mm": truth,
            "measured_mm": float(np.median(matched["measured"][mask])),
            "error_mm": float(np.median(matched["measured"][mask]) - truth),
            "error_rel": float(np.median(matched["measured"][mask]) / truth - 1.0),
            "n_points": int(mask.sum()),
            "resolved": int(int(matched["order"][mask][0]) in resolved_orders),
        })
    return rows


def summarize(rows, key="error_rel"):
    """Mean, its 95% t interval, and the count, over a list of segment rows."""
    values = np.array([row[key] for row in rows], float)
    if len(values) == 0:
        return None
    mean = float(values.mean())
    if len(values) < 2:
        return {"mean": mean, "low": None, "high": None, "n": 1}
    half = float(student_t.ppf(0.975, len(values) - 1)) * float(values.std(ddof=1)) / np.sqrt(len(values))
    return {"mean": mean, "low": mean - half, "high": mean + half, "n": len(values)}


def run_chain(segments, spacing, args, workdir, tag):
    """Rasterizes the tree at `spacing`, runs centerline.py, returns its points."""
    spacing = phantom.as_triple(spacing)
    rng = np.random.default_rng(args.seed)
    volume, origin = phantom.rasterize(segments, spacing, args.margin)
    mask = phantom.degrade(volume, spacing, phantom.default_blur(spacing, args.blur),
                           args.noise, args.threshold, rng)
    mask_path = os.path.join(workdir, f"ph_{tag}.nii.gz")
    points_path = os.path.join(workdir, f"pt_{tag}.csv")
    if os.path.exists(points_path):
        os.remove(points_path)
    phantom.write_mask(mask, spacing, origin, mask_path)

    command = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "centerline.py"),
               "--input", mask_path, "--ordering", "strahler_dd", "--no-report",
               "--csv", points_path, "--output", os.path.join(workdir, f"cl_{tag}.nii.gz")]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0 or not os.path.exists(points_path):
        raise SystemExit(f"centerline.py failed at {tag}:\n{done.stderr[-800:]}")
    table = np.genfromtxt(points_path, delimiter=",", names=True)
    points = np.column_stack([table["x_mm"], table["y_mm"], table["z_mm"]])
    return points, np.asarray(table["radius_mm"], float), int(mask.sum())


def print_table(title, groups, columns=("n", "truth", "measured", "error_mm", "error_rel")):
    """One stratum per line, with the interval on the mean relative error."""
    print(f"\n{title}")
    print("  stratum          seg   cross-plane vox   true r    measured r    error      "
          "relative error   95% CI on the mean")
    for name, rows in groups:
        if not rows:
            print(f"  {name:<15} {0:5d}   (no segment survives the junction and off-axis cuts)")
            continue
        stats = summarize(rows)
        cross = float(np.mean([row["cross_mm"] for row in rows]))
        truth = float(np.mean([row["truth_mm"] for row in rows]))
        measured = float(np.mean([row["measured_mm"] for row in rows]))
        error = float(np.mean([row["error_mm"] for row in rows]))
        interval = ("        --" if stats["low"] is None
                    else f"[{stats['low']:+6.1%}, {stats['high']:+6.1%}]")
        print(f"  {name:<15} {len(rows):5d} {cross:17.3f} {truth:8.3f} {measured:13.3f} "
              f"{error:+9.3f} {stats['mean']:+16.1%}   {interval}")


def angle_groups(rows, edges=ANGLE_EDGES):
    """The segment rows cut into angle-with-z bins, low to high."""
    groups = []
    for low, high in zip(edges, edges[1:]):
        groups.append((f"{low:.0f}-{high:.0f} deg",
                       [row for row in rows if low <= row["angle_deg"] < high
                        or (high == edges[-1] and row["angle_deg"] == high)]))
    return groups


def print_two_way(rows, orders):
    """
    Relative error by imposed order and by angle, the confound taken out.

    The one-way table cannot separate an orientation effect from a size
    effect, because in a tree grown from a trunk the thick vessels are the
    ones near z. This one compares angles WITHIN an order, where the calibre
    is fixed by construction and only the direction varies.
    """
    print("\nrelative error by imposed order and angle with z (n segments in brackets)")
    header = "  order   true d (mm)  " + "".join(f"{low:.0f}-{high:.0f} deg".rjust(18)
                                                 for low, high in zip(ANGLE_EDGES, ANGLE_EDGES[1:]))
    print(header)
    for order in orders:
        here = [row for row in rows if row["order"] == order]
        if not here:
            continue
        cells = []
        for _, group in angle_groups(here):
            stats = summarize(group)
            cells.append("-".rjust(18) if stats is None
                         else f"{stats['mean']:+.1%} [{stats['n']}]".rjust(18))
        print(f"  {order:5d} {2.0 * np.mean([r['truth_mm'] for r in here]):11.2f}  " + "".join(cells))


def print_slope(rows):
    """
    The orientation effect isolated: error against cross-plane voxel, within
    each order, pooled.

    A slope in mm of error per mm of cross-plane voxel. It is fitted on the
    residuals after each order's own mean has been removed, which is what
    "within order" means here: the between-order differences are calibre
    effects and are removed rather than fitted, so what is left is the part
    of the radius error that only the orientation explains.
    """
    x, y, used, span = [], [], [], []
    for order in sorted({row["order"] for row in rows}):
        here = [row for row in rows if row["order"] == order]
        if len(here) < 4:
            continue
        cross = np.array([row["cross_mm"] for row in here])
        error = np.array([row["error_rel"] for row in here])
        if cross.std() < 1e-6:
            continue
        x.append(cross - cross.mean())
        y.append(error - error.mean())
        used.append(order)
        span.append((cross.min(), cross.max()))
    if not x:
        print("\nwithin-order slope: not estimable -- the orders left have no spread in "
              "orientation, or too few segments")
        return
    x, y = np.concatenate(x), np.concatenate(y)
    slope, _ = np.polyfit(x, y, 1)
    residual = y - slope * x
    scatter = float((x ** 2).sum())
    se = np.sqrt(float((residual ** 2).sum()) / (len(x) - 2) / scatter)
    half = float(student_t.ppf(0.975, len(x) - 2)) * se
    reach = float(max(high for _, high in span) - min(low for low, _ in span))
    print(f"\nwithin-order slope: {slope:+.1%} of relative radius error per mm of cross-plane "
          f"voxel [{slope - half:+.1%}, {slope + half:+.1%}], on {len(x)} segments over "
          f"orders {used}")
    # a slope per millimetre extrapolates well past the orientations that
    # actually occur, and reads as a much larger effect than the tree
    # contains. What the experiment supports is the difference between its
    # own best- and worst-oriented vessels
    print(f"  the orientations present span {reach:.3f} mm of cross-plane voxel, so the effect "
          f"between the best- and worst-oriented vessel of an order is {abs(slope) * reach:.1%} "
          f"of its radius. Quote that, not the slope")
    if (slope - half > 0) == (slope + half > 0):
        print("  the interval excludes zero: at fixed calibre the radius the chain reports depends "
              "on which way the vessel runs through the grid. On an anisotropic acquisition that "
              "is a bias on every per-order diameter, and it is not visible anywhere in the "
              "chain's own output, because by then the mask has been resampled to isotropic")
    else:
        print("  the interval covers zero: no orientation effect is resolved at this spacing and "
              "this number of segments. Report the width, not the point")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spacing", default="1.25,0.799,1.25", metavar="MM",
                        help="Acquired voxel size: one number, or three comma-separated. The "
                             "audit is worth running on an anisotropic grid; on an isotropic one "
                             "it measures the chain's radius bias with no orientation term to "
                             "find. Default: 1.25,0.799,1.25")
    parser.add_argument("--control", action="store_true",
                        help="Also run the same tree isotropic at the FINEST axis of --spacing, "
                             "which is the grid the chain resamples onto. Any orientation effect "
                             "that survives there belongs to the chain; the difference between "
                             "the two runs is what the anisotropy of the acquisition cost")
    parser.add_argument("--orders", type=int, default=7, help="Strahler orders. Default: 7")
    parser.add_argument("--root-diameter", type=float, default=20.0,
                        help="Trunk diameter (mm). Default: 20")
    parser.add_argument("--root-length", type=float, default=40.0,
                        help="Trunk length (mm). Default: 40")
    parser.add_argument("--rd", type=float, default=1.50, help="Imposed diameter ratio. Default: 1.50")
    parser.add_argument("--rl", type=float, default=1.40, help="Imposed length ratio. Default: 1.40")
    parser.add_argument("--angle", type=float, default=70.0,
                        help="Full bifurcation angle in degrees. It is also what spreads the "
                             "vessels over orientation, so a very small value leaves the audit "
                             "with no angles to compare. Default: 70")
    parser.add_argument("--side-branches", type=int, default=0, metavar="N",
                        help="Side branches per element. Adds orientations and asymmetric "
                             "junctions. Default: 0")
    parser.add_argument("--side-drop", type=int, default=2, metavar="K",
                        help="Order drop of a side branch. Default: 2")
    parser.add_argument("--jitter", type=float, default=0.0,
                        help="Lognormal scatter on diameters and lengths. Default: 0")
    parser.add_argument("--junction-margin", type=float, default=1.0, metavar="RADII",
                        help="Local radii trimmed off each end of a segment before its points "
                             "count. Inside that the inscribed sphere is the junction cavity, not "
                             "the vessel. Default: 1")
    parser.add_argument("--max-offset", type=float, default=0.5, metavar="RADII",
                        help="A centerline point further than this many local radii from the "
                             "nearest true axis is not on that vessel and is dropped. Default: 0.5")
    parser.add_argument("--blur", type=float, nargs="+", default=None, metavar="MM",
                        help="Blur sigma (mm), one value or three. Default: 2/3 voxel per axis")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise level. Default: 0.05")
    parser.add_argument("--threshold", type=float, default=0.5, help="Rebinarization. Default: 0.5")
    parser.add_argument("--margin", type=float, default=4.0, help="Empty margin (mm). Default: 4")
    parser.add_argument("--seed", type=int, default=0, help="Random seed. Default: 0")
    parser.add_argument("--points-csv", help="Per-segment audit table to write")
    parser.add_argument("--keep", help="Directory to keep the phantoms and point CSVs in")
    args = parser.parse_args()

    spacing = phantom.as_triple([float(v) for v in str(args.spacing).replace("x", ",").split(",")])
    workdir = args.keep or tempfile.mkdtemp(prefix="radius_audit_")
    os.makedirs(workdir, exist_ok=True)
    print(f"working in {workdir}")

    segments = phantom.build_tree(args.orders, args.root_diameter, args.root_length,
                                  args.rd, args.rl, np.radians(0.5 * args.angle), args.jitter,
                                  np.random.default_rng(args.seed),
                                  args.side_branches, args.side_drop)
    usable = phantom.usable_orders(segments, args.orders, spacing)
    print(f"tree: {len(segments)} segments, {args.orders} orders, "
          f"orders {usable} clear three coarse voxels")

    runs = [("acquired", spacing)]
    if args.control:
        runs.append(("control", phantom.as_triple(float(spacing.min()))))

    # Both runs are held to the orders the ACQUIRED grid resolves, the
    # control included. A finer grid keeps more of the thin end alive, and
    # the thin end is where the radius is censored -- so comparing each run
    # over whatever it happened to reach would compare two different sets of
    # vessels and report the difference as an effect of the spacing.
    audits = {}
    for tag, grid in runs:
        points, radii, voxels = run_chain(segments, grid, args, workdir, tag)
        matched = match(points, radii, segments, grid, args.junction_margin, args.max_offset)
        rows = per_segment(matched, usable)
        resolved = [row for row in rows if row["resolved"]]
        censored = [row for row in rows if not row["resolved"]]
        audits[tag] = (grid, rows)
        print(f"\n=== {tag} grid {np.round(grid, 3).tolist()} mm "
              f"({grid.max() / grid.min():.2f}:1, {voxels} voxels) ===")
        print(f"  {matched['n_points']} centerline points -> {matched['n_kept']} kept "
              f"({matched['n_off_axis']} over {args.max_offset} radii off any axis, the rest inside "
              f"{args.junction_margin} radii of a junction), on {len(rows)} segments")
        if censored:
            floor = summarize(censored)
            print(f"  {len(censored)} of them are under three coarse voxels (orders "
                  f"{sorted({row['order'] for row in censored})}) and report {floor['mean']:+.1%}, "
                  f"which is the distance transform returning its floor rather than measuring: "
                  f"left out of everything below")
        overall = summarize(resolved)
        if overall is None:
            print("  nothing resolved survives the cuts -- loosen --junction-margin, add orders, "
                  "or lower the spacing")
            continue
        print(f"  radius bias over the resolved orders {sorted({row['order'] for row in resolved})}"
              f": {overall['mean']:+.1%} [{overall['low']:+.1%}, {overall['high']:+.1%}] on "
              f"{overall['n']} segments")
        print_table("relative error by angle with z (0 = along z, 90 = across it)",
                    angle_groups(resolved))
        print("  read this table for reference only: angle and calibre are confounded in a tree "
              "grown from a trunk, the thick vessels being the ones near z")
        print_two_way(resolved, sorted({row["order"] for row in resolved}))
        print_slope(resolved)

    if args.control and len(audits) == len(runs):
        print("\n=== what the anisotropy cost ===")
        for tag, _ in runs:
            grid, rows = audits[tag]
            stats = summarize([row for row in rows if row["resolved"]])
            if stats is None:
                continue
            print(f"  {tag:<10} {np.round(grid, 3).tolist()} mm: radius bias {stats['mean']:+.1%} "
                  f"[{stats['low']:+.1%}, {stats['high']:+.1%}] on {stats['n']} segments")
        print("  the control is the same tree, over the same orders, on the grid the chain "
              "resamples the other one onto. Whatever separates the two lines is information the "
              "acquisition never had, and no step of the chain can put it back -- which is why a "
              "phantom rasterized isotropic at the target size cannot stand in for the study")

    if args.points_csv:
        columns = ("segment", "order", "angle_deg", "cross_mm", "truth_mm", "measured_mm",
                   "error_mm", "error_rel", "n_points", "resolved")
        with open(args.points_csv, "w") as handle:
            handle.write(",".join(("grid",) + columns) + "\n")
            for tag, (grid, rows) in audits.items():
                for row in rows:
                    handle.write(tag + "," + ",".join(
                        f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])
                        for c in columns) + "\n")
        print(f"\nwrote {args.points_csv}")


if __name__ == "__main__":
    main()
