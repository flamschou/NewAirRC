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

The tree is a binary tree in Strahler order. Left symmetric, every order-n
segment splits into two of order n-1, so

    R_b = 2 exactly
    R_d = --rd exactly       D_n = D_1 * R_d^(n-1)
    R_l = --rl exactly       L_n = L_1 * R_l^(n-1)

and, being symmetric, elements and segments coincide: every element is one
segment. That tree calibrates R_d and R_l and cannot calibrate R_b, whose
value is 2 by construction -- the theoretical floor, while what is worth
measuring in a real tree is precisely the excess over 2. `--side-branches`
lifts that restriction: an element of order n then carries side branches of
order n - `--side-drop` along its length, which is the monopodial pattern of
a real lung, and R_b becomes a known number above 2 (see `imposed_rb`).

Two voxel sizes matter and they are not the same one. `--spacing` takes
three values as readily as one, and it should: a study acquires anisotropic
voxels and `centerline.py` upsamples them to the finest of the three. An
isotropic phantom rasterized at the *target* size therefore skips the step
that destroys the information -- its boundary is known to 0.8 mm in all
three directions, where the study's is known to 0.8 mm along one axis and
to the slice thickness along the others. Same working grid, different
information. Rasterize at the ACQUIRED size and let the chain upsample.

Usage:
    python -m analysis.phantom --orders 8 --spacing 1.5 --output tree.nii.gz
    python -m analysis.phantom --orders 8 --spacing 1.25 0.799 1.25 --output tree.nii.gz
    python -m analysis.phantom --orders 8 --side-branches 1 --side-drop 2 --output tree.nii.gz
    python -m analysis.centerline --input tree.nii.gz --ordering strahler_dd --fit-orders 1 8
"""
import argparse

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

# The truth has to be fitted with the estimator the chain uses, not with a
# second one written here: R_b on an asymmetric tree is a regression slope,
# not a closed form, and two regressions that differ in their weighting or
# their order range would show up in the comparison as a bias of the chain.
# There is one copy of the fit and both sides call it.
from .centerline import branching_ratios


def unit(vector):
    """Normalizes a vector, leaving the zero vector alone."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def parse_spacing(text):
    """
    One --spacing entry to a 3-vector: "0.8" or "1.25,0.799,1.25".

    The comma form is the point of the option. A study acquires anisotropic
    voxels; bracketing them with two isotropic runs answers a question about
    two grids that do not exist rather than the one that does.

    Shared by phantom, calibrate and radius_audit so the three read a spacing
    the same way -- calibrate sweeps a LIST of grids, so space already
    separates its entries and the comma is the only separator left inside
    one; the other two accept either form and mean the same thing by both.
    """
    return as_triple([float(v) for v in str(text).replace("x", ",").split(",")])


def as_triple(value, name="spacing", allow_zero=False):
    """
    Accepts a scalar or a length-3 sequence, returns a float array of 3.

    Everything downstream works on a 3-vector so the isotropic and the
    anisotropic case go through exactly the same code path -- an isotropic
    run is an anisotropic one whose three values happen to be equal, not a
    separate branch that could drift from it.

    A spacing of zero is meaningless and rejected; a blur of zero is a
    request, and a bad one -- see `degrade` -- but it is the caller's to make.
    """
    values = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
    if values.size == 1:
        values = np.repeat(values, 3)
    if values.size != 3 or not np.all(values >= 0 if allow_zero else values > 0):
        raise ValueError(f"--{name} takes one or three "
                         f"{'non-negative' if allow_zero else 'positive'} values, got {value}")
    return values


def build_tree(orders, root_diameter, root_length, rd, rl, half_angle, taper_jitter, rng,
               side_branches=0, side_drop=2, side_half_angle=None):
    """
    Lays out the segments of the tree, from the trunk down.

    Each bifurcation is planar and symmetric, and its plane is turned by 90
    degrees at the next generation -- the alternation is what keeps the two
    halves of the tree from growing into each other, which would weld them
    together on the raster and give the skeleton the loops it is meant to be
    tested against.

    With `side_branches` > 0 the tree stops being symmetric. An element of
    order n is then cut into `side_branches` + 1 collinear pieces, and at
    every joint but the last a branch of order n - `side_drop` leaves the
    axis, the axis itself carrying straight on. Strahler is unharmed: the
    two children of such a joint have different orders, so the parent keeps
    the larger one and the element stays order n from end to end, while the
    last joint is the true bifurcation into two elements of order n - 1.
    What changes is the count: each order now receives 2 elements per parent
    AND `side_branches` per element `side_drop` orders above it, so R_b rises
    above 2 by a known amount. That is the only construction here that makes
    R_b a measurement rather than a tautology.

    `taper_jitter` multiplies every diameter and length by a lognormal draw.
    With 0 the tree is exact and the ratios come back to the third decimal;
    the interesting question is how much scatter the fit tolerates, so run it
    at 0 first, then at the dispersion of a real tree (0.1 to 0.2).

    Diameter is a function of the order alone, so R_d stays exactly `rd`
    however asymmetric the tree gets. Length is imposed on the ELEMENT, and
    the pieces share it: the element-counted R_l is exactly `rl`, and the
    segment-counted one is not, because an order-1 element carries no side
    branch and is therefore one long piece where an order-2 element is
    several short ones. That gap is a property of this tree, not an error of
    the chain, which is why the truth is refitted rather than assumed -- see
    `truth_ratios`.

    Returns a list of segment dicts, trunk first.
    """
    segments = []
    element_count = [0]
    side_half_angle = half_angle if side_half_angle is None else side_half_angle

    def grow(start, direction, up, order, parent):
        diameter = root_diameter * rd ** (order - orders)
        length = root_length * rl ** (order - orders)
        if taper_jitter > 0:
            diameter *= float(rng.lognormal(0.0, taper_jitter))
            length *= float(rng.lognormal(0.0, taper_jitter))

        # a side branch that would fall below order 1 has nowhere to go, so
        # the bottom `side_drop` orders of the tree are plain dichotomous
        joints = side_branches if order - side_drop >= 1 else 0
        element = element_count[0]
        element_count[0] += 1
        piece = length / (joints + 1)
        node, here = np.asarray(start, dtype=float), parent
        for j in range(joints + 1):
            end = node + direction * piece
            index = len(segments)
            segments.append({
                "segment_id": index, "parent_id": here, "element_id": element,
                "strahler": order, "length_mm": float(piece), "diameter_mm": float(diameter),
                "x0_mm": float(node[0]), "y0_mm": float(node[1]), "z0_mm": float(node[2]),
                "x1_mm": float(end[0]), "y1_mm": float(end[1]), "z1_mm": float(end[2]),
            })
            node, here = end, index
            if j == joints:
                break
            # the side branch peels off in the current plane, alternating
            # sides, and the plane turns a quarter turn at the next joint --
            # the same anti-collision rule the bifurcations follow
            in_plane = unit(up - np.dot(up, direction) * direction)
            normal = unit(np.cross(direction, in_plane))
            sign = 1.0 if j % 2 == 0 else -1.0
            side = unit(np.cos(side_half_angle) * direction + sign * np.sin(side_half_angle) * in_plane)
            grow(end, side, normal, order - side_drop, index)
            up = normal

        if order <= 1:
            return
        # the bifurcation plane is spanned by the parent direction and `up`;
        # the children take the old plane normal as their own `up`, which
        # rotates the next bifurcation a quarter turn
        in_plane = unit(up - np.dot(up, direction) * direction)
        normal = unit(np.cross(direction, in_plane))
        for sign in (1.0, -1.0):
            child = unit(np.cos(half_angle) * direction + sign * np.sin(half_angle) * in_plane)
            grow(end, child, normal, order - 1, here)

    grow(np.zeros(3), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), orders, -1)
    return segments


def imposed_rb(side_branches, side_drop):
    """
    The branching ratio the construction converges to, in closed form.

    Counting elements, order n receives two per element of order n + 1 and
    `side_branches` per element of order n + `side_drop`, so

        N_n = 2 N_{n+1} + s N_{n+k}

    and a geometric N_n = x N_{n+1} solves x^k = 2 x^(k-1) + s. With s = 0
    that is x = 2, the symmetric tree. With s = 1, k = 2 it is 1 + sqrt(2).

    This is the asymptote, reached in the middle of a tall tree. A real
    phantom is finite and its top few orders count 1, 2, 4 elements whatever
    else is true, so the slope actually fitted over the measurable orders is
    near this value and not equal to it. `truth_ratios` gives that slope, and
    the gap between the two is the finite-size effect of the phantom -- which
    is a floor on how tightly any R_b can be attributed, and belongs in the
    report next to the bias.
    """
    if side_branches <= 0:
        return 2.0
    coefficients = np.zeros(side_drop + 1)
    coefficients[0] = 1.0            # x^k
    coefficients[1] = -2.0           # -2 x^(k-1)
    coefficients[-1] -= side_branches
    roots = np.roots(coefficients)
    real = [float(r.real) for r in roots if abs(r.imag) < 1e-9 and r.real > 0]
    return max(real) if real else float("nan")


def rasterize(segments, spacing, margin):
    """
    Draws the segments as capsules on a grid, with an anti-aliased surface.

    A voxel gets the fraction of it that the vessel covers, approximated from
    its distance to the axis. Drawing hard binary capsules instead would put
    every wall exactly on a voxel boundary, which is the one thing a real
    acquisition never does and which would flatter the radius estimates by
    removing the partial volume the chain has to cope with.

    The width of that partial-volume ramp is the width of the voxel ALONG
    THE WALL NORMAL, not a single number: on an anisotropic grid a wall
    facing the thin axis is resolved to that axis and a wall facing the thick
    one is smeared over it, and that difference, not the voxel count, is what
    an anisotropic acquisition actually does to a vessel. The ramp is
    therefore the spacings averaged with the components of the radial
    direction as weights, which reduces to the plain voxel size when the grid
    is isotropic and to the exact axis spacing for a wall facing an axis.

    Returns the occupancy volume in [0, 1] and the world origin of voxel 0.
    """
    spacing = as_triple(spacing)
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
        radial = points - (p0 + t * axis)
        distance = np.linalg.norm(radial, axis=-1)
        weights = np.abs(radial)
        total = weights.sum(axis=-1)
        width = np.where(total > 0, (weights * spacing).sum(axis=-1) / np.where(total > 0, total, 1.0),
                         spacing.mean())
        coverage = np.clip(0.5 + (r - distance) / width, 0.0, 1.0)
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

    The blur is a 3-vector of millimetres, and it defaults to two thirds of
    the voxel ALONG EACH AXIS: a thick slice is not merely sampled coarsely,
    it is integrated over its thickness, and a point spread function held
    isotropic while the sampling is not would put the anisotropy back into
    the grid alone -- which is the very thing the chain then undoes by
    upsampling. Pass a scalar `--blur` to impose an isotropic PSF on an
    anisotropic grid, which is a different physical claim and rarely true.
    """
    spacing, blur_mm = as_triple(spacing), as_triple(blur_mm, "blur", allow_zero=True)
    if np.any(blur_mm > 0):
        volume = gaussian_filter(volume, blur_mm / spacing)
    if noise > 0:
        volume = volume + rng.normal(0.0, noise, volume.shape).astype(np.float32)
    return volume >= threshold


def write_mask(mask, spacing, origin, path):
    """Writes the rasterized phantom with the affine its coordinates imply."""
    affine = np.eye(4)
    affine[:3, :3] = np.diag(as_triple(spacing))
    affine[:3, 3] = origin
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), path)


def default_blur(spacing, blur=None):
    """
    Point spread function used when none is given: two thirds of a voxel,
    per axis, so an anisotropic grid gets an anisotropic PSF.
    """
    return (2.0 / 3.0) * as_triple(spacing) if blur is None else as_triple(blur, "blur", allow_zero=True)


def write_segments_csv(path, segments):
    """The ground truth, one row per segment."""
    columns = ("segment_id", "parent_id", "element_id", "strahler", "length_mm", "diameter_mm",
               "x0_mm", "y0_mm", "z0_mm", "x1_mm", "y1_mm", "z1_mm")
    rows = [",".join(columns)]
    for segment in segments:
        rows.append(",".join(f"{segment[c]:.4f}" if isinstance(segment[c], float) else str(segment[c])
                             for c in columns))
    with open(path, "w") as handle:
        handle.write("\n".join(rows) + "\n")


def elements(segments):
    """The imposed segments regrouped into elements, one dict per element."""
    groups = {}
    for segment in segments:
        groups.setdefault(segment["element_id"], []).append(segment)
    out = []
    for pieces in groups.values():
        out.append({"strahler": pieces[0]["strahler"],
                    "diameter_mm": pieces[0]["diameter_mm"],
                    "length_mm": float(sum(p["length_mm"] for p in pieces))})
    return out


def truth_summary(segments, counting="element"):
    """
    The imposed tree aggregated order by order, in the shape `centerline.py`
    aggregates the measured one, so the same fit can be run on both.
    """
    parts = elements(segments) if counting == "element" else segments
    rows = []
    for order in sorted({p["strahler"] for p in parts}):
        group = [p for p in parts if p["strahler"] == order]
        rows.append({
            "order": order,
            "n_branches": len(group),
            "mean_diameter_mm": float(np.mean([p["diameter_mm"] for p in group])),
            "mean_length_mm": float(np.mean([p["length_mm"] for p in group])),
        })
    return rows


def truth_ratios(segments, counting="element", order_range=None):
    """
    The three ratios OF THE PHANTOM, fitted the way the chain fits them.

    For R_d, and for R_l counted on elements, this returns the imposed value
    to the last decimal and is a check that the construction is what it says.
    For R_b it is the only honest target: on an asymmetric tree R_b is the
    slope of a regression over a finite number of orders, not a closed form,
    so comparing the chain's slope to the asymptote of `imposed_rb` would
    charge the chain for the phantom's own truncation. Fit both sides over
    the same orders and the difference is the chain.

    `order_range` is (min, max) inclusive, and it matters: the top orders of
    any finite tree count 1, 2, 4 whatever the rule is, so a range that
    reaches the trunk drags the fitted R_b towards 2.
    """
    result = branching_ratios(truth_summary(segments, counting), "strahler", order_range)
    return {name: (fit["ratio"] if fit else None) for name, fit in result["fits"].items()}


def usable_orders(segments, orders, spacing):
    """
    The orders whose mean diameter clears three voxels, at this spacing.

    A mechanical criterion, computed from the imposed tree before anything is
    measured, so the fit range of the calibration run is fixed in advance
    rather than chosen once the answer is visible.

    Three voxels of the COARSEST axis. The chain resamples to the finest one
    and applies its own floor there, which on a 1.6:1 grid is a floor 1.6
    times too permissive: upsampling adds samples, not information, and a
    vessel two slices across does not become resolved by being interpolated
    onto four. The optimistic count is printed next to this one so the gap
    is visible, but the fit range is set by the honest criterion.
    """
    spacing = as_triple(spacing)
    keep = []
    for order in range(1, orders + 1):
        group = [s for s in segments if s["strahler"] == order]
        if group and float(np.mean([s["diameter_mm"] for s in group])) / spacing.max() >= 3.0:
            keep.append(order)
    return keep


def print_truth(segments, orders, spacing, rd, rl, side_branches=0, side_drop=2):
    """
    Prints what the chain should find, and where the grid stops allowing it.

    The last columns are the whole reason for this file: an order whose
    diameter is under three voxels cannot be measured, only guessed, so the
    fit range that `centerline.py` will honestly support is known here in
    advance rather than discovered afterwards. On an anisotropic grid there
    are two such counts, one per end of the voxel, and the truth is the
    coarse one.
    """
    spacing = as_triple(spacing)
    fine, coarse = spacing.min(), spacing.max()
    parts = elements(segments)
    usable = usable_orders(segments, orders, spacing)
    fitted = truth_ratios(segments, "element",
                          (min(usable), max(usable)) if usable else None)
    asymptote = imposed_rb(side_branches, side_drop)
    fitted_rb = fitted["R_b"]
    if side_branches <= 0:
        header = "R_b = 2.000"
    elif fitted_rb is None:
        header = f"R_b = {asymptote:.3f} asymptotically, too few orders resolve to fit it"
    else:
        header = f"R_b = {fitted_rb:.3f} over the measurable orders ({asymptote:.3f} asymptotically)"
    # on a symmetric tree every bifurcation has d_parent = R_d * d_child, so
    # d_p^n = 2 d_c^n solves in closed form and centerline.py's histogram of
    # Murray exponents has a single value to be compared against. With side
    # branches the junctions are of two kinds and the histogram has two modes
    murray = np.log(2.0) / np.log(rd) if rd > 1.0 else float("inf")
    print(f"\n=== ground truth ({header}, R_d = {rd:.3f}, R_l = {rl:.3f}, "
          f"Murray exponent at the symmetric junctions = {murray:.3f}) ===")
    print("ord  elem   seg   diameter_mm   length_mm   diam in coarse vox   in fine vox")
    counts = {}
    for element in parts:
        counts[element["strahler"]] = counts.get(element["strahler"], 0) + 1
    for order in range(1, orders + 1):
        group = [s for s in segments if s["strahler"] == order]
        if not group:
            continue
        diameter = float(np.mean([s["diameter_mm"] for s in group]))
        length = float(np.mean([e["length_mm"] for e in parts if e["strahler"] == order]))
        flag = "" if order in usable else "   <- under 3 coarse voxels, not measurable"
        print(f"{order:3d} {counts.get(order, 0):5d} {len(group):5d} {diameter:13.3f} {length:11.3f} "
              f"{diameter / coarse:20.2f} {diameter / fine:13.2f}{flag}")

    if side_branches > 0 and fitted_rb is not None:
        print(f"the tree is asymmetric: every element of order n carries {side_branches} side "
              f"branch(es) of order n-{side_drop}, so R_b sits above 2 by construction and IS "
              f"calibratable. Its truth is the fitted {fitted['R_b']:.3f} rather than the "
              f"asymptotic {asymptote:.3f}: a finite tree counts 1, 2, 4 "
              f"elements at the top whatever its rule, and the chain is not to be charged for that")
    elif side_branches > 0:
        print(f"the tree is asymmetric (R_b -> {asymptote:.3f}) but too few orders resolve at this "
              f"spacing to fit its truth, so nothing here can be compared to a measured R_b")
    else:
        print("R_b = 2.000 and the Murray exponent are properties of the construction, not targets: "
              "the tree is a symmetric binary tree, so every junction splits in two and R_b is 2 by "
              "definition, and log(2)/log(R_d) follows from R_d alone. This phantom calibrates R_d "
              "and R_l only. Use --side-branches to impose an asymmetry and make R_b a measurement")
    if not np.allclose(spacing, spacing[0]):
        print(f"the grid is anisotropic ({np.round(spacing, 3).tolist()} mm, {coarse / fine:.2f}:1). "
              f"centerline.py will upsample it to {fine:.3f} mm to measure on, and floor its fit at "
              f"{3.0 * coarse:.2f} mm -- three voxels of the coarse axis, the same rule as the "
              f"coarse column above. Counting the floor on the resampled grid instead would allow "
              f"{3.0 * fine:.2f} mm and admit an order this acquisition never resolved")
    if usable:
        print(f"the chain can only be held to orders {min(usable)}..{max(usable)} at "
              f"{np.round(spacing, 3).tolist()} mm; run centerline.py with "
              f"--fit-orders {min(usable)} {max(usable)}")
    else:
        print(f"no order resolves at {np.round(spacing, 3).tolist()} mm -- lower --spacing or raise "
              f"--root-diameter")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="phantom.nii.gz", help="Mask to write (nifti)")
    parser.add_argument("--segments-csv", help="Ground truth segment table to write")
    parser.add_argument("--orders", type=int, default=7, help="Number of Strahler orders. Default: 7")
    parser.add_argument("--root-diameter", type=float, default=20.0,
                        help="Diameter (mm) of the trunk. Default: 20")
    parser.add_argument("--root-length", type=float, default=40.0,
                        help="Length (mm) of the trunk. Default: 40")
    parser.add_argument("--rd", type=float, default=1.50, help="Imposed diameter ratio. Default: 1.50")
    parser.add_argument("--rl", type=float, default=1.40, help="Imposed length ratio. Default: 1.40")
    parser.add_argument("--angle", type=float, default=70.0,
                        help="Full bifurcation angle in degrees. Default: 70")
    parser.add_argument("--side-branches", type=int, default=0, metavar="N",
                        help="Side branches carried by every element, which is what lifts R_b above "
                             "2 and makes it calibratable. 0 gives the symmetric tree. Default: 0")
    parser.add_argument("--side-drop", type=int, default=2, metavar="K",
                        help="How many orders below its parent a side branch is. 1 puts it one "
                             "order down, which is a near-symmetric trifurcation; 2 is the "
                             "monopodial pattern of a lung. Default: 2")
    parser.add_argument("--side-angle", type=float, default=None,
                        help="Full take-off angle of the side branches, in degrees. Default: --angle")
    parser.add_argument("--jitter", type=float, default=0.0,
                        help="Lognormal spread applied to every diameter and length. 0 gives an "
                             "exact tree. Default: 0")
    parser.add_argument("--spacing", nargs="+", default=["1.5"], metavar="MM",
                        help="Voxel size (mm): one value for an isotropic grid, three for the "
                             "anisotropic grid of a real acquisition, space- or comma-separated "
                             "('1.25 0.799 1.25' and '1.25,0.799,1.25' are the same grid). Give "
                             "the ACQUIRED size, not the size the chain resamples to. "
                             "Default: 1.5")
    parser.add_argument("--blur", type=float, nargs="+", default=None, metavar="MM",
                        help="Gaussian blur sigma (mm) applied before thresholding, one value or "
                             "three. Default: two thirds of the voxel along each axis")
    parser.add_argument("--noise", type=float, default=0.05,
                        help="SD of the noise added to the blurred occupancy. Default: 0.05")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Occupancy at which a voxel becomes vessel. Default: 0.5")
    parser.add_argument("--margin", type=float, default=4.0, help="Empty margin (mm). Default: 4")
    parser.add_argument("--seed", type=int, default=0, help="Random seed. Default: 0")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    spacing = parse_spacing(",".join(args.spacing))
    blur = default_blur(spacing, args.blur)

    segments = build_tree(args.orders, args.root_diameter, args.root_length, args.rd, args.rl,
                          np.radians(0.5 * args.angle), args.jitter, rng,
                          args.side_branches, args.side_drop,
                          None if args.side_angle is None else np.radians(0.5 * args.side_angle))
    print(f"tree: {len(segments)} segments, {len(elements(segments))} elements, "
          f"{sum(1 for s in segments if s['strahler'] == 1)} terminals, {args.orders} orders")

    volume, origin = rasterize(segments, spacing, args.margin)
    print(f"raster: shape={volume.shape} at {np.round(spacing, 3).tolist()} mm")
    mask = degrade(volume, spacing, blur, args.noise, args.threshold, rng)
    print(f"mask: {int(mask.sum())} voxels ({mask.sum() * float(np.prod(spacing)) / 1000.0:.2f} mL), "
          f"blur sigma {np.round(blur, 2).tolist()} mm, noise {args.noise}")

    write_mask(mask, spacing, origin, args.output)
    print(f"wrote {args.output}")
    if args.segments_csv:
        write_segments_csv(args.segments_csv, segments)
        print(f"wrote {args.segments_csv}")

    print_truth(segments, args.orders, spacing, args.rd, args.rl, args.side_branches, args.side_drop)


if __name__ == "__main__":
    main()
