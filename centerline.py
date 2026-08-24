# -*- coding: utf-8 -*-
"""
centerline.py

Extracts the centerline (curve-skeleton) of a pulmonary artery mask stored
as a nifti volume.

Pipeline:
    1. binarize the mask (optionally isolating a single label value)
    2. clean it up: fill internal cavities, keep the largest component
    3. resample to an isotropic grid so the skeleton is not biased by the
       slice thickness
    4. thin the volume with Lee's 3D skeletonization
    5. turn the skeleton voxels into a graph, merge junction clusters,
       prune the short spurious side branches created by thinning, and cut
       the loops, which are always welds between touching vessels
    6. smooth the branches, within the digitization error, before measuring
       anything: a raw voxel path is ~10% longer than the vessel it follows
    7. estimate a local radius from the distance transform and number the
       branches -- along the main path, by Strahler order, or by
       diameter-defined Strahler, see `compute_orders`
    8. group the segments into elements and fit R_b, R_d and R_l as the
       slopes of the semi-log plots of the count, diameter and length
       against the order, with their confidence intervals. Two floors
       decide which orders enter that fit: --fit-min-voxels censors the
       thin end, where the distance transform has no range left, and
       --fit-min-branches the trunk end, where an order holds one element
       and that element may be a chain spliced across the hilum rather
       than a vessel -- see --element-flare, which names them

Outputs:
    --output        nifti centerline mask, on the input grid. Defaults to
                    <input>_centerline.nii.gz next to the input mask
    --csv           one row per centerline point (voxel + world mm + radius)
    --branches-csv  one row per segment (length, radii, all four orderings)
    --elements-csv  one row per element (same-order run of segments)
    --orders-csv    one row per order
    --ratios-csv    R_b / R_d / R_l with their confidence intervals
    --bifurcations-csv  one row per junction (the three angles, area ratio, Murray)
    --orphans-csv   one row per component outside the main tree, with its
                    wall-to-wall distance to it and where to look
    --reconnect-csv one row per severed pedicle: reclaimed or refused, and why
    --branches-csv-repaired  segments after the repair, with synthetic_fraction
    --repaired-mask the repaired mask, label 1 segmented / label 2 reclaimed
    --bridge-csv    the bridging curve: centerline recovered per dilation radius
    --sweep-csv     the ratios against the pruning strength, one row per k
    --vtk           legacy VTK polydata polylines, for Slicer / ParaView

Usage:
    python centerline.py --input artery.nii.gz
    python centerline.py --input seg.nii.gz --label 2 --csv points.csv --vtk cl.vtk
    python centerline.py --input artery.nii.gz --ordering strahler_dd --fit-orders 1 6
    python centerline.py --input av_seg.nii.gz --label 4 --compare-label 3 --bridge-sweep
    python centerline.py --input av_seg.nii.gz --label 4 --compare-label 3 --reconnect-severed
    python centerline.py --input artery.nii.gz --angle-sweep 0 0.5 1 1.5 2 2.5

See phantom.py for the calibration tree that says how much of what this
prints is anatomy and how much is the voxel size.
"""
import argparse
import itertools
import os
import textwrap
from collections import defaultdict, deque

import networkx as nx
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, zoom
from scipy.ndimage import label as connected_components
from scipy.optimize import brentq
from scipy.spatial import cKDTree
from scipy.stats import t as student_t
from skimage.draw import line_nd
from skimage.graph import MCP_Geometric
from skimage.morphology import skeletonize

# Half of the 26-neighbourhood: the lexicographically positive offsets, so
# every neighbouring pair is visited exactly once.
NEIGHBOR_OFFSETS = np.array([o for o in itertools.product((-1, 0, 1), repeat=3) if o > (0, 0, 0)], dtype=int)


# --------------------------------------------------------------------------- #
# mask preparation
# --------------------------------------------------------------------------- #
def load_mask(path, label=None):
    """Loads a nifti and returns (bool mask, affine, voxel spacing in mm)."""
    img = nib.load(path)
    data = np.asarray(img.dataobj)
    mask = (data == label) if label is not None else (data > 0)
    spacing = np.linalg.norm(img.affine[:3, :3], axis=0)
    return np.ascontiguousarray(mask, dtype=bool), img.affine, spacing


def component_volume_fraction(mask):
    """
    Number of connected components (26-connectivity) and the share of the
    mask volume held by the largest one.

    Volume is the optimistic view of fragmentation: a single fat trunk
    outweighs hundreds of broken peripheral twigs. The same fraction
    measured on centerline length, where every branch counts the same,
    is the honest one -- see `component_lengths`.
    """
    components, n_components = connected_components(mask, structure=np.ones((3, 3, 3), dtype=int))
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return n_components, float(sizes.max()) / float(mask.sum())


def resample_isotropic(mask, affine, spacing, target=None):
    """
    Resamples the mask to isotropic voxels.

    The mask is interpolated linearly as a float occupancy and re-thresholded
    at 0.5, not sampled with the nearest neighbour: nearest neighbour just
    replicates the coarse voxels, so the surface keeps the steps of the
    input grid and the thinning follows them. The 0.5 isosurface of the
    linear interpolant sits, to first order, where the original boundary was.

    Downsampling (a `--spacing` coarser than the input) is not anti-aliased
    and will drop the thinnest vessels; the default target is the finest
    input spacing, which only ever upsamples.

    Returns the new mask, its affine and the voxel->voxel mapping used later
    to project the centerline back onto the input grid. scipy's `grid_mode`
    convention is `in = (out + 0.5) / factor - 0.5`.
    """
    target = float(spacing.min()) if target is None else float(target)
    factors = spacing / target
    if np.allclose(factors, 1.0, atol=1e-3):
        return mask, affine, np.ones(3), spacing.copy()

    resampled = zoom(mask.astype(np.float32), factors, order=1, grid_mode=True, mode="nearest") > 0.5

    # voxel_in = M @ voxel_out + t, folded into the affine
    m = np.diag(1.0 / factors)
    t = 0.5 / factors - 0.5
    transform = np.eye(4)
    transform[:3, :3] = m
    transform[:3, 3] = t
    return resampled, affine @ transform, factors, np.full(3, target)


# --------------------------------------------------------------------------- #
# skeleton -> graph
# --------------------------------------------------------------------------- #
def build_voxel_graph(skeleton, spacing):
    """
    Builds a graph whose nodes are the skeleton voxels and whose edges link
    26-neighbours, weighted by the physical distance in mm.

    Returns (graph, positions) with positions[i] the voxel coordinates of node i.
    """
    coords = np.argwhere(skeleton)
    index = np.full(np.array(skeleton.shape) + 2, -1, dtype=np.int64)
    index[tuple((coords + 1).T)] = np.arange(len(coords))

    graph = nx.Graph()
    graph.add_nodes_from(range(len(coords)))
    for offset in NEIGHBOR_OFFSETS:
        neighbors = index[tuple((coords + 1 + offset).T)]
        found = np.nonzero(neighbors >= 0)[0]
        weight = float(np.linalg.norm(offset * spacing))
        graph.add_weighted_edges_from((int(a), int(b), weight) for a, b in zip(found, neighbors[found]))

    return graph, coords.astype(float)


def update_edge_weights(graph, positions, spacing):
    """
    Recomputes every edge length from the current node positions.

    Contracting a junction moves a node to the centroid of its cluster, and
    the edges reaching it have to follow -- otherwise a branch can end up
    shorter than the straight line between its two ends.
    """
    for u, v, data in graph.edges(data=True):
        data["weight"] = float(np.linalg.norm((positions[u] - positions[v]) * spacing))


def contract_junction_clusters(graph, positions, mask):
    """
    Merges each cluster of touching degree>=3 voxels into a single node.

    Thinning leaves small clumps at bifurcations; without this every clump
    would be reported as several nearby junctions. The merged node sits at
    the centroid of the cluster, unless the centroid falls outside the mask
    (concave junction) in which case one of the original voxels is kept.
    """
    junctions = [n for n, d in graph.degree() if d >= 3]
    for cluster in list(nx.connected_components(graph.subgraph(junctions))):
        if len(cluster) < 2:
            continue
        cluster = list(cluster)
        keep = cluster[0]
        centroid = positions[cluster].mean(axis=0)
        for other in cluster[1:]:
            nx.contracted_nodes(graph, keep, other, self_loops=False, copy=False)
        if mask[tuple(np.rint(centroid).astype(int))]:
            positions[keep] = centroid
    return graph


def extract_branches(graph):
    """
    Splits the graph into branches: maximal paths whose interior nodes all
    have degree 2. Returns a list of node paths.
    """
    key_nodes = {n for n, d in graph.degree() if d != 2}
    branches = []
    seen_edges = set()

    for start in key_nodes:
        for neighbor in graph.neighbors(start):
            if frozenset((start, neighbor)) in seen_edges:
                continue
            seen_edges.add(frozenset((start, neighbor)))
            path = [start, neighbor]
            previous, current = start, neighbor
            while current not in key_nodes:
                nxt = [n for n in graph.neighbors(current) if n != previous]
                if not nxt:
                    break
                seen_edges.add(frozenset((current, nxt[0])))
                path.append(nxt[0])
                previous, current = current, nxt[0]
            branches.append(path)

    # Whatever is left is made of pure degree-2 rings, which have no key node.
    remaining = set(graph.nodes) - {n for path in branches for n in path}
    while remaining:
        start = next(iter(remaining))
        if graph.degree(start) == 0:
            branches.append([start])
            remaining.discard(start)
            continue
        path = [start]
        previous, current = start, next(iter(graph.neighbors(start)), None)
        while current is not None and current != start:
            path.append(current)
            nxt = [n for n in graph.neighbors(current) if n != previous]
            previous, current = current, (nxt[0] if nxt else None)
        path.append(start)
        branches.append(path)
        remaining -= set(path)

    return branches


def path_length(graph, path):
    """Physical length of a node path, in mm."""
    return float(sum(graph[a][b]["weight"] for a, b in zip(path, path[1:])))


def prune_spurs(graph, radii, min_length, radius_factor=1.0, max_iterations=20):
    """
    Iteratively removes the terminal branches that thinning grows on every
    bump of the surface.

    A terminal branch goes if it is shorter than `min_length` mm or than
    `radius_factor` times the vessel radius at the junction it hangs from --
    the second rule is what clears the fans that appear inside wide vessels
    and at the flat ends of a cropped mask. Branches whose both ends are
    free are left alone (they are whole components, handled by
    `drop_small_components`).
    """
    total_removed = 0
    for _ in range(max_iterations):
        doomed = set()
        for path in extract_branches(graph):
            ends_free = (graph.degree(path[0]) == 1, graph.degree(path[-1]) == 1)
            if sum(ends_free) != 1:
                continue
            junction = path[-1] if ends_free[0] else path[0]
            threshold = max(min_length, radius_factor * radii[junction])
            if path_length(graph, path) >= threshold:
                continue
            # keep the junction end, drop the free side
            doomed.update(path[:-1] if ends_free[0] else path[1:])
        if not doomed:
            break
        graph.remove_nodes_from(doomed)
        total_removed += len(doomed)
    return total_removed


def component_lengths(graph):
    """
    Total branch length of every connected component, longest first.

    Measured on the skeleton of the whole mask, before it is restricted to
    its main component: the point is precisely to weigh what is about to be
    thrown away. Lengths are the raw voxel paths, which inflates them all by
    the same ~10%, so their ratio is unaffected.

    Returns a list of (length in mm, set of nodes).
    """
    parts = []
    for nodes in nx.connected_components(graph):
        length = float(sum(d["weight"] for _, _, d in graph.subgraph(nodes).edges(data=True)))
        parts.append((length, nodes))
    return sorted(parts, key=lambda part: -part[0])


def break_cycles(graph, radii, max_breaks=500):
    """
    Cuts every cycle of the skeleton, at its weakest edge.

    An arterial tree has no anastomosis, so a loop is always an artefact:
    two vessels running side by side that partial volume welded together, or
    an artery and a vein left fused by the segmentation. Left in place, a
    loop is not merely cosmetic -- it makes the tree unorderable. Strahler
    counts up from the leaves through a supposed DAG, and a branch caught in
    a loop has no well-defined depth, so the orders downstream of it are
    quietly wrong rather than visibly missing.

    Each cycle of the cycle basis loses the edge whose thinner endpoint has
    the smallest radius, which is where two vessels are most likely to be
    merely touching. Shortest cycles go first: they are the tightest welds
    and cutting them often opens the longer ones too. The number of cuts is
    reported, and the cycle count before cutting stays in the quality
    metrics -- this fixes the ordering, it does not fix the mask.

    Returns the list of (u, v, radius) removed.
    """
    broken = []
    while len(broken) < max_breaks:
        cycles = nx.cycle_basis(graph)
        if not cycles:
            break
        cycle = min(cycles, key=len)
        edges = [(u, v) for u, v in zip(cycle, cycle[1:] + cycle[:1]) if graph.has_edge(u, v)]
        if not edges:
            break
        u, v = min(edges, key=lambda e: min(radii[e[0]], radii[e[1]]))
        graph.remove_edge(u, v)
        broken.append((int(u), int(v), float(min(radii[u], radii[v]))))
    return broken


def drop_small_components(graph, min_length):
    """Removes connected components whose total length is below `min_length` mm."""
    removed = 0
    for component in list(nx.connected_components(graph)):
        subgraph = graph.subgraph(component)
        length = float(sum(d["weight"] for _, _, d in subgraph.edges(data=True)))
        if length < min_length:
            graph.remove_nodes_from(component)
            removed += 1
    return removed


# --------------------------------------------------------------------------- #
# branch ordering
# --------------------------------------------------------------------------- #
def order_branches(graph, branches, positions, radii, root_voxel=None):
    """
    Orients every branch away from a root and labels it with its generation.

    The root defaults to the free end with the largest radius, i.e. the
    trunk of the pulmonary artery. Returns the branches as (path, generation)
    with each path running proximal -> distal.
    """
    if graph.number_of_nodes() == 0:
        return []

    adjacency = defaultdict(list)
    for branch_id, path in enumerate(branches):
        adjacency[path[0]].append((branch_id, path[-1]))
        adjacency[path[-1]].append((branch_id, path[0]))

    # the root has to be a branch extremity, otherwise the traversal below
    # starts inside a branch and reaches nothing
    candidates = list(adjacency)
    if root_voxel is not None:
        distances = np.linalg.norm(positions[candidates] - np.asarray(root_voxel, float), axis=1)
        root = candidates[int(distances.argmin())]
    else:
        leaves = [n for n in candidates if graph.degree(n) == 1] or candidates
        root = max(leaves, key=lambda n: radii[n])

    ordered = [None] * len(branches)
    visited = set()
    queue = deque([(root, 0)])
    seen_nodes = {root}
    while queue:
        node, generation = queue.popleft()
        for branch_id, other in adjacency[node]:
            if branch_id in visited:
                continue
            visited.add(branch_id)
            path = branches[branch_id]
            ordered[branch_id] = (path if path[0] == node else path[::-1], generation)
            if other not in seen_nodes:
                seen_nodes.add(other)
                queue.append((other, generation + 1))

    # Components that the root cannot reach keep their original orientation.
    for branch_id, path in enumerate(branches):
        if ordered[branch_id] is None:
            ordered[branch_id] = (path, -1)
    return ordered


# --------------------------------------------------------------------------- #
# anatomical analysis
# --------------------------------------------------------------------------- #
def smooth_centerline(ordered, world, voxel_size, iterations=20, max_shift=None):
    """
    Smooths every branch, junctions pinned in place.

    A digitized centerline wobbles by up to half a voxel around the true
    axis, and measuring it step by step accumulates that wobble: on a
    diagonal tube the raw polyline is ~25% longer than the vessel. Lengths,
    tortuosities and directions must therefore be read on a smoothed curve.

    Laplacian smoothing is applied, each point kept within `max_shift` of
    where it started (half a voxel by default) so the correction stays
    within the digitization error and real curvature survives. The two
    endpoints never move, which keeps the branches connected at junctions.
    """
    max_shift = 0.5 * voxel_size if max_shift is None else max_shift
    smooth = world.copy()
    for nodes, _ in ordered:
        if len(nodes) < 3:
            continue
        anchor = world[nodes]
        points = anchor.copy()
        for _ in range(iterations):
            moved = points.copy()
            moved[1:-1] += 0.5 * (points[:-2] + points[2:] - 2.0 * points[1:-1])
            offset = moved - anchor
            distance = np.linalg.norm(offset, axis=1, keepdims=True)
            excess = distance > max_shift
            points = np.where(excess, anchor + offset * (max_shift / np.maximum(distance, 1e-9)), moved)
        smooth[nodes] = points
    return smooth


def polyline_length(points):
    """Length of a polyline, in the units of its coordinates."""
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


# How far from the junction node the direction fit is allowed to start, in
# multiples of the local junction radius. Inside that ball the centerline is
# not the vessel axis: the daughters share the blob left by the maximal
# inscribed sphere, and their skeletons still lean along the parent before
# peeling off. Fitting from distance 0 therefore reads the bend, not the
# branch, and inflates every bifurcation angle by tens of degrees. One
# parental radius is the same rule of thumb `trunk_calibre` trims with; use
# --angle-sweep to see the plateau it sits on.
DIRECTION_OFFSET = 1.0


def branch_geometry(nodes, world, radii, junction_radius, voxel_size, from_start=True,
                    direction_offset=DIRECTION_OFFSET):
    """
    Direction and calibre of a branch as seen from one of its ends.

    Both are measured away from that end, because at a junction the branches
    merge into a single blob: the radius there is inflated by the
    neighbouring vessels and the direction still bends out of the parent.
    The two use different scales, and both are clamped to the branch so a
    wide vessel is never measured at its far end:

    - the direction is fitted over 2.5 local radii, starting
      `direction_offset` radii away from the end so the blob is left behind
      entirely, and stopping one radius short of the far end so the blob
      there is not picked up instead. Over a couple of voxels only it would
      snap to the axes of the grid, which piles the bifurcation angles up at
      90 degrees. It is the first principal component of the points in that
      window, not the chord to its last point, so every point contributes.
    - the calibre is the median radius over [1, 2] local radii (at most the
      proximal half of the branch), just past the junction blob and close
      enough that the vessel has not tapered yet.

    A branch shorter than the offset has no clean window to fit: rather than
    report a direction measured inside the blob and let it pass for a
    measurement, the fit falls back to the whole branch and the third return
    value goes False, so the bifurcations resting on it can be counted and
    excluded. That count is part of the result -- pushing the offset out
    buys angles at the cost of junctions, and both have to be quoted.

    Returns (unit direction leaving the end, radius in mm, direction is clean).
    """
    order = np.asarray(nodes if from_start else nodes[::-1])
    points = world[order]
    distance = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    total = float(distance[-1])

    reach = max(2.5 * float(junction_radius), 3.0 * voxel_size)
    start = float(direction_offset) * float(junction_radius)
    # stop short of the blob at the other end, but never collapse the window
    stop = min(start + reach, max(total - float(junction_radius), start + 3.0 * voxel_size))
    window = (distance >= start) & (distance <= stop)
    clean = bool(window.sum() >= 3)
    if not clean:
        # too short to clear the blob: fall back to the old window, which
        # starts at the node, and flag the direction as contaminated
        window = distance <= min(reach, max(0.5 * total, 3.0 * voxel_size))
    sample = points[window]
    if len(sample) < 2:
        sample = points[:2]
    axis = np.linalg.svd(sample - sample.mean(axis=0), full_matrices=False)[2][0]
    # principal components have no sign: point it away from the junction
    direction = axis * np.sign(np.dot(axis, sample[-1] - sample[0]) or 1.0)

    low = min(max(float(junction_radius), 2.0 * voxel_size), max(0.25 * total, 2.0 * voxel_size))
    high = min(max(2.0 * float(junction_radius), low + 2.0 * voxel_size),
               max(0.5 * total, low + 2.0 * voxel_size))
    calibre_window = (distance >= min(low, total)) & (distance <= high)
    if not calibre_window.any():
        calibre_window = distance >= min(low, total)
    return direction, float(np.median(radii[order[calibre_window]])), clean


def trunk_calibre(nodes, world, radii, head_radius, tail_radius, voxel_size):
    """
    Median radius of a branch with the junction blobs cut off, in mm.

    This is the branch calibre every order-wise statistic should use. The
    plain mean over the branch (`mean_radius_mm`) averages in the ends, where
    the maximal inscribed sphere is not the vessel at all but the cavity of
    the bifurcation, which fills with the neighbouring vessels: it inflates
    the calibre of short branches far more than long ones, i.e. it inflates
    the high orders more than the low ones, which is exactly the direction
    that biases the slope of log(D) against the order.

    Each end is trimmed by its own local junction radius -- one parental
    radius, the rule of thumb -- clamped to 20% of the branch so something
    always survives on a short branch. A free end is not a junction: pass
    `head_radius` or `tail_radius` = 0 there and only the last voxel, whose
    distance transform is unreliable, is dropped.
    """
    order = np.asarray(nodes)
    points = world[order]
    distance = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    total = float(distance[-1])

    head_cut = min(max(float(head_radius), voxel_size), 0.2 * total)
    tail_cut = min(max(float(tail_radius), voxel_size), 0.2 * total)
    window = (distance >= head_cut) & (distance <= total - tail_cut)
    if not window.any():
        window = np.zeros(len(order), dtype=bool)
        window[len(order) // 2] = True
    return float(np.median(radii[order[window]]))


def murray_exponent(parent_radius, child_radii):
    """
    Solves sum(r_child^n) = r_parent^n, the exponent of Murray's law.

    n = 3 is the theoretical optimum for laminar flow, n = 2 means the
    cross-section is conserved across the bifurcation. Returns None when
    the radii make the equation unsolvable (a child wider than its parent).
    """
    ratios = np.asarray(child_radii, float) / parent_radius
    if parent_radius <= 0 or ratios.max() >= 1.0 or ratios.sum() <= 1.0:
        return None
    return float(brentq(lambda n: np.sum(ratios ** n) - 1.0, 1e-3, 50.0))


def branch_table(graph, ordered, smooth, radii, voxel_size, direction_offset=DIRECTION_OFFSET,
                 synthetic=None):
    """
    Per-branch measurements, in branch order (proximal -> distal), read on
    the smoothed centerline.

    Tortuosity is the ratio of the path length to the straight distance
    between the two ends: 1.0 is a straight branch. The proximal and distal
    calibres are measured away from the junctions (see `branch_geometry`),
    unlike `mean_radius_mm` which averages the whole branch.

    `synthetic` is the boolean map of voxels a repair put back, and
    `synthetic_fraction` is the share of a branch's skeleton nodes that fall
    in them. Skeleton nodes sit roughly one voxel apart, so that share is a
    length fraction to within the sampling. It is 0 for every branch when no
    repair ran, which is what keeps the ordinary path unchanged.
    """
    table = []
    for branch_id, (nodes, generation) in enumerate(ordered):
        points = smooth[nodes]
        length = polyline_length(points)
        chord = float(np.linalg.norm(points[-1] - points[0]))
        values = radii[nodes]

        share = 0.0 if synthetic is None else float(synthetic[nodes].mean())
        head_axis, head_calibre, head_clean = branch_geometry(
            nodes, smooth, radii, radii[nodes[0]], voxel_size, direction_offset=direction_offset)
        tail_axis, tail_calibre, tail_clean = branch_geometry(
            nodes, smooth, radii, radii[nodes[-1]], voxel_size, from_start=False,
            direction_offset=direction_offset)
        # a free end carries no junction blob, so nothing has to be cut there
        head_junction = radii[nodes[0]] if graph.degree(nodes[0]) > 1 else 0.0
        tail_junction = radii[nodes[-1]] if graph.degree(nodes[-1]) > 1 else 0.0
        calibre = trunk_calibre(nodes, smooth, radii, head_junction, tail_junction, voxel_size)

        table.append({
            "branch_id": branch_id,
            "bfs_generation": generation,
            "n_points": len(nodes),
            "length_mm": length,
            "chord_mm": chord,
            "tortuosity": length / chord if chord > 0 else 1.0,
            "calibre_mm": calibre,
            "mean_radius_mm": float(values.mean()),
            "min_radius_mm": float(values.min()),
            "max_radius_mm": float(values.max()),
            "proximal_calibre_mm": head_calibre,
            "distal_calibre_mm": tail_calibre,
            "tip_radius_mm": float(radii[nodes[-1]]),
            "is_terminal": int(graph.degree(nodes[-1]) == 1),
            "nodes": nodes,
            "head_axis": head_axis,
            "tail_axis": tail_axis,
            # tail_axis looks back UP the branch, towards its start: the
            # direction the branch leaves a junction by is -tail_axis
            "head_axis_clean": head_clean,
            "tail_axis_clean": tail_clean,
            "synthetic_fraction": share,
        })
    return table


def compute_orders(table):
    """
    Numbers the branches three ways, since the natural traversal order is
    not the anatomical one.

    - `bfs_generation` counts junctions from the root, so a trunk giving off
      collaterals is renumbered at every one of them: the interlobar artery
      ends up several "generations" deep while still being the same vessel.
    - `generation` follows the main path: at a junction the widest daughter
      inherits the parent's number and only the others are incremented, so
      the number tracks the vessel, not the count of junctions passed.
    - `strahler` orders from the periphery instead: a tip is 1, and a
      junction of two branches of equal order n yields n+1, otherwise the
      largest order carries through.
    - `strahler_dd` is the diameter-defined refinement of the previous one,
      computed separately by `diameter_defined_strahler`, and is the ordering
      to fit the ratios on: it does not inherit the depth of leaves that are
      truncation artefacts.

    With `generation` or `strahler`, the mean calibre must vary
    monotonically -- a violation means the tree leaks into a neighbouring
    structure or two vessels have been fused.
    """
    starts = defaultdict(list)
    for entry in table:
        starts[entry["nodes"][0]].append(entry["branch_id"])
    for entry in table:
        entry["children"] = [c for c in starts.get(entry["nodes"][-1], []) if c != entry["branch_id"]]
        entry["generation"] = None

    by_depth = sorted(table, key=lambda e: e["bfs_generation"])
    for entry in by_depth:
        if entry["generation"] is None:
            entry["generation"] = 0 if entry["bfs_generation"] == 0 else -1
        if entry["generation"] < 0:
            continue
        children = entry["children"]
        main = max(children, key=lambda c: table[c]["proximal_calibre_mm"], default=None)
        for child in children:
            table[child]["generation"] = entry["generation"] + (0 if child == main else 1)

    for entry in reversed(by_depth):
        orders = sorted((table[c].get("strahler", 1) for c in entry["children"]), reverse=True)
        if not orders:
            entry["strahler"] = 1
        elif len(orders) > 1 and orders[0] == orders[1]:
            entry["strahler"] = orders[0] + 1
        else:
            entry["strahler"] = orders[0]
    return table


def diameter_defined_strahler(table, max_iterations=15):
    """
    Re-orders the tree with the diameter-defined Strahler scheme of Jiang,
    Kassab and Fung (1994), writing the result into `strahler_dd`.

    Classic Strahler sets the order from the topological depth below a
    branch, so it is only as good as the leaves -- and in an in-vivo mask the
    leaves are not the real terminals, they are wherever the segmentation ran
    out of contrast. Two vessels of identical calibre end up several orders
    apart because one happened to be truncated earlier. The diameter-defined
    variant breaks that dependence: a parent is promoted above its largest
    daughter only if its diameter clears the threshold that separates the two
    orders, so the order tracks calibre and truncation costs one order at
    most, locally.

    The iteration: start from classic Strahler, take the mean and SD of the
    diameter within each order, put the boundary between orders n and n+1 at
    (Dn + SDn + Dn+1 - SDn+1) / 2, re-order the whole tree against those
    boundaries, and repeat until nothing moves.

    Two deliberate departures, both worth checking against the paper before
    any of this is quoted:
      - it runs on segments, not on elements. Kassab orders elements, but the
        elements are themselves defined by the ordering, so doing it properly
        means nesting the two fixed points. The segment diameters within one
        element are close, so the boundaries move little, but this is not the
        published algorithm.
      - at the top order there is no n+1 and therefore no boundary; the
        classic rule (promote when the two largest daughters tie) is used
        there.

    Returns True if the iteration converged.
    """
    by_depth = sorted(table, key=lambda e: e["bfs_generation"])
    orders = {entry["branch_id"]: entry["strahler"] for entry in table}
    seen, converged = [], False

    for _ in range(max_iterations):
        groups = defaultdict(list)
        for entry in table:
            groups[orders[entry["branch_id"]]].append(2.0 * entry["calibre_mm"])
        stats = {order: (float(np.mean(values)),
                         float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
                 for order, values in groups.items()}
        boundary = {}
        for order in stats:
            if order + 1 in stats:
                (mean_low, sd_low), (mean_high, sd_high) = stats[order], stats[order + 1]
                boundary[order] = 0.5 * (mean_low + sd_low + mean_high - sd_high)

        updated = {}
        for entry in reversed(by_depth):
            children = entry["children"]
            if not children:
                updated[entry["branch_id"]] = 1
                continue
            child_orders = sorted((updated.get(c, 1) for c in children), reverse=True)
            top = child_orders[0]
            if top in boundary:
                promote = 2.0 * entry["calibre_mm"] > boundary[top]
            else:
                promote = len(child_orders) > 1 and child_orders[1] == top
            updated[entry["branch_id"]] = top + (1 if promote else 0)

        if updated == orders:
            converged = True
            break
        signature = tuple(updated[entry["branch_id"]] for entry in table)
        orders = updated
        if signature in seen:  # two-cycle: the boundaries flip a branch back and forth
            break
        seen.append(signature)

    for entry in table:
        entry["strahler_dd"] = orders[entry["branch_id"]]
    return converged


def flare(entry):
    """
    The distal calibre over the proximal one, along a segment or an element.

    Near 1 for a vessel: an artery tapers, gently, and it never widens by
    much over its own length. Well above 1 only when the run of segments
    grouped under one order is not one vessel -- a chain that keeps its
    Strahler order across the hilum comes out as a single element several
    times wider at its end than at its start, because it is four vessels
    end to end and not one.

    That is the direct symptom of a heterogeneous aggregation, and a more
    specific one than tortuosity: a real vessel can wander around a
    junction and still be a vessel, but it cannot triple its calibre.
    """
    head = entry.get("proximal_calibre_mm") or 0.0
    if head <= 0:
        return float("nan")
    return float(entry["distal_calibre_mm"] / head)


def flared(entry, limit):
    """
    Whether `entry` is an element that flares, and so is not one vessel.

    The restriction to elements of two segments or more is what makes the
    check specific rather than merely alarming. A one-segment element was
    aggregated from nothing: whatever its two end calibres do, the answer
    cannot be that the grouping put two vessels in one row. Left in, those
    rows dominate the list -- on the LIDC example 46 of 52 hits are
    single-segment order-1 twigs whose end calibres disagree for reasons
    that have nothing to do with elements -- and the six that mean something
    are lost in them.

    The flare itself is still measured and written out for every element,
    since it is a measurement; only the flag is narrowed.
    """
    return entry.get("n_segments", 1) >= 2 and flare(entry) > limit


def build_elements(table, order_key, smooth):
    """
    Groups consecutive segments of the same order into elements.

    A segment is the piece of vessel between two bifurcations. An element is
    the whole run of segments that keep the same order, which happens every
    time a small lateral branch leaves a trunk without raising the trunk's
    order: the trunk is cut into two segments there, but anatomically it is
    one vessel.

    The distinction is not cosmetic. Counting elements instead of segments
    changes both N_n and the mean length L_n, so it changes R_b and R_l -- in
    the case of L_n by whatever the mean number of segments per element is,
    which is not small. Horsfield-ordered lengths in the literature are
    normally elemental; comparing segmental lengths against them understates
    L_n at every order. Hence both are computed here and neither is implied.

    Returns element dicts carrying the same keys `order_summary` reads.
    """
    order_of = {entry["branch_id"]: entry[order_key] for entry in table}
    successor = {}
    for entry in table:
        same = [c for c in entry["children"] if order_of[c] == entry[order_key]]
        if not same:
            continue
        # a Strahler junction gives at most one child the parent's order; the
        # diameter-defined variant can give it to both, and then the element
        # follows the wider daughter
        successor[entry["branch_id"]] = max(same, key=lambda c: table[c]["proximal_calibre_mm"])

    tails = set(successor.values())
    elements = []
    for entry in table:
        if entry["branch_id"] in tails:
            continue
        members, current = [], entry["branch_id"]
        while current is not None:
            members.append(table[current])
            current = successor.get(current)
            if current is not None and len(members) > len(table):
                break  # a cycle in the successor map would loop forever
        nodes = [n for member in members for n in member["nodes"]]
        length = float(sum(member["length_mm"] for member in members))
        chord = float(np.linalg.norm(smooth[nodes[-1]] - smooth[nodes[0]]))
        weights = np.array([member["length_mm"] for member in members])
        weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(members), 1.0 / len(members))
        element = {
            "element_id": len(elements),
            "n_segments": len(members),
            "branch_ids": [member["branch_id"] for member in members],
            "order": entry[order_key],
            "length_mm": length,
            "chord_mm": chord,
            "tortuosity": length / chord if chord > 0 else 1.0,
            "calibre_mm": float(np.dot(weights, [m["calibre_mm"] for m in members])),
            "mean_radius_mm": float(np.dot(weights, [m["mean_radius_mm"] for m in members])),
            "proximal_calibre_mm": members[0]["proximal_calibre_mm"],
            "distal_calibre_mm": members[-1]["distal_calibre_mm"],
            "tip_radius_mm": members[-1]["tip_radius_mm"],
            "is_terminal": members[-1]["is_terminal"],
            "nodes": nodes,
            "synthetic_fraction": float(np.dot(weights, [m["synthetic_fraction"] for m in members])),
        }
        element["flare"] = flare(element)
        elements.append(element)
    return elements


def vector_angle(u, v):
    """Angle between two vectors, in degrees."""
    cosine = float(np.dot(u, v)) / (float(np.linalg.norm(u)) * float(np.linalg.norm(v)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def analyze_bifurcations(table, order_key, min_radius, positions=None, factors=None):
    """
    Measures every bifurcation: the parent branch that ends there and the
    daughters that leave it. Radii and directions come from the branch
    table, i.e. measured a couple of radii away from the junction.

    All three angles of the junction are reported, not just the one between
    the daughters, because on their own the daughter-daughter angles cannot
    be told apart from a convention error. The parent leaves the junction
    along -tail_axis (tail_axis is fitted looking back up the branch), and
    getting that sign wrong turns every parent-daughter angle into its
    supplement -- a clean, constant offset that no daughter-daughter
    statistic would reveal.

    `planarity_defect_deg` is (parent-d1) + (parent-d2) - (d1-d2). For any
    three unit vectors it is >= 0, and it is 0 exactly when the three are
    coplanar with the parent lying between the daughters, which is what a
    bifurcation looks like. Its median over the tree is therefore the
    self-consistency check on the angle measurement as a whole: a few
    degrees is anatomy, and a large positive median is the signature of a
    parent direction that points somewhere the geometry does not allow --
    a reversed sign lands it near 360 - 2*theta.

    `angle_clean` flags the junctions where all three directions were
    fitted clear of the junction blob (see `branch_geometry`). The angles of
    the others are measured inside the bend and read far too wide; they are
    kept in the table but must be counted separately.

    `well_resolved` flags the junctions where the parent and every daughter
    are wider than `min_radius` (a few voxels). Below that the radii
    saturate on the voxel size and the derived quantities -- the area ratio
    and above all Murray's exponent, which is a ratio raised to a power --
    stop meaning anything.

    With `positions` and `factors` each junction also carries its voxel on
    the input grid. The angles are a distribution and the tails of that
    distribution are the interesting part: a junction whose planarity defect
    runs to tens of degrees is not a bifurcation seen from a bad angle, it is
    usually two neighbouring vessels welded by partial volume, and the only
    way to settle that is to open the coordinate in a viewer.
    """
    bifurcations = []
    for parent in table:
        children = [table[c] for c in parent["children"]]
        if len(children) < 2:
            continue
        parent_radius = parent["distal_calibre_mm"]
        child_radii = [c["proximal_calibre_mm"] for c in children]
        # the parent arrives at the junction along -tail_axis; both daughters
        # leave it along their own head_axis. All three now point out of the
        # junction the same way, which is what makes them comparable
        parent_axis = -np.asarray(parent["tail_axis"], float)

        angle = parent_1 = parent_2 = defect = None
        if len(children) == 2:
            first, second = children[0]["head_axis"], children[1]["head_axis"]
            angle = vector_angle(first, second)
            parent_1 = vector_angle(parent_axis, first)
            parent_2 = vector_angle(parent_axis, second)
            defect = parent_1 + parent_2 - angle

        clean = int(parent["tail_axis_clean"] and all(c["head_axis_clean"] for c in children))
        node = parent["nodes"][-1]
        i, j, k = (np.rint((positions[node] + 0.5) / factors - 0.5).astype(int)
                   if positions is not None and factors is not None else (-1, -1, -1))
        bifurcations.append({
            "node": node, "i": int(i), "j": int(j), "k": int(k),
            "order": parent[order_key],
            "n_children": len(children),
            "parent_radius_mm": parent_radius,
            "min_child_radius_mm": float(min(child_radii)),
            "area_ratio": float(np.sum(np.square(child_radii)) / parent_radius ** 2) if parent_radius > 0 else None,
            "asymmetry": float(min(child_radii) / max(child_radii)) if max(child_radii) > 0 else None,
            "murray_exponent": murray_exponent(parent_radius, child_radii),
            "angle_deg": angle,
            "angle_parent_first_deg": parent_1,
            "angle_parent_second_deg": parent_2,
            "planarity_defect_deg": defect,
            "angle_clean": clean,
            "well_resolved": int(min(parent_radius, *child_radii) >= min_radius),
        })
    return bifurcations


def order_summary(table, order_key, flare_limit=1.5):
    """
    Aggregates the branch measurements order by order.

    `flare_limit` only decides what gets counted in `n_flared`; it changes
    no mean. An order whose elements flare is one whose mean diameter is a
    summary of things that are not each a vessel, and that is worth having
    in the per-order table next to the mean it explains.

    `calibre_spread` is the 90th percentile of the member calibres over the
    10th, and it is the
    other way an order can fail to be a summary -- not one row that is two
    vessels, but several rows that are not the same size. The two are
    distinct and neither implies the other: flare is measured ALONG a row,
    spread ACROSS the rows of an order. Segments never flare by definition
    (a segment was aggregated from nothing) yet a segment order can spread
    badly, which is the case the flare check cannot see.

    Percentiles and not max over min: an order of 1274 segments has two
    extreme rows whose ratio is 8 or 11 and says nothing about the other
    1272. Measured that way every order of the bundled example looks
    broken; measured on the 10-90 band they run 1.24 to 1.82 and the
    statistic starts discriminating.

    A diameter-defined Strahler ordering promotes on calibre, so its orders
    are supposed to come out narrow in spread. One that does not is telling
    you the promotion failed there.
    """
    rows = []
    for order in sorted({b[order_key] for b in table}):
        branches = [b for b in table if b[order_key] == order]
        terminal = [b for b in branches if b["is_terminal"]]
        lengths = np.array([b["length_mm"] for b in branches])
        calibres = np.array([b["calibre_mm"] for b in branches])
        flares = [f for f in (flare(b) for b in branches) if np.isfinite(f)]
        chains = sum(1 for b in branches if flared(b, flare_limit))
        rows.append({
            "order": order,
            "n_branches": len(branches),
            "n_terminal": len(terminal),
            "total_length_mm": float(lengths.sum()),
            "mean_length_mm": float(lengths.mean()),
            "sd_length_mm": float(lengths.std(ddof=1)) if len(lengths) > 1 else 0.0,
            "mean_diameter_mm": float(2.0 * calibres.mean()),
            "sd_diameter_mm": float(2.0 * calibres.std(ddof=1)) if len(calibres) > 1 else 0.0,
            "mean_radius_mm": float(np.mean([b["mean_radius_mm"] for b in branches])),
            "mean_proximal_calibre_mm": float(np.mean([b["proximal_calibre_mm"] for b in branches])),
            "mean_distal_calibre_mm": float(np.mean([b["distal_calibre_mm"] for b in branches])),
            "mean_tortuosity": float(np.mean([b["tortuosity"] for b in branches])),
            "calibre_spread": (float(np.percentile(calibres, 90) / np.percentile(calibres, 10))
                               if np.percentile(calibres, 10) > 0 else None),
            "max_flare": float(max(flares)) if flares else None,
            "n_flared": chains,
            "mean_tip_radius_mm": float(np.mean([b["tip_radius_mm"] for b in terminal])) if terminal else None,
        })
    return rows


# The sign of one step of each ordering: +1 when the order grows towards the
# trunk (Strahler), -1 when it grows towards the periphery (generation). The
# branching ratios are all defined per step towards the trunk, so the fitted
# slopes have to be flipped for the peripheral orderings.
ORDER_DIRECTION = {"strahler": 1, "strahler_dd": 1, "generation": -1, "bfs_generation": -1}

# How far an order may span in calibre, 90th percentile of its members over
# the 10th, before its mean stops being a summary of one population. Not a filter: an order
# this wide is reported and left in, because unlike flare it has no single
# culprit row to point at and the right response depends on the tree.
ORDER_SPREAD_LIMIT = 2.0


def semilog_fit(orders, values):
    """
    Least squares of log10(values) on the order.

    Returns the slope, its 95% confidence interval and R2. The interval is
    the textbook t interval on the slope of a straight line, which with the
    five or six usable orders of an in-vivo tree is wide -- that width is the
    result, not a failure: it is what says whether a ratio of 1.55 and one of
    1.75 can be told apart at this resolution.
    """
    x = np.asarray(orders, float)
    y = np.log10(np.asarray(values, float))
    n = len(x)
    scatter = float(((x - x.mean()) ** 2).sum())
    if n < 3 or scatter <= 0 or not np.isfinite(y).all():
        return None

    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    ss_res = float((residual ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    half = float(student_t.ppf(0.975, n - 2)) * np.sqrt(ss_res / (n - 2) / scatter)
    return {
        "slope": float(slope), "intercept": float(intercept),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "slope_low": float(slope - half), "slope_high": float(slope + half),
        "n_orders": n,
    }


RATIO_KEYS = (("R_b", "n_branches", -1), ("R_d", "mean_diameter_mm", 1), ("R_l", "mean_length_mm", 1))


def contiguous_run(rows):
    """
    The run of consecutive orders holding the lowest one that survived.

    Censoring rows out of the middle of the range leaves a hole, and a hole
    is not neutral. The slope is fitted against the order NUMBER, so a
    surviving island above the gap sits far from the mean of the retained
    orders and carries leverage in proportion to that distance squared. Drop
    order 5 from 1..6 and the lone point at 6 is 2.8 orders off a mean of
    3.2: it supplies 7.84 of the 14.8 total scatter, more than half the
    weight deciding the slope, and it is by construction the least reliable
    point in the set -- it survived only because the filter that removed its
    neighbour did not reach it. That is how a filter meant to protect the
    fit ends up handing it to the order it was protecting against.

    The anchor is the lowest order, not the longest run. For a Strahler
    ordering the low end is the periphery, where an order holds hundreds of
    branches and its mean is worth something, while the high end is the
    trunk, where it holds one or two. Anchoring low keeps the reliable
    block; picking whichever run happens to be longer would make the range
    depend on where the holes fell, which is exactly the data-dependence the
    pre-specification is meant to rule out.
    """
    if not rows:
        return rows
    ordered = sorted(rows, key=lambda row: row["order"])
    keep = [ordered[0]]
    for row in ordered[1:]:
        if row["order"] != keep[-1]["order"] + 1:
            break
        keep.append(row)
    return keep


def branching_ratios(summary, ordering, order_range=None, min_diameter=0.0, prespecified=False,
                     min_branches=3):
    """
    Horsfield's three ratios, read as the slopes of the semi-log plots of the
    branch count, the mean diameter and the mean length against the order.

    R_b = 10^-slope(log N), R_d = 10^slope(log D), R_l = 10^slope(log L), all
    per step towards the trunk. Each is returned with the 95% interval of its
    slope carried through the same exponential, its R2, and the orders the fit
    actually rests on.

    The three are fitted over one single range of orders, not three: they
    describe the same tree and quoting them over different ranges would make
    them incomparable. Orders whose mean diameter falls under `min_diameter`
    -- three voxels, where the distance transform stops resolving anything --
    are dropped, and an explicit `order_range` overrides that filter. Fix the
    range before looking at the numbers: chosen afterwards it is a knob, and
    it is the one that most easily turns any tree into a published value.

    `prespecified` is that claim, and it is an input, not something read off
    the arguments. Whether a range was fixed in advance depends on the order
    in which the operator did things, which no flag can witness: passing
    --fit-orders after seeing the table and passing it before produce the
    same command line. So it defaults to False for both the explicit range
    and the diameter filter, and only --prespecified sets it.

    `min_branches` censors the other end of the range, on the same footing.
    At the top of a Strahler tree an order holds one or two elements, and an
    element there is a chain of segments that kept its order across the
    hilum: one row whose radius can triple from end to end, whose median
    calibre therefore summarises nothing, and which IS its order's mean.
    Such an order's SD is zero by construction, which is the tell in the
    per-order table. Dropping it is mechanical and declarable in advance,
    exactly like the diameter floor, and it takes the unreliable orders off
    both ends of the range rather than only the thin one.

    It costs R_b something, and honestly: N is an exact count, not an
    estimate, and the trunk-ward orders this removes are the ones that
    anchor the log N line. But the three ratios are fitted over one single
    range by construction, so the choice is between paying that and quoting
    ratios that do not rest on the same orders.

    Flared elements are gone before this function sees the summary, taken
    out one by one rather than by the order that held them -- see
    `summarize`. Excluding the whole order would scale the wrong way: the
    larger an order, the likelier it holds at least one flared element, so
    the rule would preferentially destroy the best-sampled orders. On the
    bundled example it removes orders of 1084, 72 and 28 elements over 4, 3
    and 1 bad rows, and leaves the fit one point.

    Whatever that removes, the range that comes out is then cut back to
    the orders consecutive with the lowest survivor -- see `contiguous_run`
    for why a hole in the middle is worse than the order it removed.
    """
    direction = ORDER_DIRECTION[ordering]
    rows = [row for row in summary if row["order"] >= 0 and row["n_branches"] > 0
            and row["mean_diameter_mm"] > 0 and row["mean_length_mm"] > 0]
    claim = "pre-specified" if prespecified else "NOT declared pre-specified"
    if order_range is not None:
        low, high = order_range
        rows = [row for row in rows if low <= row["order"] <= high]
        selection = f"orders {low}..{high} ({claim})"
    else:
        rows = [row for row in rows if row["mean_diameter_mm"] >= min_diameter]
        selection = f"orders where the mean diameter clears {min_diameter:.2f} mm ({claim})"

    thin = [row["order"] for row in rows if row["n_branches"] < min_branches]
    if min_branches > 1:
        rows = [row for row in rows if row["n_branches"] >= min_branches]
        selection += f", orders carrying fewer than {min_branches} dropped"

    kept = contiguous_run(rows)
    surviving = {row["order"] for row in kept}
    gap = [row["order"] for row in rows if row["order"] not in surviving]
    rows = kept
    if gap:
        selection += ", cut back to the contiguous run"

    orders = [row["order"] for row in rows]
    fits = {}
    for name, key, sign in RATIO_KEYS:
        fit = semilog_fit(orders, [row[key] for row in rows])
        if fit is None:
            fits[name] = None
            continue
        bounds = sorted(10.0 ** (sign * direction * bound)
                        for bound in (fit["slope_low"], fit["slope_high"]))
        fits[name] = dict(fit, ratio=10.0 ** (sign * direction * fit["slope"]),
                          ratio_low=bounds[0], ratio_high=bounds[1])
    return {"selection": selection, "rows": rows, "orders": orders, "fits": fits,
            "min_diameter": min_diameter, "prespecified": bool(prespecified),
            "min_branches": min_branches, "dropped_thin": thin, "dropped_gap": gap}


def count_peak(rows, rise=1.2):
    """
    The order at which the branch count peaks, if it peaks in the middle.

    A truncated tree counted in generations does not give a decreasing N: it
    gives a hump, because the deep generations are cut off before the tree
    would naturally thin out. Fitting a line through that returns a slope,
    an R_b and even an R2, all of which describe the parabola and not the
    tree. Worth refusing out loud rather than printing.

    Returns None when the count is monotone, or when the rise before the
    peak is too small to matter.
    """
    counts = [row["n_branches"] for row in rows]
    if len(counts) < 3:
        return None
    peak = int(np.argmax(counts))
    if peak == 0 or peak == len(counts) - 1 or counts[peak] < rise * counts[0]:
        return None
    return rows[peak]["order"]


def print_ratios(result, ordering, counting):
    """Prints the ratio table, with what the fit rests on underneath it."""
    rows = result["rows"]
    dropped = result.get("dropped_thin") or []
    print(f"\n=== branching ratios ({ordering}, {counting}s) ===")
    if dropped:
        print(f"  dropped order(s) {', '.join(str(o) for o in dropped)}: fewer than "
              f"{result['min_branches']} {counting}s each. At the top of the tree an order holds "
              f"one or two, its SD is zero by construction, and a single heterogeneous "
              f"{counting} is the whole of its mean. Mechanical rule, both ends of the range, "
              f"like the diameter floor")
    gap = result.get("dropped_gap") or []
    if gap:
        print(f"  dropped order(s) {', '.join(str(o) for o in gap)}: left stranded above the hole "
              f"the cuts opened. The slope is fitted against the order number, so an island past "
              f"a gap sits far from the mean of what is left and carries leverage in proportion "
              f"to that distance squared -- it would decide the slope it was least entitled to "
              f"decide. The range is cut back to the orders consecutive with the lowest survivor")
    if len(rows) < 3:
        print(f"  {result['selection']}: {len(rows)} usable order(s), at least 3 are needed")
        return
    print(f"  fit over {result['selection']}, {len(rows)} points")
    print("  ratio                              value   95% CI            R2")
    labels = {"R_b": "R_b  branching  (10^-slope N)", "R_d": "R_d  diameter   (10^slope D)",
              "R_l": "R_l  length     (10^slope L)"}
    for name, _, _ in RATIO_KEYS:
        fit = result["fits"][name]
        if fit is None:
            print(f"  {labels[name]:<33} -")
            continue
        print(f"  {labels[name]:<33} {fit['ratio']:6.3f}   [{fit['ratio_low']:5.3f}, "
              f"{fit['ratio_high']:5.3f}]   {fit['r2']:.3f}")
    print("  order      : " + " ".join(f"{row['order']:7d}" for row in rows))
    print("  N          : " + " ".join(f"{row['n_branches']:7d}" for row in rows))
    print("  D mean (mm): " + " ".join(f"{row['mean_diameter_mm']:7.2f}" for row in rows))
    print("  L mean (mm): " + " ".join(f"{row['mean_length_mm']:7.2f}" for row in rows))
    flared = [row["order"] for row in rows if row.get("n_flared")]
    if flared:
        print(f"  WARNING: order(s) {', '.join(str(o) for o in flared)} still carry a flared "
              f"{counting} (see the flare table). Their mean diameter averages runs that are not "
              f"each one vessel, and it is inside the fit")

    wide = [row for row in rows if (row.get("calibre_spread") or 0) > ORDER_SPREAD_LIMIT]
    if wide:
        detail = ", ".join(f"{row['order']} ({row['calibre_spread']:.2f}x)" for row in wide)
        print(f"  WARNING: order(s) {detail} span more than {ORDER_SPREAD_LIMIT:.1f}x in calibre "
              f"across their 10-90 band, and are inside the fit. That is not "
              f"flare -- flare is one row that is two vessels, this is several rows that are not "
              f"the same size -- and {ordering} promotes on calibre, so an order this wide means "
              f"the promotion failed there. Its mean diameter is a summary of a mixture")

    for name, floor_value, sense in (("R_b", 2.0, "a binary tree cannot branch less than 2:1"),
                                     ("R_d", 1.0, "vessels cannot narrow towards the trunk"),
                                     ("R_l", 1.0, "vessels cannot shorten towards the trunk")):
        fit = result["fits"][name]
        if fit is None or fit["ratio"] >= floor_value:
            continue
        print(f"  WARNING: {name} = {fit['ratio']:.3f} is below {floor_value:.1f} -- "
              f"{sense}. R2 = {fit['r2']:.3f}, so if that is high the trend is real and the "
              f"defect is in what was measured, not in the fit. Read it as a symptom and quote "
              f"it as one: an R_l under 1 is the signature of order-1 elements running long "
              f"because the bifurcations that should have interrupted them were never "
              f"segmented, while the orders above them are bounded by junctions that were")

    floor = result["min_diameter"]
    edge = min(rows, key=lambda row: row["mean_diameter_mm"])
    if floor > 0 and edge["mean_diameter_mm"] <= 1.15 * floor:
        print(f"  WARNING: order {edge['order']} sits on the censoring boundary "
              f"({edge['mean_diameter_mm']:.2f} mm against a floor of {floor:.2f} mm). The radius "
              f"there is not merely imprecise, it is truncated from below -- the distance "
              f"transform cannot return less than 1.5 voxels, so that order's diameter is the "
              f"grid, not the vessel, and it anchors the steep end of every fit. Pre-specify a "
              f"range that stops one order earlier and check the ratios do not move")

    peak = count_peak(rows)
    if peak is not None:
        print(f"  WARNING: N rises to order {peak} before falling. R_b is the slope of a straight "
              f"line, and this is a hump -- the fitted value is an artefact of the fit, whatever "
              f"its R2 says. It is the normal signature of a truncated tree counted in "
              f"generations: either order by Strahler, or start the range past the peak")
    if ORDER_DIRECTION[ordering] < 0:
        print(f"  WARNING: {ordering} counts one bifurcation per step, a Strahler order counts "
              f"several -- the order only rises where two branches of equal order meet. The "
              f"three ratios are therefore mechanically smaller here than their Strahler "
              f"counterparts on the very same tree, and are NOT comparable to published values. "
              f"Use --ordering strahler_dd before interpreting any of them")


def print_flare(elements, limit):
    """
    The elements that come out wider at their end than at their start.

    An element is meant to be one vessel: the run of segments that keep the
    same order because a lateral branch left the trunk without raising it.
    That construction is safe in the periphery, where the segments it joins
    really are one vessel, and it is not safe at the top of the tree, where
    a chain can keep its order across the hilum and be aggregated into a
    single row that starts on one vessel and ends on another.

    Nothing in the per-order table shows this. The mean diameter of such an
    order is the median calibre of that row, and a median is a summary of a
    thing that has no single calibre -- 7.1 mm for a run whose radius goes
    from 3.8 to 11.2. So the check is here, per element, before the ratios:
    the ratio of the two end calibres, which for a real artery is at most a
    little under 1 and for a spliced chain is far above it.

    Tortuosity catches the same elements sometimes, and less specifically:
    a vessel is allowed to wander. It is not allowed to triple its calibre.

    Only elements of two segments or more are eligible -- see `flared`.
    """
    chains = [e for e in elements if e["n_segments"] >= 2 and np.isfinite(flare(e))]
    ranked = sorted(((flare(e), e) for e in chains), key=lambda pair: -pair[0])
    over = [pair for pair in ranked if pair[0] > limit]
    print(f"\n=== element flare (distal / proximal calibre) ===")
    if not over:
        worst = ranked[0][0] if ranked else float("nan")
        print(f"  none of the {len(chains)} multi-segment elements exceeds {limit:.2f} "
              f"(worst {worst:.2f}): every chain ends near the calibre it started at, so no "
              f"order's mean diameter is averaging a run that is two vessels")
        return
    print(f"  {len(over)} of {len(chains)} multi-segment elements exceed {limit:.2f} "
          f"({len(elements)} elements in all; single-segment ones cannot be a bad aggregation "
          f"and are not counted)")
    print("  element  order  n_seg    prox    dist   flare   length   tort")
    for value, e in over[:12]:
        print(f"  {e['element_id']:7d}  {e['order']:5d}  {e['n_segments']:5d}  "
              f"{e['proximal_calibre_mm']:6.2f}  {e['distal_calibre_mm']:6.2f}  {value:6.2f}  "
              f"{e['length_mm']:7.1f}  {e['tortuosity']:5.2f}")
    if len(over) > 12:
        print(f"  ... and {len(over) - 12} more")
    hit = sorted({e["order"] for _, e in over})
    print(f"  a flared element is a heterogeneous aggregation, not a vessel that widens: the "
          f"chain kept one order across a junction and was grouped into one row. Its calibre_mm "
          f"is a median over a radius that moves by the flare factor, and it is the whole of its "
          f"order's mean wherever that order carries one or two elements")
    print(f"  all {len(over)} are excluded from the order statistics and from the fit, one row at "
          f"a time; order(s) {', '.join(str(o) for o in hit)} keep their remaining elements. The "
          f"tree keeps them too -- only the averages lose them, as with --max-synthetic")


def ratio_rows(result, ordering, counting, spacing=None, floor_mm=None, subject=None):
    """
    The ratio table as flat dicts, one per ratio, for --ratios-csv.

    The acquisition travels with the number. A cohort assembled from these
    files has a fit range that moves from subject to subject -- by design,
    since the floor is mechanical and lands on a different order at every
    spacing -- so a reader given the ratios alone cannot tell a difference in
    anatomy from a difference in resolution. The spacing, the anisotropy, the
    floor that was applied and the orders it left are therefore columns, not
    something to be reconstructed from a log.
    """
    orders = result["orders"]
    spacing = None if spacing is None else np.asarray(spacing, float)
    out = []
    for name, _, _ in RATIO_KEYS:
        fit = result["fits"][name]
        out.append({
            "subject": subject or "",
            "spacing_x_mm": None if spacing is None else float(spacing[0]),
            "spacing_y_mm": None if spacing is None else float(spacing[1]),
            "spacing_z_mm": None if spacing is None else float(spacing[2]),
            "anisotropy": None if spacing is None else float(spacing.max() / spacing.min()),
            "fit_floor_mm": floor_mm,
            "ratio": name, "ordering": ordering, "counting": counting,
            "value": fit["ratio"] if fit else None,
            "ci_low": fit["ratio_low"] if fit else None,
            "ci_high": fit["ratio_high"] if fit else None,
            "r2": fit["r2"] if fit else None,
            "slope": fit["slope"] if fit else None,
            "n_orders": fit["n_orders"] if fit else len(orders),
            "order_min": min(orders) if orders else None,
            "order_max": max(orders) if orders else None,
            "fit_min_branches": result.get("min_branches"),
            "prespecified": int(result["prespecified"]),
        })
    return out


def fit_floor(spacing, min_voxels, override_mm=None):
    """
    Smallest mean diameter an order may have and still enter the fit, in mm.

    Three voxels OF THE COARSE AXIS of the acquired grid. Under that the
    distance transform has no dynamic range left to report -- but which voxel
    counts is not a detail. The chain resamples to the finest axis before it
    measures anything, and a floor counted there is the anisotropy ratio too
    permissive: upsampling adds samples, not information, and a vessel two
    slices across does not become resolved by being interpolated onto four.
    The order that slips in is censored from below, sits at the smallest
    radius the transform can return, and anchors the steep end of every
    slope. Measured on the phantom at 1.19/0.80/1.31 mm it moved R_d by 5.6%.

    Counting on the coarse axis is the same rule on every grid rather than a
    correction applied to some: on an isotropic input the three axes agree
    and this is the criterion it has always used. Fixing the ratio instead --
    hard-coding the 4.93 working voxels that one anisotropy happens to need
    -- would cut too high on a rounder volume and too low on a flatter one,
    which across a cohort is the same trap the other way round.

    Returns (floor in mm, one line saying how it was arrived at).
    """
    spacing = np.asarray(spacing, float)
    coarse, fine = float(spacing.max()), float(spacing.min())
    if override_mm is not None:
        return float(override_mm), f"{override_mm:.2f} mm, given as --fit-min-diameter"
    floor = min_voxels * coarse
    if coarse <= 1.05 * fine:
        return floor, f"{floor:.2f} mm = {min_voxels:g} x {coarse:.3f} mm, an isotropic grid"
    return floor, (f"{floor:.2f} mm = {min_voxels:g} x {coarse:.3f} mm, the coarse axis of a "
                   f"{coarse / fine:.2f}:1 acquisition (counting it on the {fine:.3f} mm the mask "
                   f"is resampled to would have allowed {min_voxels * fine:.2f} mm)")


def resolution_floor(voxel_size):
    """
    Smallest radius the distance transform can return, in mm.

    A skeleton voxel one step away from the background is at exactly one
    voxel from it, and the half-voxel wall correction is added on top, so
    nothing can be reported below 1.5 voxels. A tip sitting on that value
    is not a measurement, it is the grid.
    """
    return 1.5 * voxel_size


def find_breakpoints(table, positions, smooth, factors, max_order, min_radius):
    """
    Terminal branches that stop too early: a low order (still close to the
    main path) yet a calibre well above the resolution floor.

    A vessel several millimetres wide does not simply end -- the tree is
    broken there, and these tips are the most likely attachment points of
    the fragments that the largest-component filter drops. Unlike every
    other metric here this one localizes: it returns coordinates to open in
    a viewer.

    Ordering is always the main-path one, since under Strahler every leaf
    is order 1 by construction.
    """
    breakpoints = []
    for entry in table:
        if not entry["is_terminal"] or entry["generation"] < 0:
            continue
        if entry["generation"] > max_order or entry["tip_radius_mm"] <= min_radius:
            continue
        tip = entry["nodes"][-1]
        i, j, k = np.rint((positions[tip] + 0.5) / factors - 0.5).astype(int)
        x, y, z = smooth[tip]
        breakpoints.append({
            "branch_id": entry["branch_id"], "generation": entry["generation"],
            "strahler": entry["strahler"], "tip_radius_mm": entry["tip_radius_mm"],
            "length_mm": entry["length_mm"], "i": int(i), "j": int(j), "k": int(k),
            "x_mm": float(x), "y_mm": float(y), "z_mm": float(z),
        })
    return sorted(breakpoints, key=lambda b: -b["tip_radius_mm"])


def union_labels(mask, other):
    """
    26-connected labelling of the two classes taken together.

    This settles a failure mode that neither the gap threshold nor the
    overlap test can see. A fragment can be correctly classified, and the
    trunk too, while the few millimetres of vessel joining them were given to
    the other class: at a crossing the two trees run so close that the
    separation becomes a coin toss, and losing it severs the artery in two.
    The fragment is then an orphan not because it is wrong, nor because there
    is a hole, but because its pedicle was taken.

    Labelling the union tells the three cases apart with no threshold at all:
    if a fragment and the trunk are separate in one class and joined in the
    union, the cut ran through the other class and the fault is the A/V
    classification head. If they are separate in the union too, there is a
    real hole, and that is bridging or a plain false negative.
    """
    return connected_components(mask | other, structure=np.ones((3, 3, 3), dtype=int))[0]


def bridging_curve(parts, positions, mask, spacing, dilations):
    """
    How much of the tree closes up as the mask is dilated, radius by radius.

    This replaces the hand-placed gap threshold with a measurement. Dilating
    the mask by r closes any gap up to 2r -- both walls advance -- so the
    curve of "centerline length in the largest component" against r says at
    what bridging distance the tree actually becomes one object, and its
    shape says whether it ever does:

      - a sharp collapse in the component count around one or two voxels,
        with the length fraction jumping to near 1, means the fragments were
        real vessel separated by thin gaps. The knee is the bridging radius
        that can be justified, and the plateau after it is the answer.
      - a slow, steady climb with no knee means the fragments are genuinely
        somewhere else, and dilating is merely gluing unrelated things
        together. No threshold is defensible in that case, and the fragments
        have to be explained rather than bridged.

    One Euclidean distance transform of the background serves every radius,
    so the sweep costs one EDT and one labelling per point. Components are
    attributed by looking up where each existing skeleton component lands
    after dilation, which is exact: dilation only ever merges.
    """
    outside = distance_transform_edt(~mask, sampling=spacing)
    total = float(sum(length for length, _ in parts))
    anchors = [(length, np.rint(positions[next(iter(nodes))]).astype(int)) for length, nodes in parts]

    rows = []
    for radius in dilations:
        labels, n_labels = connected_components(outside <= radius, structure=np.ones((3, 3, 3), dtype=int))
        totals = defaultdict(float)
        members = defaultdict(int)
        for length, anchor_voxel in anchors:
            label = int(labels[tuple(anchor_voxel)])
            totals[label] += length
            members[label] += 1
        best = max(totals, key=lambda label: totals[label])
        rows.append({
            "dilation_mm": float(radius),
            "gap_bridged_mm": float(2.0 * radius),
            "n_mask_components": int(n_labels),
            "n_centerline_components": len(totals),
            "n_merged_into_largest": members[best],
            "length_fraction_largest": totals[best] / total if total else 0.0,
            "length_gained_mm": totals[best] - parts[0][0],
        })

    # share of the length that started outside the main component and has
    # come back, which is the only scale-free way to read the curve
    missing = total - parts[0][0]
    for row in rows:
        row["recovered_fraction"] = row["length_gained_mm"] / missing if missing > 0 else 0.0
    return rows


def print_bridging(rows, voxel_size):
    """Prints the bridging curve and reads its shape out loud."""
    if not rows:
        return
    print("\n=== bridging curve (mask dilated, centerline re-attributed) ===")
    print("  dilation  closes gaps  mask comps  centerline comps  merged  in largest  of missing")
    for row in rows:
        print(f"  {row['dilation_mm']:6.2f} mm  {row['gap_bridged_mm']:8.2f} mm  "
              f"{row['n_mask_components']:10d}  {row['n_centerline_components']:16d}  "
              f"{row['n_merged_into_largest']:6d}  {row['length_fraction_largest']:10.1%}  "
              f"{row['recovered_fraction']:10.1%}")

    # The knee is read on the share of the *missing* length recovered, not on
    # the length fraction itself: a tree that already starts at 95% cannot
    # gain ten points however cleanly its fragments reattach, and judging it
    # on the raw fraction would call every well-connected tree knee-less.
    steps = [(rows[i]["recovered_fraction"] - rows[i - 1]["recovered_fraction"], i)
             for i in range(1, len(rows))]
    if not steps:
        return
    jump, index = max(steps)
    last = rows[-1]
    if jump >= 0.25:
        print(f"  knee at {rows[index]['dilation_mm']:.2f} mm of dilation "
              f"({rows[index]['gap_bridged_mm']:.2f} mm of gap closed): {jump:.0%} of the missing "
              f"centerline reattaches in that single step. That is the bridging radius the data "
              f"supports, and what it absorbs was separated vessel, not false positive")
    else:
        print(f"  no knee: the recovery climbs smoothly to {last['recovered_fraction']:.0%} of the "
              f"missing length at {last['dilation_mm']:.2f} mm ({last['gap_bridged_mm']:.2f} mm of "
              f"gap). The fragments are not sitting just off the tree -- dilating that far glues "
              f"unrelated objects together, so they have to be explained rather than bridged")
    if last["recovered_fraction"] < 0.9:
        print(f"  {1.0 - last['recovered_fraction']:.0%} of the missing length never reattaches, "
              f"even at {last['dilation_mm']:.2f} mm")


BRIDGE_COLUMNS = ("dilation_mm", "gap_bridged_mm", "n_mask_components", "n_centerline_components",
                  "n_merged_into_largest", "length_fraction_largest", "length_gained_mm",
                  "recovered_fraction")


def analyze_orphans(parts, positions, world, radii, factors, wall_gap,
                    compare_distance=None, compare_inside=None, union=None):
    """
    Describes every component that is not the main one, and above all how far
    each sits from it.

    The fraction of centerline outside the main component says how much of the
    tree is not analysed. It does not say why, and the two possible reasons
    call for opposite corrections:

      - a break in continuity. The fragment is real vessel, the model simply
        lost a few voxels of contrast somewhere upstream. It sits a
        millimetre or two from the main tree, wall to wall, usually in line
        with a branch that ends abruptly. The fix is upstream connectivity
        (a bridging term in the loss, a morphological closing) and the model
        is better than the report suggests.
      - a false positive. The fragment is somewhere else entirely, often
        thin, often nowhere near a vessel end. The fix is specificity, and
        the model is worse than the report suggests.

    So the distance to the main component is the discriminating measurement,
    and it is reported wall to wall, not centreline to centreline: two
    vessels whose axes are 4 mm apart are touching if both are 2 mm across.
    `gap_wall_mm` is that distance, floored at zero, and `wall_gap` is the
    threshold under which a fragment is counted as bridgeable.

    Coordinates of both ends of the gap are given on the input grid, so each
    one can be opened directly in a viewer -- which is the only way to settle
    the ambiguous ones.

    A size threshold alone cannot make that call and neither can this
    distance on its own -- a real artery that dropped out over 1 to 2 cm of
    poor contrast lands on the far side of any reasonable gap threshold, so
    "isolated" here means "not within `wall_gap`", nothing more. Two things
    settle it, and both are done elsewhere: `bridging_curve` replaces the
    hand-placed threshold with the dilation radius at which the tree actually
    closes up, and `compare_distance` cross-checks each fragment against a
    second segmentation.

    That second mask is the decisive one when the model separates arteries
    from veins: a long, coherent, well-calibred fragment that is not attached
    to the arterial trunk but does attach to the venous one is not a false
    positive at all, it is an A/V labelling error, and the fix is in the
    classification head rather than in the sensitivity. `compare_gap_mm` is
    the wall-to-wall distance to that mask, `compare_overlap` the share of the
    fragment's centerline actually inside it once dilated, and `nearer` says
    which of the two trees the fragment belongs to on distance alone.

    Returns one row per orphan, longest first.
    """
    if len(parts) < 2:
        return []

    main = np.array(sorted(parts[0][1]))
    tree = cKDTree(world[main])

    def anchor_voxel(nodes):
        return tuple(np.rint(positions[next(iter(nodes))]).astype(int))

    main_union = int(union[anchor_voxel(parts[0][1])]) if union is not None else None
    rows = []
    for index, (length, nodes) in enumerate(parts[1:], start=1):
        nodes = np.array(sorted(nodes))
        distance, nearest = tree.query(world[nodes])
        closest = int(np.argmin(distance))
        orphan_node = int(nodes[closest])
        main_node = int(main[nearest[closest]])
        axis_gap = float(distance[closest])
        values = radii[nodes]

        def voxel(node):
            return np.rint((positions[node] + 0.5) / factors - 0.5).astype(int)

        i, j, k = voxel(orphan_node)
        mi, mj, mk = voxel(main_node)
        wall_distance = max(0.0, axis_gap - float(radii[orphan_node]) - float(radii[main_node]))

        compare_gap, overlap, nearer = None, None, ""
        if compare_distance is not None:
            grid = tuple(np.rint(positions[nodes]).astype(int).T)
            # the EDT of the other mask already measures to its surface, so
            # only this fragment's own radius has to come off
            compare_gap = max(0.0, float((compare_distance[grid] - radii[nodes]).min()))
            overlap = float(compare_inside[grid].mean())
            nearer = "compare" if compare_gap < wall_distance else "main"

        reconnects = None
        if union is not None:
            reconnects = int(int(union[anchor_voxel(nodes)]) == main_union)

        rows.append({
            "compare_gap_mm": compare_gap, "compare_overlap": overlap, "nearer": nearer,
            "union_reconnects": reconnects,
            # working-grid skeleton nodes of the narrowest crossing, kept for
            # the repair; not CSV columns
            "orphan_node": orphan_node, "main_node": main_node, "nodes": nodes,
            "component_id": index,
            "length_mm": float(length),
            "n_points": int(len(nodes)),
            "median_radius_mm": float(np.median(values)),
            "max_radius_mm": float(values.max()),
            "gap_axis_mm": axis_gap,
            "gap_wall_mm": wall_distance,
            "i": int(i), "j": int(j), "k": int(k),
            "main_i": int(mi), "main_j": int(mj), "main_k": int(mk),
        })

    for row in rows:
        row["bridgeable"] = int(row["gap_wall_mm"] <= wall_gap)
    return sorted(rows, key=lambda row: -row["length_mm"])


# The three things an orphan component can be. They are not variations of one
# defect, they have nothing to do with each other, and each has its own fix --
# so a single "52% connectivity" number aggregates three unrelated problems
# and points at none of them.
ORPHAN_POPULATIONS = ("severed", "hole", "dust", "detached")

POPULATION_NOTES = {
    "severed": ("A/V classification cut the pedicle",
                "rejoins the trunk as soon as the class boundary is ignored, so the vessel is "
                "there and correctly classified -- repairable in post-processing, no retraining"),
    "hole": ("a real hole, in neither class",
             "the vessel is missing over several millimetres: a frank false negative, from low "
             "contrast or motion. Retraining, or geometric bridging if you accept it"),
    "dust": ("speckle under the resolution floor",
             "too short to be vessel and sitting on the quantization floor -- a size filter "
             "removes it with no argument"),
    "detached": ("detached, cause undetermined",
                 "no comparison mask was given, so the A/V cut and the real hole cannot be "
                 "told apart -- pass --compare-label or --compare-mask"),
}


def classify_orphans(orphans, dust_length, max_detour=3.0, max_calibre_ratio=2.0,
                     max_elbow_deg=60.0):
    """
    Sorts the orphan components into the three populations, in priority order.

    Size comes first: a two-millimetre speck that happens to touch the other
    class is speckle, not a severed pedicle, and letting it into that group
    would inflate exactly the number the repair is judged on.

    Severed then demands more than reachability. Reachability through the
    union is nearly free -- the other tree is connected, so almost any
    fragment touching it has some path to the trunk -- and a partition built
    on it alone reads as though most of the missing vessel were a
    classification error recoverable in post-processing. Requiring the path
    to be short, calibre-compatible and in line with the fragment moves the
    ones that only had a long way round into `hole`, where they say what they
    actually are: vessel absent from both classes, which is sensitivity, and
    which no post-processing recovers.

    `severed_reject` keeps the reason for each demotion.
    """
    for row in orphans:
        row["severed_reject"] = ""
        if row["length_mm"] < dust_length:
            row["population"] = "dust"
        elif row.get("union_reconnects") is None:
            row["population"] = "detached"
        elif row["union_reconnects"]:
            reject = severed_verdict(row, max_detour, max_calibre_ratio, max_elbow_deg)
            row["severed_reject"] = reject or ""
            row["population"] = "hole" if reject else "severed"
        else:
            row["population"] = "hole"
    return orphans


def orphan_populations(orphans, min_length=0.0):
    """Count, length and median calibre of each population."""
    kept = [row for row in orphans if row["length_mm"] >= min_length]
    out = {}
    for name in ORPHAN_POPULATIONS:
        group = [row for row in kept if row["population"] == name]
        out[name] = {
            "n": len(group),
            "length_mm": float(sum(row["length_mm"] for row in group)),
            "median_radius_mm": float(np.median([row["median_radius_mm"] for row in group]))
            if group else None,
        }
    out["total_length_mm"] = float(sum(row["length_mm"] for row in kept))
    return out


def orphan_split(orphans, min_length=0.0):
    """Totals on each side of the bridgeable line, kept for the gap-based view."""
    kept = [row for row in orphans if row["length_mm"] >= min_length]
    bridgeable = [row for row in kept if row["bridgeable"]]
    isolated = [row for row in kept if not row["bridgeable"]]
    return {
        "n_bridgeable": len(bridgeable), "n_isolated": len(isolated),
        "length_bridgeable_mm": float(sum(row["length_mm"] for row in bridgeable)),
        "length_isolated_mm": float(sum(row["length_mm"] for row in isolated)),
    }


def print_orphans(orphans, wall_gap, compare_name=None, dust_length=10.0, show=8):
    """
    Prints the three populations of orphan components, then the worst of them.

    Reported separately on purpose. They are three unrelated defects with
    three different corrections and three different costs, and the aggregate
    "fraction of centerline outside the main tree" is the one number that
    hides which of them dominates.
    """
    if not orphans:
        return
    totals = orphan_populations(orphans)
    big = orphan_populations(orphans, dust_length)
    substantial = sorted((row for row in orphans if row["length_mm"] >= dust_length),
                         key=lambda row: -row["length_mm"])

    print(f"\n=== orphan components ({len(orphans)} outside the main tree, "
          f"{totals['total_length_mm']:.0f} mm) ===")
    for name in ORPHAN_POPULATIONS:
        group = totals[name]
        if not group["n"]:
            continue
        headline, note = POPULATION_NOTES[name]
        share = group["length_mm"] / totals["total_length_mm"] if totals["total_length_mm"] else 0.0
        radius = (f"median r {group['median_radius_mm']:.2f} mm"
                  if group["median_radius_mm"] is not None else "")
        print(f"  {name:<9} {group['n']:4d} comps  {group['length_mm']:8.1f} mm  {share:5.1%}  "
              f"{radius:<18}  {headline}")
        for wrapped in textwrap.wrap(note, 92):
            print(f"      {wrapped}")
    if compare_name:
        print(f"  severed / hole was decided by labelling the union with {compare_name}, "
              f"then requiring the union path to be a crossing rather than a way round")
        demoted = [row for row in orphans if row.get("severed_reject")]
        if demoted:
            length = sum(row["length_mm"] for row in demoted)
            print(f"  {len(demoted)} fragment(s), {length:.0f} mm, reach the trunk through the "
                  f"union but not by a credible pedicle, and are counted as holes:")
            for row in sorted(demoted, key=lambda r: -r["length_mm"])[:show]:
                print(f"    fragment {row['component_id']:3d} ({row['length_mm']:6.1f} mm): "
                      f"{row['severed_reject']}")
            if len(demoted) > show:
                print(f"    ... {len(demoted) - show} more, see severed_reject in --orphans-csv")
            print(f"  a demotion moves the correction from post-processing to retraining, so the "
                  f"three thresholds behind it are --severed-max-detour, --severed-max-calibre "
                  f"and --severed-max-elbow")
    print(f"  the split at {dust_length:.0f} mm is --dust-length; above it, the components are "
          f"the ones worth repairing")

    if not substantial:
        return
    compared = [row for row in substantial if row["compare_gap_mm"] is not None]
    header = "  len_mm  n_pts  med_r  max_r  gap_wall  gap_axis"
    header += "   cmp_gap  cmp_ovl" if compared else ""
    print(header + "  population  fragment voxel      nearest on trunk")
    for row in substantial[:show]:
        line = (f"  {row['length_mm']:6.1f} {row['n_points']:6d} {row['median_radius_mm']:6.2f} "
                f"{row['max_radius_mm']:6.2f} {row['gap_wall_mm']:9.2f} {row['gap_axis_mm']:9.2f}")
        if compared:
            line += (f" {row['compare_gap_mm']:9.2f} {row['compare_overlap']:8.2f}"
                     if row["compare_gap_mm"] is not None else " " * 18)
        print(line + f"  {row['population']:>10}  ({row['i']:4d},{row['j']:4d},{row['k']:4d})  "
                     f"({row['main_i']:4d},{row['main_j']:4d},{row['main_k']:4d})")
    if len(substantial) > show:
        print(f"  ... {len(substantial) - show} more over {dust_length:.0f} mm, "
              f"use --orphans-csv for the full list")
    if compared:
        inside = [row for row in compared if row["compare_overlap"] >= 0.5]
        print(f"  {len(inside)}/{len(compared)} of them have more than half their centerline "
              f"inside {compare_name}: those would be outright A/V swaps rather than cut pedicles")
        if len(inside) < len(compared):
            print("  proximity alone proves nothing -- arteries and veins run alongside each other "
                  "everywhere, so only the overlap column and the union test carry evidence")



def quality_metrics(graph, table, bifurcations, breakpoints, parts, volume_fraction,
                    n_volume_components, voxel_size, min_radius, max_order,
                    n_cycles=0, n_cycles_broken=0, orphans=(), volume_ml=None,
                    n_elements=None, ordering=""):
    """
    The five numbers that say whether a segmentation can be trusted.

    They are independent on purpose: 1 and 4 are topological (is the tree in
    one piece, and does it wrongly loop back on itself), 2 says how deep it
    goes, 3 whether the vessels have a plausible calibre, 5 where it breaks.
    A defect in one implies nothing about the others.
    """
    total_length = sum(length for length, _ in parts)
    populations, split = orphan_populations(orphans), orphan_split(orphans)
    leaves = [b for b in table if b["is_terminal"]]
    orders = [b["order"] for b in table if b["order"] >= 0]
    floor = resolution_floor(voxel_size)
    at_floor = [b for b in leaves if b["tip_radius_mm"] <= floor * 1.001]
    murray = np.array([b["murray_exponent"] for b in bifurcations
                       if b["well_resolved"] and b["murray_exponent"] is not None])

    return {
        # the structural block: what the symmetric control is read on, so two
        # runs on the two classes concatenate into a comparable pair of rows
        "ordering": ordering,
        "mask_volume_ml": volume_ml,
        "n_segments": len(table),
        "n_elements": n_elements,
        "total_length_mm": float(sum(b["length_mm"] for b in table)),
        "max_order": max(orders) if orders else None,
        "n_components": len(parts),
        "n_volume_components": n_volume_components,
        "largest_component_volume_fraction": volume_fraction,
        "largest_component_length_fraction": parts[0][0] / total_length if total_length else 0.0,
        "length_outside_largest_mm": total_length - parts[0][0],
        "n_fragments_over_10mm": sum(1 for length, _ in parts[1:] if length >= 10.0),
        "n_orphans_bridgeable": split["n_bridgeable"],
        "n_orphans_isolated": split["n_isolated"],
        "orphan_length_bridgeable_mm": split["length_bridgeable_mm"],
        "orphan_length_isolated_mm": split["length_isolated_mm"],
        "n_orphans_severed": populations["severed"]["n"],
        "n_orphans_holed": populations["hole"]["n"],
        "n_orphans_dust": populations["dust"]["n"],
        "orphan_length_severed_mm": populations["severed"]["length_mm"],
        "orphan_length_holed_mm": populations["hole"]["length_mm"],
        "orphan_length_dust_mm": populations["dust"]["length_mm"],
        "n_leaves": len(leaves),
        "resolution_floor_mm": floor,
        "leaves_at_floor_fraction": len(at_floor) / len(leaves) if leaves else 0.0,
        "n_murray": int(murray.size),
        "murray_median": float(np.median(murray)) if murray.size else None,
        "murray_q1": float(np.percentile(murray, 25)) if murray.size else None,
        "murray_q3": float(np.percentile(murray, 75)) if murray.size else None,
        "n_cycles": int(n_cycles),
        "n_cycles_broken": int(n_cycles_broken),
        "n_breakpoints": len(breakpoints),
        "breakpoints_fraction": len(breakpoints) / len(leaves) if leaves else 0.0,
        "breakpoint_min_radius_mm": min_radius,
        "breakpoint_max_order": max_order,
    }


def print_quality(metrics, breakpoints):
    """Prints the five quality metrics, then the worst breakpoints."""
    def line(label, value, comment):
        print(f"{label:<48}: {value:>6}   {comment}")

    print("\n=== quality metrics ===")
    line("1. centerline length in the largest component",
         f"{metrics['largest_component_length_fraction']:.1%}",
         f"({metrics['length_outside_largest_mm']:.0f} mm outside, {metrics['n_components']} components, "
         f"{metrics['n_fragments_over_10mm']} of them over 10 mm; "
         f"{metrics['orphan_length_bridgeable_mm']:.0f} mm broken off, "
         f"{metrics['orphan_length_isolated_mm']:.0f} mm isolated)")
    line("   the same fraction measured on mask volume",
         f"{metrics['largest_component_volume_fraction']:.1%}",
         "(optimistic: the trunks weigh more than the twigs)")
    line(f"2. leaves at the resolution floor ({metrics['resolution_floor_mm']:.2f} mm)",
         f"{metrics['leaves_at_floor_fraction']:.1%}",
         f"({round(metrics['leaves_at_floor_fraction'] * metrics['n_leaves'])}/{metrics['n_leaves']} leaves; "
         f"near 100% the image is the limit, well below it the model stops early)")
    if metrics["murray_median"] is not None:
        line("3. Murray exponent, vessels over 3 voxels",
             f"{metrics['murray_median']:.2f}",
             f"(IQR {metrics['murray_q1']:.2f}-{metrics['murray_q3']:.2f}, n={metrics['n_murray']}; "
             f"3 is the optimum, the mean is meaningless here)")
    else:
        line("3. Murray exponent, vessels over 3 voxels", "-", "(no bifurcation resolved well enough)")
    line("4. cycles in the skeleton", f"{metrics['n_cycles']}",
         f"(an artery tree has no anastomosis, expected 0; {metrics['n_cycles_broken']} cut "
         f"to make the tree orderable)")
    line(f"5. leaves ending early (order <= {metrics['breakpoint_max_order']}, "
         f"r > {metrics['breakpoint_min_radius_mm']:.2f} mm)",
         f"{metrics['n_breakpoints']}", f"({metrics['breakpoints_fraction']:.1%} of leaves)")
    for entry in breakpoints[:5]:
        print(f"     r={entry['tip_radius_mm']:5.2f} mm  order {entry['generation']}  "
              f"voxel ({entry['i']}, {entry['j']}, {entry['k']})  "
              f"world ({entry['x_mm']:.1f}, {entry['y_mm']:.1f}, {entry['z_mm']:.1f}) mm")
    if len(breakpoints) > 5:
        print(f"     ... {len(breakpoints) - 5} more, use --breakpoints-csv for the full list")


def describe(values):
    """mean / median / p10 / p90 / range of a sample, as a printable string."""
    values = np.asarray([v for v in values if v is not None], float)
    if values.size == 0:
        return "n/a"
    p10, p90 = np.percentile(values, (10, 90))
    return (f"mean {values.mean():6.2f}  median {np.median(values):6.2f}  "
            f"p10-p90 {p10:5.2f}-{p90:5.2f}  range {values.min():5.2f}-{values.max():5.2f}")


def check_monotonicity(rows, order_key):
    """
    Lists the orders whose calibre goes the wrong way.

    Along the main path a vessel can only narrow, and a Strahler order can
    only widen. Any inversion is a defect of the segmentation -- a leak into
    a neighbouring vein, two vessels fused by partial volume -- or a
    mis-rooted tree, so this doubles as a quality check.
    """
    rows = [row for row in rows if row["order"] >= 0]
    if order_key in ("strahler", "strahler_dd"):
        rows = rows[::-1]
    return [(before["order"], after["order"], before["mean_diameter_mm"], after["mean_diameter_mm"])
            for before, after in zip(rows, rows[1:]) if after["mean_diameter_mm"] > before["mean_diameter_mm"]]


def print_angles(bifurcations, show=6):
    """
    The three angles of the junction, over the bifurcations whose directions
    were fitted clear of the junction blob.

    Both counts are printed because they trade off: the offset that buys a
    clean direction is the offset that disqualifies the short branches, and
    an angle quoted without the fraction it was measured on is not a
    measurement. The contaminated ones are shown underneath for comparison,
    never merged in.

    The loss is not only in power. What disqualifies a junction is a daughter
    too short relative to the parental calibre, and that is a property of the
    branching, not noise: a small lateral branch leaving a wide trunk is
    exactly the configuration that gets dropped. The surviving sample is
    therefore biased towards the less asymmetric junctions, and the median
    angle it gives is the median over that subset, not over the tree. It has
    to be quoted that way.
    """
    pairs = [b for b in bifurcations if b["angle_deg"] is not None]
    if not pairs:
        print("angle (deg)       : no two-daughter junction to measure")
        return
    clean = [b for b in pairs if b["angle_clean"]]
    print(f"angles over the {len(clean)}/{len(pairs)} two-daughter junctions with a direction "
          f"fitted clear of the blob:")
    if clean and len(clean) < len(pairs):
        kept = float(np.median([b["asymmetry"] for b in clean if b["asymmetry"] is not None]))
        dropped = [b["asymmetry"] for b in pairs if not b["angle_clean"] and b["asymmetry"] is not None]
        print(f"  selection: what is dropped is a daughter too short for the parental calibre, "
              f"which is a kind of branching, not noise -- median asymmetry {kept:.2f} here "
              f"against {np.median(dropped):.2f} among the dropped. Quote the angles as the median "
              f"over this subset, not over the tree")
    if clean:
        parent_child = [b["angle_parent_first_deg"] for b in clean] + \
                       [b["angle_parent_second_deg"] for b in clean]
        print(f"  parent-daughter : {describe(parent_child)}")
        print(f"  daughter-daughter: {describe([b['angle_deg'] for b in clean])}")
        defects = [b["planarity_defect_deg"] for b in clean]
        print(f"  planarity defect : {describe(defects)}   "
              f"(0 = parent between the daughters in one plane)")
        median = float(np.median(defects))
        if median > 30.0:
            print(f"  WARNING: the median planarity defect is {median:.1f} deg. The parent "
                  f"direction does not lie between the daughters, which no bifurcation does. "
                  f"That is a convention fault in the directions, not anatomy -- check the sign "
                  f"of the parent axis (tail_axis looks back up the branch) before reading any "
                  f"angle below")
        # the tail is the interesting part: a near-planar median with a few
        # junctions far off it is not a worse measurement, it is a shortlist
        skew = sorted((b for b in clean if b["planarity_defect_deg"] > max(30.0, 3.0 * median)),
                      key=lambda b: -b["planarity_defect_deg"])
        if skew and median <= 30.0:
            print(f"  {len(skew)} junction(s) are far from planar against a median of "
                  f"{median:.1f} deg. A bifurcation is planar to a few degrees, so these are "
                  f"more likely two vessels welded by partial volume than a branching -- same "
                  f"family as a skeleton cycle. Open them:")
            for entry in skew[:show]:
                print(f"    defect {entry['planarity_defect_deg']:6.1f} deg  parent r "
                      f"{entry['parent_radius_mm']:4.2f} mm  at ({entry['i']:4d},{entry['j']:4d},"
                      f"{entry['k']:4d})")
            if len(skew) > show:
                print(f"    ... {len(skew) - show} more, use --bifurcations-csv")
    contaminated = [b for b in pairs if not b["angle_clean"]]
    if contaminated:
        print(f"  the other {len(contaminated)}: branch too short to leave the blob, "
              f"daughter-daughter {describe([b['angle_deg'] for b in contaminated])} -- "
              f"measured inside the bend, reads wide, do not quote")


def print_analysis(graph, table, summary, bifurcations, order_key, min_radius, counting="segment"):
    """Prints the anatomical report: per order, leaves, bifurcations, tree."""
    label = {"generation": "generation (main path)", "strahler": "Strahler order",
             "strahler_dd": "diameter-defined Strahler order",
             "bfs_generation": "generation (junctions from the root)"}[order_key]
    print(f"\n=== {counting}s per {label} ===")
    print("ord    n  term   length_mm  mean_len    sd_len  mean_dia    sd_dia  mean_rad  tort  tip_rad")
    for row in summary:
        tip = f"{row['mean_tip_radius_mm']:7.2f}" if row["mean_tip_radius_mm"] is not None else "      -"
        print(f"{row['order']:3d} {row['n_branches']:4d} {row['n_terminal']:5d} "
              f"{row['total_length_mm']:11.1f} {row['mean_length_mm']:9.1f} {row['sd_length_mm']:9.1f} "
              f"{row['mean_diameter_mm']:9.2f} {row['sd_diameter_mm']:9.2f} "
              f"{row['mean_radius_mm']:9.2f} {row['mean_tortuosity']:5.2f} {tip}")

    inversions = check_monotonicity(summary, order_key)
    if inversions:
        print(f"calibre monotonicity: VIOLATED at {len(inversions)} step(s) -- " +
              ", ".join(f"{a}->{b} (diameter {ra:.2f} -> {rb:.2f} mm)" for a, b, ra, rb in inversions))
        print("  a vessel cannot widen downstream: check for a leak into a vein, "
              "two vessels fused, or a wrong root")
    else:
        print("calibre monotonicity: ok (radius decreases at every step)")

    leaves = [b for b in table if b["is_terminal"]]
    print(f"\n=== terminal branches ({len(leaves)} leaves) ===")
    if leaves:
        print(f"tip radius (mm)   : {describe([b['tip_radius_mm'] for b in leaves])}")
        print(f"mean radius (mm)  : {describe([b['mean_radius_mm'] for b in leaves])}")
        print(f"length (mm)       : {describe([b['length_mm'] for b in leaves])}")
        print(f"order             : {describe([b[order_key] for b in leaves])}")

    junctions = sum(1 for _, d in graph.degree() if d >= 3)
    resolved = [b for b in bifurcations if b["well_resolved"]]
    print(f"\n=== bifurcations ({len(bifurcations)} measured / {junctions} junctions) ===")
    if bifurcations:
        daughters = [b["n_children"] for b in bifurcations]
        trifurcations = sum(1 for n in daughters if n > 2)
        print(f"daughters per junction: mean {np.mean(daughters):.2f} "
              f"({trifurcations} junctions with more than 2)")
        print(f"parent radius (mm): {describe([b['parent_radius_mm'] for b in bifurcations])}")
        print(f"asymmetry ratio   : {describe([b['asymmetry'] for b in bifurcations])}")
        print_angles(bifurcations)
        print(f"\nrestricted to the {len(resolved)} bifurcations where the parent and both "
              f"daughters exceed {min_radius:.2f} mm:")
        if resolved:
            print(f"area ratio        : {describe([b['area_ratio'] for b in resolved])}")
            print(f"Murray exponent   : {describe([b['murray_exponent'] for b in resolved])}")
            print_angles(resolved)
        else:
            print("  none -- the mask does not resolve any vessel well enough")

    print("\n=== tree ===")
    lengths = np.array([b["length_mm"] for b in table])
    print(f"tortuosity        : {describe([b['tortuosity'] for b in table])}")
    print(f"branch length (mm): {describe(lengths)}")
    orphans = next((row for row in summary if row["order"] < 0), None)
    if orphans:
        print(f"unreachable from the root: {orphans['n_branches']} branches "
              f"({orphans['total_length_mm']:.1f} mm), reported as order -1")


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #
def paint_centerline(table, positions, shape, factors, mode="binary"):
    """
    Rasterizes the branches onto a volume of the given shape, joining
    consecutive points with a discrete 3D line so the result stays connected
    even when the working grid is finer than the output grid.

    Voxel positions are the raw skeleton ones, not the smoothed curve: the
    output has to land on the voxels of the mask.
    """
    volume = np.zeros(shape, dtype=np.uint8)
    for entry in table:
        order = entry["order"]
        # unreachable branches (order -1) are painted 255 rather than dropped
        value = {"binary": 1, "order": order + 1 if order >= 0 else 255,
                 "branch": entry["branch_id"] % 255 + 1}[mode]
        voxels = np.rint((positions[entry["nodes"]] + 0.5) / factors - 0.5).astype(int)
        voxels = np.clip(voxels, 0, np.array(shape) - 1)
        for start, stop in zip(voxels, voxels[1:]):
            volume[line_nd(start, stop, endpoint=True)] = value
        volume[tuple(voxels[-1])] = value
    return volume


def write_points_csv(path, table, positions, smooth, radii, factors):
    """
    One row per centerline point, in branch order.

    The voxel indices are those of the skeleton on the input grid, the
    millimetre coordinates those of the smoothed curve.
    """
    rows = ["branch_id,order,strahler,point_index,i,j,k,x_mm,y_mm,z_mm,radius_mm"]
    for entry in table:
        nodes = entry["nodes"]
        source = np.rint((positions[nodes] + 0.5) / factors - 0.5).astype(int)
        for k, node in enumerate(nodes):
            i, j, l = source[k]
            x, y, z = smooth[node]
            rows.append(f"{entry['branch_id']},{entry['order']},{entry['strahler']},{k},"
                        f"{i},{j},{l},{x:.3f},{y:.3f},{z:.3f},{radii[node]:.3f}")
    with open(path, "w") as handle:
        handle.write("\n".join(rows) + "\n")


def write_table_csv(path, table, columns):
    """Writes a list of dicts as a CSV, keeping only `columns`."""
    rows = [",".join(columns)]
    for entry in table:
        rows.append(",".join(
            "" if entry[c] is None else f"{entry[c]:.4f}" if isinstance(entry[c], float) else str(entry[c])
            for c in columns
        ))
    with open(path, "w") as handle:
        handle.write("\n".join(rows) + "\n")


BRANCH_COLUMNS = ("branch_id", "generation", "strahler", "strahler_dd", "bfs_generation", "n_points",
                  "length_mm", "chord_mm", "tortuosity", "calibre_mm",
                  "mean_radius_mm", "min_radius_mm", "max_radius_mm",
                  "proximal_calibre_mm", "distal_calibre_mm", "tip_radius_mm", "is_terminal",
                  "synthetic_fraction")

ORDER_COLUMNS = ("order", "n_branches", "n_terminal", "total_length_mm",
                 "mean_length_mm", "sd_length_mm", "mean_diameter_mm", "sd_diameter_mm",
                 "mean_radius_mm", "mean_proximal_calibre_mm", "mean_distal_calibre_mm",
                 "mean_tortuosity", "calibre_spread", "max_flare", "n_flared",
                 "mean_tip_radius_mm")

BIFURCATION_COLUMNS = ("node", "i", "j", "k",
                       "order", "n_children", "parent_radius_mm", "min_child_radius_mm",
                       "area_ratio", "asymmetry", "murray_exponent", "angle_deg",
                       "angle_parent_first_deg", "angle_parent_second_deg",
                       "planarity_defect_deg", "angle_clean", "well_resolved")

ELEMENT_COLUMNS = ("element_id", "order", "n_segments", "length_mm", "chord_mm", "tortuosity",
                   "calibre_mm", "mean_radius_mm", "proximal_calibre_mm", "distal_calibre_mm",
                   "flare", "tip_radius_mm", "is_terminal", "synthetic_fraction")

RATIO_COLUMNS = ("subject", "spacing_x_mm", "spacing_y_mm", "spacing_z_mm", "anisotropy",
                 "fit_floor_mm", "ratio", "ordering", "counting", "value", "ci_low", "ci_high",
                 "r2", "slope", "n_orders", "order_min", "order_max", "fit_min_branches",
                 "prespecified")

RECLAIM_COLUMNS = ("component_id", "repair", "length_mm", "median_radius_mm", "max_radius_mm",
                   "gap_wall_mm", "gap_axis_mm", "path_mm", "detour", "path_calibre_mm",
                   "elbow_deg", "reclaimed_ml",
                   "i", "j", "k", "main_i", "main_j", "main_k")

BREAKPOINT_COLUMNS = ("branch_id", "generation", "strahler", "tip_radius_mm", "length_mm",
                      "i", "j", "k", "x_mm", "y_mm", "z_mm")

ORPHAN_COLUMNS = ("component_id", "length_mm", "n_points", "median_radius_mm", "max_radius_mm",
                  "gap_axis_mm", "gap_wall_mm", "bridgeable",
                  "population", "compare_gap_mm", "compare_overlap", "nearer", "union_reconnects",
                  "path_mm", "detour", "path_calibre_mm", "elbow_deg", "severed_reject",
                  "i", "j", "k", "main_i", "main_j", "main_k")

QUALITY_COLUMNS = ("ordering", "mask_volume_ml", "n_segments", "n_elements", "n_leaves",
                   "total_length_mm", "max_order",
                   "largest_component_length_fraction", "largest_component_volume_fraction",
                   "length_outside_largest_mm", "n_components", "n_fragments_over_10mm",
                   "n_orphans_bridgeable", "n_orphans_isolated",
                   "orphan_length_bridgeable_mm", "orphan_length_isolated_mm",
                   "n_orphans_severed", "n_orphans_holed", "n_orphans_dust",
                   "orphan_length_severed_mm", "orphan_length_holed_mm", "orphan_length_dust_mm",
                   "leaves_at_floor_fraction", "resolution_floor_mm",
                   "murray_median", "murray_q1", "murray_q3", "n_murray",
                   "n_cycles", "n_cycles_broken", "n_breakpoints", "breakpoints_fraction")


def write_vtk(path, table, smooth, radii):
    """Legacy ASCII VTK polydata: one polyline per branch, radius as scalar."""
    points, lines, scalars, offset = [], [], [], 0
    for entry in table:
        nodes = entry["nodes"]
        points.extend(f"{x:.4f} {y:.4f} {z:.4f}" for x, y, z in smooth[nodes])
        scalars.extend(f"{radii[n]:.4f}" for n in nodes)
        lines.append(" ".join(str(v) for v in [len(nodes), *range(offset, offset + len(nodes))]))
        offset += len(nodes)

    with open(path, "w") as handle:
        handle.write("# vtk DataFile Version 3.0\ncenterline\nASCII\nDATASET POLYDATA\n")
        handle.write(f"POINTS {len(points)} float\n" + "\n".join(points) + "\n")
        handle.write(f"LINES {len(lines)} {sum(len(l.split()) for l in lines)}\n" + "\n".join(lines) + "\n")
        handle.write(f"POINT_DATA {len(points)}\nSCALARS radius_mm float 1\nLOOKUP_TABLE default\n")
        handle.write("\n".join(scalars) + "\n")


CENTERLINE_SUFFIX = "_centerline"


def default_output_path(mask_path, suffix=CENTERLINE_SUFFIX):
    """"artery.nii.gz" -> "artery_centerline.nii.gz", in the same directory."""
    directory, filename = os.path.split(mask_path)
    if filename.endswith(".nii.gz"):
        stem, ext = filename[: -len(".nii.gz")], ".nii.gz"
    else:
        stem, ext = os.path.splitext(filename)
    return os.path.join(directory, f"{stem}{suffix}{ext}")


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
def skeletonize_graph(work_mask, work_affine, work_spacing, verbose=True):
    """
    The whole mask-to-graph stage: radii, thinning, voxel graph, world
    coordinates. Everything downstream of it works on the graph alone.

    It is a function rather than a stretch of `main` because the repaired
    mask has to go through exactly the same stage as the original one. Any
    difference between the two runs would be indistinguishable from the
    effect being measured, so there is only one copy of it.

    Returns (radius_map, skeleton, graph, positions, radii, voxel size, world).
    """
    # the EDT stops at the last inside voxel centre, so the wall sits about
    # half a voxel further out
    radius_map = distance_transform_edt(work_mask, sampling=work_spacing) + 0.5 * work_spacing.min()
    skeleton = skeletonize(work_mask) > 0  # older skimage returns 0/255 uint8 in 3D
    if verbose:
        print(f"skeleton: {int(skeleton.sum())} voxels")

    graph, positions = build_voxel_graph(skeleton, work_spacing)
    contract_junction_clusters(graph, positions, work_mask)
    update_edge_weights(graph, positions, work_spacing)
    radii = radius_map[tuple(np.rint(positions).astype(int).T)]

    voxel_size = float(work_spacing.min())
    world = positions @ work_affine[:3, :3].T + work_affine[:3, 3]
    return radius_map, skeleton, graph, positions, radii, voxel_size, world


def build_tree(base_graph, positions, radii, world, voxel_size, args, radius_factor,
               synthetic=None):
    """
    Everything between the raw skeleton graph and the ordered branch table:
    pruning, component filtering, cycle breaking, ordering, smoothing.

    `base_graph` is never modified -- it is copied on entry -- so this can be
    called repeatedly with different pruning strengths on one skeleton, which
    is what the sensitivity sweep does. Returns None when nothing survives.
    """
    graph = base_graph.copy()
    pruned = prune_spurs(graph, radii, args.min_branch_length, radius_factor)
    if graph.number_of_nodes() == 0:
        return None

    # measure the fragmentation before dropping anything, then restrict
    parts = component_lengths(graph)
    if not args.all_components and len(parts) > 1:
        graph.remove_nodes_from(set().union(*(nodes for _, nodes in parts[1:])))
    dropped = drop_small_components(graph, args.min_component_length)
    if graph.number_of_nodes() == 0:
        return None

    cycles = graph.number_of_edges() - graph.number_of_nodes() + nx.number_connected_components(graph)
    broken = [] if args.keep_cycles else break_cycles(graph, radii)
    # a cut loop becomes a dead end, which is a spur like any other
    repruned = prune_spurs(graph, radii, args.min_branch_length, radius_factor) if broken else 0
    if graph.number_of_nodes() == 0:
        return None

    root_voxel = None
    if args.root is not None:
        root_voxel = np.asarray(args.root, float) * args.factors + 0.5 * (args.factors - 1.0)
    branches = extract_branches(graph)
    ordered = order_branches(graph, branches, positions, radii, root_voxel)
    smooth = smooth_centerline(ordered, world, voxel_size, args.smoothing, args.max_shift)
    table = compute_orders(branch_table(graph, ordered, smooth, radii, voxel_size,
                                        getattr(args, "angle_offset", DIRECTION_OFFSET),
                                        synthetic))
    converged = diameter_defined_strahler(table)

    return {"graph": graph, "table": table, "smooth": smooth, "parts": parts,
            "pruned": pruned + repruned, "dropped": dropped, "broken": broken,
            "cycles": int(cycles), "dd_converged": converged}


def union_paths(work_mask, other, orphans, positions, radii, spacing):
    """
    Measures the route through the other class that makes each candidate
    fragment reachable, and whether that route is a crossing or a detour.

    `union_reconnects` on its own is far too generous a test. It asks only
    whether a path exists, and the venous tree is one connected object: once
    a fragment touches it anywhere, a path to the trunk almost always exists,
    running centimetres down the vein and back up. That path is not a severed
    pedicle. Calling it one turns the whole missing-vessel budget into a
    post-processing problem when it is a sensitivity problem.

    So the path is measured, not just found, on three counts:

    - `detour`, the path length over the straight axis-to-axis gap. A real
      crossing gives a path barely longer than the direct distance; a run
      down the vein gives a multiple of it. This is the discriminating one.
    - `path_calibre_mm`, the median half-width of the union along the stretch
      actually traversed in the other class. A pedicle taken by the classifier
      is the width of the artery it belongs to. Three times that width is a
      vein being walked down, not an artery being crossed.
    - `elbow_deg`, the angle between the fragment's own axis at the cut and
      the direction to where the path attaches. A vessel continues into its
      pedicle; a detour leaves at an angle.

    One Dijkstra serves both the classification and the repair, and the path
    is kept on the row so `reclaim_pedicles` never recomputes it.
    """
    candidates = [row for row in orphans if row.get("union_reconnects")]
    for row in orphans:
        row.setdefault("path_mm", None)
        row.setdefault("detour", None)
        row.setdefault("path_calibre_mm", None)
        row.setdefault("elbow_deg", None)
        row["path"] = None
    if not candidates:
        return

    union = work_mask | other
    inside = distance_transform_edt(union, sampling=spacing)
    caps = {row["component_id"]: float(min(radii[row["orphan_node"]], radii[row["main_node"]]))
            for row in candidates}
    # the cost is 1/EDT, clipped at the calibre being rejoined: unclipped, a
    # wide vessel is arbitrarily cheap and the cheapest route from a fragment
    # millimetres away runs down the nearest trunk and back
    costs = np.where(union, 1.0 / np.clip(inside, 1e-6, max(caps.values())), np.inf)

    # the trunk is the mask component the main tree sits in, not the tree
    # itself: the path has to be allowed to leave the trunk anywhere
    labels = connected_components(work_mask, structure=np.ones((3, 3, 3), dtype=int))[0]
    trunk_voxel = tuple(np.rint(positions[candidates[0]["main_node"]]).astype(int))
    starts = np.argwhere(labels == labels[trunk_voxel])

    mcp = MCP_Geometric(costs, sampling=tuple(float(s) for s in spacing), fully_connected=True)
    mcp.find_costs(starts.tolist())

    taken = other & ~work_mask
    for row in candidates:
        anchor = tuple(np.rint(positions[row["orphan_node"]]).astype(int))
        try:
            path = np.asarray(mcp.traceback(anchor))
        except ValueError:
            path = None
        if path is None or len(path) < 2:
            continue

        row["path"] = path
        row["path_mm"] = float(np.linalg.norm(np.diff(path, axis=0) * spacing, axis=1).sum())
        gap = row["gap_axis_mm"]
        row["detour"] = row["path_mm"] / gap if gap > 0 else float("inf")

        crossed = taken[tuple(path.T)]
        row["path_calibre_mm"] = float(np.median(inside[tuple(path[crossed].T)])) if crossed.any() else 0.0

        # the fragment's own axis where it was cut, against the direction to
        # the point the path attaches on the trunk
        cap = caps[row["component_id"]]
        points = positions[row["nodes"]] * spacing
        origin = positions[row["orphan_node"]] * spacing
        near = points[np.linalg.norm(points - origin, axis=1) <= max(3.0 * cap, 5.0 * min(spacing))]
        toward = (path[0] - path[-1]) * spacing
        if len(near) >= 3 and np.linalg.norm(toward) > 0:
            axis = np.linalg.svd(near - near.mean(axis=0), full_matrices=False)[2][0]
            # the axis has no sign, so the acute angle is the whole claim
            row["elbow_deg"] = min(vector_angle(axis, toward), 180.0 - vector_angle(axis, toward))


def severed_verdict(row, max_detour, max_calibre_ratio, max_elbow_deg):
    """
    Whether a union path is a mislabelled crossing or a detour, and why not.

    Returns None when the fragment passes, otherwise the short reason it
    failed. The reason is kept and reported: a fragment demoted from severed
    to hole is a change of diagnosis -- post-processing to retraining -- and
    it has to be auditable rather than a count that moved.
    """
    if row.get("path") is None:
        return "no union path"
    if row["detour"] > max_detour:
        return f"detour x{row['detour']:.1f}"
    cap = row["median_radius_mm"]
    if cap > 0 and row["path_calibre_mm"] > max_calibre_ratio * cap:
        return f"crosses r={row['path_calibre_mm']:.1f} mm"
    if row["elbow_deg"] is not None and row["elbow_deg"] > max_elbow_deg:
        return f"elbow {row['elbow_deg']:.0f} deg"
    return None


def reclaim_pedicles(work_mask, other, orphans, spacing):
    """
    Puts back the vessel that the A/V classifier gave to the other class.

    Only the fragments that `severed_verdict` cleared get here, so the path
    is known to be a short, calibre-compatible, in-line crossing. Around it a
    tube is opened, no wider than the narrower of the two vessel ends being
    joined and always clipped to the union. Both caps are what keeps the
    repair from swallowing the vein it crosses: the reclaimed pedicle can be
    no wider than the artery it restores, and can contain no voxel that was
    not already segmented as vessel.

    The reclaimed voxels are returned separately and never merged away: a
    repaired mask that cannot say which of its voxels were invented is not a
    measurement, it is an assertion. Everything downstream -- the painted
    label, the per-branch synthetic fraction, the option to drop those
    branches from the statistics -- reads that one array.

    Returns (repaired mask, reclaimed voxels, one report row per fragment).
    """
    severed = [row for row in orphans if row["population"] == "severed"]
    if not severed:
        return work_mask, np.zeros_like(work_mask), []

    union = work_mask | other
    reclaimed = np.zeros_like(work_mask)
    rows = []
    for row in severed:
        box, gained = paint_pedicle(row["path"], float(row["median_radius_mm"]),
                                    union, work_mask, spacing)
        reclaimed[box] |= gained
        rows.append(dict(row, repair="reclaimed",
                         reclaimed_ml=float(gained.sum() * np.prod(spacing) / 1000.0)))

    return work_mask | reclaimed, reclaimed, rows


def paint_pedicle(path, cap, union, work_mask, spacing):
    """
    Opens the tube around one path, and returns (box, voxels gained in it).

    The tube is capped at `cap`, the fragment's own radius, and intersected
    with the union, so the repair can neither be wider than the artery it
    restores nor contain a voxel that was not already segmented as vessel.
    It is painted in a bounding box so the EDT stays cheap.
    """
    pad = np.ceil(cap / spacing).astype(int) + 1
    low = np.maximum(path.min(axis=0) - pad, 0)
    high = np.minimum(path.max(axis=0) + pad + 1, work_mask.shape)
    box = tuple(slice(a, b) for a, b in zip(low, high))
    seed = np.ones(tuple(high - low), dtype=bool)
    seed[tuple((path - low).T)] = False
    tube = (distance_transform_edt(seed, sampling=spacing) <= cap) & union[box]
    return box, tube & ~work_mask[box]


def sweep_detour(orphans, work_mask, other, spacing, thresholds, dust_length,
                 max_calibre_ratio, max_elbow_deg):
    """
    What each detour threshold buys and what it costs, in one pass.

    The threshold is the only free parameter of the repair, and picking it by
    eye on a list of fragments is how a knob gets tuned to a result. The
    trade is not obvious either: a fragment reached by a long way round is
    reattached by a long tube, so the volume taken from the other class and
    the length of fabricated centerline both grow faster than the orphan
    length recovered.

    The cost is reported per step, not cumulatively. A running average over
    everything admitted so far is dominated by the fragments already in it,
    so an expensive step barely moves it and the column reads flat whatever
    happens -- the marginal figure answers "what did this step cost", which
    is the question the threshold actually asks.

    Even so the column decides nothing on its own. It is volume per
    millimetre of orphan recovered, so a long fragment reached by a long thin
    tube is cheap by it while still being a detour. Read it against `detour`,
    never instead of it.

    Each fragment's tube is painted once; the thresholds only change which
    ones are summed, so the whole table costs one pass over the candidates.
    """
    union = work_mask | other
    priced = []
    for row in orphans:
        if row.get("path") is None or row["length_mm"] < dust_length:
            continue  # same precedence as classify_orphans: size first
        if severed_verdict(row, float("inf"), max_calibre_ratio, max_elbow_deg):
            continue  # fails on calibre or elbow, no threshold brings it back
        _, gained = paint_pedicle(row["path"], float(row["median_radius_mm"]),
                                  union, work_mask, spacing)
        priced.append((row["detour"], row["length_mm"],
                       float(gained.sum() * np.prod(spacing) / 1000.0)))

    missing = sum(row["length_mm"] for row in orphans)
    rows = []
    previous = (0, 0.0, 0.0)
    for threshold in sorted(thresholds):
        kept = [entry for entry in priced if entry[0] <= threshold]
        length = sum(entry[1] for entry in kept)
        volume = sum(entry[2] for entry in kept)
        added = length - previous[1]
        rows.append({"detour": float(threshold), "n": len(kept), "length_mm": length,
                     "reclaimed_ml": volume,
                     "share": length / missing if missing > 0 else 0.0,
                     "n_added": len(kept) - previous[0], "added_mm": added,
                     "step_ml_per_mm": (volume - previous[2]) / added if added > 0 else float("nan")})
        previous = (len(kept), length, volume)
    return rows


def print_detour_sweep(rows, missing):
    """
    Prints the detour sweep, cost column last because it is the deciding one.

    Rows that admit no new fragment are marked. A flat stretch of the table
    is not a plateau in the cost, it is an empty stretch of the detour
    distribution, and the two read identically if the marker is missing: a
    threshold picked in the middle of an empty range is arbitrary, whatever
    the mL/mm column next to it says.
    """
    print("\n=== severed fragments vs the detour threshold ===")
    print(f"  shares are of the {missing:.0f} mm of centerline outside the main tree, the same "
          f"denominator as the orphan report")
    print("  detour   n   recovered_mm   share_of_missing   reclaimed_mL |  +frags    +mm   "
          "mL/mm of the step")
    for row in rows:
        step = ("       -      -                  -   (nothing new)" if row["n_added"] == 0 else
                f"  {row['n_added']:6d} {row['added_mm']:6.1f} {row['step_ml_per_mm']:18.4f}")
        print(f"  {row['detour']:6.2f} {row['n']:3d} {row['length_mm']:14.1f} "
              f"{row['share']:18.1%} {row['reclaimed_ml']:14.2f} |{step}")

    steps = [row for row in rows if row["n_added"] > 0]
    if len(steps) < 2:
        print("  the threshold admits fragments in at most one step over this range: it is not "
              "being tested. Widen the range before reading anything into the cost column")
        return
    costs = [row["step_ml_per_mm"] for row in steps]
    if max(costs) < 2.0 * min(costs):
        print(f"  the marginal cost stays within a factor two across every step "
              f"({min(costs):.4f} to {max(costs):.4f} mL/mm), so it separates nothing here: "
              f"fragments admitted late cost about what the early ones cost. The threshold then "
              f"has to be argued from the detour distribution itself, not from this column")
    else:
        peak = max(steps, key=lambda row: row["step_ml_per_mm"])
        print(f"  the marginal cost peaks at detour {peak['detour']:.2f} "
              f"({peak['step_ml_per_mm']:.4f} mL/mm for {peak['added_mm']:.1f} mm)")
    print("  volume per millimetre of orphan recovered: a long fragment reached by a long thin "
          "tube is cheap by this measure and still a detour. It qualifies the detour criterion, "
          "it does not replace it")


def print_reclaim(rows, orphans, spacing, show=8):
    """
    Prints what the repair took back, against what it is a repair of.

    The share of the missing centerline reattached is printed next to the
    absolute figure, because the absolute figure alone reads as a result. A
    repair that puts back one percent of what is missing has not repaired the
    tree, and the before/after ratios that follow it are then measuring the
    stability of the fit against a small perturbation, not the effect of
    reconnection. Which is worth knowing, but is a different claim.
    """
    print(f"\n=== reconnection of severed pedicles ===")
    total = sum(row["reclaimed_ml"] for row in rows)
    length = sum(row["length_mm"] for row in rows)
    missing = sum(row["length_mm"] for row in orphans)
    print(f"  reclaimed {len(rows)} fragment(s), putting {length:.0f} mm of orphan centerline "
          f"back in reach of the root")
    print(f"  that is {length / missing:.1%} of the {missing:.0f} mm outside the main tree")
    print(f"  cost: {total:.2f} mL taken back from the other class "
          f"({total * 1000.0 / np.prod(spacing):.0f} voxels)")
    if rows:
        print("  frag   orphan_mm   gap_axis   path_mm   detour   reclaimed_mL")
        for row in sorted(rows, key=lambda r: -r["length_mm"])[:show]:
            print(f"  {row['component_id']:4d} {row['length_mm']:11.1f} {row['gap_axis_mm']:10.2f} "
                  f"{row['path_mm']:9.2f} {row['detour']:8.2f} {row['reclaimed_ml']:14.3f}")
        if len(rows) > show:
            print(f"  ... {len(rows) - show} more")
    if missing > 0 and length / missing < 0.10:
        print(f"  WARNING: under a tenth of the missing centerline was reattached. Read the "
              f"before/after ratios below as a sensitivity test -- how far the fit moves under a "
              f"perturbation this small -- and NOT as the impact of reconnection. If they move by "
              f"more than the perturbation, that is a result about the fit, not about the repair")


def synthetic_share(table):
    """
    How much of the centerline runs through voxels a repair put back.

    This is the number that decides whether the repaired tree can be quoted.
    A few percent is a pedicle restored: the reattached fragment is real
    vessel, measured on its own voxels, and only the crossing it hangs by is
    reconstructed. A large share means the centerline is largely tracing the
    reconstruction, and then the morphometry is describing the repair.
    """
    total = float(sum(entry["length_mm"] for entry in table))
    made = float(sum(entry["length_mm"] * entry["synthetic_fraction"] for entry in table))
    touched = [entry for entry in table if entry["synthetic_fraction"] > 0.0]
    return {"total_mm": total, "synthetic_mm": made, "n_branches": len(touched),
            "fraction": made / total if total > 0 else 0.0}


def print_synthetic(share):
    """Prints the synthetic share, and says what it licenses."""
    print(f"\n=== synthetic centerline ===")
    print(f"  {share['synthetic_mm']:.1f} mm of {share['total_mm']:.1f} mm "
          f"({share['fraction']:.2%}) runs through voxels the repair put back, "
          f"across {share['n_branches']} branch(es)")
    print("  every branch carries synthetic_fraction in --branches-csv and --elements-csv, and "
          "--max-synthetic drops them from the statistics while leaving the tree connected")
    if share["fraction"] > 0.10:
        print(f"  WARNING: over a tenth of the centerline is reconstructed. Re-run with "
              f"--max-synthetic 0.5 and check the ratios hold: above this share they may be "
              f"describing the repair rather than the tree")


def compare_measurable(before, after):
    """
    How many junctions became measurable, before and after the repair.

    A reconnected pedicle lengthens the segment it attaches to, and a segment
    that was too short to clear the junction blob can cross that threshold
    once it is whole again. So the repair is expected to buy back angles it
    was never aimed at, and that recovery is worth reporting on its own: it
    is the one effect of the reconnection that does not depend on any fitted
    slope.
    """
    def count(bifurcations):
        pairs = [b for b in bifurcations if b["angle_deg"] is not None]
        return len([b for b in pairs if b["angle_clean"]]), len(pairs)

    (clean_before, pairs_before), (clean_after, pairs_after) = count(before), count(after)
    print("\n=== measurable bifurcations before / after reconnection ===")
    print(f"  directions fitted clear of the blob: {clean_before}/{pairs_before} before, "
          f"{clean_after}/{pairs_after} after ({clean_after - clean_before:+d})")
    if clean_after > clean_before:
        print("  the repair bought back angles it was not aimed at: reattached pedicles lengthen "
              "the segments they join, and some now clear the direction-fit offset")


def compare_ratios(before, after, ordering, counting):
    """
    The three ratios before and after the repair, side by side.

    This difference is the measurement the repair exists to produce. R_l is
    the one to watch: a missing lateral branch fuses two segments into one,
    which inflates the mean length at the low orders and flattens the slope,
    so if the severed pedicles were really what depressed R_l, putting them
    back has to raise it. If it does not move, the missing bifurcations are
    elsewhere and the truncation explanation was the wrong one.
    """
    print(f"\n=== ratios before / after reconnection ({ordering}, {counting}s) ===")
    print("  ratio     before       after      change    before CI            after CI")
    for name, _, _ in RATIO_KEYS:
        old_fit, new_fit = before["fits"][name], after["fits"][name]
        if old_fit is None or new_fit is None:
            print(f"  {name:<8}  -")
            continue
        change = new_fit["ratio"] - old_fit["ratio"]
        print(f"  {name:<8} {old_fit['ratio']:7.3f}     {new_fit['ratio']:7.3f}   "
              f"{change:+8.3f}    [{old_fit['ratio_low']:5.3f}, {old_fit['ratio_high']:5.3f}]   "
              f"[{new_fit['ratio_low']:5.3f}, {new_fit['ratio_high']:5.3f}]")
    print(f"  fitted over {before['selection']} then {after['selection']}")
    if before["orders"] != after["orders"]:
        print(f"  WARNING: the fit rests on orders {before['orders']} before and {after['orders']} "
              f"after. The two numbers are not the same measurement -- reconnection changed which "
              f"orders clear the diameter floor. Pin the range with --fit-orders to compare")


def summarize(result, ordering, order_range, min_diameter, prespecified=False,
              max_synthetic=1.0, min_branches=3, flare_limit=1.5):
    """
    Numbers the branches with the chosen ordering, then aggregates them both
    ways -- one row per order for segments, one for elements -- and fits the
    ratios on each.

    Flared elements leave on the same footing and for the same reason: a
    row that starts on one vessel and ends on another is not a member of
    its order's population, so it comes out of the averages while the tree
    keeps it. Taking it out one row at a time is what makes the check
    usable -- excluding every ORDER that holds one destroys the orders that
    hold the most rows, which are the ones worth fitting.

    `max_synthetic` drops the branches that owe more than that share of
    their length to a repair, from the statistics only. The tree keeps them:
    they are what makes the reattached fragment reachable at all, so
    removing them from the graph would undo the ordering the repair was run
    to obtain. What is at stake is narrower -- whether the ratios are being
    carried by centerline through invented voxels -- and that is answered by
    leaving the structure alone and taking the pedicles out of the averages.
    """
    table, smooth = result["table"], result["smooth"]
    for entry in table:
        entry["order"] = entry[ordering]
    elements = build_elements(table, ordering, smooth)
    if max_synthetic < 1.0:
        keep = [e for e in table if e["synthetic_fraction"] <= max_synthetic]
        keep_elements = [e for e in elements if e["synthetic_fraction"] <= max_synthetic]
    else:
        keep, keep_elements = table, elements
    # a segment can never be flared (`flared` requires two members), so this
    # touches the element statistics only, which is where aggregation happens
    keep_elements = [e for e in keep_elements if not flared(e, flare_limit)]
    summaries = {"segment": order_summary(keep, "order", flare_limit),
                 "element": order_summary(keep_elements, "order", flare_limit)}
    ratios = {counting: branching_ratios(rows, ordering, order_range, min_diameter, prespecified,
                                         min_branches)
              for counting, rows in summaries.items()}
    return elements, summaries, ratios


def sweep_angle_offset(graph, table, smooth, radii, voxel_size, order_key, min_radius, offsets):
    """
    Re-measures the junction angles with the direction fitted from a range of
    distances off the node, in multiples of the local junction radius.

    Nothing but the direction window changes: the tree, the branches and the
    radii are the ones already built. The point is to show whether the angles
    sit on a plateau or slide with the window -- if they slide, the number
    quoted is a property of the fit, not of the anatomy. The count of
    measurable junctions falls as the offset grows, and reading the two
    columns together is the whole exercise: the offset to keep is the
    smallest one past which the angles stop moving.
    """
    rows = []
    for offset in offsets:
        shifted = [dict(entry) for entry in table]
        for entry in shifted:
            nodes = entry["nodes"]
            head, _, head_clean = branch_geometry(nodes, smooth, radii, radii[nodes[0]],
                                                  voxel_size, direction_offset=offset)
            tail, _, tail_clean = branch_geometry(nodes, smooth, radii, radii[nodes[-1]], voxel_size,
                                                  from_start=False, direction_offset=offset)
            entry["head_axis"], entry["head_axis_clean"] = head, head_clean
            entry["tail_axis"], entry["tail_axis_clean"] = tail, tail_clean
        bifurcations = analyze_bifurcations(shifted, order_key, min_radius)
        pairs = [b for b in bifurcations if b["angle_deg"] is not None]
        clean = [b for b in pairs if b["angle_clean"]]
        source = clean or pairs
        parent_child = [b["angle_parent_first_deg"] for b in source] + \
                       [b["angle_parent_second_deg"] for b in source]
        rows.append({
            "offset": float(offset), "n_pairs": len(pairs), "n_clean": len(clean),
            "parent_child_deg": float(np.median(parent_child)) if source else float("nan"),
            "children_deg": float(np.median([b["angle_deg"] for b in source])) if source else float("nan"),
            "defect_deg": float(np.median([b["planarity_defect_deg"] for b in source])) if source else float("nan"),
        })
    return rows


def print_angle_sweep(rows):
    """Prints the angle sweep, with what it costs in measurable junctions."""
    print("\n=== angle vs direction-fit offset ===")
    print("  the offset is the distance from the junction node at which the direction fit starts,")
    print("  in multiples of the local junction radius. 0 is the fit that starts inside the blob")
    print("  offset   measurable   parent-daughter   daughter-daughter   planarity defect")
    for row in rows:
        share = f"{row['n_clean']}/{row['n_pairs']}"
        print(f"  {row['offset']:6.2f}   {share:>10}   {row['parent_child_deg']:15.1f}   "
              f"{row['children_deg']:17.1f}   {row['defect_deg']:16.1f}")
    moving = [row for row in rows if row["n_clean"] >= 3]
    if len(moving) >= 2:
        spread = max(r["children_deg"] for r in moving) - min(r["children_deg"] for r in moving)
        print(f"  the daughter-daughter median moves {spread:.1f} deg across the sweep. Read the "
              f"offset off the plateau, not off the reference value")


def sweep_pruning(base_graph, positions, radii, world, voxel_size, args, factors_list):
    """
    Re-runs the whole post-skeleton stage for a range of pruning strengths.

    Pruning is the one free parameter of this pipeline that no measurement
    constrains, and it acts precisely on the terminal branches, i.e. on the
    lowest orders, i.e. on the steepest end of every semi-log fit. If the
    ratios move across a plausible range of k then they are a property of the
    pruning and not of the tree, and the sweep is the only thing that can
    tell the two apart. Report it next to the ratios, not instead of them.

    Both countings come out, one row each, for the same reason `print_ratios`
    prints both: segments and elements give different N and different mean
    lengths, so they give different R_b and R_l, and which of the two is
    stable under pruning is not something to decide by a default flag. On a
    tree whose segmental length fit is poor, the segmental sweep can only
    report that poor fit wobbling.
    """
    rows = []
    for k in factors_list:
        result = build_tree(base_graph, positions, radii, world, voxel_size, args, k)
        if result is None:
            rows.extend({"radius_factor": k, "counting": counting, "n_branches": 0,
                         "n_elements": 0, "n_leaves": 0, "R_b": None, "R_d": None, "R_l": None,
                         "r2_b": None, "r2_d": None, "r2_l": None,
                         "order_min": None, "order_max": None}
                        for counting in ("segment", "element"))
            continue
        elements, summaries, ratios = summarize(result, args.ordering, args.fit_orders,
                                                args.fit_floor_mm, args.prespecified,
                                                min_branches=args.fit_min_branches,
                                                flare_limit=args.element_flare)
        for counting in ("segment", "element"):
            fits = ratios[counting]["fits"]
            orders = ratios[counting]["orders"]
            rows.append({
                "radius_factor": k,
                "counting": counting,
                "n_branches": len(result["table"]),
                "n_elements": len(elements),
                "n_leaves": sum(1 for b in result["table"] if b["is_terminal"]),
                "R_b": fits["R_b"]["ratio"] if fits["R_b"] else None,
                "R_d": fits["R_d"]["ratio"] if fits["R_d"] else None,
                "R_l": fits["R_l"]["ratio"] if fits["R_l"] else None,
                "r2_b": fits["R_b"]["r2"] if fits["R_b"] else None,
                "r2_d": fits["R_d"]["r2"] if fits["R_d"] else None,
                "r2_l": fits["R_l"]["r2"] if fits["R_l"] else None,
                "order_min": min(orders) if orders else None,
                "order_max": max(orders) if orders else None,
            })
    return rows


SWEEP_MIN_R2 = 0.90


def print_sweep(all_rows, ordering):
    """
    Prints the ratios against the pruning strength, and their spread.

    The spread is computed only over the rows whose fit rests on the same
    orders. A row that gained or lost an order is not a different pruning of
    the same measurement, it is a different measurement: the fit range moved
    under it, and its ratio differs for that reason before any effect of the
    pruning. Pooling it into the range attributes to k something k did not do.

    And a ratio is only called pruning-driven when its fits are worth
    interpreting. A column of R2 near zero scatters as much as one likes
    without saying anything about the pruning -- the scatter is the fit
    failing, not k acting -- so it is reported as an unusable fit instead.
    """
    for counting in ("segment", "element"):
        print_sweep_table([row for row in all_rows if row["counting"] == counting],
                          ordering, counting)


def print_sweep_table(rows, ordering, counting):
    """One counting's worth of the pruning sweep."""
    print(f"\n=== pruning sensitivity ({ordering}, {counting}s) ===")
    print("    k  branches  elements  leaves   orders    R_b    R2     R_d    R2     R_l    R2")
    for row in rows:
        def cell(value, width=6, digits=3):
            return f"{value:{width}.{digits}f}" if value is not None else " " * (width - 1) + "-"
        span = (f"{row['order_min']:2d}..{row['order_max']:<2d}"
                if row["order_min"] is not None else "     -")
        print(f"{row['radius_factor']:5.2f} {row['n_branches']:9d} {row['n_elements']:9d} "
              f"{row['n_leaves']:7d}   {span}  {cell(row['R_b'])} {cell(row['r2_b'], 5, 3)} "
              f"{cell(row['R_d'])} {cell(row['r2_d'], 5, 3)} "
              f"{cell(row['R_l'])} {cell(row['r2_l'], 5, 3)}")
    if len({row["n_branches"] for row in rows}) < max(2, len(rows) // 2):
        print("  note: most of the sweep gave the same tree -- the absolute floor "
              "--min-branch-length is what is pruning, not k. Set it to 0 to sweep k alone")
    spans = [(row["order_min"], row["order_max"]) for row in rows if row["order_min"] is not None]
    modal = max(set(spans), key=spans.count) if spans else None
    comparable = [row for row in rows if (row["order_min"], row["order_max"]) == modal]
    dropped = [row for row in rows if row not in comparable and row["order_min"] is not None]
    if dropped:
        print(f"  the spread below is over the {len(comparable)} row(s) fitted on orders "
              f"{modal[0]}..{modal[1]}; k=" +
              ", ".join(f"{row['radius_factor']:.2f}" for row in dropped) +
              " left out, their fit rests on a different order range and their ratios differ for "
              "that reason before any effect of the pruning")

    for name, r2_key in (("R_b", "r2_b"), ("R_d", "r2_d"), ("R_l", "r2_l")):
        usable = [row for row in comparable if row[name] is not None]
        values = np.array([row[name] for row in usable], float)
        if values.size < 2:
            continue
        spread = float(values.max() - values.min()) / float(values.mean())
        best = max((row[r2_key] or 0.0) for row in usable)
        line = f"  {name}: {values.min():.3f}..{values.max():.3f} over the sweep ({spread:.1%} of its mean)"
        if best < SWEEP_MIN_R2:
            line += (f"   <- but no fit here clears R2 {SWEEP_MIN_R2:.2f} (best {best:.3f}): "
                     f"that scatter is the fit failing, not the pruning acting")
        elif spread > 0.10:
            line += "   <- driven by the pruning, not by the tree"
        print(line)


SWEEP_COLUMNS = ("radius_factor", "counting", "n_branches", "n_elements", "n_leaves",
                 "order_min", "order_max", "R_b", "r2_b", "R_d", "r2_d", "R_l", "r2_l")


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Pulmonary artery mask (nifti)")
    parser.add_argument("--label", type=int, default=None, help="Label value to isolate. Default: any nonzero voxel")
    parser.add_argument("--output", help="Centerline mask to write (nifti, input grid). Default: "
                                         "<input>_centerline.nii.gz next to the input mask")
    parser.add_argument("--csv", help="Per-point CSV to write")
    parser.add_argument("--branches-csv", help="Per-branch (segment) CSV to write")
    parser.add_argument("--elements-csv", help="Per-element CSV to write")
    parser.add_argument("--orders-csv", help="Per-order CSV to write, for the --count unit")
    parser.add_argument("--ratios-csv", help="R_b / R_d / R_l with their confidence intervals")
    parser.add_argument("--bifurcations-csv", help="Per-bifurcation CSV to write")
    parser.add_argument("--breakpoints-csv", help="CSV of the leaves that end too early, with their coordinates")
    parser.add_argument("--orphans-csv", help="One row per component outside the main tree, with "
                                              "its distance to the tree and its coordinates")
    parser.add_argument("--bridge-csv", help="The bridging curve, one row per dilation radius")
    parser.add_argument("--quality-csv", help="Single-row CSV of the quality metrics, to concatenate over cases")
    parser.add_argument("--sweep-csv", help="Ratios against the pruning strength, one row per k "
                                            "and counting (both are always swept)")
    parser.add_argument("--vtk", help="Legacy VTK polydata to write")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip the per-order, leaf and bifurcation tables. The ratios and the "
                             "quality metrics are printed either way")
    parser.add_argument("--ordering", choices=("generation", "strahler", "strahler_dd", "bfs_generation"),
                        default="generation",
                        help="How branches are numbered in the report and in --paint order. "
                             "generation: the widest daughter continues the parent (main path); "
                             "strahler: counted up from the tips; "
                             "strahler_dd: diameter-defined Strahler, the one to fit the ratios on; "
                             "bfs_generation: raw junction count from the root. Default: generation")
    parser.add_argument("--count", choices=("segment", "element"), default="segment",
                        help="Unit of the per-order table and of the ratios. A segment runs between "
                             "two bifurcations, an element is a run of segments of the same order. "
                             "Both are always fitted and both are printed, here and in "
                             "--sweep-k; this picks only the one the per-order table and "
                             "--orders-csv describe. Default: segment")
    parser.add_argument("--fit-orders", type=int, nargs=2, metavar=("MIN", "MAX"), default=None,
                        help="Orders the ratios are fitted over. Fix this before looking at the "
                             "results. Default: every order whose mean diameter clears "
                             "--fit-min-voxels")
    parser.add_argument("--prespecified", action="store_true",
                        help="Declare that the RULE fixing the fit range was chosen BEFORE the "
                             "per-order table was looked at. The rule is what can be pre-specified, "
                             "not the range: the diameter floor is mechanical, but it lands on a "
                             "different order for every spacing, so across a cohort the range moves "
                             "from subject to subject by design. Report it that way. Only you know "
                             "whether the rule came first -- the same --fit-orders is typed either "
                             "way -- so the program will not infer it. Sets prespecified=1 in "
                             "--ratios-csv; without it the range is recorded as undeclared")
    parser.add_argument("--fit-min-voxels", type=float, default=3.0,
                        help="An order whose mean diameter is under this many voxels is left out of "
                             "the fit: under three voxels of diameter the distance transform is "
                             "quantized to the grid and has no dynamic range left. Counted on the "
                             "COARSE axis of the acquired grid, which is the axis that decides "
                             "whether an order was resolved -- counting it on the resampled grid "
                             "would be the anisotropy ratio too permissive, and would admit an "
                             "order the acquisition never saw. Identical on an isotropic input, "
                             "where the three axes agree. Default: 3")
    parser.add_argument("--subject", default=None,
                        help="Identifier written into --ratios-csv, so the per-subject files of a "
                             "cohort concatenate into a table that can be read. Defaults to the "
                             "input filename")
    parser.add_argument("--fit-min-branches", type=int, default=3, metavar="N",
                        help="An order carried by fewer than N segments or elements is dropped "
                             "from the fit. At the top of a Strahler tree an order holds one or "
                             "two elements, its SD is zero by construction, and one heterogeneous "
                             "element is the whole of its mean -- see --element-flare. The rule is "
                             "mechanical and pre-specifiable, like the diameter floor, and it "
                             "censors the other end of the same range. Costs R_b the trunk-ward "
                             "orders that anchor its line, which is the price of fitting the "
                             "three ratios over one range. 1 disables it. Default: 3")
    parser.add_argument("--element-flare", type=float, default=1.5, metavar="RATIO",
                        help="Report every element whose distal calibre exceeds its proximal one "
                             "by more than this factor. A vessel tapers; a run several times wider "
                             "at its end is a chain of segments that kept one order across a "
                             "junction and got aggregated into one element, whose median calibre "
                             "summarises nothing. Default: 1.5")
    parser.add_argument("--fit-min-diameter", type=float, default=None, metavar="MM",
                        help="Diameter floor of the fit in millimetres, overriding the rule above "
                             "outright. Use it to hold a cohort to one floor when the volumes were "
                             "acquired at different spacings -- at the cost of the subjects whose "
                             "grid cannot support it, which then have to be dropped rather than "
                             "quietly fitted below their own resolution")
    parser.add_argument("--sweep-k", type=float, nargs="+", default=None, metavar="K",
                        help="Re-run the analysis for each of these --radius-factor values and "
                             "tabulate the ratios against them, e.g. --sweep-k 0.5 1 1.5 2 2.5 3")
    parser.add_argument("--paint", choices=("binary", "order", "branch"), default="binary",
                        help="Voxel value in the output mask. Default: binary")
    parser.add_argument("--smoothing", type=int, default=20,
                        help="Laplacian smoothing iterations on the centerline before measuring "
                             "lengths and angles. 0 measures the raw voxel path. Default: 20")
    parser.add_argument("--max-shift", type=float, default=None,
                        help="How far (mm) smoothing may move a point. Default: half a voxel")
    parser.add_argument("--breakpoint-order", type=int, default=1,
                        help="A terminal branch at this main-path order or below is suspect. Default: 1")
    parser.add_argument("--breakpoint-radius", type=float, default=None,
                        help="...provided its tip is wider than this (mm), i.e. it cannot just be "
                             "the tree fading out. Default: 2 voxels")
    parser.add_argument("--reconnect-severed", action="store_true",
                        help="Take back from the other class the voxels that reconnect every "
                             "severed fragment to the trunk, then re-run the whole morphometry on "
                             "the repaired mask and print the ratios before and after. Needs "
                             "--compare-label or --compare-mask, since severed is defined against "
                             "the other class")
    parser.add_argument("--repaired-mask", help="Where to write the mask --reconnect-severed "
                                                "produced, on the working grid")
    parser.add_argument("--max-synthetic", type=float, default=1.0, metavar="FRACTION",
                        help="Drop from the per-order statistics every branch owing more than this "
                             "share of its length to reclaimed voxels. The tree keeps them, so the "
                             "ordering is unchanged; only the averages lose them. 1.0 keeps "
                             "everything. Default: 1.0")
    parser.add_argument("--branches-csv-repaired",
                        help="Per-segment CSV of the tree AFTER --reconnect-severed, carrying "
                             "synthetic_fraction so the reconstructed centerline can be found again")
    parser.add_argument("--reconnect-csv", help="One row per severed fragment: whether it was "
                                                "reclaimed or refused, the union path length, the "
                                                "volume taken back and where to look")
    parser.add_argument("--severed-max-detour", type=float, default=3.0, metavar="RATIO",
                        help="Longest union path, as a multiple of the direct axis-to-axis gap, "
                             "that still counts as a mislabelled crossing. Existence of a path "
                             "proves nothing -- the other tree is connected -- so this is what "
                             "separates severed from hole. Use --severed-sweep to see what each "
                             "value buys and costs before moving it. Default: 2")
    parser.add_argument("--severed-sweep", type=float, nargs="+", default=None, metavar="RATIO",
                        help="Tabulate recovered length, reclaimed volume and mL per mm against "
                             "the detour threshold, e.g. --severed-sweep 1 1.5 2 2.5 3 4")
    parser.add_argument("--severed-max-calibre", type=float, default=2.0, metavar="RATIO",
                        help="Widest vessel the path may run through in the other class, as a "
                             "multiple of the fragment's own radius. Beyond it the route is going "
                             "down a vein, not across one. Default: 2")
    parser.add_argument("--severed-max-elbow", type=float, default=60.0, metavar="DEGREES",
                        help="Largest angle between the fragment's axis at the cut and the "
                             "direction to where the path attaches. A vessel continues into its "
                             "pedicle; a detour leaves at an angle. Default: 60")
    parser.add_argument("--angle-offset", type=float, default=DIRECTION_OFFSET,
                        help="Distance from a junction at which the branch direction fit starts, "
                             "in multiples of the local junction radius. Inside that ball the "
                             "centerline still bends out of the parent, and fitting from 0 inflates "
                             f"every angle. Default: {DIRECTION_OFFSET}")
    parser.add_argument("--angle-sweep", type=float, nargs="+", default=None, metavar="OFFSET",
                        help="Re-measure the junction angles at these direction-fit offsets and "
                             "tabulate them against the count of measurable junctions, "
                             "e.g. --angle-sweep 0 0.5 1 1.5 2 2.5")
    parser.add_argument("--murray-min-voxels", type=float, default=3.0,
                        help="Murray's exponent and the area ratio are only summarized over the "
                             "bifurcations whose parent and daughters are wider than this many "
                             "voxels. Default: 3")
    parser.add_argument("--min-branch-length", type=float, default=3.0,
                        help="Terminal branches shorter than this (mm) are pruned. Default: 3")
    parser.add_argument("--radius-factor", type=float, default=1.0,
                        help="Also prune a terminal branch shorter than this many local radii. 0 disables. Default: 1")
    parser.add_argument("--min-component-length", type=float, default=10.0,
                        help="Skeleton components shorter than this (mm) are dropped. Default: 10")
    parser.add_argument("--orphan-gap", type=float, default=3.0,
                        help="A component whose wall is within this many mm of the main tree is "
                             "counted as broken off rather than as a false positive. Default: 3")
    parser.add_argument("--compare-mask", help="A second segmentation (typically the venous "
                                               "prediction) to cross-check the orphan components "
                                               "against. Must be on the input grid. Omit it and "
                                               "pass --compare-label alone to take the other class "
                                               "out of the input file itself")
    parser.add_argument("--compare-label", type=int, default=None,
                        help="Label value to isolate for the comparison. Read from --compare-mask "
                             "if given, otherwise from --input, which is the usual case for a "
                             "multi-class segmentation: --label 4 --compare-label 3")
    parser.add_argument("--compare-dilate", type=float, default=1.0,
                        help="Voxels of dilation applied to --compare-mask before measuring how "
                             "much of a fragment lies inside it. Default: 1")
    parser.add_argument("--bridge-sweep", type=float, nargs="*", default=None, metavar="MM",
                        help="Dilate the mask by each of these radii (mm) and report how much "
                             "centerline ends up in the largest component. Replaces the "
                             "hand-placed --orphan-gap with the radius at which the tree actually "
                             "closes. Pass no value for one to six voxels")
    parser.add_argument("--dust-length", type=float, default=10.0,
                        help="An orphan component shorter than this (mm) is counted as speckle "
                             "rather than as a severed vessel or a hole. Default: 10")
    parser.add_argument("--keep-cycles", action="store_true",
                        help="Do not cut the loops of the skeleton. They are artefacts and they make "
                             "the Strahler orders downstream of them meaningless, so this is for "
                             "inspection only")
    parser.add_argument("--spacing", type=float, default=None,
                        help="Isotropic voxel size (mm) used for skeletonization. Default: smallest input spacing")
    parser.add_argument("--no-resample", action="store_true", help="Skeletonize on the input grid")
    parser.add_argument("--no-fill-holes", action="store_true", help="Do not fill internal cavities")
    parser.add_argument("--all-components", action="store_true", help="Do not restrict the mask to its largest component")
    parser.add_argument("--root", type=int, nargs=3, metavar=("I", "J", "K"),
                        help="Voxel (input grid) closest to the trunk, used as generation 0")
    args = parser.parse_args()

    # one multi-class file holds both trees more often than two files do, so
    # --compare-label alone reads the other class out of --input
    compare_path = args.compare_mask or (args.input if args.compare_label is not None else None)
    if compare_path == args.input and args.compare_label == args.label:
        raise SystemExit("--compare-label is the same class as --label; there is nothing to "
                         "compare the tree against but itself")

    mask, affine, spacing = load_mask(args.input, args.label)
    if not mask.any():
        raise SystemExit("mask is empty, nothing to extract")
    print(f"mask: {args.input}  shape={mask.shape}  spacing={np.round(spacing, 3).tolist()} mm")
    print(f"mask volume: {int(mask.sum())} voxels ({mask.sum() * np.prod(spacing) / 1000.0:.2f} mL)")

    # the mask is skeletonized whole: restricting it to its main component
    # before thinning would hide how much centerline is being dropped
    n_volume_components, volume_fraction = component_volume_fraction(mask)
    print(f"connected components: {n_volume_components}, largest holds {volume_fraction:.1%} of the volume")
    if not args.no_fill_holes:
        filled = binary_fill_holes(mask)
        print(f"filled {int(filled.sum() - mask.sum())} cavity voxels")
        mask = filled

    factors = np.ones(3)
    work_mask, work_affine, work_spacing = mask, affine, spacing
    if not args.no_resample:
        work_mask, work_affine, factors, work_spacing = resample_isotropic(mask, affine, spacing, args.spacing)
        if not np.allclose(factors, 1.0):
            print(f"resampled to {np.round(work_spacing, 3).tolist()} mm, shape={work_mask.shape}")
    # The floor of the fit is counted on the ACQUIRED grid, once, here, and
    # every later use reads it off args rather than recomputing it from a
    # voxel size that by then means the resampled one (see `fit_floor`).
    args.fit_floor_mm, floor_rule = fit_floor(spacing, args.fit_min_voxels, args.fit_min_diameter)
    print(f"acquired grid: {np.round(spacing, 3).tolist()} mm, "
          f"{spacing.max() / spacing.min():.2f}:1 anisotropy; diameter floor of the fit "
          f"{floor_rule}")
    args.factors = factors

    radius_map, skeleton, base_graph, positions, radii, voxel_size, world = skeletonize_graph(
        work_mask, work_affine, work_spacing)

    compare_distance = compare_inside = compare_union = compare_name = None
    if compare_path:
        other, other_affine, other_spacing = load_mask(compare_path, args.compare_label)
        if other.shape != mask.shape:
            raise SystemExit(f"--compare-mask has shape {other.shape}, the input has {mask.shape}; "
                             f"they have to be on the same grid")
        # counted before resampling: the working grid is isotropic at the
        # smallest input spacing, so its voxels are smaller and a count taken
        # there is not comparable to the input-grid count printed for --label
        compare_voxels = int(other.sum())
        compare_ml = compare_voxels * float(np.prod(other_spacing)) / 1000.0
        if not args.no_resample:
            other = resample_isotropic(other, other_affine, other_spacing, args.spacing)[0]
        if other.shape != work_mask.shape or not other.any():
            raise SystemExit("--compare-mask is empty or did not resample onto the working grid")
        compare_distance = distance_transform_edt(~other, sampling=work_spacing)
        compare_inside = compare_distance <= args.compare_dilate * voxel_size
        compare_union = union_labels(work_mask, other)
        compare_name = os.path.basename(compare_path)
        if compare_path == args.input:
            compare_name = f"label {args.compare_label} of the same file"
        print(f"compare mask: {compare_name}, {compare_voxels} voxels ({compare_ml:.2f} mL)")
    result = build_tree(base_graph, positions, radii, world, voxel_size, args, args.radius_factor)
    if result is None:
        raise SystemExit("nothing left after pruning, lower --min-branch-length or --radius-factor")

    graph, table, smooth, parts = result["graph"], result["table"], result["smooth"], result["parts"]
    print(f"pruned {result['pruned']} spur voxels, kept {graph.number_of_nodes()} skeleton voxels "
          f"in {nx.number_connected_components(graph)} component(s), dropped {result['dropped']} short ones")
    if result["broken"]:
        thinnest = min(r for _, _, r in result["broken"])
        print(f"cut {len(result['broken'])} loop(s) out of the skeleton "
              f"(thinnest cut at r={thinnest:.2f} mm) -- they are welds between touching "
              f"vessels, and the orders downstream of them would be undefined")
    if not result["dd_converged"]:
        print("diameter-defined Strahler did not converge; the last iteration is reported")

    elements, summaries, ratios = summarize(result, args.ordering, args.fit_orders,
                                            args.fit_floor_mm, args.prespecified,
                                            min_branches=args.fit_min_branches,
                                            flare_limit=args.element_flare)
    summary = summaries[args.count]
    bifurcations = analyze_bifurcations(table, "order", args.murray_min_voxels * voxel_size,
                                        positions, factors)

    lengths = np.array([b["length_mm"] for b in table])
    raw = sum(polyline_length(world[b["nodes"]]) for b in table)
    endpoints = sum(1 for _, d in graph.degree() if d == 1)
    junctions = sum(1 for _, d in graph.degree() if d >= 3)
    print(f"branches: {len(table)} segments in {len(elements)} elements  "
          f"endpoints: {endpoints}  junctions: {junctions}")
    print(f"total centerline length: {lengths.sum():.1f} mm  (longest branch {lengths.max():.1f} mm, "
          f"raw voxel path {raw:.1f} mm)")
    print(f"{args.ordering}: {min(r['order'] for r in summary)}..{max(r['order'] for r in summary)}  "
          f"radius: {radii.min():.2f}..{radii.max():.2f} mm")
    if not args.no_report:
        print_analysis(graph, table, summary, bifurcations, args.ordering,
                       args.murray_min_voxels * voxel_size, args.count)
    # the ratios are the point of the run, so they survive --no-report
    print_flare(elements, args.element_flare)
    for counting in ("segment", "element"):
        print_ratios(ratios[counting], args.ordering, counting)

    if args.angle_sweep:
        print_angle_sweep(sweep_angle_offset(graph, table, smooth, radii, voxel_size, "order",
                                             args.murray_min_voxels * voxel_size, args.angle_sweep))

    sweep = None
    if args.sweep_k:
        sweep = sweep_pruning(base_graph, positions, radii, world, voxel_size, args, args.sweep_k)
        print_sweep(sweep, args.ordering)

    breakpoint_radius = args.breakpoint_radius
    if breakpoint_radius is None:
        breakpoint_radius = 2.0 * voxel_size
    breakpoints = find_breakpoints(table, positions, smooth, factors, args.breakpoint_order,
                                   breakpoint_radius)
    orphans = analyze_orphans(parts, positions, world, radii, factors, args.orphan_gap,
                              compare_distance, compare_inside, compare_union)
    if compare_union is not None:
        # one Dijkstra, feeding both the severed/hole split and the repair
        union_paths(work_mask, other, orphans, positions, radii, work_spacing)
    orphans = classify_orphans(orphans, args.dust_length, args.severed_max_detour,
                               args.severed_max_calibre, args.severed_max_elbow)
    metrics = quality_metrics(graph, table, bifurcations, breakpoints, parts, volume_fraction,
                              n_volume_components, voxel_size, breakpoint_radius, args.breakpoint_order,
                              result["cycles"], len(result["broken"]), orphans,
                              float(mask.sum() * np.prod(spacing) / 1000.0), len(elements),
                              args.ordering)
    print_quality(metrics, breakpoints)
    print_orphans(orphans, args.orphan_gap, compare_name, args.dust_length)

    if args.severed_sweep:
        if compare_union is None:
            raise SystemExit("--severed-sweep needs a second class: pass --compare-label "
                             "or --compare-mask")
        print_detour_sweep(sweep_detour(orphans, work_mask, other, work_spacing,
                                        args.severed_sweep, args.dust_length,
                                        args.severed_max_calibre, args.severed_max_elbow),
                           sum(row["length_mm"] for row in orphans))

    if args.reconnect_severed:
        if compare_union is None:
            raise SystemExit("--reconnect-severed needs a second class to take the pedicles back "
                             "from: pass --compare-label or --compare-mask")
        repaired, added, reclaimed = reclaim_pedicles(work_mask, other, orphans, work_spacing)
        print_reclaim(reclaimed, orphans, work_spacing)
        if args.reconnect_csv:
            write_table_csv(args.reconnect_csv, reclaimed, RECLAIM_COLUMNS)
            print(f"wrote {args.reconnect_csv}")
        if any(row["repair"] == "reclaimed" for row in reclaimed):
            print(f"repaired mask volume: {int(repaired.sum())} voxels "
                  f"({repaired.sum() * np.prod(work_spacing) / 1000.0:.2f} mL on the working grid, "
                  f"was {work_mask.sum() * np.prod(work_spacing) / 1000.0:.2f} mL)")
            fixed = skeletonize_graph(repaired, work_affine, work_spacing)
            # which skeleton nodes of the repaired tree sit in reclaimed voxels
            born = added[tuple(np.rint(fixed[3]).astype(int).T)]
            fixed_result = build_tree(fixed[2], fixed[3], fixed[4], fixed[6], fixed[5], args,
                                      args.radius_factor, born)
            if fixed_result is None:
                print("  nothing survived pruning on the repaired mask, ratios unchanged")
            else:
                fixed_elements, _, fixed_ratios = summarize(
                    fixed_result, args.ordering, args.fit_orders,
                    args.fit_floor_mm, args.prespecified, args.max_synthetic,
                    min_branches=args.fit_min_branches, flare_limit=args.element_flare)
                print_synthetic(synthetic_share(fixed_result["table"]))
                fixed_bifurcations = analyze_bifurcations(
                    fixed_result["table"], "order", args.murray_min_voxels * voxel_size,
                    fixed[3], factors)
                compare_measurable(bifurcations, fixed_bifurcations)
                for counting in ("segment", "element"):
                    compare_ratios(ratios[counting], fixed_ratios[counting], args.ordering, counting)
                if args.branches_csv_repaired:
                    write_table_csv(args.branches_csv_repaired, fixed_result["table"], BRANCH_COLUMNS)
                    print(f"wrote {args.branches_csv_repaired}")
                if args.repaired_mask:
                    # label 1 is the original mask, label 2 the voxels taken
                    # back: the file says on its own what in it was invented
                    painted = repaired.astype(np.uint8)
                    painted[added] = 2
                    nib.save(nib.Nifti1Image(painted, work_affine), args.repaired_mask)
                    print(f"wrote {args.repaired_mask} (label 1 = segmented, "
                          f"label 2 = {int(added.sum())} reclaimed voxels)")
                    # this is the segmentation plus the pedicles, not the tree:
                    # only the tree is restricted to its main component, and
                    # the fragments the repair did not take are all still here
                    n_left = connected_components(repaired, structure=np.ones((3, 3, 3), int))[1]
                    print(f"  it holds {n_left} connected component(s): the file is the "
                          f"segmentation with the pedicles put back, not the analysed tree. "
                          f"Everything the repair did not reattach -- holes, dust, refused "
                          f"fragments -- is still in it, exactly as in the input")

    bridge = None
    if args.bridge_sweep is not None:
        dilations = sorted(args.bridge_sweep or list(voxel_size * np.arange(1.0, 7.0)))
        if dilations[0] > 0:
            dilations.insert(0, 0.0)
        bridge = bridging_curve(parts, positions, work_mask, work_spacing, dilations)
        print_bridging(bridge, voxel_size)

    output = args.output or default_output_path(args.input)
    volume = paint_centerline(table, positions, mask.shape, factors, args.paint)
    nib.save(nib.Nifti1Image(volume, affine), output)
    print(f"wrote {output} ({int((volume > 0).sum())} voxels, paint={args.paint})")
    if args.csv:
        write_points_csv(args.csv, table, positions, smooth, radii, factors)
        print(f"wrote {args.csv}")
    if args.branches_csv:
        write_table_csv(args.branches_csv, table, BRANCH_COLUMNS)
        print(f"wrote {args.branches_csv}")
    if args.elements_csv:
        write_table_csv(args.elements_csv, elements, ELEMENT_COLUMNS)
        print(f"wrote {args.elements_csv}")
    if args.orders_csv:
        write_table_csv(args.orders_csv, summary, ORDER_COLUMNS)
        print(f"wrote {args.orders_csv}")
    if args.ratios_csv:
        rows = [row for counting in ("segment", "element")
                for row in ratio_rows(ratios[counting], args.ordering, counting,
                                      spacing, args.fit_floor_mm,
                                      args.subject or os.path.basename(args.input))]
        write_table_csv(args.ratios_csv, rows, RATIO_COLUMNS)
        print(f"wrote {args.ratios_csv}")
    if args.bifurcations_csv:
        write_table_csv(args.bifurcations_csv, bifurcations, BIFURCATION_COLUMNS)
        print(f"wrote {args.bifurcations_csv}")
    if args.breakpoints_csv:
        write_table_csv(args.breakpoints_csv, breakpoints, BREAKPOINT_COLUMNS)
        print(f"wrote {args.breakpoints_csv}")
    if args.orphans_csv:
        write_table_csv(args.orphans_csv, orphans, ORPHAN_COLUMNS)
        print(f"wrote {args.orphans_csv}")
    if args.bridge_csv and bridge:
        write_table_csv(args.bridge_csv, bridge, BRIDGE_COLUMNS)
        print(f"wrote {args.bridge_csv}")
    if args.quality_csv:
        write_table_csv(args.quality_csv, [metrics], QUALITY_COLUMNS)
        print(f"wrote {args.quality_csv}")
    if args.sweep_csv and sweep:
        write_table_csv(args.sweep_csv, sweep, SWEEP_COLUMNS)
        print(f"wrote {args.sweep_csv}")
    if args.vtk:
        write_vtk(args.vtk, table, smooth, radii)
        print(f"wrote {args.vtk}")


if __name__ == "__main__":
    main()
