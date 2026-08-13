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
    5. turn the skeleton voxels into a graph, merge junction clusters and
       prune the short spurious side branches created by thinning
    6. smooth the branches, within the digitization error, before measuring
       anything: a raw voxel path is ~10% longer than the vessel it follows
    7. estimate a local radius from the distance transform and number the
       branches, either along the main path (the widest daughter continues
       its parent) or by Strahler order -- see `compute_orders`

Outputs:
    --output        nifti centerline mask, on the input grid. Defaults to
                    <input>_centerline.nii.gz next to the input mask
    --csv           one row per centerline point (voxel + world mm + radius)
    --branches-csv  one row per branch (length, radii, generation, Strahler)
    --orders-csv    one row per generation / Strahler order
    --bifurcations-csv  one row per junction (angle, area ratio, Murray)
    --vtk           legacy VTK polydata polylines, for Slicer / ParaView

Usage:
    python centerline.py --input artery.nii.gz
    python centerline.py --input seg.nii.gz --label 2 --csv points.csv --vtk cl.vtk
"""
import argparse
import itertools
import os
from collections import defaultdict, deque

import networkx as nx
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, zoom
from scipy.ndimage import label as connected_components
from scipy.optimize import brentq
from skimage.draw import line_nd
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
    Resamples the mask to isotropic voxels (nearest neighbour).

    Returns the new mask, its affine and the voxel->voxel mapping used later
    to project the centerline back onto the input grid. scipy's `grid_mode`
    convention is `in = (out + 0.5) / factor - 0.5`.
    """
    target = float(spacing.min()) if target is None else float(target)
    factors = spacing / target
    if np.allclose(factors, 1.0, atol=1e-3):
        return mask, affine, np.ones(3), spacing.copy()

    resampled = zoom(mask.astype(np.uint8), factors, order=0, grid_mode=True, mode="nearest") > 0

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


def branch_geometry(nodes, world, radii, junction_radius, voxel_size, from_start=True):
    """
    Direction and calibre of a branch as seen from one of its ends.

    Both are measured away from that end, because at a junction the branches
    merge into a single blob: the radius there is inflated by the
    neighbouring vessels and the direction still bends out of the parent.
    The two use different scales, and both are clamped to the branch so a
    wide vessel is never measured at its far end:

    - the direction is fitted over 2.5 local radii (at most half the
      branch). Over a couple of voxels only it would snap to the axes of the
      grid, which piles the bifurcation angles up at 90 degrees. It is the
      first principal component of the points in that window, not the chord
      to its last point, so every point contributes.
    - the calibre is the median radius over [1, 2] local radii (at most the
      proximal half of the branch), just past the junction blob and close
      enough that the vessel has not tapered yet.

    Returns (unit direction leaving the end, radius in mm).
    """
    order = np.asarray(nodes if from_start else nodes[::-1])
    points = world[order]
    distance = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    total = float(distance[-1])

    span = min(max(2.5 * float(junction_radius), 3.0 * voxel_size), max(0.5 * total, 3.0 * voxel_size))
    sample = points[distance <= span]
    if len(sample) < 2:
        sample = points[:2]
    axis = np.linalg.svd(sample - sample.mean(axis=0), full_matrices=False)[2][0]
    # principal components have no sign: point it away from the junction
    direction = axis * np.sign(np.dot(axis, sample[-1] - sample[0]) or 1.0)

    low = min(max(float(junction_radius), 2.0 * voxel_size), max(0.25 * total, 2.0 * voxel_size))
    high = min(max(2.0 * float(junction_radius), low + 2.0 * voxel_size),
               max(0.5 * total, low + 2.0 * voxel_size))
    window = (distance >= min(low, total)) & (distance <= high)
    if not window.any():
        window = distance >= min(low, total)
    return direction, float(np.median(radii[order[window]]))


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


def branch_table(graph, ordered, smooth, radii, voxel_size):
    """
    Per-branch measurements, in branch order (proximal -> distal), read on
    the smoothed centerline.

    Tortuosity is the ratio of the path length to the straight distance
    between the two ends: 1.0 is a straight branch. The proximal and distal
    calibres are measured away from the junctions (see `branch_geometry`),
    unlike `mean_radius_mm` which averages the whole branch.
    """
    table = []
    for branch_id, (nodes, generation) in enumerate(ordered):
        points = smooth[nodes]
        length = polyline_length(points)
        chord = float(np.linalg.norm(points[-1] - points[0]))
        values = radii[nodes]

        head_axis, head_calibre = branch_geometry(nodes, smooth, radii, radii[nodes[0]], voxel_size)
        tail_axis, tail_calibre = branch_geometry(nodes, smooth, radii, radii[nodes[-1]], voxel_size,
                                                  from_start=False)

        table.append({
            "branch_id": branch_id,
            "bfs_generation": generation,
            "n_points": len(nodes),
            "length_mm": length,
            "chord_mm": chord,
            "tortuosity": length / chord if chord > 0 else 1.0,
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


def analyze_bifurcations(table, order_key, min_radius):
    """
    Measures every bifurcation: the parent branch that ends there and the
    daughters that leave it. Radii and directions come from the branch
    table, i.e. measured a couple of radii away from the junction.

    `well_resolved` flags the junctions where the parent and every daughter
    are wider than `min_radius` (a few voxels). Below that the radii
    saturate on the voxel size and the derived quantities -- the area ratio
    and above all Murray's exponent, which is a ratio raised to a power --
    stop meaning anything.
    """
    bifurcations = []
    for parent in table:
        children = [table[c] for c in parent["children"]]
        if len(children) < 2:
            continue
        parent_radius = parent["distal_calibre_mm"]
        child_radii = [c["proximal_calibre_mm"] for c in children]

        angle = None
        if len(children) == 2:
            cosine = float(np.clip(np.dot(children[0]["head_axis"], children[1]["head_axis"]), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cosine)))

        bifurcations.append({
            "node": parent["nodes"][-1],
            "order": parent[order_key],
            "n_children": len(children),
            "parent_radius_mm": parent_radius,
            "min_child_radius_mm": float(min(child_radii)),
            "area_ratio": float(np.sum(np.square(child_radii)) / parent_radius ** 2) if parent_radius > 0 else None,
            "asymmetry": float(min(child_radii) / max(child_radii)) if max(child_radii) > 0 else None,
            "murray_exponent": murray_exponent(parent_radius, child_radii),
            "angle_deg": angle,
            "well_resolved": int(min(parent_radius, *child_radii) >= min_radius),
        })
    return bifurcations


def order_summary(table, order_key):
    """Aggregates the branch measurements order by order."""
    rows = []
    for order in sorted({b[order_key] for b in table}):
        branches = [b for b in table if b[order_key] == order]
        terminal = [b for b in branches if b["is_terminal"]]
        lengths = np.array([b["length_mm"] for b in branches])
        rows.append({
            "order": order,
            "n_branches": len(branches),
            "n_terminal": len(terminal),
            "total_length_mm": float(lengths.sum()),
            "mean_length_mm": float(lengths.mean()),
            "mean_radius_mm": float(np.mean([b["mean_radius_mm"] for b in branches])),
            "mean_proximal_calibre_mm": float(np.mean([b["proximal_calibre_mm"] for b in branches])),
            "mean_distal_calibre_mm": float(np.mean([b["distal_calibre_mm"] for b in branches])),
            "mean_tortuosity": float(np.mean([b["tortuosity"] for b in branches])),
            "mean_tip_radius_mm": float(np.mean([b["tip_radius_mm"] for b in terminal])) if terminal else None,
        })
    return rows


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


def quality_metrics(graph, table, bifurcations, breakpoints, parts, volume_fraction,
                    n_volume_components, voxel_size, min_radius, max_order):
    """
    The five numbers that say whether a segmentation can be trusted.

    They are independent on purpose: 1 and 4 are topological (is the tree in
    one piece, and does it wrongly loop back on itself), 2 says how deep it
    goes, 3 whether the vessels have a plausible calibre, 5 where it breaks.
    A defect in one implies nothing about the others.
    """
    total_length = sum(length for length, _ in parts)
    leaves = [b for b in table if b["is_terminal"]]
    floor = resolution_floor(voxel_size)
    at_floor = [b for b in leaves if b["tip_radius_mm"] <= floor * 1.001]
    murray = np.array([b["murray_exponent"] for b in bifurcations
                       if b["well_resolved"] and b["murray_exponent"] is not None])
    cycles = graph.number_of_edges() - graph.number_of_nodes() + nx.number_connected_components(graph)

    return {
        "n_components": len(parts),
        "n_volume_components": n_volume_components,
        "largest_component_volume_fraction": volume_fraction,
        "largest_component_length_fraction": parts[0][0] / total_length if total_length else 0.0,
        "length_outside_largest_mm": total_length - parts[0][0],
        "n_fragments_over_10mm": sum(1 for length, _ in parts[1:] if length >= 10.0),
        "n_leaves": len(leaves),
        "resolution_floor_mm": floor,
        "leaves_at_floor_fraction": len(at_floor) / len(leaves) if leaves else 0.0,
        "n_murray": int(murray.size),
        "murray_median": float(np.median(murray)) if murray.size else None,
        "murray_q1": float(np.percentile(murray, 25)) if murray.size else None,
        "murray_q3": float(np.percentile(murray, 75)) if murray.size else None,
        "n_cycles": int(cycles),
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
         f"{metrics['n_fragments_over_10mm']} of them over 10 mm)")
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
         "(an artery tree has no anastomosis, expected 0)")
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
    if order_key == "strahler":
        rows = rows[::-1]
    return [(before["order"], after["order"], before["mean_radius_mm"], after["mean_radius_mm"])
            for before, after in zip(rows, rows[1:]) if after["mean_radius_mm"] > before["mean_radius_mm"]]


def print_analysis(graph, table, summary, bifurcations, order_key, min_radius):
    """Prints the anatomical report: per order, leaves, bifurcations, tree."""
    label = {"generation": "generation (main path)", "strahler": "Strahler order",
             "bfs_generation": "generation (junctions from the root)"}[order_key]
    print(f"\n=== branches per {label} ===")
    print("ord    n  term   length_mm  mean_len  mean_rad  prox_cal  dist_cal  tort  tip_rad")
    for row in summary:
        tip = f"{row['mean_tip_radius_mm']:7.2f}" if row["mean_tip_radius_mm"] is not None else "      -"
        print(f"{row['order']:3d} {row['n_branches']:4d} {row['n_terminal']:5d} "
              f"{row['total_length_mm']:11.1f} {row['mean_length_mm']:9.1f} "
              f"{row['mean_radius_mm']:9.2f} {row['mean_proximal_calibre_mm']:9.2f} "
              f"{row['mean_distal_calibre_mm']:9.2f} {row['mean_tortuosity']:5.2f} {tip}")

    inversions = check_monotonicity(summary, order_key)
    if inversions:
        print(f"calibre monotonicity: VIOLATED at {len(inversions)} step(s) -- " +
              ", ".join(f"{a}->{b} ({ra:.2f} -> {rb:.2f} mm)" for a, b, ra, rb in inversions))
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
        print(f"angle (deg)       : {describe([b['angle_deg'] for b in bifurcations])}")
        print(f"\nrestricted to the {len(resolved)} bifurcations where the parent and both "
              f"daughters exceed {min_radius:.2f} mm:")
        if resolved:
            print(f"area ratio        : {describe([b['area_ratio'] for b in resolved])}")
            print(f"Murray exponent   : {describe([b['murray_exponent'] for b in resolved])}")
            print(f"angle (deg)       : {describe([b['angle_deg'] for b in resolved])}")
        else:
            print("  none -- the mask does not resolve any vessel well enough")

    print("\n=== tree ===")
    lengths = np.array([b["length_mm"] for b in table])
    print(f"tortuosity        : {describe([b['tortuosity'] for b in table])}")
    print(f"branch length (mm): {describe(lengths)}")
    reachable = [row for row in summary if row["order"] >= 0]
    if order_key != "strahler" and len(reachable) > 1:
        # geometric growth up to the widest order: past that peak the tree
        # stops splitting and the ratio only measures how it dies out
        peak = max(range(len(reachable)), key=lambda k: reachable[k]["n_branches"])
        if peak > 0:
            growth = (reachable[peak]["n_branches"] / reachable[0]["n_branches"]) ** (1.0 / peak)
            print(f"growth ratio      : {growth:.2f} branches per {order_key} "
                  f"up to {reachable[peak]['order']} (widest)")
    cycles = graph.number_of_edges() - graph.number_of_nodes() + nx.number_connected_components(graph)
    if cycles:
        print(f"loops             : {cycles} cycle(s) in the skeleton -- vessels touching each "
              f"other in the mask, which also shifts the generations downstream")
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


BRANCH_COLUMNS = ("branch_id", "generation", "strahler", "bfs_generation", "n_points",
                  "length_mm", "chord_mm", "tortuosity",
                  "mean_radius_mm", "min_radius_mm", "max_radius_mm",
                  "proximal_calibre_mm", "distal_calibre_mm", "tip_radius_mm", "is_terminal")

ORDER_COLUMNS = ("order", "n_branches", "n_terminal", "total_length_mm", "mean_length_mm",
                 "mean_radius_mm", "mean_proximal_calibre_mm", "mean_distal_calibre_mm",
                 "mean_tortuosity", "mean_tip_radius_mm")

BIFURCATION_COLUMNS = ("node", "order", "n_children", "parent_radius_mm", "min_child_radius_mm",
                       "area_ratio", "asymmetry", "murray_exponent", "angle_deg", "well_resolved")

BREAKPOINT_COLUMNS = ("branch_id", "generation", "strahler", "tip_radius_mm", "length_mm",
                      "i", "j", "k", "x_mm", "y_mm", "z_mm")

QUALITY_COLUMNS = ("largest_component_length_fraction", "largest_component_volume_fraction",
                   "length_outside_largest_mm", "n_components", "n_fragments_over_10mm",
                   "leaves_at_floor_fraction", "n_leaves", "resolution_floor_mm",
                   "murray_median", "murray_q1", "murray_q3", "n_murray",
                   "n_cycles", "n_breakpoints", "breakpoints_fraction")


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
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Pulmonary artery mask (nifti)")
    parser.add_argument("--label", type=int, default=None, help="Label value to isolate. Default: any nonzero voxel")
    parser.add_argument("--output", help="Centerline mask to write (nifti, input grid). Default: "
                                         "<input>_centerline.nii.gz next to the input mask")
    parser.add_argument("--csv", help="Per-point CSV to write")
    parser.add_argument("--branches-csv", help="Per-branch CSV to write")
    parser.add_argument("--orders-csv", help="Per-order (generation or Strahler) CSV to write")
    parser.add_argument("--bifurcations-csv", help="Per-bifurcation CSV to write")
    parser.add_argument("--breakpoints-csv", help="CSV of the leaves that end too early, with their coordinates")
    parser.add_argument("--quality-csv", help="Single-row CSV of the quality metrics, to concatenate over cases")
    parser.add_argument("--vtk", help="Legacy VTK polydata to write")
    parser.add_argument("--no-report", action="store_true", help="Skip the printed anatomical report")
    parser.add_argument("--ordering", choices=("generation", "strahler", "bfs_generation"), default="generation",
                        help="How branches are numbered in the report and in --paint order. "
                             "generation: the widest daughter continues the parent (main path); "
                             "strahler: counted up from the tips; "
                             "bfs_generation: raw junction count from the root. Default: generation")
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
    parser.add_argument("--spacing", type=float, default=None,
                        help="Isotropic voxel size (mm) used for skeletonization. Default: smallest input spacing")
    parser.add_argument("--no-resample", action="store_true", help="Skeletonize on the input grid")
    parser.add_argument("--no-fill-holes", action="store_true", help="Do not fill internal cavities")
    parser.add_argument("--all-components", action="store_true", help="Do not restrict the mask to its largest component")
    parser.add_argument("--root", type=int, nargs=3, metavar=("I", "J", "K"),
                        help="Voxel (input grid) closest to the trunk, used as generation 0")
    args = parser.parse_args()

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

    # the EDT stops at the last inside voxel centre, so the wall sits about
    # half a voxel further out
    radius_map = distance_transform_edt(work_mask, sampling=work_spacing) + 0.5 * work_spacing.min()
    skeleton = skeletonize(work_mask) > 0  # older skimage returns 0/255 uint8 in 3D
    print(f"skeleton: {int(skeleton.sum())} voxels")

    graph, positions = build_voxel_graph(skeleton, work_spacing)
    contract_junction_clusters(graph, positions, work_mask)
    update_edge_weights(graph, positions, work_spacing)
    radii = radius_map[tuple(np.rint(positions).astype(int).T)]

    pruned = prune_spurs(graph, radii, args.min_branch_length, args.radius_factor)
    if graph.number_of_nodes() == 0:
        raise SystemExit("nothing left after pruning, lower --min-branch-length")

    # measure the fragmentation before dropping anything, then restrict
    parts = component_lengths(graph)
    if not args.all_components:
        graph.remove_nodes_from(set().union(*(nodes for _, nodes in parts[1:])) if len(parts) > 1 else [])
    dropped = drop_small_components(graph, args.min_component_length)
    print(f"pruned {pruned} spur voxels, kept {graph.number_of_nodes()} skeleton voxels "
          f"in {nx.number_connected_components(graph)} component(s), dropped {dropped} short ones")
    if graph.number_of_nodes() == 0:
        raise SystemExit("nothing left, lower --min-component-length")

    branches = extract_branches(graph)
    root_voxel = None
    if args.root is not None:
        root_voxel = np.asarray(args.root, float) * factors + 0.5 * (factors - 1.0)
    ordered = order_branches(graph, branches, positions, radii, root_voxel)

    voxel_size = float(work_spacing.min())
    world = positions @ work_affine[:3, :3].T + work_affine[:3, 3]
    smooth = smooth_centerline(ordered, world, voxel_size, args.smoothing, args.max_shift)
    table = compute_orders(branch_table(graph, ordered, smooth, radii, voxel_size))
    for entry in table:
        entry["order"] = entry[args.ordering]
    summary = order_summary(table, "order")
    bifurcations = analyze_bifurcations(table, "order", args.murray_min_voxels * voxel_size)

    lengths = np.array([b["length_mm"] for b in table])
    raw = sum(polyline_length(world[b["nodes"]]) for b in table)
    endpoints = sum(1 for _, d in graph.degree() if d == 1)
    junctions = sum(1 for _, d in graph.degree() if d >= 3)
    print(f"branches: {len(table)}  endpoints: {endpoints}  junctions: {junctions}")
    print(f"total centerline length: {lengths.sum():.1f} mm  (longest branch {lengths.max():.1f} mm, "
          f"raw voxel path {raw:.1f} mm)")
    print(f"{args.ordering}: {min(r['order'] for r in summary)}..{max(r['order'] for r in summary)}  "
          f"radius: {radii.min():.2f}..{radii.max():.2f} mm")
    if not args.no_report:
        print_analysis(graph, table, summary, bifurcations, args.ordering,
                       args.murray_min_voxels * voxel_size)

    breakpoint_radius = args.breakpoint_radius
    if breakpoint_radius is None:
        breakpoint_radius = 2.0 * voxel_size
    breakpoints = find_breakpoints(table, positions, smooth, factors, args.breakpoint_order,
                                   breakpoint_radius)
    metrics = quality_metrics(graph, table, bifurcations, breakpoints, parts, volume_fraction,
                              n_volume_components, voxel_size, breakpoint_radius, args.breakpoint_order)
    print_quality(metrics, breakpoints)

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
    if args.orders_csv:
        write_table_csv(args.orders_csv, summary, ORDER_COLUMNS)
        print(f"wrote {args.orders_csv}")
    if args.bifurcations_csv:
        write_table_csv(args.bifurcations_csv, bifurcations, BIFURCATION_COLUMNS)
        print(f"wrote {args.bifurcations_csv}")
    if args.breakpoints_csv:
        write_table_csv(args.breakpoints_csv, breakpoints, BREAKPOINT_COLUMNS)
        print(f"wrote {args.breakpoints_csv}")
    if args.quality_csv:
        write_table_csv(args.quality_csv, [metrics], QUALITY_COLUMNS)
        print(f"wrote {args.quality_csv}")
    if args.vtk:
        write_vtk(args.vtk, table, smooth, radii)
        print(f"wrote {args.vtk}")


if __name__ == "__main__":
    main()
