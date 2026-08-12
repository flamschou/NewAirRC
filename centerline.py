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
    6. estimate a local radius at every centerline point from the distance
       transform, and index branches by generation from the trunk

Outputs:
    --output        nifti centerline mask, on the input grid. Defaults to
                    <input>_centerline.nii.gz next to the input mask
    --csv           one row per centerline point (voxel + world mm + radius)
    --branches-csv  one row per branch (length, radius, generation)
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


def keep_largest_component(mask):
    """Drops every connected component but the biggest one (26-connectivity)."""
    components, n_components = connected_components(mask, structure=np.ones((3, 3, 3), dtype=int))
    if n_components <= 1:
        return mask, n_components
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return components == int(sizes.argmax()), n_components


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
        return mask, affine, np.ones(3), np.array([1.0, 1.0, 1.0])

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
def branch_geometry(nodes, world, radii, span_mm, from_start=True):
    """
    Direction and calibre of a branch as seen from one of its ends.

    Everything is measured `span_mm` away from that end -- one diameter of
    the vessel meeting there -- because near a junction the branches merge
    into a single blob: the radius is inflated by the neighbouring vessels
    and the direction still bends out of the parent. The radius is the
    median over [span, 2*span], falling back to the far end of the branch
    when it is too short.

    Returns (unit direction leaving the end, radius in mm).
    """
    order = nodes if from_start else nodes[::-1]
    points = world[order]
    distance = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])

    reach = min(int(np.searchsorted(distance, span_mm)), len(order) - 1)
    vector = points[reach] - points[0]
    norm = np.linalg.norm(vector)
    direction = vector / norm if norm > 0 else np.zeros(3)

    window = (distance >= min(span_mm, distance[-1])) & (distance <= 2.0 * span_mm)
    if not window.any():
        window = distance >= min(span_mm, distance[-1])
    return direction, float(np.median(radii[np.asarray(order)[window]]))


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


def branch_table(graph, ordered, world, radii):
    """
    Per-branch measurements, in branch order (proximal -> distal).

    Tortuosity is the ratio of the path length to the straight distance
    between the two ends: 1.0 is a straight branch.
    """
    table = []
    for branch_id, (nodes, generation) in enumerate(ordered):
        length = path_length(graph, nodes)
        chord = float(np.linalg.norm(world[nodes[-1]] - world[nodes[0]]))
        values = radii[nodes]
        table.append({
            "branch_id": branch_id,
            "generation": generation,
            "n_points": len(nodes),
            "length_mm": length,
            "chord_mm": chord,
            "tortuosity": length / chord if chord > 0 else 1.0,
            "mean_radius_mm": float(values.mean()),
            "min_radius_mm": float(values.min()),
            "max_radius_mm": float(values.max()),
            "proximal_radius_mm": float(radii[nodes[0]]),
            "distal_radius_mm": float(radii[nodes[-1]]),
            "is_terminal": int(graph.degree(nodes[-1]) == 1),
            "nodes": nodes,
        })
    return table


def analyze_bifurcations(ordered, table, world, radii):
    """
    Measures every bifurcation: the parent branch that ends there and the
    daughters that leave it.

    Returns one dict per junction with the parent/daughter radii, the area
    ratio (sum of daughter sections over the parent section), Murray's
    exponent and the angle between the two daughters.
    """
    parent_of = {}
    children_of = defaultdict(list)
    for branch_id, (nodes, _) in enumerate(ordered):
        parent_of[nodes[-1]] = branch_id
        children_of[nodes[0]].append(branch_id)

    bifurcations = []
    for node, children in children_of.items():
        if node not in parent_of or len(children) < 2:
            continue
        parent = table[parent_of[node]]
        # one junction diameter is the scale over which the vessels separate
        span = max(4.0, 2.0 * float(radii[node]))
        _, parent_radius = branch_geometry(parent["nodes"], world, radii, span, from_start=False)
        geometry = [branch_geometry(table[c]["nodes"], world, radii, span) for c in children]
        directions = [d for d, _ in geometry]
        child_radii = [r for _, r in geometry]

        angle = None
        if len(children) == 2:
            cosine = float(np.clip(np.dot(directions[0], directions[1]), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cosine)))

        bifurcations.append({
            "node": node,
            "generation": parent["generation"],
            "n_children": len(children),
            "parent_radius_mm": parent_radius,
            "child_radii_mm": sorted(child_radii, reverse=True),
            "area_ratio": float(np.sum(np.square(child_radii)) / parent_radius ** 2) if parent_radius > 0 else None,
            "asymmetry": float(min(child_radii) / max(child_radii)) if max(child_radii) > 0 else None,
            "murray_exponent": murray_exponent(parent_radius, child_radii),
            "angle_deg": angle,
        })
    return bifurcations


def generation_table(table):
    """Aggregates the branch measurements generation by generation."""
    rows = []
    for generation in sorted({b["generation"] for b in table}):
        branches = [b for b in table if b["generation"] == generation]
        terminal = [b for b in branches if b["is_terminal"]]
        lengths = np.array([b["length_mm"] for b in branches])
        rows.append({
            "generation": generation,
            "n_branches": len(branches),
            "n_terminal": len(terminal),
            "total_length_mm": float(lengths.sum()),
            "mean_length_mm": float(lengths.mean()),
            "mean_radius_mm": float(np.mean([b["mean_radius_mm"] for b in branches])),
            "mean_proximal_radius_mm": float(np.mean([b["proximal_radius_mm"] for b in branches])),
            "mean_distal_radius_mm": float(np.mean([b["distal_radius_mm"] for b in branches])),
            "mean_tortuosity": float(np.mean([b["tortuosity"] for b in branches])),
            "mean_tip_radius_mm": float(np.mean([b["distal_radius_mm"] for b in terminal])) if terminal else None,
        })
    return rows


def describe(values):
    """mean / median / p10 / p90 / range of a sample, as a printable string."""
    values = np.asarray([v for v in values if v is not None], float)
    if values.size == 0:
        return "n/a"
    p10, p90 = np.percentile(values, (10, 90))
    return (f"mean {values.mean():6.2f}  median {np.median(values):6.2f}  "
            f"p10-p90 {p10:5.2f}-{p90:5.2f}  range {values.min():5.2f}-{values.max():5.2f}")


def print_analysis(graph, table, generations, bifurcations):
    """Prints the anatomical report: per generation, leaves, bifurcations."""
    print("\n=== branches per generation ===")
    print("gen    n  term   length_mm  mean_len  mean_rad  prox_rad  dist_rad  tort  tip_rad")
    for row in generations:
        tip = f"{row['mean_tip_radius_mm']:7.2f}" if row["mean_tip_radius_mm"] is not None else "      -"
        print(f"{row['generation']:3d} {row['n_branches']:4d} {row['n_terminal']:5d} "
              f"{row['total_length_mm']:11.1f} {row['mean_length_mm']:9.1f} "
              f"{row['mean_radius_mm']:9.2f} {row['mean_proximal_radius_mm']:9.2f} "
              f"{row['mean_distal_radius_mm']:9.2f} {row['mean_tortuosity']:5.2f} {tip}")

    leaves = [b for b in table if b["is_terminal"]]
    print(f"\n=== terminal branches ({len(leaves)} leaves) ===")
    if leaves:
        print(f"tip radius (mm)   : {describe([b['distal_radius_mm'] for b in leaves])}")
        print(f"mean radius (mm)  : {describe([b['mean_radius_mm'] for b in leaves])}")
        print(f"length (mm)       : {describe([b['length_mm'] for b in leaves])}")
        print(f"generation        : {describe([b['generation'] for b in leaves])}")

    junctions = sum(1 for _, d in graph.degree() if d >= 3)
    print(f"\n=== bifurcations ({len(bifurcations)} measured / {junctions} junctions) ===")
    if bifurcations:
        daughters = [b["n_children"] for b in bifurcations]
        trifurcations = sum(1 for n in daughters if n > 2)
        print(f"daughters per junction: mean {np.mean(daughters):.2f} "
              f"({trifurcations} junctions with more than 2)")
        print(f"parent radius (mm): {describe([b['parent_radius_mm'] for b in bifurcations])}")
        print(f"area ratio        : {describe([b['area_ratio'] for b in bifurcations])}")
        print(f"asymmetry ratio   : {describe([b['asymmetry'] for b in bifurcations])}")
        print(f"Murray exponent   : {describe([b['murray_exponent'] for b in bifurcations])}")
        print(f"angle (deg)       : {describe([b['angle_deg'] for b in bifurcations])}")

    print("\n=== tree ===")
    lengths = np.array([b["length_mm"] for b in table])
    print(f"tortuosity        : {describe([b['tortuosity'] for b in table])}")
    print(f"branch length (mm): {describe(lengths)}")
    reachable = [row for row in generations if row["generation"] >= 0]
    if len(reachable) > 1:
        # geometric growth up to the widest generation: past that peak the
        # tree stops splitting and the ratio only measures how it dies out
        peak = max(range(len(reachable)), key=lambda k: reachable[k]["n_branches"])
        if peak > 0:
            growth = (reachable[peak]["n_branches"] / reachable[0]["n_branches"]) ** (1.0 / peak)
            print(f"growth ratio      : {growth:.2f} branches per generation "
                  f"up to gen {reachable[peak]['generation']} (widest)")
    cycles = graph.number_of_edges() - graph.number_of_nodes() + nx.number_connected_components(graph)
    if cycles:
        print(f"loops             : {cycles} cycle(s) in the skeleton -- vessels touching each "
              f"other in the mask, which also shifts the generations downstream")
    orphans = next((row for row in generations if row["generation"] < 0), None)
    if orphans:
        print(f"unreachable from the root: {orphans['n_branches']} branches "
              f"({orphans['total_length_mm']:.1f} mm), reported as generation -1")


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #
def paint_centerline(ordered, positions, shape, factors, mode="binary"):
    """
    Rasterizes the branches onto a volume of the given shape, joining
    consecutive points with a discrete 3D line so the result stays connected
    even when the working grid is finer than the output grid.
    """
    volume = np.zeros(shape, dtype=np.uint8)
    for branch_id, (path, generation) in enumerate(ordered):
        # unreachable branches (generation -1) are painted 255 rather than dropped
        value = {"binary": 1, "generation": generation + 1 if generation >= 0 else 255,
                 "branch": branch_id % 255 + 1}[mode]
        voxels = np.rint((positions[path] + 0.5) / factors - 0.5).astype(int)
        voxels = np.clip(voxels, 0, np.array(shape) - 1)
        for start, stop in zip(voxels, voxels[1:]):
            volume[line_nd(start, stop, endpoint=True)] = value
        volume[tuple(voxels[-1])] = value
    return volume


def write_points_csv(path, ordered, positions, radii, affine, factors):
    """One row per centerline point, in branch order."""
    rows = ["branch_id,generation,point_index,i,j,k,x_mm,y_mm,z_mm,radius_mm"]
    for branch_id, (nodes, generation) in enumerate(ordered):
        voxels = positions[nodes]
        world = voxels @ affine[:3, :3].T + affine[:3, 3]
        source = np.rint((voxels + 0.5) / factors - 0.5).astype(int)
        for k, node in enumerate(nodes):
            i, j, l = source[k]
            x, y, z = world[k]
            rows.append(f"{branch_id},{generation},{k},{i},{j},{l},{x:.3f},{y:.3f},{z:.3f},{radii[node]:.3f}")
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


BRANCH_COLUMNS = ("branch_id", "generation", "n_points", "length_mm", "chord_mm", "tortuosity",
                  "mean_radius_mm", "min_radius_mm", "max_radius_mm",
                  "proximal_radius_mm", "distal_radius_mm", "is_terminal")

GENERATION_COLUMNS = ("generation", "n_branches", "n_terminal", "total_length_mm", "mean_length_mm",
                      "mean_radius_mm", "mean_proximal_radius_mm", "mean_distal_radius_mm",
                      "mean_tortuosity", "mean_tip_radius_mm")

BIFURCATION_COLUMNS = ("node", "generation", "n_children", "parent_radius_mm",
                       "area_ratio", "asymmetry", "murray_exponent", "angle_deg")


def write_vtk(path, ordered, positions, radii, affine):
    """Legacy ASCII VTK polydata: one polyline per branch, radius as scalar."""
    points, lines, scalars, offset = [], [], [], 0
    for nodes, _ in ordered:
        world = positions[nodes] @ affine[:3, :3].T + affine[:3, 3]
        points.extend(f"{x:.4f} {y:.4f} {z:.4f}" for x, y, z in world)
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
    parser.add_argument("--generations-csv", help="Per-generation CSV to write")
    parser.add_argument("--bifurcations-csv", help="Per-bifurcation CSV to write")
    parser.add_argument("--vtk", help="Legacy VTK polydata to write")
    parser.add_argument("--no-report", action="store_true", help="Skip the printed anatomical report")
    parser.add_argument("--paint", choices=("binary", "generation", "branch"), default="binary",
                        help="Voxel value in the output mask. Default: binary")
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

    if not args.all_components:
        mask, n_components = keep_largest_component(mask)
        print(f"connected components: {n_components}, kept the largest ({int(mask.sum())} voxels)")
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
    dropped = drop_small_components(graph, args.min_component_length)
    print(f"pruned {pruned} spur voxels, dropped {dropped} small components")
    if graph.number_of_nodes() == 0:
        raise SystemExit("nothing left after pruning, lower --min-branch-length / --min-component-length")

    branches = extract_branches(graph)
    root_voxel = None
    if args.root is not None:
        root_voxel = np.asarray(args.root, float) * factors + 0.5 * (factors - 1.0)
    ordered = order_branches(graph, branches, positions, radii, root_voxel)

    world = positions @ work_affine[:3, :3].T + work_affine[:3, 3]
    table = branch_table(graph, ordered, world, radii)
    generations = generation_table(table)
    bifurcations = analyze_bifurcations(ordered, table, world, radii)

    lengths = np.array([b["length_mm"] for b in table])
    depth = max(row["generation"] for row in generations)
    endpoints = sum(1 for _, d in graph.degree() if d == 1)
    junctions = sum(1 for _, d in graph.degree() if d >= 3)
    print(f"branches: {len(table)}  endpoints: {endpoints}  junctions: {junctions}")
    print(f"total centerline length: {lengths.sum():.1f} mm  (longest branch {lengths.max():.1f} mm)")
    print(f"generations: 0..{depth}  radius: {radii.min():.2f}..{radii.max():.2f} mm")
    if not args.no_report:
        print_analysis(graph, table, generations, bifurcations)

    output = args.output or default_output_path(args.input)
    volume = paint_centerline(ordered, positions, mask.shape, factors, args.paint)
    nib.save(nib.Nifti1Image(volume, affine), output)
    print(f"wrote {output} ({int((volume > 0).sum())} voxels, paint={args.paint})")
    if args.csv:
        write_points_csv(args.csv, ordered, positions, radii, work_affine, factors)
        print(f"wrote {args.csv}")
    if args.branches_csv:
        write_table_csv(args.branches_csv, table, BRANCH_COLUMNS)
        print(f"wrote {args.branches_csv}")
    if args.generations_csv:
        write_table_csv(args.generations_csv, generations, GENERATION_COLUMNS)
        print(f"wrote {args.generations_csv}")
    if args.bifurcations_csv:
        write_table_csv(args.bifurcations_csv, bifurcations, BIFURCATION_COLUMNS)
        print(f"wrote {args.bifurcations_csv}")
    if args.vtk:
        write_vtk(args.vtk, ordered, positions, radii, work_affine)
        print(f"wrote {args.vtk}")


if __name__ == "__main__":
    main()
