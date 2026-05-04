import pandas as pd
import numpy as np
import geopandas as gpd
import triangle as tr
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString
from shapely.ops import unary_union, snap, linemerge
import collections
import rasterio
import os
from pathlib import Path


def sample_elevations_from_dem(points_xy, dem_path):
    """Samples elevation values for a list of (x, y) points from a DEM."""
    print(f"  - Reading DEM: {dem_path}")
    with rasterio.open(dem_path) as src:
        coords = [tuple(p) for p in points_xy]
        elevations = [val[0] for val in src.sample(coords)]
        print(f"  - Sampled {len(elevations)} points.")
        return np.array(elevations)

def _normalize_coords(xy, target_span=1e4):
    xy = np.asarray(xy, float)
    origin = xy.min(axis=0)
    shifted = xy - origin
    span = shifted.max(axis=0)
    scale = max(span.max() / target_span, 1.0)
    return shifted / scale, origin, scale

def _denormalize_coords(xy_norm, origin, scale):
    return np.asarray(xy_norm) * scale + origin


def _cumlens(coords):
    d = np.diff(coords, axis=0)
    seg = np.sqrt((d * d).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return s, seg

def _remove_spikes(coords, angle_tol_deg):
    if len(coords) <= 2:
        return coords
    keep = [coords[0]]
    for i in range(1, len(coords) - 1):
        a, b, c = coords[i - 1], coords[i], coords[i + 1]
        u, v = b - a, c - b
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu == 0 or nv == 0:
            continue
        cosang = float(np.dot(u, v) / (nu * nv))
        if not (cosang > 0.9999 or cosang < -0.9999):
            keep.append(b)
    keep.append(coords[-1])
    return np.asarray(keep)

def _node_streams(lines, tau_snap=1.0):
    """Snap near-coincident vertices and node all intersections once."""
    ml = linemerge(unary_union(snap(unary_union(lines), unary_union(lines), tau_snap)))
    if isinstance(ml, LineString):
        return [ml]
    return list(ml.geoms)

def _resample_with_min_spacing(line, spacing, m_min, angle_tol_deg=2.0, keep_original_vertices=False):
    coords = np.asarray(line.coords, float)
    if coords.ndim == 2 and coords.shape[1] > 2:
        coords = coords[:, :2]
    coords = _remove_spikes(coords, angle_tol_deg) if angle_tol_deg > 0 else coords
    s, seg = _cumlens(coords)
    L = float(s[-1])
    if L == 0:
        return coords[:1]

    s_grid = np.arange(0.0, L + 1e-9, spacing)
    s_orig = s if keep_original_vertices else np.array([0.0, L])
    all_s = np.unique(np.round(np.concatenate([s_orig, s_grid]), 9))

    kept = []
    seg_idx = 0
    for t in all_s:
        while seg_idx < len(seg) - 1 and s[seg_idx + 1] < t - 1e-12:
            seg_idx += 1
        k = np.where(np.isclose(s, t, rtol=0, atol=1e-9))[0]
        if k.size:
            pt = coords[k[0]]
        else:
            t0, t1 = s[seg_idx], s[seg_idx + 1]
            if t1 <= t0:
                continue
            a = (t - t0) / (t1 - t0)
            pt = coords[seg_idx] * (1 - a) + coords[seg_idx + 1] * a
        if not kept or np.linalg.norm(pt - kept[-1]) >= m_min - 1e-12:
            kept.append(pt)

    if np.linalg.norm(kept[-1] - coords[-1]) > 1e-9:
        kept.append(coords[-1])
    out = [kept[0]]
    for p in kept[1:]:
        if np.linalg.norm(p - out[-1]) >= m_min - 1e-12:
            out.append(p)
    if np.linalg.norm(out[-1] - coords[-1]) > 1e-9:
        out.append(coords[-1])
    return np.asarray(out)

def _build_stream_pslg(resampled_edges, m_min, dedupe_radius):
    clean_edges = []
    for arr in (resampled_edges or []):
        if arr is None:
            continue
        a = np.asarray(arr, float)
        if a.ndim != 2 or a.shape[0] < 2:
            continue
        if a.shape[1] > 2:
            a = a[:, :2]
        clean_edges.append(a)

    if not clean_edges:
        return np.empty((0, 2), dtype=float), []

    P = np.vstack(clean_edges)
    kdt = cKDTree(P)
    pairs = kdt.query_pairs(r=max(float(dedupe_radius), 1e-9))

    parent = np.arange(len(P))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in pairs:
        union(i, j)

    rep_to_new = {}
    new_index = np.empty(len(P), dtype=int)
    uniq = []
    for i in range(len(P)):
        r = find(i)
        if r not in rep_to_new:
            rep_to_new[r] = len(uniq)
            uniq.append(P[r])
        new_index[i] = rep_to_new[r]
    uniq = np.asarray(uniq, float)

    segs = set()
    cursor = 0
    for arr in clean_edges:
        n = len(arr)
        idxs = new_index[cursor:cursor + n]
        cursor += n
        prev = idxs[0]
        for cur in idxs[1:]:
            if cur != prev:
                segs.add((min(prev, cur), max(prev, cur)))
            prev = cur

    tiny = max(0.5, 0.25 * float(dedupe_radius))
    segments = [(i, j) for i, j in segs if np.linalg.norm(uniq[i] - uniq[j]) >= tiny - 1e-12]
    return uniq, segments

# ==============================================================================
# The Main Meshing Function
# ==============================================================================

def generate_mesh_from_points(
    points_file,
    stream_shapefile,
    watershed_shapefile,
    dem_file,
    boundary_buffer_dist,
    boundary_spacing,
    tree_centroids_shapefile=None,
    tree_remove_radius=5.0,
    stream_clearance_radius=15.0,
    stream_point_spacing=25.0,
    mesh_quality_opts='p'
):
    # -------------------------------------------------------------------------
    # Parameter validation
    # -------------------------------------------------------------------------
    MIN_BUFFER_DIST = 30.0
    if boundary_buffer_dist < MIN_BUFFER_DIST:
        raise ValueError(
            f"boundary_buffer_dist={boundary_buffer_dist}m is below the minimum allowed "
            f"value of {MIN_BUFFER_DIST}m. A buffer this small places the no-flow boundary "
            f"too close to the watershed edge and is unlikely to produce a valid tRIBS mesh."
        )

    if boundary_spacing > 2 * boundary_buffer_dist:
        optimal_spacing = 2 * boundary_buffer_dist
        optimal_buffer  = boundary_spacing / 2
        print(
            f"\nWARNING: boundary_spacing ({boundary_spacing}m) is more than 2× "
            f"boundary_buffer_dist ({boundary_buffer_dist}m). Concave corners of the "
            f"watershed boundary may be under-sampled, causing boundary segments to cut "
            f"back into the watershed.\n"
            f"  To fix, choose ONE of:\n"
            f"    - Reduce boundary_spacing to <= {optimal_spacing:.0f}m "
            f"(optimal for your current buffer)\n"
            f"    - Increase boundary_buffer_dist to >= {optimal_buffer:.0f}m "
            f"(optimal for your current spacing)"
            f"\n  If the recommendations above are not followed then a detailed check of the output shapefiles is required to verify that boundary nodes (Code 1) are surrounding the entire TIN, no interior nodes (Code 0) are on the boundary.\n"
        )

    # Read DEM resolution once here so stream spacing guidance can reference it
    with rasterio.open(dem_file) as _src:
        _dem_res = max(_src.res[0], _src.res[1])  # take the coarser axis for non-square pixels

    # -------------------------------------------------------------------------
    # Step 1: Buffered boundary nodes from watershed shapefile
    # -------------------------------------------------------------------------
    print("\nStep 1: Generating buffered boundary from watershed shapefile...")
    watershed_gdf = gpd.read_file(watershed_shapefile)
    watershed_geom = watershed_gdf.geometry.unary_union
    buffered_geom = watershed_geom.buffer(boundary_buffer_dist)
    ext = buffered_geom.exterior  # already-ordered polygon ring

    total_bnd_len = ext.length
    n_bnd = max(3, int(round(total_bnd_len / boundary_spacing)))
    bnd_pts = np.array([
        [ext.interpolate(i / n_bnd, normalized=True).x,
         ext.interpolate(i / n_bnd, normalized=True).y]
        for i in range(n_bnd)
    ])
    bnd_codes = np.ones(n_bnd, dtype=int)
    print(f"  {n_bnd} boundary nodes at ~{boundary_spacing}m spacing (buffer={boundary_buffer_dist}m)")

    # -------------------------------------------------------------------------
    # Step 2: Stream network — node, locate outlet, resample, build PSLG
    # -------------------------------------------------------------------------
    print("\nStep 2: Building stream network...")
    streams_gdf = gpd.read_file(stream_shapefile)
    raw_lines = []
    for g in streams_gdf.geometry.dropna():
        if isinstance(g, LineString):
            raw_lines.append(g)
        elif g.geom_type == "MultiLineString":
            raw_lines.extend(list(g.geoms))

    lines = _node_streams(raw_lines, tau_snap=1.0)

    # Collect all degree-1 endpoints (potential outlet or tributary inlets)
    endpoint_count = collections.Counter()
    for ln in lines:
        c = list(ln.coords)
        endpoint_count[tuple(c[0][:2])] += 1
        endpoint_count[tuple(c[-1][:2])] += 1
    degree1_pts = [np.array(pt) for pt, cnt in endpoint_count.items() if cnt == 1]
    if not degree1_pts:
        raise RuntimeError("Stream network has no degree-1 endpoints; cannot locate outlet.")

    # Read points file once: code-2 node used as outlet spatial hint, code-0 kept for interior
    base_df = pd.read_csv(points_file, sep=r"\s+", skiprows=1, header=None,
                          names=['x', 'y', 'z', 'code'], comment='#')

    # Stream spacing guidance: compare user setting against DEM resolution and
    # the mean nearest-neighbour distance of the interior terrain points.
    _interior_xy = base_df[base_df['code'] == 0][['x', 'y']].values
    if len(_interior_xy) > 1:
        _nn_dists, _ = cKDTree(_interior_xy).query(_interior_xy, k=2)
        _mean_nn = float(_nn_dists[:, 1].mean())
    else:
        _mean_nn = None
    
    outlet_rows = base_df[base_df['code'] == 2]

    if not outlet_rows.empty:
        hint_xy = outlet_rows.iloc[0][['x', 'y']].values.astype(float)
        dists = [np.linalg.norm(pt - hint_xy) for pt in degree1_pts]
    else:
        # Fall back: degree-1 endpoint nearest the actual watershed boundary
        ws_boundary = watershed_geom.boundary
        dists = [ws_boundary.distance(Point(pt)) for pt in degree1_pts]

    stream_outlet_xy = degree1_pts[int(np.argmin(dists))]
    print(f"  Stream outlet: ({stream_outlet_xy[0]:.2f}, {stream_outlet_xy[1]:.2f})")

    # Nearest buffered boundary node becomes the outlet (code 2)
    bnd_kdt = cKDTree(bnd_pts)
    _, outlet_bnd_local_idx = bnd_kdt.query(stream_outlet_xy)
    outlet_bnd_local_idx = int(outlet_bnd_local_idx)
    bnd_codes[outlet_bnd_local_idx] = 2
    outlet_bnd_xy = bnd_pts[outlet_bnd_local_idx]
    print(f"  Outlet boundary node: idx={outlet_bnd_local_idx} "
          f"({outlet_bnd_xy[0]:.2f}, {outlet_bnd_xy[1]:.2f})")

    # Resample stream edges
    MIN_STREAM_SPACING = 5.0
    DEDUPE_RADIUS = max(1.5, 0.05 * float(stream_point_spacing))
    resampled_edges = []
    for ln in lines:
        pts = _resample_with_min_spacing(
            ln, spacing=stream_point_spacing, m_min=MIN_STREAM_SPACING,
            angle_tol_deg=2.0, keep_original_vertices=False
        )
        if len(pts) >= 2:
            resampled_edges.append(pts)

    unified_stream_geom = linemerge(unary_union(lines))
    total_len_km = sum(ln.length for ln in lines) / 1000.0

    stream_nodes_xy, stream_segments_local = _build_stream_pslg(
        resampled_edges, m_min=MIN_STREAM_SPACING, dedupe_radius=DEDUPE_RADIUS
    )
    stream_nodes_df = pd.DataFrame(stream_nodes_xy, columns=['x', 'y'])
    stream_nodes_df['code'] = 3
    print(f"  {len(stream_nodes_df)} stream nodes from {total_len_km:.2f} km "
          f"at ~{stream_point_spacing}m spacing")

    # Longest stream flow path via two-sweep BFS on the assembled PSLG graph
    if stream_segments_local:
        _sadj = collections.defaultdict(list)
        for _i, _j in stream_segments_local:
            _d = float(np.linalg.norm(stream_nodes_xy[_i] - stream_nodes_xy[_j]))
            _sadj[_i].append((_j, _d))
            _sadj[_j].append((_i, _d))

        def _bfs_far(adj, src):
            dist = {src: 0.0}
            q = collections.deque([src])
            far, dmax = src, 0.0
            while q:
                v = q.popleft()
                for u, d in adj[v]:
                    if u not in dist:
                        dist[u] = dist[v] + d
                        if dist[u] > dmax:
                            dmax, far = dist[u], u
                        q.append(u)
            return far, dmax

        _u, _ = _bfs_far(_sadj, next(iter(_sadj)))
        _, _longest_m = _bfs_far(_sadj, _u)
        _longest_km = _longest_m / 1000.0
    else:
        _longest_km = 0.0

    print(f"\n  Stream spacing guidance (your setting: {stream_point_spacing}m):")
    print(f"    DEM resolution       : {_dem_res:.1f}m  ← minimum meaningful spacing")
    if _mean_nn is not None:
        print(f"    Mean interior spacing: {_mean_nn:.1f}m  ← suggested maximum")
        print(f"    Suggested range      : {_dem_res:.1f}m – {_mean_nn:.1f}m")
    print(f"    Longest flow path    : {_longest_km:.2f}km (from assembled breakline)")
    if stream_point_spacing < _dem_res:
        print(
            f"  WARNING: stream_point_spacing ({stream_point_spacing}m) is finer than the "
            f"DEM resolution ({_dem_res:.1f}m). This adds detail that does not exist in the "
            f"source data. Consider increasing to at least {_dem_res:.1f}m."
        )
    if _mean_nn is not None and stream_point_spacing > _mean_nn:
        print(
            f"  WARNING: stream_point_spacing ({stream_point_spacing}m) is coarser than "
            f"the mean interior point spacing ({_mean_nn:.1f}m). The stream network will "
            f"be represented at a lower resolution than the surrounding terrain. "
            f"Consider decreasing to at most {_mean_nn:.1f}m."
        )

    # -------------------------------------------------------------------------
    # Step 3: Interior points (code 0 only from points file)
    # -------------------------------------------------------------------------
    print("\nStep 3: Preparing interior points...")
    interior_pts = base_df[base_df['code'] == 0][['x', 'y', 'code']].copy().reset_index(drop=True)

    # Remove interior points too close to each other
    INTERIOR_REMOVE_RADIUS = 5.0
    if len(interior_pts) > 1:
        int_kdt = cKDTree(interior_pts[['x', 'y']].values)
        close_pairs = int_kdt.query_pairs(r=INTERIOR_REMOVE_RADIUS)
        drop_set = {j for _, j in close_pairs}
        if drop_set:
            print(f"  Removing {len(drop_set)} interior points within {INTERIOR_REMOVE_RADIUS}m of another.")
            interior_pts = interior_pts.drop(index=list(drop_set)).reset_index(drop=True)

    # Optional tree centroids
    tree_nodes_df = pd.DataFrame(columns=['x', 'y', 'code'])
    if tree_centroids_shapefile:
        print("  Processing tree centroids...")
        trees_gdf = gpd.read_file(tree_centroids_shapefile)
        tree_pts_xy = np.array([[p.x, p.y] for p in trees_gdf.geometry])
        if tree_pts_xy.size > 0:
            tree_kdt = cKDTree(tree_pts_xy)
            indices_near_tree = tree_kdt.query_ball_point(
                interior_pts[['x', 'y']].values, r=tree_remove_radius
            )
            drop_set = {interior_pts.index[i] for i, nbrs in enumerate(indices_near_tree) if nbrs}
            if drop_set:
                print(f"  Removing {len(drop_set)} interior points within {tree_remove_radius}m of a tree.")
                interior_pts = interior_pts.drop(list(drop_set)).reset_index(drop=True)
            tree_nodes_df = pd.DataFrame(tree_pts_xy, columns=['x', 'y'])
            tree_nodes_df['code'] = 0

    frames = [df for df in [interior_pts, tree_nodes_df] if not df.empty]
    all_interior = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['x', 'y', 'code'])

    # Stream clearance zone on interior points only
    if stream_clearance_radius > 0 and not all_interior.empty:
        print(f"  Applying {stream_clearance_radius}m stream clearance zone...")
        interior_gs = gpd.GeoSeries.from_xy(all_interior['x'], all_interior['y'])
        dist_to_stream = interior_gs.distance(unified_stream_geom)
        too_close = dist_to_stream < stream_clearance_radius
        n_removed = int(too_close.sum())
        if n_removed:
            print(f"  Removing {n_removed} interior points inside clearance zone.")
            all_interior = all_interior[~too_close.values].reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Step 4: Combine all points and sample elevations
    # -------------------------------------------------------------------------
    print("\nStep 4: Combining points and sampling elevations from DEM...")
    bnd_df = pd.DataFrame({'x': bnd_pts[:, 0], 'y': bnd_pts[:, 1], 'code': bnd_codes})
    # Boundary nodes first: their row indices (0..n_bnd-1) are stable in final_input_df
    final_non_stream_pts = pd.concat([bnd_df, all_interior], ignore_index=True)
    n_non_stream = len(final_non_stream_pts)

    final_input_df = pd.concat([final_non_stream_pts, stream_nodes_df], ignore_index=True)
    all_xy = final_input_df[['x', 'y']].values
    final_input_df['z'] = sample_elevations_from_dem(all_xy, dem_file)

    # -------------------------------------------------------------------------
    # Step 5: PSLG segments
    # -------------------------------------------------------------------------
    print("\nStep 5: Defining mesh constraints...")

    # Boundary ring: consecutive pairs close the polygon
    boundary_segs = [(i, (i + 1) % n_bnd) for i in range(n_bnd)]

    # Stream segments shifted into final_input_df index space
    stream_segs = [(i + n_non_stream, j + n_non_stream) for i, j in stream_segments_local]

    # Outlet reach: single 2-node segment connecting the stream outlet to the
    # buffered boundary outlet node. stream_outlet_xy is a preserved endpoint
    # of the stream network so it exists in stream_nodes_xy.
    stream_kdt = cKDTree(stream_nodes_xy)
    _, stream_outlet_local_idx = stream_kdt.query(stream_outlet_xy)
    stream_outlet_global_idx = int(stream_outlet_local_idx) + n_non_stream
    outlet_reach_seg = (outlet_bnd_local_idx, stream_outlet_global_idx)

    all_segs = set(tuple(sorted(s)) for s in boundary_segs + stream_segs + [outlet_reach_seg])
    segments = [list(s) for s in all_segs]

    # -------------------------------------------------------------------------
    # Step 6: Triangulate
    # -------------------------------------------------------------------------
    print("\nStep 6: Triangulating...")
    Vn, origin_xy, scale_xy = _normalize_coords(all_xy)
    # YY: no Steiner points on any PSLG segment (boundary ring or stream segments).
    # Interior Steiner points are still added as needed by the quality options.
    # This ensures every stream-coded node is an original PSLG node with a
    # monotonically-enforced elevation, not a DEM-raw Steiner interpolation.
    mesh = tr.triangulate({'vertices': Vn, 'segments': segments}, f"p{mesh_quality_opts}DjYY")
    mesh_vertices_xy = _denormalize_coords(np.asarray(mesh['vertices']), origin_xy, scale_xy)

    # -------------------------------------------------------------------------
    # Step 7: Elevations for Steiner vertices
    # -------------------------------------------------------------------------
    print("\nStep 7: Assigning elevations to mesh vertices...")
    mesh_vertices_z = sample_elevations_from_dem(mesh_vertices_xy, dem_file)

    # -------------------------------------------------------------------------
    # Step 7b: Enforce monotonic descent along stream network
    # -------------------------------------------------------------------------
    # FlowDirs() in tRIBS marks any node with steepest-slope <= 0 as kSink.
    # At 10m DEM resolution stream nodes often land in the same raster cell,
    # producing flat or inverted segments that trigger lakelist errors.  Walk
    # the assembled PSLG from the outlet outward (upstream) and nudge any
    # upstream node that is not strictly higher than its downstream neighbour.
    print("\nStep 7b: Enforcing monotonic descent along stream network...")
    if stream_segments_local:
        _tmp_kdt = cKDTree(mesh_vertices_xy)
        _, _stream_mesh_idx = _tmp_kdt.query(stream_nodes_xy)
        _mono_z = mesh_vertices_z[_stream_mesh_idx].copy()

        _mono_adj = collections.defaultdict(list)
        for _mi, _mj in stream_segments_local:
            _mono_adj[_mi].append(_mj)
            _mono_adj[_mj].append(_mi)

        MIN_STREAM_GRADIENT = 0.001  # 1 mm per meter minimum slope
        _outlet_local = int(stream_outlet_local_idx)
        # The outlet terminal often samples a bank elevation rather than the
        # thalweg (especially at 10 m DEM resolution).  Starting BFS from it
        # would incorrectly raise all upstream nodes.  Since FillLakes always
        # drains to kOpenBoundary regardless of elevation, the terminal's own
        # elevation is irrelevant,skip it and root the BFS at its upstream
        # neighbour instead.
        _outlet_upstream = _mono_adj[_outlet_local]
        if _outlet_upstream:
            _bfs_root = _outlet_upstream[0]
            _mono_visited = {_outlet_local, _bfs_root}
            _mono_queue = collections.deque([_bfs_root])
        else:
            _bfs_root = _outlet_local
            _mono_visited = {_outlet_local}
            _mono_queue = collections.deque([_outlet_local])
        _n_adjusted = 0

        while _mono_queue:
            _dn = _mono_queue.popleft()          # downstream node
            for _up in _mono_adj[_dn]:
                if _up not in _mono_visited:
                    _mono_visited.add(_up)
                    _seg_len = float(np.linalg.norm(stream_nodes_xy[_up] - stream_nodes_xy[_dn]))
                    _min_z = _mono_z[_dn] + max(_seg_len * MIN_STREAM_GRADIENT, 0.01)
                    if _mono_z[_up] <= _mono_z[_dn]:
                        _mono_z[_up] = _min_z
                        _n_adjusted += 1
                    _mono_queue.append(_up)

        mesh_vertices_z[_stream_mesh_idx] = _mono_z
        if _n_adjusted:
            print(f"  Adjusted {_n_adjusted} of {len(stream_nodes_xy)} stream nodes "
                  f"(min gradient enforced: {MIN_STREAM_GRADIENT * 1000:.1f} mm/m).")
        else:
            print("  Stream elevations already monotonically decreasing — no adjustments needed.")
    else:
        print("  No stream segments — skipping.")

    final_vertices_3d = np.column_stack([mesh_vertices_xy, mesh_vertices_z])

    # -------------------------------------------------------------------------
    # Step 8: Node codes
    # -------------------------------------------------------------------------
    print("\nStep 8: Assigning node codes...")
    n_mesh_verts = len(mesh_vertices_xy)
    final_node_codes = np.zeros(n_mesh_verts, dtype=int)

    mesh_kdt = cKDTree(mesh_vertices_xy)
    _, mesh_idx_for_orig = mesh_kdt.query(all_xy)

    for orig_idx, mesh_idx in enumerate(mesh_idx_for_orig):
        final_node_codes[mesh_idx] = final_input_df['code'].iloc[orig_idx]

    is_original = np.zeros(n_mesh_verts, dtype=bool)
    is_original[mesh_idx_for_orig] = True

    # Vectorized Steiner vertex code assignment
    new_vert_indices = np.where(~is_original)[0]
    if new_vert_indices.size > 0:
        new_vert_gs = gpd.GeoSeries.from_xy(
            mesh_vertices_xy[new_vert_indices, 0],
            mesh_vertices_xy[new_vert_indices, 1]
        )
        # Build unified geometry of every stream PSLG segment + outlet reach
        stream_pslg_geom = unary_union([
            LineString([all_xy[i], all_xy[j]])
            for i, j in stream_segs + [outlet_reach_seg]
        ])
        dist_to_stream = new_vert_gs.distance(stream_pslg_geom)
        # Use the actual PSLG chord segments (not the smooth exterior ring) so
        # that Steiner points placed on the straight chords are within tolerance.
        bnd_pslg_geom = unary_union([
            LineString([all_xy[i], all_xy[(i + 1) % n_bnd]])
            for i in range(n_bnd)
        ])
        dist_to_bnd = new_vert_gs.distance(bnd_pslg_geom)

        STEINER_TOL = 0.1
        on_stream = dist_to_stream.values < STEINER_TOL
        on_bnd = (~on_stream) & (dist_to_bnd.values < STEINER_TOL)
        final_node_codes[new_vert_indices[on_stream]] = 3
        final_node_codes[new_vert_indices[on_bnd]] = 1

    print(f"  Final codes — 0(Interior):{(final_node_codes==0).sum()}  "
          f"1(Boundary):{(final_node_codes==1).sum()}  "
          f"2(Outlet):{(final_node_codes==2).sum()}  "
          f"3(Stream):{(final_node_codes==3).sum()}")

    return final_vertices_3d, np.array(mesh['triangles']), final_node_codes

# ==============================================================================
# tRIBS Mesh File Writing Functions
# ==============================================================================

def write_diagnostic_shapefile(output_base, vertices_3d, triangles, node_codes, crs=None):
    """
    Writes three diagnostic shapefiles:
      {output_base}_triangles.shp
      {output_base}_nodes.shp
      {output_base}_edges.shp
    Code values: 0=Interior, 1=Boundary, 2=Outlet, 3=Stream
    """
    from shapely.geometry import Polygon as SPoly
    print(f"\n--- Writing diagnostic shapefiles to {output_base}_*.shp ---")

    polys, tri_codes = [], []
    for tri_indices in triangles:
        polys.append(SPoly(vertices_3d[tri_indices][:, :2]))
        codes_in_tri = node_codes[tri_indices]
        if 2 in codes_in_tri:
            tri_codes.append(2)
        elif 3 in codes_in_tri:
            tri_codes.append(3)
        elif 1 in codes_in_tri:
            tri_codes.append(1)
        else:
            tri_codes.append(0)
    gpd.GeoDataFrame({'code': tri_codes}, geometry=polys, crs=crs).to_file(
        f"{output_base}_triangles.shp", driver='ESRI Shapefile'
    )

    pts = [Point(v[0], v[1]) for v in vertices_3d]
    gpd.GeoDataFrame(
        {'code': node_codes, 'elev': vertices_3d[:, 2]},
        geometry=pts, crs=crs
    ).to_file(f"{output_base}_nodes.shp", driver='ESRI Shapefile')

    _priority = {0: 1, 1: 2, 3: 0, 2: 3}
    seen = set()
    edge_lines, edge_codes = [], []
    for tri in triangles:
        for k in range(3):
            i, j = tri[k], tri[(k + 1) % 3]
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            edge_lines.append(LineString([vertices_3d[i, :2], vertices_3d[j, :2]]))
            ci, cj = int(node_codes[i]), int(node_codes[j])
            edge_codes.append(ci if _priority.get(ci, 0) >= _priority.get(cj, 0) else cj)
    gpd.GeoDataFrame({'code': edge_codes}, geometry=edge_lines, crs=crs).to_file(
        f"{output_base}_edges.shp", driver='ESRI Shapefile'
    )
    print(f"  Wrote {len(polys)} triangles, {len(pts)} nodes, {len(edge_lines)} edges.")

def write_tRIBS_mesh_files(output_prefix, output_path, vertices, triangles, node_codes):
    print(f"\n--- Preparing to write tRIBS mesh files ---")
    nnodes, ntri = len(vertices), len(triangles)

    # Ensure CCW winding
    for i in range(ntri):
        p0, p1, p2 = triangles[i]
        area = 0.5 * (
            vertices[p0, 0] * (vertices[p1, 1] - vertices[p2, 1]) +
            vertices[p1, 0] * (vertices[p2, 1] - vertices[p0, 1]) +
            vertices[p2, 0] * (vertices[p0, 1] - vertices[p1, 1])
        )
        if area < 0:
            triangles[i] = [p0, p2, p1]

    undirected_edges = {
        tuple(sorted((tri[i], tri[(i + 1) % 3])))
        for tri in triangles for i in range(3)
    }
    edge_list, directed_edge_to_id = [], {}
    for p1, p2 in sorted(undirected_edges):
        directed_edge_to_id[(p1, p2)] = len(edge_list); edge_list.append([p1, p2])
        directed_edge_to_id[(p2, p1)] = len(edge_list); edge_list.append([p2, p1])
    nedges = len(edge_list)

    spokes = collections.defaultdict(list)
    for i, edge in enumerate(edge_list):
        spokes[edge[0]].append(i)

    node_edgid = -np.ones(nnodes, dtype=int)
    edge_nextid = -np.ones(nedges, dtype=int)

    for node_id, edge_ids in spokes.items():
        angles = [
            (np.arctan2(
                vertices[edge_list[eid][1], 1] - vertices[node_id, 1],
                vertices[edge_list[eid][1], 0] - vertices[node_id, 0]
            ), eid)
            for eid in edge_ids
        ]
        angles.sort()
        sorted_eids = [eid for _, eid in angles]
        node_edgid[node_id] = sorted_eids[0]
        for i in range(len(sorted_eids)):
            edge_nextid[sorted_eids[i]] = sorted_eids[(i + 1) % len(sorted_eids)]

    undirected_edge_to_tris = collections.defaultdict(list)
    for i, tri in enumerate(triangles):
        for j in range(3):
            key = tuple(sorted((tri[j], tri[(j + 1) % 3])))
            undirected_edge_to_tris[key].append(i)

    tri_neighbors = -np.ones((ntri, 3), dtype=int)
    for i, tri in enumerate(triangles):
        for j in range(3):
            key = tuple(sorted((tri[j], tri[(j + 1) % 3])))
            nbrs = undirected_edge_to_tris[key]
            if len(nbrs) == 2:
                tri_neighbors[i, j] = nbrs[1] if nbrs[0] == i else nbrs[0]

    with open(f"{output_path}{output_prefix}.z", "w") as f:
        f.write("0.000000\n")
        f.write(f"{nnodes}\n")
        np.savetxt(f, vertices[:, 2], fmt='%.6f')

    with open(f"{output_path}{output_prefix}.nodes", "w") as f:
        f.write("0.000000\n")
        f.write(f"{nnodes}\n")
        for i in range(nnodes):
            if node_edgid[i] == -1:
                raise RuntimeError(f"Node {i} is isolated (no edges).")
            f.write(f"{vertices[i, 0]:.6f} {vertices[i, 1]:.6f} {node_edgid[i]} {node_codes[i]}\n")

    with open(f"{output_path}{output_prefix}.edges", "w") as f:
        f.write("0.000000\n")
        f.write(f"{nedges}\n")
        for i in range(nedges):
            f.write(f"{edge_list[i][0]} {edge_list[i][1]} {edge_nextid[i]}\n")

    with open(f"{output_path}{output_prefix}.tri", "w") as f:
        f.write("0.000000\n")
        f.write(f"{ntri}\n")
        for i in range(ntri):
            p0, p1, p2 = triangles[i]
            n0 = tri_neighbors[i, 1]   # opposite p0 → shares edge p1-p2
            n1 = tri_neighbors[i, 2]   # opposite p1 → shares edge p2-p0
            n2 = tri_neighbors[i, 0]   # opposite p2 → shares edge p0-p1
            e0 = directed_edge_to_id[(p0, p2)]  # origin=p0, dest=p2
            e1 = directed_edge_to_id[(p1, p0)]  # origin=p1, dest=p0
            e2 = directed_edge_to_id[(p2, p1)]  # origin=p2, dest=p1
            f.write(f"{p0} {p1} {p2} {n0} {n1} {n2} {e0} {e1} {e2}\n")

    print("\n--- All tRIBS mesh files have been generated successfully. ---")


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    name = "SMF"
    output_path = 'outputs/'

    points_file       = '/Users/cjceders/repos/tRIBS-Workshop-Sandbox/workspaces/SMF_pytRIBS/smf_init_data/mesh/SMF.points'
    stream_shapefile  = '/Users/cjceders/repos/tRIBS-Workshop-Sandbox/workspaces/SMF_pytRIBS/smf_demo/data/preprocessing/SMF_stream.shp'
    watershed_shapefile = '/Users/cjceders/repos/tRIBS-Workshop-Sandbox/workspaces/SMF_pytRIBS/smf_demo/data/preprocessing/SMF_boundary.shp'
    dem_file          = '/Users/cjceders/repos/tRIBS-Workshop-Sandbox/workspaces/SMF_pytRIBS/smf_init_data/USGS_10m_clip.tif'
    tree_file         = None

    # --- KEY PARAMETERS ---
    boundary_buffer_dist = 30.0   # meters: buffer applied to watershed polygon outward
    boundary_spacing     = 50.0   # meters: spacing between nodes on the buffered boundary ring
    stream_spacing       = 40.0    # meters: spacing between stream nodes
    stream_clear_radius  = 10.0    # meters: removes interior points this close to streams
    tree_cull_radius     = 5.0     # meters: removes interior points this close to tree centroids
    quality_opts         = 'q10a15050' # Triangle quality options — do NOT include YY here (hardcoded internally)
        # q: Minimum angle (degrees). Triangle adds interior Steiner points to meet this.
        # a: Maximum triangle area (same units as coordinates). Set too low and it can explode mesh size.
        # YY is hardcoded in generate_mesh_from_points() to prevent Steiner points on PSLG segments
        #    (boundary ring + stream segments). Interior Steiner points are still added by q/a as normal.
        # For general use, q10 is a good starting point.
        # https://www.cs.cmu.edu/~quake/triangle.switch.html

    #  SET WORKING DIRECTORY TO SCRIPT'S LOCATION
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")

    print("--- Generating tRIBS TIN mesh ---")
    try:
        final_vertices, final_triangles, final_node_codes = generate_mesh_from_points(
            points_file=points_file,
            stream_shapefile=stream_shapefile,
            watershed_shapefile=watershed_shapefile,
            dem_file=dem_file,
            boundary_buffer_dist=boundary_buffer_dist,
            boundary_spacing=boundary_spacing,
            tree_centroids_shapefile=tree_file,
            tree_remove_radius=tree_cull_radius,
            stream_clearance_radius=stream_clear_radius,
            stream_point_spacing=stream_spacing,
            mesh_quality_opts=quality_opts
        )

        write_diagnostic_shapefile(
            output_base=f'{output_path}/{name}_tin_{quality_opts}',
            vertices_3d=final_vertices,
            triangles=final_triangles,
            node_codes=final_node_codes,
            crs="EPSG:26912"
        )

        write_tRIBS_mesh_files(f'{name}_mesh', output_path, final_vertices, final_triangles, final_node_codes)

    except Exception as e:
        import traceback
        print(f"\nPROCESS FAILED: {e}")
        traceback.print_exc()
