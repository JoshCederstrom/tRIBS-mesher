import pandas as pd
import numpy as np
import geopandas as gpd
import triangle as tr
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union, snap, linemerge
import collections
import rasterio
import os
from pathlib import Path

def _order_boundary_points(boundary_pts_df):
    """Orders boundary points to form a contiguous polygon edge."""
    if len(boundary_pts_df) < 3: return []
    points = boundary_pts_df[['x', 'y']].values
    original_indices = boundary_pts_df.index.tolist()
    kdt = cKDTree(points)
    num_points = len(points)
    idx_map = {i: original_indices[i] for i in range(num_points)}
    ordered_indices, remaining_kdt_indices = [], set(range(num_points))
    current_kdt_idx = remaining_kdt_indices.pop()
    ordered_indices.append(idx_map[current_kdt_idx])
    while remaining_kdt_indices:
        _, indices = kdt.query(points[current_kdt_idx], k=min(10, len(remaining_kdt_indices) + 1))
        next_kdt_idx = -1
        for idx in indices[1:]:
            if idx in remaining_kdt_indices:
                next_kdt_idx = idx
                break
        if next_kdt_idx == -1:
            next_kdt_idx = remaining_kdt_indices.pop()
        else:
            remaining_kdt_indices.remove(next_kdt_idx)
        ordered_indices.append(idx_map[next_kdt_idx])
        current_kdt_idx = next_kdt_idx
    return ordered_indices

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
    seg = np.sqrt((d*d).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return s, seg

def _remove_spikes(coords, angle_tol_deg):
    if len(coords) <= 2: return coords
    keep = [coords[0]]
    cos_hi = np.cos(np.deg2rad(180 - angle_tol_deg))
    for i in range(1, len(coords)-1):
        a, b, c = coords[i-1], coords[i], coords[i+1]
        u, v = b - a, c - b
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu == 0 or nv == 0: continue
        cosang = float(np.dot(u, v) / (nu*nv))
        # keep if it's not a near-0° or near-180° spike
        if not (cosang > 0.9999 or cosang < -0.9999):
            keep.append(b)
    keep.append(coords[-1])
    return np.asarray(keep)

def _node_streams(lines, tau_snap=1.0):
    """Snap near-coincident vertices and node all intersections once."""
    ml = linemerge(unary_union(snap(unary_union(lines), unary_union(lines), tau_snap)))
    if isinstance(ml, LineString): return [ml]
    return list(ml.geoms)

def _resample_with_min_spacing(line, spacing, m_min, angle_tol_deg=2.0, keep_original_vertices=False):
    """
    If keep_original_vertices=False (default), keep ONLY endpoints + uniform grid,
    enforcing >= m_min between consecutive kept points. This avoids blowing up counts
    when source polylines are densely sampled.
    """
    coords = np.asarray(line.coords, float)
    if coords.ndim == 2 and coords.shape[1] > 2:
        coords = coords[:, :2]  # drop Z/M if present
    coords = _remove_spikes(coords, angle_tol_deg) if angle_tol_deg > 0 else coords
    s, seg = _cumlens(coords); L = float(s[-1])
    if L == 0: 
        return coords[:1]

    # --- decide which arclengths to keep ---
    s_grid = np.arange(0.0, L + 1e-9, spacing)
    if keep_original_vertices:
        s_orig = s  # endpoints + all original vertices
    else:
        s_orig = np.array([0.0, L])  # endpoints only

    all_s = np.unique(np.round(np.concatenate([s_orig, s_grid]), 9))

    kept = []
    seg_idx = 0
    for t in all_s:
        while seg_idx < len(seg)-1 and s[seg_idx+1] < t - 1e-12:
            seg_idx += 1
        # exact vertex?
        k = np.where(np.isclose(s, t, rtol=0, atol=1e-9))[0]
        if k.size:
            pt = coords[k[0]]
        else:
            t0, t1 = s[seg_idx], s[seg_idx+1]
            if t1 <= t0: 
                continue
            a = (t - t0) / (t1 - t0)
            pt = coords[seg_idx] * (1 - a) + coords[seg_idx+1] * a

        if not kept or np.linalg.norm(pt - kept[-1]) >= m_min - 1e-12:
            kept.append(pt)

    # ALWAYS include the last endpoint for connectivity
    if np.linalg.norm(kept[-1] - coords[-1]) > 1e-9:
        kept.append(coords[-1])

    # final dedupe against tiny numeric jitter
    out = [kept[0]]
    for p in kept[1:]:
        if np.linalg.norm(p - out[-1]) >= m_min - 1e-12:
            out.append(p)
    # ensure final endpoint again (harmless if already there)
    if np.linalg.norm(out[-1] - coords[-1]) > 1e-9:
        out.append(coords[-1])
    return np.asarray(out)

def _split_line_at_point(line, point):
    """
    Splits a shapely LineString at a projected point.
    Returns a list of one or two new LineStrings.
    """
    coords = list(line.coords)
    if point.coords[0] in coords:
        return [line] # Point is already a vertex, no split needed

    distance = line.project(point)
    if distance <= 1e-9 or distance >= line.length - 1e-9:
        return [line] # Point is effectively at an endpoint

    pre_coords = []
    post_coords = [point.coords[0]]
    
    current_dist = 0.0
    for i in range(len(coords) - 1):
        p1 = Point(coords[i])
        p2 = Point(coords[i+1])
        segment = LineString([p1, p2])
        
        # Check if the projection falls on this segment
        if current_dist <= distance < current_dist + segment.length:
            pre_coords.extend(coords[:i+1])
            pre_coords.append(point.coords[0])
            post_coords.extend(coords[i+1:])
            
            line1 = LineString(pre_coords)
            line2 = LineString(post_coords)
            return [line1, line2]
            
        current_dist += segment.length
        
    return [line] # Should not be reached if point is on line

def _build_stream_pslg(resampled_edges, m_min, dedupe_radius):
    """
    Global dedupe within ~dedupe_radius and emit (points, segments).
    Always returns (points_xy[N,2], segments[List[Tuple[int,int]]]).
    """
    # Filter out Nones/short arrays and force XY only
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

    # --- decoupled dedupe radius ---
    kdt = cKDTree(P)
    pairs = kdt.query_pairs(r=max(float(dedupe_radius), 1e-9))

    # Union-find for merging near-coincident points
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

    # Build segments per resampled edge
    segs = set()
    cursor = 0
    for arr in clean_edges:
        n = len(arr)
        idxs = new_index[cursor:cursor + n]
        cursor += n
        prev = idxs[0]
        for cur in idxs[1:]:
            if cur != prev:
                segs.add((prev, cur))
            prev = cur

    # Drop only truly tiny segments (keep connectivity)
    tiny = max(0.5, 0.25 * float(dedupe_radius))  # meters
    segments = []
    for i, j in segs:
        if np.linalg.norm(uniq[i] - uniq[j]) >= tiny - 1e-12:
            segments.append((i, j))

    return uniq, segments

# ==============================================================================
# The Main Meshing Function 
# ==============================================================================

def generate_mesh_from_points(
    points_file, stream_shapefile, dem_file,
    tree_centroids_shapefile=None,
    tree_remove_radius=5.0,
    stream_clearance_radius=15.0,
    stream_point_spacing=25.0,
    mesh_quality_opts='p'
):
    # Step 1: Generate new stream vertices from shapefile
    print("\nStep 1: Generating stream vertices from shapefile...")
    streams_gdf = gpd.read_file(stream_shapefile)

    # explode to pure LineStrings
    raw_lines = []
    for g in streams_gdf.geometry.dropna():
        if isinstance(g, LineString):
            raw_lines.append(g)
        elif g.geom_type == "MultiLineString":
            raw_lines.extend(list(g.geoms))

    # 1) snap & node junctions once to avoid hairline gaps
    lines = _node_streams(raw_lines, tau_snap=1.0)

    # 2) resample each noded edge with merge-on-arclength densifier
    MIN_STREAM_SPACING = 5.0  # meters: hard minimum consecutive spacing
    # Choose a dedupe radius independent of min spacing (scale with spacing)
    DEDUPE_RADIUS = max(1.5, 0.05 * float(stream_point_spacing))  # e.g., 1.5 m at 30 m spacing
    resampled_edges = []
    for ln in lines:
        pts = _resample_with_min_spacing(
            ln, spacing=stream_point_spacing, m_min=MIN_STREAM_SPACING, angle_tol_deg=2.0,
            keep_original_vertices=False
        )
        if len(pts) >= 2:
            resampled_edges.append(pts)

    # --- Topologically integrate the OUTLET point into the stream network ---
    try:
        ba = pd.read_csv(points_file, sep=r"\s+", skiprows=1, header=None,
                         names=['x','y','z','code'], comment='#')
        outlet_rows = ba[ba['code'] == 2]
        if not outlet_rows.empty and lines:
            outlet_xy = outlet_rows.iloc[0][['x','y']].values.astype(float)
            outlet_pt = Point(float(outlet_xy[0]), float(outlet_xy[1]))

            # Find the single stream linestring closest to the outlet
            min_dist = float('inf')
            closest_line_idx = -1
            for i, line in enumerate(lines):
                dist = line.distance(outlet_pt)
                if dist < min_dist:
                    min_dist = dist
                    closest_line_idx = i
            
            if closest_line_idx != -1:
                # The line to be modified
                line_to_split = lines[closest_line_idx]
                
                # Find the point 'q' on that line to connect to
                q_point = line_to_split.interpolate(line_to_split.project(outlet_pt))
                
                # A) Create the new connector edge from 'q' to the outlet
                connector_line = LineString([q_point, outlet_pt])
                
                # B) Split the original line at 'q'
                split_parts = _split_line_at_point(line_to_split, q_point)
                
                # C) Update the main 'lines' list by replacing the old line
                #    with its split parts and the new connector.
                #    This ensures the topology is correct before resampling.
                lines.pop(closest_line_idx) # Remove the original
                lines.extend(split_parts)   # Add the split parts
                lines.append(connector_line)      # Add the new connector
                
                print("  SUCCESS: Topologically integrated outlet into the stream network.")
        else:
            print("  SKIPPED: Outlet integration (no outlet point found or no streams).")

    except Exception as _e:
        print(f"  [ERROR] Outlet integration failed: {_e}")

    # 3) build a PSLG piece (points + within-edge segments), with global de-dup
    # Build PSLG from resampled edges
    stream_nodes_xy, stream_segments_local = _build_stream_pslg(
        resampled_edges, m_min=MIN_STREAM_SPACING, dedupe_radius=DEDUPE_RADIUS
    )
    stream_nodes_df = pd.DataFrame(stream_nodes_xy, columns=['x','y'])
    stream_nodes_df['code'] = 3
    print(f"Generated {len(stream_nodes_df)} stream nodes with spacing ~{stream_point_spacing}m "
          f"(min consecutive {MIN_STREAM_SPACING}m).")

    # unified geometry for clearance buffer later
    unified_stream_geom = linemerge(unary_union(lines))

    total_len_km = sum(ln.length for ln in lines) / 1000.0
    expected = total_len_km * 1000.0 / stream_point_spacing
    print(f"  ~Total stream length: {total_len_km:.2f} km; expected ~{expected:.0f} nodes @ {stream_point_spacing} m")
    print(f"  Actual stream nodes: {len(stream_nodes_df)} (m_min={MIN_STREAM_SPACING} m, dedupe={DEDUPE_RADIUS} m)")

    # Step 2: Read and prepare all non-stream points (Boundary, Outlet, Interior, Trees)
    print("\nStep 2: Preparing all non-stream points...")
    base_df_all = pd.read_csv(points_file, sep=r"\s+", skiprows=1, header=None, names=['x','y','z','code'], comment='#')
    # Keep only boundary and original interior points for now
    non_stream_pts = base_df_all[base_df_all['code'] != 3][['x', 'y', 'code']].copy().reset_index(drop=True)
    # Step 2.1: Remove interior points that are too close to one another
    INTERIOR_REMOVE_RADIUS = 5.0  # meters
    # Mask out only the interior points
    interior_mask = non_stream_pts['code'] == 0
    interior_pts = non_stream_pts.loc[interior_mask, ['x','y']].values

    if len(interior_pts) > 1:
        # Build a KD‐tree on those interior points
        interior_tree = cKDTree(interior_pts)
        # Find all unique pairs closer than the threshold
        close_pairs = interior_tree.query_pairs(r=INTERIOR_REMOVE_RADIUS)

        # Map those pair‐indices back to your DataFrame index
        drop_indices = set()
        interior_indices = non_stream_pts.loc[interior_mask].index.tolist()
        for i, j in close_pairs:
            # arbitrarily drop the “j”th point of each pair
            drop_indices.add(interior_indices[j])

        if drop_indices:
            print(f"Removing {len(drop_indices)} interior points within "
                f"{INTERIOR_REMOVE_RADIUS} m of another interior point.")
            non_stream_pts = (non_stream_pts
                            .drop(index=list(drop_indices))
                            .reset_index(drop=True))
        
    tree_nodes_df = pd.DataFrame()
    if tree_centroids_shapefile:
        print("\n--- Processing Tree Centroids ---")
        trees_gdf = gpd.read_file(tree_centroids_shapefile)
        
        # A) Remove original interior points near where trees will be placed
        tree_points_xy = np.array([[p.x, p.y] for p in trees_gdf.geometry])
        if tree_points_xy.size > 0:
            tree_kdt = cKDTree(tree_points_xy)
            interior_points_df = non_stream_pts[non_stream_pts['code'] == 0]
            if not interior_points_df.empty:
                indices_to_drop = tree_kdt.query_ball_point(interior_points_df[['x', 'y']].values, r=tree_remove_radius)
                drop_set = {interior_points_df.index[i] for i, n in enumerate(indices_to_drop) if n}
                if drop_set:
                    print(f"Removing {len(drop_set)} original interior points within {tree_remove_radius}m of a tree centroid.")
                    non_stream_pts = non_stream_pts.drop(list(drop_set))

            # B) Prepare the tree nodes dataframe to be added
            tree_nodes_df = pd.DataFrame(tree_points_xy, columns=['x', 'y'])
            tree_nodes_df['code'] = 0 # Trees are treated as interior points

    # Consolidate all non-stream points (Boundary, remaining Interior, and new Trees)
    final_non_stream_pts = pd.concat([non_stream_pts, tree_nodes_df], ignore_index=True)

    # Step 3: Enforce stream clearance zone on ALL interior-type points
    if stream_clearance_radius > 0:
        print(f"\nStep 3: Creating a {stream_clearance_radius}m clearance zone around streams...")
        
        # Select all interior-type points (original interior AND trees)
        interior_mask = final_non_stream_pts['code'] == 0
        all_interior_points_df = final_non_stream_pts[interior_mask]
        
        if not all_interior_points_df.empty:
            # Create a GeoSeries of the points to check
            interior_geoseries = gpd.GeoSeries.from_xy(
                all_interior_points_df['x'], 
                all_interior_points_df['y']
            )
            
            # Directly calculate the distance from each point to the unified stream geometry.
            # This is much faster than creating a buffer.
            distances_to_stream = interior_geoseries.distance(unified_stream_geom)
            
            # Create a boolean mask of points that are TOO CLOSE to the stream
            points_to_remove_mask = distances_to_stream < stream_clearance_radius
            
            # Get the original DataFrame indices of the points to drop
            indices_to_drop = all_interior_points_df.index[points_to_remove_mask]
            
            if not indices_to_drop.empty:
                print(f"Removing {len(indices_to_drop)} interior-type points within the stream clearance zone.")
                final_non_stream_pts = final_non_stream_pts.drop(indices_to_drop)

    # Step 4: Combine all points and get elevations from DEM
    print("\nStep 4: Combining points and sampling elevations from DEM...")
    final_input_df = pd.concat([final_non_stream_pts, stream_nodes_df], ignore_index=True).reset_index(drop=True)
    all_input_xy = final_input_df[['x', 'y']].values
    final_input_df['z'] = sample_elevations_from_dem(all_input_xy, dem_file)

    # Step 5: Define mesh constraints
    print("\nStep 5: Defining mesh constraints...")
    n_non_stream = len(final_non_stream_pts)
    
    # Boundary segments
    boundary_df = final_input_df[final_input_df['code'].isin([1, 2])].copy()
    ordered_boundary_indices = _order_boundary_points(boundary_df)
    boundary_segs_indices = [(ordered_boundary_indices[i], ordered_boundary_indices[(i + 1) % len(ordered_boundary_indices)]) for i in range(len(ordered_boundary_indices))]
    
    # Stream segments (use the ones built from resampled edges; shift by n_non_stream)
    stream_segs_indices = [(i + n_non_stream, j + n_non_stream) for (i, j) in stream_segments_local]

    all_segs_indices = set(tuple(sorted(s)) for s in (boundary_segs_indices + stream_segs_indices))
    segments = [list(s) for s in all_segs_indices]
    
    # Step 6: Triangulate
    print("\nStep 6: Generating Conforming Delaunay Mesh...")
    V = final_input_df[['x','y']].values
    Vn, origin_xy, scale_xy = _normalize_coords(V)  # reduce dynamic range
    mesh_data = {'vertices': Vn, 'segments': segments}

    # keep your quality opts; add 'j' to gently joggle if needed
    tri_opts = f"p{mesh_quality_opts}Dj"
    mesh = tr.triangulate(mesh_data, tri_opts)

    # denormalize mesh vertices back to map coords for downstream steps
    mesh_vertices_xy = _denormalize_coords(np.asarray(mesh['vertices']), origin_xy, scale_xy)
    
    # Step 7: Assign elevations to ALL mesh vertices from DEM
    print("\nStep 7: Assigning final elevations to all mesh vertices from DEM...")
    #mesh_vertices_xy = np.array(mesh['vertices'])
    mesh_vertices_z = sample_elevations_from_dem(mesh_vertices_xy, dem_file)
    final_vertices_3d = np.column_stack([mesh_vertices_xy, mesh_vertices_z])

    # Step 8: Assign final node codes
    print("\nStep 8: Assigning final node codes...")
    n_mesh_verts = len(mesh_vertices_xy)
    final_node_codes = np.zeros(n_mesh_verts, dtype=int)
    
    mesh_kdt = cKDTree(mesh_vertices_xy)
    _, mesh_indices_for_originals = mesh_kdt.query(final_input_df[['x', 'y']].values)
    
    for original_idx, mesh_idx in enumerate(mesh_indices_for_originals):
        final_node_codes[mesh_idx] = final_input_df['code'].iloc[original_idx]

    is_original_node = np.zeros(n_mesh_verts, dtype=bool)
    is_original_node[mesh_indices_for_originals] = True

    final_boundary_segs = [LineString(final_input_df.loc[s, ['x', 'y']].values) for s in boundary_segs_indices]
    final_stream_segs = [LineString(final_input_df.loc[s, ['x', 'y']].values) for s in stream_segs_indices]

    for i in range(n_mesh_verts):
        if is_original_node[i]: continue
        p = Point(mesh_vertices_xy[i])
        if any(p.distance(seg) < 0.1 for seg in final_stream_segs):
            final_node_codes[i] = 3
        elif any(p.distance(seg) < 0.1 for seg in final_boundary_segs):
            final_node_codes[i] = 1

    print(f"Final code counts: 0 (Interior): {np.sum(final_node_codes==0)}, 1 (Boundary): {np.sum(final_node_codes==1)}, 2 (Outlet): {np.sum(final_node_codes==2)}, 3 (Stream): {np.sum(final_node_codes==3)}")
    
    return final_vertices_3d, np.array(mesh['triangles']), final_node_codes


# ==============================================================================
# tRIBS Mesh File Writing Functions
# ==============================================================================
def write_diagnostic_shapefile(output_base, vertices_3d, triangles, node_codes, crs=None):
    """
    Writes three diagnostic shapefiles sharing a common base path:
      {output_base}_triangles.shp  — mesh triangles, coded by dominant node type
      {output_base}_nodes.shp      — mesh nodes with their boundary codes
      {output_base}_edges.shp      — mesh edges coded by highest-priority endpoint code

    Code values: 0=Interior, 1=Boundary, 2=Outlet, 3=Stream
    Edge code priority: 2 > 3 > 1 > 0
    """
    print(f"\n--- Writing diagnostic shapefiles to {output_base}_*.shp ---")

    # --- Triangles ---
    polys, tri_codes = [], []
    for tri_indices in triangles:
        polys.append(Polygon(vertices_3d[tri_indices][:, :2]))
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

    # --- Nodes ---
    pts = [Point(v[0], v[1]) for v in vertices_3d]
    gpd.GeoDataFrame({'code': node_codes}, geometry=pts, crs=crs).to_file(
        f"{output_base}_nodes.shp", driver='ESRI Shapefile'
    )

    # --- Edges ---
    _priority = {0: 1, 1: 2, 3: 0, 2: 3}
    seen = set()
    lines, edge_codes = [], []
    for tri in triangles:
        for k in range(3):
            i, j = tri[k], tri[(k + 1) % 3]
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            lines.append(LineString([vertices_3d[i, :2], vertices_3d[j, :2]]))
            ci, cj = int(node_codes[i]), int(node_codes[j])
            edge_codes.append(ci if _priority.get(ci, 0) >= _priority.get(cj, 0) else cj)
    gpd.GeoDataFrame({'code': edge_codes}, geometry=lines, crs=crs).to_file(
        f"{output_base}_edges.shp", driver='ESRI Shapefile'
    )

    print(f"  Wrote {len(polys)} triangles, {len(pts)} nodes, {len(lines)} edges.")

def write_tRIBS_mesh_files(output_prefix, output_path, vertices, triangles, node_codes):
    print(f"\n--- Preparing to write tRIBS mesh files ---")
    nnodes, ntri = len(vertices), len(triangles)
    
    # Ensure triangle winding is counter-clockwise
    for i in range(ntri):
        p0, p1, p2 = triangles[i]
        area = 0.5 * (vertices[p0, 0] * (vertices[p1, 1] - vertices[p2, 1]) + \
                      vertices[p1, 0] * (vertices[p2, 1] - vertices[p0, 1]) + \
                      vertices[p2, 0] * (vertices[p0, 1] - vertices[p1, 1]))
        if area < 0:
            triangles[i] = [p0, p2, p1]
            
    undirected_edges = {tuple(sorted((tri[i], tri[(i + 1) % 3]))) for tri in triangles for i in range(3)}
    edge_list, directed_edge_to_id = [], {}
    for p1, p2 in sorted(list(undirected_edges)):
        directed_edge_to_id[(p1, p2)] = len(edge_list); edge_list.append([p1, p2])
        directed_edge_to_id[(p2, p1)] = len(edge_list); edge_list.append([p2, p1])
    nedges = len(edge_list)
    
    spokes = collections.defaultdict(list)
    for i, edge in enumerate(edge_list): spokes[edge[0]].append(i)
    node_edgid, edge_nextid = -np.ones(nnodes, dtype=int), -np.ones(nedges, dtype=int)
    
    for node_id, edge_ids in spokes.items():
        if not edge_ids: continue
        angles = [(np.arctan2(vertices[edge_list[eid][1], 1] - vertices[node_id, 1], vertices[edge_list[eid][1], 0] - vertices[node_id, 0]), eid) for eid in edge_ids]
        angles.sort()
        sorted_edge_ids = [eid for _, eid in angles]
        if not sorted_edge_ids: continue
        node_edgid[node_id] = sorted_edge_ids[0]
        for i in range(len(sorted_edge_ids)):
            edge_nextid[sorted_edge_ids[i]] = sorted_edge_ids[(i + 1) % len(sorted_edge_ids)]
            
    undirected_edge_to_tris = collections.defaultdict(list)
    for i, tri in enumerate(triangles):
        for j in range(3):
            p1, p2 = tri[j], tri[(j+1)%3]
            undirected_edge_to_tris[tuple(sorted((p1, p2)))].append(i)
    tri_neighbors = -np.ones((ntri, 3), dtype=int)
    
    for i, tri in enumerate(triangles):
        for j in range(3):
            p1, p2 = tri[j], tri[(j+1)%3]
            neighbor_tris = undirected_edge_to_tris[tuple(sorted((p1, p2)))]
            if len(neighbor_tris) == 2:
                tri_neighbors[i, j] = neighbor_tris[1] if neighbor_tris[0] == i else neighbor_tris[0]
                
    with open(f"{output_path}{output_prefix}.z", "w") as f:
        f.write("0.000000\n"); f.write(f"{nnodes}\n")
        np.savetxt(f, vertices[:, 2], fmt='%.6f')
    with open(f"{output_path}{output_prefix}.nodes", "w") as f:
        f.write("0.000000\n"); f.write(f"{nnodes}\n")
        for i in range(nnodes):
            if node_edgid[i] == -1: raise RuntimeError(f"Node {i} is isolated.")
            f.write(f"{vertices[i, 0]:.6f} {vertices[i, 1]:.6f} {node_edgid[i]} {node_codes[i]}\n")
    with open(f"{output_path}{output_prefix}.edges", "w") as f:
        f.write("0.000000\n"); f.write(f"{nedges}\n")
        for i in range(nedges): f.write(f"{edge_list[i][0]} {edge_list[i][1]} {edge_nextid[i]}\n")
    with open(f"{output_path}{output_prefix}.tri", "w") as f:
        f.write("0.000000\n"); f.write(f"{ntri}\n")
        for i in range(ntri):
            p0, p1, p2 = triangles[i]
            n0, n1, n2 = tri_neighbors[i, 1], tri_neighbors[i, 2], tri_neighbors[i, 0]
            e0, e1, e2 = directed_edge_to_id[(p0, p2)], directed_edge_to_id[(p1, p0)], directed_edge_to_id[(p2, p1)]
            f.write(f"{p0} {p1} {p2} {n0} {n1} {n2} {e0} {e1} {e2}\n")
    print("\n--- All tRIBS mesh files have been generated successfully. ---")


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    # --- FILE PATHS ---
    name = "SMF"
    output_path = f'outputs/'
    points_file = f'/Users/cjceders/repos/tRIBS-Workshop-Sandbox/workspaces/SMF_pytRIBS/smf_init_data/mesh/SMF.points'          # Interior terrain points from pytRIBS wavelet transform
    stream_shapefile = f'/Users/cjceders/repos/tRIBS-Workshop-Sandbox/workspaces/SMF_pytRIBS/smf_demo/data/preprocessing/SMF_stream.shp'
    dem_file = "/Users/cjceders/repos/tRIBS-Workshop-Sandbox/workspaces/SMF_pytRIBS/smf_init_data/USGS_10m_clip.tif"
    tree_file = None  # Set to None to disable

    print("--- Generating mesh with DEM, Uniform Streams, and Tree Centroids (v10) ---")
    
    # --- KEY PARAMETERS TO TUNE ---
    stream_spacing = 20.0            # meters. Distance between generated stream points.
    stream_clear_radius = 10.0       # meters. Removes interior/tree points this close to streams.
                                     # In cases with an extremely dense set of tree points increasing this can help.
    tree_cull_radius = 5.0           # meters. Removes original interior points this close to trees.
    quality_opts = 'q10a15050'      # Triangle quality options example q10a20000
        # q: Min angle in degrees, will add in extra vertices to remove triangles not meeting criteria.
        # a: Max area in units from points, will add extra triangles to make sure no triangle exceeds this area.
        #    Be careful with this, if you set this too low, it can cause the mesh to fail.
        # YY: Under no circumstances insert any new vertices on your input segments (boundary or stream segments).
        #    Could be used if you are having issue with stream connectivity when reading into tRIBS but
        #    generate_mesh_from_points() should set generated points that fall within 0.1m of the steam into stream nodes.
        #    Can help if with an extremely dense set of tree points.
        # For general use, it seems like q10 is a good place to start. 
        # https://www.cs.cmu.edu/~quake/triangle.switch.html

    #  SET WORKING DIRECTORY TO SCRIPT'S LOCATION
    # Get the absolute path to the directory where this script is located
    script_dir = Path(__file__).resolve().parent
    # Change the current working directory to the script's directory
    os.chdir(script_dir)
    # Optional: Print the new working directory to confirm
    print(f"Working directory set to: {os.getcwd()}")
    
    try:
        final_vertices, final_triangles, final_node_codes = generate_mesh_from_points(
            points_file=points_file,
            stream_shapefile=stream_shapefile,
            dem_file=dem_file,
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