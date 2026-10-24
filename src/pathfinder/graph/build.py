"""Build the citywide pedestrian graph from extracted OSM ways.

Pipeline
--------
1. Classify every highway way (``classify.classify``).
2. For streets with no usable sidewalk tags, detect separately mapped sidewalks
   geometrically (perpendicular probes); streets whose pedestrians are served by separate
   sidewalk ways are removed so the router never walks down the middle of a street that has
   real sidewalks.
3. Split ways into edges at shared nodes and way endpoints. Graph vertices are OSM node ids,
   so grade separation (bridges, tunnels) is respected exactly as mapped: two ways cross only
   if they share a node.
4. Bridge small mapping gaps: a dead-end within ``GAP_TOLERANCE_M`` of another edge gets an
   explicit ``connector`` edge with ``representation = "inferred_connector"`` so the UI can
   disclose it. Gaps are never closed silently. Wider gaps (up to ``SURFACE_GAP_MAX_M``) are
   bridged only where the region's planimetric survey shows walkable surface all the way
   across (a region without one gets no such bridges); being near is otherwise never
   treated as being connected.
5. Label connected components.

Outputs (GRAPH_DIR): ``edges.parquet`` (EPSG:26985), ``nodes.parquet``, ``build_stats.json``.
"""

from __future__ import annotations

import json
import logging
from collections import Counter

import geopandas as gpd
import igraph as ig
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import substring

from ..config import ANALYSIS_CRS, GRAPH_DIR, INTERIM_DIR, ensure_dirs
from ..fetch import raw_path
from ..sources import REGION
from .classify import PEDESTRIAN_WAY, STREET_CENTERLINE, classify

log = logging.getLogger(__name__)

# --- separate-sidewalk detection parameters
PROBE_SPACING_M = 10.0  # sample spacing along a street
PROBE_END_SKIP_M = 12.0  # ignore samples near intersections (cross-street sidewalks)
PROBE_NEAR_M = 2.0  # probes start this far from the centerline
PROBE_FAR_M = 30.0  # ...and reach this far (wide DC avenues)
SIDE_COVERAGE_THRESHOLD = 0.6  # fraction of samples that must find a sidewalk on a side

# --- gap bridging
GAP_TOLERANCE_M = 3.0
# A dead end farther than GAP_TOLERANCE_M from the next walkway is joined to it only when the
# straight line between them lies on planimetric sidewalk surface (see surface_gap_test).
# 15 m covers a plaza or a wide corner; beyond that a straight line is no longer a fair
# stand-in for the path someone would walk.
SURFACE_GAP_MAX_M = 15.0
SURFACE_GAP_MIN_COVER = 0.9  # share of the connector that must lie on mapped walkable surface
_NOT_A_ROAD = {"footway", "path", "steps", "pedestrian", "cycleway", "corridor", "elevator", "track", "bridleway"}

INFERRED_CONNECTOR = "inferred_connector"
SURFACE_GAP_REASON = "pedestrian dead end joined across planimetric sidewalk surface"
# Planimetric classes that are walkable surface for that test. A walk away from the street
# (Montgomery's "Non-Street Sidewalk") is surface to walk on, though not a street sidewalk.
WALKABLE_SURFACE = ("Sidewalk", "Hidden Sidewalk", "Non-Street Sidewalk")


# ============================================================ separate sidewalk detection


def _side_probes(streets: gpd.GeoSeries, far_left, far_right):
    """Perpendicular probes on both sides of each street every PROBE_SPACING_M, away from its
    ends, from PROBE_NEAR_M out to the side's reach. Returns (probes, street index, side)."""
    reach = {"L": np.broadcast_to(np.asarray(far_left, dtype=float), (len(streets),)),
             "R": np.broadcast_to(np.asarray(far_right, dtype=float), (len(streets),))}
    probe_geoms, probe_street, probe_side = [], [], []
    for i, geom in enumerate(streets.values):
        L = geom.length
        if L <= 2 * PROBE_END_SKIP_M:
            continue
        dists = np.arange(PROBE_END_SKIP_M, L - PROBE_END_SKIP_M + 1e-9, PROBE_SPACING_M)
        for d in dists:
            p = geom.interpolate(d)
            a = geom.interpolate(max(d - 1.0, 0))
            b = geom.interpolate(min(d + 1.0, L))
            tx, ty = b.x - a.x, b.y - a.y
            norm = np.hypot(tx, ty)
            if norm == 0:
                continue
            nx, ny = -ty / norm, tx / norm  # left normal
            for side, sgn in (("L", 1.0), ("R", -1.0)):
                far = max(reach[side][i], PROBE_NEAR_M + 0.5)
                probe_geoms.append(
                    LineString(
                        [
                            (p.x + sgn * nx * PROBE_NEAR_M, p.y + sgn * ny * PROBE_NEAR_M),
                            (p.x + sgn * nx * far, p.y + sgn * ny * far),
                        ]
                    )
                )
                probe_street.append(i)
                probe_side.append(side)
    return probe_geoms, np.asarray(probe_street, dtype=int), np.asarray(probe_side)


def sidewalk_side_coverage(
    streets: gpd.GeoSeries, sidewalks: gpd.GeoSeries,
    far_left: np.ndarray | float = PROBE_FAR_M, far_right: np.ndarray | float = PROBE_FAR_M,
    offsets: bool = False,
) -> tuple[np.ndarray, ...]:
    """Fraction of sample points along each street with a sidewalk line found on the left
    and right side within [PROBE_NEAR_M, far] (default PROBE_FAR_M; per street and side when
    a reach must stop short, e.g. at the median of a divided road). NaN when a street is too
    short to sample away from its ends.

    With ``offsets``, also the median distance from the street to the nearest line found on
    each side (NaN where none is), as two more arrays: where a sidewalk found on that side
    lies."""
    probe_geoms, probe_street, probe_side = _side_probes(streets, far_left, far_right)
    tree = shapely.STRtree(np.asarray(sidewalks.values))
    hit = np.zeros(len(probe_geoms), dtype=bool)
    first = np.full(len(probe_geoms), np.nan)
    if probe_geoms:
        probes = np.asarray(probe_geoms, dtype=object)
        pi, gi = tree.query(probes, predicate="intersects")
        hit[np.unique(pi)] = True
        if offsets and len(pi):
            geoms = np.asarray(sidewalks.values)
            starts = shapely.get_point(probes[pi], 0)
            d = shapely.distance(starts, shapely.intersection(probes[pi], geoms[gi])) + PROBE_NEAR_M
            first = pd.Series(d).groupby(pi).min().reindex(range(len(probes))).to_numpy()
    cov_l = np.full(len(streets), np.nan)
    cov_r = np.full(len(streets), np.nan)
    off_l = np.full(len(streets), np.nan)
    off_r = np.full(len(streets), np.nan)
    for side, out, off in (("L", cov_l, off_l), ("R", cov_r, off_r)):
        m = probe_side == side
        hits = np.bincount(probe_street[m], weights=hit[m].astype(float), minlength=len(streets))
        tot = np.bincount(probe_street[m], minlength=len(streets))
        valid = tot > 0
        out[valid] = hits[valid] / tot[valid]
        if offsets:
            med = pd.Series(first[m]).groupby(probe_street[m]).median()
            off[med.index.to_numpy()] = med.to_numpy()
    if offsets:
        return cov_l, cov_r, off_l, off_r
    return cov_l, cov_r


# ============================================================ classification


def classify_ways(ways: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    tag_cols = [c for c in ways.columns if c not in ("osm_way_id", "node_refs", "geometry", "all_tags")]
    # Parquet round-trips missing tags as None or NaN; only real strings are tags.
    recs = [classify({k: v for k, v in zip(tag_cols, row) if isinstance(v, str)})
            for row in ways[tag_cols].itertuples(index=False, name=None)]
    ways = ways.copy()
    ways["kind"] = [c.kind for c in recs]
    ways["representation"] = [c.representation for c in recs]
    ways["class_reason"] = [c.reason for c in recs]
    ways["sidewalk_left"] = [c.sidewalk_left for c in recs]
    ways["sidewalk_right"] = [c.sidewalk_right for c in recs]
    ways["sidewalk_cov_left"] = np.nan
    ways["sidewalk_cov_right"] = np.nan

    cand = ways["kind"] == "street_unknown"
    sw = ways.loc[ways["kind"] == "sidewalk", "geometry"]
    cov_l, cov_r = sidewalk_side_coverage(ways.loc[cand, "geometry"], sw)
    ways.loc[cand, "sidewalk_cov_left"] = cov_l
    ways.loc[cand, "sidewalk_cov_right"] = cov_r

    left_ok = (ways["sidewalk_left"] == "separate") | (ways["sidewalk_cov_left"] >= SIDE_COVERAGE_THRESHOLD)
    right_ok = (ways["sidewalk_right"] == "separate") | (ways["sidewalk_cov_right"] >= SIDE_COVERAGE_THRESHOLD)
    drop = cand & left_ok & right_ok
    ways.loc[drop, ["kind", "representation"]] = None
    ways.loc[drop, "class_reason"] = "separate sidewalks detected geometrically"
    log.info("separate-sidewalk detection: %d of %d untagged streets removed", int(drop.sum()), int(cand.sum()))
    return ways


# ============================================================ topology


def split_into_edges(walk: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Split ways at every node shared with another walkable way (or repeated within one)
    and at way endpoints."""
    counts: Counter = Counter()
    endpoints = set()
    for refs in walk["node_refs"]:
        counts.update(refs)
        endpoints.add(refs[0])
        endpoints.add(refs[-1])
    is_vertex = lambda r: r in endpoints or counts[r] >= 2  # noqa: E731

    keep_cols = [c for c in walk.columns if c not in ("node_refs", "geometry")]
    rows, geoms = [], []
    for rec, refs, geom in zip(walk[keep_cols].to_dict("records"), walk["node_refs"], walk.geometry.values):
        coords = list(geom.coords)
        if len(coords) != len(refs):
            raise ValueError(f"way {rec['osm_way_id']}: coords/refs length mismatch")
        start = 0
        for j in range(1, len(refs)):
            if is_vertex(refs[j]) or j == len(refs) - 1:
                seg = coords[start : j + 1]
                if len(seg) >= 2:
                    line = LineString(seg)
                    if line.length > 0:
                        r = dict(rec)
                        r["u"], r["v"] = int(refs[start]), int(refs[j])
                        rows.append(r)
                        geoms.append(line)
                start = j
    edges = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=geoms, crs=walk.crs)
    edges["length_m"] = edges.geometry.length
    return edges


def node_table(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    starts = shapely.get_point(edges.geometry.values, 0)
    ends = shapely.get_point(edges.geometry.values, -1)
    pts = pd.DataFrame(
        {
            "node_id": np.concatenate([edges["u"].to_numpy(), edges["v"].to_numpy()]),
            "x": np.concatenate([shapely.get_x(starts), shapely.get_x(ends)]),
            "y": np.concatenate([shapely.get_y(starts), shapely.get_y(ends)]),
        }
    ).drop_duplicates("node_id")
    deg = pd.concat([edges["u"], edges["v"]]).value_counts()
    pts["degree"] = pts["node_id"].map(deg).astype(int)
    return gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts.x, pts.y), crs=edges.crs).reset_index(drop=True)


STUB_MAX_M = 30.0
_PEDESTRIAN_KINDS_FOR_BRIDGING = {
    "sidewalk", "crossing", "footway", "path", "shared_path", "steps", "pedestrian_street",
    "traffic_island", "access_aisle", "corridor",
}


def prune_removed_road_stubs(edges: gpd.GeoDataFrame, removed_nodes: set[int]) -> tuple[gpd.GeoDataFrame, int]:
    """Drop short street-centerline dead ends that terminate on the centerline of a road
    that was removed because its sidewalks are mapped separately.

    These stubs are the piece of a side street between its crossing node and the main road's
    centerline. They add no connectivity, and leaving them would let origins/destinations
    snap into the middle of a main road. Single pass by design: only the stub touching the
    removed road is dropped, never a chain of edges.
    """
    deg = pd.concat([edges["u"], edges["v"]]).value_counts()
    is_center = edges["representation"] == STREET_CENTERLINE
    short = edges["length_m"] <= STUB_MAX_M
    u_dead = edges["u"].map(deg).eq(1) & edges["u"].isin(removed_nodes)
    v_dead = edges["v"].map(deg).eq(1) & edges["v"].isin(removed_nodes)
    # the other end must stay connected, otherwise this is an isolated piece, not a stub
    u_other_ok = edges["v"].map(deg).ge(2)
    v_other_ok = edges["u"].map(deg).ge(2)
    drop = is_center & short & ((u_dead & u_other_ok) | (v_dead & v_other_ok))
    return edges[~drop].reset_index(drop=True), int(drop.sum())


def _layer_key(edges: gpd.GeoDataFrame) -> pd.Series:
    layer = edges["layer"].fillna("0").astype(str) if "layer" in edges else pd.Series("0", index=edges.index)
    return layer.where(layer != "", "0")


def surface_gap_test(planim: gpd.GeoDataFrame, ways: gpd.GeoDataFrame, barriers: gpd.GeoDataFrame):
    """Evidence test for bridging a gap wider than GAP_TOLERANCE_M.

    Returns ``accept(kind, line)``: True when the straight connector ``line`` from a dead end
    of an edge of ``kind`` runs on walkable surface in the region's planimetric survey — sidewalk
    polygons, plus painted-crosswalk polygons when the dead end is itself a crossing that
    stops short — and crosses no road centerline, railway, waterway or stairs polygon.
    """
    desc = planim["DESCRIPTION"].str.strip()
    sidewalk = np.asarray(planim.geometry[desc.isin(WALKABLE_SURFACE)].values)
    crosswalk = np.asarray(planim.geometry[desc.eq("Crosswalk")].values)
    blockers = np.concatenate([
        np.asarray(planim.geometry[desc.eq("Stairs")].values),
        np.asarray(ways.geometry[~ways["highway"].isin(_NOT_A_ROAD)].values),
        np.asarray(barriers.geometry.values),
    ])
    trees = {"sidewalk": (shapely.STRtree(sidewalk), sidewalk), "crosswalk": (shapely.STRtree(crosswalk), crosswalk)}
    block_tree = shapely.STRtree(blockers)

    def covered(line: LineString, name: str) -> float:
        tree, geoms = trees[name]
        hits = tree.query(line, predicate="intersects")
        return float(sum(line.intersection(geoms[i]).length for i in hits))

    def accept(kind: str, line: LineString) -> bool:
        if line.length <= 1.0:
            return False
        # the ends touch the edges being joined (the target may be a street); only what lies
        # strictly between them can block
        if len(block_tree.query(substring(line, 0.5, line.length - 0.5), predicate="intersects")):
            return False
        on = covered(line, "sidewalk") + (covered(line, "crosswalk") if kind == "crossing" else 0.0)
        return on / line.length >= SURFACE_GAP_MIN_COVER

    return accept


def bridge_gaps(edges: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, tolerance: float = GAP_TOLERANCE_M,
                accept=None, reason: str | None = None, skip_nodes: frozenset[int] = frozenset(),
                forbidden=None) -> tuple[gpd.GeoDataFrame, int]:
    """Connect dead ends of pedestrian ways to the nearest point of another edge within
    ``tolerance``, using explicit inferred connector edges. With ``accept(kind, line)`` a
    connector is made only if the test passes for the straight line it would follow.

    - ``skip_nodes`` are dead ends that are real, not mapping slips (a way cut at the fence of a
      closed area, see ``clip_restricted``): they are never bridged.
    - No connector may enter ``forbidden`` (closed areas, shrunk a little so the fence line
      itself does not count).

    - Only pedestrian-way dead ends are bridged (a sidewalk stopping 1.5 m short of another
      sidewalk is a mapping gap; a driveway ending is not).
    - Targets must be on the same OSM ``layer`` so bridges/tunnels are never short-circuited.
    - If the nearest point is interior to the target edge, the target is split there; the two
      halves keep all attributes. Mapped geometry is otherwise never changed.
    Returns (edges, connectors_added).
    """
    deg = nodes.set_index("node_id")["degree"]
    inc = pd.concat([
        edges[["u", "edge_id", "kind"]].rename(columns={"u": "node_id"}),
        edges[["v", "edge_id", "kind"]].rename(columns={"v": "node_id"}),
    ])
    inc = inc[inc["node_id"].map(deg).eq(1) & inc["kind"].isin(_PEDESTRIAN_KINDS_FOR_BRIDGING) & ~inc["node_id"].isin(skip_nodes)]
    if inc.empty:
        return edges, 0
    pos = nodes.set_index("node_id")
    layer = _layer_key(edges)
    e_by_id = edges.set_index("edge_id")
    layer_by_id = pd.Series(layer.to_numpy(), index=edges["edge_id"].to_numpy())
    tree = shapely.STRtree(edges.geometry.values)
    edge_ids = edges["edge_id"].to_numpy()

    splits: dict[int, list[tuple[float, int]]] = {}  # edge_id -> [(t, src_node)]
    vertex_links: list[tuple[int, int]] = []
    for nid, own_eid, own_kind in zip(inc["node_id"], inc["edge_id"], inc["kind"]):
        p = Point(pos.at[nid, "x"], pos.at[nid, "y"])
        cands = tree.query(p, predicate="dwithin", distance=tolerance)
        best, best_d = None, np.inf
        for ci in cands:
            eid = edge_ids[ci]
            if eid == own_eid:
                continue
            row = e_by_id.loc[eid]
            if nid in (row["u"], row["v"]) or layer_by_id[eid] != layer_by_id[own_eid]:
                continue
            d = row.geometry.distance(p)
            if d < best_d:
                best, best_d = eid, d
        if best is None:
            continue
        g = e_by_id.at[best, "geometry"]
        t = g.project(p)
        line = LineString([p, g.interpolate(t)])
        if accept is not None and not accept(own_kind, line):
            continue
        if forbidden is not None and line.intersects(forbidden):
            continue
        # prefer an existing vertex when it is itself within tolerance (avoids sliver
        # splits); otherwise split at the projection so connectors never exceed tolerance
        u_id, v_id = int(e_by_id.at[best, "u"]), int(e_by_id.at[best, "v"])
        du = p.distance(Point(g.coords[0]))
        dv = p.distance(Point(g.coords[-1]))
        if t <= 0.5 and du <= tolerance:
            vertex_links.append((nid, u_id))
        elif t >= g.length - 0.5 and dv <= tolerance:
            vertex_links.append((nid, v_id))
        elif 0 < t < g.length:
            splits.setdefault(int(best), []).append((t, int(nid)))

    next_id = int(min(nodes["node_id"].min(), 0)) - 1  # synthetic node ids are negative

    def coords_of_src(nid: int) -> tuple[float, float]:
        return float(pos.at[nid, "x"]), float(pos.at[nid, "y"])
    new_rows, drop_ids, links = [], [], list(vertex_links)
    for eid, items in splits.items():
        row = e_by_id.loc[eid]
        g = row.geometry
        items.sort()
        cuts: list[tuple[float, int]] = []  # (t, new_node_id)
        for t, src in items:
            if cuts and t - cuts[-1][0] < 0.5 and Point(coords_of_src(src)).distance(g.interpolate(cuts[-1][0])) <= tolerance:
                links.append((src, cuts[-1][1]))
                continue
            cuts.append((t, next_id))
            links.append((src, next_id))
            next_id -= 1
        bounds = [(0.0, int(row["u"]))] + cuts + [(g.length, int(row["v"]))]
        for (t0, n0), (t1, n1) in zip(bounds[:-1], bounds[1:]):
            r = row.to_dict()
            r["geometry"] = substring(g, t0, t1)
            r["u"], r["v"] = n0, n1
            r["length_m"] = r["geometry"].length
            r["split_from_edge"] = int(eid)
            new_rows.append(r)
        drop_ids.append(eid)

    out = edges[~edges["edge_id"].isin(drop_ids)].copy()
    if new_rows:
        out = pd.concat([out, gpd.GeoDataFrame(new_rows, geometry="geometry", crs=edges.crs)], ignore_index=True)

    # node coordinates for connector geometry (including newly created split nodes)
    coords = {int(k): (x, y) for k, x, y in zip(nodes["node_id"], nodes["x"], nodes["y"])}
    for r in new_rows:
        g = r["geometry"]
        coords.setdefault(int(r["u"]), g.coords[0])
        coords.setdefault(int(r["v"]), g.coords[-1])
    seen, conn_rows = set(), []
    for s, t in links:
        key = (min(s, t), max(s, t))
        if s == t or key in seen:
            continue
        seen.add(key)
        # a link to an existing vertex or a merged split point follows a line not tested above
        if forbidden is not None and LineString([coords[s], coords[t]]).intersects(forbidden):
            continue
        conn_rows.append({"u": s, "v": t, "kind": "connector", "representation": INFERRED_CONNECTOR,
                          "class_reason": reason or f"pedestrian dead end within {tolerance} m of another edge",
                          "osm_way_id": -1, "geometry": LineString([coords[s], coords[t]])})
    if conn_rows:
        conn = gpd.GeoDataFrame(conn_rows, geometry="geometry", crs=edges.crs)
        conn["length_m"] = conn.geometry.length
        out = pd.concat([out, conn], ignore_index=True)
    out["edge_id"] = np.arange(len(out), dtype=np.int64)
    log.info("gap bridging: %d inferred connectors (%d edges split)", len(conn_rows), len(drop_ids))
    return out, len(conn_rows)


def near_miss_dead_ends(edges: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, radius: float) -> int:
    """Pedestrian dead ends that have another edge on the same layer within ``radius`` yet
    are not joined to it. Reported, never bridged: with no evidence of a connection, near is
    not connected."""
    deg = nodes.set_index("node_id")["degree"]
    pos = nodes.set_index("node_id")
    layer = _layer_key(edges).to_numpy()
    tree = shapely.STRtree(edges.geometry.values)
    eu, ev = edges["u"].to_numpy(), edges["v"].to_numpy()
    ped = edges["kind"].isin(_PEDESTRIAN_KINDS_FOR_BRIDGING).to_numpy()
    n = 0
    for i in np.flatnonzero(ped):
        for nid in (eu[i], ev[i]):
            if deg[nid] != 1:
                continue
            near = tree.query(Point(pos.at[nid, "x"], pos.at[nid, "y"]), predicate="dwithin", distance=radius)
            if any(j != i and nid not in (eu[j], ev[j]) and layer[j] == layer[i] for j in near):
                n += 1
    return n


def label_components(edges: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    idx = {nid: i for i, nid in enumerate(nodes["node_id"])}
    g = ig.Graph(n=len(nodes), edges=list(zip(edges["u"].map(idx), edges["v"].map(idx))), directed=False)
    memb = np.asarray(g.connected_components().membership)
    # component 0 = largest by total edge length
    comp_of_edge = memb[edges["u"].map(idx).to_numpy()]
    lengths = pd.Series(edges["length_m"].to_numpy()).groupby(comp_of_edge).sum().sort_values(ascending=False)
    remap = {c: r for r, c in enumerate(lengths.index)}
    # isolated nodes cannot exist (nodes come from edges), so every component has length
    nodes = nodes.copy()
    nodes["component"] = [remap[c] for c in memb]
    edges = edges.copy()
    edges["component"] = [remap[c] for c in comp_of_edge]
    return edges, nodes


RESTRICTED_MIN_PIECE_M = 1.0  # a sliver outside the fence shorter than this is dropped with the rest
# How far inside a closed area a connector may reach before it counts as entering: the fence is
# drawn as a line, and connectors between two cut ends lie along it.
RESTRICTED_FENCE_TOLERANCE_M = 0.5


def read_restricted_areas() -> gpd.GeoDataFrame:
    path = INTERIM_DIR / "osm_restricted_areas.parquet"
    if not path.exists():
        raise RuntimeError("osm_restricted_areas.parquet missing; rerun `pf extract-osm`")
    return gpd.read_parquet(path)


def clip_restricted(edges: gpd.GeoDataFrame, areas: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, set[int], dict]:
    """Remove every part of every edge that lies inside an area closed to the public
    (osm.RESTRICTED_AREA_TAGS).

    A military installation's streets and paths are mapped like public ones, but a walker cannot
    get past the gate: routing through one would invent a route. An edge that crosses the fence is
    cut there and keeps only its parts outside, so a gate road leads up to the fence and stops, and
    a street that clips a corner of the area loses that corner (it is no longer one walkable line).
    The point where a way is cut is a new node (negative id, like the gap bridger's), and those
    nodes are returned so the gap bridger never joins them up again. Returns the edges, the cut
    nodes and what was removed, by area."""
    if areas.empty:
        return edges, set(), {}
    areas = areas.to_crs(edges.crs)
    union = areas.geometry.union_all()
    hit = edges.geometry.intersects(union).to_numpy() & ~edges.geometry.touches(union).to_numpy()
    if not hit.any():
        return edges, set(), {}
    next_id = int(min(edges["u"].min(), edges["v"].min(), 0)) - 1
    cut_ids: dict[tuple[float, float], int] = {}  # one node per cut point, shared by edges cut there

    def node_at(xy, original_xy, original_id: int) -> int:
        nonlocal next_id
        if shapely.Point(xy).distance(shapely.Point(original_xy)) < 1e-6:
            return original_id
        key = (round(xy[0], 2), round(xy[1], 2))
        if key not in cut_ids:
            cut_ids[key] = next_id
            next_id -= 1
        return cut_ids[key]

    removed: dict[str, dict] = {}
    names = list(areas["name"].fillna("unnamed"))
    rows = []
    for rec in edges[hit].to_dict("records"):
        g = rec["geometry"]
        inside = g.intersection(union).length
        name = next((n for n, a in zip(names, areas.geometry) if g.intersects(a)), "unnamed")
        r = removed.setdefault(name, {"edges_cut": 0, "edges_removed": 0, "km": 0.0})
        r["km"] += inside / 1000
        outside = g.difference(union)
        parts = [q for q in getattr(outside, "geoms", [outside]) if q.geom_type == "LineString" and q.length >= RESTRICTED_MIN_PIECE_M]
        if not parts:
            r["edges_removed"] += 1
            continue
        r["edges_cut"] += 1
        a, b = g.coords[0], g.coords[-1]
        for q in parts:
            # keep the original direction of travel along the way: a piece runs u -> v
            if g.project(shapely.Point(q.coords[0])) > g.project(shapely.Point(q.coords[-1])):
                q = shapely.reverse(q)
            piece = dict(rec)
            piece["geometry"] = q
            piece["u"] = node_at(q.coords[0], a, int(rec["u"]))
            piece["v"] = node_at(q.coords[-1], b, int(rec["v"]))
            piece["length_m"] = q.length
            rows.append(piece)
    out = edges[~hit]
    if rows:
        out = pd.concat([out, gpd.GeoDataFrame(rows, geometry="geometry", crs=edges.crs)], ignore_index=True)
    for r in removed.values():
        r["km"] = round(r["km"], 1)
    log.info("restricted areas: %s; %d cut points at the fence", removed, len(cut_ids))
    return out.reset_index(drop=True), set(cut_ids.values()), removed


# ============================================================ entry point


def build() -> dict:
    ensure_dirs()
    ways = gpd.read_parquet(INTERIM_DIR / "osm_ways.parquet").to_crs(ANALYSIS_CRS)
    ways = classify_ways(ways)
    reasons = ways.loc[ways["kind"].isna(), "class_reason"].value_counts().to_dict()
    # Scope: the region plus a buffer so boundary streets (in DC: Western/Eastern/Southern
    # Ave, whose sidewalks may lie in Maryland) stay routable. The region's inventories do
    # not cover the buffer, so edges outside the boundary are flagged ``outside_region``.
    buffer_m = REGION.buffer_m
    boundary = gpd.read_parquet(raw_path(REGION.boundary)).to_crs(ANALYSIS_CRS).geometry.union_all()
    in_scope = ways.geometry.intersects(boundary.buffer(buffer_m))
    out_of_scope = ways["kind"].notna() & ~in_scope
    ways.loc[out_of_scope, ["kind", "representation"]] = None
    ways.loc[out_of_scope, "class_reason"] = f"outside the region + {buffer_m:.0f} m"
    walk = ways[ways["kind"].notna()].reset_index(drop=True)
    walk["outside_region"] = ~walk.geometry.within(boundary)
    removed = ways["class_reason"].isin(
        ["sidewalks mapped separately", "separate sidewalks detected geometrically",
         "service road with separate sidewalks"]
    )
    removed_nodes = {n for refs in ways.loc[removed, "node_refs"] for n in refs}

    edges = split_into_edges(walk)
    # edge-level scope: long ways (e.g. the C&O towpath) can touch the buffer while running
    # for kilometers outside it; keep an edge only if most of it is in scope
    scope = boundary.buffer(buffer_m)
    inside_len = edges.geometry.intersection(scope).length
    keep = inside_len >= 0.5 * edges["length_m"]
    n_out = int((~keep).sum())
    edges = edges[keep].reset_index(drop=True)
    areas = read_restricted_areas()
    edges, fence_nodes, restricted = clip_restricted(edges, areas)
    # connectors may not enter a closed area; the fence line itself (where cut ends lie) is allowed
    fence = areas.to_crs(ANALYSIS_CRS).geometry.union_all().buffer(-RESTRICTED_FENCE_TOLERANCE_M) if len(areas) else None
    no_entry = dict(skip_nodes=frozenset(fence_nodes), forbidden=fence)
    edges["outside_region"] = ~edges.geometry.within(boundary)
    edges, n_stubs = prune_removed_road_stubs(edges, removed_nodes)
    edges["edge_id"] = np.arange(len(edges), dtype=np.int64)
    nodes = node_table(edges)
    edges, n_conn = bridge_gaps(edges, nodes, **no_entry)
    n_surface = 0
    if REGION.role("planimetrics"):
        from .roles import read as read_role

        planim, _ = read_role("planimetrics")
        barriers = gpd.read_parquet(INTERIM_DIR / "osm_barriers.parquet").to_crs(ANALYSIS_CRS)
        edges, n_surface = bridge_gaps(edges, node_table(edges), tolerance=SURFACE_GAP_MAX_M,
                                       accept=surface_gap_test(planim, ways, barriers), reason=SURFACE_GAP_REASON, **no_entry)
    else:
        log.info("gap bridging: no planimetric survey for %s; only gaps up to %.0f m are bridged",
                 REGION.id, GAP_TOLERANCE_M)
    # recomputed after bridging so connectors and split halves carry the flag too
    edges["outside_region"] = ~edges.geometry.within(boundary)
    nodes = node_table(edges)
    # A kept edge that touches the centerline of a road removed for having separately mapped
    # sidewalks meets that road with no crossing edge; enrich uses this to price the
    # unmodeled crossing (see enrich.enrich_implicit_crossings).
    nodes["on_removed_road"] = nodes["node_id"].isin(removed_nodes)
    edges, nodes = label_components(edges, nodes)

    edges = edges.drop(columns=["all_tags"], errors="ignore")
    edges.to_parquet(GRAPH_DIR / "edges_base.parquet")
    nodes.to_parquet(GRAPH_DIR / "nodes.parquet")

    comp_len = edges.groupby("component")["length_m"].sum().sort_values(ascending=False)
    stats = {
        "ways_total": int(len(ways)),
        "ways_walkable": int(len(walk)),
        "excluded_by_reason": {k: int(v) for k, v in reasons.items()},
        "edges": int(len(edges)),
        "nodes": int(len(nodes)),
        "km_by_kind": edges.groupby("kind")["length_m"].sum().div(1000).round(1).to_dict(),
        "km_by_representation": edges.groupby("representation")["length_m"].sum().div(1000).round(1).to_dict(),
        "components": int(edges["component"].nunique()),
        "largest_component_share_of_length": round(float(comp_len.iloc[0] / comp_len.sum()), 4),
        "dead_end_nodes": int((nodes["degree"] == 1).sum()),
        "dead_ends_near_another_edge_unbridged": near_miss_dead_ends(edges, nodes, SURFACE_GAP_MAX_M),
        "near_miss_radius_m": SURFACE_GAP_MAX_M,
        "removed_road_stubs_pruned": n_stubs,
        "edges_dropped_outside_scope": n_out,
        "restricted_areas": restricted,
        "restricted_fence_cut_points": len(fence_nodes),
        "inferred_connectors": n_conn + n_surface,
        "inferred_connectors_across_mapped_sidewalk_surface": n_surface,
    }
    (GRAPH_DIR / "build_stats.json").write_text(json.dumps(stats, indent=2))
    log.info("graph: %s", json.dumps(stats))
    return stats


__all__ = ["build", "classify_ways", "split_into_edges", "bridge_gaps", "surface_gap_test", "sidewalk_side_coverage",
           "PEDESTRIAN_WAY", "STREET_CENTERLINE", "INFERRED_CONNECTOR"]
