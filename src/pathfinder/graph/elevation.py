"""Sustained slope of each edge from the region's 1 m bare-earth DTM (role ``dtm``).

Outputs per edge (graded as ``evidence`` attributes):
- ``grade_max_pct``  steepest sustained grade: the steepest least-squares slope over a
                     WINDOW_M stretch centered on the edge, × 100
- ``grade_avg_pct``  |Δz| over the whole edge / length × 100
- ``rise_m``         signed z(v) − z(u), for direction-aware routing later
plus ``grade_max_spread_pct``, ``grade_quality``, ``grade_dtm_discontinuity`` and
``grade_separation_masked`` describing how far the reading can be trusted.

How the reading is made, and why
--------------------------------
* **A fitted slope over 10 m, not a difference over 6 m.** "Sustained" is tied to the
  longest run an ADA ramp may have at 8.33 % (30 ft ≈ 9 m). The slope is a least-squares fit
  through 1 m samples, which uses every sample in the window instead of only its two ends.
* **Short edges borrow their surroundings from the network.** A third of pedestrian edges
  are shorter than 6 m. Their profile is extended along the edges that continue them most
  directly (around a corner if nothing runs straighter, through several short pieces if
  need be), so the window follows mapped walkways instead of a straight line projected into
  whatever lies beyond the edge.
* **Three parallel lines, and they must agree.** The mapped line is only accurate to a metre
  or two, and a metre sideways can be a curb, a tree box or a retaining wall. The profile is
  read on the line and 1 m to each side; the value is the median of the three and it is
  VERIFIED only when they agree within REPEATABILITY_TOL_PCT. On a sample of 6,000 sidewalks,
  shifting the line 1 m each way flipped the 8.33 % verdict for 16 % of edges with the old
  6 m end-difference and for 3 % of the edges this rule keeps as verified.

The DTM is bare earth, so it cannot describe bridge decks, tunnels, indoor corridors, stairs,
elevators or escalators — those edges stay ``unknown`` (stairs/elevators are handled by
step-free logic instead) or get an inferred end-to-end grade.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from pyproj import Transformer
from shapely.geometry import Point

from .. import evidence as ev
from ..config import INTERIM_DIR
from ..fetch import load_manifest, raw_path
from ..sources import REGION, SOURCES

log = logging.getLogger(__name__)

SAMPLE_SPACING_M = 1.0  # the DTM's own resolution
WINDOW_M = 10.0  # "sustained": about the longest run an ADA ramp may have at 8.33 % (30 ft)
MIN_WINDOW_SPAN_M = 6.0  # a window cut short at a dead end must still span this much
MIN_PROFILE_M = 4.0  # with less profile than this no slope is claimed at all
LATERAL_OFFSETS_M = (-1.0, 0.0, 1.0)  # parallel lines read for every profile
REPEATABILITY_TOL_PCT = 3.0  # the three lines must agree this closely for VERIFIED
MAX_CONTINUATION_HOPS = 8  # edges a profile may run through beyond its own (junctions are made of 1 m pieces)
DIRECTION_PROBE_M = 3.0  # an edge's direction at a node is taken over its first 3 m
NOT_TERRAIN_KINDS = {"steps", "escalator", "elevator", "corridor"}
# Readings outside this range are not terrain (spikes, fill values). Maryland runs from sea
# level to Backbone Mountain (1,012 m); DC from about 0 to 125 m.
TERRAIN_RANGE_M = (-50.0, 1100.0)


class TileSampler:
    """Nearest-pixel sampling across a directory of equally sized GeoTIFF tiles."""

    def __init__(self, tile_dir: Path):
        self.tiles: list[tuple[tuple[float, float, float, float], Path]] = []
        for p in sorted(tile_dir.glob("*.tif")):
            with rasterio.open(p) as ds:
                b = ds.bounds
                self.tiles.append(((b.left, b.bottom, b.right, b.top), p))
                self.crs = ds.crs
        if not self.tiles:
            raise FileNotFoundError(f"no DTM tiles in {tile_dir}")

    def sample(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = np.full(len(x), np.nan, dtype=np.float64)
        for (l, b, r, t), path in self.tiles:
            m = (x >= l) & (x < r) & (y > b) & (y <= t)
            if not m.any():
                continue
            with rasterio.open(path) as ds:
                arr = ds.read(1, masked=True)
                rows, cols = rasterio.transform.rowcol(ds.transform, x[m], y[m])
                rows = np.clip(np.asarray(rows), 0, ds.height - 1)
                cols = np.clip(np.asarray(cols), 0, ds.width - 1)
                vals = arr[rows, cols]
                v = np.asarray(vals.filled(np.nan), dtype=np.float64)
                v[(v < TERRAIN_RANGE_M[0]) | (v > TERRAIN_RANGE_M[1])] = np.nan
                # exportImage fills areas outside the LiDAR footprint with exactly 0.0
                # instead of nodata (DC beyond its boundary; Maryland's county services
                # beyond the county); real terrain is never an exact float zero.
                v[v == 0.0] = np.nan
                out[m] = v
        return out


# ============================================================ profiles along the network


def continuations(u: np.ndarray, v: np.ndarray, geoms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each end of each edge, the edge that continues it most directly through that node.

    Returns ``(cont, cont_end)``, both shaped (n, 2) with column 0 = the u end and column 1 =
    the v end. ``cont`` is the position of the continuing edge (-1 at a dead end) and
    ``cont_end`` says which of *its* ends is at the shared node (0 = its start, 1 = its end).
    The least-deflected neighbor is taken whatever the angle: someone walking the edge has
    to go somewhere, and any mapped walkway is a better place to read the slope than open
    ground.
    """
    n = len(geoms)
    length = shapely.length(geoms)
    probe = np.minimum(length, DIRECTION_PROBE_M)
    start = shapely.get_coordinates(shapely.get_point(geoms, 0))
    end = shapely.get_coordinates(shapely.get_point(geoms, -1))
    from_start = shapely.get_coordinates(shapely.line_interpolate_point(geoms, probe)) - start
    from_end = shapely.get_coordinates(shapely.line_interpolate_point(geoms, length - probe)) - end
    dirs = np.vstack([from_start, from_end])
    norm = np.hypot(dirs[:, 0], dirs[:, 1])
    dirs = dirs / np.where(norm > 0, norm, np.nan)[:, None]

    node = np.concatenate([u, v])
    edge = np.concatenate([np.arange(n), np.arange(n)])
    which = np.concatenate([np.zeros(n, dtype=np.int8), np.ones(n, dtype=np.int8)])
    order = np.argsort(node, kind="stable")
    breaks = np.flatnonzero(np.diff(node[order])) + 1
    cont = np.full((n, 2), -1, dtype=np.int64)
    cont_end = np.full((n, 2), -1, dtype=np.int8)
    for grp in np.split(order, breaks):
        if len(grp) < 2:
            continue
        d = dirs[grp]
        # continuing straight on from edge i means leaving along -d_i; deflection to j is
        # the angle between -d_i and d_j
        cosang = -(d @ d.T)
        cosang[np.isnan(cosang)] = -2.0
        cosang[edge[grp][:, None] == edge[grp][None, :]] = -2.0  # never continue onto itself
        j = np.argmax(cosang, axis=1)
        ok = cosang[np.arange(len(grp)), j] > -2.0  # some other edge with a usable direction
        cont[edge[grp][ok], which[grp][ok]] = edge[grp][j[ok]]
        cont_end[edge[grp][ok], which[grp][ok]] = which[grp][j[ok]]
    return cont, cont_end


def build_profiles(geoms: np.ndarray, cont: np.ndarray, cont_end: np.ndarray) -> dict[str, np.ndarray]:
    """Sample points for every edge: its own line every SAMPLE_SPACING_M plus up to
    WINDOW_M / 2 along the continuing edge at each end.

    Flat arrays, grouped by ``owner`` in ascending order:
    ``owner`` (edge the profile is for), ``src`` (edge the sample lies on), ``dist`` (metres
    from the owner's start; negative before it, greater than its length beyond it), ``x``,
    ``y`` and the unit left normal ``nx``, ``ny``.
    """
    half = WINDOW_M / 2
    length = shapely.length(geoms)
    owner, src, dist, xy, normal = [], [], [], [], []
    for i, g in enumerate(geoms):
        L = length[i]
        nseg = max(int(np.ceil(L / SAMPLE_SPACING_M)), 1)
        ds = [np.linspace(0.0, L, nseg + 1)]
        pts = [shapely.get_coordinates(shapely.line_interpolate_point(g, ds[0]))]
        srcs = [np.full(nseg + 1, i)]
        for side in (0, 1):
            # walk outward through the chain of continuing edges until half a window is covered
            k_all, p_all, s_all = [], [], []
            cur, cur_side, covered, seen = i, side, 0.0, {i}
            for _ in range(MAX_CONTINUATION_HOPS):
                f = cont[cur, cur_side]
                if f < 0 or f in seen or covered >= half:
                    break
                seen.add(f)
                Lf = length[f]
                first = np.floor(covered / SAMPLE_SPACING_M) + 1
                k = np.arange(first, np.floor(min(half, covered + Lf) / SAMPLE_SPACING_M) + 1) * SAMPLE_SPACING_M
                if len(k):
                    into = k - covered  # distance into f from the node it shares with the chain
                    along = into if cont_end[cur, cur_side] == 0 else Lf - into
                    k_all.append(k)
                    p_all.append(shapely.get_coordinates(shapely.line_interpolate_point(geoms[f], along)))
                    s_all.append(np.full(len(k), f))
                covered += Lf
                cur, cur_side = f, 1 - cont_end[cur, cur_side]  # leave f by its far end
            if not k_all:
                continue
            k, p, s = np.concatenate(k_all), np.vstack(p_all), np.concatenate(s_all)
            if side == 0:  # before the start: farthest sample first
                ds.insert(0, -k[::-1])
                pts.insert(0, p[::-1])
                srcs.insert(0, s[::-1])
            else:
                ds.append(L + k)
                pts.append(p)
                srcs.append(s)
        p = np.vstack(pts)
        t = np.gradient(p, axis=0) if len(p) > 1 else np.zeros_like(p)
        tn = np.hypot(t[:, 0], t[:, 1])
        t = t / np.where(tn > 0, tn, 1.0)[:, None]
        owner.append(np.full(len(p), i))
        src.append(np.concatenate(srcs))
        dist.append(np.concatenate(ds))
        xy.append(p)
        normal.append(np.column_stack([-t[:, 1], t[:, 0]]))
    xy, normal = np.vstack(xy), np.vstack(normal)
    return {"owner": np.concatenate(owner), "src": np.concatenate(src), "dist": np.concatenate(dist),
            "x": xy[:, 0], "y": xy[:, 1], "nx": normal[:, 0].astype(np.float32), "ny": normal[:, 1].astype(np.float32)}


# ============================================================ slope from a profile

MIN_VALID_FRACTION = 0.8
MEDIAN_WINDOW = 5  # samples (~5 m): suppresses single-pixel spikes from walls/curbs


def _rolling_median(z: np.ndarray, w: int) -> np.ndarray:
    """Centered rolling median; ends are padded by repeating the end value (``nearest``),
    which cannot introduce artificial elevation changes the way zero padding does."""
    if len(z) < 3:
        return z
    from scipy.ndimage import median_filter

    return median_filter(z, size=min(w, len(z) | 1), mode="nearest")


def _smooth(z: np.ndarray) -> np.ndarray:
    """Rolling median within each run of valid samples; gaps stay NaN and are never bridged."""
    out = z.astype(float, copy=True)
    ok = ~np.isnan(out)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(np.int8), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        out[a:b] = _rolling_median(out[a:b], MEDIAN_WINDOW)
    return out


def sustained_grade(dist: np.ndarray, z: np.ndarray, start: float, end: float) -> float:
    """Steepest sustained grade (%) over the stretch ``[start, end]`` of a profile.

    For every sample in the stretch, a line is fitted by least squares through the samples
    within WINDOW_M / 2 of it; the steepest fit wins. A window with a missing sample is not
    used, and one cut short by the end of the profile must still span MIN_WINDOW_SPAN_M. If
    no window qualifies, the whole profile is fitted when it is complete and at least
    MIN_PROFILE_M long; otherwise NaN (no claim).
    """
    zs = _smooth(z)
    bad = np.isnan(zs)
    x = dist - dist.mean()  # centered: keeps the running sums well conditioned on long edges
    zz = np.where(bad, 0.0, zs)

    def csum(v: np.ndarray) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(v)])

    cb, cx, cz, cxx, cxz = csum(bad.astype(float)), csum(x), csum(zz), csum(x * x), csum(x * zz)
    centers = np.flatnonzero((dist >= start - 1e-9) & (dist <= end + 1e-9))
    lo = np.searchsorted(dist, dist[centers] - WINDOW_M / 2 - 1e-9, side="left")
    hi = np.searchsorted(dist, dist[centers] + WINDOW_M / 2 + 1e-9, side="right")
    n = hi - lo
    span = dist[np.maximum(hi - 1, 0)] - dist[lo]
    ok = (cb[hi] - cb[lo] == 0) & (n >= 3) & (span >= MIN_WINDOW_SPAN_M)
    if ok.any():
        lo, hi, n = lo[ok], hi[ok], n[ok]
        sx, sz, sxx, sxz = cx[hi] - cx[lo], cz[hi] - cz[lo], cxx[hi] - cxx[lo], cxz[hi] - cxz[lo]
        slope = (n * sxz - sx * sz) / (n * sxx - sx * sx)
        return float(np.max(np.abs(slope)) * 100)
    if not bad.any() and len(x) >= 3 and x[-1] - x[0] >= MIN_PROFILE_M:
        return float(abs(np.polyfit(x, zs, 1)[0]) * 100)
    return float("nan")


SUSPECT_GRADE_PCT = 25.0  # above this, a sustained grade on a mapped way is downgraded
# A bare-earth step this tall within DISCONTINUITY_RUN_M (>= 150 %) is a wall, not a surface
# anyone walks: stairs top out near 70 %. It means the mapped way and the DTM part company
# there — a sidewalk on a deck over a sunken road, a path along a retaining wall — so the
# DTM says nothing reliable about the walkway and the edge's grade is left UNKNOWN.
DISCONTINUITY_STEP_M = 3.0
DISCONTINUITY_RUN_M = 2.0
SEPARATION_MASK_M = 12.0  # DTM samples this close to a grade-separation point are ignored
ENDPOINT_TOLERANCE_M = 3.0


def _steps(dist: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Indices ``i`` where the smoothed profile changes by DISCONTINUITY_STEP_M or more
    between sample ``i`` and a later sample at most DISCONTINUITY_RUN_M away. Samples
    separated by missing data are never compared."""
    zs = _smooth(z)
    hits = []
    for k in range(1, int(round(DISCONTINUITY_RUN_M / SAMPLE_SPACING_M)) + 1):
        if len(zs) <= k:
            break
        close = dist[k:] - dist[:-k] <= DISCONTINUITY_RUN_M * 1.01
        # every sample in between must be valid, so the pair is not straddling a gap
        valid = ~np.isnan(zs)
        run = np.convolve(valid.astype(int), np.ones(k + 1, dtype=int), mode="valid") == k + 1
        hits.append(np.flatnonzero(close & run & (np.abs(zs[k:] - zs[:-k]) >= DISCONTINUITY_STEP_M)))
    return np.unique(np.concatenate(hits)) if hits else np.array([], dtype=int)


def _cross_median(zs_by_line: np.ndarray) -> np.ndarray:
    """Median across the parallel lines at each sample; NaN where no line has data."""
    out = np.full(zs_by_line.shape[1], np.nan)
    has = ~np.isnan(zs_by_line).all(axis=0)
    out[has] = np.nanmedian(zs_by_line[:, has], axis=0)
    return out


def profile_grades(dist: np.ndarray, zs_by_line: np.ndarray, length: float,
                   own_masked: np.ndarray | None = None) -> dict[str, float | bool]:
    """Grades for one edge from its profile read on parallel lines.

    ``zs_by_line`` is (lines, samples); ``dist`` runs from negative (before the edge) to
    beyond ``length`` (after it). Returns ``max`` (median over lines of the sustained grade),
    ``spread`` (max − min over lines; NaN unless every line gave a value), ``avg``, ``rise``
    and ``discontinuity``. Intentionally masked samples (grade separation) do not count
    against coverage; missing DTM data does.
    """
    nan = {"max": np.nan, "spread": np.nan, "avg": np.nan, "rise": np.nan, "discontinuity": False}
    own = (dist >= -1e-9) & (dist <= length + 1e-9)
    center = zs_by_line[len(zs_by_line) // 2]
    considered = int(own.sum()) - (int(own_masked.sum()) if own_masked is not None else 0)
    valid_own = int((~np.isnan(center[own])).sum())
    if valid_own < 2 or considered <= 0 or valid_own / considered < MIN_VALID_FRACTION:
        return nan

    # the cross-section median is the walkway's own profile: one-sided walls drop out
    zmed = _cross_median(zs_by_line)
    steps = _steps(dist, zmed)
    if len(steps):
        k_max = int(round(DISCONTINUITY_RUN_M / SAMPLE_SPACING_M))
        own_idx = np.flatnonzero(own)
        first, last = own_idx[0], own_idx[-1]
        if ((steps + k_max > first) & (steps < last)).any():
            return {**nan, "discontinuity": True}
        # a step out in a continuation: do not look past it
        zs_by_line = zs_by_line.copy()
        for s in steps:
            if s < first:
                zs_by_line[:, : s + 1] = np.nan
            else:
                zs_by_line[:, s + 1:] = np.nan
        zmed = _cross_median(zs_by_line)

    per_line = np.array([sustained_grade(dist, z, 0.0, length) for z in zs_by_line])
    if np.isnan(per_line).all():
        return nan
    mx = float(np.nanmedian(per_line))
    spread = float(per_line.max() - per_line.min()) if not np.isnan(per_line).any() else np.nan

    zs = _smooth(zmed)
    ok = own & ~np.isnan(zs)
    d_own, z_own = dist[ok], zs[ok]
    span = d_own[-1] - d_own[0] if len(d_own) >= 2 else 0.0
    rise = float(z_own[-1] - z_own[0]) if len(d_own) >= 2 else np.nan
    # over a few metres the end-to-end difference is DTM noise; the fitted window is the
    # better statement about a short edge
    avg = abs(rise) / span * 100 if span >= MIN_PROFILE_M else mx
    return {"max": max(mx, avg), "spread": spread, "avg": float(avg), "rise": rise, "discontinuity": False}


# ============================================================ grade separation


def grade_separation_points(edges: gpd.GeoDataFrame, highways: gpd.GeoDataFrame,
                            barriers: gpd.GeoDataFrame) -> dict[int, list[tuple[float, float]]]:
    """Points where an edge passes over/under another linear feature.

    In OSM, ways that meet at grade share a node. A crossing point that is not a shared node
    position of both geometries is therefore grade-separated (or a rare topology error).

    Masking policy for the bare-earth DTM:
    - feature tagged ``tunnel`` (below the walkway): walkway is on the surface -> not masked
      (except waterways: culverts are carved into the hydro-enforced DTM -> masked);
    - anything else, including ``bridge`` features above the walkway -> masked, because
      bare-earth processing removes bridge decks and interpolates the gap from abutments and
      embankments, which is wrong for a sidewalk passing underneath.
    """
    def is_yes(s: pd.Series) -> pd.Series:
        return s.notna() & ~s.isin(["no", "0"])

    hw = highways[~is_yes(highways["tunnel"])][["geometry"]]
    rail = barriers[barriers["railway"].notna() & ~is_yes(barriers["tunnel"])][["geometry"]]
    water = barriers[barriers["waterway"].notna()][["geometry"]]
    feats = pd.concat([hw, rail, water], ignore_index=True)
    fgeom = feats.geometry.values
    tree = shapely.STRtree(fgeom)
    ei, fi = tree.query(edges.geometry.values, predicate="intersects")
    out: dict[int, list[tuple[float, float]]] = {}
    egeom = edges.geometry.values

    def on_vertex(p: Point, g) -> bool:
        c = np.asarray(g.coords)
        return bool(np.min(np.hypot(c[:, 0] - p.x, c[:, 1] - p.y)) < 0.05)

    for a, b in zip(ei, fi):
        g, f = egeom[a], fgeom[b]
        inter = g.intersection(f)
        pts = [p for p in getattr(inter, "geoms", [inter]) if p.geom_type == "Point"]
        if not pts:
            continue
        e0, e1 = Point(g.coords[0]), Point(g.coords[-1])
        for p in pts:
            if p.distance(e0) <= ENDPOINT_TOLERANCE_M or p.distance(e1) <= ENDPOINT_TOLERANCE_M:
                continue
            if on_vertex(p, g) and on_vertex(p, f):
                continue  # shared OSM node => at-grade junction
            out.setdefault(int(edges.index[a]), []).append((p.x, p.y))
    return out


# ============================================================ entry point


def enrich_slope(edges: gpd.GeoDataFrame) -> None:
    for a in ("grade_avg_pct", "grade_max_pct"):
        ev.init_unknown(edges, a)
    edges["rise_m"] = np.nan
    # flags the coverage report reads; they stay False where no reading was taken
    for col in ("grade_separation_masked", "grade_dtm_discontinuity"):
        edges[col] = False
    edges["grade_quality"] = None
    edges["grade_max_spread_pct"] = np.nan
    dtm = REGION.role("dtm")
    if dtm is None or dtm not in load_manifest():
        log.warning("no elevation model for %s (%s); slope left unknown", REGION.id, dtm or "none registered")
        return
    sampler = TileSampler(raw_path(dtm))
    dtm_as_of = SOURCES[dtm].as_of  # when the terrain was flown, not when it was fetched

    def flag(col: str) -> pd.Series:
        return edges[col].notna() & ~edges[col].isin(["no", "0"]) if col in edges else pd.Series(False, index=edges.index)

    layer = edges["layer"].fillna("0") if "layer" in edges else pd.Series("0", index=edges.index)
    terrain = (
        ~edges["kind"].isin(NOT_TERRAIN_KINDS)
        & ~flag("bridge") & ~flag("tunnel") & ~flag("indoor") & ~flag("covered")
        & layer.isin(["0", ""])
    )
    sub = edges[terrain]
    geoms = np.asarray(sub.geometry.values)
    # profiles only ever continue onto other terrain edges: the DTM is wrong under a deck
    cont, cont_end = continuations(sub["u"].to_numpy(), sub["v"].to_numpy(), geoms)
    prof = build_profiles(geoms, cont, cont_end)
    owner, src, dist, x, y = prof["owner"], prof["src"], prof["dist"], prof["x"], prof["y"]
    log.info("slope: %d profile samples for %d edges", len(x), len(sub))

    # mask samples near grade-separation points of the edge they lie on
    highways = gpd.read_parquet(INTERIM_DIR / "osm_ways.parquet", columns=["tunnel", "bridge", "geometry"]).to_crs(edges.crs)
    barriers = gpd.read_parquet(INTERIM_DIR / "osm_barriers.parquet").to_crs(edges.crs)
    sep = grade_separation_points(sub, highways, barriers)
    pos_of = {lbl: i for i, lbl in enumerate(sub.index)}
    by_src = np.argsort(src, kind="stable")
    src_bounds = np.searchsorted(src[by_src], np.arange(len(sub) + 1))
    masked = np.zeros(len(x), dtype=bool)
    for lbl, pts in sep.items():
        i = pos_of[lbl]
        on_edge = by_src[src_bounds[i]:src_bounds[i + 1]]
        for px, py in pts:
            masked[on_edge] |= np.hypot(x[on_edge] - px, y[on_edge] - py) <= SEPARATION_MASK_M
    edges["grade_separation_masked"] = False
    edges.loc[list(sep), "grade_separation_masked"] = True
    log.info("slope: %d edges have grade-separation samples masked", len(sep))

    tr = Transformer.from_crs(edges.crs, sampler.crs, always_xy=True)
    z = np.empty((len(LATERAL_OFFSETS_M), len(x)), dtype=np.float32)
    for k, off in enumerate(LATERAL_OFFSETS_M):
        tx, ty = tr.transform(x + off * prof["nx"], y + off * prof["ny"])
        z[k] = sampler.sample(np.asarray(tx), np.asarray(ty))
    z[:, masked] = np.nan

    # build_profiles emits samples grouped by owner in ascending order
    assert np.all(np.diff(owner) >= 0)
    bounds = np.searchsorted(owner, np.arange(len(sub) + 1))
    length = shapely.length(geoms)
    n = len(sub)
    avg, mx, rise, spread = (np.full(n, np.nan) for _ in range(4))
    disc = np.zeros(n, dtype=bool)
    for i in range(n):
        a, b = bounds[i], bounds[i + 1]
        own = (dist[a:b] >= -1e-9) & (dist[a:b] <= length[i] + 1e-9)
        r = profile_grades(dist[a:b], z[:, a:b].astype(np.float64), float(length[i]), masked[a:b] & own)
        avg[i], mx[i], rise[i], spread[i], disc[i] = r["avg"], r["max"], r["rise"], r["spread"], r["discontinuity"]

    # DTM discontinuity: the reading describes a wall or a deck edge, not the walkway
    edges["grade_dtm_discontinuity"] = pd.Series(disc, index=sub.index).reindex(edges.index, fill_value=False)
    log.info("slope: %d edges left unknown because the DTM profile has a wall-like step", int(disc.sum()))

    vals_avg = pd.Series(np.round(avg, 1), index=sub.index).reindex(edges.index)
    vals_max = pd.Series(np.round(mx, 1), index=sub.index).reindex(edges.index)
    edges["grade_max_spread_pct"] = pd.Series(np.round(spread, 1), index=sub.index).reindex(edges.index)
    # Two reasons a reading is kept but only as INFERRED, so it never counts as verified:
    # - very steep: either a genuinely steep way or a mapped line that sits a few meters off
    #   the real path on a hillside, sampling the side slope. We cannot tell which, and in
    #   both interpretations it fails any wheelchair slope requirement.
    # - not repeatable: the parallel lines disagree, so the value depends on where exactly
    #   the mapper drew the line.
    suspect = vals_max > SUSPECT_GRADE_PCT
    unrepeatable = ~suspect & vals_max.notna() & ~(edges["grade_max_spread_pct"] <= REPEATABILITY_TOL_PCT)
    edges["grade_quality"] = np.where(vals_max.isna(), None,
                                      np.where(suspect, "suspect", np.where(unrepeatable, "not_repeatable", "ok")))
    ok = terrain & ~suspect & ~unrepeatable
    ev.fill_where_better(edges, "grade_avg_pct", ok, vals_avg, ev.VERIFIED, dtm, dtm_as_of)
    ev.fill_where_better(edges, "grade_max_pct", ok, vals_max, ev.VERIFIED, dtm, dtm_as_of)
    ev.fill_where_better(edges, "grade_avg_pct", terrain & ~ok, vals_avg, ev.INFERRED, dtm, dtm_as_of)
    ev.fill_where_better(edges, "grade_max_pct", terrain & ~ok, vals_max, ev.INFERRED, dtm, dtm_as_of)
    edges.loc[sub.index, "rise_m"] = np.round(rise, 2)
    log.info("slope: measured %d of %d edges (%d not repeatable, %d suspect)", int(vals_max.notna().sum()), len(edges),
             int(unrepeatable.sum()), int(suspect.sum()))

    # Bridges/tunnels/indoor: the deck is not in the bare-earth DTM, but its ends usually
    # meet the ground at abutments/portals. The end-to-end grade is an INFERRED estimate
    # (decks are engineered to modest grades; local humps are not captured).
    structural = ~terrain & ~edges["kind"].isin(NOT_TERRAIN_KINDS) & (edges["length_m"] >= MIN_WINDOW_SPAN_M)
    st = edges[structural]
    if len(st):
        ends = np.array([(g.coords[0], g.coords[-1]) for g in st.geometry.values])  # (n, 2, 2)
        ex, ey = tr.transform(ends[:, :, 0].ravel(), ends[:, :, 1].ravel())
        ez = sampler.sample(np.asarray(ex), np.asarray(ey)).reshape(-1, 2)
        grade = np.abs(ez[:, 1] - ez[:, 0]) / st["length_m"].to_numpy() * 100
        g = pd.Series(np.round(grade, 1), index=st.index).reindex(edges.index)
        g = g.where(g <= SUSPECT_GRADE_PCT)  # implausible => leave unknown
        ev.fill_where_better(edges, "grade_avg_pct", structural, g, ev.INFERRED, f"derived:{dtm}_endpoints", dtm_as_of)
        ev.fill_where_better(edges, "grade_max_pct", structural, g, ev.INFERRED, f"derived:{dtm}_endpoints", dtm_as_of)
        edges.loc[st.index, "rise_m"] = np.round(ez[:, 1] - ez[:, 0], 2)
        log.info("slope: %d structural edges given inferred end-to-end grade", int(g.notna().sum()))
