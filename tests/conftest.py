"""Shared fixtures: synthetic edges/networks with every column the cost model reads."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from pathfinder.config import ANALYSIS_CRS, GRAPH_DIR

GRADED_DEFAULTS = {
    # attr: (value, status)
    "crossing_control": (None, "unknown"),
    "signal_lpi": (None, "unknown"),
    "signal_aps": (None, "unknown"),
    "signal_hawk": (None, "unknown"),
    "signal_all_ped_phase": (None, "unknown"),
    "crossed_lanes": (None, "unknown"),
    "crossed_width_m": (None, "unknown"),
    "crossed_speed_mph": (None, "unknown"),
    "crossed_aadt": (None, "unknown"),
    "kerb_a": (None, "unknown"),
    "kerb_b": (None, "unknown"),
    "ramp_check_a": (None, "unknown"),
    "ramp_check_b": (None, "unknown"),
    "step_free": (True, "verified"),
    "surface_class": ("smooth", "verified"),
    "sidewalk_presence": (None, "unknown"),
    "sidewalk_condition": (None, "unknown"),
    "sidewalk_width_m": (None, "unknown"),
    "street_lanes": (None, "unknown"),
    "street_speed_mph": (None, "unknown"),
    "grade_avg_pct": (1.0, "verified"),
    "grade_max_pct": (1.0, "verified"),
}


def make_edges(rows: list[dict], nodes_xy: dict[int, tuple[float, float]]) -> gpd.GeoDataFrame:
    """Build an edges GeoDataFrame. Each row needs ``u``, ``v``; anything else overrides
    defaults. A graded attribute given as a value implies status ``verified`` unless
    ``<attr>_status`` is also given."""
    recs = []
    for i, r in enumerate(rows):
        rec = {"edge_id": i, "kind": "sidewalk", "representation": "pedestrian_way", "osm_way_id": 1000 + i,
               "offstreet": False, "ps_obstacles": 0, "ps_surface_problems": 0, "component": 0,
               "crossing_total_length_m": np.nan, "outside_region": False,
               "xing_u": False, "xing_u_control": None, "xing_u_width_m": np.nan, "xing_u_kerb": None,
               "xing_u_kerb_source": None, "xing_u_kerb_as_of": None,
               "xing_v": False, "xing_v_control": None, "xing_v_width_m": np.nan, "xing_v_kerb": None,
               "xing_v_kerb_source": None, "xing_v_kerb_as_of": None}
        for attr, (val, st) in GRADED_DEFAULTS.items():
            rec[attr], rec[f"{attr}_status"] = val, st
            rec[f"{attr}_source"], rec[f"{attr}_as_of"] = ("test" if val is not None else None), None
        for k, v in r.items():
            rec[k] = v
            if k in GRADED_DEFAULTS and f"{k}_status" not in r:
                rec[f"{k}_status"] = "unknown" if v is None else "verified"
        rec["geometry"] = LineString([nodes_xy[rec["u"]], nodes_xy[rec["v"]]])
        recs.append(rec)
    e = gpd.GeoDataFrame(recs, geometry="geometry", crs=ANALYSIS_CRS)
    e["length_m"] = e.geometry.length
    cr = e["kind"].eq("crossing")
    e.loc[cr, "crossing_total_length_m"] = e.loc[cr].groupby("osm_way_id")["length_m"].transform("sum")
    return e


def make_nodes(nodes_xy: dict[int, tuple[float, float]]) -> gpd.GeoDataFrame:
    df = pd.DataFrame([{"node_id": k, "x": x, "y": y} for k, (x, y) in nodes_xy.items()])
    df["component"] = 0
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs=ANALYSIS_CRS)


def grid(n: int = 8, spacing: float = 100.0) -> tuple[dict[int, tuple[float, float]], list[dict]]:
    """n x n street grid of sidewalk edges."""
    xy = {r * n + c: (c * spacing, r * spacing) for r in range(n) for c in range(n)}
    rows = []
    for r in range(n):
        for c in range(n):
            a = r * n + c
            if c + 1 < n:
                rows.append({"u": a, "v": a + 1})
            if r + 1 < n:
                rows.append({"u": a, "v": a + n})
    return xy, rows


@pytest.fixture
def built_graph():
    """The real citywide graph, or skip if Phase 0 artifacts are not built."""
    p = GRAPH_DIR / "edges.parquet"
    if not p.exists():
        pytest.skip("graph not built (run pf fetch / extract-osm / build-graph / enrich)")
    return gpd.read_parquet(p)
