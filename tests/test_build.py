import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, box

from pathfinder.config import ANALYSIS_CRS
from pathfinder.graph.build import (
    INFERRED_CONNECTOR,
    SURFACE_GAP_MAX_M,
    SURFACE_GAP_REASON,
    bridge_gaps,
    clip_restricted,
    near_miss_dead_ends,
    node_table,
    prune_removed_road_stubs,
    sidewalk_side_coverage,
    split_into_edges,
    surface_gap_test,
)


def _ways(recs):
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "coords"} for r in recs])
    return gpd.GeoDataFrame(df, geometry=[LineString(r["coords"]) for r in recs], crs=ANALYSIS_CRS)


def test_split_at_shared_nodes_and_endpoints():
    ways = _ways([
        {"osm_way_id": 1, "kind": "sidewalk", "representation": "pedestrian_way",
         "node_refs": [1, 2, 3], "coords": [(0, 0), (50, 0), (100, 0)]},
        {"osm_way_id": 2, "kind": "crossing", "representation": "pedestrian_way",
         "node_refs": [2, 4], "coords": [(50, 0), (50, 20)]},
    ])
    e = split_into_edges(ways)
    assert sorted(map(tuple, e[["u", "v"]].to_numpy())) == [(1, 2), (2, 3), (2, 4)]
    assert e["length_m"].sum() == pytest.approx(120.0)


def test_interior_non_shared_nodes_do_not_split():
    ways = _ways([{"osm_way_id": 1, "kind": "sidewalk", "representation": "pedestrian_way",
                   "node_refs": [1, 2, 3], "coords": [(0, 0), (50, 1), (100, 0)]}])
    e = split_into_edges(ways)
    assert len(e) == 1 and (e.u.iat[0], e.v.iat[0]) == (1, 3)


def test_sidewalk_side_coverage_detects_both_sides():
    street = gpd.GeoSeries([LineString([(0, 0), (200, 0)])], crs=ANALYSIS_CRS)
    sidewalks = gpd.GeoSeries([LineString([(0, 8), (200, 8)]), LineString([(0, -8), (200, -8)])], crs=ANALYSIS_CRS)
    left, right = sidewalk_side_coverage(street, sidewalks)
    assert left[0] == pytest.approx(1.0) and right[0] == pytest.approx(1.0)
    left, right = sidewalk_side_coverage(street, sidewalks.iloc[:1])
    assert left[0] == pytest.approx(1.0) and right[0] == pytest.approx(0.0)


def test_sidewalk_side_coverage_ignores_far_parallel_features():
    street = gpd.GeoSeries([LineString([(0, 0), (200, 0)])], crs=ANALYSIS_CRS)
    far = gpd.GeoSeries([LineString([(0, 60), (200, 60)])], crs=ANALYSIS_CRS)  # next block over
    left, _ = sidewalk_side_coverage(street, far)
    assert left[0] == 0.0


def test_prune_removed_road_stubs_only_drops_short_centerline_stubs():
    xy = [((0, 0), (20, 0)), ((20, 0), (40, 0)), ((20, 0), (20, 50))]
    e = gpd.GeoDataFrame(
        {"u": [1, 2, 2], "v": [2, 3, 4], "kind": ["street_unknown"] * 3,
         "representation": ["street_centerline"] * 3},
        geometry=[LineString(c) for c in xy], crs=ANALYSIS_CRS)
    e["length_m"] = e.geometry.length
    # node 3 lies on a removed road; node 1 does not
    out, n = prune_removed_road_stubs(e, removed_nodes={3})
    assert n == 1 and set(zip(out.u, out.v)) == {(1, 2), (2, 4)}


def test_bridge_gaps_splits_target_and_discloses_connector():
    e = gpd.GeoDataFrame(
        {"u": [1, 3], "v": [2, 4], "kind": ["sidewalk", "sidewalk"], "representation": ["pedestrian_way"] * 2,
         "osm_way_id": [10, 11], "layer": [None, None]},
        geometry=[LineString([(0, 0), (100, 0)]), LineString([(50, 2), (50, 40)])], crs=ANALYSIS_CRS)
    e["length_m"] = e.geometry.length
    e["edge_id"] = np.arange(len(e))
    nodes = node_table(e)
    out, n = bridge_gaps(e, nodes)
    assert n == 1
    conn = out[out["representation"] == INFERRED_CONNECTOR]
    assert len(conn) == 1 and conn["length_m"].iat[0] == pytest.approx(2.0)
    # target split into two halves that preserve total length and attributes
    halves = out[out["osm_way_id"] == 10]
    assert len(halves) == 2 and halves["length_m"].sum() == pytest.approx(100.0)


def test_bridge_gaps_respects_layers():
    e = gpd.GeoDataFrame(
        {"u": [1, 3], "v": [2, 4], "kind": ["sidewalk", "sidewalk"], "representation": ["pedestrian_way"] * 2,
         "osm_way_id": [10, 11], "layer": ["1", None]},  # a bridge sidewalk above
        geometry=[LineString([(0, 0), (100, 0)]), LineString([(50, 2), (50, 40)])], crs=ANALYSIS_CRS)
    e["length_m"] = e.geometry.length
    e["edge_id"] = np.arange(len(e))
    out, n = bridge_gaps(e, node_table(e))
    assert n == 0


def _gap_fixture(gap_m: float):
    """A sidewalk that stops ``gap_m`` short of another sidewalk running across its end."""
    e = gpd.GeoDataFrame(
        {"u": [1, 3], "v": [2, 4], "kind": ["sidewalk", "sidewalk"], "representation": ["pedestrian_way"] * 2,
         "osm_way_id": [10, 11], "layer": [None, None]},
        geometry=[LineString([(0, 0), (100, 0)]), LineString([(50, gap_m), (50, 60)])], crs=ANALYSIS_CRS)
    e["length_m"] = e.geometry.length
    e["edge_id"] = np.arange(len(e))
    return e


def _evidence(sidewalk=None, road=None, stairs=None):
    polys, desc = [], []
    for g, d in ((sidewalk, "Sidewalk"), (stairs, "Stairs")):
        if g is not None:
            polys.append(g)
            desc.append(d)
    planim = gpd.GeoDataFrame({"DESCRIPTION": pd.Series(desc, dtype="object")}, geometry=polys, crs=ANALYSIS_CRS)
    ways = gpd.GeoDataFrame({"highway": ["residential"] if road is not None else []},
                            geometry=[road] if road is not None else [], crs=ANALYSIS_CRS)
    barriers = gpd.GeoDataFrame(geometry=[], crs=ANALYSIS_CRS)
    return surface_gap_test(planim, ways, barriers)


def test_wide_gap_is_bridged_only_across_mapped_sidewalk_surface():
    e = _gap_fixture(8.0)
    # no evidence: near is not connected
    out, n = bridge_gaps(e, node_table(e), tolerance=SURFACE_GAP_MAX_M, accept=_evidence())
    assert n == 0 and len(out) == 2
    # planimetric sidewalk surface spans the gap
    out, n = bridge_gaps(e, node_table(e), tolerance=SURFACE_GAP_MAX_M,
                         accept=_evidence(sidewalk=box(40, -1, 60, 9)), reason=SURFACE_GAP_REASON)
    conn = out[out["representation"] == INFERRED_CONNECTOR]
    assert n == 1 and conn["length_m"].iat[0] == pytest.approx(8.0)
    assert conn["class_reason"].iat[0] == SURFACE_GAP_REASON


def test_wide_gap_is_not_bridged_across_a_road_or_stairs():
    e = _gap_fixture(8.0)
    surface = box(40, -1, 60, 9)
    road = LineString([(0, 4), (100, 4)])
    assert bridge_gaps(e, node_table(e), tolerance=SURFACE_GAP_MAX_M, accept=_evidence(surface, road=road))[1] == 0
    stairs = box(48, 2, 52, 6)
    assert bridge_gaps(e, node_table(e), tolerance=SURFACE_GAP_MAX_M, accept=_evidence(surface, stairs=stairs))[1] == 0
    # surface under only half the gap is not enough
    assert bridge_gaps(e, node_table(e), tolerance=SURFACE_GAP_MAX_M, accept=_evidence(box(40, -1, 60, 4)))[1] == 0


def test_near_miss_dead_ends_are_counted_not_bridged():
    e = _gap_fixture(8.0)
    assert near_miss_dead_ends(e, node_table(e), SURFACE_GAP_MAX_M) == 1
    assert near_miss_dead_ends(e, node_table(e), 5.0) == 0


# ---------------------------------------------------------------- closed areas (Fort Detrick)

BASE = box(0, 0, 100, 100)


def _base():
    return gpd.GeoDataFrame({"name": ["Base"]}, geometry=[BASE], crs=ANALYSIS_CRS)


def _street(way, refs, coords, kind="street_unknown", rep="street_centerline"):
    return {"osm_way_id": way, "kind": kind, "representation": rep, "node_refs": refs, "coords": coords}


def test_gate_road_is_cut_at_the_fence_and_nothing_inside_remains():
    """The review case: a gate road a third inside the base was kept whole, so a route could run
    20 m behind the gate. Now it stops at the fence, at a new dead-end node."""
    e = split_into_edges(_ways([_street(3, [5, 6], [(-40, 50), (20, 50)])]))
    kept, fence, removed = clip_restricted(e, _base())
    assert len(kept) == 1
    g = kept.geometry.iloc[0]
    assert g.length == pytest.approx(40.0) and g.coords[-1] == pytest.approx((0.0, 50.0))
    assert kept["u"].iloc[0] == 5 and kept["v"].iloc[0] < 0 and fence == {int(kept["v"].iloc[0])}
    assert kept.geometry.intersection(BASE).length.sum() == pytest.approx(0.0)
    assert removed == {"Base": {"edges_cut": 1, "edges_removed": 0, "km": 0.0}}


def test_street_through_a_corner_loses_the_corner_and_no_longer_connects_through():
    """An edge passing through a narrow part of the area must not stay walkable across it."""
    e = split_into_edges(_ways([_street(1, [1, 2], [(-10, 20), (20, -10)])]))  # inside the base from (0, 10) to (10, 0)
    e2 = split_into_edges(_ways([_street(2, [3, 4], [(-20, 50), (50, 50)])]))  # half inside: kept part outside
    e = pd.concat([e, e2], ignore_index=True)
    kept, fence, _ = clip_restricted(e, _base())
    assert kept.geometry.intersection(BASE.buffer(-0.01)).length.sum() == pytest.approx(0.0)
    way1 = kept[kept["osm_way_id"] == 1]
    assert len(way1) == 2  # the two sides of the corner
    ends = [{int(r.u), int(r.v)} for r in way1.itertuples()]
    assert {n for n in ends[0] | ends[1] if n > 0} == {1, 2} and len((ends[0] | ends[1]) & fence) == 2
    assert not (ends[0] & ends[1])  # the two pieces share no node: nothing joins them across the corner
    assert len(fence) == 3


def test_edges_wholly_inside_are_removed_and_the_fence_street_is_untouched():
    e = split_into_edges(_ways([
        _street(1, [1, 2], [(10, 50), (90, 50)]),   # behind the gate
        _street(2, [3, 4], [(-5, 0), (-5, 100)]),   # public road along the fence
        _street(4, [7, 8], [(0, 0), (0, 100)]),     # drawn on the fence line itself
    ]))
    kept, fence, removed = clip_restricted(e, _base())
    assert sorted(kept["osm_way_id"]) == [2, 4]
    assert fence == set()
    assert removed == {"Base": {"edges_cut": 0, "edges_removed": 1, "km": 0.1}}


def test_no_restricted_areas_keeps_everything():
    e = split_into_edges(_ways([{"osm_way_id": 1, "kind": "sidewalk", "representation": "pedestrian_way",
                                 "node_refs": [1, 2], "coords": [(0, 0), (10, 0)]}]))
    empty = gpd.GeoDataFrame({"name": []}, geometry=[], crs=ANALYSIS_CRS)
    kept, fence, removed = clip_restricted(e, empty)
    assert len(kept) == 1 and fence == set() and removed == {}


def test_bridging_never_rejoins_ends_cut_at_the_fence():
    """Two footways cut at the fence 2 m apart: a mapping-slip connector between their ends would
    run along the fence and invent a link. Cut ends are real dead ends."""
    e = split_into_edges(_ways([
        {"osm_way_id": 1, "kind": "footway", "representation": "pedestrian_way", "node_refs": [1, 2], "coords": [(-20, 50), (30, 50)]},
        {"osm_way_id": 2, "kind": "footway", "representation": "pedestrian_way", "node_refs": [3, 5, 4], "coords": [(-30, 90), (-2, 52), (30, 52)]},
    ]))
    kept, fence, _ = clip_restricted(e, _base())
    kept["edge_id"] = np.arange(len(kept))
    # without the rule the two cut ends, 2 m apart on the fence, would be joined
    assert bridge_gaps(kept, node_table(kept), forbidden=BASE.buffer(-0.5))[1] == 1
    assert bridge_gaps(kept, node_table(kept), skip_nodes=frozenset(fence), forbidden=BASE.buffer(-0.5))[1] == 0


def test_bridging_never_enters_a_closed_area():
    e = split_into_edges(_ways([
        {"osm_way_id": 1, "kind": "footway", "representation": "pedestrian_way", "node_refs": [1, 2], "coords": [(50, 120), (50, 101)]},
        {"osm_way_id": 2, "kind": "footway", "representation": "pedestrian_way", "node_refs": [3, 4], "coords": [(40, 99), (60, 99)]},
    ]))
    e["edge_id"] = np.arange(len(e))
    assert bridge_gaps(e, node_table(e))[1] == 1  # 2 m apart: an ordinary mapping slip
    assert bridge_gaps(e, node_table(e), forbidden=BASE.buffer(-0.5))[1] == 0  # but the target is behind the fence
