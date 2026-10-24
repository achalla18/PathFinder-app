"""The trip planner's answer: sourced, dated, and never more certain than the data."""

import json

import pytest
from pyproj import Transformer

from pathfinder.config import ANALYSIS_CRS, WGS84
from pathfinder.routing.cost import Preferences, Requirements
from pathfinder.routing.service import SNAP_MAX_M, TripPlanner

from .conftest import make_edges, make_nodes

X0, Y0 = 397000.0, 137000.0  # central DC in EPSG:26985
TO_LONLAT = Transformer.from_crs(ANALYSIS_CRS, WGS84, always_xy=True)


def _lonlat(x: float, y: float) -> tuple[float, float]:
    return TO_LONLAT.transform(X0 + x, Y0 + y)


@pytest.fixture
def planner(tmp_path):
    """0 → 3 straight across an unmarked crossing with one unknown curb, or around it."""
    xy = {0: (0, 0), 1: (100, 0), 2: (115, 0), 3: (215, 0), 4: (0, 60), 5: (215, 60)}
    xy = {k: (X0 + x, Y0 + y) for k, (x, y) in xy.items()}
    e = make_edges([
        {"u": 0, "v": 1},
        {"u": 1, "v": 2, "kind": "crossing", "crossing_control": "unmarked", "crossed_speed_mph": 35,
         "kerb_a": "lowered", "kerb_b": None},
        {"u": 2, "v": 3}, {"u": 0, "v": 4}, {"u": 4, "v": 5}, {"u": 5, "v": 3},
    ], xy)
    e["crossing_control_source"], e["crossing_control_as_of"] = "osm_dc", "2026-10-04"
    for col in ("name", "along_street", "crossed_street", "crossed_lanes"):
        e[col] = None
    e["reported_person_crimes"], e["reported_person_crimes_night"] = 0, 0
    e.loc[[0, 1, 2], "reported_person_crimes"] = 9  # the straight way runs along a block with reports
    e.loc[[0, 1, 2], "reported_person_crimes_night"] = 6
    e.loc[[0, 1, 2], "reported_crime_rate_day"] = 1.0
    e.loc[[0, 1, 2], "reported_crime_rate_night"] = 1.6
    e.loc[1, "crossed_street"] = "TEST AVE"
    e.to_parquet(tmp_path / "edges.parquet")
    make_nodes(xy).to_parquet(tmp_path / "nodes.parquet")
    return TripPlanner(tmp_path)


def test_plan_explains_the_recommendation_with_sourced_conditions(planner):
    pref = Preferences(w_cross=1, w_slope=0, w_sidewalk=0, budget_s=600)
    out = planner.plan(_lonlat(0, 0), _lonlat(215, 0), Requirements(), pref)
    json.dumps(out)  # the whole answer must be JSON-serializable as returned
    assert out["status"] == "ok" and not out["recommended_is_fastest"]
    assert out["recommended"]["minutes"] > out["fastest"]["minutes"]
    cmp = out["comparison"]
    assert cmp["places_that_differ"] == 1 and cmp["felt_minutes_saved"] > 0
    crossing = next(c for c in cmp["crux"]["fastest_route"]["conditions"] if c["type"] == "crossing")
    assert crossing["street"] == "TEST AVE"
    assert crossing["control"] == {"value": "unmarked", "status": "verified", "source": "osm_dc", "as_of": "2026-10-04"}
    # the curb with no evidence is reported as unknown, not left out
    assert {"value": None, "status": "unknown", "source": None, "as_of": None} in crossing["curb_ramps"]
    assert not any(c["type"] == "crossing" for c in cmp["crux"]["this_route"]["conditions"])
    # geometry runs from origin to destination
    line = out["recommended"]["geometry"]
    assert line[0] == pytest.approx(list(_lonlat(0, 0)), abs=1e-5) and line[-1] == pytest.approx(list(_lonlat(215, 0)), abs=1e-5)
    assert "safe" not in json.dumps(out).lower()


def test_plan_offers_the_smallest_compromise_when_requirements_cannot_be_met(planner):
    # step-free with likely evidence: the crossing's unknown curb blocks the straight way,
    # and steepening the detour blocks that too
    planner.net.edges.loc[4, "grade_max_pct"] = 9.0
    planner.net._feasible_cache.clear()
    req = Requirements(step_free=True, max_grade_pct=8.33, certainty="allow_likely")
    out = planner.plan(_lonlat(0, 0), _lonlat(215, 0), req)
    assert out["status"] == "no_feasible_route"
    rel = out["relaxation"]
    assert rel["compromise"] == "accept places with no evidence"
    assert rel["requirements"]["step_free"] and rel["requirements"]["max_grade_pct"] == 8.33
    assert rel["edges_outside_original_requirements"] == 1
    assert any(c["type"] == "crossing" for c in rel["where_it_applies"])


def test_plan_refuses_points_far_from_the_network(planner):
    out = planner.plan(_lonlat(0, 0), _lonlat(215, SNAP_MAX_M + 200))
    assert out["status"] == "not_near_network" and "recommended" not in out


def test_a_point_in_the_middle_of_a_long_walkway_is_on_the_network(tmp_path):
    """Regression: endpoints snapped to nodes only, so the middle of a long edge was refused
    and the walk to the node was left out of the trip."""
    xy = {0: (X0, Y0), 1: (X0 + 400, Y0), 2: (X0 + 400, Y0 + 100)}
    e = make_edges([{"u": 0, "v": 1}, {"u": 1, "v": 2}], xy)
    for col in ("name", "along_street", "crossed_street", "crossed_lanes"):
        e[col] = None
    e["reported_person_crimes"], e["reported_person_crimes_night"] = 0, 0
    e.to_parquet(tmp_path / "edges.parquet")
    make_nodes(xy).to_parquet(tmp_path / "nodes.parquet")
    planner = TripPlanner(tmp_path)

    out = planner.plan(_lonlat(200, 5), _lonlat(400, 100))  # 200 m from either node of the first edge
    assert out["status"] == "ok"
    assert out["origin"]["snap_distance_m"] == pytest.approx(5.0, abs=0.1)
    assert out["fastest"]["length_m"] == 300  # 200 m of the first edge + the whole second edge
    assert out["fastest"]["minutes"] == pytest.approx(300 / 1.2 / 60, abs=0.05)
    assert out["fastest"]["geometry"][0] == pytest.approx(list(_lonlat(200, 0)), abs=1e-5)

    same = planner.plan(_lonlat(100, 0), _lonlat(350, 0))  # both on the first edge
    assert same["status"] == "ok" and same["fastest"]["length_m"] == 250
    assert same["fastest"]["geometry"][-1] == pytest.approx(list(_lonlat(350, 0)), abs=1e-5)
    # regression: short trips came back without the option list the web page draws from
    for short in (same, planner.plan(_lonlat(390, 0), _lonlat(400, 10))):  # same edge; either side of one node
        assert short["status"] == "ok"
        (only,) = short["route_options"]
        assert only["is_fastest"] and only["is_recommended"] and only["geometry"] == short["fastest"]["geometry"]
        assert len(short["triangle"]) == 66 and {t[3] for t in short["triangle"]} == {0}


def test_endpoints_avoid_walkways_that_break_the_requirements(planner):
    # the destination is on the crossing with an unknown curb; step-free trips are placed on
    # the nearest walkway that meets the requirement instead, and the response says so
    req = Requirements(step_free=True, certainty="allow_likely")
    out = planner.plan(_lonlat(0, 0), _lonlat(107, 0), req)
    assert out["status"] == "ok"
    assert out["destination"]["walkway_meets_requirements"] and out["destination"]["nearer_walkway_passed_over"]
    plain = planner.plan(_lonlat(0, 0), _lonlat(107, 0))
    assert not plain["destination"]["nearer_walkway_passed_over"]


def test_each_distinct_place_is_described_once_and_shared_between_options(planner):
    """Speed: options that leave the fastest route at the same place share one description."""
    out = planner.plan(_lonlat(0, 0), _lonlat(215, 0), Requirements(), Preferences(w_cross=1, w_slope=0, w_sidewalk=0, budget_s=600))
    detour = next(o for o in out["route_options"] if not o["is_fastest"])
    assert out["comparison"] is detour["compared_with_fastest"]
    assert planner.evidence([]) == {}


def test_a_plan_says_nothing_of_crime_or_darkness_and_certifies_nothing(planner):
    """PathFinder is an accessibility app: no crime, after-dark or lighting fields, and the
    disclaimer says it certifies no route as accessible (rule 4)."""
    out = planner.plan(_lonlat(0, 0), _lonlat(215, 0), Requirements(), Preferences(budget_s=600))
    text = json.dumps(out).lower()
    for word in ("crime", "night", "dark", "light"):
        assert word not in text, word
    assert set(out["priorities"]) == {"cross", "slope", "sidewalk"}
    assert set(out["fastest"]["felt_minutes"]) == {"cross", "slope", "sidewalk"}
    assert "does not certify" in out["disclaimer"]


def test_a_trip_end_inside_a_closed_area_is_named_and_routed_only_to_its_fence(tmp_path):
    """Fort Detrick: a destination behind the gate snaps to the nearest public walkway outside
    (its walkways were removed at build time), and the answer says why the route stops there."""
    import geopandas as gpd
    from shapely.geometry import box

    graph, interim = tmp_path / "graph", tmp_path / "interim"
    graph.mkdir(); interim.mkdir()
    xy = {k: (X0 + x, Y0 + y) for k, (x, y) in {0: (0, 0), 1: (215, 0)}.items()}
    e = make_edges([{"u": 0, "v": 1}], xy)
    for col in ("name", "along_street", "crossed_street", "crossed_lanes"):
        e[col] = None
    e.to_parquet(graph / "edges.parquet")
    make_nodes(xy).to_parquet(graph / "nodes.parquet")
    gpd.GeoDataFrame({"name": ["Base"]}, geometry=[box(X0 + 230, Y0 - 50, X0 + 600, Y0 + 50)], crs=ANALYSIS_CRS) \
        .to_crs(WGS84).to_parquet(interim / "osm_restricted_areas.parquet")
    planner = TripPlanner(graph)

    near_fence = planner.plan(_lonlat(0, 0), _lonlat(260, 10))
    assert near_fence["status"] == "ok"
    assert near_fence["destination"]["inside_closed_area"] == "Base"
    assert "inside_closed_area" not in near_fence["origin"]
    deep = planner.plan(_lonlat(0, 0), _lonlat(560, 0))
    assert deep["status"] == "not_near_network" and "Base, which is not open to the public" in deep["message"]
