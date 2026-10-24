"""The barrier map: recorded barriers and unrecorded places, from the same graph as routing."""

from .conftest import make_edges
from .test_api import _body, client  # noqa: F401  (fixture)
from .test_service import _lonlat, planner  # noqa: F401  (fixture)

XY = {0: (0, 0), 1: (12, 0), 2: (100, 0), 3: (200, 0), 4: (300, 0)}


def _edges():
    from pathfinder.analysis.barriers import barriers

    e = make_edges([
        {"u": 0, "v": 1, "kind": "crossing", "kerb_a": "raised", "kerb_a_source": "osm_test", "kerb_b": None, "crossed_street": "MAIN ST"},
        {"u": 1, "v": 2, "kind": "street_unknown", "representation": "street_centerline", "sidewalk_presence": "none",
         "name": "OAK ST", "xing_v": True, "xing_v_kerb": "unknown"},
        {"u": 2, "v": 3, "kind": "street_unknown", "representation": "street_centerline", "name": "ELM ST"},
        {"u": 3, "v": 4, "kind": "sidewalk", "grade_max_pct": 11.0},
    ], XY)
    for c in ("crossed_street", "name", "along_street"):
        if c not in e:
            e[c] = None
    return e, barriers(e)


def test_barriers_and_unknowns_are_both_listed_with_their_evidence():
    _, b = _edges()
    got = sorted(zip(b["type"], b["kind"]))
    assert got == sorted([("curb_barrier", "barrier"), ("curb_unknown", "unknown"), ("no_sidewalk", "barrier"),
                          ("junction_curb_unknown", "unknown"), ("sidewalk_unknown", "unknown"), ("steep", "barrier")])
    curb = b[b["type"] == "curb_barrier"].iloc[0]
    assert curb["status"] == "verified" and curb["source"] == "osm_test" and curb.geometry.geom_type == "Point"
    assert b[b["type"] == "no_sidewalk"].iloc[0]["street"] == "OAK ST"


def test_a_place_beyond_the_region_line_is_left_out():
    from pathfinder.analysis.barriers import barriers

    e, _ = _edges()
    e["outside_region"] = True
    assert barriers(e).empty


def test_barrier_api_filters_by_bbox_and_rejects_a_bad_one(client):  # noqa: F811
    o, d = _lonlat(-50, -50), _lonlat(400, 50)
    r = client.get("/api/barriers", params={"bbox": f"{o[0]},{o[1]},{d[0]},{d[1]}"})
    assert r.status_code == 200
    out = r.json()
    assert out["type"] == "FeatureCollection" and "truncated" in out and "unlicensed_sources" in out
    assert {"features", "count", "truncated"} <= set(out["sidewalks"])
    assert sum(out["counts"].values()) >= len(out["features"])
    capped = client.get("/api/barriers", params={"bbox": f"{o[0]},{o[1]},{d[0]},{d[1]}", "limit": 1}).json()
    assert len(capped["features"]) <= 1 and capped["counts"] == out["counts"]
    for f in out["features"]:
        assert {"type", "kind", "status", "source", "as_of", "what"} <= set(f["properties"])
    assert client.get("/api/barriers", params={"bbox": "1,1,0,0"}).status_code == 422
    assert client.get("/api/barriers", params={"bbox": "nope"}).status_code == 422


def test_a_failing_ramp_and_a_missing_tactile_pad_are_barriers_with_the_survey_as_source():
    from pathfinder.analysis.barriers import barriers

    e = make_edges([{"u": 0, "v": 1, "kind": "crossing", "kerb_a": "ramp_present", "kerb_b": "ramp_present",
                     "ramp_check_a": "fails", "ramp_check_a_source": "frederick_ramps", "ramp_check_a_as_of": "2025-04-03",
                     "ramp_check_b": "no_tactile_pad", "crossed_street": "MARKET ST"}], XY)
    for c in ("name", "along_street"):
        e[c] = None
    b = barriers(e)
    assert sorted(b["type"]) == ["ramp_fails_check", "ramp_no_tactile_pad"]
    f = b[b["type"] == "ramp_fails_check"].iloc[0]
    assert f["source"] == "frederick_ramps" and f["as_of"] == "2025-04-03" and f["kind"] == "barrier"


def test_a_sidewalk_in_the_lowest_condition_band_or_too_narrow_is_listed_with_its_survey():
    from pathfinder.analysis.barriers import CONDITION_BARRIER_BELOW, barriers

    e = make_edges([{"u": 0, "v": 1, "sidewalk_condition": CONDITION_BARRIER_BELOW - 1, "sidewalk_condition_source": "frederick_sidewalks",
                     "sidewalk_condition_as_of": "2025-04-02", "sidewalk_width_m": 0.9},
                    {"u": 1, "v": 2, "sidewalk_condition": CONDITION_BARRIER_BELOW + 30, "sidewalk_width_m": 1.22}], XY)
    for c in ("name", "along_street", "crossed_street"):
        e[c] = None
    b = barriers(e)
    assert sorted(b["type"]) == ["sidewalk_narrow", "sidewalk_poor_condition"]
    f = b[b["type"] == "sidewalk_poor_condition"].iloc[0]
    assert f["source"] == "frederick_sidewalks" and f["as_of"] == "2025-04-02" and f["edge_id"] == 0
