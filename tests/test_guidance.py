"""On-screen guidance: steps in walking order, and never more certain than the data."""

import numpy as np

from pathfinder.routing.cost import Preferences, Requirements
from pathfinder.routing.guidance import _inventory_style, build_steps

from .test_service import _lonlat, planner  # noqa: F401  (fixture)

UNKNOWN = {"value": None, "status": "unknown", "source": None, "as_of": None}


def _row(**kw) -> dict:
    row = {"kind": "sidewalk", "name": None, "along_street": None, "osm_way_id": 1, "crossed_street": None, "sidewalk_presence": None,
           "sidewalk_presence_status": "unknown", "sidewalk_presence_source": None, "sidewalk_presence_as_of": None,
           "step_free_as_of": None, "xing_u": False, "xing_v": False,
           "xing_u_control": None, "xing_v_control": None, "xing_u_kerb": None, "xing_v_kerb": None,
           "xing_u_kerb_source": None, "xing_v_kerb_source": None, "xing_u_kerb_as_of": None, "xing_v_kerb_as_of": None}
    for attr in ("crossing_control", "kerb_a", "kerb_b"):
        row.update({attr: None, f"{attr}_status": "unknown", f"{attr}_source": None, f"{attr}_as_of": None})
    return {**row, **kw}


def _line(*pts) -> np.ndarray:
    return np.array(pts, dtype=float)


def _walk(parts, rows, degree, forward=None):
    """Steps for edges walked with no partial walkway at either end."""
    start, end = parts[0][:1], parts[-1][-1:]
    return build_steps([start, *parts, end], [None, *rows, None], [True, *(forward or [True] * len(parts)), True], [0, *degree, 0])


def test_a_turn_is_announced_only_where_there_is_a_choice_of_ways():
    parts = [_line((0, 0), (0, 100)), _line((0, 100), (-100, 100))]  # north, then west: a left turn
    at_a_junction = _walk(parts, [_row(), _row(name="K Street Northwest")], [3])
    assert [s["type"] for s in at_a_junction] == ["start", "turn", "arrive"]
    assert at_a_junction[0]["heading"] == "north"
    assert at_a_junction[1] == {"type": "turn", "vertex": 2, "direction": "left", "onto": "K ST NW"}
    a_bend_in_a_path = _walk(parts, [_row(), _row()], [2])
    assert [s["type"] for s in a_bend_in_a_path] == ["start", "arrive"]


def test_a_turn_names_the_street_a_sidewalk_runs_beside_unless_it_is_the_one_already_walked():
    # a sidewalk beside K ST, a nameless stub at the corner, then a sidewalk beside 9TH ST
    parts = [_line((0, 0), (0, 100)), _line((0, 100), (-2, 102)), _line((-2, 102), (-100, 102))]
    rows = [_row(along_street="K ST NW"), _row(), _row(along_street="9TH ST NW")]
    turn = next(s for s in _walk(parts, rows, [3, 3]) if s["type"] == "turn")
    assert turn["onto"] == "9TH ST NW"
    rows[2] = _row(along_street="K ST NW")  # the street bends: the same street is not "onto" anything
    assert "onto" not in next(s for s in _walk(parts, rows, [3, 3]) if s["type"] == "turn")


def test_osm_street_names_are_written_as_the_inventory_writes_them():
    assert _inventory_style("C Street Southeast") == "C ST SE"
    assert _inventory_style("North Capitol Street Northwest") == "NORTH CAPITOL ST NW"
    assert _inventory_style("20th Street Northwest Cycletrack") == "20th St NW Cycletrack"
    assert _inventory_style("Minnesota Avenue Southeast Cycleway") == "Minnesota Ave SE Cycleway"
    assert _inventory_style("Blagden Alley Northwest") == "BLAGDEN ALLEY NW"
    for name in ("Rock Creek Trail", "Elm Walk", "Entry Road", "Northwest Branch Trail"):
        assert _inventory_style(name) == name
    # so a sidewalk beside a street is the same way as the street, and no turn names it
    parts = [_line((0, 0), (0, 100)), _line((0, 100), (-2, 102)), _line((-2, 102), (-100, 102))]
    rows = [_row(kind="street_unknown", name="K Street Northwest"), _row(), _row(along_street="K ST NW")]
    assert "onto" not in next(s for s in _walk(parts, rows, [3, 3]) if s["type"] == "turn")


def test_right_slight_and_straight_are_told_apart():
    north = _line((0, 0), (0, 100))
    kinds = lambda to: [s.get("direction") for s in _walk([north, _line((0, 100), to)], [_row(), _row()], [4]) if s["type"] == "turn"]  # noqa: E731
    assert kinds((100, 100)) == ["right"]
    assert kinds((60, 200)) == ["slight_right"]
    assert kinds((10, 200)) == []  # straight on: nothing to say
    assert kinds((-20, 0)) == ["sharp_left"]


def test_a_corner_mapped_as_several_short_pieces_is_one_turn():
    parts = [_line((0, 0), (0, 100)), _line((0, 100), (3, 103)), _line((3, 103), (100, 103))]
    turns = [s for s in _walk(parts, [_row(), _row(), _row()], [3, 3]) if s["type"] == "turn"]
    assert len(turns) == 1 and turns[0]["direction"] == "right"


def test_a_crossing_carries_its_claims_and_unknown_stays_unknown():
    crossing = _row(kind="crossing", crossed_street="TEST AVE", crossing_control="signal", crossing_control_status="verified",
                    crossing_control_source="osm_dc", crossing_control_as_of="2026-10-04", osm_way_id=7)
    parts = [_line((0, 0), (100, 0)), _line((100, 0), (108, 0)), _line((108, 0), (115, 0)), _line((115, 0), (200, 0))]
    steps = _walk(parts, [_row(), crossing, dict(crossing), _row()], [2, 2, 2])
    cross = [s for s in steps if s["type"] == "cross"]
    assert len(cross) == 1  # one crossing, though mapped as two edges
    assert cross[0]["street"] == "TEST AVE" and cross[0]["length_m"] == 15 and cross[0]["vertex"] == 3
    assert cross[0]["control"] == {"value": "signal", "status": "verified", "source": "osm_dc", "as_of": "2026-10-04"}
    assert cross[0]["curb_ramps"] == [UNKNOWN, UNKNOWN]  # no curb evidence: said, not assumed


def test_unmapped_crossings_are_steps_too_and_take_the_worse_half():
    # two street-centerline edges meeting at a junction, each flagged at that end; one half has a signal on record
    a = _row(kind="street_unknown", xing_v=True, xing_v_control="signal", xing_v_kerb="ramp_likely",
             xing_v_kerb_source="frederick_ramps", xing_v_kerb_as_of="2025-04-02")
    b = _row(kind="street_unknown", xing_v=True, xing_v_control=None, xing_v_kerb="ramp_likely",
             xing_v_kerb_source="frederick_ramps", xing_v_kerb_as_of="2025-04-02")
    parts = [_line((0, 0), (100, 0)), _line((100, 0), (200, 0))]  # b is walked from its v end
    steps = [s for s in _walk(parts, [a, b], [2], forward=[True, False]) if s["type"] == "unmapped_crossing"]
    assert len(steps) == 1 and steps[0]["vertex"] == 2
    assert steps[0]["control"] == UNKNOWN  # the halves disagree: not a signal
    assert steps[0]["curb_ramps"] == {"value": "ramp_likely", "status": "inferred", "source": "frederick_ramps", "as_of": "2025-04-02"}


def test_halves_that_agree_on_the_curb_keep_every_source_and_the_older_date():
    a = _row(kind="alley", xing_v=True, xing_v_kerb="no_curb", xing_v_kerb_source="osm_md", xing_v_kerb_as_of="2026-10-15")
    b = _row(kind="street_unknown", xing_v=True, xing_v_kerb="no_curb", xing_v_kerb_source="frederick_sidewalks",
             xing_v_kerb_as_of="2025-03-11")
    parts = [_line((0, 0), (100, 0)), _line((100, 0), (200, 0))]
    steps = [s for s in _walk(parts, [a, b], [2], forward=[True, False]) if s["type"] == "unmapped_crossing"]
    assert steps[0]["curb_ramps"] == {"value": "no_curb", "status": "inferred", "source": "frederick_sidewalks, osm_md",
                                      "as_of": "2025-03-11"}


def test_stairs_and_streets_without_a_sidewalk_are_named_before_they_are_reached():
    rows = [_row(), _row(kind="steps", step_free_as_of="2026-10-04"), _row(kind="street_no_sidewalk", sidewalk_presence="none",
            sidewalk_presence_status="verified", sidewalk_presence_source="sidewalks_2021", sidewalk_presence_as_of="2021")]
    parts = [_line((0, 0), (50, 0)), _line((50, 0), (60, 0)), _line((60, 0), (160, 0))]
    steps = _walk(parts, rows, [2, 2])
    assert [s["type"] for s in steps] == ["start", "steps", "no_sidewalk", "arrive"]
    assert steps[1]["length_m"] == 10 and steps[1]["status"] == "verified"
    assert steps[2] == {"type": "no_sidewalk", "vertex": 5, "length_m": 100, "status": "verified", "source": "sidewalks_2021", "as_of": "2021"}


def test_every_route_option_has_steps_tied_to_its_own_geometry(planner):  # noqa: F811
    out = planner.plan(_lonlat(0, 0), _lonlat(215, 0), Requirements(), Preferences(w_cross=1, w_slope=0, w_sidewalk=0, budget_s=600))
    assert len(out["route_options"]) > 1
    for o in out["route_options"]:
        steps = o["steps"]
        assert steps[0]["type"] == "start" and steps[-1] == {"type": "arrive", "vertex": len(o["geometry"]) - 1}
        assert [s["vertex"] for s in steps] == sorted(s["vertex"] for s in steps)
    straight = next(o for o in out["route_options"] if o["is_fastest"])
    cross = next(s for s in straight["steps"] if s["type"] == "cross")
    assert cross["street"] == "TEST AVE" and cross["control"]["value"] == "unmarked"
    assert {"value": None, "status": "unknown", "source": None, "as_of": None} in cross["curb_ramps"]
    around = next(o for o in out["route_options"] if not o["is_fastest"])
    assert not any(s["type"] == "cross" for s in around["steps"])


def test_a_trip_along_one_walkway_still_starts_and_arrives(planner):  # noqa: F811
    out = planner.plan(_lonlat(10, 0), _lonlat(60, 0))
    assert [s["type"] for s in out["route_options"][0]["steps"]] == ["start", "arrive"]
