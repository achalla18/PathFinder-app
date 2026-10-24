"""Explanations must agree with the routes they explain."""

import numpy as np
import pytest

from pathfinder.routing.cost import Preferences, Requirements
from pathfinder.routing.explain import (
    WORTH_SHOWING_MIN_GAIN_S,
    crux,
    felt_cost,
    felt_gain,
    minimal_relaxation,
    section_deltas,
    shown_route,
    tipping_point,
    worth_showing,
)
from pathfinder.routing.router import Network

from .conftest import make_edges, make_nodes


def _two_ways(bad_crossing: dict, detour_m: float = 60.0) -> Network:
    """0 → 3 either straight across a crossing (0-1-2-3) or around it (0-4-5-3)."""
    xy = {0: (0, 0), 1: (100, 0), 2: (115, 0), 3: (215, 0), 4: (0, detour_m), 5: (215, detour_m)}
    e = make_edges([
        {"u": 0, "v": 1}, {"u": 1, "v": 2, "kind": "crossing", **bad_crossing}, {"u": 2, "v": 3},
        {"u": 0, "v": 4}, {"u": 4, "v": 5}, {"u": 5, "v": 3},
    ], xy)
    return Network(e, make_nodes(xy))


def test_felt_gain_matches_what_the_router_minimized():
    net = _two_ways({"crossing_control": "unmarked", "crossed_speed_mph": 35, "kerb_a": "lowered", "kerb_b": "lowered"})
    pref = Preferences(w_cross=1, w_slope=0, w_sidewalk=0, budget_s=600)
    rs = net.route_set(0, 3, Requirements(), pref)
    assert rs.recommended.key != rs.fastest.key  # the detour avoids the crossing
    assert felt_cost(rs.recommended, pref) <= felt_cost(rs.fastest, pref)
    assert felt_gain(rs.recommended, rs.fastest, pref) > 0


def test_section_deltas_add_up_to_the_route_difference():
    net = _two_ways({"crossing_control": "unmarked", "crossed_speed_mph": 35, "kerb_a": "lowered", "kerb_b": "lowered"})
    pref = Preferences(w_cross=1, w_slope=0, w_sidewalk=0, budget_s=600)
    rs = net.route_set(0, 3, Requirements(), pref)
    deltas = section_deltas(net, rs.recommended, rs.fastest, pref)
    assert len(deltas) == 1
    assert sum(d.net_s for d in deltas) == pytest.approx(-felt_gain(rs.recommended, rs.fastest, pref))
    c = crux(net, rs.recommended, rs.fastest, pref)
    assert c.time_s > 0 and c.felt_s["cross"] < 0  # slower, but spares the crossing
    assert set(c.section.b_edges) == {0, 1, 2}


def test_trivial_improvements_are_not_worth_showing():
    net = _two_ways({"crossing_control": "signal", "signal_lpi": True, "kerb_a": "lowered", "kerb_b": "lowered"},
                    detour_m=1.0)
    pref = Preferences(budget_s=600)
    rs = net.route_set(0, 3, Requirements(), pref)
    for r in rs.routes:
        if r.key != rs.fastest.key:
            assert felt_gain(r, rs.fastest, pref) < WORTH_SHOWING_MIN_GAIN_S
            assert not worth_showing(r, rs.fastest, pref)
    assert not worth_showing(rs.fastest, rs.fastest, pref)


def test_tipping_point_names_the_nearest_priorities_with_a_different_winner():
    net = _two_ways({"crossing_control": "unmarked", "crossed_speed_mph": 35, "kerb_a": "lowered", "kerb_b": "lowered"})
    pref = Preferences(w_cross=0, w_slope=1, w_sidewalk=0, budget_s=600)  # crossings do not matter to this user
    rs = net.route_set(0, 3, Requirements(), pref)
    assert rs.recommended.key == rs.fastest.key
    tp = tipping_point(rs, pref)
    assert tp is not None and tp.route.key != rs.recommended.key
    assert tp.shift["cross"] > 0  # caring more about crossings is what flips it
    assert tp.distance == pytest.approx(tp.shift["cross"] / 100)
    assert np.isclose(sum(tp.shift.values()), 0.0)


def test_no_tipping_point_when_one_route_wins_everywhere():
    xy = {0: (0, 0), 1: (100, 0)}
    net = Network(make_edges([{"u": 0, "v": 1}], xy), make_nodes(xy))
    rs = net.route_set(0, 1)
    assert tipping_point(rs, Preferences()) is None


def test_minimal_relaxation_finds_the_smallest_compromise_and_where_it_bites():
    # the only way through has one crossing end with no curb evidence
    xy = {0: (0, 0), 1: (100, 0), 2: (115, 0), 3: (215, 0)}
    e = make_edges([{"u": 0, "v": 1}, {"u": 1, "v": 2, "kind": "crossing", "kerb_a": "lowered", "kerb_b": None},
                    {"u": 2, "v": 3}], xy)
    net = Network(e, make_nodes(xy))
    req = Requirements(step_free=True, max_grade_pct=8.33, certainty="verified_only")
    assert net.route_set(0, 3, req).status == "no_feasible_route"
    r = minimal_relaxation(net, 0, 3, req)
    assert r.requirements.certainty == "allow_unknown" and r.requirements.step_free
    assert r.requirements.max_grade_pct == 8.33  # nothing else was given up
    assert r.newly_used_edges == (1,)


def test_minimal_relaxation_never_passes_known_stairs_while_step_free_holds():
    xy = {0: (0, 0), 1: (100, 0)}
    e = make_edges([{"u": 0, "v": 1, "kind": "steps", "step_free": False}], xy)
    net = Network(e, make_nodes(xy))
    r = minimal_relaxation(net, 0, 1, Requirements(step_free=True, certainty="allow_likely"))
    assert r.label == "allow steps and unverified curbs" and not r.requirements.step_free
    assert r.newly_used_edges == (0,)


def test_tipping_point_is_measured_from_the_route_actually_shown():
    """Regression: the baseline was the optimizer's choice even when it was not worth showing,
    so "what would it take" could point at the route already on screen."""
    net = _two_ways({"crossing_control": "signal", "signal_lpi": True, "kerb_a": "lowered", "kerb_b": "lowered"},
                    detour_m=1.0)
    pref = Preferences(budget_s=600)
    rs = net.route_set(0, 3, Requirements(), pref)
    # the optimizer prefers the detour here, but by too little to show: the fastest is on screen
    assert rs.recommended.key != rs.fastest.key
    assert shown_route(rs.recommended, rs.fastest, pref).key == rs.fastest.key
    tp = tipping_point(rs, pref)
    # the tipping point is where the detour becomes worth showing, not where the optimizer
    # happens to switch, and never the route already on screen
    assert tp.route.key != rs.fastest.key
    there = Preferences(w_cross=tp.weights[0], w_slope=tp.weights[1], w_sidewalk=tp.weights[2], budget_s=600)
    assert felt_gain(tp.route, rs.fastest, there) >= WORTH_SHOWING_MIN_GAIN_S
    assert tp.shift["cross"] > 0


def test_tipping_point_never_returns_the_shown_route():
    net = _two_ways({"crossing_control": "unmarked", "crossed_speed_mph": 35, "kerb_a": "lowered", "kerb_b": "lowered"})
    for w in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)):
        pref = Preferences(w_cross=w[0], w_slope=w[1], w_sidewalk=w[2], budget_s=600)
        rs = net.route_set(0, 3, Requirements(), pref)
        tp = tipping_point(rs, pref)
        if tp is not None:
            assert tp.route.key != shown_route(rs.recommended, rs.fastest, pref).key
