"""The 'missing data is never favorable' rule, expressed as penalty orderings."""

import numpy as np
import pytest

from pathfinder.routing.cost import (
    Preferences,
    Requirements,
    PENALTY_DIMS,
    crossing_penalty,
    edge_penalties,
    feasible_mask,
    sidewalk_penalty,
    slope_penalty,
)

from .conftest import make_edges

XY = {0: (0, 0), 1: (15, 0)}


def _one(**kw):
    return make_edges([{"u": 0, "v": 1, **kw}], XY)


def test_unknown_crossing_control_penalized_at_least_like_unmarked():
    unk = crossing_penalty(_one(kind="crossing"))[0]
    unmarked = crossing_penalty(_one(kind="crossing", crossing_control="unmarked"))[0]
    marked = crossing_penalty(_one(kind="crossing", crossing_control="marked_uncontrolled"))[0]
    signal = crossing_penalty(_one(kind="crossing", crossing_control="signal"))[0]
    assert unk >= unmarked > marked > signal > 0


def test_lpi_and_speed_move_crossing_penalty_in_the_right_direction():
    base = crossing_penalty(_one(kind="crossing", crossing_control="signal", signal_lpi=False))[0]
    lpi = crossing_penalty(_one(kind="crossing", crossing_control="signal", signal_lpi=True))[0]
    assert lpi < base
    slow = crossing_penalty(_one(kind="crossing", crossing_control="marked_uncontrolled", crossed_speed_mph=20))[0]
    fast = crossing_penalty(_one(kind="crossing", crossing_control="marked_uncontrolled", crossed_speed_mph=35))[0]
    assert fast > slow


def test_crossing_penalty_counts_once_across_fragments():
    xy = {0: (0, 0), 1: (6, 0), 2: (15, 0)}
    split = make_edges([{"u": 0, "v": 1, "kind": "crossing", "osm_way_id": 7, "crossing_control": "unmarked"},
                        {"u": 1, "v": 2, "kind": "crossing", "osm_way_id": 7, "crossing_control": "unmarked"}], xy)
    whole = make_edges([{"u": 0, "v": 2, "kind": "crossing", "osm_way_id": 7, "crossing_control": "unmarked"}],
                       {0: (0, 0), 2: (15, 0)})
    assert crossing_penalty(split).sum() == pytest.approx(crossing_penalty(whole).sum())


def test_unknown_kerb_between_passable_and_barrier():
    """Curb ramps are part of crossing a street."""
    ok = crossing_penalty(_one(kind="crossing", kerb_a="lowered", kerb_b="ramp_present"))[0]
    unk = crossing_penalty(_one(kind="crossing", kerb_a="lowered"))[0]
    bad = crossing_penalty(_one(kind="crossing", kerb_a="lowered", kerb_b="no_ramp"))[0]
    assert ok < unk < bad


def test_unknown_grade_is_penalized():
    assert slope_penalty(_one(grade_max_pct=None))[0] > slope_penalty(_one(grade_max_pct=2.0))[0]
    assert slope_penalty(_one(grade_max_pct=10.0))[0] > slope_penalty(_one(grade_max_pct=6.0))[0]


def test_the_three_dimensions_are_crossings_slope_and_sidewalk():
    assert PENALTY_DIMS == ("cross", "slope", "sidewalk")
    assert list(edge_penalties(_one())) == ["P_cross", "P_slope", "P_sidewalk"]


def test_walking_in_the_roadway_costs_most_and_unknown_sits_between():
    street = dict(kind="street_unknown", representation="street_centerline")
    pen = lambda v: sidewalk_penalty(_one(**street, sidewalk_presence=v))[0]  # noqa: E731
    assert pen("both") < pen("one_side") < pen(None) < pen("none")
    # a mapped sidewalk in good condition carries no presence penalty at all
    assert sidewalk_penalty(_one(kind="sidewalk", surface_class="smooth", sidewalk_condition=85.0))[0] == 0


def test_steps_count_under_sidewalk_not_slope():
    steps = _one(kind="steps", step_free=False)
    assert sidewalk_penalty(steps)[0] > 0 and slope_penalty(steps)[0] == 0


def test_step_free_requirement():
    xy = {0: (0, 0), 1: (10, 0), 2: (20, 0), 3: (30, 0)}
    e = make_edges([
        {"u": 0, "v": 1, "kind": "steps", "step_free": False},
        {"u": 1, "v": 2, "kind": "crossing", "kerb_a": "lowered", "kerb_b": None},
        {"u": 2, "v": 3, "kind": "crossing", "kerb_a": "lowered", "kerb_b": "flush"},
        {"u": 0, "v": 3, "kind": "footway", "step_free": True, "step_free_status": "inferred"},
    ], xy)
    likely = feasible_mask(e, Requirements(step_free=True, certainty="allow_likely"))
    assert likely.tolist() == [False, False, True, True]
    verified = feasible_mask(e, Requirements(step_free=True, certainty="verified_only"))
    assert verified.tolist() == [False, False, True, False]
    anything = feasible_mask(e, Requirements(step_free=True, certainty="allow_unknown"))
    assert anything.tolist() == [False, True, True, True]  # known stairs never pass


def test_max_grade_requirement():
    xy = {0: (0, 0), 1: (10, 0)}
    steep = _one(grade_max_pct=12.0)
    assert not feasible_mask(steep, Requirements(max_grade_pct=8.33))[0]
    suspect = _one(grade_max_pct=30.0, grade_max_pct_status="inferred")
    assert not feasible_mask(suspect, Requirements(max_grade_pct=8.33, certainty="allow_unknown"))[0]
    unknown = make_edges([{"u": 0, "v": 1, "grade_max_pct": None}], xy)
    assert not feasible_mask(unknown, Requirements(max_grade_pct=8.33, certainty="allow_likely"))[0]
    assert feasible_mask(unknown, Requirements(max_grade_pct=8.33, certainty="allow_unknown"))[0]


def test_preferences_normalize_weights():
    p = Preferences(w_cross=2, w_slope=1, w_sidewalk=1)
    assert np.allclose(p.weights, (0.5, 0.25, 0.25))
    assert np.allclose(p.multipliers(), (1.5, 0.75, 0.75))  # equal weights would give 1, 1, 1
    assert np.allclose(p.at((0, 1, 0)).weights, (0, 1, 0))
    with pytest.raises(ValueError):
        Preferences(w_cross=0, w_slope=0, w_sidewalk=0)


def test_implicit_crossing_costs_half_a_crossing_per_flagged_end():
    """Walking a street centerline through a junction must not be a free crossing."""
    plain = _one(kind="street_unknown", representation="street_centerline")
    one_end = _one(kind="street_unknown", representation="street_centerline", xing_u=True)
    both = _one(kind="street_unknown", representation="street_centerline", xing_u=True, xing_v=True)
    base = crossing_penalty(plain)[0]
    half = crossing_penalty(one_end)[0] - base
    assert half > 0
    assert crossing_penalty(both)[0] - base == pytest.approx(2 * half)
    # two halves cost what a mapped crossing of unknown control and the default width costs
    mapped = crossing_penalty(_one(kind="crossing", crossed_width_m=10.0))[0]
    assert 2 * half == pytest.approx(mapped)
    signal = _one(kind="street_unknown", representation="street_centerline", xing_u=True, xing_u_control="signal")
    assert 0 < crossing_penalty(signal)[0] - base < half


def test_implicit_crossing_kerbs_follow_the_certainty_setting():
    def mask(kerb, certainty):
        e = _one(kind="street_unknown", representation="street_centerline", xing_v=True, xing_v_kerb=kerb)
        return bool(feasible_mask(e, Requirements(step_free=True, certainty=certainty))[0])

    # inferred by construction: never "verified only"
    assert not any(mask(k, "verified_only") for k in ("no_curb", "ramp_likely", "unknown", "no_ramp_reported"))
    assert mask("ramp_likely", "allow_likely") and mask("no_curb", "allow_likely")
    assert not mask("unknown", "allow_likely") and mask("unknown", "allow_unknown")
    assert not mask("no_ramp_reported", "allow_unknown")
    assert not mask(None, "allow_likely")  # a flagged end with no kerb evidence is unknown
    # unknown curbs are penalized between passable and barrier, as on mapped crossings
    pen = lambda k: crossing_penalty(_one(kind="street_unknown", xing_v=True, xing_v_kerb=k))[0]  # noqa: E731
    assert pen("ramp_likely") < pen("unknown") < pen("no_ramp_reported")


def test_inferred_kerb_does_not_pass_verified_only():
    e = _one(kind="crossing", kerb_a="lowered", kerb_b="lowered", kerb_b_status="inferred")
    assert not feasible_mask(e, Requirements(step_free=True, certainty="verified_only"))[0]
    assert feasible_mask(e, Requirements(step_free=True, certainty="allow_likely"))[0]


def test_a_ramp_that_fails_its_survey_costs_more_but_stays_usable():
    ok = _one(kind="crossing", kerb_a="ramp_present", kerb_b="ramp_present", ramp_check_a="passed")
    fails = _one(kind="crossing", kerb_a="ramp_present", kerb_b="ramp_present", ramp_check_a="fails")
    assert crossing_penalty(fails)[0] > crossing_penalty(ok)[0]
    assert feasible_mask(fails, Requirements(step_free=True, certainty="verified_only"))[0]


def test_sidewalk_condition_costs_by_band_and_unknown_is_not_free():
    from pathfinder.routing.cost import sidewalk_penalty

    good, poor, unknown = (sidewalk_penalty(_one(sidewalk_condition=v))[0] for v in (85.0, 20.0, None))
    assert good < unknown < poor
    # a path or a crossing is not charged for having no sidewalk survey
    assert sidewalk_penalty(_one(kind="path"))[0] == sidewalk_penalty(_one(kind="path", sidewalk_condition=20.0))[0]


def test_a_sidewalk_on_at_least_one_side_is_priced_as_one_side():
    street = dict(kind="street_unknown", representation="street_centerline")
    pen = lambda v: sidewalk_penalty(_one(**street, sidewalk_presence=v))[0]  # noqa: E731
    assert pen("at_least_one_side") == pen("one_side") < pen(None)
