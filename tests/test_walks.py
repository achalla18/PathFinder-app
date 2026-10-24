"""Walk recordings: what they suggest for the guidance distances, and what they cannot."""

import json

import pytest

from pathfinder.analysis.walks import FORMAT, OFF_ROUTE_FLOOR_M, load, report, summarize


def _walk(off, *, speed=1.3, every=2.0, replanned_after=None, next_place=None, simulated=False, off_route=40, alert=80):
    """A recording with one fix every ``every`` seconds at the given distances from the route."""
    samples = [{"t": int(i * every * 1000), "lat": 38.9, "lon": -77.0, "accuracy_m": 8.0, "off_route_m": float(o),
                "along_m": round(i * every * speed), "next_place_m": None if next_place is None else next_place[i],
                "next_step_m": None} for i, o in enumerate(off)]
    events = [{"t": samples[0]["t"], "type": "route", "geometry": []}]
    if replanned_after is not None:
        events.append({"t": samples[replanned_after]["t"] + 100, "type": "replanned", "off_route_m": off[replanned_after]})
    return {"format": FORMAT, "started": "2026-10-13T12:00:00Z", "simulated": simulated,
            "thresholds_m": {"off_route": off_route, "arrived": 20, "arrived_radius": 30, "next_place_alert": alert},
            "samples": samples, "events": events}


def test_the_off_route_distance_covers_the_fixes_of_walkers_who_stayed_on_the_route():
    s = summarize([_walk([3, 5, 8, 4, 6] * 20 + [33])])  # one stray fix in 101
    assert s["fixes_on_route"] == 101 and s["replans"] == 0
    assert s["off_route_m"]["p50"] == 5.0
    assert s["suggested"]["off_route_m"] == OFF_ROUTE_FLOOR_M  # 99% are within 8 m; never under the floor


def test_fixes_that_led_to_a_replan_are_not_counted_as_noise_on_the_route():
    off = [4] * 60 + [70, 90, 110] + [5] * 20
    with_replan = summarize([_walk(off, replanned_after=62)])
    assert with_replan["replans"] == 1 and with_replan["fixes_on_route"] == 80
    assert with_replan["suggested"]["off_route_m"] == OFF_ROUTE_FLOOR_M
    # the same fixes with no re-plan are noise the distance has to cover
    assert summarize([_walk(off)])["suggested"]["off_route_m"] > 60
    assert with_replan["replans_at_distance"]["60 m"] == 1 and with_replan["replans_at_distance"]["80 m"] == 1


def test_noisy_fixes_ask_for_a_wider_distance_and_a_single_stray_fix_triggers_nothing():
    s = summarize([_walk(([10, 25, 48, 30] * 25))])
    assert s["suggested"]["off_route_m"] == 50  # 99% within 48 m, rounded up to 5 m
    assert s["replans_at_distance"]["40 m"] == 0  # never two in a row beyond 40 m


def test_alert_distance_follows_walking_speed_and_notice_is_measured():
    ahead = [200 - 3 * i for i in range(70)]  # 1.5 m/s, a fix every 2 s, towards one decision point
    s = summarize([_walk([3] * 70, speed=1.5, next_place=ahead)])
    assert s["walking_speed_mps_p50"] == 1.5
    assert s["suggested"]["next_place_alert_m"] == 70  # 45 s at 1.5 m/s, rounded up to 5 m
    assert s["decision_point_notice_s"]["n"] == 1
    assert 40 <= s["decision_point_notice_s"]["p50"] <= 46  # announced at 80 m, reached at 15 m


def test_simulated_and_too_short_walks_suggest_nothing(tmp_path):
    s = summarize([_walk([0] * 100, simulated=True), _walk([5] * 10)])
    assert s["walks"] == 1 and s["simulated_walks_ignored"] == 1 and s["suggested"] == {}
    path = tmp_path / "walk.json"
    path.write_text(json.dumps(_walk([0] * 100, simulated=True)))
    assert "Nothing to suggest" in report([path])
    assert "38.9" not in report([path])  # no coordinates in the output
    path.write_text(json.dumps({"format": "something-else"}))
    with pytest.raises(ValueError):
        load(path)
