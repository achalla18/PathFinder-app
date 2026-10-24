"""Mobility profiles: starting requirements and priorities, always overridable."""

import pytest

from pathfinder.routing import profiles
from pathfinder.routing.cost import Requirements

from .test_api import _body, client  # noqa: F401  (fixture)
from .test_service import _lonlat, planner  # noqa: F401  (fixture)


def test_each_profile_has_a_mix_that_sums_to_one_and_wheeled_ones_need_no_steps():
    for p in profiles.PROFILES.values():
        assert sum(p.mix) == pytest.approx(1.0)
    for pid in ("manual_wheelchair", "power_wheelchair", "stroller"):
        assert profiles.get(pid).requirements.step_free
    assert profiles.get("manual_wheelchair").requirements.max_grade_pct == profiles.ADA_RAMP_MAX_GRADE_PCT
    assert profiles.get("walker_cane").requirements == Requirements()
    with pytest.raises(KeyError):
        profiles.get("jetpack")


def test_an_explicit_choice_overrides_the_profile():
    req, mix = profiles.resolve("manual_wheelchair", {"certainty": "verified_only"}, (1, 0, 0))
    assert req.step_free and req.max_grade_pct == 8.33 and req.certainty == "verified_only"
    assert mix == (1, 0, 0)
    req, mix = profiles.resolve(None)
    assert req == Requirements() and mix == profiles.get("custom").mix


def test_the_api_applies_a_profile_and_lets_explicit_fields_win(client):  # noqa: F811
    o, d = _lonlat(0, 0), _lonlat(215, 0)
    out = client.post("/api/plan", json=_body(o, d, profile="stroller")).json()
    assert out["requirements"]["step_free"] is True
    assert out["priorities"]["sidewalk"] == pytest.approx(0.45)
    out = client.post("/api/plan", json=_body(o, d, profile="stroller", step_free=False,
                                              priorities={"cross": 1, "slope": 0, "sidewalk": 0})).json()
    assert out["requirements"]["step_free"] is False and out["priorities"]["cross"] == pytest.approx(1.0)
    assert client.post("/api/plan", json=_body(o, d, profile="jetpack")).status_code == 422
    assert client.post("/api/plan", json=_body(o, d, night=True)).status_code == 200  # unknown fields are ignored
    region = client.get("/api/region").json()
    assert {p["id"] for p in region["profiles"]} == set(profiles.PROFILES) and "reported_crime" not in region
