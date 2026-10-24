"""Mobility profiles: a starting point for someone who gets around a particular way.

A profile is a named set of hard requirements plus a starting priority mix over the three
accessibility dimensions (crossings, slope, sidewalk; see ``cost``). It is only a starting
point: the user can change any requirement or move the mix, and an explicit choice always
wins over the profile's (``resolve``).

The numbers are product judgments made before any tester has used them, and are to be
revised with disabled testers (docs/PLAN.md, Phase E). Each says why it is what it is.
A low-vision profile waits for accessible-signal (APS) data: without it there is nothing to
route on that matters to a blind walker beyond what the other profiles already weigh.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .cost import Requirements

# ADA maximum running slope for a ramp (1:12). A manual wheelchair user pushing up a longer
# sustained grade than this is beyond what accessible design asks anyone to manage.
ADA_RAMP_MAX_GRADE_PCT = 8.33
# Power chairs and scooters climb more; 12% is a conservative manufacturer-typical limit for
# sustained climbs, kept below the steepest rated values.
POWER_CHAIR_MAX_GRADE_PCT = 12.0


@dataclass(frozen=True)
class Profile:
    id: str
    label: str
    description: str
    requirements: Requirements
    mix: tuple[float, float, float]  # crossings, slope, sidewalk; sums to 1


PROFILES: dict[str, Profile] = {p.id: p for p in (
    Profile("manual_wheelchair", "Manual wheelchair",
            "Step-free, no sustained grade over 8.33% (the ADA ramp limit). Slope weighs most: every "
            "climb is pushed by hand.",
            Requirements(step_free=True, max_grade_pct=ADA_RAMP_MAX_GRADE_PCT), (0.35, 0.40, 0.25)),
    Profile("power_wheelchair", "Power wheelchair or scooter",
            "Step-free, no sustained grade over 12%. Curbs and broken or missing sidewalk matter more "
            "than hills.",
            Requirements(step_free=True, max_grade_pct=POWER_CHAIR_MAX_GRADE_PCT), (0.40, 0.20, 0.40)),
    Profile("walker_cane", "Walker or cane",
            "Steps are allowed but cost a lot. Long or busy crossings, rough surfaces and walking in "
            "the roadway weigh about equally with slope.",
            Requirements(), (0.35, 0.30, 0.35)),
    Profile("stroller", "Stroller or cart",
            "Step-free. A good sidewalk matters most; hills matter least.",
            Requirements(step_free=True), (0.30, 0.25, 0.45)),
    Profile("custom", "Custom", "Set your own requirements and priorities.", Requirements(), (1 / 3, 1 / 3, 1 / 3)),
)}
DEFAULT_PROFILE = "custom"


def get(profile_id: str) -> Profile:
    try:
        return PROFILES[profile_id]
    except KeyError:
        raise KeyError(f"unknown profile {profile_id!r}; one of {sorted(PROFILES)}") from None


def resolve(profile_id: str | None, overrides: dict | None = None,
            mix: tuple[float, float, float] | None = None) -> tuple[Requirements, tuple[float, float, float]]:
    """A profile's requirements and mix with the user's explicit choices applied on top.
    ``overrides`` holds only the requirement fields the user actually set."""
    p = get(profile_id or DEFAULT_PROFILE)
    return replace(p.requirements, **(overrides or {})), (mix if mix is not None else p.mix)


def as_json() -> list[dict]:
    """The profiles as the page shows them."""
    return [{"id": p.id, "label": p.label, "description": p.description,
             "requirements": {"step_free": p.requirements.step_free, "max_grade_pct": p.requirements.max_grade_pct,
                              "certainty": p.requirements.certainty},
             "mix": {"cross": p.mix[0], "slope": p.mix[1], "sidewalk": p.mix[2]}} for p in PROFILES.values()]
