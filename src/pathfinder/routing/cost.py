"""Cost model: requirements (hard filters) and preferences ("felt minutes").

PathFinder is an accessibility app: every dimension below is about getting along a walk with
a mobility disability (wheelchair, walker, cane, stroller).

Design
------
* **Requirements are never traded off.** They remove edges before search. Missing evidence
  is handled by the user's certainty setting — never assumed favorable.
* **Preferences** convert conditions into *felt seconds*, a time-equivalent penalty, so
  every explanation is in minutes. For each edge we precompute one penalty per dimension at
  full weight:

  - **cross**: crossing a street — curb ramps at both ends, signal control, crossing
    distance and traffic speed;
  - **slope**: sustained grade;
  - **sidewalk**: whether there is a sidewalk or the walk is in the roadway, its surface,
    reported obstacles, and steps when step-free is not a requirement.

  A request combines them with the user's simplex weights ``w`` and an ``intensity``::

      cost(e) = time(e) + intensity * 3 * (w_cross*P_cross(e) + w_slope*P_slope(e)
                                           + w_sidewalk*P_sidewalk(e))

  (The factor 3 makes equal weights reproduce the full penalty table.)
* **Unknown is penalized, not ignored.** Each table has an explicit value for unknown
  evidence, chosen to sit between "good" and "bad" — so the router prefers documented
  routes without pretending unknowns are barriers.
* **A crossing counts once.** Crossing penalties are apportioned across a crossing's
  fragments by length share.
* **Unmapped crossings count too.** Where the graph follows a street centerline, passing a
  junction crosses a street with no crossing edge. Each flagged edge end
  (``enrich.enrich_implicit_crossings``) is charged half a crossing of unknown control,
  with its one curb.

All numbers live in ``PENALTIES``. They are product judgments, versioned so results are
reproducible, and are expected to be revised with disabled testers (docs/PLAN.md, Phase E).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from ..config import DEFAULT_WALK_SPEED_MPS
from ..graph.enrich import KERB_BARRIER, KERB_PASSABLE, SIDEWALK_PRESENT

# v2 (2026-10-05): adds implicit crossings at street-centerline junctions.
# v3 (2026-10-05): adds the opt-in reported-crime penalty (off by default, so default
#                  routes are identical to v2).
# v4 (2026-10-05): reported crime is measured per block, harm- and recency-weighted, shrunk,
#                  and separately for day and night (still opt-in; defaults unchanged).
# v5 (2026-10-15): accessibility refocus. Dimensions are crossings / slope / sidewalk; curb
#                  ramps move into crossings; walking in the roadway and missing sidewalk
#                  become one sidewalk table (the sum of the two old ones); after-dark and
#                  reported-crime penalties are removed.
# v6 (2026-10-16): a curb ramp that fails its inventory's own check costs extra.
# v7 (2026-10-16): a sidewalk's surveyed condition index (City of Frederick) costs by band.
# v8 (2026-10-16): a sidewalk recorded on at least one side is priced as one side.
PENALTY_VERSION = "v8"

PENALTIES = {
    "crossing": {
        # base felt seconds per physical crossing, by control type
        "base": {"signal": 10.0, "marked_uncontrolled": 25.0, "unmarked": 45.0, "unknown": 45.0},
        "hawk_base": 15.0,
        "lpi_credit": -3.0,
        # extra seconds per meter of crossing distance, by control type
        "per_m": {"signal": 0.5, "marked_uncontrolled": 1.5, "unmarked": 2.5, "unknown": 2.5},
        # speed multiplier for uncontrolled crossings: (speed/25)^1.5; unknown speed -> 30
        "unknown_speed_mph": 30.0,
        "speed_ref_mph": 25.0,
        "speed_exp": 1.5,
        # implicit crossings with no roadway inventory block at the junction: the median
        # cross-section width at junctions that do have one is 9.8 m
        "implicit_default_width_m": 10.0,
        # curb at each crossing end (implicit crossings: one curb per flagged edge end)
        "kerb_barrier": 60.0,
        "kerb_unknown": 30.0,
        # a ramp is there but failed its survey (too steep, no landing, damaged): usable, so
        # never a hard barrier, but felt more than an unrecorded curb's uncertainty is not
        "ramp_fails": 40.0,
    },
    "slope": {
        "grade_free_pct": 5.0,  # no penalty at or below this sustained grade
        "per_pct_per_100m": 10.0,  # felt seconds per % above free grade, per 100 m
        "unknown_grade_per_100m": 20.0,
    },
    "sidewalk": {
        # along street-centerline edges: a sidewalk on neither side means walking in the
        # roadway (v4's "street_per_100m" and "no_sidewalk_per_100m", summed)
        # "at_least_one_side" (a survey recorded one side, the other is not known) is priced as
        # one side: the walker has the recorded sidewalk, and no more is assumed
        "presence_per_100m": {"none": 100.0, "one_side": 25.0, "at_least_one_side": 25.0, "both": 0.0, "unknown": 60.0},
        "shared_with_cars_per_100m": {"alley": 10.0, "service": 10.0},
        "surface_per_100m": {"smooth": 0.0, "rough": 15.0, "unpaved": 30.0, "unknown": 5.0},
        "ps_problem_each": 5.0,  # Project Sidewalk obstacle / surface problem labels (dated)
        # the survey's condition index (0-100, higher is better), felt seconds per 100 m:
        # (lower bound of band, penalty), checked from the top; below every band, "worst".
        # The city has published no cutoffs, so these bands are product judgments: from 70
        # (the city averages 78, "satisfactory") no charge, then rising as panels heave and
        # crack (the survey's main defect is level differences between panels). Unknown
        # costs a little, as an unknown surface does: it is not assumed good.
        "condition_per_100m": {"bands": ((70.0, 0.0), (55.0, 10.0), (40.0, 25.0)), "worst": 45.0, "unknown": 5.0},
        "steps_base": 30.0,  # only applies when step-free is not a requirement
        "steps_per_m": 2.0,
    },
}


# ============================================================ request objects

CERTAINTY_LEVELS = ("verified_only", "allow_likely", "allow_unknown")
PENALTY_DIMS = ("cross", "slope", "sidewalk")  # column order of Network.pen()


@dataclass(frozen=True)
class Requirements:
    step_free: bool = False
    max_grade_pct: float | None = None  # e.g. 8.33 (ADA running slope for ramps)
    certainty: str = "allow_likely"

    def __post_init__(self) -> None:
        if self.certainty not in CERTAINTY_LEVELS:
            raise ValueError(f"certainty must be one of {CERTAINTY_LEVELS}")


@dataclass(frozen=True)
class Preferences:
    w_cross: float = 1 / 3
    w_slope: float = 1 / 3
    w_sidewalk: float = 1 / 3
    intensity: float = 1.0
    budget_s: float = 300.0  # max extra real walking seconds over the fastest feasible route
    walk_speed_mps: float = DEFAULT_WALK_SPEED_MPS
    weights: tuple[float, float, float] = field(init=False)

    def __post_init__(self) -> None:
        w = np.array([self.w_cross, self.w_slope, self.w_sidewalk], dtype=float)
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError("weights must be non-negative and not all zero")
        object.__setattr__(self, "weights", tuple(w / w.sum()))

    def at(self, w: tuple[float, float, float]) -> "Preferences":
        """These preferences at another point of the priority mix."""
        return replace(self, w_cross=w[0], w_slope=w[1], w_sidewalk=w[2])

    def multipliers(self, weights: tuple[float, float, float] | None = None) -> np.ndarray:
        """What each penalty dimension (PENALTY_DIMS) is multiplied by in the felt cost, at
        this user's priorities or at another point ``weights`` of the priority mix. The
        factor 3 makes equal weights reproduce the full penalty tables."""
        w = self.weights if weights is None else weights
        return self.intensity * 3 * np.asarray(w, dtype=float)


# ============================================================ per-edge penalties


def _status_known(edges: pd.DataFrame, attr: str) -> pd.Series:
    return edges[f"{attr}_status"].ne("unknown")


def _kerb_penalty(edges: pd.DataFrame, P: dict) -> np.ndarray:
    """Curbs: both outer ends of mapped crossings, and the one curb of each implicit
    crossing end."""
    c = P["crossing"]
    out = np.zeros(len(edges))
    m = edges["kind"].eq("crossing")
    share = (edges["length_m"] / edges["crossing_total_length_m"]).fillna(1.0).clip(0, 1)
    for end in ("kerb_a", "kerb_b"):
        k = edges[end].where(_status_known(edges, end))
        pen = np.where(k.isin(KERB_BARRIER), c["kerb_barrier"], np.where(k.isin(KERB_PASSABLE), 0.0, c["kerb_unknown"]))
        out = out + np.where(m, pen * share, 0.0)
        check = edges.get(f"ramp_check_{end[-1]}")
        if check is not None:
            fails = check.eq("fails") & edges[f"ramp_check_{end[-1]}_status"].ne("unknown")
            out = out + np.where(m & fails, c["ramp_fails"] * share, 0.0)
    implicit_kerb = {"no_curb": 0.0, "ramp_likely": 0.0, "unknown": c["kerb_unknown"], "no_ramp_reported": c["kerb_barrier"]}
    for end in ("u", "v"):
        pen = edges[f"xing_{end}_kerb"].fillna("unknown").map(implicit_kerb).astype(float)
        out = out + np.where(edges[f"xing_{end}"].to_numpy(dtype=bool), pen, 0.0)
    return np.asarray(out, dtype=float)


def crossing_penalty(edges: pd.DataFrame, P: dict = PENALTIES) -> np.ndarray:
    """Crossing streets: control, distance and traffic speed, plus the curbs at the ends."""
    c = P["crossing"]
    m = edges["kind"].eq("crossing").to_numpy()
    ctrl = edges["crossing_control"].where(_status_known(edges, "crossing_control"), "unknown").fillna("unknown")
    base = ctrl.map(c["base"]).astype(float)
    hawk = edges["signal_hawk"].eq(True)
    base = base.where(~hawk, c["hawk_base"])
    base = base + np.where(edges["signal_lpi"].eq(True), c["lpi_credit"], 0.0)
    # crossing distance: the roadway inventory's cross-section width when known, else mapped crossing length
    dist = pd.to_numeric(edges["crossed_width_m"], errors="coerce").fillna(edges.get("crossing_total_length_m"))
    dist = dist.fillna(edges["length_m"])
    per_m = ctrl.map(c["per_m"]).astype(float)
    speed = pd.to_numeric(edges["crossed_speed_mph"], errors="coerce").fillna(c["unknown_speed_mph"])
    speed_mult = np.where(ctrl.isin(["marked_uncontrolled", "unmarked", "unknown"]),
                          (speed / c["speed_ref_mph"]) ** c["speed_exp"], 1.0)
    per_crossing = (base + per_m * dist) * speed_mult
    share = (edges["length_m"] / edges["crossing_total_length_m"]).fillna(1.0).clip(0, 1)
    out = np.where(m, per_crossing * share, 0.0)

    # implicit crossings: half a crossing per flagged edge end; the speed of the crossed
    # street is never known here
    unknown_mult = (c["unknown_speed_mph"] / c["speed_ref_mph"]) ** c["speed_exp"]
    for end in ("u", "v"):
        ictrl = edges[f"xing_{end}_control"].fillna("unknown")
        width = pd.to_numeric(edges[f"xing_{end}_width_m"], errors="coerce").fillna(c["implicit_default_width_m"])
        full = (ictrl.map(c["base"]).astype(float) + ictrl.map(c["per_m"]).astype(float) * width)
        full = full * np.where(ictrl.eq("signal"), 1.0, unknown_mult)
        out = out + np.where(edges[f"xing_{end}"].to_numpy(dtype=bool), 0.5 * full, 0.0)
    return np.asarray(out + _kerb_penalty(edges, P), dtype=float)


def slope_penalty(edges: pd.DataFrame, P: dict = PENALTIES) -> np.ndarray:
    a = P["slope"]
    per100 = edges["length_m"].to_numpy() / 100.0
    g = pd.to_numeric(edges["grade_max_pct"], errors="coerce")
    pen = np.where(g.isna(), a["unknown_grade_per_100m"],
                   np.clip(g - a["grade_free_pct"], 0, None) * a["per_pct_per_100m"]) * per100
    # stairs/elevators have no terrain grade by design; their cost is under sidewalk
    return np.asarray(np.where(edges["kind"].isin(["steps", "escalator", "elevator"]), 0.0, pen), dtype=float)


def condition_per_100m(edges: pd.DataFrame, c: dict) -> np.ndarray:
    """Felt seconds per 100 m for the surveyed sidewalk condition. Only ways a condition
    survey can speak for are charged for not having one: sidewalk-level ways and streets
    with a sidewalk on record. Elsewhere (paths, crossings, the roadway) it does not apply."""
    if "sidewalk_condition" not in edges:
        return np.zeros(len(edges))
    v = pd.to_numeric(edges["sidewalk_condition"].where(_status_known(edges, "sidewalk_condition")), errors="coerce")
    out = np.full(len(edges), c["worst"])
    for lo, pen in sorted(c["bands"]):  # lowest bound first; higher ones overwrite
        out = np.where(v >= lo, pen, out)
    out = np.where(v.isna(), c["unknown"], out)
    presence = edges["sidewalk_presence"].where(_status_known(edges, "sidewalk_presence"))
    applies = ((edges["kind"].eq("sidewalk") & edges["representation"].eq("pedestrian_way"))
               | (edges["representation"].eq("street_centerline") & presence.isin(SIDEWALK_PRESENT)))
    return np.where(applies.to_numpy(), out, 0.0)


def sidewalk_penalty(edges: pd.DataFrame, P: dict = PENALTIES) -> np.ndarray:
    """The way itself: a sidewalk or the roadway, its surface, obstacles, and steps."""
    a = P["sidewalk"]
    per100 = edges["length_m"].to_numpy() / 100.0
    is_street = edges["kind"].isin(["street_with_sidewalk", "street_no_sidewalk", "street_unknown"]).to_numpy()
    presence = edges["sidewalk_presence"].where(_status_known(edges, "sidewalk_presence"), "unknown").fillna("unknown")
    presence = presence.where(~edges["kind"].eq("street_no_sidewalk"), "none")
    out = np.where(is_street, presence.map(a["presence_per_100m"]).astype(float).to_numpy() * per100, 0.0)
    out = out + edges["kind"].map(a["shared_with_cars_per_100m"]).fillna(0.0).to_numpy() * per100

    surf = edges["surface_class"].where(_status_known(edges, "surface_class"), "unknown").fillna("unknown")
    out = out + surf.map(a["surface_per_100m"]).astype(float).to_numpy() * per100
    out = out + condition_per_100m(edges, a["condition_per_100m"]) * per100
    probs = edges.get("ps_obstacles", 0) + edges.get("ps_surface_problems", 0)
    out = out + np.asarray(probs, dtype=float) * a["ps_problem_each"]

    steps = edges["kind"].isin(["steps", "escalator"]).to_numpy()
    out = out + np.where(steps, a["steps_base"] + a["steps_per_m"] * edges["length_m"], 0.0)
    return np.asarray(out, dtype=float)


def edge_penalties(edges: pd.DataFrame, P: dict = PENALTIES) -> pd.DataFrame:
    return pd.DataFrame(
        {"P_cross": crossing_penalty(edges, P), "P_slope": slope_penalty(edges, P),
         "P_sidewalk": sidewalk_penalty(edges, P)},
        index=edges.index,
    )


# ============================================================ requirements


def feasible_mask(edges: pd.DataFrame, req: Requirements) -> np.ndarray:
    """Boolean mask of edges usable under the hard requirements and certainty setting."""
    ok = np.ones(len(edges), dtype=bool)
    accept = {"verified_only": {"verified"}, "allow_likely": {"verified", "inferred"},
              "allow_unknown": {"verified", "inferred", "unknown"}}[req.certainty]

    if req.step_free:
        sf = edges["step_free"]
        st = edges["step_free_status"]
        ok &= ~(sf.eq(False)).to_numpy()  # known stairs/escalators are always excluded
        ok &= st.isin(accept).to_numpy() | sf.eq(False).to_numpy()
        # curb ends of crossings
        m = edges["kind"].eq("crossing").to_numpy()
        for end in ("kerb_a", "kerb_b"):
            k, s = edges[end], edges[f"{end}_status"]
            barrier = k.isin(KERB_BARRIER) & s.ne("unknown")
            ok &= ~(m & (barrier | ~s.isin(accept)).to_numpy())
        # implicit crossings are inferred by construction: never "verified only"; curb
        # evidence there is at best likely
        for end in ("u", "v"):
            k = edges[f"xing_{end}_kerb"].fillna("unknown")
            passes = (k.isin(["no_curb", "ramp_likely"]) & ("inferred" in accept)) | (k.eq("unknown") & ("unknown" in accept))
            ok &= ~(edges[f"xing_{end}"].to_numpy(dtype=bool) & ~passes.to_numpy())
        # elevators: operational status needs live WMATA data (Phase 1); unknown until then
        if "unknown" not in accept:
            ok &= ~edges["kind"].eq("elevator").to_numpy()

    if req.max_grade_pct is not None:
        g = pd.to_numeric(edges["grade_max_pct"], errors="coerce")
        s = edges["grade_max_pct_status"]
        no_terrain = edges["kind"].isin(["steps", "escalator", "elevator", "corridor"]).to_numpy()
        too_steep = (g > req.max_grade_pct).to_numpy()
        status_bad = ~s.isin(accept).to_numpy()
        # a measured (any status) grade over the limit always fails; missing/untrusted
        # evidence fails unless the certainty setting accepts it
        ok &= no_terrain | (~too_steep & ~status_bad)
    return ok
