"""Explanations for a route set (Phase 1): is an alternative worth showing, where is the
crux, how far are the user's priorities from a different answer, and what is the smallest
compromise when nothing satisfies the requirements.

Everything here is arithmetic on quantities the router already uses, so an explanation can
never disagree with the route it explains. Nothing in this module calls a route "safe": it
reports felt minutes, real minutes and the recorded conditions behind them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .cost import PENALTY_DIMS, Preferences, Requirements
from .divergence import Section, sections
from .router import Network, Route, RouteSet

DIMS = PENALTY_DIMS  # the three-way priority mix: crossings, slope, sidewalk

# An alternative is offered as a separate choice only if it is felt to be at least this much
# better than the fastest route at the user's own priorities. Below it the two are
# equivalent for practical purposes and showing both would be noise: in the Phase 0
# experiment a large share of "different" recommendations differed by less than this.
WORTH_SHOWING_MIN_GAIN_S = 30.0


def felt_cost(route: Route, pref: Preferences) -> float:
    """What the router minimized: real seconds plus weighted felt seconds."""
    return route.time_s + float(sum(m * route.penalties[d] for d, m in zip(PENALTY_DIMS, pref.multipliers())))


def felt_gain(alt: Route, base: Route, pref: Preferences) -> float:
    """Felt seconds ``alt`` saves over ``base`` at the user's priorities (negative if worse)."""
    return felt_cost(base, pref) - felt_cost(alt, pref)


def worth_showing(alt: Route, base: Route, pref: Preferences) -> bool:
    return alt.key != base.key and felt_gain(alt, base, pref) >= WORTH_SHOWING_MIN_GAIN_S


def shown_route(best: Route, fastest: Route, pref: Preferences) -> Route:
    """The route actually presented as the recommendation at priorities ``pref``: the
    optimizer's choice when it is worth showing over the fastest route, else the fastest."""
    return best if worth_showing(best, fastest, pref) else fastest


# ------------------------------------------------------------------ crux


@dataclass
class SectionDelta:
    """One place where route A leaves route B, and what A gains or loses there (A minus B)."""
    section: Section
    time_s: float  # real seconds; positive = A is slower here
    felt_s: dict[str, float]  # per dimension at full weight; negative = A is better here
    weighted_felt_s: float  # at the user's priorities and intensity
    net_s: float  # time_s + weighted_felt_s; negative = A wins here


def section_deltas(net: Network, a: Route, b: Route, pref: Preferences) -> list[SectionDelta]:
    t = net.time_s(pref.walk_speed_mps)
    P = net.pen()
    mult = pref.multipliers()
    out = []
    for s in sections(a.edge_ids, a.node_ids, b.edge_ids, b.node_ids, net.length):
        if not s.clean:
            continue  # B does not run between the same two nodes in order; no like-for-like comparison
        ae, be = list(s.a_edges), list(s.b_edges)
        dp = P[ae].sum(axis=0) - P[be].sum(axis=0)
        dt = float(t[ae].sum() - t[be].sum())
        weighted = float(dp @ mult)
        out.append(SectionDelta(s, dt, dict(zip(PENALTY_DIMS, map(float, dp))), weighted, dt + weighted))
    return out


def crux(net: Network, a: Route, b: Route, pref: Preferences) -> SectionDelta | None:
    """The single place that most decides for A over B: the section where A's net
    advantage (felt saving minus extra time) is largest. None if the routes do not differ in
    a comparable section."""
    deltas = section_deltas(net, a, b, pref)
    return min(deltas, key=lambda d: d.net_s) if deltas else None


# ------------------------------------------------------------------ what would it take


@dataclass
class TippingPoint:
    weights: tuple[float, float, float]  # nearest priorities at which another route wins
    route: Route
    shift: dict[str, float]  # change in each dimension's weight, in percentage points
    distance: float  # total weight moved (half the L1 distance), 0–1


def tipping_point(rs: RouteSet, pref: Preferences) -> TippingPoint | None:
    """How far the user's priorities are from a different recommendation.

    Looks at the weight grid the route set was computed on and returns the grid point
    closest to the user's weights where the route that would be *shown* differs from the one
    shown now. Both sides go through ``shown_route``, so a winner that is not worth showing
    over the fastest route counts as the fastest route, exactly as it would on screen.
    Distance is the share of total priority that has to move (0.2 = "shift 20 points").
    Resolution is the grid step, so a nearer boundary than one step is reported as one step.
    """
    if rs.recommended is None:
        return None
    now = shown_route(rs.recommended, rs.fastest, pref)
    w0 = np.asarray(pref.weights)
    best = None
    for winner in rs.routes:
        for w in winner.weights_won:
            there = pref.at(w)
            r = shown_route(winner, rs.fastest, there)
            if r.key == now.key:
                continue
            d = float(np.abs(np.asarray(w) - w0).sum() / 2)
            if best is None or d < best[0]:
                best = (d, w, r)
    if best is None:
        return None
    d, w, r = best
    return TippingPoint(tuple(w), r, {k: round(float(100 * (w[i] - w0[i])), 1) for i, k in enumerate(DIMS)}, d)


# ------------------------------------------------------------------ minimal relaxation

# Compromises offered when no route satisfies the requirements, smallest first. Each is one
# change the user can understand and accept or refuse; requirements are never relaxed
# silently, and dropping a requirement outright is always the last resort.
GRADE_STEPS_PCT = (10.0, 12.5)  # 1:10 and 1:8 — the steeper ramp slopes older standards allowed


CERTAINTY_LABELS = {"allow_likely": "accept likely (inferred) evidence",
                    "allow_unknown": "accept places with no evidence"}


@dataclass
class Relaxation:
    label: str
    requirements: Requirements
    route_set: RouteSet
    newly_used_edges: tuple[int, ...]  # edges on the fastest route that the original requirements exclude


def _relaxations(req: Requirements) -> list[tuple[str, Requirements]]:
    out: list[tuple[str, Requirements]] = []
    order = ("verified_only", "allow_likely", "allow_unknown")
    looser = order[order.index(req.certainty) + 1:]
    for c in looser:
        out.append((CERTAINTY_LABELS[c], replace(req, certainty=c)))
    if req.max_grade_pct is not None:
        for g in GRADE_STEPS_PCT:
            if g > req.max_grade_pct:
                out.append((f"allow slopes up to {g:g} %", replace(req, max_grade_pct=g)))
        out.append(("drop the slope limit", replace(req, max_grade_pct=None)))
    if req.step_free:
        out.append(("allow steps and unverified curbs", replace(req, step_free=False)))
    return out


def minimal_relaxation(net: Network, origin: int, dest: int, req: Requirements,
                       pref: Preferences = Preferences()) -> Relaxation | None:
    """The smallest single compromise that makes the trip possible, and where it bites.

    Tries each relaxation in order of size and returns the first that yields a route,
    together with the edges of its fastest route that the original requirements exclude —
    the places the user is being asked to accept. None if no single relaxation helps.
    """
    blocked = ~net.feasible(req)
    for label, relaxed in _relaxations(req):
        rs = net.route_set(origin, dest, relaxed, pref)
        if rs.status == "ok":
            used = tuple(e for e in rs.fastest.edge_ids if blocked[e])
            return Relaxation(label, relaxed, rs, used)
    return None
