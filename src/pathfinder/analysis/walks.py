"""Tune walking guidance from walk recordings (Phase 1).

A tester can switch on "Record this walk" in the page and afterwards save a file: one line
of numbers per position fix (where the phone said it was, how far that was from the route,
how far the next decision point and the next step were) and what the page did (re-planned,
arrived). The file never leaves the phone unless the walker passes it on.

This module reads such files and says what they imply for the distances the page uses:
how far from the route counts as off it, and how early a decision point is announced. It
prints numbers only; no coordinates from a recording are written anywhere.

What it cannot know is whether a walker who was re-routed had really left the route or was
only reported there by a poor fix. The off-route suggestion is therefore "wide enough to
cover the fixes of walkers who were never re-routed", and every figure is shown with the
number of walks and fixes behind it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

FORMAT = "pathfinder-walk-1"
# The share of fixes, taken while on the route, that the off-route distance should cover.
# The page re-plans only after two fixes in a row beyond it, so one stray fix in a hundred
# does nothing; many more than that and walkers get re-routed while standing on the route.
OFF_ROUTE_COVER_PCT = 99.0
OFF_ROUTE_FLOOR_M = 20.0  # a sidewalk on the other side of an avenue is this far from the route
ROUND_TO_M = 5.0
# Enough warning to read the screen and look up before the place where the routes part.
NOTICE_TARGET_S = 45.0
ARRIVING_M = 15.0  # a decision point this close has been reached
MOVING_MIN_MPS = 0.3  # slower than this between fixes is standing still, not walking
MOVING_MAX_MPS = 3.0  # faster is a jump in the position, not a walker
MIN_FIXES = 30  # fewer fixes than this on route and no distance is suggested
CANDIDATES_M = (20.0, 30.0, 40.0, 50.0, 60.0, 80.0)


def load(path) -> dict:
    """One recording, checked to be in the format the page writes."""
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    if rec.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} recording")
    return rec


def _runs(over: np.ndarray) -> list[tuple[int, int]]:
    """Stretches of consecutive True values as (first index, length)."""
    out, start = [], None
    for i, flag in enumerate(over):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i - start))
            start = None
    if start is not None:
        out.append((start, len(over) - start))
    return out


def _round_up(x: float) -> float:
    return math.ceil(x / ROUND_TO_M) * ROUND_TO_M


def summarize(recordings: list[dict]) -> dict:
    """What a set of recordings says about the guidance distances. Simulated walks (the
    page's demo mode) are counted and otherwise ignored: their positions sit exactly on
    the route."""
    real = [r for r in recordings if not r.get("simulated")]
    out: dict = {"walks": len(real), "simulated_walks_ignored": len(recordings) - len(real)}
    off_on_route: list[float] = []
    accuracy: list[float] = []
    speeds: list[float] = []
    intervals: list[float] = []
    notices: list[float] = []
    replans = arrivals = 0
    would_replan = dict.fromkeys(CANDIDATES_M, 0)
    thresholds: set[float] = set()
    alerts: set[float] = set()
    for rec in real:
        s = rec.get("samples", [])
        if len(s) < 2:
            continue
        t = np.array([x["t"] for x in s], dtype=float) / 1000.0
        off = np.array([x["off_route_m"] for x in s], dtype=float)
        along = np.array([x["along_m"] for x in s], dtype=float)
        used = float(rec.get("thresholds_m", {}).get("off_route", np.nan))
        alert = float(rec.get("thresholds_m", {}).get("next_place_alert", np.nan))
        thresholds.add(used)
        alerts.add(alert)
        accuracy += [x["accuracy_m"] for x in s if x.get("accuracy_m") is not None]
        events = rec.get("events", [])
        replanned_at = np.array([e["t"] for e in events if e["type"] == "replanned"], dtype=float) / 1000.0
        replans += len(replanned_at)
        arrivals += sum(e["type"] == "arrived" for e in events)

        # fixes that led to a re-plan may be a walker who really left the route: they say
        # nothing about noise on it, so the stretch beyond the distance in use that ends at
        # each re-plan is set aside
        keep = np.ones(len(s), dtype=bool)
        if not np.isnan(used):
            for start, n in _runs(off > used):
                end_t = t[start + n - 1]
                if len(replanned_at) and np.any((replanned_at >= t[start]) & (replanned_at <= end_t + 5.0)):
                    keep[start:start + n] = False
        off_on_route += off[keep].tolist()
        for c in CANDIDATES_M:  # two fixes in a row beyond the distance is what triggers a re-plan
            would_replan[c] += sum(n >= 2 for _, n in _runs(off > c))

        dt, d_along = np.diff(t), np.diff(along)
        ok = dt > 0
        v = d_along[ok] / dt[ok]
        speeds += v[(v >= MOVING_MIN_MPS) & (v <= MOVING_MAX_MPS)].tolist()
        intervals += dt[ok].tolist()

        # how long before each decision point its alert was on screen
        ahead = np.array([np.nan if x.get("next_place_m") is None else x["next_place_m"] for x in s], dtype=float)
        if not np.isnan(alert):
            shown = None
            for i in range(len(s)):
                if np.isnan(ahead[i]) or ahead[i] > alert:
                    shown = None
                elif shown is None and ahead[i] > ARRIVING_M:
                    shown = t[i]
                elif shown is not None and ahead[i] <= ARRIVING_M:
                    notices.append(t[i] - shown)
                    shown = None

    pct = lambda a, q: round(float(np.percentile(a, q)), 1) if len(a) else None  # noqa: E731
    out["fixes_on_route"] = len(off_on_route)
    out["replans"], out["arrivals"] = replans, arrivals
    out["off_route_m"] = {f"p{q}": pct(off_on_route, q) for q in (50, 90, 95, 99)}
    out["reported_accuracy_m"] = {f"p{q}": pct(accuracy, q) for q in (50, 90)}
    out["seconds_between_fixes_p50"] = pct(intervals, 50)
    out["walking_speed_mps_p50"] = round(float(np.median(speeds)), 2) if speeds else None
    out["replans_at_distance"] = {f"{int(c)} m": would_replan[c] for c in CANDIDATES_M}
    out["in_use"] = {"off_route_m": sorted(x for x in thresholds if not np.isnan(x)),
                     "next_place_alert_m": sorted(x for x in alerts if not np.isnan(x))}
    out["decision_point_notice_s"] = {"n": len(notices), "p50": pct(notices, 50), "p10": pct(notices, 10)}

    suggest: dict = {}
    if len(off_on_route) >= MIN_FIXES:
        suggest["off_route_m"] = max(_round_up(float(np.percentile(off_on_route, OFF_ROUTE_COVER_PCT))), OFF_ROUTE_FLOOR_M)
    if len(speeds) >= MIN_FIXES:
        suggest["next_place_alert_m"] = _round_up(float(np.median(speeds)) * NOTICE_TARGET_S)
    out["suggested"] = suggest
    return out


def report(paths: list) -> str:
    """A plain-text summary of the recordings at ``paths``."""
    s = summarize([load(p) for p in paths])
    lines = [f"{s['walks']} recorded walk(s), {s['fixes_on_route']} position fixes on route"
             + (f" ({s['simulated_walks_ignored']} simulated walk(s) ignored)" if s["simulated_walks_ignored"] else ""),
             f"Re-planned {s['replans']} time(s); arrived {s['arrivals']} time(s).", ""]
    if not s["fixes_on_route"]:
        return "\n".join(lines + ["Nothing to suggest: there are no real walks to go on."])
    o, a = s["off_route_m"], s["reported_accuracy_m"]
    lines += [
        f"Distance from the route while on it: half of fixes within {o['p50']} m, 90% within {o['p90']} m, "
        f"95% within {o['p95']} m, 99% within {o['p99']} m.",
        f"Accuracy the phone itself reported: half within {a['p50']} m, 90% within {a['p90']} m." if a["p50"] is not None
        else "The phones reported no accuracy.",
        f"A fix every {s['seconds_between_fixes_p50']} s; walking speed {s['walking_speed_mps_p50']} m/s (medians).",
        "", "Re-plans there would have been at each off-route distance (two fixes in a row beyond it):",
        "  " + ", ".join(f"{k}: {v}" for k, v in s["replans_at_distance"].items()),
        f"  In use when recorded: {', '.join(f'{x:g} m' for x in s['in_use']['off_route_m']) or 'not recorded'}.", "",
    ]
    n = s["decision_point_notice_s"]
    lines.append(f"Decision points were announced {n['p50']} s ahead (median of {n['n']}; one in ten under {n['p10']} s)."
                 if n["n"] else "No decision point was reached with its alert showing, so notice time is not measured.")
    lines.append("")
    sug = s["suggested"]
    if "off_route_m" in sug:
        lines.append(f"Suggested off-route distance: {sug['off_route_m']:g} m "
                     f"(covers {OFF_ROUTE_COVER_PCT:g}% of fixes on route, never under {OFF_ROUTE_FLOOR_M:g} m).")
    else:
        lines.append(f"No off-route distance suggested: fewer than {MIN_FIXES} fixes on route.")
    if "next_place_alert_m" in sug:
        lines.append(f"Suggested decision-point alert distance: {sug['next_place_alert_m']:g} m "
                     f"({NOTICE_TARGET_S:g} s of notice at the walking speed recorded).")
    lines.append("These are suggestions from the walks given; a re-routed walker may really have left the route, "
                 "which the recording cannot tell.")
    return "\n".join(lines)
