"""On-screen guidance (Phase 1): the steps of one route, in the order they are walked.

A step is something a walker has to do or would want to know before reaching it: a turn
where there is a choice of ways, a street to cross, stairs, a stretch with no sidewalk. Each
is tied to a vertex of the route's geometry, so the page can say how far ahead it is from
the walker's own position without asking the server again.

OSM sidewalks carry no street name, so a sidewalk is named after the street it runs beside
(``along_street``, from the roadway inventory or, without one, OSM's streets) and a crossing after the street it
crosses (``crossed_street``): "turn left onto K ST NW", "cross 9TH ST NW". Where neither
is known the step says what to do and names nothing.

The product rules hold here as everywhere. A crossing's control and curbs are given as
claims with status, source and date; an unrecorded one is ``unknown``, never assumed to be
a signal. Crossings that are not mapped (``xing_u`` / ``xing_v`` on street centerlines) are
steps too. Nothing here changes a route.
"""

from __future__ import annotations

import math
import re

import numpy as np

from ..graph.enrich import signal_inventory_as_of
from ..sources import REGION

# A turn is measured between where the walker was this far before the junction and where
# they are this far after it, not between the two segments that meet there: mapped corners
# are often two or three short pieces a few metres long, each turning a little.
TURN_WINDOW_M = 12.0
# Below this the way ahead reads as "straight on" and nothing is said. Street grids meet at
# 90 degrees and DC's avenues cut them at about 30 or more; mapping wobble is well under.
TURN_MIN_DEG = 30.0
TURN_SLIGHT_BELOW_DEG = 55.0
TURN_SHARP_ABOVE_DEG = 125.0
# A turn is announced only where the walker could have gone another way: a bend in a path
# with no branch needs no instruction.
CHOICE_MIN_DEGREE = 3

_COMPASS = ("north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest")
# Steps at the same spot are given in the order they are met: turn to face the street, then cross.
_ORDER = {"start": 0, "turn": 1, "unmapped_crossing": 2, "cross": 2, "steps": 3, "no_sidewalk": 3, "arrive": 9}
# The roadway inventory's abbreviations, as found in its ROUTENAME values.
_STREET_TYPES = {
    "street": "ST", "avenue": "AVE", "road": "RD", "drive": "DR", "place": "PL", "lane": "LN",
    "terrace": "TER", "circle": "CIR", "court": "CT", "way": "WAY", "parkway": "PKWY",
    "boulevard": "BLVD", "square": "SQ", "crescent": "CRES", "row": "ROW", "walk": "WALK",
    "mews": "MEWS", "bridge": "BRG", "expressway": "EXPY", "freeway": "FWY", "plaza": "PLZ",
    "alley": "ALLEY",  # named alleys are not in the inventory; only the quadrant is shortened
}
_QUADRANTS = {"northwest": "NW", "northeast": "NE", "southeast": "SE", "southwest": "SW"}
_STREET_NAME = re.compile(
    rf"^(?P<base>.+?) (?P<type>{'|'.join(_STREET_TYPES)}) (?P<quad>{'|'.join(_QUADRANTS)})\b", re.IGNORECASE)


def _claim(row: dict, attr: str) -> dict:
    """A value with what backs it. Unknown is stated, never left out."""
    return {"value": row[attr], "status": row[f"{attr}_status"], "source": row[f"{attr}_source"], "as_of": row[f"{attr}_as_of"]}


def _inventory_style(name: str) -> str:
    """An OSM street name written as the roadway inventory writes it. OSM spells names out
    ("C Street Southeast") and the inventory abbreviates them ("C ST SE"); left alone, one
    trip reads "onto C ST SE" at one corner and "onto C Street Southeast" at the next, and
    a sidewalk beside a street seems to be a different way from the street itself. A name
    that ends in a street type and quadrant becomes the inventory's form exactly; one with
    words after them (a cycle track beside the street) is abbreviated and keeps them. Trail
    and park names have no quadrant and are left as they are."""
    m = _STREET_NAME.search(name)
    if not m:
        return name
    kind, quad = _STREET_TYPES[m["type"].lower()], _QUADRANTS[m["quad"].lower()]
    if m.end() == len(name):
        return f"{m['base']} {kind} {quad}".upper()
    return f"{m['base']} {kind.title()} {quad}{name[m.end():]}"


def _name(row: dict | None) -> str | None:
    """What the way is called: its own name, or the street a sidewalk runs beside."""
    if not row:
        return None
    return _inventory_style(row["name"]) if row["name"] else row["along_street"]


def _no_sidewalk(row: dict) -> bool:
    return row["kind"] == "street_no_sidewalk" or row["sidewalk_presence"] == "none"


def _half_crossing(row: dict, end: str) -> dict:
    """The claims for one unmapped half crossing at an edge's ``u`` or ``v`` end."""
    control, kerb = row[f"xing_{end}_control"], row[f"xing_{end}_kerb"]
    return {"control": {"value": control, "status": "inferred" if control else "unknown",
                        "source": REGION.role("signals") if control else None,
                                            "as_of": signal_inventory_as_of(REGION.role("signals")) if control else None},
            "curb_ramps": junction_kerb_claim(row, end)}


def junction_kerb_claim(row: dict, end: str) -> dict:
    """The curb claim at one flagged end: inferred from the records at the junction, with their
    source and oldest date (``enrich.enrich_implicit_crossings``)."""
    kerb = row[f"xing_{end}_kerb"]
    known = bool(kerb) and kerb != "unknown"
    return {"value": kerb, "status": "inferred" if known else "unknown",
            "source": row[f"xing_{end}_kerb_source"] if known else None,
            "as_of": row[f"xing_{end}_kerb_as_of"] if known else None}


def _less_favorable(a: dict, b: dict) -> dict:
    """Two halves of one unmapped crossing that disagree: the claim is unknown, not the better half.
    Halves that agree on value and status but rest on different records keep both sources and
    the older date."""
    if a == b:
        return a
    if a["value"] == b["value"] and a["status"] == b["status"]:
        sources = sorted({s for c in (a, b) if c["source"] for s in c["source"].split(", ")})
        dates = [c["as_of"] for c in (a, b)]
        return {**a, "source": ", ".join(sources) or None, "as_of": None if None in dates else min(dates)}
    return {"value": None, "status": "unknown", "source": None, "as_of": None}


def build_steps(parts: list[np.ndarray], rows: list[dict | None], forward: list[bool], join_degree: list[int]) -> list[dict]:
    """The steps of a route.

    ``parts`` are the pieces of the route's geometry in walking order, as (n, 2) arrays in
    the analysis CRS, exactly as they are stacked into the geometry sent to the page (so
    ``vertex`` indexes that geometry). ``rows[i]`` is the evidence row of the edge ``parts[i]``
    lies on, or None for the partial walkway at either end of the trip. ``forward[i]`` says
    whether the edge is walked from its ``u`` end. ``join_degree[i]`` is how many walkways
    meet at the node between ``parts[i]`` and ``parts[i + 1]``.
    """
    if len(join_degree) != len(parts) - 1:
        raise ValueError("one join between each pair of consecutive parts")
    xy = np.vstack(parts)
    counts = [len(p) for p in parts]
    first = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)
    last = first + np.asarray(counts) - 1
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    total = float(cum[-1])

    def point(s: float) -> np.ndarray:
        return np.array([np.interp(s, cum, xy[:, 0]), np.interp(s, cum, xy[:, 1])])

    steps: list[dict] = []
    start = {"type": "start", "vertex": 0}
    if total > 1.0:
        d = point(min(TURN_WINDOW_M, total)) - xy[0]
        start["heading"] = _COMPASS[round(math.degrees(math.atan2(d[0], d[1])) / 45) % 8]
    steps.append(start)

    # ---- turns, where there is a choice of ways
    turns: list[tuple[float, float, int]] = []  # (metres along, signed degrees: left is positive, part before the join)
    for i, degree in enumerate(join_degree):
        s = float(cum[last[i]])
        # a turn in the last few metres is the arrival itself
        if degree < CHOICE_MIN_DEGREE or s < 1.0 or total - s < TURN_WINDOW_M:
            continue
        a, b = point(max(s - TURN_WINDOW_M, 0.0)), point(min(s + TURN_WINDOW_M, total))
        here = xy[last[i]]
        before, after = here - a, b - here
        if not before.any() or not after.any():
            continue
        angle = math.degrees(math.atan2(before[0] * after[1] - before[1] * after[0], float(before @ after)))
        if abs(angle) >= TURN_MIN_DEG:
            turns.append((s, angle, i))
    # one corner mapped as several junctions a few metres apart is one turn: keep its sharpest reading
    kept: list[tuple[float, float, int]] = []
    for t in turns:
        if kept and t[0] - kept[-1][0] < TURN_WINDOW_M:
            if abs(t[1]) > abs(kept[-1][1]):
                kept[-1] = t
        else:
            kept.append(t)
    for s, angle, i in kept:
        size = abs(angle)
        how = "slight_" if size < TURN_SLIGHT_BELOW_DEG else "sharp_" if size > TURN_SHARP_ABOVE_DEG else ""
        step = {"type": "turn", "vertex": int(last[i]), "direction": how + ("left" if angle > 0 else "right")}
        # the name of the way turned onto, where it has one and it is not the way already walked
        # (the last named way before the turn: corners are full of short nameless pieces)
        was = next((n for n in (_name(rows[k]) for k in range(i, -1, -1)) if n), None)
        for j in range(i + 1, len(parts)):
            if cum[first[j]] - s > TURN_WINDOW_M:
                break
            name = _name(rows[j])
            if name:
                if name != was:
                    step["onto"] = name
                break
        steps.append(step)

    # ---- what lies along the way
    i = 0
    while i < len(parts):
        r = rows[i]
        if r is None:
            i += 1
            continue
        j = i
        if r["kind"] == "crossing":
            # a crossing split into several edges (at an island, say) is still one crossing
            while j + 1 < len(parts) and rows[j + 1] and rows[j + 1]["kind"] == "crossing" and rows[j + 1]["osm_way_id"] == r["osm_way_id"]:
                j += 1
            steps.append({"type": "cross", "vertex": int(first[i]), "street": r["crossed_street"],
                          "length_m": round(float(cum[last[j]] - cum[first[i]])),
                          "control": _claim(r, "crossing_control"), "curb_ramps": [_claim(r, "kerb_a"), _claim(r, "kerb_b")]})
        elif r["kind"] in ("steps", "escalator"):
            while j + 1 < len(parts) and rows[j + 1] and rows[j + 1]["kind"] == r["kind"]:
                j += 1
            steps.append({"type": "steps", "vertex": int(first[i]), "escalator": r["kind"] == "escalator",
                          "length_m": round(float(cum[last[j]] - cum[first[i]])),
                          "status": "verified", "source": REGION.osm, "as_of": r["step_free_as_of"]})
        elif _no_sidewalk(r):
            while j + 1 < len(parts) and rows[j + 1] and _no_sidewalk(rows[j + 1]) and rows[j + 1]["kind"] not in ("crossing", "steps", "escalator"):
                j += 1
            steps.append({"type": "no_sidewalk", "vertex": int(first[i]),
                          "length_m": round(float(cum[last[j]] - cum[first[i]])),
                          **{k: v for k, v in _claim(r, "sidewalk_presence").items() if k != "value"}})
        i = j + 1

    # ---- crossings that are not mapped: each flagged end of a street-centerline edge is half of one
    halves: list[tuple[int, dict]] = []
    for i, r in enumerate(rows):
        if r is None:
            continue
        ends = ("u", "v") if forward[i] else ("v", "u")
        for end, vertex in zip(ends, (first[i], last[i])):
            if r[f"xing_{end}"]:
                halves.append((int(vertex), _half_crossing(r, end)))
    for vertex, half in halves:
        prev = steps[-1]
        if prev["type"] == "unmapped_crossing" and abs(cum[prev["vertex"]] - cum[vertex]) < 1.0:
            # the two halves either side of one junction: one crossing, no better than its worse half
            prev["control"] = _less_favorable(prev["control"], half["control"])
            prev["curb_ramps"] = _less_favorable(prev["curb_ramps"], half["curb_ramps"])
        else:
            steps.append({"type": "unmapped_crossing", "vertex": vertex, **half})

    steps.append({"type": "arrive", "vertex": int(len(xy) - 1)})
    return sorted(steps, key=lambda s: (cum[s["vertex"]], _ORDER[s["type"]]))
