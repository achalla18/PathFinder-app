"""Plan one trip and explain it (Phase 1): the JSON-ready answer the app will render.

``TripPlanner`` wraps the routable network with what a request needs beyond routing:
snapping coordinates to the graph, route geometry, and the dated, sourced evidence behind
every statement in an explanation. Nothing returned here calls a route "safe"; every
condition carries its status, source and date.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from pyproj import Transformer
from shapely.geometry import Point
from shapely.ops import substring

from ..config import ANALYSIS_CRS, GRAPH_DIR, WGS84
from .cost import PENALTY_DIMS, PENALTY_VERSION, Preferences, Requirements
from .explain import (
    WORTH_SHOWING_MIN_GAIN_S,
    SectionDelta,
    felt_cost,
    felt_gain,
    minimal_relaxation,
    section_deltas,
    shown_route,
    tipping_point,
)
from .guidance import build_steps, junction_kerb_claim
from ..graph.enrich import signal_inventory_as_of
from ..sources import REGION
from .router import Network, Route, simplex_grid

# Farther than this from any walkway the request is refused rather than quietly started
# somewhere else ("near is not connected").
SNAP_MAX_M = 150.0

DISCLAIMER = (
    "Felt minutes are PathFinder's time-equivalent weights for recorded conditions. PathFinder "
    "does not certify any route or place as accessible: conditions are as recorded by the named "
    "source on the given date, may have changed, and unknown means nobody has recorded it."
)

_EVIDENCE_ATTRS = ("crossing_control", "kerb_a", "kerb_b", "ramp_check_a", "ramp_check_b", "step_free", "grade_max_pct",
                   "sidewalk_presence", "surface_class", "sidewalk_condition", "sidewalk_width_m")
_EVIDENCE_COLUMNS = ["edge_id", "kind", "representation", "osm_way_id", "length_m", "name", "along_street", "crossed_street",
                     "crossed_lanes", "crossed_speed_mph", "signal_lpi", "signal_hawk",
                     *[c for a in _EVIDENCE_ATTRS for c in (a, f"{a}_status", f"{a}_source", f"{a}_as_of")],
                     *[f"xing_{end}{s}" for end in ("u", "v") for s in ("", "_control", "_kerb", "_kerb_source", "_kerb_as_of")]]


def _claim(row: dict, attr: str) -> dict:
    """A value with what backs it. Unknown is stated, never left out."""
    return {"value": row[attr], "status": row[f"{attr}_status"], "source": row[f"{attr}_source"], "as_of": row[f"{attr}_as_of"]}


def _number(v) -> float | None:
    """Graded values arrive as whatever the source recorded; a slope is a number or nothing."""
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


@dataclass
class Snap:
    """A requested point placed on the network: the walkway it is on and where along it."""
    edge: int
    along_m: float  # distance from the edge's u end to the snapped point
    distance_m: float  # from the requested point to the walkway
    nearest_edge_meets_requirements: bool  # False when a nearer walkway was passed over


@dataclass
class Legs:
    """The partial edges walked from the origin point to the first node and from the last
    node to the destination point. They are the same for every route of a trip."""
    time_s: float
    length_m: float
    penalties: np.ndarray  # felt seconds per dimension at full weight
    start_xy: np.ndarray  # origin point -> first node
    end_xy: np.ndarray  # last node -> destination point


class TripPlanner:
    def __init__(self, graph_dir=GRAPH_DIR):
        self.graph_dir = graph_dir
        self.net = Network.load(graph_dir)
        geom = gpd.read_parquet(graph_dir / "edges.parquet", columns=["edge_id", "u", "v", "geometry"])
        if not (geom["edge_id"].to_numpy() == np.arange(len(geom))).all():
            raise RuntimeError("edge_id must equal row position in edges.parquet")
        self._geom = np.asarray(geom.geometry.values)
        self._geom_u, self._geom_v = geom["u"].to_numpy(), geom["v"].to_numpy()
        # trips start and end on the main network; islands (mostly private lots) lead nowhere
        self._main_edges = np.flatnonzero(self.net.edges["component"].to_numpy() == 0)
        self._tree = shapely.STRtree(self._geom[self._main_edges])
        self._to_xy = Transformer.from_crs(WGS84, ANALYSIS_CRS, always_xy=True)
        self._to_lonlat = Transformer.from_crs(ANALYSIS_CRS, WGS84, always_xy=True)
        # values with status, source and date for every edge; row position == edge_id
        self._evidence = pq.read_table(graph_dir / "edges.parquet", columns=_EVIDENCE_COLUMNS)
        # how many walkways meet at each node: guidance announces a turn only where there is a choice
        self._node_ids, self._node_degree = np.unique(np.concatenate([self._geom_u, self._geom_v]), return_counts=True)
        # Areas closed to the public (graph.build.clip_restricted removed their walkways). A trip
        # end inside one still snaps to the nearest public walkway, outside the fence; the answer
        # says so rather than implying the place itself can be walked to.
        areas = graph_dir.parent / "interim" / "osm_restricted_areas.parquet"
        self._closed = gpd.read_parquet(areas).to_crs(ANALYSIS_CRS) if areas.exists() else None

    def closed_area_at(self, lon: float, lat: float) -> str | None:
        """The name of the closed area a point lies in, if any."""
        if self._closed is None or self._closed.empty:
            return None
        p = Point(*self._to_xy.transform(lon, lat))
        hit = self._closed[self._closed.geometry.contains(p)]
        return None if hit.empty else str(hit["name"].iloc[0] or "an area closed to the public")

    # ------------------------------------------------------------------ helpers
    def snap(self, lon: float, lat: float, usable: np.ndarray | None = None) -> Snap | None:
        """Place a point on the nearest walkway within SNAP_MAX_M — anywhere along it, not
        only at its ends. With ``usable`` (a mask over edges) the nearest walkway that meets
        the requirements is taken; None if there is none in range."""
        p = Point(*self._to_xy.transform(lon, lat))
        near = self._main_edges[self._tree.query(p, predicate="dwithin", distance=SNAP_MAX_M)]
        if len(near) == 0:
            return None
        dist = shapely.distance(self._geom[near], p)
        nearest = near[np.argmin(dist)]
        if usable is not None:
            ok = usable[near]
            if not ok.any():
                return None
            near, dist = near[ok], dist[ok]
        e = int(near[np.argmin(dist)])
        return Snap(e, float(self._geom[e].project(p)), float(dist.min()), bool(e == nearest))

    def _leg(self, s: Snap, node: int, t_vec: np.ndarray, P: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
        """Walk along the snapped edge between the snapped point and one of its end nodes:
        (seconds, meters, felt seconds per dimension, xy from the point to the node)."""
        g = self._geom[s.edge]
        L = g.length
        to_u, to_v = s.along_m, L - s.along_m
        at_u, at_v = self._geom_u[s.edge] == node, self._geom_v[s.edge] == node
        toward_u = at_u and (not at_v or to_u <= to_v)  # a loop edge has the node at both ends
        d = to_u if toward_u else to_v
        share = d / L if L > 0 else 0.0
        pen = P[s.edge] * share
        xy = shapely.get_coordinates(substring(g, s.along_m, 0.0 if toward_u else L))
        return float(t_vec[s.edge] * share), float(d), pen, xy

    def _endpoints(self, so: Snap, sd: Snap, req: Requirements, pref: Preferences) -> tuple[int, int] | None:
        """Which end of each snapped edge to route between: the pair with the least total
        walking time, partial edges included. None if no pair is connected under ``req``."""
        t_vec, P = self.net.time_s(pref.walk_speed_mps), self.net.pen()
        best = None
        for no in dict.fromkeys((int(self._geom_u[so.edge]), int(self._geom_v[so.edge]))):
            for nd in dict.fromkeys((int(self._geom_u[sd.edge]), int(self._geom_v[sd.edge]))):
                between = 0.0
                if no != nd:
                    r = self.net.fastest(no, nd, req, pref)
                    if r is None:
                        continue
                    between = r.time_s
                total = self._leg(so, no, t_vec, P)[0] + between + self._leg(sd, nd, t_vec, P)[0]
                if best is None or total < best[0]:
                    best = (total, no, nd)
        return None if best is None else (best[1], best[2])

    def _legs(self, so: Snap, sd: Snap, no: int, nd: int, pref: Preferences) -> Legs:
        t_vec = self.net.time_s(pref.walk_speed_mps)
        P = self.net.pen()
        (t1, m1, p1, xy1), (t2, m2, p2, xy2) = self._leg(so, no, t_vec, P), self._leg(sd, nd, t_vec, P)
        return Legs(t1 + t2, m1 + m2, p1 + p2, xy1, xy2[::-1])

    def _line(self, route: Route, legs: Legs) -> list[list[float]]:
        """Trip geometry as [lon, lat] pairs: origin point, each edge in travel direction,
        destination point."""
        coords: list[np.ndarray] = [legs.start_xy]
        for eid, start in zip(route.edge_ids, route.node_ids[:-1]):
            c = shapely.get_coordinates(self._geom[eid])
            coords.append(c if self._geom_u[eid] == start else c[::-1])
        coords.append(legs.end_xy)
        xy = np.vstack(coords)
        lon, lat = self._to_lonlat.transform(xy[:, 0], xy[:, 1])
        return [[round(float(a), 5), round(float(b), 5)] for a, b in zip(lon, lat)]

    def _stretch(self, edge_ids, start_node: int) -> list[list[float]]:
        """Geometry of consecutive edges walked from ``start_node``, as [lon, lat] pairs."""
        coords, node = [], start_node
        for eid in edge_ids:
            c = shapely.get_coordinates(self._geom[eid])
            forward = self._geom_u[eid] == node
            coords.append(c if forward else c[::-1])
            node = self._geom_v[eid] if forward else self._geom_u[eid]
        xy = np.vstack(coords)
        lon, lat = self._to_lonlat.transform(xy[:, 0], xy[:, 1])
        return [[round(float(a), 5), round(float(b), 5)] for a, b in zip(lon, lat)]

    def _steps(self, route: Route, legs: Legs, ev: dict[int, dict]) -> list[dict]:
        """On-screen guidance for one route: its steps in walking order, each tied to a
        vertex of the geometry ``_line`` returns (the parts are stacked the same way)."""
        parts, rows, forward = [legs.start_xy], [None], [True]
        for eid, start in zip(route.edge_ids, route.node_ids[:-1]):
            fwd = bool(self._geom_u[eid] == start)
            c = shapely.get_coordinates(self._geom[eid])
            parts.append(c if fwd else c[::-1])
            rows.append(ev[int(eid)])
            forward.append(fwd)
        parts.append(legs.end_xy)
        rows.append(None)
        forward.append(True)
        # the partial walkways at both ends meet the route at its first and last node
        joins = np.asarray(route.node_ids if len(route.node_ids) == len(parts) - 1 else [], dtype=self._node_ids.dtype)
        degree = self._node_degree[np.searchsorted(self._node_ids, joins)].tolist() if len(joins) else [0] * (len(parts) - 1)
        return build_steps(parts, rows, forward, degree)

    def evidence(self, edge_ids) -> dict[int, dict]:
        """Evidence rows (values with status, source, date) for a handful of edges, as plain
        dicts keyed by edge id. Read once per request: explanations touch the same few
        hundred edges many times over."""
        ids = sorted({int(e) for e in edge_ids})
        return dict(zip(ids, self._evidence.take(ids).to_pylist())) if ids else {}

    def _conditions(self, edge_ids, ev: dict[int, dict], pref: Preferences) -> list[dict]:
        """What is recorded along a stretch, one entry per notable condition."""
        rows = [ev[int(e)] for e in edge_ids]
        out: list[dict] = []
        length = lambda keep: round(sum(r["length_m"] for r in rows if keep(r)), 1)  # noqa: E731
        seen_ways = set()
        for r in rows:
            if r["kind"] == "crossing" and r["osm_way_id"] not in seen_ways:  # one entry per physical crossing
                seen_ways.add(r["osm_way_id"])
                out.append({"type": "crossing", "street": r["crossed_street"],
                            "control": _claim(r, "crossing_control"), "curb_ramps": [_claim(r, "kerb_a"), _claim(r, "kerb_b")],
                            "ramp_checks": [_claim(r, "ramp_check_a"), _claim(r, "ramp_check_b")]})
        for end in ("u", "v"):
            for r in rows:
                if r[f"xing_{end}"]:
                    control = r[f"xing_{end}_control"]
                    out.append({"type": "unmapped_crossing", "street": r["name"],
                                "control": {"value": control, "status": "inferred" if control else "unknown",
                                            "source": REGION.role("signals") if control else None,
                                            "as_of": signal_inventory_as_of(REGION.role("signals")) if control else None},
                                "curb_ramps": junction_kerb_claim(r, end),
                                "note": "Half of a crossing at a junction where sidewalks and crossings are not mapped separately."})
        steps = [r for r in rows if r["kind"] in ("steps", "escalator")]
        if steps:
            out.append({"type": "steps", "length_m": round(sum(r["length_m"] for r in steps), 1),
                        "status": "verified", "source": REGION.osm, "as_of": steps[0]["step_free_as_of"]})
        graded = [r for r in rows if _number(r["grade_max_pct"]) is not None]
        if graded:
            r = max(graded, key=lambda r: _number(r["grade_max_pct"]))
            out.append({"type": "steepest_sustained_grade_pct", **_claim(r, "grade_max_pct"), "value": _number(r["grade_max_pct"])})
        if any(r["grade_max_pct_status"] == "unknown" for r in rows):
            out.append({"type": "slope_unknown", "length_m": length(lambda r: r["grade_max_pct_status"] == "unknown"),
                        "status": "unknown", "source": None, "as_of": None})
        # the survey's own index and width, lowest and narrowest along the stretch
        for attr, kind, pick in (("sidewalk_condition", "lowest_sidewalk_condition", min),
                                 ("sidewalk_width_m", "narrowest_sidewalk_width_m", min)):
            rated = [r for r in rows if _number(r[attr]) is not None]
            if rated:
                r = pick(rated, key=lambda r: _number(r[attr]))
                out.append({"type": kind, **_claim(r, attr), "value": _number(r[attr]),
                            "rated_length_m": length(lambda r: _number(r[attr]) is not None)})
        no_sidewalk = lambda r: r["kind"] == "street_no_sidewalk" or r["sidewalk_presence"] == "none"  # noqa: E731
        first = next((r for r in rows if no_sidewalk(r)), None)
        if first is not None:
            out.append({"type": "no_sidewalk", "length_m": length(no_sidewalk),
                        **{k: v for k, v in _claim(first, "sidewalk_presence").items() if k != "value"}})
        one_side = lambda r: r["sidewalk_presence"] == "at_least_one_side"  # noqa: E731
        first = next((r for r in rows if one_side(r)), None)
        if first is not None:
            out.append({"type": "sidewalk_one_side_recorded", "length_m": length(one_side),
                        **{k: v for k, v in _claim(first, "sidewalk_presence").items() if k != "value"},
                        "note": "A sidewalk is recorded on one side; whether the other side has one is not recorded."})
        links = sum(r["representation"] == "inferred_connector" for r in rows)
        if links:
            out.append({"type": "inferred_link", "count": links, "status": "inferred",
                        "source": "derived:gap_bridging", "as_of": None,
                        "note": "A gap between mapped walkways, assumed walkable."})
        return out

    def _route(self, r: Route, pref: Preferences, legs: Legs) -> dict:
        """One route as shown to the user, door to door: the partial edges at both ends count."""
        pen = {k: v + float(legs.penalties[i]) for i, (k, v) in enumerate(r.penalties.items())}
        felt = felt_cost(r, pref) + legs.time_s + float(legs.penalties @ pref.multipliers())
        return {"minutes": round((r.time_s + legs.time_s) / 60, 1), "length_m": round(r.length_m + legs.length_m),
                "felt_minutes": {k: round(v / 60, 1) for k, v in pen.items()},
                "felt_total_minutes": round(felt / 60, 1),
                "priority_region_share": round(len(r.weights_won) / len(simplex_grid()), 2) if r.weights_won else None,
                "geometry": self._line(r, legs)}

    def _section(self, d: SectionDelta, ev: dict[int, dict], pref: Preferences, done: dict) -> dict:
        """One place where a route leaves the fastest one. Many options leave it at the same
        place in the same way, so each distinct place is described once per request."""
        s = d.section
        key = (s.a_edges, s.b_edges)
        if key in done:
            return done[key]
        done[key] = out = self._describe_section(d, ev, pref)
        return out

    def _describe_section(self, d: SectionDelta, ev: dict[int, dict], pref: Preferences) -> dict:
        s = d.section
        here = self._stretch(s.a_edges, s.start_node)
        return {"at": here[len(here) // 2],  # where to pin this place on the map
                "extra_minutes": round(d.time_s / 60, 1),
                "felt_minutes_saved": {k: round(-v / 60, 1) for k, v in d.felt_s.items()},
                "net_felt_minutes_saved": round(-d.net_s / 60, 1),
                "this_route": {"length_m": round(s.a_length_m), "geometry": here,
                               "conditions": self._conditions(s.a_edges, ev, pref)},
                "fastest_route": {"length_m": round(s.b_length_m), "geometry": self._stretch(s.b_edges, s.start_node),
                                  "conditions": self._conditions(s.b_edges, ev, pref)}}

    @staticmethod
    def _only_option(route: dict, steps: list[dict]) -> dict:
        """A successful trip with a single way to walk it, in the same shape as any other
        plan: one option that is both the fastest and the recommendation, winning the whole
        priorities triangle."""
        option = {"id": 0, **route, "is_fastest": True, "is_recommended": True, "priority_region_share": 1.0, "steps": steps}
        triangle = [[round(a, 2), round(b, 2), round(c, 2), 0] for a, b, c in simplex_grid()]
        return {"fastest": route, "recommended": route, "recommended_is_fastest": True, "alternatives": [],
                "what_would_it_take": None, "route_options": [option], "triangle": triangle}

    def _comparison(self, r: Route, fastest: Route, pref: Preferences, deltas: list[SectionDelta],
                    ev: dict[int, dict], done: dict) -> dict:
        """How ``r`` differs from the fastest route: totals, the crux and the other places."""
        return {"extra_minutes": round((r.time_s - fastest.time_s) / 60, 1),
                "felt_minutes_saved": round(felt_gain(r, fastest, pref) / 60, 1),
                "places_that_differ": len(deltas),
                "crux": self._section(deltas[0], ev, pref, done) if deltas else None,
                "other_places": [self._section(x, ev, pref, done) for x in deltas[1:]]}

    def _options(self, rs, pref: Preferences, legs: Legs, shown: Route) -> tuple[list[dict], list[list[float]]]:
        """Every route that would be shown somewhere on the priorities triangle, and the
        triangle itself: for each grid point ``[w_cross, w_slope, w_sidewalk, option id]``.
        The fastest route is always option 0."""
        options: list[Route] = [rs.fastest]
        index = {rs.fastest.key: 0}
        triangle = []
        for winner in rs.routes:
            for w in winner.weights_won:
                r = shown_route(winner, rs.fastest, pref.at(w))
                if r.key not in index:
                    index[r.key] = len(options)
                    options.append(r)
                triangle.append([round(w[0], 2), round(w[1], 2), round(w[2], 2), index[r.key]])
        if shown.key not in index:  # the user's own priorities need not be a grid point
            index[shown.key] = len(options)
            options.append(shown)
        # every option's places first, so the evidence behind all of them is read in one go
        deltas = [sorted(section_deltas(self.net, r, rs.fastest, pref), key=lambda x: x.net_s) if i else []
                  for i, r in enumerate(options)]
        # (guidance needs every edge of every option; the places need only their own)
        ev = self.evidence(e for r in options for e in r.edge_ids)
        done: dict = {}
        out = []
        for i, r in enumerate(options):
            o = {"id": i, **self._route(r, pref, legs), "is_fastest": i == 0, "is_recommended": r.key == shown.key}
            o["priority_region_share"] = round(sum(1 for x in triangle if x[3] == i) / max(len(triangle), 1), 2)
            if i:
                o["compared_with_fastest"] = self._comparison(r, rs.fastest, pref, deltas[i], ev, done)
            o["steps"] = self._steps(r, legs, ev)
            out.append(o)
        return out, sorted(triangle)

    # ------------------------------------------------------------------ main entry
    def plan(self, origin: tuple[float, float], dest: tuple[float, float], req: Requirements = Requirements(),
             pref: Preferences = Preferences()) -> dict:
        """Plan a trip between two (lon, lat) points."""
        out: dict = {"penalty_version": PENALTY_VERSION, "disclaimer": DISCLAIMER,
                     "requirements": asdict(req), "priorities": dict(zip(PENALTY_DIMS, pref.weights))}
        usable = self.net.feasible(req)
        snaps = []
        for name, pt in (("origin", origin), ("destination", dest)):
            s = self.snap(*pt, usable)
            meets = s is not None
            if s is None:  # no walkway in range meets the requirements: place it on the nearest one and say so
                s = self.snap(*pt)
            if s is None:
                closed = self.closed_area_at(*pt)
                where = f" The {name} is inside {closed}, which is not open to the public." if closed else ""
                return {**out, "status": "not_near_network",
                        "message": f"There is no public walkway within {SNAP_MAX_M:.0f} m of the {name}.{where}" if closed
                        else f"There is no mapped walkway within {SNAP_MAX_M:.0f} m of the {name}."}
            out[name] = {"snap_distance_m": round(s.distance_m, 1), "walkway_meets_requirements": meets,
                         "nearer_walkway_passed_over": meets and not s.nearest_edge_meets_requirements}
            closed = self.closed_area_at(*pt)
            if closed:
                out[name]["inside_closed_area"] = closed
            snaps.append(s)
        so, sd = snaps
        blocked_ends = [s.edge for s in snaps if not usable[s.edge]]

        if so.edge == sd.edge and not blocked_ends:
            g = self._geom[so.edge]
            share = abs(so.along_m - sd.along_m) / g.length if g.length > 0 else 0.0
            t_vec = self.net.time_s(pref.walk_speed_mps)
            xy = shapely.get_coordinates(substring(g, so.along_m, sd.along_m))
            legs = Legs(float(t_vec[so.edge] * share), float(g.length * share),
                        self.net.pen()[so.edge] * share, xy, xy[-1:])
            here = Route((), (int(self._geom_u[so.edge]),), 0.0, 0.0, dict.fromkeys(PENALTY_DIMS, 0.0))
            return {**out, "status": "ok", **self._only_option(self._route(here, pref, legs), self._steps(here, legs, {})),
                    "note": "Origin and destination are on the same stretch of walkway."}

        ends = None if blocked_ends else self._endpoints(so, sd, req, pref)
        if ends is None:
            # nothing meets the requirements; look for the smallest compromise between the
            # nearest ends of the two walkways
            t_vec = self.net.time_s(pref.walk_speed_mps)
            o, d = (min(dict.fromkeys((int(self._geom_u[s.edge]), int(self._geom_v[s.edge]))),
                        key=lambda n, s=s: self._leg(s, n, t_vec, self.net.pen())[0]) for s in snaps)
            out["status"] = "no_feasible_route"
            rel = minimal_relaxation(self.net, o, d, req, pref) if o != d else None
            if rel is None:
                return {**out, "relaxation": None,
                        "message": "No route satisfies the requirements, and no single compromise makes one possible."}
            outside = [*blocked_ends, *rel.newly_used_edges]
            ev = self.evidence(outside)
            return {**out, "relaxation": {"compromise": rel.label, "requirements": asdict(rel.requirements),
                                          "route": self._route(rel.route_set.fastest, pref, self._legs(so, sd, o, d, pref)),
                                          "where_it_applies": self._conditions(outside, ev, pref),
                                          "edges_outside_original_requirements": len(outside)}}
        o, d = ends
        legs = self._legs(so, sd, o, d, pref)
        if o == d:
            here = Route((), (o,), 0.0, 0.0, dict.fromkeys(PENALTY_DIMS, 0.0))
            return {**out, "status": "ok", **self._only_option(self._route(here, pref, legs), self._steps(here, legs, {}))}
        rs = self.net.route_set(o, d, req, pref)
        out["status"] = rs.status
        fastest, best = rs.fastest, rs.recommended
        shown = shown_route(best, fastest, pref)
        show = shown.key != fastest.key
        out["fastest"] = self._route(fastest, pref, legs)
        out["recommended"] = self._route(shown, pref, legs)
        out["recommended_is_fastest"] = not show
        if best.key != fastest.key and not show:
            out["note"] = (f"A slightly different route scores under {WORTH_SHOWING_MIN_GAIN_S:.0f} felt seconds better "
                           "than the fastest; the two are treated as equivalent.")
        out["route_options"], out["triangle"] = self._options(rs, pref, legs, shown)
        if show:  # the recommended route's comparison, also reachable through its option
            out["comparison"] = next(o for o in out["route_options"] if o["is_recommended"])["compared_with_fastest"]
        out["alternatives"] = [self._route(r, pref, legs) for r in rs.routes
                               if r.key not in (shown.key, fastest.key) and abs(felt_gain(r, shown, pref)) >= WORTH_SHOWING_MIN_GAIN_S]
        tp = tipping_point(rs, pref)
        out["what_would_it_take"] = None if tp is None else {
            "shift_percentage_points": tp.shift, "share_of_priority_moved": round(tp.distance, 2),
            "then": self._route(tp.route, pref, legs)}
        return out
