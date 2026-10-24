"""Route sets across the preference simplex, with exact ellipse pruning.

Algorithm for one request (origin o, destination d, requirements R, preferences P)
-------------------------------------------------------------------------------
1. Feasible edges = ``cost.feasible_mask(R)``. Requirements are never traded off.
2. **Fastest feasible route** by Dijkstra on walking time inside an ellipse with foci o, d.
   Exactness: every edge's time is ≥ its length / speed, so a path faster than the one found
   (time T*) has length < T*·speed. If T*·speed ≤ the ellipse's length bound, all such paths
   lie inside the ellipse, so the result is globally optimal; otherwise the ellipse grows.
3. **Budget**: T_max = T* + P.budget_s. The ellipse with major axis T_max·speed contains every
   path that satisfies the budget (Euclidean distance ≤ path length), so pruning to it loses
   nothing. Inside it, an edge is kept only if the fastest way from o to it, along it, and on
   to d fits in T_max: every path within the budget consists of such edges, so this loses
   nothing either, and it leaves about a third of the ellipse (two extra searches).
4. **Weight grid**: for each w on a simplex grid (step 0.1 → 66 points), minimize felt cost.
   If the minimizer breaks the budget, Lagrangian relaxation adds λ·time and finds the
   smallest λ that satisfies it. The search is between the over-budget minimizer and the
   fastest route: try the λ at which the two cost the same; a path cheaper than both there
   replaces whichever of them it sides with; when there is none, the within-budget one is the
   answer (Handler & Zang 1980). That takes a few searches, where bisecting λ took a dozen.
   (Like all weighted-sum methods this finds routes on the convex hull of the time/penalty
   trade-off; documented limitation.)
5. Distinct routes are kept with the set of weights for which each wins — the regions of the
   Tradeoff Triangle.
"""

from __future__ import annotations

import time as _time
import warnings
from dataclasses import dataclass, field
from functools import lru_cache

import geopandas as gpd
import igraph as ig
import numpy as np
import pandas as pd

from ..config import GRAPH_DIR
from .cost import PENALTIES, PENALTY_DIMS, Preferences, Requirements, edge_penalties, feasible_mask

_GRADED_FOR_ROUTING = [
    "crossing_control", "signal_lpi", "signal_hawk", "crossed_width_m", "crossed_speed_mph",
    "kerb_a", "kerb_b", "ramp_check_a", "ramp_check_b", "step_free", "surface_class", "sidewalk_presence", "grade_max_pct",
    "sidewalk_condition",
]
# Everything cost.py / router.py read; keep in sync when the cost model uses a new column.
ROUTING_COLUMNS = [
    "edge_id", "u", "v", "length_m", "kind", "representation", "component", "osm_way_id",
    "crossing_total_length_m", "ps_obstacles", "ps_surface_problems",
    *[c for a in _GRADED_FOR_ROUTING for c in (a, f"{a}_status")],
    *[f"xing_{end}{suffix}" for end in ("u", "v") for suffix in ("", "_control", "_width_m", "_kerb")],
]

ELEVATOR_WAIT_S = 45.0
STEPS_TIME_FACTOR = 2.0
MIN_ELLIPSE_SLACK_M = 200.0
# The budget search ends by itself when no path beats both of its ends; each step replaces one
# end with a different path, so this is a guard against a float tie, never the usual exit.
MAX_BUDGET_STEPS = 40


def simplex_grid(step: float = 0.1) -> list[tuple[float, float, float]]:
    n = int(round(1 / step))
    return [(i / n, j / n, (n - i - j) / n) for i in range(n + 1) for j in range(n + 1 - i)]


@dataclass
class Route:
    edge_ids: tuple[int, ...]
    node_ids: tuple[int, ...]
    time_s: float
    length_m: float
    penalties: dict[str, float]  # raw full-weight felt seconds per dimension
    weights_won: list[tuple[float, float, float]] = field(default_factory=list)

    @property
    def key(self) -> tuple[int, ...]:
        return self.edge_ids


@dataclass
class RouteSet:
    origin: int
    destination: int
    fastest: Route | None
    routes: list[Route]
    t_max_s: float | None
    subgraph_edges: int
    elapsed_s: float
    status: str  # "ok" | "no_feasible_route" | "same_node"
    recommended: Route | None = None  # the route for the user's own weights


class Network:
    """The routable graph plus per-edge penalty and time arrays (aligned with ``edges``)."""

    def __init__(self, edges: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame):
        self.edges = edges.reset_index(drop=True)
        self.nodes = nodes.reset_index(drop=True)
        self.node_index = pd.Series(np.arange(len(self.nodes)), index=self.nodes["node_id"].to_numpy())
        self.node_ids = self.nodes["node_id"].to_numpy()
        self.x = self.nodes["x"].to_numpy()
        self.y = self.nodes["y"].to_numpy()
        self.eu = self.node_index.loc[self.edges["u"].to_numpy()].to_numpy()
        self.ev = self.node_index.loc[self.edges["v"].to_numpy()].to_numpy()
        self.length = self.edges["length_m"].to_numpy(dtype=float)
        self.kind = self.edges["kind"].to_numpy()
        P = edge_penalties(self.edges, PENALTIES)
        self._pen = P[[f"P_{d}" for d in PENALTY_DIMS]].to_numpy(dtype=float)
        self.graph = ig.Graph(n=len(self.nodes), edges=list(zip(self.eu, self.ev)), directed=False)
        self._feasible_cache: dict[Requirements, np.ndarray] = {}

    @classmethod
    def load(cls, graph_dir=GRAPH_DIR, lean: bool = True) -> "Network":
        """Load the routable graph. ``lean`` reads only the columns routing needs (no
        geometry): ~5x less memory, which matters when many worker processes each hold a
        copy. Use ``lean=False`` when route geometry is required."""
        if not lean:
            return cls(gpd.read_parquet(graph_dir / "edges.parquet"), gpd.read_parquet(graph_dir / "nodes.parquet"))
        import pyarrow.parquet as pq

        missing = set(ROUTING_COLUMNS) - set(pq.read_schema(graph_dir / "edges.parquet").names)
        if missing:  # a graph built before the cost model changed must be rebuilt, not half-read
            raise RuntimeError(f"edges.parquet lacks {sorted(missing)}; rerun `pf build-graph` and `pf enrich`")
        edges = pd.read_parquet(graph_dir / "edges.parquet", columns=ROUTING_COLUMNS)
        nodes = pd.read_parquet(graph_dir / "nodes.parquet", columns=["node_id", "x", "y", "component"])
        return cls(edges, nodes)

    # ------------------------------------------------------------------ helpers
    def feasible(self, req: Requirements) -> np.ndarray:
        if req not in self._feasible_cache:
            self._feasible_cache[req] = feasible_mask(self.edges, req)
        return self._feasible_cache[req]

    def pen(self) -> np.ndarray:
        """Per-edge felt seconds at full weight, one column per PENALTY_DIMS."""
        return self._pen

    def time_s(self, speed: float) -> np.ndarray:
        t = self.length / speed
        t = np.where(np.isin(self.kind, ["steps"]), t * STEPS_TIME_FACTOR, t)
        return np.where(self.kind == "elevator", t + ELEVATOR_WAIT_S, t)

    def _ellipse_edges(self, o: int, d: int, max_len: float, feasible: np.ndarray) -> np.ndarray:
        dn = np.hypot(self.x - self.x[o], self.y - self.y[o]) + np.hypot(self.x - self.x[d], self.y - self.y[d])
        inside = dn <= max_len
        return np.nonzero(feasible & inside[self.eu] & inside[self.ev])[0]

    def _sub(self, eids: np.ndarray) -> tuple[ig.Graph, np.ndarray, np.ndarray, np.ndarray]:
        """The graph on these edges, its node positions (sorted), and each edge's two ends as
        positions in that graph."""
        verts = np.unique(np.concatenate([self.eu[eids], self.ev[eids]]))
        lu, lv = np.searchsorted(verts, self.eu[eids]), np.searchsorted(verts, self.ev[eids])
        g = ig.Graph(n=len(verts), edges=np.column_stack([lu, lv]).tolist(), directed=False)
        return g, verts, lu, lv

    @staticmethod
    def _local(verts: np.ndarray, node: int) -> int | None:
        """Where a node sits in a subgraph, or None if it is not in it."""
        i = int(np.searchsorted(verts, node))
        return i if i < len(verts) and verts[i] == node else None

    @staticmethod
    def _path(g: ig.Graph, s: int, t: int, w: np.ndarray) -> list[int] | None:
        with warnings.catch_warnings():
            # unreachable targets are expected while the fastest-route ellipse is growing
            warnings.simplefilter("ignore", RuntimeWarning)
            p = g.get_shortest_paths(s, to=t, weights=w.tolist(), output="epath")[0]
        return p if p else None

    def _make_route(self, eids: np.ndarray, local_path: list[int], o: int, t_vec: np.ndarray, P: np.ndarray) -> Route:
        ge = eids[np.asarray(local_path)]
        nodes = [o]
        for e in ge:
            a, b = self.eu[e], self.ev[e]
            nodes.append(b if a == nodes[-1] else a)
        pen = P[ge].sum(axis=0)
        return Route(
            edge_ids=tuple(int(e) for e in ge),
            node_ids=tuple(int(self.node_ids[n]) for n in nodes),
            time_s=float(t_vec[ge].sum()),
            length_m=float(self.length[ge].sum()),
            penalties={d: float(v) for d, v in zip(PENALTY_DIMS, pen)},
        )

    def _fastest(self, o: int, d: int, feas: np.ndarray, t_vec: np.ndarray, speed: float,
                 P: np.ndarray) -> Route | None:
        """Fastest feasible route between two node positions, exact via ellipse growth."""
        de = float(np.hypot(self.x[o] - self.x[d], self.y[o] - self.y[d]))
        bound = max(1.5 * de, de + MIN_ELLIPSE_SLACK_M)
        while True:
            eids = self._ellipse_edges(o, d, bound, feas)
            if len(eids):
                g, verts, _, _ = self._sub(eids)
                so, sd = self._local(verts, o), self._local(verts, d)
                if so is not None and sd is not None:
                    p = self._path(g, so, sd, t_vec[eids])
                    if p is not None:
                        r = self._make_route(eids, p, o, t_vec, P)
                        if r.time_s * speed <= bound:
                            return r
                        bound = r.time_s * speed + 1.0  # guaranteed to contain the optimum
                        continue
            if bound > 60_000:  # larger than DC: genuinely no feasible route
                return None
            bound *= 2

    def fastest(self, origin_node: int, dest_node: int, req: Requirements = Requirements(),
                pref: Preferences = Preferences()) -> Route | None:
        """Fastest feasible route only (no route set). None if there is none."""
        o, d = int(self.node_index[origin_node]), int(self.node_index[dest_node])
        return self._fastest(o, d, self.feasible(req), self.time_s(pref.walk_speed_mps), pref.walk_speed_mps,
                             self.pen())

    # ------------------------------------------------------------------ main entry
    def route_set(self, origin_node: int, dest_node: int, req: Requirements = Requirements(),
                  pref: Preferences = Preferences(), grid_step: float = 0.1) -> RouteSet:
        t0 = _time.perf_counter()
        o, d = int(self.node_index[origin_node]), int(self.node_index[dest_node])
        if o == d:
            return RouteSet(origin_node, dest_node, None, [], None, 0, 0.0, "same_node")
        feas = self.feasible(req)
        speed = pref.walk_speed_mps
        t_vec = self.time_s(speed)
        P = self.pen()

        # ---- 2. fastest feasible, exact via ellipse growth
        fastest = self._fastest(o, d, feas, t_vec, speed, P)
        if fastest is None:
            return RouteSet(origin_node, dest_node, None, [], None, 0, _time.perf_counter() - t0, "no_feasible_route")

        # ---- 3. budget ellipse
        t_max = fastest.time_s + pref.budget_s
        eids = self._ellipse_edges(o, d, t_max * speed + 1.0, feas)
        g, verts, lu, lv = self._sub(eids)
        so, sd = self._local(verts, o), self._local(verts, d)
        te = t_vec[eids]
        # keep only edges some within-budget path could use (see the module docstring)
        w_time = te.tolist()
        from_o = np.asarray(g.distances(so, weights=w_time)[0])
        from_d = np.asarray(g.distances(sd, weights=w_time)[0])
        keep = np.minimum(from_o[lu] + from_d[lv], from_o[lv] + from_d[lu]) + te <= t_max + 1e-6
        eids = eids[keep]
        g, verts, _, _ = self._sub(eids)
        so, sd = self._local(verts, o), self._local(verts, d)
        te = t_vec[eids]
        Pe = P[eids]
        quickest = self._path(g, so, sd, te)  # within budget by construction: where λ → ∞ leads

        # ---- 4. weight grid with Lagrangian budget enforcement
        def best_for(w: tuple[float, float, float]) -> list[int]:
            felt = Pe @ pref.multipliers(w)

            def measure(path: list[int]) -> tuple[float, float]:
                i = np.asarray(path)
                return float(te[i].sum()), float(felt[i].sum())

            over = self._path(g, so, sd, te + felt)
            t_over, f_over = measure(over)
            if t_over <= t_max + 1e-6:
                return over
            within = quickest
            t_in, f_in = measure(within)
            for _ in range(MAX_BUDGET_STEPS):
                if t_over - t_in <= 1e-9:
                    break
                rate = (f_in - f_over) / (t_over - t_in)  # 1 + λ at which the two cost the same
                tied = f_over + rate * t_over
                mid = self._path(g, so, sd, felt + rate * te)
                t_mid, f_mid = measure(mid)
                if f_mid + rate * t_mid >= tied - 1e-9 * max(1.0, abs(tied)):
                    break  # nothing beats both: `within` is the route at the smallest λ
                if t_mid <= t_max + 1e-6:
                    within, t_in, f_in = mid, t_mid, f_mid
                else:
                    over, t_over, f_over = mid, t_mid, f_mid
            return within

        found: dict[tuple[int, ...], Route] = {}
        for w in simplex_grid(grid_step):
            r = self._make_route(eids, best_for(w), o, t_vec, P)
            found.setdefault(r.key, r).weights_won.append(w)
        rec = self._make_route(eids, best_for(pref.weights), o, t_vec, P)
        recommended = found.setdefault(rec.key, rec)

        routes = sorted(found.values(), key=lambda r: -len(r.weights_won))
        return RouteSet(origin_node, dest_node, fastest, routes, t_max, len(eids),
                        _time.perf_counter() - t0, "ok", recommended)


@lru_cache(maxsize=1)
def default_network() -> Network:
    return Network.load()
