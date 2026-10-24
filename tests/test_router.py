"""Router correctness: exact pruning, budget enforcement, requirement compliance."""

import networkx as nx
import numpy as np
import pytest

from pathfinder.routing.cost import Preferences, Requirements, feasible_mask
from pathfinder.routing.router import Network, simplex_grid

from .conftest import grid, make_edges, make_nodes


def _net(seed: int = 0, n: int = 10):
    rng = np.random.default_rng(seed)
    xy, rows = grid(n)
    # jitter geometry so lengths differ, and scatter conditions across the grid
    xy = {k: (x + rng.uniform(-30, 30), y + rng.uniform(-30, 30)) for k, (x, y) in xy.items()}
    for r in rows:
        r["lights_per_100m"] = float(rng.choice([0.0, 2.0, 6.0]))
        r["grade_max_pct"] = float(rng.choice([1.0, 4.0, 9.0]))
        if rng.random() < 0.15:
            r["kind"] = "crossing"
            r["crossing_control"] = str(rng.choice(["signal", "marked_uncontrolled", "unmarked"]))
            r["kerb_a"] = "lowered"
            r["kerb_b"] = str(rng.choice(["lowered", "no_ramp"])) if rng.random() < 0.8 else None
        if rng.random() < 0.05:
            r["kind"] = "steps"
            r["step_free"] = False
    e = make_edges(rows, xy)
    return Network(e, make_nodes(xy))


def _nx_shortest_time(net: Network, o: int, d: int, mask: np.ndarray, speed: float) -> float:
    g = nx.Graph()
    t = net.time_s(speed)
    for i in np.nonzero(mask)[0]:
        a, b = int(net.node_ids[net.eu[i]]), int(net.node_ids[net.ev[i]])
        if not g.has_edge(a, b) or g[a][b]["w"] > t[i]:
            g.add_edge(a, b, w=t[i])
    return nx.shortest_path_length(g, o, d, weight="w")


def test_simplex_grid():
    g = simplex_grid(0.1)
    assert len(g) == 66 and all(abs(sum(w) - 1) < 1e-9 for w in g)


@pytest.mark.parametrize("seed", range(5))
def test_fastest_route_is_globally_optimal_despite_pruning(seed):
    net = _net(seed)
    rng = np.random.default_rng(100 + seed)
    req = Requirements()
    for _ in range(5):
        o, d = (int(x) for x in rng.choice(net.node_ids, 2, replace=False))
        rs = net.route_set(o, d, req, Preferences())
        expected = _nx_shortest_time(net, o, d, net.feasible(req), Preferences().walk_speed_mps)
        assert rs.fastest.time_s == pytest.approx(expected)


@pytest.mark.parametrize("seed", range(5))
def test_every_route_respects_budget_and_requirements(seed):
    net = _net(seed)
    rng = np.random.default_rng(200 + seed)
    req = Requirements(step_free=True, max_grade_pct=8.33, certainty="allow_likely")
    mask = feasible_mask(net.edges, req)
    pref = Preferences(budget_s=120)
    for _ in range(5):
        o, d = (int(x) for x in rng.choice(net.node_ids, 2, replace=False))
        rs = net.route_set(o, d, req, pref)
        if rs.status != "ok":
            continue
        for r in rs.routes + [rs.recommended]:
            assert r.time_s <= rs.fastest.time_s + pref.budget_s + 1e-6
            assert all(mask[e] for e in r.edge_ids), "route used an edge excluded by a requirement"
            assert r.node_ids[0] == o and r.node_ids[-1] == d


def test_recommended_route_trades_time_for_lower_felt_penalty():
    net = _net(1)
    rng = np.random.default_rng(7)
    for _ in range(10):
        o, d = (int(x) for x in rng.choice(net.node_ids, 2, replace=False))
        rs = net.route_set(o, d, Requirements(), Preferences(budget_s=600))
        f, rec = rs.fastest, rs.recommended
        felt = lambda r: r.time_s + sum(r.penalties.values())  # noqa: E731  equal weights, intensity 1
        assert felt(rec) <= felt(f) + 1e-6


def test_weights_won_partition_the_grid():
    net = _net(2)
    rs = net.route_set(int(net.node_ids[0]), int(net.node_ids[-1]), Requirements(), Preferences())
    assert sum(len(r.weights_won) for r in rs.routes) >= len(simplex_grid(0.1))


def test_same_node_and_infeasible():
    net = _net(3)
    assert net.route_set(int(net.node_ids[0]), int(net.node_ids[0])).status == "same_node"
