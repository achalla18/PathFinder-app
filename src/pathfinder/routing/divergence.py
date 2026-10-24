"""Where do two routes differ? (the Crux Map's core computation)

Two routes from the same origin to the same destination share some edges and diverge
elsewhere. A *divergence section* is a maximal run of route A's edges that route B does not
use, together with B's edges between the same two rejoin nodes. The Crux Map shows only
these sections; the crux is the section contributing the largest felt-cost difference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Section:
    a_edges: tuple[int, ...]
    b_edges: tuple[int, ...]
    start_node: int
    end_node: int
    a_length_m: float
    b_length_m: float
    clean: bool  # False when B does not traverse start->end in order (loops / reordering)


def sections(a_edges: tuple[int, ...], a_nodes: tuple[int, ...], b_edges: tuple[int, ...],
             b_nodes: tuple[int, ...], length: np.ndarray) -> list[Section]:
    """Divergence sections of A relative to B.

    ``*_nodes`` has one more element than ``*_edges`` (node i precedes edge i)."""
    if len(a_nodes) != len(a_edges) + 1 or len(b_nodes) != len(b_edges) + 1:
        raise ValueError("node sequences must be one longer than edge sequences")
    b_set = set(b_edges)
    b_pos = {}
    for i, n in enumerate(b_nodes):
        b_pos.setdefault(n, i)
    out: list[Section] = []
    i = 0
    while i < len(a_edges):
        if a_edges[i] in b_set:
            i += 1
            continue
        j = i
        while j < len(a_edges) and a_edges[j] not in b_set:
            j += 1
        s, t = a_nodes[i], a_nodes[j]
        ps, pt = b_pos.get(s), b_pos.get(t)
        clean = ps is not None and pt is not None and pt > ps
        b_run = tuple(b_edges[ps:pt]) if clean else ()
        a_run = tuple(a_edges[i:j])
        out.append(Section(a_run, b_run, s, t, float(length[list(a_run)].sum()),
                           float(length[list(b_run)].sum()) if b_run else float("nan"), clean))
        i = j
    return out


def divergent_share(a_edges: tuple[int, ...], b_edges: tuple[int, ...], length: np.ndarray) -> float:
    """Share of A's length not shared with B."""
    total = float(length[list(a_edges)].sum())
    if total == 0:
        return 0.0
    b_set = set(b_edges)
    return float(sum(length[e] for e in a_edges if e not in b_set) / total)
