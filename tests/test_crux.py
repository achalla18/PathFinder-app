"""Checkpoint recovery and the fixed sample of the crux experiment."""

import json

import pandas as pd

from pathfinder.analysis.crux import OD_SNAP_MAX_M, PROFILES, fix_sample, load_checkpoint, resolve_sample


def _rows(od: int, profiles=PROFILES) -> list[str]:
    return [json.dumps({"od": od, "profile": p, "status": "ok"}) for p in profiles]


def test_checkpoint_keeps_only_od_pairs_with_every_profile(tmp_path):
    """Regression: an OD pair with any one profile row used to count as done."""
    partial = list(PROFILES)[:2]
    lines = _rows(0) + _rows(1, partial) + _rows(2) + ['{"od": 3, "profile": "everyday_d']  # truncated write
    ckpt = tmp_path / "ckpt.jsonl"
    ckpt.write_text("\n".join(lines))

    rows, done = load_checkpoint(ckpt, range(4))

    assert done == {0, 2}
    assert len(rows) == 2 * len(PROFILES)
    assert {(r["od"], r["profile"]) for r in rows} == {(od, p) for od in (0, 2) for p in PROFILES}


def test_checkpoint_ignores_od_pairs_outside_the_sample(tmp_path):
    ckpt = tmp_path / "ckpt.jsonl"
    ckpt.write_text("\n".join(_rows(0) + _rows(7)) + "\n")
    rows, done = load_checkpoint(ckpt, range(2))
    assert done == {0}
    assert len(rows) == len(PROFILES)


def _nodes(xy: dict[int, tuple[float, float]], off_network=()) -> pd.DataFrame:
    return pd.DataFrame([{"node_id": n, "x": 397000.0 + x, "y": 137000.0 + y, "component": int(n in off_network)}
                         for n, (x, y) in xy.items()])


def test_the_fixed_sample_finds_the_same_trips_on_a_rebuilt_graph():
    before = _nodes({1: (0, 0), 2: (500, 0), 3: (0, 900), 4: (700, 900)})
    pairs = pd.DataFrame({"origin": [1, 3], "dest": [2, 4], "euclid_m": [500.0, 700.0], "band": ["400-800", "400-800"]})
    sample = fix_sample(before, pairs)
    # a rebuild renumbers every node and moves them a little; node 4's walkway is gone
    after = _nodes({10: (2, 1), 20: (501, -1), 30: (0, 903), 40: (700 + 2 * OD_SNAP_MAX_M, 900)})
    ods = resolve_sample(after, sample)
    assert list(ods.index) == [0]  # trip 1 lost an end; trip 0 keeps its number
    assert (ods.loc[0, "origin"], ods.loc[0, "dest"]) == (10, 20)
    assert abs(ods.loc[0, "euclid_m"] - 499.0) < 0.1  # measured between the nodes actually used


def test_the_fixed_sample_only_lands_on_the_main_network():
    sample = fix_sample(_nodes({1: (0, 0), 2: (500, 0)}),
                        pd.DataFrame({"origin": [1], "dest": [2], "euclid_m": [500.0], "band": ["400-800"]}))
    # the nearest node to the destination is on an island; the main network is too far away
    after = _nodes({10: (0, 0), 20: (500, 0), 30: (400, 0)}, off_network={20})
    assert resolve_sample(after, sample).empty
