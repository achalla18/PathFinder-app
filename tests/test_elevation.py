import numpy as np
import pytest
from shapely.geometry import LineString

from pathfinder.graph.elevation import (
    WINDOW_M,
    build_profiles,
    continuations,
    profile_grades,
    sustained_grade,
)

D = np.arange(-5.0, 105.1, 1.0)  # a 100 m edge with 5 m of continuation at each end


def _lines(z):
    return np.vstack([z, z, z])


def test_flat_profile_has_zero_grade():
    # regression: zero-padded smoothing once produced >100% grades at edge ends
    r = profile_grades(D, _lines(np.full(len(D), 50.0)), 100.0)
    assert r["avg"] == pytest.approx(0.0) and r["max"] == pytest.approx(0.0) and r["rise"] == pytest.approx(0.0)


def test_uniform_slope_is_recovered():
    r = profile_grades(D, _lines(10 + 0.05 * D), 100.0)
    assert r["avg"] == pytest.approx(5.0, abs=0.1)
    assert r["max"] == pytest.approx(5.0, abs=0.1)
    assert r["rise"] == pytest.approx(5.0, abs=0.1)
    assert r["spread"] == pytest.approx(0.0, abs=1e-6)


def test_single_pixel_spike_is_suppressed():
    z = np.full(len(D), 20.0)
    z[30] = 23.0  # a wall/curb pixel
    assert profile_grades(D, _lines(z), 100.0)["max"] < 1.0


def test_short_steep_stretch_is_found_but_not_a_two_metre_blip():
    z = np.full(len(D), 20.0)
    z[D >= 40] += np.minimum(D[D >= 40] - 40, 12) * 0.10  # 12 m at 10 %
    assert sustained_grade(D, z, 0.0, 100.0) == pytest.approx(10.0, abs=0.5)
    blip = np.full(len(D), 20.0)
    blip[D >= 40] += np.minimum(D[D >= 40] - 40, 2) * 0.10  # a 2 m curb-ramp-sized rise
    assert sustained_grade(D, blip, 0.0, 100.0) < 4.0


def test_missing_data_makes_grade_unknown():
    z = 10 + 0.05 * D
    z[: len(z) // 2] = np.nan  # half the profile outside the DTM
    assert np.isnan(profile_grades(D, _lines(z), 100.0)["max"])


def test_intentionally_masked_samples_do_not_count_against_coverage():
    z = 10 + 0.05 * D
    masked = np.zeros(len(z), dtype=bool)
    masked[15:35] = True
    z[masked] = np.nan
    r = profile_grades(D, _lines(z), 100.0, masked)
    assert r["max"] == pytest.approx(5.0, abs=0.2)


def test_wall_like_step_makes_the_grade_a_discontinuity_not_a_number():
    # a sidewalk running off a deck: 13.6 m drop between two samples
    z = np.where(D < 50, 6.8, -6.8)
    r = profile_grades(D, _lines(z), 100.0)
    assert r["discontinuity"] and np.isnan(r["max"])


def test_wall_beyond_the_edge_only_shortens_the_window():
    z = (10 + 0.02 * D).copy()
    z[D > 103] -= 12.0  # the neighbor drops off a deck 3 m past the end of this edge
    r = profile_grades(D, _lines(z), 100.0)
    assert not r["discontinuity"]
    assert r["max"] == pytest.approx(2.0, abs=0.2)


def test_lines_that_disagree_report_their_spread():
    z = 10 + 0.05 * D
    left = z.copy()
    left[30:45] += np.linspace(0, 2.5, 15)  # a bank or wall on one side only
    r = profile_grades(D, np.vstack([left, z, z]), 100.0)
    assert r["max"] == pytest.approx(5.0, abs=0.1)  # the median line is unaffected
    assert r["spread"] > 10


def test_too_little_profile_claims_nothing():
    d = np.array([0.0, 1.0, 2.0])
    assert np.isnan(sustained_grade(d, np.array([1.0, 1.3, 1.6]), 0.0, 2.0))


def test_short_edge_is_measured_along_the_edge_that_continues_it():
    # a 2 m edge between a long straight neighbor and a side branch at a right angle
    geoms = np.array([LineString([(0, 0), (50, 0)]), LineString([(50, 0), (52, 0)]),
                      LineString([(52, 0), (100, 0)]), LineString([(52, 0), (52, 40)])], dtype=object)
    u, v = np.array([1, 2, 3, 3]), np.array([2, 3, 4, 5])
    cont, cont_end = continuations(u, v, geoms)
    assert cont[1].tolist() == [0, 2]  # straight on both ways, never the side branch
    assert cont_end[1].tolist() == [1, 0]
    prof = build_profiles(geoms, cont, cont_end)
    own = prof["owner"] == 1
    assert prof["dist"][own].min() == pytest.approx(-WINDOW_M / 2)
    assert prof["dist"][own].max() == pytest.approx(2 + WINDOW_M / 2)
    assert np.allclose(prof["y"][own], 0.0)  # every sample is on the straight neighbors
    assert np.all(np.diff(prof["x"][own]) > 0)


def test_profile_continues_around_a_corner_and_stops_at_a_dead_end():
    geoms = np.array([LineString([(0, 0), (20, 0)]), LineString([(20, 0), (20, 3)])], dtype=object)
    cont, cont_end = continuations(np.array([1, 2]), np.array([2, 3]), geoms)
    assert cont[1, 0] == 0 and cont_end[1, 0] == 1  # the only way on is around the corner
    assert cont[1, 1] == -1  # nothing beyond node 3
    prof = build_profiles(geoms, cont, cont_end)
    own = prof["owner"] == 1
    assert prof["dist"][own].min() == pytest.approx(-WINDOW_M / 2) and prof["dist"][own].max() == pytest.approx(3.0)


def test_profile_runs_through_a_chain_of_short_pieces():
    # 1 m edges, as at a junction: the middle one needs three neighbors each way for half a window
    xs = [0, 20, 21, 22, 23, 24, 25, 26, 27, 47]
    geoms = np.array([LineString([(a, 0), (b, 0)]) for a, b in zip(xs[:-1], xs[1:])], dtype=object)
    u, v = np.arange(len(geoms)), np.arange(1, len(geoms) + 1)
    prof = build_profiles(geoms, *continuations(u, v, geoms))
    own = prof["owner"] == 4  # the edge from x=23 to x=24
    assert prof["dist"][own].min() == pytest.approx(-WINDOW_M / 2) and prof["dist"][own].max() == pytest.approx(1 + WINDOW_M / 2)
    assert np.allclose(np.diff(prof["x"][own]), 1.0)  # evenly spaced across the pieces, no repeats
    assert set(prof["src"][own]) == {0, 1, 2, 3, 4, 5, 6, 7, 8}
