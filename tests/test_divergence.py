import numpy as np

from pathfinder.routing.divergence import divergent_share, sections

LEN = np.ones(20) * 10.0


def test_identical_routes_have_no_sections():
    assert sections((1, 2, 3), (0, 1, 2, 3), (1, 2, 3), (0, 1, 2, 3), LEN) == []


def test_single_detour_is_one_section():
    # A: 0-1-2-3-4 via edges 1,2,3,4 ; B shares first and last edge, detours in the middle
    a_e, a_n = (1, 2, 3, 4), (0, 1, 2, 3, 4)
    b_e, b_n = (1, 7, 8, 9, 4), (0, 1, 10, 11, 3, 4)
    s = sections(a_e, a_n, b_e, b_n, LEN)
    assert len(s) == 1
    assert s[0].a_edges == (2, 3) and s[0].b_edges == (7, 8, 9) and s[0].clean
    assert s[0].start_node == 1 and s[0].end_node == 3


def test_two_separate_detours():
    a_e, a_n = (1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)
    b_e, b_n = (11, 2, 3, 14, 5), (0, 1, 2, 3, 4, 5)
    assert len(sections(a_e, a_n, b_e, b_n, LEN)) == 2


def test_divergent_share():
    assert divergent_share((1, 2, 3, 4), (1, 9, 9, 4), LEN) == 0.5
