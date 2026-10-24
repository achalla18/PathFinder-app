"""Invariants on the real citywide graph (skipped when artifacts are absent)."""

import pytest

from pathfinder import evidence as ev
from pathfinder.graph.build import GAP_TOLERANCE_M, SURFACE_GAP_MAX_M, SURFACE_GAP_REASON
from pathfinder.graph.elevation import REPEATABILITY_TOL_PCT, SUSPECT_GRADE_PCT
from pathfinder.graph.enrich import GRADED_ATTRS
from pathfinder.sources import REGION

pytestmark = pytest.mark.data


def test_graded_attributes_obey_evidence_rules(built_graph):
    for a in GRADED_ATTRS:
        ev.validate(built_graph, a)


def test_stairs_are_never_step_free(built_graph):
    steps = built_graph[built_graph["kind"].isin(["steps", "escalator"])]
    assert len(steps) > 0
    assert (steps["step_free"] == False).all()  # noqa: E712
    assert (steps["step_free_status"] == "verified").all()


def test_kerbs_are_identical_across_crossing_fragments(built_graph):
    cr = built_graph[built_graph["kind"] == "crossing"]
    per = cr.groupby("osm_way_id")[["kerb_a", "kerb_b", "kerb_a_status", "kerb_b_status"]].nunique(dropna=False)
    assert (per.max(axis=1) <= 1).all()


def test_structural_edges_never_have_verified_dtm_grade(built_graph):
    struct = built_graph["bridge"].notna() & ~built_graph["bridge"].isin(["no"])
    assert not (built_graph.loc[struct, "grade_max_pct_status"] == "verified").any()


def test_inferred_connectors_are_short_and_disclosed(built_graph):
    c = built_graph[built_graph["representation"] == "inferred_connector"]
    assert (c["kind"] == "connector").all()
    surface = c["class_reason"] == SURFACE_GAP_REASON
    # wide gaps are bridged only across a planimetric survey, and DC's has many
    assert surface.any() == bool(REGION.role("planimetrics"))
    assert (c.loc[~surface, "length_m"] <= GAP_TOLERANCE_M + 1e-6).all()
    # wider gaps are bridged only on evidence, and never beyond the stated maximum
    assert (c.loc[surface, "length_m"] > GAP_TOLERANCE_M - 0.5).all()
    assert (c.loc[surface, "length_m"] <= SURFACE_GAP_MAX_M + 1e-6).all()


def test_inferred_connectors_are_never_verified_step_free(built_graph):
    c = built_graph[built_graph["representation"] == "inferred_connector"]
    assert len(c) > 0
    assert (c["step_free_status"] == "inferred").all()


def test_absence_is_never_verification_outside_inventory_coverage(built_graph):
    out = built_graph[~built_graph["covered_planimetric"]]
    assert len(out) > 0
    by_absence = out["step_free_source"].fillna("").str.startswith("derived:absent")
    assert (out.loc[by_absence, "step_free_status"] == "inferred").all()
    from_polygons = out[out["sidewalk_presence_source"] == REGION.role("planimetrics")]
    # outside coverage the survey's silence proves nothing: it may say "both" or "at least one
    # side", never "one side" (the other side has none) or "none"
    assert from_polygons["sidewalk_presence"].isin(["both", "at_least_one_side"]).all()


def test_a_region_without_an_inventory_never_confirms_absence(built_graph):
    """Rule 1: with no stairs inventory, nothing is verified by absence."""
    if REGION.covers("planimetric") is None:
        assert not built_graph["step_free_source"].eq("derived:absent_from_osm_steps_and_planimetric_stairs").any()
    # (signals without an inventory: see test_enrich, a painted crossing stays inferred)


def test_every_edge_knows_whether_it_is_outside_the_region(built_graph):
    assert built_graph["outside_region"].notna().all()


def test_implicit_crossings_only_on_street_centerlines_and_always_with_kerb_evidence(built_graph):
    for end in ("u", "v"):
        x = built_graph[built_graph[f"xing_{end}"]]
        assert len(x) > 0
        assert (x["representation"] == "street_centerline").all()
        assert x[f"xing_{end}_kerb"].isin(["no_curb", "ramp_likely", "unknown", "no_ramp_reported"]).all()
        assert built_graph.loc[~built_graph[f"xing_{end}"], f"xing_{end}_kerb"].isna().all()


def test_contradictory_signal_tags_are_never_a_verified_signal(built_graph):
    c = built_graph[built_graph["crossing_signal_tags_conflict"]]
    assert len(c) > 0 or REGION.id != "dc"  # DC's OSM has some; a small town may have none
    assert (c["crossing_control"] != "signal").all()
    assert (c["crossing_control_status"] != "verified").all()


def test_verified_grades_are_repeatable_and_plausible(built_graph):
    v = built_graph[(built_graph["grade_max_pct_status"] == "verified") & (built_graph["grade_max_pct_source"] == REGION.role("dtm"))]
    assert (v["grade_max_spread_pct"] <= REPEATABILITY_TOL_PCT).all()
    assert (v["grade_max_pct"].astype(float) <= SUSPECT_GRADE_PCT).all()


def test_network_is_mostly_one_component(built_graph):
    if REGION.id != "dc":
        pytest.skip("a pilot region's connectivity is a finding (coverage report), not an invariant")
    share = built_graph.loc[built_graph["component"] == 0, "length_m"].sum() / built_graph["length_m"].sum()
    assert share > 0.95


def test_dtm_discontinuities_leave_grade_unknown(built_graph):
    d = built_graph[built_graph["grade_dtm_discontinuity"]]
    assert len(d) > 0 or REGION.role("dtm") is None
    assert (d["grade_max_pct_status"] == "unknown").all()
    assert (d["grade_avg_pct_status"] == "unknown").all()


def test_no_walkway_enters_an_area_closed_to_the_public(built_graph):
    """Fort Detrick: nothing routable may lie inside a closed area, not even a few metres of a
    gate road, and no connector may cross into one (graph.build.clip_restricted)."""
    import geopandas as gpd

    from pathfinder.config import INTERIM_DIR

    path = INTERIM_DIR / "osm_restricted_areas.parquet"
    if not path.exists():
        pytest.skip("restricted areas not extracted")
    areas = gpd.read_parquet(path).to_crs(built_graph.crs)
    if areas.empty:
        return
    inside = built_graph.geometry.intersection(areas.geometry.union_all().buffer(-0.5)).length
    bad = built_graph.loc[inside > 0.01, ["edge_id", "kind", "name"]]
    assert bad.empty, bad.head().to_dict("records")
