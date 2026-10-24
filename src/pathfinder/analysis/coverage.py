"""Phase 0 coverage report: how much of the region's pedestrian network do we actually know?

Writes ``reports/<region>/phase0/coverage.md`` plus CSVs and PNG maps. Every number is computed from the built graph; nothing is
hand-entered. Reporting districts (DC: wards) appear only for a region that has them.
"""

from __future__ import annotations

import json
import logging

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..config import ANALYSIS_CRS, GRAPH_DIR, REPORTS_DIR  # noqa: E402
from ..fetch import load_manifest, raw_path  # noqa: E402
from ..graph.enrich import SIDEWALK_PRESENT, SURVEY_SIDEWALK_LINES  # noqa: E402
from ..sources import REGION  # noqa: E402

log = logging.getLogger(__name__)
OUT = REPORTS_DIR / "phase0"

# (attribute, scope, weight) — scope selects the edges where the attribute is meaningful
ATTRS = [
    ("crossing_control", "crossings", "count"),
    ("signal_lpi", "signalized crossings", "count"),
    ("signal_aps", "signalized crossings", "count"),
    ("signal_hawk", "signalized crossings", "count"),
    ("signal_all_ped_phase", "signalized crossings", "count"),
    ("crossed_lanes", "crossings", "count"),
    ("crossed_width_m", "crossings", "count"),
    ("crossed_speed_mph", "crossings", "count"),
    ("crossed_aadt", "crossings", "count"),
    ("kerb_a", "crossings", "count"),
    ("kerb_b", "crossings", "count"),
    ("ramp_check_a", "crossings", "count"),
    ("ramp_check_b", "crossings", "count"),
    ("step_free", "all edges", "length"),
    ("grade_avg_pct", "all edges", "length"),
    ("grade_max_pct", "all edges", "length"),
    ("surface_class", "pedestrian ways", "length"),
    ("sidewalk_presence", "street centerlines", "length"),
    ("sidewalk_condition", "sidewalks", "length"),
    ("sidewalk_width_m", "sidewalks", "length"),
    ("street_lanes", "street centerlines", "length"),
    ("street_speed_mph", "street centerlines", "length"),
]


def _scope(e: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "crossings":
        return e[e["kind"] == "crossing"].drop_duplicates("osm_way_id")
    if scope == "signalized crossings":
        c = e[e["kind"] == "crossing"].drop_duplicates("osm_way_id")
        return c[c["crossing_control"] == "signal"]
    if scope == "pedestrian ways":
        return e[e["representation"] == "pedestrian_way"]
    if scope == "sidewalks":  # sidewalk-level ways, and streets with a sidewalk on record
        known = e["sidewalk_presence_status"].ne("unknown") & e["sidewalk_presence"].isin(SIDEWALK_PRESENT)
        return e[(e["kind"].eq("sidewalk") & e["representation"].eq("pedestrian_way"))
                 | (e["representation"].eq("street_centerline") & known)]
    if scope == "street centerlines":
        return e[e["representation"] == "street_centerline"]
    return e


def _shares(df: pd.DataFrame, attr: str, weight: str) -> dict:
    w = df["length_m"] if weight == "length" else pd.Series(1.0, index=df.index)
    tot = w.sum()
    return {s: round(float(w[df[f"{attr}_status"] == s].sum() / tot * 100), 1) if tot else 0.0
            for s in ("verified", "inferred", "unknown")}


def attribute_table(e: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for attr, scope, weight in ATTRS:
        df = _scope(e, scope)
        sh = _shares(df, attr, weight)
        srcs = df[f"{attr}_source"].value_counts(normalize=True).head(3)
        ages = pd.to_datetime(df[f"{attr}_as_of"], errors="coerce").dt.year.dropna()
        rows.append({
            "attribute": attr, "scope": scope, "weighted_by": weight, "n": len(df), **sh,
            "top_sources": "; ".join(f"{k} {v:.0%}" for k, v in srcs.items()),
            "evidence_year_median": int(ages.median()) if len(ages) else None,
            "evidence_year_range": f"{int(ages.min())}–{int(ages.max())}" if len(ages) else "",
        })
    return pd.DataFrame(rows)


def with_ward(e: gpd.GeoDataFrame, wards: gpd.GeoDataFrame, field: str = "WARD") -> gpd.GeoDataFrame:
    """Edges with the reporting district (DC: ward) containing their midpoint (null beyond
    the region's line). The column is called ``ward`` whatever the region calls them."""
    mid = gpd.GeoDataFrame(e[["edge_id"]], geometry=e.geometry.interpolate(0.5, normalized=True), crs=e.crs)
    j = gpd.sjoin(mid, wards[[field, "geometry"]].rename(columns={field: "WARD"}), predicate="within", how="left")
    j = j[~j.index.duplicated()]
    return e.assign(ward=j["WARD"].reindex(e.index))


def attribute_ward_table(e: gpd.GeoDataFrame) -> pd.DataFrame:
    """Verified / inferred / unknown share of every graded attribute in every ward (long
    format). ``e`` must carry ``ward``."""
    rows = []
    for ward, g in e.groupby("ward"):
        for attr, scope, weight in ATTRS:
            df = _scope(g, scope)
            rows.append({"ward": int(ward), "attribute": attr, "scope": scope, "weighted_by": weight,
                         "n": len(df), **_shares(df, attr, weight)})
    return pd.DataFrame(rows)


def ward_table(e: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = []
    for ward, g in e.groupby("ward"):
        cr = g[g["kind"] == "crossing"].drop_duplicates("osm_way_id")
        ped_len = g["length_m"].sum()
        both_kerbs = ((cr["kerb_a_status"] != "unknown") & (cr["kerb_b_status"] != "unknown")).mean() if len(cr) else np.nan
        sc = g[g["representation"] == "street_centerline"]
        rows.append({
            "ward": int(ward),
            "network_km": round(ped_len / 1000, 1),
            "sidewalk_level_share_pct": round(g.loc[g["representation"] == "pedestrian_way", "length_m"].sum() / ped_len * 100, 1),
            "crossings": len(cr),
            "crossings_both_kerbs_known_pct": round(both_kerbs * 100, 1),
            "crossings_control_known_pct": round((cr["crossing_control_status"] != "unknown").mean() * 100, 1),
            "centerline_sidewalk_known_pct": round(_shares(sc, "sidewalk_presence", "length")["verified"], 1) if len(sc) else None,
            "grade_known_pct": round(100 - _shares(g, "grade_max_pct", "length")["unknown"], 1),
        })
    return pd.DataFrame(rows)


def _planimetric_has(description: str) -> bool:
    """Whether the region's planimetric survey records this class at all. A flag that counts
    what such a class did not confirm means nothing when no source records the class
    (Montgomery's sidewalk layer has no crosswalks or stairs)."""
    from ..graph.roles import read

    planim, _ = read("planimetrics")
    return bool(planim["DESCRIPTION"].astype("string").str.strip().eq(description).any())


def conflicts(e: pd.DataFrame) -> dict:
    """Counts of cross-source disagreements; None where no source records what is compared."""
    cr = e[e["kind"] == "crossing"].drop_duplicates("osm_way_id")
    has_crosswalks = _planimetric_has("Crosswalk")
    return {
        "osm_signal_without_signal_inventory_within_40m": int(((cr["crossing"] == "traffic_signals") & ~cr["signal_inventory_within_40m"]).sum()),
        "osm_unsignalized_with_signal_inventory_within_40m": int((cr["crossing_control"].isin(["marked_uncontrolled", "unmarked"])
                                                             & cr["crossing_control_status"].eq("verified") & cr["signal_inventory_within_40m"]).sum()),
        "osm_marked_signalization_not_recorded_with_signal_inventory_within_40m": int(
            cr["crossing_control_source"].eq("derived:osm_marked_signalization_not_recorded").sum()),
        "osm_signal_tags_contradict_each_other": int(cr["crossing_signal_tags_conflict"].sum()),
        "crossings_marked_in_osm_without_planimetric_crosswalk": int(((cr["crossing_control"] == "marked_uncontrolled") & ~cr["planimetric_crosswalk"]).sum())
        if has_crosswalks else None,
        "unmapped_stairs_found_via_planimetrics": int(e["planimetric_stairs_overlap"].sum()),
        "kerb_evidence_from_project_sidewalk_pre_2018": int(((cr["kerb_a_source"] == "project_sidewalk_dc") &
                                                             (pd.to_datetime(cr["kerb_a_as_of"], errors="coerce").dt.year < 2018)).sum()),
        "grade_readings_downgraded_suspect": int((e["grade_quality"] == "suspect").sum()),
        "grade_readings_downgraded_not_repeatable": int((e["grade_quality"] == "not_repeatable").sum()),
        "edges_with_grade_separation_masking": int(e["grade_separation_masked"].sum()),
        "implicit_crossing_edge_ends": int(e["xing_u"].sum() + e["xing_v"].sum()),
        "implicit_crossing_ends_with_unknown_kerb": int((e["xing_u_kerb"].eq("unknown")).sum() + (e["xing_v_kerb"].eq("unknown")).sum()),
        "edges_outside_region_inventory_coverage_on_street": int((e["outside_region"].astype(bool) & ~e["offstreet"]).sum()),
        "grade_readings_discarded_dtm_discontinuity": int(e["grade_dtm_discontinuity"].sum()),
    }


# The accessibility go/no-go bar, fixed 2026-10-15 before any accessibility source was loaded
# (docs/PROGRESS.md, Decisions log). Criterion 4 (a manual-wheelchair route for the trip
# sample) comes from `pf crux`. Measured inside the region's line.
BAR = {"curb_evidence": 60.0, "slope_verified": 80.0, "sidewalk_evidence": 60.0, "main_component": 90.0}


def accessibility_bar(e: pd.DataFrame, stats: dict) -> list[dict]:
    """Criteria 1, 2, 3 and 5 of the accessibility bar, with what each counts."""
    e = e[~e["outside_region"].astype(bool)]
    cr = e[e["kind"] == "crossing"].drop_duplicates("osm_way_id")
    ends = 2 * len(cr)
    known = int(cr["kerb_a_status"].ne("unknown").sum() + cr["kerb_b_status"].ne("unknown").sum())
    for end in ("u", "v"):
        f = e[f"xing_{end}"].fillna(False).astype(bool)
        ends += int(f.sum())
        known += int((f & e[f"xing_{end}_kerb"].ne("unknown")).sum())
    L = e["length_m"]
    slope = L[e["grade_max_pct_status"].eq("verified")].sum() / L.sum() * 100
    sidewalk_level = e["representation"].eq("pedestrian_way")
    street_known = e["representation"].eq("street_centerline") & e["sidewalk_presence_status"].ne("unknown")
    one_side = street_known & e["sidewalk_presence"].eq("at_least_one_side")
    sw = L[sidewalk_level | street_known].sum() / L.sum() * 100
    rows = [
        ("1. Curb evidence at crossing ends, implicit included", known / max(ends, 1) * 100, BAR["curb_evidence"],
         f"{known:,} of {ends:,} ends"),
        ("2. Slope verified, share of walkable length", slope, BAR["slope_verified"], ""),
        ("3. Sidewalk evidence, share of walkable length", sw, BAR["sidewalk_evidence"],
         f"sidewalk-level ways {L[sidewalk_level].sum() / L.sum() * 100:.1f}%, streets with presence recorded "
         f"{L[street_known].sum() / L.sum() * 100:.1f}% (of which a sidewalk on at least one side only: "
         f"{L[one_side].sum() / L.sum() * 100:.1f}%)"),
        ("5. Main component, share of length (whole graph)", stats["largest_component_share_of_length"] * 100,
         BAR["main_component"], ""),
    ]
    return [{"criterion": c, "value_pct": round(v, 1), "bar_pct": b, "result": "passes" if v >= b else "misses",
             "counts": n} for c, v, b, n in rows]


def _map(e: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame, title: str, col: pd.Series, colors: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 8.5), dpi=110)
    boundary.boundary.plot(ax=ax, color="#999999", linewidth=0.6)
    for val, color in colors.items():
        sub = e[col == val]
        if len(sub):
            if sub.geom_type.iloc[0] == "Point":
                sub.plot(ax=ax, color=color, markersize=0.6, label=f"{val} ({len(sub):,})")
            else:
                sub.plot(ax=ax, color=color, linewidth=0.35 if len(sub) > 5000 else 0.9, label=f"{val} ({len(sub):,})")
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def maps(e: gpd.GeoDataFrame) -> list[str]:
    boundary = gpd.read_parquet(raw_path(REGION.boundary)).to_crs(ANALYSIS_CRS)
    out = []
    # one point per physical crossing (kerb attributes are identical across its fragments)
    cr = e[e["kind"] == "crossing"].drop_duplicates("osm_way_id")
    cr = cr.set_geometry(cr.geometry.interpolate(0.5, normalized=True))
    both = (cr["kerb_a_status"] != "unknown") & (cr["kerb_b_status"] != "unknown")
    barrier = cr["kerb_a"].isin(["raised", "no_ramp"]) | cr["kerb_b"].isin(["raised", "no_ramp"])
    status = pd.Series(np.where(barrier, "barrier at an end", np.where(both, "ramps/flush both ends", "an end unknown")), index=cr.index)
    p = OUT / "map_crossing_kerbs.png"
    _map(cr, boundary, "Crossings: curb-ramp evidence at both ends (one point per crossing)", status,
         {"ramps/flush both ends": "#2b8cbe", "an end unknown": "#bdbdbd", "barrier at an end": "#e34a33"}, p)
    out.append(p.name)
    rep = e["representation"].map({"pedestrian_way": "sidewalk-level / pedestrian way",
                                   "street_centerline": "street centerline (sidewalk not separate)",
                                   "inferred_connector": "inferred connector"})
    p = OUT / "map_representation.png"
    _map(e, boundary, "How each walkable segment is represented", rep,
         {"sidewalk-level / pedestrian way": "#2b8cbe", "street centerline (sidewalk not separate)": "#fdae61",
          "inferred connector": "#d7191c"}, p)
    out.append(p.name)
    centerline = e["representation"].eq("street_centerline")
    known = e["sidewalk_presence_status"].ne("unknown")
    presence = e["sidewalk_presence"]
    sw = pd.Series(np.where(~centerline, "sidewalk or path mapped as its own way",
                            np.where(known & presence.eq("both"), "street: sidewalk both sides recorded",
                                     np.where(known & presence.isin(["one_side", "at_least_one_side"]),
                                              "street: sidewalk on one side recorded",
                                              np.where(known, "street: no sidewalk recorded", "street: sidewalk unknown")))),
                   index=e.index)
    p = OUT / "map_sidewalk_evidence.png"
    _map(e, boundary, "Sidewalk evidence along each walkable segment", sw,
         {"sidewalk or path mapped as its own way": "#2b8cbe", "street: sidewalk both sides recorded": "#74c476",
          "street: sidewalk on one side recorded": "#c2e699", "street: no sidewalk recorded": "#e34a33",
          "street: sidewalk unknown": "#bdbdbd"}, p)
    out.append(p.name)
    lines_path = GRAPH_DIR / SURVEY_SIDEWALK_LINES
    if lines_path.exists():
        lines = gpd.read_parquet(lines_path)
        if len(lines):
            p = OUT / "map_estimated_sidewalks.png"
            fig, ax = plt.subplots(figsize=(7.5, 8.5), dpi=110)
            boundary.boundary.plot(ax=ax, color="#999999", linewidth=0.6)
            e[e["representation"].eq("pedestrian_way")].plot(ax=ax, color="#2b8cbe", linewidth=0.35,
                                                             label="sidewalk or path mapped in OSM")
            lines.plot(ax=ax, color="#e6550d", linewidth=0.5,
                       label=f"sidewalk line estimated from the survey ({len(lines):,}, {lines.length.sum() / 1000:.0f} km)")
            ax.set_title("Estimated sidewalk lines (survey sidewalks beside streets OSM maps as centerlines)", fontsize=9)
            ax.set_axis_off()
            ax.legend(loc="lower left", fontsize=8, frameon=False)
            fig.tight_layout()
            fig.savefig(p)
            plt.close(fig)
            out.append(p.name)
    return out


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]

    def fmt(v) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, float) and v.is_integer() and abs(v) >= 100:
            return f"{int(v):,}"
        return str(v)

    for row in df.itertuples(index=False):  # itertuples keeps per-column dtypes
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    e = gpd.read_parquet(GRAPH_DIR / "edges.parquet")
    stats = json.loads((GRAPH_DIR / "build_stats.json").read_text())
    at = attribute_table(e)
    label = REGION.report_district_label
    district_md: list[str] = []
    if REGION.role("report_districts"):
        wards = gpd.read_parquet(raw_path(REGION.role("report_districts"))).to_crs(ANALYSIS_CRS)
        ew = with_ward(e, wards, REGION.report_district_field)
        wt = ward_table(ew)
        awt = attribute_ward_table(ew)
        known = awt.assign(known=(100 - awt["unknown"]).round(1)).pivot(index="attribute", columns="ward", values="known")
        known = known.reindex([a for a, _, _ in ATTRS]).reset_index()
        known.columns = [str(c) if c == "attribute" else f"{label} {c}" for c in known.columns]
        wt.to_csv(OUT / f"coverage_by_{label}.csv", index=False)
        awt.to_csv(OUT / f"coverage_attributes_by_{label}.csv", index=False)
        district_md = [
            f"## By {label}",
            "",
            _md_table(wt),
            "",
            f"## Each attribute by {label}: % of its scope with any evidence (verified or inferred)",
            "",
            f"Verified / inferred / unknown shares per {label} are in `coverage_attributes_by_{label}.csv`.",
            "",
            _md_table(known),
            "",
        ]
    cf = conflicts(e)
    imgs = maps(e)
    at.to_csv(OUT / "coverage_attributes.csv", index=False)
    (OUT / "coverage_conflicts.json").write_text(json.dumps(cf, indent=2))
    manifest = load_manifest()
    present = [f"{r} = `{sid}`" for r, sid in REGION.roles.items()]
    rep_km = stats["km_by_representation"]
    lines_path = GRAPH_DIR / SURVEY_SIDEWALK_LINES
    est = gpd.read_parquet(lines_path) if lines_path.exists() else None
    n_lines, km_lines = (len(est), est.length.sum() / 1000) if est is not None else (0, 0.0)
    sidewalk_share = rep_km.get("pedestrian_way", 0.0) / max(sum(rep_km.values()), 1e-9) * 100

    md = [
        f"# Phase 0 — Data coverage report: {REGION.name}",
        "",
        f"_Generated by `pf coverage` from the graph built on OSM extract dated "
        f"{manifest.get(REGION.osm, {}).get('last_modified', 'unknown')}. Do not edit by hand._",
        "",
        *([
            f"Sources beyond OSM and the boundary: {'; '.join(present) if present else 'none yet'}. "
            "An attribute with no source stays unknown; absence of a record counts as evidence only "
            f"where an inventory covers the place (planimetric: {REGION.covers('planimetric')}, "
            f"signals: {REGION.covers('signals')}).",
            "",
        ]),
        "Status meanings: **verified** = directly observed by a source (incl. confirmed absence in an "
        "inventory comprehensive for that place); **inferred** = produced by a documented heuristic; "
        "**unknown** = no evidence. Unknown is never treated as favorable.",
        "",
        "## Network",
        "",
        f"- Walkable edges: **{stats['edges']:,}**, nodes: **{stats['nodes']:,}**",
        f"- Largest connected component holds **{stats['largest_component_share_of_length']*100:.1f}%** of walkable length "
        f"({stats['components']} components in total; the rest are mostly private-lot service roads)",
        f"- Length by representation (km): {stats['km_by_representation']}"
        f" — sidewalk-level share of length: **{sidewalk_share:.1f}%**",
        f"- Removed-road stubs pruned: {stats['removed_road_stubs_pruned']:,}; inferred gap connectors: {stats['inferred_connectors']:,} "
        f"(each disclosed as `inferred_connector`), of which {stats['inferred_connectors_across_mapped_sidewalk_surface']:,} "
        f"bridge a wider gap across planimetric sidewalk surface",
        f"- Pedestrian dead ends left unbridged: {stats['dead_ends_near_another_edge_unbridged']:,} lie within "
        f"{stats['near_miss_radius_m']:.0f} m of another walkway with no evidence of a connection (near is not connected)",
        "",
        "## Accessibility bar (criteria 1, 2, 3, 5; criterion 4 is in crux.md)",
        "",
        "Inside the region's line. Fixed on 2026-10-15 before any accessibility source was loaded.",
        "",
        _md_table(pd.DataFrame(accessibility_bar(e, stats))),
        "",
        *([f"- Estimated sidewalk lines (map layer only; the router walks the street centerline, which carries "
           f"the same survey evidence): {n_lines:,}, {km_lines:.1f} km. See `map_estimated_sidewalks.png`.", ""]
          if n_lines else []),
        "## Attribute coverage",
        "",
        _md_table(at),
        "",
        *district_md,
        "## Cross-source conflicts and quality flags",
        "",
        *[f"- `{k}`: {v:,}" if v is not None else f"- `{k}`: n/a (no source records it)" for k, v in cf.items()],
        "",
        "## Maps",
        "",
        *[f"![{i}]({i})" for i in imgs],
        "",
    ]
    (OUT / "coverage.md").write_text("\n".join(md), encoding="utf-8")
    log.info("coverage report written to %s", OUT / "coverage.md")
