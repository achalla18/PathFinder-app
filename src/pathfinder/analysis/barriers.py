"""The city barrier map: where the recorded evidence shows an accessibility barrier, and
where nothing is recorded at all.

Barriers come from the enriched graph alone, so the map and the router can never disagree:
a place shown here as a barrier is one the router avoids or charges for, from the same
source and date. Unknown places are listed as such, never dropped: for a city, the gaps are
the survey worklist.

Rule 4 applies here too: nothing is called accessible or compliant. A barrier is "what the
named source recorded on the given date"; an unknown is "nobody has recorded it". Rule 7:
while a region's evidence includes unlicensed data, the report and map are internal.
"""

from __future__ import annotations

import json
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from ..config import ANALYSIS_CRS, GRAPH_DIR, REPORTS_DIR, WGS84
from ..fetch import raw_path
from ..graph.enrich import KERB_BARRIER, _crossing_outer_ends
from ..routing.cost import PENALTIES
from ..routing.profiles import ADA_RAMP_MAX_GRADE_PCT
from ..sources import REGION, SOURCES

log = logging.getLogger(__name__)

OUT = REPORTS_DIR / "barriers"

# A sidewalk is listed when its survey's condition index falls below the cost model's lowest
# band (the band the router charges most for), so map and router agree.
CONDITION_BARRIER_BELOW = min(lo for lo, _ in PENALTIES["sidewalk"]["condition_per_100m"]["bands"])
# Narrower than the 4 ft (1.22 m) pedestrian access route of the US Access Board's PROWAG
# (2023); 1.2 m so that a width recorded as exactly 4 ft is not listed.
NARROW_SIDEWALK_M = 1.2

# What each type means on the map and in the report; ``barrier`` types are recorded
# barriers, ``unknown`` types are places where the evidence is missing.
TYPES = {
    "curb_barrier": ("barrier", "Crossing end with a raised curb or no curb ramp"),
    "junction_curb_barrier": ("barrier", "Junction corner where curb ramps are reported missing"),
    "ramp_fails_check": ("barrier", "Curb ramp that fails its survey (too steep, no landing or damaged)"),
    "ramp_no_tactile_pad": ("barrier", "Curb ramp with no working tactile warning pad"),
    "steep": ("barrier", f"Sustained grade over {ADA_RAMP_MAX_GRADE_PCT}% (the ADA ramp limit)"),
    "steps": ("barrier", "Steps"),
    "no_sidewalk": ("barrier", "Street with no sidewalk recorded: walking in the roadway"),
    "sidewalk_poor_condition": ("barrier", f"Sidewalk with a survey condition index under {CONDITION_BARRIER_BELOW:g} of 100"),
    "sidewalk_narrow": ("barrier", f"Sidewalk recorded narrower than {NARROW_SIDEWALK_M} m (4 ft)"),
    "curb_unknown": ("unknown", "Crossing end where the curb is not recorded"),
    "junction_curb_unknown": ("unknown", "Junction corner where the curb is not recorded"),
    "sidewalk_unknown": ("unknown", "Street where it is not recorded whether there is a sidewalk"),
}
# Streets whose missing sidewalk record matters: service roads and alleys are driven, not
# walked along, and the sidewalk surveys do not grade them (enrich.NO_PLANIMETRIC_PRESENCE_KINDS).
STREET_KINDS = ("street_unknown", "street_with_sidewalk", "street_no_sidewalk")


def _claim(row: pd.Series, attr: str) -> dict:
    return {"status": row[f"{attr}_status"], "source": row[f"{attr}_source"], "as_of": row[f"{attr}_as_of"]}


def barriers(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Every recorded barrier and every unrecorded place, one row each, in ANALYSIS_CRS.
    Columns: type, kind (barrier | unknown), status, source, as_of, length_m, street,
    edge_id, geometry (a point for curbs, the segment for lines)."""
    rows: list[dict] = []

    # --- mapped crossings: both curb ends, one row per physical crossing end
    cr = edges[edges["kind"].eq("crossing")]
    if len(cr):
        groups, a_pts, b_pts = _crossing_outer_ends(edges)
        first = cr.assign(grp=groups[cr.index]).drop_duplicates("grp").set_index("grp")
        for end, pts in (("kerb_a", a_pts), ("kerb_b", b_pts)):
            for grp, r in first.iterrows():
                status = r[f"{end}_status"]
                check = f"ramp_check_{end[-1]}"
                verdict = r.get(check) if r.get(f"{check}_status", "unknown") != "unknown" else None
                if status == "unknown":
                    t, attr = "curb_unknown", end
                elif r[end] in KERB_BARRIER:
                    t, attr = "curb_barrier", end
                elif verdict == "fails":
                    t, attr = "ramp_fails_check", check
                elif verdict == "no_tactile_pad":
                    t, attr = "ramp_no_tactile_pad", check
                else:
                    continue
                rows.append({"type": t, **_claim(r, attr), "length_m": np.nan, "street": r.get("crossed_street"),
                             "edge_id": int(r["edge_id"]), "geometry": pts.loc[grp]})

    # --- junctions where streets are crossed without a crossing edge: one curb per flagged end
    for end in ("u", "v"):
        m = edges[f"xing_{end}"].fillna(False).astype(bool)
        kerb = edges[f"xing_{end}_kerb"].fillna("unknown")
        for t, keep in (("junction_curb_barrier", kerb.eq("no_ramp_reported")), ("junction_curb_unknown", kerb.eq("unknown"))):
            sel = edges[m & keep]
            if sel.empty:
                continue
            idx = 0 if end == "u" else -1
            pts = [shapely.Point(shapely.get_coordinates(g)[idx]) for g in sel.geometry.values]
            for (_, r), pt in zip(sel.iterrows(), pts):
                barrier = t.endswith("barrier")
                rows.append({"type": t, "status": "inferred" if barrier else "unknown",
                             "source": r[f"xing_{end}_kerb_source"] if barrier else None,
                             "as_of": r[f"xing_{end}_kerb_as_of"] if barrier else None, "length_m": np.nan, "street": r.get("name"),
                             "edge_id": int(r["edge_id"]), "geometry": pt})

    # --- lines
    g = pd.to_numeric(edges["grade_max_pct"], errors="coerce")
    steep = edges[(g > ADA_RAMP_MAX_GRADE_PCT) & edges["grade_max_pct_status"].ne("unknown")]
    for _, r in steep.iterrows():
        rows.append({"type": "steep", **_claim(r, "grade_max_pct"), "length_m": r["length_m"],
                     "street": r.get("along_street") or r.get("name"), "edge_id": int(r["edge_id"]), "geometry": r.geometry})
    for _, r in edges[edges["kind"].isin(["steps", "escalator"])].iterrows():
        rows.append({"type": "steps", **_claim(r, "step_free"), "length_m": r["length_m"], "street": r.get("name"),
                     "edge_id": int(r["edge_id"]), "geometry": r.geometry})
    for t, attr, below in (("sidewalk_poor_condition", "sidewalk_condition", CONDITION_BARRIER_BELOW),
                           ("sidewalk_narrow", "sidewalk_width_m", NARROW_SIDEWALK_M)):
        if attr not in edges:
            continue
        v = pd.to_numeric(edges[attr], errors="coerce")
        for _, r in edges[(v < below) & edges[f"{attr}_status"].ne("unknown")].iterrows():
            rows.append({"type": t, **_claim(r, attr), "length_m": r["length_m"],
                         "street": r.get("along_street") or r.get("name"), "edge_id": int(r["edge_id"]),
                         "geometry": r.geometry})
    street = edges["representation"].eq("street_centerline") & edges["kind"].isin(STREET_KINDS)
    none = street & (edges["kind"].eq("street_no_sidewalk")
                     | (edges["sidewalk_presence"].eq("none") & edges["sidewalk_presence_status"].ne("unknown")))
    unknown = street & ~none & edges["sidewalk_presence_status"].eq("unknown")
    for t, sel in (("no_sidewalk", edges[none]), ("sidewalk_unknown", edges[unknown])):
        for _, r in sel.iterrows():
            rows.append({"type": t, **_claim(r, "sidewalk_presence"), "length_m": r["length_m"], "street": r.get("name"),
                         "edge_id": int(r["edge_id"]), "geometry": r.geometry})

    out = gpd.GeoDataFrame(rows, columns=["type", "status", "source", "as_of", "length_m", "street", "edge_id", "geometry"],
                           geometry="geometry", crs=edges.crs)
    out.insert(1, "kind", out["type"].map(lambda t: TYPES[t][0]))
    # outside the region's line no inventory speaks for a place; keep only what is inside it
    if "outside_region" in edges:
        inside = ~edges.set_index("edge_id")["outside_region"].astype(bool)
        out = out[out["edge_id"].map(inside).fillna(False).astype(bool)]
    return out.reset_index(drop=True)


def summary(b: gpd.GeoDataFrame) -> pd.DataFrame:
    """Count and length per type, with how much of it is verified."""
    rows = []
    for t, (kind, label) in TYPES.items():
        g = b[b["type"] == t]
        rows.append({"type": t, "kind": kind, "what": label, "count": len(g),
                     "km": round(g["length_m"].sum() / 1000, 1) if g["length_m"].notna().any() else None,
                     "verified_pct": round((g["status"] == "verified").mean() * 100, 1) if len(g) else None})
    return pd.DataFrame(rows)


def by_tract(b: gpd.GeoDataFrame, tracts: gpd.GeoDataFrame, field: str) -> pd.DataFrame:
    """Per reporting district: recorded barriers and unknowns, per km of walkable network,
    so a large tract does not look worse only for being large."""
    pts = b.assign(geometry=b.geometry.representative_point())
    j = gpd.sjoin(pts, tracts[[field, "geometry"]], predicate="within", how="inner")
    counts = j.groupby([field, "kind"]).size().unstack(fill_value=0)
    for k in ("barrier", "unknown"):
        if k not in counts:
            counts[k] = 0
    return counts[["barrier", "unknown"]].rename(columns={"barrier": "barriers", "unknown": "unknowns"}).reset_index()


def _unlicensed_sources() -> list[str]:
    """Sources held back under rule 7: their license line says "Internal use only"."""
    return sorted(sid for sid, s in SOURCES.items() if "internal use" in s.license.lower())


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    edges = gpd.read_parquet(GRAPH_DIR / "edges.parquet")
    b = barriers(edges)
    s = summary(b)
    walk_km = edges.loc[~edges["outside_region"].astype(bool), "length_m"].sum() / 1000
    tract_md: list[str] = []
    if REGION.role("report_districts"):
        tracts = gpd.read_parquet(raw_path(REGION.role("report_districts"))).to_crs(ANALYSIS_CRS)
        mids = edges.assign(geometry=edges.geometry.interpolate(0.5, normalized=True))
        net = gpd.sjoin(mids[["length_m", "geometry"]], tracts[[REGION.report_district_field, "geometry"]],
                        predicate="within").groupby(REGION.report_district_field)["length_m"].sum() / 1000
        t = by_tract(b, tracts, REGION.report_district_field).merge(
            net.rename("network_km").reset_index(), on=REGION.report_district_field, how="left")
        t = t[t["network_km"] >= 1.0]  # slivers of tracts barely inside the line say nothing
        t["barriers_per_km"] = (t["barriers"] / t["network_km"]).round(2)
        t["unknowns_per_km"] = (t["unknowns"] / t["network_km"]).round(2)
        t["network_km"] = t["network_km"].round(1)
        t = t.sort_values("unknowns_per_km", ascending=False)
        t.to_csv(OUT / "barriers_by_tract.csv", index=False)
        tract_md = [f"## By {REGION.report_district_label} (most unknowns per km first)", "",
                    "Tracts with at least 1 km of walkable network inside the line. Full table: "
                    "[`barriers_by_tract.csv`](barriers_by_tract.csv).", "", _md(t.head(15)), ""]
    _map(edges, b, OUT / "map_barriers.png")
    unlicensed = _unlicensed_sources()
    md = [
        f"# Accessibility barriers: {REGION.name}",
        "",
        "_Generated by `pf barriers` from the enriched graph. Do not edit by hand._",
        "",
        *([f"**Internal.** Built in part on data with no stated open license ({', '.join(unlicensed)}); "
           "not to be shown outside until the owners give written permission (rule 7).", ""] if unlicensed else []),
        "A **barrier** is what a named source recorded on the given date; an **unknown** is a place "
        "nobody has recorded. Nothing here certifies a place as accessible. Slope is unknown wherever "
        "no elevation model is loaded, so no steep segment can appear before one is.",
        "",
        f"Walkable network inside the line: {walk_km:,.0f} km.",
        "",
        _md(s),
        "",
        *tract_md,
        "![map_barriers.png](map_barriers.png)",
        "",
    ]
    (OUT / "barriers.md").write_text("\n".join(md), encoding="utf-8")
    counts = s.set_index("type")["count"].to_dict()
    (OUT / "barriers_summary.json").write_text(json.dumps(counts, indent=2))
    log.info("barriers: %s -> %s", counts, OUT / "barriers.md")
    return counts


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(v) else f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)
                                     for v in r.values) + " |")
    return "\n".join(out)


def _map(edges: gpd.GeoDataFrame, b: gpd.GeoDataFrame, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 9), dpi=110)
    edges.plot(ax=ax, color="#e0e0e0", linewidth=0.3)
    style = {"sidewalk_unknown": ("#bdbdbd", 0.8), "no_sidewalk": ("#e34a33", 1.2), "steep": ("#984ea3", 1.4),
             "steps": ("#ff7f00", 1.4), "sidewalk_poor_condition": ("#8c510a", 1.4),
             "sidewalk_narrow": ("#01665e", 1.4)}
    for t, (color, lw) in style.items():
        g = b[b["type"] == t]
        if len(g):
            g.plot(ax=ax, color=color, linewidth=lw, label=TYPES[t][1])
    for t, color, size in (("curb_unknown", "#969696", 2), ("junction_curb_unknown", "#969696", 2),
                           ("curb_barrier", "#b30000", 9), ("junction_curb_barrier", "#b30000", 9)):
        g = b[b["type"] == t]
        if len(g):
            g.plot(ax=ax, color=color, markersize=size, label=TYPES[t][1])
    ax.set_axis_off()
    ax.set_title(f"Accessibility barriers and unknowns: {REGION.name}", fontsize=10)
    ax.legend(loc="lower left", fontsize=6, frameon=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


ESTIMATED_SIDEWALK = "Sidewalk recorded by a survey; the line is drawn where it is estimated to run"


def estimated_sidewalks(lines: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """The estimated sidewalk lines (``enrich.estimate_survey_sidewalk_lines``) in the barrier
    map's columns, kind ``evidence``: not a barrier, but where a survey shows a sidewalk that
    OSM does not map. Inside the region's line only."""
    name = edges.set_index("edge_id")["name"] if "name" in edges else pd.Series(dtype="object")
    inside = ~edges.set_index("edge_id")["outside_region"].astype(bool)
    lines = lines[lines["edge_id"].map(inside).fillna(False).astype(bool)]
    out = gpd.GeoDataFrame({
        "type": "sidewalk_estimated", "kind": "evidence", "status": lines["status"], "source": lines["source"],
        "as_of": lines["as_of"], "length_m": lines.length, "street": lines["edge_id"].map(name),
        "edge_id": lines["edge_id"],
    }, geometry=lines.geometry.values, crs=lines.crs)
    return out.reset_index(drop=True)


def as_geojson(b: gpd.GeoDataFrame) -> dict:
    """For the API: WGS84 features with their evidence."""
    g = b.to_crs(WGS84)
    g = g.assign(what=g["type"].map(lambda t: ESTIMATED_SIDEWALK if t == "sidewalk_estimated" else TYPES[t][1]),
                 length_m=g["length_m"].round(1))
    return json.loads(g.to_json(drop_id=True))
