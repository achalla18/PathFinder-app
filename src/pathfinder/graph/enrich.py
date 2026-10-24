"""Attach evidence-graded attributes to graph edges.

Every attribute follows ``evidence`` conventions (value / status / source / as_of). Sources
are applied in priority order and later sources only fill gaps (``fill_where_better``), so
the precedence is explicit in code:

    OSM (current)  >  Project Sidewalk (human-labeled imagery, mostly 2014–2017)
                   >  planimetrics (DC: 2010 / 2015–2021 capture)  >  derived heuristics

Which source plays each part (signals, planimetrics, roadway inventory, ...) is the region's
choice (``regions``); a role the region lacks grades nothing, so its attributes stay unknown.

Status semantics: ``verified`` = directly observed by a source (age is carried separately in
``*_as_of``); ``inferred`` = produced by a heuristic in this module; ``unknown`` = no
evidence. Absence of a record is never evidence of a good condition.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.ops import substring

from .. import evidence as ev
from ..config import ANALYSIS_CRS, GRAPH_DIR, INTERIM_DIR
from ..fetch import load_manifest, raw_path
from ..regions import COVERAGE_FAMILIES
from ..regions.base import AS_OF_OSM
from ..sources import REGION
from . import post_ada, roles
from .elevation import enrich_slope

log = logging.getLogger(__name__)

FT = 0.3048
OSM_AS_OF = None  # filled from the manifest (extract Last-Modified date)

# ---------------------------------------------------------------- parameters (documented)
KERB_ENDPOINT_RADIUS_M = 4.0  # kerb node counts for a crossing end if within this distance
PS_RAMP_RADIUS_M = 8.0  # Project Sidewalk curb-ramp label search radius at crossing ends
RAMP2010_RADIUS_M = 8.0
SIGNAL_RADIUS_M = 40.0  # inventory signal points are intersection-level
OFFSTREET_DIST_M = 40.0  # farther than this from any roadway inventory block => off-street
CROSSWALK_POLY_MIN_OVERLAP = 0.5  # share of crossing length inside a planimetric crosswalk
STAIRS_POLY_MIN_OVERLAP_M = 2.0
PS_MIN_VALID_YEAR = 2007  # Street View imagery in DC predates this rarely; earlier = bad date
CROSSING_NODE_SNAP_M = 0.3  # an OSM crossing node belongs to a crossing way if it lies on it
# --- unmodeled crossings where street centerlines meet (see enrich_implicit_crossings)
JUNCTION_SIGNAL_RADIUS_M = 30.0  # signal inventory points sit at the intersection itself
# A junction that is not a street intersection (a driveway or service road meeting a street)
# takes a signal only this close: the legs of a signalized intersection sat 4-14 m from its
# record, while parking-lot entrances 19-29 m down the street were wrongly signalized at 30 m
# (Frederick and Gaithersburg imagery, 2026-10-16).
DRIVEWAY_SIGNAL_RADIUS_M = 15.0
JUNCTION_ROAD_RADIUS_M = 15.0  # roadway inventory blocks meeting at the junction
JUNCTION_RAMP_RADIUS_M = 20.0  # curb-ramp evidence at the corners of a junction
JUNCTION_MIN_RAMPS = 2  # one crossing needs a ramp on each side
# A field sheet's coordinates are the graph's own crossing ends and junction nodes (from the
# blocked-trip worklist, rounded to 1e-6 degrees, about 0.1 m), so a tight radius matches its
# place and never the curb across the street or the next junction.
FIELD_CHECK_RADIUS_M = 3.0
# The street a sidewalk runs beside, for wording only ("turn left onto K ST NW").
ALONG_STREET_MAX_M = 30.0  # sidewalk to centerline: half of DC's widest avenues (160 ft) plus the planting strip
ALONG_STREET_MAX_ANGLE_DEG = 30.0  # the sidewalk must run with the street, not across it
ALONG_STREET_MIN_EDGE_M = 5.0  # shorter pieces (corner stubs) have no direction worth trusting
ALONG_STREET_LOCAL_M = 5.0  # the street's own direction is taken this far either side of the nearest point
STREET_KINDS = ("street_with_sidewalk", "street_no_sidewalk", "street_unknown", "pedestrian_street")
SHARED_ROAD_KINDS = ("alley", "service")


def _read(source_id: str) -> gpd.GeoDataFrame:
    return gpd.read_parquet(raw_path(source_id)).to_crs(ANALYSIS_CRS)


def _ms_to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, unit="ms", errors="coerce").dt.date.astype("string")


OSM = REGION.osm  # source id of the region's OSM extract


def _osm_as_of() -> str | None:
    lm = load_manifest().get(OSM, {}).get("last_modified")
    return str(pd.to_datetime(lm).date()) if lm else None


def _max_capture_date(capture_ms: pd.Series) -> str | None:
    """Latest capture date of a planimetric layer, or None for an empty one."""
    d = pd.to_datetime(capture_ms, unit="ms", errors="coerce").max()
    return None if pd.isna(d) else str(d.date())


# ================================================================ crossings


def crossing_groups(edges: gpd.GeoDataFrame) -> pd.Series:
    """A physical crossing is one OSM crossing way, possibly split into several edges (at the
    road node, islands, gap-bridging splits). Penalties are later apportioned by length so a
    crossing is counted exactly once."""
    g = pd.Series(pd.NA, index=edges.index, dtype="Int64")
    m = edges["kind"] == "crossing"
    g[m] = edges.loc[m, "osm_way_id"].astype("int64")
    return g


def signal_inventory_as_of(source_id: str | None) -> str | None:
    """The date a signal inventory describes, where its records carry none (the region's
    ``Source.as_of``)."""
    if source_id is None or source_id not in REGION.sources:
        return None
    as_of = REGION.sources[source_id].as_of
    return _osm_as_of() if as_of == AS_OF_OSM else as_of


def enrich_crossing_control(edges: gpd.GeoDataFrame, signals: gpd.GeoDataFrame, planim: gpd.GeoDataFrame,
                            osm_nodes: gpd.GeoDataFrame, signals_source: str | None = "traffic_signals",
                            planim_source: str | None = "sidewalks_2021") -> None:
    """Control type of each mapped crossing: ``signal`` / ``marked_uncontrolled`` / ``unmarked``.

    OSM records signalization in several places, and all of them are read: the way's
    ``crossing`` and ``crossing:signals`` tags and the same tags on the crossing node that
    lies on the way. ``crossing=marked`` alone says the crossing is painted, not whether it is
    signalized. Rules:

    - signal evidence and nothing against it          -> signal (VERIFIED)
    - evidence of no signal and nothing for a signal  -> marked/unmarked (VERIFIED)
    - both (OSM contradicts itself)                   -> marked/unmarked (INFERRED): the
      less favorable reading, never the signal
    - neither, but painted                            -> marked_uncontrolled; VERIFIED when no
      inventory signal is nearby, INFERRED when one is (it may well be signalized, but a quarter
      of such crossings with node evidence are slip lanes or mid-block crossings that are not),
      and INFERRED where no signal inventory covers the place (``covered_signals``): there,
      no signal on record is not evidence of no signal
    """
    m = edges["kind"] == "crossing"
    osm_as_of = _osm_as_of()
    groups = crossing_groups(edges)

    cr = edges["crossing"].fillna("")
    mk = edges["crossing:markings"].fillna("")
    ws = edges["crossing:signals"].fillna("")

    # --- signal tags on the crossing node(s) lying on each crossing way
    cn = osm_nodes[osm_nodes["highway"].eq("crossing") | osm_nodes["crossing"].notna()]
    cn = cn.assign(n_sig=cn["crossing"].eq("traffic_signals") | cn["crossing:signals"].eq("yes"),
                   n_nosig=cn["crossing:signals"].eq("no") | cn["crossing"].isin(["uncontrolled", "unmarked"]))
    nj = gpd.sjoin_nearest(cn[["n_sig", "n_nosig", "geometry"]], edges.loc[m, ["geometry"]].assign(grp=groups[m]),
                           how="inner", max_distance=CROSSING_NODE_SNAP_M)
    node_sig = groups.map(nj.groupby("grp")["n_sig"].any()).fillna(False).astype(bool)
    node_nosig = groups.map(nj.groupby("grp")["n_nosig"].any()).fillna(False).astype(bool)

    sig = cr.eq("traffic_signals") | ws.eq("yes") | node_sig
    nosig = ws.eq("no") | cr.isin(["uncontrolled", "unmarked"]) | node_nosig
    painted = (cr.isin(["marked", "zebra"]) | (mk.ne("") & ~mk.eq("no"))
               | (cr.eq("uncontrolled") & ~mk.eq("no"))) & ~cr.eq("unmarked")
    unsignalized = pd.Series(None, index=edges.index, dtype="object")
    unsignalized[painted] = "marked_uncontrolled"
    unsignalized[(cr.eq("unmarked") | mk.eq("no")) & ~painted] = "unmarked"

    # --- inventory signal features (nearest active signal within radius of crossing midpoint)
    mid = edges.loc[m].geometry.interpolate(0.5, normalized=True)
    mids = gpd.GeoDataFrame({"eidx": mid.index}, geometry=mid.values, crs=edges.crs)
    sg = signals[signals["STATUS"].eq("Active")][["LPI", "APS", "HAWK", "ALL_PED_PHASE", "PED_RECALL_MODE", "geometry"]]
    j = gpd.sjoin_nearest(mids, sg, how="left", max_distance=SIGNAL_RADIUS_M, distance_col="sig_d")
    j = j.sort_values("sig_d").drop_duplicates("eidx").set_index("eidx")
    near = j["sig_d"].notna()
    edges["signal_inventory_within_40m"] = False
    edges.loc[near[near].index, "signal_inventory_within_40m"] = True
    ddot = edges["signal_inventory_within_40m"]

    ev.init_unknown(edges, "crossing_control")
    signal_val = pd.Series("signal", index=edges.index)
    covered = edges["covered_signals"].astype(bool)
    ev.fill_where_better(edges, "crossing_control", m & sig & ~nosig, signal_val, ev.VERIFIED, OSM, osm_as_of)
    ev.fill_where_better(edges, "crossing_control", m & nosig & ~sig, unsignalized, ev.VERIFIED, OSM, osm_as_of)
    ev.fill_where_better(edges, "crossing_control", m & ~sig & ~nosig & ~ddot & covered, unsignalized, ev.VERIFIED,
                         OSM, osm_as_of)
    ev.fill_where_better(edges, "crossing_control", m & ~sig & ~nosig & ~ddot & ~covered, unsignalized, ev.INFERRED,
                         "derived:osm_marked_no_signal_inventory", osm_as_of)
    ev.fill_where_better(edges, "crossing_control", m & ~sig & ~nosig & ddot, unsignalized, ev.INFERRED,
                         "derived:osm_marked_signalization_not_recorded", osm_as_of)
    ev.fill_where_better(edges, "crossing_control", m & sig & nosig, unsignalized, ev.INFERRED,
                         "derived:osm_signal_tags_conflict", osm_as_of)
    edges["crossing_signal_tags_conflict"] = m & sig & nosig

    is_signal = edges["crossing_control"].eq("signal")
    signals_as_of = signal_inventory_as_of(signals_source)
    for attr, col, yes in (
        ("signal_lpi", "LPI", {"Yes"}),
        ("signal_aps", "APS", {"Yes"}),
        ("signal_hawk", "HAWK", {"Yes"}),
        ("signal_all_ped_phase", "ALL_PED_PHASE", {"All-Ped Phase", "TOD All-Ped Phase", "Barnes Dance"}),
    ):
        vals = pd.Series(None, index=edges.index, dtype="object")
        # an inventory that leaves the field empty says nothing: the attribute stays unknown
        v = j.loc[near, col].dropna()
        vals.loc[v.index] = v.isin(yes).to_numpy()
        if attr == "signal_aps":  # partial/planned APS is not an accessible signal today
            vals.loc[v.index[v.isin(["Planned", "Partial"])]] = False
        ev.init_unknown(edges, attr)
        ev.fill_where_better(edges, attr, m & is_signal, vals, ev.VERIFIED, signals_source, signals_as_of)

    # --- marking evidence from planimetric crosswalk polygons
    cw = planim[planim["DESCRIPTION"].str.strip().eq("Crosswalk")]
    cw_as_of = _ms_to_date(cw["CAPTUREYEAR"])
    ce = edges.loc[m, ["geometry"]]
    pairs = gpd.sjoin(ce, gpd.GeoDataFrame({"as_of": cw_as_of.values}, geometry=cw.geometry.values, crs=cw.crs),
                      predicate="intersects", how="inner")
    if len(pairs):
        cwg = cw.geometry.values
        overlap = [ce.geometry.loc[i].intersection(cwg[r]).length / max(ce.geometry.loc[i].length, 1e-9)
                   for i, r in zip(pairs.index, pairs["index_right"])]
        pairs = pairs.assign(overlap=overlap)
        best = pairs.sort_values("overlap", ascending=False)
        best = best[~best.index.duplicated()]
        hit = best[best["overlap"] >= CROSSWALK_POLY_MIN_OVERLAP]
        edges["planimetric_crosswalk"] = False
        edges.loc[hit.index, "planimetric_crosswalk"] = True
        edges["planimetric_crosswalk_as_of"] = None
        edges.loc[hit.index, "planimetric_crosswalk_as_of"] = hit["as_of"].astype(object)
        # unknown OSM control + painted crosswalk in planimetrics => marked (dated). Paint says
        # nothing about signals: it never upgrades a control that is only inferred, and where
        # OSM's signal tags conflict or a inventory signal is nearby the result stays INFERRED.
        painted_only = m & edges["planimetric_crosswalk"] & edges["crossing_control_status"].eq(ev.UNKNOWN)
        # and without a signal inventory "no signal nearby" is not known either
        signal_open = (sig & nosig) | (ddot & ~nosig) | ~covered
        marked_val = pd.Series("marked_uncontrolled", index=edges.index)
        ev.fill_where_better(edges, "crossing_control", painted_only & ~signal_open, marked_val, ev.VERIFIED,
                             planim_source, edges["planimetric_crosswalk_as_of"])
        ev.fill_where_better(edges, "crossing_control", painted_only & signal_open, marked_val, ev.INFERRED,
                             planim_source, edges["planimetric_crosswalk_as_of"])
    else:
        edges["planimetric_crosswalk"] = False
        edges["planimetric_crosswalk_as_of"] = None


def enrich_crossed_road(edges: gpd.GeoDataFrame, roadway: gpd.GeoDataFrame, source: str = "roadway_block",
                        as_of: str | None = None) -> None:
    """Attributes of the road being crossed, from the roadway inventory block (DC: DDOT
    Roadway Block) the crossing intersects most perpendicularly. ``as_of`` dates values
    that carry no year of their own (for the OSM fallback: the extract's date)."""
    m = edges["kind"] == "crossing"
    groups = crossing_groups(edges)
    # use the full crossing (union of its fragments) to find the road
    cg = edges.loc[m, ["geometry"]].assign(grp=groups[m]).dissolve("grp")
    cg["geometry"] = cg.geometry.apply(lambda g: shapely.line_merge(g) if g.geom_type == "MultiLineString" else g)
    rb = roadway[["TOTALTRAVELLANES", "TOTALCROSSSECTIONWIDTH", "SPEEDLIMITS_OB", "AADT", "AADT_YEAR",
                  "DCFUNCTIONALCLASS", "ROUTENAME", "geometry"]].reset_index(drop=True)
    pairs = gpd.sjoin(cg, rb, predicate="intersects", how="inner")

    def _angle(a, b) -> float:
        def bearing(g):
            c = np.asarray(g.coords if g.geom_type == "LineString" else g.geoms[0].coords)
            d = c[-1] - c[0]
            return np.arctan2(d[1], d[0])
        diff = abs(bearing(a) - bearing(b)) % np.pi
        return min(diff, np.pi - diff)  # 0 = parallel, pi/2 = perpendicular

    if len(pairs):
        ang = [_angle(cg.geometry.loc[g], rb.geometry.iloc[r]) for g, r in zip(pairs.index, pairs["index_right"])]
        pairs = pairs.assign(angle=ang).sort_values(["angle", "TOTALTRAVELLANES"], ascending=False)
        best = pairs[~pairs.index.duplicated()]
        best = best[best["angle"] >= np.deg2rad(30)]  # reject near-parallel matches
    else:
        best = pairs
    lookup = best.reindex(groups[m].to_numpy())
    lookup.index = groups[m].index

    def put(attr: str, values: pd.Series, value_as_of=None) -> None:
        ev.init_unknown(edges, attr)
        ev.fill_where_better(edges, attr, m, values.reindex(edges.index), ev.VERIFIED, source,
                             as_of if value_as_of is None else value_as_of)

    lanes = lookup["TOTALTRAVELLANES"].where(lookup["TOTALTRAVELLANES"] > 0)
    put("crossed_lanes", lanes)
    width = pd.to_numeric(lookup["TOTALCROSSSECTIONWIDTH"], errors="coerce")
    put("crossed_width_m", (width.where(width > 0) * FT).round(1))
    put("crossed_speed_mph", pd.to_numeric(lookup["SPEEDLIMITS_OB"], errors="coerce"))
    aadt_year = lookup["AADT_YEAR"].dropna().astype(int).astype(str).reindex(lookup.index)
    put("crossed_aadt", pd.to_numeric(lookup["AADT"], errors="coerce"), aadt_year.reindex(edges.index))
    # the full name with its type and quadrant ("MINNESOTA AVE NE"): "MINNESOTA" alone could be a
    # street, an avenue or a place, in any quadrant
    edges["crossed_street"] = lookup["ROUTENAME"].reindex(edges.index)

    # crossing length: full physical crossing length, shared by its fragments
    total = edges.loc[m].groupby(groups[m])["length_m"].sum()
    edges["crossing_total_length_m"] = groups.map(total).astype(float)


KERB_PASSABLE = {"lowered", "flush", "ramp_present"}
KERB_BARRIER = {"raised", "no_ramp"}


def _crossing_outer_ends(edges: gpd.GeoDataFrame) -> tuple[pd.Series, gpd.GeoSeries, gpd.GeoSeries]:
    """The two outer ends (at the curbs) of each physical crossing.

    Crossing ways are often split at the road centerline node; a fragment's inner end is in
    the middle of the street where no kerb exists. Outer ends are the endpoints of the
    merged crossing geometry (or, if fragments do not merge into one line, the two
    fragment endpoints farthest apart)."""
    m = edges["kind"] == "crossing"
    groups = crossing_groups(edges)
    merged = edges.loc[m, ["geometry"]].assign(grp=groups[m]).dissolve("grp").geometry
    a_pts, b_pts = [], []
    for g in merged.values:
        g = shapely.line_merge(g) if g.geom_type == "MultiLineString" else g
        if g.geom_type == "LineString":
            a_pts.append(shapely.Point(g.coords[0]))
            b_pts.append(shapely.Point(g.coords[-1]))
        else:
            pts = np.array([c for part in g.geoms for c in (part.coords[0], part.coords[-1])])
            dmat = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
            i, j = np.unravel_index(np.argmax(dmat), dmat.shape)
            a_pts.append(shapely.Point(pts[i]))
            b_pts.append(shapely.Point(pts[j]))
    a = gpd.GeoSeries(a_pts, index=merged.index, crs=edges.crs)
    b = gpd.GeoSeries(b_pts, index=merged.index, crs=edges.crs)
    return groups, a, b


def _inventory_kerb(ramps: gpd.GeoDataFrame) -> pd.Series:
    """ramp_present for each record of a ramp inventory, or what it records (KERB: a surveyed
    corner may have no ramp). An inventory of ramps alone (DC 2010) records only ramps."""
    if "KERB" not in ramps:
        return pd.Series("ramp_present", index=ramps.index, dtype="object")
    return ramps["KERB"].fillna("ramp_present")


def _inventory_check(ramps: gpd.GeoDataFrame) -> pd.Series:
    """The inventory's own verdict on each ramp (CHECK), or none when it gives none."""
    return ramps["CHECK"] if "CHECK" in ramps else pd.Series(None, index=ramps.index, dtype="object")


def enrich_kerbs(edges: gpd.GeoDataFrame, osm_nodes: gpd.GeoDataFrame, ps: gpd.GeoDataFrame,
                 ramps2010: gpd.GeoDataFrame, ps_source: str | None = "project_sidewalk_dc",
                 ramps_source: str | None = "sidewalk_ramps_2010",
                 field: gpd.GeoDataFrame | None = None, field_source: str | None = None,
                 parcels: gpd.GeoDataFrame | None = None) -> None:
    """Curb condition at both outer ends of each physical crossing: ``kerb_a`` / ``kerb_b``
    in {lowered, flush, raised, ramp_present, no_ramp}, copied to every fragment of the
    crossing. Passable for wheelchairs: KERB_PASSABLE.

    The team's field checks come first: a curb seen on site is the most recent observation
    of it. Where one says there is no ramp (or a flush curb), an inventory verdict on a ramp
    nearby is not carried, since it describes a ramp that is not at this end.

    Last, where no source records the curb, a crossing end in a post-ADA neighbourhood
    (``post_ada``, tested 2026-10-21) gets an inferred ramp."""
    m = edges["kind"] == "crossing"
    osm_as_of = _osm_as_of()
    groups, end_a, end_b = _crossing_outer_ends(edges)
    ends = {"a": end_a, "b": end_b}
    kerb = osm_nodes[osm_nodes["kerb"].notna() | osm_nodes["barrier"].eq("kerb")].copy()
    kv = kerb["kerb"].fillna("unspecified").str.split(";").str[0]
    kv = kv.replace({"yes": None, "unspecified": None, "no": "flush"})  # kerb=no => no kerb => flush
    kerb = kerb.assign(kval=kv)[["kval", "geometry"]]
    kerb = kerb[kerb["kval"].isin(["lowered", "flush", "raised"])]

    ps_ramp = ps[ps["label_type"].isin(["CurbRamp", "NoCurbRamp"])].copy()
    ps_ramp["kval"] = ps_ramp["label_type"].map({"CurbRamp": "ramp_present", "NoCurbRamp": "no_ramp"})
    ps_ramp["as_of"] = ps_ramp["image_date"]
    # a ramp inventory says ramp_present for each record, or no_ramp for a surveyed corner
    # without one (KERB), and may carry its own verdict on the ramp (CHECK)
    r10 = ramps2010.assign(kval=_inventory_kerb(ramps2010), CHECK=_inventory_check(ramps2010),
                           as_of=_ms_to_date(ramps2010["CAPTUREYEAR"]))

    c = field[field["KIND"].eq("corner")] if field is not None else None
    fc = (gpd.GeoDataFrame({"kval": c["KERB"].to_numpy(), "as_of": c["VISITED"].to_numpy()}, geometry=c.geometry.values, crs=edges.crs)
          if c is not None and len(c) else gpd.GeoDataFrame({"kval": [], "as_of": []}, geometry=[], crs=edges.crs))

    for end, pts in ends.items():
        attr = f"kerb_{end}"
        ev.init_unknown(edges, attr)
        p = gpd.GeoDataFrame({"grp": pts.index}, geometry=pts.values, crs=edges.crs)
        for src_gdf, radius, status, src in (
            (fc, FIELD_CHECK_RADIUS_M, ev.VERIFIED, field_source),
            (kerb.assign(as_of=osm_as_of), KERB_ENDPOINT_RADIUS_M, ev.VERIFIED, OSM),
            (ps_ramp[["kval", "as_of", "geometry"]], PS_RAMP_RADIUS_M, ev.VERIFIED, ps_source),
            (r10[["kval", "as_of", "geometry"]], RAMP2010_RADIUS_M, ev.VERIFIED, ramps_source),
        ):
            if src is None or src_gdf.empty:  # the region has no such source, or it holds nothing
                continue
            j = gpd.sjoin_nearest(p, src_gdf, how="inner", max_distance=radius, distance_col="d")
            j = j.sort_values("d").drop_duplicates("grp").set_index("grp")
            vals = groups.map(j["kval"])
            as_of = groups.map(j["as_of"])
            ev.fill_where_better(edges, attr, m, vals, status, src, as_of)

        # the inventory's verdict on the ramp at this end, from the same nearest record
        if parcels is not None and len(parcels):
            pa = post_ada.neighbourhood(gpd.GeoSeries(pts.values, index=pts.index, crs=edges.crs),
                                        post_ada.built_parcels(parcels))["post_ada"]
            vals = groups.map(pd.Series(np.where(pa, "ramp_present", None), index=pa.index, dtype="object"))
            ev.fill_where_better(edges, attr, m, vals, ev.INFERRED, post_ada.SOURCE, post_ada.as_of(parcels))

        check = f"ramp_check_{end}"
        ev.init_unknown(edges, check)
        if ramps_source is not None and len(r10):
            j = gpd.sjoin_nearest(p, r10[["CHECK", "as_of", "geometry"]], how="inner",
                                  max_distance=RAMP2010_RADIUS_M, distance_col="d")
            j = j.sort_values("d").drop_duplicates("grp").set_index("grp")
            ev.fill_where_better(edges, check, m, groups.map(j["CHECK"]), ev.VERIFIED, ramps_source, groups.map(j["as_of"]))
        seen_no_ramp = edges[f"{attr}_source"].eq(field_source) & edges[attr].ne("ramp_present") if field_source else None
        if seen_no_ramp is not None and seen_no_ramp.any():
            edges.loc[seen_no_ramp, [check, f"{check}_source", f"{check}_as_of"]] = None
            edges.loc[seen_no_ramp, f"{check}_status"] = ev.UNKNOWN


# ================================================================ accessibility


def enrich_step_free(edges: gpd.GeoDataFrame, planim: gpd.GeoDataFrame, planim_source: str | None = "sidewalks_2021") -> None:
    osm_as_of = _osm_as_of()
    ev.init_unknown(edges, "step_free")
    steps = edges["kind"].isin(["steps", "escalator"])
    ev.fill_where_better(edges, "step_free", steps, pd.Series(False, index=edges.index), ev.VERIFIED, OSM, osm_as_of)

    # planimetric stairs polygons crossing a non-steps edge reveal unmapped stairs
    stairs = planim[planim["DESCRIPTION"].str.strip().eq("Stairs")]
    st = gpd.GeoDataFrame({"as_of": _ms_to_date(stairs["CAPTUREYEAR"]).values}, geometry=stairs.geometry.values, crs=stairs.crs)
    cand = edges.loc[~steps & edges["representation"].eq("pedestrian_way"), ["geometry"]]
    pairs = gpd.sjoin(cand, st, predicate="intersects", how="inner")
    if len(pairs):
        ov = [cand.geometry.loc[i].intersection(st.geometry.iloc[r]).length for i, r in zip(pairs.index, pairs["index_right"])]
        pairs = pairs.assign(ov=ov)
        hit = pairs[pairs["ov"] >= STAIRS_POLY_MIN_OVERLAP_M]
        hit = hit[~hit.index.duplicated()]
        edges["planimetric_stairs_overlap"] = False
        edges.loc[hit.index, "planimetric_stairs_overlap"] = True
        ev.fill_where_better(edges, "step_free", edges.index.isin(hit.index), pd.Series(False, index=edges.index),
                             ev.VERIFIED, planim_source, hit["as_of"].reindex(edges.index))
    else:
        edges["planimetric_stairs_overlap"] = False

    # Gap-bridging connectors are links this pipeline invented, so nothing about their
    # passability was observed and they can never be VERIFIED step-free. They are also too
    # short (<= GAP_TOLERANCE_M) for the STAIRS_POLY_MIN_OVERLAP_M test above, and a gap
    # between two sidewalk ends that sits on a stairs polygon is most likely the unmapped
    # stairs themselves, so any contact with a stairs polygon marks the connector not
    # step-free.
    conn = edges["representation"].eq("inferred_connector")
    cpairs = gpd.sjoin(edges.loc[conn, ["geometry"]], st, predicate="intersects", how="inner")
    cpairs = cpairs[~cpairs.index.duplicated()]
    conn_hit = edges.index.isin(cpairs.index)
    edges.loc[conn_hit, "planimetric_stairs_overlap"] = True
    ev.fill_where_better(edges, "step_free", conn_hit, pd.Series(False, index=edges.index),
                         ev.INFERRED, planim_source, cpairs["as_of"].reindex(edges.index))
    ev.fill_where_better(edges, "step_free", conn & ~conn_hit, pd.Series(True, index=edges.index),
                         ev.INFERRED, "derived:inferred_connector_no_stairs_polygon")

    # A mapper's wheelchair tag is a direct observation of what this attribute stands for
    # (passable without steps by a wheelchair user). ``wheelchair=no`` fails the requirement
    # whatever the reason; ``yes`` verifies it unless an inventory above found stairs.
    wc = edges["wheelchair"].fillna("").str.lower()
    ev.fill_where_better(edges, "step_free", wc.eq("no"), pd.Series(False, index=edges.index),
                         ev.VERIFIED, OSM, osm_as_of)
    ev.fill_where_better(edges, "step_free", wc.isin(["yes", "designated"]), pd.Series(True, index=edges.index),
                         ev.VERIFIED, OSM, osm_as_of)

    # Remaining edges. Stairs are captured by two independent inventories: OSM (steps are
    # mapped as their own ways) and planimetrics (in DC a comprehensive aerial inventory of
    # stairs in public space). Where an edge is on-street and inside planimetric coverage
    # (``covered_planimetric``) and neither inventory has stairs, step-free is VERIFIED by
    # confirmed absence. Off-street paths (parks, NPS land, campuses, tree canopy), the buffer
    # beyond the region's line, and a region without such an inventory are outside reliable
    # coverage, so there it is only INFERRED.
    planim_as_of = _max_capture_date(stairs["CAPTUREYEAR"])
    rest = edges["step_free_status"].eq(ev.UNKNOWN)
    true = pd.Series(True, index=edges.index)
    covered = edges["covered_planimetric"].astype(bool)
    ev.fill_where_better(edges, "step_free", rest & covered, true, ev.VERIFIED,
                         "derived:absent_from_osm_steps_and_planimetric_stairs", planim_as_of)
    ev.fill_where_better(edges, "step_free", rest & edges["offstreet"], true, ev.INFERRED,
                         "derived:absent_from_osm_steps_offstreet")
    ev.fill_where_better(edges, "step_free", rest & ~covered & ~edges["offstreet"], true, ev.INFERRED,
                         "derived:absent_from_osm_steps_outside_inventory")


SURFACE_GOOD = {"paved", "asphalt", "concrete", "concrete:plates", "concrete:lanes", "paving_stones", "metal", "wood", "rubber", "tartan"}
SURFACE_ROUGH = {"sett", "cobblestone", "unhewn_cobblestone", "bricks", "brick", "compacted", "fine_gravel"}
SMOOTHNESS_ROUGH = {"bad"}
SMOOTHNESS_BAD = {"very_bad", "horrible", "very_horrible", "impassable"}
SURFACE_BAD = {"gravel", "unpaved", "dirt", "ground", "grass", "sand", "mud", "earth", "pebblestone", "woodchips", "rock", "stepping_stones"}


def enrich_surface(edges: gpd.GeoDataFrame) -> None:
    s = edges["surface"].fillna("").str.lower()
    cls = pd.Series(None, index=edges.index, dtype="object")
    cls[s.isin(SURFACE_GOOD)] = "smooth"
    cls[s.isin(SURFACE_ROUGH)] = "rough"
    cls[s.isin(SURFACE_BAD)] = "unpaved"
    # smoothness is the mapper's judgment of the surface as found; it can only make the
    # class worse than the material alone suggests, never better
    sm = edges["smoothness"].fillna("").str.lower()
    cls[sm.isin(SMOOTHNESS_ROUGH) & ~cls.eq("unpaved")] = "rough"
    cls[sm.isin(SMOOTHNESS_BAD)] = "unpaved"
    ev.init_unknown(edges, "surface_class")
    ev.fill_where_better(edges, "surface_class", pd.Series(True, index=edges.index), cls, ev.VERIFIED, OSM, _osm_as_of())


def enrich_project_sidewalk_counts(edges: gpd.GeoDataFrame, ps: gpd.GeoDataFrame) -> None:
    """Counts of human-labeled problems near each edge (context + Phase 2 features)."""
    for label, col, radius in (("Obstacle", "ps_obstacles", 5.0), ("SurfaceProblem", "ps_surface_problems", 5.0),
                               ("NoSidewalk", "ps_no_sidewalk", 12.0)):
        pts = ps[ps["label_type"].eq(label)][["median_severity", "geometry"]]
        j = gpd.sjoin_nearest(pts, edges[["geometry"]], how="inner", max_distance=radius)
        counts = j.groupby("index_right").size()
        edges[col] = counts.reindex(edges.index).fillna(0).astype(int)


def _side_coverage_polygons(lines: gpd.GeoSeries, polys: gpd.GeoSeries, far_left=None,
                            far_right=None, offsets: bool = False) -> tuple[np.ndarray, ...]:
    from .build import PROBE_FAR_M, sidewalk_side_coverage

    return sidewalk_side_coverage(lines, polys.boundary,
                                  PROBE_FAR_M if far_left is None else far_left,
                                  PROBE_FAR_M if far_right is None else far_right, offsets=offsets)


# A one-way street edge with another one-way street edge running alongside (within the
# probes' reach, bearings within this angle) is one carriageway of a divided road. Its
# probes toward the twin stop halfway to it, so the far carriageway's sidewalk is not read
# as this one's (Gaithersburg: 36.7 km of "both sides" were divided-road halves, 2026-10-15).
DIVIDED_TWIN_MAX_ANGLE_DEG = 20.0


def divided_road_reach(lines: gpd.GeoSeries, oneway: pd.Series, others: gpd.GeoSeries,
                       others_oneway: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Per street, the probe reach on its left and right: PROBE_FAR_M, or half the distance to
    a parallel one-way twin on that side."""
    from .build import PROBE_FAR_M

    far_l = np.full(len(lines), PROBE_FAR_M)
    far_r = np.full(len(lines), PROBE_FAR_M)
    ow = oneway.fillna("").isin(["yes", "-1", "true", "1"]).to_numpy()
    tw = others[others_oneway.fillna("").isin(["yes", "-1", "true", "1"]).to_numpy()]
    if not ow.any() or tw.empty:
        return far_l, far_r
    tree = shapely.STRtree(np.asarray(tw.values))
    tgeoms = np.asarray(tw.values)

    def bearing(g):
        c = shapely.get_coordinates(g)
        return np.arctan2(c[-1, 1] - c[0, 1], c[-1, 0] - c[0, 0])

    for i, g in enumerate(lines.values):
        if not ow[i] or g.length < 1.0:
            continue
        mid = g.interpolate(0.5, normalized=True)
        a = g.interpolate(max(g.length / 2 - 1.0, 0)); b = g.interpolate(min(g.length / 2 + 1.0, g.length))
        tx, ty = b.x - a.x, b.y - a.y
        br = bearing(g)
        for j in tree.query(mid.buffer(PROBE_FAR_M)):
            t = tgeoms[j]
            d = t.distance(mid)
            if d <= 3.0 or t.equals(g):  # itself, or a continuation touching it
                continue
            diff = abs(bearing(t) - br) % np.pi
            if min(diff, np.pi - diff) > np.deg2rad(DIVIDED_TWIN_MAX_ANGLE_DEG):
                continue
            q = shapely.Point(shapely.get_coordinates(shapely.shortest_line(mid, t))[-1])
            left = tx * (q.y - mid.y) - ty * (q.x - mid.x) > 0
            if left:
                far_l[i] = min(far_l[i], d / 2)
            else:
                far_r[i] = min(far_r[i], d / 2)
    return far_l, far_r


# Centerline kinds the side probes do not grade. A parking aisle, lot road or alley has a
# street's or a building's sidewalk within the probes' reach (``build.PROBE_FAR_M``) without
# having one of its own: in Gaithersburg 5 of 6 sampled service edges read "both sides" that
# way (imagery, 2026-10-15). Only OSM tags grade them.
NO_PLANIMETRIC_PRESENCE_KINDS = ("service", "alley")
# Values of ``sidewalk_presence`` that say a sidewalk is there. "at_least_one_side": a survey
# recorded a sidewalk on one side and nothing on the other, where its silence proves nothing
# (outside planimetric coverage), so whether the other side has one is not known.
SIDEWALK_PRESENT = ("both", "one_side", "at_least_one_side")
# A sidewalk found on one side only is weaker evidence than one on both, so it must lie this
# close to the street (median probe distance). Beyond 15 m, 6 of 8 sampled Gaithersburg
# finds were building-front walks across a parking lot or the far side of a divided road;
# under 5 m, 8 of 8 were sidewalks at the curb (imagery, 2026-10-16). About 4% of one-side
# finds lie farther out.
ONE_SIDE_MAX_OFFSET_M = 15.0


def enrich_centerline_sidewalks(edges: gpd.GeoDataFrame, planim: gpd.GeoDataFrame,
                                planim_source: str | None = "sidewalks_2021") -> None:
    """For street-centerline edges: does a sidewalk exist alongside? Value in
    {both, one_side, at_least_one_side, none}."""
    m = edges["representation"].eq("street_centerline")
    osm_as_of = _osm_as_of()
    left, right = edges["sidewalk_left"].fillna("unknown"), edges["sidewalk_right"].fillna("unknown")
    yes = lambda s: s.isin(["yes", "separate"])  # noqa: E731
    osm_val = pd.Series(None, index=edges.index, dtype="object")
    known = ~left.eq("unknown") & ~right.eq("unknown") & ~left.eq("n/a")
    osm_val[known & yes(left) & yes(right)] = "both"
    osm_val[known & (yes(left) ^ yes(right))] = "one_side"
    osm_val[known & ~yes(left) & ~yes(right)] = "none"
    ev.init_unknown(edges, "sidewalk_presence")
    ev.fill_where_better(edges, "sidewalk_presence", m, osm_val, ev.VERIFIED, OSM, osm_as_of)
    if planim_source is None:
        return

    sw = planim[planim["DESCRIPTION"].str.strip().isin(["Sidewalk", "Hidden Sidewalk"])]
    todo = m & edges["sidewalk_presence_status"].eq(ev.UNKNOWN) & ~edges["kind"].isin(NO_PLANIMETRIC_PRESENCE_KINDS)
    streets = m & ~edges["kind"].isin(NO_PLANIMETRIC_PRESENCE_KINDS)
    far_l, far_r = divided_road_reach(edges.loc[todo, "geometry"], edges.loc[todo, "oneway"],
                                      edges.loc[streets, "geometry"], edges.loc[streets, "oneway"])
    cl, cr, off_l, off_r = _side_coverage_polygons(edges.loc[todo, "geometry"], sw.geometry, far_l, far_r, offsets=True)
    val = pd.Series(None, index=edges.index, dtype="object")
    lo, ro = cl >= 0.6, cr >= 0.6
    # the side a one-side find is on must be near the street (see ONE_SIDE_MAX_OFFSET_M)
    near_one = np.where(lo, off_l, off_r) <= ONE_SIDE_MAX_OFFSET_M
    measured = ~np.isnan(cl)
    v = np.where(lo & ro, "both", np.where(lo | ro, "one_side", "none"))
    v = np.where(measured, v, None)
    # A missing polygon is evidence of a missing sidewalk only inside inventory coverage:
    # planimetrics stop at the region's line and are unreliable off-street (tree canopy,
    # campuses, parks). Elsewhere a sidewalk found on one side is recorded as that and no more
    # (at_least_one_side), and none found records nothing.
    uncovered = ~edges.loc[todo, "covered_planimetric"].astype(bool).to_numpy()
    v = np.where(uncovered & (v == "one_side"), np.where(near_one, "at_least_one_side", None), v)
    v = np.where(uncovered & (v == "none"), None, v)
    val.loc[edges.index[todo]] = v
    # Dated by the oldest capture among the polygons beside the street: a survey mixing
    # captures (Montgomery: 2014 to 2023) must not lend a 2014 sidewalk its newest year.
    as_of = _side_capture_dates(edges.loc[todo & val.notna(), "geometry"], sw)
    ev.fill_where_better(edges, "sidewalk_presence", todo, val, ev.VERIFIED, planim_source, as_of)


# Matching walkways to a condition survey's lines (City of Frederick, 2026-10-16). A mapped
# sidewalk lies within SURVEY_MATCH_M of the surveyed line: at 3 m, 229 of Frederick's 327 km
# of OSM sidewalk edges match, and widening to 10 m adds only 4 km, so the rest is simply not
# in the survey. An edge is graded when the segments beside it cover SURVEY_MIN_SHARE of its
# length; each segment covering at least SURVEY_SEGMENT_MIN_SHARE of it counts, so a segment
# merely touching an end does not.
SURVEY_MATCH_M = 3.0
SURVEY_MIN_SHARE = 0.6
SURVEY_SEGMENT_MIN_SHARE = 0.25
# A street centerline's sidewalks run beside it, up to the side probes' reach; a surveyed line
# there counts only when it runs alongside (bearing within this angle), not across a corner.
SURVEY_PARALLEL_MAX_DEG = 20.0


def _bearing(g) -> float:
    c = shapely.get_coordinates(g)
    return float(np.arctan2(c[-1, 1] - c[0, 1], c[-1, 0] - c[0, 0]))


def _survey_matches(lines: gpd.GeoSeries, seg_lines: gpd.GeoSeries, reach_m: float, parallel: bool) -> pd.DataFrame:
    """Pairs (edge index, segment position, share of the edge's length the segment covers)
    for segments within ``reach_m`` of each edge and covering SURVEY_SEGMENT_MIN_SHARE of it.
    The share is measured along the edge: the edge's length inside the segment's band."""
    if lines.empty or seg_lines.empty:
        return pd.DataFrame(columns=["eidx", "seg", "share"])
    bands = seg_lines.buffer(reach_m, cap_style="flat")
    tree = shapely.STRtree(np.asarray(bands.values))
    rows = []
    for eidx, g in zip(lines.index, lines.values):
        if g.length < 1.0:
            continue
        for s in tree.query(g, predicate="intersects"):
            if parallel:
                diff = abs(_bearing(g) - _bearing(seg_lines.iloc[s])) % np.pi
                if min(diff, np.pi - diff) > np.deg2rad(SURVEY_PARALLEL_MAX_DEG):
                    continue
            share = g.intersection(bands.iloc[s]).length / g.length
            if share >= SURVEY_SEGMENT_MIN_SHARE:
                rows.append((eidx, s, share))
    return pd.DataFrame(rows, columns=["eidx", "seg", "share"])


def enrich_sidewalk_condition(edges: gpd.GeoDataFrame, planim: gpd.GeoDataFrame,
                              planim_source: str | None = None) -> None:
    """The condition index and recorded width of the sidewalk an edge walks on, from a
    survey that records them per segment (planimetrics columns CONDITION, WIDTH_M, LINE;
    City of Frederick). Graded ``sidewalk_condition`` (the survey's own 0-100 index, higher is
    better) and ``sidewalk_width_m``.

    - Sidewalk-level edges: the segments the edge lies on (within SURVEY_MATCH_M). VERIFIED.
    - Street centerlines with a sidewalk on record: the segments running alongside, both
      sides. Which side the walker takes is not known, so the worse side stands for both;
      INFERRED.
    Several segments along one edge give the lowest index and the narrowest width, and an
    edge stays unknown when any segment counted for it has no value: missing data is never
    favorable. Dated by the earliest survey date among the segments.
    """
    for attr in ("sidewalk_condition", "sidewalk_width_m"):
        ev.init_unknown(edges, attr)
    if planim_source is None or "LINE" not in planim:
        return
    sw = planim[planim["DESCRIPTION"].str.strip().eq("Sidewalk")].reset_index(drop=True)
    seg_lines = gpd.GeoSeries(sw["LINE"].to_numpy(), crs=planim.crs)
    seg_date = pd.Series(_ms_to_date(sw["CAPTUREYEAR"]).to_numpy())
    from .build import PROBE_FAR_M

    side = edges["representation"].eq("pedestrian_way") & ~edges["kind"].isin(["crossing", "steps", "escalator"])
    presence_known = edges["sidewalk_presence_status"].ne(ev.UNKNOWN) & edges["sidewalk_presence"].isin(SIDEWALK_PRESENT)
    centre = edges["representation"].eq("street_centerline") & presence_known
    for sel, reach, parallel, status in ((side, SURVEY_MATCH_M, False, ev.VERIFIED),
                                         (centre, PROBE_FAR_M, True, ev.INFERRED)):
        pairs = _survey_matches(edges.loc[sel, "geometry"], seg_lines, reach, parallel)
        if pairs.empty:
            continue
        covered = pairs.groupby("eidx")["share"].sum()
        keep = covered[covered >= SURVEY_MIN_SHARE].index
        pairs = pairs[pairs["eidx"].isin(keep)]
        as_of = pairs.assign(d=seg_date.iloc[pairs["seg"]].to_numpy()).groupby("eidx")["d"].min()
        for attr, col in (("sidewalk_condition", "CONDITION"), ("sidewalk_width_m", "WIDTH_M")):
            v = pairs.assign(v=pd.to_numeric(sw[col], errors="coerce").iloc[pairs["seg"]].to_numpy())
            g = v.groupby("eidx")["v"]
            val = g.min().where(g.count() == g.size())  # any segment without a value: unknown
            m = edges.index.isin(val.dropna().index)
            ev.fill_where_better(edges, attr, pd.Series(m, index=edges.index), val.reindex(edges.index),
                                 status, planim_source, as_of.reindex(edges.index))


# Estimated sidewalk lines (owner's decision, 2026-10-16): where a survey shows a sidewalk
# beside a street that OSM maps only as a centerline, a line is drawn where the sidewalk is,
# for maps. The probes find the sidewalk's street-side edge; the line is drawn this far
# beyond it, half of a common 1.5 m sidewalk.
SIDEWALK_LINE_INSET_M = 0.75
# The line stops short of the junctions at either end by its offset (at most this share of
# its length), so lines along crossing streets do not run into each other at the corner.
SIDEWALK_LINE_END_TRIM_MAX = 0.25
SURVEY_SIDEWALK_LINES = "survey_sidewalks.parquet"  # in GRAPH_DIR, written by ``enrich``
SURVEY_LINE_KINDS = ("street_unknown", "street_with_sidewalk", "pedestrian_street")
OSM_SIDEWALK_KINDS = ("sidewalk", "footway", "path", "shared_path")


def estimate_survey_sidewalk_lines(edges: gpd.GeoDataFrame, planim: gpd.GeoDataFrame,
                                   planim_source: str | None) -> gpd.GeoDataFrame:
    """One estimated sidewalk line per street side where the region's survey shows a
    sidewalk (side coverage >= build.SIDE_COVERAGE_THRESHOLD) and OSM maps none (no
    separately mapped walkway covering that side, no ``sidewalk:<side>=no``).

    The lines are a map layer, not walkways: the router keeps walking the street centerline,
    whose ``sidewalk_presence`` carries the same survey evidence, because a line beside the
    street would have to bend back to every junction node and the network's connections
    would be invented. The sidewalk is the survey's (``source``, ``as_of``); where the line is
    drawn is estimated (status ``inferred``). Columns: edge_id, side, offset_m, status, source,
    as_of, geometry."""
    from .build import SIDE_COVERAGE_THRESHOLD, sidewalk_side_coverage

    cols = ["edge_id", "side", "offset_m", "status", "source", "as_of"]
    empty = gpd.GeoDataFrame({c: [] for c in cols}, geometry=gpd.GeoSeries([], crs=edges.crs), crs=edges.crs)
    if planim_source is None:
        return empty
    sw = planim[planim["DESCRIPTION"].str.strip().isin(["Sidewalk", "Hidden Sidewalk"])]
    cand = edges["representation"].eq("street_centerline") & edges["kind"].isin(SURVEY_LINE_KINDS)
    streets = edges["representation"].eq("street_centerline") & ~edges["kind"].isin(NO_PLANIMETRIC_PRESENCE_KINDS)
    if not cand.any() or sw.empty:
        return empty
    lines = edges.loc[cand, "geometry"]
    far_l, far_r = divided_road_reach(lines, edges.loc[cand, "oneway"],
                                      edges.loc[streets, "geometry"], edges.loc[streets, "oneway"])
    cl, cr, off_l, off_r = sidewalk_side_coverage(lines, sw.geometry.boundary, far_l, far_r, offsets=True)
    osm_ways = edges.loc[edges["representation"].eq("pedestrian_way") & edges["kind"].isin(OSM_SIDEWALK_KINDS), "geometry"]
    ol, or_ = sidewalk_side_coverage(lines, osm_ways, far_l, far_r) if len(osm_ways) else (np.zeros(len(lines)),) * 2
    dates = _side_capture_dates(lines, sw)
    rows = []
    for k, (eidx, g) in enumerate(zip(lines.index, lines.values)):
        for side, cov, osm_cov, off, tag in (("left", cl[k], ol[k], off_l[k], "sidewalk_left"),
                                             ("right", cr[k], or_[k], off_r[k], "sidewalk_right")):
            if not (cov >= SIDE_COVERAGE_THRESHOLD) or osm_cov >= SIDE_COVERAGE_THRESHOLD or np.isnan(off):
                continue
            if off > ONE_SIDE_MAX_OFFSET_M:  # too far out to be this street's sidewalk, or to draw it
                continue
            if str(edges.at[eidx, tag]) == "no":  # OSM says there is none: the less favorable reading
                continue
            d = float(off) + SIDEWALK_LINE_INSET_M
            line = g.offset_curve(d if side == "left" else -d)
            if line.is_empty or line.geom_type != "LineString":
                continue
            trim = min(d, SIDEWALK_LINE_END_TRIM_MAX * line.length)
            rows.append({"edge_id": int(edges.at[eidx, "edge_id"]), "side": side, "offset_m": round(d, 1),
                         "status": ev.INFERRED, "source": planim_source, "as_of": dates.get(eidx),
                         "geometry": substring(line, trim, line.length - trim)})
    if not rows:
        return empty
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=edges.crs)


def _side_capture_dates(lines: gpd.GeoSeries, polys: gpd.GeoDataFrame) -> pd.Series:
    """Per street edge, the earliest capture date of the planimetric polygons within the
    side probes' reach (``build.PROBE_FAR_M``); None where none lies there."""
    from .build import PROBE_FAR_M

    band = gpd.GeoDataFrame(geometry=lines.buffer(PROBE_FAR_M, cap_style="flat"), crs=lines.crs)
    pts = gpd.GeoDataFrame({"d": pd.to_datetime(polys["CAPTUREYEAR"], unit="ms", errors="coerce").values},
                           geometry=polys.geometry.values, crs=polys.crs)
    j = gpd.sjoin(band, pts, predicate="intersects", how="inner")
    first = j.groupby(level=0)["d"].min()
    return first.dt.date.astype("string").reindex(lines.index).astype(object).where(lambda s: s.notna(), None)


# ================================================================ unmodeled crossings


def enrich_implicit_crossings(edges: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, signals: gpd.GeoDataFrame,
                              roadway: gpd.GeoDataFrame, ps: gpd.GeoDataFrame, ramps2010: gpd.GeoDataFrame,
                              ps_source: str | None = "project_sidewalk_dc",
                              ramps_source: str | None = "sidewalk_ramps_2010",
                              field: gpd.GeoDataFrame | None = None, field_source: str | None = None,
                              parcels: gpd.GeoDataFrame | None = None) -> None:
    """Flag the ends of street-centerline edges where the walker crosses a street that has no
    crossing edge, so the cost model and requirements can treat that crossing like any other.

    Where sidewalks are not mapped separately the graph follows the street centerline, and
    walking through an intersection crosses the other street without traversing a
    ``crossing`` edge. Left alone, those crossings would cost nothing and need no curb ramp,
    making the least-mapped parts of the city (in DC, Wards 7 and 8) look better than the
    best-mapped ones. Missing data is never favorable.

    An end of a centerline edge is an implicit crossing when its node
    - lies on the centerline of a road removed because its sidewalks are mapped separately, or
    - joins three or more street edges (a street edge) / two or more (an alley or service road).
    Each flagged end stands for half a crossing: straight through a junction is two ends, one
    crossing. Turning a corner also counts as one, which overstates turns on purpose.

    Per end ``x`` in {u, v}:
    - ``xing_x``          bool
    - ``xing_x_control``  "signal" when an active inventory signal is at the junction, else null
                          (unknown). The junction node is the intersection itself, so slip
                          lanes and mid-block crossings cannot be confused with it.
    - ``xing_x_width_m``  widest roadway inventory block at the junction (which street is crossed
                          depends on the direction of travel), null if none
    - ``xing_x_kerb``     "no_curb" (walking in the roadway: no sidewalk, alley, service road),
                          "ramp_likely" (>= JUNCTION_MIN_RAMPS ramp records at the corners and
                          no missing-ramp label), "no_ramp_reported", or "unknown"
    - ``xing_x_kerb_source`` / ``xing_x_kerb_as_of``  what backs a known kerb value: the ramp
                          sources counted and their oldest record's date, or for "no_curb" OSM
                          (alley, service road, street without sidewalk) or the survey that
                          shows no sidewalk; null where the kerb is unknown
    A junction with no ramp records in a post-ADA neighbourhood (``post_ada``, tested
    2026-10-21) is "ramp_likely" from the parcels' years built. A field check of the junction
    (within FIELD_CHECK_RADIUS_M of its node) replaces either: it was seen on site, and is dated
    by the visit.
    Everything here is inferred; an implicit crossing never satisfies "verified only".
    """
    cl = edges["representation"].eq("street_centerline")
    street = cl & edges["kind"].isin(STREET_KINDS)
    shared = cl & edges["kind"].isin(SHARED_ROAD_KINDS)
    street_deg = pd.concat([edges.loc[street, "u"], edges.loc[street, "v"]]).value_counts()
    on_removed = set(nodes.loc[nodes["on_removed_road"], "node_id"])
    flags = {}
    for end in ("u", "v"):
        sd = edges[end].map(street_deg).fillna(0)
        rem = edges[end].isin(on_removed)
        flags[end] = (street & ((sd >= 3) | rem)) | (shared & ((sd >= 2) | rem))
    jn_ids = set(edges.loc[flags["u"], "u"]) | set(edges.loc[flags["v"], "v"])
    jn = nodes.loc[nodes["node_id"].isin(jn_ids), ["node_id", "geometry"]].reset_index(drop=True)

    sg = signals[signals["STATUS"].eq("Active")][["geometry"]]
    sj = gpd.sjoin_nearest(jn, sg, how="inner", max_distance=JUNCTION_SIGNAL_RADIUS_M, distance_col="sig_d")
    street_junction = sj["node_id"].map(street_deg).fillna(0).ge(3) | sj["node_id"].isin(on_removed)
    has_signal = set(sj.loc[street_junction | sj["sig_d"].le(DRIVEWAY_SIGNAL_RADIUS_M), "node_id"])
    wj = gpd.sjoin_nearest(jn, roadway[["TOTALCROSSSECTIONWIDTH", "geometry"]], how="inner",
                           max_distance=JUNCTION_ROAD_RADIUS_M)
    width_ft = pd.to_numeric(wj["TOTALCROSSSECTIONWIDTH"], errors="coerce")
    width = (width_ft.where(width_ft > 0) * FT).groupby(wj["node_id"]).max().round(1)

    corners = gpd.GeoDataFrame({"node_id": jn["node_id"].to_numpy()},
                               geometry=jn.geometry.buffer(JUNCTION_RAMP_RADIUS_M).to_numpy(), crs=edges.crs)

    # every ramp / missing-ramp record at a corner, with the source and date it carries
    inventory_kerb = _inventory_kerb(ramps2010)
    recs = []
    if ps_source is not None:
        p = ps[ps["label_type"].isin(["CurbRamp", "NoCurbRamp"])]
        recs.append(gpd.GeoDataFrame({"kval": p["label_type"].map({"CurbRamp": "ramp_present", "NoCurbRamp": "no_ramp"}),
                                      "src": ps_source, "as_of": p["image_date"]}, geometry=p.geometry, crs=edges.crs))
    if ramps_source is not None:
        r = ramps2010[inventory_kerb.isin(["ramp_present", "no_ramp"])]
        recs.append(gpd.GeoDataFrame({"kval": inventory_kerb[r.index], "src": ramps_source,
                                      "as_of": _ms_to_date(r["CAPTUREYEAR"])}, geometry=r.geometry, crs=edges.crs))
    recs = [g for g in recs if len(g)]
    at = (gpd.sjoin(pd.concat(recs, ignore_index=True), corners, predicate="within") if recs
          else pd.DataFrame(columns=["kval", "src", "as_of", "node_id"]))
    at["as_of"] = at["as_of"].astype(object).where(at["as_of"].notna(), None)

    def evidence(kval: str) -> pd.DataFrame:
        """Per junction: how many records say ``kval``, which sources, and the oldest record's
        date (a count of several records is as old as its oldest one; any undated record
        leaves the date unknown)."""
        g = at[at["kval"].eq(kval)].groupby("node_id")
        return pd.DataFrame({"n": g.size(), "src": g["src"].agg(lambda s: ", ".join(sorted(set(s)))),
                             "as_of": g["as_of"].agg(lambda s: None if s.isna().any() else min(s))})

    ramps, no_ramp = evidence("ramp_present"), evidence("no_ramp")
    node_kerb = pd.Series("unknown", index=jn["node_id"].to_numpy(), dtype="object")
    node_src = pd.Series(None, index=node_kerb.index, dtype="object")
    node_as_of = pd.Series(None, index=node_kerb.index, dtype="object")
    for value, e, hit in (("ramp_likely", ramps, ramps["n"] >= JUNCTION_MIN_RAMPS), ("no_ramp_reported", no_ramp, no_ramp["n"] > 0)):
        ids = e.index[hit]
        node_kerb[ids], node_src[ids], node_as_of[ids] = value, e.loc[ids, "src"], e.loc[ids, "as_of"]
    if parcels is not None and len(parcels):
        pa = post_ada.neighbourhood(gpd.GeoSeries(jn.geometry.values, index=jn["node_id"].to_numpy(), crs=edges.crs),
                                    post_ada.built_parcels(parcels))["post_ada"]
        ids = node_kerb.index[node_kerb.eq("unknown").to_numpy() & pa.reindex(node_kerb.index).fillna(False).to_numpy()]
        node_kerb[ids], node_src[ids], node_as_of[ids] = "ramp_likely", post_ada.SOURCE, post_ada.as_of(parcels)
    if field is not None and field_source is not None:
        fj = field[field["KIND"].eq("junction")]
        if len(fj):
            hit = gpd.sjoin_nearest(jn, fj[["KERB", "VISITED", "geometry"]], how="inner",
                                    max_distance=FIELD_CHECK_RADIUS_M, distance_col="d")
            hit = hit.sort_values("d").drop_duplicates("node_id").set_index("node_id")
            node_kerb[hit.index], node_src[hit.index], node_as_of[hit.index] = hit["KERB"], field_source, hit["VISITED"]

    presence_known = edges["sidewalk_presence_status"].ne(ev.UNKNOWN)
    by_kind = shared | edges["kind"].eq("street_no_sidewalk")  # OSM says the walker is in the roadway
    by_survey = ~by_kind & presence_known & edges["sidewalk_presence"].eq("none")
    in_roadway = by_kind | by_survey
    osm_as_of = _osm_as_of()
    for end in ("u", "v"):
        f = flags[end]
        n = edges[end]
        edges[f"xing_{end}"] = f
        edges[f"xing_{end}_control"] = pd.Series(np.where(f & n.isin(has_signal), "signal", None),
                                                 index=edges.index, dtype="object")
        edges[f"xing_{end}_width_m"] = n.map(width).where(f).astype(float)
        kerb = n.map(node_kerb).where(~in_roadway, "no_curb")
        edges[f"xing_{end}_kerb"] = kerb.where(f).astype("object")
        # what backs the curb value: the ramp records at the junction, or what put the walker in the roadway
        src = n.map(node_src).where(~in_roadway, None)
        src = src.where(~by_kind, OSM).where(~by_survey, edges["sidewalk_presence_source"])
        as_of = n.map(node_as_of).where(~in_roadway, None)
        as_of = as_of.where(~by_kind, osm_as_of).where(~by_survey, edges["sidewalk_presence_as_of"])
        edges[f"xing_{end}_kerb_source"] = src.where(f & kerb.ne("unknown")).astype("object")
        edges[f"xing_{end}_kerb_as_of"] = as_of.where(f & kerb.ne("unknown")).astype("object")
    log.info("implicit crossings: %d edge ends at %d junctions (%d with a inventory signal)",
             int(flags["u"].sum() + flags["v"].sum()), len(jn), len(has_signal))


def enrich_along_street(edges: gpd.GeoDataFrame, roadway: gpd.GeoDataFrame) -> None:
    """``along_street``: the street a separately mapped sidewalk runs beside. OSM sidewalks
    carry no name, so guidance could not say which street a turn leads onto. The name is
    that of the nearest roadway inventory block within ALONG_STREET_MAX_M that runs the same way
    as the sidewalk. It is used for wording only: it grades nothing and enters no cost."""
    edges["along_street"] = None
    m = edges["kind"].eq("sidewalk") & (edges["length_m"] >= ALONG_STREET_MIN_EDGE_M)
    if not m.any() or "ROUTENAME" not in roadway:
        return
    side = edges.loc[m, ["geometry"]]
    mid = side.geometry.interpolate(0.5, normalized=True)
    rb = roadway[["ROUTENAME", "geometry"]].reset_index(drop=True)
    pairs = gpd.sjoin(gpd.GeoDataFrame(geometry=mid.buffer(ALONG_STREET_MAX_M), crs=edges.crs), rb[["geometry"]],
                      predicate="intersects", how="inner")
    if pairs.empty:
        return
    roads = rb.geometry.to_numpy()[pairs["index_right"].to_numpy()]
    points = mid.loc[pairs.index].to_numpy()
    dist = shapely.distance(points, roads)
    # the street's direction where the sidewalk is beside it (a block can curve)
    at = shapely.line_locate_point(roads, points)
    a = shapely.get_coordinates(shapely.line_interpolate_point(roads, np.maximum(at - ALONG_STREET_LOCAL_M, 0.0)))
    b = shapely.get_coordinates(shapely.line_interpolate_point(roads, np.minimum(at + ALONG_STREET_LOCAL_M, shapely.length(roads))))
    walk = side.geometry.loc[pairs.index]
    w = shapely.get_coordinates(shapely.line_interpolate_point(walk.to_numpy(), 1.0, normalized=True)) \
        - shapely.get_coordinates(shapely.line_interpolate_point(walk.to_numpy(), 0.0, normalized=True))
    r = b - a
    diff = np.abs(np.arctan2(w[:, 1], w[:, 0]) - np.arctan2(r[:, 1], r[:, 0])) % np.pi
    angle = np.minimum(diff, np.pi - diff)  # 0 = parallel
    ok = angle <= np.deg2rad(ALONG_STREET_MAX_ANGLE_DEG)
    best = (pd.DataFrame({"edge": pairs.index.to_numpy()[ok], "dist": dist[ok],
                          "name": rb["ROUTENAME"].to_numpy()[pairs["index_right"].to_numpy()[ok]]})
            .sort_values("dist").drop_duplicates("edge").set_index("edge")["name"])
    edges.loc[best.index, "along_street"] = best


def enrich_offstreet(edges: gpd.GeoDataFrame, roadway: gpd.GeoDataFrame) -> None:
    """``offstreet``: farther than OFFSTREET_DIST_M from any street centerline (DC: DDOT
    roadway block), i.e. outside the public street right-of-way that inventories (signals,
    planimetrics) cover reliably. Used to decide when absence of a record means 'unknown'.

    ``covered_<family>`` (one per ``regions.COVERAGE_FAMILIES``) says where the absence of a
    record in that family's inventory counts as evidence: on-street and inside the region's
    line when the region's inventory covers its boundary, on every edge when it covers
    everything, nowhere when the region has none. An edge in the buffer beyond the line is
    not covered even when it runs along a boundary street."""
    d = gpd.sjoin_nearest(edges[["geometry"]], roadway[["geometry"]], how="left",
                          max_distance=OFFSTREET_DIST_M, distance_col="rd")
    rd = d.groupby(level=0)["rd"].min().reindex(edges.index)
    edges["offstreet"] = rd.isna()
    inside = ~edges["offstreet"] & ~edges["outside_region"].astype(bool)
    for family in COVERAGE_FAMILIES:
        scope = REGION.covers(family)
        edges[f"covered_{family}"] = inside if scope == "boundary" else pd.Series(scope == "everywhere", index=edges.index)


# ================================================================ traffic context


def enrich_centerline_traffic(edges: gpd.GeoDataFrame, roadway: gpd.GeoDataFrame, source: str = "roadway_block",
                              as_of: str | None = None) -> None:
    """Lanes / speed of the street itself for street-centerline edges (exposure when walking
    in or beside the roadway)."""
    m = edges["representation"].eq("street_centerline")
    rb = roadway[["TOTALTRAVELLANES", "SPEEDLIMITS_OB", "geometry"]]
    mid = gpd.GeoDataFrame(geometry=edges.loc[m].geometry.interpolate(0.5, normalized=True), crs=edges.crs)
    j = gpd.sjoin_nearest(mid, rb, how="left", max_distance=15.0, distance_col="d")
    j = j[~j.index.duplicated()]
    ev.init_unknown(edges, "street_lanes")
    ev.fill_where_better(edges, "street_lanes", m, j["TOTALTRAVELLANES"].where(j["TOTALTRAVELLANES"] > 0).reindex(edges.index),
                         ev.VERIFIED, source, as_of)
    ev.init_unknown(edges, "street_speed_mph")
    ev.fill_where_better(edges, "street_speed_mph", m, pd.to_numeric(j["SPEEDLIMITS_OB"], errors="coerce").reindex(edges.index),
                         ev.VERIFIED, source, as_of)


# ================================================================ entry point

GRADED_ATTRS = [
    "crossing_control", "signal_lpi", "signal_aps", "signal_hawk", "signal_all_ped_phase",
    "crossed_lanes", "crossed_width_m", "crossed_speed_mph", "crossed_aadt",
    "kerb_a", "kerb_b", "ramp_check_a", "ramp_check_b", "step_free", "surface_class", "sidewalk_presence",
    "sidewalk_condition", "sidewalk_width_m",
    "street_lanes", "street_speed_mph", "grade_avg_pct", "grade_max_pct",
]


def _prepare_project_sidewalk() -> tuple[gpd.GeoDataFrame, str | None]:
    ps, sid = roles.read("project_sidewalk")
    d = pd.to_datetime(ps["avg_image_capture_date"], utc=True, errors="coerce")
    d = d.where(d.dt.year >= PS_MIN_VALID_YEAR)
    ps["image_date"] = d.dt.date.astype("string")
    return ps, sid


def enrich() -> gpd.GeoDataFrame:
    edges = gpd.read_parquet(GRAPH_DIR / "edges_base.parquet")
    nodes = gpd.read_parquet(GRAPH_DIR / "nodes.parquet")
    osm_nodes = gpd.read_parquet(INTERIM_DIR / "osm_nodes.parquet").to_crs(ANALYSIS_CRS)
    signals, signals_src = roles.read("signals")
    planim, planim_src = roles.read("planimetrics")
    roadway, roadway_src = roles.read("roadway")
    ramps, ramps_src = roles.read("ramps")
    ps, ps_src = _prepare_project_sidewalk()
    field, field_src = roles.read("field_checks")
    parcels, parcels_src = roles.read("parcels")
    # Without a roadway inventory the streets come from the region's own OSM extract
    # (roles.osm_streets); what they say is OSM's evidence, dated as the extract.
    if roadway_src == roles.OSM_ROADWAY:
        roadway_src, roadway_as_of = OSM, _osm_as_of()
    else:
        roadway_as_of = None
    log.info("enrich %s: signals=%s planimetrics=%s roadway=%s ramps=%s project_sidewalk=%s field_checks=%s (%d answered)",
             REGION.id, signals_src, planim_src, roadway_src, ramps_src, ps_src, field_src, len(field))

    steps = [
        ("offstreet", lambda: enrich_offstreet(edges, roadway)),
        ("crossing control", lambda: enrich_crossing_control(edges, signals, planim, osm_nodes, signals_src, planim_src)),
        ("crossed road", lambda: enrich_crossed_road(edges, roadway, roadway_src, roadway_as_of)),
        ("street alongside", lambda: enrich_along_street(edges, roadway)),
        ("kerbs", lambda: enrich_kerbs(edges, osm_nodes, ps, ramps, ps_src, ramps_src, field, field_src,
                                       parcels if parcels_src else None)),
        ("step-free", lambda: enrich_step_free(edges, planim, planim_src)),
        ("surface", lambda: enrich_surface(edges)),
        ("project sidewalk counts", lambda: enrich_project_sidewalk_counts(edges, ps)),
        ("centerline sidewalks", lambda: enrich_centerline_sidewalks(edges, planim, planim_src)),
        ("sidewalk condition", lambda: enrich_sidewalk_condition(edges, planim, planim_src)),
        ("implicit crossings", lambda: enrich_implicit_crossings(edges, nodes, signals, roadway, ps, ramps, ps_src, ramps_src,
                                                                     field, field_src, parcels if parcels_src else None)),
        ("centerline traffic", lambda: enrich_centerline_traffic(edges, roadway, roadway_src, roadway_as_of)),
        ("slope", lambda: enrich_slope(edges)),
    ]
    for name, fn in steps:
        log.info("enrich: %s", name)
        fn()
    for a in GRADED_ATTRS:
        ev.validate(edges, a)
    lines = estimate_survey_sidewalk_lines(edges, planim, planim_src)
    lines.to_parquet(GRAPH_DIR / SURVEY_SIDEWALK_LINES)
    log.info("estimated sidewalk lines: %d (%.1f km)", len(lines), lines.length.sum() / 1000)
    return edges
