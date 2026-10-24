"""Read the region's source for each role the pipeline uses, in one canonical shape.

The enrichment code reads DC's field names (``TOTALTRAVELLANES``, ``DESCRIPTION``, ...),
because DC was the first region. Those names are the canonical schema: a source for another
region is renamed and recoded into them by an adapter registered here, so ``enrich`` never
needs to know which city it is in.

A role the region has no source for comes back as an *empty* frame with the canonical
columns, and ``source`` None. Every join against it finds nothing, so each attribute it
would have graded stays ``unknown``; callers that would read an empty frame as "none here"
must check ``source`` and leave their output unknown instead.

One role has a fallback: without a roadway inventory, the street centerlines come from the
region's own OSM extract (names, ``lanes`` and ``maxspeed`` tags, as mapped). They place a
crossing on its street and say what is off-street; they carry OSM's evidence, no more.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

import geopandas as gpd
import numpy as np
import pandas as pd

from ..config import ANALYSIS_CRS, INTERIM_DIR
from ..fetch import raw_path
from ..sources import REGION

log = logging.getLogger(__name__)

CANONICAL: dict[str, tuple[str, ...]] = {
    "signals": ("STATUS", "LPI", "APS", "HAWK", "ALL_PED_PHASE", "PED_RECALL_MODE"),
    "planimetrics": ("DESCRIPTION", "CAPTUREYEAR"),
    "roadway": ("TOTALTRAVELLANES", "TOTALCROSSSECTIONWIDTH", "SPEEDLIMITS_OB", "AADT", "AADT_YEAR",
                "DCFUNCTIONALCLASS", "ROUTENAME"),
    # KERB: ramp_present | no_ramp (a surveyed corner without one); CHECK: the survey's
    # verdict on the ramp, passed | fails | no_tactile_pad, or null when it gives none
    "ramps": ("CAPTUREYEAR", "KERB", "CHECK"),
    "project_sidewalk": ("label_type", "median_severity", "avg_image_capture_date"),
    "report_districts": (),
    # KIND: corner (one curb at the end of a mapped crossing) | junction (a street crossed
    # where no crossing is mapped); KERB: enrich's value for that kind; VISITED: ISO day
    "field_checks": ("KIND", "KERB", "VISITED"),
    # YEARBLT: the main structure's year built (0 or blank: none); SDATDATE: "2026MAY"
    "parcels": ("YEARBLT", "SDATDATE"),
}

# Montgomery Planning's capture codes (SOURCE_CODE) -> capture year. The layer dates a
# polygon by its capture only; the year is carried as 1 January, as DC's CAPTUREYEAR is.
MONTGOMERY_CAPTURE_YEAR = {1: 2000, 2: 2005, 3: 2009, 4: 2012, 5: 2014, 6: 2017, 7: 2020, 8: 2023}
# Montgomery FEATURE_CODEs -> planimetric DESCRIPTION. Only 1206 "Street Sidewalk" is read
# as a sidewalk beside a street. 1208 "Non-Street Sidewalk" holds private front walks, patio
# walks and walks along building fronts; counted as sidewalk, it put "both sides" on 20.5 km
# of parking aisles, lot roads and alleys (checked on imagery 2026-10-15, 11 of 12 sampled
# edges wrong; the city's own layer agreed on 5 of 322 edges). It stays walkable surface for
# bridging gaps between path ends. Cost: from the 2017 capture on, Montgomery coded most new
# polygons 1208, so recent street sidewalks there stay unknown.
MONTGOMERY_CLASSES = {1206: "Sidewalk", 1208: "Non-Street Sidewalk"}


def montgomery_planimetrics(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Montgomery Planning Sidewalks & Driveways -> planimetrics. It has no stairs or
    crosswalk class, so those descriptions never occur and nothing is read from their
    absence (the region must not name it for planimetric coverage on that account)."""
    sw = gdf[gdf["FEATURE_CODE"].isin(MONTGOMERY_CLASSES)]
    year = sw["SOURCE_CODE"].map(MONTGOMERY_CAPTURE_YEAR)
    capture_ms = pd.to_datetime(year.astype("Int64").astype("string") + "-01-01", errors="coerce")
    return gpd.GeoDataFrame({
        "DESCRIPTION": sw["FEATURE_CODE"].map(MONTGOMERY_CLASSES).to_numpy(),
        "CAPTUREYEAR": (capture_ms - pd.Timestamp("1970-01-01")) // pd.Timedelta(milliseconds=1),
    }, geometry=sw.geometry.make_valid().values, crs=gdf.crs).reset_index(drop=True)


# Frederick's sidewalk lines carry a width in feet, often 0 (not recorded); drawn as surface
# of this width when not. 4 ft is the city's most common recorded width.
FREDERICK_DEFAULT_WIDTH_FT = 4.0
# The survey's Sidewalk Condition Index (0-100, higher is better) holds exactly 85.999 on
# 1,208 of 4,649 segments, photographed or not (2026-10-16): a fill value, not a score. The
# city has published no category cutoffs for the index, so only the number is carried.
FREDERICK_SCI_FILL = 85.999


def frederick_planimetrics(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """City of Frederick sidewalk lines -> sidewalk surface polygons, each dated by the first
    survey photo of its segment. A segment never photographed has no date and is left out.

    Optional columns read by ``enrich.enrich_sidewalk_condition``: CONDITION (the city's
    index, null for the fill value), WIDTH_M (null where the survey recorded no width) and
    LINE (the surveyed line itself, for matching walkways to it)."""
    photos = pd.read_parquet(raw_path("frederick_sidewalk_photos"))
    first = photos.set_index("Segment_ID")["first_ms"]
    sw = gdf[gdf["Segment_ID"].isin(first.index)]
    width_m = sw["Width_ft"].where(sw["Width_ft"] > 0, FREDERICK_DEFAULT_WIDTH_FT) * 0.3048
    sci = pd.to_numeric(sw["SCI"], errors="coerce")
    return gpd.GeoDataFrame({
        "DESCRIPTION": "Sidewalk",
        "CAPTUREYEAR": sw["Segment_ID"].map(first).to_numpy(),
        "CONDITION": sci.where(~sci.round(3).eq(FREDERICK_SCI_FILL)).to_numpy(),
        "WIDTH_M": (sw["Width_ft"].where(sw["Width_ft"] > 0) * 0.3048).round(2).to_numpy(),
        "LINE": sw.geometry.to_numpy(),
    }, geometry=sw.geometry.buffer(width_m / 2, cap_style="flat").values, crs=gdf.crs).reset_index(drop=True)


# The City of Frederick's verdicts (ADA_Description). Every ramp that passed has a tactile
# pad, so the two pad verdicts mean the pad is what is wrong: no obstacle to wheels, but a
# blind walker gets no warning of the street. "Need_Verified" says nothing yet.
FREDERICK_RAMP_CHECK = {
    "Passed": "passed",
    "Fail": "fails",
    "Combination_No_landing": "fails",
    "Damage_Need_Repair": "fails",
    "Tactile_Warning_Pad_No": "no_tactile_pad",
    "Tactile_Warning_Pad_Damage": "no_tactile_pad",
}


def frederick_ramps(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """City of Frederick curb ramp survey -> ramps. A "No ramp" record is a surveyed corner
    with no ramp (KERB no_ramp, no verdict on a ramp that is not there)."""
    no_ramp = gdf["Ramp_Type"].astype("string").str.strip().eq("No ramp")
    day = pd.to_datetime(gdf["Data_Date"].astype("string").str.strip(), format="%Y%m%d", errors="coerce")
    check = gdf["ADA_Description"].astype("string").str.strip().map(FREDERICK_RAMP_CHECK)
    return gpd.GeoDataFrame({
        "CAPTUREYEAR": ((day - pd.Timestamp("1970-01-01")) // pd.Timedelta(milliseconds=1)).astype("Int64"),
        "KERB": np.where(no_ramp, "no_ramp", "ramp_present"),
        "CHECK": check.where(~no_ramp),
    }, geometry=gdf.geometry.values, crs=gdf.crs).reset_index(drop=True)


def dc_ramps_2010(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """DC's 2010 planimetric ramps -> ramps. Every record is a ramp seen on imagery; the layer
    gives no verdict on it."""
    return gpd.GeoDataFrame({"CAPTUREYEAR": gdf["CAPTUREYEAR"], "KERB": "ramp_present", "CHECK": None},
                            geometry=gdf.geometry.values, crs=gdf.crs).reset_index(drop=True)


def montgomery_signals(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Montgomery County's signal layer -> signals. It records where signals are, nothing about
    their pedestrian features: LPI, APS, HAWK, all-pedestrian phase and recall stay null, and
    each attribute graded from them stays unknown. A proposed signal is not active."""
    proposed = gdf["OWNER_DESC"].astype("string").str.strip().eq("Proposed").fillna(False)
    none = pd.Series(None, index=gdf.index, dtype="object")
    return gpd.GeoDataFrame({
        "STATUS": np.where(proposed, "Proposed", "Active"),
        "LPI": none, "APS": none, "HAWK": none, "ALL_PED_PHASE": none, "PED_RECALL_MODE": none,
    }, geometry=gdf.geometry.values, crs=gdf.crs).reset_index(drop=True)


# A record of an undated signal inventory is used only with an OSM traffic_signals node this
# close: Frederick County's records sit a median 3 m from one, 90% within 6.4 m (2026-10-16).
SIGNAL_OSM_CONFIRM_M = 30.0
FREDERICK_SIGNAL_TYPES = {"FUL_SIG", "FULL_SIGNAL", "Operational Full Color Traffic Signal"}


def frederick_county_signals(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Frederick County's signal layer -> signals: full traffic signals only (not school
    beacons or flashers), each confirmed by an OSM signal node within SIGNAL_OSM_CONFIRM_M.
    The layer has no edit date, so OSM's confirmation is what dates it. No pedestrian
    features: LPI, APS and the rest stay null."""
    full = gdf["SIGNAL_TYPE"].astype("string").str.strip().isin(FREDERICK_SIGNAL_TYPES).fillna(False)
    sig = gdf[full.to_numpy()]
    nodes = gpd.read_parquet(INTERIM_DIR / "osm_nodes.parquet", columns=["highway", "geometry"]).to_crs(sig.crs)
    osm = nodes[nodes["highway"].eq("traffic_signals")]
    near = gpd.sjoin_nearest(sig[["geometry"]], osm[["geometry"]], how="inner", max_distance=SIGNAL_OSM_CONFIRM_M)
    sig = sig[sig.index.isin(near.index)]
    none = pd.Series(None, index=sig.index, dtype="object")
    return gpd.GeoDataFrame({
        "STATUS": "Active",
        "LPI": none, "APS": none, "HAWK": none, "ALL_PED_PHASE": none, "PED_RECALL_MODE": none,
    }, geometry=sig.geometry.values, crs=sig.crs).reset_index(drop=True)


# What a field sheet's ``result`` may say, per kind of place, and the value enrich carries.
# A corner is one curb: what is there. A junction is every corner a walker might use, so its
# answer is one of the junction values enrich already uses (``enrich_implicit_crossings``).
FIELD_RESULTS = {
    "corner": {"ramp": "ramp_present", "flush": "flush", "no_ramp": "no_ramp"},
    "junction": {"all_ramped": "ramp_likely", "missing_ramp": "no_ramp_reported"},
}
FIELD_NOT_ANSWERED = {"", "unclear"}  # not visited yet, or could not tell: no evidence either way


def field_checks(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """The team's field sheet -> field_checks. Rows with no result (not visited) or
    "unclear" carry nothing. Anything else must be a known result for its kind with a visit
    day; a typo is an error, never a silent skip, because a skipped row would leave a corner
    someone actually checked looking unrecorded."""
    res = gdf["result"].fillna("").astype(str).str.strip().str.lower()
    kind = gdf["kind"].fillna("").astype(str).str.strip().str.lower()
    keep = ~res.isin(FIELD_NOT_ANSWERED)
    kerb = pd.Series([FIELD_RESULTS.get(k, {}).get(r) for k, r in zip(kind, res)], index=gdf.index, dtype="object")
    bad = keep & kerb.isna()
    if bad.any():
        rows = gdf.loc[bad, ["place", "kind", "result"]].to_dict("records")
        raise ValueError(f"field sheet: unknown kind or result in {rows}; allowed: {FIELD_RESULTS} or blank/unclear")
    day = pd.to_datetime(gdf["visited_on"].fillna("").astype(str).str.strip(), format="%Y-%m-%d", errors="coerce")
    undated = keep & day.isna()
    if undated.any():
        raise ValueError(f"field sheet: a result needs visited_on as YYYY-MM-DD: {gdf.loc[undated, 'place'].tolist()}")
    g = gdf[keep]
    return gpd.GeoDataFrame({"KIND": kind[keep].to_numpy(), "KERB": kerb[keep].to_numpy(),
                             "VISITED": day[keep].dt.date.astype(str).to_numpy()},
                            geometry=g.geometry.values, crs=g.crs).reset_index(drop=True)


# source id -> function turning that source's raw frame into the role's canonical columns.
# DC's sources are the canonical schema and need none.
ADAPTERS: dict[str, Callable[[gpd.GeoDataFrame], gpd.GeoDataFrame]] = {
    "montgomery_sidewalks": montgomery_planimetrics,
    "frederick_sidewalks": frederick_planimetrics,
    "frederick_ramps": frederick_ramps,
    "montgomery_signals": montgomery_signals,
    "frederick_county_signals": frederick_county_signals,
    "frederick_field_checks": field_checks,
    "sidewalk_ramps_2010": dc_ramps_2010,
}

# OSM highway values that are streets for traffic (the roadway fallback). Service roads and
# alleys are left out, as they are from DC's roadway inventory.
OSM_STREET_HIGHWAYS = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary",
    "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential", "living_street",
}
OSM_ROADWAY = "osm_streets"  # marker for the fallback; its evidence is the OSM extract's


def empty(role: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({c: pd.Series(dtype="object") for c in CANONICAL[role]},
                            geometry=gpd.GeoSeries([], crs=ANALYSIS_CRS), crs=ANALYSIS_CRS)


def _mph(maxspeed: str | None) -> float:
    """OSM ``maxspeed`` as mph: "25 mph" -> 25, "40" (km/h by OSM convention) -> 24.9. Values
    such as "signals" or "none" say nothing usable and give NaN."""
    if not maxspeed:
        return np.nan
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(mph|km/h|kmh)?\s*$", str(maxspeed).split(";")[0])
    if not m:
        return np.nan
    v = float(m[1])
    return v if m[2] == "mph" else round(v / 1.609344, 1)


def osm_streets() -> gpd.GeoDataFrame:
    """Street centerlines from the region's OSM extract, in the roadway role's columns."""
    w = gpd.read_parquet(INTERIM_DIR / "osm_ways.parquet", columns=["highway", "name", "lanes", "maxspeed", "geometry"])
    w = w[w["highway"].isin(OSM_STREET_HIGHWAYS)].to_crs(ANALYSIS_CRS)
    lanes = pd.to_numeric(w["lanes"].astype("string").str.split(";").str[0], errors="coerce")
    out = gpd.GeoDataFrame({
        "TOTALTRAVELLANES": lanes.where(lanes > 0),
        "TOTALCROSSSECTIONWIDTH": np.nan,
        "SPEEDLIMITS_OB": w["maxspeed"].map(_mph),
        "AADT": np.nan,
        "AADT_YEAR": np.nan,
        "DCFUNCTIONALCLASS": w["highway"],
        "ROUTENAME": w["name"],
    }, geometry=w.geometry.values, crs=ANALYSIS_CRS).reset_index(drop=True)
    return out


def read(role: str) -> tuple[gpd.GeoDataFrame, str | None]:
    """The region's source for ``role`` in canonical columns (EPSG:26985), and its source id.
    No source: an empty frame and None (or, for the roadway, the OSM fallback and
    ``OSM_ROADWAY``)."""
    sid = REGION.role(role)
    if sid is None:
        if role == "roadway":
            return osm_streets(), OSM_ROADWAY
        return empty(role), None
    gdf = gpd.read_parquet(raw_path(sid)).to_crs(ANALYSIS_CRS)
    if sid in ADAPTERS:
        gdf = ADAPTERS[sid](gdf)
    missing = [c for c in CANONICAL.get(role, ()) if c not in gdf]
    if missing:
        raise ValueError(f"{sid} (role {role}) lacks canonical columns {missing}; register an adapter")
    return gdf, sid
