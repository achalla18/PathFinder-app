"""Maryland pilot regions in the 6th congressional district (plan of 2026-10-15).

Each pilot starts from OpenStreetMap and the city limits alone (Phase B: does OSM map enough
sidewalks and crossings here to be worth more?). Every county or state inventory is added
later, one source at a time, after its records have been checked against imagery; until
then no role is filled, so every attribute it would grade stays ``unknown`` and no absence
counts as evidence.
"""

from __future__ import annotations

from datetime import date

from .base import AS_OF_OSM, Region, Source

GEOFABRIK_MD = "https://download.geofabrik.de/north-america/us/maryland-latest.osm.pbf"
IMAP_MUNICIPAL = "https://mdgeodata.md.gov/imap/rest/services/Boundaries/MD_PoliticalBoundaries/FeatureServer/5"
VERIFIED = date(2026, 10, 15)
# The cut taken from the statewide OSM extract: the city's own extent plus about 1 km, so
# that ways crossing the line are complete and the scope clip (``buffer_m``) has room.
CLIP_MARGIN_DEG = 0.01


def _box(extent: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = extent
    m = CLIP_MARGIN_DEG
    return (round(xmin - m, 4), round(ymin - m, 4), round(xmax + m, 4), round(ymax + m, 4))


def _osm(region_id: str, name: str, extent: tuple[float, float, float, float]) -> Source:
    return Source(
        id=f"osm_{region_id}",
        kind="pbf_clip",
        url=GEOFABRIK_MD,
        bbox=_box(extent),
        title=f"OpenStreetMap, {name} (cut from Geofabrik's Maryland extract)",
        license="ODbL 1.0 — attribution and share-alike for derived databases",
        limitations=(
            "Volunteer data. Suburban sidewalk mapping is uneven (Nichols, Groth & Quinn, Prev. "
            "Chronic Dis. 2026: thinner in low-density tracts). Absence of a tag is unknown, not 'no'."
        ),
        verified_on=VERIFIED,
        tags=("network",),
    )


def _limits(region_id: str, name: str, where: str) -> Source:
    return Source(
        id=f"{region_id}_boundary",
        kind="arcgis_layer",
        url=IMAP_MUNICIPAL,
        where=where,
        title=f"{name} city limits (MD iMAP Municipal Boundaries - Detailed)",
        license="No license stated on the service; MD iMAP public data (MD iMAP, DoIT, MDP)",
        limitations="One polygon per annexation; the pipeline unions them. Administrative line only.",
        verified_on=VERIFIED,
        tags=("boundary",),
    )


TIGER_TRACTS_2020 = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/10"


def _tracts(region_id: str, extent: tuple[float, float, float, float]) -> Source:
    return Source(
        id=f"{region_id}_tracts",
        kind="arcgis_layer",
        url=TIGER_TRACTS_2020,
        bbox=_box(extent),
        title="2020 Census tracts (Census Bureau TIGERweb)",
        license="Public domain (U.S. Census Bureau)",
        limitations="Statistical areas, not neighbourhoods; a tract can straddle the city line.",
        verified_on=VERIFIED,
        tags=("report_districts",),
    )


# One dated world build; builds are kept for a limited time, so move this to a current date
# (https://build-metadata.protomaps.dev/builds.json) when re-fetching.
PROTOMAPS_BUILD = "https://build.protomaps.com/20261021.pmtiles"


def _basemap(region_id: str, name: str) -> Source:
    return Source(
        id=f"basemap_{region_id}",
        kind="pmtiles_extract",
        url=PROTOMAPS_BUILD,
        title=f"Protomaps basemap vector tiles, {name} extract (OpenStreetMap data)",
        license="ODbL 1.0 (OpenStreetMap data) — attribution required; tileset schema by Protomaps (BSD-3)",
        limitations="Backdrop only: nothing is routed or graded from it. Tiles over the region's box to zoom 15.",
        verified_on=date(2026, 10, 21),
        tags=("basemap",),
    )


BASEMAP_FONTS = Source(
    id="basemap_fonts",
    kind="glyphs",
    url="https://protomaps.github.io/basemaps-assets/fonts",
    title="Noto Sans map label glyphs (Protomaps basemaps-assets)",
    license="SIL Open Font License 1.1",
    limitations="Latin ranges only; labels in other scripts are not drawn.",
    verified_on=date(2026, 10, 21),
    tags=("basemap",),
)


def _region(region_id: str, name: str, muni: str, extent: tuple[float, float, float, float],
            center: tuple[float, float], extra: tuple[Source, ...] = (),
            roles: dict[str, str] | None = None) -> Region:
    """A pilot region. ``extra`` sources are registered (fetched, in the manifest) but play
    no role until their records have been inspected and the region names them in ``roles``."""
    osm = _osm(region_id, name, extent)
    limits = _limits(region_id, name, f"MUN_NAME='{muni}'")
    tracts = _tracts(region_id, extent)
    basemap = _basemap(region_id, name)
    return Region(
        id=region_id,
        name=name,
        sources={s.id: s for s in (osm, limits, tracts, basemap, BASEMAP_FONTS, *extra)},
        osm=osm.id,
        boundary=limits.id,
        buffer_m=200.0,
        center=center,
        bbox=osm.bbox,
        # reports break down by census tract: where the evidence is thin, and for whom
        roles={"report_districts": tracts.id, **(roles or {})},
        report_district_field="GEOID",
        report_district_label="census tract",
        coverage={"planimetric": None, "signals": None},
        basemap=basemap.id,
    )


# Planimetric sidewalk surveys (Phase C, step 4). Neither service states a license: internal
# use only until written permission (product rule 7).
MONTGOMERY_SIDEWALKS = "https://montgomeryplans.org/server/rest/services/Overlays/Sidewalks___Driveways/FeatureServer/3"
GAITHERSBURG_SIDEWALKS = "https://maps.gaithersburgmd.gov/arcgis/rest/services/layers/Sidewalks_Public/FeatureServer/0"


def _montgomery_sidewalks(extent: tuple[float, float, float, float]) -> Source:
    return Source(
        id="montgomery_sidewalks",
        kind="arcgis_layer",
        url=MONTGOMERY_SIDEWALKS,
        where="FEATURE_CODE IN (1206, 1208)",
        bbox=_box(extent),
        title="Montgomery Planning planimetrics: street and non-street sidewalks (Sidewalks & Driveways)",
        license="No license stated on the service (M-NCPPC Montgomery Planning). Internal use only",
        limitations=(
            "Aerial capture, dated only by year (SOURCE_CODE: 2014, 2017, 2020, 2023 captures; in "
            "the Gaithersburg box most polygons are from 2017). Classes are street sidewalk (1206) "
            "and non-street sidewalk (1208); there is no stairs or crosswalk class, so the layer "
            "says nothing about either. Tree canopy can hide a sidewalk. Editor fields are dropped on "
            "ingest."
        ),
        verified_on=VERIFIED,
        drop_fields=("CREATED_USER", "LAST_EDITED_USER"),
        tags=("planimetrics",),
    )


def _gaithersburg_sidewalks() -> Source:
    return Source(
        id="gaithersburg_sidewalks",
        kind="arcgis_layer",
        url=GAITHERSBURG_SIDEWALKS,
        title="City of Gaithersburg Public Sidewalks (planimetric, from aerial photography)",
        license="No license stated; copyright City of Gaithersburg. Internal use only",
        limitations=(
            "The city's public sidewalks only (private walks are not in it). Paved/unpaved code "
            "and update method per polygon. Staff and editor fields are dropped on ingest."
        ),
        verified_on=VERIFIED,
        drop_fields=("UPDATE_STAFF_ID", "created_user", "last_edited_user"),
        tags=("planimetrics",),
    )


FREDERICK_SIDEWALKS = "https://spires.cityoffrederick.com/ArcGIS/rest/services/Sidewalks/FeatureServer/2"
FREDERICK_SIDEWALK_PHOTOS = "https://spires.cityoffrederick.com/ArcGIS/rest/services/Sidewalks/FeatureServer/1"


def _frederick_sidewalks() -> Source:
    return Source(
        id="frederick_sidewalks",
        kind="arcgis_layer",
        url=FREDERICK_SIDEWALKS,
        title="City of Frederick Sidewalks (condition survey: segment lines with SCI, width, surface)",
        license="No license stated (City of Frederick). Internal use only until written permission",
        limitations=(
            "Lines along the sidewalk, not surface outlines; the city's sidewalks only. The records "
            "carry no date: a segment is dated by its first survey photo (frederick_sidewalk_photos, "
            "Mar-May 2025), and the 815 segments with no photo (82 km) are not used. No stairs or "
            "crosswalks. Editor fields are dropped on ingest."
        ),
        verified_on=VERIFIED,
        drop_fields=("created_user", "last_edited_user"),
        tags=("planimetrics",),
    )


FREDERICK_RAMPS = "https://spires.cityoffrederick.com/ArcGIS/rest/services/Sidewalks/FeatureServer/0"


def _frederick_ramps() -> Source:
    return Source(
        id="frederick_ramps",
        kind="arcgis_layer",
        url=FREDERICK_RAMPS,
        title="City of Frederick curb ramp survey (2025: type, running and landing slope checks, tactile pad)",
        license="No license stated (City of Frederick). Internal use only until written permission",
        limitations=(
            "6,366 corners surveyed 2025-03-18 to 2025-06-10, each dated (Data_Date). 'No ramp' "
            "records a corner without one. ADA_Description is the survey's overall verdict. The "
            "city's corners only; state roads are MDOT SHA's. Editor fields are dropped on ingest."
        ),
        verified_on=VERIFIED,
        drop_fields=("created_user", "last_edited_user"),
        tags=("ramps",),
    )


def _frederick_sidewalk_photos() -> Source:
    return Source(
        id="frederick_sidewalk_photos",
        kind="arcgis_first_date",
        url=FREDERICK_SIDEWALK_PHOTOS,
        group_field="Segment_ID",
        date_field="Date_Time_Operation",
        title="City of Frederick sidewalk survey photos: first photo date per sidewalk segment",
        license="No license stated (City of Frederick). Internal use only until written permission",
        limitations="311,547 photos taken 2025-03-25 to 2025-05-23; only each segment's earliest date is kept.",
        verified_on=VERIFIED,
        tags=("planimetrics",),
    )


def _field_checks(region_id: str) -> Source:
    return Source(
        id=f"{region_id}_field_checks",
        kind="local_csv",
        url=f"field/{region_id}.csv",
        title="PathFinder field checks: curbs at the corners and junctions that block trips, seen on site",
        license="The PathFinder team's own observations; no restrictions",
        limitations=(
            "Only the places on the sheet, as seen on the day of the visit (visited_on). Unvisited "
            "rows and \"unclear\" carry nothing. A corner answer is what one curb has; a junction "
            "answer covers every corner with a sidewalk and stays inferred, because which corners "
            "a walker uses is not known. See field/README.md."
        ),
        verified_on=date(2026, 10, 20),
        tags=("field_checks",),
    )


SDAT_PARCEL_POINTS = "https://mdgeodata.md.gov/imap/rest/services/PlanningCadastre/MD_PropertyData/MapServer/0"


def _parcels(region_id: str, extent: tuple[float, float, float, float]) -> Source:
    return Source(
        id=f"{region_id}_parcels",
        kind="arcgis_layer",
        url=SDAT_PARCEL_POINTS,
        bbox=_box(extent),
        out_fields=("YEARBLT", "LU", "SDATDATE"),
        title="Maryland tax parcel points (SDAT, via MD iMAP): year built and land use",
        license="No license stated on the service; MD iMAP public data (MD iMAP, MDP, SDAT)",
        limitations=(
            "One point per tax account, near the parcel's centre. YEARBLT is the main structure's "
            "year built as assessed (0 or blank: no structure); it dates the buildings, not the "
            "street. Owner names and addresses are never fetched (out_fields)."
        ),
        verified_on=date(2026, 10, 21),
        tags=("parcels",),
    )


MONTGOMERY_SIGNALS = "https://services.arcgis.com/hdlG36cOubTzNDBb/arcgis/rest/services/Traffic_Signals_Bill_1123/FeatureServer/0"


def _montgomery_signals(extent: tuple[float, float, float, float]) -> Source:
    return Source(
        id="montgomery_signals",
        kind="arcgis_layer",
        url=MONTGOMERY_SIGNALS,
        bbox=_box(extent),
        # the layer's records carry no dates; its data were last edited on this day
        as_of="2023-11-09",
        title="Montgomery County traffic signals (MCDOT, Bill 11-23 review layer)",
        license="No license stated on the layer (Montgomery County DOT, ArcGIS Online). Internal use only",
        limitations=(
            "Signal locations only, county-wide and including SHA- and city-owned signals. Its LPI "
            "and no-turn-on-red fields are empty and it has no accessible-signal field. Not complete: "
            "of 84 signalized intersections OSM maps inside Gaithersburg, 17 have no record here, at "
            "least 5 of them real intersections with crosswalks (imagery, 2026-10-16); the rest are "
            "I-270 ramp signals. Data last edited 2023-11-09, so newer signals are missing."
        ),
        verified_on=date(2026, 10, 16),
        tags=("signals",),
    )


FREDERICK_COUNTY_SIGNALS = "https://fcgis.frederickcountymd.gov/server_pub/rest/services/DPW/Traffic_Signals/MapServer/0"


def _frederick_county_signals(extent: tuple[float, float, float, float]) -> Source:
    return Source(
        id="frederick_county_signals",
        kind="arcgis_layer",
        url=FREDERICK_COUNTY_SIGNALS,
        bbox=_box(extent),
        # the layer gives no edit date (activation dates 2006-2011 on a few records); a record
        # is used only where an OSM signal confirms it, and is dated by the OSM extract
        as_of=AS_OF_OSM,
        title="Frederick County DPW traffic signals (city-, county- and SHA-maintained)",
        license="No license stated on the service (Frederick County DPW). Internal use only",
        limitations=(
            "Signal locations only: its pedestrian-crossing and signal-head counts are empty on "
            "most records, and it has no accessible-signal or LPI field. Mostly the City of "
            "Frederick's signals (71 of 75 inside the line); state-road signals are largely missing: "
            "of 116 signalized intersections OSM maps inside the line, 45 have no record "
            "(2026-10-16). No edit date: records without an OSM signal within "
            "roles.SIGNAL_OSM_CONFIRM_M are not used."
        ),
        verified_on=date(2026, 10, 16),
        tags=("signals",),
    )


IMAP_LIDAR = "https://mdgeodata.md.gov/lidar/rest/services"


def _dtm(region_id: str, county: str, extent: tuple[float, float, float, float], as_of: str,
         title: str, license: str, limitations: str) -> Source:
    return Source(
        id=f"{region_id}_dtm",
        kind="image_service",
        url=f"{IMAP_LIDAR}/{county.title()}/MD_{county}_dem_m/ImageServer",
        bbox=_box(extent),
        as_of=as_of,
        title=title,
        license=license,
        limitations=limitations,
        verified_on=VERIFIED,
        tags=("dtm",),
    )


_GAITHERSBURG_EXTENT = (-77.2583, 39.1027, -77.1712, 39.1684)
_FREDERICK_EXTENT = (-77.4782, 39.3924, -77.3452, 39.4886)
# Extents: MD iMAP Municipal Boundaries - Detailed, returnExtentOnly in EPSG:4326 (2026-10-15).
GAITHERSBURG = _region("gaithersburg", "Gaithersburg, MD", "Gaithersburg",
                       _GAITHERSBURG_EXTENT, center=(-77.2014, 39.1434),
                       extra=(_montgomery_sidewalks(_GAITHERSBURG_EXTENT), _gaithersburg_sidewalks(),
                              _montgomery_signals(_GAITHERSBURG_EXTENT),
                              _dtm("gaithersburg", "montgomery", _GAITHERSBURG_EXTENT, "2020",
                                   "Montgomery County bare-earth DEM, metres (lidar flown late 2020 to early 2021; MD iMAP)",
                                   "MD iMAP public data; no license stated on the service",
                                   "Dated by the start of the flight season (late 2020): the service gives no day. "
                                   "Native 1 ft grid in EPSG:2893, exported by the server on the 1 m analysis grid. "
                                   "Bare earth: says nothing about bridges, decks or indoor ways.")),
                       # Montgomery's polygons checked against MD iMAP six-inch imagery and against
                       # the city's own layer (98.5% of its street-side area within 1.5 m) on
                       # 2026-10-15. Planimetric coverage stays None: the layer has no stairs class,
                       # so "no stairs on record" would prove nothing.
                       # Signals checked against OSM and imagery on 2026-10-16: 81 of 84 records
                       # inside the line sit at an OSM-mapped signal. Signal coverage stays None:
                       # the layer misses real signals, so "no signal on record" proves nothing.
                       roles={"planimetrics": "montgomery_sidewalks", "dtm": "gaithersburg_dtm",
                              "signals": "montgomery_signals"})
FREDERICK = _region("frederick", "Frederick, MD", "Frederick",
                    _FREDERICK_EXTENT, center=(-77.4105, 39.4143),
                    extra=(_frederick_sidewalks(), _frederick_sidewalk_photos(), _frederick_ramps(),
                           _frederick_county_signals(_FREDERICK_EXTENT), _field_checks("frederick"),
                           _parcels("frederick", _FREDERICK_EXTENT),
                           _dtm("frederick", "frederick", _FREDERICK_EXTENT, "2021-12-14",
                                "Frederick County bare-earth DEM, metres (USGS MD Western 2021 lidar; MD iMAP)",
                                "Public domain (USGS 3DEP, contract G17PC00015), served by MD iMAP",
                                "Flown 2021-12-14 to 2021-12-21 (work unit MD_Western_2, NOAA InPort item 70420); "
                                "19.6 cm vertical accuracy (NVA). Exported by the server on the 1 m analysis grid. "
                                "Bare earth: says nothing about bridges, decks or indoor ways.")),
                    # The city's 2025 surveys, checked on imagery and for placement (sidewalks
                    # 2026-10-15; ramps 2026-10-16: 12 of 12 sampled records at real curb
                    # transitions, 2,381 of 3,606 mapped crossing ends with a record within 8 m).
                    # Coverage stays None: no stairs class, and the city's sidewalks only.
                    # Signals: 73 of 75 records inside the line sit within 50 m of an OSM signal
                    # (median 3 m), 2026-10-16. Coverage None: 45 of OSM's 116 signalized
                    # intersections have no record.
                    # Field checks: the team's own visits to places that block trips (field/README.md)
                    # Parcels: the post-ADA ramp rule passed its bar against the city's ramp survey on
                    # 2026-10-21 (reports/frederick/post_ada/evaluation.md). Not for Gaithersburg: it has
                    # no ramp survey to test the rule against.
                    roles={"planimetrics": "frederick_sidewalks", "dtm": "frederick_dtm", "ramps": "frederick_ramps",
                           "signals": "frederick_county_signals", "field_checks": "frederick_field_checks",
                           "parcels": "frederick_parcels"})
