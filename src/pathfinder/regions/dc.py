"""Washington, DC: the city PathFinder's pipeline was first written for (as DistrictDirect).

DC publishes its accessibility inventories openly (Open Data DC, CC BY 4.0): planimetric
sidewalks, a 2010 curb-ramp survey, DDOT's roadway inventory and signals, plus Project
Sidewalk's crowdsourced labels and a 2024 LiDAR terrain model. The sources are those DistrictDirect
registered and checked (its ``sources.py``, verified 2026-10-04); its crime, crash, street-light and
high-injury layers are left out because no PathFinder role reads them.

Their field names are the pipeline's canonical schema, so every role reads as is except the
ramp survey, which carries presence only (``roles.dc_ramps_2010``).

Inventory coverage stays None: the sidewalk layer has no stairs class, so "no stairs on record"
would prove nothing, and a sidewalk missing from it is not confirmed absent.
"""

from __future__ import annotations

from datetime import date

from .base import Region, Source

DCGIS = "https://maps2.dcgis.dc.gov/dcgis/rest/services"
DCGIS_IMAGERY = "https://imagery.dcgis.dc.gov/dcgis/rest/services"
OPEN_DATA_DC = "CC BY 4.0 (Open Data DC)"
VERIFIED = date(2026, 10, 4)  # by DistrictDirect


def _layer(service: str, layer: int) -> str:
    return f"{DCGIS}/{service}/MapServer/{layer}"


_SOURCES = (
    Source(
        id="osm_dc",
        kind="http_file",
        url="https://download.geofabrik.de/north-america/us/district-of-columbia-latest.osm.pbf",
        title="OpenStreetMap extract for DC (Geofabrik)",
        license="ODbL 1.0 — attribution and share-alike for derived databases",
        limitations=("Volunteer data; separate sidewalk mapping is incomplete in places (2023 Meta "
                     "tasking project targeted DC gaps). Absence of a tag is unknown, not 'no'."),
        verified_on=VERIFIED,
        tags=("network",),
    ),
    Source(
        id="dc_boundary",
        kind="arcgis_layer",
        url=_layer("DCGIS_DATA/Administrative_Other_Boundaries_WebMercator", 10),
        title="DC Boundary",
        license=OPEN_DATA_DC,
        limitations="Administrative boundary.",
        verified_on=VERIFIED,
        tags=("boundary",),
    ),
    Source(
        id="wards_2022",
        kind="arcgis_layer",
        url=_layer("DCGIS_DATA/Administrative_Other_Boundaries_WebMercator", 53),
        title="Wards 2022",
        license=OPEN_DATA_DC,
        limitations="Reporting unit only.",
        verified_on=VERIFIED,
        tags=("boundary",),
    ),
    Source(
        id="roadway_block",
        kind="arcgis_layer",
        url=_layer("DCGIS_DATA/Transportation_WebMercator", 163),
        title="DDOT Roadway Block",
        license=OPEN_DATA_DC,
        limitations=("Inventory attributes (lanes, widths, speed limits, AADT, sidewalk widths) may be "
                     "missing or dated per block; AADT has its own year field."),
        verified_on=VERIFIED,
        tags=("crossing", "traffic"),
    ),
    Source(
        id="traffic_signals",
        kind="arcgis_layer",
        url=_layer("DCGIS_DATA/Transportation_Signs_Signals_Lights_WebMercator", 170),
        title="DDOT Traffic Signals",
        license=OPEN_DATA_DC,
        limitations=("Intersection-level points; does not say which legs have pedestrian heads. "
                     "Feature flags (LPI, APS, HAWK) reflect inventory, not real-time operation."),
        verified_on=VERIFIED,
        tags=("crossing", "accessibility"),
    ),
    Source(
        id="sidewalks_2021",
        kind="arcgis_layer",
        url=_layer("DCGIS_DATA/Planimetrics_2021", 11),
        title="Planimetric Sidewalks 2021",
        license=OPEN_DATA_DC,
        limitations="Photogrammetric footprint (2021); shape only, not surface condition or obstructions.",
        verified_on=VERIFIED,
        tags=("accessibility", "network", "planimetrics"),
    ),
    Source(
        id="sidewalk_ramps_2010",
        kind="arcgis_layer",
        url=_layer("DCGIS_DATA/Planimetrics_2010", 2),
        title="Planimetric Sidewalk Ramps 2010",
        license=OPEN_DATA_DC,
        limitations=("Captured 2010 — stale. Presence evidence only, carried with its date; says nothing "
                     "about ramp compliance."),
        verified_on=VERIFIED,
        tags=("accessibility",),
    ),
    Source(
        id="project_sidewalk_dc",
        kind="project_sidewalk",
        url="https://sidewalk-dc.cs.washington.edu/v3/api/labelClusters",
        title="Project Sidewalk DC label clusters",
        license="CC0 1.0 (cite Saha et al., CHI 2019 in research)",
        limitations=("Crowdsourced from street-level imagery captured mostly 2014–2016; age is carried "
                     "as as_of. Contributor IDs are dropped on ingest."),
        verified_on=VERIFIED,
        tags=("accessibility",),
    ),
    Source(
        id="dtm_2024",
        kind="image_service",
        url=f"{DCGIS_IMAGERY}/Lidar/Hydro_Enforced_DTM_2024/ImageServer",
        title="2024 LiDAR hydro-enforced DTM (1 m)",
        license=OPEN_DATA_DC,
        limitations=("Bare-earth: bridge decks are not represented (use endpoint interpolation), and "
                     "sidewalks on untagged decks read the ground below. Areas beyond the District line "
                     "are filled with exactly 0.0."),
        verified_on=VERIFIED,
        as_of="2024",
        tags=("elevation", "dtm"),
    ),
    Source(
        id="basemap_dc",
        kind="pmtiles_extract",
        # One dated world build; builds are kept for a limited time, so move this to a current
        # date (https://build-metadata.protomaps.dev/builds.json) when re-fetching.
        url="https://build.protomaps.com/20261006.pmtiles",
        title="Protomaps basemap vector tiles, DC extract (OpenStreetMap data)",
        license="ODbL 1.0 (OpenStreetMap data) — attribution required; tileset schema by Protomaps (BSD-3)",
        limitations="Backdrop only: nothing is routed or graded from it. Tiles over the DC box to zoom 15.",
        verified_on=date(2026, 10, 6),
        tags=("basemap",),
    ),
    Source(
        id="basemap_fonts",
        kind="glyphs",
        url="https://protomaps.github.io/basemaps-assets/fonts",
        title="Noto Sans map label glyphs (Protomaps basemaps-assets)",
        license="SIL Open Font License 1.1",
        limitations="Latin ranges only; labels in other scripts are not drawn.",
        verified_on=date(2026, 10, 6),
        tags=("basemap",),
    ),
)

DC = Region(
    id="dc",
    name="Washington, DC",
    sources={s.id: s for s in _SOURCES},
    osm="osm_dc",
    boundary="dc_boundary",
    buffer_m=200.0,
    center=(-77.0365, 38.8977),
    bbox=(-77.1300, 38.7800, -76.8990, 39.0060),
    roles={"planimetrics": "sidewalks_2021", "roadway": "roadway_block", "signals": "traffic_signals",
           "ramps": "sidewalk_ramps_2010", "project_sidewalk": "project_sidewalk_dc", "dtm": "dtm_2024",
           "report_districts": "wards_2022"},
    coverage={"planimetric": None, "signals": None},
    report_district_field="WARD",
    report_district_label="ward",
    street_quadrants=True,
    basemap="basemap_dc",
)
