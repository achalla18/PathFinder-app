"""What a region is: its sources, its boundary, and which inventories cover which places.

The pipeline was written for DC. Everything that differs between places lives here, so the
same graph build, enrichment, routing and reports run on another region without code that
names it. A region states three things the code cannot guess:

- **Sources** it has, and for each *role* the pipeline knows (traffic signals, planimetric
  surfaces, roadway inventory, ...) which source plays it. A role with no source is simply
  absent: every attribute it would have graded stays ``unknown``. Missing data is never
  favorable.
- **Inventory coverage**: where the absence of a record counts as evidence ("no stairs on
  record" => step-free). Only a comprehensive inventory can say that, and only inside the
  area it surveyed. A region that does not name an inventory for a family gets no
  confirmed absence anywhere.
- **Presentation**: name, map centre, how street names are written, which transit notices
  exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

# Roles the pipeline reads. Each source registered for a role is turned into that role's
# canonical columns by an adapter (``graph.roles``); the canonical columns are DC's field
# names because DC was the first region and its code reads them.
ROLES = (
    "signals",           # traffic signal points: STATUS, LPI, APS, HAWK, ALL_PED_PHASE, PED_RECALL_MODE
    "planimetrics",      # surface polygons: DESCRIPTION (Sidewalk/Hidden Sidewalk/Crosswalk/Stairs), CAPTUREYEAR
    "roadway",           # street centerlines with inventory attributes (lanes, width, speed, AADT, name)
    "ramps",             # curb ramp points: CAPTUREYEAR
    "project_sidewalk",  # crowdsourced labels
    "dtm",               # bare-earth elevation tiles
    "report_districts",  # reporting units for the coverage report (DC: wards)
    "field_checks",      # the team's own on-site observations of curbs: KIND, KERB, VISITED
    "parcels",           # tax parcel points with the year built: YEARBLT, SDATDATE
)

# Families of "absence is evidence". Each is "boundary" (the inventory covers public streets
# inside the region's boundary), "everywhere" (every edge in the graph, buffer included) or
# None (no confirmed absence anywhere).
COVERAGE_FAMILIES = ("planimetric", "signals")
COVERAGE_SCOPES = (None, "boundary", "everywhere")
AS_OF_OSM = "osm_extract"


@dataclass(frozen=True)
class Source:
    id: str
    # "arcgis_layer" | "http_file" | "project_sidewalk" | "image_service" | "pmtiles_extract" | "glyphs"
    # | "wmata_json" (fetched with WMATA_API_KEY) | "live" (read at request time, never stored)
    # | "pbf_clip" (a bounding-box cut of a larger OSM extract; see fetch.fetch_pbf_clip)
    # | "arcgis_first_date" (per ``group_field`` value, the earliest ``date_field``: dates a
    #   layer's records from a related layer too large to fetch, e.g. survey photos)
    # | "local_csv" (a sheet kept in the repo, ``url`` relative to the repo root; refetched
    #   whenever its checksum changes, see fetch.fetch_local_csv)
    kind: str
    url: str
    title: str
    license: str
    limitations: str
    where: str = "1=1"
    verified_on: date = date(2026, 10, 4)
    tags: tuple[str, ...] = field(default_factory=tuple)
    # lon/lat box (xmin, ymin, xmax, ymax): the cut taken out of ``url`` (pbf_clip), or the
    # features fetched (arcgis_layer: those intersecting it; a county layer for one town)
    bbox: tuple[float, float, float, float] | None = None
    # arcgis_layer: fields dropped on ingest. Privacy: editor and staff identifiers are not
    # needed for routing and are never stored.
    drop_fields: tuple[str, ...] = field(default_factory=tuple)
    # arcgis_layer: when set, the only fields requested. Privacy: a layer with personal fields
    # (owner names and addresses) is fetched with the fields the pipeline needs and no others.
    out_fields: tuple[str, ...] = field(default_factory=tuple)
    # arcgis_first_date only
    group_field: str | None = None
    date_field: str | None = None
    # The date the data describes, where the records carry none of their own (a LiDAR
    # flight); carried as the as_of of everything graded from it. AS_OF_OSM: the source is used
    # only where the region's OSM extract confirms it, and is dated by that extract.
    as_of: str | None = None


@dataclass(frozen=True)
class Region:
    id: str
    name: str
    sources: dict[str, Source]
    osm: str                 # source id of the region's OSM extract
    boundary: str            # source id of the polygon that defines the region
    # Walkways this far outside the boundary stay routable (a boundary street's far
    # sidewalk); no inventory covers them, so they are flagged ``outside_region``.
    buffer_m: float
    center: tuple[float, float]               # lon, lat: where the map opens
    bbox: tuple[float, float, float, float]   # lon/lat box with margin: map tiles, tiled APIs
    roles: dict[str, str] = field(default_factory=dict)
    coverage: dict[str, str | None] = field(default_factory=dict)
    report_district_field: str | None = None  # attribute of the report_districts source
    report_district_label: str = "district"
    street_quadrants: bool = False            # DC names carry NW/NE/SE/SW
    transit_notices: str | None = None        # "wmata" or None
    basemap: str | None = None                # source id of the local map tiles

    def role(self, name: str) -> str | None:
        if name not in ROLES:
            raise KeyError(f"unknown role {name!r}")
        return self.roles.get(name)

    def covers(self, family: str) -> str | None:
        if family not in COVERAGE_FAMILIES:
            raise KeyError(f"unknown coverage family {family!r}")
        return self.coverage.get(family)

    def __post_init__(self) -> None:
        for sid in (self.osm, self.boundary, *self.roles.values(), *( [self.basemap] if self.basemap else [])):
            if sid not in self.sources:
                raise ValueError(f"region {self.id}: source {sid!r} is not registered")
        for r in self.roles:
            self.role(r)
        for f, v in self.coverage.items():
            self.covers(f)
            if v not in COVERAGE_SCOPES:
                raise ValueError(f"region {self.id}: coverage {f}={v!r} must be one of {COVERAGE_SCOPES}")
