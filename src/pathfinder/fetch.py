"""Download every registered source into RAW_DIR with a provenance manifest.

Design rules:
- Fetches are idempotent: an existing artifact is skipped unless ``force=True``.
- Every artifact gets a manifest entry (source id, URL, query, fetch time, record count,
  sha256) so any derived result can be traced back to exactly what was downloaded.
- ArcGIS layers are fetched by object-id chunks, then the record count is checked against
  the server's id list; a mismatch is an error, never a silent partial download.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import shape

from .config import RAW_DIR, REPO_ROOT, SHARED_DIR, WGS84, ensure_dirs
from .sources import REGION, SOURCES, Source

log = logging.getLogger(__name__)

MANIFEST = RAW_DIR / "manifest.json"
USER_AGENT = "PathFinder/0.1 (pedestrian navigation research; contact via repo owner)"
_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT


# --------------------------------------------------------------------------- utilities


def _request(method: str, url: str, *, retries: int = 5, timeout: int = 180, **kw) -> requests.Response:
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            r = _session.request(method, url, timeout=timeout, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == retries:
                raise
            log.warning("request failed (%s), retry %d/%d in %.0fs", e, attempt, retries, delay)
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def _record(source: Source, path: Path, **extra) -> None:
    m = load_manifest()
    m[source.id] = {
        "title": source.title,
        "url": source.url,
        "where": source.where,
        "license": source.license,
        "path": str(path.relative_to(RAW_DIR)),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": _sha256(path) if path.is_file() else None,
        **extra,
    }
    MANIFEST.write_text(json.dumps(m, indent=2, sort_keys=True))


def raw_path(source_id: str) -> Path:
    s = SOURCES[source_id]
    if s.kind == "http_file":
        return RAW_DIR / Path(s.url).name
    if s.kind in ("image_service", "glyphs"):
        return RAW_DIR / source_id  # directory of tiles / glyph ranges
    if s.kind == "pmtiles_extract":
        return RAW_DIR / f"{source_id}.pmtiles"
    if s.kind == "pbf_clip":
        return RAW_DIR / f"{source_id}.osm.pbf"
    return RAW_DIR / f"{source_id}.parquet"


# --------------------------------------------------------------------------- ArcGIS


def _arcgis_bbox(source: Source) -> dict:
    if not source.bbox:
        return {}
    return {"geometry": ",".join(map(str, source.bbox)), "geometryType": "esriGeometryEnvelope",
            "inSR": "4326", "spatialRel": "esriSpatialRelIntersects"}


def _arcgis_ids(source: Source) -> list[int]:
    r = _request(
        "GET",
        f"{source.url}/query",
        params={"where": source.where, "returnIdsOnly": "true", "f": "json", **_arcgis_bbox(source)},
    ).json()
    if "error" in r:
        raise RuntimeError(f"{source.id}: {r['error']}")
    return sorted(r.get("objectIds") or [])


def _arcgis_chunk(source: Source, ids: list[int]) -> list[dict]:
    r = _request(
        "POST",
        f"{source.url}/query",
        data={
            "objectIds": ",".join(map(str, ids)),
            "outFields": ",".join(source.out_fields) or "*",
            "outSR": "4326",
            "returnGeometry": "true",
            "f": "geojson",
        },
    ).json()
    if "error" in r:
        raise RuntimeError(f"{source.id}: {r['error']}")
    return r["features"]


def fetch_arcgis(source: Source, *, chunk: int = 500) -> Path:
    out = raw_path(source.id)
    ids = _arcgis_ids(source)
    log.info("%s: %d records", source.id, len(ids))
    feats: list[dict] = []
    for i in range(0, len(ids), chunk):
        feats.extend(_arcgis_chunk(source, ids[i : i + chunk]))
    if len(feats) != len(ids):
        raise RuntimeError(f"{source.id}: expected {len(ids)} features, got {len(feats)}")
    geoms = [shape(f["geometry"]) if f.get("geometry") else None for f in feats]
    props = pd.DataFrame([f["properties"] for f in feats])
    props = props.drop(columns=[c for c in source.drop_fields if c in props])
    gdf = gpd.GeoDataFrame(props, geometry=geoms, crs=WGS84)
    gdf.to_parquet(out)
    extra = {"bbox": list(source.bbox)} if source.bbox else {}
    _record(source, out, records=len(gdf), null_geometries=int(gdf.geometry.isna().sum()), **extra)
    return out


# Ids per grouped-statistics request; halved while a response comes back at the server's
# record limit, so no group is silently cut off.
FIRST_DATE_ID_SPAN = 1500


def fetch_arcgis_first_date(source: Source) -> Path:
    """The earliest ``date_field`` per ``group_field`` value, by grouped statistics over id
    ranges (servers often refuse to page aggregated queries). Stored without geometry."""
    out = raw_path(source.id)
    g, dfield = source.group_field, source.date_field

    def stats(where: str, s: list[dict], grouped: bool) -> list[dict]:
        r = _request("GET", f"{source.url}/query", params={
            "where": where, "outStatistics": json.dumps(s), "f": "json",
            **({"groupByFieldsForStatistics": g} if grouped else {})}).json()
        if "error" in r:
            raise RuntimeError(f"{source.id}: {r['error']}")
        return [f["attributes"] for f in r["features"]]

    span = stats(source.where, [{"statisticType": "min", "onStatisticField": g, "outStatisticFieldName": "lo"},
                                {"statisticType": "max", "onStatisticField": g, "outStatisticFieldName": "hi"}], False)[0]
    rows: list[dict] = []
    a, step = int(span["lo"]), FIRST_DATE_ID_SPAN
    while a <= int(span["hi"]):
        got = stats(f"({source.where}) AND {g} >= {a} AND {g} < {a + step}",
                    [{"statisticType": "min", "onStatisticField": dfield, "outStatisticFieldName": "first_ms"}], True)
        if len(got) >= 1000 and step > 1:  # at or near a record limit: retry narrower
            step //= 2
            continue
        rows.extend(got)
        a += step
    df = pd.DataFrame(rows, columns=[g, "first_ms"])
    df.to_parquet(out)
    _record(source, out, records=len(df))
    return out


# --------------------------------------------------------------------------- plain files


def fetch_http_file(source: Source) -> Path:
    out = raw_path(source.id)
    tmp = out.with_suffix(out.suffix + ".part")
    with _session.get(source.url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(out)
    _record(source, out, last_modified=r.headers.get("Last-Modified"), bytes=out.stat().st_size)
    return out


# --------------------------------------------------------------------------- sheets kept in the repo

FIELD_SHEET_COLUMNS = ("place", "kind", "lat", "lon", "visited_on", "result")


def _local_csv(source: Source) -> Path:
    return REPO_ROOT / source.url


def fetch_local_csv(source: Source) -> Path:
    """A sheet kept in the repo (the team's field checks) -> raw parquet with point geometry
    from its lat/lon. The sheet's own checksum goes in the manifest, so a result always traces
    back to the exact sheet it came from; ``fetch`` reads it again whenever the sheet changes."""
    src = _local_csv(source)
    df = pd.read_csv(src, dtype=str, keep_default_na=False)
    missing = [c for c in FIELD_SHEET_COLUMNS if c not in df]
    if missing:
        raise ValueError(f"{source.id}: {src} lacks columns {missing}")
    lat, lon = pd.to_numeric(df["lat"], errors="coerce"), pd.to_numeric(df["lon"], errors="coerce")
    if lat.isna().any() or lon.isna().any():
        raise ValueError(f"{source.id}: rows without a numeric lat/lon: {df.loc[lat.isna() | lon.isna(), 'place'].tolist()}")
    out = raw_path(source.id)
    gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(lon, lat), crs=WGS84).to_parquet(out)
    _record(source, out, sheet_sha256=_sha256(src), count=len(df))
    return out


# --------------------------------------------------------------------------- Project Sidewalk

def fetch_project_sidewalk(source: Source, *, tiles: int = 4) -> Path:
    """Fetch label clusters in a grid of bbox tiles (the API can time out on all of DC)."""
    out = raw_path(source.id)
    xmin, ymin, xmax, ymax = REGION.bbox
    dx, dy = (xmax - xmin) / tiles, (ymax - ymin) / tiles
    feats: dict[int, dict] = {}
    for i in range(tiles):
        for j in range(tiles):
            bbox = f"{xmin + i*dx:.5f},{ymin + j*dy:.5f},{xmin + (i+1)*dx:.5f},{ymin + (j+1)*dy:.5f}"
            r = _request("GET", source.url, params={"bbox": bbox, "filetype": "geojson"}, timeout=600).json()
            for f in r["features"]:
                feats[f["properties"]["label_cluster_id"]] = f
    rows = []
    geoms = []
    for f in feats.values():
        p = dict(f["properties"])
        # Privacy: contributor identifiers are not needed for routing and are dropped.
        p.pop("users", None)
        p.pop("label_ids", None)
        p["tag_counts"] = json.dumps(p.get("tag_counts") or {})
        rows.append(p)
        geoms.append(shape(f["geometry"]))
    gdf = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=geoms, crs=WGS84)
    gdf.to_parquet(out)
    _record(source, out, records=len(gdf))
    return out


# --------------------------------------------------------------------------- DTM tiles


# A DTM cut to a region's box is exported on the analysis grid at this resolution: the
# county services differ in projection and units (Montgomery: EPSG:2893, US feet, 1 ft
# pixels), and 1 m is what the slope method samples at (graph/elevation.py).
DTM_RESOLUTION_M = 1.0


def fetch_image_service(source: Source, *, tile_px: int = 2000) -> Path:
    """Export a DTM as GeoTIFF tiles.

    Without a ``bbox``: the whole service at native resolution in its own CRS (DC). With one:
    only that box, on the analysis grid (EPSG:26985) at DTM_RESOLUTION_M, the server doing the
    reprojection; a county-wide service is never downloaded for one town.
    2000 px tiles: the DC server returns HTTP 500 for some dense 4000 px requests.
    Every tile is opened with rasterio after download; a corrupt tile is an error.
    """
    import rasterio
    out_dir = raw_path(source.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _request("GET", source.url, params={"f": "json"}).json()
    if source.bbox:
        from pyproj import Transformer

        from .config import ANALYSIS_CRS

        tr = Transformer.from_crs("EPSG:4326", ANALYSIS_CRS, always_xy=True)
        xs, ys = tr.transform([source.bbox[0], source.bbox[2], source.bbox[0], source.bbox[2]],
                              [source.bbox[1], source.bbox[1], source.bbox[3], source.bbox[3]])
        ext = {"xmin": float(min(xs)) // 1, "ymin": float(min(ys)) // 1, "xmax": float(max(xs)) // 1 + 1,
               "ymax": float(max(ys)) // 1 + 1}
        wkid = int(str(ANALYSIS_CRS).split(":")[-1])
        px = DTM_RESOLUTION_M
    else:
        ext = meta["extent"]
        wkid = ext["spatialReference"].get("latestWkid") or ext["spatialReference"]["wkid"]
        px = float(meta["pixelSizeX"])
    step = tile_px * px
    n = 0
    x = ext["xmin"]
    while x < ext["xmax"]:
        y = ext["ymin"]
        while y < ext["ymax"]:
            x2, y2 = min(x + step, ext["xmax"]), min(y + step, ext["ymax"])
            w, h = round((x2 - x) / px), round((y2 - y) / px)
            tile = out_dir / f"dtm_{tile_px}px_{int(x)}_{int(y)}.tif"
            if not tile.exists():
                r = _request(
                    "GET",
                    f"{source.url}/exportImage",
                    params={
                        "bbox": f"{x},{y},{x2},{y2}",
                        "bboxSR": wkid,
                        "imageSR": wkid,
                        "size": f"{w},{h}",
                        "format": "tiff",
                        "pixelType": "F32",
                        "interpolation": "RSP_NearestNeighbor",
                        "compression": "LZW",
                        "f": "image",
                    },
                    timeout=600,
                )
                if not r.content.startswith((b"II*\x00", b"MM\x00*")):
                    raise RuntimeError(f"{source.id}: non-TIFF response for tile {tile.name}: {r.content[:200]!r}")
                tmp = tile.with_suffix(".part")
                tmp.write_bytes(r.content)
                with rasterio.open(tmp) as ds:
                    if ds.crs is None or ds.width != w or ds.height != h:
                        raise RuntimeError(f"{source.id}: bad tile {tile.name}: crs={ds.crs} size={ds.width}x{ds.height}")
                tmp.replace(tile)
            n += 1
            y = y2
        x = x + step
    _record(source, out_dir, tiles=n, crs=f"EPSG:{wkid}", pixel_size_m=px)
    return out_dir


# --------------------------------------------------------------------------- OSM clip


def _shared_download(url: str, refresh: bool = False) -> tuple[Path, str | None]:
    """A large file several regions cut from (a statewide OSM extract), downloaded once into
    SHARED_DIR. Returns its path and the server's Last-Modified, kept beside it, which dates
    every clip taken from it. With ``refresh``, a newer build on the server replaces it: edits
    mapped in OSM reach the pipeline only through a new extract (Geofabrik builds daily)."""
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    out = SHARED_DIR / Path(url).name
    stamp = out.with_name(out.name + ".last_modified")
    if refresh and out.exists():
        served = _request("HEAD", url).headers.get("Last-Modified") or ""
        kept = stamp.read_text().strip() if stamp.exists() else ""
        if served and served != kept:
            log.info("%s: the server has a newer build (%s; kept %s), downloading it", out.name, served, kept or "undated")
            out.unlink()
        else:
            log.info("%s: the kept build is the server's current one (%s)", out.name, kept)
    if not out.exists():
        tmp = out.with_suffix(out.suffix + ".part")
        with _session.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            stamp.write_text(r.headers.get("Last-Modified") or "")
        tmp.replace(out)
    if not stamp.exists():
        # downloaded by hand: ask the server what it serves now, which is only right if the
        # file is that same build, so say where the date came from
        lm = _request("HEAD", url).headers.get("Last-Modified") or ""
        stamp.write_text(lm)
        log.warning("%s: no Last-Modified recorded with the download; took the server's current one (%s)", out.name, lm)
    return out, stamp.read_text().strip() or None


# Multipolygon relations a town clip keeps: areas the network is clipped by (osm.RESTRICTED_AREA_TAGS).
CLIP_RELATION_TAGS = (("landuse", "military"),)


def clip_pbf(src: Path, out: Path, bbox: tuple[float, float, float, float]) -> dict:
    """Cut a bounding box out of an OSM extract, keeping every way that has a node inside it
    complete (with all its nodes, also those outside), plus tagged nodes inside it.

    Streaming, without a node-location index: node coordinates are read directly, the ids
    of nodes inside the box are kept in memory (a town is a few hundred thousand), and the
    nodes the kept ways need from outside the box are added by ``BackReferenceWriter`` in a
    second read of the file. The only relations kept are the multipolygon areas in
    ``CLIP_RELATION_TAGS`` with a member way in the box (the writer adds their other members):
    Fort Detrick, a closed Army post inside Frederick, is mapped only as a relation.
    """
    import osmium

    xmin, ymin, xmax, ymax = bbox
    inside: set[int] = set()
    kept_ways: set[int] = set()
    n_ways = n_nodes = n_rels = 0
    tmp = out.with_suffix(".part.osm.pbf")
    tmp.unlink(missing_ok=True)
    with osmium.BackReferenceWriter(str(tmp), ref_src=str(src), overwrite=True) as writer:
        for obj in osmium.FileProcessor(str(src), osmium.osm.NODE | osmium.osm.WAY | osmium.osm.RELATION):
            if obj.is_node():
                loc = obj.location
                if loc.valid() and xmin <= loc.lon <= xmax and ymin <= loc.lat <= ymax:
                    inside.add(obj.id)
                    if obj.tags:
                        writer.add_node(obj)
                        n_nodes += 1
            elif obj.is_way():
                if any(n.ref in inside for n in obj.nodes):
                    writer.add_way(obj)
                    kept_ways.add(obj.id)
                    n_ways += 1
            elif any(obj.tags.get(k) == v for k, v in CLIP_RELATION_TAGS) and \
                    any(m.type == "w" and m.ref in kept_ways for m in obj.members):
                writer.add_relation(obj)
                n_rels += 1
    tmp.replace(out)
    return {"ways": n_ways, "tagged_nodes_inside": n_nodes, "nodes_inside": len(inside), "area_relations": n_rels}


def fetch_pbf_clip(source: Source, refresh: bool = False) -> Path:
    if not source.bbox:
        raise RuntimeError(f"{source.id}: a pbf_clip source needs a bbox")
    src, last_modified = _shared_download(source.url, refresh=refresh)
    out = raw_path(source.id)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = clip_pbf(src, out, source.bbox)
    log.info("%s: %s", source.id, counts)
    _record(source, out, last_modified=last_modified, bbox=list(source.bbox), bytes=out.stat().st_size,
            parent_sha256=_sha256(src), **counts)
    return out


# --------------------------------------------------------------------------- WMATA

# The fields kept from each WMATA list, under our own names.
_WMATA_FIELDS = {
    "wmata_entrances": ("Entrances", {"ID": "entrance_id", "Name": "name", "StationCode1": "station_code_1",
                                      "StationCode2": "station_code_2", "Lat": "lat", "Lon": "lon"}),
    "wmata_stations": ("Stations", {"Code": "code", "Name": "name", "Lat": "lat", "Lon": "lon"}),
}


def fetch_wmata_json(source: Source) -> Path:
    import os

    key = os.environ.get("WMATA_API_KEY")
    if not key:
        raise RuntimeError(f"{source.id}: set WMATA_API_KEY (in .env) to fetch from WMATA")
    root, fields = _WMATA_FIELDS[source.id]
    rows = _request("GET", source.url, headers={"api_key": key}).json()[root]
    df = pd.DataFrame(rows)[list(fields)].rename(columns=fields)
    out = raw_path(source.id)
    df.to_parquet(out, index=False)
    _record(source, out, records=len(df))
    return out


def _live_only(source: Source) -> Path:
    raise RuntimeError(f"{source.id} is read live when a trip is planned; there is nothing to fetch")


# --------------------------------------------------------------------------- basemap

# The backdrop map is detailed enough at zoom 15 (the tileset's own maximum); the page
# over-zooms from there.
BASEMAP_MAX_ZOOM = 15
# The label fonts the page's map style asks for, and the glyph ranges DC's labels need:
# Basic Latin + Latin-1, Latin Extended-A, general punctuation.
FONT_STACKS = ("Noto Sans Regular", "Noto Sans Medium", "Noto Sans Italic")
FONT_RANGES = (0, 256, 8192)


def tiles_covering(bbox: tuple[float, float, float, float], zoom: int) -> list[tuple[int, int, int]]:
    """Every web-mercator tile (z, x, y) that touches a lon/lat box at one zoom."""
    import math

    def xy(lon: float, lat: float) -> tuple[int, int]:
        n = 2 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
        return min(max(x, 0), n - 1), min(max(y, 0), n - 1)

    (x0, y1), (x1, y0) = xy(bbox[0], bbox[1]), xy(bbox[2], bbox[3])
    return [(zoom, x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


def fetch_pmtiles_extract(source: Source) -> Path:
    """Copy the tiles over the region out of a remote PMTiles archive, by HTTP range requests, into
    a small local archive. A tile the archive does not have is an error, not a hole."""
    from functools import lru_cache

    from pmtiles.reader import Reader
    from pmtiles.tile import zxy_to_tileid
    from pmtiles.writer import write

    @lru_cache(maxsize=512)  # the header and directories are read again for every tile
    def get_bytes(offset: int, length: int) -> bytes:
        return _request("GET", source.url, headers={"Range": f"bytes={offset}-{offset + length - 1}"}).content

    reader = Reader(get_bytes)
    header = reader.header()
    top = min(BASEMAP_MAX_ZOOM, header["max_zoom"])
    wanted = sorted((zxy_to_tileid(*t), t) for z in range(header["min_zoom"], top + 1) for t in tiles_covering(REGION.bbox, z))
    out = raw_path(source.id)
    tmp = out.with_suffix(out.suffix + ".part")
    with write(tmp) as w:
        for i, (tile_id, (z, x, y)) in enumerate(wanted, 1):
            data = reader.get(z, x, y)
            if data is None:
                raise RuntimeError(f"{source.id}: tile {z}/{x}/{y} is missing from {source.url}")
            w.write_tile(tile_id, bytes(data))
            if i % 100 == 0:
                log.info("%s: %d/%d tiles", source.id, i, len(wanted))
        w.finalize({"tile_type": header["tile_type"], "tile_compression": header["tile_compression"],
                    "min_lon_e7": int(REGION.bbox[0] * 1e7), "min_lat_e7": int(REGION.bbox[1] * 1e7),
                    "max_lon_e7": int(REGION.bbox[2] * 1e7), "max_lat_e7": int(REGION.bbox[3] * 1e7)}, reader.metadata())
    tmp.replace(out)
    _record(source, out, tiles=len(wanted), max_zoom=top, bbox=list(REGION.bbox), bytes=out.stat().st_size)
    return out


def fetch_glyphs(source: Source) -> Path:
    out_dir = raw_path(source.id)
    n = 0
    for stack in FONT_STACKS:
        (out_dir / stack).mkdir(parents=True, exist_ok=True)
        for start in FONT_RANGES:
            name = f"{start}-{start + 255}.pbf"
            (out_dir / stack / name).write_bytes(_request("GET", f"{source.url}/{requests.utils.quote(stack)}/{name}").content)
            n += 1
    _record(source, out_dir, files=n, stacks=list(FONT_STACKS), ranges=list(FONT_RANGES))
    return out_dir


# --------------------------------------------------------------------------- entry point

_FETCHERS = {
    "arcgis_layer": fetch_arcgis,
    "arcgis_first_date": fetch_arcgis_first_date,
    "http_file": fetch_http_file,
    "pbf_clip": fetch_pbf_clip,
    "project_sidewalk": fetch_project_sidewalk,
    "image_service": fetch_image_service,
    "pmtiles_extract": fetch_pmtiles_extract,
    "glyphs": fetch_glyphs,
    "wmata_json": fetch_wmata_json,
    "live": _live_only,
    "local_csv": fetch_local_csv,
}


def fetch(source_ids: list[str] | None = None, *, force: bool = False) -> None:
    ensure_dirs()
    import os

    everything = not source_ids
    for sid in source_ids or [i for i, s in SOURCES.items() if s.kind != "live"]:
        s = SOURCES[sid]
        if everything and s.kind == "wmata_json" and not os.environ.get("WMATA_API_KEY"):
            # Elevator notices are optional: a plain `pf fetch` must not need WMATA credentials.
            # Asked for by name, the same source fails loudly instead (fetch_wmata_json).
            log.info("skip %s (no WMATA_API_KEY; elevator notices stay off)", sid)
            continue
        p = raw_path(sid)
        done = sid in load_manifest() and p.exists()
        if done and s.kind == "local_csv":  # an edited sheet is a new version: read it again
            done = load_manifest()[sid].get("sheet_sha256") == _sha256(_local_csv(s))
        if done and not force:
            log.info("skip %s (already fetched)", sid)
            continue
        log.info("fetching %s", sid)
        if s.kind == "pbf_clip" and force:  # also take a newer statewide extract, with new OSM edits
            _FETCHERS[s.kind](s, refresh=True)
        else:
            _FETCHERS[s.kind](s)
