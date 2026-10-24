"""Place search over a local gazetteer built from the OSM extract.

Searching happens on our own server against data already on disk, so what a person types
(often their home or where they are going) is never sent to a third-party geocoder.

``build()`` (run by ``pf extract-osm``) writes INTERIM_DIR/places.parquet: named places
(shops, amenities, stations, parks, buildings) and street addresses, one row each with a
point. ``PlaceIndex.search`` matches every word of the query as a prefix of a word of the
place, after expanding the usual street abbreviations on both sides.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import osmium
import pandas as pd

from .config import INTERIM_DIR, ensure_dirs
from .fetch import raw_path
from .sources import REGION

log = logging.getLogger(__name__)

PLACES_FILE = "places.parquet"
# Tags that make a named OSM object a place someone might search for, most specific first.
PLACE_KEYS = ("railway", "public_transport", "amenity", "shop", "tourism", "leisure", "office", "historic",
              "healthcare", "craft", "building")
# Ranking when several places match equally well: stations and landmarks before shops, shops
# before plain buildings and addresses.
KIND_RANK = {"railway": 0, "public_transport": 0, "tourism": 1, "historic": 1, "leisure": 1, "amenity": 2,
             "healthcare": 2, "shop": 3, "office": 3, "craft": 3, "building": 4, "address": 5}
ABBREVIATIONS = {
    "st": "street", "ave": "avenue", "av": "avenue", "rd": "road", "blvd": "boulevard", "dr": "drive", "pl": "place",
    "ct": "court", "ln": "lane", "ter": "terrace", "pkwy": "parkway", "hwy": "highway", "sq": "square", "cir": "circle",
    "nw": "northwest", "ne": "northeast", "sw": "southwest", "se": "southeast", "n": "north", "s": "south",
    "e": "east", "w": "west", "mt": "mount", "ft": "fort", "&": "and",
}
MAX_RESULTS = 8
# OSM often maps one place twice (a node and its building). Entries with the same name and
# kind this close together are that; farther apart they are different branches of a chain.
SAME_PLACE_M = 75.0


def normalize(text: str) -> str:
    """Lower-case words with street abbreviations spelled out, so "1600 Penn Ave NW" and
    "1600 Pennsylvania Avenue Northwest" share their words."""
    words = re.findall(r"[a-z0-9&]+", text.lower().replace("'", ""))
    return " ".join(ABBREVIATIONS.get(w, w) for w in words)


class _Handler(osmium.SimpleHandler):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict] = []

    def _add(self, tags, lon: float, lat: float) -> None:
        name = tags.get("name")
        kind = next((k for k in PLACE_KEYS if k in tags), None)
        if name and kind and not (kind == "building" and tags.get("building") in ("yes", "residential", "house", "garage")):
            self.rows.append({"label": name, "detail": tags.get(kind).replace("_", " "), "kind": kind, "lon": lon, "lat": lat})
        number, street = tags.get("addr:housenumber"), tags.get("addr:street")
        if number and street:
            unit = f" #{tags['addr:unit']}" if "addr:unit" in tags else ""
            self.rows.append({"label": f"{number} {street}{unit}", "detail": name or "address", "kind": "address",
                              "lon": lon, "lat": lat})

    def node(self, n) -> None:
        if n.tags:
            self._add(n.tags, n.location.lon, n.location.lat)

    def way(self, w) -> None:
        if not w.tags or ("name" not in w.tags and "addr:housenumber" not in w.tags):
            return
        pts = [(nd.location.lon, nd.location.lat) for nd in w.nodes if nd.location.valid()]
        if pts:  # the middle of the outline is close enough to route to
            lon, lat = np.mean(pts, axis=0)
            self._add(w.tags, float(lon), float(lat))


def merge_same_place(df: pd.DataFrame) -> pd.DataFrame:
    """Drop repeated mappings of one physical place: same normalized name and kind within
    SAME_PLACE_M of an entry already kept. Branches of a chain across town all survive."""
    lat0 = np.deg2rad(df["lat"].mean())
    x = (np.deg2rad(df["lon"]) * np.cos(lat0) * 6_371_008.8).to_numpy()
    y = (np.deg2rad(df["lat"]) * 6_371_008.8).to_numpy()
    keep = np.ones(len(df), dtype=bool)
    for idx in df.groupby(["search", "kind"], sort=False).indices.values():
        if len(idx) == 1:
            continue
        kept: list[int] = []
        for i in idx:
            if kept and np.min(np.hypot(x[kept] - x[i], y[kept] - y[i])) <= SAME_PLACE_M:
                keep[i] = False
            else:
                kept.append(i)
    return df[keep].reset_index(drop=True)


def build(pbf: Path | None = None) -> Path:
    ensure_dirs()
    h = _Handler()
    h.apply_file(str(pbf or raw_path(REGION.osm)), locations=True)
    df = pd.DataFrame(h.rows)
    df["search"] = df["label"].map(normalize)
    df = merge_same_place(df[df["search"].ne("")].reset_index(drop=True))
    out = INTERIM_DIR / PLACES_FILE
    df.to_parquet(out)
    log.info("places: %d (%s)", len(df), df["kind"].value_counts().to_dict())
    return out


class PlaceIndex:
    def __init__(self, path: Path | None = None):
        path = path or INTERIM_DIR / PLACES_FILE
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run `pf extract-osm` to build the place index")
        df = pd.read_parquet(path)
        self.df = df
        self.search_text = df["search"]
        self.padded = " " + df["search"]  # so " word" finds a word start anywhere
        self.rank = df["kind"].map(KIND_RANK).fillna(9).to_numpy()
        self.lon, self.lat = df["lon"].to_numpy(), df["lat"].to_numpy()

    def search(self, query: str, near: tuple[float, float] | None = None, limit: int = MAX_RESULTS) -> list[dict]:
        """Places whose words start with every word of ``query``. Exact names first, then
        names that begin with the query, then by kind and by distance from ``near`` (lon, lat)."""
        words = re.findall(r"[a-z0-9&]+", query.lower().replace("'", ""))
        if sum(map(len, words)) < 2:
            return []
        hit = np.ones(len(self.df), dtype=bool)
        for word in words:
            # "st" may be the start of "station" or short for "street": accept either reading
            m = self.padded.str.contains(" " + word, regex=False).to_numpy()
            if word in ABBREVIATIONS:
                m = m | self.padded.str.contains(" " + ABBREVIATIONS[word], regex=False).to_numpy()
            hit = hit & m
        idx = np.flatnonzero(hit)
        if len(idx) == 0:
            return []
        text = self.search_text.iloc[idx]
        typed, spelled = " ".join(words), normalize(query)
        exact = text.eq(typed) | text.eq(spelled)
        starts = text.str.startswith(typed) | text.str.startswith(spelled)
        quality = np.where(exact, 0, np.where(starts, 1, 2))
        if near is not None:  # degrees are fine for ordering within one city
            dist = np.hypot((self.lon[idx] - near[0]) * np.cos(np.deg2rad(near[1])), self.lat[idx] - near[1])
        else:
            dist = np.zeros(len(idx))
        # ties go to the nearer place, then to the shorter name (the station before its parking garage)
        order = np.lexsort((text.str.len().to_numpy(), dist, self.rank[idx], quality))[:limit]
        rows = self.df.iloc[idx[order]]
        return [{"label": r.label, "detail": r.detail, "kind": r.kind, "lon": round(r.lon, 6), "lat": round(r.lat, 6)}
                for r in rows.itertuples()]
