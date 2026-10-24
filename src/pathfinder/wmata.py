"""Metro elevator outages near the ends of a trip, from WMATA's live feed.

What this is for: someone walking to or from a Metro station needs to know, before setting
off, that the station's elevator is out. It is a notice on the trip, with its source and the
time it was read. It does not change any route: the walking graph follows streets, and
(checked 2026-10-06) none of its elevator links is a Metro elevator.

Product rules that shape it:
- Missing data is never favorable. If the feed cannot be read, a trip that ends at a station
  is told the elevator status is unknown, never left to assume all is well.
- Every claim carries source and date: each notice has ``source`` and ``as_of``.
- Nothing about the trip is sent to WMATA. The whole outage list is fetched, the same for
  everyone, and matched to the trip here.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer

from .config import ANALYSIS_CRS, WGS84
from .fetch import raw_path
from .sources import REGION, SOURCES

log = logging.getLogger(__name__)

API_KEY_ENV = "WMATA_API_KEY"
# Live, never stored. A region with Metro stations (Gaithersburg's nearest is Shady Grove)
# registers wmata_entrances / wmata_stations and sets transit_notices="wmata" to turn this on.
OUTAGES_URL = "https://api.wmata.com/Incidents.svc/json/ElevatorIncidents"
SOURCE_NAME = "WMATA elevator and escalator outages (live)"
# A trip that starts or ends this close to a station entrance is plausibly a trip to or from
# the station. Entrances of one station are up to a few hundred metres apart, so this is
# measured to the nearest entrance, not the station's centre.
ENTRANCE_RADIUS_M = 150.0
# WMATA refreshes the list every 20-30 s; asking once a minute is current enough for a walk
# and far inside the free tier (50,000 calls a day).
REFRESH_S = 60.0
# The plan must not wait on WMATA: past this the status is reported as unknown.
TIMEOUT_S = 4.0

_to_metres = Transformer.from_crs(WGS84, ANALYSIS_CRS, always_xy=True)


class StationIndex:
    """Station entrances, and which stations a point is at."""

    def __init__(self, entrances: pd.DataFrame, stations: pd.DataFrame):
        self._codes = [tuple(c for c in (a, b) if c) for a, b in zip(entrances["station_code_1"], entrances["station_code_2"])]
        self._x, self._y = _to_metres.transform(entrances["lon"].to_numpy(), entrances["lat"].to_numpy())
        self.names: dict[str, str] = dict(zip(stations["code"], stations["name"]))

    @classmethod
    def load(cls, entrances: Path | None = None, stations: Path | None = None) -> "StationIndex":
        return cls(pd.read_parquet(entrances or raw_path("wmata_entrances")), pd.read_parquet(stations or raw_path("wmata_stations")))

    @staticmethod
    def available() -> bool:
        if REGION.transit_notices != "wmata" or not {"wmata_entrances", "wmata_stations"} <= SOURCES.keys():
            return False
        return raw_path("wmata_entrances").is_file() and raw_path("wmata_stations").is_file()

    def stations_at(self, lonlat: tuple[float, float]) -> list[str]:
        """Codes of every station with an entrance within ENTRANCE_RADIUS_M (a station where
        lines cross has two codes, and an outage is filed under either)."""
        x, y = _to_metres.transform(*lonlat)
        near = np.nonzero(np.hypot(self._x - x, self._y - y) <= ENTRANCE_RADIUS_M)[0]
        return sorted({c for i in near for c in self._codes[i]})


class OutageFeed:
    """The current outage list, re-read at most once per REFRESH_S. ``current()`` returns
    (outages, read_at), or (None, None) when the list could not be read."""

    def __init__(self, api_key: str, get=None):
        self._key = api_key
        self._get = get or self._http_get
        self._lock = threading.Lock()
        self._at = 0.0
        self._value: tuple[list[dict] | None, str | None] = (None, None)

    def _http_get(self) -> list[dict]:
        r = requests.get(OUTAGES_URL, headers={"api_key": self._key}, timeout=TIMEOUT_S)
        r.raise_for_status()
        return r.json()["ElevatorIncidents"]

    def current(self) -> tuple[list[dict] | None, str | None]:
        with self._lock:
            if time.monotonic() - self._at >= REFRESH_S or self._at == 0.0:
                try:
                    outages = self._get()
                    # checked here, inside the guard: a reply of the wrong shape must become
                    # "unknown", never an error in a plan that has already found its routes
                    if not isinstance(outages, list) or not all(isinstance(o, dict) for o in outages):
                        raise ValueError("outage list is not a list of records")
                    self._value = (outages, datetime.now(timezone.utc).isoformat(timespec="seconds"))
                except Exception as e:  # network, quota, a changed format: all mean "not known right now"
                    log.warning("WMATA outage feed could not be read: %s", type(e).__name__)
                    self._value = (None, None)
                self._at = time.monotonic()
            return self._value


class ElevatorNotices:
    def __init__(self, stations: StationIndex, feed: OutageFeed):
        self.stations = stations
        self.feed = feed

    @classmethod
    def from_environment(cls) -> "ElevatorNotices | None":
        """None when there is no API key or the station list has not been fetched: the
        feature is then absent, and the response says nothing about elevators."""
        key = os.environ.get(API_KEY_ENV)
        if not key or not StationIndex.available():
            return None
        return cls(StationIndex.load(), OutageFeed(key))

    def for_trip(self, origin: tuple[float, float], destination: tuple[float, float]) -> list[dict]:
        """Notices for the stations at either end of a trip; empty when neither end is at one
        or every elevator there is in service."""
        ends = {end: self.stations.stations_at(at) for end, at in (("origin", origin), ("destination", destination))}
        if not any(ends.values()):
            return []
        outages, read_at = self.feed.current()
        out: list[dict] = []
        for end, codes in ends.items():
            if not codes:
                continue
            station = self.stations.names.get(codes[0], codes[0])
            if outages is None:
                out.append({"end": end, "station": station, "status": "unknown", "source": SOURCE_NAME,
                            "message": "Elevator status at this station could not be checked just now."})
                continue
            for o in outages:
                if o.get("UnitType") == "ELEVATOR" and o.get("StationCode") in codes:
                    out.append({"end": end, "station": o.get("StationName") or station, "status": "out_of_service",
                                "unit": o.get("UnitName"), "where": o.get("LocationDescription"),
                                "reason": o.get("SymptomDescription"),
                                # WMATA gives these in Eastern time, without an offset
                                "out_since_eastern": o.get("DateOutOfServ"),
                                "estimated_return_eastern": o.get("EstimatedReturnToService"),
                                "source": SOURCE_NAME, "as_of": read_at})
        return out
