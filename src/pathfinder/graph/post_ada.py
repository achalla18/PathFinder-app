"""Post-ADA curb ramps: a place is likely ramped when every building around it is newer than
the ADA's curb-ramp rule for new streets.

ADA Title II has required curb ramps where new or altered streets meet sidewalks since 1992.
The streets in a subdivision are laid before its houses, so a junction whose oldest nearby
building dates from 1995 or later was most likely built with ramps. The rule says nothing about
older places, and it is inferred: it names no ramp anyone saw.

The rule and its bar were fixed on 2026-10-21 before any parcel was fetched
(docs/PROGRESS.md, Decisions log); tools/post_ada/evaluate.py scores it against the City of
Frederick's ramp survey. The constants below are that protocol and are not tuned.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

POST_ADA_YEAR = 1995  # ADA Title II curb ramps on new streets from 1992, plus three years' margin
PARCEL_RADIUS_M = 60.0  # the lots around a corner: about two to four on each street
MIN_BUILT_PARCELS = 4  # fewer says too little about the neighbourhood
YEAR_MIN, YEAR_MAX = 1700, 2026  # YEARBLT outside this is no structure (0, blank) or an error
SOURCE = "derived:post_ada_parcels"
# Sidewalks: subdivision streets only. Arterials and unclassified roads are often older
# alignments, state-built or rural, whatever the age of the houses beside them. Fixed with the
# rule's bar on 2026-10-21 (docs/PROGRESS.md, Decisions log). The sidewalk rule FAILED its bar
# (62.7% right in street photos against 95%) and is not used: these constants are kept only so
# tools/post_ada/sidewalk_check.py can reproduce that test. Never wire them into enrich.
SIDEWALK_HIGHWAYS = ("residential", "living_street", "tertiary")
SIDEWALK_CLAIM = "both"


def built_parcels(parcels: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Parcels with a structure, with ``year`` as an integer."""
    y = pd.to_numeric(parcels["YEARBLT"], errors="coerce")
    keep = y.between(YEAR_MIN, YEAR_MAX)
    return gpd.GeoDataFrame({"year": y[keep].astype(int).to_numpy()}, geometry=parcels.geometry[keep].values, crs=parcels.crs)


def neighbourhood(places: gpd.GeoSeries, built: gpd.GeoDataFrame) -> pd.DataFrame:
    """Per place: built parcels within PARCEL_RADIUS_M, the oldest one's year, and whether the
    place is post-ADA. Index follows ``places``."""
    zone = gpd.GeoDataFrame({"pid": np.arange(len(places))}, geometry=places.buffer(PARCEL_RADIUS_M).values, crs=places.crs)
    j = gpd.sjoin(built, zone, predicate="within")
    g = j.groupby("pid")["year"]
    out = pd.DataFrame({"n_built": g.size(), "oldest": g.min()}).reindex(np.arange(len(places)))
    out["n_built"] = out["n_built"].fillna(0).astype(int)
    out["post_ada"] = (out["n_built"] >= MIN_BUILT_PARCELS) & (out["oldest"] >= POST_ADA_YEAR)
    out.index = places.index
    return out


def sidewalk_candidates(edges: gpd.GeoDataFrame) -> pd.Series:
    """Street-centerline edges the sidewalk rule may speak for: a subdivision road class and
    not a service road or alley. Whether presence is already known is the caller's matter."""
    from .enrich import NO_PLANIMETRIC_PRESENCE_KINDS

    return (edges["representation"].eq("street_centerline") & edges["highway"].isin(SIDEWALK_HIGHWAYS)
            & ~edges["kind"].isin(NO_PLANIMETRIC_PRESENCE_KINDS))


def as_of(parcels: gpd.GeoDataFrame) -> str | None:
    """The parcel data's date ("2026MAY" -> "2026-05"), or None when it gives none."""
    d = pd.to_datetime(parcels["SDATDATE"].dropna().astype(str), format="%Y%b", errors="coerce").dropna()
    return None if d.empty else d.max().strftime("%Y-%m")
