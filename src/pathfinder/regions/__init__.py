"""Regions the pipeline can run on. ``PF_REGION`` picks one (default ``gaithersburg``).

Each pilot town also has a public build (``<town>_public``) made from open sources only: the
one to show anyone outside the team (rule 7)."""

from __future__ import annotations

from dataclasses import replace

from ..config import REGION_ID
from .base import COVERAGE_FAMILIES, ROLES, Region, Source
from .dc import DC
from .maryland import FREDERICK, GAITHERSBURG

PUBLIC_SUFFIX = "_public"
# Roles a public build leaves out although their source is open: the post-ADA ramp rule was
# tested against the City of Frederick's internal survey, so it is not shown outside until that
# use is agreed.
PUBLIC_DROPPED_ROLES = ("parcels",)


def is_internal(source: Source) -> bool:
    """Held back under rule 7: the license line says "Internal use only"."""
    return "internal use" in source.license.lower()


def public(region: Region) -> Region:
    """The region built from open sources only, for anything shown outside the team (rule 7):
    every source whose license says "Internal use only" is left out with the roles it plays, so
    what it graded stays unknown. Its own id gives it its own data and report folders."""
    sources = {k: s for k, s in region.sources.items() if not is_internal(s)}
    roles = {r: sid for r, sid in region.roles.items() if sid in sources and r not in PUBLIC_DROPPED_ROLES}
    return replace(region, id=region.id + PUBLIC_SUFFIX, sources=sources, roles=roles,
                   coverage={f: None for f in region.coverage})


# DC's sources are all open, so it is its own public build.
REGIONS: dict[str, Region] = {DC.id: DC, **{r.id: r for base in (GAITHERSBURG, FREDERICK) for r in (base, public(base))}}


def get(region_id: str) -> Region:
    try:
        return REGIONS[region_id]
    except KeyError:
        raise KeyError(f"unknown region {region_id!r}; known: {', '.join(REGIONS)}") from None


def current() -> Region:
    return get(REGION_ID)


__all__ = ["COVERAGE_FAMILIES", "PUBLIC_SUFFIX", "REGIONS", "ROLES", "Region", "Source", "current", "get",
           "is_internal", "public"]
