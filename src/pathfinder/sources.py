"""Registry of every external data source, for the current region.

Each entry is the single source of truth for where a dataset comes from, what it can and
cannot tell us, and when its availability was last verified. Fetchers read this registry;
nothing else should hard-code a URL. The entries themselves live with their region
(``regions/dc.py``, ``regions/maryland.py``); ``PF_REGION`` picks which registry this is.
"""

from __future__ import annotations

from .regions import Source, current

REGION = current()
SOURCES: dict[str, Source] = REGION.sources


def by_tag(tag: str) -> list[Source]:
    return [s for s in SOURCES.values() if tag in s.tags]


__all__ = ["REGION", "SOURCES", "Source", "by_tag"]
