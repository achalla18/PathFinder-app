"""Paths and spatial constants.

Raw and derived data live OUTSIDE the repository by default: they are multi-gigabyte,
reproducible from ``pf fetch``, and must never be committed or land in a cloud-synced
folder. Override with the ``PF_DATA_DIR`` environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env(path: Path | None = None) -> None:
    """Read ``KEY=value`` lines from the repo's ``.env`` (never committed) into the
    environment. A variable that is already set wins, so a deployment's own settings are
    never overridden by a file."""
    path = path or REPO_ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()

# The region every command works on (see ``regions``). Each region has its own data and
# report folders, so building one never overwrites another's graph, trips or reports.
DEFAULT_REGION = "gaithersburg"
REGION_ID = os.environ.get("PF_REGION", DEFAULT_REGION).strip().lower() or DEFAULT_REGION

DATA_ROOT = Path(os.environ.get("PF_DATA_DIR", Path.home() / ".pathfinder" / "data"))
DATA_DIR = DATA_ROOT / "regions" / REGION_ID
SHARED_DIR = DATA_ROOT / "shared"  # inputs several regions cut from (the statewide OSM extract)
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
GRAPH_DIR = DATA_DIR / "graph"
REPORTS_DIR = REPO_ROOT / "reports" / REGION_ID  # small, human-readable outputs are committed

# NAD83 / Maryland (meters): Maryland State Plane. All distances,
# buffers and lengths are computed here.
ANALYSIS_CRS = "EPSG:26985"
# Exchange/storage CRS.
WGS84 = "EPSG:4326"

# Walking speed used to convert length to time. 1.2 m/s is a common planning value for
# general pedestrians; profile-specific speeds are applied in routing.cost.
DEFAULT_WALK_SPEED_MPS = 1.2


def ensure_dirs() -> None:
    for d in (RAW_DIR, INTERIM_DIR, GRAPH_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
