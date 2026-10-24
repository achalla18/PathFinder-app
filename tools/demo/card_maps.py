"""The map on each place's card on the demo landing page: every walkway inside the place's line,
indigo where something records whether there is a sidewalk, faint where nothing does, and a
coral dot at each crossing end whose curb is recorded. The density of colour is the point:
it shows at a glance how much each place's data knows.

    python tools/demo/card_maps.py      # web/landing/img/<region>.png for every demo place

Built from the same graphs the servers route on (open-data builds only: rule 7).
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pathfinder.config import DATA_ROOT, REPO_ROOT  # noqa: E402
from pathfinder.demo import PLACES  # noqa: E402
from pathfinder.regions import REGIONS  # noqa: E402

OUT = REPO_ROOT / "web" / "landing" / "img"
INDIGO, CORAL, FAINT = "#5b5bd6", "#e4572e", "#b9b4a8"
WIDTH_IN, DPI = 9.0, 110


def render(region_id: str) -> Path:
    region = REGIONS[region_id]
    folder = DATA_ROOT / "regions" / region_id
    e = gpd.read_parquet(folder / "graph" / "edges.parquet")
    line = gpd.read_parquet(folder / "raw" / f"{region.boundary}.parquet").to_crs(e.crs).union_all()
    e = e[~e["outside_region"].astype(bool)]
    known = e["representation"].eq("pedestrian_way") | (
        e["representation"].eq("street_centerline") & e["sidewalk_presence_status"].ne("unknown"))
    cr = e[e["kind"].eq("crossing")]
    ends = [cr.geometry.apply(lambda g, i=i: g.interpolate(i, normalized=True))[cr[f"kerb_{x}_status"].ne("unknown")]
            for i, x in ((0.0, "a"), (1.0, "b"))]
    xmin, ymin, xmax, ymax = line.bounds
    fig = plt.figure(figsize=(WIDTH_IN, WIDTH_IN * 0.62), dpi=DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    gpd.GeoSeries([line], crs=e.crs).boundary.plot(ax=ax, color=FAINT, linewidth=0.8, linestyle=(0, (3, 2)))
    e[~known].plot(ax=ax, color=FAINT, linewidth=0.35, alpha=0.7)
    e[known].plot(ax=ax, color=INDIGO, linewidth=0.55)
    for pts in ends:
        if len(pts):
            gpd.GeoSeries(pts.values, crs=e.crs).plot(ax=ax, color=CORAL, markersize=0.6)
    # fill the card: centre on the place and keep the card's shape
    cx, cy, w, h = (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin
    aspect = 0.62
    if h / w > aspect:
        w = h / aspect
    else:
        h = w * aspect
    ax.set_xlim(cx - w / 2 * 1.04, cx + w / 2 * 1.04)
    ax.set_ylim(cy - h / 2 * 1.04, cy + h / 2 * 1.04)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{region_id}.png"
    fig.savefig(out, transparent=True)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for p in PLACES:
        print(render(p.region))
