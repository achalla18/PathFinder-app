"""Write the field sheet for the current region from the blocked-trip worklist.

    python tools/field_checks/make_sheet.py            # field/<region>.csv, refuses to overwrite
    python tools/field_checks/make_sheet.py --update   # add worklist places not yet on the sheet

The sheet is the team's record of what was seen on site (field/README.md). It is never
rewritten: answered rows are evidence, and ``--update`` only appends new places. Run
``tools/blocked_trips/blocked_trips.py`` first.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from pathfinder.config import REGION_ID, REPO_ROOT, REPORTS_DIR

COLUMNS = ["place", "kind", "what_to_check", "street", "lat", "lon", "map", "trips_blocked", "visited_on", "result", "notes"]
MAP_LINK = "https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=20/{lat}/{lon}"  # opens on a phone, pin on the spot
WHAT = {
    "corner": "the curb at this end of the marked crosswalk: ramp / flush / no_ramp",
    "junction": "every corner with a sidewalk: all_ramped / missing_ramp",
}


def sheet_rows(worklist: pd.DataFrame) -> pd.DataFrame:
    """One row per worklist place the field can answer: the curb at a crossing end, or a
    junction crossed without a mapped crossing. Recorded barriers are already known."""
    w = worklist[worklist["reason"].str.contains("curb unknown")]
    kind = w["place"].str.split(":").str[0]
    w = w[kind.isin(list(WHAT))]
    kind = kind[w.index]
    lat, lon = w["lat"].map("{:.6f}".format), w["lon"].map("{:.6f}".format)
    return pd.DataFrame({
        "place": w["place"], "kind": kind, "what_to_check": kind.map(WHAT), "street": w["detail"],
        "lat": lat, "lon": lon, "map": [MAP_LINK.format(lat=a, lon=o) for a, o in zip(lat, lon)],
        "trips_blocked": w["n_trips"],
        "visited_on": "", "result": "", "notes": "",
    })[COLUMNS]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--update", action="store_true", help="append worklist places not yet on the sheet")
    args = ap.parse_args()
    out = REPO_ROOT / "field" / f"{REGION_ID}.csv"
    rows = sheet_rows(pd.read_csv(REPORTS_DIR / "blocked" / "worklist.csv"))
    if out.exists():
        if not args.update:
            print(f"{out} exists; it holds answers and is never rewritten (use --update to add places)", file=sys.stderr)
            return 1
        have = pd.read_csv(out, dtype=str, keep_default_na=False)
        new = rows[~rows["place"].isin(have["place"])]
        pd.concat([have, new.astype(str)], ignore_index=True).to_csv(out, index=False)
        print(f"{out}: {len(new)} places added, {len(have)} kept")
        return 0
    out.parent.mkdir(exist_ok=True)
    rows.to_csv(out, index=False)
    print(f"{out}: {len(rows)} places")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
