"""Add barrier reports sent from the app's map onto the team's field sheet, as places to visit.

    PF_REGION=frederick_public python tools/field_checks/add_reports.py

Reports are kept by the server that received them (<data dir>/reports/barrier_reports.csv; never
sent anywhere). A report is someone's word, not evidence: it becomes an unanswered row, and only a
team member's visit (visited_on + result) changes what PathFinder says. A curb report becomes a
``corner`` the visitor answers as usual; anything else becomes a ``note`` row, which the pipeline
never reads, for the team to look at. Reports already on the sheet are not added again.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from make_sheet import COLUMNS, MAP_LINK, WHAT  # noqa: E402

from pathfinder.api import REPORTS_FILE  # noqa: E402
from pathfinder.config import REPO_ROOT  # noqa: E402
from pathfinder.sources import REGION  # noqa: E402

CURB_REPORTS = {"broken_ramp", "no_ramp"}
LABELS = {"broken_ramp": "broken curb ramp", "no_ramp": "no curb ramp", "blocked_sidewalk": "sidewalk blocked",
          "no_sidewalk": "no sidewalk", "too_steep": "too steep", "other": "other"}
NOTE_WHAT = "a report from the app: look, and note what you find (not read by the pipeline)"


def rows(reports: pd.DataFrame) -> pd.DataFrame:
    curb = reports["kind"].isin(CURB_REPORTS)
    lat = pd.to_numeric(reports["lat"]).map("{:.6f}".format)
    lon = pd.to_numeric(reports["lon"]).map("{:.6f}".format)
    # the same report always gets the same place name, so a second run adds nothing
    place = [f"report:{uuid.uuid5(uuid.NAMESPACE_URL, f'{d}|{k}|{a}|{o}|{n}').hex[:10]}"
             for d, k, a, o, n in zip(reports["day"], reports["kind"], lat, lon, reports["note"])]
    return pd.DataFrame({
        "place": place, "kind": curb.map({True: "corner", False: "note"}),
        "what_to_check": curb.map({True: WHAT["corner"], False: NOTE_WHAT}), "street": "",
        "lat": lat, "lon": lon, "map": [MAP_LINK.format(lat=a, lon=o) for a, o in zip(lat, lon)],
        "trips_blocked": "", "visited_on": "", "result": "",
        "notes": [f"reported {d} in the app: {LABELS.get(k, k)}" + (f" ({n})" if n else "")
                  for d, k, n in zip(reports["day"], reports["kind"], reports["note"])],
    })[COLUMNS]


def main() -> int:
    sid = REGION.role("field_checks")
    if sid is None:
        print(f"{REGION.id} has no field sheet (no field_checks role)", file=sys.stderr)
        return 1
    if not REPORTS_FILE.exists():
        print(f"no reports yet ({REPORTS_FILE})")
        return 0
    sheet = REPO_ROOT / REGION.sources[sid].url
    have = pd.read_csv(sheet, dtype=str, keep_default_na=False)
    new = rows(pd.read_csv(REPORTS_FILE, dtype=str, keep_default_na=False))
    new = new[~new["place"].isin(have["place"])]
    pd.concat([have, new], ignore_index=True).to_csv(sheet, index=False)
    print(f"{sheet}: {len(new)} reports added as places to visit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
