# Field checks

The team's own record of curbs seen on site, one sheet per region (`<region>.csv`). Each row is
a place that blocks manual-wheelchair trips because nothing records its curbs
(`reports/<region>/blocked/worklist.csv`). `pf fetch` reads the sheet into the pipeline as the
source `<region>_field_checks`, and reads it again whenever the file changes.

These are the team's observations, so no other owner restricts them (rule 7 does not apply).

## Filling a row

Open the `map` link, stand at the pin, and fill three columns:

- `visited_on`: the day you looked, `YYYY-MM-DD`.
- `result`: one of the answers below, or `unclear` if you could not tell. Leave it blank if you
  have not been there.
- `notes`: anything else worth knowing, such as a steep or broken ramp, a missing sidewalk, a
  parked car or construction. Notes are not read by the pipeline. Never write anyone's name.

| `kind` | Where to look | `result` |
|---|---|---|
| `corner` | The one curb at the pin, where the marked crosswalk meets the sidewalk | `ramp` (a curb ramp), `flush` (no curb: sidewalk and street meet level), `no_ramp` (a raised curb with no ramp) |
| `junction` | Every corner of the intersection at the pin that has a sidewalk | `all_ramped` (each has a ramp or flush curb), `missing_ramp` (at least one has a raised curb with no ramp) |

Do not change `place`, `kind`, `lat` or `lon`. To check a place not on the sheet, add a row with
its own `place` name, a `kind`, and the coordinates of the curb (corner) or the middle of the
intersection (junction).

## What happens to an answer

- **Corner:** the curb becomes `verified`, credited to the field check and dated by
  `visited_on`. It takes priority over every other source. If the corner has no ramp, or is
  flush, the city's verdict on a ramp nearby is not carried.
- **Junction:** the answer replaces the count of ramp records there. It stays `inferred`, as
  every junction claim does, because which corners a walker uses is not known.
- A typo in `kind` or `result`, or a result without a date, stops `pf enrich` with the row named.
  It is never skipped silently.

Then rerun `pf enrich`, `pf coverage`, `pf barriers`, `pf crux --fresh` and
`python tools/blocked_trips/blocked_trips.py` to see what opened.

## Reports from the app

Anyone using the app can report a barrier from the map (Barriers → Report a barrier). Reports
stay on the computer running the server. To put them on this sheet as places to visit, run
`python tools/field_checks/add_reports.py` (with `PF_REGION` set to the region).

- A curb report (broken or missing ramp) becomes a `corner` row: visit it and answer it like any
  corner.
- Anything else becomes a `note` row. The pipeline never reads these. Write what you found in
  `notes`, and leave `result` blank: a result on a `note` row stops `pf enrich`.

A report alone changes nothing PathFinder says; only your visit does.

To add new worklist places after a rerun:
`python tools/field_checks/make_sheet.py --update`. It never rewrites answered rows.
