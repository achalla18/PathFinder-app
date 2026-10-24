# PathFinder

A city accessibility app for **Maryland's 6th congressional district**: Gaithersburg,
Germantown, Frederick, Hagerstown, Cumberland, Frostburg and the towns between.

**For people getting around with a wheelchair, walker, cane or stroller** it plans
step-free routes from a mobility profile. It shows **where** sensible routes differ, usually
one to three decisive places: a missing curb ramp, a steep block, a street with no sidewalk.
It shows **what is known vs. unknown** about each of them, with source and date, and **how
far your priorities would have to move** before a different route wins, across crossings &
curbs, slope and sidewalk. Hard requirements (step-free, maximum slope) are never traded off.

**For cities and advocates** it maps the barriers that are on record, and the places nobody
has recorded, by census tract: the survey worklist.

It never certifies a route or place as accessible. It says what was recorded, by whom and
when.

PathFinder grew out of DistrictDirect, the same idea built and tested for Washington, DC;
it is a separate app with its own data, repo and command.

Built for the Congressional App Challenge (Maryland's 6th district).

## Run the demo

```bash
pf demo     # Washington, DC and the open-data builds of Gaithersburg and Frederick
```

It opens http://127.0.0.1:8000. The regions must be built first (see Quick start).

## Data and licensing

- This repository holds code and summary reports only; map data is downloaded by `pf fetch`
  and never committed.
- The public builds (`dc`, `gaithersburg_public`, `frederick_public`) use openly licensed
  data only. Some city and county sources have no open license; the app marks them
  "Internal use only", leaves them out of the public builds, and no data or results from
  them are published here.
- Code: [MIT License](LICENSE). Map data © OpenStreetMap contributors (ODbL); boundaries and
  elevation from MD iMAP; Washington, DC data from Open Data DC (CC BY 4.0).

## Quick start

```bash
python -m uv venv ~/.venvs/pathfinder --python 3.13
python -m uv pip install --python ~/.venvs/pathfinder/Scripts/python.exe -e ".[dev]"

export PF_REGION=gaithersburg_public  # or frederick_public, dc; data goes to ~/.pathfinder/data/regions/<region>
pf fetch                         # city limits + an OSM cut of the statewide Maryland extract
pf extract-osm
pf build-graph
pf enrich
pf coverage                      # -> reports/gaithersburg_public/phase0/coverage.md
pf barriers                      # -> reports/gaithersburg_public/barriers/barriers.md
pf route 39.1434,-77.2014 39.1210,-77.2370 --profile manual_wheelchair
python -m pytest -q
```

