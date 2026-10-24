"""What blocks the sampled trips that have no manual-wheelchair route (criterion 4).

For each trip `pf crux` found infeasible under the manual-wheelchair requirement (step-free,
8.33%, allow likely), the router is re-run with looser settings: first allow unknown, then also
no grade limit. On the resulting route set, the route needing the fewest fixes is taken, and every
place on it that fails the real requirement is listed with its reason: a curb with no record at
a marked crossing or at an unmarked junction crossing, unknown slope, a measured slope over the
limit, or steps.

The worklist is a greedy pick of the fewest unknown places that, if each turned out to be
passable, would open the most trips. It is an upper bound: a checked corner may turn out to have no
ramp. Places blocked by measured slope are never on it, since a check cannot clear them.

Usage (after `pf crux`, with PF_REGION set):  python tools/blocked_trips/blocked_trips.py
Writes reports/<region>/blocked/. Internal while the region uses unlicensed data (rule 7).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from pathfinder.config import ANALYSIS_CRS, DATA_DIR, RAW_DIR, REGION_ID, REPORTS_DIR
from pathfinder.graph.enrich import KERB_BARRIER, _crossing_outer_ends
from pathfinder.routing.cost import Preferences, Requirements, feasible_mask
from pathfinder.routing.router import Network

GRADE_LIMIT = 8.33  # the manual-wheelchair profile's limit, as in pf crux
REQ = Requirements(step_free=True, max_grade_pct=GRADE_LIMIT, certainty="allow_likely")
LOOSE = [("allow_unknown", Requirements(step_free=True, max_grade_pct=GRADE_LIMIT, certainty="allow_unknown")),
         ("allow_unknown+no_grade", Requirements(step_free=True, max_grade_pct=None, certainty="allow_unknown"))]
# OSM classes that in these pilots are mostly state roads (MDOT SHA); the extract has no route refs.
MAJOR = {"trunk", "trunk_link", "primary", "primary_link"}
BAR_TRIPS_PCT = 50.0  # criterion 4 (docs/PROGRESS.md, Decisions log 2026-10-15)
ACCEPT = {"verified", "inferred"}
NO_TERRAIN = {"steps", "escalator", "elevator", "corridor"}


def place_reasons(r, node_streets, node_major, name_major) -> list[tuple]:
    """(key, reason, detail, major_road) for each requirement this edge fails."""
    out = []
    if r.kind == "crossing":
        for end in ("kerb_a", "kerb_b"):
            k, s = r[end], r[f"{end}_status"]
            major = bool(name_major.get(r.crossed_street or "", False)) or r.highway in MAJOR
            if k in KERB_BARRIER and s != "unknown":
                out.append((f"corner:{r.edge_id}:{end}", f"recorded {k}", r.crossed_street, major))
            elif s not in ACCEPT:
                out.append((f"corner:{r.edge_id}:{end}", "curb unknown, marked crossing", r.crossed_street, major))
    for end in ("u", "v"):
        if bool(r[f"xing_{end}"]):
            k = r[f"xing_{end}_kerb"] if isinstance(r[f"xing_{end}_kerb"], str) else "unknown"
            if k not in ("no_curb", "ramp_likely"):
                node = r[end]
                others = sorted(node_streets[node] - {r["name"]})
                out.append((f"junction:{node}", "curb unknown, unmarked junction crossing" if k == "unknown" else k,
                            "; ".join(others[:2]) or r["name"], node_major[node]))
    if r.step_free is False:
        out.append((f"edge:{r.edge_id}", "steps", r["name"], False))
    elif r.step_free_status not in ACCEPT and r.kind != "crossing":
        out.append((f"edge:{r.edge_id}", "step-free unknown", r["name"], r.highway in MAJOR))
    if r.kind not in NO_TERRAIN:
        g = pd.to_numeric(r.grade_max_pct, errors="coerce")
        if pd.notna(g) and g > GRADE_LIMIT:
            out.append((f"edge:{r.edge_id}", "steeper than 8.33%", f"{g:.1f}% ({r.grade_max_pct_status})", r.highway in MAJOR))
        elif r.grade_max_pct_status not in ACCEPT:
            out.append((f"edge:{r.edge_id}", "slope unknown", r["name"], r.highway in MAJOR))
    return out


def greedy(need: dict[int, set[str]]) -> list[tuple[str, int]]:
    """Places in pick order with the trips opened so far, favouring places that finish trips."""
    opened, chosen, order = set(), set(), []
    while len(opened) < len(need):
        score = Counter()
        for t, ks in need.items():
            if t not in opened:
                rest = ks - chosen
                for k in rest:
                    score[k] += 1 / len(rest)
        k = score.most_common(1)[0][0]
        chosen.add(k)
        opened |= {t for t, ks in need.items() if ks <= chosen}
        order.append((k, len(opened)))
    return order


def main() -> None:
    out_dir = REPORTS_DIR / "blocked"
    out_dir.mkdir(parents=True, exist_ok=True)
    net = Network.load(lean=False)
    E = net.edges
    ok_real = feasible_mask(E, REQ)
    res = pd.read_parquet(DATA_DIR / "analysis" / "crux_results.parquet")
    sf = res[res.profile == "stepfree_likely"]
    blocked = sf[sf.status == "no_feasible_route"]

    street = E[E["kind"].str.startswith("street")]
    node_streets, node_major = defaultdict(set), defaultdict(bool)
    for u, v, n, h in zip(street.u, street.v, street.name.fillna(""), street.highway.fillna("")):
        for x in (u, v):
            if n:
                node_streets[x].add(n)
            node_major[x] |= h in MAJOR
    name_major = street.groupby(street.name.fillna(""))["highway"].agg(lambda s: bool(set(s) & MAJOR)).to_dict()

    trips, places = [], {}
    for od in blocked.itertuples():
        found = None
        for label, rq in LOOSE:
            rs = net.route_set(int(od.origin), int(od.dest), rq, Preferences())
            if rs.status != "ok":
                continue
            best = None
            for route in rs.routes:
                items = {}
                for e in route.edge_ids:
                    if not ok_real[int(e)]:
                        r = E.iloc[int(e)]
                        for p in place_reasons(r, node_streets, node_major, name_major):
                            items[p[0]] = p + (bool(r.outside_region),)
                if best is None or len(items) < len(best):
                    best = items
            found = (label, best)
            break
        if found is None:
            trips.append({"od": od.od, "band": od.band, "opened_by": "none (no step-free route at all)"})
            continue
        label, items = found
        kinds = Counter(p[1] for p in items.values())
        trips.append({"od": od.od, "band": od.band, "opened_by": label, "n_places": len(items),
                      "reasons": "; ".join(f"{k} x{n}" for k, n in kinds.most_common()),
                      "only_unknowns": all("unknown" in p[1] for p in items.values()),
                      "any_major_road": any(p[3] for p in items.values()),
                      "any_outside_line": any(p[4] for p in items.values())})
        for key, p in items.items():
            d = places.setdefault(key, {"reason": p[1], "detail": p[2], "major_road": p[3],
                                        "outside_line": p[4], "trips": set()})
            d["trips"].add(int(od.od))
    T = pd.DataFrame(trips)

    # where each place is, and how far the nearest ramp-inventory record is from the curb itself
    groups, end_a, end_b = _crossing_outer_ends(E)
    ramps = gpd.read_parquet(RAW_DIR / f"{REGION_ID}_ramps.parquet").to_crs(E.crs) \
        if (RAW_DIR / f"{REGION_ID}_ramps.parquet").exists() else None
    tree = shapely.STRtree(ramps.geometry.values) if ramps is not None else None
    eidx = pd.Series(np.arange(len(E)), index=E.edge_id.to_numpy())
    nidx = net.nodes.set_index("node_id")
    rows = []
    for k, v in places.items():
        kind, ident = k.split(":")[:2]
        if kind == "corner":
            g = groups.iloc[eidx[int(ident)]]
            pt = (end_a if k.endswith("kerb_a") else end_b).loc[g]
        elif kind == "junction":
            pt = nidx.geometry.loc[int(ident)]
        else:
            pt = E.geometry.iloc[eidx[int(ident)]].interpolate(0.5, normalized=True)
        d = round(pt.distance(ramps.geometry.values[tree.nearest(pt)]), 1) if tree is not None else None
        rows.append({"place": k, **{x: v[x] for x in ("reason", "detail", "major_road", "outside_line")},
                     "nearest_ramp_record_m": d, "n_trips": len(v["trips"]),
                     "trips": ",".join(map(str, sorted(v["trips"]))), "geometry": pt})
    P = gpd.GeoDataFrame(rows, geometry="geometry", crs=E.crs).to_crs(ANALYSIS_CRS)
    tracts = gpd.read_parquet(RAW_DIR / f"{REGION_ID}_tracts.parquet").to_crs(ANALYSIS_CRS)[["GEOID", "geometry"]]
    P = gpd.sjoin(P, tracts, how="left", predicate="within").drop(columns="index_right")
    ll = P.geometry.to_crs(4326)
    P["lon"], P["lat"] = ll.x.round(6), ll.y.round(6)
    P = P.drop(columns="geometry").sort_values("n_trips", ascending=False)

    unknown = set(P.place[P.reason.str.contains("unknown")])
    need = defaultdict(set)
    for k, ts in zip(P.place, P.trips):
        for t in ts.split(","):
            need[int(t)].add(k)
    need = {t: ks for t, ks in need.items() if ks <= unknown}
    order = greedy(need)
    n_total, n_ok = len(sf), int((sf.status == "ok").sum())
    short = int(np.ceil(BAR_TRIPS_PCT / 100 * n_total)) - n_ok
    upto = next((i + 1 for i, (_, o) in enumerate(order) if o >= short), None)
    W = P.set_index("place").loc[[k for k, _ in order[:upto or len(order)]]].reset_index()
    W["trips_opened_so_far"] = [o for _, o in order[:len(W)]]

    T.to_csv(out_dir / "blocked_trips.csv", index=False)
    P.to_csv(out_dir / "blocking_places.csv", index=False)
    W.to_csv(out_dir / "worklist.csv", index=False)

    curbs = P[P.reason.str.startswith("curb unknown")]
    no_rec = curbs.nearest_ramp_record_m.gt(50)
    tract = (curbs[no_rec & ~curbs.outside_line].groupby("GEOID")
             .agg(places=("place", "size"), trip_links=("n_trips", "sum"),
                  streets=("detail", lambda s: ", ".join(pd.Series(s.dropna().unique()).head(4))))
             .sort_values("trip_links", ascending=False))
    summary = {
        "trips": n_total, "with_route": n_ok, "blocked": len(blocked), "short_of_bar": short,
        "opened_by": T.opened_by.value_counts().to_dict(),
        "blocked_only_by_unknowns": int(T.only_unknowns.fillna(False).sum()),
        "touching_major_road": int(T.any_major_road.fillna(False).sum()),
        "touching_outside_line": int(T.any_outside_line.fillna(False).sum()),
        "places_by_reason": P.reason.value_counts().to_dict(),
        "unknown_curbs_inside_line": int((~curbs.outside_line).sum()),
        "unknown_curbs_inside_line_no_ramp_record_within_50m": int((no_rec & ~curbs.outside_line).sum()),
        "places_to_close_gap": upto, "worklist_inside_line": int((~W.outside_line).sum()) if len(W) else 0,
        "greedy": {n: o for n, (_, o) in enumerate(order, 1) if n in (10, 20, 30, 40, 50, 75, 100, len(order))},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")

    pct = lambda n: f"{n / n_total * 100:.1f}%"
    md = [
        f"# What blocks manual-wheelchair trips: {REGION_ID}",
        "",
        "**Internal use only** while this region uses unlicensed data (AGENTS.md rule 7). Generated by "
        "`tools/blocked_trips/blocked_trips.py` from `pf crux`'s trip sample; do not hand-edit.",
        "",
        f"Trips with a manual-wheelchair route (step-free, 8.33%, allow likely): **{n_ok} of {n_total} "
        f"({pct(n_ok)})**. Criterion 4 needs {BAR_TRIPS_PCT:.0f}%: {max(short, 0)} more trips.",
        "",
        "## What would open the blocked trips",
        "",
        "| Opened by | Trips |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in summary["opened_by"].items()],
        "",
        f"- Blocked only by places with no record (opened if every one turned out passable): "
        f"**{summary['blocked_only_by_unknowns']}**.",
        f"- Touching a trunk or primary road (mostly state roads): {summary['touching_major_road']}.",
        f"- Touching a place outside the city line: {summary['touching_outside_line']}.",
        "",
        "## Blocking places, by reason",
        "",
        "| Reason | Places |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in summary["places_by_reason"].items()],
        "",
        f"Inside the city line, {summary['unknown_curbs_inside_line']} blocking curbs have no record; "
        f"{summary['unknown_curbs_inside_line_no_ramp_record_within_50m']} of them have no ramp-inventory "
        "record within 50 m, so the inventory does not reach them. By census tract:",
        "",
        "| Tract | Places | Trip links | Streets |", "|---|---|---|---|",
        *[f"| {g} | {r.places} | {r.trip_links} | {r.streets} |" for g, r in tract.head(8).iterrows()],
        "",
        "## Worklist",
        "",
        (f"The **{upto} places** in `worklist.csv` would open the {max(short, 0)} trips the bar needs, *if every one "
         f"turned out to have a usable ramp*: an upper bound. {summary['worklist_inside_line']} are inside the city line. "
         if upto else "No set of unknown places alone closes the gap. ")
        + "Measured slopes are never on the list. Opened per places checked: "
        + ", ".join(f"{n}: {o}" for n, o in summary["greedy"].items()) + ".",
    ]
    (out_dir / "blocked.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
