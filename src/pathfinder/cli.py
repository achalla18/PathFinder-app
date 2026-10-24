"""Command-line entry point: ``pf <command>``."""

from __future__ import annotations

import argparse
import logging


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pf", description="PathFinder data pipeline")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download registered sources into the raw data dir")
    f.add_argument("sources", nargs="*", help="source ids (default: all)")
    f.add_argument("--force", action="store_true", help="re-download even if present")

    sub.add_parser("sources", help="list registered sources")
    sub.add_parser("extract-osm", help="extract pedestrian-relevant OSM ways and nodes")
    sub.add_parser("build-graph", help="build the citywide pedestrian graph topology")
    sub.add_parser("enrich", help="attach evidence-graded attributes to graph edges")
    sub.add_parser("coverage", help="write the Phase 0 data coverage report")
    sub.add_parser("barriers", help="write the city barrier report -> reports/<region>/barriers/")
    sub.add_parser("crux-report", help="regenerate the crux report from saved results")
    c = sub.add_parser("crux", help="run the Phase 0 crux-hypothesis experiment")
    c.add_argument("--per-bin", type=int, default=334, help="OD pairs per distance band")
    c.add_argument("--workers", type=int, default=None)
    c.add_argument("--fresh", action="store_true", help="discard the checkpoint and place the fixed sample on the current graph")
    c.add_argument("--resample", action="store_true", help="draw a new sample (earlier runs stop being comparable)")
    w = sub.add_parser("walks", help="read walk recordings saved from the page; suggests off-route and alert distances")
    w.add_argument("files", nargs="+", help="recording files (walk-*.json)")
    s = sub.add_parser("serve", help="run the web app's API (and the built web app, if present)")
    s.add_argument("--host", default="127.0.0.1", help="non-local hosts need PF_APP_PASSWORD set")
    s.add_argument("--port", type=int, default=8000)
    d = sub.add_parser("demo", help="landing page to choose DC, Gaithersburg or Frederick, with each place's app (local)")
    d.add_argument("--no-browser", action="store_true", help="do not open the browser")
    r = sub.add_parser("route", help="plan and explain one trip; prints JSON")
    # lat,lon (as copied from a map): DC longitudes are negative and would parse as options
    r.add_argument("origin", help="lat,lon")
    r.add_argument("dest", help="lat,lon")
    r.add_argument("--profile", default=None, help="mobility profile: manual_wheelchair, power_wheelchair, "
                   "walker_cane, stroller, custom (sets requirements and priorities; flags below override)")
    r.add_argument("--step-free", action="store_const", const=True, default=None, help="require a step-free route")
    r.add_argument("--max-grade", type=float, default=None, help="max sustained grade in %% (e.g. 8.33)")
    r.add_argument("--certainty", choices=["verified_only", "allow_likely", "allow_unknown"], default=None)
    r.add_argument("--weights", default=None, help="priorities as cross,slope,sidewalk (any positive scale)")
    r.add_argument("--budget-min", type=float, default=5.0, help="max extra real minutes over the fastest route")
    r.add_argument("--no-geometry", action="store_true", help="omit route coordinates from the output")

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.cmd == "fetch":
        from .fetch import fetch

        fetch(args.sources or None, force=args.force)
    elif args.cmd == "sources":
        from .sources import REGION, SOURCES

        print(f"region: {REGION.id} ({REGION.name}); set PF_REGION to change")
        for s in SOURCES.values():
            print(f"{s.id:28s} {s.kind:16s} {s.title}")
    elif args.cmd == "extract-osm":
        from .osm import extract
        from .places import build as build_places

        extract()
        build_places()
    elif args.cmd == "build-graph":
        from .graph.build import build

        build()
    elif args.cmd == "enrich":
        from .config import GRAPH_DIR
        from .graph.enrich import enrich

        enrich().to_parquet(GRAPH_DIR / "edges.parquet")
    elif args.cmd == "barriers":
        from .analysis.barriers import run as barriers_run

        barriers_run()
    elif args.cmd == "coverage":
        from .analysis.coverage import run as coverage_run

        coverage_run()
    elif args.cmd == "crux-report":
        from .analysis.crux import write_report

        write_report()
    elif args.cmd == "crux":
        import json

        from .analysis.crux import run

        print(json.dumps(run(args.per_bin, args.workers, args.fresh, args.resample), indent=2))
    elif args.cmd == "walks":
        from .analysis.walks import report

        print(report(args.files))
    elif args.cmd == "serve":
        from .api import serve

        serve(args.host, args.port)
    elif args.cmd == "demo":
        from .demo import run

        run(open_browser=not args.no_browser)
    elif args.cmd == "route":
        import json

        from .routing.cost import Preferences
        from .routing.profiles import resolve as resolve_profile
        from .routing.service import TripPlanner

        def lonlat(s: str) -> tuple[float, float]:
            lat, lon = (float(v) for v in s.split(","))
            return lon, lat

        overrides = {k: v for k, v in (("step_free", args.step_free), ("max_grade_pct", args.max_grade),
                                       ("certainty", args.certainty)) if v is not None}
        mix = tuple(float(v) for v in args.weights.split(",")) if args.weights else None
        req, mix = resolve_profile(args.profile, overrides, mix)
        pref = Preferences(w_cross=mix[0], w_slope=mix[1], w_sidewalk=mix[2], budget_s=args.budget_min * 60)
        result = TripPlanner().plan(lonlat(args.origin), lonlat(args.dest), req, pref)
        if args.no_geometry:
            def strip(o):
                if isinstance(o, dict):
                    return {k: strip(v) for k, v in o.items() if k != "geometry"}
                return [strip(v) for v in o] if isinstance(o, list) else o
            result = strip(result)
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
