"""HTTP API for the web app (Phase 1): one endpoint that plans and explains a trip.

Deployment stays private. The server binds to this machine only unless a password is set
(``PF_APP_PASSWORD``); with a password every request needs it (HTTP Basic, any user name).
Nothing about a plan or search request is stored: the caller's location arrives only as
trip endpoints and is gone when the response is sent. The one thing kept is what a tester
chooses to send through the feedback form: their answers and comment, with the day, and no
location, address or device details.
"""

from __future__ import annotations

import csv
import json
import mimetypes
import os
import secrets
import threading
from collections import OrderedDict
from datetime import date
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Response, status
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from .basemap import Basemap
from .config import DATA_DIR, REPO_ROOT
from .places import PlaceIndex
from .routing.cost import PENALTY_VERSION, Preferences
from .analysis.barriers import _unlicensed_sources as unlicensed_sources
from .routing.profiles import PROFILES
from .routing.profiles import as_json as profiles_json
from .routing.profiles import resolve as resolve_profile
from .routing.service import TripPlanner
from .sources import REGION, SOURCES
from .regions import is_internal
from .wmata import ElevatorNotices

PASSWORD_ENV = "PF_APP_PASSWORD"
WEB_DIST = REPO_ROOT / "web" / "dist"
MAX_BUDGET_MIN = 30.0
FEEDBACK_FILE = DATA_DIR / "feedback" / "feedback.jsonl"
# Tiles and glyphs only change when they are fetched again; a day spares a phone re-asking.
BASEMAP_CACHE = "private, max-age=86400"
# One map view at a time; a town has a few thousand barrier records in all.
BARRIERS_MAX_FEATURES = 5000
# The page, its service worker and its manifest name the hashed asset files: a phone must check
# them each time, or it would keep running an old build. (The hashed files themselves are kept for good.)
REVALIDATE_PATHS = {"/", "/index.html", "/sw.js", "/manifest.webmanifest"}
mimetypes.add_type("application/manifest+json", ".webmanifest")
_feedback_lock = threading.Lock()
# Recent plans, by trip and settings. The network does not change while a server runs, so the
# same request has the same answer; a demo's example trips (warmed by `pf demo`) come back at once
# instead of re-running a search that can take seconds (a "no route" trip tries compromises).
PLAN_CACHE_SIZE = 64
_basic = HTTPBasic(auto_error=False)


class LatLon(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class Priorities(BaseModel):
    """The mix of the three accessibility dimensions (any positive scale)."""
    cross: float = Field(1.0, ge=0)
    slope: float = Field(1.0, ge=0)
    sidewalk: float = Field(1.0, ge=0)

    @model_validator(mode="after")
    def _not_all_zero(self) -> "Priorities":
        if self.cross + self.slope + self.sidewalk <= 0:
            raise ValueError("at least one priority must be above zero")
        return self


class PlanRequest(BaseModel):
    """A trip. ``profile`` fills the requirements and priorities; any of them sent explicitly
    overrides the profile's."""
    origin: LatLon
    destination: LatLon
    profile: Literal[tuple(PROFILES)] | None = None  # type: ignore[valid-type]
    step_free: bool = False
    max_grade_pct: float | None = Field(None, gt=0, le=30)
    certainty: Literal["verified_only", "allow_likely", "allow_unknown"] = "allow_likely"
    priorities: Priorities = Priorities()
    budget_min: float = Field(5.0, ge=0, le=MAX_BUDGET_MIN)


Answer = Literal["yes", "partly", "no"]


class Feedback(BaseModel):
    """What a tester tells us. Every question is optional; an empty form is refused."""
    felt_minutes_clear: Answer | None = None  # did "felt minutes" make sense?
    places_clear: Answer | None = None  # were the places where routes differ understandable?
    better_than_route_list: Answer | None = None  # preferred over a plain list of routes?
    walked: bool = False  # sent after walking a route with guidance, not only planning one
    comment: str = Field("", max_length=2000)

    @model_validator(mode="after")
    def _not_empty(self) -> "Feedback":
        if not (self.felt_minutes_clear or self.places_clear or self.better_than_route_list or self.comment.strip()):
            raise ValueError("nothing to send")
        return self


# What someone can report from the map. Curb reports become corners on the team's field sheet
# (tools/field_checks/add_reports.py); the rest become notes to look at. A report never changes
# a route: only a team member's visit does.
REPORT_KINDS = ("broken_ramp", "no_ramp", "blocked_sidewalk", "no_sidewalk", "too_steep", "other")
REPORTS_FILE = DATA_DIR / "reports" / "barrier_reports.csv"
REPORT_COLUMNS = ("day", "kind", "lat", "lon", "note")


class BarrierReport(BaseModel):
    """A barrier someone saw: where, what, and an optional note. Nothing about who sent it."""
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    kind: Literal[REPORT_KINDS]  # type: ignore[valid-type]
    note: str = Field("", max_length=300)


def _round_repr(obj) -> str:
    """A settings object as a cache key: its fields, with floats rounded (a mix sent as 0.3333333
    by the page and computed as 1/3 here is the same request)."""
    import dataclasses

    fields = dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else vars(obj)
    return repr(sorted((k, round(v, 4) if isinstance(v, float) else v) for k, v in fields.items()))


def _require_password(credentials: HTTPBasicCredentials | None = Depends(_basic)) -> None:
    expected = os.environ.get(PASSWORD_ENV)
    if not expected:
        return  # no password configured: the server is only reachable from this machine (see `pf serve`)
    if credentials is None or not secrets.compare_digest(credentials.password.encode(), expected.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "password required", headers={"WWW-Authenticate": "Basic"})


HOME_ENV = "PF_HOME"
PLACES_ENV = "PF_PLACES"  # every place's server, for the page's place menu (set by `pf demo`)
# Example trips the page offers before anything is chosen, one file per region (demo/<region>.json):
# [{"label": ..., "from": [lon, lat, name], "to": [lon, lat, name]}, ...]
EXAMPLES_DIR = REPO_ROOT / "demo"


def _examples() -> list[dict]:
    path = EXAMPLES_DIR / f"{REGION.id}.json"
    if not path.exists():
        return []
    out = []
    for e in json.loads(path.read_text(encoding="utf-8")):
        ends = [e.get("from"), e.get("to")]
        if isinstance(e.get("label"), str) and all(isinstance(x, list) and len(x) == 3 for x in ends):
            extra = {k: e[k] for k in ("profile", "certainty", "label_es", "note_es") if isinstance(e.get(k), str)}
            out.append({"label": e["label"], "note": str(e.get("note", "")), **extra,
                        "from": [float(ends[0][0]), float(ends[0][1]), str(ends[0][2])],
                        "to": [float(ends[1][0]), float(ends[1][1]), str(ends[1][2])]})
    return out


def _places() -> list[dict]:
    """The places the page can switch to directly: [{"name", "url"}] from ``PF_PLACES``. Anything
    that is not a web address is dropped."""
    try:
        items = json.loads(os.environ.get(PLACES_ENV, "") or "[]")
    except ValueError:
        return []
    return [{"name": str(i["name"]), "url": i["url"]} for i in items if isinstance(i, dict) and isinstance(i.get("name"), str)
            and isinstance(i.get("url"), str) and i["url"].startswith(("http://", "https://"))]


def _home() -> str | None:
    """Where "Change place" leads: the page listing every place (``PF_HOME``, set by ``pf demo``).
    Each server serves one region, so changing place opens another server's page."""
    url = os.environ.get(HOME_ENV, "").strip()
    return url if url.startswith(("http://", "https://")) else None


def create_app(planner: TripPlanner | None = None, places: PlaceIndex | None = None,
               basemap: Basemap | None = None, feedback_file=None,
               elevators: ElevatorNotices | None = None) -> FastAPI:
    app = FastAPI(title="PathFinder", docs_url=None, redoc_url=None, dependencies=[Depends(_require_password)])
    # a plan with thirty routes is about 2 MB of coordinates as JSON and a tenth of that compressed
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    state: dict = {"planner": planner, "places": places, "barriers": None, "sidewalks": None}
    if basemap is None and Basemap.available():
        basemap = Basemap()
    feedback_path = feedback_file or FEEDBACK_FILE
    if elevators is None:
        elevators = ElevatorNotices.from_environment()

    app.state.dd = state
    plans: OrderedDict = OrderedDict()
    plan_lock = threading.Lock()

    def get_planner() -> TripPlanner:
        if state["planner"] is None:  # loaded on first use so the process starts fast
            state["planner"] = TripPlanner()
        return state["planner"]

    @app.get("/api/region")
    def region() -> dict:
        """Where the page opens and what this region can say, so one page serves any region."""
        return {"id": REGION.id, "name": REGION.name, "center": list(REGION.center), "bbox": list(REGION.bbox),
                "profiles": profiles_json(), "transit_notices": elevators is not None,
                "sources": {sid: s.title for sid, s in SOURCES.items()},
                "public": not any(is_internal(s) for s in SOURCES.values()), "home": _home(), "places": _places(), "examples": _examples()}

    @app.get("/api/health")
    def health() -> dict:
        # the page draws its backdrop from here when the tiles have been fetched
        return {"status": "ok", "basemap": basemap is not None, "basemap_max_zoom": basemap.max_zoom if basemap else None,
                "center": list(REGION.center)}

    @app.get("/tiles/{z}/{x}/{y}.mvt")
    def tile(z: int, x: int, y: int) -> Response:
        data = basemap.tile(z, x, y) if basemap else None
        if data is None:
            return Response(status_code=204)
        headers = {"Cache-Control": BASEMAP_CACHE}
        if basemap.gzipped:
            headers["Content-Encoding"] = "gzip"
        return Response(data, media_type="application/x-protobuf", headers=headers)

    @app.get("/fonts/{stack}/{name}")
    def glyphs(stack: str, name: str) -> Response:
        if basemap is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        try:
            data = basemap.glyphs(stack, name)
        except ValueError:
            raise HTTPException(status.HTTP_404_NOT_FOUND) from None
        return Response(data, media_type="application/x-protobuf", headers={"Cache-Control": BASEMAP_CACHE})

    @app.post("/api/feedback", status_code=status.HTTP_201_CREATED)
    def feedback(body: Feedback) -> dict:
        """Keep a tester's answers. Only what they typed and ticked is written, with the day."""
        row = {"day": date.today().isoformat(), "penalty_version": PENALTY_VERSION, **body.model_dump()}
        with _feedback_lock:
            feedback_path.parent.mkdir(parents=True, exist_ok=True)
            with feedback_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {"status": "saved"}

    @app.post("/api/report", status_code=status.HTTP_201_CREATED)
    def report(body: BarrierReport) -> dict:
        """Keep a barrier report on this computer for the team to check on site. It is not sent
        anywhere and changes no route; inside the region's box only."""
        xmin, ymin, xmax, ymax = REGION.bbox
        if not (xmin <= body.lon <= xmax and ymin <= body.lat <= ymax):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"that point is outside {REGION.name}")
        row = {"day": date.today().isoformat(), "kind": body.kind, "lat": round(body.lat, 6), "lon": round(body.lon, 6),
               "note": " ".join(body.note.split())}
        with _feedback_lock:
            REPORTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            new = not REPORTS_FILE.exists()
            with REPORTS_FILE.open("a", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
                if new:
                    w.writeheader()
                w.writerow(row)
        return {"status": "saved"}

    @app.get("/api/barriers")
    def barrier_map(bbox: str = Query(pattern=r"^-?[\d.]+,-?[\d.]+,-?[\d.]+,-?[\d.]+$"),
                    limit: int = Query(BARRIERS_MAX_FEATURES, ge=1, le=BARRIERS_MAX_FEATURES)) -> dict:
        """Recorded barriers and unrecorded places inside ``bbox`` (minlon,minlat,maxlon,maxlat),
        each with its source and date, from the same graph the planner routes on."""
        from .analysis.barriers import as_geojson, barriers

        xmin, ymin, xmax, ymax = (float(v) for v in bbox.split(","))
        if not (xmin < xmax and ymin < ymax):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "bbox must be minlon,minlat,maxlon,maxlat")
        if state["barriers"] is None:
            import geopandas as gpd

            edges = gpd.read_parquet(get_planner().graph_dir / "edges.parquet")
            state["barriers"] = barriers(edges).to_crs("EPSG:4326")
            from .analysis.barriers import estimated_sidewalks
            from .graph.enrich import SURVEY_SIDEWALK_LINES

            path = get_planner().graph_dir / SURVEY_SIDEWALK_LINES
            lines = gpd.read_parquet(path) if path.exists() else None
            state["sidewalks"] = (estimated_sidewalks(lines, edges).to_crs("EPSG:4326")
                                  if lines is not None and len(lines) else None)
        b = state["barriers"]
        inside = b.cx[xmin:xmax, ymin:ymax]
        sw = state["sidewalks"]
        sw_inside = sw.cx[xmin:xmax, ymin:ymax] if sw is not None else None
        # A capped answer keeps every barrier and spreads what is left over the unknown types
        # in turn, so no kind of gap disappears from the map; the counts are always complete.
        turn = inside.groupby("type").cumcount()
        inside = inside.assign(_turn=turn.where(inside["kind"] == "unknown", -1)).sort_values("_turn", kind="stable")
        out = as_geojson(inside.drop(columns="_turn").head(limit))
        # estimated sidewalk lines travel beside the barriers, never in their place
        sidewalks = as_geojson(sw_inside.head(limit)) if sw_inside is not None and len(sw_inside) else \
            {"type": "FeatureCollection", "features": []}
        return {**out, "truncated": len(inside) > limit, "counts": inside["type"].value_counts().to_dict(),
                "sidewalks": {**sidewalks, "count": 0 if sw_inside is None else len(sw_inside),
                              "truncated": sw_inside is not None and len(sw_inside) > limit},
                "unlicensed_sources": unlicensed_sources()}

    @app.get("/api/search")
    def search(q: str = Query(min_length=2, max_length=120), lat: float | None = Query(None, ge=-90, le=90),
               lon: float | None = Query(None, ge=-180, le=180)) -> dict:
        """Find places and addresses by name. Answered from the local index; the query is
        not logged or passed on."""
        if state["places"] is None:
            state["places"] = PlaceIndex()
        near = (lon, lat) if lat is not None and lon is not None else None
        return {"results": state["places"].search(q, near)}

    @app.post("/api/plan")
    def plan(body: PlanRequest) -> dict:
        chosen = body.model_fields_set
        overrides = {k: getattr(body, k) for k in ("step_free", "max_grade_pct", "certainty") if k in chosen}
        p = body.priorities
        mix = (p.cross, p.slope, p.sidewalk) if "priorities" in chosen else None
        req, mix = resolve_profile(body.profile, overrides, mix)
        pref = Preferences(w_cross=mix[0], w_slope=mix[1], w_sidewalk=mix[2], budget_s=body.budget_min * 60)
        origin, destination = (body.origin.lon, body.origin.lat), (body.destination.lon, body.destination.lat)
        key = (tuple(round(v, 6) for v in (*origin, *destination)), _round_repr(req), _round_repr(pref))
        with plan_lock:
            out = plans.get(key)
            if out is not None:
                plans.move_to_end(key)
        if out is None:
            out = get_planner().plan(origin, destination, req, pref)
            with plan_lock:
                plans[key] = out
                while len(plans) > PLAN_CACHE_SIZE:
                    plans.popitem(last=False)
        out = dict(out)  # the cached answer stays as planned; live notices are added to a copy
        if elevators is not None:
            # Metro elevators out of service at a station the trip starts or ends at. A notice
            # only: no route depends on them (see wmata.py).
            out["transit_notices"] = elevators.for_trip(origin, destination)
        return out

    @app.middleware("http")
    async def cache_built_assets(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/assets/") and response.status_code == 200:
            # file names carry a hash of their content, so a copy can be kept for good
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        elif request.url.path in REVALIDATE_PATHS:
            response.headers["Cache-Control"] = "no-cache"
        return response

    if WEB_DIST.exists():  # the built web app, when present (`npm run build` in web/)
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
    return app


def warm_up(app_state: dict) -> None:
    """Load the network and the place index now, so the first person to ask does not wait for them."""
    if app_state["planner"] is None:
        app_state["planner"] = TripPlanner()
    if app_state["places"] is None:
        app_state["places"] = PlaceIndex()


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    if host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get(PASSWORD_ENV):
        raise SystemExit(f"Refusing to listen on {host} without a password: set {PASSWORD_ENV} first. "
                         "Deployment stays private unless the owner authorizes public access.")
    app = create_app()
    threading.Thread(target=warm_up, args=(app.state.dd,), daemon=True).start()
    uvicorn.run(app, host=host, port=port)
