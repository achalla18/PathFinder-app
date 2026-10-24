"""The demo: a landing page to choose a place, and each place's navigation app behind it.

A server serves one region (``PF_REGION`` is read once, at import), so ``pf demo`` starts one
``pf serve`` per place on its own port and serves the landing page itself. Every place shown is
built from open data only (rule 7): DC, and the Maryland towns' public builds.

    pf demo            # http://127.0.0.1:8000, then pick a place

Local only: it is for recording the demo video, not for deployment.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass

import requests

from .api import PASSWORD_ENV
from .config import REPO_ROOT
from .regions import REGIONS, is_internal

LANDING_DIR = REPO_ROOT / "web" / "landing"
HOST = "127.0.0.1"
LANDING_PORT = 8000


@dataclass(frozen=True)
class Place:
    region: str
    name: str
    port: int
    md6: bool          # in Maryland's 6th congressional district
    blurb: str
    blurb_es: str

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.port}/"


PLACES = (
    Place("dc", "Washington, DC", 8001, False,
          "A city that publishes its sidewalks, curb ramps, signals and terrain as open data.",
          "Una ciudad que publica sus aceras, rampas, semáforos y terreno como datos abiertos."),
    Place("gaithersburg_public", "Gaithersburg", 8002, True,
          "Montgomery County. Built from OpenStreetMap and public terrain data.",
          "Condado de Montgomery. Hecho con OpenStreetMap y datos públicos del terreno."),
    Place("frederick_public", "Frederick", 8003, True,
          "Frederick County. Built from OpenStreetMap and public terrain data.",
          "Condado de Frederick. Hecho con OpenStreetMap y datos públicos del terreno."),
)

# coverage.md's accessibility-bar rows: "| 1. Curb evidence ... | 17.1 | 60.0 | misses | ..."
_ROW = re.compile(r"^\|\s*(\d)\.\s[^|]*\|\s*([\d.]+)\s*\|", re.M)


def evidence(region_id: str) -> dict[str, float | None]:
    """Share of curbs, slope and sidewalks with evidence, from the region's coverage report."""
    path = REPO_ROOT / "reports" / region_id / "phase0" / "coverage.md"
    rows = dict(_ROW.findall(path.read_text(encoding="utf-8"))) if path.exists() else {}
    val = lambda k: float(rows[k]) if k in rows else None  # noqa: E731
    return {"curbs": val("1"), "slope": val("2"), "sidewalks": val("3")}


def places_json() -> list[dict]:
    return [{"id": p.region, "name": p.name, "url": p.url, "md6": p.md6, "blurb": p.blurb, "blurb_es": p.blurb_es,
             "image": f"img/{p.region}.png", "evidence": evidence(p.region)}
            for p in PLACES]


def landing_app():
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="PathFinder demo", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def fresh(request, call_next):
        # the page is edited between takes: never let the browser show a stale copy
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/places")
    def places() -> list[dict]:
        return places_json()

    app.mount("/", StaticFiles(directory=LANDING_DIR, html=True), name="landing")
    return app


def _check_open() -> None:
    """Refuse to start if any place would show data held back under rule 7."""
    for p in PLACES:
        held = [s.id for s in REGIONS[p.region].sources.values() if is_internal(s)]
        if held:
            raise SystemExit(f"{p.region} holds internal sources {held}; the demo shows open data only")


def _wait(url: str, procs: list[subprocess.Popen], timeout_s: float = 120) -> None:
    # The servers inherit PF_APP_PASSWORD (from the environment or the repo's .env) and then ask
    # every request for it, health checks included; any user name is accepted.
    password = os.environ.get(PASSWORD_ENV)
    auth = ("pathfinder", password) if password else None
    end = time.time() + timeout_s
    while time.time() < end:
        if any(p.poll() is not None for p in procs):
            raise SystemExit("a place's server stopped while starting; see its log above")
        try:
            r = requests.get(url + "api/health", timeout=2, auth=auth)
            if r.status_code < 400:
                return
            if r.status_code == 401:
                raise SystemExit(f"{url} refused the demo's password ({PASSWORD_ENV}); check it")
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise SystemExit(f"{url} did not answer within {timeout_s:.0f} s")


def warm_examples(places=PLACES) -> int:
    """Plan each place's example trips once, as the page does when one is clicked, so that the
    server's answer is ready (api.PLAN_CACHE_SIZE) and nothing stalls on camera. Returns how many
    were planned."""
    password = os.environ.get(PASSWORD_ENV)
    auth = ("pathfinder", password) if password else None
    n = 0
    for p in places:
        path = REPO_ROOT / "demo" / f"{p.region}.json"
        if not path.exists():
            continue
        for e in json.loads(path.read_text(encoding="utf-8")):
            body = {"origin": {"lon": e["from"][0], "lat": e["from"][1]}, "destination": {"lon": e["to"][0], "lat": e["to"][1]}}
            body.update(_page_settings(p.url, e, auth))
            try:
                requests.post(p.url + "api/plan", json=body, auth=auth, timeout=120).raise_for_status()
                n += 1
            except requests.RequestException as err:
                print(f"  could not warm {p.name}: {e['label']} ({err})", flush=True)
    return n


def _page_settings(url: str, example: dict, auth) -> dict:
    """The settings the page sends for an example: its profile's requirements and mix, with the
    example's evidence level (App.tsx, the example buttons)."""
    profiles = requests.get(url + "api/region", auth=auth, timeout=30).json().get("profiles", [])
    prof = next((x for x in profiles if x["id"] == example.get("profile")), None)
    if prof is None:
        return {}
    r, mix = prof["requirements"], prof["mix"]
    return {"step_free": r["step_free"], "max_grade_pct": r["max_grade_pct"],
            "certainty": example.get("certainty") or r["certainty"], "budget_min": 5,
            "priorities": {"cross": mix["cross"], "slope": mix["slope"], "sidewalk": mix["sidewalk"]}}


def run(open_browser: bool = True) -> None:
    import uvicorn

    _check_open()
    home = f"http://{HOST}:{LANDING_PORT}/"
    procs = []
    try:
        for p in PLACES:
            env = {**os.environ, "PF_REGION": p.region, "PF_HOME": home,
                   "PF_PLACES": json.dumps([{"name": q.name, "url": q.url} for q in PLACES])}
            procs.append(subprocess.Popen([sys.executable, "-m", "pathfinder.cli", "serve", "--port", str(p.port)], env=env))
        for p in PLACES:
            _wait(p.url, procs)
            print(f"  {p.name}: {p.url}", flush=True)
        print(f"PathFinder demo: {home}  (Ctrl-C stops everything)", flush=True)
        threading.Thread(target=lambda: print(f"  example trips ready: {warm_examples()} planned", flush=True),
                         daemon=True).start()
        if open_browser:
            threading.Timer(1.0, webbrowser.open, args=(home,)).start()
        uvicorn.run(landing_app(), host=HOST, port=LANDING_PORT, log_level="warning")
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
