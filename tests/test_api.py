"""The HTTP API: validation, the password gate, and the planner's answer passed through."""

import pytest
from fastapi.testclient import TestClient

from pathfinder import api

from .test_service import _lonlat, planner  # noqa: F401  (fixture)


def _body(o, d, **kw):
    return {"origin": {"lon": o[0], "lat": o[1]}, "destination": {"lon": d[0], "lat": d[1]}, **kw}


@pytest.fixture
def client(planner, monkeypatch):  # noqa: F811
    monkeypatch.delenv(api.PASSWORD_ENV, raising=False)
    return TestClient(api.create_app(planner))


def test_plan_carries_elevator_outages_at_a_station_the_trip_ends_at(planner, monkeypatch):  # noqa: F811
    import pandas as pd

    from pathfinder.wmata import ElevatorNotices, OutageFeed, StationIndex

    monkeypatch.delenv(api.PASSWORD_ENV, raising=False)
    o, d = _lonlat(0, 0), _lonlat(215, 0)
    entrances = pd.DataFrame([{"entrance_id": "1", "name": "east", "station_code_1": "X01", "station_code_2": "Y01", "lon": d[0], "lat": d[1]}])
    stations = pd.DataFrame([{"code": "X01", "name": "Test Station", "lon": d[0], "lat": d[1]}])
    feed = [{"UnitType": "ELEVATOR", "StationCode": "Y01", "StationName": "Test Station", "UnitName": "Y01X01",
             "LocationDescription": "Elevator between street and mezzanine", "SymptomDescription": "Service Call",
             "DateOutOfServ": "2026-10-06T08:00:00", "EstimatedReturnToService": None},
            {"UnitType": "ESCALATOR", "StationCode": "X01", "UnitName": "X01E01"},
            {"UnitType": "ELEVATOR", "StationCode": "Z09", "UnitName": "Z09X01"}]
    calls = []

    def get():
        calls.append(1)
        if len(calls) > 1:
            raise OSError("down")
        return feed

    notices = ElevatorNotices(StationIndex(entrances, stations), OutageFeed("k", get))
    c = TestClient(api.create_app(planner, elevators=notices))
    (n,) = c.post("/api/plan", json=_body(o, d)).json()["transit_notices"]
    # the other code of the same station counts; escalators and other stations do not
    assert n["end"] == "destination" and n["unit"] == "Y01X01" and n["status"] == "out_of_service"
    assert n["source"] and n["as_of"]
    assert c.post("/api/plan", json=_body(o, d)).json()["transit_notices"] == [n] and len(calls) == 1  # not re-read within a minute
    # a feed that cannot be read is "unknown", never silence
    notices.feed._at = -1e9
    (u,) = c.post("/api/plan", json=_body(o, d)).json()["transit_notices"]
    assert u["status"] == "unknown" and u["station"] == "Test Station"
    # ...and so is a reply of the wrong shape: the routes still come back
    for bad in ({"Message": "quota exceeded"}, [None], [{"UnitType": "ELEVATOR"}, "x"]):
        notices.feed._get = lambda bad=bad: bad
        notices.feed._at = -1e9
        r = c.post("/api/plan", json=_body(o, d))
        assert r.status_code == 200 and r.json()["status"] == "ok"
        assert [x["status"] for x in r.json()["transit_notices"]] == ["unknown"]
    # with no elevator source configured the response says nothing about elevators
    monkeypatch.setattr(ElevatorNotices, "from_environment", classmethod(lambda cls: None))
    assert "transit_notices" not in TestClient(api.create_app(planner)).post("/api/plan", json=_body(o, d)).json()


def test_fetching_everything_asks_for_every_stored_source(monkeypatch):
    from pathfinder import fetch
    from pathfinder.sources import SOURCES

    asked = []
    monkeypatch.delenv("WMATA_API_KEY", raising=False)
    monkeypatch.setattr(fetch, "ensure_dirs", lambda: None)
    monkeypatch.setattr(fetch, "load_manifest", lambda: {})
    monkeypatch.setattr(fetch, "_FETCHERS", {k: (lambda s: asked.append(s.id)) for k in fetch._FETCHERS})
    fetch.fetch()
    assert set(asked) == {i for i, s in SOURCES.items() if s.kind != "live"}


def test_feedback_keeps_only_what_the_tester_sent(planner, monkeypatch, tmp_path):  # noqa: F811
    import json

    monkeypatch.delenv(api.PASSWORD_ENV, raising=False)
    out = tmp_path / "feedback.jsonl"
    c = TestClient(api.create_app(planner, feedback_file=out))
    assert c.post("/api/feedback", json={}).status_code == 422  # an empty form is not a response
    assert c.post("/api/feedback", json={"felt_minutes_clear": "maybe"}).status_code == 422
    r = c.post("/api/feedback", json={"felt_minutes_clear": "partly", "walked": True, "comment": "the bars confused me"},
               headers={"User-Agent": "a-phone"})
    assert r.status_code == 201
    (row,) = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert set(row) == {"day", "penalty_version", "felt_minutes_clear", "places_clear", "better_than_route_list", "walked", "comment"}
    assert row["felt_minutes_clear"] == "partly" and row["walked"] is True


def test_basemap_tiles_and_glyphs_are_served_locally(planner, monkeypatch, tmp_path):  # noqa: F811
    import gzip

    from pmtiles.tile import Compression, TileType, zxy_to_tileid
    from pmtiles.writer import write

    from pathfinder.basemap import Basemap

    monkeypatch.delenv(api.PASSWORD_ENV, raising=False)
    tiles, fonts = tmp_path / "t.pmtiles", tmp_path / "fonts"
    with write(tiles) as w:
        w.write_tile(zxy_to_tileid(1, 0, 0), gzip.compress(b"tile"))
        w.finalize({"tile_type": TileType.MVT, "tile_compression": Compression.GZIP}, {})
    (fonts / "Noto Sans Regular").mkdir(parents=True)
    (fonts / "Noto Sans Regular" / "0-255.pbf").write_bytes(b"glyphs")
    (tmp_path / "secret.pbf").write_bytes(b"no")
    c = TestClient(api.create_app(planner, basemap=Basemap(tiles, fonts)))
    assert c.get("/api/health").json()["basemap"] is True
    r = c.get("/tiles/1/0/0.mvt")
    assert r.status_code == 200 and r.content == b"tile"  # stored gzipped, sent as such, decoded by the client
    assert c.get("/tiles/1/1/1.mvt").status_code == 204 and c.get("/tiles/9/0/0.mvt").status_code == 204
    assert c.get("/fonts/Noto Sans Regular/0-255.pbf").content == b"glyphs"
    assert c.get("/fonts/Noto Sans Regular/256-511.pbf").content == b""  # not fetched: no glyphs, not an error
    assert c.get("/fonts/Nope/0-255.pbf").status_code == 404
    assert c.get("/fonts/%2E%2E/secret.pbf").status_code == 404


def test_plan_returns_route_options_and_the_priorities_triangle(client):
    r = client.post("/api/plan", json=_body(_lonlat(0, 0), _lonlat(215, 0), priorities={"cross": 1, "slope": 0, "sidewalk": 0},
                                            budget_min=10))
    assert r.status_code == 200
    out = r.json()
    assert out["status"] == "ok"
    options = out["route_options"]
    assert options[0]["is_fastest"] and "compared_with_fastest" not in options[0]
    assert sum(o["is_recommended"] for o in options) == 1
    detour = next(o for o in options if not o["is_fastest"])
    assert detour["compared_with_fastest"]["crux"]["fastest_route"]["conditions"][0]["type"] == "crossing"
    # every grid point of the triangle names an option, and the shares add up
    assert len(out["triangle"]) == 66 and {t[3] for t in out["triangle"]} <= {o["id"] for o in options}
    assert sum(o["priority_region_share"] for o in options) == pytest.approx(1.0, abs=0.02)


def test_plan_rejects_bad_input(client):
    o, d = _lonlat(0, 0), _lonlat(215, 0)
    assert client.post("/api/plan", json=_body(o, d, priorities={"cross": 0, "slope": 0, "sidewalk": 0})).status_code == 422
    assert client.post("/api/plan", json=_body(o, d, certainty="whatever")).status_code == 422
    assert client.post("/api/plan", json=_body(o, d, budget_min=500)).status_code == 422
    assert client.post("/api/plan", json={"origin": {"lat": 999, "lon": 0}, "destination": {"lat": 0, "lon": 0}}).status_code == 422


def test_password_gate(planner, monkeypatch):  # noqa: F811
    monkeypatch.setenv(api.PASSWORD_ENV, "open sesame")
    c = TestClient(api.create_app(planner))
    assert c.get("/api/health").status_code == 401
    assert c.get("/api/health", auth=("anyone", "wrong")).status_code == 401
    assert c.get("/api/health", auth=("anyone", "open sesame")).status_code == 200


def test_server_refuses_to_listen_publicly_without_a_password(monkeypatch):
    monkeypatch.delenv(api.PASSWORD_ENV, raising=False)
    with pytest.raises(SystemExit):
        api.serve(host="0.0.0.0")


def test_large_answers_are_compressed(client):
    r = client.post("/api/plan", json=_body(_lonlat(0, 0), _lonlat(215, 0)), headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200 and r.headers.get("content-encoding") == "gzip"
    assert r.json()["status"] == "ok"  # and still read as the same answer


def test_built_page_is_revalidated_and_hashed_assets_are_kept(planner, monkeypatch, tmp_path):  # noqa: F811
    """The page and its service worker name the hashed files of a build; a phone that kept them
    for good would keep running an old build. The hashed files themselves never change."""
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><title>t</title>")
    (tmp_path / "sw.js").write_text("// sw")
    (tmp_path / "manifest.webmanifest").write_text("{}")
    (tmp_path / "assets" / "index-abc123.js").write_text("// js")
    monkeypatch.setattr(api, "WEB_DIST", tmp_path)
    monkeypatch.delenv(api.PASSWORD_ENV, raising=False)
    c = TestClient(api.create_app(planner))
    for path in ("/", "/sw.js", "/manifest.webmanifest"):
        r = c.get(path)
        assert r.status_code == 200 and r.headers["cache-control"] == "no-cache", path
    assert c.get("/manifest.webmanifest").headers["content-type"].startswith("application/manifest+json")
    assert "immutable" in c.get("/assets/index-abc123.js").headers["cache-control"]


def test_barrier_report_keeps_only_place_kind_note_and_day(planner, monkeypatch, tmp_path):  # noqa: F811
    import csv

    monkeypatch.delenv(api.PASSWORD_ENV, raising=False)
    monkeypatch.setattr(api, "REPORTS_FILE", tmp_path / "reports" / "barrier_reports.csv")
    c = TestClient(api.create_app(planner))
    lon, lat = api.REGION.center
    assert c.post("/api/report", json={"lat": lat, "lon": lon, "kind": "broken_ramp", "note": "  cracked,\n  lip  "},
                  headers={"User-Agent": "a-phone"}).status_code == 201
    assert c.post("/api/report", json={"lat": lat, "lon": lon, "kind": "pothole"}).status_code == 422  # not a kind we take
    assert c.post("/api/report", json={"lat": 0.0, "lon": 0.0, "kind": "other"}).status_code == 422  # outside the region
    (row,) = list(csv.DictReader(api.REPORTS_FILE.open(encoding="utf-8")))
    assert set(row) == {"day", "kind", "lat", "lon", "note"} and row["kind"] == "broken_ramp" and row["note"] == "cracked, lip"


def test_the_same_trip_is_planned_once_and_answered_from_memory_after(planner, monkeypatch):  # noqa: F811
    """pf demo warms each example trip; when the page asks for it again it must not wait for a
    second search. Live notices still go only on the copy that is sent."""
    monkeypatch.delenv(api.PASSWORD_ENV, raising=False)
    calls = []
    real = planner.plan
    monkeypatch.setattr(planner, "plan", lambda *a, **k: calls.append(1) or real(*a, **k))
    c = TestClient(api.create_app(planner))
    o, d = _lonlat(0, 0), _lonlat(215, 0)
    first = c.post("/api/plan", json=_body(o, d)).json()
    again = c.post("/api/plan", json=_body(o, d)).json()
    assert len(calls) == 1 and first == again
    c.post("/api/plan", json={**_body(o, d), "certainty": "allow_unknown"})
    assert len(calls) == 2  # other settings, another search
