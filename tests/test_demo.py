"""The demo launcher and the landing page's numbers (pf demo)."""
import json

from pathfinder import api, demo


def test_landing_page_reads_each_places_evidence_from_its_coverage_report(tmp_path, monkeypatch):
    report = tmp_path / "reports" / "somewhere" / "phase0"
    report.mkdir(parents=True)
    (report / "coverage.md").write_text(
        "| Criterion | Value % | Bar % | Result | Counts |\n|---|---|---|---|---|\n"
        "| 1. Curb evidence at crossing ends, implicit included | 17.1 | 60.0 | misses | 1,670 of 9,771 ends |\n"
        "| 2. Slope verified, share of walkable length | 95.8 | 80.0 | passes |  |\n"
        "| 3. Sidewalk evidence, share of walkable length | 42.2 | 60.0 | misses | ... |\n", encoding="utf-8")
    monkeypatch.setattr(demo, "REPO_ROOT", tmp_path)
    assert demo.evidence("somewhere") == {"curbs": 17.1, "slope": 95.8, "sidewalks": 42.2}
    assert demo.evidence("nowhere") == {"curbs": None, "slope": None, "sidewalks": None}


def test_every_demo_place_is_built_from_open_data_only():
    demo._check_open()  # raises SystemExit if any place holds an internal source
    assert len({p.port for p in demo.PLACES}) == len(demo.PLACES) and demo.LANDING_PORT not in {p.port for p in demo.PLACES}


def test_example_trips_keep_only_well_formed_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "EXAMPLES_DIR", tmp_path)
    good = {"label": "To the library", "note": "avoids a steep block", "from": [-77.41, 39.41, "Home"], "to": [-77.40, 39.42, "Library"]}
    (tmp_path / f"{api.REGION.id}.json").write_text(json.dumps([good, {"label": "broken", "from": [1, 2]}]), encoding="utf-8")
    assert api._examples() == [good]


def test_readiness_probe_passes_the_password_the_servers_ask_for(monkeypatch):
    """With PF_APP_PASSWORD set, every place's server asks for it on /api/health too; the
    launcher must send it, or it waits out its timeout and stops a demo that was healthy."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv(api.PASSWORD_ENV, "s3cret")
    client = TestClient(api.create_app())  # the real password gate; /api/health needs no planner
    seen = []

    def get(url, timeout, auth=None):
        seen.append(auth)
        return client.get("/api/health", auth=auth)

    monkeypatch.setattr(demo.requests, "get", get)
    running = type("Proc", (), {"poll": lambda self: None})()
    demo._wait("http://127.0.0.1:8001/", [running], timeout_s=5)
    assert seen == [("pathfinder", "s3cret")]


def test_readiness_probe_stops_at_once_on_a_wrong_password(monkeypatch):
    import pytest
    from fastapi.testclient import TestClient

    monkeypatch.setenv(api.PASSWORD_ENV, "right")
    client = TestClient(api.create_app())
    # the server expects another password than the one that reaches it: it answers 401, and the
    # launcher must say so at once instead of waiting out its timeout
    monkeypatch.setattr(demo.requests, "get", lambda url, timeout, auth=None: client.get("/api/health", auth=("x", "wrong")))
    running = type("Proc", (), {"poll": lambda self: None})()
    with pytest.raises(SystemExit, match="refused"):
        demo._wait("http://127.0.0.1:8001/", [running], timeout_s=5)


def test_reports_become_unanswered_field_sheet_rows_once():
    """A report is a place to visit, never an answer: it reaches the sheet with no result, a
    curb report as a corner and anything else as a note the pipeline does not read."""
    import importlib.util

    import pandas as pd

    from pathfinder.config import REPO_ROOT

    spec = importlib.util.spec_from_file_location("add_reports", REPO_ROOT / "tools" / "field_checks" / "add_reports.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    reports = pd.DataFrame([{"day": "2026-10-22", "kind": "no_ramp", "lat": "39.41", "lon": "-77.41", "note": ""},
                            {"day": "2026-10-22", "kind": "blocked_sidewalk", "lat": "39.42", "lon": "-77.40", "note": "bins"}])
    rows = mod.rows(reports)
    assert rows["kind"].tolist() == ["corner", "note"]
    assert (rows["result"] == "").all() and (rows["visited_on"] == "").all()
    assert rows["place"].tolist() == mod.rows(reports)["place"].tolist()  # same report, same place: added once
    assert "bins" in rows["notes"].iat[1]

