"""Place search: matching, abbreviations, ranking."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pathfinder import api
from pathfinder.places import PlaceIndex, merge_same_place, normalize


@pytest.fixture
def index(tmp_path):
    rows = [
        ("Union Station", "station", "railway", -77.0064, 38.8973),
        ("Union Market", "marketplace", "amenity", -76.9979, 38.9086),
        ("Station Kitchen", "restaurant", "amenity", -77.03, 38.90),
        ("1600 Pennsylvania Avenue Northwest", "The White House", "address", -77.0365, 38.8977),
        ("1600 Pennsylvania Avenue Southeast", "address", "address", -76.9820, 38.8800),
        ("Ben's Chili Bowl", "restaurant", "amenity", -77.0288, 38.9172),
    ]
    df = pd.DataFrame(rows, columns=["label", "detail", "kind", "lon", "lat"])
    df["search"] = df["label"].map(normalize)
    path = tmp_path / "places.parquet"
    df.to_parquet(path)
    return PlaceIndex(path)


def test_normalize_spells_out_street_abbreviations():
    assert normalize("1600 Penn. Ave NW") == "1600 penn avenue northwest"
    assert normalize("Ben's Chili Bowl") == "bens chili bowl"


def test_every_word_must_match_the_start_of_a_word(index):
    assert [r["label"] for r in index.search("union st")] == ["Union Station"]
    assert [r["label"] for r in index.search("bens chili")] == ["Ben's Chili Bowl"]
    assert index.search("nion") == []  # not a word start
    assert index.search("u") == []  # too short to be worth searching


def test_abbreviated_address_finds_the_full_one_and_nearer_wins(index):
    both = index.search("1600 pennsylvania ave")
    assert {r["label"] for r in both} == {"1600 Pennsylvania Avenue Northwest", "1600 Pennsylvania Avenue Southeast"}
    assert index.search("1600 pennsylvania ave nw")[0]["detail"] == "The White House"
    near_se = index.search("1600 pennsylvania ave", near=(-76.98, 38.88))
    assert near_se[0]["label"].endswith("Southeast")


def test_names_that_start_with_the_query_come_before_names_that_contain_it(index):
    assert [r["label"] for r in index.search("station")] == ["Station Kitchen", "Union Station"]


def test_search_endpoint(index, monkeypatch):
    monkeypatch.delenv(api.PASSWORD_ENV, raising=False)
    c = TestClient(api.create_app(places=index))
    r = c.get("/api/search", params={"q": "union"})
    assert r.status_code == 200 and r.json()["results"][0]["label"] == "Union Station"  # the station ranks above the market
    assert c.get("/api/search", params={"q": "u"}).status_code == 422


def test_duplicate_mappings_merge_but_branches_of_a_chain_do_not():
    """Regression: every branch of a chain but the first was dropped from the index."""
    rows = [
        ("Safeway", "supermarket", "shop", -77.0300, 38.9100),  # a node...
        ("Safeway", "supermarket", "shop", -77.0302, 38.9101),  # ...and its building, 20 m away
        ("Safeway", "supermarket", "shop", -76.9900, 38.8800),  # another branch across town
        ("Safeway", "fuel", "amenity", -77.0300, 38.9100),  # same name, different kind of place
    ]
    df = pd.DataFrame(rows, columns=["label", "detail", "kind", "lon", "lat"])
    df["search"] = df["label"].map(normalize)
    out = merge_same_place(df)
    assert len(out) == 3
    assert sorted(out.loc[out["kind"] == "shop", "lon"]) == [-77.03, -76.99]
