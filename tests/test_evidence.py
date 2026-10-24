import pandas as pd
import pytest

from pathfinder import evidence as ev


def test_null_values_are_always_unknown():
    df = pd.DataFrame(index=range(3))
    ev.set_attr(df, "x", [1.0, None, 3.0], ev.VERIFIED, "src", "2024-01-01")
    assert df["x_status"].tolist() == ["verified", "unknown", "verified"]
    assert df["x_source"].isna().tolist() == [False, True, False]
    ev.validate(df, "x")


def test_fill_only_improves_evidence_rank():
    df = pd.DataFrame(index=range(4))
    ev.init_unknown(df, "x")
    all_rows = pd.Series(True, index=df.index)
    n = ev.fill_where_better(df, "x", all_rows, pd.Series(["a", "b", None, "d"]), ev.INFERRED, "heur")
    assert n == 3
    # a verified source upgrades inferred rows...
    n = ev.fill_where_better(df, "x", all_rows, pd.Series(["A", None, "C", None]), ev.VERIFIED, "obs")
    assert n == 2
    assert df["x"].tolist() == ["A", "b", "C", "d"]
    assert df["x_status"].tolist() == ["verified", "inferred", "verified", "inferred"]
    # ...but an inferred source never overwrites verified evidence, and same rank never replaces
    assert ev.fill_where_better(df, "x", all_rows, pd.Series(["z"] * 4), ev.INFERRED, "heur2") == 0
    assert ev.fill_where_better(df, "x", all_rows, pd.Series(["z"] * 4), ev.VERIFIED, "obs2") == 2
    ev.validate(df, "x")


def test_validate_rejects_null_value_with_known_status():
    df = pd.DataFrame({"x": [None], "x_status": ["verified"], "x_source": ["s"], "x_as_of": [None]})
    with pytest.raises(ValueError):
        ev.validate(df, "x")


def test_validate_rejects_bad_status():
    df = pd.DataFrame({"x": [1], "x_status": ["probably"], "x_source": ["s"], "x_as_of": [None]})
    with pytest.raises(ValueError):
        ev.validate(df, "x")
