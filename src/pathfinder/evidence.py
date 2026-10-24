"""Evidence grading for every network attribute.

Each graded attribute ``X`` on an edge is stored as four columns:

- ``X``          the value (may be null)
- ``X_status``   one of VERIFIED / INFERRED / UNKNOWN
- ``X_source``   source id from ``sources.SOURCES`` (or "derived:<method>")
- ``X_as_of``    ISO date (or year) the underlying observation describes, if known

``VERIFIED`` means a source directly observed the value — including *confirmed absence* in an
inventory that is comprehensive for that place (e.g., no stairs in DC's planimetric stairs
inventory for an on-street edge). Absence in an inventory that does not cover the place is
never verification.

Rules (enforced by tests):
1. A null value is always UNKNOWN. Missing data is never favorable.
2. INFERRED means "a model or heuristic produced this"; it never satisfies a
   "verified only" requirement.
3. Status is about the evidence, not about whether the condition is good or bad: a verified
   raised kerb is VERIFIED *and* a barrier.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

VERIFIED = "verified"
INFERRED = "inferred"
UNKNOWN = "unknown"
STATUSES = (VERIFIED, INFERRED, UNKNOWN)

# Status order for "best available evidence" merges.
_RANK = {VERIFIED: 0, INFERRED: 1, UNKNOWN: 2}


def columns(attr: str) -> list[str]:
    return [attr, f"{attr}_status", f"{attr}_source", f"{attr}_as_of"]


def set_attr(
    df: pd.DataFrame,
    attr: str,
    values: pd.Series | np.ndarray | Iterable,
    status: str | pd.Series,
    source: str | pd.Series,
    as_of: str | pd.Series | None = None,
) -> None:
    """Assign a graded attribute; nulls are forced to UNKNOWN regardless of ``status``."""
    vals = pd.Series(values, index=df.index)
    df[attr] = vals
    st = pd.Series(status, index=df.index, dtype="object")
    st[vals.isna()] = UNKNOWN
    df[f"{attr}_status"] = st
    src = pd.Series(source, index=df.index, dtype="object")
    src[vals.isna()] = None
    df[f"{attr}_source"] = src
    ao = pd.Series(as_of, index=df.index, dtype="object")
    ao[vals.isna()] = None
    df[f"{attr}_as_of"] = ao


def init_unknown(df: pd.DataFrame, attr: str) -> None:
    df[attr] = pd.Series([None] * len(df), index=df.index, dtype="object")
    df[f"{attr}_status"] = UNKNOWN
    df[f"{attr}_source"] = None
    df[f"{attr}_as_of"] = None


def fill_where_better(
    df: pd.DataFrame,
    attr: str,
    mask: pd.Series,
    values: pd.Series,
    status: str,
    source: str,
    as_of: pd.Series | str | None = None,
) -> int:
    """Overwrite ``attr`` only where ``mask`` holds, the new value is non-null, and the new
    evidence is strictly better ranked than what is there. Returns rows updated."""
    values = pd.Series(values, index=df.index)
    cur_rank = df[f"{attr}_status"].map(_RANK)
    upd = mask & values.notna() & (_RANK[status] < cur_rank)
    if not upd.any():
        return 0
    df.loc[upd, attr] = values[upd]
    df.loc[upd, f"{attr}_status"] = status
    df.loc[upd, f"{attr}_source"] = source
    ao = pd.Series(as_of, index=df.index, dtype="object")
    df.loc[upd, f"{attr}_as_of"] = ao[upd]
    return int(upd.sum())


def validate(df: pd.DataFrame, attr: str) -> None:
    """Raise if any graded-attribute invariant is violated."""
    st = df[f"{attr}_status"]
    bad = set(st.dropna().unique()) - set(STATUSES)
    if bad:
        raise ValueError(f"{attr}: invalid statuses {bad}")
    if st.isna().any():
        raise ValueError(f"{attr}: null status")
    null_val_not_unknown = df[attr].isna() & (st != UNKNOWN)
    if null_val_not_unknown.any():
        raise ValueError(f"{attr}: {int(null_val_not_unknown.sum())} null values not marked unknown")
    unknown_with_value = df[attr].notna() & (st == UNKNOWN)
    if unknown_with_value.any():
        raise ValueError(f"{attr}: {int(unknown_with_value.sum())} unknown rows carry a value")
    known_without_source = df[f"{attr}_source"].isna() & (st != UNKNOWN)
    if known_without_source.any():
        raise ValueError(f"{attr}: {int(known_without_source.sum())} graded rows have no source")
