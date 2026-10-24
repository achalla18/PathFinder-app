"""Phase 0 experiment: does the Crux Map idea hold citywide?

Hypothesis H1: when the route recommended for a user's priorities differs from the fastest
route, the two differ in only a few places (≤ 3 divergence sections) in most trips.

Also measured: how often priorities change the route at all, route diversity across the
preference triangle, extra time spent, circuity of fastest routes (a network-gap detector),
step-free feasibility under each certainty setting with relaxation diagnostics, and routing
latency.

Important caveat (stated in the report): results depend on the penalty tables, which are
product judgments, not measurements. A sensitivity profile with doubled intensity is
included so conclusions are not artifacts of one setting.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from pyproj import Transformer

from ..config import ANALYSIS_CRS, DATA_DIR, REPORTS_DIR, WGS84
from ..routing.cost import PENALTY_VERSION, Preferences, Requirements
from ..routing.divergence import divergent_share, sections
from ..routing.explain import WORTH_SHOWING_MIN_GAIN_S, felt_gain

log = logging.getLogger(__name__)

SEED = 20261004
# The trips are a fixed list of coordinates kept with the reports, so that a run after the
# graph has changed walks the same trips as the run before it and the two can be compared
# trip for trip. (Node ids do not survive a rebuild; places do.)
OD_SAMPLE = REPORTS_DIR / "phase0" / "crux_od_pairs.csv"
# A sampled point is placed on the nearest node of the current main network. Farther than
# this and the walkway it was drawn from is no longer there: the trip is left out and counted.
OD_SNAP_MAX_M = 30.0
DIST_BINS = [(400, 800), (800, 1500), (1500, 2500)]
# A small town's network may have few or no destinations this far from some origins; past this
# many draws per wanted trip the band is reported as unfillable instead of looping forever.
SAMPLE_MAX_TRIES_PER_PAIR = 200
ORIGIN_KINDS = {"sidewalk", "crossing", "footway", "pedestrian_street", "path", "shared_path",
                "street_with_sidewalk", "street_no_sidewalk", "street_unknown"}

# Scenarios run for every sampled trip. The step-free ones carry the manual-wheelchair
# profile's requirements (step-free, 8.33%) at the three certainty settings.
PROFILES: dict[str, tuple[Requirements, Preferences]] = {
    "everyday_day": (Requirements(), Preferences()),
    # the walker/cane profile: no requirement, its own mix (routing/profiles.py)
    "walker_cane": (Requirements(), Preferences(w_cross=0.35, w_slope=0.30, w_sidewalk=0.35)),
    "everyday_day_x2": (Requirements(), Preferences(intensity=2.0)),
    "stepfree_likely": (Requirements(step_free=True, max_grade_pct=8.33, certainty="allow_likely"), Preferences()),
    "stepfree_verified": (Requirements(step_free=True, max_grade_pct=8.33, certainty="verified_only"), Preferences()),
    "stepfree_unknown_ok": (Requirements(step_free=True, max_grade_pct=8.33, certainty="allow_unknown"), Preferences()),
}
RELAXATIONS = {
    "drop_max_grade": Requirements(step_free=True, max_grade_pct=None, certainty="allow_likely"),
    "allow_unknown": Requirements(step_free=True, max_grade_pct=8.33, certainty="allow_unknown"),
    "grade_12pct": Requirements(step_free=True, max_grade_pct=12.0, certainty="allow_likely"),
}


def sample_od_pairs(net, n_per_bin: int, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    e = net.edges
    cand = e[(e["component"] == 0) & e["kind"].isin(ORIGIN_KINDS)]
    # length-weighted node sampling: pick an edge proportional to length, then one end
    w = cand["length_m"].to_numpy() / cand["length_m"].sum()
    nodes = net.nodes.set_index("node_id")
    comp0 = net.nodes[net.nodes["component"] == 0]
    cx, cy, cid = comp0["x"].to_numpy(), comp0["y"].to_numpy(), comp0["node_id"].to_numpy()
    rows = []
    for lo, hi in DIST_BINS:
        got = tries = 0
        while got < n_per_bin:
            tries += 1
            if tries > SAMPLE_MAX_TRIES_PER_PAIR * n_per_bin:
                raise RuntimeError(f"band {lo}-{hi} m: only {got} of {n_per_bin} trips found in {tries - 1} draws; "
                                   "the main network may be too small for this band")
            ei = rng.choice(len(cand), p=w)
            o = int(cand["u"].iat[ei] if rng.random() < 0.5 else cand["v"].iat[ei])
            ox, oy = nodes.at[o, "x"], nodes.at[o, "y"]
            d = np.hypot(cx - ox, cy - oy)
            pool = np.nonzero((d >= lo) & (d < hi))[0]
            if len(pool) == 0:
                continue
            # destinations: also length-weighted, restricted to the distance band
            j = pool[rng.integers(len(pool))]
            rows.append({"origin": o, "dest": int(cid[j]), "euclid_m": float(d[j]), "band": f"{lo}-{hi}"})
            got += 1
    return pd.DataFrame(rows)


def fix_sample(nodes: pd.DataFrame, ods: pd.DataFrame) -> pd.DataFrame:
    """Node-id pairs as places: the coordinates of both ends, which outlive the graph they
    were sampled from."""
    xy = nodes.set_index("node_id")[["x", "y"]]
    to_lonlat = Transformer.from_crs(ANALYSIS_CRS, WGS84, always_xy=True)
    out = pd.DataFrame({"band": ods["band"].to_numpy()}, index=pd.RangeIndex(len(ods), name="od"))
    for end in ("origin", "dest"):
        p = xy.loc[ods[end].to_numpy()]
        lon, lat = to_lonlat.transform(p["x"].to_numpy(), p["y"].to_numpy())
        out[f"{end}_lon"], out[f"{end}_lat"] = np.round(lon, 6), np.round(lat, 6)
    return out


def resolve_sample(nodes: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """The fixed sample on this graph: each end placed on the nearest node of the main
    network. Trips with an end more than OD_SNAP_MAX_M from it are dropped; the index keeps
    each remaining trip's number in the sample."""
    main = nodes[nodes["component"] == 0]
    cx, cy, cid = main["x"].to_numpy(), main["y"].to_numpy(), main["node_id"].to_numpy()
    to_xy = Transformer.from_crs(WGS84, ANALYSIS_CRS, always_xy=True)
    ends = {}
    for end in ("origin", "dest"):
        x, y = to_xy.transform(sample[f"{end}_lon"].to_numpy(), sample[f"{end}_lat"].to_numpy())
        near = [int(np.argmin(np.hypot(cx - a, cy - b))) for a, b in zip(x, y)]
        ends[end] = (cid[near], cx[near], cy[near], np.hypot(cx[near] - x, cy[near] - y))
    (o, ox, oy, od_), (d, dx, dy, dd) = ends["origin"], ends["dest"]
    out = pd.DataFrame({"origin": o, "dest": d, "euclid_m": np.hypot(dx - ox, dy - oy), "band": sample["band"].to_numpy()},
                       index=sample.index)
    # two ends that now share a node are no trip at all
    return out[(od_ <= OD_SNAP_MAX_M) & (dd <= OD_SNAP_MAX_M) & (out["origin"] != out["dest"])]


# ------------------------------------------------------------------ worker

_NET = None


def _init_worker() -> None:
    global _NET
    from ..routing.router import Network

    _NET = Network.load()


@dataclass
class Result:
    od: int
    profile: str
    status: str
    n_routes: int = 0
    fastest_time_s: float = float("nan")
    fastest_len_m: float = float("nan")
    circuity: float = float("nan")
    rec_is_fastest: bool | None = None
    rec_extra_s: float = float("nan")
    rec_felt_gain_s: float = float("nan")  # felt seconds the recommendation saves over the fastest
    rec_sections: int | None = None
    rec_divergent_share: float = float("nan")
    rec_section_lengths: str = ""
    alt_sections_median: float = float("nan")
    alt_sections_max: int | None = None
    largest_region_share: float = float("nan")
    elapsed_s: float = float("nan")
    subgraph_edges: int = 0
    relaxation: str = ""


def _run(args) -> list[dict]:
    od_idx, origin, dest, euclid = args
    net = _NET
    out = []
    for name, (req, pref) in PROFILES.items():
        rs = net.route_set(origin, dest, req, pref)
        r = Result(od_idx, name, rs.status, elapsed_s=rs.elapsed_s, subgraph_edges=rs.subgraph_edges)
        if rs.status == "ok":
            f, rec = rs.fastest, rs.recommended
            r.n_routes = len(rs.routes)
            r.fastest_time_s, r.fastest_len_m = f.time_s, f.length_m
            r.circuity = f.length_m / euclid
            r.rec_is_fastest = rec.key == f.key
            r.rec_extra_s = rec.time_s - f.time_s
            r.rec_felt_gain_s = felt_gain(rec, f, pref)
            secs = sections(rec.edge_ids, rec.node_ids, f.edge_ids, f.node_ids, net.length)
            r.rec_sections = len(secs)
            r.rec_divergent_share = divergent_share(rec.edge_ids, f.edge_ids, net.length)
            r.rec_section_lengths = ";".join(f"{s.a_length_m:.0f}" for s in secs)
            alt = [len(sections(x.edge_ids, x.node_ids, f.edge_ids, f.node_ids, net.length)) for x in rs.routes if x.key != f.key]
            if alt:
                r.alt_sections_median = float(np.median(alt))
                r.alt_sections_max = int(max(alt))
            total = sum(len(x.weights_won) for x in rs.routes)
            r.largest_region_share = max(len(x.weights_won) for x in rs.routes) / total
        elif rs.status == "no_feasible_route" and name == "stepfree_likely":
            fixes = [k for k, rq in RELAXATIONS.items() if net.route_set(origin, dest, rq, pref).status == "ok"]
            r.relaxation = ",".join(fixes) if fixes else "none_single"
        out.append(asdict(r))
    return out


def load_checkpoint(path, ods) -> tuple[list[dict], set[int]]:
    """Read checkpoint rows, keeping only OD pairs that have a row for every profile.

    An interrupted run can leave a truncated last line or an OD pair with only some of its
    profile rows. Counting such a pair as done would silently drop its missing profiles from
    the statistics, so partial pairs are discarded here (and logged) and get recomputed.
    ``ods`` are the numbers of the trips in this run.
    """
    ods = set(ods)
    by_od: dict[int, dict[str, dict]] = {}
    unreadable = 0
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                unreadable += 1
                continue
            by_od.setdefault(int(row["od"]), {})[row["profile"]] = row
    complete = {od: d for od, d in by_od.items() if set(d) == set(PROFILES) and od in ods}
    if unreadable or len(complete) != len(by_od):
        log.warning("crux checkpoint: dropping %d unreadable line(s) and %d partial or out-of-sample OD pair(s)",
                    unreadable, len(by_od) - len(complete))
    rows = [d[p] for d in complete.values() for p in PROFILES]
    return rows, set(complete)


def run(n_per_bin: int = 334, workers: int | None = None, fresh: bool = False, resample: bool = False) -> dict:
    """Run (or resume) the experiment.

    Results are checkpointed per OD pair to a JSONL file as they complete, so an interrupted
    run (e.g. killed under memory pressure) resumes where it stopped. Any change to the graph
    or penalties requires ``fresh=True``, which discards the checkpoint and places the fixed
    sample (OD_SAMPLE) on the current graph again: the same trips, so runs can be compared.
    ``resample=True`` draws a new sample, after which earlier runs are no longer comparable.
    """
    from ..routing.router import Network

    out_dir = DATA_DIR / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    od_path, ckpt = out_dir / "crux_od_pairs.csv", out_dir / "crux_checkpoint.jsonl"
    if resample:
        OD_SAMPLE.unlink(missing_ok=True)
    if fresh or resample:
        od_path.unlink(missing_ok=True)
        ckpt.unlink(missing_ok=True)
    if od_path.exists():
        ods = pd.read_csv(od_path, index_col="od")
    else:
        net = Network.load()
        if not OD_SAMPLE.exists():
            OD_SAMPLE.parent.mkdir(parents=True, exist_ok=True)
            fix_sample(net.nodes, sample_od_pairs(net, n_per_bin)).to_csv(OD_SAMPLE)
        sample = pd.read_csv(OD_SAMPLE, index_col="od")
        if len(sample) != n_per_bin * len(DIST_BINS):
            raise RuntimeError(f"the fixed sample has {len(sample)} pairs ({len(sample) // len(DIST_BINS)} per band); "
                               "pass that as --per-bin, or --resample to draw a new one")
        ods = resolve_sample(net.nodes, sample)
        log.info("crux: %d of %d sampled trips are on the current network", len(ods), len(sample))
        del net
        ods.to_csv(od_path)
    done: set[int] = set()
    if ckpt.exists():
        rows, done = load_checkpoint(ckpt, ods.index)
        # rewrite so the file holds only complete pairs before new rows are appended
        tmp = ckpt.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(row, default=float) + "\n" for row in rows))
        tmp.replace(ckpt)
    args = [(i, int(r.origin), int(r.dest), float(r.euclid_m)) for i, r in ods.iterrows() if i not in done]
    # Each worker holds its own copy of the network; keep the default conservative.
    workers = workers or min(4, max(1, mp.cpu_count() - 2))
    log.info("crux experiment: %d OD pairs (%d already done) x %d profiles on %d workers",
             len(ods), len(done), len(PROFILES), workers)
    if args:
        with mp.Pool(workers, initializer=_init_worker) as pool, ckpt.open("a") as f:
            for n, chunk in enumerate(pool.imap_unordered(_run, args, chunksize=2), 1):
                # one write per OD pair, so an interruption cannot split its profiles
                f.write("".join(json.dumps(row, default=float) + "\n" for row in chunk))
                f.flush()
                if n % 50 == 0:
                    log.info("crux: %d / %d OD pairs", len(done) + n, len(ods))
    rows, done = load_checkpoint(ckpt, ods.index)
    if len(done) != len(ods):
        raise RuntimeError(f"checkpoint incomplete: {len(done)} of {len(ods)} OD pairs have all profiles; rerun to resume")
    res = pd.DataFrame(rows).merge(ods.reset_index(), on="od")
    res.to_parquet(out_dir / "crux_results.parquet")
    return write_report(res)


def write_report(res: pd.DataFrame | None = None) -> dict:
    """Summarize saved results into reports/phase0/ (JSON, markdown, chart)."""
    if res is None:
        res = pd.read_parquet(DATA_DIR / "analysis" / "crux_results.parquet")
    summary = summarize(res)
    if OD_SAMPLE.exists():
        summary["n_sampled"] = int(len(pd.read_csv(OD_SAMPLE)))
    rep = REPORTS_DIR / "phase0"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "crux_summary.json").write_text(json.dumps(summary, indent=2))
    _chart(res, rep / "crux_sections.png")
    (rep / "crux.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def _chart(res: pd.DataFrame, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profiles = ["everyday_day", "walker_cane", "stepfree_likely"]
    fig, axes = plt.subplots(1, len(profiles), figsize=(11, 3.4), dpi=110, sharey=True)
    for ax, name in zip(axes, profiles):
        g = res[(res["profile"] == name) & (res["status"] == "ok") & (res["rec_is_fastest"] == False)]  # noqa: E712
        counts = g["rec_sections"].clip(upper=8).value_counts().sort_index()
        share = counts / counts.sum() * 100 if counts.sum() else counts
        ax.bar([str(int(i)) if i < 8 else "8+" for i in share.index], share.values, color="#2b8cbe")
        ax.set_title(f"{name}\n(n={len(g)} trips where route changed)", fontsize=9)
        ax.set_xlabel("divergence sections vs. fastest")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("% of trips")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _markdown(s: dict) -> str:
    P = s["profiles"]
    lines = [
        "# Phase 0 — Crux hypothesis experiment",
        "",
        "_Generated by `pf crux` / `write_report`. Do not edit by hand._",
        "",
        f"**Design.** {s['n_od_pairs']:,} origin–destination pairs sampled citywide (length-weighted "
        f"origins on walkable edges; destinations stratified into 400–800 m, 800–1,500 m and "
        f"1,500–2,500 m straight-line bands; seed {SEED}). The pairs are a fixed list of coordinates "
        f"(`crux_od_pairs.csv`, {s.get('n_sampled', s['n_od_pairs']):,} pairs) placed on the current network, so runs on "
        f"different graph versions walk the same trips; pairs with an end more than {OD_SNAP_MAX_M:.0f} m from the "
        "network, or whose two ends now share a node, are left out. Each pair is routed under six profiles. "
        "For each, the route recommended for the profile's priorities (equal weights unless noted) is "
        "compared with the fastest route that satisfies the same requirements.",
        "",
        "**H1:** when the recommended route differs from the fastest, they differ in ≤ 3 places "
        "(divergence sections) for most trips.",
        "",
        f"**Caveat.** Recommendations depend on the {PENALTY_VERSION} penalty tables (`routing/cost.py`), which are "
        "product judgments, not measurements. `everyday_day_x2` doubles their intensity as a "
        "sensitivity check. These results test whether the *interaction* is viable on the region's real "
        "network; they say nothing about whether routes are safer.",
        "",
        "| profile | feasible % | route changed % | H1: ≤3 sections % (when changed) | extra min p50 / p90 | median routes in set | latency p50 / p95 s |",
        "|---|---|---|---|---|---|---|",
    ]
    worth_lines = [
        "",
        "## Is the difference worth showing?",
        "",
        f"A recommendation that is felt to be less than {WORTH_SHOWING_MIN_GAIN_S:.0f} s better than the fastest route "
        "(at the profile's own priorities) is not offered as a separate choice (`routing/explain.py`).",
        "",
        "| profile | worth showing, % of feasible trips | % of changed routes | H1: ≤3 sections % (when worth showing) | felt gain min p50 (when changed) |",
        "|---|---|---|---|---|",
        *[f"| {name} | {p.get('worth_showing_pct_of_feasible', '—')} | {p.get('worth_showing_pct_of_changed', '—')} | "
          f"{p.get('H1_sections_le3_pct_when_worth_showing', '—')} | {p.get('rec_felt_gain_min_p50_when_changed', '—')} |"
          for name, p in P.items()],
    ]
    for name, p in P.items():
        lines.append(
            f"| {name} | {p['feasible_pct']} | {p.get('rec_differs_from_fastest_pct', '')} | "
            f"{p.get('H1_sections_le3_pct_when_changed', '—')} | "
            f"{p.get('rec_extra_min_p50_when_changed', '—')} / {p.get('rec_extra_min_p90_when_changed', '—')} | "
            f"{p.get('median_routes_in_set', '')} | {p.get('latency_p50_s', '')} / {p.get('latency_p95_s', '')} |"
        )
    ed = P.get("everyday_day", {})
    lines += [
        "",
        "## H1 by distance band (everyday_day)",
        "",
        "| band (m) | trips where route changed | ≤3 sections % | median sections |",
        "|---|---|---|---|",
        *[f"| {b} | {v['n']} | {v['sections_le3_pct']} | {v['sections_p50']} |"
          for b, v in sorted(ed.get("H1_by_distance_band", {}).items(), key=lambda kv: int(kv[0].split('-')[0]))],
        "",
        "![sections](crux_sections.png)",
        *worth_lines,
        "",
        "## Network circuity (fastest route length / straight-line distance)",
        "",
        f"- median {ed.get('circuity_p50')}, p95 {ed.get('circuity_p95')}, "
        f"trips with circuity > 2: {ed.get('circuity_gt_2_count')} (candidates for missing links or real barriers "
        "such as rail yards, freeways and rivers)",
        "",
        "## Step-free feasibility",
        "",
        "Requirement: step-free + max sustained grade 8.33 %. Infeasible trips under `allow_likely` were re-run "
        "with single relaxations to find what blocks them:",
        "",
        *[f"- `{k}`: {v}" for k, v in P.get("stepfree_likely", {}).get("infeasible_relaxations", {}).items()],
        "",
    ]
    return "\n".join(lines)


def _pct(x: pd.Series) -> float:
    return round(float(x.mean()) * 100, 1) if len(x) else float("nan")


def summarize(res: pd.DataFrame) -> dict:
    s: dict = {"n_od_pairs": int(res["od"].nunique()), "profiles": {}}
    for name, g in res.groupby("profile"):
        ok = g[g["status"] == "ok"]
        changed = ok[ok["rec_is_fastest"] == False]  # noqa: E712
        p = {
            "n": int(len(g)),
            "feasible_pct": _pct(g["status"] == "ok"),
            "rec_differs_from_fastest_pct": _pct(ok["rec_is_fastest"] == False),  # noqa: E712
            "median_routes_in_set": float(ok["n_routes"].median()) if len(ok) else None,
            "latency_p50_s": round(float(ok["elapsed_s"].median()), 3) if len(ok) else None,
            "latency_p95_s": round(float(ok["elapsed_s"].quantile(0.95)), 3) if len(ok) else None,
            "subgraph_edges_p50": float(ok["subgraph_edges"].median()) if len(ok) else None,
        }
        if len(changed):
            p.update({
                "H1_sections_le3_pct_when_changed": _pct(changed["rec_sections"] <= 3),
                "sections_distribution_when_changed": changed["rec_sections"].value_counts().sort_index().astype(int).to_dict(),
                "rec_extra_min_p50_when_changed": round(float(changed["rec_extra_s"].median() / 60), 2),
                "rec_extra_min_p90_when_changed": round(float(changed["rec_extra_s"].quantile(0.9) / 60), 2),
                "rec_divergent_share_p50_when_changed": round(float(changed["rec_divergent_share"].median()), 3),
                "rec_felt_gain_min_p50_when_changed": round(float(changed["rec_felt_gain_s"].median() / 60), 2),
            })
            worth = changed[changed["rec_felt_gain_s"] >= WORTH_SHOWING_MIN_GAIN_S]
            p["worth_showing_pct_of_feasible"] = round(len(worth) / len(ok) * 100, 1)
            p["worth_showing_pct_of_changed"] = round(len(worth) / len(changed) * 100, 1)
            if len(worth):
                p["H1_sections_le3_pct_when_worth_showing"] = _pct(worth["rec_sections"] <= 3)
            by_band = {}
            for band, gb in changed.groupby("band"):
                by_band[band] = {"n": int(len(gb)), "sections_le3_pct": _pct(gb["rec_sections"] <= 3),
                                 "sections_p50": float(gb["rec_sections"].median())}
            p["H1_by_distance_band"] = by_band
        if name == "everyday_day" and len(ok):
            p["circuity_p50"] = round(float(ok["circuity"].median()), 3)
            p["circuity_p95"] = round(float(ok["circuity"].quantile(0.95)), 3)
            p["circuity_gt_2_count"] = int((ok["circuity"] > 2).sum())
        if name == "stepfree_likely":
            inf = g[g["status"] == "no_feasible_route"]
            p["infeasible_relaxations"] = inf["relaxation"].value_counts().to_dict()
        s["profiles"][name] = p
    return s
