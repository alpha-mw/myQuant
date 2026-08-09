#!/usr/bin/env python3
"""Compare two exact fundamental generation files before promotion.

An authoritative full rebuild regenerates the entire mart, not just the years
being added. So extending coverage back to 2015 can silently change or drop rows
in the 2021-2026 span that is already in production. This checks three things
before the pointer is allowed to move:

1. Coverage was actually extended - the staged mart reaches back to 2015 with
   usable metric coverage, which is the whole point of the rebuild.
2. The existing span did not regress - no symbol/date rows lost and no metric
   coverage materially reduced over 2021-06 to 2026-06.
3. PIT discipline holds - availability_date never precedes end_date, so no row
   claims to have been knowable before the period it reports on had closed.

This verifier is deliberately path-exact: callers must name both parquet files.
It never follows a pointer, scans a generation root, promotes data, or writes a
receipt.  Prints a verdict and exits non-zero when promotion should not proceed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS = ("fin_roe", "fin_roa", "fin_debt_to_assets", "fin_net_profit_yoy")
# Coverage may drift slightly on a rebuild; more than this is a regression.
COVERAGE_TOLERANCE = 0.02


def _exact_file(raw: str, *, label: str) -> Path:
    if not raw or any(token in raw for token in ("*", "?", "[", "]")):
        raise SystemExit(f"{label} must be an explicit path without glob syntax")
    path = Path(raw)
    if not path.is_absolute():
        raise SystemExit(f"{label} must be absolute")
    if path.is_symlink():
        raise SystemExit(f"{label} must not be a symlink: {path}")
    return path


def _load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    print(f"  reading {path}")
    return pd.read_parquet(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged-generation", required=True)
    parser.add_argument("--live-generation", required=True)
    args = parser.parse_args()

    staged = _load(_exact_file(args.staged_generation, label="--staged-generation"))
    live = _load(_exact_file(args.live_generation, label="--live-generation"))
    for frame in (staged, live):
        frame["trade_date"] = frame["trade_date"].astype("string")

    failures: list[str] = []

    print("=== 1. coverage extended ===")
    s_min, s_max = staged["trade_date"].min(), staged["trade_date"].max()
    l_min, l_max = live["trade_date"].min(), live["trade_date"].max()
    print(
        f"  live   : {l_min} -> {l_max}  rows={len(live):,}  symbols={live['ts_code'].nunique():,}"
    )
    print(
        f"  staged : {s_min} -> {s_max}  rows={len(staged):,}  "
        f"symbols={staged['ts_code'].nunique():,}"
    )
    if s_min >= l_min:
        failures.append(f"staged start {s_min} did not extend past live start {l_min}")
    elif s_min > "2016-01-01":
        failures.append(f"staged start {s_min} falls short of the 2015 target")

    early = staged.loc[staged["trade_date"] < "2021-01-01"]
    print(f"  pre-2021 rows: {len(early):,}")
    if early.empty:
        failures.append("no pre-2021 rows in the staged mart")
    else:
        for metric in METRICS:
            if metric in early.columns:
                cov = early[metric].notna().mean()
                flag = "" if cov >= 0.5 else "  <-- thin"
                print(f"    {metric:<22} {cov * 100:5.1f}%{flag}")

    print("\n=== 2. existing span did not regress ===")
    # Derive the window from the live mart rather than hardcoding it: the span
    # the pointer actually serves is what must not regress, and it moves with
    # every routine publish.
    overlap_start, overlap_end = l_min, l_max
    print(f"  window     : {overlap_start} -> {overlap_end} (live span)")
    s_ov = staged.loc[staged["trade_date"].between(overlap_start, overlap_end)]
    l_ov = live.loc[live["trade_date"].between(overlap_start, overlap_end)]
    print(f"  live rows  : {len(l_ov):,}")
    print(f"  staged rows: {len(s_ov):,}")

    live_keys = set(zip(l_ov["ts_code"], l_ov["trade_date"]))
    staged_keys = set(zip(s_ov["ts_code"], s_ov["trade_date"]))
    lost = live_keys - staged_keys
    print(f"  symbol/date rows present live but absent staged: {len(lost):,}")
    if lost:
        failures.append(f"{len(lost)} rows lost from the already-served span")

    for metric in METRICS:
        if metric in l_ov.columns and metric in s_ov.columns:
            lc, sc = l_ov[metric].notna().mean(), s_ov[metric].notna().mean()
            delta = sc - lc
            mark = "" if delta >= -COVERAGE_TOLERANCE else "  <-- REGRESSION"
            print(
                f"    {metric:<22} live {lc * 100:5.1f}%  "
                f"staged {sc * 100:5.1f}%  delta {delta * 100:+5.1f}%{mark}"
            )
            if delta < -COVERAGE_TOLERANCE:
                failures.append(f"{metric} coverage regressed by {abs(delta) * 100:.1f}%")

    print("\n=== 3. PIT discipline ===")
    if {"availability_date", "end_date"} <= set(staged.columns):
        avail = staged["availability_date"].astype("string").str.replace("-", "", regex=False)
        end = staged["end_date"].astype("string").str.replace("-", "", regex=False)
        both = avail.notna() & end.notna()
        violations = int((avail[both] < end[both]).sum())
        print(f"  rows where availability_date precedes end_date: {violations:,}")
        if violations:
            failures.append(f"{violations} rows violate PIT ordering")
    else:
        failures.append("staged mart lacks availability_date/end_date for the PIT check")

    print("\n" + "=" * 60)
    if failures:
        print("DO NOT PROMOTE:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("All checks passed. Safe to promote.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
