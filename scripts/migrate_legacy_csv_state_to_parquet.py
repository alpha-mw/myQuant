#!/usr/bin/env python3
"""One-off migration from legacy local CSV strategy state to Parquet sidecars.

This script is intentionally outside production packages. It is the only
allowed CSV reader after the Parquet read cleanup.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _copy_if_present(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source, encoding="utf-8-sig")
    frame.to_parquet(target, index=False)
    return True


def migrate_cn_aggressive(base_dir: Path) -> list[str]:
    written: list[str] = []
    if not base_dir.exists():
        return written
    for run_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        for stem in ("ledger_after_manual_switch", "pnl_summary"):
            source = run_dir / f"{stem}.csv"
            target = run_dir / f"{stem}.parquet"
            if _copy_if_present(source, target):
                written.append(str(target))
    return written


def migrate_us_simulated(base_dir: Path) -> list[str]:
    written: list[str] = []
    for stem in ("latest_positions", "latest_trade_log"):
        source = base_dir / f"{stem}.csv"
        target = base_dir / f"{stem}.parquet"
        if _copy_if_present(source, target):
            written.append(str(target))
    if base_dir.exists():
        for run_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
            for stem in ("ledger", "trade_log", "pnl_summary"):
                source = run_dir / f"{stem}.csv"
                target = run_dir / f"{stem}.parquet"
                if _copy_if_present(source, target):
                    written.append(str(target))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cn-aggressive-dir",
        type=Path,
        default=Path("results/strategy_records/CN/aggressive_tech_manufacturing"),
    )
    parser.add_argument(
        "--us-simulated-dir",
        type=Path,
        default=Path("results/strategy_records/US/simulated_portfolio_10000"),
    )
    args = parser.parse_args()
    written = [
        *migrate_cn_aggressive(args.cn_aggressive_dir),
        *migrate_us_simulated(args.us_simulated_dir),
    ]
    for path in written:
        print(path)
    print(f"wrote_parquet_sidecars={len(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
