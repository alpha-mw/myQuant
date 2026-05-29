#!/usr/bin/env python3
"""Build audited clean daily-bar layers from local market CSV snapshots."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from quant_investor.market.config import get_market_settings
from quant_investor.market.daily_cleaner import (
    DailyCleanConfig,
    clean_market_daily_data,
    latest_download_report_target,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "离线清洗 myQuant 本地日线 CSV。raw 数据不被修改；"
            "clean 层和 quarantine/audit 文件写入 data/clean。"
        )
    )
    parser.add_argument("--market", required=True, choices=["CN", "US"])
    parser.add_argument("--raw-dir", default="")
    parser.add_argument("--clean-dir", default="")
    parser.add_argument("--audit-dir", default="")
    parser.add_argument(
        "--target-date",
        default="auto",
        help="清洁层要求的最新交易日；auto 使用最新 download_report，可传 none 关闭 stale 标记。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只写 audit，不写 clean CSV。",
    )
    parser.add_argument(
        "--max-quarantine-rows-per-symbol",
        type=int,
        default=5000,
    )
    return parser


def _resolve_target_date(raw_dir: Path, market: str, value: str) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"", "none", "off", "false"}:
        return None
    if normalized == "auto":
        return latest_download_report_target(raw_dir, market)
    return value


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    settings = get_market_settings(args.market)
    market = settings.market
    raw_dir = Path(args.raw_dir or settings.data_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_dir = Path(args.clean_dir or f"data/clean/{market.lower()}_daily")
    audit_dir = Path(args.audit_dir or f"data/clean/audit/{market.lower()}_{timestamp}")
    target_date = _resolve_target_date(raw_dir, market, args.target_date)

    manifest = clean_market_daily_data(
        DailyCleanConfig(
            market=market,
            raw_dir=raw_dir,
            clean_dir=clean_dir,
            audit_dir=audit_dir,
            latest_required_date=target_date,
            write_clean=not args.dry_run,
            max_quarantine_rows_per_symbol=args.max_quarantine_rows_per_symbol,
        )
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
