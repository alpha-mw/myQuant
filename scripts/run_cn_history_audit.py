#!/usr/bin/env python3
"""Run a full-window CN strict-Parquet coverage audit.

This command is maintenance-only.  It reads canonical Parquet bars and calls
only Tushare reference/evidence endpoints when ``--allow-online`` is supplied;
it never invokes market analysis, portfolio review, execution, or broker APIs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from quant_investor.config import config
from quant_investor.market.cn_history_audit import run_cn_history_audit
from quant_investor.market.download_cn import CNFullMarketDownloader


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", default="CN", choices=["CN"])
    parser.add_argument("--days", type=int, default=100)
    parser.add_argument(
        "--end-date",
        default="auto",
        help="YYYYMMDD or auto (latest_complete_trade_date).",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="data/cn_market_full")
    parser.add_argument(
        "--allow-online",
        action="store_true",
        help="Authorize read-only Tushare trade_cal/suspend_d/bak_daily evidence calls.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict:
    args = _parse_args(argv)
    if not args.allow_online:
        raise SystemExit(
            "--allow-online is required for exact trade_cal and non-trading evidence."
        )
    if not str(getattr(config, "TUSHARE_TOKEN", "") or "").strip():
        raise SystemExit("TUSHARE_TOKEN is not configured.")
    if not str(getattr(config, "TUSHARE_URL", "") or "").strip():
        raise SystemExit("TUSHARE_URL is not configured.")

    downloader = CNFullMarketDownloader(data_dir=str(Path(args.output_root)))
    provider = getattr(downloader, "pro", None)
    if provider is None:
        raise SystemExit("Configured Tushare provider is unavailable.")

    report, output_path = run_cn_history_audit(
        data_root=Path(args.data_root),
        output_root=Path(args.output_root),
        days=int(args.days),
        end_date=str(args.end_date),
        allow_online=True,
        provider=provider,
        suspended_loader=downloader._load_latest_suspended_symbols,
    )
    summary = {
        "market": args.market,
        "output_path": str(output_path),
        "history_audit_status": report["history_audit_status"],
        "audited_trade_dates_count": report["audited_trade_dates_count"],
        "prior_trade_dates_reused": report["prior_trade_dates_reused"],
        "history_primary_absence_dates": report[
            "history_primary_absence_dates"
        ],
        "history_unresolved_gap_dates": report[
            "history_unresolved_gap_dates"
        ],
        "synthetic_bar_count": report["synthetic_bar_count"],
        "portfolio_data_ready": report["portfolio_data_ready"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if report["history_audit_status"] != "passed":
        raise SystemExit(2)
    return report


if __name__ == "__main__":  # pragma: no cover
    main()
