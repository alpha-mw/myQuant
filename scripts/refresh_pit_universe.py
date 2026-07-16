#!/usr/bin/env python3
"""Refresh CN point-in-time listing membership from Tushare.

Default mode is dry-run: it fetches the three stock_basic statuses and prints
counts, but does not write local artifacts unless --execute is supplied.  An
executed refresh publishes an immutable generation and atomically advances the
latest discovery manifest; it never overwrites the frozen legacy Parquet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from quant_investor.config import config
from quant_investor.credential_utils import create_tushare_pro
from quant_investor.market.pit_universe import (
    PITUniverseStore,
    estimate_historical_bar_backfill_cost,
    refresh_pit_universe_from_tushare,
)

try:
    import tushare as ts
except ImportError:  # pragma: no cover - exercised by local operator env
    ts = None


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", default="CN", choices=["CN"])
    parser.add_argument(
        "--source-root",
        default=config.PIT_UNIVERSE_SOURCE_ROOT,
        help="Canonical PIT membership root.",
    )
    parser.add_argument(
        "--raw-root",
        default="data/cn_universe/raw",
        help="Raw JSONL PIT stock_basic snapshot root.",
    )
    parser.add_argument(
        "--compat-path",
        default="data/cn_universe/stock_basic_membership_latest.json",
        help="Human-readable compatibility export path.",
    )
    parser.add_argument(
        "--components-path",
        default="data/cn_universe/cn_index_components.json",
        help="Current full-A component scope that the refreshed PIT must cover.",
    )
    parser.add_argument("--execute", action="store_true", help="Write local PIT membership artifacts.")
    parser.add_argument(
        "--allow-online",
        action="store_true",
        help="Acknowledge that this command will call Tushare.",
    )
    parser.add_argument(
        "--missing-trade-dates",
        type=int,
        default=0,
        help="Optional dry-run cost estimate for date-scoped historical bar repair.",
    )
    parser.add_argument(
        "--unresolved-symbol-dates",
        type=int,
        default=0,
        help="Optional dry-run cost estimate for symbol-scoped tail repair.",
    )
    parser.add_argument("--output", default="", help="Optional JSON report path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict:
    args = _parse_args(argv)
    if not args.allow_online:
        raise SystemExit("--allow-online is required because this command calls Tushare.")
    if args.execute and not bool(getattr(config, "PIT_UNIVERSE_BACKFILL_ENABLED", False)):
        raise SystemExit("Set PIT_UNIVERSE_BACKFILL_ENABLED=1 to write PIT universe artifacts.")
    if ts is None:
        raise SystemExit("tushare is not installed.")
    if not config.TUSHARE_TOKEN:
        raise SystemExit("TUSHARE_TOKEN is not set.")

    pro = create_tushare_pro(ts, config.TUSHARE_TOKEN, config.TUSHARE_URL)
    store = PITUniverseStore(
        root_dir=Path(args.source_root),
        raw_root=Path(args.raw_root),
        compatibility_path=Path(args.compat_path),
    )
    components_path = Path(args.components_path)
    if not components_path.exists():
        raise SystemExit(
            f"Current full-A components are missing: {components_path}"
        )
    components_payload = json.loads(
        components_path.read_text(encoding="utf-8")
    )
    required_symbols = sorted(
        {
            str(symbol or "").strip().upper()
            for symbol in components_payload.get("full_a", []) or []
            if str(symbol or "").strip()
        }
    )
    if not required_symbols:
        raise SystemExit("Current full-A component scope is empty.")
    report = refresh_pit_universe_from_tushare(
        pro,
        store=store,
        execute=bool(args.execute),
        required_symbols=required_symbols,
    )
    if args.execute:
        required_generation_evidence = (
            "generation_manifest_path",
            "generation_manifest_sha256",
            "canonical_path",
            "canonical_sha256",
        )
        missing_generation_evidence = [
            key for key in required_generation_evidence if not report.get(key)
        ]
        if missing_generation_evidence:
            raise RuntimeError(
                "PIT generation publish omitted binding evidence: "
                + ",".join(missing_generation_evidence)
            )
    report["market"] = args.market
    report["components_evidence"] = {
        "path": str(components_path),
        "symbol_count": len(required_symbols),
    }
    report["backfill_cost_estimate"] = estimate_historical_bar_backfill_cost(
        missing_trade_dates=int(args.missing_trade_dates),
        unresolved_symbol_dates=int(args.unresolved_symbol_dates),
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":  # pragma: no cover
    main()
