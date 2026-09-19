"""Collect per-trade-date suspension evidence for the historical sample.

Two jobs, one contract:

  * 600068.SH, which was a member for a handful of days and traded on none of
    them — the case the admission rule exists for;
  * the days on which 600311.SH and 000004.SZ were members and yet have no bar,
    which the day-count reconciliation identified but did not explain.

A date with no verifying suspension record stays "unexplained". Closing a count
is not the same as explaining it, and this script will not report the former as
though it were the latter.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from quant_investor.market.cn_research_suspension_evidence import (
    SuspensionEvidenceError,
    build_suspension_evidence,
    expected_trading_days,
    verify_suspension_evidence,
)


def _provider():
    import tushare as ts

    from quant_investor import config as C
    from quant_investor.config import TUSHARE_OFFICIAL_URL

    token = os.environ.get("TUSHARE_TOKEN") or getattr(C, "TUSHARE_TOKEN", "")
    pro = ts.pro_api(token, timeout=60)
    pro._DataApi__http_url = TUSHARE_OFFICIAL_URL
    return pro


def _suspend_payload(pro, trade_date: str, seen: dict, failures: dict) -> dict | None:
    """One query per trade date, cached; two failures of a kind stops the run."""
    if trade_date in seen:
        return seen[trade_date]
    try:
        frame = pro.suspend_d(trade_date=trade_date)
    except Exception as exc:  # noqa: BLE001 - classified, then surfaced
        kind = type(exc).__name__
        failures[kind] = failures.get(kind, 0) + 1
        if failures[kind] >= 2:
            raise SystemExit(
                f"suspend_d failed twice with {kind}; stopping rather than retrying blindly"
            ) from exc
        time.sleep(1.0)
        return _suspend_payload(pro, trade_date, seen, failures)
    rows = [
        {
            "ts_code": str(row.get("ts_code") or ""),
            "trade_date": str(row.get("trade_date") or ""),
            "suspend_type": str(row.get("suspend_type") or ""),
        }
        for _, row in frame.iterrows()
    ] if len(frame) else []
    payload = {"query_params": {"trade_date": trade_date}, "rows": rows}
    seen[trade_date] = payload
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window-start", default="20210904")
    ap.add_argument("--window-end", default="20260904")
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pointer = json.loads((root / "parquet" / "cn" / "_latest.json").read_text())
    membership_path = Path(pointer["coverage"]["pit_membership_path"])
    membership = pd.read_parquet(membership_path)
    membership_ref = {
        "path": str(membership_path),
        "sha256": pointer["coverage"]["pit_membership_sha256"],
    }

    pro = _provider()
    calendar = pro.trade_cal(
        exchange="SSE", start_date=args.window_start, end_date=args.window_end
    )
    open_dates = sorted(
        str(row["cal_date"])
        for _, row in calendar.iterrows()
        if int(row["is_open"]) == 1
    )
    calendar_ref = {
        "api": "tushare.trade_cal",
        "exchange": "SSE",
        "query_params": {
            "start_date": args.window_start,
            "end_date": args.window_end,
        },
        "open_day_count": len(open_dates),
    }

    seen: dict[str, dict] = {}
    failures: dict[str, int] = {}
    results: dict[str, dict] = {}

    # --- 600068.SH: member throughout, traded on nothing -------------------
    row = membership[membership["symbol"] == "600068.SH"]
    if len(row):
        record = row.iloc[0]
        window = expected_trading_days(
            symbol="600068.SH",
            effective_from=str(record["effective_from"]),
            effective_to=str(record["effective_to"] or ""),
            window_start=args.window_start,
            window_end=args.window_end,
            calendar_open_dates=open_dates,
        )
        payloads = {
            date: _suspend_payload(pro, date, seen, failures)
            for date in window.trade_dates
        }
        document = build_suspension_evidence(
            window=window,
            per_date_payloads=payloads,
            calendar_reference=calendar_ref,
            membership_reference=membership_ref,
        )
        path = out_dir / "suspension_evidence_600068.SH.json"
        path.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")
        try:
            verified = verify_suspension_evidence(
                document,
                symbol="600068.SH",
                expected_trade_dates=window.trade_dates,
                per_date_payloads=payloads,
            )
            results["600068.SH"] = {
                "status": "VERIFIED",
                "expected_days": len(window.trade_dates),
                "expected_window": [window.window_start, window.window_end],
                "verified": verified,
                "evidence_path": str(path),
            }
        except SuspensionEvidenceError as exc:
            results["600068.SH"] = {
                "status": "REJECTED",
                "reason": str(exc),
                "expected_days": len(window.trade_dates),
                "unexplained_dates": document["unexplained_dates"],
                "evidence_path": str(path),
            }

    # --- the member-but-no-bar days for the two delisted identities --------
    cache_path = root / "provider_price_cache.parquet"
    if not cache_path.exists():
        # The gap analysis needs the sample's own prices; without them the
        # 600068-style admission evidence above still stands on its own.
        report = {
            "schema_version": "cn-research-suspension-evidence-report.v1",
            "research_only": True,
            "calendar_reference": calendar_ref,
            "membership_reference": membership_ref,
            "results": results,
            "suspend_d_queries": len(seen),
            "gap_analysis": "skipped: provider_price_cache.parquet absent",
        }
        (out_dir / "suspension_evidence_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=1)[:1800])
        return 0
    price_cache = pd.read_parquet(cache_path)
    price_cache["trade_date"] = price_cache["trade_date"].astype(str)
    priced = {
        symbol: set(group["trade_date"])
        for symbol, group in price_cache.groupby("ts_code")
    }
    for symbol in ("600311.SH", "000004.SZ"):
        record = membership[membership["symbol"] == symbol].iloc[0]
        window = expected_trading_days(
            symbol=symbol,
            effective_from=str(record["effective_from"]),
            effective_to=str(record["effective_to"] or ""),
            window_start=args.window_start,
            window_end=args.window_end,
            calendar_open_dates=open_dates,
        )
        gap_dates = [
            date for date in window.trade_dates if date not in priced.get(symbol, set())
        ]
        payloads = {
            date: _suspend_payload(pro, date, seen, failures) for date in gap_dates
        }
        explained, unexplained = [], []
        for date in gap_dates:
            rows = [
                item
                for item in payloads[date]["rows"]
                if item["ts_code"].upper() == symbol
                and item["suspend_type"].upper() == "S"
            ]
            (explained if rows else unexplained).append(date)
        results[symbol] = {
            "member_days": len(window.trade_dates),
            "member_days_without_a_bar": len(gap_dates),
            "explained_by_verified_suspension": len(explained),
            "unexplained_missing": len(unexplained),
            "unexplained_dates": unexplained,
            "explained_dates_sample": explained[:5],
            "note": (
                "unexplained days remain unexplained; a closed day count is not "
                "coverage"
            ),
        }

    report = {
        "schema_version": "cn-research-suspension-evidence-report.v1",
        "research_only": True,
        "calendar_reference": calendar_ref,
        "membership_reference": membership_ref,
        "results": results,
        "suspend_d_queries": len(seen),
    }
    (out_dir / "suspension_evidence_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=1)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
