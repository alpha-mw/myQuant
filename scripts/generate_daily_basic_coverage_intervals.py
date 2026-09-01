"""Generate the v2 daily_basic coverage declaration from the run's own data.

Beijing Stock Exchange symbols carry no Tushare daily_basic before the exchange
opened on 2021-11-15, while the five-year rebuild window opens 2021-08-31. That
gap is a property of the provider, not of the companies, and the audit needs it
declared or it treats every one of those symbols as an incomplete fetch.

Every value here is measured from the staged table this run actually fetched —
no date is carried over from the retired v1 declaration, whose reason was right
but whose boundary date was two weeks early.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/Users/maxwell/mySpace/myQuant")

from quant_investor.factors.pit_fundamentals import normalize_ts_code
from quant_investor.market.fundamental_mart import build_canonical_scope_evidence
from quant_investor.market.fundamental_provider_contract import canonical_json_sha256

REASON = "PROVIDER_COVERAGE_BOUNDARY"
AUTHORITY = "PROVIDER_METADATA_RECEIPT"
SCHEMA = "daily-basic-coverage-intervals.v2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged-daily", required=True)
    ap.add_argument("--scope", required=True)
    ap.add_argument("--market-pointer", required=True)
    ap.add_argument("--membership", required=True)
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--daily-start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--outcomes",
        required=True,
        help="request_outcomes.json; only symbols the audit flagged are declared",
    )
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    daily_path = Path(args.staged_daily)
    raw = daily_path.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    daily = pd.read_parquet(daily_path, columns=["ts_code", "trade_date"])
    daily["ts_code"] = daily["ts_code"].map(normalize_ts_code)
    daily["trade_date"] = daily["trade_date"].astype(str)

    # Declare only what the audit actually flagged. Asserting a provider
    # boundary for a symbol the audit is content with would be inventing
    # authority, which is how the retired v1 file went wrong.
    outcomes = json.loads(Path(args.outcomes).read_text())["outcomes"]
    flagged = {
        normalize_ts_code(row["symbol"])
        for row in outcomes
        if row.get("table") == "daily_basic" and row.get("history_complete") is not True
    }
    # The scope evidence must be built over the whole canonical scope — it
    # verifies the symbol set — so filter only when emitting intervals.
    symbols = sorted(daily["ts_code"].dropna().unique())
    evidence = build_canonical_scope_evidence(
        symbols,
        canonical_path=args.scope,
        market_pointer_path=args.market_pointer,
        membership_path=args.membership,
        as_of=args.as_of,
        daily_start=args.daily_start,
    )
    identities = dict(evidence.get("listing_identities", {}) or {})
    listing_dates = dict(evidence.get("listing_dates", {}) or {})
    history_ends = dict(evidence.get("history_end_dates", {}) or {})

    window_start = args.daily_start
    observed_first = (
        daily[daily["ts_code"].isin(flagged)].groupby("ts_code")["trade_date"].min()
    )

    intervals: list[dict[str, str]] = []
    skipped: list[str] = []
    for symbol, first in observed_first.items():
        if first <= window_start:
            continue  # nothing missing at the front of the window
        listing_start = str(listing_dates.get(symbol) or "")
        listing_end = str(history_ends.get(symbol) or "")
        identity = str(identities.get(symbol) or "")
        if not (listing_start and listing_end and identity):
            skipped.append(f"{symbol}: missing listing evidence")
            continue
        # The gap runs from the later of the window and the listing to the day
        # before the provider's first observation.
        effective_from = max(window_start, listing_start)
        effective_to = (
            pd.Timestamp(first) - pd.Timedelta(days=1)
        ).strftime("%Y%m%d")
        if effective_from > effective_to:
            skipped.append(f"{symbol}: listing starts inside the covered gap")
            continue
        if effective_to > min(listing_end, args.as_of):
            effective_to = min(listing_end, args.as_of)
        canonical = {
            "symbol": symbol,
            "listing_identity": identity,
            "reason": REASON,
            "authority": AUTHORITY,
            "effective_from": effective_from,
            "effective_to": effective_to,
            "available_at": args.as_of,
            "cutoff": args.as_of,
            "source_sha256": source_sha,
        }
        intervals.append({"interval_id": canonical_json_sha256(canonical), **canonical})

    intervals.sort(key=lambda row: row["interval_id"])
    record = {"schema_version": SCHEMA, "intervals": intervals}
    payload = {**record, "record_sha256": canonical_json_sha256(record)}

    summary = {
        "symbols_flagged_by_audit": len(flagged),
        "intervals_declared": len(intervals),
        "skipped": skipped,
        "source_table": str(daily_path),
        "source_sha256": source_sha,
        "window_start": window_start,
        "cutoff": args.as_of,
        "effective_to_values": sorted({row["effective_to"] for row in intervals}),
        "effective_from_values": sorted({row["effective_from"] for row in intervals}),
        "sample": intervals[:2],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.write:
        Path(args.out).write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        )
        print(f"\nwritten: {args.out}")
    else:
        print("\n(dry run; pass --write to persist)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
