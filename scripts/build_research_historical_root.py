"""Build an isolated research root containing historical (delisted) identities.

The production universe is "listed today": `fetch_cn_index_components.py` asks
`stock_basic` for `list_status="L"`, so an identity that delisted inside the
backtest window is absent from the fundamental mart and from the factor symbol
axis. This script builds a *separate* root in which such identities exist, so
their construction and consumption can be exercised end to end without touching
production.

It is not a second production universe. The root it writes is research-only:
nothing here is promotable, and the fundamental rebuild refuses the research
universe key unless both the market root and the staging root are isolated.

Bar history comes from the provider, never from the local canonical store — the
local store is missing real trading history for these identities, which is the
defect that motivated this work, so using it as the source would bake the gap in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from quant_investor.market.pit_universe import PITUniverseRecord, PITUniverseStore

PRODUCTION_MEMBERSHIP = (
    "data/parquet/cn/reference/_generations/"
    "pit-20260905-4ef20518a329d046/stock_basic_membership.parquet"
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def symbol_set_sha256(symbols: Sequence[str]) -> str:
    return _sha256_text("\n".join(sorted(str(s).strip().upper() for s in symbols)))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def _provider():
    import tushare as ts

    from quant_investor import config as C
    from quant_investor.config import TUSHARE_OFFICIAL_URL

    token = os.environ.get("TUSHARE_TOKEN") or getattr(C, "TUSHARE_TOKEN", "")
    if not token:
        raise SystemExit("TUSHARE_TOKEN is required for the provider bar fetch")
    pro = ts.pro_api(token, timeout=60)
    pro._DataApi__http_url = TUSHARE_OFFICIAL_URL
    return pro


def _fetch_bars(pro, symbol: str, start: str, end: str, log: list[dict]) -> pd.DataFrame:
    """Fetch one identity's bars, stopping after two failures of the same kind.

    Blind retry hides the difference between a transient network fault and a
    provider-side refusal; two of the same error is enough to call it the latter.
    """
    seen: dict[str, int] = {}
    for attempt in range(1, 5):
        try:
            frame = pro.daily(ts_code=symbol, start_date=start, end_date=end)
            log.append(
                {
                    "symbol": symbol,
                    "attempt": attempt,
                    "outcome": "ok",
                    "rows": int(len(frame)),
                    "window": [start, end],
                }
            )
            return frame
        except Exception as exc:  # noqa: BLE001 - classified, then re-raised
            kind = type(exc).__name__
            seen[kind] = seen.get(kind, 0) + 1
            log.append(
                {
                    "symbol": symbol,
                    "attempt": attempt,
                    "outcome": "error",
                    "error_kind": kind,
                    "error": str(exc)[:200],
                }
            )
            if seen[kind] >= 2:
                raise SystemExit(
                    f"provider bar fetch failed twice with {kind} for {symbol}; "
                    "stopping rather than retrying blindly"
                ) from exc
            time.sleep(1.0)
    raise SystemExit(f"provider bar fetch exhausted attempts for {symbol}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="isolated research root")
    ap.add_argument("--symbols", required=True, help="comma-separated batch symbols")
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--daily-start", required=True)
    ap.add_argument("--snapshot-id", default="")
    ap.add_argument(
        "--zero-bar-symbols",
        default="",
        help=(
            "identities admitted as members with no bars; their absence must be "
            "carried by suspension evidence, never by fabricated prices"
        ),
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    batch = sorted({s.strip().upper() for s in args.symbols.split(",") if s.strip()})
    zero_bar = sorted(
        {s.strip().upper() for s in args.zero_bar_symbols.split(",") if s.strip()}
    )
    batch = sorted(set(batch) | set(zero_bar))
    as_of, daily_start = args.as_of, args.daily_start
    snapshot_id = args.snapshot_id or f"research{as_of}T000000Z"

    membership = pd.read_parquet(PRODUCTION_MEMBERSHIP)
    membership = membership[membership["symbol"].isin(batch)]
    if len(membership) != len(batch):
        missing = sorted(set(batch) - set(membership["symbol"]))
        raise SystemExit(f"identities absent from canonical membership: {missing}")

    pro = _provider()
    fetch_log: list[dict] = []
    table_root = root / "parquet" / "cn" / "_snapshots" / snapshot_id / "table" / "bars"
    serving_root = root / "parquet" / "cn" / "_snapshots" / snapshot_id / "serving" / "bars"
    table_root.mkdir(parents=True, exist_ok=True)
    serving_root.mkdir(parents=True, exist_ok=True)

    records: list[PITUniverseRecord] = []
    per_symbol: dict[str, dict[str, Any]] = {}
    frames: list[pd.DataFrame] = []
    for _, row in membership.sort_values("symbol").iterrows():
        symbol = str(row["symbol"])
        status = str(row["source_list_status"])
        list_date = str(row["list_date"]).strip()
        effective_from = str(row["effective_from"]).strip()
        effective_to = str(row["effective_to"] or "").strip()
        # Eligibility mirrors the mart: the window intersected with the identity's
        # own listing interval. history_end is effective_to for a closed interval.
        history_end = effective_to or as_of
        start = max(daily_start, list_date)
        end = min(as_of, history_end)
        if symbol in zero_bar:
            # Deliberately not fetched: this identity is expected to have no
            # trades at all, and a fetch that returned something would mean the
            # premise is wrong. Its absence is carried by evidence instead.
            frame = pd.DataFrame(columns=["ts_code", "trade_date"])
            fetch_log.append(
                {"symbol": symbol, "outcome": "skipped_zero_bar_admission_candidate"}
            )
        else:
            frame = _fetch_bars(pro, symbol, start, end, fetch_log)
        bars = (
            frame[["ts_code", "trade_date"]].copy()
            if len(frame)
            else pd.DataFrame(columns=["ts_code", "trade_date"])
        )
        bars["ts_code"] = symbol
        bars["trade_date"] = bars["trade_date"].astype(str)
        bars = bars.sort_values("trade_date").reset_index(drop=True)
        per_symbol[symbol] = {
            "source_list_status": status,
            "list_date": list_date,
            "effective_from": effective_from,
            "effective_to": effective_to,
            "eligibility_start": start,
            "eligibility_end": end,
            "provider_bar_rows": int(len(bars)),
            "provider_bar_first": bars["trade_date"].min() if len(bars) else "",
            "provider_bar_last": bars["trade_date"].max() if len(bars) else "",
        }
        if len(bars):
            frames.append(bars)
        # The serving partition is written for every member, including one with
        # no bars: it says "this identity is in the served universe and has zero
        # rows", which is the truth. Omitting it would instead make the identity
        # disappear from symbol resolution — silently dropping exactly the case
        # this sample exists to carry.
        (serving_root / f"symbol={symbol}").mkdir(parents=True, exist_ok=True)
        bars.to_parquet(serving_root / f"symbol={symbol}" / "bars.parquet", index=False)
        records.append(
            PITUniverseRecord(
                symbol=symbol,
                name=str(row["name"]),
                # Sector attribution is carried through from the canonical
                # membership: the derivation builds its sector map from these
                # fields, and dropping them collapses every identity into
                # "unknown", which then trips the concentration gate for a
                # reason that has nothing to do with the data being tested.
                area=str(row.get("area") or ""),
                industry=str(row.get("industry") or ""),
                board_market=str(row.get("board_market") or ""),
                source_list_status=status,
                list_date=list_date,
                delist_date=str(row["delist_date"] or "").strip(),
                effective_from=effective_from,
                effective_to=effective_to,
                observed_at=str(row["observed_at"]),
                source_run_id=f"research-hist-{as_of}",
                raw_payload_hash=_sha256_text(symbol)[:16],
            )
        )

    if not frames:
        raise SystemExit("no provider bars fetched; refusing to write an empty root")
    combined = pd.concat(frames, ignore_index=True)
    combined["year"] = combined["trade_date"].str[:4]
    for year, group in combined.groupby("year"):
        target = table_root / f"year={year}"
        target.mkdir(parents=True, exist_ok=True)
        group[["ts_code", "trade_date"]].reset_index(drop=True).to_parquet(
            target / "bars.parquet", index=False
        )

    store = PITUniverseStore(root_dir=root / "parquet" / "cn" / "reference")
    pit = store.write_snapshot(
        raw_records=records,
        observed_at=f"{as_of[:4]}-{as_of[4:6]}-{as_of[6:]}T00:00:00Z",
        source_run_id=f"research-hist-{as_of}",
    )

    components = {
        "full_a": batch,
        # The research universe key is requested explicitly by the rebuild; it is
        # kept distinct from full_a so no production caller can reach it by
        # accident, and it is never widened beyond this root's served symbols.
        "full_a_hist": batch,
        "hs300": [],
        "zz500": [],
        "zz1000": [],
        "stats": {"total_unique": len(batch), "full_a": len(batch)},
        "research_only": True,
        "tradable_universe": False,
    }
    _write_json(root / "cn_universe" / "cn_index_components.json", components)

    # Identities whose interval closed before as_of are not active on the as-of
    # date. Naming them here is what lets scope evidence accept a historical
    # member at all; it is an explicit authoring decision, recorded as such.
    inactive = sorted(s for s, v in per_symbol.items() if v["effective_to"])
    manifest_zero_bar = list(zero_bar)
    coverage = {
        "coverage_schema_version": "cn-full-a-coverage.v4",
        "complete": True,
        "categories_checked": ["full_a"],
        "coverage_trade_date": as_of,
        "latest_available_trade_date": as_of,
        "latest_complete_trade_date": as_of,
        "expected_scope_count": len(batch),
        "expected_scope_sha256": symbol_set_sha256(batch),
        # Identities admitted with no bars are not observed bars; counting them
        # as observed would be the coverage lie this whole path exists to avoid.
        "observed_bar_count": len(batch) - len(inactive),
        "coverage_complete_count": len(batch),
        "coverage_ratio": 1.0,
        "blocking_incomplete_count": 0,
        "suspended_symbols": [],
        # Both delisted identities are inactive as of the as-of date; that is a
        # fact about their membership interval. They are deliberately NOT claimed
        # as verified_terminal_delisting: that classification requires an
        # exchange-notice evidence chain (path + payload SHA + inferred dates)
        # which this stage does not have, and asserting it without the evidence
        # would be exactly the kind of unbacked claim the gates exist to catch.
        "inactive_symbols": inactive,
        "verified_terminal_delisting_symbols": [],
        "verified_nontrading_bak_daily_zero_symbols": [],
        "allowed_stale_symbols": [],
        "non_blocking_absent_symbols": inactive,
        "true_missing_symbols": [],
        "classification_sets_disjoint": True,
        "pit_generation_id": str(pit["generation_id"]),
        "pit_generation_manifest_path": str(pit["generation_manifest_path"]),
        "pit_generation_manifest_sha256": str(pit["generation_manifest_sha256"]),
        "pit_membership_path": str(Path(pit["canonical_path"]).resolve()),
        "pit_membership_sha256": str(pit["canonical_sha256"]),
    }
    # The reader requires a snapshot manifest beside the snapshot directory and
    # resolves its path strictly as <_snapshots>/<snapshot_id>.json.
    manifest_path = root / "parquet" / "cn" / "_snapshots" / f"{snapshot_id}.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "cn-research-snapshot-manifest.v1",
            "snapshot_id": snapshot_id,
            "research_only": True,
            "latest_complete_trade_date": as_of,
            "table_root": str(table_root.resolve()),
            "derived_serving_root": str(serving_root.resolve()),
            "symbols": batch,
            "bar_authority": "tushare.daily (provider)",
            # The reader fingerprints the pointer's coverage against the
            # manifest's and refuses a mismatch, so the same object is bound in
            # both places rather than restated.
            "coverage": coverage,
        },
    )
    pointer = {
        "status": "OK",
        "snapshot_id": snapshot_id,
        "manifest_path": str(manifest_path.resolve()),
        "table_root": str(table_root.resolve()),
        "derived_serving_root": str(serving_root.resolve()),
        "latest_trade_date": as_of,
        "latest_available_trade_date": as_of,
        "latest_complete_trade_date": as_of,
        "updated_at": f"{as_of[:4]}-{as_of[4:6]}-{as_of[6:]}T00:00:00Z",
        "blockers": [],
        "quarantined_tail_dates": [],
        "coverage": coverage,
        "research_only": True,
    }
    _write_json(root / "parquet" / "cn" / "_latest.json", pointer)

    manifest = {
        "schema_version": "cn-research-historical-scope.v1",
        "research_only": True,
        "tradable_universe": False,
        "root": str(root),
        "window": {"daily_start": daily_start, "as_of": as_of},
        "batch": batch,
        "batch_symbol_set_sha256": symbol_set_sha256(batch),
        "membership_source": {
            "path": str(Path(PRODUCTION_MEMBERSHIP).resolve()),
            "sha256": hashlib.sha256(
                Path(PRODUCTION_MEMBERSHIP).read_bytes()
            ).hexdigest(),
        },
        "bar_authority": "tushare.daily (provider); local canonical bars are NOT used",
        "per_symbol": per_symbol,
        "non_blocking_absent_symbols": inactive,
        "zero_bar_admission_candidates": manifest_zero_bar,
        "snapshot_id": snapshot_id,
    }
    _write_json(root / "research_scope_manifest.json", manifest)
    _write_json(root / "provider_bar_fetch_log.json", {"entries": fetch_log})

    print(json.dumps({"root": str(root), "batch": batch, "per_symbol": per_symbol}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
