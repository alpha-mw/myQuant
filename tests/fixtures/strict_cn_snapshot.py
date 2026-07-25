from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

from quant_investor.market.pit_universe import PITUniverseRecord, PITUniverseStore


def symbol_set_sha256(symbols: Sequence[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(str(symbol).strip().upper() for symbol in symbols)).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def v4_snapshot_paths(data_root: Path, snapshot_id: str) -> tuple[Path, Path, Path]:
    snapshot_root = data_root / "parquet" / "cn" / "_snapshots"
    return (
        snapshot_root / snapshot_id / "table" / "bars",
        snapshot_root / snapshot_id / "serving" / "bars",
        snapshot_root / f"{snapshot_id}.json",
    )


def write_components(data_root: Path, symbols: Sequence[str]) -> None:
    payload = {
        "full_a": [str(symbol).strip().upper() for symbol in symbols],
        "hs300": [],
        "zz500": [],
        "zz1000": [],
        "stats": {"total_unique": len({str(symbol).strip().upper() for symbol in symbols})},
    }
    write_json(data_root / "cn_universe" / "cn_index_components.json", payload)


def write_pit_generation(
    data_root: Path,
    symbols: Sequence[str],
    *,
    observed_at: str | None = None,
) -> dict:
    normalized = [str(symbol).strip().upper() for symbol in symbols]
    symbol_hash = hashlib.sha256("\n".join(sorted(normalized)).encode("utf-8")).hexdigest()
    if observed_at is None:
        observed_at = f"2026-01-01T00:00:{int(symbol_hash[:2], 16) % 60:02d}Z"
    source_run_id = f"test-strict-cn-snapshot-{symbol_hash[:16]}"
    store = PITUniverseStore(root_dir=data_root / "parquet" / "cn" / "reference")
    records = [
        PITUniverseRecord(
            symbol=str(symbol).strip().upper(),
            name=f"{str(symbol).strip().upper()} fixture",
            source_list_status="L",
            list_date="20000101",
            effective_from="20000101",
            effective_to="",
            observed_at=observed_at,
            source_run_id=source_run_id,
            raw_payload_hash=hashlib.sha256(
                str(symbol).strip().upper().encode("utf-8")
            ).hexdigest()[:16],
        )
        for symbol in normalized
    ]
    return store.write_snapshot(
        raw_records=records,
        observed_at=observed_at,
        source_run_id=source_run_id,
    )


def coverage_v4(
    data_root: Path,
    symbols: Sequence[str],
    *,
    trade_date: str,
    observed_bar_count: int | None = None,
) -> dict:
    normalized = [str(symbol).strip().upper() for symbol in symbols]
    write_components(data_root, normalized)
    pit = write_pit_generation(data_root, normalized)
    count = len(normalized)
    observed = count if observed_bar_count is None else int(observed_bar_count)
    return {
        "coverage_schema_version": "cn-full-a-coverage.v4",
        "complete": True,
        "categories_checked": ["full_a"],
        "coverage_trade_date": str(trade_date),
        "latest_available_trade_date": str(trade_date),
        "latest_complete_trade_date": str(trade_date),
        "expected_scope_count": count,
        "expected_scope_sha256": symbol_set_sha256(normalized),
        "observed_bar_count": observed,
        "coverage_complete_count": observed,
        "coverage_ratio": 1.0 if count == observed else (observed / count if count else 0.0),
        "blocking_incomplete_count": 0,
        "suspended_symbols": [],
        "inactive_symbols": [],
        "verified_terminal_delisting_symbols": [],
        "verified_nontrading_bak_daily_zero_symbols": [],
        "allowed_stale_symbols": [],
        "non_blocking_absent_symbols": [],
        "true_missing_symbols": [],
        "classification_sets_disjoint": True,
        "pit_generation_id": str(pit["generation_id"]),
        "pit_generation_manifest_path": str(pit["generation_manifest_path"]),
        "pit_generation_manifest_sha256": str(pit["generation_manifest_sha256"]),
        "pit_membership_path": str(pit["canonical_path"]),
        "pit_membership_sha256": str(pit["canonical_sha256"]),
    }
