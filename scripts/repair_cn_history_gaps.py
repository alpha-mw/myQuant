#!/usr/bin/env python3
"""Repair up to five proven CN historical symbol-date bar gaps atomically."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from quant_investor.config import config
from quant_investor.market.cn_history_audit import _read_canonical_window
from quant_investor.market.cn_nontrading_evidence import (
    canonical_json_sha256,
    file_sha256,
)
from quant_investor.market.cn_terminal_delisting_evidence import (
    read_terminal_delisting_evidence,
    terminal_delist_dates,
)
from quant_investor.market.download import CNParquetBatchMaintainer
from quant_investor.market.market_data_reader import (
    MarketDataReader,
    coverage_fingerprint,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", default="CN", choices=["CN"])
    parser.add_argument("--audit-path", required=True)
    parser.add_argument("--trade-dates", nargs="+", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="data/cn_market_full")
    parser.add_argument("--allow-online", action="store_true")
    return parser.parse_args(argv)


def _compact_date(value: object) -> str:
    digits = "".join(character for character in str(value or "") if character.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _validate_source_audit_payload(audit: dict) -> None:
    work = dict(audit)
    declared_sha256 = str(work.pop("audit_sha256", "") or "")
    if declared_sha256 != canonical_json_sha256(work):
        raise SystemExit("Source audit payload SHA is invalid.")
    if audit.get("schema_version") != "myquant-cn-history-audit.v4":
        raise SystemExit("Source audit schema is not supported.")
    if audit.get("audit_method") != "full_recompute_from_canonical":
        raise SystemExit("Source audit is not a full canonical recomputation.")
    if audit.get("full_window_recomputed") is not True:
        raise SystemExit("Source audit did not recompute the full window.")
    if audit.get("prior_trade_dates_reused") != 0:
        raise SystemExit("Source audit reused prior date classifications.")
    if audit.get("audited_trade_dates_count") != 100 or len(
        audit.get("audited_trade_dates", []) or []
    ) != 100:
        raise SystemExit("Source audit date count is inconsistent.")
    canonical = audit.get("canonical", {}) or {}
    window = audit.get("canonical_window_evidence", {}) or {}
    pit = audit.get("pit_membership_evidence", {}) or {}
    if (canonical.get("storage_validation", {}) or {}).get("status") != "passed":
        raise SystemExit("Source audit canonical validation did not pass.")
    if window.get("table_serving_match") is not True:
        raise SystemExit("Source audit table/serving evidence did not match.")
    if not canonical.get("snapshot_id") or not pit.get("sha256"):
        raise SystemExit("Source audit provenance binding is incomplete.")


def _stale_target_keys(
    current_keys: set[tuple[str, str]],
    expected_by_date: dict[str, list[str]],
) -> list[tuple[str, str]]:
    return sorted(
        (trade_date, symbol)
        for trade_date, symbols in expected_by_date.items()
        for symbol in symbols
        if (trade_date, symbol) in current_keys
    )


def _validate_active_coverage_pit_binding(
    *,
    binding: Mapping[str, Any],
    coverage: Mapping[str, Any],
    audit_pit: Mapping[str, Any],
) -> None:
    if binding.get("status") != "passed":
        raise SystemExit(
            "Active market coverage PIT binding is invalid: "
            + ",".join(binding.get("blockers", []) or [])
        )
    expected_path = str(coverage.get("pit_membership_path") or "").strip()
    expected_sha256 = str(
        coverage.get("pit_membership_sha256") or ""
    ).strip().lower()
    if (
        str(audit_pit.get("path") or "").strip() != expected_path
        or str(audit_pit.get("sha256") or "").strip().lower()
        != expected_sha256
        or str(binding.get("canonical_sha256") or "").strip().lower()
        != expected_sha256
    ):
        raise SystemExit("Source audit PIT binding is stale.")

    coverage_schema = str(
        coverage.get("coverage_schema_version") or ""
    ).strip()
    if coverage_schema != "cn-full-a-coverage.v4":
        return
    expected_generation = {
        "coverage_schema_version": coverage_schema,
        "generation_id": str(coverage.get("pit_generation_id") or "").strip(),
        "generation_manifest_path": str(
            coverage.get("pit_generation_manifest_path") or ""
        ).strip(),
        "generation_manifest_sha256": str(
            coverage.get("pit_generation_manifest_sha256") or ""
        ).strip().lower(),
    }
    observed_generation = {
        "coverage_schema_version": str(
            audit_pit.get("coverage_schema_version") or ""
        ).strip(),
        "generation_id": str(audit_pit.get("generation_id") or "").strip(),
        "generation_manifest_path": str(
            audit_pit.get("generation_manifest_path") or ""
        ).strip(),
        "generation_manifest_sha256": str(
            audit_pit.get("generation_manifest_sha256") or ""
        ).strip().lower(),
    }
    if observed_generation != expected_generation:
        raise SystemExit("Source audit PIT generation binding is stale.")


def main(argv: Sequence[str] | None = None) -> dict:
    args = _parse_args(argv)
    if not args.allow_online:
        raise SystemExit("--allow-online is required for exact historical repair calls.")
    dates = sorted({_compact_date(value) for value in args.trade_dates if _compact_date(value)})
    if not dates or len(dates) > 5:
        raise SystemExit("Provide between one and five unique trade dates.")
    audit_path = Path(args.audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    _validate_source_audit_payload(audit)
    per_date = {
        str(row.get("trade_date") or ""): row
        for row in audit.get("per_date", []) or []
        if isinstance(row, dict)
    }
    expected_by_date: dict[str, list[str]] = {}
    for trade_date in dates:
        row = per_date.get(trade_date)
        if row is None:
            raise SystemExit(f"Trade date is not in the audit: {trade_date}")
        expected = sorted(
            {
                str(symbol or "").strip().upper()
                for symbol in row.get("true_missing_symbols", []) or []
                if str(symbol or "").strip()
            }
        )
        if not expected:
            raise SystemExit(f"Audit has no true missing symbols for {trade_date}")
        expected_by_date[trade_date] = expected

    if not str(getattr(config, "TUSHARE_TOKEN", "") or "").strip():
        raise SystemExit("TUSHARE_TOKEN is not configured.")
    maintainer = CNParquetBatchMaintainer(
        data_dir=str(Path(args.output_root)),
        data_root=Path(args.data_root),
    )
    store = maintainer.store
    before_validation = store.validate_latest()
    if before_validation.get("status") != "passed":
        raise SystemExit("Latest canonical is not healthy; historical repair refused.")
    latest_path = Path(args.data_root) / "parquet" / "cn" / "_latest.json"
    before_latest = json.loads(latest_path.read_text(encoding="utf-8"))
    before_latest_sha256 = file_sha256(latest_path)
    audit_canonical = audit.get("canonical", {}) or {}
    if before_latest.get("snapshot_id") != audit_canonical.get("snapshot_id"):
        raise SystemExit(
            "Source audit snapshot is stale; rerun the full history audit."
        )
    if before_latest_sha256 != audit_canonical.get("latest_sha256"):
        raise SystemExit("Source audit latest-pointer binding is stale.")
    manifest_path = Path(str(before_latest.get("manifest_path") or ""))
    if not manifest_path.is_absolute():
        manifest_path = Path.cwd() / manifest_path
    if (
        str(manifest_path) != str(audit_canonical.get("manifest_path") or "")
        or file_sha256(manifest_path)
        != audit_canonical.get("manifest_sha256")
    ):
        raise SystemExit("Source audit manifest binding is stale.")
    reader = MarketDataReader(
        market="CN",
        data_root=Path(args.data_root),
        mode_policy="strict",
    )
    pit_binding = reader.coverage_bound_pit(refresh=True)
    if (
        reader._load_latest_payload() != before_latest
        or file_sha256(latest_path) != before_latest_sha256
    ):
        raise SystemExit("Active market pointer changed during repair preflight.")
    audit_pit = audit.get("pit_membership_evidence", {}) or {}
    active_coverage = before_latest.get("coverage", {}) or {}
    if not isinstance(active_coverage, Mapping):
        raise SystemExit("Active market coverage is invalid.")
    _validate_active_coverage_pit_binding(
        binding=pit_binding,
        coverage=active_coverage,
        audit_pit=audit_pit,
    )
    audit_terminal = audit.get("terminal_delisting_evidence", {}) or {}
    terminal_symbols = sorted(
        {
            str(symbol or "").strip().upper()
            for symbol in audit_terminal.get("symbols", []) or []
            if str(symbol or "").strip()
        }
    )
    if terminal_symbols:
        raw_terminal_path = str(audit_terminal.get("path") or "").strip()
        if not raw_terminal_path:
            raise SystemExit(
                "Source audit terminal-delisting path is missing."
            )
        terminal_path = Path(raw_terminal_path)
        if (
            not terminal_path.exists()
            or file_sha256(terminal_path)
            != str(audit_terminal.get("file_sha256") or "")
        ):
            raise SystemExit(
                "Source audit terminal-delisting file binding is stale."
            )
        terminal_payload, terminal_blockers = read_terminal_delisting_evidence(
            terminal_path,
            target_trade_date=str(audit.get("effective_trade_date") or ""),
            candidate_symbols=terminal_symbols,
            pit_membership_path=str(audit_pit.get("path") or ""),
            pit_membership_sha256=str(audit_pit.get("sha256") or ""),
        )
        if terminal_blockers:
            raise SystemExit(
                "Source audit terminal-delisting evidence is stale: "
                + ",".join(terminal_blockers)
            )
        if (
            str(terminal_payload.get("payload_sha256") or "")
            != str(audit_terminal.get("payload_sha256") or "")
            or terminal_delist_dates(terminal_payload)
            != dict(audit_terminal.get("inferred_delist_dates", {}) or {})
        ):
            raise SystemExit(
                "Source audit terminal-delisting payload binding is stale."
            )
        latest_terminal_symbols = sorted(
            {
                str(symbol or "").strip().upper()
                for symbol in before_latest.get("coverage", {}).get(
                    "verified_terminal_delisting_symbols", []
                )
                or []
                if str(symbol or "").strip()
            }
        )
        if latest_terminal_symbols != terminal_symbols:
            raise SystemExit(
                "Source audit terminal-delisting coverage binding is stale."
            )
    selected_dates = [
        _compact_date(value)
        for value in audit.get("audited_trade_dates", []) or []
        if _compact_date(value)
    ]
    snapshot = reader._require_snapshot()
    canonical_bars, current_window = _read_canonical_window(
        reader,
        table_root=snapshot.table_root,
        serving_root=snapshot.serving_root,
        start_date=selected_dates[0],
        end_date=selected_dates[-1],
        selected_dates=selected_dates,
    )
    audit_window = audit.get("canonical_window_evidence", {}) or {}
    for key in ("table_sha256", "serving_sha256", "table_row_count"):
        if current_window.get(key) != audit_window.get(key):
            raise SystemExit(
                f"Source audit canonical window is stale: {key}."
            )
    current_keys = set(
        zip(canonical_bars["trade_date"], canonical_bars["ts_code"])
    )
    stale_target_keys = _stale_target_keys(
        current_keys,
        expected_by_date,
    )
    if stale_target_keys:
        raise SystemExit(
            "Source audit gap has already been filled; rerun the full audit: "
            f"{stale_target_keys[:20]}"
        )
    latest_complete = _compact_date(before_latest.get("latest_complete_trade_date"))
    if not latest_complete or any(trade_date >= latest_complete for trade_date in dates):
        raise SystemExit("Every repair date must be earlier than latest_complete_trade_date.")
    before_coverage = dict(before_latest.get("coverage", {}) or {})
    before_fingerprint = coverage_fingerprint(before_coverage)

    repaired_frames: list[pd.DataFrame] = []
    endpoint_evidence: dict[str, dict[str, object]] = {}
    for trade_date in dates:
        daily, daily_error = maintainer._fetch_endpoint(
            "daily",
            trade_date,
            "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
        )
        adj, adj_error = maintainer._fetch_endpoint(
            "adj_factor",
            trade_date,
            "ts_code,trade_date,adj_factor",
        )
        daily_basic, daily_basic_error = maintainer._fetch_endpoint(
            "daily_basic",
            trade_date,
            "ts_code,trade_date,turnover_rate,volume_ratio,pe,pb,total_mv,circ_mv",
        )
        if daily_error or adj_error:
            raise SystemExit(
                f"Endpoint failure for {trade_date}: daily={daily_error}, adj={adj_error}"
            )
        expected = set(expected_by_date[trade_date])
        bars = maintainer._build_bars_frame(daily, adj, daily_basic)
        repaired = bars.loc[bars["ts_code"].isin(expected)].copy()
        observed = set(repaired.get("ts_code", pd.Series(dtype=str)).astype(str))
        if observed != expected:
            raise SystemExit(
                f"Repair rows do not match audit gap for {trade_date}: "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        repaired_frames.append(repaired)
        endpoint_evidence[trade_date] = {
            "expected_symbols": sorted(expected),
            "repaired_row_count": int(len(repaired)),
            "daily_row_count": int(len(daily)),
            "adj_factor_row_count": int(len(adj)),
            "daily_basic_row_count": int(len(daily_basic)),
            "daily_basic_warning": str(daily_basic_error or ""),
        }

    incoming = pd.concat(repaired_frames, ignore_index=True, sort=False)
    manifest = store.upsert_bars(
        incoming,
        target_trade_date=max(dates),
        target_trade_dates=dates,
        source="cn_history_gap_bounded_repair",
        metadata={
            "status": "OK",
            "latest_available_trade_date": latest_complete,
            "latest_complete_trade_date": latest_complete,
            "coverage": {
                "coverage_trade_date": max(dates),
                "historical_repair_trade_dates": dates,
                "historical_repair_row_count": int(len(incoming)),
            },
            "historical_repair_trade_dates": dates,
            "historical_repair_source_audit": str(audit_path),
            "blockers": [],
        },
        expected_latest_pointer_sha256=before_latest_sha256,
    )
    after_validation = store.validate_latest()
    after_latest = json.loads(latest_path.read_text(encoding="utf-8"))
    after_fingerprint = coverage_fingerprint(after_latest.get("coverage", {}))
    if after_validation.get("status") != "passed":
        raise SystemExit("Post-repair storage validation failed.")
    if after_fingerprint != before_fingerprint:
        raise SystemExit("Latest coverage fingerprint changed during historical repair.")
    if _compact_date(after_latest.get("latest_complete_trade_date")) != latest_complete:
        raise SystemExit("Latest complete trade date changed during historical repair.")

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    report = {
        "schema_version": "cn-history-gap-bounded-repair.v1",
        "generated_at": generated_at,
        "market": args.market,
        "source_audit_path": str(audit_path),
        "repair_trade_dates": dates,
        "repair_trade_date_count": len(dates),
        "repaired_row_count": int(len(incoming)),
        "endpoint_evidence": endpoint_evidence,
        "before_coverage_fingerprint": before_fingerprint,
        "after_coverage_fingerprint": after_fingerprint,
        "latest_coverage_unchanged": True,
        "latest_complete_trade_date": latest_complete,
        "snapshot_id": manifest.get("snapshot_id"),
        "manifest_path": manifest.get("manifest_path"),
        "historical_upsert_coverage_preserved": manifest.get(
            "historical_upsert_coverage_preserved"
        ),
        "storage_validation": after_validation,
        "synthetic_bar_count": 0,
        "no_analysis_or_trading_side_effects": True,
    }
    output_path = Path(args.output_root) / (
        "history_repair_"
        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        + ".json"
    )
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "repair_trade_dates": dates,
                "repaired_row_count": int(len(incoming)),
                "latest_coverage_unchanged": True,
                "latest_complete_trade_date": latest_complete,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return report


if __name__ == "__main__":  # pragma: no cover
    main()
