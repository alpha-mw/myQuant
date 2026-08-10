from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, cast

import pandas as pd
import pytest

import quant_investor.market.fundamental_generation as generation
import quant_investor.market.fundamental_mart as fundamental_mart
from quant_investor.market.fundamental_generation import (
    FundamentalGenerationError,
    load_fundamental_pointer,
    pointer_sha256,
    preflight_staged_fundamental_promotion,
    promote_staged_fundamental_generation,
    publish_fundamental_generation,
)
from quant_investor.market.fundamental_provider_contract import (
    FUNDAMENTAL_DERIVATION_CONTRACT,
    FUNDAMENTAL_FETCH_CHECKPOINT_SCHEMA,
    FUNDAMENTAL_FETCH_PIT_CONTRACT,
    FUNDAMENTAL_PROVIDER_MANIFEST_SCHEMA,
    FUNDAMENTAL_REQUEST_OUTCOME_SCHEMA,
    canonical_json_sha256,
    frame_fingerprint,
    frame_logical_schema,
)


def _tables(symbol: str) -> dict[str, pd.DataFrame]:
    trade_dates = pd.bdate_range("2023-05-10", "2024-05-10")
    return {
        "fundamental_period": pd.DataFrame(
            [
                {
                    "ts_code": symbol,
                    "end_date": "20221231",
                    "availability_date": "20230501",
                }
            ]
        ),
        "fundamental_daily": pd.DataFrame(
            {
                "ts_code": symbol,
                "trade_date": trade_dates.strftime("%Y%m%d"),
                "end_date": "20221231",
                "availability_date": "20230501",
            }
        ),
        "fundamental_quarantine": pd.DataFrame(
            columns=["ts_code", "quarantine_reason"]
        ),
    }


def _raw_tables(symbols: list[str]) -> dict[str, pd.DataFrame]:
    financial_rows: dict[str, list[dict[str, object]]] = {
        table: []
        for table in ("fina_indicator", "income", "balancesheet", "cashflow")
    }
    daily_rows: list[dict[str, object]] = []
    forecast_rows: list[dict[str, object]] = []
    trade_dates = pd.bdate_range("2023-05-10", "2024-05-10")
    for symbol_index, symbol in enumerate(symbols):
        financial_rows["fina_indicator"].append(
            {
                "ts_code": symbol,
                "ann_date": "20230501",
                "end_date": "20221231",
                "roe_dt": 10.0 + symbol_index,
                "roe": 10.0 + symbol_index,
                "roa": 5.0,
                "debt_to_assets": 40.0,
                "netprofit_yoy": 12.0,
            }
        )
        financial_rows["income"].append(
            {
                "ts_code": symbol,
                "ann_date": "20230501",
                "f_ann_date": "",
                "end_date": "20221231",
                "n_income": 100.0,
                "n_income_attr_p": 90.0,
                "update_flag": "1",
            }
        )
        financial_rows["balancesheet"].append(
            {
                "ts_code": symbol,
                "ann_date": "20230501",
                "f_ann_date": "",
                "end_date": "20221231",
                "total_liab": 400.0,
                "total_assets": 1000.0,
                "update_flag": "1",
            }
        )
        financial_rows["cashflow"].append(
            {
                "ts_code": symbol,
                "ann_date": "20230501",
                "f_ann_date": "",
                "end_date": "20221231",
                "n_cashflow_act": 120.0,
                "c_pay_acq_const_fiolta": 20.0,
                "free_cashflow": 100.0,
                "update_flag": "1",
            }
        )
        for period_index, (period, announcement) in enumerate(
            (
                ("20230630", "20231028"),
                ("20230930", "20240128"),
                ("20231231", "20240429"),
            ),
            start=1,
        ):
            financial_rows["fina_indicator"].append(
                {
                    "ts_code": symbol,
                    "ann_date": announcement,
                    "end_date": period,
                    "roe_dt": 10.0 + symbol_index + period_index,
                    "roe": 10.0 + symbol_index + period_index,
                    "roa": 5.0,
                    "debt_to_assets": 40.0,
                    "netprofit_yoy": 12.0,
                }
            )
            financial_rows["income"].append(
                {
                    "ts_code": symbol,
                    "ann_date": announcement,
                    "f_ann_date": "",
                    "end_date": period,
                    "n_income": 100.0 + period_index,
                    "n_income_attr_p": 90.0 + period_index,
                    "update_flag": "1",
                }
            )
            financial_rows["balancesheet"].append(
                {
                    "ts_code": symbol,
                    "ann_date": announcement,
                    "f_ann_date": "",
                    "end_date": period,
                    "total_liab": 400.0,
                    "total_assets": 1000.0,
                    "update_flag": "1",
                }
            )
            financial_rows["cashflow"].append(
                {
                    "ts_code": symbol,
                    "ann_date": announcement,
                    "f_ann_date": "",
                    "end_date": period,
                    "n_cashflow_act": 120.0,
                    "c_pay_acq_const_fiolta": 20.0,
                    "free_cashflow": 100.0,
                    "update_flag": "1",
                }
            )
        daily_rows.extend(
            {
                "ts_code": symbol,
                "trade_date": trade_date.strftime("%Y%m%d"),
                "total_mv": 100_000.0 + symbol_index * 10_000.0,
                "circ_mv": 80_000.0,
                "pe": 10.0,
                "pb": 1.5,
            }
            for trade_date in trade_dates
        )
        forecast_rows.append(
            {
                "ts_code": symbol,
                "ann_date": "20230505",
                "end_date": "20231231",
                "type": "预增",
                "p_change_min": 10.0,
                "p_change_max": 20.0,
                "net_profit_min": 110.0,
                "net_profit_max": 120.0,
                "last_parent_net": 100.0,
                "summary": "fixture forecast",
                "change_reason": "fixture",
                "update_flag": "1",
            }
        )
    return {
        **{table: pd.DataFrame(rows) for table, rows in financial_rows.items()},
        "daily_basic": pd.DataFrame(daily_rows),
        "forecast": pd.DataFrame(forecast_rows),
    }


def _clean_outcomes(
    raw_tables: dict[str, pd.DataFrame],
    symbols: list[str],
) -> list[dict[str, object]]:
    zero_fields = {
        "rows_hard_invalid": 0,
        "rows_filtered_future": 0,
        "rows_filtered_missing_availability": 0,
        "rows_filtered_core_values": 0,
        "rows_deduplicated": 0,
        "rows_discarded_request_malformed": 0,
        "rows_hard_invalid_schema": 0,
        "rows_hard_invalid_symbol": 0,
        "rows_hard_invalid_availability_date": 0,
        "rows_hard_invalid_end_date": 0,
        "rows_hard_invalid_end_after_availability": 0,
        "rows_hard_invalid_core_numeric": 0,
    }
    outcomes: list[dict[str, object]] = []
    for symbol in symbols:
        for table in generation.FUNDAMENTAL_RAW_TABLES:
            rows = int(
                raw_tables[table]["ts_code"].astype(str).eq(symbol).sum()
            )
            outcome: dict[str, object] = {
                "schema_version": FUNDAMENTAL_REQUEST_OUTCOME_SCHEMA,
                "symbol": symbol,
                "table": table,
                "status": "success",
                "error": "",
                "attempts": 1,
                "provider_calls": 1,
                "rows_received": rows,
                "rows": rows,
                **zero_fields,
            }
            if table == "daily_basic":
                outcome["history_complete"] = True
            outcomes.append(outcome)
    return sorted(
        outcomes,
        key=lambda item: (str(item["symbol"]), str(item["table"])),
    )


def _publish_offline(
    root: Path,
    *,
    run_id: str = "canonical-old",
    expected_pointer_sha256: str | None = None,
) -> None:
    publish_fundamental_generation(
        root=root,
        run_id=run_id,
        tables=_tables("000001.SZ"),
        metadata={
            "run_id": run_id,
            "source_priority": "manual_offline_snapshot",
            "gate2_passed": True,
        },
        expected_pointer_sha256=expected_pointer_sha256,
    )


def _publish_verified_primary(
    root: Path,
    *,
    run_id: str = "verified-stage",
    tables_override: dict[str, pd.DataFrame] | None = None,
    market_pointer_override: dict[str, object] | None = None,
    scope_exception_symbols: list[str] | None = None,
    pointer_exception_symbols: list[str] | None = None,
    membership_override: pd.DataFrame | None = None,
    requested_daily_start: str = "20230510",
    requested_financial_start: str = "20210510",
    canonical_bar_start: str = "20230510",
    expected_pointer_sha256: str | None = None,
) -> None:
    symbols = sorted(
        set(
            (
                tables_override["fundamental_daily"]["ts_code"].astype(str).tolist()
                if tables_override is not None
                else ["000002.SZ"]
            )
        )
    )
    evidence_dir = root / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    scope_path = evidence_dir / "scope.json"
    market_pointer_path = evidence_dir / "market_latest.json"
    membership_path = evidence_dir / "membership.parquet"
    scope_path.write_text(json.dumps({"full_a": symbols}), encoding="utf-8")
    membership = membership_override
    if membership is None:
        membership = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "list_date": "20230510",
                    "effective_from": "20230510",
                    "effective_to": "",
                    "industry": "fixture-sector",
                }
                for symbol in symbols
            ]
        )
    elif "industry" not in membership.columns:
        membership = membership.assign(industry="fixture-sector")
    membership.to_parquet(membership_path, index=False)
    symbol_set_sha256 = hashlib.sha256("\n".join(symbols).encode()).hexdigest()
    bars_root = evidence_dir / "canonical-bars"
    bars_root.mkdir(exist_ok=True)
    pd.DataFrame(
        [
            {
                "ts_code": symbol,
                "trade_date": trade_date.strftime("%Y%m%d"),
            }
            for symbol in symbols
            for trade_date in pd.bdate_range(canonical_bar_start, "2024-05-10")
        ]
    ).to_parquet(bars_root / "part.parquet", index=False)
    market_pointer_payload: dict[str, object] = (
        dict(market_pointer_override)
        if market_pointer_override is not None
        else {
            "snapshot_id": "fixture-market-snapshot",
            "latest_complete_trade_date": "20240510",
            "table_root": str(bars_root.resolve()),
            "coverage": {
                "expected_scope_count": len(symbols),
                "expected_scope_sha256": symbol_set_sha256,
                "non_blocking_absent_symbols": list(
                    pointer_exception_symbols or []
                ),
                "pit_membership_path": str(membership_path.resolve()),
                "pit_membership_sha256": hashlib.sha256(
                    membership_path.read_bytes()
                ).hexdigest(),
            },
        }
    )
    market_pointer_payload.setdefault("table_root", str(bars_root.resolve()))
    market_pointer_path.write_text(
        json.dumps(market_pointer_payload),
        encoding="utf-8",
    )
    membership_by_symbol = membership.assign(
        _symbol=membership["symbol"].astype("string").str.strip().str.upper()
    ).set_index("_symbol")
    listing_dates = {
        symbol: str(membership_by_symbol.at[symbol, "list_date"]).strip()
        for symbol in symbols
    }
    history_end_dates = {symbol: "20240510" for symbol in symbols}
    membership_sha256 = hashlib.sha256(membership_path.read_bytes()).hexdigest()
    listing_identities = {
        symbol: canonical_json_sha256(
            {
                "symbol": symbol,
                "listing_date": listing_dates[symbol],
                "effective_from": str(
                    membership_by_symbol.at[symbol, "effective_from"]
                ).strip(),
                "history_end": history_end_dates[symbol],
                "membership_sha256": membership_sha256,
            }
        )
        for symbol in symbols
    }
    canonical_bar_first_dates = {symbol: canonical_bar_start for symbol in symbols}
    canonical_bar_last_dates = {symbol: "20240510" for symbol in symbols}
    bar_file_evidence = fundamental_mart._canonical_bar_file_evidence(
        bars_root.resolve()
    )
    bar_bounds_sha = hashlib.sha256(
        "\n".join(
            f"{symbol}|{canonical_bar_first_dates[symbol]}|{canonical_bar_last_dates[symbol]}"
            for symbol in symbols
        ).encode()
    ).hexdigest()
    eligibility_sha = hashlib.sha256(
        "\n".join(
            "|".join(
                (
                    symbol,
                    listing_dates[symbol],
                    history_end_dates[symbol],
                    canonical_bar_first_dates[symbol],
                    canonical_bar_last_dates[symbol],
                )
            )
            for symbol in symbols
        ).encode()
    ).hexdigest()
    canonical_scope_evidence = {
        "canonical_path": str(scope_path.resolve()),
        "canonical_file_sha256": hashlib.sha256(scope_path.read_bytes()).hexdigest(),
        "canonical_market_pointer_path": str(market_pointer_path.resolve()),
        "canonical_market_pointer_sha256": hashlib.sha256(
            market_pointer_path.read_bytes()
        ).hexdigest(),
        "canonical_market_snapshot_id": "fixture-market-snapshot",
        "canonical_market_trade_date": "20240510",
        "canonical_membership_path": str(membership_path.resolve()),
        "canonical_membership_sha256": membership_sha256,
        "symbol_count": len(symbols),
        "symbol_set_sha256": symbol_set_sha256,
        "listing_dates": listing_dates,
        "history_end_dates": history_end_dates,
        "listing_identities": listing_identities,
        "canonical_bar_first_dates": canonical_bar_first_dates,
        "canonical_bar_last_dates": canonical_bar_last_dates,
        "canonical_bar_table_root": str(bars_root.resolve()),
        "canonical_bar_file_count": len(bar_file_evidence),
        "canonical_bar_files_sha256": canonical_json_sha256(bar_file_evidence),
        "canonical_bar_bounds_sha256": bar_bounds_sha,
        "daily_history_coverage_interval_path": "",
        "daily_history_coverage_interval_source_sha256": "",
        "daily_history_coverage_intervals": [],
        "canonical_bar_daily_start": requested_daily_start,
        "canonical_bar_as_of": "20240510",
        "history_eligibility_sha256": eligibility_sha,
        "non_blocking_absent_symbols": list(scope_exception_symbols or []),
    }
    raw_tables = _raw_tables(symbols)
    outcomes = _clean_outcomes(raw_tables, symbols)
    audit_policy = fundamental_mart.FundamentalEndpointAuditPolicy()
    outcomes = fundamental_mart._attach_daily_history_coverage(
        symbols,
        outcomes,
        raw_tables,
        daily_start=requested_daily_start,
        as_of="20240510",
        scope_evidence=canonical_scope_evidence,
        policy=audit_policy,
    )
    outcomes = fundamental_mart._attach_financial_coverage(
        symbols,
        outcomes,
        raw_tables,
        financial_start=requested_financial_start,
        as_of="20240510",
        scope_evidence=canonical_scope_evidence,
        policy=audit_policy,
    )
    binding = fundamental_mart._checkpoint_binding(
        symbols=symbols,
        years=1,
        start_date=requested_daily_start,
        financial_start_date=requested_financial_start,
        as_of="20240510",
        canonical_scope_evidence=canonical_scope_evidence,
    )
    checkpoint_root = fundamental_mart._safe_checkpoint_root(
        evidence_dir / "checkpoint"
    )
    checkpoint = fundamental_mart._write_fetch_checkpoint(
        checkpoint_root,
        binding=binding,
        tables=raw_tables,
        outcomes=outcomes,
        expected_pointer_sha256="",
        expected_revision=0,
    )
    raw_tables = checkpoint.tables
    outcomes = checkpoint.outcomes
    membership_bytes = membership_path.read_bytes()
    derivation_timestamp = "2024-05-11T00:00:00Z"
    try:
        derived_tables, derivation_evidence = (
            fundamental_mart.rederive_fundamental_tables_v3(
                raw_tables,
                membership_bytes=membership_bytes,
                membership_sha256=hashlib.sha256(membership_bytes).hexdigest(),
                as_of="20240510",
                symbols=symbols,
                non_blocking_absent_symbols=list(scope_exception_symbols or []),
                run_id=run_id,
                source="live_tushare",
                derivation_timestamp=derivation_timestamp,
            )
        )
    except (TypeError, ValueError):
        # A few legacy negative fixtures intentionally bind malformed membership.
        # They are rejected by the earlier membership gate, before replay.
        derived_tables = {
            "fundamental_period": pd.concat(
                [_tables(symbol)["fundamental_period"] for symbol in symbols],
                ignore_index=True,
            ),
            "fundamental_daily": pd.concat(
                [_tables(symbol)["fundamental_daily"] for symbol in symbols],
                ignore_index=True,
            ),
            "fundamental_quarantine": pd.DataFrame(
                columns=["ts_code", "quarantine_reason"]
            ),
        }
        derivation_evidence = {
            "contract_version": FUNDAMENTAL_DERIVATION_CONTRACT,
            "membership_sha256": hashlib.sha256(membership_bytes).hexdigest(),
            "as_of": "20240510",
            "selection_rule": (
                "latest_active_membership_interval_as_of_else_latest_expired"
            ),
            "selected_symbol_count": len(symbols),
            "expired_fallback_symbol_count": 0,
            "expired_fallback_symbols_sha256": hashlib.sha256(b"").hexdigest(),
            "sector_map_sha256": canonical_json_sha256(
                {symbol: "fixture-sector" for symbol in symbols}
            ),
            "derivation_timestamp": derivation_timestamp,
            "run_id": run_id,
            "source": "live_tushare",
            "raw_table_fingerprints": {
                table: frame_fingerprint(raw_tables[table])
                for table in generation.FUNDAMENTAL_RAW_TABLES
            },
            "output_frame_fingerprints": {
                table: frame_fingerprint(frame)
                for table, frame in derived_tables.items()
            },
        }
    tables = tables_override or derived_tables
    selection_rule = "latest_active_membership_interval_as_of_else_latest_expired"
    derivation = {
        "contract_version": FUNDAMENTAL_DERIVATION_CONTRACT,
        "pit_membership_path": str(membership_path.resolve()),
        "pit_membership_sha256": hashlib.sha256(membership_bytes).hexdigest(),
        "sector_selection_rule": selection_rule,
        **derivation_evidence,
    }
    endpoint_audit = fundamental_mart._build_endpoint_audit(
        symbols,
        outcomes,
        policy=audit_policy,
        daily_basic_empty_exception_symbols=list(scope_exception_symbols or []),
    )
    provider_manifest = {
        "schema_version": FUNDAMENTAL_PROVIDER_MANIFEST_SCHEMA,
        "provider": "tushare",
        "run_id": run_id,
        "authoritative_full_rebuild": True,
        "pit_contract_version": FUNDAMENTAL_FETCH_PIT_CONTRACT,
        "request_fields": dict(fundamental_mart.SOURCE_REQUEST_FIELDS),
        "strict_pit_as_of": "20240510",
        "daily_start_date": requested_daily_start,
        "financial_start_date": requested_financial_start,
        "years": 1,
        "requests_attempted": len(symbols) * len(generation.FUNDAMENTAL_RAW_TABLES),
        "requests_succeeded_with_rows": len(outcomes),
        "requests_empty": 0,
        "requests_failed": 0,
        "requests_malformed": 0,
        "symbol_table_outcomes": outcomes,
        "request_outcome_accounting_sha256": canonical_json_sha256(outcomes),
        "raw_row_counts": {
            table: len(raw_tables[table])
            for table in generation.FUNDAMENTAL_RAW_TABLES
        },
        "raw_table_fingerprints": {
            table: frame_fingerprint(raw_tables[table])
            for table in generation.FUNDAMENTAL_RAW_TABLES
        },
        "canonical_scope_evidence": canonical_scope_evidence,
        "checkpoint": {
            "schema_version": FUNDAMENTAL_FETCH_CHECKPOINT_SCHEMA,
            "root": str(checkpoint_root),
            "generation_id": checkpoint.generation_id,
            "revision": checkpoint.revision,
            "pointer_sha256": checkpoint.pointer_sha256,
            "manifest_sha256": checkpoint.manifest_sha256,
            "binding_sha256": checkpoint.binding_sha256,
            "outcome_accounting_sha256": checkpoint.outcome_accounting_sha256,
            "table_evidence_sha256": checkpoint.table_evidence_sha256,
        },
        "derivation": derivation,
        "raw_to_derived_binding_sha256": canonical_json_sha256(derivation),
        "endpoint_audit": endpoint_audit,
    }
    metadata = {
        "run_id": run_id,
        "provider_status": "live_tushare",
        "source_priority": "tushare_primary",
        "source_provenance": "live_tushare_explicit",
        "provider_manifest": provider_manifest,
        "gate2_passed": True,
    }
    attestation = generation._issue_primary_generation_attestation(
        tables=tables,
        metadata=metadata,
        source="live_tushare",
        provider_manifest_sha256=generation._metadata_sha256(provider_manifest),
        raw_table_fingerprints=tuple(
            (
                name,
                frame_fingerprint(raw_tables[name]),
            )
            for name in generation.FUNDAMENTAL_RAW_TABLES
        ),
    )
    publish_fundamental_generation(
        root=root,
        run_id=run_id,
        tables=tables,
        metadata=metadata,
        _primary_attestation=attestation,
        expected_pointer_sha256=expected_pointer_sha256,
    )


def _write_canonical_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _resign_staged_primary(
    root: Path,
    pointer: dict[str, object],
    manifest: dict[str, object],
) -> None:
    metadata = dict(cast(dict[str, Any], manifest["metadata"]))
    provider = dict(cast(dict[str, Any], metadata["provider_manifest"]))
    table_manifest = dict(cast(dict[str, Any], manifest["tables"]))
    envelope = dict(cast(dict[str, Any], manifest["primary_provenance"]))
    envelope.pop("envelope_sha256", None)
    envelope["provider_manifest_sha256"] = generation._metadata_sha256(provider)
    envelope["metadata_sha256"] = generation._metadata_sha256(metadata)
    envelope["raw_table_fingerprints"] = dict(
        provider["raw_table_fingerprints"]
    )
    envelope["output_frame_fingerprints"] = {
        table: dict(table_manifest[table])["frame_fingerprint"]
        for table in generation.FUNDAMENTAL_TABLES
    }
    envelope["output_parquet_sha256"] = {
        table: dict(table_manifest[table])["sha256"]
        for table in generation.FUNDAMENTAL_TABLES
    }
    envelope["envelope_sha256"] = generation._metadata_sha256(envelope)
    manifest["primary_provenance"] = envelope
    pointer["primary_provenance"] = envelope
    manifest_path = root / str(pointer["manifest_path"])
    _write_canonical_json(manifest_path, manifest)
    _write_canonical_json(
        root / generation.FUNDAMENTAL_POINTER_FILENAME,
        pointer,
    )


def _mutate_provider_and_resign(
    root: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    pointer_path = root / generation.FUNDAMENTAL_POINTER_FILENAME
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    manifest_path = root / pointer["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = dict(manifest["metadata"])
    provider = dict(metadata["provider_manifest"])
    mutate(provider)
    metadata["provider_manifest"] = provider
    manifest["metadata"] = metadata
    _resign_staged_primary(root, pointer, manifest)


def _mutate_derived_table_and_resign(
    root: Path,
    table_name: str,
    mutate: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    pointer_path = root / generation.FUNDAMENTAL_POINTER_FILENAME
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    manifest_path = root / pointer["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    table_path = root / dict(pointer["tables"])[table_name]
    frame = pd.read_parquet(table_path)
    updated = mutate(frame.copy())
    updated.to_parquet(table_path, index=False)
    payload = table_path.read_bytes()
    readback = pd.read_parquet(table_path)
    table_manifest = dict(manifest["tables"])
    table_manifest[table_name] = {
        "rows": len(readback),
        "columns": list(readback.columns),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "frame_fingerprint": frame_fingerprint(readback),
        "logical_schema": frame_logical_schema(readback),
    }
    manifest["tables"] = table_manifest
    _resign_staged_primary(root, pointer, manifest)


def _mutate_checkpoint_raw_and_resign(
    root: Path,
    table_name: str,
    mutate: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    mutate_outcomes: (
        Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None
    ) = None,
) -> None:
    staged = load_fundamental_pointer(root)
    assert staged is not None
    provider = dict(staged["manifest"]["metadata"]["provider_manifest"])
    captured = generation._capture_provider_checkpoint_v3(provider)
    raw_tables = {
        name: frame.copy() for name, frame in captured.tables.items()
    }
    raw_tables[table_name] = mutate(raw_tables[table_name].copy())
    outcomes = [dict(outcome) for outcome in captured.outcomes]
    if mutate_outcomes is not None:
        outcomes = mutate_outcomes(outcomes)
    checkpoint_pointer = json.loads(
        (captured.root / "latest.json").read_text(encoding="utf-8")
    )
    checkpoint_manifest = json.loads(
        (captured.root / checkpoint_pointer["manifest_path"]).read_text(
            encoding="utf-8"
        )
    )
    forged_checkpoint = fundamental_mart._write_fetch_checkpoint(
        captured.root,
        binding=checkpoint_manifest["binding"],
        tables=raw_tables,
        outcomes=outcomes,
        expected_pointer_sha256=captured.pointer_sha256,
        expected_revision=captured.revision,
    )

    def bind_checkpoint(provider_value: dict[str, Any]) -> None:
        provider_value["symbol_table_outcomes"] = forged_checkpoint.outcomes
        provider_value["request_outcome_accounting_sha256"] = (
            forged_checkpoint.outcome_accounting_sha256
        )
        provider_value["raw_row_counts"] = {
            name: len(forged_checkpoint.tables[name])
            for name in generation.FUNDAMENTAL_RAW_TABLES
        }
        fingerprints = {
            name: frame_fingerprint(forged_checkpoint.tables[name])
            for name in generation.FUNDAMENTAL_RAW_TABLES
        }
        provider_value["raw_table_fingerprints"] = fingerprints
        provider_value["checkpoint"] = {
            **dict(provider_value["checkpoint"]),
            "generation_id": forged_checkpoint.generation_id,
            "revision": forged_checkpoint.revision,
            "pointer_sha256": forged_checkpoint.pointer_sha256,
            "manifest_sha256": forged_checkpoint.manifest_sha256,
            "binding_sha256": forged_checkpoint.binding_sha256,
            "outcome_accounting_sha256": (
                forged_checkpoint.outcome_accounting_sha256
            ),
            "table_evidence_sha256": forged_checkpoint.table_evidence_sha256,
        }
        derivation = dict(provider_value["derivation"])
        derivation["raw_table_fingerprints"] = fingerprints
        provider_value["derivation"] = derivation
        provider_value["raw_to_derived_binding_sha256"] = canonical_json_sha256(
            derivation
        )

    _mutate_provider_and_resign(root, bind_checkpoint)


def test_real_v3_checkpoint_writer_evidence_promotes(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)

    staged_pointer = json.loads(
        (staging / generation.FUNDAMENTAL_POINTER_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    staged_manifest = json.loads(
        (staging / staged_pointer["manifest_path"]).read_text(encoding="utf-8")
    )
    provider = dict(staged_manifest["metadata"]["provider_manifest"])
    checkpoint = dict(provider["checkpoint"])
    checkpoint_root = Path(checkpoint["root"])
    checkpoint_pointer = json.loads(
        (checkpoint_root / "latest.json").read_text(encoding="utf-8")
    )
    checkpoint_manifest = json.loads(
        (checkpoint_root / checkpoint_pointer["manifest_path"]).read_text(
            encoding="utf-8"
        )
    )
    exact_table_evidence_sha256 = canonical_json_sha256(
        dict(checkpoint_manifest["table_files"])
    )
    assert checkpoint_manifest["table_evidence_sha256"] == exact_table_evidence_sha256
    assert checkpoint["table_evidence_sha256"] == exact_table_evidence_sha256

    result = promote_staged_fundamental_generation(
        staging_root=staging,
        canonical_root=canonical,
        expected_pointer_sha256=pointer_sha256(canonical),
    )

    assert result["promoted"] is True


def test_promotion_rejects_bound_canonical_bar_file_drift(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)
    staged_pointer = json.loads(
        (staging / generation.FUNDAMENTAL_POINTER_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    staged_manifest = json.loads(
        (staging / staged_pointer["manifest_path"]).read_text(encoding="utf-8")
    )
    scope = staged_manifest["metadata"]["provider_manifest"][
        "canonical_scope_evidence"
    ]
    bars_root = Path(scope["canonical_bar_table_root"])
    bars = pd.read_parquet(bars_root / "part.parquet")
    bars.loc[0, "trade_date"] = "20230511"
    bars.to_parquet(bars_root / "part.parquet", index=False)

    with pytest.raises(
        FundamentalGenerationError,
        match="scope evidence changed|bar dataset",
    ):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256=pointer_sha256(canonical),
        )


@pytest.mark.parametrize("listing_date", ["19961217", "20210831"])
def test_promotion_uses_bound_canonical_bar_start_for_long_history_gap(
    tmp_path: Path,
    listing_date: str,
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(
        staging,
        membership_override=pd.DataFrame(
            [
                {
                    "symbol": "000002.SZ",
                    "list_date": listing_date,
                    "effective_from": listing_date,
                    "effective_to": "",
                    "industry": "fixture-sector",
                }
            ]
        ),
        requested_daily_start="20220101",
        requested_financial_start="20230630",
        canonical_bar_start="20230510",
    )

    result = promote_staged_fundamental_generation(
        staging_root=staging,
        canonical_root=canonical,
        expected_pointer_sha256=pointer_sha256(canonical),
    )

    assert result["promoted"] is True


def test_promote_verified_primary_generation_with_cas(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)
    before = pointer_sha256(canonical)
    old_generation = canonical / generation.FUNDAMENTAL_GENERATIONS_DIRNAME / "canonical-old"

    result = promote_staged_fundamental_generation(
        staging_root=staging,
        canonical_root=canonical,
        expected_pointer_sha256=before,
    )

    installed = load_fundamental_pointer(canonical)
    assert result["promoted"] is True
    assert result["previous_pointer_sha256"] == before
    assert result["pointer_sha256"] == pointer_sha256(canonical)
    assert installed is not None
    assert installed["generation_id"] == "verified-stage"
    assert installed["primary_provenance_verified"] is True
    assert installed["metadata"]["gate2_passed"] is True
    assert old_generation.is_dir()
    lock = canonical / generation.FUNDAMENTAL_PROMOTION_LOCK_FILENAME
    assert lock.is_file() and not lock.is_symlink()
    assert os.stat(lock).st_mode & 0o077 == 0


def test_optional_phase_recorder_observes_commit_boundaries(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)
    events: list[tuple[str, dict[str, Any]]] = []

    result = promote_staged_fundamental_generation(
        staging_root=staging,
        canonical_root=canonical,
        expected_pointer_sha256=pointer_sha256(canonical),
        phase_recorder=lambda event, payload: events.append(
            (event, dict(payload))
        ),
    )

    assert result["promoted"] is True
    assert [event for event, _payload in events] == [
        "PRECAS_VALIDATED",
        "CAS_COMMITTED",
        "POSTCHECK_PASSED",
    ]
    assert events[1][1]["pointer_sha256"] == result["pointer_sha256"]


def test_promotion_preflight_is_read_only_and_matches_commit_target(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)
    before = pointer_sha256(canonical)

    preflight = preflight_staged_fundamental_promotion(
        staging_root=staging,
        canonical_root=canonical,
        expected_pointer_sha256=before,
    )

    assert pointer_sha256(canonical) == before
    assert preflight["candidate_generation_id"] == "verified-stage"
    assert preflight["provider_evidence"] is None
    result = promote_staged_fundamental_generation(
        staging_root=staging,
        canonical_root=canonical,
        expected_pointer_sha256=before,
    )
    assert result["pointer_sha256"] == preflight["candidate_pointer_sha256"]


def test_successful_promotion_rederives_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)
    original = fundamental_mart.rederive_fundamental_tables_v3
    calls = 0

    def count_rederive(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        fundamental_mart,
        "rederive_fundamental_tables_v3",
        count_rederive,
    )
    result = promote_staged_fundamental_generation(
        staging_root=staging,
        canonical_root=canonical,
        expected_pointer_sha256=pointer_sha256(canonical),
    )

    assert result["promoted"] is True
    assert calls == 1


def test_promotion_rejects_v2_provider_contract_before_pointer_change(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)
    before = pointer_sha256(canonical)

    def downgrade(provider: dict[str, Any]) -> None:
        provider["pit_contract_version"] = "myquant-fundamental-fetch-pit.v2"

    _mutate_provider_and_resign(staging, downgrade)
    with pytest.raises(FundamentalGenerationError, match="PIT contract"):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256=before,
        )
    assert pointer_sha256(canonical) == before


@pytest.mark.parametrize(
    ("table_name", "mutate"),
    [
        pytest.param(
            "forecast",
            lambda frame: frame.assign(ann_date=""),
            id="missing-availability",
        ),
        pytest.param(
            "daily_basic",
            lambda frame: frame.assign(total_mv=0.0),
            id="core-unusable",
        ),
    ],
)
def test_promotion_rejects_resigned_checkpoint_accepted_raw_laundering(
    tmp_path: Path,
    table_name: str,
    mutate: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)
    before = pointer_sha256(canonical)
    _mutate_checkpoint_raw_and_resign(staging, table_name, mutate)

    with pytest.raises(
        FundamentalGenerationError,
        match="accepted raw contains rejected rows",
    ):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256=before,
        )
    assert pointer_sha256(canonical) == before


def test_promotion_preserves_clean_provider_filter_accounting(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)
    before = pointer_sha256(canonical)

    def inflate_received_rows(
        outcomes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        for outcome in outcomes:
            if outcome["table"] == "forecast":
                outcome["rows_received"] = int(outcome["rows_received"]) + 1
                outcome["rows_filtered_future"] = 1
                break
        return outcomes

    _mutate_checkpoint_raw_and_resign(
        staging,
        "forecast",
        lambda frame: frame,
        mutate_outcomes=inflate_received_rows,
    )
    result = promote_staged_fundamental_generation(
        staging_root=staging,
        canonical_root=canonical,
        expected_pointer_sha256=before,
    )
    assert result["promoted"] is True


def test_promotion_rejects_provider_weakened_financial_policy(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)
    before = pointer_sha256(canonical)

    def weaken_financial_policy(provider: dict[str, Any]) -> None:
        audit = dict(provider["endpoint_audit"])
        policy = dict(audit["policy"])
        policy.update(
            {
                "financial_period_min_coverage_ratio": 0.0,
                "financial_max_consecutive_missing_baseline_periods": 99,
                "financial_require_latest_baseline": False,
            }
        )
        audit["policy"] = policy
        provider["endpoint_audit"] = audit

    _mutate_provider_and_resign(staging, weaken_financial_policy)
    with pytest.raises(
        FundamentalGenerationError,
        match="authoritative promotion policy",
    ):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256=before,
        )
    assert pointer_sha256(canonical) == before


def test_promotion_rejects_checkpoint_exact_byte_hash_drift(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)
    before = pointer_sha256(canonical)
    staged = load_fundamental_pointer(staging)
    assert staged is not None
    provider = staged["manifest"]["metadata"]["provider_manifest"]
    checkpoint_root = Path(provider["checkpoint"]["root"])
    checkpoint_pointer = json.loads(
        (checkpoint_root / "latest.json").read_text(encoding="utf-8")
    )
    checkpoint_manifest_path = checkpoint_root / checkpoint_pointer["manifest_path"]
    checkpoint_manifest = json.loads(
        checkpoint_manifest_path.read_text(encoding="utf-8")
    )
    raw_table = (
        checkpoint_manifest_path.parent
        / checkpoint_manifest["table_files"]["daily_basic"]["path"]
    )
    with raw_table.open("ab") as handle:
        handle.write(b"checkpoint-tamper")

    with pytest.raises(FundamentalGenerationError, match="checkpoint table SHA"):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256=before,
        )
    assert pointer_sha256(canonical) == before


def test_promotion_rejects_provider_outcomes_or_raw_fingerprint_drift(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    _publish_offline(canonical)
    before = pointer_sha256(canonical)

    outcomes_stage = tmp_path / "outcomes-stage"
    _publish_verified_primary(outcomes_stage)

    def forge_outcome(provider: dict[str, Any]) -> None:
        outcomes = [dict(value) for value in provider["symbol_table_outcomes"]]
        outcomes[0]["error"] = "forged-provider-only"
        provider["symbol_table_outcomes"] = outcomes
        provider["request_outcome_accounting_sha256"] = canonical_json_sha256(
            outcomes
        )

    _mutate_provider_and_resign(outcomes_stage, forge_outcome)
    with pytest.raises(
        FundamentalGenerationError,
        match="outcome.*mismatch",
    ):
        promote_staged_fundamental_generation(
            staging_root=outcomes_stage,
            canonical_root=canonical,
            expected_pointer_sha256=before,
        )

    fingerprint_stage = tmp_path / "fingerprint-stage"
    _publish_verified_primary(fingerprint_stage, run_id="fingerprint-stage")

    def forge_fingerprint(provider: dict[str, Any]) -> None:
        fingerprints = dict(provider["raw_table_fingerprints"])
        fingerprints["daily_basic"] = "0" * 64
        provider["raw_table_fingerprints"] = fingerprints
        derivation = dict(provider["derivation"])
        derivation["raw_table_fingerprints"] = fingerprints
        provider["derivation"] = derivation
        provider["raw_to_derived_binding_sha256"] = canonical_json_sha256(
            derivation
        )

    _mutate_provider_and_resign(fingerprint_stage, forge_fingerprint)
    with pytest.raises(FundamentalGenerationError, match="raw fingerprint mismatch"):
        promote_staged_fundamental_generation(
            staging_root=fingerprint_stage,
            canonical_root=canonical,
            expected_pointer_sha256=before,
        )
    assert pointer_sha256(canonical) == before


@pytest.mark.parametrize(
    ("table_name", "mutate"),
    [
        (
            "fundamental_period",
            lambda frame: frame.assign(fin_roe=frame["fin_roe"] + 0.01),
        ),
        (
            "fundamental_daily",
            lambda frame: frame.assign(sector="forged-sector"),
        ),
        (
            "fundamental_daily",
            lambda frame: frame.assign(forecast_summary="forged-forecast"),
        ),
        (
            "fundamental_quarantine",
            lambda _frame: pd.DataFrame(
                [
                    {
                        "ts_code": "000002.SZ",
                        "quarantine_reason": "forged-quarantine",
                    }
                ]
            ),
        ),
    ],
)
def test_promotion_rejects_resigned_derived_projection_tamper(
    tmp_path: Path,
    table_name: str,
    mutate: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)
    before = pointer_sha256(canonical)
    _mutate_derived_table_and_resign(staging, table_name, mutate)

    with pytest.raises(FundamentalGenerationError, match="raw-to-derived replay mismatch"):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256=before,
        )
    assert pointer_sha256(canonical) == before


def test_promotion_allows_only_synthetic_fetch_timestamp_projection_drift(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)

    def change_synthetic_time(frame: pd.DataFrame) -> pd.DataFrame:
        frame["fetched_at"] = "2099-01-01T00:00:00Z"
        return frame

    _mutate_derived_table_and_resign(
        staging,
        "fundamental_period",
        change_synthetic_time,
    )
    result = promote_staged_fundamental_generation(
        staging_root=staging,
        canonical_root=canonical,
        expected_pointer_sha256=pointer_sha256(canonical),
    )
    assert result["promoted"] is True


def test_promotion_rejects_derivation_or_membership_drift(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    _publish_offline(canonical)
    before = pointer_sha256(canonical)

    derivation_stage = tmp_path / "derivation-stage"
    _publish_verified_primary(derivation_stage)

    def forge_derivation(provider: dict[str, Any]) -> None:
        derivation = dict(provider["derivation"])
        derivation["selection_rule"] = "forged-selection-rule"
        provider["derivation"] = derivation
        provider["raw_to_derived_binding_sha256"] = canonical_json_sha256(
            derivation
        )

    _mutate_provider_and_resign(derivation_stage, forge_derivation)
    with pytest.raises(FundamentalGenerationError, match="derivation contract"):
        promote_staged_fundamental_generation(
            staging_root=derivation_stage,
            canonical_root=canonical,
            expected_pointer_sha256=before,
        )

    membership_stage = tmp_path / "membership-stage"
    _publish_verified_primary(membership_stage, run_id="membership-stage")
    staged = load_fundamental_pointer(membership_stage)
    assert staged is not None
    membership_path = Path(
        staged["manifest"]["metadata"]["provider_manifest"]
        ["canonical_scope_evidence"]["canonical_membership_path"]
    )
    with membership_path.open("ab") as handle:
        handle.write(b"membership-tamper")
    with pytest.raises(FundamentalGenerationError, match="scope evidence drifted"):
        promote_staged_fundamental_generation(
            staging_root=membership_stage,
            canonical_root=canonical,
            expected_pointer_sha256=before,
        )
    assert pointer_sha256(canonical) == before


def test_promotion_recomputes_financial_coverage_and_rejects_forged_audit(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)
    before = pointer_sha256(canonical)
    staged = load_fundamental_pointer(staging)
    assert staged is not None
    provider = staged["manifest"]["metadata"]["provider_manifest"]
    old_outcomes = {
        (value["symbol"], value["table"]): dict(value)
        for value in provider["symbol_table_outcomes"]
    }
    sparse_raw = _raw_tables(["000002.SZ"])
    for table in ("fina_indicator", "income", "balancesheet", "cashflow"):
        sparse_raw[table] = sparse_raw[table][
            sparse_raw[table]["end_date"].eq("20221231")
        ].reset_index(drop=True)
    forged_outcomes = _clean_outcomes(sparse_raw, ["000002.SZ"])
    for outcome in forged_outcomes:
        if outcome["table"] in {
            "fina_indicator",
            "income",
            "balancesheet",
            "cashflow",
        }:
            declared = old_outcomes[(outcome["symbol"], outcome["table"])]
            outcome["financial_coverage"] = declared["financial_coverage"]
            outcome["financial_coverage_passed"] = declared[
                "financial_coverage_passed"
            ]
    checkpoint_root = Path(provider["checkpoint"]["root"])
    pointer = json.loads(
        (checkpoint_root / "latest.json").read_text(encoding="utf-8")
    )
    checkpoint_manifest = json.loads(
        (checkpoint_root / pointer["manifest_path"]).read_text(encoding="utf-8")
    )
    forged_checkpoint = fundamental_mart._write_fetch_checkpoint(
        checkpoint_root,
        binding=checkpoint_manifest["binding"],
        tables=sparse_raw,
        outcomes=forged_outcomes,
        expected_pointer_sha256=provider["checkpoint"]["pointer_sha256"],
        expected_revision=provider["checkpoint"]["revision"],
    )

    def bind_forged_checkpoint(provider_value: dict[str, Any]) -> None:
        provider_value["symbol_table_outcomes"] = forged_checkpoint.outcomes
        provider_value["request_outcome_accounting_sha256"] = (
            forged_checkpoint.outcome_accounting_sha256
        )
        provider_value["raw_row_counts"] = {
            table: len(forged_checkpoint.tables[table])
            for table in generation.FUNDAMENTAL_RAW_TABLES
        }
        fingerprints = {
            table: frame_fingerprint(forged_checkpoint.tables[table])
            for table in generation.FUNDAMENTAL_RAW_TABLES
        }
        provider_value["raw_table_fingerprints"] = fingerprints
        provider_value["checkpoint"] = {
            **dict(provider_value["checkpoint"]),
            "generation_id": forged_checkpoint.generation_id,
            "revision": forged_checkpoint.revision,
            "pointer_sha256": forged_checkpoint.pointer_sha256,
            "manifest_sha256": forged_checkpoint.manifest_sha256,
            "binding_sha256": forged_checkpoint.binding_sha256,
            "outcome_accounting_sha256": (
                forged_checkpoint.outcome_accounting_sha256
            ),
            "table_evidence_sha256": forged_checkpoint.table_evidence_sha256,
        }
        derivation = dict(provider_value["derivation"])
        derivation["raw_table_fingerprints"] = fingerprints
        provider_value["derivation"] = derivation
        provider_value["raw_to_derived_binding_sha256"] = canonical_json_sha256(
            derivation
        )

    _mutate_provider_and_resign(staging, bind_forged_checkpoint)
    with pytest.raises(FundamentalGenerationError, match="financial coverage"):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256=before,
        )
    assert pointer_sha256(canonical) == before


def test_primary_manifest_binds_exact_readback_after_object_string_roundtrip(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    tables = _tables("000002.SZ")
    tables["fundamental_daily"].insert(
        len(tables["fundamental_daily"].columns),
        "sector",
        pd.Series(
            ["bank"] * len(tables["fundamental_daily"]),
            dtype=object,
        ),
    )
    assert str(tables["fundamental_daily"]["sector"].dtype) == "object"
    prewrite_fingerprint = generation._frame_fingerprint(
        tables["fundamental_daily"]
    )

    _publish_verified_primary(staging, tables_override=tables)

    pointer = load_fundamental_pointer(staging)
    assert pointer is not None
    assert pointer["primary_provenance"]["schema_version"] == (
        "cn-fundamental-primary-provenance.v2"
    )
    manifest = pointer["manifest"]
    table_manifest = manifest["tables"]["fundamental_daily"]
    relative_table_path = Path(pointer["tables"]["fundamental_daily"])
    assert not relative_table_path.is_absolute()
    assert ".." not in relative_table_path.parts
    table_path = staging / relative_table_path
    payload = table_path.read_bytes()
    readback = pd.read_parquet(table_path)
    assert hashlib.sha256(payload).hexdigest() == table_manifest["sha256"]
    assert len(readback) == table_manifest["rows"]
    assert list(readback.columns) == table_manifest["columns"]
    assert generation.frame_fingerprint(readback) == table_manifest[
        "frame_fingerprint"
    ]
    assert generation.frame_logical_schema(readback) == table_manifest[
        "logical_schema"
    ]
    assert prewrite_fingerprint == table_manifest["frame_fingerprint"]
    assert pointer["primary_provenance"]["output_frame_fingerprints"][
        "fundamental_daily"
    ] == table_manifest["frame_fingerprint"]


def test_primary_runtime_pointer_uses_sha_and_metadata_without_semantic_rescan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "primary-runtime"
    _publish_verified_primary(root)
    generation._validate_fundamental_pointer_cached.cache_clear()

    def reject_semantic_rescan(*_args: Any, **_kwargs: Any):
        raise AssertionError("primary runtime must not rescan every scalar")

    monkeypatch.setattr(
        generation,
        "_readback_table_contract",
        reject_semantic_rescan,
    )

    pointer = load_fundamental_pointer(root)

    assert pointer is not None
    assert pointer["primary_provenance_verified"] is True


def test_primary_runtime_pointer_rejects_hardlinked_table(tmp_path: Path) -> None:
    root = tmp_path / "primary-hardlink"
    _publish_verified_primary(root)
    pointer = json.loads(
        (root / generation.FUNDAMENTAL_POINTER_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    table = root / pointer["tables"]["fundamental_daily"]
    os.link(table, root / "fundamental_daily.alias.parquet")
    generation._validate_fundamental_pointer_cached.cache_clear()

    with pytest.raises(FundamentalGenerationError, match="hard-linked"):
        load_fundamental_pointer(root)


def test_primary_runtime_pointer_detects_manifest_replacement_during_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "primary-manifest-race"
    _publish_verified_primary(root)
    pointer = json.loads(
        (root / generation.FUNDAMENTAL_POINTER_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    manifest_path = root / pointer["manifest_path"]
    original = generation._readback_primary_table_identity
    replaced = False

    def replace_manifest_after_table(*args: Any, **kwargs: Any):
        nonlocal replaced
        result = original(*args, **kwargs)
        if not replaced:
            replaced = True
            replacement = manifest_path.with_name("manifest.replacement.json")
            replacement.write_bytes(manifest_path.read_bytes())
            os.replace(replacement, manifest_path)
        return result

    generation._validate_fundamental_pointer_cached.cache_clear()
    monkeypatch.setattr(
        generation,
        "_readback_primary_table_identity",
        replace_manifest_after_table,
    )

    with pytest.raises(
        FundamentalGenerationError,
        match="manifest changed during validation",
    ):
        load_fundamental_pointer(root)
    assert replaced is True


def test_streaming_parquet_readback_preserves_exact_frame_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "streaming"
    tables = _tables("000002.SZ")
    tables["fundamental_daily"] = pd.DataFrame(
        {
            "ts_code": ["000002.SZ", "000002.SZ", "000002.SZ"],
            "trade_date": ["20240508", "20240509", "20240510"],
            "end_date": ["20221231", "20221231", "20221231"],
            "availability_date": ["20230501", "20230501", "20230501"],
            "sector": [None, "银行", "bank"],
            "fin_roe": [float("nan"), float("inf"), 0.3],
            "nullable_bool": pd.Series([pd.NA, True, False], dtype="boolean"),
            "nullable_int": pd.Series([pd.NA, 1, 2], dtype="Int64"),
            "bytes_value": [None, b"a", b"\x00"],
            "date_value": [None, date(2024, 5, 9), date(2024, 5, 10)],
            "timestamp_value": [pd.NaT, pd.Timestamp("2024-05-09"), pd.Timestamp("2024-05-10")],
            "timedelta_value": [pd.NaT, pd.Timedelta(days=1), pd.Timedelta(days=2)],
            "decimal_value": [None, Decimal("1.25"), Decimal("2.50")],
        }
    )
    expected = frame_fingerprint(tables["fundamental_daily"])
    monkeypatch.setattr(
        generation,
        "FUNDAMENTAL_STREAMING_READBACK_MIN_ROWS",
        1,
    )
    monkeypatch.setattr(
        generation,
        "FUNDAMENTAL_STREAMING_ROW_GROUP_SIZE",
        1,
    )
    original_readback = generation._readback_table_contract

    def reject_full_daily_readback(*args: Any, **kwargs: Any):
        if kwargs.get("table_name") == "fundamental_daily":
            raise AssertionError("streaming daily must not use full-byte readback")
        return original_readback(*args, **kwargs)

    monkeypatch.setattr(
        generation,
        "_readback_table_contract",
        reject_full_daily_readback,
    )

    publish_fundamental_generation(
        root=root,
        run_id="streaming-identity",
        tables=tables,
        metadata={
            "run_id": "streaming-identity",
            "source_priority": "manual_offline_snapshot",
            "gate2_passed": True,
        },
    )

    pointer = json.loads(
        (root / generation.FUNDAMENTAL_POINTER_FILENAME).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (root / pointer["manifest_path"]).read_text(encoding="utf-8")
    )
    table_manifest = manifest["tables"]["fundamental_daily"]
    assert table_manifest["frame_fingerprint"] == expected
    assert table_manifest["rows"] == 3
    assert table_manifest["columns"] == list(tables["fundamental_daily"].columns)
    table_path = root / pointer["tables"]["fundamental_daily"]
    parquet = generation.pq.ParquetFile(table_path)
    assert parquet.num_row_groups == 3
    assert max(
        parquet.metadata.row_group(index).num_rows
        for index in range(parquet.num_row_groups)
    ) <= 1
    full_readback = pd.read_parquet(table_path)
    assert frame_fingerprint(full_readback) == expected
    assert frame_logical_schema(full_readback) == table_manifest["logical_schema"]


def test_streaming_parquet_readback_rejects_inode_swap_before_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "streaming-race"
    tables = _tables("000002.SZ")
    monkeypatch.setattr(
        generation,
        "FUNDAMENTAL_STREAMING_READBACK_MIN_ROWS",
        1,
    )
    original_streaming = generation._streaming_parquet_table_evidence
    swapped = False

    def swap_then_read(path: Path, **kwargs: Any):
        nonlocal swapped
        if kwargs.get("table_name") == "fundamental_daily" and not swapped:
            swapped = True
            replacement = path.with_name("replacement.parquet")
            replacement.write_bytes(path.read_bytes())
            os.replace(replacement, path)
        return original_streaming(path, **kwargs)

    monkeypatch.setattr(
        generation,
        "_streaming_parquet_table_evidence",
        swap_then_read,
    )
    with pytest.raises(
        FundamentalGenerationError,
        match="changed before readback",
    ):
        publish_fundamental_generation(
            root=root,
            run_id="streaming-race",
            tables=tables,
            metadata={
                "run_id": "streaming-race",
                "source_priority": "manual_offline_snapshot",
                "gate2_passed": True,
            },
        )

    assert swapped is True
    assert not (root / generation.FUNDAMENTAL_POINTER_FILENAME).exists()


def test_streaming_parquet_readback_rejects_inode_swap_between_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "streaming.parquet"
    pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": ["20240509", "20240510"],
        }
    ).to_parquet(path, index=False, engine="pyarrow", row_group_size=1)
    file_sha256, signature = generation._stable_file_sha256(path)
    original_schema = generation.frame_logical_schema
    swapped = False

    def swap_after_first_schema(frame: pd.DataFrame):
        nonlocal swapped
        result = original_schema(frame)
        if not swapped:
            swapped = True
            replacement = tmp_path / "replacement.parquet"
            replacement.write_bytes(path.read_bytes())
            os.replace(replacement, path)
        return result

    monkeypatch.setattr(generation, "frame_logical_schema", swap_after_first_schema)
    with pytest.raises(
        FundamentalGenerationError,
        match="changed during readback",
    ):
        generation._streaming_parquet_table_evidence(
            path,
            table_name="fundamental_daily",
            file_sha256=file_sha256,
            expected_signature=signature,
        )
    assert swapped is True


def test_promotion_rejects_forged_readback_fingerprint_before_pointer_change(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)
    before = pointer_sha256(canonical)
    pointer = json.loads(
        (staging / generation.FUNDAMENTAL_POINTER_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    manifest_path = staging / pointer["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tables"]["fundamental_daily"]["frame_fingerprint"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        FundamentalGenerationError,
        match="frame fingerprint mismatch",
    ):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256=before,
        )

    assert pointer_sha256(canonical) == before


def test_sorted_key_primary_artifacts_load_and_promote(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)
    pointer_path = staging / generation.FUNDAMENTAL_POINTER_FILENAME
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    manifest_path = staging / pointer["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    pointer_path.write_text(
        json.dumps(
            pointer,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    staged = load_fundamental_pointer(staging)
    assert staged is not None
    assert staged["primary_provenance_verified"] is True
    promoted = promote_staged_fundamental_generation(
        staging_root=staging,
        canonical_root=canonical,
        expected_pointer_sha256=pointer_sha256(canonical),
    )

    assert promoted["promoted"] is True
    assert load_fundamental_pointer(canonical)[
        "primary_provenance_verified"
    ] is True


def test_authoritative_loader_and_promotion_reject_provenance_v1(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)
    before = pointer_sha256(canonical)
    pointer_path = staging / generation.FUNDAMENTAL_POINTER_FILENAME
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    manifest_path = staging / pointer["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    envelope = dict(manifest["primary_provenance"])
    envelope.pop("envelope_sha256")
    envelope["schema_version"] = "cn-fundamental-primary-provenance.v1"
    envelope["envelope_sha256"] = generation._metadata_sha256(envelope)
    pointer["primary_provenance"] = envelope
    manifest["primary_provenance"] = envelope
    pointer_path.write_text(
        json.dumps(pointer, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        FundamentalGenerationError,
        match="primary provenance contract mismatch",
    ):
        load_fundamental_pointer(staging)
    with pytest.raises(FundamentalGenerationError):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256=before,
        )

    assert pointer_sha256(canonical) == before


def test_non_primary_legacy_manifest_remains_readable(tmp_path: Path) -> None:
    root = tmp_path / "offline"
    _publish_offline(root)
    pointer = json.loads(
        (root / generation.FUNDAMENTAL_POINTER_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    manifest_path = root / pointer["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for table_manifest in manifest["tables"].values():
        table_manifest.pop("frame_fingerprint")
        table_manifest.pop("logical_schema")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    loaded = load_fundamental_pointer(root)

    assert loaded is not None
    assert loaded["generation_id"] == "canonical-old"
    assert loaded["primary_provenance_verified"] is False


def test_publish_semantic_readback_failure_keeps_predecessor_pointer_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical"
    _publish_offline(canonical)
    pointer_path = canonical / generation.FUNDAMENTAL_POINTER_FILENAME
    before_bytes = pointer_path.read_bytes()
    before = hashlib.sha256(before_bytes).hexdigest()

    def reject_roundtrip(*_args, **_kwargs):
        raise ValueError("simulated semantic mismatch")

    monkeypatch.setattr(
        generation,
        "assert_frame_semantics_equal",
        reject_roundtrip,
    )
    with pytest.raises(
        FundamentalGenerationError,
        match="semantic readback mismatch",
    ):
        _publish_verified_primary(
            canonical,
            run_id="rejected-primary",
            expected_pointer_sha256=before,
        )

    assert pointer_path.read_bytes() == before_bytes
    assert not (
        canonical
        / generation.FUNDAMENTAL_GENERATIONS_DIRNAME
        / "rejected-primary"
    ).exists()


def test_promotion_repairs_hash_valid_primary_predecessor_missing_envelope(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_verified_primary(canonical, run_id="legacy-primary")
    pointer_path = canonical / generation.FUNDAMENTAL_POINTER_FILENAME
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    manifest_path = canonical / pointer["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pointer.pop("primary_provenance")
    manifest.pop("primary_provenance")
    pointer_path.write_text(json.dumps(pointer, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _publish_verified_primary(staging, run_id="replacement-primary")
    before = pointer_sha256(canonical)

    result = promote_staged_fundamental_generation(
        staging_root=staging,
        canonical_root=canonical,
        expected_pointer_sha256=before,
    )

    assert result["generation_id"] == "replacement-primary"
    assert load_fundamental_pointer(canonical)["primary_provenance_verified"] is True


def test_promotion_rejects_unverified_staging(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_offline(staging, run_id="offline-stage")
    before = pointer_sha256(canonical)

    with pytest.raises(
        FundamentalGenerationError,
        match="not verified primary gate2",
    ):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256=before,
        )

    assert pointer_sha256(canonical) == before
    assert not (canonical / generation.FUNDAMENTAL_GENERATIONS_DIRNAME / "offline-stage").exists()


def test_promotion_rejects_pointer_cas_drift(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)

    with pytest.raises(FundamentalGenerationError, match="CAS mismatch"):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256="0" * 64,
        )

    assert load_fundamental_pointer(canonical)["generation_id"] == "canonical-old"


def test_normal_publisher_and_promotion_share_pointer_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)
    expected = pointer_sha256(canonical)
    pointer_write_reached = threading.Event()
    release_pointer_write = threading.Event()
    original_atomic = generation._atomic_write_json

    def hold_concurrent_pointer(path, payload):
        if (
            path.name == generation.FUNDAMENTAL_POINTER_FILENAME
            and payload.get("generation_id") == "concurrent-writer"
        ):
            pointer_write_reached.set()
            assert release_pointer_write.wait(timeout=5)
        return original_atomic(path, payload)

    monkeypatch.setattr(generation, "_atomic_write_json", hold_concurrent_pointer)
    publisher_error: list[BaseException] = []
    promotion_error: list[BaseException] = []

    def publish_concurrently():
        try:
            _publish_offline(
                canonical,
                run_id="concurrent-writer",
                expected_pointer_sha256=expected,
            )
        except BaseException as exc:  # pragma: no cover - assertion aid
            publisher_error.append(exc)

    def promote_concurrently():
        try:
            promote_staged_fundamental_generation(
                staging_root=staging,
                canonical_root=canonical,
                expected_pointer_sha256=expected,
            )
        except BaseException as exc:
            promotion_error.append(exc)

    publisher = threading.Thread(target=publish_concurrently)
    publisher.start()
    assert pointer_write_reached.wait(timeout=5)
    promotion = threading.Thread(target=promote_concurrently)
    promotion.start()
    time.sleep(0.05)
    assert promotion.is_alive()
    release_pointer_write.set()
    publisher.join(timeout=5)
    promotion.join(timeout=5)

    assert publisher_error == []
    assert len(promotion_error) == 1
    assert "CAS mismatch" in str(promotion_error[0])
    assert load_fundamental_pointer(canonical)["generation_id"] == "concurrent-writer"


def test_stale_normal_publisher_cannot_overwrite_promoted_generation(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)
    stale_predecessor = pointer_sha256(canonical)
    promote_staged_fundamental_generation(
        staging_root=staging,
        canonical_root=canonical,
        expected_pointer_sha256=stale_predecessor,
    )

    with pytest.raises(
        FundamentalGenerationError,
        match="predecessor pointer CAS mismatch",
    ):
        _publish_offline(
            canonical,
            run_id="stale-writer",
            expected_pointer_sha256=stale_predecessor,
        )

    assert load_fundamental_pointer(canonical)["generation_id"] == "verified-stage"


def test_promotion_rejects_symlink_root_and_lock(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)
    expected = pointer_sha256(canonical)
    alias = tmp_path / "canonical-alias"
    alias.symlink_to(canonical, target_is_directory=True)

    with pytest.raises(FundamentalGenerationError, match="symlink rejected"):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=alias,
            expected_pointer_sha256=expected,
        )

    lock = canonical / generation.FUNDAMENTAL_PROMOTION_LOCK_FILENAME
    lock.unlink()
    lock.symlink_to(tmp_path / "outside-lock")
    with pytest.raises(FundamentalGenerationError, match="lock is unsafe"):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256=expected,
        )


def test_promotion_rejects_staging_table_hash_tamper(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)
    pointer = load_fundamental_pointer(staging)
    assert pointer is not None
    table = staging / pointer["tables"]["fundamental_daily"]
    with table.open("ab") as handle:
        handle.write(b"tamper")
    before = pointer_sha256(canonical)

    with pytest.raises(FundamentalGenerationError, match="table hash mismatch"):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256=before,
        )

    assert pointer_sha256(canonical) == before


def test_promotion_rejects_period_end_after_availability(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    tables = _tables("000002.SZ")
    tables["fundamental_period"]["end_date"] = "20251231"
    _publish_verified_primary(staging, tables_override=tables)

    with pytest.raises(
        FundamentalGenerationError,
        match="period end is after availability",
    ):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256=pointer_sha256(canonical),
        )


def test_promotion_rejects_daily_end_after_availability(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    tables = _tables("000002.SZ")
    tables["fundamental_daily"]["end_date"] = "20250501"
    _publish_verified_primary(staging, tables_override=tables)

    with pytest.raises(
        FundamentalGenerationError,
        match="daily end is after availability",
    ):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256=pointer_sha256(canonical),
        )


def test_promotion_rejects_market_pointer_missing_membership_binding(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(
        staging,
        market_pointer_override={
            "snapshot_id": "fixture-market-snapshot",
            "latest_complete_trade_date": "20240510",
        },
    )

    with pytest.raises(
        FundamentalGenerationError,
        match="canonical market",
    ):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256=pointer_sha256(canonical),
        )


def test_promotion_rejects_scope_exception_not_bound_by_market_pointer(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(
        staging,
        scope_exception_symbols=["000002.SZ"],
        pointer_exception_symbols=[],
    )

    with pytest.raises(
        FundamentalGenerationError,
        match="PIT membership eligibility binding is invalid",
    ):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256=pointer_sha256(canonical),
        )


def test_promotion_rejects_out_of_scope_period_symbol(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    tables = _tables("000002.SZ")
    extra = tables["fundamental_period"].copy()
    extra["ts_code"] = "999999.SZ"
    tables["fundamental_period"] = pd.concat(
        [tables["fundamental_period"], extra],
        ignore_index=True,
    )
    _publish_verified_primary(staging, tables_override=tables)

    with pytest.raises(
        FundamentalGenerationError,
        match="out-of-scope symbols: fundamental_period",
    ):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256=pointer_sha256(canonical),
        )


def test_promotion_rejects_null_effective_to_as_open_interval(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(
        staging,
        membership_override=pd.DataFrame(
            [
                {
                    "symbol": "000002.SZ",
                    "list_date": "20230510",
                    "effective_from": "20230510",
                    "effective_to": pd.NA,
                }
            ]
        ),
    )

    with pytest.raises(
        FundamentalGenerationError,
        match="required date is null",
    ):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256=pointer_sha256(canonical),
        )


def test_promotion_rejects_suffixed_membership_effective_from(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(
        staging,
        membership_override=pd.DataFrame(
            [
                {
                    "symbol": "000002.SZ",
                    "list_date": "20230510",
                    "effective_from": "20230510junk",
                    "effective_to": "",
                }
            ]
        ),
    )

    with pytest.raises(
        FundamentalGenerationError,
        match="effective_from is invalid",
    ):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256=pointer_sha256(canonical),
        )


def test_promotion_rejects_majority_single_day_symbol_histories(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    full = _tables("000002.SZ")
    short_a = _tables("000003.SZ")
    short_b = _tables("000004.SZ")
    tables = {
        "fundamental_period": pd.concat(
            [
                full["fundamental_period"],
                short_a["fundamental_period"],
                short_b["fundamental_period"],
            ],
            ignore_index=True,
        ),
        "fundamental_daily": pd.concat(
            [
                full["fundamental_daily"],
                short_a["fundamental_daily"].tail(1),
                short_b["fundamental_daily"].tail(1),
            ],
            ignore_index=True,
        ),
        "fundamental_quarantine": full["fundamental_quarantine"],
    }
    _publish_verified_primary(staging, tables_override=tables)

    with pytest.raises(
        FundamentalGenerationError,
        match="per-symbol daily history incomplete",
    ):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256=pointer_sha256(canonical),
        )


def test_promotion_rejects_history_clustered_away_from_middle_months(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    tables = _tables("000002.SZ")
    clustered = pd.bdate_range("2023-05-10", "2023-10-10").append(
        pd.DatetimeIndex([pd.Timestamp("2024-05-10")])
    )
    tables["fundamental_daily"] = pd.DataFrame(
        {
            "ts_code": "000002.SZ",
            "trade_date": clustered.strftime("%Y%m%d"),
            "end_date": "20221231",
            "availability_date": "20230501",
        }
    )
    _publish_verified_primary(staging, tables_override=tables)

    with pytest.raises(
        FundamentalGenerationError,
        match="per-symbol daily history incomplete",
    ):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256=pointer_sha256(canonical),
        )


def test_post_switch_failure_rolls_pointer_back_by_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)
    before = pointer_sha256(canonical)
    original_load = generation.load_fundamental_pointer
    failed = False

    def fail_once(root: str | Path):
        nonlocal failed
        pointer = original_load(root)
        if (
            not failed
            and Path(root).resolve() == canonical.resolve()
            and pointer is not None
            and pointer.get("generation_id") == "verified-stage"
        ):
            failed = True
            raise FundamentalGenerationError("simulated post-switch failure")
        return pointer

    monkeypatch.setattr(generation, "load_fundamental_pointer", fail_once)
    with pytest.raises(FundamentalGenerationError, match="pointer rolled back"):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256=before,
        )

    assert pointer_sha256(canonical) == before
    assert original_load(canonical)["generation_id"] == "canonical-old"
    assert (canonical / generation.FUNDAMENTAL_GENERATIONS_DIRNAME / "verified-stage").is_dir()


def test_post_switch_canonical_scope_drift_rolls_pointer_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)
    before = pointer_sha256(canonical)
    staged = load_fundamental_pointer(staging)
    assert staged is not None
    scope = staged["manifest"]["metadata"]["provider_manifest"][
        "canonical_scope_evidence"
    ]
    bars_path = Path(scope["canonical_bar_table_root"]) / "part.parquet"
    original_revalidate = generation._revalidate_captured_primary_scope
    calls = 0

    def drift_on_post_switch(captured: Any) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            bars = pd.read_parquet(bars_path)
            bars.loc[0, "trade_date"] = "20230511"
            bars.to_parquet(bars_path, index=False)
        return original_revalidate(captured)

    monkeypatch.setattr(
        generation,
        "_revalidate_captured_primary_scope",
        drift_on_post_switch,
    )
    with pytest.raises(FundamentalGenerationError, match="pointer rolled back"):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256=before,
        )

    assert calls == 2
    assert pointer_sha256(canonical) == before
    assert load_fundamental_pointer(canonical)["generation_id"] == "canonical-old"


def test_post_switch_installed_table_drift_rolls_pointer_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)
    before = pointer_sha256(canonical)
    original_validate = generation._validate_installed_promotion_identity

    def tamper_then_validate(**kwargs: Any):
        table_path = kwargs["final_root"] / "fundamental_daily.parquet"
        with table_path.open("ab") as handle:
            handle.write(b"post-switch-drift")
        return original_validate(**kwargs)

    monkeypatch.setattr(
        generation,
        "_validate_installed_promotion_identity",
        tamper_then_validate,
    )
    with pytest.raises(FundamentalGenerationError, match="pointer rolled back"):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256=before,
        )

    assert pointer_sha256(canonical) == before
    assert load_fundamental_pointer(canonical)["generation_id"] == "canonical-old"


def test_scope_drift_during_installed_validation_rolls_pointer_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)
    before = pointer_sha256(canonical)
    staged = load_fundamental_pointer(staging)
    assert staged is not None
    scope = staged["manifest"]["metadata"]["provider_manifest"][
        "canonical_scope_evidence"
    ]
    bars_path = Path(scope["canonical_bar_table_root"]) / "part.parquet"
    original_validate = generation._validate_installed_promotion_identity

    def validate_then_drift(**kwargs: Any):
        installed = original_validate(**kwargs)
        bars = pd.read_parquet(bars_path)
        bars.loc[0, "trade_date"] = "20230511"
        bars.to_parquet(bars_path, index=False)
        return installed

    monkeypatch.setattr(
        generation,
        "_validate_installed_promotion_identity",
        validate_then_drift,
    )
    with pytest.raises(FundamentalGenerationError, match="pointer rolled back"):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256=before,
        )

    assert pointer_sha256(canonical) == before
    assert load_fundamental_pointer(canonical)["generation_id"] == "canonical-old"


def test_post_switch_pointer_race_is_not_rolled_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _publish_offline(canonical)
    _publish_verified_primary(staging)
    before = pointer_sha256(canonical)
    pointer_path = canonical / generation.FUNDAMENTAL_POINTER_FILENAME
    old_pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    race_pointer = {**old_pointer, "race_marker": True}
    race_bytes = (
        json.dumps(race_pointer, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()
    race_hash = hashlib.sha256(race_bytes).hexdigest()
    original_load = generation.load_fundamental_pointer
    raced = False

    def race_after_switch(root: str | Path):
        nonlocal raced
        pointer = original_load(root)
        if (
            not raced
            and Path(root).resolve() == canonical.resolve()
            and pointer is not None
            and pointer.get("generation_id") == "verified-stage"
        ):
            raced = True
            pointer_path.write_bytes(race_bytes)
            raise FundamentalGenerationError("simulated pointer race")
        return pointer

    monkeypatch.setattr(
        generation,
        "load_fundamental_pointer",
        race_after_switch,
    )
    with pytest.raises(
        FundamentalGenerationError,
        match="pointer drift; rollback not attempted",
    ):
        promote_staged_fundamental_generation(
            staging_root=staging,
            canonical_root=canonical,
            expected_pointer_sha256=before,
        )

    assert pointer_sha256(canonical) == race_hash
    assert json.loads(pointer_path.read_text(encoding="utf-8"))["race_marker"] is True
