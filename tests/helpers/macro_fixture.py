from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from quant_investor.market import macro_mart


_POINTER_SHA = "1" * 64
_MARKET_INPUT_FILES = [
    {
        "path": "year=2024/month=05/part.parquet",
        "size_bytes": 1,
        "sha256": "2" * 64,
    }
]
_MARKET_FILES_SHA = macro_mart._canonical_json_sha256(
    {"files": _MARKET_INPUT_FILES}
)


def _formula_universe(*, trade_date: str) -> dict[str, Any]:
    symbols = [f"{index:06d}.SZ" for index in range(100)]
    empty_sha = macro_mart._formula_symbol_set_sha256([])
    symbol_sha = macro_mart._formula_symbol_set_sha256(symbols)
    return {
        "schema_version": macro_mart.MARKET_FORMULA_UNIVERSE_SCHEMA,
        "selection_rule": macro_mart.MARKET_FORMULA_SELECTION_RULE,
        "target_trade_date": str(trade_date).replace("-", ""),
        "input_symbol_count": 100,
        "target_terminal_symbol_count": 100,
        "stale_symbol_count": 0,
        "scored_symbol_count": 100,
        "input_row_count": 100,
        "target_terminal_row_count": 100,
        "stale_row_count": 0,
        "input_symbol_set_sha256": symbol_sha,
        "target_terminal_symbol_set_sha256": symbol_sha,
        "stale_symbol_set_sha256": empty_sha,
        "scored_symbol_set_sha256": symbol_sha,
    }


def _bundle(*, trade_date: str) -> dict[str, Any]:
    fetched_at = "2024-05-10T08:00:00+00:00"
    endpoints: dict[str, Any] = {}
    selected: dict[str, Any] = {}
    for endpoint, spec in sorted(macro_mart._ENDPOINT_SPECS.items()):
        values = {
            field: 8.6 if field == "m2_yoy" else 1.0
            for field in spec["value_fields"]
        }
        record = {"month": "202404", **values}
        records = [record]
        endpoints[endpoint] = {
            "endpoint": endpoint,
            "query": {"start_m": "202401", "end_m": "202405"},
            "columns": sorted(record),
            "row_count": 1,
            "records": records,
            "records_sha256": hashlib.sha256(
                macro_mart._canonical_json_bytes({"records": records})
            ).hexdigest(),
        }
        selected[endpoint] = {
            "month": "202404",
            "values": values,
            "observed_available_at": fetched_at,
            "official_release_timestamp_known": False,
            "max_release_lag_days": int(spec["max_release_lag_days"]),
            "conservative_available_by": "2024-06-14",
            "transform_role": (
                "policy_signal" if endpoint == "cn_m" else "context_only"
            ),
        }
    return {
        "schema_version": macro_mart.PROVIDER_BUNDLE_SCHEMA,
        "provider_id": "tushare_pro",
        "source": "tushare_primary",
        "source_priority": "tushare_primary",
        "trade_date": trade_date,
        "fetched_at": fetched_at,
        "decision_cutoff_at": fetched_at,
        "live_requested": True,
        "historical_replay_eligible": False,
        "official_release_timestamps_claimed": False,
        "query_window": {"start_month": "202401", "end_month": "202405"},
        "endpoints": endpoints,
        "selected_inputs": selected,
    }


def bind_macro_generation(
    root: Path,
    *,
    generation_id: str,
    row: Mapping[str, Any],
) -> tuple[Path, Path, Path, Path]:
    """Write a hash-consistent canonical fixture without invoking live code."""

    generation = root / "_generations" / generation_id
    generation.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([dict(row)])
    table = generation / "part.parquet"
    frame.to_parquet(table, index=False)
    table_sha = hashlib.sha256(table.read_bytes()).hexdigest()

    provider_bundle = _bundle(trade_date=str(row["trade_date"]))
    provider_path = generation / "provider_bundle.json"
    provider_path.write_bytes(
        macro_mart._canonical_json_bytes(provider_bundle) + b"\n"
    )
    provider_sha = hashlib.sha256(provider_path.read_bytes()).hexdigest()
    output_frame_sha = macro_mart._frame_sha256(frame)
    formula_universe = _formula_universe(
        trade_date=str(row["trade_date"])
    )
    formula_universe_sha = macro_mart._canonical_json_sha256(
        formula_universe
    )
    provenance: dict[str, Any] = {
        "schema_version": macro_mart.PRIMARY_PROVENANCE_SCHEMA,
        "status": "verified_live_tushare",
        "source": "tushare_primary",
        "source_priority": "tushare_primary",
        "trade_date": str(row["trade_date"]),
        "fetched_at": "2024-05-10T08:00:00+00:00",
        "provider_bundle_sha256": provider_sha,
        "canonical_market_pointer_sha256": _POINTER_SHA,
        "market_input_files_sha256": _MARKET_FILES_SHA,
        "market_formula_universe_sha256": formula_universe_sha,
        "output_frame_sha256": output_frame_sha,
        "output_parquet_sha256": table_sha,
        "transform_version": macro_mart.TRANSFORM_VERSION,
        "historical_replay_eligible": False,
    }
    provenance["envelope_sha256"] = macro_mart._canonical_json_sha256(
        provenance
    )
    manifest_payload = {
        "schema_version": macro_mart.CANONICAL_MANIFEST_SCHEMA,
        "generation_id": generation_id,
        "table": "macro_daily",
        "table_path": "part.parquet",
        "parquet_sha256": table_sha,
        "provider_bundle_path": "provider_bundle.json",
        "provider_bundle_sha256": provider_sha,
        "row_count": 1,
        "source": "tushare_primary",
        "source_priority": "tushare_primary",
        "provider_status": "verified_provider_snapshot",
        "pit_status": "market_point_in_time",
        "as_of": str(row["trade_date"]),
        "decision_cutoff_at": "2024-05-10T08:00:00+00:00",
        "historical_replay_eligible": False,
        "transform_version": macro_mart.TRANSFORM_VERSION,
        "market_input_files": _MARKET_INPUT_FILES,
        "market_input_files_sha256": _MARKET_FILES_SHA,
        "market_formula_universe": formula_universe,
        "market_formula_universe_sha256": formula_universe_sha,
        "primary_provenance": provenance,
        "production_eligible": True,
    }
    manifest = generation / "manifest.json"
    manifest.write_text(
        json.dumps(manifest_payload, sort_keys=True),
        encoding="utf-8",
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    catalog = root.parent / "_catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": macro_mart.STRICT_CATALOG_SCHEMA,
                "required_tables": ["macro_daily"],
                "tables": {
                    "macro_daily": {
                        "path": table.relative_to(root.parent).as_posix(),
                        "generation_manifest": manifest.relative_to(
                            root.parent
                        ).as_posix(),
                        "generation_id": generation_id,
                        "parquet_sha256": table_sha,
                        "generation_manifest_sha256": manifest_sha,
                        "provider_bundle_sha256": provider_sha,
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return catalog, table, manifest, provider_path
