from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from quant_investor.market import macro_mart
from quant_investor.macro.contracts import (
    MacroObservation,
    canonical_hash,
    parse_timestamp,
    published_cutoff,
)
from quant_investor.macro.registry import NATIONAL_INDICATORS
from quant_investor.macro.snapshot import build_macro_snapshot
from quant_investor.macro.store import pointer_sha256, publish_observations
from quant_investor.macro.v15_controls import (
    V15_MACRO_CONTROL_SCHEMA_VERSION,
    build_v15_macro_controls,
)


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


def make_v15_controls(
    *,
    macro_score: float = 0.2,
    liquidity_score: float = 0.4,
    volatility_percentile: float = 45.0,
    policy_signal: str = "neutral",
) -> dict[str, Any]:
    """Build a compact, hash-bound DAG fixture for the v15 control contract."""

    controls: dict[str, Any] = {
        "schema_version": V15_MACRO_CONTROL_SCHEMA_VERSION,
        "production_control_projection": True,
        "macro_score": float(macro_score),
        "liquidity_score": float(liquidity_score),
        "volatility_percentile": float(volatility_percentile),
        "policy_signal": str(policy_signal),
    }
    controls["semantic_sha256"] = canonical_hash(controls)
    return controls


def write_ready_macro_observations(
    root: Path,
    *,
    as_of: str,
    run_id: str = "macro-observations-ready",
    decision_cutoff_at: str | None = None,
) -> str:
    """Publish a production-like v2, 81.25%-coverage snapshot fixture."""

    target = pd.Timestamp(str(as_of))
    logical_as_of = target.date().isoformat()
    decision_cutoff = (
        published_cutoff(logical_as_of)
        if decision_cutoff_at is None
        else parse_timestamp(
            decision_cutoff_at,
            field_name="decision_cutoff_at",
        )
    )
    selected = [
        item
        for item in NATIONAL_INDICATORS
        if item.indicator_id
        not in {
            "cn.gdp_yoy",
            "market.breadth",
            "market.volatility_percentile",
        }
    ]
    observations: list[dict[str, Any]] = []
    for definition in selected:
        for offset in (3, 2, 1):
            period_end = target - pd.offsets.MonthEnd(offset)
            available = period_end + pd.Timedelta(days=1)
            timestamp = available.tz_localize("UTC").isoformat()
            observations.append(
                {
                    "indicator_id": definition.indicator_id,
                    "dimension_type": "national",
                    "industry_chain": "",
                    "period_end": period_end.date().isoformat(),
                    "release_at": timestamp,
                    "available_at": timestamp,
                    "vintage_id": "initial",
                    "value": 1.0,
                    "unit": definition.unit,
                    "frequency": definition.frequency,
                    "source_system": "nbs_official",
                    "source_record_id": (
                        f"fixture:{definition.indicator_id}:"
                        f"{period_end:%Y%m%d}"
                    ),
                    "source_url": "https://www.stats.gov.cn/fixture",
                    "fetched_at": timestamp,
                    "quality_status": "pass",
                }
            )
    normalized = [MacroObservation.from_mapping(item) for item in observations]
    snapshot = build_macro_snapshot(
        normalized,
        market="CN",
        as_of=logical_as_of,
        decision_cutoff_at=decision_cutoff,
    ).to_dict()
    content_hashes = sorted(item.content_hash for item in normalized)
    evidence_body = (
        json.dumps(
            {
                "schema_version": "macro-test-fixture-evidence.v1",
                "market": "CN",
                "as_of": logical_as_of,
                "decision_cutoff_at": decision_cutoff.isoformat(),
                "observation_content_hashes": content_hashes,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    evidence_sha256 = hashlib.sha256(evidence_body).hexdigest()
    result = publish_observations(
        normalized,
        root=root,
        run_id=run_id,
        metadata={
            "schema_version": "macro-production-observation-bundle.v1",
            "market": "CN",
            "as_of": logical_as_of,
            "decision_cutoff_at": decision_cutoff.isoformat(),
            "official_bundle_manifest_sha256": "1" * 64,
            "official_plan_sha256": "2" * 64,
            "local_bootstrap_plan_sha256": "3" * 64,
            "local_snapshot_manifest_sha256": "4" * 64,
            "local_coverage_contract_sha256": "5" * 64,
            "local_effective_available_at": max(
                item.available_at for item in normalized
            ),
            "validated_snapshot_hash": snapshot["snapshot_hash"],
            "atomic_combined_publication": True,
            "authority": "test_fixture",
        },
        evidence_bytes={evidence_sha256: evidence_body},
        evidence_metadata={
            evidence_sha256: {
                "extension": ".bin",
                "evidence_kind": "macro_test_fixture_source_bundle",
                "schema_version": "macro-test-fixture-evidence.v1",
                "market": "CN",
                "as_of": logical_as_of,
                "size_bytes": len(evidence_body),
            }
        },
        observation_evidence={
            content_hash: [evidence_sha256]
            for content_hash in content_hashes
        },
    )
    assert result["promoted"] is True
    return pointer_sha256(root)


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
        "schema_version": macro_mart.LEGACY_PROVIDER_BUNDLE_SCHEMA,
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
    snapshot_payload: dict[str, Any] = {
        "schema_version": "macro-snapshot.v2",
        "market": "CN",
        "as_of": str(row["trade_date"]),
        "readiness_status": "pass",
        "national_states": {
            "growth": 0.48,
            "credit_liquidity": 0.4,
            "inflation": 0.0,
            "policy_fiscal": 0.0,
            "property": 0.0,
            "external": 0.0,
            "market_confirmation": 0.0,
        },
        "coverage": {"national": 0.8125},
    }
    snapshot_payload["snapshot_hash"] = canonical_hash(snapshot_payload)
    observation_generation = {
        "generation_id": "macro-observations-g1",
        "pointer_sha256": "3" * 64,
        "parquet_sha256": "4" * 64,
        "manifest_sha256": "5" * 64,
        "content_set_hash": "6" * 64,
        "row_count": 39,
    }
    row_volatility = float(row["volatility_percentile"])
    controls = build_v15_macro_controls(
        snapshot_payload,
        volatility_percentile=(
            row_volatility if 0.0 <= row_volatility <= 100.0 else 45.0
        ),
        observation_generation=observation_generation,
    )
    snapshot_path = generation / "macro_snapshot.json"
    snapshot_path.write_bytes(
        macro_mart._canonical_json_bytes(snapshot_payload) + b"\n"
    )
    snapshot_sha = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    controls_path = generation / "v15_controls.json"
    controls_path.write_bytes(
        macro_mart._canonical_json_bytes(controls) + b"\n"
    )
    controls_sha = hashlib.sha256(controls_path.read_bytes()).hexdigest()
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
        "macro_snapshot_sha256": snapshot_sha,
        "v15_controls_sha256": controls_sha,
        "macro_observation_pointer_sha256": observation_generation[
            "pointer_sha256"
        ],
        "v15_controls_semantic_sha256": controls["semantic_sha256"],
        "transform_version": macro_mart.V15_TRANSFORM_VERSION,
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
        "transform_version": macro_mart.V15_TRANSFORM_VERSION,
        "market_input_files": _MARKET_INPUT_FILES,
        "market_input_files_sha256": _MARKET_FILES_SHA,
        "market_formula_universe": formula_universe,
        "market_formula_universe_sha256": formula_universe_sha,
        "macro_snapshot_path": "macro_snapshot.json",
        "macro_snapshot_sha256": snapshot_sha,
        "v15_controls_path": "v15_controls.json",
        "v15_controls_sha256": controls_sha,
        "v15_controls_schema_version": controls["schema_version"],
        "v15_controls_semantic_sha256": controls["semantic_sha256"],
        "macro_observation_generation": observation_generation,
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
                        "macro_snapshot_sha256": snapshot_sha,
                        "v15_controls_sha256": controls_sha,
                        "v15_controls_semantic_sha256": controls[
                            "semantic_sha256"
                        ],
                        "macro_observation_generation": observation_generation,
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return catalog, table, manifest, provider_path
