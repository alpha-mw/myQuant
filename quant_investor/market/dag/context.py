from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass, field, fields as dataclass_fields
from datetime import timedelta
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from quant_investor.agent_protocol import (
    ActionLabel,
    AgentStatus,
    BranchVerdict,
    ConfidenceLabel,
    DataQualityIssue,
    Direction,
    GlobalContext,
)
from quant_investor.branch_contracts import BranchResult
from quant_investor.config import config
from quant_investor.funnel.deterministic_funnel import FunnelConfig, FunnelOutput
from quant_investor.factors.runtime import (
    MinedFactorScorer,
    ProductionEvaluationContext,
    _mint_production_evaluation_context,
    production_frame_validation_blocker,
    production_symbol_set_sha256,
    validate_production_evaluation_context,
)
from quant_investor.market.config import get_market_settings
from quant_investor.market.dag.packets import (
    _clamp,
    _build_cross_section_quant,
    _build_global_quant_verdict,
    _build_market_snapshot,
    _build_quant_branch_result_with_validation,
    _build_symbol_tradability,
)
from quant_investor.market.data_quality import build_data_quality_diagnostics
from quant_investor.market.pit_universe import (
    PIT_UNIVERSE_MANIFEST_SCHEMA_VERSION,
    PIT_UNIVERSE_SCHEMA_VERSION,
    PITUniverseRecord,
    SUPPORTED_LIST_STATUSES,
    evaluate_listing_status,
    records_by_symbol,
)
from quant_investor.market.branch_readiness import (
    STATUS_BLOCK,
    assess_macro_readiness,
    assess_branch_data_readiness,
    load_macro_record,
    macro_generation_identity,
    write_branch_readiness_report,
)
from quant_investor.market.read_result import MarketDataReadResult
from quant_investor.market.runtime_profile import profile_stage
from quant_investor.llm_gateway import detect_provider
from quant_investor.model_roles import ModelRoleResolution
from quant_investor.macro.v15_controls import (
    V15_MACRO_CONTROL_SCHEMA_VERSION,
)
from quant_investor.reporting.run_artifacts import build_model_role_metadata
from quant_investor.regime.engine import MarkovRegimeEngine
from quant_investor.regime.scope import (
    REGIME_SCOPE_INSUFFICIENT,
    REGIME_SCOPE_MARKET_REFERENCE,
    RegimeScope,
    build_regime_scope,
    deterministic_symbol_sample,
    reference_universe_key_for_market,
)


DAG_RUNTIME_PRICE_VOLUME_COLUMNS: tuple[str, ...] = (
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
    "adj_close",
)
DAG_RUNTIME_LOOKBACK_CALENDAR_DAYS = 420
DAG_SINGLE_NAME_WEIGHT_CAP = 0.50
DAG_NEUTRAL_TARGET_EXPOSURE = 0.55


def _validated_pinned_macro_controls(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    raw = manifest.get("v15_controls")
    if not isinstance(raw, Mapping):
        raise ValueError("macro_v15_controls_missing_or_invalid")
    controls = dict(raw)
    if (
        controls.get("schema_version")
        != V15_MACRO_CONTROL_SCHEMA_VERSION
        or controls.get("production_control_projection") is not True
        or any(
            field_name not in controls
            for field_name in (
                "macro_score",
                "liquidity_score",
                "volatility_percentile",
                "policy_signal",
                "semantic_sha256",
            )
        )
    ):
        raise ValueError("macro_v15_controls_missing_or_invalid")
    return controls


@dataclass
class MarketContextState:
    all_symbols: list[str]
    read_results: dict[str, MarketDataReadResult]
    frames: dict[str, pd.DataFrame]
    tradability_snapshot: dict[str, dict[str, Any]]
    data_quality_issues: list[DataQualityIssue]
    quarantined_symbols: list[str]
    researchable_symbols: list[str]
    candidate_symbols: list[str]
    provider_health: dict[str, dict[str, Any]]
    market_snapshot: dict[str, Any]
    macro_verdict: BranchVerdict
    global_quant_verdict: BranchVerdict
    quant_result: BranchResult
    global_context: GlobalContext
    model_roles: Any
    funnel_output: FunnelOutput
    resolver_snapshot: dict[str, Any] = field(default_factory=dict)
    branch_data_readiness: dict[str, Any] = field(default_factory=dict)
    branch_data_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class _MarkovReferenceInput:
    scope: RegimeScope
    frames: dict[str, pd.DataFrame]
    tradability_snapshot: dict[str, dict[str, Any]]
    cross_section_quant: dict[str, Any]


def _is_quarantined_read_result(read_result: Any) -> bool:
    issues = list(getattr(read_result, "issues", []) or [])
    return bool(issues)


def _provider_label(resolution: ModelRoleResolution) -> str:
    metadata = dict(resolution.metadata or {})
    if (
        metadata.get("review_layer_mode") == "codex_handoff"
        or resolution.resolved_model == "codex-handoff"
    ):
        return "codex"
    try:
        return detect_provider(resolution.resolved_model)
    except Exception:
        return ""


def _compact_runtime_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return ""
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _runtime_lookback_start_date(
    latest_trade_date: Any,
    *,
    calendar_days: int = DAG_RUNTIME_LOOKBACK_CALENDAR_DAYS,
) -> str:
    compact = _compact_runtime_date(latest_trade_date)
    if not compact:
        return ""
    parsed = pd.to_datetime(compact, format="%Y%m%d", errors="coerce")
    if pd.isna(parsed):
        return ""
    return (pd.Timestamp(parsed) - timedelta(days=max(int(calendar_days), 1))).strftime("%Y%m%d")


def _call_accepts_keyword(callable_obj: Callable[..., Any], keyword: str) -> bool:
    try:
        parameters = signature(callable_obj).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(
        parameter.name == keyword or parameter.kind == Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _read_symbol_frames_with_projection(
    batch_reader: Callable[..., Any],
    symbols: list[str],
    *,
    universe_key: str,
    start_date: str = "",
    end_date: str = "",
) -> dict[str, MarketDataReadResult]:
    kwargs: dict[str, Any] = {"universe_key": universe_key}
    if _call_accepts_keyword(batch_reader, "columns"):
        kwargs["columns"] = DAG_RUNTIME_PRICE_VOLUME_COLUMNS
    if start_date and _call_accepts_keyword(batch_reader, "start_date"):
        kwargs["start_date"] = start_date
    if end_date and _call_accepts_keyword(batch_reader, "end_date"):
        kwargs["end_date"] = end_date
    return dict(batch_reader(symbols, **kwargs) or {})


def _read_symbol_frame_with_projection(
    reader: Callable[..., Any],
    symbol: str,
    *,
    universe_key: str,
    start_date: str = "",
    end_date: str = "",
) -> MarketDataReadResult:
    kwargs: dict[str, Any] = {"universe_key": universe_key}
    if _call_accepts_keyword(reader, "columns"):
        kwargs["columns"] = DAG_RUNTIME_PRICE_VOLUME_COLUMNS
    if start_date and _call_accepts_keyword(reader, "start_date"):
        kwargs["start_date"] = start_date
    if end_date and _call_accepts_keyword(reader, "end_date"):
        kwargs["end_date"] = end_date
    return reader(symbol, **kwargs)


def _researchable_frame_subset(
    frames: Mapping[str, pd.DataFrame],
    researchable_symbols: list[str],
) -> dict[str, pd.DataFrame]:
    """Preserve researchable order while excluding every quarantined frame."""

    return {
        symbol: frames[symbol]
        for symbol in researchable_symbols
        if symbol in frames
    }


def _readback_sha256(path_value: Any) -> tuple[str, str, bytes] | None:
    try:
        raw_path = Path(str(path_value or "")).expanduser()
        if raw_path.is_symlink():
            return None
        path = raw_path.resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if not path.is_file():
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    return str(path), hashlib.sha256(raw).hexdigest(), raw


def _resolved_path_string(path_value: Any) -> str:
    if not isinstance(path_value, (str, Path)) or not str(path_value).strip():
        return ""
    try:
        return str(Path(path_value).expanduser().resolve())
    except (OSError, RuntimeError, ValueError):
        return ""


def _resolved_snapshot_path_string(path_value: Any, pointer_path: Any) -> str:
    if not isinstance(path_value, (str, Path)) or not str(path_value).strip():
        return ""
    raw_path = Path(path_value).expanduser()
    if raw_path.is_absolute():
        return _resolved_path_string(raw_path)
    candidates = [Path.cwd() / raw_path]
    pointer = _resolved_path_string(pointer_path)
    if pointer:
        candidates.extend(parent / raw_path for parent in Path(pointer).parents)
    for candidate in candidates:
        try:
            if candidate.exists():
                return str(candidate.resolve())
        except (OSError, RuntimeError):
            continue
    return _resolved_path_string(raw_path)


def _canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _build_production_evaluation_context(
    *,
    market: str,
    universe_key: str,
    symbols: list[str],
    reader_snapshot: Mapping[str, Any],
    scoped_data_snapshot: Mapping[str, Any],
    read_results: Mapping[str, MarketDataReadResult] | None = None,
) -> tuple[ProductionEvaluationContext | None, list[str]]:
    """Mint Quant context only after dual-source and artifact readback."""

    blockers: list[str] = []
    normalized_market = str(market or "").strip().upper()
    normalized_universe = str(universe_key or "").strip()
    normalized_symbols = [str(symbol) for symbol in symbols]
    raw_gate = scoped_data_snapshot.get("strict_parquet_gate")
    if isinstance(raw_gate, Mapping):
        gate = dict(raw_gate)
    else:
        gate = {}
        blockers.append("production_snapshot_gate_metadata_invalid")
    if reader_snapshot.get("healthy") is not True or gate.get("healthy") is not True:
        blockers.append("production_snapshot_not_healthy")
    for label, payload in (("reader", reader_snapshot), ("gate", gate)):
        if str(payload.get("status") or "").upper() != "OK":
            blockers.append(f"production_snapshot_{label}_status_invalid")
        raw_blockers = payload.get("blockers")
        if not isinstance(raw_blockers, list) or raw_blockers:
            blockers.append(f"production_snapshot_{label}_blockers_invalid")
    scoped_market = str(scoped_data_snapshot.get("market") or normalized_market).upper()
    scoped_universe = str(
        scoped_data_snapshot.get("universe_key") or normalized_universe
    )
    if scoped_market != normalized_market:
        blockers.append("production_snapshot_market_mismatch")
    if scoped_universe != normalized_universe:
        blockers.append("production_snapshot_universe_mismatch")

    evaluation_as_of = _compact_runtime_date(
        scoped_data_snapshot.get("local_latest_trade_date")
        or scoped_data_snapshot.get("latest_trade_date")
    )
    reader_latest = _compact_runtime_date(
        reader_snapshot.get("latest_complete_trade_date")
    )
    gate_latest = _compact_runtime_date(gate.get("latest_complete_trade_date"))
    if not evaluation_as_of or reader_latest != evaluation_as_of or gate_latest != evaluation_as_of:
        blockers.append("production_latest_complete_trade_date_mismatch")
    reader_snapshot_id = str(reader_snapshot.get("snapshot_id") or "").strip()
    gate_snapshot_id = str(gate.get("snapshot_id") or "").strip()
    if not reader_snapshot_id or reader_snapshot_id != gate_snapshot_id:
        blockers.append("production_snapshot_id_mismatch")
    for path_key in ("latest_pointer_path", "manifest_path"):
        reader_path = str(reader_snapshot.get(path_key) or "").strip()
        gate_path = str(gate.get(path_key) or "").strip()
        if gate_path and reader_path != gate_path:
            blockers.append(f"production_snapshot_{path_key}_mismatch")

    artifact_hashes: dict[str, str] = {}
    artifact_paths: dict[str, str] = {}
    pointer_readback = _readback_sha256(reader_snapshot.get("latest_pointer_path"))
    manifest_readback = _readback_sha256(reader_snapshot.get("manifest_path"))
    if pointer_readback is None:
        blockers.append("production_snapshot_pointer_readback_missing")
    else:
        artifact_paths["snapshot_pointer"] = pointer_readback[0]
        artifact_hashes["snapshot_pointer"] = pointer_readback[1]
    if manifest_readback is None:
        blockers.append("production_snapshot_manifest_readback_missing")
    else:
        artifact_paths["snapshot_manifest"] = manifest_readback[0]
        artifact_hashes["snapshot_manifest"] = manifest_readback[1]
    snapshot_payloads: dict[str, Mapping[str, Any]] = {}
    for label, readback in (
        ("pointer", pointer_readback),
        ("manifest", manifest_readback),
    ):
        if readback is None:
            continue
        try:
            payload = json.loads(readback[2].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            blockers.append(f"production_snapshot_{label}_json_invalid")
            continue
        if not isinstance(payload, Mapping):
            blockers.append(f"production_snapshot_{label}_json_invalid")
            continue
        snapshot_payloads[label] = payload
        if str(payload.get("snapshot_id") or "") != reader_snapshot_id:
            blockers.append(f"production_snapshot_{label}_id_mismatch")
        payload_latest = _compact_runtime_date(
            payload.get("latest_complete_trade_date")
            or payload.get("latest_trade_date")
        )
        if payload_latest != evaluation_as_of:
            blockers.append(f"production_snapshot_{label}_date_mismatch")
        if str(payload.get("status") or "").upper() != "OK":
            blockers.append(f"production_snapshot_{label}_status_invalid")
        raw_payload_blockers = payload.get("blockers")
        if not isinstance(raw_payload_blockers, list) or raw_payload_blockers:
            blockers.append(f"production_snapshot_{label}_blockers_invalid")

    pointer_payload = snapshot_payloads.get("pointer", {})
    manifest_payload = snapshot_payloads.get("manifest", {})
    if manifest_payload.get("readback_validated") is not True:
        blockers.append("production_snapshot_manifest_not_readback_validated")
    if str(manifest_payload.get("market") or "").upper() != normalized_market:
        blockers.append("production_snapshot_manifest_market_mismatch")
    for payload in (reader_snapshot, gate, pointer_payload):
        payload_market = str(payload.get("market") or "").upper()
        if payload_market and payload_market != normalized_market:
            blockers.append("production_snapshot_market_mismatch")

    def serving_path(payload: Mapping[str, Any]) -> Any:
        return payload.get("serving_root") or payload.get("derived_serving_root")
    path_sources = {
        "manifest_path": (
            reader_snapshot.get("manifest_path"),
            gate.get("manifest_path"),
            pointer_payload.get("manifest_path"),
            manifest_payload.get("manifest_path"),
        ),
        "table_root": (
            reader_snapshot.get("table_root"),
            gate.get("table_root"),
            pointer_payload.get("table_root"),
            manifest_payload.get("table_root"),
        ),
        "serving_root": (
            serving_path(reader_snapshot),
            serving_path(gate),
            serving_path(pointer_payload),
            serving_path(manifest_payload),
        ),
    }
    verified_snapshot_paths: dict[str, str] = {}
    for label, raw_paths in path_sources.items():
        normalized_paths = [
            _resolved_snapshot_path_string(
                path,
                reader_snapshot.get("latest_pointer_path"),
            )
            for path in raw_paths
        ]
        if any(not path for path in normalized_paths) or len(set(normalized_paths)) != 1:
            blockers.append(f"production_snapshot_{label}_mismatch")
            continue
        verified_snapshot_paths[label] = normalized_paths[0]

    open_day_proof_sha256 = ""
    raw_calendar = scoped_data_snapshot.get("open_day_calendar")
    if not isinstance(raw_calendar, Mapping):
        blockers.append("production_open_day_calendar_metadata_invalid")
    else:
        calendar_readback = _readback_sha256(raw_calendar.get("path"))
        if calendar_readback is None:
            blockers.append("production_open_day_calendar_readback_missing")
        else:
            artifact_paths["open_day_calendar"] = calendar_readback[0]
            artifact_hashes["open_day_calendar"] = calendar_readback[1]
            open_day_proof_sha256 = calendar_readback[1]
            try:
                calendar_payload = json.loads(
                    calendar_readback[2].decode("utf-8")
                )
            except (UnicodeDecodeError, json.JSONDecodeError):
                calendar_payload = None
            if not isinstance(calendar_payload, Mapping):
                blockers.append("production_open_day_calendar_json_invalid")
            else:
                if set(calendar_payload) != {
                    "schema_version",
                    "market",
                    "open_dates",
                }:
                    blockers.append(
                        "production_open_day_calendar_fields_invalid"
                    )
                if calendar_payload.get("schema_version") != "market-open-days.v1":
                    blockers.append("production_open_day_calendar_schema_invalid")
                if str(calendar_payload.get("market") or "").upper() != normalized_market:
                    blockers.append("production_open_day_calendar_market_mismatch")
                raw_open_dates = calendar_payload.get("open_dates")
                if (
                    not isinstance(raw_open_dates, list)
                    or not raw_open_dates
                ):
                    blockers.append("production_open_day_calendar_dates_invalid")
                else:
                    open_dates: set[str] = set()
                    invalid_open_date = False
                    for value in raw_open_dates:
                        if (
                            not isinstance(value, str)
                            or re.fullmatch(r"[0-9]{8}", value) is None
                        ):
                            invalid_open_date = True
                            break
                        parsed = pd.to_datetime(
                            value,
                            format="%Y%m%d",
                            errors="coerce",
                        )
                        if (
                            pd.isna(parsed)
                            or int(parsed.dayofweek) >= 5
                        ):
                            invalid_open_date = True
                            break
                        open_dates.add(value)
                    if invalid_open_date or len(open_dates) != len(raw_open_dates):
                        blockers.append("production_open_day_calendar_dates_invalid")
                    elif evaluation_as_of not in open_dates:
                        blockers.append("production_evaluation_as_of_not_open_day")

    provenance: dict[str, dict[str, Any]] = {}
    for symbol in normalized_symbols:
        read_result = (read_results or {}).get(symbol)
        if read_results is None:
            break
        if read_result is None:
            blockers.append(f"production_read_result_missing:{symbol}")
            continue
        if not isinstance(read_result, MarketDataReadResult):
            blockers.append(f"production_read_result_invalid:{symbol}")
            continue
        if not isinstance(read_result.metadata, Mapping):
            blockers.append(f"production_read_result_metadata_invalid:{symbol}")
            continue
        metadata = dict(read_result.metadata)
        if (
            str(read_result.symbol) != symbol
            or str(read_result.universe_key) != normalized_universe
            or str(metadata.get("snapshot_id") or "") != reader_snapshot_id
            or _compact_runtime_date(metadata.get("latest_complete_trade_date"))
            != evaluation_as_of
        ):
            blockers.append(f"production_read_result_provenance_mismatch:{symbol}")
            continue
        if not isinstance(read_result.resolver_trace, Mapping):
            blockers.append(f"production_read_result_resolver_invalid:{symbol}")
            continue
        storage_layer = str(metadata.get("storage_layer") or "")
        resolution_strategy = str(
            metadata.get("resolution_strategy")
            or read_result.resolver_trace.get("resolution_strategy")
            or ""
        )
        resolved_path = _resolved_path_string(read_result.path)
        if metadata.get("resolved") is not True:
            blockers.append(f"production_read_result_not_resolved:{symbol}")
            continue
        if storage_layer == "canonical_batch":
            expected_path = verified_snapshot_paths.get("table_root", "")
            mode_valid = resolution_strategy == "strict_parquet_canonical_batch"
        elif storage_layer == "serving":
            serving_root = verified_snapshot_paths.get("serving_root", "")
            expected_path = _resolved_path_string(
                Path(serving_root) / f"symbol={symbol}" / "bars.parquet"
            ) if serving_root else ""
            mode_valid = resolution_strategy == "strict_parquet_serving"
        else:
            expected_path = ""
            mode_valid = False
        if not mode_valid:
            blockers.append(f"production_read_result_storage_mode_invalid:{symbol}")
            continue
        if not resolved_path or resolved_path != expected_path:
            blockers.append(f"production_read_result_path_mismatch:{symbol}")
            continue
        provenance[symbol] = {
            "symbol": symbol,
            "universe_key": normalized_universe,
            "snapshot_id": reader_snapshot_id,
            "latest_complete_trade_date": evaluation_as_of,
            "storage_layer": storage_layer,
            "resolution_strategy": resolution_strategy,
            "resolved": True,
            "path": resolved_path,
        }
    read_result_provenance_sha256 = ""
    if read_results is not None and len(provenance) == len(normalized_symbols):
        read_result_provenance_sha256 = _canonical_payload_sha256(provenance)
    else:
        blockers.append("production_read_result_provenance_incomplete")

    pit_status = "not_applicable"
    pit_as_of = ""
    pit_proof_sha256 = ""
    pit_na_reason = "market_not_cn"
    if normalized_market == "CN":
        pit_status = "verified"
        pit_na_reason = ""
        raw_pit = scoped_data_snapshot.get("pit_universe")
        if isinstance(raw_pit, Mapping):
            pit = dict(raw_pit)
        else:
            pit = {}
            blockers.append("production_cn_pit_metadata_invalid")
        if not bool(getattr(config, "PIT_UNIVERSE_ENABLED", False)):
            blockers.append("production_cn_pit_membership_disabled")
        if not bool(getattr(config, "PIT_UNIVERSE_REQUIRED", False)):
            blockers.append("production_cn_pit_membership_not_required")
        if (
            pit.get("enabled") is not True
            or pit.get("required") is not True
            or pit.get("status") != "applied"
        ):
            blockers.append("production_cn_pit_membership_not_applied")
        pit_as_of = _compact_runtime_date(pit.get("as_of"))
        if pit_as_of != evaluation_as_of:
            blockers.append("production_pit_membership_as_of_mismatch")
        missing_count = pit.get("missing_count")
        coverage_ratio = pit.get("coverage_ratio")
        if (
            isinstance(missing_count, bool)
            or not isinstance(missing_count, int)
            or missing_count != 0
            or isinstance(coverage_ratio, bool)
            or not isinstance(coverage_ratio, (int, float))
            or float(coverage_ratio) != 1.0
        ):
            blockers.append("production_cn_pit_membership_partial")
        raw_statuses = pit.get("statuses")
        if isinstance(raw_statuses, Mapping):
            statuses = dict(raw_statuses)
        else:
            statuses = {}
            blockers.append("production_cn_pit_statuses_invalid")
        claimed_statuses: dict[str, dict[str, Any]] = {}
        for symbol in normalized_symbols:
            status = statuses.get(symbol)
            if not isinstance(status, Mapping):
                blockers.append(f"production_cn_pit_status_missing:{symbol}")
                continue
            if (
                str(status.get("symbol") or "") != symbol
                or _compact_runtime_date(status.get("date")) != evaluation_as_of
                or status.get("in_universe") is not True
                or status.get("research_eligible") is not True
            ):
                blockers.append(f"production_cn_pit_status_mismatch:{symbol}")
                continue
            claimed_statuses[symbol] = dict(status)
        pit_manifest_readback = _readback_sha256(pit.get("manifest_path"))
        pit_canonical_readback = _readback_sha256(pit.get("canonical_path"))
        authoritative_statuses: dict[str, dict[str, Any]] = {}
        if pit_manifest_readback is None or pit_canonical_readback is None:
            blockers.append("production_cn_pit_artifact_readback_missing")
        else:
            artifact_paths["pit_manifest"] = pit_manifest_readback[0]
            artifact_hashes["pit_manifest"] = pit_manifest_readback[1]
            artifact_paths["pit_canonical"] = pit_canonical_readback[0]
            artifact_hashes["pit_canonical"] = pit_canonical_readback[1]
            manifest_source = ""
            manifest_observed_at = ""
            try:
                pit_manifest = json.loads(
                    pit_manifest_readback[2].decode("utf-8")
                )
            except (UnicodeDecodeError, json.JSONDecodeError):
                pit_manifest = None
            if not isinstance(pit_manifest, Mapping):
                blockers.append("production_cn_pit_manifest_mismatch")
            else:
                manifest_source = str(pit_manifest.get("source") or "")
                manifest_observed_at = str(
                    pit_manifest.get("observed_at") or ""
                )
                manifest_source_run_id = str(
                    pit_manifest.get("source_run_id") or ""
                )
                if (
                    pit_manifest.get("schema_version")
                    != PIT_UNIVERSE_MANIFEST_SCHEMA_VERSION
                    or pit_manifest.get("membership_schema_version")
                    != PIT_UNIVERSE_SCHEMA_VERSION
                    or manifest_source_run_id
                    != str(pit.get("snapshot_id") or "")
                    or isinstance(pit_manifest.get("row_count"), bool)
                    or not isinstance(pit_manifest.get("row_count"), int)
                ):
                    blockers.append("production_cn_pit_manifest_mismatch")
                if manifest_source != "tushare.stock_basic":
                    blockers.append(
                        "production_cn_pit_manifest_source_invalid"
                    )
                if not manifest_observed_at:
                    blockers.append(
                        "production_cn_pit_manifest_observed_at_invalid"
                    )
                try:
                    declared_canonical_path = Path(
                        str(pit_manifest.get("canonical_path") or "")
                    ).expanduser().resolve()
                except (OSError, RuntimeError, ValueError):
                    declared_canonical_path = None
                if (
                    declared_canonical_path is None
                    or str(declared_canonical_path) != pit_canonical_readback[0]
                ):
                    blockers.append("production_cn_pit_manifest_path_mismatch")
            try:
                canonical_frame = pd.read_parquet(io.BytesIO(pit_canonical_readback[2]))
            except Exception:
                canonical_frame = None
                blockers.append("production_cn_pit_canonical_parquet_invalid")
            canonical_records: list[PITUniverseRecord] = []
            if canonical_frame is not None:
                expected_canonical_columns = [
                    item.name for item in dataclass_fields(PITUniverseRecord)
                ]
                if list(canonical_frame.columns) != expected_canonical_columns:
                    blockers.append(
                        "production_cn_pit_canonical_columns_mismatch"
                    )
                try:
                    canonical_records = [
                        PITUniverseRecord.from_dict(row)
                        for row in canonical_frame.to_dict(orient="records")
                    ]
                except Exception:
                    canonical_records = []
                    blockers.append(
                        "production_cn_pit_canonical_records_invalid"
                    )
            if canonical_frame is not None:
                if (
                    "schema_version" not in canonical_frame.columns
                    or not canonical_frame["schema_version"].eq(
                        PIT_UNIVERSE_SCHEMA_VERSION
                    ).all()
                ):
                    blockers.append(
                        "production_cn_pit_canonical_schema_mismatch"
                    )
                if (
                    isinstance(pit_manifest, Mapping)
                    and pit_manifest.get("row_count") != len(canonical_frame)
                ):
                    blockers.append("production_cn_pit_manifest_row_count_mismatch")
                canonical_symbols = [record.symbol for record in canonical_records]
                if (
                    any(not symbol for symbol in canonical_symbols)
                    or len(canonical_symbols) != len(set(canonical_symbols))
                ):
                    blockers.append("production_cn_pit_canonical_symbols_invalid")
                canonical_by_symbol = records_by_symbol(canonical_records)
                expected_source_run_id = str(pit.get("snapshot_id") or "")
                if any(
                    record.source_run_id != expected_source_run_id
                    for record in canonical_records
                ):
                    blockers.append(
                        "production_cn_pit_canonical_source_run_mismatch"
                    )
                if any(
                    record.source_list_status not in SUPPORTED_LIST_STATUSES
                    for record in canonical_records
                ):
                    blockers.append(
                        "production_cn_pit_canonical_list_status_invalid"
                    )
                if any(
                    record.source != manifest_source
                    or record.source != "tushare.stock_basic"
                    for record in canonical_records
                ):
                    blockers.append(
                        "production_cn_pit_canonical_source_mismatch"
                    )
                if any(
                    record.observed_at != manifest_observed_at
                    for record in canonical_records
                ):
                    blockers.append(
                        "production_cn_pit_canonical_observed_at_mismatch"
                    )
                for symbol in normalized_symbols:
                    record = canonical_by_symbol.get(symbol)
                    if record is None:
                        blockers.append(
                            f"production_cn_pit_canonical_record_missing:{symbol}"
                        )
                        continue
                    authoritative = evaluate_listing_status(
                        record,
                        symbol=symbol,
                        as_of=evaluation_as_of,
                    ).to_dict()
                    claimed = claimed_statuses.get(symbol)
                    if (
                        record.source_run_id != expected_source_run_id
                        or claimed != authoritative
                        or authoritative.get("in_universe") is not True
                        or authoritative.get("research_eligible") is not True
                    ):
                        blockers.append(
                            f"production_cn_pit_canonical_status_mismatch:{symbol}"
                        )
                        continue
                    authoritative_statuses[symbol] = authoritative
        if len(authoritative_statuses) == len(normalized_symbols):
            pit_proof_sha256 = _canonical_payload_sha256(
                {
                    "as_of": pit_as_of,
                    "snapshot_id": pit.get("snapshot_id"),
                    "statuses": authoritative_statuses,
                    "manifest_sha256": artifact_hashes.get("pit_manifest", ""),
                    "canonical_sha256": artifact_hashes.get("pit_canonical", ""),
                }
            )

    resolved_artifact_owners: dict[str, str] = {}
    resolved_artifact_file_owners: dict[tuple[int, int], str] = {}
    for name, artifact_path in artifact_paths.items():
        resolved_artifact_path = _resolved_path_string(artifact_path)
        prior_name = resolved_artifact_owners.get(resolved_artifact_path)
        if prior_name is not None:
            blockers.append("production_verified_artifact_path_reused")
            if "open_day_calendar" in {name, prior_name}:
                blockers.append(
                    "production_open_day_calendar_not_independent"
                )
        else:
            resolved_artifact_owners[resolved_artifact_path] = name
        try:
            file_stat = Path(resolved_artifact_path).stat()
        except (OSError, ValueError):
            blockers.append(
                f"production_verified_artifact_identity_invalid:{name}"
            )
            continue
        file_identity = (int(file_stat.st_dev), int(file_stat.st_ino))
        prior_file_name = resolved_artifact_file_owners.get(file_identity)
        if prior_file_name is not None:
            blockers.append("production_verified_artifact_file_reused")
            if "open_day_calendar" in {name, prior_file_name}:
                blockers.append(
                    "production_open_day_calendar_not_independent"
                )
        else:
            resolved_artifact_file_owners[file_identity] = name

    if blockers:
        return None, list(dict.fromkeys(blockers))
    assert manifest_readback is not None
    context = _mint_production_evaluation_context(
        evaluation_as_of=evaluation_as_of,
        market=normalized_market,
        universe_key=normalized_universe,
        universe_sha256=production_symbol_set_sha256(normalized_symbols),
        snapshot_id=reader_snapshot_id,
        latest_complete_trade_date=evaluation_as_of,
        pit_membership_status=pit_status,
        pit_membership_as_of=pit_as_of,
        pit_membership_proof_sha256=pit_proof_sha256,
        pit_membership_not_applicable_reason=pit_na_reason,
        open_day_proof_sha256=open_day_proof_sha256,
        read_result_provenance_sha256=read_result_provenance_sha256,
        verified_artifact_paths=artifact_paths,
        verified_artifact_sha256s=artifact_hashes,
    )
    context_blockers = validate_production_evaluation_context(
        context,
        expected_symbols=normalized_symbols,
    )
    if context_blockers:
        return None, context_blockers
    return context, []


def _frame_summaries_from_tradability(
    tradability_snapshot: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for symbol, payload in tradability_snapshot.items():
        state = dict(payload.get("market_state", {}) or {})
        summaries[str(symbol)] = {
            "rows": int(state.get("rows", 0) or 0),
            "latest_close": float(state.get("latest_close", 0.0) or 0.0),
            "average_return": float(state.get("average_return", 0.0) or 0.0),
            "volatility": float(state.get("volatility", 0.0) or 0.0),
        }
    return summaries


def _macro_v2_observer_metadata(*, market: str, as_of: str) -> dict[str, Any]:
    """Build one run-level observer diagnostic without touching decisions."""

    enabled = bool(getattr(config, "MACRO_V2_OBSERVER_ENABLED", False))
    kill_switch = bool(
        getattr(config, "MACRO_V2_OBSERVER_KILL_SWITCH", True)
    )
    production_enabled = bool(
        getattr(config, "MACRO_V2_PRODUCTION_ENABLED", False)
    )
    production_kill_switch = bool(
        getattr(config, "MACRO_V2_PRODUCTION_KILL_SWITCH", True)
    )
    base = {
        "schema_version": "macro-observer-runtime.v2",
        "enabled": enabled,
        "kill_switch": kill_switch,
        "active": False,
        "production_enabled": production_enabled,
        "production_kill_switch": production_kill_switch,
        "observer_only": True,
        "production_eligible": False,
        "applied": False,
    }
    if not enabled or kill_switch:
        base["reason"] = (
            "kill_switch_active" if kill_switch else "observer_disabled"
        )
        return base
    observations_root = Path(
        str(
            getattr(
                config,
                "MACRO_V2_OBSERVATIONS_PATH",
                "data/parquet/cn/macro_observations",
            )
        )
    )
    try:
        from quant_investor.macro.observer import build_macro_observer
        from quant_investor.macro.store import load_observations

        observations, generation = load_observations(observations_root)
        return build_macro_observer(
            observations,
            market=market,
            as_of=as_of,
            enabled=True,
            kill_switch=False,
            persist=True,
            output_root=str(
                getattr(
                    config,
                    "MACRO_V2_OBSERVER_OUTPUT_DIR",
                    "results/v15/macro_observer",
                )
            ),
            production_enabled=production_enabled,
            production_kill_switch=production_kill_switch,
            generation_provenance=generation,
        )
    except Exception as exc:
        base["reason"] = "observer_build_failed"
        base["blockers"] = [f"{type(exc).__name__}:{exc}"]
        return base


def _insufficient_markov_reference_input(
    *,
    market: str,
    universe_key: str,
    requested_symbol_count: int,
    explicit_symbol_count: int,
    unsampled_symbol_count: int,
    sampled: bool,
    min_market_sample: int,
    diagnostics: list[str],
) -> _MarkovReferenceInput:
    scope = build_regime_scope(
        market=market,
        base_universe_key=universe_key,
        source_universe_key=universe_key,
        requested_symbol_count=requested_symbol_count,
        source_symbol_count=0,
        explicit_symbol_count=explicit_symbol_count,
        unsampled_symbol_count=unsampled_symbol_count,
        sampled=sampled,
        min_market_sample=min_market_sample,
        source_description="no_valid_market_reference",
        diagnostics=diagnostics,
        force_scope=REGIME_SCOPE_INSUFFICIENT,
    )
    return _MarkovReferenceInput(
        scope=scope,
        frames={},
        tradability_snapshot={},
        cross_section_quant={
            "candidate_count": 0,
            "sample_count": 0,
            "average_return": 0.0,
            "average_volatility": 0.0,
            "breadth": 0.0,
        },
    )


def _build_reference_tradability(
    *,
    read_results: Mapping[str, MarketDataReadResult],
    trend_windows: tuple[int, ...],
    volume_spike_threshold: float,
    breakout_distance_pct: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], list[str]]:
    reference_frames: dict[str, pd.DataFrame] = {}
    reference_tradability: dict[str, dict[str, Any]] = {}
    diagnostics: list[str] = []
    for symbol in sorted(read_results):
        read_result = read_results[symbol]
        if _is_quarantined_read_result(read_result):
            diagnostics.append(f"markov_reference_symbol_quarantined:{symbol}")
            continue
        frame = read_result.frame
        if frame is None or frame.empty:
            diagnostics.append(f"markov_reference_symbol_empty:{symbol}")
            continue
        reference_frames[symbol] = frame
        reference_tradability[symbol] = _build_symbol_tradability(
            symbol,
            read_result,
            trend_windows=trend_windows,
            volume_spike_threshold=volume_spike_threshold,
            breakout_distance_pct=breakout_distance_pct,
        )
    return reference_frames, reference_tradability, diagnostics[:20]


def _resolve_markov_reference_input(
    *,
    market: str,
    universe_key: str,
    requested_symbols: list[str],
    current_frames: Mapping[str, pd.DataFrame],
    current_tradability_snapshot: Mapping[str, Mapping[str, Any]],
    current_cross_section_quant: Mapping[str, Any],
    shared_reader: Any,
    as_of: str,
    runtime_lookback_start_date: str,
    explicit_symbol_count: int,
    unsampled_symbol_count: int,
    sampled: bool,
    trend_windows: tuple[int, ...],
    volume_spike_threshold: float,
    breakout_distance_pct: float,
    runtime_profiler: Any | None = None,
) -> _MarkovReferenceInput:
    min_market_sample = max(int(getattr(config, "MARKOV_REGIME_MIN_MARKET_SAMPLE", 30) or 30), 1)
    max_reference_symbols = max(
        int(getattr(config, "MARKOV_REGIME_MAX_REFERENCE_SYMBOLS", 300) or 300),
        min_market_sample,
    )
    requested_count = len(requested_symbols)
    unsampled_count = int(unsampled_symbol_count or requested_count)
    current_scope = build_regime_scope(
        market=market,
        base_universe_key=universe_key,
        source_universe_key=universe_key,
        requested_symbol_count=requested_count,
        source_symbol_count=len(current_frames or {}),
        explicit_symbol_count=explicit_symbol_count,
        unsampled_symbol_count=unsampled_count,
        sampled=sampled,
        min_market_sample=min_market_sample,
        source_description="dag_current_universe",
    )
    if current_scope.regime_scope == "full_market" and current_scope.production_eligible:
        return _MarkovReferenceInput(
            scope=current_scope,
            frames=dict(current_frames),
            tradability_snapshot={
                str(symbol): dict(payload)
                for symbol, payload in current_tradability_snapshot.items()
            },
            cross_section_quant=dict(current_cross_section_quant),
        )

    diagnostics = list(current_scope.diagnostics)
    diagnostics.append(
        f"markov_requested_pool_not_market_scope:{current_scope.regime_scope}"
    )
    reference_universe_key = reference_universe_key_for_market(market, config)
    list_symbols = getattr(shared_reader, "list_symbols", None)
    batch_reader = getattr(shared_reader, "read_symbol_frames", None)
    if not callable(list_symbols) or not callable(batch_reader):
        diagnostics.append("markov_reference_reader_unavailable")
        return _insufficient_markov_reference_input(
            market=market,
            universe_key=universe_key,
            requested_symbol_count=requested_count,
            explicit_symbol_count=explicit_symbol_count,
            unsampled_symbol_count=unsampled_count,
            sampled=sampled,
            min_market_sample=min_market_sample,
            diagnostics=diagnostics,
        )

    try:
        reference_symbols = list_symbols(reference_universe_key)
    except Exception as exc:
        diagnostics.append(f"markov_reference_universe_list_failed:{exc}")
        return _insufficient_markov_reference_input(
            market=market,
            universe_key=universe_key,
            requested_symbol_count=requested_count,
            explicit_symbol_count=explicit_symbol_count,
            unsampled_symbol_count=unsampled_count,
            sampled=sampled,
            min_market_sample=min_market_sample,
            diagnostics=diagnostics,
        )
    selected_reference_symbols, reference_sampled, reference_unsampled_count = deterministic_symbol_sample(
        reference_symbols,
        max_reference_symbols,
    )
    if len(selected_reference_symbols) < min_market_sample:
        diagnostics.append(
            f"markov_reference_symbol_count_below_min:{len(selected_reference_symbols)}<{min_market_sample}"
        )
        return _insufficient_markov_reference_input(
            market=market,
            universe_key=universe_key,
            requested_symbol_count=requested_count,
            explicit_symbol_count=explicit_symbol_count,
            unsampled_symbol_count=reference_unsampled_count,
            sampled=reference_sampled,
            min_market_sample=min_market_sample,
            diagnostics=diagnostics,
        )

    with profile_stage(
        runtime_profiler,
        "dag_markov_reference_read",
        {
            "source_universe_key": reference_universe_key,
            "source_symbol_count": len(selected_reference_symbols),
            "sampled": reference_sampled,
        },
    ) as reference_metadata:
        reference_metadata["min_market_sample"] = min_market_sample
        reference_metadata["max_reference_symbols"] = max_reference_symbols
        reference_metadata["unsampled_symbol_count"] = reference_unsampled_count
        reference_read_results = _read_symbol_frames_with_projection(
            batch_reader,
            selected_reference_symbols,
            universe_key=reference_universe_key,
            start_date=runtime_lookback_start_date,
            end_date=as_of,
        )
        reference_metadata["batch_result_count"] = len(reference_read_results)

    reference_frames, reference_tradability, reference_notes = _build_reference_tradability(
        read_results=reference_read_results,
        trend_windows=trend_windows,
        volume_spike_threshold=volume_spike_threshold,
        breakout_distance_pct=breakout_distance_pct,
    )
    diagnostics.extend(reference_notes)
    reference_frame_summaries = _frame_summaries_from_tradability(reference_tradability)
    reference_cross_section = _build_cross_section_quant(
        reference_frames,
        frame_summaries=reference_frame_summaries,
    )
    reference_scope = build_regime_scope(
        market=market,
        base_universe_key=universe_key,
        source_universe_key=reference_universe_key,
        requested_symbol_count=requested_count,
        source_symbol_count=len(reference_frames),
        explicit_symbol_count=explicit_symbol_count,
        unsampled_symbol_count=reference_unsampled_count,
        sampled=reference_sampled,
        min_market_sample=min_market_sample,
        source_description="local_canonical_market_reference",
        diagnostics=diagnostics,
        force_scope=(
            REGIME_SCOPE_MARKET_REFERENCE
            if len(reference_frames) >= min_market_sample
            else REGIME_SCOPE_INSUFFICIENT
        ),
    )
    return _MarkovReferenceInput(
        scope=reference_scope,
        frames=reference_frames,
        tradability_snapshot=reference_tradability,
        cross_section_quant=reference_cross_section,
    )




def _holding_single_review_active(
    *,
    recall_context: Mapping[str, Any] | None,
    symbols: list[str],
) -> bool:
    context = recall_context if isinstance(recall_context, Mapping) else {}
    holding_symbol = str(context.get("holding_symbol") or "").strip().upper()
    if not holding_symbol:
        return False
    normalized = [
        str(symbol).strip().upper()
        for symbol in symbols
        if str(symbol).strip()
    ]
    return len(normalized) == 1 and normalized[0] == holding_symbol


def _blocked_macro_verdict(
    *,
    blockers: list[str],
    generation_identity: Mapping[str, str],
) -> BranchVerdict:
    diagnostic_notes = ["canonical_macro_readiness_blocked", *blockers]
    return BranchVerdict(
        agent_name="MacroAgent",
        thesis=(
            "Canonical Macro evidence is blocked; this neutral diagnostic "
            "does not authorize a decision."
        ),
        status=AgentStatus.VETOED,
        direction=Direction.NEUTRAL,
        action=ActionLabel.HOLD,
        confidence_label=ConfidenceLabel.VERY_LOW,
        final_score=0.0,
        final_confidence=0.0,
        diagnostic_notes=diagnostic_notes,
        metadata={
            "regime": "neutral",
            "target_gross_exposure": DAG_NEUTRAL_TARGET_EXPOSURE,
            "style_bias": "balanced_quality",
            "decision_authorized": False,
            "macro_data_readiness_status": STATUS_BLOCK,
            "canonical_macro_generation": dict(generation_identity),
            "blockers": list(blockers),
        },
    )


def _resolve_effective_data_state(
    *,
    scoped_data_snapshot: Mapping[str, Any],
    download_stage: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], str, str]:
    snapshot_latest_trade_date = str(
        scoped_data_snapshot.get("local_latest_trade_date")
        or scoped_data_snapshot.get("latest_trade_date")
        or ""
    )
    snapshot_freshness_mode = str(
        scoped_data_snapshot.get("freshness_mode") or "stable"
    )
    completeness_payload = (
        dict(
            download_stage.get("completeness_after")
            or download_stage.get("completeness_before")
            or {}
        )
        if download_stage
        else {}
    )
    effective_latest_trade_date = str(
        completeness_payload.get("latest_trade_date")
        or snapshot_latest_trade_date
    )
    effective_freshness_mode = str(
        completeness_payload.get("freshness_mode")
        or snapshot_freshness_mode
        or "stable"
    )
    return (
        completeness_payload,
        effective_latest_trade_date,
        effective_freshness_mode,
    )




def _prepare_market_context(
    *,
    market: str,
    universe_key: str,
    selected_categories: list[str],
    symbols: list[str],
    company_profile_map: Mapping[str, Mapping[str, Any]],
    shared_reader: Any,
    scoped_data_snapshot: Mapping[str, Any],
    download_stage: Mapping[str, Any] | None,
    enable_agent_layer: bool,
    agent_timeout: float,
    master_timeout: float,
    master_reasoning_effort: str,
    branch_model_resolution: ModelRoleResolution,
    master_model_resolution: ModelRoleResolution,
    branch_candidate_models: list[str],
    master_candidate_models: list[str],
    company_name_map: Mapping[str, str],
    funnel_profile: str,
    max_candidates: int,
    trend_windows: tuple[int, ...],
    volume_spike_threshold: float,
    breakout_distance_pct: float,
    sector_bucket_limit: int,
    macro_agent: Any,
    funnel_cls: Any,
    provider_health_detector: Callable[..., dict[str, dict[str, Any]]],
    runtime_profiler: Any | None = None,
    explicit_symbol_count: int = 0,
    unsampled_symbol_count: int = 0,
    sampled: bool = False,
    recall_context: Mapping[str, Any] | None = None,
) -> MarketContextState:
    settings = get_market_settings(market)
    all_symbols = list(symbols)
    resolver_snapshot = shared_reader.snapshot()

    read_results: dict[str, MarketDataReadResult] = {}
    frames: dict[str, pd.DataFrame] = {}
    tradability_snapshot: dict[str, dict[str, Any]] = {}
    data_quality_issues: list[DataQualityIssue] = []
    quarantined_symbols: list[str] = []
    researchable_symbols: list[str] = []
    industry_map: dict[str, str] = {}
    symbol_market_state: dict[str, dict[str, Any]] = {}
    batch_read_results: dict[str, MarketDataReadResult] = {}
    raw_read_results: dict[str, MarketDataReadResult] = {}
    frame_summaries: dict[str, dict[str, Any]] = {}
    quant_frame_validation_blockers: dict[str, str] = {}
    quant_contract_eligibility_blockers: dict[str, list[str]] = {}
    runtime_end_date = _compact_runtime_date(
        scoped_data_snapshot.get("local_latest_trade_date")
        or scoped_data_snapshot.get("latest_trade_date")
    )
    runtime_as_of = pd.to_datetime(
        runtime_end_date,
        format="%Y%m%d",
        errors="coerce",
    )
    runtime_lookback_start_date = _runtime_lookback_start_date(runtime_end_date)
    with profile_stage(
        runtime_profiler,
        "dag_batch_read",
        {"symbol_count": len(all_symbols), "universe_key": universe_key},
    ) as stage_metadata:
        stage_metadata["projected_columns"] = list(DAG_RUNTIME_PRICE_VOLUME_COLUMNS)
        stage_metadata["projected_column_count"] = len(DAG_RUNTIME_PRICE_VOLUME_COLUMNS)
        stage_metadata["runtime_lookback_calendar_days"] = DAG_RUNTIME_LOOKBACK_CALENDAR_DAYS
        if runtime_lookback_start_date:
            stage_metadata["runtime_lookback_start_date"] = runtime_lookback_start_date
        batch_reader = getattr(shared_reader, "read_symbol_frames", None)
        if callable(batch_reader):
            batch_read_results = _read_symbol_frames_with_projection(
                batch_reader,
                all_symbols,
                universe_key=universe_key,
                start_date=runtime_lookback_start_date,
                end_date=runtime_end_date,
            )
        per_symbol_fallback_count = 0
        for symbol in all_symbols:
            read_result = batch_read_results.get(symbol)
            if read_result is None:
                per_symbol_fallback_count += 1
                read_result = _read_symbol_frame_with_projection(
                    shared_reader.read_symbol_frame,
                    symbol,
                    universe_key=universe_key,
                    start_date=runtime_lookback_start_date,
                    end_date=runtime_end_date,
                )
            raw_read_results[symbol] = read_result
        stage_metadata["batch_result_count"] = len(batch_read_results)
        stage_metadata["per_symbol_fallback_count"] = per_symbol_fallback_count
    with profile_stage(
        runtime_profiler,
        "dag_tradability_snapshot",
        {"symbol_count": len(all_symbols), "universe_key": universe_key},
    ) as stage_metadata:
        for symbol in all_symbols:
            profile = dict(company_profile_map.get(symbol, {}) or {})
            read_result = raw_read_results[symbol]
            read_results[symbol] = read_result
            frames[symbol] = read_result.frame
            frame_blocker = None
            if runtime_end_date and not pd.isna(runtime_as_of):
                frame_blocker = production_frame_validation_blocker(
                    read_result.frame,
                    symbol=symbol,
                    evaluation_as_of=pd.Timestamp(runtime_as_of),
                )
            if frame_blocker:
                quant_frame_validation_blockers[symbol] = frame_blocker
                data_quality_issues.append(
                    DataQualityIssue(
                        path=str(read_result.path or ""),
                        symbol=symbol,
                        category=str(read_result.category or ""),
                        universe_key=universe_key,
                        issue_type=frame_blocker.split(":", 1)[0],
                        severity="error",
                        message=(
                            "Production research frame excluded: "
                            f"{frame_blocker}"
                        ),
                        resolver_strategy=str(
                            read_result.resolver_trace.get(
                                "resolution_strategy",
                                "",
                            )
                        ),
                        metadata={
                            "blocker": frame_blocker,
                            "evaluation_as_of": runtime_end_date,
                        },
                    )
                )
            tradability = _build_symbol_tradability(
                symbol,
                read_result,
                company_name=company_name_map.get(symbol, ""),
                sector=str(profile.get("sector", "") or profile.get("industry", "")),
                industry=str(profile.get("industry", "") or profile.get("sector", "")),
                trend_windows=trend_windows,
                volume_spike_threshold=volume_spike_threshold,
                breakout_distance_pct=breakout_distance_pct,
            )
            tradability_snapshot[symbol] = tradability
            market_state = dict(tradability.get("market_state", {}) or {})
            frame_summaries[symbol] = {
                "rows": int(market_state.get("rows", 0) or 0),
                "latest_close": float(market_state.get("latest_close", 0.0) or 0.0),
                "average_return": float(market_state.get("average_return", 0.0) or 0.0),
                "volatility": float(market_state.get("volatility", 0.0) or 0.0),
            }
            symbol_market_state[symbol] = dict(tradability_snapshot[symbol].get("market_state", {}) or {})
            industry_label = str(tradability_snapshot[symbol].get("industry") or tradability_snapshot[symbol].get("sector") or "").strip()
            if industry_label:
                industry_map[symbol] = industry_label
            data_quality_issues.extend(read_result.issues)
            if frame_blocker or _is_quarantined_read_result(read_result):
                quarantined_symbols.append(symbol)
            else:
                researchable_symbols.append(symbol)
        stage_metadata["researchable_count"] = len(researchable_symbols)
        stage_metadata["quarantined_count"] = len(quarantined_symbols)
        stage_metadata["issue_count"] = len(data_quality_issues)

    symbols = list(researchable_symbols)
    researchable_frames = _researchable_frame_subset(frames, symbols)
    production_factor_scorer = MinedFactorScorer()
    production_runtime_plan = (
        production_factor_scorer.build_production_runtime_plan(
            researchable_frames
        )
    )
    if production_runtime_plan.filter_applied:
        quant_contract_eligibility_blockers = {
            symbol: list(blockers)
            for symbol, blockers in (
                production_runtime_plan.symbol_blockers.items()
            )
        }
        for symbol, blockers in quant_contract_eligibility_blockers.items():
            read_result = read_results[symbol]
            data_quality_issues.append(
                DataQualityIssue(
                    path=str(read_result.path or ""),
                    symbol=symbol,
                    category=str(read_result.category or ""),
                    universe_key=universe_key,
                    issue_type="production_factor_runtime_ineligible",
                    severity="error",
                    message=(
                        "Production factor runtime input excluded: "
                        + ";".join(blockers)
                    ),
                    resolver_strategy=str(
                        read_result.resolver_trace.get(
                            "resolution_strategy",
                            "",
                        )
                    ),
                    metadata={
                        "blockers": list(blockers),
                        "evaluation_as_of": runtime_end_date,
                    },
                )
            )
            if symbol not in quarantined_symbols:
                quarantined_symbols.append(symbol)
        researchable_symbols = list(
            production_runtime_plan.eligible_symbols
        )
        symbols = list(researchable_symbols)
        researchable_frames = _researchable_frame_subset(frames, symbols)
    stage_metadata["researchable_count"] = len(researchable_symbols)
    stage_metadata["quarantined_count"] = len(quarantined_symbols)
    stage_metadata["issue_count"] = len(data_quality_issues)
    researchable_frame_summaries = {
        symbol: frame_summaries[symbol]
        for symbol in symbols
        if symbol in frame_summaries
    }
    (
        production_evaluation_context,
        production_evaluation_context_blockers,
    ) = _build_production_evaluation_context(
        market=settings.market,
        universe_key=universe_key,
        symbols=symbols,
        reader_snapshot=resolver_snapshot,
        scoped_data_snapshot=scoped_data_snapshot,
        read_results={symbol: read_results[symbol] for symbol in symbols},
    )

    with profile_stage(
        runtime_profiler,
        "dag_quant_context",
        {"researchable_count": len(symbols), "universe_key": universe_key},
    ) as stage_metadata:
        with profile_stage(
            runtime_profiler,
            "dag_cross_section_quant",
            {"researchable_count": len(symbols), "frame_count": len(frames)},
        ) as cross_section_metadata:
            cross_section_quant = _build_cross_section_quant(
                researchable_frames,
                frame_summaries=researchable_frame_summaries,
            )
            cross_section_metadata["breadth"] = float(cross_section_quant.get("breadth", 0.0))
            cross_section_metadata["average_return"] = float(
                cross_section_quant.get("average_return", 0.0)
            )
            cross_section_metadata["average_volatility"] = float(
                cross_section_quant.get("average_volatility", 0.0)
            )
        (
            completeness_payload,
            effective_latest_trade_date,
            effective_freshness_mode,
        ) = _resolve_effective_data_state(
            scoped_data_snapshot=scoped_data_snapshot,
            download_stage=download_stage,
        )
        pinned_macro_record, pinned_macro_manifest = load_macro_record(
            as_of=effective_latest_trade_date,
        )
        pinned_macro_readiness = assess_macro_readiness(
            macro_record=pinned_macro_record,
            manifest=pinned_macro_manifest,
            as_of=effective_latest_trade_date,
        )
        try:
            pinned_macro_controls = _validated_pinned_macro_controls(
                pinned_macro_manifest
            )
        except ValueError:
            pinned_macro_controls = {}
            pinned_macro_readiness.status = STATUS_BLOCK
            pinned_macro_readiness.blockers = list(
                dict.fromkeys(
                    [
                        *pinned_macro_readiness.blockers,
                        "macro_v15_controls_missing_or_invalid",
                    ]
                )
            )
        pinned_macro_blocked = pinned_macro_readiness.status == STATUS_BLOCK
        pinned_macro_identity = macro_generation_identity(pinned_macro_manifest)
        if pinned_macro_blocked:
            macro_overview = {
                "regime": "neutral",
                "macro_score": 0.0,
                "liquidity_score": 0.0,
                "volatility_percentile": 50.0,
                "policy_signal": "neutral",
            }
        else:
            macro_overview = {
                "regime": "neutral",
                "macro_score": float(pinned_macro_controls["macro_score"]),
                "liquidity_score": float(
                    pinned_macro_controls["liquidity_score"]
                ),
                "volatility_percentile": float(
                    pinned_macro_controls["volatility_percentile"]
                ),
                "policy_signal": str(
                    pinned_macro_controls["policy_signal"]
                ),
            }
        with profile_stage(
            runtime_profiler,
            "dag_market_snapshot",
            {
                "researchable_count": len(symbols),
                "latest_trade_date": effective_latest_trade_date,
                "universe_key": universe_key,
            },
        ) as market_snapshot_metadata:
            market_snapshot = _build_market_snapshot(
                market=settings.market,
                universe_key=universe_key,
                frames=researchable_frames,
                global_summary={"candidate_count": len(symbols)},
                latest_trade_date=effective_latest_trade_date,
                macro_overview=macro_overview,
                frame_summaries=researchable_frame_summaries,
            )
            market_snapshot.update(
                {
                    "macro_data_readiness_status": pinned_macro_readiness.status,
                    "canonical_macro_generation": dict(pinned_macro_identity),
                    "v15_controls_semantic_sha256": str(
                        pinned_macro_controls.get("semantic_sha256") or ""
                    ),
                    "decision_authorized": not pinned_macro_blocked,
                }
            )
            market_snapshot_metadata["snapshot_key_count"] = len(market_snapshot)

        with profile_stage(
            runtime_profiler,
            "dag_macro_verdict",
            {"market": settings.market, "universe_key": universe_key},
        ) as macro_metadata:
            if pinned_macro_blocked:
                macro_verdict = _blocked_macro_verdict(
                    blockers=list(pinned_macro_readiness.blockers),
                    generation_identity=pinned_macro_identity,
                )
            else:
                macro_verdict = macro_agent.run(
                    {"market_snapshot": market_snapshot}
                )
                macro_verdict.metadata = dict(macro_verdict.metadata or {})
                macro_verdict.metadata.update(
                    {
                        "decision_authorized": True,
                        "macro_data_readiness_status": (
                            pinned_macro_readiness.status
                        ),
                        "canonical_macro_generation": dict(
                            pinned_macro_identity
                        ),
                        "v15_controls_semantic_sha256": str(
                            pinned_macro_controls.get("semantic_sha256") or ""
                        ),
                    }
                )
            macro_metadata["macro_regime"] = str(
                macro_verdict.metadata.get("regime", "neutral")
            )
            macro_metadata["macro_score"] = float(macro_verdict.final_score)
            macro_metadata["macro_data_readiness_status"] = (
                pinned_macro_readiness.status
            )
            macro_metadata["canonical_macro_generation"] = dict(
                pinned_macro_identity
            )
        macro_overview["regime"] = str(macro_verdict.metadata.get("regime", "neutral"))
        market_snapshot.update(macro_overview)
        market_snapshot["macro_agent_score"] = float(macro_verdict.final_score)
        with profile_stage(
            runtime_profiler,
            "dag_quant_branch_result",
            {"researchable_count": len(symbols), "frame_count": len(frames)},
        ) as quant_branch_metadata:
            (
                quant_result,
                quant_validation_token,
            ) = _build_quant_branch_result_with_validation(
                frames=researchable_frames,
                frame_summaries=researchable_frame_summaries,
                evaluation_context=production_evaluation_context,
                evaluation_context_blockers=(
                    production_evaluation_context_blockers
                ),
                scorer=production_factor_scorer,
                production_runtime_plan=production_runtime_plan,
            )
            quant_branch_metadata["scored_symbol_count"] = len(quant_result.symbol_scores)
        with profile_stage(
            runtime_profiler,
            "dag_global_quant_verdict",
            {"researchable_count": len(symbols), "universe_key": universe_key},
        ) as global_quant_metadata:
            global_quant_verdict = _build_global_quant_verdict(
                cross_section_quant=cross_section_quant,
                symbol_count=len(symbols),
                quant_result=quant_result,
                validation_token=quant_validation_token,
            )
            global_quant_metadata["global_quant_score"] = float(global_quant_verdict.final_score)
            global_quant_metadata["global_quant_confidence"] = float(
                global_quant_verdict.final_confidence
            )
            global_quant_metadata["production_quant_evidence"] = bool(
                global_quant_verdict.metadata.get("production_quant_evidence", False)
            )
        stage_metadata["macro_regime"] = str(macro_verdict.metadata.get("regime", "neutral"))
        stage_metadata["breadth"] = float(cross_section_quant.get("breadth", 0.0))
    liquidity_scores = {
        symbol: float(
            max(
                min(
                    1.0,
                    max(
                        0.0,
                        float(frame_summaries.get(symbol, {}).get("rows", 0) or 0)
                        / 250.0,
                    ),
                ),
                tradability_snapshot.get(symbol, {}).get("liquidity_score", 0.0),
            )
        )
        for symbol, frame in frames.items()
    }
    illiquid_symbols = [symbol for symbol, score in liquidity_scores.items() if score < 0.10]
    sector_strengths: dict[str, float] = {}
    sector_members: dict[str, list[float]] = {}
    for symbol, info in tradability_snapshot.items():
        sector = str(info.get("industry") or info.get("sector") or "").strip()
        if not sector or sector == "unknown":
            continue
        sector_members.setdefault(sector, []).append(float(info.get("momentum_strength", 0.0)))
    if sector_members:
        sector_avgs = {
            sector: sum(values) / max(len(values), 1)
            for sector, values in sector_members.items()
        }
        ordered = sorted(sector_avgs.items(), key=lambda item: (-item[1], item[0]))
        total = max(len(ordered) - 1, 1)
        for rank, (sector, score) in enumerate(ordered):
            percentile = 1.0 if len(ordered) == 1 else 1.0 - (rank / total)
            sector_strengths[sector] = _clamp(0.55 * percentile + 0.45 * float(score), 0.0, 1.0)

    style_exposures: dict[str, Any] = {
        "style_bias": macro_verdict.metadata.get("style_bias", "balanced"),
        "default": 0.50,
    }
    for symbol, info in tradability_snapshot.items():
        sector = str(info.get("industry") or info.get("sector") or "unknown")
        sector_strength = float(sector_strengths.get(sector, 0.50))
        momentum_strength = float(info.get("momentum_strength", 0.0))
        style_exposures[symbol] = {
            "prior": _clamp(0.35 + 0.35 * sector_strength + 0.30 * momentum_strength, 0.15, 0.90),
            "sector": str(info.get("sector") or sector),
            "industry": sector,
            "momentum_strength": momentum_strength,
        }

    target_exposure = float(macro_verdict.metadata.get("target_gross_exposure", 0.5))
    max_single_weight = DAG_SINGLE_NAME_WEIGHT_CAP
    if (
        not pinned_macro_blocked
        and str(funnel_profile or "").strip().lower() == "momentum_leader"
    ):
        breadth = float(cross_section_quant.get("breadth", 0.0))
        weak_regime = str(macro_verdict.metadata.get("regime", "neutral")) in {"趋势下跌", "震荡高波"}
        if weak_regime or float(macro_verdict.final_score) < 0.0 or breadth < 0.48:
            target_exposure = min(target_exposure, 0.45) * 0.75
        elif str(macro_verdict.metadata.get("regime", "neutral")) == "趋势上涨" and breadth > 0.55:
            target_exposure = min(target_exposure * 1.08, 0.72)

    macro_agent_regime = str(macro_verdict.metadata.get("regime", "neutral"))
    effective_macro_regime = macro_agent_regime
    baseline_target_exposure = float(target_exposure)
    baseline_max_single_weight = float(max_single_weight)
    risk_budget: dict[str, Any] = {
        "target_exposure": target_exposure,
        "max_single_weight": max_single_weight,
        "sector_bucket_limit": int(sector_bucket_limit),
        "baseline_target_exposure": baseline_target_exposure,
        "baseline_max_single_weight": baseline_max_single_weight,
    }
    markov_target = str(
        getattr(config, "MARKOV_REGIME_EXECUTION_TARGET", "production") or "production"
    ).strip().lower()
    markov_enabled = bool(getattr(config, "MARKOV_REGIME_ENABLED", True)) and markov_target != "disabled"
    markov_payload: dict[str, Any] = {
        "enabled": False,
        "status": "disabled",
        "execution_mode": "disabled",
        "production_eligible": False,
        "baseline_target_exposure": baseline_target_exposure,
        "applied_target_exposure": baseline_target_exposure,
        "baseline_max_single_weight": baseline_max_single_weight,
        "applied_max_single_weight": baseline_max_single_weight,
    }
    if pinned_macro_blocked:
        markov_payload.update(
            {
                "status": "blocked_by_canonical_macro_readiness",
                "execution_mode": "not_run",
                "diagnostic_notes": [
                    "markov_not_run_canonical_macro_readiness_blocked",
                    *list(pinned_macro_readiness.blockers),
                ],
                "canonical_macro_generation": dict(pinned_macro_identity),
            }
        )
    elif markov_enabled:
        markov_reference_input = _resolve_markov_reference_input(
            market=settings.market,
            universe_key=universe_key,
            requested_symbols=list(all_symbols),
            current_frames=frames,
            current_tradability_snapshot=tradability_snapshot,
            current_cross_section_quant=cross_section_quant,
            shared_reader=shared_reader,
            as_of=effective_latest_trade_date,
            runtime_lookback_start_date=runtime_lookback_start_date,
            explicit_symbol_count=explicit_symbol_count,
            unsampled_symbol_count=unsampled_symbol_count,
            sampled=sampled,
            trend_windows=trend_windows,
            volume_spike_threshold=volume_spike_threshold,
            breakout_distance_pct=breakout_distance_pct,
            runtime_profiler=runtime_profiler,
        )
        markov_engine = MarkovRegimeEngine(
            history_path=str(getattr(config, "MARKOV_REGIME_HISTORY_PATH", "results/regime/markov_regime_history.jsonl")),
            enabled=True,
            execution_target=markov_target or "production",
            persist_enabled=bool(getattr(config, "MARKOV_REGIME_PERSIST_ENABLED", True)),
        )
        regime_signal = markov_engine.run(
            market=settings.market,
            universe_key=markov_reference_input.scope.source_universe_key,
            as_of=effective_latest_trade_date,
            frames=markov_reference_input.frames,
            tradability_snapshot=markov_reference_input.tradability_snapshot,
            cross_section_quant=markov_reference_input.cross_section_quant,
            macro_verdict=macro_verdict,
            market_snapshot=market_snapshot,
            scope=markov_reference_input.scope,
        )
        markov_payload = regime_signal.to_dict()
        markov_payload["execution_target"] = markov_engine.execution_target
        markov_payload["execution_mode"] = "production"
        markov_payload["enabled"] = True
        markov_payload["baseline_target_exposure"] = baseline_target_exposure
        markov_payload["baseline_max_single_weight"] = baseline_max_single_weight
        if regime_signal.production_eligible:
            markov_payload["status"] = "applied"
            effective_macro_regime = regime_signal.dominant_regime
            target_exposure = min(
                baseline_target_exposure,
                regime_signal.suggested_gross_exposure_cap,
            )
            max_single_weight = min(
                baseline_max_single_weight,
                regime_signal.suggested_max_single_weight,
            )
            applied_turnover_cap = regime_signal.turnover_cap
        else:
            markov_payload["status"] = (
                regime_signal.status or "not_applied_insufficient_market_scope"
            )
            target_exposure = baseline_target_exposure
            max_single_weight = baseline_max_single_weight
            applied_turnover_cap = None
        markov_payload["applied_target_exposure"] = target_exposure
        markov_payload["applied_gross_exposure_cap"] = target_exposure
        markov_payload["applied_max_single_weight"] = max_single_weight
        markov_payload["applied_turnover_cap"] = applied_turnover_cap
        risk_budget.update(
            {
                "target_exposure": target_exposure,
                "max_single_weight": max_single_weight,
                "markov_enabled": True,
                "markov_regime_enabled": True,
                "markov_execution_mode": "production",
                "markov_production_eligible": bool(regime_signal.production_eligible),
                "markov_status": str(markov_payload.get("status") or ""),
                "markov_regime_scope": regime_signal.regime_scope,
                "markov_scope_key": regime_signal.scope_key,
                "markov_source_universe_key": regime_signal.source_universe_key,
                "markov_source_symbol_count": regime_signal.source_symbol_count,
                "markov_requested_symbol_count": regime_signal.requested_symbol_count,
                "markov_dominant_regime": regime_signal.dominant_regime,
                "markov_probabilities": dict(regime_signal.probabilities),
                "markov_confidence": regime_signal.confidence,
                "markov_transition_risk": regime_signal.transition_risk,
                "markov_baseline_target_exposure": baseline_target_exposure,
                "markov_applied_target_exposure": target_exposure,
                "markov_applied_gross_exposure_cap": target_exposure,
                "markov_baseline_max_single_weight": baseline_max_single_weight,
                "markov_applied_max_single_weight": max_single_weight,
                "markov_turnover_cap": applied_turnover_cap,
                "markov_history_record_count": regime_signal.history_record_count,
                "markov_transition_matrix_source": regime_signal.transition_matrix_source,
                "markov_diagnostic_notes": list(regime_signal.diagnostic_notes),
            }
        )
        if applied_turnover_cap is not None:
            risk_budget["turnover_cap"] = applied_turnover_cap


    provider_health = provider_health_detector(
        agent_model=branch_model_resolution.primary_model,
        master_model=master_model_resolution.primary_model,
    )
    global_context = GlobalContext(
        market=settings.market,
        universe_key=universe_key,
        rebalance_date=effective_latest_trade_date,
        latest_trade_date=effective_latest_trade_date,
        universe_symbols=list(all_symbols),
        universe_hash="",
        industry_map=industry_map,
        liquidity_filter={
            "candidate_count": len(all_symbols),
            "researchable_count": len(symbols),
            "quarantined_count": len(quarantined_symbols),
            "category_count": len(selected_categories),
            "suspended": list(quarantined_symbols),
            "illiquid": list(illiquid_symbols),
            "liquidity_scores": liquidity_scores,
            "sector_bucket_limit": int(sector_bucket_limit),
        },
        macro_regime=effective_macro_regime,
        cross_section_quant={**cross_section_quant, "macro_score": float(macro_verdict.final_score)},
        style_exposures=style_exposures,
        correlation_matrix={},
        risk_budget=risk_budget,
        data_quality_issues=data_quality_issues,
        data_quality_diagnostics=build_data_quality_diagnostics(
            total_symbols=all_symbols,
            researchable_symbols=researchable_symbols,
            shortlistable_symbols=[],
            final_selected_symbols=[],
            quarantined_symbols=quarantined_symbols,
            issues=data_quality_issues,
        ),
        model_capability_map=provider_health,
        symbol_name_map=dict(company_name_map),
        data_quality_quarantine=list(quarantined_symbols),
        freshness_mode=effective_freshness_mode,
        effective_target_trade_date=str(
            completeness_payload.get("effective_target_trade_date")
            or effective_latest_trade_date
        ),
        regime_params={"markov": markov_payload},
        macro_data=(
            dict(pinned_macro_controls) if not pinned_macro_blocked else {}
        ),
        universe_tiers={
            "total": list(all_symbols),
            "researchable": list(researchable_symbols),
            "shortlistable": [],
            "final_selected": [],
        },
        metadata={
            "resolver": resolver_snapshot,
            "resolver_directory_priority": list((resolver_snapshot or {}).get("directory_priority", [])),
            "physical_directories_used_for_full_a": list((resolver_snapshot or {}).get("physical_directories_used_for_full_a", [])),
            "data_quality_issue_count": len(data_quality_issues),
            "candidate_count": len(all_symbols),
            "researchable_count": len(symbols),
            "quarantined_count": len(quarantined_symbols),
            "quarantined_symbols": list(quarantined_symbols[:32]),
            "global_quant_verdict": global_quant_verdict.to_dict(),
            "production_evaluation_context": (
                production_evaluation_context.to_metadata()
                if production_evaluation_context is not None
                else {}
            ),
            "production_evaluation_context_blockers": list(
                production_evaluation_context_blockers
            ),
            "quant_frame_validation_blockers": dict(
                quant_frame_validation_blockers
            ),
            "quant_contract_eligibility_blockers": dict(
                quant_contract_eligibility_blockers
            ),
            "provider_health": {},
            "data_snapshot": dict(scoped_data_snapshot),
            "symbol_market_state": symbol_market_state,
            "markov_regime": markov_payload,
            "markov_regime_diagnostic_notes": list(markov_payload.get("diagnostic_notes", []) or []),
            "macro_agent_regime": macro_agent_regime,
            "canonical_macro_generation": dict(pinned_macro_identity),
            "canonical_macro_readiness": pinned_macro_readiness.to_dict(),
            "decision_authorized": not pinned_macro_blocked,
            "selection_profile": {
                "funnel_profile": str(funnel_profile or "classic").strip().lower() or "classic",
                "trend_windows": list(trend_windows),
                "volume_spike_threshold": float(volume_spike_threshold),
                "breakout_distance_pct": float(breakout_distance_pct),
                "max_candidates": int(max_candidates),
                "sector_bucket_limit": int(sector_bucket_limit),
            },
        },
    )
    global_context.model_capability_map = provider_health
    global_context.metadata["provider_health"] = provider_health
    global_context.metadata["macro_v2_observer"] = (
        _macro_v2_observer_metadata(
            market=settings.market,
            as_of=effective_latest_trade_date,
        )
    )
    if symbols:
        import hashlib

        global_context.universe_hash = hashlib.sha256(",".join(sorted(symbols)).encode("utf-8")).hexdigest()[:16]

    role_metadata = {
        "resolver": resolver_snapshot,
        "data_quality_issue_count": len(data_quality_issues),
        "agent_layer_enabled": bool(enable_agent_layer),
        "provider_health": provider_health,
        "ordered_review_models": {
            "branch": list(branch_candidate_models),
            "master": list(master_candidate_models),
        },
    }
    for key, value in {
        **dict(branch_model_resolution.metadata or {}),
        **dict(master_model_resolution.metadata or {}),
    }.items():
        role_metadata.setdefault(str(key), value)
    role_metadata.setdefault(
        "review_layer_mode",
        "local_llm" if enable_agent_layer else "disabled",
    )

    model_roles = build_model_role_metadata(
        branch_model=branch_model_resolution.primary_model,
        master_model=master_model_resolution.primary_model,
        agent_fallback_model=branch_model_resolution.fallback_model,
        master_fallback_model=master_model_resolution.fallback_model,
        resolved_branch_model=branch_model_resolution.resolved_model,
        resolved_master_model=master_model_resolution.resolved_model,
        master_reasoning_effort=master_reasoning_effort,
        branch_provider=_provider_label(branch_model_resolution),
        master_provider=_provider_label(master_model_resolution),
        branch_timeout=agent_timeout,
        master_timeout=master_timeout,
        agent_layer_enabled=bool(enable_agent_layer),
        branch_fallback_used=bool(branch_model_resolution.fallback_used),
        master_fallback_used=bool(master_model_resolution.fallback_used),
        branch_fallback_reason=str(branch_model_resolution.fallback_reason),
        master_fallback_reason=str(master_model_resolution.fallback_reason),
        universe_key=universe_key,
        universe_size=len(symbols),
        universe_hash=global_context.universe_hash,
        metadata=role_metadata,
    )

    funnel = funnel_cls(
        FunnelConfig(
            max_candidates=int(max_candidates or getattr(config, "FUNNEL_MAX_CANDIDATES", 500) or 500),
            profile=str(funnel_profile or "classic").strip().lower() or "classic",
            trend_windows=tuple(int(item) for item in trend_windows if int(item) > 0) or tuple(getattr(config, "FUNNEL_TREND_WINDOWS", (20, 60, 120))),
            volume_spike_threshold=float(volume_spike_threshold),
            breakout_distance_pct=float(breakout_distance_pct),
            sector_bucket_limit=int(sector_bucket_limit if str(funnel_profile or "").strip().lower() == "momentum_leader" else 0),
        )
    )
    with profile_stage(
        runtime_profiler,
        "dag_funnel",
        {"researchable_count": len(researchable_symbols), "max_candidates": int(max_candidates)},
    ) as stage_metadata:
        funnel_output = funnel.run(
            quant_result=quant_result,
            global_context=global_context,
        )
        stage_metadata["candidate_count"] = len(getattr(funnel_output, "candidates", []) or [])
        stage_metadata["excluded_count"] = len(getattr(funnel_output, "excluded_symbols", {}) or {})
    candidate_symbols = [symbol for symbol in funnel_output.candidates if symbol in researchable_symbols]
    holding_review_funnel_override = _holding_single_review_active(
        recall_context=recall_context,
        symbols=researchable_symbols,
    )
    if holding_review_funnel_override:
        candidate_symbols = list(researchable_symbols)
    with profile_stage(
        runtime_profiler,
        "dag_branch_readiness",
        {"candidate_count": len(candidate_symbols), "universe_key": universe_key},
    ) as stage_metadata:
        branch_governance_report = assess_branch_data_readiness(
            frames=frames,
            read_results=read_results,
            candidate_symbols=candidate_symbols,
            market=settings.market,
            category=universe_key,
            as_of=effective_latest_trade_date,
            pinned_macro_record=pinned_macro_record,
            pinned_macro_manifest=pinned_macro_manifest,
        )
        branch_governance_artifacts = write_branch_readiness_report(branch_governance_report)
        stage_metadata["blocked_symbol_count"] = len(branch_governance_report.blocked_symbols)
        stage_metadata["quantifiable_universe_count"] = len(branch_governance_report.quantifiable_universe)
        stage_metadata["investable_universe_count"] = len(branch_governance_report.investable_universe)
    branch_data_readiness = branch_governance_report.to_dict(include_branch_data=False)
    branch_data_payload = dict(branch_governance_report.branch_data)
    macro_ready = branch_governance_report.readiness.get("macro")
    macro_blocked = pinned_macro_blocked or bool(
        macro_ready and macro_ready.status == STATUS_BLOCK
    )
    if macro_blocked:
        macro_verdict.metadata = dict(macro_verdict.metadata or {})
        macro_verdict.metadata["decision_authorized"] = False
        market_snapshot["decision_authorized"] = False
    blocked_symbols = set(branch_governance_report.blocked_symbols)
    holding_review_readiness_override = _holding_single_review_active(
        recall_context=recall_context,
        symbols=candidate_symbols,
    )
    holding_review_funnel_override_applied = (
        holding_review_funnel_override and not macro_blocked
    )
    holding_review_readiness_override_applied = (
        holding_review_readiness_override and not macro_blocked
    )
    for symbol in list(blocked_symbols):
        if symbol in candidate_symbols:
            funnel_output.excluded_symbols.setdefault(symbol, "branch_data_readiness_block")
    if macro_blocked:
        for symbol in candidate_symbols:
            funnel_output.excluded_symbols.setdefault(symbol, "macro_data_readiness_block")
        candidate_symbols = []
    else:
        if not holding_review_readiness_override:
            candidate_symbols = [symbol for symbol in candidate_symbols if symbol not in blocked_symbols]
    funnel_output.candidates = list(candidate_symbols)
    funnel_output.candidate_scores = {
        symbol: score
        for symbol, score in dict(funnel_output.candidate_scores).items()
        if symbol in set(candidate_symbols)
    }
    funnel_output.funnel_metadata = dict(funnel_output.funnel_metadata or {})
    funnel_output.funnel_metadata.update(
        {
            "branch_data_governance_status": {
                branch: readiness.status
                for branch, readiness in branch_governance_report.readiness.items()
            },
            "branch_data_blocked_count": len(branch_governance_report.blocked_symbols),
            "macro_data_readiness_block": macro_blocked,
            "holding_review_funnel_override": (
                holding_review_funnel_override_applied
            ),
            "holding_review_funnel_override_requested": (
                holding_review_funnel_override
            ),
            "holding_review_branch_readiness_override": (
                holding_review_readiness_override_applied
            ),
            "holding_review_branch_readiness_override_requested": (
                holding_review_readiness_override
            ),
        }
    )
    if not macro_blocked and branch_data_payload.get("macro_data"):
        market_snapshot.update(dict(branch_data_payload.get("macro_data") or {}))
        global_context.macro_data.update(dict(branch_data_payload.get("macro_data") or {}))
    global_context.universe_tiers = {
        "total": list(all_symbols),
        "researchable": list(researchable_symbols),
        "shortlistable": list(candidate_symbols),
        "final_selected": [],
    }
    global_context.data_quality_diagnostics = build_data_quality_diagnostics(
        total_symbols=all_symbols,
        researchable_symbols=researchable_symbols,
        shortlistable_symbols=candidate_symbols,
        final_selected_symbols=[],
        quarantined_symbols=quarantined_symbols,
        issues=data_quality_issues,
    )
    global_context.metadata["candidate_count"] = len(candidate_symbols)
    global_context.metadata["shortlistable_count"] = len(candidate_symbols)
    global_context.metadata["branch_data_readiness"] = branch_data_readiness
    global_context.metadata["branch_readiness_artifacts"] = branch_governance_artifacts
    global_context.metadata["branch_fusion_blocked"] = macro_blocked
    global_context.metadata["decision_authorized"] = not macro_blocked
    global_context.metadata["blocked_branch_symbols"] = list(branch_governance_report.blocked_symbols[:128])
    global_context.metadata["holding_review_funnel_override"] = (
        holding_review_funnel_override_applied
    )
    global_context.metadata["holding_review_funnel_override_requested"] = (
        holding_review_funnel_override
    )
    global_context.metadata["holding_review_branch_readiness_override"] = (
        holding_review_readiness_override_applied
    )
    global_context.metadata[
        "holding_review_branch_readiness_override_requested"
    ] = holding_review_readiness_override
    global_context.metadata["quantifiable_universe_count"] = len(branch_governance_report.quantifiable_universe)
    global_context.metadata["investable_universe_count"] = len(branch_governance_report.investable_universe)
    candidate_sector_counts: dict[str, int] = {}
    for symbol in candidate_symbols:
        sector = str(industry_map.get(symbol) or tradability_snapshot.get(symbol, {}).get("industry") or tradability_snapshot.get(symbol, {}).get("sector") or "").strip()
        if not sector or sector == "unknown":
            continue
        candidate_sector_counts[sector] = candidate_sector_counts.get(sector, 0) + 1
    global_context.metadata["candidate_sector_counts"] = candidate_sector_counts

    return MarketContextState(
        all_symbols=all_symbols,
        read_results=read_results,
        frames=frames,
        tradability_snapshot=tradability_snapshot,
        data_quality_issues=data_quality_issues,
        quarantined_symbols=quarantined_symbols,
        researchable_symbols=researchable_symbols,
        candidate_symbols=candidate_symbols,
        provider_health=provider_health,
        market_snapshot=market_snapshot,
        macro_verdict=macro_verdict,
        global_quant_verdict=global_quant_verdict,
        quant_result=quant_result,
        global_context=global_context,
        model_roles=model_roles,
        funnel_output=funnel_output,
        resolver_snapshot=resolver_snapshot,
        branch_data_readiness=branch_data_readiness,
        branch_data_payload=branch_data_payload,
    )
