"""Deterministic source adapters and computations for the v17 shadow runtime.

External lifecycle payloads may identify a run, carry a deep-research answer,
or propose target weights.  They are never allowed to supply an authoritative
rank, timing decision, risk overlay, cost, turnover, or portfolio result.  This
module rebuilds those values from content-addressed local source objects and
the pure v17 policy functions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from decimal import Decimal
import hashlib
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_investor.factors.price_volume import compute_price_volume_factor
from quant_investor.market.pit_universe import (
    PITUniverseRecord,
    REASON_CONFLICTING_STATUS_ROWS,
    evaluate_listing_status,
    records_by_symbol,
)

from .contracts import (
    Availability,
    V17ContractError,
    parse_utc_timestamp,
    require_authority_false,
    require_bool,
    require_exact_keys,
    require_identifier,
    require_nonempty_string,
    require_number,
    require_ratio,
    require_symbol,
)
from .deep_research import (
    COVERAGE_SECTIONS,
    LAYER_NAMES,
    SEVERE_RED_FLAGS,
    SIGNAL_WEIGHTS,
    DeepResearchEvaluation,
    evaluate_deep_research,
)
from .forward_calibration import (
    HORIZONS,
    assess_fundamental_eligibility,
    calibrate_forward_returns,
)
from .fundamental_scoring import ALL_METRICS, score_fundamental_universe
from .holdings import validate_holdings_snapshot
from .optimizer import FeasiblePortfolio, ProposedTrade, optimize_lexicographic
from .permissions import determine_trade_permission
from .pretrade import (
    evaluate_pretrade,
    validate_execution_cost_policy,
    validate_pretrade_result,
)
from .quant_timing import (
    FACTOR_RESOURCE_SHA256,
    assert_factor_source_binding,
    calibrate_timing_probabilities,
    compute_latest_scores,
    decide_timing,
)
from .regime_overlay import (
    build_available_overlay_input,
    build_disabled_overlay_input,
    build_unavailable_overlay_input,
    compute_regime_portfolio_overlay,
)
from .risk_policy import validate_portfolio_risk_policy_snapshot
from .semantic import (
    canonical_json_bytes,
    require_sha256,
    seal_semantic,
    validate_semantic_seal,
)
from .source_bindings import PORTFOLIO_REQUIRED_ROLES, SourceBindingBundle
from .source_maintain import MAX_SOURCE_OBJECT_BYTES
from .storage import file_sha256, read_json

DETERMINISTIC_RESULT_VERSION = "myquant.v17.deterministic-result.v1"
DEEP_REQUEST_VERSION = "myquant.v17.deep-research-request.v1"
DEEP_EVALUATION_VERSION = "myquant.v17.deep-research-evaluation.v1"
PORTFOLIO_COMPUTATION_VERSION = "myquant.v17.portfolio-computation.v1"
RANK_OUTPUT_VERSION = "myquant.v17.shadow-rank-output.v1"
PORTFOLIO_OUTPUT_VERSION = "myquant.v17.shadow-portfolio-output.v1"

PIT_MEMBERSHIP_SOURCE_VERSION = "myquant.v17.pit-membership-source.v1"
FUNDAMENTAL_SNAPSHOT_SOURCE_VERSION = "myquant.v17.fundamental-snapshot-source.v1"
FUNDAMENTAL_HISTORY_SOURCE_VERSION = "myquant.v17.fundamental-history-source.v1"
FORWARD_CALIBRATION_SOURCE_VERSION = "myquant.v17.forward-calibration-source.v1"
QUANT_CALIBRATION_SOURCE_VERSION = "myquant.v17.quant-calibration-source.v1"
MARKET_SNAPSHOT_SOURCE_VERSION = "myquant.v17.market-snapshot-source.v1"
MARKET_POINTER_SOURCE_VERSION = "myquant.v17.market-pointer-source.v1"
OPEN_DAY_CALENDAR_SOURCE_VERSION = "myquant.v17.cn-open-day-calendar-source.v1"
BENCHMARK_TOTAL_RETURN_SOURCE_VERSION = "myquant.v17.benchmark-total-return-source.v1"
DIVIDEND_TOTAL_RETURN_SOURCE_VERSION = "myquant.v17.dividend-total-return-source.v1"
DELISTING_CASH_SOURCE_VERSION = "myquant.v17.delisting-cash-source.v1"
DEEP_EVIDENCE_SOURCE_VERSION = "myquant.v17.deep-evidence-source.v1"
TRADABILITY_SOURCE_VERSION = "myquant.v17.tradability-evidence-source.v1"
RISK_MODEL_INPUT_SOURCE_VERSION = "myquant.v17.risk-model-input-source.v1"
CLUSTER_MAPPING_SOURCE_VERSION = "myquant.v17.cluster-mapping-source.v1"
RANK_PROJECTION_RECEIPT_VERSION = "myquant.v17.rank-projection-verification-receipt.v1"
CALIBRATION_INPUT_MANIFEST_VERSION = "myquant.v17.calibration-input-manifest.v1"
CALIBRATION_RECEIPT_VERSION = "myquant.v17.calibration-verification-receipt.v1"
HOLDINGS_POINTER_VERSION = "myquant.v17.holdings-pointer.v1"
PORTFOLIO_PROJECTION_RECEIPT_VERSION = "myquant.v17.portfolio-projection-verification-receipt.v1"
FUNDAMENTAL_CALIBRATION_RAW_SOURCE_VERSION = "myquant.v17.fundamental-calibration-raw-predictors.v1"
QUANT_CALIBRATION_RAW_SOURCE_VERSION = "myquant.v17.quant-calibration-raw-bars.v1"
CALIBRATION_PIT_HISTORY_SOURCE_VERSION = "myquant.v17.calibration-pit-history.v1"
REGIME_MAPPING_SOURCE_VERSION = "myquant.v17.regime-mapping-source.v1"
REGIME_INPUT_SOURCE_VERSION = "myquant.v17.regime-input-source.v1"

_SOURCE_ENVELOPE_BASE_KEYS = frozenset(
    {"version", "market", "cutoff", "authority", "semantic_sha256"}
)
_PIT_ROW_KEYS = frozenset(
    {
        "symbol",
        "industry",
        "in_universe",
        "research_eligible",
        "membership_conflict",
        "membership_is_pit",
        "universe_id",
        "availability",
    }
)
_FUNDAMENTAL_ROW_KEYS = frozenset(
    {
        "symbol",
        "availability",
        "flow_basis",
        "balance_sheet_basis",
        "capex_sign_convention",
        "net_profit_ttm",
        "market_cap",
        "cfo_ttm",
        "capex_ttm",
        "fin_roe",
        "fin_roa",
        "fin_ocf_to_profit",
        "fin_fcf_to_profit",
        "fin_net_profit_yoy",
        "forecast_revision",
        "fin_debt_to_assets",
    }
)
_HISTORY_ROW_KEYS = frozenset(
    {"symbol", "trade_date", "availability", "is_open_day", "metric", "value"}
)
_FORWARD_ROW_KEYS = frozenset(
    {
        "symbol",
        "industry",
        "fundamental_score",
        "score_decile",
        "horizon",
        "cross_section_date",
        "availability",
        "predictor_available_at",
        "label_available_at",
        "age_open_days",
        "realized_open_days",
        "is_pit_month_end",
        "is_mature",
        "stock_start_trade_date",
        "stock_end_trade_date",
        "benchmark_start_trade_date",
        "benchmark_end_trade_date",
        "stock_total_return",
        "benchmark_total_return",
        "benchmark_symbol",
        "stock_return_includes_dividends",
        "benchmark_return_is_pre_tax_total_return",
        "delisted",
        "official_terminal_cash_settlement",
    }
)
_QUANT_CALIBRATION_ROW_KEYS = frozenset(
    {
        "horizon",
        "symbol",
        "cross_section_date",
        "availability",
        "predictor_available_at",
        "label_available_at",
        "age_open_days",
        "target_start_trade_date",
        "target_end_trade_date",
        "realized_open_days",
        "is_mature",
        "is_pit",
        "target_definition",
        "pv_blend_volstab19x2_mom90_amihud5_w80",
        "pv_short_reversal_25d",
        "pv_downside_volatility_15d",
        "stock_total_return",
        "benchmark_total_return",
        "excess_return",
    }
)
_MARKET_SYMBOL_KEYS = frozenset(
    {
        "symbol",
        "bars",
    }
)
_BAR_KEYS = frozenset({"trade_date", "availability", "is_open_day", "close", "volume", "amount"})
_CALENDAR_SESSION_KEYS = frozenset({"trade_date", "availability", "is_open_day"})
_TRADABILITY_ROW_KEYS = frozenset({"symbol", "effective_trade_date", "availability", "status"})
_TRADABILITY_STATUSES = frozenset(
    {"TRADABLE", "SUSPENDED", "LIMIT_BLOCKED", "DELISTED", "OTHER_BLOCKED"}
)
_CLUSTER_ROW_KEYS = frozenset({"symbol", "cluster", "availability"})
_FUNDAMENTAL_RAW_PREDICTOR_ROW_KEYS = (
    _PIT_ROW_KEYS | _FUNDAMENTAL_ROW_KEYS | frozenset({"trade_date", "is_open_day"})
)
_QUANT_RAW_CROSS_SECTION_KEYS = frozenset({"cross_section_date", "symbols"})
_QUANT_RAW_SYMBOL_KEYS = frozenset({"symbol", "bars"})
_CALIBRATION_PIT_HISTORY_ROW_KEYS = frozenset(
    {"cross_section_date", "eligible_symbols", "dispositions"}
)
_BENCHMARK_TOTAL_RETURN_ROW_KEYS = frozenset(
    {
        "start_trade_date",
        "end_trade_date",
        "start_total_return_index",
        "end_total_return_index",
        "availability",
    }
)
_DIVIDEND_TOTAL_RETURN_ROW_KEYS = frozenset(
    {
        "symbol",
        "start_trade_date",
        "end_trade_date",
        "start_total_return_index",
        "end_total_return_index",
        "availability",
    }
)
_DELISTING_ROW_KEYS = frozenset(
    {
        "symbol",
        "end_trade_date",
        "delisted",
        "official_terminal_cash_settlement",
        "availability",
    }
)
_DEEP_EVIDENCE_ENTRY_KEYS = frozenset(
    {
        "evidence_id",
        "symbol",
        "kind",
        "available_at",
        "locator",
        "content",
        "located_content_sha256",
    }
)
_REGIME_MAPPING_CELL_KEYS = frozenset({"gross_cap", "cash_floor"})

_DEEP_EVIDENCE_KIND_CLAIMS: Mapping[str, Mapping[str, frozenset[str]]] = {
    "financial_filings": {
        "layers": frozenset({"raw_facts", "derived_metrics", "research_inferences", "risk_alerts"}),
        "coverage": frozenset(
            {
                "financial_reports_and_three_statement_reconciliation",
                "normalization_and_reversible_adjustments",
                "segments",
                "counterevidence",
                "falsification_conditions",
                "continuous_monitoring_items",
            }
        ),
        "signals": frozenset({"financial", "business_model"}),
        "red_flags": frozenset(
            {
                "audit_or_going_concern",
                "restatement_or_three_statement_failure",
                "fraud_or_material_penalty",
                "liquidity_or_refinancing_break",
                "customer_or_supplier_concentration_break",
                "listing_or_delisting_risk",
            }
        ),
    },
    "governance_ownership": {
        "layers": frozenset(
            {"raw_facts", "research_inferences", "investment_judgments", "risk_alerts"}
        ),
        "coverage": frozenset(
            {
                "management_and_governance",
                "ownership",
                "catalysts",
                "counterevidence",
                "falsification_conditions",
                "continuous_monitoring_items",
            }
        ),
        "signals": frozenset({"management", "competitiveness"}),
        "red_flags": frozenset(
            {
                "fraud_or_material_penalty",
                "controller_appropriation_or_pledge_crisis",
                "material_related_party_or_governance_conflict",
                "liquidity_or_refinancing_break",
                "core_thesis_falsified",
            }
        ),
    },
    "industry_competition": {
        "layers": frozenset(
            {
                "raw_facts",
                "derived_metrics",
                "research_inferences",
                "investment_judgments",
                "risk_alerts",
            }
        ),
        "coverage": frozenset(
            {
                "industry_and_competition",
                "products_and_technology",
                "catalysts",
                "counterevidence",
                "falsification_conditions",
                "continuous_monitoring_items",
            }
        ),
        "signals": frozenset({"business_model", "industry", "competitiveness"}),
        "red_flags": frozenset(
            {
                "customer_or_supplier_concentration_break",
                "product_or_technology_obsolescence",
                "core_thesis_falsified",
            }
        ),
    },
    "valuation_scenarios": {
        "layers": frozenset(
            {"derived_metrics", "research_inferences", "investment_judgments", "risk_alerts"}
        ),
        "coverage": frozenset(
            {
                "dcf",
                "reverse_dcf",
                "comparable_companies",
                "sotp_if_applicable",
                "bull_base_bear_scenarios",
                "catalysts",
                "counterevidence",
                "falsification_conditions",
                "continuous_monitoring_items",
            }
        ),
        "signals": frozenset({"valuation", "financial"}),
        "red_flags": frozenset(
            {"liquidity_or_refinancing_break", "listing_or_delisting_risk", "core_thesis_falsified"}
        ),
    },
}


class V17PipelineError(V17ContractError):
    """A bound source or deterministic v17 computation is invalid."""


def _strict_sequence(value: Any, *, label: str, allow_empty: bool = False) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise V17PipelineError(f"{label} must be an array")
    items = list(value)
    if not allow_empty and not items:
        raise V17PipelineError(f"{label} must be nonempty")
    return items


def _validate_source_envelope(
    payload: Mapping[str, Any],
    *,
    version: str,
    cutoff: str,
    extra_keys: frozenset[str],
    label: str,
) -> dict[str, Any]:
    sealed = validate_semantic_seal(payload)
    require_exact_keys(
        sealed,
        _SOURCE_ENVELOPE_BASE_KEYS | extra_keys,
        label=label,
    )
    if sealed.get("version") != version:
        raise V17PipelineError(f"{label} version mismatch")
    if sealed.get("market") != "CN" or sealed.get("cutoff") != cutoff:
        raise V17PipelineError(f"{label} market/cutoff binding mismatch")
    parse_utc_timestamp(sealed.get("cutoff"), label=f"{label}.cutoff")
    require_authority_false(sealed.get("authority"))
    return sealed


def _read_bound_source(
    bundle: SourceBindingBundle,
    role: str,
) -> dict[str, Any]:
    item = bundle.by_role[role]
    if item["availability"] != Availability.AVAILABLE.value:
        raise V17PipelineError(f"source unavailable: {role}")
    path = bundle.object_path(role)
    expected = require_sha256(item["byte_sha256"], label=f"{role} byte SHA-256")
    before = file_sha256(path)
    if before != expected:
        raise V17PipelineError(f"source byte SHA mismatch: {role}")
    payload = read_json(path, max_bytes=MAX_SOURCE_OBJECT_BYTES)
    if file_sha256(path) != before:
        raise V17PipelineError(f"source changed during read: {role}")
    return validate_semantic_seal(payload)


def _read_bound_raw_json(
    bundle: SourceBindingBundle,
    role: str,
) -> dict[str, Any]:
    item = bundle.by_role[role]
    if item["availability"] != Availability.AVAILABLE.value:
        raise V17PipelineError(f"canonical lineage unavailable: {role}")
    path = bundle.object_path(role)
    expected = require_sha256(item["byte_sha256"], label=f"{role} byte SHA-256")
    before = file_sha256(path)
    if before != expected:
        raise V17PipelineError(f"canonical lineage byte SHA mismatch: {role}")
    payload = read_json(path, max_bytes=MAX_SOURCE_OBJECT_BYTES)
    if file_sha256(path) != before:
        raise V17PipelineError(f"canonical lineage changed during read: {role}")
    if not isinstance(payload, Mapping):
        raise V17PipelineError(f"canonical lineage JSON must be an object: {role}")
    return dict(payload)


def _raw_json_semantic_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(payload))).hexdigest()


def _bound_source_semantic_sha(bundle: SourceBindingBundle, role: str) -> str:
    return str(_read_bound_source(bundle, role)["semantic_sha256"])


def _validate_rank_projection_receipt(
    bundle: SourceBindingBundle,
    cutoff: str,
) -> None:
    receipt = _validate_source_envelope(
        _read_bound_source(bundle, "rank_projection_verification_receipt"),
        version=RANK_PROJECTION_RECEIPT_VERSION,
        cutoff=cutoff,
        extra_keys=frozenset(
            {
                "verifier_id",
                "verification_status",
                "source_byte_sha256s",
                "source_semantic_sha256s",
                "canonical_bar_replay_passed",
                "pit_projection_replay_passed",
                "fundamental_projection_replay_passed",
            }
        ),
        label="Rank projection verification receipt",
    )
    byte_roles = frozenset(
        {
            "market_pointer",
            "market_snapshot_manifest",
            "market_snapshot",
            "cn_open_day_calendar",
            "pit_membership_generation_manifest",
            "pit_membership_canonical",
            "pit_membership",
            "fundamental_generation_pointer",
            "fundamental_generation_manifest",
            "fundamental_snapshot",
            "fundamental_metric_history",
        }
    )
    semantic_roles = frozenset(
        {
            "market_snapshot",
            "cn_open_day_calendar",
            "pit_membership",
            "fundamental_snapshot",
            "fundamental_metric_history",
        }
    )
    if (
        receipt["verifier_id"] != "codex_delegated_reviewer"
        or receipt["verification_status"] != "PASSED"
    ):
        raise V17PipelineError("rank projection verification authority missing")
    if (
        not isinstance(receipt["source_byte_sha256s"], Mapping)
        or set(receipt["source_byte_sha256s"]) != byte_roles
    ):
        raise V17PipelineError("rank projection byte binding set mismatch")
    if (
        not isinstance(receipt["source_semantic_sha256s"], Mapping)
        or set(receipt["source_semantic_sha256s"]) != semantic_roles
    ):
        raise V17PipelineError("rank projection semantic binding set mismatch")
    if any(
        receipt["source_byte_sha256s"][role] != _source_sha(bundle, role) for role in byte_roles
    ) or any(
        receipt["source_semantic_sha256s"][role] != _bound_source_semantic_sha(bundle, role)
        for role in semantic_roles
    ):
        raise V17PipelineError("rank projection verification receipt drift")
    for field in (
        "canonical_bar_replay_passed",
        "pit_projection_replay_passed",
        "fundamental_projection_replay_passed",
    ):
        if receipt[field] is not True:
            raise V17PipelineError(f"rank projection replay failed: {field}")


def _calibration_generator_sha256(kind: str) -> str:
    module_names = (
        ("fundamental_scoring.py", "forward_calibration.py")
        if kind == "FUNDAMENTAL"
        else ("quant_timing.py",)
    )
    manifest = {name: file_sha256(Path(__file__).with_name(name)) for name in module_names}
    if kind == "QUANT":
        manifest["price_volume.py"] = assert_factor_source_binding()
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def _calibration_observation_keys(
    rows: Sequence[Mapping[str, Any]],
) -> list[list[Any]]:
    return sorted(
        [[str(row["symbol"]), str(row["cross_section_date"]), int(row["horizon"])] for row in rows]
    )


def _load_calibration_pit_history(
    bundle: SourceBindingBundle,
    cutoff: str,
) -> dict[pd.Timestamp, tuple[str, ...]]:
    lineage = _load_pit_lineage(bundle)
    source = _validate_source_envelope(
        _read_bound_source(bundle, "calibration_pit_membership_history"),
        version=CALIBRATION_PIT_HISTORY_SOURCE_VERSION,
        cutoff=cutoff,
        extra_keys=frozenset(
            {
                "canonical_generation_id",
                "canonical_generation_manifest_sha256",
                "canonical_membership_sha256",
                "rows",
            }
        ),
        label="Calibration PIT membership history",
    )
    if (
        source["canonical_generation_id"] != lineage["generation_id"]
        or source["canonical_generation_manifest_sha256"] != lineage["manifest_sha256"]
        or source["canonical_membership_sha256"] != lineage["canonical_sha256"]
    ):
        raise V17PipelineError("calibration PIT history canonical lineage mismatch")
    rows = _validate_ordered_rows(
        source["rows"],
        label="calibration PIT history rows",
        exact_keys=_CALIBRATION_PIT_HISTORY_ROW_KEYS,
        order_key=lambda item: _as_utc_timestamp(
            item["cross_section_date"], label="calibration PIT cross section"
        ),
    )
    by_symbol = _load_canonical_pit_records(bundle, lineage)
    result: dict[pd.Timestamp, tuple[str, ...]] = {}
    for row in rows:
        date = _as_utc_timestamp(row["cross_section_date"], label="calibration PIT cross section")
        symbols_raw = _strict_sequence(
            row["eligible_symbols"], label="calibration PIT eligible_symbols"
        )
        symbols = tuple(
            require_symbol(value, label="calibration PIT symbol") for value in symbols_raw
        )
        if symbols != tuple(sorted(set(symbols))):
            raise V17PipelineError("calibration PIT eligible symbols must be sorted and unique")
        dispositions = row["dispositions"]
        if not isinstance(dispositions, Mapping):
            raise V17PipelineError("calibration PIT dispositions must be an object")
        normalized_dispositions: dict[str, str] = {}
        for symbol, reason in dispositions.items():
            require_symbol(symbol, label="calibration disposition symbol")
            require_identifier(reason, label="calibration disposition reason")
            normalized_dispositions[str(symbol)] = str(reason)
        if set(symbols).intersection(dispositions):
            raise V17PipelineError("calibration PIT disposition overlaps eligible scope")
        replayed_eligible: list[str] = []
        replayed_dispositions: dict[str, str] = {}
        for symbol, record in sorted(by_symbol.items()):
            observed_at = _as_utc_timestamp(
                record.observed_at, label="canonical PIT record observed_at"
            )
            if observed_at > date:
                replayed_dispositions[symbol] = "evidence_after_cross_section"
                continue
            status = evaluate_listing_status(
                record,
                symbol=symbol,
                as_of=date.strftime("%Y-%m-%d"),
            )
            if status.in_universe and status.research_eligible:
                replayed_eligible.append(symbol)
            else:
                replayed_dispositions[symbol] = status.reason
        if symbols != tuple(replayed_eligible) or normalized_dispositions != replayed_dispositions:
            raise V17PipelineError("calibration PIT scope/dispositions replay mismatch")
        if date in result:
            raise V17PipelineError("duplicate calibration PIT cross section")
        result[date] = symbols
    return result


def _expected_calibration_dates(
    sessions: Sequence[pd.Timestamp],
    *,
    kind: str,
) -> tuple[pd.Timestamp, ...]:
    lookback = 2520 if kind == "FUNDAMENTAL" else 1260
    maturity = 378 if kind == "FUNDAMENTAL" else 60
    start = max(0, len(sessions) - 1 - lookback)
    end = len(sessions) - 1 - maturity
    if end < start:
        return ()
    return tuple(
        session
        for position, session in enumerate(sessions)
        if start <= position <= end and _is_calendar_month_end(session, sessions)
    )


def _validate_calibration_authority(
    bundle: SourceBindingBundle,
    cutoff: str,
    *,
    kind: str,
    dataset_role: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    prefix = kind.lower()
    manifest_role = f"{prefix}_calibration_input_manifest"
    receipt_role = f"{prefix}_calibration_verification_receipt"
    manifest = _validate_source_envelope(
        _read_bound_source(bundle, manifest_role),
        version=CALIBRATION_INPUT_MANIFEST_VERSION,
        cutoff=cutoff,
        extra_keys=frozenset(
            {
                "kind",
                "generation_id",
                "training_window_start",
                "training_window_end",
                "session_list_sha256",
                "pit_membership_history_sha256",
                "raw_predictor_manifest_sha256",
                "raw_return_manifest_sha256",
                "expected_observation_keys_sha256",
                "expected_observation_count",
                "disposition_manifest_sha256",
                "generator_implementation_sha256",
                "feature_availability_contract",
                "sample_scope_complete",
            }
        ),
        label=f"{kind} calibration input manifest",
    )
    if manifest["kind"] != kind:
        raise V17PipelineError(f"{kind} calibration input kind mismatch")
    require_identifier(manifest["generation_id"], label="calibration generation_id")
    _as_utc_timestamp(manifest["training_window_start"], label="training window start")
    _as_utc_timestamp(manifest["training_window_end"], label="training window end")
    for field in (
        "session_list_sha256",
        "pit_membership_history_sha256",
        "raw_predictor_manifest_sha256",
        "raw_return_manifest_sha256",
        "expected_observation_keys_sha256",
        "disposition_manifest_sha256",
        "generator_implementation_sha256",
    ):
        require_sha256(manifest[field], label=field)
    count = manifest["expected_observation_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise V17PipelineError("calibration expected observation count invalid")
    keys = _calibration_observation_keys(rows)
    keys_sha = hashlib.sha256(canonical_json_bytes(keys)).hexdigest()
    rows_sha = hashlib.sha256(canonical_json_bytes(list(rows))).hexdigest()
    observed_dates = sorted(str(row["cross_section_date"]) for row in rows)
    sessions = _load_open_day_calendar(bundle, cutoff)
    session_list_sha = hashlib.sha256(
        canonical_json_bytes([item.isoformat() for item in sessions])
    ).hexdigest()
    raw_predictor_role = (
        "fundamental_calibration_raw_predictors"
        if kind == "FUNDAMENTAL"
        else "quant_calibration_raw_bars"
    )
    raw_return_manifest_sha = hashlib.sha256(
        canonical_json_bytes(
            {
                role: _source_sha(bundle, role)
                for role in (
                    "H00300_total_return",
                    "dividend_total_return",
                    "official_delisting_cash",
                )
            }
        )
    ).hexdigest()
    if (
        count != len(rows)
        or manifest["training_window_start"] != observed_dates[0]
        or manifest["training_window_end"] != observed_dates[-1]
        or manifest["expected_observation_keys_sha256"] != keys_sha
        or manifest["generator_implementation_sha256"] != _calibration_generator_sha256(kind)
        or manifest["session_list_sha256"] != session_list_sha
        or manifest["pit_membership_history_sha256"]
        != _source_sha(bundle, "calibration_pit_membership_history")
        or manifest["raw_predictor_manifest_sha256"] != _source_sha(bundle, raw_predictor_role)
        or manifest["raw_return_manifest_sha256"] != raw_return_manifest_sha
        or manifest["disposition_manifest_sha256"]
        != _source_sha(bundle, "calibration_pit_membership_history")
        or manifest["feature_availability_contract"]
        != "PREDICTOR_AT_OR_BEFORE_START_LABEL_AT_OR_AFTER_END"
        or manifest["sample_scope_complete"] is not True
    ):
        raise V17PipelineError(f"{kind} calibration input manifest replay mismatch")
    receipt = _validate_source_envelope(
        _read_bound_source(bundle, receipt_role),
        version=CALIBRATION_RECEIPT_VERSION,
        cutoff=cutoff,
        extra_keys=frozenset(
            {
                "kind",
                "generation_id",
                "input_manifest_sha256",
                "input_manifest_semantic_sha256",
                "dataset_sha256",
                "dataset_semantic_sha256",
                "observed_observation_keys_sha256",
                "observed_rows_sha256",
                "verifier_id",
                "verification_status",
                "predictors_replayed_from_pit",
                "complete_scope_replay_passed",
                "shuffled_label_attack_rejected",
                "missing_loser_attack_rejected",
            }
        ),
        label=f"{kind} calibration verification receipt",
    )
    dataset = _read_bound_source(bundle, dataset_role)
    expected = {
        "kind": kind,
        "generation_id": manifest["generation_id"],
        "input_manifest_sha256": _source_sha(bundle, manifest_role),
        "input_manifest_semantic_sha256": manifest["semantic_sha256"],
        "dataset_sha256": _source_sha(bundle, dataset_role),
        "dataset_semantic_sha256": dataset["semantic_sha256"],
        "observed_observation_keys_sha256": keys_sha,
        "observed_rows_sha256": rows_sha,
        "verifier_id": "codex_delegated_reviewer",
        "verification_status": "PASSED",
    }
    if any(receipt[field] != value for field, value in expected.items()):
        raise V17PipelineError(f"{kind} calibration verification receipt drift")
    for field in (
        "predictors_replayed_from_pit",
        "complete_scope_replay_passed",
        "shuffled_label_attack_rejected",
        "missing_loser_attack_rejected",
    ):
        if receipt[field] is not True:
            raise V17PipelineError(f"{kind} calibration verification failed: {field}")


def _source_sha(bundle: SourceBindingBundle, role: str) -> str:
    item = bundle.by_role[role]
    if item["availability"] != Availability.AVAILABLE.value:
        raise V17PipelineError(f"source unavailable: {role}")
    return require_sha256(item["byte_sha256"], label=f"{role} byte SHA-256")


def _validate_ordered_rows(
    raw: Any,
    *,
    label: str,
    exact_keys: frozenset[str],
    order_key: Any,
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    rows = _strict_sequence(raw, label=label, allow_empty=allow_empty)
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(rows):
        if not isinstance(item, Mapping):
            raise V17PipelineError(f"{label}[{index}] must be an object")
        require_exact_keys(item, exact_keys, label=f"{label}[{index}]")
        normalized.append(dict(item))
    if normalized != sorted(normalized, key=order_key):
        raise V17PipelineError(f"{label} must be canonically ordered")
    return normalized


def _nullable_number(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    return require_number(value, label=label)


def _canonical_trade_date(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise V17PipelineError(f"{label} must be YYYYMMDD or YYYY-MM-DD")
    if len(value) == 8 and value.isdigit():
        date_format = "%Y%m%d"
    elif (
        len(value) == 10
        and value[4] == "-"
        and value[7] == "-"
        and value.replace("-", "").isdigit()
    ):
        date_format = "%Y-%m-%d"
    else:
        raise V17PipelineError(f"{label} must be YYYYMMDD or YYYY-MM-DD")
    try:
        parsed = pd.to_datetime(value, format=date_format, errors="raise")
    except (TypeError, ValueError) as exc:
        raise V17PipelineError(f"{label} is invalid") from exc
    normalized = parsed.strftime("%Y-%m-%d")
    if parsed.strftime(date_format) != value:
        raise V17PipelineError(f"{label} is not canonical")
    return normalized


def _load_market_lineage(bundle: SourceBindingBundle) -> dict[str, Any]:
    pointer = _read_bound_raw_json(bundle, "market_pointer")
    manifest = _read_bound_raw_json(bundle, "market_snapshot_manifest")
    if pointer.get("status") != "OK" or list(pointer.get("blockers") or []):
        raise V17PipelineError("canonical market pointer is not healthy")
    snapshot_id = require_identifier(pointer.get("snapshot_id"), label="market snapshot_id")
    manifest_path = require_nonempty_string(
        pointer.get("manifest_path"), label="market manifest_path", max_chars=4096
    )
    if Path(manifest_path).name != f"{snapshot_id}.json":
        raise V17PipelineError("canonical market pointer manifest identity mismatch")
    effective_trade_date = _canonical_trade_date(
        pointer.get("latest_complete_trade_date") or pointer.get("latest_trade_date"),
        label="canonical market effective trade date",
    )
    coverage = pointer.get("coverage")
    if (
        not isinstance(coverage, Mapping)
        or coverage.get("coverage_schema_version") != "cn-full-a-coverage.v4"
    ):
        raise V17PipelineError("canonical market pointer exact v4 coverage missing")
    coverage_date = _canonical_trade_date(
        coverage.get("coverage_trade_date") or coverage.get("latest_complete_trade_date"),
        label="canonical market coverage trade date",
    )
    if coverage_date != effective_trade_date:
        raise V17PipelineError("canonical market pointer coverage date mismatch")
    if (
        manifest.get("snapshot_id") != snapshot_id
        or manifest.get("market") != "CN"
        or manifest.get("status") != "OK"
        or manifest.get("manifest_path") != manifest_path
        or _canonical_trade_date(
            manifest.get("latest_complete_trade_date") or manifest.get("latest_trade_date"),
            label="canonical market manifest trade date",
        )
        != effective_trade_date
        or manifest.get("readback_validated") is not True
    ):
        raise V17PipelineError("canonical market pointer/manifest binding invalid")
    manifest_coverage = manifest.get("coverage")
    if not isinstance(manifest_coverage, Mapping) or dict(manifest_coverage) != dict(coverage):
        raise V17PipelineError("canonical market pointer/manifest coverage mismatch")
    return {
        "snapshot_id": snapshot_id,
        "effective_trade_date": effective_trade_date,
        "pointer_sha256": _source_sha(bundle, "market_pointer"),
        "pointer_semantic_sha256": _raw_json_semantic_sha256(pointer),
        "manifest_sha256": _source_sha(bundle, "market_snapshot_manifest"),
        "manifest_semantic_sha256": _raw_json_semantic_sha256(manifest),
    }


def _load_pit_lineage(bundle: SourceBindingBundle) -> dict[str, Any]:
    manifest = _read_bound_raw_json(bundle, "pit_membership_generation_manifest")
    if (
        manifest.get("schema_version") != "cn_pit_universe_manifest.v1"
        or manifest.get("membership_schema_version") != "cn_pit_universe.v1"
    ):
        raise V17PipelineError("canonical PIT generation manifest schema invalid")
    generation_id = require_identifier(manifest.get("generation_id"), label="PIT generation_id")
    canonical_sha = require_sha256(manifest.get("canonical_sha256"), label="PIT canonical SHA-256")
    if canonical_sha != _source_sha(bundle, "pit_membership_canonical"):
        raise V17PipelineError("canonical PIT membership byte binding mismatch")
    row_count = manifest.get("row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        raise V17PipelineError("canonical PIT manifest row_count invalid")
    return {
        "generation_id": generation_id,
        "manifest_sha256": _source_sha(bundle, "pit_membership_generation_manifest"),
        "manifest_semantic_sha256": _raw_json_semantic_sha256(manifest),
        "canonical_sha256": canonical_sha,
        "row_count": row_count,
    }


def _load_canonical_pit_records(
    bundle: SourceBindingBundle,
    lineage: Mapping[str, Any],
) -> dict[str, PITUniverseRecord]:
    path = bundle.object_path("pit_membership_canonical")
    expected_sha = require_sha256(
        lineage["canonical_sha256"], label="canonical PIT membership SHA-256"
    )
    if file_sha256(path) != expected_sha:
        raise V17PipelineError("canonical PIT membership byte binding mismatch")
    try:
        raw = path.read_bytes()
        canonical_frame = pd.read_parquet(io.BytesIO(raw))
    except Exception as exc:
        raise V17PipelineError("canonical PIT membership parquet unreadable") from exc
    if hashlib.sha256(raw).hexdigest() != expected_sha or file_sha256(path) != expected_sha:
        raise V17PipelineError("canonical PIT membership changed during read")
    expected_columns = list(PITUniverseRecord.__dataclass_fields__)
    if list(canonical_frame.columns) != expected_columns:
        raise V17PipelineError("canonical PIT membership parquet schema mismatch")
    canonical_records = [
        PITUniverseRecord.from_dict(item) for item in canonical_frame.to_dict(orient="records")
    ]
    if len(canonical_records) != lineage["row_count"]:
        raise V17PipelineError("canonical PIT membership row-count mismatch")
    return records_by_symbol(canonical_records)


def _load_fundamental_lineage(bundle: SourceBindingBundle) -> dict[str, Any]:
    pointer = _read_bound_raw_json(bundle, "fundamental_generation_pointer")
    manifest = _read_bound_raw_json(bundle, "fundamental_generation_manifest")
    if (
        pointer.get("schema_version") != "cn-fundamental-pointer.v1"
        or pointer.get("status") != "OK"
    ):
        raise V17PipelineError("canonical Fundamental pointer schema/status invalid")
    generation_id = require_identifier(
        pointer.get("generation_id"), label="Fundamental generation_id"
    )
    generation_root = f"_fundamental_generations/{generation_id}"
    if pointer.get("manifest_path") != f"{generation_root}/manifest.json":
        raise V17PipelineError("canonical Fundamental manifest path mismatch")
    if (
        manifest.get("schema_version") != "cn-fundamental-generation.v1"
        or manifest.get("status") != "OK"
        or manifest.get("generation_id") != generation_id
    ):
        raise V17PipelineError("canonical Fundamental pointer/manifest generation mismatch")
    pointer_tables = pointer.get("tables")
    manifest_tables = manifest.get("tables")
    expected_tables = {"fundamental_period", "fundamental_daily", "fundamental_quarantine"}
    if (
        not isinstance(pointer_tables, Mapping)
        or not isinstance(manifest_tables, Mapping)
        or set(pointer_tables) != expected_tables
        or set(manifest_tables) != expected_tables
    ):
        raise V17PipelineError("canonical Fundamental table set mismatch")
    table_sha256s: dict[str, str] = {}
    for table in sorted(expected_tables):
        if pointer_tables[table] != f"{generation_root}/{table}.parquet":
            raise V17PipelineError(f"canonical Fundamental table path mismatch: {table}")
        cell = manifest_tables[table]
        if not isinstance(cell, Mapping):
            raise V17PipelineError(f"canonical Fundamental table manifest invalid: {table}")
        table_sha256s[table] = require_sha256(
            cell.get("sha256"), label=f"Fundamental {table} SHA-256"
        )
    return {
        "generation_id": generation_id,
        "pointer_sha256": _source_sha(bundle, "fundamental_generation_pointer"),
        "pointer_semantic_sha256": _raw_json_semantic_sha256(pointer),
        "manifest_sha256": _source_sha(bundle, "fundamental_generation_manifest"),
        "manifest_semantic_sha256": _raw_json_semantic_sha256(manifest),
        "table_sha256s": table_sha256s,
    }


def _load_pit_membership(bundle: SourceBindingBundle, cutoff: str) -> pd.DataFrame:
    lineage = _load_pit_lineage(bundle)
    canonical_by_symbol = _load_canonical_pit_records(bundle, lineage)
    source = _validate_source_envelope(
        _read_bound_source(bundle, "pit_membership"),
        version=PIT_MEMBERSHIP_SOURCE_VERSION,
        cutoff=cutoff,
        extra_keys=frozenset(
            {
                "universe_id",
                "canonical_generation_id",
                "canonical_generation_manifest_sha256",
                "canonical_generation_manifest_semantic_sha256",
                "canonical_membership_sha256",
                "rows",
            }
        ),
        label="PIT membership source",
    )
    expected_lineage = {
        "canonical_generation_id": lineage["generation_id"],
        "canonical_generation_manifest_sha256": lineage["manifest_sha256"],
        "canonical_generation_manifest_semantic_sha256": lineage["manifest_semantic_sha256"],
        "canonical_membership_sha256": lineage["canonical_sha256"],
    }
    if any(source[field] != expected for field, expected in expected_lineage.items()):
        raise V17PipelineError("PIT membership canonical lineage mismatch")
    if source.get("universe_id") != "CN/full_a":
        raise V17PipelineError("PIT membership universe must be CN/full_a")
    rows = _validate_ordered_rows(
        source["rows"],
        label="PIT membership rows",
        exact_keys=_PIT_ROW_KEYS,
        order_key=lambda item: str(item["symbol"]),
    )
    seen: set[str] = set()
    cutoff_at = parse_utc_timestamp(cutoff, label="cutoff")
    for index, row in enumerate(rows):
        symbol = require_symbol(row["symbol"], label=f"PIT rows[{index}].symbol")
        if symbol in seen:
            raise V17PipelineError(f"duplicate PIT symbol: {symbol}")
        seen.add(symbol)
        require_nonempty_string(row["industry"], label=f"PIT rows[{index}].industry")
        for field in (
            "in_universe",
            "research_eligible",
            "membership_conflict",
            "membership_is_pit",
        ):
            require_bool(row[field], label=f"PIT rows[{index}].{field}")
        if row["universe_id"] != "CN/full_a":
            raise V17PipelineError("PIT row universe mismatch")
        availability = parse_utc_timestamp(row["availability"], label="PIT availability")
        if availability > cutoff_at:
            raise V17PipelineError("PIT membership contains post-cutoff evidence")
        record = canonical_by_symbol.get(symbol)
        if record is None:
            raise V17PipelineError("PIT membership symbol missing from canonical parquet")
        observed_at = parse_utc_timestamp(
            record.observed_at, label="canonical PIT record observed_at"
        )
        status = evaluate_listing_status(
            record,
            symbol=symbol,
            as_of=cutoff_at.strftime("%Y-%m-%d"),
        )
        expected = {
            "industry": record.industry,
            "in_universe": status.in_universe,
            "research_eligible": status.research_eligible,
            "membership_conflict": status.reason == REASON_CONFLICTING_STATUS_ROWS,
            "membership_is_pit": observed_at <= cutoff_at,
            "availability": observed_at,
        }
        observed = {
            "industry": row["industry"],
            "in_universe": row["in_universe"],
            "research_eligible": row["research_eligible"],
            "membership_conflict": row["membership_conflict"],
            "membership_is_pit": row["membership_is_pit"],
            "availability": availability,
        }
        if observed != expected:
            raise V17PipelineError(f"PIT membership canonical replay mismatch: {symbol}")
    if seen != set(canonical_by_symbol):
        raise V17PipelineError("PIT membership canonical symbol coverage mismatch")
    return pd.DataFrame(rows)


def _load_fundamental_snapshot(
    bundle: SourceBindingBundle,
    cutoff: str,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    lineage = _load_fundamental_lineage(bundle)
    source = _validate_source_envelope(
        _read_bound_source(bundle, "fundamental_snapshot"),
        version=FUNDAMENTAL_SNAPSHOT_SOURCE_VERSION,
        cutoff=cutoff,
        extra_keys=frozenset(
            {
                "pit_membership_sha256",
                "canonical_generation_id",
                "canonical_pointer_sha256",
                "canonical_pointer_semantic_sha256",
                "canonical_manifest_sha256",
                "canonical_manifest_semantic_sha256",
                "canonical_table_sha256s",
                "rows",
            }
        ),
        label="Fundamental snapshot source",
    )
    if source["pit_membership_sha256"] != _source_sha(bundle, "pit_membership"):
        raise V17PipelineError("Fundamental snapshot PIT binding mismatch")
    expected_lineage = {
        "canonical_generation_id": lineage["generation_id"],
        "canonical_pointer_sha256": lineage["pointer_sha256"],
        "canonical_pointer_semantic_sha256": lineage["pointer_semantic_sha256"],
        "canonical_manifest_sha256": lineage["manifest_sha256"],
        "canonical_manifest_semantic_sha256": lineage["manifest_semantic_sha256"],
        "canonical_table_sha256s": lineage["table_sha256s"],
    }
    if any(source[field] != expected for field, expected in expected_lineage.items()):
        raise V17PipelineError("Fundamental snapshot canonical lineage mismatch")
    rows = _validate_ordered_rows(
        source["rows"],
        label="Fundamental snapshot rows",
        exact_keys=_FUNDAMENTAL_ROW_KEYS,
        order_key=lambda item: str(item["symbol"]),
        allow_empty=True,
    )
    seen: set[str] = set()
    for index, row in enumerate(rows):
        symbol = require_symbol(row["symbol"], label=f"Fundamental rows[{index}].symbol")
        if symbol in seen:
            raise V17PipelineError(f"duplicate Fundamental symbol: {symbol}")
        seen.add(symbol)
        if parse_utc_timestamp(
            row["availability"], label="Fundamental availability"
        ) > parse_utc_timestamp(cutoff, label="cutoff"):
            raise V17PipelineError("Fundamental snapshot contains post-cutoff evidence")
        for field in (
            "net_profit_ttm",
            "market_cap",
            "cfo_ttm",
            "capex_ttm",
            "fin_roe",
            "fin_roa",
            "fin_ocf_to_profit",
            "fin_fcf_to_profit",
            "fin_net_profit_yoy",
            "forecast_revision",
            "fin_debt_to_assets",
        ):
            _nullable_number(row[field], label=f"Fundamental rows[{index}].{field}")
    membership_symbols = set(membership["symbol"].astype(str))
    extras = sorted(seen - membership_symbols)
    if extras:
        raise V17PipelineError(f"Fundamental symbols outside PIT source: {extras}")

    fundamentals = pd.DataFrame(rows)
    if fundamentals.empty:
        fundamentals = pd.DataFrame(columns=sorted(_FUNDAMENTAL_ROW_KEYS))
    merged = membership.rename(columns={"availability": "membership_availability"}).merge(
        fundamentals.rename(columns={"availability": "fundamental_availability"}),
        on="symbol",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    member_at = pd.to_datetime(merged["membership_availability"], utc=True)
    fundamental_at = pd.to_datetime(merged["fundamental_availability"], utc=True, errors="coerce")
    merged["availability"] = member_at.where(
        fundamental_at.isna(), member_at.combine(fundamental_at, max)
    )
    return merged.drop(columns=["membership_availability", "fundamental_availability"])


def _load_history(bundle: SourceBindingBundle, cutoff: str) -> pd.DataFrame:
    lineage = _load_fundamental_lineage(bundle)
    source = _validate_source_envelope(
        _read_bound_source(bundle, "fundamental_metric_history"),
        version=FUNDAMENTAL_HISTORY_SOURCE_VERSION,
        cutoff=cutoff,
        extra_keys=frozenset(
            {
                "calendar_sha256",
                "canonical_generation_id",
                "canonical_pointer_sha256",
                "canonical_pointer_semantic_sha256",
                "canonical_manifest_sha256",
                "canonical_manifest_semantic_sha256",
                "canonical_table_sha256s",
                "rows",
            }
        ),
        label="Fundamental history source",
    )
    if source["calendar_sha256"] != _source_sha(bundle, "cn_open_day_calendar"):
        raise V17PipelineError("Fundamental history calendar binding mismatch")
    expected_lineage = {
        "canonical_generation_id": lineage["generation_id"],
        "canonical_pointer_sha256": lineage["pointer_sha256"],
        "canonical_pointer_semantic_sha256": lineage["pointer_semantic_sha256"],
        "canonical_manifest_sha256": lineage["manifest_sha256"],
        "canonical_manifest_semantic_sha256": lineage["manifest_semantic_sha256"],
        "canonical_table_sha256s": lineage["table_sha256s"],
    }
    if any(source[field] != expected for field, expected in expected_lineage.items()):
        raise V17PipelineError("Fundamental history canonical lineage mismatch")
    open_sessions = frozenset(_load_open_day_calendar(bundle, cutoff))
    rows = _validate_ordered_rows(
        source["rows"],
        label="Fundamental history rows",
        exact_keys=_HISTORY_ROW_KEYS,
        order_key=lambda item: (
            str(item["symbol"]),
            str(item["metric"]),
            str(item["trade_date"]),
        ),
    )
    seen: set[tuple[str, str, str]] = set()
    cutoff_at = parse_utc_timestamp(cutoff, label="cutoff")
    for index, row in enumerate(rows):
        symbol = require_symbol(row["symbol"], label=f"history[{index}].symbol")
        metric = require_identifier(row["metric"], label=f"history[{index}].metric")
        trade_date = parse_utc_timestamp(row["trade_date"], label="history.trade_date")
        available_at = parse_utc_timestamp(row["availability"], label="history.availability")
        if trade_date > cutoff_at or available_at > cutoff_at:
            raise V17PipelineError("Fundamental history contains post-cutoff evidence")
        if pd.Timestamp(trade_date).tz_convert("UTC") not in open_sessions:
            raise V17PipelineError("Fundamental history date is not a canonical open session")
        if require_bool(row["is_open_day"], label="history.is_open_day") is not True:
            raise V17PipelineError("Fundamental history contains a non-open session")
        require_number(row["value"], label=f"history[{index}].value")
        key = (symbol, metric, str(row["trade_date"]))
        if key in seen:
            raise V17PipelineError(f"duplicate Fundamental history row: {key}")
        seen.add(key)
    return pd.DataFrame(rows)


def _as_utc_timestamp(value: Any, *, label: str) -> pd.Timestamp:
    return pd.Timestamp(parse_utc_timestamp(value, label=label)).tz_convert("UTC")


def _load_open_day_calendar(
    bundle: SourceBindingBundle,
    cutoff: str,
) -> tuple[pd.Timestamp, ...]:
    lineage = _load_market_lineage(bundle)
    source = _validate_source_envelope(
        _read_bound_source(bundle, "cn_open_day_calendar"),
        version=OPEN_DAY_CALENDAR_SOURCE_VERSION,
        cutoff=cutoff,
        extra_keys=frozenset(
            {
                "calendar_id",
                "canonical_snapshot_id",
                "canonical_market_pointer_sha256",
                "canonical_market_pointer_semantic_sha256",
                "canonical_snapshot_manifest_sha256",
                "canonical_snapshot_manifest_semantic_sha256",
                "canonical_effective_trade_date",
                "sessions",
            }
        ),
        label="CN open-day calendar source",
    )
    if source["calendar_id"] != "CN.SSE_SZSE.CANONICAL_OPEN_SESSIONS":
        raise V17PipelineError("CN open-day calendar identity mismatch")
    expected_lineage = {
        "canonical_snapshot_id": lineage["snapshot_id"],
        "canonical_market_pointer_sha256": lineage["pointer_sha256"],
        "canonical_market_pointer_semantic_sha256": lineage["pointer_semantic_sha256"],
        "canonical_snapshot_manifest_sha256": lineage["manifest_sha256"],
        "canonical_snapshot_manifest_semantic_sha256": lineage["manifest_semantic_sha256"],
        "canonical_effective_trade_date": lineage["effective_trade_date"],
    }
    if any(source[field] != expected for field, expected in expected_lineage.items()):
        raise V17PipelineError("CN open-day calendar canonical lineage mismatch")
    rows = _validate_ordered_rows(
        source["sessions"],
        label="CN open-day calendar sessions",
        exact_keys=_CALENDAR_SESSION_KEYS,
        order_key=lambda item: _as_utc_timestamp(item["trade_date"], label="calendar.trade_date"),
    )
    cutoff_at = _as_utc_timestamp(cutoff, label="cutoff")
    sessions: list[pd.Timestamp] = []
    for index, row in enumerate(rows):
        trade_date = _as_utc_timestamp(
            row["trade_date"], label=f"calendar.sessions[{index}].trade_date"
        )
        available_at = _as_utc_timestamp(
            row["availability"], label=f"calendar.sessions[{index}].availability"
        )
        if require_bool(row["is_open_day"], label="calendar.is_open_day") is not True:
            raise V17PipelineError("calendar contains a non-open session")
        if trade_date > cutoff_at or available_at > cutoff_at or available_at < trade_date:
            raise V17PipelineError("calendar session PIT timing invalid")
        sessions.append(trade_date)
    if len(sessions) != len(set(sessions)):
        raise V17PipelineError("calendar sessions must be unique")
    if sessions[-1].strftime("%Y-%m-%d") != lineage["effective_trade_date"]:
        raise V17PipelineError("calendar tail is not canonical effective session")
    return tuple(sessions)


def _session_positions(sessions: Sequence[pd.Timestamp]) -> dict[pd.Timestamp, int]:
    return {session: index for index, session in enumerate(sessions)}


def _is_calendar_month_end(session: pd.Timestamp, sessions: Sequence[pd.Timestamp]) -> bool:
    month_sessions = [
        item for item in sessions if (item.year, item.month) == (session.year, session.month)
    ]
    return bool(month_sessions and session == month_sessions[-1])


def _load_market_pointer(
    bundle: SourceBindingBundle,
    cutoff: str,
    sessions: Sequence[pd.Timestamp],
) -> dict[str, Any]:
    del cutoff
    lineage = _load_market_lineage(bundle)
    if not sessions or sessions[-1].strftime("%Y-%m-%d") != lineage["effective_trade_date"]:
        raise V17PipelineError("market pointer does not identify effective cutoff session")
    return lineage


def _load_benchmark_total_return(
    bundle: SourceBindingBundle,
    cutoff: str,
) -> dict[tuple[pd.Timestamp, pd.Timestamp], dict[str, Any]]:
    source = _validate_source_envelope(
        _read_bound_source(bundle, "H00300_total_return"),
        version=BENCHMARK_TOTAL_RETURN_SOURCE_VERSION,
        cutoff=cutoff,
        extra_keys=frozenset({"benchmark_symbol", "calendar_sha256", "rows"}),
        label="H00300 total-return source",
    )
    if source["benchmark_symbol"] != "H00300.CSI":
        raise V17PipelineError("benchmark total-return identity mismatch")
    if source["calendar_sha256"] != _source_sha(bundle, "cn_open_day_calendar"):
        raise V17PipelineError("benchmark total-return calendar binding mismatch")
    rows = _validate_ordered_rows(
        source["rows"],
        label="benchmark total-return rows",
        exact_keys=_BENCHMARK_TOTAL_RETURN_ROW_KEYS,
        order_key=lambda item: (
            _as_utc_timestamp(item["start_trade_date"], label="benchmark.start"),
            _as_utc_timestamp(item["end_trade_date"], label="benchmark.end"),
        ),
    )
    result: dict[tuple[pd.Timestamp, pd.Timestamp], dict[str, Any]] = {}
    cutoff_at = _as_utc_timestamp(cutoff, label="cutoff")
    for index, row in enumerate(rows):
        start = _as_utc_timestamp(row["start_trade_date"], label=f"benchmark[{index}].start")
        end = _as_utc_timestamp(row["end_trade_date"], label=f"benchmark[{index}].end")
        available = _as_utc_timestamp(row["availability"], label="benchmark availability")
        start_index = require_number(
            row["start_total_return_index"],
            label="benchmark start index",
            minimum=0.0,
            minimum_exclusive=True,
        )
        end_index = require_number(
            row["end_total_return_index"],
            label="benchmark end index",
            minimum=0.0,
            minimum_exclusive=True,
        )
        if end <= start or available < end or available > cutoff_at:
            raise V17PipelineError("benchmark total-return PIT timing invalid")
        key = (start, end)
        if key in result:
            raise V17PipelineError("duplicate benchmark total-return interval")
        result[key] = {"return": end_index / start_index - 1.0, "availability": available}
    return result


def _load_stock_total_return(
    bundle: SourceBindingBundle,
    cutoff: str,
) -> dict[tuple[str, pd.Timestamp, pd.Timestamp], dict[str, Any]]:
    source = _validate_source_envelope(
        _read_bound_source(bundle, "dividend_total_return"),
        version=DIVIDEND_TOTAL_RETURN_SOURCE_VERSION,
        cutoff=cutoff,
        extra_keys=frozenset({"calendar_sha256", "rows"}),
        label="stock dividend total-return source",
    )
    if source["calendar_sha256"] != _source_sha(bundle, "cn_open_day_calendar"):
        raise V17PipelineError("stock total-return calendar binding mismatch")
    rows = _validate_ordered_rows(
        source["rows"],
        label="stock total-return rows",
        exact_keys=_DIVIDEND_TOTAL_RETURN_ROW_KEYS,
        order_key=lambda item: (
            str(item["symbol"]),
            _as_utc_timestamp(item["start_trade_date"], label="stock return start"),
            _as_utc_timestamp(item["end_trade_date"], label="stock return end"),
        ),
    )
    result: dict[tuple[str, pd.Timestamp, pd.Timestamp], dict[str, Any]] = {}
    cutoff_at = _as_utc_timestamp(cutoff, label="cutoff")
    for index, row in enumerate(rows):
        symbol = require_symbol(row["symbol"], label=f"stock return[{index}].symbol")
        start = _as_utc_timestamp(row["start_trade_date"], label="stock return start")
        end = _as_utc_timestamp(row["end_trade_date"], label="stock return end")
        available = _as_utc_timestamp(row["availability"], label="stock return availability")
        start_index = require_number(
            row["start_total_return_index"],
            label="stock start index",
            minimum=0.0,
            minimum_exclusive=True,
        )
        end_index = require_number(
            row["end_total_return_index"],
            label="stock end index",
            minimum=0.0,
            minimum_exclusive=True,
        )
        if end <= start or available < end or available > cutoff_at:
            raise V17PipelineError("stock total-return PIT timing invalid")
        key = (symbol, start, end)
        if key in result:
            raise V17PipelineError("duplicate stock total-return interval")
        result[key] = {"return": end_index / start_index - 1.0, "availability": available}
    return result


def _load_delisting_evidence(
    bundle: SourceBindingBundle,
    cutoff: str,
) -> dict[tuple[str, pd.Timestamp], tuple[bool, bool, pd.Timestamp]]:
    source = _validate_source_envelope(
        _read_bound_source(bundle, "official_delisting_cash"),
        version=DELISTING_CASH_SOURCE_VERSION,
        cutoff=cutoff,
        extra_keys=frozenset({"rows"}),
        label="official delisting-cash source",
    )
    rows = _validate_ordered_rows(
        source["rows"],
        label="official delisting-cash rows",
        exact_keys=_DELISTING_ROW_KEYS,
        order_key=lambda item: (
            str(item["symbol"]),
            _as_utc_timestamp(item["end_trade_date"], label="delisting end"),
        ),
    )
    result: dict[tuple[str, pd.Timestamp], tuple[bool, bool, pd.Timestamp]] = {}
    cutoff_at = _as_utc_timestamp(cutoff, label="cutoff")
    for index, row in enumerate(rows):
        symbol = require_symbol(row["symbol"], label=f"delisting[{index}].symbol")
        end = _as_utc_timestamp(row["end_trade_date"], label="delisting end")
        available = _as_utc_timestamp(row["availability"], label="delisting availability")
        delisted = require_bool(row["delisted"], label="delisted")
        terminal = require_bool(
            row["official_terminal_cash_settlement"], label="official terminal cash"
        )
        if available < end or available > cutoff_at or (not delisted and terminal):
            raise V17PipelineError("official delisting evidence invalid")
        key = (symbol, end)
        if key in result:
            raise V17PipelineError("duplicate official delisting evidence")
        result[key] = (delisted, terminal, available)
    return result


def _same_number(left: Any, right: float) -> bool:
    try:
        value = require_number(left, label="declared derived number")
    except V17ContractError:
        return False
    return math.isclose(value, right, rel_tol=0.0, abs_tol=1e-12)


def _recomputed_return_fields(
    *,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    stock_returns: Mapping[tuple[str, pd.Timestamp, pd.Timestamp], Mapping[str, Any]],
    benchmark_returns: Mapping[tuple[pd.Timestamp, pd.Timestamp], Mapping[str, Any]],
    delisting: Mapping[tuple[str, pd.Timestamp], tuple[bool, bool, pd.Timestamp]],
) -> tuple[float, float, bool, bool, pd.Timestamp]:
    stock = stock_returns.get((symbol, start, end))
    benchmark = benchmark_returns.get((start, end))
    terminal = delisting.get((symbol, end))
    if stock is None or benchmark is None or terminal is None:
        raise V17PipelineError("forward-return raw evidence interval missing")
    available = max(stock["availability"], benchmark["availability"], terminal[2])
    return (
        float(stock["return"]),
        float(benchmark["return"]),
        bool(terminal[0]),
        bool(terminal[1]),
        available,
    )


def _replay_fundamental_calibration_predictors(
    bundle: SourceBindingBundle,
    cutoff: str,
    *,
    calibration_rows: Sequence[Mapping[str, Any]],
) -> None:
    source = _validate_source_envelope(
        _read_bound_source(bundle, "fundamental_calibration_raw_predictors"),
        version=FUNDAMENTAL_CALIBRATION_RAW_SOURCE_VERSION,
        cutoff=cutoff,
        extra_keys=frozenset(
            {
                "calendar_sha256",
                "pit_generation_manifest_sha256",
                "fundamental_generation_manifest_sha256",
                "rows",
            }
        ),
        label="Fundamental calibration raw predictors",
    )
    if (
        source["calendar_sha256"] != _source_sha(bundle, "cn_open_day_calendar")
        or source["pit_generation_manifest_sha256"]
        != _source_sha(bundle, "pit_membership_generation_manifest")
        or source["fundamental_generation_manifest_sha256"]
        != _source_sha(bundle, "fundamental_generation_manifest")
    ):
        raise V17PipelineError("Fundamental calibration raw predictor lineage mismatch")
    raw_rows = _validate_ordered_rows(
        source["rows"],
        label="Fundamental calibration raw predictor rows",
        exact_keys=_FUNDAMENTAL_RAW_PREDICTOR_ROW_KEYS,
        order_key=lambda item: (
            str(item["symbol"]),
            _as_utc_timestamp(item["trade_date"], label="raw Fundamental trade date"),
        ),
    )
    calendar_sessions = _load_open_day_calendar(bundle, cutoff)
    calendar = frozenset(calendar_sessions)
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, pd.Timestamp]] = set()
    for index, raw in enumerate(raw_rows):
        row = dict(raw)
        symbol = require_symbol(row["symbol"], label=f"raw Fundamental[{index}].symbol")
        trade_date = _as_utc_timestamp(row["trade_date"], label="raw Fundamental trade date")
        available = _as_utc_timestamp(row["availability"], label="raw Fundamental availability")
        if trade_date not in calendar or available > trade_date:
            raise V17PipelineError("raw Fundamental predictor is not PIT at open session")
        if require_bool(row["is_open_day"], label="raw Fundamental is_open_day") is not True:
            raise V17PipelineError("raw Fundamental predictor contains non-open session")
        key = (symbol, trade_date)
        if key in seen:
            raise V17PipelineError("duplicate raw Fundamental predictor row")
        seen.add(key)
        row["trade_date"] = trade_date
        row["availability"] = available
        normalized.append(row)
    raw_frame = pd.DataFrame(normalized)
    pit_history = _load_calibration_pit_history(bundle, cutoff)
    calibration = pd.DataFrame(list(calibration_rows))
    calibration["cross_section_date"] = pd.to_datetime(calibration["cross_section_date"], utc=True)
    expected_dates = set(_expected_calibration_dates(calendar_sessions, kind="FUNDAMENTAL"))
    if set(calibration["cross_section_date"].unique()) != expected_dates:
        raise V17PipelineError("Fundamental calibration training-date coverage mismatch")
    raw_frame["fcf_to_price"] = (
        pd.to_numeric(raw_frame["cfo_ttm"], errors="raise")
        - pd.to_numeric(raw_frame["capex_ttm"], errors="raise")
    ) / pd.to_numeric(raw_frame["market_cap"], errors="raise")
    raw_frame["fin_fcf_to_profit"] = (
        pd.to_numeric(raw_frame["cfo_ttm"], errors="raise")
        - pd.to_numeric(raw_frame["capex_ttm"], errors="raise")
    ) / pd.to_numeric(raw_frame["net_profit_ttm"], errors="raise")
    for cross_section in sorted(calibration["cross_section_date"].unique()):
        date = pd.Timestamp(cross_section).tz_convert("UTC")
        expected_symbols = pit_history.get(date)
        if expected_symbols is None:
            raise V17PipelineError("Fundamental calibration PIT scope missing")
        snapshot = raw_frame.loc[
            (raw_frame["trade_date"] == date) & raw_frame["symbol"].isin(expected_symbols)
        ].copy()
        if tuple(sorted(snapshot["symbol"].astype(str))) != expected_symbols:
            raise V17PipelineError("Fundamental calibration raw scope is incomplete")
        historical = raw_frame.loc[
            (raw_frame["trade_date"] <= date) & raw_frame["symbol"].isin(expected_symbols)
        ]
        history_rows: list[dict[str, Any]] = []
        for raw_item in historical.to_dict("records"):
            for metric in ALL_METRICS:
                history_rows.append(
                    {
                        "symbol": raw_item["symbol"],
                        "trade_date": raw_item["trade_date"],
                        "availability": raw_item["availability"],
                        "is_open_day": True,
                        "metric": metric,
                        "value": raw_item[metric],
                    }
                )
        scored = score_fundamental_universe(
            snapshot,
            pd.DataFrame(history_rows),
            cutoff=date,
            holdings=(),
            top_n=len(expected_symbols),
        ).scored
        if not scored["status"].astype(str).eq("AVAILABLE").all():
            raise V17PipelineError("Fundamental calibration raw replay produced unavailable score")
        deciles = _score_deciles(scored)
        scored_by_symbol = scored.set_index("symbol", drop=False)
        for horizon in HORIZONS:
            cell = calibration.loc[
                (calibration["cross_section_date"] == date) & (calibration["horizon"] == horizon)
            ]
            if tuple(sorted(cell["symbol"].astype(str))) != expected_symbols:
                raise V17PipelineError("Fundamental calibration omitted eligible observations")
            for item in cell.to_dict("records"):
                symbol = str(item["symbol"])
                replayed = scored_by_symbol.loc[symbol]
                if (
                    item["industry"] != replayed["industry"]
                    or not _same_number(item["fundamental_score"], float(replayed["total_score"]))
                    or int(item["score_decile"]) != deciles[symbol]
                ):
                    raise V17PipelineError("Fundamental calibration predictor replay mismatch")


def _replay_quant_calibration_predictors(
    bundle: SourceBindingBundle,
    cutoff: str,
    *,
    calibration_rows: Sequence[Mapping[str, Any]],
) -> None:
    source = _validate_source_envelope(
        _read_bound_source(bundle, "quant_calibration_raw_bars"),
        version=QUANT_CALIBRATION_RAW_SOURCE_VERSION,
        cutoff=cutoff,
        extra_keys=frozenset(
            {"calendar_sha256", "canonical_snapshot_manifest_sha256", "cross_sections"}
        ),
        label="Quant calibration raw bars",
    )
    if source["calendar_sha256"] != _source_sha(bundle, "cn_open_day_calendar") or source[
        "canonical_snapshot_manifest_sha256"
    ] != _source_sha(bundle, "market_snapshot_manifest"):
        raise V17PipelineError("Quant calibration raw bars lineage mismatch")
    cross_sections = _validate_ordered_rows(
        source["cross_sections"],
        label="Quant raw cross sections",
        exact_keys=_QUANT_RAW_CROSS_SECTION_KEYS,
        order_key=lambda item: _as_utc_timestamp(
            item["cross_section_date"], label="Quant raw cross section"
        ),
    )
    sessions = _load_open_day_calendar(bundle, cutoff)
    positions = _session_positions(sessions)
    pit_history = _load_calibration_pit_history(bundle, cutoff)
    calibration = pd.DataFrame(list(calibration_rows))
    calibration["cross_section_date"] = pd.to_datetime(calibration["cross_section_date"], utc=True)
    factor_names = (
        "pv_blend_volstab19x2_mom90_amihud5_w80",
        "pv_short_reversal_25d",
        "pv_downside_volatility_15d",
    )
    observed_dates: set[pd.Timestamp] = set()
    for cross in cross_sections:
        date = _as_utc_timestamp(cross["cross_section_date"], label="Quant raw cross section")
        if date in observed_dates or date not in positions:
            raise V17PipelineError("Quant raw cross section invalid or duplicate")
        observed_dates.add(date)
        expected_symbols = pit_history.get(date)
        if expected_symbols is None:
            raise V17PipelineError("Quant calibration PIT scope missing")
        symbol_items = _validate_ordered_rows(
            cross["symbols"],
            label="Quant raw symbols",
            exact_keys=_QUANT_RAW_SYMBOL_KEYS,
            order_key=lambda item: str(item["symbol"]),
        )
        if tuple(str(item["symbol"]) for item in symbol_items) != expected_symbols:
            raise V17PipelineError("Quant raw predictor scope is incomplete")
        frames: dict[str, pd.DataFrame] = {}
        for item in symbol_items:
            symbol = require_symbol(item["symbol"], label="Quant raw symbol")
            bars = _validate_ordered_rows(
                item["bars"],
                label=f"Quant raw bars {symbol}",
                exact_keys=_BAR_KEYS,
                order_key=lambda row: _as_utc_timestamp(
                    row["trade_date"], label="Quant raw bar date"
                ),
            )
            normalized_bars: list[dict[str, Any]] = []
            bar_dates: list[pd.Timestamp] = []
            for raw_bar in bars:
                bar = dict(raw_bar)
                bar_date = _as_utc_timestamp(bar["trade_date"], label="Quant raw bar date")
                available = _as_utc_timestamp(
                    bar["availability"], label="Quant raw bar availability"
                )
                if bar_date not in positions or bar_date > date or available > date:
                    raise V17PipelineError("Quant raw bar violates PIT/session boundary")
                if require_bool(bar["is_open_day"], label="Quant raw is_open_day") is not True:
                    raise V17PipelineError("Quant raw bar contains non-open session")
                for field in ("close", "volume", "amount"):
                    require_number(
                        bar[field],
                        label=f"Quant raw {field}",
                        minimum=0.0,
                        minimum_exclusive=True,
                    )
                bar["trade_date"] = bar_date
                bar["availability"] = available
                bar_dates.append(bar_date)
                normalized_bars.append(bar)
            if len(bar_dates) < 91 or bar_dates[-1] != date:
                raise V17PipelineError("Quant raw factor history incomplete")
            expected_bar_dates = list(sessions[positions[bar_dates[0]] : positions[date] + 1])
            if bar_dates != expected_bar_dates:
                raise V17PipelineError("Quant raw factor history has session gaps")
            frames[symbol] = pd.DataFrame(normalized_bars)
        replayed = {factor: compute_price_volume_factor(factor, frames) for factor in factor_names}
        for horizon in (20, 60):
            cell = calibration.loc[
                (calibration["cross_section_date"] == date) & (calibration["horizon"] == horizon)
            ]
            if tuple(sorted(cell["symbol"].astype(str))) != expected_symbols:
                raise V17PipelineError("Quant calibration omitted eligible observations")
            for item in cell.to_dict("records"):
                symbol = str(item["symbol"])
                for factor in factor_names:
                    if not _same_number(item[factor], float(replayed[factor].loc[symbol])):
                        raise V17PipelineError("Quant calibration predictor replay mismatch")
    dataset_dates = set(calibration["cross_section_date"].unique())
    expected_cross_section_dates = set(_expected_calibration_dates(sessions, kind="QUANT"))
    if (
        observed_dates != {pd.Timestamp(value).tz_convert("UTC") for value in dataset_dates}
        or observed_dates != expected_cross_section_dates
    ):
        raise V17PipelineError("Quant raw cross-section coverage mismatch")


def _load_forward_observations(bundle: SourceBindingBundle, cutoff: str) -> pd.DataFrame:
    source = _validate_source_envelope(
        _read_bound_source(bundle, "fundamental_forward_calibration"),
        version=FORWARD_CALIBRATION_SOURCE_VERSION,
        cutoff=cutoff,
        extra_keys=frozenset(
            {
                "benchmark_total_return_sha256",
                "dividend_total_return_sha256",
                "official_delisting_cash_sha256",
                "calendar_sha256",
                "rows",
            }
        ),
        label="Forward calibration source",
    )
    expected_bindings = {
        "benchmark_total_return_sha256": _source_sha(bundle, "H00300_total_return"),
        "dividend_total_return_sha256": _source_sha(bundle, "dividend_total_return"),
        "official_delisting_cash_sha256": _source_sha(bundle, "official_delisting_cash"),
        "calendar_sha256": _source_sha(bundle, "cn_open_day_calendar"),
    }
    for field, expected in expected_bindings.items():
        if source[field] != expected:
            raise V17PipelineError(f"Forward calibration evidence binding mismatch: {field}")
    rows = _validate_ordered_rows(
        source["rows"],
        label="Forward calibration rows",
        exact_keys=_FORWARD_ROW_KEYS,
        order_key=lambda item: (
            str(item["industry"]),
            int(item["score_decile"]),
            int(item["horizon"]),
            str(item["cross_section_date"]),
            str(item["symbol"]),
        ),
    )
    sessions = _load_open_day_calendar(bundle, cutoff)
    positions = _session_positions(sessions)
    stock_returns = _load_stock_total_return(bundle, cutoff)
    benchmark_returns = _load_benchmark_total_return(bundle, cutoff)
    delisting = _load_delisting_evidence(bundle, cutoff)
    cutoff_at = _as_utc_timestamp(cutoff, label="cutoff")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, pd.Timestamp, int]] = set()
    for index, raw in enumerate(rows):
        row = dict(raw)
        symbol = require_symbol(row["symbol"], label=f"forward[{index}].symbol")
        require_nonempty_string(row["industry"], label=f"forward[{index}].industry")
        require_ratio(row["fundamental_score"], label="forward fundamental_score")
        horizon = row["horizon"]
        decile = row["score_decile"]
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in HORIZONS:
            raise V17PipelineError("forward horizon invalid")
        if isinstance(decile, bool) or not isinstance(decile, int) or decile not in range(1, 11):
            raise V17PipelineError("forward score decile invalid")
        start = _as_utc_timestamp(row["cross_section_date"], label="forward cross section")
        end = _as_utc_timestamp(row["stock_end_trade_date"], label="forward stock end")
        if start not in positions or end not in positions:
            raise V17PipelineError("forward interval is not on canonical open-day calendar")
        start_position = positions[start]
        expected_end_position = start_position + horizon
        if expected_end_position >= len(sessions) or sessions[expected_end_position] != end:
            raise V17PipelineError("forward interval does not span exact open sessions")
        age = len(sessions) - 1 - start_position
        availability = _as_utc_timestamp(row["availability"], label="forward availability")
        predictor_available = _as_utc_timestamp(
            row["predictor_available_at"], label="forward predictor availability"
        )
        label_available = _as_utc_timestamp(
            row["label_available_at"], label="forward label availability"
        )
        stock_return, benchmark_return, delisted, terminal, raw_available = (
            _recomputed_return_fields(
                symbol=symbol,
                start=start,
                end=end,
                stock_returns=stock_returns,
                benchmark_returns=benchmark_returns,
                delisting=delisting,
            )
        )
        exact_timestamps = all(
            _as_utc_timestamp(row[field], label=f"forward.{field}") == expected
            for field, expected in (
                ("stock_start_trade_date", start),
                ("benchmark_start_trade_date", start),
                ("benchmark_end_trade_date", end),
            )
        )
        if not exact_timestamps:
            raise V17PipelineError("forward stock/benchmark session binding mismatch")
        if (
            predictor_available > start
            or label_available < end
            or label_available < raw_available
            or label_available > cutoff_at
            or availability != label_available
        ):
            raise V17PipelineError("forward availability does not cover raw evidence")
        derived_checks = (
            row["age_open_days"] == age,
            row["realized_open_days"] == horizon,
            row["is_pit_month_end"] is _is_calendar_month_end(start, sessions),
            row["is_mature"] is True,
            row["stock_return_includes_dividends"] is True,
            row["benchmark_return_is_pre_tax_total_return"] is True,
            row["benchmark_symbol"] == "H00300.CSI",
            row["delisted"] is delisted,
            row["official_terminal_cash_settlement"] is terminal,
            _same_number(row["stock_total_return"], stock_return),
            _same_number(row["benchmark_total_return"], benchmark_return),
        )
        if not all(derived_checks):
            raise V17PipelineError("forward derived field disagrees with raw evidence")
        if delisted and not terminal:
            raise V17PipelineError("delisted sample lacks official terminal cash settlement")
        key = (symbol, start, horizon)
        if key in seen:
            raise V17PipelineError("duplicate forward calibration observation")
        seen.add(key)
        row["stock_total_return"] = stock_return
        row["benchmark_total_return"] = benchmark_return
        normalized.append(row)

    frame = pd.DataFrame(normalized)
    for _, group in frame.groupby(["industry", "cross_section_date", "horizon"], sort=True):
        ordered = group.sort_values(
            ["fundamental_score", "symbol"], ascending=[True, True], kind="mergesort"
        )
        count = len(ordered)
        expected_deciles = {
            int(index): min(10, (position * 10) // count + 1)
            for position, index in enumerate(ordered.index)
        }
        if any(
            int(frame.at[index, "score_decile"]) != value
            for index, value in expected_deciles.items()
        ):
            raise V17PipelineError("forward score decile is not reproducible from sealed scores")
    _replay_fundamental_calibration_predictors(
        bundle,
        cutoff,
        calibration_rows=rows,
    )
    _validate_calibration_authority(
        bundle,
        cutoff,
        kind="FUNDAMENTAL",
        dataset_role="fundamental_forward_calibration",
        rows=rows,
    )
    return frame


def _load_market_snapshot(
    bundle: SourceBindingBundle,
    cutoff: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], pd.DataFrame]:
    sessions = _load_open_day_calendar(bundle, cutoff)
    pointer = _load_market_pointer(bundle, cutoff, sessions)
    source = _validate_source_envelope(
        _read_bound_source(bundle, "market_snapshot"),
        version=MARKET_SNAPSHOT_SOURCE_VERSION,
        cutoff=cutoff,
        extra_keys=frozenset(
            {
                "snapshot_id",
                "calendar_sha256",
                "canonical_market_pointer_sha256",
                "canonical_market_pointer_semantic_sha256",
                "canonical_snapshot_manifest_sha256",
                "canonical_snapshot_manifest_semantic_sha256",
                "canonical_effective_trade_date",
                "benchmark_symbol",
                "benchmark_bars",
                "symbols",
            }
        ),
        label="Market snapshot source",
    )
    if source["snapshot_id"] != pointer["snapshot_id"]:
        raise V17PipelineError("Market snapshot pointer identity mismatch")
    if source["calendar_sha256"] != _source_sha(bundle, "cn_open_day_calendar"):
        raise V17PipelineError("Market snapshot calendar binding mismatch")
    if source["benchmark_symbol"] != "H00300.CSI":
        raise V17PipelineError("Market snapshot benchmark identity mismatch")
    expected_lineage = {
        "canonical_market_pointer_sha256": pointer["pointer_sha256"],
        "canonical_market_pointer_semantic_sha256": pointer["pointer_semantic_sha256"],
        "canonical_snapshot_manifest_sha256": pointer["manifest_sha256"],
        "canonical_snapshot_manifest_semantic_sha256": pointer["manifest_semantic_sha256"],
        "canonical_effective_trade_date": pointer["effective_trade_date"],
    }
    if any(source[field] != expected for field, expected in expected_lineage.items()):
        raise V17PipelineError("Market snapshot canonical lineage mismatch")

    cutoff_at = _as_utc_timestamp(cutoff, label="cutoff")

    def normalized_bars(raw: Any, *, label: str) -> pd.DataFrame:
        bars = _validate_ordered_rows(
            raw,
            label=label,
            exact_keys=_BAR_KEYS,
            order_key=lambda row: _as_utc_timestamp(row["trade_date"], label=f"{label}.trade_date"),
        )
        positions = _session_positions(sessions)
        dates: list[pd.Timestamp] = []
        normalized: list[dict[str, Any]] = []
        for bar_index, raw_bar in enumerate(bars):
            bar = dict(raw_bar)
            parsed = _as_utc_timestamp(bar["trade_date"], label=f"{label}[{bar_index}].trade_date")
            available_at = _as_utc_timestamp(
                bar["availability"], label=f"{label}[{bar_index}].availability"
            )
            if (
                parsed not in positions
                or parsed > cutoff_at
                or available_at > cutoff_at
                or available_at < parsed
            ):
                raise V17PipelineError(f"{label} PIT/open-session binding invalid")
            if require_bool(bar["is_open_day"], label=f"{label}.is_open_day") is not True:
                raise V17PipelineError(f"{label} contains a non-open session")
            require_number(
                bar["close"],
                label=f"{label}[{bar_index}].close",
                minimum=0.0,
                minimum_exclusive=True,
            )
            for field in ("volume", "amount"):
                require_number(
                    bar[field],
                    label=f"{label}[{bar_index}].{field}",
                    minimum=0.0,
                    minimum_exclusive=True,
                )
            bar["trade_date"] = parsed
            bar["availability"] = available_at
            dates.append(parsed)
            normalized.append(bar)
        if len(dates) != len(set(dates)):
            raise V17PipelineError(f"duplicate market date in {label}")
        if dates[-1] != sessions[-1]:
            raise V17PipelineError(f"{label} latest bar is not effective cutoff session")
        expected = list(sessions[positions[dates[0]] : positions[dates[-1]] + 1])
        if dates != expected:
            raise V17PipelineError(f"{label} contains missing or extra open sessions")
        return pd.DataFrame(normalized)

    benchmark_frame = normalized_bars(source["benchmark_bars"], label="H00300 benchmark bars")
    symbols = _validate_ordered_rows(
        source["symbols"],
        label="Market snapshot symbols",
        exact_keys=_MARKET_SYMBOL_KEYS,
        order_key=lambda item: str(item["symbol"]),
    )
    frames: dict[str, pd.DataFrame] = {}
    attributes: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(symbols):
        symbol = require_symbol(item["symbol"], label=f"market symbols[{index}].symbol")
        if symbol in frames:
            raise V17PipelineError(f"duplicate market symbol: {symbol}")
        frame = normalized_bars(item["bars"], label=f"{symbol}.bars")
        if len(frame) < 91:
            raise V17PipelineError(f"{symbol} lacks minimum factor history")
        adv20 = float(pd.to_numeric(frame["amount"], errors="raise").tail(20).mean())
        if not np.isfinite(adv20) or adv20 <= 0.0:
            raise V17PipelineError(f"{symbol} ADV20 cannot be derived")
        frames[symbol] = frame
        attributes[symbol] = {
            "adv20": adv20,
        }
    return frames, attributes, benchmark_frame


def _load_tradability(
    bundle: SourceBindingBundle,
    cutoff: str,
    *,
    symbols: Sequence[str],
    effective_trade_date: pd.Timestamp,
) -> dict[str, bool]:
    source = _validate_source_envelope(
        _read_bound_source(bundle, "tradability_evidence"),
        version=TRADABILITY_SOURCE_VERSION,
        cutoff=cutoff,
        extra_keys=frozenset({"market_snapshot_sha256", "effective_trade_date", "rows"}),
        label="Tradability evidence source",
    )
    if source["market_snapshot_sha256"] != _source_sha(bundle, "market_snapshot"):
        raise V17PipelineError("tradability market snapshot binding mismatch")
    if (
        _as_utc_timestamp(source["effective_trade_date"], label="tradability effective date")
        != effective_trade_date
    ):
        raise V17PipelineError("tradability effective session mismatch")
    rows = _validate_ordered_rows(
        source["rows"],
        label="tradability rows",
        exact_keys=_TRADABILITY_ROW_KEYS,
        order_key=lambda item: str(item["symbol"]),
    )
    cutoff_at = _as_utc_timestamp(cutoff, label="cutoff")
    result: dict[str, bool] = {}
    for index, row in enumerate(rows):
        symbol = require_symbol(row["symbol"], label=f"tradability[{index}].symbol")
        if symbol in result:
            raise V17PipelineError("duplicate tradability symbol")
        if (
            _as_utc_timestamp(row["effective_trade_date"], label="tradability row date")
            != effective_trade_date
        ):
            raise V17PipelineError("tradability row effective session mismatch")
        available = _as_utc_timestamp(row["availability"], label="tradability availability")
        status = row["status"]
        if (
            available < effective_trade_date
            or available > cutoff_at
            or status not in _TRADABILITY_STATUSES
        ):
            raise V17PipelineError("tradability evidence row invalid")
        result[symbol] = status == "TRADABLE"
    if set(result) != set(symbols):
        raise V17PipelineError("tradability evidence must exactly cover sealed symbols")
    return result


def _derive_risk_model_attributes(
    bundle: SourceBindingBundle,
    cutoff: str,
    *,
    frames: Mapping[str, pd.DataFrame],
    benchmark_frame: pd.DataFrame,
    risk_policy: Mapping[str, Any],
) -> dict[str, tuple[float, float]]:
    source = _validate_source_envelope(
        _read_bound_source(bundle, "risk_model_input"),
        version=RISK_MODEL_INPUT_SOURCE_VERSION,
        cutoff=cutoff,
        extra_keys=frozenset(
            {
                "market_snapshot_sha256",
                "benchmark_symbol",
                "algorithm",
                "beta_window",
                "stress_scenario",
                "benchmark_stress_shock",
            }
        ),
        label="Risk-model input source",
    )
    if source["market_snapshot_sha256"] != _source_sha(bundle, "market_snapshot"):
        raise V17PipelineError("risk-model market snapshot binding mismatch")
    if source["benchmark_symbol"] != "H00300.CSI" or source["algorithm"] != "OLS_BETA_V1":
        raise V17PipelineError("risk-model identity mismatch")
    window = source["beta_window"]
    if isinstance(window, bool) or not isinstance(window, int) or not 20 <= window <= 1260:
        raise V17PipelineError("risk-model beta_window invalid")
    if source["stress_scenario"] != risk_policy["stress_scenario"]:
        raise V17PipelineError("risk-model stress scenario does not match risk policy")
    shock = require_number(
        source["benchmark_stress_shock"],
        label="benchmark_stress_shock",
        minimum=-1.0,
        maximum=0.0,
    )
    if shock >= 0.0:
        raise V17PipelineError("benchmark_stress_shock must be negative")
    benchmark = benchmark_frame.set_index("trade_date")["close"].astype(float).pct_change().dropna()
    result: dict[str, tuple[float, float]] = {}
    for symbol, frame in frames.items():
        stock = frame.set_index("trade_date")["close"].astype(float).pct_change().dropna()
        aligned = (
            pd.concat({"stock": stock, "benchmark": benchmark}, axis=1, join="inner")
            .dropna()
            .tail(window)
        )
        if len(aligned) != window:
            raise V17PipelineError(f"risk-model history incomplete: {symbol}")
        variance = float(aligned["benchmark"].var(ddof=1))
        if not np.isfinite(variance) or variance <= 0.0:
            raise V17PipelineError("risk-model benchmark variance invalid")
        beta = float(aligned["stock"].cov(aligned["benchmark"]) / variance)
        stress_loss = max(0.0, -(beta * shock))
        if not np.isfinite(beta) or not np.isfinite(stress_loss) or stress_loss > 1.0:
            raise V17PipelineError(f"risk-model derived value invalid: {symbol}")
        result[symbol] = (beta, stress_loss)
    return result


def _load_cluster_mapping(
    bundle: SourceBindingBundle,
    cutoff: str,
    *,
    symbols: Sequence[str],
) -> dict[str, str]:
    source = _validate_source_envelope(
        _read_bound_source(bundle, "cluster_mapping"),
        version=CLUSTER_MAPPING_SOURCE_VERSION,
        cutoff=cutoff,
        extra_keys=frozenset(
            {"market_snapshot_sha256", "pit_membership_sha256", "generator_id", "rows"}
        ),
        label="Canonical cluster mapping source",
    )
    if source["market_snapshot_sha256"] != _source_sha(bundle, "market_snapshot") or source[
        "pit_membership_sha256"
    ] != _source_sha(bundle, "pit_membership"):
        raise V17PipelineError("cluster mapping canonical input binding mismatch")
    if source["generator_id"] != "V17_CANONICAL_CLUSTER_V1":
        raise V17PipelineError("cluster mapping generator identity mismatch")
    rows = _validate_ordered_rows(
        source["rows"],
        label="cluster mapping rows",
        exact_keys=_CLUSTER_ROW_KEYS,
        order_key=lambda item: str(item["symbol"]),
    )
    cutoff_at = _as_utc_timestamp(cutoff, label="cutoff")
    result: dict[str, str] = {}
    for index, row in enumerate(rows):
        symbol = require_symbol(row["symbol"], label=f"cluster[{index}].symbol")
        if symbol in result:
            raise V17PipelineError("duplicate cluster mapping symbol")
        if _as_utc_timestamp(row["availability"], label="cluster availability") > cutoff_at:
            raise V17PipelineError("cluster mapping contains post-cutoff evidence")
        result[symbol] = require_identifier(row["cluster"], label="cluster")
    if set(result) != set(symbols):
        raise V17PipelineError("cluster mapping must exactly cover sealed symbols")
    return result


def _load_quant_observations(bundle: SourceBindingBundle, cutoff: str) -> pd.DataFrame:
    source = _validate_source_envelope(
        _read_bound_source(bundle, "quant_timing_calibration"),
        version=QUANT_CALIBRATION_SOURCE_VERSION,
        cutoff=cutoff,
        extra_keys=frozenset(
            {
                "market_snapshot_sha256",
                "calendar_sha256",
                "benchmark_total_return_sha256",
                "dividend_total_return_sha256",
                "factor_resource_sha256",
                "factor_implementation_sha256",
                "rows",
            }
        ),
        label="Quant calibration source",
    )
    if source["market_snapshot_sha256"] != _source_sha(bundle, "market_snapshot"):
        raise V17PipelineError("Quant calibration market binding mismatch")
    expected_bindings = {
        "calendar_sha256": _source_sha(bundle, "cn_open_day_calendar"),
        "benchmark_total_return_sha256": _source_sha(bundle, "H00300_total_return"),
        "dividend_total_return_sha256": _source_sha(bundle, "dividend_total_return"),
        "factor_resource_sha256": FACTOR_RESOURCE_SHA256,
        "factor_implementation_sha256": assert_factor_source_binding(),
    }
    for field, expected in expected_bindings.items():
        if source[field] != expected:
            raise V17PipelineError(f"Quant calibration binding mismatch: {field}")
    rows = _validate_ordered_rows(
        source["rows"],
        label="Quant calibration rows",
        exact_keys=_QUANT_CALIBRATION_ROW_KEYS,
        order_key=lambda item: (
            int(item["horizon"]),
            str(item["cross_section_date"]),
            str(item["symbol"]),
        ),
    )
    sessions = _load_open_day_calendar(bundle, cutoff)
    positions = _session_positions(sessions)
    stock_returns = _load_stock_total_return(bundle, cutoff)
    benchmark_returns = _load_benchmark_total_return(bundle, cutoff)
    cutoff_at = _as_utc_timestamp(cutoff, label="cutoff")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, pd.Timestamp, int]] = set()
    factor_names = (
        "pv_blend_volstab19x2_mom90_amihud5_w80",
        "pv_short_reversal_25d",
        "pv_downside_volatility_15d",
    )
    for index, raw in enumerate(rows):
        row = dict(raw)
        symbol = require_symbol(row["symbol"], label=f"quant[{index}].symbol")
        horizon = row["horizon"]
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in {20, 60}:
            raise V17PipelineError("Quant calibration horizon invalid")
        start = _as_utc_timestamp(row["cross_section_date"], label="quant cross section")
        end = _as_utc_timestamp(row["target_end_trade_date"], label="quant target end")
        if start not in positions or end not in positions:
            raise V17PipelineError("Quant interval is not on canonical open-day calendar")
        start_position = positions[start]
        if start_position + horizon >= len(sessions) or sessions[start_position + horizon] != end:
            raise V17PipelineError("Quant interval does not span exact open sessions")
        if _as_utc_timestamp(row["target_start_trade_date"], label="quant target start") != start:
            raise V17PipelineError("Quant target start mismatch")
        age = len(sessions) - 1 - start_position
        available = _as_utc_timestamp(row["availability"], label="quant availability")
        predictor_available = _as_utc_timestamp(
            row["predictor_available_at"], label="quant predictor availability"
        )
        label_available = _as_utc_timestamp(
            row["label_available_at"], label="quant label availability"
        )
        stock = stock_returns.get((symbol, start, end))
        benchmark = benchmark_returns.get((start, end))
        if stock is None or benchmark is None:
            raise V17PipelineError("Quant raw total-return interval missing")
        raw_available = max(stock["availability"], benchmark["availability"])
        stock_return = float(stock["return"])
        benchmark_return = float(benchmark["return"])
        excess = stock_return - benchmark_return
        for factor in factor_names:
            require_number(row[factor], label=f"quant.{factor}")
        checks = (
            predictor_available <= start,
            label_available >= end,
            label_available == available,
            available >= raw_available,
            available <= cutoff_at,
            row["age_open_days"] == age,
            row["realized_open_days"] == horizon,
            row["is_mature"] is True,
            row["is_pit"] is True,
            row["target_definition"] == "EXCESS_RETURN_GT_ZERO",
            _same_number(row["stock_total_return"], stock_return),
            _same_number(row["benchmark_total_return"], benchmark_return),
            _same_number(row["excess_return"], excess),
        )
        if not all(checks):
            raise V17PipelineError("Quant derived field disagrees with raw evidence")
        key = (symbol, start, horizon)
        if key in seen:
            raise V17PipelineError("duplicate Quant calibration observation")
        seen.add(key)
        row["stock_total_return"] = stock_return
        row["benchmark_total_return"] = benchmark_return
        row["excess_return"] = excess
        normalized.append(row)

    frame = pd.DataFrame(normalized)
    frame["score_decile"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    for _, group in frame.groupby(["horizon", "cross_section_date"], sort=True):
        ranks = group[list(factor_names)].rank(method="average", pct=True)
        composite = ranks.mean(axis=1)
        ordered = pd.DataFrame(
            {"composite": composite, "symbol": group["symbol"]}, index=group.index
        ).sort_values(["composite", "symbol"], ascending=[True, True], kind="mergesort")
        count = len(ordered)
        for position, row_index in enumerate(ordered.index):
            frame.at[row_index, "score_decile"] = min(10, (position * 10) // count + 1)
    _replay_quant_calibration_predictors(
        bundle,
        cutoff,
        calibration_rows=rows,
    )
    _validate_calibration_authority(
        bundle,
        cutoff,
        kind="QUANT",
        dataset_role="quant_timing_calibration",
        rows=rows,
    )
    return frame


def _load_deep_evidence(
    bundle: SourceBindingBundle,
    cutoff: str,
    sealed_symbols: Sequence[str],
) -> dict[str, dict[str, Any]]:
    source = _validate_source_envelope(
        _read_bound_source(bundle, "sealed_deep_evidence"),
        version=DEEP_EVIDENCE_SOURCE_VERSION,
        cutoff=cutoff,
        extra_keys=frozenset({"entries"}),
        label="Deep evidence source",
    )
    entries = _validate_ordered_rows(
        source["entries"],
        label="Deep evidence entries",
        exact_keys=_DEEP_EVIDENCE_ENTRY_KEYS,
        order_key=lambda item: (str(item["symbol"]), str(item["evidence_id"])),
    )
    requested = set(sealed_symbols)
    result: dict[str, dict[str, Any]] = {
        symbol: {"evidence_ids": [], "claims": {}} for symbol in sealed_symbols
    }
    seen_ids: set[str] = set()
    cutoff_at = _as_utc_timestamp(cutoff, label="cutoff")
    for index, item in enumerate(entries):
        evidence_id = require_identifier(
            item["evidence_id"], label=f"evidence[{index}].evidence_id"
        )
        symbol = require_symbol(item["symbol"], label=f"evidence[{index}].symbol")
        if evidence_id in seen_ids:
            raise V17PipelineError(f"duplicate deep evidence id: {evidence_id}")
        seen_ids.add(evidence_id)
        kind = require_identifier(item["kind"], label=f"evidence[{index}].kind")
        if kind not in _DEEP_EVIDENCE_KIND_CLAIMS:
            raise V17PipelineError(f"unsupported deep evidence kind: {kind}")
        available = _as_utc_timestamp(item["available_at"], label=f"evidence[{index}].available_at")
        if available > cutoff_at:
            raise V17PipelineError("deep evidence contains post-cutoff content")
        expected_locator = f"sealed_deep_evidence#/entries/{index}/content"
        if item["locator"] != expected_locator:
            raise V17PipelineError(f"deep evidence locator mismatch: {evidence_id}")
        content = require_nonempty_string(
            item["content"], label=f"evidence[{index}].content", max_chars=500_000
        )
        digest = require_sha256(item["located_content_sha256"], label="located content SHA-256")
        if hashlib.sha256(content.encode("utf-8")).hexdigest() != digest:
            raise V17PipelineError(f"deep evidence located content digest mismatch: {evidence_id}")
        if symbol in requested:
            claims = _DEEP_EVIDENCE_KIND_CLAIMS[kind]
            result[symbol]["evidence_ids"].append(evidence_id)
            result[symbol]["claims"][evidence_id] = {
                "kind": kind,
                "layers": sorted(claims["layers"]),
                "coverage": sorted(claims["coverage"]),
                "signals": sorted(claims["signals"]),
                "red_flags": sorted(claims["red_flags"]),
            }
    required = {
        "layers": set(LAYER_NAMES),
        "coverage": set(COVERAGE_SECTIONS),
        "signals": set(SIGNAL_WEIGHTS),
        "red_flags": set(SEVERE_RED_FLAGS),
    }
    normalized: dict[str, dict[str, Any]] = {}
    for symbol in sealed_symbols:
        claims = result[symbol]["claims"]
        unions = {
            category: (
                set().union(*(set(item[category]) for item in claims.values())) if claims else set()
            )
            for category in required
        }
        normalized[symbol] = {
            "evidence_ids": tuple(sorted(result[symbol]["evidence_ids"])),
            "claims": {key: claims[key] for key in sorted(claims)},
            "ready": all(unions[category] >= values for category, values in required.items()),
        }
    return normalized


def _holdings_payload(
    bundle: SourceBindingBundle,
    *,
    cutoff: str,
    strategy_id: str,
) -> dict[str, Any] | None:
    item = bundle.by_role["holdings"]
    if (
        item["availability"] != Availability.AVAILABLE.value
        or bundle.by_role["holdings_pointer"]["availability"] != Availability.AVAILABLE.value
    ):
        return None
    payload = validate_holdings_snapshot(
        _read_bound_source(bundle, "holdings"),
        cutoff=cutoff,
    )
    if payload["strategy_id"] != strategy_id:
        raise V17PipelineError("holdings strategy binding mismatch")
    if payload["availability"] != Availability.AVAILABLE.value:
        return None
    _validate_holdings_authority(bundle, cutoff=cutoff, holdings=payload)
    return payload


def _validate_holdings_authority(
    bundle: SourceBindingBundle,
    *,
    cutoff: str,
    holdings: Mapping[str, Any],
) -> None:
    pointer = _validate_source_envelope(
        _read_bound_source(bundle, "holdings_pointer"),
        version=HOLDINGS_POINTER_VERSION,
        cutoff=cutoff,
        extra_keys=frozenset(
            {
                "generation_id",
                "effective_trade_date",
                "holdings_snapshot_sha256",
                "holdings_snapshot_semantic_sha256",
                "updated_at",
            }
        ),
        label="Holdings pointer",
    )
    require_identifier(pointer["generation_id"], label="holdings generation_id")
    lineage = _load_market_lineage(bundle)
    if pointer["effective_trade_date"] != lineage["effective_trade_date"]:
        raise V17PipelineError("holdings pointer is stale at canonical effective session")
    if pointer["holdings_snapshot_sha256"] != _source_sha(bundle, "holdings"):
        raise V17PipelineError("holdings pointer byte binding mismatch")
    if pointer["holdings_snapshot_semantic_sha256"] != holdings["semantic_sha256"]:
        raise V17PipelineError("holdings pointer semantic binding mismatch")
    as_of = _as_utc_timestamp(holdings["as_of"], label="holdings.as_of")
    if as_of.strftime("%Y-%m-%d") != lineage["effective_trade_date"]:
        raise V17PipelineError("AVAILABLE holdings snapshot is stale")
    updated = _as_utc_timestamp(pointer["updated_at"], label="holdings pointer updated_at")
    if updated < as_of or updated > _as_utc_timestamp(cutoff, label="cutoff"):
        raise V17PipelineError("holdings pointer PIT timing invalid")


def _validate_portfolio_projection_receipt(
    bundle: SourceBindingBundle,
    *,
    cutoff: str,
) -> None:
    receipt = _validate_source_envelope(
        _read_bound_source(bundle, "portfolio_projection_verification_receipt"),
        version=PORTFOLIO_PROJECTION_RECEIPT_VERSION,
        cutoff=cutoff,
        extra_keys=frozenset(
            {
                "verifier_id",
                "verification_status",
                "source_byte_sha256s",
                "source_semantic_sha256s",
                "canonical_bar_projection_replay_passed",
                "cluster_replay_passed",
                "holdings_ledger_replay_passed",
            }
        ),
        label="Portfolio projection verification receipt",
    )
    roles = frozenset(
        {
            "market_pointer",
            "market_snapshot_manifest",
            "market_snapshot",
            "pit_membership",
            "holdings",
            "holdings_pointer",
            "tradability_evidence",
            "risk_model_input",
            "cluster_mapping",
        }
    )
    semantic_roles = roles - {"market_pointer", "market_snapshot_manifest"}
    if (
        receipt["verifier_id"] != "codex_delegated_reviewer"
        or receipt["verification_status"] != "PASSED"
        or not isinstance(receipt["source_byte_sha256s"], Mapping)
        or set(receipt["source_byte_sha256s"]) != roles
        or not isinstance(receipt["source_semantic_sha256s"], Mapping)
        or set(receipt["source_semantic_sha256s"]) != semantic_roles
    ):
        raise V17PipelineError("portfolio projection verification authority missing")
    if any(
        receipt["source_byte_sha256s"][role] != _source_sha(bundle, role) for role in roles
    ) or any(
        receipt["source_semantic_sha256s"][role] != _bound_source_semantic_sha(bundle, role)
        for role in semantic_roles
    ):
        raise V17PipelineError("portfolio projection verification receipt drift")
    for field in (
        "canonical_bar_projection_replay_passed",
        "cluster_replay_passed",
        "holdings_ledger_replay_passed",
    ):
        if receipt[field] is not True:
            raise V17PipelineError(f"portfolio projection verification failed: {field}")


def _score_deciles(scored: pd.DataFrame) -> dict[str, int]:
    result: dict[str, int] = {}
    available = scored.loc[scored["status"] == "AVAILABLE"]
    for _, group in available.groupby("industry", sort=True):
        ordered = group.sort_values(
            ["total_score", "symbol"],
            ascending=[True, True],
            kind="mergesort",
        )
        count = len(ordered)
        for position, (_, row) in enumerate(ordered.iterrows()):
            result[str(row["symbol"])] = min(10, (position * 10) // count + 1)
    return result


def _json_reasons(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def compute_prepare_artifacts(
    bundle: SourceBindingBundle,
    *,
    run_id: str,
    strategy_id: str,
    cutoff: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute Fundamental ranking/calibration and the sealed deep request."""

    _validate_rank_projection_receipt(bundle, cutoff)
    membership = _load_pit_membership(bundle, cutoff)
    snapshot = _load_fundamental_snapshot(bundle, cutoff, membership)
    history = _load_history(bundle, cutoff)
    forward_observations = _load_forward_observations(bundle, cutoff)
    # Validate the deferred Quant inputs now, while deliberately not making a
    # timing decision before the Fundamental/Codex stages are complete.
    _load_market_snapshot(bundle, cutoff)
    _load_quant_observations(bundle, cutoff)

    holdings = _holdings_payload(
        bundle,
        cutoff=cutoff,
        strategy_id=strategy_id,
    )
    held_symbols = (
        tuple(item["symbol"] for item in holdings["positions"]) if holdings is not None else ()
    )
    candidates = score_fundamental_universe(
        snapshot,
        history,
        cutoff=cutoff,
        holdings=held_symbols,
        top_n=24,
    )
    if not candidates.ranked_symbols:
        raise V17PipelineError("Fundamental produced no ranked candidates")
    calibration = calibrate_forward_returns(forward_observations, cutoff=cutoff)
    deciles = _score_deciles(candidates.scored)
    scored_by_symbol = candidates.scored.set_index("symbol", drop=False)
    rows: list[dict[str, Any]] = []
    for symbol in candidates.sealed_symbols:
        row = scored_by_symbol.loc[symbol]
        decile = deciles.get(symbol)
        if decile is None:
            base_q25: dict[int, float] = {}
            base_eligible = False
            base_blockers = ["fundamental_score_unavailable"]
        else:
            eligibility = assess_fundamental_eligibility(
                calibration,
                industry=str(row["industry"]),
                score_decile=decile,
                deep_research_complete=True,
                severe_red_flags=False,
            )
            base_q25 = dict(eligibility.base_q25_by_horizon)
            base_eligible = bool(eligibility.eligible)
            base_blockers = list(eligibility.blockers)
        total_score = row["total_score"]
        rows.append(
            {
                "symbol": symbol,
                "industry": str(row["industry"]),
                "fundamental_status": str(row["status"]),
                "fundamental_score": (float(total_score) if pd.notna(total_score) else None),
                "score_decile": decile,
                "base_q25_by_horizon": {
                    str(horizon): float(base_q25[horizon])
                    for horizon in HORIZONS
                    if horizon in base_q25
                },
                "base_eligible": base_eligible,
                "base_blockers": base_blockers,
                "fundamental_blockers": _json_reasons(row["unavailable_reasons"]),
                "selected_top24": symbol in candidates.ranked_symbols,
                "appended_holding": symbol in candidates.appended_holdings,
            }
        )
    deterministic = seal_semantic(
        {
            "version": DETERMINISTIC_RESULT_VERSION,
            "run_id": run_id,
            "cutoff": cutoff,
            "ranked_symbols": list(candidates.ranked_symbols),
            "sealed_symbols": list(candidates.sealed_symbols),
            "appended_holdings": list(candidates.appended_holdings),
            "portfolio_required_roles": sorted(PORTFOLIO_REQUIRED_ROLES),
            "rows": rows,
            "authority": False,
        }
    )
    evidence = _load_deep_evidence(bundle, cutoff, candidates.sealed_symbols)
    deep_request = seal_semantic(
        {
            "version": DEEP_REQUEST_VERSION,
            "run_id": run_id,
            "cutoff": cutoff,
            "symbols": list(candidates.sealed_symbols),
            "evidence_ids_by_symbol": {
                symbol: list(evidence[symbol]["evidence_ids"])
                for symbol in candidates.sealed_symbols
            },
            "evidence_claims_by_symbol": {
                symbol: evidence[symbol]["claims"] for symbol in candidates.sealed_symbols
            },
            "evidence_readiness_by_symbol": {
                symbol: bool(evidence[symbol]["ready"]) for symbol in candidates.sealed_symbols
            },
            "authority": False,
        }
    )
    return deterministic, deep_request


def _evaluation_payload(
    symbol: str,
    result: DeepResearchEvaluation,
) -> dict[str, Any]:
    payload = asdict(result)
    return {
        "symbol": symbol,
        "status": payload["status"],
        "research_complete": payload["research_complete"],
        "f_eligible": payload["f_eligible"],
        "buy_permission_revoked": payload["buy_permission_revoked"],
        "severe_red_flags": list(payload["severe_red_flags"]),
        "weighted_signal": payload["weighted_signal"],
        "delta": payload["delta"],
        "base_q25_252": payload["base_q25_252"],
        "adjusted_q25_252": payload["adjusted_q25_252"],
        "blockers": list(payload["blockers"]),
    }


def _deep_evidence_taxonomy_blockers(
    research: Mapping[str, Any],
    *,
    claims: Mapping[str, Mapping[str, Any]],
    evidence_ready: bool,
) -> list[str]:
    blockers: list[str] = []
    if not evidence_ready:
        blockers.append("sealed_evidence_taxonomy_incomplete")

    def validate_refs(item: Any, *, category: str, claim: str, path: str) -> None:
        if not isinstance(item, Mapping):
            return
        refs = item.get("evidence_ids")
        if not isinstance(refs, list):
            return
        for evidence_id in refs:
            evidence_claim = claims.get(str(evidence_id))
            if evidence_claim is None or claim not in evidence_claim.get(category, []):
                blockers.append(f"evidence_kind_not_allowed:{path}:{evidence_id}")

    layers = research.get("layers")
    if isinstance(layers, Mapping):
        for layer in LAYER_NAMES:
            items = layers.get(layer)
            if isinstance(items, list):
                for index, item in enumerate(items):
                    validate_refs(
                        item,
                        category="layers",
                        claim=layer,
                        path=f"layers.{layer}[{index}]",
                    )
    coverage = research.get("coverage")
    if isinstance(coverage, Mapping):
        for section in COVERAGE_SECTIONS:
            validate_refs(
                coverage.get(section),
                category="coverage",
                claim=section,
                path=f"coverage.{section}",
            )
    signals = research.get("signals")
    if isinstance(signals, Mapping):
        for dimension in SIGNAL_WEIGHTS:
            validate_refs(
                signals.get(dimension),
                category="signals",
                claim=dimension,
                path=f"signals.{dimension}",
            )
    flags = research.get("severe_red_flags")
    if isinstance(flags, Mapping):
        for flag in SEVERE_RED_FLAGS:
            validate_refs(
                flags.get(flag),
                category="red_flags",
                claim=flag,
                path=f"severe_red_flags.{flag}",
            )
    return list(dict.fromkeys(blockers))


def evaluate_deep_response(
    response: Mapping[str, Any],
    *,
    deterministic: Mapping[str, Any],
    deep_request: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate every COMPLETE item against exact locally derived base values."""

    rows = {str(item["symbol"]): item for item in deterministic["rows"]}
    evaluations: list[dict[str, Any]] = []
    for item in response["review_results"]:
        symbol = str(item["symbol"])
        if item["status"] == "COMPLETE":
            base = rows[symbol]
            q25 = {
                int(horizon): float(value) for horizon, value in base["base_q25_by_horizon"].items()
            }
            evaluated = evaluate_deep_research(
                item["research"],
                sealed_symbol=symbol,
                sealed_evidence_ids=deep_request["evidence_ids_by_symbol"][symbol],
                base_q25_by_horizon=q25,
                base_eligible=bool(base["base_eligible"]),
            )
            payload = _evaluation_payload(symbol, evaluated)
            taxonomy_blockers = _deep_evidence_taxonomy_blockers(
                item["research"],
                claims=deep_request["evidence_claims_by_symbol"][symbol],
                evidence_ready=bool(deep_request["evidence_readiness_by_symbol"][symbol]),
            )
            if taxonomy_blockers:
                payload["status"] = "DEEP_RESEARCH_INVALID"
                payload["research_complete"] = False
                payload["f_eligible"] = False
                payload["adjusted_q25_252"] = None
                payload["blockers"] = list(
                    dict.fromkeys([*payload["blockers"], *taxonomy_blockers])
                )
            evaluations.append(payload)
        else:
            evaluations.append(
                {
                    "symbol": symbol,
                    "status": "DEEP_RESEARCH_UNAVAILABLE",
                    "research_complete": False,
                    "f_eligible": False,
                    "buy_permission_revoked": False,
                    "severe_red_flags": [],
                    "weighted_signal": 0.0,
                    "delta": 0.0,
                    "base_q25_252": rows[symbol]["base_q25_by_horizon"].get("252"),
                    "adjusted_q25_252": None,
                    "blockers": [f"deep_research_unavailable:{item['reason']}"],
                }
            )
    return seal_semantic(
        {
            "version": DEEP_EVALUATION_VERSION,
            "run_id": response["run_id"],
            "cutoff": response["cutoff"],
            "evaluations": evaluations,
            "received_at": response["received_at"],
            "authority": False,
        }
    )


def _validate_regime_mapping(
    bundle: SourceBindingBundle,
    *,
    name: str,
    cutoff: str,
) -> dict[str, dict[str, float]]:
    role = f"{name}_overlay_mapping"
    source = _validate_source_envelope(
        _read_bound_source(bundle, role),
        version=REGIME_MAPPING_SOURCE_VERSION,
        cutoff=cutoff,
        extra_keys=frozenset({"name", "states"}),
        label=f"{name} regime mapping",
    )
    if source["name"] != name or not isinstance(source["states"], Mapping) or not source["states"]:
        raise V17PipelineError(f"{name} regime mapping shape invalid")
    states: dict[str, dict[str, float]] = {}
    for state, item in sorted(source["states"].items()):
        canonical_state = require_identifier(state, label=f"{name} mapping state")
        if not isinstance(item, Mapping):
            raise V17PipelineError(f"{name} mapping cell must be an object")
        require_exact_keys(item, _REGIME_MAPPING_CELL_KEYS, label=f"{name}.{state}")
        states[canonical_state] = {
            "gross_cap": require_ratio(item["gross_cap"], label=f"{name}.{state}.gross_cap"),
            "cash_floor": require_ratio(item["cash_floor"], label=f"{name}.{state}.cash_floor"),
        }
    return states


def _mapped_regime_input(
    bundle: SourceBindingBundle,
    *,
    name: str,
    cutoff: str,
) -> dict[str, Any]:
    role = f"{name}_overlay_input"
    source = _validate_source_envelope(
        _read_bound_source(bundle, role),
        version=REGIME_INPUT_SOURCE_VERSION,
        cutoff=cutoff,
        extra_keys=frozenset({"name", "enabled", "availability", "state", "reason"}),
        label=f"{name} regime input",
    )
    if source["name"] != name:
        raise V17PipelineError(f"{name} regime input name mismatch")
    enabled = require_bool(source["enabled"], label=f"{name}.enabled")
    availability = source["availability"]
    state = source["state"]
    reason = source["reason"]
    if not enabled:
        if availability is not None or state is not None or reason is not None:
            raise V17PipelineError(f"disabled {name} input must not carry values")
        return build_disabled_overlay_input(name=name)
    if availability not in {Availability.AVAILABLE.value, Availability.UNAVAILABLE.value}:
        raise V17PipelineError(f"{name} availability invalid")
    if availability == Availability.UNAVAILABLE.value:
        if state is not None:
            raise V17PipelineError(f"unavailable {name} input cannot carry state")
        return build_unavailable_overlay_input(
            name=name,
            reason=require_nonempty_string(reason, label=f"{name}.reason"),
        )
    if reason is not None:
        raise V17PipelineError(f"available {name} input cannot carry reason")
    selected = require_identifier(state, label=f"{name}.state")
    mapping = _validate_regime_mapping(bundle, name=name, cutoff=cutoff)
    if selected not in mapping:
        raise V17PipelineError(f"{name} state missing from sealed mapping")
    return build_available_overlay_input(name=name, **mapping[selected])


def _rank_output(
    deterministic: Mapping[str, Any],
    deep_evaluation: Mapping[str, Any],
    timing: pd.DataFrame,
) -> dict[str, Any]:
    deep_by_symbol = {str(item["symbol"]): item for item in deep_evaluation["evaluations"]}
    timing_by_symbol = timing.set_index("symbol", drop=False)
    rows: list[dict[str, Any]] = []
    for base in deterministic["rows"]:
        symbol = str(base["symbol"])
        deep = deep_by_symbol[symbol]
        timed = timing_by_symbol.loc[symbol]
        score_decile = timed["score_decile"]
        rows.append(
            {
                "symbol": symbol,
                "fundamental_score": base["fundamental_score"],
                "fundamental_score_decile": base["score_decile"],
                "base_q25_by_horizon": dict(base["base_q25_by_horizon"]),
                "deep_status": deep["status"],
                "f_eligible": bool(deep["f_eligible"]),
                "severe_red_flags": list(deep["severe_red_flags"]),
                "adjusted_q25_252": deep["adjusted_q25_252"],
                "quant_score_decile": (int(score_decile) if not pd.isna(score_decile) else None),
                "probability_20d": (
                    float(timed["probability_20d"])
                    if np.isfinite(timed["probability_20d"])
                    else None
                ),
                "probability_60d": (
                    float(timed["probability_60d"])
                    if np.isfinite(timed["probability_60d"])
                    else None
                ),
                "timing_state": str(timed["timing_state"]),
                "selected_top24": bool(base["selected_top24"]),
                "appended_holding": bool(base["appended_holding"]),
            }
        )
    eligible_ranked = [
        symbol for symbol in deterministic["ranked_symbols"] if deep_by_symbol[symbol]["f_eligible"]
    ]
    return seal_semantic(
        {
            "version": RANK_OUTPUT_VERSION,
            "initial_ranked_symbols": list(deterministic["ranked_symbols"]),
            "eligible_ranked_symbols": eligible_ranked,
            "sealed_symbols": list(deterministic["sealed_symbols"]),
            "rows": rows,
            "authority": False,
        }
    )


def _permission_actions(permission: Mapping[str, Any]) -> frozenset[str]:
    actions = {"LOCK"}
    if permission["can_buy"]:
        actions.add("BUY")
    if permission["can_sell"]:
        actions.add("SELL")
    return frozenset(actions)


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _current_weights(holdings: Mapping[str, Any]) -> dict[str, float]:
    nav = _decimal(holdings["nav"])
    return {
        str(item["symbol"]): float(_decimal(item["market_value"]) / nav)
        for item in holdings["positions"]
    }


def _candidate_metrics(
    target: Mapping[str, float],
    market: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, float], dict[str, float], float, float]:
    industries: dict[str, float] = {}
    clusters: dict[str, float] = {}
    beta = 0.0
    stress = 0.0
    for symbol, weight in target.items():
        attributes = market[symbol]
        industries[attributes["industry"]] = industries.get(attributes["industry"], 0.0) + weight
        clusters[attributes["cluster"]] = clusters.get(attributes["cluster"], 0.0) + weight
        beta += weight * float(attributes["beta"])
        stress += weight * float(attributes["stress_loss"])
    return industries, clusters, beta, stress


def _normalize_target_weights(
    value: Any,
    *,
    sealed_symbols: frozenset[str],
) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise V17PipelineError("candidate target_weights must be an object")
    target: dict[str, float] = {}
    for raw_symbol, raw_weight in sorted(value.items()):
        symbol = require_symbol(raw_symbol, label="target weight symbol")
        if symbol not in sealed_symbols:
            raise V17PipelineError(f"candidate expanded sealed universe: {symbol}")
        weight = require_ratio(raw_weight, label=f"target_weights.{symbol}")
        if symbol in target:
            raise V17PipelineError(f"duplicate target symbol: {symbol}")
        target[symbol] = weight
    return target


def compute_finalization(
    bundle: SourceBindingBundle,
    *,
    deterministic: Mapping[str, Any],
    deep_evaluation: Mapping[str, Any],
    candidate_proposals: Sequence[Mapping[str, Any]],
    strategy_id: str,
    cutoff: str,
) -> tuple[str, dict[str, Any], dict[str, Any] | None, list[str], dict[str, Any]]:
    """Recompute timing, overlay, permissions, pretrade, and optimizer output."""

    sealed_symbols = tuple(str(item) for item in deterministic["sealed_symbols"])
    frames, market, benchmark_frame = _load_market_snapshot(bundle, cutoff)
    missing_market = sorted(set(sealed_symbols) - set(frames))
    if missing_market:
        raise V17PipelineError(f"sealed symbols missing market inputs: {missing_market}")
    latest_scores = compute_latest_scores(
        frames,
        sealed_symbols=sealed_symbols,
        cutoff=cutoff,
    )
    timing_calibration = calibrate_timing_probabilities(
        _load_quant_observations(bundle, cutoff),
        cutoff=cutoff,
    )
    timing = decide_timing(latest_scores, timing_calibration)
    rank_output = _rank_output(deterministic, deep_evaluation, timing)
    quant_blockers = list(timing_calibration.blockers)
    unready = timing.loc[timing["timing_state"] == "UNREADY", "symbol"].tolist()
    if unready:
        quant_blockers.extend(f"quant_unready:{symbol}" for symbol in unready)
    if quant_blockers:
        computation = seal_semantic(
            {
                "version": PORTFOLIO_COMPUTATION_VERSION,
                "status": "NO_PORTFOLIO",
                "permissions": [],
                "candidate_results": [],
                "optimizer_rejections": {},
                "blockers": list(dict.fromkeys(quant_blockers)),
                "authority": False,
            }
        )
        return "NO_PORTFOLIO", rank_output, None, list(dict.fromkeys(quant_blockers)), computation

    source_blockers = [f"source_unavailable:{role}" for role in bundle.portfolio_unavailable_roles]
    if source_blockers:
        computation = seal_semantic(
            {
                "version": PORTFOLIO_COMPUTATION_VERSION,
                "status": "NO_PORTFOLIO",
                "permissions": [],
                "candidate_results": [],
                "optimizer_rejections": {},
                "blockers": source_blockers,
                "authority": False,
            }
        )
        return "NO_PORTFOLIO", rank_output, None, source_blockers, computation

    _validate_portfolio_projection_receipt(bundle, cutoff=cutoff)
    holdings = validate_holdings_snapshot(_read_bound_source(bundle, "holdings"), cutoff=cutoff)
    risk = validate_portfolio_risk_policy_snapshot(
        _read_bound_source(bundle, "risk_policy"), cutoff=cutoff
    )
    cost_policy = validate_execution_cost_policy(
        _read_bound_source(bundle, "execution_cost_policy")
    )
    if holdings["strategy_id"] != strategy_id or risk["strategy_id"] != strategy_id:
        raise V17PipelineError("portfolio source strategy binding mismatch")
    if (
        holdings["availability"] != Availability.AVAILABLE.value
        or risk["availability"] != Availability.AVAILABLE.value
    ):
        raise V17PipelineError("portfolio sources changed availability after manifest validation")
    _validate_holdings_authority(bundle, cutoff=cutoff, holdings=holdings)
    effective_trade_date = next(iter(frames.values())).iloc[-1]["trade_date"]
    tradability = _load_tradability(
        bundle,
        cutoff,
        symbols=sealed_symbols,
        effective_trade_date=effective_trade_date,
    )
    risk_attributes = _derive_risk_model_attributes(
        bundle,
        cutoff,
        frames={symbol: frames[symbol] for symbol in sealed_symbols},
        benchmark_frame=benchmark_frame,
        risk_policy=risk,
    )
    clusters = _load_cluster_mapping(
        bundle,
        cutoff,
        symbols=sealed_symbols,
    )
    industry_by_symbol = {
        str(item["symbol"]): str(item["industry"]) for item in deterministic["rows"]
    }
    for symbol in sealed_symbols:
        beta, stress_loss = risk_attributes[symbol]
        market[symbol].update(
            {
                "industry": industry_by_symbol[symbol],
                "cluster": clusters[symbol],
                "tradable": tradability[symbol],
                "beta": beta,
                "stress_loss": stress_loss,
            }
        )
    base = build_available_overlay_input(
        name="base",
        gross_cap=float(risk["gross_cap"]),
        cash_floor=float(risk["cash_floor"]),
    )
    macro = _mapped_regime_input(bundle, name="macro", cutoff=cutoff)
    markov = _mapped_regime_input(bundle, name="markov", cutoff=cutoff)
    overlay = compute_regime_portfolio_overlay(base=base, macro=macro, markov=markov)
    if overlay["availability"] != Availability.AVAILABLE.value:
        blockers = ["regime_overlay_unavailable"]
        computation = seal_semantic(
            {
                "version": PORTFOLIO_COMPUTATION_VERSION,
                "status": "NO_PORTFOLIO",
                "permissions": [],
                "candidate_results": [],
                "optimizer_rejections": {},
                "blockers": blockers,
                "authority": False,
            }
        )
        return "NO_PORTFOLIO", rank_output, None, blockers, computation

    current = _current_weights(holdings)
    deep_by_symbol = {str(item["symbol"]): item for item in deep_evaluation["evaluations"]}
    timing_by_symbol = timing.set_index("symbol", drop=False)
    permissions: dict[str, dict[str, Any]] = {}
    permission_mask: dict[str, frozenset[str]] = {}
    for symbol in sealed_symbols:
        deep = deep_by_symbol[symbol]
        permission = determine_trade_permission(
            symbol=symbol,
            held=symbol in current,
            tradable=bool(market[symbol]["tradable"]),
            fundamental_eligibility=("F_ELIGIBLE" if deep["f_eligible"] else "F_INELIGIBLE"),
            severe_red_flag=bool(deep["severe_red_flags"]),
            quant_timing=str(timing_by_symbol.loc[symbol]["timing_state"]),
        )
        permissions[symbol] = permission
        permission_mask[symbol] = _permission_actions(permission)

    nav_decimal = _decimal(holdings["nav"])
    feasible: list[FeasiblePortfolio] = []
    candidate_results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(candidate_proposals):
        if not isinstance(raw, Mapping):
            raise V17PipelineError(f"candidate_proposals[{index}] must be an object")
        require_exact_keys(
            raw,
            frozenset({"candidate_id", "target_weights"}),
            label=f"candidate_proposals[{index}]",
        )
        candidate_id = require_identifier(raw["candidate_id"], label="candidate_id")
        if candidate_id in seen_ids:
            raise V17PipelineError(f"duplicate candidate proposal: {candidate_id}")
        seen_ids.add(candidate_id)
        target = _normalize_target_weights(
            raw["target_weights"], sealed_symbols=frozenset(sealed_symbols)
        )
        deltas = {
            symbol: _decimal(target.get(symbol, 0.0)) - _decimal(current.get(symbol, 0.0))
            for symbol in set(target) | set(current)
        }
        turnover_decimal = sum(
            (abs(delta) for delta in deltas.values()),
            start=Decimal("0"),
        )
        turnover = float(turnover_decimal)
        candidate_industry_weights, candidate_cluster_weights, portfolio_beta, portfolio_stress = (
            _candidate_metrics(target, market)
        )
        aggregate_blockers: list[str] = []
        if sum(target.values()) > 1.0:
            aggregate_blockers.append("target_gross_above_one")
        if turnover > 2.0:
            aggregate_blockers.append("turnover_above_two")
        if any(value > 1.0 for value in candidate_industry_weights.values()):
            aggregate_blockers.append("industry_aggregate_above_one")
        if any(value > 1.0 for value in candidate_cluster_weights.values()):
            aggregate_blockers.append("cluster_aggregate_above_one")
        if portfolio_stress > 1.0:
            aggregate_blockers.append("stress_aggregate_above_one")
        if sum(target.values()) > float(overlay["effective_gross"]):
            aggregate_blockers.append("effective_gross_exceeded")
        if any(weight > float(risk["single_name_cap"]) for weight in target.values()):
            aggregate_blockers.append("single_name_cap_exceeded")
        if any(
            value > float(risk["industry_cap"]) for value in candidate_industry_weights.values()
        ):
            aggregate_blockers.append("industry_cap_exceeded")
        if any(value > float(risk["cluster_cap"]) for value in candidate_cluster_weights.values()):
            aggregate_blockers.append("cluster_cap_exceeded")
        if abs(portfolio_beta) > float(risk["beta_cap"]):
            aggregate_blockers.append("beta_cap_exceeded")
        if portfolio_stress > float(risk["stress_loss_cap"]):
            aggregate_blockers.append("stress_cap_exceeded")
        if turnover > float(risk["turnover_cap"]):
            aggregate_blockers.append("turnover_cap_exceeded")
        if aggregate_blockers:
            expected_q25 = sum(
                weight * float(deep_by_symbol[symbol]["adjusted_q25_252"])
                for symbol, weight in target.items()
                if deep_by_symbol[symbol]["adjusted_q25_252"] is not None
            )
            candidate_results.append(
                {
                    "candidate_id": candidate_id,
                    "target_weights": target,
                    "turnover": turnover,
                    "expected_adjusted_q25": expected_q25,
                    "transaction_cost_fraction": 0.0,
                    "pretrade_results": [],
                    "blockers": aggregate_blockers,
                }
            )
            continue
        trades: list[ProposedTrade] = []
        pretrade_results: list[dict[str, Any]] = []
        candidate_blockers: list[str] = []
        total_cost_amount = Decimal("0")
        for symbol in sorted(set(target) | set(current)):
            delta = deltas[symbol]
            if delta == 0:
                continue
            action = "BUY" if delta > 0 else "SELL"
            trade = ProposedTrade(
                symbol=symbol,
                action=action,
                notional_fraction=float(abs(delta)),
            )
            trades.append(trade)
            attributes = market[symbol]
            proposal = {
                "symbol": symbol,
                "side": action,
                "trade_notional": float(abs(delta) * nav_decimal),
                "adv20": float(attributes["adv20"]),
                "position_weight_after": target.get(symbol, 0.0),
                "industry_weight_after": candidate_industry_weights.get(
                    attributes["industry"], 0.0
                ),
                "cluster_weight_after": candidate_cluster_weights.get(attributes["cluster"], 0.0),
                "beta_after": portfolio_beta,
                "stress_loss_after": portfolio_stress,
                "turnover_after": turnover,
            }
            pretrade = evaluate_pretrade(
                proposal,
                permission=permissions[symbol],
                risk_policy=risk,
                cost_policy=cost_policy,
                cutoff=cutoff,
            )
            pretrade = validate_pretrade_result(
                pretrade,
                proposal=proposal,
                permission=permissions[symbol],
                risk_policy=risk,
                cost_policy=cost_policy,
                cutoff=cutoff,
            )
            pretrade_results.append(pretrade)
            total_cost_amount += _decimal(pretrade["cost"]["amount"])
            if not pretrade["passed"]:
                failed = [item["name"] for item in pretrade["checks"] if not item["passed"]]
                candidate_blockers.append(f"pretrade_failed:{symbol}:{','.join(failed)}")
        expected_q25 = 0.0
        for symbol, weight in target.items():
            adjusted = deep_by_symbol[symbol]["adjusted_q25_252"]
            if adjusted is not None:
                expected_q25 += weight * float(adjusted)
        candidate_results.append(
            {
                "candidate_id": candidate_id,
                "target_weights": target,
                "turnover": turnover,
                "expected_adjusted_q25": expected_q25,
                "transaction_cost_fraction": float(total_cost_amount / nav_decimal),
                "pretrade_results": pretrade_results,
                "blockers": candidate_blockers,
            }
        )
        if candidate_blockers:
            continue
        feasible.append(
            FeasiblePortfolio(
                candidate_id=candidate_id,
                target_weights=target,
                trades=tuple(trades),
                expected_adjusted_q25=expected_q25,
                transaction_cost=float(total_cost_amount / nav_decimal),
                turnover=turnover,
            )
        )

    optimized = optimize_lexicographic(
        feasible,
        permission_mask=permission_mask,
        current_weights=current,
        effective_gross=float(overlay["effective_gross"]),
    )
    optimizer_rejections = {
        candidate_id: list(reasons) for candidate_id, reasons in sorted(optimized.rejected.items())
    }
    for item in candidate_results:
        if item["blockers"]:
            optimizer_rejections.setdefault(item["candidate_id"], list(item["blockers"]))
    if optimized.selected is None:
        blockers = ["no_feasible_portfolio_candidate"]
        computation = seal_semantic(
            {
                "version": PORTFOLIO_COMPUTATION_VERSION,
                "status": "INFEASIBLE",
                "permissions": [permissions[symbol] for symbol in sealed_symbols],
                "candidate_results": candidate_results,
                "optimizer_rejections": optimizer_rejections,
                "blockers": blockers,
                "authority": False,
            }
        )
        return "INFEASIBLE", rank_output, None, blockers, computation

    selected = optimized.selected
    selected_result = next(
        item for item in candidate_results if item["candidate_id"] == selected.candidate_id
    )
    portfolio_output = seal_semantic(
        {
            "version": PORTFOLIO_OUTPUT_VERSION,
            "candidate_id": selected.candidate_id,
            "target_weights": {
                symbol: float(weight) for symbol, weight in sorted(selected.target_weights.items())
            },
            "shadow_trade_deltas": [
                {
                    "symbol": trade.symbol,
                    "action": trade.action,
                    "notional_fraction": trade.notional_fraction,
                }
                for trade in selected.trades
            ],
            "expected_adjusted_q25": selected.expected_adjusted_q25,
            "transaction_cost_fraction": selected.transaction_cost,
            "net_adjusted_q25": selected.net_adjusted_q25,
            "turnover": selected.turnover,
            "regime_overlay": overlay,
            "permissions": [permissions[symbol] for symbol in sealed_symbols],
            "pretrade_results": selected_result["pretrade_results"],
            "optimizer_rejections": optimizer_rejections,
            "authority": False,
        }
    )
    computation = seal_semantic(
        {
            "version": PORTFOLIO_COMPUTATION_VERSION,
            "status": "COMPLETE",
            "permissions": [permissions[symbol] for symbol in sealed_symbols],
            "candidate_results": candidate_results,
            "optimizer_rejections": optimizer_rejections,
            "blockers": [],
            "authority": False,
        }
    )
    return "COMPLETE", rank_output, portfolio_output, [], computation


__all__ = [
    "DEEP_EVALUATION_VERSION",
    "DEEP_EVIDENCE_SOURCE_VERSION",
    "DEEP_REQUEST_VERSION",
    "DETERMINISTIC_RESULT_VERSION",
    "FORWARD_CALIBRATION_SOURCE_VERSION",
    "FUNDAMENTAL_HISTORY_SOURCE_VERSION",
    "FUNDAMENTAL_SNAPSHOT_SOURCE_VERSION",
    "MARKET_SNAPSHOT_SOURCE_VERSION",
    "PIT_MEMBERSHIP_SOURCE_VERSION",
    "PORTFOLIO_COMPUTATION_VERSION",
    "PORTFOLIO_OUTPUT_VERSION",
    "QUANT_CALIBRATION_SOURCE_VERSION",
    "RANK_OUTPUT_VERSION",
    "REGIME_INPUT_SOURCE_VERSION",
    "REGIME_MAPPING_SOURCE_VERSION",
    "V17PipelineError",
    "compute_finalization",
    "compute_prepare_artifacts",
    "evaluate_deep_response",
]
