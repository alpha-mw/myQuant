"""
A股激进科技制造策略正式复盘跟踪器。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import importlib.util
import json
import math
import shutil
import stat
import time
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import requests

from quant_investor.factors.report import (
    load_factor_library_shadow_status,
    render_factor_library_shadow_markdown,
)
from quant_investor.agents.risk_guard import risk_guard_single_name_weight_cap
from quant_investor.market.config import get_market_settings
from quant_investor.market.dag_executor import execute_market_dag
from quant_investor.market.download_cn import CNFullMarketDownloader
from quant_investor.market.market_data_reader import MarketDataReader
from quant_investor.monitoring import cn_aggressive_market_metrics as _market_metrics
from quant_investor.llm_policy import llm_handoff_reason
from quant_investor.llm_provider_priority import coerce_review_model_priority
from quant_investor.pipeline import QuantInvestor
from quant_investor.reporting.formal_diagnostics import (
    HoldingDecisionDiagnostic,
    apply_report_decision_guardrail,
    build_holding_decision_diagnostics,
    collect_formal_report_warnings,
    is_previous_day_realtime_decision_sufficient,
    render_holding_diagnostic_markdown_table,
)
from quant_investor.research_run_config import ResolvedReviewModels


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_DIR = (
    PROJECT_ROOT / "results" / "strategy_records" / "CN" / "aggressive_tech_manufacturing"
)
DEFAULT_NOTES_PATH = DEFAULT_BASE_DIR / "latest_notes_payload.md"
DEFAULT_DECISION_LOG_PATH = PROJECT_ROOT / "results" / "decision_log" / "decision_log.jsonl"
DEFAULT_INITIAL_CAPITAL = 1_000_000.0
MAX_EFFECTIVE_HOLDING_COUNT = 8
INVALID_MANUAL_LEDGER_STATUS_MARKERS = ("invalidated_price_basis_no_execution",)
VALID_MANUAL_LEDGER_STATUS_PREFIXES = (
    "filled_",
    "no_action_",
    "carry_forward_",
    "prepare_switch_",
    "rejected_",
    "no_fill_",
    "no_fills_",
    "advisory_only_",
)
TRAILING_TAKE_PROFIT_REVIEW_GIVEBACK_RATIO = 0.20
TRAILING_TAKE_PROFIT_REDUCE_GIVEBACK_RATIO = 0.35
TRAILING_TAKE_PROFIT_ENTRY_DATE_COLUMNS = (
    "manual_entry_trade_date",
    "entry_trade_date",
    "entry_date",
    "buy_trade_date",
    "buy_date",
    "first_buy_date",
    "open_date",
)
TRAILING_TAKE_PROFIT_PEAK_PRICE_COLUMNS = (
    "trailing_profit_peak_price",
    "profit_peak_price",
    "peak_price",
)
TRAILING_TAKE_PROFIT_PEAK_DATE_COLUMNS = (
    "trailing_profit_peak_trade_date",
    "profit_peak_trade_date",
    "peak_trade_date",
)
TRAILING_TAKE_PROFIT_LEDGER_COLUMNS = (
    "manual_entry_trade_date",
    "trailing_profit_tracking_start_date",
    "trailing_take_profit_status",
    "trailing_take_profit_confirmed",
    "trailing_take_profit_basis",
    "trailing_profit_peak_price",
    "trailing_profit_peak_trade_date",
    "peak_unrealized_profit",
    "current_unrealized_profit",
    "profit_giveback_ratio",
    "trailing_profit_review_price",
    "trailing_profit_reduce_price",
    "trailing_stop_price",
    "trailing_take_profit_reason",
)

CANDIDATE_POOL_COLUMNS = (
    "symbol",
    "name",
    "category",
    "theme_label",
    "candidate_source",
    "candidate_rank",
    "candidate_dag_four_branch_complete",
    "present_branches",
    "missing_branches",
    "evidence_quality",
    "bayesian_rank",
    "posterior_action_score",
    "posterior_win_rate",
    "posterior_expected_alpha",
    "posterior_confidence",
    "rank_score",
    "shortlist_action",
    "shortlist_confidence",
    "expected_upside",
    "suggested_weight",
    "portfolio_target_weight",
    "portfolio_target_position",
    "risk_flags",
    "rationale",
    "branch_quant_score",
    "branch_fundamental_score",
    "branch_intelligence_score",
    "branch_macro_score",
    "candidate_quote_status",
    "candidate_quote_timestamp",
    "candidate_quote_price",
    "new_position_eligible",
    "eligibility_blockers",
    "manual_override_required",
    "actionable",
    "codex_recommendation_score",
    "codex_recommendation_rating",
)

SWITCH_PLAN_COLUMNS = (
    "sell_symbol",
    "sell_name",
    "sell_role",
    "sell_recommended_action",
    "buy_symbol",
    "buy_name",
    "buy_theme",
    "buy_bayesian_rank",
    "buy_posterior_action_score",
    "buy_posterior_confidence",
    "buy_posterior_expected_alpha",
    "buy_portfolio_target_weight",
    "candidate_quote_status",
    "new_position_eligible",
    "manual_override_required",
    "actionable",
    "candidate_source",
    "evidence_quality",
    "priority",
    "action",
    "switch_ratio_hint",
    "trigger_threshold",
    "no_switch_condition",
    "buy_codex_recommendation_score",
    "buy_codex_recommendation_rating",
)
QUOTE_TIMEOUT = 20
DEFAULT_QUOTE_MAX_AGE_SECONDS = 300
QUOTE_INPUT_SCHEMA_VERSION = "cn_aggressive_quote_input.v1"
MANUAL_EXECUTION_SCHEMA_VERSION = "cn_aggressive_manual_execution.v3"
REALTIME_EXECUTION_PRICE_FIELDS = (
    "current",
    "realtime_price",
    "last",
    "last_price",
    "trade_price",
    "bid",
    "bid_price",
    "ask",
    "ask_price",
    "price",
)
INDEX_QUOTES = {
    "sh000001": "上证指数",
    "sz399001": "深证成指",
    "sz399006": "创业板指",
    "sh000300": "沪深300",
    "sz399905": "中证500",
    "sz399852": "中证1000",
    "sh000688": "科创50",
    "sz399673": "创业板50",
}
THEME_BASKETS = {
    "先进材料": ["688519.SH", "688295.SH"],
    "AI存储": ["688525.SH"],
    "电子制造": ["002384.SZ", "002008.SZ"],
    "电力设备": ["601179.SH", "600487.SH"],
    "光通信": ["601869.SH"],
}
CATEGORY_THEME_LABELS = {
    "hs300": "大盘核心资产",
    "zz500": "中盘制造主线",
    "zz1000": "小盘成长弹性",
    "full_a": "全市场 v13 DAG",
}
REQUIRED_DAG_BRANCHES = ("quant", "fundamental", "intelligence", "macro")
CANDIDATE_DAG_TOP_K = 12
CANDIDATE_DECAY_WATERFALL_SCHEMA_VERSION = "cn_aggressive_candidate_decay_waterfall.v1"
CANDIDATE_DECAY_SINGLE_REASON_THRESHOLD = 0.95
THEME_POOL_AUDIT_SCHEMA_VERSION = "cn_aggressive_theme_pool_audit.v1"
THEME_POOL_AUDIT_SYMBOL_SAMPLE_LIMIT = 50
MARKET_METRICS_CACHE_SCHEMA_VERSION = _market_metrics.MARKET_METRICS_CACHE_SCHEMA_VERSION
MARKET_METRICS_COMPONENT_KEYS = _market_metrics.MARKET_METRICS_COMPONENT_KEYS
MARKET_METRICS_CATEGORIES = _market_metrics.MARKET_METRICS_CATEGORIES
MARKET_METRICS_OUTPUT_COLUMNS = _market_metrics.MARKET_METRICS_OUTPUT_COLUMNS
MARKET_METRICS_REQUIRED_COLUMNS = _market_metrics.MARKET_METRICS_REQUIRED_COLUMNS
MarketMetricsBundle = _market_metrics.MarketMetricsBundle
_compute_category_breadth = _market_metrics._compute_category_breadth
_compute_full_market_metrics = _market_metrics._compute_full_market_metrics
_compute_market_metrics_and_breadth = _market_metrics._compute_market_metrics_and_breadth
_components_fingerprint = _market_metrics._components_fingerprint
_derive_stage_levels = _market_metrics._derive_stage_levels
_load_cached_market_metrics_bundle = _market_metrics._load_cached_market_metrics_bundle
_load_history_frame = _market_metrics._load_history_frame
_load_or_compute_market_metrics_bundle = (
    _market_metrics.load_or_compute_market_metrics_bundle
)
_market_metrics_cache_dir = _market_metrics._market_metrics_cache_dir
_metric_return = _market_metrics._metric_return
_normalize_market_metrics_frame = _market_metrics._normalize_market_metrics_frame
_price_series = _market_metrics._price_series
_read_frame_from_result = _market_metrics._read_frame_from_result
_reader_snapshot_payload = _market_metrics._reader_snapshot_payload
_score_full_market_metrics = _market_metrics._score_full_market_metrics
_validate_market_metrics_frame = _market_metrics._validate_market_metrics_frame
_write_market_metrics_cache = _market_metrics._write_market_metrics_cache


@dataclass
class ProposedOrder:
    symbol: str
    action: str
    shares: int
    price: float
    trade_value: float
    realized_pnl: float
    reason: str


def _load_daily_config_llm_settings() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "daily_config.py"
    if not config_path.exists():
        return {
            "review_model_priority": coerce_review_model_priority([]),
            "agent_model": "",
            "agent_fallback_model": "",
            "master_model": "",
            "master_fallback_model": "",
            "master_reasoning_effort": "",
        }

    spec = importlib.util.spec_from_file_location("_daily_cfg_for_tracker", config_path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    cfg: dict[str, Any] = dict(getattr(module, "DAILY_CONFIG", {}) or {})
    resolved = ResolvedReviewModels.from_mapping(cfg)
    payload = resolved.to_runtime_kwargs()
    payload["enable_agent_layer"] = bool(cfg.get("enable_agent_layer", True))
    return payload


def _plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        payload = value.to_dict()
        if isinstance(payload, dict):
            return dict(_jsonable(payload))
    if isinstance(value, dict):
        return dict(_jsonable(value))
    data = getattr(value, "__dict__", None)
    if isinstance(data, dict):
        return dict(_jsonable(data))
    return {}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if hasattr(value, "to_dict") and not isinstance(value, pd.DataFrame):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _serialize_reviewed_branch_verdicts(result: Any, symbol: str) -> dict[str, Any]:
    payload = dict(getattr(result, "reviewed_research_by_symbol", {}) or {}).get(symbol, {})
    reviewed = (
        {str(name): _jsonable(verdict) for name, verdict in payload.items()}
        if isinstance(payload, dict)
        else {}
    )
    branch_summaries = dict(getattr(result, "reviewed_branch_summaries", {}) or {})
    for branch_name in REQUIRED_DAG_BRANCHES:
        if branch_name in reviewed:
            continue
        verdict = branch_summaries.get(branch_name)
        if verdict is not None:
            reviewed[branch_name] = _jsonable(verdict)
    if "macro" not in reviewed:
        macro_verdict = getattr(result, "macro_verdict", None)
        if macro_verdict is not None:
            reviewed["macro"] = _jsonable(macro_verdict)
    return reviewed


def _serialize_symbol_review_bundle(result: Any, symbol: str) -> dict[str, Any]:
    review_bundle = getattr(result, "review_bundle", None)
    if review_bundle is None:
        return {}
    branch_overlays = dict(getattr(review_bundle, "branch_overlay_verdicts_by_symbol", {}) or {}).get(symbol, {})
    master_hints = dict(getattr(review_bundle, "master_hints_by_symbol", {}) or {})
    return {
        "branch_overlays": {str(name): _jsonable(verdict) for name, verdict in dict(branch_overlays).items()},
        "master_hint": _jsonable(master_hints.get(symbol)),
    }


def _llm_usage_summary_to_dict(summary: Any) -> dict[str, Any]:
    return {
        "call_count": int(getattr(summary, "call_count", 0) or 0),
        "success_count": int(getattr(summary, "success_count", 0) or 0),
        "fallback_count": int(getattr(summary, "fallback_count", 0) or 0),
        "failed_count": int(getattr(summary, "failed_count", 0) or 0),
        "total_tokens": int(getattr(summary, "total_tokens", 0) or 0),
        "estimated_cost_usd": round(float(getattr(summary, "estimated_cost_usd", 0.0) or 0.0), 8),
    }


def _empty_llm_usage_summary() -> dict[str, Any]:
    return {
        "call_count": 0,
        "success_count": 0,
        "fallback_count": 0,
        "failed_count": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }


def _codex_handoff_review_layer(source_ledger: pd.DataFrame, *, reason: str) -> dict[str, Any]:
    usage = _empty_llm_usage_summary()
    review_by_symbol: dict[str, dict[str, Any]] = {}
    session_ids: dict[str, str] = {}
    for row in source_ledger.itertuples():
        symbol = str(getattr(row, "symbol", "")).strip().upper()
        if not symbol:
            continue
        session_ids[symbol] = ""
        review_by_symbol[symbol] = {
            "llm_usage": dict(usage),
            "llm_attempt_summary": dict(usage),
            "llm_effective_summary": dict(usage),
            "llm_session_id": "",
            "ic_hint": {},
            "recommendation": {},
            "report_excerpt": "",
            "llm_degraded": False,
            "llm_degraded_reason": "",
            "reviewed_branch_verdicts": {},
            "branch_overlays": {},
            "master_hint": {},
            "recall_context": {"holding_symbol": symbol},
            "codex_handoff": True,
        }
    return {
        "reviewed_symbols": list(review_by_symbol.keys()),
        "by_symbol": review_by_symbol,
        "degraded_symbols": {},
        "llm_usage_summary": dict(usage),
        "llm_attempt_summary": dict(usage),
        "llm_effective_summary": dict(usage),
        "model_role_metadata": {
            "agent_layer_enabled": False,
            "branch_model": "codex-handoff",
            "master_model": "codex-handoff",
            "local_llm_disabled": True,
            "llm_handoff": "codex",
            "handoff_reason": reason,
        },
        "fallback_reasons": [reason],
        "session_ids": session_ids,
        "local_llm_disabled": True,
        "codex_handoff": True,
    }


def _trade_recommendation_to_dict(recommendation: Any) -> dict[str, Any]:
    if recommendation is None:
        return {}
    payload = _plain_dict(recommendation)
    if payload:
        return payload
    return {
        "symbol": str(getattr(recommendation, "symbol", "")),
        "action": str(getattr(recommendation, "action", "")),
        "weight": float(getattr(recommendation, "weight", 0.0) or 0.0),
        "confidence": float(getattr(recommendation, "confidence", 0.0) or 0.0),
        "one_line_conclusion": str(getattr(recommendation, "one_line_conclusion", "")),
        "risk_flags": list(getattr(recommendation, "risk_flags", []) or []),
        "metadata": dict(getattr(recommendation, "metadata", {}) or {}),
    }


def _branch_payload_present(payload: Any) -> bool:
    if payload is None:
        return False
    if isinstance(payload, dict):
        return bool(payload)
    return True


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, dict) else {}


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_structured_evidence(payload: dict[str, Any]) -> bool:
    evidence = payload.get("evidence")
    if isinstance(evidence, list) and evidence:
        return True
    score = _coerce_float(payload.get("final_score", payload.get("score")))
    confidence = _coerce_float(
        payload.get("final_confidence", payload.get("confidence"))
    )
    return score is not None and confidence is not None and confidence >= 0.35


def _average_mapping_ratio(values: Any) -> float | None:
    if not isinstance(values, dict) or not values:
        return None
    ratios = [
        max(0.0, min(1.0, float(value)))
        for value in values.values()
        if _coerce_float(value) is not None
    ]
    if not ratios:
        return None
    return float(sum(ratios) / len(ratios))


def _quant_governance_state(payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _as_mapping(payload.get("metadata"))
    runtime = _as_mapping(metadata.get("mined_factor_runtime"))
    registry = _as_mapping(runtime.get("registry"))
    governance = _as_mapping(registry.get("governance_runtime"))
    blockers = list(
        runtime.get("runtime_blockers")
        or governance.get("blockers")
        or metadata.get("runtime_blockers")
        or []
    )
    return {
        "governance_status": str(
            runtime.get("governance_status")
            or metadata.get("governance_status")
            or governance.get("status")
            or payload.get("governance_status")
            or "governance_blocked"
        ),
        "factor_mode": str(
            runtime.get("factor_mode")
            or metadata.get("factor_mode")
            or governance.get("factor_mode")
            or payload.get("factor_mode")
            or "governance_blocked"
        ),
        "production_eligible": (
            runtime.get(
                "production_eligible",
                governance.get("production_eligible", False),
            )
            is True
        ),
        "runtime_blockers": [
            str(item) for item in blockers if str(item).strip()
        ],
        "legacy_fallback_allowed": bool(
            runtime.get(
                "legacy_fallback_allowed",
                governance.get("legacy_fallback_allowed", False),
            )
        ),
        "runtime": runtime,
        "governance": governance,
    }


def _quant_evidence_limited(payload: dict[str, Any]) -> bool:
    governance_state = _quant_governance_state(payload)
    factor_mode = str(governance_state["factor_mode"]).strip()
    runtime = _as_mapping(governance_state.get("runtime"))
    factor_count = int(_coerce_float(runtime.get("factor_count")) or 0)
    factors_used = runtime.get("factors_used")
    applied = runtime.get("applied_to_score")
    applied_to_score = (
        bool(applied) if applied is not None else factor_count > 0
    )
    average_factor_coverage = _average_mapping_ratio(
        runtime.get("factor_coverages")
    )
    if (
        governance_state["governance_status"] != "ready"
        or factor_mode != "governed_mined_factors"
        or governance_state["production_eligible"] is not True
        or governance_state["runtime_blockers"]
        or governance_state["legacy_fallback_allowed"] is True
        or factor_count <= 0
        or not factors_used
    ):
        return True
    if not applied_to_score:
        return True
    if average_factor_coverage is not None and average_factor_coverage < 0.80:
        return True
    return not _has_structured_evidence(payload)


def _fundamental_evidence_limited(payload: dict[str, Any]) -> bool:
    if not _has_structured_evidence(payload):
        return True
    data_quality = _as_mapping(payload.get("data_quality"))
    coverage_ratio = _coerce_float(data_quality.get("coverage_ratio"))
    if coverage_ratio is not None and coverage_ratio < 0.50:
        return True
    metadata = _as_mapping(payload.get("metadata"))
    module_coverage = _as_mapping(metadata.get("module_coverage"))
    if module_coverage:
        active_modules = [
            _as_mapping(item)
            for item in module_coverage.values()
            if _as_mapping(item).get("status") != "disabled_global"
        ]
        if active_modules:
            covered = [
                _coerce_float(item.get("coverage_ratio"))
                for item in active_modules
            ]
            usable = [float(item) for item in covered if item is not None]
            if usable and sum(usable) / len(usable) < 0.50:
                return True
    return False


def _branch_evidence_limited(branch_name: str, payload: Any) -> bool:
    if not isinstance(payload, dict) or not payload:
        return False
    status = str(payload.get("status", "")).strip().lower()
    if status in {"error", "failed", "failure"}:
        return True
    notes: list[str] = []
    for key in ("diagnostic_notes", "coverage_notes", "investment_risks"):
        raw = payload.get(key)
        if isinstance(raw, list):
            notes.extend(str(item).lower() for item in raw)
        elif raw:
            notes.append(str(raw).lower())
    metadata = _as_mapping(payload.get("metadata"))
    data_quality = _as_mapping(metadata.get("data_quality")) or _as_mapping(
        payload.get("data_quality")
    )
    coverage_ratio = data_quality.get("coverage_ratio")
    if coverage_ratio is not None:
        try:
            if float(coverage_ratio) < 0.5:
                return True
        except (TypeError, ValueError):
            pass
    limited_markers = (
        "fallback",
        "placeholder",
        "provider_missing",
        "snapshot_missing",
        "runtime_error",
        "compute_error",
        "empty_factor_values",
        "证据不足",
    )
    if any(any(marker in note for marker in limited_markers) for note in notes):
        return True
    if branch_name == "quant":
        return _quant_evidence_limited(payload)
    if branch_name == "fundamental":
        return _fundamental_evidence_limited(payload)
    return False


def _build_dag_four_branch_compliance(
    *,
    review_symbols: list[str],
    effective_local_holding_symbols: list[str],
    branch_signals_by_symbol: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ordered_symbols = list(
        dict.fromkeys(
            [
                *(
                    str(symbol).strip().upper()
                    for symbol in review_symbols
                    if str(symbol).strip()
                ),
                *(
                    str(symbol).strip().upper()
                    for symbol in effective_local_holding_symbols
                    if str(symbol).strip()
                ),
            ]
        )
    )
    present_by_symbol: dict[str, list[str]] = {}
    missing_by_symbol: dict[str, list[str]] = {}
    limited_by_symbol: dict[str, list[str]] = {}
    for symbol in ordered_symbols:
        payload = dict(branch_signals_by_symbol.get(symbol, {}) or {})
        reviewed = dict(payload.get("reviewed_branch_verdicts", {}) or {})
        present = [
            branch_name
            for branch_name in REQUIRED_DAG_BRANCHES
            if _branch_payload_present(reviewed.get(branch_name))
        ]
        missing = [
            branch_name
            for branch_name in REQUIRED_DAG_BRANCHES
            if branch_name not in present
        ]
        limited = [
            branch_name
            for branch_name in present
            if _branch_evidence_limited(branch_name, reviewed.get(branch_name))
        ]
        present_by_symbol[symbol] = present
        missing_by_symbol[symbol] = missing
        if limited:
            limited_by_symbol[symbol] = limited

    complete = (
        all(not missing for missing in missing_by_symbol.values())
        if ordered_symbols
        else False
    )
    if complete and limited_by_symbol:
        reason = (
            "All required non-LLM DAG branches are materialized; "
            "limited-evidence branches remain size caps."
        )
    elif complete:
        reason = (
            "All required non-LLM DAG branches are materialized with substantive "
            "branch evidence; module-level coverage notes remain diagnostics only."
        )
    else:
        reason = (
            "Some required non-LLM DAG branches are not materialized in "
            "reviewed_branch_verdicts."
        )
    return {
        "required_branches": list(REQUIRED_DAG_BRANCHES),
        "status": "DAG四分支完整执行" if complete else "DAG四分支未完整执行",
        "complete": complete,
        "present_branch_by_symbol": present_by_symbol,
        "missing_branch_by_symbol": missing_by_symbol,
        "limited_evidence_branch_by_symbol": limited_by_symbol,
        "formal_review_symbols": list(review_symbols),
        "effective_local_holding_symbols": list(effective_local_holding_symbols),
        "reason": reason,
        "evidence_quality_adjustment": (
            "keep_limited_evidence_position_caps"
            if complete and limited_by_symbol
            else ("none" if complete else "lower_evidence_quality_and_keep_actions_watch_or_no_action")
        ),
    }


def _render_dag_compliance_markdown(compliance: dict[str, Any]) -> list[str]:
    missing_by_symbol = dict(compliance.get("missing_branch_by_symbol", {}) or {})
    present_by_symbol = dict(compliance.get("present_branch_by_symbol", {}) or {})
    limited_by_symbol = dict(compliance.get("limited_evidence_branch_by_symbol", {}) or {})
    lines = [
        "#### 5.3.1 DAG 四分支执行验收",
        "",
        f"- required_branches：`{', '.join(compliance.get('required_branches', REQUIRED_DAG_BRANCHES))}`",
        f"- status：`{compliance.get('status', 'unknown')}`",
        f"- complete：`{str(bool(compliance.get('complete', False))).lower()}`",
        "- present_branch_by_symbol：见下表。",
        "- missing_branch_by_symbol：见下表。",
        f"- limited_evidence_branch_by_symbol：{limited_by_symbol or {}}",
        f"- 原因：{compliance.get('reason', '')}",
        f"- 执行影响：{compliance.get('evidence_quality_adjustment', '')}",
        "",
        "| symbol | present | missing | limited |",
        "| --- | --- | --- | --- |",
    ]
    for symbol in missing_by_symbol:
        present = ", ".join(present_by_symbol.get(symbol, []) or ["-"])
        missing = ", ".join(missing_by_symbol.get(symbol, []) or ["-"])
        limited = ", ".join(limited_by_symbol.get(symbol, []) or ["-"])
        lines.append(f"| {symbol} | {present} | {missing} | {limited} |")
    return lines


def _summarize_factor_governance_runtime(
    branch_signals_by_symbol: Mapping[str, Any],
) -> dict[str, Any]:
    symbol_rows: dict[str, dict[str, Any]] = {}
    for raw_symbol, raw_payload in branch_signals_by_symbol.items():
        payload = _mapping_payload(raw_payload)
        reviewed = _mapping_payload(payload.get("reviewed_branch_verdicts"))
        quant = _mapping_payload(reviewed.get("quant"))
        metadata = _mapping_payload(quant.get("metadata"))
        runtime = _mapping_payload(metadata.get("mined_factor_runtime"))
        registry = _mapping_payload(runtime.get("registry"))
        governance = _mapping_payload(registry.get("governance_runtime"))
        if not (quant or metadata or runtime or governance):
            continue
        status = str(
            runtime.get("governance_status")
            or metadata.get("governance_status")
            or governance.get("status")
            or quant.get("governance_status")
            or "governance_blocked"
        )
        factor_mode = str(
            runtime.get("factor_mode")
            or metadata.get("factor_mode")
            or governance.get("factor_mode")
            or quant.get("factor_mode")
            or "governance_blocked"
        )
        blockers = list(
            runtime.get("runtime_blockers")
            or governance.get("blockers")
            or metadata.get("runtime_blockers")
            or []
        )
        blocker_texts = [str(item) for item in blockers if str(item).strip()]
        symbol_rows[str(raw_symbol)] = {
            "governance_status": status,
            "factor_mode": factor_mode,
            "confidence": _safe_float(
                runtime.get("confidence_multiplier"),
                _safe_float(governance.get("confidence_multiplier"), 0.0),
            ),
            "production_eligible": bool(
                runtime.get("production_eligible", governance.get("production_eligible", False))
            ),
            "registry_selectable_factor_count": int(
                _safe_float(
                    governance.get("production_factor_count"),
                    _safe_float(registry.get("production_factor_count"), 0.0),
                )
            ),
            "registry_selectable_factor_names": list(
                governance.get("production_factor_names")
                or registry.get("production_factor_names")
                or []
            ),
            "registry_sha256": str(
                governance.get("production_factor_set_sha256")
                or registry.get("production_factor_set_sha256")
                or ""
            ),
            "protocol_version": str(governance.get("protocol_version") or "v2"),
            "protocol_hash": str(governance.get("protocol_hash") or ""),
            "factors_used": list(runtime.get("factors_used") or []),
            "blockers": blocker_texts,
            "slot_incumbents": _jsonable(governance.get("slot_incumbents") or {}),
            "normalized_abs_weights": _jsonable(
                governance.get("normalized_abs_weights") or {}
            ),
            "family_normalized_abs_weights": _jsonable(
                governance.get("family_normalized_abs_weights") or {}
            ),
            "canonical_replay_producer_control": _jsonable(
                governance.get("canonical_replay_producer_control") or {}
            ),
            "slot_blockers": [
                item for item in blocker_texts if "slot" in item
            ],
            "family_blockers": [
                item for item in blocker_texts if "family" in item
            ],
            "risk_budget_blockers": [
                item
                for item in blocker_texts
                if "abs_weight" in item or "risk_budget" in item
            ],
            "evidence_blockers": [
                item
                for item in blocker_texts
                if "evidence" in item or "canonical" in item or "replay" in item
            ],
            "legacy_fallback_allowed": bool(
                runtime.get("legacy_fallback_allowed", governance.get("legacy_fallback_allowed", False))
            ),
        }
    if not symbol_rows:
        return {
            "schema_version": "factor_governance_runtime_summary.v1",
            "governance_status": "governance_blocked",
            "factor_mode": "governance_blocked",
            "confidence": 0.0,
            "production_eligible": False,
            "registry_selectable_factor_count": 0,
            "registry_selectable_factor_names": [],
            "registry_sha256": "",
            "protocol_version": "v2",
            "protocol_hash": "",
            "factors_used": [],
            "blockers": ["holding_quant_runtime_evidence_missing"],
            "legacy_fallback_allowed": False,
            "slot_incumbents": {},
            "normalized_abs_weights": {},
            "family_normalized_abs_weights": {},
            "canonical_replay_producer_control": {},
            "slot_blockers": [],
            "family_blockers": [],
            "risk_budget_blockers": [],
            "evidence_blockers": ["holding_quant_runtime_evidence_missing"],
            "symbol_runtime": {},
        }
    first = next(iter(symbol_rows.values()))
    status_values = {row["governance_status"] for row in symbol_rows.values()}
    mode_values = {row["factor_mode"] for row in symbol_rows.values()}
    blockers = list(
        dict.fromkeys(
            blocker
            for row in symbol_rows.values()
            for blocker in list(row.get("blockers") or [])
        )
    )
    if len(status_values) > 1 or len(mode_values) > 1:
        blockers.append("holding_factor_governance_runtime_inconsistent")
    runtime_fingerprints = {
        (
            row.get("registry_sha256"),
            row.get("protocol_hash"),
            row.get("registry_selectable_factor_count"),
            tuple(row.get("registry_selectable_factor_names") or []),
        )
        for row in symbol_rows.values()
    }
    if len(runtime_fingerprints) > 1:
        blockers.append("holding_factor_registry_runtime_inconsistent")
    for row in symbol_rows.values():
        if not list(row.get("factors_used") or []):
            blockers.append("holding_factor_runtime_factors_used_missing")
        if _safe_float(row.get("confidence"), 0.0) <= 0:
            blockers.append("holding_factor_runtime_confidence_non_positive")
        selectable_names = list(row.get("registry_selectable_factor_names") or [])
        if (
            int(row.get("registry_selectable_factor_count") or 0) <= 0
            or len(selectable_names)
            != int(row.get("registry_selectable_factor_count") or 0)
        ):
            blockers.append("holding_factor_registry_selectable_set_invalid")
        for field, blocker in (
            ("registry_sha256", "holding_factor_registry_sha256_invalid"),
            ("protocol_hash", "holding_factor_protocol_hash_invalid"),
        ):
            value = str(row.get(field) or "").lower()
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                blockers.append(blocker)
        canonical_control = _mapping_payload(
            row.get("canonical_replay_producer_control")
        )
        if not (
            canonical_control.get("producer_available") is True
            and canonical_control.get("artifact_bytes_readback_bound") is True
            and canonical_control.get("production_apply_eligible") is True
        ):
            blockers.append("holding_factor_canonical_control_not_ready")
    blockers = list(dict.fromkeys(blockers))
    ready = bool(
        status_values == {"ready"}
        and mode_values == {"governed_mined_factors"}
        and all(row.get("production_eligible") is True for row in symbol_rows.values())
        and all(
            row.get("legacy_fallback_allowed") is False
            for row in symbol_rows.values()
        )
        and not blockers
    )
    factors_used = sorted(
        {
            str(factor)
            for row in symbol_rows.values()
            for factor in list(row.get("factors_used") or [])
            if str(factor)
        }
    )
    return {
        "schema_version": "factor_governance_runtime_summary.v1",
        "governance_status": "ready" if ready else "governance_blocked",
        "factor_mode": "governed_mined_factors" if ready else "governance_blocked",
        "confidence": min(
            [_safe_float(row.get("confidence"), 0.0) for row in symbol_rows.values()]
            or [0.0]
        ) if ready else 0.0,
        "production_eligible": ready,
        "registry_selectable_factor_count": int(first["registry_selectable_factor_count"]),
        "registry_selectable_factor_names": list(first["registry_selectable_factor_names"]),
        "registry_sha256": str(first["registry_sha256"]),
        "protocol_version": str(first["protocol_version"]),
        "protocol_hash": str(first["protocol_hash"]),
        "factors_used": factors_used if ready else [],
        "blockers": blockers,
        "slot_incumbents": _jsonable(first.get("slot_incumbents") or {}),
        "normalized_abs_weights": _jsonable(
            first.get("normalized_abs_weights") or {}
        ),
        "family_normalized_abs_weights": _jsonable(
            first.get("family_normalized_abs_weights") or {}
        ),
        "canonical_replay_producer_control": _jsonable(
            first.get("canonical_replay_producer_control") or {}
        ),
        "slot_blockers": list(
            dict.fromkeys(
                item
                for row in symbol_rows.values()
                for item in list(row.get("slot_blockers") or [])
            )
        ),
        "family_blockers": list(
            dict.fromkeys(
                item
                for row in symbol_rows.values()
                for item in list(row.get("family_blockers") or [])
            )
        ),
        "risk_budget_blockers": list(
            dict.fromkeys(
                item
                for row in symbol_rows.values()
                for item in list(row.get("risk_budget_blockers") or [])
            )
        ),
        "evidence_blockers": list(
            dict.fromkeys(
                item
                for row in symbol_rows.values()
                for item in list(row.get("evidence_blockers") or [])
            )
        ),
        "legacy_fallback_allowed": any(
            bool(row.get("legacy_fallback_allowed"))
            for row in symbol_rows.values()
        ),
        "symbol_runtime": symbol_rows,
    }


def _format_factor_governance_runtime_lines(summary: Mapping[str, Any]) -> list[str]:
    names = ", ".join(str(item) for item in list(summary.get("registry_selectable_factor_names") or [])) or "none"
    used = ", ".join(str(item) for item in list(summary.get("factors_used") or [])) or "none"
    blockers = "；".join(str(item) for item in list(summary.get("blockers") or [])) or "none"
    slot_blockers = "；".join(
        str(item) for item in list(summary.get("slot_blockers") or [])
    ) or "none"
    family_blockers = "；".join(
        str(item) for item in list(summary.get("family_blockers") or [])
    ) or "none"
    risk_budget_blockers = "；".join(
        str(item) for item in list(summary.get("risk_budget_blockers") or [])
    ) or "none"
    evidence_blockers = "；".join(
        str(item) for item in list(summary.get("evidence_blockers") or [])
    ) or "none"
    return [
        f"- governance_status：`{summary.get('governance_status', 'governance_blocked')}`",
        f"- factor_mode：`{summary.get('factor_mode', 'governance_blocked')}`；confidence=`{_safe_float(summary.get('confidence')):.3f}`；production_eligible=`{str(bool(summary.get('production_eligible'))).lower()}`",
        f"- registry selectable：count=`{int(_safe_float(summary.get('registry_selectable_factor_count')))}`；names=`{names}`；SHA=`{summary.get('registry_sha256') or 'missing'}`",
        f"- protocol：version=`{summary.get('protocol_version') or 'v2'}`；hash=`{summary.get('protocol_hash') or 'missing'}`",
        f"- factors_used：`{used}`；legacy_proxy_fallback=`{str(bool(summary.get('legacy_fallback_allowed'))).lower()}`",
        f"- slot incumbents：`{json.dumps(summary.get('slot_incumbents') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- risk budgets：factor=`{json.dumps(summary.get('normalized_abs_weights') or {}, ensure_ascii=False, sort_keys=True)}`；family=`{json.dumps(summary.get('family_normalized_abs_weights') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- canonical producer control：`{json.dumps(summary.get('canonical_replay_producer_control') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- slot/family/risk/evidence blockers：slot=`{slot_blockers}`；family=`{family_blockers}`；risk=`{risk_budget_blockers}`；evidence=`{evidence_blockers}`",
        f"- blockers：`{blockers}`",
    ]


def _build_holding_dag_assessments(
    branch_signals_by_symbol: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    assessments: dict[str, dict[str, Any]] = {}
    for raw_symbol, raw_payload in branch_signals_by_symbol.items():
        symbol = str(raw_symbol).strip().upper()
        payload = _mapping_payload(raw_payload)
        reviewed = _mapping_payload(payload.get("reviewed_branch_verdicts"))
        present = [
            branch
            for branch in REQUIRED_DAG_BRANCHES
            if _branch_payload_present(reviewed.get(branch))
        ]
        missing = [branch for branch in REQUIRED_DAG_BRANCHES if branch not in present]
        limited = [
            branch
            for branch in present
            if _branch_evidence_limited(
                branch,
                _mapping_payload(reviewed.get(branch)),
            )
        ]
        quant = _mapping_payload(reviewed.get("quant"))
        quant_state = _quant_governance_state(quant)
        governance_status = str(quant_state["governance_status"])
        factor_mode = str(quant_state["factor_mode"])
        blockers = [f"missing_branch:{branch}" for branch in missing]
        blockers.extend(f"limited_evidence:{branch}" for branch in limited)
        if governance_status != "ready":
            blockers.append(f"quant_governance_status:{governance_status}")
        if factor_mode != "governed_mined_factors":
            blockers.append(f"quant_factor_mode:{factor_mode}")
        if quant_state["production_eligible"] is not True:
            blockers.append("quant_production_eligible:false")
        if quant_state["legacy_fallback_allowed"] is True:
            blockers.append("quant_legacy_fallback_allowed:true")
        blockers.extend(
            f"quant_runtime_blocker:{item}"
            for item in quant_state["runtime_blockers"]
        )
        recall_context = _mapping_payload(payload.get("recall_context"))
        recall_holding_symbol = str(recall_context.get("holding_symbol") or "").strip().upper()
        if recall_holding_symbol != symbol:
            blockers.append("recall_context_holding_symbol_mismatch")
        assessments[symbol] = {
            "holding_dag_status": "ready" if not blockers else "blocked",
            "holding_dag_blockers": list(dict.fromkeys(blockers)),
            "present_branches": present,
            "missing_branches": missing,
            "limited_evidence_branches": limited,
            "quant_governance_status": governance_status,
            "factor_mode": factor_mode,
            "quant_production_eligible": bool(
                quant_state["production_eligible"]
            ),
            "quant_legacy_fallback_allowed": bool(
                quant_state["legacy_fallback_allowed"]
            ),
            "recall_context": recall_context,
            "recall_holding_symbol": recall_holding_symbol,
            "allowed_action_scope": (
                "normal_advisory_subject_to_all_gates"
                if not blockers
                else "read_only_risk_reducing_sell_or_clear_only"
            ),
        }
    return assessments


def _run_unified_review_mainline_for_holdings(
    *,
    source_ledger: pd.DataFrame,
    latest_trade_date: str,
    source_record: str,
) -> dict[str, Any]:
    review_by_symbol: dict[str, dict[str, Any]] = {}
    degraded_symbols: dict[str, str] = {}
    llm_settings = _load_daily_config_llm_settings()
    requested_agent_layer = bool(llm_settings.pop("enable_agent_layer", True))
    handoff_reason = llm_handoff_reason()
    codex_handoff_active = bool(handoff_reason or not requested_agent_layer)
    expected_no_local_llm_reason = (
        handoff_reason
        if handoff_reason
        else ("daily_config_agent_layer_disabled_codex_handoff" if not requested_agent_layer else "")
    )
    review_models = ResolvedReviewModels.from_mapping(llm_settings)
    aggregate_attempt_usage = {
        "call_count": 0,
        "success_count": 0,
        "fallback_count": 0,
        "failed_count": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
    aggregate_effective_usage = {
        "call_count": 0,
        "success_count": 0,
        "fallback_count": 0,
        "failed_count": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
    model_role_metadata: dict[str, Any] = {}
    fallback_reasons: list[str] = []
    session_ids: dict[str, str] = {}

    for row in source_ledger.itertuples():
        symbol = str(getattr(row, "symbol", "")).strip().upper()
        if not symbol:
            continue
        current_value = _safe_float(getattr(row, "current_value", 0.0), 0.0)
        cost_basis = _safe_float(getattr(row, "cost_basis", 0.0), 0.0)
        review_capital = max(current_value, cost_basis, 10_000.0)
        investor = QuantInvestor(
            stock_pool=[symbol],
            market="CN",
            lookback_years=1.0,
            total_capital=review_capital,
            risk_level="积极",
            verbose=False,
            enable_agent_layer=requested_agent_layer,
            universe_key="full_a",
            enable_document_semantics=True,
            **review_models.to_runtime_kwargs(),
            recall_context={
                "strategy": "aggressive_tech_manufacturing",
                "source_record": source_record,
                "latest_trade_date": latest_trade_date,
                "holding_symbol": symbol,
            },
        )
        result = investor.run()
        attempt_usage = _llm_usage_summary_to_dict(getattr(result, "llm_usage_summary", None))
        effective_usage = _llm_usage_summary_to_dict(getattr(result, "llm_effective_summary", None))
        for key in ("call_count", "success_count", "fallback_count", "failed_count", "total_tokens"):
            aggregate_attempt_usage[key] += int(attempt_usage[key])
            aggregate_effective_usage[key] += int(effective_usage[key])
        aggregate_attempt_usage["estimated_cost_usd"] = round(
            float(aggregate_attempt_usage["estimated_cost_usd"]) + float(attempt_usage["estimated_cost_usd"]),
            8,
        )
        aggregate_effective_usage["estimated_cost_usd"] = round(
            float(aggregate_effective_usage["estimated_cost_usd"]) + float(effective_usage["estimated_cost_usd"]),
            8,
        )
        role_payload = _plain_dict(getattr(result, "model_role_metadata", None))
        if role_payload and not model_role_metadata:
            model_role_metadata = role_payload
        review_bundle = getattr(result, "review_bundle", None)
        fallback_reasons.extend(list(getattr(review_bundle, "fallback_reasons", []) or []))
        degraded_reason = ""
        if attempt_usage["call_count"] <= 0:
            if expected_no_local_llm_reason:
                fallback_reasons.append(f"{symbol}: {expected_no_local_llm_reason}")
            else:
                degraded_reason = (
                    f"{symbol} review-layer 未产生有效 LLM 调用，已按降级模式继续；"
                    "请核对模型配额、provider 可用性或 free-tier 限制。"
                )
                degraded_symbols[symbol] = degraded_reason
                fallback_reasons.append(degraded_reason)
        recommendations = list(getattr(getattr(result, "final_strategy", None), "recommendations", []) or [])
        if not recommendations:
            recommendations = list(getattr(getattr(result, "final_strategy", None), "trade_recommendations", []) or [])
        recommendation = next((item for item in recommendations if str(getattr(item, "symbol", "")).upper() == symbol), None)
        ic_hint = dict((getattr(result, "ic_hints_by_symbol", {}) or {}).get(symbol, {}))
        session_ids[symbol] = str(getattr(result, "llm_usage_session_id", "") or "")
        reviewed_branch_verdicts = _serialize_reviewed_branch_verdicts(result, symbol)
        symbol_review_bundle = _serialize_symbol_review_bundle(result, symbol)
        review_by_symbol[symbol] = {
            "llm_usage": attempt_usage,
            "llm_attempt_summary": attempt_usage,
            "llm_effective_summary": effective_usage,
            "llm_session_id": session_ids[symbol],
            "ic_hint": ic_hint,
            "recommendation": _trade_recommendation_to_dict(recommendation),
            "report_excerpt": str(getattr(result, "final_report", "") or "")[:2000],
            "llm_degraded": bool(degraded_reason),
            "llm_degraded_reason": degraded_reason,
            "reviewed_branch_verdicts": reviewed_branch_verdicts,
            "branch_overlays": dict(symbol_review_bundle.get("branch_overlays", {}) or {}),
            "master_hint": dict(symbol_review_bundle.get("master_hint", {}) or {}),
            "recall_context": {
                "strategy": "aggressive_tech_manufacturing",
                "source_record": source_record,
                "latest_trade_date": latest_trade_date,
                "holding_symbol": symbol,
            },
            "codex_handoff": codex_handoff_active,
            "local_llm_disabled": bool(handoff_reason),
        }
        time.sleep(0.8)

    return {
        "reviewed_symbols": list(review_by_symbol.keys()),
        "by_symbol": review_by_symbol,
        "degraded_symbols": degraded_symbols,
        "llm_usage_summary": aggregate_attempt_usage,
        "llm_attempt_summary": aggregate_attempt_usage,
        "llm_effective_summary": aggregate_effective_usage,
        "model_role_metadata": model_role_metadata,
        "fallback_reasons": sorted(dict.fromkeys(item for item in fallback_reasons if str(item).strip())),
        "session_ids": session_ids,
        "local_llm_disabled": bool(handoff_reason),
        "codex_handoff": codex_handoff_active,
        "non_llm_dag_executed": bool(review_by_symbol),
    }


def _now_local() -> datetime:
    return datetime.now().astimezone()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def _safe_pct(value: float, base: float) -> float:
    if abs(base) < 1e-12:
        return 0.0
    return value / base


def _trailing_take_profit_row_date(row: Any, columns: tuple[str, ...]) -> str:
    for column in columns:
        value = row.get(column) if isinstance(row, pd.Series) else getattr(row, column, "")
        trade_date = _compact_trade_date(value)
        if trade_date:
            return trade_date
    return ""


def _trailing_take_profit_row_float(row: Any, columns: tuple[str, ...]) -> float:
    for column in columns:
        value = row.get(column) if isinstance(row, pd.Series) else getattr(row, column, None)
        number = _safe_float(value, 0.0)
        if number > 0:
            return number
    return 0.0


def _unconfirmed_trailing_basis(previous_basis: str) -> str:
    prefix = "unconfirmed_missing_entry_anchor"
    normalized: list[str] = []
    for token in str(previous_basis or "").split(":"):
        value = token.strip()
        if not value or value == prefix or value in normalized:
            continue
        if value == "manual_ledger_entry_date":
            value = "prior_manual_ledger_entry_date_without_current_anchor"
        if value not in normalized:
            normalized.append(value)
    return ":".join([prefix, *normalized])


def _trailing_take_profit_peak_from_history(
    frame: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    current_price: float,
) -> tuple[float, str, str]:
    if frame.empty or "trade_date" not in frame.columns:
        return 0.0, "", ""

    price_column = "high" if "high" in frame.columns else "close" if "close" in frame.columns else ""
    if not price_column:
        return 0.0, "", ""

    work = frame.copy()
    work["trade_date"] = work["trade_date"].map(_compact_trade_date)
    work = work[work["trade_date"].str.len().eq(8)].copy()
    if start_date:
        work = work[work["trade_date"] >= start_date].copy()
    if end_date:
        work = work[work["trade_date"] <= end_date].copy()
    if work.empty:
        return 0.0, "", ""

    work["_trailing_price"] = pd.to_numeric(work[price_column], errors="coerce")
    work = work[work["_trailing_price"] > 0].copy()
    if work.empty:
        return 0.0, "", ""

    peak_idx = work["_trailing_price"].idxmax()
    peak_price = _safe_float(work.loc[peak_idx, "_trailing_price"], 0.0)
    peak_trade_date = str(work.loc[peak_idx, "trade_date"] or "")
    peak_source = price_column
    if current_price > peak_price:
        peak_price = current_price
        peak_trade_date = end_date or peak_trade_date
        peak_source = "realtime_current"
    return round(peak_price, 6), peak_trade_date, peak_source


def _classify_trailing_take_profit(
    *,
    current_price: float,
    buy_price: float,
    shares: int,
    peak_price: float,
    stage_stop_price: float,
    score_full_market: float,
    market_weight: float,
    realtime_quote_valid: bool,
    confirmed_basis: bool,
) -> dict[str, Any]:
    peak_unrealized_profit = max(peak_price - buy_price, 0.0) * shares
    current_unrealized_profit = max(current_price - buy_price, 0.0) * shares
    giveback_ratio = (
        (peak_unrealized_profit - current_unrealized_profit) / peak_unrealized_profit
        if peak_unrealized_profit > 0
        else 0.0
    )
    review_price = (
        buy_price + (1.0 - TRAILING_TAKE_PROFIT_REVIEW_GIVEBACK_RATIO) * (peak_price - buy_price)
        if peak_price > buy_price
        else 0.0
    )
    reduce_price = (
        buy_price + (1.0 - TRAILING_TAKE_PROFIT_REDUCE_GIVEBACK_RATIO) * (peak_price - buy_price)
        if peak_price > buy_price
        else 0.0
    )
    trailing_stop = max(stage_stop_price, review_price) if review_price > 0 else stage_stop_price

    if not realtime_quote_valid or current_price <= 0 or buy_price <= 0 or shares <= 0 or peak_price <= 0:
        status = "unconfirmed"
        reason = "实时价、持仓成本、股数或利润峰值不可验证，移动止盈不推导卖出。"
    elif peak_unrealized_profit <= 0:
        status = "no_sell_signal"
        reason = "尚无正向利润峰值，移动止盈沿用阶段止损。"
    elif not confirmed_basis:
        status = "unconfirmed"
        reason = "利润峰值缺少可验证的建仓锚点，仅保留诊断，不推导卖出。"
    elif giveback_ratio >= TRAILING_TAKE_PROFIT_REDUCE_GIVEBACK_RATIO and confirmed_basis:
        status = "reduce_risk"
        reason = (
            f"利润峰值回吐 {giveback_ratio:.2%} 已超过 "
            f"{TRAILING_TAKE_PROFIT_REDUCE_GIVEBACK_RATIO:.0%} 风险线，需用实时 quote 和人工确认减仓。"
        )
    elif (
        giveback_ratio > 0
        and confirmed_basis
        and (current_price < stage_stop_price or score_full_market < 0.60)
    ):
        status = "clear_risk"
        reason = "利润回吐叠加阶段止损/评分破位，需确认是否退出风险仓。"
    elif giveback_ratio >= TRAILING_TAKE_PROFIT_REVIEW_GIVEBACK_RATIO:
        status = "hold_with_trailing_stop"
        reason = (
            f"利润峰值回吐 {giveback_ratio:.2%} 已达到 "
            f"{TRAILING_TAKE_PROFIT_REVIEW_GIVEBACK_RATIO:.0%} 复核线，设置移动止盈观察。"
        )
    elif market_weight >= 0.18 and current_unrealized_profit > 0 and current_price <= trailing_stop:
        status = "hold_with_trailing_stop"
        reason = "盈利仓位较大且接近移动止盈复核价，需重点观察但不自动成交。"
    else:
        status = "no_sell_signal"
        reason = (
            f"利润峰值回吐 {giveback_ratio:.2%} 未达到 "
            f"{TRAILING_TAKE_PROFIT_REVIEW_GIVEBACK_RATIO:.0%} 复核线。"
        )

    return {
        "trailing_take_profit_status": status,
        "peak_unrealized_profit": round(peak_unrealized_profit, 2),
        "current_unrealized_profit": round(current_unrealized_profit, 2),
        "profit_giveback_ratio": round(max(giveback_ratio, 0.0), 6),
        "trailing_profit_review_price": round(review_price, 2) if review_price > 0 else 0.0,
        "trailing_profit_reduce_price": round(reduce_price, 2) if reduce_price > 0 else 0.0,
        "trailing_stop_price": round(trailing_stop, 2) if trailing_stop > 0 else 0.0,
        "trailing_take_profit_reason": reason,
    }


def _apply_trailing_take_profit_review(
    holdings_review: pd.DataFrame,
    *,
    reader: Any,
    analysis_trade_date: str,
) -> pd.DataFrame:
    if holdings_review.empty:
        return holdings_review

    result = holdings_review.copy()
    symbols = [
        str(symbol).strip().upper()
        for symbol in result.get("symbol", pd.Series(dtype=object)).tolist()
        if str(symbol).strip()
    ]
    history_by_symbol: dict[str, pd.DataFrame] = {}
    history_error = ""
    history_error_by_symbol: dict[str, str] = {}
    try:
        read_results = reader.read_symbol_frames(
            symbols,
            universe_key="full_a",
            end_date=analysis_trade_date,
            columns=["ts_code", "trade_date", "high", "close"],
        )
        for symbol, read_result in dict(read_results or {}).items():
            history_by_symbol[str(symbol).strip().upper()] = read_result.frame.copy()
    except Exception as exc:
        history_error = str(exc)
    for symbol in symbols:
        if not history_by_symbol.get(symbol, pd.DataFrame()).empty:
            continue
        if not hasattr(reader, "read_symbol_frame"):
            if history_error:
                history_error_by_symbol[symbol] = history_error
            continue
        try:
            read_result = reader.read_symbol_frame(
                symbol,
                universe_key="full_a",
                end_date=analysis_trade_date,
                columns=["ts_code", "trade_date", "high", "close"],
            )
            history_by_symbol[symbol] = read_result.frame.copy()
        except Exception as exc:
            history_error_by_symbol[symbol] = str(exc)

    rows: list[dict[str, Any]] = []
    end_date = _compact_trade_date(analysis_trade_date)
    for _, row in result.iterrows():
        symbol = str(row.get("symbol") or "").strip().upper()
        current_price = _safe_float(row.get("current_price"), 0.0)
        buy_price = _safe_float(row.get("buy_price"), 0.0)
        shares = int(_safe_float(row.get("shares_before"), 0.0))
        stage_stop = _safe_float(row.get("stage_stop_price"), 0.0)
        score = _safe_float(row.get("score_full_market"), 0.0)
        nav_weight = _safe_float(row.get("nav_weight"), _safe_float(row.get("market_weight"), 0.0))
        realtime_quote_valid = bool(row.get("realtime_quote_valid", True))

        manual_entry_date = _trailing_take_profit_row_date(
            row,
            TRAILING_TAKE_PROFIT_ENTRY_DATE_COLUMNS,
        )
        previous_basis = str(row.get("trailing_take_profit_basis") or "").strip()
        previous_tracking_start = _compact_trade_date(row.get("trailing_profit_tracking_start_date"))
        tracking_start_date = manual_entry_date or previous_tracking_start or end_date
        carried_peak_price = _trailing_take_profit_row_float(
            row,
            TRAILING_TAKE_PROFIT_PEAK_PRICE_COLUMNS,
        )
        carried_peak_trade_date = _trailing_take_profit_row_date(
            row,
            TRAILING_TAKE_PROFIT_PEAK_DATE_COLUMNS,
        )

        history_frame = history_by_symbol.get(symbol, pd.DataFrame())
        history_record_count = int(len(history_frame.index)) if not history_frame.empty else 0
        history_peak_price = 0.0
        history_peak_trade_date = ""
        history_peak_source = ""
        if not history_frame.empty:
            history_start_date = manual_entry_date if manual_entry_date else ""
            history_peak_price, history_peak_trade_date, history_peak_source = (
                _trailing_take_profit_peak_from_history(
                    history_frame,
                    start_date=history_start_date,
                    end_date=end_date,
                    current_price=current_price,
                )
            )

        if manual_entry_date:
            peak_price = max(history_peak_price, carried_peak_price, current_price)
            peak_trade_date = (
                history_peak_trade_date
                if history_peak_price >= carried_peak_price
                else carried_peak_trade_date
            ) or end_date
            basis = "manual_ledger_entry_date"
            confirmed_basis = True
        elif carried_peak_price > 0:
            peak_price = max(carried_peak_price, current_price)
            peak_trade_date = carried_peak_trade_date or end_date
            basis = _unconfirmed_trailing_basis(previous_basis)
            confirmed_basis = False
        elif history_peak_price > 0:
            peak_price = max(history_peak_price, current_price)
            peak_trade_date = history_peak_trade_date or end_date
            basis = "strict_parquet_available_history_unconfirmed_missing_entry_anchor"
            confirmed_basis = False
        else:
            peak_price = 0.0
            peak_trade_date = end_date
            symbol_history_error = history_error_by_symbol.get(symbol) or history_error
            basis = (
                f"unconfirmed_history_unavailable:{symbol_history_error}"
                if symbol_history_error
                else "unconfirmed_history_unavailable"
            )
            confirmed_basis = False

        classified = _classify_trailing_take_profit(
            current_price=current_price,
            buy_price=buy_price,
            shares=shares,
            peak_price=peak_price,
            stage_stop_price=stage_stop,
            score_full_market=score,
            market_weight=nav_weight,
            realtime_quote_valid=realtime_quote_valid,
            confirmed_basis=confirmed_basis,
        )
        reason = str(classified["trailing_take_profit_reason"])
        if not confirmed_basis and classified["trailing_take_profit_status"] != "unconfirmed":
            reason = (
                reason
                + " 峰值来自 strict Parquet 可得历史但缺少 ledger 买入日期，"
                "仅作为移动止盈观察，不作为自动卖出依据。"
            )

        if stage_stop <= 0:
            stop_loss_status = "missing_stop"
        elif not realtime_quote_valid or current_price <= 0:
            stop_loss_status = "pending_unverified_quote"
        elif current_price <= stage_stop:
            stop_loss_status = "triggered_at_or_below_stage_stop"
        else:
            stop_loss_status = "not_triggered_above_stage_stop"

        rows.append(
            {
                "manual_entry_trade_date": manual_entry_date,
                "trailing_profit_tracking_start_date": tracking_start_date,
                "trailing_take_profit_confirmed": bool(confirmed_basis),
                "trailing_take_profit_basis": basis,
                "trailing_profit_peak_price": round(peak_price, 2) if peak_price > 0 else 0.0,
                "trailing_profit_peak_trade_date": peak_trade_date,
                "trailing_profit_peak_source": history_peak_source or ("carried" if carried_peak_price > 0 else ""),
                "history_record_count": history_record_count,
                "stop_loss_status": stop_loss_status,
                **classified,
                "trailing_profit_reduce_price_20": classified["trailing_profit_review_price"],
                "trailing_profit_reduce_price_35": classified["trailing_profit_reduce_price"],
                "trailing_take_profit_reason": reason,
            }
        )

    trailing_frame = pd.DataFrame(rows)
    for column in trailing_frame.columns:
        result[column] = trailing_frame[column].values
    return result


def _summarize_trailing_take_profit(holdings_review: pd.DataFrame) -> dict[str, Any]:
    if holdings_review.empty or "trailing_take_profit_status" not in holdings_review.columns:
        return {
            "schema_version": "cn_aggressive_trailing_take_profit.v1",
            "status": "missing",
            "review_giveback_ratio": TRAILING_TAKE_PROFIT_REVIEW_GIVEBACK_RATIO,
            "reduce_giveback_ratio": TRAILING_TAKE_PROFIT_REDUCE_GIVEBACK_RATIO,
            "status_counts": {},
            "watch_symbols": [],
            "unconfirmed_symbols": [],
            "records": [],
        }

    records = holdings_review.to_dict(orient="records")
    status_counts = Counter(str(row.get("trailing_take_profit_status") or "unknown") for row in records)
    watch_statuses = {"hold_with_trailing_stop", "reduce_risk", "clear_risk", "cash_hold"}
    watch_symbols = [
        str(row.get("symbol"))
        for row in records
        if str(row.get("trailing_take_profit_status") or "") in watch_statuses
    ]
    unconfirmed_symbols = [
        str(row.get("symbol"))
        for row in records
        if str(row.get("trailing_take_profit_status") or "") == "unconfirmed"
    ]
    return {
        "schema_version": "cn_aggressive_trailing_take_profit.v1",
        "status": "success",
        "review_giveback_ratio": TRAILING_TAKE_PROFIT_REVIEW_GIVEBACK_RATIO,
        "reduce_giveback_ratio": TRAILING_TAKE_PROFIT_REDUCE_GIVEBACK_RATIO,
        "status_counts": dict(sorted(status_counts.items())),
        "watch_symbols": watch_symbols,
        "unconfirmed_symbols": unconfirmed_symbols,
        "max_profit_giveback_ratio": round(
            max((_safe_float(row.get("profit_giveback_ratio"), 0.0) for row in records), default=0.0),
            6,
        ),
        "records": [
            {
                "symbol": row.get("symbol"),
                "name": row.get("name"),
                "status": row.get("trailing_take_profit_status"),
                "confirmed": bool(row.get("trailing_take_profit_confirmed")),
                "basis": row.get("trailing_take_profit_basis"),
                "peak_price": row.get("trailing_profit_peak_price"),
                "current_price": row.get("current_price"),
                "profit_giveback_ratio": row.get("profit_giveback_ratio"),
                "trailing_stop_price": row.get("trailing_stop_price"),
                "review_price": row.get("trailing_profit_review_price"),
                "reduce_price": row.get("trailing_profit_reduce_price"),
                "reason": row.get("trailing_take_profit_reason"),
            }
            for row in records
        ],
    }


def _map_symbol_to_quote_code(symbol: str) -> str:
    text = str(symbol).strip().upper()
    if text.startswith(("SH", "SZ")) and "." not in text:
        return text.lower()
    code, market = text.split(".")
    prefix = "sh" if market == "SH" else "sz"
    return f"{prefix}{code}"


def _decode_quote_payload(content: bytes) -> str:
    for encoding in ("gbk", "gb18030", "utf-8"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


def _parse_quote_payload(line: str) -> dict[str, Any] | None:
    text = str(line).strip()
    if not text or "=" not in text or "~" not in text:
        return None

    prefix, payload = text.split("=", 1)
    quote_code = prefix.replace("v_", "").strip()
    payload = payload.strip().strip(";").strip('"')
    parts = payload.split("~")
    if len(parts) < 6:
        return None

    current = _safe_float(parts[3])
    prev_close = _safe_float(parts[4])
    quote_time = parts[30].strip() if len(parts) > 30 else ""
    change = _safe_float(parts[31], current - prev_close) if len(parts) > 31 else current - prev_close
    change_pct = (
        _safe_float(parts[32], _safe_pct(change, prev_close) * 100.0)
        if len(parts) > 32
        else _safe_pct(change, prev_close) * 100.0
    )
    return {
        "quote_code": quote_code,
        "source": "tencent_realtime_quote",
        "name": parts[1].strip() or quote_code,
        "current": current,
        "realtime_price": current,
        "realtime_price_field": "current",
        "prev_close": prev_close,
        "open": _safe_float(parts[5]),
        "high": _safe_float(parts[33]) if len(parts) > 33 else 0.0,
        "low": _safe_float(parts[34]) if len(parts) > 34 else 0.0,
        "time": quote_time,
        "quote_timestamp": quote_time,
        "change": change,
        "change_pct": change_pct,
    }


def _quote_timestamp(quote: dict[str, Any]) -> str:
    for key in ("time", "quote_timestamp", "timestamp", "datetime", "fetched_at"):
        value = str(quote.get(key) or "").strip()
        if value:
            return value
    return ""


def _parse_quote_datetime(value: Any, *, now: datetime) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    digits = "".join(character for character in text if character.isdigit())
    try:
        if len(digits) == 14 and len(text) <= 19:
            parsed = datetime.strptime(digits, "%Y%m%d%H%M%S")
        else:
            timestamp = pd.Timestamp(text)
            if pd.isna(timestamp):
                return None
            parsed = timestamp.to_pydatetime()
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=now.tzinfo)
    return parsed.astimezone(now.tzinfo)


def _quote_freshness(
    quote: dict[str, Any],
    *,
    now: datetime,
    max_age_seconds: int,
) -> tuple[bool, str, float | None]:
    quote_time = _parse_quote_datetime(_quote_timestamp(quote), now=now)
    if quote_time is None:
        return False, "quote_timestamp_missing_or_invalid", None
    if quote_time.date() != now.date():
        return False, "quote_not_same_local_trade_date", None
    age_seconds = (now - quote_time).total_seconds()
    if age_seconds < -5:
        return False, "quote_timestamp_in_future", age_seconds
    if age_seconds > max(0, int(max_age_seconds)):
        return False, "quote_ttl_expired", age_seconds
    return True, "fresh", age_seconds


def _resolve_realtime_execution_price(
    quote: dict[str, Any],
    *,
    now: datetime | None = None,
    max_age_seconds: int = DEFAULT_QUOTE_MAX_AGE_SECONDS,
) -> tuple[float, str]:
    """Return a validated realtime execution price and the field it came from."""
    if not isinstance(quote, dict) or not quote:
        return 0.0, ""
    if not _quote_timestamp(quote):
        return 0.0, ""
    if now is not None:
        fresh, _, _ = _quote_freshness(
            quote,
            now=now,
            max_age_seconds=max_age_seconds,
        )
        if not fresh:
            return 0.0, ""

    low = _safe_float(quote.get("low"), 0.0)
    high = _safe_float(quote.get("high"), 0.0)
    has_intraday_range = low > 0 and high > 0 and low <= high
    fields = list(REALTIME_EXECUTION_PRICE_FIELDS)
    declared_field = str(quote.get("realtime_price_field") or quote.get("execution_price_field") or "").strip()
    if declared_field and declared_field not in fields:
        fields.insert(0, declared_field)
    for field in fields:
        if field not in quote:
            continue
        price = _safe_float(quote.get(field), 0.0)
        if price <= 0:
            continue
        if has_intraday_range and not (low <= price <= high):
            continue
        return price, field
    return 0.0, ""


def _stable_read_bytes(path: Path) -> bytes:
    before = path.stat()
    first = path.read_bytes()
    middle = path.stat()
    second = path.read_bytes()
    after = path.stat()
    if (
        first != second
        or before.st_ino != middle.st_ino
        or middle.st_ino != after.st_ino
        or before.st_size != middle.st_size
        or middle.st_size != after.st_size
        or before.st_mtime_ns != middle.st_mtime_ns
        or middle.st_mtime_ns != after.st_mtime_ns
    ):
        raise RuntimeError(f"file changed during stable read: {path}")
    return first


def _load_local_quote_input(path_value: str | Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    path = Path(path_value).expanduser()
    project_root = PROJECT_ROOT.resolve(strict=True)
    resolved = path.resolve(strict=True)
    results_root = (project_root / "results").resolve(strict=True)
    try:
        resolved.relative_to(results_root)
    except ValueError as exc:
        raise RuntimeError("quote_input_path_must_be_contained_in_ignored_results") from exc
    file_stat = path.lstat()
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise RuntimeError("quote_input_must_be_regular_non_symlink_file")
    if file_stat.st_mode & 0o077:
        raise RuntimeError("quote_input_permissions_must_be_0600")
    raw = _stable_read_bytes(resolved)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("quote_input_payload_must_be_mapping")
    if str(payload.get("schema_version") or "") != QUOTE_INPUT_SCHEMA_VERSION:
        raise RuntimeError("quote_input_schema_version_invalid")
    source = str(payload.get("source") or "").strip()
    captured_at = str(payload.get("captured_at") or "").strip()
    quotes = payload.get("quotes")
    if not source or not captured_at or not isinstance(quotes, dict):
        raise RuntimeError("quote_input_required_fields_missing")
    normalized: dict[str, dict[str, Any]] = {}
    for raw_code, raw_quote in quotes.items():
        if not isinstance(raw_quote, dict):
            raise RuntimeError(f"quote_input_quote_invalid:{raw_code}")
        code = str(raw_code or raw_quote.get("quote_code") or "").strip().lower()
        if not code:
            raise RuntimeError("quote_input_quote_code_missing")
        quote = dict(raw_quote)
        quote.setdefault("quote_code", code)
        quote.setdefault("source", source)
        quote.setdefault("quote_timestamp", captured_at)
        normalized[code] = quote
    return normalized, {
        "mode": "local_quote_input",
        "path": str(resolved),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "schema_version": QUOTE_INPUT_SCHEMA_VERSION,
        "source": source,
        "captured_at": captured_at,
        "permissions": oct(stat.S_IMODE(file_stat.st_mode)),
        "stable_double_read": True,
        "provider_fallback_used": False,
    }


def _fallback_current_price(metric: Mapping[str, Any], row: Any) -> float:
    row_price = _safe_float(getattr(row, "current_price", 0.0), 0.0)
    return _safe_float(metric.get("latest_close"), row_price)


def _fallback_metric_or_row_price(
    metric: Mapping[str, Any],
    metric_field: str,
    row: Any,
    row_field: str,
    default: float,
) -> float:
    row_price = _safe_float(getattr(row, row_field, default), default)
    return _safe_float(metric.get(metric_field), row_price)


def _fetch_tencent_quotes(quote_codes: list[str]) -> dict[str, dict[str, Any]]:
    if not quote_codes:
        return {}

    chunks = [quote_codes[idx : idx + 60] for idx in range(0, len(quote_codes), 60)]
    result: dict[str, dict[str, Any]] = {}
    session = requests.Session()

    for chunk in chunks:
        url = "https://qt.gtimg.cn/q=" + ",".join(chunk)
        response = session.get(url, timeout=QUOTE_TIMEOUT)
        response.raise_for_status()
        payload = _decode_quote_payload(response.content)
        for raw_line in payload.split(";"):
            parsed = _parse_quote_payload(raw_line)
            if parsed is None:
                continue
            result[parsed["quote_code"]] = parsed
    return result


# Full-market metrics/cache helpers live in cn_aggressive_market_metrics.py.

def _summarize_theme_strength(review: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for theme, symbols in THEME_BASKETS.items():
        subset = review[review["symbol"].isin(symbols)].copy()
        if subset.empty:
            continue
        rows.append(
            {
                "theme": theme,
                "symbols": subset["symbol"].tolist(),
                "avg_today_change_pct": float(subset["today_change_pct"].mean()),
                "avg_score": float(subset["score_full_market"].mean()),
                "avg_rank": float(subset["rank_full_market"].mean()),
                "avg_ret20": float(subset["ret20"].mean()),
            }
        )
    rows.sort(
        key=lambda item: (
            item["avg_score"],
            item["avg_today_change_pct"],
            item["avg_ret20"],
            -item["avg_rank"],
            item["theme"],
        ),
        reverse=True,
    )
    return rows


def _market_style_conclusion(indices: dict[str, dict[str, Any]], breadth: dict[str, dict[str, Any]]) -> str:
    hs300 = breadth.get("hs300", {})
    zz1000 = breadth.get("zz1000", {})
    kc50 = indices.get("sh000688", {})
    hs300_idx = indices.get("sh000300", {})
    cyb = indices.get("sz399006", {})

    if (
        kc50.get("change_pct", 0.0) > hs300_idx.get("change_pct", 0.0)
        and zz1000.get("ret20_positive_ratio", 0.0) >= hs300.get("ret20_positive_ratio", 0.0)
    ):
        return "成长修复偏强，科技高弹性重新获得承接。"
    if (
        hs300.get("ret20_positive_ratio", 0.0) >= zz1000.get("ret20_positive_ratio", 0.0)
        and hs300_idx.get("change_pct", 0.0) >= cyb.get("change_pct", 0.0)
    ):
        return "风格仍偏大盘稳健，成长方向只是局部修复。"
    if kc50.get("change_pct", 0.0) > 0 and cyb.get("change_pct", 0.0) < 0:
        return "市场处于结构性修复，科创强于泛成长，资金更偏向有景气验证的硬科技。"
    return "市场仍是结构性分化，未回到全面进攻。"


def _tech_mainline_conclusion(theme_strength: list[dict[str, Any]]) -> tuple[str, str]:
    if not theme_strength:
        return "暂无足够主题样本。", "暂无足够主题样本。"

    strongest = theme_strength[0]
    weakest = theme_strength[-1]
    strong_text = (
        f"{strongest['theme']} 的完整日线强度最强，平均全市场评分 {strongest['avg_score']:.3f}，"
        f"盘中涨跌 {strongest['avg_today_change_pct']:+.2f}%。"
    )
    weak_text = (
        f"{weakest['theme']} 的完整日线强度最弱，平均全市场评分 {weakest['avg_score']:.3f}，"
        f"盘中涨跌 {weakest['avg_today_change_pct']:+.2f}%。"
    )
    return strong_text, weak_text


def _position_role(row: pd.Series) -> str:
    rank = int(row["rank_full_market"])
    price = float(row["current_price"])
    stop_price = float(row["stage_stop_price"])
    if rank <= 30 and price >= stop_price:
        return "核心持有"
    if rank <= 120 and price >= stop_price:
        return "稳定核心"
    if rank <= 260 and price >= stop_price * 0.995:
        return "观察持有"
    return "降级观察"


def _position_action(row: pd.Series) -> str:
    price = float(row["current_price"])
    stop_price = float(row["stage_stop_price"])
    score = float(row["score_full_market"])
    trailing_status = str(row.get("trailing_take_profit_status") or "").strip()
    if trailing_status == "clear_risk":
        return "清仓待确认"
    if price < stop_price and score < 0.72:
        return "减仓待确认"
    if trailing_status == "reduce_risk":
        return "减仓待确认"
    if trailing_status == "hold_with_trailing_stop":
        return "移动止盈观察"
    if price < stop_price:
        return "继续观察"
    return "继续持有"


def _position_reason(row: pd.Series) -> str:
    rank = int(row["rank_full_market"])
    score = float(row["score_full_market"])
    today = float(row["today_change_pct"])
    price = float(row["current_price"])
    stop_price = float(row["stage_stop_price"])
    trailing_status = str(row.get("trailing_take_profit_status") or "").strip()
    trailing_reason = str(row.get("trailing_take_profit_reason") or "").strip()
    trailing_stop = _safe_float(row.get("trailing_stop_price"), 0.0)
    giveback_ratio = _safe_float(row.get("profit_giveback_ratio"), 0.0)

    if trailing_status == "clear_risk":
        return (
            f"{trailing_reason} 当前价 {price:.2f}，移动止盈/复核价 {trailing_stop:.2f}，"
            f"利润回吐 {giveback_ratio:.2%}。"
        )
    if price < stop_price and score < 0.72:
        return (
            f"盘中 {today:+.2f}% 且仍低于阶段止损位 {stop_price:.2f}，"
            f"完整日线强度只在全市场第 {rank} 位，需把减仓判断留在下一次确认。"
        )
    if trailing_status in {"reduce_risk", "hold_with_trailing_stop"}:
        return (
            f"{trailing_reason} 当前价 {price:.2f}，移动止盈/复核价 {trailing_stop:.2f}，"
            f"利润回吐 {giveback_ratio:.2%}。"
        )
    if rank <= 20:
        return (
            f"完整日线仍处第一梯队，全市场排名第 {rank} 位，"
            f"盘中 {today:+.2f}% 说明主线承接仍在。"
        )
    if price < stop_price:
        return (
            f"盘中 {today:+.2f}% ，尚未重新站回阶段止损位 {stop_price:.2f}，"
            f"但全市场评分 {score:.3f} 仍未完全失真，先观察修复延续性。"
        )
    return (
        f"完整日线评分 {score:.3f}，全市场排名第 {rank} 位，"
        f"盘中 {today:+.2f}% ，继续按主线内部分化而非失效处理。"
    )


def _build_rebalance_plan(review: pd.DataFrame) -> list[ProposedOrder]:
    weak = review[
        (review["current_price"] < review["stage_stop_price"])
        & (review["score_full_market"] < 0.72)
        & (review["today_change_pct"] <= 0.5)
    ].sort_values(["market_weight", "score_full_market"], ascending=[False, True])

    if weak.empty:
        return []

    orders: list[ProposedOrder] = []
    for row in weak.head(1).itertuples():
        lot_size = 100
        shares = int(int(row.shares_before) * 0.2 // lot_size) * lot_size
        if shares <= 0:
            continue
        trade_value = round(shares * float(row.current_price), 2)
        realized = round(shares * (float(row.current_price) - float(row.buy_price)), 2)
        orders.append(
            ProposedOrder(
                symbol=row.symbol,
                action="sell",
                shares=shares,
                price=round(float(row.current_price), 2),
                trade_value=trade_value,
                realized_pnl=realized,
                reason=(
                    f"{row.symbol} 仍低于阶段止损位 {float(row.stage_stop_price):.2f}，"
                    "且完整日线已明显落后于组合主线，执行温和减仓。"
                ),
            )
        )
    return orders


def _risk_reduction_sell_gate(
    *,
    order: ProposedOrder,
    effective_ledger: pd.DataFrame,
    holdings_review: pd.DataFrame,
) -> tuple[bool, str]:
    """Classify sell-only orders that may bypass buy-side evidence gates.

    This gate does not validate realtime execution price. It only separates
    risk-reduction sell eligibility from new-risk buy/add/switch eligibility.
    """
    symbol = str(order.symbol or "").strip().upper()
    if str(order.action).strip().lower() != "sell":
        return False, "not_sell_order"
    if not symbol:
        return False, "missing_symbol"
    if int(order.shares) <= 0:
        return False, "non_positive_shares"
    if int(order.shares) % 100 != 0:
        return False, "non_board_lot_sell"
    if effective_ledger.empty:
        return False, "missing_effective_ledger"

    ledger = effective_ledger.copy()
    if "symbol" not in ledger.columns or "shares" not in ledger.columns:
        return False, "invalid_effective_ledger_schema"
    ledger["symbol"] = ledger["symbol"].astype(str).str.strip().str.upper()
    ledger_row = ledger[ledger["symbol"] == symbol]
    if ledger_row.empty:
        return False, "symbol_not_in_effective_ledger"
    held_shares = int(_safe_float(ledger_row.iloc[0].get("shares"), 0.0))
    if int(order.shares) > held_shares:
        return False, "sell_exceeds_effective_ledger_shares"

    if holdings_review.empty:
        return False, "missing_holdings_review"
    review = holdings_review.copy()
    if "symbol" not in review.columns:
        return False, "invalid_holdings_review_schema"
    review["symbol"] = review["symbol"].astype(str).str.strip().str.upper()
    review_row = review[review["symbol"] == symbol]
    if review_row.empty:
        return False, "missing_holding_diagnostics"

    row = review_row.iloc[0]
    current_price = _safe_float(row.get("current_price"), 0.0)
    stage_stop = _safe_float(row.get("stage_stop_price"), 0.0)
    score = _safe_float(row.get("score_full_market"), 1.0)
    action_text = str(row.get("recommended_action", "")).strip()
    reason_text = str(row.get("reason", "")).strip()
    below_stop = stage_stop > 0 and current_price > 0 and current_price < stage_stop
    weak_score = score < 0.72
    explicit_reduce = any(
        marker in action_text or marker in reason_text
        for marker in (
            "减仓",
            "清仓",
            "止损",
            "移动止盈",
            "profit_giveback",
            "reduce_risk",
            "clear_risk",
            "broken stop",
            "risk",
            "thesis invalidation",
        )
    )
    if not (below_stop and weak_score) and not explicit_reduce:
        return False, "no_risk_reduction_signal"
    return True, "risk_reduction_sell_eligible_pending_realtime_quote"


def _theme_label_for_symbol(symbol: str, category: str) -> str:
    for theme, symbols in THEME_BASKETS.items():
        if symbol in symbols:
            return theme
    return CATEGORY_THEME_LABELS.get(category, "正式筛选候选")


def _mapping_payload(value: Any) -> dict[str, Any]:
    payload = _jsonable(value)
    return dict(payload) if isinstance(payload, dict) else {}


def _list_payloads(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    return [_mapping_payload(item) for item in values]


def _artifact_member(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _bool_payload(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}
    return bool(value)


def _int_payload(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _theme_pool_metadata_from_dag_artifacts(
    dag_artifacts: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    funnel_summary = _mapping_payload(dag_artifacts.get("funnel_summary"))
    funnel_metadata = _mapping_payload(funnel_summary.get("funnel_metadata"))
    pool_metadata = _mapping_payload(funnel_metadata.get("theme_pool"))
    if pool_metadata:
        return pool_metadata, funnel_metadata

    funnel_output = dag_artifacts.get("funnel_output")
    funnel_metadata = _mapping_payload(_artifact_member(funnel_output, "funnel_metadata"))
    pool_metadata = _mapping_payload(funnel_metadata.get("theme_pool"))
    return pool_metadata, funnel_metadata


def _sample_symbol_map(symbols_by_bucket: dict[str, list[str]]) -> dict[str, list[str]]:
    return {
        bucket: list(symbols[:THEME_POOL_AUDIT_SYMBOL_SAMPLE_LIMIT])
        for bucket, symbols in sorted(symbols_by_bucket.items())
    }


def _theme_pool_audit_from_dag_artifacts(dag_artifacts: dict[str, Any]) -> dict[str, Any]:
    if not dag_artifacts:
        return {}
    pool_metadata, funnel_metadata = _theme_pool_metadata_from_dag_artifacts(dag_artifacts)
    if not pool_metadata:
        return {}

    policy = _mapping_payload(pool_metadata.get("policy"))
    symbol_payloads = _mapping_payload(pool_metadata.get("symbols"))
    symbols: dict[str, dict[str, Any]] = {}
    admitted_symbols_by_source: dict[str, list[str]] = {}
    excluded_symbols_by_reason: dict[str, list[str]] = {}
    excluded_reason_counts: Counter[str] = Counter()

    for raw_symbol, raw_metadata in sorted(symbol_payloads.items()):
        symbol = _symbol_key(raw_symbol)
        if not symbol:
            continue
        metadata = _mapping_payload(raw_metadata)
        admitted = _bool_payload(metadata.get("admitted"))
        source = str(metadata.get("source") or ("admitted" if admitted else "none")).strip()
        reason = str(
            metadata.get("theme_pool_reason")
            or metadata.get("reason")
            or ("admitted" if admitted else "excluded")
        ).strip()
        record = {
            "admitted": admitted,
            "source": source,
            "primary_theme_id": str(metadata.get("primary_theme_id") or ""),
            "theme_pool_score": round(_safe_float(metadata.get("theme_pool_score")), 6),
            "theme_policy_regime": str(metadata.get("theme_policy_regime") or ""),
            "theme_pool_reason": reason,
        }
        symbols[symbol] = record
        if admitted:
            admitted_symbols_by_source.setdefault(source or "admitted", []).append(symbol)
        else:
            reason_key = reason or "excluded"
            excluded_reason_counts[reason_key] += 1
            excluded_symbols_by_reason.setdefault(reason_key, []).append(symbol)

    admitted_themes = _jsonable(pool_metadata.get("admitted_themes") or [])
    rejected_themes = _jsonable(pool_metadata.get("rejected_themes") or [])
    status = str(pool_metadata.get("status") or funnel_metadata.get("theme_pool_status") or "")
    theme_pool_status = str(funnel_metadata.get("theme_pool_status") or status)
    policy_regime = str(policy.get("regime") or policy.get("policy_regime") or "")
    if not policy_regime and symbols:
        policy_regime = str(next(iter(symbols.values())).get("theme_policy_regime") or "")

    summary = {
        "schema_version": THEME_POOL_AUDIT_SCHEMA_VERSION,
        "enabled": _bool_payload(pool_metadata.get("enabled")),
        "required": _bool_payload(pool_metadata.get("required")),
        "status": status,
        "theme_pool_status": theme_pool_status,
        "score_source": str(pool_metadata.get("score_source") or ""),
        "fallback_to_raw_score": _bool_payload(pool_metadata.get("fallback_to_raw_score")),
        "policy": _jsonable(policy),
        "policy_regime": policy_regime,
        "admitted_theme_count": _int_payload(
            pool_metadata.get("admitted_theme_count"),
            len(admitted_themes) if isinstance(admitted_themes, list) else 0,
        ),
        "natural_admitted_theme_count": _int_payload(pool_metadata.get("natural_admitted_theme_count")),
        "forced_theme_count": _int_payload(pool_metadata.get("forced_theme_count")),
        "min_admitted_themes": _int_payload(pool_metadata.get("min_admitted_themes")),
        "rejected_theme_count": _int_payload(
            pool_metadata.get("rejected_theme_count"),
            len(rejected_themes) if isinstance(rejected_themes, list) else 0,
        ),
        "core_symbol_count": _int_payload(pool_metadata.get("core_symbol_count")),
        "residual_symbol_count": _int_payload(pool_metadata.get("residual_symbol_count")),
        "excluded_symbol_count": _int_payload(
            pool_metadata.get("excluded_symbol_count"),
            sum(excluded_reason_counts.values()),
        ),
        "symbol_count": len(symbols),
        "admitted_symbol_count": sum(1 for item in symbols.values() if item.get("admitted")),
        "excluded_reason_counts": dict(sorted(excluded_reason_counts.items())),
        "admitted_symbols_by_source_sample": _sample_symbol_map(admitted_symbols_by_source),
        "excluded_symbols_by_reason_sample": _sample_symbol_map(excluded_symbols_by_reason),
        "admitted_themes": admitted_themes,
        "rejected_themes": rejected_themes,
    }
    return {
        "schema_version": THEME_POOL_AUDIT_SCHEMA_VERSION,
        "summary": _jsonable(summary),
        "admitted_themes": admitted_themes,
        "rejected_themes": rejected_themes,
        "symbols": _jsonable(symbols),
        "admitted_symbols_by_source": _jsonable(admitted_symbols_by_source),
        "excluded_symbols_by_reason": _jsonable(excluded_symbols_by_reason),
    }


def _theme_label_from_theme_record(theme: dict[str, Any]) -> str:
    return str(
        theme.get("theme_name")
        or theme.get("name")
        or theme.get("theme_id")
        or "unknown_theme"
    )


def _format_theme_pool_report_lines(
    theme_pool_summary: dict[str, Any],
    *,
    candidate_pool: pd.DataFrame,
    theme_symbols: dict[str, Any] | None = None,
) -> list[str]:
    theme_summary = _mapping_payload(theme_pool_summary)
    if not theme_summary:
        return ["- Theme Candidate Pool：正式 DAG artifacts 未发现可审计 metadata，需视为可观测性缺口。"]

    theme_symbols = _mapping_payload(theme_symbols)
    reason_counts = _mapping_payload(theme_summary.get("excluded_reason_counts"))
    reason_text = (
        "；".join(f"{reason}={count}" for reason, count in reason_counts.items())
        if reason_counts
        else "无"
    )
    lines = [
        (
            f"- Theme Candidate Pool：status=`{theme_summary.get('status') or 'unknown'}`，"
            f"policy_regime=`{theme_summary.get('policy_regime') or 'unknown'}`，"
            f"admitted_themes=`{theme_summary.get('admitted_theme_count', 0)}`，"
            f"natural=`{theme_summary.get('natural_admitted_theme_count', 0)}`，"
            f"forced=`{theme_summary.get('forced_theme_count', 0)}`，"
            f"min_themes=`{theme_summary.get('min_admitted_themes', 0)}`，"
            f"rejected_themes=`{theme_summary.get('rejected_theme_count', 0)}`，"
            f"core=`{theme_summary.get('core_symbol_count', 0)}`，"
            f"residual=`{theme_summary.get('residual_symbol_count', 0)}`，"
            f"excluded=`{theme_summary.get('excluded_symbol_count', 0)}`，"
            f"excluded_reasons=`{reason_text}`。"
        )
    ]
    forced_theme_count = _int_payload(theme_summary.get("forced_theme_count"))
    if forced_theme_count > 0:
        lines.append(
            "- Theme 强制准入说明："
            f"forced=`{forced_theme_count}` 表示这些主题未自然通过 ThemeGatePolicy，"
            "只是因 `min_admitted_themes` 硬约束进入 Theme core 候选池；"
            "应按“硬加入/强制准入”解读，不等同于自然合格主题。"
        )

    admitted_themes = _list_payloads(theme_summary.get("admitted_themes"))
    if admitted_themes:
        admitted_parts = []
        for theme in admitted_themes[:5]:
            forced = _bool_payload(theme.get("forced"))
            admitted_parts.append(
                f"{_theme_label_from_theme_record(theme)} "
                f"自然通过={str(not forced).lower()} "
                f"强制准入={str(forced).lower()} "
                f"硬加入={str(forced).lower()} "
                f"score={_safe_float(theme.get('theme_score')):.3f} "
                f"rank_score={_safe_float(theme.get('theme_rank_score')):.3f} "
                f"phase={theme.get('phase') or 'unknown'} "
                f"force_reason={theme.get('force_reason') or 'none'} "
                f"original_rejection={theme.get('original_rejection_reason') or 'none'} "
                f"quality_flags={','.join(str(flag) for flag in (theme.get('quality_flags') or [])) or 'none'}"
            )
        admitted_text = "；".join(admitted_parts)
        lines.append(f"- Theme admitted 明细：{admitted_text}。")

    admitted_sample = _mapping_payload(theme_summary.get("admitted_symbols_by_source_sample"))
    candidate_core_symbols: list[str] = []
    if not candidate_pool.empty:
        for row in candidate_pool.head(12).itertuples():
            symbol = str(getattr(row, "symbol", "") or "").strip().upper()
            metadata = _mapping_payload(theme_symbols.get(symbol))
            if symbol and str(metadata.get("source") or "") == "core":
                candidate_core_symbols.append(symbol)
    core_sample = []
    for symbol in [
        *candidate_core_symbols,
        *[str(symbol).strip().upper() for symbol in admitted_sample.get("core", [])],
    ]:
        if symbol and symbol not in core_sample:
            core_sample.append(symbol)
    if core_sample:
        lines.append(
            "- Theme core候选样本："
            f"core样本=`{', '.join(core_sample[:20])}`。"
        )

    if not candidate_pool.empty:
        candidate_parts: list[str] = []
        for row in candidate_pool.head(5).itertuples():
            symbol = str(getattr(row, "symbol", "") or "").strip().upper()
            if not symbol:
                continue
            metadata = _mapping_payload(theme_symbols.get(symbol))
            if not metadata:
                continue
            name = str(getattr(row, "name", "") or "UNKNOWN_NAME")
            candidate_parts.append(
                f"{symbol} {name}: source=`{metadata.get('source') or 'none'}`，"
                f"theme=`{metadata.get('primary_theme_id') or ''}`，"
                f"score=`{_safe_float(metadata.get('theme_pool_score')):.6f}`，"
                f"reason=`{metadata.get('theme_pool_reason') or ''}`"
            )
        if candidate_parts:
            lines.append("- 正式候选 Theme Pool 归属：" + "；".join(candidate_parts) + "。")

    rejected_themes = _list_payloads(theme_summary.get("rejected_themes"))
    if rejected_themes:
        rejected_text = "；".join(
            (
                f"{_theme_label_from_theme_record(theme)} "
                f"reason={theme.get('reason') or 'unknown'} "
                f"score={_safe_float(theme.get('score')):.3f} "
                f"phase={theme.get('phase') or 'unknown'}"
            )
            for theme in rejected_themes[:5]
        )
        lines.append(f"- Theme rejected 前列：{rejected_text}。")

    return lines


def _shortlist_payloads_from_dag_artifacts(
    dag_artifacts: dict[str, Any],
) -> tuple[list[dict[str, Any]], str, bool]:
    shortlist_rows = _list_payloads(dag_artifacts.get("shortlist"))
    if shortlist_rows:
        return shortlist_rows, "dag_artifacts.shortlist", False

    for owner_key in ("report_bundle", "portfolio_decision"):
        shortlist_rows = _list_payloads(
            _artifact_member(dag_artifacts.get(owner_key), "shortlist")
        )
        if shortlist_rows:
            return shortlist_rows, f"{owner_key}.shortlist", True

    return [], "", False


def _symbol_key(value: Any) -> str:
    return str(value or "").strip().upper()


def _candidate_dag_branch_state(packet: dict[str, Any]) -> tuple[list[str], list[str], dict[str, float]]:
    branch_payloads = _mapping_payload(packet.get("branch_verdicts"))
    present = [
        branch
        for branch in REQUIRED_DAG_BRANCHES
        if _branch_payload_present(branch_payloads.get(branch))
        and not _branch_evidence_limited(
            branch,
            _mapping_payload(branch_payloads.get(branch)),
        )
    ]
    missing = [branch for branch in REQUIRED_DAG_BRANCHES if branch not in present]
    scores = {
        branch: round(_safe_float(_mapping_payload(branch_payloads.get(branch)).get("final_score")), 6)
        for branch in present
    }
    return present, missing, scores


def _positive_reason_counts(value: Any) -> dict[str, int]:
    payload = _mapping_payload(value)
    result: dict[str, int] = {}
    for reason, count in payload.items():
        text = str(reason or "").strip()
        numeric = _int_payload(count)
        if text and numeric > 0:
            result[text] = numeric
    return dict(sorted(result.items()))


def _candidate_decay_stage(
    *,
    name: str,
    label: str,
    input_count: int,
    output_count: int,
    rejection_reasons: dict[str, int] | None = None,
    note: str = "",
) -> dict[str, Any]:
    input_count = max(_int_payload(input_count), 0)
    output_count = max(_int_payload(output_count), 0)
    rejected_count = max(input_count - output_count, 0)
    reasons = _positive_reason_counts(rejection_reasons or {})
    reason_total = sum(reasons.values())
    if rejected_count > 0 and reason_total < rejected_count:
        reasons["unattributed_remainder"] = rejected_count - reason_total
    return {
        "stage": name,
        "label": label,
        "input_count": input_count,
        "output_count": output_count,
        "rejected_count": rejected_count,
        "rejection_reasons": reasons,
        "note": note,
    }


def _candidate_decay_constraint_snapshot(
    *,
    theme_pool_summary: dict[str, Any],
    dag_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dag_artifacts = dag_artifacts or {}
    policy = _mapping_payload(theme_pool_summary.get("policy"))
    funnel_summary = _mapping_payload(dag_artifacts.get("funnel_summary"))
    funnel_metadata = _mapping_payload(funnel_summary.get("funnel_metadata"))
    portfolio_payload = _mapping_payload(dag_artifacts.get("portfolio_decision"))
    risk_constraints = _mapping_payload(portfolio_payload.get("risk_constraints"))
    risk_decision = _mapping_payload(risk_constraints.get("risk_decision"))
    portfolio_metadata = _mapping_payload(portfolio_payload.get("metadata"))
    market_settings = get_market_settings("CN")
    gross_cap_value = risk_decision.get("gross_exposure_cap")
    if gross_cap_value is None:
        gross_cap_value = portfolio_payload.get("target_gross_exposure")
    portfolio_max_weight = risk_decision.get("max_weight")
    candidate_symbols = portfolio_metadata.get("candidate_symbols")
    return {
        "min_weight": "not_applied_in_candidate_dag_weight_path",
        "whole_lot": int(getattr(market_settings, "lot_size", 100) or 100),
        "gross_cap": (
            _safe_float(gross_cap_value)
            if gross_cap_value is not None
            else "not_recorded"
        ),
        "single_name_cap_env": round(risk_guard_single_name_weight_cap(), 6),
        "portfolio_max_weight": (
            _safe_float(portfolio_max_weight)
            if portfolio_max_weight is not None
            else "not_recorded"
        ),
        "theme_policy_regime": str(
            theme_pool_summary.get("policy_regime") or policy.get("regime") or ""
        ),
        "theme_min_symbol_score": _safe_float(policy.get("min_symbol_score"), 0.0),
        "theme_min_theme_score": _safe_float(policy.get("min_theme_score"), 0.0),
        "theme_effective_max_candidates": _int_payload(
            theme_pool_summary.get("effective_max_candidates")
        ),
        "theme_requested_max_candidates": _int_payload(
            theme_pool_summary.get("requested_max_candidates")
        ),
        "funnel_max_candidates": _int_payload(funnel_metadata.get("max_candidates")),
        "funnel_min_composite_score": funnel_metadata.get("min_composite_score", "not_recorded"),
        "portfolio_candidate_symbols": len(candidate_symbols) if isinstance(candidate_symbols, list) else 0,
    }


def _candidate_decay_classification(stages: list[dict[str, Any]]) -> dict[str, Any]:
    post_theme_reasons: Counter[str] = Counter()
    for stage in stages:
        if stage.get("stage") == "theme_pool":
            continue
        for reason, count in _positive_reason_counts(stage.get("rejection_reasons")).items():
            if reason == "unattributed_remainder":
                continue
            post_theme_reasons[reason] += count
    total = sum(post_theme_reasons.values())
    if total <= 0:
        return {
            "classification": "no_post_theme_decay",
            "dominant_reason": "",
            "dominant_reason_share": 0.0,
            "decision": "no candidate decay after Theme Pool was observed.",
        }
    dominant_reason, dominant_count = sorted(
        post_theme_reasons.items(),
        key=lambda item: (-item[1], item[0]),
    )[0]
    share = dominant_count / total if total else 0.0
    artifact_limited = dominant_reason.startswith("artifact_only_")
    if share >= CANDIDATE_DECAY_SINGLE_REASON_THRESHOLD and not artifact_limited:
        return {
            "classification": "single_reason_requires_constraint_review",
            "dominant_reason": dominant_reason,
            "dominant_reason_share": round(share, 6),
            "decision": "single reason >=95%; print constraint snapshot and review as freeze-exception before behavior changes.",
        }
    return {
        "classification": "by_design_or_artifact_limited",
        "dominant_reason": dominant_reason,
        "dominant_reason_share": round(share, 6),
        "decision": "causes are dispersed or historical artifacts lack per-row detail; treat empty shortlist as reportable by-design unless future waterfall isolates one mechanical cause.",
    }


def _build_candidate_decay_waterfall(
    *,
    candidate_status: dict[str, Any],
    dag_artifacts: dict[str, Any] | None = None,
    accepted_count: int = 0,
    missing_by_symbol: dict[str, list[str]] | None = None,
    zero_target_symbols: list[str] | None = None,
) -> dict[str, Any]:
    status = _mapping_payload(candidate_status)
    pipeline = _mapping_payload(status.get("dag_pipeline"))
    theme_summary = _mapping_payload(status.get("theme_candidate_pool"))
    dag_artifacts = dag_artifacts or {}
    funnel_summary = _mapping_payload(dag_artifacts.get("funnel_summary"))
    funnel_metadata = _mapping_payload(funnel_summary.get("funnel_metadata"))
    portfolio_payload = _mapping_payload(dag_artifacts.get("portfolio_decision"))
    portfolio_metadata = _mapping_payload(portfolio_payload.get("metadata"))
    candidate_symbols = [
        str(symbol).strip().upper()
        for symbol in list(portfolio_metadata.get("candidate_symbols") or [])
        if str(symbol).strip()
    ]

    symbol_count = _int_payload(theme_summary.get("symbol_count"))
    theme_output = _int_payload(
        theme_summary.get("admitted_symbol_count"),
        _int_payload(theme_summary.get("core_symbol_count")),
    )
    bayesian_count = _int_payload(pipeline.get("bayesian_record_count"))
    shortlist_count = _int_payload(pipeline.get("shortlist_count"))
    portfolio_target_count = _int_payload(pipeline.get("portfolio_target_count"))
    accepted_count = max(_int_payload(accepted_count), 0)
    stages: list[dict[str, Any]] = []

    if symbol_count or theme_output:
        stages.append(
            _candidate_decay_stage(
                name="theme_pool",
                label="full universe -> Theme Pool admitted core",
                input_count=symbol_count,
                output_count=theme_output,
                rejection_reasons=_positive_reason_counts(theme_summary.get("excluded_reason_counts")),
                note="Theme Pool hard filter; this includes non-hard-tech/non-admitted-theme exclusions.",
            )
        )

    after_theme_pool = _int_payload(funnel_metadata.get("after_theme_pool"), theme_output)
    after_score_gates = _int_payload(funnel_metadata.get("after_gates"))
    final_funnel_candidates = _int_payload(
        funnel_metadata.get("final_candidates"),
        _int_payload(funnel_summary.get("candidates_count"), len(candidate_symbols)),
    )
    if after_score_gates > 0:
        stages.append(
            _candidate_decay_stage(
                name="deterministic_score_gate",
                label="Theme Pool admitted core -> score/risk gates",
                input_count=after_theme_pool,
                output_count=after_score_gates,
                rejection_reasons={
                    "deterministic_funnel_below_min_score_or_theme_penalty": max(
                        after_theme_pool - after_score_gates,
                        0,
                    )
                },
                note="Deterministic composite score, theme penalty, and local market-state gates.",
            )
        )
    if final_funnel_candidates > 0 and after_score_gates > 0:
        stages.append(
            _candidate_decay_stage(
                name="deterministic_rank_cutoff",
                label="score/risk gate survivors -> ranked funnel candidates",
                input_count=after_score_gates,
                output_count=final_funnel_candidates,
                rejection_reasons={
                    "deterministic_funnel_rank_cutoff_or_sector_bucket_limit": max(
                        after_score_gates - final_funnel_candidates,
                        0,
                    )
                },
                note="Top-N rank cutoff and optional sector bucket limit.",
            )
        )
    elif theme_output and bayesian_count < theme_output:
        stages.append(
            _candidate_decay_stage(
                name="core_to_bayesian_record",
                label="Theme Pool admitted core -> Bayesian records",
                input_count=theme_output,
                output_count=bayesian_count,
                rejection_reasons={
                    "artifact_only_not_materialized_as_bayesian_record": max(
                        theme_output - bayesian_count,
                        0,
                    )
                },
                note="Historical record lacks in-memory DAG per-symbol reasons for deterministic score/rank/readiness decay.",
            )
        )

    if final_funnel_candidates > 0 and bayesian_count < final_funnel_candidates:
        stages.append(
            _candidate_decay_stage(
                name="branch_readiness",
                label="ranked funnel candidates -> Bayesian records",
                input_count=final_funnel_candidates,
                output_count=bayesian_count,
                rejection_reasons={
                    "branch_or_macro_readiness_block": max(final_funnel_candidates - bayesian_count, 0)
                },
                note="Branch readiness or macro readiness can remove candidates before Bayesian posterior.",
            )
        )

    bayesian_kill_switch_count = 0
    for row in _list_payloads(dag_artifacts.get("bayesian_records")):
        metadata = _mapping_payload(row.get("metadata"))
        if bool(metadata.get("kill_switch")):
            bayesian_kill_switch_count += 1
    bayesian_rejected = max(bayesian_count - shortlist_count, 0)
    bayesian_reasons = (
        {"bayesian_kill_switch": bayesian_kill_switch_count}
        if bayesian_kill_switch_count == bayesian_rejected and bayesian_rejected > 0
        else {"bayesian_shortlist_not_selected_or_kill_switch": bayesian_rejected}
    )
    stages.append(
        _candidate_decay_stage(
            name="bayesian_shortlist",
            label="Bayesian records -> shortlist",
            input_count=bayesian_count,
            output_count=shortlist_count,
            rejection_reasons=bayesian_reasons,
            note="Bayesian posterior shortlist; historical rows may only expose aggregate counts.",
        )
    )

    zero_target_symbols = list(zero_target_symbols or [])
    portfolio_rejected = max(shortlist_count - portfolio_target_count, 0)
    stages.append(
        _candidate_decay_stage(
            name="portfolio_constructor",
            label="shortlist -> positive target weights",
            input_count=shortlist_count,
            output_count=portfolio_target_count,
            rejection_reasons={
                "portfolio_constructor_zero_target_weight_or_reject": portfolio_rejected,
                "portfolio_constructor_zero_target_symbols": len(zero_target_symbols),
            },
            note="RiskGuard/ICCoordinator/PortfolioConstructor deterministic control chain.",
        )
    )

    missing_by_symbol = missing_by_symbol or {}
    missing_counter: Counter[str] = Counter()
    for branches in missing_by_symbol.values():
        for branch in branches:
            missing_counter[f"candidate_pool_missing_{branch}"] += 1
    if portfolio_target_count or accepted_count:
        stages.append(
            _candidate_decay_stage(
                name="candidate_pool_materialization",
                label="positive target weights -> report candidate rows",
                input_count=portfolio_target_count,
                output_count=accepted_count,
                rejection_reasons=dict(missing_counter),
                note="Final report materialization excludes incomplete branch/Bayesian rows and existing holdings.",
            )
        )

    classification = _candidate_decay_classification(stages)
    return {
        "schema_version": CANDIDATE_DECAY_WATERFALL_SCHEMA_VERSION,
        "record_scope": "candidate_level_dag",
        "single_reason_threshold": CANDIDATE_DECAY_SINGLE_REASON_THRESHOLD,
        "stages": stages,
        "classification": classification,
        "constraint_snapshot": _candidate_decay_constraint_snapshot(
            theme_pool_summary=theme_summary,
            dag_artifacts=dag_artifacts,
        ),
    }


def _format_candidate_decay_waterfall_report_lines(
    candidate_status: dict[str, Any],
) -> list[str]:
    waterfall = _mapping_payload(candidate_status.get("candidate_decay_waterfall"))
    if not waterfall:
        return []
    stages = [
        stage
        for stage in _list_payloads(waterfall.get("stages"))
        if _int_payload(stage.get("input_count")) or _int_payload(stage.get("output_count"))
    ]
    if not stages:
        return []
    classification = _mapping_payload(waterfall.get("classification"))
    constraints = _mapping_payload(waterfall.get("constraint_snapshot"))
    lines = [
        "",
        "#### 空 shortlist 衰减瀑布（report-only）",
        "",
        "| stage | input | output | rejected | top rejection reasons |",
        "|---|---:|---:|---:|---|",
    ]
    for stage in stages:
        reasons = _positive_reason_counts(stage.get("rejection_reasons"))
        reason_text = "；".join(
            f"{reason}={count}"
            for reason, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0]))[:5]
        ) or "无"
        lines.append(
            f"| {stage.get('label') or stage.get('stage')} | "
            f"{_int_payload(stage.get('input_count'))} | "
            f"{_int_payload(stage.get('output_count'))} | "
            f"{_int_payload(stage.get('rejected_count'))} | {reason_text} |"
        )
    lines.extend(
        [
            "",
            (
                f"- 瀑布判定：`{classification.get('classification') or 'unknown'}`，"
                f"dominant_reason=`{classification.get('dominant_reason') or 'none'}`，"
                f"share={_safe_float(classification.get('dominant_reason_share')):.2%}。"
            ),
            f"- 判读：{classification.get('decision') or 'N/A'}",
            (
                "- 约束快照："
                f"min_weight=`{constraints.get('min_weight', 'N/A')}`，"
                f"整手=`{constraints.get('whole_lot', 'N/A')}`，"
                f"gross_cap=`{constraints.get('gross_cap', 'N/A')}`，"
                f"single_name_cap_env=`{constraints.get('single_name_cap_env', 'N/A')}`，"
                f"portfolio_max_weight=`{constraints.get('portfolio_max_weight', 'N/A')}`，"
                f"theme_min_symbol_score=`{constraints.get('theme_min_symbol_score', 'N/A')}`。"
            ),
        ]
    )
    return lines


def _candidate_dag_status(
    *,
    candidate_generation_status: str,
    blocker: str,
    evaluated_symbols: list[str],
    accepted_symbols: list[str],
    present_by_symbol: dict[str, list[str]],
    missing_by_symbol: dict[str, list[str]],
    bayesian_record_count: int,
    shortlist_count: int,
    portfolio_target_count: int,
    shortlist_source: str = "",
    shortlist_fallback_used: bool = False,
    shortlist_artifact_missing: bool = False,
    theme_pool_summary: dict[str, Any] | None = None,
    error: str = "",
    dag_executed: bool = True,
) -> dict[str, Any]:
    theme_summary = _mapping_payload(theme_pool_summary)
    return {
        "candidate_generation_status": candidate_generation_status,
        "blocker": blocker,
        "required_branches": list(REQUIRED_DAG_BRANCHES),
        "candidate_source": "v13_full_market_dag",
        "theme_candidate_pool": _jsonable(theme_summary),
        "dag_pipeline": {
            "universe": "full_a",
            "deterministic_funnel": bool(dag_executed),
            "theme_candidate_pool_status": str(
                theme_summary.get("status") or theme_summary.get("theme_pool_status") or ""
            ),
            "candidate_level_four_branch": bool(dag_executed),
            "bayesian_shortlist": bool(dag_executed),
            "riskguard_ic_portfolio_constructor": bool(dag_executed),
            "bayesian_record_count": int(bayesian_record_count),
            "shortlist_count": int(shortlist_count),
            "portfolio_target_count": int(portfolio_target_count),
            "shortlist_source": shortlist_source,
            "shortlist_fallback_used": bool(shortlist_fallback_used),
            "shortlist_artifact_missing": bool(shortlist_artifact_missing),
        },
        "candidate_dag_four_branch_compliance": {
            "complete": bool(accepted_symbols),
            "evaluated_symbols": list(evaluated_symbols),
            "accepted_symbols": list(accepted_symbols),
            "present_branch_by_symbol": present_by_symbol,
            "missing_branch_by_symbol": missing_by_symbol,
            "required_branches": list(REQUIRED_DAG_BRANCHES),
        },
        "error": error,
    }


def _build_candidate_pool_from_v13_dag(
    *,
    dag_artifacts: dict[str, Any],
    held_symbols: list[str],
    max_candidates: int = CANDIDATE_DAG_TOP_K,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not dag_artifacts:
        status = _candidate_dag_status(
            candidate_generation_status="blocked",
            blocker="candidate_dag_incomplete",
            evaluated_symbols=[],
            accepted_symbols=[],
            present_by_symbol={},
            missing_by_symbol={},
            bayesian_record_count=0,
            shortlist_count=0,
            portfolio_target_count=0,
            error="missing_dag_artifacts",
            dag_executed=False,
        )
        return pd.DataFrame(columns=CANDIDATE_POOL_COLUMNS, dtype=object), status

    theme_pool_audit = _theme_pool_audit_from_dag_artifacts(dag_artifacts)
    theme_pool_summary = _mapping_payload(theme_pool_audit.get("summary"))
    held_set = {_symbol_key(symbol) for symbol in held_symbols if _symbol_key(symbol)}
    packets = {
        _symbol_key(symbol): _mapping_payload(packet)
        for symbol, packet in _mapping_payload(dag_artifacts.get("symbol_research_packets")).items()
        if _symbol_key(symbol)
    }
    shortlist_rows, shortlist_source, shortlist_fallback_used = (
        _shortlist_payloads_from_dag_artifacts(dag_artifacts)
    )
    shortlist_by_symbol = {
        _symbol_key(row.get("symbol")): row
        for row in shortlist_rows
        if _symbol_key(row.get("symbol"))
    }
    bayesian_rows = _list_payloads(dag_artifacts.get("bayesian_records"))
    bayesian_by_symbol = {
        _symbol_key(row.get("symbol")): row
        for row in bayesian_rows
        if _symbol_key(row.get("symbol"))
    }
    portfolio_payload = _mapping_payload(dag_artifacts.get("portfolio_decision"))
    target_weights = {
        _symbol_key(symbol): _safe_float(weight)
        for symbol, weight in _mapping_payload(portfolio_payload.get("target_weights")).items()
        if _symbol_key(symbol)
    }
    target_positions = {
        _symbol_key(symbol): _safe_float(weight)
        for symbol, weight in _mapping_payload(portfolio_payload.get("target_positions")).items()
        if _symbol_key(symbol)
    }

    evaluated_symbols = [
        symbol
        for symbol in shortlist_by_symbol
        if symbol not in held_set
    ]
    present_by_symbol: dict[str, list[str]] = {}
    missing_by_symbol: dict[str, list[str]] = {}
    rows: list[dict[str, Any]] = []
    zero_target_symbols: list[str] = []
    for symbol in evaluated_symbols:
        packet = packets.get(symbol, {})
        present, missing, branch_scores = _candidate_dag_branch_state(packet)
        present_by_symbol[symbol] = present
        if missing:
            missing_by_symbol[symbol] = missing
            continue
        bayesian = bayesian_by_symbol.get(symbol)
        if bayesian is None:
            missing_by_symbol[symbol] = ["bayesian"]
            continue
        shortlist = shortlist_by_symbol[symbol]
        target_weight = _safe_float(
            target_weights.get(symbol),
            _safe_float(target_positions.get(symbol), 0.0),
        )
        if target_weight <= 0:
            zero_target_symbols.append(symbol)
            continue
        category = str(packet.get("category") or shortlist.get("category") or "full_a")
        rows.append(
            {
                "symbol": symbol,
                "name": (
                    str(packet.get("company_name") or shortlist.get("company_name") or bayesian.get("company_name") or "")
                    or symbol
                ),
                "category": category,
                "theme_label": _theme_label_for_symbol(symbol, category),
                "candidate_source": "v13_full_market_dag",
                "candidate_rank": 0,
                "candidate_dag_four_branch_complete": True,
                "present_branches": ",".join(REQUIRED_DAG_BRANCHES),
                "missing_branches": "",
                "evidence_quality": "高",
                "bayesian_rank": int(_safe_float(bayesian.get("rank"), len(rows) + 1)),
                "posterior_action_score": round(_safe_float(bayesian.get("posterior_action_score")), 6),
                "posterior_win_rate": round(_safe_float(bayesian.get("posterior_win_rate")), 6),
                "posterior_expected_alpha": round(_safe_float(bayesian.get("posterior_expected_alpha")), 6),
                "posterior_confidence": round(_safe_float(bayesian.get("posterior_confidence")), 6),
                "rank_score": round(_safe_float(shortlist.get("rank_score")), 6),
                "shortlist_action": str(shortlist.get("action") or ""),
                "shortlist_confidence": round(_safe_float(shortlist.get("confidence")), 6),
                "expected_upside": round(_safe_float(shortlist.get("expected_upside")), 6),
                "suggested_weight": round(_safe_float(shortlist.get("suggested_weight")), 6),
                "portfolio_target_weight": round(target_weight, 6),
                "portfolio_target_position": round(_safe_float(target_positions.get(symbol), target_weight), 6),
                "risk_flags": "；".join(str(item).strip() for item in list(shortlist.get("risk_flags") or []) if str(item).strip()),
                "rationale": "；".join(str(item).strip() for item in list(shortlist.get("rationale") or []) if str(item).strip()),
                "branch_quant_score": branch_scores.get("quant"),
                "branch_fundamental_score": branch_scores.get("fundamental"),
                "branch_intelligence_score": branch_scores.get("intelligence"),
                "branch_macro_score": branch_scores.get("macro"),
            }
        )

    rows.sort(
        key=lambda row: (
            -float(row["portfolio_target_weight"]),
            int(row["bayesian_rank"] or 999999),
            -float(row["posterior_action_score"]),
            str(row["symbol"]),
        )
    )
    rows = rows[: max(1, int(max_candidates))]
    for index, row in enumerate(rows, start=1):
        row["candidate_rank"] = index

    positive_target_count = len(
        {
            symbol
            for symbol in {*target_weights, *target_positions}
            if _safe_float(
                target_weights.get(symbol),
                _safe_float(target_positions.get(symbol), 0.0),
            )
            > 0
        }
    )
    shortlist_artifact_missing = (
        not shortlist_rows
        and positive_target_count > 0
    )
    if rows:
        status_name = "complete"
        blocker = ""
    elif missing_by_symbol:
        status_name = "blocked"
        blocker = "candidate_dag_incomplete"
    elif shortlist_artifact_missing:
        status_name = "blocked"
        blocker = "candidate_artifact_shortlist_missing"
    elif (
        bool(theme_pool_summary)
        and _int_payload(theme_pool_summary.get("admitted_theme_count")) == 0
        and not bayesian_rows
        and not shortlist_rows
        and positive_target_count == 0
    ):
        status_name = "empty"
        blocker = "no_candidate_admitted_by_theme_pool"
    else:
        status_name = "empty"
        blocker = "no_candidate_selected_by_portfolio_constructor"
    status = _candidate_dag_status(
        candidate_generation_status=status_name,
        blocker=blocker,
        evaluated_symbols=evaluated_symbols,
        accepted_symbols=[str(row["symbol"]) for row in rows],
        present_by_symbol=present_by_symbol,
        missing_by_symbol=missing_by_symbol,
        bayesian_record_count=len(bayesian_rows),
        shortlist_count=len(shortlist_rows),
        portfolio_target_count=positive_target_count,
        shortlist_source=shortlist_source,
        shortlist_fallback_used=shortlist_fallback_used,
        shortlist_artifact_missing=shortlist_artifact_missing,
        theme_pool_summary=theme_pool_summary,
        error=(
            "shortlist artifact missing while Bayesian records or PortfolioConstructor "
            "target weights are present"
            if shortlist_artifact_missing
            else ""
        ),
    )
    status["candidate_decay_waterfall"] = _build_candidate_decay_waterfall(
        candidate_status=status,
        dag_artifacts=dag_artifacts,
        accepted_count=len(rows),
        missing_by_symbol=missing_by_symbol,
        zero_target_symbols=zero_target_symbols,
    )
    return pd.DataFrame(rows, columns=CANDIDATE_POOL_COLUMNS, dtype=object), status


def _attach_candidate_quote_gate(
    candidate_pool: pd.DataFrame,
    *,
    quote_payload: dict[str, dict[str, Any]],
    analysis_trade_date: str,
    now: datetime | None = None,
    max_age_seconds: int = DEFAULT_QUOTE_MAX_AGE_SECONDS,
) -> pd.DataFrame:
    if candidate_pool.empty:
        return candidate_pool
    result = candidate_pool.copy()
    statuses: list[str] = []
    timestamps: list[str] = []
    prices: list[float | None] = []
    eligible: list[bool] = []
    blockers: list[str] = []
    analysis_date = _compact_trade_date(analysis_trade_date)
    payload_quote_dates = [
        "".join(ch for ch in _quote_timestamp(quote) if ch.isdigit())[:8]
        for quote in quote_payload.values()
    ]
    latest_quote_date = max((date for date in payload_quote_dates if len(date) == 8), default="")
    for row in result.itertuples(index=False):
        symbol = str(getattr(row, "symbol", "") or "").strip().upper()
        quote = quote_payload.get(_map_symbol_to_quote_code(symbol), {})
        quote_time = _quote_timestamp(quote)
        price, _ = _resolve_realtime_execution_price(
            quote,
            now=now,
            max_age_seconds=max_age_seconds,
        )
        quote_date = "".join(ch for ch in quote_time if ch.isdigit())[:8]
        quote_fresh = bool(
            price > 0
            and len(quote_date) == 8
            and quote_date == latest_quote_date
            and (not analysis_date or quote_date >= analysis_date)
        )
        expected_alpha = _safe_float(getattr(row, "posterior_expected_alpha", 0.0))
        action = (
            str(getattr(row, "shortlist_action", "") or "")
            .strip()
            .lower()
            .removeprefix("actionlabel.")
        )
        row_blockers: list[str] = []
        if action != "buy":
            row_blockers.append("recommendation_not_buy")
        if expected_alpha <= 0:
            row_blockers.append("non_positive_expected_alpha")
        if not quote_fresh:
            row_blockers.append("candidate_quote_not_fresh")
        statuses.append("fresh" if quote_fresh else ("stale" if quote_time else "missing"))
        timestamps.append(quote_time)
        prices.append(round(price, 4) if price > 0 else None)
        eligible.append(not row_blockers)
        blockers.append(";".join(row_blockers))
    result["candidate_quote_status"] = statuses
    result["candidate_quote_timestamp"] = timestamps
    result["candidate_quote_price"] = prices
    result["new_position_eligible"] = eligible
    result["eligibility_blockers"] = blockers
    result["manual_override_required"] = True
    result["actionable"] = False
    return result


def _run_candidate_level_v13_dag(
    *,
    held_symbols: list[str],
    analysis_trade_date: str,
    completeness_report: dict[str, Any],
    total_capital: float,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    llm_settings = _load_daily_config_llm_settings()
    requested_agent_layer = bool(llm_settings.pop("enable_agent_layer", True))
    review_models = ResolvedReviewModels.from_mapping(llm_settings)
    data_snapshot = {
        "local_latest_trade_date": analysis_trade_date,
        "analysis_trade_date": analysis_trade_date,
        "summary_text": "CN aggressive formal candidate generation uses strict v13 DAG.",
        "completeness": _jsonable(completeness_report),
    }
    try:
        dag_artifacts = execute_market_dag(
            market="CN",
            universe="full_a",
            categories=None,
            mode="batch",
            batch_size=None,
            total_capital=total_capital,
            top_k=CANDIDATE_DAG_TOP_K,
            shortlist_size=CANDIDATE_DAG_TOP_K,
            data_snapshot=data_snapshot,
            verbose=False,
            enable_agent_layer=requested_agent_layer,
            **review_models.to_runtime_kwargs(),
            recall_context={
                "strategy": "aggressive_tech_manufacturing",
                "review_layer": "candidate_generation",
                "analysis_trade_date": analysis_trade_date,
            },
        )
    except Exception as exc:
        status = _candidate_dag_status(
            candidate_generation_status="blocked",
            blocker="candidate_dag_execution_failed",
            evaluated_symbols=[],
            accepted_symbols=[],
            present_by_symbol={},
            missing_by_symbol={},
            bayesian_record_count=0,
            shortlist_count=0,
            portfolio_target_count=0,
            error=str(exc),
            dag_executed=False,
        )
        return pd.DataFrame(columns=CANDIDATE_POOL_COLUMNS, dtype=object), status, {}
    candidate_pool, status = _build_candidate_pool_from_v13_dag(
        dag_artifacts=dag_artifacts,
        held_symbols=held_symbols,
    )
    theme_pool_audit = _theme_pool_audit_from_dag_artifacts(dag_artifacts)
    return candidate_pool, status, theme_pool_audit


def _build_switch_plan(
    *,
    holdings_review: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    completeness_passed: bool,
    decision_data_sufficient: bool,
) -> pd.DataFrame:
    if holdings_review.empty or candidate_pool.empty:
        return pd.DataFrame(columns=SWITCH_PLAN_COLUMNS)
    required_candidate_columns = {
        "candidate_source",
        "candidate_dag_four_branch_complete",
        "portfolio_target_weight",
        "posterior_action_score",
        "posterior_confidence",
        "bayesian_rank",
    }
    if not required_candidate_columns.issubset(set(candidate_pool.columns)):
        return pd.DataFrame(columns=SWITCH_PLAN_COLUMNS)
    dag_candidates = candidate_pool[
        (candidate_pool["candidate_source"].astype(str) == "v13_full_market_dag")
        & (candidate_pool["candidate_dag_four_branch_complete"].astype(bool))
        & (candidate_pool["portfolio_target_weight"].map(lambda value: _safe_float(value) > 0))
    ].copy()
    if dag_candidates.empty:
        return pd.DataFrame(columns=SWITCH_PLAN_COLUMNS)

    review = holdings_review.copy()
    review["_switch_pressure"] = review.apply(
        lambda row: (
            (3 if "减仓" in str(row.get("recommended_action", "")) else 0)
            + (2 if "降级" in str(row.get("position_role", "")) else 0)
            + (
                1
                if _safe_float(row.get("stage_stop_price")) > 0
                and _safe_float(row.get("current_price")) < _safe_float(row.get("stage_stop_price"))
                else 0
            )
        ),
        axis=1,
    )
    weak_holdings = review[review["_switch_pressure"] > 0].sort_values(
        by=["_switch_pressure", "market_weight", "symbol"],
        ascending=[False, False, True],
    ).head(3)
    if weak_holdings.empty:
        return pd.DataFrame(columns=SWITCH_PLAN_COLUMNS)
    best_candidates = dag_candidates.sort_values(
        by=["portfolio_target_weight", "posterior_action_score", "posterior_confidence", "symbol"],
        ascending=[False, False, False, True],
    ).head(3)

    rows: list[dict[str, Any]] = []
    for weak_row, candidate_row in zip(weak_holdings.itertuples(), best_candidates.itertuples()):
        target_weight = _safe_float(getattr(candidate_row, "portfolio_target_weight", 0.0))
        posterior_score = _safe_float(getattr(candidate_row, "posterior_action_score", 0.0))
        posterior_confidence = _safe_float(getattr(candidate_row, "posterior_confidence", 0.0))
        expected_alpha = _safe_float(getattr(candidate_row, "posterior_expected_alpha", 0.0))
        shortlist_action = (
            str(getattr(candidate_row, "shortlist_action", "") or "")
            .strip()
            .lower()
            .removeprefix("actionlabel.")
        )
        quote_status = str(getattr(candidate_row, "candidate_quote_status", "missing") or "missing")
        quote_fresh = quote_status == "fresh"
        superior = (
            target_weight > 0
            and posterior_score > 0
            and posterior_confidence > 0
            and expected_alpha > 0
            and shortlist_action == "buy"
        )
        data_ready = completeness_passed or decision_data_sufficient
        actionable = superior and data_ready and quote_fresh
        rows.append(
            {
                "sell_symbol": weak_row.symbol,
                "sell_name": weak_row.name,
                "sell_role": weak_row.position_role,
                "sell_recommended_action": getattr(weak_row, "recommended_action", ""),
                "buy_symbol": candidate_row.symbol,
                "buy_name": candidate_row.name,
                "buy_theme": candidate_row.theme_label,
                "buy_bayesian_rank": int(_safe_float(getattr(candidate_row, "bayesian_rank", 999999))),
                "buy_posterior_action_score": round(posterior_score, 6),
                "buy_posterior_confidence": round(posterior_confidence, 6),
                "buy_posterior_expected_alpha": round(expected_alpha, 6),
                "buy_portfolio_target_weight": round(target_weight, 6),
                "candidate_quote_status": quote_status,
                "new_position_eligible": bool(superior and data_ready and quote_fresh),
                "manual_override_required": True,
                "actionable": False,
                "candidate_source": candidate_row.candidate_source,
                "evidence_quality": candidate_row.evidence_quality,
                "priority": "high" if superior and target_weight >= 0.06 else "watch",
                "action": "pending_manual_override" if actionable else ("prepare_switch" if superior else "watch_only"),
                "switch_ratio_hint": "等待 Maxwell 人工 override" if actionable else "先观察，不执行",
                "trigger_threshold": (
                    "候选保持 candidate-level v13 DAG 四分支完整，且 PortfolioConstructor 维持正目标权重；现持仓仍未解除降级/减仓信号"
                    if not actionable
                    else "按20%试探性换仓；后续只在 candidate-level v13 DAG 仍完整且目标权重提升时递增"
                ),
                "no_switch_condition": (
                    "当前前日线+实时行情口径仍未满足决策数据要求，先不把结构优势直接转成实单"
                    if superior and not (completeness_passed or decision_data_sufficient)
                    else "若现持仓解除减仓/降级信号，或候选 Bayesian/RiskGuard/PortfolioConstructor 不再支持正目标权重，则继续持有原仓"
                ),
            }
        )
    return pd.DataFrame(rows, columns=SWITCH_PLAN_COLUMNS)


def _apply_orders(
    source_ledger: pd.DataFrame,
    orders: list[ProposedOrder],
    cash_before: float,
    quote_prices: dict[str, float],
) -> tuple[pd.DataFrame, float, float]:
    ledger = source_ledger.copy()
    ledger = ledger.drop(
        columns=[
            column
            for column in ["nav_weight", "equity_sleeve_weight"]
            if column in ledger.columns
        ]
    )
    ledger["shares"] = ledger["shares"].astype(int)
    cash_after = round(cash_before, 2)
    realized_total = 0.0
    order_map = {order.symbol: order for order in orders}

    updated_rows: list[dict[str, Any]] = []
    for row in ledger.itertuples():
        shares = int(row.shares)
        avg_cost = float(row.avg_cost)
        order = order_map.get(row.symbol)
        if order and order.action == "sell":
            shares = max(0, shares - order.shares)
            cash_after = round(cash_after + order.trade_value, 2)
            realized_total += order.realized_pnl

        price = float(quote_prices.get(row.symbol, getattr(row, "current_price", 0.0)))
        cost_basis = round(shares * avg_cost, 2)
        current_value = round(shares * price, 2)
        unrealized = round(current_value - cost_basis, 2)
        updated_rows.append(
            {
                "symbol": row.symbol,
                "name": row.name,
                "shares": shares,
                "avg_cost": round(avg_cost, 6),
                "cost_basis": cost_basis,
                "current_price": round(price, 2),
                "current_value": current_value,
                "unrealized_pnl": unrealized,
                "unrealized_pnl_pct": round(_safe_pct(unrealized, cost_basis), 6),
            }
        )

    updated = pd.DataFrame(updated_rows)
    invested = float(updated["current_value"].sum()) if not updated.empty else 0.0
    total_value = invested + cash_after
    if invested > 0:
        updated["equity_sleeve_weight"] = (updated["current_value"] / invested).round(6)
        updated["market_weight"] = updated["equity_sleeve_weight"]
    else:
        updated["equity_sleeve_weight"] = 0.0
        updated["market_weight"] = 0.0
    if total_value > 0:
        updated["nav_weight"] = (updated["current_value"] / total_value).round(6)
    else:
        updated["nav_weight"] = 0.0
    return updated, cash_after, round(realized_total, 2)


def _manual_manifest_is_valid_baseline(manifest: dict[str, Any]) -> bool:
    status = str(manifest.get("status") or "").strip()
    execution_status = str(manifest.get("execution_status") or "").strip()
    if not status or status.lower() == "ok":
        return False
    if execution_status and execution_status != status:
        return False
    if not status.startswith(VALID_MANUAL_LEDGER_STATUS_PREFIXES):
        return False
    status_text = " ".join(
        str(manifest.get(key) or "")
        for key in ("status", "execution_status", "price_basis", "note")
    )
    return not any(marker in status_text for marker in INVALID_MANUAL_LEDGER_STATUS_MARKERS)


def _decision_log_symbol_matches(value: Any, symbol: str) -> bool:
    normalized = str(value or "").upper()
    tokens = {
        token.strip()
        for token in normalized.replace("；", ",").replace(";", ",").split(",")
        if token.strip()
    }
    return "PORTFOLIO" in tokens or symbol.upper() in tokens


def _same_day_manual_fill_authorization(
    *,
    decision_log_path: Path,
    trade_date: str,
    symbol: str,
    order_action: str,
    order_shares: int,
    allowed_root: Path = DEFAULT_DECISION_LOG_PATH.parent,
) -> dict[str, Any]:
    result = {
        "authorized": False,
        "decision_log_path": str(decision_log_path),
        "trade_date": trade_date,
        "symbol": symbol,
        "order_action": order_action,
        "order_shares": int(order_shares),
        "advisory_event_id": "",
        "human_action_event_id": "",
        "blockers": [],
    }
    if not decision_log_path.is_file() or decision_log_path.is_symlink():
        result["blockers"] = ["decision_log_missing_or_unsafe"]
        return result
    try:
        resolved_path = decision_log_path.resolve(strict=True)
        resolved_path.relative_to(allowed_root.resolve(strict=True))
    except (FileNotFoundError, OSError, ValueError):
        result["blockers"] = ["decision_log_outside_allowed_root"]
        return result
    if stat.S_IMODE(resolved_path.stat().st_mode) & (stat.S_IWGRP | stat.S_IWOTH):
        result["blockers"] = ["decision_log_permissions_unsafe"]
        return result
    result["decision_log_path"] = str(resolved_path)
    try:
        raw = _stable_read_bytes(resolved_path).decode("utf-8")
    except Exception as exc:
        result["blockers"] = [f"decision_log_unreadable:{exc}"]
        return result

    entries: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            result["blockers"] = [
                f"decision_log_invalid_json_line:{line_number}"
            ]
            return result
        if not isinstance(payload, dict):
            result["blockers"] = [
                f"decision_log_non_object_line:{line_number}"
            ]
            return result
        event_id = str(payload.get("event_id") or "").strip()
        if (
            payload.get("schema_version") != "decision_log.v1"
            or not event_id
            or event_id in seen_event_ids
        ):
            result["blockers"] = [
                f"decision_log_schema_or_event_id_invalid:{line_number}"
            ]
            return result
        seen_event_ids.add(event_id)
        if _compact_trade_date(payload.get("trade_date")) == trade_date:
            entries.append(payload)

    advisories: dict[str, dict[str, Any]] = {}
    allowed_advisory_actions = (
        {"sell", "reduce_risk", "clear_risk"}
        if order_action == "sell"
        else {"buy", "add_risk"}
    )
    for entry in entries:
        if str(entry.get("event_type") or "") != "advisory":
            continue
        if not _decision_log_symbol_matches(entry.get("symbol"), symbol):
            continue
        metadata = _mapping_payload(entry.get("metadata"))
        try:
            advisory_shares = int(metadata.get("shares"))
        except (TypeError, ValueError):
            advisory_shares = -1
        if (
            str(entry.get("action") or "").strip().lower()
            not in allowed_advisory_actions
            or advisory_shares != int(order_shares)
        ):
            continue
        event_id = str(entry.get("event_id") or "").strip()
        if event_id:
            advisories[event_id] = entry

    for entry in entries:
        if str(entry.get("event_type") or "") != "human_action":
            continue
        if not _decision_log_symbol_matches(entry.get("symbol"), symbol):
            continue
        metadata = _mapping_payload(entry.get("metadata"))
        paired_event_id = str(metadata.get("paired_advisory_event_id") or "").strip()
        if paired_event_id not in advisories:
            continue
        if (
            metadata.get("effective_ledger_unchanged") is True
            or metadata.get("no_real_order_placed") is True
            or metadata.get("requires_maxwell_reconfirmation_after_fresh_quote") is True
        ):
            continue
        allowed_actions = (
            {"sell", "reduce_risk", "clear_risk", "local_manual_sell"}
            if order_action == "sell"
            else {"buy", "add_risk", "local_manual_buy"}
        )
        try:
            authorized_shares = int(metadata.get("shares"))
        except (TypeError, ValueError):
            authorized_shares = -1
        explicitly_confirmed = bool(
            str(entry.get("action") or "").strip().lower() in allowed_actions
            and metadata.get("authorized") is True
            and str(metadata.get("approved_by") or "").strip().lower()
            == "maxwell"
            and authorized_shares == int(order_shares)
        )
        if not explicitly_confirmed:
            continue
        result.update(
            {
                "authorized": True,
                "advisory_event_id": paired_event_id,
                "human_action_event_id": str(entry.get("event_id") or ""),
                "blockers": [],
            }
        )
        return result

    result["blockers"] = ["same_day_paired_advisory_human_action_missing"]
    return result


def _manual_manifest_explicit_time_key(manifest: Mapping[str, Any]) -> str:
    for key in (
        "recorded_at",
        "executed_at",
        "updated_at",
        "timestamp_long",
        "record_timestamp",
    ):
        normalized = _validated_datetime_key(manifest.get(key))
        if normalized:
            return normalized
    return ""


def _manual_manifest_explicit_datetime(
    manifest: Mapping[str, Any],
) -> datetime | None:
    for key in (
        "recorded_at",
        "executed_at",
        "updated_at",
        "timestamp_long",
        "record_timestamp",
    ):
        parsed = _validated_datetime(manifest.get(key))
        if parsed is not None:
            return parsed
    return None


def _validated_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    local_tz = _now_local().tzinfo
    timezone_hint = local_tz
    parsed: datetime | None = None
    iso_text = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(iso_text)
    except ValueError:
        normalized = text
        if normalized.endswith(" CST"):
            normalized = normalized[:-4]
        elif normalized.endswith(" UTC"):
            normalized = normalized[:-4]
            timezone_hint = timezone.utc
        for format_string in (
            "%Y-%m-%d %H:%M:%S",
            "%Y%m%d_%H%M%S",
            "%Y%m%d_%H%M",
            "%Y%m%d%H%M%S",
            "%Y%m%d%H%M",
            "%Y%m%d",
        ):
            try:
                parsed = datetime.strptime(normalized, format_string)
                break
            except ValueError:
                continue
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone_hint)
    return parsed.astimezone(timezone.utc)


def _validated_datetime_key(value: Any) -> str:
    parsed = _validated_datetime(value)
    if parsed is None:
        return ""
    return parsed.astimezone(_now_local().tzinfo).strftime("%Y%m%d%H%M%S")


def _manual_manifest_order_key(run_dir: Path, manifest: dict[str, Any]) -> tuple[str, str]:
    for key in (
        "recorded_at",
        "executed_at",
        "updated_at",
        "timestamp_long",
        "record_timestamp",
    ):
        normalized = _validated_datetime_key(manifest.get(key))
        if normalized:
            return normalized, run_dir.name
    return _validated_datetime_key(run_dir.name), run_dir.name


def _resolve_manual_ledger_path(manifest_path: Path, manifest: dict[str, Any]) -> Path | None:
    next_ledger = str(manifest.get("next_ledger_path") or "").strip()
    if not next_ledger:
        return None
    declared = Path(next_ledger).expanduser()
    candidate = declared if declared.is_absolute() else manifest_path.parent / declared
    if candidate.is_symlink():
        return None
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(manifest_path.parent.resolve(strict=True))
    except (FileNotFoundError, OSError, ValueError):
        return None
    if (
        resolved.stem != "ledger_after_manual_switch"
        or resolved.suffix.lower() not in {".parquet", ".csv"}
        or not resolved.is_file()
    ):
        return None
    return resolved


def _declared_manual_ledger_sha256(
    manifest: Mapping[str, Any],
    ledger_path: Path,
) -> str:
    suffix_key = (
        "ledger_after_manual_switch_parquet_sha256"
        if ledger_path.suffix.lower() == ".parquet"
        else "ledger_after_manual_switch_csv_sha256"
    )
    for key in (
        "next_ledger_sha256",
        suffix_key,
        "ledger_after_manual_switch_sha256",
        "ledger_sha256",
    ):
        value = str(manifest.get(key) or "").strip().lower()
        if len(value) == 64 and all(character in "0123456789abcdef" for character in value):
            return value
    return ""


def _read_manual_ledger(path: Path) -> pd.DataFrame:
    raw = _stable_read_bytes(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(io.BytesIO(raw))
    if path.name == "ledger_after_manual_switch.csv":
        return pd.read_table(io.BytesIO(raw), sep=",", encoding="utf-8-sig")
    raise RuntimeError(f"manual ledger 只允许读取 ledger_after_manual_switch sidecar: {path}")


def _build_effective_manual_pnl_summary(
    *,
    manual_manifest: dict[str, Any],
    fallback_pnl_summary: pd.DataFrame,
) -> pd.DataFrame:
    if fallback_pnl_summary.empty:
        row: dict[str, Any] = {}
    else:
        row = dict(fallback_pnl_summary.iloc[-1].to_dict())

    manifest_to_pnl_fields = {
        "cash_after": "cash_after",
        "market_value_after": "market_value_after",
        "total_value_after": "total_value_after",
        "portfolio_pnl_after": "portfolio_pnl_after",
        "portfolio_return_after": "portfolio_pnl_pct_after",
        "realized_pnl_from_rebalance": "realized_pnl_from_rebalance",
        "quote_snapshot": "quote_snapshot",
    }
    for manifest_key, pnl_key in manifest_to_pnl_fields.items():
        value = manual_manifest.get(manifest_key)
        if value not in (None, ""):
            row[pnl_key] = value

    return pd.DataFrame([row])


def _required_manifest_number(
    manifest: Mapping[str, Any],
    key: str,
) -> float:
    if key not in manifest or manifest.get(key) in (None, ""):
        raise RuntimeError(f"v3_manual_manifest_financial_field_missing:{key}")
    try:
        value = float(manifest[key])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"v3_manual_manifest_financial_field_invalid:{key}"
        ) from exc
    if not math.isfinite(value):
        raise RuntimeError(f"v3_manual_manifest_financial_field_invalid:{key}")
    return value


def _manual_financial_state_sha256(
    manifest: Mapping[str, Any],
    *,
    ledger_sha256: str,
) -> str:
    payload = {
        key: manifest.get(key)
        for key in (
            "capital_cny",
            "cash_after",
            "market_value_after",
            "total_value_after",
            "portfolio_pnl_after",
            "portfolio_return_after",
        )
    }
    payload["ledger_sha256"] = ledger_sha256
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_v3_manual_financial_reconciliation(
    *,
    ledger: pd.DataFrame,
    manifest: Mapping[str, Any],
    expected_capital: float,
    ledger_sha256: str,
) -> dict[str, Any]:
    if "current_value" not in ledger.columns:
        raise RuntimeError("v3_manual_ledger_schema_missing:current_value")
    current_values = pd.to_numeric(ledger["current_value"], errors="coerce")
    if current_values.isna().any():
        raise RuntimeError("v3_manual_ledger_current_value_invalid")
    ledger_market_value = float(current_values.sum())
    cash_after = _required_manifest_number(manifest, "cash_after")
    market_value_after = _required_manifest_number(manifest, "market_value_after")
    total_value_after = _required_manifest_number(manifest, "total_value_after")
    portfolio_pnl_after = _required_manifest_number(manifest, "portfolio_pnl_after")
    portfolio_return_after = _required_manifest_number(
        manifest,
        "portfolio_return_after",
    )
    manifest_capital = _required_manifest_number(manifest, "capital_cny")
    declared_financial_sha256 = str(
        manifest.get("financial_state_sha256") or ""
    ).strip().lower()
    expected_financial_sha256 = _manual_financial_state_sha256(
        manifest,
        ledger_sha256=ledger_sha256,
    )
    if declared_financial_sha256 != expected_financial_sha256:
        raise RuntimeError("v3_manual_financial_state_sha256_mismatch")
    money_tolerance = 0.01
    if not math.isclose(
        ledger_market_value,
        market_value_after,
        rel_tol=0.0,
        abs_tol=money_tolerance,
    ):
        raise RuntimeError("v3_manual_ledger_market_value_mismatch")
    if not math.isclose(
        cash_after + market_value_after,
        total_value_after,
        rel_tol=0.0,
        abs_tol=money_tolerance,
    ):
        raise RuntimeError("v3_manual_total_value_identity_mismatch")
    implied_capital = total_value_after - portfolio_pnl_after
    if implied_capital <= 0:
        raise RuntimeError("v3_manual_implied_capital_invalid")
    if not math.isclose(
        manifest_capital,
        expected_capital,
        rel_tol=0.0,
        abs_tol=money_tolerance,
    ):
        raise RuntimeError("v3_manual_capital_vs_formal_manifest_mismatch")
    if not math.isclose(
        implied_capital,
        manifest_capital,
        rel_tol=0.0,
        abs_tol=money_tolerance,
    ):
        raise RuntimeError("v3_manual_capital_pnl_identity_mismatch")
    implied_return = portfolio_pnl_after / implied_capital
    if not math.isclose(
        portfolio_return_after,
        implied_return,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise RuntimeError("v3_manual_portfolio_return_identity_mismatch")
    return {
        "status": "verified",
        "ledger_market_value": round(ledger_market_value, 2),
        "cash_after": round(cash_after, 2),
        "market_value_after": round(market_value_after, 2),
        "total_value_after": round(total_value_after, 2),
        "portfolio_pnl_after": round(portfolio_pnl_after, 2),
        "portfolio_return_after": portfolio_return_after,
        "implied_capital": round(implied_capital, 2),
        "capital_cny": round(manifest_capital, 2),
        "financial_state_sha256": expected_financial_sha256,
    }


def _load_latest_effective_manual_record(
    base_dir: Path,
    max_record_name: str | None = None,
    expected_capital: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], Path, Path]:
    run_dirs = [
        path
        for path in base_dir.iterdir()
        if path.is_dir()
        and not path.name.startswith("_")
        and (max_record_name is None or path.name <= max_record_name)
    ]
    candidates: list[tuple[tuple[str, str], Path, dict[str, Any], Path]] = []
    validation_errors: list[str] = []
    for run_dir in run_dirs:
        manifest_path = run_dir / "manual_execution_manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest_raw = _stable_read_bytes(manifest_path)
            manifest = json.loads(manifest_raw.decode("utf-8"))
        except Exception as exc:
            validation_errors.append(f"{run_dir.name}:manifest_unreadable:{exc}")
            continue
        if not isinstance(manifest, dict):
            validation_errors.append(f"{run_dir.name}:manifest_not_mapping")
            continue
        if not _manual_manifest_is_valid_baseline(manifest):
            continue
        if str(manifest.get("schema_version") or "") == MANUAL_EXECUTION_SCHEMA_VERSION:
            explicit_time = _manual_manifest_explicit_datetime(manifest)
            if explicit_time is None:
                validation_errors.append(f"{run_dir.name}:v3_manifest_time_missing")
                continue
            now_utc = _now_local().astimezone(timezone.utc)
            if explicit_time > now_utc + timedelta(minutes=5):
                validation_errors.append(f"{run_dir.name}:v3_manifest_time_in_future")
                continue
        ledger_path = _resolve_manual_ledger_path(manifest_path, manifest)
        if ledger_path is None:
            validation_errors.append(f"{run_dir.name}:declared_next_ledger_invalid")
            continue
        candidates.append((_manual_manifest_order_key(run_dir, manifest), manifest_path, manifest, ledger_path))

    for _, manifest_path, manifest, ledger_path in sorted(
        candidates,
        key=lambda item: item[0],
        reverse=True,
    ):
        try:
            raw = _stable_read_bytes(ledger_path)
            actual_sha256 = hashlib.sha256(raw).hexdigest()
            declared_sha256 = _declared_manual_ledger_sha256(manifest, ledger_path)
            if str(manifest.get("schema_version") or "") == MANUAL_EXECUTION_SCHEMA_VERSION and not declared_sha256:
                raise RuntimeError("v3_manual_manifest_ledger_sha256_missing")
            if declared_sha256 and actual_sha256 != declared_sha256:
                raise RuntimeError("manual_ledger_sha256_mismatch")
            ledger = (
                pd.read_parquet(io.BytesIO(raw))
                if ledger_path.suffix.lower() == ".parquet"
                else pd.read_table(io.BytesIO(raw), sep=",", encoding="utf-8-sig")
            )
            required_columns = {"symbol", "shares", "avg_cost"}
            if not required_columns.issubset(set(ledger.columns)):
                raise RuntimeError(
                    "manual_ledger_schema_missing:"
                    + ",".join(sorted(required_columns - set(ledger.columns)))
                )
            declared_count = manifest.get("effective_manual_holding_count")
            if declared_count not in (None, "") and int(declared_count) != len(ledger):
                raise RuntimeError("manual_ledger_holding_count_mismatch")
            is_v3 = (
                str(manifest.get("schema_version") or "")
                == MANUAL_EXECUTION_SCHEMA_VERSION
            )
            if is_v3 and expected_capital is None:
                raise RuntimeError("v3_manual_expected_capital_missing")
            financial_reconciliation = (
                _validate_v3_manual_financial_reconciliation(
                    ledger=ledger,
                    manifest=manifest,
                    expected_capital=float(expected_capital),
                    ledger_sha256=actual_sha256,
                )
                if is_v3
                else {"status": "legacy_unsealed"}
            )
            manifest = {
                **manifest,
                "validated_ledger_sha256": actual_sha256,
                "validated_financial_reconciliation": financial_reconciliation,
                "validated_ledger_provenance": {
                    "declared_next_ledger_path": str(manifest.get("next_ledger_path") or ""),
                    "resolved_path": str(ledger_path),
                    "contained_in_manifest_directory": True,
                    "regular_non_symlink_file": True,
                    "stable_double_read": True,
                    "declared_sha256": declared_sha256,
                    "actual_sha256": actual_sha256,
                    "hash_status": "verified" if declared_sha256 else "legacy_unsealed",
                    "financial_reconciliation_status": financial_reconciliation[
                        "status"
                    ],
                },
            }
            return ledger, manifest, manifest_path, ledger_path
        except Exception as exc:
            validation_errors.append(f"{manifest_path.parent.name}:ledger_invalid:{exc}")
            continue

    raise RuntimeError(
        "策略目录下不存在有效本地/manual ledger_after_manual_switch sidecar；"
        "formal ledger.csv 已停用，无法作为连续复盘或执行基线。"
        + (" blockers=" + " | ".join(validation_errors[-5:]) if validation_errors else "")
    )


def _pnl_summary_path(record_dir: Path) -> Path | None:
    parquet_path = record_dir / "pnl_summary.parquet"
    if parquet_path.exists():
        return parquet_path
    return None


def _read_pnl_summary(path: Path) -> pd.DataFrame:
    if path.suffix.lower() != ".parquet":
        raise RuntimeError(f"pnl_summary 只允许读取 Parquet sidecar: {path}")
    return pd.read_parquet(path)


def _load_previous_record(
    base_dir: Path,
    source_record: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    run_dirs = [
        path
        for path in base_dir.iterdir()
        if path.is_dir()
        and not path.name.startswith("_")
        and (path / "manifest.json").exists()
        and _pnl_summary_path(path) is not None
    ]
    if not run_dirs:
        raise RuntimeError("策略目录下不存在上一条正式记录，无法做连续复盘。")

    if source_record:
        latest_dir = base_dir / source_record
        if not latest_dir.exists():
            raise RuntimeError(f"指定的 source_record 不存在: {source_record}")
        missing_files = []
        if not (latest_dir / "manifest.json").exists():
            missing_files.append("manifest.json")
        if _pnl_summary_path(latest_dir) is None:
            missing_files.append("pnl_summary.parquet")
        if missing_files:
            raise RuntimeError(
                f"指定的 source_record 缺失正式记录文件: {', '.join(missing_files)} ({source_record})"
            )
    else:
        latest_dir = sorted(run_dirs, key=lambda path: path.name)[-1]
    manifest_path = latest_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pnl_summary = _read_pnl_summary(_pnl_summary_path(latest_dir))
    ledger, manual_manifest, manual_manifest_path, manual_ledger_path = _load_latest_effective_manual_record(
        base_dir,
        max_record_name=latest_dir.name,
        expected_capital=_safe_float(
            manifest.get("capital_cny"),
            DEFAULT_INITIAL_CAPITAL,
        ),
    )
    manifest = {
        **manifest,
        "formal_source_record": latest_dir.name,
        "effective_manual_manifest_path": str(manual_manifest_path),
        "effective_manual_ledger_path": str(manual_ledger_path),
        "effective_manual_status": str(
            manual_manifest.get("status")
            or manual_manifest.get("execution_status")
            or ""
        ),
        "effective_manual_ledger_sha256": str(
            manual_manifest.get("validated_ledger_sha256") or ""
        ),
        "effective_manual_ledger_provenance": _jsonable(
            manual_manifest.get("validated_ledger_provenance") or {}
        ),
    }
    pnl_summary = _build_effective_manual_pnl_summary(
        manual_manifest=manual_manifest,
        fallback_pnl_summary=pnl_summary,
    )
    return ledger, manifest, pnl_summary


def _allocate_run_timestamp(base_dir: Path, now: datetime) -> str:
    base = now.strftime("%Y%m%d_%H%M")
    candidate = base
    counter = 1
    while (base_dir / candidate).exists():
        candidate = f"{base}_{counter:02d}"
        counter += 1
    return candidate


def _format_symbol_set(symbols: list[str]) -> str:
    return " / ".join(symbols)


def _format_holding_snapshot_set(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "无"
    return " / ".join(_format_holding_snapshot(row) for row in frame.itertuples())


def _format_holding_snapshot(row: Any) -> str:
    buy_price = _safe_float(
        getattr(row, "buy_price", getattr(row, "avg_cost", 0.0)),
        0.0,
    )
    unrealized_pnl = _safe_float(getattr(row, "unrealized_pnl", 0.0), 0.0)
    unrealized_pnl_pct = _safe_float(getattr(row, "unrealized_pnl_pct", 0.0), 0.0)
    return (
        f"{row.symbol}({row.name}) 持有成本 `{buy_price:.2f}`，"
        f"PNL `{_format_signed_money(unrealized_pnl)}`（{unrealized_pnl_pct:+.2%}）"
    )


def _format_top_holdings_by_unrealized_pnl(frame: pd.DataFrame, *, positive: bool) -> str:
    if frame.empty or "unrealized_pnl" not in frame.columns:
        return "无"

    filtered = frame[frame["unrealized_pnl"] > 0] if positive else frame[frame["unrealized_pnl"] < 0]
    if filtered.empty:
        return "无"

    ordered = filtered.sort_values("unrealized_pnl", ascending=not positive)
    return "；".join(
        f"{row.symbol}({row.name}) {row.unrealized_pnl:,.2f} 元"
        for row in ordered.head(3).itertuples()
    )


def _format_top_delta_vs_source_record(frame: pd.DataFrame, *, positive: bool) -> str:
    if frame.empty or "delta_vs_source_record" not in frame.columns:
        return "无"

    filtered = (
        frame[frame["delta_vs_source_record"] > 0]
        if positive
        else frame[frame["delta_vs_source_record"] < 0]
    )
    if filtered.empty:
        return "无"

    ordered = filtered.sort_values("delta_vs_source_record", ascending=not positive)
    return "；".join(
        f"{row.symbol} {row.delta_vs_source_record:,.2f} 元"
        for row in ordered.head(3).itertuples()
    )


def _format_signed_money(value: float) -> str:
    return f"{value:+,.2f} 元"


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _format_legacy_overweight_holding_lines(
    holdings_review: pd.DataFrame,
    *,
    cap: float | None = None,
) -> list[str]:
    if not isinstance(holdings_review, pd.DataFrame) or holdings_review.empty:
        return []
    if "nav_weight" not in holdings_review.columns:
        return []
    limit = risk_guard_single_name_weight_cap() if cap is None else float(cap)
    if limit <= 0.0:
        return []

    rows: list[tuple[str, str, float]] = []
    for _, row in holdings_review.iterrows():
        weight = _safe_float(row.get("nav_weight"), 0.0)
        if weight <= limit + 1e-9:
            continue
        symbol = str(row.get("symbol") or "").strip() or "UNKNOWN"
        name = str(row.get("name") or "").strip()
        rows.append((symbol, name, weight))

    if not rows:
        return []
    rows.sort(key=lambda item: (-item[2], item[0]))
    return [
        (
            f"- 超限存量持仓提示：`{symbol}`"
            + (f"（{name}）" if name else "")
            + (
                f"当前权重 {weight:.2%}，上限 {limit:.2%}；"
                "只提示，不强制卖出。建议路径：暂停新增买入，优先用新增资金与自然回撤稀释；"
                "若后续确认减仓，则分批降至上限。"
            )
        )
        for symbol, name, weight in rows
    ]


def _codex_rating_from_score(score: int) -> str:
    if score >= 80:
        return "强"
    if score >= 70:
        return "偏强"
    if score >= 60:
        return "观察"
    return "偏弱"


def _holding_codex_score(row: pd.Series) -> int:
    rank = int(_safe_float(row.get("rank_full_market"), 9999.0))
    today_change = _safe_float(row.get("today_change_pct"), 0.0)
    ret5 = _safe_float(row.get("ret5"), 0.0)
    ret20 = _safe_float(row.get("ret20"), 0.0)
    ret60 = _safe_float(row.get("ret60"), 0.0)
    current_price = _safe_float(row.get("current_price"), 0.0)
    stop_price = _safe_float(row.get("stage_stop_price"), 0.0)
    target_price = _safe_float(row.get("stage_target_price"), 0.0)
    market_weight = _safe_float(row.get("nav_weight"), _safe_float(row.get("market_weight"), 0.0))
    realtime_quote_valid = bool(row.get("realtime_quote_valid"))
    llm_degraded = bool(row.get("llm_degraded"))
    recommended_action = str(row.get("recommended_action") or "").strip()

    strength_points = 0.0
    if rank > 0 and rank < 9999:
        strength_points = 20.0 * (1.0 - _clamp((rank - 1.0) / 499.0, 0.0, 1.0))
    score_full_market = _safe_float(row.get("score_full_market"), 0.0)
    if strength_points <= 0.0:
        strength_points = 8.0 + 8.0 * _clamp(score_full_market, 0.0, 1.0)

    trend_signal = (
        _clamp(today_change / 8.0, -1.0, 1.0)
        + _clamp(ret5 / 0.15, -1.0, 1.0)
        + _clamp(ret20 / 0.35, -1.0, 1.0)
        + _clamp(ret60 / 0.60, -1.0, 1.0)
    ) / 4.0
    trend_points = 25.0 * (trend_signal + 1.0) / 2.0

    stop_buffer = _safe_pct(current_price - stop_price, stop_price) if stop_price > 0 else 0.0
    target_gap = _safe_pct(target_price - current_price, current_price) if current_price > 0 else 0.0
    risk_points = 0.0
    risk_points += 10.0 * _clamp((stop_buffer + 0.10) / 0.35, 0.0, 1.0)
    risk_points += 10.0 * _clamp((target_gap + 0.05) / 0.25, 0.0, 1.0)

    evidence_points = 10.0 if realtime_quote_valid else 5.0
    if llm_degraded:
        evidence_points -= 2.0

    concentration_penalty = 6.0 * _clamp((market_weight - 0.18) / 0.12, 0.0, 1.0)
    portfolio_fit_points = 10.0 - concentration_penalty

    action_points_map = {
        "继续持有": 15.0,
        "移动止盈观察": 12.0,
        "继续观察": 10.0,
        "减仓待确认": 5.0,
        "清仓待确认": 2.0,
    }
    action_points = action_points_map.get(recommended_action, 8.0)

    total = strength_points + trend_points + risk_points + evidence_points + portfolio_fit_points + action_points
    return int(round(_clamp(total, 0.0, 100.0)))


def _candidate_codex_score(row: pd.Series) -> int:
    posterior_action_score = _safe_float(row.get("posterior_action_score"), 0.0)
    posterior_confidence = _safe_float(row.get("posterior_confidence"), 0.0)
    expected_upside = _safe_float(row.get("expected_upside"), 0.0)
    portfolio_target_weight = _safe_float(row.get("portfolio_target_weight"), 0.0)
    branch_complete = bool(row.get("candidate_dag_four_branch_complete"))
    evidence_quality = str(row.get("evidence_quality") or "").strip()
    risk_flags = str(row.get("risk_flags") or "").strip()

    score = 0.0
    score += 35.0 * _clamp(posterior_action_score / 0.65, 0.0, 1.0)
    score += 20.0 * _clamp(posterior_confidence / 0.70, 0.0, 1.0)
    score += 15.0 * _clamp(expected_upside / 0.03, 0.0, 1.0)
    score += 10.0 * _clamp(portfolio_target_weight / 0.10, 0.0, 1.0)
    score += 10.0 if branch_complete else 0.0
    score += {"高": 10.0, "中": 6.0}.get(evidence_quality, 3.0)
    if risk_flags:
        score -= min(8.0, 2.0 * risk_flags.count("；"))
    return int(round(_clamp(score, 0.0, 100.0)))


def _format_holding_advice_line(row: Any) -> str:
    price = _safe_float(getattr(row, "current_price", 0.0), 0.0)
    buy_price = _safe_float(getattr(row, "buy_price", 0.0), 0.0)
    buy_value = _safe_float(getattr(row, "buy_value", 0.0), 0.0)
    unrealized_pnl = _safe_float(getattr(row, "unrealized_pnl", 0.0), 0.0)
    unrealized_pnl_pct = _safe_float(getattr(row, "unrealized_pnl_pct", 0.0), 0.0)
    stop_price = _safe_float(getattr(row, "stage_stop_price", 0.0), 0.0)
    target_price = _safe_float(getattr(row, "stage_target_price", 0.0), 0.0)
    stop_buffer = _safe_pct(price - stop_price, stop_price)
    hard_signal = "未触发阶段止损" if price >= stop_price else "低于阶段止损，需跟踪减仓确认"
    trailing_status = str(getattr(row, "trailing_take_profit_status", "") or "unconfirmed")
    trailing_peak = _safe_float(getattr(row, "trailing_profit_peak_price", 0.0), 0.0)
    trailing_stop = _safe_float(getattr(row, "trailing_stop_price", 0.0), 0.0)
    trailing_giveback = _safe_float(getattr(row, "profit_giveback_ratio", 0.0), 0.0)
    trailing_basis = str(getattr(row, "trailing_take_profit_basis", "") or "")
    trailing_text = (
        f"移动止盈 `{trailing_status}`，峰值 `{trailing_peak:.2f}`，"
        f"复核/止盈价 `{trailing_stop:.2f}`，利润回吐 `{trailing_giveback:.2%}`"
        + ("（峰值未锚定买入日）" if "unanchored" in trailing_basis else "")
    )
    codex_score = int(_safe_float(getattr(row, "codex_recommendation_score", 0), 0.0))
    codex_rating = str(getattr(row, "codex_recommendation_rating", "") or _codex_rating_from_score(codex_score))
    return (
        f"- `{row.symbol}`（{row.name}）：建议 `{row.recommended_action}`，"
        f"持仓角色 `{row.position_role}`；Codex 评分 `{codex_score}/100`（`{codex_rating}`）；当前价 `{price:.2f}`，"
        f"持有成本 `{buy_price:.2f}`（成本金额 `{buy_value:,.2f} 元`），"
        f"阶段止损 `{stop_price:.2f}`（缓冲 {stop_buffer:+.2%}），"
        f"阶段目标 `{target_price:.2f}`；浮动 PNL `{_format_signed_money(unrealized_pnl)}`"
        f"（{unrealized_pnl_pct:+.2%}），"
        f"较上一条记录 `{_format_signed_money(float(row.delta_vs_source_record))}`；"
        f"全市场强度排名 `{int(row.rank_full_market)}`，今日涨跌 `{float(row.today_change_pct):+.2f}%`；"
        f"{trailing_text}；"
        f"{hard_signal}。"
    )


def _format_candidate_advice_line(row: Any, switch_row: dict[str, Any] | None) -> str:
    source_label = "全市场 v13 DAG"
    relative_advantage = (
        f"相对 `{switch_row['sell_symbol']}`，候选由 PortfolioConstructor 给出目标权重 "
        f"`{float(switch_row['buy_portfolio_target_weight']):.2%}`，"
        f"Bayesian action score `{float(switch_row['buy_posterior_action_score']):.3f}`；"
        if switch_row
        else (
            f"Bayesian rank `{int(row.bayesian_rank)}`，PortfolioConstructor 目标权重 "
            f"`{float(row.portfolio_target_weight):.2%}`；"
        )
    )
    risk_flags = str(getattr(row, "risk_flags", "") or "").strip()
    if str(row.evidence_quality) == "高":
        major_risk = f"{risk_flags or '若候选 DAG 任一分支转弱，必须降级出正式候选池'}；"
    elif str(row.evidence_quality) == "中":
        major_risk = "盘中采用前一交易日稳定日线结合实时行情，收盘后需用当日日线复核；"
    else:
        major_risk = "本地 strict 快照仍有缺口，结论依赖主导本地快照延续性；"
    trigger = (
        str(switch_row["trigger_threshold"])
        if switch_row
        else "若下一轮仍保持 candidate-level v13 DAG 完整且 PortfolioConstructor 正目标权重，可继续观察或准备换仓"
    )
    codex_score = int(_safe_float(getattr(row, "codex_recommendation_score", 0), 0.0))
    codex_rating = str(getattr(row, "codex_recommendation_rating", "") or _codex_rating_from_score(codex_score))
    return (
        f"- `{row.symbol}`（{row.name}）：主线 `{row.theme_label}`，来源 `{source_label}`；"
        f"Codex 评分 `{codex_score}/100`（`{codex_rating}`）；"
        f"{relative_advantage}主要风险：{major_risk}"
        f"触发条件：{trigger}；证据质量：`{row.evidence_quality}`。"
    )


def _format_warning_count_summary(warnings: list[Any]) -> str:
    if not warnings:
        return "无"
    counts = Counter(str(warning.code) for warning in warnings)
    return "，".join(f"{code}={count}" for code, count in sorted(counts.items()))


def _format_warning_messages(warnings: list[Any], *, severity: str | None = None, limit: int = 6) -> str:
    selected = [
        warning
        for warning in warnings
        if severity is None or str(getattr(warning, "severity", "")) == severity
    ]
    if not selected:
        return "无"
    messages = [str(warning.human_message).strip() for warning in selected if str(warning.human_message).strip()]
    return "；".join(messages[:limit]) + (" ..." if len(messages) > limit else "")


def _format_trade_date(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text or "未知"


def _compact_trade_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "nat", "none", "null", "<na>"}:
        return ""
    if len(text) == 8 and text.isdigit():
        return text
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else text.replace("-", "")


def _normalize_review_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def build_parquet_canonical_completeness_report(
    *,
    reader: Any,
    components: dict[str, Any],
    categories: list[str] | tuple[str, ...] | None = None,
    allowed_stale_symbols: list[str] | tuple[str, ...] | set[str] | None = None,
    target_trade_date: str | None = None,
    early_stop_reason: str = "",
) -> dict[str, Any]:
    """Build the formal-review completeness payload from strict canonical Parquet."""

    snapshot = dict(reader.snapshot() or {})
    if snapshot.get("healthy") is False:
        blockers = [str(item) for item in snapshot.get("blockers", []) if str(item).strip()]
        raise RuntimeError("; ".join(blockers) or "strict Parquet snapshot is not healthy")

    effective_trade_date = _compact_trade_date(
        target_trade_date
        or snapshot.get("latest_complete_trade_date")
        or snapshot.get("latest_trade_date")
    )
    if not effective_trade_date:
        raise RuntimeError("strict Parquet latest_complete_trade_date is missing")

    cross_section = reader.read_cross_section(
        effective_trade_date,
        universe_key="full_a",
        columns=["ts_code", "trade_date"],
    )
    symbol_column = (
        "symbol"
        if "symbol" in cross_section.columns
        else "ts_code"
        if "ts_code" in cross_section.columns
        else ""
    )
    available_symbols = {
        _normalize_review_symbol(symbol)
        for symbol in (cross_section[symbol_column].tolist() if symbol_column else [])
        if _normalize_review_symbol(symbol)
    }
    allowed = {
        _normalize_review_symbol(symbol)
        for symbol in (allowed_stale_symbols or [])
        if _normalize_review_symbol(symbol)
    }
    coverage_payload = dict(snapshot.get("coverage", {}) or {})
    coverage_provenance_blockers = [
        str(item)
        for item in list(snapshot.get("coverage_provenance_blockers", []) or [])
        if str(item).strip()
    ]
    inactive_symbols = {
        _normalize_review_symbol(symbol)
        for symbol in coverage_payload.get("inactive_symbols", []) or []
        if _normalize_review_symbol(symbol)
    }
    suspended_symbols = {
        _normalize_review_symbol(symbol)
        for symbol in coverage_payload.get("suspended_symbols", []) or []
        if _normalize_review_symbol(symbol)
    }
    persisted_allowed_symbols = {
        _normalize_review_symbol(symbol)
        for symbol in coverage_payload.get("allowed_stale_symbols", []) or []
        if _normalize_review_symbol(symbol)
    }
    persisted_non_blocking_absent = {
        _normalize_review_symbol(symbol)
        for symbol in coverage_payload.get("non_blocking_absent_symbols", []) or []
        if _normalize_review_symbol(symbol)
    }
    full_a_symbols = {
        _normalize_review_symbol(symbol)
        for symbol in components.get("full_a", []) or []
        if _normalize_review_symbol(symbol)
    }
    expected_scope_sha256 = hashlib.sha256(
        "\n".join(sorted(full_a_symbols)).encode("utf-8")
    ).hexdigest()
    declared_scope_sha256 = str(
        coverage_payload.get("expected_scope_sha256") or ""
    ).strip().lower()
    coverage_trade_date = _compact_trade_date(
        coverage_payload.get("coverage_trade_date")
        or coverage_payload.get("latest_complete_trade_date")
    )
    coverage_claim_complete = bool(
        coverage_payload.get("complete") is True
        and int(coverage_payload.get("blocking_incomplete_count", 0) or 0) == 0
        and int(coverage_payload.get("expected_scope_count", 0) or 0)
        == len(full_a_symbols)
        and coverage_trade_date == effective_trade_date
    )
    canonical_full_a_complete = bool(
        coverage_claim_complete
        and not coverage_provenance_blockers
        and declared_scope_sha256
        and declared_scope_sha256 == expected_scope_sha256
    )
    coverage_provenance_status = (
        "verified"
        if canonical_full_a_complete
        else "legacy_unbound"
        if coverage_claim_complete and not declared_scope_sha256
        else "unverified"
    )
    target_categories = list(categories or ["full_a", "hs300", "zz500", "zz1000"])

    report: dict[str, Any] = {
        "allowed_stale_symbols": sorted(allowed),
        "complete": True,
        "blocking_incomplete_count": 0,
        "categories_checked": target_categories,
        "categories": {},
        "data_quality_issues": [],
        "pre_listing_symbols": [],
        "coverage_provenance_status": coverage_provenance_status,
        "coverage_trade_date": coverage_trade_date,
        "expected_scope_sha256": expected_scope_sha256,
        "declared_expected_scope_sha256": declared_scope_sha256,
        "coverage_provenance_blockers": coverage_provenance_blockers,
        "source": "strict_parquet_canonical",
        "snapshot_id": str(snapshot.get("snapshot_id") or ""),
        "latest_complete_trade_date": effective_trade_date,
    }
    unique_expected_symbols: set[str] = set()
    unique_complete_symbols: set[str] = set()
    unique_blocking_symbols: set[str] = set()
    verified_inactive_symbols = inactive_symbols if canonical_full_a_complete else set()
    verified_suspended_symbols = suspended_symbols if canonical_full_a_complete else set()
    verified_persisted_allowed_symbols = (
        persisted_allowed_symbols if canonical_full_a_complete else set()
    )
    verified_persisted_non_blocking_absent = (
        persisted_non_blocking_absent if canonical_full_a_complete else set()
    )

    for category in target_categories:
        category_symbols = [
            _normalize_review_symbol(symbol)
            for symbol in components.get(category, []) or []
            if _normalize_review_symbol(symbol)
        ]
        category_symbols = list(dict.fromkeys(category_symbols))
        target_set = set(category_symbols)
        unique_expected_symbols.update(target_set)
        present = target_set & available_symbols
        declared_non_blocking_absent = (
            allowed
            | verified_persisted_allowed_symbols
            | verified_inactive_symbols
            | verified_suspended_symbols
            | verified_persisted_non_blocking_absent
        )
        canonical_non_trading_absent = set()
        if canonical_full_a_complete and target_set.issubset(full_a_symbols):
            canonical_non_trading_absent = target_set - present
        non_blocking_absent = (
            (declared_non_blocking_absent | canonical_non_trading_absent)
            & target_set
        ) - present
        missing = sorted(target_set - present - non_blocking_absent)
        category_complete_count = len(present) + len(non_blocking_absent)
        unique_complete_symbols.update(present | non_blocking_absent)
        expected_count = len(category_symbols)
        blocking_count = len(missing)
        if blocking_count:
            report["complete"] = False
            unique_blocking_symbols.update(missing)

        status_counts: dict[str, int] = {}
        if present:
            status_counts["up_to_date"] = len(present)
        if non_blocking_absent:
            status_counts["non_trading_or_allowed"] = len(non_blocking_absent)
        if missing:
            status_counts["missing"] = len(missing)
        report["categories"][category] = {
            "expected": expected_count,
            "latest_trade_date": effective_trade_date,
            "pre_listing_symbols": [],
            "date_counts": {effective_trade_date: len(present)} if present else {},
            "status_counts": status_counts,
            "missing_symbols": missing,
            "stale_symbols": [],
            "suspended_stale_symbols": sorted(
                verified_suspended_symbols & target_set
            ),
            "inactive_symbols": sorted(verified_inactive_symbols & target_set),
            "canonical_non_trading_absent_symbols": sorted(
                canonical_non_trading_absent
            ),
            "unreadable_symbols": [],
            "blocking_missing_symbols": missing,
            "blocking_stale_symbols": [],
            "blocking_unreadable_symbols": [],
            "blocking_incomplete_count": blocking_count,
            "coverage_complete_count": category_complete_count,
            "coverage_ratio": (
                category_complete_count / expected_count
                if expected_count
                else 1.0
            ),
        }

    coverage_complete_count = len(unique_complete_symbols)
    expected_scope_count = len(unique_expected_symbols)
    report["blocking_incomplete_count"] = len(unique_blocking_symbols)
    coverage_ratio = (
        coverage_complete_count / expected_scope_count
        if expected_scope_count
        else 1.0
    )
    report.update(
        {
            "latest_trade_date": effective_trade_date,
            "strict_trade_date": effective_trade_date,
            "stable_trade_date": effective_trade_date,
            "effective_target_trade_date": effective_trade_date,
            "freshness_mode": "strict_parquet_canonical",
            "coverage_ratio": coverage_ratio,
            "coverage_complete_count": coverage_complete_count,
            "expected_scope_count": expected_scope_count,
            "coverage_threshold": 0.95,
            "early_stop_reason": early_stop_reason,
            "resolver": {
                "resolution_strategy": "strict_parquet_canonical",
                "snapshot_id": str(snapshot.get("snapshot_id") or ""),
                "latest_complete_trade_date": effective_trade_date,
                "table_root": str(snapshot.get("table_root") or ""),
                "serving_root": str(snapshot.get("serving_root") or ""),
                "manifest_path": str(snapshot.get("manifest_path") or ""),
                "latest_pointer_path": str(snapshot.get("latest_pointer_path") or ""),
                "physical_directories_used_for_full_a": [],
                "local_union_fallback_used": False,
                "metadata": {
                    "source": "strict_parquet_canonical",
                    "fallback_used": False,
                },
            },
            "data_quality_issue_count": 0,
        }
    )
    return report


def _dominant_trade_date(date_counts: dict[str, Any]) -> str:
    normalized = {
        str(date).strip(): int(count or 0)
        for date, count in (date_counts or {}).items()
        if str(date).strip()
    }
    if not normalized:
        return ""
    return max(normalized.items(), key=lambda item: (item[1], item[0]))[0]


def _resolve_analysis_trade_date(completeness_report: dict[str, Any]) -> str:
    categories = dict(completeness_report.get("categories", {}) or {})
    full_a = dict(categories.get("full_a", {}) or {})
    dominant = _dominant_trade_date(full_a.get("date_counts", {}) or {})
    if dominant:
        return dominant

    aggregate_counts: dict[str, int] = {}
    for payload in categories.values():
        for date, count in (payload.get("date_counts", {}) or {}).items():
            key = str(date).strip()
            if not key:
                continue
            aggregate_counts[key] = aggregate_counts.get(key, 0) + int(count or 0)
    dominant = _dominant_trade_date(aggregate_counts)
    if dominant:
        return dominant
    return str(
        completeness_report.get("effective_target_trade_date")
        or completeness_report.get("latest_trade_date")
        or ""
    )


def _build_data_status_summary(
    completeness_report: dict[str, Any],
    analysis_trade_date: str,
    decision_data_sufficient: bool = False,
) -> str:
    target_trade_date = _format_trade_date(completeness_report.get("latest_trade_date"))
    effective_target_trade_date = _format_trade_date(
        completeness_report.get("effective_target_trade_date")
        or completeness_report.get("latest_trade_date")
    )
    strict_trade_date = _format_trade_date(
        completeness_report.get("strict_trade_date")
        or completeness_report.get("latest_trade_date")
    )
    freshness_mode = str(completeness_report.get("freshness_mode") or "strict")
    blocking_count = int(completeness_report.get("blocking_incomplete_count", 0) or 0)
    coverage_ratio = float(completeness_report.get("coverage_ratio", 0.0) or 0.0)
    pre_listing_count = len(completeness_report.get("pre_listing_symbols", []) or [])
    full_a = (
        completeness_report.get("categories", {})
        .get("full_a", {})
        or {}
    )
    dominant_count = int((full_a.get("date_counts", {}) or {}).get(analysis_trade_date, 0) or 0)
    dominant_expected = int(full_a.get("expected", 0) or 0)
    suspended_stale_count = len(
        (
            completeness_report.get("categories", {})
            .get("full_a", {})
            .get("suspended_stale_symbols", [])
            or []
        )
    )

    base = f"本地主导A股日线快照位于 `{_format_trade_date(analysis_trade_date)}`"
    if _format_trade_date(analysis_trade_date) != effective_target_trade_date:
        base += (
            f"，而当前完整性目标交易日为 `{effective_target_trade_date}`"
            f"（strict 最新 `{strict_trade_date}` / 报告 latest `{target_trade_date}`）"
        )
    else:
        base += f"，当前完整性目标交易日同样为 `{effective_target_trade_date}`"
    extras = [
        f"主导快照覆盖 `{dominant_count}/{dominant_expected}`",
        f"覆盖率 `{coverage_ratio:.1%}`",
        f"停牌/长期停牌例外 `{suspended_stale_count}` 个",
        f"预上市样本 `{pre_listing_count}` 个",
        f"完整性模式 `{freshness_mode}`",
    ]
    if blocking_count > 0:
        extras.insert(0, f"阻塞缺口 `{blocking_count}` 个")
    else:
        extras.insert(0, "阻塞缺口 `0` 个")
    suffix = "（" + "；".join(extras) + "）"
    if decision_data_sufficient and _format_trade_date(analysis_trade_date) != effective_target_trade_date:
        suffix += "；盘中决策口径接受前一交易日稳定日线结合实时行情，当日 strict 日线未出不视为决策阻断。"
    return base + suffix


def _build_data_snapshot_lines(
    completeness_report: dict[str, Any],
    quote_snapshot: str,
    analysis_trade_date: str,
    decision_data_sufficient: bool = False,
) -> list[str]:
    def _symbol_date_label(item: Any) -> str:
        if isinstance(item, Mapping):
            symbol = str(item.get("symbol") or "")
            date_value = item.get("latest_local_date")
            return (
                f"{symbol}@{_format_trade_date(date_value)}"
                if date_value
                else symbol
            )
        return str(item)

    categories = dict(completeness_report.get("categories", {}) or {})
    full_a = dict(categories.get("full_a", {}) or {})
    analysis_count = int((full_a.get("date_counts", {}) or {}).get(analysis_trade_date, 0) or 0)
    lines = [
        (
            f"- 分析采用的主导本地快照：`{_format_trade_date(analysis_trade_date)}`，"
            f"`full_a` 覆盖 `{analysis_count}/{int(full_a.get('expected', 0) or 0)}`"
        ),
        f"- 本地最新交易日：`{_format_trade_date(completeness_report.get('latest_trade_date'))}`",
        f"- strict 最新交易日：`{_format_trade_date(completeness_report.get('strict_trade_date'))}`",
        f"- stable 最新交易日：`{_format_trade_date(completeness_report.get('stable_trade_date'))}`",
        f"- 当前采用目标交易日：`{_format_trade_date(completeness_report.get('effective_target_trade_date'))}`",
        f"- 新鲜度模式：`{completeness_report.get('freshness_mode') or 'strict'}`",
        (
            f"- 覆盖完成度：`{int(completeness_report.get('coverage_complete_count', 0) or 0)}/"
            f"{int(completeness_report.get('expected_scope_count', 0) or 0)}` "
            f"（{float(completeness_report.get('coverage_ratio', 0.0) or 0.0):.1%}）"
        ),
        (
            f"- 完整性状态：`{'通过' if completeness_report.get('complete') else '未通过'}`，"
            f"阻塞缺口 `{int(completeness_report.get('blocking_incomplete_count', 0) or 0)}` 个"
        ),
        (
            f"- 覆盖证据绑定：`{completeness_report.get('coverage_provenance_status') or 'unverified'}`；"
            f"coverage_trade_date=`{completeness_report.get('coverage_trade_date') or 'missing'}`；"
            f"scope_sha=`{completeness_report.get('declared_expected_scope_sha256') or 'missing'}`；"
            f"blockers=`{','.join(completeness_report.get('coverage_provenance_blockers', []) or []) or 'none'}`"
        ),
        (
            f"- 指数/持仓盘中快照：`{quote_snapshot}`；指数与持仓现价使用盘中行情，"
            "广度、主题与全市场强弱仍以本地最新日线快照为准。"
            if quote_snapshot
            else "- 指数/持仓盘中快照：`N/A`，本轮全部结论基于本地日线快照。"
        ),
    ]
    if decision_data_sufficient:
        lines.append(
            "- 决策数据口径：`盘中前日线+实时行情可用`；当日 strict 日线缺口仅作数据披露，"
            "不自动降级正式投资结论。"
        )

    pre_listing = list(completeness_report.get("pre_listing_symbols", []) or [])
    if pre_listing:
        lines.append(
            "- 预上市样本：`"
            + "，".join(
                f"{item.get('symbol')}@{_format_trade_date(item.get('list_date'))}"
                for item in pre_listing[:8]
            )
            + (" ...`" if len(pre_listing) > 8 else "`")
        )

    for category in completeness_report.get("categories_checked", []):
        payload = categories.get(category, {})
        if not payload:
            continue
        category_dominant_trade_date = _dominant_trade_date(payload.get("date_counts", {}) or {})
        category_dominant_count = int((payload.get("date_counts", {}) or {}).get(category_dominant_trade_date, 0) or 0)
        lines.append(
            f"- {category}：目标 `{_format_trade_date(payload.get('latest_trade_date'))}`，"
            f"主导本地快照 `{_format_trade_date(category_dominant_trade_date)}` "
            f"（`{category_dominant_count}/{int(payload.get('expected', 0) or 0)}`），"
            f"目标覆盖 `{int(payload.get('coverage_complete_count', 0) or 0)}/"
            f"{int(payload.get('expected', 0) or 0)}`，"
            f"阻塞缺口 `{int(payload.get('blocking_incomplete_count', 0) or 0)}`，"
            f"停牌例外 `{len(payload.get('suspended_stale_symbols', []) or [])}`"
        )
        blocking_missing = list(payload.get("blocking_missing_symbols", []) or [])
        if blocking_missing:
            lines.append(
                "- "
                + category
                + " 阻塞缺失：`"
                + "，".join(blocking_missing[:10])
                + (" ...`" if len(blocking_missing) > 10 else "`")
            )
        blocking_stale = list(payload.get("blocking_stale_symbols", []) or [])
        if blocking_stale:
            lines.append(
                "- "
                + category
                + " 阻塞旧档：`"
                + "，".join(
                    _symbol_date_label(item) for item in blocking_stale[:10]
                )
                + (" ...`" if len(blocking_stale) > 10 else "`")
            )
        suspended = list(payload.get("suspended_stale_symbols", []) or [])
        if suspended:
            lines.append(
                "- "
                + category
                + " 停牌/长期停牌例外：`"
                + "，".join(
                    _symbol_date_label(item) for item in suspended[:10]
                )
                + (" ...`" if len(suspended) > 10 else "`")
            )
    return lines


def _write_outputs(
    base_dir: Path,
    run_dir: Path,
    report_text: str,
    holdings_review: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    switch_plan_df: pd.DataFrame,
    ledger: pd.DataFrame,
    orders_df: pd.DataFrame,
    pnl_summary_df: pd.DataFrame,
    manifest: dict[str, Any],
    market_snapshot: dict[str, Any],
    theme_pool_audit: dict[str, Any] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = run_dir / "raw_exports"
    raw_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"aggressive_portfolio_{manifest['timestamp']}_formal"
    report_path = run_dir / "analysis_report.md"
    holdings_path = run_dir / "holdings_review.csv"
    candidate_path = run_dir / "candidate_pool.csv"
    switch_path = run_dir / "switch_plan.csv"
    orders_path = run_dir / "orders.csv"
    pnl_path = run_dir / "pnl_summary.csv"
    pnl_parquet_path = run_dir / "pnl_summary.parquet"
    snapshot_path = run_dir / "market_snapshot.json"
    manifest_path = run_dir / "manifest.json"
    runtime_profile_path = run_dir / "runtime_profile.json"
    theme_pool_audit_path = run_dir / "theme_pool_audit.json"

    report_path.write_text(report_text, encoding="utf-8")
    holdings_review.to_csv(holdings_path, index=False, encoding="utf-8-sig")
    candidate_pool.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    switch_plan_df.to_csv(switch_path, index=False, encoding="utf-8-sig")
    orders_df.to_csv(orders_path, index=False, encoding="utf-8-sig")
    pnl_summary_df.to_csv(pnl_path, index=False, encoding="utf-8-sig")
    pnl_summary_df.to_parquet(pnl_parquet_path, index=False)
    snapshot_path.write_text(json.dumps(market_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    runtime_profile = manifest.get("runtime_profile") or market_snapshot.get("runtime_profile") or {}
    if runtime_profile:
        runtime_profile_path.write_text(
            json.dumps(runtime_profile, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if theme_pool_audit:
        files = manifest.setdefault("files", {})
        if isinstance(files, dict):
            files["theme_pool_audit"] = "theme_pool_audit.json"
        raw_name = f"{prefix}_theme_pool_audit.json"
        raw_exports = manifest.setdefault("raw_exports", {})
        if isinstance(raw_exports, dict):
            raw_exports["theme_pool_audit"] = f"raw_exports/{raw_name}"
        theme_pool_audit_path.write_text(
            json.dumps(_jsonable(theme_pool_audit), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        audit_reference = {
            "schema_version": THEME_POOL_AUDIT_SCHEMA_VERSION,
            "file": "theme_pool_audit.json",
            "raw_export": f"raw_exports/{raw_name}",
        }
        manifest["theme_pool_audit"] = dict(audit_reference)
        market_snapshot["theme_pool_audit"] = dict(audit_reference)
        snapshot_path.write_text(
            json.dumps(market_snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    shutil.copy2(report_path, raw_dir / f"{prefix}_report.md")
    shutil.copy2(holdings_path, raw_dir / f"{prefix}_holdings_review.csv")
    shutil.copy2(candidate_path, raw_dir / f"{prefix}_candidate_pool.csv")
    shutil.copy2(switch_path, raw_dir / f"{prefix}_switch_plan.csv")
    shutil.copy2(orders_path, raw_dir / f"{prefix}_orders.csv")
    shutil.copy2(pnl_path, raw_dir / f"{prefix}_pnl_summary.csv")
    if runtime_profile_path.exists():
        shutil.copy2(runtime_profile_path, raw_dir / "runtime_profile.json")
    if theme_pool_audit and theme_pool_audit_path.exists():
        shutil.copy2(theme_pool_audit_path, raw_dir / f"{prefix}_theme_pool_audit.json")


def _holdings_review_ledger_invariant(
    holdings_review: pd.DataFrame,
    effective_ledger: pd.DataFrame,
) -> dict[str, Any]:
    review_symbols = {
        str(symbol).strip().upper()
        for symbol in holdings_review.get("symbol", pd.Series(dtype=object)).tolist()
        if str(symbol).strip()
    }
    ledger_symbols = {
        str(symbol).strip().upper()
        for symbol in effective_ledger.get("symbol", pd.Series(dtype=object)).tolist()
        if str(symbol).strip()
    }
    extra = sorted(review_symbols - ledger_symbols)
    missing = sorted(ledger_symbols - review_symbols)
    return {
        "schema_version": "cn_aggressive_holdings_review_ledger_invariant.v1",
        "status": "ok" if not extra else "warning",
        "rule": "holdings_review symbols must be a subset of ledger_after_manual_switch symbols",
        "review_symbol_count": len(review_symbols),
        "ledger_symbol_count": len(ledger_symbols),
        "extra_review_symbols": extra,
        "missing_review_symbols": missing,
    }


def _prune_holdings_review_to_effective_ledger(
    holdings_review: pd.DataFrame,
    effective_ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    invariant = _holdings_review_ledger_invariant(holdings_review, effective_ledger)
    if holdings_review.empty or "symbol" not in holdings_review.columns:
        return holdings_review.copy(), invariant
    ledger_symbols = {
        str(symbol).strip().upper()
        for symbol in effective_ledger.get("symbol", pd.Series(dtype=object)).tolist()
        if str(symbol).strip()
    }
    if not ledger_symbols:
        return holdings_review.copy(), invariant
    pruned = holdings_review.copy()
    pruned["_symbol_key"] = pruned["symbol"].astype(str).str.strip().str.upper()
    pruned = pruned[pruned["_symbol_key"].isin(ledger_symbols)].drop(columns=["_symbol_key"]).reset_index(drop=True)
    post_invariant = _holdings_review_ledger_invariant(pruned, effective_ledger)
    post_invariant["pre_prune"] = invariant
    post_invariant["pruned_extra_review_symbols"] = invariant.get("extra_review_symbols", [])
    return pruned, post_invariant


def _manual_order_rows(
    *,
    timestamp: str,
    orders: list[ProposedOrder],
    source_ledger: pd.DataFrame,
    quote_by_symbol: dict[str, dict[str, Any]],
    execution_price_rejections: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source_by_symbol = {
        str(row.symbol).strip().upper(): row
        for row in source_ledger.itertuples()
    }
    for order in orders:
        symbol = str(order.symbol).strip().upper()
        source_row = source_by_symbol.get(symbol)
        quote = quote_by_symbol.get(symbol, {})
        rows.append(
            {
                "timestamp": timestamp,
                "action": order.action,
                "symbol": symbol,
                "name": getattr(source_row, "name", ""),
                "shares": int(order.shares),
                "execution_price": round(float(order.price), 2),
                "trade_value": round(float(order.trade_value), 2),
                "realized_pnl": round(float(order.realized_pnl), 2),
                "status": "filled",
                "reason": order.reason,
                "quote_source": str(quote.get("source") or ""),
                "quote_timestamp": _quote_timestamp(quote),
                "execution_price_field": str(
                    quote.get("realtime_execution_price_field")
                    or quote.get("realtime_price_field")
                    or quote.get("execution_price_field")
                    or ""
                ),
            }
        )
    for rejected in execution_price_rejections:
        symbol = str(rejected.get("symbol") or "").strip().upper()
        source_row = source_by_symbol.get(symbol)
        quote = quote_by_symbol.get(symbol, {})
        rejection_reason = str(rejected.get("reason") or "")
        pending_authorization = rejection_reason == (
            "advisory_only_no_local_manual_fill_authorization"
        )
        rows.append(
            {
                "timestamp": timestamp,
                "action": str(rejected.get("action") or ""),
                "symbol": symbol,
                "name": getattr(source_row, "name", ""),
                "shares": int(_safe_float(rejected.get("shares"), 0.0)),
                "execution_price": rejected.get("validated_realtime_price"),
                "trade_value": None,
                "realized_pnl": None,
                "status": "pending_authorization" if pending_authorization else "rejected",
                "reason": rejection_reason,
                "quote_source": str(quote.get("source") or "") if pending_authorization else "",
                "quote_timestamp": _quote_timestamp(quote) if pending_authorization else "",
                "execution_price_field": "",
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "action",
            "symbol",
            "name",
            "shares",
            "execution_price",
            "trade_value",
            "realized_pnl",
            "status",
            "reason",
            "quote_source",
            "quote_timestamp",
            "execution_price_field",
        ],
    )


def _write_manual_execution_outputs(
    *,
    run_dir: Path,
    timestamp: str,
    timestamp_long: str,
    updated_ledger: pd.DataFrame,
    pnl_summary: dict[str, Any],
    source_ledger: pd.DataFrame,
    orders: list[ProposedOrder],
    execution_price_rejections: list[dict[str, Any]],
    quote_by_symbol: dict[str, dict[str, Any]],
    quote_snapshot: str,
    quote_error: str,
    completeness_passed: bool,
    decision_data_sufficient: bool,
    dag_four_branch_compliance: dict[str, Any],
    execution_price_gate: dict[str, Any],
    advisory_only: bool,
    quote_provenance: dict[str, Any],
) -> dict[str, Any]:
    raw_dir = run_dir / "raw_exports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ledger_csv_path = run_dir / "ledger_after_manual_switch.csv"
    ledger_parquet_path = run_dir / "ledger_after_manual_switch.parquet"
    manual_orders_path = run_dir / "manual_switch_and_take_profit_orders.csv"
    review_path = run_dir / "daily_execution_review.md"
    manifest_path = run_dir / "manual_execution_manifest.json"

    manual_orders = _manual_order_rows(
        timestamp=timestamp_long,
        orders=orders,
        source_ledger=source_ledger,
        quote_by_symbol=quote_by_symbol,
        execution_price_rejections=execution_price_rejections,
    )
    updated_ledger.to_csv(ledger_csv_path, index=False, encoding="utf-8-sig")
    updated_ledger.to_parquet(ledger_parquet_path, index=False)
    manual_orders.to_csv(manual_orders_path, index=False, encoding="utf-8-sig")
    ledger_csv_bytes = _stable_read_bytes(ledger_csv_path)
    ledger_parquet_bytes = _stable_read_bytes(ledger_parquet_path)
    orders_bytes = _stable_read_bytes(manual_orders_path)
    ledger_csv_sha256 = hashlib.sha256(ledger_csv_bytes).hexdigest()
    ledger_parquet_sha256 = hashlib.sha256(ledger_parquet_bytes).hexdigest()
    orders_sha256 = hashlib.sha256(orders_bytes).hexdigest()

    applied_trades = manual_orders[manual_orders["status"] == "filled"].to_dict(orient="records")
    rejected_trades = manual_orders[manual_orders["status"] != "filled"].to_dict(orient="records")
    pending_authorization = bool(
        not manual_orders.empty
        and manual_orders["status"].astype(str).eq("pending_authorization").any()
    )
    status = (
        "filled_local_manual_paper_rebalance"
        if applied_trades
        else (
            "advisory_only_pending_authorization_carry_forward"
            if pending_authorization
            else "rejected_no_fill_carry_forward"
            if rejected_trades
            else "no_action_carry_forward"
        )
    )
    price_basis = (
        "execution_time_realtime_quote"
        if applied_trades
        else (
            "no_fill_advisory_only_pending_authorization"
            if pending_authorization
            else "no_fill_realtime_quote_missing_or_gate_rejected"
            if rejected_trades
            else "no_fill_no_orders"
        )
    )
    source_symbols = {
        str(symbol).strip().upper()
        for symbol in source_ledger.get("symbol", pd.Series(dtype=object)).tolist()
        if str(symbol).strip()
    }
    next_symbols = {
        str(symbol).strip().upper()
        for symbol in updated_ledger.get("symbol", pd.Series(dtype=object)).tolist()
        if str(symbol).strip()
    }
    manifest = {
        "schema_version": MANUAL_EXECUTION_SCHEMA_VERSION,
        "capital_cny": pnl_summary.get("initial_capital"),
        "status": status,
        "execution_status": status,
        "manual_execution_mode": "paper_only_local_manual_no_broker",
        "advisory_only": bool(advisory_only),
        "local_manual_fills_allowed": not bool(advisory_only),
        "record_timestamp": timestamp,
        "recorded_at": timestamp_long,
        "price_basis": price_basis,
        "quote_source": str(quote_provenance.get("source") or ""),
        "quote_provenance": _jsonable(quote_provenance),
        "quote_snapshot": quote_snapshot,
        "quote_fetch_error": quote_error or "",
        "decision_data_sufficient": bool(decision_data_sufficient),
        "completeness_passed": bool(completeness_passed),
        "dag_four_branch_complete": bool(dag_four_branch_compliance.get("complete")),
        "execution_price_gate": execution_price_gate,
        "applied_local_trades": applied_trades,
        "rejected_or_pending_trades": rejected_trades,
        "effective_manual_ledger_path": ledger_csv_path.name,
        "next_ledger_path": ledger_csv_path.name,
        "next_ledger_sha256": ledger_csv_sha256,
        "ledger_after_manual_switch_csv": ledger_csv_path.name,
        "ledger_after_manual_switch_csv_sha256": ledger_csv_sha256,
        "ledger_after_manual_switch_parquet": ledger_parquet_path.name,
        "ledger_after_manual_switch_parquet_sha256": ledger_parquet_sha256,
        "manual_orders_path": manual_orders_path.name,
        "manual_orders_sha256": orders_sha256,
        "daily_execution_review_path": review_path.name,
        "ledger_provenance": {
            "declared_next_ledger_path": ledger_csv_path.name,
            "contained_in_run_directory": True,
            "regular_non_symlink_file": True,
            "stable_double_read": True,
            "declared_sha256": ledger_csv_sha256,
            "csv_sha256": ledger_csv_sha256,
            "parquet_sha256": ledger_parquet_sha256,
        },
        "effective_manual_holding_count": int(len(next_symbols)),
        "source_manual_holding_count": int(len(source_symbols)),
        "cash_after": pnl_summary.get("cash_after"),
        "market_value_after": pnl_summary.get("market_value_after"),
        "total_value_after": pnl_summary.get("total_value_after"),
        "portfolio_pnl_after": pnl_summary.get("portfolio_pnl_after"),
        "portfolio_return_after": pnl_summary.get("portfolio_pnl_pct_after"),
        "realized_pnl_from_rebalance": pnl_summary.get("realized_pnl_from_rebalance"),
        "no_broker_api_called": True,
    }
    manifest["financial_state_sha256"] = _manual_financial_state_sha256(
        manifest,
        ledger_sha256=ledger_csv_sha256,
    )
    review_lines = [
        f"# 本地/manual执行复盘 {timestamp}",
        "",
        "- 执行状态："
        + (
            f"本地/manual paper 成交 `{len(applied_trades)}` 笔；无真实券商/API下单。"
            if applied_trades
            else "未成交；无真实券商/API下单。"
        ),
        f"- 有效 manual ledger：`{ledger_csv_path}`；SHA256 `{ledger_csv_sha256}`。",
        f"- 模式：advisory_only=`{bool(advisory_only)}`，local_manual_fills_allowed=`{not bool(advisory_only)}`。",
        f"- 有效持仓数：`{len(next_symbols)}`；组合纪律上限 `{MAX_EFFECTIVE_HOLDING_COUNT}`。",
        f"- quote_snapshot：`{quote_snapshot or 'N/A'}`；价格口径 `{price_basis}`。",
        f"- 数据闸门：decision_data_sufficient=`{bool(decision_data_sufficient)}`，completeness_passed=`{bool(completeness_passed)}`。",
        f"- DAG 四分支：complete=`{bool(dag_four_branch_compliance.get('complete'))}`。",
    ]
    if applied_trades:
        review_lines.append("- 已写入本地/manual成交：")
        for row in applied_trades:
            review_lines.append(
                f"- `{row['symbol']}` {row['action']} {int(row['shares'])}股 "
                f"@ {float(row['execution_price']):.2f}；{row['reason']}"
            )
    if rejected_trades:
        review_lines.append("- 未成交/拒绝：")
        for row in rejected_trades:
            review_lines.append(
                f"- `{row['symbol']}` {row['action']} {int(row['shares'])}股：{row['reason']}"
            )
    review_path.write_text("\n".join(review_lines) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2), encoding="utf-8")

    prefix = f"aggressive_portfolio_{timestamp}"
    shutil.copy2(manifest_path, raw_dir / f"{prefix}_manual_execution_manifest.json")
    shutil.copy2(manual_orders_path, raw_dir / f"{prefix}_manual_switch_and_take_profit_orders.csv")
    shutil.copy2(ledger_csv_path, raw_dir / f"{prefix}_ledger_after_manual_switch.csv")
    shutil.copy2(ledger_parquet_path, raw_dir / f"{prefix}_ledger_after_manual_switch.parquet")
    shutil.copy2(review_path, raw_dir / f"{prefix}_daily_execution_review.md")
    return manifest


def _build_notes_payload(
    trade_date: str,
    data_status: str,
    market_core_view: str,
    pnl_summary: dict[str, Any],
    orders: list[ProposedOrder],
    switch_plan_df: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    tomorrow_focus: list[str],
    theme_pool_summary: dict[str, Any] | None = None,
    theme_symbols: dict[str, Any] | None = None,
) -> str:
    if orders:
        order_text = "；".join(
            f"{order.symbol} {order.action} {order.shares}股 @ {order.price:.2f}" for order in orders
        )
    else:
        order_text = "无，本日维持现有结构。"

    if switch_plan_df.empty:
        switch_text = "否，暂无明显优于现持仓的可执行换仓对象。"
        switch_detail_text = "维持原组合，继续跟踪候选观察池。"
    else:
        top_switch = switch_plan_df.iloc[0]
        action = str(top_switch["action"])
        if action == "switch_now":
            switch_text = "是，已形成明确换仓优先级。"
        elif action == "prepare_switch":
            switch_text = "否，但已形成预备换仓对象。"
        else:
            switch_text = "否，先保留观察池。"
        sell_label = f"{top_switch['sell_symbol']} {top_switch['sell_name']}"
        buy_label = f"{top_switch['buy_symbol']} {top_switch['buy_name']}"
        switch_detail_text = (
            f"{sell_label} -> {buy_label}，优先级 `{top_switch['priority']}`，"
            f"触发条件：{top_switch['trigger_threshold']}"
        )

    if candidate_pool.empty:
        candidate_text = "暂无可用备选。"
    else:
        top_rows = candidate_pool.head(3)
        candidate_text = "；".join(
            f"{row.symbol}({row.name})/{row.theme_label}/评分{int(getattr(row, 'codex_recommendation_score', 0))}/证据{row.evidence_quality}"
            for row in top_rows.itertuples()
        )
    theme_exposure_text = ""
    if theme_pool_summary:
        theme_lines = _format_theme_pool_report_lines(
            theme_pool_summary,
            candidate_pool=candidate_pool,
            theme_symbols=theme_symbols or {},
        )
        theme_exposure_text = "；".join(
            line.removeprefix("- ").rstrip("。") for line in theme_lines[:4]
        )

    notes_lines = [
        f"# A股日度复盘 {trade_date}",
        "",
        f"- 数据完整性：{data_status}",
        f"- 市场核心判断：{market_core_view}",
        (
            f"- 组合盈亏：截至 `{pnl_summary['quote_snapshot']}`，总资产 "
            f"`{pnl_summary['total_value_after']:,.2f} 元`，较初始资金 "
            f"`{pnl_summary['portfolio_pnl_after']:,.2f} 元` "
            f"（{pnl_summary['portfolio_pnl_pct_after']:.2%}），较上一条正式记录变动 "
            f"`{pnl_summary['delta_vs_source_record']:,.2f} 元`。"
        ),
        f"- 备选投资建议：{candidate_text}",
        f"- 是否调仓：{'是' if orders else '否'}",
        f"- 调仓内容：{order_text}",
        f"- 是否换仓：{switch_text}",
        f"- 换仓内容：{switch_detail_text}",
        f"- 明日观察重点：{'；'.join(tomorrow_focus)}",
    ]
    if theme_exposure_text:
        notes_lines.insert(6, f"- Theme 候选暴露：{theme_exposure_text}")
    return "\n".join(notes_lines)


def run_tracker(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    now = _now_local()
    advisory_only = bool(getattr(args, "advisory_only", True))
    decision_log_path = Path(
        str(
            getattr(args, "decision_log_path", DEFAULT_DECISION_LOG_PATH)
            or DEFAULT_DECISION_LOG_PATH
        )
    ).expanduser()
    decision_log_allowed_root = Path(
        str(
            getattr(
                args,
                "decision_log_allowed_root",
                DEFAULT_DECISION_LOG_PATH.parent,
            )
            or DEFAULT_DECISION_LOG_PATH.parent
        )
    ).expanduser()
    allow_live_quotes = bool(getattr(args, "allow_live_quotes", False))
    quote_input_json = str(getattr(args, "quote_input_json", "") or "").strip()
    quote_max_age_seconds = max(
        0,
        int(getattr(args, "quote_max_age_seconds", DEFAULT_QUOTE_MAX_AGE_SECONDS)),
    )
    if quote_input_json and allow_live_quotes:
        raise RuntimeError("quote_input_json_and_allow_live_quotes_are_mutually_exclusive")
    timestamp = _allocate_run_timestamp(Path(args.base_dir), now)
    timestamp_long = now.strftime("%Y-%m-%d %H:%M:%S %Z")
    base_dir = Path(args.base_dir)
    run_dir = base_dir / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)

    source_ledger, source_manifest, source_pnl = _load_previous_record(
        base_dir,
        source_record=args.source_record,
    )
    source_record = str(source_manifest.get("timestamp") or source_ledger.iloc[0].get("source_record", "unknown"))
    cash_before = _safe_float(source_pnl["cash_after"].iloc[-1] if "cash_after" in source_pnl.columns else 0.0)
    initial_capital = _safe_float(source_manifest.get("capital_cny"), DEFAULT_INITIAL_CAPITAL)
    source_total_value = _safe_float(
        source_pnl["total_value_after"].iloc[-1] if "total_value_after" in source_pnl.columns else initial_capital
    )

    cn_data_root = Path(get_market_settings('CN').data_dir)
    downloader = CNFullMarketDownloader(
        data_dir=str(cn_data_root),
        years=args.years,
        max_workers=4,
    )
    components = downloader.load_components()
    market_data_reader = MarketDataReader(market="CN", data_root=cn_data_root.parent)
    review_categories = ["full_a", "hs300", "zz500", "zz1000"]
    completeness_before = build_parquet_canonical_completeness_report(
        reader=market_data_reader,
        components=components,
        categories=review_categories,
        allowed_stale_symbols=args.allowed_stale_symbols,
    )
    attempted_backfill = False
    download_report_path = None
    completeness_after = completeness_before

    latest_trade_date = str(completeness_after.get("latest_trade_date") or "")
    analysis_trade_date = _resolve_analysis_trade_date(completeness_after)
    completeness_passed = bool(completeness_after["complete"])
    skip_market_metrics_prewarm = bool(getattr(args, "skip_market_metrics_prewarm", False))

    market_metrics_bundle = _load_or_compute_market_metrics_bundle(
        base_dir=base_dir,
        components=components,
        reader=market_data_reader,
        latest_trade_date=analysis_trade_date,
        completeness_report=completeness_after,
        skip_prewarm=skip_market_metrics_prewarm,
    )
    full_metrics = market_metrics_bundle.full_metrics
    breadth = market_metrics_bundle.breadth
    market_metrics_cache_meta = market_metrics_bundle.cache_meta

    review_layer = _run_unified_review_mainline_for_holdings(
        source_ledger=source_ledger,
        latest_trade_date=analysis_trade_date,
        source_record=source_record,
    )
    review_by_symbol = dict(review_layer.get("by_symbol", {}) or {})
    degraded_symbols = dict(review_layer.get("degraded_symbols", {}) or {})
    review_attempt_summary = dict(review_layer.get("llm_attempt_summary", review_layer.get("llm_usage_summary", {})) or {})
    review_effective_summary = dict(review_layer.get("llm_effective_summary", {}) or {})
    review_model_role_metadata = dict(review_layer.get("model_role_metadata", {}) or {})
    codex_handoff_active = bool(review_layer.get("codex_handoff", False))
    branch_signals_by_symbol = {
        symbol: {
            "reviewed_branch_verdicts": dict(payload.get("reviewed_branch_verdicts", {}) or {}),
            "branch_overlays": dict(payload.get("branch_overlays", {}) or {}),
            "master_hint": dict(payload.get("master_hint", {}) or {}),
            "ic_hint": dict(payload.get("ic_hint", {}) or {}),
            "recommendation": dict(payload.get("recommendation", {}) or {}),
            "report_excerpt": str(payload.get("report_excerpt", "") or ""),
            "recall_context": dict(payload.get("recall_context", {}) or {}),
        }
        for symbol, payload in review_by_symbol.items()
    }
    holding_dag_assessments = _build_holding_dag_assessments(
        branch_signals_by_symbol
    )

    metrics_map = {
        row.symbol: row._asdict()
        for row in full_metrics.itertuples(index=False)
    }

    holding_quote_codes = [_map_symbol_to_quote_code(symbol) for symbol in source_ledger["symbol"]]
    index_quote_codes = list(INDEX_QUOTES.keys())
    quote_error = ""
    quote_provenance: dict[str, Any]
    if quote_input_json:
        try:
            quote_payload, quote_provenance = _load_local_quote_input(quote_input_json)
        except Exception as exc:
            quote_payload = {}
            quote_error = str(exc)
            quote_provenance = {
                "mode": "local_quote_input_rejected",
                "path": quote_input_json,
                "provider_fallback_used": False,
                "error": quote_error,
            }
    elif allow_live_quotes:
        try:
            quote_payload = _fetch_tencent_quotes(index_quote_codes + holding_quote_codes)
            quote_provenance = {
                "mode": "explicit_live_provider",
                "source": "tencent_realtime_quote",
                "provider_fallback_used": False,
            }
        except Exception as exc:
            quote_payload = {}
            quote_error = str(exc)
            quote_provenance = {
                "mode": "explicit_live_provider_failed",
                "source": "tencent_realtime_quote",
                "provider_fallback_used": False,
                "error": quote_error,
            }
    else:
        quote_payload = {}
        quote_error = "provider_quotes_disabled_by_default"
        quote_provenance = {
            "mode": "provider_quotes_disabled",
            "provider_fallback_used": False,
        }
    quote_validation_now = _now_local()
    raw_quote_snapshot = max(
        [_quote_timestamp(quote_payload.get(code, {})) for code in index_quote_codes + holding_quote_codes],
        default="",
    )
    fresh_quote_timestamps = [
        _quote_timestamp(quote_payload.get(code, {}))
        for code in index_quote_codes + holding_quote_codes
        if _quote_freshness(
            quote_payload.get(code, {}),
            now=quote_validation_now,
            max_age_seconds=quote_max_age_seconds,
        )[0]
    ]
    quote_snapshot = max(fresh_quote_timestamps, default="")
    quote_provenance["raw_latest_quote_timestamp"] = raw_quote_snapshot
    quote_provenance["fresh_latest_quote_timestamp"] = quote_snapshot
    quote_provenance["fresh_quote_count"] = len(fresh_quote_timestamps)
    quote_provenance["quote_max_age_seconds"] = quote_max_age_seconds
    quote_provenance["diagnostic_quote_evaluated_at"] = (
        quote_validation_now.isoformat()
    )
    diagnostic_completeness_after = {
        **completeness_after,
        "quote_snapshot": quote_snapshot,
    }
    decision_data_sufficient = is_previous_day_realtime_decision_sufficient(
        target_date=str(completeness_after.get("effective_target_trade_date") or latest_trade_date or ""),
        dominant_local_snapshot_date=analysis_trade_date,
        completeness_state=diagnostic_completeness_after,
        quote_snapshot=quote_snapshot,
    )

    indices = {
        code: {
            **quote_payload[code],
            "name": INDEX_QUOTES[code],
        }
        for code in index_quote_codes
        if code in quote_payload
    }

    current_rows: list[dict[str, Any]] = []
    previous_value_map = {
        row.symbol: float(getattr(row, "current_value", 0.0))
        for row in source_ledger.itertuples()
    }
    for row in source_ledger.itertuples():
        symbol = row.symbol
        metric = metrics_map.get(symbol, {})
        quote = quote_payload.get(_map_symbol_to_quote_code(symbol), {})
        review_payload = review_by_symbol.get(symbol, {})
        recommendation = dict(review_payload.get("recommendation", {}) or {})
        ic_hint = dict(review_payload.get("ic_hint", {}) or {})
        master_hint = dict(review_payload.get("master_hint", {}) or {})
        llm_attempt_summary = dict(review_payload.get("llm_attempt_summary", {}) or {})
        llm_effective_summary = dict(review_payload.get("llm_effective_summary", {}) or {})
        llm_action = str(recommendation.get("action") or ic_hint.get("action") or "hold")
        llm_confidence_source = ""
        llm_confidence_value: float | None = None
        if "confidence" in recommendation and recommendation.get("confidence") is not None:
            llm_confidence_value = _safe_float(recommendation.get("confidence"))
            llm_confidence_source = "recommendation.confidence"
        elif "confidence_hint" in ic_hint and ic_hint.get("confidence_hint") is not None:
            llm_confidence_value = _safe_float(ic_hint.get("confidence_hint"))
            llm_confidence_source = "ic_hint.confidence_hint"
        elif "confidence_hint" in master_hint and master_hint.get("confidence_hint") is not None:
            llm_confidence_value = _safe_float(master_hint.get("confidence_hint"))
            llm_confidence_source = "master_hint.confidence_hint"
        if codex_handoff_active and llm_confidence_value is None:
            llm_confidence: float | None = None
            llm_confidence_source = "codex_handoff"
        else:
            llm_confidence = float(llm_confidence_value or 0.0)
        llm_conclusion = str(
            recommendation.get("one_line_conclusion")
            or ic_hint.get("thesis")
            or ""
        ).strip()
        llm_risk_flags = list(recommendation.get("risk_flags") or ic_hint.get("risk_flags") or [])
        llm_session_id = str(review_payload.get("llm_session_id", "") or "")
        quote_fresh, quote_freshness_reason, quote_age_seconds = _quote_freshness(
            quote,
            now=quote_validation_now,
            max_age_seconds=quote_max_age_seconds,
        )
        realtime_execution_price, realtime_execution_price_field = _resolve_realtime_execution_price(
            quote,
            now=quote_validation_now,
            max_age_seconds=quote_max_age_seconds,
        )
        fallback_price = _fallback_current_price(metric, row)
        current_price = realtime_execution_price if realtime_execution_price > 0 else fallback_price
        current_value = round(int(row.shares) * current_price, 2)
        buy_value = round(float(row.cost_basis), 2)
        unrealized = round(current_value - buy_value, 2)
        today_change_pct = round(_safe_float(quote.get("change_pct")), 2)
        staged_target = round(
            _fallback_metric_or_row_price(
                metric,
                "stage_target_price",
                row,
                "stage_target_price",
                current_price * 1.1,
            ),
            2,
        )
        staged_stop = round(
            _fallback_metric_or_row_price(
                metric,
                "stage_stop_price",
                row,
                "stage_stop_price",
                current_price * 0.94,
            ),
            2,
        )
        manual_entry_trade_date = _trailing_take_profit_row_date(row, TRAILING_TAKE_PROFIT_ENTRY_DATE_COLUMNS)
        carried_trailing_peak_price = _trailing_take_profit_row_float(row, TRAILING_TAKE_PROFIT_PEAK_PRICE_COLUMNS)
        carried_trailing_peak_trade_date = _trailing_take_profit_row_date(row, TRAILING_TAKE_PROFIT_PEAK_DATE_COLUMNS)
        current_rows.append(
            {
                "symbol": symbol,
                "name": row.name,
                "category": metric.get("category", ""),
                "shares_before": int(row.shares),
                "buy_price": round(float(row.avg_cost), 6),
                "buy_value": buy_value,
                "current_price": round(current_price, 2),
                "current_value": current_value,
                "unrealized_pnl": unrealized,
                "unrealized_pnl_pct": round(_safe_pct(unrealized, buy_value), 6),
                "today_change_pct": today_change_pct,
                "ret5": round(_safe_float(metric.get("ret5")), 6),
                "ret20": round(_safe_float(metric.get("ret20")), 6),
                "ret60": round(_safe_float(metric.get("ret60")), 6),
                "close_vs_ma20": round(_safe_float(metric.get("close_vs_ma20")), 6),
                "ma20_vs_ma60": round(_safe_float(metric.get("ma20_vs_ma60")), 6),
                "ma60_vs_ma120": round(_safe_float(metric.get("ma60_vs_ma120")), 6),
                "dd20": round(_safe_float(metric.get("dd20")), 6),
                "rank_full_market": int(metric["rank_full_market"]) if metric and "rank_full_market" in metric else 9999,
                "score_full_market": round(_safe_float(metric.get("score_full_market")), 6),
                "stage_target_price": staged_target,
                "stage_stop_price": staged_stop,
                "manual_entry_trade_date": manual_entry_trade_date,
                "trailing_profit_tracking_start_date": _compact_trade_date(
                    getattr(row, "trailing_profit_tracking_start_date", "")
                ),
                "trailing_take_profit_basis": str(getattr(row, "trailing_take_profit_basis", "") or ""),
                "trailing_profit_peak_price": (
                    round(carried_trailing_peak_price, 2)
                    if carried_trailing_peak_price > 0
                    else 0.0
                ),
                "trailing_profit_peak_trade_date": carried_trailing_peak_trade_date,
                "delta_vs_source_record": round(current_value - previous_value_map.get(symbol, 0.0), 2),
                "llm_action": llm_action,
                "llm_confidence": round(llm_confidence, 6) if llm_confidence is not None else None,
                "llm_conclusion": llm_conclusion,
                "llm_risk_flags": "；".join(str(item).strip() for item in llm_risk_flags if str(item).strip()),
                "llm_session_id": llm_session_id,
                "llm_attempt_calls": int(llm_attempt_summary.get("call_count", 0) or 0),
                "llm_failed_calls": int(llm_attempt_summary.get("failed_count", 0) or 0),
                "llm_effective_calls": int(llm_effective_summary.get("call_count", 0) or 0),
                "llm_confidence_source": llm_confidence_source,
                "llm_degraded": bool(review_payload.get("llm_degraded", False)),
                "realtime_quote_timestamp": _quote_timestamp(quote),
                "quote_time": _quote_timestamp(quote),
                "realtime_quote_source": str(quote.get("source") or ""),
                "realtime_quote_valid": realtime_execution_price > 0,
                "realtime_quote_fresh": bool(quote_fresh),
                "realtime_quote_freshness_reason": quote_freshness_reason,
                "realtime_quote_age_seconds": (
                    round(float(quote_age_seconds), 3)
                    if quote_age_seconds is not None
                    else None
                ),
                "realtime_execution_price": round(realtime_execution_price, 2) if realtime_execution_price > 0 else None,
                "realtime_execution_price_field": realtime_execution_price_field,
            }
        )

    holdings_review = pd.DataFrame(current_rows)
    total_market_value_before = round(float(holdings_review["current_value"].sum()), 2)
    total_value_before = round(total_market_value_before + cash_before, 2)
    if total_market_value_before > 0:
        holdings_review["equity_sleeve_weight"] = (
            holdings_review["current_value"] / total_market_value_before
        ).round(6)
        holdings_review["market_weight"] = holdings_review["equity_sleeve_weight"]
    else:
        holdings_review["equity_sleeve_weight"] = 0.0
        holdings_review["market_weight"] = 0.0
    if total_value_before > 0:
        holdings_review["nav_weight"] = (
            holdings_review["current_value"] / total_value_before
        ).round(6)
    else:
        holdings_review["nav_weight"] = 0.0
    holdings_review = _apply_trailing_take_profit_review(
        holdings_review,
        reader=market_data_reader,
        analysis_trade_date=analysis_trade_date,
    )
    holdings_review["position_role"] = holdings_review.apply(_position_role, axis=1)
    holdings_review["recommended_action"] = holdings_review.apply(_position_action, axis=1)
    holdings_review["reason"] = holdings_review.apply(_position_reason, axis=1)
    holdings_review["codex_recommendation_score"] = holdings_review.apply(_holding_codex_score, axis=1)
    holdings_review["codex_recommendation_rating"] = holdings_review["codex_recommendation_score"].map(
        _codex_rating_from_score
    )
    holdings_review = holdings_review.sort_values(
        by=["score_full_market", "today_change_pct", "symbol"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    trailing_take_profit_summary = _summarize_trailing_take_profit(holdings_review)
    candidate_pool, candidate_level_dag_status, theme_pool_audit = _run_candidate_level_v13_dag(
        held_symbols=holdings_review["symbol"].tolist(),
        analysis_trade_date=analysis_trade_date,
        completeness_report=completeness_after,
        total_capital=initial_capital,
    )
    theme_pool_summary = _mapping_payload(theme_pool_audit.get("summary"))
    if not candidate_pool.empty:
        candidate_quote_codes = [
            _map_symbol_to_quote_code(str(symbol))
            for symbol in candidate_pool["symbol"].tolist()
        ]
        missing_quote_codes = [code for code in candidate_quote_codes if code not in quote_payload]
        if missing_quote_codes and allow_live_quotes:
            try:
                quote_payload.update(_fetch_tencent_quotes(missing_quote_codes))
            except Exception as exc:
                quote_error = "; ".join(item for item in [quote_error, str(exc)] if item)
        candidate_pool = _attach_candidate_quote_gate(
            candidate_pool,
            quote_payload=quote_payload,
            analysis_trade_date=analysis_trade_date,
            now=quote_validation_now,
            max_age_seconds=quote_max_age_seconds,
        )
        candidate_pool = candidate_pool.copy()
        candidate_pool["codex_recommendation_score"] = candidate_pool.apply(_candidate_codex_score, axis=1)
        candidate_pool["codex_recommendation_rating"] = candidate_pool["codex_recommendation_score"].map(
            _codex_rating_from_score
        )
    switch_plan_df = _build_switch_plan(
        holdings_review=holdings_review,
        candidate_pool=candidate_pool,
        completeness_passed=completeness_passed,
        decision_data_sufficient=decision_data_sufficient,
    )
    if not switch_plan_df.empty and not candidate_pool.empty:
        score_map = candidate_pool.set_index("symbol")["codex_recommendation_score"].to_dict()
        rating_map = candidate_pool.set_index("symbol")["codex_recommendation_rating"].to_dict()
        switch_plan_df = switch_plan_df.copy()
        switch_plan_df["buy_codex_recommendation_score"] = switch_plan_df["buy_symbol"].map(score_map)
        switch_plan_df["buy_codex_recommendation_rating"] = switch_plan_df["buy_symbol"].map(rating_map)

    theme_strength = _summarize_theme_strength(holdings_review)
    style_text = _market_style_conclusion(indices=indices, breadth=breadth)
    strongest_theme_text, weakest_theme_text = _tech_mainline_conclusion(theme_strength)
    proposed_orders = _build_rebalance_plan(holdings_review)
    # 当前策略仍以主线内部修复为主，只在出现明确弱化共振时执行减仓。
    if len(proposed_orders) == 1 and theme_strength and theme_strength[0]["avg_score"] >= 0.8:
        proposed_orders = []
    quote_by_symbol = {
        str(row.symbol).strip().upper(): quote_payload.get(_map_symbol_to_quote_code(str(row.symbol)), {})
        for row in source_ledger.itertuples()
    }
    execution_price_rejections: list[dict[str, Any]] = []
    validated_orders: list[ProposedOrder] = []
    order_gate_decisions: list[dict[str, Any]] = []
    decision_log_authorizations: list[dict[str, Any]] = []
    data_gate_allows_new_risk = completeness_passed or decision_data_sufficient
    execution_gate_now = _now_local()
    for order in proposed_orders:
        risk_sell_allowed, risk_sell_reason = _risk_reduction_sell_gate(
            order=order,
            effective_ledger=source_ledger,
            holdings_review=holdings_review,
        )
        gate_reason = "data_gate_allows_new_risk"
        if not data_gate_allows_new_risk:
            if not risk_sell_allowed:
                execution_price_rejections.append(
                    {
                        "symbol": order.symbol,
                        "action": order.action,
                        "shares": order.shares,
                        "reason": "data_gate_blocked_non_risk_reduction_order",
                        "data_gate_allows_new_risk": False,
                        "risk_reduction_sell_gate": risk_sell_reason,
                    }
                )
                continue
            gate_reason = risk_sell_reason
        elif risk_sell_allowed:
            gate_reason = risk_sell_reason
        holding_assessment = holding_dag_assessments.get(
            str(order.symbol).strip().upper(), {}
        )
        if holding_assessment.get("holding_dag_status") != "ready":
            rejection = {
                "symbol": order.symbol,
                "action": order.action,
                "shares": order.shares,
                "reason": "holding_dag_blocked_read_only_risk_reduction_advisory",
                "data_gate_allows_new_risk": bool(data_gate_allows_new_risk),
                "risk_reduction_sell_gate": risk_sell_reason,
                "execution_gate_reason": gate_reason,
                "holding_dag_blockers": list(
                    holding_assessment.get("holding_dag_blockers")
                    or ["holding_dag_assessment_missing"]
                ),
            }
            execution_price_rejections.append(rejection)
            order_gate_decisions.append(rejection)
            continue
        order_gate_decisions.append(
            {
                "symbol": order.symbol,
                "action": order.action,
                "shares": order.shares,
                "data_gate_allows_new_risk": bool(data_gate_allows_new_risk),
                "risk_reduction_sell_gate": risk_sell_reason,
                "execution_gate_reason": gate_reason,
            }
        )
        execution_price, execution_price_field = _resolve_realtime_execution_price(
            quote_by_symbol.get(str(order.symbol).strip().upper(), {}),
            now=execution_gate_now,
            max_age_seconds=quote_max_age_seconds,
        )
        if execution_price <= 0:
            execution_price_rejections.append(
                {
                    "symbol": order.symbol,
                    "action": order.action,
                    "shares": order.shares,
                    "reason": "missing_valid_realtime_execution_price",
                    "data_gate_allows_new_risk": bool(data_gate_allows_new_risk),
                    "risk_reduction_sell_gate": risk_sell_reason,
                    "execution_gate_reason": gate_reason,
                }
            )
            continue
        quote_by_symbol[str(order.symbol).strip().upper()][
            "realtime_execution_price_field"
        ] = execution_price_field
        source_row = source_ledger[source_ledger["symbol"] == order.symbol]
        avg_cost = _safe_float(source_row["avg_cost"].iloc[0]) if not source_row.empty else order.price
        trade_value = round(order.shares * execution_price, 2)
        realized_pnl = (
            round(order.shares * (execution_price - avg_cost), 2)
            if order.action == "sell"
            else order.realized_pnl
        )
        validated_order = ProposedOrder(
            symbol=order.symbol,
            action=order.action,
            shares=order.shares,
            price=round(execution_price, 2),
            trade_value=trade_value,
            realized_pnl=realized_pnl,
            reason=order.reason,
        )
        if advisory_only:
            execution_price_rejections.append(
                {
                    "symbol": order.symbol,
                    "action": order.action,
                    "shares": order.shares,
                    "reason": "advisory_only_no_local_manual_fill_authorization",
                    "data_gate_allows_new_risk": bool(data_gate_allows_new_risk),
                    "risk_reduction_sell_gate": risk_sell_reason,
                    "execution_gate_reason": gate_reason,
                    "validated_realtime_price": round(execution_price, 2),
                }
            )
            continue
        decision_log_authorization = _same_day_manual_fill_authorization(
            decision_log_path=decision_log_path,
            trade_date=execution_gate_now.strftime("%Y%m%d"),
            symbol=str(order.symbol).strip().upper(),
            order_action=str(order.action).strip().lower(),
            order_shares=int(order.shares),
            allowed_root=decision_log_allowed_root,
        )
        decision_log_authorizations.append(decision_log_authorization)
        if not decision_log_authorization["authorized"]:
            execution_price_rejections.append(
                {
                    "symbol": order.symbol,
                    "action": order.action,
                    "shares": order.shares,
                    "reason": "same_day_decision_log_authorization_missing",
                    "data_gate_allows_new_risk": bool(data_gate_allows_new_risk),
                    "risk_reduction_sell_gate": risk_sell_reason,
                    "execution_gate_reason": gate_reason,
                    "validated_realtime_price": round(execution_price, 2),
                    "decision_log_blockers": list(
                        decision_log_authorization.get("blockers") or []
                    ),
                }
            )
            continue
        validated_orders.append(
            validated_order
        )
    orders = validated_orders
    execution_price_gate = {
        "accepted_price_fields": list(REALTIME_EXECUTION_PRICE_FIELDS),
        "requires_realtime_quote_timestamp": True,
        "requires_same_local_trade_date": True,
        "quote_max_age_seconds": quote_max_age_seconds,
        "rejects_static_daily_prices": True,
        "advisory_only": advisory_only,
        "local_manual_fills_allowed": not advisory_only,
        "requires_same_day_decision_log_pair": True,
        "decision_log_path": str(decision_log_path),
        "quote_policy": quote_provenance,
        "execution_quote_evaluated_at": execution_gate_now.isoformat(),
        "data_gate_allows_new_risk": bool(data_gate_allows_new_risk),
        "risk_reduction_sell_bypass_enabled": True,
        "order_gate_decisions": order_gate_decisions,
        "decision_log_authorizations": decision_log_authorizations,
        "rejections": execution_price_rejections,
    }

    order_rows = []
    for order in orders:
        name = source_ledger[source_ledger["symbol"] == order.symbol]["name"].iloc[0]
        order_rows.append(
            {
                "timestamp": timestamp_long,
                "action": order.action,
                "symbol": order.symbol,
                "name": name,
                "shares": order.shares,
                "price": order.price,
                "trade_value": order.trade_value,
                "realized_pnl": order.realized_pnl,
                "reason": order.reason,
            }
        )
    orders_df = pd.DataFrame(
        order_rows,
        columns=["timestamp", "action", "symbol", "name", "shares", "price", "trade_value", "realized_pnl", "reason"],
    )

    quote_prices = {row["symbol"]: float(row["current_price"]) for row in current_rows}
    if advisory_only and orders:
        raise RuntimeError("advisory_only_invariant_violation_orders_not_empty")
    updated_ledger, cash_after, realized_pnl = _apply_orders(
        source_ledger=source_ledger,
        orders=orders,
        cash_before=cash_before,
        quote_prices=quote_prices,
    )
    ledger_meta_columns = [
        "symbol",
        "stage_target_price",
        "stage_stop_price",
        "position_role",
        *[
            column
            for column in TRAILING_TAKE_PROFIT_LEDGER_COLUMNS
            if column in holdings_review.columns
        ],
    ]
    ledger_meta = holdings_review[ledger_meta_columns].rename(columns={"position_role": "thesis_status"})
    updated_ledger = updated_ledger.merge(ledger_meta, on="symbol", how="left")

    total_market_value_after = round(float(updated_ledger["current_value"].sum()), 2)
    total_value_after = round(total_market_value_after + cash_after, 2)
    portfolio_pnl_after = round(total_value_after - initial_capital, 2)
    portfolio_pnl_before = round(total_value_before - initial_capital, 2)

    float_winners = holdings_review.sort_values("unrealized_pnl", ascending=False)
    float_losers = holdings_review.sort_values("unrealized_pnl", ascending=True)

    rebalance_reason = (
        "执行温和调仓，削减已明显落后于主线的弱支线仓位。"
        if orders
        else "今天不执行调仓，继续把强修复与弱滞涨的分化再观察一个交易日。"
    )
    data_status = _build_data_status_summary(
        completeness_after,
        analysis_trade_date=analysis_trade_date,
        decision_data_sufficient=decision_data_sufficient,
    )
    tomorrow_focus = [
        "确认先进材料与光通信的修复能否延续，不让单日反弹误判为全面重启",
        "继续观察 `大族激光 / 中国西电` 是否能重新站回阶段止损位",
        "跟踪科创50 相对沪深300 的强弱差，判断资金是否继续偏向硬科技",
    ]
    trailing_watch_symbols = list(trailing_take_profit_summary.get("watch_symbols", []) or [])
    trailing_unconfirmed_symbols = list(trailing_take_profit_summary.get("unconfirmed_symbols", []) or [])
    if trailing_watch_symbols:
        tomorrow_focus.insert(
            0,
            "逐票复核移动止盈：`" + "，".join(trailing_watch_symbols[:5]) + "` 已进入利润回吐/移动止盈观察",
        )
    elif trailing_unconfirmed_symbols:
        tomorrow_focus.append(
            "补齐 manual ledger 买入/跟踪起始日期，避免移动止盈峰值只停留在 unconfirmed 估算"
        )
    if not completeness_passed and decision_data_sufficient:
        tomorrow_focus.insert(
            0,
            (
                f"收盘后确认 strict 交易日 `{_format_trade_date(completeness_after.get('strict_trade_date'))}` "
                "日线落库，并复核盘中前日线+实时行情结论"
            ),
        )
    elif not completeness_passed:
        tomorrow_focus.insert(
            0,
            (
                f"核对本地A股快照阻塞缺口 `{int(completeness_after.get('blocking_incomplete_count', 0) or 0)}` 个"
                "是否属于真实缺数还是停牌/预上市扰动"
            ),
        )
    if (
        str(completeness_after.get("strict_trade_date") or "")
        and completeness_after.get("strict_trade_date") != completeness_after.get("effective_target_trade_date")
    ):
        tomorrow_focus.insert(
            0,
            (
                f"关注 strict 交易日 `{_format_trade_date(completeness_after.get('strict_trade_date'))}`"
                " 是否在明日变成必须补齐的新主导快照"
            ),
        )
    data_snapshot_lines = _build_data_snapshot_lines(
        completeness_report=completeness_after,
        quote_snapshot=quote_snapshot,
        analysis_trade_date=analysis_trade_date,
        decision_data_sufficient=decision_data_sufficient,
    )
    dag_four_branch_compliance = _build_dag_four_branch_compliance(
        review_symbols=list(branch_signals_by_symbol.keys()),
        effective_local_holding_symbols=[
            str(symbol).strip().upper()
            for symbol in list(source_ledger["symbol"])
            if str(symbol).strip()
        ],
        branch_signals_by_symbol=branch_signals_by_symbol,
    )
    holdings_review["holding_dag_status"] = holdings_review["symbol"].map(
        lambda symbol: str(
            holding_dag_assessments.get(str(symbol).strip().upper(), {}).get(
                "holding_dag_status", "blocked"
            )
        )
    )
    holdings_review["holding_dag_blockers"] = holdings_review["symbol"].map(
        lambda symbol: ";".join(
            holding_dag_assessments.get(str(symbol).strip().upper(), {}).get(
                "holding_dag_blockers", ["holding_dag_assessment_missing"]
            )
        )
    )
    holdings_review["holding_dag_allowed_action_scope"] = holdings_review["symbol"].map(
        lambda symbol: str(
            holding_dag_assessments.get(str(symbol).strip().upper(), {}).get(
                "allowed_action_scope", "read_only_risk_reducing_sell_or_clear_only"
            )
        )
    )
    holdings_review["recall_context_holding_symbol"] = holdings_review["symbol"].map(
        lambda symbol: str(
            holding_dag_assessments.get(str(symbol).strip().upper(), {}).get(
                "recall_holding_symbol", ""
            )
        )
    )
    fundamental_coverage_by_symbol: dict[str, dict[str, Any]] = {}
    enhanced_data_flags_by_symbol: dict[str, dict[str, Any]] = {}
    intelligence_diagnostics: dict[str, dict[str, Any]] = {}
    for symbol, payload in branch_signals_by_symbol.items():
        reviewed_branchs = dict(payload.get("reviewed_branch_verdicts", {}) or {})
        fundamental_payload = dict(reviewed_branchs.get("fundamental", {}) or {})
        fundamental_meta = dict(fundamental_payload.get("metadata", {}) or {})
        fundamental_quality = dict(fundamental_meta.get("data_quality", {}) or {})
        snapshot_quality_by_symbol = dict(fundamental_quality.get("snapshot_quality_by_symbol", {}) or {})
        fundamental_coverage_by_symbol[symbol] = {
            "coverage_ratio": fundamental_quality.get("coverage_ratio"),
            "missing_modules": (
                dict(fundamental_quality.get("missing_modules", {}) or {}).get(symbol, [])
                if isinstance(fundamental_quality.get("missing_modules"), dict)
                else fundamental_quality.get("missing_modules", [])
            ),
            "snapshot_quality": dict(snapshot_quality_by_symbol.get(symbol, {}) or {}),
            "module_coverage": dict(fundamental_meta.get("module_coverage", {}) or {}),
        }
        enhanced_data_flags_by_symbol[symbol] = dict(snapshot_quality_by_symbol.get(symbol, {}) or {})
        if reviewed_branchs.get("intelligence"):
            intelligence_diagnostics[symbol] = dict(reviewed_branchs.get("intelligence", {}) or {})

    formal_warnings = collect_formal_report_warnings(
        target_date=str(completeness_after.get("effective_target_trade_date") or latest_trade_date or ""),
        dominant_local_snapshot_date=analysis_trade_date,
        completeness_state=diagnostic_completeness_after,
        holdings_review=holdings_review.to_dict(orient="records"),
        branch_diagnostics=branch_signals_by_symbol,
        fundamental_coverage_by_symbol=fundamental_coverage_by_symbol,
        enhanced_data_flags_by_symbol=enhanced_data_flags_by_symbol,
        intelligence_diagnostics=intelligence_diagnostics,
        review_layer_diagnostics={
            "effective_call_count": int(review_effective_summary.get("call_count", 0) or 0),
            "attempt_call_count": int(review_attempt_summary.get("call_count", 0) or 0),
            "degraded_symbols": list(degraded_symbols.keys()),
            "fallback_reasons": list(review_layer.get("fallback_reasons", []) or []),
            "codex_handoff": codex_handoff_active,
            "local_llm_disabled": bool(review_layer.get("local_llm_disabled", False)),
        },
    )
    holding_diagnostics = build_holding_decision_diagnostics(
        holdings_review=holdings_review.to_dict(orient="records"),
        warnings=formal_warnings,
        provisional_label_by_symbol={
            str(row.symbol): str(getattr(row, "recommended_action", ""))
            for row in holdings_review.itertuples()
        },
        data_date_by_symbol={str(row.symbol): analysis_trade_date for row in holdings_review.itertuples()},
        branch_signals_by_symbol=branch_signals_by_symbol,
    )
    decision_guardrail = apply_report_decision_guardrail(
        provisional_label="rebalance" if orders else "no_action",
        warnings=formal_warnings,
        holding_diagnostics=holding_diagnostics,
        llm_confidences=list(holdings_review["llm_confidence"]) if "llm_confidence" in holdings_review.columns else [],
    )
    if decision_guardrail.display_label in {"no_action_evidence_impaired", "hold_arbitrated"}:
        adjusted_diagnostics: list[HoldingDecisionDiagnostic] = []
        for item in holding_diagnostics:
            adjusted_diagnostics.append(
                HoldingDecisionDiagnostic(
                    symbol=item.symbol,
                    name=item.name,
                    data_date=item.data_date,
                    final_label=decision_guardrail.display_label,
                    branch_vs_final=(
                        "conflict_downgraded"
                        if item.branch_vs_final == "conflict_requires_arbitration"
                        else item.branch_vs_final
                    ),
                    llm_confidence=item.llm_confidence,
                    warning_codes=list(item.warning_codes),
                    decision_impact=(
                        item.decision_impact
                        if item.decision_impact != "none"
                        else "downgraded_final_label"
                    ),
                    arbitration_note=item.arbitration_note or decision_guardrail.arbitration_note,
                )
            )
        holding_diagnostics = adjusted_diagnostics

    diagnostic_by_symbol = {item.symbol: item for item in holding_diagnostics}

    def _diagnostic_attr(symbol: Any, attr: str, default: Any = "") -> Any:
        item = diagnostic_by_symbol.get(str(symbol).strip().upper())
        return getattr(item, attr, default) if item is not None else default

    holdings_review["report_data_date"] = holdings_review["symbol"].map(
        lambda value: _diagnostic_attr(value, "data_date", "")
    )
    holdings_review["report_guardrail_label"] = holdings_review["symbol"].map(
        lambda value: _diagnostic_attr(value, "final_label", "unknown")
    )
    holdings_review["report_branch_vs_final"] = holdings_review["symbol"].map(
        lambda value: _diagnostic_attr(value, "branch_vs_final", "unknown")
    )
    holdings_review["report_warning_codes"] = holdings_review["symbol"].map(
        lambda value: ",".join(_diagnostic_attr(value, "warning_codes", []))
    )
    holdings_review["report_decision_impact"] = holdings_review["symbol"].map(
        lambda value: _diagnostic_attr(value, "decision_impact", "none")
    )
    holdings_review["report_arbitration_note"] = holdings_review["symbol"].map(
        lambda value: _diagnostic_attr(value, "arbitration_note", "")
    )
    diagnostic_table = render_holding_diagnostic_markdown_table(holding_diagnostics)
    typed_warning_codes = sorted({warning.code for warning in formal_warnings})
    switch_rows_by_buy_symbol = {
        str(row.buy_symbol): row._asdict()
        for row in switch_plan_df.itertuples(index=False)
    }
    candidate_lines = [
        _format_candidate_advice_line(row, switch_rows_by_buy_symbol.get(str(row.symbol)))
        for row in candidate_pool.head(5).itertuples()
    ]
    factor_governance_runtime = _summarize_factor_governance_runtime(
        branch_signals_by_symbol
    )
    factor_governance_lines = _format_factor_governance_runtime_lines(
        factor_governance_runtime
    )
    factor_shadow_status = load_factor_library_shadow_status(
        root_dir=PROJECT_ROOT / "data" / "factor_library",
        as_of=now.strftime("%Y-%m-%d"),
    )
    factor_shadow_lines = render_factor_library_shadow_markdown(factor_shadow_status).splitlines()
    if factor_shadow_lines[:2] == ["## Factor Library Status (Read-only Shadow)", ""]:
        factor_shadow_lines = factor_shadow_lines[2:]
    top_switch_action = str(switch_plan_df.iloc[0]["action"]) if not switch_plan_df.empty else ""
    switch_now = top_switch_action == "switch_now"
    switch_prepare = top_switch_action == "prepare_switch"
    theme_pool_report_lines = _format_theme_pool_report_lines(
        theme_pool_summary,
        candidate_pool=candidate_pool,
        theme_symbols=_mapping_payload(theme_pool_audit.get("symbols")),
    )
    candidate_waterfall_lines = _format_candidate_decay_waterfall_report_lines(
        candidate_level_dag_status
    )
    legacy_overweight_lines = _format_legacy_overweight_holding_lines(holdings_review)
    legacy_overweight_summary_lines = (
        [
            (
                f"- 超限存量持仓：{len(legacy_overweight_lines)} 笔；"
                "本报告只提示，不强制卖出或改写目标，详见 5.2。"
            )
        ]
        if legacy_overweight_lines
        else ["- 超限存量持仓：无"]
    )

    report_lines = [
        "# A股激进科技制造策略正式复盘报告",
        "",
        "## 1. 记录信息",
        "",
        "- 市场：A股（CN）",
        "- 策略：`aggressive_tech_manufacturing`",
        f"- 上一条正式记录：`{source_record}`",
        f"- 执行基线 manifest：`{source_manifest.get('effective_manual_manifest_path') or 'UNCONFIRMED'}`",
        f"- 执行基线 ledger：`{source_manifest.get('effective_manual_ledger_path') or 'UNCONFIRMED'}`；SHA256=`{source_manifest.get('effective_manual_ledger_sha256') or 'legacy_unsealed'}`",
        f"- 本次正式记录时间：{timestamp_long}",
        f"- 盘中快照：{quote_snapshot or 'N/A'}",
        f"- 完整性校验：**{'已通过' if completeness_passed else '未通过'}**",
        f"- 决策数据口径：**{'盘中前日线+实时行情可用' if decision_data_sufficient else 'strict 日线完整性优先'}**",
        "- 分析口径：**直接基于本地已有数据，不自动补数，不把完整性校验作为正式结论前置阻断。**",
        "- 分析链路：**统一 DAG / review-layer（逐持仓主线复核）**",
        (
            f"- DAG 四分支执行状态：**{dag_four_branch_compliance['status']}**；"
            f"`complete={str(bool(dag_four_branch_compliance['complete'])).lower()}`"
        ),
        (
            f"- Candidate-level DAG：`{candidate_level_dag_status.get('candidate_generation_status', 'unknown')}`"
            + (
                f"，blocker=`{candidate_level_dag_status.get('blocker')}`"
                if candidate_level_dag_status.get("blocker")
                else ""
            )
        ),
        (
            "- Holding DAG："
            + "；".join(
                f"`{symbol}`={payload.get('holding_dag_status', 'blocked')}"
                + (
                    f"({','.join(payload.get('holding_dag_blockers', [])[:3])})"
                    if payload.get("holding_dag_blockers")
                    else ""
                )
                for symbol, payload in holding_dag_assessments.items()
            )
        ),
        "",
        "## 0. 正式结果速览",
        "",
        f"- 正式结论：**{rebalance_reason}**",
        f"- 报告展示标签：**`{decision_guardrail.display_label}`**",
        f"- 数据完整性状态：**{data_status}**",
        f"- 今日是否执行调仓：**{'是' if orders else '否'}**",
        *legacy_overweight_summary_lines,
        (
            f"- 实时成交价门禁：**未通过 {len(execution_price_rejections)} 笔，未写成交**"
            if execution_price_rejections
            else "- 实时成交价门禁：**通过或无待执行订单**"
        ),
        (
            f"- 今日是否执行换仓：**{'是' if switch_now else '否'}**"
            if switch_plan_df is not None
            else "- 今日是否执行换仓：**否**"
        ),
        (
            f"- Typed diagnostics：`{', '.join(typed_warning_codes)}`"
            if typed_warning_codes
            else "- Typed diagnostics：无"
        ),
        (
            f"- 决策护栏说明：{decision_guardrail.arbitration_note}"
            if str(decision_guardrail.arbitration_note).strip()
            else "- 决策护栏说明：无"
        ),
        (
            "- LLM 复核状态：**本地 LLM 未执行；Codex 接管解释**，本地调用 `0` 次"
            if codex_handoff_active
            else (
                f"- LLM 复核状态：**已执行**，原始尝试 `{review_attempt_summary.get('call_count', 0)}` 次"
                f"（成功 `{review_attempt_summary.get('success_count', 0)}` / 失败 `{review_attempt_summary.get('failed_count', 0)}` / "
                f"fallback `{review_attempt_summary.get('fallback_count', 0)}`），"
                f"有效输出 `{review_effective_summary.get('call_count', 0)}` 次，"
                f"`{review_attempt_summary.get('total_tokens', 0)}` tokens，"
                f"估算成本 `${float(review_attempt_summary.get('estimated_cost_usd', 0.0)):.6f}`"
            )
        ),
        (
            "- Review-layer 降级：`"
            + "，".join(sorted(degraded_symbols.keys()))
            + "`"
            if degraded_symbols
            else "- Review-layer 降级：无"
        ),
        (
            "- 备选投资建议："
            + "；".join(
                f"{row.symbol}({row.name})/{row.theme_label}"
                for row in candidate_pool.head(3).itertuples()
            )
            if not candidate_pool.empty
            else "- 备选投资建议：暂无"
        ),
        f"- 明日准备事项：{'；'.join(tomorrow_focus)}",
        "",
        "## 2. 数据状态与本地快照",
        "",
        f"- 当前运行时间：`{timestamp_long}`",
        f"- 首轮完整性：`{'通过' if completeness_before['complete'] else '未通过'}`，阻塞缺口 `{int(completeness_before['blocking_incomplete_count'])}` 个",
        f"- 是否执行补数：`{'是' if attempted_backfill else '否'}`",
        "- 本轮策略：即使存在数据滞后或局部缺口，也继续给出正式分析与正式建议，但会明确披露数据限制。",
        *data_snapshot_lines,
        "",
        "## 3. 当前组合盈亏判断",
        "",
        f"- 截至 `quote_snapshot={quote_snapshot or 'N/A'}`，组合总资产 **{total_value_after:,.2f} 元**，较初始资金 **{portfolio_pnl_after:,.2f} 元**，收益率 **{portfolio_pnl_after / initial_capital:.2%}**。",
        f"- 相对上一条正式记录 `{source_record}` 的 **{source_total_value:,.2f} 元**，当前盘中净值变动 **{total_value_after - source_total_value:,.2f} 元**。",
        "- 当前浮盈仓位：" + _format_top_holdings_by_unrealized_pnl(float_winners, positive=True),
        "- 当前浮亏前三：" + _format_top_holdings_by_unrealized_pnl(float_losers, positive=False),
        "- 相对上一条正式记录的正向收益贡献："
        + _format_top_delta_vs_source_record(holdings_review, positive=True),
        "- 相对上一条正式记录的拖累来源前三："
        + _format_top_delta_vs_source_record(holdings_review, positive=False),
        "",
        "## 4. A股整体市场风格与指数结构",
        "",
        "### 4.1 指数结构（盘中快照）",
        "",
    ]

    for code in index_quote_codes:
        payload = indices.get(code)
        if not payload:
            continue
        report_lines.append(f"- {payload['name']}：{payload['change_pct']:+.2f}%")

    report_lines.extend(
        [
            "",
            f"结论：{style_text}",
            "",
            "### 4.2 广度与市场内部状态（基于最新完整日线）",
            "",
            (
                f"- HS300：1日上涨占比 {breadth['hs300']['ret1_positive_ratio']:.1%}，"
                f"20日上涨占比 {breadth['hs300']['ret20_positive_ratio']:.1%}，"
                f"`MA20 > MA60` 占比 {breadth['hs300']['ma20_gt_ma60_ratio']:.1%}"
            ),
            (
                f"- ZZ500：1日上涨占比 {breadth['zz500']['ret1_positive_ratio']:.1%}，"
                f"20日上涨占比 {breadth['zz500']['ret20_positive_ratio']:.1%}，"
                f"`MA20 > MA60` 占比 {breadth['zz500']['ma20_gt_ma60_ratio']:.1%}"
            ),
            (
                f"- ZZ1000：1日上涨占比 {breadth['zz1000']['ret1_positive_ratio']:.1%}，"
                f"20日上涨占比 {breadth['zz1000']['ret20_positive_ratio']:.1%}，"
                f"`MA20 > MA60` 占比 {breadth['zz1000']['ma20_gt_ma60_ratio']:.1%}"
            ),
            "",
            "### 4.3 科技 / 高端制造主线强弱",
            "",
            f"- {strongest_theme_text}",
            f"- {weakest_theme_text}",
            (
                f"- 主题强弱排序：{', '.join(item['theme'] for item in theme_strength)}"
                if theme_strength
                else "- 主题强弱排序：暂无"
            ),
            "",
            "## 5. 当前策略持仓复盘",
            "",
            "### 5.1 持仓相对强弱",
            "",
            "- 今日相对最强："
            + _format_holding_snapshot_set(holdings_review.sort_values(
                ["today_change_pct", "score_full_market"], ascending=[False, False]
            ).head(3)),
            "- 今日相对最弱："
            + _format_holding_snapshot_set(holdings_review.sort_values(
                ["today_change_pct", "score_full_market"], ascending=[True, True]
            ).head(3)),
            "",
            "### 5.2 当前弱点与结构预警",
            "",
        ]
    )

    weak_rows = holdings_review[holdings_review["current_price"] < holdings_review["stage_stop_price"]]
    if weak_rows.empty:
        report_lines.append("- 目前持仓都在阶段止损位上方，尚未触发新的硬性减仓信号。")
    else:
        for row in weak_rows.itertuples():
            report_lines.append(
                f"- `{row.symbol}` 当前价 {row.current_price:.2f} 仍低于阶段止损位 {row.stage_stop_price:.2f}，"
                f"状态 `{row.position_role}`。"
            )
    report_lines.extend(legacy_overweight_lines)
    if "trailing_take_profit_status" in holdings_review.columns:
        giveback_values = pd.to_numeric(
            holdings_review.get("profit_giveback_ratio", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0.0)
        trailing_threshold_rows = holdings_review[
            giveback_values >= TRAILING_TAKE_PROFIT_REVIEW_GIVEBACK_RATIO
        ]
        trailing_watch_rows = holdings_review[
            holdings_review["trailing_take_profit_status"].astype(str).isin(
                ["hold_with_trailing_stop", "reduce_risk", "clear_risk", "cash_hold"]
            )
        ]
        trailing_unconfirmed_rows = holdings_review[
            holdings_review["trailing_take_profit_status"].astype(str).eq("unconfirmed")
        ]
        if trailing_threshold_rows.empty:
            report_lines.append("- 移动止盈：本轮未触发 20% 利润峰值回吐；逐票阈值已写入 holdings_review。")
        else:
            report_lines.append(
                "- 移动止盈 20% 阈值复核："
                + "；".join(
                    f"`{row.symbol}` 状态 `{row.trailing_take_profit_status}`，"
                    f"confirmed=`{str(bool(row.trailing_take_profit_confirmed)).lower()}`，"
                    f"回吐 {float(row.profit_giveback_ratio):.2%}，复核价 {float(row.trailing_stop_price):.2f}"
                    for row in trailing_threshold_rows.itertuples()
                )
            )
        if not trailing_watch_rows.empty:
            report_lines.append(
                "- 移动止盈可行动作状态："
                + "；".join(
                    f"`{row.symbol}` `{row.trailing_take_profit_status}`"
                    for row in trailing_watch_rows.itertuples()
                )
            )
        if not trailing_unconfirmed_rows.empty:
            report_lines.append(
                "- 移动止盈未确认："
                + "；".join(
                    f"`{row.symbol}` {row.trailing_take_profit_basis}"
                    for row in trailing_unconfirmed_rows.itertuples()
                )
            )

    report_lines.extend(
        [
            "",
            "### 5.3 统一 DAG / Review Layer 复核",
            "",
            (
                "- 本地 LLM 调用：`已禁用`；LLM 解释由 Codex 在运行后读取正式产物接管。"
                if codex_handoff_active
                else (
                    f"- 分支模型：`{review_model_role_metadata.get('resolved_branch_model') or review_model_role_metadata.get('branch_model') or 'N/A'}`；"
                    f"主模型：`{review_model_role_metadata.get('resolved_master_model') or review_model_role_metadata.get('master_model') or 'N/A'}`"
                )
            ),
            (
                f"- 原始尝试汇总：`{review_attempt_summary.get('call_count', 0)}` 次，"
                f"成功 `{review_attempt_summary.get('success_count', 0)}` 次，"
                f"失败 `{review_attempt_summary.get('failed_count', 0)}` 次，"
                f"fallback `{review_attempt_summary.get('fallback_count', 0)}` 次，"
                f"tokens `{review_attempt_summary.get('total_tokens', 0)}`，"
                f"成本 `${float(review_attempt_summary.get('estimated_cost_usd', 0.0)):.6f}`"
            ),
            (
                f"- 有效输出汇总：`{review_effective_summary.get('call_count', 0)}` 次，"
                f"成功 `{review_effective_summary.get('success_count', 0)}` 次，"
                f"tokens `{review_effective_summary.get('total_tokens', 0)}`，"
                f"成本 `${float(review_effective_summary.get('estimated_cost_usd', 0.0)):.6f}`"
            ),
            (
                "- review-layer fallback："
                + "；".join(review_layer.get("fallback_reasons", [])[:6])
                if review_layer.get("fallback_reasons")
                else "- review-layer fallback：无"
            ),
            (
                "- review-layer 降级标的："
                + "；".join(f"{symbol}: {reason}" for symbol, reason in list(degraded_symbols.items())[:6])
                if degraded_symbols
                else "- review-layer 降级标的：无"
            ),
            "",
            *_render_dag_compliance_markdown(dag_four_branch_compliance),
            "",
            "### 5.4 是否需要调仓",
            "",
            f"- 报告展示标签：`{decision_guardrail.display_label}`",
            (
                f"- 护栏仲裁说明：{decision_guardrail.arbitration_note}"
                if str(decision_guardrail.arbitration_note).strip()
                else "- 护栏仲裁说明：无"
            ),
            (
                f"- 触发 warning codes：`{', '.join(typed_warning_codes)}`"
                if typed_warning_codes
                else "- 触发 warning codes：无"
            ),
            "",
            "#### 5.4.1 决策诊断",
            "",
            diagnostic_table,
            "",
            "#### 5.4.2 正式建议",
            "",
            f"- 正式建议：**{'执行温和调仓' if orders else '本次不执行调仓'}**",
            (
                "- 原因：主线最强分支重新走强，今天更像内部修复而不是主线失效；弱支线虽然没有完全修复，但还不足以在单日里推翻长期主线。"
                if not orders
                else "- 原因：弱支线在完整日线与盘中都继续落后，已满足温和减仓条件。"
            ),
        ]
    )
    for row in holdings_review.itertuples():
        report_lines.append(_format_holding_advice_line(row))
    report_lines.extend(
        [
            "",
            "#### 5.4.3 证据质量与工程诊断",
            "",
            f"- 诊断汇总：{_format_warning_count_summary(formal_warnings)}",
            f"- Material 级证据缺口：{_format_warning_messages(formal_warnings, severity='material', limit=3)}",
            (
                f"- LLM / review-layer 状态：原始尝试 `{review_attempt_summary.get('call_count', 0)}` 次，"
                f"有效输出 `{review_effective_summary.get('call_count', 0)}` 次；"
                + (
                    "降级标的 `"
                    + "，".join(sorted(degraded_symbols.keys()))
                    + "`。"
                    if degraded_symbols
                    else "无逐标的 review-layer 降级。"
                )
            ),
            "- 工程诊断说明：provider、snapshot、旧 intelligence batch 等诊断仅用于说明证据等级，不直接作为逐票投资建议正文。",
            "",
            "##### 持仓诊断明细",
            "",
            diagnostic_table,
        ]
    )
    report_lines.extend(
        [
            "",
            "### 5.5 备选投资建议",
            "",
            (
                "- 本轮备选池来自全市场 v13 DAG：DeterministicFunnel → candidate-level 四分支 → Bayesian → RiskGuard/IC/PortfolioConstructor。"
                if not candidate_pool.empty
                else (
                    "- 本轮未提取到有效备选池；"
                    f"candidate_generation_status=`{candidate_level_dag_status.get('candidate_generation_status', 'unknown')}`，"
                    f"blocker=`{candidate_level_dag_status.get('blocker') or 'none'}`。"
                )
            ),
            *candidate_lines,
            *candidate_waterfall_lines,
            *theme_pool_report_lines,
            "",
            "### 5.6 现持仓与备选标的换仓比较",
            "",
            (
                "- 正式换仓结论：**执行换仓**，优先按 20% 试探性替换最弱持仓。"
                if switch_now
                else (
                    "- 正式换仓结论：**暂不换仓，但已形成预备换仓对象**。"
                    if switch_prepare
                    else "- 正式换仓结论：**暂不换仓**，现阶段以观察池跟踪为主。"
                )
            ),
            (
                "- 不换仓条件：当前本地 strict 快照仍不完整，先不把结构优势直接转成实单。"
                if not completeness_passed
                else "- 不换仓条件：若弱持仓重新收复阶段止损位且候选优势收敛，则维持原组合。"
            ),
            "",
        ]
    )
    if switch_plan_df.empty:
        report_lines.append("- 当前没有形成明确的一对一换仓比较。")
    else:
        for row in switch_plan_df.itertuples(index=False):
            report_lines.append(
                f"- `{row.sell_symbol}`（{row.sell_name}） vs `{row.buy_symbol}`（{row.buy_name}）："
                f"候选主线 `{row.buy_theme}`，Bayesian rank `{int(row.buy_bayesian_rank)}`，"
                f"posterior action score `{float(row.buy_posterior_action_score):.3f}`，"
                f"PortfolioConstructor 目标权重 `{float(row.buy_portfolio_target_weight):.2%}`；建议 `{row.action}`，"
                f"优先级 `{row.priority}`，比例提示 `{row.switch_ratio_hint}`；"
                f"触发阈值：{row.trigger_threshold}；不换仓条件：{row.no_switch_condition}"
            )

    report_lines.extend(
        [
            "",
            "### 5.7 FactorGovernanceProtocol v2 生产运行时",
            "",
            *factor_governance_lines,
            "",
            "### 5.8 Legacy Factor Library（只读影子观察）",
            "",
            *factor_shadow_lines,
            "",
            "## 6. 面向明日的观察重点和准备事项",
            "",
        ]
    )
    for idx, item in enumerate(tomorrow_focus, start=1):
        report_lines.append(f"{idx}. {item}")

    report_text = "\n".join(report_lines)

    pnl_summary = {
        "record_time": timestamp_long,
        "quote_snapshot": quote_snapshot,
        "initial_capital": initial_capital,
        "cash_before": round(cash_before, 2),
        "market_value_before": total_market_value_before,
        "total_value_before": total_value_before,
        "portfolio_pnl_before": portfolio_pnl_before,
        "portfolio_pnl_pct_before": round(_safe_pct(portfolio_pnl_before, initial_capital), 6),
        "realized_pnl_from_rebalance": realized_pnl,
        "cash_after": round(cash_after, 2),
        "market_value_after": total_market_value_after,
        "total_value_after": total_value_after,
        "portfolio_pnl_after": portfolio_pnl_after,
        "portfolio_pnl_pct_after": round(_safe_pct(portfolio_pnl_after, initial_capital), 6),
        "delta_vs_source_record": round(total_value_after - source_total_value, 2),
    }
    pnl_summary_df = pd.DataFrame([pnl_summary])

    manual_execution_manifest = _write_manual_execution_outputs(
        run_dir=run_dir,
        timestamp=timestamp,
        timestamp_long=timestamp_long,
        updated_ledger=updated_ledger,
        pnl_summary=pnl_summary,
        source_ledger=source_ledger,
        orders=orders,
        execution_price_rejections=execution_price_rejections,
        quote_by_symbol=quote_by_symbol,
        quote_snapshot=quote_snapshot,
        quote_error=quote_error,
        completeness_passed=completeness_passed,
        decision_data_sufficient=decision_data_sufficient,
        dag_four_branch_compliance=dag_four_branch_compliance,
        execution_price_gate=execution_price_gate,
        advisory_only=advisory_only,
        quote_provenance=quote_provenance,
    )
    holdings_review_output, holdings_review_invariant = _prune_holdings_review_to_effective_ledger(
        holdings_review,
        updated_ledger,
    )

    notes_text = _build_notes_payload(
        trade_date=now.strftime("%Y-%m-%d"),
        data_status=data_status,
        market_core_view=f"{style_text}{strongest_theme_text}{weakest_theme_text}",
        pnl_summary=pnl_summary,
        orders=orders,
        switch_plan_df=switch_plan_df,
        candidate_pool=candidate_pool,
        tomorrow_focus=tomorrow_focus,
        theme_pool_summary=theme_pool_summary,
        theme_symbols=_mapping_payload(theme_pool_audit.get("symbols")),
    )
    (base_dir / "latest_notes_payload.md").write_text(
        notes_text,
        encoding="utf-8",
    )
    formal_diagnostics_payload = {
        "warnings": [_jsonable(item.to_dict()) for item in formal_warnings],
        "holding_diagnostics": [_jsonable(item.to_dict()) for item in holding_diagnostics],
        "decision_guardrail": _jsonable(decision_guardrail.to_dict()),
        "typed_warning_codes": typed_warning_codes,
        "branch_diagnostics_by_symbol": _jsonable(branch_signals_by_symbol),
        "dag_four_branch_compliance": _jsonable(dag_four_branch_compliance),
        "candidate_level_dag_status": _jsonable(candidate_level_dag_status),
        "theme_candidate_pool": _jsonable(theme_pool_summary),
        "factor_governance_runtime": _jsonable(factor_governance_runtime),
        "trailing_take_profit": _jsonable(trailing_take_profit_summary),
        "holdings_review_ledger_invariant": _jsonable(holdings_review_invariant),
        "holding_dag_assessments": _jsonable(holding_dag_assessments),
    }
    runtime_profile = {
        "schema_version": "cn_aggressive_runtime_profile.v1",
        "market": "CN",
        "strategy": "aggressive_tech_manufacturing",
        "record_id": timestamp,
        "generated_at": _now_local().isoformat(),
        "snapshot_id": str(market_metrics_cache_meta.get("snapshot_id") or ""),
        "analysis_trade_date": analysis_trade_date,
        "total_elapsed_sec": round(time.time() - started, 3),
        "stages": [
            {
                "name": "market_metrics_prewarm",
                "status": str(market_metrics_cache_meta.get("status") or "unknown"),
                "elapsed_sec": round(float(market_metrics_cache_meta.get("compute_elapsed_sec", 0.0) or 0.0), 3),
                "metadata": _jsonable(market_metrics_cache_meta),
            }
        ],
    }

    manifest = {
        "market": "CN",
        "strategy": "aggressive_tech_manufacturing",
        "timestamp": timestamp,
        "recorded_at": timestamp_long,
        "source_record": source_record,
        "formal_record": True,
        "completeness_passed": completeness_passed,
        "decision_data_sufficient": decision_data_sufficient,
        "decision_data_mode": (
            "previous_day_daily_plus_realtime"
            if decision_data_sufficient and not completeness_passed
            else "strict_daily"
        ),
        "capital_cny": initial_capital,
        "quote_snapshot": quote_snapshot,
        "action_taken_today": bool(orders),
        "execution_price_gate": execution_price_gate,
        "analysis_chain": "unified_dag_review_layer_per_holding",
        "analysis_input_policy": (
            "previous_day_daily_plus_realtime_no_backfill_no_gate"
            if decision_data_sufficient and not completeness_passed
            else "local_snapshot_no_backfill_no_gate"
        ),
        "files": {
            "analysis_report": "analysis_report.md",
            "holdings_review": "holdings_review.csv",
            "candidate_pool": "candidate_pool.csv",
            "switch_plan": "switch_plan.csv",
            "orders": "orders.csv",
            "pnl_summary": "pnl_summary.csv",
            "market_snapshot": "market_snapshot.json",
            "runtime_profile": "runtime_profile.json",
            "manual_execution_manifest": "manual_execution_manifest.json",
            "manual_orders": "manual_switch_and_take_profit_orders.csv",
            "ledger_after_manual_switch": "ledger_after_manual_switch.csv",
            "daily_execution_review": "daily_execution_review.md",
        },
        "raw_exports": {
            "report": f"raw_exports/aggressive_portfolio_{timestamp}_formal_report.md",
            "orders": f"raw_exports/aggressive_portfolio_{timestamp}_formal_orders.csv",
            "pnl_summary": f"raw_exports/aggressive_portfolio_{timestamp}_formal_pnl_summary.csv",
            "holdings_review": f"raw_exports/aggressive_portfolio_{timestamp}_formal_holdings_review.csv",
            "candidate_pool": f"raw_exports/aggressive_portfolio_{timestamp}_formal_candidate_pool.csv",
            "switch_plan": f"raw_exports/aggressive_portfolio_{timestamp}_formal_switch_plan.csv",
            "runtime_profile": "raw_exports/runtime_profile.json",
            "manual_execution_manifest": f"raw_exports/aggressive_portfolio_{timestamp}_manual_execution_manifest.json",
            "manual_orders": f"raw_exports/aggressive_portfolio_{timestamp}_manual_switch_and_take_profit_orders.csv",
            "ledger_after_manual_switch": f"raw_exports/aggressive_portfolio_{timestamp}_ledger_after_manual_switch.csv",
            "daily_execution_review": f"raw_exports/aggressive_portfolio_{timestamp}_daily_execution_review.md",
        },
        "data_snapshot": {
            "latest_trade_date": latest_trade_date,
            "analysis_trade_date": analysis_trade_date,
            "strict_trade_date": completeness_after.get("strict_trade_date"),
            "stable_trade_date": completeness_after.get("stable_trade_date"),
            "effective_target_trade_date": completeness_after.get("effective_target_trade_date"),
            "freshness_mode": completeness_after.get("freshness_mode"),
            "coverage_ratio": completeness_after.get("coverage_ratio"),
            "blocking_incomplete_count": completeness_after.get("blocking_incomplete_count"),
            "decision_data_sufficient": decision_data_sufficient,
            "decision_data_mode": (
                "previous_day_daily_plus_realtime"
                if decision_data_sufficient and not completeness_passed
                else "strict_daily"
            ),
            "completeness": completeness_after,
            "download_report": str(download_report_path) if download_report_path else None,
            "market_metrics_cache": _jsonable(market_metrics_cache_meta),
        },
        "review_layer": {
            "reviewed_symbols": list(review_layer.get("reviewed_symbols", []) or []),
            "llm_usage_summary": review_attempt_summary,
            "llm_attempt_summary": review_attempt_summary,
            "llm_effective_summary": review_effective_summary,
            "model_role_metadata": review_model_role_metadata,
            "fallback_reasons": list(review_layer.get("fallback_reasons", []) or []),
            "degraded_symbols": degraded_symbols,
            "session_ids": dict(review_layer.get("session_ids", {}) or {}),
            "symbol_diagnostics": _jsonable(branch_signals_by_symbol),
            "codex_handoff": codex_handoff_active,
            "local_llm_disabled": bool(review_layer.get("local_llm_disabled", False)),
        },
        "dag_four_branch_compliance": _jsonable(dag_four_branch_compliance),
        "candidate_level_dag_status": _jsonable(candidate_level_dag_status),
        "candidate_generation_status": candidate_level_dag_status.get("candidate_generation_status"),
        "blocker": candidate_level_dag_status.get("blocker"),
        "theme_candidate_pool": _jsonable(theme_pool_summary),
        "factor_governance_runtime": _jsonable(factor_governance_runtime),
        "holding_dag_assessments": _jsonable(holding_dag_assessments),
        "trailing_take_profit": _jsonable(trailing_take_profit_summary),
        "candidate_pool": candidate_pool.to_dict(orient="records"),
        "switch_plan": switch_plan_df.to_dict(orient="records"),
        "formal_diagnostics": formal_diagnostics_payload,
        "holdings_review_ledger_invariant": _jsonable(holdings_review_invariant),
        "market_metrics_prewarm": _jsonable(market_metrics_cache_meta),
        "runtime_profile": runtime_profile,
        "manual_execution": _jsonable(manual_execution_manifest),
    }
    market_snapshot = {
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "quote_snapshot": quote_snapshot,
        "analysis_trade_date": analysis_trade_date,
        "indices": indices,
        "breadth": breadth,
        "market_metrics_prewarm": _jsonable(market_metrics_cache_meta),
        "runtime_profile": runtime_profile,
        "data_status": data_status,
        "completeness": completeness_after,
        "decision_data_sufficient": decision_data_sufficient,
        "decision_data_mode": (
            "previous_day_daily_plus_realtime"
            if decision_data_sufficient and not completeness_passed
            else "strict_daily"
        ),
        "execution_price_gate": execution_price_gate,
        "portfolio": {
            "total_value": total_value_after,
            "portfolio_pnl": portfolio_pnl_after,
            "portfolio_pnl_pct": round(_safe_pct(portfolio_pnl_after, initial_capital), 6),
            "delta_vs_source_record": round(total_value_after - source_total_value, 2),
        },
        "theme_strength": theme_strength,
        "candidate_pool": candidate_pool.to_dict(orient="records"),
        "switch_plan": switch_plan_df.to_dict(orient="records"),
        "review_layer": {
            "reviewed_symbols": list(review_layer.get("reviewed_symbols", []) or []),
            "llm_usage_summary": review_attempt_summary,
            "llm_attempt_summary": review_attempt_summary,
            "llm_effective_summary": review_effective_summary,
            "model_role_metadata": review_model_role_metadata,
            "fallback_reasons": list(review_layer.get("fallback_reasons", []) or []),
            "degraded_symbols": degraded_symbols,
            "session_ids": dict(review_layer.get("session_ids", {}) or {}),
            "symbol_diagnostics": _jsonable(branch_signals_by_symbol),
            "codex_handoff": codex_handoff_active,
            "local_llm_disabled": bool(review_layer.get("local_llm_disabled", False)),
        },
        "dag_four_branch_compliance": _jsonable(dag_four_branch_compliance),
        "candidate_level_dag_status": _jsonable(candidate_level_dag_status),
        "candidate_generation_status": candidate_level_dag_status.get("candidate_generation_status"),
        "blocker": candidate_level_dag_status.get("blocker"),
        "theme_candidate_pool": _jsonable(theme_pool_summary),
        "factor_governance_runtime": _jsonable(factor_governance_runtime),
        "holding_dag_assessments": _jsonable(holding_dag_assessments),
        "trailing_take_profit": _jsonable(trailing_take_profit_summary),
        "formal_diagnostics": formal_diagnostics_payload,
        "holdings_review_ledger_invariant": _jsonable(holdings_review_invariant),
        "download_report": str(download_report_path) if download_report_path else None,
        "quote_fetch_error": quote_error or None,
        "manual_execution": _jsonable(manual_execution_manifest),
    }

    _write_outputs(
        base_dir=base_dir,
        run_dir=run_dir,
        report_text=report_text,
        holdings_review=holdings_review_output,
        candidate_pool=candidate_pool,
        switch_plan_df=switch_plan_df,
        ledger=updated_ledger,
        orders_df=orders_df,
        pnl_summary_df=pnl_summary_df,
        manifest=manifest,
        market_snapshot=market_snapshot,
        theme_pool_audit=theme_pool_audit,
    )

    return {
        "timestamp": timestamp,
        "timestamp_long": timestamp_long,
        "run_dir": str(run_dir),
        "latest_trade_date": latest_trade_date,
        "analysis_trade_date": analysis_trade_date,
        "completeness_passed": completeness_passed,
        "decision_data_sufficient": decision_data_sufficient,
        "action_taken_today": bool(orders),
        "switch_action_today": top_switch_action or "none",
        "report_guardrail_label": decision_guardrail.display_label,
        "typed_warning_codes": typed_warning_codes,
        "data_status": data_status,
        "style_view": style_text,
        "review_layer_degraded_symbols": degraded_symbols,
        "codex_handoff": codex_handoff_active,
        "local_llm_disabled": bool(review_layer.get("local_llm_disabled", False)),
        "candidate_pool_top": candidate_pool.head(3).to_dict(orient="records"),
        "switch_plan": switch_plan_df.to_dict(orient="records"),
        "tech_mainline": {
            "strongest": strongest_theme_text,
            "weakest": weakest_theme_text,
        },
        "dag_four_branch_compliance": _jsonable(dag_four_branch_compliance),
        "candidate_level_dag_status": _jsonable(candidate_level_dag_status),
        "candidate_generation_status": candidate_level_dag_status.get("candidate_generation_status"),
        "blocker": candidate_level_dag_status.get("blocker"),
        "theme_candidate_pool": _jsonable(theme_pool_summary),
        "formal_diagnostics": formal_diagnostics_payload,
        "holdings_review_ledger_invariant": _jsonable(holdings_review_invariant),
        "market_metrics_prewarm": _jsonable(market_metrics_cache_meta),
        "full_market_metrics_cache": _jsonable(market_metrics_cache_meta),
        "manual_execution": _jsonable(manual_execution_manifest),
        "pnl_summary": pnl_summary,
        "elapsed_sec": round(time.time() - started, 2),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A股激进科技制造策略正式复盘跟踪器")
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--years", type=int, default=7)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--source-record", default=None)
    parser.add_argument("--allowed-stale-symbols", nargs="*", default=[])
    execution_mode = parser.add_mutually_exclusive_group()
    execution_mode.add_argument(
        "--advisory-only",
        dest="advisory_only",
        action="store_true",
        help="仅写建议和 pending/rejected 记录，不写本地模拟成交（默认）",
    )
    execution_mode.add_argument(
        "--allow-local-manual-fills",
        dest="advisory_only",
        action="store_false",
        help="显式授权在所有门禁通过后写本地/manual paper fill；仍不调用券商",
    )
    parser.set_defaults(advisory_only=True)
    quote_source = parser.add_mutually_exclusive_group()
    quote_source.add_argument(
        "--quote-input-json",
        default="",
        help="读取 results/ 下权限 0600 的本地 quote artifact；不回退外部 provider",
    )
    quote_source.add_argument(
        "--allow-live-quotes",
        action="store_true",
        default=False,
        help="显式允许 Tencent realtime quote 读取；默认禁用",
    )
    parser.add_argument(
        "--quote-max-age-seconds",
        type=int,
        default=DEFAULT_QUOTE_MAX_AGE_SECONDS,
        help="本地同日实时 quote 最大允许年龄，默认 300 秒",
    )
    parser.add_argument(
        "--decision-log-path",
        default=str(DEFAULT_DECISION_LOG_PATH),
        help="local/manual fill 必须由该 decision log 中同日 advisory+human_action 配对授权",
    )
    parser.add_argument(
        "--skip-market-metrics-prewarm",
        action="store_true",
        help="工程排障用：跳过启动前 full-market metrics 缓存预热",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_tracker(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
