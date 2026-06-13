"""Dependency-injected review runtime for the CN tracker."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from quant_investor.monitoring.cn_aggressive_review_layer import (
    _llm_usage_summary_to_dict,
    _serialize_reviewed_branch_verdicts,
    _serialize_symbol_review_bundle,
    _trade_recommendation_to_dict,
)
from quant_investor.monitoring.cn_aggressive_utils import _plain_dict, _safe_float
from quant_investor.research_run_config import ResolvedReviewModels


def run_unified_review_mainline_for_holdings(
    *,
    source_ledger: pd.DataFrame,
    latest_trade_date: str,
    source_record: str,
    quant_investor_cls: type[Any],
    load_daily_config_llm_settings: Callable[[], dict[str, Any]],
    llm_handoff_reason_fn: Callable[[], str],
    sleep_fn: Callable[[float], Any],
) -> dict[str, Any]:
    review_by_symbol: dict[str, dict[str, Any]] = {}
    degraded_symbols: dict[str, str] = {}
    llm_settings = load_daily_config_llm_settings()
    requested_agent_layer = bool(llm_settings.pop("enable_agent_layer", True))
    handoff_reason = llm_handoff_reason_fn()
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
        investor = quant_investor_cls(
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
            "codex_handoff": codex_handoff_active,
            "local_llm_disabled": bool(handoff_reason),
        }
        sleep_fn(0.8)

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
