"""Review-layer serialization and DAG compliance helpers for the CN tracker."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd

from quant_investor.llm_provider_priority import coerce_review_model_priority
from quant_investor.monitoring.cn_aggressive_utils import _jsonable, _plain_dict
from quant_investor.research_run_config import ResolvedReviewModels


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_DAG_BRANCHES = ("quant", "fundamental", "intelligence", "macro")

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


def _serialize_reviewed_branch_verdicts(result: Any, symbol: str) -> dict[str, Any]:
    payload = dict(getattr(result, "reviewed_research_by_symbol", {}) or {}).get(symbol, {})
    reviewed = (
        {str(name): _jsonable(verdict) for name, verdict in payload.items()}
        if isinstance(payload, dict)
        else {}
    )
    branch_summaries = dict(getattr(result, "reviewed_branch_summaries", {}) or {})
    for branch_name in ("quant", "macro"):
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


def _quant_evidence_limited(payload: dict[str, Any]) -> bool:
    metadata = _as_mapping(payload.get("metadata"))
    factor_mode = str(metadata.get("factor_mode", "")).strip()
    runtime = _as_mapping(metadata.get("mined_factor_runtime"))
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
        factor_mode == "legacy_proxy_fallback"
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
