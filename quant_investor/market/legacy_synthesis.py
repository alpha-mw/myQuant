"""Compatibility synthesis from market DAG artifacts to report batches."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from quant_investor.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.market.config import get_market_settings
from quant_investor.market.name_map import get_stock_name


def _dedupe_text(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _sanitize_text(text: Any) -> str:
    normalized = str(text or "").strip()
    lowered = normalized.lower()
    if not normalized:
        return ""
    if "could not infer frequency" in lowered:
        return "部分批次 K 线深度模型未完成频率对齐，已自动回退统计预测。"
    if "provider_missing" in lowered or "snapshot_missing" in lowered:
        return "部分模块当前缺少覆盖，已只计入数据覆盖说明。"
    if "provider_error" in lowered:
        return "部分数据接口本轮不可用，已只计入数据覆盖说明。"
    if "timeout" in lowered:
        return "部分批次模型阶段超时，已自动保留基础结论。"
    if lowered == "unknown":
        return "已按默认状态处理。"
    return normalized


def _canonical_branch_map(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        branch_name: payload[branch_name]
        for branch_name in CANONICAL_BRANCH_ORDER
        if branch_name in payload
    }


def _category_name(category: str, market: str = "CN") -> str:
    settings = get_market_settings(market)
    return getattr(settings, "category_labels", {}).get(category, category)


def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        payload = value.to_dict()
        return dict(payload) if isinstance(payload, Mapping) else {}
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _build_legacy_recommendation_from_dag(
    *,
    symbol: str,
    packet: Any,
    shortlist_item: Any | None,
    portfolio_decision: Any,
    category: str,
    market: str,
    total_capital: float,
) -> dict[str, Any]:
    category_label = _category_name(category, market)
    branch_scores = _canonical_branch_map(
        dict(getattr(packet, "branch_scores", {}) or {})
    )
    branch_confidences = _canonical_branch_map(
        dict(getattr(packet, "branch_confidences", {}) or {})
    )
    branch_theses = _canonical_branch_map(
        dict(getattr(packet, "branch_theses", {}) or {})
    )
    score_values = [float(value) for value in branch_scores.values()]
    confidence_values = [float(value) for value in branch_confidences.values()]
    consensus_score = sum(score_values) / max(len(score_values), 1)
    branch_positive_count = sum(1 for value in score_values if value > 0)
    target_weights = getattr(portfolio_decision, "target_weights", {}) or {}
    target_positions = (
        getattr(portfolio_decision, "target_positions", {}) or {}
    )
    suggested_weight = float(
        target_weights.get(
            symbol,
            getattr(shortlist_item, "suggested_weight", 0.0)
            if shortlist_item is not None
            else 0.0,
        )
    )
    rank_score = float(
        getattr(shortlist_item, "rank_score", consensus_score)
        if shortlist_item is not None
        else consensus_score
    )
    confidence = float(
        getattr(
            shortlist_item,
            "confidence",
            sum(confidence_values) / max(len(confidence_values), 1)
            if confidence_values
            else 0.0,
        )
    )
    metadata = getattr(packet, "metadata", {}) or {}
    current_price = float(metadata.get("latest_close", 0.0) or 0.0)
    if current_price <= 0:
        current_price = float(
            metadata.get("price_summary", {}).get("latest_close", 0.0) or 0.0
        )
    if current_price <= 0:
        current_price = max(rank_score, 0.01) * 100.0

    recommended_entry_price = current_price
    target_price = current_price * (1 + max(rank_score, 0.0) * 0.25)
    stop_loss_price = current_price * (0.92 if rank_score >= 0 else 0.88)
    support_drivers = list(getattr(shortlist_item, "rationale", []) or [])
    if not support_drivers:
        support_drivers = [
            text for text in branch_theses.values() if str(text).strip()
        ]
    drag_drivers = (
        list(getattr(packet, "diagnostic_notes", []) or [])
        + list(getattr(packet, "risk_flags", []) or [])
    )
    weight_cap_reasons = list(getattr(shortlist_item, "risk_flags", []) or [])
    action = getattr(shortlist_item, "action", None)
    action_value = (
        action.value
        if hasattr(action, "value")
        else str(action or "buy").lower()
    )
    company_name = str(
        getattr(shortlist_item, "company_name", "")
        or getattr(packet, "company_name", "")
        or get_stock_name(symbol, market=market)
    ).strip()
    return {
        "symbol": symbol,
        "company_name": company_name,
        "name": company_name,
        "category": category,
        "category_name": category_label,
        "action": action_value,
        "current_price": round(current_price, 4),
        "recommended_entry_price": round(recommended_entry_price, 4),
        "target_price": round(target_price, 4),
        "stop_loss_price": round(stop_loss_price, 4),
        "entry_price_range": {
            "low": round(current_price * 0.98, 4),
            "high": round(current_price * 1.02, 4),
        },
        "portfolio_weight": round(suggested_weight, 4),
        "suggested_weight": round(suggested_weight, 4),
        "portfolio_amount": round(suggested_weight * total_capital, 2),
        "expected_upside": round(max(rank_score, 0.0), 4),
        "confidence": round(confidence, 4),
        "consensus_score": round(consensus_score, 4),
        "model_expected_return": round(max(consensus_score, 0.0), 4),
        "branch_positive_count": int(branch_positive_count),
        "support_drivers": _dedupe_text([
            _sanitize_text(item) for item in support_drivers
        ])[:5],
        "drag_drivers": _dedupe_text([
            _sanitize_text(item) for item in drag_drivers
        ])[:5],
        "weight_cap_reasons": _dedupe_text([
            _sanitize_text(item) for item in weight_cap_reasons
        ])[:5],
        "risk_flags": _dedupe_text([
            _sanitize_text(item)
            for item in list(getattr(shortlist_item, "risk_flags", []) or [])
            + list(getattr(packet, "risk_flags", []) or [])
        ])[:5],
        "one_line_conclusion": str(
            getattr(shortlist_item, "rationale", [])[0]
            if shortlist_item is not None
            and getattr(shortlist_item, "rationale", [])
            else next(
                (
                    text
                    for text in branch_theses.values()
                    if str(text).strip()
                ),
                f"{symbol} 已进入组合候选。",
            )
        ),
        "data_source_status": "real" if current_price > 0 else "synthetic",
        "lot_size": 100,
        "macro_score": float(metadata.get("macro_score", 0.0)),
        "global_quant_score": float(
            metadata.get("global_quant_summary", {}).get("final_score", 0.0)
        ),
        "current_weight": float(target_weights.get(symbol, 0.0)),
        "target_position": float(target_positions.get(symbol, 0.0)),
    }


def synthesize_legacy_analysis_results_from_dag(
    *,
    dag_artifacts: dict[str, Any],
    market: str,
    universe: str,
    categories: list[str],
    total_capital: float,
) -> dict[str, list[dict[str, Any]]]:
    packets = dict(dag_artifacts.get("symbol_research_packets", {}) or {})
    shortlist = list(dag_artifacts.get("shortlist", []) or [])
    portfolio_decision = dag_artifacts.get("portfolio_decision")
    branch_summaries = dict(dag_artifacts.get("branch_summaries", {}) or {})
    selected_categories = list(categories or [universe])
    shortlist_by_symbol = {
        getattr(item, "symbol", ""): item for item in shortlist
    }
    symbols_by_category: dict[str, list[str]] = {
        category: [] for category in selected_categories
    }
    for symbol, packet in packets.items():
        category = str(getattr(packet, "category", "") or universe)
        if category not in symbols_by_category:
            symbols_by_category[category] = []
        symbols_by_category[category].append(symbol)

    branches_as_dict: dict[str, dict[str, Any]] = {}
    for name, branch in _canonical_branch_map(branch_summaries).items():
        branch_payload = _to_dict(branch)
        branches_as_dict[name] = {
            "score": float(
                branch_payload.get(
                    "score",
                    branch_payload.get("final_score", 0.0),
                )
            ),
            "confidence": float(
                branch_payload.get(
                    "confidence",
                    branch_payload.get("final_confidence", 0.0),
                )
            ),
            "conclusion": str(
                branch_payload.get(
                    "conclusion",
                    branch_payload.get("thesis", ""),
                )
            ),
            "support_drivers": [
                str(item)
                for item in branch_payload.get(
                    "support_drivers",
                    branch_payload.get("coverage_notes", []),
                )
            ],
            "drag_drivers": [
                str(item)
                for item in branch_payload.get(
                    "drag_drivers",
                    branch_payload.get("diagnostic_notes", []),
                )
            ],
            "investment_risks": [
                str(item)
                for item in branch_payload.get(
                    "investment_risks",
                    branch_payload.get("risks", []),
                )
            ],
            "coverage_notes": [
                str(item) for item in branch_payload.get("coverage_notes", [])
            ],
            "diagnostic_notes": [
                str(item)
                for item in branch_payload.get("diagnostic_notes", [])
            ],
            "module_coverage": dict(branch_payload.get("module_coverage", {})),
            "debate_statuses": [
                str(item) for item in branch_payload.get("debate_statuses", [])
            ],
            "metadata": dict(branch_payload.get("metadata", {})),
        }
    execution_trace = dag_artifacts.get("execution_trace")
    execution_log = []
    if execution_trace is not None:
        steps = (
            getattr(execution_trace, "steps", [])
            if hasattr(execution_trace, "steps")
            else []
        )
        for step in steps:
            conclusion = (
                getattr(step, "conclusion", "")
                if hasattr(step, "conclusion")
                else ""
            )
            stage = (
                getattr(step, "stage", "")
                if hasattr(step, "stage")
                else ""
            )
            execution_log.append(f"{stage}: {conclusion}".strip(": "))

    all_results: dict[str, list[dict[str, Any]]] = {}
    for category in selected_categories:
        symbols = list(dict.fromkeys(symbols_by_category.get(category, [])))
        recommendations = []
        risk_constraints = (
            getattr(portfolio_decision, "risk_constraints", {}) or {}
        )
        for symbol in symbols:
            packet = packets.get(symbol)
            if packet is None:
                continue
            shortlist_item = shortlist_by_symbol.get(symbol)
            recommendation = _build_legacy_recommendation_from_dag(
                symbol=symbol,
                packet=packet,
                shortlist_item=shortlist_item,
                portfolio_decision=portfolio_decision,
                category=category,
                market=market,
                total_capital=total_capital,
            )
            if recommendation.get("data_source_status") == "real":
                recommendations.append(recommendation)
        all_results[category] = [
            {
                "market": market,
                "universe": universe,
                "category": category,
                "category_name": _category_name(category, market),
                "batch_id": 1,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "stocks": symbols,
                "stock_count": len(symbols),
                "branches": branches_as_dict,
                "strategy": {
                    "target_exposure": float(
                        getattr(portfolio_decision, "target_exposure", 0.0)
                    ),
                    "style_bias": str(
                        getattr(portfolio_decision, "metadata", {}).get(
                            "style_bias",
                            getattr(portfolio_decision, "target_positions", {})
                            and "均衡"
                            or "防御",
                        )
                    ),
                    "risk_summary": dict(
                        risk_constraints.get("risk_decision", {})
                    ),
                    "candidate_symbols": [
                        getattr(item, "symbol", "") for item in shortlist
                    ],
                    "data_quality_issue_count": len(
                        dag_artifacts.get("data_quality_issues", []) or []
                    ),
                },
                "recommendations": recommendations,
                "execution_log": list(execution_log),
                "analysis_meta": {
                    "global_context": _to_dict(
                        dag_artifacts.get("global_context")
                    ),
                    "data_snapshot": dict(
                        dag_artifacts.get("data_snapshot", {})
                        or (
                            getattr(
                                dag_artifacts.get("global_context"),
                                "metadata",
                                {},
                            )
                            or {}
                        ).get("data_snapshot", {})
                    ),
                    "symbol_research_packets": {
                        symbol: _to_dict(packet)
                        for symbol, packet in packets.items()
                    },
                    "shortlist": [_to_dict(item) for item in shortlist],
                    "portfolio_decision": _to_dict(portfolio_decision),
                    "model_role_metadata": _to_dict(
                        dag_artifacts.get("model_role_metadata")
                    ),
                    "what_if_plan": _to_dict(
                        dag_artifacts.get("what_if_plan")
                    ),
                    "execution_trace": _to_dict(
                        dag_artifacts.get("execution_trace")
                    ),
                    "review_bundle": _to_dict(
                        dag_artifacts.get("review_bundle")
                    ),
                    "ic_hints_by_symbol": dict(
                        dag_artifacts.get("review_bundle").ic_hints_by_symbol
                        if dag_artifacts.get("review_bundle")
                        else {}
                    ),
                    "branch_schema_version": getattr(
                        dag_artifacts.get("review_bundle"),
                        "branch_schema_version",
                        "",
                    ),
                    "ic_protocol_version": getattr(
                        dag_artifacts.get("review_bundle"),
                        "ic_protocol_version",
                        "",
                    ),
                    "report_protocol_version": getattr(
                        dag_artifacts.get("review_bundle"),
                        "report_protocol_version",
                        "",
                    ),
                },
            }
        ]
    return all_results


__all__ = ["synthesize_legacy_analysis_results_from_dag"]
