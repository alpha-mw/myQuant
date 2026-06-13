"""Helper functions for full-market report rendering."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from quant_investor.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.market import name_map as _name_map
from quant_investor.market.config import get_market_settings

_STOCK_NAME_CACHE = _name_map._STOCK_NAME_CACHE
get_stock_name = _name_map.get_stock_name
_is_unknown_stock_name = _name_map.is_unknown_stock_name
load_stock_names = _name_map.load_stock_names

BRANCH_LABELS = {
    "quant": "量化",
    "fundamental": "基本面",
    "intelligence": "智能融合",
    "macro": "宏观",
}
BRANCH_SUPPORT_DENOMINATOR = len(CANONICAL_BRANCH_ORDER)


def _dedupe_text(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _sanitize_text(text: str) -> str:
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


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.70:
        return "高"
    if confidence >= 0.45:
        return "中"
    return "低"


def _branch_label(branch_name: str) -> str:
    return BRANCH_LABELS.get(branch_name, branch_name)


def _canonical_branch_map(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return v13 canonical branch payloads and drop legacy keys."""

    return {
        branch_name: payload[branch_name]
        for branch_name in CANONICAL_BRANCH_ORDER
        if branch_name in payload
    }


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        payload = value.to_dict()
        if isinstance(payload, dict):
            return dict(payload)
    if isinstance(value, dict):
        return dict(value)
    return {}


def _build_analysis_meta(
    all_results: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "market": "",
        "batch_count": 0,
        "total_stocks": 0,
        "category_count": 0,
        "symbols": [],
        "batch_traces": [],
        "analysis_kwargs": {},
    }
    first_meta: dict[str, Any] | None = None
    seen_categories = 0
    for category, batches in all_results.items():
        if batches:
            seen_categories += 1
        for batch in batches:
            meta["batch_count"] += 1
            meta["total_stocks"] += int(batch.get("stock_count", 0))
            meta["symbols"].extend(list(batch.get("stocks", [])))
            analysis_meta = batch.get("analysis_meta", {})
            if analysis_meta and first_meta is None:
                first_meta = dict(analysis_meta)
            if batch.get("execution_log"):
                meta["batch_traces"].append(
                    {
                        "category": category,
                        "batch_id": batch.get("batch_id"),
                        "log_tail": list(batch.get("execution_log", []))[-5:],
                    }
                )
    meta["category_count"] = seen_categories
    meta["symbols"] = list(
        dict.fromkeys(
            str(symbol) for symbol in meta["symbols"] if str(symbol).strip()
        )
    )
    if first_meta is not None:
        meta["market"] = str(first_meta.get("market", meta["market"]))
        meta["universe"] = str(
            first_meta.get("universe", meta.get("universe", ""))
        )
        meta["branch_model"] = str(first_meta.get("branch_model", ""))
        meta["master_model"] = str(first_meta.get("master_model", ""))
        meta["master_reasoning_effort"] = str(
            first_meta.get("master_reasoning_effort", "")
        )
        meta["agent_layer_enabled"] = bool(
            first_meta.get("agent_layer_enabled", False)
        )
        meta["model_role_metadata"] = dict(
            first_meta.get("model_role_metadata", {})
        )
        meta["execution_trace"] = dict(first_meta.get("execution_trace", {}))
        meta["what_if_plan"] = dict(first_meta.get("what_if_plan", {}))
        meta["global_context"] = dict(first_meta.get("global_context", {}))
        meta["data_snapshot"] = dict(first_meta.get("data_snapshot", {}))
        meta["symbol_research_packets"] = dict(
            first_meta.get("symbol_research_packets", {})
        )
        meta["shortlist"] = list(first_meta.get("shortlist", []))
        meta["portfolio_decision"] = dict(
            first_meta.get("portfolio_decision", {})
        )
        meta["review_bundle"] = dict(first_meta.get("review_bundle", {}))
        meta["ic_hints_by_symbol"] = dict(
            first_meta.get("ic_hints_by_symbol", {})
        )
        meta["branch_schema_version"] = str(
            first_meta.get("branch_schema_version", "")
        )
        meta["ic_protocol_version"] = str(
            first_meta.get("ic_protocol_version", "")
        )
        meta["report_protocol_version"] = str(
            first_meta.get("report_protocol_version", "")
        )
        meta["analysis_kwargs"] = dict(first_meta.get("analysis_kwargs", {}))
    return meta


def _default_branch_conclusion(branch_name: str, score: float) -> str:
    label = _branch_label(branch_name)
    if score >= 0.15:
        return f"{label}分支整体给出偏正面的执行结论。"
    if score <= -0.15:
        return f"{label}分支整体给出偏谨慎的执行结论。"
    return f"{label}分支整体维持中性结论。"


def category_name(category: str, market: str = "CN") -> str:
    settings = get_market_settings(market)
    return getattr(settings, "category_labels", {}).get(category, category)


def _derive_stock_support_drivers(payload: dict[str, Any]) -> list[str]:
    branch_scores = _canonical_branch_map(
        dict(payload.get("branch_scores", {}))
    )
    positive = [
        f"{_branch_label(name)}得分 {float(score):+.2f}"
        for name, score in sorted(
            branch_scores.items(), key=lambda item: item[1], reverse=True
        )
        if float(score) > 0.05
    ]
    if positive:
        return positive[:3]
    if float(payload.get("expected_upside", 0.0)) > 0.08:
        return [
            f"预期空间约 {float(payload.get('expected_upside', 0.0)):.1%}。"
        ]
    return ["当前主要依赖组合层的中性结论。"]


def _derive_stock_drag_drivers(payload: dict[str, Any]) -> list[str]:
    branch_scores = _canonical_branch_map(
        dict(payload.get("branch_scores", {}))
    )
    negative = [
        f"{_branch_label(name)}得分 {float(score):+.2f}"
        for name, score in sorted(
            branch_scores.items(), key=lambda item: item[1]
        )
        if float(score) < -0.05
    ]
    risk_flags = [
        _sanitize_text(item)
        for item in payload.get("risk_flags", [])
        if _sanitize_text(item)
    ]
    return _dedupe_text(negative[:2] + risk_flags[:2])[:3]


def _derive_stock_conclusion(payload: dict[str, Any]) -> str:
    support_count = int(payload.get("branch_positive_count", 0))
    confidence = float(payload.get("confidence", 0.0))
    expected_upside = float(payload.get("expected_upside", 0.0))
    if support_count >= BRANCH_SUPPORT_DENOMINATOR and confidence >= 0.55:
        return (
            f"{payload['symbol']} 当前获得 {support_count}/"
            f"{BRANCH_SUPPORT_DENOMINATOR} "
            f"个 v13 分支支持，预期空间约 {expected_upside:.1%}。"
        )
    if support_count >= 3 and confidence >= 0.40:
        return f"{payload['symbol']} 当前结论偏正，但更适合分批跟踪。"
    return f"{payload['symbol']} 当前信号仍需观察，暂不宜激进执行。"


def _safe_average(values: list[float], default: float = 0.0) -> float:
    normalized = [float(value) for value in values if value is not None]
    return sum(normalized) / len(normalized) if normalized else default


def _normalize_with_cap(
    raw_scores: dict[str, float],
    total_target_exposure: float,
    max_single_weight: float,
) -> dict[str, float]:
    positive_scores = {
        symbol: score for symbol, score in raw_scores.items() if score > 0
    }
    if not positive_scores or total_target_exposure <= 0:
        return {}

    remaining = dict(positive_scores)
    weights = {symbol: 0.0 for symbol in positive_scores}
    remaining_exposure = total_target_exposure

    while remaining and remaining_exposure > 1e-8:
        total_score = sum(remaining.values())
        if total_score <= 0:
            break

        overflow_symbols = []
        for symbol, score in list(remaining.items()):
            proposed = remaining_exposure * score / total_score
            if proposed > max_single_weight + 1e-8:
                weights[symbol] = max_single_weight
                remaining_exposure -= max_single_weight
                overflow_symbols.append(symbol)

        if overflow_symbols:
            for symbol in overflow_symbols:
                remaining.pop(symbol, None)
            continue

        for symbol, score in remaining.items():
            weights[symbol] = remaining_exposure * score / total_score
        break

    return {symbol: weight for symbol, weight in weights.items() if weight > 0}


def _build_market_summary(
    all_results: dict[str, list[dict[str, Any]]], market: str = "CN"
) -> dict[str, Any]:
    settings = get_market_settings(market)
    summary: dict[str, Any] = {
        "market": settings.market,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_stocks": 0,
        "total_batches": 0,
        "categories": {},
    }

    for category, results in all_results.items():
        if not results:
            continue
        category_stocks = sum(item.get("stock_count", 0) for item in results)
        summary["total_stocks"] += category_stocks
        summary["total_batches"] += len(results)

        branch_scores: dict[str, list[float]] = {}
        candidate_count = 0
        for item in results:
            candidate_count += len(
                item.get("strategy", {}).get("candidate_symbols", [])
            )
            for name, branch in _canonical_branch_map(
                dict(item.get("branches", {}))
            ).items():
                branch_scores.setdefault(name, []).append(
                    float(branch.get("score", 0.0))
                )

        summary["categories"][category] = {
            "category_name": category_name(category, settings.market),
            "batch_count": len(results),
            "stock_count": category_stocks,
            "candidate_count": candidate_count,
            "avg_target_exposure": _safe_average(
                [
                    item.get("strategy", {}).get("target_exposure", 0.0)
                    for item in results
                ]
            ),
            "avg_branch_scores": {
                name: _safe_average(scores)
                for name, scores in branch_scores.items()
            },
        }
    return summary

__all__ = [
    "BRANCH_LABELS",
    "BRANCH_SUPPORT_DENOMINATOR",
    "_STOCK_NAME_CACHE",
    "_branch_label",
    "_build_analysis_meta",
    "_build_market_summary",
    "_canonical_branch_map",
    "_confidence_label",
    "_dedupe_text",
    "_default_branch_conclusion",
    "_derive_stock_conclusion",
    "_derive_stock_drag_drivers",
    "_derive_stock_support_drivers",
    "_is_unknown_stock_name",
    "_normalize_with_cap",
    "_safe_average",
    "_sanitize_text",
    "_to_mapping",
    "category_name",
    "get_stock_name",
    "load_stock_names",
]
