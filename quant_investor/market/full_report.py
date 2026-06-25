"""Full-market report rendering for market analysis artifacts."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from quant_investor.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.market import name_map as _name_map
from quant_investor.market.config import get_market_settings
from quant_investor.reporting.theme_governance_renderer import (
    append_theme_governance_section_once,
)
from quant_investor.reporting.theme_renderer import render_theme_rotation_markdown
from quant_investor.reporting.theme_shadow_renderer import (
    append_theme_production_overlay_section_once,
    append_theme_shadow_section_once,
)

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


def _extract_theme_shadow_monitor(meta: Mapping[str, Any]) -> dict[str, Any]:
    direct = meta.get("theme_shadow_monitor")
    if isinstance(direct, Mapping):
        return dict(direct)

    global_context = meta.get("global_context")
    if isinstance(global_context, Mapping):
        context_metadata = global_context.get("metadata")
        if isinstance(context_metadata, Mapping):
            monitor = context_metadata.get("theme_shadow_monitor")
            if isinstance(monitor, Mapping):
                return dict(monitor)
    return {}


def _extract_theme_production_overlay(meta: Mapping[str, Any]) -> dict[str, Any]:
    direct = meta.get("theme_production_overlay")
    if isinstance(direct, Mapping):
        return dict(direct)

    global_context = meta.get("global_context")
    if isinstance(global_context, Mapping):
        context_metadata = global_context.get("metadata")
        if isinstance(context_metadata, Mapping):
            overlay = context_metadata.get("theme_production_overlay")
            if isinstance(overlay, Mapping):
                return dict(overlay)
    review_bundle = meta.get("review_bundle")
    if isinstance(review_bundle, Mapping):
        bundle_metadata = review_bundle.get("metadata")
        if isinstance(bundle_metadata, Mapping):
            overlay = bundle_metadata.get("theme_production_overlay")
            if isinstance(overlay, Mapping):
                return dict(overlay)
    return {}


def _extract_theme_metadata_payload(
    meta: Mapping[str, Any],
    key: str,
) -> dict[str, Any]:
    direct = meta.get(key)
    if isinstance(direct, Mapping):
        return dict(direct)

    global_context = meta.get("global_context")
    if isinstance(global_context, Mapping):
        context_metadata = global_context.get("metadata")
        if isinstance(context_metadata, Mapping):
            payload = context_metadata.get(key)
            if isinstance(payload, Mapping):
                return dict(payload)
    return {}


def _append_theme_sections_once(
    markdown: str,
    *,
    theme_rotation: Mapping[str, Any] | None,
    theme_governance: Mapping[str, Any] | None,
    theme_production_overlay: Mapping[str, Any] | None,
    theme_shadow_monitor: Mapping[str, Any] | None,
    governance_max_rows: int = 20,
) -> str:
    text = str(markdown or "")
    if "## 主题轮动雷达" not in text:
        rotation_section = render_theme_rotation_markdown(theme_rotation)
        if rotation_section:
            text = text.rstrip() + "\n\n" + rotation_section.strip() + "\n"
    text = append_theme_governance_section_once(
        text,
        theme_governance,
        max_rows=governance_max_rows,
    )
    text = append_theme_production_overlay_section_once(
        text,
        theme_production_overlay,
    )
    return append_theme_shadow_section_once(text, theme_shadow_monitor)


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
        theme_shadow_monitor = _extract_theme_shadow_monitor(first_meta)
        if theme_shadow_monitor:
            meta["theme_shadow_monitor"] = theme_shadow_monitor
        theme_production_overlay = _extract_theme_production_overlay(first_meta)
        if theme_production_overlay:
            meta["theme_production_overlay"] = theme_production_overlay
        theme_rotation = _extract_theme_metadata_payload(
            first_meta,
            "theme_rotation",
        )
        if theme_rotation:
            meta["theme_rotation"] = theme_rotation
        theme_governance = _extract_theme_metadata_payload(
            first_meta,
            "theme_governance",
        )
        if theme_governance:
            meta["theme_governance"] = theme_governance
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


def build_full_market_trade_plan(
    all_results: dict[str, list[dict[str, Any]]],
    market: str = "CN",
    total_capital: float = 1_000_000,
    top_k: int = 12,
) -> dict[str, Any]:
    settings = get_market_settings(market)
    summary = _build_market_summary(all_results, market=settings.market)
    collected: list[dict[str, Any]] = []

    for category, batches in all_results.items():
        for batch in batches:
            batch_target_exposure = float(
                batch.get("strategy", {}).get("target_exposure", 0.0)
            )
            batch_style_bias = batch.get("strategy", {}).get(
                "style_bias", "均衡"
            )
            batch_risk_summary = batch.get("strategy", {}).get(
                "risk_summary", {}
            )
            for recommendation in batch.get("recommendations", []):
                if recommendation.get("action") != "buy":
                    continue
                if recommendation.get("data_source_status") != "real":
                    continue
                payload = dict(recommendation)
                payload["category"] = category
                payload["category_name"] = category_name(
                    category, settings.market
                )
                raw_company_name = str(
                    payload.get("company_name", "") or ""
                ).strip()
                company_name = (
                    get_stock_name(
                        payload.get("symbol", ""), market=settings.market
                    )
                    if _is_unknown_stock_name(raw_company_name)
                    else raw_company_name
                )
                payload["company_name"] = company_name
                raw_name = str(payload.get("name", "") or "").strip()
                payload["name"] = (
                    company_name
                    if _is_unknown_stock_name(raw_name)
                    else raw_name
                )
                payload["batch_target_exposure"] = batch_target_exposure
                payload["style_bias"] = batch_style_bias
                payload["risk_level"] = batch_risk_summary.get(
                    "risk_level", "normal"
                )
                payload["rank_score"] = (
                    max(float(payload.get("suggested_weight", 0.0)), 0.001)
                    * (
                        1
                        + max(float(payload.get("consensus_score", 0.0)), 0.0)
                    )
                    * (
                        1
                        + max(
                            float(payload.get("model_expected_return", 0.0)),
                            0.0,
                        )
                    )
                    * (0.8 + float(payload.get("confidence", 0.0)))
                    * (
                        1
                        + float(payload.get("branch_positive_count", 0))
                        / BRANCH_SUPPORT_DENOMINATOR
                    )
                )
                collected.append(payload)

    deduped: dict[str, dict[str, Any]] = {}
    for item in sorted(
        collected, key=lambda entry: entry["rank_score"], reverse=True
    ):
        deduped.setdefault(item["symbol"], item)

    ranked = list(deduped.values())[:top_k]
    if not ranked:
        return {
            "market_summary": summary,
            "portfolio_plan": {
                "total_capital": total_capital,
                "target_exposure": 0.0,
                "planned_investment": 0.0,
                "cash_reserve": total_capital,
                "selected_count": 0,
                "style_bias": "防御",
                "max_single_weight": 0.0,
                "category_exposure": {},
                "execution_notes": [
                    "当前没有满足真实数据与买入条件的候选标的。"
                ],
            },
            "recommendations": [],
        }

    weighted_exposure_values = []
    for batches in all_results.values():
        for batch in batches:
            stock_count = max(int(batch.get("stock_count", 0)), 1)
            weighted_exposure_values.extend(
                [float(batch.get("strategy", {}).get("target_exposure", 0.0))]
                * stock_count
            )

    target_exposure = min(
        max(_safe_average(weighted_exposure_values, default=0.35), 0.15), 0.80
    )
    max_single_weight = min(
        0.12, max(0.05, target_exposure / max(len(ranked), 1) * 2.2)
    )

    active = ranked
    for _ in range(3):
        weight_map = _normalize_with_cap(
            {item["symbol"]: float(item["rank_score"]) for item in active},
            total_target_exposure=target_exposure,
            max_single_weight=max_single_weight,
        )
        filtered_active = []
        for item in active:
            weight = weight_map.get(item["symbol"], 0.0)
            entry_price = float(
                item.get("recommended_entry_price")
                or item.get("current_price")
                or 0.0
            )
            lot_size = int(
                item.get("lot_size", getattr(settings, "lot_size", 100))
            )
            if entry_price <= 0:
                continue
            minimum_ticket = entry_price * lot_size
            if total_capital * weight + 1e-8 < minimum_ticket:
                continue
            filtered_active.append(item)
        if len(filtered_active) == len(active):
            break
        active = filtered_active

    weight_map = _normalize_with_cap(
        {item["symbol"]: float(item["rank_score"]) for item in active},
        total_target_exposure=target_exposure,
        max_single_weight=max_single_weight,
    )

    final_recommendations = []
    category_exposure: dict[str, float] = {}
    style_counter = Counter()
    planned_investment = 0.0

    for rank, item in enumerate(active, start=1):
        weight = weight_map.get(item["symbol"], 0.0)
        entry_price = float(
            item.get("recommended_entry_price")
            or item.get("current_price")
            or 0.0
        )
        lot_size = int(
            item.get("lot_size", getattr(settings, "lot_size", 100))
        )
        shares = (
            int((total_capital * weight) // max(entry_price, 0.01) // lot_size)
            * lot_size
        )
        amount = shares * entry_price
        actual_weight = amount / total_capital if total_capital > 0 else 0.0
        if shares <= 0 or amount <= 0:
            continue

        final_item = dict(item)
        final_item["rank"] = rank
        final_item["portfolio_weight"] = round(actual_weight, 4)
        final_item["portfolio_amount"] = round(amount, 2)
        final_item["portfolio_shares"] = shares
        final_item["cash_buffer"] = round(total_capital * weight - amount, 2)
        final_recommendations.append(ActionConsistencyGuard.apply(final_item))
        planned_investment += amount
        category_exposure[item["category"]] = (
            category_exposure.get(item["category"], 0.0) + actual_weight
        )
        style_counter[item.get("style_bias", "均衡")] += 1

    cash_reserve = max(total_capital - planned_investment, 0.0)
    portfolio_style_bias = (
        style_counter.most_common(1)[0][0] if style_counter else "均衡"
    )
    reliability = _safe_average(
        [float(item.get("confidence", 0.0)) for item in final_recommendations],
        default=0.0,
    )
    execution_notes = [
        (
            f"全市场共扫描 {summary['total_stocks']} 只股票，"
            f"最终入选 {len(final_recommendations)} 只。"
        ),
        (
            f"组合计划投入约 {settings.currency_symbol}"
            f"{planned_investment:,.0f}，保留现金约 "
            f"{settings.currency_symbol}{cash_reserve:,.0f}。"
        ),
        f"单票上限 {max_single_weight:.1%}，优先采用分批建仓与纪律止损。",
    ]

    return {
        "market_summary": summary,
        "portfolio_plan": {
            "total_capital": total_capital,
            "target_exposure": round(
                sum(
                    item["portfolio_weight"] for item in final_recommendations
                ),
                4,
            ),
            "planned_investment": round(planned_investment, 2),
            "cash_reserve": round(cash_reserve, 2),
            "selected_count": len(final_recommendations),
            "style_bias": portfolio_style_bias,
            "max_single_weight": round(max_single_weight, 4),
            "category_exposure": {
                key: round(value, 4)
                for key, value in category_exposure.items()
            },
            "execution_notes": execution_notes,
            "reliability": round(reliability, 4),
        },
        "recommendations": final_recommendations,
    }


class ExecutiveSummaryBuilder:
    """生成面向投资决策的三句话执行摘要。"""

    def __init__(
        self,
        portfolio_plan: dict[str, Any],
        branch_summary: dict[str, dict[str, Any]],
    ) -> None:
        self.portfolio_plan = portfolio_plan
        self.branch_summary = branch_summary

    def _macro_score(self) -> float:
        return float(self.branch_summary.get("macro", {}).get("score", 0.0))

    def _reliability(self) -> float:
        values = [
            float(branch.get("confidence", 0.0))
            for branch in self.branch_summary.values()
            if branch is not None
        ]
        return _safe_average(values, default=0.0)

    def build(self) -> list[str]:
        exposure = float(self.portfolio_plan.get("target_exposure", 0.0))
        style = str(self.portfolio_plan.get("style_bias", "均衡"))
        selected = int(self.portfolio_plan.get("selected_count", 0))
        macro_score = self._macro_score()
        reliability = self._reliability()
        return [
            f"当前建议总仓位维持在 {exposure:.1%}，组合风格偏{style}。",
            f"宏观评分 {macro_score:+.2f}，本轮最终纳入 {selected} 只标的进入执行清单。",
            f"整体可信度为{_confidence_label(reliability)}，当前更适合纪律化分批执行。",
        ]


class ActionConsistencyGuard:
    """统一校验动作、分支支持度和风险文案的一致性。"""

    MIN_CONFIDENCE = 0.42
    MACRO_PRESSURE_THRESHOLD = -0.25

    @classmethod
    def apply(cls, recommendation: dict[str, Any]) -> dict[str, Any]:
        payload = dict(recommendation)
        positive_count = int(payload.get("branch_positive_count", 0))
        confidence = float(payload.get("confidence", 0.0))
        macro_score = float(payload.get("macro_score", 0.0))
        target_exposure = float(
            payload.get(
                "batch_target_exposure", payload.get("portfolio_weight", 0.0)
            )
        )
        weak_support = positive_count <= 2
        low_confidence = confidence < cls.MIN_CONFIDENCE
        macro_pressure = (
            macro_score <= cls.MACRO_PRESSURE_THRESHOLD
            or target_exposure <= 0.20
        )

        if macro_pressure or (weak_support and low_confidence):
            action = "观察"
        elif weak_support or low_confidence:
            action = "轻仓试错"
        else:
            action = "买入"

        reasons = list(payload.get("weight_cap_reasons", []))
        if weak_support:
            reasons.append(
                f"v13 四分支支持仅 {positive_count}/"
                f"{BRANCH_SUPPORT_DENOMINATOR}，不宜激进。"
            )
        if low_confidence:
            reasons.append(f"综合可信度仅 {_confidence_label(confidence)}。")
        if macro_pressure:
            reasons.append("宏观分支当前显著压仓，动作已自动下调。")

        payload["raw_action"] = payload.get("action", "")
        payload["action"] = action
        payload["weight_cap_reasons"] = _dedupe_text(
            [_sanitize_text(item) for item in reasons if item]
        )
        return payload


def _aggregate_branch_summary(
    all_results: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    total_batches = sum(len(batches) for batches in all_results.values())
    for batches in all_results.values():
        for batch in batches:
            for name, branch in _canonical_branch_map(
                dict(batch.get("branches", {}))
            ).items():
                bucket = aggregated.setdefault(
                    name,
                    {
                        "score_values": [],
                        "confidence_values": [],
                        "conclusions": [],
                        "support_drivers": [],
                        "drag_drivers": [],
                        "investment_risks": [],
                        "coverage_notes": [],
                        "diagnostic_notes": [],
                        "module_coverage": {},
                        "debate_statuses": [],
                    },
                )
                bucket["score_values"].append(float(branch.get("score", 0.0)))
                bucket["confidence_values"].append(
                    float(branch.get("confidence", 0.0))
                )
                bucket["conclusions"].append(str(branch.get("conclusion", "")))
                bucket["support_drivers"].extend(
                    str(item) for item in branch.get("support_drivers", [])
                )
                bucket["drag_drivers"].extend(
                    str(item) for item in branch.get("drag_drivers", [])
                )
                bucket["investment_risks"].extend(
                    str(item)
                    for item in branch.get(
                        "investment_risks", branch.get("risks", [])
                    )
                )
                bucket["coverage_notes"].extend(
                    str(item) for item in branch.get("coverage_notes", [])
                )
                bucket["diagnostic_notes"].extend(
                    str(item) for item in branch.get("diagnostic_notes", [])
                )
                bucket["debate_statuses"].append(
                    str(branch.get("debate_status", "skipped"))
                )
                for module_name, info in branch.get(
                    "module_coverage", {}
                ).items():
                    module_bucket = bucket["module_coverage"].setdefault(
                        module_name,
                        {
                            "label": info.get("label", module_name),
                            "available_symbols": 0,
                            "total_symbols": 0,
                            "disabled_batches": 0,
                            "status": info.get("status", "active"),
                        },
                    )
                    module_bucket["available_symbols"] += int(
                        info.get("available_symbols", 0)
                    )
                    module_bucket["total_symbols"] += int(
                        info.get("total_symbols", 0)
                    )
                    if info.get("status") == "disabled_global":
                        module_bucket["disabled_batches"] += 1
                        module_bucket["status"] = "disabled_global"

    finalized: dict[str, dict[str, Any]] = {}
    for name, bucket in aggregated.items():
        module_notes = []
        for module_name, info in bucket["module_coverage"].items():
            label = str(info.get("label", module_name))
            available = int(info.get("available_symbols", 0))
            total = int(info.get("total_symbols", 0))
            if info.get("status") == "disabled_global":
                module_notes.append(
                    f"{label}: 0/{max(total, 1)} 标的可用，"
                    f"{int(info.get('disabled_batches', 0))}/"
                    f"{max(total_batches, 1)} 批次全局剔除。"
                )
            elif total > 0 and available < total:
                module_notes.append(
                    f"{label}: {available}/{total} 标的已覆盖。"
                )
        finalized[name] = {
            "score": _safe_average(bucket["score_values"], default=0.0),
            "confidence": _safe_average(
                bucket["confidence_values"], default=0.0
            ),
            "conclusion": next(
                (text for text in bucket["conclusions"] if str(text).strip()),
                _default_branch_conclusion(
                    name, _safe_average(bucket["score_values"], default=0.0)
                ),
            ),
            "support_drivers": _dedupe_text(
                [_sanitize_text(item) for item in bucket["support_drivers"]]
            )[:3],
            "drag_drivers": _dedupe_text(
                [_sanitize_text(item) for item in bucket["drag_drivers"]]
            )[:3],
            "investment_risks": _dedupe_text(
                [_sanitize_text(item) for item in bucket["investment_risks"]]
            )[:5],
            "coverage_notes": _dedupe_text(
                [_sanitize_text(item) for item in bucket["coverage_notes"]]
                + module_notes
            )[:6],
            "diagnostic_notes": _dedupe_text(
                [_sanitize_text(item) for item in bucket["diagnostic_notes"]]
            )[:6],
            "module_coverage": bucket["module_coverage"],
            "debate_statuses": [
                status
                for status in _dedupe_text(bucket["debate_statuses"])
                if status and status != "unknown"
            ],
        }
    return finalized


class DiagnosticsBucketizer:
    """把投资风险、覆盖信息和工程诊断拆分到不同报告区域。"""

    def __init__(
        self,
        all_results: dict[str, list[dict[str, Any]]],
        branch_summary: dict[str, dict[str, Any]],
    ) -> None:
        self.all_results = all_results
        self.branch_summary = branch_summary

    def bucket(self) -> dict[str, list[str]]:
        total_batches = max(
            sum(len(batches) for batches in self.all_results.values()), 1
        )
        investment_risks: list[str] = []
        coverage_notes: list[str] = []
        diagnostic_notes: list[str] = []
        for branch_name, branch in self.branch_summary.items():
            investment_risks.extend(branch.get("investment_risks", []))
            coverage_notes.extend(branch.get("coverage_notes", []))
            diagnostic_notes.extend(
                f"{_sanitize_text(note)}（{total_batches}/{total_batches} 批次）"
                for note in branch.get("diagnostic_notes", [])
            )
        for batches in self.all_results.values():
            for batch in batches:
                execution_log = batch.get("execution_log", [])
                for line in execution_log[-5:]:
                    sanitized = _sanitize_text(str(line))
                    if sanitized and sanitized not in diagnostic_notes:
                        diagnostic_notes.append(
                            f"{sanitized}（1/{total_batches} 批次）"
                        )
        return {
            "investment_risks": _dedupe_text(investment_risks)[:8],
            "coverage_notes": _dedupe_text(coverage_notes)[:8],
            "diagnostic_notes": _dedupe_text(diagnostic_notes)[:12],
        }


class ConclusionRenderer:
    """渲染分支与个股结论。"""

    @staticmethod
    def _coerce_mapping(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if hasattr(value, "to_dict"):
            payload = value.to_dict()
            if isinstance(payload, dict):
                return dict(payload)
        if isinstance(value, dict):
            return dict(value)
        return {}

    @staticmethod
    def render_branch(branch_name: str, branch: dict[str, Any]) -> list[str]:
        label = _branch_label(branch_name)
        conclusion = str(
            branch.get("conclusion")
            or _default_branch_conclusion(
                branch_name, float(branch.get("score", 0.0))
            )
        )
        support = _dedupe_text(branch.get("support_drivers", [])) or [
            "当前未观察到明显增量支撑。"
        ]
        drag = _dedupe_text(branch.get("drag_drivers", [])) or [
            "当前未观察到明显拖累项。"
        ]
        coverage = _dedupe_text(branch.get("coverage_notes", [])) or [
            "当前未发现显著覆盖缺口。"
        ]
        return [
            f"### {label}分支",
            f"- 平均得分: {branch_name}: {float(branch.get('score', 0.0)):+.3f}",
            f"- 结论: {conclusion}",
            f"- 主要驱动: {'；'.join(support[:2])}",
            f"- 主要拖累: {'；'.join(drag[:2])}",
            f"- 数据覆盖情况: {'；'.join(coverage[:2])}",
            (
                "- 可信度标签: "
                f"{_confidence_label(float(branch.get('confidence', 0.0)))}"
            ),
            "",
        ]

    @staticmethod
    def render_model_role_metadata(model_role_metadata: Any) -> list[str]:
        payload = ConclusionRenderer._coerce_mapping(model_role_metadata)
        if not payload:
            return ["- 当前未记录模型角色元数据。"]
        lines = ["- 模型角色元数据:"]
        for key in [
            "agent_model",
            "agent_fallback_model",
            "master_model",
            "master_fallback_model",
            "master_reasoning_effort",
        ]:
            if key in payload:
                lines.append(f"  - {key}: {payload[key]}")
        if payload.get("resolver_directory_priority"):
            lines.append(
                "  - resolver_directory_priority: "
                f"{payload['resolver_directory_priority']}"
            )
        if payload.get("physical_directories_used_for_full_a"):
            directories = ", ".join(
                str(item)
                for item in payload["physical_directories_used_for_full_a"]
            )
            lines.append(
                f"  - physical_directories_used_for_full_a: {directories}"
            )
        if payload.get("fallback_used") is not None:
            lines.append(f"  - fallback_used: {payload['fallback_used']}")
        return lines

    @staticmethod
    def render_execution_trace(execution_trace: Any) -> list[str]:
        payload = ConclusionRenderer._coerce_mapping(execution_trace)
        if not payload:
            return ["- 当前未记录执行轨迹。"]
        lines = ["- 执行轨迹:"]
        if payload.get("stage_summaries"):
            lines.append("  - stages:")
            for stage in payload.get("stage_summaries", [])[:8]:
                stage_payload = ConclusionRenderer._coerce_mapping(stage)
                if not stage_payload:
                    continue
                stage_name = stage_payload.get("stage_name", "stage")
                status = stage_payload.get("status", "")
                lines.append(f"    - {stage_name}: {status}")
        for key in [
            "resolver_directory_priority",
            "physical_directories_used_for_full_a",
            "local_union_fallback_used",
            "final_deterministic_outcome",
        ]:
            if key in payload:
                value = payload[key]
                if isinstance(value, list):
                    value = ", ".join(str(item) for item in value)
                lines.append(f"  - {key}: {value}")
        return lines

    @staticmethod
    def render_what_if_plan(what_if_plan: Any) -> list[str]:
        payload = ConclusionRenderer._coerce_mapping(what_if_plan)
        scenarios = (
            payload.get("scenarios", [])
            if isinstance(payload.get("scenarios", []), list)
            else []
        )
        if not payload and not scenarios:
            return ["- 当前未记录 what-if 计划。"]
        lines = ["- what-if 计划:"]
        for scenario in scenarios[:6]:
            scenario_payload = ConclusionRenderer._coerce_mapping(scenario)
            if not scenario_payload:
                continue
            scenario_name = scenario_payload.get("scenario_name", "scenario")
            rerun_daily = (
                "是"
                if scenario_payload.get("rerun_full_market_daily_path")
                else "否"
            )
            position_rule = scenario_payload.get(
                "position_adjustment_rule", ""
            )
            lines.extend(
                [
                    f"  - 场景: {scenario_name}",
                    f"    - 触发: {scenario_payload.get('trigger', '')}",
                    f"    - 动作: {scenario_payload.get('action', '')}",
                    f"    - 仓位调整: {position_rule}",
                    f"    - 重新跑全市场: {rerun_daily}",
                ]
            )
        return lines

    @staticmethod
    def render_run_context(
        model_role_metadata: Any,
        execution_trace: Any,
        what_if_plan: Any,
    ) -> list[str]:
        lines: list[str] = []
        lines.extend(
            ConclusionRenderer.render_model_role_metadata(model_role_metadata)
        )
        lines.extend(
            ConclusionRenderer.render_execution_trace(execution_trace)
        )
        lines.extend(ConclusionRenderer.render_what_if_plan(what_if_plan))
        return lines

    @staticmethod
    def render_stock(item: dict[str, Any], market: str) -> list[str]:
        action = str(item.get("action", "观察"))
        stock_name = str(
            item.get("company_name")
            or item.get("name")
            or get_stock_name(item["symbol"], market=market)
        ).strip()
        if _is_unknown_stock_name(stock_name):
            stock_name = get_stock_name(item["symbol"], market=market)
        support = _dedupe_text(
            item.get("support_drivers", [])
            or _derive_stock_support_drivers(item)
        )
        drag = _dedupe_text(
            item.get("drag_drivers", []) or _derive_stock_drag_drivers(item)
        )
        weight_caps = _dedupe_text(item.get("weight_cap_reasons", []))
        if action == "买入":
            conclusion = (
                f"{item['symbol']} {stock_name} 当前获得 "
                f"{int(item.get('branch_positive_count', 0))}/"
                f"{BRANCH_SUPPORT_DENOMINATOR} "
                "个 v13 分支支持，可按计划分批执行。"
            )
        elif action == "轻仓试错":
            conclusion = (
                f"{item['symbol']} {stock_name} 当前仍有正向依据，"
                "但更适合轻仓试错。"
            )
        else:
            conclusion = (
                f"{item['symbol']} {stock_name} 当前信号不足以支撑激进执行，"
                "建议继续观察。"
            )
        one_line = str(item.get("one_line_conclusion", "") or "").strip()
        if action != "买入" and ("买" in one_line or "执行" in one_line):
            one_line = ""
        pressure_text = (
            "；".join((drag + weight_caps)[:3])
            if (drag or weight_caps)
            else "当前未见额外压仓理由。"
        )
        return [
            (
                f"### {item.get('rank', '-')}. {item['symbol']} "
                f"{stock_name} ({item['category_name']})"
            ),
            f"- 一句话结论: {one_line or conclusion}",
            f"- 入选原因: {'；'.join(support[:3])}",
            f"- 压仓原因: {pressure_text}",
            f"- 执行动作: {action}",
            (
                f"- 交易参数: 现价 {item['current_price']:.2f}，"
                f"参考买点 {item['recommended_entry_price']:.2f}，"
                f"目标价 {item['target_price']:.2f}，"
                f"止损价 {item['stop_loss_price']:.2f}"
            ),
            "",
        ]


def save_candidate_index(
    all_results: dict[str, list[dict[str, Any]]],
    market: str = "CN",
    output_dir: str | None = None,
) -> str:
    settings = get_market_settings(market)
    target_dir = Path(output_dir or settings.analysis_output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, list[str]] = {}
    for category, batches in all_results.items():
        items: list[dict[str, Any]] = []
        for batch in batches:
            items.extend(batch.get("recommendations", []))
        ranked_symbols = [
            item["symbol"]
            for item in sorted(
                items,
                key=lambda rec: (
                    float(rec.get("consensus_score", 0.0)),
                    float(rec.get("suggested_weight", 0.0)),
                ),
                reverse=True,
            )
            if item.get("data_source_status") == "real"
        ]
        payload[category] = list(dict.fromkeys(ranked_symbols))

    output_file = target_dir / "all_candidates.json"
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    return str(output_file)


def _build_full_market_report_bundle(
    all_results: dict[str, list[dict[str, Any]]],
    *,
    market: str,
    total_capital: float,
    top_k: int,
) -> tuple[dict[str, Any], Any | None]:
    plan = build_full_market_trade_plan(
        all_results,
        market=market,
        total_capital=total_capital,
        top_k=top_k,
    )
    return plan, None


def generate_full_report(
    all_results: dict[str, list[dict[str, Any]]],
    market: str = "CN",
    output_dir: str | None = None,
    total_capital: float = 1_000_000,
    top_k: int = 12,
) -> dict[str, str]:
    settings = get_market_settings(market)
    target_dir = Path(output_dir or settings.analysis_output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 80}")
    print(f"📊 生成{settings.market_name}全市场综合分析报告")
    print(f"{'=' * 80}")

    load_stock_names(settings.market)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan, report_bundle = _build_full_market_report_bundle(
        all_results,
        market=settings.market,
        total_capital=total_capital,
        top_k=top_k,
    )
    summary = plan["market_summary"]
    portfolio_plan = _to_mapping(
        getattr(report_bundle, "portfolio_plan", None)
    ) or _to_mapping(plan.get("portfolio_plan"))
    target_weights = dict(portfolio_plan.get("target_weights", {}) or {})
    position_limits = dict(portfolio_plan.get("position_limits", {}) or {})
    category_exposure = dict(portfolio_plan.get("category_exposure", {}) or {})
    execution_notes = list(portfolio_plan.get("execution_notes", []) or [])
    if not portfolio_plan:
        portfolio_plan = {}
    portfolio_plan.setdefault("target_weights", target_weights)
    portfolio_plan.setdefault("position_limits", position_limits)
    portfolio_plan.setdefault("category_exposure", category_exposure)
    portfolio_plan.setdefault("execution_notes", execution_notes)
    portfolio_plan.setdefault(
        "target_exposure",
        float(
            portfolio_plan.get(
                "target_exposure",
                sum(target_weights.values()) if target_weights else 0.0,
            )
        ),
    )
    portfolio_plan.setdefault(
        "planned_investment",
        float(portfolio_plan["target_exposure"]) * float(total_capital),
    )
    portfolio_plan.setdefault(
        "cash_reserve",
        float(total_capital)
        * float(
            portfolio_plan.get(
                "cash_ratio", 1.0 - float(portfolio_plan["target_exposure"])
            )
        ),
    )
    portfolio_plan.setdefault(
        "style_bias", portfolio_plan.get("style_bias", "均衡")
    )
    portfolio_plan.setdefault(
        "max_single_weight",
        max(
            position_limits.values(),
            default=max(target_weights.values(), default=0.0),
        ),
    )
    portfolio_plan.setdefault("selected_count", len(target_weights))
    raw_recommendations = list(plan.get("recommendations", []) or [])
    recommendations = [
        ActionConsistencyGuard.apply(item) for item in raw_recommendations
    ]
    plan["recommendations"] = recommendations
    plan["portfolio_plan"] = portfolio_plan
    branch_summary = _aggregate_branch_summary(all_results)
    diagnostics = DiagnosticsBucketizer(all_results, branch_summary).bucket()
    executive_summary = ExecutiveSummaryBuilder(
        portfolio_plan, branch_summary
    ).build()
    analysis_meta = _build_analysis_meta(all_results)

    report_lines = [
        (
            f"# {settings.report_flag} "
            f"{settings.market_name}全市场组合级交易建议报告\n"
        ),
        f"**生成时间**: {summary['generated_at']}\n",
        "**分析架构**: Quant-Investor V13 四分支研究契约\n",
        (
            f"**分析覆盖**: {summary['total_stocks']} 只股票，"
            f"{summary['total_batches']} 个批次\n"
        ),
        f"**分析 universe**: {analysis_meta.get('universe', 'full_a')}\n",
        "\n## 三句话执行摘要\n",
    ]
    data_snapshot_summary = str(
        (analysis_meta.get("data_snapshot", {}) or {}).get("summary_text", "")
    ).strip()
    if data_snapshot_summary:
        report_lines.insert(5, f"**本地数据快照**: {data_snapshot_summary}\n")
    for line in executive_summary:
        report_lines.append(f"- {line}\n")

    cash_reserve_text = (
        f"{settings.currency_symbol}{portfolio_plan['cash_reserve']:,.0f}"
    )
    reliability_label = _confidence_label(
        float(portfolio_plan.get("reliability", 0.0))
    )
    report_lines.extend(
        [
            "\n## 为什么当前总仓位是这个水平\n",
            (
                f"- 当前计划总仓位为 {portfolio_plan['target_exposure']:.1%}，"
                f"计划投入 {settings.currency_symbol}"
                f"{portfolio_plan['planned_investment']:,.0f}，预留现金 "
                f"{cash_reserve_text}。\n"
            ),
            (
                f"- 组合风格偏 {portfolio_plan['style_bias']}，"
                f"单票上限控制在 {portfolio_plan['max_single_weight']:.1%}，"
                f"本轮最终纳入 {portfolio_plan['selected_count']} 只标的。\n"
            ),
            "- 类别暴露为 "
            + (
                " / ".join(
                    f"{category_name(category, settings.market)} {weight:.1%}"
                    for category, weight in portfolio_plan[
                        "category_exposure"
                    ].items()
                )
                if portfolio_plan["category_exposure"]
                else "暂无"
            )
            + "。\n",
        ]
    )

    report_lines.extend(
        [
            "\n## 数据覆盖与可信度摘要\n",
            (
                f"- 本次汇总 {summary['total_batches']}/"
                f"{summary['total_batches']} 批次，覆盖 "
                f"{summary['total_stocks']}/{summary['total_stocks']} 标的。\n"
            ),
            (
                f"- 组合层平均可信度为 {reliability_label}。\n"
            ),
        ]
    )
    for note in diagnostics["coverage_notes"][:5]:
        report_lines.append(f"- {_sanitize_text(note)}\n")
    if diagnostics["investment_risks"]:
        report_lines.append(
            f"- 需要前置注意的投资风险: {'；'.join(diagnostics['investment_risks'][:3])}\n"
        )
    if analysis_meta.get("model_role_metadata"):
        report_lines.extend(["\n## 模型角色与执行轨迹\n"])
        render_run_context = getattr(
            ConclusionRenderer, "render_run_context", None
        )
        if callable(render_run_context):
            report_lines.extend(
                render_run_context(
                    analysis_meta.get("model_role_metadata"),
                    analysis_meta.get("execution_trace"),
                    analysis_meta.get("what_if_plan"),
                )
            )
        else:
            report_lines.extend(
                ConclusionRenderer.render_model_role_metadata(
                    analysis_meta.get("model_role_metadata")
                )
            )
            report_lines.extend(
                ConclusionRenderer.render_execution_trace(
                    analysis_meta.get("execution_trace")
                )
            )
            report_lines.extend(
                ConclusionRenderer.render_what_if_plan(
                    analysis_meta.get("what_if_plan")
                )
            )

    if recommendations:
        report_lines.extend(
            [
                "\n## 最终推荐标的\n",
                (
                    "| 排名 | 代码 | 名称 | 类别 | 现价 | 推荐买入价 | "
                    "目标卖出价 | 止损价 | 推荐仓位 | 金额 | "
                    "预期空间 | v13分支支持 |\n"
                ),
                (
                    "|:---:|:---|:---|:---|---:|---:|---:|---:|"
                    "---:|---:|---:|---:|\n"
                ),
            ]
        )
        for item in recommendations:
            stock_name = str(
                item.get("company_name")
                or item.get("name")
                or get_stock_name(item["symbol"], market=settings.market)
            ).strip()
            if _is_unknown_stock_name(stock_name):
                stock_name = get_stock_name(
                    item["symbol"], market=settings.market
                )
            current_price = float(item.get("current_price", 0))
            entry_low = float(
                item.get("entry_price_range", {}).get(
                    "low", current_price * 0.99
                )
            )
            entry_high = float(
                item.get("entry_price_range", {}).get(
                    "high", current_price * 1.01
                )
            )
            display_entry_price = float(
                item.get("recommended_entry_price")
                or (entry_low + entry_high) / 2
                or current_price
            )
            report_lines.append(
                f"| {item['rank']} | {item['symbol']} | {stock_name} | "
                f"{item['category_name']} | {settings.currency_symbol}"
                f"{current_price:.2f} | {settings.currency_symbol}"
                f"{display_entry_price:.2f} | {settings.currency_symbol}"
                f"{item['target_price']:.2f} | {settings.currency_symbol}"
                f"{item['stop_loss_price']:.2f} | "
                f"{item['portfolio_weight']:.1%} | {settings.currency_symbol}"
                f"{item['portfolio_amount']:,.0f} | "
                f"{float(item['expected_upside']):.1%} | "
                f"{item['branch_positive_count']}/"
                f"{BRANCH_SUPPORT_DENOMINATOR} |\n"
            )

        for item in recommendations[:12]:
            report_lines.extend(
                ConclusionRenderer.render_stock(item, settings.market)
            )
            display_entry_price = float(
                item.get("recommended_entry_price")
                or item.get("current_price")
                or 0.01
            )
            max_loss = (
                float(item.get("stop_loss_price", 0.0))
                / max(display_entry_price, 0.01)
                - 1
            ) * 100
            report_lines.append(f"- 最大亏损: {max_loss:.1f}%\n")
            risk_observation = (
                "；".join(
                    _dedupe_text(
                        [
                            _sanitize_text(flag)
                            for flag in item.get("risk_flags", [])
                        ]
                    )[:3]
                )
                or "无"
            )
            report_lines.append(
                f"- 风险观察: {risk_observation}\n"
            )
            report_lines.append("")
    else:
        report_lines.append("\n## 最终推荐标的\n")
        report_lines.append(
            "- 当前没有满足条件的最终候选，建议继续以现金和观察仓位为主。\n"
        )

    report_lines.append("\n## v13 四分支结论\n")
    for branch_name in CANONICAL_BRANCH_ORDER:
        if branch_name in branch_summary:
            report_lines.extend(
                line + "\n"
                for line in ConclusionRenderer.render_branch(
                    branch_name, branch_summary[branch_name]
                )
            )

    report_lines.append("## 附录：工程诊断与详细运行日志\n")
    if diagnostics["diagnostic_notes"]:
        for note in diagnostics["diagnostic_notes"]:
            report_lines.append(f"- {note}\n")
    else:
        report_lines.append(
            f"- 当前未记录新增工程诊断（{summary['total_batches']}/"
            f"{summary['total_batches']} 批次）。\n"
        )

    for category, batches in all_results.items():
        for batch in batches:
            if not batch.get("execution_log"):
                continue
            log_count = len(batch.get("execution_log", []))
            report_lines.append(
                f"- {category_name(category, settings.market)} 批次 "
                f"{batch.get('batch_id', '-')}: "
                f"附带 {log_count}/{log_count} 条运行日志。\n"
            )

    summary_lines = [
        f"# {settings.report_flag} {settings.market_name}全市场分析摘要\n",
        f"**生成时间**: {summary['generated_at']}\n",
        (
            f"**分析覆盖**: {summary['total_stocks']} 只股票，"
            f"{summary['total_batches']} 个批次\n"
        ),
        f"**分析 universe**: {analysis_meta.get('universe', 'full_a')}\n",
        "\n## 三句话执行摘要\n",
    ]
    if data_snapshot_summary:
        summary_lines.insert(4, f"**本地数据快照**: {data_snapshot_summary}\n")
    for line in executive_summary:
        summary_lines.append(f"- {line}\n")
    summary_lines.append("\n## 执行提醒\n")
    for note in portfolio_plan["execution_notes"]:
        summary_lines.append(f"- {_sanitize_text(note)}\n")

    summary_file = target_dir / f"{settings.market}_Full_Report_{timestamp}.md"
    data_file = target_dir / f"{settings.market}_Trade_Data_{timestamp}.json"
    report_file = target_dir / f"{settings.market}_Trade_Report_{timestamp}.md"
    summary_text = _append_theme_sections_once(
        "".join(summary_lines),
        theme_rotation=analysis_meta.get("theme_rotation"),
        theme_governance=analysis_meta.get("theme_governance"),
        theme_production_overlay=analysis_meta.get("theme_production_overlay"),
        theme_shadow_monitor=analysis_meta.get("theme_shadow_monitor"),
        governance_max_rows=8,
    )
    report_text = _append_theme_sections_once(
        "".join(report_lines),
        theme_rotation=analysis_meta.get("theme_rotation"),
        theme_governance=analysis_meta.get("theme_governance"),
        theme_production_overlay=analysis_meta.get("theme_production_overlay"),
        theme_shadow_monitor=analysis_meta.get("theme_shadow_monitor"),
        governance_max_rows=20,
    )
    with open(summary_file, "w", encoding="utf-8") as file:
        file.write(summary_text)
    with open(report_file, "w", encoding="utf-8") as file:
        file.write(report_text)
    with open(data_file, "w", encoding="utf-8") as file:
        json.dump(plan, file, indent=2, ensure_ascii=False)

    candidate_file = save_candidate_index(
        all_results, market=settings.market, output_dir=str(target_dir)
    )
    return {
        "summary_report": str(summary_file),
        "trade_report": str(report_file),
        "trade_data": str(data_file),
        "candidate_index": candidate_file,
    }


__all__ = [
    "ActionConsistencyGuard",
    "BRANCH_LABELS",
    "BRANCH_SUPPORT_DENOMINATOR",
    "ConclusionRenderer",
    "DiagnosticsBucketizer",
    "ExecutiveSummaryBuilder",
    "build_full_market_trade_plan",
    "category_name",
    "generate_full_report",
    "save_candidate_index",
]
