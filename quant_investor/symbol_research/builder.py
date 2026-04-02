from __future__ import annotations

from typing import Any

from quant_investor.branch_contracts import BranchResult, UnifiedDataBundle
from quant_investor.contracts import GlobalContext, SymbolResearchPacket


def build_symbol_research_packets(
    *,
    data_bundle: UnifiedDataBundle,
    branch_results: dict[str, BranchResult],
    calibrated_signals: dict[str, Any],
    global_context: GlobalContext,
) -> dict[str, SymbolResearchPacket]:
    packets: dict[str, SymbolResearchPacket] = {}
    for symbol in data_bundle.symbols:
        packets[symbol] = _build_symbol_research_packet(
            symbol=symbol,
            data_bundle=data_bundle,
            branch_results=branch_results,
            calibrated_signals=calibrated_signals,
            global_context=global_context,
        )
    return packets


def _aggregate_branch_metric(
    calibrated_signals: dict[str, Any],
    symbol: str,
    metric_name: str,
) -> float:
    values: list[float] = []
    for signal in calibrated_signals.values():
        if signal is None:
            continue
        metric = getattr(signal, metric_name, {}) or {}
        if not isinstance(metric, dict):
            continue
        if symbol not in metric:
            continue
        try:
            values.append(float(metric.get(symbol, 0.0)))
        except (TypeError, ValueError):
            continue
    if not values:
        return 0.0
    return sum(values) / len(values)


def _build_symbol_research_packet(
    *,
    symbol: str,
    data_bundle: UnifiedDataBundle,
    branch_results: dict[str, BranchResult],
    calibrated_signals: dict[str, Any],
    global_context: GlobalContext,
) -> SymbolResearchPacket:
    kline = branch_results.get("kline")
    fundamental = branch_results.get("fundamental")
    intelligence = branch_results.get("intelligence")
    quant_signal = calibrated_signals.get("quant")

    def _branch_score(branch: BranchResult | None) -> float:
        if branch is None:
            return 0.0
        return float(branch.symbol_scores.get(symbol, branch.score))

    kline_view = {
        "score": _branch_score(kline),
        "predicted_return": float((kline.signals.get("predicted_return", {}) or {}).get(symbol, 0.0)) if kline else 0.0,
        "trend_regime": str((kline.signals.get("trend_regime", {}) or {}).get(symbol, "")) if kline else "",
        "branch_mode": kline.metadata.get("branch_mode", "") if kline else "",
    }
    fundamental_view = {
        "score": _branch_score(fundamental),
        "bull_case": list((fundamental.signals.get("bull_case", {}) or {}).get(symbol, [])) if fundamental else [],
        "bear_case": list((fundamental.signals.get("bear_case", {}) or {}).get(symbol, [])) if fundamental else [],
        "quality_breakdown": dict((fundamental.signals.get("quality_breakdown", {}) or {}).get(symbol, {})) if fundamental else {},
        "branch_mode": fundamental.metadata.get("branch_mode", "") if fundamental else "",
    }
    intelligence_view = {
        "score": _branch_score(intelligence),
        "event_risk": float((intelligence.signals.get("event_risk_score", {}) or {}).get(symbol, 0.0)) if intelligence else 0.0,
        "sentiment": float((intelligence.signals.get("sentiment_score", {}) or {}).get(symbol, 0.0)) if intelligence else 0.0,
        "money_flow": float((intelligence.signals.get("money_flow_score", {}) or {}).get(symbol, 0.0)) if intelligence else 0.0,
        "branch_mode": intelligence.metadata.get("branch_mode", "") if intelligence else "",
    }

    merged_score = _aggregate_branch_metric(calibrated_signals, symbol, "symbol_convictions")
    confidence = _aggregate_branch_metric(calibrated_signals, symbol, "symbol_confidences")
    risk_flags = _collect_risk_flags(symbol, data_bundle, branch_results)

    evidence = {
        "support": _collect_supporting_lines(symbol, branch_results),
        "drag": _collect_drag_lines(symbol, branch_results),
        "risks": list(risk_flags),
    }

    metadata = {
        "packet_ref": f"{global_context.cache_key}:{symbol}",
        "branch_scores": {
            name: _branch_score(branch)
            for name, branch in branch_results.items()
        },
        "quant_factor_score": float((global_context.quant_factor_scores or {}).get(symbol, 0.0)),
        "liquidity_score": float((global_context.liquidity_snapshot or {}).get(symbol, {}).get("liquidity_score", 0.0)),
    }

    return SymbolResearchPacket(
        symbol=symbol,
        market=data_bundle.market,
        as_of_date=str(data_bundle.metadata.get("end_date", global_context.as_of_date)),
        global_context_ref=global_context.cache_key,
        kline_view=kline_view,
        fundamental_view=fundamental_view,
        intelligence_view=intelligence_view,
        merged_score=merged_score,
        confidence=confidence,
        risk_flags=list(risk_flags),
        evidence=evidence,
        metadata=metadata,
    )


def _collect_risk_flags(
    symbol: str,
    data_bundle: UnifiedDataBundle,
    branch_results: dict[str, BranchResult],
) -> list[str]:
    flags: list[str] = []
    symbol_meta = data_bundle.symbol_provenance().get(symbol, {})
    if symbol_meta.get("data_source_status") != "real":
        flags.append("非纯真实数据源")
    for branch in branch_results.values():
        for risk in branch.risks:
            text = str(risk)
            if symbol in text or branch.branch_name in {"macro", "quant"}:
                flags.append(text)
    deduped: list[str] = []
    for item in flags:
        if item and item not in deduped:
            deduped.append(item)
    return deduped[:6]


def _collect_supporting_lines(symbol: str, branch_results: dict[str, BranchResult]) -> list[str]:
    lines: list[str] = []
    kline = branch_results.get("kline")
    if kline is not None and kline.score > 0:
        lines.append(f"K线评分 {kline.symbol_scores.get(symbol, kline.score):+.2f}")
    fundamental = branch_results.get("fundamental")
    if fundamental is not None and fundamental.score > 0:
        lines.append(f"基本面评分 {fundamental.symbol_scores.get(symbol, fundamental.score):+.2f}")
    intelligence = branch_results.get("intelligence")
    if intelligence is not None and intelligence.score > 0:
        lines.append(f"情报评分 {intelligence.symbol_scores.get(symbol, intelligence.score):+.2f}")
    return lines[:4]


def _collect_drag_lines(symbol: str, branch_results: dict[str, BranchResult]) -> list[str]:
    lines: list[str] = []
    for branch_name in ("kline", "fundamental", "intelligence"):
        branch = branch_results.get(branch_name)
        if branch is None:
            continue
        if branch.symbol_scores.get(symbol, branch.score) < 0:
            lines.append(f"{branch_name} 偏弱 {branch.symbol_scores.get(symbol, branch.score):+.2f}")
    return lines[:4]
