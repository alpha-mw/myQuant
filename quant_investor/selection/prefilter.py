from __future__ import annotations

from typing import Any

from quant_investor.branch_contracts import UnifiedDataBundle
from quant_investor.contracts import GlobalContext, ShortlistItem, SymbolResearchPacket


def build_shortlist_items(
    *,
    data_bundle: UnifiedDataBundle,
    symbol_packets: dict[str, SymbolResearchPacket],
    global_context: GlobalContext,
    max_items: int = 12,
) -> list[ShortlistItem]:
    shortlist: list[ShortlistItem] = []
    for symbol, packet in symbol_packets.items():
        liquidity_score = float((global_context.liquidity_snapshot or {}).get(symbol, {}).get("liquidity_score", 0.0))
        downside_risk = min(
            1.0,
            max(0.0, -float(packet.merged_score)) * 0.6 + min(len(packet.risk_flags), 6) * 0.08,
        )
        rank_score = (
            0.55 * float(packet.merged_score)
            + 0.20 * float(packet.confidence)
            + 0.15 * liquidity_score
            - 0.20 * downside_risk
        )
        shortlist.append(
            ShortlistItem(
                symbol=symbol,
                rank_score=round(rank_score, 4),
                expected_return=round(float(packet.kline_view.get("predicted_return", packet.merged_score * 0.1)), 4),
                downside_risk=round(downside_risk, 4),
                liquidity_score=round(liquidity_score, 4),
                conviction=_rank_to_conviction(rank_score),
                packet_ref=packet.metadata.get("packet_ref", f"{global_context.cache_key}:{symbol}"),
                sector=str((data_bundle.fundamentals.get(symbol, {}) or {}).get("sector", "")),
                rationale=_build_rationale(packet),
                metadata={
                    "global_context_ref": global_context.cache_key,
                    "liquidity_snapshot": dict((global_context.liquidity_snapshot or {}).get(symbol, {})),
                },
            )
        )
    shortlist.sort(key=lambda item: item.rank_score, reverse=True)
    return shortlist[:max_items]


def _rank_to_conviction(rank_score: float) -> str:
    if rank_score >= 0.5:
        return "strong_buy"
    if rank_score >= 0.15:
        return "buy"
    if rank_score <= -0.5:
        return "strong_sell"
    if rank_score <= -0.15:
        return "sell"
    return "neutral"


def _build_rationale(packet: SymbolResearchPacket) -> str:
    lines = []
    if packet.kline_view.get("predicted_return") is not None:
        lines.append(f"K线预期收益 {float(packet.kline_view.get('predicted_return', 0.0)):+.2%}")
    if packet.fundamental_view.get("score") is not None:
        lines.append(f"基本面分数 {float(packet.fundamental_view.get('score', 0.0)):+.2f}")
    if packet.intelligence_view.get("score") is not None:
        lines.append(f"情报分数 {float(packet.intelligence_view.get('score', 0.0)):+.2f}")
    if packet.risk_flags:
        lines.append("风险: " + "；".join(packet.risk_flags[:2]))
    return " / ".join(lines) if lines else "暂无明显增量理由"
