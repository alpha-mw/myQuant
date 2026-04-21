from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Any, Mapping

from quant_investor.agent_protocol import ActionLabel, BranchVerdict, ICDecision, RiskDecision
from quant_investor.agents.agent_contracts import BaseBranchAgentOutput
from quant_investor.branch_contracts import BranchResult
from quant_investor.market.dag.common import _dedupe_texts, _score_to_action


def _branch_conviction_from_action(action: ActionLabel) -> str:
    if action == ActionLabel.BUY:
        return "buy"
    if action == ActionLabel.SELL:
        return "sell"
    return "neutral"


def _aggregate_branch_summaries(
    research_by_symbol: Mapping[str, Mapping[str, BranchVerdict]],
) -> dict[str, BranchVerdict]:
    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "scores": [],
            "confidences": [],
            "theses": [],
            "risks": [],
            "coverage": [],
            "diagnostic": [],
            "symbols": [],
        }
    )
    for symbol, branch_map in research_by_symbol.items():
        for branch_name, verdict in branch_map.items():
            bucket = buckets[branch_name]
            bucket["scores"].append(float(verdict.final_score))
            bucket["confidences"].append(float(verdict.final_confidence))
            bucket["theses"].append(str(verdict.thesis))
            bucket["risks"].extend(verdict.investment_risks)
            bucket["coverage"].extend(verdict.coverage_notes)
            bucket["diagnostic"].extend(verdict.diagnostic_notes)
            bucket["symbols"].append(symbol)
    result: dict[str, BranchVerdict] = {}
    for branch_name, bucket in buckets.items():
        avg_score = fmean(bucket["scores"]) if bucket["scores"] else 0.0
        avg_conf = fmean(bucket["confidences"]) if bucket["confidences"] else 0.0
        result[branch_name] = BranchVerdict(
            agent_name=branch_name,
            thesis=next((text for text in bucket["theses"] if text.strip()), f"{branch_name} 分支已完成全市场汇总。"),
            symbol=None,
            final_score=float(avg_score),
            final_confidence=float(avg_conf),
            investment_risks=_dedupe_texts(list(bucket["risks"]))[:8],
            coverage_notes=_dedupe_texts(list(bucket["coverage"]))[:8],
            diagnostic_notes=_dedupe_texts(list(bucket["diagnostic"]))[:8],
            metadata={
                "branch_name": branch_name,
                "symbol_count": len(bucket["symbols"]),
                "symbols": list(dict.fromkeys(bucket["symbols"]))[:15],
            },
        )
    return result


def _build_branch_results(
    research_by_symbol: Mapping[str, Mapping[str, BranchVerdict]],
    branch_summaries: Mapping[str, BranchVerdict],
) -> dict[str, BranchResult]:
    branch_scores_by_symbol: dict[str, dict[str, float]] = defaultdict(dict)
    for symbol, branch_map in research_by_symbol.items():
        for branch_name, verdict in branch_map.items():
            branch_scores_by_symbol[branch_name][symbol] = float(verdict.final_score)

    results: dict[str, BranchResult] = {}
    for branch_name, verdict in branch_summaries.items():
        results[branch_name] = BranchResult(
            branch_name=branch_name,
            final_score=float(verdict.final_score),
            final_confidence=float(verdict.final_confidence),
            symbol_scores=dict(branch_scores_by_symbol.get(branch_name, {})),
            conclusion=str(verdict.thesis),
            investment_risks=list(verdict.investment_risks),
            coverage_notes=list(verdict.coverage_notes),
            diagnostic_notes=list(verdict.diagnostic_notes),
            metadata=dict(verdict.metadata),
        )
    return results


def _branch_verdicts_to_master_reports(
    branch_summaries: Mapping[str, BranchVerdict],
    research_by_symbol: Mapping[str, Mapping[str, BranchVerdict]],
) -> dict[str, BaseBranchAgentOutput]:
    reports: dict[str, BaseBranchAgentOutput] = {}
    for branch_name, summary in branch_summaries.items():
        symbol_views: dict[str, str] = {}
        for symbol, branches in research_by_symbol.items():
            verdict = branches.get(branch_name)
            if verdict is None:
                continue
            symbol_views[symbol] = verdict.thesis
        action = _score_to_action(float(summary.final_score))
        reports[branch_name] = BaseBranchAgentOutput(
            branch_name=branch_name,
            conviction=_branch_conviction_from_action(action),
            conviction_score=float(summary.final_score),
            confidence=float(summary.final_confidence),
            key_insights=list(summary.coverage_notes[:4]) or [summary.thesis],
            risk_flags=list(summary.investment_risks[:4]),
            disagreements_with_algo=list(summary.diagnostic_notes[:4]),
            symbol_views=symbol_views,
            reasoning=str(summary.thesis),
        )
    return reports


def _attach_symbol_to_ic_decision(
    ic_decision: ICDecision,
    *,
    symbol: str,
    risk_decision: RiskDecision,
    current_weight: float,
    tradability_info: Mapping[str, Any],
    ic_hint: Mapping[str, Any] | None = None,
    shortlist_item: Any | None = None,
) -> ICDecision:
    payload = ic_decision
    payload.selected_symbols = [symbol] if symbol not in risk_decision.blocked_symbols and payload.action in {ActionLabel.BUY, ActionLabel.HOLD} else []
    payload.rejected_symbols = [] if payload.selected_symbols else [symbol]
    payload.metadata = dict(payload.metadata)
    shortlist_meta = dict(getattr(shortlist_item, "metadata", {}) or {}) if shortlist_item is not None else {}
    momentum_strength = float(shortlist_meta.get("momentum_strength", tradability_info.get("momentum_strength", 0.0)) or 0.0)
    calibrated_confidence = float(shortlist_meta.get("history_confidence", getattr(shortlist_item, "confidence", payload.final_confidence)) or payload.final_confidence)
    fake_breakout_penalty = float(shortlist_meta.get("fake_breakout_penalty", tradability_info.get("fake_breakout_risk", 0.0)) or 0.0)
    payload.metadata.update(
        {
            "symbol": symbol,
            "company_name": str(tradability_info.get("company_name", "")),
            "risk_action_cap": risk_decision.action_cap.value,
            "current_weight": current_weight,
            "sector": str(tradability_info.get("sector", "unknown")),
            "industry": str(tradability_info.get("industry", tradability_info.get("sector", "unknown"))),
            "symbol_momentum_strengths": {symbol: momentum_strength},
            "symbol_calibrated_confidences": {symbol: calibrated_confidence},
            "symbol_fake_breakout_penalties": {symbol: fake_breakout_penalty},
            "symbol_candidates": [
                {
                    "symbol": symbol,
                    "company_name": str(tradability_info.get("company_name", "")),
                    "score": float(payload.final_score),
                    "confidence": float(payload.final_confidence),
                    "action": payload.action.value,
                    "current_weight": current_weight,
                    "momentum_strength": momentum_strength,
                    "calibrated_confidence": calibrated_confidence,
                    "fake_breakout_penalty": fake_breakout_penalty,
                    "one_line_conclusion": payload.thesis,
                }
            ],
        }
    )
    if ic_hint:
        payload.metadata["llm_master_hint"] = dict(ic_hint)
    return payload
