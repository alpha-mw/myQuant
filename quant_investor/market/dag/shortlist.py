from __future__ import annotations

from statistics import fmean
from typing import Any, Mapping

from quant_investor.agent_protocol import BranchVerdict, ShortlistItem
from quant_investor.market.dag.common import _dedupe_texts, _score_to_action


def _build_shortlist(
    research_by_symbol: Mapping[str, Mapping[str, BranchVerdict]],
    ic_hints_by_symbol: Mapping[str, Mapping[str, Any]],
    macro_verdict: BranchVerdict,
    company_name_map: Mapping[str, str],
    top_k: int,
    allowed_symbols: set[str] | None = None,
    blocked_symbols: set[str] | None = None,
) -> list[ShortlistItem]:
    shortlist: list[ShortlistItem] = []
    for symbol, branch_map in research_by_symbol.items():
        if allowed_symbols is not None and symbol not in allowed_symbols:
            continue
        if blocked_symbols is not None and symbol in blocked_symbols:
            continue
        scores = [float(verdict.final_score) for verdict in branch_map.values()]
        confidences = [float(verdict.final_confidence) for verdict in branch_map.values()]
        avg_score = fmean(scores) if scores else 0.0
        avg_conf = fmean(confidences) if confidences else 0.0
        hint = dict(ic_hints_by_symbol.get(symbol, {}))
        hint_score = float(hint.get("score", avg_score))
        hint_conf = float(hint.get("confidence", avg_conf))
        rank_score = 0.55 * avg_score + 0.25 * hint_score + 0.20 * float(macro_verdict.final_score)
        action = _score_to_action(rank_score)
        rationale = _dedupe_texts(
            [verdict.thesis for verdict in branch_map.values()] + [str(hint.get("thesis", ""))]
        )
        shortlist.append(
            ShortlistItem(
                symbol=symbol,
                company_name=company_name_map.get(symbol, ""),
                category=str(next(iter(branch_map.values())).metadata.get("category", "")) if branch_map else "",
                rank_score=float(rank_score),
                action=action,
                confidence=float(max(avg_conf, hint_conf)),
                expected_upside=max(float(rank_score), 0.0),
                suggested_weight=max(float(hint.get("confidence", avg_conf)), 0.0) * 0.1,
                risk_flags=_dedupe_texts([item for verdict in branch_map.values() for item in verdict.investment_risks]),
                rationale=rationale[:5],
                metadata={
                    "branch_scores": {name: float(verdict.final_score) for name, verdict in branch_map.items()},
                    "macro_score": float(macro_verdict.final_score),
                    "ic_hint": hint,
                },
            )
        )
    shortlist.sort(key=lambda item: (-float(item.rank_score), item.symbol))
    return shortlist[:top_k]


def _build_shortlist_from_bayesian_records(
    *,
    posterior_results: list[Any],
    company_name_map: Mapping[str, str],
    top_k: int,
) -> list[ShortlistItem]:
    ranked = sorted(
        posterior_results,
        key=lambda item: (
            -float(getattr(item, "posterior_action_score", 0.0) if not isinstance(item, dict) else item.get("posterior_action_score", 0.0)),
            str(getattr(item, "symbol", "") if not isinstance(item, dict) else item.get("symbol", "")),
        ),
    )
    shortlist: list[ShortlistItem] = []
    for record in ranked:
        symbol = str(getattr(record, "symbol", "") if not isinstance(record, dict) else record.get("symbol", ""))
        company_name = (
            str(getattr(record, "company_name", "") if not isinstance(record, dict) else record.get("company_name", ""))
            or company_name_map.get(symbol, "")
        )
        action_score = float(
            getattr(record, "posterior_action_score", 0.0)
            if not isinstance(record, dict)
            else record.get("posterior_action_score", 0.0)
        )
        win_rate = float(
            getattr(record, "posterior_win_rate", 0.0)
            if not isinstance(record, dict)
            else record.get("posterior_win_rate", 0.0)
        )
        confidence = float(
            getattr(record, "posterior_confidence", 0.0)
            if not isinstance(record, dict)
            else record.get("posterior_confidence", 0.0)
        )
        expected_alpha = float(
            getattr(record, "posterior_expected_alpha", 0.0)
            if not isinstance(record, dict)
            else record.get("posterior_expected_alpha", 0.0)
        )
        metadata = (
            dict(getattr(record, "metadata", {}) or {})
            if not isinstance(record, dict)
            else dict(record.get("metadata", {}) or {})
        )
        if bool(metadata.get("kill_switch", False)):
            continue
        risk_flags: list[str] = []
        fake_breakout_penalty = float(metadata.get("fake_breakout_penalty", 0.0) or 0.0)
        setup_failure_penalty = float(metadata.get("setup_failure_penalty", 0.0) or 0.0)
        crowding_penalty = float(metadata.get("crowding_penalty", 0.0) or 0.0)
        if fake_breakout_penalty >= 0.35:
            risk_flags.append(f"fake_breakout_risk={fake_breakout_penalty:.2f}")
        if setup_failure_penalty >= 0.25:
            risk_flags.append(f"setup_failure_penalty={setup_failure_penalty:.2f}")
        if crowding_penalty >= 0.20:
            risk_flags.append(f"crowding_penalty={crowding_penalty:.2f}")
        shortlist.append(
            ShortlistItem(
                symbol=symbol,
                company_name=company_name,
                category=str(metadata.get("category", "")),
                rank_score=action_score,
                action=_score_to_action(action_score),
                confidence=confidence,
                expected_upside=max(expected_alpha, 0.0),
                suggested_weight=max(
                    0.0,
                    min(
                        0.2,
                        float(metadata.get("momentum_strength", action_score) or action_score)
                        * confidence
                        * (1.0 - min(fake_breakout_penalty, 0.85) * 0.45)
                        * 0.2,
                    ),
                ),
                risk_flags=risk_flags,
                rationale=[
                    f"posterior_action_score={action_score:.3f}",
                    f"posterior_win_rate={win_rate:.3f}",
                    f"momentum_strength={float(metadata.get('momentum_strength', 0.0) or 0.0):.3f}",
                ],
                metadata={
                    **metadata,
                    "posterior_action_score": action_score,
                    "posterior_win_rate": win_rate,
                    "posterior_confidence": confidence,
                    "posterior_expected_alpha": expected_alpha,
                    "posterior_edge_after_costs": float(metadata.get("posterior_edge_after_costs", 0.0)),
                    "posterior_capacity_penalty": float(metadata.get("posterior_capacity_penalty", 0.0)),
                    "rank": int(getattr(record, "rank", 0) if not isinstance(record, dict) else record.get("rank", 0)),
                },
            )
        )
        if len(shortlist) >= top_k:
            break
    return shortlist
