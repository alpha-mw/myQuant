from __future__ import annotations

from typing import Any, Mapping

from quant_investor.agent_protocol import BranchVerdict, GlobalContext, ShortlistItem
from quant_investor.agents.agent_contracts import MasterAgentInput
from quant_investor.agents.master_agent import MasterAgent
from quant_investor.market.dag.common import _run_async_coroutine_safely


def _portfolio_master_advisory(
    *,
    master_agent: MasterAgent,
    macro_verdict: BranchVerdict,
    shortlist: list[ShortlistItem],
    global_context: GlobalContext,
    evidence_pack: dict[str, Any],
    recall_context: Mapping[str, Any] | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    advice_meta: dict[str, Any] = {}
    try:
        agent_input = MasterAgentInput(
            evidence_pack=dict(evidence_pack),
            branch_reports={},
            risk_report=None,
            ensemble_baseline=dict(
                evidence_pack.get(
                    "portfolio_constraints",
                    {
                        "aggregate_score": float(sum(item.rank_score for item in shortlist) / len(shortlist)) if shortlist else 0.0,
                        "selected_count": len(shortlist),
                    },
                )
            ),
            market_regime=str(global_context.macro_regime or macro_verdict.metadata.get("regime", "neutral")),
            candidate_symbols=[item.symbol for item in shortlist],
            recall_context=dict(recall_context or {}),
        )
        output = _run_async_coroutine_safely(lambda: master_agent.deliberate(agent_input))
        advice_meta = {
            "status": "success",
            "reason": "",
            "final_conviction": output.final_conviction,
            "final_score": float(output.final_score),
            "confidence": float(output.confidence),
            "top_picks": [item.model_dump() if hasattr(item, "model_dump") else item.__dict__ for item in output.top_picks],
            "portfolio_narrative": output.portfolio_narrative,
            "risk_adjusted_exposure": float(output.risk_adjusted_exposure),
            "evidence_pack_token_count": int(
                evidence_pack.get("trace_fragments", {}).get("budget", {}).get("token_count", 0) or 0
            ),
        }
        return output, advice_meta
    except Exception as exc:
        advice_meta = {
            "status": "fallback",
            "reason": str(exc),
            "final_conviction": "neutral",
            "final_score": 0.0,
            "confidence": 0.5,
            "top_picks": [],
            "portfolio_narrative": "MasterAgent fallback advisory.",
            "risk_adjusted_exposure": float(global_context.risk_budget.get("target_exposure", 0.0)),
            "evidence_pack_token_count": int(
                evidence_pack.get("trace_fragments", {}).get("budget", {}).get("token_count", 0) or 0
            ),
        }
        return None, advice_meta
