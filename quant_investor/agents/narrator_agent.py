"""
只读型 NarratorAgent。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from quant_investor.agent_protocol import (
    ActionLabel,
    AgentStatus,
    BranchVerdict,
    ConfidenceLabel,
    Direction,
    EventNote,
    ICDecision,
    PortfolioPlan,
    ReportBundle,
)
from quant_investor.agents.base import BaseAgent
from quant_investor.reporting.conclusion_renderer import ConclusionRenderer
from quant_investor.reporting.diagnostics_bucketizer import DiagnosticsBucketizer
from quant_investor.reporting.executive_summary import ExecutiveSummaryBuilder


class NarratorAgent(BaseAgent):
    """只把结构化决策结果渲染为报告。"""

    agent_name = "NarratorAgent"

    def run(self, payload: Mapping[str, Any]) -> ReportBundle:
        envelope = self.ensure_payload(payload)
        self.require_keys(
            envelope,
            "macro_verdict",
            "branch_summaries",
            "ic_decisions",
            "portfolio_plan",
            "run_diagnostics",
        )

        macro_verdict = self.copy_value(envelope["macro_verdict"])
        portfolio_plan = self.copy_value(envelope["portfolio_plan"])
        if not isinstance(macro_verdict, BranchVerdict):
            raise TypeError("macro_verdict 必须是 BranchVerdict")
        if not isinstance(portfolio_plan, PortfolioPlan):
            raise TypeError("portfolio_plan 必须是 PortfolioPlan")

        ic_decisions = self._normalize_ic_decisions(self.copy_value(envelope["ic_decisions"]))
        branch_verdicts = self._normalize_branch_summaries(
            self.copy_value(envelope["branch_summaries"])
        )

        bucketed = DiagnosticsBucketizer(
            branch_summaries=branch_verdicts,
            run_diagnostics=self.copy_value(envelope["run_diagnostics"]),
        ).bucket()
        executive_summary = ExecutiveSummaryBuilder(
            macro_verdict=macro_verdict,
            branch_summaries=branch_verdicts,
            ic_decisions=ic_decisions,
            portfolio_plan=portfolio_plan,
        ).build()
        market_view = ConclusionRenderer.render_market_view(
            macro_verdict=macro_verdict,
            ic_decisions=ic_decisions,
            portfolio_plan=portfolio_plan,
        )
        branch_conclusions = ConclusionRenderer.render_branch_conclusions(branch_verdicts)
        stock_cards = ConclusionRenderer.render_stock_cards(
            ic_decisions=ic_decisions,
            portfolio_plan=portfolio_plan,
        )
        markdown_report = ConclusionRenderer.render_markdown(
            executive_summary=executive_summary,
            market_view=market_view,
            branch_conclusions=branch_conclusions,
            stock_cards=stock_cards,
            coverage_summary=bucketed["coverage_summary"],
            appendix_diagnostics=bucketed["appendix_diagnostics"],
        )

        diagnostics = [
            EventNote(
                title=f"diagnostic_{index}",
                message=message,
            )
            for index, message in enumerate(bucketed["appendix_diagnostics"][1:], start=1)
        ]

        return ReportBundle(
            headline=executive_summary[0],
            summary=" ".join(executive_summary),
            macro_verdict=macro_verdict,
            branch_verdicts=branch_verdicts,
            ic_decision=ic_decisions[0] if ic_decisions else None,
            ic_decisions=ic_decisions,
            portfolio_plan=portfolio_plan,
            markdown_report=markdown_report,
            executive_summary=executive_summary,
            market_view=market_view,
            branch_conclusions=branch_conclusions,
            stock_cards=stock_cards,
            coverage_summary=bucketed["coverage_summary"],
            appendix_diagnostics=bucketed["appendix_diagnostics"],
            highlights=executive_summary,
            warnings=bucketed["investment_risks"],
            diagnostics=diagnostics,
            metadata={
                "narrator_read_only": True,
                "stock_card_count": len(stock_cards),
                "coverage_count": bucketed["counts"]["coverage_count"],
                "diagnostic_count": bucketed["counts"]["diagnostic_count"],
                "investment_risk_count": bucketed["counts"]["investment_risk_count"],
            },
        )

    @staticmethod
    def _normalize_ic_decisions(payload: Any) -> list[ICDecision]:
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
            raise TypeError("ic_decisions 必须是 ICDecision 列表")
        decisions = [item for item in payload if isinstance(item, ICDecision)]
        if len(decisions) != len(payload):
            raise TypeError("ic_decisions 中存在非 ICDecision 项")
        return decisions

    def _normalize_branch_summaries(
        self,
        payload: Any,
    ) -> dict[str, BranchVerdict]:
        if not isinstance(payload, Mapping):
            raise TypeError("branch_summaries 必须是 Mapping")
        result: dict[str, BranchVerdict] = {}
        for name, branch in payload.items():
            key = str(name)
            if isinstance(branch, BranchVerdict):
                result[key] = branch
                continue
            if not isinstance(branch, Mapping):
                raise TypeError("branch_summaries 的值必须是 BranchVerdict 或 Mapping")
            thesis = str(branch.get("thesis") or branch.get("conclusion") or "").strip()
            if not thesis:
                thesis = f"{key} 分支已生成结构化结论。"
            result[key] = BranchVerdict(
                agent_name=str(branch.get("agent_name") or key),
                thesis=thesis,
                status=branch.get("status", AgentStatus.SUCCESS),
                direction=branch.get(
                    "direction",
                    self.score_to_direction(float(branch.get("score", 0.0))),
                ),
                action=branch.get(
                    "action",
                    self.score_to_action(float(branch.get("score", 0.0))),
                ),
                confidence_label=branch.get(
                    "confidence_label",
                    self.confidence_to_label(float(branch.get("confidence", 0.0))),
                ),
                final_score=float(branch.get("final_score", branch.get("score", 0.0))),
                final_confidence=float(
                    branch.get("final_confidence", branch.get("confidence", 0.0))
                ),
                investment_risks=[str(item) for item in branch.get("investment_risks", branch.get("risks", []))],
                coverage_notes=[str(item) for item in branch.get("coverage_notes", [])],
                diagnostic_notes=[str(item) for item in branch.get("diagnostic_notes", [])],
                metadata=dict(branch.get("metadata", {})),
            )
        return result
