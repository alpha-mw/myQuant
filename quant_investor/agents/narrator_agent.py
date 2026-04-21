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
    ExecutionTrace,
    ExecutionTraceStep,
    ICDecision,
    ModelRoleMetadata,
    PortfolioPlan,
    ReportBundle,
    StockReviewBundle,
    WhatIfPlan,
    WhatIfScenario,
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
        review_bundle = self._normalize_review_bundle(self.copy_value(envelope.get("review_bundle")))
        ic_hints_by_symbol = self._normalize_ic_hints_by_symbol(
            self.copy_value(envelope.get("ic_hints_by_symbol"))
        )
        model_role_metadata = self._normalize_model_role_metadata(
            self.copy_value(envelope.get("model_role_metadata"))
        )
        execution_trace = self._normalize_execution_trace(
            self.copy_value(envelope.get("execution_trace"))
        )
        what_if_plan = self._normalize_what_if_plan(
            self.copy_value(envelope.get("what_if_plan"))
        )
        global_context = self.copy_value(envelope.get("global_context"))
        symbol_research_packets = self.copy_value(envelope.get("symbol_research_packets") or {})
        shortlist = self.copy_value(envelope.get("shortlist") or [])
        portfolio_decision = self.copy_value(envelope.get("portfolio_decision"))
        bayesian_records = self.copy_value(envelope.get("bayesian_records") or [])
        funnel_summary = self.copy_value(envelope.get("funnel_summary") or {})

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
        review_sections = ConclusionRenderer.render_review_sections(review_bundle)
        markdown_report = ConclusionRenderer.render_markdown(
            executive_summary=executive_summary,
            market_view=market_view,
            branch_conclusions=branch_conclusions,
            stock_cards=stock_cards,
            coverage_summary=bucketed["coverage_summary"],
            appendix_diagnostics=bucketed["appendix_diagnostics"],
        )
        bayesian_section = ConclusionRenderer.render_bayesian_section(
            bayesian_records=bayesian_records,
            funnel_summary=funnel_summary,
            symbol_name_map=getattr(global_context, "symbol_name_map", {}) if global_context is not None else {},
        )
        run_context = ConclusionRenderer.render_run_context(
            model_role_metadata=model_role_metadata,
            execution_trace=execution_trace,
            what_if_plan=what_if_plan,
        )
        if review_sections:
            markdown_report = markdown_report.rstrip() + "\n\n" + "\n".join(review_sections).strip() + "\n"
        if bayesian_section:
            markdown_report = markdown_report.rstrip() + "\n\n" + "\n".join(bayesian_section).strip() + "\n"
        if run_context:
            markdown_report = markdown_report.rstrip() + "\n\n" + "\n".join(run_context).strip() + "\n"

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
            global_context=global_context,
            symbol_research_packets=dict(symbol_research_packets),
            shortlist=list(shortlist),
            portfolio_decision=portfolio_decision,
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
            review_bundle=review_bundle,
            ic_hints_by_symbol=ic_hints_by_symbol,
            model_role_metadata=model_role_metadata,
            execution_trace=execution_trace,
            what_if_plan=what_if_plan,
            metadata={
                "narrator_read_only": True,
                "stock_card_count": len(stock_cards),
                "coverage_count": bucketed["counts"]["coverage_count"],
                "diagnostic_count": bucketed["counts"]["diagnostic_count"],
                "investment_risk_count": bucketed["counts"]["investment_risk_count"],
                "funnel_summary": funnel_summary,
                "bayesian_record_count": len(bayesian_records),
                "shortlist_count": len(shortlist),
                "final_selected_count": len(getattr(portfolio_decision, "target_weights", {}) or {}),
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

    @staticmethod
    def _normalize_review_bundle(payload: Any) -> StockReviewBundle | None:
        if payload is None:
            return None
        if isinstance(payload, StockReviewBundle):
            return payload
        if not isinstance(payload, Mapping):
            return None
        return StockReviewBundle(
            agent_name=str(payload.get("agent_name") or "StockReviewOrchestrator"),
            branch_overlay_verdicts_by_symbol=dict(payload.get("branch_overlay_verdicts_by_symbol", {})),
            master_hints_by_symbol=dict(payload.get("master_hints_by_symbol", {})),
            ic_hints_by_symbol=dict(payload.get("ic_hints_by_symbol", {})),
            branch_summaries=dict(payload.get("branch_summaries", {})),
            macro_verdict=payload.get("macro_verdict"),
            risk_decision=payload.get("risk_decision"),
            telemetry=list(payload.get("telemetry", [])),
            fallback_reasons=[str(item) for item in payload.get("fallback_reasons", [])],
            metadata=dict(payload.get("metadata", {})),
        )

    @staticmethod
    def _normalize_ic_hints_by_symbol(payload: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(payload, Mapping):
            return {}
        result: dict[str, dict[str, Any]] = {}
        for symbol, hint in payload.items():
            if isinstance(hint, Mapping):
                result[str(symbol)] = dict(hint)
        return result

    @staticmethod
    def _normalize_model_role_metadata(payload: Any) -> ModelRoleMetadata | None:
        if payload is None:
            return None
        if isinstance(payload, ModelRoleMetadata):
            return payload
        if not isinstance(payload, Mapping):
            return None
        return ModelRoleMetadata(
            branch_model=str(payload.get("branch_model", "")),
            master_model=str(payload.get("master_model", "")),
            branch_provider=str(payload.get("branch_provider", "")),
            master_provider=str(payload.get("master_provider", "")),
            branch_timeout=float(payload.get("branch_timeout", 0.0)),
            master_timeout=float(payload.get("master_timeout", 0.0)),
            agent_layer_enabled=bool(payload.get("agent_layer_enabled", False)),
            branch_role=str(payload.get("branch_role", "per-stock analysis")),
            master_role=str(
                payload.get(
                    "master_role",
                    "master synthesis / portfolio-level judgment before deterministic risk and sizing",
                )
            ),
            metadata=dict(payload.get("metadata", {})),
        )

    @staticmethod
    def _normalize_execution_trace(payload: Any) -> ExecutionTrace | None:
        if payload is None:
            return None
        if isinstance(payload, ExecutionTrace):
            return payload
        if not isinstance(payload, Mapping):
            return None
        return ExecutionTrace(
            model_roles=NarratorAgent._normalize_model_role_metadata(payload.get("model_roles"))
            or ModelRoleMetadata(),
            key_parameters=dict(payload.get("key_parameters", {})),
            steps=[
                step
                if isinstance(step, ExecutionTraceStep)
                else ExecutionTraceStep(
                    stage=str(step.get("stage", "")),
                    role=str(step.get("role", "")),
                    model=str(step.get("model", "")),
                    success=bool(step.get("success", True)),
                    conclusion=str(step.get("conclusion", "")),
                    parameters=dict(step.get("parameters", {})),
                    fallback_reason=str(step.get("fallback_reason", "")),
                    timeout_seconds=float(step.get("timeout_seconds", 0.0)),
                    metadata=dict(step.get("metadata", {})),
                )
                for step in payload.get("steps", [])
                if isinstance(step, (ExecutionTraceStep, Mapping))
            ],
            final_deterministic_outcome=dict(payload.get("final_deterministic_outcome", {})),
            metadata=dict(payload.get("metadata", {})),
        )

    @staticmethod
    def _normalize_what_if_plan(payload: Any) -> WhatIfPlan | None:
        if payload is None:
            return None
        if isinstance(payload, WhatIfPlan):
            return payload
        if not isinstance(payload, Mapping):
            return None
        return WhatIfPlan(
            scenarios=[
                scenario
                if isinstance(scenario, WhatIfScenario)
                else WhatIfScenario(
                    scenario_name=str(scenario.get("scenario_name", "")),
                    trigger=str(scenario.get("trigger", "")),
                    monitoring_indicators=[str(item) for item in scenario.get("monitoring_indicators", [])],
                    action=str(scenario.get("action", "")),
                    position_adjustment_rule=str(scenario.get("position_adjustment_rule", "")),
                    rerun_full_market_daily_path=bool(scenario.get("rerun_full_market_daily_path", False)),
                    metadata=dict(scenario.get("metadata", {})),
                )
                for scenario in payload.get("scenarios", [])
                if isinstance(scenario, (WhatIfScenario, Mapping))
            ],
            metadata=dict(payload.get("metadata", {})),
            generated_by=str(payload.get("generated_by", "deterministic")),
        )
