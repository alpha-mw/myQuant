from __future__ import annotations

from typing import Any

from quant_investor.branch_contracts import PortfolioStrategy, TradeRecommendation
from quant_investor.llm_gateway import current_usage_session_id, update_usage_markdown
from quant_investor.pipeline.result_types import QuantInvestorPipelineResult
from quant_investor.versioning import (
    ARCHITECTURE_VERSION,
    BRANCH_SCHEMA_VERSION,
    CALIBRATION_SCHEMA_VERSION,
    DEBATE_TEMPLATE_VERSION,
    IC_PROTOCOL_VERSION,
    REPORT_PROTOCOL_VERSION,
)


def portfolio_decision_to_strategy(
    *,
    portfolio_decision: Any,
    report_bundle: Any,
    global_context: Any,
) -> PortfolioStrategy:
    shortlist = list(getattr(portfolio_decision, "shortlist", []) or [])
    target_weights = dict(getattr(portfolio_decision, "target_weights", {}) or {})
    recommendations: list[TradeRecommendation] = []
    for item in shortlist:
        rationale = list(getattr(item, "rationale", []) or [])
        recommendations.append(
            TradeRecommendation(
                symbol=str(getattr(item, "symbol", "")),
                action=str(getattr(getattr(item, "action", None), "value", "hold")),
                weight=float(target_weights.get(getattr(item, "symbol", ""), getattr(item, "suggested_weight", 0.0)) or 0.0),
                suggested_weight=float(getattr(item, "suggested_weight", 0.0) or 0.0),
                confidence=float(getattr(item, "confidence", 0.0) or 0.0),
                expected_upside=float(getattr(item, "expected_upside", 0.0) or 0.0),
                rationale="；".join(rationale[:2]) or str(getattr(item, "symbol", "")),
                one_line_conclusion="；".join(rationale[:1]) or str(getattr(item, "symbol", "")),
                risk_flags=list(getattr(item, "risk_flags", []) or []),
                metadata={"company_name": str(getattr(item, "company_name", ""))},
            )
        )
    return PortfolioStrategy(
        target_exposure=float(getattr(portfolio_decision, "target_exposure", 0.0) or 0.0),
        total_exposure=float(getattr(portfolio_decision, "target_exposure", 0.0) or 0.0),
        gross_exposure=float(getattr(portfolio_decision, "target_gross_exposure", 0.0) or 0.0),
        net_exposure=float(getattr(portfolio_decision, "target_net_exposure", 0.0) or 0.0),
        cash_ratio=float(getattr(portfolio_decision, "cash_ratio", 1.0) or 1.0),
        style_bias=str(getattr(global_context, "style_exposures", {}).get("style_bias", "均衡")),
        candidate_symbols=[str(getattr(item, "symbol", "")) for item in shortlist],
        target_weights=target_weights,
        target_positions=dict(getattr(portfolio_decision, "target_positions", {}) or {}),
        position_limits=dict(getattr(getattr(report_bundle, "portfolio_plan", None), "position_limits", {}) or {}),
        execution_notes=list(getattr(getattr(report_bundle, "portfolio_plan", None), "execution_notes", []) or []),
        branch_consensus={
            name: float(getattr(verdict, "final_score", 0.0) or 0.0)
            for name, verdict in dict(getattr(report_bundle, "branch_verdicts", {}) or {}).items()
        },
        risk_summary=dict(getattr(portfolio_decision, "risk_constraints", {}) or {}),
        provenance_summary={
            "generated_from": "market_dag",
            "tracked_symbols": [str(getattr(item, "symbol", "")) for item in shortlist],
        },
        research_mode="production",
        summary=str(getattr(report_bundle, "summary", "") or ""),
        notes=list(getattr(report_bundle, "executive_summary", []) or []),
        metadata={
            "generated_from": "market_dag",
            "report_headline": str(getattr(report_bundle, "headline", "") or ""),
        },
        recommendations=recommendations,
        trade_recommendations=recommendations,
        blocked_symbols=list(getattr(getattr(report_bundle, "portfolio_plan", None), "blocked_symbols", []) or []),
        rejected_symbols=list(getattr(getattr(report_bundle, "portfolio_plan", None), "rejected_symbols", []) or []),
        sector_preferences=[],
        stop_loss_policy={},
        symbol_convictions={symbol: float(weight) for symbol, weight in target_weights.items()},
    )


def build_pipeline_result_from_dag(
    *,
    dag_artifacts: dict[str, Any],
    usage_records: list[Any],
    usage_summary: Any,
    effective_usage_records: list[Any],
    effective_usage_summary: Any,
    total_time: float,
    enable_agent_layer: bool,
    execution_log: list[str],
) -> QuantInvestorPipelineResult:
    report_bundle = dag_artifacts.get("report_bundle")
    portfolio_decision = dag_artifacts.get("portfolio_decision")
    global_context = dag_artifacts.get("global_context")
    data_snapshot = dict(
        dag_artifacts.get("data_snapshot", {})
        or (getattr(global_context, "metadata", {}) or {}).get("data_snapshot", {})
    )
    final_strategy = portfolio_decision_to_strategy(
        portfolio_decision=portfolio_decision,
        report_bundle=report_bundle,
        global_context=global_context,
    )
    final_report = update_usage_markdown(
        getattr(report_bundle, "markdown_report", "") or "",
        usage_summary,
        title="## LLM 可观测性（统一 DAG）",
    )
    result = QuantInvestorPipelineResult(
        architecture_version=ARCHITECTURE_VERSION,
        branch_schema_version=BRANCH_SCHEMA_VERSION,
        ic_protocol_version=getattr(report_bundle, "ic_protocol_version", IC_PROTOCOL_VERSION),
        report_protocol_version=getattr(report_bundle, "report_protocol_version", REPORT_PROTOCOL_VERSION),
        calibration_schema_version=CALIBRATION_SCHEMA_VERSION,
        debate_template_version=DEBATE_TEMPLATE_VERSION,
        data_bundle=None,
        branch_results=dict(dag_artifacts.get("branch_results", {})),
        calibrated_signals={},
        risk_results=dag_artifacts.get("risk_decision"),
        final_strategy=final_strategy,
        final_report=final_report,
        execution_log=list(execution_log) + ["[INFO] unified market DAG executed", "single-mainline 完成"],
        layer_timings={"total": total_time},
        total_time=total_time,
        agent_orchestration=dict(dag_artifacts),
        agent_portfolio_plan=dag_artifacts.get("portfolio_plan"),
        agent_report_bundle=report_bundle,
        agent_ic_decisions=dag_artifacts.get("ic_decisions"),
        agent_review_bundle=dag_artifacts.get("review_bundle"),
        ic_hints_by_symbol=dict(getattr(dag_artifacts.get("review_bundle"), "ic_hints_by_symbol", {}) or {}),
        model_role_metadata=dag_artifacts.get("model_role_metadata"),
        execution_trace=dag_artifacts.get("execution_trace"),
        what_if_plan=dag_artifacts.get("what_if_plan"),
        llm_usage_records=list(usage_records),
        llm_usage_summary=usage_summary,
        llm_effective_records=list(effective_usage_records),
        llm_effective_summary=effective_usage_summary,
        llm_usage_session_id=current_usage_session_id() or "",
        data_snapshot=data_snapshot,
        raw_data={},
        factor_data={},
        model_predictions={},
        macro_signal=str(getattr(getattr(report_bundle, "macro_verdict", None), "metadata", {}).get("policy_signal", "🟡")),
        macro_summary=str(getattr(getattr(report_bundle, "macro_verdict", None), "thesis", "") or ""),
        baseline_strategy=final_strategy,
        baseline_risk_result=dag_artifacts.get("risk_decision"),
        macro_verdict=getattr(report_bundle, "macro_verdict", None),
        reviewed_research_by_symbol=dict(dag_artifacts.get("branch_verdicts_by_symbol", {})),
        reviewed_branch_summaries=dict(dag_artifacts.get("branch_summaries", {})),
        branch_review_outputs=(dict(dag_artifacts.get("branch_summaries", {})) if enable_agent_layer else {}),
        master_review_output=(dag_artifacts.get("portfolio_master_output") if enable_agent_layer else None),
        risk_review_output=dag_artifacts.get("risk_decision"),
        review_bundle=dag_artifacts.get("review_bundle"),
        symbol_review_bundle={},
        agent_layer_enabled=enable_agent_layer,
        pipeline_mode="bayesian",
        global_context=global_context,
        funnel_output=dag_artifacts.get("funnel_output"),
        bayesian_records=list(dag_artifacts.get("bayesian_records", [])),
        shortlist_evidence=list(dag_artifacts.get("shortlist_evidence", [])),
        bayesian_shortlist_symbols=[item.symbol for item in list(dag_artifacts.get("shortlist", []))],
    )
    result.final_strategy.metadata["agent_layer_enabled"] = enable_agent_layer
    provenance_summary = dict(getattr(result.final_strategy, "provenance_summary", {}))
    provenance_summary["agent_layer_enabled"] = enable_agent_layer
    result.final_strategy.provenance_summary = provenance_summary
    return result
