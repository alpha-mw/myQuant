from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from quant_investor.market.dag.evidence import (
    MASTER_EVIDENCE_PACK_FIELD_LIMIT,
    MASTER_EVIDENCE_PACK_SHORTLIST_LIMIT,
)


@dataclass
class ReportingArtifactsState:
    data_quality_summary: dict[str, Any]
    data_quality_diagnostics: Any
    what_if_plan: Any
    execution_trace: Any
    report_bundle: Any
    dag_artifacts: dict[str, Any]


def _build_reporting_artifacts(
    *,
    market: str,
    universe_key: str,
    all_symbols: list[str],
    researchable_symbols: list[str],
    candidate_symbols: list[str],
    quarantined_symbols: list[str],
    data_quality_issues: list[Any],
    read_results: Mapping[str, Any],
    shared_reader: Any,
    global_context: Any,
    provider_health: Mapping[str, Any],
    model_roles: Any,
    funnel_summary: dict[str, Any],
    bayesian_records: list[Any],
    review_bundle: Any,
    ic_hints_by_symbol: Mapping[str, Mapping[str, Any]],
    macro_verdict: Any,
    branch_summaries: Mapping[str, Any],
    branch_verdicts_by_symbol: Mapping[str, Mapping[str, Any]],
    branch_results: Mapping[str, Any],
    ic_decisions: list[Any],
    portfolio_plan: Any,
    portfolio_decision: Any,
    symbol_research_packets: Mapping[str, Any],
    shortlist: list[Any],
    portfolio_master_output: Any | None,
    portfolio_master_meta: Mapping[str, Any],
    portfolio_master_reliability: float,
    risk_decision: Any,
    tradability_snapshot: Mapping[str, Any],
    scoped_data_snapshot: Mapping[str, Any],
    download_stage: Mapping[str, Any] | None,
    category_count: int,
    funnel_output: Any,
    global_quant_verdict: Any,
    narrator_agent_cls: Any,
    build_data_quality_diagnostics_fn: Callable[..., Any],
    build_what_if_plan_fn: Callable[..., Any],
    build_execution_trace_fn: Callable[..., Any],
    build_bayesian_trace_fn: Callable[..., Any],
) -> ReportingArtifactsState:
    data_quality_summary = {
        "data_quality_issue_count": len(data_quality_issues),
        "quarantined_count": len(quarantined_symbols),
        "researchable_count": len(researchable_symbols),
        "shortlistable_count": len(candidate_symbols),
        "shortlist_count": len(shortlist),
        "quarantined_symbols": list(quarantined_symbols[:32]),
        "investment_risks": [issue.message for issue in data_quality_issues[:8]],
        "coverage_notes": [
            f"可研究覆盖 {len(researchable_symbols)} / 总覆盖 {len(all_symbols)}",
            f"漏斗候选 {len(candidate_symbols)} / 可研究覆盖 {len(researchable_symbols)}",
            f"Bayesian shortlist {len(shortlist)} / 漏斗候选 {len(candidate_symbols)}",
            f"resolver={shared_reader.snapshot().get('resolution_strategy', '')}",
        ],
        "diagnostic_notes": [
            f"{issue.symbol or 'unknown'}: {issue.message}"
            for issue in data_quality_issues[:8]
        ],
    }
    data_quality_diagnostics = build_data_quality_diagnostics_fn(
        total_symbols=all_symbols,
        researchable_symbols=researchable_symbols,
        shortlistable_symbols=candidate_symbols,
        final_selected_symbols=list(portfolio_plan.target_weights),
        quarantined_symbols=quarantined_symbols,
        issues=data_quality_issues,
    )
    global_context.data_quality_diagnostics = data_quality_diagnostics
    global_context.universe_tiers = {
        "total": list(all_symbols),
        "researchable": list(researchable_symbols),
        "shortlistable": list(candidate_symbols),
        "final_selected": list(portfolio_plan.target_weights),
    }
    funnel_summary["final_selected_count"] = len(portfolio_plan.target_weights)
    funnel_summary["compression_ratio"] = (
        f"{len(all_symbols)} -> {len(candidate_symbols)} -> {len(shortlist)} -> {len(portfolio_plan.target_weights)}"
    )
    what_if_plan = build_what_if_plan_fn(
        portfolio_plan=portfolio_plan,
        market_summary={
            "candidate_count": len(candidate_symbols),
            "selected_count": len(portfolio_plan.target_weights),
            "macro_score": float(macro_verdict.final_score),
        },
        model_roles=model_roles,
        candidate_count=len(candidate_symbols),
        selected_count=len(portfolio_plan.target_weights),
    )
    execution_trace = build_execution_trace_fn(
        model_roles=model_roles,
        download_stage=download_stage,
        analysis_meta={
            "market": market,
            "universe": universe_key,
            "batch_count": 1,
            "category_count": category_count,
            "total_stocks": len(all_symbols),
            "researchable_count": len(researchable_symbols),
            "quarantined_count": len(quarantined_symbols),
            "shortlistable_count": len(candidate_symbols),
            "shortlist_count": len(shortlist),
            "final_selected_count": len(portfolio_plan.target_weights),
            "quarantined_symbols": list(quarantined_symbols[:32]),
            "data_quality_issue_count": len(data_quality_issues),
            "fallback_reasons": list(review_bundle.fallback_reasons),
            "master_success": bool(portfolio_master_output is not None),
            "ic_hints_count": len(ic_hints_by_symbol),
            "company_name_coverage": sum(1 for item in shortlist if item.company_name),
            "evidence_pack_token_count": int(review_bundle.metadata.get("evidence_pack_token_count", 0) or 0),
            "evidence_pack_field_limit": MASTER_EVIDENCE_PACK_FIELD_LIMIT,
            "evidence_pack_shortlist_limit": MASTER_EVIDENCE_PACK_SHORTLIST_LIMIT,
            "candidate_count": len(candidate_symbols),
            "funnel_summary": funnel_summary,
            "bayesian_record_count": len(bayesian_records),
            "bayesian_trace": build_bayesian_trace_fn(bayesian_records=bayesian_records),
            "resolver": shared_reader.snapshot(),
            "global_context": global_context.to_dict(),
            "provider_health": provider_health,
            "data_quality_diagnostics": data_quality_diagnostics.to_dict(),
        },
        portfolio_plan={
            "selected_count": len(portfolio_plan.target_weights),
            "target_exposure": float(portfolio_plan.target_exposure),
            "max_single_weight": float(portfolio_plan.position_limits.get(next(iter(portfolio_plan.position_limits), ""), 0.0)) if portfolio_plan.position_limits else 0.0,
            "risk_veto": bool(risk_decision.veto),
            "action_cap": risk_decision.action_cap.value,
            "risk_summary": risk_decision.to_dict(),
            "execution_notes": portfolio_plan.execution_notes,
            "target_weights": dict(portfolio_plan.target_weights),
            "reliability": portfolio_master_reliability,
            "style_bias": global_context.style_exposures.get("style_bias", "balanced"),
            "provider_health": provider_health,
        },
        persistence_note="结构化 DAG 结果已准备完毕。",
    )
    portfolio_decision.what_if_plan = what_if_plan
    portfolio_decision.execution_trace = execution_trace

    review_bundle.branch_summaries = branch_summaries
    review_bundle.risk_decision = risk_decision
    review_bundle.metadata.update(
        {
            "portfolio_master_output": portfolio_master_output.model_dump() if portfolio_master_output is not None and hasattr(portfolio_master_output, "model_dump") else dict(portfolio_master_meta),
            "data_quality_summary": data_quality_summary,
            "evidence_pack_token_count": review_bundle.metadata.get("evidence_pack_token_count", 0),
        }
    )

    narrator_agent = narrator_agent_cls()
    report_bundle = narrator_agent.run(
        {
            "macro_verdict": macro_verdict,
            "branch_summaries": branch_summaries,
            "ic_decisions": ic_decisions,
            "portfolio_plan": portfolio_plan,
            "review_bundle": review_bundle,
            "ic_hints_by_symbol": ic_hints_by_symbol,
            "model_role_metadata": model_roles,
            "execution_trace": execution_trace,
            "what_if_plan": what_if_plan,
            "global_context": global_context,
            "symbol_research_packets": symbol_research_packets,
            "shortlist": shortlist,
            "portfolio_decision": portfolio_decision,
            "bayesian_records": bayesian_records,
            "funnel_summary": funnel_summary,
            "run_diagnostics": {
                **data_quality_summary,
                "coverage_notes": data_quality_summary["coverage_notes"]
                + [
                    f"physical_directories={len(shared_reader.snapshot().get('physical_directories_used_for_full_a', []))}",
                ],
                "diagnostic_notes": data_quality_summary["diagnostic_notes"]
                + [
                    f"{symbol}: {issue.message}"
                    for symbol, read_result in list(read_results.items())[:5]
                    for issue in read_result.issues[:1]
                ],
                "resolver": shared_reader.snapshot(),
                "global_context": global_context.to_dict(),
            },
        }
    )

    dag_artifacts = {
        "global_context": global_context,
        "symbol_research_packets": dict(symbol_research_packets),
        "branch_verdicts_by_symbol": dict(branch_verdicts_by_symbol),
        "branch_summaries": dict(branch_summaries),
        "branch_results": dict(branch_results),
        "funnel_output": funnel_output,
        "funnel_summary": funnel_summary,
        "bayesian_records": bayesian_records,
        "global_quant_verdict": global_quant_verdict,
        "macro_verdict": macro_verdict,
        "risk_decision": risk_decision,
        "ic_decisions": ic_decisions,
        "portfolio_plan": portfolio_plan,
        "portfolio_decision": portfolio_decision,
        "portfolio_master_output": portfolio_master_output,
        "portfolio_master_meta": dict(portfolio_master_meta),
        "review_bundle": review_bundle,
        "model_role_metadata": model_roles,
        "what_if_plan": what_if_plan,
        "execution_trace": execution_trace,
        "tradability_snapshot": tradability_snapshot,
        "data_quality_issues": [issue.to_dict() for issue in data_quality_issues],
        "data_quality_summary": data_quality_summary,
        "data_quality_diagnostics": data_quality_diagnostics,
        "provider_health": provider_health,
        "resolver": shared_reader.snapshot(),
        "data_snapshot": dict(scoped_data_snapshot),
        "report_bundle": report_bundle,
    }
    return ReportingArtifactsState(
        data_quality_summary=data_quality_summary,
        data_quality_diagnostics=data_quality_diagnostics,
        what_if_plan=what_if_plan,
        execution_trace=execution_trace,
        report_bundle=report_bundle,
        dag_artifacts=dag_artifacts,
    )
