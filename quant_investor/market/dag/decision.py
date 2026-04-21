from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from quant_investor.agent_protocol import BayesianDecisionRecord, ICDecision, PortfolioDecision, RiskDecision
from quant_investor.bayesian.calibration import CalibrationStore
from quant_investor.market.dag.common import _dedupe_texts
from quant_investor.market.dag.evidence import _build_master_evidence_pack
from quant_investor.market.dag.shortlist import _build_shortlist_from_bayesian_records
from quant_investor.reporting.run_artifacts import build_funnel_summary


@dataclass
class BayesianSelectionState:
    bayesian_records: list[BayesianDecisionRecord]
    shortlist: list[Any]
    funnel_summary: dict[str, Any]
    evidence_pack: dict[str, Any]
    portfolio_master_output: Any | None
    portfolio_master_meta: dict[str, Any]
    portfolio_master_reliability: float


@dataclass
class PortfolioConstructionState:
    risk_decision: RiskDecision
    ic_decisions: list[ICDecision]
    portfolio_plan: Any
    portfolio_decision: PortfolioDecision


def _run_bayesian_selection_phase(
    *,
    candidate_symbols: list[str],
    company_name_map: Mapping[str, str],
    symbol_research_packets: Mapping[str, Any],
    research_by_symbol: Mapping[str, Mapping[str, Any]],
    branch_summaries: Mapping[str, Any],
    branch_results: Mapping[str, Any],
    macro_verdict: Any,
    global_context: Any,
    model_roles: Any,
    resolver_snapshot: Mapping[str, Any],
    data_quality_issues: list[Any],
    top_k: int,
    all_symbols: list[str],
    funnel_output: Any,
    provider_health: Mapping[str, Any],
    master_timeout: float,
    master_reasoning_effort: str,
    master_model_resolution: Any,
    master_candidate_models: list[str],
    recall_context: Mapping[str, Any] | None,
    hierarchical_prior_builder_cls: Any,
    likelihood_mapper_cls: Any,
    posterior_engine_cls: Any,
    master_agent_cls: Any,
    llm_client_cls: Any,
    portfolio_master_advisory_fn: Callable[..., tuple[Any | None, dict[str, Any]]],
) -> BayesianSelectionState:
    prior_builder = hierarchical_prior_builder_cls()
    try:
        likelihood_mapper = likelihood_mapper_cls(
            calibration_store=CalibrationStore(),
            recall_context=recall_context,
            global_context=global_context,
        )
    except TypeError:
        likelihood_mapper = likelihood_mapper_cls()
    posterior_engine = posterior_engine_cls()
    bayesian_records: list[BayesianDecisionRecord] = []
    degraded_map = {
        "kline": provider_health.get("kline", {}).get("mode") != "hybrid",
        "fundamental": False,
        "intelligence": False,
        "quant": False,
    }
    for symbol in candidate_symbols:
        prior = prior_builder.build_prior(symbol, global_context)
        likelihoods = likelihood_mapper.compute_likelihoods(
            branch_results=branch_results,
            symbol=symbol,
            candidate_symbols=set(candidate_symbols),
        )
        posterior = posterior_engine.compute_posterior(
            prior,
            likelihoods,
            symbol=symbol,
            company_name=company_name_map.get(symbol, ""),
            regime=global_context.macro_regime or "未知",
            is_degraded=degraded_map,
        )
        bayesian_records.append(
            BayesianDecisionRecord(
                symbol=symbol,
                company_name=company_name_map.get(symbol, ""),
                prior=posterior.prior.to_dict(),
                likelihoods=posterior.likelihoods.to_dict(),
                posterior_win_rate=posterior.posterior_win_rate,
                posterior_expected_alpha=posterior.posterior_expected_alpha,
                posterior_confidence=posterior.posterior_confidence,
                posterior_action_score=posterior.posterior_action_score,
                posterior_edge_after_costs=posterior.posterior_edge_after_costs,
                posterior_capacity_penalty=posterior.posterior_capacity_penalty,
                correlation_discount=posterior.correlation_discount,
                coverage_discount=posterior.coverage_discount,
                data_quality_penalty=posterior.data_quality_penalty,
                fallback_penalty=posterior.fallback_penalty,
                regime_adjustment=posterior.regime_adjustment,
                evidence_sources=list(posterior.evidence_sources),
                action_threshold_used=posterior.action_threshold_used,
                metadata={
                    "category": str(symbol_research_packets[symbol].category),
                    "posterior_edge_after_costs": posterior.posterior_edge_after_costs,
                    "posterior_capacity_penalty": posterior.posterior_capacity_penalty,
                    "profile": str((global_context.metadata or {}).get("selection_profile", {}).get("funnel_profile", "classic")),
                    "momentum_strength": float((posterior.metadata or {}).get("momentum_strength", 0.0) if isinstance(getattr(posterior, "metadata", {}), Mapping) else 0.0),
                    "fake_breakout_penalty": float((posterior.metadata or {}).get("fake_breakout_penalty", 0.0) if isinstance(getattr(posterior, "metadata", {}), Mapping) else 0.0),
                    "setup_failure_penalty": float((posterior.metadata or {}).get("setup_failure_penalty", 0.0) if isinstance(getattr(posterior, "metadata", {}), Mapping) else 0.0),
                    "crowding_penalty": float((posterior.metadata or {}).get("crowding_penalty", 0.0) if isinstance(getattr(posterior, "metadata", {}), Mapping) else 0.0),
                    "history_confidence": float((posterior.metadata or {}).get("history_confidence", 0.0) if isinstance(getattr(posterior, "metadata", {}), Mapping) else 0.0),
                    "calibration_samples": dict((posterior.metadata or {}).get("calibration_samples", {}) or {}) if isinstance(getattr(posterior, "metadata", {}), Mapping) else {},
                    "kill_switch": bool((posterior.metadata or {}).get("kill_switch", False)) if isinstance(getattr(posterior, "metadata", {}), Mapping) else False,
                    "sector": str((posterior.metadata or {}).get("sector", "")) if isinstance(getattr(posterior, "metadata", {}), Mapping) else "",
                },
            )
        )
    bayesian_records.sort(key=lambda item: (-float(item.posterior_action_score), item.symbol))
    for index, record in enumerate(bayesian_records, start=1):
        record.rank = index
        record.metadata = dict(record.metadata or {})
        record.metadata["rank"] = index

    shortlist = _build_shortlist_from_bayesian_records(
        posterior_results=bayesian_records,
        company_name_map=company_name_map,
        top_k=top_k,
    )
    for item in shortlist:
        branch_map = research_by_symbol.get(item.symbol, {})
        item.risk_flags = _dedupe_texts([risk for verdict in branch_map.values() for risk in verdict.investment_risks])[:5]
        item.rationale = _dedupe_texts(
            list(item.rationale)
            + [verdict.thesis for verdict in branch_map.values()]
        )[:5]

    funnel_summary = build_funnel_summary(
        universe_size=len(all_symbols),
        candidates_count=len(candidate_symbols),
        shortlist_count=len(shortlist),
        final_selected_count=0,
        excluded_symbols=funnel_output.excluded_symbols,
        funnel_metadata=funnel_output.funnel_metadata,
    )
    evidence_pack = _build_master_evidence_pack(
        shortlist=shortlist,
        branch_summaries=branch_summaries,
        macro_verdict=macro_verdict,
        risk_constraints=global_context.risk_budget,
        model_roles=model_roles,
        resolver_snapshot=resolver_snapshot,
        data_quality_issues=data_quality_issues,
        company_name_map=company_name_map,
        top_k=top_k,
    )

    portfolio_master_agent = master_agent_cls(
        llm_client=llm_client_cls(timeout=master_timeout),
        model=master_model_resolution.resolved_model,
        candidate_models=list(master_candidate_models),
        fallback_model=master_model_resolution.fallback_model,
        reasoning_effort=master_reasoning_effort,
        timeout=master_timeout,
    )
    portfolio_master_output, portfolio_master_meta = portfolio_master_advisory_fn(
        master_agent=portfolio_master_agent,
        macro_verdict=macro_verdict,
        shortlist=shortlist,
        global_context=global_context,
        evidence_pack=evidence_pack,
        recall_context=recall_context,
    )
    return BayesianSelectionState(
        bayesian_records=bayesian_records,
        shortlist=shortlist,
        funnel_summary=funnel_summary,
        evidence_pack=evidence_pack,
        portfolio_master_output=portfolio_master_output,
        portfolio_master_meta=portfolio_master_meta,
        portfolio_master_reliability=float(portfolio_master_meta.get("confidence", 0.0) or 0.0),
    )


def _run_portfolio_construction_phase(
    *,
    shortlist: list[Any],
    branch_summaries: Mapping[str, Any],
    macro_verdict: Any,
    global_context: Any,
    data_quality_issues: list[Any],
    ic_hints_by_symbol: Mapping[str, Mapping[str, Any]],
    research_by_symbol: Mapping[str, Mapping[str, Any]],
    tradability_snapshot: Mapping[str, Mapping[str, Any]],
    funnel_summary: Mapping[str, Any],
    bayesian_records: list[Any],
    candidate_symbols: list[str],
    portfolio_master_output: Any | None,
    portfolio_master_meta: Mapping[str, Any],
    risk_guard_cls: Any,
    ic_coordinator_cls: Any,
    portfolio_constructor_cls: Any,
    attach_symbol_to_ic_decision_fn: Callable[..., ICDecision],
) -> PortfolioConstructionState:
    risk_guard = risk_guard_cls()
    risk_decision = risk_guard.run(
        {
            "branch_verdicts": branch_summaries,
            "macro_verdict": macro_verdict,
            "portfolio_state": {
                "candidate_symbols": [item.symbol for item in shortlist],
                "current_weights": {},
            },
            "constraints": {
                "gross_exposure_cap": float(global_context.risk_budget.get("target_exposure", 0.55)),
                "max_weight": float(global_context.risk_budget.get("max_single_weight", 0.12)),
                "risk_flags": _dedupe_texts([issue.message for issue in data_quality_issues[:8]]),
                "data_quality_issue_count": len(data_quality_issues),
            },
        }
    )

    ic_coordinator = ic_coordinator_cls()
    shortlisted_symbols = [item.symbol for item in shortlist]
    shortlist_by_symbol = {item.symbol: item for item in shortlist}
    ic_decisions: list[ICDecision] = []
    for symbol in shortlisted_symbols:
        decision = ic_coordinator.run(
            {
                "branch_verdicts": research_by_symbol[symbol],
                "risk_decision": risk_decision,
                "ic_hints": ic_hints_by_symbol.get(symbol, {}),
            }
        )
        decision = attach_symbol_to_ic_decision_fn(
            decision,
            symbol=symbol,
            risk_decision=risk_decision,
            current_weight=0.0,
            tradability_info=tradability_snapshot[symbol],
            ic_hint=ic_hints_by_symbol.get(symbol, {}),
            shortlist_item=shortlist_by_symbol.get(symbol),
        )
        ic_decisions.append(decision)

    sector_bucket_limit = int(global_context.risk_budget.get("sector_bucket_limit", 0) or 0)
    sector_caps: dict[str, float] = {}
    if sector_bucket_limit > 0:
        for symbol in shortlisted_symbols:
            tradability = dict(tradability_snapshot.get(symbol, {}) or {})
            sector = str(tradability.get("industry") or tradability.get("sector") or "").strip()
            if not sector or sector == "unknown":
                continue
            base_cap = float(risk_decision.max_weight) * max(sector_bucket_limit, 1) * 1.05
            sector_caps[sector] = min(float(risk_decision.gross_exposure_cap), max(base_cap, float(risk_decision.max_weight)))

    position_limits = dict(risk_decision.position_limits)
    for symbol in shortlisted_symbols:
        item = shortlist_by_symbol.get(symbol)
        if item is None:
            continue
        tradability = dict(tradability_snapshot.get(symbol, {}) or {})
        base_limit = float(position_limits.get(symbol, risk_decision.max_weight))
        momentum_strength = float(getattr(item, "metadata", {}).get("momentum_strength", 0.0) or tradability.get("momentum_strength", 0.0) or 0.0)
        fake_breakout_penalty = float(getattr(item, "metadata", {}).get("fake_breakout_penalty", 0.0) or tradability.get("fake_breakout_risk", 0.0) or 0.0)
        liquidity_score = float(tradability.get("liquidity_score", 1.0) or 1.0)
        adjusted_limit = base_limit
        adjusted_limit *= 0.78 + 0.22 * max(momentum_strength, 0.0)
        adjusted_limit *= 1.0 - min(fake_breakout_penalty, 0.80) * 0.35
        adjusted_limit *= 0.75 + 0.25 * max(liquidity_score, 0.20)
        position_limits[symbol] = max(0.04, min(base_limit, adjusted_limit))

    portfolio_constructor = portfolio_constructor_cls()
    portfolio_plan = portfolio_constructor.run(
        {
            "ic_decisions": ic_decisions,
            "macro_verdict": macro_verdict,
            "risk_limits": {
                "gross_exposure_cap": float(risk_decision.gross_exposure_cap),
                "max_weight": float(risk_decision.max_weight),
                "position_limits": position_limits,
                "blocked_symbols": list(risk_decision.blocked_symbols),
                "sector_caps": sector_caps,
            },
            "existing_portfolio": {"current_weights": {}},
            "tradability_snapshot": tradability_snapshot,
        }
    )

    portfolio_decision = PortfolioDecision(
        status=portfolio_plan.status,
        shortlist=shortlist,
        target_exposure=float(portfolio_plan.target_exposure),
        target_gross_exposure=float(portfolio_plan.target_gross_exposure),
        target_net_exposure=float(portfolio_plan.target_net_exposure),
        cash_ratio=float(portfolio_plan.cash_ratio),
        target_weights=dict(portfolio_plan.target_weights),
        target_positions=dict(portfolio_plan.target_positions),
        risk_constraints={
            "risk_decision": risk_decision.to_dict(),
            "tradability_snapshot": tradability_snapshot,
        },
        master_hints={
            "portfolio_master_output": portfolio_master_output.model_dump() if portfolio_master_output is not None and hasattr(portfolio_master_output, "model_dump") else dict(portfolio_master_meta),
        },
        metadata={
            "portfolio_master_meta": dict(portfolio_master_meta),
            "risk_summary": risk_decision.to_dict(),
            "branch_summary_count": len(branch_summaries),
            "funnel_summary": dict(funnel_summary),
            "bayesian_record_count": len(bayesian_records),
            "bayesian_top_symbols": [record.symbol for record in bayesian_records[: min(len(bayesian_records), 10)]],
            "candidate_symbols": list(candidate_symbols),
            "shortlist_symbols": [item.symbol for item in shortlist],
        },
    )
    return PortfolioConstructionState(
        risk_decision=risk_decision,
        ic_decisions=ic_decisions,
        portfolio_plan=portfolio_plan,
        portfolio_decision=portfolio_decision,
    )
