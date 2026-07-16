from __future__ import annotations

from copy import copy
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

from quant_investor.agent_protocol import (
    ActionLabel,
    AgentStatus,
    BayesianDecisionRecord,
    ConfidenceLabel,
    Direction,
    ICDecision,
    PortfolioDecision,
    RiskDecision,
)
from quant_investor.bayesian.calibration import CalibrationStore
from quant_investor.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.branch_contracts import BranchResult
from quant_investor.market.dag.common import _dedupe_texts
from quant_investor.market.dag.assembly import _aggregate_branch_summaries
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
    counterfactual_shortlist: list[Any]
    counterfactual_by_symbol: dict[str, dict[str, Any]]
    counterfactual_bayesian_records: list[BayesianDecisionRecord]


@dataclass
class PortfolioConstructionState:
    risk_decision: RiskDecision
    ic_decisions: list[ICDecision]
    portfolio_plan: Any
    portfolio_decision: PortfolioDecision


def _enum_text(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip().upper()




def _is_codex_handoff_model_roles(model_roles: Any) -> bool:
    metadata = dict(getattr(model_roles, "metadata", {}) or {})
    return bool(
        metadata.get("review_layer_mode") == "codex_handoff"
        or metadata.get("codex_handoff_pending")
        or getattr(model_roles, "resolved_master_model", "") == "codex-handoff"
        or getattr(model_roles, "resolved_branch_model", "") == "codex-handoff"
    )


def _compact_markov_regime_metadata(global_context: Any) -> dict[str, Any]:
    regime_params = getattr(global_context, "regime_params", {}) or {}
    markov_payload = {}
    if isinstance(regime_params, Mapping):
        candidate = regime_params.get("markov", {})
        if isinstance(candidate, Mapping):
            markov_payload = dict(candidate)
    if not markov_payload:
        metadata = getattr(global_context, "metadata", {}) or {}
        if isinstance(metadata, Mapping) and isinstance(metadata.get("markov_regime"), Mapping):
            markov_payload = dict(metadata.get("markov_regime", {}) or {})
    if not markov_payload:
        return {}
    if markov_payload.get("enabled") is False or markov_payload.get("status") == "disabled":
        return {
            "enabled": False,
            "status": "disabled",
        }
    production_eligible = bool(markov_payload.get("production_eligible", False))
    if not production_eligible:
        return {
            "enabled": True,
            "execution_mode": str(markov_payload.get("execution_mode") or "production"),
            "production_eligible": False,
            "status": str(
                markov_payload.get("status")
                or "not_applied_insufficient_market_scope"
            ),
            "regime_scope": str(markov_payload.get("regime_scope") or ""),
            "scope_key": str(markov_payload.get("scope_key") or ""),
        }
    probabilities = markov_payload.get("probabilities", {})
    compact_probabilities = (
        {str(key): float(value) for key, value in probabilities.items()}
        if isinstance(probabilities, Mapping)
        else {}
    )
    return {
        "enabled": True,
        "execution_mode": str(markov_payload.get("execution_mode") or "production"),
        "production_eligible": True,
        "regime_scope": str(markov_payload.get("regime_scope") or ""),
        "scope_key": str(markov_payload.get("scope_key") or ""),
        "dominant_regime": str(markov_payload.get("dominant_regime") or ""),
        "confidence": float(markov_payload.get("confidence", 0.0) or 0.0),
        "transition_risk": float(markov_payload.get("transition_risk", 0.0) or 0.0),
        "probabilities": compact_probabilities,
    }




def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _likelihood_branch_degraded_map(
    *,
    branch_summaries: Mapping[str, Any],
    branch_results: Mapping[str, Any],
) -> dict[str, bool]:
    """Map only the two v14 likelihood branches to explicit degraded state."""

    degraded: dict[str, bool] = {}
    for branch_name in ("quant", "fundamental"):
        summary = branch_summaries.get(branch_name)
        result = branch_results.get(branch_name)
        summary_degraded = bool(
            summary is not None
            and _enum_text(getattr(summary, "status", AgentStatus.SUCCESS))
            in {"DEGRADED", "VETOED"}
        )
        result_metadata = dict(getattr(result, "metadata", {}) or {})
        result_degraded = bool(
            result is not None
            and (
                not bool(getattr(result, "success", True))
                or str(result_metadata.get("degraded_reason") or "").strip()
            )
        )
        degraded[branch_name] = summary_degraded or result_degraded
    return degraded


def _require_exact_canonical_branches(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping")
    expected = set(CANONICAL_BRANCH_ORDER)
    actual = set(payload)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing branches: " + ", ".join(missing))
        if unexpected:
            details.append("unsupported branches: " + ", ".join(unexpected))
        raise ValueError(
            f"{label} must contain exactly the canonical branches "
            f"{list(CANONICAL_BRANCH_ORDER)!r}; {'; '.join(details)}"
        )


def _require_valid_canonical_branch_results(
    branch_results: Mapping[str, Any],
) -> None:
    _require_exact_canonical_branches(
        branch_results,
        label="Bayesian selection branch_results",
    )
    for branch_name in CANONICAL_BRANCH_ORDER:
        result = branch_results[branch_name]
        if not isinstance(result, BranchResult):
            raise ValueError(
                "Bayesian selection branch_results must contain BranchResult objects"
            )
        result.validate()
        if result.branch_name != branch_name:
            raise ValueError(
                "Bayesian selection branch result key/name mismatch: "
                f"{branch_name!r} != {result.branch_name!r}"
            )


def _fundamental_counterfactual_score(
    *,
    symbol: str,
    research_by_symbol: Mapping[str, Mapping[str, Any]],
) -> tuple[float | None, str]:
    fundamental = research_by_symbol.get(symbol, {}).get("fundamental")
    metadata = dict(getattr(fundamental, "metadata", {}) or {})
    runtime = dict(metadata.get("fundamental_research_runtime", {}) or {})
    if runtime.get("blockers") or not runtime.get("request_id"):
        return None, ""
    if bool(runtime.get("counterfactual", False)):
        return _optional_float(runtime.get("counterfactual_adjusted_score")), "with_dossier"
    if bool(runtime.get("applied", False)):
        return _optional_float(metadata.get("deterministic_base_score")), "without_dossier"
    return None, ""


def _counterfactual_branch_results(
    *,
    branch_results: Mapping[str, Any],
    symbol: str,
    fundamental_score: float,
) -> dict[str, Any]:
    _require_valid_canonical_branch_results(branch_results)
    copied = dict(branch_results)
    fundamental = branch_results["fundamental"]
    alternate = copy(fundamental)
    alternate.symbol_scores = dict(getattr(fundamental, "symbol_scores", {}) or {})
    alternate.symbol_scores[symbol] = float(fundamental_score)
    copied["fundamental"] = alternate
    _require_valid_canonical_branch_results(copied)
    return copied


def _build_counterfactual_control_inputs(
    *,
    research_by_symbol: Mapping[str, Mapping[str, Any]],
    counterfactual_by_symbol: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rebuilt: dict[str, dict[str, Any]] = {}
    for symbol, branch_map in research_by_symbol.items():
        _require_exact_canonical_branches(
            branch_map,
            label=f"counterfactual research_by_symbol[{symbol!r}]",
        )
        alternative_map = dict(branch_map)
        alternative = counterfactual_by_symbol.get(symbol)
        fundamental = branch_map.get("fundamental")
        if alternative is not None and fundamental is not None:
            score = float(alternative["fundamental_score"])
            variant = str(alternative["basis"])
            actual_metadata = dict(getattr(fundamental, "metadata", {}) or {})
            deterministic = dict(
                actual_metadata.get("fundamental_deterministic_control_input", {}) or {}
            )
            direction = (
                Direction.BULLISH
                if score >= 0.15
                else Direction.BEARISH if score <= -0.15 else Direction.NEUTRAL
            )
            action = (
                ActionLabel.BUY
                if score >= 0.25
                else ActionLabel.SELL if score <= -0.35 else ActionLabel.HOLD
            )
            alternative_map["fundamental"] = replace(
                fundamental,
                thesis=str(deterministic.get("thesis") or fundamental.thesis),
                status=AgentStatus(
                    str(deterministic.get("status") or fundamental.status.value).lower()
                ),
                final_score=score,
                final_confidence=float(
                    deterministic.get("final_confidence", fundamental.final_confidence)
                ),
                direction=direction,
                action=action,
                confidence_label=ConfidenceLabel(
                    str(
                        deterministic.get("confidence_label")
                        or fundamental.confidence_label.value
                    )
                ),
                investment_risks=list(
                    deterministic.get("investment_risks", fundamental.investment_risks)
                    or []
                ),
                coverage_notes=list(
                    deterministic.get("coverage_notes", fundamental.coverage_notes) or []
                ),
                diagnostic_notes=list(
                    deterministic.get("diagnostic_notes", fundamental.diagnostic_notes)
                    or []
                ),
                metadata={
                    **{
                        key: value
                        for key, value in actual_metadata.items()
                        if key not in {"overlay", "fundamental_research_runtime"}
                    },
                    **(
                        {
                            "fundamental_research_runtime": {
                                **dict(
                                    actual_metadata.get(
                                        "fundamental_research_runtime", {}
                                    )
                                    or {}
                                ),
                                "effective_mode": "counterfactual_replay",
                                "applied": True,
                                "counterfactual": False,
                                "measurement_only": True,
                            }
                        }
                        if variant == "with_dossier"
                        else {}
                    ),
                    "fundamental_research_variant": variant,
                    "fundamental_research_counterfactual_replay": True,
                },
            )
        _require_exact_canonical_branches(
            alternative_map,
            label=f"rebuilt counterfactual research_by_symbol[{symbol!r}]",
        )
        rebuilt[symbol] = alternative_map
    summaries = _aggregate_branch_summaries(rebuilt)
    _require_exact_canonical_branches(
        summaries,
        label="counterfactual branch_summaries",
    )
    return rebuilt, summaries


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
    _require_valid_canonical_branch_results(branch_results)
    _require_exact_canonical_branches(
        branch_summaries,
        label="Bayesian selection branch_summaries",
    )
    if branch_summaries["macro"] != macro_verdict:
        raise ValueError(
            "Bayesian selection macro_verdict must match branch_summaries['macro']"
        )

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
    markov_regime_metadata = _compact_markov_regime_metadata(global_context)
    bayesian_records: list[BayesianDecisionRecord] = []
    counterfactual_bayesian_records: list[BayesianDecisionRecord] = []
    counterfactual_by_symbol: dict[str, dict[str, Any]] = {}
    degraded_map = _likelihood_branch_degraded_map(
        branch_summaries=branch_summaries,
        branch_results=branch_results,
    )
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
        counterfactual_score, counterfactual_basis = _fundamental_counterfactual_score(
            symbol=symbol,
            research_by_symbol=research_by_symbol,
        )
        if counterfactual_score is not None:
            counterfactual_likelihoods = likelihood_mapper.compute_likelihoods(
                branch_results=_counterfactual_branch_results(
                    branch_results=branch_results,
                    symbol=symbol,
                    fundamental_score=counterfactual_score,
                ),
                symbol=symbol,
                candidate_symbols=set(candidate_symbols),
            )
            counterfactual_posterior = posterior_engine.compute_posterior(
                prior,
                counterfactual_likelihoods,
                symbol=symbol,
                company_name=company_name_map.get(symbol, ""),
                regime=global_context.macro_regime or "未知",
                is_degraded=degraded_map,
            )
            counterfactual_by_symbol[symbol] = {
                "basis": counterfactual_basis,
                "fundamental_score": float(counterfactual_score),
                "posterior_action_score": float(
                    counterfactual_posterior.posterior_action_score
                ),
                "posterior_win_rate": float(counterfactual_posterior.posterior_win_rate),
                "posterior_expected_alpha": float(
                    counterfactual_posterior.posterior_expected_alpha
                ),
                "posterior_confidence": float(
                    counterfactual_posterior.posterior_confidence
                ),
                "posterior_edge_after_costs": float(
                    counterfactual_posterior.posterior_edge_after_costs
                ),
                "kill_switch": bool(
                    (counterfactual_posterior.metadata or {}).get("kill_switch", False)
                ),
            }
            counterfactual_bayesian_records.append(
                BayesianDecisionRecord(
                    symbol=symbol,
                    company_name=company_name_map.get(symbol, ""),
                    prior=counterfactual_posterior.prior.to_dict(),
                    likelihoods=counterfactual_posterior.likelihoods.to_dict(),
                    posterior_win_rate=counterfactual_posterior.posterior_win_rate,
                    posterior_expected_alpha=counterfactual_posterior.posterior_expected_alpha,
                    posterior_confidence=counterfactual_posterior.posterior_confidence,
                    posterior_action_score=counterfactual_posterior.posterior_action_score,
                    posterior_edge_after_costs=counterfactual_posterior.posterior_edge_after_costs,
                    posterior_capacity_penalty=counterfactual_posterior.posterior_capacity_penalty,
                    correlation_discount=counterfactual_posterior.correlation_discount,
                    coverage_discount=counterfactual_posterior.coverage_discount,
                    data_quality_penalty=counterfactual_posterior.data_quality_penalty,
                    fallback_penalty=counterfactual_posterior.fallback_penalty,
                    regime_adjustment=counterfactual_posterior.regime_adjustment,
                    evidence_sources=list(counterfactual_posterior.evidence_sources),
                    action_threshold_used=counterfactual_posterior.action_threshold_used,
                    metadata={
                        "fundamental_research_variant": counterfactual_basis,
                        "fundamental_score": float(counterfactual_score),
                    },
                )
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
                    "crowding_penalty": float((posterior.metadata or {}).get("crowding_penalty", 0.0) if isinstance(getattr(posterior, "metadata", {}), Mapping) else 0.0),
                    "history_confidence": float((posterior.metadata or {}).get("history_confidence", 0.0) if isinstance(getattr(posterior, "metadata", {}), Mapping) else 0.0),
                    "calibration_samples": dict((posterior.metadata or {}).get("calibration_samples", {}) or {}) if isinstance(getattr(posterior, "metadata", {}), Mapping) else {},
                    "kill_switch": bool((posterior.metadata or {}).get("kill_switch", False)) if isinstance(getattr(posterior, "metadata", {}), Mapping) else False,
                    "sector": str((posterior.metadata or {}).get("sector", "")) if isinstance(getattr(posterior, "metadata", {}), Mapping) else "",
                    "markov_regime": markov_regime_metadata,
                },
            )
        )
    bayesian_records.sort(key=lambda item: (-float(item.posterior_action_score), item.symbol))
    for index, record in enumerate(bayesian_records, start=1):
        record.rank = index
        record.metadata = dict(record.metadata or {})
        record.metadata["rank"] = index

    counterfactual_record_by_symbol = {
        record.symbol: record for record in counterfactual_bayesian_records
    }
    counterfactual_bayesian_records = [
        counterfactual_record_by_symbol.get(record.symbol)
        or replace(
            record,
            metadata={
                **dict(record.metadata or {}),
                "fundamental_research_variant": "unchanged_no_eligible_dossier",
            },
        )
        for record in bayesian_records
    ]
    counterfactual_bayesian_records.sort(
        key=lambda item: (-float(item.posterior_action_score), item.symbol)
    )
    for index, record in enumerate(counterfactual_bayesian_records, start=1):
        record.rank = index
        record.metadata = dict(record.metadata or {})
        record.metadata["rank"] = index

    counterfactual_shortlist: list[Any] = []
    if counterfactual_by_symbol:
        counterfactual_shortlist = _build_shortlist_from_bayesian_records(
            posterior_results=counterfactual_bayesian_records,
            company_name_map=company_name_map,
            top_k=top_k,
        )
        by_symbol = {item.symbol: item for item in counterfactual_shortlist}
        for record in counterfactual_bayesian_records:
            payload = counterfactual_by_symbol.get(record.symbol)
            if payload is None:
                continue
            item = by_symbol.get(record.symbol)
            payload["rank"] = int(record.rank)
            payload["shortlisted"] = item is not None
            payload["pre_control_suggested_weight"] = (
                float(item.suggested_weight) if item is not None else 0.0
            )

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

    codex_handoff_review = _is_codex_handoff_model_roles(model_roles)
    if (
        bool(getattr(model_roles, "agent_layer_enabled", False))
        and not codex_handoff_review
    ):
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
    elif codex_handoff_review:
        portfolio_master_output = None
        portfolio_master_meta = {
            "status": "codex_handoff_pending",
            "reason": "local_llm_disabled_codex_handoff",
            "review_layer_mode": "codex_handoff",
            "codex_handoff_pending": True,
            "final_conviction": "neutral",
            "final_score": 0.0,
            "confidence": 0.0,
            "top_picks": [],
            "portfolio_narrative": (
                "Portfolio Master advisory packaged for Codex handoff; "
                "deterministic pipeline continued."
            ),
            "risk_adjusted_exposure": float(
                global_context.risk_budget.get("target_exposure", 0.0)
            ),
            "evidence_pack_token_count": int(
                evidence_pack.get("trace_fragments", {})
                .get("budget", {})
                .get("token_count", 0)
                or 0
            ),
            "evidence_pack": evidence_pack,
        }
    else:
        portfolio_master_output = None
        portfolio_master_meta = {
            "status": "disabled",
            "reason": "agent_layer_disabled",
            "final_conviction": "neutral",
            "final_score": 0.0,
            "confidence": 0.0,
            "top_picks": [],
            "portfolio_narrative": "Portfolio Master advisory disabled by no-agent mode.",
            "risk_adjusted_exposure": float(global_context.risk_budget.get("target_exposure", 0.0)),
            "evidence_pack_token_count": int(
                evidence_pack.get("trace_fragments", {}).get("budget", {}).get("token_count", 0) or 0
            ),
        }
    portfolio_master_meta = dict(portfolio_master_meta)
    portfolio_master_meta["advisory_only"] = True
    portfolio_master_meta["deterministic_control_chain_effect"] = "none"
    return BayesianSelectionState(
        bayesian_records=bayesian_records,
        shortlist=shortlist,
        funnel_summary=funnel_summary,
        evidence_pack=evidence_pack,
        portfolio_master_output=portfolio_master_output,
        portfolio_master_meta=portfolio_master_meta,
        portfolio_master_reliability=float(portfolio_master_meta.get("confidence", 0.0) or 0.0),
        counterfactual_shortlist=counterfactual_shortlist,
        counterfactual_by_symbol=counterfactual_by_symbol,
        counterfactual_bayesian_records=counterfactual_bayesian_records,
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
    shortlisted_symbols = [item.symbol for item in shortlist]
    risk_constraints = {
        "gross_exposure_cap": float(global_context.risk_budget.get("target_exposure", 0.55)),
        "max_weight": float(global_context.risk_budget.get("max_single_weight", 0.12)),
        "risk_flags": _dedupe_texts([issue.message for issue in data_quality_issues[:8]]),
        "data_quality_issue_count": len(data_quality_issues),
    }
    turnover_cap_present = "turnover_cap" in dict(global_context.risk_budget or {})
    turnover_cap = _optional_float(global_context.risk_budget.get("turnover_cap"))
    if turnover_cap_present:
        risk_constraints["turnover_cap"] = turnover_cap
    risk_decision = risk_guard.run(
        {
            "branch_verdicts": branch_summaries,
            "macro_verdict": macro_verdict,
            "portfolio_state": {
                "candidate_symbols": shortlisted_symbols,
                "current_weights": {},
            },
            "constraints": risk_constraints,
        }
    )

    ic_coordinator = ic_coordinator_cls()
    shortlist_by_symbol = {item.symbol: item for item in shortlist}
    ic_decisions: list[ICDecision] = []
    for symbol in shortlisted_symbols:
        advisory_ic_hint = ic_hints_by_symbol.get(symbol, {})
        decision = ic_coordinator.run(
            {
                "branch_verdicts": research_by_symbol[symbol],
                "risk_decision": risk_decision,
                "ic_hints": {},
            }
        )
        decision = attach_symbol_to_ic_decision_fn(
            decision,
            symbol=symbol,
            risk_decision=risk_decision,
            current_weight=0.0,
            tradability_info=tradability_snapshot[symbol],
            ic_hint=advisory_ic_hint,
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

    portfolio_risk_limits = {
        "gross_exposure_cap": float(risk_decision.gross_exposure_cap),
        "max_weight": float(risk_decision.max_weight),
        "position_limits": position_limits,
        "blocked_symbols": list(risk_decision.blocked_symbols),
        "sector_caps": sector_caps,
        "turnover_cap": turnover_cap,
    }

    portfolio_constructor = portfolio_constructor_cls()
    portfolio_plan = portfolio_constructor.run(
        {
            "ic_decisions": ic_decisions,
            "macro_verdict": macro_verdict,
            "risk_limits": portfolio_risk_limits,
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
