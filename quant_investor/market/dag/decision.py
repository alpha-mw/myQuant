from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from quant_investor.agent_protocol import (
    AgentStatus,
    BayesianDecisionRecord,
    ICDecision,
    PortfolioDecision,
    RiskDecision,
)
from quant_investor.bayesian.calibration import CalibrationStore
from quant_investor.config import config
from quant_investor.governance import replay_v13_1
from quant_investor.market.dag.common import _dedupe_texts
from quant_investor.market.dag.evidence import _build_master_evidence_pack
from quant_investor.market.dag.shortlist import _build_shortlist_from_bayesian_records
from quant_investor.market.dag.theme_context import (
    build_theme_portfolio_constraints,
    build_theme_risk_constraints,
    extract_symbol_theme_metadata,
)
from quant_investor.reporting.run_artifacts import build_funnel_summary
from quant_investor.themes.protocol_v2 import (
    persist_theme_formal_reconciliation_artifact,
    reconcile_theme_protocol_v2,
)


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


def _enum_text(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip().upper()


def _post_control_theme_reconciliation(
    *,
    global_context: Any,
    shortlist: list[Any],
    tradability_snapshot: Mapping[str, Mapping[str, Any]],
    risk_decision: RiskDecision,
    portfolio_plan: Any,
) -> dict[str, Any]:
    metadata = getattr(global_context, "metadata", {})
    rotation = (
        metadata.get("theme_rotation", {})
        if isinstance(metadata, Mapping)
        else {}
    )
    if not isinstance(rotation, dict):
        return {"status": "unavailable", "formal_pool": []}
    protocol = rotation.get("protocol_v2", {})
    if not isinstance(protocol, dict):
        return {"status": "unavailable", "formal_pool": []}
    if (
        protocol.get("formal_enabled") is not True
        or protocol.get("formal_kill_switch") is True
    ):
        artifact = {
            "schema_version": "theme_formal_reconciliation.v1",
            "status": "observer_only",
            "formal_pool": [],
            "formal_symbols": [],
            "blockers": [
                "formal_kill_switch_active"
                if protocol.get("formal_kill_switch") is True
                else "formal_switch_disabled"
            ],
        }
        rotation["formal_reconciliation"] = artifact
        return artifact

    activation_blockers: list[str] = []
    if not bool(getattr(config, "THEME_PORTFOLIO_CAP_ENABLED", False)):
        activation_blockers.append("theme_portfolio_cap_required_for_formal")
    if not bool(
        getattr(config, "THEME_FORMAL_RECONCILIATION_PERSIST_ENABLED", False)
    ):
        activation_blockers.append("formal_reconciliation_persistence_disabled")
    activation_blockers.extend(
        _theme_plan_cap_proof_blockers(
            portfolio_plan=portfolio_plan,
            current_protocol_hash=str(protocol.get("protocol_hash") or ""),
        )
    )
    activation_blockers.extend(
        _theme_joint_manifest_blockers(
            current_protocol_hash=str(protocol.get("protocol_hash") or "")
        )
    )
    if activation_blockers:
        artifact = {
            "schema_version": "theme_formal_reconciliation.v1",
            "status": "blocked",
            "formal_pool": [],
            "formal_symbols": [],
            "blockers": activation_blockers,
        }
        persistence_status = {
            "status": (
                "disabled"
                if "formal_reconciliation_persistence_disabled"
                in activation_blockers
                else "not_attempted"
            ),
            "path": "",
            "readback_verified": False,
        }
        rotation["formal_reconciliation"] = artifact
        rotation["formal_reconciliation_persistence"] = persistence_status
        if isinstance(metadata, dict):
            metadata["theme_formal_reconciliation"] = artifact
            metadata["theme_formal_reconciliation_persistence"] = (
                persistence_status
            )
        return artifact

    shortlist_by_symbol = {
        str(getattr(item, "symbol", "")).strip().upper(): item
        for item in shortlist
        if str(getattr(item, "symbol", "")).strip()
    }
    target_weights = dict(getattr(portfolio_plan, "target_weights", {}) or {})
    blocked_symbols = {
        str(symbol).strip().upper()
        for symbol in list(getattr(risk_decision, "blocked_symbols", []) or [])
    }
    rejected_symbols = {
        str(symbol).strip().upper()
        for symbol in list(getattr(portfolio_plan, "rejected_symbols", []) or [])
    }
    plan_blocked_symbols = {
        str(symbol).strip().upper()
        for symbol in list(getattr(portfolio_plan, "blocked_symbols", []) or [])
    }
    risk_allows_buy = (
        _enum_text(getattr(risk_decision, "status", "")) == "SUCCESS"
        and not bool(getattr(risk_decision, "hard_veto", False))
        and not bool(getattr(risk_decision, "veto", False))
        and _enum_text(getattr(risk_decision, "action_cap", "")) == "BUY"
    )
    outcomes: dict[str, dict[str, Any]] = {}
    candidate_symbols = sorted(
        set(shortlist_by_symbol)
        | {
            str(symbol).strip().upper()
            for symbol in dict(
                rotation.get("symbol_theme_membership_details", {}) or {}
            )
        }
    )
    for symbol in candidate_symbols:
        tradability = dict(tradability_snapshot.get(symbol, {}) or {})
        shortlist_item = shortlist_by_symbol.get(symbol)
        shortlist_metadata = dict(
            getattr(shortlist_item, "metadata", {}) or {}
        )
        try:
            liquidity_score = float(tradability.get("liquidity_score"))
        except (TypeError, ValueError):
            liquidity_score = 0.0
        try:
            data_quality_issue_count = int(
                tradability.get("data_quality_issue_count")
            )
        except (TypeError, ValueError):
            data_quality_issue_count = -1
        positive_edge = False
        if shortlist_item is not None:
            try:
                edge_after_costs = float(
                    shortlist_metadata.get("posterior_edge_after_costs")
                )
            except (TypeError, ValueError):
                edge_after_costs = 0.0
            positive_edge = (
                _enum_text(getattr(shortlist_item, "action", "")) == "BUY"
                and edge_after_costs > 0.0
            )
        portfolio_weight = float(target_weights.get(symbol, 0.0) or 0.0)
        outcomes[symbol] = {
            "data_pass": (
                tradability.get("tradable") is True
                and bool(str(tradability.get("source_path") or "").strip())
                and data_quality_issue_count == 0
            ),
            "tradability_pass": tradability.get("tradable") is True,
            "liquidity_pass": liquidity_score > 0.0,
            "positive_edge_or_buy": positive_edge,
            "risk_guard_pass": (
                risk_allows_buy and symbol not in blocked_symbols
            ),
            "portfolio_constructor_pass": (
                _enum_text(getattr(portfolio_plan, "status", "")) == "SUCCESS"
                and portfolio_weight > 0.0
                and symbol not in rejected_symbols
                and symbol not in plan_blocked_symbols
            ),
            "portfolio_weight": portfolio_weight,
            "decision_id": (
                f"{str(protocol.get('protocol_hash') or '')[:16]}:"
                f"{str(protocol.get('as_of') or rotation.get('as_of') or '')}:"
                f"{symbol}"
            ),
        }
    try:
        artifact = reconcile_theme_protocol_v2(
            prequalification=protocol,
            symbol_membership_details=dict(
                rotation.get("symbol_theme_membership_details", {}) or {}
            ),
            symbol_outcomes=outcomes,
            as_of=str(protocol.get("as_of") or rotation.get("as_of") or ""),
            expected_protocol_hash=str(protocol.get("protocol_hash") or ""),
            run_id=str(getattr(global_context, "universe_hash", "") or ""),
        )
    except (TypeError, ValueError) as exc:
        artifact = {
            "schema_version": "theme_formal_reconciliation.v1",
            "status": "blocked",
            "formal_pool": [],
            "formal_symbols": [],
            "blockers": [f"post_control_reconciliation_blocked:{exc}"],
        }
    persistence_status: dict[str, Any] = {
        "status": "not_attempted",
        "path": "",
        "readback_verified": False,
    }
    if artifact.get("status") in {"formal", "valid_empty"}:
        if not bool(
            getattr(config, "THEME_FORMAL_RECONCILIATION_PERSIST_ENABLED", False)
        ):
            persistence_status["status"] = "disabled"
            artifact = {
                "schema_version": "theme_formal_reconciliation.v1",
                "status": "blocked",
                "formal_pool": [],
                "formal_symbols": [],
                "blockers": ["formal_reconciliation_persistence_disabled"],
            }
        else:
            try:
                persistence_status = persist_theme_formal_reconciliation_artifact(
                    str(
                        getattr(
                            config,
                            "THEME_FORMAL_RECONCILIATION_DIR",
                            "private/theme_reconciliation",
                        )
                    ),
                    artifact,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                persistence_status = {
                    "status": "error",
                    "path": "",
                    "readback_verified": False,
                    "error": str(exc),
                }
                artifact = {
                    "schema_version": "theme_formal_reconciliation.v1",
                    "status": "blocked",
                    "formal_pool": [],
                    "formal_symbols": [],
                    "blockers": [
                        f"formal_reconciliation_persistence_blocked:{exc}"
                    ],
                }
    rotation["formal_reconciliation"] = artifact
    rotation["formal_reconciliation_persistence"] = persistence_status
    if isinstance(metadata, dict):
        metadata["theme_formal_reconciliation"] = artifact
        metadata["theme_formal_reconciliation_persistence"] = persistence_status
    return artifact


def _theme_plan_cap_proof_blockers(
    *,
    portfolio_plan: Any,
    current_protocol_hash: str,
) -> list[str]:
    plan_metadata = getattr(portfolio_plan, "metadata", {})
    if not isinstance(plan_metadata, Mapping):
        return ["theme_portfolio_cap_execution_proof_missing"]
    blockers: list[str] = []
    if plan_metadata.get("theme_portfolio_cap_enabled") is not True:
        blockers.append("theme_portfolio_cap_not_applied")
    lane = plan_metadata.get("theme_tactical_lane")
    if not isinstance(lane, Mapping):
        blockers.append("theme_tactical_lane_execution_proof_missing")
        lane = {}
    status = str(lane.get("status") or "")
    if status not in {"active", "closed_by_markov"}:
        blockers.append("theme_tactical_lane_status_invalid")
    if (
        not current_protocol_hash
        or str(lane.get("protocol_hash") or "") != current_protocol_hash
    ):
        blockers.append("theme_tactical_lane_protocol_hash_mismatch")
    if lane.get("applied") is not True:
        blockers.append("theme_tactical_lane_not_applied")
    diagnostic_values = [
        *list(plan_metadata.get("theme_portfolio_diagnostic_notes") or []),
        *list(getattr(portfolio_plan, "construction_notes", []) or []),
        *list(getattr(portfolio_plan, "execution_notes", []) or []),
    ]
    if status == "blocked_malformed" or any(
        "malformed" in str(note).strip().lower()
        for note in diagnostic_values
    ):
        blockers.append("theme_portfolio_cap_malformed_diagnostic")
    return list(dict.fromkeys(blockers))


def _theme_joint_manifest_blockers(*, current_protocol_hash: str) -> list[str]:
    if replay_v13_1.CANONICAL_JOINT_REPLAY_PRODUCER_AVAILABLE is not True:
        return ["canonical_joint_replay_producer_not_implemented"]
    path_text = str(
        getattr(config, "THEME_V2_JOINT_MANIFEST_PATH", "") or ""
    ).strip()
    expected_sha = str(
        getattr(config, "THEME_V2_EXPECTED_JOINT_MANIFEST_SHA256", "") or ""
    ).strip().lower()
    blockers: list[str] = []
    if not path_text:
        return ["theme_joint_manifest_path_missing"]
    if len(expected_sha) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha
    ):
        return ["theme_joint_manifest_expected_sha256_missing"]
    try:
        verification = replay_v13_1.verify_joint_replay_manifest(
            path_text,
            expected_artifact_sha256=expected_sha,
            expected_theme_protocol_hash=current_protocol_hash,
        )
    except (OSError, TypeError, ValueError) as exc:
        return [f"theme_joint_manifest_verification_error:{exc}"]
    blockers.extend(
        str(blocker)
        for blocker in list(verification.get("blockers") or [])
        if str(blocker)
    )
    if verification.get("ready") is not True:
        blockers.append("theme_joint_manifest_canonical_verification_failed")
    return list(dict.fromkeys(blockers))


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


def _compact_theme_pool_symbol_metadata(
    *,
    funnel_output: Any,
    symbol: str,
) -> dict[str, Any]:
    funnel_metadata = getattr(funnel_output, "funnel_metadata", {}) or {}
    if not isinstance(funnel_metadata, Mapping):
        return {}
    theme_pool = funnel_metadata.get("theme_pool", {})
    if not isinstance(theme_pool, Mapping) or not theme_pool:
        return {}
    symbol_map = theme_pool.get("symbols", {})
    policy = theme_pool.get("policy", {})
    symbol_payload = (
        dict(symbol_map.get(symbol, {}) or {})
        if isinstance(symbol_map, Mapping)
        else {}
    )
    policy_payload = policy if isinstance(policy, Mapping) else {}
    risk_flags = symbol_payload.get("risk_flags", [])
    if isinstance(risk_flags, (str, bytes)):
        compact_risk_flags: list[str] = []
    else:
        try:
            compact_risk_flags = [
                str(item)
                for item in list(risk_flags or [])
                if str(item or "").strip()
            ]
        except TypeError:
            compact_risk_flags = []
    return {
        "admitted": bool(symbol_payload.get("admitted", False)),
        "source": str(symbol_payload.get("source") or "none"),
        "primary_theme_id": str(symbol_payload.get("primary_theme_id") or ""),
        "primary_theme_name": str(symbol_payload.get("primary_theme_name") or ""),
        "theme_score": float(symbol_payload.get("theme_score", 0.0) or 0.0),
        "symbol_theme_score": float(symbol_payload.get("symbol_theme_score", 0.0) or 0.0),
        "theme_pool_score": float(symbol_payload.get("theme_pool_score", 0.0) or 0.0),
        "bucket": str(symbol_payload.get("bucket") or "none"),
        "phase": str(symbol_payload.get("phase") or ""),
        "risk_flags": compact_risk_flags,
        "candidate_intent": str(symbol_payload.get("candidate_intent") or ""),
        "score_penalty": float(symbol_payload.get("score_penalty", 0.0) or 0.0),
        "theme_forced_admission": bool(symbol_payload.get("theme_forced_admission", False)),
        "theme_policy_regime": str(
            symbol_payload.get("theme_policy_regime")
            or policy_payload.get("regime")
            or ""
        ),
        "theme_pool_reason": str(symbol_payload.get("theme_pool_reason") or ""),
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    markov_regime_metadata = _compact_markov_regime_metadata(global_context)
    bayesian_records: list[BayesianDecisionRecord] = []
    degraded_map = {
        "quant": False,
        "fundamental": False,
        "intelligence": False,
        "macro": False,
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
                    "theme_rotation": extract_symbol_theme_metadata(
                        global_context=global_context,
                        symbol=symbol,
                    ),
                    "theme_pool": _compact_theme_pool_symbol_metadata(
                        funnel_output=funnel_output,
                        symbol=symbol,
                    ),
                    "markov_regime": markov_regime_metadata,
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
    theme_risk_constraints = build_theme_risk_constraints(
        global_context=global_context,
        symbols=shortlisted_symbols,
        enabled=config.THEME_RISK_GUARD_ENABLED,
        overextended_gross_cap=config.THEME_RISK_OVEREXTENDED_GROSS_CAP,
        overextended_max_weight=config.THEME_RISK_OVEREXTENDED_MAX_WEIGHT,
        distribution_gross_cap=config.THEME_RISK_DISTRIBUTION_GROSS_CAP,
        distribution_max_weight=config.THEME_RISK_DISTRIBUTION_MAX_WEIGHT,
        fake_breakout_max_weight=config.THEME_RISK_FAKE_BREAKOUT_MAX_WEIGHT,
    )
    risk_constraints.update(theme_risk_constraints)
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

    theme_portfolio_constraints = build_theme_portfolio_constraints(
        global_context=global_context,
        symbols=shortlisted_symbols,
        enabled=config.THEME_PORTFOLIO_CAP_ENABLED,
        max_theme_exposure=config.THEME_PORTFOLIO_MAX_THEME_EXPOSURE,
        overextended_max_theme_exposure=config.THEME_PORTFOLIO_OVEREXTENDED_MAX_THEME_EXPOSURE,
        distribution_max_theme_exposure=config.THEME_PORTFOLIO_DISTRIBUTION_MAX_THEME_EXPOSURE,
    )
    portfolio_risk_limits = {
        "gross_exposure_cap": float(risk_decision.gross_exposure_cap),
        "max_weight": float(risk_decision.max_weight),
        "position_limits": position_limits,
        "blocked_symbols": list(risk_decision.blocked_symbols),
        "sector_caps": sector_caps,
        "turnover_cap": turnover_cap,
        "theme_portfolio_cap_enabled": theme_portfolio_constraints["theme_portfolio_cap_enabled"],
        "theme_exposure_map": theme_portfolio_constraints["theme_exposure_map"],
        "theme_caps": theme_portfolio_constraints["theme_caps"],
        "theme_names": theme_portfolio_constraints["theme_names"],
        "theme_phases": theme_portfolio_constraints["theme_phases"],
        "theme_tactical_lane": theme_portfolio_constraints["theme_tactical_lane"],
        "theme_portfolio_diagnostic_notes": theme_portfolio_constraints[
            "diagnostic_notes"
        ],
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
    theme_formal_reconciliation = _post_control_theme_reconciliation(
        global_context=global_context,
        shortlist=shortlist,
        tradability_snapshot=tradability_snapshot,
        risk_decision=risk_decision,
        portfolio_plan=portfolio_plan,
    )
    _apply_theme_formal_reconciliation_to_plan(
        portfolio_plan=portfolio_plan,
        reconciliation=theme_formal_reconciliation,
        global_context=global_context,
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
            "theme_formal_reconciliation": theme_formal_reconciliation,
            "theme_formal_reconciliation_persistence": dict(
                getattr(global_context, "metadata", {}).get(
                    "theme_formal_reconciliation_persistence", {}
                )
                or {}
            ),
        },
    )
    return PortfolioConstructionState(
        risk_decision=risk_decision,
        ic_decisions=ic_decisions,
        portfolio_plan=portfolio_plan,
        portfolio_decision=portfolio_decision,
    )


def _apply_theme_formal_reconciliation_to_plan(
    *,
    portfolio_plan: Any,
    reconciliation: Mapping[str, Any],
    global_context: Any,
) -> None:
    metadata = getattr(global_context, "metadata", {})
    rotation = metadata.get("theme_rotation", {}) if isinstance(metadata, Mapping) else {}
    protocol = rotation.get("protocol_v2", {}) if isinstance(rotation, Mapping) else {}
    formal_active = bool(
        isinstance(protocol, Mapping)
        and protocol.get("formal_enabled") is True
        and protocol.get("formal_kill_switch") is not True
    )
    if not formal_active:
        return
    current_weights = dict(getattr(portfolio_plan, "target_weights", {}) or {})
    current_positions = dict(getattr(portfolio_plan, "target_positions", {}) or {})
    allowed_symbols = {
        str(symbol)
        for symbol in list(reconciliation.get("formal_symbols") or [])
        if str(symbol)
    }
    valid_status = str(reconciliation.get("status") or "") == "formal"
    kept_weights = (
        {
            symbol: weight
            for symbol, weight in current_weights.items()
            if symbol in allowed_symbols
        }
        if valid_status
        else {}
    )
    removed = sorted(set(current_weights) - set(kept_weights))
    if not removed and valid_status:
        return
    kept_positions = {
        symbol: value
        for symbol, value in current_positions.items()
        if symbol in kept_weights
    }
    target_gross = round(sum(abs(float(weight)) for weight in kept_weights.values()), 6)
    target_net = round(sum(float(weight) for weight in kept_weights.values()), 6)
    portfolio_plan.target_weights = kept_weights
    portfolio_plan.target_positions = kept_positions
    portfolio_plan.target_exposure = target_gross
    portfolio_plan.target_gross_exposure = target_gross
    portfolio_plan.target_net_exposure = target_net
    portfolio_plan.cash_ratio = max(0.0, min(1.0, 1.0 - target_net))
    if not kept_weights:
        portfolio_plan.status = AgentStatus.VETOED
    portfolio_plan.blocked_symbols = sorted(
        set(getattr(portfolio_plan, "blocked_symbols", []) or []) | set(removed)
    )
    portfolio_plan.rejected_symbols = sorted(
        set(getattr(portfolio_plan, "rejected_symbols", []) or []) | set(removed)
    )
    plan_metadata = getattr(portfolio_plan, "metadata", None)
    if isinstance(plan_metadata, dict):
        plan_metadata["theme_formal_fail_closed"] = True
        plan_metadata["theme_formal_removed_symbols"] = removed
        plan_metadata["theme_formal_reconciliation_status"] = str(
            reconciliation.get("status") or "blocked"
        )
