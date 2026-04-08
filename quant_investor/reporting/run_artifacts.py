"""
Structured run artifact builders for model roles, execution trace, and what-if plans.
"""

from __future__ import annotations

from typing import Any, Mapping

from quant_investor.agent_protocol import (
    ExecutionTrace,
    ExecutionTraceStep,
    ModelRoleMetadata,
    WhatIfPlan,
    WhatIfScenario,
)


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if hasattr(value, "to_dict"):
        payload = value.to_dict()
        if isinstance(payload, Mapping):
            return {str(key): item for key, item in payload.items()}
    if hasattr(value, "__dict__"):
        return {
            str(key): item
            for key, item in value.__dict__.items()
            if not key.startswith("_")
        }
    return {}


def build_model_role_metadata(
    *,
    branch_model: str = "",
    master_model: str = "",
    agent_fallback_model: str = "",
    master_fallback_model: str = "",
    resolved_branch_model: str = "",
    resolved_master_model: str = "",
    master_reasoning_effort: str = "high",
    branch_provider: str = "",
    master_provider: str = "",
    branch_timeout: float = 0.0,
    master_timeout: float = 0.0,
    agent_layer_enabled: bool = False,
    branch_fallback_used: bool = False,
    master_fallback_used: bool = False,
    branch_fallback_reason: str = "",
    master_fallback_reason: str = "",
    universe_key: str = "",
    universe_size: int = 0,
    universe_hash: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> ModelRoleMetadata:
    payload = dict(metadata or {})
    payload.setdefault("branch_role", "per-stock analysis")
    payload.setdefault(
        "master_role",
        "master synthesis / portfolio-level judgment before deterministic risk and sizing",
    )
    return ModelRoleMetadata(
        branch_model=str(branch_model or ""),
        master_model=str(master_model or ""),
        agent_fallback_model=str(agent_fallback_model or ""),
        master_fallback_model=str(master_fallback_model or ""),
        resolved_branch_model=str(resolved_branch_model or ""),
        resolved_master_model=str(resolved_master_model or ""),
        master_reasoning_effort=str(master_reasoning_effort or "").strip() or "high",
        branch_provider=str(branch_provider or ""),
        master_provider=str(master_provider or ""),
        branch_timeout=float(branch_timeout or 0.0),
        master_timeout=float(master_timeout or 0.0),
        agent_layer_enabled=bool(agent_layer_enabled),
        branch_fallback_used=bool(branch_fallback_used),
        master_fallback_used=bool(master_fallback_used),
        branch_fallback_reason=str(branch_fallback_reason or ""),
        master_fallback_reason=str(master_fallback_reason or ""),
        universe_key=str(universe_key or ""),
        universe_size=int(universe_size or 0),
        universe_hash=str(universe_hash or ""),
        metadata=payload,
    )


def build_what_if_plan(
    *,
    portfolio_plan: Mapping[str, Any] | Any,
    market_summary: Mapping[str, Any] | Any,
    model_roles: Mapping[str, Any] | ModelRoleMetadata | None = None,
    candidate_count: int | None = None,
    selected_count: int | None = None,
) -> WhatIfPlan:
    portfolio = _as_mapping(portfolio_plan)
    market = _as_mapping(market_summary)
    roles = _as_mapping(model_roles) if model_roles is not None else {}

    total_candidate_count = int(candidate_count if candidate_count is not None else market.get("candidate_count", 0))
    total_selected_count = int(selected_count if selected_count is not None else portfolio.get("selected_count", 0))
    target_exposure = float(portfolio.get("target_exposure", 0.0))
    max_single_weight = float(portfolio.get("max_single_weight", 0.0))
    reliability = float(portfolio.get("reliability", 0.0))
    macro_score = float(market.get("macro_score", market.get("avg_macro_score", 0.0)))
    style_bias = str(portfolio.get("style_bias", "均衡"))

    scenarios = [
        WhatIfScenario(
            scenario_name="macro_turns_weaker",
            trigger="宏观评分转弱，或宏观相关风险指标持续走低",
            monitoring_indicators=[
                f"macro_score <= -0.25 (current={macro_score:+.2f})",
                "volatility_percentile 上行",
                "liquidity_score 下行",
            ],
            action="降低整体风险偏好，优先保留现金与高置信度持仓。",
            position_adjustment_rule=(
                f"若宏观持续转弱，将目标总暴露从 {target_exposure:.1%} 下调至 "
                f"max({target_exposure:.1%} * 0.8, 25%)，并优先剔除低置信度仓位。"
            ),
            rerun_full_market_daily_path=True,
            metadata={
                "style_bias": style_bias,
                "reliability": reliability,
                "model_roles": {
                    "branch_model": roles.get("branch_model", ""),
                    "master_model": roles.get("master_model", ""),
                },
            },
        ),
        WhatIfScenario(
            scenario_name="single_name_stop_loss_or_reversal",
            trigger="单票跌破止损，或分支/主协调信号出现明显反转",
            monitoring_indicators=[
                "当前价 vs 止损价",
                "branch score 方向变化",
                "ic_hints_by_symbol 中的 action/confidence 变化",
            ],
            action="对触发标的执行减仓或退出，保留组合级风险约束不变。",
            position_adjustment_rule=(
                f"若单票触发，先将其目标权重压缩至 0 或不超过 {max_single_weight:.1%}，"
                "并保持组合其余仓位按 deterministic 规则重算。"
            ),
            rerun_full_market_daily_path=False,
            metadata={
                "selected_count": total_selected_count,
                "confidence_threshold": max(reliability, 0.4),
            },
        ),
        WhatIfScenario(
            scenario_name="candidate_set_decays",
            trigger="候选集衰减，或可执行标的数量明显下降",
            monitoring_indicators=[
                f"selected_count < max(candidate_count * 0.7, 1) (current={total_selected_count})",
                f"candidate_count (current={total_candidate_count})",
                "branch_positive_count 分布",
            ],
            action="收紧筛选条件并重新扫描全市场候选，必要时重跑 daily path。",
            position_adjustment_rule=(
                "若候选数量持续下降，则将组合仓位下调到更保守区间，"
                "并把剩余仓位集中到高置信度、低冲突标的。"
            ),
            rerun_full_market_daily_path=True,
            metadata={
                "candidate_count": total_candidate_count,
                "selected_count": total_selected_count,
                "portfolio_style_bias": style_bias,
            },
        ),
    ]

    return WhatIfPlan(
        scenarios=scenarios,
        metadata={
            "candidate_count": total_candidate_count,
            "selected_count": total_selected_count,
            "target_exposure": target_exposure,
            "max_single_weight": max_single_weight,
            "reliability": reliability,
        },
        generated_by="deterministic",
    )


def build_execution_trace(
    *,
    model_roles: ModelRoleMetadata | Mapping[str, Any],
    download_stage: Mapping[str, Any] | None = None,
    analysis_meta: Mapping[str, Any] | None = None,
    portfolio_plan: Mapping[str, Any] | Any | None = None,
    persistence_note: str = "",
) -> ExecutionTrace:
    role_payload = _as_mapping(model_roles)
    model_role_metadata = (
        model_roles
        if isinstance(model_roles, ModelRoleMetadata)
        else build_model_role_metadata(
            branch_model=str(role_payload.get("branch_model", "")),
            master_model=str(role_payload.get("master_model", "")),
            agent_fallback_model=str(role_payload.get("agent_fallback_model", "")),
            master_fallback_model=str(role_payload.get("master_fallback_model", "")),
            resolved_branch_model=str(role_payload.get("resolved_branch_model", "")),
            resolved_master_model=str(role_payload.get("resolved_master_model", "")),
            master_reasoning_effort=str(role_payload.get("master_reasoning_effort", "")),
            branch_provider=str(role_payload.get("branch_provider", "")),
            master_provider=str(role_payload.get("master_provider", "")),
            branch_timeout=float(role_payload.get("branch_timeout", 0.0)),
            master_timeout=float(role_payload.get("master_timeout", 0.0)),
            agent_layer_enabled=bool(role_payload.get("agent_layer_enabled", False)),
            branch_fallback_used=bool(role_payload.get("branch_fallback_used", False)),
            master_fallback_used=bool(role_payload.get("master_fallback_used", False)),
            branch_fallback_reason=str(role_payload.get("branch_fallback_reason", "")),
            master_fallback_reason=str(role_payload.get("master_fallback_reason", "")),
            universe_key=str(role_payload.get("universe_key", "")),
            universe_size=int(role_payload.get("universe_size", 0)),
            universe_hash=str(role_payload.get("universe_hash", "")),
            metadata=role_payload.get("metadata", {}),
        )
    )
    analysis = _as_mapping(analysis_meta or {})
    portfolio = _as_mapping(portfolio_plan or {})
    download = _as_mapping(download_stage or {})
    resolver_payload = _as_mapping(download.get("resolver"))
    if resolver_payload.get("after"):
        resolver_payload = _as_mapping(resolver_payload.get("after"))
    elif resolver_payload.get("before"):
        resolver_payload = _as_mapping(resolver_payload.get("before"))
    if not resolver_payload:
        resolver_payload = _as_mapping(
            _as_mapping(download.get("completeness_after")).get("resolver")
            or _as_mapping(download.get("completeness_before")).get("resolver")
            or analysis.get("resolver")
            or portfolio.get("resolver")
        )

    batch_count = int(analysis.get("batch_count", 0))
    category_count = int(analysis.get("category_count", 0))
    total_stocks = int(analysis.get("total_stocks", 0))
    researchable_count = int(analysis.get("researchable_count", 0))
    quarantined_count = int(analysis.get("quarantined_count", 0))
    quarantined_symbols = list(analysis.get("quarantined_symbols", []) or [])
    data_quality_issue_count = int(analysis.get("data_quality_issue_count", 0))
    candidate_count = int(analysis.get("candidate_count", 0))
    shortlistable_count = int(analysis.get("shortlistable_count", candidate_count))
    shortlist_count = int(analysis.get("shortlist_count", shortlistable_count))
    final_selected_count = int(portfolio.get("selected_count", analysis.get("final_selected_count", 0)))
    target_exposure = float(portfolio.get("target_exposure", 0.0))
    max_single_weight = float(portfolio.get("max_single_weight", 0.0))
    evidence_pack_token_count = int(analysis.get("evidence_pack_token_count", 0))
    evidence_pack_field_limit = int(analysis.get("evidence_pack_field_limit", 0))
    evidence_pack_shortlist_limit = int(analysis.get("evidence_pack_shortlist_limit", 0))
    company_name_coverage = int(analysis.get("company_name_coverage", 0))
    bayesian_record_count = int(analysis.get("bayesian_record_count", 0))
    reliability = float(portfolio.get("reliability", 0.0))
    funnel_summary = _as_mapping(analysis.get("funnel_summary"))

    steps = [
        ExecutionTraceStep(
            stage="data_check_download",
            role="system",
            model="deterministic",
            success=str(download.get("status", "")).lower() not in {"failed", "error"},
            conclusion=(
                f"数据阶段完成：{download.get('status', 'unknown')}，"
                f"覆盖状态为 {download.get('reason', 'n/a')}。"
            ),
            parameters={
                "market": download.get("market", analysis.get("market", "")),
                "categories": download.get("categories", []),
            },
            fallback_reason=str(download.get("warning", "")),
            metadata={
                "completeness_before": download.get("completeness_before"),
                "completeness_after": download.get("completeness_after"),
                "resolver": resolver_payload,
                "quarantined_count": quarantined_count,
            },
        ),
        ExecutionTraceStep(
            stage="global_context_build",
            role="system",
            model="deterministic",
            success=True,
            conclusion=(
                f"GlobalContext 已完成一次性构建，覆盖 {total_stocks or 0} 只股票，"
                f"其中可研究 {researchable_count or 0} 只，隔离 {quarantined_count or 0} 只。"
            ),
            parameters={
                "universe_key": model_role_metadata.universe_key,
                "universe_size": model_role_metadata.universe_size,
                "data_quality_issue_count": data_quality_issue_count,
            },
            metadata={
                "global_context": analysis.get("global_context", {}),
                "resolver": resolver_payload,
            },
        ),
        ExecutionTraceStep(
            stage="candidate_review",
            role="branch",
            model=model_role_metadata.branch_model,
            success=bool(batch_count or total_stocks),
            conclusion=(
                f"完成 {category_count or len(analysis.get('categories', []))} 个类别、"
                f"{batch_count or 0} 个执行切片的候选复审，覆盖候选 {shortlistable_count or 0} 只。"
            ),
            parameters={
                "agent_layer_enabled": model_role_metadata.agent_layer_enabled,
                "branch_timeout": model_role_metadata.branch_timeout,
                "master_reasoning_effort": model_role_metadata.master_reasoning_effort,
                "branch_fallback_used": model_role_metadata.branch_fallback_used,
                "master_fallback_used": model_role_metadata.master_fallback_used,
                "candidate_review_count": shortlistable_count,
            },
            fallback_reason=_join_texts(analysis.get("fallback_reasons", [])),
            metadata={
                "symbol_research_packets": analysis.get("symbol_research_packets", {}),
                "model_role_metadata": model_role_metadata.to_dict(),
                "quarantined_count": quarantined_count,
                "quarantined_symbols": quarantined_symbols[:32],
            },
        ),
        ExecutionTraceStep(
            stage="deterministic_funnel",
            role="deterministic",
            model="quant+kline gates",
            success=bool(candidate_count or researchable_count),
            conclusion=(
                f"Deterministic Funnel 已将可研究标的 {researchable_count or 0} 只压缩到候选 {candidate_count or 0} 只。"
            ),
            parameters={
                "researchable_count": researchable_count,
                "candidate_count": candidate_count,
            },
            metadata={"funnel_summary": funnel_summary},
        ),
        ExecutionTraceStep(
            stage="bayesian_decision",
            role="deterministic",
            model="prior/likelihood/posterior",
            success=bool(bayesian_record_count or shortlist_count),
            conclusion=(
                f"Bayesian 层已完成候选排序，产出 {bayesian_record_count or 0} 条 posterior 记录，"
                f"形成 shortlist {shortlist_count or 0} 只。"
            ),
            parameters={
                "candidate_count": candidate_count,
                "bayesian_record_count": bayesian_record_count,
                "shortlist_count": shortlist_count,
            },
            metadata={"bayesian_trace": analysis.get("bayesian_trace", [])},
        ),
        ExecutionTraceStep(
            stage="master_synthesis",
            role="master",
            model=model_role_metadata.master_model,
            success=bool(analysis.get("master_success", True)),
            conclusion=(
                "Master synthesis 已生成结构化建议，并通过 hints 向后续 deterministic 控制链传递。"
            ),
            parameters={
                "master_timeout": model_role_metadata.master_timeout,
                "master_reasoning_effort": model_role_metadata.master_reasoning_effort,
                "advisory_only": True,
                "master_fallback_used": model_role_metadata.master_fallback_used,
            },
            fallback_reason=_join_texts(analysis.get("master_fallback_reasons", [])),
            metadata={
                "ic_hints_count": int(analysis.get("ic_hints_count", 0)),
                "shortlist": analysis.get("shortlist", []),
                "evidence_pack_token_count": evidence_pack_token_count,
                "evidence_pack_field_limit": evidence_pack_field_limit,
                "evidence_pack_shortlist_limit": evidence_pack_shortlist_limit,
                "company_name_coverage": company_name_coverage,
            },
        ),
        ExecutionTraceStep(
            stage="deterministic_risk_and_sizing",
            role="deterministic",
            model="RiskGuard/PortfolioConstructor",
            success=True,
            conclusion=(
                f"RiskGuard 与组合构建器完成最终决策：目标暴露 {target_exposure:.1%}，"
                f"单票上限 {max_single_weight:.1%}，精选标的 {final_selected_count} 只。"
            ),
            parameters={
                "risk_veto": bool(portfolio.get("risk_veto", False)),
                "action_cap": portfolio.get("action_cap", ""),
                "reliability": reliability,
            },
            metadata={
                "risk_summary": portfolio.get("risk_summary", {}),
                "execution_notes": portfolio.get("execution_notes", []),
                "target_weights": portfolio.get("target_weights", {}),
            },
        ),
    ]

    if persistence_note:
        steps.append(
            ExecutionTraceStep(
                stage="report_persistence",
                role="system",
                model="filesystem/history-store",
                success=True,
                conclusion=persistence_note,
            )
        )

    final_outcome = {
        "selected_count": final_selected_count,
        "target_exposure": target_exposure,
        "max_single_weight": max_single_weight,
        "reliability": reliability,
        "hard_veto": bool(portfolio.get("risk_veto", False)),
        "final_action_cap": portfolio.get("action_cap", ""),
    }
    return ExecutionTrace(
        model_roles=model_role_metadata,
        key_parameters={
            "total_stocks": total_stocks,
            "batch_count": batch_count,
            "category_count": category_count,
            "data_quality_issue_count": data_quality_issue_count,
            "total_universe_count": total_stocks,
            "researchable_count": researchable_count,
            "candidate_count": candidate_count,
            "quarantined_count": quarantined_count,
            "shortlistable_count": shortlistable_count,
            "shortlist_count": shortlist_count,
            "final_selected_count": final_selected_count,
            "selected_count": final_selected_count,
            "target_exposure": target_exposure,
            "max_single_weight": max_single_weight,
            "evidence_pack_token_count": evidence_pack_token_count,
            "evidence_pack_field_limit": evidence_pack_field_limit,
            "evidence_pack_shortlist_limit": evidence_pack_shortlist_limit,
            "company_name_coverage": company_name_coverage,
            "bayesian_record_count": bayesian_record_count,
            "reliability": reliability,
            "master_reasoning_effort": model_role_metadata.master_reasoning_effort,
            "universe_key": model_role_metadata.universe_key,
            "universe_size": model_role_metadata.universe_size,
            "branch_fallback_used": model_role_metadata.branch_fallback_used,
            "master_fallback_used": model_role_metadata.master_fallback_used,
            "branch_fallback_reason": model_role_metadata.branch_fallback_reason,
            "master_fallback_reason": model_role_metadata.master_fallback_reason,
            "resolver_strategy": resolver_payload.get("resolution_strategy", ""),
        },
        resolver_directory_priority=list(resolver_payload.get("directory_priority", [])),
        physical_directories_used_for_full_a=list(resolver_payload.get("physical_directories_used_for_full_a", [])),
        local_union_fallback_used=bool(resolver_payload.get("local_union_fallback_used", False)),
        resolved_file_paths_by_symbol=dict(resolver_payload.get("resolved_file_paths_by_symbol", {})),
        resolution_strategy=str(resolver_payload.get("resolution_strategy", "")),
        steps=steps,
        final_deterministic_outcome=final_outcome,
        metadata={
            "analysis_summary": analysis,
            "download_summary": download,
            "resolver": resolver_payload,
            "data_quality_issue_count": data_quality_issue_count,
        },
    )


def build_bayesian_trace(
    *,
    bayesian_records: list[Any],
    shortlist_limit: int = 20,
) -> list[dict[str, Any]]:
    """Build structured prior/likelihood/posterior trace for shortlisted symbols."""
    traces: list[dict[str, Any]] = []
    for record in bayesian_records[:shortlist_limit]:
        entry: dict[str, Any]
        if hasattr(record, "to_dict"):
            entry = record.to_dict()
        else:
            entry = {
                "symbol": getattr(record, "symbol", ""),
                "company_name": getattr(record, "company_name", ""),
                "posterior_win_rate": getattr(record, "posterior_win_rate", 0.0),
                "posterior_action_score": getattr(record, "posterior_action_score", 0.0),
                "posterior_confidence": getattr(record, "posterior_confidence", 0.0),
                "rank": getattr(record, "rank", 0),
            }
        traces.append(entry)
    return traces


def build_funnel_summary(
    *,
    universe_size: int = 0,
    candidates_count: int = 0,
    shortlist_count: int = 0,
    final_selected_count: int = 0,
    excluded_symbols: dict[str, str] | None = None,
    funnel_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build universe compression statistics."""
    return {
        "universe_size": universe_size,
        "candidates_count": candidates_count,
        "shortlist_count": shortlist_count,
        "final_selected_count": final_selected_count,
        "compression_ratio": (
            f"{universe_size} -> {candidates_count} -> {shortlist_count} -> {final_selected_count}"
        ),
        "excluded_count": len(excluded_symbols or {}),
        "funnel_metadata": dict(funnel_metadata or {}),
    }


def build_data_quality_diagnostics(
    *,
    quarantine_symbols: list[str] | None = None,
    data_quality_issues: list[Any] | None = None,
) -> dict[str, Any]:
    """Build quarantine summary for data quality diagnostics."""
    quarantine = list(quarantine_symbols or [])
    issues = list(data_quality_issues or [])
    return {
        "quarantined_count": len(quarantine),
        "quarantined_symbols": quarantine[:50],
        "issue_count": len(issues),
        "issue_summary": [
            str(getattr(issue, "description", issue))[:200]
            for issue in issues[:20]
        ],
    }


def _join_texts(values: Any) -> str:
    if not isinstance(values, list):
        return str(values or "")
    texts = [str(item).strip() for item in values if str(item).strip()]
    return "；".join(texts[:3])
