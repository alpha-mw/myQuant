from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import pandas as pd

from quant_investor import QuantInvestor
from quant_investor.agent_protocol import (
    ActionLabel,
    DataQualityDiagnostics,
    ExecutionTrace,
    ExecutionTraceStep,
    GlobalContext,
    ICDecision,
    ModelRoleMetadata,
    PortfolioDecision,
    PortfolioPlan,
    ReportBundle,
    RiskDecision,
    RiskLevel,
    ShortlistItem,
    StockReviewBundle,
    SymbolResearchPacket,
    WhatIfPlan,
    WhatIfScenario,
)
from quant_investor.agents.agent_contracts import (
    AgentEnhancedStrategy,
    BaseBranchAgentOutput,
    MasterAgentOutput,
    RiskAgentOutput,
    SymbolRecommendation,
)
from quant_investor.agents.narrator_agent import NarratorAgent
from quant_investor.branch_contracts import BranchResult, PortfolioStrategy, TradeRecommendation, UnifiedDataBundle
import quant_investor.market.analyze as market_analyze
import quant_investor.pipeline.mainline as mainline_module
import quant_investor.versioning as versioning


def make_price_frame(symbol: str, periods: int = 8) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=periods, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "close": [10.0 + index * 0.2 for index in range(periods)],
            "symbol": [symbol] * periods,
        }
    )


def make_data_bundle(symbols: list[str], market: str = "CN") -> UnifiedDataBundle:
    bundle = UnifiedDataBundle(
        market=market,
        symbols=list(symbols),
        symbol_data={symbol: make_price_frame(symbol) for symbol in symbols},
        metadata={
            "provenance_summary": {"fixture": "integration"},
            "coverage_summary": {"symbols": len(symbols)},
        },
    )
    bundle.stock_pool = list(symbols)
    bundle.stock_data = bundle.symbol_data
    bundle.current_prices = {symbol: 10.0 + index for index, symbol in enumerate(symbols)}
    bundle.stock_names = {symbol: symbol for symbol in symbols}
    bundle.provenance_summary = {"fixture": "integration"}
    bundle.coverage_summary = {"symbols": len(symbols)}
    return bundle


def make_trade_recommendation(
    symbol: str,
    *,
    action: str,
    weight: float,
    confidence: float,
    thesis: str,
    support_drivers: list[str] | None = None,
    drag_drivers: list[str] | None = None,
    weight_cap_reasons: list[str] | None = None,
) -> TradeRecommendation:
    return TradeRecommendation(
        symbol=symbol,
        action=action,
        weight=weight,
        confidence=confidence,
        rationale=thesis,
        one_line_conclusion=thesis,
        support_drivers=list(support_drivers or [f"{symbol} support"]),
        drag_drivers=list(drag_drivers or [f"{symbol} drag"]),
        weight_cap_reasons=list(weight_cap_reasons or [f"{symbol} weight cap"]),
    )


def make_branch_result(
    name: str,
    *,
    symbol_scores: dict[str, float],
    thesis: str,
    score: float,
    confidence: float,
    investment_risks: list[str] | None = None,
    coverage_notes: list[str] | None = None,
    diagnostic_notes: list[str] | None = None,
    signals: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> BranchResult:
    payload = BranchResult(
        branch_name=name,
        signals=dict(signals or {}),
        explanation=thesis,
        symbol_scores=dict(symbol_scores),
        metadata=dict(metadata or {}),
        final_score=score,
        final_confidence=confidence,
        conclusion=thesis,
        thesis_points=[thesis],
        investment_risks=list(investment_risks or [f"投资风险: {name} 风险暴露上升"]),
        coverage_notes=list(coverage_notes or [f"覆盖说明: {name} 覆盖充分"]),
        diagnostic_notes=list(diagnostic_notes or [f"工程异常: {name} 使用缓存快照"]),
        support_drivers=[f"{name} support driver"],
        drag_drivers=[f"{name} drag driver"],
        weight_cap_reasons=[f"{name} weight cap"],
        module_coverage={
            name: {
                "label": name,
                "available_symbols": len(symbol_scores),
                "total_symbols": len(symbol_scores),
                "disabled_batches": 0,
                "status": "active",
            }
        },
    )
    payload.metadata.setdefault("debate_status", "reviewed")
    if name == "macro":
        payload.signals.setdefault("liquidity_signal", "🟡")
    if name == "kline":
        payload.signals.setdefault("predicted_return", symbol_scores)
        payload.metadata.setdefault("branch_mode", "fast-screen")
    return payload


def make_strategy(
    weights: dict[str, float],
    *,
    thesis_prefix: str,
    action: str = "buy",
) -> PortfolioStrategy:
    recommendations = [
        make_trade_recommendation(
            symbol,
            action=action if weight > 0 else "watch",
            weight=weight,
            confidence=0.72,
            thesis=f"{thesis_prefix}: {symbol}",
            support_drivers=[f"{symbol} support"],
            drag_drivers=[f"{symbol} drag"],
            weight_cap_reasons=[f"{symbol} cap"],
        )
        for symbol, weight in weights.items()
    ]
    total_exposure = sum(weights.values())
    return PortfolioStrategy(
        recommendations=recommendations,
        target_weights=dict(weights),
        target_positions=dict(weights),
        position_limits={symbol: max(weight, 0.2) for symbol, weight in weights.items()},
        total_exposure=total_exposure,
        gross_exposure=total_exposure,
        net_exposure=total_exposure,
        cash_ratio=max(0.0, 1.0 - total_exposure),
        summary=f"{thesis_prefix} strategy",
        notes=[f"{thesis_prefix} notes"],
        style_bias="均衡",
        branch_consensus={
            "kline": 0.71,
            "quant": 0.68,
            "fundamental": 0.73,
            "intelligence": 0.63,
            "macro": 0.58,
        },
        risk_summary={
            "risk_level": "medium",
            "warnings": [f"{thesis_prefix} risk summary"],
            "max_single_position": max(weights.values(), default=0.0),
        },
        execution_notes=[f"{thesis_prefix} execution notes"],
        research_mode="production" if total_exposure > 0 else "research_only",
    )


def make_artifacts(
    symbols: list[str],
    *,
    thesis_prefix: str,
    veto: bool = False,
) -> dict[str, object]:
    if veto:
        weights: dict[str, float] = {symbol: 0.0 for symbol in symbols[:1]}
        strategy = make_strategy(weights, thesis_prefix=thesis_prefix, action="watch")
        risk_decision = RiskDecision(
            hard_veto=True,
            veto=True,
            risk_level=RiskLevel.HIGH,
            action_cap=ActionLabel.AVOID,
            blocked_symbols=list(symbols),
            position_limits={symbol: 0.0 for symbol in symbols},
            reasons=["RiskGuard veto: tail risk above hard threshold"],
        )
        ic_decision = ICDecision(
            thesis=f"{thesis_prefix} veto thesis",
            action=ActionLabel.WATCH,
            selected_symbols=[],
            rejected_symbols=list(symbols),
            rationale_points=["RiskGuard veto cannot be overridden"],
        )
        portfolio_plan = PortfolioPlan(
            target_exposure=0.0,
            target_gross_exposure=0.0,
            target_net_exposure=0.0,
            cash_ratio=1.0,
            target_weights={},
            target_positions={},
            position_limits={symbol: 0.0 for symbol in symbols},
            blocked_symbols=list(symbols),
            rejected_symbols=list(symbols),
            execution_notes=["RiskGuard veto engaged"],
            construction_notes=["No live trade due to hard veto"],
        )
    else:
        base_weights = {symbol: weight for symbol, weight in zip(symbols, [0.35, 0.30, 0.20], strict=False)}
        strategy = make_strategy(base_weights, thesis_prefix=thesis_prefix, action="buy")
        risk_decision = RiskDecision(
            hard_veto=False,
            veto=False,
            risk_level=RiskLevel.MEDIUM,
            action_cap=ActionLabel.BUY,
            blocked_symbols=[],
            position_limits={symbol: min(weight + 0.05, 0.4) for symbol, weight in base_weights.items()},
            reasons=["RiskGuard approved within exposure cap"],
        )
        ic_decision = ICDecision(
            thesis=f"{thesis_prefix} consensus thesis",
            action=ActionLabel.BUY,
            selected_symbols=list(base_weights),
            rejected_symbols=[symbol for symbol in symbols if symbol not in base_weights],
            rationale_points=["Consensus backed by multi-branch evidence"],
        )
        portfolio_plan = PortfolioPlan(
            target_exposure=sum(base_weights.values()),
            target_gross_exposure=sum(base_weights.values()),
            target_net_exposure=sum(base_weights.values()),
            cash_ratio=max(0.0, 1.0 - sum(base_weights.values())),
            target_weights=dict(base_weights),
            target_positions=dict(base_weights),
            position_limits={symbol: min(weight + 0.05, 0.4) for symbol, weight in base_weights.items()},
            blocked_symbols=[],
            rejected_symbols=[symbol for symbol in symbols if symbol not in base_weights],
            execution_notes=["PortfolioConstructor deterministic weights"],
            construction_notes=["Narrator is read-only"],
        )

    branch_results = {
        "kline": make_branch_result(
            "kline",
            symbol_scores={symbol: 0.71 - index * 0.05 for index, symbol in enumerate(symbols)},
            thesis=f"{thesis_prefix} kline thesis",
            score=0.71,
            confidence=0.83,
            investment_risks=["投资风险: 趋势拥挤导致回撤放大"],
            coverage_notes=["覆盖说明: Kline 快筛覆盖正常"],
            diagnostic_notes=["工程异常: Kline 使用离线快照"],
            signals={"screen_mode": "fast-screen"},
            metadata={"branch_mode": "fast-screen"},
        ),
        "quant": make_branch_result(
            "quant",
            symbol_scores={symbol: 0.68 - index * 0.03 for index, symbol in enumerate(symbols)},
            thesis=f"{thesis_prefix} quant thesis",
            score=0.68,
            confidence=0.78,
            investment_risks=["投资风险: 因子衰减速度上升"],
            coverage_notes=["覆盖说明: 量化因子覆盖完整"],
            diagnostic_notes=["工程异常: 因子缓存命中"],
        ),
        "fundamental": make_branch_result(
            "fundamental",
            symbol_scores={symbol: 0.73 - index * 0.02 for index, symbol in enumerate(symbols)},
            thesis=f"{thesis_prefix} fundamental thesis",
            score=0.73,
            confidence=0.81,
            investment_risks=["投资风险: 盈利修正下行"],
            coverage_notes=["覆盖说明: 财报覆盖充足"],
            diagnostic_notes=["工程异常: 文档语义降级到摘要"],
        ),
        "intelligence": make_branch_result(
            "intelligence",
            symbol_scores={symbol: 0.63 - index * 0.02 for index, symbol in enumerate(symbols)},
            thesis=f"{thesis_prefix} intelligence thesis",
            score=0.63,
            confidence=0.76,
            investment_risks=["投资风险: 事件波动抬升"],
            coverage_notes=["覆盖说明: 新闻覆盖正常"],
            diagnostic_notes=["工程异常: 情绪接口限流后回退"],
        ),
        "macro": make_branch_result(
            "macro",
            symbol_scores={symbol: 0.58 for symbol in symbols},
            thesis=f"{thesis_prefix} macro thesis",
            score=0.58,
            confidence=0.74,
            investment_risks=["投资风险: 宏观流动性收紧"],
            coverage_notes=["覆盖说明: 宏观覆盖正常"],
            diagnostic_notes=["工程异常: 宏观接口使用缓存"],
            signals={"liquidity_signal": "🟡"},
        ),
    }

    run_diagnostics = {
        "investment_risks": ["投资风险: 市场波动率中枢抬升"],
        "coverage_notes": ["覆盖说明: 部分长尾标的使用替代数据"],
        "diagnostic_notes": ["工程异常: second-pass 超时后使用缓存"],
    }
    return {
        "branch_results": branch_results,
        "risk_decision": risk_decision,
        "ic_decision": ic_decision,
        "portfolio_plan": portfolio_plan,
        "strategy": strategy,
        "run_diagnostics": run_diagnostics,
    }


def build_report_bundle(
    branch_results: dict[str, BranchResult],
    *,
    risk_decision: RiskDecision,
    ic_decision: ICDecision,
    portfolio_plan: PortfolioPlan,
    run_diagnostics: dict[str, list[str]] | None = None,
    dag_payload: dict[str, object] | None = None,
) -> tuple[dict[str, object], ReportBundle]:
    narrator = NarratorAgent()
    branch_verdicts = {
        name: narrator.branch_result_to_verdict(
            branch_result,
            thesis=branch_result.conclusion or branch_result.explanation or name,
        )
        for name, branch_result in branch_results.items()
    }
    payload = {
        "macro_verdict": branch_verdicts["macro"],
        "branch_summaries": branch_verdicts,
        "ic_decisions": [ic_decision],
        "portfolio_plan": portfolio_plan,
        "run_diagnostics": run_diagnostics or {},
    }
    if dag_payload:
        payload.update(dag_payload)
    bundle = narrator.run(payload)
    bundle = ReportBundle(
        architecture_version=versioning.ARCHITECTURE_VERSION,
        branch_schema_version=versioning.BRANCH_SCHEMA_VERSION,
        ic_protocol_version=versioning.IC_PROTOCOL_VERSION,
        report_protocol_version=versioning.REPORT_PROTOCOL_VERSION,
        headline=bundle.headline,
        summary=bundle.summary,
        macro_verdict=bundle.macro_verdict,
        branch_verdicts=bundle.branch_verdicts,
        risk_decision=risk_decision,
        ic_decision=ic_decision,
        ic_decisions=[ic_decision],
        portfolio_plan=portfolio_plan,
        markdown_report=bundle.markdown_report,
        executive_summary=bundle.executive_summary,
        market_view=bundle.market_view,
        branch_conclusions=bundle.branch_conclusions,
        stock_cards=bundle.stock_cards,
        coverage_summary=bundle.coverage_summary,
        appendix_diagnostics=bundle.appendix_diagnostics,
        highlights=bundle.highlights,
        warnings=bundle.warnings,
        diagnostics=bundle.diagnostics,
        metadata={**bundle.metadata, "narrator_read_only": True},
    )
    return branch_verdicts, bundle


def make_dag_artifacts(
    symbols: list[str],
    *,
    thesis_prefix: str,
    veto: bool = False,
    review_enabled: bool = False,
) -> dict[str, object]:
    artifacts = make_artifacts(symbols, thesis_prefix=thesis_prefix, veto=veto)
    selected_symbols = list(artifacts["portfolio_plan"].target_weights)
    shortlist_symbols = selected_symbols or list(symbols[:1])
    shortlist = [
        ShortlistItem(
            symbol=symbol,
            company_name=symbol,
            category="full_a",
            rank_score=0.91 - index * 0.05,
            action=ActionLabel.BUY if symbol in selected_symbols else ActionLabel.WATCH,
            confidence=0.82 - index * 0.04,
            expected_upside=0.12 - index * 0.01,
            suggested_weight=float(artifacts["portfolio_plan"].target_weights.get(symbol, 0.0)),
            risk_flags=[f"{symbol} risk"],
            rationale=[f"{thesis_prefix} posterior summary {symbol}"],
            metadata={
                "posterior_action_score": round(0.91 - index * 0.05, 4),
                "posterior_win_rate": round(0.66 - index * 0.03, 4),
                "posterior_confidence": round(0.82 - index * 0.04, 4),
                "posterior_edge_after_costs": round(0.05 - index * 0.01, 4),
                "posterior_capacity_penalty": round(0.01 + index * 0.002, 4),
            },
        )
        for index, symbol in enumerate(shortlist_symbols)
    ]
    model_role_metadata = ModelRoleMetadata(
        branch_model="deepseek-chat" if review_enabled else "deterministic",
        master_model="moonshot-v1-128k" if review_enabled else "deterministic",
        resolved_branch_model="deepseek-chat" if review_enabled else "deterministic",
        resolved_master_model="moonshot-v1-128k" if review_enabled else "deterministic",
        master_reasoning_effort="high",
        agent_layer_enabled=review_enabled,
        universe_key="full_a",
        universe_size=len(symbols),
    )
    execution_trace = ExecutionTrace(
        model_roles=model_role_metadata,
        key_parameters={
            "total_universe_count": len(symbols),
            "researchable_count": len(symbols),
            "shortlistable_count": len(shortlist_symbols),
            "shortlist_count": len(shortlist),
            "final_selected_count": len(selected_symbols),
        },
        resolution_strategy="logical_full_a",
        steps=[
            ExecutionTraceStep(
                stage="global_context",
                role="system",
                model="deterministic",
                success=True,
                conclusion=f"{len(symbols)} symbols resolved",
            ),
            ExecutionTraceStep(
                stage="candidate_review",
                role="system",
                model="deterministic",
                success=True,
                conclusion=f"{len(shortlist_symbols)} candidates reviewed",
            ),
            ExecutionTraceStep(
                stage="deterministic_portfolio_decision",
                role="system",
                model="deterministic",
                success=True,
                conclusion=f"{len(selected_symbols)} names selected",
            ),
        ],
        final_deterministic_outcome={"selected_count": len(selected_symbols)},
    )
    what_if_plan = WhatIfPlan(
        scenarios=[
            WhatIfScenario(
                scenario_name="macro_turns_weaker",
                trigger="macro score deteriorates",
                action="reduce exposure",
            ),
            WhatIfScenario(
                scenario_name="candidate_set_decays",
                trigger="posteriors lose edge",
                action="re-run funnel",
            ),
        ]
    )
    global_context = GlobalContext(
        market="CN",
        universe_key="full_a",
        rebalance_date="2026-03-26",
        latest_trade_date="2026-03-26",
        effective_target_trade_date="2026-03-26",
        macro_regime="neutral",
        universe_symbols=list(symbols),
        symbol_name_map={symbol: symbol for symbol in symbols},
        style_exposures={"style_bias": "均衡"},
        freshness_mode="stable",
        universe_tiers={
            "total": list(symbols),
            "researchable": list(symbols),
            "shortlistable": list(shortlist_symbols),
            "final_selected": list(selected_symbols),
        },
        data_quality_diagnostics=DataQualityDiagnostics(
            total_universe_count=len(symbols),
            researchable_universe_count=len(symbols),
            shortlistable_universe_count=len(shortlist_symbols),
            final_selected_universe_count=len(selected_symbols),
        ),
    )
    symbol_research_packets = {
        symbol: SymbolResearchPacket(
            symbol=symbol,
            company_name=symbol,
            market="CN",
            category="full_a",
            branch_scores={
                name: float(branch.symbol_scores.get(symbol, branch.final_score))
                for name, branch in artifacts["branch_results"].items()
            },
            branch_confidences={
                name: float(branch.final_confidence or branch.confidence or 0.0)
                for name, branch in artifacts["branch_results"].items()
            },
            branch_theses={
                name: str(branch.conclusion or branch.explanation or name)
                for name, branch in artifacts["branch_results"].items()
            },
            risk_flags=[f"{symbol} risk"],
            coverage_notes=["fixture coverage"],
            diagnostic_notes=["fixture diagnostics"],
        )
        for symbol in symbols
    }
    portfolio_decision = PortfolioDecision(
        shortlist=shortlist,
        target_exposure=float(artifacts["portfolio_plan"].target_exposure),
        target_gross_exposure=float(artifacts["portfolio_plan"].target_gross_exposure),
        target_net_exposure=float(artifacts["portfolio_plan"].target_net_exposure),
        cash_ratio=float(artifacts["portfolio_plan"].cash_ratio),
        target_weights=dict(artifacts["portfolio_plan"].target_weights),
        target_positions=dict(artifacts["portfolio_plan"].target_positions),
        risk_constraints={
            "blocked_symbols": list(artifacts["portfolio_plan"].blocked_symbols),
            "position_limits": dict(artifacts["portfolio_plan"].position_limits),
        },
        master_hints={
            item.symbol: {"action": item.action.value, "score": item.rank_score, "confidence": item.confidence}
            for item in shortlist
        },
        what_if_plan=what_if_plan,
        execution_trace=execution_trace,
        metadata={"selected_count": len(selected_symbols)},
    )
    review_bundle = StockReviewBundle(
        branch_summaries={},
        ic_hints_by_symbol={
            item.symbol: {
                "action": item.action.value,
                "score": item.rank_score,
                "confidence": item.confidence,
                "rationale_points": list(item.rationale),
            }
            for item in shortlist
        },
        fallback_reasons=[] if review_enabled else ["review_disabled"],
    )
    portfolio_master_output = (
        MasterAgentOutput(
            final_conviction="buy",
            final_score=0.66,
            confidence=0.84,
            debate_resolution=[f"{thesis_prefix} master resolution"],
            conviction_drivers=[f"{thesis_prefix} conviction driver"],
            top_picks=[
                SymbolRecommendation(
                    symbol=item.symbol,
                    action=item.action.value,
                    conviction="buy",
                    rationale=item.rationale[0],
                    target_weight=item.suggested_weight,
                )
                for item in shortlist
            ],
            risk_adjusted_exposure=float(artifacts["portfolio_plan"].target_exposure),
        )
        if review_enabled
        else None
    )
    bayesian_records = [
        {
            "symbol": item.symbol,
            "company_name": item.company_name,
            "posterior_action_score": item.metadata["posterior_action_score"],
            "posterior_win_rate": item.metadata["posterior_win_rate"],
            "posterior_confidence": item.metadata["posterior_confidence"],
            "posterior_edge_after_costs": item.metadata["posterior_edge_after_costs"],
            "posterior_capacity_penalty": item.metadata["posterior_capacity_penalty"],
            "rank": index + 1,
        }
        for index, item in enumerate(shortlist)
    ]
    funnel_summary = {
        "total_universe_count": len(symbols),
        "researchable_count": len(symbols),
        "shortlistable_count": len(shortlist_symbols),
        "final_selected_count": len(selected_symbols),
        "compression_ratio": f"{len(symbols)} -> {len(shortlist_symbols)} -> {len(selected_symbols)}",
    }
    branch_verdicts, report_bundle = build_report_bundle(
        artifacts["branch_results"],
        risk_decision=artifacts["risk_decision"],
        ic_decision=artifacts["ic_decision"],
        portfolio_plan=artifacts["portfolio_plan"],
        run_diagnostics=artifacts["run_diagnostics"],
        dag_payload={
            "global_context": global_context,
            "symbol_research_packets": symbol_research_packets,
            "shortlist": shortlist,
            "portfolio_decision": portfolio_decision,
            "model_role_metadata": model_role_metadata,
            "execution_trace": execution_trace,
            "what_if_plan": what_if_plan,
            "review_bundle": review_bundle,
            "ic_hints_by_symbol": dict(review_bundle.ic_hints_by_symbol),
            "bayesian_records": bayesian_records,
            "funnel_summary": funnel_summary,
        },
    )
    branch_verdicts_by_symbol = {
        symbol: {
            name: deepcopy(verdict)
            for name, verdict in branch_verdicts.items()
        }
        for symbol in symbols
    }
    for symbol, verdicts in branch_verdicts_by_symbol.items():
        for verdict in verdicts.values():
            verdict.symbol = symbol
    return {
        "global_context": global_context,
        "symbol_research_packets": symbol_research_packets,
        "branch_verdicts_by_symbol": branch_verdicts_by_symbol,
        "branch_summaries": branch_verdicts,
        "macro_verdict": branch_verdicts["macro"],
        "risk_decision": artifacts["risk_decision"],
        "ic_decisions": [artifacts["ic_decision"]],
        "shortlist": shortlist,
        "portfolio_plan": artifacts["portfolio_plan"],
        "portfolio_decision": portfolio_decision,
        "review_bundle": review_bundle,
        "model_role_metadata": model_role_metadata,
        "what_if_plan": what_if_plan,
        "execution_trace": execution_trace,
        "tradability_snapshot": {symbol: {"tradable": True} for symbol in symbols},
        "data_quality_issues": [],
        "data_quality_summary": {"researchable_count": len(symbols)},
        "resolver": {"resolution_strategy": "logical_full_a"},
        "report_bundle": report_bundle,
        "portfolio_master_output": portfolio_master_output,
        "portfolio_master_meta": {"confidence": 0.84} if review_enabled else {},
        "branch_results": artifacts["branch_results"],
        "bayesian_records": bayesian_records,
        "funnel_output": SimpleNamespace(
            candidates=list(shortlist_symbols),
            excluded_symbols={},
            funnel_metadata=funnel_summary,
        ),
        "funnel_summary": funnel_summary,
    }


def make_review_result(
    symbols: list[str],
    *,
    thesis_prefix: str,
) -> AgentEnhancedStrategy:
    branch_outputs = {
        branch_name: BaseBranchAgentOutput(
            branch_name=branch_name,
            conviction="buy" if branch_name != "macro" else "neutral",
            conviction_score=0.62 if branch_name != "macro" else 0.18,
            confidence=0.81,
            key_insights=[f"{thesis_prefix} {branch_name} review"],
            risk_flags=[f"投资风险: {branch_name} review risk"],
            symbol_views={symbol: f"{thesis_prefix} {branch_name} view {symbol}" for symbol in symbols},
            reasoning="review reasoning should stay advisory",
        )
        for branch_name in ["kline", "quant", "fundamental", "intelligence", "macro"]
    }
    master_output = MasterAgentOutput(
        final_conviction="buy",
        final_score=0.66,
        confidence=0.84,
        debate_resolution=[f"{thesis_prefix} master resolution"],
        conviction_drivers=[f"{thesis_prefix} conviction driver"],
        consensus_areas=[f"{thesis_prefix} consensus area"],
        disagreement_areas=[f"{thesis_prefix} disagreement area"],
        top_picks=[
            SymbolRecommendation(
                symbol=symbol,
                action="buy",
                conviction="strong_buy",
                rationale=f"{thesis_prefix} top pick {symbol}",
                target_weight=0.95,
            )
            for symbol in symbols
        ],
        portfolio_narrative="This narrative should not determine final weights.",
        risk_adjusted_exposure=0.72,
    )
    risk_output = RiskAgentOutput(
        risk_assessment="elevated",
        max_recommended_exposure=0.72,
        risk_warnings=[f"投资风险: {thesis_prefix} review cap"],
        position_sizing_overrides={
            symbol: {"max_weight": min(0.35, 0.2 + index * 0.05)}
            for index, symbol in enumerate(symbols)
        },
        reasoning="risk reasoning should stay advisory",
    )
    return AgentEnhancedStrategy(
        algorithmic_strategy={"mode": "fixture"},
        agent_strategy=master_output,
        agent_layer_success=True,
        fallback_used=False,
        branch_agent_outputs=branch_outputs,
        risk_agent_output=risk_output,
        agent_layer_timings={"branch_review": 0.01, "master_review": 0.01},
    )


def assert_protocol_bundle(
    report_bundle: ReportBundle,
    *,
    risk_decision: RiskDecision,
    ic_decision: ICDecision,
    portfolio_plan: PortfolioPlan,
) -> None:
    assert isinstance(report_bundle, ReportBundle)
    assert report_bundle.agent_name == "NarratorAgent"
    assert report_bundle.metadata.get("narrator_read_only") is True
    assert report_bundle.macro_verdict is not None
    assert report_bundle.macro_verdict.thesis.strip()
    assert report_bundle.risk_decision == risk_decision
    assert report_bundle.ic_decision == ic_decision
    assert report_bundle.portfolio_plan == portfolio_plan
    assert report_bundle.portfolio_plan.target_weights == portfolio_plan.target_weights
    assert report_bundle.ic_decision.action == ic_decision.action
    assert report_bundle.markdown_report.strip()
    assert report_bundle.coverage_summary
    assert report_bundle.appendix_diagnostics
    assert report_bundle.warnings
    assert all(verdict.thesis.strip() for verdict in report_bundle.branch_verdicts.values())
    assert any("覆盖说明" in note for note in report_bundle.coverage_summary)
    assert not any("工程异常" in note for note in report_bundle.coverage_summary)
    assert any("工程异常" in note for note in report_bundle.appendix_diagnostics)
    assert not any("覆盖说明" in note for note in report_bundle.appendix_diagnostics)
    assert any("投资风险" in note for note in report_bundle.warnings)


def run_stubbed_quant_path(
    monkeypatch,
    *,
    symbols: list[str],
    thesis_prefix: str,
    veto: bool = False,
    kline_backend: str = "hybrid",
    review_enabled: bool = False,
) -> SimpleNamespace:
    artifacts = make_artifacts(symbols, thesis_prefix=thesis_prefix, veto=veto)
    dag_artifacts = make_dag_artifacts(
        symbols,
        thesis_prefix=thesis_prefix,
        veto=veto,
        review_enabled=review_enabled,
    )
    review_result = make_review_result(symbols, thesis_prefix=thesis_prefix) if review_enabled else None
    monkeypatch.setattr(mainline_module, "_execute_market_dag", lambda **_kwargs: dag_artifacts, raising=False)

    investor = QuantInvestor(
        stock_pool=symbols,
        market="CN",
        kline_backend=kline_backend,
        verbose=False,
        enable_agent_layer=review_enabled,
    )
    result = investor.run()
    return SimpleNamespace(
        investor=investor,
        result=result,
        artifacts=artifacts,
        review_result=review_result,
    )


def run_stubbed_full_market_path(
    monkeypatch,
    tmp_path: Path,
    *,
    symbols: list[str],
    category: str = "core",
    analysis_kwargs: dict[str, object] | None = None,
) -> SimpleNamespace:
    artifacts = make_artifacts(symbols, thesis_prefix="full-market", veto=False)
    dag_artifacts = make_dag_artifacts(
        symbols,
        thesis_prefix="full-market",
        veto=False,
        review_enabled=bool((analysis_kwargs or {}).get("enable_agent_layer", False)),
    )
    captured_dag_kwargs: list[dict[str, object]] = []

    def _save_candidate_index(_all_results, *, market: str, output_dir: str):
        path = Path(output_dir) / f"{market}_candidates.json"
        path.write_text(json.dumps({"symbols": symbols}, ensure_ascii=False), encoding="utf-8")
        return str(path)

    def _build_full_market_report_bundle(_all_results, *, market: str, total_capital: float, top_k: int):
        report_bundle = dag_artifacts["report_bundle"]
        plan = {
            "market_summary": {
                "generated_at": "2026-03-24T00:00:00Z",
                "total_stocks": len(symbols),
                "total_batches": 1,
                "market": market,
                "top_k": top_k,
                "total_capital": total_capital,
            },
            "portfolio_plan": {
                "target_weights": dict(artifacts["portfolio_plan"].target_weights),
                "cash_ratio": artifacts["portfolio_plan"].cash_ratio,
                "blocked_symbols": list(artifacts["portfolio_plan"].blocked_symbols),
            },
            "selected_symbols": list(artifacts["portfolio_plan"].target_weights),
        }
        return plan, report_bundle

    def _fake_execute_market_dag(**kwargs):
        captured_dag_kwargs.append(dict(kwargs))
        return dag_artifacts

    monkeypatch.setattr(market_analyze, "execute_market_dag", _fake_execute_market_dag, raising=False)
    monkeypatch.setattr(
        market_analyze,
        "get_market_settings",
        lambda market: SimpleNamespace(
            market=market,
            default_batch_size=len(symbols),
            analysis_output_dir=tmp_path,
            market_name="中国A股",
            report_flag="CN",
            currency_symbol="¥",
        ),
    )
    monkeypatch.setattr(market_analyze, "normalize_categories", lambda _market, categories=None: list(categories or [category]))
    monkeypatch.setattr(market_analyze, "category_name", lambda _category, _market: "核心样本")
    monkeypatch.setattr(market_analyze, "get_all_local_symbols", lambda _category, market="CN": list(symbols))
    monkeypatch.setattr(market_analyze, "load_stock_names", lambda _market: None)
    monkeypatch.setattr(market_analyze, "save_candidate_index", _save_candidate_index, raising=False)
    monkeypatch.setattr(
        market_analyze,
        "_build_full_market_report_bundle",
        _build_full_market_report_bundle,
        raising=False,
    )

    output = market_analyze.run_market_analysis(
        market="CN",
        mode="sample",
        categories=[category],
        total_capital=1_000_000,
        top_k=min(2, len(symbols)),
        verbose=False,
        **dict(analysis_kwargs or {}),
    )
    _plan, report_bundle = _build_full_market_report_bundle(
        {},
        market="CN",
        total_capital=1_000_000,
        top_k=min(2, len(symbols)),
    )
    return SimpleNamespace(
        output=output,
        report_bundle=report_bundle,
        captured_dag_kwargs=captured_dag_kwargs,
        artifacts=artifacts,
    )
