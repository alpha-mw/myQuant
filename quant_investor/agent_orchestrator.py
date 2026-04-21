#!/usr/bin/env python3
"""
新架构 AgentOrchestrator。

固定执行顺序：
1. 加载 UnifiedDataBundle
2. 运行一次 MacroAgent
3. 对每个 symbol 运行 4 个 research agents
4. 对每个 symbol 运行 RiskGuard
5. 对每个 symbol 运行 ICCoordinator
6. 运行一次 PortfolioConstructor
7. 运行一次 NarratorAgent
8. 持久化输出

为兼容过渡，也提供使用旧 branch_results 的 bridge 入口。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

import pandas as pd

from quant_investor.agent_protocol import (
    ActionLabel,
    AgentStatus,
    BranchVerdict,
    CoverageScope,
    ICDecision,
    PortfolioPlan,
    ReportBundle,
    RiskDecision,
    RiskLevel,
)
from quant_investor.agents.fundamental_agent import FundamentalAgent
from quant_investor.agents.ic_coordinator import ICCoordinator
from quant_investor.agents.intelligence_agent import IntelligenceAgent
from quant_investor.agents.kline_agent import KlineAgent
from quant_investor.agents.macro_agent import MacroAgent
from quant_investor.agents.narrator_agent import NarratorAgent
from quant_investor.agents.portfolio_constructor import PortfolioConstructor
from quant_investor.agents.quant_agent import QuantAgent
from quant_investor.agents.risk_guard import RiskGuard
from quant_investor.branch_contracts import BranchResult, UnifiedDataBundle
from quant_investor.reporting.run_artifacts import (
    build_execution_trace,
    build_model_role_metadata,
    build_what_if_plan,
)
from quant_investor.versioning import (
    ARCHITECTURE_VERSION,
    BRANCH_SCHEMA_VERSION,
    IC_PROTOCOL_VERSION,
    REPORT_PROTOCOL_VERSION,
)


class ControlChainOrchestrator:
    """把新协议层 agents 串为固定调用链。"""

    @staticmethod
    def _version_info() -> dict[str, str]:
        return {
            "architecture_version": ARCHITECTURE_VERSION,
            "branch_schema_version": BRANCH_SCHEMA_VERSION,
            "ic_protocol_version": IC_PROTOCOL_VERSION,
            "report_protocol_version": REPORT_PROTOCOL_VERSION,
        }

    def __init__(
        self,
        macro_agent: MacroAgent | None = None,
        kline_agent: KlineAgent | None = None,
        quant_agent: QuantAgent | None = None,
        fundamental_agent: FundamentalAgent | None = None,
        intelligence_agent: IntelligenceAgent | None = None,
        risk_guard: RiskGuard | None = None,
        ic_coordinator: ICCoordinator | None = None,
        portfolio_constructor: PortfolioConstructor | None = None,
        narrator_agent: NarratorAgent | None = None,
    ) -> None:
        self.macro_agent = macro_agent or MacroAgent()
        self.kline_agent = kline_agent or KlineAgent()
        self.quant_agent = quant_agent or QuantAgent()
        self.fundamental_agent = fundamental_agent or FundamentalAgent()
        self.intelligence_agent = intelligence_agent or IntelligenceAgent()
        self.risk_guard = risk_guard or RiskGuard()
        self.ic_coordinator = ic_coordinator or ICCoordinator()
        self.portfolio_constructor = portfolio_constructor or PortfolioConstructor()
        self.narrator_agent = narrator_agent or NarratorAgent()

    def run(
        self,
        *,
        data_bundle: UnifiedDataBundle | None = None,
        data_loader: Any | None = None,
        constraints: Mapping[str, Any] | None = None,
        existing_portfolio: Mapping[str, Any] | None = None,
        tradability_snapshot: Mapping[str, Any] | None = None,
        review_bundle: Any | None = None,
        persist_dir: str | Path | None = None,
        persist_outputs: bool = True,
    ) -> dict[str, Any]:
        """走完整的新 agent 调用链。"""

        bundle = self._load_data_bundle(data_bundle=data_bundle, data_loader=data_loader)
        normalized_constraints = self._normalize_constraints(constraints)
        normalized_portfolio = self._normalize_existing_portfolio(existing_portfolio)
        normalized_tradability = self._normalize_tradability_snapshot(
            bundle,
            tradability_snapshot,
        )

        macro_verdict = self.macro_agent.run(
            {
                "market_snapshot": self._build_market_snapshot(bundle),
            }
        )
        research_by_symbol = self._run_research_agents_per_symbol(bundle)
        return self._finalize(
            data_bundle=bundle,
            macro_verdict=macro_verdict,
            research_by_symbol=research_by_symbol,
            constraints=normalized_constraints,
            existing_portfolio=normalized_portfolio,
            tradability_snapshot=normalized_tradability,
            review_bundle=review_bundle,
            persist_dir=persist_dir,
            persist_outputs=persist_outputs,
        )

    def run_with_precomputed_research(
        self,
        *,
        data_bundle: UnifiedDataBundle,
        branch_results: Mapping[str, BranchResult],
        constraints: Mapping[str, Any] | None = None,
        existing_portfolio: Mapping[str, Any] | None = None,
        tradability_snapshot: Mapping[str, Any] | None = None,
        review_bundle: Any | None = None,
        persist_dir: str | Path | None = None,
        persist_outputs: bool = False,
    ) -> dict[str, Any]:
        """兼容过渡入口：复用旧 pipeline 已算出的 branch_results。"""

        normalized_constraints = self._normalize_constraints(constraints)
        normalized_portfolio = self._normalize_existing_portfolio(existing_portfolio)
        normalized_tradability = self._normalize_tradability_snapshot(
            data_bundle,
            tradability_snapshot,
        )
        macro_verdict = self._build_macro_verdict_from_branch_results(branch_results, data_bundle)
        research_by_symbol = self._build_symbol_verdicts_from_branch_results(
            branch_results,
            data_bundle,
        )
        return self._finalize(
            data_bundle=data_bundle,
            macro_verdict=macro_verdict,
            research_by_symbol=research_by_symbol,
            constraints=normalized_constraints,
            existing_portfolio=normalized_portfolio,
            tradability_snapshot=normalized_tradability,
            ic_hints_by_symbol={},
            review_bundle=review_bundle,
            persist_dir=persist_dir,
            persist_outputs=persist_outputs,
        )

    def run_with_structured_research(
        self,
        *,
        data_bundle: UnifiedDataBundle,
        macro_verdict: BranchVerdict,
        research_by_symbol: Mapping[str, Mapping[str, BranchVerdict]],
        constraints: Mapping[str, Any] | None = None,
        existing_portfolio: Mapping[str, Any] | None = None,
        tradability_snapshot: Mapping[str, Any] | None = None,
        ic_hints_by_symbol: Mapping[str, Mapping[str, Any]] | None = None,
        review_bundle: Any | None = None,
        persist_dir: str | Path | None = None,
        persist_outputs: bool = False,
    ) -> dict[str, Any]:
        """使用已结构化的 research verdicts 执行唯一控制链。"""

        if not isinstance(macro_verdict, BranchVerdict):
            raise TypeError("macro_verdict 必须是 BranchVerdict")

        normalized_constraints = self._normalize_constraints(constraints)
        normalized_portfolio = self._normalize_existing_portfolio(existing_portfolio)
        normalized_tradability = self._normalize_tradability_snapshot(
            data_bundle,
            tradability_snapshot,
        )
        normalized_hints = self._normalize_ic_hints_by_symbol(ic_hints_by_symbol)
        normalized_research = self._normalize_research_by_symbol(research_by_symbol)
        return self._finalize(
            data_bundle=data_bundle,
            macro_verdict=macro_verdict,
            research_by_symbol=normalized_research,
            constraints=normalized_constraints,
            existing_portfolio=normalized_portfolio,
            tradability_snapshot=normalized_tradability,
            ic_hints_by_symbol=normalized_hints,
            review_bundle=review_bundle,
            persist_dir=persist_dir,
            persist_outputs=persist_outputs,
        )

    @staticmethod
    def _load_data_bundle(
        *,
        data_bundle: UnifiedDataBundle | None,
        data_loader: Any | None,
    ) -> UnifiedDataBundle:
        if isinstance(data_bundle, UnifiedDataBundle):
            return data_bundle
        if callable(data_loader):
            loaded = data_loader()
            if isinstance(loaded, UnifiedDataBundle):
                return loaded
            raise TypeError("data_loader 必须返回 UnifiedDataBundle")
        raise ValueError("必须提供 data_bundle 或 data_loader")

    @staticmethod
    def _normalize_constraints(constraints: Mapping[str, Any] | None) -> dict[str, Any]:
        if constraints is None:
            return {}
        if not isinstance(constraints, Mapping):
            raise TypeError("constraints 必须是 Mapping")
        return dict(constraints)

    @staticmethod
    def _normalize_existing_portfolio(
        existing_portfolio: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if existing_portfolio is None:
            return {"current_weights": {}}
        if not isinstance(existing_portfolio, Mapping):
            raise TypeError("existing_portfolio 必须是 Mapping")
        payload = dict(existing_portfolio)
        if "current_weights" not in payload and "positions" in payload:
            payload["current_weights"] = dict(payload.get("positions", {}))
        payload.setdefault("current_weights", {})
        return payload

    def _run_research_agents_per_symbol(
        self,
        data_bundle: UnifiedDataBundle,
    ) -> dict[str, dict[str, BranchVerdict]]:
        research_by_symbol: dict[str, dict[str, BranchVerdict]] = {}
        for symbol in data_bundle.symbols:
            branch_payload = {
                "data_bundle": data_bundle,
                "stock_pool": [symbol],
                "market": data_bundle.market,
                "verbose": False,
            }
            research_by_symbol[symbol] = {
                "kline": self._ensure_symbol_verdict(
                    self.kline_agent.run(
                        {
                            **branch_payload,
                            "mode": "shortlist",
                        }
                    ),
                    symbol=symbol,
                    branch_name="kline",
                ),
                "quant": self._ensure_symbol_verdict(
                    self.quant_agent.run(branch_payload),
                    symbol=symbol,
                    branch_name="quant",
                ),
                "fundamental": self._ensure_symbol_verdict(
                    self.fundamental_agent.run(branch_payload),
                    symbol=symbol,
                    branch_name="fundamental",
                ),
                "intelligence": self._ensure_symbol_verdict(
                    self.intelligence_agent.run(
                        {
                            **branch_payload,
                            "market_regime": data_bundle.macro_data.get("regime"),
                        }
                    ),
                    symbol=symbol,
                    branch_name="intelligence",
                ),
            }
        return research_by_symbol

    def _build_macro_verdict_from_branch_results(
        self,
        branch_results: Mapping[str, BranchResult],
        data_bundle: UnifiedDataBundle,
    ) -> BranchVerdict:
        macro_branch = branch_results.get("macro")
        if isinstance(macro_branch, BranchResult):
            verdict = self.macro_agent.branch_result_to_verdict(
                macro_branch,
                thesis=str(macro_branch.conclusion or "宏观分支已生成结构化判断。"),
                metadata={
                    "regime": str(
                        macro_branch.signals.get("macro_regime")
                        or macro_branch.signals.get("risk_level")
                        or "neutral"
                    ),
                    "target_gross_exposure": self._macro_target_exposure_from_score(
                        float(macro_branch.score)
                    ),
                    "style_bias": self._style_bias_from_macro_score(float(macro_branch.score)),
                },
                scope=CoverageScope.MARKET,
            )
            verdict.symbol = None
            verdict.metadata["symbol"] = None
            return verdict

        return self.macro_agent.run(
            {
                "market_snapshot": self._build_market_snapshot(data_bundle),
            }
        )

    def _build_symbol_verdicts_from_branch_results(
        self,
        branch_results: Mapping[str, BranchResult],
        data_bundle: UnifiedDataBundle,
    ) -> dict[str, dict[str, BranchVerdict]]:
        agent_map = {
            "kline": self.kline_agent,
            "quant": self.quant_agent,
            "fundamental": self.fundamental_agent,
            "intelligence": self.intelligence_agent,
        }
        research_by_symbol = {
            symbol: {}
            for symbol in data_bundle.symbols
        }

        for branch_name, agent in agent_map.items():
            branch_result = branch_results.get(branch_name)
            if not isinstance(branch_result, BranchResult):
                continue
            pool_verdict = agent.branch_result_to_verdict(
                branch_result,
                thesis=str(branch_result.conclusion or "").strip()
                or f"{branch_name} 分支已生成结构化判断。",
            )
            for symbol in data_bundle.symbols:
                research_by_symbol[symbol][branch_name] = self._ensure_symbol_verdict(
                    pool_verdict,
                    symbol=symbol,
                    branch_name=branch_name,
                )
        return research_by_symbol

    def _finalize(
        self,
        *,
        data_bundle: UnifiedDataBundle,
        macro_verdict: BranchVerdict,
        research_by_symbol: Mapping[str, Mapping[str, BranchVerdict]],
        constraints: Mapping[str, Any],
        existing_portfolio: Mapping[str, Any],
        tradability_snapshot: Mapping[str, Any],
        ic_hints_by_symbol: Mapping[str, Mapping[str, Any]],
        review_bundle: Any | None,
        persist_dir: str | Path | None,
        persist_outputs: bool,
    ) -> dict[str, Any]:
        risk_by_symbol: dict[str, RiskDecision] = {}
        ic_by_symbol: dict[str, ICDecision] = {}

        for symbol in data_bundle.symbols:
            branch_verdicts = dict(research_by_symbol.get(symbol, {}))
            current_weight = float(existing_portfolio.get("current_weights", {}).get(symbol, 0.0))
            risk_decision = self.risk_guard.run(
                {
                    "branch_verdicts": branch_verdicts,
                    "macro_verdict": macro_verdict,
                    "portfolio_state": {
                        "candidate_symbols": [symbol],
                        "current_weights": {symbol: current_weight} if current_weight else {},
                    },
                    "constraints": constraints,
                }
            )
            risk_by_symbol[symbol] = risk_decision

            ic_decision = self.ic_coordinator.run(
                {
                    "branch_verdicts": branch_verdicts,
                    "risk_decision": risk_decision,
                    "ic_hints": dict(ic_hints_by_symbol.get(symbol, {})),
                }
            )
            ic_by_symbol[symbol] = self._attach_symbol_to_ic_decision(
                ic_decision=ic_decision,
                symbol=symbol,
                risk_decision=risk_decision,
                current_weight=current_weight,
                tradability_info=tradability_snapshot.get(symbol, {}),
                ic_hint=ic_hints_by_symbol.get(symbol, {}),
            )

        aggregated_risk_limits = self._aggregate_risk_limits(
            constraints=constraints,
            risk_by_symbol=risk_by_symbol,
            macro_verdict=macro_verdict,
        )
        portfolio_plan = self.portfolio_constructor.run(
            {
                "ic_decisions": list(ic_by_symbol.values()),
                "macro_verdict": macro_verdict,
                "risk_limits": aggregated_risk_limits,
                "existing_portfolio": existing_portfolio,
                "tradability_snapshot": tradability_snapshot,
            }
        )

        branch_summaries = self._aggregate_branch_summaries(research_by_symbol)
        run_diagnostics = self._build_run_diagnostics(research_by_symbol)
        aggregated_risk_decision = self._aggregate_report_risk_decision(
            risk_by_symbol=risk_by_symbol,
            aggregated_risk_limits=aggregated_risk_limits,
        )
        review_metadata = dict(getattr(review_bundle, "metadata", {}) or {}) if review_bundle is not None else {}
        model_role_metadata = build_model_role_metadata(
            branch_model=str(review_metadata.get("branch_model", "")),
            master_model=str(review_metadata.get("master_model", "")),
            agent_fallback_model=str(review_metadata.get("branch_fallback_model", "")),
            master_fallback_model=str(review_metadata.get("master_fallback_model", "")),
            resolved_branch_model=str(review_metadata.get("branch_primary_model", review_metadata.get("branch_model", ""))),
            resolved_master_model=str(review_metadata.get("master_primary_model", review_metadata.get("master_model", ""))),
            master_reasoning_effort=str(review_metadata.get("master_reasoning_effort", "")),
            branch_provider=str(review_metadata.get("branch_provider", "")),
            master_provider=str(review_metadata.get("master_provider", "")),
            branch_timeout=float(review_metadata.get("branch_timeout", 0.0)),
            master_timeout=float(review_metadata.get("master_timeout", 0.0)),
            agent_layer_enabled=bool(review_metadata.get("agent_layer_enabled", False)),
            branch_fallback_used=bool(review_metadata.get("branch_fallback_used", False)),
            master_fallback_used=bool(review_metadata.get("master_fallback_used", False)),
            branch_fallback_reason=str(review_metadata.get("branch_fallback_reason", "")),
            master_fallback_reason=str(review_metadata.get("master_fallback_reason", "")),
            universe_key=str(review_metadata.get("universe_key", "")),
            universe_size=int(review_metadata.get("symbol_count", 0)),
            universe_hash=str(review_metadata.get("universe_hash", "")),
            metadata=review_metadata,
        )
        what_if_plan = build_what_if_plan(
            portfolio_plan=portfolio_plan,
            market_summary={
                "candidate_count": len(ic_by_symbol),
                "selected_count": len([item for item in ic_by_symbol.values() if item.action == ActionLabel.BUY]),
                "macro_score": float(macro_verdict.final_score),
            },
            model_roles=model_role_metadata,
            candidate_count=len(ic_by_symbol),
            selected_count=len([item for item in ic_by_symbol.values() if item.action == ActionLabel.BUY]),
        )
        execution_trace = build_execution_trace(
            model_roles=model_role_metadata,
            analysis_meta={
                "batch_count": len(research_by_symbol),
                "category_count": len(research_by_symbol),
                "total_stocks": len(data_bundle.symbols),
                "fallback_reasons": list(getattr(review_bundle, "fallback_reasons", []) or []),
                "master_success": bool(review_bundle is not None),
                "ic_hints_count": len(ic_hints_by_symbol),
            },
            portfolio_plan={
                "selected_count": len([item for item in ic_by_symbol.values() if item.action == ActionLabel.BUY]),
                "target_exposure": float(portfolio_plan.target_gross_exposure),
                "max_single_weight": max(portfolio_plan.position_limits.values(), default=0.0),
                "risk_veto": bool(aggregated_risk_decision.veto),
                "action_cap": aggregated_risk_decision.action_cap.value,
                "risk_summary": portfolio_plan.metadata.get("risk_summary", {}),
                "execution_notes": portfolio_plan.execution_notes,
            },
        )
        report_bundle = self.narrator_agent.run(
            {
                "macro_verdict": macro_verdict,
                "branch_summaries": branch_summaries,
                "ic_decisions": list(ic_by_symbol.values()),
                "portfolio_plan": portfolio_plan,
                "run_diagnostics": run_diagnostics,
                "review_bundle": review_bundle,
                "ic_hints_by_symbol": dict(ic_hints_by_symbol),
                "model_role_metadata": model_role_metadata,
                "execution_trace": execution_trace,
                "what_if_plan": what_if_plan,
            }
        )
        report_bundle = ReportBundle(
            architecture_version=report_bundle.architecture_version,
            branch_schema_version=report_bundle.branch_schema_version,
            ic_protocol_version=report_bundle.ic_protocol_version,
            report_protocol_version=report_bundle.report_protocol_version,
            headline=report_bundle.headline,
            summary=report_bundle.summary,
            macro_verdict=report_bundle.macro_verdict,
            branch_verdicts=report_bundle.branch_verdicts,
            risk_decision=aggregated_risk_decision,
            ic_decision=next(iter(ic_by_symbol.values()), None),
            ic_decisions=list(ic_by_symbol.values()),
            review_bundle=report_bundle.review_bundle,
            ic_hints_by_symbol=dict(report_bundle.ic_hints_by_symbol),
            model_role_metadata=report_bundle.model_role_metadata,
            execution_trace=report_bundle.execution_trace,
            what_if_plan=report_bundle.what_if_plan,
            portfolio_plan=portfolio_plan,
            markdown_report=report_bundle.markdown_report,
            executive_summary=report_bundle.executive_summary,
            market_view=report_bundle.market_view,
            branch_conclusions=report_bundle.branch_conclusions,
            stock_cards=report_bundle.stock_cards,
            coverage_summary=report_bundle.coverage_summary,
            appendix_diagnostics=report_bundle.appendix_diagnostics,
            highlights=report_bundle.highlights,
            warnings=report_bundle.warnings,
            diagnostics=report_bundle.diagnostics,
            metadata={**report_bundle.metadata, "narrator_read_only": True},
        )

        persisted_paths = (
            self._persist_outputs(
                macro_verdict=macro_verdict,
                research_by_symbol=research_by_symbol,
                risk_by_symbol=risk_by_symbol,
                ic_by_symbol=ic_by_symbol,
                portfolio_plan=portfolio_plan,
                report_bundle=report_bundle,
                persist_dir=persist_dir,
            )
            if persist_outputs
            else {}
        )
        return {
            "data_bundle": data_bundle,
            **self._version_info(),
            "macro_verdict": macro_verdict,
            "research_by_symbol": research_by_symbol,
            "risk_by_symbol": risk_by_symbol,
            "ic_by_symbol": ic_by_symbol,
            "portfolio_plan": portfolio_plan,
            "report_bundle": report_bundle,
            "persisted_paths": persisted_paths,
            "review_bundle": review_bundle,
        }

    def _attach_symbol_to_ic_decision(
        self,
        *,
        ic_decision: ICDecision,
        symbol: str,
        risk_decision: RiskDecision,
        current_weight: float,
        tradability_info: Mapping[str, Any],
        ic_hint: Mapping[str, Any] | None = None,
    ) -> ICDecision:
        payload = deepcopy(ic_decision)
        position_mode = self._position_mode_from_action(
            action=payload.action,
            risk_decision=risk_decision,
            current_weight=current_weight,
            symbol=symbol,
        )
        payload.selected_symbols = [symbol] if position_mode == "target" else []
        payload.rejected_symbols = [symbol] if position_mode != "target" else []
        payload.metadata = dict(payload.metadata)
        payload.metadata.update(
            {
                "symbol": symbol,
                "risk_action_cap": risk_decision.action_cap.value,
                "symbol_candidates": [
                    {
                        "symbol": symbol,
                        "score": payload.final_score,
                        "confidence": payload.final_confidence,
                        "action": payload.action.value,
                        "position_mode": position_mode,
                        "current_weight": current_weight,
                        "sector": str(tradability_info.get("sector", "unknown")),
                        "one_line_conclusion": payload.thesis,
                    }
                ],
            }
        )
        if ic_hint:
            payload.metadata["llm_master_hint"] = dict(ic_hint)
        return payload

    @staticmethod
    def _normalize_research_by_symbol(
        payload: Mapping[str, Mapping[str, BranchVerdict]],
    ) -> dict[str, dict[str, BranchVerdict]]:
        if not isinstance(payload, Mapping):
            raise TypeError("research_by_symbol 必须是 Mapping[str, Mapping[str, BranchVerdict]]")
        result: dict[str, dict[str, BranchVerdict]] = {}
        for symbol, branch_map in payload.items():
            if not isinstance(branch_map, Mapping):
                raise TypeError("research_by_symbol 的 value 必须是 Mapping[str, BranchVerdict]")
            result[str(symbol)] = {
                str(branch_name): verdict
                for branch_name, verdict in branch_map.items()
                if isinstance(verdict, BranchVerdict)
            }
        return result

    @staticmethod
    def _normalize_ic_hints_by_symbol(
        payload: Mapping[str, Mapping[str, Any]] | None,
    ) -> dict[str, dict[str, Any]]:
        if payload is None:
            return {}
        if not isinstance(payload, Mapping):
            raise TypeError("ic_hints_by_symbol 必须是 Mapping[str, Mapping[str, Any]]")
        result: dict[str, dict[str, Any]] = {}
        for symbol, hint in payload.items():
            if not isinstance(hint, Mapping):
                continue
            result[str(symbol)] = dict(hint)
        return result

    @staticmethod
    def _position_mode_from_action(
        *,
        action: ActionLabel,
        risk_decision: RiskDecision,
        current_weight: float,
        symbol: str,
    ) -> str:
        if symbol in risk_decision.blocked_symbols:
            return "reject"
        if risk_decision.veto and risk_decision.gross_exposure_cap <= 0.0:
            return "reject"
        if action is ActionLabel.WATCH:
            return "watch"
        if action in {ActionLabel.AVOID, ActionLabel.SELL}:
            return "reject"
        if action is ActionLabel.HOLD and current_weight <= 0.0:
            return "watch"
        return "target"

    def _aggregate_risk_limits(
        self,
        *,
        constraints: Mapping[str, Any],
        risk_by_symbol: Mapping[str, RiskDecision],
        macro_verdict: BranchVerdict,
    ) -> dict[str, Any]:
        global_gross_cap = min(
            float(constraints.get("gross_exposure_cap", 1.0)),
            float(macro_verdict.metadata.get("target_gross_exposure", 1.0)),
        )
        max_weight = float(constraints.get("max_weight", 1.0))
        blocked_symbols: set[str] = set(str(symbol) for symbol in constraints.get("blocked_symbols", []))
        position_limits: dict[str, float] = {}
        global_caps: list[float] = [global_gross_cap]
        action_caps: list[ActionLabel] = []

        for symbol, decision in risk_by_symbol.items():
            action_caps.append(decision.action_cap)
            blocked_symbols.update(decision.blocked_symbols)
            symbol_limit = float(decision.position_limits.get(symbol, decision.max_weight))
            position_limits[symbol] = min(position_limits.get(symbol, 1.0), symbol_limit)
            if self._is_global_veto(decision, symbol):
                global_caps.append(float(decision.gross_exposure_cap))
                max_weight = min(max_weight, float(decision.max_weight))
            else:
                max_weight = min(max_weight, max(float(decision.max_weight), 0.0) or max_weight)

        if position_limits:
            max_weight = min(max_weight, max(position_limits.values()))

        aggregated_action_cap = ActionLabel.BUY
        for action_cap in action_caps:
            aggregated_action_cap = self.portfolio_constructor.more_restrictive_action(
                aggregated_action_cap,
                action_cap,
            )

        sector_caps = constraints.get("sector_caps", {})
        turnover_cap = constraints.get("turnover_cap")
        aggregated = {
            "gross_exposure_cap": max(0.0, min(global_caps) if global_caps else global_gross_cap),
            "max_weight": max(0.0, min(max_weight, 1.0)),
            "position_limits": position_limits,
            "blocked_symbols": sorted(blocked_symbols),
            "sector_caps": dict(sector_caps) if isinstance(sector_caps, Mapping) else {},
            "turnover_cap": turnover_cap,
            "action_cap": aggregated_action_cap.value,
        }
        return aggregated

    @staticmethod
    def _is_global_veto(decision: RiskDecision, symbol: str) -> bool:
        blocked = set(decision.blocked_symbols)
        if not decision.veto:
            return False
        return not blocked or blocked != {symbol}

    def _aggregate_branch_summaries(
        self,
        research_by_symbol: Mapping[str, Mapping[str, BranchVerdict]],
    ) -> dict[str, BranchVerdict]:
        per_branch: dict[str, list[BranchVerdict]] = {}
        for symbol_payload in research_by_symbol.values():
            for branch_name, verdict in symbol_payload.items():
                per_branch.setdefault(branch_name, []).append(verdict)

        aggregated: dict[str, BranchVerdict] = {}
        for branch_name, verdicts in per_branch.items():
            thesis = verdicts[0].thesis if verdicts else f"{branch_name} 分支已生成结构化结论。"
            mean_score = sum(item.final_score for item in verdicts) / max(len(verdicts), 1)
            mean_confidence = sum(item.final_confidence for item in verdicts) / max(len(verdicts), 1)
            risk_notes: list[str] = []
            coverage_notes: list[str] = []
            diagnostic_notes: list[str] = []
            for item in verdicts:
                risk_notes.extend(item.investment_risks)
                coverage_notes.extend(item.coverage_notes)
                diagnostic_notes.extend(item.diagnostic_notes)
            aggregated[branch_name] = BranchVerdict(
                agent_name=verdicts[0].agent_name if verdicts else branch_name,
                thesis=thesis,
                status=AgentStatus.SUCCESS
                if all(item.status == AgentStatus.SUCCESS for item in verdicts)
                else AgentStatus.DEGRADED,
                direction=self.portfolio_constructor.score_to_direction(mean_score),
                action=self.portfolio_constructor.score_to_action(mean_score),
                confidence_label=self.portfolio_constructor.confidence_to_label(mean_confidence),
                final_score=mean_score,
                final_confidence=mean_confidence,
                investment_risks=self._dedupe_texts(risk_notes),
                coverage_notes=self._dedupe_texts(coverage_notes),
                diagnostic_notes=self._dedupe_texts(diagnostic_notes),
                metadata={"symbol_count": len(verdicts)},
            )
        return aggregated

    def _aggregate_report_risk_decision(
        self,
        *,
        risk_by_symbol: Mapping[str, RiskDecision],
        aggregated_risk_limits: Mapping[str, Any],
    ) -> RiskDecision:
        action_cap_raw = aggregated_risk_limits.get("action_cap", ActionLabel.BUY.value)
        action_cap = (
            action_cap_raw if isinstance(action_cap_raw, ActionLabel)
            else ActionLabel(str(action_cap_raw).strip().lower())
        )
        veto = any(decision.veto for decision in risk_by_symbol.values())
        level_rank = {
            RiskLevel.LOW: 0,
            RiskLevel.MEDIUM: 1,
            RiskLevel.HIGH: 2,
            RiskLevel.EXTREME: 3,
        }
        risk_level = RiskLevel.LOW
        reasons: list[str] = []
        events: list[Any] = []
        for decision in risk_by_symbol.values():
            reasons.extend(decision.reasons)
            events.extend(decision.events)
            if level_rank.get(decision.risk_level, 0) > level_rank.get(risk_level, 0):
                risk_level = decision.risk_level

        gross_exposure_cap = float(aggregated_risk_limits.get("gross_exposure_cap", 1.0))
        is_restricted = gross_exposure_cap < 1.0 or action_cap is not ActionLabel.BUY
        return RiskDecision(
            status=(
                AgentStatus.VETOED
                if veto
                else AgentStatus.DEGRADED if is_restricted else AgentStatus.SUCCESS
            ),
            risk_level=risk_level,
            hard_veto=veto,
            veto=veto,
            action_cap=action_cap,
            max_weight=float(aggregated_risk_limits.get("max_weight", 1.0)),
            gross_exposure_cap=gross_exposure_cap,
            target_exposure_cap=gross_exposure_cap,
            blocked_symbols=list(aggregated_risk_limits.get("blocked_symbols", [])),
            position_limits=dict(aggregated_risk_limits.get("position_limits", {})),
            reasons=self._dedupe_texts(reasons) or ["未触发额外聚合风险约束。"],
            events=events,
            metadata={"aggregated_from_symbols": len(risk_by_symbol)},
        )

    @staticmethod
    def _build_run_diagnostics(
        research_by_symbol: Mapping[str, Mapping[str, BranchVerdict]],
    ) -> dict[str, list[str]]:
        diagnostics: list[str] = []
        for symbol_payload in research_by_symbol.values():
            for verdict in symbol_payload.values():
                diagnostics.extend(verdict.diagnostic_notes)
        return {"diagnostic_notes": ControlChainOrchestrator._dedupe_texts(diagnostics)}

    def _normalize_tradability_snapshot(
        self,
        data_bundle: UnifiedDataBundle,
        snapshot: Mapping[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        if isinstance(snapshot, Mapping):
            source = snapshot.get("symbols") if isinstance(snapshot.get("symbols"), Mapping) else snapshot
            result = {
                str(symbol): dict(info)
                for symbol, info in source.items()
                if isinstance(info, Mapping)
            }
        else:
            result = {}

        for symbol in data_bundle.symbols:
            result.setdefault(symbol, self._default_tradability_for_symbol(data_bundle, symbol))
        return result

    def _default_tradability_for_symbol(
        self,
        data_bundle: UnifiedDataBundle,
        symbol: str,
    ) -> dict[str, Any]:
        frame = data_bundle.symbol_data.get(symbol)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            avg_volume = float(frame.get("volume", pd.Series([1_000_000])).tail(20).mean() or 1_000_000)
            latest_close = float(frame.get("close", pd.Series([1.0])).iloc[-1] or 1.0)
            liquidity_score = min(max(avg_volume / 5_000_000, 0.2), 1.0)
            return {
                "is_tradable": True,
                "sector": str(data_bundle.fundamentals.get(symbol, {}).get("sector", "unknown")),
                "liquidity_score": liquidity_score,
                "latest_close": latest_close,
                "avg_volume_20d": avg_volume,
            }
        return {
            "is_tradable": True,
            "sector": str(data_bundle.fundamentals.get(symbol, {}).get("sector", "unknown")),
            "liquidity_score": 0.5,
        }

    def _build_market_snapshot(self, data_bundle: UnifiedDataBundle) -> dict[str, Any]:
        macro_data = dict(data_bundle.macro_data or {})
        combined = data_bundle.combined_frame()
        if not combined.empty and "close" in combined.columns:
            market_prices = combined.groupby("date")["close"].mean()
            market_return = market_prices.pct_change().dropna()
            recent_return = float(market_return.tail(20).mean()) if len(market_return) >= 5 else 0.0
            volatility = float(market_return.tail(20).std()) if len(market_return) >= 5 else 0.0
            macro_data.setdefault("macro_score", max(-1.0, min(1.0, recent_return * 20)))
            macro_data.setdefault("liquidity_score", max(-1.0, min(1.0, 0.35 - volatility * 10)))
            macro_data.setdefault(
                "volatility_percentile",
                max(0.0, min(100.0, 50.0 + volatility * 500)),
            )
        macro_data.setdefault("regime", macro_data.get("risk_level", "neutral"))
        macro_data.setdefault("policy_signal", macro_data.get("signal", "neutral"))
        return macro_data

    def _ensure_symbol_verdict(
        self,
        verdict: BranchVerdict,
        *,
        symbol: str,
        branch_name: str,
    ) -> BranchVerdict:
        if not isinstance(verdict, BranchVerdict):
            raise TypeError(f"{branch_name} agent 必须返回 BranchVerdict")
        payload = deepcopy(verdict)
        legacy_scores = payload.metadata.get("legacy_symbol_scores", {})
        symbol_score = float(legacy_scores.get(symbol, payload.final_score))
        payload.symbol = symbol
        payload.direction = self.portfolio_constructor.score_to_direction(symbol_score)
        payload.action = self.portfolio_constructor.score_to_action(symbol_score)
        payload.final_score = max(-1.0, min(1.0, symbol_score))
        payload.metadata = dict(payload.metadata)
        payload.metadata.update(
            {
                "symbol": symbol,
                "source_branch": branch_name,
                "symbol_score": symbol_score,
            }
        )
        for item in payload.evidence:
            item.scope = CoverageScope.SYMBOL
            item.symbols = [symbol]
            item.score = payload.final_score
        return payload

    @staticmethod
    def _macro_target_exposure_from_score(score: float) -> float:
        return max(0.1, min(1.0, 0.55 + float(score) * 0.35))

    @staticmethod
    def _style_bias_from_macro_score(score: float) -> str:
        if score <= -0.25:
            return "defensive_quality"
        if score >= 0.25:
            return "cyclical_growth"
        return "balanced_quality"

    @staticmethod
    def _dedupe_texts(values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    def _persist_outputs(
        self,
        *,
        macro_verdict: BranchVerdict,
        research_by_symbol: Mapping[str, Mapping[str, BranchVerdict]],
        risk_by_symbol: Mapping[str, RiskDecision],
        ic_by_symbol: Mapping[str, ICDecision],
        portfolio_plan: PortfolioPlan,
        report_bundle: ReportBundle,
        persist_dir: str | Path | None,
    ) -> dict[str, str]:
        base_dir = (
            Path(persist_dir)
            if persist_dir is not None
            else Path(tempfile.mkdtemp(prefix="quant_investor_agent_orchestrator_"))
        )
        base_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        files = {
            "macro_verdict": run_dir / "macro_verdict.json",
            "research_by_symbol": run_dir / "research_by_symbol.json",
            "risk_by_symbol": run_dir / "risk_by_symbol.json",
            "ic_by_symbol": run_dir / "ic_by_symbol.json",
            "portfolio_plan": run_dir / "portfolio_plan.json",
            "report_bundle": run_dir / "report_bundle.json",
            "markdown_report": run_dir / "report.md",
        }

        files["macro_verdict"].write_text(
            json.dumps(self._serialize(macro_verdict), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        files["research_by_symbol"].write_text(
            json.dumps(self._serialize(research_by_symbol), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        files["risk_by_symbol"].write_text(
            json.dumps(self._serialize(risk_by_symbol), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        files["ic_by_symbol"].write_text(
            json.dumps(self._serialize(ic_by_symbol), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        files["portfolio_plan"].write_text(
            json.dumps(self._serialize(portfolio_plan), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        files["report_bundle"].write_text(
            json.dumps(self._serialize(report_bundle), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        files["markdown_report"].write_text(report_bundle.markdown_report, encoding="utf-8")

        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    **self._version_info(),
                    "files": {key: str(path) for key, path in files.items()},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "run_dir": str(run_dir),
            "manifest": str(manifest_path),
            **{key: str(path) for key, path in files.items()},
        }

    def _serialize(self, value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if is_dataclass(value):
            return {
                key: self._serialize(item)
                for key, item in asdict(value).items()
            }
        if isinstance(value, Mapping):
            return {
                str(key): self._serialize(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [self._serialize(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        return value


AgentOrchestrator = ControlChainOrchestrator
