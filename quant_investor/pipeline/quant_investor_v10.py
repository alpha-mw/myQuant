"""
Quant Investor V10.0 Multi-Agent 入口。

V10 = V9 量化管线 + Multi-Agent IC Enhancement Layer。
Agent 层是后置增强，不替换任何 V9 计算逻辑。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from quant_investor.agents.agent_contracts import AgentEnhancedStrategy
from quant_investor.agents.orchestrator import AgentOrchestrator
from quant_investor.logger import get_logger
from quant_investor.pipeline.quant_investor_v9 import QuantInvestorV9, V9PipelineResult
from quant_investor.versioning import ARCHITECTURE_VERSION_V10, AGENT_SCHEMA_VERSION

_logger = get_logger("QuantInvestorV10")


@dataclass
class V10PipelineResult(V9PipelineResult):
    """V10 完整流水线结果：V9 结果 + Agent 增强策略。"""

    agent_enhanced_strategy: Optional[AgentEnhancedStrategy] = None
    agent_layer_enabled: bool = False
    agent_schema_version: str = AGENT_SCHEMA_VERSION


class QuantInvestorV10(QuantInvestorV9):
    """V10 = V9 + Multi-Agent IC Enhancement Layer。"""

    def __init__(
        self,
        stock_pool: list[str],
        market: str = "CN",
        lookback_years: float = 1.0,
        total_capital: float = 1_000_000.0,
        risk_level: str = "中等",
        enable_macro: bool = True,
        enable_backtest: bool = False,
        enable_alpha_mining: bool = True,
        enable_quant: bool = True,
        enable_kline: bool = True,
        enable_fundamental: bool = True,
        enable_intelligence: bool = True,
        enable_branch_debate: bool = False,
        kline_backend: str = "hybrid",
        allow_synthetic_for_research: bool = False,
        debate_top_k: int = 3,
        debate_min_abs_score: float = 0.08,
        debate_timeout_sec: float = 8.0,
        debate_model: str = "gpt-5.4-mini",
        enable_document_semantics: bool = True,
        verbose: bool = True,
        enable_kronos: bool | None = None,
        enable_llm_debate: bool | None = None,
        # V10 agent layer params
        enable_agent_layer: bool = True,
        agent_model: str = "",
        master_model: str = "",
        agent_timeout: float = 15.0,
        master_timeout: float = 30.0,
        agent_total_timeout: float = 120.0,
    ) -> None:
        super().__init__(
            stock_pool=stock_pool,
            market=market,
            lookback_years=lookback_years,
            total_capital=total_capital,
            risk_level=risk_level,
            enable_macro=enable_macro,
            enable_backtest=enable_backtest,
            enable_alpha_mining=enable_alpha_mining,
            enable_quant=enable_quant,
            enable_kline=enable_kline,
            enable_fundamental=enable_fundamental,
            enable_intelligence=enable_intelligence,
            enable_branch_debate=enable_branch_debate,
            kline_backend=kline_backend,
            allow_synthetic_for_research=allow_synthetic_for_research,
            debate_top_k=debate_top_k,
            debate_min_abs_score=debate_min_abs_score,
            debate_timeout_sec=debate_timeout_sec,
            debate_model=debate_model,
            enable_document_semantics=enable_document_semantics,
            verbose=verbose,
            enable_kronos=enable_kronos,
            enable_llm_debate=enable_llm_debate,
        )
        self.enable_agent_layer = enable_agent_layer
        self.agent_model = agent_model
        self.master_model = master_model
        self.agent_timeout = agent_timeout
        self.master_timeout = master_timeout
        self.agent_total_timeout = agent_total_timeout

    def run(self) -> V10PipelineResult:
        t0 = time.time()
        self._log("=" * 60)
        self._log("🚀 Quant Investor V10.0 启动（Multi-Agent IC Architecture）")
        self._log(f"标的: {self.stock_pool}")
        self._log(f"Agent 层: {'启用' if self.enable_agent_layer else '禁用'}")
        if self.enable_agent_layer:
            self._log(f"分支模型: {self.agent_model or '(未指定)'}")
            self._log(f"IC 模型: {self.master_model or '(未指定)'}")
        self._log("=" * 60)

        # Step 1: Run full V9 pipeline
        v9_result = super().run()

        # Step 2: Build V10 result from V9
        v10_result = V10PipelineResult(
            architecture_version=ARCHITECTURE_VERSION_V10,
            branch_schema_version=v9_result.branch_schema_version,
            calibration_schema_version=v9_result.calibration_schema_version,
            debate_template_version=v9_result.debate_template_version,
            data_bundle=v9_result.data_bundle,
            branch_results=v9_result.branch_results,
            calibrated_signals=v9_result.calibrated_signals,
            risk_results=v9_result.risk_results,
            final_strategy=v9_result.final_strategy,
            final_report=v9_result.final_report,
            execution_log=list(v9_result.execution_log),
            layer_timings=dict(v9_result.layer_timings),
            raw_data=v9_result.raw_data,
            factor_data=v9_result.factor_data,
            model_predictions=v9_result.model_predictions,
            macro_signal=v9_result.macro_signal,
            macro_summary=v9_result.macro_summary,
            v7_decision=v9_result.v7_decision,
            llm_ensemble_results=v9_result.llm_ensemble_results,
            agent_layer_enabled=self.enable_agent_layer,
        )

        # Step 3: Run agent enhancement layer
        if self.enable_agent_layer and self.agent_model and self.master_model:
            self._log("🤖 启动 Multi-Agent IC 层...")
            t_agent = time.time()
            try:
                orchestrator = AgentOrchestrator(
                    branch_model=self.agent_model,
                    master_model=self.master_model,
                    timeout_per_agent=self.agent_timeout,
                    master_timeout=self.master_timeout,
                    total_timeout=self.agent_total_timeout,
                )

                # Reconstruct ensemble output from final_strategy
                ensemble_output: dict[str, Any] = {}
                if hasattr(v9_result.final_strategy, "branch_consensus"):
                    ensemble_output["branch_consensus"] = v9_result.final_strategy.branch_consensus
                    # Compute aggregate_score from branch consensus
                    scores = v9_result.final_strategy.branch_consensus
                    if scores:
                        ensemble_output["aggregate_score"] = sum(scores.values()) / len(scores)
                    else:
                        ensemble_output["aggregate_score"] = 0.0
                else:
                    ensemble_output["aggregate_score"] = 0.0

                # Detect market regime from macro branch
                market_regime = "default"
                macro_br = v9_result.branch_results.get("macro")
                if macro_br and hasattr(macro_br, "signals") and isinstance(macro_br.signals, dict):
                    market_regime = str(macro_br.signals.get("regime", "default"))

                agent_result = orchestrator.enhance_sync(
                    branch_results=v9_result.branch_results,
                    calibrated_signals=v9_result.calibrated_signals,
                    risk_result=v9_result.risk_results,
                    ensemble_output=ensemble_output,
                    data_bundle=v9_result.data_bundle,
                    market_regime=market_regime,
                    algorithmic_strategy=v9_result.final_strategy,
                )
                v10_result.agent_enhanced_strategy = agent_result
                agent_time = time.time() - t_agent
                v10_result.layer_timings["agent_layer"] = agent_time

                if agent_result.agent_layer_success:
                    self._log(f"✅ IC 层完成，耗时 {agent_time:.1f}s")
                    self._log(f"   IC 最终研判: {agent_result.agent_strategy.final_conviction}")
                    self._log(f"   IC 分数: {agent_result.agent_strategy.final_score:.3f}")
                    # Append IC narrative to report
                    v10_result.final_report += self._build_agent_report_section(agent_result)
                else:
                    self._log(f"⚠️ IC 层降级（fallback），耗时 {agent_time:.1f}s")
            except Exception as exc:
                agent_time = time.time() - t_agent
                self._log(f"❌ IC 层异常: {exc}，耗时 {agent_time:.1f}s")
                v10_result.layer_timings["agent_layer"] = agent_time
        elif self.enable_agent_layer:
            self._log("⚠️ Agent 层已启用但未指定模型（agent_model/master_model），跳过")

        v10_result.total_time = time.time() - t0
        self._log(f"✅ V10 分析完成，总耗时 {v10_result.total_time:.1f}s")
        return v10_result

    @staticmethod
    def _build_agent_report_section(agent_result: AgentEnhancedStrategy) -> str:
        """将 IC 层结果追加到 markdown 报告。"""
        if not agent_result.agent_strategy:
            return ""

        ic = agent_result.agent_strategy
        lines = [
            "",
            "---",
            "",
            "## 🤖 IC 投资委员会研判（V10 Agent Layer）",
            "",
            f"**最终研判**: {ic.final_conviction}  |  **分数**: {ic.final_score:.3f}  |  **置信度**: {ic.confidence:.2f}",
            f"**风险调整后敞口**: {ic.risk_adjusted_exposure:.1%}",
            "",
        ]

        if ic.portfolio_narrative:
            lines.append(f"### 投资论点")
            lines.append(ic.portfolio_narrative)
            lines.append("")

        if ic.consensus_areas:
            lines.append("### IC 共识")
            for item in ic.consensus_areas:
                lines.append(f"- {item}")
            lines.append("")

        if ic.disagreement_areas:
            lines.append("### IC 分歧")
            for item in ic.disagreement_areas:
                lines.append(f"- {item}")
            lines.append("")

        if ic.debate_resolution:
            lines.append("### 分歧调解")
            for item in ic.debate_resolution:
                lines.append(f"- {item}")
            lines.append("")

        if ic.top_picks:
            lines.append("### 精选标的")
            for pick in ic.top_picks:
                lines.append(f"- **{pick.symbol}** ({pick.action}) — {pick.rationale}")
            lines.append("")

        if ic.dissenting_views:
            lines.append("### 少数派意见")
            for item in ic.dissenting_views:
                lines.append(f"- {item}")
            lines.append("")

        # Branch agent summaries
        if agent_result.branch_agent_outputs:
            lines.append("### 各分支 SubAgent 摘要")
            for name, output in agent_result.branch_agent_outputs.items():
                if output is None:
                    lines.append(f"- **{name}**: ❌ 未完成")
                    continue
                lines.append(f"- **{name}**: {output.conviction} (score={output.conviction_score:.3f})")
                if output.key_insights:
                    for insight in output.key_insights[:3]:
                        lines.append(f"  - {insight}")
            lines.append("")

        # Risk agent summary
        risk = agent_result.risk_agent_output
        if risk:
            lines.append("### 风控 SubAgent 评估")
            lines.append(f"- 风险等级: **{risk.risk_assessment}**")
            lines.append(f"- 建议最大敞口: {risk.max_recommended_exposure:.1%}")
            if risk.risk_warnings:
                for w in risk.risk_warnings[:3]:
                    lines.append(f"- ⚠️ {w}")
            lines.append("")

        # Timings
        if agent_result.agent_layer_timings:
            lines.append("### Agent 层耗时")
            for k, v in agent_result.agent_layer_timings.items():
                lines.append(f"- {k}: {v:.1f}s")
            lines.append("")

        return "\n".join(lines)
