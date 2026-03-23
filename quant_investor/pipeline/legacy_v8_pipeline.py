"""
冻结的 V8 legacy 研究编排器。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_investor.branch_contracts import BranchResult, PortfolioStrategy, UnifiedDataBundle
from quant_investor.pipeline.parallel_research_pipeline import (
    BranchPerformanceTracker,
    ParallelResearchPipeline,
    _clamp,
    _safe_mean,
)
from quant_investor.versioning import (
    ARCHITECTURE_VERSION_V8,
    BRANCH_SCHEMA_VERSION_V8,
    LEGACY_BRANCH_ORDER,
    LEGACY_BRANCH_WEIGHTS,
)


class LegacyV8ParallelResearchPipeline(ParallelResearchPipeline):
    """V8 legacy frozen 分支编排器。"""

    BRANCH_ORDER = list(LEGACY_BRANCH_ORDER)
    ARCHITECTURE_VERSION = ARCHITECTURE_VERSION_V8
    BRANCH_SCHEMA_VERSION = BRANCH_SCHEMA_VERSION_V8

    def __init__(
        self,
        stock_pool: list[str],
        market: str = "CN",
        lookback_years: float = 1.0,
        total_capital: float = 1_000_000.0,
        risk_level: str = "中等",
        enable_alpha_mining: bool = True,
        enable_kline: bool = True,
        enable_intelligence: bool = True,
        enable_llm_debate: bool = True,
        enable_macro: bool = True,
        kline_backend: str = "heuristic",
        allow_synthetic_for_research: bool = False,
        verbose: bool = True,
        enable_kronos: bool | None = None,
    ) -> None:
        super().__init__(
            stock_pool=stock_pool,
            market=market,
            lookback_years=lookback_years,
            total_capital=total_capital,
            risk_level=risk_level,
            enable_alpha_mining=enable_alpha_mining,
            enable_quant=True,
            enable_kline=enable_kline,
            enable_fundamental=False,
            enable_intelligence=enable_intelligence,
            enable_branch_debate=False,
            enable_macro=enable_macro,
            kline_backend=kline_backend,
            allow_synthetic_for_research=allow_synthetic_for_research,
            enable_document_semantics=False,
            verbose=verbose,
            enable_kronos=enable_kronos,
            enable_llm_debate=enable_llm_debate,
        )
        self.enable_llm_debate = enable_llm_debate
        self.enable_branch_debate = False
        self.enable_fundamental = False
        self._branch_tracker = BranchPerformanceTracker(
            default_weights=LEGACY_BRANCH_WEIGHTS,
            architecture_version=self.architecture_version,
            branch_schema_version=self.branch_schema_version,
        )

    def _enabled_branch_flags(self) -> dict[str, bool]:
        return {
            "kline": self.enable_kline,
            "quant": True,
            "llm_debate": self.enable_llm_debate,
            "intelligence": self.enable_intelligence,
            "macro": self.enable_macro,
        }

    def _run_branches(self, data_bundle: UnifiedDataBundle) -> dict[str, BranchResult]:
        results: dict[str, BranchResult] = {}
        for name in self.BRANCH_ORDER:
            if not self._enabled_branch_flags().get(name, False):
                continue
            try:
                branch_method = getattr(self, f"_run_{name}_branch")
                branch_result = self._annotate_branch_result(branch_method(data_bundle))
            except Exception as exc:
                branch_result = self._degraded_branch_result(
                    branch_name=name,
                    explanation=f"{name} 分支异常，已降级为中性结果。",
                    degraded_reason=str(exc),
                    risks=[f"{name} 分支失败: {exc}"],
                )
            results[name] = branch_result
        return results

    def _run_llm_debate_branch(self, data_bundle: UnifiedDataBundle) -> BranchResult:
        """V8 legacy: 结构化多空辩论顶层分支。"""
        symbol_scores: dict[str, float] = {}
        bull_case: dict[str, list[str]] = {}
        bear_case: dict[str, list[str]] = {}
        key_risks: dict[str, list[str]] = {}

        for symbol in self.stock_pool:
            fundamentals = data_bundle.fundamentals.get(symbol, {})
            events = data_bundle.event_data.get(symbol, [])
            sentiment = data_bundle.sentiment_data.get(symbol, {})

            bullish_points: list[str] = []
            bearish_points: list[str] = []
            risks: list[str] = []
            score = 0.0

            roe = float(fundamentals.get("roe", 0.0))
            growth = float(fundamentals.get("profit_growth", fundamentals.get("revenue_growth", 0.0)))
            debt = float(fundamentals.get("debt_ratio", 0.4))
            event_impact = _safe_mean([float(item.get("impact", 0.0)) for item in events])
            fear_greed = float(sentiment.get("fear_greed", 0.0))

            if roe > 0.12:
                bullish_points.append("盈利能力较强，ROE 表现优于中性水平。")
                score += 0.25
            else:
                bearish_points.append("盈利能力尚未形成明显优势。")
                score -= 0.10

            if growth > 0.10:
                bullish_points.append("成长信号积极，利润或收入保持扩张。")
                score += 0.20
            elif growth < -0.05:
                bearish_points.append("成长动能偏弱，基本面改善需要更多验证。")
                score -= 0.20

            if debt > 0.55:
                risks.append("资产负债率偏高，需要关注偿债压力。")
                bearish_points.append("负债水平较高，财务弹性受限。")
                score -= 0.15

            if event_impact > 0.05:
                bullish_points.append("近期事件流偏正面，有利于估值修复。")
                score += 0.15
            elif event_impact < -0.05:
                bearish_points.append("近期事件流偏负面，可能压制风险偏好。")
                score -= 0.15

            if fear_greed > 0.2:
                bullish_points.append("市场情绪边际回暖。")
                score += 0.10
            elif fear_greed < -0.2:
                bearish_points.append("市场情绪仍偏谨慎。")
                score -= 0.10

            bull_case[symbol] = bullish_points or ["暂无强烈看多证据，维持中性观察。"]
            bear_case[symbol] = bearish_points or ["暂无明显看空论据。"]
            key_risks[symbol] = risks
            symbol_scores[symbol] = _clamp(score, -1.0, 1.0)

        return BranchResult(
            branch_name="llm_debate",
            score=_safe_mean(list(symbol_scores.values())),
            confidence=0.58,
            signals={
                "bull_case": bull_case,
                "bear_case": bear_case,
                "key_risks": key_risks,
                "llm_confidence": 0.58,
            },
            risks=["第一版为结构化研究辩论器，新闻与公告摘要仍以占位适配为主。"],
            explanation="LLM 辩论分支围绕基本面、行业、事件和情绪构建多空论据，但不直接负责最终仓位裁决。",
            symbol_scores=symbol_scores,
            metadata={
                "bull_case": bull_case,
                "bear_case": bear_case,
                "branch_mode": "structured_research_debate",
                "debate_mode": "structured_rules",
                "reliability": 0.58,
                "horizon_days": 10,
            },
        )

    def _run_intelligence_branch(self, data_bundle: UnifiedDataBundle) -> BranchResult:
        """V8 legacy intelligence: 仍含财务质量主分。"""
        financial_health: dict[str, float] = {}
        event_risk: dict[str, float] = {}
        sentiment_score: dict[str, float] = {}
        breadth_score: dict[str, float] = {}
        alerts: list[str] = []
        symbol_scores: dict[str, float] = {}

        advancing = 0
        declining = 0
        for symbol, df in data_bundle.symbol_data.items():
            fundamentals = data_bundle.fundamentals.get(symbol, {})
            sentiments = data_bundle.sentiment_data.get(symbol, {})
            events = data_bundle.event_data.get(symbol, [])

            roe = float(fundamentals.get("roe", 0.0))
            gross_margin = float(fundamentals.get("gross_margin", 0.0))
            debt_ratio = float(fundamentals.get("debt_ratio", 0.5))
            pe = float(fundamentals.get("pe", 15.0))
            pb = float(fundamentals.get("pb", 2.0))

            piotroski_like = (
                (1 if roe > 0.1 else 0)
                + (1 if gross_margin > 0.25 else 0)
                + (1 if debt_ratio < 0.45 else 0)
            ) / 3
            beneish_risk = 1.0 - _clamp((gross_margin * 2 + roe - debt_ratio), 0.0, 1.0)
            altman_like = _clamp(1.5 - debt_ratio + roe, 0.0, 1.0)
            valuation_score = _clamp((20 - pe) / 20 + (2.5 - pb) / 3, -1.0, 1.0)

            financial_score = _clamp(
                (piotroski_like - beneish_risk + altman_like + valuation_score) / 3,
                -1.0,
                1.0,
            )
            event_score = _clamp(
                -_safe_mean([abs(float(item.get("impact", 0.0))) for item in events]),
                -1.0,
                0.0,
            )
            senti = _clamp(
                0.5 * float(sentiments.get("fear_greed", 0.0))
                + 0.3 * float(sentiments.get("money_flow", 0.0))
                + 0.2 * float(sentiments.get("breadth", 0.0)),
                -1.0,
                1.0,
            )

            if df["close"].pct_change().iloc[-1] >= 0:
                advancing += 1
            else:
                declining += 1

            breadth = 0.0
            if advancing + declining > 0:
                breadth = (advancing - declining) / (advancing + declining)

            if debt_ratio > 0.65:
                alerts.append(f"{symbol} 负债率偏高，需结合现金流进一步验证。")
            if beneish_risk > 0.6:
                alerts.append(f"{symbol} 财务舞弊风险代理信号偏高。")

            financial_health[symbol] = financial_score
            event_risk[symbol] = event_score
            sentiment_score[symbol] = senti
            breadth_score[symbol] = breadth

            regime = self._market_regime
            if regime in {"趋势上涨", "趋势下跌"}:
                w_fin, w_sen, w_evt = 0.35, 0.40, 0.25
            elif regime == "震荡低波":
                w_fin, w_sen, w_evt = 0.55, 0.20, 0.25
            elif regime == "震荡高波":
                w_fin, w_sen, w_evt = 0.40, 0.25, 0.35
            else:
                w_fin, w_sen, w_evt = 0.50, 0.25, 0.25
            symbol_scores[symbol] = _clamp(
                w_fin * financial_score + w_sen * senti + w_evt * event_score,
                -1.0,
                1.0,
            )

        return BranchResult(
            branch_name="intelligence",
            score=_safe_mean(list(symbol_scores.values())),
            confidence=0.62,
            signals={
                "intelligence_score": symbol_scores,
                "financial_health_score": financial_health,
                "event_risk_score": event_risk,
                "sentiment_score": sentiment_score,
                "breadth_score": breadth_score,
                "alerts": alerts,
            },
            risks=alerts[:5],
            explanation="多维智能融合分支独立整合财务质量、事件风险、情绪、资金流和市场广度信号。",
            symbol_scores=symbol_scores,
            metadata={
                "financial_health_score": financial_health,
                "event_risk_score": event_risk,
                "sentiment_score": sentiment_score,
                "breadth_score": breadth_score,
                "branch_mode": "structured_intelligence_fusion",
                "reliability": 0.72,
                "horizon_days": 10,
            },
        )

    def _run_ensemble_layer(
        self,
        data_bundle: UnifiedDataBundle,
        branch_results: dict[str, BranchResult],
        risk_result: Any,
        calibrated_signals: dict[str, Any] | None = None,
    ) -> PortfolioStrategy:
        """V8 legacy 融合逻辑。"""
        if calibrated_signals is None:
            calibrated_signals = self._calibrate_signals(data_bundle, branch_results)
        branch_consensus = {
            name: round(signal.aggregate_expected_return, 4)
            for name, signal in calibrated_signals.items()
        }
        raw_symbol_convictions = self._aggregate_symbol_scores(calibrated_signals)
        expected_returns = self._aggregate_expected_returns(calibrated_signals)
        synthetic_symbols = set(data_bundle.synthetic_symbols())
        degraded_symbols = set(data_bundle.degraded_symbols())
        macro_overlay = self._macro_overlay_factor(calibrated_signals.get("macro"))
        candidate_symbols = [
            symbol
            for symbol, score in sorted(raw_symbol_convictions.items(), key=lambda item: item[1], reverse=True)
            if score > 0 and (self.allow_synthetic_for_research or symbol not in synthetic_symbols)
        ][: min(8, len(data_bundle.real_symbols()) or len(self.stock_pool))]

        research_score = _safe_mean(list(branch_consensus.values()))
        disagreement = np.std([branch.score for branch in branch_results.values()]) if branch_results else 0.0
        base_exposure = 1 - risk_result.position_sizing.cash_ratio
        exposure_penalty = _clamp(disagreement, 0.0, 0.35)
        target_exposure = _clamp(base_exposure * (1 - exposure_penalty), 0.1, 0.95)
        if synthetic_symbols and not self.allow_synthetic_for_research:
            target_exposure = min(target_exposure, 0.45)
        if risk_result.risk_level == "danger":
            target_exposure = min(target_exposure, 0.25)
        elif risk_result.risk_level == "warning":
            target_exposure = min(target_exposure, 0.55)
        if not data_bundle.real_symbols():
            target_exposure = 0.0
        target_exposure = _clamp(target_exposure * macro_overlay, 0.0, 0.95)

        quorum_max = getattr(self, "_quorum_max_exposure", 0.95)
        quorum_rel = getattr(self, "_quorum_reliability", 1.0)
        target_exposure = min(target_exposure, quorum_max)
        if quorum_rel <= 0.0:
            target_exposure = 0.0

        style_bias = self._infer_style_bias(branch_results, research_score)
        sector_preferences = self._infer_sector_preferences(style_bias, branch_results.get("macro"))
        stop_loss_policy = {
            "portfolio_drawdown_limit": abs(self.risk_layer.max_drawdown_limit),
            "default_stop_loss_pct": abs(self.risk_layer.stop_loss_pct),
            "default_take_profit_pct": self.risk_layer.take_profit_pct,
        }
        position_limits = self._optimize_positions(
            data_bundle=data_bundle,
            candidate_symbols=candidate_symbols,
            expected_returns=expected_returns,
            convictions=raw_symbol_convictions,
            target_exposure=target_exposure,
            fallback_weights=risk_result.position_sizing.risk_adjusted_weights,
        )

        execution_notes = [
            f"研究共识得分 {research_score:+.2f}，分支分歧度 {disagreement:.2f}。",
            f"当前风险等级为 {risk_result.risk_level}，目标总仓位调整为 {target_exposure:.0%}。",
        ]
        if synthetic_symbols:
            execution_notes.append(
                "存在降级/模拟数据标的，已默认排除其进入候选池。"
                if not self.allow_synthetic_for_research
                else "存在降级/模拟数据标的，当前以研究模式允许其保留。"
            )
        for branch in branch_results.values():
            execution_notes.extend(branch.risks[:1])
        research_mode = "production"
        research_only_reason = ""
        if quorum_rel <= 0.0:
            research_mode = "research_only"
            research_only_reason = "quorum_insufficient"
            target_exposure = 0.0
            execution_notes.append("分支成功数不足（≤1），进入 research_only 模式。")
        elif synthetic_symbols or any(not branch.success for branch in branch_results.values()):
            research_mode = "degraded"
        if not candidate_symbols:
            execution_notes.append("暂无明确优势标的，建议以现金和防御仓位为主。")
            if not data_bundle.real_symbols():
                research_mode = "research_only"
                research_only_reason = "no_real_candidates_after_provenance_filter"
                target_exposure = 0.0
                position_limits = {}

        trade_recommendations = self._build_trade_recommendations(
            data_bundle=data_bundle,
            branch_results=branch_results,
            calibrated_signals=calibrated_signals,
            risk_result=risk_result,
            candidate_symbols=candidate_symbols,
            position_limits=position_limits,
            expected_returns=expected_returns,
            convictions=raw_symbol_convictions,
        )

        return PortfolioStrategy(
            **self._version_payload(),
            target_exposure=target_exposure,
            style_bias=style_bias,
            sector_preferences=sector_preferences,
            candidate_symbols=candidate_symbols,
            position_limits=position_limits,
            stop_loss_policy=stop_loss_policy,
            execution_notes=execution_notes[:8],
            branch_consensus=branch_consensus,
            symbol_convictions=raw_symbol_convictions,
            risk_summary={
                "risk_level": risk_result.risk_level,
                "volatility": risk_result.risk_metrics.volatility,
                "max_drawdown": risk_result.risk_metrics.max_drawdown,
                "warnings": risk_result.risk_warnings,
                "planned_investment": self.total_capital * target_exposure,
                "cash_reserve": self.total_capital * (1 - target_exposure),
                "max_single_position": self.risk_layer.max_position_size,
            },
            provenance_summary={
                "synthetic_symbols": sorted(synthetic_symbols),
                "degraded_symbols": sorted(degraded_symbols),
                "branch_modes": {
                    name: str(signal.branch_mode) for name, signal in calibrated_signals.items()
                },
                "branch_reliability": {
                    name: float(signal.reliability) for name, signal in calibrated_signals.items()
                },
                "macro_overlay": macro_overlay,
                "research_only_reason": research_only_reason,
                "symbol_provenance": data_bundle.symbol_provenance(),
                **self._version_payload(),
            },
            research_mode=research_mode,
            trade_recommendations=trade_recommendations,
        )

    def _collect_symbol_risks(
        self,
        symbol: str,
        data_bundle: UnifiedDataBundle,
        branch_results: dict[str, BranchResult],
        branch_positive_count: int,
        active_branch_count: int,
        confidence: float,
        expected_drawdown: float,
        trend_regime: str,
    ) -> list[str]:
        risk_flags = super()._collect_symbol_risks(
            symbol=symbol,
            data_bundle=data_bundle,
            branch_results=branch_results,
            branch_positive_count=branch_positive_count,
            active_branch_count=active_branch_count,
            confidence=confidence,
            expected_drawdown=expected_drawdown,
            trend_regime=trend_regime,
        )
        llm_branch = branch_results.get("llm_debate")
        if llm_branch is not None:
            llm_risks = llm_branch.signals.get("key_risks", {}).get(symbol, [])
            risk_flags.extend(str(item) for item in llm_risks[:2])
        return list(dict.fromkeys(risk_flags))

    def _infer_style_bias(
        self,
        branch_results: dict[str, BranchResult],
        research_score: float,
    ) -> str:
        quant_score = branch_results.get("quant", BranchResult("quant")).score
        intelligence_score = branch_results.get("intelligence", BranchResult("intelligence")).score
        macro_score = branch_results.get("macro", BranchResult("macro")).score
        if research_score < -0.2 or macro_score < -0.2:
            return "防御"
        if intelligence_score > 0.2 and quant_score < 0:
            return "高质量"
        if quant_score > 0.2:
            return "成长"
        return "均衡"

    @staticmethod
    def _default_branch_mode(branch_name: str) -> str:
        return {
            "kline": "kline_heuristic",
            "quant": "alpha_research",
            "llm_debate": "structured_research_debate",
            "intelligence": "structured_intelligence_fusion",
            "macro": "macro_terminal",
        }.get(branch_name, "unknown")

    def _build_markdown_report(self, result: Any) -> str:
        report = super()._build_markdown_report(result)
        return report.replace("# V9 五路并行研究投资策略报告", "# V8 Legacy Frozen 投资策略报告", 1)
