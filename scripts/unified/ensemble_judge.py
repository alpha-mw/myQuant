"""
Ensemble Judge Layer — 集成裁判层
=====================================
V10 架构终端决策层：汇聚五大分支信号，经过市场状态感知与动态权重
调整，输出可直接执行的投资建议。

输入（来自五大并行分支 + 风控层）：
  branch_kronos      → KronosBranchResult
  branch_quant       → QuantBranchResult
  branch_debate      → DebateBranchResult
  branch_intelligence→ IntelligenceBranchResult
  branch_macro       → MacroBranchResult
  risk_result        → RiskLayerResult

输出：
  EnsembleJudgment   每只股票的最终投资裁定 + 组合配置
  EnsembleReport     Markdown综合报告

市场状态感知（RegimeDetector）：
  TREND_UP / TREND_DOWN / VOLATILE / RANGE_BOUND / CRISIS
  — 各状态下自动调整分支权重，例如：
    危机状态 → 风控层权重上调，Kronos/宏观为主，LLM辩论降权
    趋势行情 → Kronos预测 + 量化因子为主
    震荡行情 → 基本面/财务/新闻信号更可靠

集成算法：
  1. 每个分支输出归一化 [-1, +1] 方向信号
  2. 按市场状态加权求和 → 综合得分
  3. 置信度 = 各分支方向一致性比率
  4. 宏观层作为"一票否决"条件（极端风险时强制降仓）
  5. 风控层对最终仓位施加硬约束
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from logger import get_logger

_logger = get_logger("EnsembleJudge")


# ---------------------------------------------------------------------------
# 市场状态枚举
# ---------------------------------------------------------------------------

class MarketRegime(Enum):
    TREND_UP    = "趋势上行"
    TREND_DOWN  = "趋势下行"
    VOLATILE    = "高波动"
    RANGE_BOUND = "震荡盘整"
    CRISIS      = "极端风险"


# ---------------------------------------------------------------------------
# 分支信号标准化结构
# ---------------------------------------------------------------------------

@dataclass
class BranchSignal:
    """单个分支归一化后的信号"""
    branch_name: str
    score: float           # [-1, +1]，负=看空，正=看多
    confidence: float      # [0, 1]
    label: str             # "强烈买入/买入/中性/卖出/强烈卖出"
    raw: Any = None        # 原始分支结果

    def is_valid(self) -> bool:
        return not (np.isnan(self.score) or np.isnan(self.confidence))


@dataclass
class StockJudgment:
    """单只股票的最终裁定"""
    symbol: str
    stock_name: str = ""

    # 各分支信号
    branch_signals: list[BranchSignal] = field(default_factory=list)

    # 集成结果
    ensemble_score: float = 0.0       # [-1, +1]
    ensemble_confidence: float = 0.0
    decision: str = "中性"             # 强烈买入/买入/中性/卖出/强烈卖出
    suggested_weight: float = 0.0      # 建议仓位权重 [0, 1]
    risk_adjusted_weight: float = 0.0  # 风控约束后仓位

    # 市场状态
    market_regime: MarketRegime = MarketRegime.RANGE_BOUND
    regime_override: bool = False      # 宏观/风控一票否决

    # 解释
    key_drivers: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class EnsembleJudgment:
    """全组合集成裁定"""
    stock_judgments: dict[str, StockJudgment] = field(default_factory=dict)
    portfolio_allocation: dict[str, float] = field(default_factory=dict)
    market_regime: MarketRegime = MarketRegime.RANGE_BOUND
    overall_market_score: float = 0.0
    total_invested_pct: float = 0.0
    cash_pct: float = 1.0
    report: str = ""
    elapsed: float = 0.0


# ---------------------------------------------------------------------------
# 市场状态检测器
# ---------------------------------------------------------------------------

class RegimeDetector:
    """
    根据宏观信号 + 价格动量判断当前市场状态。
    当宏观数据不可用时退化为价格动量判断。
    """

    # 各市场状态下的分支权重
    REGIME_WEIGHTS: dict[MarketRegime, dict[str, float]] = {
        MarketRegime.TREND_UP: {
            "kronos": 0.30, "quant": 0.25, "debate": 0.15,
            "intelligence": 0.15, "macro": 0.15,
        },
        MarketRegime.TREND_DOWN: {
            "kronos": 0.25, "quant": 0.20, "debate": 0.15,
            "intelligence": 0.15, "macro": 0.25,
        },
        MarketRegime.VOLATILE: {
            "kronos": 0.20, "quant": 0.15, "debate": 0.15,
            "intelligence": 0.20, "macro": 0.30,
        },
        MarketRegime.RANGE_BOUND: {
            "kronos": 0.20, "quant": 0.20, "debate": 0.20,
            "intelligence": 0.25, "macro": 0.15,
        },
        MarketRegime.CRISIS: {
            "kronos": 0.15, "quant": 0.10, "debate": 0.10,
            "intelligence": 0.15, "macro": 0.50,
        },
    }

    def detect(
        self,
        macro_result: Any = None,
        portfolio_volatility: float = 0.0,
        market_trend_5d: float = 0.0,
    ) -> MarketRegime:
        """检测市场状态"""
        # 危机判断：宏观极端风险或波动率极高
        macro_risk = self._extract_macro_risk(macro_result)
        if macro_risk == "extreme" or portfolio_volatility > 0.04:
            return MarketRegime.CRISIS

        # 高波动判断
        if portfolio_volatility > 0.025:
            if market_trend_5d > 0.01:
                return MarketRegime.TREND_UP
            elif market_trend_5d < -0.01:
                return MarketRegime.TREND_DOWN
            return MarketRegime.VOLATILE

        # 趋势判断（低波动）
        if market_trend_5d > 0.008:
            return MarketRegime.TREND_UP
        if market_trend_5d < -0.008:
            return MarketRegime.TREND_DOWN

        return MarketRegime.RANGE_BOUND

    @staticmethod
    def _extract_macro_risk(macro_result: Any) -> str:
        if macro_result is None:
            return "normal"
        # 尝试从宏观结果中提取风险级别
        risk_level = getattr(macro_result, "risk_level", None)
        if risk_level is None:
            risk_level = getattr(macro_result, "macro_risk_level", "normal")
        level_str = str(risk_level).lower()
        if any(x in level_str for x in ["极高", "极端", "extreme", "crisis", "high_risk"]):
            return "extreme"
        if any(x in level_str for x in ["高", "high", "elevated"]):
            return "high"
        return "normal"

    def get_weights(self, regime: MarketRegime) -> dict[str, float]:
        return dict(self.REGIME_WEIGHTS[regime])


# ---------------------------------------------------------------------------
# 信号标准化工具
# ---------------------------------------------------------------------------

class SignalNormalizer:
    """把各分支的异构结果统一成 BranchSignal"""

    # 决策文本 → 方向数值映射
    LABEL_SCORE: dict[str, float] = {
        "强烈买入": 1.0, "strong_buy": 1.0,
        "买入": 0.6,     "buy": 0.6,
        "中性": 0.0,     "neutral": 0.0, "持有": 0.0, "hold": 0.0,
        "卖出": -0.6,    "sell": -0.6,
        "强烈卖出": -1.0, "strong_sell": -1.0,
    }

    def from_kronos(self, kronos_result: Any, symbol: str) -> BranchSignal:
        """标准化 Kronos 分支结果"""
        if kronos_result is None:
            return BranchSignal("kronos", 0.0, 0.0, "中性")

        try:
            # IntelligenceLayerResult 中的 kronos 子结果
            per_stock = getattr(kronos_result, "per_stock_signals", {})
            sig = per_stock.get(symbol)
            if sig:
                kronos_part = getattr(sig, "kronos_signal", None)
                if kronos_part:
                    score = float(getattr(kronos_part, "score", 0.0))
                    conf = float(getattr(kronos_part, "confidence", 0.5))
                    label = self._score_to_label(score)
                    return BranchSignal("kronos", score, conf, label, kronos_part)

            # 直接 KronosPortfolioSignal
            stock_forecasts = getattr(kronos_result, "stock_forecasts", {})
            fc = stock_forecasts.get(symbol)
            if fc:
                direction = getattr(fc, "direction", "中性")
                conf = float(getattr(fc, "confidence", 0.5))
                score = self.LABEL_SCORE.get(direction, 0.0)
                return BranchSignal("kronos", score, conf, direction, fc)

        except Exception:
            pass
        return BranchSignal("kronos", 0.0, 0.3, "中性")

    def from_quant(self, quant_result: Any, symbol: str) -> BranchSignal:
        """标准化传统量化分支结果"""
        if quant_result is None:
            return BranchSignal("quant", 0.0, 0.0, "中性")
        try:
            # QuantPipelineResult: model_predictions 是 pd.Series indexed by symbol
            predictions = getattr(quant_result, "model_predictions", None)
            if predictions is not None and hasattr(predictions, "get"):
                raw_score = predictions.get(symbol, 0.0)
                if raw_score is not None:
                    score = float(np.clip(raw_score, -1, 1))
                    label = self._score_to_label(score)
                    return BranchSignal("quant", score, 0.6, label, quant_result)

            # 从因子数据中提取
            factor_data = getattr(quant_result, "factor_data", None)
            if factor_data is not None and symbol in str(factor_data):
                return BranchSignal("quant", 0.0, 0.4, "中性")

        except Exception:
            pass
        return BranchSignal("quant", 0.0, 0.3, "中性")

    def from_debate(self, debate_result: Any, symbol: str) -> BranchSignal:
        """标准化 LLM 多空辩论结果"""
        if debate_result is None:
            return BranchSignal("debate", 0.0, 0.0, "中性")
        try:
            # DecisionLayerResult → InvestmentDecision per symbol
            decisions = getattr(debate_result, "investment_decisions", [])
            for d in decisions:
                if getattr(d, "symbol", "") == symbol:
                    dec_str = getattr(d, "decision", "持有")
                    conf = float(getattr(d, "confidence", 0.5))
                    score = self.LABEL_SCORE.get(dec_str, 0.0)
                    return BranchSignal("debate", score, conf, dec_str, d)

            # DebateResult 直接
            debates = getattr(debate_result, "debate_results", [])
            for dr in debates:
                if getattr(dr, "symbol", "") == symbol:
                    final = getattr(dr, "final_decision", None)
                    if final:
                        dec_str = getattr(final, "decision", "持有")
                        conf = float(getattr(final, "confidence", 0.5))
                        score = self.LABEL_SCORE.get(dec_str, 0.0)
                        return BranchSignal("debate", score, conf, dec_str, dr)

        except Exception:
            pass
        return BranchSignal("debate", 0.0, 0.3, "中性")

    def from_intelligence(self, intel_result: Any, symbol: str) -> BranchSignal:
        """标准化多维智能融合结果"""
        if intel_result is None:
            return BranchSignal("intelligence", 0.0, 0.0, "中性")
        try:
            per_stock = getattr(intel_result, "per_stock_signals", {})
            sig = per_stock.get(symbol)
            if sig:
                score = float(getattr(sig, "combined_score", 0.0))
                score = float(np.clip(score * 2 - 1, -1, 1))  # [0,1] → [-1,+1]
                conf = float(getattr(sig, "confidence", 0.5))
                label = getattr(sig, "signal_label", self._score_to_label(score))
                return BranchSignal("intelligence", score, conf, label, sig)

        except Exception:
            pass
        return BranchSignal("intelligence", 0.0, 0.3, "中性")

    def from_macro(self, macro_result: Any) -> BranchSignal:
        """标准化宏观分支结果（不针对单股，输出整体市场方向）"""
        if macro_result is None:
            return BranchSignal("macro", 0.0, 0.0, "中性")
        try:
            # MacroRiskTerminal → macro_signal
            signal_str = (
                getattr(macro_result, "macro_signal", None) or
                getattr(macro_result, "signal", "中性")
            )
            score_map = {
                "强烈看多": 0.8, "看多": 0.5, "中性": 0.0,
                "看空": -0.5, "强烈看空": -0.8,
                "bullish": 0.5, "bearish": -0.5, "neutral": 0.0,
            }
            score = score_map.get(str(signal_str), 0.0)
            conf = 0.7  # 宏观信号通常较高可信度
            risk_level = RegimeDetector._extract_macro_risk(macro_result)
            if risk_level == "extreme":
                score = min(score, -0.5)  # 极端风险强制偏空
            return BranchSignal("macro", score, conf, str(signal_str), macro_result)

        except Exception:
            pass
        return BranchSignal("macro", 0.0, 0.3, "中性")

    @staticmethod
    def _score_to_label(score: float) -> str:
        if score >= 0.7:
            return "强烈买入"
        if score >= 0.3:
            return "买入"
        if score <= -0.7:
            return "强烈卖出"
        if score <= -0.3:
            return "卖出"
        return "中性"


# ---------------------------------------------------------------------------
# 核心 Ensemble 引擎
# ---------------------------------------------------------------------------

class EnsembleJudgeEngine:
    """
    五分支信号集成裁判引擎。

    Parameters
    ----------
    max_single_position : 单股最大仓位（默认0.25）
    min_confidence      : 最低置信度阈值，低于此不建仓（默认0.35）
    macro_veto_threshold: 宏观一票否决得分（得分<=此值时强制清仓）
    custom_weights      : 覆盖默认的市场状态权重 {"kronos":..., "quant":...}
    """

    def __init__(
        self,
        max_single_position: float = 0.25,
        min_confidence: float = 0.35,
        macro_veto_threshold: float = -0.6,
        custom_weights: Optional[dict[str, float]] = None,
    ) -> None:
        self.max_single_position = max_single_position
        self.min_confidence = min_confidence
        self.macro_veto_threshold = macro_veto_threshold
        self.custom_weights = custom_weights

        self._regime_detector = RegimeDetector()
        self._normalizer = SignalNormalizer()

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def judge(
        self,
        stock_pool: list[str],
        stock_names: Optional[dict[str, str]] = None,
        *,
        kronos_result: Any = None,
        quant_result: Any = None,
        debate_result: Any = None,
        intelligence_result: Any = None,
        macro_result: Any = None,
        risk_result: Any = None,
        portfolio_volatility: float = 0.0,
        market_trend_5d: float = 0.0,
    ) -> EnsembleJudgment:
        """执行全量集成裁判，返回 EnsembleJudgment"""
        t0 = time.time()
        names = stock_names or {}

        # 1. 检测市场状态
        regime = self._regime_detector.detect(
            macro_result=macro_result,
            portfolio_volatility=portfolio_volatility,
            market_trend_5d=market_trend_5d,
        )
        weights = self.custom_weights or self._regime_detector.get_weights(regime)
        _logger.info(f"[EnsembleJudge] 市场状态={regime.value}  权重={weights}")

        # 2. 宏观一票否决检查
        macro_signal = self._normalizer.from_macro(macro_result)
        macro_veto = macro_signal.score <= self.macro_veto_threshold

        # 3. 逐股裁定
        judgments: dict[str, StockJudgment] = {}
        for symbol in stock_pool:
            j = self._judge_single(
                symbol=symbol,
                stock_name=names.get(symbol, symbol),
                regime=regime,
                weights=weights,
                macro_signal=macro_signal,
                macro_veto=macro_veto,
                kronos_result=kronos_result,
                quant_result=quant_result,
                debate_result=debate_result,
                intelligence_result=intelligence_result,
                risk_result=risk_result,
            )
            judgments[symbol] = j

        # 4. 组合配置
        allocation = self._allocate_portfolio(judgments, risk_result)

        # 5. 整体市场打分
        overall_score = float(np.mean([
            j.ensemble_score for j in judgments.values()
        ])) if judgments else 0.0

        # 6. 生成报告
        result = EnsembleJudgment(
            stock_judgments=judgments,
            portfolio_allocation=allocation,
            market_regime=regime,
            overall_market_score=overall_score,
            total_invested_pct=sum(allocation.values()),
            cash_pct=max(0.0, 1.0 - sum(allocation.values())),
            elapsed=round(time.time() - t0, 2),
        )
        result.report = self._generate_report(result, weights)
        return result

    # ------------------------------------------------------------------
    # 单股裁定
    # ------------------------------------------------------------------

    def _judge_single(
        self,
        symbol: str,
        stock_name: str,
        regime: MarketRegime,
        weights: dict[str, float],
        macro_signal: BranchSignal,
        macro_veto: bool,
        kronos_result: Any,
        quant_result: Any,
        debate_result: Any,
        intelligence_result: Any,
        risk_result: Any,
    ) -> StockJudgment:
        n = self._normalizer
        j = StockJudgment(symbol=symbol, stock_name=stock_name, market_regime=regime)

        # 收集各分支信号
        signals = {
            "kronos":       n.from_kronos(kronos_result, symbol),
            "quant":        n.from_quant(quant_result, symbol),
            "debate":       n.from_debate(debate_result, symbol),
            "intelligence": n.from_intelligence(intelligence_result, symbol),
            "macro":        macro_signal,
        }
        j.branch_signals = list(signals.values())

        # 加权集成
        total_w = 0.0
        ensemble_score = 0.0
        valid_signals: list[BranchSignal] = []

        for name, sig in signals.items():
            if not sig.is_valid():
                continue
            w = weights.get(name, 0.0)
            ensemble_score += w * sig.score * sig.confidence
            total_w += w * sig.confidence
            valid_signals.append(sig)

        if total_w > 0:
            ensemble_score /= total_w
        j.ensemble_score = float(np.clip(ensemble_score, -1.0, 1.0))

        # 方向一致性 = 置信度
        if valid_signals:
            direction_positive = sum(1 for s in valid_signals if s.score > 0.1)
            direction_negative = sum(1 for s in valid_signals if s.score < -0.1)
            n_valid = len(valid_signals)
            majority = max(direction_positive, direction_negative)
            j.ensemble_confidence = majority / n_valid if n_valid > 0 else 0.0
        else:
            j.ensemble_confidence = 0.0

        # 宏观一票否决
        if macro_veto:
            j.regime_override = True
            j.ensemble_score = min(j.ensemble_score, -0.3)
            j.risk_flags.append(f"宏观一票否决（宏观得分={macro_signal.score:.2f}）")

        # 决策标签
        j.decision = SignalNormalizer._score_to_label(j.ensemble_score)

        # 建议仓位：低置信度 → 不建仓
        if j.ensemble_confidence < self.min_confidence or j.ensemble_score <= 0:
            j.suggested_weight = 0.0
        else:
            base_w = (j.ensemble_score + 1) / 2  # [0, 1]
            j.suggested_weight = float(np.clip(
                base_w * j.ensemble_confidence * self.max_single_position,
                0.0,
                self.max_single_position,
            ))

        # 风控硬约束
        j.risk_adjusted_weight = self._apply_risk_constraints(
            symbol, j.suggested_weight, risk_result
        )

        # 关键驱动因素
        j.key_drivers = self._extract_key_drivers(signals, weights)

        # 综合摘要
        j.summary = (
            f"{stock_name}({symbol}): {j.decision} | "
            f"综合得分={j.ensemble_score:.2f} | 置信度={j.ensemble_confidence:.0%} | "
            f"建议仓位={j.risk_adjusted_weight:.1%}"
        )
        return j

    # ------------------------------------------------------------------
    # 组合配置
    # ------------------------------------------------------------------

    def _allocate_portfolio(
        self,
        judgments: dict[str, StockJudgment],
        risk_result: Any,
    ) -> dict[str, float]:
        """
        归一化仓位分配：
        1. 过滤掉评级为中性/卖出/强烈卖出的股票
        2. 按风控调整后权重分配
        3. 总仓位受宏观信号约束
        """
        raw: dict[str, float] = {
            sym: j.risk_adjusted_weight
            for sym, j in judgments.items()
            if j.risk_adjusted_weight > 0.01
        }
        if not raw:
            return {}

        total = sum(raw.values())
        # 最大总仓位：宏观一票否决时上限60%
        any_veto = any(j.regime_override for j in judgments.values())
        max_total = 0.60 if any_veto else 0.95

        if total > max_total:
            scale = max_total / total
            raw = {k: v * scale for k, v in raw.items()}

        return {k: round(v, 4) for k, v in raw.items()}

    @staticmethod
    def _apply_risk_constraints(
        symbol: str, weight: float, risk_result: Any
    ) -> float:
        """将风控层的仓位约束施加到建议权重上"""
        if risk_result is None:
            return weight
        try:
            pos_sizing = getattr(risk_result, "position_sizing", None)
            if pos_sizing:
                risk_weights = getattr(pos_sizing, "risk_adjusted_weights", {})
                cap = risk_weights.get(symbol, weight)
                return float(min(weight, cap))
        except Exception:
            pass
        return weight

    @staticmethod
    def _extract_key_drivers(
        signals: dict[str, BranchSignal],
        weights: dict[str, float],
    ) -> list[str]:
        """提取对最终决策贡献最大的分支"""
        contributions = []
        for name, sig in signals.items():
            if sig.is_valid() and abs(sig.score) > 0.1:
                w = weights.get(name, 0.0)
                contrib = w * abs(sig.score) * sig.confidence
                contributions.append((contrib, name, sig.label))
        contributions.sort(reverse=True)
        drivers = []
        for _, name, label in contributions[:3]:
            branch_cn = {
                "kronos": "Kronos图形预测", "quant": "传统量化因子",
                "debate": "LLM多空辩论",  "intelligence": "多维情报融合",
                "macro": "宏观环境",
            }.get(name, name)
            drivers.append(f"{branch_cn}({label})")
        return drivers

    # ------------------------------------------------------------------
    # 报告生成
    # ------------------------------------------------------------------

    def _generate_report(
        self, result: EnsembleJudgment, weights: dict[str, float]
    ) -> str:
        lines = [
            "# Ensemble Judge Layer — 集成裁判报告\n\n",
            f"**市场状态**: {result.market_regime.value}  ",
            f"**市场综合得分**: {result.overall_market_score:+.2f}  ",
            f"**总投资仓位**: {result.total_invested_pct:.1%}  ",
            f"**现金比例**: {result.cash_pct:.1%}\n\n",
            "## 分支权重（当前市场状态）\n\n",
            "| 分支 | 权重 |\n|------|------|\n",
        ]
        branch_cn = {
            "kronos": "Branch 1: Kronos图形预测",
            "quant": "Branch 2: 传统量化因子",
            "debate": "Branch 3: LLM多空辩论",
            "intelligence": "Branch 4: 多维情报融合",
            "macro": "Branch 5: 宏观环境",
        }
        for k, v in weights.items():
            lines.append(f"| {branch_cn.get(k, k)} | {v:.0%} |\n")

        lines.append("\n## 个股裁定汇总\n\n")
        lines.append(
            "| 股票 | 综合得分 | 置信度 | 决策 | 建议仓位 | 关键驱动 |\n"
            "|------|---------|--------|------|---------|----------|\n"
        )
        for sym, j in result.stock_judgments.items():
            override_flag = " ⚠️" if j.regime_override else ""
            drivers_str = " / ".join(j.key_drivers[:2]) if j.key_drivers else "—"
            lines.append(
                f"| {j.stock_name}({sym}){override_flag} "
                f"| {j.ensemble_score:+.2f} "
                f"| {j.ensemble_confidence:.0%} "
                f"| {j.decision} "
                f"| {j.risk_adjusted_weight:.1%} "
                f"| {drivers_str} |\n"
            )

        lines.append("\n## 各分支信号明细\n\n")
        for sym, j in result.stock_judgments.items():
            lines.append(f"\n### {j.stock_name}({sym})\n\n")
            lines.append("| 分支 | 信号 | 得分 | 置信度 |\n|------|------|------|--------|\n")
            for sig in j.branch_signals:
                cn = branch_cn.get(sig.branch_name, sig.branch_name)
                lines.append(
                    f"| {cn} | {sig.label} | {sig.score:+.2f} | {sig.confidence:.0%} |\n"
                )
            if j.risk_flags:
                lines.append(f"\n**风险警示**: {'; '.join(j.risk_flags)}\n")

        lines.append("\n## 组合配置\n\n")
        lines.append("| 股票 | 配置权重 |\n|------|----------|\n")
        for sym, w in result.portfolio_allocation.items():
            name = result.stock_judgments.get(sym, StockJudgment(sym)).stock_name or sym
            lines.append(f"| {name}({sym}) | {w:.1%} |\n")
        lines.append(f"| **现金** | **{result.cash_pct:.1%}** |\n")

        lines.append(f"\n*集成耗时: {result.elapsed}s*\n")
        return "".join(lines)
