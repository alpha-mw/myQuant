#!/usr/bin/env python3
"""
Decision Layer - 决策层 (第6层)

功能:
1. 多模型多空辩论 - 5个专业分析模型
2. 深度公司研究 - 产品、竞争格局、行业趋势、政策
3. 综合1-5层所有信息
4. 生成具体投资建议 - 决策、仓位、目标价、止损

使用 multi_model_debate.py 中的 MultiModelDebateSystem
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from multi_model_debate import MultiModelDebateSystem, DebateResult, InvestmentDecision


@dataclass
class DecisionLayerResult:
    """决策层结果"""
    debate_results: List[DebateResult] = field(default_factory=list)
    investment_decisions: List[InvestmentDecision] = field(default_factory=list)
    portfolio_allocation: Dict[str, float] = field(default_factory=dict)
    market_outlook: str = ""
    final_report: str = ""


class DecisionLayer:
    """
    决策层 - LLM多模型多空辩论
    """
    
    def __init__(self, api_key: Optional[str] = None, verbose: bool = True):
        self.debate_system = MultiModelDebateSystem(verbose=verbose)
        self.verbose = verbose
        self.result = DecisionLayerResult()
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[DecisionLayer] {msg}")
    
    def run_decision_process(
        self,
        symbols: List[str],
        quant_data: Dict[str, Dict],
        macro_data: Dict,
        risk_data: Dict
    ) -> DecisionLayerResult:
        """
        运行完整决策流程
        
        对每只股票进行多模型多空辩论，生成投资决策
        """
        self._log("=" * 80)
        self._log("【第6层】决策层 - LLM多模型多空辩论")
        self._log("=" * 80)
        
        result = DecisionLayerResult()
        
        # 对每只股票进行辩论
        for symbol in symbols:
            self._log(f"\n分析股票: {symbol}")
            
            # 获取该股票的数据
            symbol_quant = quant_data.get(symbol, {})
            
            # 执行多模型辩论
            debate_result = self.debate_system.conduct_debate(
                symbol=symbol,
                quant_data=symbol_quant,
                macro_data=macro_data,
                risk_data=risk_data
            )
            
            result.debate_results.append(debate_result)
            result.investment_decisions.append(debate_result.investment_decision)
            
            self._log(f"决策: {debate_result.investment_decision.decision} "
                     f"(置信度{debate_result.investment_decision.confidence:.0%})")
        
        # 生成组合配置
        result.portfolio_allocation = self._generate_portfolio_allocation(
            result.investment_decisions,
            macro_data
        )
        
        # 生成市场展望
        result.market_outlook = self._generate_market_outlook(
            result.investment_decisions,
            macro_data
        )
        
        # 生成最终报告
        result.final_report = self._generate_comprehensive_report(result)
        
        self._log("\n决策层完成")
        
        return result
    
    def _generate_portfolio_allocation(
        self,
        decisions: List[InvestmentDecision],
        macro_data: Dict
    ) -> Dict[str, float]:
        """生成组合配置"""
        # 基于宏观信号调整基础仓位
        base_position = {
            "🔴": 0.3,
            "🟡": 0.5,
            "🟢": 0.8,
            "🔵": 1.0
        }.get(macro_data.get('signal', '🟡'), 0.5)
        
        # 筛选买入建议
        buy_decisions = [d for d in decisions 
                        if d.decision in ["强烈买入", "买入"]]
        
        if not buy_decisions:
            return {"CASH": 1.0}
        
        # 按置信度和建议仓位加权
        total_weight = sum(d.confidence * d.position_size for d in buy_decisions)
        
        allocation = {}
        for d in buy_decisions:
            weight = (d.confidence * d.position_size / total_weight) * base_position
            allocation[d.symbol] = min(weight, 0.2)  # 单票不超过20%
        
        # 归一化
        total = sum(allocation.values())
        if total > 0:
            allocation = {k: v/total * base_position for k, v in allocation.items()}
        
        cash_ratio = 1 - sum(allocation.values())
        if cash_ratio > 0:
            allocation["CASH"] = cash_ratio
        
        return allocation
    
    def _generate_market_outlook(
        self,
        decisions: List[InvestmentDecision],
        macro_data: Dict
    ) -> str:
        """生成市场展望"""
        strong_buy = sum(1 for d in decisions if d.decision == "强烈买入")
        buy = sum(1 for d in decisions if d.decision == "买入")
        hold = sum(1 for d in decisions if d.decision == "持有")
        sell = sum(1 for d in decisions if d.decision in ["卖出", "强烈卖出"])
        
        if strong_buy >= 2 or buy >= 3:
            outlook = "积极看多，精选优质标的"
        elif sell >= 2:
            outlook = "防御为主，降低仓位"
        elif hold >= len(decisions) * 0.5:
            outlook = "震荡市，均衡配置"
        else:
            outlook = "结构性机会，精选个股"
        
        # 结合宏观信号
        macro_signal = macro_data.get('signal', '🟡')
        if macro_signal == "🔴":
            outlook += " | 宏观环境不利，严格控制风险"
        elif macro_signal == "🟢":
            outlook += " | 宏观环境支持，可适当积极"
        
        return outlook
    
    def _generate_comprehensive_report(self, result: DecisionLayerResult) -> str:
        """生成综合报告"""
        lines = []
        
        lines.append("# 🎯 量化投资决策报告")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # 市场展望
        lines.append("## 📊 市场展望")
        lines.append(result.market_outlook)
        lines.append("")
        
        # 组合配置
        lines.append("## 💼 组合配置建议")
        lines.append("| 标的 | 配置比例 |")
        lines.append("|:---|:---:|")
        for symbol, weight in sorted(result.portfolio_allocation.items(), 
                                     key=lambda x: x[1], reverse=True):
            lines.append(f"| {symbol} | {weight:.1%} |")
        lines.append("")
        
        # 个股决策
        lines.append("## 📈 个股投资决策")
        lines.append("")
        
        for decision in result.investment_decisions:
            emoji = {
                "强烈买入": "🔥",
                "买入": "🟢",
                "持有": "🟡",
                "卖出": "🔴",
                "强烈卖出": "⛔"
            }.get(decision.decision, "⚪")
            
            lines.append(f"### {emoji} {decision.symbol}")
            lines.append(f"- **决策**: {decision.decision}")
            lines.append(f"- **置信度**: {decision.confidence:.0%}")
            lines.append(f"- **建议仓位**: {decision.position_size:.0%}")
            if decision.target_price:
                lines.append(f"- **目标价**: ¥{decision.target_price:.2f}")
            if decision.stop_loss:
                lines.append(f"- **止损位**: ¥{decision.stop_loss:.2f}")
            lines.append(f"- **投资周期**: {decision.time_horizon}")
            
            lines.append("\n**决策逻辑**:")
            for logic in decision.logic_chain[:3]:
                lines.append(f"- {logic}")
            
            lines.append("\n**模型共识**:")
            for model, bias in decision.model_consensus.items():
                lines.append(f"- {model}: {bias}")
            
            lines.append("")
        
        # 风险提示
        lines.append("## ⚠️ 风险提示")
        all_risks = []
        for d in result.investment_decisions:
            all_risks.extend(d.opposing_concerns)
        
        unique_risks = list(set(all_risks))[:5]
        for risk in unique_risks:
            lines.append(f"- {risk}")
        
        return "\n".join(lines)


# ==================== 测试 ====================

if __name__ == '__main__':
    print("=" * 80)
    print("Decision Layer - 测试")
    print("=" * 80)
    
    decision_layer = DecisionLayer(verbose=True)
    
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    quant_data = {
        "AAPL": {"predicted_return": 0.15, "predicted_volatility": 0.25, "sharpe_ratio": 1.2, "factors": ["momentum", "value"]},
        "MSFT": {"predicted_return": 0.12, "predicted_volatility": 0.22, "sharpe_ratio": 1.1, "factors": ["quality", "growth"]},
        "GOOGL": {"predicted_return": 0.08, "predicted_volatility": 0.28, "sharpe_ratio": 0.8, "factors": ["value"]}
    }
    
    macro_data = {"signal": "🟡", "risk_level": "中风险"}
    risk_data = {"risk_level": "normal", "volatility": 0.25}
    
    result = decision_layer.run_decision_process(
        symbols=symbols,
        quant_data=quant_data,
        macro_data=macro_data,
        risk_data=risk_data
    )
    
    print("\n" + "=" * 80)
    print("最终报告")
    print("=" * 80)
    print(result.final_report)
