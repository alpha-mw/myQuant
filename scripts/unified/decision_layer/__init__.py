#!/usr/bin/env python3
"""
统一决策层
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class DecisionOutput:
    """决策层输出"""
    ratings: List[Dict] = field(default_factory=list)
    analysis: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionLayerResult:
    """决策层结果 - 新架构"""
    debate_results: List = field(default_factory=list)
    investment_decisions: List = field(default_factory=list)
    stock_recommendations: List = field(default_factory=list)  # 添加这个属性
    portfolio_allocation: Dict[str, float] = field(default_factory=dict)
    market_outlook: str = ""
    final_report: str = ""


class DecisionLayer:
    """决策层 - 简化版，避免循环导入"""
    
    def __init__(self, api_key: Optional[str] = None, verbose: bool = True):
        self.verbose = verbose
        self.result = DecisionLayerResult()
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[DecisionLayer] {msg}")
    
    def run_decision_process(self, symbols, quant_data, macro_data, risk_data):
        """运行决策流程 - 简化版"""
        self._log("=" * 60)
        self._log("【第6层】决策层 - 生成投资建议")
        
        # 生成简化报告
        report_lines = []
        report_lines.append("# 投资决策报告")
        report_lines.append(f"**分析标的**: {', '.join(symbols)}")
        report_lines.append("")
        
        # 基于宏观信号生成建议
        macro_signal = macro_data.get('signal', '🟡')
        if macro_signal == '🔴':
            outlook = "宏观高风险，防御为主"
        elif macro_signal == '🟢':
            outlook = "宏观低风险，积极布局"
        else:
            outlook = "宏观中风险，精选个股"
        
        report_lines.append(f"**市场展望**: {outlook}")
        report_lines.append("")
        
        report_lines.append("## 个股建议")
        for symbol in symbols:
            report_lines.append(f"- {symbol}: 建议关注")
        
        self.result.final_report = "\n".join(report_lines)
        self.result.market_outlook = outlook
        
        return self.result


class UnifiedDecisionLayer:
    """统一决策层 - 兼容旧接口"""
    
    def __init__(self, llm_preference: Optional[List[str]] = None, verbose: bool = True):
        self.llm_preference = llm_preference or ["openai"]
        self.verbose = verbose
        
    def _log(self, msg: str):
        if self.verbose:
            print(f"  [DecisionLayer] {msg}")
    
    def process(self, ranked_stocks: List[Dict], data_bundle: Any) -> DecisionOutput:
        """处理决策 - 兼容旧接口"""
        output = DecisionOutput()
        
        self._log("生成投资建议...")
        
        for stock in ranked_stocks[:5]:
            output.ratings.append({
                'stock': stock.get('code'),
                'rating': '买入' if stock.get('composite_score', 0) > 0 else '持有',
                'score': stock.get('composite_score', 0),
                'reason': '基于多因子模型评分'
            })
        
        output.analysis = f"分析了 {len(ranked_stocks)} 只股票，推荐关注前 5 名"
        self._log(f"生成 {len(output.ratings)} 条投资评级")
        
        return output


__all__ = ['DecisionLayer', 'DecisionLayerResult', 'UnifiedDecisionLayer', 'DecisionOutput']
