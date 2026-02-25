#!/usr/bin/env python3
"""
统一版本集成 MacroRiskTerminal V6.3
多市场宏观风控终端
"""

import sys
import os
from typing import Optional

# 添加 unified 路径
unified_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if unified_dir not in sys.path:
    sys.path.insert(0, unified_dir)

# 导入 MacroRiskTerminal
from macro_risk_terminal import (
    create_terminal, detect_market,
    CNMacroRiskTerminal, USMacroRiskTerminal,
    RiskTerminalReport
)

from pipeline import MasterPipelineUnified, UnifiedReport


class UnifiedWithMacroRisk:
    """
    统一版本 + 宏观风控终端集成
    
    将宏观风控信号与量化投资流程结合
    """
    
    def __init__(
        self,
        market: str = "US",
        stock_pool: Optional[list] = None,
        lookback_years: int = 1,
        enable_macro_risk: bool = True,
        verbose: bool = True
    ):
        self.market = market
        self.stock_pool = stock_pool
        self.lookback_years = lookback_years
        self.enable_macro_risk = enable_macro_risk
        self.verbose = verbose
        
        # 初始化量化流水线
        self.quant_pipeline = MasterPipelineUnified(
            market=market,
            stock_pool=stock_pool,
            lookback_years=lookback_years,
            verbose=verbose
        )
        
        # 初始化宏观风控终端
        self.macro_terminal = None
        if enable_macro_risk:
            try:
                self.macro_terminal = create_terminal(market=market)
                if verbose:
                    print(f"[MacroRisk] {market} 宏观风控终端已加载")
            except Exception as e:
                if verbose:
                    print(f"[MacroRisk] 加载失败: {e}")
    
    def run(self) -> dict:
        """
        执行完整分析 (量化 + 宏观)
        
        Returns:
            包含量化报告和宏观风控报告的字典
        """
        results = {
            'quant_report': None,
            'macro_report': None,
            'combined_signal': None,
            'final_recommendation': None
        }
        
        # 1. 运行量化分析
        print("\n" + "="*60)
        print("Step 1: 量化投资分析")
        print("="*60)
        quant_report = self.quant_pipeline.run()
        results['quant_report'] = quant_report
        
        # 2. 运行宏观风控分析
        if self.macro_terminal:
            print("\n" + "="*60)
            print("Step 2: 宏观风控分析")
            print("="*60)
            try:
                macro_report = self.macro_terminal.generate_risk_report()
                results['macro_report'] = macro_report
                
                # 打印宏观报告
                print(self.macro_terminal.format_report_markdown(macro_report))
            except Exception as e:
                print(f"宏观风控分析失败: {e}")
        
        # 3. 综合信号
        results['combined_signal'] = self._combine_signals(
            quant_report, results['macro_report']
        )
        
        results['final_recommendation'] = self._generate_final_recommendation(
            results['combined_signal']
        )
        
        return results
    
    def _combine_signals(self, quant_report, macro_report) -> dict:
        """综合量化和宏观信号"""
        signal = {
            'quant_signal': 'neutral',
            'macro_signal': 'neutral',
            'combined': 'neutral',
            'risk_level': 'medium'
        }
        
        # 量化信号
        if quant_report and quant_report.risk_output:
            if quant_report.risk_output.portfolio:
                signal['quant_signal'] = 'bullish'
        
        # 宏观信号
        if macro_report:
            signal['macro_signal'] = macro_report.overall_signal
            signal['risk_level'] = macro_report.overall_risk_level
        
        # 综合判断
        if signal['macro_signal'] in ['🔴', '高风险']:
            signal['combined'] = 'high_risk'
        elif signal['macro_signal'] in ['🟢', '🔵', '低风险', '极低风险']:
            signal['combined'] = 'favorable'
        else:
            signal['combined'] = 'neutral'
        
        return signal
    
    def _generate_final_recommendation(self, signal: dict) -> str:
        """生成最终建议"""
        if signal['combined'] == 'high_risk':
            return "宏观风险较高，建议降低仓位，以防御为主"
        elif signal['combined'] == 'favorable':
            return "宏观环境有利，可积极配置推荐标的"
        else:
            return "宏观环境中性，保持正常仓位，精选个股"


# 便捷函数
def analyze_with_macro_risk(
    market: str = "US",
    stocks: Optional[list] = None,
    **kwargs
) -> dict:
    """
    带宏观风控的量化分析
    
    示例:
        results = analyze_with_macro_risk(
            market="US",
            stocks=["AAPL", "MSFT", "NVDA"]
        )
        print(results['final_recommendation'])
    """
    analyzer = UnifiedWithMacroRisk(
        market=market,
        stock_pool=stocks,
        **kwargs
    )
    return analyzer.run()


if __name__ == '__main__':
    print("="*70)
    print("Quant-Investor Unified + MacroRiskTerminal V6.3")
    print("="*70)
    
    # 测试美股分析
    results = analyze_with_macro_risk(
        market="US",
        stocks=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
        lookback_years=0.5,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("最终建议")
    print("="*70)
    print(results['final_recommendation'])
    
    if results['combined_signal']:
        print(f"\n综合信号: {results['combined_signal']}")
