#!/usr/bin/env python3
"""
Quant-Investor Unified v7.0 + MacroRiskTerminal V6.3 Enhanced
完全基于指标体系文档的集成版本
"""

import sys
import os
from typing import Optional, Dict, Any, List

# 添加路径
unified_dir = os.path.dirname(os.path.abspath(__file__))
if unified_dir not in sys.path:
    sys.path.insert(0, unified_dir)

# 导入新的 V6.3 宏观风控终端
from macro_terminal_v63 import (
    create_terminal, detect_market,
    CNMacroRiskTerminal, USMacroRiskTerminal,
    RiskTerminalReport, MacroRiskTerminalBase
)

# 导入量化流水线
from pipeline import MasterPipelineUnified, UnifiedReport


class UnifiedWithMacroV63:
    """
    统一版本 + MacroRiskTerminal V6.3 完整集成
    
    基于完整指标体系文档实现
    """
    
    VERSION = "7.0.0-unified-v6.3"
    
    def __init__(
        self,
        market: str = "US",
        stock_pool: Optional[List[str]] = None,
        lookback_years: float = 1.0,
        enable_macro_risk: bool = True,
        macro_weight: float = 0.3,  # 宏观信号权重
        verbose: bool = True
    ):
        self.market = market.upper()
        self.stock_pool = stock_pool
        self.lookback_years = lookback_years
        self.enable_macro_risk = enable_macro_risk
        self.macro_weight = macro_weight
        self.verbose = verbose
        
        # 初始化量化流水线
        self.quant_pipeline = MasterPipelineUnified(
            market=self.market,
            stock_pool=stock_pool,
            lookback_years=lookback_years,
            verbose=verbose
        )
        
        # 初始化宏观风控终端
        self.macro_terminal: Optional[MacroRiskTerminalBase] = None
        if enable_macro_risk:
            try:
                self.macro_terminal = create_terminal(market=self.market)
                if verbose:
                    print(f"[MacroRisk V6.3] {self.market} 宏观风控终端已加载")
                    print(f"[MacroRisk V6.3] 指标体系版本: 完整版")
            except Exception as e:
                if verbose:
                    print(f"[MacroRisk V6.3] 加载失败: {e}")
    
    def run(self) -> Dict[str, Any]:
        """
        执行完整分析
        
        Returns:
            包含量化报告、宏观报告、综合判断的字典
        """
        results = {
            'version': self.VERSION,
            'market': self.market,
            'quant_report': None,
            'macro_report': None,
            'macro_markdown': None,
            'combined_analysis': None,
            'final_signal': None,
            'position_advice': None,
            'strategy_adjustment': None
        }
        
        # Step 1: 量化分析
        if self.verbose:
            print("\n" + "="*70)
            print("Step 1: 量化投资分析")
            print("="*70)
        
        quant_report = self.quant_pipeline.run()
        results['quant_report'] = quant_report
        
        # Step 2: 宏观风控分析
        if self.macro_terminal:
            if self.verbose:
                print("\n" + "="*70)
                print("Step 2: 宏观风控分析 (V6.3 完整指标体系)")
                print("="*70)
            
            try:
                macro_report = self.macro_terminal.generate_risk_report()
                results['macro_report'] = macro_report
                results['macro_markdown'] = self.macro_terminal.format_report_markdown(macro_report)
                
                if self.verbose:
                    print(results['macro_markdown'])
            except Exception as e:
                if self.verbose:
                    print(f"宏观风控分析失败: {e}")
        
        # Step 3: 综合分析
        results['combined_analysis'] = self._combine_analysis(
            quant_report, results['macro_report']
        )
        
        results['final_signal'] = results['combined_analysis']['signal']
        results['position_advice'] = results['combined_analysis']['position']
        results['strategy_adjustment'] = results['combined_analysis']['strategy']
        
        return results
    
    def _combine_analysis(self, quant_report: UnifiedReport, 
                         macro_report: Optional[RiskTerminalReport]) -> Dict[str, str]:
        """
        综合量化和宏观分析
        
        基于指标体系文档的综合风控信号规则
        """
        analysis = {
            'quant_signal': 'neutral',
            'macro_signal': 'neutral',
            'macro_modules': {},
            'signal': '🟡 中风险',
            'position': '50% 仓位',
            'strategy': '控制仓位，精选个股',
            'detail': ''
        }
        
        # 量化信号判断
        if quant_report and quant_report.risk_output:
            if quant_report.risk_output.portfolio:
                # 根据组合风险特征判断
                port = quant_report.risk_output.portfolio
                if port.volatility > 0.30:
                    analysis['quant_signal'] = 'high_risk'
                elif port.volatility < 0.20:
                    analysis['quant_signal'] = 'low_risk'
                else:
                    analysis['quant_signal'] = 'medium'
        
        # 宏观信号判断 (基于指标体系文档)
        if macro_report:
            analysis['macro_signal'] = macro_report.overall_signal
            analysis['macro_modules'] = {
                m.module_name_en: m.overall_signal for m in macro_report.modules
            }
            
            # 使用宏观终端的综合判断
            if macro_report.overall_risk_level == "高风险":
                analysis['signal'] = '🔴 高风险'
                analysis['position'] = '≤30% 仓位'
                analysis['strategy'] = '防御为主，优先现金和低波动资产'
            elif macro_report.overall_risk_level == "中风险":
                analysis['signal'] = '🟡 中风险'
                analysis['position'] = '30%-60% 仓位'
                analysis['strategy'] = '控制仓位，精选高质量个股'
            elif macro_report.overall_risk_level == "低风险":
                analysis['signal'] = '🟢 低风险'
                analysis['position'] = '60%-90% 仓位'
                analysis['strategy'] = '正常配置，积极布局成长股'
            elif macro_report.overall_risk_level == "极低风险":
                analysis['signal'] = '🔵 极低风险'
                analysis['position'] = '80%-100% 仓位'
                analysis['strategy'] = '加大配置，逆向布局超跌优质股'
        
        # 生成详细说明
        details = []
        if quant_report:
            details.append(f"量化: {len(quant_report.data_bundle.stock_universe)}只股票")
        if macro_report:
            red_modules = [n for n, s in analysis['macro_modules'].items() if s == '🔴']
            yellow_modules = [n for n, s in analysis['macro_modules'].items() if s == '🟡']
            if red_modules:
                details.append(f"宏观风险模块: {', '.join(red_modules)}")
            if yellow_modules:
                details.append(f"宏观注意模块: {', '.join(yellow_modules)}")
        
        analysis['detail'] = '; '.join(details)
        
        return analysis
    
    def generate_full_report(self, results: Dict[str, Any]) -> str:
        """生成完整的Markdown报告"""
        lines = []
        
        lines.append("# Quant-Investor 统一版投资分析报告")
        lines.append(f"**版本**: {self.VERSION}")
        lines.append(f"**市场**: {self.market}")
        lines.append(f"**时间**: {results['quant_report'].timestamp if results['quant_report'] else ''}")
        lines.append("")
        
        # 综合结论
        lines.append("## 🎯 综合结论")
        lines.append("")
        lines.append(f"**风控信号**: {results['final_signal']}")
        lines.append(f"**仓位建议**: {results['position_advice']}")
        lines.append(f"**策略调整**: {results['strategy_adjustment']}")
        if results['combined_analysis'].get('detail'):
            lines.append(f"**分析详情**: {results['combined_analysis']['detail']}")
        lines.append("")
        
        # 量化分析摘要
        if results['quant_report']:
            quant = results['quant_report']
            lines.append("## 📊 量化分析摘要")
            lines.append("")
            lines.append(f"- **分析标的**: {len(quant.data_bundle.stock_universe)} 只股票")
            lines.append(f"- **有效因子**: {len(quant.factor_output.effective_factors) if quant.factor_output else 0} 个")
            lines.append(f"- **模型排名**: {len(quant.model_output.ranked_stocks) if quant.model_output else 0} 只")
            if quant.risk_output and quant.risk_output.portfolio:
                port = quant.risk_output.portfolio
                lines.append(f"- **组合配置**: {port.weights}")
                lines.append(f"- **预期收益**: {port.expected_return*100:.2f}%")
                lines.append(f"- **预期波动**: {port.volatility*100:.2f}%")
            lines.append("")
        
        # 宏观风控报告
        if results['macro_markdown']:
            lines.append("## 🌍 宏观风控分析")
            lines.append("")
            lines.append(results['macro_markdown'])
            lines.append("")
        
        return "\n".join(lines)


# ==================== 便捷函数 ====================

def analyze_complete(
    market: str = "US",
    stocks: Optional[List[str]] = None,
    lookback_years: float = 1.0,
    enable_macro: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    完整分析函数 - 一键获取量化+宏观分析
    
    示例:
        results = analyze_complete(
            market="US",
            stocks=["AAPL", "MSFT", "NVDA"],
            lookback_years=0.5
        )
        
        print(results['final_signal'])
        print(results['position_advice'])
        
        # 导出完整报告
        with open('report.md', 'w') as f:
            f.write(analyzer.generate_full_report(results))
    """
    analyzer = UnifiedWithMacroV63(
        market=market,
        stock_pool=stocks,
        lookback_years=lookback_years,
        enable_macro_risk=enable_macro,
        verbose=verbose
    )
    return analyzer.run()


# ==================== 主程序 ====================

if __name__ == '__main__':
    print("="*70)
    print("Quant-Investor Unified v7.0 + MacroRiskTerminal V6.3")
    print("完整指标体系集成版")
    print("="*70)
    
    # 美股示例
    print("\n【美股分析示例】")
    results = analyze_complete(
        market="US",
        stocks=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
        lookback_years=0.5,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("最终结论")
    print("="*70)
    print(f"信号: {results['final_signal']}")
    print(f"仓位: {results['position_advice']}")
    print(f"策略: {results['strategy_adjustment']}")
    
    # 导出报告
    analyzer = UnifiedWithMacroV63(market="US", verbose=False)
    report_md = analyzer.generate_full_report(results)
    
    report_path = '/root/.openclaw/workspace/myQuant/scripts/unified/full_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
    
    print(f"\n完整报告已保存: {report_path}")
