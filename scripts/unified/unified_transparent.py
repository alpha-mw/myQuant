#!/usr/bin/env python3
"""
Quant-Investor Unified v7.0 + MacroRiskTerminal V6.3 Transparent
完全透明化集成版本
"""

import sys
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

# 添加路径
unified_dir = os.path.dirname(os.path.abspath(__file__))
if unified_dir not in sys.path:
    sys.path.insert(0, unified_dir)

# 导入透明化宏观终端
from macro_terminal_transparent import (
    create_terminal, detect_market,
    MacroRiskTerminalBase,
    RiskTerminalReport
)

# 导入量化流水线
from pipeline import MasterPipelineUnified, UnifiedReport
from logger import get_logger


class UnifiedTransparent:
    """
    完全透明化的统一分析框架
    
    所有分析步骤都记录详细日志，可追溯
    """
    
    VERSION = "7.0.0-transparent"
    
    def __init__(
        self,
        market: str = "US",
        stock_pool: Optional[List[str]] = None,
        lookback_years: float = 1.0,
        enable_macro: bool = True,
        verbose: bool = True
    ):
        self.market = market.upper()
        self.stock_pool = stock_pool
        self.lookback_years = lookback_years
        self.enable_macro = enable_macro
        self.verbose = verbose
        self.execution_log: List[str] = []
        self._logger = get_logger("UnifiedTransparent", verbose)

        # 初始化量化流水线
        self.quant_pipeline = MasterPipelineUnified(
            market=self.market,
            stock_pool=stock_pool,
            lookback_years=lookback_years,
            verbose=verbose
        )
        
        # 初始化宏观终端
        self.macro_terminal: Optional[MacroRiskTerminalBase] = None
        if enable_macro:
            try:
                self.macro_terminal = create_terminal(market=self.market)
                self._log(f"宏观风控终端初始化成功: {self.market}")
            except Exception as e:
                self._log(f"宏观风控终端初始化失败: {e}")
    
    def _log(self, msg: str) -> None:
        """记录执行日志"""
        self.execution_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        self._logger.info(msg)
    
    def run(self) -> Dict[str, Any]:
        """
        执行完全透明化的分析流程
        
        Returns:
            包含完整分析过程和结果的字典
        """
        self._log("=" * 80)
        self._log(f"Quant-Investor Unified {self.VERSION} 开始执行")
        self._log(f"市场: {self.market}, 股票: {self.stock_pool}, 回测: {self.lookback_years}年")
        self._log("=" * 80)
        
        results = {
            'version': self.VERSION,
            'market': self.market,
            'execution_log': self.execution_log,
            'quant_report': None,
            'macro_report': None,
            'macro_markdown': None,
            'combined_analysis': None,
            'final_recommendation': None
        }
        
        # Step 1: 量化分析
        self._log("\n【Step 1/3】量化投资分析")
        self._log("-" * 80)
        
        try:
            quant_report = self.quant_pipeline.run()
            results['quant_report'] = quant_report
            self._log(f"量化分析完成: {len(quant_report.data_bundle.stock_universe)}只股票")
        except Exception as e:
            self._log(f"量化分析失败: {e}")
        
        # Step 2: 宏观风控分析
        if self.macro_terminal:
            self._log("\n【Step 2/3】宏观风控分析 (完全透明化)")
            self._log("-" * 80)
            
            try:
                macro_report = self.macro_terminal.generate_risk_report()
                results['macro_report'] = macro_report
                results['macro_markdown'] = self.macro_terminal.format_report_markdown(macro_report)
                self._log(f"宏观分析完成: {len(macro_report.modules)}个模块")
            except Exception as e:
                self._log(f"宏观分析失败: {e}")
        
        # Step 3: 综合分析
        self._log("\n【Step 3/3】综合信号判断")
        self._log("-" * 80)
        
        combined = self._combine_analysis(
            results.get('quant_report'),
            results.get('macro_report')
        )
        results['combined_analysis'] = combined
        results['final_recommendation'] = combined['recommendation']
        
        self._log(f"综合信号: {combined['signal']}")
        self._log(f"仓位建议: {combined['position']}")
        self._log(f"策略调整: {combined['strategy']}")
        
        self._log("\n" + "=" * 80)
        self._log("分析流程完成")
        self._log("=" * 80)
        
        return results
    
    def _combine_analysis(self, quant_report, macro_report) -> Dict[str, str]:
        """
        综合量化和宏观分析 - 完全透明化
        
        展示完整的推理过程
        """
        self._log("开始综合信号计算...")
        
        analysis = {
            'quant_signal': 'neutral',
            'macro_signal': 'neutral',
            'macro_modules': {},
            'reasoning_steps': [],
            'signal': '🟡 中风险',
            'position': '50% 仓位',
            'strategy': '控制仓位，精选个股',
            'recommendation': ''
        }
        
        # 量化信号
        if quant_report and quant_report.risk_output:
            if quant_report.risk_output.portfolio:
                port = quant_report.risk_output.portfolio
                volatility = port.volatility
                
                self._log(f"量化组合波动率: {volatility*100:.1f}%")
                
                if volatility > 0.30:
                    analysis['quant_signal'] = 'high_risk'
                    analysis['reasoning_steps'].append(f"量化: 波动率{volatility*100:.1f}% > 30%，高风险")
                elif volatility < 0.20:
                    analysis['quant_signal'] = 'low_risk'
                    analysis['reasoning_steps'].append(f"量化: 波动率{volatility*100:.1f}% < 20%，低风险")
                else:
                    analysis['quant_signal'] = 'medium'
                    analysis['reasoning_steps'].append(f"量化: 波动率{volatility*100:.1f}% 正常")
        
        # 宏观信号
        if macro_report:
            analysis['macro_signal'] = macro_report.overall_signal
            analysis['macro_modules'] = {
                m.module_name: {
                    'signal': m.overall_signal,
                    'indicators': [ind.name for ind in m.indicators]
                }
                for m in macro_report.modules
            }
            
            self._log(f"宏观信号: {macro_report.overall_signal} {macro_report.overall_risk_level}")
            
            # 使用宏观终端的综合判断
            analysis['signal'] = f"{macro_report.overall_signal} {macro_report.overall_risk_level}"
            analysis['position'] = self._signal_to_position(macro_report.overall_signal)
            analysis['strategy'] = macro_report.recommendation
            
            analysis['reasoning_steps'].append(
                f"宏观: {macro_report.overall_signal} ({macro_report.overall_risk_level})"
            )
            
            # 列出风险模块
            red_modules = [m.module_name for m in macro_report.modules if m.overall_signal == '🔴']
            yellow_modules = [m.module_name for m in macro_report.modules if m.overall_signal == '🟡']
            
            if red_modules:
                analysis['reasoning_steps'].append(f"风险模块: {', '.join(red_modules)}")
            if yellow_modules:
                analysis['reasoning_steps'].append(f"注意模块: {', '.join(yellow_modules)}")
        
        # 生成最终建议
        analysis['recommendation'] = f"{analysis['signal']} | {analysis['position']} | {analysis['strategy']}"
        
        self._log("综合判断推理:")
        for step in analysis['reasoning_steps']:
            self._log(f"  - {step}")
        
        return analysis
    
    def _signal_to_position(self, signal: str) -> str:
        """信号转仓位建议"""
        mapping = {
            "🔴": "≤30% 仓位",
            "🟡": "30%-60% 仓位",
            "🟢": "60%-90% 仓位",
            "🔵": "80%-100% 仓位"
        }
        return mapping.get(signal, "50% 仓位")
    
    def generate_full_report(self, results: Dict[str, Any]) -> str:
        """生成完整的Markdown报告"""
        lines = []
        
        lines.append("# Quant-Investor 统一版投资分析报告 (完全透明化)")
        lines.append(f"**版本**: {self.VERSION}")
        lines.append(f"**市场**: {self.market}")
        lines.append(f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # 综合结论
        lines.append("## 🎯 综合结论")
        lines.append("")
        
        if results['combined_analysis']:
            ca = results['combined_analysis']
            lines.append(f"| 项目 | 内容 |")
            lines.append(f"|:---|:---|")
            lines.append(f"| 综合信号 | {ca['signal']} |")
            lines.append(f"| 仓位建议 | {ca['position']} |")
            lines.append(f"| 策略调整 | {ca['strategy']} |")
            lines.append("")
            
            if ca['reasoning_steps']:
                lines.append("### 推理过程")
                lines.append("")
                for step in ca['reasoning_steps']:
                    lines.append(f"- {step}")
                lines.append("")
        
        # 量化分析
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
        
        # 宏观风控
        if results['macro_markdown']:
            lines.append("## 🌍 宏观风控分析 (完全透明化)")
            lines.append("")
            lines.append(results['macro_markdown'])
            lines.append("")
        
        # 执行日志
        lines.append("## 📝 执行日志")
        lines.append("")
        lines.append("```")
        for log in results['execution_log']:
            lines.append(log)
        lines.append("```")
        lines.append("")
        
        return "\n".join(lines)


# ==================== 便捷函数 ====================

def analyze_transparent(
    market: str = "US",
    stocks: Optional[List[str]] = None,
    lookback_years: float = 1.0,
    enable_macro: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    完全透明化的分析函数
    
    示例:
        results = analyze_transparent(
            market="US",
            stocks=["AAPL", "MSFT", "NVDA"],
            lookback_years=0.5
        )
        
        print(results['final_recommendation'])
        
        # 保存完整报告
        analyzer = UnifiedTransparent(market="US", verbose=False)
        with open('report.md', 'w') as f:
            f.write(analyzer.generate_full_report(results))
    """
    analyzer = UnifiedTransparent(
        market=market,
        stock_pool=stocks,
        lookback_years=lookback_years,
        enable_macro=enable_macro,
        verbose=verbose
    )
    return analyzer.run()


# ==================== 主程序 ====================

if __name__ == '__main__':
    print("="*80)
    print("Quant-Investor Unified v7.0 + MacroRiskTerminal V6.3")
    print("完全透明化集成版本")
    print("="*80)
    
    # 运行示例
    results = analyze_transparent(
        market="US",
        stocks=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
        lookback_years=0.5,
        verbose=True
    )
    
    print("\n" + "="*80)
    print("最终结论")
    print("="*80)
    print(f"{results['final_recommendation']}")
    
    # 保存报告
    analyzer = UnifiedTransparent(market="US", verbose=False)
    report_md = analyzer.generate_full_report(results)
    
    report_path = '/root/.openclaw/workspace/myQuant/scripts/unified/transparent_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
    
    print(f"\n完整报告已保存: {report_path}")
