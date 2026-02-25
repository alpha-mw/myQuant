#!/usr/bin/env python3
"""
Quant-Investor Unified v7.0 + MacroRiskTerminal V6.3 (Tushare优先)
完整集成版本

特性:
1. Tushare作为首选数据源 (使用提供的token和URL)
2. 量化投资分析
3. 宏观风控分析 (CN-US多市场)
4. 完全透明化报告
"""

import sys
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

# 添加路径
unified_dir = os.path.dirname(os.path.abspath(__file__))
if unified_dir not in sys.path:
    sys.path.insert(0, unified_dir)

# 导入Tushare优先的宏观终端
from macro_terminal_tushare import (
    create_terminal, detect_market,
    MacroRiskTerminalBase,
    RiskTerminalReport,
    TUSHARE_TOKEN,
    TUSHARE_URL
)

# 导入量化流水线
from pipeline import MasterPipelineUnified


class UnifiedTushare:
    """
    Tushare优先的统一分析框架
    """
    
    VERSION = "7.0.0-tushare"
    
    def __init__(
        self,
        market: str = "CN",
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
                self._log(f"宏观风控终端初始化成功: {self.market} (Tushare优先)")
            except Exception as e:
                self._log(f"宏观风控终端初始化失败: {e}")
    
    def _log(self, msg: str):
        timestamp = datetime.now().strftime('%H:%M:%S')
        entry = f"[{timestamp}] {msg}"
        self.execution_log.append(entry)
        if self.verbose:
            print(entry)
    
    def run(self) -> Dict[str, Any]:
        """执行完整分析"""
        self._log("=" * 80)
        self._log(f"Quant-Investor Unified {self.VERSION} (Tushare优先)")
        self._log(f"市场: {self.market}, 股票: {self.stock_pool}")
        self._log(f"Tushare URL: {TUSHARE_URL}")
        self._log("=" * 80)
        
        results = {
            'version': self.VERSION,
            'market': self.market,
            'tushare_url': TUSHARE_URL,
            'execution_log': self.execution_log,
            'quant_report': None,
            'macro_report': None,
            'macro_markdown': None,
            'final_recommendation': None
        }
        
        # Step 1: 量化分析
        self._log("\n【Step 1/2】量化投资分析")
        try:
            quant_report = self.quant_pipeline.run()
            results['quant_report'] = quant_report
            self._log(f"量化分析完成: {len(quant_report.data_bundle.stock_universe)}只股票")
        except Exception as e:
            self._log(f"量化分析失败: {e}")
        
        # Step 2: 宏观风控分析 (Tushare优先)
        if self.macro_terminal:
            self._log("\n【Step 2/2】宏观风控分析 (Tushare优先)")
            try:
                macro_report = self.macro_terminal.generate_risk_report()
                results['macro_report'] = macro_report
                results['macro_markdown'] = self.macro_terminal.format_report_markdown(macro_report)
                results['final_recommendation'] = f"{macro_report.overall_signal} {macro_report.overall_risk_level} | {macro_report.recommendation}"
                self._log(f"宏观分析完成: {len(macro_report.modules)}个模块")
                self._log(f"综合信号: {macro_report.overall_signal} {macro_report.overall_risk_level}")
            except Exception as e:
                self._log(f"宏观分析失败: {e}")
        
        self._log("\n" + "=" * 80)
        self._log("分析流程完成")
        self._log("=" * 80)
        
        return results
    
    def generate_full_report(self, results: Dict[str, Any]) -> str:
        """生成完整报告"""
        lines = []
        
        lines.append("# Quant-Investor 统一版投资分析报告 (Tushare优先)")
        lines.append(f"**版本**: {self.VERSION}")
        lines.append(f"**市场**: {self.market}")
        lines.append(f"**Tushare URL**: {TUSHARE_URL}")
        lines.append(f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        if results['final_recommendation']:
            lines.append("## 🎯 综合结论")
            lines.append("")
            lines.append(f"**{results['final_recommendation']}**")
            lines.append("")
        
        if results['quant_report']:
            quant = results['quant_report']
            lines.append("## 📊 量化分析摘要")
            lines.append("")
            lines.append(f"- **分析标的**: {len(quant.data_bundle.stock_universe)} 只股票")
            if quant.risk_output and quant.risk_output.portfolio:
                port = quant.risk_output.portfolio
                lines.append(f"- **组合配置**: {port.weights}")
            lines.append("")
        
        if results['macro_markdown']:
            lines.append("## 🌍 宏观风控分析 (Tushare优先)")
            lines.append("")
            lines.append(results['macro_markdown'])
            lines.append("")
        
        return "\n".join(lines)


# 便捷函数
def analyze_with_tushare(
    market: str = "CN",
    stocks: Optional[List[str]] = None,
    lookback_years: float = 1.0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Tushare优先的分析函数
    
    示例:
        results = analyze_with_tushare(
            market="CN",
            stocks=["000001.SZ", "600000.SH"],
            lookback_years=1.0
        )
        print(results['final_recommendation'])
    """
    analyzer = UnifiedTushare(
        market=market,
        stock_pool=stocks,
        lookback_years=lookback_years,
        verbose=verbose
    )
    return analyzer.run()


if __name__ == '__main__':
    print("="*80)
    print("Quant-Investor Unified v7.0 + MacroRiskTerminal V6.3")
    print("Tushare优先集成版本")
    print(f"Tushare URL: {TUSHARE_URL}")
    print("="*80)
    
    # A股示例
    print("\n【A股分析示例】")
    results = analyze_with_tushare(
        market="CN",
        stocks=["000001.SZ", "600000.SH", "000858.SZ"],
        lookback_years=0.5,
        verbose=True
    )
    
    print("\n" + "="*80)
    print("最终结论")
    print("="*80)
    print(results['final_recommendation'])
    
    # 保存报告
    analyzer = UnifiedTushare(market="CN", verbose=False)
    report_md = analyzer.generate_full_report(results)
    
    report_path = '/root/.openclaw/workspace/myQuant/scripts/unified/tushare_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
    
    print(f"\n完整报告已保存: {report_path}")
