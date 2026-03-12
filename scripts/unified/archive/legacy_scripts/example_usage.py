#!/usr/bin/env python3
"""
Quant-Investor Unified v7.0 + MacroRiskTerminal V6.3
统一版本完整示例

功能:
1. 量化投资分析 (数据/因子/模型/决策/风控)
2. 宏观风控分析 (多市场适配)
3. 综合信号判断
"""

import sys
import os

# 添加路径
unified_dir = os.path.dirname(os.path.abspath(__file__))
if unified_dir not in sys.path:
    sys.path.insert(0, unified_dir)

from unified_with_macro import analyze_with_macro_risk


def main():
    print("="*70)
    print("Quant-Investor Unified v7.0 + MacroRiskTerminal V6.3")
    print("="*70)
    
    # 示例1: 美股分析
    print("\n【示例1】美股分析 (AAPL, MSFT, NVDA, GOOGL, AMZN)")
    print("-"*70)
    
    results_us = analyze_with_macro_risk(
        market="US",
        stocks=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
        lookback_years=0.5,
        enable_macro_risk=True,
        verbose=True
    )
    
    print("\n📊 量化分析结果:")
    quant = results_us['quant_report']
    print(f"  - 股票数量: {len(quant.data_bundle.stock_universe)}")
    print(f"  - 有效因子: {len(quant.factor_output.effective_factors)}")
    print(f"  - 模型排名: {len(quant.model_output.ranked_stocks)}")
    if quant.risk_output and quant.risk_output.portfolio:
        print(f"  - 组合权重: {quant.risk_output.portfolio.weights}")
    
    print("\n🌍 宏观风控结果:")
    macro = results_us['macro_report']
    if macro:
        print(f"  - 综合信号: {macro.overall_signal} {macro.overall_risk_level}")
        print(f"  - 建议: {macro.recommendation}")
        for m in macro.modules:
            print(f"  - {m.module_name}: {m.overall_signal}")
    
    print("\n🎯 最终建议:")
    print(f"  {results_us['final_recommendation']}")
    
    # 示例2: A股分析 (需要Tushare token)
    print("\n\n【示例2】A股分析 (需要配置Tushare Token)")
    print("-"*70)
    print("""
# 使用方式:
results_cn = analyze_with_macro_risk(
    market="CN",
    stocks=["000001.SZ", "600000.SH", "000858.SZ"],  # 平安银行、浦发银行、五粮液
    lookback_years=1,
    enable_macro_risk=True,
    verbose=True
)
    """)
    
    # 示例3: 快速分析
    print("\n【示例3】快速分析函数")
    print("-"*70)
    print("""
from unified_with_macro import analyze_with_macro_risk

# 一行代码完成分析
results = analyze_with_macro_risk(
    market="US",
    stocks=["TSLA", "META", "AMD"],
    lookback_years=0.5
)

# 获取关键结果
print(results['final_recommendation'])
print(results['combined_signal'])

# 获取详细报告
quant_report = results['quant_report']
macro_report = results['macro_report']

# 导出Markdown报告
if macro_report:
    markdown = results.get('macro_terminal').format_report_markdown(macro_report)
    with open('macro_report.md', 'w') as f:
        f.write(markdown)
    """)
    
    print("\n" + "="*70)
    print("集成版本功能清单")
    print("="*70)
    print("""
量化投资层 (V2.7-V6.0整合):
  ✅ 数据获取 (yfinance/Tushare)
  ✅ 因子计算 (动量/波动率/均值回归等)
  ✅ ML模型 (XGBoost/LightGBM/RandomForest)
  ✅ 组合优化 (最大夏普比率)
  ✅ 风险评估 (VaR/波动率/回撤)

宏观风控层 (V6.3):
  ✅ 多市场适配 (CN/US)
  ✅ 货币政策分析
  ✅ 经济增长分析
  ✅ 估值分析 (巴菲特指标)
  ✅ 通胀分析
  ✅ 情绪与收益率曲线

综合信号:
  ✅ 量化+宏观双维度判断
  ✅ 仓位建议
  ✅ 策略调整建议
    """)


if __name__ == '__main__':
    main()
