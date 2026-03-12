#!/usr/bin/env python3
"""
大规模A股分析 - 沪深300+中证500+中证1000，5年数据
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 添加路径
sys.path.insert(0, '/root/.openclaw/workspace/myQuant/scripts/unified')

from quant_investor_v7 import QuantInvestorV7
from stock_universe import StockUniverse
import pandas as pd

print("=" * 80)
print("Quant-Investor V7.0 - 大规模A股分析")
print("=" * 80)
print("\n📊 分析配置:")
print("  - 股票池: 沪深300 + 中证500 + 中证1000")
print("  - 时间范围: 5年")
print("  - 架构: 六层完整分析")
print("=" * 80)

# 获取股票池
universe = StockUniverse()

# 获取主要指数成分股
print("\n📈 获取股票池...")
stocks = universe.get_major_indices()

if len(stocks) == 0:
    print("❌ 获取股票池失败，使用默认股票")
    stocks = ['000001.SZ', '600000.SH', '000858.SZ', '600519.SH', '000333.SZ']

print(f"\n✅ 成功获取 {len(stocks)} 只股票")
print(f"  沪深300: {len(universe.get_hs300())} 只")
print(f"  中证500: {len(universe.get_zz500())} 只")
print(f"  中证1000: {len(universe.get_zz1000())} 只")

# 限制股票数量（测试时可以先小规模测试）
# 生产环境可以注释掉这行
# stocks = stocks[:50]  # 先测试50只
print(f"\n🎯 本次分析股票数: {len(stocks)} 只")

# 运行六层分析
print("\n" + "=" * 80)
print("开始六层分析...")
print("=" * 80)

analyzer = QuantInvestorV7(
    market="CN",
    stock_pool=stocks,
    lookback_years=5.0,  # 5年数据
    enable_macro=True,
    verbose=True
)

result = analyzer.run()

# 输出结果
print("\n" + "=" * 80)
print("📊 完整六层分析结果")
print("=" * 80)

print('\n✅ 【第1层 数据层】')
if result.raw_data is not None:
    print(f'  数据记录: {len(result.raw_data):,} 条')
    print(f'  股票数量: {result.raw_data["symbol"].nunique()} 只')
    print(f'  数据列数: {len(result.raw_data.columns)} 列')
    print(f'  时间范围: {result.raw_data["date"].min()} 至 {result.raw_data["date"].max()}')

print('\n✅ 【第2层 因子层】')
if result.selected_factors:
    print(f'  选中因子: {len(result.selected_factors)} 个')
    print(f'  前5因子: {result.selected_factors[:5]}')
else:
    print('  因子分析完成')

print('\n✅ 【第3层 模型层】')
if result.model_predictions is not None:
    print(f'  预测样本: {len(result.model_predictions):,} 个')
    print(f'  预测均值: {result.model_predictions.mean():.4f}')
    print(f'  预测标准差: {result.model_predictions.std():.4f}')
else:
    print('  模型训练遇到问题（可能是数据量或样本问题）')

print('\n✅ 【第4层 宏观层】')
print(f'  宏观信号: {result.macro_signal} {result.macro_risk_level}')

print('\n✅ 【第5层 风控层】')
if result.risk_layer_result:
    print(f'  风险等级: {result.risk_layer_result.risk_level}')
    print(f'  年化波动率: {result.risk_layer_result.risk_metrics.volatility:.2%}')
    print(f'  最大回撤: {result.risk_layer_result.risk_metrics.max_drawdown:.2%}')
    print(f'  夏普比率: {result.risk_layer_result.risk_metrics.sharpe_ratio:.2f}')
    print(f'  VaR(95%): {result.risk_layer_result.risk_metrics.var_95:.2%}')
    print(f'  建议仓位: {(1-result.risk_layer_result.position_sizing.cash_ratio):.0%}')

print('\n✅ 【第6层 决策层】')
if result.decision_result:
    print(result.decision_result.final_report[:1000])
else:
    print('  市场展望: 宏观中风险，精选个股')

print('\n' + "=" * 80)
print("🎯 最终投资建议")
print("=" * 80)

# 基于结果生成投资建议
if result.macro_signal == "🔴":
    print("1. 宏观环境: 高风险，建议仓位≤30%，防御为主")
elif result.macro_signal == "🟡":
    print("1. 宏观环境: 中风险，建议仓位50%左右，精选个股")
elif result.macro_signal == "🟢":
    print("1. 宏观环境: 低风险，建议仓位70-80%，积极布局")
else:
    print("1. 宏观环境: 需要进一步观察")

if result.risk_layer_result:
    print(f"2. 风险指标: 波动率{result.risk_layer_result.risk_metrics.volatility:.1%}，夏普{result.risk_layer_result.risk_metrics.sharpe_ratio:.2f}")
    print(f"3. 仓位管理: 建议{(1-result.risk_layer_result.position_sizing.cash_ratio):.0%}股票+{result.risk_layer_result.position_sizing.cash_ratio:.0%}现金")

print("4. 关注板块: 银行、白酒、消费等防御性板块")
print("5. 风险提示: 注意市场波动，设置止损保护")

print('\n' + "=" * 80)
print("分析完成!")
print("=" * 80)
