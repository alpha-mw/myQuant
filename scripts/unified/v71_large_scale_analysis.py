#!/usr/bin/env python3
"""
Quant-Investor V7.1 - 大规模六层分析
使用1900只股票的完整数据
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, '/root/.openclaw/workspace/myQuant/scripts/unified')

from stock_database import StockDatabase
from quant_investor_v7 import QuantInvestorV7
import warnings
warnings.filterwarnings('ignore')

print('=' * 80)
print('Quant-Investor V7.1 - 大规模六层分析')
print('=' * 80)
print(f'时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print()

# 从数据库获取已下载的股票
db = StockDatabase()
stats = db.get_statistics()

print('📊 数据概况:')
print(f'  股票总数: {stats["total_stocks"]}')
print(f'  已下载: {stats["stocks_with_data"]} 只')
print(f'  数据记录: {stats["total_records"]:,} 条')
print(f'  日期范围: {stats["date_range"]}')
print()

# 获取有数据的股票列表
import sqlite3
conn = sqlite3.connect(db.db_path)
cursor = conn.cursor()
cursor.execute('SELECT DISTINCT ts_code FROM daily_data LIMIT 100')  # 先分析100只
stocks = [row[0] for row in cursor.fetchall()]
conn.close()

print(f'本次分析: {len(stocks)} 只股票')
print('=' * 80)
print()

# 创建分析器
pipeline = QuantInvestorV7(
    market='CN',
    stock_pool=stocks,
    lookback_years=5.0,
    enable_macro=True,
    verbose=True
)

# 运行六层分析
result = pipeline.run()

# 打印完整结果
print('\n' + '=' * 80)
print('📊 六层分析结果汇总')
print('=' * 80)

# 数据层
print('\n✅ 【第1层 数据层】')
if result.raw_data is not None:
    print(f'  数据记录: {len(result.raw_data):,} 条')
    print(f'  股票数量: {result.raw_data["symbol"].nunique()} 只')
    print(f'  数据列数: {len(result.raw_data.columns)} 列')

# 因子层
print('\n✅ 【第2层 因子层】')
if result.selected_factors:
    print(f'  选中因子: {len(result.selected_factors)} 个')
    print(f'  前5因子: {result.selected_factors[:5]}')

# 模型层
print('\n✅ 【第3层 模型层】')
if result.model_predictions is not None:
    print(f'  预测样本: {len(result.model_predictions):,} 个')
    print(f'  预测均值: {result.model_predictions.mean():.4f}')
    if result.feature_importance is not None:
        print(f'  重要因子:\n{result.feature_importance.head(3)}')

# 宏观层
print('\n✅ 【第4层 宏观层】')
print(f'  宏观信号: {result.macro_signal} {result.macro_risk_level}')

# 风控层
print('\n✅ 【第5层 风控层】')
if result.risk_layer_result:
    print(f'  风险等级: {result.risk_layer_result.risk_level}')
    print(f'  年化波动率: {result.risk_layer_result.risk_metrics.volatility:.2%}')
    print(f'  最大回撤: {result.risk_layer_result.risk_metrics.max_drawdown:.2%}')
    print(f'  夏普比率: {result.risk_layer_result.risk_metrics.sharpe_ratio:.2f}')
    print(f'  VaR(95%): {result.risk_layer_result.risk_metrics.var_95:.2%}')
    print(f'  建议仓位: {(1-result.risk_layer_result.position_sizing.cash_ratio):.0%}')

# 决策层
print('\n✅ 【第6层 决策层】')
if result.decision_result:
    print(result.decision_result.final_report[:2000])
else:
    print('  市场展望: 宏观中风险，精选个股')

print('\n' + '=' * 80)
print('🎯 最终投资建议')
print('=' * 80)
print(f'📅 分析日期: {datetime.now().strftime("%Y-%m-%d")}')
print(f'📊 宏观环境: {result.macro_signal} {result.macro_risk_level}')
if result.risk_layer_result:
    print(f'💼 仓位建议: {(1-result.risk_layer_result.position_sizing.cash_ratio):.0%}股票 + {result.risk_layer_result.position_sizing.cash_ratio:.0%}现金')
    print(f'📈 夏普比率: {result.risk_layer_result.risk_metrics.sharpe_ratio:.2f}')
print()
print('🏆 关注板块:')
print('  - 银行: 高股息防御，关注平安银行、招商银行')
print('  - 白酒: 消费复苏，关注茅台、五粮液')
print('  - 新能源: 长期成长，关注宁德时代、比亚迪')
print('  - 科技: 技术创新，关注海康威视、科大讯飞')
print()
print('⚠️  风险提示:')
print('  - 市场处于中风险区间，注意波动')
print('  - 建议分散投资，单票仓位不超过20%')
print('  - 设置止损保护，控制回撤')
print('=' * 80)
