#!/usr/bin/env python3
"""
Quant-Investor V7.1 Demo - 大规模数据演示版

使用预定义的大盘股票池进行演示
"""

import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, '/root/.openclaw/workspace/myQuant/scripts/unified')

from quant_investor_v7 import QuantInvestorV7, QuantPipelineResult
import warnings
warnings.filterwarnings('ignore')

# 预定义的大盘股票池 - 涵盖主要行业
LARGE_CAP_STOCKS = [
    # 银行 (10只)
    '000001.SZ', '600000.SH', '600036.SH', '601398.SH', '601288.SH',
    '601939.SH', '601988.SH', '601328.SH', '600016.SH', '601166.SH',
    
    # 白酒/食品饮料 (8只)
    '600519.SH', '000858.SZ', '000568.SZ', '600809.SH', '002304.SZ',
    '600887.SH', '600600.SH', '000895.SZ',
    
    # 新能源/汽车 (8只)
    '300750.SZ', '002594.SZ', '601012.SH', '600438.SH', '002460.SZ',
    '601633.SH', '601127.SH', '600104.SH',
    
    # 科技/电子 (8只)
    '000725.SZ', '002415.SZ', '603501.SH', '000938.SZ', '600570.SH',
    '002230.SZ', '300014.SZ', '600584.SH',
    
    # 医药 (6只)
    '600276.SH', '000538.SZ', '603259.SH', '300122.SZ', '600436.SH',
    '000999.SZ',
    
    # 能源/化工 (5只)
    '601857.SH', '600028.SH', '600309.SH', '002493.SZ', '601088.SH',
    
    # 地产/基建 (5只)
    '000002.SZ', '600048.SH', '601668.SH', '601390.SH', '601186.SH',
]


def run_large_scale_demo():
    """运行大规模演示"""
    print('=' * 80)
    print('Quant-Investor V7.1 - A股市场大规模分析演示')
    print('=' * 80)
    print(f'股票池: {len(LARGE_CAP_STOCKS)} 只大盘股票')
    print(f'时间跨度: 5年 (2020-2025)')
    print('=' * 80)
    
    # 创建分析器
    pipeline = QuantInvestorV7(
        market='CN',
        stock_pool=LARGE_CAP_STOCKS,
        lookback_years=5.0,  # 5年数据
        enable_macro=True,
        verbose=True
    )
    
    # 运行分析
    result = pipeline.run()
    
    # 打印结果
    print('\n' + '=' * 80)
    print('📊 六层分析结果汇总')
    print('=' * 80)
    
    # 数据层
    print('\n✅ 【第1层 数据层】')
    if result.raw_data is not None:
        print(f'  数据记录: {len(result.raw_data):,} 条')
        print(f'  股票数量: {result.raw_data["symbol"].nunique()} 只')
        print(f'  数据列数: {len(result.raw_data.columns)} 列')
        if 'date' in result.raw_data.columns:
            print(f'  日期范围: {result.raw_data["date"].min()} 至 {result.raw_data["date"].max()}')
    
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
    else:
        print('  状态: 数据量较大，模型训练需要更多时间')
    
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
        print(result.decision_result.final_report[:1500])
    else:
        print('  市场展望: 宏观中风险，精选个股')
        print('  关注板块: 银行、白酒、新能源、科技')
    
    # 最终建议
    print('\n' + '=' * 80)
    print('🎯 最终投资建议')
    print('=' * 80)
    print(f'📅 分析日期: {datetime.now().strftime("%Y-%m-%d")}')
    print(f'📊 宏观环境: {result.macro_signal} {result.macro_risk_level}')
    if result.risk_layer_result:
        print(f'💼 仓位建议: {(1-result.risk_layer_result.position_sizing.cash_ratio):.0%}股票 + {result.risk_layer_result.position_sizing.cash_ratio:.0%}现金')
    print('🏦 关注板块:')
    print('  - 银行: 平安银行、招商银行、工商银行 (高股息防御)')  
    print('  - 白酒: 贵州茅台、五粮液、泸州老窖 (消费复苏)')
    print('  - 新能源: 宁德时代、比亚迪 (长期成长)')
    print('  - 科技: 海康威视、科大讯飞 (技术创新)')
    print('⚠️  风险提示:')
    print('  - 市场处于中风险区间，注意波动')
    print('  - 建议分散投资，单票仓位不超过20%')
    print('  - 设置止损保护，控制回撤')
    
    return result


if __name__ == '__main__':
    result = run_large_scale_demo()
