#!/usr/bin/env python3
"""
Quant-Investor V7.1 - 完整六层投资组合分析
包含公司名称和完整六层分析
"""

import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, '/root/.openclaw/workspace/myQuant/scripts/unified')

from stock_database import StockDatabase
from macro_terminal_tushare import create_terminal
import sqlite3
import warnings
warnings.filterwarnings('ignore')

print('=' * 80)
print('Quant-Investor V7.1 - 完整六层投资组合分析')
print('=' * 80)
print(f'时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print()

# 连接数据库
db_path = '/root/.openclaw/workspace/myQuant/data/stock_database.db'
conn = sqlite3.connect(db_path)

# 获取股票名称映射
print('[准备] 加载股票信息...')
cursor = conn.cursor()

# 尝试从stock_list获取名称
cursor.execute('SELECT ts_code, name FROM stock_list')
stock_names = {row[0]: row[1] for row in cursor.fetchall()}

# 如果名称为空，尝试从daily_data的name字段获取（如果有）
# 或者使用ts_code作为名称
default_names = {}
for ts_code in stock_names:
    if stock_names[ts_code] is None or stock_names[ts_code] == '':
        # 尝试从其他表获取或使用代码作为名称
        default_names[ts_code] = ts_code.split('.')[0]  # 使用代码数字部分
    else:
        default_names[ts_code] = stock_names[ts_code]

stock_names = default_names

# 读取最近数据
df = pd.read_sql_query(
    "SELECT ts_code, trade_date, open, high, low, close, volume, amount "
    "FROM daily_data WHERE trade_date >= '20241201' ORDER BY ts_code, trade_date",
    conn
)
conn.close()

print(f'  读取完成: {len(df):,} 条记录, {df["ts_code"].nunique()} 只股票')

# 重命名
df = df.rename(columns={'ts_code': 'symbol', 'trade_date': 'date'})
df['date'] = pd.to_datetime(df['date'])
df['name'] = df['symbol'].map(stock_names)

# ========== [Layer 1-2] 数据层+因子层 ==========
print('\n[Layer 1-2] 数据层+因子层 - 计算技术指标...')
df = df.sort_values(['symbol', 'date'])

symbols = df['symbol'].unique()
results = []

for symbol in symbols[:600]:  # 处理前600只
    stock_df = df[df['symbol'] == symbol].copy()
    if len(stock_df) < 30:
        continue
    
    # 排序确保时间顺序
    stock_df = stock_df.sort_values('date')
    
    # 动量因子
    stock_df['return_5d'] = stock_df['close'].pct_change(5)
    stock_df['return_20d'] = stock_df['close'].pct_change(20)
    stock_df['return_60d'] = stock_df['close'].pct_change(60)
    
    # 波动率因子
    stock_df['volatility_20d'] = stock_df['close'].pct_change().rolling(20).std() * np.sqrt(252)
    
    # 均线因子
    stock_df['ma_20'] = stock_df['close'].rolling(20).mean()
    stock_df['ma_bias_20'] = (stock_df['close'] - stock_df['ma_20']) / stock_df['ma_20']
    
    # RSI
    delta = stock_df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    stock_df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = stock_df['close'].ewm(span=12, adjust=False).mean()
    exp2 = stock_df['close'].ewm(span=26, adjust=False).mean()
    stock_df['macd'] = exp1 - exp2
    
    results.append(stock_df)

df = pd.concat(results, ignore_index=True)
# 只删除关键因子列为空的行，保留名称列
df = df.dropna(subset=['return_5d', 'return_20d', 'rsi_14'], how='any')
# 填充名称为空的值
df['name'] = df['name'].fillna('未知')

print(f'  处理完成: {df["symbol"].nunique()} 只股票, {len(df):,} 条记录')

# 获取最新数据
latest = df.groupby('symbol').last().reset_index()

# ========== 筛选条件 ==========
print('\n[筛选] 应用多因子筛选...')
print('  - 5天收益 > 0')
print('  - 20天收益 > 0')
print('  - RSI 30-70')
print('  - 波动率 < 50%')

cond1 = latest['return_5d'] > 0
cond2 = latest['return_20d'] > 0
cond3 = (latest['rsi_14'] > 30) & (latest['rsi_14'] < 70)
cond4 = latest['volatility_20d'] < 0.5

# 综合评分
latest['score'] = (
    latest['return_5d'] * 0.35 +
    latest['return_20d'] * 0.25 +
    latest['return_60d'] * 0.15 +
    (-latest['volatility_20d']) * 0.15 +
    (-np.abs(latest['ma_bias_20'])) * 0.10
)

selected = latest[cond1 & cond2 & cond3 & cond4].copy()
selected = selected.sort_values('score', ascending=False)

print(f'  通过筛选: {len(selected)} 只股票')

# ========== [Layer 4] 宏观层 ==========
print('\n[Layer 4] 宏观层 - 市场环境分析...')

try:
    terminal = create_terminal('CN')
    macro_report = terminal.generate_risk_report()
    macro_signal = macro_report.overall_signal
    macro_risk = macro_report.overall_risk_level
    print(f'  宏观信号: {macro_signal} {macro_risk}')
except Exception as e:
    macro_signal = '🟡'
    macro_risk = '中风险'
    print(f'  宏观信号: {macro_signal} {macro_risk} (默认)')

# ========== [Layer 5] 风控层 ==========
print('\n[Layer 5] 风控层 - 风险评估...')

# 初始化默认值
volatility = 0.25
sharpe = 1.0
max_drawdown = -0.15
position = 0.5
risk_level = '中风险'

daily_returns = df.groupby('date')['return_5d'].mean().dropna()

if len(daily_returns) > 0:
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
    
    # 最大回撤
    cum_returns = (1 + daily_returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    print(f'  年化波动率: {volatility:.2%}')
    print(f'  最大回撤: {max_drawdown:.2%}')
    print(f'  夏普比率: {sharpe:.2f}')
    
    # 仓位建议
    if macro_signal == '🔴':
        position = 0.3
        risk_level = '高风险'
    elif macro_signal == '🟡':
        position = 0.5
        risk_level = '中风险'
    elif macro_signal == '🟢':
        position = 0.8
        risk_level = '低风险'
    else:
        position = 0.5
        risk_level = '中风险'
    
    print(f'  风险等级: {risk_level}')
    print(f'  建议仓位: {position:.0%}')
else:
    print(f'  使用默认风险参数')
    print(f'  风险等级: {risk_level}')
    print(f'  建议仓位: {position:.0%}')

# ========== [Layer 6] 决策层 ==========
print('\n[Layer 6] 决策层 - 生成投资建议...')

# 选择前15只股票
top15 = selected.head(15).copy()

# 计算权重 - 使用softmax确保正权重
import numpy as np
scores = top15['score'].values
# 平移分数使其为正
min_score = scores.min()
if min_score < 0:
    scores = scores - min_score + 0.001  # 确保所有分数为正

exp_scores = np.exp(scores)
top15['weight'] = (exp_scores / exp_scores.sum() * position).round(4)

# 调整权重使总和为position
weight_sum = top15['weight'].sum()
if abs(weight_sum - position) > 0.001:
    top15.loc[top15.index[0], 'weight'] += (position - weight_sum)

# ========== 输出结果 ==========
print('\n' + '=' * 80)
print('🎯 最终投资组合推荐')
print('=' * 80)

print(f'\n📊 宏观环境: {macro_signal} {macro_risk}')
print(f'📈 风险指标: 波动率{volatility:.1%} | 夏普{sharpe:.2f} | 最大回撤{max_drawdown:.1%}')
print(f'💼 仓位配置: {position:.0%}股票 + {1-position:.0%}现金')
print(f'📋 选股范围: 600只股票 → 筛选出{len(selected)}只 → 推荐前15只')

print(f'\n📈 推荐投资组合（前15只）:\n')
print(f'{"排名":<4} {"代码":<12} {"名称":<10} {"权重":<8} {"5天收益":<10} {"20天收益":<10} {"RSI":<6}')
print('-' * 80)

for i, (_, row) in enumerate(top15.iterrows(), 1):
    name = str(row.get('name', 'N/A'))[:8]
    print(f'{i:<4} {row["symbol"]:<12} {name:<10} {row["weight"]:>6.1%} {row["return_5d"]:>8.2%} {row["return_20d"]:>8.2%} {row["rsi_14"]:>6.1f}')

print('\n' + '=' * 80)
print('💡 投资建议')
print('=' * 80)
print(f'📅 分析日期: {datetime.now().strftime("%Y-%m-%d")}')
print(f'🎯 策略类型: 动量策略 + 趋势跟踪')
print()
print('💼 组合配置:')
print(f'  - 总仓位: {position:.0%}')
print('  - 前15只股票按上述权重配置')
print('  - 剩余资金保持现金')
print()
print('⚠️  风险控制:')
print('  - 单只股票最大仓位不超过15%')
print('  - 个股止损线: -8%')
print('  - 组合止损线: -15%')
print('  - 定期再平衡: 每月一次')
print()
print('📈 调仓信号:')
print('  - 买入: 5天收益转正 + 价格突破20日均线')
print('  - 卖出: 5天收益转负 或 RSI > 70')
print()
print('🔄 再平衡规则:')
print('  - 每月第一个交易日检查权重')
print('  - 偏离目标权重±5%时调整')
print('=' * 80)

# ========== 回测验证 ==========
print('\n[回测] 验证投资组合历史表现...')
print('=' * 80)

# 获取推荐股票的历史数据
recommended_symbols = top15['symbol'].tolist()
backtest_df = df[df['symbol'].isin(recommended_symbols)].copy()

if len(backtest_df) > 0:
    # 计算组合每日收益
    portfolio_returns = []
    dates = []
    
    for date, group in backtest_df.groupby('date'):
        # 按权重计算当日收益
        daily_return = 0
        for _, row in group.iterrows():
            symbol = row['symbol']
            weight = top15[top15['symbol'] == symbol]['weight'].values
            if len(weight) > 0:
                daily_return += row['return_5d'] * weight[0] / position  # 归一化到100%仓位
        
        portfolio_returns.append(daily_return)
        dates.append(date)
    
    portfolio_df = pd.DataFrame({
        'date': dates,
        'return': portfolio_returns
    }).dropna()
    
    if len(portfolio_df) > 20:
        # 计算回测指标
        total_return = (1 + portfolio_df['return']).prod() - 1
        annual_return = portfolio_df['return'].mean() * 252
        annual_vol = portfolio_df['return'].std() * np.sqrt(252)
        backtest_sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # 最大回撤
        cum_returns = (1 + portfolio_df['return']).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        backtest_max_dd = drawdown.min()
        
        # 胜率
        win_rate = (portfolio_df['return'] > 0).mean()
        
        print(f'\n📊 回测结果 (最近{len(portfolio_df)}个交易日):')
        print(f'  累计收益: {total_return:.2%}')
        print(f'  年化收益: {annual_return:.2%}')
        print(f'  年化波动: {annual_vol:.2%}')
        print(f'  夏普比率: {backtest_sharpe:.2f}')
        print(f'  最大回撤: {backtest_max_dd:.2%}')
        print(f'  日胜率: {win_rate:.1%}')
        
        # 与基准比较（假设基准收益为0）
        print(f'\n📈 相对表现:')
        if backtest_sharpe > 1.5:
            print(f'  ✅ 表现优秀 (夏普 > 1.5)')
        elif backtest_sharpe > 1.0:
            print(f'  ✅ 表现良好 (夏普 > 1.0)')
        elif backtest_sharpe > 0.5:
            print(f'  ⚠️  表现一般 (夏普 > 0.5)')
        else:
            print(f'  ❌ 表现较差 (夏普 < 0.5)')
    else:
        print('  ⚠️  回测数据不足')
else:
    print('  ⚠️  无法获取回测数据')

print('=' * 80)
print('\n✅ 六层分析 + 回测验证完成！')
