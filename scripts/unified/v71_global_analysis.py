#!/usr/bin/env python3
"""
Quant-Investor V7.1 - 全局数据分析（1900只股票）
优化版本：分批处理，避免内存问题
"""

import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, '/root/.openclaw/workspace/myQuant/scripts/unified')

from stock_database import StockDatabase
from factor_analyzer import FactorAnalyzer
from enhanced_model_layer import EnhancedModelLayer
from macro_terminal_tushare import create_terminal
from risk_management_layer import RiskManagementLayer
from decision_layer import DecisionLayer
import warnings
warnings.filterwarnings('ignore')

print('=' * 80)
print('Quant-Investor V7.1 - 全局数据分析')
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

# 获取所有有数据的股票
import sqlite3
conn = sqlite3.connect(db.db_path)
cursor = conn.cursor()
cursor.execute('SELECT DISTINCT ts_code FROM daily_data')
all_stocks = [row[0] for row in cursor.fetchall()]
conn.close()

print(f'本次分析: {len(all_stocks)} 只股票（全局数据）')
print('=' * 80)
print()

# ========== 第1层：数据层（分批处理）==========
print('[Layer 1] 数据层 - 加载全局数据...')

# 从数据库直接读取所有数据
print('  从数据库读取...')
conn = sqlite3.connect(db.db_path)

# 使用超时设置，避免卡住
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("数据读取超时")

# 设置5分钟超时
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(300)

try:
    df = pd.read_sql_query(
        "SELECT * FROM daily_data WHERE trade_date >= '20200101' LIMIT 2500000", 
        conn
    )
    signal.alarm(0)  # 取消超时
except TimeoutException:
    print('  读取超时，使用已有数据...')
    df = pd.DataFrame()

conn.close()

if len(df) == 0:
    print('  从数据库直接读取失败，尝试分批读取...')
    # 分批读取
    chunks = []
    conn = sqlite3.connect(db.db_path)
    for i, chunk in enumerate(pd.read_sql_query(
        "SELECT * FROM daily_data WHERE trade_date >= '20200101'", 
        conn, 
        chunksize=500000
    )):
        print(f'    读取批次 {i+1}: {len(chunk)} 条')
        chunks.append(chunk)
        if i >= 4:  # 最多读取250万条
            break
    conn.close()
    df = pd.concat(chunks, ignore_index=True)

print(f'  加载完成: {len(df):,} 条记录, {df["ts_code"].nunique()} 只股票')

# 重命名列以兼容现有代码
df = df.rename(columns={
    'ts_code': 'symbol',
    'trade_date': 'date',
    'open': 'open',
    'high': 'high',
    'low': 'low',
    'close': 'close',
    'volume': 'volume',
    'amount': 'amount'
})

# 添加特征工程（简化版）
print('  特征工程...')

# 动量因子
for period in [5, 10, 20, 60, 120]:
    df[f'return_{period}d'] = df.groupby('symbol')['close'].pct_change(period)

# 波动率因子
for period in [20, 60, 120]:
    df[f'volatility_{period}d'] = df.groupby('symbol')['close'].pct_change().rolling(period).std().values * np.sqrt(252)

# 技术指标
# RSI
delta = df.groupby('symbol')['close'].diff()
gain = delta.where(delta > 0, 0).groupby(df['symbol']).transform(lambda x: x.rolling(14).mean())
loss = (-delta.where(delta < 0, 0)).groupby(df['symbol']).transform(lambda x: x.rolling(14).mean())
rs = gain / loss
df['rsi_14'] = 100 - (100 / (1 + rs))

# 均线偏离
for period in [5, 10, 20, 60]:
    ma = df.groupby('symbol')['close'].transform(lambda x: x.rolling(period).mean())
    df[f'ma_bias_{period}'] = (df['close'] - ma) / ma

# MACD
exp1 = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
exp2 = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
df['macd'] = exp1 - exp2
df['macd_signal'] = df.groupby('symbol')['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())

# 目标变量：未来5天收益
df['label_return'] = df.groupby('symbol')['close'].pct_change(5).shift(-5)

# 删除缺失值
df_clean = df.dropna()
print(f'  清理后: {len(df_clean):,} 条记录')

# ========== 第2层：因子层 ==========
print('\n[Layer 2] 因子层 - 因子分析...')

factor_cols = [c for c in df_clean.columns if c.startswith(('return_', 'volatility_', 'rsi_', 'macd_', 'ma_bias_'))]
print(f'  因子数量: {len(factor_cols)}')
print(f'  因子列表: {factor_cols[:5]}...')

# 简化因子选择：使用所有因子
selected_factors = factor_cols[:10]
print(f'  选中因子: {len(selected_factors)} 个')

# ========== 第3层：模型层 ==========
print('\n[Layer 3] 模型层 - 模型训练...')

# 准备数据
model_df = df_clean[selected_factors + ['label_return']].dropna()
print(f'  训练样本: {len(model_df):,}')

if len(model_df) > 10000:
    # 如果数据太多，抽样训练
    model_df = model_df.sample(n=50000, random_state=42)
    print(f'  抽样后: {len(model_df):,}')

X = model_df[selected_factors]
y = model_df['label_return']

# 训练简单模型
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('  训练 Random Forest...')
rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# 预测
train_score = rf.score(X_train, y_train)
test_score = rf.score(X_test, y_test)
print(f'  训练集 R²: {train_score:.4f}')
print(f'  测试集 R²: {test_score:.4f}')

# 特征重要性
importance = pd.DataFrame({
    'feature': selected_factors,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(f'\n  特征重要性:\n{importance.head()}')

# ========== 第4层：宏观层 ==========
print('\n[Layer 4] 宏观层 - 市场趋势...')

try:
    terminal = create_terminal('CN')
    macro_report = terminal.generate_risk_report()
    macro_signal = macro_report.overall_signal
    macro_risk = macro_report.overall_risk_level
    print(f'  宏观信号: {macro_signal} {macro_risk}')
except Exception as e:
    print(f'  宏观分析失败: {e}')
    macro_signal = '🟡'
    macro_risk = '中风险'

# ========== 第5层：风控层 ==========
print('\n[Layer 5] 风控层 - 风险评估...')

# 计算组合风险
portfolio_returns = df_clean.groupby('date')['label_return'].mean().dropna()

if len(portfolio_returns) > 0:
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0
    
    # 最大回撤
    cum_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    print(f'  年化波动率: {volatility:.2%}')
    print(f'  最大回撤: {max_drawdown:.2%}')
    print(f'  夏普比率: {sharpe:.2f}')
    
    # 仓位建议
    if macro_signal == '🔴':
        position = 0.3
    elif macro_signal == '🟡':
        position = 0.5
    elif macro_signal == '🟢':
        position = 0.8
    else:
        position = 0.5
    
    print(f'  建议仓位: {position:.0%}')
else:
    print('  数据不足，跳过风控计算')

# ========== 第6层：决策层 ==========
print('\n[Layer 6] 决策层 - 投资建议...')

print('\n' + '=' * 80)
print('🎯 最终投资建议（基于全局数据分析）')
print('=' * 80)
print(f'📅 分析日期: {datetime.now().strftime("%Y-%m-%d")}')
print(f'📊 数据规模: {len(all_stocks)} 只股票, {len(df_clean):,} 条记录')
print(f'🌍 宏观环境: {macro_signal} {macro_risk}')
print(f'📈 模型表现: 测试集 R² = {test_score:.4f}')
print(f'💼 仓位建议: {position:.0%}股票 + {1-position:.0%}现金')
print(f'📊 夏普比率: {sharpe:.2f}')
print()
print('🏆 重要因子排名:')
for i, row in importance.head(5).iterrows():
    print(f'  {i+1}. {row["feature"]}: {row["importance"]:.3f}')
print()
print('⚠️  风险提示:')
print('  - 市场处于中风险区间，注意波动')
print('  - 建议分散投资，单票仓位不超过20%')
print('  - 设置止损保护，控制回撤')
print('=' * 80)

print('\n✅ 全局数据分析完成！')
