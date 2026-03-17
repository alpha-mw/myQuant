#!/usr/bin/env python3
"""
Legacy Backtrader 回测脚本（V7 兼容保留）

该脚本不是 V8 主线的可信回测入口；V8 主线请使用 `portfolio_backtest.py`。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3

from backtest_engine import BacktestEngine, MomentumStrategy, RiskManagedStrategy
from logging_config import get_logger

logger = get_logger("backtest_runner")


def load_stock_data(db_path: str, symbols: list, start_date: str, end_date: str) -> dict:
    """
    从数据库加载股票数据
    """
    conn = sqlite3.connect(db_path)
    data_dict = {}
    
    for symbol in symbols:
        query = f"""
        SELECT ts_code as symbol, trade_date as date, 
               open, high, low, close, volume, amount
        FROM daily_data 
        WHERE ts_code = '{symbol}' 
          AND trade_date >= '{start_date}' 
          AND trade_date <= '{end_date}'
        ORDER BY trade_date
        """
        
        df = pd.read_sql_query(query, conn)
        
        if len(df) > 30:  # 至少需要30天数据
            df['date'] = pd.to_datetime(df['date'])
            data_dict[symbol] = df
            logger.info(f"加载 {symbol}: {len(df)} 条记录")
        else:
            logger.warning(f"{symbol} 数据不足，跳过")
    
    conn.close()
    return data_dict


def run_full_backtest():
    """
    运行完整回测
    """
    logger.info("=" * 80)
    logger.info("Quant-Investor V7.0 - Backtrader事件驱动回测")
    logger.info("=" * 80)
    
    # 配置
    DB_PATH = '/root/.openclaw/workspace/myQuant/data/stock_database.db'
    INITIAL_CASH = 1000000.0  # 初始资金100万
    START_DATE = '20240101'
    END_DATE = '20250301'
    
    # 测试股票列表（前15只推荐股票）
    test_symbols = [
        '600808.SH', '002938.SZ', '300579.SZ', '688578.SH', '688696.SH',
        '000792.SZ', '300433.SZ', '301200.SZ', '002698.SZ', '300866.SZ',
        '002472.SZ', '002073.SZ', '600366.SH', '002967.SZ', '605118.SH'
    ]
    
    logger.info(f"回测区间: {START_DATE} - {END_DATE}")
    logger.info(f"初始资金: {INITIAL_CASH:,.2f}")
    logger.info(f"测试股票: {len(test_symbols)} 只")
    
    # 加载数据
    logger.info("\n[1/4] 加载股票数据...")
    data_dict = load_stock_data(DB_PATH, test_symbols, START_DATE, END_DATE)
    
    if len(data_dict) < 5:
        logger.error("可用股票数量不足，无法运行回测")
        return
    
    # 创建回测引擎
    logger.info("\n[2/4] 初始化回测引擎...")
    engine = BacktestEngine(
        initial_cash=INITIAL_CASH,
        commission=0.0003,      # 手续费0.03%
        stamp_duty=0.001,       # 印花税0.1%
        slippage=0.001          # 滑点0.1%
    )
    
    # 添加数据
    logger.info("\n[3/4] 添加数据到回测引擎...")
    for symbol, df in data_dict.items():
        engine.add_data(df, symbol)
    
    # 运行回测
    logger.info("\n[4/4] 运行回测...")
    logger.info("-" * 80)
    
    result = engine.run(
        RiskManagedStrategy,
        max_positions=15,
        max_drawdown=0.15,
        position_pct=0.07,
    )
    
    # 输出结果
    logger.info("-" * 80)
    logger.info("\n📊 回测结果汇总:")
    logger.info(f"  初始资金: {result['initial_value']:,.2f}")
    logger.info(f"  最终资金: {result['final_value']:,.2f}")
    logger.info(f"  总收益率: {result['total_return']*100:.2f}%")
    logger.info(f"  夏普比率: {result['sharpe_ratio']:.2f}")
    logger.info(f"  最大回撤: {result['max_drawdown']*100:.2f}%")
    
    # 交易统计
    trades = result.get('trades', {})
    if trades:
        total_trades = trades.get('total', {}).get('total', 0)
        won_trades = trades.get('won', {}).get('total', 0)
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
        
        logger.info(f"\n📈 交易统计:")
        logger.info(f"  总交易次数: {total_trades}")
        logger.info(f"  盈利次数: {won_trades}")
        logger.info(f"  胜率: {win_rate:.1f}%")
    
    logger.info("\n" + "=" * 80)
    logger.info("回测完成!")
    logger.info("=" * 80)
    
    return result


def run_simple_test():
    """
    运行简单测试（单只股票）
    """
    logger.info("运行简单回测测试...")
    
    DB_PATH = '/root/.openclaw/workspace/myQuant/data/stock_database.db'
    
    # 加载单只股票数据
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT ts_code as symbol, trade_date as date, open, high, low, close, volume "
        "FROM daily_data WHERE ts_code = '000001.SZ' AND trade_date >= '20240101' "
        "ORDER BY trade_date",
        conn
    )
    conn.close()
    
    if len(df) < 30:
        logger.error("数据不足")
        return
    
    df['date'] = pd.to_datetime(df['date'])
    
    # 创建引擎并运行
    engine = BacktestEngine(initial_cash=100000)
    engine.add_data(df, '000001.SZ')
    
    result = engine.run(MomentumStrategy)
    
    logger.info(f"测试完成，收益率: {result['total_return']*100:.2f}%")
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Quant-Investor V7.0 回测系统')
    parser.add_argument('--test', action='store_true', help='运行简单测试')
    parser.add_argument('--full', action='store_true', help='运行完整回测')
    
    args = parser.parse_args()
    
    if args.test:
        run_simple_test()
    elif args.full:
        run_full_backtest()
    else:
        # 默认运行完整回测
        run_full_backtest()
