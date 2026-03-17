#!/usr/bin/env python3
"""
美股全市场分析 - 下载过去3年数据并进行五路并行研究分析
"""

import sys
import os
import warnings
from datetime import datetime, timedelta
from typing import List
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, str(os.path.dirname(__file__)))

from quant_investor_v8 import QuantInvestorV8
from enhanced_data_layer import EnhancedDataLayer
import yfinance as yf


# 美股大盘股列表 (作为"全市场"代表)
US_STOCK_UNIVERSE = [
    # 科技巨头 (Mag7)
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA',
    # 半导体
    'AVGO', 'AMD', 'INTC', 'QCOM', 'AMAT', 'MU', 'LRCX', 'ADI', 'KLAC', 'MRVL', 'NXPI', 'SNPS', 'CDNS',
    # 软件/互联网
    'ORCL', 'ADBE', 'CRM', 'NFLX', 'INTU', 'NOW', 'PANW', 'SNOW', 'ZM', 'UBER', 'LYFT', 'ABNB', 'DDOG', 'CRWD',
    # 金融
    'BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'C', 'AXP', 'PNC', 'USB',
    # 医疗保健
    'LLY', 'JNJ', 'UNH', 'ABBV', 'MRK', 'PFE', 'TMO', 'VRTX', 'REGN', 'GILD', 'AMGN', 'BIIB', 'DHR', 'ABT',
    # 消费品
    'WMT', 'PG', 'COST', 'KO', 'PEP', 'MCD', 'HD', 'LOW', 'NKE', 'SBUX', 'TGT', 'DG',
    # 能源
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX',
    # 工业
    'GE', 'CAT', 'BA', 'HON', 'UPS', 'RTX', 'LMT', 'MMM', 'DE', 'CSX', 'UNP', 'FDX',
    # 通信
    'VZ', 'T', 'CMCSA', 'CHTR', 'TMUS',
    # 材料
    'LIN', 'APD', 'SHW', 'FCX', 'NEM', 'DOW', 'DD',
    # 房地产
    'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'O', 'WELL', 'SPG',
    # 公用事业
    'NEE', 'SO', 'DUK', 'AEP', 'EXC', 'SRE'
]


def download_us_data(symbols: List[str], years: int = 3, save_dir: str = 'data/us_market'):
    """
    下载美股历史数据
    
    Args:
        symbols: 股票代码列表
        years: 下载年数
        save_dir: 保存目录
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    print(f"=" * 80)
    print(f"📥 美股数据下载")
    print(f"=" * 80)
    print(f"股票数量: {len(symbols)} 只")
    print(f"时间范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    print(f"=" * 80)
    
    success_count = 0
    failed_symbols = []
    
    for i, symbol in enumerate(symbols, 1):
        try:
            print(f"[{i}/{len(symbols)}] 下载 {symbol}...", end=" ")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if df.empty:
                print("❌ 无数据")
                failed_symbols.append(symbol)
                continue
            
            # 保存为CSV
            df.reset_index(inplace=True)
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            df.to_csv(f"{save_dir}/{symbol}.csv", index=False)
            print(f"✅ {len(df)} 条")
            success_count += 1
            
        except Exception as e:
            print(f"❌ 失败: {e}")
            failed_symbols.append(symbol)
    
    print(f"=" * 80)
    print(f"下载完成: {success_count}/{len(symbols)} 只成功")
    if failed_symbols:
        print(f"失败列表: {failed_symbols}")
    print(f"=" * 80)
    
    return success_count, failed_symbols


def analyze_us_market(sample_size: int = 50, years: int = 3):
    """
    对美股进行五路并行研究分析
    
    Args:
        sample_size: 抽样分析的股票数量 (全市场太多，抽样分析)
        years: 数据年数
    """
    print("=" * 80)
    print("🚀 美股全市场五路并行研究分析")
    print("=" * 80)
    print(f"\n📊 分析配置:")
    print(f"  - 股票池: 美股大盘股 {sample_size} 只 (从 {len(US_STOCK_UNIVERSE)} 只中抽样)")
    print(f"  - 时间范围: {years} 年")
    print(f"  - 市场: US")
    print(f"  - 架构: V8.0 五路并行研究")
    print("=" * 80)
    
    # 抽样
    import random
    random.seed(42)
    sampled_stocks = random.sample(US_STOCK_UNIVERSE, min(sample_size, len(US_STOCK_UNIVERSE)))
    
    print(f"\n📈 分析股票列表:")
    for i, stock in enumerate(sampled_stocks, 1):
        print(f"  {i:2d}. {stock}")
    
    # 运行五路并行研究分析
    print("\n" + "=" * 80)
    print("开始五路并行研究分析...")
    print("=" * 80)
    
    analyzer = QuantInvestorV8(
        stock_pool=sampled_stocks,
        market='US',
        total_capital=1_000_000,
        risk_level='中等',
        enable_macro=True,
        enable_kronos=True,
        enable_intelligence=True,
        enable_llm_debate=True,
        verbose=True
    )
    
    result = analyzer.run()
    
    # 输出详细结果
    print("\n" + "=" * 80)
    print("📊 五路并行研究分析结果")
    print("=" * 80)
    
    print("\n【五大研究分支结果】")
    for name, branch in result.branch_results.items():
        print(f"\n{name.upper()}:")
        print(f"  得分: {branch.score:+.2f}")
        print(f"  置信度: {branch.confidence:.0%}")
        if hasattr(branch, 'signals') and branch.signals:
            print(f"  信号: {branch.signals}")
        if hasattr(branch, 'summary') and branch.summary:
            print(f"  摘要: {branch.summary[:100]}...")
    
    print("\n【集成裁判结果】")
    strategy = result.final_strategy
    print(f"  目标敞口: {strategy.target_exposure:.0%}")
    print(f"  风格偏好: {strategy.style_bias}")
    print(f"  候选标的: {strategy.candidate_symbols}")
    
    print("\n【完整报告】")
    print(result.final_report)
    
    # 保存结果
    save_results(result, sampled_stocks)
    
    return result, sampled_stocks


def save_results(result, stocks: List[str], output_dir: str = 'results'):
    """保存分析结果"""
    import os
    import json
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存报告
    report_file = f"{output_dir}/us_market_report_{timestamp}.md"
    with open(report_file, 'w') as f:
        f.write(result.final_report)
    print(f"\n📄 报告已保存: {report_file}")
    
    # 保存分支结果摘要
    summary = {
        'timestamp': timestamp,
        'stocks': stocks,
        'market': 'US',
        'branches': {},
        'final_strategy': {
            'target_exposure': result.final_strategy.target_exposure,
            'style_bias': result.final_strategy.style_bias,
            'candidate_symbols': result.final_strategy.candidate_symbols
        }
    }
    
    for name, branch in result.branch_results.items():
        summary['branches'][name] = {
            'score': branch.score,
            'confidence': branch.confidence,
            'signals': branch.signals if hasattr(branch, 'signals') else None
        }
    
    summary_file = f"{output_dir}/us_market_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📊 摘要已保存: {summary_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='美股全市场分析')
    parser.add_argument('--download', action='store_true', help='先下载数据')
    parser.add_argument('--analyze', action='store_true', help='执行分析')
    parser.add_argument('--sample', type=int, default=50, help='抽样股票数 (默认50)')
    parser.add_argument('--years', type=int, default=3, help='数据年数 (默认3)')
    
    args = parser.parse_args()
    
    if args.download:
        download_us_data(US_STOCK_UNIVERSE, years=args.years)
    
    if args.analyze or not args.download:
        analyze_us_market(sample_size=args.sample, years=args.years)
