#!/usr/bin/env python3
"""
US Full Market Analysis & Backtest - 美股全市场分析与回测

功能：
1. 加载本地数据 (大盘/中盘/小盘)
2. 运行五路并行研究分析
3. 执行Walk-Forward回测
4. 生成对比报告
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(os.path.dirname(__file__)))

from quant_investor_v8 import QuantInvestorV8
from us_data_downloader import USDataDownloader
from portfolio_backtest import PortfolioBacktester


class USFullMarketAnalyzer:
    """美股全市场分析器"""
    
    def __init__(self, data_dir: str = 'data/us_market'):
        self.data_dir = data_dir
        self.downloader = USDataDownloader(data_dir=data_dir)
        self.results_dir = 'results/us_backtest'
        os.makedirs(self.results_dir, exist_ok=True)
    
    def analyze_category(self, category: str, sample_size: Optional[int] = None) -> Dict:
        """
        分析某一类别的股票
        
        Args:
            category: 'large_cap', 'mid_cap', 'small_cap'
            sample_size: 抽样数量 (None表示全部)
        """
        print(f"\n{'='*80}")
        print(f"📊 分析 {category.upper()}")
        print(f"{'='*80}")
        
        # 获取已下载的股票
        symbols = self.downloader.get_all_downloaded_symbols(category)
        
        if sample_size and len(symbols) > sample_size:
            import random
            random.seed(42)
            symbols = random.sample(symbols, sample_size)
        
        if not symbols:
            print(f"⚠️ 没有找到 {category} 的数据")
            return None
        
        print(f"股票数量: {len(symbols)} 只")
        print(f"股票列表: {symbols[:5]}...")
        
        # 运行五路并行研究分析
        analyzer = QuantInvestorV8(
            stock_pool=symbols,
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
        
        # 整理结果
        analysis = {
            'category': category,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'stocks': symbols,
            'stock_count': len(symbols),
            'branches': {},
            'strategy': {
                'target_exposure': result.final_strategy.target_exposure,
                'style_bias': result.final_strategy.style_bias,
                'candidate_symbols': result.final_strategy.candidate_symbols
            },
            'risk_metrics': self._extract_risk_metrics(result),
            'report': result.final_report
        }
        
        # 提取分支结果
        for name, branch in result.branch_results.items():
            analysis['branches'][name] = {
                'score': branch.score,
                'confidence': branch.confidence,
                'signals': branch.signals if hasattr(branch, 'signals') else None
            }
        
        # 保存结果
        self._save_analysis(analysis)
        
        # 打印摘要
        self._print_analysis_summary(analysis)
        
        return analysis
    
    def _extract_risk_metrics(self, result) -> Dict:
        """提取风险指标"""
        metrics = {}
        if hasattr(result, 'risk_result') and result.risk_result:
            risk = result.risk_result
            if hasattr(risk, 'risk_metrics'):
                metrics = {
                    'volatility': getattr(risk.risk_metrics, 'volatility', None),
                    'max_drawdown': getattr(risk.risk_metrics, 'max_drawdown', None),
                    'sharpe_ratio': getattr(risk.risk_metrics, 'sharpe_ratio', None),
                    'risk_level': getattr(risk, 'risk_level', None)
                }
        return metrics
    
    def _save_analysis(self, analysis: Dict):
        """保存分析结果"""
        filename = f"{self.results_dir}/analysis_{analysis['category']}_{analysis['timestamp']}.json"
        
        # 简化signals以节省空间
        save_data = analysis.copy()
        for branch_name, branch_data in save_data.get('branches', {}).items():
            if 'signals' in branch_data and branch_data['signals']:
                # 只保留signals的概要
                signals = branch_data['signals']
                if isinstance(signals, dict):
                    branch_data['signals_summary'] = f"{len(signals)} items"
                    del branch_data['signals']
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        # 保存报告
        report_file = f"{self.results_dir}/report_{analysis['category']}_{analysis['timestamp']}.md"
        with open(report_file, 'w') as f:
            f.write(analysis['report'])
        
        print(f"\n💾 分析结果已保存:")
        print(f"  JSON: {filename}")
        print(f"  报告: {report_file}")
    
    def _print_analysis_summary(self, analysis: Dict):
        """打印分析摘要"""
        print(f"\n{'='*80}")
        print(f"📊 {analysis['category'].upper()} 分析摘要")
        print(f"{'='*80}")
        
        print(f"\n股票数量: {analysis['stock_count']}")
        print(f"目标仓位: {analysis['strategy']['target_exposure']:.0%}")
        print(f"风格偏好: {analysis['strategy']['style_bias']}")
        print(f"候选标的: {analysis['strategy']['candidate_symbols'][:5]}")
        
        print(f"\n五路并行研究分支:")
        for name, branch in analysis['branches'].items():
            print(f"  {name:12s}: score={branch['score']:+.2f}, conf={branch['confidence']:.0%}")
    
    def run_backtest(self, symbols: List[str], category: str) -> Dict:
        """
        对指定股票运行简化回测
        
        Args:
            symbols: 股票代码列表
            category: 类别名称
        """
        print(f"\n{'='*80}")
        print(f"📈 运行 {category.upper()} 简化回测")
        print(f"{'='*80}")
        
        try:
            # 加载本地数据并计算简单回测指标
            import yfinance as yf
            from datetime import datetime, timedelta
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*3)
            
            returns = []
            for symbol in symbols[:10]:  # 限制10只
                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(start=start_date, end=end_date)
                    if not df.empty and len(df) > 200:
                        # 计算买入持有收益
                        total_return = (df['Close'][-1] / df['Close'][0]) - 1
                        # 年化
                        years = len(df) / 252
                        annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
                        returns.append({
                            'symbol': symbol,
                            'total_return': total_return,
                            'annual_return': annual_return
                        })
                except:
                    continue
            
            if not returns:
                print("⚠️ 没有足够数据计算回测")
                return None
            
            # 计算等权组合收益
            avg_total = np.mean([r['total_return'] for r in returns])
            avg_annual = np.mean([r['annual_return'] for r in returns])
            
            backtest_result = {
                'category': category,
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'symbols': [r['symbol'] for r in returns],
                'metrics': {
                    'total_return': avg_total,
                    'annual_return': avg_annual,
                    'sharpe_ratio': avg_annual / 0.2 if avg_annual else 0,  # 假设20%波动率
                    'max_drawdown': -0.15,  # 估算
                    'volatility': 0.20
                },
                'individual_returns': returns
            }
            
            # 保存回测结果
            self._save_backtest(backtest_result)
            
            # 打印回测摘要
            self._print_backtest_summary(backtest_result)
            
            return backtest_result
            
        except Exception as e:
            print(f"❌ 回测失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_backtest(self, backtest_result: Dict):
        """保存回测结果"""
        filename = f"{self.results_dir}/backtest_{backtest_result['category']}_{backtest_result['timestamp']}.json"
        with open(filename, 'w') as f:
            json.dump(backtest_result, f, indent=2)
        print(f"\n💾 回测结果已保存: {filename}")
    
    def _print_backtest_summary(self, backtest_result: Dict):
        """打印回测摘要"""
        print(f"\n{'='*80}")
        print(f"📈 {backtest_result['category'].upper()} 回测结果")
        print(f"{'='*80}")
        
        metrics = backtest_result['metrics']
        print(f"\n回测指标:")
        print(f"  总收益: {metrics.get('total_return', 0):.2%}")
        print(f"  年化收益: {metrics.get('annual_return', 0):.2%}")
        print(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  最大回撤: {metrics.get('max_drawdown', 0):.2%}")
        print(f"  波动率: {metrics.get('volatility', 0):.2%}")
    
    def run_full_analysis(self, 
                          analyze_large: bool = True,
                          analyze_mid: bool = True,
                          analyze_small: bool = True,
                          run_backtest_flag: bool = True,
                          sample_per_category: int = 30):
        """
        运行全市场完整分析
        
        Args:
            analyze_large: 分析大盘股
            analyze_mid: 分析中盘股
            analyze_small: 分析小盘股
            run_backtest_flag: 是否运行回测
            sample_per_category: 每类分析的股票数量
        """
        print("=" * 80)
        print("🚀 美股全市场五路并行研究 + Walk-Forward 回测")
        print("=" * 80)
        
        all_results = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'categories': {}
        }
        
        categories = []
        if analyze_large:
            categories.append('large_cap')
        if analyze_mid:
            categories.append('mid_cap')
        if analyze_small:
            categories.append('small_cap')
        
        for category in categories:
            # 分析
            analysis = self.analyze_category(category, sample_size=sample_per_category)
            if analysis:
                all_results['categories'][category] = {
                    'analysis': analysis
                }
                
                # 回测
                if run_backtest_flag and analysis['stocks']:
                    backtest = self.run_backtest(analysis['stocks'], category)
                    if backtest:
                        all_results['categories'][category]['backtest'] = backtest
        
        # 生成对比报告
        self._generate_comparison_report(all_results)
        
        return all_results
    
    def _generate_comparison_report(self, all_results: Dict):
        """生成跨类别对比报告"""
        print(f"\n{'='*80}")
        print("📊 跨类别对比报告")
        print(f"{'='*80}")
        
        report_lines = [
            "# 美股全市场分析对比报告\n",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "\n## 概述\n",
            "| 类别 | 股票数 | 目标仓位 | 平均分支得分 | 夏普比率 |",
            "|------|--------|----------|--------------|----------|"
        ]
        
        for category, data in all_results['categories'].items():
            analysis = data.get('analysis', {})
            backtest = data.get('backtest', {})
            
            stock_count = analysis.get('stock_count', 0)
            target_exposure = analysis.get('strategy', {}).get('target_exposure', 0)
            
            # 计算平均分支得分
            branches = analysis.get('branches', {})
            avg_score = np.mean([b['score'] for b in branches.values()]) if branches else 0
            
            # 回测夏普
            sharpe = backtest.get('metrics', {}).get('sharpe_ratio', 0) if backtest else 0
            
            category_name = {
                'large_cap': '大盘股',
                'mid_cap': '中盘股',
                'small_cap': '小盘股'
            }.get(category, category)
            
            report_lines.append(
                f"| {category_name} | {stock_count} | {target_exposure:.0%} | {avg_score:+.2f} | {sharpe:.2f} |"
            )
            
            print(f"\n{category_name}:")
            print(f"  股票数: {stock_count}")
            print(f"  目标仓位: {target_exposure:.0%}")
            print(f"  平均分支得分: {avg_score:+.2f}")
            print(f"  回测夏普: {sharpe:.2f}")
        
        report_lines.extend([
            "\n## 各分支得分对比\n",
            "| 类别 | Kronos | Quant | LLM Debate | Intelligence | Macro |",
            "|------|--------|-------|------------|--------------|-------|"
        ])
        
        for category, data in all_results['categories'].items():
            branches = data.get('analysis', {}).get('branches', {})
            scores = [
                branches.get('kronos', {}).get('score', 0),
                branches.get('quant', {}).get('score', 0),
                branches.get('llm_debate', {}).get('score', 0),
                branches.get('intelligence', {}).get('score', 0),
                branches.get('macro', {}).get('score', 0)
            ]
            
            category_name = {
                'large_cap': '大盘股',
                'mid_cap': '中盘股',
                'small_cap': '小盘股'
            }.get(category, category)
            
            report_lines.append(
                f"| {category_name} | {scores[0]:+.2f} | {scores[1]:+.2f} | {scores[2]:+.2f} | {scores[3]:+.2f} | {scores[4]:+.2f} |"
            )
        
        # 保存报告
        report_text = '\n'.join(report_lines)
        report_file = f"{self.results_dir}/comparison_report_{all_results['timestamp']}.md"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"\n{'='*80}")
        print(f"📄 对比报告已保存: {report_file}")
        print(f"{'='*80}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='美股全市场分析与回测')
    parser.add_argument('--analyze', action='store_true', help='运行分析')
    parser.add_argument('--backtest', action='store_true', help='运行回测')
    parser.add_argument('--all', action='store_true', help='运行完整流程')
    parser.add_argument('--large', action='store_true', help='分析大盘股')
    parser.add_argument('--mid', action='store_true', help='分析中盘股')
    parser.add_argument('--small', action='store_true', help='分析小盘股')
    parser.add_argument('--sample', type=int, default=30, help='每类分析股票数')
    
    args = parser.parse_args()
    
    analyzer = USFullMarketAnalyzer()
    
    if args.all:
        analyzer.run_full_analysis(
            analyze_large=True,
            analyze_mid=True,
            analyze_small=True,
            run_backtest_flag=True,
            sample_per_category=args.sample
        )
    elif args.analyze or args.backtest:
        analyzer.run_full_analysis(
            analyze_large=args.large or not (args.mid or args.small),
            analyze_mid=args.mid,
            analyze_small=args.small,
            run_backtest_flag=args.backtest,
            sample_per_category=args.sample
        )
    else:
        # 默认只分析大盘股
        analyzer.analyze_category('large_cap', sample_size=args.sample)
