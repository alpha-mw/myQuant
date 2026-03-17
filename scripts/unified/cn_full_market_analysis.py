#!/usr/bin/env python3
"""
CN Full Market Analysis - A股全市场五路并行研究分析

分析：
- 沪深300 (大盘股)
- 中证500 (中盘股)
- 中证1000 (小盘股)
"""

import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Optional
import random

sys.path.insert(0, str(os.path.dirname(__file__)))

from quant_investor_v8 import QuantInvestorV8


def load_symbols_from_local(category: str, limit: Optional[int] = None) -> List[str]:
    """从本地数据目录加载股票代码"""
    data_dir = f'data/cn_market_full/{category}'
    
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return []
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    symbols = [f.replace('.csv', '') for f in csv_files]
    
    if limit and len(symbols) > limit:
        # 使用固定随机种子保证可重复性
        random.seed(42)
        symbols = random.sample(symbols, limit)
    
    category_name = {
        'hs300': '沪深300',
        'zz500': '中证500',
        'zz1000': '中证1000'
    }.get(category, category)
    
    print(f"📊 {category_name}: 加载 {len(symbols)} 只股票")
    return symbols


def analyze_category(category: str, sample_size: Optional[int] = 50) -> Dict:
    """
    分析某一类别的股票
    
    Args:
        category: 'hs300', 'zz500', 'zz1000'
        sample_size: 抽样数量 (None表示全部)
    """
    category_name = {
        'hs300': '沪深300 (大盘股)',
        'zz500': '中证500 (中盘股)',
        'zz1000': '中证1000 (小盘股)'
    }.get(category, category)
    
    print(f"\n{'='*80}")
    print(f"🔍 分析 {category_name}")
    print(f"{'='*80}")
    
    # 加载股票列表
    symbols = load_symbols_from_local(category, sample_size)
    
    if not symbols:
        print(f"⚠️ 没有找到 {category} 的数据")
        return None
    
    print(f"分析股票数: {len(symbols)}")
    print(f"前10只: {symbols[:10]}")
    
    # 运行五路并行研究分析
    print(f"\n⏳ 启动 Quant-Investor V8.0 分析引擎...")
    
    analyzer = QuantInvestorV8(
        stock_pool=symbols,
        market='CN',
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
        'category_name': category_name,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'stocks': symbols,
        'stock_count': len(symbols),
        'branches': {},
        'strategy': {
            'target_exposure': result.final_strategy.target_exposure,
            'style_bias': result.final_strategy.style_bias,
            'candidate_symbols': result.final_strategy.candidate_symbols
        },
        'report': result.final_report
    }
    
    # 提取分支结果
    for name, branch in result.branch_results.items():
        analysis['branches'][name] = {
            'score': branch.score,
            'confidence': branch.confidence,
            'signals': branch.signals if hasattr(branch, 'signals') else None
        }
    
    # 打印摘要
    print_summary(analysis)
    
    return analysis


def print_summary(analysis: Dict):
    """打印分析摘要"""
    print(f"\n{'='*80}")
    print(f"📊 {analysis['category_name']} 分析结果")
    print(f"{'='*80}")
    
    print(f"\n股票数量: {analysis['stock_count']}")
    print(f"目标仓位: {analysis['strategy']['target_exposure']:.0%}")
    print(f"风格偏好: {analysis['strategy']['style_bias']}")
    print(f"候选标的: {analysis['strategy']['candidate_symbols'][:10]}")
    
    print(f"\n五路并行研究分支:")
    for name, branch in analysis['branches'].items():
        print(f"  {name:12s}: score={branch['score']:+.2f}, conf={branch['confidence']:.0%}")


def save_analysis(analysis: Dict, output_dir: str = 'results/cn_analysis'):
    """保存分析结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存JSON
    json_file = f"{output_dir}/analysis_{analysis['category']}_{analysis['timestamp']}.json"
    
    # 简化数据以节省空间
    save_data = analysis.copy()
    for branch_name, branch_data in save_data.get('branches', {}).items():
        if 'signals' in branch_data and branch_data['signals']:
            signals = branch_data['signals']
            if isinstance(signals, dict):
                branch_data['signals_summary'] = f"{len(signals)} items"
                del branch_data['signals']
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    # 保存报告
    report_file = f"{output_dir}/report_{analysis['category']}_{analysis['timestamp']}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(analysis['report'])
    
    print(f"\n💾 分析结果已保存:")
    print(f"  JSON: {json_file}")
    print(f"  报告: {report_file}")


def generate_comparison_report(all_results: Dict, output_dir: str = 'results/cn_analysis'):
    """生成跨类别对比报告"""
    print(f"\n{'='*80}")
    print("📊 A股跨市场层级对比报告")
    print(f"{'='*80}")
    
    report_lines = [
        "# A股全市场五路并行研究对比报告\n",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "\n## 概述\n",
        "| 市场层级 | 股票数 | 目标仓位 | 风格偏好 | 候选标的数 |",
        "|:---|:---:|:---:|:---:|:---:|:--"
    ]
    
    for category, data in all_results['categories'].items():
        if data is None:
            continue
        strategy = data.get('strategy', {})
        candidates = strategy.get('candidate_symbols', [])
        
        report_lines.append(
            f"| {data['category_name']} | {data['stock_count']} | "
            f"{strategy.get('target_exposure', 0):.0%} | "
            f"{strategy.get('style_bias', 'N/A')} | "
            f"{len(candidates)} |"
        )
    
    report_lines.extend([
        "\n## 五路并行研究分支得分对比\n",
        "| 市场层级 | Kronos | Quant | LLM Debate | Intelligence | Macro |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:--"
    ])
    
    for category, data in all_results['categories'].items():
        if data is None:
            continue
        branches = data.get('branches', {})
        scores = [
            branches.get('kronos', {}).get('score', 0),
            branches.get('quant', {}).get('score', 0),
            branches.get('llm_debate', {}).get('score', 0),
            branches.get('intelligence', {}).get('score', 0),
            branches.get('macro', {}).get('score', 0)
        ]
        
        report_lines.append(
            f"| {data['category_name']} | {scores[0]:+.2f} | {scores[1]:+.2f} | "
            f"{scores[2]:+.2f} | {scores[3]:+.2f} | {scores[4]:+.2f} |"
        )
    
    # 添加详细分析
    report_lines.append("\n## 各市场层级详细分析\n")
    
    for category, data in all_results['categories'].items():
        if data is None:
            continue
        report_lines.append(f"\n### {data['category_name']}\n")
        report_lines.append(f"- **股票数量**: {data['stock_count']}\n")
        report_lines.append(f"- **目标仓位**: {data['strategy']['target_exposure']:.0%}\n")
        report_lines.append(f"- **风格偏好**: {data['strategy']['style_bias']}\n")
        report_lines.append(f"- **候选标的**: {', '.join(data['strategy']['candidate_symbols'][:15])}\n")
        
        # 分支得分
        report_lines.append("\n**五路并行研究分支得分**:\n")
        for name, branch in data['branches'].items():
            report_lines.append(f"- {name}: {branch['score']:+.2f} (置信度: {branch['confidence']:.0%})\n")
    
    # 保存报告
    report_text = '\n'.join(report_lines)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"{output_dir}/comparison_report_{timestamp}.md"
    
    os.makedirs(output_dir, exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n📄 对比报告已保存: {report_file}")


def run_full_analysis(
    analyze_hs300: bool = True,
    analyze_zz500: bool = True,
    analyze_zz1000: bool = True,
    sample_per_category: int = 50
):
    """
    运行A股全市场完整分析
    
    Args:
        analyze_hs300: 分析沪深300
        analyze_zz500: 分析中证500
        analyze_zz1000: 分析中证1000
        sample_per_category: 每类分析的股票数量
    """
    print("=" * 80)
    print("🇨🇳 A股全市场五路并行研究分析")
    print("=" * 80)
    print(f"\n分析配置:")
    print(f"  - 沪深300: {'✅' if analyze_hs300 else '❌'} (抽样 {sample_per_category} 只)")
    print(f"  - 中证500: {'✅' if analyze_zz500 else '❌'} (抽样 {sample_per_category} 只)")
    print(f"  - 中证1000: {'✅' if analyze_zz1000 else '❌'} (抽样 {sample_per_category} 只)")
    print(f"  - 架构: Quant-Investor V8.0 五路并行研究")
    print("=" * 80)
    
    all_results = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'config': {
            'sample_per_category': sample_per_category
        },
        'categories': {}
    }
    
    # 分析沪深300
    if analyze_hs300:
        analysis = analyze_category('hs300', sample_size=sample_per_category)
        if analysis:
            save_analysis(analysis)
            all_results['categories']['hs300'] = analysis
    
    # 分析中证500
    if analyze_zz500:
        analysis = analyze_category('zz500', sample_size=sample_per_category)
        if analysis:
            save_analysis(analysis)
            all_results['categories']['zz500'] = analysis
    
    # 分析中证1000
    if analyze_zz1000:
        analysis = analyze_category('zz1000', sample_size=sample_per_category)
        if analysis:
            save_analysis(analysis)
            all_results['categories']['zz1000'] = analysis
    
    # 生成对比报告
    if len(all_results['categories']) > 1:
        generate_comparison_report(all_results)
    
    print("\n" + "=" * 80)
    print("✅ A股全市场分析完成!")
    print("=" * 80)
    
    return all_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='A股全市场五路并行研究分析')
    parser.add_argument('--all', action='store_true', help='分析全部三类市场')
    parser.add_argument('--hs300', action='store_true', help='分析沪深300')
    parser.add_argument('--zz500', action='store_true', help='分析中证500')
    parser.add_argument('--zz1000', action='store_true', help='分析中证1000')
    parser.add_argument('--sample', type=int, default=50, help='每类分析股票数 (默认50)')
    
    args = parser.parse_args()
    
    if args.all:
        run_full_analysis(
            analyze_hs300=True,
            analyze_zz500=True,
            analyze_zz1000=True,
            sample_per_category=args.sample
        )
    elif args.hs300 or args.zz500 or args.zz1000:
        run_full_analysis(
            analyze_hs300=args.hs300,
            analyze_zz500=args.zz500,
            analyze_zz1000=args.zz1000,
            sample_per_category=args.sample
        )
    else:
        # 默认分析沪深300
        run_full_analysis(
            analyze_hs300=True,
            analyze_zz500=False,
            analyze_zz1000=False,
            sample_per_category=args.sample
        )
