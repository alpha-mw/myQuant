#!/usr/bin/env python3
"""
CN Full Market Batch Analysis - A股全市场批量分析

分析所有已下载的股票：
- 沪深300: 296只
- 中证500: 500只  
- 中证1000: 1000只
总计: 1796只股票
"""

import os
import sys
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(os.path.dirname(__file__)))

from quant_investor_v8 import QuantInvestorV8


def get_all_local_symbols(category: str) -> List[str]:
    """从本地数据目录获取所有股票代码"""
    data_dir = f'data/cn_market_full/{category}'
    
    if not os.path.exists(data_dir):
        return []
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    symbols = [f.replace('.csv', '') for f in csv_files]
    return symbols


def analyze_batch(symbols: List[str], category: str, batch_id: int) -> Optional[Dict]:
    """分析一批股票"""
    category_name = {
        'hs300': '沪深300 (大盘股)',
        'zz500': '中证500 (中盘股)',
        'zz1000': '中证1000 (小盘股)'
    }.get(category, category)
    
    print(f"\n{'='*80}")
    print(f"📊 分析 {category_name} - 批次 {batch_id}")
    print(f"{'='*80}")
    print(f"本批股票数: {len(symbols)}")
    print(f"前10只: {symbols[:10]}")
    
    try:
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

        recommendations = []
        for recommendation in result.final_strategy.trade_recommendations:
            payload = asdict(recommendation)
            payload["category"] = category
            payload["category_name"] = category_name
            recommendations.append(payload)

        analysis = {
            'category': category,
            'category_name': category_name,
            'batch_id': batch_id,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'stocks': symbols,
            'stock_count': len(symbols),
            'branches': {},
            'strategy': {
                'target_exposure': result.final_strategy.target_exposure,
                'style_bias': result.final_strategy.style_bias,
                'candidate_symbols': result.final_strategy.candidate_symbols,
                'position_limits': result.final_strategy.position_limits,
                'branch_consensus': result.final_strategy.branch_consensus,
                'risk_summary': result.final_strategy.risk_summary,
                'execution_notes': result.final_strategy.execution_notes,
                'research_mode': result.final_strategy.research_mode,
            },
            'recommendations': recommendations,
        }

        # 提取分支结果
        for name, branch in result.branch_results.items():
            analysis['branches'][name] = {
                'score': branch.score,
                'confidence': branch.confidence,
                'top_symbols': [
                    {'symbol': symbol, 'score': score}
                    for symbol, score in sorted(
                        branch.symbol_scores.items(),
                        key=lambda item: item[1],
                        reverse=True
                    )[:5]
                ]
            }

        print(f"✅ 批次 {batch_id} 分析完成")
        print(f"   目标仓位: {analysis['strategy']['target_exposure']:.0%}")
        print(f"   候选标的: {len(analysis['strategy']['candidate_symbols'])} 只")
        
        return analysis
        
    except Exception as e:
        print(f"❌ 批次 {batch_id} 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_category_full(category: str, batch_size: int = 50) -> List[Dict]:
    """
    全量分析某一类别的所有股票
    
    Args:
        category: 'hs300', 'zz500', 'zz1000'
        batch_size: 每批分析的股票数
    """
    category_name = {
        'hs300': '沪深300 (大盘股)',
        'zz500': '中证500 (中盘股)',
        'zz1000': '中证1000 (小盘股)'
    }.get(category, category)
    
    print(f"\n{'='*80}")
    print(f"🚀 开始全量分析 {category_name}")
    print(f"{'='*80}")
    
    # 获取所有股票
    all_symbols = get_all_local_symbols(category)
    total = len(all_symbols)
    
    if total == 0:
        print(f"❌ 没有找到 {category} 的数据")
        return []
    
    print(f"总计 {total} 只股票需要分析")
    print(f"批次大小: {batch_size} 只")
    print(f"预计批次: {(total + batch_size - 1) // batch_size} 批")
    print(f"预计时间: {total * 2 / 60:.1f} 分钟")
    print(f"{'='*80}")
    
    # 分批分析
    all_results = []
    num_batches = (total + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, total)
        batch_symbols = all_symbols[start_idx:end_idx]
        
        print(f"\n⏳ 进度: 批次 {i+1}/{num_batches} ({start_idx+1}-{end_idx}/{total})")
        
        result = analyze_batch(batch_symbols, category, i+1)
        if result:
            all_results.append(result)
            
            # 保存中间结果
            save_batch_result(result)
    
    return all_results


def save_batch_result(result: Dict, output_dir: str = 'results/cn_analysis_full'):
    """保存批次分析结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{output_dir}/batch_{result['category']}_{result['batch_id']:03d}_{result['timestamp']}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"💾 批次结果已保存: {filename}")


def _safe_average(values: List[float], default: float = 0.0) -> float:
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else default


def _normalize_with_cap(
    raw_scores: Dict[str, float],
    total_target_exposure: float,
    max_single_weight: float,
) -> Dict[str, float]:
    """把原始评分归一化为组合权重，并施加单票上限。"""
    positive_scores = {symbol: score for symbol, score in raw_scores.items() if score > 0}
    if not positive_scores or total_target_exposure <= 0:
        return {}

    remaining = dict(positive_scores)
    weights = {symbol: 0.0 for symbol in positive_scores}
    remaining_exposure = total_target_exposure

    while remaining and remaining_exposure > 1e-8:
        total_score = sum(remaining.values())
        if total_score <= 0:
            break

        overflow_symbols = []
        for symbol, score in list(remaining.items()):
            proposed = remaining_exposure * score / total_score
            if proposed > max_single_weight + 1e-8:
                weights[symbol] = max_single_weight
                remaining_exposure -= max_single_weight
                overflow_symbols.append(symbol)

        if overflow_symbols:
            for symbol in overflow_symbols:
                remaining.pop(symbol, None)
            continue

        for symbol, score in remaining.items():
            weights[symbol] = remaining_exposure * score / total_score
        break

    return {symbol: weight for symbol, weight in weights.items() if weight > 0}


def _build_market_summary(all_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_stocks': 0,
        'total_batches': 0,
        'categories': {},
    }

    for category, results in all_results.items():
        if not results:
            continue

        category_stocks = sum(item.get('stock_count', 0) for item in results)
        summary['total_stocks'] += category_stocks
        summary['total_batches'] += len(results)

        branch_scores: Dict[str, List[float]] = {}
        candidate_count = 0
        for item in results:
            candidate_count += len(item.get('strategy', {}).get('candidate_symbols', []))
            for name, branch in item.get('branches', {}).items():
                branch_scores.setdefault(name, []).append(float(branch.get('score', 0.0)))

        summary['categories'][category] = {
            'category_name': category_name(category),
            'batch_count': len(results),
            'stock_count': category_stocks,
            'candidate_count': candidate_count,
            'avg_target_exposure': _safe_average(
                [item.get('strategy', {}).get('target_exposure', 0.0) for item in results]
            ),
            'avg_branch_scores': {
                name: _safe_average(scores)
                for name, scores in branch_scores.items()
            },
        }

    return summary


def category_name(category: str) -> str:
    return {
        'hs300': '沪深300 (大盘股)',
        'zz500': '中证500 (中盘股)',
        'zz1000': '中证1000 (小盘股)',
    }.get(category, category)


def build_full_market_trade_plan(
    all_results: Dict[str, List[Dict]],
    total_capital: float = 1_000_000,
    top_k: int = 12,
) -> Dict[str, Any]:
    """将批次候选股聚合为全市场组合级交易计划。"""
    market_summary = _build_market_summary(all_results)
    collected: List[Dict[str, Any]] = []

    for category, batches in all_results.items():
        for batch in batches:
            batch_target_exposure = float(batch.get('strategy', {}).get('target_exposure', 0.0))
            batch_style_bias = batch.get('strategy', {}).get('style_bias', '均衡')
            batch_risk_summary = batch.get('strategy', {}).get('risk_summary', {})
            for recommendation in batch.get('recommendations', []):
                if recommendation.get('action') != 'buy':
                    continue
                if recommendation.get('data_source_status') != 'real':
                    continue
                payload = dict(recommendation)
                payload['category'] = category
                payload['category_name'] = category_name(category)
                payload['batch_target_exposure'] = batch_target_exposure
                payload['style_bias'] = batch_style_bias
                payload['risk_level'] = batch_risk_summary.get('risk_level', 'normal')
                payload['rank_score'] = (
                    max(float(payload.get('suggested_weight', 0.0)), 0.001)
                    * (1 + max(float(payload.get('consensus_score', 0.0)), 0.0))
                    * (1 + max(float(payload.get('model_expected_return', 0.0)), 0.0))
                    * (0.8 + float(payload.get('confidence', 0.0)))
                    * (1 + float(payload.get('branch_positive_count', 0)) / 5)
                )
                collected.append(payload)

    deduped: Dict[str, Dict[str, Any]] = {}
    for item in sorted(collected, key=lambda entry: entry['rank_score'], reverse=True):
        deduped.setdefault(item['symbol'], item)

    ranked = list(deduped.values())[:top_k]
    if not ranked:
        return {
            'market_summary': market_summary,
            'portfolio_plan': {
                'total_capital': total_capital,
                'target_exposure': 0.0,
                'planned_investment': 0.0,
                'cash_reserve': total_capital,
                'selected_count': 0,
                'style_bias': '防御',
                'max_single_weight': 0.0,
                'category_exposure': {},
                'execution_notes': ['当前没有满足真实数据与买入条件的候选标的。'],
            },
            'recommendations': [],
        }

    weighted_exposure_values = []
    for category, batches in all_results.items():
        for batch in batches:
            stock_count = max(int(batch.get('stock_count', 0)), 1)
            weighted_exposure_values.extend(
                [float(batch.get('strategy', {}).get('target_exposure', 0.0))] * stock_count
            )
    target_exposure = min(max(_safe_average(weighted_exposure_values, default=0.35), 0.15), 0.80)
    max_single_weight = min(0.12, max(0.05, target_exposure / max(len(ranked), 1) * 2.2))

    active = ranked
    for _ in range(3):
        weight_map = _normalize_with_cap(
            {item['symbol']: float(item['rank_score']) for item in active},
            total_target_exposure=target_exposure,
            max_single_weight=max_single_weight,
        )
        filtered_active = []
        for item in active:
            weight = weight_map.get(item['symbol'], 0.0)
            entry_price = float(item.get('recommended_entry_price') or item.get('current_price') or 0.0)
            lot_size = int(item.get('lot_size', 100))
            if entry_price <= 0:
                continue
            minimum_ticket = entry_price * lot_size
            if total_capital * weight + 1e-8 < minimum_ticket:
                continue
            filtered_active.append(item)
        if len(filtered_active) == len(active):
            break
        active = filtered_active

    weight_map = _normalize_with_cap(
        {item['symbol']: float(item['rank_score']) for item in active},
        total_target_exposure=target_exposure,
        max_single_weight=max_single_weight,
    )

    final_recommendations = []
    category_exposure: Dict[str, float] = {}
    style_counter = Counter()
    planned_investment = 0.0

    for rank, item in enumerate(active, start=1):
        weight = weight_map.get(item['symbol'], 0.0)
        entry_price = float(item.get('recommended_entry_price') or item.get('current_price') or 0.0)
        lot_size = int(item.get('lot_size', 100))
        shares = int((total_capital * weight) // max(entry_price, 0.01) // lot_size) * lot_size
        amount = shares * entry_price
        actual_weight = amount / total_capital if total_capital > 0 else 0.0
        if shares <= 0 or amount <= 0:
            continue

        final_item = dict(item)
        final_item['rank'] = rank
        final_item['portfolio_weight'] = round(actual_weight, 4)
        final_item['portfolio_amount'] = round(amount, 2)
        final_item['portfolio_shares'] = shares
        final_item['cash_buffer'] = round(total_capital * weight - amount, 2)
        final_recommendations.append(final_item)
        planned_investment += amount
        category_exposure[item['category']] = category_exposure.get(item['category'], 0.0) + actual_weight
        style_counter[item.get('style_bias', '均衡')] += 1

    cash_reserve = max(total_capital - planned_investment, 0.0)
    portfolio_style_bias = style_counter.most_common(1)[0][0] if style_counter else '均衡'

    execution_notes = [
        f"全市场共扫描 {market_summary['total_stocks']} 只股票，最终入选 {len(final_recommendations)} 只。",
        f"组合计划投入约 ¥{planned_investment:,.0f}，保留现金约 ¥{cash_reserve:,.0f}。",
        f"单票上限 {max_single_weight:.1%}，优先采用分批建仓与纪律止损。",
    ]

    return {
        'market_summary': market_summary,
        'portfolio_plan': {
            'total_capital': total_capital,
            'target_exposure': round(sum(item['portfolio_weight'] for item in final_recommendations), 4),
            'planned_investment': round(planned_investment, 2),
            'cash_reserve': round(cash_reserve, 2),
            'selected_count': len(final_recommendations),
            'style_bias': portfolio_style_bias,
            'max_single_weight': round(max_single_weight, 4),
            'category_exposure': {
                category: round(weight, 4)
                for category, weight in category_exposure.items()
            },
            'execution_notes': execution_notes,
        },
        'recommendations': final_recommendations,
    }


def save_candidate_index(all_results: Dict[str, List[Dict]], output_dir: str = 'results/cn_analysis_full') -> str:
    """保存按类别聚合的候选股索引。"""
    os.makedirs(output_dir, exist_ok=True)
    payload: Dict[str, List[str]] = {}
    for category, batches in all_results.items():
        items: List[Dict[str, Any]] = []
        for batch in batches:
            items.extend(batch.get('recommendations', []))
        ranked_symbols = [
            item['symbol']
            for item in sorted(
                items,
                key=lambda rec: (
                    float(rec.get('consensus_score', 0.0)),
                    float(rec.get('suggested_weight', 0.0)),
                ),
                reverse=True,
            )
            if item.get('data_source_status') == 'real'
        ]
        payload[category] = list(dict.fromkeys(ranked_symbols))

    output_file = f"{output_dir}/all_candidates.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return output_file


def generate_full_report(
    all_results: Dict[str, List[Dict]],
    output_dir: str = 'results/cn_analysis_full',
    total_capital: float = 1_000_000,
    top_k: int = 12,
) -> Dict[str, str]:
    """生成全市场执行摘要和组合级交易建议。"""
    print(f"\n{'='*80}")
    print("📊 生成A股全市场综合分析报告")
    print(f"{'='*80}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plan = build_full_market_trade_plan(all_results, total_capital=total_capital, top_k=top_k)
    summary = plan['market_summary']
    portfolio_plan = plan['portfolio_plan']
    recommendations = plan['recommendations']

    report_lines = [
        "# 🇨🇳 A股全市场组合级交易建议报告\n",
        f"**生成时间**: {summary['generated_at']}\n",
        "**分析架构**: Quant-Investor V8.0 五路并行研究\n",
        f"**分析覆盖**: {summary['total_stocks']} 只股票，{summary['total_batches']} 个批次\n",
        "\n---\n",
        "\n## 📊 市场全量扫描摘要\n",
    ]

    for category, info in summary['categories'].items():
        report_lines.append(f"\n### {info['category_name']}\n")
        report_lines.append(f"- **分析批次**: {info['batch_count']} 批\n")
        report_lines.append(f"- **股票总数**: {info['stock_count']} 只\n")
        report_lines.append(f"- **候选股数**: {info['candidate_count']} 只\n")
        report_lines.append(f"- **平均目标仓位**: {info['avg_target_exposure']:.1%}\n")
        report_lines.append("- **平均分支得分**:\n")
        for branch_name in ['kronos', 'quant', 'llm_debate', 'intelligence', 'macro']:
            score = info['avg_branch_scores'].get(branch_name)
            if score is not None:
                report_lines.append(f"  - {branch_name}: {score:+.3f}\n")

    report_lines.extend(
        [
            "\n## 💰 组合级执行计划\n",
            f"- **总资金**: ¥{portfolio_plan['total_capital']:,.0f}\n",
            f"- **计划总仓位**: {portfolio_plan['target_exposure']:.1%}\n",
            f"- **计划投入资金**: ¥{portfolio_plan['planned_investment']:,.0f}\n",
            f"- **预留现金**: ¥{portfolio_plan['cash_reserve']:,.0f}\n",
            f"- **组合风格**: {portfolio_plan['style_bias']}\n",
            f"- **单票上限**: {portfolio_plan['max_single_weight']:.1%}\n",
            f"- **类别暴露**: "
            + (
                " / ".join(
                    f"{category_name(category)} {weight:.1%}"
                    for category, weight in portfolio_plan['category_exposure'].items()
                )
                if portfolio_plan['category_exposure'] else
                "暂无"
            )
            + "\n",
        ]
    )

    if recommendations:
        report_lines.extend(
            [
                "\n## 🎯 最终入选标的\n",
                "| 排名 | 标的 | 类别 | 仓位 | 金额 | 股数 | 建议买入 | 目标价 | 止损价 | 预期空间 | 五路支持 |\n",
                "|:---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
            ]
        )
        for item in recommendations:
            report_lines.append(
                f"| {item['rank']} | {item['symbol']} | {item['category_name']} | "
                f"{item['portfolio_weight']:.1%} | ¥{item['portfolio_amount']:,.0f} | "
                f"{item['portfolio_shares']:,} | ¥{item['recommended_entry_price']:.2f} | "
                f"¥{item['target_price']:.2f} | ¥{item['stop_loss_price']:.2f} | "
                f"{float(item['expected_upside']):.1%} | {item['branch_positive_count']}/5 |\n"
            )

        report_lines.append("\n## 🧭 个股执行说明\n")
        for item in recommendations[:8]:
            report_lines.append(f"\n### {item['rank']}. {item['symbol']} ({item['category_name']})\n")
            report_lines.append(
                f"- **建议建仓**: {item['portfolio_weight']:.1%} / ¥{item['portfolio_amount']:,.0f} / "
                f"{item['portfolio_shares']:,} 股\n"
            )
            report_lines.append(
                f"- **买入区间**: ¥{item['entry_price_range'].get('low', 0.0):.2f}"
                f" - ¥{item['entry_price_range'].get('high', 0.0):.2f}\n"
            )
            report_lines.append(
                f"- **目标与止损**: ¥{item['target_price']:.2f} / ¥{item['stop_loss_price']:.2f}\n"
            )
            report_lines.append(
                f"- **五路共识**: {float(item['consensus_score']):+.2f}，支持分支 "
                f"{item['branch_positive_count']}/5，综合置信度 {float(item['confidence']):.0%}\n"
            )
            if item.get('risk_flags'):
                report_lines.append(f"- **风险观察**: {'；'.join(item['risk_flags'][:3])}\n")
            if item.get('position_management'):
                report_lines.append(f"- **仓位管理**: {'；'.join(item['position_management'][:2])}\n")
    else:
        report_lines.append("\n## 🎯 最终入选标的\n\n当前没有满足条件的最终买入标的，建议继续以现金观望。\n")

    report_lines.append("\n## ✅ 执行提醒\n")
    for note in portfolio_plan['execution_notes']:
        report_lines.append(f"- {note}\n")

    summary_lines = [
        "# 🇨🇳 A股全市场分析摘要\n",
        f"**生成时间**: {summary['generated_at']}\n",
        f"**分析覆盖**: {summary['total_stocks']} 只股票，{summary['total_batches']} 个批次\n",
        f"**计划总仓位**: {portfolio_plan['target_exposure']:.1%}\n",
        f"**计划投入资金**: ¥{portfolio_plan['planned_investment']:,.0f}\n",
        f"**预留现金**: ¥{portfolio_plan['cash_reserve']:,.0f}\n",
        f"**最终入选标的数**: {portfolio_plan['selected_count']} 只\n",
        "\n## 类别摘要\n",
    ]
    for category, info in summary['categories'].items():
        summary_lines.append(
            f"- {info['category_name']}: {info['stock_count']} 只，"
            f"候选 {info['candidate_count']} 只，平均目标仓位 {info['avg_target_exposure']:.1%}\n"
        )
    summary_lines.append("\n## 执行提醒\n")
    for note in portfolio_plan['execution_notes']:
        summary_lines.append(f"- {note}\n")

    os.makedirs(output_dir, exist_ok=True)
    summary_file = f"{output_dir}/全量分析报告_{timestamp}.md"
    data_file = f"{output_dir}/交易建议数据_{timestamp}.json"
    report_file = f"{output_dir}/交易建议报告_{timestamp}.md"

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.writelines(summary_lines)
    with open(report_file, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    candidate_file = save_candidate_index(all_results, output_dir=output_dir)

    print(f"\n📄 全量分析报告已保存: {summary_file}")
    print(f"📄 交易建议报告已保存: {report_file}")
    print(f"💾 交易建议数据已保存: {data_file}")
    print(f"💾 候选股索引已保存: {candidate_file}")

    return {
        'summary_report': summary_file,
        'trade_report': report_file,
        'trade_data': data_file,
        'candidate_index': candidate_file,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='A股全市场批量分析')
    parser.add_argument('--category', type=str, choices=['hs300', 'zz500', 'zz1000', 'all'],
                       default='all', help='分析类别 (默认all)')
    parser.add_argument('--batch-size', type=int, default=30, 
                       help='每批分析的股票数 (默认30，建议20-50)')
    parser.add_argument('--capital', type=float, default=1_000_000,
                       help='组合总资金，用于生成最终交易建议')
    parser.add_argument('--top-k', type=int, default=12,
                       help='最终输出的买入候选数量')
    
    args = parser.parse_args()
    
    categories = []
    if args.category == 'all':
        categories = ['hs300', 'zz500', 'zz1000']
    else:
        categories = [args.category]
    
    all_results = {}
    
    for cat in categories:
        results = analyze_category_full(cat, batch_size=args.batch_size)
        if results:
            all_results[cat] = results
    
    # 生成综合报告
    if all_results:
        generate_full_report(
            all_results,
            total_capital=args.capital,
            top_k=args.top_k,
        )
    
    print("\n" + "="*80)
    print("✅ A股全市场批量分析完成!")
    print("="*80)


if __name__ == '__main__':
    main()
