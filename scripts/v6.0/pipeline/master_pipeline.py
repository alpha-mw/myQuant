#!/usr/bin/env python3
"""
Quant-Investor V6.0 - Master Pipeline (统一流水线)

V6.0的唯一入口，严格按照分层架构串联所有模块：
  数据层 → 因子层 → 模型层 → 决策层 → 风控层 → 报告生成

用户只需提供市场、股票池等最简参数，即可触发完整的端到端分析。
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path

# 添加V6.0模块路径
V6_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if V6_DIR not in sys.path:
    sys.path.insert(0, V6_DIR)

from data_layer.unified_data_layer import UnifiedDataLayer, UnifiedDataBundle
from factor_layer.unified_factor_layer import UnifiedFactorLayer, FactorLayerOutput
from model_layer.unified_model_layer import UnifiedModelLayer, ModelLayerOutput
from decision_layer.unified_decision_layer import UnifiedDecisionLayer, DecisionLayerOutput
from risk_layer.unified_risk_layer import UnifiedRiskLayer, RiskLayerOutput


# ==================== 报告生成器 ====================

@dataclass
class PipelineReport:
    """流水线最终报告"""
    # 基本信息
    market: str = ""
    run_date: str = ""
    duration_seconds: float = 0.0
    
    # 各层摘要
    data_stats: Dict = field(default_factory=dict)
    factor_stats: Dict = field(default_factory=dict)
    model_stats: Dict = field(default_factory=dict)
    decision_stats: Dict = field(default_factory=dict)
    risk_stats: Dict = field(default_factory=dict)
    
    # 核心结果
    final_recommendations: List[Dict] = field(default_factory=list)
    portfolio_weights: Dict[str, float] = field(default_factory=dict)
    risk_alerts: List[str] = field(default_factory=list)
    
    # 详细结果引用
    data_bundle: Any = None
    factor_output: Any = None
    model_output: Any = None
    decision_output: Any = None
    risk_output: Any = None


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir or os.path.expanduser("~/.quant_investor/reports"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_markdown(self, report: PipelineReport) -> str:
        """生成Markdown格式的完整报告"""
        lines = []
        
        # 标题
        lines.append(f"# 📊 Quant-Investor V6.0 投资分析报告")
        lines.append(f"")
        lines.append(f"**生成时间**: {report.run_date}")
        lines.append(f"**市场**: {report.market}")
        lines.append(f"**分析耗时**: {report.duration_seconds:.1f} 秒")
        lines.append(f"")
        
        # 执行摘要
        lines.append(f"## 📋 执行摘要")
        lines.append(f"")
        lines.append(f"| 分析阶段 | 关键指标 |")
        lines.append(f"|---------|---------|")
        lines.append(f"| 数据层 | 股票: {report.data_stats.get('valid_stocks', 'N/A')} 只, "
                     f"日期范围: {report.data_stats.get('date_range', 'N/A')} |")
        lines.append(f"| 因子层 | 有效因子: {report.factor_stats.get('effective_factors', 'N/A')} 个, "
                     f"最佳因子: {report.factor_stats.get('top_factor', 'N/A')} |")
        lines.append(f"| 模型层 | 训练模型: {report.model_stats.get('models_trained', 'N/A')} 个, "
                     f"最佳模型: {report.model_stats.get('best_model', 'N/A')} |")
        lines.append(f"| 决策层 | 深度分析: {report.decision_stats.get('stocks_analyzed', 'N/A')} 只, "
                     f"LLM: {report.decision_stats.get('llm_providers', 'N/A')} |")
        lines.append(f"| 风控层 | 组合方法: {report.risk_stats.get('optimization_method', 'N/A')}, "
                     f"夏普: {report.risk_stats.get('sharpe_ratio', 0):.2f} |")
        lines.append(f"")
        
        # 最终推荐
        lines.append(f"## 🏆 最终投资推荐")
        lines.append(f"")
        
        if report.final_recommendations:
            lines.append(f"| 排名 | 代码 | 名称 | 量化得分 | 定性评分 | 综合得分 | 投资评级 | 权重 |")
            lines.append(f"|------|------|------|---------|---------|---------|---------|------|")
            
            for i, rec in enumerate(report.final_recommendations, 1):
                code = rec.get('code', '')
                name = rec.get('name', code)
                quant = rec.get('quant_score', 0)
                qual = rec.get('qualitative_score', 0)
                final = rec.get('final_score', 0)
                rating = rec.get('investment_rating', '待分析')
                weight = report.portfolio_weights.get(code, 0)
                
                lines.append(f"| {i} | {code} | {name} | {quant:.3f} | "
                           f"{qual:.1f}/10 | {final:.3f} | {rating} | {weight:.1%} |")
            lines.append(f"")
        
        # 组合概况
        if report.risk_stats:
            lines.append(f"## 📈 组合概况")
            lines.append(f"")
            lines.append(f"| 指标 | 值 |")
            lines.append(f"|------|-----|")
            lines.append(f"| 预期年化收益 | {report.risk_stats.get('expected_return', 0):.2%} |")
            lines.append(f"| 预期年化波动 | {report.risk_stats.get('expected_volatility', 0):.2%} |")
            lines.append(f"| 夏普比率 | {report.risk_stats.get('sharpe_ratio', 0):.2f} |")
            lines.append(f"| 最大回撤 | {report.risk_stats.get('max_drawdown', 0):.2%} |")
            lines.append(f"")
        
        # 决策层详情
        if report.decision_output and hasattr(report.decision_output, 'debate_results'):
            lines.append(f"## 🧠 深度分析详情")
            lines.append(f"")
            
            for code, debate in report.decision_output.debate_results.items():
                lines.append(f"### {debate.company_name} ({code})")
                lines.append(f"")
                lines.append(f"**投资评级**: {debate.investment_rating} | "
                           f"**综合评分**: {debate.final_score:.1f}/10 | "
                           f"**置信度**: {debate.final_confidence:.0%}")
                lines.append(f"")
                
                if debate.consensus:
                    lines.append(f"**综合结论**: {debate.consensus}")
                    lines.append(f"")
                
                if debate.bull_case:
                    lines.append(f"**多方观点**: {debate.bull_case}")
                    lines.append(f"")
                
                if debate.bear_case:
                    lines.append(f"**空方观点**: {debate.bear_case}")
                    lines.append(f"")
                
                if debate.valuation_summary:
                    lines.append(f"**估值分析**: {debate.valuation_summary}")
                    lines.append(f"")
                
                # 各Agent评分
                if debate.agent_analyses:
                    lines.append(f"| 分析师 | 评分 | 置信度 | 核心观点 |")
                    lines.append(f"|--------|------|--------|---------|")
                    for agent_name, analysis in debate.agent_analyses.items():
                        summary = analysis.summary[:80] if analysis.summary else "N/A"
                        lines.append(f"| {agent_name} | {analysis.score:.1f}/10 | "
                                   f"{analysis.confidence:.0%} | {summary} |")
                    lines.append(f"")
        
        # 有效因子
        if report.factor_output and hasattr(report.factor_output, 'effective_factors'):
            lines.append(f"## 🔬 有效因子 (Top 15)")
            lines.append(f"")
            lines.append(f"| 排名 | 因子名称 | 类别 | IC均值 | IR | 多空收益 | 有效性得分 |")
            lines.append(f"|------|---------|------|--------|-----|---------|-----------|")
            
            for i, f in enumerate(report.factor_output.effective_factors[:15], 1):
                lines.append(f"| {i} | {f.name} | {f.category} | {f.ic_mean:+.4f} | "
                           f"{f.ir:+.3f} | {f.long_short_return:+.2%} | {f.effectiveness_score:.2f} |")
            lines.append(f"")
        
        # 风险预警
        if report.risk_alerts:
            lines.append(f"## ⚠️ 风险预警")
            lines.append(f"")
            for alert in report.risk_alerts:
                lines.append(f"- {alert}")
            lines.append(f"")
        
        # 免责声明
        lines.append(f"---")
        lines.append(f"*本报告由 Quant-Investor V6.0 自动生成，仅供参考，不构成投资建议。"
                     f"投资有风险，入市需谨慎。*")
        
        return "\n".join(lines)
    
    def save_report(self, report: PipelineReport, filename: str = None) -> str:
        """保存报告到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"v6_report_{report.market}_{timestamp}.md"
        
        filepath = self.output_dir / filename
        content = self.generate_markdown(report)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(filepath)


# ==================== Master Pipeline V6.0 ====================

class MasterPipelineV6:
    """
    V6.0 统一流水线 (Master Pipeline)
    
    整个技能的唯一入口，串联所有5个分析层。
    
    使用方式：
        pipeline = MasterPipelineV6(market="US")
        report = pipeline.run()
    """
    
    def __init__(self, market: str = "US", stock_pool: List[str] = None,
                  lookback_years: int = 3, llm_preference: List[str] = None,
                  optimization_method: str = 'max_sharpe',
                  total_capital: float = 1000000,
                  top_n_candidates: int = 20, top_n_final: int = 10,
                  max_debate_stocks: int = 5, max_debate_rounds: int = 1,
                  verbose: bool = True, output_dir: str = None):
        """
        初始化V6.0统一流水线
        
        Args:
            market: 市场类型 ("CN" 或 "US")
            stock_pool: 自定义股票池 (可选)
            lookback_years: 历史数据回溯年数
            llm_preference: LLM偏好顺序
            optimization_method: 组合优化方法 ('max_sharpe'/'risk_parity'/'min_variance'/'equal_weight')
            total_capital: 总投资资金
            top_n_candidates: 因子层候选股票数
            top_n_final: 模型层最终排名数
            max_debate_stocks: 决策层深度分析股票数
            max_debate_rounds: 辩论轮数
            verbose: 是否打印详细信息
            output_dir: 报告输出目录
        """
        self.market = market
        self.stock_pool = stock_pool
        self.lookback_years = lookback_years
        self.llm_preference = llm_preference
        self.optimization_method = optimization_method
        self.total_capital = total_capital
        self.top_n_candidates = top_n_candidates
        self.top_n_final = top_n_final
        self.max_debate_stocks = max_debate_stocks
        self.max_debate_rounds = max_debate_rounds
        self.verbose = verbose
        
        # 报告生成器
        self.report_generator = ReportGenerator(output_dir)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"🚀 Quant-Investor V6.0 Master Pipeline")
            print(f"{'='*70}")
            print(f"   市场: {market}")
            print(f"   股票池: {'自定义' if stock_pool else '指数成分股'}")
            print(f"   回溯年数: {lookback_years}")
            print(f"   优化方法: {optimization_method}")
            print(f"   候选股票: {top_n_candidates} → 排名: {top_n_final} → 深度分析: {max_debate_stocks}")
            print(f"{'='*70}")
    
    def run(self) -> PipelineReport:
        """
        运行完整的V6.0分析流水线
        
        Returns:
            PipelineReport: 完整的分析报告
        """
        start_time = time.time()
        report = PipelineReport(
            market=self.market,
            run_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        try:
            # ========== 第1层: 数据层 ==========
            self._print_phase("1/5", "数据层", "获取与清洗数据")
            
            data_layer = UnifiedDataLayer(
                market=self.market, lookback_years=self.lookback_years,
                verbose=self.verbose
            )
            data_bundle = data_layer.fetch_all(stock_pool=self.stock_pool)
            report.data_bundle = data_bundle
            report.data_stats = data_bundle.stats
            
            if data_bundle.panel_data is None or len(data_bundle.panel_data) == 0:
                raise ValueError("数据获取失败: 面板数据为空")
            
            # ========== 第2层: 因子层 ==========
            self._print_phase("2/5", "因子层", "计算与验证因子")
            
            factor_layer = UnifiedFactorLayer(
                verbose=self.verbose, top_n_stocks=self.top_n_candidates
            )
            benchmark_returns = data_layer.get_benchmark_returns(data_bundle)
            factor_output = factor_layer.process(
                data_bundle.panel_data, benchmark_returns
            )
            report.factor_output = factor_output
            report.factor_stats = factor_output.stats
            
            # ========== 第3层: 模型层 ==========
            self._print_phase("3/5", "模型层", "ML建模与信号生成")
            
            model_layer = UnifiedModelLayer(
                verbose=self.verbose, top_n_stocks=self.top_n_final
            )
            model_output = model_layer.predict(
                factor_matrix=factor_output.factor_matrix,
                panel=data_bundle.panel_data,
                candidate_stocks=factor_output.candidate_stocks
            )
            report.model_output = model_output
            report.model_stats = model_output.stats
            
            # ========== 第4层: 决策层 ==========
            self._print_phase("4/5", "决策层", "多Agent深度分析")
            
            # 使用模型层排名，如果模型层没有结果则使用因子层候选
            ranked_stocks = model_output.ranked_stocks or factor_output.candidate_stocks
            
            # 如果有focus_stocks，只对用户关注的股票进行深度分析
            focus_stocks = data_bundle.focus_stocks
            if focus_stocks:
                # 从排名中筛选focus股票，保留排名顺序
                focus_ranked = [s for s in ranked_stocks if s.get('code') in focus_stocks]
                # 如果排名中没有focus股票，直接构建
                if not focus_ranked:
                    focus_ranked = [{'code': c, 'name': c, 'composite_score': 0.5} for c in focus_stocks]
                ranked_stocks = focus_ranked
                if self.verbose:
                    print(f"  🎯 聚焦用户关注股票: {len(ranked_stocks)} 只")
            
            # 构建量化摘要
            quant_summary = self._build_quant_summary(factor_output, model_output)
            
            decision_layer = UnifiedDecisionLayer(
                llm_preference=self.llm_preference,
                verbose=self.verbose,
                max_debate_rounds=self.max_debate_rounds
            )
            decision_output = decision_layer.analyze(
                ranked_stocks=ranked_stocks,
                data_bundle=data_bundle,
                quant_summary=quant_summary,
                max_stocks=self.max_debate_stocks
            )
            report.decision_output = decision_output
            report.decision_stats = decision_output.stats
            
            # ========== 第5层: 风控层 ==========
            self._print_phase("5/5", "风控层", "组合优化与风险评估")
            
            # 使用决策层推荐，如果没有则使用排名股票
            recommendations = decision_output.final_recommendations or ranked_stocks
            
            # 确保风控层只对focus股票构建组合
            if focus_stocks:
                recommendations = [r for r in recommendations if r.get('code') in focus_stocks]
                if not recommendations:
                    recommendations = [{'code': c, 'name': c} for c in focus_stocks]
            
            risk_layer = UnifiedRiskLayer(verbose=self.verbose)
            risk_output = risk_layer.process(
                recommendations=recommendations,
                data_bundle=data_bundle,
                optimization_method=self.optimization_method,
                total_capital=self.total_capital
            )
            report.risk_output = risk_output
            report.risk_stats = risk_output.stats
            
            # ========== 汇总结果 ==========
            report.final_recommendations = decision_output.final_recommendations or ranked_stocks
            report.portfolio_weights = risk_output.portfolio.weights if risk_output.portfolio else {}
            report.risk_alerts = risk_output.risk_alerts
            
        except Exception as e:
            if self.verbose:
                print(f"\n  ❌ 流水线执行出错: {e}")
                import traceback
                traceback.print_exc()
        
        report.duration_seconds = time.time() - start_time
        
        # 生成并保存报告
        if self.verbose:
            self._print_phase("完成", "报告生成", "生成投资分析报告")
        
        report_path = self.report_generator.save_report(report)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"🎉 V6.0 分析完成!")
            print(f"   总耗时: {report.duration_seconds:.1f} 秒")
            print(f"   推荐股票: {len(report.final_recommendations)} 只")
            print(f"   报告路径: {report_path}")
            print(f"{'='*70}")
        
        return report
    
    def _print_phase(self, phase_num: str, phase_name: str, description: str):
        """打印阶段信息"""
        if self.verbose:
            print(f"\n{'━'*70}")
            print(f"  [{phase_num}] {phase_name}: {description}")
            print(f"{'━'*70}")
    
    def _build_quant_summary(self, factor_output: FactorLayerOutput,
                              model_output: ModelLayerOutput) -> str:
        """构建量化分析摘要（传递给决策层）"""
        lines = []
        
        # 因子分析摘要
        lines.append("## 量化因子分析")
        lines.append(f"- 有效因子数: {factor_output.stats.get('effective_factors', 0)}")
        lines.append(f"- 最佳因子: {factor_output.stats.get('top_factor', 'N/A')} "
                    f"(IC={factor_output.stats.get('top_factor_ic', 0):.4f})")
        
        if factor_output.effective_factors:
            lines.append("\n### Top 5 有效因子:")
            for i, f in enumerate(factor_output.effective_factors[:5], 1):
                lines.append(f"  {i}. {f.name}: IC={f.ic_mean:+.4f}, IR={f.ir:+.3f}")
        
        # 模型预测摘要
        lines.append(f"\n## 机器学习模型")
        lines.append(f"- 训练模型数: {model_output.stats.get('models_trained', 0)}")
        lines.append(f"- 最佳模型: {model_output.stats.get('best_model', 'N/A')}")
        
        if model_output.feature_importance:
            lines.append("\n### Top 5 重要特征:")
            for i, (feat, imp) in enumerate(list(model_output.feature_importance.items())[:5], 1):
                lines.append(f"  {i}. {feat}: {imp:.4f}")
        
        return "\n".join(lines)


# ==================== 便捷函数 ====================

def run_analysis(market: str = "US", stock_pool: List[str] = None,
                  lookback_years: int = 3, verbose: bool = True,
                  **kwargs) -> PipelineReport:
    """
    便捷函数：运行V6.0完整分析
    
    Args:
        market: 市场 ("US" 或 "CN")
        stock_pool: 自定义股票池
        lookback_years: 回溯年数
        verbose: 详细输出
        **kwargs: 其他参数传递给MasterPipelineV6
    
    Returns:
        PipelineReport: 分析报告
    
    示例:
        # 分析美股
        report = run_analysis("US")
        
        # 分析指定股票
        report = run_analysis("US", stock_pool=["AAPL", "MSFT", "NVDA"])
        
        # 分析A股
        report = run_analysis("CN", lookback_years=3)
    """
    pipeline = MasterPipelineV6(
        market=market, stock_pool=stock_pool,
        lookback_years=lookback_years, verbose=verbose,
        **kwargs
    )
    return pipeline.run()


if __name__ == "__main__":
    print("=" * 70)
    print("Quant-Investor V6.0 Master Pipeline")
    print("=" * 70)
    
    # 运行分析
    report = run_analysis(
        market="US",
        stock_pool=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
        lookback_years=1,
        verbose=True,
        top_n_candidates=5,
        top_n_final=5,
        max_debate_stocks=3
    )
    
    print(f"\n最终推荐:")
    for rec in report.final_recommendations:
        print(f"  {rec.get('code', 'N/A')}: {rec.get('name', 'N/A')} - {rec.get('investment_rating', 'N/A')}")
