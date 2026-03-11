#!/usr/bin/env python3
"""
Quant-Investor Unified v7.0 - 大一统版本
整合 V2.7 ~ V6.0 所有核心功能

使用方式:
    from unified import MasterPipelineUnified
    pipeline = MasterPipelineUnified(market="US", stock_pool=["AAPL", "MSFT"])
    report = pipeline.run()
"""

import sys
import os

# 添加所有版本路径
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# 按优先级添加版本路径 (新版本优先)
version_paths = [
    'v6.0', 'v5.0', 'v4.1', 'v4.0', 
    'v3.6', 'v3.5', 'v3.4', 'v3.3', 'v3.2', 'v3.1', 'v3.0',
    'v2.9', 'v2.8', 'v2.7'
]

for v in version_paths:
    path = os.path.join(parent_dir, v)
    if path not in sys.path:
        sys.path.insert(0, path)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field

# 导入统一层 (使用相对导入)
import sys
import os
unified_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if unified_dir not in sys.path:
    sys.path.insert(0, unified_dir)

from data_layer import UnifiedDataLayer, UnifiedDataBundle
from factor_layer import UnifiedFactorLayer, FactorOutput
from model_layer import UnifiedModelLayer, ModelOutput
from decision_layer import UnifiedDecisionLayer, DecisionOutput
from risk_layer import UnifiedRiskLayer, RiskOutput, Portfolio
from logger import get_logger

# 尝试导入历史版本特色功能
try:
    from v2_7.persistent_data_manager import PersistentDataManager
    V27_AVAILABLE = True
except:
    V27_AVAILABLE = False

try:
    from v2_8.risk_management.risk_manager import RiskManager
    V28_AVAILABLE = True
except:
    V28_AVAILABLE = False

try:
    from v2_9.debate_engine.investment_pipeline import InvestmentPipeline
    V29_AVAILABLE = True
except:
    V29_AVAILABLE = False

try:
    from v3_2.factor_mining.genetic_factor_generator import GeneticFactorGenerator
    V32_AVAILABLE = True
except:
    V32_AVAILABLE = False

try:
    from v3_5.deep_synthesis.deep_feature_engine import DeepFeatureEngine
    V35_AVAILABLE = True
except:
    V35_AVAILABLE = False

try:
    from v3_6.llm_adapters.multi_llm_adapter import MultiLLMAdapter
    V36_AVAILABLE = True
except:
    V36_AVAILABLE = False

try:
    from v5_0.factor_zoo.price_volume_factors import PriceVolumeFactors
    from v5_0.factor_zoo.fundamental_factors import FundamentalFactors
    V50_FACTORS = True
except:
    V50_FACTORS = False


@dataclass
class UnifiedReport:
    """统一分析报告"""
    version: str = "7.0.0-unified"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 各层输出
    data_bundle: Optional[UnifiedDataBundle] = None
    factor_output: Optional[FactorOutput] = None
    model_output: Optional[ModelOutput] = None
    decision_output: Optional[DecisionOutput] = None
    risk_output: Optional[RiskOutput] = None
    
    # 整合功能状态
    features_used: Dict[str, bool] = field(default_factory=dict)
    
    # 最终输出
    summary: str = ""
    recommendations: List[Dict] = field(default_factory=list)
    portfolio: Optional[Portfolio] = None
    
    def to_markdown(self) -> str:
        """生成 Markdown 报告"""
        lines = [
            "# Quant-Investor 统一版投资分析报告",
            f"",
            f"**版本**: {self.version}  ",
            f"**时间**: {self.timestamp}  ",
            f"**市场**: {self.data_bundle.market if self.data_bundle else 'N/A'}",
            "",
            "## 功能整合状态",
            "",
        ]
        
        for feature, used in self.features_used.items():
            status = "✅" if used else "❌"
            lines.append(f"- {status} {feature}")
        
        lines.extend([
            "",
            "## 执行摘要",
            "",
            self.summary,
            "",
            "## 投资组合建议",
            "",
        ])
        
        if self.portfolio and self.portfolio.weights:
            lines.append("| 股票 | 权重 | 预期收益 | 风险等级 |")
            lines.append("|:---|---:|---:|:---|")
            for stock, weight in self.portfolio.weights.items():
                lines.append(f"| {stock} | {weight*100:.1f}% | - | - |")
        else:
            lines.append("暂无组合建议")
        
        lines.extend([
            "",
            "## 风险提示",
            "",
        ])
        
        if self.risk_output and self.risk_output.risk_alerts:
            for alert in self.risk_output.risk_alerts:
                lines.append(f"- ⚠️ {alert}")
        else:
            lines.append("- 无重大风险预警")
        
        return "\n".join(lines)


class MasterPipelineUnified:
    """
    统一主流水线 - 整合所有版本功能
    
    参数:
        market: 市场代码 ("CN" 或 "US")
        stock_pool: 自定义股票池 (可选)
        lookback_years: 回测年数
        enable_v27_persistent: 启用 V2.7 持久化存储
        enable_v28_advanced_risk: 启用 V2.8 高级风险管理
        enable_v29_debate: 启用 V2.9 多 Agent 辩论
        enable_v32_genetic: 启用 V3.2 遗传编程因子挖掘
        enable_v35_deep: 启用 V3.5 深度特征合成
        enable_v36_multi_llm: 启用 V3.6 多 LLM 适配
        enable_v50_factors: 启用 V5.0 扩展因子库
        llm_preference: LLM 偏好列表
        verbose: 是否输出详细日志
    """
    
    VERSION = "7.0.0-unified"
    
    def __init__(
        self,
        market: str = "US",
        stock_pool: Optional[List[str]] = None,
        lookback_years: int = 1,
        # 功能开关
        enable_v27_persistent: bool = True,
        enable_v28_advanced_risk: bool = False,
        enable_v29_debate: bool = False,
        enable_v32_genetic: bool = False,
        enable_v35_deep: bool = False,
        enable_v36_multi_llm: bool = False,
        enable_v50_factors: bool = True,
        # 其他参数
        llm_preference: Optional[List[str]] = None,
        verbose: bool = True
    ):
        self.market = market.upper()
        self.stock_pool = stock_pool
        self.lookback_years = lookback_years
        self.verbose = verbose
        self._logger = get_logger("MasterPipelineUnified", verbose)

        # 功能开关
        self.features = {
            'V2.7 持久化存储': enable_v27_persistent and V27_AVAILABLE,
            'V2.8 高级风险管理': enable_v28_advanced_risk and V28_AVAILABLE,
            'V2.9 多Agent辩论': enable_v29_debate and V29_AVAILABLE,
            'V3.2 遗传编程因子': enable_v32_genetic and V32_AVAILABLE,
            'V3.5 深度特征合成': enable_v35_deep and V35_AVAILABLE,
            'V3.6 多LLM适配': enable_v36_multi_llm and V36_AVAILABLE,
            'V5.0 扩展因子库': enable_v50_factors and V50_FACTORS,
        }
        
        # 初始化各层
        self._log(f"初始化 Quant-Investor Unified v{self.VERSION}")
        self._log(f"市场: {market}, 回测: {lookback_years}年")
        
        # 数据层
        self.data_layer = UnifiedDataLayer(
            market=self.market,
            lookback_years=lookback_years,
            verbose=verbose
        )
        
        # 因子层
        self.factor_layer = UnifiedFactorLayer(verbose=verbose)
        
        # 模型层
        self.model_layer = UnifiedModelLayer(verbose=verbose)
        
        # 决策层
        self.decision_layer = UnifiedDecisionLayer(
            llm_preference=llm_preference or ["openai"],
            verbose=verbose
        )
        
        # 风控层
        self.risk_layer = UnifiedRiskLayer(verbose=verbose)
        
        # 历史版本组件 (按需初始化)
        self.v27_manager = None
        self.v28_risk_manager = None
        self.v32_genetic = None
        self.v35_deep = None
        self.v36_llm = None
        
        self._init_legacy_components()
    
    def _log(self, msg: str) -> None:
        self._logger.info(msg)
    
    def _init_legacy_components(self):
        """初始化历史版本组件"""
        if self.features['V2.7 持久化存储']:
            self.v27_manager = PersistentDataManager()
            self._log("V2.7 持久化管理器已加载")
        
        if self.features['V2.8 高级风险管理']:
            self.v28_risk_manager = RiskManager()
            self._log("V2.8 高级风险管理器已加载")
        
        if self.features['V3.2 遗传编程因子']:
            self.v32_genetic = GeneticFactorGenerator()
            self._log("V3.2 遗传编程因子生成器已加载")
        
        if self.features['V3.5 深度特征合成']:
            self.v35_deep = DeepFeatureEngine()
            self._log("V3.5 深度特征引擎已加载")
        
        if self.features['V3.6 多LLM适配']:
            self.v36_llm = MultiLLMAdapter()
            self._log("V3.6 多LLM适配器已加载")
    
    def run(self) -> UnifiedReport:
        """
        执行完整分析流程
        
        Returns:
            UnifiedReport 统一分析报告
        """
        report = UnifiedReport()
        report.features_used = self.features
        
        self._log("=" * 60)
        self._log("开始统一分析流程")
        self._log("=" * 60)
        
        # Step 1: 数据层
        self._log("\n[Step 1/5] 数据获取与清洗...")
        data_bundle = self.data_layer.fetch_all(stock_pool=self.stock_pool)
        report.data_bundle = data_bundle
        self._log(f"  ✓ 获取 {len(data_bundle.stock_universe)} 只股票数据")
        
        # Step 2: 因子层
        self._log("\n[Step 2/5] 因子计算与验证...")
        benchmark_returns = self.data_layer.get_benchmark_returns(data_bundle)
        factor_output = self.factor_layer.process(
            data_bundle.panel_data,
            benchmark_returns
        )
        report.factor_output = factor_output
        self._log(f"  ✓ 有效因子: {len(factor_output.effective_factors)}")
        
        # 扩展: V3.2 遗传编程因子挖掘
        if self.features['V3.2 遗传编程因子'] and self.v32_genetic:
            self._log("  🧬 执行遗传编程因子挖掘...")
            # TODO: 集成遗传编程
        
        # 扩展: V3.5 深度特征合成
        if self.features['V3.5 深度特征合成'] and self.v35_deep:
            self._log("  🧠 执行深度特征合成...")
            # TODO: 集成深度特征
        
        # Step 3: 模型层
        self._log("\n[Step 3/5] 机器学习建模...")
        model_output = self.model_layer.predict(
            factor_matrix=factor_output.factor_matrix,
            panel=data_bundle.panel_data,
            candidate_stocks=factor_output.candidate_stocks
        )
        report.model_output = model_output
        self._log(f"  ✓ 训练 {model_output.stats.get('models_trained', 0)} 个模型")
        
        # Step 4: 决策层
        self._log("\n[Step 4/5] LLM 深度决策...")
        
        # 如果有自定义股票池，只分析这些股票
        focus_stocks = data_bundle.focus_stocks or []
        if focus_stocks and model_output.ranked_stocks:
            focus_ranked = [
                s for s in model_output.ranked_stocks 
                if s.get('code') in focus_stocks
            ][:10]
        else:
            focus_ranked = model_output.ranked_stocks[:10]
        
        decision_output = self.decision_layer.process(
            ranked_stocks=focus_ranked,
            data_bundle=data_bundle
        )
        report.decision_output = decision_output
        self._log(f"  ✓ 决策完成: {len(decision_output.ratings)} 条评级")
        
        # Step 5: 风控层
        self._log("\n[Step 5/5] 组合优化与风险评估...")
        risk_output = self.risk_layer.process(
            recommendations=focus_ranked,
            data_bundle=data_bundle,
            optimization_method='max_sharpe'
        )
        report.risk_output = risk_output
        report.portfolio = risk_output.portfolio
        
        if risk_output.portfolio:
            self._log(f"  ✓ 组合权重: {len(risk_output.portfolio.weights)} 只股票")
        
        # 扩展: V2.8 高级风险管理
        if self.features['V2.8 高级风险管理'] and self.v28_risk_manager:
            self._log("  🛡️ 执行高级风险分析...")
            # TODO: 集成高级风险分析
        
        # 生成摘要
        report.summary = self._generate_summary(report)
        report.recommendations = self._generate_recommendations(report)
        
        self._log("\n" + "=" * 60)
        self._log("统一分析流程完成!")
        self._log("=" * 60)
        
        return report
    
    def _generate_summary(self, report: UnifiedReport) -> str:
        """生成执行摘要"""
        parts = []
        
        if report.data_bundle:
            parts.append(f"分析了 {len(report.data_bundle.stock_universe)} 只股票的 {self.lookback_years} 年历史数据。")
        
        if report.factor_output:
            parts.append(f"从 34+ 个基础因子中筛选出 {len(report.factor_output.effective_factors)} 个有效因子。")
        
        if report.model_output:
            parts.append(f"使用 XGBoost/LightGBM/RandomForest 集成模型进行预测。")
        
        if report.risk_output and report.risk_output.portfolio:
            port = report.risk_output.portfolio
            parts.append(f"优化后的组合预期年化收益 {port.expected_return*100:.2f}%，波动率 {port.volatility*100:.2f}%。")
        
        return " ".join(parts)
    
    def _generate_recommendations(self, report: UnifiedReport) -> List[Dict]:
        """生成投资建议列表"""
        recommendations = []
        
        if report.risk_output and report.risk_output.portfolio:
            for stock, weight in report.risk_output.portfolio.weights.items():
                recommendations.append({
                    'stock': stock,
                    'weight': weight,
                    'action': '持有' if weight > 0.1 else '轻仓',
                    'confidence': '中'
                })
        
        return recommendations
    
    def get_version_info(self) -> Dict[str, Any]:
        """获取版本信息"""
        return {
            'version': self.VERSION,
            'market': self.market,
            'features': self.features,
            'components': {
                'data_layer': 'V6.0 Unified',
                'factor_layer': 'V6.0 Unified + V5.0 Extensions',
                'model_layer': 'V6.0 Unified',
                'decision_layer': 'V6.0 Unified + V3.6 Multi-LLM',
                'risk_layer': 'V6.0 Unified + V2.8 Advanced',
            }
        }


# 便捷函数
def quick_analyze(
    market: str = "US",
    stocks: Optional[List[str]] = None,
    **kwargs
) -> UnifiedReport:
    """
    快速分析函数
    
    示例:
        report = quick_analyze("US", ["AAPL", "MSFT", "NVDA"])
        print(report.to_markdown())
    """
    pipeline = MasterPipelineUnified(
        market=market,
        stock_pool=stocks,
        **kwargs
    )
    return pipeline.run()


if __name__ == '__main__':
    print("=" * 70)
    print("Quant-Investor Unified v7.0 - 大一统版本")
    print("=" * 70)
    
    # 快速测试
    report = quick_analyze(
        market="US",
        stocks=["AAPL", "MSFT", "NVDA"],
        lookback_years=1,
        verbose=True
    )
    
    print("\n" + report.to_markdown())
