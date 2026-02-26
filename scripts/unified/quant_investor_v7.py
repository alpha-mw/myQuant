#!/usr/bin/env python3
"""
Quant-Investor V7.0 - 新架构整合版本

架构:
1. 数据层 (Data Layer) - 数据获取与清理
2. 因子层 (Factor Layer) - 特征工程与因子检验
3. 模型层 (Model Layer) - ML模型训练与预测
4. 宏观数据层 (Macro Layer) - 第0层风控，市场趋势判断
5. 决策层 (Decision Layer) - LLM深度分析

流程:
原始数据 → 数据层 → 因子层 → 模型层 → 宏观层 → 决策层 → 最终投资建议
"""

import sys
import os
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

# 添加路径
unified_dir = os.path.dirname(os.path.abspath(__file__))
if unified_dir not in sys.path:
    sys.path.insert(0, unified_dir)

# 导入各层
from enhanced_data_layer import EnhancedDataLayer, DataCleaner, FeatureEngineer
from factor_analyzer import FactorAnalyzer
from enhanced_model_layer import EnhancedModelLayer
from macro_terminal_tushare import create_terminal, MacroRiskTerminalBase


# ==================== 配置 ====================

TUSHARE_TOKEN = "33d6ebd3bad7812192d768a191e29ebe653a1839b3f63ec8a0dd7da94172"
TUSHARE_URL = 'http://lianghua.nanyangqiankun.top'


# ==================== 数据结构 ====================

@dataclass
class QuantPipelineResult:
    """量化流水线结果"""
    # 数据层输出
    raw_data: Optional[pd.DataFrame] = None
    cleaned_data: Optional[pd.DataFrame] = None
    
    # 因子层输出
    factor_data: Optional[pd.DataFrame] = None
    factor_analysis: Optional[Dict] = None
    selected_factors: List[str] = field(default_factory=list)
    
    # 模型层输出
    model_predictions: Optional[pd.Series] = None
    model_results: Optional[Dict] = None
    feature_importance: Optional[pd.DataFrame] = None
    
    # 宏观层输出
    macro_report: Optional[Any] = None
    macro_signal: str = ""
    macro_risk_level: str = ""
    
    # 决策层输出
    llm_analysis: str = ""
    final_recommendation: str = ""
    
    # 执行日志
    execution_log: List[str] = field(default_factory=list)


# ==================== 新架构主类 ====================

class QuantInvestorV7:
    """
    Quant-Investor V7.0 - 五层架构
    
    1. 数据层: 获取OHLCV、基本面、宏观数据
    2. 因子层: 计算因子、因子检验、筛选
    3. 模型层: 训练ML模型、生成预测
    4. 宏观层: 市场趋势判断、风险信号
    5. 决策层: LLM综合分析、生成建议
    """
    
    VERSION = "7.0.0-new-arch"
    
    def __init__(
        self,
        market: str = "CN",
        stock_pool: Optional[List[str]] = None,
        lookback_years: float = 1.0,
        enable_macro: bool = True,
        enable_llm: bool = False,
        verbose: bool = True
    ):
        self.market = market.upper()
        self.stock_pool = stock_pool or []
        self.lookback_years = lookback_years
        self.enable_macro = enable_macro
        self.enable_llm = enable_llm
        self.verbose = verbose
        
        # 初始化各层
        self._init_layers()
        
        # 结果存储
        self.result = QuantPipelineResult()
    
    def _log(self, msg: str, layer: str = ""):
        """记录日志"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        prefix = f"[{layer}]" if layer else "[Main]"
        entry = f"[{timestamp}] {prefix} {msg}"
        self.result.execution_log.append(entry)
        if self.verbose:
            print(entry)
    
    def _init_layers(self):
        """初始化各层组件"""
        # 1. 数据层
        self.data_layer = EnhancedDataLayer(
            market=self.market,
            verbose=self.verbose
        )
        
        # 2. 因子层
        self.factor_analyzer = FactorAnalyzer(
            verbose=self.verbose
        )
        
        # 3. 模型层
        self.model_layer = EnhancedModelLayer(
            verbose=self.verbose
        )
        
        # 4. 宏观层
        self.macro_layer: Optional[MacroRiskTerminalBase] = None
        if self.enable_macro:
            try:
                self.macro_layer = create_terminal(market=self.market)
                self._log("宏观层初始化成功", "Macro")
            except Exception as e:
                self._log(f"宏观层初始化失败: {e}", "Macro")
        
        # 5. 决策层 (LLM) - 预留接口
        self.llm_enabled = self.enable_llm
    
    # ==================== 第1层: 数据层 ====================
    
    def _layer1_data(self) -> bool:
        """
        数据层: 获取并清理数据
        
        输入: 股票池、时间范围
        输出: 清理后的数据
        """
        self._log("=" * 60, "Layer1")
        self._log("【第1层】数据层 - 数据获取与清理", "Layer1")
        self._log("=" * 60, "Layer1")
        
        if not self.stock_pool:
            self._log("股票池为空", "Layer1")
            return False
        
        all_data = []
        
        for symbol in self.stock_pool:
            try:
                self._log(f"获取数据: {symbol}", "Layer1")
                
                # 计算日期范围
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365 * self.lookback_years)
                
                # 获取数据 (包含特征工程)
                df = self.data_layer.fetch_and_process(
                    symbol=symbol,
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d'),
                    label_periods=5
                )
                
                if not df.empty:
                    all_data.append(df)
                    self._log(f"{symbol}: {len(df)} 行数据", "Layer1")
                else:
                    self._log(f"{symbol}: 无数据", "Layer1")
                    
            except Exception as e:
                self._log(f"{symbol} 数据获取失败: {e}", "Layer1")
        
        if not all_data:
            self._log("没有获取到任何数据", "Layer1")
            return False
        
        # 合并数据
        combined_df = pd.concat(all_data, ignore_index=True)
        self.result.raw_data = combined_df
        
        self._log(f"数据层完成: 共 {len(combined_df)} 行", "Layer1")
        return True
    
    # ==================== 第2层: 因子层 ====================
    
    def _layer2_factor(self) -> bool:
        """
        因子层: 因子计算、检验、筛选
        
        输入: 清理后的数据
        输出: 筛选后的有效因子
        """
        self._log("=" * 60, "Layer2")
        self._log("【第2层】因子层 - 因子计算与检验", "Layer2")
        self._log("=" * 60, "Layer2")
        
        if self.result.raw_data is None or self.result.raw_data.empty:
            self._log("无输入数据", "Layer2")
            return False
        
        df = self.result.raw_data
        
        # 识别因子列
        factor_cols = [c for c in df.columns if c.startswith((
            'return_', 'volatility_', 'rsi_', 'macd_', 'ma_bias_',
            'momentum_', 'atr_', 'volume_ratio_', 'amihud_',
            'pe', 'pb', 'roe', 'gross_margin'
        ))]
        
        if not factor_cols:
            self._log("未找到因子列", "Layer2")
            return False
        
        self._log(f"发现 {len(factor_cols)} 个因子", "Layer2")
        
        # 因子检验
        try:
            analysis_results = self.factor_analyzer.comprehensive_factor_test(
                df,
                factor_cols=factor_cols,
                return_col='label_return'
            )
            
            self.result.factor_analysis = analysis_results
            
            # 获取综合评分最高的因子
            comprehensive = analysis_results.get('comprehensive_score', pd.DataFrame())
            if not comprehensive.empty:
                # 选择前10个有效因子
                selected = comprehensive[comprehensive.get('有效性', '') == '有效'].head(10)
                self.result.selected_factors = selected['因子'].tolist()
                
                self._log(f"筛选出 {len(self.result.selected_factors)} 个有效因子", "Layer2")
                for i, (_, row) in enumerate(selected.head(5).iterrows(), 1):
                    self._log(f"  {i}. {row['因子']}: 综合得分={row.get('综合得分', 0):.4f}", "Layer2")
            
        except Exception as e:
            self._log(f"因子检验失败: {e}", "Layer2")
            # 使用所有因子作为备选
            self.result.selected_factors = factor_cols[:10]
        
        self.result.factor_data = df
        self._log("因子层完成", "Layer2")
        return True
    
    # ==================== 第3层: 模型层 ====================
    
    def _layer3_model(self) -> bool:
        """
        模型层: 训练ML模型，生成预测
        
        输入: 筛选后的因子
        输出: 模型预测结果
        """
        self._log("=" * 60, "Layer3")
        self._log("【第3层】模型层 - ML模型训练与预测", "Layer3")
        self._log("=" * 60, "Layer3")
        
        if self.result.factor_data is None or not self.result.selected_factors:
            self._log("无输入数据", "Layer3")
            return False
        
        df = self.result.factor_data
        feature_cols = self.result.selected_factors
        
        # 确保有足够的数据
        if len(df) < 100:
            self._log("数据量不足，跳过模型训练", "Layer3")
            return False
        
        try:
            # 训练所有模型
            model_results = self.model_layer.train_all_models(
                df,
                feature_cols=feature_cols,
                label_col='label_return',
                task='regression',
                use_lstm=False
            )
            
            self.result.model_results = model_results
            
            # 获取集成预测
            ensemble_pred = self.model_layer.ensemble_predict(
                list(model_results.values())
            )
            
            self.result.model_predictions = ensemble_pred
            
            # 特征重要性
            importance = self.model_layer.get_feature_importance_ranking()
            self.result.feature_importance = importance
            
            # 输出模型性能
            self._log("模型性能:", "Layer3")
            for name, result in model_results.items():
                if result.model is not None:
                    mse = result.val_metrics.get('mse', 0)
                    self._log(f"  {name}: Val MSE={mse:.6f}", "Layer3")
            
            if not ensemble_pred.empty:
                self._log(f"集成预测: {len(ensemble_pred)} 个预测值", "Layer3")
            
        except Exception as e:
            self._log(f"模型训练失败: {e}", "Layer3")
            return False
        
        self._log("模型层完成", "Layer3")
        return True
    
    # ==================== 第4层: 宏观层 ====================
    
    def _layer4_macro(self) -> bool:
        """
        宏观层: 市场趋势判断，第0层风控
        
        输入: 市场代码
        输出: 宏观风险信号
        """
        self._log("=" * 60, "Layer4")
        self._log("【第4层】宏观层 - 市场趋势判断 (第0层风控)", "Layer4")
        self._log("=" * 60, "Layer4")
        
        if not self.macro_layer:
            self._log("宏观层未启用", "Layer4")
            return False
        
        try:
            # 生成宏观风控报告
            macro_report = self.macro_layer.generate_risk_report()
            self.result.macro_report = macro_report
            self.result.macro_signal = macro_report.overall_signal
            self.result.macro_risk_level = macro_report.overall_risk_level
            
            self._log(f"宏观信号: {macro_report.overall_signal} {macro_report.overall_risk_level}", "Layer4")
            
            # 输出各模块信号
            for module in macro_report.modules:
                self._log(f"  {module.module_name}: {module.overall_signal}", "Layer4")
            
        except Exception as e:
            self._log(f"宏观分析失败: {e}", "Layer4")
            return False
        
        self._log("宏观层完成", "Layer4")
        return True
    
    # ==================== 第5层: 决策层 ====================
    
    def _layer5_decision(self) -> bool:
        """
        决策层: LLM深度分析，生成最终建议
        
        输入: 模型预测 + 宏观信号
        输出: 最终投资建议
        """
        self._log("=" * 60, "Layer5")
        self._log("【第5层】决策层 - 生成投资建议", "Layer5")
        self._log("=" * 60, "Layer5")
        
        # 整合各层信息生成建议
        recommendations = []
        
        # 1. 基于模型预测
        if self.result.model_predictions is not None:
            avg_pred = self.result.model_predictions.mean()
            if avg_pred > 0.02:
                recommendations.append("模型预测乐观，建议增配")
            elif avg_pred < -0.02:
                recommendations.append("模型预测悲观，建议减配")
            else:
                recommendations.append("模型预测中性，维持配置")
        
        # 2. 基于宏观信号
        if self.result.macro_signal:
            signal_map = {
                "🔴": "宏观高风险，防御为主",
                "🟡": "宏观中风险，控制仓位",
                "🟢": "宏观低风险，积极布局",
                "🔵": "宏观极低风险，逆向布局"
            }
            recommendations.append(signal_map.get(self.result.macro_signal, ""))
        
        # 3. 基于因子分析
        if self.result.selected_factors:
            recommendations.append(f"重点关注因子: {', '.join(self.result.selected_factors[:3])}")
        
        # 生成最终建议
        final_recommendation = " | ".join(filter(None, recommendations))
        self.result.final_recommendation = final_recommendation
        
        self._log(f"最终建议: {final_recommendation}", "Layer5")
        self._log("决策层完成", "Layer5")
        return True
    
    # ==================== 主流程 ====================
    
    def run(self) -> QuantPipelineResult:
        """
        执行完整五层流程
        
        数据层 → 因子层 → 模型层 → 宏观层 → 决策层
        """
        self._log("=" * 80)
        self._log(f"Quant-Investor V7.0 开始执行")
        self._log(f"版本: {self.VERSION}")
        self._log(f"市场: {self.market}")
        self._log(f"股票池: {self.stock_pool}")
        self._log("=" * 80)
        
        # 执行五层流程
        success = True
        
        # Layer 1: 数据层
        if not self._layer1_data():
            success = False
        
        # Layer 2: 因子层
        if success and not self._layer2_factor():
            success = False
        
        # Layer 3: 模型层
        if success and not self._layer3_model():
            success = False
        
        # Layer 4: 宏观层
        if self.enable_macro:
            self._layer4_macro()
        
        # Layer 5: 决策层
        self._layer5_decision()
        
        self._log("=" * 80)
        self._log("流程执行完成")
        self._log("=" * 80)
        
        return self.result
    
    def generate_report(self) -> str:
        """生成完整报告"""
        lines = []
        
        lines.append("# Quant-Investor V7.0 投资分析报告")
        lines.append(f"**版本**: {self.VERSION}")
        lines.append(f"**市场**: {self.market}")
        lines.append(f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # 最终建议
        lines.append("## 🎯 最终投资建议")
        lines.append("")
        lines.append(f"**{self.result.final_recommendation}**")
        lines.append("")
        
        # 各层输出摘要
        lines.append("## 📊 分析摘要")
        lines.append("")
        
        # 数据层
        if self.result.raw_data is not None:
            lines.append(f"**数据层**: {len(self.result.raw_data)} 行数据")
        
        # 因子层
        if self.result.selected_factors:
            lines.append(f"**因子层**: {len(self.result.selected_factors)} 个有效因子")
            lines.append(f"  - 主要因子: {', '.join(self.result.selected_factors[:5])}")
        
        # 模型层
        if self.result.model_predictions is not None:
            pred_mean = self.result.model_predictions.mean()
            lines.append(f"**模型层**: 平均预测收益 {pred_mean*100:.2f}%")
        
        # 宏观层
        if self.result.macro_signal:
            lines.append(f"**宏观层**: {self.result.macro_signal} {self.result.macro_risk_level}")
        
        lines.append("")
        
        # 执行日志
        lines.append("## 📝 执行日志")
        lines.append("")
        lines.append("```")
        for log in self.result.execution_log[-20:]:  # 最后20条
            lines.append(log)
        lines.append("```")
        lines.append("")
        
        return "\n".join(lines)


# ==================== 便捷函数 ====================

def analyze(
    market: str = "CN",
    stocks: Optional[List[str]] = None,
    lookback_years: float = 1.0,
    verbose: bool = True
) -> QuantPipelineResult:
    """
    便捷分析函数
    
    示例:
        result = analyze(
            market="CN",
            stocks=["000001.SZ", "600000.SH"],
            lookback_years=1.0
        )
        print(result.final_recommendation)
    """
    pipeline = QuantInvestorV7(
        market=market,
        stock_pool=stocks,
        lookback_years=lookback_years,
        enable_macro=True,
        verbose=verbose
    )
    return pipeline.run()


if __name__ == '__main__':
    print("=" * 80)
    print("Quant-Investor V7.0 - 新架构五层模型")
    print("=" * 80)
    
    # 运行示例
    result = analyze(
        market="CN",
        stocks=["000001.SZ", "600000.SH"],
        lookback_years=0.5,
        verbose=True
    )
    
    print("\n" + "=" * 80)
    print("最终报告")
    print("=" * 80)
    print(result.final_recommendation)
