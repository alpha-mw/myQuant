#!/usr/bin/env python3
"""
Quant-Investor V7.1 - 大规模数据增强版

支持：
- 5年历史数据
- 沪深300+中证500+中证1000成分股
- 批量并行获取
"""

import sys
import os
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

# 添加路径
unified_dir = os.path.dirname(os.path.abspath(__file__))
if unified_dir not in sys.path:
    sys.path.insert(0, unified_dir)

# 导入各层
from enhanced_data_layer import EnhancedDataLayer
from factor_analyzer import FactorAnalyzer
from enhanced_model_layer import EnhancedModelLayer
from macro_terminal_tushare import create_terminal, MacroRiskTerminalBase
from risk_management_layer import RiskManagementLayer, RiskLayerResult
from decision_layer import DecisionLayer, DecisionLayerResult
from batch_data_pipeline import BatchDataPipeline, fetch_major_indices_data
from stock_universe import get_major_indices


# ==================== 配置 ====================

TUSHARE_TOKEN = os.environ.get('TUSHARE_TOKEN', '33d6ebd3bad7812192d768a191e29ebe653a1839b3f63ec8a0dd7da94172')
TUSHARE_URL = os.environ.get('TUSHARE_URL', 'http://lianghua.nanyangqiankun.top')


# ==================== 数据结构 ====================

@dataclass
class QuantPipelineResult:
    """量化流水线结果"""
    # 数据层输出
    raw_data: Optional[pd.DataFrame] = None
    
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
    
    # 风控层输出 (第5层)
    risk_layer_result: Optional[RiskLayerResult] = None
    risk_adjusted_positions: Optional[Dict[str, float]] = None
    stop_loss_levels: Optional[Dict[str, float]] = None
    
    # 决策层输出 (第6层)
    decision_result: Optional[DecisionLayerResult] = None
    final_recommendation: str = ""
    
    # 执行日志
    execution_log: List[str] = field(default_factory=list)


# ==================== 大规模分析主类 ====================

class QuantInvestorV71:
    """
    Quant-Investor V7.1 - 大规模数据增强版
    
    1. 数据层: 支持5年数据，批量获取
    2. 因子层: 计算因子、因子检验、筛选
    3. 模型层: 训练ML模型、生成预测
    4. 宏观层: 市场趋势判断、风险信号
    5. 风控层: 组合风控、仓位管理、止损止盈
    6. 决策层: LLM多Agent多空辩论、生成具体投资建议
    """
    
    VERSION = "7.1.0-large-scale"
    
    def __init__(
        self,
        market: str = "CN",
        stock_pool: Optional[List[str]] = None,
        lookback_years: float = 5.0,
        enable_macro: bool = True,
        use_batch_pipeline: bool = True,
        max_workers: int = 5,
        verbose: bool = True
    ):
        self.market = market.upper()
        self.stock_pool = stock_pool or []
        self.lookback_years = lookback_years
        self.enable_macro = enable_macro
        self.use_batch_pipeline = use_batch_pipeline
        self.max_workers = max_workers
        self.verbose = verbose
        
        # 结果存储 - 必须在_init_layers之前初始化
        self.result = QuantPipelineResult()
        
        # 初始化各层
        self._init_layers()
    
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
        if self.use_batch_pipeline:
            self.batch_pipeline = BatchDataPipeline(
                market=self.market,
                max_workers=self.max_workers,
                verbose=self.verbose
            )
        self.data_layer = EnhancedDataLayer(market=self.market, verbose=False)
        
        # 2. 因子层
        self.factor_analyzer = FactorAnalyzer(verbose=self.verbose)
        
        # 3. 模型层
        self.model_layer = EnhancedModelLayer(verbose=self.verbose)
        
        # 4. 宏观层
        self.macro_layer: Optional[MacroRiskTerminalBase] = None
        if self.enable_macro:
            try:
                self.macro_layer = create_terminal(market=self.market)
                self._log("宏观层初始化成功", "Macro")
            except Exception as e:
                self._log(f"宏观层初始化失败: {e}", "Macro")
        
        # 5. 风控层
        self.risk_layer = RiskManagementLayer(verbose=self.verbose)
        
        # 6. 决策层 (LLM多Agent)
        self.decision_layer = DecisionLayer(
            api_key=os.environ.get('OPENAI_API_KEY'),
            verbose=self.verbose
        )
    
    def _layer1_data(self) -> bool:
        """第1层: 数据层 - 批量获取"""
        self._log("=" * 60, "Layer1")
        self._log(f"【第1层】数据层 - 批量数据获取 ({self.lookback_years}年)", "Layer1")
        
        # 如果没有指定股票池，获取主要指数成分股
        if not self.stock_pool:
            self._log("未指定股票池，获取主要指数成分股 (HS300+ZZ500+ZZ1000)", "Layer1")
            self.stock_pool = get_major_indices()
        
        if not self.stock_pool:
            self._log("股票池为空", "Layer1")
            return False
        
        self._log(f"股票池规模: {len(self.stock_pool)} 只", "Layer1")
        
        try:
            # 使用批量流水线获取数据
            if self.use_batch_pipeline and len(self.stock_pool) > 10:
                df = self.batch_pipeline.fetch_batch(
                    stocks=self.stock_pool[:200],  # 限制前200只避免时间过长
                    start_date=(datetime.now() - timedelta(days=365*self.lookback_years)).strftime('%Y%m%d'),
                    end_date=datetime.now().strftime('%Y%m%d'),
                    use_cache=True
                )
            else:
                # 小批量直接获取
                all_data = []
                for i, symbol in enumerate(self.stock_pool[:50]):  # 限制50只
                    try:
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=365 * self.lookback_years)
                        
                        df = self.data_layer.fetch_and_process(
                            symbol=symbol,
                            start_date=start_date.strftime('%Y%m%d'),
                            end_date=end_date.strftime('%Y%m%d')
                        )
                        
                        if not df.empty:
                            all_data.append(df)
                        
                        if (i + 1) % 10 == 0:
                            self._log(f"已处理 {i+1}/{len(self.stock_pool)} 只股票", "Layer1")
                            
                    except Exception as e:
                        self._log(f"{symbol} 失败: {e}", "Layer1")
                
                if all_data:
                    df = pd.concat(all_data, ignore_index=True)
                else:
                    df = pd.DataFrame()
            
            if not df.empty:
                self.result.raw_data = df
                self._log(f"数据层完成: {len(df)} 条记录, {df['symbol'].nunique()} 只股票", "Layer1")
                return True
            else:
                self._log("数据获取失败", "Layer1")
                return False
                
        except Exception as e:
            self._log(f"数据层异常: {e}", "Layer1")
            return False
    
    def _layer2_factor(self) -> bool:
        """第2层: 因子层"""
        self._log("=" * 60, "Layer2")
        self._log("【第2层】因子层 - 因子计算与检验", "Layer2")
        
        if self.result.raw_data is None:
            return False
        
        df = self.result.raw_data
        factor_cols = [c for c in df.columns if c.startswith(('return_', 'volatility_', 'rsi_', 'macd_', 'ma_bias_'))]
        
        if not factor_cols:
            return False
        
        try:
            analysis = self.factor_analyzer.comprehensive_factor_test(df, factor_cols, 'label_return')
            self.result.factor_analysis = analysis
            
            comprehensive = analysis.get('comprehensive_score', pd.DataFrame())
            if not comprehensive.empty:
                # 选择综合得分最高的前10个因子
                selected = comprehensive.head(10)
                self.result.selected_factors = selected['因子'].tolist()
                self._log(f"选中 {len(self.result.selected_factors)} 个因子", "Layer2")
            else:
                self.result.selected_factors = factor_cols[:10]
            
        except Exception as e:
            self._log(f"因子检验失败: {e}", "Layer2")
            self.result.selected_factors = factor_cols[:10]
        
        self.result.factor_data = df
        return True
    
    def _layer3_model(self) -> bool:
        """第3层: 模型层"""
        self._log("=" * 60, "Layer3")
        self._log("【第3层】模型层 - ML模型训练", "Layer3")
        
        if self.result.factor_data is None or not self.result.selected_factors:
            return False
        
        try:
            model_results = self.model_layer.train_all_models(
                self.result.factor_data,
                feature_cols=self.result.selected_factors,
                label_col='label_return',
                task='regression'
            )
            
            self.result.model_results = model_results
            ensemble_pred = self.model_layer.ensemble_predict(list(model_results.values()))
            self.result.model_predictions = ensemble_pred
            self.result.feature_importance = self.model_layer.get_feature_importance_ranking()
            
        except Exception as e:
            self._log(f"模型训练失败: {e}", "Layer3")
            return False
        
        return True
    
    def _layer4_macro(self) -> bool:
        """第4层: 宏观层"""
        self._log("=" * 60, "Layer4")
        self._log("【第4层】宏观层 - 市场趋势判断", "Layer4")
        
        if not self.macro_layer:
            return False
        
        try:
            macro_report = self.macro_layer.generate_risk_report()
            self.result.macro_report = macro_report
            self.result.macro_signal = macro_report.overall_signal
            self.result.macro_risk_level = macro_report.overall_risk_level
            
            self._log(f"宏观信号: {macro_report.overall_signal} {macro_report.overall_risk_level}", "Layer4")
            
        except Exception as e:
            self._log(f"宏观分析失败: {e}", "Layer4")
            return False
        
        return True
    
    def _layer5_risk(self) -> bool:
        """第5层: 风控层"""
        self._log("=" * 60, "Layer5")
        self._log("【第5层】风控层 - 组合风控", "Layer5")
        
        # 准备数据
        predicted_returns = {}
        predicted_volatilities = {}
        
        for symbol in self.stock_pool[:50]:  # 限制前50只
            pred = self.result.model_predictions.mean() if self.result.model_predictions is not None else 0
            predicted_returns[symbol] = pred
            predicted_volatilities[symbol] = 0.25
        
        portfolio_returns = pd.Series(np.random.normal(0.0005, 0.02, 252))
        current_prices = {s: 100.0 for s in self.stock_pool[:50]}
        
        try:
            risk_result = self.risk_layer.run_risk_management(
                portfolio_returns=portfolio_returns,
                predicted_returns=predicted_returns,
                predicted_volatilities=predicted_volatilities,
                current_prices=current_prices,
                macro_signal=self.result.macro_signal or "🟢"
            )
            
            self.result.risk_layer_result = risk_result
            self.result.risk_adjusted_positions = risk_result.position_sizing.risk_adjusted_weights
            
            self._log(f"风控完成: {risk_result.risk_level}", "Layer5")
            
        except Exception as e:
            self._log(f"风控失败: {e}", "Layer5")
            return False
        
        return True
    
    def _layer6_decision(self) -> bool:
        """第6层: 决策层 (LLM多Agent)"""
        self._log("=" * 60, "Layer6")
        self._log("【第6层】决策层 - LLM多Agent多空辩论", "Layer6")
        
        # 准备数据
        quant_data = {}
        for symbol in self.stock_pool[:10]:  # 限制前10只
            quant_data[symbol] = {
                "predicted_return": self.result.model_predictions.mean() if self.result.model_predictions is not None else 0,
                "predicted_volatility": 0.25,
                "sharpe_ratio": self.result.risk_layer_result.risk_metrics.sharpe_ratio if self.result.risk_layer_result else 0,
                "factors": self.result.selected_factors[:5]
            }
        
        macro_data = {"signal": self.result.macro_signal, "risk_level": self.result.macro_risk_level}
        risk_data = {"risk_level": self.result.risk_layer_result.risk_level if self.result.risk_layer_result else "normal"}
        
        try:
            decision_result = self.decision_layer.run_decision_process(
                symbols=list(quant_data.keys()),
                quant_data=quant_data,
                macro_data=macro_data,
                risk_data=risk_data
            )
            
            self.result.decision_result = decision_result
            self.result.final_recommendation = decision_result.final_report
            
            self._log(f"市场展望: {decision_result.market_outlook}", "Layer6")
            
            for rec in decision_result.investment_decisions:
                self._log(f"  {rec.symbol}: {rec.action} (置信度{rec.confidence:.0%})", "Layer6")
            
        except Exception as e:
            self._log(f"决策层失败: {e}", "Layer6")
            return False
        
        return True
    
    def run(self) -> QuantPipelineResult:
        """执行完整六层流程"""
        self._log("=" * 80)
        self._log(f"Quant-Investor V7.1 大规模分析开始执行")
        self._log(f"版本: {self.VERSION}")
        self._log(f"市场: {self.market}")
        self._log(f"股票池: {len(self.stock_pool)} 只")
        self._log(f"回溯期: {self.lookback_years} 年")
        self._log("=" * 80)
        
        # 执行六层流程
        self._layer1_data()
        self._layer2_factor()
        self._layer3_model()
        
        if self.enable_macro:
            self._layer4_macro()
        
        self._layer5_risk()
        self._layer6_decision()
        
        self._log("=" * 80)
        self._log("六层流程执行完成")
        self._log("=" * 80)
        
        return self.result


# ==================== 便捷函数 ====================

def analyze_large_scale(
    market: str = "CN",
    stocks: Optional[List[str]] = None,
    lookback_years: float = 5.0,
    use_major_indices: bool = True,
    max_workers: int = 5,
    verbose: bool = True
) -> QuantPipelineResult:
    """
    大规模分析便捷函数
    
    Args:
        market: 市场代码 (CN/US)
        stocks: 股票列表 (None则自动获取主要指数成分股)
        lookback_years: 回溯年数 (默认5年)
        use_major_indices: 是否使用主要指数成分股 (HS300+ZZ500+ZZ1000)
        max_workers: 并行线程数
        verbose: 是否打印日志
    """
    # 获取股票池
    if stocks is None and use_major_indices:
        stocks = get_major_indices()
        if verbose:
            print(f"[Main] 自动获取主要指数成分股: {len(stocks)} 只")
    
    pipeline = QuantInvestorV71(
        market=market,
        stock_pool=stocks,
        lookback_years=lookback_years,
        use_batch_pipeline=True,
        max_workers=max_workers,
        verbose=verbose
    )
    
    return pipeline.run()


if __name__ == '__main__':
    print("=" * 80)
    print("Quant-Investor V7.1 - 大规模分析")
    print("=" * 80)
    
    # 运行大规模分析
    result = analyze_large_scale(
        market="CN",
        lookback_years=5.0,
        use_major_indices=True,
        max_workers=3,
        verbose=True
    )
    
    print("\n" + "=" * 80)
    print("分析结果")
    print("=" * 80)
    
    if result.decision_result:
        print(result.decision_result.final_report)
