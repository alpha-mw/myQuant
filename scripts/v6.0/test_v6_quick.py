#!/usr/bin/env python3
"""
V6.0 快速测试脚本 - 验证数据扩充和因子/模型层
跳过LLM决策层以节省时间
"""
import sys
sys.path.insert(0, '.')

from data_layer.unified_data_layer import UnifiedDataLayer
from factor_layer.unified_factor_layer import UnifiedFactorLayer
from model_layer.unified_model_layer import UnifiedModelLayer
from risk_layer.unified_risk_layer import UnifiedRiskLayer

print("=" * 70)
print("V6.0 快速测试: 数据扩充 + 因子层 + 模型层 + 风控层")
print("=" * 70)

# 1. 数据层 - 测试自动扩充
print("\n[1/4] 数据层: 测试样本扩充...")
data_layer = UnifiedDataLayer(market="US", lookback_years=1, verbose=True)
data_bundle = data_layer.fetch_all(stock_pool=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"])

print(f"\n  focus_stocks: {data_bundle.focus_stocks}")
print(f"  total stocks in universe: {len(data_bundle.stock_universe)}")
print(f"  panel shape: {data_bundle.panel_data.shape if data_bundle.panel_data is not None else 'None'}")
print(f"  unique stocks in panel: {data_bundle.panel_data['stock_code'].nunique() if data_bundle.panel_data is not None else 0}")

# 2. 因子层
print("\n[2/4] 因子层: 测试因子验证...")
factor_layer = UnifiedFactorLayer(verbose=True, top_n_stocks=20)
benchmark_returns = data_layer.get_benchmark_returns(data_bundle)
factor_output = factor_layer.process(data_bundle.panel_data, benchmark_returns)

print(f"\n  有效因子: {len(factor_output.effective_factors)}")
print(f"  候选股票: {len(factor_output.candidate_stocks)}")
if factor_output.factor_matrix is not None:
    print(f"  因子矩阵: {factor_output.factor_matrix.shape}")

# 3. 模型层
print("\n[3/4] 模型层: 测试ML训练...")
model_layer = UnifiedModelLayer(verbose=True, top_n_stocks=20)
model_output = model_layer.predict(
    factor_matrix=factor_output.factor_matrix,
    panel=data_bundle.panel_data,
    candidate_stocks=factor_output.candidate_stocks
)

print(f"\n  训练模型: {model_output.stats.get('models_trained', 0)}")
print(f"  排名股票: {len(model_output.ranked_stocks)}")

# 4. 风控层 (使用focus_stocks)
print("\n[4/4] 风控层: 测试组合优化...")
focus_stocks = data_bundle.focus_stocks
if focus_stocks:
    focus_ranked = [s for s in model_output.ranked_stocks if s.get('code') in focus_stocks]
    if not focus_ranked:
        focus_ranked = [{'code': c, 'name': c, 'composite_score': 0.5} for c in focus_stocks]
else:
    focus_ranked = model_output.ranked_stocks[:10]

risk_layer = UnifiedRiskLayer(verbose=True)
risk_output = risk_layer.process(
    recommendations=focus_ranked,
    data_bundle=data_bundle,
    optimization_method='max_sharpe'
)

print(f"\n  组合权重: {len(risk_output.portfolio.weights) if risk_output.portfolio else 0}")
print(f"  风险预警: {len(risk_output.risk_alerts)}")

print("\n" + "=" * 70)
print("✅ V6.0 快速测试完成!")
print("=" * 70)
