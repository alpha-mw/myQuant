"""
风险管理模块完整测试脚本

测试内容:
1. 风险度量指标计算
2. 因子风险分解
3. 综合风险评估
4. 策略对比
5. 报告生成
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from risk_metrics import RiskMetrics, VaRMethod
from factor_risk import FactorRiskAnalyzer
from risk_manager import RiskManager, RiskLevel


def create_test_data(n_days: int = 504, seed: int = 42):
    """创建测试数据"""
    np.random.seed(seed)
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')
    
    # 模拟因子收益率
    factor_returns = pd.DataFrame({
        'market': np.random.normal(0.0003, 0.01, n_days),
        'size': np.random.normal(0.0001, 0.005, n_days),
        'value': np.random.normal(0.0001, 0.006, n_days),
        'momentum': np.random.normal(0.0002, 0.008, n_days),
    }, index=dates)
    
    # 模拟策略收益率（与因子相关）
    strategy_returns = pd.Series(
        0.0001 +  # Alpha
        1.2 * factor_returns['market'].values +
        0.3 * factor_returns['size'].values +
        -0.2 * factor_returns['value'].values +
        0.5 * factor_returns['momentum'].values +
        np.random.normal(0, 0.005, n_days),  # 特异性收益
        index=dates
    )
    
    # 模拟基准收益率
    benchmark_returns = pd.Series(
        factor_returns['market'].values + np.random.normal(0, 0.002, n_days),
        index=dates
    )
    
    return strategy_returns, benchmark_returns, factor_returns


def test_risk_metrics():
    """测试风险度量指标"""
    print("\n" + "=" * 60)
    print("测试1: 风险度量指标计算")
    print("=" * 60)
    
    strategy_returns, benchmark_returns, _ = create_test_data()
    rm = RiskMetrics(risk_free_rate=0.02)
    
    # 测试各项指标
    tests_passed = 0
    tests_total = 0
    
    # Sharpe比率
    tests_total += 1
    sharpe = rm.sharpe_ratio(strategy_returns)
    if isinstance(sharpe, float) and not np.isnan(sharpe):
        print(f"  ✓ Sharpe Ratio: {sharpe:.4f}")
        tests_passed += 1
    else:
        print(f"  ✗ Sharpe Ratio 计算失败")
    
    # Sortino比率
    tests_total += 1
    sortino = rm.sortino_ratio(strategy_returns)
    if isinstance(sortino, float) and not np.isnan(sortino):
        print(f"  ✓ Sortino Ratio: {sortino:.4f}")
        tests_passed += 1
    else:
        print(f"  ✗ Sortino Ratio 计算失败")
    
    # 最大回撤
    tests_total += 1
    prices = (1 + strategy_returns).cumprod()
    max_dd = rm.maximum_drawdown(prices)
    if isinstance(max_dd, float) and max_dd <= 0:
        print(f"  ✓ Maximum Drawdown: {max_dd*100:.2f}%")
        tests_passed += 1
    else:
        print(f"  ✗ Maximum Drawdown 计算失败")
    
    # VaR
    tests_total += 1
    var_95 = rm.value_at_risk(strategy_returns, 0.95, VaRMethod.HISTORICAL)
    if isinstance(var_95, float) and var_95 <= 0:
        print(f"  ✓ VaR (95%): {var_95*100:.2f}%")
        tests_passed += 1
    else:
        print(f"  ✗ VaR 计算失败")
    
    # CVaR
    tests_total += 1
    cvar_95 = rm.conditional_var(strategy_returns, 0.95)
    if isinstance(cvar_95, float) and cvar_95 <= var_95:
        print(f"  ✓ CVaR (95%): {cvar_95*100:.2f}%")
        tests_passed += 1
    else:
        print(f"  ✗ CVaR 计算失败")
    
    # Beta
    tests_total += 1
    beta = rm.calculate_beta(strategy_returns, benchmark_returns)
    if isinstance(beta, float) and not np.isnan(beta):
        print(f"  ✓ Beta: {beta:.4f}")
        tests_passed += 1
    else:
        print(f"  ✗ Beta 计算失败")
    
    # Alpha
    tests_total += 1
    alpha = rm.calculate_alpha(strategy_returns, benchmark_returns)
    if isinstance(alpha, float) and not np.isnan(alpha):
        print(f"  ✓ Alpha: {alpha*100:.2f}%")
        tests_passed += 1
    else:
        print(f"  ✗ Alpha 计算失败")
    
    print(f"\n  测试结果: {tests_passed}/{tests_total} 通过")
    return tests_passed == tests_total


def test_factor_risk():
    """测试因子风险分解"""
    print("\n" + "=" * 60)
    print("测试2: 因子风险分解")
    print("=" * 60)
    
    strategy_returns, _, factor_returns = create_test_data()
    analyzer = FactorRiskAnalyzer()
    
    tests_passed = 0
    tests_total = 0
    
    # 因子暴露计算
    tests_total += 1
    exposures = analyzer.calculate_factor_exposures(strategy_returns, factor_returns)
    if len(exposures) == len(factor_returns.columns):
        print(f"  ✓ 因子暴露计算成功，共{len(exposures)}个因子")
        for exp in exposures:
            sig = "***" if exp.is_significant else ""
            print(f"    - {exp.factor_name}: Beta={exp.exposure:.4f} {sig}")
        tests_passed += 1
    else:
        print(f"  ✗ 因子暴露计算失败")
    
    # 风险分解
    tests_total += 1
    decomposition = analyzer.decompose_risk(strategy_returns, factor_returns)
    if decomposition.total_risk > 0 and 0 <= decomposition.r_squared <= 1:
        print(f"  ✓ 风险分解成功")
        print(f"    - 总风险: {decomposition.total_risk*100:.2f}%")
        print(f"    - 系统性风险: {decomposition.systematic_risk*100:.2f}%")
        print(f"    - 特异性风险: {decomposition.idiosyncratic_risk*100:.2f}%")
        print(f"    - R²: {decomposition.r_squared:.4f}")
        tests_passed += 1
    else:
        print(f"  ✗ 风险分解失败")
    
    # 压力测试
    tests_total += 1
    stress_results = analyzer.stress_test(strategy_returns, factor_returns)
    if len(stress_results) > 0:
        print(f"  ✓ 压力测试成功")
        tests_passed += 1
    else:
        print(f"  ✗ 压力测试失败")
    
    print(f"\n  测试结果: {tests_passed}/{tests_total} 通过")
    return tests_passed == tests_total


def test_risk_manager():
    """测试综合风险管理器"""
    print("\n" + "=" * 60)
    print("测试3: 综合风险管理器")
    print("=" * 60)
    
    strategy_returns, benchmark_returns, factor_returns = create_test_data()
    rm = RiskManager()
    
    tests_passed = 0
    tests_total = 0
    
    # 综合评估
    tests_total += 1
    evaluation = rm.evaluate_strategy(
        strategy_returns,
        benchmark_returns,
        factor_returns,
        strategy_name="测试策略"
    )
    if 'risk_level' in evaluation and 'basic_metrics' in evaluation:
        print(f"  ✓ 综合评估成功")
        print(f"    - 风险等级: {evaluation['risk_level'].value}")
        print(f"    - 预警数量: {len(evaluation['alerts'])}")
        tests_passed += 1
    else:
        print(f"  ✗ 综合评估失败")
    
    # 策略对比
    tests_total += 1
    strategies = {
        '策略A': strategy_returns,
        '策略B': pd.Series(np.random.normal(0.0003, 0.012, len(strategy_returns)), index=strategy_returns.index),
    }
    comparison = rm.compare_strategies(strategies, benchmark_returns)
    if len(comparison) == 2:
        print(f"  ✓ 策略对比成功")
        tests_passed += 1
    else:
        print(f"  ✗ 策略对比失败")
    
    # 报告生成
    tests_total += 1
    report = rm.generate_comprehensive_report(
        strategy_returns,
        benchmark_returns,
        factor_returns,
        strategy_name="测试策略"
    )
    if len(report) > 1000 and "风险等级" in report:
        print(f"  ✓ 报告生成成功 (长度: {len(report)} 字符)")
        tests_passed += 1
    else:
        print(f"  ✗ 报告生成失败")
    
    print(f"\n  测试结果: {tests_passed}/{tests_total} 通过")
    return tests_passed == tests_total


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 60)
    print("测试4: 边界情况处理")
    print("=" * 60)
    
    rm = RiskMetrics()
    
    tests_passed = 0
    tests_total = 0
    
    # 空数据
    tests_total += 1
    try:
        empty_returns = pd.Series([], dtype=float)
        sharpe = rm.sharpe_ratio(empty_returns)
        print(f"  ✓ 空数据处理: Sharpe={sharpe}")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ 空数据处理失败: {e}")
    
    # 常数收益率
    tests_total += 1
    try:
        const_returns = pd.Series([0.001] * 100)
        sharpe = rm.sharpe_ratio(const_returns)
        if sharpe == 0 or np.isinf(sharpe):
            print(f"  ✓ 常数收益率处理: Sharpe={sharpe}")
            tests_passed += 1
        else:
            print(f"  ✗ 常数收益率处理异常: Sharpe={sharpe}")
    except Exception as e:
        print(f"  ✗ 常数收益率处理失败: {e}")
    
    # 极端收益率
    tests_total += 1
    try:
        extreme_returns = pd.Series(np.random.normal(0, 0.5, 100))  # 50%日波动率
        var = rm.value_at_risk(extreme_returns, 0.95)
        if isinstance(var, float) and not np.isnan(var):
            print(f"  ✓ 极端收益率处理: VaR={var*100:.2f}%")
            tests_passed += 1
        else:
            print(f"  ✗ 极端收益率处理异常")
    except Exception as e:
        print(f"  ✗ 极端收益率处理失败: {e}")
    
    print(f"\n  测试结果: {tests_passed}/{tests_total} 通过")
    return tests_passed == tests_total


def main():
    """运行所有测试"""
    print("=" * 60)
    print("Quant-Investor V2.8 风险管理模块测试")
    print("=" * 60)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # 运行测试
    results.append(("风险度量指标", test_risk_metrics()))
    results.append(("因子风险分解", test_factor_risk()))
    results.append(("综合风险管理器", test_risk_manager()))
    results.append(("边界情况处理", test_edge_cases()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过！风险管理模块运行正常。")
    else:
        print("部分测试失败，请检查相关模块。")
    print("=" * 60)
    
    return all_passed


if __name__ == '__main__':
    main()
