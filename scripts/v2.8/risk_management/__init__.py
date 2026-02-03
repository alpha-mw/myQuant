"""
Quant-Investor V2.8 - 风险管理模块

核心组件:
- RiskMetrics: 风险度量指标计算器
- FactorRiskAnalyzer: 因子风险分解与归因分析
- RiskManager: 综合风险管理器

V2.8 核心特性:
1. 全面的风险度量指标：Sharpe、Sortino、Calmar、VaR、CVaR等
2. 因子风险分解：基于Barra模型思想，分解系统性和特异性风险
3. 风险归因分析：分析各因子和各资产对总风险的贡献
4. 情景分析与压力测试：模拟不同市场情景下的风险表现
5. 综合风险报告：一站式生成专业的风险分析报告
"""

from .risk_metrics import (
    RiskMetrics,
    VaRMethod,
    DrawdownInfo,
)

from .factor_risk import (
    FactorRiskAnalyzer,
    FactorExposure,
    RiskDecomposition,
)

from .risk_manager import (
    RiskManager,
    RiskLevel,
    RiskAlert,
)

__all__ = [
    # 风险度量
    'RiskMetrics',
    'VaRMethod',
    'DrawdownInfo',
    
    # 因子风险
    'FactorRiskAnalyzer',
    'FactorExposure',
    'RiskDecomposition',
    
    # 综合管理
    'RiskManager',
    'RiskLevel',
    'RiskAlert',
]

__version__ = '2.8.0'
