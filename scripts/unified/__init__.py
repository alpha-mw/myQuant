"""
Quant-Investor Unified Version (统一版本)
整合 V2.7 ~ V6.0 所有核心功能的大一统框架

架构: 分层解耦设计
- 数据层 (Data Layer): V2.7持久化 + V3.0全景数据 + V6.0统一接口
- 因子层 (Factor Layer): V3.2因子挖掘 + V3.4因子库 + V3.5深度合成 + V5.0因子动物园 + V6.0统一接口
- 模型层 (Model Layer): V5.0 ML模型 + V6.0统一接口
- 决策层 (Decision Layer): V2.9多Agent辩论 + V3.6多LLM适配 + V6.0统一接口
- 风控层 (Risk Layer): V2.8风险管理 + V4.1基准对比 + V5.0高级风控 + V6.0统一接口
- 智能层 (Intelligence Layer): V3.1动态监控 + V3.3因子分析

Author: Unified Framework
Date: 2026-02-25
"""

__version__ = "7.0.0-unified"
__all__ = [
    'UnifiedDataLayer',
    'UnifiedFactorLayer', 
    'UnifiedModelLayer',
    'UnifiedDecisionLayer',
    'UnifiedRiskLayer',
    'MasterPipelineUnified',
    'IntelligenceMonitor',
]

# 延迟导入，避免循环依赖
def _get_layers():
    from .data_layer import UnifiedDataLayer
    from .factor_layer import UnifiedFactorLayer
    from .model_layer import UnifiedModelLayer
    from .decision_layer import UnifiedDecisionLayer
    from .risk_layer import UnifiedRiskLayer
    from .intelligence_layer import IntelligenceMonitor
    from .pipeline import MasterPipelineUnified
    return {
        'UnifiedDataLayer': UnifiedDataLayer,
        'UnifiedFactorLayer': UnifiedFactorLayer,
        'UnifiedModelLayer': UnifiedModelLayer,
        'UnifiedDecisionLayer': UnifiedDecisionLayer,
        'UnifiedRiskLayer': UnifiedRiskLayer,
        'IntelligenceMonitor': IntelligenceMonitor,
        'MasterPipelineUnified': MasterPipelineUnified,
    }

# 版本信息
VERSION_INFO = {
    'version': '7.0.0-unified',
    'codename': 'Omnibus',
    'release_date': '2026-02-25',
    'integrated_versions': ['v2.7', 'v2.8', 'v2.9', 'v3.0', 'v3.1', 'v3.2', 'v3.3', 'v3.4', 'v3.5', 'v3.6', 'v4.0', 'v4.1', 'v5.0', 'v6.0'],
    'features': {
        'data': ['persistent_storage', 'multi_source', 'incremental_update', 'alternative_data'],
        'factor': ['500+_factors', 'genetic_mining', 'deep_synthesis', 'auto_validation'],
        'model': ['xgboost', 'lightgbm', 'randomforest', 'ensemble', 'transformer_optional'],
        'decision': ['multi_llm', 'multi_agent_debate', 'qualitative_analysis'],
        'risk': ['var_cvar', 'factor_decomposition', 'stress_test', 'benchmark_comparison'],
        'intelligence': ['factor_decay_monitor', 'dynamic_optimization'],
    }
}

def get_version_info():
    """获取版本信息"""
    return VERSION_INFO.copy()

def list_integrated_features():
    """列出所有整合的功能"""
    info = get_version_info()
    print(f"Quant-Investor Unified v{info['version']}")
    print(f"代号: {info['codename']}")
    print(f"发布日期: {info['release_date']}")
    print(f"\n整合版本: {', '.join(info['integrated_versions'])}")
    print("\n核心功能:")
    for layer, features in info['features'].items():
        print(f"  [{layer.upper()}] {', '.join(features)}")
