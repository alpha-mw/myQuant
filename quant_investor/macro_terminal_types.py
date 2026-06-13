"""Protocol dataclasses for macro risk terminal reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DataAcquisitionStep:
    """数据获取步骤记录"""

    timestamp: str
    data_source: str
    data_type: str
    attempt_method: str
    params: dict[str, Any]
    result_status: str
    result_summary: str
    error_message: str = ""
    fallback_plan: str = ""


@dataclass
class AnalysisStep:
    """分析步骤记录"""

    timestamp: str
    step_name: str
    input_data: str
    analysis_method: str
    reasoning_process: str
    conclusion: str
    confidence: str = ""


@dataclass
class IndicatorResult:
    """单个指标的分析结果"""

    name: str
    value: float = 0.0
    unit: str = ""
    status: str = ""
    signal: str = "🟡"
    data_source: str = ""
    data_date: str = ""
    acquisition_steps: list[DataAcquisitionStep] = field(default_factory=list)
    historical_ref: str = ""
    analysis_steps: list[AnalysisStep] = field(default_factory=list)
    analysis_detail: str = ""
    threshold_rules: str = ""


@dataclass
class ModuleResult:
    """单个模块的分析结果"""

    module_name: str
    module_name_en: str
    indicators: list[IndicatorResult] = field(default_factory=list)
    overall_signal: str = "🟡"
    module_analysis_log: list[AnalysisStep] = field(default_factory=list)


@dataclass
class RiskTerminalReport:
    """宏观风控终端完整报告"""

    timestamp: str = ""
    version: str = "V6.4"
    market: str = ""
    market_name: str = ""
    modules: list[ModuleResult] = field(default_factory=list)
    overall_signal: str = "🟡"
    overall_risk_level: str = ""
    recommendation: str = ""
    execution_log: list[str] = field(default_factory=list)


__all__ = [
    "AnalysisStep",
    "DataAcquisitionStep",
    "IndicatorResult",
    "ModuleResult",
    "RiskTerminalReport",
]
