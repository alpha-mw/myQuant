"""
公开编排入口。
"""

from quant_investor.pipeline.current import CurrentPipelineResult, QuantInvestor, QuantInvestorCurrent
from quant_investor.pipeline.quant_investor_v8 import QuantInvestorV8, V8PipelineResult
from quant_investor.pipeline.quant_investor_v9 import QuantInvestorV9, V9PipelineResult
from quant_investor.pipeline.quant_investor_v10 import QuantInvestorV10, V10PipelineResult

QuantInvestorLatest = QuantInvestorV10

__all__ = [
    "QuantInvestor",
    "QuantInvestorCurrent",
    "QuantInvestorV8",
    "QuantInvestorV9",
    "QuantInvestorV10",
    "QuantInvestorLatest",
    "CurrentPipelineResult",
    "V8PipelineResult",
    "V9PipelineResult",
    "V10PipelineResult",
]
