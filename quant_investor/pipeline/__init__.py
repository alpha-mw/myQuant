"""
公开编排入口。
"""

from quant_investor.pipeline.mainline import QuantInvestor, QuantInvestorPipelineResult
from quant_investor.pipeline.quant_investor_v8 import QuantInvestorV8, V8PipelineResult
from quant_investor.pipeline.quant_investor_v9 import QuantInvestorV9, V9PipelineResult
from quant_investor.pipeline.quant_investor_v10 import QuantInvestorV10, V10PipelineResult

QuantInvestorLatest = QuantInvestorV10

__all__ = [
    "QuantInvestor",
    "QuantInvestorPipelineResult",
    "QuantInvestorV8",
    "QuantInvestorV9",
    "QuantInvestorV10",
    "QuantInvestorLatest",
    "V8PipelineResult",
    "V9PipelineResult",
    "V10PipelineResult",
]
