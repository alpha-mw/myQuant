"""
当前稳定主线入口。

`QuantInvestor` 明确绑定到当前稳定主线，而不是 newest/latest 试验线。
当前 stable = `QuantInvestorV9`。
"""

from __future__ import annotations

from quant_investor.pipeline.quant_investor_v9 import QuantInvestorV9, V9PipelineResult

QuantInvestorCurrent = QuantInvestorV9
CurrentPipelineResult = V9PipelineResult
QuantInvestor = QuantInvestorCurrent

__all__ = [
    "QuantInvestor",
    "QuantInvestorCurrent",
    "CurrentPipelineResult",
]
