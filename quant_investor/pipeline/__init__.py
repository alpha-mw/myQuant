"""Public pipeline entrypoints."""

from quant_investor.pipeline.mainline import QuantInvestor
from quant_investor.pipeline.result_types import QuantInvestorPipelineResult

__all__ = ["QuantInvestor", "QuantInvestorPipelineResult"]
