#!/usr/bin/env python3
"""
V8 主线公共入口。

官方导出仅保留五路并行研究主线对象与标准契约对象。
"""

from pathlib import Path
import sys

_PACKAGE_DIR = Path(__file__).resolve().parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

from .branch_contracts import (
    BranchResult,
    CalibratedBranchSignal,
    PortfolioStrategy,
    ResearchPipelineResult,
    UnifiedDataBundle,
)
from .quant_investor_v8 import QuantInvestorV8, V8PipelineResult

__all__ = [
    "QuantInvestorV8",
    "V8PipelineResult",
    "UnifiedDataBundle",
    "BranchResult",
    "CalibratedBranchSignal",
    "PortfolioStrategy",
    "ResearchPipelineResult",
]
