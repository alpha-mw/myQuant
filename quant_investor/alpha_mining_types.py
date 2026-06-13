"""Data contracts for the alpha mining framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class FactorProfile:
    """一个已验证 Alpha 因子的完整档案"""

    name: str
    category: str
    formula_desc: str
    ic_mean: float
    ic_std: float
    ir: float
    ic_positive_rate: float
    decay_halflife: int
    annual_turnover: float
    long_short_return: float
    max_drawdown: float
    correlation_with_existing: float
    capacity_score: float
    origin: str = "systematic"


@dataclass
class MiningResult:
    """Alpha 挖掘的完整结果"""

    systematic_factors: list[FactorProfile] = field(default_factory=list)
    genetic_factors: list[FactorProfile] = field(default_factory=list)
    llm_factors: list[FactorProfile] = field(default_factory=list)
    selected_factors: list[FactorProfile] = field(default_factory=list)
    factor_correlation_matrix: Optional[pd.DataFrame] = None
    mining_report: str = ""


__all__ = [
    "FactorProfile",
    "MiningResult",
]
