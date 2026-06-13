from __future__ import annotations

import pandas as pd

from quant_investor import alpha_mining
from quant_investor import alpha_mining_types


def test_alpha_mining_types_are_split_and_reexported() -> None:
    assert alpha_mining.FactorProfile is alpha_mining_types.FactorProfile
    assert alpha_mining.MiningResult is alpha_mining_types.MiningResult

    result = alpha_mining.MiningResult(factor_correlation_matrix=pd.DataFrame())
    assert isinstance(result.factor_correlation_matrix, pd.DataFrame)
