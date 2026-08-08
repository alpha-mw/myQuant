from __future__ import annotations

import pandas as pd
import pytest

from quant_investor.factors.aquant_expression_v5 import (
    ALLOWED_NAMES,
    AquantExpressionInputsV5,
    evaluate_aquant_expression_v5,
)


def _inputs():
    dates = pd.to_datetime(["2026-08-06", "2026-08-07"])
    columns = ["000001.SZ", "600000.SH"]
    matrix = pd.DataFrame([[1.0, 2.0], [2.0, 1.0]], index=dates, columns=columns)
    matrices = {name: matrix.copy() for name in ALLOWED_NAMES}
    matrices["pb"] = pd.DataFrame([[1.0, 2.0], [4.0, 2.0]], index=dates, columns=columns)
    matrices["total_mv"] = pd.DataFrame([[10.0, 20.0], [40.0, 20.0]], index=dates, columns=columns)
    return AquantExpressionInputsV5(matrices=matrices, diagnostics={})


def test_v5_evaluates_daily_basic_value_and_size_without_changing_v4():
    book = evaluate_aquant_expression_v5("cs_rank(1.0 / pb)", _inputs())
    size = evaluate_aquant_expression_v5("-cs_rank(total_mv)", _inputs())
    assert book.loc[pd.Timestamp("2026-08-07"), "600000.SH"] == 1.0
    assert size.loc[pd.Timestamp("2026-08-07"), "600000.SH"] == -0.5


def test_v5_rejects_unknown_names_and_unsafe_ast():
    with pytest.raises(ValueError, match="unsupported v5 expression name"):
        evaluate_aquant_expression_v5("cs_rank(secret_field)", _inputs())
    with pytest.raises(ValueError, match="unsupported v5 expression syntax"):
        evaluate_aquant_expression_v5("pb.__class__", _inputs())
