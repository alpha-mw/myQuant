"""A signal-day weight must not depend on data from after the signal day.

Candidate selection historically dropped a symbol whose *forward* execution
return was missing. That decision is made on the signal date using information
from after it, and because the eligible count sets the quantile boundaries, it
moves the other symbols' weights too — a symbol with no future silently
re-weights its peers' past.

These tests pin both halves down: the legacy behaviour is preserved by default
and documented as lookahead-bearing, and the corrected semantics are invariant
under truncation of later prices while still refusing to invent a return.
"""
from __future__ import annotations

import pytest

from quant_investor.factors.backtest import (
    build_quantile_weight_matrix,
    compute_daily_backtest_records,
)
from quant_investor.factors.matrix import (
    FIELD_CLOSE,
    FactorMatrix,
    MatrixDataBundle,
    MatrixDataContract,
)
from quant_investor.factors.schema import FactorBacktestConfig

SYMBOLS = ["AAA.SZ", "BBB.SZ", "CCC.SZ"]
DATES = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]


def _bundle(closes: list[list[float | None]]) -> MatrixDataBundle:
    contract = MatrixDataContract(
        contract_id="signal-time-semantics",
        universe="test",
        symbols=list(SYMBOLS),
        dates=list(DATES),
        required_fields=[FIELD_CLOSE],
    )
    return MatrixDataBundle(
        bundle_id="signal-time-semantics-bundle",
        contract=contract,
        fields={FIELD_CLOSE: closes},
    )


def _factor() -> FactorMatrix:
    return FactorMatrix(
        matrix_id="signal-time-semantics-factor",
        expression="f",
        symbols=list(SYMBOLS),
        dates=list(DATES),
        values=[
            [3.0, 3.0, 3.0, 3.0, 3.0],
            [2.0, 2.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ],
    )


def _closes(truncate_after: int | None = None) -> list[list[float | None]]:
    rows: list[list[float | None]] = []
    for base in (10.0, 20.0, 30.0):
        row: list[float | None] = []
        for index in range(len(DATES)):
            if truncate_after is not None and index > truncate_after:
                row.append(None)
            else:
                row.append(base + index)
        rows.append(row)
    return rows


def _config(*, requires_future: bool) -> FactorBacktestConfig:
    return FactorBacktestConfig(
        config_id="signal-time-semantics-config",
        execution_price="close",
        quantile_count=2,
        long_quantile=2,
        long_short=False,
        selection_requires_future_return=requires_future,
    )


def test_default_config_preserves_the_legacy_selection_rule() -> None:
    """Changing the default would silently re-weight every existing result."""
    assert FactorBacktestConfig(config_id="c").selection_requires_future_return is True


@pytest.mark.parametrize("cut_index", [1, 2, 3])
def test_signal_day_weights_are_invariant_to_later_prices(cut_index: int) -> None:
    factor = _factor()
    config = _config(requires_future=False)

    baseline = build_quantile_weight_matrix(factor, _bundle(_closes()), config)
    truncated = build_quantile_weight_matrix(
        factor, _bundle(_closes(truncate_after=cut_index)), config
    )

    for row in range(len(SYMBOLS)):
        for column in range(cut_index + 1):
            assert (baseline.long_weights[row][column] or 0.0) == pytest.approx(
                truncated.long_weights[row][column] or 0.0
            ), f"weight moved at {SYMBOLS[row]} {DATES[column]} after truncating later prices"


def test_legacy_rule_lets_later_prices_move_an_earlier_weight() -> None:
    """The defect, stated as a test so it cannot be reintroduced unnoticed."""
    factor = _factor()
    config = _config(requires_future=True)

    baseline = build_quantile_weight_matrix(factor, _bundle(_closes()), config)
    truncated = build_quantile_weight_matrix(
        factor, _bundle(_closes(truncate_after=2)), config
    )

    moved = [
        (SYMBOLS[row], DATES[column])
        for row in range(len(SYMBOLS))
        for column in range(3)
        if (baseline.long_weights[row][column] or 0.0)
        != (truncated.long_weights[row][column] or 0.0)
    ]
    assert moved, "expected the legacy rule to move at least one signal-day weight"


def test_unevaluable_return_is_null_and_its_day_is_kept() -> None:
    """A return we cannot evaluate must not become a zero return, or vanish."""
    factor = _factor()
    config = _config(requires_future=False)
    bundle = _bundle(_closes(truncate_after=2))
    weights = build_quantile_weight_matrix(factor, bundle, config)

    records = compute_daily_backtest_records(
        factor, bundle, config, weights, mode="long_only", holding_period_days=1
    )

    assert records, "records must still be emitted when returns are unevaluable"
    unevaluable = [record for record in records if record.long_return is None]
    assert unevaluable, "a truncated series must produce unevaluable returns"
    assert all(
        record.long_return != 0.0 for record in unevaluable
    ), "an unevaluable return must not be reported as zero"
