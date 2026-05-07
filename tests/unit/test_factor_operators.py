from __future__ import annotations

import copy

import pytest

from quant_investor.factors.operators import (
    cs_booksize,
    cs_indneut,
    cs_rank,
    cs_winsorize,
    cs_zscore,
    div,
    log,
    sqrt,
    ts_corr,
    ts_delay,
    ts_mean,
    ts_std,
)


def test_ts_delay_preserves_shape_and_does_not_mutate() -> None:
    values = [[1.0, 2.0, None, 4.0], [10.0, 20.0, 30.0, 40.0]]
    original = copy.deepcopy(values)

    assert ts_delay(values, 2) == [
        [None, None, 1.0, 2.0],
        [None, None, 10.0, 20.0],
    ]
    assert values == original


def test_ts_mean_and_ts_std_use_full_windows() -> None:
    values = [[1.0, 2.0, None, 4.0], [10.0, 20.0, 30.0, 40.0]]

    assert ts_mean(values, 2) == [
        [None, 1.5, None, None],
        [None, 15.0, 25.0, 35.0],
    ]
    assert ts_std(values, 2) == [
        [None, 0.5, None, None],
        [None, 5.0, 5.0, 5.0],
    ]


def test_ts_corr_handles_known_pairs_and_invalid_windows() -> None:
    assert ts_corr([[1.0, 2.0, 3.0, 4.0]], [[2.0, 4.0, 6.0, 8.0]], 3) == [
        [None, None, 1.0, 1.0]
    ]
    assert ts_corr([[1.0, 2.0, 3.0]], [[1.0, 1.0, 1.0]], 3) == [[None, None, None]]
    assert ts_corr([[1.0, None, 3.0]], [[1.0, 2.0, 3.0]], 3) == [[None, None, 1.0]]


def test_cs_rank_ranks_per_date_and_ignores_missing() -> None:
    values = [[1.0, None, 2.0], [3.0, 2.0, 2.0], [2.0, 1.0, None]]

    assert cs_rank(values) == [
        [0.0, None, 0.5],
        [1.0, 1.0, 0.5],
        [0.5, 0.0, None],
    ]


def test_cs_zscore_works_on_known_cross_section() -> None:
    result = cs_zscore([[1.0], [3.0], [2.0]])

    assert result[0][0] == pytest.approx(-1.224744871)
    assert result[1][0] == pytest.approx(1.224744871)
    assert result[2][0] == pytest.approx(0.0)


def test_cs_winsorize_clips_outliers_by_quantile() -> None:
    assert cs_winsorize([[1.0], [2.0], [100.0]], lower_quantile=0.0, upper_quantile=0.5) == [
        [1.0],
        [2.0],
        [2.0],
    ]


def test_cs_indneut_subtracts_industry_mean() -> None:
    result = cs_indneut(
        [[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]],
        industry_by_symbol={"AAA": "Tech", "BBB": "Tech", "CCC": "Finance"},
        symbols=["AAA", "BBB", "CCC"],
    )

    assert result == [[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]


def test_cs_booksize_normalizes_positive_and_negative_books() -> None:
    assert cs_booksize([[1.0, -2.0, 0.0], [3.0, -6.0, None], [-4.0, 8.0, 0.0]]) == [
        [0.25, -0.25, 0.0],
        [0.75, -0.75, None],
        [-1.0, 1.0, 0.0],
    ]


def test_invalid_elementwise_values_return_none() -> None:
    assert div([[1.0, 2.0, None]], [[0.0, 2.0, 3.0]]) == [[None, 1.0, None]]
    assert log([[1.0, 0.0, -1.0]]) == [[0.0, None, None]]
    assert sqrt([[4.0, 0.0, -1.0]]) == [[2.0, 0.0, None]]


def test_window_arguments_must_be_positive_integers() -> None:
    with pytest.raises(ValueError, match="window"):
        ts_mean([[1.0]], 0)
    with pytest.raises(ValueError, match="periods"):
        ts_delay([[1.0]], -1)
