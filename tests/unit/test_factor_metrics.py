from __future__ import annotations

import math

import pytest

from quant_investor.factors.metrics import (
    ReturnMetricSummary,
    TurnoverMetricSummary,
    annualized_return_from_daily,
    annualized_vol_from_daily,
    build_return_metric_summary,
    build_turnover_metric_summary,
    cumulative_return,
    filter_none_finite,
    max_drawdown_from_returns,
    positive_ratio,
    sharpe_from_daily,
)


def test_return_metric_summary_round_trip() -> None:
    summary = build_return_metric_summary(
        "fixture.before_cost",
        [0.01, None, -0.02, 0.03],
        metadata={"slice": "fixture"},
    )

    assert ReturnMetricSummary.from_dict(summary.to_dict()).to_dict() == summary.to_dict()
    assert summary.sample_count == 3
    assert summary.metadata == {"slice": "fixture"}


def test_turnover_metric_summary_round_trip() -> None:
    summary = build_turnover_metric_summary(
        "fixture.turnover",
        [0.10, 0.20, 0.50],
        turnover_budget=0.15,
    )

    assert TurnoverMetricSummary.from_dict(summary.to_dict()).to_dict() == summary.to_dict()
    assert summary.average_turnover == pytest.approx(0.2666666667)
    assert summary.budget_breach_count == 2
    assert summary.budget_breach_ratio == pytest.approx(2 / 3)


def test_cumulative_return_known_case() -> None:
    assert cumulative_return([0.10, -0.10]) == pytest.approx(-0.01)


def test_max_drawdown_known_case() -> None:
    returns = [0.10, -0.20, 0.05]

    assert max_drawdown_from_returns(returns) == pytest.approx(0.20)


def test_annualized_return_vol_and_sharpe_are_deterministic() -> None:
    returns = [0.01, 0.03, -0.02]
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / len(returns)
    expected_ann_ret = mean * 252
    expected_ann_vol = math.sqrt(variance) * math.sqrt(252)

    assert annualized_return_from_daily(returns) == pytest.approx(expected_ann_ret)
    assert annualized_vol_from_daily(returns) == pytest.approx(expected_ann_vol)
    assert sharpe_from_daily(returns) == pytest.approx(expected_ann_ret / expected_ann_vol)


def test_positive_ratio_known_case() -> None:
    assert positive_ratio([0.01, 0.00, -0.01, None, 0.02]) == pytest.approx(0.50)


def test_none_values_are_ignored() -> None:
    assert filter_none_finite([None, 0.01, None, -0.02]) == [0.01, -0.02]
    assert cumulative_return([None, 0.10]) == pytest.approx(0.10)


def test_nan_and_infinite_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        filter_none_finite([float("nan")])
    with pytest.raises(ValueError, match="finite"):
        filter_none_finite([float("inf")])
    with pytest.raises(ValueError, match="JSON-serializable"):
        build_return_metric_summary("bad", [0.01], metadata={"bad": float("nan")})
