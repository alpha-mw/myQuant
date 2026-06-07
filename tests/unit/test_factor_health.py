from __future__ import annotations

from quant_investor.factors.governance import (
    FactorGateEvaluator,
    FactorLifecycleState,
    FactorRecord,
)
from quant_investor.factors.health import (
    FactorHealthAction,
    FactorHealthStatus,
    apply_health_decision,
    classify_factor_health,
)


def _base_metrics(**overrides):
    metrics = {
        "no_future_leakage": True,
        "uses_availability_date": True,
        "point_in_time_rebalance": True,
        "adjusted_price_consistent": True,
        "tradability_rules_defined": True,
        "missingness_explained": True,
        "coverage_rate": 0.75,
        "nan_rate": 0.10,
        "monthly_coverage_min": 0.60,
        "max_sector_coverage_share": 0.45,
        "max_size_bucket_coverage_share": 0.55,
        "extreme_value_ratio": 0.01,
        "icir": 0.62,
        "mean_rankic": 0.035,
        "positive_ic_ratio": 0.58,
        "rankic_direction_stable": True,
        "max_single_year_ic_contribution": 0.35,
        "top_bottom_spread": 0.08,
        "top_quantile_return": 0.05,
        "monotonicity": 0.55,
        "long_short_from_long_side": True,
        "turnover": 4.0,
        "cost_adjusted_return": 0.04,
        "slippage_sensitivity_ok": True,
        "execution_realism": True,
        "capacity_pressure": 0.30,
        "neutralized_icir": 0.31,
        "existing_factor_corr": 0.25,
        "style_exposure_only": False,
        "oos_positive_ratio": 0.60,
        "parameter_stability": True,
        "date_range_robustness": True,
        "rebalance_frequency_robustness": True,
        "universe_robustness": True,
        "regime_robustness": True,
        "master_return_delta": 0.025,
        "sharpe_delta": 0.10,
        "max_drawdown_delta": -0.01,
        "turnover_delta": 0.05,
        "execution_cost_delta": 0.002,
        "signal_corr": 0.20,
    }
    metrics.update(overrides)
    return metrics


def _evaluation(name: str, **metric_overrides):
    metrics = _base_metrics(**metric_overrides)
    review = FactorGateEvaluator().evaluate(factor_name=name, metrics=metrics)
    return {
        "name": name,
        "metrics": metrics,
        "review": review.to_dict(),
        "diagnostics": {"evaluation_end_date": "2026-06-30", "rankic_count": 21},
    }


def test_factor_health_first_failure_enters_watchlist_without_weight_change():
    record = FactorRecord(
        name="weak_factor",
        state=FactorLifecycleState.PRODUCTION_FACTOR,
        weight=0.05,
    )

    decision = classify_factor_health(
        record,
        _evaluation("weak_factor", icir=0.20),
    )

    assert decision.status == FactorHealthStatus.WATCHLIST
    assert decision.action == FactorHealthAction.WATCHLIST
    assert decision.consecutive_failures == 1
    assert decision.new_weight == record.weight


def test_factor_health_reduces_then_deprecates_after_repeated_failures():
    record = FactorRecord(
        name="decaying_factor",
        state=FactorLifecycleState.PRODUCTION_FACTOR,
        weight=0.05,
    )
    evaluation = _evaluation(
        "decaying_factor",
        icir=0.20,
        positive_ic_ratio=0.40,
    )

    reduce_decision = classify_factor_health(
        record,
        evaluation,
        previous_failure_count=1,
    )
    assert reduce_decision.action == FactorHealthAction.REDUCE_WEIGHT
    apply_health_decision(
        record,
        reduce_decision,
        reviewed_at="2026-07-01T00:00:00Z",
    )
    assert record.weight == 0.025
    assert record.state == FactorLifecycleState.PRODUCTION_FACTOR

    deprecate_decision = classify_factor_health(
        record,
        evaluation,
        previous_failure_count=2,
    )
    assert deprecate_decision.action == FactorHealthAction.DEPRECATE
    apply_health_decision(
        record,
        deprecate_decision,
        reviewed_at="2026-08-01T00:00:00Z",
    )
    assert record.weight == 0.0
    assert record.state == FactorLifecycleState.DEPRECATED
    assert record.deprecated_reason


def test_factor_health_duplicate_evaluation_observes_without_double_counting():
    record = FactorRecord(
        name="duplicate_window_factor",
        state=FactorLifecycleState.PRODUCTION_FACTOR,
        weight=0.05,
    )

    decision = classify_factor_health(
        record,
        _evaluation("duplicate_window_factor", icir=0.20),
        previous_failure_count=1,
        count_failure=False,
    )

    assert decision.status == FactorHealthStatus.WATCHLIST
    assert decision.action == FactorHealthAction.OBSERVE
    assert decision.consecutive_failures == 1
    assert decision.new_weight == record.weight
