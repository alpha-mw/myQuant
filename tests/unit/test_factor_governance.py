import pandas as pd

from quant_investor.factors.governance import (
    FactorGateEvaluator,
    FactorLifecycleState,
    FactorRecord,
    GateResult,
)
from quant_investor.factors.runtime import MinedFactorRegistry, score_with_mined_factors


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


def test_gate1_failure_rejects_factor():
    review = FactorGateEvaluator().evaluate(
        factor_name="bad_factor",
        metrics=_base_metrics(no_future_leakage=False),
    )
    assert review.decision.value == "reject"
    assert review.target_state == FactorLifecycleState.DRAFT
    assert not review.gate_results[0].passed


def test_all_gates_passed_becomes_production_candidate_not_production_factor():
    review = FactorGateEvaluator().evaluate(
        factor_name="candidate_factor",
        metrics=_base_metrics(),
    )
    assert review.decision.value == "production_candidate"
    assert review.target_state == FactorLifecycleState.PRODUCTION_CANDIDATE


def test_initial_backtest_without_oos_stays_paper_factor():
    review = FactorGateEvaluator().evaluate(
        factor_name="paper_only",
        metrics=_base_metrics(oos_positive_ratio=0.40, parameter_stability=False),
    )
    assert review.decision.value == "paper_factor"
    assert review.target_state == FactorLifecycleState.PAPER_FACTOR


def test_quant_runtime_only_consumes_production_factor_with_all_gates_passed():
    passed_gates = [
        GateResult(gate_id=i, gate_key=f"gate_{i}", title=f"Gate {i}", passed=True)
        for i in range(1, 9)
    ]
    production = FactorRecord(
        name="momentum_1m",
        state=FactorLifecycleState.PRODUCTION_FACTOR,
        implementation="alpha_mining.FactorLibrary:momentum_1m",
        weight=1.0,
        gate_results=passed_gates,
    )
    paper = FactorRecord(
        name="volatility_penalty",
        state=FactorLifecycleState.PAPER_FACTOR,
        implementation="builtin:volatility_penalty",
        weight=1.0,
        gate_results=passed_gates,
    )
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    frames = {
        "AAA": pd.DataFrame({"date": dates, "close": range(80), "volume": [1000] * 80}),
        "BBB": pd.DataFrame(
            {"date": dates, "close": list(reversed(range(80))), "volume": [1000] * 80}
        ),
    }
    result = score_with_mined_factors(
        frames, registry=MinedFactorRegistry.from_records([production, paper])
    )
    assert result.factor_count == 1
    assert result.factors_used == ["momentum_1m"]
    assert "volatility_penalty" in result.skipped_factors
    assert set(result.symbol_scores) == {"AAA", "BBB"}


def test_quant_runtime_consumes_price_volume_production_factor():
    passed_gates = [
        GateResult(gate_id=i, gate_key=f"gate_{i}", title=f"Gate {i}", passed=True)
        for i in range(1, 9)
    ]
    production = FactorRecord(
        name="pv_short_reversal_5d",
        state=FactorLifecycleState.PRODUCTION_FACTOR,
        implementation="price_volume:pv_short_reversal_5d",
        weight=1.0,
        gate_results=passed_gates,
    )
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    frames = {
        "AAA": pd.DataFrame(
            {
                "trade_date": dates,
                "adj_close": list(range(100, 180)),
                "vol": [1000] * 80,
                "amount": [100_000] * 80,
            }
        ),
        "BBB": pd.DataFrame(
            {
                "trade_date": dates,
                "adj_close": list(reversed(range(100, 180))),
                "vol": [1000] * 80,
                "amount": [100_000] * 80,
            }
        ),
    }
    result = score_with_mined_factors(frames, registry=MinedFactorRegistry.from_records([production]))
    assert result.factor_count == 1
    assert result.factors_used == ["pv_short_reversal_5d"]
    assert result.coverage_rate == 1.0
    assert result.symbol_scores["BBB"] > result.symbol_scores["AAA"]
