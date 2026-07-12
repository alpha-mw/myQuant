import pandas as pd

from quant_investor.factors.governance import (
    FactorGateEvaluator,
    FactorLifecycleState,
    FactorRecord,
    GateResult,
)
from quant_investor.factors.runtime import (
    MinedFactorRegistry,
    score_with_mined_factors,
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
        metrics=_base_metrics(
            oos_positive_ratio=0.40,
            parameter_stability=False,
        ),
    )
    assert review.decision.value == "paper_factor"
    assert review.target_state == FactorLifecycleState.PAPER_FACTOR


def test_quant_runtime_only_consumes_production_factor_with_all_gates_passed():
    passed_gates = [
        GateResult(
            gate_id=i,
            gate_key=f"gate_{i}",
            title=f"Gate {i}",
            passed=True,
        )
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
        "AAA": pd.DataFrame(
            {"date": dates, "close": range(80), "volume": [1000] * 80}
        ),
        "BBB": pd.DataFrame(
            {
                "date": dates,
                "close": list(reversed(range(80))),
                "volume": [1000] * 80,
            }
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
        GateResult(
            gate_id=i,
            gate_key=f"gate_{i}",
            title=f"Gate {i}",
            passed=True,
        )
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
    result = score_with_mined_factors(
        frames,
        registry=MinedFactorRegistry.from_records([production]),
    )
    assert result.factor_count == 1
    assert result.factors_used == ["pv_short_reversal_5d"]
    assert result.coverage_rate == 1.0
    assert result.symbol_scores["BBB"] > result.symbol_scores["AAA"]


def test_quant_runtime_batches_price_volume_frame_preparation(monkeypatch):
    import quant_investor.factors.price_volume as price_volume

    passed_gates = [
        GateResult(
            gate_id=i,
            gate_key=f"gate_{i}",
            title=f"Gate {i}",
            passed=True,
        )
        for i in range(1, 9)
    ]
    factors = [
        FactorRecord(
            name=name,
            state=FactorLifecycleState.PRODUCTION_FACTOR,
            implementation=f"price_volume:{name}",
            weight=1.0,
            gate_results=passed_gates,
        )
        for name in (
            "pv_short_reversal_5d",
            "pv_volume_stability_10d",
            "pv_amihud_illiquidity_5d",
        )
    ]
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    frames = {
        "AAA": pd.DataFrame(
            {
                "trade_date": dates,
                "adj_close": list(range(100, 180)),
                "vol": [1000 + idx for idx in range(80)],
                "amount": [100_000 + idx * 100 for idx in range(80)],
            }
        ),
        "BBB": pd.DataFrame(
            {
                "trade_date": dates,
                "adj_close": list(reversed(range(100, 180))),
                "vol": [1500 + idx for idx in range(80)],
                "amount": [150_000 + idx * 100 for idx in range(80)],
            }
        ),
    }
    ordered_frame_calls = {"count": 0}
    original_ordered_frame = price_volume._ordered_frame

    def _counting_ordered_frame(frame, *args, **kwargs):
        ordered_frame_calls["count"] += 1
        return original_ordered_frame(frame, *args, **kwargs)

    monkeypatch.setattr(price_volume, "_ordered_frame", _counting_ordered_frame)

    result = score_with_mined_factors(
        frames,
        registry=MinedFactorRegistry.from_records(factors),
    )

    assert result.factor_count == 3
    assert set(result.factors_used) == {factor.name for factor in factors}
    assert ordered_frame_calls["count"] <= len(frames)


def test_price_volume_preparation_skips_sort_for_ordered_frames(monkeypatch):
    import quant_investor.factors.price_volume as price_volume

    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    frames = {
        "AAA": pd.DataFrame(
            {
                "trade_date": dates,
                "adj_close": list(range(100, 180)),
                "vol": [1000 + idx for idx in range(80)],
                "amount": [100_000 + idx * 100 for idx in range(80)],
            }
        ),
        "BBB": pd.DataFrame(
            {
                "trade_date": dates,
                "adj_close": list(reversed(range(100, 180))),
                "vol": [1500 + idx for idx in range(80)],
                "amount": [150_000 + idx * 100 for idx in range(80)],
            }
        ),
    }
    sort_values_calls = {"count": 0}
    original_sort_values = pd.DataFrame.sort_values

    def _counting_sort_values(self, *args, **kwargs):
        sort_values_calls["count"] += 1
        return original_sort_values(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "sort_values", _counting_sort_values)

    prepared = price_volume.prepare_price_volume_frames(
        frames,
        include_amihud_base=True,
    )

    assert set(prepared) == {"AAA", "BBB"}
    assert sort_values_calls["count"] == 0


def test_price_volume_preparation_skips_numeric_coercion_for_numeric_frames(monkeypatch):
    import quant_investor.factors.price_volume as price_volume

    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    frames = {
        "AAA": pd.DataFrame(
            {
                "trade_date": dates,
                "adj_close": [100.0 + idx for idx in range(80)],
                "vol": [1000.0 + idx for idx in range(80)],
                "amount": [100_000.0 + idx * 100 for idx in range(80)],
            }
        ),
        "BBB": pd.DataFrame(
            {
                "trade_date": dates,
                "adj_close": [180.0 - idx for idx in range(80)],
                "vol": [1500.0 + idx for idx in range(80)],
                "amount": [150_000.0 + idx * 100 for idx in range(80)],
            }
        ),
    }
    to_numeric_calls = {"count": 0}
    original_to_numeric = pd.to_numeric

    def _counting_to_numeric(*args, **kwargs):
        to_numeric_calls["count"] += 1
        return original_to_numeric(*args, **kwargs)

    monkeypatch.setattr(pd, "to_numeric", _counting_to_numeric)

    prepared = price_volume.prepare_price_volume_frames(
        frames,
        include_amihud_base=True,
    )

    assert set(prepared) == {"AAA", "BBB"}
    assert float(prepared["AAA"].close.iloc[-1]) == 179.0
    assert to_numeric_calls["count"] == 0


def test_price_volume_preparation_tails_ordered_frames_to_lookback():
    import quant_investor.factors.price_volume as price_volume

    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    frames = {
        "AAA": pd.DataFrame(
            {
                "trade_date": dates,
                "adj_close": list(range(100, 180)),
                "vol": [1000 + idx for idx in range(80)],
                "amount": [100_000 + idx * 100 for idx in range(80)],
            }
        ),
    }

    prepared = price_volume.prepare_price_volume_frames(
        frames,
        include_amihud_base=True,
        lookback_rows=30,
    )

    assert len(prepared["AAA"].close) == 30
    assert float(prepared["AAA"].close.iloc[0]) == 150.0


def test_quant_runtime_passes_price_volume_required_lookback(monkeypatch):
    import quant_investor.factors.price_volume as price_volume

    passed_gates = [
        GateResult(
            gate_id=i,
            gate_key=f"gate_{i}",
            title=f"Gate {i}",
            passed=True,
        )
        for i in range(1, 9)
    ]
    factors = [
        FactorRecord(
            name=name,
            state=FactorLifecycleState.PRODUCTION_FACTOR,
            implementation=f"price_volume:{name}",
            weight=1.0,
            gate_results=passed_gates,
        )
        for name in (
            "pv_blend_volstab19x2_mom90_amihud5_w75",
            "pv_low_dollar_volume_30d",
        )
    ]
    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    frames = {
        "AAA": pd.DataFrame(
            {
                "trade_date": dates,
                "adj_close": list(range(100, 400)),
                "vol": [1000 + idx for idx in range(300)],
                "amount": [100_000 + idx * 100 for idx in range(300)],
            }
        ),
        "BBB": pd.DataFrame(
            {
                "trade_date": dates,
                "adj_close": list(reversed(range(100, 400))),
                "vol": [1500 + idx for idx in range(300)],
                "amount": [150_000 + idx * 100 for idx in range(300)],
            }
        ),
    }
    prepared_lookback_rows: list[int] = []
    original_prepare = price_volume.prepare_price_volume_frames

    def _counting_prepare(frames_arg, *args, **kwargs):
        prepared_lookback_rows.append(int(kwargs.get("lookback_rows", 0) or 0))
        return original_prepare(frames_arg, *args, **kwargs)

    monkeypatch.setattr(price_volume, "prepare_price_volume_frames", _counting_prepare)

    result = score_with_mined_factors(
        frames,
        registry=MinedFactorRegistry.from_records(factors),
    )

    assert result.factor_count == 2
    assert prepared_lookback_rows == [91]


def test_quant_runtime_computes_prepared_amihud_base_without_pandas_pct_change(monkeypatch):
    passed_gates = [
        GateResult(
            gate_id=i,
            gate_key=f"gate_{i}",
            title=f"Gate {i}",
            passed=True,
        )
        for i in range(1, 9)
    ]
    factors = [
        FactorRecord(
            name=name,
            state=FactorLifecycleState.PRODUCTION_FACTOR,
            implementation=f"price_volume:{name}",
            weight=1.0,
            gate_results=passed_gates,
        )
        for name in (
            "pv_amihud_illiquidity_10d",
            "pv_amihud_illiquidity_15d",
            "pv_amihud_illiquidity_20d",
        )
    ]
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    frames = {
        "AAA": pd.DataFrame(
            {
                "trade_date": dates,
                "adj_close": list(range(100, 180)),
                "vol": [1000 + idx for idx in range(80)],
                "amount": [100_000 + idx * 100 for idx in range(80)],
            }
        ),
        "BBB": pd.DataFrame(
            {
                "trade_date": dates,
                "adj_close": list(reversed(range(100, 180))),
                "vol": [1500 + idx for idx in range(80)],
                "amount": [150_000 + idx * 100 for idx in range(80)],
            }
        ),
    }
    pct_change_calls = {"count": 0}
    original_pct_change = pd.Series.pct_change

    def _counting_pct_change(self, *args, **kwargs):
        pct_change_calls["count"] += 1
        return original_pct_change(self, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "pct_change", _counting_pct_change)

    result = score_with_mined_factors(
        frames,
        registry=MinedFactorRegistry.from_records(factors),
    )

    assert result.factor_count == 3
    assert set(result.factors_used) == {factor.name for factor in factors}
    assert pct_change_calls["count"] == 0


def test_quant_runtime_reuses_amihud_illiquidity_windows(monkeypatch):
    import quant_investor.factors.price_volume as price_volume

    passed_gates = [
        GateResult(
            gate_id=i,
            gate_key=f"gate_{i}",
            title=f"Gate {i}",
            passed=True,
        )
        for i in range(1, 9)
    ]
    factors = [
        FactorRecord(
            name=name,
            state=FactorLifecycleState.PRODUCTION_FACTOR,
            implementation=f"price_volume:{name}",
            weight=1.0,
            gate_results=passed_gates,
        )
        for name in (
            "pv_amihud_illiquidity_10d",
            "pv_amihud_illiquidity_15d",
            "pv_amihud_illiquidity_20d",
            "pv_amihud_illiquidity_25d",
        )
    ]
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    frames = {
        "AAA": pd.DataFrame(
            {
                "trade_date": dates,
                "adj_close": list(range(100, 180)),
                "vol": [1000 + idx for idx in range(80)],
                "amount": [100_000 + idx * 100 for idx in range(80)],
            }
        ),
        "BBB": pd.DataFrame(
            {
                "trade_date": dates,
                "adj_close": list(reversed(range(100, 180))),
                "vol": [1500 + idx for idx in range(80)],
                "amount": [150_000 + idx * 100 for idx in range(80)],
            }
        ),
    }
    scalar_calls = {"count": 0}
    original_amihud = price_volume._amihud_illiquidity_from_prepared

    def _counting_amihud(*args, **kwargs):
        scalar_calls["count"] += 1
        return original_amihud(*args, **kwargs)

    monkeypatch.setattr(
        price_volume,
        "_amihud_illiquidity_from_prepared",
        _counting_amihud,
    )

    result = score_with_mined_factors(
        frames,
        registry=MinedFactorRegistry.from_records(factors),
    )

    assert result.factor_count == 4
    assert set(result.factors_used) == {factor.name for factor in factors}
    assert scalar_calls["count"] == 0


def test_quant_runtime_reuses_volume_stability_windows(monkeypatch):
    import quant_investor.factors.price_volume as price_volume

    passed_gates = [
        GateResult(
            gate_id=i,
            gate_key=f"gate_{i}",
            title=f"Gate {i}",
            passed=True,
        )
        for i in range(1, 9)
    ]
    factors = [
        FactorRecord(
            name=name,
            state=FactorLifecycleState.PRODUCTION_FACTOR,
            implementation=f"price_volume:{name}",
            weight=1.0,
            gate_results=passed_gates,
        )
        for name in (
            "pv_volume_stability_10d",
            "pv_volume_stability_15d",
            "pv_volume_stability_20d",
        )
    ]
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    frames = {
        "AAA": pd.DataFrame(
            {
                "trade_date": dates,
                "adj_close": list(range(100, 180)),
                "vol": [1000 + idx for idx in range(80)],
                "amount": [100_000 + idx * 100 for idx in range(80)],
            }
        ),
        "BBB": pd.DataFrame(
            {
                "trade_date": dates,
                "adj_close": list(reversed(range(100, 180))),
                "vol": [1500 + idx for idx in range(80)],
                "amount": [150_000 + idx * 100 for idx in range(80)],
            }
        ),
    }
    scalar_calls = {"count": 0}
    original_volstab = price_volume._volume_stability_from_volume

    def _counting_volstab(*args, **kwargs):
        scalar_calls["count"] += 1
        return original_volstab(*args, **kwargs)

    monkeypatch.setattr(
        price_volume,
        "_volume_stability_from_volume",
        _counting_volstab,
    )

    result = score_with_mined_factors(
        frames,
        registry=MinedFactorRegistry.from_records(factors),
    )

    assert result.factor_count == 3
    assert set(result.factors_used) == {factor.name for factor in factors}
    assert scalar_calls["count"] == 0


def test_quant_runtime_reuses_low_dollar_volume_windows(monkeypatch):
    import quant_investor.factors.price_volume as price_volume

    passed_gates = [
        GateResult(
            gate_id=i,
            gate_key=f"gate_{i}",
            title=f"Gate {i}",
            passed=True,
        )
        for i in range(1, 9)
    ]
    factors = [
        FactorRecord(
            name=name,
            state=FactorLifecycleState.PRODUCTION_FACTOR,
            implementation=f"price_volume:{name}",
            weight=1.0,
            gate_results=passed_gates,
        )
        for name in (
            "pv_low_dollar_volume_15d",
            "pv_low_dollar_volume_20d",
            "pv_low_dollar_volume_25d",
            "pv_low_dollar_volume_30d",
        )
    ]
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    frames = {
        "AAA": pd.DataFrame(
            {
                "trade_date": dates,
                "adj_close": list(range(100, 180)),
                "vol": [1000 + idx for idx in range(80)],
                "amount": [100_000 + idx * 100 for idx in range(80)],
            }
        ),
        "BBB": pd.DataFrame(
            {
                "trade_date": dates,
                "adj_close": list(reversed(range(100, 180))),
                "vol": [1500 + idx for idx in range(80)],
                "amount": [150_000 + idx * 100 for idx in range(80)],
            }
        ),
    }
    scalar_calls = {"count": 0}
    original_low_dollar_volume = price_volume._low_dollar_volume_from_amount

    def _counting_low_dollar_volume(*args, **kwargs):
        scalar_calls["count"] += 1
        return original_low_dollar_volume(*args, **kwargs)

    monkeypatch.setattr(
        price_volume,
        "_low_dollar_volume_from_amount",
        _counting_low_dollar_volume,
    )

    result = score_with_mined_factors(
        frames,
        registry=MinedFactorRegistry.from_records(factors),
    )

    assert result.factor_count == 4
    assert set(result.factors_used) == {factor.name for factor in factors}
    assert scalar_calls["count"] == 0


def test_quant_runtime_reuses_blend_components_across_weight_variants(monkeypatch):
    import quant_investor.factors.price_volume as price_volume

    passed_gates = [
        GateResult(
            gate_id=i,
            gate_key=f"gate_{i}",
            title=f"Gate {i}",
            passed=True,
        )
        for i in range(1, 9)
    ]
    factors = [
        FactorRecord(
            name=name,
            state=FactorLifecycleState.PRODUCTION_FACTOR,
            implementation=f"price_volume:{name}",
            weight=1.0,
            gate_results=passed_gates,
        )
        for name in (
            "pv_blend_volstab19x2_mom90_amihud5_w75",
            "pv_blend_volstab19x2_mom90_amihud5_w70",
        )
    ]
    dates = pd.date_range("2024-01-01", periods=130, freq="B")
    frames = {
        "AAA": pd.DataFrame(
            {
                "trade_date": dates,
                "adj_close": list(range(100, 230)),
                "vol": [1000 + (idx % 3) * 10 for idx in range(130)],
                "amount": [100_000 + idx * 100 for idx in range(130)],
            }
        ),
        "BBB": pd.DataFrame(
            {
                "trade_date": dates,
                "adj_close": list(reversed(range(100, 230))),
                "vol": [1800 + (idx % 5) * 30 for idx in range(130)],
                "amount": [160_000 + idx * 80 for idx in range(130)],
            }
        ),
        "CCC": pd.DataFrame(
            {
                "trade_date": dates,
                "adj_close": [120 + (idx % 20) for idx in range(130)],
                "vol": [1400 + (idx % 11) * 40 for idx in range(130)],
                "amount": [130_000 + idx * 90 for idx in range(130)],
            }
        ),
    }
    calls = {"volstab": 0, "momentum": 0, "amihud": 0}
    original_volstab = price_volume._volume_stability_smooth_from_volume
    original_momentum = price_volume._momentum_from_close
    original_amihud = price_volume._amihud_illiquidity_from_prepared

    def _counting_volstab(*args, **kwargs):
        calls["volstab"] += 1
        return original_volstab(*args, **kwargs)

    def _counting_momentum(*args, **kwargs):
        calls["momentum"] += 1
        return original_momentum(*args, **kwargs)

    def _counting_amihud(*args, **kwargs):
        calls["amihud"] += 1
        return original_amihud(*args, **kwargs)

    monkeypatch.setattr(price_volume, "_volume_stability_smooth_from_volume", _counting_volstab)
    monkeypatch.setattr(price_volume, "_momentum_from_close", _counting_momentum)
    monkeypatch.setattr(price_volume, "_amihud_illiquidity_from_prepared", _counting_amihud)

    result = score_with_mined_factors(
        frames,
        registry=MinedFactorRegistry.from_records(factors),
    )

    assert result.factor_count == 2
    assert set(result.factors_used) == {factor.name for factor in factors}
    assert calls == {
        "volstab": len(frames),
        "momentum": len(frames),
        "amihud": len(frames),
    }


def test_quant_runtime_single_symbol_coverage_uses_factor_availability():
    passed_gates = [
        GateResult(
            gate_id=i,
            gate_key=f"gate_{i}",
            title=f"Gate {i}",
            passed=True,
        )
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
    }

    result = score_with_mined_factors(
        frames,
        registry=MinedFactorRegistry.from_records([production]),
    )

    assert result.factor_count == 1
    assert result.factor_coverages["pv_short_reversal_5d"] == 1.0
    assert result.coverage_rate == 1.0
    assert result.to_metadata()["applied_to_score"] is True


def test_quant_runtime_consumes_blended_price_volume_factor():
    passed_gates = [
        GateResult(
            gate_id=i,
            gate_key=f"gate_{i}",
            title=f"Gate {i}",
            passed=True,
        )
        for i in range(1, 9)
    ]
    production = FactorRecord(
        name="pv_blend_volstab19x2_mom90_amihud5_w75",
        state=FactorLifecycleState.PRODUCTION_FACTOR,
        implementation=(
            "price_volume:pv_blend_volstab19x2_mom90_amihud5_w75"
        ),
        weight=1.0,
        gate_results=passed_gates,
    )
    dates = pd.date_range("2024-01-01", periods=130, freq="B")
    frames = {
        "AAA": pd.DataFrame(
            {
                "trade_date": dates,
                "adj_close": list(range(100, 230)),
                "vol": [1000 + (idx % 3) * 10 for idx in range(130)],
                "amount": [100_000 + idx * 100 for idx in range(130)],
            }
        ),
        "BBB": pd.DataFrame(
            {
                "trade_date": dates,
                "adj_close": list(reversed(range(100, 230))),
                "vol": [1800 + (idx % 5) * 30 for idx in range(130)],
                "amount": [160_000 + idx * 80 for idx in range(130)],
            }
        ),
        "CCC": pd.DataFrame(
            {
                "trade_date": dates,
                "adj_close": [120 + (idx % 20) for idx in range(130)],
                "vol": [1400 + (idx % 11) * 40 for idx in range(130)],
                "amount": [130_000 + idx * 90 for idx in range(130)],
            }
        ),
    }

    result = score_with_mined_factors(
        frames,
        registry=MinedFactorRegistry.from_records([production]),
    )

    assert result.factor_count == 1
    assert result.factors_used == ["pv_blend_volstab19x2_mom90_amihud5_w75"]
    assert (
        result.factor_coverages["pv_blend_volstab19x2_mom90_amihud5_w75"]
        == 1.0
    )
    assert set(result.symbol_scores) == {"AAA", "BBB", "CCC"}


def test_default_registry_keeps_only_current_full_a_champion():
    registry = MinedFactorRegistry.load()
    selectable = {factor.name: factor for factor in registry.selectable_factors()}

    name = "pv_low_dollar_volume_5d"
    factor = selectable[name]
    assert factor.state == FactorLifecycleState.PRODUCTION_FACTOR
    assert factor.weight == 0.05
    assert factor.all_gates_passed()
    assert factor.implementation == f"price_volume:{name}"

    by_name = {factor.name: factor for factor in registry.factors}
    deprecated = by_name["pv_blend_volstab19x2_mom90_amihud5_w70"]
    assert deprecated.state == FactorLifecycleState.DEPRECATED
    assert deprecated.weight == 0.0
