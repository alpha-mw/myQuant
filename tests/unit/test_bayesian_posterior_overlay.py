from __future__ import annotations

import math

import pytest

from quant_investor.bayesian.calibration_v2 import (
    GROUP_ALL_HORIZONS,
    GROUP_ALL_MARKETS,
    GROUP_ALL_REGIMES,
    TARGET_POSTERIOR_WIN_RATE,
    CalibrationBucket,
    CalibrationCurve,
    CalibrationCurveKey,
    CalibrationModelV2,
)
from quant_investor.bayesian.posterior_overlay import (
    OVERLAY_METADATA_KEY,
    EdgeCostConfig,
    attach_overlay_metadata,
    bps_to_decimal_return,
    build_calibrated_posterior_overlay,
    build_calibrated_posterior_overlays,
    clamp_probability,
    horizon_label_for_days,
)
from quant_investor.bayesian.types import PosteriorResult


def _bucket(index: int, probability: float, bucket_count: int = 2) -> CalibrationBucket:
    lower = index / bucket_count
    upper = (index + 1) / bucket_count
    return CalibrationBucket(
        bucket_index=index,
        lower_bound=lower,
        upper_bound=upper,
        center=(lower + upper) / 2.0,
        total_count=10,
        positive_count=int(probability * 10),
        raw_mean=(lower + upper) / 2.0,
        empirical_rate=probability,
        prior_alpha=0.0,
        prior_beta=0.0,
        calibrated_probability=probability,
    )


def _curve(
    key: CalibrationCurveKey,
    *,
    low_probability: float = 0.20,
    high_probability: float = 0.80,
    total_examples: int = 20,
) -> CalibrationCurve:
    return CalibrationCurve(
        key=key,
        bucket_count=2,
        prior_strength=0.0,
        total_examples=total_examples,
        positive_examples=int(total_examples * 0.5),
        base_rate=0.5,
        buckets=[
            _bucket(0, low_probability),
            _bucket(1, high_probability),
        ],
    )


def _model(*curves: CalibrationCurve) -> CalibrationModelV2:
    return CalibrationModelV2(
        model_id="calibration-v2-test",
        trained_at="2026-04-26T00:00:00Z",
        bucket_count=2,
        prior_strength=0.0,
        min_examples_per_curve=1,
        curves=list(curves),
    )


def _posterior(symbol: str = "000001.SZ", win_rate: float = 0.75) -> PosteriorResult:
    return PosteriorResult(
        symbol=symbol,
        company_name="测试公司",
        posterior_win_rate=win_rate,
        posterior_expected_alpha=0.04,
        posterior_confidence=0.70,
        posterior_action_score=0.65,
        posterior_edge_after_costs=0.02,
        posterior_capacity_penalty=0.01,
        action_threshold_used=0.55,
        metadata={"source": "unit"},
    )


def test_numeric_helpers() -> None:
    assert bps_to_decimal_return(100.0) == pytest.approx(0.01)
    assert clamp_probability(-0.1) == 0.0
    assert clamp_probability(1.1) == 1.0
    assert horizon_label_for_days(20) == "20D"

    with pytest.raises(ValueError, match="finite"):
        clamp_probability(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        bps_to_decimal_return(float("inf"))


def test_edge_cost_config_validation() -> None:
    assert EdgeCostConfig().transaction_cost_bps == 0.0

    with pytest.raises(ValueError, match="non-negative"):
        EdgeCostConfig(transaction_cost_bps=-1.0)
    with pytest.raises(ValueError, match="calibration_blend_weight"):
        EdgeCostConfig(calibration_blend_weight=1.1)
    with pytest.raises(ValueError, match="max_probability_adjustment"):
        EdgeCostConfig(max_probability_adjustment=-0.1)


def test_calibration_overlay_computes_edge_after_costs() -> None:
    model = _model(_curve(CalibrationCurveKey()))
    result = _posterior(win_rate=0.75)
    config = EdgeCostConfig(
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
        market_impact_bps=15.0,
        risk_capital_charge=0.002,
    )

    overlay = build_calibrated_posterior_overlay(
        result,
        model,
        market="CN",
        horizon_days=20,
        macro_regime="趋势上涨",
        edge_cost_config=config,
    )

    assert 0.0 <= overlay.calibrated_posterior_win_rate <= 1.0
    assert overlay.calibrated_posterior_win_rate == pytest.approx(0.80)
    assert overlay.calibrated_posterior_expected_alpha == pytest.approx((0.80 - 0.50) * 0.10)
    assert overlay.edge_breakdown.total_cost_penalty == pytest.approx(0.015)
    assert overlay.calibrated_edge_after_costs == pytest.approx(0.015)
    assert overlay.diagnostics.model_id == "calibration-v2-test"
    assert overlay.diagnostics.selected_curve_examples == 20
    assert overlay.metadata["posterior_result_metadata"]["source"] == "unit"

    round_trip = type(overlay).from_dict(overlay.to_dict())
    assert round_trip.diagnostics.model_id == overlay.diagnostics.model_id
    assert round_trip.edge_breakdown.total_cost_penalty == pytest.approx(overlay.edge_breakdown.total_cost_penalty)


def test_probability_adjustment_cap() -> None:
    model = _model(_curve(CalibrationCurveKey(), low_probability=0.95, high_probability=0.95))
    config = EdgeCostConfig(max_probability_adjustment=0.05)

    overlay = build_calibrated_posterior_overlay(_posterior(win_rate=0.40), model, edge_cost_config=config)

    assert overlay.diagnostics.cap_applied is True
    assert overlay.diagnostics.probability_delta_after_cap == pytest.approx(0.05)
    assert overlay.calibrated_posterior_win_rate == pytest.approx(0.45)


def test_blend_behavior_before_cap() -> None:
    model = _model(_curve(CalibrationCurveKey(), high_probability=0.90))
    result = _posterior(win_rate=0.70)

    raw_overlay = build_calibrated_posterior_overlay(
        result,
        model,
        edge_cost_config=EdgeCostConfig(calibration_blend_weight=0.0),
    )
    half_overlay = build_calibrated_posterior_overlay(
        result,
        model,
        edge_cost_config=EdgeCostConfig(calibration_blend_weight=0.5),
    )
    full_overlay = build_calibrated_posterior_overlay(
        result,
        model,
        edge_cost_config=EdgeCostConfig(calibration_blend_weight=1.0),
    )

    assert raw_overlay.calibrated_posterior_win_rate == pytest.approx(0.70)
    assert half_overlay.calibrated_posterior_win_rate == pytest.approx(0.80)
    assert full_overlay.calibrated_posterior_win_rate == pytest.approx(0.90)


def test_batch_overlay_preserves_order_rejects_duplicate_and_does_not_mutate() -> None:
    model = _model(_curve(CalibrationCurveKey()))
    first = _posterior("000001.SZ")
    second = _posterior("000002.SZ", win_rate=0.25)

    overlays = build_calibrated_posterior_overlays([first, second], model)

    assert [overlay.symbol for overlay in overlays] == ["000001.SZ", "000002.SZ"]
    assert OVERLAY_METADATA_KEY not in first.metadata
    assert first.posterior_win_rate == pytest.approx(0.75)
    with pytest.raises(ValueError, match="Duplicate"):
        build_calibrated_posterior_overlays([first, _posterior("000001.SZ")], model)


def test_attach_overlay_metadata_mutation_modes_preserve_core_fields() -> None:
    model = _model(_curve(CalibrationCurveKey()))
    result = _posterior()
    overlay = build_calibrated_posterior_overlay(result, model)

    copied = attach_overlay_metadata(result, overlay, mutate=False)

    assert copied is not result
    assert OVERLAY_METADATA_KEY in copied.metadata
    assert OVERLAY_METADATA_KEY not in result.metadata
    assert copied.posterior_win_rate == result.posterior_win_rate
    assert copied.posterior_expected_alpha == result.posterior_expected_alpha
    assert copied.posterior_action_score == result.posterior_action_score
    assert copied.posterior_edge_after_costs == result.posterior_edge_after_costs

    mutated = attach_overlay_metadata(result, overlay, mutate=True)
    assert mutated is result
    assert OVERLAY_METADATA_KEY in result.metadata
    assert result.posterior_win_rate == pytest.approx(0.75)


def test_calibration_v2_select_curve_fallback_order_and_calibrate_behavior() -> None:
    exact_key = CalibrationCurveKey(TARGET_POSTERIOR_WIN_RATE, "CN", "20D", "趋势上涨")
    exact = _curve(exact_key, high_probability=0.66, total_examples=40)
    global_curve = _curve(
        CalibrationCurveKey(TARGET_POSTERIOR_WIN_RATE, GROUP_ALL_MARKETS, GROUP_ALL_HORIZONS, GROUP_ALL_REGIMES),
        high_probability=0.80,
        total_examples=20,
    )
    model = _model(global_curve, exact)

    selected_exact = model.select_curve(
        TARGET_POSTERIOR_WIN_RATE,
        market="CN",
        horizon_label="20D",
        macro_regime="趋势上涨",
    )
    selected_fallback = model.select_curve(
        TARGET_POSTERIOR_WIN_RATE,
        market="US",
        horizon_label="20D",
        macro_regime="震荡",
    )

    assert selected_exact is exact
    assert selected_fallback is global_curve
    assert model.select_curve("missing_target", market="CN", horizon_label="20D") is None
    assert model.calibrate(
        TARGET_POSTERIOR_WIN_RATE,
        0.75,
        market="CN",
        horizon_label="20D",
        macro_regime="趋势上涨",
    ) == pytest.approx(0.66)
    assert model.calibrate("missing_target", -0.4) == pytest.approx(0.3)
