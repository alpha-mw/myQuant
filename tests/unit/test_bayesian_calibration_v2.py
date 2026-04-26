from __future__ import annotations

import math
from pathlib import Path

import pytest

from quant_investor.bayesian.calibration_v2 import (
    GROUP_ALL_HORIZONS,
    GROUP_ALL_MARKETS,
    GROUP_ALL_REGIMES,
    TARGET_POSTERIOR_WIN_RATE,
    CalibrationCurveKey,
    CalibrationTrainingExample,
    CalibrationV2Store,
    brier_score,
    bucket_index_for_value,
    build_calibration_curve,
    build_calibration_report,
    build_training_examples,
    log_loss,
    normalize_score_to_unit_interval,
    train_calibration_model,
)
from quant_investor.bayesian.outcome_ledger import (
    OUTCOME_STATUS_MISSING_DATA,
    OUTCOME_STATUS_RESOLVED,
    OutcomeLedgerStore,
    OutcomeRecord,
    PredictionRecord,
    make_outcome_id,
)
from quant_investor.branch_config import CANONICAL_BRANCH_ORDER


def _prediction(prediction_id: str, posterior: float = 0.62, rank: int = 1) -> PredictionRecord:
    return PredictionRecord(
        prediction_id=prediction_id,
        run_id="run-cn-full-a-20260426",
        run_date="2026-04-26",
        rebalance_date="2026-04-26",
        latest_trade_date="2026-04-25",
        horizon_days=20,
        horizon_label="20D",
        symbol=f"00000{rank}.SZ",
        company_name="测试公司",
        market="CN",
        universe_key="full_a",
        universe_hash="hash123",
        macro_regime="趋势上涨",
        rank=rank,
        prior={"composite_prior": 0.55},
        likelihoods={"quant_likelihood": 0.64},
        branch_scores={
            "quant": 0.4,
            "kline": -0.4,
            "intelligence": 0.1,
            "fundamental": 0.2,
            "macro": -0.2,
            "noncanonical": 0.99,
        },
        branch_confidences={"quant": 0.7},
        posterior_win_rate=posterior,
        posterior_expected_alpha=0.03,
        posterior_confidence=0.7,
        posterior_action_score=0.6,
        posterior_edge_after_costs=0.02,
        action_threshold_used=0.55,
    )


def _outcome(prediction: PredictionRecord, *, status: str = OUTCOME_STATUS_RESOLVED, excess: float | None = 0.04, realized: float | None = 0.02) -> OutcomeRecord:
    return OutcomeRecord(
        outcome_id=make_outcome_id(
            prediction_id=prediction.prediction_id,
            resolution_date="2026-05-26",
            status=status,
        ),
        prediction_id=prediction.prediction_id,
        run_id=prediction.run_id,
        symbol=prediction.symbol,
        market=prediction.market,
        horizon_days=prediction.horizon_days,
        horizon_label=prediction.horizon_label,
        run_date=prediction.run_date,
        resolution_date="2026-05-26",
        status=status,
        realized_return=realized,
        benchmark_return=0.01,
        excess_return=excess,
    )


def test_normalize_score_to_unit_interval() -> None:
    assert normalize_score_to_unit_interval(0.8) == pytest.approx(0.8)
    assert normalize_score_to_unit_interval(-0.4) == pytest.approx(0.3)
    assert normalize_score_to_unit_interval(2.0) == pytest.approx(1.0)
    assert normalize_score_to_unit_interval(-2.0) == pytest.approx(0.0)

    with pytest.raises(ValueError, match="finite"):
        normalize_score_to_unit_interval(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        normalize_score_to_unit_interval(float("inf"))


def test_build_training_examples_uses_resolved_outcomes_and_deterministic_order() -> None:
    prediction = _prediction("pred-1")
    missing_prediction = _prediction("pred-2", rank=2)
    outcomes = [
        _outcome(prediction, excess=0.04, realized=-0.02),
        _outcome(missing_prediction, status=OUTCOME_STATUS_MISSING_DATA, excess=0.10, realized=0.10),
    ]

    examples = build_training_examples([prediction, missing_prediction], outcomes)

    assert [example.target_name for example in examples] == [
        TARGET_POSTERIOR_WIN_RATE,
        *(f"branch:{branch}" for branch in CANONICAL_BRANCH_ORDER),
    ]
    assert examples[0].realized_label == 1
    assert examples[0].excess_return == pytest.approx(0.04)
    assert examples[2].target_name == "branch:kline"
    assert examples[2].raw_value == pytest.approx(-0.4)
    assert all("noncanonical" not in example.target_name for example in examples)


def test_build_training_examples_filters_small_returns() -> None:
    prediction = _prediction("pred-1")
    examples = build_training_examples([prediction], [_outcome(prediction, excess=0.001)], min_abs_return=0.01)

    assert examples == []


def test_bucket_curve_includes_empty_buckets_and_beta_binomial_probability() -> None:
    examples = [
        CalibrationTrainingExample(normalized_value=0.0, raw_value=0.0, realized_label=1),
        CalibrationTrainingExample(normalized_value=1.0, raw_value=1.0, realized_label=0),
    ]
    curve = build_calibration_curve(
        examples,
        CalibrationCurveKey(),
        bucket_count=2,
        prior_strength=4.0,
    )

    assert bucket_index_for_value(0.0, 2) == 0
    assert bucket_index_for_value(1.0, 2) == 1
    assert len(curve.buckets) == 2
    first = curve.buckets[0]
    assert first.total_count == 1
    assert first.positive_count == 1
    assert first.calibrated_probability == pytest.approx((1 + 0.25 * 4.0) / (1 + 4.0))


def test_build_calibration_curve_rejects_invalid_normalized_values() -> None:
    examples = [CalibrationTrainingExample(normalized_value=-0.1, raw_value=-0.1, realized_label=1)]

    with pytest.raises(ValueError, match="normalized_value"):
        build_calibration_curve(examples, CalibrationCurveKey(), bucket_count=2)


def test_model_training_creates_global_fallback_and_calibrates() -> None:
    examples: list[CalibrationTrainingExample] = []
    for index in range(4):
        label = 1 if index % 2 == 0 else 0
        examples.append(
            CalibrationTrainingExample(
                prediction_id=f"pred-{index}",
                target_name=TARGET_POSTERIOR_WIN_RATE,
                market="CN",
                horizon_label="20D",
                macro_regime="趋势上涨",
                raw_value=0.2 + index * 0.2,
                normalized_value=0.2 + index * 0.2,
                realized_label=label,
            )
        )
        examples.append(
            CalibrationTrainingExample(
                prediction_id=f"pred-{index}",
                target_name="branch:quant",
                market="CN",
                horizon_label="20D",
                macro_regime="趋势上涨",
                raw_value=0.1 + index * 0.1,
                normalized_value=0.1 + index * 0.1,
                realized_label=label,
            )
        )

    model = train_calibration_model(examples, bucket_count=5, min_examples_per_curve=30, trained_at="2026-04-26T00:00:00Z")

    assert model.get_curve(CalibrationCurveKey(TARGET_POSTERIOR_WIN_RATE, GROUP_ALL_MARKETS, GROUP_ALL_HORIZONS, GROUP_ALL_REGIMES)) is not None
    assert model.get_curve(CalibrationCurveKey("branch:quant", GROUP_ALL_MARKETS, GROUP_ALL_HORIZONS, GROUP_ALL_REGIMES)) is not None
    assert 0.0 <= model.calibrate(TARGET_POSTERIOR_WIN_RATE, 0.7, market="CN", horizon_label="20D", macro_regime="趋势上涨") <= 1.0
    assert model.calibrate("missing_target", -0.4) == pytest.approx(0.3)


def test_metric_helpers_return_known_values() -> None:
    assert brier_score([0.25, 0.75], [0, 1]) == pytest.approx(0.0625)
    expected_log_loss = -0.5 * (math.log(0.75) + math.log(0.75))
    assert log_loss([0.25, 0.75], [0, 1]) == pytest.approx(expected_log_loss)


def test_build_calibration_report_includes_raw_and_calibrated_metrics() -> None:
    examples = [
        CalibrationTrainingExample(
            prediction_id="pred-1",
            target_name=TARGET_POSTERIOR_WIN_RATE,
            market="CN",
            horizon_label="20D",
            macro_regime="趋势上涨",
            raw_value=0.8,
            normalized_value=0.8,
            realized_label=1,
        ),
        CalibrationTrainingExample(
            prediction_id="pred-2",
            target_name=TARGET_POSTERIOR_WIN_RATE,
            market="CN",
            horizon_label="20D",
            macro_regime="趋势上涨",
            raw_value=0.2,
            normalized_value=0.2,
            realized_label=0,
        ),
    ]
    model = train_calibration_model(examples, bucket_count=5, min_examples_per_curve=30, trained_at="2026-04-26T00:00:00Z")
    report = build_calibration_report(model, examples, generated_at="2026-04-26T00:00:00Z")

    assert report.total_examples == 2
    assert len(report.metric_summaries) == 1
    summary = report.metric_summaries[0]
    assert summary.raw_brier_score is not None
    assert summary.calibrated_brier_score is not None
    assert summary.raw_log_loss is not None
    assert summary.calibrated_log_loss is not None


def test_calibration_v2_store_trains_from_ledger_and_round_trips(tmp_path: Path) -> None:
    ledger_store = OutcomeLedgerStore(tmp_path / "ledger")
    predictions = [_prediction("pred-1", posterior=0.7), _prediction("pred-2", posterior=0.3, rank=2)]
    for prediction in predictions:
        ledger_store.append_prediction(prediction)
    ledger_store.append_outcome(_outcome(predictions[0], excess=0.05, realized=0.04))
    ledger_store.append_outcome(_outcome(predictions[1], excess=-0.02, realized=0.01))

    calibration_store = CalibrationV2Store(tmp_path / "calibration")
    model, report = calibration_store.train_from_ledger(
        ledger_store,
        bucket_count=5,
        min_examples_per_curve=30,
        metadata={"test": True},
    )

    assert calibration_store.model_path.exists()
    assert calibration_store.report_path.exists()
    assert len(model.curves) >= 2
    assert report.total_examples == 2 * (1 + len(CANONICAL_BRANCH_ORDER))
    assert calibration_store.load_model().model_id == model.model_id
    assert calibration_store.load_report().model_id == report.model_id


def test_empty_report_has_no_summaries() -> None:
    model = train_calibration_model([], trained_at="2026-04-26T00:00:00Z")
    report = build_calibration_report(model, [], generated_at="2026-04-26T00:00:00Z")

    assert report.total_examples == 0
    assert report.metric_summaries == []
