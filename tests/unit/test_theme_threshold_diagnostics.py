from __future__ import annotations

from quant_investor.themes.calibration import (
    build_theme_calibration_report,
    build_threshold_diagnostics,
    evaluate_threshold,
)
from quant_investor.themes.replay import ThemeCalibrationDataset, ThemeReplayRecord


def _record(
    *,
    idx: int,
    phase: str = "confirmed_rotation",
    symbol_theme_score: float = 0.78,
    theme_score: float = 76.0,
    alpha_5d: float | None = 0.04,
    alpha_10d: float | None = 0.05,
    alpha_20d: float | None = 0.06,
    hit_5d: bool | None = True,
    hit_10d: bool | None = True,
    drawdown: float | None = -0.03,
    risk_flags: list[str] | None = None,
    data_available: bool = True,
) -> ThemeReplayRecord:
    return ThemeReplayRecord(
        symbol=f"000{idx:03d}.SZ",
        primary_theme_id="theme::ai",
        primary_theme_name="AI",
        phase=phase,
        symbol_theme_score=symbol_theme_score,
        theme_score=theme_score,
        risk_flags=list(risk_flags or []),
        forward_alpha_5d=alpha_5d,
        forward_alpha_10d=alpha_10d,
        forward_alpha_20d=alpha_20d,
        hit_5d=hit_5d,
        hit_10d=hit_10d,
        max_drawdown_10d=drawdown,
        max_runup_10d=0.08,
        data_available=data_available,
    )


def test_evaluate_threshold_candidate_boost():
    records = [_record(idx=i) for i in range(12)]

    diagnostic = evaluate_threshold(
        records,
        threshold_name="symbol_theme_score >= 0.70",
        threshold_value=0.70,
        predicate=lambda record: record.symbol_theme_score >= 0.70,
        total_record_count=len(records),
        min_sample=10,
    )

    assert diagnostic.recommended_action == "candidate_boost"
    assert diagnostic.pass_min_sample is True
    assert diagnostic.pass_alpha is True
    assert diagnostic.pass_hit_rate is True
    assert diagnostic.pass_drawdown is True


def test_evaluate_threshold_risk_reduce_on_negative_alpha():
    records = [
        _record(idx=i, alpha_5d=-0.03, hit_5d=False, drawdown=-0.04)
        for i in range(12)
    ]

    diagnostic = evaluate_threshold(
        records,
        threshold_name="phase == overextended",
        threshold_value="overextended",
        predicate=lambda record: True,
        total_record_count=len(records),
        min_sample=10,
    )

    assert diagnostic.recommended_action == "risk_reduce"
    assert "negative_alpha" in diagnostic.diagnostic_flags


def test_evaluate_threshold_watch_only_low_sample():
    records = [_record(idx=i) for i in range(3)]

    diagnostic = evaluate_threshold(
        records,
        threshold_name="phase == confirmed_rotation",
        threshold_value="confirmed_rotation",
        predicate=lambda record: True,
        total_record_count=len(records),
        min_sample=10,
    )

    assert diagnostic.recommended_action == "watch_only"
    assert "low_sample" in diagnostic.diagnostic_flags


def test_build_threshold_diagnostics_contains_expected_thresholds():
    dataset = ThemeCalibrationDataset(records=[_record(idx=i) for i in range(12)])

    diagnostics = build_threshold_diagnostics(dataset, min_sample=10)
    names = {diagnostic.threshold_name for diagnostic in diagnostics}

    assert "symbol_theme_score >= 0.70" in names
    assert "phase == confirmed_rotation" in names
    assert "no risk flags" in names


def test_build_theme_calibration_report_contains_phase_and_bucket_summaries():
    records = [
        _record(idx=0, phase="confirmed_rotation", symbol_theme_score=0.78),
        _record(idx=1, phase="distribution", symbol_theme_score=0.42),
    ]
    dataset = ThemeCalibrationDataset(records=records)

    report = build_theme_calibration_report(dataset, min_sample=1)

    assert "confirmed_rotation" in report.phase_summaries
    assert "distribution" in report.phase_summaries
    assert "0.70-0.85" in report.score_bucket_summaries
    assert "0.40-0.55" in report.score_bucket_summaries


def test_build_theme_calibration_report_recommended_thresholds():
    dataset = ThemeCalibrationDataset(records=[_record(idx=i) for i in range(12)])

    report = build_theme_calibration_report(dataset, min_sample=10)

    assert report.recommended_thresholds
    assert all(
        diagnostic.recommended_action == "candidate_boost"
        for diagnostic in report.recommended_thresholds
    )


def test_theme_calibration_report_to_dataframe_and_markdown():
    dataset = ThemeCalibrationDataset(records=[_record(idx=i) for i in range(12)])
    report = build_theme_calibration_report(dataset, min_sample=10)

    summary_frame = report.to_dataframe()
    threshold_frame = report.thresholds_dataframe()
    markdown = report.to_markdown()

    assert not summary_frame.empty
    assert not threshold_frame.empty
    assert "## Theme Calibration Diagnostics" in markdown
    assert (
        "This report is offline calibration only and does not change trading behavior."
        in markdown
    )


def test_build_theme_calibration_report_empty_safe():
    report = build_theme_calibration_report(ThemeCalibrationDataset(), min_sample=10)

    assert report.record_count == 0
    assert "insufficient_calibration_sample" in report.warnings
    assert "no_threshold_passed_conservative_filters" in report.warnings
