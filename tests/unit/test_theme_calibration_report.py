from __future__ import annotations

from quant_investor.themes.calibration import (
    evidence_quality,
    score_bucket,
    summarize_records,
    theme_score_bucket,
)
from quant_investor.themes.replay import ThemeReplayRecord


def _record(
    *,
    symbol: str,
    phase: str = "confirmed_rotation",
    symbol_theme_score: float = 0.75,
    theme_score: float = 72.0,
    alpha_5d: float | None = 0.03,
    alpha_10d: float | None = 0.04,
    alpha_20d: float | None = 0.05,
    hit_5d: bool | None = True,
    hit_10d: bool | None = True,
    drawdown: float | None = -0.03,
    runup: float | None = 0.07,
    data_available: bool = True,
) -> ThemeReplayRecord:
    return ThemeReplayRecord(
        symbol=symbol,
        primary_theme_id="theme::ai",
        primary_theme_name="AI",
        phase=phase,
        symbol_theme_score=symbol_theme_score,
        theme_score=theme_score,
        forward_alpha_5d=alpha_5d,
        forward_alpha_10d=alpha_10d,
        forward_alpha_20d=alpha_20d,
        hit_5d=hit_5d,
        hit_10d=hit_10d,
        max_drawdown_10d=drawdown,
        max_runup_10d=runup,
        data_available=data_available,
    )


def test_summarize_records_positive_group():
    records = [_record(symbol=f"000{i:03d}.SZ") for i in range(12)]

    summary = summarize_records(records, group_key="confirmed_rotation")

    assert summary.record_count == 12
    assert summary.available_count == 12
    assert summary.avg_forward_alpha_5d is not None
    assert summary.avg_forward_alpha_5d > 0
    assert summary.hit_rate_5d is not None
    assert summary.hit_rate_5d > 0.5
    assert summary.evidence_quality == "weak"
    assert evidence_quality(summary.available_count) == "weak"


def test_summarize_records_low_sample_flag():
    records = [_record(symbol=f"000{i:03d}.SZ") for i in range(3)]

    summary = summarize_records(records, group_key="small_sample", min_sample=10)

    assert summary.available_count == 3
    assert summary.evidence_quality == "insufficient"
    assert "low_sample" in summary.diagnostic_flags


def test_score_bucket_boundaries():
    assert score_bucket(None) == "missing"
    assert score_bucket(0.10) == "0.00-0.40"
    assert score_bucket(0.40) == "0.40-0.55"
    assert score_bucket(0.55) == "0.55-0.70"
    assert score_bucket(0.70) == "0.70-0.85"
    assert score_bucket(0.85) == "0.85-1.00"


def test_theme_score_bucket_boundaries():
    assert theme_score_bucket(None) == "missing"
    assert theme_score_bucket(10.0) == "0-40"
    assert theme_score_bucket(40.0) == "40-55"
    assert theme_score_bucket(55.0) == "55-70"
    assert theme_score_bucket(70.0) == "70-85"
    assert theme_score_bucket(85.0) == "85-100"
