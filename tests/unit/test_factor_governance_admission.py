from __future__ import annotations

import pytest

from quant_investor.factors.admission import (
    build_library_entry_from_decision,
    build_production_factor_library,
    evaluate_backtest_against_thresholds,
    propose_admission_decision,
)
from quant_investor.factors.schema import (
    ADMISSION_DECISION_APPROVE_PAPER_TRADING,
    ADMISSION_DECISION_REJECT,
    FACTOR_STATUS_PAPER_TRADING,
    FACTOR_STATUS_PRODUCTION,
    FACTOR_STATUS_REJECTED,
    FACTOR_STATUS_VALIDATED_RESEARCH,
    VALIDATION_VERDICT_FAIL,
    VALIDATION_VERDICT_PASS,
    VALIDATION_VERDICT_WARN,
    FactorAdmissionDecision,
    FactorBacktestResult,
    FactorLibraryEntry,
    FactorValidationThresholds,
    make_admission_decision_id,
    make_backtest_result_id,
)


def _result(**overrides: object) -> FactorBacktestResult:
    payload = {
        "result_id": make_backtest_result_id(
            factor_id="factor-momentum-test",
            factor_version="v1",
            config_id="config-weekly",
        ),
        "factor_id": "factor-momentum-test",
        "factor_version": "v1",
        "config_id": "config-weekly",
        "start_date": "2021-01-01",
        "end_date": "2025-12-31",
        "sample_days": 1000,
        "coverage_ratio": 0.92,
        "missing_ratio": 0.08,
        "ann_ret": 0.12,
        "ann_vol": 0.16,
        "sharpe": 1.20,
        "max_drawdown": 0.12,
        "turnover_avg": 0.25,
        "long_num_avg": 100.0,
        "short_num_avg": 100.0,
        "rank_ic_mean": 0.035,
        "ic_mean": 0.030,
        "icir": 0.50,
        "ic_t_stat": 4.0,
        "positive_ic_ratio": 0.62,
        "top_bottom_spread": 0.08,
        "after_cost_top_bottom_spread": 0.06,
        "before_cost_sharpe": 1.10,
        "after_cost_sharpe": 0.95,
        "monotonicity_score": 1.0,
        "capacity_estimate": 1000000.0,
        "slice_metrics": {"2024": {"rank_ic_mean": 0.03}},
        "metadata": {
            "point_in_time_passed": True,
            "validation_generated_at": "2026-04-27",
        },
    }
    payload.update(overrides)
    return FactorBacktestResult.from_dict(payload)


def test_strong_synthetic_backtest_passes_thresholds() -> None:
    report = evaluate_backtest_against_thresholds(_result())

    assert report.overall_verdict == VALIDATION_VERDICT_PASS
    assert report.recommended_status == FACTOR_STATUS_VALIDATED_RESEARCH
    assert report.failed_gates == []
    assert report.gate_results["rank_ic_mean"] == VALIDATION_VERDICT_PASS
    assert report.thresholds.to_dict() == FactorValidationThresholds().to_dict()


def test_weak_synthetic_backtest_fails_and_recommends_rejected() -> None:
    weak = _result(
        sample_days=200,
        coverage_ratio=0.50,
        rank_ic_mean=-0.01,
        icir=0.05,
        ic_t_stat=0.2,
        after_cost_sharpe=0.10,
        positive_ic_ratio=0.40,
        after_cost_top_bottom_spread=-0.01,
    )
    report = evaluate_backtest_against_thresholds(weak)

    assert report.overall_verdict == VALIDATION_VERDICT_FAIL
    assert report.recommended_status == FACTOR_STATUS_REJECTED
    assert "sample_days" in report.failed_gates
    assert "after_cost_spread" in report.failed_gates


def test_missing_metrics_fail_relevant_gates() -> None:
    missing = _result(
        rank_ic_mean=None,
        icir=None,
        ic_t_stat=None,
        after_cost_sharpe=None,
        positive_ic_ratio=None,
        after_cost_top_bottom_spread=None,
    )
    report = evaluate_backtest_against_thresholds(missing)

    assert report.overall_verdict == VALIDATION_VERDICT_FAIL
    assert {
        "rank_ic_mean",
        "icir",
        "ic_t_stat",
        "after_cost_sharpe",
        "positive_ic_ratio",
        "after_cost_spread",
    }.issubset(set(report.failed_gates))


def test_warning_case_recommends_paper_trading_status() -> None:
    thresholds = FactorValidationThresholds(max_drawdown=0.10)
    report = evaluate_backtest_against_thresholds(_result(max_drawdown=0.20), thresholds)

    assert report.overall_verdict == VALIDATION_VERDICT_WARN
    assert report.recommended_status == FACTOR_STATUS_PAPER_TRADING
    assert report.warning_gates == ["max_drawdown"]


def test_point_in_time_gate_fails_when_required_evidence_is_missing() -> None:
    payload_metadata = {"validation_generated_at": "2026-04-27"}
    report = evaluate_backtest_against_thresholds(_result(metadata=payload_metadata))

    assert report.overall_verdict == VALIDATION_VERDICT_FAIL
    assert "point_in_time" in report.failed_gates
    assert report.point_in_time_snapshot["reason"] == "missing_point_in_time_evidence"


def test_propose_admission_decision_does_not_auto_approve_production_on_pass() -> None:
    report = evaluate_backtest_against_thresholds(_result())
    decision = propose_admission_decision(
        report,
        decided_at="2026-04-27",
        decided_by="system",
    )

    assert decision.decision == ADMISSION_DECISION_APPROVE_PAPER_TRADING
    assert decision.target_status == FACTOR_STATUS_PAPER_TRADING
    assert decision.target_status != FACTOR_STATUS_PRODUCTION
    assert "Production approval is manual" in decision.rationale


def test_propose_admission_decision_for_warn_and_fail() -> None:
    warn_report = evaluate_backtest_against_thresholds(
        _result(max_drawdown=0.20),
        FactorValidationThresholds(max_drawdown=0.10),
    )
    warn_decision = propose_admission_decision(warn_report, decided_at="2026-04-27")
    assert warn_decision.decision in {"needs_research", ADMISSION_DECISION_APPROVE_PAPER_TRADING}
    assert warn_decision.conditions == ["resolve_warning_gate:max_drawdown"]

    fail_report = evaluate_backtest_against_thresholds(_result(sample_days=1))
    fail_decision = propose_admission_decision(fail_report, decided_at="2026-04-27")
    assert fail_decision.decision == ADMISSION_DECISION_REJECT
    assert fail_decision.target_status == FACTOR_STATUS_REJECTED


def test_build_library_entry_from_decision_maps_paper_trading() -> None:
    report = evaluate_backtest_against_thresholds(_result())
    decision = propose_admission_decision(report, decided_at="2026-04-27", decided_by="system")

    entry = build_library_entry_from_decision(decision, owner="research", tags=["momentum", "cn"])

    assert entry.factor_id == decision.factor_id
    assert entry.status == FACTOR_STATUS_PAPER_TRADING
    assert entry.paper_trading_since == "2026-04-27"
    assert entry.validation_report_id == report.report_id
    assert entry.tags == ["cn", "momentum"]


def test_build_library_entry_rejects_implicit_production() -> None:
    bad_decision = FactorAdmissionDecision(
        decision_id=make_admission_decision_id(
            factor_id="factor-a",
            factor_version="v1",
            decision=ADMISSION_DECISION_APPROVE_PAPER_TRADING,
            decided_at="2026-04-27",
        ),
        factor_id="factor-a",
        factor_version="v1",
        validation_report_id="report-a",
        decision=ADMISSION_DECISION_APPROVE_PAPER_TRADING,
        target_status=FACTOR_STATUS_PRODUCTION,
        decided_at="2026-04-27",
        decided_by="system",
        rationale="bad production target",
    )

    with pytest.raises(ValueError, match="production"):
        build_library_entry_from_decision(bad_decision)


def test_production_library_filters_only_production_and_sorts() -> None:
    paper = FactorLibraryEntry(
        factor_id="factor-c",
        factor_version="v1",
        status=FACTOR_STATUS_PAPER_TRADING,
        admission_decision_id="decision-c",
        validation_report_id="report-c",
        paper_trading_since="2026-04-27",
    )
    prod_b = FactorLibraryEntry(
        factor_id="factor-b",
        factor_version="v1",
        status=FACTOR_STATUS_PRODUCTION,
        admission_decision_id="decision-b",
        validation_report_id="report-b",
        production_since="2026-04-27",
    )
    prod_a = FactorLibraryEntry(
        factor_id="factor-a",
        factor_version="v1",
        status=FACTOR_STATUS_PRODUCTION,
        admission_decision_id="decision-a",
        validation_report_id="report-a",
        production_since="2026-04-27",
    )

    library = build_production_factor_library([paper, prod_b, prod_a], generated_at="2026-04-27")

    assert [entry.factor_id for entry in library.entries] == ["factor-a", "factor-b"]


def test_production_library_rejects_duplicate_production_entries() -> None:
    prod = FactorLibraryEntry(
        factor_id="factor-a",
        factor_version="v1",
        status=FACTOR_STATUS_PRODUCTION,
        admission_decision_id="decision-a",
        validation_report_id="report-a",
        production_since="2026-04-27",
    )

    with pytest.raises(ValueError, match="Duplicate"):
        build_production_factor_library([prod, prod], generated_at="2026-04-27")
