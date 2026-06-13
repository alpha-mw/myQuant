from __future__ import annotations

import pytest

from quant_investor.factors import schema as schema_module
from quant_investor.factors import schema_primitives
from quant_investor.factors.schema import (
    ADMISSION_DECISION_APPROVE_PAPER_TRADING,
    ADMISSION_DECISION_APPROVE_PRODUCTION,
    FACTOR_FAMILY_MOMENTUM,
    FACTOR_FAMILY_PRICE,
    FACTOR_STATUS_PAPER_TRADING,
    FACTOR_STATUS_PRODUCTION,
    FACTOR_STATUS_REJECTED,
    FACTOR_STATUS_RESEARCH_CANDIDATE,
    VALIDATION_VERDICT_PASS,
    FactorAdmissionDecision,
    FactorBacktestConfig,
    FactorBacktestResult,
    FactorDefinition,
    FactorLibraryEntry,
    FactorValidationReport,
    FactorValidationThresholds,
    ProductionFactorLibrary,
    make_admission_decision_id,
    make_backtest_config_id,
    make_backtest_result_id,
    make_factor_id,
    make_production_library_id,
    make_validation_report_id,
)


def test_factor_schema_primitives_are_split_and_reexported() -> None:
    assert schema_module.FACTOR_STATUS_DRAFT is schema_primitives.FACTOR_STATUS_DRAFT
    assert schema_module.FACTOR_FAMILY_CUSTOM is schema_primitives.FACTOR_FAMILY_CUSTOM
    assert schema_module.SUPPORTED_FACTOR_STATUSES is schema_primitives.SUPPORTED_FACTOR_STATUSES
    assert schema_module.DEFAULT_FACTOR_LIBRARY_DIR == schema_primitives.DEFAULT_FACTOR_LIBRARY_DIR
    assert schema_module._json_safe is schema_primitives.json_safe
    assert schema_module._finite_float is schema_primitives.finite_float
    assert schema_module._short_hash is schema_primitives.short_hash


def _definition() -> FactorDefinition:
    factor_id = make_factor_id(
        factor_name="Sixty Day Momentum",
        factor_family=FACTOR_FAMILY_MOMENTUM,
        expression="close / delay(close, 60) - 1",
    )
    return FactorDefinition(
        factor_id=factor_id,
        factor_name="Sixty Day Momentum",
        factor_family=FACTOR_FAMILY_MOMENTUM,
        status=FACTOR_STATUS_RESEARCH_CANDIDATE,
        version="v1",
        expression="close / delay(close, 60) - 1",
        input_fields=["close", "close", "trade_date"],
        data_sources=["local_csv", "local_csv", "vendor_snapshot"],
        universe="CN",
        benchmark="CSI300",
        expected_direction=1.0,
        rebalance_frequency="weekly",
        lookback_window=60,
        delay_days=1,
        execution_price="next_open",
        winsorization_rule="mad_3",
        standardization_rule="zscore",
        neutralization_rule="industry_size",
        missing_value_rule="drop",
        point_in_time_required=True,
        st_filter=True,
        suspension_filter=True,
        limit_up_down_filter=True,
        new_listing_min_days=120,
        adjustment_rule="forward_adjusted",
        industry_neutral=True,
        size_neutral=True,
        economic_rationale="Momentum captures persistent medium-horizon trend behavior.",
        owner="research",
        created_at="2026-04-27",
        metadata={"fixture": True},
    )


def _backtest_config() -> FactorBacktestConfig:
    config = FactorBacktestConfig(
        config_id="placeholder",
        universe="CN",
        benchmark="CSI300",
        start_date="2021-01-01",
        end_date="2025-12-31",
        rebalance_frequency="weekly",
        delay_days=1,
        execution_price="next_open",
        long_short=True,
        long_only=False,
        quantile_count=5,
        long_quantile=5,
        short_quantile=1,
        transaction_cost_bps=5.0,
        slippage_bps=3.0,
        market_impact_bps=2.0,
        max_participation_rate=0.10,
        min_coverage_ratio=0.80,
        neutralize_industry=True,
        neutralize_size=True,
    )
    config.config_id = make_backtest_config_id(config)
    return config


def _backtest_result() -> FactorBacktestResult:
    config = _backtest_config()
    result_id = make_backtest_result_id(
        factor_id="factor-momentum-test",
        factor_version="v1",
        config_id=config.config_id,
    )
    return FactorBacktestResult(
        result_id=result_id,
        factor_id="factor-momentum-test",
        factor_version="v1",
        config_id=config.config_id,
        start_date="2021-01-01",
        end_date="2025-12-31",
        sample_days=1000,
        coverage_ratio=0.92,
        missing_ratio=0.08,
        ann_ret=0.12,
        ann_vol=0.16,
        sharpe=1.20,
        max_drawdown=0.15,
        turnover_avg=0.25,
        long_num_avg=100.0,
        short_num_avg=100.0,
        rank_ic_mean=0.035,
        ic_mean=0.030,
        icir=0.50,
        ic_t_stat=4.0,
        positive_ic_ratio=0.60,
        top_bottom_spread=0.08,
        after_cost_top_bottom_spread=0.06,
        before_cost_sharpe=1.10,
        after_cost_sharpe=0.95,
        monotonicity_score=1.0,
        capacity_estimate=1000000.0,
        slice_metrics={"2024": {"rank_ic_mean": 0.03}},
        metadata={"point_in_time_passed": True},
    )


def _validation_report() -> FactorValidationReport:
    result = _backtest_result()
    return FactorValidationReport(
        report_id=make_validation_report_id(
            factor_id=result.factor_id,
            factor_version=result.factor_version,
            backtest_result_id=result.result_id,
        ),
        factor_id=result.factor_id,
        factor_version=result.factor_version,
        generated_at="2026-04-27",
        backtest_result_id=result.result_id,
        thresholds=FactorValidationThresholds(),
        overall_verdict=VALIDATION_VERDICT_PASS,
        gate_results={"sample_days": VALIDATION_VERDICT_PASS},
        failed_gates=[],
        warning_gates=[],
        metric_snapshot={"rank_ic_mean": result.rank_ic_mean},
        recommended_status=FACTOR_STATUS_RESEARCH_CANDIDATE,
        rationale="schema fixture report",
    )


def _paper_decision() -> FactorAdmissionDecision:
    report = _validation_report()
    return FactorAdmissionDecision(
        decision_id=make_admission_decision_id(
            factor_id=report.factor_id,
            factor_version=report.factor_version,
            decision=ADMISSION_DECISION_APPROVE_PAPER_TRADING,
            decided_at="2026-04-27",
        ),
        factor_id=report.factor_id,
        factor_version=report.factor_version,
        validation_report_id=report.report_id,
        decision=ADMISSION_DECISION_APPROVE_PAPER_TRADING,
        target_status=FACTOR_STATUS_PAPER_TRADING,
        decided_at="2026-04-27",
        decided_by="research",
        rationale="Approve paper trading for schema fixture.",
        conditions=["b", "a", "a"],
    )


def test_factor_definition_round_trip_and_deterministic_lists() -> None:
    definition = _definition()
    round_trip = FactorDefinition.from_dict(definition.to_dict())

    assert round_trip.to_dict() == definition.to_dict()
    assert round_trip.input_fields == ["close", "trade_date"]
    assert round_trip.data_sources == ["local_csv", "vendor_snapshot"]


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("status", "live", "status"),
        ("factor_family", "alchemy", "factor_family"),
        ("expected_direction", 0.0, "expected_direction"),
        ("delay_days", 0, "delay_days"),
    ],
)
def test_factor_definition_rejects_invalid_contract_values(
    field_name: str,
    value: object,
    match: str,
) -> None:
    payload = _definition().to_dict()
    payload[field_name] = value

    with pytest.raises(ValueError, match=match):
        FactorDefinition.from_dict(payload)


def test_factor_definition_rejects_missing_required_text_and_negative_listing_days() -> None:
    payload = _definition().to_dict()
    payload["factor_name"] = ""
    with pytest.raises(ValueError, match="factor_name"):
        FactorDefinition.from_dict(payload)

    payload = _definition().to_dict()
    payload["new_listing_min_days"] = -1
    with pytest.raises(ValueError, match="new_listing_min_days"):
        FactorDefinition.from_dict(payload)


def test_backtest_config_validates_dates_quantiles_and_costs() -> None:
    assert FactorBacktestConfig.from_dict(_backtest_config().to_dict()).to_dict() == _backtest_config().to_dict()

    bad_date = _backtest_config().to_dict()
    bad_date["start_date"] = "2026-01-01"
    bad_date["end_date"] = "2025-01-01"
    with pytest.raises(ValueError, match="start_date"):
        FactorBacktestConfig.from_dict(bad_date)

    bad_quantiles = _backtest_config().to_dict()
    bad_quantiles["long_quantile"] = 6
    with pytest.raises(ValueError, match="long_quantile"):
        FactorBacktestConfig.from_dict(bad_quantiles)

    bad_cost = _backtest_config().to_dict()
    bad_cost["transaction_cost_bps"] = -1.0
    with pytest.raises(ValueError, match="transaction_cost_bps"):
        FactorBacktestConfig.from_dict(bad_cost)


def test_validation_threshold_defaults_validate_and_reject_invalid_values() -> None:
    thresholds = FactorValidationThresholds()
    assert thresholds.min_sample_days == 750
    assert FactorValidationThresholds.from_dict(thresholds.to_dict()).to_dict() == thresholds.to_dict()

    with pytest.raises(ValueError, match="min_coverage_ratio"):
        FactorValidationThresholds(min_coverage_ratio=1.5)
    with pytest.raises(ValueError, match="production_revalidation_days"):
        FactorValidationThresholds(production_revalidation_days=0)


def test_backtest_result_rejects_invalid_ratios_and_non_finite_metrics() -> None:
    payload = _backtest_result().to_dict()
    payload["coverage_ratio"] = 1.2
    with pytest.raises(ValueError, match="coverage_ratio"):
        FactorBacktestResult.from_dict(payload)

    payload = _backtest_result().to_dict()
    payload["missing_ratio"] = -0.1
    with pytest.raises(ValueError, match="missing_ratio"):
        FactorBacktestResult.from_dict(payload)

    payload = _backtest_result().to_dict()
    payload["sharpe"] = float("inf")
    with pytest.raises(ValueError, match="sharpe"):
        FactorBacktestResult.from_dict(payload)


def test_validation_report_validates_verdict_status_and_rationale() -> None:
    report = _validation_report()
    assert FactorValidationReport.from_dict(report.to_dict()).to_dict() == report.to_dict()

    payload = report.to_dict()
    payload["overall_verdict"] = "maybe"
    with pytest.raises(ValueError, match="overall_verdict"):
        FactorValidationReport.from_dict(payload)

    payload = report.to_dict()
    payload["recommended_status"] = "unknown"
    with pytest.raises(ValueError, match="recommended_status"):
        FactorValidationReport.from_dict(payload)

    payload = report.to_dict()
    payload["rationale"] = ""
    with pytest.raises(ValueError, match="rationale"):
        FactorValidationReport.from_dict(payload)


def test_admission_decision_requires_validation_report_for_production() -> None:
    with pytest.raises(ValueError, match="approve_production"):
        FactorAdmissionDecision(
            decision_id="decision-prod",
            factor_id="factor-a",
            factor_version="v1",
            validation_report_id=None,
            decision=ADMISSION_DECISION_APPROVE_PRODUCTION,
            target_status=FACTOR_STATUS_PRODUCTION,
            decided_at="2026-04-27",
            decided_by="ic",
            rationale="manual production approval",
        )


def test_admission_decision_round_trip_and_condition_sorting() -> None:
    decision = _paper_decision()
    round_trip = FactorAdmissionDecision.from_dict(decision.to_dict())

    assert round_trip.to_dict() == decision.to_dict()
    assert round_trip.conditions == ["a", "b"]


def test_library_entry_production_requires_dates_and_validation() -> None:
    decision = _paper_decision()
    entry = FactorLibraryEntry(
        factor_id=decision.factor_id,
        factor_version=decision.factor_version,
        status=FACTOR_STATUS_PAPER_TRADING,
        admission_decision_id=decision.decision_id,
        validation_report_id=decision.validation_report_id,
        paper_trading_since="2026-04-27",
        tags=["momentum", "momentum", "cn"],
    )
    assert FactorLibraryEntry.from_dict(entry.to_dict()).tags == ["cn", "momentum"]

    with pytest.raises(ValueError, match="production_since"):
        FactorLibraryEntry(
            factor_id="factor-a",
            factor_version="v1",
            status=FACTOR_STATUS_PRODUCTION,
            admission_decision_id="decision-a",
            validation_report_id="report-a",
        )
    with pytest.raises(ValueError, match="validation_report_id"):
        FactorLibraryEntry(
            factor_id="factor-a",
            factor_version="v1",
            status=FACTOR_STATUS_PRODUCTION,
            admission_decision_id="decision-a",
            production_since="2026-04-27",
        )


def test_production_factor_library_rejects_non_production_and_duplicates() -> None:
    prod = FactorLibraryEntry(
        factor_id="factor-b",
        factor_version="v1",
        status=FACTOR_STATUS_PRODUCTION,
        admission_decision_id="decision-b",
        validation_report_id="report-b",
        production_since="2026-04-27",
    )
    prod_earlier = FactorLibraryEntry(
        factor_id="factor-a",
        factor_version="v1",
        status=FACTOR_STATUS_PRODUCTION,
        admission_decision_id="decision-a",
        validation_report_id="report-a",
        production_since="2026-04-27",
    )
    library = ProductionFactorLibrary(
        library_id=make_production_library_id([prod, prod_earlier]),
        generated_at="2026-04-27",
        entries=[prod, prod_earlier],
    )
    assert [entry.factor_id for entry in library.entries] == ["factor-a", "factor-b"]

    non_prod = FactorLibraryEntry(
        factor_id="factor-c",
        factor_version="v1",
        status=FACTOR_STATUS_REJECTED,
        admission_decision_id="decision-c",
        validation_report_id="report-c",
    )
    with pytest.raises(ValueError, match="production"):
        ProductionFactorLibrary(library_id="library", generated_at="2026-04-27", entries=[non_prod])
    with pytest.raises(ValueError, match="Duplicate"):
        ProductionFactorLibrary(library_id="library", generated_at="2026-04-27", entries=[prod, prod])


def test_deterministic_id_helpers_are_stable() -> None:
    definition = _definition()
    config = _backtest_config()
    result = _backtest_result()
    report = _validation_report()
    decision = _paper_decision()
    entry = FactorLibraryEntry(
        factor_id=definition.factor_id,
        factor_version=definition.version,
        status=FACTOR_STATUS_PRODUCTION,
        admission_decision_id="decision-prod",
        validation_report_id="report-prod",
        production_since="2026-04-27",
    )

    assert make_factor_id(
        factor_name=definition.factor_name,
        factor_family=FACTOR_FAMILY_MOMENTUM,
        expression=definition.expression,
    ) == definition.factor_id
    assert make_backtest_config_id(config) == make_backtest_config_id(config)
    assert make_backtest_result_id(
        factor_id=result.factor_id,
        factor_version=result.factor_version,
        config_id=result.config_id,
    ) == result.result_id
    assert make_validation_report_id(
        factor_id=report.factor_id,
        factor_version=report.factor_version,
        backtest_result_id=report.backtest_result_id,
    ) == report.report_id
    assert make_admission_decision_id(
        factor_id=decision.factor_id,
        factor_version=decision.factor_version,
        decision=decision.decision,
        decided_at=decision.decided_at,
    ) == decision.decision_id
    assert make_production_library_id([entry]) == make_production_library_id([entry])


def test_factor_family_price_is_supported() -> None:
    payload = _definition().to_dict()
    payload["factor_family"] = FACTOR_FAMILY_PRICE
    payload["factor_id"] = make_factor_id(
        factor_name=payload["factor_name"],
        factor_family=FACTOR_FAMILY_PRICE,
        expression=payload["expression"],
    )

    assert FactorDefinition.from_dict(payload).factor_family == FACTOR_FAMILY_PRICE
