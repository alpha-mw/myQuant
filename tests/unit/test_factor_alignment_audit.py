from __future__ import annotations

import importlib
import json

import pytest

import quant_investor.factors.alignment_audit as alignment_audit
from quant_investor.factors.alignment_audit import (
    ALIGNMENT_AUDIT_FAIL,
    ALIGNMENT_AUDIT_PASS,
    ALIGNMENT_AUDIT_WARN,
    ALIGNMENT_ISSUE_ALIGNMENT_GAP,
    ALIGNMENT_ISSUE_BLOCKER,
    ALIGNMENT_ISSUE_DERIVED_VWAP_MISSING,
    ALIGNMENT_ISSUE_PRICE_FIELD_MISSING,
    ALIGNMENT_ISSUE_RETURN_MATRIX_LOOKAHEAD,
    ALIGNMENT_ISSUE_SAME_DAY_EXECUTION,
    ALIGNMENT_POLICY_CUSTOM,
    ALIGNMENT_POLICY_SIGNAL_T_EXECUTE_T_PLUS_1,
    AlignmentAuditRecord,
    FactorBacktestAlignmentAuditConfig,
    FactorBacktestAlignmentIssue,
    audit_execution_return_matrix_alignment,
    audit_factor_backtest_alignment,
    expected_alignment_tuples,
    make_alignment_audit_config_id,
    make_alignment_issue_id,
    make_alignment_record_id,
    render_alignment_audit_markdown,
    validate_strictly_ascending_dates,
)
from quant_investor.factors.backtest import (
    EXECUTION_PRICE_CLOSE,
    EXECUTION_PRICE_OPEN,
    EXECUTION_PRICE_VWAP,
    SingleFactorBacktestRun,
    build_execution_return_matrix,
    run_single_factor_backtest,
)
from quant_investor.factors.matrix import (
    FIELD_AMOUNT,
    FIELD_CLOSE,
    FIELD_OPEN,
    FIELD_VOLUME,
    FIELD_VWAP,
    FactorMatrix,
    MatrixDataBundle,
    MatrixDataContract,
    compute_coverage,
    make_factor_matrix_id,
    make_matrix_bundle_id,
    make_matrix_contract_id,
)
from quant_investor.factors.schema import FactorBacktestConfig, make_backtest_config_id
from quant_investor.versioning import FACTOR_BACKTEST_ALIGNMENT_AUDIT_SCHEMA_VERSION


SYMBOLS = ["AAA", "BBB", "CCC"]
DATES = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]


def test_alignment_audit_contracts_are_split_and_reexported() -> None:
    alignment_audit_types = importlib.import_module(
        "quant_investor.factors.alignment_audit_types"
    )

    assert (
        alignment_audit.FactorBacktestAlignmentIssue
        is alignment_audit_types.FactorBacktestAlignmentIssue
    )
    assert (
        alignment_audit.FactorBacktestAlignmentAuditConfig
        is alignment_audit_types.FactorBacktestAlignmentAuditConfig
    )
    assert (
        alignment_audit.AlignmentAuditRecord
        is alignment_audit_types.AlignmentAuditRecord
    )
    assert (
        alignment_audit.FactorBacktestAlignmentAuditReport
        is alignment_audit_types.FactorBacktestAlignmentAuditReport
    )
    assert (
        alignment_audit.make_alignment_audit_config_id
        is alignment_audit_types.make_alignment_audit_config_id
    )
    assert (
        alignment_audit.make_alignment_issue_id
        is alignment_audit_types.make_alignment_issue_id
    )
    assert (
        alignment_audit.make_alignment_record_id
        is alignment_audit_types.make_alignment_record_id
    )
    assert (
        alignment_audit.make_alignment_audit_report_id
        is alignment_audit_types.make_alignment_audit_report_id
    )


def _contract(required_fields: list[str]) -> MatrixDataContract:
    return MatrixDataContract(
        contract_id=make_matrix_contract_id(
            universe="CN",
            benchmark="CSI300",
            symbols=SYMBOLS,
            dates=DATES,
        ),
        universe="CN",
        benchmark="CSI300",
        symbols=SYMBOLS,
        dates=DATES,
        required_fields=required_fields,
        field_sources={field_name: "fixture" for field_name in required_fields},
        point_in_time_flags={field_name: True for field_name in required_fields},
        metadata={"preserve_symbol_order": True},
    )


def _bundle(
    *,
    include_vwap: bool = False,
    include_amount_volume: bool = True,
    include_open: bool = True,
    include_close: bool = True,
) -> MatrixDataBundle:
    vwap = [
        [10.0, 11.0, 12.0, 13.0, 14.0],
        [20.0, 19.0, 18.0, 17.0, 16.0],
        [30.0, 30.0, 33.0, 33.0, 36.0],
    ]
    fields = {}
    if include_open:
        fields[FIELD_OPEN] = [
            [9.0, 10.0, 12.0, 15.0, 15.0],
            [18.0, 20.0, 18.0, 18.0, 17.0],
            [28.0, 30.0, 31.0, 31.0, 34.0],
        ]
    if include_close:
        fields[FIELD_CLOSE] = [
            [10.0, 12.0, 12.0, 13.0, 15.0],
            [20.0, 18.0, 18.0, 16.0, 16.0],
            [30.0, 31.0, 33.0, 34.0, 36.0],
        ]
    if include_amount_volume:
        volume = [[100.0 for _date in DATES] for _symbol in SYMBOLS]
        amount = [
            [price * volume[row_index][column_index] for column_index, price in enumerate(row)]
            for row_index, row in enumerate(vwap)
        ]
        fields[FIELD_AMOUNT] = amount
        fields[FIELD_VOLUME] = volume
    if include_vwap:
        fields[FIELD_VWAP] = vwap
    contract = _contract(required_fields=list(fields))
    return MatrixDataBundle(
        bundle_id=make_matrix_bundle_id(
            contract_id=contract.contract_id,
            field_names=fields,
        ),
        contract=contract,
        fields=fields,
        universe_mask=[[True for _date in DATES] for _symbol in SYMBOLS],
        tradability_mask=[[True for _date in DATES] for _symbol in SYMBOLS],
        metadata={"fixture": True},
    )


def _factor_matrix() -> FactorMatrix:
    values = [
        [3.0, 1.0, 3.0, 2.0, 1.0],
        [1.0, 3.0, 1.0, 2.0, 3.0],
        [2.0, 2.0, 2.0, 3.0, 2.0],
    ]
    coverage_ratio, missing_ratio = compute_coverage(values)
    return FactorMatrix(
        matrix_id=make_factor_matrix_id(
            expression="alignment_fixture",
            symbols=SYMBOLS,
            dates=DATES,
        ),
        factor_id="alignment-fixture",
        factor_version="v1",
        expression="alignment_fixture",
        symbols=SYMBOLS,
        dates=DATES,
        values=values,
        coverage_ratio=coverage_ratio,
        missing_ratio=missing_ratio,
        metadata={"expected_direction": 1.0},
    )


def _config(*, execution_price: str = EXECUTION_PRICE_VWAP) -> FactorBacktestConfig:
    config = FactorBacktestConfig(
        config_id="placeholder",
        universe="CN",
        benchmark="CSI300",
        start_date="2026-01-01",
        end_date="2026-01-03",
        rebalance_frequency="daily",
        delay_days=1,
        execution_price=execution_price,
        long_short=True,
        long_only=False,
        quantile_count=3,
        long_quantile=3,
        short_quantile=1,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        market_impact_bps=0.0,
        min_coverage_ratio=0.0,
    )
    config.config_id = make_backtest_config_id(config)
    return config


def _run() -> SingleFactorBacktestRun:
    return run_single_factor_backtest(_factor_matrix(), _bundle(), _config())


def test_validate_strictly_ascending_dates() -> None:
    validate_strictly_ascending_dates(DATES)

    with pytest.raises(ValueError, match="duplicates"):
        validate_strictly_ascending_dates(["2026-01-01", "2026-01-01"])
    with pytest.raises(ValueError, match="strictly ascending"):
        validate_strictly_ascending_dates(["2026-01-02", "2026-01-01"])
    with pytest.raises(ValueError, match="ISO dates"):
        validate_strictly_ascending_dates(["bad-date"])


def test_expected_alignment_tuples_delay_one() -> None:
    alignments = expected_alignment_tuples(
        DATES[:3],
        delay_days=1,
        holding_period_days=1,
    )

    assert alignments == [
        {
            "signal_date": "2026-01-01",
            "execution_start_date": "2026-01-02",
            "execution_end_date": "2026-01-03",
            "signal_index": 0,
            "execution_start_index": 1,
            "execution_end_index": 2,
            "delay_days": 1,
            "holding_period_days": 1,
            "execution_price": EXECUTION_PRICE_VWAP,
        }
    ]
    assert expected_alignment_tuples(DATES[:2], delay_days=1, holding_period_days=1) == []


def test_audit_config_and_dataclass_round_trips() -> None:
    config = FactorBacktestAlignmentAuditConfig(
        config_id="alignment-config-fixture",
        expected_policy=ALIGNMENT_POLICY_SIGNAL_T_EXECUTE_T_PLUS_1,
        expected_delay_days=1,
        expected_holding_period_days=1,
        execution_price=EXECUTION_PRICE_VWAP,
        metadata={"fixture": True},
    )
    config.config_id = make_alignment_audit_config_id(config)

    assert FactorBacktestAlignmentAuditConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()
    json.dumps(config.to_dict(), ensure_ascii=False, sort_keys=True)

    with pytest.raises(ValueError, match="expected_delay_days"):
        FactorBacktestAlignmentAuditConfig(
            config_id="bad-delay",
            expected_delay_days=0,
        )
    with pytest.raises(ValueError, match="expected_holding_period_days"):
        FactorBacktestAlignmentAuditConfig(
            config_id="bad-hold",
            expected_holding_period_days=0,
        )
    with pytest.raises(ValueError, match="allow_custom_policy"):
        FactorBacktestAlignmentAuditConfig(
            config_id="bad-custom",
            expected_policy=ALIGNMENT_POLICY_CUSTOM,
            allow_custom_policy=False,
        )

    issue = FactorBacktestAlignmentIssue(
        issue_id=make_alignment_issue_id(
            issue_code=ALIGNMENT_ISSUE_ALIGNMENT_GAP,
            signal_date="2026-01-01",
            execution_start_date="2026-01-02",
            execution_end_date="2026-01-03",
            message="fixture issue",
        ),
        issue_code=ALIGNMENT_ISSUE_ALIGNMENT_GAP,
        severity=ALIGNMENT_ISSUE_BLOCKER,
        message="fixture issue",
        signal_date="2026-01-01",
        execution_start_date="2026-01-02",
        execution_end_date="2026-01-03",
        metadata={"fixture": True},
    )
    assert FactorBacktestAlignmentIssue.from_dict(issue.to_dict()).to_dict() == issue.to_dict()

    record = AlignmentAuditRecord(
        record_id=make_alignment_record_id(
            signal_date="2026-01-01",
            execution_start_date="2026-01-02",
            execution_end_date="2026-01-03",
            execution_price=EXECUTION_PRICE_VWAP,
        ),
        signal_date="2026-01-01",
        execution_start_date="2026-01-02",
        execution_end_date="2026-01-03",
        signal_index=0,
        execution_start_index=1,
        execution_end_index=2,
        delay_days=1,
        holding_period_days=1,
        execution_price=EXECUTION_PRICE_VWAP,
        expected_return_source_index=1,
        observed_weight_source_index=0,
        passed=True,
        issue_codes=[],
    )
    assert AlignmentAuditRecord.from_dict(record.to_dict()).to_dict() == record.to_dict()


def test_positive_factor_backtest_alignment_audit() -> None:
    matrix = _factor_matrix()
    bundle = _bundle()
    config = _config()
    run = run_single_factor_backtest(matrix, bundle, config)

    report = audit_factor_backtest_alignment(
        factor_matrix=matrix,
        bundle=bundle,
        config=config,
        run=run,
        generated_at="2026-04-27T00:00:00",
    )

    assert report.schema_version == FACTOR_BACKTEST_ALIGNMENT_AUDIT_SCHEMA_VERSION
    assert report.verdict == ALIGNMENT_AUDIT_PASS
    assert report.issue_count == 0
    assert [(record.signal_index, record.execution_start_index, record.execution_end_index) for record in report.records] == [
        (0, 1, 2),
        (1, 2, 3),
        (2, 3, 4),
    ]
    assert all(record.expected_return_source_index == record.execution_start_index for record in report.records)
    assert all(record.observed_weight_source_index == record.signal_index for record in report.records)
    assert report.metadata["non_runtime_impact"] is True


def test_same_day_run_record_emits_blocker() -> None:
    matrix = _factor_matrix()
    bundle = _bundle()
    config = _config()
    run_payload = run_single_factor_backtest(matrix, bundle, config).to_dict()
    run_payload["daily_records"][0]["execution_start_date"] = run_payload["daily_records"][0]["signal_date"]
    malformed_run = SingleFactorBacktestRun.from_dict(run_payload)

    report = audit_factor_backtest_alignment(
        factor_matrix=matrix,
        bundle=bundle,
        config=config,
        run=malformed_run,
        generated_at="2026-04-27T00:00:00",
    )

    assert report.verdict == ALIGNMENT_AUDIT_FAIL
    assert any(issue.issue_code == ALIGNMENT_ISSUE_SAME_DAY_EXECUTION for issue in report.issues)


def test_execution_return_matrix_audit_detects_correct_shift_and_missing_fields() -> None:
    bundle = _bundle()
    correct_returns = build_execution_return_matrix(
        bundle,
        execution_price=EXECUTION_PRICE_VWAP,
        holding_period_days=1,
    )

    assert audit_execution_return_matrix_alignment(
        bundle=bundle,
        execution_return_matrix=correct_returns,
        execution_price=EXECUTION_PRICE_VWAP,
        holding_period_days=1,
    ) == []

    close_values = bundle.get_field(FIELD_CLOSE)
    shifted_previous = []
    for row in close_values:
        shifted_row = [None]
        for column_index in range(1, len(row)):
            shifted_row.append(float(row[column_index]) / float(row[column_index - 1]) - 1.0)
        shifted_previous.append(shifted_row)
    shifted_issues = audit_execution_return_matrix_alignment(
        bundle=bundle,
        execution_return_matrix=shifted_previous,
        execution_price=EXECUTION_PRICE_CLOSE,
        holding_period_days=1,
    )
    assert any(issue.issue_code == ALIGNMENT_ISSUE_RETURN_MATRIX_LOOKAHEAD for issue in shifted_issues)

    missing_vwap_bundle = _bundle(
        include_vwap=False,
        include_amount_volume=False,
        include_open=True,
        include_close=True,
    )
    placeholder_returns = [[None for _date in DATES] for _symbol in SYMBOLS]
    missing_vwap_issues = audit_execution_return_matrix_alignment(
        bundle=missing_vwap_bundle,
        execution_return_matrix=placeholder_returns,
        execution_price=EXECUTION_PRICE_VWAP,
        holding_period_days=1,
    )
    assert [issue.issue_code for issue in missing_vwap_issues] == [ALIGNMENT_ISSUE_DERIVED_VWAP_MISSING]

    missing_close_bundle = _bundle(
        include_vwap=False,
        include_amount_volume=True,
        include_open=True,
        include_close=False,
    )
    missing_close_issues = audit_execution_return_matrix_alignment(
        bundle=missing_close_bundle,
        execution_return_matrix=placeholder_returns,
        execution_price=EXECUTION_PRICE_CLOSE,
        holding_period_days=1,
    )
    assert [issue.issue_code for issue in missing_close_issues] == [ALIGNMENT_ISSUE_PRICE_FIELD_MISSING]

    missing_open_bundle = _bundle(
        include_vwap=False,
        include_amount_volume=True,
        include_open=False,
        include_close=True,
    )
    missing_open_issues = audit_execution_return_matrix_alignment(
        bundle=missing_open_bundle,
        execution_return_matrix=placeholder_returns,
        execution_price=EXECUTION_PRICE_OPEN,
        holding_period_days=1,
    )
    assert [issue.issue_code for issue in missing_open_issues] == [ALIGNMENT_ISSUE_PRICE_FIELD_MISSING]


def test_run_record_count_mismatch_is_warning() -> None:
    matrix = _factor_matrix()
    bundle = _bundle()
    config = _config()
    run_payload = run_single_factor_backtest(matrix, bundle, config).to_dict()
    run_payload["daily_records"] = run_payload["daily_records"][:-1]
    short_run = SingleFactorBacktestRun.from_dict(run_payload)

    report = audit_factor_backtest_alignment(
        factor_matrix=matrix,
        bundle=bundle,
        config=config,
        run=short_run,
        generated_at="2026-04-27T00:00:00",
    )

    assert report.verdict == ALIGNMENT_AUDIT_WARN
    assert any(issue.issue_code == ALIGNMENT_ISSUE_ALIGNMENT_GAP for issue in report.issues)


def test_markdown_renderer_contains_required_sections() -> None:
    report = audit_factor_backtest_alignment(
        factor_matrix=_factor_matrix(),
        bundle=_bundle(),
        config=_config(),
        run=_run(),
        generated_at="2026-04-27T00:00:00",
    )

    markdown = render_alignment_audit_markdown(report)

    assert "Verdict: `pass`" in markdown
    assert "## Alignment Records" in markdown
    assert "## Issues" in markdown
    assert "This alignment audit is offline-only" in markdown
    assert "| Signal | Execute start | Execute end |" in markdown


def test_audit_inputs_are_not_mutated() -> None:
    matrix = _factor_matrix()
    bundle = _bundle()
    config = _config()
    run = run_single_factor_backtest(matrix, bundle, config)
    before = (matrix.to_dict(), bundle.to_dict(), run.to_dict())

    audit_factor_backtest_alignment(
        factor_matrix=matrix,
        bundle=bundle,
        config=config,
        run=run,
        generated_at="2026-04-27T00:00:00",
        metadata={"nested": {"value": True}},
    )

    assert (matrix.to_dict(), bundle.to_dict(), run.to_dict()) == before
