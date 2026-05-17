from __future__ import annotations

import pytest

from quant_investor.factors.matrix import (
    MatrixDataBundle,
    MatrixDataContract,
    make_matrix_bundle_id,
    make_matrix_contract_id,
)
from quant_investor.factors.store import FactorTradabilityAuditStore
from quant_investor.factors.tradability import (
    EXECUTION_AUDIT_STATUS_FEASIBLE,
    FIELD_AMOUNT,
    FIELD_LISTING_DAYS,
    FIELD_VALID_PRICE,
    FIELD_VALID_VOLUME,
    FIELD_VOLUME,
    FIELD_VWAP,
    TRADE_DIRECTION_BUY,
    TRADABILITY_AUDIT_PASS,
    AShareTradabilityConfig,
    ExecutionTransitionAuditRecord,
    FactorExecutionFeasibilityReport,
    build_ashare_tradability_mask,
    build_tradability_audit_report,
    make_execution_feasibility_report_id,
    make_execution_transition_record_id,
    make_tradability_config_id,
    render_execution_feasibility_markdown,
    render_tradability_audit_markdown,
)


SYMBOLS = ["AAA"]
DATES = ["2026-01-01", "2026-01-02"]


def _config() -> AShareTradabilityConfig:
    config = AShareTradabilityConfig(config_id="placeholder")
    config.config_id = make_tradability_config_id(config)
    return config


def _bundle() -> MatrixDataBundle:
    fields = {
        FIELD_VWAP: [[10.0, 11.0]],
        FIELD_VOLUME: [[1000.0, 1000.0]],
        FIELD_AMOUNT: [[10000.0, 11000.0]],
        FIELD_LISTING_DAYS: [[100, 101]],
        FIELD_VALID_PRICE: [[True, True]],
        FIELD_VALID_VOLUME: [[True, True]],
    }
    contract = MatrixDataContract(
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
        optional_fields=list(fields),
        field_sources={field_name: "fixture" for field_name in fields},
        point_in_time_flags={field_name: True for field_name in fields},
        metadata={"preserve_symbol_order": True},
    )
    return MatrixDataBundle(
        bundle_id=make_matrix_bundle_id(contract_id=contract.contract_id, field_names=fields),
        contract=contract,
        fields=fields,
        universe_mask=[[True, True]],
        tradability_mask=[[True, True]],
        metadata={"fixture": True},
    )


def _mask():
    return build_ashare_tradability_mask(_bundle(), config=_config())


def _tradability_report():
    return build_tradability_audit_report(
        _mask(),
        generated_at="2026-04-27T00:00:00Z",
    )


def _execution_report():
    mask = _mask()
    record = ExecutionTransitionAuditRecord(
        record_id=make_execution_transition_record_id(
            symbol="AAA",
            signal_date="2026-01-01",
            execution_date="2026-01-02",
            target_weight=1.0,
        ),
        symbol="AAA",
        signal_date="2026-01-01",
        execution_date="2026-01-02",
        previous_weight=0.0,
        target_weight=1.0,
        trade_weight=1.0,
        trade_direction=TRADE_DIRECTION_BUY,
        can_buy=True,
        can_sell=True,
        can_trade=True,
        status=EXECUTION_AUDIT_STATUS_FEASIBLE,
        issue_codes=[],
    )
    return FactorExecutionFeasibilityReport(
        report_id=make_execution_feasibility_report_id(
            backtest_run_id="run-fixture",
            weight_matrix_id="weights-fixture",
            generated_at="2026-04-27T00:00:00Z",
        ),
        generated_at="2026-04-27T00:00:00Z",
        factor_matrix_id="matrix-fixture",
        backtest_run_id="run-fixture",
        weight_matrix_id="weights-fixture",
        mask_id=mask.mask_id,
        total_transitions=1,
        feasible_transitions=1,
        blocked_transitions=0,
        partially_feasible_transitions=0,
        blocked_buy_count=0,
        blocked_sell_count=0,
        blocked_symbols=[],
        issue_count=0,
        blocker_count=0,
        warning_count=0,
        info_count=0,
        transition_records=[record],
        issues=[],
        verdict=TRADABILITY_AUDIT_PASS,
    )


def test_append_and_read_tradability_mask(tmp_path) -> None:
    store = FactorTradabilityAuditStore(tmp_path / "tradability")
    mask = _mask()

    store.append_tradability_mask(mask)

    assert store.read_tradability_masks()[0].to_dict() == mask.to_dict()
    assert store.get_tradability_mask_ids() == {mask.mask_id}


def test_append_and_read_reports(tmp_path) -> None:
    store = FactorTradabilityAuditStore(tmp_path / "tradability")
    tradability_report = _tradability_report()
    execution_report = _execution_report()

    store.append_tradability_audit_report(tradability_report)
    store.append_execution_feasibility_report(execution_report)

    assert store.read_tradability_audit_reports()[0].to_dict() == tradability_report.to_dict()
    assert store.read_execution_feasibility_reports()[0].to_dict() == execution_report.to_dict()
    assert store.get_tradability_audit_report_ids() == {tradability_report.report_id}
    assert store.get_execution_feasibility_report_ids() == {execution_report.report_id}


def test_duplicate_ids_raise(tmp_path) -> None:
    store = FactorTradabilityAuditStore(tmp_path / "tradability")
    mask = _mask()
    tradability_report = _tradability_report()
    execution_report = _execution_report()

    store.append_tradability_mask(mask)
    store.append_tradability_audit_report(tradability_report)
    store.append_execution_feasibility_report(execution_report)

    with pytest.raises(ValueError, match="Duplicate mask_id"):
        store.append_tradability_mask(mask)
    with pytest.raises(ValueError, match="Duplicate report_id"):
        store.append_tradability_audit_report(tradability_report)
    with pytest.raises(ValueError, match="Duplicate report_id"):
        store.append_execution_feasibility_report(execution_report)


def test_malformed_json_raises_clear_value_error(tmp_path) -> None:
    store = FactorTradabilityAuditStore(tmp_path / "tradability")
    store.tradability_masks_path.parent.mkdir(parents=True, exist_ok=True)
    store.tradability_masks_path.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed JSON"):
        store.read_tradability_masks()


def test_save_and_load_markdowns_and_create_directories(tmp_path) -> None:
    root = tmp_path / "missing" / "tradability"
    store = FactorTradabilityAuditStore(root)
    tradability_markdown = render_tradability_audit_markdown(_tradability_report())
    execution_markdown = render_execution_feasibility_markdown(_execution_report())

    tradability_path = store.save_tradability_audit_markdown(tradability_markdown)
    execution_path = store.save_execution_feasibility_markdown(execution_markdown)

    assert tradability_path == store.tradability_audit_markdown_path
    assert execution_path == store.execution_feasibility_markdown_path
    assert store.load_tradability_audit_markdown() == tradability_markdown
    assert store.load_execution_feasibility_markdown() == execution_markdown
    assert root.exists()
