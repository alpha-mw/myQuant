from __future__ import annotations

import json
import math

import pytest

from quant_investor.data_quality_contract import (
    ISSUE_MISSING_REQUIRED_FIELD,
    ISSUE_SEVERITY_BLOCKER,
    DataQualityAssessment,
    DataQualityIssue,
    build_tradability_status,
    make_issue_id,
)
from quant_investor.risk_tensor import (
    EXECUTION_BLOCKED,
    EXECUTION_FEASIBLE,
    EXECUTION_PARTIALLY_FEASIBLE,
    RISK_ISSUE_ADV_CAP_EXCEEDED,
    RISK_ISSUE_DATA_QUARANTINE,
    RISK_ISSUE_LOW_LIQUIDITY,
    RISK_ISSUE_MAX_ORDER_VALUE_EXCEEDED,
    RISK_ISSUE_POSITION_TOO_LARGE,
    RISK_ISSUE_SECTOR_CONCENTRATION,
    RISK_ISSUE_TURNOVER_EXCEEDED,
    RISK_ISSUE_UNTRADABLE,
    RISK_SEVERITY_BLOCKER,
    RISK_SEVERITY_WARNING,
    ExecutionFeasibility,
    ExecutionFeasibilityReport,
    LiquidityProfile,
    PortfolioRiskTensor,
    RiskIssue,
    RiskTensorStore,
    StressScenarioResult,
    SymbolExposure,
    SymbolRiskTensor,
    bps_to_decimal_return,
    build_execution_feasibility,
    build_execution_feasibility_report,
    build_portfolio_risk_tensor,
    build_risk_guard_context_patch,
    build_symbol_risk_tensor,
    clamp_unit_interval,
    make_execution_report_id,
    make_portfolio_tensor_id,
    make_risk_issue_id,
    make_symbol_tensor_id,
    validate_finite_number,
    weighted_average,
)
from quant_investor.versioning import RISK_TENSOR_SCHEMA_VERSION


def _liquidity(**overrides: object) -> LiquidityProfile:
    payload = {
        "symbol": "000001.SZ",
        "market": "CN",
        "as_of": "2026-04-26",
        "adv": 1_000_000.0,
        "liquidity_score": 0.9,
        "max_order_value": 20_000.0,
        "max_participation_rate": 0.10,
        "estimated_spread_bps": 6.0,
        "estimated_market_impact_bps": 2.0,
    }
    payload.update(overrides)
    return LiquidityProfile(**payload)


def _exposure(symbol: str = "000001.SZ", **overrides: object) -> SymbolExposure:
    payload = {
        "symbol": symbol,
        "market": "CN",
        "as_of": "2026-04-26",
        "sector": "Technology",
        "industry": "Hardware",
        "beta": 1.1,
        "style_exposures": {"momentum": 1.0, "value": -0.2},
        "factor_exposures": {"quality": 0.5},
        "correlation_cluster": "cluster-a",
    }
    payload.update(overrides)
    return SymbolExposure(**payload)


def _symbol_tensor(
    symbol: str = "000001.SZ",
    *,
    target_weight: float = 0.10,
    current_weight: float = 0.0,
    liquidity: LiquidityProfile | None = None,
    exposure: SymbolExposure | None = None,
    is_tradable: bool = True,
    max_weight: float | None = None,
    portfolio_value: float | None = 100_000.0,
) -> SymbolRiskTensor:
    return build_symbol_risk_tensor(
        symbol=symbol,
        market="CN",
        as_of="2026-04-26",
        latest_trade_date="2026-04-25",
        target_weight=target_weight,
        current_weight=current_weight,
        tradability_status=type("Status", (), {"is_tradable": is_tradable, "reasons": []})(),
        exposure=exposure or _exposure(symbol=symbol),
        liquidity=liquidity or _liquidity(symbol=symbol),
        portfolio_value=portfolio_value,
        max_weight=max_weight,
        stress_shocks={"selloff": -0.10},
    )


def test_risk_tensor_dataclass_round_trips() -> None:
    issue = RiskIssue(
        issue_id=make_risk_issue_id(
            symbol="000001.SZ",
            market="CN",
            as_of="2026-04-26",
            issue_type=RISK_ISSUE_LOW_LIQUIDITY,
            message="low liquidity",
        ),
        symbol="000001.SZ",
        market="CN",
        as_of="2026-04-26",
        issue_type=RISK_ISSUE_LOW_LIQUIDITY,
        severity=RISK_SEVERITY_WARNING,
        message="low liquidity",
        value=0.10,
        limit=0.20,
    )
    exposure = _exposure()
    liquidity = _liquidity()
    execution = build_execution_feasibility(
        symbol="000001.SZ",
        market="CN",
        as_of="2026-04-26",
        target_weight=0.05,
        portfolio_value=100_000.0,
        liquidity=liquidity,
    )
    stress = StressScenarioResult(
        scenario_name="selloff",
        symbol="000001.SZ",
        market="CN",
        as_of="2026-04-26",
        shock_return=-0.20,
        position_weight=0.05,
        estimated_loss=0.01,
    )
    symbol_tensor = SymbolRiskTensor(
        tensor_id=make_symbol_tensor_id(
            symbol="000001.SZ",
            market="CN",
            as_of="2026-04-26",
            latest_trade_date="2026-04-25",
        ),
        symbol="000001.SZ",
        market="CN",
        as_of="2026-04-26",
        latest_trade_date="2026-04-25",
        target_weight=0.05,
        exposure=exposure,
        liquidity=liquidity,
        execution=execution,
        stress_results=[stress],
        issues=[issue],
        risk_score=0.20,
    )
    portfolio_tensor = build_portfolio_risk_tensor(
        symbol_tensors=[symbol_tensor],
        market="CN",
        as_of="2026-04-26",
    )
    report = build_execution_feasibility_report(
        symbol_tensors=[symbol_tensor],
        market="CN",
        as_of="2026-04-26",
    )

    assert RiskIssue.from_dict(issue.to_dict()).to_dict() == issue.to_dict()
    assert SymbolExposure.from_dict(exposure.to_dict()).to_dict() == exposure.to_dict()
    assert LiquidityProfile.from_dict(liquidity.to_dict()).to_dict() == liquidity.to_dict()
    assert ExecutionFeasibility.from_dict(execution.to_dict()).to_dict() == execution.to_dict()
    assert StressScenarioResult.from_dict(stress.to_dict()).to_dict() == stress.to_dict()
    assert SymbolRiskTensor.from_dict(symbol_tensor.to_dict()).to_dict() == symbol_tensor.to_dict()
    assert PortfolioRiskTensor.from_dict(portfolio_tensor.to_dict()).to_dict() == portfolio_tensor.to_dict()
    assert ExecutionFeasibilityReport.from_dict(report.to_dict()).to_dict() == report.to_dict()

    with pytest.raises(ValueError, match="severity"):
        RiskIssue(severity="urgent")
    with pytest.raises(ValueError, match="style_exposures"):
        SymbolExposure(style_exposures={"bad": math.inf})


def test_deterministic_ids_are_stable() -> None:
    issue_args = {
        "symbol": "000001.SZ",
        "market": "CN",
        "as_of": "2026-04-26",
        "issue_type": RISK_ISSUE_UNTRADABLE,
        "message": "blocked",
    }
    symbol_args = {
        "symbol": "000001.SZ",
        "market": "CN",
        "as_of": "2026-04-26",
        "latest_trade_date": "2026-04-25",
    }
    tensor_ids = ["b", "a", "c"]
    portfolio_args = {"market": "CN", "as_of": "2026-04-26", "symbol_tensor_ids": tensor_ids}

    assert make_risk_issue_id(**issue_args) == make_risk_issue_id(**issue_args)
    assert make_symbol_tensor_id(**symbol_args) == make_symbol_tensor_id(**symbol_args)
    assert make_portfolio_tensor_id(**portfolio_args) == make_portfolio_tensor_id(**portfolio_args)
    assert make_execution_report_id(**portfolio_args) == make_execution_report_id(**portfolio_args)
    assert make_portfolio_tensor_id(**portfolio_args) == make_portfolio_tensor_id(
        market="CN",
        as_of="2026-04-26",
        symbol_tensor_ids=list(reversed(tensor_ids)),
    )


def test_numeric_helpers_validate_and_compute() -> None:
    assert clamp_unit_interval(-1.0) == 0.0
    assert clamp_unit_interval(1.5) == 1.0
    assert bps_to_decimal_return(100.0) == pytest.approx(0.01)
    validate_finite_number(1.0, field_name="field")

    with pytest.raises(ValueError, match="field"):
        validate_finite_number(float("nan"), field_name="field")
    with pytest.raises(ValueError, match="field"):
        validate_finite_number(float("inf"), field_name="field")

    assert weighted_average({"a": 1.0, "b": 3.0}, {"a": 1.0, "b": 1.0}) == pytest.approx(2.0)
    assert weighted_average({}, {}) is None
    assert weighted_average({"a": 1.0}, {"a": 0.0}) is None


def test_execution_feasibility_statuses_and_costs() -> None:
    clean = build_execution_feasibility(
        symbol="000001.SZ",
        market="CN",
        as_of="2026-04-26",
        target_weight=0.05,
        portfolio_value=100_000.0,
        liquidity=_liquidity(),
    )
    untradable = build_execution_feasibility(
        symbol="000001.SZ",
        market="CN",
        as_of="2026-04-26",
        target_weight=0.05,
        portfolio_value=100_000.0,
        liquidity=_liquidity(),
        is_tradable=False,
        tradability_reasons=["suspended"],
    )
    too_large = build_execution_feasibility(
        symbol="000001.SZ",
        market="CN",
        as_of="2026-04-26",
        target_weight=0.30,
        portfolio_value=100_000.0,
        liquidity=_liquidity(),
        max_weight=0.20,
    )
    partial = build_execution_feasibility(
        symbol="000001.SZ",
        market="CN",
        as_of="2026-04-26",
        target_weight=0.30,
        portfolio_value=100_000.0,
        liquidity=_liquidity(max_order_value=10_000.0, adv=None, max_participation_rate=None),
    )
    zero_allowed = build_execution_feasibility(
        symbol="000001.SZ",
        market="CN",
        as_of="2026-04-26",
        target_weight=0.01,
        portfolio_value=100_000.0,
        liquidity=_liquidity(max_order_value=0.0, adv=None, max_participation_rate=None),
    )
    adv_capped = build_execution_feasibility(
        symbol="000001.SZ",
        market="CN",
        as_of="2026-04-26",
        target_weight=0.03,
        portfolio_value=100_000.0,
        liquidity=_liquidity(adv=10_000.0, max_order_value=20_000.0, max_participation_rate=0.10),
    )

    assert clean.status == EXECUTION_FEASIBLE
    assert clean.requested_trade_value == pytest.approx(5_000.0)
    assert clean.allowed_trade_value == pytest.approx(20_000.0)
    assert clean.estimated_slippage_bps == pytest.approx(3.0)
    assert clean.estimated_market_impact_bps == pytest.approx(2.0)
    assert clean.estimated_transaction_cost_bps == pytest.approx(5.0)
    assert untradable.status == EXECUTION_BLOCKED
    assert RISK_ISSUE_UNTRADABLE in untradable.blocked_reasons
    assert too_large.status == EXECUTION_BLOCKED
    assert RISK_ISSUE_POSITION_TOO_LARGE in too_large.blocked_reasons
    assert partial.status == EXECUTION_PARTIALLY_FEASIBLE
    assert RISK_ISSUE_MAX_ORDER_VALUE_EXCEEDED in partial.warning_reasons
    assert zero_allowed.status == EXECUTION_BLOCKED
    assert RISK_ISSUE_MAX_ORDER_VALUE_EXCEEDED in zero_allowed.blocked_reasons
    assert adv_capped.status == EXECUTION_PARTIALLY_FEASIBLE
    assert RISK_ISSUE_ADV_CAP_EXCEEDED in adv_capped.warning_reasons


def test_symbol_risk_tensor_builder_handles_quality_tradability_liquidity_and_stress() -> None:
    clean = _symbol_tensor(target_weight=0.02)
    low_liquidity = _symbol_tensor(
        symbol="000002.SZ",
        liquidity=_liquidity(symbol="000002.SZ", liquidity_score=0.10),
    )
    untradable = _symbol_tensor(symbol="000003.SZ", is_tradable=False)
    issue = DataQualityIssue(
        issue_id=make_issue_id(
            symbol="000004.SZ",
            market="CN",
            as_of="2026-04-26",
            issue_type=ISSUE_MISSING_REQUIRED_FIELD,
            field_name="close",
        ),
        symbol="000004.SZ",
        market="CN",
        as_of="2026-04-26",
        field_name="close",
        issue_type=ISSUE_MISSING_REQUIRED_FIELD,
        severity=ISSUE_SEVERITY_BLOCKER,
        message="missing close",
    )
    assessment = DataQualityAssessment(
        assessment_id="dq-assessment",
        snapshot_id="snapshot",
        symbol="000004.SZ",
        market="CN",
        as_of="2026-04-26",
        quarantine=True,
        quarantine_reasons=["missing close"],
        blocker_count=1,
        issue_count=1,
        data_quality_score=0.0,
        issues=[issue],
    )
    quarantined = build_symbol_risk_tensor(
        symbol="000004.SZ",
        market="CN",
        as_of="2026-04-26",
        latest_trade_date="2026-04-25",
        target_weight=0.10,
        data_quality_assessment=assessment,
        liquidity=_liquidity(symbol="000004.SZ"),
        portfolio_value=100_000.0,
        stress_shocks={"crash": -0.25},
    )

    exposure = _exposure(symbol="000005.SZ")
    exposure_before = exposure.to_dict()
    liquidity = _liquidity(symbol="000005.SZ")
    liquidity_before = liquidity.to_dict()
    build_symbol_risk_tensor(
        symbol="000005.SZ",
        market="CN",
        as_of="2026-04-26",
        latest_trade_date="2026-04-25",
        target_weight=0.10,
        exposure=exposure,
        liquidity=liquidity,
        portfolio_value=100_000.0,
    )

    assert clean.risk_score < 0.10
    assert any(result.estimated_loss > 0 for result in clean.stress_results)
    assert any(issue.issue_type == RISK_ISSUE_LOW_LIQUIDITY for issue in low_liquidity.issues)
    assert any(issue.issue_type == RISK_ISSUE_UNTRADABLE for issue in untradable.issues)
    assert any(issue.issue_type == RISK_ISSUE_DATA_QUARANTINE for issue in quarantined.issues)
    assert quarantined.quarantine is True
    assert quarantined.is_researchable is False
    assert quarantined.data_quality_score == 0.0
    assert exposure.to_dict() == exposure_before
    assert liquidity.to_dict() == liquidity_before


def test_portfolio_risk_tensor_computes_exposures_issues_and_caps() -> None:
    first = _symbol_tensor(
        symbol="000001.SZ",
        target_weight=0.30,
        exposure=_exposure(
            symbol="000001.SZ",
            sector="Technology",
            style_exposures={"momentum": 1.0},
            factor_exposures={"quality": 0.5},
            beta=1.0,
            correlation_cluster="cluster-a",
        ),
    )
    second = _symbol_tensor(
        symbol="000002.SZ",
        target_weight=0.25,
        exposure=_exposure(
            symbol="000002.SZ",
            sector="Technology",
            style_exposures={"momentum": 0.5},
            factor_exposures={"quality": 1.0},
            beta=1.2,
            correlation_cluster="cluster-a",
        ),
    )
    blocked = _symbol_tensor(
        symbol="000003.SZ",
        target_weight=0.10,
        is_tradable=False,
        exposure=_exposure(
            symbol="000003.SZ",
            sector=None,
            style_exposures={},
            factor_exposures={},
            correlation_cluster=None,
        ),
    )
    portfolio = build_portfolio_risk_tensor(
        symbol_tensors=[second, blocked, first],
        market="CN",
        as_of="2026-04-26",
        turnover_estimate=0.30,
        sector_cap=0.40,
        gross_exposure_cap=0.50,
        max_weight=0.20,
        turnover_cap=0.20,
    )
    issue_types = [issue.issue_type for issue in portfolio.portfolio_issues]

    assert [tensor.symbol for tensor in portfolio.symbol_tensors] == ["000001.SZ", "000002.SZ", "000003.SZ"]
    assert portfolio.gross_exposure == pytest.approx(0.65)
    assert portfolio.net_exposure == pytest.approx(0.65)
    assert portfolio.long_exposure == pytest.approx(0.65)
    assert portfolio.short_exposure == pytest.approx(0.0)
    assert portfolio.sector_weights["Technology"] == pytest.approx(0.55)
    assert portfolio.sector_weights["UNKNOWN"] == pytest.approx(0.10)
    assert portfolio.style_exposures["momentum"] == pytest.approx(0.425)
    assert portfolio.factor_exposures["quality"] == pytest.approx(0.40)
    assert "000003.SZ" in portfolio.blocked_symbols
    assert portfolio.max_weight_by_symbol["000003.SZ"] == 0.0
    assert RISK_ISSUE_POSITION_TOO_LARGE in issue_types
    assert RISK_ISSUE_SECTOR_CONCENTRATION in issue_types
    assert RISK_ISSUE_TURNOVER_EXCEEDED in issue_types
    assert 0.0 <= portfolio.risk_score <= 1.0


def test_execution_feasibility_report_groups_and_aggregates_deterministically() -> None:
    feasible = _symbol_tensor(symbol="000001.SZ", target_weight=0.02)
    partial = _symbol_tensor(
        symbol="000002.SZ",
        target_weight=0.30,
        liquidity=_liquidity(symbol="000002.SZ", max_order_value=10_000.0, adv=None),
    )
    blocked = _symbol_tensor(symbol="000003.SZ", target_weight=0.05, is_tradable=False)
    report = build_execution_feasibility_report(
        symbol_tensors=[blocked, partial, feasible],
        market="CN",
        as_of="2026-04-26",
    )

    assert report.feasible_symbols == ["000001.SZ"]
    assert report.partially_feasible_symbols == ["000002.SZ"]
    assert report.blocked_symbols == ["000003.SZ"]
    assert list(report.execution_by_symbol) == ["000001.SZ", "000002.SZ", "000003.SZ"]
    assert report.total_requested_trade_value == pytest.approx(37_000.0)
    assert report.total_allowed_trade_value == pytest.approx(50_000.0)
    assert report.aggregate_estimated_cost_bps == pytest.approx(5.0)
    assert any(issue.severity == RISK_SEVERITY_BLOCKER for issue in report.issues)


def test_risk_guard_context_patch_is_json_serializable_and_offline_only() -> None:
    blocked = _symbol_tensor(symbol="000003.SZ", is_tradable=False)
    feasible = _symbol_tensor(symbol="000001.SZ")
    portfolio = build_portfolio_risk_tensor(
        symbol_tensors=[blocked, feasible],
        market="CN",
        as_of="2026-04-26",
        max_weight=0.20,
    )
    report = build_execution_feasibility_report(
        symbol_tensors=[blocked, feasible],
        market="CN",
        as_of="2026-04-26",
    )
    patch = build_risk_guard_context_patch(portfolio, report)

    json.dumps(patch, ensure_ascii=False)
    assert patch["blocked_symbols"] == ["000003.SZ"]
    assert patch["max_weight_by_symbol"]["000003.SZ"] == 0.0
    assert patch["execution_status_by_symbol"]["000003.SZ"] == EXECUTION_BLOCKED
    assert patch["metadata"]["risk_tensor_schema_version"] == RISK_TENSOR_SCHEMA_VERSION
    assert patch["metadata"]["execution_report_id"] == report.report_id


def test_risk_tensor_store_round_trips_and_rejects_duplicates_and_bad_json(tmp_path) -> None:
    symbol_tensor = _symbol_tensor()
    portfolio_tensor = build_portfolio_risk_tensor(
        symbol_tensors=[symbol_tensor],
        market="CN",
        as_of="2026-04-26",
    )
    report = build_execution_feasibility_report(
        symbol_tensors=[symbol_tensor],
        market="CN",
        as_of="2026-04-26",
    )
    store = RiskTensorStore(tmp_path)

    store.append_symbol_tensor(symbol_tensor)
    assert store.read_symbol_tensors()[0].tensor_id == symbol_tensor.tensor_id
    with pytest.raises(ValueError, match="Duplicate tensor_id"):
        store.append_symbol_tensor(symbol_tensor)
    assert store.append_symbol_tensors([]) == 0

    store.append_portfolio_tensor(portfolio_tensor)
    assert store.read_portfolio_tensors()[0].tensor_id == portfolio_tensor.tensor_id
    with pytest.raises(ValueError, match="Duplicate tensor_id"):
        store.append_portfolio_tensor(portfolio_tensor)

    store.append_execution_report(report)
    assert store.read_execution_reports()[0].report_id == report.report_id
    with pytest.raises(ValueError, match="Duplicate report_id"):
        store.append_execution_report(report)

    bad_store = RiskTensorStore(tmp_path / "bad")
    bad_store.symbol_tensors_path.parent.mkdir(parents=True, exist_ok=True)
    bad_store.symbol_tensors_path.write_text("{bad json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed JSON"):
        bad_store.read_symbol_tensors()


def test_phase5_quality_and_tradability_objects_bridge_by_duck_typing() -> None:
    tradability = build_tradability_status(
        symbol="000001.SZ",
        market="CN",
        as_of="2026-04-26",
        latest_trade_date="2026-04-25",
        is_suspended=True,
        liquidity_score=0.10,
        adv=50_000.0,
        max_order_value=1_000.0,
    )
    assessment = DataQualityAssessment(
        assessment_id="dq-assessment",
        snapshot_id="snapshot",
        symbol="000001.SZ",
        market="CN",
        as_of="2026-04-26",
        quarantine=True,
        quarantine_reasons=["lookahead"],
        blocker_count=1,
        issue_count=1,
        data_quality_score=0.0,
        tradability_reasons=list(tradability.reasons),
    )
    tensor = build_symbol_risk_tensor(
        symbol="000001.SZ",
        market="CN",
        as_of="2026-04-26",
        latest_trade_date="2026-04-25",
        target_weight=0.05,
        data_quality_assessment=assessment,
        tradability_status=tradability,
        portfolio_value=100_000.0,
    )

    issue_types = [issue.issue_type for issue in tensor.issues]
    assert tensor.quarantine is True
    assert tensor.is_researchable is False
    assert tensor.is_tradable is False
    assert tensor.liquidity.adv == pytest.approx(50_000.0)
    assert tensor.liquidity.liquidity_score == pytest.approx(0.10)
    assert tensor.liquidity.max_order_value == pytest.approx(1_000.0)
    assert tensor.execution.status == EXECUTION_BLOCKED
    assert RISK_ISSUE_DATA_QUARANTINE in issue_types
    assert RISK_ISSUE_UNTRADABLE in issue_types
    assert RISK_ISSUE_LOW_LIQUIDITY in issue_types
    assert RISK_ISSUE_MAX_ORDER_VALUE_EXCEEDED in tensor.execution.warning_reasons
