from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest

from quant_investor.bayesian.posterior_overlay import CalibratedPosteriorOverlay
from quant_investor.portfolio_optimizer import (
    CONSTRAINT_BLOCKED_SYMBOL,
    CONSTRAINT_MIN_EDGE,
    CONSTRAINT_RISK_SCORE,
    CONSTRAINT_TURNOVER_CAP,
    PLAN_STATUS_OPTIMIZED,
    ConstraintViolation,
    OptimizationCandidate,
    OptimizedPortfolioPlan,
    PortfolioOptimizerConfig,
    PortfolioOptimizerStore,
    RebalanceInput,
    RebalanceResult,
    WalkForwardResult,
    bps_to_decimal_return,
    build_candidate_from_overlay,
    build_candidates_from_overlays,
    build_portfolio_constructor_patch,
    compound_returns,
    compute_sector_weights,
    estimate_turnover,
    evaluate_rebalance,
    make_constraint_violation_id,
    make_plan_id,
    make_rebalance_id,
    make_walk_forward_run_id,
    max_drawdown_from_returns,
    optimize_portfolio,
    run_walk_forward_loop,
    validate_finite_number,
)
from quant_investor.risk_tensor import (
    EXECUTION_BLOCKED,
    RISK_ISSUE_DATA_QUARANTINE,
    RISK_SEVERITY_BLOCKER,
    ExecutionFeasibility,
    LiquidityProfile,
    RiskIssue,
    SymbolExposure,
    SymbolRiskTensor,
)
from quant_investor.versioning import PORTFOLIO_OPTIMIZER_SCHEMA_VERSION


def test_portfolio_optimizer_contracts_are_split_and_reexported() -> None:
    optimizer = importlib.import_module("quant_investor.portfolio_optimizer")
    contracts = importlib.import_module("quant_investor.portfolio_optimizer_types")

    assert optimizer.PortfolioOptimizerConfig is contracts.PortfolioOptimizerConfig
    assert optimizer.OptimizationCandidate is contracts.OptimizationCandidate
    assert optimizer.ConstraintViolation is contracts.ConstraintViolation
    assert optimizer.OptimizedPortfolioPlan is contracts.OptimizedPortfolioPlan
    assert optimizer.RebalanceInput is contracts.RebalanceInput
    assert optimizer.RebalanceResult is contracts.RebalanceResult
    assert optimizer.WalkForwardResult is contracts.WalkForwardResult
    assert optimizer.bps_to_decimal_return is contracts.bps_to_decimal_return
    assert optimizer.estimate_turnover is contracts.estimate_turnover
    assert optimizer.make_plan_id is contracts.make_plan_id


def _candidate(symbol: str, **overrides: object) -> OptimizationCandidate:
    payload = {
        "symbol": symbol,
        "market": "CN",
        "as_of": "2026-04-26",
        "company_name": f"{symbol} Inc",
        "sector": "Technology",
        "current_weight": 0.0,
        "max_weight": None,
        "expected_alpha": 0.04,
        "edge_after_costs": 0.03,
        "confidence": 0.80,
        "action_score": 0.70,
        "risk_score": 0.20,
        "liquidity_score": 0.80,
        "estimated_transaction_cost_bps": 5.0,
        "estimated_slippage_bps": 2.0,
        "estimated_market_impact_bps": 1.0,
        "is_blocked": False,
        "block_reasons": [],
    }
    payload.update(overrides)
    return OptimizationCandidate(**payload)


def _violation() -> ConstraintViolation:
    return ConstraintViolation(
        violation_id=make_constraint_violation_id(
            symbol="000001.SZ",
            constraint_type=CONSTRAINT_MIN_EDGE,
            message="low edge",
        ),
        symbol="000001.SZ",
        constraint_type=CONSTRAINT_MIN_EDGE,
        severity="info",
        value=0.0,
        limit=0.01,
        message="low edge",
    )


def _plan() -> OptimizedPortfolioPlan:
    return OptimizedPortfolioPlan(
        plan_id="plan-test",
        as_of="2026-04-26",
        market="CN",
        status=PLAN_STATUS_OPTIMIZED,
        objective_value=0.01,
        target_weights={"000001.SZ": 0.10},
        current_weights={"000001.SZ": 0.02},
        trade_weights={"000001.SZ": 0.08},
        selected_symbols=["000001.SZ"],
        blocked_symbols=[],
        rejected_symbols=[],
        cash_weight=0.90,
        gross_exposure=0.10,
        net_exposure=0.10,
        long_exposure=0.10,
        turnover_estimate=0.08,
        sector_weights={"Technology": 0.10},
        violations=[_violation()],
        candidate_count=1,
        metadata={
            "config": PortfolioOptimizerConfig(transaction_cost_bps=10.0, slippage_bps=5.0).to_dict(),
            "optimization_method": "deterministic_greedy_v1",
        },
    )


def test_dataclass_round_trips() -> None:
    config = PortfolioOptimizerConfig(metadata={"source": "unit"})
    candidate = _candidate("000001.SZ", block_reasons=["b", "a"], confidence=1.2)
    violation = _violation()
    plan = _plan()
    rebalance_input = RebalanceInput(
        as_of="2026-04-26",
        evaluation_date="2026-04-29",
        market="CN",
        candidates=[candidate],
        forward_returns={"000001.SZ": 0.02},
        benchmark_return=0.01,
        current_weights={"000001.SZ": 0.02},
    )
    rebalance_result = evaluate_rebalance(plan, evaluation_date="2026-04-29", forward_returns={"000001.SZ": 0.02})
    walk_forward = WalkForwardResult(
        run_id="wf-test",
        market="CN",
        start_date="2026-04-26",
        end_date="2026-04-29",
        rebalance_count=1,
        cumulative_gross_return=0.002,
        cumulative_net_return=0.0019,
        max_drawdown=0.0,
        average_turnover=0.08,
        total_estimated_cost_return=0.0001,
        plans=[plan],
        rebalance_results=[rebalance_result],
    )

    assert PortfolioOptimizerConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()
    assert OptimizationCandidate.from_dict(candidate.to_dict()).block_reasons == ["a", "b"]
    assert ConstraintViolation.from_dict(violation.to_dict()).to_dict() == violation.to_dict()
    assert OptimizedPortfolioPlan.from_dict(plan.to_dict()).to_dict() == plan.to_dict()
    assert RebalanceInput.from_dict(rebalance_input.to_dict()).to_dict() == rebalance_input.to_dict()
    assert RebalanceResult.from_dict(rebalance_result.to_dict()).to_dict() == rebalance_result.to_dict()
    assert WalkForwardResult.from_dict(walk_forward.to_dict()).to_dict() == walk_forward.to_dict()


def test_deterministic_ids_are_stable() -> None:
    assert make_constraint_violation_id(symbol="A", constraint_type="x", message="m") == make_constraint_violation_id(
        symbol="A",
        constraint_type="x",
        message="m",
    )
    assert make_plan_id(market="CN", as_of="2026-04-26", symbols=["b", "a"]) == make_plan_id(
        market="CN",
        as_of="2026-04-26",
        symbols=["a", "b"],
    )
    assert make_rebalance_id(plan_id="plan", evaluation_date="2026-04-29") == make_rebalance_id(
        plan_id="plan",
        evaluation_date="2026-04-29",
    )
    assert make_walk_forward_run_id(
        market="CN",
        start_date="2026-04-26",
        end_date="2026-05-01",
        rebalance_dates=["2026-04-26"],
    ) == make_walk_forward_run_id(
        market="CN",
        start_date="2026-04-26",
        end_date="2026-05-01",
        rebalance_dates=["2026-04-26"],
    )


def test_numeric_helpers() -> None:
    assert bps_to_decimal_return(100.0) == pytest.approx(0.01)
    assert estimate_turnover({"a": 0.10, "b": 0.20}, {"a": 0.20, "c": 0.05}) == pytest.approx(0.35)
    assert compute_sector_weights({"a": 0.10, "b": 0.20}, {"a": "Tech", "b": None}) == {
        "Tech": 0.10,
        "UNKNOWN": 0.20,
    }
    assert compound_returns([0.10, -0.10]) == pytest.approx(-0.01)
    assert max_drawdown_from_returns([0.10, -0.20, 0.05]) == pytest.approx(0.20)
    with pytest.raises(ValueError, match="finite"):
        validate_finite_number(float("nan"), field_name="bad")
    with pytest.raises(ValueError, match="finite"):
        validate_finite_number(float("inf"), field_name="bad")


def test_candidate_bridge_from_stubs_and_duplicate_overlay_rejection() -> None:
    overlay = SimpleNamespace(
        symbol="000001.SZ",
        company_name="测试公司",
        market="CN",
        calibrated_posterior_expected_alpha=0.05,
        calibrated_edge_after_costs=0.03,
        calibrated_posterior_action_score=0.70,
        diagnostics=SimpleNamespace(metadata={"posterior_confidence": 0.65}),
        metadata={"posterior_confidence": 0.10},
    )
    risk_tensor = SimpleNamespace(
        symbol="000001.SZ",
        market="CN",
        as_of="2026-04-26",
        exposure=SimpleNamespace(sector="Technology"),
        risk_score=0.30,
        liquidity=SimpleNamespace(liquidity_score=0.75, estimated_market_impact_bps=3.0),
        execution=SimpleNamespace(
            status="blocked",
            blocked_reasons=["suspended"],
            estimated_transaction_cost_bps=9.0,
            estimated_slippage_bps=2.0,
            estimated_market_impact_bps=None,
        ),
        quarantine=False,
        is_tradable=True,
        issues=[SimpleNamespace(issue_type="data_quarantine", severity="blocker")],
        metadata={"max_weight": 0.08},
    )

    candidate = build_candidate_from_overlay(overlay, risk_tensor=risk_tensor, current_weight=0.04)

    assert candidate.symbol == "000001.SZ"
    assert candidate.expected_alpha == pytest.approx(0.05)
    assert candidate.edge_after_costs == pytest.approx(0.03)
    assert candidate.sector == "Technology"
    assert candidate.risk_score == pytest.approx(0.30)
    assert candidate.liquidity_score == pytest.approx(0.75)
    assert candidate.estimated_transaction_cost_bps == pytest.approx(9.0)
    assert candidate.estimated_market_impact_bps == pytest.approx(3.0)
    assert candidate.is_blocked is True
    assert candidate.block_reasons == ["data_quarantine", "suspended"]
    assert candidate.max_weight == pytest.approx(0.08)

    with pytest.raises(ValueError, match="Duplicate"):
        build_candidates_from_overlays([overlay, overlay])


def test_optimizer_clean_case_respects_caps_and_ordering() -> None:
    candidates = [
        _candidate("000001.SZ", edge_after_costs=0.04, confidence=0.90, action_score=0.80, sector="Technology"),
        _candidate("000002.SZ", edge_after_costs=0.03, confidence=0.80, action_score=0.70, sector="Industrial"),
        _candidate("000003.SZ", edge_after_costs=0.02, confidence=0.70, action_score=0.60, sector="Technology"),
    ]
    config = PortfolioOptimizerConfig(max_weight=0.20, gross_exposure_cap=0.50, sector_cap=0.30, turnover_cap=None)

    plan = optimize_portfolio(candidates, config=config, market="CN", as_of="2026-04-26")

    assert plan.status == PLAN_STATUS_OPTIMIZED
    assert plan.gross_exposure <= 0.50
    assert all(weight <= 0.20 + 1e-12 for weight in plan.target_weights.values())
    assert plan.sector_weights["Technology"] <= 0.30 + 1e-12
    assert plan.selected_symbols == sorted(plan.target_weights, key=lambda symbol: (-plan.target_weights[symbol], symbol))
    assert plan.objective_value > 0.0


def test_optimizer_rejection_and_constraint_violations() -> None:
    candidates = [
        _candidate("000001.SZ", is_blocked=True, block_reasons=["blocked"]),
        _candidate("000002.SZ", edge_after_costs=-0.01),
        _candidate("000003.SZ", risk_score=0.95),
        _candidate("000004.SZ", edge_after_costs=0.04, risk_score=0.10),
    ]
    config = PortfolioOptimizerConfig(min_edge_after_costs=0.0, max_risk_score=0.80, turnover_cap=None)

    plan = optimize_portfolio(candidates, config=config, market="CN", as_of="2026-04-26")
    violation_types = {violation.constraint_type for violation in plan.violations}

    assert "000001.SZ" in plan.blocked_symbols
    assert plan.rejected_symbols == ["000001.SZ", "000002.SZ", "000003.SZ"]
    assert CONSTRAINT_BLOCKED_SYMBOL in violation_types
    assert CONSTRAINT_MIN_EDGE in violation_types
    assert CONSTRAINT_RISK_SCORE in violation_types


def test_turnover_cap_scales_trades_and_records_violation() -> None:
    candidates = [
        _candidate("000001.SZ", edge_after_costs=0.05, confidence=0.90),
        _candidate("000002.SZ", edge_after_costs=0.04, confidence=0.80),
    ]
    config = PortfolioOptimizerConfig(max_weight=0.30, gross_exposure_cap=0.60, sector_cap=None, turnover_cap=0.10)

    plan = optimize_portfolio(
        candidates,
        config=config,
        market="CN",
        as_of="2026-04-26",
        current_weights={"000003.SZ": 0.30},
    )

    assert plan.turnover_estimate <= 0.10 + 1e-12
    assert any(violation.constraint_type == CONSTRAINT_TURNOVER_CAP for violation in plan.violations)


def test_rebalance_evaluation_uses_supplied_returns_and_costs() -> None:
    plan = _plan()
    result = evaluate_rebalance(
        plan,
        evaluation_date="2026-04-29",
        forward_returns={"000001.SZ": 0.05},
        benchmark_return=0.01,
    )

    assert result.realized_gross_return == pytest.approx(0.005)
    assert result.missing_return_symbols == []
    assert result.estimated_cost_return == pytest.approx(0.08 * 0.0015)
    assert result.realized_net_return == pytest.approx(0.005 - 0.08 * 0.0015)
    assert result.excess_return == pytest.approx(result.realized_net_return - 0.01)

    missing = evaluate_rebalance(plan, evaluation_date="2026-04-30", forward_returns={})
    assert missing.missing_return_symbols == ["000001.SZ"]
    assert missing.realized_gross_return == pytest.approx(0.0)


def test_walk_forward_loop_compounds_and_rolls_current_weights() -> None:
    config = PortfolioOptimizerConfig(max_weight=0.30, gross_exposure_cap=0.30, sector_cap=None, turnover_cap=None)
    first_candidates = [_candidate("000001.SZ", edge_after_costs=0.05, confidence=0.90)]
    second_candidates = [_candidate("000002.SZ", edge_after_costs=0.06, confidence=0.90)]
    inputs = [
        RebalanceInput(
            as_of="2026-04-26",
            evaluation_date="2026-04-29",
            market="CN",
            candidates=first_candidates,
            forward_returns={"000001.SZ": 0.10},
            benchmark_return=0.02,
            current_weights={},
        ),
        RebalanceInput(
            as_of="2026-04-29",
            evaluation_date="2026-05-06",
            market="CN",
            candidates=second_candidates,
            forward_returns={"000002.SZ": -0.05},
            benchmark_return=-0.01,
            current_weights={},
        ),
    ]

    result = run_walk_forward_loop(inputs, config=config)

    assert result.rebalance_count == 2
    assert len(result.plans) == 2
    assert len(result.rebalance_results) == 2
    assert result.plans[1].current_weights == result.plans[0].target_weights
    assert result.cumulative_net_return == pytest.approx(compound_returns([r.realized_net_return for r in result.rebalance_results]))
    assert result.average_turnover == pytest.approx(sum(r.turnover_estimate for r in result.rebalance_results) / 2.0)
    assert result.max_drawdown >= 0.0


def test_portfolio_constructor_patch_is_json_serializable() -> None:
    patch = build_portfolio_constructor_patch(_plan())

    json.dumps(patch, ensure_ascii=False, sort_keys=True)
    assert patch["target_weights"] == {"000001.SZ": 0.10}
    assert patch["blocked_symbols"] == []
    assert patch["turnover_estimate"] == pytest.approx(0.08)
    assert patch["sector_weights"] == {"Technology": 0.10}
    assert patch["violations"]
    assert patch["metadata"]["portfolio_optimizer_schema_version"] == PORTFOLIO_OPTIMIZER_SCHEMA_VERSION


def test_store_round_trips_and_rejects_duplicates_and_bad_json(tmp_path) -> None:
    plan = _plan()
    rebalance = evaluate_rebalance(plan, evaluation_date="2026-04-29", forward_returns={"000001.SZ": 0.01})
    walk_forward = WalkForwardResult(
        run_id="wf-store",
        market="CN",
        start_date="2026-04-26",
        end_date="2026-04-29",
        rebalance_count=1,
        cumulative_gross_return=rebalance.realized_gross_return,
        cumulative_net_return=rebalance.realized_net_return,
        max_drawdown=0.0,
        average_turnover=rebalance.turnover_estimate,
        total_estimated_cost_return=rebalance.estimated_cost_return,
        plans=[plan],
        rebalance_results=[rebalance],
    )
    store = PortfolioOptimizerStore(tmp_path)

    store.append_plan(plan)
    assert store.read_plans()[0].plan_id == plan.plan_id
    with pytest.raises(ValueError, match="Duplicate plan_id"):
        store.append_plan(plan)

    store.append_rebalance_result(rebalance)
    assert store.read_rebalance_results()[0].rebalance_id == rebalance.rebalance_id
    with pytest.raises(ValueError, match="Duplicate rebalance_id"):
        store.append_rebalance_result(rebalance)

    store.append_walk_forward_result(walk_forward)
    assert store.read_walk_forward_results()[0].run_id == walk_forward.run_id
    with pytest.raises(ValueError, match="Duplicate run_id"):
        store.append_walk_forward_result(walk_forward)

    bad_store = PortfolioOptimizerStore(tmp_path / "bad")
    bad_store.optimized_plans_path.parent.mkdir(parents=True, exist_ok=True)
    bad_store.optimized_plans_path.write_text("{bad json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed JSON"):
        bad_store.read_plans()


def test_actual_phase4_and_phase6_bridge_instances() -> None:
    overlay = CalibratedPosteriorOverlay(
        symbol="000001.SZ",
        company_name="真实测试",
        market="CN",
        calibrated_posterior_expected_alpha=0.04,
        calibrated_edge_after_costs=0.025,
        calibrated_posterior_action_score=0.66,
        metadata={"posterior_confidence": 0.70},
    )
    risk_tensor = SymbolRiskTensor(
        tensor_id="tensor",
        symbol="000001.SZ",
        market="CN",
        as_of="2026-04-26",
        latest_trade_date="2026-04-25",
        exposure=SymbolExposure(symbol="000001.SZ", market="CN", as_of="2026-04-26", sector="Technology"),
        liquidity=LiquidityProfile(
            symbol="000001.SZ",
            market="CN",
            as_of="2026-04-26",
            liquidity_score=0.60,
            estimated_market_impact_bps=4.0,
        ),
        execution=ExecutionFeasibility(
            symbol="000001.SZ",
            market="CN",
            as_of="2026-04-26",
            status=EXECUTION_BLOCKED,
            blocked_reasons=["untradable"],
            estimated_transaction_cost_bps=8.0,
            estimated_slippage_bps=2.0,
        ),
        issues=[
            RiskIssue(
                issue_id="issue",
                symbol="000001.SZ",
                market="CN",
                as_of="2026-04-26",
                issue_type=RISK_ISSUE_DATA_QUARANTINE,
                severity=RISK_SEVERITY_BLOCKER,
                message="quarantine",
            )
        ],
        risk_score=0.40,
    )

    candidate = build_candidate_from_overlay(overlay, risk_tensor=risk_tensor)

    assert candidate.symbol == "000001.SZ"
    assert candidate.sector == "Technology"
    assert candidate.confidence == pytest.approx(0.70)
    assert candidate.risk_score == pytest.approx(0.40)
    assert candidate.is_blocked is True
    assert candidate.block_reasons == [RISK_ISSUE_DATA_QUARANTINE, "untradable"]
