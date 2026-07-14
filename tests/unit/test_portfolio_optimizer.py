from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import quant_investor.portfolio_optimizer as optimizer_module
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
from quant_investor.versioning import PORTFOLIO_OPTIMIZER_SCHEMA_VERSION


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


_OVERLAY_PROVENANCE_CASES = [
    (
        "payload-key",
        {"calibrated_posterior_overlay": {"schema_version": "v2"}},
    ),
    (
        "schema-key",
        {"posterior_overlay_schema_version": "2026-07-14.posterior-overlay.v2"},
    ),
    ("shadow-mode", {"overlay_mode": "shadow"}),
    (
        "report-contract",
        {
            "report_only": True,
            "production_eligible": False,
            "production_weight": 0.0,
        },
    ),
    ("source-type", {"source_type": "posterior_overlay"}),
]


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


def test_candidate_bridges_fail_closed_before_reading_overlay_or_risk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: list[str] = []

    class Poison:
        def __getattribute__(self, name: str):
            reads.append(name)
            raise AssertionError(f"bridge read forbidden input: {name}")

    with pytest.raises(ValueError, match="report-only"):
        build_candidate_from_overlay(
            Poison(),
            risk_tensor=Poison(),
            current_weight=0.35,
            metadata=Poison(),
        )
    assert reads == []

    legacy_v1 = {
        "schema_version": "2026-04-26.posterior-overlay.v1",
        "symbol": "000001.SZ",
        "calibrated_posterior_expected_alpha": 0.05,
        "calibrated_edge_after_costs": 0.03,
        "calibrated_posterior_action_score": 0.70,
    }
    with pytest.raises(ValueError, match="report-only"):
        build_candidate_from_overlay(legacy_v1, current_weight=0.35)

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
    constructed = 0

    def counting_candidate(**kwargs):
        nonlocal constructed
        constructed += 1
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(
        optimizer_module,
        "OptimizationCandidate",
        counting_candidate,
    )
    with pytest.raises(ValueError, match="report-only"):
        build_candidates_from_overlays([overlay, overlay])
    assert constructed == 0


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


def test_bridge_cannot_convert_report_only_overlay_into_forced_exit() -> None:
    class RiskTensorPoison:
        def __getattribute__(self, name: str):
            raise AssertionError(
                f"bridge read risk field before fail-closed rejection: {name}"
            )

    report_only_overlay = SimpleNamespace(
        schema_version="2026-07-14.posterior-overlay.v2",
        overlay_mode="shadow",
        report_only=True,
        production_eligible=False,
        production_weight=0.0,
    )

    with pytest.raises(ValueError, match="report-only"):
        build_candidate_from_overlay(
            report_only_overlay,
            risk_tensor=RiskTensorPoison(),
            current_weight=0.40,
        )


@pytest.mark.parametrize(
    ("marker_name", "marker"),
    _OVERLAY_PROVENANCE_CASES,
    ids=[case[0] for case in _OVERLAY_PROVENANCE_CASES],
)
def test_optimizer_rejects_recursive_overlay_candidate_metadata_before_scoring(
    marker_name: str,
    marker: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = OptimizationCandidate.from_dict(
        _candidate(
            "000001.SZ",
            current_weight=0.40,
            metadata={"nested": [{"marker": marker}]},
        ).to_dict()
    )
    score_reads = 0

    def explode_score(*args, **kwargs):
        nonlocal score_reads
        score_reads += 1
        raise AssertionError(f"scored overlay-derived candidate: {marker_name}")

    monkeypatch.setattr(optimizer_module, "_candidate_adjusted_score", explode_score)

    with pytest.raises(ValueError, match="overlay"):
        optimize_portfolio(
            [candidate],
            current_weights={candidate.symbol: 0.40},
        )
    assert score_reads == 0


@pytest.mark.parametrize(
    ("marker_name", "marker"),
    _OVERLAY_PROVENANCE_CASES,
    ids=[case[0] for case in _OVERLAY_PROVENANCE_CASES],
)
def test_constructor_patch_rejects_recursive_overlay_plan_metadata(
    marker_name: str,
    marker: dict[str, object],
) -> None:
    payload = _plan().to_dict()
    payload["metadata"] = {"nested": [{"marker": marker}]}
    round_tripped = OptimizedPortfolioPlan.from_dict(payload)

    with pytest.raises(ValueError, match="overlay"):
        build_portfolio_constructor_patch(round_tripped)


def test_optimizer_rejects_report_only_ineligible_marker_without_weight() -> None:
    candidate = _candidate(
        "CURRENT",
        current_weight=0.40,
        metadata={
            "nested": {
                "report_only": True,
                "production_eligible": False,
            }
        },
    )

    with pytest.raises(ValueError, match="overlay"):
        optimize_portfolio(
            [candidate],
            current_weights={candidate.symbol: 0.40},
        )


def test_constructor_patch_rejects_report_only_ineligible_without_weight() -> None:
    payload = _plan().to_dict()
    payload["metadata"] = {
        "nested": {
            "report_only": True,
            "eligible": False,
        }
    }
    plan = OptimizedPortfolioPlan.from_dict(payload)

    with pytest.raises(ValueError, match="overlay"):
        build_portfolio_constructor_patch(plan)


@pytest.mark.parametrize("location", ["input_metadata", "config_metadata"])
def test_optimizer_rejects_overlay_provenance_destined_for_plan_metadata(
    location: str,
) -> None:
    marker = {"nested": {"source_type": "posterior_overlay"}}
    kwargs: dict[str, object] = {}
    if location == "input_metadata":
        kwargs["metadata"] = marker
    else:
        kwargs["config"] = PortfolioOptimizerConfig(metadata=marker)

    with pytest.raises(ValueError, match="overlay"):
        optimize_portfolio([_candidate("000001.SZ")], **kwargs)


def test_optimizer_preserves_non_overlay_metadata_behavior() -> None:
    metadata = {
        "nested": {
            "overlay_mode": "off",
            "report_only": False,
            "production_eligible": False,
            "production_weight": 0.0,
            "source_type": "posterior",
        }
    }
    candidate = _candidate("NORMAL", metadata=metadata)

    plan = optimize_portfolio(
        [candidate],
        config=PortfolioOptimizerConfig(
            max_weight=0.20,
            sector_cap=None,
            turnover_cap=None,
        ),
    )
    patch = build_portfolio_constructor_patch(plan)

    assert plan.target_weights == {"NORMAL": pytest.approx(0.20)}
    assert patch["target_weights"] == {"NORMAL": pytest.approx(0.20)}


@pytest.mark.parametrize(
    "schema_version",
    [
        "2026-07-14.posterior-overlay.v2",
        "2026-04-26.portfolio-optimizer.v1",
        "evil-schema.v999",
    ],
)
def test_optimizer_execution_rejects_non_current_candidate_schema_before_scoring(
    schema_version: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _candidate("CURRENT", current_weight=0.40).to_dict()
    payload["schema_version"] = schema_version
    observed = OptimizationCandidate.from_dict(payload)
    score_reads = 0

    def explode_score(*args, **kwargs):
        nonlocal score_reads
        score_reads += 1
        raise AssertionError("non-current candidate reached scoring")

    monkeypatch.setattr(optimizer_module, "_candidate_adjusted_score", explode_score)
    assert observed.schema_version == schema_version

    with pytest.raises(ValueError, match="schema"):
        optimize_portfolio(
            [observed],
            current_weights={observed.symbol: 0.40},
        )
    assert score_reads == 0


@pytest.mark.parametrize(
    "schema_version",
    [
        "2026-07-14.posterior-overlay.v2",
        "2026-04-26.portfolio-optimizer.v1",
        "evil-schema.v999",
    ],
)
def test_constructor_patch_rejects_non_current_plan_schema(
    schema_version: str,
) -> None:
    payload = _plan().to_dict()
    payload["schema_version"] = schema_version
    observed = OptimizedPortfolioPlan.from_dict(payload)
    assert observed.schema_version == schema_version

    with pytest.raises(ValueError, match="schema"):
        build_portfolio_constructor_patch(observed)


_FINAL_NORMALIZED_PROVENANCE_CASES: list[
    tuple[str, dict[object, object]]
] = [
    (
        "schema-v2-value",
        {"note": "2026-07-14.posterior-overlay.v2"},
    ),
    (
        "schema-v1-value",
        {"note": "2026-04-26.posterior-overlay.v1"},
    ),
    (
        "standalone-report-only",
        {"Report-Only": True},
    ),
    (
        "marker-string",
        {"note": "Calibrated Posterior Overlay"},
    ),
    (
        "schema-marker-string",
        {"note": "Posterior-Overlay-Schema.Version"},
    ),
    (
        "mode-list-pair",
        {"pairs": [["Overlay Mode", "SHADOW"]]},
    ),
    (
        "source-list-pair",
        {"pairs": [["Source-Type", "Posterior Overlay"]]},
    ),
    (
        "non-string-key",
        {("Overlay", "Mode"): "Shadow"},
    ),
    (
        "case-separated-key",
        {"SOURCE.TYPE": "POSTERIOR-OVERLAY"},
    ),
]


@pytest.mark.parametrize(
    "location",
    ["candidate", "plan", "input_metadata", "config_metadata"],
)
@pytest.mark.parametrize(
    ("marker_name", "metadata"),
    _FINAL_NORMALIZED_PROVENANCE_CASES,
    ids=[case[0] for case in _FINAL_NORMALIZED_PROVENANCE_CASES],
)
def test_final_recursive_detector_rejects_normalized_overlay_markers(
    location: str,
    marker_name: str,
    metadata: dict[object, object],
) -> None:
    if location == "candidate":
        candidate = _candidate("MARKED", metadata=metadata)
        with pytest.raises(ValueError, match="overlay"):
            optimize_portfolio([candidate])
    elif location == "plan":
        payload = _plan().to_dict()
        payload["metadata"] = metadata
        with pytest.raises(ValueError, match="overlay"):
            build_portfolio_constructor_patch(
                OptimizedPortfolioPlan.from_dict(payload)
            )
    elif location == "input_metadata":
        with pytest.raises(ValueError, match="overlay"):
            optimize_portfolio(
                [_candidate("MARKED")],
                metadata=metadata,  # type: ignore[arg-type]
            )
    else:
        config = PortfolioOptimizerConfig(
            metadata=metadata,  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match="overlay"):
            optimize_portfolio([_candidate("MARKED")], config=config)


@pytest.mark.parametrize(
    "location",
    ["candidate", "plan", "input_metadata", "config_metadata"],
)
def test_final_recursive_detector_preserves_normalized_ordinary_metadata(
    location: str,
) -> None:
    metadata = {
        "Report-Only": False,
        "Source.Type": "Posterior",
        "Overlay Mode": "OFF",
        "note": "posterior calibration observation",
    }
    if location == "candidate":
        plan = optimize_portfolio([_candidate("NORMAL", metadata=metadata)])
        assert plan.target_weights
    elif location == "plan":
        payload = _plan().to_dict()
        payload["metadata"] = metadata
        patch = build_portfolio_constructor_patch(
            OptimizedPortfolioPlan.from_dict(payload)
        )
        assert patch["target_weights"]
    elif location == "input_metadata":
        plan = optimize_portfolio(
            [_candidate("NORMAL")],
            metadata=metadata,
        )
        assert plan.target_weights
    else:
        plan = optimize_portfolio(
            [_candidate("NORMAL")],
            config=PortfolioOptimizerConfig(metadata=metadata),
        )
        assert plan.target_weights


@pytest.mark.parametrize("artifact", ["candidate", "plan"])
@pytest.mark.parametrize("attack", ["unknown", "missing"])
def test_current_v2_from_dict_requires_exact_top_level_fields(
    artifact: str,
    attack: str,
) -> None:
    if artifact == "candidate":
        payload = _candidate("CURRENT").to_dict()
        parser = OptimizationCandidate.from_dict
        missing_field = "symbol"
    else:
        payload = _plan().to_dict()
        parser = OptimizedPortfolioPlan.from_dict
        missing_field = "plan_id"
    if attack == "unknown":
        payload.update(
            {
                "overlay_mode": "shadow",
                "report_only": True,
                "production_eligible": False,
                "production_weight": 0.0,
            }
        )
    else:
        payload.pop(missing_field)

    with pytest.raises(ValueError, match="fields"):
        parser(payload)


class EqualCurrent:
    def __str__(self) -> str:
        return PORTFOLIO_OPTIMIZER_SCHEMA_VERSION

    def __eq__(self, other: object) -> bool:
        return other == PORTFOLIO_OPTIMIZER_SCHEMA_VERSION

    def __ne__(self, other: object) -> bool:
        return not self == other


@pytest.mark.parametrize("artifact", ["candidate", "plan"])
def test_from_dict_rejects_non_exact_string_schema(artifact: str) -> None:
    payload = (
        _candidate("CURRENT").to_dict()
        if artifact == "candidate"
        else _plan().to_dict()
    )
    payload["schema_version"] = EqualCurrent()
    parser = (
        OptimizationCandidate.from_dict
        if artifact == "candidate"
        else OptimizedPortfolioPlan.from_dict
    )

    with pytest.raises(TypeError, match="schema"):
        parser(payload)


@pytest.mark.parametrize("artifact", ["candidate", "plan"])
def test_execution_guards_reject_equal_current_objects(artifact: str) -> None:
    if artifact == "candidate":
        candidate = _candidate("CURRENT", schema_version=EqualCurrent())
        with pytest.raises(ValueError, match="schema"):
            optimize_portfolio([candidate])
    else:
        plan = _plan()
        plan.schema_version = EqualCurrent()  # type: ignore[assignment]
        with pytest.raises(ValueError, match="schema"):
            build_portfolio_constructor_patch(plan)


@pytest.mark.parametrize("artifact", ["candidate", "plan"])
def test_legacy_payload_remains_observable_but_not_executable(
    artifact: str,
) -> None:
    if artifact == "candidate":
        observed = OptimizationCandidate.from_dict(
            {
                "schema_version": "2026-04-26.portfolio-optimizer.v1",
                "symbol": "OBSERVED",
                "unknown_control": {"report_only": False},
            }
        )
        assert observed.symbol == "OBSERVED"
        with pytest.raises(ValueError, match="schema"):
            optimize_portfolio([observed])
    else:
        observed = OptimizedPortfolioPlan.from_dict(
            {
                "schema_version": "legacy-plan.observation.v1",
                "plan_id": "observed-plan",
                "unknown_control": {"report_only": False},
            }
        )
        assert observed.plan_id == "observed-plan"
        with pytest.raises(ValueError, match="schema"):
            build_portfolio_constructor_patch(observed)
