from __future__ import annotations

import json

import pandas as pd

from quant_investor.factors.governance import (
    FactorAdmissionDecision,
    FactorLifecycleState,
    FactorRecord,
    GateResult,
)
from quant_investor.factors.runtime import MinedFactorRegistry
from scripts.mine_quant_branch_factors import (
    MiningCandidate,
    apply_production_candidate_registry_updates,
    build_candidate_catalog,
    candidate_maturity_context,
    compute_formulaic_signal,
)
from scripts.retest_aquant_alpha_mix_8gate import RetestContext


def _context() -> RetestContext:
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    columns = ["000001.SZ", "000002.SZ"]
    matrix = pd.DataFrame(1.0, index=dates, columns=columns)
    return RetestContext(
        frames={},
        universe_by_symbol={symbol: "fixture" for symbol in columns},
        adj_close=matrix,
        volume=matrix,
        amount=matrix,
        forward_return=matrix,
        rebalance_dates=list(dates),
        biweekly_dates=list(dates),
        existing_composite=None,
        existing_blocker="",
    )


def test_candidate_maturity_context_uses_signal_coverage_only():
    context = _context()
    signal = pd.DataFrame(
        [
            [None, None],
            [1.0, None],
            [2.0, None],
            [3.0, 4.0],
        ],
        index=context.adj_close.index,
        columns=context.adj_close.columns,
    )

    matured, start = candidate_maturity_context(
        context,
        signal,
        base_start="2024-01-02",
        min_signal_coverage=0.60,
    )

    assert start == "2024-01-05"
    assert list(matured.adj_close.index) == [pd.Timestamp("2024-01-05")]


def test_compute_formulaic_signal_rank_blends_primitives():
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    columns = ["000001.SZ", "000002.SZ"]
    left = pd.DataFrame([[1.0, 2.0], [4.0, 3.0]], index=dates, columns=columns)
    right = pd.DataFrame(
        [[9.0, 8.0], [5.0, 6.0]],
        index=dates,
        columns=columns,
    )
    candidate = MiningCandidate(
        name="formula_fixture",
        family="fixture",
        category="formulaic_research",
        implementation="research_formula:rank_blend",
        description="fixture",
        params={"left": "left", "right": "right", "left_weight": 0.25},
    )

    actual = compute_formulaic_signal(
        candidate,
        {"left": left, "right": right},
    )
    expected = left.rank(axis=1, pct=True).mul(0.25) + right.rank(
        axis=1,
        pct=True,
    ).mul(0.75)

    pd.testing.assert_frame_equal(actual, expected)


def _gate_rows() -> list[dict[str, object]]:
    return [
        {
            "gate_id": gate_id,
            "gate_key": f"gate_{gate_id}",
            "title": f"Gate {gate_id}",
            "passed": True,
            "reasons": [],
            "metrics": {},
            "severity": "info",
        }
        for gate_id in range(1, 9)
    ]


def _qualified_result(name: str = "daily_fixture_factor") -> dict[str, object]:
    return {
        "name": name,
        "family": "momentum",
        "category": "momentum",
        "implementation": "price_volume:pv_momentum_20d",
        "expression": "",
        "window": 20,
        "params": {},
        "description": "fixture daily mining factor",
        "effective_analysis_start_date": "2024-01-02",
        "decision": FactorAdmissionDecision.PRODUCTION_CANDIDATE.value,
        "target_state": FactorLifecycleState.PRODUCTION_CANDIDATE.value,
        "gates_passed": 8,
        "passed_gate_ids": list(range(1, 9)),
        "failed_gate_ids": [],
        "gate_results": _gate_rows(),
        "metrics": {
            "horizon_days": 30,
            "mean_rankic": 0.035,
            "icir": 0.85,
            "positive_ic_ratio": 0.62,
            "top_bottom_spread": 0.04,
            "master_return_delta": 0.012,
        },
        "blockers": [],
        "summary": "passes fixture gates",
    }


def _write_registry(path, registry: MinedFactorRegistry) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": registry.schema_version,
                "metadata": registry.metadata,
                "factors": [record.to_dict() for record in registry.factors],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_candidate_catalog_covers_daily_mining_dimensions():
    candidates = build_candidate_catalog((5, 20, 60, 120))
    families = {candidate.family for candidate in candidates}
    categories = {candidate.category for candidate in candidates}

    assert len(candidates) >= 120
    assert {
        "momentum",
        "short_reversal",
        "volume_stability",
        "low_dollar_volume",
        "high_dollar_volume",
        "amihud_illiquidity",
        "volatility_penalty",
        "downside_volatility",
        "price_efficiency",
        "dollar_volume_growth",
        "fin_roe",
        "momentum_liquidity",
        "fundamental_quality_value",
    }.issubset(families)
    assert {
        "momentum",
        "reversal",
        "trading_activity",
        "liquidity",
        "capacity",
        "risk",
        "trend_quality",
        "formulaic_research",
    }.issubset(categories)


def test_registry_write_adds_zero_weight_production_candidate(tmp_path):
    registry_path = tmp_path / "mined_factors.json"
    source_notes = [{"title": "fixture source", "url": "https://example.com"}]

    manifest = apply_production_candidate_registry_updates(
        registry_path=registry_path,
        qualified_results=[_qualified_result()],
        run_timestamp="2026-06-07T04:30:00",
        run_id="daily_factor_mining_fixture",
        report_path="reports/factor_governance/daily_mining/fixture.json",
        owner="test owner",
        source_notes=source_notes,
        horizon_days=30,
        max_candidates=5,
        write=True,
    )

    registry = MinedFactorRegistry.load(registry_path)
    assert manifest["status"] == "updated"
    assert manifest["written_factors"] == ["daily_fixture_factor"]
    assert registry.selectable_factors() == []
    assert registry.non_selectable_reasons() == {
        "daily_fixture_factor": "state=production_candidate"
    }
    record = registry.factors[0]
    assert record.state == FactorLifecycleState.PRODUCTION_CANDIDATE
    assert record.weight == 0.0
    assert record.admission_decision == (
        FactorAdmissionDecision.PRODUCTION_CANDIDATE
    )
    assert record.metadata["manual_promotion_required"] is True
    assert record.metadata["runtime_effect"] == (
        "none_until_manual_production_factor_promotion"
    )
    assert record.metadata["source_notes"] == source_notes


def test_registry_write_does_not_override_existing_production_factor(tmp_path):
    registry_path = tmp_path / "mined_factors.json"
    existing = FactorRecord(
        name="daily_fixture_factor",
        state=FactorLifecycleState.PRODUCTION_FACTOR,
        implementation="price_volume:pv_momentum_20d",
        weight=0.05,
        gate_results=[
            GateResult.from_dict(row) for row in _gate_rows()
        ],
        admission_decision=FactorAdmissionDecision.PRODUCTION_CANDIDATE,
    )
    _write_registry(
        registry_path,
        MinedFactorRegistry.from_records([existing]),
    )

    manifest = apply_production_candidate_registry_updates(
        registry_path=registry_path,
        qualified_results=[_qualified_result()],
        run_timestamp="2026-06-07T04:30:00",
        run_id="daily_factor_mining_fixture",
        report_path="reports/factor_governance/daily_mining/fixture.json",
        owner="test owner",
        horizon_days=30,
        max_candidates=5,
        write=True,
    )

    registry = MinedFactorRegistry.load(registry_path)
    record = registry.factors[0]
    assert manifest["status"] == "no_registry_changes"
    assert manifest["skipped_factors"] == [
        {
            "name": "daily_fixture_factor",
            "reason": "existing_production_factor",
        }
    ]
    assert record.state == FactorLifecycleState.PRODUCTION_FACTOR
    assert record.weight == 0.05
