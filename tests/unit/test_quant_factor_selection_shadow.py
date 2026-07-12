from __future__ import annotations

import pandas as pd

from quant_investor.factors.governance import (
    FactorLifecycleState,
    FactorRecord,
    GateResult,
)
from quant_investor.factors.runtime import MinedFactorRegistry, score_with_mined_factors
from scripts.run_quant_factor_selection_shadow import (
    DEFAULT_SHADOW_EFFECTIVE_SHARE,
    _covered_uncovered_selection_bias,
    append_observation_once,
    build_baseline_identity,
    build_arm_specs,
    combine_runtime_components,
    compare_rankings,
    compute_runtime_components,
    ensure_preregistration,
    ensure_baseline_contract,
    governed_experiment_config,
    preregistration_policy,
    validate_registry_contract,
)


def _gates() -> list[GateResult]:
    return [
        GateResult(
            gate_id=gate_id,
            gate_key=f"gate_{gate_id}",
            title=f"Gate {gate_id}",
            passed=True,
        )
        for gate_id in range(1, 9)
    ]


def _record(name: str, implementation: str, *, weight: float = 0.05) -> FactorRecord:
    return FactorRecord(
        name=name,
        state=FactorLifecycleState.PRODUCTION_FACTOR,
        implementation=implementation,
        weight=weight,
        gate_results=_gates(),
    )


def _frames() -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2025-01-02", periods=80, freq="B")
    frames: dict[str, pd.DataFrame] = {}
    for index, symbol in enumerate(["000001.SZ", "000002.SZ", "000003.SZ", "600000.SH"]):
        close = pd.Series(range(100 + index * 5, 180 + index * 5), dtype=float)
        close = close.add(pd.Series(range(80), dtype=float).pow(1 + index / 10).mul(0.01))
        frames[symbol] = pd.DataFrame(
            {
                "trade_date": dates.strftime("%Y%m%d"),
                "close": close,
                "vol": pd.Series(range(1000 + index * 50, 1080 + index * 50), dtype=float),
                "amount": close.mul(1000 + index * 50),
            }
        )
    return frames


def test_runtime_components_recompose_exact_direct_runtime_score() -> None:
    records = [
        _record("return", "builtin:short_term_return"),
        _record("risk", "builtin:volatility_penalty"),
    ]
    frames = _frames()
    components, details, skipped, covered = compute_runtime_components(frames, records)
    actual = combine_runtime_components(
        components,
        {record.name: record.weight for record in records},
    )
    direct = score_with_mined_factors(
        frames,
        registry=MinedFactorRegistry.from_records(records),
    )

    assert skipped == {}
    assert set(details) == {"return", "risk"}
    assert set(covered) == {"return", "risk"}
    expected = pd.Series(direct.symbol_scores, dtype=float).reindex(actual.index)
    pd.testing.assert_series_equal(actual, expected, check_names=False, atol=1e-12, rtol=0.0)


def test_arm_protocol_builds_a_all_bi_ci_and_three_percent_d() -> None:
    records = [
        _record("old_a", "builtin:short_term_return"),
        _record("old_b", "builtin:volatility_penalty"),
    ]
    arms = build_arm_specs(
        records,
        candidate_name="candidate",
    )

    assert [arm.arm_type for arm in arms].count("baseline") == 1
    assert [arm.arm_type for arm in arms].count("leave_one_out") == 2
    assert [arm.arm_type for arm in arms].count("one_for_one_replacement") == 2
    additive = [arm for arm in arms if arm.arm_type == "additive_shadow"][0]
    effective_share = additive.factor_weights["candidate"] / sum(
        additive.factor_weights.values()
    )
    assert abs(effective_share - DEFAULT_SHADOW_EFFECTIVE_SHARE) < 1e-12
    replacements = [arm for arm in arms if arm.arm_type == "one_for_one_replacement"]
    assert all(arm.factor_weights["candidate"] == 0.05 for arm in replacements)


def test_default_shadow_weight_is_three_percent_for_old14() -> None:
    records = [
        _record(f"old_{index}", "builtin:short_term_return", weight=0.01 + index / 1000)
        for index in range(14)
    ]
    additive = build_arm_specs(records, candidate_name="candidate")[-1]
    candidate_weight = additive.factor_weights["candidate"]
    effective_share = candidate_weight / sum(additive.factor_weights.values())
    assert abs(effective_share - DEFAULT_SHADOW_EFFECTIVE_SHARE) < 1e-12


def test_rank_comparison_reports_top_n_membership_flips() -> None:
    baseline = pd.Series({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})
    arm = pd.Series({"A": 4.0, "B": 1.0, "C": 2.0, "D": 3.0})

    comparison = compare_rankings(baseline, arm, symbols=list(baseline.index), top_ns=(2,))

    assert comparison["rank_spearman"] < 1.0
    assert comparison["top_n"]["2"]["entered"] == ["D"]
    assert comparison["top_n"]["2"]["exited"] == ["B"]
    assert comparison["top_n"]["2"]["membership_flip_count"] == 2


def test_registry_contract_requires_zero_weight_8_gate_candidate() -> None:
    production = _record("old", "builtin:short_term_return")
    candidate = FactorRecord(
        name="candidate",
        state=FactorLifecycleState.PRODUCTION_CANDIDATE,
        implementation="aquant_expression:candidate",
        weight=0.0,
        gate_results=_gates(),
        metadata={"expression": "cs_rank(fin_net_profit_yoy)"},
    )
    registry = MinedFactorRegistry.from_records([production, candidate])

    active, selected, blockers = validate_registry_contract(
        registry,
        candidate_name="candidate",
        expected_production_factor_count=1,
    )

    assert [record.name for record in active] == ["old"]
    assert selected is candidate
    assert blockers == []


def test_preregistration_is_create_once_and_hash_locked(tmp_path) -> None:
    path = tmp_path / "preregistration.json"
    policy = preregistration_policy()

    created, blockers = ensure_preregistration(
        path,
        policy=policy,
        created_at="2026-07-11T00:00:00+08:00",
    )
    matched, repeated_blockers = ensure_preregistration(
        path,
        policy=policy,
        created_at="2026-07-12T00:00:00+08:00",
    )
    changed = {**policy, "min_month_end_rankic_count": 99}
    blocked, changed_blockers = ensure_preregistration(
        path,
        policy=changed,
        created_at="2026-07-13T00:00:00+08:00",
    )

    assert blockers == []
    assert created["status"] == "created"
    assert repeated_blockers == []
    assert matched["status"] == "matched"
    assert blocked["status"] == "blocked"
    assert "preregistration_policy_mismatch" in changed_blockers


def test_governed_experiment_config_rejects_weight_and_threshold_drift() -> None:
    registered = preregistration_policy()["experiment_config"]

    assert governed_experiment_config() == registered
    assert (
        governed_experiment_config(candidate_shadow_weight_override=9.0)
        != registered
    )
    assert governed_experiment_config(min_candidate_coverage=0.01) != registered


def test_observation_ledger_deduplicates_snapshot_as_of_key(tmp_path) -> None:
    path = tmp_path / "observation_ledger.jsonl"
    row = {
        "observation_key": "snapshot|20260708|candidate",
        "snapshot_id": "snapshot",
        "as_of": "20260708",
        "candidate": "candidate",
    }

    first, blockers = append_observation_once(path, row)
    repeated, repeated_blockers = append_observation_once(path, row)

    assert blockers == []
    assert repeated_blockers == []
    assert first["status"] == "appended"
    assert repeated["status"] == "duplicate_not_appended"
    assert first["unique_observation_count"] == 1
    assert repeated["unique_observation_count"] == 1
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_covered_uncovered_bias_reports_selection_rate_ratio() -> None:
    covered = [f"C{index:02d}" for index in range(20)]
    uncovered = [f"U{index:02d}" for index in range(20)]
    symbols = covered + uncovered
    baseline = pd.Series({symbol: float(40 - index) for index, symbol in enumerate(symbols)})
    top_order = uncovered[:15] + covered[:5] + uncovered[15:] + covered[5:]
    additive = pd.Series(
        {symbol: float(40 - index) for index, symbol in enumerate(top_order)}
    )

    bias = _covered_uncovered_selection_bias(
        candidate_name="candidate",
        covered_symbols=set(covered),
        selection_symbols=symbols,
        baseline_scores=baseline,
        additive_scores=additive,
    )

    top20 = bias["D_add_candidate_3pct"]["top_n"]["20"]
    assert top20["covered_selection_rate"] == 0.25
    assert top20["uncovered_selection_rate"] == 0.75
    assert abs(top20["covered_to_uncovered_selection_rate_ratio"] - 1 / 3) < 1e-12


def test_baseline_contract_is_create_once_and_blocks_identity_drift(tmp_path) -> None:
    path = tmp_path / "baseline_contract.json"
    production = [_record("old", "builtin:short_term_return")]
    candidate = FactorRecord(
        name="candidate",
        version="v1",
        state=FactorLifecycleState.PRODUCTION_CANDIDATE,
        implementation="aquant_expression:candidate",
        weight=0.0,
        gate_results=_gates(),
        metadata={"expression": "cs_rank(fin_net_profit_yoy)"},
    )
    identity = build_baseline_identity(production, candidate)
    start_audit = {
        "snapshot_id": "snapshot-a",
        "manifest_sha256": "manifest-a",
        "full_registry_sha256": "registry-a",
    }

    created, blockers = ensure_baseline_contract(
        path,
        identity=identity,
        start_audit=start_audit,
        created_at="2026-07-11T00:00:00+08:00",
    )
    matched, repeated_blockers = ensure_baseline_contract(
        path,
        identity=identity,
        start_audit={**start_audit, "snapshot_id": "snapshot-b"},
        created_at="2026-07-12T00:00:00+08:00",
    )
    drifted_identity = {
        **identity,
        "ranking_contract": {**identity["ranking_contract"], "top_n": [10, 20]},
    }
    blocked, drift_blockers = ensure_baseline_contract(
        path,
        identity=drifted_identity,
        start_audit=start_audit,
        created_at="2026-07-13T00:00:00+08:00",
    )

    assert blockers == []
    assert created["status"] == "created"
    assert repeated_blockers == []
    assert matched["status"] == "matched"
    assert matched["start_audit"]["snapshot_id"] == "snapshot-a"
    assert blocked["status"] == "blocked"
    assert "baseline_contract_identity_mismatch" in drift_blockers


def test_baseline_contract_locks_experiment_weights_and_thresholds(tmp_path) -> None:
    path = tmp_path / "baseline_contract_v2.json"
    production = [_record("old", "builtin:short_term_return")]
    candidate = FactorRecord(
        name="candidate",
        version="v1",
        state=FactorLifecycleState.PRODUCTION_CANDIDATE,
        implementation="aquant_expression:candidate",
        weight=0.0,
        gate_results=_gates(),
        metadata={"expression": "cs_rank(fin_net_profit_yoy)"},
    )
    identity = build_baseline_identity(
        production,
        candidate,
        experiment_config=governed_experiment_config(),
    )
    start_audit = {"snapshot_id": "snapshot-a"}
    _, blockers = ensure_baseline_contract(
        path,
        identity=identity,
        start_audit=start_audit,
        created_at="2026-07-11T00:00:00+08:00",
    )
    drifted = build_baseline_identity(
        production,
        candidate,
        experiment_config=governed_experiment_config(
            replacement_weight_override=9.0,
        ),
    )
    blocked, drift_blockers = ensure_baseline_contract(
        path,
        identity=drifted,
        start_audit=start_audit,
        created_at="2026-07-12T00:00:00+08:00",
    )

    assert blockers == []
    assert blocked["status"] == "blocked"
    assert "baseline_contract_identity_mismatch" in drift_blockers
