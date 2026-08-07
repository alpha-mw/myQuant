from __future__ import annotations

import copy
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from quant_investor.factors.governance import (
    GATE_SPECS,
    FactorLifecycleState,
    FactorRecord,
    GateResult,
)
from quant_investor.factors.governance_evidence import (
    RAW_REPLAY_SCHEMA_VERSION,
    SNAPSHOT_EVIDENCE_SCHEMA_VERSION,
    build_registry_mutation_plan_from_evidence,
    load_governance_replay_evidence,
    produce_governance_replay_evidence,
    replay_arm_hash,
    verify_governance_replay_evidence,
    write_governance_replay_evidence,
)
from quant_investor.factors.governance_protocol_v2 import (
    CANONICAL_FULL_CHAIN_PRODUCER_BLOCKER,
    FactorEvidenceWindow,
    FactorSlot,
    RegistryMutationPlan,
    advance_failure_streak,
    apply_governed_transition,
    assess_candidate_maturity,
    benjamini_hochberg_by_family,
    block_bootstrap_paired_delta_ci,
    build_slot_risk_budget,
    evaluate_c_arm,
    governance_runtime_status,
    load_mutation_budget_ledger,
    monthly_mutation_budget_blockers,
    protocol_hash,
    protocol_policy,
    reserve_monthly_mutation_budget,
    validate_purged_walk_forward,
)
from quant_investor.factors.registry_store import (
    apply_factor_record_patch,
    load_registry_snapshot_strict,
)
from quant_investor.factors.runtime import MinedFactorRegistry


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _artifact_hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _gates() -> list[GateResult]:
    return [
        GateResult(
            gate_id=spec.gate_id,
            gate_key=spec.key,
            title=spec.title,
            passed=True,
            metrics={
                "artifact_hash": _artifact_hash(f"gate-{spec.gate_id}"),
                **({"coverage_rate": 1.0} if spec.gate_id == 2 else {}),
            },
        )
        for spec in GATE_SPECS
    ]


def _record(
    name: str,
    *,
    state: FactorLifecycleState,
    weight: float,
    family: str = "momentum",
    cluster: str = "price_momentum",
) -> FactorRecord:
    return FactorRecord(
        name=name,
        state=state,
        implementation=f"price_volume:{name}",
        category=family,
        weight=weight,
        gate_results=_gates(),
        metadata={
            "factor_family": family,
            "dominant_primitive_cluster": cluster,
        },
    )


def _write_registry(path: Path, records: list[FactorRecord]) -> None:
    payload = MinedFactorRegistry.from_records(records)
    path.write_text(
        json.dumps(
            {
                "schema_version": payload.schema_version,
                "metadata": {"fixture": True},
                "factors": [record.to_dict() for record in records],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _evidence_window() -> FactorEvidenceWindow:
    cohort_starts = [
        date(2024, 1, 1) + timedelta(days=index * 31)
        for index in range(8)
    ]
    return FactorEvidenceWindow(
        window_id="cohort-20260731",
        snapshot_id="snapshot-001",
        data_hash=_artifact_hash("data"),
        code_hash=_artifact_hash("code"),
        cost_hash=_artifact_hash("cost"),
        forward_cohort_ids=[f"cohort-{index}" for index in range(8)],
        month_end_rankic_dates=[],
        evaluation_hash=_artifact_hash("evaluation"),
        purge_days=30,
        embargo_days=30,
        walk_forward_fold_count=1,
        observed_at="2026-07-31T17:00:00+08:00",
        forward_cohorts=[
            {
                "cohort_id": f"cohort-{index}",
                "start": start.isoformat(),
                "end": (start + timedelta(days=29)).isoformat(),
                "horizon_days": 30,
            }
            for index, start in enumerate(cohort_starts)
        ],
    )


def _valid_days() -> list[str]:
    start = date(2026, 4, 3)
    return [(start + timedelta(days=index)).isoformat() for index in range(120)]


def _raw_replay(challenger: FactorRecord | None = None) -> dict:
    challenger = challenger or _record(
        "challenger",
        state=FactorLifecycleState.PRODUCTION_CANDIDATE,
        weight=0.0,
    )
    dates = _valid_days()
    arms: dict[str, dict] = {}
    for name, daily_return in (
        ("A", 0.00010),
        ("B", 0.00010),
        ("C", 0.00015),
        ("D", 0.00015),
    ):
        arms[name] = {
            "trading_dates": dates,
            "after_cost_daily_returns": [daily_return] * len(dates),
            "stage_artifact_hashes": {
                stage: _artifact_hash(f"{name}:{stage}")
                for stage in (
                    "quant",
                    "theme",
                    "bayesian",
                    "risk_guard",
                    "portfolio_constructor",
                )
            },
        }
    return {
        "schema_version": RAW_REPLAY_SCHEMA_VERSION,
        "as_of": "2026-07-31",
        "snapshot_evidence": {
            "schema_version": SNAPSHOT_EVIDENCE_SCHEMA_VERSION,
            "source": "strict_parquet_snapshot",
            "snapshot_id": "snapshot-001",
            "manifest_sha256": _artifact_hash("snapshot-manifest"),
            "latest_complete_trade_date": "2026-07-31",
            "valid_trading_days": dates,
        },
        "slot": FactorSlot(
            family="momentum",
            dominant_primitive_cluster="price_momentum",
            incumbent="incumbent",
            reserve="challenger",
        ).to_dict(),
        "challenger_record": challenger.to_dict(),
        "evidence_window": _evidence_window().to_dict(),
        "arms": arms,
        "health_evidence": {
            "failure_window_ids": ["mature-w1", "mature-w2", "mature-w3"],
            "artifact_hash": _artifact_hash("health-evidence"),
        },
        "selection_evidence": {
            "family_fdr_q_value": 0.08,
            "artifact_hash": _artifact_hash("selection-evidence"),
        },
        "limits_evidence": {
            key: {
                "measured": measured,
                "limit": limit,
                "artifact_hash": _artifact_hash(f"{key}-evidence"),
            }
            for key, measured, limit in (
                ("turnover", 0.20, 0.30),
                ("slippage", 0.002, 0.005),
                ("tail_risk", 0.01, 0.02),
            )
        },
        "walk_forward_evidence": {
            "purged": True,
            "purge_days": 30,
            "embargo_days": 30,
            "folds": [
                {
                    "fold_id": "1",
                    "train_end": "2026-05-31",
                    "validation_start": "2026-07-01",
                    "validation_end": "2026-07-20",
                    "evidence_hash": _artifact_hash("walk-forward-fold-1"),
                }
            ],
        },
    }


def _rehash_evidence(evidence: dict) -> None:
    evidence.pop("evidence_hash", None)
    evidence["evidence_hash"] = _hash(evidence)


def _registry_records() -> list[FactorRecord]:
    records = [
        _record(
            "incumbent",
            state=FactorLifecycleState.PRODUCTION_FACTOR,
            weight=0.10,
        ),
        _record(
            "challenger",
            state=FactorLifecycleState.PRODUCTION_CANDIDATE,
            weight=0.0,
        ),
    ]
    records.extend(
        _record(
            f"other-{index}",
            state=FactorLifecycleState.PRODUCTION_FACTOR,
            weight=0.10,
            family=f"family-{index}",
            cluster=f"cluster-{index}",
        )
        for index in range(5)
    )
    return records


def _plan_fixture(tmp_path: Path):
    registry_path = tmp_path / "mined_factors.json"
    _write_registry(registry_path, _registry_records())
    evidence = produce_governance_replay_evidence(_raw_replay())
    ledger_path = tmp_path / "state" / "monthly_budget.jsonl"
    wal_path = tmp_path / "wal" / "transition.json"
    plan, valid_days = build_registry_mutation_plan_from_evidence(
        registry_path=registry_path,
        evidence=evidence,
        wal_path=wal_path,
        budget_ledger_path=ledger_path,
    )
    return registry_path, evidence, plan, valid_days, ledger_path, wal_path


def test_versioned_contracts_are_content_addressed_and_slot_scoped(tmp_path) -> None:
    registry_path, evidence, plan, _, _, _ = _plan_fixture(tmp_path)

    assert plan.transition.slot.slot_id == "momentum::price_momentum"
    assert plan.protocol_hash == protocol_hash()
    assert len(protocol_hash()) == 64
    assert protocol_policy()["mutation"]["rollback_refunds_monthly_budget"] is False
    policy = protocol_policy()
    producer_contract = policy["canonical_replay_producer_contract"]
    assert producer_contract["authority"]["producer_implemented"] is True
    assert producer_contract["authority"]["production_apply_eligible"] is False
    assert "canonical_replay_producer_control" not in policy
    assert plan.evidence_hash == evidence["evidence_hash"]
    assert plan.to_dict()["mutation_plan_hash"] == plan.mutation_plan_hash
    assert plan.expected_registry_sha256 == (
        load_registry_snapshot_strict(registry_path).registry_sha256
    )
    tampered = plan.to_dict()
    tampered["expected_registry_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="mutation_plan_hash mismatch"):
        RegistryMutationPlan.from_dict(tampered)
    missing_evidence_hash = plan.to_dict()
    missing_evidence_hash["evidence_hash"] = ""
    with pytest.raises(ValueError, match="evidence_hash"):
        RegistryMutationPlan.from_dict(missing_evidence_hash)


@pytest.mark.parametrize(
    ("month_ends", "cohorts", "mature"),
    [(12, 0, True), (0, 8, True), (11, 7, False)],
)
def test_candidate_maturity_requires_12_month_ends_or_8_nonoverlap_30d_cohorts(
    month_ends: int,
    cohorts: int,
    mature: bool,
) -> None:
    starts = [date(2024, 1, 1) + timedelta(days=index * 31) for index in range(cohorts)]
    result = assess_candidate_maturity(
        month_end_rankic_dates=[
            f"2025-{index + 1:02d}-28" for index in range(month_ends)
        ],
        forward_cohorts=[
            {
                "cohort_id": f"c{index}",
                "start": starts[index].isoformat(),
                "end": (starts[index] + timedelta(days=29)).isoformat(),
                "horizon_days": 30,
            }
            for index in range(cohorts)
        ],
    )
    assert result["mature"] is mature


def test_bh_fdr_is_applied_within_family_at_q_point_10() -> None:
    rows = benjamini_hochberg_by_family(
        [
            {"name": "m1", "family": "momentum", "p_value": 0.01},
            {"name": "m2", "family": "momentum", "p_value": 0.04},
            {"name": "m3", "family": "momentum", "p_value": 0.20},
            {"name": "v1", "family": "value", "p_value": 0.08},
        ]
    )
    by_name = {row["name"]: row for row in rows}
    assert by_name["m1"]["fdr_passed"] is True
    assert by_name["m2"]["fdr_passed"] is True
    assert by_name["m3"]["fdr_passed"] is False
    assert by_name["v1"]["fdr_passed"] is True


def test_purged_walk_forward_requires_hashes_purge_and_30d_embargo() -> None:
    evidence = _raw_replay()["walk_forward_evidence"]
    assert validate_purged_walk_forward(evidence)["passed"] is True
    evidence["embargo_days"] = 29
    assert validate_purged_walk_forward(evidence)["passed"] is False


def test_paired_block_bootstrap_is_deterministic_and_hashes_actual_samples() -> None:
    deltas = [0.00005 + ((index % 5) - 2) * 0.000001 for index in range(120)]
    first = block_bootstrap_paired_delta_ci(deltas)
    assert first == block_bootstrap_paired_delta_ci(deltas)
    assert first["sample_count"] == 120
    assert first["ci_lower"] > 0.0
    assert first["sample_hash"] == _hash(deltas)


def test_data_blocked_neither_increments_nor_clears_failure_streak() -> None:
    state = advance_failure_streak([], "failure", "w1")
    state = advance_failure_streak(
        state["failure_window_ids"], "data_blocked", "w2"
    )
    assert state["failure_window_ids"] == ["w1"]
    state = advance_failure_streak(state["failure_window_ids"], "failure", "w1")
    assert state["failure_count"] == 1
    state = advance_failure_streak(state["failure_window_ids"], "healthy", "w3")
    assert state["failure_window_ids"] == []


def test_c_arm_ignores_forged_self_reported_delta_and_full_chain_booleans() -> None:
    evidence = produce_governance_replay_evidence(_raw_replay())
    evidence["paired_after_cost_daily_deltas"] = [-0.50] * 120
    evidence["paired_delta_ci95_lower"] = -0.50
    evidence["full_control_chain_evaluated"] = False
    _rehash_evidence(evidence)

    result = evaluate_c_arm(evidence)

    assert result["passed"] is True
    assert result["paired_after_cost_sample_count"] == 120
    assert result["annualized_net_excess_improvement"] == pytest.approx(0.0126)
    assert result["after_cost_paired_delta_ci95_lower"] > 0.0


def test_c_arm_recomputes_and_hash_checks_actual_a_c_after_cost_arrays() -> None:
    evidence = produce_governance_replay_evidence(_raw_replay())
    stale_arm_hash = copy.deepcopy(evidence)
    stale_arm_hash["arms"]["C"]["after_cost_daily_returns"] = [0.00005] * 120
    _rehash_evidence(stale_arm_hash)
    stale_result = evaluate_c_arm(stale_arm_hash)
    assert stale_result["passed"] is False
    assert any(
        blocker.startswith("canonical_governance_evidence_invalid")
        for blocker in stale_result["blockers"]
    )
    assert "arm_C_hash_mismatch" in stale_result["blockers"]

    negative_actual = copy.deepcopy(evidence)
    negative_actual["arms"]["C"]["after_cost_daily_returns"] = [0.00005] * 120
    negative_actual["arms"]["C"]["arm_hash"] = replay_arm_hash(
        negative_actual["arms"]["C"]
    )
    _rehash_evidence(negative_actual)
    negative_result = evaluate_c_arm(negative_actual)
    assert negative_result["passed"] is False
    assert "paired_delta_ci95_lower_not_positive" in negative_result["blockers"]
    assert negative_result["annualized_net_excess_improvement"] < 0.0


def test_only_canonical_producer_artifact_can_build_apply_plan(tmp_path) -> None:
    evidence = produce_governance_replay_evidence(_raw_replay())
    evidence_path = tmp_path / "evidence.json"
    write_governance_replay_evidence(evidence_path, evidence)
    assert load_governance_replay_evidence(evidence_path) == evidence
    assert verify_governance_replay_evidence(evidence) == evidence

    handwritten = tmp_path / "handwritten-plan.json"
    handwritten.write_text(
        json.dumps(
            {
                "transition_plan": {"passed": True},
                "valid_trading_days": ["2026-07-31"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema"):
        load_governance_replay_evidence(handwritten)


def test_retired_v2_governance_clis_are_removed() -> None:
    """Both v2 CLIs were pure refusal stubs; they are gone rather than kept as ones."""

    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    assert not (scripts_dir / "build_factor_governance_replay_evidence.py").exists()
    assert not (scripts_dir / "rollback_factor_governance_transition.py").exists()


def test_slot_risk_budget_retains_previous_when_evidence_is_insufficient() -> None:
    previous = {"incumbent": 0.18, "same_family": 0.16, "challenger": 0.0}
    preserved = build_slot_risk_budget(
        previous_weights=previous,
        family_by_factor={name: "momentum" for name in previous},
        incumbent="incumbent",
        challenger="challenger",
        evidence_sufficient=False,
    )
    assert preserved["weights"] == previous
    assert preserved["status"] == "retained_previous_weights"
    blocked = build_slot_risk_budget(
        previous_weights=previous,
        family_by_factor={name: "momentum" for name in previous},
        incumbent="incumbent",
        challenger="challenger",
        evidence_sufficient=True,
    )
    assert blocked["status"] == "blocked"
    assert "family_abs_weight_above_0.35" in blocked["blockers"]


def test_handwritten_normalized_replay_can_report_but_never_apply(
    tmp_path,
) -> None:
    registry_path, evidence, plan, valid_days, ledger_path, wal_path = (
        _plan_fixture(tmp_path)
    )
    before = load_registry_snapshot_strict(registry_path)

    producer = evidence["producer"]
    assert producer["artifact_bytes_readback_bound"] is False
    assert producer["production_apply_eligible"] is False

    report = apply_governed_transition(
        registry_path,
        plan,
        expected_protocol_hash=protocol_hash(),
        valid_trading_days=valid_days,
        write=False,
    )
    blocked_apply = apply_governed_transition(
        registry_path,
        plan,
        expected_protocol_hash=protocol_hash(),
        valid_trading_days=valid_days,
        write=True,
    )
    after = load_registry_snapshot_strict(registry_path)

    assert report["status"] == "report_only_ready"
    assert report["apply_requested"] is False
    assert blocked_apply["status"] == "blocked"
    assert CANONICAL_FULL_CHAIN_PRODUCER_BLOCKER in blocked_apply["blockers"]
    assert blocked_apply["apply_requested"] is True
    assert blocked_apply["changed_record_names"] == []
    assert after.registry_sha256 == before.registry_sha256
    assert not wal_path.exists()
    assert not ledger_path.exists()


def _historical_inverse_wal_fixture(tmp_path: Path) -> dict[str, object]:
    registry_path = tmp_path / "historical_registry.json"
    record = _record(
        "historical-incumbent",
        state=FactorLifecycleState.PRODUCTION_FACTOR,
        weight=0.10,
    )
    _write_registry(registry_path, [record])
    before = load_registry_snapshot_strict(registry_path)
    transition_hash = _artifact_hash("historical-transition")
    mutation_plan_hash = _artifact_hash("historical-mutation-plan")
    evidence_hash = _artifact_hash("historical-evidence")
    ledger_path = tmp_path / "state" / "historical_budget.jsonl"
    reservation = reserve_monthly_mutation_budget(
        ledger_path,
        month="2026-07",
        transition_id="historical-transition",
        transition_hash=transition_hash,
        mutation_plan_hash=mutation_plan_hash,
        evidence_hash=evidence_hash,
        before_registry_sha256=before.registry_sha256,
    )
    updated = FactorRecord.from_dict(record.to_dict())
    updated.weight = 0.09
    wal_path = tmp_path / "wal" / "historical-transition.json"
    applied = apply_factor_record_patch(
        registry_path,
        {record.name: updated},
        expected_registry_sha256=before.registry_sha256,
        expected_record_sha256s={
            record.name: before.record_sha256s[record.name]
        },
        mutation_id="historical-transition",
        reason="historical valid v2 transition fixture",
        manifest_metadata={
            "protocol_version": "v2",
            "protocol_hash": protocol_hash(),
            "transition_hash": transition_hash,
            "mutation_plan_hash": mutation_plan_hash,
            "evidence_hash": evidence_hash,
            "mutation_budget_ledger_path": str(ledger_path),
            "mutation_budget_reservation": reservation,
        },
        journal_path=wal_path,
        write=True,
    )
    return {
        "registry_path": registry_path,
        "before": before,
        "after": load_registry_snapshot_strict(registry_path),
        "ledger_path": ledger_path,
        "wal_path": wal_path,
        "applied": applied,
    }


def test_historical_inverse_wal_state_exhausts_the_monthly_budget(tmp_path) -> None:
    fixture = _historical_inverse_wal_fixture(tmp_path)
    registry_path = fixture["registry_path"]
    ledger_path = fixture["ledger_path"]
    wal_path = fixture["wal_path"]
    after_apply = fixture["after"]
    applied = fixture["applied"]
    assert isinstance(registry_path, Path)
    assert isinstance(ledger_path, Path)
    assert isinstance(wal_path, Path)
    ledger_before_rollback = ledger_path.read_bytes()
    wal_sha = hashlib.sha256(wal_path.read_bytes()).hexdigest()
    common_rollback_args = [
        "--registry-path",
        str(registry_path),
        "--inverse-wal",
        str(wal_path),
        "--mutation-budget-ledger",
        str(ledger_path),
        "--protocol-version",
        "v2",
        "--expected-protocol-hash",
        protocol_hash(),
        "--expected-current-registry-sha256",
        after_apply.registry_sha256,
        "--expected-inverse-wal-sha256",
        wal_sha,
        "--expected-transition-hash",
        applied["transition_hash"],
        "--expected-mutation-plan-hash",
        applied["mutation_plan_hash"],
        "--expected-evidence-hash",
        applied["evidence_hash"],
    ]

    assert monthly_mutation_budget_blockers(ledger_path, "2026-07") == [
        "monthly_transition_budget_exhausted"
    ]
    assert load_registry_snapshot_strict(registry_path).registry_sha256 == (
        after_apply.registry_sha256
    )
    assert len(load_mutation_budget_ledger(ledger_path)) == 1


def test_non_month_end_and_snapshot_calendar_tampering_fail_closed(tmp_path) -> None:
    registry_path, _, plan, valid_days, _, _ = _plan_fixture(tmp_path)
    wrong_days = valid_days[:-1]
    result = apply_governed_transition(
        registry_path,
        plan,
        expected_protocol_hash=protocol_hash(),
        valid_trading_days=wrong_days,
        write=False,
    )
    assert result["status"] == "blocked"
    assert "valid_trading_days_not_from_evidence_artifact" in result["blockers"]
    assert "not_last_valid_trading_day" in result["blockers"]


@pytest.mark.parametrize(
    ("before_weights", "after_weights", "expected_blocker"),
    [
        (
            {"incumbent": 0.09, "challenger": 0.0},
            {"incumbent": 0.0, "challenger": 0.10},
            "incumbent_before_weight_mismatch",
        ),
        (
            {"incumbent": 0.10, "challenger": 0.0},
            {"incumbent": 0.0, "challenger": 0.09},
            "challenger_after_weight_mismatch",
        ),
    ],
)
def test_transition_plan_cannot_misstate_before_or_after_weights(
    tmp_path,
    before_weights,
    after_weights,
    expected_blocker,
) -> None:
    registry_path, _, plan, valid_days, _, _ = _plan_fixture(tmp_path)
    payload = plan.to_dict()
    payload["transition"]["before_weights"] = before_weights
    payload["transition"]["after_weights"] = after_weights
    payload["transition"].pop("transition_hash", None)
    payload.pop("mutation_plan_hash", None)
    misstated = RegistryMutationPlan.from_dict(payload)
    result = apply_governed_transition(
        registry_path,
        misstated,
        expected_protocol_hash=protocol_hash(),
        valid_trading_days=valid_days,
        write=False,
    )
    assert result["status"] == "blocked"
    assert expected_blocker in result["blockers"]


def test_zero_production_factors_is_governance_blocked_without_legacy_proxy() -> None:
    registry = MinedFactorRegistry.from_records(
        [
            _record(
                "candidate",
                state=FactorLifecycleState.PRODUCTION_CANDIDATE,
                weight=0.0,
            )
        ]
    )
    status = governance_runtime_status(registry)
    assert status["status"] == "governance_blocked"
    assert status["factor_mode"] == "governance_blocked"
    assert status["confidence_multiplier"] == 0.0
    assert status["legacy_fallback_allowed"] is False
    assert status["production_factor_count"] == 0
    assert "no_selectable_production_factors" in status["blockers"]


def _runtime_ready_registry(monkeypatch) -> MinedFactorRegistry:
    import quant_investor.factors.governance_protocol_v2 as protocol_module

    records = [
        _record(
            f"factor-{index}",
            state=FactorLifecycleState.PRODUCTION_FACTOR,
            weight=0.20,
            family=f"family-{index}",
            cluster=f"cluster-{index}",
        )
        for index in range(5)
    ]
    monkeypatch.setattr(
        protocol_module,
        "canonical_replay_producer_control",
        lambda: {
            "producer_implemented": True,
            "local_bytes_readback_verified": True,
            "canonical_producer_authenticated": True,
            "production_apply_authorized": True,
            "production_apply_eligible": True,
            "blocker": "",
        },
    )
    monkeypatch.setattr(
        protocol_module,
        "validate_production_runtime_contracts",
        lambda records, metadata: {
            "status": "ready",
            "contracts": {
                record.name: {
                    "required_columns": ["trade_date", "adj_close"],
                    "lookback_rows": 2,
                    "gate2_min_coverage_rate": 1.0,
                    "min_cross_section": 20,
                }
                for record in records
            },
            "contracts_sha256": _artifact_hash("contracts"),
            "implementation_code_sha256s": {
                record.name: _artifact_hash(f"code-{record.name}")
                for record in records
            },
            "blockers": [],
        },
    )
    monkeypatch.setattr(
        protocol_module,
        "validate_quant_production_activation",
        lambda *args, **kwargs: {"status": "ready", "blockers": []},
    )
    registry = MinedFactorRegistry.from_records(records)
    manifest = registry.selectable_manifest()
    registry.metadata = {
        **manifest,
        "factor_governance_protocol_version": "v2",
        "factor_governance_protocol_hash": protocol_hash(),
        "factor_governance_last_evidence_hash": _artifact_hash("evidence"),
        "factor_governance_last_evaluation_hash": _artifact_hash("evaluation"),
        "factor_governance_evidence_schema": "factor-governance-replay-evidence.v2",
        "factor_governance_production_apply_eligible": True,
        "factor_governance_production_apply_blocker": "",
        "strict_loader": True,
    }
    return registry


def test_runtime_requires_protocol_manifest_slots_budgets_and_canonical_evidence(
    monkeypatch,
) -> None:
    registry = _runtime_ready_registry(monkeypatch)
    status = governance_runtime_status(registry)

    assert status["status"] == "ready"
    assert status["confidence_multiplier"] == 1.0
    assert status["blockers"] == []

    registry.metadata["production_factor_count"] = 4
    drifted = governance_runtime_status(registry)
    assert drifted["status"] == "governance_blocked"
    assert "registry_production_factor_count_mismatch" in drifted["blockers"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_runtime_rejects_non_finite_selectable_factor_weight(
    monkeypatch,
    value: float,
) -> None:
    registry = _runtime_ready_registry(monkeypatch)
    registry.factors[0].weight = value

    status = governance_runtime_status(registry)

    assert status["status"] == "governance_blocked"
    assert "factor_weight_non_finite:factor-0" in status["blockers"]
    assert "production_factor_total_abs_weight_non_finite" in status["blockers"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_runtime_rejects_non_finite_selectable_factor_direction(
    monkeypatch,
    value: float,
) -> None:
    registry = _runtime_ready_registry(monkeypatch)
    registry.factors[0].direction = value

    status = governance_runtime_status(registry)

    assert status["status"] == "governance_blocked"
    assert "factor_direction_non_finite:factor-0" in status["blockers"]


def test_runtime_rejects_non_finite_total_abs_weight(monkeypatch) -> None:
    registry = _runtime_ready_registry(monkeypatch)
    for record in registry.factors:
        record.weight = 1e308

    status = governance_runtime_status(registry)

    assert status["status"] == "governance_blocked"
    assert "production_factor_total_abs_weight_non_finite" in status["blockers"]
    assert not any(
        blocker.startswith("factor_weight_non_finite:")
        for blocker in status["blockers"]
    )


def test_runtime_rejects_effectively_zero_weight_or_direction(monkeypatch) -> None:
    registry = _runtime_ready_registry(monkeypatch)
    registry.factors[0].weight = 1e-18
    registry.factors[1].direction = 0.0

    status = governance_runtime_status(registry)

    assert status["status"] == "governance_blocked"
    assert "factor_weight_zero_or_negligible:factor-0" in status["blockers"]
    assert "factor_direction_zero_or_negligible:factor-1" in status["blockers"]


def test_current_baseline_registry_is_not_v2_runtime_ready() -> None:
    registry = MinedFactorRegistry.load()

    status = governance_runtime_status(registry)

    assert status["status"] == "governance_blocked"
    assert status["confidence_multiplier"] == 0.0
    assert CANONICAL_FULL_CHAIN_PRODUCER_BLOCKER in status["blockers"]
    assert "registry_protocol_hash_mismatch" in status["blockers"]
