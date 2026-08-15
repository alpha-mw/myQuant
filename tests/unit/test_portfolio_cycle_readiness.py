from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

import quant_investor.portfolio_cycle.readiness as readiness_module
from quant_investor.portfolio_cycle.readiness import (
    ALL_BLOCKER_CODES,
    AUTHORITY_FLAGS,
    DECISION_INPUT_READINESS_SCHEMA_ID,
    GATE_NAMES,
    PHASE_CAPABILITY,
    POST_RUN_STATES,
    PRE_RUN_STATES,
    PUBLIC_CYCLE_STATUS_SCHEMA_ID,
    GateEvidence,
    MainlineBlocker,
    MainlineResolution,
    ReadinessBlocker,
    build_decision_input_readiness,
    build_public_cycle_status,
    derive_decision_input_readiness,
)
from quant_investor.portfolio_cycle.contracts import PortfolioCycleError
from quant_investor.portfolio_cycle.contracts import (
    ArtifactRef,
    HOLDINGS_ACCOUNTING_POLICY_SCHEMA_ID,
    HOLDINGS_LEDGER_SCHEMA_ID,
    HOLDINGS_MANIFEST_SCHEMA_ID,
    HOLDINGS_POINTER_SCHEMA_ID,
    HOLDINGS_PRICE_SOURCE_SCHEMA_ID,
    IDENTITY_DECLARATION_SCHEMA_ID,
    MoneyTotals,
    PROTOCOL,
    VerifiedHoldingsBaseline,
    VerifiedStrategyIdentity,
)

STRATEGY = "cn-mainline"


def _ref(name: str, *, schema_id: str | None = None) -> dict[str, str]:
    return {
        "schema_id": schema_id or f"myquant.test.{name}",
        "relative_path": f"data/private/tests/{name}.json",
        "byte_sha256": (name.encode("ascii").hex() + "0" * 64)[:64],
    }


def _artifact_ref(name: str, *, schema_id: str) -> ArtifactRef:
    return ArtifactRef(**_ref(name, schema_id=schema_id))


def _identity(*, strategy_id: str = STRATEGY) -> VerifiedStrategyIdentity:
    return VerifiedStrategyIdentity(
        verified=True,
        declaration_ref=_artifact_ref("identity", schema_id=IDENTITY_DECLARATION_SCHEMA_ID),
        historical_label="aggressive_tech_manufacturing",
        canonical_strategy_id=strategy_id,
        declared_by="maxwell",
        declared_at="2026-08-06T00:00:00Z",
        authority_kind="owner_declaration",
        provenance="explicit-owner-input",
    )


def _holdings(*, strategy_id: str = STRATEGY) -> VerifiedHoldingsBaseline:
    zero = Decimal("0.0000")
    return VerifiedHoldingsBaseline(
        verified=True,
        canonical_strategy_id=strategy_id,
        account_id="paper-account",
        currency="CNY",
        trade_date="2026-08-05",
        as_of="2026-08-05T07:00:00Z",
        valuation_at="2026-08-05T07:00:00Z",
        decision_cutoff="2026-08-05T07:00:00Z",
        pointer_updated_at="2026-08-05T07:01:00Z",
        pointer_ref=_artifact_ref("holdings-pointer", schema_id=HOLDINGS_POINTER_SCHEMA_ID),
        manifest_ref=_artifact_ref("holdings-manifest", schema_id=HOLDINGS_MANIFEST_SCHEMA_ID),
        accounting_policy_ref=_artifact_ref(
            "accounting-policy", schema_id=HOLDINGS_ACCOUNTING_POLICY_SCHEMA_ID
        ),
        price_source_ref=_artifact_ref("price-source", schema_id=HOLDINGS_PRICE_SOURCE_SCHEMA_ID),
        ledger_ref=_artifact_ref("holdings-ledger", schema_id=HOLDINGS_LEDGER_SCHEMA_ID),
        totals=MoneyTotals(
            contributed_capital=zero,
            cash=zero,
            cost_basis=zero,
            market_value=zero,
            unrealized_pnl=zero,
            realized_pnl=zero,
            nav=zero,
        ),
        positions=(),
    )


def _gates(*, verified: bool, override: str | None = None) -> dict[str, GateEvidence]:
    return {
        name: GateEvidence(
            verified=(not verified if name == override else verified),
            ref=(
                _ref(f"gate-{name}") if (not verified if name == override else verified) else None
            ),
        )
        for name in GATE_NAMES
    }


def _pre(
    *,
    identity: object | None = None,
    holdings: object | None = None,
    gates: dict[str, GateEvidence] | None = None,
    synthetic_only: bool = True,
) -> dict[str, object]:
    return build_decision_input_readiness(
        identity=_identity() if identity is None else identity,
        holdings=_holdings() if holdings is None else holdings,
        gates=_gates(verified=False) if gates is None else gates,
        synthetic_only=synthetic_only,
    )


def _active_resolution(public_run: dict[str, object] | None = None) -> MainlineResolution:
    dto: dict[str, object] = {
        "state": "ACTIVE",
        "canonical_strategy_id": STRATEGY,
        "active_pointer_ref": _ref("active-pointer"),
        "mainline_run_ref": _ref("mainline-run"),
        "formal_output_ref": _ref("formal-output"),
        "portfolio_output_ref": _ref("portfolio-output"),
        "source_closure_ref": _ref("source-closure"),
    }
    if public_run is not None:
        dto = public_run
    return MainlineResolution("ACTIVE", dto)


def test_frozen_schema_states_and_blocker_vocabulary() -> None:
    assert DECISION_INPUT_READINESS_SCHEMA_ID == (
        "myquant.portfolio-cycle.decision-input-readiness"
    )
    assert PUBLIC_CYCLE_STATUS_SCHEMA_ID == "myquant.portfolio-cycle.public-cycle-status"
    assert PRE_RUN_STATES == {"BLOCKED", "FOUNDATION_VALIDATED"}
    assert POST_RUN_STATES == {
        "BLOCKED",
        "PUBLIC_CLOSURE_ACTIVE_FOUNDATION_ONLY",
    }
    assert "FULL_CYCLE_CLOSED" not in PRE_RUN_STATES | POST_RUN_STATES
    assert {blocker.value for blocker in ReadinessBlocker} <= ALL_BLOCKER_CODES
    assert {
        f"MAINLINE_BLOCKED:{blocker.value}" for blocker in MainlineBlocker
    } <= ALL_BLOCKER_CODES


def test_missing_and_invalid_foundation_inputs_are_distinct() -> None:
    missing = build_decision_input_readiness(
        identity=None,
        holdings=None,
        synthetic_only=False,
    )
    assert missing["state"] == "BLOCKED"
    assert "STRATEGY_ID_UNCONFIRMED" in missing["blockers"]
    assert "HOLDINGS_BASELINE_UNAVAILABLE" in missing["blockers"]
    assert missing["canonical_strategy_id"] is None
    assert missing["decision_cutoff"] is None

    invalid = build_decision_input_readiness(
        identity=None,
        holdings=None,
        identity_invalid=True,
        holdings_invalid=True,
        synthetic_only=False,
    )
    assert "IDENTITY_DECLARATION_INVALID" in invalid["blockers"]
    assert "HOLDINGS_BASELINE_INVALID" in invalid["blockers"]
    assert "STRATEGY_ID_UNCONFIRMED" not in invalid["blockers"]
    assert "HOLDINGS_BASELINE_UNAVAILABLE" not in invalid["blockers"]


def test_verified_identity_and_holdings_only_validate_foundation() -> None:
    result = _pre(synthetic_only=False)
    assert result["state"] == "FOUNDATION_VALIDATED"
    assert result["foundation_validated"] is True
    assert result["business_ready"] is False
    assert result["business_cycle_closed"] is False
    assert result["phase_capability"] == PHASE_CAPABILITY
    assert result["canonical_strategy_id"] == STRATEGY
    assert result["decision_cutoff"] == "2026-08-05T07:00:00Z"
    assert result["blockers"] == sorted(set(result["blockers"]))
    assert "STRICT_CN_DATA_UNVERIFIED" in result["blockers"]
    assert "CAPABILITY_BLOCKED_MAINLINE_PUBLISHER" in result["blockers"]
    assert "PAPER_SIMULATION_UNAVAILABLE" in result["blockers"]
    assert "LEARNING_RUNTIME_UNAVAILABLE" in result["blockers"]


@pytest.mark.parametrize(
    ("gate_name", "expected_blocker"),
    [
        ("strict_cn_data", "STRICT_CN_DATA_UNVERIFIED"),
        ("pit", "PIT_UNVERIFIED"),
        ("fundamental", "FUNDAMENTAL_UNVERIFIED"),
        ("macro", "MACRO_UNVERIFIED"),
        ("release_calendar", "RELEASE_CALENDAR_UNVERIFIED"),
        ("factor_active_set", "FACTOR_ACTIVE_SET_UNAVAILABLE"),
        ("risk_policy", "RISK_POLICY_UNAVAILABLE"),
        ("portfolio_policy", "PORTFOLIO_POLICY_UNAVAILABLE"),
        (
            "mainline_publisher",
            "CAPABILITY_BLOCKED_MAINLINE_PUBLISHER",
        ),
        ("paper_simulation", "PAPER_SIMULATION_UNAVAILABLE"),
        ("learning_runtime", "LEARNING_RUNTIME_UNAVAILABLE"),
    ],
)
def test_each_explicit_gate_has_one_fail_closed_blocker(
    gate_name: str, expected_blocker: str
) -> None:
    result = _pre(gates=_gates(verified=True, override=gate_name))
    assert result["state"] == "FOUNDATION_VALIDATED"
    assert result["business_ready"] is False
    assert result["blockers"] == [expected_blocker]


def test_all_explicit_gates_still_do_not_claim_business_readiness() -> None:
    result = _pre(gates=_gates(verified=True))
    assert result["state"] == "FOUNDATION_VALIDATED"
    assert result["blockers"] == []
    assert result["business_ready"] is False
    assert result["business_cycle_closed"] is False
    assert result["operational_authority"] is False
    assert set(result["authority"].values()) == {False}


def test_identity_holdings_lineage_mismatch_fails_closed() -> None:
    result = build_decision_input_readiness(
        identity=_identity(),
        holdings=_holdings(strategy_id="other-strategy"),
        gates=_gates(verified=True),
        synthetic_only=False,
    )
    assert result["state"] == "BLOCKED"
    assert result["holdings"] is None
    assert result["blockers"] == ["HOLDINGS_BASELINE_INVALID"]


def test_requested_cutoff_mismatch_invalidates_holdings_without_substitution() -> None:
    result = build_decision_input_readiness(
        identity=_identity(),
        holdings=_holdings(),
        gates=_gates(verified=True),
        decision_cutoff="2026-08-05T08:00:00Z",
        synthetic_only=False,
    )
    assert result["state"] == "BLOCKED"
    assert result["decision_cutoff"] == "2026-08-05T08:00:00Z"
    assert result["decision_cutoff_verified"] is False
    assert result["holdings"] is None
    assert result["blockers"] == ["HOLDINGS_BASELINE_INVALID"]


def test_input_contract_rejects_contradictory_or_inexact_gates() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        build_decision_input_readiness(
            identity=_identity(),
            holdings=_holdings(),
            identity_invalid=True,
            synthetic_only=False,
        )

    gates = _gates(verified=False)
    gates["pit"] = GateEvidence(False, _ref("forbidden-ref"))
    with pytest.raises(ValueError, match="must not carry"):
        _pre(gates=gates)

    gates = _gates(verified=False)
    gates["pit"] = GateEvidence(True, None)
    with pytest.raises(ValueError, match="requires an exact ref"):
        _pre(gates=gates)

    with pytest.raises(ValueError, match="exact Phase 1 gate names"):
        _pre(gates={"pit": GateEvidence(False)})


def test_read_only_orchestration_keeps_unbound_cutoff_as_context_only(
    tmp_path: Path,
) -> None:
    missing_workspace = tmp_path / "missing-workspace"
    result = derive_decision_input_readiness(
        missing_workspace,
        strategy_id=None,
        decision_cutoff="2026-08-05T08:00:00Z",
        synthetic_only=False,
    )
    assert not missing_workspace.exists()
    assert result["state"] == "BLOCKED"
    assert result["decision_cutoff"] == "2026-08-05T08:00:00Z"
    assert result["decision_cutoff_verified"] is False
    assert result["canonical_strategy_id"] is None
    assert "STRATEGY_ID_UNCONFIRMED" in result["blockers"]
    assert "HOLDINGS_BASELINE_UNAVAILABLE" in result["blockers"]


def test_read_only_orchestration_requires_path_sha_pairs_and_canonical_cutoff(
    tmp_path: Path,
) -> None:
    with pytest.raises(PortfolioCycleError) as unpaired:
        derive_decision_input_readiness(
            tmp_path,
            identity_path="governance/identity.json",
            decision_cutoff="2026-08-05T08:00:00Z",
        )
    assert unpaired.value.code == "PORTFOLIO_CYCLE_ARGUMENTS_INVALID"

    with pytest.raises(PortfolioCycleError) as bad_cutoff:
        derive_decision_input_readiness(
            tmp_path,
            decision_cutoff="2026-08-05T08:00:00+08:00",
        )
    assert bad_cutoff.value.code == "PORTFOLIO_CYCLE_ARGUMENTS_INVALID"


@pytest.mark.parametrize(
    "code",
    [
        "PORTFOLIO_CYCLE_STORAGE_SECURITY",
        "PORTFOLIO_CYCLE_STABLE_READ_FAILED",
        "PORTFOLIO_CYCLE_READ_BOUND_EXCEEDED",
    ],
)
def test_orchestration_reraises_security_and_read_integrity_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code: str,
) -> None:
    def fail_identity(*args: object, **kwargs: object) -> object:
        raise PortfolioCycleError(code, "exact read failed")

    monkeypatch.setattr(readiness_module, "resolve_strategy_identity", fail_identity)
    with pytest.raises(PortfolioCycleError) as exc_info:
        derive_decision_input_readiness(
            tmp_path,
            identity_path="governance/identity.json",
            identity_sha256="0" * 64,
            decision_cutoff="2026-08-05T08:00:00Z",
        )
    assert exc_info.value.code == code


def test_orchestration_maps_ordinary_exact_artifact_failure_to_invalid_blocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_identity(*args: object, **kwargs: object) -> object:
        raise PortfolioCycleError("PORTFOLIO_CYCLE_BYTE_SHA_MISMATCH", "identity bytes changed")

    monkeypatch.setattr(readiness_module, "resolve_strategy_identity", fail_identity)
    result = derive_decision_input_readiness(
        tmp_path,
        identity_path="governance/identity.json",
        identity_sha256="0" * 64,
        decision_cutoff="2026-08-05T08:00:00Z",
    )
    assert result["state"] == "BLOCKED"
    assert "IDENTITY_DECLARATION_INVALID" in result["blockers"]


def test_output_is_deterministic_and_blockers_are_sorted_unique() -> None:
    normal = _gates(verified=False)
    reversed_order = dict(reversed(list(normal.items())))
    first = _pre(gates=normal)
    second = _pre(gates=reversed_order)
    assert first == second
    assert first["blockers"] == sorted(first["blockers"])
    assert len(first["blockers"]) == len(set(first["blockers"]))
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )


@pytest.mark.parametrize("synthetic_only", [False, True])
def test_synthetic_flag_never_upgrades_authority(synthetic_only: bool) -> None:
    pre = _pre(synthetic_only=synthetic_only)
    post = build_public_cycle_status(
        decision_input_readiness=pre,
        mainline_resolution=_active_resolution(),
        synthetic_only=synthetic_only,
    )
    for result in (pre, post):
        assert result["synthetic_only"] is synthetic_only
        assert result["operational_authority"] is False
        assert result["authority"] == AUTHORITY_FLAGS
        assert set(result["authority"].values()) == {False}
        assert "FULL_CYCLE_CLOSED" not in json.dumps(result, sort_keys=True)


def test_active_public_closure_is_still_foundation_only_and_no_business_cycle() -> None:
    pre = _pre(gates=_gates(verified=False))
    post = build_public_cycle_status(
        decision_input_readiness=pre,
        mainline_resolution=_active_resolution(),
        synthetic_only=True,
    )
    assert post["schema_id"] == PUBLIC_CYCLE_STATUS_SCHEMA_ID
    assert post["state"] == "PUBLIC_CLOSURE_ACTIVE_FOUNDATION_ONLY"
    assert post["public_closure_active"] is True
    assert post["foundation_validated"] is True
    assert post["business_ready"] is False
    assert post["business_cycle_closed"] is False
    assert post["paper_state"] == "UNAVAILABLE"
    assert post["learning_state"] == "UNAVAILABLE"
    assert post["blockers"] == sorted(set(post["blockers"]))
    assert post["blockers"].count("PAPER_SIMULATION_UNAVAILABLE") == 1
    assert post["blockers"].count("LEARNING_RUNTIME_UNAVAILABLE") == 1


def test_active_mainline_cannot_override_blocked_foundation() -> None:
    pre = build_decision_input_readiness(
        identity=None,
        holdings=None,
        gates=_gates(verified=True),
        synthetic_only=True,
    )
    post = build_public_cycle_status(
        decision_input_readiness=pre,
        mainline_resolution=_active_resolution(),
        synthetic_only=True,
    )
    assert post["state"] == "BLOCKED"
    assert post["public_closure_active"] is True
    assert post["business_cycle_closed"] is False
    assert "STRATEGY_ID_UNCONFIRMED" in post["blockers"]
    assert "HOLDINGS_BASELINE_UNAVAILABLE" in post["blockers"]


def test_uninitialized_and_exact_blocked_mainline_states_are_preserved() -> None:
    pre = _pre(gates=_gates(verified=True))
    uninitialized = build_public_cycle_status(
        decision_input_readiness=pre,
        mainline_resolution=MainlineResolution("MAINLINE_UNINITIALIZED", None),
        synthetic_only=True,
    )
    assert uninitialized["state"] == "BLOCKED"
    assert "MAINLINE_UNINITIALIZED" in uninitialized["blockers"]

    blocked_code = "MAINLINE_BLOCKED:SOURCE_CLOSURE_INVALID"
    blocked = build_public_cycle_status(
        decision_input_readiness=pre,
        mainline_resolution=MainlineResolution(blocked_code, None),
        synthetic_only=True,
    )
    assert blocked["state"] == "BLOCKED"
    assert blocked_code in blocked["blockers"]
    assert blocked["mainline_derived_state"] == blocked_code


def test_active_mainline_projection_is_read_only_and_still_insufficient(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "input-marker"
    marker.write_bytes(b"immutable")
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    resolution = _active_resolution()
    assert resolution.is_active

    post = build_public_cycle_status(
        decision_input_readiness=_pre(),
        mainline_resolution=resolution,
        synthetic_only=True,
    )
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert post["state"] == "PUBLIC_CLOSURE_ACTIVE_FOUNDATION_ONLY"
    assert post["business_cycle_closed"] is False
    assert post["operational_authority"] is False
    assert post["authority"]["mainline_write"] is False
    assert post["authority"]["paper_ledger_write_authorized"] is False


def test_post_run_rejects_noncanonical_pre_status_or_synthetic_mismatch() -> None:
    pre = _pre(synthetic_only=True)
    with pytest.raises(ValueError, match="not a Phase 1 status"):
        build_public_cycle_status(
            decision_input_readiness=pre,
            mainline_resolution=_active_resolution(),
            synthetic_only=False,
        )

    forged = dict(pre)
    forged["business_ready"] = True
    with pytest.raises(ValueError, match="not a Phase 1 status"):
        build_public_cycle_status(
            decision_input_readiness=forged,
            mainline_resolution=_active_resolution(),
            synthetic_only=True,
        )


def test_post_run_rejects_forged_foundation_without_exact_bindings() -> None:
    forged = {
        "schema_id": DECISION_INPUT_READINESS_SCHEMA_ID,
        "protocol": PROTOCOL,
        "state": "FOUNDATION_VALIDATED",
        "phase_capability": PHASE_CAPABILITY,
        "synthetic_only": True,
        "read_only": True,
        "operational_authority": False,
        "foundation_validated": True,
        "business_ready": False,
        "business_cycle_closed": False,
        "canonical_strategy_id": STRATEGY,
        "decision_cutoff": None,
        "decision_cutoff_verified": False,
        "identity": None,
        "holdings": None,
        "gates": {name: {"verified": True, "ref": _ref(name)} for name in GATE_NAMES},
        "blockers": [],
        "authority": dict(AUTHORITY_FLAGS),
    }
    with pytest.raises(ValueError, match="derived fields are inconsistent"):
        build_public_cycle_status(
            decision_input_readiness=forged,
            mainline_resolution=_active_resolution(),
            synthetic_only=True,
        )


def test_pre_run_rejects_duck_typed_or_malformed_verified_results() -> None:
    class Duck:
        verified = True

    duck = build_decision_input_readiness(
        identity=Duck(),
        holdings=_holdings(),
        synthetic_only=False,
    )
    assert duck["state"] == "BLOCKED"
    assert "IDENTITY_DECLARATION_INVALID" in duck["blockers"]

    bad_currency = _holdings()
    object.__setattr__(bad_currency, "currency", "USD")
    currency_result = build_decision_input_readiness(
        identity=_identity(),
        holdings=bad_currency,
        synthetic_only=False,
    )
    assert currency_result["state"] == "BLOCKED"
    assert "HOLDINGS_BASELINE_INVALID" in currency_result["blockers"]

    bad_ref = _identity()
    object.__setattr__(
        bad_ref,
        "declaration_ref",
        _artifact_ref("identity", schema_id="myquant.test.wrong"),
    )
    ref_result = build_decision_input_readiness(
        identity=bad_ref,
        holdings=_holdings(),
        synthetic_only=False,
    )
    assert ref_result["state"] == "BLOCKED"
    assert "IDENTITY_DECLARATION_INVALID" in ref_result["blockers"]
