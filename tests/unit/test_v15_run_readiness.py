from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from quant_investor.monitoring.v15_run_readiness import (
    build_v15_run_readiness,
    canonical_sha256,
    load_v15_run_readiness,
    write_v15_run_readiness,
)


def _macro_evidence(*, session_lag: int = 0) -> dict[str, object]:
    evidence: dict[str, object] = {
        "schema_version": "macro-readiness-evidence.v1",
        "market": "CN",
        "macro_logical_date": "2026-07-15",
        "target_session_date": "2026-07-15",
        "target_decision_cutoff_at": "2026-07-15T07:00:00+00:00",
        "max_session_lag": 2,
        "macro_release_calendar_generation_id": "calendar-generation",
        "macro_release_calendar_pointer_sha256": "1" * 64,
        "macro_release_calendar_manifest_sha256": "2" * 64,
        "macro_release_calendar_semantic_sha256": "3" * 64,
        "macro_release_calendar_registry_sha256": "4" * 64,
        "macro_release_calendar_plan_sha256": "5" * 64,
        "macro_release_calendar_capture_manifest_sha256": "6" * 64,
        "macro_release_calendar_market_open_days_sha256": "7" * 64,
        "macro_release_calendar_critical_policy_sha256": "8" * 64,
        "validated_release_calendar_ancestry": [],
        "evaluation": {
            "ready": True,
            "session_lag": {
                "ready": True,
                "session_lag": session_lag,
                "macro_logical_date": "2026-07-15",
                "target_session_date": "2026-07-15",
                "blockers": [],
            },
            "critical_event_gap": {
                "ready": True,
                "window_start_exclusive": "2026-07-15T07:00:00+00:00",
                "window_end_inclusive": "2026-07-15T07:00:00+00:00",
                "relevant_event_ids": [],
                "resolved_event_ids": [],
                "blocking_event_ids": [],
                "blockers": [],
            },
            "blockers": [],
        },
    }
    return _resign_macro_evidence(evidence)


def _resign_macro_evidence(
    evidence: dict[str, object],
) -> dict[str, object]:
    semantic_payload = dict(evidence)
    semantic_payload.pop("semantic_sha256", None)
    evidence["semantic_sha256"] = canonical_sha256(semantic_payload)
    return evidence


def _branches(
    *,
    ready: bool = True,
    macro_evidence: dict[str, object] | None = None,
    macro_binding: str | None = None,
):
    branches = {
        name: {
            "status": "pass" if ready else "block",
            "blockers": [] if ready else ["missing"],
        }
        for name in ("quant", "fundamental", "macro")
    }
    evidence = macro_evidence or _macro_evidence()
    identity_fields = (
        "macro_release_calendar_generation_id",
        "macro_release_calendar_pointer_sha256",
        "macro_release_calendar_manifest_sha256",
        "macro_release_calendar_semantic_sha256",
        "macro_release_calendar_registry_sha256",
        "macro_release_calendar_plan_sha256",
        "macro_release_calendar_capture_manifest_sha256",
        "macro_release_calendar_market_open_days_sha256",
        "macro_release_calendar_critical_policy_sha256",
    )
    binding = (
        macro_binding
        if macro_binding is not None
        else str(evidence.get("semantic_sha256") or "")
    )
    branches["macro"]["metadata"] = {
        "canonical_identity": {
            **{
                field_name: evidence.get(field_name, "")
                for field_name in identity_fields
            },
            "macro_readiness_evidence_semantic_sha256": binding,
        },
        "macro_readiness_evidence": evidence,
    }
    return branches


def _payload(**overrides):
    portfolio = {"valid": True, "target_weights": {"000001.SZ": 0.1}}
    values = {
        "run_id": "run-1",
        "generated_at": "2026-07-16T10:00:00+08:00",
        "analysis_trade_date": "20260715",
        "market_data_ready": True,
        "market_data_blockers": [],
        "branch_readiness": _branches(),
        "branch_objects": {name: True for name in ("quant", "fundamental", "macro")},
        "factor_governance": {
            "governance_status": "ready",
            "production_eligible": True,
            "registry_file_sha256": "a" * 64,
            "production_factor_set_sha256": "b" * 64,
            "blockers": [],
        },
        "candidate_decision": {"candidate_decision_status": "complete", "blocker": ""},
        "portfolio_constructor": portfolio,
        "human_authorization": {
            "authorized": True,
            "run_id": "run-1",
            "analysis_trade_date": "20260715",
            "portfolio_sha256": canonical_sha256(portfolio),
        },
        "risk_reduction_quote_gate": {"authorized": False, "blockers": ["fresh_quote_missing"]},
        "material_warnings": [],
    }
    values.update(overrides)
    return build_v15_run_readiness(**values)


@pytest.mark.parametrize("session_lag", [0, 1, 2])
def test_v2_persists_compact_pinned_macro_readiness(
    session_lag: int,
) -> None:
    evidence = _macro_evidence(session_lag=session_lag)
    payload = _payload(
        branch_readiness=_branches(macro_evidence=evidence)
    )

    assert payload["schema_version"] == "v15_run_readiness.v2"
    assert payload["branch_data_ready"]["macro"] is True
    assert payload["new_risk_authorized"] is True
    assert payload["macro_readiness"] == {
        "macro_readiness_evidence_semantic_sha256": evidence[
            "semantic_sha256"
        ],
        "macro_release_calendar_generation_id": "calendar-generation",
        "macro_release_calendar_semantic_sha256": "3" * 64,
        "macro_release_calendar_market_open_days_sha256": "7" * 64,
        "macro_release_calendar_critical_policy_sha256": "8" * 64,
        "macro_logical_date": "2026-07-15",
        "target_session_date": "2026-07-15",
        "target_decision_cutoff_at": "2026-07-15T07:00:00+00:00",
        "session_lag": session_lag,
        "key_event_blocker_ids": [],
    }


def test_missing_macro_evidence_fails_only_macro_branch_closed() -> None:
    branches = _branches()
    branches["macro"]["metadata"].pop("macro_readiness_evidence")

    payload = _payload(branch_readiness=branches)

    assert payload["branch_data_ready"] == {
        "quant": True,
        "fundamental": True,
        "macro": False,
    }
    assert payload["new_risk_authorized"] is False
    assert (
        "branch_data_not_ready:macro:"
        "macro_release_readiness_evidence_missing"
        in payload["blockers"]
    )
    assert payload["macro_readiness"][
        "macro_readiness_evidence_semantic_sha256"
    ] == ""
    assert payload["macro_readiness"]["session_lag"] is None


@pytest.mark.parametrize(
    ("case", "expected_blocker"),
    [
        (
            "tampered",
            "macro_release_readiness_evidence_tampered",
        ),
        (
            "schema",
            "macro_release_readiness_evidence_contract_invalid",
        ),
        (
            "identity_missing",
            "macro_release_calendar_binding_missing",
        ),
        (
            "target_mismatch",
            "macro_release_readiness_evidence_target_mismatch",
        ),
    ],
)
def test_macro_evidence_contract_failures_are_precise(
    case: str,
    expected_blocker: str,
) -> None:
    evidence = _macro_evidence()
    if case == "tampered":
        evidence["macro_logical_date"] = "2026-07-14"
    elif case == "schema":
        evidence["schema_version"] = "macro-readiness-evidence.v0"
        _resign_macro_evidence(evidence)
    elif case == "identity_missing":
        evidence.pop("macro_release_calendar_market_open_days_sha256")
        _resign_macro_evidence(evidence)
    elif case == "target_mismatch":
        evidence["target_session_date"] = "2026-07-16"
        _resign_macro_evidence(evidence)

    payload = _payload(
        branch_readiness=_branches(macro_evidence=evidence)
    )

    assert payload["branch_data_ready"]["macro"] is False
    assert payload["branch_data_ready"]["quant"] is True
    assert payload["branch_data_ready"]["fundamental"] is True
    assert (
        f"branch_data_not_ready:macro:{expected_blocker}"
        in payload["blockers"]
    )


def test_macro_evidence_binding_mismatch_fails_closed() -> None:
    payload = _payload(
        branch_readiness=_branches(macro_binding="f" * 64)
    )

    assert payload["branch_data_ready"]["macro"] is False
    assert (
        "branch_data_not_ready:macro:"
        "macro_release_readiness_evidence_binding_mismatch"
        in payload["blockers"]
    )


def test_macro_semantic_sha_and_binding_must_be_lowercase_hex() -> None:
    evidence = _macro_evidence()
    evidence["semantic_sha256"] = "G" * 64
    payload = _payload(
        branch_readiness=_branches(
            macro_evidence=evidence,
            macro_binding="G" * 64,
        )
    )

    assert payload["branch_data_ready"]["macro"] is False
    assert (
        "branch_data_not_ready:macro:"
        "macro_release_readiness_evidence_tampered"
        in payload["blockers"]
    )
    assert (
        "branch_data_not_ready:macro:"
        "macro_release_readiness_evidence_binding_invalid"
        in payload["blockers"]
    )


@pytest.mark.parametrize("invalid_lag", [-1, 3, True, None, "1"])
def test_macro_session_lag_must_be_an_integer_from_zero_to_two(
    invalid_lag: object,
) -> None:
    evidence = _macro_evidence()
    evaluation = evidence["evaluation"]
    assert isinstance(evaluation, dict)
    lag = evaluation["session_lag"]
    assert isinstance(lag, dict)
    lag["session_lag"] = invalid_lag
    _resign_macro_evidence(evidence)

    payload = _payload(
        branch_readiness=_branches(macro_evidence=evidence)
    )

    assert payload["branch_data_ready"]["macro"] is False
    assert payload["macro_readiness"]["session_lag"] is None
    assert (
        "branch_data_not_ready:macro:macro_release_session_lag_invalid"
        in payload["blockers"]
    )


def test_not_ready_evaluation_persists_key_event_blocker_ids() -> None:
    evidence = _macro_evidence(session_lag=1)
    evaluation = evidence["evaluation"]
    assert isinstance(evaluation, dict)
    evaluation["ready"] = False
    evaluation["blockers"] = [
        "macro_release_critical_event_in_gap:critical-gdp"
    ]
    gap = evaluation["critical_event_gap"]
    assert isinstance(gap, dict)
    gap["ready"] = False
    gap["blocking_event_ids"] = ["critical-gdp"]
    gap["blockers"] = [
        "macro_release_critical_event_in_gap:critical-gdp"
    ]
    _resign_macro_evidence(evidence)

    payload = _payload(
        branch_readiness=_branches(macro_evidence=evidence)
    )

    assert payload["branch_data_ready"]["macro"] is False
    assert payload["macro_readiness"]["key_event_blocker_ids"] == [
        "critical-gdp"
    ]
    assert (
        "branch_data_not_ready:macro:macro_release_readiness_blocked"
        in payload["blockers"]
    )


def test_lag_zero_resolved_post_close_event_blocks_new_risk() -> None:
    event_id = "critical-gdp-resolved"
    event_blocker = f"macro_release_critical_event_in_gap:{event_id}"
    evidence = _macro_evidence(session_lag=0)
    evidence["target_decision_cutoff_at"] = (
        "2026-07-15T08:00:00+00:00"
    )
    evaluation = evidence["evaluation"]
    assert isinstance(evaluation, dict)
    evaluation["ready"] = False
    evaluation["blockers"] = [event_blocker]
    gap = evaluation["critical_event_gap"]
    assert isinstance(gap, dict)
    gap["ready"] = False
    gap["window_end_inclusive"] = "2026-07-15T08:00:00+00:00"
    gap["relevant_event_ids"] = [event_id]
    gap["resolved_event_ids"] = []
    gap["blocking_event_ids"] = [event_id]
    gap["blockers"] = [event_blocker]
    _resign_macro_evidence(evidence)

    payload = _payload(
        branch_readiness=_branches(macro_evidence=evidence)
    )

    assert payload["branch_data_ready"]["macro"] is False
    assert payload["new_risk_authorized"] is False
    assert payload["macro_readiness"]["session_lag"] == 0
    assert payload["macro_readiness"]["key_event_blocker_ids"] == [
        event_id
    ]
    assert (
        "branch_data_not_ready:macro:macro_release_readiness_blocked"
        in payload["blockers"]
    )


def test_missing_human_authorization_fails_closed() -> None:
    payload = _payload(human_authorization=None)
    assert payload["new_risk_authorized"] is False
    assert "new_risk_human_authorization_missing_or_invalid" in payload["blockers"]


def test_materialized_and_ready_are_independent() -> None:
    payload = _payload(branch_readiness=_branches(ready=False))
    assert all(payload["branch_objects_materialized"].values())
    assert not any(payload["branch_data_ready"].values())
    assert payload["new_risk_authorized"] is False


def test_empty_candidate_has_exact_portfolio_blocker() -> None:
    payload = _payload(
        candidate_decision={
            "candidate_decision_status": "empty",
            "blocker": "no_candidate_selected_by_portfolio_constructor",
        },
        portfolio_constructor={"valid": True, "target_weights": {}},
        human_authorization=None,
    )
    assert payload["candidate_decision_status"] == "empty"
    assert "no_candidate_selected_by_portfolio_constructor" in payload["blockers"]
    assert payload["macro_readiness"]["target_session_date"] == (
        "2026-07-15"
    )


def test_v1_readiness_artifacts_are_rejected(tmp_path: Path) -> None:
    legacy = {**_payload(), "schema_version": "v15_run_readiness.v1"}

    with pytest.raises(ValueError, match="invalid v15 run readiness schema"):
        write_v15_run_readiness(tmp_path / "rejected.json", legacy)

    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(ValueError, match="invalid v15 run readiness artifact"):
        load_v15_run_readiness(
            path,
            expected_sha256=canonical_sha256(legacy),
        )


def test_atomic_owner_only_roundtrip(tmp_path: Path) -> None:
    payload = _payload()
    path = tmp_path / "v15_run_readiness.json"
    reference = write_v15_run_readiness(path, payload)
    assert reference["schema_version"] == "v15_run_readiness.v2"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert load_v15_run_readiness(path, expected_sha256=reference["sha256"]) == payload
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_v15_run_readiness(path, expected_sha256="0" * 64)
