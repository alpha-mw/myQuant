from copy import deepcopy

import pytest

from quant_investor.operations.daily_contract import ContractError
from quant_investor.market.future_calendar_context import (
    validate_loop_context,
    validate_future_state,
    future_mode,
)

REF = {"path": "evidence.json", "sha256": "a" * 64}


def context():
    return {
        "schema_version": "cn-daily-factor-loop.v2",
        "release_install_input_ref": REF,
        "release_repository_root": "/fixture/repository",
        "release_commit": "b" * 40,
        "calendar_capture_parent": "/fixture/captures",
        "initial_calendar_receipt_ref": None,
        "next_session_calendar_mode": "SYNTHETIC_FIXTURE_ONLY",
    }


def state():
    return {
        "schema_version": "cn-daily-factor-state.v2",
        "phase": "CORE_READY",
        "trade_date": "20260902",
        "context_sha256": "c" * 64,
        "calendar_receipt_ref": REF,
        "core_checkpoint_ref": REF,
        "core_observation_refs": {"LOW": REF, "W80": REF},
        "future_capture_refs": None,
        "fixture_transport_evidence_ref": None,
        "next_session_calendar_proof_ref": None,
        "next_session_calendar_failure_ref": None,
        "core_handoff_ref": None,
    }


def validate(value, mode="SYNTHETIC_FIXTURE_ONLY"):
    return validate_future_state(value, context_sha256="c" * 64, trade_date="20260902", mode=mode)


def test_v1_remains_permissive_and_disabled():
    value = {"schema_version": "cn-daily-factor-loop.v1", "historical_extra": True}
    assert validate_loop_context(value) == value
    assert future_mode(value) == "DISABLED"


def test_v2_mode_does_not_grant_provenance():
    assert future_mode(context()) == "SYNTHETIC_FIXTURE_ONLY"
    assert validate(state()) == state()


@pytest.mark.parametrize("fault", ["extra", "missing", "mode", "commit", "path"])
def test_context_rejects_invalid_fields(fault):
    value = deepcopy(context())
    if fault == "extra":
        value["caller_clock"] = "20260902"
    elif fault == "missing":
        value.pop("initial_calendar_receipt_ref")
    elif fault == "mode":
        value["next_session_calendar_mode"] = "LIVE"
    elif fault == "commit":
        value["release_commit"] = "bad"
    else:
        value["calendar_capture_parent"] = "/fixture/../captures"
    with pytest.raises(ContractError):
        validate_loop_context(value)


@pytest.mark.parametrize(
    "fault",
    [
        "extra",
        "date",
        "sha",
        "phase",
        "early_proof",
        "conflict",
        "missing_evidence",
        "wrong_capture_root",
    ],
)
def test_state_rejects_invalid_recovery_binding(fault):
    value = deepcopy(state())
    if fault == "extra":
        value["latest"] = True
    elif fault == "date":
        value["trade_date"] = "20260903"
    elif fault == "sha":
        value["context_sha256"] = "d" * 64
    elif fault == "phase":
        value["phase"] = "UNKNOWN"
    elif fault == "early_proof":
        value["next_session_calendar_proof_ref"] = REF
    elif fault == "conflict":
        value.update(
            phase="FUTURE_BOUND",
            next_session_calendar_proof_ref=REF,
            next_session_calendar_failure_ref=REF,
        )
    else:
        value.update(
            phase="CAPTURE_BOUND",
            future_capture_refs={
                "execution_ref": {
                    "relative_path": "capture/capture-execution.json",
                    "byte_sha256": "a" * 64,
                },
                "success_ref": {
                    "relative_path": "capture/capture-success.json",
                    "byte_sha256": "b" * 64,
                },
            },
        )
        if fault == "wrong_capture_root":
            value["fixture_transport_evidence_ref"] = REF
            value["future_capture_refs"]["success_ref"][
                "relative_path"
            ] = "other/capture-success.json"
    with pytest.raises(ContractError):
        validate(value)


def test_disabled_state_and_failure_state_are_distinct():
    value = state()
    value["phase"] = "FUTURE_BOUND"
    assert validate(value, "DISABLED") == value
    with pytest.raises(ContractError):
        validate(value)
    value["next_session_calendar_failure_ref"] = REF
    assert validate(value) == value
    with pytest.raises(ContractError):
        validate(value, "DISABLED")
    value.update(phase="CORE_PUBLISHED", core_handoff_ref=REF)
    assert validate(value) == value
