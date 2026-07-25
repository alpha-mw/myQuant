from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import pytest

from quant_investor.v17_v2_contract.action_matrix import (
    ACTIONS,
    ACTION_MATRIX_RESOURCE_SHA256,
    BUSINESS_TERMINAL_STATES,
    CHECKPOINTS,
    HARD_STOP_TERMINAL_STATES,
    NONTERMINAL_STATES,
    STATES,
    STATE_MACHINE_RESOURCE_SHA256,
    TERMINAL_STATES,
    VERSIONS,
    ActionDecision,
    ActionMatrixError,
    action_matrix_resource,
    decide_action,
    load_state_machine_resource,
    matching_rule_ids,
    matrix_cardinality,
)
from quant_investor.v17_v2_contract.namespace import NAMESPACE_MAP

RESOURCE_ROOT = (
    Path(__file__).resolve().parents[2] / "quant_investor" / "v17_v2_contract" / "resources"
)


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _single_outcome(
    *,
    version: str,
    action: str,
    state: str,
    checkpoint: str,
):
    decision = decide_action(
        version=version,
        action=action,
        state=state,
        checkpoint=checkpoint,
    )
    assert len(decision.outcomes) == 1
    return decision, decision.outcomes[0]


def test_action_matrix_domain_cardinality_and_unique_decision() -> None:
    cells = tuple(itertools.product(VERSIONS, ACTIONS, STATES, CHECKPOINTS))

    assert len(VERSIONS) == 5
    assert len(ACTIONS) == 8
    assert len(STATES) == 13
    assert len(CHECKPOINTS) == 3
    assert matrix_cardinality() == 1_560
    assert len(cells) == 1_560
    assert len(set(cells)) == 1_560

    for version, action, state, checkpoint in cells:
        assert (
            len(
                matching_rule_ids(
                    version=version,
                    action=action,
                    state=state,
                    checkpoint=checkpoint,
                )
            )
            == 1
        )
        decision = decide_action(
            version=version,
            action=action,
            state=state,
            checkpoint=checkpoint,
        )
        assert isinstance(decision, ActionDecision)
        assert decision.outcomes
        for outcome in decision.outcomes:
            assert type(outcome.command_commit) is bool
            assert type(outcome.business_acceptance) is bool
            assert type(outcome.terminal) is bool
            assert outcome.exit_code in {0, 2}
            assert outcome.latest_effect in {"UNCHANGED", "PUBLISHED", "REPAIRED"}


def test_action_matrix_and_state_machine_resources_are_canonical_and_hash_bound() -> None:
    matrix_path = RESOURCE_ROOT / "action_matrix.v1.json"
    machine_path = RESOURCE_ROOT / "state_machine.v1.json"

    matrix_raw = matrix_path.read_bytes()
    machine_raw = machine_path.read_bytes()
    assert matrix_raw == _canonical_bytes(json.loads(matrix_raw))
    assert machine_raw == _canonical_bytes(json.loads(machine_raw))
    assert hashlib.sha256(matrix_raw).hexdigest() == ACTION_MATRIX_RESOURCE_SHA256
    assert hashlib.sha256(machine_raw).hexdigest() == STATE_MACHINE_RESOURCE_SHA256
    matrix = action_matrix_resource()
    machine = load_state_machine_resource()
    assert matrix["schema"] == matrix["version"] == ("myquant.v17.v2.action-matrix.v1")
    assert machine["schema"] == machine["version"] == ("myquant.v17.v2.state-machine.v1")
    assert matrix["protocol_version"] == machine["protocol_version"] == ("myquant.v17.v2")
    assert matrix["package_version"] == machine["package_version"] == "17.0.0"


def test_action_matrix_write_namespaces_match_namespace_contract_exactly() -> None:
    expected = {
        namespace_id: specification.path_template
        for namespace_id, specification in NAMESPACE_MAP.items()
    }
    assert action_matrix_resource()["namespace_paths"] == expected


def test_v2_contract_import_does_not_import_v1_package() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = """
import json
import sys
import quant_investor.v17_v2_contract
import quant_investor.v17_v2_contract.action_matrix
print(json.dumps(sorted(
    name for name in sys.modules
    if name == "quant_investor.v17" or name.startswith("quant_investor.v17.")
)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []


@pytest.mark.parametrize("checkpoint", CHECKPOINTS)
@pytest.mark.parametrize("state", TERMINAL_STATES)
@pytest.mark.parametrize("action", ACTIONS)
def test_v1_terminal_only_status_and_artifact_are_allowed(
    checkpoint: str,
    state: str,
    action: str,
) -> None:
    decision, outcome = _single_outcome(
        version="v1",
        action=action,
        state=state,
        checkpoint=checkpoint,
    )
    if action in {"READ_STATUS", "READ_ARTIFACT"}:
        assert decision.allowed is True
        assert decision.read_only is True
        assert outcome.target_state == state
        assert outcome.terminal is True
        assert outcome.business_acceptance is True
        assert outcome.exit_code == 0
    else:
        assert decision.allowed is False
        assert decision.read_only is False
        assert decision.allowed_write_namespaces == ()
        assert outcome.command_commit is False
        assert outcome.business_acceptance is False
        assert outcome.exit_code == 2
        assert outcome.latest_effect == "UNCHANGED"


@pytest.mark.parametrize("checkpoint", CHECKPOINTS)
@pytest.mark.parametrize("state", NONTERMINAL_STATES)
@pytest.mark.parametrize("action", ACTIONS)
def test_all_v1_nonterminal_actions_are_zero_write(
    checkpoint: str,
    state: str,
    action: str,
) -> None:
    decision, outcome = _single_outcome(
        version="v1",
        action=action,
        state=state,
        checkpoint=checkpoint,
    )
    assert decision.allowed is False
    assert decision.allowed_write_namespaces == ()
    assert outcome.command_commit is False
    assert outcome.business_acceptance is False
    assert outcome.target_state is None
    assert outcome.exit_code == 2
    assert outcome.latest_effect == "UNCHANGED"


@pytest.mark.parametrize("version", ("unknown", "malformed"))
@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("state", STATES)
@pytest.mark.parametrize("checkpoint", CHECKPOINTS)
def test_unknown_and_malformed_protocols_fail_closed(
    version: str,
    action: str,
    state: str,
    checkpoint: str,
) -> None:
    decision, outcome = _single_outcome(
        version=version,
        action=action,
        state=state,
        checkpoint=checkpoint,
    )
    assert decision.allowed is False
    assert decision.allowed_write_namespaces == ()
    assert outcome.command_commit is False
    assert outcome.business_acceptance is False
    assert outcome.exit_code == 2
    assert outcome.latest_effect == "UNCHANGED"


@pytest.mark.parametrize("state", STATES[1:])
@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("checkpoint", CHECKPOINTS)
def test_absent_protocol_with_nonmissing_state_fails_closed(
    state: str,
    action: str,
    checkpoint: str,
) -> None:
    decision, outcome = _single_outcome(
        version="ABSENT",
        action=action,
        state=state,
        checkpoint=checkpoint,
    )
    assert decision.allowed is False
    assert decision.reason == "absent_protocol_state_conflict"
    assert decision.allowed_write_namespaces == ()
    assert outcome.exit_code == 2
    assert outcome.command_commit is False


@pytest.mark.parametrize("checkpoint", ("PRE_IMPORT", "ACCEPTED"))
def test_no_action_can_write_or_commit_before_initialized(checkpoint: str) -> None:
    for version, action, state in itertools.product(VERSIONS, ACTIONS, STATES):
        decision = decide_action(
            version=version,
            action=action,
            state=state,
            checkpoint=checkpoint,
        )
        assert decision.allowed_write_namespaces == ()
        assert all(outcome.command_commit is False for outcome in decision.outcomes)
        assert all(outcome.latest_effect == "UNCHANGED" for outcome in decision.outcomes)


@pytest.mark.parametrize(
    ("version", "action", "state", "writes", "retry_cas", "target"),
    [
        (
            "ABSENT",
            "SOURCE_MAINTAIN",
            "MISSING",
            ("SOURCE_OBJECTS", "SOURCE_MANIFESTS", "SOURCE_LOCATORS"),
            "EMPTY",
            None,
        ),
        (
            "ABSENT",
            "RISK_POLICY_SEAL",
            "MISSING",
            ("SOURCE_OBJECTS",),
            "ABSENT_TARGET",
            None,
        ),
        (
            "ABSENT",
            "SHADOW_PREPARE",
            "MISSING",
            (
                "RUN_ROOT",
                "RUN_LEDGER",
                "RUN_LOCK",
                "RUN_EVENTS",
                "RUN_RECEIPTS",
                "MODELS",
                "OUTCOMES",
                "LATEST",
                "LATEST_LOCK",
            ),
            "EMPTY",
            "DEEP_REQUEST_READY",
        ),
        (
            "v2",
            "SHADOW_PREPARE",
            "PREPARED",
            (
                "RUN_LEDGER",
                "RUN_LOCK",
                "RUN_EVENTS",
                "RUN_RECEIPTS",
                "MODELS",
                "OUTCOMES",
                "LATEST",
                "LATEST_LOCK",
            ),
            "CURRENT_LEDGER_SHA",
            "DEEP_REQUEST_READY",
        ),
        (
            "v2",
            "REPAIR_LATEST",
            "SHADOW_PORTFOLIO_INFEASIBLE",
            ("LATEST", "LATEST_LOCK"),
            "CURRENT_LEDGER_AND_LATEST_SHA",
            "SHADOW_PORTFOLIO_INFEASIBLE",
        ),
    ],
)
def test_initialized_write_namespaces_are_exact(
    version: str,
    action: str,
    state: str,
    writes: tuple[str, ...],
    retry_cas: str,
    target: str | None,
) -> None:
    decision = decide_action(
        version=version,
        action=action,
        state=state,
        checkpoint="INITIALIZED",
    )
    matching_successes = [
        outcome
        for outcome in decision.outcomes
        if outcome.business_acceptance and outcome.target_state == target
    ]
    assert len(matching_successes) == 1
    outcome = matching_successes[0]
    assert decision.allowed is True
    assert decision.read_only is False
    assert decision.allowed_write_namespaces == writes
    assert decision.retry_cas == retry_cas
    assert outcome.target_state == target
    assert outcome.command_commit is True
    assert outcome.business_acceptance is True
    assert outcome.exit_code == 0


@pytest.mark.parametrize(
    ("version", "state"),
    [
        ("ABSENT", "MISSING"),
        ("v2", "PREPARED"),
        ("v2", "DETERMINISTIC_COMPLETE"),
    ],
)
def test_initialized_prepare_outcomes_include_only_success_or_durable_hard_stop(
    version: str,
    state: str,
) -> None:
    decision = decide_action(
        version=version,
        action="SHADOW_PREPARE",
        state=state,
        checkpoint="INITIALIZED",
    )
    assert tuple(outcome.target_state for outcome in decision.outcomes) == (
        "DEEP_REQUEST_READY",
        "HARD_STOP_SNAPSHOT_DRIFT",
        "HARD_STOP_INVALID_EVIDENCE",
    )
    assert {"OUTCOMES", "LATEST", "LATEST_LOCK"}.issubset(decision.allowed_write_namespaces)
    success, *hard_stops = decision.outcomes
    assert success.business_acceptance is True
    assert success.command_commit is True
    assert success.exit_code == 0
    assert success.latest_effect == "UNCHANGED"
    for outcome in hard_stops:
        assert outcome.command_commit is True
        assert outcome.business_acceptance is False
        assert outcome.terminal is True
        assert outcome.exit_code == 2
        assert outcome.latest_effect == "PUBLISHED"


def test_initialized_receive_outcomes_are_complete_and_ordered() -> None:
    decision = decide_action(
        version="v2",
        action="SHADOW_RECEIVE",
        state="DEEP_REQUEST_READY",
        checkpoint="INITIALIZED",
    )
    assert decision.allowed_write_namespaces == (
        "RUN_LEDGER",
        "RUN_LOCK",
        "RUN_EVENTS",
        "RUN_RECEIPTS",
        "OUTCOMES",
        "LATEST",
        "LATEST_LOCK",
    )
    assert tuple(outcome.target_state for outcome in decision.outcomes) == (
        "DEEP_RESPONSE_RECEIVED",
        "HARD_STOP_SNAPSHOT_DRIFT",
        "HARD_STOP_INVALID_EVIDENCE",
    )
    success, *hard_stops = decision.outcomes
    assert success.command_commit is True
    assert success.business_acceptance is True
    assert success.exit_code == 0
    assert success.latest_effect == "UNCHANGED"
    assert tuple(outcome.target_state for outcome in hard_stops) == (
        "HARD_STOP_SNAPSHOT_DRIFT",
        "HARD_STOP_INVALID_EVIDENCE",
    )
    for outcome in hard_stops:
        assert outcome.command_commit is True
        assert outcome.business_acceptance is False
        assert outcome.terminal is True
        assert outcome.exit_code == 2
        assert outcome.latest_effect == "PUBLISHED"


def test_failure_semantics_are_explicit_and_outside_cartesian_domain() -> None:
    payload = action_matrix_resource()
    failures = payload["failure_semantics"]
    assert set(payload["domains"]) == {
        "actions",
        "checkpoints",
        "states",
        "versions",
    }
    assert tuple(failures) == (
        "CAS_CONFLICT",
        "POST_INITIALIZED_UNCOMMITTED",
        "PRE_IMPORT_REJECTION",
        "PRE_INITIALIZED_VALIDATION_FAILURE",
        "TERMINAL_LATEST_PUBLICATION_FAILURE",
    )
    for failure_id in (
        "PRE_IMPORT_REJECTION",
        "PRE_INITIALIZED_VALIDATION_FAILURE",
        "CAS_CONFLICT",
    ):
        semantics = failures[failure_id]
        assert semantics["command_commit"] is False
        assert semantics["business_acceptance"] is False
        assert semantics["allowed_write_namespaces"] == []
        assert semantics["exit_code"] == 2
        assert semantics["latest_effect"] == "UNCHANGED"
        assert semantics["receipt_effect"] == "NONE"

    uncommitted = failures["POST_INITIALIZED_UNCOMMITTED"]
    assert uncommitted["command_commit"] is False
    assert uncommitted["business_acceptance"] is False
    assert uncommitted["allowed_write_namespaces"] == ["RUN_RECEIPTS"]
    assert uncommitted["exit_code"] == 2
    assert uncommitted["latest_effect"] == "UNCHANGED"
    assert uncommitted["receipt_effect"] == "UNPUBLISHED_NOT_COMMITTED"
    assert uncommitted["required_next_action"] == "READ_STATUS"

    latest_failure = failures["TERMINAL_LATEST_PUBLICATION_FAILURE"]
    assert latest_failure["command_commit"] is True
    assert latest_failure["business_acceptance"] is False
    assert latest_failure["allowed_write_namespaces"] == ["RUN_RECEIPTS"]
    assert latest_failure["exit_code"] == 2
    assert latest_failure["latest_effect"] == "UNCHANGED"
    assert latest_failure["receipt_effect"] == "TERMINAL_UNPUBLISHED"
    assert latest_failure["required_next_action"] == "REPAIR_LATEST"


@pytest.mark.parametrize(
    ("state", "expected_targets"),
    [
        (
            "DEEP_RESPONSE_RECEIVED",
            BUSINESS_TERMINAL_STATES + HARD_STOP_TERMINAL_STATES,
        ),
        (
            "PORTFOLIO_COMPLETE",
            (
                "SHADOW_COMPLETE_AWAITING_HUMAN_DECISION",
                "HARD_STOP_SNAPSHOT_DRIFT",
                "HARD_STOP_INVALID_EVIDENCE",
            ),
        ),
    ],
)
def test_initialized_finalize_outcomes_freeze_business_and_hard_stop_exits(
    state: str,
    expected_targets: tuple[str, ...],
) -> None:
    decision = decide_action(
        version="v2",
        action="SHADOW_FINALIZE",
        state=state,
        checkpoint="INITIALIZED",
    )
    assert tuple(outcome.target_state for outcome in decision.outcomes) == (expected_targets)
    for outcome in decision.outcomes:
        assert outcome.command_commit is True
        assert outcome.latest_effect == "PUBLISHED"
        if outcome.target_state in BUSINESS_TERMINAL_STATES:
            assert outcome.business_acceptance is True
            assert outcome.exit_code == 0
        else:
            assert outcome.target_state in HARD_STOP_TERMINAL_STATES
            assert outcome.business_acceptance is False
            assert outcome.exit_code == 2


@pytest.mark.parametrize(
    "action",
    ("SHADOW_PREPARE", "SHADOW_RECEIVE", "SHADOW_FINALIZE"),
)
@pytest.mark.parametrize("state", HARD_STOP_TERMINAL_STATES)
@pytest.mark.parametrize("checkpoint", CHECKPOINTS)
def test_hard_stop_exact_replay_is_read_only_but_exits_two(
    action: str,
    state: str,
    checkpoint: str,
) -> None:
    decision, outcome = _single_outcome(
        version="v2",
        action=action,
        state=state,
        checkpoint=checkpoint,
    )
    assert decision.allowed is True
    assert decision.read_only is True
    assert decision.allowed_write_namespaces == ()
    assert decision.retry_cas == "EXACT_REPLAY"
    assert outcome.target_state == state
    assert outcome.command_commit is False
    assert outcome.business_acceptance is False
    assert outcome.terminal is True
    assert outcome.exit_code == 2
    assert outcome.latest_effect == "UNCHANGED"


@pytest.mark.parametrize("state", BUSINESS_TERMINAL_STATES)
@pytest.mark.parametrize("checkpoint", CHECKPOINTS)
def test_business_terminal_finalize_replay_exits_zero(
    state: str,
    checkpoint: str,
) -> None:
    decision, outcome = _single_outcome(
        version="v2",
        action="SHADOW_FINALIZE",
        state=state,
        checkpoint=checkpoint,
    )
    assert decision.allowed is True
    assert decision.read_only is True
    assert outcome.target_state == state
    assert outcome.command_commit is False
    assert outcome.business_acceptance is True
    assert outcome.terminal is True
    assert outcome.exit_code == 0
    assert outcome.latest_effect == "UNCHANGED"


@pytest.mark.parametrize(
    ("action", "state"),
    [
        ("SHADOW_PREPARE", "DEEP_REQUEST_READY"),
        ("SHADOW_RECEIVE", "DEEP_RESPONSE_RECEIVED"),
    ],
)
@pytest.mark.parametrize("checkpoint", CHECKPOINTS)
def test_nonterminal_exact_retry_is_read_only(
    action: str,
    state: str,
    checkpoint: str,
) -> None:
    decision, outcome = _single_outcome(
        version="v2",
        action=action,
        state=state,
        checkpoint=checkpoint,
    )
    assert decision.allowed is True
    assert decision.read_only is True
    assert decision.retry_cas == "EXACT_REPLAY"
    assert decision.allowed_write_namespaces == ()
    assert outcome.target_state == state
    assert outcome.command_commit is False
    assert outcome.business_acceptance is True
    assert outcome.exit_code == 0


@pytest.mark.parametrize(
    ("action", "state"),
    [
        ("SHADOW_RECEIVE", "PREPARED"),
        ("SHADOW_RECEIVE", "PORTFOLIO_COMPLETE"),
        ("SHADOW_FINALIZE", "PREPARED"),
        ("SHADOW_FINALIZE", "DEEP_REQUEST_READY"),
    ],
)
@pytest.mark.parametrize("checkpoint", CHECKPOINTS)
def test_wrong_receive_or_finalize_state_is_preimport_rejected(
    action: str,
    state: str,
    checkpoint: str,
) -> None:
    decision, outcome = _single_outcome(
        version="v2",
        action=action,
        state=state,
        checkpoint=checkpoint,
    )
    assert decision.allowed is False
    assert decision.allowed_write_namespaces == ()
    assert outcome.target_state is None
    assert outcome.command_commit is False
    assert outcome.business_acceptance is False
    assert outcome.exit_code == 2
    assert outcome.latest_effect == "UNCHANGED"


def test_state_machine_terminal_semantics_are_immutable_and_explicit() -> None:
    machine = load_state_machine_resource()
    assert tuple(machine["nonterminal_states"]) == NONTERMINAL_STATES
    assert tuple(machine["terminal_states"]) == TERMINAL_STATES
    assert machine["protocol_roots"] == {
        "private_source": "data/private/v17_sources/protocol-v2",
        "shadow_results": "results/v17_shadow/protocol-v2",
    }
    for state in TERMINAL_STATES:
        assert machine["transitions"][state] == []
        semantics = machine["terminal_semantics"][state]
        assert semantics["target_state"] == state
        assert semantics["terminal"] is True
        assert semantics["command_commit"] is True
        assert semantics["latest_effect"] == "PUBLISHED"
        if state in BUSINESS_TERMINAL_STATES:
            assert semantics["business_acceptance"] is True
            assert semantics["exit_code"] == 0
        else:
            assert semantics["business_acceptance"] is False
            assert semantics["exit_code"] == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", "V3"),
        ("action", "UNKNOWN_ACTION"),
        ("state", "NOT_A_STATE"),
        ("checkpoint", "AFTER_WRITE"),
    ],
)
def test_out_of_domain_callers_fail_closed(field: str, value: str) -> None:
    kwargs = {
        "version": "v2",
        "action": "READ_STATUS",
        "state": "PREPARED",
        "checkpoint": "PRE_IMPORT",
    }
    kwargs[field] = value
    with pytest.raises(ActionMatrixError):
        decide_action(**kwargs)
