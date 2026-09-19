"""Lifecycle and provenance invariants, not a synthetic full-DAG proof."""

import pytest

from quant_investor.operations.daily_contract import (
    ContractError,
    EOD_NODE_IDS,
    GRAPH,
    GRAPH_SHA256,
    NODE_IDS,
    NodeState,
    availability,
    completion_candidate,
    dependencies,
    failure,
    transition,
    validate_ref,
)


def test_eod_completion_excludes_next_session_morning():
    assert len(NODE_IDS) == 17 and len(EOD_NODE_IDS) == 16
    assert "morning" not in EOD_NODE_IDS
    assert len(GRAPH_SHA256) == 64
    states = dict.fromkeys(EOD_NODE_IDS, NodeState.SUCCEEDED)
    assert completion_candidate(states)
    with pytest.raises(ContractError, match="EOD_NODE_SET_INVALID"):
        completion_candidate({**states, "morning": NodeState.SUCCEEDED})
    for node in EOD_NODE_IDS:
        for state in NodeState:
            result = completion_candidate({**states, node: state})
            assert result is (state == NodeState.SUCCEEDED)


def test_ready_pool_does_not_wait_for_auxiliaries_and_store_does_not_wait_for_alpha():
    states = dict.fromkeys(NODE_IDS, NodeState.BLOCKED)
    for node in ("factor", "low_observation", "w80_observation"):
        states[node] = NodeState.SUCCEEDED
    assert dependencies("top100", states) == ()
    states["w80_observation"] = NodeState.PARTIAL
    assert dependencies("top100", states) == ("w80_observation",)
    for node in ("calendar", "market", "corporate_action_recon"):
        states[node] = NodeState.SUCCEEDED
    assert dependencies("store", states) == ()
    store = next(row for row in GRAPH if row.node_id == "store")
    assert store.after == ("decision",)


def test_graph_is_acyclic_and_observation_batch_retains_separate_child_states():
    visited = set()
    for node in GRAPH:
        assert set(node.requires + node.after) <= visited
        visited.add(node.node_id)
    batch = [n.node_id for n in GRAPH if n.writer_group == "observation_batch"]
    assert batch == ["low_observation", "w80_observation"]


def test_success_cannot_be_restarted_and_unknown_failure_cannot_be_retried():
    with pytest.raises(ContractError, match="NODE_TRANSITION_INVALID"):
        transition(NodeState.SUCCEEDED, NodeState.READY)
    assert transition(NodeState.SUCCEEDED, NodeState.STALE) == NodeState.STALE
    with pytest.raises(ContractError, match="TERMINAL_FAILURE_NOT_RETRYABLE"):
        transition(NodeState.FAILED, NodeState.READY, failure_code="WRITER_FAILED", reconciled=True)
    with pytest.raises(ContractError, match="RECOVERY_RECONCILIATION_REQUIRED"):
        transition(NodeState.FAILED, NodeState.READY, failure_code="PROVIDER_UNAVAILABLE")
    assert (
        transition(
            NodeState.FAILED, NodeState.READY, failure_code="PROVIDER_UNAVAILABLE", reconciled=True
        )
        == NodeState.READY
    )
    assert failure("POST_WRITE_IN_DOUBT", next_node="top100")["retryable"] is False


@pytest.mark.parametrize(
    "path", ["/etc/passwd", "../escape", "a/../b", "a//b", "a/./b", ".", "a\\b"]
)
def test_refs_reject_escape_and_noncanonical_paths(path):
    with pytest.raises(ContractError, match="REF_PATH_INVALID"):
        validate_ref({"path": path, "sha256": "a" * 64})


def timing(**overrides):
    return availability(
        **{
            "required_nodes": frozenset({"factor", "low_observation", "w80_observation"}),
            "proven_available_at": {
                "factor": "2026-09-04T07:00:00Z",
                "low_observation": "2026-09-04T07:00:00Z",
                "w80_observation": "2026-09-04T07:00:00Z",
            },
            "deadline": "2026-09-04T07:00:00Z",
            "deadline_ref": {"path": "policies/exact-deadline.json", "sha256": "a" * 64},
            "synthetic": False,
            **overrides,
        }
    )


def test_deadline_equality_allowed_but_late_observation_never_realtime_oos():
    assert timing()["prospective"] is True
    result = timing(
        proven_available_at={
            "factor": "2026-09-04T06:59:59Z",
            "low_observation": "2026-09-06T03:32:05Z",
            "w80_observation": "2026-09-06T03:32:05Z",
        }
    )
    assert result["classification"] == "LATE_REGISTERED"
    assert result["prospective"] is False


@pytest.mark.parametrize(
    "override",
    [
        {"deadline": None},
        {"deadline_ref": None},
        {"synthetic": True},
        {"recovered_unknown": True},
        {"recomputed": True},
        {"proven_available_at": {"factor": None, "low_observation": None, "w80_observation": None}},
    ],
)
def test_missing_or_recovered_or_synthetic_evidence_never_eligible(override):
    assert timing(**override)["prospective"] is False


def test_no_naive_clock_and_no_silent_dropping_required_timing_nodes():
    with pytest.raises(ContractError, match="TIMESTAMP_INVALID"):
        timing(deadline="2026-09-04T07:00:00")
    with pytest.raises(ContractError, match="TIMING_NODE_SET_INVALID"):
        timing(proven_available_at={"factor": "2026-09-04T07:00:00Z"})


@pytest.mark.parametrize("value", [None, True, 2, "reference", ["path", "sha256"]])
def test_non_mapping_ref_is_typed_contract_failure(value):
    with pytest.raises(ContractError, match="REF_SCHEMA_INVALID"):
        validate_ref(value)
