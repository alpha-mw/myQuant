"""Pure contracts for the CN daily evidence coordinator.

This module cannot invoke providers or subsystem writers. Domain readiness and
investment admission are deliberately not inferred from lifecycle transitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
from pathlib import PurePosixPath
import re
from typing import Mapping

from quant_investor.contracts import canonical_json_bytes


class ContractError(ValueError):
    """A deterministic orchestration contract violation."""


class NodeState(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    READY = "READY"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    PARTIAL = "PARTIAL"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    STALE = "STALE"


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    requires: tuple[str, ...] = ()
    after: tuple[str, ...] = ()
    eod_required: bool = True
    writer_group: str | None = None


# Code-owned topology. A request cannot inject nodes, edges, or callbacks.
GRAPH = (
    NodeSpec("calendar", writer_group="maintenance_core"),
    NodeSpec("pit", requires=("calendar",), writer_group="maintenance_core"),
    NodeSpec("market", requires=("calendar", "pit"), writer_group="maintenance_core"),
    NodeSpec("factor", requires=("calendar", "pit", "market")),
    NodeSpec("low_observation", requires=("factor",), writer_group="observation_batch"),
    NodeSpec("w80_observation", requires=("factor",), writer_group="observation_batch"),
    NodeSpec("top100", requires=("factor", "low_observation", "w80_observation")),
    NodeSpec("theme", requires=("top100",)),
    NodeSpec("industry", requires=("top100",)),
    NodeSpec("exposure", requires=("theme",)),
    NodeSpec("fundamental", requires=("pit",)),
    NodeSpec("macro", requires=("calendar", "pit", "market")),
    NodeSpec("corporate_action_recon", requires=("market",)),
    NodeSpec(
        "decision",
        requires=("top100", "theme", "industry", "exposure", "fundamental", "macro"),
    ),
    NodeSpec(
        "store", requires=("calendar", "market", "corporate_action_recon"), after=("decision",)
    ),
    NodeSpec("dashboard", requires=("store",), after=("decision",)),
    NodeSpec("morning", eod_required=False),
)
NODE_IDS = frozenset(node.node_id for node in GRAPH)
EOD_NODE_IDS = frozenset(node.node_id for node in GRAPH if node.eod_required)
GRAPH_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "schema_version": "cn-daily-evidence-graph.v1",
            "nodes": [
                {**asdict(node), "requires": list(node.requires), "after": list(node.after)}
                for node in GRAPH
            ],
        }
    )
).hexdigest()


@dataclass(frozen=True)
class FailureRule:
    retryable: bool
    owner_action_required: bool
    canonical_state_may_have_changed: bool = False


FAILURES = {
    "INPUT_MISSING": FailureRule(True, False),
    "INPUT_STALE": FailureRule(True, False),
    "POINTER_MISMATCH": FailureRule(False, False),
    "SHA_MISMATCH": FailureRule(False, True),
    "UPSTREAM_INCOMPLETE": FailureRule(True, False),
    "PROVIDER_UNAVAILABLE": FailureRule(True, False),
    "AUTHORIZATION_BLOCKED": FailureRule(False, True),
    "SCHEMA_MISMATCH": FailureRule(False, True),
    "IDEMPOTENCY_CONFLICT": FailureRule(False, True, True),
    "DATE_MISMATCH": FailureRule(False, False),
    "CALENDAR_MISMATCH": FailureRule(False, False),
    "CORPORATE_ACTION_UNRECONCILED": FailureRule(False, True),
    "POLICY_BLOCKED": FailureRule(False, True),
    "IO_TRANSIENT": FailureRule(True, False),
    "WRITER_FAILED": FailureRule(False, False, True),
    "POST_WRITE_IN_DOUBT": FailureRule(False, False, True),
    "VALIDATION_FAILED": FailureRule(False, False),
    "UNSUPPORTED_LINEAGE_GAP": FailureRule(False, True),
}


def failure(code: str, *, next_node: str) -> dict:
    if code not in FAILURES or next_node not in NODE_IDS:
        raise ContractError("FAILURE_CONTRACT_INVALID")
    return {"code": code, **asdict(FAILURES[code]), "recommended_next_node": next_node}


def transition(
    source: NodeState,
    target: NodeState,
    *,
    failure_code: str | None = None,
    reconciled: bool = False,
) -> NodeState:
    """Validate lifecycle changes; callers separately replay dependencies/refs."""
    if not isinstance(source, NodeState) or not isinstance(target, NodeState):
        raise ContractError("NODE_STATE_INVALID")
    allowed = {
        NodeState.NOT_STARTED: {NodeState.READY, NodeState.BLOCKED, NodeState.SKIPPED},
        NodeState.READY: {NodeState.RUNNING, NodeState.BLOCKED, NodeState.SKIPPED},
        NodeState.RUNNING: {
            NodeState.SUCCEEDED,
            NodeState.PARTIAL,
            NodeState.BLOCKED,
            NodeState.FAILED,
        },
        NodeState.SUCCEEDED: {NodeState.STALE},
        NodeState.PARTIAL: {NodeState.READY, NodeState.STALE},
        NodeState.BLOCKED: {NodeState.READY, NodeState.SKIPPED},
        NodeState.FAILED: {NodeState.READY},
        NodeState.SKIPPED: {NodeState.READY, NodeState.BLOCKED},
        NodeState.STALE: set(),
    }
    if target not in allowed[source]:
        raise ContractError("NODE_TRANSITION_INVALID")
    if source == NodeState.FAILED:
        if failure_code not in FAILURES or not FAILURES[failure_code].retryable:
            raise ContractError("TERMINAL_FAILURE_NOT_RETRYABLE")
        if not reconciled:
            raise ContractError("RECOVERY_RECONCILIATION_REQUIRED")
    return target


def dependencies(node_id: str, states: Mapping[str, NodeState]) -> tuple[str, ...]:
    if node_id not in NODE_IDS or set(states) - NODE_IDS:
        raise ContractError("GRAPH_NODE_INVALID")
    if any(not isinstance(value, NodeState) for value in states.values()):
        raise ContractError("NODE_STATE_INVALID")
    node = next(row for row in GRAPH if row.node_id == node_id)
    # `after` is an ordering barrier, not an admission or data dependency.
    return tuple(name for name in node.requires if states.get(name) != NodeState.SUCCEEDED)


def completion_candidate(states: Mapping[str, NodeState]) -> bool:
    """Necessary lifecycle condition only; never a substitute for native validation."""
    if set(states) != EOD_NODE_IDS:
        raise ContractError("EOD_NODE_SET_INVALID")
    if any(not isinstance(value, NodeState) for value in states.values()):
        raise ContractError("NODE_STATE_INVALID")
    return all(value == NodeState.SUCCEEDED for value in states.values())


def validate_ref(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ContractError("REF_SCHEMA_INVALID")
    path, sha = value["path"], value["sha256"]
    if type(path) is not str or type(sha) is not str:
        raise ContractError("REF_SCHEMA_INVALID")
    parts = PurePosixPath(path)
    if (
        not path
        or parts.is_absolute()
        or str(parts) != path
        or ".." in parts.parts
        or "\\" in path
        or "\x00" in path
        or path == "."
    ):
        raise ContractError("REF_PATH_INVALID")
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ContractError("REF_SHA_INVALID")
    return dict(value)


def utc_stamp(value: str) -> datetime:
    if type(value) is not str:
        raise ContractError("TIMESTAMP_INVALID")
    try:
        instant = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ContractError("TIMESTAMP_INVALID") from exc
    if instant.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ContractError("TIMESTAMP_NONCANONICAL")
    return instant


def availability(
    *,
    proven_available_at: Mapping[str, str | None],
    required_nodes: frozenset[str],
    deadline: str | None,
    deadline_ref: Mapping[str, str] | None,
    synthetic: bool,
    recovered_unknown: bool = False,
    recomputed: bool = False,
) -> dict:
    """Classify already-validated timing evidence, never infer old publication time.

    A ledger validator must prove deadline policy/Calendar semantics and source
    timestamps before using this pure comparison. Missing proof stays ineligible.
    """
    if not required_nodes or not required_nodes <= EOD_NODE_IDS:
        raise ContractError("TIMING_NODE_SET_INVALID")
    if set(proven_available_at) != required_nodes:
        raise ContractError("TIMING_NODE_SET_INVALID")
    for flag in (synthetic, recovered_unknown, recomputed):
        if type(flag) is not bool:
            raise ContractError("TIMING_FLAG_INVALID")
    stamps = [utc_stamp(value) for value in proven_available_at.values() if value is not None]
    bound = utc_stamp(deadline) if deadline is not None else None
    if deadline_ref is not None:
        validate_ref(deadline_ref)
    known = bound is not None and deadline_ref is not None and len(stamps) == len(required_nodes)
    if synthetic or recomputed:
        classification = "RETROSPECTIVE_RECOMPUTE"
    elif recovered_unknown or not known:
        classification = "UNKNOWN_LEGACY"
    elif bound is not None and all(stamp <= bound for stamp in stamps):
        classification = "CONTEMPORANEOUS"
    else:
        classification = "LATE_REGISTERED"
    return {
        "classification": classification,
        "prospective": classification == "CONTEMPORANEOUS",
        "synthetic": synthetic,
        "authority": "NONE",
    }
