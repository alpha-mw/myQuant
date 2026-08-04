"""Read-only deployability audit helpers for V4 regime-evidence v2 chains.

This module does not read, write, or validate V4 artifacts.  It records the
structural consequence of the V4 v2 predecessor chain so Sprint 1D can report
whether a long-lived chain is deployable under the sealed V4 replay limits.
Actual V4 replay outcomes are supplied by tests or an external audit caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable

V4_REGIME_REPLAY_MAX_DEPTH: Final = 5
V4_REGIME_REPLAY_MAX_NODES: Final = 16
V4_REGIME_REPLAY_MAX_REFERENCES: Final = 128
V4_REGIME_REPLAY_MAX_VALIDATION_SECONDS: Final = 10


@dataclass(frozen=True)
class ReplayLimitSnapshot:
    """The V4 producer replay limits relevant to a predecessor chain."""

    max_depth: int = V4_REGIME_REPLAY_MAX_DEPTH
    max_nodes: int = V4_REGIME_REPLAY_MAX_NODES
    max_references: int = V4_REGIME_REPLAY_MAX_REFERENCES
    max_validation_seconds: int = V4_REGIME_REPLAY_MAX_VALIDATION_SECONDS


@dataclass(frozen=True)
class ObservedReplayOutcome:
    """Observed result from an actual synthetic V4 replay/build probe."""

    max_successful_sessions: int
    first_failing_session: int | None
    first_failure_blocker: str | None


@dataclass(frozen=True)
class ChainLengthAudit:
    """Deterministic deployability estimate for one requested chain length."""

    session_count: int
    estimated_closure_depth: int
    estimated_node_count: int
    estimated_byte_count: int
    replay_duration_seconds: str
    peak_memory_bytes: str
    replay_status: str
    blocker_codes: tuple[str, ...]
    first_failing_session: int | None
    first_failure_blocker: str | None


@dataclass(frozen=True)
class MissedSessionRecoveryAudit:
    """Result of S0, S1, missing S2, then attempted S3 recovery."""

    scenario: str
    replay_status: str
    blocker_codes: tuple[str, ...]
    liveness_blocker: str | None


@dataclass(frozen=True)
class RegimeChainDeployabilityAudit:
    """Read-only audit summary for the V4 regime-evidence v2 chain shape."""

    v4_limits: ReplayLimitSnapshot
    length_results: tuple[ChainLengthAudit, ...]
    missed_session_recovery: MissedSessionRecoveryAudit
    scalability_blocker: str | None
    liveness_blocker: str | None


def estimate_predecessor_chain_depth(session_count: int) -> int:
    """Return the deepest evidence/model predecessor path for a v2 chain.

    A normal v2 evidence points to its model snapshot; the model snapshot points
    to the contiguous predecessor evidence.  The deepest replay path therefore
    alternates evidence -> model -> predecessor evidence for every retained
    session and grows linearly with history.
    """

    if type(session_count) is not int or session_count < 1:
        raise ValueError("session_count must be a positive integer")
    return session_count * 2


def estimate_predecessor_chain_nodes(session_count: int) -> int:
    """Return a conservative unique-node estimate for one latest replay.

    The latest session contributes evidence, feature, model, transition, and
    sealed feature source terminals.  Each retained predecessor can recursively
    replay the same registered closure shape, so the deployability audit uses a
    deliberately conservative linear estimate.
    """

    if type(session_count) is not int or session_count < 1:
        raise ValueError("session_count must be a positive integer")
    nodes_per_session = 8
    return session_count * nodes_per_session


def estimate_predecessor_chain_bytes(
    session_count: int,
    *,
    bytes_per_session: int = 4096,
) -> int:
    """Return a deterministic byte estimate for reporting, not validation."""

    if type(bytes_per_session) is not int or bytes_per_session < 1:
        raise ValueError("bytes_per_session must be a positive integer")
    if type(session_count) is not int or session_count < 1:
        raise ValueError("session_count must be a positive integer")
    return session_count * bytes_per_session


def audit_regime_chain_deployability(
    *,
    session_counts: Iterable[int],
    observed_replay: ObservedReplayOutcome,
    missed_session_recovery: MissedSessionRecoveryAudit,
    limits: ReplayLimitSnapshot = ReplayLimitSnapshot(),
) -> RegimeChainDeployabilityAudit:
    """Build a read-only deployability report for requested chain lengths."""

    results: list[ChainLengthAudit] = []
    scalability_blocker: str | None = None
    for session_count in tuple(session_counts):
        depth = estimate_predecessor_chain_depth(session_count)
        nodes = estimate_predecessor_chain_nodes(session_count)
        estimated_bytes = estimate_predecessor_chain_bytes(session_count)
        blockers: list[str] = []
        if depth > limits.max_depth:
            blockers.append("V4_REGIME_CHAIN_SCALABILITY_GAP")
        if nodes > limits.max_nodes:
            blockers.append("V4_REGIME_CHAIN_NODE_LIMIT_GAP")
        if (
            observed_replay.first_failing_session is not None
            and session_count >= observed_replay.first_failing_session
        ):
            blockers.append("V4_REGIME_REPLAY_FIRST_FAILURE_CONFIRMED")
        replay_status = "BLOCKED" if blockers else "REPLAYABLE_WITHIN_OBSERVED_BOUND"
        if session_count >= 260 and blockers and scalability_blocker is None:
            scalability_blocker = "V4_REGIME_CHAIN_SCALABILITY_GAP"
        results.append(
            ChainLengthAudit(
                session_count=session_count,
                estimated_closure_depth=depth,
                estimated_node_count=nodes,
                estimated_byte_count=estimated_bytes,
                replay_duration_seconds=("NOT_MEASURED_AFTER_FIRST_FAILURE"),
                peak_memory_bytes="NOT_MEASURED_AFTER_FIRST_FAILURE",
                replay_status=replay_status,
                blocker_codes=tuple(blockers),
                first_failing_session=observed_replay.first_failing_session,
                first_failure_blocker=observed_replay.first_failure_blocker,
            )
        )
    liveness_blocker = missed_session_recovery.liveness_blocker
    return RegimeChainDeployabilityAudit(
        v4_limits=limits,
        length_results=tuple(results),
        missed_session_recovery=missed_session_recovery,
        scalability_blocker=scalability_blocker,
        liveness_blocker=liveness_blocker,
    )


__all__ = [
    "ChainLengthAudit",
    "MissedSessionRecoveryAudit",
    "ObservedReplayOutcome",
    "RegimeChainDeployabilityAudit",
    "ReplayLimitSnapshot",
    "audit_regime_chain_deployability",
    "estimate_predecessor_chain_bytes",
    "estimate_predecessor_chain_depth",
    "estimate_predecessor_chain_nodes",
]
