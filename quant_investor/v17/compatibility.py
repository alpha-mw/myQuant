"""Pure compatibility policy for retiring the v17 v1 shadow protocol.

This module deliberately performs no filesystem access.  It freezes the
version-dispatch matrix that a later CLI/runtime adapter must consult before
any directory creation, lock acquisition, receipt write, or artifact import.
Existing v1 terminal runs remain inspectable; every v1 write path and every
nonterminal v1 run is retired with exit code 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .contracts import V17ContractError

V1_LEDGER_VERSION: Final = "myquant.v17.shadow-ledger.v1"

V1_TERMINAL_STATES: Final = (
    "SHADOW_COMPLETE_AWAITING_HUMAN_DECISION",
    "SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
    "SHADOW_PORTFOLIO_INFEASIBLE",
    "HARD_STOP_SNAPSHOT_DRIFT",
    "HARD_STOP_INVALID_EVIDENCE",
)
V1_NONTERMINAL_STATES: Final = (
    "PREPARED",
    "DETERMINISTIC_COMPLETE",
    "DEEP_REQUEST_READY",
    "DEEP_RESPONSE_RECEIVED",
    "PORTFOLIO_COMPLETE",
)

READ_STATUS: Final = "READ_STATUS"
READ_ARTIFACT: Final = "READ_ARTIFACT"
CREATE_RUN: Final = "CREATE_RUN"
ADVANCE_RUN: Final = "ADVANCE_RUN"
RECEIVE_RESPONSE: Final = "RECEIVE_RESPONSE"
FINALIZE_RUN: Final = "FINALIZE_RUN"
REPAIR_LATEST: Final = "REPAIR_LATEST"

V1_READ_ONLY_ACTIONS: Final = (READ_STATUS, READ_ARTIFACT)
V1_MUTATING_ACTIONS: Final = (
    CREATE_RUN,
    ADVANCE_RUN,
    RECEIVE_RESPONSE,
    FINALIZE_RUN,
    REPAIR_LATEST,
)
V1_ACTIONS: Final = V1_READ_ONLY_ACTIONS + V1_MUTATING_ACTIONS


@dataclass(frozen=True)
class V1CompatibilityDecision:
    """One deterministic, side-effect-free v1 compatibility decision."""

    allowed: bool
    read_only: bool
    reason: str
    exit_code: int


class V17CompatibilityError(V17ContractError):
    """Raised before any retired v1 operation is allowed to touch storage."""

    exit_code = 2

    def __init__(self, decision: V1CompatibilityDecision) -> None:
        super().__init__(decision.reason)
        self.decision = decision


def decide_v1_compatibility(
    *,
    action: str,
    state: str | None,
    ledger_version: str = V1_LEDGER_VERSION,
) -> V1CompatibilityDecision:
    """Return the frozen v1 matrix decision without performing I/O."""

    if ledger_version != V1_LEDGER_VERSION:
        raise V17CompatibilityError(
            V1CompatibilityDecision(False, False, "unsupported_v1_ledger_version", 2)
        )
    if action not in V1_ACTIONS:
        raise V17CompatibilityError(
            V1CompatibilityDecision(False, False, "unknown_v1_compatibility_action", 2)
        )
    if action == CREATE_RUN:
        return V1CompatibilityDecision(False, False, "v1_new_run_retired", 2)
    if state is None:
        raise V17CompatibilityError(V1CompatibilityDecision(False, False, "v1_state_required", 2))
    if state not in V1_TERMINAL_STATES and state not in V1_NONTERMINAL_STATES:
        raise V17CompatibilityError(V1CompatibilityDecision(False, False, "unknown_v1_state", 2))
    if state in V1_NONTERMINAL_STATES:
        return V1CompatibilityDecision(False, False, "v1_nonterminal_retired", 2)
    if action in V1_READ_ONLY_ACTIONS:
        return V1CompatibilityDecision(True, True, "v1_terminal_read_only", 0)
    if action == REPAIR_LATEST:
        return V1CompatibilityDecision(False, False, "v1_latest_repair_retired", 2)
    return V1CompatibilityDecision(False, False, "v1_terminal_mutation_retired", 2)


def require_v1_compatibility(
    *,
    action: str,
    state: str | None,
    ledger_version: str = V1_LEDGER_VERSION,
) -> V1CompatibilityDecision:
    """Return an allowed read decision or raise the fixed exit-2 rejection."""

    decision = decide_v1_compatibility(
        action=action,
        state=state,
        ledger_version=ledger_version,
    )
    if not decision.allowed:
        raise V17CompatibilityError(decision)
    return decision


__all__ = [
    "ADVANCE_RUN",
    "CREATE_RUN",
    "FINALIZE_RUN",
    "READ_ARTIFACT",
    "READ_STATUS",
    "RECEIVE_RESPONSE",
    "REPAIR_LATEST",
    "V1_ACTIONS",
    "V1CompatibilityDecision",
    "V1_LEDGER_VERSION",
    "V1_MUTATING_ACTIONS",
    "V1_NONTERMINAL_STATES",
    "V1_READ_ONLY_ACTIONS",
    "V1_TERMINAL_STATES",
    "V17CompatibilityError",
    "decide_v1_compatibility",
    "require_v1_compatibility",
]
