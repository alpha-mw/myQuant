"""Pure FactorGovernanceProtocol v4.1 cycle-state contract.

The contract models only the immutable state artifacts for one explicitly
identified governance cycle.  It deliberately performs no file discovery,
registry access, production mutation, replay, statistics, or network work.
Callers supply the normalized predecessor artifact and both CAS identities for
every transition.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any


STATE_SCHEMA_VERSION = "factor-governance-cycle-state.v4.1"
SCHEMA_VERSION = STATE_SCHEMA_VERSION
PROTOCOL_VERSION = "v4"
GENESIS_SHA256 = "0" * 64

PRECOMMITTED = "PRECOMMITTED"
DISCOVERY = "DISCOVERY"
HOLDOUT_READY = "HOLDOUT_READY"
HOLDOUT_UNSEALED_FINALIZING = "HOLDOUT_UNSEALED_FINALIZING"
TERMINAL = "TERMINAL"

STATE_SEQUENCE = (
    PRECOMMITTED,
    DISCOVERY,
    HOLDOUT_READY,
    HOLDOUT_UNSEALED_FINALIZING,
    TERMINAL,
)
STATES = STATE_SEQUENCE
NEXT_STATE_BY_STATE: dict[str, str | None] = {
    state: STATE_SEQUENCE[index + 1] if index + 1 < len(STATE_SEQUENCE) else None
    for index, state in enumerate(STATE_SEQUENCE)
}
PREDECESSOR_STATE_BY_STATE: dict[str, str | None] = {
    state: STATE_SEQUENCE[index - 1] if index else None
    for index, state in enumerate(STATE_SEQUENCE)
}
HOLDOUT_UNSEALED_BY_STATE = {
    PRECOMMITTED: False,
    DISCOVERY: False,
    HOLDOUT_READY: False,
    HOLDOUT_UNSEALED_FINALIZING: True,
    TERMINAL: True,
}

_STATE_FIELDS = {
    "schema_version",
    "protocol_version",
    "cycle_id",
    "cycle_root_sha256",
    "state",
    "expected_predecessor_state",
    "predecessor",
    "source_chain_node_sha256",
    "holdout_unsealed",
    "terminal_reason",
    "allowed_next_state",
    "state_semantic_sha256",
}
_PREDECESSOR_FIELDS = {"kind", "byte_sha256", "semantic_sha256"}
_SAFE_CYCLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}")


class FactorGovernanceCycleStateV4_1Error(ValueError):
    """Raised when a v4.1 cycle-state artifact fails closed."""


# A compact alias is convenient for callers that do not use underscores in
# versioned class names.
FactorGovernanceCycleStateV41Error = FactorGovernanceCycleStateV4_1Error


def canonical_json_bytes(value: Any) -> bytes:
    """Return compact sorted finite JSON bytes with no trailing newline."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, TypeError, ValueError) as exc:
        raise FactorGovernanceCycleStateV4_1Error(
            f"value is not canonical finite JSON: {exc}"
        ) from exc


def semantic_sha256(value: Any) -> str:
    """Hash canonical semantic JSON bytes, explicitly excluding a newline."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_semantic_sha256_v4_1(value: Any) -> str:
    """Version-labelled alias for :func:`semantic_sha256`."""

    return semantic_sha256(value)


def canonical_file_bytes(value: Any) -> bytes:
    """Return the one canonical file representation used for byte CAS hashes."""

    return canonical_json_bytes(value) + b"\n"


def byte_sha256(value: Any) -> str:
    """Hash canonical artifact bytes, including exactly one final newline."""

    return hashlib.sha256(canonical_file_bytes(value)).hexdigest()


def cycle_state_byte_sha256_v4_1(value: Any) -> str:
    """Validate a cycle artifact and return its canonical byte identity."""

    return byte_sha256(validate_cycle_state_v4_1(value))


def _exact_object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FactorGovernanceCycleStateV4_1Error(f"{label} must be an object")
    payload = dict(value)
    if any(type(key) is not str for key in payload):
        raise FactorGovernanceCycleStateV4_1Error(
            f"{label} field names must be strings"
        )
    missing = sorted(fields - set(payload))
    unknown = sorted(set(payload) - fields)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise FactorGovernanceCycleStateV4_1Error(
            f"{label} fields invalid: {';'.join(details)}"
        )
    return payload


def _sha256(value: Any, label: str, *, allow_genesis: bool = False) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FactorGovernanceCycleStateV4_1Error(
            f"{label} must be lowercase SHA-256"
        )
    if value == GENESIS_SHA256 and not allow_genesis:
        raise FactorGovernanceCycleStateV4_1Error(
            f"{label} must be a nonzero lowercase SHA-256"
        )
    return value


def _cycle_id(value: Any) -> str:
    if (
        type(value) is not str
        or _SAFE_CYCLE_ID.fullmatch(value) is None
        or ".." in value
    ):
        raise FactorGovernanceCycleStateV4_1Error(
            "cycle_id must be an exact safe non-empty path segment"
        )
    return value


def _state(value: Any, label: str = "state") -> str:
    if type(value) is not str or value not in STATE_SEQUENCE:
        raise FactorGovernanceCycleStateV4_1Error(
            f"{label} must be one of {','.join(STATE_SEQUENCE)}"
        )
    return value


def _terminal_reason(value: Any, *, state: str) -> str | None:
    if state != TERMINAL:
        if value is not None:
            raise FactorGovernanceCycleStateV4_1Error(
                "terminal_reason must be null before TERMINAL"
            )
        return None
    if type(value) is not str or not value or value != value.strip():
        raise FactorGovernanceCycleStateV4_1Error(
            "terminal_reason must be an exact non-empty string at TERMINAL"
        )
    return value


def _predecessor_descriptor(value: Any, *, state: str) -> dict[str, str]:
    payload = _exact_object(value, _PREDECESSOR_FIELDS, "predecessor")
    kind = payload["kind"]
    if type(kind) is not str:
        raise FactorGovernanceCycleStateV4_1Error(
            "predecessor.kind must be a string"
        )
    if state == PRECOMMITTED:
        expected = {
            "kind": "genesis",
            "byte_sha256": GENESIS_SHA256,
            "semantic_sha256": GENESIS_SHA256,
        }
        if payload != expected:
            raise FactorGovernanceCycleStateV4_1Error(
                "PRECOMMITTED predecessor must be exact genesis"
            )
        return expected
    if kind != "cycle_state":
        raise FactorGovernanceCycleStateV4_1Error(
            "non-genesis predecessor.kind must be cycle_state"
        )
    return {
        "kind": kind,
        "byte_sha256": _sha256(
            payload["byte_sha256"], "predecessor.byte_sha256"
        ),
        "semantic_sha256": _sha256(
            payload["semantic_sha256"], "predecessor.semantic_sha256"
        ),
    }


def _state_semantic_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key != "state_semantic_sha256"
    }


def validate_cycle_state_v4_1(
    value: Mapping[str, Any],
    *,
    expected_cycle_id: str | None = None,
    expected_cycle_root_sha256: str | None = None,
    expected_state: str | None = None,
) -> dict[str, Any]:
    """Validate and normalize one self-sealed v4.1 cycle-state artifact."""

    payload = _exact_object(value, _STATE_FIELDS, "cycle state")
    # This catches non-JSON values before any normalized artifact is returned.
    canonical_json_bytes(payload)

    if payload["schema_version"] != STATE_SCHEMA_VERSION:
        raise FactorGovernanceCycleStateV4_1Error("cycle state schema mismatch")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise FactorGovernanceCycleStateV4_1Error("protocol_version must be v4")

    normalized_cycle_id = _cycle_id(payload["cycle_id"])
    normalized_root = _sha256(
        payload["cycle_root_sha256"], "cycle_root_sha256"
    )
    normalized_state = _state(payload["state"])
    expected_predecessor_state = PREDECESSOR_STATE_BY_STATE[normalized_state]
    if payload["expected_predecessor_state"] != expected_predecessor_state:
        raise FactorGovernanceCycleStateV4_1Error(
            "expected_predecessor_state does not match the exact state sequence"
        )
    predecessor = _predecessor_descriptor(
        payload["predecessor"], state=normalized_state
    )
    source_chain_node_sha = _sha256(
        payload["source_chain_node_sha256"], "source_chain_node_sha256"
    )

    holdout_unsealed = payload["holdout_unsealed"]
    if type(holdout_unsealed) is not bool:
        raise FactorGovernanceCycleStateV4_1Error(
            "holdout_unsealed must be a boolean"
        )
    if holdout_unsealed is not HOLDOUT_UNSEALED_BY_STATE[normalized_state]:
        raise FactorGovernanceCycleStateV4_1Error(
            "holdout_unsealed violates the monotonic state contract"
        )

    terminal_reason = _terminal_reason(
        payload["terminal_reason"], state=normalized_state
    )
    allowed_next_state = NEXT_STATE_BY_STATE[normalized_state]
    if payload["allowed_next_state"] != allowed_next_state:
        raise FactorGovernanceCycleStateV4_1Error(
            "allowed_next_state does not match the exact state sequence"
        )

    supplied_semantic_sha = _sha256(
        payload["state_semantic_sha256"], "state_semantic_sha256"
    )
    expected_semantic_sha = semantic_sha256(_state_semantic_payload(payload))
    if supplied_semantic_sha != expected_semantic_sha:
        raise FactorGovernanceCycleStateV4_1Error(
            "state_semantic_sha256 mismatch"
        )

    if expected_cycle_id is not None:
        if normalized_cycle_id != _cycle_id(expected_cycle_id):
            raise FactorGovernanceCycleStateV4_1Error("cycle_id identity mismatch")
    if expected_cycle_root_sha256 is not None:
        if normalized_root != _sha256(
            expected_cycle_root_sha256, "expected_cycle_root_sha256"
        ):
            raise FactorGovernanceCycleStateV4_1Error(
                "cycle_root_sha256 identity mismatch"
            )
    if expected_state is not None and normalized_state != _state(
        expected_state, "expected_state"
    ):
        raise FactorGovernanceCycleStateV4_1Error("state identity mismatch")

    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": normalized_cycle_id,
        "cycle_root_sha256": normalized_root,
        "state": normalized_state,
        "expected_predecessor_state": expected_predecessor_state,
        "predecessor": predecessor,
        "source_chain_node_sha256": source_chain_node_sha,
        "holdout_unsealed": holdout_unsealed,
        "terminal_reason": terminal_reason,
        "allowed_next_state": allowed_next_state,
        "state_semantic_sha256": supplied_semantic_sha,
    }


def validate_factor_governance_cycle_state_v4_1(
    value: Mapping[str, Any],
    *,
    expected_cycle_id: str | None = None,
    expected_cycle_root_sha256: str | None = None,
    expected_state: str | None = None,
) -> dict[str, Any]:
    """Long-form alias for :func:`validate_cycle_state_v4_1`."""

    return validate_cycle_state_v4_1(
        value,
        expected_cycle_id=expected_cycle_id,
        expected_cycle_root_sha256=expected_cycle_root_sha256,
        expected_state=expected_state,
    )


def _seal_state(payload: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(payload)
    sealed["state_semantic_sha256"] = semantic_sha256(payload)
    return validate_cycle_state_v4_1(sealed)


def build_genesis_cycle_state_v4_1(
    *,
    cycle_id: str,
    cycle_root_sha256: str,
    source_chain_node_sha256: str,
) -> dict[str, Any]:
    """Build the sole genesis state, ``PRECOMMITTED``."""

    normalized_cycle_id = _cycle_id(cycle_id)
    normalized_root = _sha256(cycle_root_sha256, "cycle_root_sha256")
    normalized_source = _sha256(
        source_chain_node_sha256, "source_chain_node_sha256"
    )
    return _seal_state(
        {
            "schema_version": STATE_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "cycle_id": normalized_cycle_id,
            "cycle_root_sha256": normalized_root,
            "state": PRECOMMITTED,
            "expected_predecessor_state": None,
            "predecessor": {
                "kind": "genesis",
                "byte_sha256": GENESIS_SHA256,
                "semantic_sha256": GENESIS_SHA256,
            },
            "source_chain_node_sha256": normalized_source,
            "holdout_unsealed": False,
            "terminal_reason": None,
            "allowed_next_state": DISCOVERY,
        }
    )


def validate_genesis_cycle_state_v4_1(
    value: Mapping[str, Any],
    *,
    expected_cycle_id: str | None = None,
    expected_cycle_root_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate that an artifact is the exact ``PRECOMMITTED`` genesis."""

    return validate_cycle_state_v4_1(
        value,
        expected_cycle_id=expected_cycle_id,
        expected_cycle_root_sha256=expected_cycle_root_sha256,
        expected_state=PRECOMMITTED,
    )


def _transition_predecessor(
    *,
    predecessor: Mapping[str, Any],
    predecessor_byte_sha256: str,
    expected_predecessor_byte_sha256: str,
    expected_predecessor_semantic_sha256: str,
    cycle_id: str,
    cycle_root_sha256: str,
    next_state: str,
) -> tuple[dict[str, Any], str, str]:
    normalized_cycle_id = _cycle_id(cycle_id)
    normalized_root = _sha256(cycle_root_sha256, "cycle_root_sha256")
    requested_state = _state(next_state, "next_state")
    normalized_predecessor = validate_cycle_state_v4_1(predecessor)

    if normalized_predecessor["cycle_id"] != normalized_cycle_id:
        raise FactorGovernanceCycleStateV4_1Error(
            "cross-cycle predecessor substitution"
        )
    if normalized_predecessor["cycle_root_sha256"] != normalized_root:
        raise FactorGovernanceCycleStateV4_1Error(
            "cross-root predecessor substitution"
        )

    supplied_byte_sha = _sha256(
        predecessor_byte_sha256, "predecessor_byte_sha256"
    )
    expected_byte_sha = _sha256(
        expected_predecessor_byte_sha256,
        "expected_predecessor_byte_sha256",
    )
    expected_semantic_sha = _sha256(
        expected_predecessor_semantic_sha256,
        "expected_predecessor_semantic_sha256",
    )
    actual_byte_sha = byte_sha256(normalized_predecessor)
    actual_semantic_sha = normalized_predecessor["state_semantic_sha256"]
    if supplied_byte_sha != actual_byte_sha:
        raise FactorGovernanceCycleStateV4_1Error(
            "predecessor byte SHA does not match the normalized artifact"
        )
    if expected_byte_sha != supplied_byte_sha:
        raise FactorGovernanceCycleStateV4_1Error(
            "stale predecessor byte SHA CAS"
        )
    if expected_semantic_sha != actual_semantic_sha:
        raise FactorGovernanceCycleStateV4_1Error(
            "stale predecessor semantic SHA CAS"
        )

    predecessor_state = normalized_predecessor["state"]
    if predecessor_state == TERMINAL:
        raise FactorGovernanceCycleStateV4_1Error(
            "TERMINAL cycle state cannot be reopened or restarted"
        )
    if (
        predecessor_state == HOLDOUT_UNSEALED_FINALIZING
        and requested_state == HOLDOUT_UNSEALED_FINALIZING
    ):
        raise FactorGovernanceCycleStateV4_1Error(
            "second or replayed holdout unseal is forbidden"
        )
    required_next = NEXT_STATE_BY_STATE[predecessor_state]
    if requested_state != required_next:
        raise FactorGovernanceCycleStateV4_1Error(
            f"transition must advance exactly {predecessor_state} -> {required_next}"
        )
    return normalized_predecessor, actual_byte_sha, actual_semantic_sha


def build_next_cycle_state_v4_1(
    *,
    predecessor: Mapping[str, Any],
    predecessor_byte_sha256: str,
    expected_predecessor_byte_sha256: str,
    expected_predecessor_semantic_sha256: str,
    cycle_id: str,
    cycle_root_sha256: str,
    next_state: str,
    source_chain_node_sha256: str,
    terminal_reason: str | None = None,
) -> dict[str, Any]:
    """Build exactly the next state after passing byte and semantic CAS."""

    normalized_predecessor, actual_byte_sha, actual_semantic_sha = (
        _transition_predecessor(
            predecessor=predecessor,
            predecessor_byte_sha256=predecessor_byte_sha256,
            expected_predecessor_byte_sha256=expected_predecessor_byte_sha256,
            expected_predecessor_semantic_sha256=(
                expected_predecessor_semantic_sha256
            ),
            cycle_id=cycle_id,
            cycle_root_sha256=cycle_root_sha256,
            next_state=next_state,
        )
    )
    normalized_source = _sha256(
        source_chain_node_sha256, "source_chain_node_sha256"
    )
    normalized_terminal_reason = _terminal_reason(
        terminal_reason, state=next_state
    )
    artifact = _seal_state(
        {
            "schema_version": STATE_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "cycle_id": normalized_predecessor["cycle_id"],
            "cycle_root_sha256": normalized_predecessor["cycle_root_sha256"],
            "state": next_state,
            "expected_predecessor_state": normalized_predecessor["state"],
            "predecessor": {
                "kind": "cycle_state",
                "byte_sha256": actual_byte_sha,
                "semantic_sha256": actual_semantic_sha,
            },
            "source_chain_node_sha256": normalized_source,
            "holdout_unsealed": HOLDOUT_UNSEALED_BY_STATE[next_state],
            "terminal_reason": normalized_terminal_reason,
            "allowed_next_state": NEXT_STATE_BY_STATE[next_state],
        }
    )
    return validate_next_cycle_state_v4_1(
        artifact,
        predecessor=normalized_predecessor,
        predecessor_byte_sha256=actual_byte_sha,
        expected_predecessor_byte_sha256=actual_byte_sha,
        expected_predecessor_semantic_sha256=actual_semantic_sha,
        cycle_id=normalized_predecessor["cycle_id"],
        cycle_root_sha256=normalized_predecessor["cycle_root_sha256"],
    )


def validate_next_cycle_state_v4_1(
    value: Mapping[str, Any],
    *,
    predecessor: Mapping[str, Any],
    predecessor_byte_sha256: str,
    expected_predecessor_byte_sha256: str,
    expected_predecessor_semantic_sha256: str,
    cycle_id: str,
    cycle_root_sha256: str,
) -> dict[str, Any]:
    """Validate one transition artifact against its explicit predecessor CAS."""

    normalized = validate_cycle_state_v4_1(
        value,
        expected_cycle_id=cycle_id,
        expected_cycle_root_sha256=cycle_root_sha256,
    )
    normalized_predecessor, actual_byte_sha, actual_semantic_sha = (
        _transition_predecessor(
            predecessor=predecessor,
            predecessor_byte_sha256=predecessor_byte_sha256,
            expected_predecessor_byte_sha256=expected_predecessor_byte_sha256,
            expected_predecessor_semantic_sha256=(
                expected_predecessor_semantic_sha256
            ),
            cycle_id=cycle_id,
            cycle_root_sha256=cycle_root_sha256,
            next_state=normalized["state"],
        )
    )
    expected_descriptor = {
        "kind": "cycle_state",
        "byte_sha256": actual_byte_sha,
        "semantic_sha256": actual_semantic_sha,
    }
    if normalized["expected_predecessor_state"] != normalized_predecessor["state"]:
        raise FactorGovernanceCycleStateV4_1Error(
            "transition predecessor state identity mismatch"
        )
    if normalized["predecessor"] != expected_descriptor:
        raise FactorGovernanceCycleStateV4_1Error(
            "transition predecessor byte/semantic descriptor mismatch"
        )
    return normalized


def build_factor_governance_cycle_state_genesis_v4_1(
    *,
    cycle_id: str,
    cycle_root_sha256: str,
    source_chain_node_sha256: str,
) -> dict[str, Any]:
    """Long-form genesis builder alias."""

    return build_genesis_cycle_state_v4_1(
        cycle_id=cycle_id,
        cycle_root_sha256=cycle_root_sha256,
        source_chain_node_sha256=source_chain_node_sha256,
    )


def build_factor_governance_cycle_state_transition_v4_1(
    *,
    predecessor: Mapping[str, Any],
    predecessor_byte_sha256: str,
    expected_predecessor_byte_sha256: str,
    expected_predecessor_semantic_sha256: str,
    cycle_id: str,
    cycle_root_sha256: str,
    state: str,
    source_chain_node_sha256: str,
    terminal_reason: str | None = None,
) -> dict[str, Any]:
    """Long-form transition builder using ``state`` as the target name."""

    return build_next_cycle_state_v4_1(
        predecessor=predecessor,
        predecessor_byte_sha256=predecessor_byte_sha256,
        expected_predecessor_byte_sha256=expected_predecessor_byte_sha256,
        expected_predecessor_semantic_sha256=(
            expected_predecessor_semantic_sha256
        ),
        cycle_id=cycle_id,
        cycle_root_sha256=cycle_root_sha256,
        next_state=state,
        source_chain_node_sha256=source_chain_node_sha256,
        terminal_reason=terminal_reason,
    )


__all__ = [
    "DISCOVERY",
    "FactorGovernanceCycleStateV41Error",
    "FactorGovernanceCycleStateV4_1Error",
    "GENESIS_SHA256",
    "HOLDOUT_READY",
    "HOLDOUT_UNSEALED_BY_STATE",
    "HOLDOUT_UNSEALED_FINALIZING",
    "NEXT_STATE_BY_STATE",
    "PRECOMMITTED",
    "PREDECESSOR_STATE_BY_STATE",
    "PROTOCOL_VERSION",
    "SCHEMA_VERSION",
    "STATES",
    "STATE_SCHEMA_VERSION",
    "STATE_SEQUENCE",
    "TERMINAL",
    "build_factor_governance_cycle_state_genesis_v4_1",
    "build_factor_governance_cycle_state_transition_v4_1",
    "build_genesis_cycle_state_v4_1",
    "build_next_cycle_state_v4_1",
    "byte_sha256",
    "canonical_file_bytes",
    "canonical_json_bytes",
    "canonical_semantic_sha256_v4_1",
    "cycle_state_byte_sha256_v4_1",
    "semantic_sha256",
    "validate_cycle_state_v4_1",
    "validate_factor_governance_cycle_state_v4_1",
    "validate_genesis_cycle_state_v4_1",
    "validate_next_cycle_state_v4_1",
]
