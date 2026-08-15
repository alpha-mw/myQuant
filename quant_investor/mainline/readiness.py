"""Mainline-owned composition of candidate-bound readiness artifacts.

The input Intelligence readiness remains candidate-free.  This module alone
validates a Mainline candidate and binds its exact ref into the shared compiled
readiness contract.  Neither path has generation or activation authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from quant_investor.contracts import seal_artifact
from quant_investor.intelligence._common import (
    IntelligenceError,
    artifact_identity,
    artifact_payload,
    artifact_ref,
    canonical_value,
    identifier,
    require_no_future,
    timestamp,
    validate_artifact_ref,
)
from quant_investor.intelligence.runtime import validate_readiness

from .candidate import validate_mainline_candidate

_ACTIVE_ADMISSION_ROUTES: Final = frozenset({"BOOTSTRAP_EXCEPTION", "PROSPECTIVE_ADMISSION"})
_MAINLINE_ABSENCE_BLOCKER: Final = "MAINLINE_CANDIDATE_ABSENT"


def _validated_blockers(payload: Mapping[str, Any]) -> list[str]:
    blockers = payload.get("blockers")
    if type(blockers) is not list:
        raise IntelligenceError("Mainline readiness blockers must be a list")
    normalized = [
        identifier(code, label=f"blockers[{index}]") for index, code in enumerate(blockers)
    ]
    if normalized != sorted(set(normalized), key=lambda item: item.encode("ascii")):
        raise IntelligenceError("Mainline readiness blockers must be sorted and unique")
    return normalized


def _validate_factor_projection(payload: Mapping[str, Any]) -> str:
    factor_state = payload.get("factor_state")
    route = payload.get("admission_route")
    producer = identifier(payload.get("producer_identity"), label="producer_identity")
    factor_status_ref = payload.get("factor_status_ref")
    if factor_state not in {"READY", "BLOCKED"}:
        raise IntelligenceError("Mainline readiness Factor state is invalid")
    if factor_status_ref is not None:
        factor_status_ref = validate_artifact_ref(
            factor_status_ref,
            label="factor_status_ref",
        )
        if factor_status_ref["kind"] != "factor.status":
            raise IntelligenceError("Mainline readiness Factor status kind is invalid")
    if factor_state == "READY":
        if (
            route not in _ACTIVE_ADMISSION_ROUTES
            or factor_status_ref is None
            or producer not in {"NOT_CLAIMED", "PROSPECTIVE_GOVERNANCE"}
        ):
            raise IntelligenceError("ready Mainline readiness Factor binding is invalid")
    elif route != "NONE":
        raise IntelligenceError("blocked Mainline readiness admission route is invalid")
    return factor_state


def validate_mainline_readiness(
    artifact: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    """Validate one candidate-bound readiness artifact without resolving storage."""

    normalized, payload = artifact_payload(
        artifact,
        expected_kind="intelligence_readiness",
    )
    artifact_identity(payload.get("readiness_id"), label="readiness_id")
    factor_state = _validate_factor_projection(payload)
    blockers = _validated_blockers(payload)
    if _MAINLINE_ABSENCE_BLOCKER in blockers:
        raise IntelligenceError("Mainline readiness retains candidate absence")
    candidate_ref = validate_artifact_ref(
        payload.get("mainline_candidate_ref"),
        label="mainline_candidate_ref",
    )
    if candidate_ref["kind"] != "mainline_candidate":
        raise IntelligenceError("Mainline readiness candidate kind is invalid")
    mainline_state = payload.get("mainline_state")
    investment_state = payload.get("investment_state")
    if mainline_state == "READY":
        if factor_state != "READY" or blockers or investment_state != "PAPER_CANDIDATE":
            raise IntelligenceError("ready Mainline readiness state is inconsistent")
    elif mainline_state == "BLOCKED":
        if investment_state != "BLOCKED":
            raise IntelligenceError("blocked Mainline readiness state is inconsistent")
    else:
        raise IntelligenceError("Mainline readiness state is invalid")
    return normalized


def compose_mainline_readiness(
    readiness: Mapping[str, Any] | bytes,
    *,
    mainline_candidate: Mapping[str, Any] | bytes,
    assessed_at: str | None = None,
    readiness_id: str | None = None,
) -> dict[str, Any]:
    """Bind one validated candidate to candidate-free Intelligence readiness."""

    base = validate_readiness(readiness)
    instant = timestamp(assessed_at or base["created_at"], label="assessed_at")
    require_no_future(base, as_of=instant, label="intelligence_readiness")
    candidate = validate_mainline_candidate(mainline_candidate)
    require_no_future(candidate, as_of=instant, label="mainline_candidate")
    base_payload = base["payload"]
    blockers = [code for code in base_payload["blockers"] if code != _MAINLINE_ABSENCE_BLOCKER]
    if base_payload["factor_state"] == "READY" and not blockers:
        mainline_state = "READY"
        investment_state = "PAPER_CANDIDATE"
    else:
        mainline_state = "BLOCKED"
        investment_state = "BLOCKED"
    identity = readiness_id or base["artifact_id"]
    payload = {
        "admission_route": base_payload["admission_route"],
        "blockers": blockers,
        "factor_state": base_payload["factor_state"],
        "factor_status_ref": base_payload["factor_status_ref"],
        "investment_state": investment_state,
        "mainline_candidate_ref": artifact_ref(candidate),
        "mainline_state": mainline_state,
        "producer_identity": base_payload["producer_identity"],
        "readiness_id": artifact_identity(identity, label="readiness_id"),
    }
    canonical_value(payload)
    return validate_mainline_readiness(
        seal_artifact("intelligence_readiness", payload, created_at=instant)
    )


__all__ = ["compose_mainline_readiness", "validate_mainline_readiness"]
