"""Offline skeptical advisory review with no deterministic control authority."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, Final

from .._common import (
    IntelligenceError,
    NO_AUTHORITY,
    artifact_payload,
    artifact_identity,
    artifact_ref,
    build_artifact,
    business_identity,
    company_code,
    decimal_text,
    decimal_value,
    identifier,
    require_artifact_ref,
    require_no_future,
    timestamp,
    validate_public_https_url,
)
from ..investment_decision import validate_investment_decision

ADVISORY_STATUSES: Final = frozenset({"AVAILABLE", "ADVISORY_UNAVAILABLE"})
MAX_ADVISORY_DELTA: Final = Decimal("0.10")


def _facts(values: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise IntelligenceError("validated_facts must be a sequence")
    rows: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    seen: set[str] = set()
    for index, value in enumerate(values):
        if type(value) is not dict or set(value) != {
            "confidence",
            "fact",
            "source_id",
            "source_url",
        }:
            raise IntelligenceError(f"validated_facts[{index}] shape is invalid")
        source_id = identifier(value["source_id"], label=f"validated_facts[{index}].source_id")
        fact = artifact_identity(value["fact"], label=f"validated_facts[{index}].fact")
        url = validate_public_https_url(
            value["source_url"], label=f"validated_facts[{index}].source_url"
        )
        confidence = decimal_value(
            value["confidence"],
            label=f"validated_facts[{index}].confidence",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
        identity = f"{source_id}\0{url}\0{fact}"
        if identity in seen:
            raise IntelligenceError("validated fact closure is duplicated")
        seen.add(identity)
        source_ids.add(source_id)
        rows.append(
            {
                "confidence": decimal_text(confidence),
                "fact": fact,
                "source_id": source_id,
                "source_url": url,
            }
        )
    rows.sort(
        key=lambda row: (
            row["source_id"].encode("utf-8"),
            row["source_url"].encode("utf-8"),
            row["fact"].encode("utf-8"),
        )
    )
    return rows, len(source_ids)


def review_advisory(
    *,
    decision: Mapping[str, Any] | bytes,
    proposed_percentile: Any | None,
    validated_facts: Sequence[Mapping[str, Any]],
    as_of: str,
    unresolved_source_conflict: bool = False,
    capability_available: bool = True,
    review_id: str | None = None,
) -> dict[str, Any]:
    """Replay prevalidated facts and emit a bounded, advisory-only challenge."""

    cutoff = timestamp(as_of, label="as_of")
    decision_artifact = validate_investment_decision(decision)
    require_no_future(decision_artifact, as_of=cutoff, label="investment_decision")
    decision_payload = decision_artifact["payload"]
    if type(unresolved_source_conflict) is not bool or type(capability_available) is not bool:
        raise IntelligenceError("advisory capability flags must be boolean")
    facts, independent_sources = _facts(validated_facts)
    deterministic_value = decision_payload.get("deterministic_percentile")
    deterministic = (
        None
        if deterministic_value is None
        else decimal_value(
            deterministic_value,
            label="deterministic_percentile",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
    )
    proposed = None
    if proposed_percentile is not None:
        proposed = decimal_value(
            proposed_percentile,
            label="proposed_percentile",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
    reasons: list[str] = []
    status = "AVAILABLE"
    if not capability_available or deterministic is None:
        status = "ADVISORY_UNAVAILABLE"
        reasons.append("ADVISORY_CAPABILITY_UNAVAILABLE")
    if unresolved_source_conflict:
        reasons.append("UNRESOLVED_SOURCE_CONFLICT")
    if independent_sources < 2:
        reasons.append("INSUFFICIENT_VALIDATED_FACT_AUTHORITY")
    may_adjust = status == "AVAILABLE" and not reasons and proposed is not None
    advisory = deterministic
    if may_adjust:
        delta = abs(proposed - deterministic)
        if delta > MAX_ADVISORY_DELTA:
            raise IntelligenceError("advisory change exceeds ten percent")
        advisory = proposed
    elif deterministic is not None:
        delta = Decimal("0")
    else:
        delta = Decimal("0")
    code = company_code(decision_payload.get("company_code"))
    return build_artifact(
        kind="advisory_review",
        identity_field="review_id",
        identity=review_id
        or business_identity(
            kind="advisory_review",
            identity_inputs={
                "as_of": cutoff,
                "company_code": code,
                "decision_id": decision_artifact["artifact_id"],
            },
        ),
        created_at=cutoff,
        fields={
            "absolute_delta": decimal_text(delta),
            "advisory_percentile": None if advisory is None else decimal_text(advisory),
            "as_of": cutoff,
            "company_code": code,
            "decision_ref": artifact_ref(decision_artifact),
            "deterministic_decision_state": decision_payload["state"],
            "deterministic_percentile": (
                None if deterministic is None else decimal_text(deterministic)
            ),
            "reason_codes": sorted(set(reasons), key=lambda item: item.encode("ascii")),
            "status": status,
            "validated_facts": facts,
        },
    )


def replay_advisory(
    artifact: Mapping[str, Any] | bytes,
    *,
    decision: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    """Validate an advisory artifact entirely offline against its exact decision."""

    normalized, payload = artifact_payload(artifact, expected_kind="advisory_review")
    decision_artifact = validate_investment_decision(decision)
    require_artifact_ref(payload.get("decision_ref"), decision_artifact, label="decision_ref")
    if payload.get("deterministic_decision_state") != decision_artifact["payload"].get("state"):
        raise IntelligenceError("advisory review changed deterministic decision authority")
    if payload.get("status") not in ADVISORY_STATUSES:
        raise IntelligenceError("advisory status is invalid")
    if (
        payload.get("authority") != NO_AUTHORITY
        or payload.get("research_only") is not True
        or payload.get("production") is not False
        or payload.get("run_state") != "INACTIVE"
    ):
        raise IntelligenceError("advisory review authority is invalid")
    company_code(payload.get("company_code"))
    timestamp(payload.get("as_of"), label="advisory.as_of")
    facts, _ = _facts(payload.get("validated_facts"))
    if facts != payload.get("validated_facts"):
        raise IntelligenceError("advisory fact closure is not canonical")
    decision_percentile = decision_artifact["payload"].get("deterministic_percentile")
    if payload.get("deterministic_percentile") != decision_percentile:
        raise IntelligenceError("advisory deterministic percentile binding differs")
    delta = decimal_value(
        payload.get("absolute_delta"),
        label="advisory.absolute_delta",
        minimum=Decimal("0"),
        maximum=MAX_ADVISORY_DELTA,
    )
    deterministic = (
        None
        if decision_percentile is None
        else decimal_value(
            decision_percentile,
            label="decision.deterministic_percentile",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
    )
    advisory_value = payload.get("advisory_percentile")
    advisory = (
        None
        if advisory_value is None
        else decimal_value(
            advisory_value,
            label="advisory.advisory_percentile",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
    )
    expected_delta = (
        Decimal("0") if deterministic is None or advisory is None else abs(advisory - deterministic)
    )
    if delta != expected_delta:
        raise IntelligenceError("advisory delta does not replay")
    if payload["reason_codes"] and delta != Decimal("0"):
        raise IntelligenceError("blocked advisory review cannot change rank")
    if payload["status"] == "ADVISORY_UNAVAILABLE" and advisory != deterministic:
        raise IntelligenceError("unavailable advisory cannot change rank")
    return normalized


__all__ = [
    "ADVISORY_STATUSES",
    "IntelligenceError",
    "MAX_ADVISORY_DELTA",
    "replay_advisory",
    "review_advisory",
    "validate_public_https_url",
]
