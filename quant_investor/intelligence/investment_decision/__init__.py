"""Five-state deterministic investment research decision.

The decision is research authority only.  No state is an order, trade, target
position, or mainline activation instruction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, Final

from .._common import (
    IntelligenceError,
    NO_AUTHORITY,
    artifact_payload,
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
    validate_artifact_ref,
)
from ..decision_context import validate_decision_context

DECISION_STATES: Final = (
    "THESIS_INVALIDATED",
    "INSUFFICIENT_EVIDENCE",
    "WATCHLIST",
    "RESEARCH_APPROVED",
    "PAPER_CANDIDATE",
)


def _thresholds(value: Mapping[str, Any]) -> tuple[Decimal, Decimal, dict[str, str]]:
    if type(value) is not dict or set(value) != {"paper_candidate", "research_approved"}:
        raise IntelligenceError("decision thresholds shape is invalid")
    research = decimal_value(
        value["research_approved"],
        label="thresholds.research_approved",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    paper = decimal_value(
        value["paper_candidate"],
        label="thresholds.paper_candidate",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    if paper < research:
        raise IntelligenceError("paper threshold cannot be below research threshold")
    return (
        research,
        paper,
        {
            "paper_candidate": decimal_text(paper),
            "research_approved": decimal_text(research),
        },
    )


def _decision_state(
    *,
    context: Mapping[str, Any],
    percentile: Decimal | None,
    research_threshold: Decimal,
    paper_threshold: Decimal,
    additional_vetoes: Sequence[str],
) -> tuple[str, list[str], list[str]]:
    blockers = list(context.get("blocker_codes", []))
    if context.get("hypothesis_status") == "INVALIDATED":
        return "THESIS_INVALIDATED", blockers, ["PREREGISTERED_HYPOTHESIS_INVALIDATED"]
    if context.get("status") != "AVAILABLE" or percentile is None:
        if percentile is None:
            blockers.append("DETERMINISTIC_RANK_UNAVAILABLE")
        return (
            "INSUFFICIENT_EVIDENCE",
            sorted(set(blockers), key=lambda item: item.encode("ascii")),
            ["DECISION_CONTEXT_INCOMPLETE"],
        )
    vetoes = sorted(
        set(context.get("hard_risk_codes", [])) | set(additional_vetoes),
        key=lambda item: item.encode("ascii"),
    )
    if vetoes:
        return "WATCHLIST", [], ["HARD_RISK_VETO", *vetoes]
    if percentile < research_threshold:
        return "WATCHLIST", [], ["DETERMINISTIC_THRESHOLD_NOT_MET"]
    if percentile >= paper_threshold:
        return "PAPER_CANDIDATE", [], ["PAPER_THRESHOLD_MET"]
    return "RESEARCH_APPROVED", [], ["RESEARCH_THRESHOLD_MET"]


def make_investment_decision(
    *,
    context: Mapping[str, Any] | bytes,
    deterministic_percentile: Any | None,
    thresholds: Mapping[str, Any],
    as_of: str,
    hard_veto_codes: Sequence[str] = (),
    decision_id: str | None = None,
) -> dict[str, Any]:
    """Produce one deterministic five-state research decision."""

    cutoff = timestamp(as_of, label="as_of")
    context_artifact = validate_decision_context(context)
    require_no_future(context_artifact, as_of=cutoff, label="decision_context")
    context_payload = context_artifact["payload"]
    if context_payload.get("as_of") != cutoff:
        raise IntelligenceError("decision and context must share an exact cutoff")
    percentile = None
    if deterministic_percentile is not None:
        percentile = decimal_value(
            deterministic_percentile,
            label="deterministic_percentile",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
    research_threshold, paper_threshold, normalized_thresholds = _thresholds(thresholds)
    if isinstance(hard_veto_codes, (str, bytes)) or not isinstance(hard_veto_codes, Sequence):
        raise IntelligenceError("hard_veto_codes must be a sequence")
    vetoes = [
        identifier(value, label=f"hard_veto_codes[{index}]")
        for index, value in enumerate(hard_veto_codes)
    ]
    if len(vetoes) != len(set(vetoes)):
        raise IntelligenceError("hard_veto_codes must be nonempty and unique")
    state, blockers, reasons = _decision_state(
        context=context_payload,
        percentile=percentile,
        research_threshold=research_threshold,
        paper_threshold=paper_threshold,
        additional_vetoes=vetoes,
    )
    code = company_code(context_payload.get("company_code"))
    return build_artifact(
        kind="investment_decision",
        identity_field="decision_id",
        identity=decision_id
        or business_identity(
            kind="investment_decision",
            identity_inputs={
                "as_of": cutoff,
                "company_code": code,
                "context_id": context_artifact["artifact_id"],
            },
        ),
        created_at=cutoff,
        fields={
            "as_of": cutoff,
            "blocker_codes": blockers,
            "company_code": code,
            "context_ref": artifact_ref(context_artifact),
            "deterministic_percentile": None if percentile is None else decimal_text(percentile),
            "reason_codes": sorted(set(reasons), key=lambda item: item.encode("ascii")),
            "state": state,
            "thresholds": normalized_thresholds,
        },
    )


def validate_investment_decision(  # noqa: C901 - five-state replay gate
    artifact: Mapping[str, Any] | bytes,
    *,
    context: Mapping[str, Any] | bytes | None = None,
) -> dict[str, Any]:
    normalized, payload = artifact_payload(artifact, expected_kind="investment_decision")
    if payload.get("state") not in DECISION_STATES:
        raise IntelligenceError("investment decision state is invalid")
    if (
        payload.get("authority") != NO_AUTHORITY
        or payload.get("research_only") is not True
        or payload.get("production") is not False
        or payload.get("run_state") != "INACTIVE"
    ):
        raise IntelligenceError("investment decision authority is invalid")
    company_code(payload.get("company_code"))
    timestamp(payload.get("as_of"), label="investment_decision.as_of")
    context_ref = payload.get("context_ref")
    if type(context_ref) is not dict or context_ref.get("kind") != "decision_context":
        raise IntelligenceError("investment decision context ref is invalid")
    validate_artifact_ref(context_ref, label="context_ref")
    _, _, normalized_thresholds = _thresholds(payload.get("thresholds"))
    if payload.get("thresholds") != normalized_thresholds:
        raise IntelligenceError("investment decision thresholds are not canonical")
    percentile = payload.get("deterministic_percentile")
    if percentile is not None:
        parsed_percentile = decimal_value(
            percentile,
            label="investment_decision.deterministic_percentile",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
        if percentile != decimal_text(parsed_percentile):
            raise IntelligenceError("investment decision percentile is not canonical")
    for field in ("blocker_codes", "reason_codes"):
        values = payload.get(field)
        if type(values) is not list or values != sorted(
            {identifier(value, label=f"{field} value") for value in values},
            key=lambda item: item.encode("ascii"),
        ):
            raise IntelligenceError(f"investment decision {field} is invalid")
    if payload["state"] == "PAPER_CANDIDATE" and payload.get("blocker_codes"):
        raise IntelligenceError("paper candidate cannot retain blockers")
    if context is not None:
        validated_context = validate_decision_context(context)
        require_artifact_ref(payload.get("context_ref"), validated_context, label="context_ref")
        if validated_context["payload"].get("company_code") != payload.get("company_code"):
            raise IntelligenceError("investment decision company closure differs")
    return normalized


__all__ = [
    "DECISION_STATES",
    "IntelligenceError",
    "make_investment_decision",
    "validate_investment_decision",
]
