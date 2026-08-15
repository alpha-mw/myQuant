"""Source-bound context assembly for deterministic investment decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from .._common import (
    IntelligenceError,
    NO_AUTHORITY,
    artifact_payload,
    artifact_ref,
    build_artifact,
    business_identity,
    company_code,
    identifier,
    require_no_future,
    timestamp,
    validate_artifact_ref,
)
from ..fundamental import validate_fundamental_assessment
from ..industry import validate_industry_assessment
from ..theme import validate_theme_assessment

CONTEXT_STATUSES: Final = frozenset({"AVAILABLE", "INSUFFICIENT_EVIDENCE", "BLOCKED"})
HYPOTHESIS_STATUSES: Final = frozenset({"VALID", "INVALIDATED", "UNTESTED"})
RISK_STATUSES: Final = frozenset({"AVAILABLE", "BLOCKED", "UNAVAILABLE"})


def _component(
    artifact: Mapping[str, Any] | bytes | None,
    *,
    kind: str,
    company: str,
    as_of: str,
) -> tuple[dict[str, str] | None, dict[str, Any] | None]:
    if artifact is None:
        return None, None
    validators = {
        "fundamental_assessment": validate_fundamental_assessment,
        "industry_assessment": validate_industry_assessment,
        "theme_assessment": validate_theme_assessment,
    }
    validated = validators[kind](artifact)
    require_no_future(validated, as_of=as_of, label=kind)
    payload = validated["payload"]
    if payload.get("company_code") != company:
        raise IntelligenceError(f"{kind} is bound to another company")
    return artifact_ref(validated), payload


def _evidence_refs(values: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise IntelligenceError("evidence_refs must be a sequence")
    rows = [
        validate_artifact_ref(value, label=f"evidence_refs[{index}]")
        for index, value in enumerate(values)
    ]
    keys = [(row["kind"], row["artifact_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise IntelligenceError("evidence closure is duplicated")
    return sorted(
        rows,
        key=lambda row: (row["kind"].encode("ascii"), row["artifact_id"].encode("ascii")),
    )


def build_decision_context(  # noqa: C901 - deterministic cross-domain gate
    *,
    company: str,
    as_of: str,
    hypothesis_status: str,
    risk_status: str,
    evidence_refs: Sequence[Mapping[str, Any]],
    industry_assessment: Mapping[str, Any] | bytes | None,
    theme_assessment: Mapping[str, Any] | bytes | None,
    fundamental_assessment: Mapping[str, Any] | bytes | None,
    quant_ref: Mapping[str, Any] | None,
    risk_codes: Sequence[str] = (),
    context_id: str | None = None,
) -> dict[str, Any]:
    """Assemble exact component evidence; never infer a missing layer."""

    code = company_code(company)
    cutoff = timestamp(as_of, label="as_of")
    if hypothesis_status not in HYPOTHESIS_STATUSES:
        raise IntelligenceError("hypothesis_status is invalid")
    if risk_status not in RISK_STATUSES:
        raise IntelligenceError("risk_status is invalid")
    industry_ref, industry = _component(
        industry_assessment,
        kind="industry_assessment",
        company=code,
        as_of=cutoff,
    )
    theme_ref, theme = _component(
        theme_assessment,
        kind="theme_assessment",
        company=code,
        as_of=cutoff,
    )
    fundamental_ref, fundamental = _component(
        fundamental_assessment,
        kind="fundamental_assessment",
        company=code,
        as_of=cutoff,
    )
    quant = None if quant_ref is None else validate_artifact_ref(quant_ref, label="quant_ref")
    evidence = _evidence_refs(evidence_refs)
    blockers: list[str] = []
    if quant is None:
        blockers.append("QUANT_EVIDENCE_MISSING")
    if industry is None or industry.get("status") != "AVAILABLE":
        blockers.append("INDUSTRY_IDENTITY_UNAVAILABLE")
    if theme is None or theme.get("status") != "AVAILABLE":
        blockers.append("THEME_EVIDENCE_UNAVAILABLE")
    if fundamental is None or fundamental.get("status") in {"MISSING", "BLOCKED"}:
        blockers.append("FUNDAMENTAL_EVIDENCE_UNAVAILABLE")
    if not evidence:
        blockers.append("SOURCE_EVIDENCE_MISSING")
    if risk_status != "AVAILABLE":
        blockers.append("RISK_EVIDENCE_UNAVAILABLE")
    if hypothesis_status == "UNTESTED":
        blockers.append("HYPOTHESIS_UNTESTED")
    if isinstance(risk_codes, (str, bytes)) or not isinstance(risk_codes, Sequence):
        raise IntelligenceError("risk_codes must be a sequence")
    normalized_risks = sorted(
        {identifier(value, label=f"risk_codes[{index}]") for index, value in enumerate(risk_codes)},
        key=lambda item: item.encode("ascii"),
    )
    if len(normalized_risks) != len(risk_codes):
        raise IntelligenceError("risk_codes must be nonempty unique strings")
    if risk_status == "BLOCKED":
        status = "BLOCKED"
    elif blockers:
        status = "INSUFFICIENT_EVIDENCE"
    else:
        status = "AVAILABLE"
    return build_artifact(
        kind="decision_context",
        identity_field="context_id",
        identity=context_id
        or business_identity(
            kind="decision_context",
            identity_inputs={"as_of": cutoff, "company_code": code},
        ),
        created_at=cutoff,
        fields={
            "as_of": cutoff,
            "blocker_codes": sorted(set(blockers), key=lambda item: item.encode("ascii")),
            "company_code": code,
            "component_refs": {
                "fundamental": fundamental_ref,
                "industry": industry_ref,
                "quant": quant,
                "theme": theme_ref,
            },
            "evidence_refs": evidence,
            "hard_risk_codes": normalized_risks,
            "hypothesis_status": hypothesis_status,
            "risk_status": risk_status,
            "status": status,
        },
    )


def validate_decision_context(  # noqa: C901 - exact cross-domain closure
    artifact: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    normalized, payload = artifact_payload(artifact, expected_kind="decision_context")
    if payload.get("status") not in CONTEXT_STATUSES:
        raise IntelligenceError("decision context status is invalid")
    if payload.get("hypothesis_status") not in HYPOTHESIS_STATUSES:
        raise IntelligenceError("decision context hypothesis status is invalid")
    if payload.get("risk_status") not in RISK_STATUSES:
        raise IntelligenceError("decision context risk status is invalid")
    if (
        payload.get("authority") != NO_AUTHORITY
        or payload.get("research_only") is not True
        or payload.get("production") is not False
        or payload.get("run_state") != "INACTIVE"
    ):
        raise IntelligenceError("decision context authority is invalid")
    company_code(payload.get("company_code"))
    timestamp(payload.get("as_of"), label="decision_context.as_of")
    components = payload.get("component_refs")
    if type(components) is not dict or set(components) != {
        "fundamental",
        "industry",
        "quant",
        "theme",
    }:
        raise IntelligenceError("decision context component closure is invalid")
    expected_kinds = {
        "fundamental": "fundamental_assessment",
        "industry": "industry_assessment",
        "theme": "theme_assessment",
    }
    for field, reference in components.items():
        if reference is None:
            continue
        normalized_ref = validate_artifact_ref(reference, label=f"component_refs.{field}")
        expected_kind = expected_kinds.get(field)
        if expected_kind is not None and normalized_ref["kind"] != expected_kind:
            raise IntelligenceError("decision context component kind is invalid")
    evidence = payload.get("evidence_refs")
    if type(evidence) is not list or _evidence_refs(evidence) != evidence:
        raise IntelligenceError("decision context evidence refs are not canonical")
    for field in ("blocker_codes", "hard_risk_codes"):
        values = payload.get(field)
        if type(values) is not list or values != sorted(
            {identifier(value, label=f"{field} value") for value in values},
            key=lambda item: item.encode("ascii"),
        ):
            raise IntelligenceError(f"decision context {field} is invalid")
    if payload["status"] == "AVAILABLE" and (
        any(reference is None for reference in components.values())
        or payload.get("blocker_codes") != []
    ):
        raise IntelligenceError("available decision context is incomplete")
    return normalized


__all__ = [
    "CONTEXT_STATUSES",
    "HYPOTHESIS_STATUSES",
    "IntelligenceError",
    "RISK_STATUSES",
    "build_decision_context",
    "validate_decision_context",
]
