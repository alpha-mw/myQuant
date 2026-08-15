"""Deterministic Industry identity and component assessment.

Identity is source-bound.  Missing or conflicting membership remains explicit
and can never be repaired by a model, a neutral score, or inferred taxonomy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, Final

from .._common import (
    IntelligenceError,
    NO_AUTHORITY,
    artifact_payload,
    build_artifact,
    business_identity,
    company_code,
    decimal_text,
    decimal_value,
    identifier,
    timestamp,
)

INDUSTRY_STATUSES: Final = frozenset({"AVAILABLE", "AMBIGUOUS", "UNMAPPED", "RETIRED"})


def _membership_rows(  # noqa: C901 - precedence and PIT identity gate
    memberships: Sequence[Mapping[str, Any]],
    *,
    providers: Sequence[str],
    as_of: str,
) -> tuple[list[dict[str, Any]], str | None, str | None, list[str]]:
    if isinstance(memberships, (str, bytes)) or not isinstance(memberships, Sequence):
        raise IntelligenceError("memberships must be a sequence")
    provider_order = [
        identifier(value, label=f"provider_precedence[{index}]")
        for index, value in enumerate(providers)
    ]
    if not provider_order or len(provider_order) != len(set(provider_order)):
        raise IntelligenceError("provider_precedence must be ordered, nonempty and unique")
    cutoff = timestamp(as_of, label="as_of")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(memberships):
        if type(row) is not dict:
            raise IntelligenceError(f"memberships[{index}] must be an object")
        allowed = {
            "available_at",
            "effective_from",
            "effective_to",
            "exposure",
            "industry_id",
            "provider",
            "retired",
        }
        if not set(row).issubset(allowed) or not {
            "available_at",
            "effective_from",
            "exposure",
            "industry_id",
            "provider",
            "retired",
        }.issubset(row):
            raise IntelligenceError(f"memberships[{index}] shape is invalid")
        provider = identifier(row["provider"], label=f"memberships[{index}].provider")
        industry_id = identifier(row["industry_id"], label=f"memberships[{index}].industry_id")
        available_at = timestamp(row["available_at"], label=f"memberships[{index}].available_at")
        effective_from = timestamp(
            row["effective_from"], label=f"memberships[{index}].effective_from"
        )
        effective_to = row.get("effective_to")
        if effective_to is not None:
            effective_to = timestamp(effective_to, label=f"memberships[{index}].effective_to")
            if effective_to < effective_from:
                raise IntelligenceError("industry membership chronology is invalid")
        if available_at > cutoff or effective_from > cutoff:
            raise IntelligenceError("industry membership contains future evidence")
        retired = row["retired"]
        if type(retired) is not bool:
            raise IntelligenceError("industry retired flag must be boolean")
        key = (provider, industry_id)
        if key in seen:
            raise IntelligenceError("industry membership closure is duplicated")
        seen.add(key)
        normalized.append(
            {
                "available_at": available_at,
                "effective_from": effective_from,
                "effective_to": effective_to,
                "exposure": decimal_text(
                    decimal_value(
                        row["exposure"],
                        label=f"memberships[{index}].exposure",
                        minimum=Decimal("0"),
                        maximum=Decimal("1"),
                    )
                ),
                "industry_id": industry_id,
                "provider": provider,
                "retired": retired,
            }
        )
    active = [
        row
        for row in normalized
        if not row["retired"]
        and row["effective_from"] <= cutoff
        and (row["effective_to"] is None or row["effective_to"] >= cutoff)
    ]
    retired_only = bool(normalized) and not active and all(row["retired"] for row in normalized)
    for provider in provider_order:
        selected = [row for row in active if row["provider"] == provider]
        if not selected:
            continue
        total = sum((Decimal(row["exposure"]) for row in selected), Decimal("0"))
        if total != Decimal("1"):
            raise IntelligenceError("industry exposures must sum exactly to one")
        selected.sort(key=lambda row: row["industry_id"].encode("ascii"))
        maximum = max(Decimal(row["exposure"]) for row in selected)
        primary = [row["industry_id"] for row in selected if Decimal(row["exposure"]) == maximum]
        if len(primary) != 1:
            return [], provider, None, ["SAME_PRECEDENCE_CLASSIFICATION_CONFLICT"]
        return selected, provider, primary[0], []
    if retired_only:
        return [], None, None, ["INDUSTRY_MEMBERSHIP_RETIRED"]
    return [], None, None, ["NO_ADMISSIBLE_MEMBERSHIP"]


def _component(
    metric_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str | None, str]:
    if not metric_rows:
        return [], None, "MISSING"
    if isinstance(metric_rows, (str, bytes)) or not isinstance(metric_rows, Sequence):
        raise IntelligenceError("metric_rows must be a sequence")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    total_weight = Decimal("0")
    weighted = Decimal("0")
    missing = False
    for index, row in enumerate(metric_rows):
        if type(row) is not dict or set(row) != {"metric_id", "status", "value", "weight"}:
            raise IntelligenceError(f"metric_rows[{index}] shape is invalid")
        metric_id = identifier(row["metric_id"], label=f"metric_rows[{index}].metric_id")
        if metric_id in seen:
            raise IntelligenceError("industry metric is duplicated")
        seen.add(metric_id)
        status = row["status"]
        if status not in {"AVAILABLE", "MISSING"}:
            raise IntelligenceError("industry metric status is invalid")
        weight = decimal_value(
            row["weight"],
            label=f"metric_rows[{index}].weight",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
        total_weight += weight
        value_text: str | None = None
        if status == "AVAILABLE":
            value = decimal_value(row["value"], label=f"metric_rows[{index}].value")
            value_text = decimal_text(value)
            weighted += value * weight
        elif row["value"] is not None:
            raise IntelligenceError("missing industry metric cannot carry a value")
        else:
            missing = True
        rows.append(
            {
                "metric_id": metric_id,
                "status": status,
                "value": value_text,
                "weight": decimal_text(weight),
            }
        )
    if total_weight != Decimal("1"):
        raise IntelligenceError("industry metric weights must sum exactly to one")
    rows.sort(key=lambda row: row["metric_id"].encode("ascii"))
    return rows, None if missing else decimal_text(weighted), "MISSING" if missing else "AVAILABLE"


def assess_industry(
    *,
    company: str,
    memberships: Sequence[Mapping[str, Any]],
    provider_precedence: Sequence[str],
    as_of: str,
    metric_rows: Sequence[Mapping[str, Any]] = (),
    assessment_id: str | None = None,
) -> dict[str, Any]:
    """Build an inactive, source-bound Industry assessment."""

    code = company_code(company)
    cutoff = timestamp(as_of, label="as_of")
    exposure_rows, provider, primary, reasons = _membership_rows(
        memberships,
        providers=provider_precedence,
        as_of=cutoff,
    )
    metrics, component_score, component_status = _component(metric_rows)
    if reasons == ["SAME_PRECEDENCE_CLASSIFICATION_CONFLICT"]:
        status = "AMBIGUOUS"
    elif reasons == ["INDUSTRY_MEMBERSHIP_RETIRED"]:
        status = "RETIRED"
    elif reasons:
        status = "UNMAPPED"
    else:
        status = "AVAILABLE"
    if status != "AVAILABLE":
        component_score = None
        component_status = "MISSING"
    return build_artifact(
        kind="industry_assessment",
        identity_field="assessment_id",
        identity=assessment_id
        or business_identity(
            kind="industry_assessment",
            identity_inputs={"as_of": cutoff, "company_code": code},
        ),
        created_at=cutoff,
        fields={
            "as_of": cutoff,
            "company_code": code,
            "component_score": component_score,
            "component_status": component_status,
            "exposures": [
                {"industry_id": row["industry_id"], "exposure": row["exposure"]}
                for row in exposure_rows
            ],
            "metric_rows": metrics,
            "primary_industry_id": primary,
            "provider": provider,
            "reason_codes": reasons,
            "status": status,
        },
    )


def validate_industry_assessment(artifact: Mapping[str, Any] | bytes) -> dict[str, Any]:
    normalized, payload = artifact_payload(artifact, expected_kind="industry_assessment")
    if payload.get("status") not in INDUSTRY_STATUSES:
        raise IntelligenceError("industry assessment status is invalid")
    if payload.get("research_only") is not True or payload.get("production") is not False:
        raise IntelligenceError("industry assessment authority is invalid")
    if payload.get("authority") != NO_AUTHORITY or payload.get("run_state") != "INACTIVE":
        raise IntelligenceError("industry assessment is not inactive")
    company_code(payload.get("company_code"))
    timestamp(payload.get("as_of"), label="industry.as_of")
    exposures = payload.get("exposures")
    if type(exposures) is not list:
        raise IntelligenceError("industry exposures are invalid")
    if payload["status"] == "AVAILABLE":
        total = sum(
            (
                decimal_value(
                    row.get("exposure"),
                    label="industry.exposure",
                    minimum=Decimal("0"),
                    maximum=Decimal("1"),
                )
                for row in exposures
            ),
            Decimal("0"),
        )
        if not exposures or total != Decimal("1") or payload.get("primary_industry_id") is None:
            raise IntelligenceError("available industry assessment is incomplete")
    elif exposures or payload.get("primary_industry_id") is not None:
        raise IntelligenceError("unavailable industry identity cannot carry exposures")
    return normalized


__all__ = [
    "INDUSTRY_STATUSES",
    "IntelligenceError",
    "assess_industry",
    "validate_industry_assessment",
]
