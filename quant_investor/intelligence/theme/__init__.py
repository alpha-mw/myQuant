"""Deterministic Theme exposure, component, and concentration-risk assessment."""

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

THEME_STATUSES: Final = frozenset(
    {"AVAILABLE", "AMBIGUOUS", "NO_MEMBERSHIP", "RETIRED", "UNMAPPED"}
)


def _normalize_memberships(
    memberships: Sequence[Mapping[str, Any]],
    *,
    as_of: str,
) -> list[dict[str, Any]]:
    if isinstance(memberships, (str, bytes)) or not isinstance(memberships, Sequence):
        raise IntelligenceError("memberships must be a sequence")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(memberships):
        if type(row) is not dict:
            raise IntelligenceError(f"memberships[{index}] must be an object")
        required = {
            "available_at",
            "exposure",
            "exposure_basis",
            "provider",
            "status",
            "theme_id",
        }
        allowed = required | {"parent_theme_id", "score"}
        if not required.issubset(row) or not set(row).issubset(allowed):
            raise IntelligenceError(f"memberships[{index}] shape is invalid")
        available_at = timestamp(row["available_at"], label=f"memberships[{index}].available_at")
        if available_at > as_of:
            raise IntelligenceError("theme membership contains future evidence")
        provider = identifier(row["provider"], label=f"memberships[{index}].provider")
        theme_id = identifier(row["theme_id"], label=f"memberships[{index}].theme_id")
        key = (provider, theme_id)
        if key in seen:
            raise IntelligenceError("theme membership is duplicated")
        seen.add(key)
        status = row["status"]
        if status not in {"ACTIVE", "RETIRED", "UNRESOLVED"}:
            raise IntelligenceError("theme membership status is invalid")
        parent = row.get("parent_theme_id")
        if parent is not None:
            parent = identifier(parent, label=f"memberships[{index}].parent_theme_id")
        score = row.get("score")
        score_text = None
        if score is not None:
            score_text = decimal_text(
                decimal_value(
                    score,
                    label=f"memberships[{index}].score",
                    minimum=Decimal("0"),
                    maximum=Decimal("1"),
                )
            )
        rows.append(
            {
                "available_at": available_at,
                "exposure": decimal_text(
                    decimal_value(
                        row["exposure"],
                        label=f"memberships[{index}].exposure",
                        minimum=Decimal("0"),
                        maximum=Decimal("1"),
                    )
                ),
                "exposure_basis": identifier(
                    row["exposure_basis"],
                    label=f"memberships[{index}].exposure_basis",
                ),
                "parent_theme_id": parent,
                "provider": provider,
                "score": score_text,
                "status": status,
                "theme_id": theme_id,
            }
        )
    return rows


def _select_provider(
    rows: Sequence[dict[str, Any]],
    provider_precedence: Sequence[str],
) -> tuple[list[dict[str, Any]], str | None]:
    providers = [
        identifier(value, label=f"provider_precedence[{index}]")
        for index, value in enumerate(provider_precedence)
    ]
    if not providers or len(providers) != len(set(providers)):
        raise IntelligenceError("provider_precedence must be ordered, nonempty and unique")
    for provider in providers:
        selected = [row for row in rows if row["provider"] == provider]
        if selected:
            return selected, provider
    return [], None


def _risk_rows(
    risks: Sequence[Mapping[str, Any]],
    *,
    selected_theme_ids: set[str],
) -> tuple[list[dict[str, Any]], str, list[str]]:
    if isinstance(risks, (str, bytes)) or not isinstance(risks, Sequence):
        raise IntelligenceError("risk_rows must be a sequence")
    rows: list[dict[str, Any]] = []
    vetoes: set[str] = set()
    maximum = Decimal("0")
    seen: set[str] = set()
    for index, row in enumerate(risks):
        if type(row) is not dict or set(row) != {"hard_veto_codes", "severity", "theme_id"}:
            raise IntelligenceError(f"risk_rows[{index}] shape is invalid")
        theme_id = identifier(row["theme_id"], label=f"risk_rows[{index}].theme_id")
        if theme_id not in selected_theme_ids or theme_id in seen:
            raise IntelligenceError("theme risk closure is not exact")
        seen.add(theme_id)
        severity = decimal_value(
            row["severity"],
            label=f"risk_rows[{index}].severity",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
        codes = row["hard_veto_codes"]
        if not isinstance(codes, Sequence) or isinstance(codes, (str, bytes)):
            raise IntelligenceError("theme hard veto codes must be a sequence")
        normalized_codes = sorted(
            {identifier(code, label="theme hard veto code") for code in codes},
            key=lambda item: item.encode("ascii"),
        )
        if len(normalized_codes) != len(codes):
            raise IntelligenceError("theme hard veto codes must be unique")
        vetoes.update(normalized_codes)
        maximum = max(maximum, severity)
        rows.append(
            {
                "hard_veto_codes": normalized_codes,
                "severity": decimal_text(severity),
                "theme_id": theme_id,
            }
        )
    rows.sort(key=lambda row: row["theme_id"].encode("ascii"))
    return rows, decimal_text(maximum), sorted(vetoes, key=lambda item: item.encode("ascii"))


def assess_theme(
    *,
    company: str,
    memberships: Sequence[Mapping[str, Any]],
    provider_precedence: Sequence[str],
    catalog_complete: bool,
    as_of: str,
    risk_rows: Sequence[Mapping[str, Any]] = (),
    assessment_id: str | None = None,
) -> dict[str, Any]:
    """Build one inactive Theme assessment without identity inference."""

    code = company_code(company)
    cutoff = timestamp(as_of, label="as_of")
    if type(catalog_complete) is not bool:
        raise IntelligenceError("catalog_complete must be boolean")
    all_rows = _normalize_memberships(memberships, as_of=cutoff)
    selected, provider = _select_provider(all_rows, provider_precedence)
    reasons: list[str] = []
    status = "AVAILABLE"
    active = [row for row in selected if row["status"] == "ACTIVE"]
    if not selected:
        status = "NO_MEMBERSHIP" if catalog_complete else "UNMAPPED"
        reasons = [
            "COMPLETE_CATALOG_NO_MEMBERSHIP" if catalog_complete else "THEME_CATALOG_INCOMPLETE"
        ]
    elif not active and all(row["status"] == "RETIRED" for row in selected):
        status = "RETIRED"
        reasons = ["THEME_MEMBERSHIP_RETIRED"]
    elif any(row["status"] == "UNRESOLVED" for row in selected):
        status = "UNMAPPED"
        reasons = ["THEME_MEMBERSHIP_UNRESOLVED"]
    else:
        active_ids = {row["theme_id"] for row in active}
        if any(row["parent_theme_id"] in active_ids for row in active):
            status = "AMBIGUOUS"
            reasons = ["PARENT_CHILD_MEMBERSHIP_CONFLICT"]
        total = sum((Decimal(row["exposure"]) for row in active), Decimal("0"))
        if total != Decimal("1"):
            status = "AMBIGUOUS"
            reasons = ["THEME_EXPOSURE_WEIGHT_INVALID"]
    exposure_rows: list[dict[str, Any]] = []
    component_score: str | None = None
    component_status = "MISSING"
    normalized_risks: list[dict[str, Any]] = []
    overall_severity = decimal_text(Decimal("0"))
    veto_codes: list[str] = []
    if status == "AVAILABLE":
        active.sort(key=lambda row: row["theme_id"].encode("ascii"))
        exposure_rows = [
            {
                "exposure": row["exposure"],
                "exposure_basis": row["exposure_basis"],
                "theme_id": row["theme_id"],
            }
            for row in active
        ]
        if all(row["score"] is not None for row in active):
            score = sum(
                (Decimal(row["score"]) * Decimal(row["exposure"]) for row in active),
                Decimal("0"),
            )
            component_score = decimal_text(score)
            component_status = "AVAILABLE"
        normalized_risks, overall_severity, veto_codes = _risk_rows(
            risk_rows,
            selected_theme_ids={row["theme_id"] for row in active},
        )
    elif risk_rows:
        raise IntelligenceError("theme risk cannot attach to unresolved identity")
    return build_artifact(
        kind="theme_assessment",
        identity_field="assessment_id",
        identity=assessment_id
        or business_identity(
            kind="theme_assessment",
            identity_inputs={"as_of": cutoff, "company_code": code},
        ),
        created_at=cutoff,
        fields={
            "as_of": cutoff,
            "company_code": code,
            "component_score": component_score,
            "component_status": component_status,
            "exposures": exposure_rows,
            "hard_veto_codes": veto_codes,
            "overall_severity": overall_severity,
            "provider": provider,
            "reason_codes": reasons,
            "risk_rows": normalized_risks,
            "status": status,
        },
    )


def validate_theme_assessment(artifact: Mapping[str, Any] | bytes) -> dict[str, Any]:
    normalized, payload = artifact_payload(artifact, expected_kind="theme_assessment")
    if payload.get("status") not in THEME_STATUSES:
        raise IntelligenceError("theme assessment status is invalid")
    if (
        payload.get("authority") != NO_AUTHORITY
        or payload.get("research_only") is not True
        or payload.get("production") is not False
        or payload.get("run_state") != "INACTIVE"
    ):
        raise IntelligenceError("theme assessment authority is invalid")
    company_code(payload.get("company_code"))
    timestamp(payload.get("as_of"), label="theme.as_of")
    exposures = payload.get("exposures")
    if type(exposures) is not list:
        raise IntelligenceError("theme exposures are invalid")
    if payload["status"] == "AVAILABLE":
        total = sum(
            (
                decimal_value(
                    row.get("exposure"),
                    label="theme.exposure",
                    minimum=Decimal("0"),
                    maximum=Decimal("1"),
                )
                for row in exposures
            ),
            Decimal("0"),
        )
        if not exposures or total != Decimal("1"):
            raise IntelligenceError("available theme assessment is incomplete")
    elif exposures or payload.get("component_score") is not None:
        raise IntelligenceError("unavailable theme identity cannot carry a score")
    return normalized


__all__ = [
    "THEME_STATUSES",
    "IntelligenceError",
    "assess_theme",
    "validate_theme_assessment",
]
