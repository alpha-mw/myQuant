"""Owner-sealed Theme governance and source-bound economic exposure states."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from ._common import (
    IntelligenceError,
    artifact_payload,
    artifact_ref,
    build_artifact,
    business_identity,
    company_code,
    require_artifact_ref,
    require_no_future,
    timestamp,
)

STRATEGY_ID: Final = "aggressive_tech_manufacturing"
GOVERNANCE_KIND: Final = "theme_governance_policy"
EXPOSURE_PROJECTION_KIND: Final = "theme_economic_exposure_projection"
OWNER_APPROVED_AT: Final = "2026-08-26T13:37:53Z"
EFFECTIVE_SIGNAL_DATE: Final = "20260827"

PRIMARY_THEME_IDS: Final = (
    "TUSHARE_DC:BK0480.DC",
    "TUSHARE_DC:BK0523.DC",
    "TUSHARE_DC:BK0800.DC",
    "TUSHARE_DC:BK0877.DC",
    "TUSHARE_DC:BK0891.DC",
    "TUSHARE_DC:BK0917.DC",
    "TUSHARE_DC:BK0922.DC",
    "TUSHARE_DC:BK0963.DC",
    "TUSHARE_DC:BK1090.DC",
    "TUSHARE_DC:BK1134.DC",
    "TUSHARE_DC:BK1184.DC",
)
FALLBACK_ALIAS_ROWS: Final = (
    {
        "fallback_theme_id": "TUSHARE_TDX:880548.TDX",
        "mapping_rule": "EXACT_REGISTRY_NAME",
        "primary_theme_id": "TUSHARE_DC:BK0963.DC",
    },
    {
        "fallback_theme_id": "TUSHARE_TDX:880550.TDX",
        "mapping_rule": "OWNER_APPROVED_ALIAS",
        "primary_theme_id": "TUSHARE_DC:BK0877.DC",
    },
    {
        "fallback_theme_id": "TUSHARE_TDX:880703.TDX",
        "mapping_rule": "EXACT_REGISTRY_NAME",
        "primary_theme_id": "TUSHARE_DC:BK1184.DC",
    },
    {
        "fallback_theme_id": "TUSHARE_TDX:880731.TDX",
        "mapping_rule": "EXACT_REGISTRY_NAME",
        "primary_theme_id": "TUSHARE_DC:BK0523.DC",
    },
    {
        "fallback_theme_id": "TUSHARE_TDX:880799.TDX",
        "mapping_rule": "EXACT_REGISTRY_NAME",
        "primary_theme_id": "TUSHARE_DC:BK0922.DC",
    },
    {
        "fallback_theme_id": "TUSHARE_TDX:880904.TDX",
        "mapping_rule": "EXACT_REGISTRY_NAME",
        "primary_theme_id": "TUSHARE_DC:BK1090.DC",
    },
    {
        "fallback_theme_id": "TUSHARE_TDX:880948.TDX",
        "mapping_rule": "EXACT_REGISTRY_NAME",
        "primary_theme_id": "TUSHARE_DC:BK0800.DC",
    },
    {
        "fallback_theme_id": "TUSHARE_TDX:880952.TDX",
        "mapping_rule": "OWNER_APPROVED_ALIAS",
        "primary_theme_id": "TUSHARE_DC:BK0891.DC",
    },
)
FALLBACK_THEME_IDS: Final = tuple(row["fallback_theme_id"] for row in FALLBACK_ALIAS_ROWS)
TECHNOLOGY_THEME_IDS: Final = tuple(
    sorted((*PRIMARY_THEME_IDS, *FALLBACK_THEME_IDS), key=lambda value: value.encode("ascii"))
)
EXPOSURE_LEVELS: Final = ("HIGH", "LOW", "MEDIUM", "UNVERIFIED")
EVIDENCE_SOURCE_PRECEDENCE: Final = (
    "ANNUAL_REPORT",
    "INTERIM_REPORT",
    "COMPANY_ANNOUNCEMENT",
    "INVESTOR_RELATIONS_RECORD",
    "PRODUCT_CUSTOMER_EVIDENCE",
    "REVENUE_STRUCTURE",
    "ORDER_CAPACITY",
    "CAPITAL_EXPENDITURE",
)


def approved_theme_governance_policy() -> dict[str, Any]:
    """Return the exact Theme policy approved by Maxwell on 2026-08-26."""

    domain_rows = [
        {
            "domain_id": "ADVANCED_MATERIALS",
            "fallback_theme_ids": ["TUSHARE_TDX:880731.TDX"],
            "primary_theme_ids": ["TUSHARE_DC:BK0523.DC"],
            "state": "ACTIVE",
        },
        {
            "domain_id": "AI_INFRASTRUCTURE",
            "fallback_theme_ids": [
                "TUSHARE_TDX:880799.TDX",
                "TUSHARE_TDX:880948.TDX",
            ],
            "primary_theme_ids": [
                "TUSHARE_DC:BK0800.DC",
                "TUSHARE_DC:BK0922.DC",
                "TUSHARE_DC:BK1134.DC",
            ],
            "state": "ACTIVE",
        },
        {
            "domain_id": "COMMERCIAL_SPACE",
            "fallback_theme_ids": ["TUSHARE_TDX:880548.TDX"],
            "primary_theme_ids": [
                "TUSHARE_DC:BK0480.DC",
                "TUSHARE_DC:BK0963.DC",
            ],
            "state": "ACTIVE",
        },
        {
            "domain_id": "ELECTRONICS_INTERCONNECT",
            "fallback_theme_ids": ["TUSHARE_TDX:880550.TDX"],
            "primary_theme_ids": ["TUSHARE_DC:BK0877.DC"],
            "state": "ACTIVE",
        },
        {
            "domain_id": "INTELLIGENT_VEHICLE",
            "fallback_theme_ids": [],
            "primary_theme_ids": [],
            "state": "EXPANSION_PENDING",
        },
        {
            "domain_id": "ROBOTICS_AUTOMATION",
            "fallback_theme_ids": [
                "TUSHARE_TDX:880703.TDX",
                "TUSHARE_TDX:880904.TDX",
            ],
            "primary_theme_ids": [
                "TUSHARE_DC:BK1090.DC",
                "TUSHARE_DC:BK1184.DC",
            ],
            "state": "ACTIVE",
        },
        {
            "domain_id": "SEMICONDUCTOR",
            "fallback_theme_ids": ["TUSHARE_TDX:880952.TDX"],
            "primary_theme_ids": [
                "TUSHARE_DC:BK0891.DC",
                "TUSHARE_DC:BK0917.DC",
            ],
            "state": "ACTIVE",
        },
    ]
    return build_artifact(
        kind=GOVERNANCE_KIND,
        identity_field="policy_id",
        identity=business_identity(
            kind=GOVERNANCE_KIND,
            identity_inputs={
                "effective_signal_date": EFFECTIVE_SIGNAL_DATE,
                "owner_approved_at": OWNER_APPROVED_AT,
                "strategy_id": STRATEGY_ID,
            },
        ),
        created_at=OWNER_APPROVED_AT,
        fields={
            "domain_rows": domain_rows,
            "effective_from": OWNER_APPROVED_AT,
            "effective_signal_date": EFFECTIVE_SIGNAL_DATE,
            "evidence_source_precedence": list(EVIDENCE_SOURCE_PRECEDENCE),
            "exposure_levels": list(EXPOSURE_LEVELS),
            "fallback_alias_rows": [dict(row) for row in FALLBACK_ALIAS_ROWS],
            "fallback_provider": "TUSHARE_TDX",
            "fallback_rule": "ONLY_REGISTERED_DC_FALLBACK_COMPANY_KEYSET",
            "membership_is_economic_exposure": False,
            "owner_approved_at": OWNER_APPROVED_AT,
            "primary_provider": "TUSHARE_DC",
            "primary_theme_ids": list(PRIMARY_THEME_IDS),
            "status": "ACTIVE",
            "strategy_id": STRATEGY_ID,
            "technology_theme_ids": list(TECHNOLOGY_THEME_IDS),
        },
    )


def validate_theme_governance_policy(value: Mapping[str, Any] | bytes) -> dict[str, Any]:
    normalized, _payload = artifact_payload(value, expected_kind=GOVERNANCE_KIND)
    if normalized != approved_theme_governance_policy():
        raise IntelligenceError("Theme governance policy does not replay approved bytes")
    return normalized


def build_unverified_economic_exposure_projection(
    *,
    as_of: str,
    daily_policy: Mapping[str, Any] | bytes,
    theme_projection: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    """Produce only honest UNVERIFIED exposure states until source evidence exists."""

    cutoff = timestamp(as_of, label="as_of")
    governance = approved_theme_governance_policy()
    governance_payload = governance["payload"]
    daily, daily_payload = artifact_payload(daily_policy, expected_kind="daily_research_policy")
    theme, theme_payload = artifact_payload(
        theme_projection, expected_kind="theme_membership_projection"
    )
    require_no_future(theme, as_of=cutoff, label="theme projection")
    require_artifact_ref(theme_payload["policy_ref"], daily, label="theme.policy_ref")
    if (
        daily_payload["strategy_id"] != STRATEGY_ID
        or daily_payload["technology_policy_state"] != "ACTIVE"
        or daily_payload["effective_signal_date"] != EFFECTIVE_SIGNAL_DATE
        or daily_payload["technology_theme_ids"] != governance_payload["technology_theme_ids"]
        or daily_payload["theme_provider_precedence"]
        != [governance_payload["primary_provider"], governance_payload["fallback_provider"]]
    ):
        raise IntelligenceError("daily Theme policy differs from owner governance")
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for row in theme_payload["company_rows"]:
        company = company_code(row["company_code"])
        matched = row["technology_theme_ids"]
        gate = (
            "UNAVAILABLE"
            if row["status"] == "UNMAPPED"
            else "PASS" if matched else "REJECT_NON_TECH"
        )
        reason_codes = (
            ["ECONOMIC_EXPOSURE_SOURCE_REQUIRED"]
            if gate == "PASS"
            else ["TECHNOLOGY_MEMBERSHIP_NOT_ADMITTED"]
        )
        if gate == "PASS":
            blockers.append(f"ECONOMIC_EXPOSURE_UNVERIFIED:{company}")
        rows.append(
            {
                "company_code": company,
                "economic_exposure_state": "UNVERIFIED",
                "evidence_refs": [],
                "membership_theme_ids": matched,
                "reason_codes": reason_codes,
                "technology_gate": gate,
            }
        )
    return build_artifact(
        kind=EXPOSURE_PROJECTION_KIND,
        identity_field="projection_id",
        identity=business_identity(
            kind=EXPOSURE_PROJECTION_KIND,
            identity_inputs={
                "company_set_sha256": theme_payload["company_set_sha256"],
                "policy_id": governance["artifact_id"],
                "trade_date": theme_payload["trade_date"],
            },
        ),
        created_at=cutoff,
        fields={
            "as_of": cutoff,
            "blocker_codes": sorted(blockers),
            "company_rows": rows,
            "company_set_sha256": theme_payload["company_set_sha256"],
            "daily_policy_ref": artifact_ref(daily),
            "governance_policy_ref": artifact_ref(governance),
            "status": "BLOCKED" if blockers else "READY",
            "strategy_id": STRATEGY_ID,
            "theme_projection_ref": artifact_ref(theme),
        },
    )


__all__ = [
    "EFFECTIVE_SIGNAL_DATE",
    "EXPOSURE_LEVELS",
    "EXPOSURE_PROJECTION_KIND",
    "FALLBACK_ALIAS_ROWS",
    "GOVERNANCE_KIND",
    "OWNER_APPROVED_AT",
    "PRIMARY_THEME_IDS",
    "STRATEGY_ID",
    "TECHNOLOGY_THEME_IDS",
    "approved_theme_governance_policy",
    "build_unverified_economic_exposure_projection",
    "validate_theme_governance_policy",
]
