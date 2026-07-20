"""Structurally nonauthorizing v16 readiness-v2 migration foundation.

This schema cannot authorize new risk.  A later authorization-capable schema
requires a separate review and version.  Inputs are typed evidence bundles;
caller-supplied readiness booleans, paths, and standalone hashes are absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import re
from collections.abc import Mapping
from typing import Any

from .factor_carrier import (
    FactorProductionSetCarrierV4Error,
    FactorProductionSetEvidenceBundleV4,
)
from quant_investor.v16.evidence_v2.contracts import (
    EvidenceV2Error,
    seal_semantic,
    validate_semantic_seal,
)
from quant_investor.v16.evidence_v2.schedule import (
    AttemptGenesisEvidenceBundleV3,
    ScheduleAnchorBindingV3,
    validate_bound_lineage_v3,
)

SCHEMA_VERSION = "v16_run_readiness.v2"
ARCHITECTURE_VERSION = "16.0.0"
ARTIFACT_FILENAME = "v16_run_readiness_v2.json"
BRANCH_SCHEMA_VERSION = "v16.four-branch"
FORMAL_BRANCHES = ("quant", "fundamental", "macro", "llm")
FORMAL_BRANCH_WEIGHT = "0.25"
SCHEDULE_LINEAGE_READBACK_SCHEMA = "v16.schedule-lineage-readback.v3"
FOUNDATION_BLOCKERS = (
    "calendar_recheck_capture_time_not_independently_evidenced",
    "calendar_recheck_transport_freshness_not_independently_attested",
    "calibration_source_recomputation_not_integrated",
    "codex_authority_chain_v2_not_integrated",
    "dashboard_activation_receipt_v2_not_integrated",
    "evidence_v2_disconnected_from_authorizing_consumers",
    "global_attempt_registry_authority_not_integrated",
    "production_pointer_switch_not_authorized",
    "provisional_journal_head_not_bound_to_external_anti_rollback_authority",
    "readiness_v2_foundation_schema_nonauthorizing",
)

_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class V16ReadinessV2Error(ValueError):
    """Raised when readiness-v2 foundation evidence fails closed."""


def _exact(value: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise V16ReadinessV2Error(f"{label} fields mismatch")
    return dict(value)


def _id(value: Any, *, label: str) -> str:
    text = str(value or "")
    if not _ID_PATTERN.fullmatch(text):
        raise V16ReadinessV2Error(f"{label} must be a safe identifier")
    return text


def _iso_date(value: Any, *, label: str) -> str:
    text = str(value or "")
    try:
        normalized = date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise V16ReadinessV2Error(f"{label} must be ISO date") from exc
    if normalized != text:
        raise V16ReadinessV2Error(f"{label} must be canonical ISO date")
    return text


def _utc(value: Any, *, label: str) -> str:
    text = str(value or "")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise V16ReadinessV2Error(f"{label} must be ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise V16ReadinessV2Error(f"{label} must be UTC")
    normalized = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if normalized != text:
        raise V16ReadinessV2Error(f"{label} must be canonical UTC")
    return text


@dataclass(frozen=True)
class ScheduleLineageEvidenceBundleV3:
    genesis: AttemptGenesisEvidenceBundleV3
    schedule_anchors: tuple[ScheduleAnchorBindingV3, ...]

    def read(self) -> dict[str, Any]:
        if not isinstance(self.genesis, AttemptGenesisEvidenceBundleV3):
            raise V16ReadinessV2Error("schedule lineage genesis has the wrong type")
        if (
            type(self.schedule_anchors) is not tuple
            or not self.schedule_anchors
            or any(
                not isinstance(item, ScheduleAnchorBindingV3)
                for item in self.schedule_anchors
            )
        ):
            raise V16ReadinessV2Error(
                "schedule lineage requires a non-empty tuple of v3 anchors"
            )
        try:
            projection = validate_bound_lineage_v3(
                genesis=self.genesis,
                schedule_anchors=self.schedule_anchors,
            )
        except EvidenceV2Error as exc:
            raise V16ReadinessV2Error(str(exc)) from exc
        genesis = self.genesis.read()
        schedules = [item.evidence.read() for item in self.schedule_anchors]
        return seal_semantic(
            {
                "schema_version": SCHEDULE_LINEAGE_READBACK_SCHEMA,
                "protocol_attempt_id": genesis["protocol_attempt_id"],
                "epochs": [item["epoch"] for item in schedules],
                "schedule_declaration_refs": [
                    item.evidence.schedule.reference.to_dict()
                    for item in self.schedule_anchors
                ],
                "activation_candidate": False,
                "new_risk_authorized": False,
                "production_apply_enabled": False,
                "readiness_status": "no_new_risk",
                "blockers": projection["blockers"],
            }
        )


@dataclass(frozen=True)
class ReadinessEvidenceBundleV2:
    factor_production_set: FactorProductionSetEvidenceBundleV4
    schedule_lineage: ScheduleLineageEvidenceBundleV3 | None = None

    def read(self) -> tuple[dict[str, Any], dict[str, Any] | None]:
        if not isinstance(
            self.factor_production_set,
            FactorProductionSetEvidenceBundleV4,
        ):
            raise V16ReadinessV2Error("Factor production-set bundle has the wrong type")
        if self.schedule_lineage is not None and not isinstance(
            self.schedule_lineage,
            ScheduleLineageEvidenceBundleV3,
        ):
            raise V16ReadinessV2Error("schedule lineage bundle has the wrong type")
        try:
            factor = self.factor_production_set.read()
        except FactorProductionSetCarrierV4Error as exc:
            raise V16ReadinessV2Error(str(exc)) from exc
        schedule = None if self.schedule_lineage is None else self.schedule_lineage.read()
        return factor, schedule


def _blocker_projection(
    *,
    factor: Mapping[str, Any],
    schedule: Mapping[str, Any] | None,
) -> tuple[list[str], list[dict[str, str]]]:
    sources: dict[str, str] = {
        blocker: "readiness_v2_foundation" for blocker in FOUNDATION_BLOCKERS
    }
    if schedule is None:
        sources["schedule_v3_lineage_missing"] = "schedule_lineage"
    else:
        raw_schedule_blockers = schedule.get("blockers")
        if not isinstance(raw_schedule_blockers, list):
            raise V16ReadinessV2Error("schedule lineage blockers must be a list")
        for blocker in raw_schedule_blockers:
            text = str(blocker or "")
            if not text:
                raise V16ReadinessV2Error("schedule lineage blocker is empty")
            sources[text] = "schedule_lineage"
    raw_factor_blockers = factor.get("blockers")
    if not isinstance(raw_factor_blockers, list):
        raise V16ReadinessV2Error("Factor carrier blockers must be a list")
    for blocker in raw_factor_blockers:
        text = str(blocker or "")
        if not text:
            raise V16ReadinessV2Error("Factor carrier blocker is empty")
        sources[f"factor_v4:{text}"] = "factor_production_set"
    blockers = sorted(sources)
    return blockers, [
        {"blocker": blocker, "source": sources[blocker]} for blocker in blockers
    ]


def build_v16_run_readiness_v2(
    *,
    run_id: str,
    generated_at: str,
    analysis_trade_date: str,
    evidence: ReadinessEvidenceBundleV2,
) -> dict[str, Any]:
    if not isinstance(evidence, ReadinessEvidenceBundleV2):
        raise V16ReadinessV2Error("readiness v2 requires ReadinessEvidenceBundleV2")
    factor, schedule = evidence.read()
    blockers, blocker_sources = _blocker_projection(factor=factor, schedule=schedule)
    schedule_refs = (
        []
        if evidence.schedule_lineage is None
        else [
            item.evidence.schedule.reference.to_dict()
            for item in evidence.schedule_lineage.schedule_anchors
        ]
    )
    return seal_semantic(
        {
            "schema_version": SCHEMA_VERSION,
            "architecture_version": ARCHITECTURE_VERSION,
            "artifact_filename": ARTIFACT_FILENAME,
            "run_id": _id(run_id, label="run_id"),
            "generated_at": _utc(generated_at, label="generated_at"),
            "analysis_trade_date": _iso_date(
                analysis_trade_date,
                label="analysis_trade_date",
            ),
            "branch_schema_version": BRANCH_SCHEMA_VERSION,
            "formal_branches": [
                {"branch": branch, "weight": FORMAL_BRANCH_WEIGHT}
                for branch in FORMAL_BRANCHES
            ],
            "retrieval_role": "evidence_only_no_scoring_or_weight",
            "risk_advisor_role": "advisory_only",
            "evidence_refs": {
                "factor_production_set_carrier": (
                    evidence.factor_production_set.carrier.reference.to_dict()
                ),
                "schedule_v3_declarations": schedule_refs,
            },
            "factor_production_set": factor,
            "schedule_lineage": (
                {
                    "present": False,
                    "schema_version": None,
                    "protocol_attempt_id": None,
                    "epochs": [],
                    "blockers": ["schedule_v3_lineage_missing"],
                }
                if schedule is None
                else schedule
            ),
            "activation_candidate": False,
            "new_risk_authorized": False,
            "readiness_status": "no_new_risk",
            "broker_side_effects": False,
            "blockers": blockers,
            "blocker_sources": blocker_sources,
        }
    )


def validate_v16_run_readiness_v2(
    value: Mapping[str, Any],
    *,
    evidence: ReadinessEvidenceBundleV2,
) -> dict[str, Any]:
    try:
        payload = validate_semantic_seal(value)
    except EvidenceV2Error as exc:
        raise V16ReadinessV2Error(str(exc)) from exc
    payload = _exact(
        payload,
        {
            "schema_version",
            "architecture_version",
            "artifact_filename",
            "run_id",
            "generated_at",
            "analysis_trade_date",
            "branch_schema_version",
            "formal_branches",
            "retrieval_role",
            "risk_advisor_role",
            "evidence_refs",
            "factor_production_set",
            "schedule_lineage",
            "activation_candidate",
            "new_risk_authorized",
            "readiness_status",
            "broker_side_effects",
            "blockers",
            "blocker_sources",
            "semantic_sha256",
        },
        label="v16 readiness v2",
    )
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["architecture_version"] != ARCHITECTURE_VERSION
        or payload["artifact_filename"] != ARTIFACT_FILENAME
        or payload["branch_schema_version"] != BRANCH_SCHEMA_VERSION
    ):
        raise V16ReadinessV2Error("readiness v2 identity mismatch")
    if any(
        payload[field] is not False
        for field in (
            "activation_candidate",
            "new_risk_authorized",
            "broker_side_effects",
        )
    ) or payload["readiness_status"] != "no_new_risk":
        raise V16ReadinessV2Error("readiness v2 foundation must remain no_new_risk")
    rebuilt = build_v16_run_readiness_v2(
        run_id=payload["run_id"],
        generated_at=payload["generated_at"],
        analysis_trade_date=payload["analysis_trade_date"],
        evidence=evidence,
    )
    if rebuilt != payload:
        raise V16ReadinessV2Error("readiness v2 drifts from reopened evidence bundle")
    return payload


__all__ = [
    "ARCHITECTURE_VERSION",
    "ARTIFACT_FILENAME",
    "BRANCH_SCHEMA_VERSION",
    "FORMAL_BRANCHES",
    "FOUNDATION_BLOCKERS",
    "ReadinessEvidenceBundleV2",
    "SCHEDULE_LINEAGE_READBACK_SCHEMA",
    "SCHEMA_VERSION",
    "ScheduleLineageEvidenceBundleV3",
    "V16ReadinessV2Error",
    "build_v16_run_readiness_v2",
    "validate_v16_run_readiness_v2",
]
