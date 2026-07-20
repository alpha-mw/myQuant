"""Structurally nonauthorizing readiness for v16 source-status evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import re
from collections.abc import Mapping
from typing import Any

from .calibration_plan_v3 import (
    CALIBRATION_SOURCE_STATUS_SCHEMA,
    FORMAL_BRANCHES,
    PRIVATE_ROOT_POLICY,
    READINESS_V3_SCHEMA,
)
from .calibration_source_v3 import (
    CalibrationSourceStatusEvidenceBundleV3,
    validate_calibration_source_status_v3,
)
from .contracts import (
    BoundCanonicalArtifact,
    EvidenceV2Error,
    seal_semantic,
    validate_semantic_seal,
)
from .factor_carrier import (
    FactorProductionSetCarrierV4Error,
    FactorProductionSetEvidenceBundleV4,
)
from .schedule import AttemptGenesisEvidenceBundleV3
from .schedule_v4 import ScheduleAnchorBindingV4, validate_bound_lineage_v4

SCHEMA_VERSION = READINESS_V3_SCHEMA
ARCHITECTURE_VERSION = "16.0.0"
ARTIFACT_FILENAME = "v16_run_readiness_v3.json"
FORMAL_BRANCH_WEIGHT = "0.25"
SCHEDULE_LINEAGE_READBACK_SCHEMA = "v16.schedule-lineage-readback.v4"
FOUNDATION_BLOCKERS = (
    "calendar_recheck_capture_time_not_independently_evidenced",
    "calendar_recheck_transport_freshness_not_independently_attested",
    "codex_authority_chain_v2_not_integrated",
    "dashboard_activation_receipt_v2_not_integrated",
    "evidence_v2_disconnected_from_authorizing_consumers",
    "global_attempt_registry_authority_not_integrated",
    "production_pointer_switch_not_authorized",
    "provisional_journal_head_not_bound_to_external_anti_rollback_authority",
    "readiness_v3_source_status_schema_nonauthorizing",
)

_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class V16ReadinessV3Error(ValueError):
    """Raised when readiness-v3 source evidence fails closed."""


def _exact(value: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise V16ReadinessV3Error(f"{label} fields mismatch")
    return dict(value)


def _identifier(value: Any, *, label: str) -> str:
    text = str(value or "")
    if not _ID_PATTERN.fullmatch(text):
        raise V16ReadinessV3Error(f"{label} must be a safe identifier")
    return text


def _iso_date(value: Any, *, label: str) -> str:
    text = str(value or "")
    try:
        normalized = date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise V16ReadinessV3Error(f"{label} must be an ISO date") from exc
    if normalized != text:
        raise V16ReadinessV3Error(f"{label} must be a canonical ISO date")
    return text


def _utc(value: Any, *, label: str) -> str:
    text = str(value or "")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise V16ReadinessV3Error(f"{label} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise V16ReadinessV3Error(f"{label} must be UTC")
    normalized = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if normalized != text:
        raise V16ReadinessV3Error(f"{label} must be canonical UTC")
    return text


@dataclass(frozen=True)
class ScheduleLineageEvidenceBundleV4:
    genesis: AttemptGenesisEvidenceBundleV3
    schedule_anchors: tuple[ScheduleAnchorBindingV4, ...]

    def read(self) -> dict[str, Any]:
        if not isinstance(self.genesis, AttemptGenesisEvidenceBundleV3):
            raise V16ReadinessV3Error("schedule lineage v4 genesis has the wrong type")
        if (
            type(self.schedule_anchors) is not tuple
            or not self.schedule_anchors
            or any(
                not isinstance(item, ScheduleAnchorBindingV4)
                for item in self.schedule_anchors
            )
        ):
            raise V16ReadinessV3Error(
                "schedule lineage v4 requires a non-empty tuple of anchors"
            )
        try:
            projection = validate_bound_lineage_v4(
                genesis=self.genesis,
                schedule_anchors=self.schedule_anchors,
            )
        except EvidenceV2Error as exc:
            raise V16ReadinessV3Error(str(exc)) from exc
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
class ReadinessEvidenceBundleV3:
    factor_production_set: FactorProductionSetEvidenceBundleV4
    schedule_lineage: ScheduleLineageEvidenceBundleV4
    calibration_status: BoundCanonicalArtifact
    calibration_evidence: CalibrationSourceStatusEvidenceBundleV3

    def read(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        if not isinstance(
            self.factor_production_set,
            FactorProductionSetEvidenceBundleV4,
        ):
            raise V16ReadinessV3Error("Factor production-set bundle has the wrong type")
        if not isinstance(self.schedule_lineage, ScheduleLineageEvidenceBundleV4):
            raise V16ReadinessV3Error("schedule lineage v4 bundle has the wrong type")
        if not isinstance(
            self.calibration_evidence,
            CalibrationSourceStatusEvidenceBundleV3,
        ) or not isinstance(self.calibration_status, BoundCanonicalArtifact):
            raise V16ReadinessV3Error("calibration source bundle has the wrong type")
        if (
            self.calibration_status.reference.artifact_schema
            != CALIBRATION_SOURCE_STATUS_SCHEMA
            or self.calibration_status.reference.root_policy != PRIVATE_ROOT_POLICY
        ):
            raise V16ReadinessV3Error("calibration source status ref is invalid")
        try:
            factor = self.factor_production_set.read()
            schedule = self.schedule_lineage.read()
            calibration = validate_calibration_source_status_v3(
                self.calibration_status.read(),
                evidence=self.calibration_evidence,
            )
        except (EvidenceV2Error, FactorProductionSetCarrierV4Error) as exc:
            raise V16ReadinessV3Error(str(exc)) from exc
        if (
            calibration["protocol_attempt_id"] != schedule["protocol_attempt_id"]
            or calibration["schedule_ref"]
            not in schedule["schedule_declaration_refs"]
        ):
            raise V16ReadinessV3Error("calibration/schedule readiness lineage drift")
        return factor, schedule, calibration


def _blocker_projection(
    *,
    factor: Mapping[str, Any],
    schedule: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> tuple[list[str], list[dict[str, str]]]:
    rows = [
        {"blocker": blocker, "source": "readiness_v3_foundation"}
        for blocker in FOUNDATION_BLOCKERS
    ]
    for source, payload in (
        ("factor_production_set", factor),
        ("schedule_lineage", schedule),
    ):
        blockers = payload.get("blockers")
        if not isinstance(blockers, list):
            raise V16ReadinessV3Error(f"{source} blockers must be a list")
        rows.extend(
            {"blocker": str(blocker), "source": source} for blocker in blockers
        )
    calibration_sources = calibration.get("blocker_sources")
    if not isinstance(calibration_sources, list):
        raise V16ReadinessV3Error("calibration blocker sources must be a list")
    for item in calibration_sources:
        if not isinstance(item, Mapping) or set(item) != {"blocker", "source"}:
            raise V16ReadinessV3Error("calibration blocker source row is invalid")
        rows.append(
            {
                "blocker": str(item["blocker"]),
                "source": f"calibration_source:{item['source']}",
            }
        )
    if any(not item["blocker"] or not item["source"] for item in rows):
        raise V16ReadinessV3Error("readiness v3 blocker source row is empty")
    rows.sort(key=lambda item: (item["blocker"], item["source"]))
    return sorted({item["blocker"] for item in rows}), rows


def build_v16_run_readiness_v3(
    *,
    run_id: str,
    generated_at: str,
    analysis_trade_date: str,
    evidence: ReadinessEvidenceBundleV3,
) -> dict[str, Any]:
    if not isinstance(evidence, ReadinessEvidenceBundleV3):
        raise V16ReadinessV3Error("readiness v3 requires ReadinessEvidenceBundleV3")
    factor, schedule, calibration = evidence.read()
    blockers, blocker_sources = _blocker_projection(
        factor=factor,
        schedule=schedule,
        calibration=calibration,
    )
    return seal_semantic(
        {
            "schema_version": SCHEMA_VERSION,
            "architecture_version": ARCHITECTURE_VERSION,
            "artifact_filename": ARTIFACT_FILENAME,
            "run_id": _identifier(run_id, label="run_id"),
            "generated_at": _utc(generated_at, label="generated_at"),
            "analysis_trade_date": _iso_date(
                analysis_trade_date,
                label="analysis_trade_date",
            ),
            "formal_branches": [
                {"branch": branch, "weight": FORMAL_BRANCH_WEIGHT}
                for branch in FORMAL_BRANCHES
            ],
            "retrieval_role": "evidence_only_no_scoring_or_weight",
            "risk_advisor_role": "advisory_only",
            "factor_production_set": factor,
            "schedule_lineage": schedule,
            "calibration_source_status": calibration,
            "evidence_refs": {
                "factor_production_set_carrier": (
                    evidence.factor_production_set.carrier.reference.to_dict()
                ),
                "schedule_v4_declarations": schedule["schedule_declaration_refs"],
                "calibration_source_status": evidence.calibration_status.reference.to_dict(),
            },
            "activation_candidate": False,
            "new_risk_authorized": False,
            "readiness_status": "no_new_risk",
            "broker_side_effects": False,
            "blockers": blockers,
            "blocker_sources": blocker_sources,
        }
    )


def validate_v16_run_readiness_v3(
    value: Mapping[str, Any],
    *,
    evidence: ReadinessEvidenceBundleV3,
) -> dict[str, Any]:
    try:
        payload = validate_semantic_seal(value)
    except EvidenceV2Error as exc:
        raise V16ReadinessV3Error(str(exc)) from exc
    fields = {
        "schema_version",
        "architecture_version",
        "artifact_filename",
        "run_id",
        "generated_at",
        "analysis_trade_date",
        "formal_branches",
        "retrieval_role",
        "risk_advisor_role",
        "factor_production_set",
        "schedule_lineage",
        "calibration_source_status",
        "evidence_refs",
        "activation_candidate",
        "new_risk_authorized",
        "readiness_status",
        "broker_side_effects",
        "blockers",
        "blocker_sources",
        "semantic_sha256",
    }
    payload = _exact(payload, fields, label="v16 readiness v3")
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["architecture_version"] != ARCHITECTURE_VERSION
        or payload["artifact_filename"] != ARTIFACT_FILENAME
    ):
        raise V16ReadinessV3Error("readiness v3 identity mismatch")
    if any(
        payload[field] is not False
        for field in (
            "activation_candidate",
            "new_risk_authorized",
            "broker_side_effects",
        )
    ) or payload["readiness_status"] != "no_new_risk":
        raise V16ReadinessV3Error("readiness v3 must remain no_new_risk")
    rebuilt = build_v16_run_readiness_v3(
        run_id=str(payload["run_id"]),
        generated_at=str(payload["generated_at"]),
        analysis_trade_date=str(payload["analysis_trade_date"]),
        evidence=evidence,
    )
    if rebuilt != payload:
        raise V16ReadinessV3Error("readiness v3 drifts from reopened evidence")
    return payload


__all__ = [
    "ARCHITECTURE_VERSION",
    "ARTIFACT_FILENAME",
    "FOUNDATION_BLOCKERS",
    "FORMAL_BRANCH_WEIGHT",
    "ReadinessEvidenceBundleV3",
    "SCHEMA_VERSION",
    "SCHEDULE_LINEAGE_READBACK_SCHEMA",
    "ScheduleLineageEvidenceBundleV4",
    "V16ReadinessV3Error",
    "build_v16_run_readiness_v3",
    "validate_v16_run_readiness_v3",
]
