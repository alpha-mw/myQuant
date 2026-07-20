"""Evidence-v2 Dashboard projection and nonauthorizing source status."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from .candidate_report_source_v2 import (
    CandidateReportSourceEvidenceBundleV2,
    validate_candidate_report_source_v2,
)
from .codex_authority_plan_v2 import PRIVATE_ROOT_POLICY
from .contracts import (
    BoundCanonicalArtifact,
    EvidenceV2Error,
    seal_semantic,
    validate_semantic_seal,
)
from .publication_plan_v2 import (
    CANDIDATE_REPORT_SCHEMA,
    DASHBOARD_SNAPSHOT_SCHEMA,
    DASHBOARD_SOURCE_STATUS_SCHEMA,
    PublicationPlanEvidenceBundleV2,
)

DASHBOARD_SOURCE_BLOCKERS = (
    "dashboard_contract_v16_evidence_v2_not_integrated_with_app",
    "codex_requirement_unsupported:requirement="
    "dashboard_snapshot_delivery_attestation_protocol",
    "codex_requirement_unsupported:requirement="
    "dashboard_activation_receipt_v2_protocol",
)


def _exact(value: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise EvidenceV2Error(f"{label} fields mismatch")
    return dict(value)


def _matches_planned_reference(
    artifact: BoundCanonicalArtifact,
    planned: Mapping[str, Any],
) -> bool:
    return (
        artifact.reference.absolute_path == planned["absolute_path"]
        and artifact.reference.artifact_schema == planned["artifact_schema"]
        and artifact.reference.root_policy == planned["root_policy"]
    )


def _monotonic_blockers(
    source: Mapping[str, Any],
    *,
    additions: tuple[str, ...] = (),
    source_label: str,
) -> tuple[list[str], list[dict[str, str]]]:
    source_rows = source.get("blocker_sources")
    if not isinstance(source_rows, list):
        raise EvidenceV2Error(f"{source_label} blocker sources must be a list")
    rows: list[dict[str, str]] = []
    for item in source_rows:
        if not isinstance(item, Mapping) or set(item) != {"blocker", "source"}:
            raise EvidenceV2Error(f"{source_label} blocker source row is invalid")
        rows.append(
            {
                "blocker": str(item["blocker"]),
                "source": f"{source_label}:{item['source']}",
            }
        )
    rows.extend(
        {"blocker": blocker, "source": "dashboard_source_v2"}
        for blocker in additions
    )
    if any(not item["blocker"] or not item["source"] for item in rows):
        raise EvidenceV2Error("Dashboard blocker source row is empty")
    rows.sort(key=lambda item: (item["blocker"], item["source"]))
    blockers = sorted({item["blocker"] for item in rows})
    inherited = {str(item) for item in source.get("blockers", [])}
    if not inherited.issubset(blockers):
        raise EvidenceV2Error("Dashboard blockers are not monotonic")
    return blockers, rows


@dataclass(frozen=True)
class DashboardReportEvidenceBundleV2:
    publication_plan: PublicationPlanEvidenceBundleV2
    candidate_report: BoundCanonicalArtifact
    report_evidence: CandidateReportSourceEvidenceBundleV2

    def read(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        if (
            not isinstance(self.publication_plan, PublicationPlanEvidenceBundleV2)
            or not isinstance(self.candidate_report, BoundCanonicalArtifact)
            or self.candidate_report.reference.artifact_schema
            != CANDIDATE_REPORT_SCHEMA
            or self.candidate_report.reference.root_policy != PRIVATE_ROOT_POLICY
            or not isinstance(
                self.report_evidence,
                CandidateReportSourceEvidenceBundleV2,
            )
            or self.report_evidence.publication_plan.plan.reference
            != self.publication_plan.plan.reference
        ):
            raise EvidenceV2Error("Dashboard report evidence bundle is invalid")
        plan, readiness = self.publication_plan.read()
        if not _matches_planned_reference(
            self.candidate_report,
            plan["planned_artifacts"]["candidate_report"],
        ):
            raise EvidenceV2Error("Dashboard candidate report path drifts from plan")
        report = validate_candidate_report_source_v2(
            self.candidate_report.read(),
            evidence=self.report_evidence,
        )
        if (
            report["protocol_attempt_id"] != plan["protocol_attempt_id"]
            or report["run_id"] != plan["run_id"]
            or report["readiness_v4_ref"]
            != self.publication_plan.readiness_v4.reference.to_dict()
        ):
            raise EvidenceV2Error("Dashboard report lineage drift")
        return plan, readiness, report


def build_dashboard_snapshot_v2(
    *,
    evidence: DashboardReportEvidenceBundleV2,
) -> dict[str, Any]:
    if not isinstance(evidence, DashboardReportEvidenceBundleV2):
        raise EvidenceV2Error("Dashboard snapshot requires its typed evidence bundle")
    plan, readiness, report = evidence.read()
    blockers, blocker_sources = _monotonic_blockers(
        report,
        source_label="candidate_report",
    )
    return seal_semantic(
        {
            "schema_version": DASHBOARD_SNAPSHOT_SCHEMA,
            "architecture_version": report["architecture_version"],
            "protocol_attempt_id": plan["protocol_attempt_id"],
            "run_id": plan["run_id"],
            "generated_at": report["generated_at"],
            "analysis_trade_date": report["analysis_trade_date"],
            "source_refs": {
                "publication_plan": evidence.publication_plan.plan.reference.to_dict(),
                "readiness_v4": evidence.publication_plan.readiness_v4.reference.to_dict(),
                "candidate_report": evidence.candidate_report.reference.to_dict(),
            },
            "formal_branches": report["formal_branches"],
            "retrieval_role": report["retrieval_role"],
            "risk_advisor_role": report["risk_advisor_role"],
            "menu": report["menu"],
            "cash_ratio": report["cash_ratio"],
            "positive_weight_count": report["positive_weight_count"],
            "target_plus_cash": report["target_plus_cash"],
            "readiness": {
                "status": "no_new_risk",
                "activation_candidate": False,
                "new_risk_authorized": False,
                "production_apply_enabled": False,
                "production_pointer_switch_authorized": False,
                "codex_activation_authorized": False,
                "dashboard_activation_authorized": False,
                "sealed_live_human_receipt_verified": False,
                "broker_side_effects": False,
                "source_readiness_status": readiness["readiness_status"],
            },
            "projection_validation_complete": True,
            "authority_source_complete": False,
            "blockers": blockers,
            "blocker_sources": blocker_sources,
        }
    )


def validate_dashboard_snapshot_v2(
    value: Mapping[str, Any],
    *,
    evidence: DashboardReportEvidenceBundleV2,
) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    payload = _exact(
        payload,
        {
            "schema_version",
            "architecture_version",
            "protocol_attempt_id",
            "run_id",
            "generated_at",
            "analysis_trade_date",
            "source_refs",
            "formal_branches",
            "retrieval_role",
            "risk_advisor_role",
            "menu",
            "cash_ratio",
            "positive_weight_count",
            "target_plus_cash",
            "readiness",
            "projection_validation_complete",
            "authority_source_complete",
            "blockers",
            "blocker_sources",
            "semantic_sha256",
        },
        label="Dashboard evidence-v2 snapshot",
    )
    if payload["schema_version"] != DASHBOARD_SNAPSHOT_SCHEMA:
        raise EvidenceV2Error("Dashboard evidence-v2 snapshot schema mismatch")
    rebuilt = build_dashboard_snapshot_v2(evidence=evidence)
    if rebuilt != payload:
        raise EvidenceV2Error("Dashboard evidence-v2 snapshot drifts from report")
    return payload


@dataclass(frozen=True)
class DashboardSnapshotEvidenceBundleV2:
    report: DashboardReportEvidenceBundleV2
    snapshot: BoundCanonicalArtifact

    def read(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        if (
            not isinstance(self.report, DashboardReportEvidenceBundleV2)
            or not isinstance(self.snapshot, BoundCanonicalArtifact)
            or self.snapshot.reference.artifact_schema != DASHBOARD_SNAPSHOT_SCHEMA
            or self.snapshot.reference.root_policy != PRIVATE_ROOT_POLICY
        ):
            raise EvidenceV2Error("Dashboard snapshot evidence bundle is invalid")
        plan, readiness, report = self.report.read()
        if not _matches_planned_reference(
            self.snapshot,
            plan["planned_artifacts"]["dashboard_snapshot"],
        ):
            raise EvidenceV2Error("Dashboard snapshot path drifts from plan")
        snapshot = validate_dashboard_snapshot_v2(
            self.snapshot.read(),
            evidence=self.report,
        )
        return plan, readiness, report, snapshot


def build_dashboard_source_status_v2(
    *,
    evidence: DashboardSnapshotEvidenceBundleV2,
) -> dict[str, Any]:
    if not isinstance(evidence, DashboardSnapshotEvidenceBundleV2):
        raise EvidenceV2Error("Dashboard status requires its typed evidence bundle")
    plan, _readiness, report, snapshot = evidence.read()
    blockers, blocker_sources = _monotonic_blockers(
        snapshot,
        additions=DASHBOARD_SOURCE_BLOCKERS,
        source_label="dashboard_snapshot",
    )
    return seal_semantic(
        {
            "schema_version": DASHBOARD_SOURCE_STATUS_SCHEMA,
            "protocol_attempt_id": plan["protocol_attempt_id"],
            "run_id": plan["run_id"],
            "publication_plan_ref": evidence.report.publication_plan.plan.reference.to_dict(),
            "readiness_v4_ref": evidence.report.publication_plan.readiness_v4.reference.to_dict(),
            "candidate_report_ref": evidence.report.candidate_report.reference.to_dict(),
            "dashboard_snapshot_ref": evidence.snapshot.reference.to_dict(),
            "artifact_role": "source_status_only",
            "projection_validation_complete": True,
            "publication_delivery_attested": False,
            "dashboard_app_integrated": False,
            "dashboard_activation_receipt_verified": False,
            "authority_source_complete": False,
            "readiness_status": "no_new_risk",
            "report_readiness_status": report["readiness_status"],
            "blockers": blockers,
            "blocker_sources": blocker_sources,
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
            "production_pointer_switch_authorized": False,
            "dashboard_activation_authorized": False,
            "broker_side_effects": False,
        }
    )


def validate_dashboard_source_status_v2(
    value: Mapping[str, Any],
    *,
    evidence: DashboardSnapshotEvidenceBundleV2,
) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    payload = _exact(
        payload,
        {
            "schema_version",
            "protocol_attempt_id",
            "run_id",
            "publication_plan_ref",
            "readiness_v4_ref",
            "candidate_report_ref",
            "dashboard_snapshot_ref",
            "artifact_role",
            "projection_validation_complete",
            "publication_delivery_attested",
            "dashboard_app_integrated",
            "dashboard_activation_receipt_verified",
            "authority_source_complete",
            "readiness_status",
            "report_readiness_status",
            "blockers",
            "blocker_sources",
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
            "production_pointer_switch_authorized",
            "dashboard_activation_authorized",
            "broker_side_effects",
            "semantic_sha256",
        },
        label="Dashboard source status v2",
    )
    if payload["schema_version"] != DASHBOARD_SOURCE_STATUS_SCHEMA:
        raise EvidenceV2Error("Dashboard source status v2 schema mismatch")
    rebuilt = build_dashboard_source_status_v2(evidence=evidence)
    if rebuilt != payload:
        raise EvidenceV2Error("Dashboard source status v2 drifts from evidence")
    return payload


__all__ = [
    "DASHBOARD_SOURCE_BLOCKERS",
    "DashboardReportEvidenceBundleV2",
    "DashboardSnapshotEvidenceBundleV2",
    "build_dashboard_snapshot_v2",
    "build_dashboard_source_status_v2",
    "validate_dashboard_snapshot_v2",
    "validate_dashboard_source_status_v2",
]
