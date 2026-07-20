"""Terminal, nonauthorizing completeness receipt for publication artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from .candidate_report_source_v2 import validate_candidate_report_source_v2
from .codex_authority_plan_v2 import PRIVATE_ROOT_POLICY
from .contracts import (
    BoundCanonicalArtifact,
    EvidenceV2Error,
    seal_semantic,
    validate_semantic_seal,
)
from .dashboard_source_v2 import (
    DashboardSnapshotEvidenceBundleV2,
    validate_dashboard_snapshot_v2,
    validate_dashboard_source_status_v2,
)
from .publication_plan_v2 import (
    CANDIDATE_REPORT_SCHEMA,
    DASHBOARD_SNAPSHOT_SCHEMA,
    DASHBOARD_SOURCE_STATUS_SCHEMA,
    PUBLICATION_AGGREGATE_SCHEMA,
    PublicationPlanEvidenceBundleV2,
)

AGGREGATE_ARTIFACT_ORDER = (
    "publication_plan",
    "readiness_v4",
    "candidate_report",
    "dashboard_snapshot",
    "dashboard_source_status",
)
AGGREGATE_BLOCKERS = (
    "publication_aggregate_not_production_authority",
    "publication_bundle_delivery_not_externally_attested",
)


def _exact(value: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise EvidenceV2Error(f"{label} fields mismatch")
    return dict(value)


def _matches_planned(
    artifact: BoundCanonicalArtifact,
    planned: Mapping[str, Any],
) -> bool:
    return (
        artifact.reference.absolute_path == planned["absolute_path"]
        and artifact.reference.artifact_schema == planned["artifact_schema"]
        and artifact.reference.root_policy == planned["root_policy"]
    )


@dataclass(frozen=True)
class PublicationAggregateEvidenceBundleV2:
    publication_plan: PublicationPlanEvidenceBundleV2
    candidate_report: BoundCanonicalArtifact
    dashboard_snapshot: BoundCanonicalArtifact
    dashboard_source_status: BoundCanonicalArtifact
    dashboard_evidence: DashboardSnapshotEvidenceBundleV2

    def read(
        self,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        if (
            not isinstance(self.publication_plan, PublicationPlanEvidenceBundleV2)
            or not isinstance(self.candidate_report, BoundCanonicalArtifact)
            or self.candidate_report.reference.artifact_schema
            != CANDIDATE_REPORT_SCHEMA
            or self.candidate_report.reference.root_policy != PRIVATE_ROOT_POLICY
            or not isinstance(self.dashboard_snapshot, BoundCanonicalArtifact)
            or self.dashboard_snapshot.reference.artifact_schema
            != DASHBOARD_SNAPSHOT_SCHEMA
            or self.dashboard_snapshot.reference.root_policy != PRIVATE_ROOT_POLICY
            or not isinstance(self.dashboard_source_status, BoundCanonicalArtifact)
            or self.dashboard_source_status.reference.artifact_schema
            != DASHBOARD_SOURCE_STATUS_SCHEMA
            or self.dashboard_source_status.reference.root_policy
            != PRIVATE_ROOT_POLICY
            or not isinstance(
                self.dashboard_evidence,
                DashboardSnapshotEvidenceBundleV2,
            )
        ):
            raise EvidenceV2Error("publication aggregate evidence bundle is invalid")
        plan, readiness = self.publication_plan.read()
        planned = plan["planned_artifacts"]
        for key, artifact in (
            ("candidate_report", self.candidate_report),
            ("dashboard_snapshot", self.dashboard_snapshot),
            ("dashboard_source_status", self.dashboard_source_status),
        ):
            if not _matches_planned(artifact, planned[key]):
                raise EvidenceV2Error(
                    f"publication aggregate artifact drifts from plan: {key}"
                )
        report_evidence = self.dashboard_evidence.report.report_evidence
        if (
            self.dashboard_evidence.report.publication_plan.plan.reference
            != self.publication_plan.plan.reference
            or self.dashboard_evidence.report.candidate_report.reference
            != self.candidate_report.reference
            or self.dashboard_evidence.snapshot.reference
            != self.dashboard_snapshot.reference
        ):
            raise EvidenceV2Error("publication aggregate nested evidence ref drift")
        report = validate_candidate_report_source_v2(
            self.candidate_report.read(),
            evidence=report_evidence,
        )
        snapshot = validate_dashboard_snapshot_v2(
            self.dashboard_snapshot.read(),
            evidence=self.dashboard_evidence.report,
        )
        status = validate_dashboard_source_status_v2(
            self.dashboard_source_status.read(),
            evidence=self.dashboard_evidence,
        )
        if (
            report["protocol_attempt_id"] != plan["protocol_attempt_id"]
            or report["run_id"] != plan["run_id"]
            or snapshot["protocol_attempt_id"] != plan["protocol_attempt_id"]
            or snapshot["run_id"] != plan["run_id"]
            or status["protocol_attempt_id"] != plan["protocol_attempt_id"]
            or status["run_id"] != plan["run_id"]
        ):
            raise EvidenceV2Error("publication aggregate lineage drift")
        return plan, readiness, report, snapshot, status


def _artifact_row(
    artifact_id: str,
    artifact: BoundCanonicalArtifact,
) -> dict[str, Any]:
    if not artifact.payload:
        raise EvidenceV2Error("publication aggregate artifact payload is empty")
    return {
        "artifact_id": artifact_id,
        "artifact_ref": artifact.reference.to_dict(),
        "byte_size": len(artifact.payload),
    }


def _blocker_projection(
    status: Mapping[str, Any],
) -> tuple[list[str], list[dict[str, str]]]:
    source_rows = status.get("blocker_sources")
    if not isinstance(source_rows, list):
        raise EvidenceV2Error("Dashboard status blocker sources must be a list")
    rows: list[dict[str, str]] = []
    for item in source_rows:
        if not isinstance(item, Mapping) or set(item) != {"blocker", "source"}:
            raise EvidenceV2Error("Dashboard status blocker source row is invalid")
        rows.append(
            {
                "blocker": str(item["blocker"]),
                "source": f"dashboard_source_status:{item['source']}",
            }
        )
    rows.extend(
        {"blocker": blocker, "source": "publication_aggregate_v2"}
        for blocker in AGGREGATE_BLOCKERS
    )
    if any(not item["blocker"] or not item["source"] for item in rows):
        raise EvidenceV2Error("publication aggregate blocker source row is empty")
    rows.sort(key=lambda item: (item["blocker"], item["source"]))
    blockers = sorted({item["blocker"] for item in rows})
    inherited = {str(item) for item in status.get("blockers", [])}
    if not inherited.issubset(blockers):
        raise EvidenceV2Error("publication aggregate blockers are not monotonic")
    return blockers, rows


def build_publication_aggregate_v2(
    *,
    evidence: PublicationAggregateEvidenceBundleV2,
) -> dict[str, Any]:
    if not isinstance(evidence, PublicationAggregateEvidenceBundleV2):
        raise EvidenceV2Error("publication aggregate requires its typed evidence bundle")
    plan, _readiness, _report, _snapshot, status = evidence.read()
    blockers, blocker_sources = _blocker_projection(status)
    artifacts = [
        _artifact_row("publication_plan", evidence.publication_plan.plan),
        _artifact_row("readiness_v4", evidence.publication_plan.readiness_v4),
        _artifact_row("candidate_report", evidence.candidate_report),
        _artifact_row("dashboard_snapshot", evidence.dashboard_snapshot),
        _artifact_row("dashboard_source_status", evidence.dashboard_source_status),
    ]
    if [item["artifact_id"] for item in artifacts] != list(
        AGGREGATE_ARTIFACT_ORDER
    ):
        raise EvidenceV2Error("publication aggregate artifact order drift")
    refs = [item["artifact_ref"] for item in artifacts]
    identities = [
        (
            item["absolute_path"],
            item["artifact_schema"],
            item["byte_sha256"],
            item["semantic_sha256"],
        )
        for item in refs
    ]
    if len(identities) != len(set(identities)):
        raise EvidenceV2Error("publication aggregate artifact identities are duplicated")
    return seal_semantic(
        {
            "schema_version": PUBLICATION_AGGREGATE_SCHEMA,
            "protocol_attempt_id": plan["protocol_attempt_id"],
            "run_id": plan["run_id"],
            "artifact_order": list(AGGREGATE_ARTIFACT_ORDER),
            "artifacts": artifacts,
            "publication_artifact_set_complete": True,
            "publication_delivery_attested": False,
            "readiness_status": "no_new_risk",
            "blockers": blockers,
            "blocker_sources": blocker_sources,
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
            "production_pointer_switch_authorized": False,
            "codex_activation_authorized": False,
            "dashboard_activation_authorized": False,
            "sealed_live_human_receipt_verified": False,
            "broker_side_effects": False,
        }
    )


def validate_publication_aggregate_v2(
    value: Mapping[str, Any],
    *,
    evidence: PublicationAggregateEvidenceBundleV2,
) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    payload = _exact(
        payload,
        {
            "schema_version",
            "protocol_attempt_id",
            "run_id",
            "artifact_order",
            "artifacts",
            "publication_artifact_set_complete",
            "publication_delivery_attested",
            "readiness_status",
            "blockers",
            "blocker_sources",
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
            "production_pointer_switch_authorized",
            "codex_activation_authorized",
            "dashboard_activation_authorized",
            "sealed_live_human_receipt_verified",
            "broker_side_effects",
            "semantic_sha256",
        },
        label="publication aggregate v2",
    )
    if payload["schema_version"] != PUBLICATION_AGGREGATE_SCHEMA:
        raise EvidenceV2Error("publication aggregate v2 schema mismatch")
    rebuilt = build_publication_aggregate_v2(evidence=evidence)
    if rebuilt != payload:
        raise EvidenceV2Error("publication aggregate v2 drifts from evidence")
    return payload


__all__ = [
    "AGGREGATE_ARTIFACT_ORDER",
    "AGGREGATE_BLOCKERS",
    "PublicationAggregateEvidenceBundleV2",
    "build_publication_aggregate_v2",
    "validate_publication_aggregate_v2",
]
