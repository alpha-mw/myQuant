"""Predeclared, nonauthorizing publication plan for v16 evidence-v2."""

from __future__ import annotations

from dataclasses import dataclass
import posixpath
from collections.abc import Mapping
from typing import Any

from .codex_authority_plan_v2 import PRIVATE_ROOT_POLICY, READINESS_V4_SCHEMA
from .contracts import (
    BoundCanonicalArtifact,
    EvidenceRef,
    EvidenceV2Error,
    seal_semantic,
    validate_semantic_seal,
)
from .readiness_v4 import ReadinessEvidenceBundleV4, validate_v16_run_readiness_v4

PUBLICATION_PLAN_SCHEMA = "v16.publication-source-plan.v2"
CANDIDATE_REPORT_SCHEMA = "v16.candidate-source-report.v2"
DASHBOARD_SNAPSHOT_SCHEMA = "dashboard_contract.v16.evidence-v2"
DASHBOARD_SOURCE_STATUS_SCHEMA = "v16.dashboard-source-status.v2"
PUBLICATION_AGGREGATE_SCHEMA = "v16.publication-aggregate.v2"
PUBLICATION_PLAN_FILENAME = "publication_source_plan_v2.json"

PLANNED_PUBLICATION_SCHEMAS = {
    "candidate_report": CANDIDATE_REPORT_SCHEMA,
    "dashboard_snapshot": DASHBOARD_SNAPSHOT_SCHEMA,
    "dashboard_source_status": DASHBOARD_SOURCE_STATUS_SCHEMA,
    "publication_aggregate": PUBLICATION_AGGREGATE_SCHEMA,
}
PUBLICATION_OUTPUT_ORDER = tuple(PLANNED_PUBLICATION_SCHEMAS)
PUBLICATION_REQUIREMENTS = (
    "candidate_report_publication_attestation_protocol",
    "dashboard_snapshot_delivery_attestation_protocol",
    "dashboard_activation_receipt_v2_protocol",
)


def _exact(value: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise EvidenceV2Error(f"{label} fields mismatch")
    return dict(value)


def _identifier(value: Any, *, label: str) -> str:
    text = str(value or "")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-:"
    if (
        not text
        or text != text.strip()
        or len(text) > 128
        or any(character not in allowed for character in text)
    ):
        raise EvidenceV2Error(f"{label} is not a safe identifier")
    return text


def _private_root(value: Any) -> str:
    text = str(value or "")
    if (
        not text.startswith("/")
        or text.startswith("//")
        or text.endswith("/")
        or "\x00" in text
        or posixpath.normpath(text) != text
    ):
        raise EvidenceV2Error("publication private root must be canonical and absolute")
    return text


def _direct_child(path: Any, *, root: str, label: str) -> str:
    text = str(path or "")
    if (
        not text.startswith("/")
        or text.startswith("//")
        or text.endswith("/")
        or "\x00" in text
        or posixpath.normpath(text) != text
        or posixpath.dirname(text) != root
    ):
        raise EvidenceV2Error(f"{label} must be a direct private-root child")
    return text


@dataclass(frozen=True)
class PlannedPublicationArtifactV2:
    absolute_path: str
    artifact_schema: str
    root_policy: str = PRIVATE_ROOT_POLICY

    def validate_under(self, private_root: str) -> None:
        root = _private_root(private_root)
        _direct_child(
            self.absolute_path,
            root=root,
            label="planned publication artifact",
        )
        if self.root_policy != PRIVATE_ROOT_POLICY or not self.artifact_schema:
            raise EvidenceV2Error("planned publication artifact schema/root is invalid")

    def to_dict(self) -> dict[str, str]:
        return {
            "absolute_path": self.absolute_path,
            "artifact_schema": self.artifact_schema,
            "root_policy": self.root_policy,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlannedPublicationArtifactV2":
        payload = _exact(
            value,
            {"absolute_path", "artifact_schema", "root_policy"},
            label="planned publication artifact v2",
        )
        return cls(**{key: str(payload[key]) for key in payload})


def build_publication_source_plan_v2(
    *,
    protocol_attempt_id: str,
    run_id: str,
    private_root: str,
    plan_absolute_path: str,
    readiness_v4_ref: EvidenceRef,
    planned_artifacts: Mapping[str, PlannedPublicationArtifactV2],
) -> dict[str, Any]:
    root = _private_root(private_root)
    plan_path = _direct_child(
        plan_absolute_path,
        root=root,
        label="publication plan path",
    )
    if posixpath.basename(plan_path) != PUBLICATION_PLAN_FILENAME:
        raise EvidenceV2Error("publication plan path has the wrong filename")
    if (
        not isinstance(readiness_v4_ref, EvidenceRef)
        or readiness_v4_ref.artifact_schema != READINESS_V4_SCHEMA
        or readiness_v4_ref.root_policy != PRIVATE_ROOT_POLICY
    ):
        raise EvidenceV2Error("publication readiness-v4 ref is invalid")
    if not isinstance(planned_artifacts, Mapping) or list(planned_artifacts) != list(
        PLANNED_PUBLICATION_SCHEMAS
    ):
        raise EvidenceV2Error("publication planned artifact keys/order mismatch")
    normalized: dict[str, dict[str, str]] = {}
    paths = [plan_path]
    for key, expected_schema in PLANNED_PUBLICATION_SCHEMAS.items():
        artifact = planned_artifacts[key]
        if not isinstance(artifact, PlannedPublicationArtifactV2):
            raise EvidenceV2Error("publication planned artifact has the wrong type")
        artifact.validate_under(root)
        if artifact.artifact_schema != expected_schema:
            raise EvidenceV2Error(f"publication planned schema mismatch: {key}")
        normalized[key] = artifact.to_dict()
        paths.append(artifact.absolute_path)
    if len(paths) != len(set(paths)):
        raise EvidenceV2Error("publication plan/output paths must be unique")
    return seal_semantic(
        {
            "schema_version": PUBLICATION_PLAN_SCHEMA,
            "protocol_attempt_id": _identifier(
                protocol_attempt_id,
                label="protocol_attempt_id",
            ),
            "run_id": _identifier(run_id, label="run_id"),
            "private_root": root,
            "plan_absolute_path": plan_path,
            "readiness_v4_ref": readiness_v4_ref.to_dict(),
            "planned_artifacts": normalized,
            "output_order": list(PUBLICATION_OUTPUT_ORDER),
            "unsupported_requirement_ids": list(PUBLICATION_REQUIREMENTS),
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_publication_source_plan_v2(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    payload = _exact(
        payload,
        {
            "schema_version",
            "protocol_attempt_id",
            "run_id",
            "private_root",
            "plan_absolute_path",
            "readiness_v4_ref",
            "planned_artifacts",
            "output_order",
            "unsupported_requirement_ids",
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
            "semantic_sha256",
        },
        label="publication source plan v2",
    )
    if payload["schema_version"] != PUBLICATION_PLAN_SCHEMA:
        raise EvidenceV2Error("publication source plan v2 schema mismatch")
    artifacts = payload["planned_artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        PLANNED_PUBLICATION_SCHEMAS
    ):
        raise EvidenceV2Error("publication planned artifact keys mismatch")
    rebuilt = build_publication_source_plan_v2(
        protocol_attempt_id=str(payload["protocol_attempt_id"]),
        run_id=str(payload["run_id"]),
        private_root=str(payload["private_root"]),
        plan_absolute_path=str(payload["plan_absolute_path"]),
        readiness_v4_ref=EvidenceRef.from_dict(payload["readiness_v4_ref"]),
        planned_artifacts={
            key: PlannedPublicationArtifactV2.from_dict(artifacts[key])
            for key in PLANNED_PUBLICATION_SCHEMAS
        },
    )
    if rebuilt != payload:
        raise EvidenceV2Error("publication source plan v2 is not canonical")
    return payload


@dataclass(frozen=True)
class PublicationPlanEvidenceBundleV2:
    plan: BoundCanonicalArtifact
    readiness_v4: BoundCanonicalArtifact
    readiness_evidence: ReadinessEvidenceBundleV4

    def read(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if (
            not isinstance(self.plan, BoundCanonicalArtifact)
            or self.plan.reference.artifact_schema != PUBLICATION_PLAN_SCHEMA
            or self.plan.reference.root_policy != PRIVATE_ROOT_POLICY
            or not isinstance(self.readiness_v4, BoundCanonicalArtifact)
            or self.readiness_v4.reference.artifact_schema != READINESS_V4_SCHEMA
            or self.readiness_v4.reference.root_policy != PRIVATE_ROOT_POLICY
            or not isinstance(self.readiness_evidence, ReadinessEvidenceBundleV4)
        ):
            raise EvidenceV2Error("publication plan evidence bundle is invalid")
        plan = validate_publication_source_plan_v2(self.plan.read())
        if (
            self.plan.reference.absolute_path != plan["plan_absolute_path"]
            or plan["readiness_v4_ref"] != self.readiness_v4.reference.to_dict()
        ):
            raise EvidenceV2Error("publication plan bound path/readiness ref drift")
        readiness = validate_v16_run_readiness_v4(
            self.readiness_v4.read(),
            evidence=self.readiness_evidence,
        )
        if (
            readiness["protocol_attempt_id"] != plan["protocol_attempt_id"]
            or readiness["run_id"] != plan["run_id"]
        ):
            raise EvidenceV2Error("publication plan/readiness lineage drift")
        return plan, readiness


__all__ = [
    "CANDIDATE_REPORT_SCHEMA",
    "DASHBOARD_SNAPSHOT_SCHEMA",
    "DASHBOARD_SOURCE_STATUS_SCHEMA",
    "PLANNED_PUBLICATION_SCHEMAS",
    "PRIVATE_ROOT_POLICY",
    "PUBLICATION_AGGREGATE_SCHEMA",
    "PUBLICATION_OUTPUT_ORDER",
    "PUBLICATION_PLAN_FILENAME",
    "PUBLICATION_PLAN_SCHEMA",
    "PUBLICATION_REQUIREMENTS",
    "PlannedPublicationArtifactV2",
    "PublicationPlanEvidenceBundleV2",
    "build_publication_source_plan_v2",
    "validate_publication_source_plan_v2",
]
