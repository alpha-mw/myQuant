"""Aggregate fail-closed readiness for the v16 Codex authority source lane."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from .codex_authority_plan_v2 import (
    CODEX_IC_STATUS_SCHEMA,
    EXECUTION_SOURCE_STATUS_SCHEMA,
    HANDOFF_SOURCE_STATUS_SCHEMA,
    PRIVATE_ROOT_POLICY,
    READINESS_V3_SCHEMA,
    READINESS_V4_SCHEMA,
    CodexAuthorityPlanEvidenceBundleV2,
)
from .codex_ic_source_v2 import (
    CodexICSourceEvidenceBundleV2,
    validate_codex_ic_source_status_v2,
)
from .contracts import (
    BoundCanonicalArtifact,
    EvidenceRef,
    EvidenceV2Error,
    seal_semantic,
    validate_semantic_seal,
)
from .execution_handoff_source_v2 import (
    ExecutionHandoffSourceEvidenceBundleV2,
    validate_execution_source_status_v2,
    validate_handoff_source_status_v2,
)
from .readiness_v3 import (
    ARCHITECTURE_VERSION,
    ReadinessEvidenceBundleV3,
    V16ReadinessV3Error,
    validate_v16_run_readiness_v3,
)

SCHEMA_VERSION = READINESS_V4_SCHEMA
ARTIFACT_FILENAME = "v16_run_readiness_v4.json"
READINESS_V4_FOUNDATION_BLOCKERS = (
    "codex_activation_authority_not_integrated",
    "dashboard_activation_receipt_v2_not_integrated",
    "live_human_identity_signature_protocol_not_integrated",
    "production_pointer_switch_not_authorized",
    "readiness_v4_source_status_schema_nonauthorizing",
)


def _exact(value: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise EvidenceV2Error(f"{label} fields mismatch")
    return dict(value)


def _matches_planned_reference(
    reference: EvidenceRef,
    planned: Mapping[str, Any],
) -> bool:
    return (
        reference.absolute_path == planned["absolute_path"]
        and reference.artifact_schema == planned["artifact_schema"]
        and reference.root_policy == planned["root_policy"]
    )


@dataclass(frozen=True)
class ValidatedReadinessSourceV4:
    readiness_v3: dict[str, Any]
    plan: dict[str, Any]
    ic_status: dict[str, Any]
    execution_status: dict[str, Any]
    handoff_status: dict[str, Any]


@dataclass(frozen=True)
class ReadinessEvidenceBundleV4:
    readiness_v3: BoundCanonicalArtifact
    readiness_v3_evidence: ReadinessEvidenceBundleV3
    plan: CodexAuthorityPlanEvidenceBundleV2
    ic_status: BoundCanonicalArtifact
    ic_evidence: CodexICSourceEvidenceBundleV2
    execution_status: BoundCanonicalArtifact
    handoff_status: BoundCanonicalArtifact
    execution_handoff_evidence: ExecutionHandoffSourceEvidenceBundleV2

    def read(self) -> ValidatedReadinessSourceV4:
        if (
            not isinstance(self.readiness_v3, BoundCanonicalArtifact)
            or not isinstance(self.readiness_v3_evidence, ReadinessEvidenceBundleV3)
            or not isinstance(self.plan, CodexAuthorityPlanEvidenceBundleV2)
            or not isinstance(self.ic_status, BoundCanonicalArtifact)
            or not isinstance(self.ic_evidence, CodexICSourceEvidenceBundleV2)
            or not isinstance(self.execution_status, BoundCanonicalArtifact)
            or not isinstance(self.handoff_status, BoundCanonicalArtifact)
            or not isinstance(
                self.execution_handoff_evidence,
                ExecutionHandoffSourceEvidenceBundleV2,
            )
        ):
            raise EvidenceV2Error("readiness-v4 evidence bundle types are invalid")
        if (
            self.readiness_v3.reference.artifact_schema != READINESS_V3_SCHEMA
            or self.readiness_v3.reference.root_policy != PRIVATE_ROOT_POLICY
        ):
            raise EvidenceV2Error("readiness-v4 readiness-v3 ref is invalid")
        try:
            readiness_v3 = validate_v16_run_readiness_v3(
                self.readiness_v3.read(),
                evidence=self.readiness_v3_evidence,
            )
        except V16ReadinessV3Error as exc:
            raise EvidenceV2Error(str(exc)) from exc
        plan = self.plan.read()
        if (
            self.plan.readiness_v3.reference != self.readiness_v3.reference
            or plan["readiness_v3_ref"] != self.readiness_v3.reference.to_dict()
            or readiness_v3["run_id"] != plan["run_id"]
        ):
            raise EvidenceV2Error("readiness-v4 plan/readiness-v3 lineage drift")

        planned = plan["planned_artifacts"]
        for key, artifact, schema in (
            ("ic_status", self.ic_status, CODEX_IC_STATUS_SCHEMA),
            ("execution_status", self.execution_status, EXECUTION_SOURCE_STATUS_SCHEMA),
            ("handoff_status", self.handoff_status, HANDOFF_SOURCE_STATUS_SCHEMA),
        ):
            if (
                artifact.reference.artifact_schema != schema
                or artifact.reference.root_policy != PRIVATE_ROOT_POLICY
                or not _matches_planned_reference(artifact.reference, planned[key])
            ):
                raise EvidenceV2Error(f"readiness-v4 artifact drifts from plan: {key}")
        if (
            self.ic_evidence.plan.plan.reference != self.plan.plan.reference
            or self.execution_handoff_evidence.plan.plan.reference
            != self.plan.plan.reference
            or self.execution_handoff_evidence.ic_status.reference
            != self.ic_status.reference
        ):
            raise EvidenceV2Error("readiness-v4 typed evidence lineage drift")
        ic_status = validate_codex_ic_source_status_v2(
            self.ic_status.read(),
            evidence=self.ic_evidence,
        )
        execution_status = validate_execution_source_status_v2(
            self.execution_status.read(),
            evidence=self.execution_handoff_evidence,
        )
        handoff_status = validate_handoff_source_status_v2(
            self.handoff_status.read(),
            evidence=self.execution_handoff_evidence,
        )
        for label, status in (
            ("IC", ic_status),
            ("execution", execution_status),
            ("handoff", handoff_status),
        ):
            if (
                status["protocol_attempt_id"] != plan["protocol_attempt_id"]
                or status["run_id"] != plan["run_id"]
            ):
                raise EvidenceV2Error(f"readiness-v4 {label} status lineage drift")
        return ValidatedReadinessSourceV4(
            readiness_v3=readiness_v3,
            plan=plan,
            ic_status=ic_status,
            execution_status=execution_status,
            handoff_status=handoff_status,
        )


def _blocker_projection(
    *,
    validated: ValidatedReadinessSourceV4,
) -> tuple[list[str], list[dict[str, str]]]:
    rows = [
        {"blocker": blocker, "source": "readiness_v4_foundation"}
        for blocker in READINESS_V4_FOUNDATION_BLOCKERS
    ]
    for source, payload in (
        ("readiness_v3", validated.readiness_v3),
        ("codex_ic_source", validated.ic_status),
        ("execution_source", validated.execution_status),
        ("handoff_source", validated.handoff_status),
    ):
        source_rows = payload.get("blocker_sources")
        if not isinstance(source_rows, list):
            raise EvidenceV2Error(f"{source} blocker sources must be a list")
        for item in source_rows:
            if not isinstance(item, Mapping) or set(item) != {"blocker", "source"}:
                raise EvidenceV2Error(f"{source} blocker source row is invalid")
            rows.append(
                {
                    "blocker": str(item["blocker"]),
                    "source": f"{source}:{item['source']}",
                }
            )
    if any(not item["blocker"] or not item["source"] for item in rows):
        raise EvidenceV2Error("readiness-v4 blocker source row is empty")
    rows.sort(key=lambda item: (item["blocker"], item["source"]))
    return sorted({item["blocker"] for item in rows}), rows


def build_v16_run_readiness_v4(
    *,
    evidence: ReadinessEvidenceBundleV4,
) -> dict[str, Any]:
    if not isinstance(evidence, ReadinessEvidenceBundleV4):
        raise EvidenceV2Error("readiness-v4 requires its typed evidence bundle")
    validated = evidence.read()
    blockers, blocker_sources = _blocker_projection(validated=validated)
    return seal_semantic(
        {
            "schema_version": SCHEMA_VERSION,
            "architecture_version": ARCHITECTURE_VERSION,
            "artifact_filename": ARTIFACT_FILENAME,
            "protocol_attempt_id": validated.plan["protocol_attempt_id"],
            "run_id": validated.plan["run_id"],
            "generated_at": validated.readiness_v3["generated_at"],
            "analysis_trade_date": validated.readiness_v3["analysis_trade_date"],
            "formal_branches": validated.readiness_v3["formal_branches"],
            "retrieval_role": validated.readiness_v3["retrieval_role"],
            "risk_advisor_role": validated.readiness_v3["risk_advisor_role"],
            "evidence_refs": {
                "readiness_v3": evidence.readiness_v3.reference.to_dict(),
                "codex_authority_plan": evidence.plan.plan.reference.to_dict(),
                "codex_ic_status": evidence.ic_status.reference.to_dict(),
                "execution_status": evidence.execution_status.reference.to_dict(),
                "handoff_status": evidence.handoff_status.reference.to_dict(),
            },
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
            "production_pointer_switch_authorized": False,
            "codex_activation_authorized": False,
            "dashboard_activation_authorized": False,
            "sealed_live_human_receipt_verified": False,
            "broker_side_effects": False,
            "readiness_status": "no_new_risk",
            "blockers": blockers,
            "blocker_sources": blocker_sources,
        }
    )


def validate_v16_run_readiness_v4(
    value: Mapping[str, Any],
    *,
    evidence: ReadinessEvidenceBundleV4,
) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    payload = _exact(
        payload,
        {
            "schema_version",
            "architecture_version",
            "artifact_filename",
            "protocol_attempt_id",
            "run_id",
            "generated_at",
            "analysis_trade_date",
            "formal_branches",
            "retrieval_role",
            "risk_advisor_role",
            "evidence_refs",
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
            "production_pointer_switch_authorized",
            "codex_activation_authorized",
            "dashboard_activation_authorized",
            "sealed_live_human_receipt_verified",
            "broker_side_effects",
            "readiness_status",
            "blockers",
            "blocker_sources",
            "semantic_sha256",
        },
        label="v16 readiness v4",
    )
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["architecture_version"] != ARCHITECTURE_VERSION
        or payload["artifact_filename"] != ARTIFACT_FILENAME
    ):
        raise EvidenceV2Error("readiness-v4 identity mismatch")
    rebuilt = build_v16_run_readiness_v4(evidence=evidence)
    if rebuilt != payload:
        raise EvidenceV2Error("readiness-v4 drifts from reopened evidence")
    return payload


__all__ = [
    "ARTIFACT_FILENAME",
    "READINESS_V4_FOUNDATION_BLOCKERS",
    "SCHEMA_VERSION",
    "ReadinessEvidenceBundleV4",
    "ValidatedReadinessSourceV4",
    "build_v16_run_readiness_v4",
    "validate_v16_run_readiness_v4",
]
