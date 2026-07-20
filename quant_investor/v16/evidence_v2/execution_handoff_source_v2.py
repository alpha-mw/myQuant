"""Fail-closed v16 execution and handoff source-status projections.

The projections reopen only the source plan and recomputed Codex IC status.
They do not accept legacy capital maps, human receipts, authorization booleans,
execution plans, market state, order eligibility, or handoff delivery claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from .codex_authority_plan_v2 import (
    CODEX_IC_STATUS_SCHEMA,
    EXECUTION_HANDOFF_REQUIREMENTS,
    EXECUTION_SOURCE_STATUS_SCHEMA,
    HANDOFF_SOURCE_STATUS_SCHEMA,
    PRIVATE_ROOT_POLICY,
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

EXECUTION_REQUIREMENTS = EXECUTION_HANDOFF_REQUIREMENTS[:2]
HANDOFF_REQUIREMENTS = EXECUTION_HANDOFF_REQUIREMENTS[2:]
PERMANENT_EXECUTION_HANDOFF_BLOCKERS = (
    "bare_capital_execution_handoff_mapping_not_accepted_as_evidence",
    "caller_human_authorized_boolean_not_accepted_as_evidence",
    "codex_authority_v2_disconnected_from_authorizing_consumers",
    "external_antirollback_authority_not_integrated",
    "legacy_codex_human_authorization_v1_not_accepted_as_live_authority",
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
class ValidatedExecutionHandoffSourceV2:
    plan: dict[str, Any]
    ic_status: dict[str, Any]


@dataclass(frozen=True)
class ExecutionHandoffSourceEvidenceBundleV2:
    plan: CodexAuthorityPlanEvidenceBundleV2
    ic_status: BoundCanonicalArtifact
    ic_evidence: CodexICSourceEvidenceBundleV2

    def read(self) -> ValidatedExecutionHandoffSourceV2:
        if (
            not isinstance(self.plan, CodexAuthorityPlanEvidenceBundleV2)
            or not isinstance(self.ic_status, BoundCanonicalArtifact)
            or not isinstance(self.ic_evidence, CodexICSourceEvidenceBundleV2)
        ):
            raise EvidenceV2Error("execution/handoff source bundle types are invalid")
        plan = self.plan.read()
        if (
            self.ic_status.reference.artifact_schema != CODEX_IC_STATUS_SCHEMA
            or self.ic_status.reference.root_policy != PRIVATE_ROOT_POLICY
            or not _matches_planned_reference(
                self.ic_status.reference,
                plan["planned_artifacts"]["ic_status"],
            )
            or self.ic_evidence.plan.plan.reference != self.plan.plan.reference
        ):
            raise EvidenceV2Error("execution/handoff IC status drifts from source plan")
        ic_status = validate_codex_ic_source_status_v2(
            self.ic_status.read(),
            evidence=self.ic_evidence,
        )
        if (
            ic_status["protocol_attempt_id"] != plan["protocol_attempt_id"]
            or ic_status["run_id"] != plan["run_id"]
            or ic_status["source_plan_ref"] != self.plan.plan.reference.to_dict()
        ):
            raise EvidenceV2Error("execution/handoff IC lineage drift")
        return ValidatedExecutionHandoffSourceV2(plan=plan, ic_status=ic_status)


def _blocker_projection(
    *,
    status_kind: str,
    requirement_ids: tuple[str, ...],
    ic_status: Mapping[str, Any],
) -> tuple[list[str], list[dict[str, str]]]:
    rows = [
        {
            "blocker": f"codex_requirement_unsupported:requirement={requirement}",
            "source": f"{status_kind}_source:requirement:{requirement}",
        }
        for requirement in requirement_ids
    ]
    rows.extend(
        {
            "blocker": blocker,
            "source": f"{status_kind}_source_status",
        }
        for blocker in (
            *PERMANENT_EXECUTION_HANDOFF_BLOCKERS,
            f"{status_kind}_source_recomputation_incomplete",
        )
    )
    ic_sources = ic_status.get("blocker_sources")
    if not isinstance(ic_sources, list):
        raise EvidenceV2Error("Codex IC blocker sources must be a list")
    for item in ic_sources:
        if not isinstance(item, Mapping) or set(item) != {"blocker", "source"}:
            raise EvidenceV2Error("Codex IC blocker source row is invalid")
        rows.append(
            {
                "blocker": str(item["blocker"]),
                "source": f"codex_ic_source:{item['source']}",
            }
        )
    if any(not item["blocker"] or not item["source"] for item in rows):
        raise EvidenceV2Error("execution/handoff blocker source row is empty")
    rows.sort(key=lambda item: (item["blocker"], item["source"]))
    return sorted({item["blocker"] for item in rows}), rows


def _build_source_status(
    *,
    evidence: ExecutionHandoffSourceEvidenceBundleV2,
    schema_version: str,
    status_kind: str,
    requirement_ids: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(evidence, ExecutionHandoffSourceEvidenceBundleV2):
        raise EvidenceV2Error(
            "execution/handoff source status requires its typed evidence bundle"
        )
    validated = evidence.read()
    blockers, blocker_sources = _blocker_projection(
        status_kind=status_kind,
        requirement_ids=requirement_ids,
        ic_status=validated.ic_status,
    )
    return seal_semantic(
        {
            "schema_version": schema_version,
            "protocol_attempt_id": validated.plan["protocol_attempt_id"],
            "run_id": validated.plan["run_id"],
            "source_plan_ref": evidence.plan.plan.reference.to_dict(),
            "codex_ic_status_ref": evidence.ic_status.reference.to_dict(),
            "artifact_role": "source_status_only",
            "unsupported_requirement_ids": list(requirement_ids),
            "source_recomputation_complete": False,
            "readiness_status": "no_new_risk",
            "blockers": blockers,
            "blocker_sources": blocker_sources,
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def build_execution_source_status_v2(
    *,
    evidence: ExecutionHandoffSourceEvidenceBundleV2,
) -> dict[str, Any]:
    return _build_source_status(
        evidence=evidence,
        schema_version=EXECUTION_SOURCE_STATUS_SCHEMA,
        status_kind="execution",
        requirement_ids=EXECUTION_REQUIREMENTS,
    )


def build_handoff_source_status_v2(
    *,
    evidence: ExecutionHandoffSourceEvidenceBundleV2,
) -> dict[str, Any]:
    return _build_source_status(
        evidence=evidence,
        schema_version=HANDOFF_SOURCE_STATUS_SCHEMA,
        status_kind="handoff",
        requirement_ids=HANDOFF_REQUIREMENTS,
    )


def _validate_source_status(
    value: Mapping[str, Any],
    *,
    evidence: ExecutionHandoffSourceEvidenceBundleV2,
    schema_version: str,
    builder: Any,
) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    payload = _exact(
        payload,
        {
            "schema_version",
            "protocol_attempt_id",
            "run_id",
            "source_plan_ref",
            "codex_ic_status_ref",
            "artifact_role",
            "unsupported_requirement_ids",
            "source_recomputation_complete",
            "readiness_status",
            "blockers",
            "blocker_sources",
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
            "semantic_sha256",
        },
        label=f"{schema_version} source status",
    )
    if payload["schema_version"] != schema_version:
        raise EvidenceV2Error("execution/handoff source status schema mismatch")
    rebuilt = builder(evidence=evidence)
    if rebuilt != payload:
        raise EvidenceV2Error("execution/handoff source status drifts from evidence")
    return payload


def validate_execution_source_status_v2(
    value: Mapping[str, Any],
    *,
    evidence: ExecutionHandoffSourceEvidenceBundleV2,
) -> dict[str, Any]:
    return _validate_source_status(
        value,
        evidence=evidence,
        schema_version=EXECUTION_SOURCE_STATUS_SCHEMA,
        builder=build_execution_source_status_v2,
    )


def validate_handoff_source_status_v2(
    value: Mapping[str, Any],
    *,
    evidence: ExecutionHandoffSourceEvidenceBundleV2,
) -> dict[str, Any]:
    return _validate_source_status(
        value,
        evidence=evidence,
        schema_version=HANDOFF_SOURCE_STATUS_SCHEMA,
        builder=build_handoff_source_status_v2,
    )


__all__ = [
    "EXECUTION_REQUIREMENTS",
    "HANDOFF_REQUIREMENTS",
    "PERMANENT_EXECUTION_HANDOFF_BLOCKERS",
    "ExecutionHandoffSourceEvidenceBundleV2",
    "ValidatedExecutionHandoffSourceV2",
    "build_execution_source_status_v2",
    "build_handoff_source_status_v2",
    "validate_execution_source_status_v2",
    "validate_handoff_source_status_v2",
]
