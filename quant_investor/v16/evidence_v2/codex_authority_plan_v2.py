"""Predeclared source plan for the disconnected v16 Codex authority lane."""

from __future__ import annotations

from dataclasses import dataclass
import posixpath
from collections.abc import Mapping
from typing import Any

from .contracts import (
    BoundCanonicalArtifact,
    EvidenceRef,
    EvidenceV2Error,
    seal_semantic,
    validate_semantic_seal,
)

CODEX_AUTHORITY_PLAN_SCHEMA = "v16.codex-authority-source-plan.v2"
CODEX_IC_STATUS_SCHEMA = "v16.codex-ic-source-status.v2"
EXECUTION_SOURCE_STATUS_SCHEMA = "v16.execution-source-status.v2"
HANDOFF_SOURCE_STATUS_SCHEMA = "v16.handoff-source-status.v2"
READINESS_V4_SCHEMA = "v16_run_readiness.v4"
FULL_UNION_POSTERIOR_SCHEMA = "v16.full-union-posterior-evidence.v2"
READINESS_V3_SCHEMA = "v16_run_readiness.v3"
MENU_SCHEMA = "codex-review-menu.v1"
STAGE2_REQUEST_SCHEMA = "codex-review-stage2-request.v1"
STAGE2_RESPONSE_SCHEMA = "codex-review-stage2-response.v1"
PRIVATE_ROOT_POLICY = "v16.private-evidence-root.v2"

MENU_REQUIREMENTS = (
    "menu_position_source_contract",
    "menu_reference_price_source_contract",
    "risk_advisory_source_contract",
    "stage2_model_execution_attestation",
)
EXECUTION_HANDOFF_REQUIREMENTS = (
    "execution_plan_source_contract",
    "execution_market_state_source_contract",
    "live_human_identity_signature_protocol",
    "handoff_delivery_attestation_protocol",
)
UNSUPPORTED_REQUIREMENTS = MENU_REQUIREMENTS + EXECUTION_HANDOFF_REQUIREMENTS

PLANNED_ARTIFACT_SCHEMAS = {
    "menu": MENU_SCHEMA,
    "stage2_request": STAGE2_REQUEST_SCHEMA,
    "stage2_response": STAGE2_RESPONSE_SCHEMA,
    "ic_status": CODEX_IC_STATUS_SCHEMA,
    "execution_status": EXECUTION_SOURCE_STATUS_SCHEMA,
    "handoff_status": HANDOFF_SOURCE_STATUS_SCHEMA,
    "readiness_v4": READINESS_V4_SCHEMA,
}


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
        raise EvidenceV2Error("Codex authority private root must be canonical and absolute")
    return text


@dataclass(frozen=True)
class PlannedCodexArtifactV2:
    absolute_path: str
    artifact_schema: str
    root_policy: str = PRIVATE_ROOT_POLICY

    def validate_under(self, private_root: str) -> None:
        root = _private_root(private_root)
        path = str(self.absolute_path or "")
        if (
            not path.startswith("/")
            or path.startswith("//")
            or path.endswith("/")
            or "\x00" in path
            or posixpath.normpath(path) != path
        ):
            raise EvidenceV2Error("planned Codex artifact path is not canonical")
        try:
            common = posixpath.commonpath((root, path))
        except ValueError as exc:
            raise EvidenceV2Error("planned Codex artifact escapes private root") from exc
        if common != root or path == root:
            raise EvidenceV2Error("planned Codex artifact must be a private-root child")
        if self.root_policy != PRIVATE_ROOT_POLICY or not self.artifact_schema:
            raise EvidenceV2Error("planned Codex artifact schema/root policy is invalid")

    def to_dict(self) -> dict[str, str]:
        return {
            "absolute_path": self.absolute_path,
            "artifact_schema": self.artifact_schema,
            "root_policy": self.root_policy,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlannedCodexArtifactV2":
        payload = _exact(
            value,
            {"absolute_path", "artifact_schema", "root_policy"},
            label="planned Codex artifact v2",
        )
        return cls(**{key: str(payload[key]) for key in payload})


def _source_ref(reference: EvidenceRef, *, schema: str, label: str) -> dict[str, str]:
    if (
        not isinstance(reference, EvidenceRef)
        or reference.artifact_schema != schema
        or reference.root_policy != PRIVATE_ROOT_POLICY
    ):
        raise EvidenceV2Error(f"Codex authority {label} ref is invalid")
    return reference.to_dict()


def build_codex_authority_source_plan_v2(
    *,
    protocol_attempt_id: str,
    run_id: str,
    private_root: str,
    full_union_posterior_ref: EvidenceRef,
    readiness_v3_ref: EvidenceRef,
    planned_artifacts: Mapping[str, PlannedCodexArtifactV2],
) -> dict[str, Any]:
    root = _private_root(private_root)
    if not isinstance(planned_artifacts, Mapping) or list(planned_artifacts) != list(
        PLANNED_ARTIFACT_SCHEMAS
    ):
        raise EvidenceV2Error("Codex authority planned artifact keys/order mismatch")
    normalized: dict[str, dict[str, str]] = {}
    paths: list[str] = []
    for key, expected_schema in PLANNED_ARTIFACT_SCHEMAS.items():
        artifact = planned_artifacts[key]
        if not isinstance(artifact, PlannedCodexArtifactV2):
            raise EvidenceV2Error("Codex authority planned artifact has the wrong type")
        artifact.validate_under(root)
        if artifact.artifact_schema != expected_schema:
            raise EvidenceV2Error(f"Codex authority planned schema mismatch: {key}")
        normalized[key] = artifact.to_dict()
        paths.append(artifact.absolute_path)
    if len(paths) != len(set(paths)):
        raise EvidenceV2Error("Codex authority future artifact paths must be unique")
    return seal_semantic(
        {
            "schema_version": CODEX_AUTHORITY_PLAN_SCHEMA,
            "protocol_attempt_id": _identifier(
                protocol_attempt_id,
                label="protocol_attempt_id",
            ),
            "run_id": _identifier(run_id, label="run_id"),
            "private_root": root,
            "full_union_posterior_ref": _source_ref(
                full_union_posterior_ref,
                schema=FULL_UNION_POSTERIOR_SCHEMA,
                label="full-union posterior",
            ),
            "readiness_v3_ref": _source_ref(
                readiness_v3_ref,
                schema=READINESS_V3_SCHEMA,
                label="readiness-v3",
            ),
            "planned_artifacts": normalized,
            "unsupported_requirement_ids": list(UNSUPPORTED_REQUIREMENTS),
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_codex_authority_source_plan_v2(
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
            "full_union_posterior_ref",
            "readiness_v3_ref",
            "planned_artifacts",
            "unsupported_requirement_ids",
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
            "semantic_sha256",
        },
        label="Codex authority source plan v2",
    )
    if payload["schema_version"] != CODEX_AUTHORITY_PLAN_SCHEMA:
        raise EvidenceV2Error("Codex authority source plan v2 schema mismatch")
    artifacts = payload["planned_artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        PLANNED_ARTIFACT_SCHEMAS
    ):
        raise EvidenceV2Error("Codex authority planned artifact keys mismatch")
    rebuilt = build_codex_authority_source_plan_v2(
        protocol_attempt_id=str(payload["protocol_attempt_id"]),
        run_id=str(payload["run_id"]),
        private_root=str(payload["private_root"]),
        full_union_posterior_ref=EvidenceRef.from_dict(
            payload["full_union_posterior_ref"]
        ),
        readiness_v3_ref=EvidenceRef.from_dict(payload["readiness_v3_ref"]),
        planned_artifacts={
            key: PlannedCodexArtifactV2.from_dict(artifacts[key])
            for key in PLANNED_ARTIFACT_SCHEMAS
        },
    )
    if rebuilt != payload:
        raise EvidenceV2Error("Codex authority source plan v2 is not canonical")
    return payload


@dataclass(frozen=True)
class CodexAuthorityPlanEvidenceBundleV2:
    plan: BoundCanonicalArtifact
    full_union_posterior: BoundCanonicalArtifact
    readiness_v3: BoundCanonicalArtifact

    def read(self) -> dict[str, Any]:
        if (
            self.plan.reference.artifact_schema != CODEX_AUTHORITY_PLAN_SCHEMA
            or self.plan.reference.root_policy != PRIVATE_ROOT_POLICY
        ):
            raise EvidenceV2Error("Codex authority plan evidence ref is invalid")
        plan = validate_codex_authority_source_plan_v2(self.plan.read())
        for field, artifact, schema in (
            (
                "full_union_posterior_ref",
                self.full_union_posterior,
                FULL_UNION_POSTERIOR_SCHEMA,
            ),
            ("readiness_v3_ref", self.readiness_v3, READINESS_V3_SCHEMA),
        ):
            if (
                not isinstance(artifact, BoundCanonicalArtifact)
                or artifact.reference.artifact_schema != schema
                or artifact.reference.root_policy != PRIVATE_ROOT_POLICY
                or plan[field] != artifact.reference.to_dict()
            ):
                raise EvidenceV2Error(f"Codex authority plan evidence ref drift: {field}")
            artifact.read()
        return plan


__all__ = [
    "CODEX_AUTHORITY_PLAN_SCHEMA",
    "CODEX_IC_STATUS_SCHEMA",
    "EXECUTION_HANDOFF_REQUIREMENTS",
    "EXECUTION_SOURCE_STATUS_SCHEMA",
    "FULL_UNION_POSTERIOR_SCHEMA",
    "HANDOFF_SOURCE_STATUS_SCHEMA",
    "MENU_REQUIREMENTS",
    "PLANNED_ARTIFACT_SCHEMAS",
    "PRIVATE_ROOT_POLICY",
    "READINESS_V3_SCHEMA",
    "READINESS_V4_SCHEMA",
    "UNSUPPORTED_REQUIREMENTS",
    "CodexAuthorityPlanEvidenceBundleV2",
    "PlannedCodexArtifactV2",
    "build_codex_authority_source_plan_v2",
    "validate_codex_authority_source_plan_v2",
]
