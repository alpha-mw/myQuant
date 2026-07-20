"""Source validation before a v16 calibration target can be computed."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

from .calibration_plan_v3 import (
    CALIBRATION_PLAN_V3_SCHEMA,
    COST_STATUS_SCHEMA,
    PRIVATE_ROOT_POLICY,
    STOCK_SOURCE_SET_SCHEMA,
    TARGET_STATUS_SCHEMA,
    validate_calibration_universe_plan_v3,
)
from .contracts import (
    BoundCanonicalArtifact,
    BoundRawArtifact,
    EvidenceRef,
    EvidenceV2Error,
    seal_semantic,
    validate_semantic_seal,
)
from .schedule_v4 import ScheduleAnchorBindingV4, validate_schedule_anchor_binding_v4
from .target import (
    ADJUSTMENT_FACTOR_EVIDENCE_SCHEMA,
    INDEX_MANIFEST_SCHEMA,
    INDEX_TABLE_SCHEMA,
    PIT_MEMBERSHIP_EVIDENCE_SCHEMA,
    STOCK_MARK_TABLE_SCHEMA,
    SUSPENSION_EVIDENCE_SCHEMA,
    StockMarkSourceBundle,
    prepare_stock_mark_sources,
    validate_h00300_manifest_with_parquet,
    validate_stock_mark_evidence_from_sources,
)

EIGHT_COMPONENT_COST_REQUIREMENT = "eight_component_cost_model"
STOCK_SOURCE_KEYS = (
    "market_parquet",
    "adjustment_factors",
    "pit_membership",
    "suspensions",
)
STOCK_SOURCE_SCHEMAS = {
    "market_parquet": STOCK_MARK_TABLE_SCHEMA,
    "adjustment_factors": ADJUSTMENT_FACTOR_EVIDENCE_SCHEMA,
    "pit_membership": PIT_MEMBERSHIP_EVIDENCE_SCHEMA,
    "suspensions": SUSPENSION_EVIDENCE_SCHEMA,
}
STOCK_SOURCE_ROOT_POLICIES = {
    "market_parquet": "v16.governed-data-root.v2",
    "adjustment_factors": PRIVATE_ROOT_POLICY,
    "pit_membership": PRIVATE_ROOT_POLICY,
    "suspensions": PRIVATE_ROOT_POLICY,
}


def _exact(value: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise EvidenceV2Error(f"{label} fields mismatch")
    return dict(value)


def _identifier(value: Any, *, label: str) -> str:
    text = str(value or "")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if (
        not text
        or text != text.strip()
        or len(text) > 128
        or any(character not in allowed for character in text)
    ):
        raise EvidenceV2Error(f"{label} is not a safe identifier")
    return text


def _matches_planned_reference(
    reference: EvidenceRef,
    planned: Mapping[str, Any],
) -> bool:
    return (
        reference.absolute_path == planned["absolute_path"]
        and reference.artifact_schema == planned["artifact_schema"]
        and reference.root_policy == planned["root_policy"]
    )


def _bound_plan(plan: BoundCanonicalArtifact) -> dict[str, Any]:
    if (
        not isinstance(plan, BoundCanonicalArtifact)
        or plan.reference.artifact_schema != CALIBRATION_PLAN_V3_SCHEMA
        or plan.reference.root_policy != PRIVATE_ROOT_POLICY
    ):
        raise EvidenceV2Error("target v4 requires a bound calibration plan v3")
    return validate_calibration_universe_plan_v3(plan.read())


def _sample(plan: Mapping[str, Any], sample_id: str) -> dict[str, Any]:
    matches = [item for item in plan["sample_plans"] if item["sample_id"] == sample_id]
    if len(matches) != 1:
        raise EvidenceV2Error("target v4 sample is missing or ambiguous in the plan")
    return dict(matches[0])


def _read_source_artifact(
    artifact: BoundCanonicalArtifact | BoundRawArtifact,
) -> EvidenceRef:
    if isinstance(artifact, BoundCanonicalArtifact):
        artifact.read()
    elif not isinstance(artifact, BoundRawArtifact):
        raise EvidenceV2Error("cost source has the wrong bound artifact type")
    return artifact.reference


def _cost_blocker(sample_id: str) -> str:
    return (
        "calibration_cost_requirement_unsupported:"
        f"sample={sample_id}:requirement={EIGHT_COMPONENT_COST_REQUIREMENT}"
    )


def _target_blocker(sample_id: str) -> str:
    return (
        "calibration_target_outcome_blocked:"
        f"sample={sample_id}:dependency={EIGHT_COMPONENT_COST_REQUIREMENT}"
    )


def build_stock_source_set_v3(
    *,
    protocol_attempt_id: str,
    source_refs: Mapping[str, EvidenceRef],
) -> dict[str, Any]:
    if not isinstance(source_refs, Mapping) or list(source_refs) != list(STOCK_SOURCE_KEYS):
        raise EvidenceV2Error("stock source-set keys/order mismatch")
    normalized: dict[str, dict[str, str]] = {}
    for key in STOCK_SOURCE_KEYS:
        reference = source_refs[key]
        if (
            not isinstance(reference, EvidenceRef)
            or reference.artifact_schema != STOCK_SOURCE_SCHEMAS[key]
            or reference.root_policy != STOCK_SOURCE_ROOT_POLICIES[key]
        ):
            raise EvidenceV2Error(f"stock source-set ref is invalid: {key}")
        normalized[key] = reference.to_dict()
    if len({item["byte_sha256"] for item in normalized.values()}) != len(normalized):
        raise EvidenceV2Error("stock source-set refs must have distinct byte identities")
    return seal_semantic(
        {
            "schema_version": STOCK_SOURCE_SET_SCHEMA,
            "protocol_attempt_id": _identifier(
                protocol_attempt_id,
                label="protocol_attempt_id",
            ),
            "source_refs": normalized,
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_stock_source_set_v3(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_attempt_id",
        "source_refs",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    payload = _exact(payload, fields, label="stock source-set v3")
    if payload["schema_version"] != STOCK_SOURCE_SET_SCHEMA or not isinstance(
        payload["source_refs"], Mapping
    ):
        raise EvidenceV2Error("stock source-set v3 schema/refs mismatch")
    if set(payload["source_refs"]) != set(STOCK_SOURCE_KEYS):
        raise EvidenceV2Error("stock source-set v3 ref keys mismatch")
    rebuilt = build_stock_source_set_v3(
        protocol_attempt_id=str(payload["protocol_attempt_id"]),
        source_refs={
            key: EvidenceRef.from_dict(payload["source_refs"][key])
            for key in STOCK_SOURCE_KEYS
        },
    )
    if rebuilt != payload:
        raise EvidenceV2Error("stock source-set v3 is not canonical")
    return payload


def build_cost_source_status_v3(
    *,
    plan: BoundCanonicalArtifact,
    sample_id: str,
    source_artifacts: Sequence[BoundCanonicalArtifact | BoundRawArtifact],
) -> dict[str, Any]:
    plan_payload = _bound_plan(plan)
    sample = _sample(plan_payload, sample_id)
    expected = [EvidenceRef.from_dict(item) for item in sample["cost_source_refs"]]
    actual = [_read_source_artifact(item) for item in source_artifacts]
    if [item.to_dict() for item in actual] != [item.to_dict() for item in expected]:
        raise EvidenceV2Error("cost source artifacts drift from the pre-s0 plan")
    blocker = _cost_blocker(sample_id)
    return seal_semantic(
        {
            "schema_version": COST_STATUS_SCHEMA,
            "protocol_attempt_id": plan_payload["protocol_attempt_id"],
            "epoch": plan_payload["epoch"],
            "schedule_id": plan_payload["schedule_id"],
            "sample_id": sample["sample_id"],
            "symbol": sample["symbol"],
            "cost_source_refs": [item.to_dict() for item in actual],
            "unsupported_requirement_id": EIGHT_COMPONENT_COST_REQUIREMENT,
            "source_recomputation_complete": False,
            "blockers": [blocker],
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_cost_source_status_v3(
    value: Mapping[str, Any],
    *,
    plan: BoundCanonicalArtifact,
    source_artifacts: Sequence[BoundCanonicalArtifact | BoundRawArtifact],
) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_attempt_id",
        "epoch",
        "schedule_id",
        "sample_id",
        "symbol",
        "cost_source_refs",
        "unsupported_requirement_id",
        "source_recomputation_complete",
        "blockers",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    payload = _exact(payload, fields, label="cost source status v3")
    if payload["schema_version"] != COST_STATUS_SCHEMA:
        raise EvidenceV2Error("cost source status v3 schema mismatch")
    rebuilt = build_cost_source_status_v3(
        plan=plan,
        sample_id=str(payload["sample_id"]),
        source_artifacts=source_artifacts,
    )
    if rebuilt != payload:
        raise EvidenceV2Error("cost source status v3 drifts from bound sources")
    return payload


@dataclass(frozen=True)
class TargetSourceEvidenceBundleV4:
    plan: BoundCanonicalArtifact
    schedule_anchor: ScheduleAnchorBindingV4
    stock_marks: BoundCanonicalArtifact
    stock_source_set: BoundCanonicalArtifact
    stock_sources: StockMarkSourceBundle
    benchmark_manifest: BoundCanonicalArtifact
    benchmark_parquet: BoundRawArtifact
    cost_status: BoundCanonicalArtifact
    cost_source_artifacts: tuple[BoundCanonicalArtifact | BoundRawArtifact, ...]

    def read(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        plan = _bound_plan(self.plan)
        schedule = validate_schedule_anchor_binding_v4(self.schedule_anchor)
        if (
            schedule["protocol_attempt_id"] != plan["protocol_attempt_id"]
            or schedule["epoch"] != plan["epoch"]
            or schedule["schedule_id"] != plan["schedule_id"]
            or schedule["calibration_plan_ref"] != self.plan.reference.to_dict()
        ):
            raise EvidenceV2Error("target v4 schedule/plan lineage mismatch")
        stock_payload = self.stock_marks.read()
        sample = _sample(plan, str(stock_payload.get("sample_id") or ""))
        if not _matches_planned_reference(
            self.stock_marks.reference,
            sample["artifacts"]["stock_marks"],
        ) or not _matches_planned_reference(
            self.cost_status.reference,
            sample["artifacts"]["cost_status"],
        ):
            raise EvidenceV2Error("target v4 future artifact paths drift from plan")
        slots = [item for item in schedule["slots"] if item["slot_id"] == sample["slot_id"]]
        if len(slots) != 1:
            raise EvidenceV2Error("target v4 schedule slot is missing or ambiguous")
        slot = slots[0]
        if (
            sample["cohort_start_date"] != slot["target_sessions"][0]
            or sample["cohort_end_date"] != slot["target_sessions"][-1]
        ):
            raise EvidenceV2Error("target v4 cohort window drifts from schedule")
        if (
            self.stock_source_set.reference.to_dict() != sample["stock_source_set_ref"]
            or self.stock_source_set.reference.artifact_schema != STOCK_SOURCE_SET_SCHEMA
        ):
            raise EvidenceV2Error("target v4 stock source-set ref drifts from plan")
        source_set = validate_stock_source_set_v3(self.stock_source_set.read())
        if source_set["protocol_attempt_id"] != plan["protocol_attempt_id"]:
            raise EvidenceV2Error("target v4 stock source-set attempt drifts from plan")
        actual_stock_refs = {
            "market_parquet": self.stock_sources.market_parquet.reference,
            "adjustment_factors": self.stock_sources.adjustment_factors.reference,
            "pit_membership": self.stock_sources.pit_membership.reference,
            "suspensions": self.stock_sources.suspensions.reference,
        }
        if source_set["source_refs"] != {
            key: actual_stock_refs[key].to_dict() for key in STOCK_SOURCE_KEYS
        }:
            raise EvidenceV2Error("target v4 stock source refs drift from source-set")
        stock_sources = prepare_stock_mark_sources(self.stock_sources)
        stock, _entry, _exit = validate_stock_mark_evidence_from_sources(
            stock_payload,
            sources=stock_sources,
            entry_date=sample["cohort_start_date"],
            exit_date=sample["cohort_end_date"],
        )
        if (
            stock["protocol_attempt_id"] != plan["protocol_attempt_id"]
            or stock["symbol"] != sample["symbol"]
            or stock["slot_id"] != sample["slot_id"]
            or stock["schedule_ref"] != self.schedule_anchor.evidence.schedule.reference.to_dict()
            or stock_sources.calendar_ref.to_dict() != schedule["open_session_calendar"]
        ):
            raise EvidenceV2Error("target v4 stock boundary lineage mismatch")
        if (
            self.benchmark_manifest.reference.to_dict()
            != sample["benchmark_manifest_ref"]
            or self.benchmark_manifest.reference.artifact_schema != INDEX_MANIFEST_SCHEMA
            or self.benchmark_parquet.reference.artifact_schema != INDEX_TABLE_SCHEMA
        ):
            raise EvidenceV2Error("target v4 benchmark refs drift from plan")
        manifest, table = validate_h00300_manifest_with_parquet(
            self.benchmark_manifest.read(),
            parquet_payload=self.benchmark_parquet.payload,
        )
        if (
            manifest["table_ref"] != self.benchmark_parquet.reference.to_dict()
            or manifest["calendar_ref"] != schedule["open_session_calendar"]
        ):
            raise EvidenceV2Error("target v4 benchmark manifest lineage drift")
        benchmark_dates = {item["trade_date"] for item in table["rows"]}
        missing = [item for item in slot["target_sessions"] if item not in benchmark_dates]
        if missing:
            raise EvidenceV2Error("target v4 benchmark lacks exact target-session rows")
        cost = validate_cost_source_status_v3(
            self.cost_status.read(),
            plan=self.plan,
            source_artifacts=self.cost_source_artifacts,
        )
        if cost["sample_id"] != sample["sample_id"]:
            raise EvidenceV2Error("target v4 cost status sample drift")
        return plan, sample, stock


def build_target_source_status_v3(
    *,
    evidence: TargetSourceEvidenceBundleV4,
) -> dict[str, Any]:
    if not isinstance(evidence, TargetSourceEvidenceBundleV4):
        raise EvidenceV2Error("target source status requires TargetSourceEvidenceBundleV4")
    plan, sample, _stock = evidence.read()
    blocker = _target_blocker(sample["sample_id"])
    return seal_semantic(
        {
            "schema_version": TARGET_STATUS_SCHEMA,
            "protocol_attempt_id": plan["protocol_attempt_id"],
            "epoch": plan["epoch"],
            "schedule_id": plan["schedule_id"],
            "slot_id": sample["slot_id"],
            "sample_id": sample["sample_id"],
            "symbol": sample["symbol"],
            "schedule_ref": evidence.schedule_anchor.evidence.schedule.reference.to_dict(),
            "stock_marks_ref": evidence.stock_marks.reference.to_dict(),
            "stock_source_refs": [
                evidence.stock_sources.market_parquet.reference.to_dict(),
                evidence.stock_sources.adjustment_factors.reference.to_dict(),
                evidence.stock_sources.pit_membership.reference.to_dict(),
                evidence.stock_sources.suspensions.reference.to_dict(),
            ],
            "benchmark_manifest_ref": evidence.benchmark_manifest.reference.to_dict(),
            "benchmark_parquet_ref": evidence.benchmark_parquet.reference.to_dict(),
            "cost_status_ref": evidence.cost_status.reference.to_dict(),
            "source_recomputation_complete": False,
            "blockers": [blocker],
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_target_source_status_v3(
    value: Mapping[str, Any],
    *,
    evidence: TargetSourceEvidenceBundleV4,
) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_attempt_id",
        "epoch",
        "schedule_id",
        "slot_id",
        "sample_id",
        "symbol",
        "schedule_ref",
        "stock_marks_ref",
        "stock_source_refs",
        "benchmark_manifest_ref",
        "benchmark_parquet_ref",
        "cost_status_ref",
        "source_recomputation_complete",
        "blockers",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    payload = _exact(payload, fields, label="target source status v3")
    if payload["schema_version"] != TARGET_STATUS_SCHEMA:
        raise EvidenceV2Error("target source status v3 schema mismatch")
    rebuilt = build_target_source_status_v3(evidence=evidence)
    if rebuilt != payload:
        raise EvidenceV2Error("target source status v3 drifts from bound sources")
    return payload


__all__ = [
    "EIGHT_COMPONENT_COST_REQUIREMENT",
    "TargetSourceEvidenceBundleV4",
    "build_cost_source_status_v3",
    "build_stock_source_set_v3",
    "build_target_source_status_v3",
    "validate_cost_source_status_v3",
    "validate_stock_source_set_v3",
    "validate_target_source_status_v3",
]
