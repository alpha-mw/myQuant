"""Schedule-v4 binding for the calibration source plan v3."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

from .calendar import OPEN_SESSION_CALENDAR_SCHEMA
from .calendar_recheck import CALENDAR_RECHECK_SCHEMA, CalendarRecheckEvidenceBundle
from .calibration_plan_v3 import (
    CALIBRATION_PLAN_V3_SCHEMA,
    CalibrationPlanEvidenceBundleV3,
    SCHEDULE_V4_SCHEMA,
)
from .contracts import (
    BoundCanonicalArtifact,
    EvidenceRef,
    EvidenceV2Error,
    nonauthorizing_projection,
    seal_semantic,
    validate_semantic_seal,
)
from .runtime_identity import MODEL_BUNDLE_SCHEMA, RUNTIME_CAPSULE_SCHEMA
from .schedule import (
    ATTEMPT_GENESIS_V3_SCHEMA,
    EPOCH_ORDER,
    MODEL_BRANCHES,
    PRIVATE_ROOT_POLICY,
    AttemptGenesisEvidenceBundleV3,
    FrozenModelEvidenceBundle,
    _exact,
    _id,
    _reference,
    _validate_slots_v3,
)
from .session_clock import SESSION_CLOCK_SCHEMA
from .timestamp import TimestampAnchorBinding

LINEAGE_V4_BLOCKERS = (
    "calendar_recheck_capture_time_not_independently_evidenced",
    "calendar_recheck_transport_freshness_not_independently_attested",
    "evidence_v2_disconnected_from_authorizing_consumers",
    "global_attempt_registry_authority_not_integrated",
    "provisional_journal_head_not_bound_to_external_anti_rollback_authority",
)


def build_schedule_declaration_v4(
    *,
    protocol_attempt_id: str,
    epoch: str,
    schedule_id: str,
    seed_hex: str,
    genesis_ref: EvidenceRef,
    runtime_capsule: EvidenceRef,
    open_session_calendar: EvidenceRef,
    session_clock: EvidenceRef,
    calendar_recheck_ref: EvidenceRef,
    model_bundle_refs: Mapping[str, EvidenceRef] | None,
    calibration_plan_ref: EvidenceRef | None,
    slots: Sequence[Mapping[str, Any]],
    calendar: Mapping[str, Any],
    session_clock_value: Mapping[str, Any],
) -> dict[str, Any]:
    epoch_name = str(epoch)
    if epoch_name not in EPOCH_ORDER:
        raise EvidenceV2Error("schedule v4 epoch must be A, B, or C")
    seed = str(seed_hex)
    if len(seed) != 64 or any(character not in "0123456789abcdef" for character in seed):
        raise EvidenceV2Error("schedule v4 seed must be exactly 32 lowercase-hex bytes")
    for reference, schema, label in (
        (genesis_ref, ATTEMPT_GENESIS_V3_SCHEMA, "genesis_ref"),
        (open_session_calendar, OPEN_SESSION_CALENDAR_SCHEMA, "open_session_calendar"),
        (session_clock, SESSION_CLOCK_SCHEMA, "session_clock"),
        (calendar_recheck_ref, CALENDAR_RECHECK_SCHEMA, "calendar_recheck_ref"),
    ):
        if reference.artifact_schema != schema or reference.root_policy != PRIVATE_ROOT_POLICY:
            raise EvidenceV2Error(f"schedule v4 {label} is invalid")
    if (
        runtime_capsule.artifact_schema != RUNTIME_CAPSULE_SCHEMA
        or runtime_capsule.root_policy != PRIVATE_ROOT_POLICY
    ):
        raise EvidenceV2Error("schedule v4 runtime capsule ref is invalid")
    if epoch_name == "A":
        if model_bundle_refs is not None or calibration_plan_ref is not None:
            raise EvidenceV2Error("epoch A must not bind model or calibration evidence")
    else:
        if not isinstance(model_bundle_refs, Mapping) or list(model_bundle_refs) != list(
            MODEL_BRANCHES
        ):
            raise EvidenceV2Error("epoch B/C must bind ordered four-branch model refs")
        if any(
            reference.artifact_schema != MODEL_BUNDLE_SCHEMA
            or reference.root_policy != PRIVATE_ROOT_POLICY
            for reference in model_bundle_refs.values()
        ):
            raise EvidenceV2Error("epoch B/C model refs are not frozen private bundles")
        if (
            calibration_plan_ref is None
            or calibration_plan_ref.artifact_schema != CALIBRATION_PLAN_V3_SCHEMA
            or calibration_plan_ref.root_policy != PRIVATE_ROOT_POLICY
        ):
            raise EvidenceV2Error("epoch B/C calibration plan v3 ref is invalid")
    normalized_slots = _validate_slots_v3(
        list(slots),
        epoch=epoch_name,
        calendar=calendar,
        session_clock=session_clock_value,
    )
    return seal_semantic(
        {
            "schema_version": SCHEDULE_V4_SCHEMA,
            "protocol_attempt_id": _id(protocol_attempt_id, label="protocol_attempt_id"),
            "epoch": epoch_name,
            "schedule_id": _id(schedule_id, label="schedule_id"),
            "seed_hex": seed,
            "genesis_ref": genesis_ref.to_dict(),
            "runtime_capsule": runtime_capsule.to_dict(),
            "open_session_calendar": open_session_calendar.to_dict(),
            "session_clock": session_clock.to_dict(),
            "calendar_recheck_ref": calendar_recheck_ref.to_dict(),
            "model_bundle_refs": (
                None
                if model_bundle_refs is None
                else {branch: model_bundle_refs[branch].to_dict() for branch in MODEL_BRANCHES}
            ),
            "calibration_plan_ref": (
                None if calibration_plan_ref is None else calibration_plan_ref.to_dict()
            ),
            "slots": normalized_slots,
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_schedule_declaration_v4(
    value: Mapping[str, Any],
    *,
    calendar: Mapping[str, Any],
    session_clock_value: Mapping[str, Any],
) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_attempt_id",
        "epoch",
        "schedule_id",
        "seed_hex",
        "genesis_ref",
        "runtime_capsule",
        "open_session_calendar",
        "session_clock",
        "calendar_recheck_ref",
        "model_bundle_refs",
        "calibration_plan_ref",
        "slots",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    payload = _exact(payload, fields, label="schedule declaration v4")
    if payload["schema_version"] != SCHEDULE_V4_SCHEMA:
        raise EvidenceV2Error("unsupported schedule declaration v4 schema")
    _id(payload["protocol_attempt_id"], label="protocol_attempt_id")
    _id(payload["schedule_id"], label="schedule_id")
    epoch = str(payload["epoch"])
    if epoch not in EPOCH_ORDER:
        raise EvidenceV2Error("schedule v4 epoch is invalid")
    seed = str(payload["seed_hex"])
    if len(seed) != 64 or any(character not in "0123456789abcdef" for character in seed):
        raise EvidenceV2Error("schedule v4 seed is invalid")
    refs = {
        "genesis_ref": (ATTEMPT_GENESIS_V3_SCHEMA, payload["genesis_ref"]),
        "open_session_calendar": (
            OPEN_SESSION_CALENDAR_SCHEMA,
            payload["open_session_calendar"],
        ),
        "session_clock": (SESSION_CLOCK_SCHEMA, payload["session_clock"]),
        "calendar_recheck_ref": (
            CALENDAR_RECHECK_SCHEMA,
            payload["calendar_recheck_ref"],
        ),
    }
    for label, (schema, raw) in refs.items():
        reference = EvidenceRef.from_dict(_reference(raw, label=label))
        if reference.artifact_schema != schema or reference.root_policy != PRIVATE_ROOT_POLICY:
            raise EvidenceV2Error(f"schedule v4 {label} drift")
        payload[label] = reference.to_dict()
    payload["runtime_capsule"] = _reference(
        payload["runtime_capsule"], label="runtime_capsule"
    )
    runtime_ref = EvidenceRef.from_dict(payload["runtime_capsule"])
    if (
        runtime_ref.artifact_schema != RUNTIME_CAPSULE_SCHEMA
        or runtime_ref.root_policy != PRIVATE_ROOT_POLICY
    ):
        raise EvidenceV2Error("schedule v4 runtime capsule ref drift")
    if epoch == "A":
        if payload["model_bundle_refs"] is not None:
            raise EvidenceV2Error("epoch A must not bind model bundles")
        if payload["calibration_plan_ref"] is not None:
            raise EvidenceV2Error("epoch A must not bind a calibration plan")
    else:
        model_refs = payload["model_bundle_refs"]
        if not isinstance(model_refs, Mapping) or list(model_refs) != list(MODEL_BRANCHES):
            raise EvidenceV2Error("epoch B/C model ref order drift")
        payload["model_bundle_refs"] = {
            branch: _reference(model_refs[branch], label=f"{branch} model bundle")
            for branch in MODEL_BRANCHES
        }
        if any(
            EvidenceRef.from_dict(reference).artifact_schema != MODEL_BUNDLE_SCHEMA
            or EvidenceRef.from_dict(reference).root_policy != PRIVATE_ROOT_POLICY
            for reference in payload["model_bundle_refs"].values()
        ):
            raise EvidenceV2Error("epoch B/C model refs are invalid")
        if payload["calibration_plan_ref"] is None:
            raise EvidenceV2Error("epoch B/C calibration plan is missing")
        payload["calibration_plan_ref"] = _reference(
            payload["calibration_plan_ref"], label="calibration_plan_ref"
        )
        plan_ref = EvidenceRef.from_dict(payload["calibration_plan_ref"])
        if (
            plan_ref.artifact_schema != CALIBRATION_PLAN_V3_SCHEMA
            or plan_ref.root_policy != PRIVATE_ROOT_POLICY
        ):
            raise EvidenceV2Error("epoch B/C calibration plan ref is invalid")
    payload["slots"] = _validate_slots_v3(
        payload["slots"],
        epoch=epoch,
        calendar=calendar,
        session_clock=session_clock_value,
    )
    if any(
        payload[field] is not False
        for field in (
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
        )
    ):
        raise EvidenceV2Error("schedule v4 must be permanently nonauthorizing")
    return payload


@dataclass(frozen=True)
class ScheduleEvidenceBundleV4:
    schedule: BoundCanonicalArtifact
    genesis: AttemptGenesisEvidenceBundleV3
    calendar_recheck: CalendarRecheckEvidenceBundle
    model_bundles: tuple[FrozenModelEvidenceBundle, ...]
    calibration_plan: CalibrationPlanEvidenceBundleV3 | None

    def read(self) -> dict[str, Any]:
        if (
            self.schedule.reference.artifact_schema != SCHEDULE_V4_SCHEMA
            or self.schedule.reference.root_policy != PRIVATE_ROOT_POLICY
        ):
            raise EvidenceV2Error("schedule evidence v4 ref is invalid")
        genesis = self.genesis.read()
        calendar = self.genesis.calendar.read()
        clock = self.genesis.session_clock.read()
        recheck = self.calendar_recheck.read()
        schedule = validate_schedule_declaration_v4(
            self.schedule.read(),
            calendar=calendar,
            session_clock_value=clock,
        )
        if (
            schedule["genesis_ref"] != self.genesis.genesis.reference.to_dict()
            or schedule["runtime_capsule"] != genesis["runtime_capsule"]
            or schedule["open_session_calendar"] != genesis["open_session_calendar"]
            or schedule["session_clock"] != genesis["session_clock"]
            or schedule["calendar_recheck_ref"]
            != self.calendar_recheck.recheck.reference.to_dict()
        ):
            raise EvidenceV2Error("schedule evidence v4 genesis/recheck refs drift")
        if (
            recheck["protocol_attempt_id"] != schedule["protocol_attempt_id"]
            or recheck["epoch"] != schedule["epoch"]
            or recheck["schedule_id"] != schedule["schedule_id"]
            or recheck["first_s0_open_at"]
            != min(slot["s0_open_at"] for slot in schedule["slots"])
        ):
            raise EvidenceV2Error("schedule v4 calendar recheck lineage drift")
        if schedule["protocol_attempt_id"] != genesis["protocol_attempt_id"]:
            raise EvidenceV2Error("schedule evidence v4 protocol attempt drift")
        if schedule["epoch"] == "A":
            if self.model_bundles or self.calibration_plan is not None:
                raise EvidenceV2Error("epoch A schedule v4 carries model evidence")
        else:
            model_payloads = [bundle.read() for bundle in self.model_bundles]
            if [payload["branch"] for payload in model_payloads] != list(MODEL_BRANCHES):
                raise EvidenceV2Error("schedule v4 model evidence order drift")
            model_refs = {
                branch: bundle.model_bundle.reference.to_dict()
                for branch, bundle in zip(MODEL_BRANCHES, self.model_bundles)
            }
            if schedule["model_bundle_refs"] != model_refs:
                raise EvidenceV2Error("schedule v4 model evidence refs drift")
            if any(
                payload["protocol_attempt_id"] != schedule["protocol_attempt_id"]
                for payload in model_payloads
            ):
                raise EvidenceV2Error("schedule v4 model evidence attempt drift")
            if self.calibration_plan is None:
                raise EvidenceV2Error("epoch B/C schedule v4 lacks calibration plan")
            plan = self.calibration_plan.read()
            if (
                schedule["calibration_plan_ref"]
                != self.calibration_plan.plan.reference.to_dict()
                or plan["protocol_attempt_id"] != schedule["protocol_attempt_id"]
                or plan["epoch"] != schedule["epoch"]
                or plan["schedule_id"] != schedule["schedule_id"]
                or plan["runtime_capsule_ref"] != schedule["runtime_capsule"]
                or plan["model_bundle_refs"] != model_refs
            ):
                raise EvidenceV2Error("schedule v4 calibration-plan lineage drift")
        return schedule

    def model_evidence_identity(self) -> tuple[tuple[tuple[str, ...], ...], ...]:
        self.read()
        return tuple(bundle.evidence_identity() for bundle in self.model_bundles)


@dataclass(frozen=True)
class ScheduleAnchorBindingV4:
    evidence: ScheduleEvidenceBundleV4
    timestamp: TimestampAnchorBinding


def validate_schedule_anchor_binding_v4(
    binding: ScheduleAnchorBindingV4,
) -> dict[str, Any]:
    if not isinstance(binding, ScheduleAnchorBindingV4):
        raise EvidenceV2Error("schedule anchor binding v4 has the wrong type")
    schedule = binding.evidence.read()
    attempt, receipt = binding.timestamp.read()
    if any(
        artifact.reference.root_policy != PRIVATE_ROOT_POLICY
        for artifact in (
            binding.evidence.schedule,
            binding.timestamp.attempt,
            binding.timestamp.validation_receipt,
        )
    ):
        raise EvidenceV2Error("schedule v4 anchor artifacts must use the private root")
    first_s0_open = min(slot["s0_open_at"] for slot in schedule["slots"])
    if (
        receipt["anchored_artifact_ref"] != binding.evidence.schedule.reference.to_dict()
        or receipt["anchor_kind"] != "schedule_declaration"
        or receipt["anchor_not_before"] is not None
        or receipt["anchor_not_after"] != first_s0_open
        or attempt["protocol_attempt_id"] != schedule["protocol_attempt_id"]
    ):
        raise EvidenceV2Error("schedule v4 RFC3161 pre-s0 anchor lineage mismatch")
    return schedule


def validate_bound_lineage_v4(
    *,
    genesis: AttemptGenesisEvidenceBundleV3,
    schedule_anchors: Sequence[ScheduleAnchorBindingV4],
) -> dict[str, Any]:
    normalized_genesis = genesis.read()
    normalized_schedules = [
        validate_schedule_anchor_binding_v4(item) for item in schedule_anchors
    ]
    epochs = [item["epoch"] for item in normalized_schedules]
    if epochs != list(EPOCH_ORDER[: len(epochs)]) or len(epochs) > len(EPOCH_ORDER):
        raise EvidenceV2Error("bound lineage v4 must contain one ordered A/B/C prefix")
    attempt_id = normalized_genesis["protocol_attempt_id"]
    for item, binding in zip(normalized_schedules, schedule_anchors):
        if (
            item["protocol_attempt_id"] != attempt_id
            or item["genesis_ref"] != genesis.genesis.reference.to_dict()
            or item["runtime_capsule"] != normalized_genesis["runtime_capsule"]
            or item["open_session_calendar"]
            != normalized_genesis["open_session_calendar"]
            or item["session_clock"] != normalized_genesis["session_clock"]
            or binding.evidence.genesis.genesis.reference != genesis.genesis.reference
        ):
            raise EvidenceV2Error("bound lineage v4 immutable genesis identity drift")
    schedule_ids = [item["schedule_id"] for item in normalized_schedules]
    if len(schedule_ids) != len(set(schedule_ids)):
        raise EvidenceV2Error("bound lineage v4 schedule IDs must be unique")
    for previous, current in zip(normalized_schedules, normalized_schedules[1:]):
        previous_end = max(
            session for slot in previous["slots"] for session in slot["target_sessions"]
        )
        current_start = min(slot["s0_date"] for slot in current["slots"])
        if current_start <= previous_end:
            raise EvidenceV2Error("bound lineage v4 epoch windows overlap")
    frozen = [
        binding.evidence.model_evidence_identity()
        for binding in schedule_anchors
        if binding.evidence.read()["epoch"] in {"B", "C"}
    ]
    if len(frozen) == 2 and frozen[0] != frozen[1]:
        raise EvidenceV2Error("epoch C full model evidence drifts from frozen epoch B")
    projection = nonauthorizing_projection(blockers=list(LINEAGE_V4_BLOCKERS))
    if projection["blockers"] != list(LINEAGE_V4_BLOCKERS):
        raise EvidenceV2Error("bound lineage v4 blocker set drift")
    return projection


__all__ = [
    "LINEAGE_V4_BLOCKERS",
    "ScheduleAnchorBindingV4",
    "ScheduleEvidenceBundleV4",
    "build_schedule_declaration_v4",
    "validate_bound_lineage_v4",
    "validate_schedule_anchor_binding_v4",
    "validate_schedule_declaration_v4",
]
