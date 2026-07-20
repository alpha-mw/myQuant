"""Immutable one-attempt schedule and lineage contracts for evidence-v2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from collections.abc import Mapping, Sequence
import os
from pathlib import Path
from typing import Any

from .contracts import (
    BoundCanonicalArtifact,
    BoundRawArtifact,
    EvidenceRef,
    EvidenceV2Error,
    nonauthorizing_projection,
    seal_semantic,
    validate_semantic_seal,
)
from .timestamp import TimestampAnchorBinding
from .runtime_identity import MODEL_BUNDLE_SCHEMA
from .calendar import (
    CalendarEvidenceBundle,
    OPEN_SESSION_CALENDAR_SCHEMA,
    PRIVATE_ROOT_POLICY as CALENDAR_PRIVATE_ROOT_POLICY,
    validate_open_session_calendar,
)
from .calendar_recheck import CALENDAR_RECHECK_SCHEMA
from .calendar_recheck import CalendarRecheckEvidenceBundle
from .session_clock import (
    SessionClockEvidenceBundle,
    SESSION_CLOCK_SCHEMA,
    validate_session_clock,
)
from .runtime_identity import (
    RUNTIME_CAPSULE_SCHEMA,
    RUNTIME_COMPONENT_ORDER,
    validate_frozen_model_bundle,
    validate_runtime_capsule,
)
from .secure_io import (
    RootPolicy,
    load_bound_canonical_artifact,
    load_bound_raw_artifact,
)

ATTEMPT_GENESIS_SCHEMA = "v16.evidence-attempt-genesis.v2"
SCHEDULE_DECLARATION_SCHEMA = "v16.evidence-schedule-declaration.v2"
LINEAGE_EVENT_SCHEMA = "v16.evidence-lineage-event.v2"
TRANSITION_GRAPH_SCHEMA = "factor-v4.transition-graph.v2"
EPOCH_ORDER = ("A", "B", "C")
MODEL_BRANCHES = ("quant", "fundamental", "macro", "llm")
MAX_ATTEMPTS_V16 = 1
CALIBRATION_UNIVERSE_SCHEMA = "v16.calibration-universe-plan.v2"
LAMBDA_FOLD_SCHEMA = "v16.lambda-fold-evidence.v2"
PRIVATE_ROOT_POLICY = "v16.private-evidence-root.v2"
ATTEMPT_GENESIS_V3_SCHEMA = "v16.evidence-attempt-genesis.v3"
SCHEDULE_DECLARATION_V3_SCHEMA = "v16.evidence-schedule-declaration.v3"

if PRIVATE_ROOT_POLICY != CALENDAR_PRIVATE_ROOT_POLICY:
    raise RuntimeError("evidence-v2 private root policy constants drift")


def _exact(value: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise EvidenceV2Error(f"{label} fields mismatch")
    return dict(value)


def _id(value: Any, *, label: str) -> str:
    text = str(value or "")
    if (
        not text
        or text != text.strip()
        or len(text) > 128
        or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in text
        )
    ):
        raise EvidenceV2Error(f"{label} is not a safe identifier")
    return text


def _utc(value: Any, *, label: str) -> datetime:
    text = str(value or "")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceV2Error(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise EvidenceV2Error(f"{label} must be UTC")
    canonical = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if canonical != text:
        raise EvidenceV2Error(f"{label} must use canonical UTC form")
    return parsed


def _date(value: Any, *, label: str) -> str:
    text = str(value or "")
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise EvidenceV2Error(f"{label} must be ISO date") from exc
    if parsed.isoformat() != text:
        raise EvidenceV2Error(f"{label} must be canonical ISO date")
    return text


def _reference(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise EvidenceV2Error(f"{label} must be an EvidenceRef")
    return EvidenceRef.from_dict(value).to_dict()


def validate_transition_graph(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    expected = {
        "schema_version",
        "protocol_attempt_id",
        "transitions",
        "semantic_sha256",
    }
    payload = _exact(payload, expected, label="transition graph")
    if payload["schema_version"] != TRANSITION_GRAPH_SCHEMA:
        raise EvidenceV2Error("unsupported transition graph schema")
    _id(payload["protocol_attempt_id"], label="protocol_attempt_id")
    transitions = payload["transitions"]
    if not isinstance(transitions, list) or not transitions:
        raise EvidenceV2Error("transition graph must be nonempty")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    fields = {
        "transition_id",
        "mode",
        "incumbent",
        "challenger",
        "arm_factor_sets",
    }
    for index, raw in enumerate(transitions):
        row = _exact(raw, fields, label=f"transitions[{index}]")
        transition_id = _id(row["transition_id"], label="transition_id")
        if transition_id in seen:
            raise EvidenceV2Error("transition IDs must be unique")
        seen.add(transition_id)
        mode = str(row["mode"])
        if mode not in {"add", "replace"}:
            raise EvidenceV2Error("transition mode must be add or replace")
        incumbent = row["incumbent"]
        if mode == "add" and incumbent is not None:
            raise EvidenceV2Error("add transition incumbent must be null")
        if mode == "replace":
            incumbent = _id(incumbent, label="incumbent")
        challenger = _id(row["challenger"], label="challenger")
        arms = row["arm_factor_sets"]
        if not isinstance(arms, Mapping) or list(arms) != ["A", "B", "C", "D"]:
            raise EvidenceV2Error("arm factor-set refs must preserve A/B/C/D order")
        normalized.append(
            {
                "transition_id": transition_id,
                "mode": mode,
                "incumbent": incumbent,
                "challenger": challenger,
                "arm_factor_sets": {
                    arm: _reference(arms[arm], label=f"arm {arm} factor set")
                    for arm in ("A", "B", "C", "D")
                },
            }
        )
    payload["transitions"] = normalized
    return payload


def build_attempt_genesis(
    *,
    protocol_attempt_id: str,
    runtime_capsule: EvidenceRef,
    proposed_factor_graph: EvidenceRef,
    open_session_calendar: EvidenceRef,
) -> dict[str, Any]:
    return seal_semantic(
        {
            "schema_version": ATTEMPT_GENESIS_SCHEMA,
            "protocol_version": "v16",
            "protocol_attempt_id": _id(protocol_attempt_id, label="protocol_attempt_id"),
            "max_attempts_v16": MAX_ATTEMPTS_V16,
            "epoch_order": list(EPOCH_ORDER),
            "runtime_capsule": runtime_capsule.to_dict(),
            "proposed_factor_graph": proposed_factor_graph.to_dict(),
            "open_session_calendar": open_session_calendar.to_dict(),
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_attempt_genesis(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    expected = {
        "schema_version",
        "protocol_version",
        "protocol_attempt_id",
        "max_attempts_v16",
        "epoch_order",
        "runtime_capsule",
        "proposed_factor_graph",
        "open_session_calendar",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    payload = _exact(payload, expected, label="attempt genesis")
    if (
        payload["schema_version"] != ATTEMPT_GENESIS_SCHEMA
        or payload["protocol_version"] != "v16"
        or payload["max_attempts_v16"] != MAX_ATTEMPTS_V16
        or payload["epoch_order"] != list(EPOCH_ORDER)
    ):
        raise EvidenceV2Error("attempt genesis protocol envelope mismatch")
    _id(payload["protocol_attempt_id"], label="protocol_attempt_id")
    _reference(payload["runtime_capsule"], label="runtime_capsule")
    _reference(payload["proposed_factor_graph"], label="proposed_factor_graph")
    _reference(payload["open_session_calendar"], label="open_session_calendar")
    if any(
        payload[field] is not False
        for field in (
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
        )
    ):
        raise EvidenceV2Error("attempt genesis must be permanently nonauthorizing")
    return payload


def build_schedule_declaration(
    *,
    protocol_attempt_id: str,
    epoch: str,
    schedule_id: str,
    seed_hex: str,
    runtime_capsule: EvidenceRef,
    open_session_calendar: EvidenceRef,
    model_bundle_refs: Mapping[str, EvidenceRef] | None,
    calibration_universe_ref: EvidenceRef | None,
    slots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    epoch_name = str(epoch)
    if epoch_name not in EPOCH_ORDER:
        raise EvidenceV2Error("epoch must be A, B, or C")
    seed = str(seed_hex)
    if len(seed) != 64 or any(character not in "0123456789abcdef" for character in seed):
        raise EvidenceV2Error("schedule seed must be exactly 32 lowercase-hex bytes")
    if epoch_name == "A":
        if model_bundle_refs is not None:
            raise EvidenceV2Error("epoch A must not bind trained model bundles")
        if calibration_universe_ref is not None:
            raise EvidenceV2Error("epoch A must not bind a B/C calibration universe")
    elif not isinstance(model_bundle_refs, Mapping) or set(model_bundle_refs) != set(
        MODEL_BRANCHES
    ):
        raise EvidenceV2Error("epoch B/C must bind exactly four model bundles")
    elif calibration_universe_ref is None:
        raise EvidenceV2Error("epoch B/C must bind a predeclared calibration universe")
    elif any(
        reference.artifact_schema != MODEL_BUNDLE_SCHEMA
        or reference.root_policy != PRIVATE_ROOT_POLICY
        for reference in model_bundle_refs.values()
    ):
        raise EvidenceV2Error("epoch B/C model refs are not frozen private bundles")
    elif (
        calibration_universe_ref.artifact_schema != CALIBRATION_UNIVERSE_SCHEMA
        or calibration_universe_ref.root_policy != PRIVATE_ROOT_POLICY
    ):
        raise EvidenceV2Error("epoch B/C calibration universe ref is invalid")
    normalized_slots = _validate_slots(list(slots), epoch=epoch_name)
    return seal_semantic(
        {
            "schema_version": SCHEDULE_DECLARATION_SCHEMA,
            "protocol_attempt_id": _id(protocol_attempt_id, label="protocol_attempt_id"),
            "epoch": epoch_name,
            "schedule_id": _id(schedule_id, label="schedule_id"),
            "seed_hex": seed,
            "runtime_capsule": runtime_capsule.to_dict(),
            "open_session_calendar": open_session_calendar.to_dict(),
            "model_bundle_refs": (
                None
                if model_bundle_refs is None
                else {branch: model_bundle_refs[branch].to_dict() for branch in MODEL_BRANCHES}
            ),
            "calibration_universe_ref": (
                None if calibration_universe_ref is None else calibration_universe_ref.to_dict()
            ),
            "slots": normalized_slots,
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def _validate_slots(slots: list[Any], *, epoch: str) -> list[dict[str, Any]]:
    if not slots:
        raise EvidenceV2Error("schedule must contain at least one slot")
    expected_window = 30 if epoch == "B" else 20
    fields = {
        "slot_id",
        "s0_date",
        "s0_open_at",
        "s0_close_at",
        "decision_cutoff_at",
        "s1_open_at",
        "target_sessions",
    }
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_sessions: set[str] = set()
    prior_s0: str | None = None
    for index, raw in enumerate(slots):
        row = _exact(raw, fields, label=f"slots[{index}]")
        slot_id = _id(row["slot_id"], label="slot_id")
        if slot_id in seen_ids:
            raise EvidenceV2Error("schedule slot IDs must be unique")
        seen_ids.add(slot_id)
        s0_date = _date(row["s0_date"], label="s0_date")
        s0_open = _utc(row["s0_open_at"], label="s0_open_at")
        s0_close = _utc(row["s0_close_at"], label="s0_close_at")
        cutoff = _utc(row["decision_cutoff_at"], label="decision_cutoff_at")
        s1_open = _utc(row["s1_open_at"], label="s1_open_at")
        if not s0_open < s0_close < cutoff < s1_open:
            raise EvidenceV2Error("slot timestamp chronology is invalid")
        if s0_open.date().isoformat() != s0_date:
            raise EvidenceV2Error("s0_date does not match s0_open_at")
        if prior_s0 is not None and s0_date <= prior_s0:
            raise EvidenceV2Error("schedule slots must be ordered by s0_date")
        prior_s0 = s0_date
        sessions_raw = row["target_sessions"]
        if not isinstance(sessions_raw, list) or len(sessions_raw) != expected_window:
            raise EvidenceV2Error(
                f"epoch {epoch} target window must contain {expected_window} sessions"
            )
        sessions = [_date(item, label="target_session") for item in sessions_raw]
        if sessions != sorted(sessions) or len(sessions) != len(set(sessions)):
            raise EvidenceV2Error("target sessions must be sorted and distinct")
        if seen_sessions.intersection(sessions):
            raise EvidenceV2Error("schedule target windows must not overlap")
        seen_sessions.update(sessions)
        if s1_open.date().isoformat() != sessions[0]:
            raise EvidenceV2Error("s1_open_at must match first target session")
        normalized.append(
            {
                "slot_id": slot_id,
                "s0_date": s0_date,
                "s0_open_at": row["s0_open_at"],
                "s0_close_at": row["s0_close_at"],
                "decision_cutoff_at": row["decision_cutoff_at"],
                "s1_open_at": row["s1_open_at"],
                "target_sessions": sessions,
            }
        )
    return normalized


def validate_schedule_declaration(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    expected = {
        "schema_version",
        "protocol_attempt_id",
        "epoch",
        "schedule_id",
        "seed_hex",
        "runtime_capsule",
        "open_session_calendar",
        "model_bundle_refs",
        "calibration_universe_ref",
        "slots",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    payload = _exact(payload, expected, label="schedule declaration")
    if payload["schema_version"] != SCHEDULE_DECLARATION_SCHEMA:
        raise EvidenceV2Error("unsupported schedule declaration schema")
    _id(payload["protocol_attempt_id"], label="protocol_attempt_id")
    _id(payload["schedule_id"], label="schedule_id")
    epoch = str(payload["epoch"])
    if epoch not in EPOCH_ORDER:
        raise EvidenceV2Error("schedule epoch is invalid")
    seed = str(payload["seed_hex"])
    if len(seed) != 64 or any(character not in "0123456789abcdef" for character in seed):
        raise EvidenceV2Error("schedule seed is invalid")
    _reference(payload["runtime_capsule"], label="runtime_capsule")
    _reference(payload["open_session_calendar"], label="open_session_calendar")
    if epoch == "A":
        if payload["model_bundle_refs"] is not None:
            raise EvidenceV2Error("epoch A must not bind trained model bundles")
        if payload["calibration_universe_ref"] is not None:
            raise EvidenceV2Error("epoch A must not bind a B/C calibration universe")
    else:
        model_refs = payload["model_bundle_refs"]
        if not isinstance(model_refs, Mapping) or set(model_refs) != set(MODEL_BRANCHES):
            raise EvidenceV2Error("epoch B/C must bind exactly four model bundles")
        payload["model_bundle_refs"] = {
            branch: _reference(
                model_refs[branch],
                label=f"{branch} model bundle",
            )
            for branch in MODEL_BRANCHES
        }
        if any(
            EvidenceRef.from_dict(reference).artifact_schema != MODEL_BUNDLE_SCHEMA
            or EvidenceRef.from_dict(reference).root_policy != PRIVATE_ROOT_POLICY
            for reference in payload["model_bundle_refs"].values()
        ):
            raise EvidenceV2Error("epoch B/C model refs are not frozen private bundles")
        if payload["calibration_universe_ref"] is None:
            raise EvidenceV2Error("epoch B/C calibration universe is missing")
        payload["calibration_universe_ref"] = _reference(
            payload["calibration_universe_ref"],
            label="calibration_universe_ref",
        )
        universe_ref = EvidenceRef.from_dict(payload["calibration_universe_ref"])
        if (
            universe_ref.artifact_schema != CALIBRATION_UNIVERSE_SCHEMA
            or universe_ref.root_policy != PRIVATE_ROOT_POLICY
        ):
            raise EvidenceV2Error("epoch B/C calibration universe ref is invalid")
    payload["slots"] = _validate_slots(payload["slots"], epoch=epoch)
    if any(
        payload[field] is not False
        for field in (
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
        )
    ):
        raise EvidenceV2Error("schedule declaration must be nonauthorizing")
    return payload


def build_attempt_genesis_v3(
    *,
    protocol_attempt_id: str,
    runtime_capsule: EvidenceRef,
    proposed_factor_graph: EvidenceRef,
    open_session_calendar: EvidenceRef,
    session_clock: EvidenceRef,
) -> dict[str, Any]:
    if (
        runtime_capsule.artifact_schema != RUNTIME_CAPSULE_SCHEMA
        or runtime_capsule.root_policy != PRIVATE_ROOT_POLICY
        or proposed_factor_graph.artifact_schema != TRANSITION_GRAPH_SCHEMA
        or proposed_factor_graph.root_policy != PRIVATE_ROOT_POLICY
    ):
        raise EvidenceV2Error("genesis v3 runtime or factor-graph ref is invalid")
    for reference, schema, label in (
        (open_session_calendar, OPEN_SESSION_CALENDAR_SCHEMA, "open_session_calendar"),
        (session_clock, SESSION_CLOCK_SCHEMA, "session_clock"),
    ):
        if (
            reference.artifact_schema != schema
            or reference.root_policy != PRIVATE_ROOT_POLICY
        ):
            raise EvidenceV2Error(f"genesis v3 {label} ref is invalid")
    return seal_semantic(
        {
            "schema_version": ATTEMPT_GENESIS_V3_SCHEMA,
            "protocol_version": "v16",
            "protocol_attempt_id": _id(protocol_attempt_id, label="protocol_attempt_id"),
            "max_attempts_v16": MAX_ATTEMPTS_V16,
            "epoch_order": list(EPOCH_ORDER),
            "runtime_capsule": runtime_capsule.to_dict(),
            "proposed_factor_graph": proposed_factor_graph.to_dict(),
            "open_session_calendar": open_session_calendar.to_dict(),
            "session_clock": session_clock.to_dict(),
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_attempt_genesis_v3(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_version",
        "protocol_attempt_id",
        "max_attempts_v16",
        "epoch_order",
        "runtime_capsule",
        "proposed_factor_graph",
        "open_session_calendar",
        "session_clock",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    payload = _exact(payload, fields, label="attempt genesis v3")
    if (
        payload["schema_version"] != ATTEMPT_GENESIS_V3_SCHEMA
        or payload["protocol_version"] != "v16"
        or payload["max_attempts_v16"] != MAX_ATTEMPTS_V16
        or payload["epoch_order"] != list(EPOCH_ORDER)
    ):
        raise EvidenceV2Error("attempt genesis v3 protocol envelope mismatch")
    _id(payload["protocol_attempt_id"], label="protocol_attempt_id")
    runtime = EvidenceRef.from_dict(
        _reference(payload["runtime_capsule"], label="runtime_capsule")
    )
    graph = EvidenceRef.from_dict(
        _reference(payload["proposed_factor_graph"], label="proposed_factor_graph")
    )
    calendar = EvidenceRef.from_dict(
        _reference(payload["open_session_calendar"], label="open_session_calendar")
    )
    clock = EvidenceRef.from_dict(
        _reference(payload["session_clock"], label="session_clock")
    )
    if (
        runtime.artifact_schema != RUNTIME_CAPSULE_SCHEMA
        or runtime.root_policy != PRIVATE_ROOT_POLICY
        or graph.artifact_schema != TRANSITION_GRAPH_SCHEMA
        or graph.root_policy != PRIVATE_ROOT_POLICY
    ):
        raise EvidenceV2Error("attempt genesis v3 runtime/factor refs drift")
    if (
        calendar.artifact_schema != OPEN_SESSION_CALENDAR_SCHEMA
        or clock.artifact_schema != SESSION_CLOCK_SCHEMA
        or calendar.root_policy != PRIVATE_ROOT_POLICY
        or clock.root_policy != PRIVATE_ROOT_POLICY
    ):
        raise EvidenceV2Error("attempt genesis v3 calendar/clock refs drift")
    if any(
        payload[field] is not False
        for field in (
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
        )
    ):
        raise EvidenceV2Error("attempt genesis v3 must be permanently nonauthorizing")
    return payload


def _validate_slots_v3(
    slots: list[Any],
    *,
    epoch: str,
    calendar: Mapping[str, Any],
    session_clock: Mapping[str, Any],
) -> list[dict[str, Any]]:
    normalized_calendar = validate_open_session_calendar(calendar)
    normalized_clock = validate_session_clock(session_clock)
    if not slots:
        raise EvidenceV2Error("schedule v3 must contain at least one slot")
    expected_window = 30 if epoch == "B" else 20
    open_sessions = list(normalized_calendar["open_sessions"])
    open_index = {value: index for index, value in enumerate(open_sessions)}
    effective_from = str(normalized_clock["effective_from"])
    fields = {
        "slot_id",
        "s0_date",
        "s0_open_at",
        "s0_close_at",
        "decision_cutoff_at",
        "s1_open_at",
        "target_sessions",
    }
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    previous_target_end: str | None = None
    for index, raw in enumerate(slots):
        row = _exact(raw, fields, label=f"slots[{index}]")
        slot_id = _id(row["slot_id"], label="slot_id")
        if slot_id in seen_ids:
            raise EvidenceV2Error("schedule v3 slot IDs must be unique")
        seen_ids.add(slot_id)
        s0_date = _date(row["s0_date"], label="s0_date")
        if s0_date not in open_index or s0_date < effective_from:
            raise EvidenceV2Error("schedule v3 s0 is not an eligible bound-calendar session")
        if previous_target_end is not None and s0_date <= previous_target_end:
            raise EvidenceV2Error("schedule v3 slots overlap or are not strictly ordered")
        expected_open = f"{s0_date}T01:15:00Z"
        expected_close = f"{s0_date}T07:00:00Z"
        expected_cutoff = f"{s0_date}T07:30:00Z"
        if (
            row["s0_open_at"] != expected_open
            or row["s0_close_at"] != expected_close
            or row["decision_cutoff_at"] != expected_cutoff
        ):
            raise EvidenceV2Error("schedule v3 s0 clock boundaries drift")
        s0_open = _utc(row["s0_open_at"], label="s0_open_at")
        s0_close = _utc(row["s0_close_at"], label="s0_close_at")
        cutoff = _utc(row["decision_cutoff_at"], label="decision_cutoff_at")
        sessions_raw = row["target_sessions"]
        if not isinstance(sessions_raw, list) or len(sessions_raw) != expected_window:
            raise EvidenceV2Error(
                f"epoch {epoch} target window must contain {expected_window} sessions"
            )
        sessions = [_date(item, label="target_session") for item in sessions_raw]
        start = open_index[s0_date] + 1
        expected_sessions = open_sessions[start : start + expected_window]
        if len(expected_sessions) != expected_window or sessions != expected_sessions:
            raise EvidenceV2Error(
                "schedule v3 target window must be the next consecutive open sessions"
            )
        expected_s1 = f"{sessions[0]}T01:15:00Z"
        if row["s1_open_at"] != expected_s1:
            raise EvidenceV2Error("schedule v3 s1_open_at is not the next official open")
        s1_open = _utc(row["s1_open_at"], label="s1_open_at")
        if not s0_open < s0_close < cutoff < s1_open:
            raise EvidenceV2Error("schedule v3 slot timestamp chronology is invalid")
        previous_target_end = sessions[-1]
        normalized.append(
            {
                "slot_id": slot_id,
                "s0_date": s0_date,
                "s0_open_at": expected_open,
                "s0_close_at": expected_close,
                "decision_cutoff_at": expected_cutoff,
                "s1_open_at": expected_s1,
                "target_sessions": sessions,
            }
        )
    return normalized


def build_schedule_declaration_v3(
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
    calibration_universe_ref: EvidenceRef | None,
    slots: Sequence[Mapping[str, Any]],
    calendar: Mapping[str, Any],
    session_clock_value: Mapping[str, Any],
) -> dict[str, Any]:
    epoch_name = str(epoch)
    if epoch_name not in EPOCH_ORDER:
        raise EvidenceV2Error("schedule v3 epoch must be A, B, or C")
    seed = str(seed_hex)
    if len(seed) != 64 or any(character not in "0123456789abcdef" for character in seed):
        raise EvidenceV2Error("schedule v3 seed must be exactly 32 lowercase-hex bytes")
    for reference, schema, label in (
        (genesis_ref, ATTEMPT_GENESIS_V3_SCHEMA, "genesis_ref"),
        (open_session_calendar, OPEN_SESSION_CALENDAR_SCHEMA, "open_session_calendar"),
        (session_clock, SESSION_CLOCK_SCHEMA, "session_clock"),
        (calendar_recheck_ref, CALENDAR_RECHECK_SCHEMA, "calendar_recheck_ref"),
    ):
        if (
            reference.artifact_schema != schema
            or reference.root_policy != PRIVATE_ROOT_POLICY
        ):
            raise EvidenceV2Error(f"schedule v3 {label} is invalid")
    if (
        runtime_capsule.artifact_schema != RUNTIME_CAPSULE_SCHEMA
        or runtime_capsule.root_policy != PRIVATE_ROOT_POLICY
    ):
        raise EvidenceV2Error("schedule v3 runtime capsule ref is invalid")
    if epoch_name == "A":
        if model_bundle_refs is not None or calibration_universe_ref is not None:
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
            calibration_universe_ref is None
            or calibration_universe_ref.artifact_schema != CALIBRATION_UNIVERSE_SCHEMA
            or calibration_universe_ref.root_policy != PRIVATE_ROOT_POLICY
        ):
            raise EvidenceV2Error("epoch B/C calibration universe ref is invalid")
    normalized_slots = _validate_slots_v3(
        list(slots),
        epoch=epoch_name,
        calendar=calendar,
        session_clock=session_clock_value,
    )
    return seal_semantic(
        {
            "schema_version": SCHEDULE_DECLARATION_V3_SCHEMA,
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
            "calibration_universe_ref": (
                None if calibration_universe_ref is None else calibration_universe_ref.to_dict()
            ),
            "slots": normalized_slots,
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_schedule_declaration_v3(
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
        "calibration_universe_ref",
        "slots",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    payload = _exact(payload, fields, label="schedule declaration v3")
    if payload["schema_version"] != SCHEDULE_DECLARATION_V3_SCHEMA:
        raise EvidenceV2Error("unsupported schedule declaration v3 schema")
    _id(payload["protocol_attempt_id"], label="protocol_attempt_id")
    _id(payload["schedule_id"], label="schedule_id")
    epoch = str(payload["epoch"])
    if epoch not in EPOCH_ORDER:
        raise EvidenceV2Error("schedule v3 epoch is invalid")
    seed = str(payload["seed_hex"])
    if len(seed) != 64 or any(character not in "0123456789abcdef" for character in seed):
        raise EvidenceV2Error("schedule v3 seed is invalid")
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
            raise EvidenceV2Error(f"schedule v3 {label} drift")
        payload[label] = reference.to_dict()
    payload["runtime_capsule"] = _reference(
        payload["runtime_capsule"], label="runtime_capsule"
    )
    runtime_ref = EvidenceRef.from_dict(payload["runtime_capsule"])
    if (
        runtime_ref.artifact_schema != RUNTIME_CAPSULE_SCHEMA
        or runtime_ref.root_policy != PRIVATE_ROOT_POLICY
    ):
        raise EvidenceV2Error("schedule v3 runtime capsule ref drift")
    if epoch == "A":
        if payload["model_bundle_refs"] is not None:
            raise EvidenceV2Error("epoch A must not bind model bundles")
        if payload["calibration_universe_ref"] is not None:
            raise EvidenceV2Error("epoch A must not bind a calibration universe")
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
        if payload["calibration_universe_ref"] is None:
            raise EvidenceV2Error("epoch B/C calibration universe is missing")
        payload["calibration_universe_ref"] = _reference(
            payload["calibration_universe_ref"],
            label="calibration_universe_ref",
        )
        universe_ref = EvidenceRef.from_dict(payload["calibration_universe_ref"])
        if (
            universe_ref.artifact_schema != CALIBRATION_UNIVERSE_SCHEMA
            or universe_ref.root_policy != PRIVATE_ROOT_POLICY
        ):
            raise EvidenceV2Error("epoch B/C calibration universe ref is invalid")
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
        raise EvidenceV2Error("schedule v3 must be permanently nonauthorizing")
    return payload


@dataclass(frozen=True)
class EvidenceLoadLocation:
    reference: EvidenceRef
    root: str
    policy: RootPolicy

    def __post_init__(self) -> None:
        if not isinstance(self.reference, EvidenceRef) or not isinstance(
            self.policy, RootPolicy
        ):
            raise EvidenceV2Error("evidence load location fields have invalid types")
        root = Path(self.root)
        text = str(root)
        if (
            not root.is_absolute()
            or "\x00" in text
            or os.path.normpath(text) != text
            or text.startswith("//")
            or text.endswith("/")
        ):
            raise EvidenceV2Error("evidence load root must be canonical and absolute")
        try:
            Path(self.reference.absolute_path).relative_to(root)
        except ValueError as exc:
            raise EvidenceV2Error("evidence load reference escapes its explicit root") from exc
        if self.reference.root_policy != self.policy.policy_id:
            raise EvidenceV2Error("evidence load location policy mismatch")


def _load_canonical(location: EvidenceLoadLocation) -> BoundCanonicalArtifact:
    if not isinstance(location, EvidenceLoadLocation):
        raise EvidenceV2Error("canonical load requires an EvidenceLoadLocation")
    return load_bound_canonical_artifact(
        root=location.root,
        reference=location.reference,
        policy=location.policy,
    )


def _load_raw(location: EvidenceLoadLocation) -> BoundRawArtifact:
    if not isinstance(location, EvidenceLoadLocation):
        raise EvidenceV2Error("raw load requires an EvidenceLoadLocation")
    return load_bound_raw_artifact(
        root=location.root,
        reference=location.reference,
        policy=location.policy,
    )


def _reference_identity(label: str, reference: EvidenceRef) -> tuple[str, ...]:
    return (
        label,
        reference.schema_version,
        reference.artifact_schema,
        reference.absolute_path,
        reference.byte_sha256,
        reference.semantic_sha256,
        reference.root_policy,
    )


@dataclass(frozen=True)
class RuntimeCapsuleEvidenceBundle:
    capsule: BoundCanonicalArtifact
    components: tuple[BoundRawArtifact, ...]

    @classmethod
    def load(
        cls,
        *,
        capsule: EvidenceLoadLocation,
        components: Sequence[EvidenceLoadLocation],
    ) -> "RuntimeCapsuleEvidenceBundle":
        return cls(
            capsule=_load_canonical(capsule),
            components=tuple(_load_raw(location) for location in components),
        )

    def read(self) -> dict[str, Any]:
        if (
            self.capsule.reference.artifact_schema != RUNTIME_CAPSULE_SCHEMA
            or self.capsule.reference.root_policy != PRIVATE_ROOT_POLICY
        ):
            raise EvidenceV2Error("runtime capsule bundle ref is invalid")
        payload = validate_runtime_capsule(self.capsule.read())
        raw_components = payload["components"]
        if (
            not isinstance(raw_components, list)
            or len(raw_components) != len(RUNTIME_COMPONENT_ORDER)
            or len(self.components) != len(RUNTIME_COMPONENT_ORDER)
        ):
            raise EvidenceV2Error("runtime capsule transitive component count drift")
        for index, (component, artifact) in enumerate(
            zip(raw_components, self.components)
        ):
            if (
                component["component_id"] != RUNTIME_COMPONENT_ORDER[index]
                or component["artifact_ref"] != artifact.reference.to_dict()
            ):
                raise EvidenceV2Error("runtime capsule transitive component ref drift")
        return payload


@dataclass(frozen=True)
class FactorGraphEvidenceBundle:
    graph: BoundCanonicalArtifact
    factor_sets: tuple[BoundCanonicalArtifact, ...]

    @classmethod
    def load(
        cls,
        *,
        graph: EvidenceLoadLocation,
        factor_sets: Sequence[EvidenceLoadLocation],
    ) -> "FactorGraphEvidenceBundle":
        return cls(
            graph=_load_canonical(graph),
            factor_sets=tuple(_load_canonical(location) for location in factor_sets),
        )

    def read(self) -> dict[str, Any]:
        if self.graph.reference.artifact_schema != TRANSITION_GRAPH_SCHEMA:
            raise EvidenceV2Error("factor graph bundle schema drift")
        graph = validate_transition_graph(self.graph.read())
        ordered_refs: list[dict[str, str]] = []
        seen: set[tuple[str, ...]] = set()
        for transition in graph["transitions"]:
            for arm in ("A", "B", "C", "D"):
                reference = transition["arm_factor_sets"][arm]
                identity = tuple(reference[key] for key in sorted(reference))
                if identity not in seen:
                    seen.add(identity)
                    ordered_refs.append(reference)
        if len(self.factor_sets) != len(ordered_refs):
            raise EvidenceV2Error("factor graph transitive factor-set count drift")
        for expected, artifact in zip(ordered_refs, self.factor_sets):
            if expected != artifact.reference.to_dict():
                raise EvidenceV2Error("factor graph transitive factor-set ref drift")
            artifact.read()
        return graph


@dataclass(frozen=True)
class AttemptGenesisEvidenceBundleV3:
    genesis: BoundCanonicalArtifact
    runtime: RuntimeCapsuleEvidenceBundle
    factor_graph: FactorGraphEvidenceBundle
    calendar: CalendarEvidenceBundle
    session_clock: SessionClockEvidenceBundle

    def read(self) -> dict[str, Any]:
        if (
            self.genesis.reference.artifact_schema != ATTEMPT_GENESIS_V3_SCHEMA
            or self.genesis.reference.root_policy != PRIVATE_ROOT_POLICY
        ):
            raise EvidenceV2Error("attempt genesis v3 bundle ref is invalid")
        genesis = validate_attempt_genesis_v3(self.genesis.read())
        runtime = self.runtime.read()
        graph = self.factor_graph.read()
        calendar = self.calendar.read()
        clock = self.session_clock.read()
        if (
            genesis["runtime_capsule"] != self.runtime.capsule.reference.to_dict()
            or genesis["proposed_factor_graph"]
            != self.factor_graph.graph.reference.to_dict()
            or genesis["open_session_calendar"]
            != self.calendar.calendar.reference.to_dict()
            or genesis["session_clock"]
            != self.session_clock.session_clock.reference.to_dict()
        ):
            raise EvidenceV2Error("attempt genesis v3 transitive refs drift")
        attempt_id = genesis["protocol_attempt_id"]
        if (
            runtime["protocol_attempt_id"] != attempt_id
            or graph["protocol_attempt_id"] != attempt_id
            or calendar["calendar_id"] != "cn.open-session-calendar.2026.v1"
            or clock["session_clock_id"] != "cn.listed-equity-auction-clock.2026.v1"
        ):
            raise EvidenceV2Error("attempt genesis v3 transitive identity drift")
        return genesis


@dataclass(frozen=True)
class FrozenModelEvidenceBundle:
    model_bundle: BoundCanonicalArtifact
    training_schedule: BoundCanonicalArtifact
    training_capture: BoundCanonicalArtifact
    feature_contract: BoundCanonicalArtifact
    hyperparameters: BoundCanonicalArtifact
    serialized_model: BoundRawArtifact
    llm_tokenizer: BoundRawArtifact | None = None
    llm_inference_config: BoundCanonicalArtifact | None = None
    llm_provider_attestation: BoundCanonicalArtifact | None = None

    @classmethod
    def load(
        cls,
        *,
        model_bundle: EvidenceLoadLocation,
        training_schedule: EvidenceLoadLocation,
        training_capture: EvidenceLoadLocation,
        feature_contract: EvidenceLoadLocation,
        hyperparameters: EvidenceLoadLocation,
        serialized_model: EvidenceLoadLocation,
        llm_tokenizer: EvidenceLoadLocation | None = None,
        llm_inference_config: EvidenceLoadLocation | None = None,
        llm_provider_attestation: EvidenceLoadLocation | None = None,
    ) -> "FrozenModelEvidenceBundle":
        return cls(
            model_bundle=_load_canonical(model_bundle),
            training_schedule=_load_canonical(training_schedule),
            training_capture=_load_canonical(training_capture),
            feature_contract=_load_canonical(feature_contract),
            hyperparameters=_load_canonical(hyperparameters),
            serialized_model=_load_raw(serialized_model),
            llm_tokenizer=(None if llm_tokenizer is None else _load_raw(llm_tokenizer)),
            llm_inference_config=(
                None
                if llm_inference_config is None
                else _load_canonical(llm_inference_config)
            ),
            llm_provider_attestation=(
                None
                if llm_provider_attestation is None
                else _load_canonical(llm_provider_attestation)
            ),
        )

    def read(self) -> dict[str, Any]:
        if (
            self.model_bundle.reference.artifact_schema != MODEL_BUNDLE_SCHEMA
            or self.model_bundle.reference.root_policy != PRIVATE_ROOT_POLICY
        ):
            raise EvidenceV2Error("frozen model evidence bundle ref is invalid")
        payload = validate_frozen_model_bundle(self.model_bundle.read())
        expected: dict[str, BoundCanonicalArtifact | BoundRawArtifact] = {
            "training_schedule_ref": self.training_schedule,
            "training_capture_ref": self.training_capture,
            "feature_contract_ref": self.feature_contract,
            "hyperparameter_ref": self.hyperparameters,
            "serialized_model_ref": self.serialized_model,
        }
        for field, artifact in expected.items():
            if payload[field] != artifact.reference.to_dict():
                raise EvidenceV2Error(f"frozen model transitive ref drift: {field}")
            if isinstance(artifact, BoundCanonicalArtifact):
                artifact.read()
        provider = payload["llm_provider_build"]
        optional = (
            self.llm_tokenizer,
            self.llm_inference_config,
            self.llm_provider_attestation,
        )
        if payload["branch"] == "llm":
            if provider is None or any(artifact is None for artifact in optional):
                raise EvidenceV2Error("LLM model transitive provider evidence is incomplete")
            assert self.llm_tokenizer is not None
            assert self.llm_inference_config is not None
            assert self.llm_provider_attestation is not None
            provider_refs: dict[str, BoundCanonicalArtifact | BoundRawArtifact] = {
                "tokenizer_ref": self.llm_tokenizer,
                "inference_config_ref": self.llm_inference_config,
                "provider_attestation_ref": self.llm_provider_attestation,
            }
            for field, artifact in provider_refs.items():
                if provider[field] != artifact.reference.to_dict():
                    raise EvidenceV2Error(f"LLM provider transitive ref drift: {field}")
                if isinstance(artifact, BoundCanonicalArtifact):
                    artifact.read()
        elif any(artifact is not None for artifact in optional):
            raise EvidenceV2Error("non-LLM model has LLM-only transitive evidence")
        return payload

    def evidence_identity(self) -> tuple[tuple[str, ...], ...]:
        payload = self.read()
        identities: list[tuple[str, ...]] = [
            _reference_identity("model_bundle", self.model_bundle.reference)
        ]
        for field in (
            "training_schedule_ref",
            "training_capture_ref",
            "feature_contract_ref",
            "hyperparameter_ref",
            "serialized_model_ref",
        ):
            reference = EvidenceRef.from_dict(payload[field])
            identities.append(_reference_identity(field, reference))
        provider = payload["llm_provider_build"]
        if provider is not None:
            for field in (
                "tokenizer_ref",
                "inference_config_ref",
                "provider_attestation_ref",
            ):
                reference = EvidenceRef.from_dict(provider[field])
                identities.append(_reference_identity(field, reference))
        return tuple(identities)


@dataclass(frozen=True)
class CalibrationUniverseEvidenceBundle:
    universe: BoundCanonicalArtifact
    lambda_folds: tuple[BoundCanonicalArtifact, ...]

    @classmethod
    def load(
        cls,
        *,
        universe: EvidenceLoadLocation,
        lambda_folds: Sequence[EvidenceLoadLocation],
    ) -> "CalibrationUniverseEvidenceBundle":
        return cls(
            universe=_load_canonical(universe),
            lambda_folds=tuple(_load_canonical(location) for location in lambda_folds),
        )

    def read(self) -> dict[str, Any]:
        from .metrics import (
            validate_calibration_universe_plan,
            validate_lambda_fold_evidence,
        )

        if (
            self.universe.reference.artifact_schema != CALIBRATION_UNIVERSE_SCHEMA
            or self.universe.reference.root_policy != PRIVATE_ROOT_POLICY
        ):
            raise EvidenceV2Error("calibration universe bundle ref is invalid")
        payload = validate_calibration_universe_plan(self.universe.read())
        expected_refs = [
            item
            for branch in MODEL_BRANCHES
            for item in payload["lambda_fold_refs"][branch]
        ]
        if len(expected_refs) != len(self.lambda_folds):
            raise EvidenceV2Error("calibration universe lambda-fold count drift")
        for expected, artifact in zip(expected_refs, self.lambda_folds):
            if (
                expected != artifact.reference.to_dict()
                or artifact.reference.artifact_schema != LAMBDA_FOLD_SCHEMA
            ):
                raise EvidenceV2Error("calibration universe lambda-fold ref drift")
            validate_lambda_fold_evidence(artifact.read())
        return payload


@dataclass(frozen=True)
class ScheduleEvidenceBundleV3:
    schedule: BoundCanonicalArtifact
    genesis: AttemptGenesisEvidenceBundleV3
    calendar_recheck: CalendarRecheckEvidenceBundle
    model_bundles: tuple[FrozenModelEvidenceBundle, ...]
    calibration_universe: CalibrationUniverseEvidenceBundle | None

    def read(self) -> dict[str, Any]:
        if (
            self.schedule.reference.artifact_schema != SCHEDULE_DECLARATION_V3_SCHEMA
            or self.schedule.reference.root_policy != PRIVATE_ROOT_POLICY
        ):
            raise EvidenceV2Error("schedule evidence v3 ref is invalid")
        genesis = self.genesis.read()
        calendar = self.genesis.calendar.read()
        clock = self.genesis.session_clock.read()
        recheck = self.calendar_recheck.read()
        schedule = validate_schedule_declaration_v3(
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
            raise EvidenceV2Error("schedule evidence v3 genesis/recheck refs drift")
        if (
            recheck["protocol_attempt_id"] != schedule["protocol_attempt_id"]
            or recheck["epoch"] != schedule["epoch"]
            or recheck["schedule_id"] != schedule["schedule_id"]
            or recheck["first_s0_open_at"]
            != min(slot["s0_open_at"] for slot in schedule["slots"])
        ):
            raise EvidenceV2Error("schedule-specific calendar recheck lineage drift")
        if schedule["protocol_attempt_id"] != genesis["protocol_attempt_id"]:
            raise EvidenceV2Error("schedule evidence v3 protocol attempt drift")
        if schedule["epoch"] == "A":
            if self.model_bundles or self.calibration_universe is not None:
                raise EvidenceV2Error("epoch A schedule bundle carries model evidence")
        else:
            model_payloads = [bundle.read() for bundle in self.model_bundles]
            if [payload["branch"] for payload in model_payloads] != list(MODEL_BRANCHES):
                raise EvidenceV2Error("schedule model evidence order drift")
            model_refs = {
                branch: bundle.model_bundle.reference.to_dict()
                for branch, bundle in zip(MODEL_BRANCHES, self.model_bundles)
            }
            if schedule["model_bundle_refs"] != model_refs:
                raise EvidenceV2Error("schedule model evidence refs drift")
            if any(
                payload["protocol_attempt_id"] != schedule["protocol_attempt_id"]
                for payload in model_payloads
            ):
                raise EvidenceV2Error("schedule model evidence attempt drift")
            if self.calibration_universe is None:
                raise EvidenceV2Error("epoch B/C schedule bundle lacks calibration universe")
            universe = self.calibration_universe.read()
            if (
                schedule["calibration_universe_ref"]
                != self.calibration_universe.universe.reference.to_dict()
                or universe["protocol_attempt_id"] != schedule["protocol_attempt_id"]
                or universe["epoch"] != schedule["epoch"]
                or universe["schedule_id"] != schedule["schedule_id"]
                or universe["model_bundle_refs"] != model_refs
            ):
                raise EvidenceV2Error("schedule calibration-universe lineage drift")
        return schedule

    def model_evidence_identity(self) -> tuple[tuple[tuple[str, ...], ...], ...]:
        self.read()
        return tuple(bundle.evidence_identity() for bundle in self.model_bundles)


@dataclass(frozen=True)
class ScheduleAnchorBindingV3:
    evidence: ScheduleEvidenceBundleV3
    timestamp: TimestampAnchorBinding


def validate_schedule_anchor_binding_v3(
    binding: ScheduleAnchorBindingV3,
) -> dict[str, Any]:
    if not isinstance(binding, ScheduleAnchorBindingV3):
        raise EvidenceV2Error("schedule anchor binding v3 has the wrong type")
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
        raise EvidenceV2Error("schedule v3 anchor artifacts must use the private root")
    first_s0_open = min(slot["s0_open_at"] for slot in schedule["slots"])
    if (
        receipt["anchored_artifact_ref"]
        != binding.evidence.schedule.reference.to_dict()
        or receipt["anchor_kind"] != "schedule_declaration"
        or receipt["anchor_not_before"] is not None
        or receipt["anchor_not_after"] != first_s0_open
        or attempt["protocol_attempt_id"] != schedule["protocol_attempt_id"]
    ):
        raise EvidenceV2Error("schedule v3 RFC3161 pre-s0 anchor lineage mismatch")
    return schedule


_LINEAGE_V3_BLOCKERS = (
    "calendar_recheck_capture_time_not_independently_evidenced",
    "calendar_recheck_transport_freshness_not_independently_attested",
    "evidence_v2_disconnected_from_authorizing_consumers",
    "global_attempt_registry_authority_not_integrated",
    "provisional_journal_head_not_bound_to_external_anti_rollback_authority",
)


def validate_bound_lineage_v3(
    *,
    genesis: AttemptGenesisEvidenceBundleV3,
    schedule_anchors: Sequence[ScheduleAnchorBindingV3],
) -> dict[str, Any]:
    normalized_genesis = genesis.read()
    normalized_schedules = [
        validate_schedule_anchor_binding_v3(item) for item in schedule_anchors
    ]
    epochs = [item["epoch"] for item in normalized_schedules]
    if epochs != list(EPOCH_ORDER[: len(epochs)]) or len(epochs) > len(EPOCH_ORDER):
        raise EvidenceV2Error("bound lineage v3 must contain one ordered A/B/C prefix")
    attempt_id = normalized_genesis["protocol_attempt_id"]
    for item, binding in zip(normalized_schedules, schedule_anchors):
        if (
            item["protocol_attempt_id"] != attempt_id
            or item["genesis_ref"] != genesis.genesis.reference.to_dict()
            or item["runtime_capsule"] != normalized_genesis["runtime_capsule"]
            or item["open_session_calendar"]
            != normalized_genesis["open_session_calendar"]
            or item["session_clock"] != normalized_genesis["session_clock"]
            or binding.evidence.genesis.genesis.reference
            != genesis.genesis.reference
        ):
            raise EvidenceV2Error("bound lineage v3 immutable genesis identity drift")
    schedule_ids = [item["schedule_id"] for item in normalized_schedules]
    if len(schedule_ids) != len(set(schedule_ids)):
        raise EvidenceV2Error("bound lineage v3 schedule IDs must be unique")
    for previous, current in zip(normalized_schedules, normalized_schedules[1:]):
        previous_end = max(
            session for slot in previous["slots"] for session in slot["target_sessions"]
        )
        current_start = min(slot["s0_date"] for slot in current["slots"])
        if current_start <= previous_end:
            raise EvidenceV2Error("bound lineage v3 epoch windows overlap")
    frozen = [
        binding.evidence.model_evidence_identity()
        for binding in schedule_anchors
        if binding.evidence.read()["epoch"] in {"B", "C"}
    ]
    if len(frozen) == 2 and frozen[0] != frozen[1]:
        raise EvidenceV2Error("epoch C full model evidence drifts from frozen epoch B")
    projection = nonauthorizing_projection(blockers=list(_LINEAGE_V3_BLOCKERS))
    if projection["blockers"] != list(_LINEAGE_V3_BLOCKERS):
        raise EvidenceV2Error("bound lineage v3 blocker set drift")
    return projection


@dataclass(frozen=True)
class ScheduleAnchorBinding:
    schedule: BoundCanonicalArtifact
    timestamp: TimestampAnchorBinding


def validate_schedule_anchor_binding(
    binding: ScheduleAnchorBinding,
) -> dict[str, Any]:
    if not isinstance(binding, ScheduleAnchorBinding):
        raise EvidenceV2Error("schedule anchor binding has the wrong type")
    schedule = validate_schedule_declaration(binding.schedule.read())
    attempt, receipt = binding.timestamp.read()
    if any(
        artifact.reference.root_policy != PRIVATE_ROOT_POLICY
        for artifact in (
            binding.schedule,
            binding.timestamp.attempt,
            binding.timestamp.validation_receipt,
        )
    ):
        raise EvidenceV2Error("schedule anchor artifacts must use the private root")
    first_s0_open = min(slot["s0_open_at"] for slot in schedule["slots"])
    if (
        receipt["anchored_artifact_ref"] != binding.schedule.reference.to_dict()
        or receipt["anchor_kind"] != "schedule_declaration"
        or receipt["anchor_not_before"] is not None
        or receipt["anchor_not_after"] != first_s0_open
        or attempt["protocol_attempt_id"] != schedule["protocol_attempt_id"]
    ):
        raise EvidenceV2Error("schedule RFC3161 pre-s0 anchor lineage mismatch")
    return schedule


def validate_bound_lineage(
    *,
    genesis: Mapping[str, Any],
    schedule_anchors: Sequence[ScheduleAnchorBinding],
) -> dict[str, Any]:
    normalized_genesis = validate_attempt_genesis(genesis)
    normalized_schedules = [validate_schedule_anchor_binding(item) for item in schedule_anchors]
    epochs = [item["epoch"] for item in normalized_schedules]
    if epochs != list(EPOCH_ORDER[: len(epochs)]) or len(epochs) > len(EPOCH_ORDER):
        raise EvidenceV2Error("bound lineage must contain at most one ordered A/B/C schedule")
    attempt_id = normalized_genesis["protocol_attempt_id"]
    if any(item["protocol_attempt_id"] != attempt_id for item in normalized_schedules):
        raise EvidenceV2Error("bound lineage protocol attempt IDs drift")
    runtime_ref = normalized_genesis["runtime_capsule"]
    if any(item["runtime_capsule"] != runtime_ref for item in normalized_schedules):
        raise EvidenceV2Error("bound lineage runtime capsule drifts from genesis")
    calendar_ref = normalized_genesis["open_session_calendar"]
    if any(item["open_session_calendar"] != calendar_ref for item in normalized_schedules):
        raise EvidenceV2Error("bound lineage open-session calendar drifts from genesis")
    schedule_ids = [item["schedule_id"] for item in normalized_schedules]
    if len(schedule_ids) != len(set(schedule_ids)):
        raise EvidenceV2Error("bound lineage schedule IDs must be unique")
    for previous, current in zip(normalized_schedules, normalized_schedules[1:]):
        previous_end = max(
            session for slot in previous["slots"] for session in slot["target_sessions"]
        )
        current_start = min(slot["s0_date"] for slot in current["slots"])
        if current_start <= previous_end:
            raise EvidenceV2Error("bound lineage epoch windows overlap")
    frozen_refs = [
        item["model_bundle_refs"] for item in normalized_schedules if item["epoch"] in {"B", "C"}
    ]
    if len(frozen_refs) == 2 and frozen_refs[0] != frozen_refs[1]:
        raise EvidenceV2Error("epoch C model bundles drift from frozen epoch A bundles")
    return nonauthorizing_projection(
        blockers=[
            "global_attempt_registry_authority_not_integrated",
            "evidence_v2_disconnected_from_authorizing_consumers",
        ]
    )


__all__ = [
    "ATTEMPT_GENESIS_SCHEMA",
    "ATTEMPT_GENESIS_V3_SCHEMA",
    "AttemptGenesisEvidenceBundleV3",
    "CalibrationUniverseEvidenceBundle",
    "EPOCH_ORDER",
    "EvidenceLoadLocation",
    "FactorGraphEvidenceBundle",
    "FrozenModelEvidenceBundle",
    "LINEAGE_EVENT_SCHEMA",
    "MAX_ATTEMPTS_V16",
    "MODEL_BRANCHES",
    "RuntimeCapsuleEvidenceBundle",
    "ScheduleAnchorBinding",
    "ScheduleAnchorBindingV3",
    "ScheduleEvidenceBundleV3",
    "SCHEDULE_DECLARATION_SCHEMA",
    "SCHEDULE_DECLARATION_V3_SCHEMA",
    "TRANSITION_GRAPH_SCHEMA",
    "build_attempt_genesis",
    "build_attempt_genesis_v3",
    "build_schedule_declaration",
    "build_schedule_declaration_v3",
    "validate_attempt_genesis",
    "validate_attempt_genesis_v3",
    "validate_bound_lineage",
    "validate_bound_lineage_v3",
    "validate_schedule_anchor_binding",
    "validate_schedule_anchor_binding_v3",
    "validate_schedule_declaration",
    "validate_schedule_declaration_v3",
    "validate_transition_graph",
]
