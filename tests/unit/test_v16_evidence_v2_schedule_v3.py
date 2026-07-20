from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from quant_investor.v16.evidence_v2.calendar import (
    OPEN_SESSIONS,
    bind_calendar_artifact,
    build_declared_open_session_calendar,
)
from quant_investor.v16.evidence_v2.calendar_recheck import CALENDAR_RECHECK_SCHEMA
from quant_investor.v16.evidence_v2.contracts import (
    EVIDENCE_REF_SCHEMA,
    EvidenceRef,
    EvidenceV2Error,
    canonical_json_bytes,
    seal_semantic,
    semantic_sha256,
)
from quant_investor.v16.evidence_v2.runtime_identity import (
    MODEL_BUNDLE_SCHEMA,
    RUNTIME_CAPSULE_SCHEMA,
)
from quant_investor.v16.evidence_v2.schedule import (
    ATTEMPT_GENESIS_V3_SCHEMA,
    CALIBRATION_UNIVERSE_SCHEMA,
    SCHEDULE_DECLARATION_V3_SCHEMA,
    TRANSITION_GRAPH_SCHEMA,
    EvidenceLoadLocation,
    build_attempt_genesis_v3,
    build_schedule_declaration_v3,
    validate_attempt_genesis_v3,
    validate_schedule_declaration_v3,
)
from quant_investor.v16.evidence_v2.secure_io import PRIVATE_EVIDENCE_POLICY
from quant_investor.v16.evidence_v2.session_clock import (
    bind_session_clock_artifact,
    build_declared_session_clock,
)


def _ref(name: str, schema: str) -> EvidenceRef:
    return EvidenceRef(
        schema_version=EVIDENCE_REF_SCHEMA,
        artifact_schema=schema,
        absolute_path=f"/private/v16/{name}.json",
        byte_sha256=hashlib.sha256(f"{name}:bytes".encode()).hexdigest(),
        semantic_sha256=hashlib.sha256(f"{name}:semantic".encode()).hexdigest(),
        root_policy="v16.private-evidence-root.v2",
    )


def _bound_ref(name: str, value: dict[str, object]) -> EvidenceRef:
    raw = canonical_json_bytes(value)
    return EvidenceRef(
        schema_version=EVIDENCE_REF_SCHEMA,
        artifact_schema=str(value["schema_version"]),
        absolute_path=f"/private/v16/{name}.json",
        byte_sha256=hashlib.sha256(raw).hexdigest(),
        semantic_sha256=semantic_sha256(value),
        root_policy="v16.private-evidence-root.v2",
    )


def _calendar_and_clock() -> tuple[dict[str, object], dict[str, object], EvidenceRef, EvidenceRef]:
    root = "/private/v16/calendar-sources"
    calendar = build_declared_open_session_calendar(root)
    clock = build_declared_session_clock(root)
    calendar_ref = bind_calendar_artifact(
        calendar,
        absolute_path="/private/v16/calendar.json",
    ).reference
    clock_ref = bind_session_clock_artifact(
        clock,
        absolute_path="/private/v16/clock.json",
    ).reference
    return calendar, clock, calendar_ref, clock_ref


def _slot(epoch: str, s0_date: str = "2026-07-06") -> dict[str, object]:
    count = 30 if epoch == "B" else 20
    index = OPEN_SESSIONS.index(s0_date)
    targets = list(OPEN_SESSIONS[index + 1 : index + 1 + count])
    return {
        "slot_id": f"{epoch.lower()}-slot-001",
        "s0_date": s0_date,
        "s0_open_at": f"{s0_date}T01:15:00Z",
        "s0_close_at": f"{s0_date}T07:00:00Z",
        "decision_cutoff_at": f"{s0_date}T07:30:00Z",
        "s1_open_at": f"{targets[0]}T01:15:00Z",
        "target_sessions": targets,
    }


def _genesis() -> tuple[dict[str, object], EvidenceRef, EvidenceRef, EvidenceRef, EvidenceRef]:
    _calendar, _clock, calendar_ref, clock_ref = _calendar_and_clock()
    runtime_ref = _ref("runtime", RUNTIME_CAPSULE_SCHEMA)
    graph_ref = _ref("factor-graph", TRANSITION_GRAPH_SCHEMA)
    genesis = build_attempt_genesis_v3(
        protocol_attempt_id="attempt-v16-001",
        runtime_capsule=runtime_ref,
        proposed_factor_graph=graph_ref,
        open_session_calendar=calendar_ref,
        session_clock=clock_ref,
    )
    return genesis, _bound_ref("genesis", genesis), runtime_ref, calendar_ref, clock_ref


def test_genesis_v3_binds_calendar_and_clock_and_stays_nonauthorizing() -> None:
    genesis, _ref_value, _runtime, _calendar, _clock = _genesis()

    assert validate_attempt_genesis_v3(genesis) == genesis
    assert genesis["schema_version"] == ATTEMPT_GENESIS_V3_SCHEMA
    assert genesis["activation_candidate"] is False
    assert genesis["new_risk_authorized"] is False
    assert genesis["production_apply_enabled"] is False


def test_epoch_a_schedule_v3_uses_exact_next_20_open_sessions() -> None:
    calendar, clock, _calendar_ref, _clock_ref = _calendar_and_clock()
    genesis, genesis_ref, runtime_ref, calendar_ref, clock_ref = _genesis()
    assert validate_attempt_genesis_v3(genesis) == genesis
    schedule = build_schedule_declaration_v3(
        protocol_attempt_id="attempt-v16-001",
        epoch="A",
        schedule_id="schedule-a",
        seed_hex="a" * 64,
        genesis_ref=genesis_ref,
        runtime_capsule=runtime_ref,
        open_session_calendar=calendar_ref,
        session_clock=clock_ref,
        calendar_recheck_ref=_ref("recheck-a", CALENDAR_RECHECK_SCHEMA),
        model_bundle_refs=None,
        calibration_universe_ref=None,
        slots=[_slot("A")],
        calendar=calendar,
        session_clock_value=clock,
    )

    assert validate_schedule_declaration_v3(
        schedule,
        calendar=calendar,
        session_clock_value=clock,
    ) == schedule
    assert schedule["schema_version"] == SCHEDULE_DECLARATION_V3_SCHEMA
    assert schedule["slots"][0]["target_sessions"] == list(
        OPEN_SESSIONS[
            OPEN_SESSIONS.index("2026-07-06") + 1 : OPEN_SESSIONS.index("2026-07-06")
            + 21
        ]
    )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda slot: slot.update(
                {
                    "s0_date": "2026-10-01",
                    "s0_open_at": "2026-10-01T01:15:00Z",
                    "s0_close_at": "2026-10-01T07:00:00Z",
                    "decision_cutoff_at": "2026-10-01T07:30:00Z",
                }
            ),
            "eligible bound-calendar session",
        ),
        (
            lambda slot: slot["target_sessions"].__setitem__(
                1, slot["target_sessions"][2]
            ),
            "next consecutive open sessions",
        ),
        (
            lambda slot: slot.update({"s0_open_at": "2026-07-06T01:16:00Z"}),
            "clock boundaries drift",
        ),
    ],
)
def test_schedule_v3_rejects_closure_skip_or_clock_drift(mutator, message: str) -> None:
    calendar, clock, _calendar_ref, _clock_ref = _calendar_and_clock()
    _genesis_value, genesis_ref, runtime_ref, calendar_ref, clock_ref = _genesis()
    slot = _slot("A")
    mutator(slot)

    with pytest.raises(EvidenceV2Error, match=message):
        build_schedule_declaration_v3(
            protocol_attempt_id="attempt-v16-001",
            epoch="A",
            schedule_id="schedule-a",
            seed_hex="a" * 64,
            genesis_ref=genesis_ref,
            runtime_capsule=runtime_ref,
            open_session_calendar=calendar_ref,
            session_clock=clock_ref,
            calendar_recheck_ref=_ref("recheck-a", CALENDAR_RECHECK_SCHEMA),
            model_bundle_refs=None,
            calibration_universe_ref=None,
            slots=[slot],
            calendar=calendar,
            session_clock_value=clock,
        )


def test_epoch_b_requires_ordered_four_model_refs_and_bound_universe() -> None:
    calendar, clock, _calendar_ref, _clock_ref = _calendar_and_clock()
    _genesis_value, genesis_ref, runtime_ref, calendar_ref, clock_ref = _genesis()
    models = {
        branch: _ref(f"model-{branch}", MODEL_BUNDLE_SCHEMA)
        for branch in ("quant", "fundamental", "macro", "llm")
    }
    schedule = build_schedule_declaration_v3(
        protocol_attempt_id="attempt-v16-001",
        epoch="B",
        schedule_id="schedule-b",
        seed_hex="b" * 64,
        genesis_ref=genesis_ref,
        runtime_capsule=runtime_ref,
        open_session_calendar=calendar_ref,
        session_clock=clock_ref,
        calendar_recheck_ref=_ref("recheck-b", CALENDAR_RECHECK_SCHEMA),
        model_bundle_refs=models,
        calibration_universe_ref=_ref("universe-b", CALIBRATION_UNIVERSE_SCHEMA),
        slots=[_slot("B")],
        calendar=calendar,
        session_clock_value=clock,
    )
    assert validate_schedule_declaration_v3(
        schedule,
        calendar=calendar,
        session_clock_value=clock,
    ) == schedule

    unordered = {key: models[key] for key in ("llm", "quant", "fundamental", "macro")}
    with pytest.raises(EvidenceV2Error, match="ordered four-branch"):
        build_schedule_declaration_v3(
            protocol_attempt_id="attempt-v16-001",
            epoch="B",
            schedule_id="schedule-b",
            seed_hex="b" * 64,
            genesis_ref=genesis_ref,
            runtime_capsule=runtime_ref,
            open_session_calendar=calendar_ref,
            session_clock=clock_ref,
            calendar_recheck_ref=_ref("recheck-b", CALENDAR_RECHECK_SCHEMA),
            model_bundle_refs=unordered,
            calibration_universe_ref=_ref("universe-b", CALIBRATION_UNIVERSE_SCHEMA),
            slots=[_slot("B")],
            calendar=calendar,
            session_clock_value=clock,
        )


def test_schedule_v3_rejects_authorization_even_when_resealed() -> None:
    calendar, clock, _calendar_ref, _clock_ref = _calendar_and_clock()
    _genesis_value, genesis_ref, runtime_ref, calendar_ref, clock_ref = _genesis()
    schedule = build_schedule_declaration_v3(
        protocol_attempt_id="attempt-v16-001",
        epoch="A",
        schedule_id="schedule-a",
        seed_hex="a" * 64,
        genesis_ref=genesis_ref,
        runtime_capsule=runtime_ref,
        open_session_calendar=calendar_ref,
        session_clock=clock_ref,
        calendar_recheck_ref=_ref("recheck-a", CALENDAR_RECHECK_SCHEMA),
        model_bundle_refs=None,
        calibration_universe_ref=None,
        slots=[_slot("A")],
        calendar=calendar,
        session_clock_value=clock,
    )
    mutated = deepcopy(schedule)
    mutated.pop("semantic_sha256")
    mutated["new_risk_authorized"] = True
    with pytest.raises(EvidenceV2Error, match="nonauthorizing"):
        validate_schedule_declaration_v3(
            seal_semantic(mutated),
            calendar=calendar,
            session_clock_value=clock,
        )


def test_evidence_load_location_has_explicit_root_and_no_kind_switch() -> None:
    reference = _ref("runtime", RUNTIME_CAPSULE_SCHEMA)
    location = EvidenceLoadLocation(
        reference=reference,
        root="/private/v16",
        policy=PRIVATE_EVIDENCE_POLICY,
    )
    assert set(location.__dataclass_fields__) == {"reference", "root", "policy"}

    with pytest.raises(EvidenceV2Error, match="escapes"):
        EvidenceLoadLocation(
            reference=reference,
            root="/different/private/root",
            policy=PRIVATE_EVIDENCE_POLICY,
        )
