from __future__ import annotations

from copy import deepcopy

import pytest

from quant_investor.v16.evidence_v2.calibration_plan_v3 import (
    CALIBRATION_PLAN_V3_SCHEMA,
    SCHEDULE_V4_SCHEMA,
)
from quant_investor.v16.evidence_v2.calendar_recheck import CALENDAR_RECHECK_SCHEMA
from quant_investor.v16.evidence_v2.contracts import EvidenceV2Error, seal_semantic
from quant_investor.v16.evidence_v2.runtime_identity import MODEL_BUNDLE_SCHEMA
from quant_investor.v16.evidence_v2.schedule import (
    CALIBRATION_UNIVERSE_SCHEMA,
    build_schedule_declaration_v3,
)
from quant_investor.v16.evidence_v2.schedule_v4 import (
    build_schedule_declaration_v4,
    validate_schedule_declaration_v4,
)
from tests.unit.test_v16_evidence_v2_schedule_v3 import (
    _calendar_and_clock,
    _genesis,
    _ref,
    _slot,
)


def _inputs():
    calendar, clock, _calendar_ref, _clock_ref = _calendar_and_clock()
    _genesis_value, genesis_ref, runtime_ref, calendar_ref, clock_ref = _genesis()
    models = {
        branch: _ref(f"model-{branch}", MODEL_BUNDLE_SCHEMA)
        for branch in ("quant", "fundamental", "macro", "llm")
    }
    return calendar, clock, genesis_ref, runtime_ref, calendar_ref, clock_ref, models


def test_schedule_v4_binds_only_calibration_plan_v3() -> None:
    calendar, clock, genesis_ref, runtime_ref, calendar_ref, clock_ref, models = _inputs()
    schedule = build_schedule_declaration_v4(
        protocol_attempt_id="attempt-v16-001",
        epoch="B",
        schedule_id="schedule-b",
        seed_hex="b" * 64,
        genesis_ref=genesis_ref,
        runtime_capsule=runtime_ref,
        open_session_calendar=calendar_ref,
        session_clock=clock_ref,
        calendar_recheck_ref=_ref(
            "recheck-b",
            CALENDAR_RECHECK_SCHEMA,
        ),
        model_bundle_refs=models,
        calibration_plan_ref=_ref("plan-b", CALIBRATION_PLAN_V3_SCHEMA),
        slots=[_slot("B")],
        calendar=calendar,
        session_clock_value=clock,
    )

    validated = validate_schedule_declaration_v4(
        schedule,
        calendar=calendar,
        session_clock_value=clock,
    )
    assert validated["schema_version"] == SCHEDULE_V4_SCHEMA
    assert validated["calibration_plan_ref"]["artifact_schema"] == CALIBRATION_PLAN_V3_SCHEMA


def test_schedule_v4_rejects_v2_universe_and_schedule_v3_payload() -> None:
    calendar, clock, genesis_ref, runtime_ref, calendar_ref, clock_ref, models = _inputs()
    with pytest.raises(EvidenceV2Error, match="calibration plan v3 ref"):
        build_schedule_declaration_v4(
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
            calibration_plan_ref=_ref("universe-b", CALIBRATION_UNIVERSE_SCHEMA),
            slots=[_slot("B")],
            calendar=calendar,
            session_clock_value=clock,
        )

    schedule_v3 = build_schedule_declaration_v3(
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
    with pytest.raises(EvidenceV2Error, match="fields mismatch"):
        validate_schedule_declaration_v4(
            schedule_v3,
            calendar=calendar,
            session_clock_value=clock,
        )


def test_schedule_v4_rejects_resealed_authorization() -> None:
    calendar, clock, genesis_ref, runtime_ref, calendar_ref, clock_ref, _models = _inputs()
    schedule = build_schedule_declaration_v4(
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
        calibration_plan_ref=None,
        slots=[_slot("A")],
        calendar=calendar,
        session_clock_value=clock,
    )
    tampered = deepcopy(schedule)
    tampered.pop("semantic_sha256")
    tampered["new_risk_authorized"] = True
    with pytest.raises(EvidenceV2Error, match="permanently nonauthorizing"):
        validate_schedule_declaration_v4(
            seal_semantic(tampered),
            calendar=calendar,
            session_clock_value=clock,
        )
