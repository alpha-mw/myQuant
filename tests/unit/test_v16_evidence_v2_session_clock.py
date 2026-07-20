from __future__ import annotations

from copy import deepcopy

import pytest

from quant_investor.v16.evidence_v2.calendar import (
    CLOCK_EXCLUDED_SCOPES,
    CLOCK_SCOPE_ID,
    CLOCK_SEGMENTS,
)
from quant_investor.v16.evidence_v2.contracts import EvidenceV2Error, seal_semantic
from quant_investor.v16.evidence_v2.session_clock import (
    CLOCK_BINDING_IDS,
    SESSION_CLOCK_EFFECTIVE_FROM,
    build_declared_session_clock,
    validate_session_clock,
)


def test_declared_session_clock_is_exact_and_nonauthorizing() -> None:
    clock = build_declared_session_clock("/private/v16/calendar-sources")

    assert validate_session_clock(clock) == clock
    assert clock["scope_id"] == CLOCK_SCOPE_ID
    assert clock["segments"] == [dict(item) for item in CLOCK_SEGMENTS]
    assert clock["excluded_scopes"] == list(CLOCK_EXCLUDED_SCOPES)
    assert clock["effective_from"] == SESSION_CLOCK_EFFECTIVE_FROM
    assert len(clock["source_bindings"]) == 16
    assert [item["source_binding_id"] for item in clock["source_bindings"]] == list(
        CLOCK_BINDING_IDS
    )
    assert clock["activation_candidate"] is False
    assert clock["new_risk_authorized"] is False
    assert clock["production_apply_enabled"] is False


def test_session_clock_rejects_segment_or_effective_date_drift() -> None:
    clock = build_declared_session_clock("/private/v16/calendar-sources")
    mutated = deepcopy(clock)
    mutated.pop("semantic_sha256")
    mutated["segments"][0]["start_local_time"] = "09:16:00"
    with pytest.raises(EvidenceV2Error, match="frozen values drift"):
        validate_session_clock(seal_semantic(mutated))

    mutated = deepcopy(clock)
    mutated.pop("semantic_sha256")
    mutated["effective_from"] = "2026-01-01"
    with pytest.raises(EvidenceV2Error, match="frozen values drift"):
        validate_session_clock(seal_semantic(mutated))


def test_session_clock_rejects_authorization() -> None:
    clock = build_declared_session_clock("/private/v16/calendar-sources")
    mutated = deepcopy(clock)
    mutated.pop("semantic_sha256")
    mutated["production_apply_enabled"] = True
    with pytest.raises(EvidenceV2Error, match="nonauthorizing"):
        validate_session_clock(seal_semantic(mutated))
