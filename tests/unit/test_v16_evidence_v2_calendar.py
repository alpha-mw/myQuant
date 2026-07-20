from __future__ import annotations

from copy import deepcopy

import pytest

from quant_investor.v16.evidence_v2.calendar import (
    BINDING_SPECS,
    CALENDAR_BINDING_IDS,
    CLOSED_WEEKDAY_DATES,
    EXCLUDED_SOURCE_NAMES,
    OPEN_SESSIONS,
    SOURCE_INVENTORY,
    build_declared_open_session_calendar,
    declared_source_bindings,
    parse_source_semantics,
    validate_open_session_calendar,
    validate_source_binding,
)
from quant_investor.v16.evidence_v2.contracts import EvidenceV2Error, seal_semantic


def _synthetic_html(binding_id: str) -> bytes:
    spec = next(item for item in BINDING_SPECS if item.binding_id == binding_id)
    markers = "".join(group[0] for group in spec.marker_groups)
    return f"<html><body>{markers}</body></html>".encode()


def test_declared_calendar_is_exact_and_permanently_nonauthorizing() -> None:
    root = "/private/v16/calendar-sources"
    calendar = build_declared_open_session_calendar(root)

    assert validate_open_session_calendar(calendar) == calendar
    assert len(OPEN_SESSIONS) == 242
    assert len(CLOSED_WEEKDAY_DATES) == 19
    assert len(calendar["source_bindings"]) == 22
    assert [item["source_binding_id"] for item in calendar["source_bindings"]] == list(
        CALENDAR_BINDING_IDS
    )
    assert calendar["activation_candidate"] is False
    assert calendar["new_risk_authorized"] is False
    assert calendar["production_apply_enabled"] is False


def test_source_inventory_has_22_consumed_and_two_explicit_exclusions() -> None:
    assert len(SOURCE_INVENTORY) == 24
    assert sum(item.consumed for item in SOURCE_INVENTORY) == 22
    assert EXCLUDED_SOURCE_NAMES == {
        "myquant-szse-2023-rule-implementation-report.pdf",
        "myquant-szse-2026-closures.html",
    }


def test_declared_registry_has_28_exact_binding_rows() -> None:
    bindings = declared_source_bindings("/private/v16/calendar-sources")

    assert len(bindings) == 28
    assert all(validate_source_binding(item) == item for item in bindings)
    assert len({item["source_binding_id"] for item in bindings}) == 28
    assert all(
        item["authority_scope"]
        == "declared_exchange_url_semantic_correspondence_only"
        for item in bindings
    )


def test_html_parser_accepts_semantic_equivalence_but_rejects_marker_drift() -> None:
    binding_id = "cn.sse.rule-notice.current.2026.v1"
    raw = _synthetic_html(binding_id)
    first = parse_source_semantics(binding_id, raw)
    second = parse_source_semantics(
        binding_id,
        raw.replace(b"<body>", b"<body><nav>byte drift</nav>"),
    )
    assert first == second

    spec = next(item for item in BINDING_SPECS if item.binding_id == binding_id)
    missing = spec.marker_groups[-1][0].encode()
    with pytest.raises(EvidenceV2Error, match="semantic marker group"):
        parse_source_semantics(binding_id, raw.replace(missing, b"removed"))


def test_binary_profile_rejects_any_byte_drift() -> None:
    with pytest.raises(EvidenceV2Error, match="binary source byte SHA drift"):
        parse_source_semantics(
            "cn.sse.rule-binary.current.2026.calendar.v1",
            b"not-the-frozen-docx",
        )


def test_calendar_rejects_authorization_or_binding_mutation() -> None:
    calendar = build_declared_open_session_calendar("/private/v16/calendar-sources")
    mutated = deepcopy(calendar)
    mutated.pop("semantic_sha256")
    mutated["new_risk_authorized"] = True
    with pytest.raises(EvidenceV2Error, match="nonauthorizing"):
        validate_open_session_calendar(seal_semantic(mutated))

    binding = deepcopy(calendar["source_bindings"][0])
    binding["retrieval_method"] = "caller_supplied"
    with pytest.raises(EvidenceV2Error, match="retrieval_method drift"):
        validate_source_binding(binding)
