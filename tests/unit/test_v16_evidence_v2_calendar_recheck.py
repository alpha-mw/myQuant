from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from quant_investor.v16.evidence_v2.calendar import (
    BINDING_SPECS,
    BoundSource,
    declared_source_bindings,
)
from quant_investor.v16.evidence_v2.calendar_recheck import (
    CALENDAR_RECHECK_BLOCKERS,
    CALENDAR_RECHECK_SCHEMA,
    REQUIRED_RECHECK_BINDING_IDS,
    REQUIRED_RECHECK_PARSER_IDS,
    TRANSPORT_FRESHNESS_STATUS,
    _compare_one_source,
    _validate_calendar_recheck_payload,
)
from quant_investor.v16.evidence_v2.contracts import (
    BoundRawArtifact,
    EVIDENCE_REF_SCHEMA,
    EvidenceRef,
    EvidenceV2Error,
    seal_semantic,
)


def _artifact_ref(name: str, schema: str) -> EvidenceRef:
    return EvidenceRef(
        schema_version=EVIDENCE_REF_SCHEMA,
        artifact_schema=schema,
        absolute_path=f"/private/v16/{name}.json",
        byte_sha256=hashlib.sha256(f"{name}:bytes".encode()).hexdigest(),
        semantic_sha256=hashlib.sha256(f"{name}:semantic".encode()).hexdigest(),
        root_policy="v16.private-evidence-root.v2",
    )


def _recheck_payload() -> dict[str, object]:
    declared = {
        item["source_binding_id"]: item
        for item in declared_source_bindings("/private/v16/calendar-sources")
    }
    rows = []
    for binding_id in REQUIRED_RECHECK_BINDING_IDS:
        binding = declared[binding_id]
        rows.append(
            {
                "source_binding_id": binding_id,
                "raw_ref": binding["raw_ref"],
                "semantic_projection_sha256": binding[
                    "semantic_projection_sha256"
                ],
                "parser_contract_id": binding["parser_contract_id"],
                "comparison_status": "byte_and_semantic_match",
            }
        )
    return seal_semantic(
        {
            "schema_version": CALENDAR_RECHECK_SCHEMA,
            "recheck_id": "recheck-a-001",
            "protocol_attempt_id": "attempt-v16-001",
            "epoch": "A",
            "schedule_id": "schedule-a",
            "first_s0_open_at": "2026-07-06T01:15:00Z",
            "bound_calendar_ref": _artifact_ref(
                "calendar", "v16.open-session-calendar.v1"
            ).to_dict(),
            "bound_session_clock_ref": _artifact_ref(
                "clock", "v16.session-clock.v1"
            ).to_dict(),
            "required_source_binding_ids": list(REQUIRED_RECHECK_BINDING_IDS),
            "source_rechecks": rows,
            "parser_contract_ids": list(REQUIRED_RECHECK_PARSER_IDS),
            "transport_freshness_status": TRANSPORT_FRESHNESS_STATUS,
            "authority_scope": "declared_exchange_url_semantic_correspondence_only",
            "blockers": list(CALENDAR_RECHECK_BLOCKERS),
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def test_recheck_payload_is_exact_and_remains_blocked() -> None:
    payload = _recheck_payload()

    assert _validate_calendar_recheck_payload(payload) == payload
    assert payload["blockers"] == list(CALENDAR_RECHECK_BLOCKERS)
    assert len(payload["source_rechecks"]) == 9
    assert payload["new_risk_authorized"] is False


def test_recheck_rejects_transport_claim_or_binary_byte_drift_status() -> None:
    payload = _recheck_payload()
    mutated = deepcopy(payload)
    mutated.pop("semantic_sha256")
    mutated["transport_freshness_status"] = "attested"
    with pytest.raises(EvidenceV2Error, match="status or blocker drift"):
        _validate_calendar_recheck_payload(seal_semantic(mutated))

    mutated = deepcopy(payload)
    mutated.pop("semantic_sha256")
    mutated["source_rechecks"][4][
        "comparison_status"
    ] = "semantic_match_with_byte_drift"
    with pytest.raises(EvidenceV2Error, match="comparison status drift"):
        _validate_calendar_recheck_payload(seal_semantic(mutated))


def test_html_recheck_accepts_byte_drift_only_after_semantic_reparse() -> None:
    binding_id = "cn.sse.rule-notice.current.2026.v1"
    spec = next(item for item in BINDING_SPECS if item.binding_id == binding_id)
    raw = (
        "<html><body>"
        + "".join(group[0] for group in spec.marker_groups)
        + "<nav>new navigation bytes</nav></body></html>"
    ).encode()
    reference = EvidenceRef(
        schema_version=EVIDENCE_REF_SCHEMA,
        artifact_schema=str(spec.semantic["schema_version"]),
        absolute_path="/private/v16/rechecks/sse-current.html",
        byte_sha256=hashlib.sha256(raw).hexdigest(),
        semantic_sha256=str(spec.semantic["semantic_sha256"]),
        root_policy="v16.private-evidence-root.v2",
    )
    source = BoundSource(
        binding_id=binding_id,
        artifact=BoundRawArtifact(reference=reference, payload=raw),
    )
    expected = {
        item["source_binding_id"]: item
        for item in declared_source_bindings("/private/v16/calendar-sources")
    }[binding_id]

    row = _compare_one_source(expected, source)

    assert row["comparison_status"] == "semantic_match_with_byte_drift"
    assert row["semantic_projection_sha256"] == spec.semantic["semantic_sha256"]


def test_recheck_rejects_source_order_drift() -> None:
    payload = _recheck_payload()
    mutated = deepcopy(payload)
    mutated.pop("semantic_sha256")
    mutated["source_rechecks"][0], mutated["source_rechecks"][1] = (
        mutated["source_rechecks"][1],
        mutated["source_rechecks"][0],
    )
    with pytest.raises(EvidenceV2Error, match="entry identity drift"):
        _validate_calendar_recheck_payload(seal_semantic(mutated))
