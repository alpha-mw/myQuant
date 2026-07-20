"""Schedule-specific, pre-anchor calendar and clock source recheck."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .calendar import (
    BINDING_SPEC_BY_ID,
    CALENDAR_BINDING_IDS,
    PRIVATE_ROOT_POLICY,
    SOURCE_AUTHORITY_SCOPE,
    BoundSource,
    CalendarEvidenceBundle,
    bind_calendar_artifact,
    build_open_session_calendar,
    load_private_source_bindings,
    parse_source_semantics,
)
from .contracts import (
    BoundCanonicalArtifact,
    EVIDENCE_REF_SCHEMA,
    EvidenceRef,
    EvidenceV2Error,
    canonical_json_bytes,
    seal_semantic,
    sha256_bytes,
    validate_semantic_seal,
)
from .session_clock import (
    CLOCK_BINDING_IDS,
    SessionClockEvidenceBundle,
    bind_session_clock_artifact,
    build_session_clock,
)

CALENDAR_RECHECK_SCHEMA = "v16.calendar-pre-anchor-recheck.v1"
TRANSPORT_FRESHNESS_STATUS = "not_independently_attested"
CALENDAR_RECHECK_BLOCKERS = (
    "calendar_recheck_capture_time_not_independently_evidenced",
    "calendar_recheck_transport_freshness_not_independently_attested",
    "evidence_v2_disconnected_from_authorizing_consumers",
)
REQUIRED_RECHECK_BINDING_IDS = (
    "cn.sse.active-closure-schedule.2026.v1",
    "cn.szse.active-closure-schedule.2026.v1",
    "cn.bse.active-closure-schedule.2026.v1",
    "cn.sse.rule-notice.current.2026.v1",
    "cn.sse.rule-binary.current.2026.clock.v1",
    "cn.szse.rule-notice.current.2026.v1",
    "cn.szse.rule-binary.current.2026.clock.v1",
    "cn.bse.rule-inline.current.2026.calendar.v1",
    "cn.bse.rule-inline.current.2026.clock.v1",
)
REQUIRED_RECHECK_PARSER_IDS = (
    "cn.sse.active-closure-html.2026.v1",
    "cn.szse.active-closure-html.2026.v1",
    "cn.bse.active-closure-html.month-day.2026.v1",
    "cn.sse.rule-notice-explicit-html.2026.v1",
    "exact_byte_sha_to_code_frozen_profile_v1",
    "cn.szse.rule-notice-explicit-html.2026.v1",
    "cn.bse.inline-calendar-rule-html.v1",
    "cn.bse.inline-session-clock-rule-html.v1",
)
_ENTRY_FIELDS = {
    "source_binding_id",
    "raw_ref",
    "semantic_projection_sha256",
    "parser_contract_id",
    "comparison_status",
}
_TOP_FIELDS = {
    "schema_version",
    "recheck_id",
    "protocol_attempt_id",
    "epoch",
    "schedule_id",
    "first_s0_open_at",
    "bound_calendar_ref",
    "bound_session_clock_ref",
    "required_source_binding_ids",
    "source_rechecks",
    "parser_contract_ids",
    "transport_freshness_status",
    "authority_scope",
    "blockers",
    "activation_candidate",
    "new_risk_authorized",
    "production_apply_enabled",
    "semantic_sha256",
}


def _exact(value: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise EvidenceV2Error(f"{label} fields mismatch")
    return dict(value)


def _identifier(value: Any, *, label: str) -> str:
    text = str(value or "")
    if (
        not text
        or text != text.strip()
        or len(text) > 128
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in text
        )
    ):
        raise EvidenceV2Error(f"{label} is not a safe identifier")
    return text


def _utc(value: Any, *, label: str) -> str:
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
    return text


def _expected_binding_map(
    calendar_bundle: CalendarEvidenceBundle,
    session_clock_bundle: SessionClockEvidenceBundle,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Mapping[str, Any]]]:
    calendar = calendar_bundle.read()
    session_clock = session_clock_bundle.read()
    by_id: dict[str, Mapping[str, Any]] = {}
    for binding in (*calendar["source_bindings"], *session_clock["source_bindings"]):
        binding_id = str(binding["source_binding_id"])
        prior = by_id.get(binding_id)
        if prior is not None and prior != binding:
            raise EvidenceV2Error("calendar and clock source-binding registries disagree")
        by_id[binding_id] = binding
    if any(binding_id not in by_id for binding_id in REQUIRED_RECHECK_BINDING_IDS):
        raise EvidenceV2Error("bound calendar/clock is missing a recheck source")
    return calendar, session_clock, by_id


def _compare_sources(
    expected_by_id: Mapping[str, Mapping[str, Any]],
    observed_sources: Sequence[BoundSource],
) -> list[dict[str, Any]]:
    if [source.binding_id for source in observed_sources] != list(
        REQUIRED_RECHECK_BINDING_IDS
    ):
        raise EvidenceV2Error("observed recheck sources must preserve the exact order")
    if len({source.binding_id for source in observed_sources}) != 9:
        raise EvidenceV2Error("observed recheck sources must be unique")

    rows = [
        _compare_one_source(expected_by_id[source.binding_id], source)
        for source in observed_sources
    ]

    bse_calendar = rows[7]["raw_ref"]
    bse_clock = rows[8]["raw_ref"]
    for field in ("absolute_path", "byte_sha256", "root_policy"):
        if bse_calendar[field] != bse_clock[field]:
            raise EvidenceV2Error("BSE calendar/clock recheck physical alias drift")
    if (
        bse_calendar["artifact_schema"] == bse_clock["artifact_schema"]
        or bse_calendar["semantic_sha256"] == bse_clock["semantic_sha256"]
    ):
        raise EvidenceV2Error("BSE calendar/clock semantic aliases must remain distinct")
    return rows


def _compare_one_source(
    expected: Mapping[str, Any],
    source: BoundSource,
) -> dict[str, Any]:
    spec = BINDING_SPEC_BY_ID[source.binding_id]
    expected_ref = EvidenceRef.from_dict(expected["raw_ref"])
    observed_ref = source.artifact.reference
    semantic = parse_source_semantics(source.binding_id, source.artifact.payload)
    semantic_hash = str(semantic["semantic_sha256"])
    if (
        observed_ref.root_policy != PRIVATE_ROOT_POLICY
        or observed_ref.artifact_schema != semantic["schema_version"]
        or observed_ref.semantic_sha256 != semantic_hash
        or semantic_hash != expected_ref.semantic_sha256
    ):
        raise EvidenceV2Error(f"{source.binding_id} observed semantic identity drift")
    byte_match = observed_ref.byte_sha256 == expected_ref.byte_sha256
    if spec.parser_contract_id == "exact_byte_sha_to_code_frozen_profile_v1":
        if not byte_match:
            raise EvidenceV2Error(f"{source.binding_id} binary recheck byte drift")
        status = "byte_and_semantic_match"
    else:
        status = (
            "byte_and_semantic_match"
            if byte_match
            else "semantic_match_with_byte_drift"
        )
    return {
        "source_binding_id": source.binding_id,
        "raw_ref": observed_ref.to_dict(),
        "semantic_projection_sha256": semantic_hash,
        "parser_contract_id": spec.parser_contract_id,
        "comparison_status": status,
    }


def build_calendar_recheck(
    *,
    recheck_id: str,
    protocol_attempt_id: str,
    epoch: str,
    schedule_id: str,
    first_s0_open_at: str,
    calendar_bundle: CalendarEvidenceBundle,
    session_clock_bundle: SessionClockEvidenceBundle,
    observed_sources: Sequence[BoundSource],
) -> dict[str, Any]:
    epoch_name = str(epoch)
    if epoch_name not in {"A", "B", "C"}:
        raise EvidenceV2Error("calendar recheck epoch must be A, B, or C")
    calendar, session_clock, expected = _expected_binding_map(
        calendar_bundle,
        session_clock_bundle,
    )
    rows = _compare_sources(expected, observed_sources)
    payload = seal_semantic(
        {
            "schema_version": CALENDAR_RECHECK_SCHEMA,
            "recheck_id": _identifier(recheck_id, label="recheck_id"),
            "protocol_attempt_id": _identifier(
                protocol_attempt_id,
                label="protocol_attempt_id",
            ),
            "epoch": epoch_name,
            "schedule_id": _identifier(schedule_id, label="schedule_id"),
            "first_s0_open_at": _utc(first_s0_open_at, label="first_s0_open_at"),
            "bound_calendar_ref": calendar_bundle.calendar.reference.to_dict(),
            "bound_session_clock_ref": (
                session_clock_bundle.session_clock.reference.to_dict()
            ),
            "required_source_binding_ids": list(REQUIRED_RECHECK_BINDING_IDS),
            "source_rechecks": rows,
            "parser_contract_ids": list(REQUIRED_RECHECK_PARSER_IDS),
            "transport_freshness_status": TRANSPORT_FRESHNESS_STATUS,
            "authority_scope": SOURCE_AUTHORITY_SCOPE,
            "blockers": list(CALENDAR_RECHECK_BLOCKERS),
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )
    # Keep local names alive so static analysis cannot mistake the full read for ref-only use.
    if calendar["calendar_id"] is None or session_clock["session_clock_id"] is None:
        raise EvidenceV2Error("bound calendar or session clock identity is absent")
    return _validate_calendar_recheck_payload(payload)


def _validate_calendar_recheck_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _exact(
        validate_semantic_seal(value),
        _TOP_FIELDS,
        label="calendar recheck",
    )
    if payload["schema_version"] != CALENDAR_RECHECK_SCHEMA:
        raise EvidenceV2Error("unsupported calendar recheck schema")
    _identifier(payload["recheck_id"], label="recheck_id")
    _identifier(payload["protocol_attempt_id"], label="protocol_attempt_id")
    _identifier(payload["schedule_id"], label="schedule_id")
    if payload["epoch"] not in {"A", "B", "C"}:
        raise EvidenceV2Error("calendar recheck epoch is invalid")
    _utc(payload["first_s0_open_at"], label="first_s0_open_at")
    calendar_ref = EvidenceRef.from_dict(payload["bound_calendar_ref"])
    clock_ref = EvidenceRef.from_dict(payload["bound_session_clock_ref"])
    if (
        calendar_ref.artifact_schema != "v16.open-session-calendar.v1"
        or clock_ref.artifact_schema != "v16.session-clock.v1"
        or calendar_ref.root_policy != PRIVATE_ROOT_POLICY
        or clock_ref.root_policy != PRIVATE_ROOT_POLICY
    ):
        raise EvidenceV2Error("calendar recheck bound artifact refs drift")
    if payload["required_source_binding_ids"] != list(REQUIRED_RECHECK_BINDING_IDS):
        raise EvidenceV2Error("calendar recheck required-source order drift")
    if payload["parser_contract_ids"] != list(REQUIRED_RECHECK_PARSER_IDS):
        raise EvidenceV2Error("calendar recheck parser registry drift")
    if (
        payload["transport_freshness_status"] != TRANSPORT_FRESHNESS_STATUS
        or payload["authority_scope"] != SOURCE_AUTHORITY_SCOPE
        or payload["blockers"] != list(CALENDAR_RECHECK_BLOCKERS)
    ):
        raise EvidenceV2Error("calendar recheck status or blocker drift")
    if any(
        payload[field] is not False
        for field in (
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
        )
    ):
        raise EvidenceV2Error("calendar recheck must be permanently nonauthorizing")

    raw_rows = payload["source_rechecks"]
    if not isinstance(raw_rows, list) or len(raw_rows) != 9:
        raise EvidenceV2Error("calendar recheck must contain exactly nine entries")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        row = _exact(raw, _ENTRY_FIELDS, label=f"source_rechecks[{index}]")
        binding_id = REQUIRED_RECHECK_BINDING_IDS[index]
        spec = BINDING_SPEC_BY_ID[binding_id]
        if (
            row["source_binding_id"] != binding_id
            or row["parser_contract_id"] != spec.parser_contract_id
        ):
            raise EvidenceV2Error("calendar recheck entry identity drift")
        reference = EvidenceRef.from_dict(row["raw_ref"])
        if (
            reference.root_policy != PRIVATE_ROOT_POLICY
            or reference.artifact_schema != spec.semantic["schema_version"]
            or reference.semantic_sha256 != spec.semantic["semantic_sha256"]
            or row["semantic_projection_sha256"] != reference.semantic_sha256
        ):
            raise EvidenceV2Error("calendar recheck entry EvidenceRef drift")
        allowed_statuses = {"byte_and_semantic_match"}
        if spec.parser_contract_id != "exact_byte_sha_to_code_frozen_profile_v1":
            allowed_statuses.add("semantic_match_with_byte_drift")
        if row["comparison_status"] not in allowed_statuses:
            raise EvidenceV2Error("calendar recheck comparison status drift")
        row["raw_ref"] = reference.to_dict()
        rows.append(row)
    payload["source_rechecks"] = rows
    return payload


@dataclass(frozen=True)
class CalendarRecheckEvidenceBundle:
    recheck: BoundCanonicalArtifact
    calendar: CalendarEvidenceBundle
    session_clock: SessionClockEvidenceBundle
    observed_sources: tuple[BoundSource, ...]

    def read(self) -> dict[str, Any]:
        declared = _validate_calendar_recheck_payload(self.recheck.read())
        if (
            declared["bound_calendar_ref"] != self.calendar.calendar.reference.to_dict()
            or declared["bound_session_clock_ref"]
            != self.session_clock.session_clock.reference.to_dict()
        ):
            raise EvidenceV2Error("calendar recheck bundle bound refs drift")
        rebuilt = build_calendar_recheck(
            recheck_id=str(declared["recheck_id"]),
            protocol_attempt_id=str(declared["protocol_attempt_id"]),
            epoch=str(declared["epoch"]),
            schedule_id=str(declared["schedule_id"]),
            first_s0_open_at=str(declared["first_s0_open_at"]),
            calendar_bundle=self.calendar,
            session_clock_bundle=self.session_clock,
            observed_sources=self.observed_sources,
        )
        if rebuilt != declared:
            raise EvidenceV2Error("calendar recheck bundle does not recompute exactly")
        return declared


def validate_calendar_recheck(
    bundle: CalendarRecheckEvidenceBundle,
) -> dict[str, Any]:
    if not isinstance(bundle, CalendarRecheckEvidenceBundle):
        raise EvidenceV2Error("calendar recheck validation requires the full evidence bundle")
    return bundle.read()


def bind_calendar_recheck_artifact(
    value: Mapping[str, Any],
    *,
    absolute_path: str,
) -> BoundCanonicalArtifact:
    payload = _validate_calendar_recheck_payload(value)
    raw = canonical_json_bytes(payload)
    return BoundCanonicalArtifact(
        reference=EvidenceRef(
            schema_version=EVIDENCE_REF_SCHEMA,
            artifact_schema=CALENDAR_RECHECK_SCHEMA,
            absolute_path=absolute_path,
            byte_sha256=sha256_bytes(raw),
            semantic_sha256=str(payload["semantic_sha256"]),
            root_policy=PRIVATE_ROOT_POLICY,
        ),
        payload=raw,
    )


def validate_private_calendar_recheck_acceptance(
    root: str,
) -> dict[str, Any]:
    """Run the full local comparison without claiming capture-time freshness."""

    root_path = str(root)
    sources = load_private_source_bindings(root_path)
    by_id = {source.binding_id: source for source in sources}
    calendar_value = build_open_session_calendar(sources)
    clock_value = build_session_clock(sources)
    calendar_bundle = CalendarEvidenceBundle(
        calendar=bind_calendar_artifact(
            calendar_value,
            absolute_path=f"{root_path}/acceptance-open-session-calendar.json",
        ),
        sources=tuple(by_id[binding_id] for binding_id in CALENDAR_BINDING_IDS),
    )
    clock_bundle = SessionClockEvidenceBundle(
        session_clock=bind_session_clock_artifact(
            clock_value,
            absolute_path=f"{root_path}/acceptance-session-clock.json",
        ),
        sources=tuple(by_id[binding_id] for binding_id in CLOCK_BINDING_IDS),
    )
    observed = tuple(by_id[binding_id] for binding_id in REQUIRED_RECHECK_BINDING_IDS)
    recheck_value = build_calendar_recheck(
        recheck_id="local-private-source-acceptance",
        protocol_attempt_id="disconnected-v16-acceptance",
        epoch="A",
        schedule_id="local-private-source-acceptance-a",
        first_s0_open_at="2026-07-06T01:15:00Z",
        calendar_bundle=calendar_bundle,
        session_clock_bundle=clock_bundle,
        observed_sources=observed,
    )
    bundle = CalendarRecheckEvidenceBundle(
        recheck=bind_calendar_recheck_artifact(
            recheck_value,
            absolute_path=f"{root_path}/acceptance-calendar-recheck.json",
        ),
        calendar=calendar_bundle,
        session_clock=clock_bundle,
        observed_sources=observed,
    )
    return validate_calendar_recheck(bundle)


__all__ = [
    "CALENDAR_RECHECK_BLOCKERS",
    "CALENDAR_RECHECK_SCHEMA",
    "CalendarRecheckEvidenceBundle",
    "REQUIRED_RECHECK_BINDING_IDS",
    "REQUIRED_RECHECK_PARSER_IDS",
    "TRANSPORT_FRESHNESS_STATUS",
    "bind_calendar_recheck_artifact",
    "build_calendar_recheck",
    "validate_calendar_recheck",
    "validate_private_calendar_recheck_acceptance",
]
