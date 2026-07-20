"""Frozen listed-equity auction clock for disconnected v16 evidence-v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .calendar import (
    BINDING_SPECS,
    CLOCK_EXCLUDED_SCOPES,
    CLOCK_SCOPE_ID,
    CLOCK_SEGMENTS,
    PRIVATE_ROOT_POLICY,
    SOURCE_DOCUMENT_SET_BY_ID,
    BoundSource,
    _binding_from_source,
    _exact,
    declared_source_bindings,
    validate_source_binding,
    validate_source_document_set,
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

SESSION_CLOCK_SCHEMA = "v16.session-clock.v1"
SESSION_CLOCK_ID = "cn.listed-equity-auction-clock.2026.v1"
SESSION_CLOCK_TIMEZONE = "Asia/Shanghai"
SESSION_CLOCK_EFFECTIVE_FROM = "2026-07-06"
SESSION_CLOCK_AUTHORITY_SCOPE = "official_exchange_trading_rules_only"

CLOCK_SOURCE_SET_IDS = (
    "cn.mainboard-registration-effective-date.2023.v1",
    "cn.sse.rule-notice-history.2023-2026.v1",
    "cn.szse.rule-notice-history.2023-2026.v1",
    "cn.sse.clock-rule-history.2023-2026.v1",
    "cn.szse.clock-rule-history.2023-2026.v1",
    "cn.bse.clock-rule-history.2021-2026.v1",
    SESSION_CLOCK_ID,
)

_CLOSURE_BINDING_IDS = frozenset(
    {
        "cn.sse.annual-closure-notice.2026.v1",
        "cn.sse.active-closure-schedule.2026.v1",
        "cn.szse.annual-closure-notice.2026.v1",
        "cn.szse.active-closure-schedule.2026.v1",
        "cn.bse.annual-closure-notice.2026.v1",
        "cn.bse.active-closure-schedule.2026.v1",
    }
)
CLOCK_BINDING_IDS = tuple(
    spec.binding_id
    for spec in BINDING_SPECS
    if spec.binding_id not in _CLOSURE_BINDING_IDS
    and not spec.binding_id.endswith(".calendar.v1")
)
if len(CLOCK_BINDING_IDS) != 16:
    raise RuntimeError("session clock must bind exactly 16 source rows")


@dataclass(frozen=True)
class SessionClockEvidenceBundle:
    session_clock: BoundCanonicalArtifact
    sources: tuple[BoundSource, ...]

    def read(self) -> dict[str, Any]:
        if self.session_clock.reference.root_policy != PRIVATE_ROOT_POLICY:
            raise EvidenceV2Error("session-clock evidence bundle must use the private root")
        declared = validate_session_clock(self.session_clock.read())
        rebuilt = build_session_clock(self.sources)
        if rebuilt != declared:
            raise EvidenceV2Error("session-clock evidence bundle does not recompute exactly")
        return declared


def _clock_semantics(binding: Mapping[str, Any]) -> Mapping[str, Any] | None:
    profile = binding["selected_profile"]
    if isinstance(profile, Mapping) and profile.get("domain") == "clock":
        return profile
    projection = binding["semantic_projection"]
    if isinstance(projection, Mapping) and projection.get("schema_version") == (
        "v16.inline-session-clock-rule-projection.v1"
    ):
        return projection
    return None


def _session_clock_payload_from_bindings(
    bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized = [validate_source_binding(value) for value in bindings]
    by_id = {str(value["source_binding_id"]): value for value in normalized}
    if len(by_id) != len(normalized) or tuple(by_id) != CLOCK_BINDING_IDS:
        raise EvidenceV2Error("session clock must bind the exact ordered 16-source registry")
    source_roots = {
        str(Path(EvidenceRef.from_dict(value["raw_ref"]).absolute_path).parent)
        for value in normalized
    }
    if len(source_roots) != 1:
        raise EvidenceV2Error("session-clock source bindings must share one explicit root")

    current_ids = (
        "cn.sse.rule-binary.current.2026.clock.v1",
        "cn.szse.rule-binary.current.2026.clock.v1",
        "cn.bse.rule-inline.current.2026.clock.v1",
    )
    current_semantics = [_clock_semantics(by_id[binding_id]) for binding_id in current_ids]
    if any(value is None for value in current_semantics):
        raise EvidenceV2Error("current session-clock source is not a clock projection")
    for value in current_semantics:
        assert value is not None
        if (
            value["scope_id"] != CLOCK_SCOPE_ID
            or value["segments"] != [dict(item) for item in CLOCK_SEGMENTS]
            or value["excluded_scopes"] != list(CLOCK_EXCLUDED_SCOPES)
            or value["effective_from"] != SESSION_CLOCK_EFFECTIVE_FROM
            or value["effective_to_exclusive"] is not None
            or value["legal_status"] != "effective"
        ):
            raise EvidenceV2Error("three-exchange current session clocks disagree")

    history_pairs = (
        (
            "cn.sse.rule-binary.prior.2023.clock.v1",
            "cn.sse.rule-binary.current.2026.clock.v1",
        ),
        (
            "cn.szse.rule-binary.prior.2023.clock.v1",
            "cn.szse.rule-binary.current.2026.clock.v1",
        ),
        (
            "cn.bse.rule-inline.prior.2021.clock.v1",
            "cn.bse.rule-inline.current.2026.clock.v1",
        ),
    )
    for prior_id, current_id in history_pairs:
        prior = _clock_semantics(by_id[prior_id])
        current = _clock_semantics(by_id[current_id])
        if (
            prior is None
            or current is None
            or prior["effective_to_exclusive"] != current["effective_from"]
        ):
            raise EvidenceV2Error("session-clock legal intervals are not gapless")

    return seal_semantic(
        {
            "schema_version": SESSION_CLOCK_SCHEMA,
            "session_clock_id": SESSION_CLOCK_ID,
            "market": "CN",
            "timezone": SESSION_CLOCK_TIMEZONE,
            "scope_id": CLOCK_SCOPE_ID,
            "segments": [dict(item) for item in CLOCK_SEGMENTS],
            "excluded_scopes": list(CLOCK_EXCLUDED_SCOPES),
            "effective_from": SESSION_CLOCK_EFFECTIVE_FROM,
            "source_document_sets": [
                dict(SOURCE_DOCUMENT_SET_BY_ID[set_id])
                for set_id in CLOCK_SOURCE_SET_IDS
            ],
            "source_bindings": normalized,
            "authority_scope": SESSION_CLOCK_AUTHORITY_SCOPE,
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def build_session_clock(sources: Sequence[BoundSource]) -> dict[str, Any]:
    by_id: dict[str, BoundSource] = {}
    for source in sources:
        if not isinstance(source, BoundSource) or source.binding_id in by_id:
            raise EvidenceV2Error("session-clock bound sources are invalid or duplicated")
        by_id[source.binding_id] = source
    all_binding_ids = {spec.binding_id for spec in BINDING_SPECS}
    if set(by_id) != set(CLOCK_BINDING_IDS) and set(by_id) != all_binding_ids:
        raise EvidenceV2Error("session-clock bound source set is incomplete or has extras")
    if len({str(Path(source.artifact.reference.absolute_path).parent) for source in sources}) != 1:
        raise EvidenceV2Error("session-clock bound sources must share one explicit root")
    bindings = [_binding_from_source(by_id[binding_id]) for binding_id in CLOCK_BINDING_IDS]
    return _session_clock_payload_from_bindings(bindings)


def build_declared_session_clock(root: str | Path) -> dict[str, Any]:
    all_bindings = declared_source_bindings(root)
    by_id = {str(item["source_binding_id"]): item for item in all_bindings}
    return _session_clock_payload_from_bindings(
        [by_id[binding_id] for binding_id in CLOCK_BINDING_IDS]
    )


def validate_session_clock(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "session_clock_id",
        "market",
        "timezone",
        "scope_id",
        "segments",
        "excluded_scopes",
        "effective_from",
        "source_document_sets",
        "source_bindings",
        "authority_scope",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    payload = _exact(payload, fields, label="session clock")
    if (
        payload["schema_version"] != SESSION_CLOCK_SCHEMA
        or payload["session_clock_id"] != SESSION_CLOCK_ID
        or payload["market"] != "CN"
        or payload["timezone"] != SESSION_CLOCK_TIMEZONE
        or payload["scope_id"] != CLOCK_SCOPE_ID
        or payload["segments"] != [dict(item) for item in CLOCK_SEGMENTS]
        or payload["excluded_scopes"] != list(CLOCK_EXCLUDED_SCOPES)
        or payload["effective_from"] != SESSION_CLOCK_EFFECTIVE_FROM
        or payload["authority_scope"] != SESSION_CLOCK_AUTHORITY_SCOPE
    ):
        raise EvidenceV2Error("session-clock frozen values drift")
    if any(
        payload[field] is not False
        for field in (
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
        )
    ):
        raise EvidenceV2Error("session clock must be permanently nonauthorizing")
    sets = [validate_source_document_set(item) for item in payload["source_document_sets"]]
    if [item["source_document_set_id"] for item in sets] != list(
        CLOCK_SOURCE_SET_IDS
    ):
        raise EvidenceV2Error("session-clock source-set order drift")
    bindings = [validate_source_binding(item) for item in payload["source_bindings"]]
    if [item["source_binding_id"] for item in bindings] != list(CLOCK_BINDING_IDS):
        raise EvidenceV2Error("session-clock binding order drift")
    rebuilt = _session_clock_payload_from_bindings(bindings)
    if rebuilt != payload:
        raise EvidenceV2Error("session clock does not recompute exactly")
    return payload


def bind_session_clock_artifact(
    value: Mapping[str, Any],
    *,
    absolute_path: str,
) -> BoundCanonicalArtifact:
    payload = validate_session_clock(value)
    raw = canonical_json_bytes(payload)
    return BoundCanonicalArtifact(
        reference=EvidenceRef(
            schema_version=EVIDENCE_REF_SCHEMA,
            artifact_schema=SESSION_CLOCK_SCHEMA,
            absolute_path=absolute_path,
            byte_sha256=sha256_bytes(raw),
            semantic_sha256=str(payload["semantic_sha256"]),
            root_policy=PRIVATE_ROOT_POLICY,
        ),
        payload=raw,
    )


__all__ = [
    "CLOCK_BINDING_IDS",
    "CLOCK_SOURCE_SET_IDS",
    "SESSION_CLOCK_AUTHORITY_SCOPE",
    "SESSION_CLOCK_EFFECTIVE_FROM",
    "SESSION_CLOCK_ID",
    "SESSION_CLOCK_SCHEMA",
    "SESSION_CLOCK_TIMEZONE",
    "SessionClockEvidenceBundle",
    "bind_session_clock_artifact",
    "build_declared_session_clock",
    "build_session_clock",
    "validate_session_clock",
]
