"""Deterministic, offline contracts for the three-round public search lane."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from typing import Any, Final

from .._core import canonical_bytes, identifier, require_exact_keys
from ._contracts import (
    artifact,
    artifact_ref,
    canonical_url,
    closed_artifact,
    exact_source_ref,
    fail,
    identifiers,
    same,
    text,
    texts,
    when,
)
from .models import (
    CITATION_LEAD_VERSION,
    I5ContractError,
    PUBLIC_CAPABILITY_VERSION,
    PUBLIC_PACKET_VERSION,
    ROLE,
    ROUND_ORDER,
    SEARCH_REQUEST_VERSION,
    SEARCH_RESPONSE_VERSION,
    SEARCH_RUN_VERSION,
    SEARCH_RUN_STATUS_VERSION,
    SEARCH_SOURCE_VERSION,
)

_PACKET_FIELDS: Final = {
    "declassification_evidence_ref",
    "public_identity",
    "public_industry_ids",
    "public_theme_ids",
    "thesis",
    "search_questions",
    "market_data_cutoff",
    "target_knowledge_window",
}
_PUBLIC_IDENTITY_FIELDS: Final = {"company_code", "display_name"}
_WINDOW_FIELDS: Final = {"not_before", "not_after"}
_CAPABILITY_FIELDS: Final = {
    "organization",
    "project",
    "credential_ref",
    "client_namespace",
    "history_namespace",
    "cache_namespace",
    "receipt_store_namespace",
    "endpoint",
    "exact_model",
    "request_mode",
    "system_prompt_sha256",
    "request_schema_sha256",
    "output_schema_sha256",
    "tool_configuration",
    "control_evidence_ref",
    "not_before",
    "expires_at",
    "role",
    "no_fallback",
}
_REQUEST_FIELDS: Final = {
    "packet_ref",
    "capability_ref",
    "round",
    "round_index",
    "query",
    "request_configuration",
    "system_prompt_sha256",
    "request_schema_sha256",
    "output_schema_sha256",
}
_RESPONSE_FIELDS: Final = {
    "request_ref",
    "provider_response_id",
    "response_text",
    "tool_events",
    "output_schema_sha256",
}
PUBLIC_SEARCH_PROMPT: Final = (
    "myquant.i5.public-search-prompt.v1\n"
    "你是怀疑型 AI 投委会的公开检索规划员。只使用已脱密公开包，主动寻找反证、缺口和冲突；"
    "不得请求或推断私有持仓、权重、凭证、内部路径或交易动作。"
)
PUBLIC_SEARCH_PROMPT_SHA256: Final = hashlib.sha256(
    PUBLIC_SEARCH_PROMPT.encode("utf-8")
).hexdigest()
PUBLIC_REQUEST_SCHEMA_BYTES: Final = canonical_bytes(
    {
        "exact_fields": sorted(_REQUEST_FIELDS),
        "name": "myquant.i5.public-search-request-schema.v1",
        "rounds": list(ROUND_ORDER),
        "tool": "web_search",
    }
)
PUBLIC_REQUEST_SCHEMA_SHA256: Final = hashlib.sha256(PUBLIC_REQUEST_SCHEMA_BYTES).hexdigest()
PUBLIC_OUTPUT_SCHEMA_BYTES: Final = canonical_bytes(
    {
        "exact_fields": sorted(_RESPONSE_FIELDS),
        "name": "myquant.i5.public-search-output-schema.v1",
        "required_tool_event": {"status": "COMPLETED", "type": "web_search"},
    }
)
PUBLIC_OUTPUT_SCHEMA_SHA256: Final = hashlib.sha256(PUBLIC_OUTPUT_SCHEMA_BYTES).hexdigest()
_SOURCE_FIELDS: Final = {
    "request_ref",
    "provider_response_id",
    "url",
    "title",
    "publisher",
    "publication_hint",
    "media_kind",
    "status",
}
_CITATION_FIELDS: Final = {
    "provider_response_id",
    "source_ref",
    "citation_text",
    "citation_url",
}
_RUN_FIELDS: Final = {
    "packet_ref",
    "capability_ref",
    "evidence_collection_started_at",
    "search_completed_at",
    "round_rows",
    "status",
}
_ROUND_ROW_FIELDS: Final = {
    "round",
    "round_index",
    "request_ref",
    "response_ref",
    "source_refs",
    "citation_lead_refs",
}
_RUN_STATUS_FIELDS: Final = {
    "packet_ref",
    "capability_ref",
    "evidence_collection_started_at",
    "status_recorded_at",
    "status",
    "round_rows",
}
_STATUS_ROW_FIELDS: Final = {
    "round",
    "round_index",
    "status",
    "request_ref",
    "response_ref",
    "source_refs",
    "citation_lead_refs",
    "failure_code",
}


def _assert_no_canary(value: str, canaries: Sequence[str], *, label: str) -> None:
    for canary in canaries:
        if type(canary) is not str or not canary:
            fail("privacy canaries must be nonempty strings")
        if canary in value:
            fail(f"{label} contains a private canary")


def build_declassified_public_packet(
    *,
    company_code: str,
    display_name: str,
    public_industry_ids: Sequence[str],
    public_theme_ids: Sequence[str],
    thesis: str,
    search_questions: Sequence[str],
    market_data_cutoff: str,
    target_knowledge_not_before: str,
    target_knowledge_not_after: str,
    created_at: str,
    declassification_evidence_ref: Mapping[str, Any],
    privacy_canaries: Sequence[str] = (),
) -> dict[str, Any]:
    created = when(created_at, label="created_at")
    cutoff = when(market_data_cutoff, label="market_data_cutoff")
    not_before = when(target_knowledge_not_before, label="target_knowledge_not_before")
    not_after = when(target_knowledge_not_after, label="target_knowledge_not_after")
    if not cutoff <= created <= not_before <= not_after:
        fail("public packet timeline is invalid")
    declassification_ref = exact_source_ref(
        declassification_evidence_ref,
        label="declassification_evidence_ref",
    )
    if declassification_ref["available_at"] > created or declassification_ref["cutoff"] > created:
        fail("declassification evidence is future-dated")
    name = text(display_name, label="display_name")
    thesis_text = text(thesis, label="thesis")
    questions = texts(search_questions, label="search_questions", minimum=1, maximum=32)
    for label, value in (
        ("company_code", company_code),
        ("display_name", name),
        ("thesis", thesis_text),
        *((f"search_questions[{index}]", value) for index, value in enumerate(questions)),
    ):
        _assert_no_canary(value, privacy_canaries, label=label)
    return artifact(
        version=PUBLIC_PACKET_VERSION,
        identity_field="packet_id",
        timestamp_value=created,
        payload={
            "declassification_evidence_ref": declassification_ref,
            "public_identity": {
                "company_code": identifier(company_code, label="company_code"),
                "display_name": name,
            },
            "public_industry_ids": identifiers(public_industry_ids, label="public_industry_ids"),
            "public_theme_ids": identifiers(public_theme_ids, label="public_theme_ids"),
            "thesis": thesis_text,
            "search_questions": questions,
            "market_data_cutoff": cutoff,
            "target_knowledge_window": {
                "not_before": not_before,
                "not_after": not_after,
            },
        },
    )


def validate_declassified_public_packet(
    document: Mapping[str, Any],
    *,
    declassification_evidence_ref: Mapping[str, Any],
    privacy_canaries: Sequence[str] = (),
) -> dict[str, Any]:
    row = closed_artifact(
        document,
        version=PUBLIC_PACKET_VERSION,
        identity_field="packet_id",
        payload_fields=_PACKET_FIELDS,
    )
    require_exact_keys(row["public_identity"], _PUBLIC_IDENTITY_FIELDS, label="public_identity")
    require_exact_keys(
        row["target_knowledge_window"], _WINDOW_FIELDS, label="target_knowledge_window"
    )
    authorized_ref = exact_source_ref(
        declassification_evidence_ref,
        label="authorized_declassification_evidence_ref",
    )
    if row["declassification_evidence_ref"] != authorized_ref:
        fail("public packet declassification evidence differs from authorized closure")
    expected = build_declassified_public_packet(
        company_code=row["public_identity"]["company_code"],
        display_name=row["public_identity"]["display_name"],
        public_industry_ids=row["public_industry_ids"],
        public_theme_ids=row["public_theme_ids"],
        thesis=row["thesis"],
        search_questions=row["search_questions"],
        market_data_cutoff=row["market_data_cutoff"],
        target_knowledge_not_before=row["target_knowledge_window"]["not_before"],
        target_knowledge_not_after=row["target_knowledge_window"]["not_after"],
        created_at=row["timestamp"],
        declassification_evidence_ref=row["declassification_evidence_ref"],
        privacy_canaries=privacy_canaries,
    )
    same(row, expected, label="public packet")
    return expected


def build_public_search_capability(
    *,
    organization: str,
    project: str,
    credential_ref: str,
    client_namespace: str,
    history_namespace: str,
    cache_namespace: str,
    receipt_store_namespace: str,
    endpoint: str,
    exact_model: str,
    control_evidence_ref: Mapping[str, Any],
    not_before: str,
    expires_at: str,
    created_at: str,
) -> dict[str, Any]:
    created = when(created_at, label="created_at")
    start = when(not_before, label="not_before")
    end = when(expires_at, label="expires_at")
    if not created <= start < end:
        fail("public capability validity interval is invalid")
    endpoint_url = canonical_url(endpoint, label="endpoint")
    if not endpoint_url.startswith("https://"):
        fail("public search endpoint must use HTTPS")
    evidence = exact_source_ref(control_evidence_ref, label="control_evidence_ref")
    if evidence["available_at"] > start:
        fail("public capability evidence was unavailable at activation")
    return artifact(
        version=PUBLIC_CAPABILITY_VERSION,
        identity_field="capability_id",
        timestamp_value=created,
        payload={
            "organization": identifier(organization, label="organization"),
            "project": identifier(project, label="project"),
            "credential_ref": identifier(credential_ref, label="credential_ref"),
            "client_namespace": identifier(client_namespace, label="client_namespace"),
            "history_namespace": identifier(history_namespace, label="history_namespace"),
            "cache_namespace": identifier(cache_namespace, label="cache_namespace"),
            "receipt_store_namespace": identifier(
                receipt_store_namespace, label="receipt_store_namespace"
            ),
            "endpoint": endpoint_url,
            "exact_model": identifier(exact_model, label="exact_model"),
            "request_mode": "RESPONSES_WEB_SEARCH",
            "system_prompt_sha256": PUBLIC_SEARCH_PROMPT_SHA256,
            "request_schema_sha256": PUBLIC_REQUEST_SCHEMA_SHA256,
            "output_schema_sha256": PUBLIC_OUTPUT_SCHEMA_SHA256,
            "tool_configuration": [{"type": "web_search"}],
            "control_evidence_ref": evidence,
            "not_before": start,
            "expires_at": end,
            "role": ROLE,
            "no_fallback": True,
        },
    )


def validate_public_search_capability(document: Mapping[str, Any]) -> dict[str, Any]:
    row = closed_artifact(
        document,
        version=PUBLIC_CAPABILITY_VERSION,
        identity_field="capability_id",
        payload_fields=_CAPABILITY_FIELDS,
    )
    expected = build_public_search_capability(
        organization=row["organization"],
        project=row["project"],
        credential_ref=row["credential_ref"],
        client_namespace=row["client_namespace"],
        history_namespace=row["history_namespace"],
        cache_namespace=row["cache_namespace"],
        receipt_store_namespace=row["receipt_store_namespace"],
        endpoint=row["endpoint"],
        exact_model=row["exact_model"],
        control_evidence_ref=row["control_evidence_ref"],
        not_before=row["not_before"],
        expires_at=row["expires_at"],
        created_at=row["timestamp"],
    )
    same(row, expected, label="public capability")
    return expected


def build_search_request(
    *,
    packet: Mapping[str, Any],
    capability: Mapping[str, Any],
    round_name: str,
    round_index: int,
    query: str,
    requested_at: str,
) -> dict[str, Any]:
    capability_doc = validate_public_search_capability(capability)
    packet_doc = validate_declassified_public_packet(
        packet,
        declassification_evidence_ref=capability_doc["control_evidence_ref"],
    )
    requested = when(requested_at, label="requested_at")
    if round_name not in ROUND_ORDER or round_index != ROUND_ORDER.index(round_name) + 1:
        fail("search round or index is invalid")
    if not capability_doc["not_before"] <= requested <= capability_doc["expires_at"]:
        fail("search request is outside capability validity")
    window = packet_doc["target_knowledge_window"]
    if not window["not_before"] <= requested <= window["not_after"]:
        fail("search request is outside target knowledge window")
    configuration = {
        "background": False,
        "endpoint": capability_doc["endpoint"],
        "model": capability_doc["exact_model"],
        "previous_response_id": None,
        "store": False,
        "tool_choice": "required",
        "tools": [{"type": "web_search"}],
    }
    return artifact(
        version=SEARCH_REQUEST_VERSION,
        identity_field="search_request_id",
        timestamp_value=requested,
        payload={
            "packet_ref": artifact_ref(packet_doc, identity_field="packet_id"),
            "capability_ref": artifact_ref(capability_doc, identity_field="capability_id"),
            "round": round_name,
            "round_index": round_index,
            "query": text(query, label="query"),
            "request_configuration": configuration,
            "system_prompt_sha256": capability_doc["system_prompt_sha256"],
            "request_schema_sha256": capability_doc["request_schema_sha256"],
            "output_schema_sha256": capability_doc["output_schema_sha256"],
        },
    )


def validate_search_request(
    document: Mapping[str, Any],
    *,
    packet: Mapping[str, Any],
    capability: Mapping[str, Any],
) -> dict[str, Any]:
    row = closed_artifact(
        document,
        version=SEARCH_REQUEST_VERSION,
        identity_field="search_request_id",
        payload_fields=_REQUEST_FIELDS,
    )
    expected = build_search_request(
        packet=packet,
        capability=capability,
        round_name=row["round"],
        round_index=row["round_index"],
        query=row["query"],
        requested_at=row["timestamp"],
    )
    same(row, expected, label="search request")
    return expected


def build_search_source(
    *,
    request: Mapping[str, Any],
    provider_response_id: str,
    url: str,
    title: str,
    publisher: str | None,
    publication_hint: str | None,
    media_kind: str,
    discovered_at: str,
) -> dict[str, Any]:
    if media_kind not in {"HTML", "PDF", "UNKNOWN"}:
        fail("search source media_kind is invalid")
    return artifact(
        version=SEARCH_SOURCE_VERSION,
        identity_field="source_id",
        timestamp_value=when(discovered_at, label="discovered_at"),
        payload={
            "request_ref": artifact_ref(request, identity_field="search_request_id"),
            "provider_response_id": identifier(provider_response_id, label="provider_response_id"),
            "url": canonical_url(url, label="source.url"),
            "title": text(title, label="source.title"),
            "publisher": None if publisher is None else text(publisher, label="publisher"),
            "publication_hint": (
                None
                if publication_hint is None
                else text(publication_hint, label="publication_hint")
            ),
            "media_kind": media_kind,
            "status": "DISCOVERED",
        },
    )


def validate_search_source(
    document: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    provider_response_id: str,
) -> dict[str, Any]:
    row = closed_artifact(
        document,
        version=SEARCH_SOURCE_VERSION,
        identity_field="source_id",
        payload_fields=_SOURCE_FIELDS,
    )
    expected = build_search_source(
        request=request,
        provider_response_id=provider_response_id,
        url=row["url"],
        title=row["title"],
        publisher=row["publisher"],
        publication_hint=row["publication_hint"],
        media_kind=row["media_kind"],
        discovered_at=row["timestamp"],
    )
    same(row, expected, label="search source")
    return expected


def build_citation_lead(
    *,
    provider_response_id: str,
    source: Mapping[str, Any],
    citation_text: str,
    citation_url: str,
    cited_at: str,
) -> dict[str, Any]:
    return artifact(
        version=CITATION_LEAD_VERSION,
        identity_field="citation_lead_id",
        timestamp_value=when(cited_at, label="cited_at"),
        payload={
            "provider_response_id": identifier(provider_response_id, label="provider_response_id"),
            "source_ref": artifact_ref(source, identity_field="source_id"),
            "citation_text": text(citation_text, label="citation_text"),
            "citation_url": canonical_url(citation_url, label="citation_url"),
        },
    )


def validate_citation_lead(
    document: Mapping[str, Any],
    *,
    provider_response_id: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    row = closed_artifact(
        document,
        version=CITATION_LEAD_VERSION,
        identity_field="citation_lead_id",
        payload_fields=_CITATION_FIELDS,
    )
    expected = build_citation_lead(
        provider_response_id=provider_response_id,
        source=source,
        citation_text=row["citation_text"],
        citation_url=row["citation_url"],
        cited_at=row["timestamp"],
    )
    same(row, expected, label="citation lead")
    return expected


def build_search_response(
    *,
    request: Mapping[str, Any],
    provider_response_id: str,
    response_text: str,
    responded_at: str,
) -> dict[str, Any]:
    if (
        request.get("system_prompt_sha256") != PUBLIC_SEARCH_PROMPT_SHA256
        or request.get("request_schema_sha256") != PUBLIC_REQUEST_SCHEMA_SHA256
        or request.get("output_schema_sha256") != PUBLIC_OUTPUT_SCHEMA_SHA256
    ):
        fail("search request prompt/schema binding is invalid")
    requested_at = when(request["timestamp"], label="request.timestamp")
    responded = when(responded_at, label="responded_at")
    if responded < requested_at:
        fail("search response predates request")
    return artifact(
        version=SEARCH_RESPONSE_VERSION,
        identity_field="search_response_id",
        timestamp_value=responded,
        payload={
            "request_ref": artifact_ref(request, identity_field="search_request_id"),
            "provider_response_id": identifier(provider_response_id, label="provider_response_id"),
            "response_text": text(response_text, label="response_text"),
            "tool_events": [{"status": "COMPLETED", "type": "web_search"}],
            "output_schema_sha256": PUBLIC_OUTPUT_SCHEMA_SHA256,
        },
    )


def validate_search_response(
    document: Mapping[str, Any], *, request: Mapping[str, Any]
) -> dict[str, Any]:
    row = closed_artifact(
        document,
        version=SEARCH_RESPONSE_VERSION,
        identity_field="search_response_id",
        payload_fields=_RESPONSE_FIELDS,
    )
    expected = build_search_response(
        request=request,
        provider_response_id=row["provider_response_id"],
        response_text=row["response_text"],
        responded_at=row["timestamp"],
    )
    same(row, expected, label="search response")
    return expected


def _validate_round_bundle(
    value: Mapping[str, Any],
    *,
    packet: Mapping[str, Any],
    capability: Mapping[str, Any],
    expected_name: str,
    expected_index: int,
) -> dict[str, Any]:
    require_exact_keys(value, {"request", "response", "sources", "citation_leads"}, label="round")
    request = validate_search_request(value["request"], packet=packet, capability=capability)
    if request["round"] != expected_name or request["round_index"] != expected_index:
        fail("search run round order is invalid")
    response = validate_search_response(value["response"], request=request)
    if not value["sources"] or not value["citation_leads"]:
        fail("every search round requires source and citation lead receipts")
    sources = [
        validate_search_source(
            row,
            request=request,
            provider_response_id=response["provider_response_id"],
        )
        for row in value["sources"]
    ]
    citations_by_source = {row["source_ref"]["artifact_id"] for row in value["citation_leads"]}
    citations = []
    for source in sources:
        source_id = source["source_id"]
        for lead in value["citation_leads"]:
            if lead["source_ref"]["artifact_id"] == source_id:
                citations.append(
                    validate_citation_lead(
                        lead,
                        provider_response_id=response["provider_response_id"],
                        source=source,
                    )
                )
    if citations_by_source - {source["source_id"] for source in sources}:
        fail("citation lead refers to an unknown source")
    timeline_rows = [*sources, *citations]
    if any(
        not request["timestamp"] <= row["timestamp"] <= response["timestamp"]
        for row in timeline_rows
    ):
        fail("search source/citation timeline is invalid")
    return {
        "request": request,
        "response": response,
        "sources": sources,
        "citation_leads": citations,
    }


def build_search_run(
    *,
    packet: Mapping[str, Any],
    capability: Mapping[str, Any],
    round_bundles: Sequence[Mapping[str, Any]],
    evidence_collection_started_at: str,
    search_completed_at: str,
) -> dict[str, Any]:
    capability_doc = validate_public_search_capability(capability)
    packet_doc = validate_declassified_public_packet(
        packet,
        declassification_evidence_ref=capability_doc["control_evidence_ref"],
    )
    started = when(evidence_collection_started_at, label="evidence_collection_started_at")
    completed = when(search_completed_at, label="search_completed_at")
    if len(round_bundles) != 3:
        fail("new I5 search run must contain exactly three rounds")
    validated = [
        _validate_round_bundle(
            value,
            packet=packet_doc,
            capability=capability_doc,
            expected_name=ROUND_ORDER[index],
            expected_index=index + 1,
        )
        for index, value in enumerate(round_bundles)
    ]
    times = [row["request"]["timestamp"] for row in validated]
    times.extend(row["response"]["timestamp"] for row in validated)
    if started > min(times) or completed < max(times) or started < packet_doc["market_data_cutoff"]:
        fail("search run timeline is invalid")
    rows = []
    for index, value in enumerate(validated):
        rows.append(
            {
                "round": ROUND_ORDER[index],
                "round_index": index + 1,
                "request_ref": artifact_ref(value["request"], identity_field="search_request_id"),
                "response_ref": artifact_ref(
                    value["response"], identity_field="search_response_id"
                ),
                "source_refs": sorted(
                    [
                        artifact_ref(source, identity_field="source_id")
                        for source in value["sources"]
                    ],
                    key=lambda row: row["artifact_id"],
                ),
                "citation_lead_refs": sorted(
                    [
                        artifact_ref(lead, identity_field="citation_lead_id")
                        for lead in value["citation_leads"]
                    ],
                    key=lambda row: row["artifact_id"],
                ),
            }
        )
    return artifact(
        version=SEARCH_RUN_VERSION,
        identity_field="search_run_id",
        timestamp_value=completed,
        payload={
            "packet_ref": artifact_ref(packet_doc, identity_field="packet_id"),
            "capability_ref": artifact_ref(capability_doc, identity_field="capability_id"),
            "evidence_collection_started_at": started,
            "search_completed_at": completed,
            "round_rows": rows,
            "status": "COMPLETE",
        },
    )


def validate_search_run(
    document: Mapping[str, Any],
    *,
    packet: Mapping[str, Any],
    capability: Mapping[str, Any],
    round_bundles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    row = closed_artifact(
        document,
        version=SEARCH_RUN_VERSION,
        identity_field="search_run_id",
        payload_fields=_RUN_FIELDS,
    )
    for index, value in enumerate(row["round_rows"]):
        require_exact_keys(value, _ROUND_ROW_FIELDS, label=f"round_rows[{index}]")
    expected = build_search_run(
        packet=packet,
        capability=capability,
        round_bundles=round_bundles,
        evidence_collection_started_at=row["evidence_collection_started_at"],
        search_completed_at=row["search_completed_at"],
    )
    same(row, expected, label="search run")
    return expected


def _completed_status_row(
    *,
    round_name: str,
    round_index: int,
    bundle_value: Mapping[str, Any],
    packet: Mapping[str, Any],
    capability: Mapping[str, Any],
    recorded: str,
) -> dict[str, Any]:
    bundle = _validate_round_bundle(
        bundle_value,
        packet=packet,
        capability=capability,
        expected_name=round_name,
        expected_index=round_index,
    )
    if bundle["response"]["timestamp"] > recorded:
        fail("completed round is later than run status")
    return {
        "round": round_name,
        "round_index": round_index,
        "status": "COMPLETED",
        "request_ref": artifact_ref(bundle["request"], identity_field="search_request_id"),
        "response_ref": artifact_ref(bundle["response"], identity_field="search_response_id"),
        "source_refs": sorted(
            [artifact_ref(row, identity_field="source_id") for row in bundle["sources"]],
            key=lambda row: row["artifact_id"],
        ),
        "citation_lead_refs": sorted(
            [
                artifact_ref(row, identity_field="citation_lead_id")
                for row in bundle["citation_leads"]
            ],
            key=lambda row: row["artifact_id"],
        ),
        "failure_code": None,
    }


def _incomplete_status_row(
    *, round_name: str, round_index: int, status: str, failure_code: Any
) -> dict[str, Any]:
    if type(failure_code) is not str or not failure_code:
        fail("failed or missing round requires a failure code")
    try:
        reason = identifier(failure_code, label="failure_code")
    except Exception as exc:
        raise I5ContractError(str(exc)) from exc
    return {
        "round": round_name,
        "round_index": round_index,
        "status": status,
        "request_ref": None,
        "response_ref": None,
        "source_refs": [],
        "citation_lead_refs": [],
        "failure_code": reason,
    }


def _search_status_row(
    *,
    round_name: str,
    round_index: int,
    outcome: Mapping[str, Any],
    completed_round_bundles: Mapping[str, Mapping[str, Any]],
    packet: Mapping[str, Any],
    capability: Mapping[str, Any],
    recorded: str,
) -> dict[str, Any]:
    row = require_exact_keys(
        outcome,
        {"status", "failure_code"},
        label=f"round_statuses.{round_name}",
    )
    status = row["status"]
    if status == "COMPLETED":
        if row["failure_code"] is not None or round_name not in completed_round_bundles:
            fail("completed round lacks its exact closure")
        return _completed_status_row(
            round_name=round_name,
            round_index=round_index,
            bundle_value=completed_round_bundles[round_name],
            packet=packet,
            capability=capability,
            recorded=recorded,
        )
    if status not in {"FAILED", "MISSING"} or round_name in completed_round_bundles:
        fail("failed or missing round closure is invalid")
    return _incomplete_status_row(
        round_name=round_name,
        round_index=round_index,
        status=status,
        failure_code=row["failure_code"],
    )


def build_search_run_status(
    *,
    packet: Mapping[str, Any],
    capability: Mapping[str, Any],
    round_statuses: Mapping[str, Mapping[str, Any]],
    completed_round_bundles: Mapping[str, Mapping[str, Any]],
    evidence_collection_started_at: str,
    status_recorded_at: str,
) -> dict[str, Any]:
    """Record a failed/missing three-round topology without creating a complete run."""

    capability_doc = validate_public_search_capability(capability)
    packet_doc = validate_declassified_public_packet(
        packet,
        declassification_evidence_ref=capability_doc["control_evidence_ref"],
    )
    if type(round_statuses) is not dict or set(round_statuses) != set(ROUND_ORDER):
        fail("search run status must describe exactly three rounds")
    if type(completed_round_bundles) is not dict:
        fail("completed_round_bundles must be an object")
    started = when(evidence_collection_started_at, label="evidence_collection_started_at")
    recorded = when(status_recorded_at, label="status_recorded_at")
    if started < packet_doc["market_data_cutoff"] or recorded < started:
        fail("search run status timeline is invalid")
    rows = []
    for index, round_name in enumerate(ROUND_ORDER):
        rows.append(
            _search_status_row(
                round_name=round_name,
                round_index=index + 1,
                outcome=round_statuses[round_name],
                completed_round_bundles=completed_round_bundles,
                packet=packet_doc,
                capability=capability_doc,
                recorded=recorded,
            )
        )
    complete_names = {row["round"] for row in rows if row["status"] == "COMPLETED"}
    if set(completed_round_bundles) != complete_names:
        fail("completed round closure set is not exact")
    if len(complete_names) == 3:
        fail("complete three-round topology must use build_search_run")
    return artifact(
        version=SEARCH_RUN_STATUS_VERSION,
        identity_field="search_run_status_id",
        timestamp_value=recorded,
        payload={
            "packet_ref": artifact_ref(packet_doc, identity_field="packet_id"),
            "capability_ref": artifact_ref(capability_doc, identity_field="capability_id"),
            "evidence_collection_started_at": started,
            "status_recorded_at": recorded,
            "status": "INCOMPLETE",
            "round_rows": rows,
        },
    )


def validate_search_run_status(
    document: Mapping[str, Any],
    *,
    packet: Mapping[str, Any],
    capability: Mapping[str, Any],
    round_statuses: Mapping[str, Mapping[str, Any]],
    completed_round_bundles: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    row = closed_artifact(
        document,
        version=SEARCH_RUN_STATUS_VERSION,
        identity_field="search_run_status_id",
        payload_fields=_RUN_STATUS_FIELDS,
    )
    for index, value in enumerate(row["round_rows"]):
        require_exact_keys(value, _STATUS_ROW_FIELDS, label=f"round_rows[{index}]")
    expected = build_search_run_status(
        packet=packet,
        capability=capability,
        round_statuses=round_statuses,
        completed_round_bundles=completed_round_bundles,
        evidence_collection_started_at=row["evidence_collection_started_at"],
        status_recorded_at=row["status_recorded_at"],
    )
    same(row, expected, label="search run status")
    return expected


__all__ = [
    "PUBLIC_OUTPUT_SCHEMA_SHA256",
    "PUBLIC_REQUEST_SCHEMA_SHA256",
    "PUBLIC_SEARCH_PROMPT",
    "PUBLIC_SEARCH_PROMPT_SHA256",
    "build_citation_lead",
    "build_declassified_public_packet",
    "build_public_search_capability",
    "build_search_request",
    "build_search_response",
    "build_search_run",
    "build_search_run_status",
    "build_search_source",
    "validate_citation_lead",
    "validate_declassified_public_packet",
    "validate_public_search_capability",
    "validate_search_request",
    "validate_search_response",
    "validate_search_run",
    "validate_search_run_status",
    "validate_search_source",
]
