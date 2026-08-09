"""SSRF-safe capture policy and offline validation contracts for I5."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ipaddress
from typing import Any, Final

from .._core import canonical_bytes, identifier, require_exact_keys, sha256
from ._contracts import (
    artifact,
    artifact_ref,
    canonical_url,
    closed_artifact,
    digest_bytes,
    exact_source_ref,
    fail,
    same,
    text,
    texts,
    validate_digest,
    when,
)
from .models import (
    CAPTURE_POLICY_VERSION,
    CAPTURE_RECEIPT_VERSION,
    I5ContractError,
    SOURCE_CLASSES,
    VALIDATED_FACT_VERSION,
)
from .public_search import validate_search_source

_MIB: Final = 1024 * 1024
_ALLOWED_MIME: Final = frozenset({"text/html", "application/xhtml+xml"})
_ALLOWED_CHARSETS: Final = frozenset({"UTF-8", "GB18030", "GBK", "BIG5"})
_ALLOWED_ENCODINGS: Final = frozenset({"identity", "gzip", "deflate", "br"})
_POLICY_FIELDS: Final = {
    "schemes",
    "ports",
    "max_redirects",
    "connect_timeout_seconds",
    "read_timeout_seconds",
    "total_timeout_seconds",
    "max_header_bytes",
    "max_compressed_bytes",
    "max_decoded_bytes",
    "max_decompression_ratio",
    "max_html_nodes",
    "max_html_depth",
    "allowed_mime_types",
    "allowed_charsets",
    "allowed_content_encodings",
    "parser_identity",
    "parser_version",
    "parser_options_sha256",
    "decoder_manifest_sha256",
    "transport_policy_ref",
    "security_requirements",
}
_RECEIPT_FIELDS: Final = {
    "source_ref",
    "policy_ref",
    "status",
    "requested_url",
    "final_url",
    "redirect_chain",
    "response_status",
    "selected_headers",
    "content_encoding",
    "mime_type",
    "charset",
    "compressed_entity",
    "decoded_entity",
    "parser_input",
    "parser_identity",
    "parser_version",
    "parser_options_sha256",
    "html_node_count",
    "html_max_depth",
    "publication_evidence_ref",
    "transport_attestation_ref",
    "security_observations",
    "reason_codes",
}
_HOP_FIELDS: Final = {
    "url",
    "resolved_addresses",
    "peer_ip",
    "status_code",
    "location",
}
_OBSERVATION_FIELDS: Final = {
    "redirect_chain",
    "selected_headers",
    "content_encoding",
    "mime_type",
    "charset",
    "html_node_count",
    "html_max_depth",
    "header_bytes_total",
    "connect_timeout_seconds",
    "read_timeout_seconds",
    "total_timeout_seconds",
    "transfer_decoding_complete",
    "proxy_environment_used",
    "cookies_sent",
    "authorization_sent",
    "library_reresolution",
    "hostname_sni_preserved",
}
_FACT_FIELDS: Final = {
    "capture_ref",
    "subject_id",
    "classification",
    "claim",
    "locator",
    "source_url",
    "source_class",
    "canonical_source_id",
    "original_source_id",
    "syndication_group_id",
    "conflict_status",
    "captured_at",
    "publication_evidence_ref",
    "prompt_injection",
}


def build_capture_policy(
    *,
    parser_identity: str,
    parser_version: str,
    parser_options_sha256: str,
    decoder_manifest_sha256: str,
    transport_policy_ref: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    return artifact(
        version=CAPTURE_POLICY_VERSION,
        identity_field="capture_policy_id",
        timestamp_value=when(created_at, label="created_at"),
        payload={
            "schemes": ["http", "https"],
            "ports": [80, 443],
            "max_redirects": 5,
            "connect_timeout_seconds": 5,
            "read_timeout_seconds": 15,
            "total_timeout_seconds": 30,
            "max_header_bytes": 64 * 1024,
            "max_compressed_bytes": 5 * _MIB,
            "max_decoded_bytes": 20 * _MIB,
            "max_decompression_ratio": 20,
            "max_html_nodes": 100_000,
            "max_html_depth": 128,
            "allowed_mime_types": sorted(_ALLOWED_MIME),
            "allowed_charsets": sorted(_ALLOWED_CHARSETS),
            "allowed_content_encodings": sorted(_ALLOWED_ENCODINGS),
            "parser_identity": identifier(parser_identity, label="parser_identity"),
            "parser_version": identifier(parser_version, label="parser_version"),
            "parser_options_sha256": sha256(parser_options_sha256, label="parser_options_sha256"),
            "decoder_manifest_sha256": sha256(
                decoder_manifest_sha256, label="decoder_manifest_sha256"
            ),
            "transport_policy_ref": exact_source_ref(
                transport_policy_ref, label="transport_policy_ref"
            ),
            "security_requirements": {
                "all_dns_addresses_public": True,
                "peer_ip_must_match_validated_dns": True,
                "revalidate_every_redirect": True,
                "preserve_hostname_and_sni": True,
                "library_reresolution_forbidden": True,
                "proxy_environment_forbidden": True,
                "cookies_forbidden": True,
                "authorization_headers_forbidden": True,
                "charset_guessing_forbidden": True,
            },
        },
    )


def validate_capture_policy(document: Mapping[str, Any]) -> dict[str, Any]:
    row = closed_artifact(
        document,
        version=CAPTURE_POLICY_VERSION,
        identity_field="capture_policy_id",
        payload_fields=_POLICY_FIELDS,
    )
    expected = build_capture_policy(
        parser_identity=row["parser_identity"],
        parser_version=row["parser_version"],
        parser_options_sha256=row["parser_options_sha256"],
        decoder_manifest_sha256=row["decoder_manifest_sha256"],
        transport_policy_ref=row["transport_policy_ref"],
        created_at=row["timestamp"],
    )
    same(row, expected, label="capture policy")
    return expected


def _public_ip(value: Any, *, label: str) -> str:
    if type(value) is not str or value != value.strip():
        fail(f"{label} IP address is invalid")
    try:
        parsed = ipaddress.ip_address(value)
    except ValueError as exc:
        raise ValueError(f"{label} IP address is invalid") from exc
    if not parsed.is_global or parsed.is_multicast or parsed.is_reserved:
        fail(f"{label} IP address is not public")
    if parsed.compressed != value:
        fail(f"{label} IP address is not canonical")
    return value


def _validate_hop(value: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    row = require_exact_keys(value, _HOP_FIELDS, label=f"redirect_chain[{index}]")
    addresses = [
        _public_ip(address, label=f"redirect_chain[{index}].resolved_addresses")
        for address in row["resolved_addresses"]
    ]
    if not addresses or len(addresses) != len(set(addresses)):
        fail("DNS address set is empty or duplicated")
    addresses = sorted(addresses, key=lambda address: ipaddress.ip_address(address).packed)
    peer = _public_ip(row["peer_ip"], label=f"redirect_chain[{index}].peer_ip")
    if peer not in addresses:
        fail("actual peer IP is outside validated DNS set")
    if type(row["status_code"]) is not int or not 100 <= row["status_code"] <= 599:
        fail("HTTP status is invalid")
    location = row["location"]
    if location is not None:
        location = canonical_url(location, label=f"redirect_chain[{index}].location")
    return {
        "url": canonical_url(row["url"], label=f"redirect_chain[{index}].url"),
        "resolved_addresses": addresses,
        "peer_ip": peer,
        "status_code": row["status_code"],
        "location": location,
    }


def _validate_redirect_chain(
    value: Sequence[Mapping[str, Any]], *, requested_url: str
) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        fail("redirect_chain must be a sequence")
    if not 1 <= len(value) <= 6:
        fail("redirect_chain exceeds five redirects")
    rows = [_validate_hop(row, index=index) for index, row in enumerate(value)]
    if rows[0]["url"] != requested_url:
        fail("redirect chain does not begin at requested URL")
    for index, row in enumerate(rows[:-1]):
        if not 300 <= row["status_code"] <= 399 or row["location"] != rows[index + 1]["url"]:
            fail("redirect hop is not exact")
    if rows[-1]["location"] is not None or 300 <= rows[-1]["status_code"] <= 399:
        fail("final response cannot remain a redirect")
    return rows


def _validate_headers(value: Mapping[str, Any]) -> dict[str, str]:
    if type(value) is not dict:
        fail("selected_headers must be an object")
    headers: dict[str, str] = {}
    for key, raw in value.items():
        if type(key) is not str or key != key.lower() or not key.isascii():
            fail("selected header names must be lowercase ASCII")
        headers[key] = text(raw, label=f"selected_headers.{key}")
    if len(canonical_bytes(headers)) > 64 * 1024:
        fail("selected headers exceed 64 KiB")
    return headers


def _validate_security_observation(observation: Mapping[str, Any]) -> None:
    required_false = (
        "proxy_environment_used",
        "cookies_sent",
        "authorization_sent",
        "library_reresolution",
    )
    if any(observation[field] is not False for field in required_false):
        fail("capture transport leaked authority or bypassed DNS binding")
    if observation["hostname_sni_preserved"] is not True:
        fail("capture transport did not preserve hostname/SNI")


def _validate_entity_sizes(
    *,
    compressed_entity: bytes,
    decoded_entity: bytes,
    parser_input: bytes,
    charset: str,
) -> None:
    if len(compressed_entity) > 5 * _MIB or len(decoded_entity) > 20 * _MIB:
        fail("captured entity exceeds size limit")
    ratio = len(decoded_entity) / max(1, len(compressed_entity))
    if ratio > 20:
        fail("captured entity exceeds decompression ratio")
    try:
        expected_parser_input = decoded_entity.decode(charset, errors="strict").encode("utf-8")
    except (LookupError, UnicodeDecodeError) as exc:
        raise I5ContractError("decoded entity is invalid for the declared charset") from exc
    if parser_input != expected_parser_input:
        fail("parser input must be the exact strict UTF-8 projection of decoded entity bytes")


def _validate_response_representation(
    observation: Mapping[str, Any], *, headers: Mapping[str, str]
) -> None:
    if observation["content_encoding"] not in _ALLOWED_ENCODINGS:
        fail("content encoding is forbidden")
    if observation["mime_type"] not in _ALLOWED_MIME:
        fail("MIME type is forbidden")
    if observation["charset"] not in _ALLOWED_CHARSETS:
        fail("charset is missing, conflicting, guessed, or forbidden")
    content_type = headers.get("content-type", "")
    if content_type.lower().count("charset=") != 1:
        fail("content-type charset is missing or conflicting")
    if observation["mime_type"].lower() not in content_type.lower():
        fail("content-type MIME conflicts with selected MIME")
    declared_charset = content_type.lower().split("charset=", 1)[1].strip()
    if declared_charset.upper() != observation["charset"]:
        fail("content-type charset conflicts with selected charset")
    encoding_header = headers.get("content-encoding", "identity")
    if encoding_header != observation["content_encoding"]:
        fail("content-encoding header conflicts with selected decoder")


def _validate_parser_limits(observation: Mapping[str, Any]) -> None:
    node_count = observation["html_node_count"]
    if type(node_count) is not int or not 0 <= node_count <= 100_000:
        fail("HTML node count is invalid")
    max_depth = observation["html_max_depth"]
    if type(max_depth) is not int or not 0 <= max_depth <= 128:
        fail("HTML depth is invalid")


def _validate_capture_observation(
    observation: Mapping[str, Any], *, requested_url: str
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    row = require_exact_keys(observation, _OBSERVATION_FIELDS, label="transport_observation")
    _validate_security_observation(row)
    chain = _validate_redirect_chain(row["redirect_chain"], requested_url=requested_url)
    headers = _validate_headers(row["selected_headers"])
    if type(row["header_bytes_total"]) is not int or not (
        len(canonical_bytes(headers)) <= row["header_bytes_total"] <= 64 * 1024
    ):
        fail("HTTP headers exceed 64 KiB or lack an exact byte count")
    if (
        row["connect_timeout_seconds"],
        row["read_timeout_seconds"],
        row["total_timeout_seconds"],
    ) != (5, 15, 30):
        fail("capture timeouts differ from fixed transport policy")
    if row["transfer_decoding_complete"] is not True:
        fail("HTTP transfer decoding is incomplete")
    _validate_response_representation(row, headers=headers)
    _validate_parser_limits(row)
    return chain, headers


def build_capture_receipt(
    *,
    source: Mapping[str, Any],
    source_request: Mapping[str, Any],
    provider_response_id: str,
    policy: Mapping[str, Any],
    transport_observation: Mapping[str, Any],
    compressed_entity: bytes,
    decoded_entity: bytes,
    parser_input: bytes,
    publication_evidence_ref: Mapping[str, Any] | None,
    transport_attestation_ref: Mapping[str, Any],
    captured_at: str,
) -> dict[str, Any]:
    source_doc = validate_search_source(
        source,
        request=source_request,
        provider_response_id=provider_response_id,
    )
    if source_doc["media_kind"] == "PDF":
        fail("PDF is discovery-only and cannot be captured in I5 v1")
    policy_doc = validate_capture_policy(policy)
    captured = when(captured_at, label="captured_at")
    if captured < source_doc["timestamp"]:
        fail("capture predates discovery")
    chain, headers = _validate_capture_observation(
        transport_observation, requested_url=source_doc["url"]
    )
    _validate_entity_sizes(
        compressed_entity=compressed_entity,
        decoded_entity=decoded_entity,
        parser_input=parser_input,
        charset=transport_observation["charset"],
    )
    publication = None
    if publication_evidence_ref is not None:
        publication = exact_source_ref(publication_evidence_ref, label="publication_evidence_ref")
        if publication["available_at"] > captured or publication["cutoff"] > captured:
            fail("publication evidence is future-dated")
    attestation = exact_source_ref(transport_attestation_ref, label="transport_attestation_ref")
    if attestation["available_at"] > captured or attestation["cutoff"] > captured:
        fail("transport attestation is future-dated")
    observation = dict(transport_observation)
    return artifact(
        version=CAPTURE_RECEIPT_VERSION,
        identity_field="capture_receipt_id",
        timestamp_value=captured,
        payload={
            "source_ref": artifact_ref(source_doc, identity_field="source_id"),
            "policy_ref": artifact_ref(policy_doc, identity_field="capture_policy_id"),
            "status": "VALIDATED",
            "requested_url": source_doc["url"],
            "final_url": chain[-1]["url"],
            "redirect_chain": chain,
            "response_status": chain[-1]["status_code"],
            "selected_headers": headers,
            "content_encoding": observation["content_encoding"],
            "mime_type": observation["mime_type"],
            "charset": observation["charset"],
            "compressed_entity": digest_bytes(compressed_entity, label="compressed_entity"),
            "decoded_entity": digest_bytes(decoded_entity, label="decoded_entity"),
            "parser_input": digest_bytes(parser_input, label="parser_input"),
            "parser_identity": policy_doc["parser_identity"],
            "parser_version": policy_doc["parser_version"],
            "parser_options_sha256": policy_doc["parser_options_sha256"],
            "html_node_count": observation["html_node_count"],
            "html_max_depth": observation["html_max_depth"],
            "publication_evidence_ref": publication,
            "transport_attestation_ref": attestation,
            "security_observations": {
                "authorization_sent": False,
                "cookies_sent": False,
                "hostname_sni_preserved": True,
                "library_reresolution": False,
                "proxy_environment_used": False,
            },
            "reason_codes": [],
        },
    )


def validate_capture_receipt(
    document: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    source_request: Mapping[str, Any],
    provider_response_id: str,
    policy: Mapping[str, Any],
    transport_observation: Mapping[str, Any],
    compressed_entity: bytes,
    decoded_entity: bytes,
    parser_input: bytes,
) -> dict[str, Any]:
    row = closed_artifact(
        document,
        version=CAPTURE_RECEIPT_VERSION,
        identity_field="capture_receipt_id",
        payload_fields=_RECEIPT_FIELDS,
    )
    validate_digest(row["compressed_entity"], raw=compressed_entity, label="compressed_entity")
    validate_digest(row["decoded_entity"], raw=decoded_entity, label="decoded_entity")
    validate_digest(row["parser_input"], raw=parser_input, label="parser_input")
    expected = build_capture_receipt(
        source=source,
        source_request=source_request,
        provider_response_id=provider_response_id,
        policy=policy,
        transport_observation=transport_observation,
        compressed_entity=compressed_entity,
        decoded_entity=decoded_entity,
        parser_input=parser_input,
        publication_evidence_ref=row["publication_evidence_ref"],
        transport_attestation_ref=row["transport_attestation_ref"],
        captured_at=row["timestamp"],
    )
    same(row, expected, label="capture receipt")
    return expected


def build_capture_disposition(
    *,
    source: Mapping[str, Any],
    policy: Mapping[str, Any],
    status: str,
    reason_codes: Sequence[str],
    recorded_at: str,
) -> dict[str, Any]:
    if status not in {"CAPTURED", "MISMATCH", "BLOCKED"}:
        fail("capture disposition status is invalid")
    reasons = texts(reason_codes, label="reason_codes", minimum=1)
    return artifact(
        version=CAPTURE_RECEIPT_VERSION,
        identity_field="capture_receipt_id",
        timestamp_value=when(recorded_at, label="recorded_at"),
        payload={
            "source_ref": artifact_ref(source, identity_field="source_id"),
            "policy_ref": artifact_ref(policy, identity_field="capture_policy_id"),
            "status": status,
            "requested_url": source["url"],
            "final_url": None,
            "redirect_chain": [],
            "response_status": None,
            "selected_headers": {},
            "content_encoding": None,
            "mime_type": None,
            "charset": None,
            "compressed_entity": None,
            "decoded_entity": None,
            "parser_input": None,
            "parser_identity": None,
            "parser_version": None,
            "parser_options_sha256": None,
            "html_node_count": None,
            "html_max_depth": None,
            "publication_evidence_ref": None,
            "transport_attestation_ref": None,
            "security_observations": {},
            "reason_codes": reasons,
        },
    )


def _claim_locator(
    parser_input: bytes, *, claim: str, byte_start: int, byte_end: int
) -> dict[str, Any]:
    if type(byte_start) is not int or type(byte_end) is not int:
        fail("fact locator offsets must be integers")
    if not 0 <= byte_start < byte_end <= len(parser_input):
        fail("fact locator is outside parser input")
    located = parser_input[byte_start:byte_end]
    if located != claim.encode("utf-8"):
        fail("fact claim does not exactly match byte-safe locator")
    return {
        "byte_start": byte_start,
        "byte_end": byte_end,
        "matched_text_sha256": digest_bytes(located, label="located_text")["byte_sha256"],
        "parser_input_sha256": digest_bytes(parser_input, label="parser_input")["byte_sha256"],
    }


def build_validated_fact(
    *,
    capture_receipt: Mapping[str, Any],
    capture_closure: Mapping[str, Any],
    parser_input: bytes,
    subject_id: str,
    claim: str,
    byte_start: int,
    byte_end: int,
    source_class: str,
    canonical_source_id: str,
    original_source_id: str | None,
    syndication_group_id: str | None,
    conflict_status: str,
    prompt_injection_detected: bool,
    validated_at: str,
) -> dict[str, Any]:
    capture = validate_capture_receipt(
        capture_receipt, parser_input=parser_input, **dict(capture_closure)
    )
    validated = when(validated_at, label="validated_at")
    if not capture["timestamp"] <= validated:
        fail("fact validation predates capture")
    if source_class not in SOURCE_CLASSES:
        fail("fact source class is invalid")
    if conflict_status not in {"NONE", "RESOLVED", "UNRESOLVED"}:
        fail("fact conflict status is invalid")
    if type(prompt_injection_detected) is not bool:
        fail("prompt injection flag must be boolean")
    original_id = None
    if original_source_id is not None:
        original_id = identifier(original_source_id, label="original_source_id")
    if source_class in {"FIRST_PARTY", "ORIGINAL_SOURCE"} and original_id is None:
        fail("qualifying fact requires an identified original source")
    group_id = None
    if syndication_group_id is not None:
        group_id = identifier(syndication_group_id, label="syndication_group_id")
    claim_text = text(claim, label="claim")
    return artifact(
        version=VALIDATED_FACT_VERSION,
        identity_field="fact_id",
        timestamp_value=validated,
        payload={
            "capture_ref": artifact_ref(capture, identity_field="capture_receipt_id"),
            "subject_id": identifier(subject_id, label="subject_id"),
            "classification": "FACT",
            "claim": claim_text,
            "locator": _claim_locator(
                parser_input,
                claim=claim_text,
                byte_start=byte_start,
                byte_end=byte_end,
            ),
            "source_url": capture["final_url"],
            "source_class": source_class,
            "canonical_source_id": identifier(canonical_source_id, label="canonical_source_id"),
            "original_source_id": original_id,
            "syndication_group_id": group_id,
            "conflict_status": conflict_status,
            "captured_at": capture["timestamp"],
            "publication_evidence_ref": capture["publication_evidence_ref"],
            "prompt_injection": {
                "detected": prompt_injection_detected,
                "treatment": "UNTRUSTED_DATA_ONLY",
            },
        },
    )


def validate_validated_fact(
    document: Mapping[str, Any],
    *,
    capture_receipt: Mapping[str, Any],
    capture_closure: Mapping[str, Any],
    parser_input: bytes,
) -> dict[str, Any]:
    row = closed_artifact(
        document,
        version=VALIDATED_FACT_VERSION,
        identity_field="fact_id",
        payload_fields=_FACT_FIELDS,
    )
    expected = build_validated_fact(
        capture_receipt=capture_receipt,
        capture_closure=capture_closure,
        parser_input=parser_input,
        subject_id=row["subject_id"],
        claim=row["claim"],
        byte_start=row["locator"]["byte_start"],
        byte_end=row["locator"]["byte_end"],
        source_class=row["source_class"],
        canonical_source_id=row["canonical_source_id"],
        original_source_id=row["original_source_id"],
        syndication_group_id=row["syndication_group_id"],
        conflict_status=row["conflict_status"],
        prompt_injection_detected=row["prompt_injection"]["detected"],
        validated_at=row["timestamp"],
    )
    same(row, expected, label="validated fact")
    return expected


__all__ = [
    "build_capture_disposition",
    "build_capture_policy",
    "build_capture_receipt",
    "build_validated_fact",
    "validate_capture_policy",
    "validate_capture_receipt",
    "validate_validated_fact",
]
