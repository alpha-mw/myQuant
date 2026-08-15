"""Exact non-envelope records for protected contextual-validation custody."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import re
import stat
from typing import Any, Final

from quant_investor.contracts import (
    ContractError,
    canonical_json_bytes,
    get_contract,
    parse_canonical_json_bytes,
)

from .errors import SystemContractError

EMPTY_POINTER_SHA256: Final = "EMPTY"
_OBJECT_REF_FIELDS: Final = frozenset(
    {"kind", "contract_sha256", "artifact_id", "semantic_sha256", "byte_sha256"}
)

VALIDATION_INTENT_DOMAIN: Final = "myquant-validation-run-intent"
VALIDATION_PREPARED_DOMAIN: Final = "myquant-validation-run-prepared"
VALIDATION_CUSTODY_RECORD_DOMAIN: Final = "myquant-validation-custody-record"
SOURCE_VERIFICATION_SNAPSHOT_DOMAIN: Final = "myquant-source-verification-snapshot"
VALIDATION_COMPLETION_DOMAIN: Final = "myquant-validation-run-completion"

VALIDATION_INTENT_CONTRACT_SHA256: Final = (
    "4c91bfd608e6b1409d95501ca7389ed62e0a112cdeba51c093ccce38dde9c435"
)
VALIDATION_PREPARED_CONTRACT_SHA256: Final = (
    "ddf9108cfa5ee5b7b228f271e8e7996ce49d5480cb52901ecf3e35b1bc6aacc0"
)
VALIDATION_CUSTODY_RECORD_CONTRACT_SHA256: Final = (
    "df7494449e9c5404b7cd6d51d40732151591cd7d3415c3b90709791e3796e6f1"
)
SOURCE_VERIFICATION_SNAPSHOT_CONTRACT_SHA256: Final = (
    "cafedfacc0a7ac5eaccf10f9bffd07b2c119af011e303f60e7b6a9c5b6c89693"
)
VALIDATION_COMPLETION_CONTRACT_SHA256: Final = (
    "eb982739c098a65e8ab5fa894b0fd7de0ac3678111939518ea31ca9d4fca2ef9"
)

VALIDATION_INTENT_FIELDS: Final = frozenset(
    {
        "authority",
        "candidate_state_pointer_sha256",
        "candidate_state_ref",
        "clock_source",
        "component_registry_sha256",
        "domain",
        "factor_source_object_count",
        "factor_source_stat_tree_sha256",
        "factor_source_total_bytes",
        "factor_validator_manifest_ref",
        "installed_code_manifest_sha256",
        "intent_contract_sha256",
        "intent_id",
        "intrinsic_receipt_ref",
        "maximum_total_factor_source_bytes",
        "outcome",
        "plan_sha256",
        "release_manifest_ref",
        "semantic_sha256",
        "trusted_at",
        "validation_lane",
        "validation_namespace_id",
        "validation_profile_id",
        "validation_request_ref",
    }
)
VALIDATION_PREPARED_FIELDS: Final = frozenset(
    {
        "authority",
        "clock_source",
        "contextual_result_ref",
        "domain",
        "intent_id",
        "intent_semantic_sha256",
        "intent_sha256",
        "outcome",
        "plan_sha256",
        "prepared_contract_sha256",
        "prepared_id",
        "semantic_sha256",
        "trusted_at",
        "validation_attestation_ref",
        "validation_namespace_id",
        "validation_request_ref",
    }
)
VALIDATION_CUSTODY_RECORD_FIELDS: Final = frozenset(
    {
        "domain",
        "record_contract_sha256",
        "record_id",
        "validation_request_ref",
        "attestation_ref",
        "contextual_result_ref",
        "release_manifest_ref",
        "component_registry_sha256",
        "recorded_at",
        "os_actor",
        "outcome",
        "authority",
        "semantic_sha256",
    }
)
SOURCE_STAT_FIELDS: Final = frozenset(
    {
        "st_ctime_ns",
        "st_dev",
        "st_gid",
        "st_ino",
        "st_mode",
        "st_mtime_ns",
        "st_nlink",
        "st_size",
        "st_uid",
    }
)
SOURCE_STAT_ROW_FIELDS: Final = frozenset(
    {
        "source_binding_sha256",
        "source_object_ref",
        "stat_identity",
        "stat_identity_sha256",
    }
)
SOURCE_VERIFICATION_SNAPSHOT_FIELDS: Final = frozenset(
    {
        "authority",
        "cache_contract_sha256",
        "domain",
        "factor_source_total_bytes",
        "installed_code_manifest_sha256",
        "maximum_total_factor_source_bytes",
        "outcome",
        "semantic_sha256",
        "source_object_count",
        "source_object_refs",
        "source_stat_rows",
        "source_stat_tree_sha256",
        "unique_source_binding_count",
        "validation_attestation_ref",
    }
)
VALIDATION_COMPLETION_FIELDS: Final = frozenset(
    {
        "authority",
        "clock_source",
        "completion_contract_sha256",
        "completion_id",
        "contextual_result_ref",
        "custody_record_sha256",
        "domain",
        "intent_semantic_sha256",
        "intent_sha256",
        "outcome",
        "prepared_sha256",
        "semantic_sha256",
        "source_verification_snapshot_sha256",
        "trusted_at",
        "validation_attestation_ref",
        "validation_namespace_id",
        "validation_request_ref",
    }
)

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")


def _sha(value: Any, *, label: str, empty_allowed: bool = False) -> str:
    if empty_allowed and value == EMPTY_POINTER_SHA256:
        return value
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SystemContractError(f"{label} must be lowercase SHA-256")
    return value


def _text(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value.encode("utf-8", errors="strict")) > 512
        or any(ord(character) < 0x20 for character in value)
    ):
        raise SystemContractError(f"{label} must be canonical text")
    return value


def _timestamp(value: Any, *, label: str) -> str:
    text = _text(value, label=label)
    try:
        parsed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise SystemContractError(f"{label} must be canonical UTC seconds") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != text:
        raise SystemContractError(f"{label} must be canonical UTC seconds")
    return text


def _ref_key(ref: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        ref[field]
        for field in (
            "kind",
            "contract_sha256",
            "artifact_id",
            "semantic_sha256",
            "byte_sha256",
        )
    )


def _typed_ref(value: Any, *, label: str, kind: str) -> dict[str, str]:
    ref = validate_object_ref(value, label=label)
    if ref["kind"] != kind:
        raise SystemContractError(f"{label} has the wrong compiled kind")
    return ref


def validate_object_ref(value: Any, *, label: str = "object_ref") -> dict[str, str]:
    if type(value) is not dict or set(value) != set(_OBJECT_REF_FIELDS):
        raise SystemContractError(f"{label} fields are not exact")
    try:
        definition = get_contract(value.get("kind"), value.get("contract_sha256"))
    except ContractError as exc:
        raise SystemContractError(f"{label} contract pair is not compiled") from exc
    artifact_id = _text(value.get("artifact_id"), label=f"{label}.artifact_id")
    for field in ("semantic_sha256", "byte_sha256"):
        _sha(value.get(field), label=f"{label}.{field}")
    return {
        "kind": definition.kind,
        "contract_sha256": definition.contract_sha256,
        "artifact_id": artifact_id,
        "semantic_sha256": value["semantic_sha256"],
        "byte_sha256": value["byte_sha256"],
    }


def _exact_document(
    value: Mapping[str, Any] | bytes,
    *,
    fields: frozenset[str],
    label: str,
) -> dict[str, Any]:
    try:
        if isinstance(value, bytes):
            document = parse_canonical_json_bytes(value, label=label)
        else:
            document = dict(value)
        canonical_json_bytes(document)
    except (ContractError, TypeError, ValueError) as exc:
        raise SystemContractError(f"{label} is not canonical JSON") from exc
    if type(document) is not dict or set(document) != set(fields):
        raise SystemContractError(f"{label} fields are not exact")
    return document


def _semantic(document: Mapping[str, Any]) -> str:
    preimage = {key: value for key, value in document.items() if key != "semantic_sha256"}
    return hashlib.sha256(canonical_json_bytes(preimage)).hexdigest()


def _finish(document: dict[str, Any]) -> dict[str, Any]:
    document["semantic_sha256"] = _semantic(document)
    canonical_json_bytes(document)
    return document


def validation_intent_id(validation_namespace_id: str, validation_request_id: str) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": "myquant-validation-run-intent-id",
                "validation_namespace_id": validation_namespace_id,
                "validation_request_id": validation_request_id,
            }
        )
    ).hexdigest()


def prepared_id(intent_id: str) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {"domain": "myquant-validation-run-prepared-id", "intent_id": intent_id}
        )
    ).hexdigest()


def completion_id(intent_id: str) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {"domain": "myquant-validation-run-completion-id", "intent_id": intent_id}
        )
    ).hexdigest()


def custody_record_id(attestation_ref: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": "myquant-validation-custody-record-id",
                "attestation_ref": validate_object_ref(attestation_ref),
            }
        )
    ).hexdigest()


def build_validation_intent(**values: Any) -> dict[str, Any]:
    document = {
        **values,
        "domain": VALIDATION_INTENT_DOMAIN,
        "intent_contract_sha256": VALIDATION_INTENT_CONTRACT_SHA256,
        "outcome": "PLANNED",
        "clock_source": "SYSTEM_UTC",
        "authority": "NON_AUTHORIZING",
    }
    if set(document) != set(VALIDATION_INTENT_FIELDS - {"semantic_sha256"}):
        raise SystemContractError("validation intent builder fields are not exact")
    return validate_validation_intent(_finish(document))


def validate_validation_intent(  # noqa: C901
    value: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    row = _exact_document(value, fields=VALIDATION_INTENT_FIELDS, label="validation intent")
    if (
        row["domain"] != VALIDATION_INTENT_DOMAIN
        or row["intent_contract_sha256"] != VALIDATION_INTENT_CONTRACT_SHA256
        or row["outcome"] != "PLANNED"
        or row["clock_source"] != "SYSTEM_UTC"
        or row["authority"] != "NON_AUTHORIZING"
        or row["semantic_sha256"] != _semantic(row)
    ):
        raise SystemContractError("validation intent constants/hash mismatch")
    for field in (
        "component_registry_sha256",
        "factor_source_stat_tree_sha256",
        "installed_code_manifest_sha256",
        "plan_sha256",
    ):
        _sha(row[field], label=f"validation intent.{field}")
    _sha(
        row["candidate_state_pointer_sha256"],
        label="validation intent.candidate_state_pointer_sha256",
        empty_allowed=True,
    )
    request_ref = _typed_ref(
        row["validation_request_ref"],
        label="validation intent.validation_request_ref",
        kind="system.validation_run_request",
    )
    _typed_ref(
        row["release_manifest_ref"],
        label="validation intent.release_manifest_ref",
        kind="system.release",
    )
    _typed_ref(
        row["factor_validator_manifest_ref"],
        label="validation intent.factor_validator_manifest_ref",
        kind="factor.validator_manifest",
    )
    _typed_ref(
        row["intrinsic_receipt_ref"],
        label="validation intent.intrinsic_receipt_ref",
        kind="factor.validation_receipt",
    )
    if row["candidate_state_ref"] is not None:
        _typed_ref(
            row["candidate_state_ref"],
            label="validation intent.candidate_state_ref",
            kind="factor.composite_state",
        )
        if row["candidate_state_pointer_sha256"] == EMPTY_POINTER_SHA256:
            raise SystemContractError("validation intent candidate pointer is EMPTY")
    elif row["candidate_state_pointer_sha256"] != EMPTY_POINTER_SHA256:
        raise SystemContractError("validation intent empty candidate has a pointer SHA")
    for field in (
        "intent_id",
        "validation_namespace_id",
        "validation_profile_id",
        "validation_lane",
    ):
        _text(row[field], label=f"validation intent.{field}")
    _timestamp(row["trusted_at"], label="validation intent.trusted_at")
    if row["intent_id"] != validation_intent_id(
        row["validation_namespace_id"], request_ref["artifact_id"]
    ):
        raise SystemContractError("validation intent deterministic identity mismatch")
    if type(row["factor_source_object_count"]) is not int or row["factor_source_object_count"] <= 0:
        raise SystemContractError("validation intent source count is invalid")
    for field in ("factor_source_total_bytes", "maximum_total_factor_source_bytes"):
        if type(row[field]) is not int or row[field] <= 0:
            raise SystemContractError(f"validation intent.{field} is invalid")
    if row["factor_source_total_bytes"] > row["maximum_total_factor_source_bytes"]:
        raise SystemContractError("validation intent source bytes exceed the frozen bound")
    return row


def build_validation_prepared(**values: Any) -> dict[str, Any]:
    document = {
        **values,
        "domain": VALIDATION_PREPARED_DOMAIN,
        "prepared_contract_sha256": VALIDATION_PREPARED_CONTRACT_SHA256,
        "outcome": "PREPARED",
        "clock_source": "SYSTEM_UTC",
        "authority": "NON_AUTHORIZING",
    }
    if set(document) != set(VALIDATION_PREPARED_FIELDS - {"semantic_sha256"}):
        raise SystemContractError("validation prepared builder fields are not exact")
    return validate_validation_prepared(_finish(document))


def validate_validation_prepared(value: Mapping[str, Any] | bytes) -> dict[str, Any]:
    row = _exact_document(value, fields=VALIDATION_PREPARED_FIELDS, label="validation prepared")
    if (
        row["domain"] != VALIDATION_PREPARED_DOMAIN
        or row["prepared_contract_sha256"] != VALIDATION_PREPARED_CONTRACT_SHA256
        or row["outcome"] != "PREPARED"
        or row["clock_source"] != "SYSTEM_UTC"
        or row["authority"] != "NON_AUTHORIZING"
        or row["semantic_sha256"] != _semantic(row)
        or row["prepared_id"] != prepared_id(row["intent_id"])
    ):
        raise SystemContractError("validation prepared constants/hash mismatch")
    for field in ("intent_semantic_sha256", "intent_sha256", "plan_sha256"):
        _sha(row[field], label=f"validation prepared.{field}")
    request_ref = _typed_ref(
        row["validation_request_ref"],
        label="validation prepared.validation_request_ref",
        kind="system.validation_run_request",
    )
    _typed_ref(
        row["contextual_result_ref"],
        label="validation prepared.contextual_result_ref",
        kind="factor.contextual_validation_result",
    )
    _typed_ref(
        row["validation_attestation_ref"],
        label="validation prepared.validation_attestation_ref",
        kind="system.validation_attestation",
    )
    for field in ("intent_id", "prepared_id", "validation_namespace_id"):
        _text(row[field], label=f"validation prepared.{field}")
    _timestamp(row["trusted_at"], label="validation prepared.trusted_at")
    if row["intent_id"] != validation_intent_id(
        row["validation_namespace_id"], request_ref["artifact_id"]
    ):
        raise SystemContractError("validation prepared intent binding mismatch")
    return row


def build_custody_record(**values: Any) -> dict[str, Any]:
    document = {
        **values,
        "domain": VALIDATION_CUSTODY_RECORD_DOMAIN,
        "record_contract_sha256": VALIDATION_CUSTODY_RECORD_CONTRACT_SHA256,
        "outcome": "VALIDATED",
        "authority": "NON_AUTHORIZING",
    }
    if set(document) != set(VALIDATION_CUSTODY_RECORD_FIELDS - {"semantic_sha256"}):
        raise SystemContractError("validation custody record builder fields are not exact")
    return validate_custody_record(_finish(document))


def validate_custody_record(value: Mapping[str, Any] | bytes) -> dict[str, Any]:
    row = _exact_document(
        value, fields=VALIDATION_CUSTODY_RECORD_FIELDS, label="validation custody record"
    )
    if (
        row["domain"] != VALIDATION_CUSTODY_RECORD_DOMAIN
        or row["record_contract_sha256"] != VALIDATION_CUSTODY_RECORD_CONTRACT_SHA256
        or row["outcome"] != "VALIDATED"
        or row["authority"] != "NON_AUTHORIZING"
        or row["semantic_sha256"] != _semantic(row)
        or row["record_id"] != custody_record_id(row["attestation_ref"])
    ):
        raise SystemContractError("validation custody record constants/hash mismatch")
    _typed_ref(
        row["validation_request_ref"],
        label="validation custody record.validation_request_ref",
        kind="system.validation_run_request",
    )
    _typed_ref(
        row["attestation_ref"],
        label="validation custody record.attestation_ref",
        kind="system.validation_attestation",
    )
    _typed_ref(
        row["contextual_result_ref"],
        label="validation custody record.contextual_result_ref",
        kind="factor.contextual_validation_result",
    )
    _typed_ref(
        row["release_manifest_ref"],
        label="validation custody record.release_manifest_ref",
        kind="system.release",
    )
    _sha(row["component_registry_sha256"], label="component_registry_sha256")
    _text(row["record_id"], label="validation custody record.record_id")
    _timestamp(row["recorded_at"], label="validation custody record.recorded_at")
    _text(row["os_actor"], label="validation custody record.os_actor")
    return row


def build_source_verification_snapshot(**values: Any) -> dict[str, Any]:
    document = {
        **values,
        "domain": SOURCE_VERIFICATION_SNAPSHOT_DOMAIN,
        "cache_contract_sha256": SOURCE_VERIFICATION_SNAPSHOT_CONTRACT_SHA256,
        "outcome": "VALIDATED",
        "authority": "NON_AUTHORIZING",
    }
    if set(document) != set(SOURCE_VERIFICATION_SNAPSHOT_FIELDS - {"semantic_sha256"}):
        raise SystemContractError("source snapshot builder fields are not exact")
    return validate_source_verification_snapshot(_finish(document))


def validate_source_verification_snapshot(  # noqa: C901
    value: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    row = _exact_document(
        value,
        fields=SOURCE_VERIFICATION_SNAPSHOT_FIELDS,
        label="source verification snapshot",
    )
    if (
        row["domain"] != SOURCE_VERIFICATION_SNAPSHOT_DOMAIN
        or row["cache_contract_sha256"] != SOURCE_VERIFICATION_SNAPSHOT_CONTRACT_SHA256
        or row["outcome"] != "VALIDATED"
        or row["authority"] != "NON_AUTHORIZING"
        or row["semantic_sha256"] != _semantic(row)
    ):
        raise SystemContractError("source verification snapshot constants/hash mismatch")
    _typed_ref(
        row["validation_attestation_ref"],
        label="snapshot.attestation_ref",
        kind="system.validation_attestation",
    )
    refs = row.get("source_object_refs")
    rows = row.get("source_stat_rows")
    if type(refs) is not list or not refs or type(rows) is not list or not rows:
        raise SystemContractError("source snapshot ref/stat rows must be lists")
    normalized_refs = [
        _typed_ref(
            ref,
            label=f"snapshot.source_object_refs[{index}]",
            kind="system.source_object",
        )
        for index, ref in enumerate(refs)
    ]
    ref_keys = [_ref_key(ref) for ref in normalized_refs]
    if ref_keys != sorted(ref_keys) or len(ref_keys) != len(set(ref_keys)):
        raise SystemContractError("source snapshot refs are not sorted and unique")
    stat_ref_keys: list[tuple[str, ...]] = []
    binding_shas: list[str] = []
    for index, stat_row in enumerate(rows):
        if type(stat_row) is not dict or set(stat_row) != set(SOURCE_STAT_ROW_FIELDS):
            raise SystemContractError("source snapshot stat row fields are not exact")
        stat_ref = _typed_ref(
            stat_row["source_object_ref"],
            label=f"snapshot.source_stat_rows[{index}].ref",
            kind="system.source_object",
        )
        stat_ref_keys.append(_ref_key(stat_ref))
        stat_identity = stat_row.get("stat_identity")
        if type(stat_identity) is not dict or set(stat_identity) != set(SOURCE_STAT_FIELDS):
            raise SystemContractError("source snapshot stat identity fields are not exact")
        if any(type(item) is not int for item in stat_identity.values()):
            raise SystemContractError("source snapshot stat identity values are invalid")
        if (
            not stat.S_ISREG(stat_identity["st_mode"])
            or stat_identity["st_nlink"] != 1
            or stat_identity["st_size"] <= 0
            or any(
                stat_identity[field] < 0
                for field in (
                    "st_ctime_ns",
                    "st_dev",
                    "st_gid",
                    "st_ino",
                    "st_mtime_ns",
                    "st_uid",
                )
            )
        ):
            raise SystemContractError("source snapshot stat identity is unsafe")
        _sha(stat_row["source_binding_sha256"], label="source binding SHA")
        binding_shas.append(stat_row["source_binding_sha256"])
        _sha(stat_row["stat_identity_sha256"], label="source stat identity SHA")
        if (
            stat_row["stat_identity_sha256"]
            != hashlib.sha256(canonical_json_bytes(stat_identity)).hexdigest()
        ):
            raise SystemContractError("source stat identity SHA mismatch")
    _sha(row["source_stat_tree_sha256"], label="source stat tree SHA")
    _sha(row["installed_code_manifest_sha256"], label="installed code manifest SHA")
    if (
        binding_shas != sorted(binding_shas)
        or len(binding_shas) != len(set(binding_shas))
        or sorted(stat_ref_keys) != ref_keys
        or len(stat_ref_keys) != len(ref_keys)
        or row["source_stat_tree_sha256"] != hashlib.sha256(canonical_json_bytes(rows)).hexdigest()
    ):
        raise SystemContractError("source snapshot stat tree/ref projection mismatch")
    for field in (
        "source_object_count",
        "unique_source_binding_count",
        "factor_source_total_bytes",
        "maximum_total_factor_source_bytes",
    ):
        if type(row[field]) is not int or row[field] <= 0:
            raise SystemContractError(f"source snapshot.{field} is invalid")
    if (
        row["source_object_count"] != len(refs)
        or row["unique_source_binding_count"] != len(rows)
        or row["factor_source_total_bytes"]
        != sum(stat_row["stat_identity"]["st_size"] for stat_row in rows)
        or row["factor_source_total_bytes"] > row["maximum_total_factor_source_bytes"]
    ):
        raise SystemContractError("source snapshot count/byte projection mismatch")
    return row


def build_validation_completion(**values: Any) -> dict[str, Any]:
    document = {
        **values,
        "domain": VALIDATION_COMPLETION_DOMAIN,
        "completion_contract_sha256": VALIDATION_COMPLETION_CONTRACT_SHA256,
        "outcome": "VALIDATED",
        "clock_source": "SYSTEM_UTC",
        "authority": "NON_AUTHORIZING",
    }
    if set(document) != set(VALIDATION_COMPLETION_FIELDS - {"semantic_sha256"}):
        raise SystemContractError("validation completion builder fields are not exact")
    return validate_validation_completion(_finish(document))


def validate_validation_completion(value: Mapping[str, Any] | bytes) -> dict[str, Any]:
    row = _exact_document(value, fields=VALIDATION_COMPLETION_FIELDS, label="validation completion")
    if (
        row["domain"] != VALIDATION_COMPLETION_DOMAIN
        or row["completion_contract_sha256"] != VALIDATION_COMPLETION_CONTRACT_SHA256
        or row["outcome"] != "VALIDATED"
        or row["clock_source"] != "SYSTEM_UTC"
        or row["authority"] != "NON_AUTHORIZING"
        or row["semantic_sha256"] != _semantic(row)
        or row["completion_id"]
        != completion_id(
            validation_intent_id(
                row["validation_namespace_id"],
                row["validation_request_ref"]["artifact_id"],
            )
        )
    ):
        raise SystemContractError("validation completion constants/hash mismatch")
    for field in (
        "intent_semantic_sha256",
        "intent_sha256",
        "custody_record_sha256",
        "prepared_sha256",
        "source_verification_snapshot_sha256",
    ):
        _sha(row[field], label=f"validation completion.{field}")
    request_ref = _typed_ref(
        row["validation_request_ref"],
        label="validation completion.validation_request_ref",
        kind="system.validation_run_request",
    )
    _typed_ref(
        row["contextual_result_ref"],
        label="validation completion.contextual_result_ref",
        kind="factor.contextual_validation_result",
    )
    _typed_ref(
        row["validation_attestation_ref"],
        label="validation completion.validation_attestation_ref",
        kind="system.validation_attestation",
    )
    for field in ("completion_id", "validation_namespace_id"):
        _text(row[field], label=f"validation completion.{field}")
    _timestamp(row["trusted_at"], label="validation completion.trusted_at")
    if row["completion_id"] != completion_id(
        validation_intent_id(row["validation_namespace_id"], request_ref["artifact_id"])
    ):
        raise SystemContractError("validation completion deterministic identity mismatch")
    return row


__all__ = [
    name for name in globals() if name.startswith("VALIDATION_") or name.startswith("SOURCE_")
] + [
    "build_custody_record",
    "build_source_verification_snapshot",
    "build_validation_completion",
    "build_validation_intent",
    "build_validation_prepared",
    "completion_id",
    "custody_record_id",
    "prepared_id",
    "validate_custody_record",
    "validate_source_verification_snapshot",
    "validate_validation_completion",
    "validate_validation_intent",
    "validate_validation_prepared",
    "validation_intent_id",
]
