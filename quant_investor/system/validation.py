"""Generic, exact-once System runner for fixed Factor contextual validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import importlib
import os
from pathlib import PurePosixPath
import re
import resource
import sys
import time
from typing import Any, Final

from quant_investor.contracts import (
    ContractError,
    canonical_json_bytes,
    get_contract,
    seal_artifact,
    validate_artifact,
)

from .components import (
    BOOTSTRAP_VALIDATION_PROFILE,
    COMPONENT_REGISTRY_SHA256,
    CONTEXTUAL_VALIDATOR_PACKAGE,
    MAXIMUM_DECODED_SOURCE_CELLS,
    MAXIMUM_DECODED_SOURCE_ROWS,
    MAXIMUM_FACTOR_SOURCE_OBJECT_BYTES,
    MAXIMUM_VALIDATION_RSS_BYTES,
    PROSPECTIVE_VALIDATION_PROFILE,
    STRICT_SOURCE_DECODER_ID,
    component_registry,
    validate_installed_component_manifest,
    validation_profile,
)
from .errors import (
    SystemContractError,
    SystemImmutableConflict,
    SystemNotFound,
    SystemPreconditionError,
    SystemSecurityError,
    SystemStorageError,
)
from .release import installed_code_manifest_sha256
from .storage import (
    EMPTY_POINTER_SHA256,
    SOURCE_VERIFICATION_CACHE_ROOT,
    VALIDATION_CUSTODY_ROOT,
    VALIDATION_REQUESTS_ROOT,
    VALIDATION_RUNS_ROOT,
)
from .store import (
    OBJECT_REF_FIELDS,
    OBJECT_REF_SORT_FIELDS,
    _domain_identity,
    _require_text,
    _require_timestamp,
    _utc_now,
    object_ref_for_artifact,
    validate_object_ref,
    validation_namespace_path_sha256,
)
from .validation_records import (
    build_custody_record,
    build_source_verification_snapshot,
    build_validation_completion,
    build_validation_intent,
    build_validation_prepared,
    completion_id,
    custody_record_id,
    prepared_id,
    validate_custody_record,
    validate_source_verification_snapshot,
    validate_validation_completion,
    validate_validation_intent,
    validate_validation_prepared,
    validation_intent_id,
)

VALIDATION_REQUEST_KIND: Final = "system.validation_run_request"
VALIDATION_ATTESTATION_KIND: Final = "system.validation_attestation"
CONTEXTUAL_RESULT_KIND: Final = "factor.contextual_validation_result"
FACTOR_VALIDATOR_MANIFEST_KIND: Final = "factor.validator_manifest"
FACTOR_VALIDATION_RECEIPT_KIND: Final = "factor.validation_receipt"
FACTOR_COMPOSITE_STATE_KIND: Final = "factor.composite_state"
FACTOR_SOURCE_ATTESTATION_KIND: Final = "factor.source_decode_attestation"
FACTOR_CUSTODY_RECORD_KIND: Final = "factor.custody_record"
SOURCE_OBJECT_KIND: Final = "system.source_object"
INSTALLED_COMPONENT_KIND: Final = "system.installed_component_manifest"

MAXIMUM_VALIDATION_OBJECTS: Final = 20_000
MAXIMUM_VALIDATION_OPEN_FDS: Final = 64
MAXIMUM_BOOTSTRAP_VALIDATION_SECONDS: Final = 30.0
MAXIMUM_PROSPECTIVE_VALIDATION_SECONDS: Final = 180.0

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_REF_FIELDS: Final = set(OBJECT_REF_FIELDS)


def _ref_key(ref: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(ref[field] for field in OBJECT_REF_SORT_FIELDS)


def _sorted_refs(values: Sequence[Mapping[str, Any]], *, label: str) -> list[dict[str, str]]:
    refs = [
        validate_object_ref(value, label=f"{label}[{index}]") for index, value in enumerate(values)
    ]
    refs.sort(key=_ref_key)
    keys = [_ref_key(ref) for ref in refs]
    if len(keys) != len(set(keys)):
        raise SystemContractError(f"{label} contains duplicate refs")
    return refs


def _business_identity(prefix: str, inputs: Mapping[str, Any]) -> str:
    return f"{prefix}-{hashlib.sha256(canonical_json_bytes(dict(inputs))).hexdigest()}"


def _bootstrap_namespace(intrinsic_receipt_ref: Mapping[str, Any]) -> str:
    return _business_identity(
        "factor-validation-namespace",
        {
            "validation_profile_id": BOOTSTRAP_VALIDATION_PROFILE,
            "intrinsic_receipt_ref": validate_object_ref(intrinsic_receipt_ref),
        },
    )


def _prospective_namespace(
    *,
    exchange_calendar_ref: Mapping[str, Any],
    implementation_manifest_ref: Mapping[str, Any],
    factor_validator_manifest_ref: Mapping[str, Any],
) -> str:
    return _business_identity(
        "factor-validation-namespace",
        {
            "validation_profile_id": PROSPECTIVE_VALIDATION_PROFILE,
            "exchange_calendar_ref": validate_object_ref(exchange_calendar_ref),
            "implementation_manifest_ref": validate_object_ref(implementation_manifest_ref),
            "factor_validator_manifest_ref": validate_object_ref(factor_validator_manifest_ref),
        },
    )


def _request_id(payload_without_id: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {"domain": "myquant-validation-run-request-id", **dict(payload_without_id)}
        )
    ).hexdigest()


def _run_paths(namespace_id: str, intent_id: str) -> dict[str, PurePosixPath]:
    root = VALIDATION_RUNS_ROOT / validation_namespace_path_sha256(namespace_id)
    return {
        "root": root,
        "lock": root / ".lock",
        "intents": root / "intents",
        "prepared": root / "prepared",
        "completions": root / "completions",
        "intent": root / "intents" / intent_id,
        "prepared_record": root / "prepared" / intent_id,
        "completion": root / "completions" / intent_id,
    }


def _initialize_run_namespace(store: Any, namespace_id: str) -> None:
    paths = _run_paths(namespace_id, "0" * 64)
    for key in ("root", "intents", "prepared", "completions"):
        store._storage.ensure_directory(paths[key])
    with store._storage.exclusive_lock(paths["lock"]):
        pass


def _resolve_request_namespace(
    store: Any,
    *,
    profile_id: str,
    factor_validator_manifest_ref: Mapping[str, Any],
    intrinsic_receipt_ref: Mapping[str, Any],
    candidate_state_ref: Mapping[str, Any] | None,
) -> str:
    if profile_id == BOOTSTRAP_VALIDATION_PROFILE:
        if candidate_state_ref is not None:
            raise SystemContractError("bootstrap validation cannot bind candidate state")
        return _bootstrap_namespace(intrinsic_receipt_ref)
    if profile_id != PROSPECTIVE_VALIDATION_PROFILE or candidate_state_ref is None:
        raise SystemContractError("prospective validation requires candidate state")
    candidate_ref = validate_object_ref(candidate_state_ref, label="candidate_state_ref")
    if candidate_ref["kind"] != FACTOR_COMPOSITE_STATE_KIND:
        raise SystemContractError("candidate state has the wrong compiled kind")
    candidate = store.get_object(candidate_ref)
    preregistration_ref = validate_object_ref(
        candidate["payload"].get("preregistration_ref"),
        label="candidate_state.preregistration_ref",
    )
    preregistration = store.get_object(preregistration_ref)
    payload = preregistration.get("payload")
    if type(payload) is not dict:
        raise SystemContractError("candidate preregistration payload is invalid")
    namespace = _prospective_namespace(
        exchange_calendar_ref=payload.get("exchange_calendar_ref"),
        implementation_manifest_ref=payload.get("implementation_manifest_ref"),
        factor_validator_manifest_ref=factor_validator_manifest_ref,
    )
    if candidate.get("payload", {}).get("custody_namespace_id") != namespace:
        raise SystemContractError("candidate custody namespace does not match mine roots")
    return namespace


def build_validation_run_request(
    store: Any,
    *,
    release_manifest_ref: Mapping[str, Any],
    factor_validator_manifest_ref: Mapping[str, Any],
    intrinsic_receipt_ref: Mapping[str, Any],
    candidate_state_ref: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive and publish the sole request for one exact validation closure."""

    release_ref = validate_object_ref(release_manifest_ref, label="release_manifest_ref")
    validator_ref = validate_object_ref(
        factor_validator_manifest_ref, label="factor_validator_manifest_ref"
    )
    receipt_ref = validate_object_ref(intrinsic_receipt_ref, label="intrinsic_receipt_ref")
    if release_ref["kind"] != "system.release":
        raise SystemContractError("validation request release kind is invalid")
    if validator_ref["kind"] != FACTOR_VALIDATOR_MANIFEST_KIND:
        raise SystemContractError("validation request Factor manifest kind is invalid")
    if receipt_ref["kind"] != FACTOR_VALIDATION_RECEIPT_KIND:
        raise SystemContractError("validation request receipt kind is invalid")
    for label, ref in (
        ("release_manifest_ref", release_ref),
        ("factor_validator_manifest_ref", validator_ref),
        ("intrinsic_receipt_ref", receipt_ref),
    ):
        artifact = store.get_object(ref)
        if object_ref_for_artifact(artifact) != ref:
            raise SystemContractError(f"validation request {label} binding mismatch")
    candidate_ref = (
        validate_object_ref(candidate_state_ref, label="candidate_state_ref")
        if candidate_state_ref is not None
        else None
    )
    profile_id = (
        BOOTSTRAP_VALIDATION_PROFILE if candidate_ref is None else PROSPECTIVE_VALIDATION_PROFILE
    )
    namespace = _resolve_request_namespace(
        store,
        profile_id=profile_id,
        factor_validator_manifest_ref=validator_ref,
        intrinsic_receipt_ref=receipt_ref,
        candidate_state_ref=candidate_ref,
    )
    body = {
        "validation_profile_id": profile_id,
        "component_registry_sha256": COMPONENT_REGISTRY_SHA256,
        "validation_namespace_id": namespace,
        "release_manifest_ref": release_ref,
        "factor_validator_manifest_ref": validator_ref,
        "intrinsic_receipt_ref": receipt_ref,
        "candidate_state_ref": candidate_ref,
    }
    payload = {"validation_request_id": _request_id(body), **body}
    request_path = VALIDATION_REQUESTS_ROOT / f"{payload['validation_request_id']}.json"
    store._storage.ensure_directory(VALIDATION_REQUESTS_ROOT)
    with store._storage.exclusive_lock(VALIDATION_REQUESTS_ROOT / ".lock"):
        existing = store._storage.read_optional(request_path)
        if existing is not None:
            request = validate_artifact(existing.data, expected_kind=VALIDATION_REQUEST_KIND)
            if request["payload"] != payload:
                raise SystemImmutableConflict("validation request identity conflict")
            ref = object_ref_for_artifact(request)
            if store.get_object(ref) != request:
                raise SystemContractError("validation request object binding mismatch")
        else:
            request = seal_artifact(
                VALIDATION_REQUEST_KIND,
                payload,
                created_at=_utc_now(),
            )
            ref = store.put_object(request)
            stored = store._storage.write_exact_once(request_path, canonical_json_bytes(request))
            if stored.data != canonical_json_bytes(request):
                raise SystemStorageError("validation request exact readback mismatch")
    _initialize_run_namespace(store, namespace)
    return {"validation_request": request, "validation_request_ref": ref}


def _load_request(  # noqa: C901
    store: Any, value: Mapping[str, Any] | bytes
) -> tuple[dict[str, Any], dict[str, str]]:
    if type(value) is bytes:
        try:
            parsed = validate_artifact(value, expected_kind=VALIDATION_REQUEST_KIND)
        except ContractError as exc:
            raise SystemContractError("validation request envelope is invalid") from exc
        request = parsed
        ref = object_ref_for_artifact(request)
    elif type(value) is dict and set(value) == _REF_FIELDS:
        ref = validate_object_ref(value, label="validation_request_ref")
        if ref["kind"] != VALIDATION_REQUEST_KIND:
            raise SystemContractError("validation request ref has the wrong kind")
        request = store.get_object(ref)
    elif isinstance(value, Mapping):
        try:
            request = validate_artifact(dict(value), expected_kind=VALIDATION_REQUEST_KIND)
        except ContractError as exc:
            raise SystemContractError("validation request envelope is invalid") from exc
        ref = object_ref_for_artifact(request)
    else:
        raise SystemContractError("validation request must be an exact ref or envelope")
    if store.get_object(ref) != request:
        raise SystemContractError("validation request must already be stored exactly")
    index = store._storage.read(VALIDATION_REQUESTS_ROOT / f"{request['artifact_id']}.json")
    if index.data != canonical_json_bytes(request):
        raise SystemContractError("validation request deterministic index mismatch")
    return request, ref


def _validate_request_identity(store: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(request["payload"])
    profile_id = payload.get("validation_profile_id")
    validation_profile(profile_id)
    if payload.get("component_registry_sha256") != COMPONENT_REGISTRY_SHA256:
        raise SystemContractError("validation request component registry mismatch")
    expected_namespace = _resolve_request_namespace(
        store,
        profile_id=profile_id,
        factor_validator_manifest_ref=payload.get("factor_validator_manifest_ref"),
        intrinsic_receipt_ref=payload.get("intrinsic_receipt_ref"),
        candidate_state_ref=payload.get("candidate_state_ref"),
    )
    if payload.get("validation_namespace_id") != expected_namespace:
        raise SystemContractError("validation request namespace mismatch")
    body = {key: value for key, value in payload.items() if key != "validation_request_id"}
    if payload.get("validation_request_id") != _request_id(body):
        raise SystemContractError("validation request deterministic identity mismatch")
    return payload


def _walk_refs(value: Any) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    if type(value) is dict:
        if set(value) == _REF_FIELDS:
            result.append(validate_object_ref(value))
        else:
            for item in value.values():
                result.extend(_walk_refs(item))
    elif type(value) is list:
        for item in value:
            result.extend(_walk_refs(item))
    return result


def _resolve_recursive_closure(
    store: Any, roots: Sequence[Mapping[str, Any]]
) -> dict[tuple[str, ...], tuple[dict[str, str], dict[str, Any]]]:
    queue = [validate_object_ref(ref) for ref in roots]
    resolved: dict[tuple[str, ...], tuple[dict[str, str], dict[str, Any]]] = {}
    while queue:
        ref = queue.pop(0)
        key = _ref_key(ref)
        if key in resolved:
            continue
        if len(resolved) >= MAXIMUM_VALIDATION_OBJECTS:
            raise SystemSecurityError("validation closure object bound exceeded")
        artifact = store.get_object(ref)
        if object_ref_for_artifact(artifact) != ref:
            raise SystemContractError("validation closure ref mismatch")
        resolved[key] = (ref, artifact)
        queue.extend(_walk_refs(artifact["payload"]))
    return resolved


def _receipt_closure(store: Any, receipt_ref: Mapping[str, Any]) -> dict[str, Any]:
    ref = validate_object_ref(receipt_ref, label="intrinsic_receipt_ref")
    receipt = store.get_object(ref)
    payload = receipt.get("payload")
    if (
        receipt.get("kind") != FACTOR_VALIDATION_RECEIPT_KIND
        or type(payload) is not dict
        or payload.get("validated") is not True
        or payload.get("authority") != "NON_AUTHORIZING"
    ):
        raise SystemContractError("intrinsic receipt is not valid and non-authorizing")
    policy_ref = validate_object_ref(payload.get("policy_ref"), label="receipt.policy_ref")
    active_ref = validate_object_ref(payload.get("active_set_ref"), label="receipt.active_set_ref")
    evidence = payload.get("evidence_refs")
    if type(evidence) is not list:
        raise SystemContractError("intrinsic receipt evidence refs are invalid")
    evidence_refs = _sorted_refs(evidence, label="receipt.evidence_refs")
    if evidence_refs != evidence:
        raise SystemContractError("intrinsic receipt evidence refs are not canonical")
    return {
        "intrinsic_receipt_ref": ref,
        "intrinsic_receipt": receipt,
        "policy_ref": policy_ref,
        "active_set_ref": active_ref,
        "evidence_refs": evidence_refs,
    }


def _custody_chain(store: Any, candidate: Mapping[str, Any] | None) -> list[dict[str, str]]:
    if candidate is None:
        return []
    payload = candidate.get("payload")
    if type(payload) is not dict:
        raise SystemContractError("candidate composite payload is invalid")
    count = payload.get("custody_record_count")
    head = payload.get("custody_head_ref")
    if type(count) is not int or count < 0:
        raise SystemContractError("candidate custody record count is invalid")
    if count == 0:
        if head is not None:
            raise SystemContractError("empty candidate custody chain has a head")
        return []
    current = validate_object_ref(head, label="candidate.custody_head_ref")
    descending: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    while current is not None:
        key = _ref_key(current)
        if key in seen or len(descending) >= count:
            raise SystemContractError("candidate custody chain is cyclic or oversized")
        seen.add(key)
        if current["kind"] != FACTOR_CUSTODY_RECORD_KIND:
            raise SystemContractError("candidate custody head has the wrong kind")
        artifact = store.get_object(current)
        descending.append(current)
        previous = artifact["payload"].get("previous_custody_ref")
        current = (
            validate_object_ref(previous, label="custody.previous_custody_ref")
            if previous is not None
            else None
        )
    if len(descending) != count:
        raise SystemContractError("candidate custody chain count mismatch")
    return list(reversed(descending))


def _validate_component_closure(  # noqa: C901
    store: Any,
    *,
    profile: Mapping[str, Any],
    request: Mapping[str, Any],
    factor_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_payload = factor_manifest.get("payload")
    if type(manifest_payload) is not dict:
        raise SystemContractError("Factor validator manifest payload is invalid")
    if manifest_payload.get("release_manifest_ref") != request["release_manifest_ref"]:
        raise SystemContractError("Factor validator manifest release mismatch")
    contextual_ref = validate_object_ref(
        manifest_payload.get("contextual_validator_component_ref"),
        label="factor_validator_manifest.contextual_component_ref",
    )
    decoder_ref = validate_object_ref(
        manifest_payload.get("source_decoder_component_ref"),
        label="factor_validator_manifest.decoder_component_ref",
    )
    contextual = validate_installed_component_manifest(store.get_object(contextual_ref))
    decoder = validate_installed_component_manifest(store.get_object(decoder_ref))
    contextual_payload = contextual["payload"]
    decoder_payload = decoder["payload"]
    release_ref = request["release_manifest_ref"]
    expected_entrypoint = {
        "module_name": profile["callback_module"],
        "qualified_name": profile["callback_qualified_name"],
    }
    if (
        contextual_payload["component_registry_sha256"] != COMPONENT_REGISTRY_SHA256
        or contextual_payload["component_id"] != f"{request['validation_profile_id']}-component"
        or contextual_payload["component_role"] != "CONTEXTUAL_VALIDATOR"
        or contextual_payload["package_name"] != CONTEXTUAL_VALIDATOR_PACKAGE
        or len(contextual_payload["entrypoints"]) != 1
        or {
            key: contextual_payload["entrypoints"][0][key]
            for key in ("module_name", "qualified_name")
        }
        != expected_entrypoint
        or contextual_payload["allowed_source_formats"] != []
        or contextual_payload["fallback_allowed"] is not False
        or contextual_payload["release_manifest_ref"] != release_ref
    ):
        raise SystemContractError("contextual validator component does not match profile")
    registry_decoder = component_registry()["source_decoder"]
    if (
        decoder_payload["component_registry_sha256"] != COMPONENT_REGISTRY_SHA256
        or decoder_payload["component_id"] != STRICT_SOURCE_DECODER_ID
        or decoder_payload["component_role"] != "SOURCE_DECODER"
        or decoder_payload["module_names"] != registry_decoder["module_names"]
        or decoder_payload["allowed_source_formats"] != ["PARQUET"]
        or decoder_payload["fallback_allowed"] is not False
        or decoder_payload["release_manifest_ref"] != release_ref
        or len(decoder_payload["entrypoints"]) != 1
        or {
            key: decoder_payload["entrypoints"][0][key] for key in ("module_name", "qualified_name")
        }
        != {
            "module_name": registry_decoder["module_name"],
            "qualified_name": registry_decoder["qualified_name"],
        }
    ):
        raise SystemContractError("source decoder component does not match fixed registry")

    rows = manifest_payload.get("implementation_rows")
    if type(rows) is not list or not rows:
        raise SystemContractError("Factor implementation component rows are absent")
    implementation_refs: list[dict[str, str]] = []
    factor_ids: list[str] = []
    implementation_ids: list[str] = []
    for index, row in enumerate(rows):
        if type(row) is not dict:
            raise SystemContractError("Factor implementation row is invalid")
        ref = validate_object_ref(
            row.get("implementation_component_ref"),
            label=f"implementation_rows[{index}].component_ref",
        )
        component = validate_installed_component_manifest(store.get_object(ref))
        component_payload = component["payload"]
        if (
            component_payload["component_registry_sha256"] != COMPONENT_REGISTRY_SHA256
            or component_payload["component_role"] != "SOURCE_IMPLEMENTATION"
            or component_payload["component_id"] != row.get("implementation_id")
            or component_payload["release_manifest_ref"] != release_ref
            or component_payload["allowed_source_formats"] != []
            or component_payload["fallback_allowed"] is not False
            or len(component_payload["entrypoints"]) != 1
        ):
            raise SystemContractError("Factor implementation component role is invalid")
        entrypoint = component_payload["entrypoints"][0]
        if (
            entrypoint["module_name"] != row.get("module_name")
            or entrypoint["qualified_name"] != row.get("qualified_name")
            or entrypoint["code_sha256"] != row.get("code_sha256")
        ):
            raise SystemContractError("Factor implementation AST identity mismatch")
        implementation_refs.append(ref)
        factor_ids.append(
            _require_text(row.get("factor_id"), label=f"implementation_rows[{index}].factor_id")
        )
        implementation_ids.append(
            _require_text(
                row.get("implementation_id"),
                label=f"implementation_rows[{index}].implementation_id",
            )
        )
    if factor_ids != sorted(factor_ids, key=lambda value: value.encode("utf-8")):
        raise SystemContractError("Factor implementation rows are not in UTF-8 order")
    if len(factor_ids) != len(set(factor_ids)) or len(implementation_ids) != len(
        set(implementation_ids)
    ):
        raise SystemContractError("Factor implementation identities are duplicated")
    implementation_refs = _sorted_refs(implementation_refs, label="implementation_component_refs")
    contracts = manifest_payload.get("validated_contracts")
    if type(contracts) is not list or not contracts:
        raise SystemContractError("Factor validated contract rows are absent")
    contract_keys: list[tuple[str, str]] = []
    for row in contracts:
        if type(row) is not dict:
            raise SystemContractError("Factor validated contract row is invalid")
        definition = get_contract(row.get("kind"), row.get("contract_sha256"))
        if row != {
            "kind": definition.kind,
            "contract_sha256": definition.contract_sha256,
            "json_schema_sha256": definition.json_schema_sha256,
            "validator_code_sha256": definition.validator_code_sha256,
        }:
            raise SystemContractError("Factor validated contract row has drifted")
        contract_keys.append((definition.kind, definition.contract_sha256))
    if contract_keys != sorted(contract_keys) or len(contract_keys) != len(set(contract_keys)):
        raise SystemContractError("Factor validated contract rows are not canonical")
    return {
        "contextual_validator_component_ref": contextual_ref,
        "source_decoder_component_ref": decoder_ref,
        "implementation_component_refs": implementation_refs,
        "compiled_contracts": list(contracts),
    }


def _source_snapshot_rows(
    store: Any,
    source_refs: Sequence[Mapping[str, Any]],
    *,
    maximum_total_bytes: int,
    full_hash: bool,
) -> dict[str, Any]:
    refs = _sorted_refs(source_refs, label="factor_source_object_refs")
    bindings: set[str] = set()
    rows: list[dict[str, Any]] = []
    total = 0
    formats: set[str] = set()
    for ref in refs:
        if ref["kind"] != SOURCE_OBJECT_KIND:
            raise SystemContractError("Factor source closure contains a non-source object")
        inspected = store.inspect_source_object(
            ref,
            full_hash=full_hash,
            maximum_bytes=MAXIMUM_FACTOR_SOURCE_OBJECT_BYTES,
        )
        if inspected["source_format"] not in {"JSON", "PARQUET"}:
            raise SystemContractError("Factor source format is not admitted by fixed profiles")
        formats.add(inspected["source_format"])
        binding_sha = hashlib.sha256(
            canonical_json_bytes(
                {
                    "domain": "myquant-source-binding",
                    "source_root_id": inspected["source_root_id"],
                    "relative_path": inspected["relative_path"],
                }
            )
        ).hexdigest()
        if binding_sha in bindings:
            raise SystemContractError("Factor source physical bindings are duplicated")
        bindings.add(binding_sha)
        total += inspected["size"]
        if total > maximum_total_bytes:
            raise SystemSecurityError("Factor source closure exceeds its profile byte bound")
        stat_identity = inspected["stat_identity"]
        rows.append(
            {
                "source_binding_sha256": binding_sha,
                "source_object_ref": ref,
                "stat_identity": stat_identity,
                "stat_identity_sha256": hashlib.sha256(
                    canonical_json_bytes(stat_identity)
                ).hexdigest(),
            }
        )
    rows.sort(key=lambda row: row["source_binding_sha256"])
    if not refs or "PARQUET" not in formats:
        raise SystemContractError("Factor source closure lacks strict PARQUET input")
    tree_sha = hashlib.sha256(canonical_json_bytes(rows)).hexdigest()
    return {
        "source_object_refs": refs,
        "source_stat_rows": rows,
        "source_stat_tree_sha256": tree_sha,
        "factor_source_total_bytes": total,
        "source_object_count": len(refs),
        "unique_source_binding_count": len(bindings),
    }


def _validate_source_attestation_closure(
    store: Any,
    *,
    source_attestation_refs: Sequence[Mapping[str, Any]],
    source_object_refs: Sequence[Mapping[str, Any]],
    factor_validator_manifest_ref: Mapping[str, Any],
    components: Mapping[str, Any],
) -> None:
    """Verify generic decoder/component/descriptor bindings for sealed source rows."""

    expected_sources = {
        _ref_key(validate_object_ref(ref, label="factor_source_object_ref"))
        for ref in source_object_refs
    }
    decoder_component = store.get_object(components["source_decoder_component_ref"])
    registry_decoder = component_registry()["source_decoder"]
    decoder_matches = [
        row
        for row in decoder_component["payload"]["entrypoints"]
        if row["module_name"] == registry_decoder["module_name"]
        and row["qualified_name"] == registry_decoder["qualified_name"]
    ]
    if len(decoder_matches) != 1:
        raise SystemContractError("fixed source decoder entrypoint is ambiguous")
    decoder_code_sha256 = decoder_matches[0]["code_sha256"]

    for attestation_ref in _sorted_refs(source_attestation_refs, label="source_attestation_refs"):
        if attestation_ref["kind"] != FACTOR_SOURCE_ATTESTATION_KIND:
            raise SystemContractError("source attestation ref has the wrong kind")
        payload = store.get_object(attestation_ref)["payload"]
        decoder = payload["decoder_contract"]
        if (
            decoder["decoder_id"] != STRICT_SOURCE_DECODER_ID
            or decoder["factor_validator_manifest_ref"] != factor_validator_manifest_ref
            or decoder["contextual_validator_component_ref"]
            != components["contextual_validator_component_ref"]
            or decoder["source_decoder_component_ref"] != components["source_decoder_component_ref"]
            or decoder["decoder_code_sha256"] != decoder_code_sha256
            or decoder["implementation_component_refs"]
            != components["implementation_component_refs"]
            or decoder["allowed_source_formats"] != ["PARQUET"]
            or decoder["fallback_allowed"] is not False
        ):
            raise SystemContractError("source attestation decoder closure has drifted")
        for binding in payload["source_bindings"]:
            source_ref = validate_object_ref(
                binding["source_object_ref"], label="source binding object ref"
            )
            if _ref_key(source_ref) not in expected_sources:
                raise SystemContractError("source attestation escapes the source closure")
            source = store.get_object(source_ref)
            source_payload = source["payload"]
            if (
                source_payload["source_format"] != "PARQUET"
                or source_payload["media_type"] != "application/vnd.apache.parquet"
                or binding["source_root_id"] != source_payload["source_root_id"]
                or binding["source_object_created_at"] != source["created_at"]
                or binding["media_type"] != source_payload["media_type"]
                or binding["source_format"] != source_payload["source_format"]
                or binding["source_byte_sha256"] != source_payload["byte_sha256"]
            ):
                raise SystemContractError("source attestation descriptor binding has drifted")
            inspected = store.inspect_source_object(
                source_ref,
                full_hash=False,
                maximum_bytes=MAXIMUM_FACTOR_SOURCE_OBJECT_BYTES,
            )
            row_count = binding["row_count"]
            column_count = binding["column_count"]
            cell_count = binding["decoded_cell_count"]
            if (
                binding["source_byte_count"] != inspected["size"]
                or type(row_count) is not int
                or not 0 < row_count <= MAXIMUM_DECODED_SOURCE_ROWS
                or type(column_count) is not int
                or column_count <= 0
                or type(cell_count) is not int
                or cell_count != row_count * column_count
                or cell_count > MAXIMUM_DECODED_SOURCE_CELLS
            ):
                raise SystemContractError("source attestation resource projection is invalid")


def _derive_plan(
    store: Any, request_payload: Mapping[str, Any], *, full_source_hash: bool
) -> dict[str, Any]:
    profile = validation_profile(request_payload["validation_profile_id"])
    receipt = _receipt_closure(store, request_payload["intrinsic_receipt_ref"])
    candidate_ref = request_payload["candidate_state_ref"]
    candidate = store.get_object(candidate_ref) if candidate_ref is not None else None
    candidate_pointer_sha = EMPTY_POINTER_SHA256
    if candidate_ref is not None:
        candidate_state = store.read_candidate_state(request_payload["validation_namespace_id"])
        if candidate_state is None or candidate_state["candidate_state_ref"] != candidate_ref:
            raise SystemPreconditionError("validation candidate state is not current")
        candidate_pointer_sha = candidate_state["pointer_byte_sha256"]
        if candidate["payload"].get("intrinsic_receipt_ref") != receipt["intrinsic_receipt_ref"]:
            raise SystemContractError("candidate state intrinsic receipt binding mismatch")
    roots = [
        receipt["intrinsic_receipt_ref"],
        receipt["policy_ref"],
        *receipt["evidence_refs"],
        receipt["active_set_ref"],
        request_payload["factor_validator_manifest_ref"],
    ]
    if candidate_ref is not None:
        roots.append(candidate_ref)
    closure = _resolve_recursive_closure(store, roots)
    artifacts = [artifact for _, artifact in closure.values()]
    source_refs = [
        ref for ref, artifact in closure.values() if artifact["kind"] == SOURCE_OBJECT_KIND
    ]
    source_attestation_refs = [
        ref
        for ref, artifact in closure.values()
        if artifact["kind"] == FACTOR_SOURCE_ATTESTATION_KIND
    ]
    factor_manifest = store.get_object(request_payload["factor_validator_manifest_ref"])
    if factor_manifest["kind"] != FACTOR_VALIDATOR_MANIFEST_KIND:
        raise SystemContractError("validation request Factor manifest kind is invalid")
    components = _validate_component_closure(
        store,
        profile=profile,
        request=request_payload,
        factor_manifest=factor_manifest,
    )
    source_projection = _source_snapshot_rows(
        store,
        source_refs,
        maximum_total_bytes=profile["maximum_total_factor_source_bytes"],
        full_hash=full_source_hash,
    )
    custody_refs = _custody_chain(store, candidate)
    custody_tree_sha = hashlib.sha256(canonical_json_bytes(custody_refs)).hexdigest()
    custody_head_ref = custody_refs[-1] if custody_refs else None
    source_attestation_refs = _sorted_refs(source_attestation_refs, label="source_attestation_refs")
    _validate_source_attestation_closure(
        store,
        source_attestation_refs=source_attestation_refs,
        source_object_refs=source_projection["source_object_refs"],
        factor_validator_manifest_ref=request_payload["factor_validator_manifest_ref"],
        components=components,
    )
    del artifacts
    release = store.get_object(request_payload["release_manifest_ref"])
    release_payload = release["payload"]
    installed_sha = installed_code_manifest_sha256()
    if release_payload.get("code_manifest_sha256") != installed_sha:
        raise SystemPreconditionError("installed code manifest differs from request release")
    plan = {
        "domain": "myquant-validation-run-plan",
        "validation_namespace_id": request_payload["validation_namespace_id"],
        "validation_profile_id": request_payload["validation_profile_id"],
        "validation_lane": profile["validation_lane"],
        "component_registry_sha256": COMPONENT_REGISTRY_SHA256,
        "release_manifest_ref": request_payload["release_manifest_ref"],
        "installed_code_manifest_sha256": installed_sha,
        "factor_validator_manifest_ref": request_payload["factor_validator_manifest_ref"],
        **components,
        "intrinsic_receipt_ref": receipt["intrinsic_receipt_ref"],
        "policy_ref": receipt["policy_ref"],
        "evidence_refs": receipt["evidence_refs"],
        "active_set_ref": receipt["active_set_ref"],
        "candidate_state_ref": candidate_ref,
        "candidate_state_pointer_sha256": candidate_pointer_sha,
        "source_attestation_refs": source_attestation_refs,
        "source_object_refs": source_projection["source_object_refs"],
        "source_stat_rows": source_projection["source_stat_rows"],
        "source_stat_tree_sha256": source_projection["source_stat_tree_sha256"],
        "factor_source_total_bytes": source_projection["factor_source_total_bytes"],
        "maximum_total_factor_source_bytes": profile["maximum_total_factor_source_bytes"],
        "custody_record_refs": custody_refs,
        "custody_head_ref": custody_head_ref,
        "custody_tree_sha256": custody_tree_sha,
        "compiled_contracts": components["compiled_contracts"],
    }
    return {
        "profile": profile,
        "receipt": receipt,
        "candidate": candidate,
        "factor_manifest": factor_manifest,
        "release": release,
        "plan": plan,
        "plan_sha256": hashlib.sha256(canonical_json_bytes(plan)).hexdigest(),
    }


def _parse_time(value: str) -> datetime:
    return datetime.strptime(
        _require_timestamp(value, label="trusted time"), "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=timezone.utc)


def _clock_floor(store: Any, namespace_id: str, derived: Mapping[str, Any]) -> datetime | None:
    values: list[str] = []
    candidate = derived.get("candidate")
    if candidate is not None:
        values.append(candidate["payload"]["last_stored_at"])
    for ref in derived["plan"]["custody_record_refs"]:
        values.append(store.get_object(ref)["payload"]["stored_at"])
    paths = _run_paths(namespace_id, "0" * 64)
    try:
        intent_names = store._storage.list_directory_names(paths["intents"], directories_only=True)
    except SystemNotFound:
        intent_names = ()
    for name in intent_names:
        if _SHA256_RE.fullmatch(name) is None:
            raise SystemSecurityError("validation intent directory name is invalid")
        files = store._storage.read_exact_directory(
            paths["intents"] / name, expected_names=frozenset({"intent.json"})
        )
        values.append(validate_validation_intent(files["intent.json"].data)["trusted_at"])
    return max((_parse_time(value) for value in values), default=None)


def _read_record_directory(
    store: Any,
    path: PurePosixPath,
    filename: str,
    validator: Any,
) -> tuple[dict[str, Any], bytes] | None:
    try:
        files = store._storage.read_exact_directory(path, expected_names=frozenset({filename}))
    except SystemNotFound:
        return None
    raw = files[filename].data
    return validator(raw), raw


def _open_fd_count() -> int:
    try:
        return len(os.listdir("/dev/fd"))
    except OSError:
        return 0


def _maximum_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _invoke_callback(
    store: Any,
    *,
    profile: Mapping[str, Any],
    validation_request: Mapping[str, Any],
    trusted_at: str,
) -> dict[str, Any]:
    if _open_fd_count() > MAXIMUM_VALIDATION_OPEN_FDS:
        raise SystemSecurityError("validation runner file descriptor bound exceeded")
    before_rss = _maximum_rss_bytes()
    if before_rss > MAXIMUM_VALIDATION_RSS_BYTES:
        raise SystemSecurityError("contextual validation RSS bound exceeded")
    started = time.monotonic()
    try:
        module = importlib.import_module(profile["callback_module"])
        callback = getattr(module, profile["callback_qualified_name"])
    except (ImportError, AttributeError) as exc:
        raise SystemPreconditionError("compiled validation callback is unavailable") from exc
    if (
        not callable(callback)
        or getattr(callback, "__module__", None) != profile["callback_module"]
    ):
        raise SystemPreconditionError("compiled validation callback identity mismatch")
    result = callback(
        system_store=store,
        validation_request=dict(validation_request),
        trusted_at=trusted_at,
    )
    elapsed = time.monotonic() - started
    maximum_seconds = (
        MAXIMUM_BOOTSTRAP_VALIDATION_SECONDS
        if profile["validation_lane"] == "BOOTSTRAP"
        else MAXIMUM_PROSPECTIVE_VALIDATION_SECONDS
    )
    if elapsed > maximum_seconds:
        raise SystemSecurityError("contextual validation time bound exceeded")
    if _open_fd_count() > MAXIMUM_VALIDATION_OPEN_FDS:
        raise SystemSecurityError("validation runner file descriptor bound exceeded")
    after_rss = _maximum_rss_bytes()
    if after_rss > MAXIMUM_VALIDATION_RSS_BYTES:
        raise SystemSecurityError("contextual validation RSS bound exceeded")
    if type(result) is not dict:
        raise SystemContractError("compiled validation callback did not return an exact payload")
    return dict(result)


def _validate_context_payload(payload: Mapping[str, Any], derived: Mapping[str, Any]) -> None:
    plan = derived["plan"]
    expected = {
        "validation_namespace_id": plan["validation_namespace_id"],
        "lane": plan["validation_lane"],
        "intrinsic_receipt_ref": plan["intrinsic_receipt_ref"],
        "policy_ref": plan["policy_ref"],
        "evidence_refs": plan["evidence_refs"],
        "active_set_ref": plan["active_set_ref"],
        "composite_state_ref": plan["candidate_state_ref"],
        "factor_validator_manifest_ref": plan["factor_validator_manifest_ref"],
        "contextual_validator_component_ref": plan["contextual_validator_component_ref"],
        "source_decoder_component_ref": plan["source_decoder_component_ref"],
        "implementation_component_refs": plan["implementation_component_refs"],
        "source_attestation_refs": plan["source_attestation_refs"],
        "source_object_refs": plan["source_object_refs"],
        "custody_record_refs": plan["custody_record_refs"],
        "custody_tree_sha256": plan["custody_tree_sha256"],
        "custody_head_ref": plan["custody_head_ref"],
        "validated": True,
        "blockers": [],
        "authority": "NON_AUTHORIZING",
    }
    if {key: payload.get(key) for key in expected} != expected:
        raise SystemContractError("contextual result differs from derived validation closure")


def _attestation_payload(
    *,
    request_ref: Mapping[str, Any],
    request_payload: Mapping[str, Any],
    derived: Mapping[str, Any],
    intent: Mapping[str, Any],
    intent_sha256: str,
    contextual_result_ref: Mapping[str, Any],
    trusted_at: str,
) -> dict[str, Any]:
    plan = derived["plan"]
    release = derived["release"]["payload"]
    identity = {
        "validation_request_ref": dict(request_ref),
        "validation_plan_sha256": derived["plan_sha256"],
        "contextual_result_ref": dict(contextual_result_ref),
        "validated_at": trusted_at,
    }
    return {
        "attestation_id": _domain_identity("system.validation_attestation", **identity),
        "validation_request_ref": dict(request_ref),
        "validation_profile_id": request_payload["validation_profile_id"],
        "component_registry_sha256": COMPONENT_REGISTRY_SHA256,
        "validation_namespace_id": request_payload["validation_namespace_id"],
        "validation_lane": plan["validation_lane"],
        "validation_intent_sha256": intent_sha256,
        "validation_plan_sha256": derived["plan_sha256"],
        "candidate_state_ref": plan["candidate_state_ref"],
        "candidate_state_pointer_sha256": plan["candidate_state_pointer_sha256"],
        "contextual_result_ref": dict(contextual_result_ref),
        "intrinsic_receipt_ref": plan["intrinsic_receipt_ref"],
        "policy_ref": plan["policy_ref"],
        "evidence_refs": plan["evidence_refs"],
        "active_set_ref": plan["active_set_ref"],
        "source_object_refs": plan["source_object_refs"],
        "release_manifest_ref": request_payload["release_manifest_ref"],
        "release_identity": {
            "release_id": release["release_id"],
            "code_sha256": release["code_sha256"],
            "wheel_sha256": release["wheel_sha256"],
            "code_manifest_sha256": release["code_manifest_sha256"],
        },
        "installed_code_manifest_sha256": plan["installed_code_manifest_sha256"],
        "compiled_contracts": plan["compiled_contracts"],
        "factor_validator_manifest_ref": plan["factor_validator_manifest_ref"],
        "contextual_validator_component_ref": plan["contextual_validator_component_ref"],
        "source_decoder_component_ref": plan["source_decoder_component_ref"],
        "implementation_component_refs": plan["implementation_component_refs"],
        "source_attestation_refs": plan["source_attestation_refs"],
        "custody_record_refs": plan["custody_record_refs"],
        "custody_head_ref": plan["custody_head_ref"],
        "custody_tree_sha256": plan["custody_tree_sha256"],
        "factor_source_stat_tree_sha256": plan["source_stat_tree_sha256"],
        "factor_source_total_bytes": plan["factor_source_total_bytes"],
        "maximum_total_factor_source_bytes": plan["maximum_total_factor_source_bytes"],
        "validated_at": trusted_at,
        "clock_source": "SYSTEM_UTC",
        "outcome": "VALIDATED",
        "authority": "NON_AUTHORIZING",
    }


def _expected_intent(
    *,
    intent_id: str,
    request_ref: Mapping[str, Any],
    request_payload: Mapping[str, Any],
    derived: Mapping[str, Any],
    trusted_at: str,
) -> dict[str, Any]:
    plan = derived["plan"]
    return build_validation_intent(
        candidate_state_pointer_sha256=plan["candidate_state_pointer_sha256"],
        candidate_state_ref=plan["candidate_state_ref"],
        component_registry_sha256=COMPONENT_REGISTRY_SHA256,
        factor_source_object_count=len(plan["source_object_refs"]),
        factor_source_stat_tree_sha256=plan["source_stat_tree_sha256"],
        factor_source_total_bytes=plan["factor_source_total_bytes"],
        factor_validator_manifest_ref=request_payload["factor_validator_manifest_ref"],
        installed_code_manifest_sha256=plan["installed_code_manifest_sha256"],
        intent_id=intent_id,
        intrinsic_receipt_ref=request_payload["intrinsic_receipt_ref"],
        maximum_total_factor_source_bytes=plan["maximum_total_factor_source_bytes"],
        plan_sha256=derived["plan_sha256"],
        release_manifest_ref=request_payload["release_manifest_ref"],
        trusted_at=trusted_at,
        validation_lane=plan["validation_lane"],
        validation_namespace_id=request_payload["validation_namespace_id"],
        validation_profile_id=request_payload["validation_profile_id"],
        validation_request_ref=dict(request_ref),
    )


def _assert_plan_stable(
    store: Any,
    *,
    request_payload: Mapping[str, Any],
    derived: Mapping[str, Any],
) -> None:
    """Reopen every bound source/component and reject drift before completion."""

    observed = _derive_plan(store, request_payload, full_source_hash=False)
    if observed["plan_sha256"] != derived["plan_sha256"] or observed["plan"] != derived["plan"]:
        raise SystemPreconditionError("validation plan changed during execution")


def _publish_custody(
    store: Any,
    *,
    request_ref: Mapping[str, Any],
    derived: Mapping[str, Any],
    contextual_result: Mapping[str, Any],
    contextual_result_ref: Mapping[str, Any],
    attestation: Mapping[str, Any],
    attestation_ref: Mapping[str, Any],
    trusted_at: str,
) -> tuple[dict[str, Any], str]:
    record = build_custody_record(
        record_id=custody_record_id(attestation_ref),
        validation_request_ref=dict(request_ref),
        attestation_ref=dict(attestation_ref),
        contextual_result_ref=dict(contextual_result_ref),
        release_manifest_ref=derived["plan"]["release_manifest_ref"],
        component_registry_sha256=COMPONENT_REGISTRY_SHA256,
        recorded_at=trusted_at,
        os_actor=f"uid:{os.geteuid()}",
    )
    files = {
        "contextual_result.json": canonical_json_bytes(contextual_result),
        "attestation.json": canonical_json_bytes(attestation),
        "record.json": canonical_json_bytes(record),
    }
    directory = VALIDATION_CUSTODY_ROOT / attestation_ref["byte_sha256"]
    readback = store._storage.write_atomic_directory(directory, files)
    if any(readback[name].data != raw for name, raw in files.items()):
        raise SystemStorageError("protected validation custody readback mismatch")
    return record, hashlib.sha256(files["record.json"]).hexdigest()


def _publish_snapshot(
    store: Any,
    *,
    attestation_ref: Mapping[str, Any],
    derived: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    plan = derived["plan"]
    snapshot = build_source_verification_snapshot(
        factor_source_total_bytes=plan["factor_source_total_bytes"],
        installed_code_manifest_sha256=plan["installed_code_manifest_sha256"],
        maximum_total_factor_source_bytes=plan["maximum_total_factor_source_bytes"],
        source_object_count=len(plan["source_object_refs"]),
        source_object_refs=plan["source_object_refs"],
        source_stat_rows=plan["source_stat_rows"],
        source_stat_tree_sha256=plan["source_stat_tree_sha256"],
        unique_source_binding_count=len(plan["source_stat_rows"]),
        validation_attestation_ref=dict(attestation_ref),
    )
    raw = canonical_json_bytes(snapshot)
    directory = SOURCE_VERIFICATION_CACHE_ROOT / attestation_ref["byte_sha256"]
    readback = store._storage.write_atomic_directory(directory, {"snapshot.json": raw})
    if readback["snapshot.json"].data != raw:
        raise SystemStorageError("source snapshot exact readback mismatch")
    return snapshot, hashlib.sha256(raw).hexdigest()


def _result_projection(
    *,
    request: Mapping[str, Any],
    request_ref: Mapping[str, Any],
    intent: Mapping[str, Any],
    prepared: Mapping[str, Any],
    contextual_result: Mapping[str, Any],
    contextual_result_ref: Mapping[str, Any],
    attestation: Mapping[str, Any],
    attestation_ref: Mapping[str, Any],
    custody_record: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
    completion: Mapping[str, Any],
    completion_sha256: str,
) -> dict[str, Any]:
    return {
        "outcome": "VALIDATED",
        "validation_request": dict(request),
        "validation_request_ref": dict(request_ref),
        "validation_intent": dict(intent),
        "validation_prepared": dict(prepared),
        "contextual_result": dict(contextual_result),
        "contextual_result_ref": dict(contextual_result_ref),
        "validation_attestation": dict(attestation),
        "validation_attestation_ref": dict(attestation_ref),
        "custody_record": dict(custody_record),
        "source_verification_snapshot": dict(source_snapshot),
        "validation_completion": dict(completion),
        "completion_sha256": completion_sha256,
    }


def run_validation(
    store: Any, request_value: Mapping[str, Any] | bytes
) -> dict[str, Any]:  # noqa: C901
    """Execute or exactly recover one fixed-profile contextual validation run."""

    request, request_ref = _load_request(store, request_value)
    request_payload = _validate_request_identity(store, request)
    intent_id = validation_intent_id(
        request_payload["validation_namespace_id"], request_payload["validation_request_id"]
    )
    paths = _run_paths(request_payload["validation_namespace_id"], intent_id)
    with store._storage.exclusive_lock(paths["lock"]):
        existing_completion = _read_record_directory(
            store,
            paths["completion"],
            "completion.json",
            validate_validation_completion,
        )
        if existing_completion is not None:
            return resolve_validation_attestation(
                store,
                existing_completion[0]["validation_attestation_ref"],
                verification_level="full",
            )

        derived = _derive_plan(store, request_payload, full_source_hash=True)
        existing_intent = _read_record_directory(
            store, paths["intent"], "intent.json", validate_validation_intent
        )
        if existing_intent is None:
            trusted_at = _utc_now()
            floor = _clock_floor(store, request_payload["validation_namespace_id"], derived)
            if floor is not None and _parse_time(trusted_at) <= floor:
                raise SystemPreconditionError(
                    "system clock is not strictly after validation custody",
                    code="SYSTEM_CLOCK_ROLLBACK",
                )
            intent = _expected_intent(
                intent_id=intent_id,
                request_ref=request_ref,
                request_payload=request_payload,
                derived=derived,
                trusted_at=trusted_at,
            )
            intent_raw = canonical_json_bytes(intent)
            store._storage.write_atomic_directory(paths["intent"], {"intent.json": intent_raw})
        else:
            intent, intent_raw = existing_intent
            trusted_at = intent["trusted_at"]
            if _parse_time(_utc_now()) < _parse_time(trusted_at):
                raise SystemPreconditionError(
                    "system clock precedes incomplete validation intent",
                    code="SYSTEM_CLOCK_ROLLBACK",
                )
            if intent != _expected_intent(
                intent_id=intent_id,
                request_ref=request_ref,
                request_payload=request_payload,
                derived=derived,
                trusted_at=trusted_at,
            ):
                raise SystemImmutableConflict("validation intent plan has drifted")
        intent_sha = hashlib.sha256(intent_raw).hexdigest()

        existing_prepared = _read_record_directory(
            store,
            paths["prepared_record"],
            "prepared.json",
            validate_validation_prepared,
        )
        if existing_prepared is None:
            context_payload = _invoke_callback(
                store,
                profile=derived["profile"],
                validation_request=request,
                trusted_at=trusted_at,
            )
            _validate_context_payload(context_payload, derived)
            contextual_result = seal_artifact(
                CONTEXTUAL_RESULT_KIND, context_payload, created_at=trusted_at
            )
            contextual_result_ref = store.put_object(contextual_result)
            attestation_payload = _attestation_payload(
                request_ref=request_ref,
                request_payload=request_payload,
                derived=derived,
                intent=intent,
                intent_sha256=intent_sha,
                contextual_result_ref=contextual_result_ref,
                trusted_at=trusted_at,
            )
            attestation = seal_artifact(
                VALIDATION_ATTESTATION_KIND,
                attestation_payload,
                created_at=trusted_at,
            )
            attestation_ref = store.put_object(attestation)
            prepared = build_validation_prepared(
                contextual_result_ref=contextual_result_ref,
                intent_id=intent_id,
                intent_semantic_sha256=intent["semantic_sha256"],
                intent_sha256=intent_sha,
                plan_sha256=derived["plan_sha256"],
                prepared_id=prepared_id(intent_id),
                trusted_at=trusted_at,
                validation_attestation_ref=attestation_ref,
                validation_namespace_id=request_payload["validation_namespace_id"],
                validation_request_ref=request_ref,
            )
            prepared_raw = canonical_json_bytes(prepared)
            store._storage.write_atomic_directory(
                paths["prepared_record"], {"prepared.json": prepared_raw}
            )
        else:
            prepared, prepared_raw = existing_prepared
            if (
                prepared["intent_sha256"] != intent_sha
                or prepared["intent_semantic_sha256"] != intent["semantic_sha256"]
                or prepared["plan_sha256"] != derived["plan_sha256"]
                or prepared["validation_request_ref"] != request_ref
                or prepared["trusted_at"] != trusted_at
            ):
                raise SystemImmutableConflict("validation prepared mapping has drifted")
            contextual_result_ref = prepared["contextual_result_ref"]
            attestation_ref = prepared["validation_attestation_ref"]
            contextual_result = store.get_object(contextual_result_ref)
            attestation = store.get_object(attestation_ref)
            _validate_context_payload(contextual_result["payload"], derived)
            expected_attestation_payload = _attestation_payload(
                request_ref=request_ref,
                request_payload=request_payload,
                derived=derived,
                intent=intent,
                intent_sha256=intent_sha,
                contextual_result_ref=contextual_result_ref,
                trusted_at=trusted_at,
            )
            if attestation["payload"] != expected_attestation_payload:
                raise SystemImmutableConflict("prepared attestation has drifted")

        _assert_plan_stable(
            store,
            request_payload=request_payload,
            derived=derived,
        )
        custody_record, custody_sha = _publish_custody(
            store,
            request_ref=request_ref,
            derived=derived,
            contextual_result=contextual_result,
            contextual_result_ref=contextual_result_ref,
            attestation=attestation,
            attestation_ref=attestation_ref,
            trusted_at=trusted_at,
        )
        snapshot, snapshot_sha = _publish_snapshot(
            store, attestation_ref=attestation_ref, derived=derived
        )
        _assert_plan_stable(
            store,
            request_payload=request_payload,
            derived=derived,
        )
        completion = build_validation_completion(
            completion_id=completion_id(intent_id),
            contextual_result_ref=contextual_result_ref,
            custody_record_sha256=custody_sha,
            intent_semantic_sha256=intent["semantic_sha256"],
            intent_sha256=intent_sha,
            prepared_sha256=hashlib.sha256(prepared_raw).hexdigest(),
            source_verification_snapshot_sha256=snapshot_sha,
            trusted_at=trusted_at,
            validation_attestation_ref=attestation_ref,
            validation_namespace_id=request_payload["validation_namespace_id"],
            validation_request_ref=request_ref,
        )
        completion_raw = canonical_json_bytes(completion)
        store._storage.write_atomic_directory(
            paths["completion"], {"completion.json": completion_raw}
        )
        return _result_projection(
            request=request,
            request_ref=request_ref,
            intent=intent,
            prepared=prepared,
            contextual_result=contextual_result,
            contextual_result_ref=contextual_result_ref,
            attestation=attestation,
            attestation_ref=attestation_ref,
            custody_record=custody_record,
            source_snapshot=snapshot,
            completion=completion,
            completion_sha256=hashlib.sha256(completion_raw).hexdigest(),
        )


def resolve_validation_attestation(  # noqa: C901
    store: Any,
    attestation_value: Mapping[str, Any],
    *,
    verification_level: str,
) -> dict[str, Any]:
    """Resolve the unique completion/custody/snapshot for one attestation ref."""

    if verification_level not in {"stat", "full"}:
        raise SystemContractError("validation verification level is invalid")
    attestation_ref = validate_object_ref(attestation_value, label="validation_attestation_ref")
    if attestation_ref["kind"] != VALIDATION_ATTESTATION_KIND:
        raise SystemContractError("validation attestation ref has the wrong kind")
    attestation = store.get_object(attestation_ref)
    payload = attestation["payload"]
    if (
        payload.get("outcome") != "VALIDATED"
        or payload.get("authority") != "NON_AUTHORIZING"
        or payload.get("clock_source") != "SYSTEM_UTC"
    ):
        raise SystemContractError("validation attestation is not validated/non-authorizing")
    request = store.get_object(payload["validation_request_ref"])
    request_payload = _validate_request_identity(store, request)
    request_ref = object_ref_for_artifact(request)
    if request_ref != payload["validation_request_ref"]:
        raise SystemContractError("validation attestation request binding mismatch")
    intent_id = validation_intent_id(
        request_payload["validation_namespace_id"], request_payload["validation_request_id"]
    )
    paths = _run_paths(request_payload["validation_namespace_id"], intent_id)
    intent_row = _read_record_directory(
        store, paths["intent"], "intent.json", validate_validation_intent
    )
    prepared_row = _read_record_directory(
        store,
        paths["prepared_record"],
        "prepared.json",
        validate_validation_prepared,
    )
    completion_row = _read_record_directory(
        store,
        paths["completion"],
        "completion.json",
        validate_validation_completion,
    )
    if intent_row is None or prepared_row is None or completion_row is None:
        raise SystemPreconditionError("validation run is incomplete")
    intent, intent_raw = intent_row
    prepared, prepared_raw = prepared_row
    completion, completion_raw = completion_row
    contextual_result_ref = prepared["contextual_result_ref"]
    if prepared["validation_attestation_ref"] != attestation_ref:
        raise SystemContractError("prepared attestation ref mismatch")
    contextual_result = store.get_object(contextual_result_ref)
    derived = _derive_plan(
        store,
        request_payload,
        full_source_hash=verification_level == "full",
    )
    if intent != _expected_intent(
        intent_id=intent_id,
        request_ref=request_ref,
        request_payload=request_payload,
        derived=derived,
        trusted_at=intent["trusted_at"],
    ):
        raise SystemContractError("validation intent cannot be reconstructed")
    expected_prepared = build_validation_prepared(
        contextual_result_ref=prepared["contextual_result_ref"],
        intent_id=intent_id,
        intent_semantic_sha256=intent["semantic_sha256"],
        intent_sha256=hashlib.sha256(intent_raw).hexdigest(),
        plan_sha256=derived["plan_sha256"],
        prepared_id=prepared_id(intent_id),
        trusted_at=intent["trusted_at"],
        validation_attestation_ref=attestation_ref,
        validation_namespace_id=request_payload["validation_namespace_id"],
        validation_request_ref=request_ref,
    )
    if prepared != expected_prepared or prepared_raw != canonical_json_bytes(expected_prepared):
        raise SystemContractError("validation prepared mapping cannot be reconstructed")
    _validate_context_payload(contextual_result["payload"], derived)
    intent_sha = hashlib.sha256(intent_raw).hexdigest()
    expected_attestation = _attestation_payload(
        request_ref=request_ref,
        request_payload=request_payload,
        derived=derived,
        intent=intent,
        intent_sha256=intent_sha,
        contextual_result_ref=contextual_result_ref,
        trusted_at=intent["trusted_at"],
    )
    if payload != expected_attestation:
        raise SystemContractError("validation attestation cannot be reconstructed")
    custody_files = store._storage.read_exact_directory(
        VALIDATION_CUSTODY_ROOT / attestation_ref["byte_sha256"],
        expected_names=frozenset({"contextual_result.json", "attestation.json", "record.json"}),
    )
    if custody_files["contextual_result.json"].data != canonical_json_bytes(
        contextual_result
    ) or custody_files["attestation.json"].data != canonical_json_bytes(attestation):
        raise SystemContractError("protected validation custody artifact mismatch")
    custody_record = validate_custody_record(custody_files["record.json"].data)
    expected_custody = build_custody_record(
        record_id=custody_record_id(attestation_ref),
        validation_request_ref=request_ref,
        attestation_ref=attestation_ref,
        contextual_result_ref=contextual_result_ref,
        release_manifest_ref=derived["plan"]["release_manifest_ref"],
        component_registry_sha256=COMPONENT_REGISTRY_SHA256,
        recorded_at=intent["trusted_at"],
        os_actor=f"uid:{os.geteuid()}",
    )
    if custody_record != expected_custody:
        raise SystemContractError("protected validation custody record mismatch")
    snapshot_files = store._storage.read_exact_directory(
        SOURCE_VERIFICATION_CACHE_ROOT / attestation_ref["byte_sha256"],
        expected_names=frozenset({"snapshot.json"}),
    )
    snapshot_raw = snapshot_files["snapshot.json"].data
    snapshot = validate_source_verification_snapshot(snapshot_raw)
    source_projection = _source_snapshot_rows(
        store,
        payload["source_object_refs"],
        maximum_total_bytes=payload["maximum_total_factor_source_bytes"],
        full_hash=verification_level == "full",
    )
    expected_snapshot = build_source_verification_snapshot(
        factor_source_total_bytes=source_projection["factor_source_total_bytes"],
        installed_code_manifest_sha256=payload["installed_code_manifest_sha256"],
        maximum_total_factor_source_bytes=payload["maximum_total_factor_source_bytes"],
        source_object_count=source_projection["source_object_count"],
        source_object_refs=source_projection["source_object_refs"],
        source_stat_rows=source_projection["source_stat_rows"],
        source_stat_tree_sha256=source_projection["source_stat_tree_sha256"],
        unique_source_binding_count=source_projection["unique_source_binding_count"],
        validation_attestation_ref=attestation_ref,
    )
    if snapshot != expected_snapshot:
        raise SystemContractError("source verification snapshot has drifted")
    expected_completion = build_validation_completion(
        completion_id=completion_id(intent_id),
        contextual_result_ref=contextual_result_ref,
        custody_record_sha256=hashlib.sha256(custody_files["record.json"].data).hexdigest(),
        intent_semantic_sha256=intent["semantic_sha256"],
        intent_sha256=intent_sha,
        prepared_sha256=hashlib.sha256(prepared_raw).hexdigest(),
        source_verification_snapshot_sha256=hashlib.sha256(snapshot_raw).hexdigest(),
        trusted_at=intent["trusted_at"],
        validation_attestation_ref=attestation_ref,
        validation_namespace_id=request_payload["validation_namespace_id"],
        validation_request_ref=request_ref,
    )
    if completion != expected_completion or completion_raw != canonical_json_bytes(
        expected_completion
    ):
        raise SystemContractError("validation completion cannot be reconstructed")
    return _result_projection(
        request=request,
        request_ref=request_ref,
        intent=intent,
        prepared=prepared,
        contextual_result=contextual_result,
        contextual_result_ref=contextual_result_ref,
        attestation=attestation,
        attestation_ref=attestation_ref,
        custody_record=custody_record,
        source_snapshot=snapshot,
        completion=completion,
        completion_sha256=hashlib.sha256(completion_raw).hexdigest(),
    )


__all__ = [
    "MAXIMUM_BOOTSTRAP_VALIDATION_SECONDS",
    "MAXIMUM_PROSPECTIVE_VALIDATION_SECONDS",
    "MAXIMUM_VALIDATION_OBJECTS",
    "MAXIMUM_VALIDATION_OPEN_FDS",
    "build_validation_run_request",
    "resolve_validation_attestation",
    "run_validation",
]
