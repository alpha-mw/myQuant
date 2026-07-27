"""Independent canonical JSON and closed-schema validation for runtime routing."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import re
from typing import Any, Mapping

TARGET_VERSION = "myquant.research-runtime.protocol-target.v1"
RUN_VERSION = "myquant.research-runtime.run.v1"
ACTIVE_POINTER_VERSION = "myquant.research-runtime.active-run-pointer.v1"
SELECTOR_VERSION = "myquant.research-runtime.default-protocol-selector.v1"
INTENT_VERSION = "myquant.research-runtime.route-transition-intent.v1"
BOOTSTRAP_RECEIPT_VERSION = "myquant.research-runtime.route-bootstrap-receipt.v1"
CUTOVER_RECEIPT_VERSION = "myquant.research-runtime.cutover-receipt.v1"
ROLLBACK_RECEIPT_VERSION = "myquant.research-runtime.rollback-receipt.v1"
V4_FORMAL_ACTIVE_POINTER_VERSION = (
    "myquant.v17.v4.formal-active-pointer.v1"
)
V4_REGISTERED_REFERENCE_VERSIONS = frozenset(
    {
        "myquant.v17.v4.canary-pointer.v1",
        "myquant.v17.v4.canary-receipt.v1",
        "myquant.v17.v4.default-eligibility-receipt.v1",
        "myquant.v17.v4.default-eligible-pointer.v1",
        "myquant.v17.v4.dual-run-comparison.v1",
        "myquant.v17.v4.formal-activation-intent.v1",
        "myquant.v17.v4.formal-activation-receipt.v1",
        "myquant.v17.v4.formal-activation-rejection.v1",
        V4_FORMAL_ACTIVE_POINTER_VERSION,
        "myquant.v17.v4.formal-output.v1",
        "myquant.v17.v4.historical-canary-policy.v1",
    }
)

_SHA_RE = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$", re.ASCII)
_UTC_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$",
    re.ASCII,
)
_AUTHORITY_KEYS = {
    "broker",
    "execution",
    "order",
    "trade",
}
_REF_KEYS = {
    "artifact_id",
    "artifact_version",
    "byte_sha256",
    "cutoff",
    "relative_path",
    "semantic_sha256",
    "strategy_id",
}
_INTENT_IDENTITY_KEYS = {"intent_id", "relative_path", "version"}


class CanonicalControlError(ValueError):
    """Canonical bytes or a neutral control artifact failed closed."""


def sha256(raw: bytes) -> str:
    if type(raw) is not bytes:
        raise CanonicalControlError("SHA-256 input must be bytes")
    return hashlib.sha256(raw).hexdigest()


def _json_value(value: Any, *, path: str = "$") -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _json_value(item, path=f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise CanonicalControlError(f"{path} contains a non-text key")
            _json_value(item, path=f"{path}.{key}")
        return
    raise CanonicalControlError(f"{path} contains an unsupported JSON value")


def canonical_bytes(value: Any) -> bytes:
    _json_value(value)
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _v4_canonical_resource_bytes(value: Any) -> bytes:
    _json_value(value)
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _v4_semantic_sha256(value: Mapping[str, Any]) -> str:
    payload = _semantic_payload(value)
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _semantic_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    return payload


def seal(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CanonicalControlError("artifact root must be an object")
    payload = _semantic_payload(value)
    payload["semantic_sha256"] = (
        _v4_semantic_sha256(payload)
        if str(payload.get("version", "")).startswith("myquant.v17.v4.")
        else sha256(canonical_bytes(payload))
    )
    validate(payload)
    return payload


def encode(value: Mapping[str, Any]) -> bytes:
    validate(value)
    document = dict(value)
    if str(document.get("version", "")).startswith("myquant.v17.v4."):
        return _v4_canonical_resource_bytes(document)
    return canonical_bytes(document)


def decode(raw: bytes, *, expected_version: str | None = None) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise CanonicalControlError("artifact input must be exact bytes")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanonicalControlError("artifact is not canonical JSON") from exc
    if type(value) is not dict or canonical_bytes(value) != raw:
        raise CanonicalControlError("artifact bytes are not canonical")
    if expected_version is not None and value.get("version") != expected_version:
        raise CanonicalControlError("artifact version mismatch")
    validate(value)
    return value


def decode_reference(
    raw: bytes,
    *,
    expected_version: str | None = None,
) -> dict[str, Any]:
    """Decode a neutral artifact or canonical v4 evidence without importing v4."""
    if type(raw) is not bytes:
        raise CanonicalControlError("artifact input must be exact bytes")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanonicalControlError("artifact is not canonical JSON") from exc
    if type(value) is not dict:
        raise CanonicalControlError("artifact bytes are not canonical")
    version = value.get("version")
    expected_raw = (
        _v4_canonical_resource_bytes(value)
        if type(version) is str and version.startswith("myquant.v17.v4.")
        else canonical_bytes(value)
    )
    if expected_raw != raw:
        raise CanonicalControlError("artifact bytes are not canonical")
    if expected_version is not None and version != expected_version:
        raise CanonicalControlError("artifact version mismatch")
    if version in _VALIDATORS:
        validate(value)
        return value
    if (
        type(version) is not str
        or not version.startswith("myquant.v17.v4.")
        or version not in V4_REGISTERED_REFERENCE_VERSIONS
    ):
        raise CanonicalControlError("referenced artifact version is not admitted")
    semantic = value.get("semantic_sha256")
    _digest(semantic, label="semantic_sha256")
    if semantic != _v4_semantic_sha256(value):
        raise CanonicalControlError("artifact semantic SHA-256 mismatch")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], *, label: str) -> None:
    if set(value) != keys:
        raise CanonicalControlError(f"{label} shape is not closed")


def _identifier(value: Any, *, label: str) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        raise CanonicalControlError(f"{label} is not a canonical identifier")
    return value


def _digest(value: Any, *, label: str, allow_empty: bool = False) -> str:
    if allow_empty and value == "EMPTY":
        return value
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise CanonicalControlError(f"{label} is not a canonical SHA-256")
    return value


def _instant(value: Any, *, label: str) -> str:
    if type(value) is not str or _UTC_RE.fullmatch(value) is None:
        raise CanonicalControlError(f"{label} is not a UTC instant")
    try:
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise CanonicalControlError(f"{label} is not a real UTC instant") from exc
    return value


def _path(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or "\\" in value
        or value.startswith("/")
        or "//" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise CanonicalControlError(f"{label} is not a canonical relative path")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise CanonicalControlError(f"{label} must be ASCII") from exc
    return value


def _authority(value: Any) -> None:
    if type(value) is not dict:
        raise CanonicalControlError("authority_ceiling must be an object")
    _exact(value, _AUTHORITY_KEYS, label="authority_ceiling")
    if any(value[key] is not False for key in _AUTHORITY_KEYS):
        raise CanonicalControlError("execution authority ceiling must remain false")


def _reference(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise CanonicalControlError(f"{label} must be an exact reference")
    _exact(value, _REF_KEYS, label=label)
    _identifier(value["artifact_id"], label=f"{label}.artifact_id")
    _identifier(value["artifact_version"], label=f"{label}.artifact_version")
    _identifier(value["strategy_id"], label=f"{label}.strategy_id")
    _instant(value["cutoff"], label=f"{label}.cutoff")
    _path(value["relative_path"], label=f"{label}.relative_path")
    _digest(value["byte_sha256"], label=f"{label}.byte_sha256")
    _digest(value["semantic_sha256"], label=f"{label}.semantic_sha256")
    return value


def _references(value: Any, *, label: str) -> None:
    if type(value) is not list:
        raise CanonicalControlError(f"{label} must be an array")
    previous: tuple[str, str] | None = None
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(value):
        ref = _reference(item, label=f"{label}[{index}]")
        key = (ref["relative_path"], ref["byte_sha256"])
        if key in seen or (previous is not None and key <= previous):
            raise CanonicalControlError(f"{label} must be uniquely sorted")
        previous = key
        seen.add(key)


def _intent_identity(value: Any) -> None:
    if type(value) is not dict:
        raise CanonicalControlError("transition_intent_ref must be an object")
    _exact(value, _INTENT_IDENTITY_KEYS, label="transition_intent_ref")
    _identifier(value["intent_id"], label="transition_intent_ref.intent_id")
    if value["version"] != INTENT_VERSION:
        raise CanonicalControlError("transition intent version mismatch")
    _path(value["relative_path"], label="transition_intent_ref.relative_path")


def _validate_target(value: Mapping[str, Any]) -> None:
    _exact(
        value,
        {
            "version",
            "target_id",
            "protocol_id",
            "strategy_scope",
            "active_run_pointer_template",
            "authority_ceiling",
            "semantic_sha256",
        },
        label="protocol target",
    )
    _identifier(value["target_id"], label="target_id")
    protocol_id = value["protocol_id"]
    if protocol_id not in {"v15", "myquant.v17.v4"}:
        raise CanonicalControlError("protocol target protocol_id is closed")
    _identifier(value["strategy_scope"], label="strategy_scope")
    template = _path(
        value["active_run_pointer_template"],
        label="active_run_pointer_template",
    )
    expected = {
        "v15": (
            "results/research_runtime_control/active_runs/"
            "v15/{strategy_id}.json"
        ),
        "myquant.v17.v4": (
            "results/v17_v4_formal_research/strategies/"
            "{strategy_id}/_active.json"
        ),
    }[protocol_id]
    if template != expected:
        raise CanonicalControlError("active-run pointer template mismatch")
    _authority(value["authority_ceiling"])


def _validate_run(value: Mapping[str, Any]) -> None:
    _exact(
        value,
        {
            "version",
            "run_id",
            "protocol_id",
            "strategy_id",
            "cutoff",
            "status",
            "evidence_refs",
            "authority_ceiling",
            "semantic_sha256",
        },
        label="immutable run",
    )
    _identifier(value["run_id"], label="run_id")
    if value["protocol_id"] not in {"v15", "myquant.v17.v4"}:
        raise CanonicalControlError("run protocol_id is closed")
    _identifier(value["strategy_id"], label="strategy_id")
    _instant(value["cutoff"], label="cutoff")
    if value["status"] != "HEALTHY":
        raise CanonicalControlError("runtime-control run must be healthy")
    _references(value["evidence_refs"], label="evidence_refs")
    _authority(value["authority_ceiling"])


def _validate_active_pointer(value: Mapping[str, Any]) -> None:
    _exact(
        value,
        {
            "version",
            "protocol_id",
            "strategy_id",
            "run_ref",
            "cutoff",
            "updated_at",
            "semantic_sha256",
        },
        label="active-run pointer",
    )
    if value["protocol_id"] not in {"v15", "myquant.v17.v4"}:
        raise CanonicalControlError("active-run protocol_id is closed")
    _identifier(value["strategy_id"], label="strategy_id")
    _reference(value["run_ref"], label="run_ref")
    _instant(value["cutoff"], label="cutoff")
    _instant(value["updated_at"], label="updated_at")


def _validate_v4_authority(value: Any) -> None:
    keys = {
        "broker",
        "execution",
        "formal_research_publication",
        "order",
        "research_runtime_default",
        "trade",
    }
    if type(value) is not dict:
        raise CanonicalControlError("v4 authority must be an object")
    _exact(value, keys, label="v4 authority")
    for key in {"broker", "execution", "order", "trade"}:
        if value[key] is not False:
            raise CanonicalControlError(
                "v4 execution authority must remain false"
            )
    for key in {"formal_research_publication", "research_runtime_default"}:
        if type(value[key]) is not bool:
            raise CanonicalControlError("v4 research authority must be boolean")


def _validate_v4_formal_pointer(value: Mapping[str, Any]) -> None:
    _exact(
        value,
        {
            "authority",
            "cutoff",
            "intent_ref",
            "pointer_id",
            "protocol_version",
            "semantic_sha256",
            "state",
            "strategy_id",
            "updated_at",
            "version",
        },
        label="v4 formal active pointer",
    )
    _validate_v4_authority(value["authority"])
    if (
        value["authority"]["formal_research_publication"] is not False
        or value["authority"]["research_runtime_default"] is not False
    ):
        raise CanonicalControlError(
            "v4 formal pointer authority exceeds its publication role"
        )
    _instant(value["cutoff"], label="cutoff")
    _reference(value["intent_ref"], label="intent_ref")
    _identifier(value["pointer_id"], label="pointer_id")
    if value["protocol_version"] != "myquant.v17.v4":
        raise CanonicalControlError("v4 protocol version mismatch")
    if value["state"] != "PENDING_COMPLETION":
        raise CanonicalControlError("v4 formal pointer state mismatch")
    _identifier(value["strategy_id"], label="strategy_id")
    _instant(value["updated_at"], label="updated_at")
    intent_ref = value["intent_ref"]
    if (
        intent_ref["strategy_id"] != value["strategy_id"]
        or intent_ref["cutoff"] != value["cutoff"]
        or intent_ref["artifact_version"]
        != "myquant.v17.v4.formal-activation-intent.v1"
    ):
        raise CanonicalControlError(
            "intent_ref scope, cutoff, or version mismatch"
        )


def _validate_selector(value: Mapping[str, Any]) -> None:
    _exact(
        value,
        {
            "version",
            "selector_id",
            "status",
            "protocol_target_ref",
            "transition_intent_ref",
            "updated_at",
            "semantic_sha256",
        },
        label="default protocol selector",
    )
    _identifier(value["selector_id"], label="selector_id")
    if value["status"] not in {
        "V15_DEFAULT",
        "RESEARCH_DEFAULT_ACTIVE",
        "ROLLED_BACK_TO_V15",
    }:
        raise CanonicalControlError("selector status is closed")
    _reference(value["protocol_target_ref"], label="protocol_target_ref")
    _intent_identity(value["transition_intent_ref"])
    _instant(value["updated_at"], label="updated_at")


_INTENT_KEYS = {
    "version",
    "intent_id",
    "transition",
    "created_at",
    "expected_selector_sha256",
    "expected_protocol_target_ref",
    "proposed_protocol_target_ref",
    "expected_target_active_pointer_sha256",
    "expected_target_run_ref",
    "proposed_selector_bytes_sha256",
    "required_evidence_refs",
    "semantic_sha256",
}


def _validate_intent(value: Mapping[str, Any]) -> None:
    _exact(value, _INTENT_KEYS, label="route transition intent")
    _identifier(value["intent_id"], label="intent_id")
    if value["transition"] not in {"BOOTSTRAP", "CUTOVER", "ROLLBACK"}:
        raise CanonicalControlError("intent transition is closed")
    _instant(value["created_at"], label="created_at")
    _digest(
        value["expected_selector_sha256"],
        label="expected_selector_sha256",
        allow_empty=True,
    )
    if value["expected_protocol_target_ref"] is not None:
        _reference(
            value["expected_protocol_target_ref"],
            label="expected_protocol_target_ref",
        )
    _reference(
        value["proposed_protocol_target_ref"],
        label="proposed_protocol_target_ref",
    )
    _digest(
        value["expected_target_active_pointer_sha256"],
        label="expected_target_active_pointer_sha256",
    )
    _reference(value["expected_target_run_ref"], label="expected_target_run_ref")
    _digest(
        value["proposed_selector_bytes_sha256"],
        label="proposed_selector_bytes_sha256",
    )
    _references(value["required_evidence_refs"], label="required_evidence_refs")


_RECEIPT_KEYS = {
    "version",
    "receipt_id",
    "intent_id",
    "transition",
    "recorded_at",
    "expected_selector_sha256",
    "expected_protocol_target_ref",
    "proposed_protocol_target_ref",
    "expected_target_active_pointer_sha256",
    "expected_target_run_ref",
    "proposed_selector_bytes_sha256",
    "required_evidence_refs",
    "observed_prevalue_sha256",
    "observed_target_active_pointer_sha256",
    "post_readback_sha256",
    "outcome",
    "semantic_sha256",
}


def _validate_receipt(value: Mapping[str, Any]) -> None:
    _exact(value, _RECEIPT_KEYS, label="route transition receipt")
    _identifier(value["receipt_id"], label="receipt_id")
    _identifier(value["intent_id"], label="intent_id")
    transition = value["transition"]
    outcomes = {
        "BOOTSTRAP": {
            "BOOTSTRAP_SUCCEEDED",
            "BOOTSTRAP_RECOVERED",
            "BOOTSTRAP_ABORTED",
            "BOOTSTRAP_CAS_BLOCKED",
        },
        "CUTOVER": {
            "CUTOVER_SUCCEEDED",
            "CUTOVER_RECOVERED",
            "CUTOVER_ABORTED",
            "CUTOVER_CAS_BLOCKED",
        },
        "ROLLBACK": {
            "ROLLBACK_SUCCEEDED",
            "ROLLBACK_RECOVERED",
            "ROLLBACK_ABORTED",
            "ROLLBACK_CAS_BLOCKED",
        },
    }
    if transition not in outcomes or value["outcome"] not in outcomes[transition]:
        raise CanonicalControlError("receipt transition/outcome mismatch")
    _instant(value["recorded_at"], label="recorded_at")
    _digest(
        value["expected_selector_sha256"],
        label="expected_selector_sha256",
        allow_empty=True,
    )
    if value["expected_protocol_target_ref"] is not None:
        _reference(
            value["expected_protocol_target_ref"],
            label="expected_protocol_target_ref",
        )
    _reference(
        value["proposed_protocol_target_ref"],
        label="proposed_protocol_target_ref",
    )
    _digest(
        value["expected_target_active_pointer_sha256"],
        label="expected_target_active_pointer_sha256",
    )
    _reference(value["expected_target_run_ref"], label="expected_target_run_ref")
    _digest(
        value["proposed_selector_bytes_sha256"],
        label="proposed_selector_bytes_sha256",
    )
    _references(value["required_evidence_refs"], label="required_evidence_refs")
    _digest(
        value["observed_prevalue_sha256"],
        label="observed_prevalue_sha256",
        allow_empty=True,
    )
    _digest(
        value["observed_target_active_pointer_sha256"],
        label="observed_target_active_pointer_sha256",
    )
    _digest(
        value["post_readback_sha256"],
        label="post_readback_sha256",
        allow_empty=True,
    )


_VALIDATORS = {
    TARGET_VERSION: _validate_target,
    RUN_VERSION: _validate_run,
    ACTIVE_POINTER_VERSION: _validate_active_pointer,
    SELECTOR_VERSION: _validate_selector,
    INTENT_VERSION: _validate_intent,
    BOOTSTRAP_RECEIPT_VERSION: _validate_receipt,
    CUTOVER_RECEIPT_VERSION: _validate_receipt,
    ROLLBACK_RECEIPT_VERSION: _validate_receipt,
    V4_FORMAL_ACTIVE_POINTER_VERSION: _validate_v4_formal_pointer,
}


def validate(value: Mapping[str, Any]) -> None:
    if type(value) is not dict:
        raise CanonicalControlError("artifact root must be an object")
    version = value.get("version")
    validator = _VALIDATORS.get(version)
    if validator is None:
        raise CanonicalControlError("artifact version is unregistered")
    validator(value)
    semantic = value.get("semantic_sha256")
    _digest(semantic, label="semantic_sha256")
    expected = (
        _v4_semantic_sha256(value)
        if str(version).startswith("myquant.v17.v4.")
        else sha256(canonical_bytes(_semantic_payload(value)))
    )
    if semantic != expected:
        raise CanonicalControlError("artifact semantic SHA-256 mismatch")


def artifact_reference(
    *,
    relative_path: str,
    document: Mapping[str, Any],
    raw: bytes,
    strategy_id: str,
    cutoff: str,
) -> dict[str, str]:
    decoded = decode_reference(raw)
    if decoded != dict(document):
        raise CanonicalControlError("artifact reference bytes drift")
    document_strategy = document.get(
        "strategy_id",
        document.get("strategy_scope"),
    )
    if (
        document_strategy is not None
        and document_strategy != strategy_id
    ):
        raise CanonicalControlError("artifact reference strategy mismatch")
    document_cutoff = document.get("cutoff")
    if document_cutoff is not None and document_cutoff != cutoff:
        raise CanonicalControlError("artifact reference cutoff mismatch")
    artifact_id = next(
        (
            document[name]
            for name in (
                "target_id",
                "run_id",
                "selector_id",
                "intent_id",
                "receipt_id",
                "pointer_id",
                "output_id",
                "comparison_id",
                "policy_id",
                "evidence_id",
            )
            if name in document
        ),
        None,
    )
    if artifact_id is None:
        artifact_id = (
            f"{document['protocol_id']}-{document['strategy_id']}"
        )
    return {
        "artifact_id": str(artifact_id),
        "artifact_version": str(document["version"]),
        "byte_sha256": sha256(raw),
        "cutoff": _instant(cutoff, label="artifact cutoff"),
        "relative_path": _path(relative_path, label="artifact path"),
        "semantic_sha256": str(document["semantic_sha256"]),
        "strategy_id": _identifier(
            strategy_id,
            label="artifact strategy_id",
        ),
    }


def intent_identity(*, intent_id: str, relative_path: str) -> dict[str, str]:
    return {
        "intent_id": _identifier(intent_id, label="intent_id"),
        "relative_path": _path(relative_path, label="intent path"),
        "version": INTENT_VERSION,
    }


def authority_ceiling() -> dict[str, bool]:
    return {key: False for key in sorted(_AUTHORITY_KEYS)}


__all__ = [
    "ACTIVE_POINTER_VERSION",
    "BOOTSTRAP_RECEIPT_VERSION",
    "CUTOVER_RECEIPT_VERSION",
    "CanonicalControlError",
    "INTENT_VERSION",
    "ROLLBACK_RECEIPT_VERSION",
    "RUN_VERSION",
    "SELECTOR_VERSION",
    "TARGET_VERSION",
    "V4_FORMAL_ACTIVE_POINTER_VERSION",
    "V4_REGISTERED_REFERENCE_VERSIONS",
    "artifact_reference",
    "authority_ceiling",
    "canonical_bytes",
    "decode",
    "decode_reference",
    "encode",
    "intent_identity",
    "seal",
    "sha256",
    "validate",
]
