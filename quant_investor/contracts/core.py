"""Stable, versionless artifact envelopes and their compiled contract registry.

The registry is deliberately process-local and code-defined.  Artifact bytes do
not select a schema from disk, and an envelope is accepted only when its exact
``(kind, contract_sha256)`` pair was registered by imported source code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import re
import threading
import unicodedata
from typing import Any, Final

MAX_CANONICAL_JSON_BYTES: Final = 8 * 1024 * 1024
MAX_CANONICAL_DEPTH: Final = 64
ARTIFACT_ENVELOPE_FIELDS: Final = frozenset(
    {
        "kind",
        "contract_sha256",
        "artifact_id",
        "created_at",
        "payload",
        "semantic_sha256",
    }
)
ARTIFACT_SEMANTIC_DOMAIN: Final = "myquant-artifact"
LEGACY_CONTRACT_FIELDS: Final = frozenset(
    {
        "contract_version",
        "protocol_version",
        "schema_id",
        "schema_version",
        "version",
    }
)

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_KIND_RE: Final = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_FIELD_RE: Final = re.compile(r"^[a-z][a-z0-9_]{0,127}$")


class ContractError(ValueError):
    """Base class for stable artifact-contract failures."""


class CanonicalJSONError(ContractError):
    """Raised when a value or byte string is not strict canonical JSON."""


class ContractRegistrationError(ContractError):
    """Raised when compiled contract registration is inconsistent."""


class UnknownContractError(ContractError):
    """Raised for an artifact pair absent from the compiled allowlist."""


class ArtifactValidationError(ContractError):
    """Raised when an artifact does not close its declared contract."""


PayloadValidator = Callable[[Mapping[str, Any]], None]


def _validate_string(value: str, *, label: str) -> None:
    try:
        raw = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise CanonicalJSONError(f"{label} is not valid UTF-8") from exc
    if len(raw) > MAX_CANONICAL_JSON_BYTES:
        raise CanonicalJSONError(f"{label} exceeds the canonical byte bound")
    if unicodedata.normalize("NFC", value) != value:
        raise CanonicalJSONError(f"{label} must be Unicode NFC")
    if any(ord(character) < 0x20 for character in value):
        raise CanonicalJSONError(f"{label} contains a control character")


def _validate_canonical_value(  # noqa: C901
    value: Any,
    *,
    label: str = "$",
    depth: int = 0,
    ancestors: frozenset[int] = frozenset(),
) -> None:
    if depth > MAX_CANONICAL_DEPTH:
        raise CanonicalJSONError(f"{label} exceeds maximum canonical depth")
    if value is None or type(value) in {bool, int}:
        return
    if type(value) is str:
        _validate_string(value, label=label)
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise CanonicalJSONError(f"{label} contains a non-finite float")
        return
    if type(value) not in {list, dict}:
        raise CanonicalJSONError(f"{label} contains an unsupported value")

    identity = id(value)
    if identity in ancestors:
        raise CanonicalJSONError(f"{label} contains a reference cycle")
    descendants = ancestors | {identity}
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_canonical_value(
                item,
                label=f"{label}[{index}]",
                depth=depth + 1,
                ancestors=descendants,
            )
        return

    for key, item in value.items():
        if type(key) is not str or not key or not key.isascii():
            raise CanonicalJSONError(f"{label} contains a noncanonical object key")
        if any(ord(character) < 0x20 for character in key):
            raise CanonicalJSONError(f"{label} contains a noncanonical object key")
        _validate_canonical_value(
            item,
            label=f"{label}.{key}",
            depth=depth + 1,
            ancestors=descendants,
        )


def canonical_json_bytes(value: Any) -> bytes:
    """Return the sole accepted UTF-8 JSON representation for ``value``.

    Non-finite floats, non-string or noncanonical keys, non-NFC strings,
    cycles, and non-JSON Python values are rejected before serialization.
    """

    _validate_canonical_value(value)
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8", errors="strict")
    if len(raw) > MAX_CANONICAL_JSON_BYTES:
        raise CanonicalJSONError("canonical JSON exceeds the byte bound")
    return raw


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalJSONError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(token: str) -> Any:
    raise CanonicalJSONError(f"non-finite JSON constant is forbidden: {token}")


def parse_canonical_json_bytes(raw: bytes, *, label: str = "artifact") -> Any:
    """Parse exact canonical bytes while rejecting duplicates and non-finite values."""

    if type(raw) is not bytes or not raw or len(raw) > MAX_CANONICAL_JSON_BYTES:
        raise CanonicalJSONError(f"{label} has an invalid canonical byte length")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except CanonicalJSONError:
        raise
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise CanonicalJSONError(f"{label} is invalid JSON") from exc
    expected = canonical_json_bytes(value)
    if raw != expected:
        raise CanonicalJSONError(f"{label} is not exact canonical JSON")
    return value


def _require_kind(value: Any) -> str:
    if type(value) is not str or _KIND_RE.fullmatch(value) is None:
        raise ContractRegistrationError("contract kind is not canonical")
    return value


def _require_field(value: Any, *, label: str) -> str:
    if type(value) is not str or _FIELD_RE.fullmatch(value) is None:
        raise ContractRegistrationError(f"{label} is not a canonical field name")
    return value


def _require_sha256(value: Any, *, label: str, error_type: type[ContractError]) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise error_type(f"{label} must be lowercase SHA-256")
    return value


def _require_timestamp(value: Any, *, label: str = "created_at") -> str:
    if type(value) is not str:
        raise ArtifactValidationError(f"{label} must be canonical UTC seconds")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ArtifactValidationError(f"{label} must be canonical UTC seconds") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ArtifactValidationError(f"{label} must be canonical UTC seconds")
    return value


def _normalize_fields(values: Any, *, label: str) -> frozenset[str]:
    if values is None:
        return frozenset()
    if isinstance(values, (str, bytes)):
        raise ContractRegistrationError(f"{label} must be a field collection")
    try:
        normalized = frozenset(_require_field(value, label=label) for value in values)
    except TypeError as exc:
        raise ContractRegistrationError(f"{label} must be a field collection") from exc
    return normalized


@dataclass(frozen=True, slots=True)
class ContractDefinition:
    """One compiled artifact contract.

    ``contract_sha256`` may pin a separately compiled source contract.  When it
    is omitted, it is deterministically derived from this declarative shape.
    A custom validator is code-bound to the registered pair and never selected
    by artifact bytes.
    """

    kind: str
    identity_field: str
    contract_sha256: str = ""
    json_schema_sha256: str = ""
    validator_code_sha256: str = ""
    required_payload_fields: frozenset[str] = field(default_factory=frozenset)
    optional_payload_fields: frozenset[str] = field(default_factory=frozenset)
    forbidden_payload_fields: frozenset[str] = field(default_factory=frozenset)
    allow_additional_payload_fields: bool = False
    validator: PayloadValidator | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        kind = _require_kind(self.kind)
        identity_field = _require_field(self.identity_field, label="identity_field")
        required = _normalize_fields(
            self.required_payload_fields, label="required_payload_fields"
        ) | {identity_field}
        optional = _normalize_fields(self.optional_payload_fields, label="optional_payload_fields")
        forbidden = _normalize_fields(
            self.forbidden_payload_fields, label="forbidden_payload_fields"
        )
        if required & optional or required & forbidden or optional & forbidden:
            raise ContractRegistrationError("contract payload field sets overlap")
        if type(self.allow_additional_payload_fields) is not bool:
            raise ContractRegistrationError("allow_additional_payload_fields must be boolean")
        if self.validator is not None and not callable(self.validator):
            raise ContractRegistrationError("contract validator must be callable")

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "identity_field", identity_field)
        object.__setattr__(self, "required_payload_fields", frozenset(required))
        object.__setattr__(self, "optional_payload_fields", optional)
        object.__setattr__(self, "forbidden_payload_fields", forbidden)

        schema_descriptor = {
            "allow_additional_payload_fields": self.allow_additional_payload_fields,
            "forbidden_payload_fields": sorted(forbidden),
            "optional_payload_fields": sorted(optional),
            "required_payload_fields": sorted(required),
        }
        json_schema_sha = (
            _require_sha256(
                self.json_schema_sha256,
                label="json_schema_sha256",
                error_type=ContractRegistrationError,
            )
            if self.json_schema_sha256
            else hashlib.sha256(canonical_json_bytes(schema_descriptor)).hexdigest()
        )
        if self.validator_code_sha256:
            validator_code_sha = _require_sha256(
                self.validator_code_sha256,
                label="validator_code_sha256",
                error_type=ContractRegistrationError,
            )
        else:
            target_validator = self.validator or type(self).validate_payload
            try:
                validator_source = inspect.getsource(target_validator).encode(
                    "utf-8", errors="strict"
                )
            except (OSError, TypeError, UnicodeError) as exc:
                raise ContractRegistrationError(
                    "validator_code_sha256 is required when validator source is unavailable"
                ) from exc
            validator_code_sha = hashlib.sha256(validator_source).hexdigest()
        catalog_entry = {
            "identity_field": identity_field,
            "json_schema_sha256": json_schema_sha,
            "kind": kind,
            "validator_code_sha256": validator_code_sha,
        }
        digest = hashlib.sha256(canonical_json_bytes(catalog_entry)).hexdigest()
        if self.contract_sha256:
            supplied = _require_sha256(
                self.contract_sha256,
                label="contract_sha256",
                error_type=ContractRegistrationError,
            )
            if supplied != digest:
                raise ContractRegistrationError(
                    "contract_sha256 does not bind the compiled catalog entry"
                )
        object.__setattr__(self, "contract_sha256", digest)
        object.__setattr__(self, "json_schema_sha256", json_schema_sha)
        object.__setattr__(self, "validator_code_sha256", validator_code_sha)

    def catalog_entry(self) -> dict[str, str]:
        """Return the exact code-compiled dispatch entry bound by the contract SHA."""

        return {
            "kind": self.kind,
            "contract_sha256": self.contract_sha256,
            "identity_field": self.identity_field,
            "json_schema_sha256": self.json_schema_sha256,
            "validator_code_sha256": self.validator_code_sha256,
        }

    def validate_payload(self, value: Any) -> dict[str, Any]:  # noqa: C901
        if type(value) is not dict:
            raise ArtifactValidationError("artifact payload must be an object")
        payload = dict(value)
        fields = set(payload)
        missing = self.required_payload_fields - fields
        forbidden = self.forbidden_payload_fields & fields
        if missing:
            raise ArtifactValidationError(
                f"artifact payload is missing required fields: {sorted(missing)}"
            )
        if forbidden:
            raise ArtifactValidationError(
                f"artifact payload contains forbidden fields: {sorted(forbidden)}"
            )
        if not self.allow_additional_payload_fields:
            expected = self.required_payload_fields | self.optional_payload_fields
            if not fields <= expected:
                raise ArtifactValidationError("artifact payload fields are not exact")
        identity = payload.get(self.identity_field)
        if (
            type(identity) is not str
            or not identity
            or identity != identity.strip()
            or len(identity.encode("utf-8", errors="strict")) > 512
            or any(ord(character) < 0x20 for character in identity)
        ):
            raise ArtifactValidationError("artifact payload identity is not canonical")
        canonical_json_bytes(payload)
        if self.validator is not None:
            try:
                self.validator(payload)
            except ContractError:
                raise
            except (TypeError, ValueError) as exc:
                raise ArtifactValidationError("artifact payload validator rejected value") from exc
        return payload


_REGISTRY_LOCK = threading.RLock()
_CONTRACTS: dict[tuple[str, str], ContractDefinition] = {}
_CONTRACTS_BY_KIND: dict[str, dict[str, ContractDefinition]] = {}
_REGISTRY_FROZEN = False


def _freeze_contract_registry() -> None:
    """Freeze pair dispatch after the package's static builtins have loaded."""

    global _REGISTRY_FROZEN
    with _REGISTRY_LOCK:
        _REGISTRY_FROZEN = True


def register_contract(
    definition: ContractDefinition | None = None,
    /,
    **definition_kwargs: Any,
) -> ContractDefinition:
    """Register one code-defined contract pair, idempotently.

    Callers may pass a ``ContractDefinition`` or its constructor keywords.
    Conflicting reuse of an already-registered pair fails closed.
    """

    if definition is None:
        definition = ContractDefinition(**definition_kwargs)
    elif definition_kwargs or not isinstance(definition, ContractDefinition):
        raise ContractRegistrationError("register_contract requires one contract definition")
    key = (definition.kind, definition.contract_sha256)
    with _REGISTRY_LOCK:
        existing = _CONTRACTS.get(key)
        if existing is not None:
            if existing != definition:
                raise ContractRegistrationError("compiled contract pair is already different")
            return existing
        if _REGISTRY_FROZEN:
            raise ContractRegistrationError("compiled contract registry is frozen")
        _CONTRACTS[key] = definition
        _CONTRACTS_BY_KIND.setdefault(definition.kind, {})[definition.contract_sha256] = definition
    return definition


def registered_contracts() -> tuple[ContractDefinition, ...]:
    """Return a deterministic snapshot of the compiled allowlist."""

    with _REGISTRY_LOCK:
        return tuple(_CONTRACTS[key] for key in sorted(_CONTRACTS))


def registered_contract_catalog() -> tuple[dict[str, str], ...]:
    """Return the deterministic compiled pair-dispatch catalog."""

    return tuple(definition.catalog_entry() for definition in registered_contracts())


def contract_catalog_sha256() -> str:
    """Hash the exact compiled catalog object used for pair dispatch."""

    document = {"contracts": list(registered_contract_catalog())}
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def get_contract(kind: str, contract_sha256: str | None = None) -> ContractDefinition:
    """Resolve one compiled definition without consulting artifact-controlled I/O."""

    try:
        normalized_kind = _require_kind(kind)
    except ContractRegistrationError as exc:
        raise UnknownContractError("artifact kind is not registered") from exc
    with _REGISTRY_LOCK:
        by_sha = _CONTRACTS_BY_KIND.get(normalized_kind, {})
        if contract_sha256 is None:
            if len(by_sha) != 1:
                raise UnknownContractError("artifact kind has no single compiled contract")
            return next(iter(by_sha.values()))
        try:
            normalized_sha = _require_sha256(
                contract_sha256,
                label="contract_sha256",
                error_type=UnknownContractError,
            )
        except UnknownContractError:
            raise
        definition = by_sha.get(normalized_sha)
        if definition is None:
            raise UnknownContractError("artifact contract pair is not allowlisted")
        return definition


def artifact_semantic_preimage(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact acyclic semantic preimage for a sealed envelope."""

    if type(artifact) is not dict:
        raise ArtifactValidationError("artifact envelope must be an object")
    fields = set(artifact)
    expected = set(ARTIFACT_ENVELOPE_FIELDS)
    if fields != expected:
        raise ArtifactValidationError("artifact envelope fields are not exact")
    definition = get_contract(artifact.get("kind"), artifact.get("contract_sha256"))
    return {
        "domain": ARTIFACT_SEMANTIC_DOMAIN,
        "kind": artifact["kind"],
        "contract_sha256": artifact["contract_sha256"],
        "identity_field": definition.identity_field,
        "artifact_id": artifact["artifact_id"],
        "created_at": artifact["created_at"],
        "payload": artifact["payload"],
    }


def _semantic_sha256(preimage: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(preimage))).hexdigest()


def seal_artifact(
    kind: str,
    payload: Mapping[str, Any],
    *,
    created_at: str,
    contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Seal one exact artifact envelope using a compiled contract pair."""

    definition = get_contract(kind, contract_sha256)
    if type(payload) is not dict:
        raise ArtifactValidationError("artifact payload must be an object")
    normalized_payload = definition.validate_payload(dict(payload))
    identity = normalized_payload[definition.identity_field]
    envelope: dict[str, Any] = {
        "kind": definition.kind,
        "contract_sha256": definition.contract_sha256,
        "artifact_id": identity,
        "created_at": _require_timestamp(created_at),
        "payload": normalized_payload,
    }
    semantic_preimage = {
        "domain": ARTIFACT_SEMANTIC_DOMAIN,
        "kind": definition.kind,
        "contract_sha256": definition.contract_sha256,
        "identity_field": definition.identity_field,
        "artifact_id": identity,
        "created_at": envelope["created_at"],
        "payload": normalized_payload,
    }
    envelope["semantic_sha256"] = _semantic_sha256(semantic_preimage)
    canonical_json_bytes(envelope)
    return envelope


def validate_artifact(
    artifact: Mapping[str, Any] | bytes,
    *,
    expected_kind: str | None = None,
    expected_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate exact bytes/fields, allowlist membership, identity, and hashes."""

    if type(artifact) is bytes:
        value = parse_canonical_json_bytes(artifact)
    elif type(artifact) is dict:
        canonical_json_bytes(artifact)
        value = dict(artifact)
    else:
        raise ArtifactValidationError("artifact must be an object or canonical bytes")
    if type(value) is not dict or set(value) != set(ARTIFACT_ENVELOPE_FIELDS):
        raise ArtifactValidationError("artifact envelope fields are not exact")

    kind = value.get("kind")
    contract_sha = value.get("contract_sha256")
    if expected_kind is not None and kind != expected_kind:
        raise ArtifactValidationError("artifact kind does not match expectation")
    if expected_contract_sha256 is not None and contract_sha != expected_contract_sha256:
        raise ArtifactValidationError("artifact contract does not match expectation")
    definition = get_contract(kind, contract_sha)
    payload = definition.validate_payload(value.get("payload"))
    identity = payload[definition.identity_field]
    if value.get("artifact_id") != identity:
        raise ArtifactValidationError("artifact_id does not match payload identity")
    _require_timestamp(value.get("created_at"))
    semantic = _require_sha256(
        value.get("semantic_sha256"),
        label="semantic_sha256",
        error_type=ArtifactValidationError,
    )
    expected_semantic = _semantic_sha256(artifact_semantic_preimage(value))
    if semantic != expected_semantic:
        raise ArtifactValidationError("artifact semantic_sha256 mismatch")
    canonical_json_bytes(value)
    return value


def artifact_byte_sha256(artifact: Mapping[str, Any] | bytes) -> str:
    """Return the exact canonical artifact-byte SHA-256 after full validation."""

    if type(artifact) is bytes:
        validate_artifact(artifact)
        raw = artifact
    else:
        normalized = validate_artifact(artifact)
        raw = canonical_json_bytes(normalized)
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "ARTIFACT_ENVELOPE_FIELDS",
    "ARTIFACT_SEMANTIC_DOMAIN",
    "ArtifactValidationError",
    "CanonicalJSONError",
    "ContractDefinition",
    "ContractError",
    "ContractRegistrationError",
    "LEGACY_CONTRACT_FIELDS",
    "MAX_CANONICAL_JSON_BYTES",
    "UnknownContractError",
    "artifact_byte_sha256",
    "artifact_semantic_preimage",
    "canonical_json_bytes",
    "contract_catalog_sha256",
    "get_contract",
    "parse_canonical_json_bytes",
    "register_contract",
    "registered_contract_catalog",
    "registered_contracts",
    "seal_artifact",
    "validate_artifact",
]
