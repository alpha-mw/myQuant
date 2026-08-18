"""Content-addressed artifacts, immutable generations, and active-pointer CAS."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from io import FileIO
import os
from pathlib import PurePosixPath
import re
import threading
import time
from types import MappingProxyType
from typing import Any, Final

from quant_investor.contracts import (
    READINESS_FIELDS,
    SYSTEM_GENERATION_MANIFEST_CONTRACT,
    ContractError,
    artifact_byte_sha256,
    canonical_json_bytes,
    contract_catalog_sha256,
    get_contract,
    parse_canonical_json_bytes,
    registered_contract_catalog,
    seal_artifact,
    validate_artifact,
)

from .components import DECODE_MEMORY_BUDGET_BYTES
from .candidate_records import (
    build_candidate_transaction_intent,
    candidate_transaction_intent_id,
    validate_candidate_transaction_intent,
)
from .errors import (
    SYSTEM_ACTIVE_POINTER_ABSENT,
    SystemCASMismatch,
    SystemActivationAuthorizationError,
    SystemContractError,
    SystemError,
    SystemImmutableConflict,
    SystemMigrationClosureError,
    SystemMigrationMarkerAbsent,
    SystemNotFound,
    SystemPreconditionError,
    SystemSecurityError,
    SystemStorageError,
)
from .release import installed_code_manifest_sha256
from .historical_activation import (
    INITIAL_ASSEMBLER_MODULE_PATH,
    INITIAL_HISTORICAL_SOURCE_MAX_BYTES,
    INITIAL_SOURCE_FORMAT_MEDIA_TYPES,
    validate_frozen_object_ref,
    validate_initial_assembler_module_path,
)
from .storage import (
    ACTIVE_POINTER_PATH,
    ACTIVATION_AUTHORIZATIONS_ROOT,
    ACTIVATION_TRANSACTIONS_ROOT,
    CANDIDATE_STATE_ROOT,
    EMPTY_POINTER_SHA256,
    FINAL_CUTOVER_AUTHORIZATIONS_ROOT,
    GENERATIONS_ROOT,
    MIGRATION_MARKER_PATH,
    OBJECTS_ROOT,
    POINTER_HISTORY_ROOT,
    VALIDATION_RUNS_ROOT,
    SecureSystemStorage,
    _PreparedInitialActivationWrite,
    source_stat_identity,
)

OBJECT_REF_FIELDS: Final = frozenset(
    {
        "kind",
        "contract_sha256",
        "artifact_id",
        "semantic_sha256",
        "byte_sha256",
    }
)
OBJECT_REF_SORT_FIELDS: Final = (
    "kind",
    "contract_sha256",
    "artifact_id",
    "semantic_sha256",
    "byte_sha256",
)
READINESS_KINDS: Final = frozenset({"system.readiness", "intelligence_readiness"})
OPERATIONAL_READINESS_KIND: Final = "intelligence_readiness"
SUSPENDED_READINESS_KIND: Final = "system.readiness"
RELEASE_KIND: Final = "system.release"
MANIFEST_KIND: Final = SYSTEM_GENERATION_MANIFEST_CONTRACT.kind
FACTOR_ACTIVE_SET_KINDS: Final = frozenset({"factor.bootstrap_set", "factor.admitted_set"})
FACTOR_STATUS_KIND: Final = "factor.status"
FACTOR_VALIDATION_RECEIPT_KIND: Final = "factor.validation_receipt"
FACTOR_COMPOSITE_STATE_KIND: Final = "factor.composite_state"
CANDIDATE_POINTER_FIELDS: Final = frozenset(
    {
        "candidate_state_ref",
        "previous_pointer_sha256",
        "stored_at",
        "os_actor",
        "authority",
    }
)
POINTER_FIELDS: Final = frozenset(
    {
        "generation_id",
        "manifest_sha256",
        "previous_pointer_sha256",
        "activated_at",
        "os_actor",
    }
)
GENERATION_STATES: Final = frozenset({"OPERATIONAL", "SYSTEM_SUSPENDED"})
OPERATIONAL_DATA_ROLES: Final = (
    "exchange_calendar",
    "fundamental_generation",
    "market_snapshot",
    "pit_membership",
)
SOURCE_FORMAT_MEDIA_TYPES: Final = MappingProxyType(
    {
        "BINARY": ("application/octet-stream", ()),
        "CSV": ("text/csv", (".csv",)),
        "JSON": ("application/json", (".json",)),
        "JSONL": ("application/x-ndjson", (".jsonl",)),
        "PARQUET": ("application/vnd.apache.parquet", (".parquet",)),
        "PYTHON": ("text/x-python", (".py",)),
        "TEXT": ("text/plain", (".txt",)),
    }
)
MAX_SOURCE_BYTES: Final = 1024 * 1024 * 1024 * 1024
_MAXIMUM_RESOURCE_LEASE_WAIT_SECONDS: Final = 180.0

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")


class _WeightedMemoryBudget:
    """One fail-closed weighted lease shared by every runner in this process."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._available = capacity
        self._condition = threading.Condition()

    @contextmanager
    def reserve(self, amount: int) -> Iterator[None]:
        deadline = time.monotonic() + _MAXIMUM_RESOURCE_LEASE_WAIT_SECONDS
        with self._condition:
            while self._available < amount:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SystemSecurityError("validation decode memory lease timed out")
                self._condition.wait(timeout=remaining)
            self._available -= amount
        try:
            yield
        finally:
            with self._condition:
                self._available += amount
                if self._available > self._capacity:
                    self._available = self._capacity
                    raise SystemStorageError("validation decode memory lease accounting failed")
                self._condition.notify_all()


_PROCESS_DECODE_MEMORY_BUDGET: Final = _WeightedMemoryBudget(DECODE_MEMORY_BUDGET_BYTES)


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SystemContractError(f"{label} must be lowercase SHA-256")
    return value


def _require_pointer_sha256(value: Any, *, label: str) -> str:
    if value == EMPTY_POINTER_SHA256:
        return EMPTY_POINTER_SHA256
    return _require_sha256(value, label=label)


def _require_text(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value.encode("utf-8", errors="strict")) > 512
        or any(ord(character) < 0x20 for character in value)
    ):
        raise SystemContractError(f"{label} must be canonical non-empty text")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _require_timestamp(value: Any, *, label: str) -> str:
    if type(value) is not str:
        raise SystemContractError(f"{label} must be canonical UTC seconds")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise SystemContractError(f"{label} must be canonical UTC seconds") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise SystemContractError(f"{label} must be canonical UTC seconds")
    return value


def _domain_identity(domain: str, **identity_inputs: Any) -> str:
    """Hash only explicitly named business-identity inputs."""

    return hashlib.sha256(
        canonical_json_bytes({"domain": domain, "identity_inputs": identity_inputs})
    ).hexdigest()


def generation_assembly_identity(
    *,
    generation_state: str,
    release_id: str,
    readiness_id: str,
    created_at: str,
) -> str:
    """Return the public deterministic identity used by generation manifests."""

    return _domain_identity(
        "system.generation_assembly",
        generation_state=_require_text(generation_state, label="generation_state"),
        release_id=_require_text(release_id, label="release_id"),
        readiness_id=_require_text(readiness_id, label="readiness_id"),
        created_at=_require_timestamp(created_at, label="created_at"),
    )


def _object_path(kind: str, byte_sha256: str) -> Any:
    return OBJECTS_ROOT / kind / f"{byte_sha256}.json"


def validation_namespace_path_sha256(validation_namespace_id: str) -> str:
    """Map an opaque validation namespace to a canonical governed path."""

    namespace = _require_text(validation_namespace_id, label="validation_namespace_id")
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": "myquant-validation-namespace-path",
                "validation_namespace_id": namespace,
            }
        )
    ).hexdigest()


def _candidate_paths(
    validation_namespace_id: str,
) -> tuple[PurePosixPath, PurePosixPath, PurePosixPath]:
    namespace_path = CANDIDATE_STATE_ROOT / validation_namespace_path_sha256(
        validation_namespace_id
    )
    return (
        namespace_path / "_current.json",
        namespace_path / "history",
        namespace_path / ".lock",
    )


def _candidate_transaction_path(
    validation_namespace_id: str,
    transaction_id: str,
) -> PurePosixPath:
    namespace_path = CANDIDATE_STATE_ROOT / validation_namespace_path_sha256(
        validation_namespace_id
    )
    transaction = _require_text(transaction_id, label="transaction_id")
    transaction_path = hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": "myquant-candidate-transaction-path",
                "transaction_id": transaction,
            }
        )
    ).hexdigest()
    return namespace_path / "transactions" / transaction_path


def validate_object_ref(value: Any, *, label: str = "object_ref") -> dict[str, str]:
    """Validate an exact content-addressed object reference and compiled pair."""

    if type(value) is not dict or set(value) != set(OBJECT_REF_FIELDS):
        raise SystemContractError(f"{label} fields are not exact")
    kind = value.get("kind")
    contract_sha256 = value.get("contract_sha256")
    try:
        definition = get_contract(kind, contract_sha256)
    except ContractError as exc:
        raise SystemContractError(f"{label} contract pair is not compiled") from exc
    artifact_id = _require_text(value.get("artifact_id"), label=f"{label}.artifact_id")
    semantic_sha256 = _require_sha256(
        value.get("semantic_sha256"), label=f"{label}.semantic_sha256"
    )
    byte_sha256 = _require_sha256(value.get("byte_sha256"), label=f"{label}.byte_sha256")
    return {
        "kind": definition.kind,
        "contract_sha256": definition.contract_sha256,
        "artifact_id": artifact_id,
        "semantic_sha256": semantic_sha256,
        "byte_sha256": byte_sha256,
    }


def object_ref_for_artifact(artifact: Mapping[str, Any]) -> dict[str, str]:
    """Build the exact ref for an already validated envelope."""

    try:
        document = validate_artifact(dict(artifact))
        byte_sha256 = artifact_byte_sha256(document)
    except ContractError as exc:
        raise SystemContractError("artifact contract closure failed") from exc
    return {
        "kind": document["kind"],
        "contract_sha256": document["contract_sha256"],
        "artifact_id": document["artifact_id"],
        "semantic_sha256": document["semantic_sha256"],
        "byte_sha256": byte_sha256,
    }


def _object_ref_sort_key(ref: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(ref[field] for field in OBJECT_REF_SORT_FIELDS)


def _validate_readiness(artifact: Mapping[str, Any]) -> dict[str, Any]:
    if artifact.get("kind") not in READINESS_KINDS:
        raise SystemContractError("generation readiness binding has the wrong kind")
    payload = artifact.get("payload")
    if type(payload) is not dict or set(payload) != set(READINESS_FIELDS):
        raise SystemContractError("readiness payload fields are not exact")
    for field in (
        "readiness_id",
        "factor_state",
        "admission_route",
        "producer_identity",
        "mainline_state",
        "investment_state",
    ):
        _require_text(payload.get(field), label=f"readiness.{field}")
    blockers = payload.get("blockers")
    if type(blockers) is not list:
        raise SystemContractError("readiness.blockers must be a list")
    normalized = [
        _require_text(blocker, label=f"readiness.blockers[{index}]")
        for index, blocker in enumerate(blockers)
    ]
    if len(normalized) != len(set(normalized)) or normalized != sorted(normalized):
        raise SystemContractError("readiness.blockers must be sorted and unique")
    factor_status_ref = payload.get("factor_status_ref")
    if factor_status_ref is not None:
        ref = validate_object_ref(factor_status_ref, label="readiness.factor_status_ref")
        if ref["kind"] != "factor.status":
            raise SystemContractError("readiness.factor_status_ref has the wrong kind")
    mainline_candidate_ref = payload.get("mainline_candidate_ref")
    if mainline_candidate_ref is not None:
        ref = validate_object_ref(mainline_candidate_ref, label="readiness.mainline_candidate_ref")
        if ref["kind"] != "mainline_candidate":
            raise SystemContractError("readiness.mainline_candidate_ref has the wrong kind")
    return dict(payload)


def _validate_release(artifact: Mapping[str, Any]) -> dict[str, Any]:
    if artifact.get("kind") != RELEASE_KIND:
        raise SystemContractError("generation release binding has the wrong kind")
    payload = artifact.get("payload")
    if type(payload) is not dict:
        raise SystemContractError("release payload must be an object")
    _require_text(payload.get("release_id"), label="release.release_id")
    _require_text(payload.get("state"), label="release.state")
    for field in ("code_sha256", "wheel_sha256", "code_manifest_sha256"):
        _require_sha256(payload.get(field), label=f"release.{field}")
    return dict(payload)


def _verify_installed_release(artifact: Mapping[str, Any]) -> str:
    payload = _validate_release(artifact)
    observed = installed_code_manifest_sha256()
    if payload["code_manifest_sha256"] != observed:
        raise SystemPreconditionError("installed code manifest does not match generation release")
    return observed


class SystemStore:
    """Stable store for unified runtime artifacts and active generation state."""

    def __init__(
        self,
        workspace_root: str | os.PathLike[str],
        *,
        source_root: str | os.PathLike[str] | None = None,
        source_root_id: str | None = None,
        max_source_bytes: int = MAX_SOURCE_BYTES,
    ) -> None:
        self._storage = SecureSystemStorage(workspace_root, max_read_bytes=64 * 1024 * 1024)
        self.workspace_root = self._storage.workspace_root
        self._source_storage = SecureSystemStorage(source_root or workspace_root)
        self.source_root = self._source_storage.workspace_root
        if type(max_source_bytes) is not int or max_source_bytes <= 0:
            raise SystemSecurityError("source byte bound is invalid")
        self.max_source_bytes = max_source_bytes
        derived_root_id = _domain_identity(
            "system.source_root", resolved_path=str(self.source_root)
        )
        self.source_root_id = _require_text(
            source_root_id or derived_root_id, label="source_root_id"
        )
        self._decode_memory_budget = _PROCESS_DECODE_MEMORY_BUDGET

    @staticmethod
    def contract_catalog() -> dict[str, Any]:
        """Return the exact compiled catalog object bound by every generation."""

        return {"contracts": list(registered_contract_catalog())}

    def put_contract_catalog(self) -> str:
        """Publish/read back the exact compiled catalog as a content object."""

        document = self.contract_catalog()
        raw = canonical_json_bytes(document)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != contract_catalog_sha256():
            raise SystemContractError("compiled contract catalog hash mismatch")
        stored = self._storage.write_exact_once(
            _object_path("system.contract_catalog", digest), raw
        )
        if stored.byte_sha256 != digest:
            raise SystemStorageError("compiled contract catalog readback mismatch")
        self.read_contract_catalog(digest)
        return digest

    def read_contract_catalog(self, catalog_sha256: str) -> dict[str, Any]:
        """Read and validate an exact content-addressed compiled catalog object."""

        digest = _require_sha256(catalog_sha256, label="contract_catalog_sha256")
        stored = self._storage.read(_object_path("system.contract_catalog", digest))
        if stored.byte_sha256 != digest:
            raise SystemContractError("contract catalog byte hash mismatch")
        try:
            document = parse_canonical_json_bytes(stored.data, label="contract catalog")
        except ContractError as exc:
            raise SystemContractError("contract catalog is not canonical") from exc
        if document != self.contract_catalog():
            raise SystemContractError("contract catalog does not match compiled allowlist")
        return document

    def _read_historical_contract_catalog(
        self, catalog_sha256: str
    ) -> tuple[dict[str, Any], dict[tuple[str, str], str]]:
        """Read an initial-generation catalog without comparing it to current code.

        The permanent migration marker is a historical anchor.  A descendant
        release may add contracts, so replay must authenticate the stored
        catalog bytes and use their non-executable dispatch metadata instead of
        requiring equality with the descendant process registry.
        """

        digest = _require_sha256(catalog_sha256, label="historical contract catalog")
        stored = self._storage.read(_object_path("system.contract_catalog", digest))
        if stored.byte_sha256 != digest:
            raise SystemContractError("historical contract catalog byte hash mismatch")
        try:
            document = parse_canonical_json_bytes(stored.data, label="historical contract catalog")
        except ContractError as exc:
            raise SystemContractError("historical contract catalog is not canonical") from exc
        if type(document) is not dict or set(document) != {"contracts"}:
            raise SystemContractError("historical contract catalog fields are not exact")
        rows = document["contracts"]
        if type(rows) is not list or not rows:
            raise SystemContractError("historical contract catalog is empty")
        dispatch: dict[tuple[str, str], str] = {}
        expected_fields = {
            "kind",
            "contract_sha256",
            "identity_field",
            "json_schema_sha256",
            "validator_code_sha256",
        }
        for row in rows:
            if type(row) is not dict or set(row) != expected_fields:
                raise SystemContractError("historical catalog row fields are not exact")
            kind = _require_text(row["kind"], label="historical catalog kind")
            contract_sha = _require_sha256(
                row["contract_sha256"], label="historical catalog contract"
            )
            identity_field = _require_text(
                row["identity_field"], label="historical catalog identity field"
            )
            _require_sha256(row["json_schema_sha256"], label="historical catalog schema")
            _require_sha256(row["validator_code_sha256"], label="historical catalog validator")
            key = (kind, contract_sha)
            if key in dispatch:
                raise SystemContractError("historical catalog dispatch is duplicated")
            dispatch[key] = identity_field
        if rows != sorted(rows, key=lambda row: (row["kind"], row["contract_sha256"])):
            raise SystemContractError("historical contract catalog is not sorted")
        return document, dispatch

    @staticmethod
    def _validate_historical_artifact(
        raw: bytes,
        *,
        dispatch: Mapping[tuple[str, str], str],
        expected_ref: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate a frozen envelope using catalog metadata, never old code."""

        try:
            artifact = parse_canonical_json_bytes(raw, label="historical artifact")
        except ContractError as exc:
            raise SystemContractError("historical artifact is not canonical") from exc
        envelope_fields = {
            "kind",
            "contract_sha256",
            "artifact_id",
            "created_at",
            "payload",
            "semantic_sha256",
        }
        if type(artifact) is not dict or set(artifact) != envelope_fields:
            raise SystemContractError("historical artifact envelope fields are not exact")
        kind = _require_text(artifact["kind"], label="historical artifact kind")
        contract_sha = _require_sha256(
            artifact["contract_sha256"], label="historical artifact contract"
        )
        identity_field = dispatch.get((kind, contract_sha))
        payload = artifact["payload"]
        if identity_field is None or type(payload) is not dict:
            raise SystemContractError("historical artifact contract pair is not anchored")
        identity = payload.get(identity_field)
        if type(identity) is not str or not identity or artifact["artifact_id"] != identity:
            raise SystemContractError("historical artifact identity differs")
        created_at = _require_timestamp(
            artifact["created_at"], label="historical artifact created_at"
        )
        semantic = _require_sha256(
            artifact["semantic_sha256"], label="historical artifact semantic SHA"
        )
        preimage = {
            "domain": "myquant-artifact",
            "kind": kind,
            "contract_sha256": contract_sha,
            "identity_field": identity_field,
            "artifact_id": identity,
            "created_at": created_at,
            "payload": payload,
        }
        if semantic != hashlib.sha256(canonical_json_bytes(preimage)).hexdigest():
            raise SystemContractError("historical artifact semantic SHA differs")
        if expected_ref is not None:
            ref = validate_frozen_object_ref(
                expected_ref,
                label="historical object ref",
                dispatch=dispatch,
            )
            observed = {
                "kind": kind,
                "contract_sha256": contract_sha,
                "artifact_id": identity,
                "semantic_sha256": semantic,
                "byte_sha256": hashlib.sha256(raw).hexdigest(),
            }
            if observed != ref:
                raise SystemContractError("historical artifact exact ref differs")
        return artifact

    def _historical_get_object(
        self,
        ref: Mapping[str, Any],
        *,
        dispatch: Mapping[tuple[str, str], str],
    ) -> dict[str, Any]:
        normalized = validate_frozen_object_ref(
            ref,
            label="historical object ref",
            dispatch=dispatch,
        )
        stored = self._storage.read(_object_path(normalized["kind"], normalized["byte_sha256"]))
        if stored.byte_sha256 != normalized["byte_sha256"]:
            raise SystemContractError("historical object byte hash differs")
        return self._validate_historical_artifact(
            stored.data, dispatch=dispatch, expected_ref=normalized
        )

    def _verify_historical_source_object(self, artifact: Mapping[str, Any]) -> None:
        payload = artifact.get("payload")
        if artifact.get("kind") != "system.source_object" or type(payload) is not dict:
            raise SystemContractError("historical source object is invalid")
        if payload.get("source_root_id") != self.source_root_id:
            raise SystemContractError("historical source root identity differs")
        path = _require_text(
            payload.get("relative_path"), label="historical source_object.relative_path"
        )
        parsed_path = PurePosixPath(path)
        if (
            parsed_path.is_absolute()
            or str(parsed_path) != path
            or "\\" in path
            or any(part in {"", ".", ".."} for part in parsed_path.parts)
        ):
            raise SystemContractError("historical source object path is not canonical")
        media = _require_text(
            payload.get("media_type"), label="historical source_object.media_type"
        )
        format_name = _require_text(
            payload.get("source_format"), label="historical source_object.source_format"
        )
        rule = INITIAL_SOURCE_FORMAT_MEDIA_TYPES.get(format_name)
        if rule is None or media != rule[0]:
            raise SystemContractError("historical source media/format binding is invalid")
        suffixes = rule[1]
        if suffixes and not path.lower().endswith(suffixes):
            raise SystemContractError("historical source path/format binding is invalid")
        expected_sha = _require_sha256(
            payload.get("byte_sha256"), label="historical source byte SHA"
        )
        observed = self._source_storage.hash_workspace_file(
            path, maximum_bytes=INITIAL_HISTORICAL_SOURCE_MAX_BYTES
        )
        if observed.byte_sha256 != expected_sha:
            raise SystemContractError("historical source bytes changed")

    def _verify_historical_source_bundle(
        self,
        artifact: Mapping[str, Any],
        *,
        dispatch: Mapping[tuple[str, str], str],
        ancestors: frozenset[str] = frozenset(),
    ) -> None:
        payload = artifact.get("payload")
        if artifact.get("kind") != "system.source_bundle" or type(payload) is not dict:
            raise SystemContractError("historical source bundle is invalid")
        rows = payload.get("sources")
        if type(rows) is not list:
            raise SystemContractError("historical source bundle rows are invalid")
        byte_sha = hashlib.sha256(canonical_json_bytes(dict(artifact))).hexdigest()
        if byte_sha in ancestors:
            raise SystemContractError("historical source bundle is cyclic")
        roles: list[str] = []
        descendants = ancestors | {byte_sha}
        for row in rows:
            if type(row) is not dict or set(row) != {"role", "source_ref"}:
                raise SystemContractError("historical source bundle row fields differ")
            roles.append(_require_text(row["role"], label="historical source role"))
            child = self._historical_get_object(row["source_ref"], dispatch=dispatch)
            if child["kind"] == "system.source_object":
                self._verify_historical_source_object(child)
            elif child["kind"] == "system.source_bundle":
                self._verify_historical_source_bundle(
                    child, dispatch=dispatch, ancestors=descendants
                )
            else:
                raise SystemContractError("historical source bundle child kind differs")
        if roles != sorted(roles) or len(roles) != len(set(roles)):
            raise SystemContractError("historical source bundle roles are not exact")

    def _verify_historical_initial_generation(  # noqa: C901
        self, generation_id: str
    ) -> dict[str, Any]:
        """Authenticate the immutable initial object graph without current validators."""

        normalized_id = _require_sha256(generation_id, label="historical generation_id")
        stored = self._storage.read(GENERATIONS_ROOT / normalized_id / "manifest.json")
        try:
            provisional = parse_canonical_json_bytes(
                stored.data, label="historical generation manifest"
            )
        except ContractError as exc:
            raise SystemContractError("historical generation manifest is not canonical") from exc
        if type(provisional) is not dict or type(provisional.get("payload")) is not dict:
            raise SystemContractError("historical generation manifest is invalid")
        catalog_sha = _require_sha256(
            provisional["payload"].get("contract_catalog_sha256"),
            label="historical manifest catalog",
        )
        catalog, dispatch = self._read_historical_contract_catalog(catalog_sha)
        manifest = self._validate_historical_artifact(stored.data, dispatch=dispatch)
        if manifest["kind"] != MANIFEST_KIND or manifest["semantic_sha256"] != normalized_id:
            raise SystemContractError("historical generation identity differs")
        manifest_ref = {
            "kind": manifest["kind"],
            "contract_sha256": manifest["contract_sha256"],
            "artifact_id": manifest["artifact_id"],
            "semantic_sha256": manifest["semantic_sha256"],
            "byte_sha256": stored.byte_sha256,
        }
        if self._historical_get_object(manifest_ref, dispatch=dispatch) != manifest:
            raise SystemContractError("historical manifest object binding differs")
        payload = manifest["payload"]
        if (
            payload.get("generation_state") != "OPERATIONAL"
            or payload.get("migration_receipt_ref") is not None
            or payload.get("migration_marker_ref") is not None
            or payload.get("mainline_ref") is not None
        ):
            raise SystemContractError("historical initial manifest state differs")

        def one(field: str) -> dict[str, Any]:
            return self._historical_get_object(payload[field], dispatch=dispatch)

        def many(field: str) -> list[dict[str, Any]]:
            refs = payload.get(field)
            if type(refs) is not list:
                raise SystemContractError(f"historical manifest {field} is invalid")
            return [self._historical_get_object(ref, dispatch=dispatch) for ref in refs]

        release = one("release_manifest_ref")
        sources = many("source_refs")
        for source in sources:
            self._verify_historical_source_bundle(source, dispatch=dispatch)
        factor_source_objects = many("factor_source_object_refs")
        for source in factor_source_objects:
            self._verify_historical_source_object(source)
        factor_policy = one("factor_policy_ref")
        factor_evidence = many("factor_evidence_refs")
        factor_active_set = one("factor_active_set_ref")
        factor_validation_attestation = one("factor_validation_attestation_ref")
        research = many("research_refs")
        fundamental_veto_subject = None
        fundamental_advisory = None
        if len(research) == 1 and research[0].get("kind") == "system.production_bootstrap_receipt":
            from .historical_activation import (
                INITIAL_PRODUCTION_RECEIPT_CONTRACT_SHA256,
                validate_initial_fundamental_advisory,
                validate_initial_fundamental_veto_subject,
                validate_initial_production_receipt,
            )

            if research[0].get("contract_sha256") == INITIAL_PRODUCTION_RECEIPT_CONTRACT_SHA256:
                historical_receipt = validate_initial_production_receipt(research[0])
                fundamental_veto_subject = validate_initial_fundamental_veto_subject(
                    self._historical_get_object(
                        historical_receipt["payload"]["fundamental_veto_subject_ref"],
                        dispatch=dispatch,
                    )
                )
                fundamental_advisory = validate_initial_fundamental_advisory(
                    self._historical_get_object(
                        historical_receipt["payload"]["fundamental_advisory_ref"],
                        dispatch=dispatch,
                    )
                )
        readiness = one("readiness_matrix_ref")
        readiness_payload = readiness.get("payload")
        if type(readiness_payload) is not dict:
            raise SystemContractError("historical readiness payload is invalid")
        factor_status_ref = readiness_payload.get("factor_status_ref")
        if factor_status_ref is None:
            raise SystemContractError("historical Factor status ref is absent")
        factor_status = self._historical_get_object(factor_status_ref, dispatch=dispatch)
        emergency_sha = _require_sha256(
            payload.get("emergency_controller_sha256"),
            label="historical emergency controller SHA",
        )
        from .controller import verify_emergency_controller

        controller = verify_emergency_controller(self, expected_sha256=emergency_sha)
        return {
            "verified": True,
            "generation_state": "OPERATIONAL",
            "generation_id": normalized_id,
            "manifest": manifest,
            "manifest_sha256": stored.byte_sha256,
            "manifest_byte_sha256": stored.byte_sha256,
            "manifest_ref": manifest_ref,
            "contract_catalog": catalog,
            "historical_contract_dispatch": dict(dispatch),
            "release": release,
            "sources": sources,
            "factor_source_object_refs": list(payload["factor_source_object_refs"]),
            "factor_source_objects": factor_source_objects,
            "factor_policy": factor_policy,
            "factor_evidence": factor_evidence,
            "factor_active_set": factor_active_set,
            "factor_validation_attestation_ref": payload["factor_validation_attestation_ref"],
            "factor_validation_attestation": factor_validation_attestation,
            "factor_validation_resolution": None,
            "factor_contextual_result": None,
            "factor_validation_completion": None,
            "factor_source_verification_snapshot": None,
            "factor_status_ref": factor_status_ref,
            "factor_status": factor_status,
            "factor_validation_receipt_ref": None,
            "factor_validation_receipt": None,
            "mainline": None,
            "research": research,
            "fundamental_veto_subject": fundamental_veto_subject,
            "fundamental_advisory": fundamental_advisory,
            "migration_receipt": None,
            "migration_marker": None,
            "readiness": readiness,
            "emergency_controller": {
                **controller,
                "historical": True,
            },
            "deployed_release_verified": False,
            "historical_release_verified": True,
        }

    def _verify_historical_suspended_generation(
        self,
        generation_id: str,
        *,
        expected_manifest_sha256: str,
        expected_manifest_contract_sha256: str,
        expected_manifest_payload_fields: Sequence[str],
    ) -> dict[str, Any]:
        """Authenticate a prebuilt suspended target without current catalog equality."""

        normalized_id = _require_sha256(generation_id, label="historical suspended generation")
        stored = self._storage.read(GENERATIONS_ROOT / normalized_id / "manifest.json")
        if stored.byte_sha256 != _require_sha256(
            expected_manifest_sha256,
            label="controller suspended manifest SHA",
        ):
            raise SystemContractError("historical suspended manifest bytes differ")
        try:
            provisional = parse_canonical_json_bytes(
                stored.data, label="historical suspended manifest"
            )
        except ContractError as exc:
            raise SystemContractError("historical suspended manifest is not canonical") from exc
        if type(provisional) is not dict or type(provisional.get("payload")) is not dict:
            raise SystemContractError("historical suspended manifest is invalid")
        catalog_sha = _require_sha256(
            provisional["payload"].get("contract_catalog_sha256"),
            label="historical suspended catalog",
        )
        catalog, dispatch = self._read_historical_contract_catalog(catalog_sha)
        manifest = self._validate_historical_artifact(stored.data, dispatch=dispatch)
        payload = manifest["payload"]
        payload_fields = tuple(expected_manifest_payload_fields)
        if (
            manifest["kind"] != MANIFEST_KIND
            or manifest["contract_sha256"]
            != _require_sha256(
                expected_manifest_contract_sha256,
                label="controller manifest contract SHA",
            )
            or set(payload) != set(payload_fields)
            or len(payload_fields) != len(set(payload_fields))
            or manifest["semantic_sha256"] != normalized_id
        ):
            raise SystemContractError("historical suspended identity differs")
        manifest_ref = {
            "kind": manifest["kind"],
            "contract_sha256": manifest["contract_sha256"],
            "artifact_id": manifest["artifact_id"],
            "semantic_sha256": manifest["semantic_sha256"],
            "byte_sha256": stored.byte_sha256,
        }
        if self._historical_get_object(manifest_ref, dispatch=dispatch) != manifest:
            raise SystemContractError("historical suspended manifest object differs")
        empty_fields = {
            "source_refs": [],
            "factor_source_object_refs": [],
            "factor_policy_ref": None,
            "factor_evidence_refs": [],
            "factor_active_set_ref": None,
            "factor_validation_attestation_ref": None,
            "mainline_ref": None,
            "research_refs": [],
            "migration_receipt_ref": None,
            "migration_marker_ref": None,
            "emergency_controller_sha256": None,
        }
        if payload.get("generation_state") != "SYSTEM_SUSPENDED" or any(
            payload.get(field) != value for field, value in empty_fields.items()
        ):
            raise SystemContractError("historical suspended closure is not minimal")
        release = self._historical_get_object(payload["release_manifest_ref"], dispatch=dispatch)
        readiness = self._historical_get_object(payload["readiness_matrix_ref"], dispatch=dispatch)
        release_payload = release.get("payload")
        readiness_payload = readiness.get("payload")
        if (
            release.get("kind") != RELEASE_KIND
            or type(release_payload) is not dict
            or set(release_payload)
            != {
                "release_id",
                "state",
                "code_sha256",
                "wheel_sha256",
                "code_manifest_sha256",
            }
            or release_payload.get("state") != "SYSTEM_SUSPENDED"
        ):
            raise SystemContractError("historical suspended release differs")
        if (
            readiness.get("kind") != SUSPENDED_READINESS_KIND
            or type(readiness_payload) is not dict
            or set(readiness_payload) != set(READINESS_FIELDS)
            or any(
                readiness_payload.get(field) != "SUSPENDED"
                for field in ("factor_state", "mainline_state", "investment_state")
            )
            or readiness_payload.get("factor_status_ref") is not None
            or readiness_payload.get("mainline_candidate_ref") is not None
            or readiness_payload.get("admission_route") != "SUSPENDED"
            or type(readiness_payload.get("blockers")) is not list
            or not readiness_payload["blockers"]
        ):
            raise SystemContractError("historical suspended readiness differs")
        return {
            "verified": True,
            "generation_state": "SYSTEM_SUSPENDED",
            "generation_id": normalized_id,
            "manifest": manifest,
            "manifest_sha256": stored.byte_sha256,
            "manifest_byte_sha256": stored.byte_sha256,
            "manifest_ref": manifest_ref,
            "contract_catalog": catalog,
            "release": release,
            "sources": [],
            "factor_source_object_refs": [],
            "factor_source_objects": [],
            "factor_policy": None,
            "factor_evidence": [],
            "factor_active_set": None,
            "factor_validation_attestation_ref": None,
            "factor_validation_attestation": None,
            "factor_validation_resolution": None,
            "factor_contextual_result": None,
            "factor_validation_completion": None,
            "factor_source_verification_snapshot": None,
            "factor_status_ref": None,
            "factor_status": None,
            "factor_validation_receipt_ref": None,
            "factor_validation_receipt": None,
            "mainline": None,
            "research": [],
            "migration_receipt": None,
            "migration_marker": None,
            "readiness": readiness,
            "emergency_controller": None,
            "deployed_release_verified": False,
            "historical_release_verified": True,
        }

    def _resolve_historical_emergency_anchor(self) -> dict[str, Any]:
        """Resolve the initial fixed controller target without marker replay."""

        chain = self._pointer_chain()
        if not chain:
            raise SystemMigrationMarkerAbsent("active pointer is absent")
        initial = chain[-1]
        pointer = initial.get("pointer")
        pointer_sha = initial.get("pointer_byte_sha256")
        if (
            type(pointer) is not dict
            or type(pointer_sha) is not str
            or pointer.get("previous_pointer_sha256") != EMPTY_POINTER_SHA256
        ):
            raise SystemMigrationClosureError("initial pointer ancestry is invalid")
        generation_id = _require_sha256(
            pointer.get("generation_id"), label="initial pointer generation"
        )
        manifest_stored = self._storage.read(GENERATIONS_ROOT / generation_id / "manifest.json")
        if manifest_stored.byte_sha256 != _require_sha256(
            pointer.get("manifest_sha256"), label="initial pointer manifest"
        ):
            raise SystemMigrationClosureError("initial pointer manifest bytes differ")
        try:
            provisional = parse_canonical_json_bytes(
                manifest_stored.data, label="historical initial manifest"
            )
        except ContractError as exc:
            raise SystemMigrationClosureError(
                "historical initial manifest is not canonical"
            ) from exc
        payload = provisional.get("payload") if type(provisional) is dict else None
        if type(payload) is not dict:
            raise SystemMigrationClosureError("historical initial manifest is invalid")
        catalog_sha = _require_sha256(
            payload.get("contract_catalog_sha256"),
            label="historical initial catalog",
        )
        _catalog, dispatch = self._read_historical_contract_catalog(catalog_sha)
        manifest = self._validate_historical_artifact(
            manifest_stored.data,
            dispatch=dispatch,
        )
        if (
            manifest.get("kind") != MANIFEST_KIND
            or manifest.get("semantic_sha256") != generation_id
            or manifest["payload"].get("generation_state") != "OPERATIONAL"
        ):
            raise SystemMigrationClosureError("historical initial generation differs")
        emergency_sha = _require_sha256(
            manifest["payload"].get("emergency_controller_sha256"),
            label="historical emergency controller SHA",
        )
        from .controller import verify_emergency_controller

        controller = verify_emergency_controller(self, expected_sha256=emergency_sha)
        return {
            "initial_pointer": pointer,
            "initial_pointer_byte_sha256": pointer_sha,
            "initial_manifest": manifest,
            "initial_manifest_byte_sha256": manifest_stored.byte_sha256,
            "controller": controller,
        }

    @staticmethod
    def _validated_artifact(value: Mapping[str, Any] | bytes) -> dict[str, Any]:
        try:
            if type(value) is bytes:
                return validate_artifact(value)
            if isinstance(value, Mapping):
                return validate_artifact(dict(value))
        except ContractError as exc:
            raise SystemContractError("artifact contract closure failed") from exc
        raise SystemContractError("artifact must be an envelope or canonical bytes")

    def put_object(self, artifact: Mapping[str, Any] | bytes) -> dict[str, str]:
        """Validate and publish one immutable content-addressed object."""

        document = self._validated_artifact(artifact)
        raw = artifact if type(artifact) is bytes else canonical_json_bytes(document)
        byte_sha256 = hashlib.sha256(raw).hexdigest()
        stored = self._storage.write_exact_once(_object_path(document["kind"], byte_sha256), raw)
        if stored.byte_sha256 != byte_sha256:
            raise SystemStorageError("content-addressed object readback hash mismatch")
        replay = self._validated_artifact(stored.data)
        if replay != document:
            raise SystemStorageError("content-addressed object replay mismatch")
        return object_ref_for_artifact(replay)

    def get_object(
        self,
        ref_or_byte_sha256: Mapping[str, Any] | str,
        *,
        kind: str | None = None,
    ) -> dict[str, Any]:
        """Securely read and fully validate one content-addressed object."""

        expected_ref: dict[str, str] | None
        if type(ref_or_byte_sha256) is str:
            byte_sha256 = _require_sha256(ref_or_byte_sha256, label="object byte SHA-256")
            expected_ref = None
            if type(kind) is not str or not kind:
                raise SystemContractError("object kind is required with a bare byte SHA")
            try:
                object_kind = get_contract(kind).kind
            except ContractError as exc:
                raise SystemContractError("object kind is not compiled") from exc
        else:
            expected_ref = validate_object_ref(ref_or_byte_sha256)
            byte_sha256 = expected_ref["byte_sha256"]
            object_kind = expected_ref["kind"]
        stored = self._storage.read(_object_path(object_kind, byte_sha256))
        if stored.byte_sha256 != byte_sha256:
            raise SystemContractError("content-addressed object byte hash mismatch")
        document = self._validated_artifact(stored.data)
        observed_ref = object_ref_for_artifact(document)
        if expected_ref is not None and observed_ref != expected_ref:
            raise SystemContractError("content-addressed object reference mismatch")
        return document

    read_object = get_object

    def build_installed_component(
        self,
        *,
        component_id: str,
        component_role: str,
        package_name: str,
        module_names: Sequence[str],
        entrypoint_specs: Sequence[tuple[str, str]],
        release_manifest_ref: Mapping[str, Any],
        allowed_source_formats: Sequence[str] = (),
        fallback_allowed: bool = False,
        created_at: str | None = None,
    ) -> dict[str, str]:
        """Derive, seal, publish, and read back one installed component manifest."""

        release_ref, release = self._resolve_ref(
            release_manifest_ref,
            label="release_manifest_ref",
            expected_kinds={RELEASE_KIND},
        )
        if release_ref is None or release is None:
            raise SystemContractError("installed component release is absent")
        _verify_installed_release(release)
        from .components import seal_installed_component_manifest

        artifact = seal_installed_component_manifest(
            component_id=_require_text(component_id, label="component_id"),
            component_role=_require_text(component_role, label="component_role"),
            package_name=_require_text(package_name, label="package_name"),
            module_names=tuple(module_names),
            entrypoint_specs=tuple(entrypoint_specs),
            release_manifest_ref=release_ref,
            allowed_source_formats=tuple(allowed_source_formats),
            fallback_allowed=fallback_allowed,
            created_at=_require_timestamp(created_at or _utc_now(), label="created_at"),
        )
        ref = self.put_object(artifact)
        from .components import validate_installed_component_manifest

        validate_installed_component_manifest(self.get_object(ref))
        return ref

    def build_contextual_validator_component(
        self,
        validation_profile_id: str,
        *,
        release_manifest_ref: Mapping[str, Any],
        created_at: str | None = None,
    ) -> dict[str, str]:
        """Build the fixed whole-package contextual-validator component."""

        from .components import CONTEXTUAL_VALIDATOR_PACKAGE, validation_profile
        from .components import _package_modules as package_modules

        profile = validation_profile(validation_profile_id)
        return self.build_installed_component(
            component_id=f"{validation_profile_id}-component",
            component_role="CONTEXTUAL_VALIDATOR",
            package_name=CONTEXTUAL_VALIDATOR_PACKAGE,
            module_names=package_modules(CONTEXTUAL_VALIDATOR_PACKAGE),
            entrypoint_specs=[(profile["callback_module"], profile["callback_qualified_name"])],
            release_manifest_ref=release_manifest_ref,
            created_at=created_at,
        )

    def build_source_decoder_component(
        self,
        *,
        release_manifest_ref: Mapping[str, Any],
        created_at: str | None = None,
    ) -> dict[str, str]:
        """Build the sole strict-PARQUET decoder component."""

        from .components import STRICT_SOURCE_DECODER_ID, component_registry

        decoder = component_registry()["source_decoder"]
        return self.build_installed_component(
            component_id=STRICT_SOURCE_DECODER_ID,
            component_role="SOURCE_DECODER",
            package_name="quant_investor.factors.governance",
            module_names=decoder["module_names"],
            entrypoint_specs=[(decoder["module_name"], decoder["qualified_name"])],
            release_manifest_ref=release_manifest_ref,
            allowed_source_formats=decoder["allowed_source_formats"],
            fallback_allowed=decoder["fallback_allowed"],
            created_at=created_at,
        )

    @staticmethod
    def _validate_candidate_pointer(
        document: Mapping[str, Any] | bytes,
        *,
        validation_namespace_id: str,
    ) -> dict[str, Any]:
        try:
            if type(document) is bytes:
                pointer = parse_canonical_json_bytes(document, label="candidate pointer")
            elif isinstance(document, Mapping):
                pointer = dict(document)
            else:
                raise SystemContractError("candidate pointer must be canonical JSON")
        except ContractError as exc:
            raise SystemContractError("candidate pointer is not canonical JSON") from exc
        if type(pointer) is not dict or set(pointer) != set(CANDIDATE_POINTER_FIELDS):
            raise SystemContractError("candidate pointer fields are not exact")
        candidate_ref = validate_object_ref(
            pointer.get("candidate_state_ref"), label="candidate_state_ref"
        )
        if candidate_ref["kind"] != FACTOR_COMPOSITE_STATE_KIND:
            raise SystemContractError("candidate pointer has the wrong compiled state kind")
        _require_pointer_sha256(
            pointer.get("previous_pointer_sha256"),
            label="candidate.previous_pointer_sha256",
        )
        _require_timestamp(pointer.get("stored_at"), label="candidate.stored_at")
        _require_text(pointer.get("os_actor"), label="candidate.os_actor")
        if pointer.get("authority") != "NON_AUTHORIZING":
            raise SystemContractError("candidate pointer must be non-authorizing")
        _require_text(validation_namespace_id, label="validation_namespace_id")
        canonical_json_bytes(pointer)
        return pointer

    def read_candidate_state(self, validation_namespace_id: str) -> dict[str, Any] | None:
        """Read the current non-authorizing composite candidate for one namespace."""

        pointer_path, _, _ = _candidate_paths(validation_namespace_id)
        stored = self._storage.read_optional(pointer_path)
        if stored is None:
            return None
        pointer = self._validate_candidate_pointer(
            stored.data, validation_namespace_id=validation_namespace_id
        )
        candidate_ref = validate_object_ref(pointer["candidate_state_ref"])
        candidate = self.get_object(candidate_ref)
        payload = candidate.get("payload")
        if (
            type(payload) is not dict
            or payload.get("authority") != "NON_AUTHORIZING"
            or payload.get("custody_namespace_id") != validation_namespace_id
        ):
            raise SystemContractError("candidate composite namespace/authority mismatch")
        return {
            "pointer": pointer,
            "pointer_byte_sha256": stored.byte_sha256,
            "candidate_state_ref": candidate_ref,
            "candidate_state": candidate,
        }

    def read_candidate_transaction(
        self,
        validation_namespace_id: str,
        transaction_id: str,
    ) -> dict[str, Any] | None:
        """Read one immutable non-authorizing candidate transaction intent."""

        namespace = _require_text(validation_namespace_id, label="validation_namespace_id")
        transaction = _require_text(transaction_id, label="transaction_id")
        directory = _candidate_transaction_path(namespace, transaction)
        try:
            files = self._storage.read_exact_directory(
                directory,
                expected_names=frozenset({"intent.json"}),
            )
        except SystemNotFound:
            return None
        intent = validate_candidate_transaction_intent(files["intent.json"].data)
        if (
            intent["intent_id"] != candidate_transaction_intent_id(namespace, transaction)
            or intent["validation_namespace_id"] != namespace
            or intent["transaction_id"] != transaction
        ):
            raise SystemContractError("candidate transaction intent path binding differs")
        return intent

    def begin_candidate_transaction(
        self,
        validation_namespace_id: str,
        transaction_id: str,
        *,
        expected_pointer_sha256: str,
        transaction_plan: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Fix the exact plan and System UTC before any candidate object write."""

        namespace = _require_text(validation_namespace_id, label="validation_namespace_id")
        transaction = _require_text(transaction_id, label="transaction_id")
        expected = _require_pointer_sha256(expected_pointer_sha256, label="expected_pointer_sha256")
        if not isinstance(transaction_plan, Mapping):
            raise SystemContractError("candidate transaction plan must be an object")
        plan = dict(transaction_plan)
        directory = _candidate_transaction_path(namespace, transaction)
        validation_root = VALIDATION_RUNS_ROOT / validation_namespace_path_sha256(namespace)
        self._storage.ensure_directory(validation_root)
        with self._storage.exclusive_lock(validation_root / ".lock"):
            existing = self.read_candidate_transaction(namespace, transaction)
            if existing is not None:
                expected_intent = build_candidate_transaction_intent(
                    validation_namespace_id=namespace,
                    transaction_id=transaction,
                    expected_pointer_sha256=expected,
                    previous_candidate_state_ref=existing["previous_candidate_state_ref"],
                    transaction_plan=plan,
                    trusted_at=existing["trusted_at"],
                )
                if existing != expected_intent:
                    raise SystemImmutableConflict(
                        "candidate transaction intent conflicts with immutable plan"
                    )
                return existing

            current = self.read_candidate_state(namespace)
            observed = EMPTY_POINTER_SHA256 if current is None else current["pointer_byte_sha256"]
            if observed != expected:
                raise SystemCASMismatch(expected, observed)
            previous_ref = None if current is None else current["candidate_state_ref"]
            trusted_at = _utc_now()
            if current is not None and trusted_at < current["pointer"]["stored_at"]:
                raise SystemPreconditionError(
                    "system clock precedes candidate state",
                    code="SYSTEM_CLOCK_ROLLBACK",
                )
            intent = build_candidate_transaction_intent(
                validation_namespace_id=namespace,
                transaction_id=transaction,
                expected_pointer_sha256=expected,
                previous_candidate_state_ref=previous_ref,
                transaction_plan=plan,
                trusted_at=trusted_at,
            )
            raw = canonical_json_bytes(intent)
            readback = self._storage.write_atomic_directory(
                directory,
                {"intent.json": raw},
            )
            if readback["intent.json"].data != raw:
                raise SystemStorageError("candidate transaction intent exact readback mismatch")
            replay = self.read_candidate_transaction(namespace, transaction)
            if replay != intent:
                raise SystemStorageError("candidate transaction intent replay mismatch")
            return intent

    def compare_and_swap_candidate_state(  # noqa: C901
        self,
        validation_namespace_id: str,
        candidate_state_ref: Mapping[str, Any],
        *,
        expected_pointer_sha256: str,
        stored_at: str | None = None,
        os_actor: str | None = None,
    ) -> dict[str, Any]:
        """CAS one Factor-produced composite candidate without granting authority."""

        namespace = _require_text(validation_namespace_id, label="validation_namespace_id")
        expected = _require_pointer_sha256(expected_pointer_sha256, label="expected_pointer_sha256")
        ref = validate_object_ref(candidate_state_ref, label="candidate_state_ref")
        if ref["kind"] != FACTOR_COMPOSITE_STATE_KIND:
            raise SystemContractError("candidate state ref has the wrong compiled kind")
        candidate = self.get_object(ref)
        payload = candidate.get("payload")
        if (
            type(payload) is not dict
            or payload.get("authority") != "NON_AUTHORIZING"
            or payload.get("custody_namespace_id") != namespace
        ):
            raise SystemContractError("candidate state namespace/authority mismatch")
        transaction_id = _require_text(
            payload.get("transaction_id"), label="candidate_state.transaction_id"
        )
        intent = self.read_candidate_transaction(namespace, transaction_id)
        if intent is None:
            raise SystemPreconditionError(
                "candidate transaction intent is absent",
                code="SYSTEM_CANDIDATE_TRANSACTION_REQUIRED",
            )
        if intent["expected_pointer_sha256"] != expected:
            raise SystemContractError("candidate transaction expected pointer differs")
        timestamp = intent["trusted_at"]
        if stored_at is not None and _require_timestamp(stored_at, label="stored_at") != timestamp:
            raise SystemContractError("candidate pointer timestamp differs from intent")
        if (
            candidate.get("created_at") != timestamp
            or payload.get("last_stored_at") != timestamp
            or payload.get("previous_composite_state_ref") != intent["previous_candidate_state_ref"]
        ):
            raise SystemContractError("candidate state differs from transaction intent")
        actor = _require_text(os_actor or f"uid:{os.geteuid()}", label="os_actor")
        pointer = {
            "candidate_state_ref": ref,
            "previous_pointer_sha256": expected,
            "stored_at": timestamp,
            "os_actor": actor,
            "authority": "NON_AUTHORIZING",
        }
        pointer_path, history_root, lock_path = _candidate_paths(namespace)
        validation_root = VALIDATION_RUNS_ROOT / validation_namespace_path_sha256(namespace)
        self._storage.ensure_directory(validation_root)
        with self._storage.exclusive_lock(validation_root / ".lock"):
            current = self._storage.read_optional(pointer_path)
            observed = EMPTY_POINTER_SHA256 if current is None else current.byte_sha256
            if observed != expected:
                raise SystemCASMismatch(expected, observed)
            expected_previous_ref = None
            if current is not None:
                current_pointer = self._validate_candidate_pointer(
                    current.data,
                    validation_namespace_id=namespace,
                )
                expected_previous_ref = current_pointer["candidate_state_ref"]
            if payload.get("previous_composite_state_ref") != expected_previous_ref:
                raise SystemContractError(
                    "candidate state predecessor does not match the authoritative head"
                )
            stored = self._storage.compare_and_swap_pointer(
                canonical_json_bytes(pointer),
                pointer_path=pointer_path,
                history_root=history_root,
                lock_path=lock_path,
                expected_sha256=expected,
            )
        result = self.read_candidate_state(namespace)
        if result is None or result["pointer_byte_sha256"] != stored.byte_sha256:
            raise SystemStorageError("candidate pointer disappeared after CAS")
        return result

    def build_validation_run_request(
        self,
        *,
        release_manifest_ref: Mapping[str, Any],
        factor_validator_manifest_ref: Mapping[str, Any],
        intrinsic_receipt_ref: Mapping[str, Any],
        candidate_state_ref: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Derive and store the deterministic fixed-profile validation request."""

        from .validation import build_validation_run_request

        return build_validation_run_request(
            self,
            release_manifest_ref=release_manifest_ref,
            factor_validator_manifest_ref=factor_validator_manifest_ref,
            intrinsic_receipt_ref=intrinsic_receipt_ref,
            candidate_state_ref=candidate_state_ref,
        )

    def run_validation(self, request: Mapping[str, Any] | bytes) -> dict[str, Any]:
        """Run/recover one exact contextual validation through the compiled callback."""

        from .validation import run_validation

        return run_validation(self, request)

    def resolve_validation_attestation(
        self,
        validation_attestation_ref: Mapping[str, Any],
        *,
        verification_level: str = "full",
    ) -> dict[str, Any]:
        """Resolve protected custody/completion for one exact attestation ref."""

        from .validation import resolve_validation_attestation

        return resolve_validation_attestation(
            self,
            validation_attestation_ref,
            verification_level=verification_level,
        )

    def assemble_from_request(self, document: Mapping[str, Any] | bytes) -> dict[str, Any]:
        """Decode one sealed assembly request and assemble its stored refs."""

        from .requests import decode_assembly_request

        return self.assemble_generation(**decode_assembly_request(document))

    @staticmethod
    def _validate_source_metadata(
        *, relative_path: Any, media_type: Any, source_format: Any
    ) -> tuple[str, str, str]:
        path = _require_text(relative_path, label="source_object.relative_path")
        media = _require_text(media_type, label="source_object.media_type")
        format_name = _require_text(source_format, label="source_object.source_format")
        rule = SOURCE_FORMAT_MEDIA_TYPES.get(format_name)
        if rule is None or media != rule[0]:
            raise SystemContractError("source object media/format binding is invalid")
        suffixes = rule[1]
        if suffixes and not path.lower().endswith(suffixes):
            raise SystemContractError("source object path/format binding is invalid")
        return path, media, format_name

    def put_source_file(
        self,
        relative_path: str,
        *,
        source_object_id: str,
        media_type: str,
        source_format: str,
        created_at: str,
        expected_byte_sha256: str | None = None,
    ) -> dict[str, str]:
        """Seal a descriptor after streaming the canonical source in place."""

        path, media, format_name = self._validate_source_metadata(
            relative_path=relative_path,
            media_type=media_type,
            source_format=source_format,
        )
        source = self._source_storage.hash_workspace_file(path, maximum_bytes=self.max_source_bytes)
        if expected_byte_sha256 is not None:
            expected = _require_sha256(expected_byte_sha256, label="expected_byte_sha256")
            if source.byte_sha256 != expected:
                raise SystemContractError("source file byte SHA does not match expectation")
        descriptor = seal_artifact(
            "system.source_object",
            {
                "source_object_id": _require_text(source_object_id, label="source_object_id"),
                "source_root_id": self.source_root_id,
                "relative_path": path,
                "media_type": media,
                "source_format": format_name,
                "byte_sha256": source.byte_sha256,
            },
            created_at=created_at,
        )
        return self.put_object(descriptor)

    def read_source_object_bytes(
        self,
        source_object_ref: Mapping[str, Any],
        *,
        maximum_bytes: int,
    ) -> tuple[dict[str, Any], bytes]:
        """Read one exact source through the owner/mode/NOFOLLOW security seam."""

        from .components import MAXIMUM_FACTOR_SOURCE_OBJECT_BYTES

        if (
            type(maximum_bytes) is not int
            or maximum_bytes <= 0
            or maximum_bytes > MAXIMUM_FACTOR_SOURCE_OBJECT_BYTES
            or maximum_bytes > self.max_source_bytes
        ):
            raise SystemSecurityError("source object byte read bound is invalid")
        ref = validate_object_ref(source_object_ref, label="source_object_ref")
        if ref["kind"] != "system.source_object":
            raise SystemContractError("source object ref has the wrong kind")
        artifact = self.get_object(ref)
        payload = artifact.get("payload")
        if type(payload) is not dict or payload.get("source_root_id") != self.source_root_id:
            raise SystemContractError("source object root identity mismatch")
        path, _, _ = self._validate_source_metadata(
            relative_path=payload.get("relative_path"),
            media_type=payload.get("media_type"),
            source_format=payload.get("source_format"),
        )
        expected_sha = _require_sha256(
            payload.get("byte_sha256"), label="source_object.byte_sha256"
        )
        stored = self._source_storage.read_workspace_file_bytes(path, maximum_bytes=maximum_bytes)
        if stored.byte_sha256 != expected_sha:
            raise SystemContractError("canonical source file byte hash mismatch")
        return dict(payload), stored.data

    @contextmanager
    def open_source_object(
        self,
        source_object_ref: Mapping[str, Any],
        *,
        maximum_bytes: int,
        decoded_reservation_bytes: int,
    ) -> Iterator[tuple[dict[str, Any], FileIO]]:
        """Open one strict-Parquet source without materializing its raw bytes.

        The returned unbuffered binary stream implements ``read``, ``readinto``,
        ``seek``, ``tell``, ``readable``, ``seekable``, and ``fileno`` for
        ``pyarrow.parquet``.  Its decoded-memory lease remains held until the
        caller exits this context, so no decoded table may escape the ``with``
        scope.
        """

        from .components import (
            MAXIMUM_DECODE_RESERVATION_BYTES,
            MAXIMUM_FACTOR_SOURCE_OBJECT_BYTES,
        )

        if (
            type(maximum_bytes) is not int
            or maximum_bytes <= 0
            or maximum_bytes > MAXIMUM_FACTOR_SOURCE_OBJECT_BYTES
            or maximum_bytes > self.max_source_bytes
        ):
            raise SystemSecurityError("source object stream byte bound is invalid")
        if (
            type(decoded_reservation_bytes) is not int
            or decoded_reservation_bytes <= 0
            or decoded_reservation_bytes > MAXIMUM_DECODE_RESERVATION_BYTES
        ):
            raise SystemSecurityError("decoded source reservation is invalid")
        ref = validate_object_ref(source_object_ref, label="source_object_ref")
        if ref["kind"] != "system.source_object":
            raise SystemContractError("source object ref has the wrong kind")
        artifact = self.get_object(ref)
        payload = artifact.get("payload")
        if type(payload) is not dict or payload.get("source_root_id") != self.source_root_id:
            raise SystemContractError("source object root identity mismatch")
        path, media, source_format = self._validate_source_metadata(
            relative_path=payload.get("relative_path"),
            media_type=payload.get("media_type"),
            source_format=payload.get("source_format"),
        )
        if source_format != "PARQUET" or media != "application/vnd.apache.parquet":
            raise SystemContractError("streaming decoder seam accepts strict PARQUET only")
        expected_sha = _require_sha256(
            payload.get("byte_sha256"), label="source_object.byte_sha256"
        )
        with self._decode_memory_budget.reserve(decoded_reservation_bytes):
            with self._source_storage.open_workspace_file(path, maximum_bytes=maximum_bytes) as (
                descriptor,
                before_identity,
            ):
                digest = hashlib.sha256()
                observed_size = 0
                os.lseek(descriptor, 0, os.SEEK_SET)
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    observed_size += len(chunk)
                after_hash = source_stat_identity(os.fstat(descriptor))
                if (
                    dict(before_identity) != after_hash
                    or observed_size != after_hash["st_size"]
                    or digest.hexdigest() != expected_sha
                ):
                    raise SystemContractError("canonical source stream hash/stat binding mismatch")
                os.lseek(descriptor, 0, os.SEEK_SET)
                duplicate = os.dup(descriptor)
                os.set_inheritable(duplicate, False)
                stream = FileIO(duplicate, mode="rb", closefd=True)
                try:
                    yield dict(payload), stream
                finally:
                    stream.close()

    def inspect_source_object(
        self,
        source_object_ref: Mapping[str, Any],
        *,
        full_hash: bool,
        maximum_bytes: int,
    ) -> dict[str, Any]:
        """Return a secure descriptor/stat projection, optionally with full byte hash."""

        if type(full_hash) is not bool:
            raise SystemContractError("full_hash must be boolean")
        from .components import MAXIMUM_FACTOR_SOURCE_OBJECT_BYTES

        if (
            type(maximum_bytes) is not int
            or maximum_bytes <= 0
            or maximum_bytes > MAXIMUM_FACTOR_SOURCE_OBJECT_BYTES
            or maximum_bytes > self.max_source_bytes
        ):
            raise SystemSecurityError("source object inspection byte bound is invalid")
        ref = validate_object_ref(source_object_ref, label="source_object_ref")
        if ref["kind"] != "system.source_object":
            raise SystemContractError("source object ref has the wrong kind")
        artifact = self.get_object(ref)
        payload = artifact.get("payload")
        if type(payload) is not dict or payload.get("source_root_id") != self.source_root_id:
            raise SystemContractError("source object root identity mismatch")
        path, media, format_name = self._validate_source_metadata(
            relative_path=payload.get("relative_path"),
            media_type=payload.get("media_type"),
            source_format=payload.get("source_format"),
        )
        expected_sha = _require_sha256(
            payload.get("byte_sha256"), label="source_object.byte_sha256"
        )
        observed = (
            self._source_storage.hash_workspace_file(path, maximum_bytes=maximum_bytes)
            if full_hash
            else self._source_storage.stat_workspace_file(path, maximum_bytes=maximum_bytes)
        )
        if full_hash and observed.byte_sha256 != expected_sha:
            raise SystemContractError("canonical source file byte hash mismatch")
        if observed.stat_identity is None:
            raise SystemStorageError("canonical source stat identity is absent")
        return {
            "source_object_ref": ref,
            "source_root_id": self.source_root_id,
            "relative_path": path,
            "media_type": media,
            "source_format": format_name,
            "byte_sha256": expected_sha,
            "size": observed.size,
            "stat_identity": dict(observed.stat_identity),
        }

    def _verify_source_object(self, artifact: Mapping[str, Any]) -> dict[str, Any]:
        if artifact.get("kind") != "system.source_object":
            raise SystemContractError("source descriptor has the wrong kind")
        payload = artifact["payload"]
        if payload.get("source_root_id") != self.source_root_id:
            raise SystemContractError("source descriptor root identity mismatch")
        path, _, _ = self._validate_source_metadata(
            relative_path=payload.get("relative_path"),
            media_type=payload.get("media_type"),
            source_format=payload.get("source_format"),
        )
        expected_sha = _require_sha256(
            payload.get("byte_sha256"), label="source_object.byte_sha256"
        )
        observed = self._source_storage.hash_workspace_file(
            path, maximum_bytes=self.max_source_bytes
        )
        if observed.byte_sha256 != expected_sha:
            raise SystemContractError("canonical source file byte hash mismatch")
        return dict(payload)

    def _verify_source_bundle(
        self,
        artifact: Mapping[str, Any],
        *,
        require_sources: bool,
        ancestors: frozenset[str] = frozenset(),
    ) -> tuple[str, ...]:
        if artifact.get("kind") != "system.source_bundle":
            raise SystemContractError("source bundle has the wrong kind")
        byte_sha = artifact_byte_sha256(dict(artifact))
        if byte_sha in ancestors:
            raise SystemContractError("source bundle closure is cyclic")
        payload = artifact["payload"]
        _require_text(payload.get("state"), label="source_bundle.state")
        values = payload.get("sources")
        if type(values) is not list or (require_sources and not values):
            raise SystemContractError("source bundle sources have invalid cardinality")
        roles: list[str] = []
        refs: list[dict[str, str]] = []
        for index, row in enumerate(values):
            if type(row) is not dict or set(row) != {"role", "source_ref"}:
                raise SystemContractError("source bundle row fields are not exact")
            roles.append(_require_text(row.get("role"), label=f"source role {index}"))
            refs.append(
                validate_object_ref(row.get("source_ref"), label=f"source bundle ref {index}")
            )
        if roles != sorted(roles) or len(roles) != len(set(roles)):
            raise SystemContractError("source bundle roles must be sorted and unique")
        descendants = ancestors | {byte_sha}
        for ref in refs:
            source = self.get_object(ref)
            if source["kind"] == "system.source_object":
                self._verify_source_object(source)
            elif source["kind"] == "system.source_bundle":
                self._verify_source_bundle(
                    source, require_sources=require_sources, ancestors=descendants
                )
            else:
                raise SystemContractError(
                    "source bundle refs must resolve to source objects or bundles"
                )
        return tuple(roles)

    def _resolve_ref(
        self,
        value: Any,
        *,
        label: str,
        expected_kinds: frozenset[str] | set[str] | None = None,
        nullable: bool = False,
    ) -> tuple[dict[str, str] | None, dict[str, Any] | None]:
        if value is None:
            if nullable:
                return None, None
            raise SystemContractError(f"{label} must be an exact stored object ref")
        ref = validate_object_ref(value, label=label)
        artifact = self.get_object(ref)
        if expected_kinds is not None and artifact["kind"] not in expected_kinds:
            raise SystemContractError(f"{label} has the wrong artifact kind")
        return ref, artifact

    def _resolve_refs(
        self,
        values: Any,
        *,
        label: str,
        minimum: int = 0,
        expected_kinds: frozenset[str] | set[str] | None = None,
    ) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
        if type(values) not in {list, tuple} or len(values) < minimum:
            raise SystemContractError(f"{label} has invalid cardinality")
        rows = [
            self._resolve_ref(
                value,
                label=f"{label}[{index}]",
                expected_kinds=expected_kinds,
            )
            for index, value in enumerate(values)
        ]
        refs = [row[0] for row in rows]
        artifacts = [row[1] for row in rows]
        if any(ref is None for ref in refs) or any(artifact is None for artifact in artifacts):
            raise SystemContractError(f"{label} contains a null ref")
        normalized_refs = [dict(ref) for ref in refs if ref is not None]
        normalized_artifacts = [dict(artifact) for artifact in artifacts if artifact is not None]
        ref_keys = [_object_ref_sort_key(ref) for ref in normalized_refs]
        if len(set(ref_keys)) != len(ref_keys):
            raise SystemContractError(f"{label} contains duplicate refs")
        ordered_rows = sorted(
            zip(normalized_refs, normalized_artifacts, strict=True),
            key=lambda row: _object_ref_sort_key(row[0]),
        )
        return [row[0] for row in ordered_rows], [row[1] for row in ordered_rows]

    def _source_bundle_descriptors(
        self,
        bundle: Mapping[str, Any],
        *,
        ancestors: frozenset[str] = frozenset(),
    ) -> tuple[dict[str, Any], ...]:
        bundle_sha = artifact_byte_sha256(dict(bundle))
        if bundle_sha in ancestors:
            raise SystemContractError("source bundle closure is cyclic")
        result: list[dict[str, Any]] = []
        for row in bundle["payload"]["sources"]:
            source = self.get_object(validate_object_ref(row["source_ref"]))
            if source["kind"] == "system.source_object":
                result.append(self._verify_source_object(source))
            elif source["kind"] == "system.source_bundle":
                result.extend(
                    self._source_bundle_descriptors(source, ancestors=ancestors | {bundle_sha})
                )
            else:
                raise SystemContractError(
                    "source bundle refs must resolve to source objects or bundles"
                )
        return tuple(result)

    def _require_bundle_formats(
        self,
        bundle: Mapping[str, Any],
        *,
        label: str,
        allowed: frozenset[str],
        required: frozenset[str] = frozenset(),
    ) -> None:
        descriptors = self._source_bundle_descriptors(bundle)
        if not descriptors:
            raise SystemContractError(f"{label} source closure is empty")
        formats = {row["source_format"] for row in descriptors}
        if not formats <= allowed or not required <= formats:
            raise SystemContractError(f"{label} source format closure is invalid")

    def _validate_operational_sources(self, sources: Sequence[Mapping[str, Any]]) -> None:
        rows_by_role: dict[str, dict[str, Any]] = {}
        for bundle in sources:
            for row in bundle["payload"]["sources"]:
                role = row["role"]
                if role in rows_by_role:
                    raise SystemContractError("operational data roles are duplicated")
                rows_by_role[role] = row
        if tuple(sorted(rows_by_role)) != OPERATIONAL_DATA_ROLES:
            raise SystemContractError("operational data role closure is not exact")

        role_rules = {
            "exchange_calendar": (
                frozenset({"BINARY", "JSON", "PARQUET"}),
                frozenset({"PARQUET"}),
            ),
            "fundamental_generation": (
                frozenset({"JSON", "PARQUET"}),
                frozenset({"JSON", "PARQUET"}),
            ),
            "market_snapshot": (
                frozenset({"JSON", "PARQUET"}),
                frozenset({"JSON", "PARQUET"}),
            ),
            "pit_membership": (
                frozenset({"JSON", "PARQUET"}),
                frozenset({"PARQUET"}),
            ),
        }
        for role in OPERATIONAL_DATA_ROLES:
            source = self.get_object(
                validate_object_ref(
                    rows_by_role[role]["source_ref"],
                    label=f"operational source {role}",
                )
            )
            if source["kind"] == "system.source_object":
                payload = self._verify_source_object(source)
                formats = frozenset({payload["source_format"]})
                allowed, required = role_rules[role]
                if not formats <= allowed or not required <= formats:
                    raise SystemContractError(f"operational source {role} format is invalid")
            elif source["kind"] == "system.source_bundle":
                allowed, required = role_rules[role]
                self._require_bundle_formats(
                    source, label=f"operational source {role}", allowed=allowed, required=required
                )
            else:
                raise SystemContractError("operational data source kind is invalid")

    def _validate_factor_generation_closure(  # noqa: C901
        self,
        *,
        policy_ref: Mapping[str, Any],
        policy: Mapping[str, Any],
        evidence_refs: Sequence[Mapping[str, Any]],
        evidence: Sequence[Mapping[str, Any]],
        active_set_ref: Mapping[str, Any],
        active_set: Mapping[str, Any],
        factor_source_object_refs: Sequence[Mapping[str, Any]],
        validation_attestation_ref: Mapping[str, Any],
        validation_attestation: Mapping[str, Any],
        readiness: Mapping[str, Any],
        factor_status: Mapping[str, Any] | None,
        verification_level: str,
    ) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        """Bind a Factor-owned validation receipt without interpreting Factor policy."""

        normalized_policy_ref = validate_object_ref(policy_ref, label="factor_policy_ref")
        normalized_active_ref = validate_object_ref(
            active_set_ref,
            label="factor_active_set_ref",
        )
        normalized_evidence_refs = [
            validate_object_ref(ref, label=f"factor_evidence_refs[{index}]")
            for index, ref in enumerate(evidence_refs)
        ]
        factor_objects = (
            (normalized_policy_ref, policy),
            (normalized_active_ref, active_set),
            *zip(normalized_evidence_refs, evidence, strict=True),
        )
        for ref, artifact in factor_objects:
            if object_ref_for_artifact(artifact) != ref:
                raise SystemContractError("Factor generation object ref mismatch")
        if not normalized_policy_ref["kind"].startswith("factor."):
            raise SystemContractError("Factor policy ref must bind a compiled Factor kind")
        if normalized_active_ref["kind"] not in FACTOR_ACTIVE_SET_KINDS:
            raise SystemContractError("Factor active-set kind is invalid")

        if factor_status is None or factor_status.get("kind") != FACTOR_STATUS_KIND:
            raise SystemContractError("Factor status binding is absent")
        status_payload = factor_status.get("payload")
        if type(status_payload) is not dict:
            raise SystemContractError("Factor status payload is invalid")
        status_active = status_payload.get("active")
        if type(status_active) is not dict:
            raise SystemContractError("Factor status active projection is invalid")

        status_set_ref = validate_object_ref(
            status_active.get("factor_set_ref"),
            label="factor_status.active.factor_set_ref",
        )
        if status_set_ref != normalized_active_ref:
            raise SystemContractError("Factor status active-set binding mismatch")
        receipt_ref = validate_object_ref(
            status_active.get("validation_receipt_ref"),
            label="factor_status.active.validation_receipt_ref",
        )
        if receipt_ref["kind"] != FACTOR_VALIDATION_RECEIPT_KIND:
            raise SystemContractError("Factor status validation receipt kind is invalid")
        receipt = self.get_object(receipt_ref)
        receipt_payload = receipt.get("payload")
        if type(receipt_payload) is not dict:
            raise SystemContractError("Factor validation receipt payload is invalid")

        receipt_policy_ref = validate_object_ref(
            receipt_payload.get("policy_ref"),
            label="factor.validation_receipt.policy_ref",
        )
        receipt_active_ref = validate_object_ref(
            receipt_payload.get("active_set_ref"),
            label="factor.validation_receipt.active_set_ref",
        )
        receipt_evidence_value = receipt_payload.get("evidence_refs")
        if type(receipt_evidence_value) is not list:
            raise SystemContractError("Factor validation receipt evidence refs must be a list")
        receipt_evidence_refs = [
            validate_object_ref(
                ref,
                label=f"factor.validation_receipt.evidence_refs[{index}]",
            )
            for index, ref in enumerate(receipt_evidence_value)
        ]
        receipt_evidence_keys = [_object_ref_sort_key(ref) for ref in receipt_evidence_refs]
        if receipt_evidence_keys != sorted(receipt_evidence_keys) or len(
            set(receipt_evidence_keys)
        ) != len(receipt_evidence_keys):
            raise SystemContractError(
                "Factor validation receipt evidence refs must be sorted and unique"
            )
        if (
            receipt_policy_ref != normalized_policy_ref
            or receipt_active_ref != normalized_active_ref
            or receipt_evidence_refs != normalized_evidence_refs
        ):
            raise SystemContractError("Factor validation receipt generation binding mismatch")
        if (
            receipt_payload.get("validated") is not True
            or receipt_payload.get("authority") != "NON_AUTHORIZING"
        ):
            raise SystemContractError("Factor validation receipt is not valid and non-authorizing")

        contextual_result_ref = validate_object_ref(
            status_active.get("contextual_result_ref"),
            label="factor_status.active.contextual_result_ref",
        )
        if contextual_result_ref["kind"] != "factor.contextual_validation_result":
            raise SystemContractError("Factor status contextual result kind is invalid")
        status_attestation_ref = validate_object_ref(
            status_active.get("validation_attestation_ref"),
            label="factor_status.active.validation_attestation_ref",
        )
        normalized_attestation_ref = validate_object_ref(
            validation_attestation_ref,
            label="factor_validation_attestation_ref",
        )
        if (
            status_attestation_ref != normalized_attestation_ref
            or object_ref_for_artifact(validation_attestation) != normalized_attestation_ref
        ):
            raise SystemContractError("Factor status/manifest attestation binding mismatch")
        resolution = self.resolve_validation_attestation(
            normalized_attestation_ref,
            verification_level=verification_level,
        )
        if (
            resolution["contextual_result_ref"] != contextual_result_ref
            or resolution["validation_attestation"] != validation_attestation
        ):
            raise SystemContractError("Factor contextual custody binding mismatch")
        attestation_payload = validation_attestation.get("payload")
        if type(attestation_payload) is not dict:
            raise SystemContractError("Factor validation attestation payload is invalid")
        normalized_factor_sources = [
            validate_object_ref(ref, label=f"factor_source_object_refs[{index}]")
            for index, ref in enumerate(factor_source_object_refs)
        ]
        if (
            attestation_payload.get("intrinsic_receipt_ref") != receipt_ref
            or attestation_payload.get("policy_ref") != normalized_policy_ref
            or attestation_payload.get("active_set_ref") != normalized_active_ref
            or attestation_payload.get("evidence_refs") != normalized_evidence_refs
            or attestation_payload.get("source_object_refs") != normalized_factor_sources
        ):
            raise SystemContractError("Factor attestation generation binding mismatch")
        context_payload = resolution["contextual_result"].get("payload")
        if (
            type(context_payload) is not dict
            or context_payload.get("lane") != attestation_payload.get("validation_lane")
            or status_active.get("lane") != context_payload.get("lane")
        ):
            raise SystemContractError("Factor validation lane binding mismatch")

        readiness_payload = readiness.get("payload")
        if type(readiness_payload) is not dict:
            raise SystemContractError("generation readiness payload is invalid")
        status_state = _require_text(
            status_payload.get("readiness"),
            label="factor_status.readiness",
        )
        if status_state != readiness_payload.get("factor_state"):
            raise SystemContractError("Factor status/readiness state mismatch")
        for field in ("admission_route", "producer_identity"):
            status_value = _require_text(
                status_active.get(field),
                label=f"factor_status.active.{field}",
            )
            if status_value != readiness_payload.get(field):
                raise SystemContractError(f"Factor status/readiness {field} mismatch")
        status_blockers = status_payload.get("blockers")
        readiness_blockers = readiness_payload.get("blockers")
        if (
            type(status_blockers) is not list
            or type(readiness_blockers) is not list
            or any(
                _require_text(value, label=f"factor_status.blockers[{index}]") != value
                for index, value in enumerate(status_blockers)
            )
            or status_blockers != sorted(status_blockers)
            or len(status_blockers) != len(set(status_blockers))
            or not set(status_blockers) <= set(readiness_blockers)
        ):
            raise SystemContractError("Factor status blockers are not preserved by readiness")
        if status_payload.get("activation_mutation_authorized") is not False:
            raise SystemContractError("Factor status must remain non-authorizing")
        if status_active.get("state") != "ACTIVE":
            raise SystemContractError("Factor status must bind the active generation set")
        if status_state == "READY" and status_blockers:
            raise SystemContractError("Factor READY status is inconsistent")

        return receipt_ref, receipt, resolution

    def assemble_generation(  # noqa: C901
        self,
        *,
        generation_state: str,
        release_manifest_ref: Mapping[str, Any],
        source_refs: Sequence[Mapping[str, Any]],
        factor_source_object_refs: Sequence[Mapping[str, Any]],
        factor_policy_ref: Mapping[str, Any] | None,
        factor_evidence_refs: Sequence[Mapping[str, Any]],
        factor_active_set_ref: Mapping[str, Any] | None,
        factor_validation_attestation_ref: Mapping[str, Any] | None,
        mainline_ref: Mapping[str, Any] | None,
        research_refs: Sequence[Mapping[str, Any]],
        migration_receipt_ref: Mapping[str, Any] | None,
        migration_marker_ref: Mapping[str, Any] | None,
        skill_tree_sha256: str,
        automation_semantic_sha256: str,
        readiness_matrix_ref: Mapping[str, Any],
        emergency_controller_sha256: str | None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        """Assemble from already-stored exact refs; never publish caller artifacts."""

        if type(generation_state) is not str or generation_state not in GENERATION_STATES:
            raise SystemContractError("generation_state is invalid")
        timestamp = _require_timestamp(created_at or _utc_now(), label="created_at")
        skill_sha = _require_sha256(skill_tree_sha256, label="skill_tree_sha256")
        automation_sha = _require_sha256(
            automation_semantic_sha256, label="automation_semantic_sha256"
        )
        if generation_state == "SYSTEM_SUSPENDED":
            if emergency_controller_sha256 is not None:
                raise SystemContractError(
                    "suspended generation emergency_controller_sha256 must be null"
                )
            emergency_sha = None
        else:
            emergency_sha = _require_sha256(
                emergency_controller_sha256,
                label="emergency_controller_sha256",
            )

        release_ref, release_artifact = self._resolve_ref(
            release_manifest_ref,
            label="release_manifest_ref",
            expected_kinds={RELEASE_KIND},
        )
        if release_ref is None or release_artifact is None:
            raise SystemContractError("release manifest ref is absent")
        _validate_release(release_artifact)
        operational = generation_state == "OPERATIONAL"
        readiness_kind = OPERATIONAL_READINESS_KIND if operational else SUSPENDED_READINESS_KIND
        readiness_ref, readiness_artifact = self._resolve_ref(
            readiness_matrix_ref,
            label="readiness_matrix_ref",
            expected_kinds={readiness_kind},
        )
        if readiness_ref is None or readiness_artifact is None:
            raise SystemContractError("readiness matrix ref is absent")
        _validate_readiness(readiness_artifact)
        normalized_sources, source_artifacts = self._resolve_refs(
            source_refs,
            label="source_refs",
            minimum=1 if operational else 0,
            expected_kinds={"system.source_bundle"},
        )
        for source_artifact in source_artifacts:
            self._verify_source_bundle(source_artifact, require_sources=operational)
        if operational:
            self._validate_operational_sources(source_artifacts)
        normalized_factor_source_refs, _ = self._resolve_refs(
            factor_source_object_refs,
            label="factor_source_object_refs",
            minimum=1 if operational else 0,
            expected_kinds={"system.source_object"},
        )
        validation_attestation_ref, _ = self._resolve_ref(
            factor_validation_attestation_ref,
            label="factor_validation_attestation_ref",
            expected_kinds={"system.validation_attestation"},
            nullable=not operational,
        )
        policy_ref, factor_policy_artifact = self._resolve_ref(
            factor_policy_ref,
            label="factor_policy_ref",
            nullable=not operational,
        )
        evidence_refs, factor_evidence_artifacts = self._resolve_refs(
            factor_evidence_refs,
            label="factor_evidence_refs",
            minimum=1 if operational else 0,
        )
        active_set_ref, factor_active_set_artifact = self._resolve_ref(
            factor_active_set_ref,
            label="factor_active_set_ref",
            nullable=not operational,
        )
        normalized_mainline_ref, mainline_artifact = self._resolve_ref(
            mainline_ref,
            label="mainline_ref",
            expected_kinds={"mainline_candidate"},
            nullable=True,
        )
        normalized_research_refs, _ = self._resolve_refs(
            research_refs, label="research_refs", minimum=0
        )
        if migration_receipt_ref is not None:
            raise SystemContractError("migration_receipt_ref must be the mandatory null tombstone")
        if migration_marker_ref is not None:
            raise SystemContractError("migration_marker_ref must be the mandatory null tombstone")
        migration_ref = None
        marker_ref = None
        if operational:
            for label, ref in (
                ("factor_policy_ref", policy_ref),
                ("factor_active_set_ref", active_set_ref),
            ):
                if ref is None:
                    raise SystemContractError(f"{label} is absent")
        elif any(
            (
                normalized_sources,
                normalized_factor_source_refs,
                policy_ref,
                evidence_refs,
                active_set_ref,
                normalized_mainline_ref,
                normalized_research_refs,
                migration_ref,
                marker_ref,
                validation_attestation_ref,
            )
        ):
            raise SystemContractError(
                "suspended generation must use the minimal null/empty closure"
            )

        readiness_payload = readiness_artifact["payload"]
        if generation_state == "SYSTEM_SUSPENDED" and (
            readiness_payload["factor_state"] != "SUSPENDED"
            or readiness_payload["mainline_state"] != "SUSPENDED"
            or readiness_payload["investment_state"] != "SUSPENDED"
            or readiness_payload["factor_status_ref"] is not None
            or readiness_payload["mainline_candidate_ref"] is not None
        ):
            raise SystemContractError("suspended generation readiness is not minimal")

        factor_status_artifact = None
        factor_status_ref = readiness_payload["factor_status_ref"]
        if factor_status_ref is not None:
            _, factor_status_artifact = self._resolve_ref(
                factor_status_ref,
                label="readiness.factor_status_ref",
                expected_kinds={"factor.status"},
            )
        candidate_ref = readiness_payload["mainline_candidate_ref"]
        if candidate_ref is not None:
            candidate_binding, _ = self._resolve_ref(
                candidate_ref,
                label="readiness.mainline_candidate_ref",
                expected_kinds={"mainline_candidate"},
            )
            if normalized_mainline_ref is None or candidate_binding != normalized_mainline_ref:
                raise SystemContractError("readiness/mainline candidate binding mismatch")
        if operational:
            if (
                policy_ref is None
                or factor_policy_artifact is None
                or active_set_ref is None
                or factor_active_set_artifact is None
                or factor_status_artifact is None
                or validation_attestation_ref is None
            ):
                raise SystemContractError("operational authority closure is invalid")
            self._validate_factor_generation_closure(
                policy_ref=policy_ref,
                policy=factor_policy_artifact,
                evidence_refs=evidence_refs,
                evidence=factor_evidence_artifacts,
                active_set_ref=active_set_ref,
                active_set=factor_active_set_artifact,
                factor_source_object_refs=normalized_factor_source_refs,
                validation_attestation_ref=validation_attestation_ref,
                validation_attestation=self.get_object(validation_attestation_ref),
                readiness=readiness_artifact,
                factor_status=factor_status_artifact,
                verification_level="stat",
            )
        del mainline_artifact

        catalog_sha = self.put_contract_catalog()
        assembly_id = generation_assembly_identity(
            generation_state=generation_state,
            release_id=release_ref["artifact_id"],
            readiness_id=readiness_ref["artifact_id"],
            created_at=timestamp,
        )

        manifest = seal_artifact(
            MANIFEST_KIND,
            {
                "assembly_id": assembly_id,
                "generation_state": generation_state,
                "contract_catalog_sha256": catalog_sha,
                "release_manifest_ref": release_ref,
                "source_refs": normalized_sources,
                "factor_source_object_refs": normalized_factor_source_refs,
                "factor_policy_ref": policy_ref,
                "factor_evidence_refs": evidence_refs,
                "factor_active_set_ref": active_set_ref,
                "factor_validation_attestation_ref": validation_attestation_ref,
                "mainline_ref": normalized_mainline_ref,
                "research_refs": normalized_research_refs,
                "migration_receipt_ref": migration_ref,
                "migration_marker_ref": marker_ref,
                "skill_tree_sha256": skill_sha,
                "automation_semantic_sha256": automation_sha,
                "readiness_matrix_ref": readiness_ref,
                "emergency_controller_sha256": emergency_sha,
            },
            created_at=timestamp,
        )
        if "generation_id" in manifest["payload"]:
            raise SystemContractError("generation_id must not occur in manifest payload")
        generation_id = manifest["semantic_sha256"]
        manifest_ref = self.put_object(manifest)
        manifest_raw = canonical_json_bytes(manifest)
        stored = self._storage.write_generation_manifest(generation_id, manifest_raw)
        if stored.byte_sha256 != manifest_ref["byte_sha256"]:
            raise SystemStorageError("generation manifest byte binding mismatch")
        return self._verify_generation(generation_id, validation_level="stat")

    def _verify_generation(  # noqa: C901
        self,
        generation_id: str,
        *,
        deployed_release_ref: Mapping[str, Any] | None = None,
        validation_level: str,
    ) -> dict[str, Any]:
        """Verify one immutable generation and resolve its complete object closure."""

        if validation_level not in {"stat", "full"}:
            raise SystemContractError("generation validation level is invalid")

        normalized_id = _require_sha256(generation_id, label="generation_id")
        stored = self._storage.read(GENERATIONS_ROOT / normalized_id / "manifest.json")
        manifest = self._validated_artifact(stored.data)
        if manifest["kind"] != MANIFEST_KIND:
            raise SystemContractError("generation manifest has the wrong kind")
        if manifest["semantic_sha256"] != normalized_id:
            raise SystemContractError("generation_id is not the manifest semantic SHA-256")
        payload = manifest["payload"]
        if type(payload) is not dict or "generation_id" in payload:
            raise SystemContractError("generation manifest payload is invalid")

        manifest_ref = object_ref_for_artifact(manifest)
        manifest_object = self.get_object(manifest_ref)
        if manifest_object != manifest:
            raise SystemContractError("generation manifest object binding mismatch")

        generation_state = payload.get("generation_state")
        if type(generation_state) is not str or generation_state not in GENERATION_STATES:
            raise SystemContractError("manifest generation_state is invalid")
        catalog_sha = _require_sha256(
            payload.get("contract_catalog_sha256"),
            label="manifest.contract_catalog_sha256",
        )
        catalog = self.read_contract_catalog(catalog_sha)
        release_ref, release = self._resolve_ref(
            payload.get("release_manifest_ref"),
            label="manifest.release_manifest_ref",
            expected_kinds={RELEASE_KIND},
        )
        if release_ref is None or release is None:
            raise SystemContractError("manifest release is absent")
        _validate_release(release)
        deployed_release_verified = False
        if deployed_release_ref is not None:
            deployed = validate_object_ref(deployed_release_ref, label="deployed_release_ref")
            if deployed != release_ref:
                raise SystemContractError("deployed release identity does not match generation")
            _verify_installed_release(release)
            deployed_release_verified = True
        source_refs, sources = self._resolve_refs(
            payload.get("source_refs"),
            label="manifest.source_refs",
            minimum=1 if generation_state == "OPERATIONAL" else 0,
            expected_kinds={"system.source_bundle"},
        )
        for source in sources:
            self._verify_source_bundle(source, require_sources=generation_state == "OPERATIONAL")
        if generation_state == "OPERATIONAL":
            self._validate_operational_sources(sources)
        factor_source_object_refs, factor_source_objects = self._resolve_refs(
            payload.get("factor_source_object_refs"),
            label="manifest.factor_source_object_refs",
            minimum=1 if generation_state == "OPERATIONAL" else 0,
            expected_kinds={"system.source_object"},
        )
        factor_validation_attestation_ref, factor_validation_attestation = self._resolve_ref(
            payload.get("factor_validation_attestation_ref"),
            label="manifest.factor_validation_attestation_ref",
            expected_kinds={"system.validation_attestation"},
            nullable=generation_state == "SYSTEM_SUSPENDED",
        )
        policy_ref, factor_policy = self._resolve_ref(
            payload.get("factor_policy_ref"),
            label="manifest.factor_policy_ref",
            nullable=generation_state == "SYSTEM_SUSPENDED",
        )
        evidence_refs, factor_evidence = self._resolve_refs(
            payload.get("factor_evidence_refs"),
            label="manifest.factor_evidence_refs",
            minimum=1 if generation_state == "OPERATIONAL" else 0,
        )
        active_set_ref, factor_active_set = self._resolve_ref(
            payload.get("factor_active_set_ref"),
            label="manifest.factor_active_set_ref",
            nullable=generation_state == "SYSTEM_SUSPENDED",
        )
        mainline_ref, mainline = self._resolve_ref(
            payload.get("mainline_ref"),
            label="manifest.mainline_ref",
            expected_kinds={"mainline_candidate"},
            nullable=True,
        )
        research_refs, research = self._resolve_refs(
            payload.get("research_refs"), label="manifest.research_refs", minimum=0
        )
        if payload.get("migration_receipt_ref") is not None:
            raise SystemContractError(
                "manifest migration_receipt_ref must be the mandatory null tombstone"
            )
        if payload.get("migration_marker_ref") is not None:
            raise SystemContractError(
                "manifest migration_marker_ref must be the mandatory null tombstone"
            )
        migration_receipt_ref = None
        migration_receipt = None
        migration_marker_ref = None
        migration_marker = None
        readiness_kind = (
            OPERATIONAL_READINESS_KIND
            if generation_state == "OPERATIONAL"
            else SUSPENDED_READINESS_KIND
        )
        readiness_ref, readiness = self._resolve_ref(
            payload.get("readiness_matrix_ref"),
            label="manifest.readiness_matrix_ref",
            expected_kinds={readiness_kind},
        )
        if readiness_ref is None or readiness is None:
            raise SystemContractError("manifest readiness is absent")
        _validate_readiness(readiness)
        _require_sha256(payload.get("skill_tree_sha256"), label="skill_tree_sha256")
        _require_sha256(
            payload.get("automation_semantic_sha256"),
            label="automation_semantic_sha256",
        )
        emergency_sha = payload.get("emergency_controller_sha256")
        emergency_controller = None
        if generation_state == "SYSTEM_SUSPENDED":
            if emergency_sha is not None:
                raise SystemContractError("suspended generation emergency SHA must be null")
        else:
            emergency_sha = _require_sha256(
                emergency_sha,
                label="emergency_controller_sha256",
            )
            from .controller import verify_emergency_controller

            emergency_controller = verify_emergency_controller(
                self,
                expected_sha256=emergency_sha,
            )

        if generation_state == "OPERATIONAL":
            for label, value in (
                ("factor_policy", factor_policy),
                ("factor_active_set", factor_active_set),
            ):
                if value is None:
                    raise SystemContractError(f"generation {label} binding is absent")
        elif any(
            (
                source_refs,
                factor_source_object_refs,
                policy_ref,
                evidence_refs,
                active_set_ref,
                mainline_ref,
                research_refs,
                migration_receipt_ref,
                migration_marker_ref,
                factor_validation_attestation_ref,
            )
        ):
            raise SystemContractError("suspended generation manifest is not a minimal closure")

        readiness_payload = readiness["payload"]
        factor_status_ref = readiness_payload["factor_status_ref"]
        factor_status = None
        if factor_status_ref is not None:
            normalized_factor_status_ref, factor_status = self._resolve_ref(
                factor_status_ref,
                label="readiness.factor_status_ref",
                expected_kinds={"factor.status"},
            )
            factor_status_ref = normalized_factor_status_ref
        candidate_ref = readiness_payload["mainline_candidate_ref"]
        if candidate_ref is not None:
            candidate_binding, _ = self._resolve_ref(
                candidate_ref,
                label="readiness.mainline_candidate_ref",
                expected_kinds={"mainline_candidate"},
            )
            if mainline_ref is None or candidate_binding != mainline_ref:
                raise SystemContractError("readiness/mainline candidate binding mismatch")
        factor_validation_receipt_ref = None
        factor_validation_receipt = None
        factor_validation_resolution = None
        if generation_state == "OPERATIONAL":
            if (
                policy_ref is None
                or factor_policy is None
                or active_set_ref is None
                or factor_active_set is None
                or factor_status is None
                or factor_validation_attestation_ref is None
                or factor_validation_attestation is None
            ):
                raise SystemContractError("operational Factor closure is incomplete")
            (
                factor_validation_receipt_ref,
                factor_validation_receipt,
                factor_validation_resolution,
            ) = self._validate_factor_generation_closure(
                policy_ref=policy_ref,
                policy=factor_policy,
                evidence_refs=evidence_refs,
                evidence=factor_evidence,
                active_set_ref=active_set_ref,
                active_set=factor_active_set,
                factor_source_object_refs=factor_source_object_refs,
                validation_attestation_ref=factor_validation_attestation_ref,
                validation_attestation=factor_validation_attestation,
                readiness=readiness,
                factor_status=factor_status,
                verification_level=validation_level,
            )
        if generation_state == "SYSTEM_SUSPENDED" and (
            readiness_payload["factor_state"] != "SUSPENDED"
            or readiness_payload["mainline_state"] != "SUSPENDED"
            or readiness_payload["investment_state"] != "SUSPENDED"
            or factor_status_ref is not None
            or candidate_ref is not None
        ):
            raise SystemContractError("suspended generation readiness is not minimal")

        return {
            "verified": True,
            "generation_state": generation_state,
            "generation_id": normalized_id,
            "manifest": manifest,
            "manifest_sha256": stored.byte_sha256,
            "manifest_byte_sha256": stored.byte_sha256,
            "manifest_ref": manifest_ref,
            "contract_catalog": catalog,
            "release": release,
            "sources": sources,
            "factor_source_object_refs": factor_source_object_refs,
            "factor_source_objects": factor_source_objects,
            "factor_policy": factor_policy,
            "factor_evidence": factor_evidence,
            "factor_active_set": factor_active_set,
            "factor_validation_attestation_ref": factor_validation_attestation_ref,
            "factor_validation_attestation": factor_validation_attestation,
            "factor_validation_resolution": factor_validation_resolution,
            "factor_contextual_result": (
                factor_validation_resolution["contextual_result"]
                if factor_validation_resolution is not None
                else None
            ),
            "factor_validation_completion": (
                factor_validation_resolution["validation_completion"]
                if factor_validation_resolution is not None
                else None
            ),
            "factor_source_verification_snapshot": (
                factor_validation_resolution["source_verification_snapshot"]
                if factor_validation_resolution is not None
                else None
            ),
            "factor_status_ref": factor_status_ref,
            "factor_status": factor_status,
            "factor_validation_receipt_ref": factor_validation_receipt_ref,
            "factor_validation_receipt": factor_validation_receipt,
            "mainline": mainline,
            "research": research,
            "migration_receipt": migration_receipt,
            "migration_marker": migration_marker,
            "readiness": readiness,
            "emergency_controller": emergency_controller,
            "deployed_release_verified": deployed_release_verified,
        }

    def verify_generation(
        self,
        generation_id: str,
        *,
        deployed_release_ref: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Deep-verify a generation, including full Factor source byte hashes."""

        verified = self._verify_generation(
            generation_id,
            deployed_release_ref=deployed_release_ref,
            validation_level="full",
        )
        research = verified.get("research")
        if (
            verified.get("generation_state") == "OPERATIONAL"
            and type(research) is list
            and len(research) == 1
            and research[0].get("kind") == "system.production_bootstrap_receipt"
        ):
            from quant_investor.factors.governance.production import (
                validate_production_bootstrap_generation_closure,
            )

            receipt = validate_production_bootstrap_generation_closure(
                store=self,
                verified_generation=verified,
                deployed_release_ref=(
                    deployed_release_ref or verified["manifest"]["payload"]["release_manifest_ref"]
                ),
                validation_mode="PRE_CAS_CURRENT",
            )
            limitations = list(receipt["payload"]["calendar_source_limitations"])
            from .fundamental_advisory import (
                validate_fundamental_advisory,
                validate_fundamental_veto_subject,
            )

            fundamental_veto_subject = validate_fundamental_veto_subject(
                self.get_object(receipt["payload"]["fundamental_veto_subject_ref"])
            )
            fundamental_advisory = validate_fundamental_advisory(
                self.get_object(receipt["payload"]["fundamental_advisory_ref"])
            )
            verified = {
                **verified,
                "calendar_authority_route": (
                    "TRUSTED_PROVIDER_DEGRADED" if limitations else "EXCHANGE_OFFICIAL"
                ),
                "calendar_authority_confidence": ("DEGRADED" if limitations else "OFFICIAL"),
                "calendar_source_limitations": limitations,
                "fundamental_veto_subject": fundamental_veto_subject,
                "fundamental_advisory": fundamental_advisory,
            }
        return verified

    @staticmethod
    def _validate_pointer(document: Mapping[str, Any] | bytes) -> dict[str, Any]:
        try:
            if type(document) is bytes:
                pointer = parse_canonical_json_bytes(document, label="active pointer")
            elif isinstance(document, Mapping):
                pointer = dict(document)
            else:  # pragma: no cover - the public annotation is exhaustive
                raise SystemContractError("active pointer must be an object")
        except ContractError as exc:
            raise SystemContractError("active pointer is not canonical JSON") from exc
        if type(pointer) is not dict or set(pointer) != set(POINTER_FIELDS):
            raise SystemContractError("active pointer fields are not exact")
        _require_sha256(pointer.get("generation_id"), label="pointer.generation_id")
        _require_sha256(pointer.get("manifest_sha256"), label="pointer.manifest_sha256")
        _require_pointer_sha256(
            pointer.get("previous_pointer_sha256"),
            label="pointer.previous_pointer_sha256",
        )
        _require_timestamp(pointer.get("activated_at"), label="pointer.activated_at")
        _require_text(pointer.get("os_actor"), label="pointer.os_actor")
        canonical_json_bytes(pointer)
        return pointer

    def activate_generation(
        self,
        generation_id: str,
        *,
        expected_pointer_sha256: str,
        activated_at: str | None = None,
        os_actor: str | None = None,
        deployed_release_ref: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """CAS a later verified generation; first activation uses its own ceremony."""

        expected = _require_pointer_sha256(expected_pointer_sha256, label="expected_pointer_sha256")
        if expected == EMPTY_POINTER_SHA256:
            raise SystemActivationAuthorizationError(
                "initial activation requires detached receipt and authorization"
            )
        self.verify_migration_completion()
        verified = self.verify_generation(generation_id, deployed_release_ref=deployed_release_ref)
        if verified["generation_state"] == "SYSTEM_SUSPENDED":
            raise SystemActivationAuthorizationError(
                "suspension requires an exact prebuilt pointer"
            )
        if verified["generation_state"] == "OPERATIONAL" and deployed_release_ref is None:
            raise SystemPreconditionError(
                "operational activation requires deployed release identity"
            )
        timestamp = _require_timestamp(activated_at or _utc_now(), label="activated_at")
        actor = _require_text(
            os_actor or "uid:%d" % self._storage.workspace_root.stat().st_uid,
            label="os_actor",
        )
        pointer = {
            "generation_id": verified["generation_id"],
            "manifest_sha256": verified["manifest_sha256"],
            "previous_pointer_sha256": expected,
            "activated_at": timestamp,
            "os_actor": actor,
        }
        raw = canonical_json_bytes(pointer)
        current = self._storage.read_optional(ACTIVE_POINTER_PATH)
        if current is not None:
            self._validate_pointer(current.data)
        stored = self._storage._compare_and_swap_active_authorized_nonempty(
            raw, expected_sha256=expected
        )
        replay = self._validate_pointer(stored.data)
        if replay != pointer:
            raise SystemStorageError("active pointer replay mismatch")
        active = self.read_active(deployed_release_ref=deployed_release_ref)
        if active is None:  # pragma: no cover - successful CAS created the pointer
            raise SystemStorageError("active pointer disappeared after CAS")
        return active

    def activate_suspended_generation(  # noqa: C901
        self,
        *,
        target_active_pointer_raw: bytes,
        expected_pointer_sha256: str,
    ) -> dict[str, Any]:
        """CAS exact presealed bytes to a verified suspended generation.

        This emergency lane deliberately does not require the permanent marker,
        so it can contain an exact initial-pointer-only crash.  It never grants
        Factor authority and it never synthesizes actor or time fields.
        """

        expected = _require_pointer_sha256(expected_pointer_sha256, label="expected_pointer_sha256")
        if expected == EMPTY_POINTER_SHA256:
            raise SystemActivationAuthorizationError(
                "emergency suspension requires a non-empty preimage"
            )
        if type(target_active_pointer_raw) is not bytes or not target_active_pointer_raw:
            raise SystemActivationAuthorizationError(
                "suspension pointer must preserve exact non-empty bytes"
            )
        pointer = self._validate_pointer(target_active_pointer_raw)
        if pointer["previous_pointer_sha256"] != expected:
            raise SystemActivationAuthorizationError("suspension pointer preimage binding mismatch")
        current_uid = os.geteuid()
        if pointer["os_actor"] != f"uid:{current_uid}:emergency-suspend":
            raise SystemActivationAuthorizationError("suspension pointer EUID differs")
        activated = datetime.strptime(
            _require_timestamp(pointer["activated_at"], label="activated_at"),
            "%Y-%m-%dT%H:%M:%SZ",
        ).replace(tzinfo=timezone.utc)
        if activated > datetime.now(timezone.utc):
            raise SystemActivationAuthorizationError(
                "suspension pointer activation time is in the future"
            )
        current = self._storage.read_optional(ACTIVE_POINTER_PATH)
        if current is None:
            raise SystemCASMismatch(expected, EMPTY_POINTER_SHA256)
        self._validate_pointer(current.data)
        if current.byte_sha256 != expected:
            raise SystemCASMismatch(expected, current.byte_sha256)
        anchor = self._resolve_historical_emergency_anchor()
        controller = anchor["controller"]
        if (
            pointer["generation_id"] != controller["generation_id"]
            or pointer["manifest_sha256"] != controller["manifest_sha256"]
        ):
            raise SystemActivationAuthorizationError(
                "suspension target differs from the initial controller"
            )
        verified = self._verify_historical_suspended_generation(
            pointer["generation_id"],
            expected_manifest_sha256=controller["manifest_sha256"],
            expected_manifest_contract_sha256=controller["manifest_contract_sha256"],
            expected_manifest_payload_fields=controller["manifest_payload_fields"],
        )
        if verified["generation_state"] != "SYSTEM_SUSPENDED":
            raise SystemActivationAuthorizationError("emergency target must be SYSTEM_SUSPENDED")
        if verified["manifest_sha256"] != pointer["manifest_sha256"]:
            raise SystemActivationAuthorizationError("suspension pointer manifest binding mismatch")
        stored = self._storage._compare_and_swap_active_authorized_nonempty(
            target_active_pointer_raw, expected_sha256=expected
        )
        if stored.data != target_active_pointer_raw:
            raise SystemStorageError("suspension pointer exact-byte readback mismatch")
        return {
            "generation_id": verified["generation_id"],
            "generation_state": "SYSTEM_SUSPENDED",
            "pointer": pointer,
            "pointer_byte_sha256": stored.byte_sha256,
            "factor_authority": "BLOCKED",
            "migration_marker_required_for_factor_active": True,
        }

    def activate_initial_generation(  # noqa: C901
        self,
        *,
        target_active_pointer_raw: bytes,
        migration_receipt_raw: bytes,
        final_cutover_authorization_raw: bytes,
        activation_authorization_raw: bytes,
        deployed_release_ref: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Perform or exactly recover the sole authorized ``EMPTY`` activation."""

        if any(
            type(raw) is not bytes or not raw
            for raw in (
                target_active_pointer_raw,
                migration_receipt_raw,
                final_cutover_authorization_raw,
                activation_authorization_raw,
            )
        ):
            raise SystemActivationAuthorizationError(
                "initial activation inputs must preserve exact non-empty bytes"
            )
        pointer = self._validate_pointer(target_active_pointer_raw)
        if pointer["previous_pointer_sha256"] != EMPTY_POINTER_SHA256:
            raise SystemActivationAuthorizationError(
                "initial activation preimage must be literal EMPTY"
            )
        try:
            from quant_investor.migration.authority import (
                validate_current_production_authorization_target,
                validate_final_cutover_authorization_closure,
            )

            final_authorization = validate_final_cutover_authorization_closure(
                final_cutover_authorization_raw,
                repository_root=self.workspace_root,
                object_resolver=self.get_object,
                deployed_release_ref=deployed_release_ref,
                validation_mode="PRE_CAS_CURRENT",
            )
        except SystemError:
            raise
        try:
            target = validate_current_production_authorization_target(
                system_store=self,
                deployed_release_ref=deployed_release_ref,
                generation_id=pointer["generation_id"],
                expected_manifest_ref=final_authorization["payload"][
                    "production_generation_manifest_ref"
                ],
                expected_receipt_ref=final_authorization["payload"][
                    "production_bootstrap_receipt_ref"
                ],
            )
            verified = target["verified_generation"]
            production_receipt = target["production_receipt"]
        except SystemError as exc:
            raise SystemActivationAuthorizationError(
                "initial authorization lacks valid production target closure"
            ) from exc
        if verified["manifest_sha256"] != pointer["manifest_sha256"]:
            raise SystemActivationAuthorizationError("pointer manifest binding mismatch")
        try:
            from .activation import (
                build_prepared_activation_transaction,
                validate_activation_authorization,
            )

            calendar_fields = (
                "calendar_authority_policy_ref",
                "calendar_compilation_ref",
                "calendar_capability_ref",
                "calendar_capture_execution_ref",
                "calendar_authorization_basis",
                "calendar_source_limitations",
                "bootstrap_admission_intent_sha256",
                "factor_dependency_sha256",
                "fundamental_veto_subject_ref",
                "fundamental_operator_veto_ref",
                "fundamental_advisory_ref",
            )
            if (
                any(
                    final_authorization["payload"][field] != production_receipt["payload"][field]
                    for field in calendar_fields
                )
                or final_authorization["payload"]["production_generation_manifest_ref"]
                != target["generation_manifest_ref"]
                or final_authorization["payload"]["production_bootstrap_receipt_ref"]
                != target["production_receipt_ref"]
                or final_authorization["payload"]["fundamental_advisory_authorized"] is not True
            ):
                raise SystemActivationAuthorizationError(
                    "final authorization target binding differs from production receipt"
                )

            authorization, marker = validate_activation_authorization(
                activation_authorization_raw,
                final_cutover_authorization=final_cutover_authorization_raw,
                migration_receipt=migration_receipt_raw,
                target_active_pointer=target_active_pointer_raw,
                target_generation_manifest=verified["manifest"],
                deployed_release_ref=deployed_release_ref,
                current_uid=os.geteuid(),
            )
            prepared = build_prepared_activation_transaction(authorization)
        except SystemError:
            raise
        except Exception as exc:
            raise SystemActivationAuthorizationError(
                "initial activation authorization closure failed"
            ) from exc
        receipt = self._validated_artifact(migration_receipt_raw)
        if receipt["kind"] != "system.migration.receipt":
            raise SystemActivationAuthorizationError("detached receipt kind is invalid")
        receipt_ref = object_ref_for_artifact(receipt)
        authorization_ref = object_ref_for_artifact(authorization)
        final_authorization_ref = object_ref_for_artifact(final_authorization)
        prepared_ref = object_ref_for_artifact(prepared)
        pointer_sha = hashlib.sha256(target_active_pointer_raw).hexdigest()

        def validate_under_lock() -> None:
            validate_final_cutover_authorization_closure(
                final_cutover_authorization_raw,
                repository_root=self.workspace_root,
                object_resolver=self.get_object,
                deployed_release_ref=deployed_release_ref,
                validation_mode="PRE_CAS_CURRENT",
            )
            locked_target = validate_current_production_authorization_target(
                system_store=self,
                deployed_release_ref=deployed_release_ref,
                generation_id=pointer["generation_id"],
                expected_manifest_ref=final_authorization["payload"][
                    "production_generation_manifest_ref"
                ],
                expected_receipt_ref=final_authorization["payload"][
                    "production_bootstrap_receipt_ref"
                ],
            )
            locked_generation = locked_target["verified_generation"]
            if locked_generation["manifest_sha256"] != pointer["manifest_sha256"]:
                raise SystemActivationAuthorizationError(
                    "pointer manifest binding drifted under activation lock"
                )
            locked_production_receipt = locked_target["production_receipt"]
            if (
                any(
                    final_authorization["payload"][field]
                    != locked_production_receipt["payload"][field]
                    for field in calendar_fields
                )
                or final_authorization["payload"]["fundamental_advisory_authorized"] is not True
            ):
                raise SystemActivationAuthorizationError(
                    "calendar authorization drifted under activation lock"
                )

        result = self._storage._commit_initial_activation(
            transaction=_PreparedInitialActivationWrite(
                pointer_raw=target_active_pointer_raw,
                receipt_raw=migration_receipt_raw,
                final_authorization_raw=final_cutover_authorization_raw,
                activation_authorization_raw=activation_authorization_raw,
                prepared_raw=canonical_json_bytes(prepared),
                marker_raw=canonical_json_bytes(marker),
            ),
            lock_validator=validate_under_lock,
        )
        active = self.read_active(deployed_release_ref=deployed_release_ref)
        if active is None:
            raise SystemStorageError("active pointer disappeared after initial activation")
        completion = active["migration_completion"]
        return {
            **active,
            "activation": {
                "authorization_ref": authorization_ref,
                "final_cutover_authorization_ref": final_authorization_ref,
                "cas_performed": bool(result["cas_performed"]),
                "marker_byte_sha256": completion["marker_byte_sha256"],
                "marker_semantic_sha256": completion["marker"]["semantic_sha256"],
                "migration_receipt_ref": receipt_ref,
                "pointer_byte_sha256": pointer_sha,
                "prepared_transaction_ref": prepared_ref,
            },
        }

    def verify_migration_completion(  # noqa: C901
        self, chain: Sequence[Mapping[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Verify the permanent marker against the anchored initial pointer."""

        pointer_chain = list(chain) if chain is not None else self._pointer_chain()
        if not pointer_chain:
            raise SystemMigrationMarkerAbsent("active pointer is absent")
        initial_row = pointer_chain[-1]
        initial_pointer = initial_row.get("pointer")
        initial_pointer_sha = initial_row.get("pointer_byte_sha256")
        if (
            type(initial_pointer) is not dict
            or type(initial_pointer_sha) is not str
            or initial_pointer.get("previous_pointer_sha256") != EMPTY_POINTER_SHA256
        ):
            raise SystemMigrationClosureError("initial pointer ancestry is invalid")
        marker_stored = self._storage.read_optional(MIGRATION_MARKER_PATH)
        if marker_stored is None:
            raise SystemMigrationMarkerAbsent("permanent migration marker is absent")
        authorization_stored = self._storage.read_optional(
            ACTIVATION_AUTHORIZATIONS_ROOT / f"{initial_pointer_sha}.json"
        )
        if authorization_stored is None:
            raise SystemMigrationClosureError("initial authorization index is absent")
        prepared_stored = self._storage.read_optional(
            ACTIVATION_TRANSACTIONS_ROOT / f"{initial_pointer_sha}.json"
        )
        if prepared_stored is None:
            raise SystemMigrationClosureError("prepared transaction index is absent")
        final_authorization_stored = self._storage.read_optional(
            FINAL_CUTOVER_AUTHORIZATIONS_ROOT / f"{initial_pointer_sha}.json"
        )
        if final_authorization_stored is None:
            raise SystemMigrationClosureError("final cutover authorization index is absent")
        try:
            from quant_investor.migration.errors import UnifiedCutoverError
            from quant_investor.migration.authority import (
                _git_blob,
                validate_final_cutover_authorization_closure,
            )
            from .historical_activation import (
                validate_initial_activation_authorization,
                validate_initial_activation_bundle,
                validate_initial_migration_receipt,
                validate_initial_permanent_marker,
            )

            generation = self._verify_historical_initial_generation(
                initial_pointer["generation_id"]
            )
            dispatch = generation["historical_contract_dispatch"]
            marker = validate_initial_permanent_marker(marker_stored.data)
            receipt_ref = validate_frozen_object_ref(
                marker["payload"]["migration_receipt_ref"],
                label="marker.migration_receipt_ref",
                dispatch=dispatch,
            )
            receipt = validate_initial_migration_receipt(
                self._historical_get_object(receipt_ref, dispatch=dispatch)
            )
            authorization = validate_initial_activation_authorization(authorization_stored.data)
            deployed_ref = validate_frozen_object_ref(
                authorization["payload"]["deployed_release_ref"],
                label="authorization.deployed_release_ref",
                dispatch=dispatch,
            )
            if generation["manifest"]["payload"]["release_manifest_ref"] != deployed_ref:
                raise SystemMigrationClosureError(
                    "initial generation/deployed release binding is invalid"
                )

            def historical_resolver(ref: Mapping[str, object]) -> Mapping[str, object]:
                return self._historical_get_object(ref, dispatch=dispatch)

            final_authorization = validate_final_cutover_authorization_closure(
                final_authorization_stored.data,
                repository_root=self.workspace_root,
                object_resolver=historical_resolver,
                deployed_release_ref=deployed_ref,
                validation_mode="HISTORICAL",
            )
            from .historical_activation import (
                validate_initial_fundamental_advisory,
                validate_initial_fundamental_veto_subject,
            )

            fundamental_veto_subject = validate_initial_fundamental_veto_subject(
                self._historical_get_object(
                    final_authorization["payload"]["fundamental_veto_subject_ref"],
                    dispatch=dispatch,
                )
            )
            fundamental_advisory = validate_initial_fundamental_advisory(
                self._historical_get_object(
                    final_authorization["payload"]["fundamental_advisory_ref"],
                    dispatch=dispatch,
                )
            )
            generation = {
                **generation,
                "fundamental_veto_subject": fundamental_veto_subject,
                "fundamental_advisory": fundamental_advisory,
            }
            assembler_path = validate_initial_assembler_module_path(INITIAL_ASSEMBLER_MODULE_PATH)
            _mode, _blob, assembler_raw = _git_blob(
                self.workspace_root,
                final_authorization["payload"]["final_integration_commit"],
                assembler_path,
            )
            historical_assembler_sha256 = hashlib.sha256(assembler_raw).hexdigest()
            from quant_investor.factors.governance.production import (
                validate_production_bootstrap_generation_closure,
            )

            try:
                production_receipt = validate_production_bootstrap_generation_closure(
                    store=self,
                    verified_generation=generation,
                    deployed_release_ref=deployed_ref,
                    validation_mode="HISTORICAL",
                    historical_assembler_sha256=historical_assembler_sha256,
                )
                for field in (
                    "calendar_authority_policy_ref",
                    "calendar_compilation_ref",
                    "calendar_capability_ref",
                    "calendar_capture_execution_ref",
                    "calendar_authorization_basis",
                    "calendar_source_limitations",
                ):
                    if field in production_receipt["payload"] and (
                        final_authorization["payload"].get(field)
                        != production_receipt["payload"][field]
                    ):
                        raise SystemMigrationClosureError(
                            "historical calendar authorization binding differs"
                        )
            except SystemError as exc:
                raise SystemMigrationClosureError(
                    "initial generation production bootstrap closure is invalid"
                ) from exc
            bundle = validate_initial_activation_bundle(
                final_authorization=final_authorization_stored.data,
                activation_authorization=authorization_stored.data,
                prepared_transaction=prepared_stored.data,
                migration_receipt=receipt,
                permanent_marker=marker_stored.data,
                active_pointer=initial_pointer,
                generation_manifest=generation["manifest"],
                deployed_release_ref=deployed_ref,
                current_uid=os.geteuid(),
            )
            validated_authorization = bundle["activation_authorization"]
            prepared = bundle["prepared_transaction"]
        except (SystemError, UnifiedCutoverError) as exc:
            if isinstance(exc, SystemMigrationClosureError):
                raise
            raise SystemMigrationClosureError("permanent migration closure is invalid") from exc
        if hashlib.sha256(canonical_json_bytes(validated_authorization)).hexdigest() != (
            hashlib.sha256(authorization_stored.data).hexdigest()
        ):
            raise SystemMigrationClosureError("authorization exact-byte identity mismatch")
        return {
            "authorization": validated_authorization,
            "authorization_byte_sha256": authorization_stored.byte_sha256,
            "initial_pointer": dict(initial_pointer),
            "initial_pointer_byte_sha256": initial_pointer_sha,
            "initial_generation": {
                key: value
                for key, value in generation.items()
                if key != "historical_contract_dispatch"
            },
            "marker": marker,
            "marker_byte_sha256": marker_stored.byte_sha256,
            "migration_receipt": receipt,
            "migration_receipt_ref": receipt_ref,
            "prepared_transaction": prepared,
            "prepared_transaction_byte_sha256": prepared_stored.byte_sha256,
            "final_cutover_authorization": final_authorization,
            "final_cutover_authorization_byte_sha256": (final_authorization_stored.byte_sha256),
        }

    def read_active(  # noqa: C901 - current-release verification with historical fallback
        self, *, deployed_release_ref: Mapping[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Return the fully resolved active generation, or ``None`` if uninitialized."""

        chain = self._pointer_chain()
        if not chain:
            return None
        completion = self.verify_migration_completion(chain)
        current = chain[0]
        pointer = current["pointer"]

        if current["pointer_byte_sha256"] == completion["initial_pointer_byte_sha256"]:
            anchored_release_ref = validate_frozen_object_ref(
                completion["authorization"]["payload"]["deployed_release_ref"],
                label="initial authorization deployed release ref",
            )
            if deployed_release_ref is not None:
                requested_release_ref = validate_frozen_object_ref(
                    deployed_release_ref,
                    label="deployed_release_ref",
                )
                if requested_release_ref != anchored_release_ref:
                    raise SystemContractError(
                        "requested release differs from initial authorization"
                    )
            else:
                requested_release_ref = anchored_release_ref
            try:
                verified = self._verify_generation(
                    pointer["generation_id"],
                    deployed_release_ref=requested_release_ref,
                    validation_level="stat",
                )
                from quant_investor.factors.governance.production import (
                    validate_production_bootstrap_generation_closure,
                )

                production_receipt = validate_production_bootstrap_generation_closure(
                    store=self,
                    verified_generation=verified,
                    deployed_release_ref=requested_release_ref,
                    validation_mode="PRE_CAS_CURRENT",
                )
                from .fundamental_advisory import (
                    validate_fundamental_advisory,
                    validate_fundamental_veto_subject,
                )

                verified = {
                    **verified,
                    "fundamental_veto_subject": validate_fundamental_veto_subject(
                        self.get_object(
                            production_receipt["payload"]["fundamental_veto_subject_ref"]
                        )
                    ),
                    "fundamental_advisory": validate_fundamental_advisory(
                        self.get_object(production_receipt["payload"]["fundamental_advisory_ref"])
                    ),
                }
            except SystemError:
                if deployed_release_ref is not None:
                    raise
                verified = completion["initial_generation"]
        else:
            try:
                verified = self._verify_generation(
                    pointer["generation_id"],
                    deployed_release_ref=deployed_release_ref,
                    validation_level="stat",
                )
            except SystemContractError:
                controller = completion["initial_generation"]["emergency_controller"]
                if (
                    pointer["generation_id"] != controller["generation_id"]
                    or pointer["manifest_sha256"] != controller["manifest_sha256"]
                ):
                    raise SystemContractError(
                        "historical suspended pointer differs from initial controller"
                    )
                verified = self._verify_historical_suspended_generation(
                    pointer["generation_id"],
                    expected_manifest_sha256=controller["manifest_sha256"],
                    expected_manifest_contract_sha256=controller["manifest_contract_sha256"],
                    expected_manifest_payload_fields=controller["manifest_payload_fields"],
                )
        if verified["manifest_sha256"] != pointer["manifest_sha256"]:
            raise SystemContractError("active pointer manifest binding mismatch")
        if (
            verified["generation_state"] == "OPERATIONAL"
            and not verified["deployed_release_verified"]
            and not verified.get("historical_release_verified", False)
        ):
            _verify_installed_release(verified["release"])
            verified = {**verified, "deployed_release_verified": True}
        return {
            "pointer": pointer,
            "pointer_byte_sha256": current["pointer_byte_sha256"],
            "migration_completion": completion,
            **verified,
        }

    def _pointer_chain(self) -> list[dict[str, Any]]:
        current = self._storage.read_optional(ACTIVE_POINTER_PATH)
        if current is None:
            return []
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        while True:
            pointer_sha = current.byte_sha256
            if pointer_sha in seen:
                raise SystemContractError("active pointer history is cyclic")
            seen.add(pointer_sha)
            pointer = self._validate_pointer(current.data)
            rows.append(
                {
                    "pointer": pointer,
                    "pointer_byte_sha256": pointer_sha,
                }
            )
            previous = pointer["previous_pointer_sha256"]
            if previous == EMPTY_POINTER_SHA256:
                return rows
            current = self._storage.read(POINTER_HISTORY_ROOT / f"{previous}.json")
            if current.byte_sha256 != previous:
                raise SystemContractError("retained previous pointer byte hash mismatch")

    def pointer_history(self, *, newest_first: bool = True) -> list[dict[str, Any]]:
        """Verify and resolve the retained pointer chain through literal ``EMPTY``."""

        if type(newest_first) is not bool:
            raise SystemContractError("newest_first must be boolean")
        rows = []
        for row in self._pointer_chain():
            pointer = row["pointer"]
            generation = self.verify_generation(pointer["generation_id"])
            if generation["manifest_sha256"] != pointer["manifest_sha256"]:
                raise SystemContractError("pointer history manifest binding mismatch")
            rows.append({**row, "generation": generation})
        return rows if newest_first else list(reversed(rows))

    def validate_active_resolution(
        self,
        resolved: Mapping[str, Any],
        *,
        deployed_release_ref: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Re-read and compare a caller-supplied resolved active-state object."""

        if type(resolved) is not dict:
            raise SystemContractError("active resolution must be an object")
        current = self.read_active(deployed_release_ref=deployed_release_ref)
        if current is None:
            raise SystemNotFound("active pointer is absent")
        if dict(resolved) != current:
            raise SystemContractError("active resolution does not match exact readback")
        return current

    def verify(
        self,
        generation_id: str | None = None,
        *,
        deployed_release_ref: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Verify a named generation or report normal uninitialized active state."""

        if generation_id is not None:
            normalized_id = _require_sha256(generation_id, label="generation_id")
            chain = self._pointer_chain()
            if chain and chain[-1]["pointer"]["generation_id"] == normalized_id:
                try:
                    current_verified = self.verify_generation(
                        normalized_id,
                        deployed_release_ref=deployed_release_ref,
                    )
                    if (
                        current_verified.get("generation_state") == "OPERATIONAL"
                        and type(current_verified.get("fundamental_advisory")) is not dict
                    ):
                        raise SystemContractError(
                            "operational generation Fundamental advisory is absent"
                        )
                    return current_verified
                except SystemError:
                    pass
                completion = self.verify_migration_completion(chain)
                historical = completion["initial_generation"]
                anchored_release = validate_frozen_object_ref(
                    completion["authorization"]["payload"]["deployed_release_ref"],
                    label="initial authorization deployed release ref",
                )
                if (
                    deployed_release_ref is not None
                    and validate_frozen_object_ref(
                        deployed_release_ref,
                        label="deployed_release_ref",
                    )
                    != anchored_release
                ):
                    raise SystemContractError(
                        "requested release differs from initial authorization"
                    )
                final_payload = completion["final_cutover_authorization"]["payload"]
                return {
                    **historical,
                    "calendar_authority_route": final_payload.get(
                        "calendar_authorization_basis", {}
                    ).get("authority_route"),
                    "calendar_authority_confidence": (
                        "DEGRADED"
                        if final_payload.get("calendar_source_limitations")
                        else "OFFICIAL"
                    ),
                    "calendar_source_limitations": list(
                        final_payload.get("calendar_source_limitations", [])
                    ),
                }
            return self.verify_generation(generation_id, deployed_release_ref=deployed_release_ref)
        active = self.read_active(deployed_release_ref=deployed_release_ref)
        if active is not None:
            return active
        return {
            "state": "UNINITIALIZED",
            "verified": False,
            "active_pointer_sha256": EMPTY_POINTER_SHA256,
            "generation_id": None,
            "blockers": [SYSTEM_ACTIVE_POINTER_ABSENT],
        }

    def status(  # noqa: C901
        self,
        *,
        deployed_release_ref: Mapping[str, Any] | None = None,
        external_routing: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a fail-closed, path-free status summary for CLI/public readers."""

        try:
            active = self.read_active(deployed_release_ref=deployed_release_ref)
            if active is None:
                return {
                    "state": "UNINITIALIZED",
                    "verified": False,
                    "active_pointer_sha256": EMPTY_POINTER_SHA256,
                    "generation_id": None,
                    "readiness": None,
                    "calendar_authority_route": None,
                    "calendar_authority_confidence": None,
                    "calendar_source_limitations": [],
                    "fundamental_advisory": None,
                    "blockers": [SYSTEM_ACTIVE_POINTER_ABSENT],
                    "external_routing_state": "UNINITIALIZED",
                }
            readiness = dict(active["readiness"]["payload"])
            blockers = list(readiness["blockers"])
            state_values = {
                readiness["factor_state"],
                readiness["mainline_state"],
                readiness["investment_state"],
            }
            if active["generation_state"] == "SYSTEM_SUSPENDED" or "SUSPENDED" in state_values:
                state = "SUSPENDED"
            elif (
                readiness["factor_state"] == "READY"
                and readiness["mainline_state"] == "UNINITIALIZED"
                and readiness["investment_state"] == "BLOCKED"
            ):
                state = "PARTIAL"
            elif blockers:
                state = "BLOCKED"
            else:
                state = "ACTIVE"
            if (
                active["generation_state"] == "OPERATIONAL"
                and not active["deployed_release_verified"]
            ):
                if state == "ACTIVE":
                    state = "PARTIAL"
                blockers = sorted(set(blockers) | {"SYSTEM_DEPLOYED_RELEASE_UNCONFIRMED"})
            routing_state = self._external_routing_state(
                active["manifest"]["payload"]["automation_semantic_sha256"],
                external_routing,
            )
            final_payload = active["migration_completion"]["final_cutover_authorization"]["payload"]
            calendar_limitations = list(final_payload["calendar_source_limitations"])
            calendar_route = (
                "TRUSTED_PROVIDER_DEGRADED" if calendar_limitations else "EXCHANGE_OFFICIAL"
            )
            fundamental_artifact = active.get("fundamental_advisory")
            fundamental_advisory = None
            if (
                type(fundamental_artifact) is dict
                and type(fundamental_artifact.get("payload")) is dict
            ):
                fundamental_payload = fundamental_artifact["payload"]
                fundamental_advisory = {
                    field: fundamental_payload[field]
                    for field in (
                        "integrity_status",
                        "required_by_active_factor_set",
                        "system_as_of_date",
                        "fundamental_snapshot_cutoff_date",
                        "calendar_age_days",
                        "open_session_age",
                        "latest_admitted_available_at",
                        "last_refresh_basis",
                        "disclosure_check",
                        "freshness_policy",
                        "default_action",
                        "operator_veto_present",
                        "effective_action",
                        "source_limitations",
                    )
                }
            if (
                fundamental_advisory is None
                or fundamental_advisory["effective_action"] != "PROCEED"
            ):
                state = "BLOCKED"
                blockers = sorted(set(blockers) | {"SYSTEM_FUNDAMENTAL_ADVISORY_INVALID"})
            return {
                "state": state,
                "verified": True,
                "active_pointer_sha256": active["pointer_byte_sha256"],
                "generation_id": active["generation_id"],
                "readiness": readiness,
                "calendar_authority_route": calendar_route,
                "calendar_authority_confidence": (
                    "DEGRADED" if calendar_limitations else "OFFICIAL"
                ),
                "calendar_source_limitations": calendar_limitations,
                "fundamental_advisory": fundamental_advisory,
                "blockers": blockers,
                "external_routing_state": routing_state,
            }
        except SystemError as exc:
            return {
                "state": "BLOCKED",
                "verified": False,
                "active_pointer_sha256": None,
                "generation_id": None,
                "readiness": None,
                "calendar_authority_route": None,
                "calendar_authority_confidence": None,
                "calendar_source_limitations": [],
                "fundamental_advisory": None,
                "blockers": [exc.code],
                "external_routing_state": "BLOCKED",
            }
        except Exception:
            return {
                "state": "BLOCKED",
                "verified": False,
                "active_pointer_sha256": None,
                "generation_id": None,
                "readiness": None,
                "fundamental_advisory": None,
                "blockers": ["SYSTEM_ERROR"],
                "external_routing_state": "BLOCKED",
            }

    @staticmethod
    def _external_routing_state(
        expected_semantic_sha256: str,
        observed: Mapping[str, Any] | None,
    ) -> str:
        if observed is None:
            return "UNCONFIRMED"
        if type(observed) is not dict or set(observed) != {
            "automation_semantic_sha256",
            "scheduler_enabled",
        }:
            raise SystemContractError("external routing observation fields are not exact")
        actual = _require_sha256(
            observed.get("automation_semantic_sha256"),
            label="external_routing.automation_semantic_sha256",
        )
        if type(observed.get("scheduler_enabled")) is not bool:
            raise SystemContractError("external routing scheduler state is invalid")
        if actual != expected_semantic_sha256:
            return "SYSTEM_EXTERNAL_ROUTING_DRIFT"
        if not observed["scheduler_enabled"]:
            return "SYSTEM_ACTIVE_AUTOMATION_DISABLED"
        return "ACTIVE"


__all__ = [
    "GENERATION_STATES",
    "MANIFEST_KIND",
    "OBJECT_REF_FIELDS",
    "OPERATIONAL_READINESS_KIND",
    "OPERATIONAL_DATA_ROLES",
    "POINTER_FIELDS",
    "READINESS_KINDS",
    "RELEASE_KIND",
    "SOURCE_FORMAT_MEDIA_TYPES",
    "SUSPENDED_READINESS_KIND",
    "SystemStore",
    "generation_assembly_identity",
    "object_ref_for_artifact",
    "validate_object_ref",
]
