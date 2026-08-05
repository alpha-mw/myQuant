"""Immutable, resumable V17 v4 forward-evidence orchestration.

The session reference is the sole discovery artifact.  Stage outputs, stage
receipts, and ``run.json`` are deliberately undiscoverable orphans until the
complete closure has been replayed and the final factor pointer has been
reread.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from enum import Enum
import hashlib
import inspect
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any, Final, NoReturn

from quant_investor.v17_v4_contract.canonical import (
    canonical_bytes,
    canonical_resource_bytes,
    load_canonical_resource,
    seal_semantic,
    strict_json_loads,
    validate_semantic_sha,
)
from quant_investor.v17_v4_contract.identities import (
    require_sha256,
    require_utc_timestamp,
)
from quant_investor.v17_v4_contract.schema_validation import (
    artifact_identity_field,
    validate_artifact,
)

from .run_profiles import (
    LifecycleLabel,
    ProfileDefinition,
    RunProfile,
    STAGE_LIFECYCLE_LABELS,
    normalize_profile,
    normalize_stage,
    profile_definition,
)
from .source_storage import (
    RUN_ROOT,
    SHADOW_ROOT,
    GovernedStore,
    ExactReferenceReader,
    SourceExactOnceConflict,
    SourceStorageError,
    SourceStorageSecurityError,
    StoredBytes,
    WriteResult,
    canonical_governed_path,
)

FORWARD_REQUEST_VERSION: Final = "myquant.v17.v4.forward-run-request.v1"
STAGE_OUTPUT_VERSION: Final = "myquant.v17.v4.forward-stage-output.v1"
STAGE_RECEIPT_VERSION: Final = "myquant.v17.v4.forward-stage-receipt.v1"
FORWARD_RUN_VERSION: Final = "myquant.v17.v4.forward-observation-run.v1"
FORWARD_SESSION_VERSION: Final = "myquant.v17.v4.forward-observation-session-ref.v1"

FORWARD_REQUEST_ROOT: Final = RUN_ROOT / "forward_requests"
FORWARD_EVIDENCE_ROOT: Final = SHADOW_ROOT / "forward_evidence"
MAX_ARTIFACT_BYTES: Final = 64 * 1024 * 1024
DISK_FREE_FLOOR_BYTES: Final = 512 * 1024 * 1024
RUN_STATE_INACTIVE: Final = "INACTIVE"
RUN_STATE_FORWARD_EVIDENCE_ACTIVE: Final = "FORWARD_EVIDENCE_ACTIVE"
RUN_STATE_EXPLORE_COMPLETE: Final = "EXPLORE_COMPLETE"
RUN_STATE_BLOCKED: Final = "BLOCKED"

_STRATEGY_RE: Final = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", re.ASCII)
_REQUEST_ID_RE: Final = re.compile(
    r"^forward-request-[0-9a-f]{64}$",
    re.ASCII,
)
_SHA_RE: Final = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_FORBIDDEN_TRUE_FIELDS: Final = frozenset(
    {
        "broker_authority",
        "broker_enabled",
        "broker_authorized",
        "execution_authority",
        "execution_enabled",
        "execution_authorized",
        "formal_authority",
        "formal_eligible",
        "mainline_authority",
        "order_authority",
        "order_enabled",
        "order_authorized",
        "policy_promotion_eligible",
        "promotion_authority",
        "promotion_eligible",
        "production",
        "production_authority",
        "provider_authority",
        "provider_calls_allowed",
        "provider_enabled",
        "provider_authorized",
        "trade_authority",
        "trade_enabled",
        "trade_authorized",
    }
)
_FORBIDDEN_BOOLEAN_NAMES: Final = frozenset(
    {
        "broker",
        "execution",
        "mainline",
        "order",
        "promotion",
        "provider",
        "production",
        "trade",
    }
)
_PIT_BOOLEAN_FIELDS: Final = frozenset(
    {
        "pit_admitted",
        "pit_complete",
        "pit_valid",
    }
)
_FUTURE_BOOLEAN_FIELDS: Final = frozenset(
    {
        "contains_future_data",
        "future_data",
        "future_data_present",
        "lookahead_present",
    }
)
_LINEAGE_FIELDS: Final = frozenset(
    {
        "lineage_receipt_refs",
        "parent_receipt_refs",
        "prior_receipt_refs",
    }
)
_STAGE_RESULT_FIELDS: Final = frozenset(
    {
        "_forward_stage_result",
        "authority",
        "completeness",
        "expected_payload_sha256",
        "future_data_present",
        "lineage_valid",
        "payload",
        "pit_valid",
        "schema_valid",
    }
)

NO_SIDE_EFFECT_FLAGS: Final = {
    "broker": False,
    "execution": False,
    "mainline": False,
    "order": False,
    "promotion": False,
    "production": False,
    "provider": False,
    "trade": False,
}

NO_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "mainline_authority": False,
    "order": False,
    "production": False,
    "research_only": True,
    "trade": False,
}


class ExecutionOutcome(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    BLOCKED = "BLOCKED"
    SKIPPED = "SKIPPED"


class Completeness(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    UNAVAILABLE = "UNAVAILABLE"


class ForwardEvidenceError(RuntimeError):
    """Fail-closed V17 v4 forward-evidence error."""

    exit_code = 2

    def __init__(self, code: str) -> None:
        self.code = code
        self.run_state = RUN_STATE_BLOCKED
        super().__init__(f"V17_V4_FORWARD_EVIDENCE_BLOCKED:{code}")


def _blocked(code: str) -> NoReturn:
    raise ForwardEvidenceError(code)


@dataclass(frozen=True)
class StageResult:
    """Deterministic result returned by one injected stage callback."""

    payload: Mapping[str, Any]
    completeness: Completeness | str = Completeness.COMPLETE
    expected_payload_sha256: str | None = None
    schema_valid: bool = True
    pit_valid: bool = True
    future_data_present: bool = False
    authority: bool = False
    lineage_valid: bool = True


@dataclass(frozen=True)
class StageContext:
    workspace_root: Path
    request: Mapping[str, Any]
    request_ref: Mapping[str, str]
    profile: RunProfile
    stage: str
    required: bool
    previous_receipt_refs: tuple[Mapping[str, str], ...]
    previous_output_refs: tuple[Mapping[str, str], ...]
    output_ref: Mapping[str, str] | None = None


StageCallback = Callable[[StageContext], StageResult | Mapping[str, Any] | bytes | None]
StageReader = Callable[..., Any]
StageValidator = Callable[..., Any]
ReferenceReader = Callable[..., Any]
EventHook = Callable[..., Any]
DiskFreeReader = Callable[[Path], int]


class _ForwardEvidenceStore(GovernedStore):
    """Writer narrowed to the request and forward-evidence roots."""

    def _canonical_path(
        self,
        value: str | PurePosixPath,
    ) -> PurePosixPath:
        path = canonical_governed_path(value)
        parts = path.parts
        if (
            len(parts) == 5
            and parts[:4] == ("data", "private", "v17_v4_runs", "forward_requests")
            and parts[4].endswith(".json")
            and _REQUEST_ID_RE.fullmatch(parts[4][:-5]) is not None
        ):
            return path
        prefix = (
            "results",
            "v17_v4_shadow",
            "forward_evidence",
            "strategies",
        )
        if len(parts) < 8 or parts[:4] != prefix:
            raise SourceStorageSecurityError("path is outside V17 v4 forward evidence")
        if _STRATEGY_RE.fullmatch(parts[4]) is None:
            raise SourceStorageSecurityError("invalid forward strategy path")
        if parts[5] == "runs":
            if _REQUEST_ID_RE.fullmatch(parts[6]) is None:
                raise SourceStorageSecurityError("invalid forward request path")
            if len(parts) == 8 and parts[7] == "run.json":
                return path
            if (
                len(parts) == 9
                and parts[7] in {"outputs", "receipts"}
                and parts[8].endswith(".json")
            ):
                try:
                    normalize_stage(parts[8][:-5])
                except ValueError as exc:
                    raise SourceStorageSecurityError("invalid forward stage path") from exc
                return path
        if (
            parts[5] == "sessions"
            and len(parts) == 8
            and _canonical_session(parts[6]) == parts[6]
            and parts[7].endswith(".json")
            and _REQUEST_ID_RE.fullmatch(parts[7][:-5]) is not None
        ):
            return path
        raise SourceStorageSecurityError("path is outside V17 v4 forward evidence")


def _canonical_session(value: Any) -> str:
    if type(value) is not str:
        _blocked("decision_session")
    try:
        normalized = date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ForwardEvidenceError("decision_session") from exc
    if normalized != value:
        _blocked("decision_session")
    return normalized


def _canonical_strategy(value: Any) -> str:
    if type(value) is not str or _STRATEGY_RE.fullmatch(value) is None:
        _blocked("strategy")
    return value


def _json_value(value: Any, *, label: str = "value") -> Any:
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if isinstance(value, Enum):
        return _json_value(value.value, label=label)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if type(key) is not str:
                _blocked(f"{label}_non_string_key")
            result[key] = _json_value(child, label=label)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(child, label=label) for child in value]
    _blocked(f"{label}_non_json")


def _resource_bytes(document: Mapping[str, Any]) -> bytes:
    try:
        normalized = _json_value(document, label="artifact")
        validate_artifact(normalized)
        return canonical_resource_bytes(
            normalized,
            max_bytes=MAX_ARTIFACT_BYTES,
        )
    except Exception as exc:
        raise ForwardEvidenceError("artifact_schema_or_size") from exc


def _semantic_body(document: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(_json_value(document, label="request"))
    body.pop("request_id", None)
    body.pop("semantic_sha256", None)
    return body


def _request_digest(document: Mapping[str, Any]) -> str:
    try:
        raw = canonical_bytes(
            _semantic_body(document),
            max_bytes=MAX_ARTIFACT_BYTES,
        )
    except Exception as exc:
        raise ForwardEvidenceError("request_schema_or_size") from exc
    return hashlib.sha256(raw).hexdigest()


def _request_shape(document: Mapping[str, Any]) -> tuple[RunProfile, str, str]:
    if document.get("version") != FORWARD_REQUEST_VERSION:
        _blocked("request_version")
    try:
        validate_artifact(dict(document))
    except Exception as exc:
        raise ForwardEvidenceError("request_schema") from exc
    try:
        profile = normalize_profile(document.get("request_profile"))
    except ValueError as exc:
        raise ForwardEvidenceError("request_profile") from exc
    strategy = _canonical_strategy(document.get("strategy_id"))
    session = _canonical_session(document.get("decision_session"))
    try:
        cutoff = require_utc_timestamp(
            document["cutoff"],
            label="request.cutoff",
        )
        created_at = require_utc_timestamp(
            document["created_at"],
            label="request.created_at",
        )
    except Exception as exc:
        raise ForwardEvidenceError("request_cutoff") from exc
    if cutoff[:10] < session or created_at < cutoff:
        _blocked("request_cutoff_binding")
    stage_ids = [
        str(row["stage_id"]) for row in document.get("stage_inputs", []) if isinstance(row, Mapping)
    ]
    if stage_ids != sorted(stage_ids) or len(stage_ids) != len(set(stage_ids)):
        _blocked("request_stage_inputs")
    allowed_stages = set(profile_definition(profile).stages)
    try:
        if any(normalize_stage(stage_id) not in allowed_stages for stage_id in stage_ids):
            _blocked("request_stage_inputs")
    except ValueError as exc:
        raise ForwardEvidenceError("request_stage_inputs") from exc
    digest = _request_digest(document)
    if document.get("request_id") != f"forward-request-{digest}":
        _blocked("request_id")
    _validate_no_authority(document)
    return profile, strategy, session


def build_forward_request(request_body: Mapping[str, Any]) -> dict[str, Any]:
    """Build one canonical, content-addressed request document."""

    body = _semantic_body(request_body)
    body.setdefault("version", FORWARD_REQUEST_VERSION)
    body.setdefault("protocol_version", "myquant.v17.v4")
    body.setdefault("authority", dict(NO_AUTHORITY))
    body.setdefault("request_profile", RunProfile.FORWARD_EVIDENCE.value)
    digest = _request_digest(body)
    request_id = f"forward-request-{digest}"
    provided_id = request_body.get("request_id")
    provided_sha = request_body.get("semantic_sha256")
    if provided_id is not None and provided_id != request_id:
        _blocked("request_id")
    document = dict(body)
    document["request_id"] = request_id
    document = seal_semantic(document)
    if provided_sha is not None and provided_sha != document["semantic_sha256"]:
        _blocked("request_semantic_sha256")
    _request_shape(document)
    return document


def _disk_reader(path: Path) -> int:
    return shutil.disk_usage(path).free


def _ensure_disk_free(
    workspace_root: Path,
    disk_free_reader: DiskFreeReader,
) -> None:
    try:
        free = disk_free_reader(workspace_root)
    except Exception as exc:
        raise ForwardEvidenceError("disk_free_unavailable") from exc
    if type(free) is not int or free < DISK_FREE_FLOOR_BYTES:
        _blocked("disk_free_below_512mib")


def _write_exact(
    store: _ForwardEvidenceStore,
    path: str,
    raw: bytes,
    *,
    disk_free_reader: DiskFreeReader,
) -> WriteResult:
    observed = store.read_optional(path)
    if observed is None:
        _ensure_disk_free(store.workspace_root, disk_free_reader)
    try:
        return store.write_exact_once(path, raw)
    except SourceExactOnceConflict as exc:
        raise ForwardEvidenceError("exact_once_conflict") from exc
    except SourceStorageError as exc:
        raise ForwardEvidenceError("storage_write") from exc


def _request_path(request_id: str) -> str:
    if _REQUEST_ID_RE.fullmatch(request_id) is None:
        _blocked("request_id")
    return str(FORWARD_REQUEST_ROOT / f"{request_id}.json")


def _run_root(strategy: str, request_id: str) -> PurePosixPath:
    return (
        FORWARD_EVIDENCE_ROOT / "strategies" / _canonical_strategy(strategy) / "runs" / request_id
    )


def _output_path(strategy: str, request_id: str, stage: str) -> str:
    return str(_run_root(strategy, request_id) / "outputs" / f"{normalize_stage(stage)}.json")


def _receipt_path(strategy: str, request_id: str, stage: str) -> str:
    return str(_run_root(strategy, request_id) / "receipts" / f"{normalize_stage(stage)}.json")


def _run_path(strategy: str, request_id: str) -> str:
    return str(_run_root(strategy, request_id) / "run.json")


def _session_path(
    strategy: str,
    decision_session: str,
    request_id: str,
) -> str:
    return str(
        FORWARD_EVIDENCE_ROOT
        / "strategies"
        / _canonical_strategy(strategy)
        / "sessions"
        / _canonical_session(decision_session)
        / f"{request_id}.json"
    )


def _artifact_ref(
    document: Mapping[str, Any],
    *,
    relative_path: str,
    raw: bytes | None = None,
) -> dict[str, str]:
    artifact_raw = _resource_bytes(document) if raw is None else raw
    version = document.get("version")
    if type(version) is not str:
        _blocked("artifact_version")
    try:
        identity_field = artifact_identity_field(version)
    except Exception as exc:
        raise ForwardEvidenceError("artifact_identity") from exc
    artifact_id = document.get(identity_field)
    semantic = document.get("semantic_sha256")
    cutoff = document.get("cutoff")
    strategy_id = document.get("strategy_id")
    if (
        type(artifact_id) is not str
        or type(semantic) is not str
        or _SHA_RE.fullmatch(semantic) is None
        or type(cutoff) is not str
        or type(strategy_id) is not str
    ):
        _blocked("artifact_semantic_sha256")
    return {
        "artifact_id": artifact_id,
        "artifact_version": version,
        "byte_sha256": hashlib.sha256(artifact_raw).hexdigest(),
        "cutoff": cutoff,
        "relative_path": str(canonical_governed_path(relative_path)),
        "semantic_sha256": semantic,
        "strategy_id": strategy_id,
    }


def _request_ref(
    document: Mapping[str, Any],
    *,
    relative_path: str,
    raw: bytes,
) -> dict[str, str]:
    return _artifact_ref(document, relative_path=relative_path, raw=raw)


def publish_forward_request(
    workspace_root: str | os.PathLike[str],
    request_body: Mapping[str, Any],
    *,
    disk_free_reader: DiskFreeReader | None = None,
) -> dict[str, Any]:
    """Publish an immutable request and return its exact path/SHA pair."""

    document = build_forward_request(request_body)
    path = _request_path(document["request_id"])
    raw = _resource_bytes(document)
    store = _ForwardEvidenceStore(
        workspace_root,
        max_read_bytes=MAX_ARTIFACT_BYTES,
        max_hash_bytes=MAX_ARTIFACT_BYTES,
    )
    result = _write_exact(
        store,
        path,
        raw,
        disk_free_reader=disk_free_reader or _disk_reader,
    )
    readback = store.read(path, result.byte_sha256)
    if readback != raw:
        _blocked("request_readback")
    return {
        "created": result.created,
        "request": document,
        "request_id": document["request_id"],
        "request_path": path,
        "request_sha256": result.byte_sha256,
    }


def _read_document(
    store: _ForwardEvidenceStore,
    path: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    try:
        raw = store.read(path, expected_sha256)
        value = load_canonical_resource(
            raw,
            label=path,
            max_bytes=MAX_ARTIFACT_BYTES,
        )
        validate_artifact(value)
    except Exception as exc:
        raise ForwardEvidenceError("artifact_readback") from exc
    if type(value) is not dict:
        _blocked("artifact_schema")
    return dict(value), raw


def _read_optional_document(
    store: _ForwardEvidenceStore,
    path: str,
) -> tuple[dict[str, Any], bytes] | None:
    try:
        observed = store.read_optional(path)
    except Exception as exc:
        raise ForwardEvidenceError("artifact_readback") from exc
    if observed is None:
        return None
    try:
        value = load_canonical_resource(
            observed.data,
            label=path,
            max_bytes=MAX_ARTIFACT_BYTES,
        )
        validate_artifact(value)
    except Exception as exc:
        raise ForwardEvidenceError("artifact_readback") from exc
    if type(value) is not dict:
        _blocked("artifact_schema")
    return dict(value), observed.data


def _invoke(callback: Callable[..., Any], *args: Any) -> Any:
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return callback(*args)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    ]
    if any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    ):
        return callback(*args)
    return callback(*args[: len(positional)])


def _validate_no_authority(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "authority":
                if child is False:
                    continue
                if isinstance(child, Mapping) and child == NO_AUTHORITY:
                    continue
                _blocked("authority")
            if key in _FORBIDDEN_TRUE_FIELDS and child is not False:
                _blocked("authority")
            if key in _FORBIDDEN_BOOLEAN_NAMES and type(child) is bool and child:
                _blocked("authority")
            _validate_no_authority(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_no_authority(child)


def _validate_sha_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key.endswith("sha256"):
                if type(child) is not str or _SHA_RE.fullmatch(child) is None:
                    _blocked("sha256")
            _validate_sha_fields(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_sha_fields(child)


def _validate_pit_and_future(
    value: Any,
    *,
    decision_session: str,
    request_cutoff: str | None,
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in _PIT_BOOLEAN_FIELDS and child is not True:
                _blocked("pit")
            if key == "pit_status" and str(child).upper() not in {
                "ADMITTED",
                "COMPLETE",
                "PASS",
                "SUCCEEDED",
                "VALID",
            }:
                _blocked("pit")
            if key in _FUTURE_BOOLEAN_FIELDS and child is not False:
                _blocked("future_data")
            if type(child) is str and (
                key == "decision_session"
                or key == "as_of_session"
                or key == "trade_date"
                or key.endswith("_session")
            ):
                try:
                    observed_session = date.fromisoformat(child).isoformat()
                except ValueError:
                    observed_session = ""
                if observed_session and observed_session > decision_session:
                    _blocked("future_data")
            if (
                request_cutoff is not None
                and type(child) is str
                and key
                in {
                    "data_cutoff",
                    "evidence_cutoff",
                    "source_cutoff",
                }
            ):
                try:
                    observed_cutoff = require_utc_timestamp(
                        child,
                        label=key,
                    )
                except Exception as exc:
                    raise ForwardEvidenceError("future_data") from exc
                if observed_cutoff > request_cutoff:
                    _blocked("future_data")
            _validate_pit_and_future(
                child,
                decision_session=decision_session,
                request_cutoff=request_cutoff,
            )
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_pit_and_future(
                child,
                decision_session=decision_session,
                request_cutoff=request_cutoff,
            )


def _validate_lineage_claims(
    value: Any,
    expected: Sequence[Mapping[str, str]],
) -> None:
    if isinstance(value, Mapping):
        if type(value.get("version")) is str and type(value.get("semantic_sha256")) is str:
            return
        for key, child in value.items():
            if key in _LINEAGE_FIELDS and child != expected:
                _blocked("lineage")
            _validate_lineage_claims(child, expected)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_lineage_claims(child, expected)


def _sorted_refs(
    references: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    return sorted(
        (dict(reference) for reference in references),
        key=lambda reference: (
            reference["relative_path"],
            reference["byte_sha256"],
            reference["artifact_id"],
        ),
    )


def _validate_reference_path(path: Any) -> str:
    if type(path) is not str:
        _blocked("reference_path")
    parsed = PurePosixPath(path)
    if (
        not path
        or "\\" in path
        or parsed.is_absolute()
        or str(parsed) != path
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        _blocked("reference_path")
    return path


def _validate_external_references(
    value: Any,
    *,
    reference_reader: ReferenceReader | None,
    context: StageContext,
) -> None:
    if isinstance(value, Mapping):
        has_path = "relative_path" in value
        has_sha = "byte_sha256" in value
        if has_path != has_sha:
            _blocked("reference_shape")
        if has_path:
            path = _validate_reference_path(value["relative_path"])
            try:
                expected = require_sha256(
                    value["byte_sha256"],
                    label="reference.byte_sha256",
                )
            except Exception as exc:
                raise ForwardEvidenceError("sha256") from exc
            if reference_reader is not None:
                try:
                    observed = _invoke(
                        reference_reader,
                        path,
                        expected,
                        dict(value),
                        context,
                    )
                except Exception as exc:
                    raise ForwardEvidenceError("reference_readback") from exc
                if isinstance(observed, StoredBytes):
                    observed = observed.data
                if hasattr(observed, "data") and isinstance(
                    observed.data,
                    bytes,
                ):
                    observed = observed.data
                if isinstance(observed, bytes):
                    if hashlib.sha256(observed).hexdigest() != expected:
                        _blocked("reference_sha256")
                elif observed is False:
                    _blocked("reference_readback")
        for child in value.values():
            _validate_external_references(
                child,
                reference_reader=reference_reader,
                context=context,
            )
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_external_references(
                child,
                reference_reader=reference_reader,
                context=context,
            )


def _allocation_roles(payload: Mapping[str, Any]) -> list[str]:
    if payload.get("version") == "myquant.v17.v4.forward-factor-allocation.v1":
        try:
            validate_artifact(dict(payload))
        except Exception as exc:
            raise ForwardEvidenceError("allocation_schema") from exc
        allocations = payload.get("allocations")
        if not isinstance(allocations, list):
            _blocked("allocation_schema")
        return [
            str(row["factor_tier"])
            for row in allocations
            if isinstance(row, Mapping) and row.get("selected") is True
        ]
    role_keys = {
        "allocation",
        "allocation_role",
        "allocation_track",
        "bucket",
        "factor_tier",
        "role",
        "track",
    }
    collection_keys = {"allocations", "allocation_roles", "roles", "tracks"}
    roles: list[str] = []

    def walk(value: Any, parent_key: str | None = None) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in role_keys and type(child) is str:
                    roles.append(child)
                elif key in collection_keys and isinstance(child, list):
                    for item in child:
                        if type(item) is str:
                            roles.append(item)
                        else:
                            walk(item, key)
                else:
                    walk(child, key)
        elif isinstance(value, list):
            for child in value:
                walk(child, parent_key)

    walk(payload)
    return roles


def _coerce_stage_result(value: Any) -> StageResult | None:
    if value is None:
        return None
    if isinstance(value, StageResult):
        return value
    if isinstance(value, bytes):
        try:
            payload = load_canonical_resource(
                value,
                label="stage callback payload",
                max_bytes=MAX_ARTIFACT_BYTES,
            )
        except Exception as exc:
            raise ForwardEvidenceError("stage_schema") from exc
        if type(payload) is not dict:
            _blocked("stage_schema")
        return StageResult(payload=payload)
    if isinstance(value, Mapping):
        if "payload" in value and (
            value.get("_forward_stage_result") is True or set(value).issubset(_STAGE_RESULT_FIELDS)
        ):
            payload = value.get("payload")
            if not isinstance(payload, Mapping):
                _blocked("stage_schema")
            return StageResult(
                payload=payload,
                completeness=value.get(
                    "completeness",
                    Completeness.COMPLETE,
                ),
                expected_payload_sha256=value.get("expected_payload_sha256"),
                schema_valid=value.get("schema_valid", True),
                pit_valid=value.get("pit_valid", True),
                future_data_present=value.get(
                    "future_data_present",
                    False,
                ),
                authority=value.get("authority", False),
                lineage_valid=value.get("lineage_valid", True),
            )
        return StageResult(payload=value)
    _blocked("stage_schema")


def _validate_stage_result(
    result: StageResult,
    *,
    context: StageContext,
    stage_validator: StageValidator | None,
    reference_reader: ReferenceReader | None,
) -> tuple[dict[str, Any], Completeness, str]:
    if (
        result.schema_valid is not True
        or result.pit_valid is not True
        or result.future_data_present is not False
        or result.authority is not False
        or result.lineage_valid is not True
    ):
        _blocked("stage_contract")
    try:
        completeness = Completeness(result.completeness)
    except ValueError as exc:
        raise ForwardEvidenceError("stage_completeness") from exc
    if completeness is Completeness.UNAVAILABLE:
        _blocked("provided_stage_unavailable")
    if context.required and completeness is not Completeness.COMPLETE:
        _blocked("required_stage_partial")
    payload = _json_value(result.payload, label="stage_payload")
    if type(payload) is not dict:
        _blocked("stage_schema")
    if payload.get("schema_valid") is False:
        _blocked("stage_schema")
    if payload.get("lineage_valid") is False:
        _blocked("lineage")
    _validate_no_authority(payload)
    _validate_sha_fields(payload)
    _validate_pit_and_future(
        payload,
        decision_session=str(context.request["decision_session"]),
        request_cutoff=context.request.get("cutoff"),
    )
    expected_lineage = [
        dict(reference) for reference in _sorted_refs(context.previous_receipt_refs)
    ]
    _validate_lineage_claims(payload, expected_lineage)
    for field, expected in (
        ("request_id", context.request["request_id"]),
        (
            "strategy_id",
            context.request.get(
                "strategy_id",
                context.request.get("strategy"),
            ),
        ),
        ("decision_session", context.request["decision_session"]),
    ):
        if field in payload and payload[field] != expected:
            _blocked("stage_binding")
    if context.stage == "allocation" and context.profile is RunProfile.FORWARD_EVIDENCE:
        roles = _allocation_roles(payload)
        if not roles or any(role.casefold() not in {"core", "challenger"} for role in roles):
            _blocked("allocation_not_core_challenger")
    _validate_external_references(
        payload,
        reference_reader=reference_reader,
        context=context,
    )
    if "semantic_sha256" in payload:
        try:
            validate_semantic_sha(payload)
        except Exception as exc:
            raise ForwardEvidenceError("stage_semantic_sha256") from exc
    if stage_validator is not None:
        try:
            validated = _invoke(stage_validator, payload, context)
        except Exception as exc:
            raise ForwardEvidenceError("stage_schema") from exc
        if validated is False:
            _blocked("stage_schema")
    payload_raw = canonical_bytes(
        payload,
        max_bytes=MAX_ARTIFACT_BYTES,
    )
    payload_sha = hashlib.sha256(payload_raw).hexdigest()
    if result.expected_payload_sha256 is not None:
        try:
            expected_payload_sha = require_sha256(
                result.expected_payload_sha256,
                label="expected_payload_sha256",
            )
        except Exception as exc:
            raise ForwardEvidenceError("stage_payload_sha256") from exc
        if payload_sha != expected_payload_sha:
            _blocked("stage_payload_sha256")
    return payload, completeness, payload_sha


def _stage_output_document(
    *,
    context: StageContext,
    payload: Mapping[str, Any],
    completeness: Completeness,
    payload_sha256: str,
) -> dict[str, Any]:
    return seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "completeness": completeness.value,
            "cutoff": context.request["cutoff"],
            "decision_session": context.request["decision_session"],
            "lineage_receipt_refs": _sorted_refs(context.previous_receipt_refs),
            "output_id": (f"{context.request['request_id']}-{context.stage}-output"),
            "payload_json": canonical_bytes(
                payload,
                max_bytes=MAX_ARTIFACT_BYTES,
            ).decode("utf-8"),
            "payload_sha256": payload_sha256,
            "protocol_version": "myquant.v17.v4",
            "recorded_at": context.request["created_at"],
            "request_ref": dict(context.request_ref),
            "stage_id": context.stage,
            "strategy_id": context.request["strategy_id"],
            "version": STAGE_OUTPUT_VERSION,
        }
    )


def _stage_receipt_document(
    *,
    context: StageContext,
    outcome: ExecutionOutcome,
    completeness: Completeness,
    output_ref: Mapping[str, str] | None,
    error_code: str | None = None,
) -> dict[str, Any]:
    receipt_id = f"{context.request['request_id']}-{context.stage}-receipt"
    return seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "blockers": [error_code] if error_code is not None else [],
            "completeness": completeness.value,
            "cutoff": context.request["cutoff"],
            "decision_session": context.request["decision_session"],
            "execution_outcome": outcome.value,
            "output_refs": ([dict(output_ref)] if output_ref is not None else []),
            "protocol_version": "myquant.v17.v4",
            "receipt_id": receipt_id,
            "recorded_at": context.request["created_at"],
            "request_ref": dict(context.request_ref),
            "stage_id": context.stage,
            "strategy_id": context.request["strategy_id"],
            "version": STAGE_RECEIPT_VERSION,
        }
    )


def _decode_stage_payload(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    payload_json = document.get("payload_json")
    if type(payload_json) is not str:
        _blocked("stage_output_schema")
    try:
        payload_raw = payload_json.encode("utf-8", errors="strict")
        payload = strict_json_loads(
            payload_raw,
            label="stage output payload_json",
            max_bytes=MAX_ARTIFACT_BYTES,
        )
    except Exception as exc:
        raise ForwardEvidenceError("stage_output_schema") from exc
    if (
        type(payload) is not dict
        or canonical_bytes(
            payload,
            max_bytes=MAX_ARTIFACT_BYTES,
        )
        != payload_raw
        or hashlib.sha256(payload_raw).hexdigest() != document.get("payload_sha256")
    ):
        _blocked("stage_output_schema")
    return dict(payload)


def _validate_stage_output_document(
    document: Mapping[str, Any],
    *,
    context: StageContext,
    stage_validator: StageValidator | None,
    reference_reader: ReferenceReader | None,
) -> tuple[dict[str, Any], Completeness]:
    try:
        validate_semantic_sha(document)
    except Exception as exc:
        raise ForwardEvidenceError("stage_output_semantic") from exc
    expected_lineage = [
        dict(reference) for reference in _sorted_refs(context.previous_receipt_refs)
    ]
    if (
        document.get("version") != STAGE_OUTPUT_VERSION
        or document.get("request_ref") != context.request_ref
        or document.get("stage_id") != context.stage
        or document.get("authority") != NO_AUTHORITY
        or document.get("cutoff") != context.request["cutoff"]
        or document.get("recorded_at") != context.request["created_at"]
        or document.get("lineage_receipt_refs") != expected_lineage
    ):
        _blocked("stage_output_binding")
    try:
        completeness = Completeness(document.get("completeness"))
    except ValueError as exc:
        raise ForwardEvidenceError("stage_output_completeness") from exc
    payload = _decode_stage_payload(document)
    result = StageResult(
        payload=payload,
        completeness=completeness,
        expected_payload_sha256=document["payload_sha256"],
    )
    normalized, _, _ = _validate_stage_result(
        result,
        context=context,
        stage_validator=stage_validator,
        reference_reader=reference_reader,
    )
    return normalized, completeness


def _validate_stage_receipt_document(
    document: Mapping[str, Any],
    *,
    context: StageContext,
) -> tuple[ExecutionOutcome, Completeness]:
    try:
        validate_semantic_sha(document)
    except Exception as exc:
        raise ForwardEvidenceError("stage_receipt_semantic") from exc
    try:
        outcome = ExecutionOutcome(document.get("execution_outcome"))
        completeness = Completeness(document.get("completeness"))
    except ValueError as exc:
        raise ForwardEvidenceError("stage_receipt_schema") from exc
    if (
        document.get("version") != STAGE_RECEIPT_VERSION
        or document.get("request_ref") != context.request_ref
        or document.get("stage_id") != context.stage
        or document.get("authority") != NO_AUTHORITY
        or document.get("cutoff") != context.request["cutoff"]
        or document.get("recorded_at") != context.request["created_at"]
    ):
        _blocked("stage_receipt_binding")
    if outcome is ExecutionOutcome.SKIPPED:
        if (
            context.required
            or completeness is not Completeness.UNAVAILABLE
            or document.get("output_refs") != []
            or document.get("blockers") != []
        ):
            _blocked("stage_skip_invalid")
    elif outcome is ExecutionOutcome.SUCCEEDED:
        if (
            completeness is Completeness.UNAVAILABLE
            or not isinstance(document.get("output_refs"), list)
            or len(document["output_refs"]) != 1
            or document.get("blockers") != []
        ):
            _blocked("stage_receipt_output")
        if context.required and completeness is not Completeness.COMPLETE:
            _blocked("required_stage_partial")
    elif (
        document.get("output_refs") != []
        or not isinstance(document.get("blockers"), list)
        or not document["blockers"]
    ):
        _blocked("blocked_receipt_code")
    return outcome, completeness


def _replay_stage_output(
    *,
    store: _ForwardEvidenceStore,
    output_ref: Mapping[str, Any],
    context: StageContext,
    stage_reader: StageReader | None,
    stage_validator: StageValidator | None,
    reference_reader: ReferenceReader | None,
) -> tuple[dict[str, Any], Completeness]:
    try:
        expected = require_sha256(
            output_ref.get("byte_sha256"),
            label="output_ref.byte_sha256",
        )
        path = str(canonical_governed_path(output_ref["relative_path"]))
    except Exception as exc:
        raise ForwardEvidenceError("stage_output_ref") from exc
    document, raw = _read_document(store, path, expected)
    observed_ref = _artifact_ref(document, relative_path=path, raw=raw)
    if observed_ref != dict(output_ref):
        _blocked("stage_output_ref")
    replay_context = replace(context, output_ref=observed_ref)
    payload, completeness = _validate_stage_output_document(
        document,
        context=replay_context,
        stage_validator=stage_validator,
        reference_reader=reference_reader,
    )
    if stage_reader is not None:
        try:
            replayed = _invoke(
                stage_reader,
                document,
                replay_context,
                raw,
                observed_ref,
            )
        except Exception as exc:
            raise ForwardEvidenceError("stage_replay") from exc
        if replayed is False:
            _blocked("stage_replay")
        if isinstance(replayed, bytes) and replayed != raw:
            _blocked("stage_replay")
        if isinstance(replayed, Mapping):
            if dict(replayed) != document and dict(replayed) != payload:
                _blocked("stage_replay")
    return document, completeness


def _read_request_stage_input(
    *,
    context: StageContext,
    stage_input: Mapping[str, Any],
) -> StageResult:
    reference = stage_input.get("artifact_ref")
    if not isinstance(reference, Mapping):
        _blocked("stage_input_ref")
    try:
        expected_sha = require_sha256(
            reference.get("byte_sha256"),
            label="stage_input.byte_sha256",
        )
        path = _validate_reference_path(reference.get("relative_path"))
        reader = ExactReferenceReader(
            context.workspace_root,
            max_read_bytes=MAX_ARTIFACT_BYTES,
            max_hash_bytes=MAX_ARTIFACT_BYTES,
        )
        raw = reader.read(path, expected_sha)
        document = load_canonical_resource(
            raw,
            label=f"stage input {context.stage}",
            max_bytes=MAX_ARTIFACT_BYTES,
        )
        validate_artifact(document)
    except Exception as exc:
        raise ForwardEvidenceError("stage_input_readback") from exc
    if type(document) is not dict:
        _blocked("stage_input_schema")
    try:
        identity_field = artifact_identity_field(document.get("version"))
        observed_ref = {
            "artifact_id": str(document[identity_field]),
            "artifact_version": str(document["version"]),
            "byte_sha256": hashlib.sha256(raw).hexdigest(),
            "cutoff": str(document["cutoff"]),
            "relative_path": path,
            "semantic_sha256": str(document["semantic_sha256"]),
            "strategy_id": str(document["strategy_id"]),
        }
    except Exception as exc:
        raise ForwardEvidenceError("stage_input_binding") from exc
    if observed_ref != dict(reference):
        _blocked("stage_input_binding")
    if (
        document.get("protocol_version") != "myquant.v17.v4"
        or document.get("cutoff") > context.request["cutoff"]
        or document.get("strategy_id") != context.request["strategy_id"]
    ):
        _blocked("stage_input_binding")
    _validate_no_authority(document)
    _validate_pit_and_future(
        document,
        decision_session=str(context.request["decision_session"]),
        request_cutoff=str(context.request["cutoff"]),
    )
    try:
        completeness = Completeness(stage_input.get("completeness"))
    except ValueError as exc:
        raise ForwardEvidenceError("stage_input_completeness") from exc
    if completeness is Completeness.UNAVAILABLE:
        _blocked("stage_input_completeness")
    payload: Mapping[str, Any] = document
    if document.get("version") == STAGE_OUTPUT_VERSION:
        if (
            document.get("stage_id") != context.stage
            or document.get("completeness") != completeness.value
        ):
            _blocked("stage_input_binding")
        payload = _decode_stage_payload(document)
    return StageResult(
        payload=payload,
        completeness=completeness,
    )


def _write_receipt(
    *,
    store: _ForwardEvidenceStore,
    strategy: str,
    request_id: str,
    context: StageContext,
    outcome: ExecutionOutcome,
    completeness: Completeness,
    output_ref: Mapping[str, str] | None,
    disk_free_reader: DiskFreeReader,
    error_code: str | None = None,
) -> dict[str, str]:
    document = _stage_receipt_document(
        context=context,
        outcome=outcome,
        completeness=completeness,
        output_ref=output_ref,
        error_code=error_code,
    )
    path = _receipt_path(strategy, request_id, context.stage)
    raw = _resource_bytes(document)
    result = _write_exact(
        store,
        path,
        raw,
        disk_free_reader=disk_free_reader,
    )
    readback, observed_raw = _read_document(
        store,
        path,
        result.byte_sha256,
    )
    _validate_stage_receipt_document(readback, context=context)
    return _artifact_ref(
        readback,
        relative_path=path,
        raw=observed_raw,
    )


def _execute_stage(
    *,
    store: _ForwardEvidenceStore,
    strategy: str,
    definition: ProfileDefinition,
    request: Mapping[str, Any],
    request_ref: Mapping[str, str],
    stage: str,
    callback: StageCallback | None,
    stage_input: Mapping[str, Any] | None,
    stage_reader: StageReader | None,
    stage_validator: StageValidator | None,
    reference_reader: ReferenceReader | None,
    previous_receipt_refs: list[Mapping[str, str]],
    previous_output_refs: list[Mapping[str, str]],
    disk_free_reader: DiskFreeReader,
    event_hook: EventHook | None,
) -> tuple[dict[str, str], dict[str, str] | None]:
    request_id = str(request["request_id"])
    required = definition.is_required(stage)
    context = StageContext(
        workspace_root=store.workspace_root,
        request=request,
        request_ref=request_ref,
        profile=definition.profile,
        stage=stage,
        required=required,
        previous_receipt_refs=tuple(previous_receipt_refs),
        previous_output_refs=tuple(previous_output_refs),
    )
    if callback is not None and stage_input is not None:
        _blocked("stage_input_callback_conflict")
    receipt_path = _receipt_path(strategy, request_id, stage)
    existing_receipt = _read_optional_document(store, receipt_path)
    if existing_receipt is not None:
        if callback is None and stage_input is not None:
            _read_request_stage_input(
                context=context,
                stage_input=stage_input,
            )
        receipt, receipt_raw = existing_receipt
        outcome, _ = _validate_stage_receipt_document(
            receipt,
            context=context,
        )
        receipt_ref = _artifact_ref(
            receipt,
            relative_path=receipt_path,
            raw=receipt_raw,
        )
        if outcome is ExecutionOutcome.BLOCKED:
            blockers = receipt.get("blockers")
            code = blockers[0] if isinstance(blockers, list) and blockers else "stage_blocked"
            _blocked(str(code))
        if outcome is ExecutionOutcome.SKIPPED:
            return receipt_ref, None
        output_ref = receipt["output_refs"][0]
        _replay_stage_output(
            store=store,
            output_ref=output_ref,
            context=context,
            stage_reader=stage_reader,
            stage_validator=stage_validator,
            reference_reader=reference_reader,
        )
        return receipt_ref, dict(output_ref)

    output_path = _output_path(strategy, request_id, stage)
    existing_output = _read_optional_document(store, output_path)
    if existing_output is not None:
        if callback is None and stage_input is not None:
            _read_request_stage_input(
                context=context,
                stage_input=stage_input,
            )
        output, output_raw = existing_output
        output_ref = _artifact_ref(
            output,
            relative_path=output_path,
            raw=output_raw,
        )
        _, completeness = _replay_stage_output(
            store=store,
            output_ref=output_ref,
            context=context,
            stage_reader=stage_reader,
            stage_validator=stage_validator,
            reference_reader=reference_reader,
        )
        receipt_ref = _write_receipt(
            store=store,
            strategy=strategy,
            request_id=request_id,
            context=context,
            outcome=ExecutionOutcome.SUCCEEDED,
            completeness=completeness,
            output_ref=output_ref,
            disk_free_reader=disk_free_reader,
        )
        return receipt_ref, output_ref

    if callback is None and stage_input is not None:
        try:
            result = _read_request_stage_input(
                context=context,
                stage_input=stage_input,
            )
            payload, completeness, payload_sha = _validate_stage_result(
                result,
                context=context,
                stage_validator=stage_validator,
                reference_reader=reference_reader,
            )
        except ForwardEvidenceError as exc:
            _write_receipt(
                store=store,
                strategy=strategy,
                request_id=request_id,
                context=context,
                outcome=ExecutionOutcome.BLOCKED,
                completeness=Completeness.PARTIAL,
                output_ref=None,
                error_code=exc.code,
                disk_free_reader=disk_free_reader,
            )
            raise
    elif callback is None:
        if not required:
            receipt_ref = _write_receipt(
                store=store,
                strategy=strategy,
                request_id=request_id,
                context=context,
                outcome=ExecutionOutcome.SKIPPED,
                completeness=Completeness.UNAVAILABLE,
                output_ref=None,
                disk_free_reader=disk_free_reader,
            )
            return receipt_ref, None
        receipt_ref = _write_receipt(
            store=store,
            strategy=strategy,
            request_id=request_id,
            context=context,
            outcome=ExecutionOutcome.BLOCKED,
            completeness=Completeness.UNAVAILABLE,
            output_ref=None,
            error_code="required_stage_absent",
            disk_free_reader=disk_free_reader,
        )
        del receipt_ref
        _blocked("required_stage_absent")

    else:
        try:
            raw_result = _invoke(callback, context)
            result = _coerce_stage_result(raw_result)
            if result is None:
                if not required:
                    receipt_ref = _write_receipt(
                        store=store,
                        strategy=strategy,
                        request_id=request_id,
                        context=context,
                        outcome=ExecutionOutcome.SKIPPED,
                        completeness=Completeness.UNAVAILABLE,
                        output_ref=None,
                        disk_free_reader=disk_free_reader,
                    )
                    return receipt_ref, None
                _blocked("required_stage_absent")
            payload, completeness, payload_sha = _validate_stage_result(
                result,
                context=context,
                stage_validator=stage_validator,
                reference_reader=reference_reader,
            )
        except ForwardEvidenceError as exc:
            _write_receipt(
                store=store,
                strategy=strategy,
                request_id=request_id,
                context=context,
                outcome=ExecutionOutcome.BLOCKED,
                completeness=Completeness.PARTIAL,
                output_ref=None,
                error_code=exc.code,
                disk_free_reader=disk_free_reader,
            )
            raise
        except Exception as exc:
            _write_receipt(
                store=store,
                strategy=strategy,
                request_id=request_id,
                context=context,
                outcome=ExecutionOutcome.BLOCKED,
                completeness=Completeness.PARTIAL,
                output_ref=None,
                error_code="stage_callback",
                disk_free_reader=disk_free_reader,
            )
            raise ForwardEvidenceError("stage_callback") from exc

    output = _stage_output_document(
        context=context,
        payload=payload,
        completeness=completeness,
        payload_sha256=payload_sha,
    )
    output_raw = _resource_bytes(output)
    output_write = _write_exact(
        store,
        output_path,
        output_raw,
        disk_free_reader=disk_free_reader,
    )
    output_ref = _artifact_ref(
        output,
        relative_path=output_path,
        raw=output_raw,
    )
    if event_hook is not None:
        _invoke(
            event_hook,
            "after_stage_output",
            replace(context, output_ref=output_ref),
        )
    _replay_stage_output(
        store=store,
        output_ref=output_ref,
        context=context,
        stage_reader=stage_reader,
        stage_validator=stage_validator,
        reference_reader=reference_reader,
    )
    if output_write.byte_sha256 != output_ref["byte_sha256"]:
        _blocked("stage_output_write_sha")
    receipt_ref = _write_receipt(
        store=store,
        strategy=strategy,
        request_id=request_id,
        context=context,
        outcome=ExecutionOutcome.SUCCEEDED,
        completeness=completeness,
        output_ref=output_ref,
        disk_free_reader=disk_free_reader,
    )
    if event_hook is not None:
        _invoke(
            event_hook,
            "after_stage_receipt",
            replace(context, output_ref=output_ref),
        )
    return receipt_ref, output_ref


def _lifecycle_labels(
    stages: Sequence[str],
    receipts: Sequence[Mapping[str, Any]],
) -> list[str]:
    outcomes = {
        stage: receipt.get("execution_outcome")
        for stage, receipt in zip(stages, receipts, strict=True)
    }
    labels: list[LifecycleLabel] = []
    if outcomes.get("source") == ExecutionOutcome.SUCCEEDED.value:
        labels.append(LifecycleLabel.SOURCE_SNAPSHOT)
    if outcomes.get("quant") == ExecutionOutcome.SUCCEEDED.value:
        labels.append(LifecycleLabel.QUANT_COMPLETE)
    if "fundamental" in outcomes:
        labels.append(LifecycleLabel.FUNDAMENTAL_PARTIAL_ALLOWED)
    if outcomes.get("fusion") == ExecutionOutcome.SUCCEEDED.value:
        labels.append(LifecycleLabel.FUSION_COMPLETE)
    if "deep" in outcomes:
        labels.append(LifecycleLabel.DEEP_OPTIONAL)
    if (
        outcomes.get("factor_universe_observation") == ExecutionOutcome.SUCCEEDED.value
        or outcomes.get("strategy_pool_observation") == ExecutionOutcome.SUCCEEDED.value
    ):
        labels.append(LifecycleLabel.SHADOW_OBSERVATION_CREATED)
    labels.append(LifecycleLabel.FORWARD_LABEL_PENDING)
    return [label.value for label in labels]


def _read_ref_document(
    store: _ForwardEvidenceStore,
    reference: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    try:
        path = str(canonical_governed_path(reference["relative_path"]))
        expected = require_sha256(
            reference["byte_sha256"],
            label="artifact_ref.byte_sha256",
        )
    except Exception as exc:
        raise ForwardEvidenceError("artifact_ref") from exc
    document, raw = _read_document(store, path, expected)
    if _artifact_ref(document, relative_path=path, raw=raw) != dict(reference):
        _blocked("artifact_ref")
    return document, raw


def _replay_receipt_closure(
    *,
    store: _ForwardEvidenceStore,
    definition: ProfileDefinition,
    request: Mapping[str, Any],
    request_ref: Mapping[str, str],
    receipt_refs: list[Mapping[str, str]],
    stage_readers: Mapping[str, StageReader],
    stage_validators: Mapping[str, StageValidator],
    reference_reader: ReferenceReader | None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if len(receipt_refs) != len(definition.stages):
        _blocked("receipt_count")
    receipts_by_stage: dict[str, tuple[dict[str, Any], dict[str, str]]] = {}
    for receipt_ref in receipt_refs:
        receipt, _ = _read_ref_document(store, receipt_ref)
        stage_id = receipt.get("stage_id")
        if (
            type(stage_id) is not str
            or stage_id in receipts_by_stage
            or stage_id not in definition.stages
        ):
            _blocked("receipt_stage")
        receipts_by_stage[stage_id] = (receipt, dict(receipt_ref))
    prior_receipts: list[Mapping[str, str]] = []
    prior_outputs: list[Mapping[str, str]] = []
    receipts: list[dict[str, Any]] = []
    outputs: list[dict[str, str]] = []
    for stage in definition.stages:
        try:
            receipt, receipt_ref = receipts_by_stage[stage]
        except KeyError as exc:
            raise ForwardEvidenceError("receipt_stage") from exc
        context = StageContext(
            workspace_root=store.workspace_root,
            request=request,
            request_ref=request_ref,
            profile=definition.profile,
            stage=stage,
            required=definition.is_required(stage),
            previous_receipt_refs=tuple(prior_receipts),
            previous_output_refs=tuple(prior_outputs),
        )
        outcome, _ = _validate_stage_receipt_document(
            receipt,
            context=context,
        )
        if outcome is ExecutionOutcome.BLOCKED:
            blockers = receipt.get("blockers")
            code = blockers[0] if isinstance(blockers, list) and blockers else "stage_blocked"
            _blocked(str(code))
        if outcome is ExecutionOutcome.SKIPPED:
            if definition.is_required(stage):
                _blocked("required_stage_skipped")
        else:
            output_ref = receipt["output_refs"][0]
            _replay_stage_output(
                store=store,
                output_ref=output_ref,
                context=context,
                stage_reader=stage_readers.get(stage),
                stage_validator=stage_validators.get(stage),
                reference_reader=reference_reader,
            )
            output_dict = dict(output_ref)
            prior_outputs.append(output_dict)
            outputs.append(output_dict)
        prior_receipts.append(dict(receipt_ref))
        receipts.append(receipt)
    return receipts, outputs


def _request_ref_readback(
    workspace_root: Path,
    *,
    request: Mapping[str, Any],
    fields: Sequence[str],
    error_code: str,
) -> None:
    reader = ExactReferenceReader(
        workspace_root,
        max_read_bytes=MAX_ARTIFACT_BYTES,
        max_hash_bytes=MAX_ARTIFACT_BYTES,
    )
    for field in fields:
        references = request.get(field)
        if not isinstance(references, list) or not references:
            _blocked(error_code)
        for index, reference in enumerate(references):
            if not isinstance(reference, Mapping):
                _blocked(error_code)
            try:
                path = _validate_reference_path(reference["relative_path"])
                expected_sha = require_sha256(
                    reference["byte_sha256"],
                    label=f"{field}[{index}].byte_sha256",
                )
                raw = reader.read(path, expected_sha)
                document = load_canonical_resource(
                    raw,
                    label=f"{field}[{index}]",
                    max_bytes=MAX_ARTIFACT_BYTES,
                )
                validate_semantic_sha(document)
                try:
                    identity_field = artifact_identity_field(document.get("version"))
                except Exception:
                    identity_field = None
                if identity_field is not None:
                    validate_artifact(document)
                    artifact_id = document.get(identity_field)
                else:
                    artifact_id = next(
                        (
                            value
                            for key, value in document.items()
                            if key.endswith("_id")
                            and key
                            not in {
                                "protocol_id",
                                "strategy_id",
                            }
                            and value == reference["artifact_id"]
                        ),
                        None,
                    )
                _validate_no_authority(document)
                _validate_sha_fields(document)
                _validate_pit_and_future(
                    document,
                    decision_session=str(request["decision_session"]),
                    request_cutoff=request.get("cutoff"),
                )
            except ForwardEvidenceError:
                raise
            except Exception as exc:
                raise ForwardEvidenceError(error_code) from exc
            if (
                artifact_id != reference["artifact_id"]
                or document.get("version") != reference["artifact_version"]
                or document.get("semantic_sha256") != reference["semantic_sha256"]
                or document.get("cutoff") != reference["cutoff"]
                or document.get("strategy_id") != reference["strategy_id"]
            ):
                _blocked(error_code)


def _factor_pointer_reread(
    callback: Callable[..., Any] | None,
    *,
    request: Mapping[str, Any],
    context: StageContext,
    reference_reader: ReferenceReader | None,
) -> None:
    _request_ref_readback(
        context.workspace_root,
        request=request,
        fields=("factor_refs",),
        error_code="factor_pointer_reread",
    )
    if callback is None:
        return
    try:
        observed = _invoke(callback, context)
    except Exception as exc:
        raise ForwardEvidenceError("factor_pointer_reread") from exc
    if observed is False:
        _blocked("factor_pointer_drift")
    expected_pointer = request.get(
        "factor_set_pointer_ref",
        request.get("factor_pointer_ref"),
    )
    expected_set = request.get("factor_set_ref")
    if observed is None or observed is True:
        return
    pointer: Any = None
    factor_set: Any = None
    if (
        isinstance(observed, tuple)
        and len(observed) == 2
        and all(isinstance(item, Mapping) for item in observed)
    ):
        pointer, factor_set = observed
    elif isinstance(observed, Mapping):
        pointer = observed.get(
            "factor_set_pointer_ref",
            observed.get("factor_pointer_ref", observed.get("pointer_ref")),
        )
        factor_set = observed.get("factor_set_ref")
        if pointer is None and expected_pointer is not None:
            pointer = observed
    else:
        _blocked("factor_pointer_reread")
    if expected_pointer is not None and dict(pointer or {}) != dict(expected_pointer):
        _blocked("factor_pointer_drift")
    if expected_set is not None and dict(factor_set or {}) != dict(expected_set):
        _blocked("factor_pointer_drift")
    replay_context = replace(context, stage="final")
    _validate_external_references(
        {"pointer": pointer, "factor_set": factor_set},
        reference_reader=reference_reader,
        context=replay_context,
    )


def _run_state(definition: ProfileDefinition) -> str:
    if definition.profile is RunProfile.FORWARD_EVIDENCE:
        return RUN_STATE_FORWARD_EVIDENCE_ACTIVE
    if definition.profile is RunProfile.EXPLORE:
        return RUN_STATE_EXPLORE_COMPLETE
    return RUN_STATE_BLOCKED


def _run_document(
    *,
    definition: ProfileDefinition,
    request: Mapping[str, Any],
    request_ref: Mapping[str, str],
    receipt_refs: Sequence[Mapping[str, str]],
    output_refs: Sequence[Mapping[str, str]],
    stage_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    completeness = (
        Completeness.PARTIAL
        if any(
            receipt.get("execution_outcome") == ExecutionOutcome.SUCCEEDED.value
            and receipt.get("completeness") == Completeness.PARTIAL.value
            for receipt in stage_receipts
        )
        else Completeness.COMPLETE
    )
    return seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "broker": False,
            "completeness": completeness.value,
            "cutoff": request["cutoff"],
            "decision_session": request["decision_session"],
            "execution": False,
            "execution_outcome": ExecutionOutcome.SUCCEEDED.value,
            "mainline_authority": False,
            "observation_refs": _sorted_refs(output_refs),
            "observation_run_id": (f"forward-observation-{request['request_id'][16:]}"),
            "order": False,
            "protocol_version": "myquant.v17.v4",
            "recorded_at": request["created_at"],
            "research_only": True,
            "request_ref": dict(request_ref),
            "run_state": _run_state(definition),
            "stage_receipt_refs": _sorted_refs(receipt_refs),
            "strategy_id": request["strategy_id"],
            "trade": False,
            "version": FORWARD_RUN_VERSION,
        }
    )


def _validate_run_document(
    run: Mapping[str, Any],
    *,
    definition: ProfileDefinition,
    request: Mapping[str, Any],
    request_ref: Mapping[str, str],
) -> None:
    try:
        validate_semantic_sha(run)
    except Exception as exc:
        raise ForwardEvidenceError("run_semantic") from exc
    if (
        run.get("version") != FORWARD_RUN_VERSION
        or run.get("request_ref") != request_ref
        or run.get("execution_outcome") != ExecutionOutcome.SUCCEEDED.value
        or run.get("completeness") not in {Completeness.COMPLETE.value, Completeness.PARTIAL.value}
        or run.get("authority") != NO_AUTHORITY
        or run.get("broker") is not False
        or run.get("execution") is not False
        or run.get("mainline_authority") is not False
        or run.get("order") is not False
        or run.get("research_only") is not True
        or run.get("run_state") != _run_state(definition)
        or run.get("trade") is not False
        or run.get("cutoff") != request["cutoff"]
        or run.get("recorded_at") != request["created_at"]
        or not isinstance(run.get("stage_receipt_refs"), list)
        or not isinstance(run.get("observation_refs"), list)
    ):
        _blocked("run_binding")


def _session_document(
    *,
    definition: ProfileDefinition,
    request: Mapping[str, Any],
    request_ref: Mapping[str, str],
    run_ref: Mapping[str, str],
) -> dict[str, Any]:
    return seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "broker": False,
            "cutoff": request["cutoff"],
            "decision_session": request["decision_session"],
            "execution": False,
            "mainline_authority": False,
            "observation_run_ref": dict(run_ref),
            "order": False,
            "protocol_version": "myquant.v17.v4",
            "published_at": request["created_at"],
            "research_only": True,
            "run_state": _run_state(definition),
            "session_ref_id": (f"forward-observation-session-{request['request_id'][16:]}"),
            "strategy_id": request["strategy_id"],
            "trade": False,
            "version": FORWARD_SESSION_VERSION,
        }
    )


def _validate_session_document(
    session: Mapping[str, Any],
    *,
    definition: ProfileDefinition,
    request: Mapping[str, Any],
    request_ref: Mapping[str, str],
) -> None:
    try:
        validate_semantic_sha(session)
    except Exception as exc:
        raise ForwardEvidenceError("session_semantic") from exc
    if (
        session.get("version") != FORWARD_SESSION_VERSION
        or session.get("decision_session") != request["decision_session"]
        or session.get("cutoff") != request["cutoff"]
        or session.get("published_at") != request["created_at"]
        or session.get("authority") != NO_AUTHORITY
        or session.get("broker") is not False
        or session.get("execution") is not False
        or session.get("mainline_authority") is not False
        or session.get("order") is not False
        or session.get("research_only") is not True
        or session.get("run_state") != _run_state(definition)
        or session.get("trade") is not False
        or not isinstance(session.get("observation_run_ref"), Mapping)
    ):
        _blocked("session_binding")


def _result(
    *,
    session: Mapping[str, Any],
    session_ref: Mapping[str, str],
    run: Mapping[str, Any],
    definition: ProfileDefinition,
    request: Mapping[str, Any],
    stage_receipts: Sequence[Mapping[str, Any]],
    created: bool,
) -> dict[str, Any]:
    state = _run_state(definition)
    return {
        "authority": dict(NO_AUTHORITY),
        "broker": False,
        "created": created,
        "execution": False,
        "mainline_authority": False,
        "lifecycle_labels": _lifecycle_labels(
            list(definition.stages),
            stage_receipts,
        ),
        "order": False,
        "profile": definition.profile.value,
        "research_only": True,
        "request_id": request["request_id"],
        "run_ref": dict(session["observation_run_ref"]),
        "run_state": state,
        "session_ref": dict(session_ref),
        "side_effects": dict(NO_SIDE_EFFECT_FLAGS),
        "trade": False,
    }


def _replay_session(
    *,
    store: _ForwardEvidenceStore,
    session_path: str,
    session_raw: bytes,
    session: Mapping[str, Any],
    definition: ProfileDefinition,
    request: Mapping[str, Any],
    request_ref: Mapping[str, str],
    stage_readers: Mapping[str, StageReader],
    stage_validators: Mapping[str, StageValidator],
    reference_reader: ReferenceReader | None,
    factor_pointer_reread: Callable[..., Any] | None,
) -> dict[str, Any]:
    _validate_session_document(
        session,
        definition=definition,
        request=request,
        request_ref=request_ref,
    )
    for stage, stage_input in _request_stage_input_map(
        request,
        definition=definition,
    ).items():
        _read_request_stage_input(
            context=StageContext(
                workspace_root=store.workspace_root,
                request=request,
                request_ref=request_ref,
                profile=definition.profile,
                stage=stage,
                required=definition.is_required(stage),
                previous_receipt_refs=(),
                previous_output_refs=(),
            ),
            stage_input=stage_input,
        )
    run, _ = _read_ref_document(
        store,
        session["observation_run_ref"],
    )
    _validate_run_document(
        run,
        definition=definition,
        request=request,
        request_ref=request_ref,
    )
    receipts, outputs = _replay_receipt_closure(
        store=store,
        definition=definition,
        request=request,
        request_ref=request_ref,
        receipt_refs=run["stage_receipt_refs"],
        stage_readers=stage_readers,
        stage_validators=stage_validators,
        reference_reader=reference_reader,
    )
    if _sorted_refs(outputs) != list(run["observation_refs"]):
        _blocked("run_observation_refs")
    expected_completeness = (
        Completeness.PARTIAL.value
        if any(
            receipt.get("execution_outcome") == ExecutionOutcome.SUCCEEDED.value
            and receipt.get("completeness") == Completeness.PARTIAL.value
            for receipt in receipts
        )
        else Completeness.COMPLETE.value
    )
    if run.get("completeness") != expected_completeness:
        _blocked("run_completeness")
    final_context = StageContext(
        workspace_root=store.workspace_root,
        request=request,
        request_ref=request_ref,
        profile=definition.profile,
        stage="final",
        required=definition.is_required("final"),
        previous_receipt_refs=tuple(run["stage_receipt_refs"]),
        previous_output_refs=tuple(run["observation_refs"]),
        output_ref=(run["observation_refs"][-1] if run["observation_refs"] else None),
    )
    _factor_pointer_reread(
        factor_pointer_reread,
        request=request,
        context=final_context,
        reference_reader=reference_reader,
    )
    session_ref = _artifact_ref(
        session,
        relative_path=session_path,
        raw=session_raw,
    )
    return _result(
        session=session,
        session_ref=session_ref,
        run=run,
        definition=definition,
        request=request,
        stage_receipts=receipts,
        created=False,
    )


def _normalize_callback_map(
    callbacks: Mapping[str, Any] | None,
    *,
    definition: ProfileDefinition,
    label: str,
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_stage, callback in (callbacks or {}).items():
        try:
            stage = normalize_stage(raw_stage)
        except ValueError as exc:
            raise ForwardEvidenceError(f"{label}_stage") from exc
        if stage not in definition.stages or stage in normalized:
            _blocked(f"{label}_stage")
        if callback is not None and not callable(callback):
            _blocked(label)
        normalized[stage] = callback
    return normalized


def _request_stage_input_map(
    request: Mapping[str, Any],
    *,
    definition: ProfileDefinition,
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in request.get("stage_inputs", []):
        if not isinstance(row, Mapping):
            _blocked("request_stage_inputs")
        try:
            stage = normalize_stage(row.get("stage_id"))
        except ValueError as exc:
            raise ForwardEvidenceError("request_stage_inputs") from exc
        if stage not in definition.stages or stage in result:
            _blocked("request_stage_inputs")
        result[stage] = row
    return result


def run_forward(
    workspace_root: str | os.PathLike[str],
    *,
    request_path: str,
    expected_request_sha256: str | None = None,
    request_sha256: str | None = None,
    stage_callbacks: Mapping[str, StageCallback] | None = None,
    stage_readers: Mapping[str, StageReader] | None = None,
    stage_validators: Mapping[str, StageValidator] | None = None,
    reference_reader: ReferenceReader | None = None,
    factor_pointer_reread: Callable[..., Any] | None = None,
    event_hook: EventHook | None = None,
    disk_free_reader: DiskFreeReader | None = None,
) -> dict[str, Any]:
    """Run one exact request and publish its session reference last.

    ``request_path`` and its exact byte SHA are the only request inputs.  All
    execution functions are injected, so offline tests and callers can replay
    deterministic stages without provider or execution access.
    """

    if (
        expected_request_sha256 is not None
        and request_sha256 is not None
        and expected_request_sha256 != request_sha256
    ):
        _blocked("request_sha_argument_conflict")
    expected_sha = expected_request_sha256 or request_sha256
    if expected_sha is None:
        _blocked("request_sha_absent")
    try:
        expected_sha = require_sha256(
            expected_sha,
            label="request_sha256",
        )
        canonical_path = str(canonical_governed_path(request_path))
    except Exception as exc:
        raise ForwardEvidenceError("request_reference") from exc
    root = Path(workspace_root)
    store = _ForwardEvidenceStore(
        root,
        max_read_bytes=MAX_ARTIFACT_BYTES,
        max_hash_bytes=MAX_ARTIFACT_BYTES,
    )
    request, request_raw = _read_document(
        store,
        canonical_path,
        expected_sha,
    )
    profile, strategy, decision_session = _request_shape(request)
    if canonical_path != _request_path(request["request_id"]):
        _blocked("request_path")
    request_reference = _request_ref(
        request,
        relative_path=canonical_path,
        raw=request_raw,
    )
    definition = profile_definition(profile)
    callbacks = _normalize_callback_map(
        stage_callbacks,
        definition=definition,
        label="stage_callback",
    )
    readers = _normalize_callback_map(
        stage_readers,
        definition=definition,
        label="stage_reader",
    )
    validators = _normalize_callback_map(
        stage_validators,
        definition=definition,
        label="stage_validator",
    )
    request_stage_inputs = _request_stage_input_map(
        request,
        definition=definition,
    )
    for stage, stage_input in request_stage_inputs.items():
        if callbacks.get(stage) is not None:
            _blocked("stage_input_callback_conflict")
        _read_request_stage_input(
            context=StageContext(
                workspace_root=root,
                request=request,
                request_ref=request_reference,
                profile=profile,
                stage=stage,
                required=definition.is_required(stage),
                previous_receipt_refs=(),
                previous_output_refs=(),
            ),
            stage_input=stage_input,
        )
    _request_ref_readback(
        root,
        request=request,
        fields=("source_refs", "factor_refs"),
        error_code="request_ref_readback",
    )
    disk_reader = disk_free_reader or _disk_reader
    session_path = _session_path(
        strategy,
        decision_session,
        request["request_id"],
    )
    existing_session = _read_optional_document(store, session_path)
    if existing_session is not None:
        session, session_raw = existing_session
        return _replay_session(
            store=store,
            session_path=session_path,
            session_raw=session_raw,
            session=session,
            definition=definition,
            request=request,
            request_ref=request_reference,
            stage_readers=readers,
            stage_validators=validators,
            reference_reader=reference_reader,
            factor_pointer_reread=factor_pointer_reread,
        )

    receipt_refs: list[Mapping[str, str]] = []
    output_refs: list[Mapping[str, str]] = []
    for stage in definition.stages:
        receipt_ref, output_ref = _execute_stage(
            store=store,
            strategy=strategy,
            definition=definition,
            request=request,
            request_ref=request_reference,
            stage=stage,
            callback=callbacks.get(stage),
            stage_input=request_stage_inputs.get(stage),
            stage_reader=readers.get(stage),
            stage_validator=validators.get(stage),
            reference_reader=reference_reader,
            previous_receipt_refs=receipt_refs,
            previous_output_refs=output_refs,
            disk_free_reader=disk_reader,
            event_hook=event_hook,
        )
        receipt_refs.append(receipt_ref)
        if output_ref is not None:
            output_refs.append(output_ref)

    stage_receipts, replayed_outputs = _replay_receipt_closure(
        store=store,
        definition=definition,
        request=request,
        request_ref=request_reference,
        receipt_refs=receipt_refs,
        stage_readers=readers,
        stage_validators=validators,
        reference_reader=reference_reader,
    )
    if replayed_outputs != [dict(reference) for reference in output_refs]:
        if _sorted_refs(replayed_outputs) != _sorted_refs(output_refs):
            _blocked("transitive_output_replay")
    final_context = StageContext(
        workspace_root=root,
        request=request,
        request_ref=request_reference,
        profile=profile,
        stage="final",
        required=definition.is_required("final"),
        previous_receipt_refs=tuple(receipt_refs),
        previous_output_refs=tuple(output_refs),
        output_ref=output_refs[-1] if output_refs else None,
    )
    run = _run_document(
        definition=definition,
        request=request,
        request_ref=request_reference,
        receipt_refs=receipt_refs,
        output_refs=output_refs,
        stage_receipts=stage_receipts,
    )
    _request_ref_readback(
        root,
        request=request,
        fields=("source_refs",),
        error_code="source_ref_reread",
    )
    run_path = _run_path(strategy, request["request_id"])
    run_raw = _resource_bytes(run)
    run_write = _write_exact(
        store,
        run_path,
        run_raw,
        disk_free_reader=disk_reader,
    )
    run_readback, observed_run_raw = _read_document(
        store,
        run_path,
        run_write.byte_sha256,
    )
    _validate_run_document(
        run_readback,
        definition=definition,
        request=request,
        request_ref=request_reference,
    )
    if event_hook is not None:
        _invoke(event_hook, "after_run", final_context)
    _factor_pointer_reread(
        factor_pointer_reread,
        request=request,
        context=final_context,
        reference_reader=reference_reader,
    )
    run_ref = _artifact_ref(
        run_readback,
        relative_path=run_path,
        raw=observed_run_raw,
    )
    session = _session_document(
        definition=definition,
        request=request,
        request_ref=request_reference,
        run_ref=run_ref,
    )
    session_raw = _resource_bytes(session)
    if event_hook is not None:
        _invoke(event_hook, "before_session", final_context)
    session_write = _write_exact(
        store,
        session_path,
        session_raw,
        disk_free_reader=disk_reader,
    )
    session_readback, observed_session_raw = _read_document(
        store,
        session_path,
        session_write.byte_sha256,
    )
    _validate_session_document(
        session_readback,
        definition=definition,
        request=request,
        request_ref=request_reference,
    )
    session_ref = _artifact_ref(
        session_readback,
        relative_path=session_path,
        raw=observed_session_raw,
    )
    return _result(
        session=session_readback,
        session_ref=session_ref,
        run=run_readback,
        definition=definition,
        request=request,
        stage_receipts=stage_receipts,
        created=session_write.created,
    )


__all__ = [
    "Completeness",
    "DISK_FREE_FLOOR_BYTES",
    "ExecutionOutcome",
    "FORWARD_EVIDENCE_ROOT",
    "FORWARD_REQUEST_ROOT",
    "FORWARD_REQUEST_VERSION",
    "FORWARD_RUN_VERSION",
    "FORWARD_SESSION_VERSION",
    "ForwardEvidenceError",
    "MAX_ARTIFACT_BYTES",
    "NO_SIDE_EFFECT_FLAGS",
    "RUN_STATE_BLOCKED",
    "RUN_STATE_EXPLORE_COMPLETE",
    "RUN_STATE_FORWARD_EVIDENCE_ACTIVE",
    "RUN_STATE_INACTIVE",
    "STAGE_OUTPUT_VERSION",
    "STAGE_RECEIPT_VERSION",
    "StageContext",
    "StageResult",
    "build_forward_request",
    "publish_forward_request",
    "run_forward",
]
