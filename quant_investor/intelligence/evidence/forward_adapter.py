"""Read-only, exact-reference bridge from V4 Observation to I0 evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import os
from pathlib import Path
import stat
from typing import Any, Final

from quant_investor.v17_v4_contract.canonical import (
    canonical_resource_bytes,
    load_canonical_resource,
    validate_semantic_sha,
)
from quant_investor.v17_v4_contract.schema_validation import validate_artifact

from .._core import (
    NO_AUTHORITY,
    EXACT_REF_FIELDS,
    IntelligenceContractError,
    assert_no_authority,
    exact_ref,
    safe_path,
    seal_content_addressed,
    sha256,
    sorted_exact_refs,
    timestamp,
    validate_content_addressed,
)

SESSION_VERSION: Final = "myquant.v17.v4.forward-observation-session-ref.v1"
RUN_VERSION: Final = "myquant.v17.v4.forward-observation-run.v1"
OBSERVATION_VERSIONS: Final = (
    "myquant.v17.v4.factor-universe-observation.v1",
    "myquant.v17.v4.strategy-pool-observation.v1",
)
LABEL_VERSION: Final = "myquant.v17.v4.forward-label.v1"
EVALUATION_VERSION: Final = "myquant.v17.v4.forward-evaluation-receipt.v1"
BUNDLE_VERSION: Final = "myquant.v17.research-intelligence.observation-evidence-bundle.v1"
MAX_FILE_BYTES: Final = 8 * 1024 * 1024
MAX_TOTAL_BYTES: Final = 64 * 1024 * 1024
MAX_PATH_PARTS: Final = 24
MAX_OBSERVATION_REFS: Final = 128
MAX_LABEL_REFS: Final = 5
MAX_EVALUATION_REFS: Final = 64
MAX_CLOSURE_REFS: Final = 512
MAX_CLOSURE_DEPTH: Final = 16
VERSION_PREFIXES: Final = {
    SESSION_VERSION: ("results/v17_v4_shadow/forward_evidence/",),
    RUN_VERSION: ("results/v17_v4_shadow/forward_evidence/",),
    OBSERVATION_VERSIONS[0]: (
        "results/v17_v4_shadow/forward_observations/",
        "data/private/v17_v4_runs/forward_observations/",
    ),
    OBSERVATION_VERSIONS[1]: (
        "results/v17_v4_shadow/forward_observations/",
        "data/private/v17_v4_runs/forward_observations/",
    ),
    LABEL_VERSION: (
        "results/v17_v4_shadow/forward_labels/",
        "data/private/v17_v4_runs/forward_labels/",
    ),
    EVALUATION_VERSION: (
        "results/v17_v4_shadow/forward_evaluations/",
        "data/private/v17_v4_runs/forward_evaluations/",
    ),
}
IDENTITY_FIELDS: Final = {
    SESSION_VERSION: "session_ref_id",
    RUN_VERSION: "observation_run_id",
    OBSERVATION_VERSIONS[0]: "observation_id",
    OBSERVATION_VERSIONS[1]: "observation_id",
    LABEL_VERSION: "label_id",
    EVALUATION_VERSION: "receipt_id",
    "myquant.v17.v4.forward-run-request.v1": "request_id",
    "myquant.v17.v4.forward-stage-output.v1": "output_id",
    "myquant.v17.v4.forward-stage-receipt.v1": "receipt_id",
    "myquant.v17.v4.forward-evidence-origin-inventory.v1": "inventory_id",
    "myquant.v17.v4.existing-factor-inventory.v1": "inventory_id",
    "myquant.v17.v4.factor-definition.v1": "factor_id",
    "myquant.v17.v4.label-market.v1": "source_id",
    "myquant.v17.v4.research-source.v1": "source_id",
}
REGISTERED_CLOSURE_VERSIONS: Final = frozenset(
    {
        "myquant.v17.v4.existing-factor-inventory.v1",
        "myquant.v17.v4.forward-evidence-origin-inventory.v1",
        "myquant.v17.v4.forward-run-request.v1",
        "myquant.v17.v4.forward-stage-output.v1",
        "myquant.v17.v4.forward-stage-receipt.v1",
    }
)
TERMINAL_CLOSURE_VERSIONS: Final = frozenset(
    {
        "myquant.v17.v4.factor-definition.v1",
        "myquant.v17.v4.label-market.v1",
        "myquant.v17.v4.research-source.v1",
    }
)
CLOSURE_VERSION_PREFIXES: Final = {
    "myquant.v17.v4.existing-factor-inventory.v1": ("results/v17_v4_shadow/forward_evaluations/",),
    "myquant.v17.v4.factor-definition.v1": ("data/private/v17_v4_sources/factors/",),
    "myquant.v17.v4.forward-evidence-origin-inventory.v1": (
        "results/v17_v4_shadow/forward_evaluations/",
    ),
    "myquant.v17.v4.forward-run-request.v1": ("data/private/v17_v4_runs/forward_requests/",),
    "myquant.v17.v4.forward-stage-output.v1": ("results/v17_v4_shadow/forward_evidence/",),
    "myquant.v17.v4.forward-stage-receipt.v1": ("results/v17_v4_shadow/forward_evidence/",),
    "myquant.v17.v4.label-market.v1": ("data/private/v17_v4_sources/",),
    "myquant.v17.v4.research-source.v1": ("data/private/v17_v4_sources/",),
}
ALLOWED_CHILD_VERSIONS: Final = {
    SESSION_VERSION: frozenset({RUN_VERSION}),
    RUN_VERSION: frozenset(
        {
            "myquant.v17.v4.forward-run-request.v1",
            "myquant.v17.v4.forward-stage-output.v1",
            "myquant.v17.v4.forward-stage-receipt.v1",
        }
    ),
    OBSERVATION_VERSIONS[0]: frozenset(
        {
            "myquant.v17.v4.factor-definition.v1",
            "myquant.v17.v4.forward-run-request.v1",
            "myquant.v17.v4.research-source.v1",
        }
    ),
    OBSERVATION_VERSIONS[1]: frozenset(
        {
            "myquant.v17.v4.factor-definition.v1",
            "myquant.v17.v4.forward-run-request.v1",
            "myquant.v17.v4.research-source.v1",
        }
    ),
    LABEL_VERSION: frozenset({RUN_VERSION, "myquant.v17.v4.label-market.v1"}),
    EVALUATION_VERSION: frozenset(
        {
            LABEL_VERSION,
            RUN_VERSION,
            "myquant.v17.v4.existing-factor-inventory.v1",
            "myquant.v17.v4.forward-evidence-origin-inventory.v1",
        }
    ),
    "myquant.v17.v4.forward-run-request.v1": frozenset(
        {
            "myquant.v17.v4.factor-definition.v1",
            "myquant.v17.v4.research-source.v1",
        }
    ),
    "myquant.v17.v4.forward-stage-receipt.v1": frozenset(
        {
            "myquant.v17.v4.forward-run-request.v1",
            "myquant.v17.v4.forward-stage-output.v1",
        }
    ),
    "myquant.v17.v4.forward-stage-output.v1": frozenset(
        {
            "myquant.v17.v4.forward-run-request.v1",
            "myquant.v17.v4.forward-stage-receipt.v1",
        }
    ),
    "myquant.v17.v4.forward-evidence-origin-inventory.v1": frozenset(
        {LABEL_VERSION, "myquant.v17.v4.forward-run-request.v1"}
    ),
    "myquant.v17.v4.existing-factor-inventory.v1": frozenset(
        {
            *OBSERVATION_VERSIONS,
            "myquant.v17.v4.factor-definition.v1",
            "myquant.v17.v4.forward-run-request.v1",
            "myquant.v17.v4.research-source.v1",
        }
    ),
    "myquant.v17.v4.factor-definition.v1": frozenset(),
    "myquant.v17.v4.label-market.v1": frozenset(),
    "myquant.v17.v4.research-source.v1": frozenset(),
}


class ExactArtifactReader:
    """Bounded descriptor-relative reader with no symlink or hardlink traversal."""

    def __init__(self, workspace_root: str) -> None:
        if type(workspace_root) is not str or not os.path.isabs(workspace_root):
            raise IntelligenceContractError("workspace_root must be absolute")
        root_path = Path(workspace_root)
        try:
            root_lstat = root_path.lstat()
        except OSError as exc:
            raise IntelligenceContractError("workspace_root cannot be inspected") from exc
        if stat.S_ISLNK(root_lstat.st_mode) or not stat.S_ISDIR(root_lstat.st_mode):
            raise IntelligenceContractError("workspace_root must be a real directory")
        self._root = workspace_root
        self._owner = os.geteuid()
        self._total_bytes = 0
        self._cache: dict[tuple[str, str], bytes] = {}
        self._path_hashes: dict[str, str] = {}

    @property
    def total_bytes(self) -> int:
        """Return unique verified bytes admitted by this reader."""

        return self._total_bytes

    def _validate_stat(self, observed: os.stat_result, *, directory: bool) -> None:
        expected_kind = stat.S_ISDIR if directory else stat.S_ISREG
        if not expected_kind(observed.st_mode):
            raise IntelligenceContractError("governed path has an invalid file type")
        if observed.st_uid != self._owner or observed.st_mode & 0o022:
            raise IntelligenceContractError("governed path ownership/mode is unsafe")
        if not directory and observed.st_nlink != 1:
            raise IntelligenceContractError("hard-linked governed artifacts are rejected")

    @staticmethod
    def _assert_case(parent_fd: int, component: str) -> None:
        matches = [
            name for name in os.listdir(parent_fd) if name.casefold() == component.casefold()
        ]
        if matches != [component]:
            raise IntelligenceContractError("case-fold drift or ambiguity detected")

    def _bounded_read(self, file_fd: int) -> tuple[bytes, os.stat_result]:
        before = os.fstat(file_fd)
        self._validate_stat(before, directory=False)
        if before.st_size > MAX_FILE_BYTES:
            raise IntelligenceContractError("governed artifact exceeds file limit")
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(file_fd, min(1024 * 1024, MAX_FILE_BYTES + 1 - observed))
            if not chunk:
                break
            observed += len(chunk)
            if observed > MAX_FILE_BYTES:
                raise IntelligenceContractError("governed artifact exceeds file limit")
            chunks.append(chunk)
        self._total_bytes += observed
        if self._total_bytes > MAX_TOTAL_BYTES:
            raise IntelligenceContractError("artifact closure exceeds total byte limit")
        return b"".join(chunks), before

    def read(self, relative_path: str, expected_sha256: str) -> bytes:
        path = safe_path(relative_path, label="relative_path")
        expected = sha256(expected_sha256, label="expected_sha256")
        prior = self._path_hashes.setdefault(path, expected)
        if prior != expected:
            raise IntelligenceContractError("same path declares different byte SHAs")
        cache_key = (path, expected)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        parts = path.split("/")
        if len(parts) > MAX_PATH_PARTS:
            raise IntelligenceContractError("governed path exceeds component limit")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        directory_flags = flags | getattr(os, "O_DIRECTORY", 0)
        descriptors: list[int] = []
        try:
            parent_fd = os.open(self._root, directory_flags)
            descriptors.append(parent_fd)
            self._validate_stat(os.fstat(parent_fd), directory=True)
            for component in parts[:-1]:
                self._assert_case(parent_fd, component)
                child_fd = os.open(component, directory_flags, dir_fd=parent_fd)
                descriptors.append(child_fd)
                self._validate_stat(os.fstat(child_fd), directory=True)
                parent_fd = child_fd
            self._assert_case(parent_fd, parts[-1])
            file_fd = os.open(parts[-1], flags, dir_fd=parent_fd)
            descriptors.append(file_fd)
            raw, before = self._bounded_read(file_fd)
            after = os.fstat(file_fd)
            path_after = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
            identity_fields = (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
            if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
                raise IntelligenceContractError("artifact identity drifted during read")
            if any(
                getattr(after, field) != getattr(path_after, field) for field in identity_fields
            ):
                raise IntelligenceContractError("artifact path identity drifted during read")
            if hashlib.sha256(raw).hexdigest() != expected:
                raise IntelligenceContractError("artifact byte SHA mismatch")
            self._cache[cache_key] = raw
            return raw
        except OSError as exc:
            raise IntelligenceContractError("governed artifact read failed") from exc
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)


def _artifact_dict(validated: Any) -> dict[str, Any]:
    if hasattr(validated, "as_dict"):
        return dict(validated.as_dict())
    if type(validated) is dict:
        return dict(validated)
    raise IntelligenceContractError("V4 validator returned an unsupported artifact")


def _load_document(
    reader: ExactArtifactReader,
    *,
    relative_path: str,
    byte_sha256: str,
    expected_version: str,
) -> tuple[dict[str, Any], bytes]:
    prefixes = VERSION_PREFIXES.get(expected_version, ())
    if not any(relative_path.startswith(prefix) for prefix in prefixes):
        raise IntelligenceContractError("artifact path is outside its version allowlist")
    raw = reader.read(relative_path, byte_sha256)
    try:
        payload = load_canonical_resource(
            raw,
            label=expected_version,
            max_bytes=MAX_FILE_BYTES,
        )
        if type(payload) is not dict or payload.get("version") != expected_version:
            raise IntelligenceContractError("artifact version mismatch")
        return _artifact_dict(validate_artifact(payload)), raw
    except IntelligenceContractError:
        raise
    except Exception as exc:
        raise IntelligenceContractError("V4 artifact validation failed") from exc


def _load_ref(
    reader: ExactArtifactReader,
    value: Mapping[str, Any],
    *,
    expected_versions: Sequence[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    reference = exact_ref(
        value,
        label="artifact_ref",
        expected_versions=expected_versions,
    )
    document, raw = _load_document(
        reader,
        relative_path=reference["relative_path"],
        byte_sha256=reference["byte_sha256"],
        expected_version=reference["artifact_version"],
    )
    identity_field = IDENTITY_FIELDS[reference["artifact_version"]]
    if (
        document.get(identity_field) != reference["artifact_id"]
        or document.get("semantic_sha256") != reference["semantic_sha256"]
        or document.get("strategy_id") != reference["strategy_id"]
        or document.get("cutoff") != reference["cutoff"]
        or canonical_resource_bytes(document) != raw
    ):
        raise IntelligenceContractError("artifact reference/document binding mismatch")
    return document, reference


def _same_run_binding(document: Mapping[str, Any], *, run: Mapping[str, Any]) -> None:
    if document.get("strategy_id") != run.get("strategy_id") or document.get(
        "decision_session"
    ) != run.get("decision_session"):
        raise IntelligenceContractError("artifact strategy/session binding mismatch")


def _ref_key(value: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(value.get("artifact_version")),
        str(value.get("relative_path")),
        str(value.get("byte_sha256")),
    )


def _find_exact_refs(value: Any) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    if type(value) is dict:
        if set(value) == EXACT_REF_FIELDS:
            refs.append(exact_ref(value, label="closure_ref"))
        else:
            for child in value.values():
                refs.extend(_find_exact_refs(child))
    elif type(value) is list:
        for child in value:
            refs.extend(_find_exact_refs(child))
    return refs


def _load_closure_document(
    reader: ExactArtifactReader,
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    ref = exact_ref(reference, label="closure_ref")
    version = ref["artifact_version"]
    prefixes = CLOSURE_VERSION_PREFIXES.get(version)
    if prefixes is None or not any(ref["relative_path"].startswith(prefix) for prefix in prefixes):
        raise IntelligenceContractError("closure ref version/path is outside the allowlist")
    raw = reader.read(ref["relative_path"], ref["byte_sha256"])
    try:
        document = load_canonical_resource(
            raw,
            label=ref["artifact_version"],
            max_bytes=MAX_FILE_BYTES,
        )
        normalized = validate_semantic_sha(document)
        if version in REGISTERED_CLOSURE_VERSIONS:
            normalized = _artifact_dict(validate_artifact(normalized))
        elif version in TERMINAL_CLOSURE_VERSIONS:
            identity_field = IDENTITY_FIELDS[version]
            if set(normalized) != {
                "cutoff",
                identity_field,
                "semantic_sha256",
                "strategy_id",
                "version",
            }:
                raise IntelligenceContractError("terminal closure document shape is not closed")
        else:
            raise IntelligenceContractError("closure ref version is not allowlisted")
    except Exception as exc:
        raise IntelligenceContractError("closure document is not canonical and sealed") from exc
    if (
        normalized.get("version") != ref["artifact_version"]
        or normalized.get("strategy_id") != ref["strategy_id"]
        or normalized.get("cutoff") != ref["cutoff"]
        or normalized.get("semantic_sha256") != ref["semantic_sha256"]
        or canonical_resource_bytes(normalized) != raw
    ):
        raise IntelligenceContractError("closure ref/document binding mismatch")
    identity_field = IDENTITY_FIELDS[version]
    if normalized.get(identity_field) != ref["artifact_id"]:
        raise IntelligenceContractError("closure artifact identity mismatch")
    return normalized


def _verify_closure(
    reader: ExactArtifactReader,
    *,
    roots: Sequence[Mapping[str, Any]],
    preverified_refs: Sequence[Mapping[str, Any]],
    closure_refs: Sequence[Mapping[str, Any]],
    as_of: str,
) -> list[dict[str, str]]:
    cutoff = timestamp(as_of, label="as_of")
    supplied = sorted_exact_refs(closure_refs, label="closure_refs")
    if not supplied or len(supplied) > MAX_CLOSURE_REFS:
        raise IntelligenceContractError("closure ref cardinality is invalid")
    supplied_by_key = {_ref_key(ref): ref for ref in supplied}
    if len(supplied_by_key) != len(supplied):
        raise IntelligenceContractError("closure refs contain conflicting duplicates")
    preverified_by_key = {
        _ref_key(ref): exact_ref(ref, label="preverified_ref") for ref in preverified_refs
    }
    if len(preverified_by_key) != len(preverified_refs):
        raise IntelligenceContractError("preverified refs contain locator conflicts")
    visited: dict[tuple[str, str, str], dict[str, str]] = {}
    active: set[tuple[str, str, str]] = set()

    def walk(document: Mapping[str, Any], *, depth: int) -> None:
        if depth > MAX_CLOSURE_DEPTH:
            raise IntelligenceContractError("artifact closure exceeds depth limit")
        parent_version = document.get("version")
        allowed_children = ALLOWED_CHILD_VERSIONS.get(parent_version)
        if allowed_children is None:
            raise IntelligenceContractError("closure parent version is not allowlisted")
        parent_strategy = document.get("strategy_id")
        parent_cutoff = timestamp(document.get("cutoff"), label="closure parent cutoff")
        if parent_cutoff > cutoff:
            raise IntelligenceContractError("closure parent is from the future")
        for child_ref in _find_exact_refs(document):
            if child_ref["artifact_version"] not in allowed_children:
                raise IntelligenceContractError("closure edge target version is not allowlisted")
            if child_ref["strategy_id"] != parent_strategy:
                raise IntelligenceContractError("closure edge strategy mismatch")
            if child_ref["cutoff"] > parent_cutoff or child_ref["cutoff"] > cutoff:
                raise IntelligenceContractError("closure edge contains future evidence")
            child_key = _ref_key(child_ref)
            if child_key in active:
                raise IntelligenceContractError("artifact closure cycle detected")
            if child_key in preverified_by_key:
                if preverified_by_key[child_key] != child_ref:
                    raise IntelligenceContractError("preverified exact ref conflict")
                continue
            if child_key in visited:
                if visited[child_key] != child_ref:
                    raise IntelligenceContractError("recursive exact ref conflict")
                continue
            expected = supplied_by_key.get(child_key)
            if expected != child_ref:
                raise IntelligenceContractError("undeclared or conflicting closure ref")
            visited[child_key] = child_ref
            active.add(child_key)
            walk(_load_closure_document(reader, child_ref), depth=depth + 1)
            active.remove(child_key)

    for root in roots:
        walk(root, depth=0)
    if set(visited) != set(supplied_by_key):
        raise IntelligenceContractError("unused closure refs are rejected")
    return supplied


def _normalize_supplied_refs(
    *,
    observation_refs: Sequence[Mapping[str, Any]],
    label_refs: Sequence[Mapping[str, Any]],
    evaluation_refs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    if not observation_refs or len(observation_refs) > MAX_OBSERVATION_REFS:
        raise IntelligenceContractError("typed observation ref cardinality is invalid")
    if len(label_refs) > MAX_LABEL_REFS or len(evaluation_refs) > MAX_EVALUATION_REFS:
        raise IntelligenceContractError("label/evaluation ref cardinality is invalid")
    observations = sorted_exact_refs(
        observation_refs,
        label="observation_refs",
        expected_versions=OBSERVATION_VERSIONS,
    )
    labels = sorted_exact_refs(
        label_refs,
        label="label_refs",
        expected_versions=(LABEL_VERSION,),
    )
    evaluations = sorted_exact_refs(
        evaluation_refs,
        label="evaluation_refs",
        expected_versions=(EVALUATION_VERSION,),
    )
    path_hashes: dict[str, str] = {}
    for reference in observations + labels + evaluations:
        path = reference["relative_path"]
        prior = path_hashes.setdefault(path, reference["byte_sha256"])
        if prior != reference["byte_sha256"]:
            raise IntelligenceContractError("one path cannot declare multiple byte SHAs")
    return observations, labels, evaluations


def _load_observations(
    reader: ExactArtifactReader,
    references: Sequence[Mapping[str, Any]],
    *,
    run: Mapping[str, Any],
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for reference in references:
        document, _ = _load_ref(
            reader,
            reference,
            expected_versions=OBSERVATION_VERSIONS,
        )
        _same_run_binding(document, run=run)
        if document.get("cutoff") != run.get("cutoff") or document.get("request_ref") != run.get(
            "request_ref"
        ):
            raise IntelligenceContractError("typed observation request/cutoff mismatch")
        documents.append(document)
    return documents


def _load_labels(
    reader: ExactArtifactReader,
    references: Sequence[Mapping[str, Any]],
    *,
    run: Mapping[str, Any],
    run_ref: Mapping[str, Any],
    as_of: str,
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for reference in references:
        document, _ = _load_ref(
            reader,
            reference,
            expected_versions=(LABEL_VERSION,),
        )
        _same_run_binding(document, run=run)
        if (
            document.get("observation_run_ref") != run_ref
            or document.get("cutoff") < run.get("cutoff")
            or document.get("cutoff") > as_of
        ):
            raise IntelligenceContractError("forward label run/cutoff mismatch")
        documents.append(document)
    return documents


def _load_evaluations(
    reader: ExactArtifactReader,
    references: Sequence[Mapping[str, Any]],
    *,
    run: Mapping[str, Any],
    run_ref: Mapping[str, Any],
    as_of: str,
    supplied_labels: Mapping[tuple[str, str, str], Mapping[str, str]],
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for reference in references:
        document, _ = _load_ref(
            reader,
            reference,
            expected_versions=(EVALUATION_VERSION,),
        )
        _same_run_binding(document, run=run)
        if (
            document.get("observation_run_ref") != run_ref
            or document.get("cutoff") < run.get("cutoff")
            or document.get("cutoff") > as_of
            or document.get("recorded_at") > as_of
        ):
            raise IntelligenceContractError("evaluation receipt run/cutoff mismatch")
        for value in document.get("label_refs", []):
            label_ref = exact_ref(
                value,
                label="evaluation.label_ref",
                expected_versions=(LABEL_VERSION,),
            )
            if supplied_labels.get(_ref_key(label_ref)) != label_ref:
                raise IntelligenceContractError("evaluation label closure was not supplied exactly")
        documents.append(document)
    return documents


def build_observation_evidence_bundle(
    *,
    workspace_root: str,
    session_relative_path: str,
    session_byte_sha256: str,
    observation_refs: Sequence[Mapping[str, Any]],
    closure_refs: Sequence[Mapping[str, Any]],
    as_of: str,
    label_refs: Sequence[Mapping[str, Any]] = (),
    evaluation_refs: Sequence[Mapping[str, Any]] = (),
    reader: ExactArtifactReader | None = None,
) -> dict[str, Any]:
    """Bind explicitly supplied typed artifacts to one exact V4 session/run closure."""

    cutoff = timestamp(as_of, label="as_of")
    session_path = safe_path(session_relative_path, label="session_relative_path")
    session_sha = sha256(session_byte_sha256, label="session_byte_sha256")
    observations, labels, evaluations = _normalize_supplied_refs(
        observation_refs=observation_refs,
        label_refs=label_refs,
        evaluation_refs=evaluation_refs,
    )

    artifact_reader = ExactArtifactReader(workspace_root) if reader is None else reader
    session, _ = _load_document(
        artifact_reader,
        relative_path=session_path,
        byte_sha256=session_sha,
        expected_version=SESSION_VERSION,
    )
    if session.get("cutoff") > cutoff or session.get("published_at") > cutoff:
        raise IntelligenceContractError("session is not available at the intelligence cutoff")
    session_ref = {
        "artifact_id": str(session["session_ref_id"]),
        "artifact_version": SESSION_VERSION,
        "byte_sha256": session_sha,
        "cutoff": str(session["cutoff"]),
        "relative_path": session_path,
        "semantic_sha256": str(session["semantic_sha256"]),
        "strategy_id": str(session["strategy_id"]),
    }
    run, run_ref = _load_ref(
        artifact_reader,
        session["observation_run_ref"],
        expected_versions=(RUN_VERSION,),
    )
    if (
        session.get("strategy_id") != run.get("strategy_id")
        or session.get("decision_session") != run.get("decision_session")
        or session.get("cutoff") != run.get("cutoff")
        or run_ref != session.get("observation_run_ref")
    ):
        raise IntelligenceContractError("session/run binding mismatch")
    if run.get("cutoff") > cutoff or run.get("recorded_at") > cutoff:
        raise IntelligenceContractError("run is not available at the intelligence cutoff")

    observation_documents = _load_observations(artifact_reader, observations, run=run)
    label_documents = _load_labels(
        artifact_reader,
        labels,
        run=run,
        run_ref=run_ref,
        as_of=cutoff,
    )
    supplied_labels = {_ref_key(reference): reference for reference in labels}
    evaluation_documents = _load_evaluations(
        artifact_reader,
        evaluations,
        run=run,
        run_ref=run_ref,
        as_of=cutoff,
        supplied_labels=supplied_labels,
    )
    verified_closure = _verify_closure(
        artifact_reader,
        roots=[
            session,
            run,
            *observation_documents,
            *label_documents,
            *evaluation_documents,
        ],
        preverified_refs=[session_ref, run_ref, *observations, *labels, *evaluations],
        closure_refs=closure_refs,
        as_of=cutoff,
    )
    authorized_evidence_refs = sorted_exact_refs(
        [*observations, *labels, *evaluations, *verified_closure],
        label="authorized_evidence_refs",
    )

    completeness_values = {str(document.get("completeness")) for document in observation_documents}
    completeness = "COMPLETE" if completeness_values == {"COMPLETE"} else "PARTIAL_OR_UNAVAILABLE"
    return seal_content_addressed(
        {
            "authority": dict(NO_AUTHORITY),
            "authorized_evidence_refs": authorized_evidence_refs,
            "completeness": completeness,
            "evaluation_refs": evaluations,
            "label_refs": labels,
            "limitations": [
                "EXPLICIT_TYPED_REFS_REQUIRED",
                "REQUEST_BOUND_TYPED_OBSERVATIONS_NOT_STAGE_PAYLOAD_DISCOVERED",
            ],
            "observation_refs": observations,
            "production": False,
            "research_only": True,
            "run_ref": run_ref,
            "session_ref": session_ref,
            "timestamp": cutoff,
            "version": BUNDLE_VERSION,
            "verified_closure_refs": verified_closure,
        },
        identity_field="bundle_id",
    )


def validate_observation_evidence_bundle(
    document: Mapping[str, Any],
    *,
    as_of: str,
) -> dict[str, Any]:
    """Validate the closed, source-authorizing summary produced by the adapter."""

    cutoff = timestamp(as_of, label="as_of")
    row = validate_content_addressed(document, identity_field="bundle_id")
    if set(row) != {
        "authority",
        "authorized_evidence_refs",
        "bundle_id",
        "completeness",
        "evaluation_refs",
        "label_refs",
        "limitations",
        "observation_refs",
        "production",
        "research_only",
        "run_ref",
        "semantic_sha256",
        "session_ref",
        "timestamp",
        "verified_closure_refs",
        "version",
    }:
        raise IntelligenceContractError("Observation bundle shape is not closed")
    if row.get("version") != BUNDLE_VERSION:
        raise IntelligenceContractError("Observation bundle version mismatch")
    if timestamp(row.get("timestamp"), label="bundle.timestamp") > cutoff:
        raise IntelligenceContractError("Observation bundle is from the future")
    assert_no_authority(row)
    if row.get("completeness") not in {"COMPLETE", "PARTIAL_OR_UNAVAILABLE"}:
        raise IntelligenceContractError("Observation bundle completeness is invalid")
    if row.get("limitations") != [
        "EXPLICIT_TYPED_REFS_REQUIRED",
        "REQUEST_BOUND_TYPED_OBSERVATIONS_NOT_STAGE_PAYLOAD_DISCOVERED",
    ]:
        raise IntelligenceContractError("Observation bundle limitations drifted")

    session_ref = exact_ref(
        row.get("session_ref"),
        label="session_ref",
        expected_versions=(SESSION_VERSION,),
    )
    run_ref = exact_ref(
        row.get("run_ref"),
        label="run_ref",
        expected_versions=(RUN_VERSION,),
    )
    observations, labels, evaluations = _normalize_supplied_refs(
        observation_refs=row.get("observation_refs"),
        label_refs=row.get("label_refs"),
        evaluation_refs=row.get("evaluation_refs"),
    )
    if (
        session_ref != row.get("session_ref")
        or run_ref != row.get("run_ref")
        or observations != row.get("observation_refs")
        or labels != row.get("label_refs")
        or evaluations != row.get("evaluation_refs")
    ):
        raise IntelligenceContractError("Observation bundle refs are not canonical")
    closure = sorted_exact_refs(row.get("verified_closure_refs"), label="verified_closure_refs")
    if (
        not closure
        or len(closure) > MAX_CLOSURE_REFS
        or closure != row.get("verified_closure_refs")
    ):
        raise IntelligenceContractError("Observation bundle closure refs are invalid")
    authorized = sorted_exact_refs(
        row.get("authorized_evidence_refs"),
        label="authorized_evidence_refs",
    )
    expected_authorized = sorted_exact_refs(
        [*observations, *labels, *evaluations, *closure],
        label="expected_authorized_evidence_refs",
    )
    if authorized != expected_authorized or authorized != row.get("authorized_evidence_refs"):
        raise IntelligenceContractError("Observation bundle authorization closure mismatch")
    if any(ref["strategy_id"] != run_ref["strategy_id"] for ref in authorized):
        raise IntelligenceContractError("Observation bundle strategy closure mismatch")
    if any(ref["cutoff"] > row["timestamp"] for ref in authorized):
        raise IntelligenceContractError("Observation bundle contains future authorized refs")
    return row


def replay_forward_evaluation_inputs(
    *,
    workspace_root: str,
    session_relative_path: str,
    session_byte_sha256: str,
    observation_refs: Sequence[Mapping[str, Any]],
    closure_refs: Sequence[Mapping[str, Any]],
    as_of: str,
    label_refs: Sequence[Mapping[str, Any]] = (),
    evaluation_refs: Sequence[Mapping[str, Any]] = (),
    reader: ExactArtifactReader | None = None,
) -> dict[str, Any]:
    """Replay one origin and return its bundle plus typed validated documents.

    The same descriptor-relative reader may be shared across origins.  Its
    content cache makes repeated exact refs count once against the global byte
    budget while preserving the first verified snapshot for deterministic
    evaluation.
    """

    artifact_reader = ExactArtifactReader(workspace_root) if reader is None else reader
    bundle = build_observation_evidence_bundle(
        workspace_root=workspace_root,
        session_relative_path=session_relative_path,
        session_byte_sha256=session_byte_sha256,
        observation_refs=observation_refs,
        closure_refs=closure_refs,
        as_of=as_of,
        label_refs=label_refs,
        evaluation_refs=evaluation_refs,
        reader=artifact_reader,
    )
    observations, labels, evaluations = _normalize_supplied_refs(
        observation_refs=observation_refs,
        label_refs=label_refs,
        evaluation_refs=evaluation_refs,
    )
    session, _ = _load_document(
        artifact_reader,
        relative_path=session_relative_path,
        byte_sha256=session_byte_sha256,
        expected_version=SESSION_VERSION,
    )
    run, run_ref = _load_ref(
        artifact_reader,
        session["observation_run_ref"],
        expected_versions=(RUN_VERSION,),
    )
    observation_documents = _load_observations(artifact_reader, observations, run=run)
    label_documents = _load_labels(
        artifact_reader,
        labels,
        run=run,
        run_ref=run_ref,
        as_of=as_of,
    )
    label_map = {_ref_key(reference): reference for reference in labels}
    evaluation_documents = _load_evaluations(
        artifact_reader,
        evaluations,
        run=run,
        run_ref=run_ref,
        as_of=as_of,
        supplied_labels=label_map,
    )
    return {
        "bundle": bundle,
        "evaluation_documents": evaluation_documents,
        "label_documents": label_documents,
        "observation_documents": observation_documents,
        "run": run,
        "session": session,
    }


__all__ = [
    "BUNDLE_VERSION",
    "EVALUATION_VERSION",
    "ExactArtifactReader",
    "LABEL_VERSION",
    "OBSERVATION_VERSIONS",
    "RUN_VERSION",
    "SESSION_VERSION",
    "build_observation_evidence_bundle",
    "replay_forward_evaluation_inputs",
    "validate_observation_evidence_bundle",
]
