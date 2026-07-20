#!/usr/bin/env python3
"""Build one private, research-only FactorGovernanceProtocol v4.1 DISCOVERY bundle.

Every participating input is supplied as an absolute path with caller-provided
byte and semantic identities.  A_quant is read only from a pinned Git commit's
object database; neither its worktree nor its Python code is executed.  This
entrypoint deliberately has no registry, proposal, apply, WAL, provider,
portfolio, replay, transaction, broker, order, or trade surface.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import quant_investor.factors.governance_cycle_state_v4_1 as cycle_state  # noqa: E402
from quant_investor.factors import (  # noqa: E402
    governance_discovery_readback_v4_1 as publication,
)
import quant_investor.factors.governance_discovery_v4_1 as discovery  # noqa: E402
import quant_investor.factors.governance_screening_v4 as screening  # noqa: E402
from quant_investor.factors import (  # noqa: E402
    governance_source_readback_v4_1 as predecessor_readback,
)
import quant_investor.factors.governance_source_v4_1 as predecessor_source  # noqa: E402


PINNED_AQUANT_COMMIT = "4424dcecc384f614b0e9fd5e36cf094e9244bad5"
EXPECTED_AQUANT_GIT_TOP_LEVEL = Path("/Users/maxwell/mySpace")
DEFAULT_GIT_EXECUTABLE = Path("/usr/bin/git")
EXPECTED_BASE_CANDIDATE_COUNT = 230
EXPECTED_AQUANT_CANDIDATE_COUNT = 100

EXPECTED_ORDERED_NAMES_SEMANTIC_SHA256 = (
    "64078f603d4484cb7f2dd167275ab25e790e10613ac7046f6da66f541d32bbab"
)
EXPECTED_COMPATIBLE_NAMES_SEMANTIC_SHA256 = (
    "38e1d7268028436dfb23deb0543816030d97adab65997babe8361d0646e97f6e"
)
EXPECTED_ALIAS_NAMES_SEMANTIC_SHA256 = (
    "abb938af17b0875f72d994697de3c3a20209ad862b5fac6c535c91b0915c597d"
)
EXPECTED_AQUANT_ACCOUNTING = {
    "source_idea_count": 100,
    "compatible_count": 43,
    "incompatible_count": 57,
    "new_candidate_count": 37,
    "structural_alias_count": 6,
    "discovery_member_count": 273,
    "selected_count": 267,
    "unselected_count": 6,
}

PREDECESSOR_INPUT_BINDING_FILENAME = "cutoff_input_binding.v4_1.json"
PREDECESSOR_DESIGN_SOURCE_FILENAME = "design_source.v4_1.json"
PREDECESSOR_SOURCE_NODE_FILENAME = "source_chain_node.v4_1.json"
PREDECESSOR_STATE_FILENAME = "cycle_state.precommitted.v4_1.json"
PREDECESSOR_READBACK_FILENAME = "source_readback_report.v4_1.json"
PREDECESSOR_DIRECTORY_ENTRIES = frozenset(
    {
        ".lock",
        PREDECESSOR_INPUT_BINDING_FILENAME,
        PREDECESSOR_DESIGN_SOURCE_FILENAME,
        PREDECESSOR_SOURCE_NODE_FILENAME,
        PREDECESSOR_STATE_FILENAME,
        PREDECESSOR_READBACK_FILENAME,
    }
)

FIXED_BLOCKERS = (
    "formal_catalog_not_materialized",
    "holdout_not_appended",
    "statistics_not_run",
    "verified_v4_replay_not_run",
    "qualification_not_evaluated",
)
FIXED_NOT_RUN_STATUSES = {
    field: discovery.NOT_RUN for field in discovery.MEASUREMENT_STATUS_FIELDS
}
FIXED_SIDE_EFFECTS = {
    field: False for field in discovery.SIDE_EFFECT_FIELDS
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_OID40_RE = re.compile(r"[0-9a-f]{40}")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}")


@dataclass(frozen=True)
class AquantSourceSpec:
    key: str
    repository_path: str
    expected_blob_oid: str
    expected_raw_sha256: str


AQUANT_SOURCE_SPECS = (
    AquantSourceSpec(
        key="generator",
        repository_path="A_quant/scripts/run_factor_batch_screen.py",
        expected_blob_oid="6de605a9ebc6c4b1f9cd730c5ffe350d11e8aef9",
        expected_raw_sha256=(
            "011b754f01db87d04f1b924025b65c6c49999de7d20cc924cc9e22812f74c312"
        ),
    ),
    AquantSourceSpec(
        key="expression",
        repository_path="A_quant/app/factor_sandbox/expression.py",
        expected_blob_oid="d8acdd565b8bba27ffaf02ec44a7029ec63e832d",
        expected_raw_sha256=(
            "df93622a33309aa28d065d6e8fd366de1ebf7d2be600b26170084f727a7dc936"
        ),
    ),
    AquantSourceSpec(
        key="operators",
        repository_path="A_quant/app/factor_sandbox/operators.py",
        expected_blob_oid="bd3365fb994a941caa62913156c2a6fb172bd697",
        expected_raw_sha256=(
            "367f0c68a1e6f8c2e7f0fe168c91e23d77689f101fd203889d5c5b1c2bdb80a1"
        ),
    ),
    AquantSourceSpec(
        key="matrix_dataset",
        repository_path="A_quant/app/factor_sandbox/matrix_dataset.py",
        expected_blob_oid="ef6f6d0a408176a0e3151d619d097c5190d60ef8",
        expected_raw_sha256=(
            "eab9ba96576d040622ae170fc36689a4ee62b64f13a91ae0efe9ff9cd8942547"
        ),
    ),
    AquantSourceSpec(
        key="schemas",
        repository_path="A_quant/app/data/schemas.py",
        expected_blob_oid="2bc56bfea1e0dd6a31a230b72422e0238312f20d",
        expected_raw_sha256=(
            "848f324ada44b1d6e4c944d7e156fa9901779da797c51d8076e7b56db0a55817"
        ),
    ),
    AquantSourceSpec(
        key="time_alignment_policy",
        repository_path="A_quant/docs/factor_time_alignment_policy.md",
        expected_blob_oid="ef4de17343b3d24bbb1560537bf0c4354b60ebdb",
        expected_raw_sha256=(
            "e913ac9909927652b37571ee47c15d06e77b28227e1ee1f588179b435471f083"
        ),
    ),
)


class FactorV4_1DiscoveryRunnerError(ValueError):
    """Raised when DISCOVERY cannot be built without weakening a boundary."""


@dataclass(frozen=True)
class BoundJsonArtifact:
    absolute_path: str
    raw_sha256: str
    semantic_sha256: str
    value: dict[str, Any]


@dataclass(frozen=True)
class BoundFile:
    absolute_path: str
    raw_sha256: str
    size_bytes: int


@dataclass(frozen=True)
class BoundGitSource:
    key: str
    repository_path: str
    blob_oid: str
    raw_sha256: str
    size_bytes: int
    data: bytes


@dataclass(frozen=True)
class BoundGitObjects:
    repository_top_level: str
    git_dir: str
    object_dir: str
    pinned_commit: str
    sources: tuple[BoundGitSource, ...]


@dataclass(frozen=True)
class BoundDiscoveryInputs:
    predecessor_input_binding: BoundJsonArtifact
    predecessor_design_source: BoundJsonArtifact
    predecessor_state: BoundJsonArtifact
    predecessor_source_node: BoundJsonArtifact
    predecessor_readback_report: BoundJsonArtifact
    base_ontology: BoundJsonArtifact
    base_catalog: BoundJsonArtifact
    local_evaluator: BoundFile
    git_objects: BoundGitObjects
    code_bindings: tuple[BoundFile, ...]
    stable_identity_sha256: str


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, TypeError, ValueError) as exc:
        raise FactorV4_1DiscoveryRunnerError(
            f"value is not canonical finite JSON: {exc}"
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise FactorV4_1DiscoveryRunnerError(
            f"{label} must be an exact lowercase SHA-256"
        )
    return value


def _oid40(value: object, label: str) -> str:
    if type(value) is not str or _OID40_RE.fullmatch(value) is None:
        raise FactorV4_1DiscoveryRunnerError(
            f"{label} must be an exact lowercase 40-hex Git object ID"
        )
    return value


def _safe_id(value: object, label: str) -> str:
    if (
        type(value) is not str
        or _SAFE_ID_RE.fullmatch(value) is None
        or ".." in value
    ):
        raise FactorV4_1DiscoveryRunnerError(
            f"{label} must be an exact safe non-empty path segment"
        )
    return value


def _absolute_path(value: object, label: str) -> Path:
    if type(value) is not str or not value:
        raise FactorV4_1DiscoveryRunnerError(
            f"{label} must be an absolute path"
        )
    path = Path(value)
    if not path.is_absolute():
        raise FactorV4_1DiscoveryRunnerError(
            f"{label} must be an absolute path"
        )
    return path


def _regular_file_bytes(
    value: object,
    label: str,
    *,
    require_private: bool,
) -> tuple[Path, bytes]:
    path = _absolute_path(value, label)
    try:
        resolved = path.resolve(strict=True)
        before = path.lstat()
    except OSError as exc:
        raise FactorV4_1DiscoveryRunnerError(f"{label} cannot be read: {exc}") from exc
    if resolved != path or not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise FactorV4_1DiscoveryRunnerError(
            f"{label} must be a resolved regular non-symlink file"
        )
    if before.st_nlink != 1:
        raise FactorV4_1DiscoveryRunnerError(f"{label} hard-link count must be one")
    if require_private:
        if stat.S_IMODE(before.st_mode) != 0o600:
            raise FactorV4_1DiscoveryRunnerError(f"{label} mode must be 0600")
        if before.st_uid != os.getuid():
            raise FactorV4_1DiscoveryRunnerError(
                f"{label} must be owned by the current user"
            )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FactorV4_1DiscoveryRunnerError(
            f"{label} cannot be securely opened: {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        identity_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_uid",
            "st_gid",
            "st_size",
        )
        if any(
            getattr(before, field) != getattr(opened, field)
            for field in identity_fields
        ):
            raise FactorV4_1DiscoveryRunnerError(
                f"{label} identity changed while opening"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        data = b"".join(chunks)
        after_fd = os.fstat(descriptor)
        if any(
            getattr(opened, field) != getattr(after_fd, field)
            for field in identity_fields
        ):
            raise FactorV4_1DiscoveryRunnerError(
                f"{label} identity changed while reading"
            )
        if len(data) != after_fd.st_size:
            raise FactorV4_1DiscoveryRunnerError(
                f"{label} size changed while reading"
            )
    finally:
        os.close(descriptor)
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise FactorV4_1DiscoveryRunnerError(
            f"{label} disappeared after reading: {exc}"
        ) from exc
    if any(
        getattr(after_fd, field) != getattr(after_path, field)
        for field in identity_fields
    ):
        raise FactorV4_1DiscoveryRunnerError(
            f"{label} path identity changed while reading"
        )
    return path, data


def _decode_json_object(data: bytes, label: str) -> dict[str, Any]:
    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FactorV4_1DiscoveryRunnerError(
                    f"{label} contains duplicate JSON field: {key}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise FactorV4_1DiscoveryRunnerError(
            f"{label} contains non-finite JSON constant: {value}"
        )

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except FactorV4_1DiscoveryRunnerError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FactorV4_1DiscoveryRunnerError(
            f"{label} is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(value, Mapping):
        raise FactorV4_1DiscoveryRunnerError(f"{label} must contain a JSON object")
    payload = dict(value)
    # Also rejects NaN/Infinity and non-string mapping keys on round-trip.
    _canonical_json_bytes(payload)
    return payload


def _bind_json_artifact(
    *,
    path_value: object,
    expected_raw_sha256: object,
    expected_semantic_sha256: object,
    label: str,
    validator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    semantic_field: str | None,
    trailing_newline: bool,
    semantic_calculator: Callable[[Mapping[str, Any]], str] | None = None,
) -> BoundJsonArtifact:
    path, raw = _regular_file_bytes(path_value, label, require_private=True)
    expected_raw = _sha256(expected_raw_sha256, f"expected {label} raw SHA-256")
    actual_raw = _sha256_bytes(raw)
    if actual_raw != expected_raw:
        raise FactorV4_1DiscoveryRunnerError(f"{label} raw SHA-256 mismatch")
    value = _decode_json_object(raw, label)
    try:
        normalized = dict(validator(value))
    except (TypeError, ValueError) as exc:
        raise FactorV4_1DiscoveryRunnerError(
            f"{label} validation failed: {exc}"
        ) from exc
    expected_bytes = _canonical_json_bytes(normalized) + (
        b"\n" if trailing_newline else b""
    )
    if raw != expected_bytes:
        raise FactorV4_1DiscoveryRunnerError(
            f"{label} bytes are not in the exact canonical representation"
        )
    if semantic_field is None:
        if semantic_calculator is None:
            raise FactorV4_1DiscoveryRunnerError(
                f"{label} semantic calculator is missing"
            )
        semantic = semantic_calculator(normalized)
    else:
        semantic = normalized.get(semantic_field)
    expected_semantic = _sha256(
        expected_semantic_sha256,
        f"expected {label} semantic SHA-256",
    )
    if semantic != expected_semantic:
        raise FactorV4_1DiscoveryRunnerError(f"{label} semantic SHA-256 mismatch")
    return BoundJsonArtifact(
        absolute_path=str(path),
        raw_sha256=actual_raw,
        semantic_sha256=expected_semantic,
        value=normalized,
    )


def _validate_predecessor_source_node(
    value: Mapping[str, Any], *, cycle_id: str
) -> dict[str, Any]:
    payload = dict(value)
    expected_fields = {
        "schema_version",
        "protocol_version",
        "cycle_id",
        "snapshot_id",
        "cutoff_date",
        "input_binding_semantic_sha256",
        "design_source_root_sha256",
        "cutoff_session_scope_semantic_sha256",
        "source_binding_sha256",
        "out_of_bound_calendar_nonparticipating",
        "serving_inventory_eligibility_prohibited",
        "semantic_sha256",
    }
    if set(payload) != expected_fields:
        raise FactorV4_1DiscoveryRunnerError(
            "predecessor source-node fields are not exact"
        )
    if payload.get("schema_version") != "factor-governance-cutoff-source-node.v4.1":
        raise FactorV4_1DiscoveryRunnerError("predecessor source-node schema mismatch")
    if payload.get("protocol_version") != "v4":
        raise FactorV4_1DiscoveryRunnerError(
            "predecessor source-node protocol mismatch"
        )
    if payload.get("cycle_id") != cycle_id:
        raise FactorV4_1DiscoveryRunnerError("predecessor source-node cycle mismatch")
    observed = _sha256(
        payload.get("semantic_sha256"),
        "predecessor source-node semantic SHA-256",
    )
    base = dict(payload)
    base.pop("semantic_sha256")
    # The frozen PRECOMMITTED source contract predates DISCOVERY's artifact
    # encoding and seals semantics with its canonical trailing newline.
    if _sha256_bytes(_canonical_json_bytes(base) + b"\n") != observed:
        raise FactorV4_1DiscoveryRunnerError(
            "predecessor source-node self hash mismatch"
        )
    return payload


def _validate_predecessor_input_binding(
    value: Mapping[str, Any], *, cycle_id: str
) -> dict[str, Any]:
    del cycle_id
    payload = dict(value)
    expected_fields = {
        "schema_version",
        "market",
        "snapshot_id",
        "cutoff_date",
        "latest_pointer",
        "snapshot_manifest",
        "components",
        "pit_generation",
        "calendar",
        "table",
        "eligibility_boundary",
        "readiness",
        "side_effects",
    }
    if set(payload) != expected_fields:
        raise FactorV4_1DiscoveryRunnerError(
            "predecessor input-binding fields are not exact"
        )
    if payload.get("schema_version") != "factor-governance-cutoff-input-binding.v4.1":
        raise FactorV4_1DiscoveryRunnerError(
            "predecessor input-binding schema mismatch"
        )
    if payload.get("market") != "CN":
        raise FactorV4_1DiscoveryRunnerError("predecessor input binding must be CN")
    if payload.get("readiness") != "EXPLORATORY_INPUT_BOUND":
        raise FactorV4_1DiscoveryRunnerError(
            "predecessor input-binding readiness mismatch"
        )
    expected_side_effects = {
        "registry": False,
        "wal": False,
        "budget": False,
        "apply": False,
        "broker": False,
        "order": False,
        "trade": False,
        "network": False,
    }
    if payload.get("side_effects") != expected_side_effects:
        raise FactorV4_1DiscoveryRunnerError(
            "predecessor input-binding side effects are not exact"
        )
    return payload


def _validate_predecessor_design_source(
    value: Mapping[str, Any], *, cycle_id: str
) -> dict[str, Any]:
    payload = dict(value)
    expected_fields = {
        "schema_version",
        "cycle_id",
        "snapshot_id",
        "cutoff_date",
        "component_symbols",
        "component_count",
        "component_symbols_semantic_sha256",
        "pit_record_count",
        "pit_records_semantic_sha256",
        "out_of_bound_calendar_nonparticipating",
        "calendar_sessions",
        "calendar_semantic_sha256",
        "session_scope_descriptors",
        "session_scope_mapping_semantic_sha256",
        "historical_table_binding_sha256",
        "historical_source_binding_sha256",
        "exploratory",
        "semantic_sha256",
    }
    if set(payload) != expected_fields:
        raise FactorV4_1DiscoveryRunnerError(
            "predecessor design-source fields are not exact"
        )
    if payload.get("schema_version") != "factor-governance-design-source.v4.1":
        raise FactorV4_1DiscoveryRunnerError(
            "predecessor design-source schema mismatch"
        )
    if payload.get("cycle_id") != cycle_id or payload.get("exploratory") is not True:
        raise FactorV4_1DiscoveryRunnerError(
            "predecessor design-source cycle/exploratory identity mismatch"
        )
    observed = _sha256(
        payload.get("semantic_sha256"),
        "predecessor design-source semantic SHA-256",
    )
    base = dict(payload)
    base.pop("semantic_sha256")
    if predecessor_source.semantic_sha256(base) != observed:
        raise FactorV4_1DiscoveryRunnerError(
            "predecessor design-source self hash mismatch"
        )
    return payload


def _validate_predecessor_readback_report(
    value: Mapping[str, Any], *, cycle_id: str
) -> dict[str, Any]:
    payload = dict(value)
    expected_fields = {
        "schema_version",
        "protocol_version",
        "cycle_id",
        "run_id",
        "readiness",
        "qualification",
        "artifacts",
        "cycle_root_semantic_sha256",
        "state_cas",
        "side_effects",
    }
    if set(payload) != expected_fields:
        raise FactorV4_1DiscoveryRunnerError(
            "predecessor readback-report fields are not exact"
        )
    if (
        payload.get("schema_version") != "factor-governance-source-readback.v4.1"
        or payload.get("protocol_version") != "v4"
        or payload.get("cycle_id") != cycle_id
        or payload.get("readiness") != "EXPLORATORY_PRECOMMITTED"
        or payload.get("qualification") is not False
    ):
        raise FactorV4_1DiscoveryRunnerError(
            "predecessor readback-report identity mismatch"
        )
    _safe_id(payload.get("run_id"), "predecessor run_id")
    expected_side_effects = {
        "registry": False,
        "wal": False,
        "budget": False,
        "apply": False,
        "broker": False,
        "order": False,
        "trade": False,
        "network": False,
    }
    if payload.get("side_effects") != expected_side_effects:
        raise FactorV4_1DiscoveryRunnerError(
            "predecessor readback-report side effects are not exact"
        )
    return payload


def _full_canonical_predecessor_semantic_sha256(value: Mapping[str, Any]) -> str:
    """Use the frozen PRECOMMITTED contract's newline-bearing semantic bytes."""

    return _sha256_bytes(_canonical_json_bytes(dict(value)) + b"\n")


def _predecessor_input_binding_semantic_sha256(
    value: Mapping[str, Any],
) -> str:
    return predecessor_readback.binding_semantic_sha256_v4_1(dict(value))


def _verify_predecessor_directory(
    artifacts: Mapping[str, BoundJsonArtifact],
) -> None:
    expected_names = set(PREDECESSOR_DIRECTORY_ENTRIES) - {".lock"}
    if set(artifacts) != expected_names:
        raise FactorV4_1DiscoveryRunnerError(
            "predecessor artifact filename set is not exact"
        )
    parents = {Path(item.absolute_path).parent for item in artifacts.values()}
    if len(parents) != 1:
        raise FactorV4_1DiscoveryRunnerError(
            "predecessor artifacts must share one exact directory"
        )
    parent = next(iter(parents))
    for filename, item in artifacts.items():
        if Path(item.absolute_path).name != filename:
            raise FactorV4_1DiscoveryRunnerError(
                f"predecessor artifact filename mismatch: {filename}"
            )
    try:
        resolved = parent.resolve(strict=True)
        before = parent.lstat()
    except OSError as exc:
        raise FactorV4_1DiscoveryRunnerError(
            f"predecessor directory cannot be read: {exc}"
        ) from exc
    if (
        resolved != parent
        or parent.is_symlink()
        or not stat.S_ISDIR(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o700
        or before.st_uid != os.getuid()
    ):
        raise FactorV4_1DiscoveryRunnerError(
            "predecessor directory must be resolved owner-only mode 0700"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(parent, flags)
    except OSError as exc:
        raise FactorV4_1DiscoveryRunnerError(
            f"predecessor directory secure open failed: {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise FactorV4_1DiscoveryRunnerError(
                "predecessor directory identity changed while opening"
            )
        entries = set(os.listdir(descriptor))
        after = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_uid) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
        ):
            raise FactorV4_1DiscoveryRunnerError(
                "predecessor directory identity changed while listing"
            )
    finally:
        os.close(descriptor)
    if entries != set(PREDECESSOR_DIRECTORY_ENTRIES):
        missing = sorted(set(PREDECESSOR_DIRECTORY_ENTRIES) - entries)
        extra = sorted(entries - set(PREDECESSOR_DIRECTORY_ENTRIES))
        raise FactorV4_1DiscoveryRunnerError(
            f"predecessor directory entries mismatch: missing={missing};extra={extra}"
        )
    lock_path, lock_bytes = _regular_file_bytes(
        str(parent / ".lock"),
        "predecessor bundle lock",
        require_private=True,
    )
    if lock_path.parent != parent or lock_bytes:
        raise FactorV4_1DiscoveryRunnerError(
            "predecessor bundle lock must be an empty private file"
        )


def _verify_predecessor_cross_bindings(
    *,
    cycle_id: str,
    input_binding: BoundJsonArtifact,
    design_source: BoundJsonArtifact,
    source_node: BoundJsonArtifact,
    state: BoundJsonArtifact,
    readback_report: BoundJsonArtifact,
) -> None:
    binding = input_binding.value
    design = design_source.value
    node = source_node.value
    normalized_state = state.value
    report = readback_report.value
    if (
        design.get("historical_table_binding_sha256")
        != input_binding.semantic_sha256
        or design.get("snapshot_id") != binding.get("snapshot_id")
        or design.get("cutoff_date") != binding.get("cutoff_date")
        or node.get("snapshot_id") != binding.get("snapshot_id")
        or node.get("cutoff_date") != binding.get("cutoff_date")
    ):
        raise FactorV4_1DiscoveryRunnerError(
            "predecessor input/design/source cross-binding mismatch"
        )
    try:
        validated_node = predecessor_readback.validate_cutoff_source_node_v4_1(
            node,
            cycle_id=cycle_id,
            input_binding=binding,
            design_source=design,
            source_binding_sha256=node["source_binding_sha256"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FactorV4_1DiscoveryRunnerError(
            f"predecessor cutoff source-node validation failed: {exc}"
        ) from exc
    if validated_node.get("semantic_sha256") != source_node.semantic_sha256:
        raise FactorV4_1DiscoveryRunnerError(
            "predecessor cutoff source-node semantic identity mismatch"
        )
    cycle_root = predecessor_readback.cycle_root_semantic_sha256_v4_1(
        cycle_id=cycle_id,
        input_binding=binding,
        design_source=design,
    )
    if normalized_state.get("cycle_root_sha256") != cycle_root:
        raise FactorV4_1DiscoveryRunnerError(
            "PRECOMMITTED cycle-root cross-binding mismatch"
        )
    if normalized_state.get("source_chain_node_sha256") != source_node.semantic_sha256:
        raise FactorV4_1DiscoveryRunnerError(
            "PRECOMMITTED source-chain cross-binding mismatch"
        )

    expected_report_descriptors = {
        PREDECESSOR_INPUT_BINDING_FILENAME: {
            "absolute_path": input_binding.absolute_path,
            "sha256": input_binding.raw_sha256,
        },
        PREDECESSOR_DESIGN_SOURCE_FILENAME: {
            "absolute_path": design_source.absolute_path,
            "sha256": design_source.raw_sha256,
        },
        PREDECESSOR_SOURCE_NODE_FILENAME: {
            "absolute_path": source_node.absolute_path,
            "sha256": source_node.raw_sha256,
        },
        PREDECESSOR_STATE_FILENAME: {
            "absolute_path": state.absolute_path,
            "sha256": state.raw_sha256,
        },
    }
    if report.get("artifacts") != expected_report_descriptors:
        raise FactorV4_1DiscoveryRunnerError(
            "predecessor readback report artifact descriptors mismatch"
        )
    expected_run_id = Path(readback_report.absolute_path).parent.name
    if report.get("run_id") != expected_run_id:
        raise FactorV4_1DiscoveryRunnerError(
            "predecessor readback report run/directory identity mismatch"
        )
    if report.get("cycle_root_semantic_sha256") != cycle_root:
        raise FactorV4_1DiscoveryRunnerError(
            "predecessor readback report cycle-root mismatch"
        )
    if report.get("state_cas") != {
        "before": "EMPTY",
        "after": state.raw_sha256,
    }:
        raise FactorV4_1DiscoveryRunnerError(
            "predecessor readback report state CAS mismatch"
        )


def _bind_local_file(
    *, path_value: object, expected_raw_sha256: object, label: str
) -> BoundFile:
    path, raw = _regular_file_bytes(path_value, label, require_private=False)
    expected = _sha256(expected_raw_sha256, f"expected {label} raw SHA-256")
    actual = _sha256_bytes(raw)
    if actual != expected:
        raise FactorV4_1DiscoveryRunnerError(f"{label} raw SHA-256 mismatch")
    return BoundFile(str(path), actual, len(raw))


def sanitized_git_environment(
    inherited: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the sole environment accepted for pinned Git-object reads."""

    # Accepting an inherited mapping is useful for proving in tests that no
    # caller-controlled variable survives.  The Git subprocess itself receives
    # only this fixed minimal environment.
    del inherited
    return {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": "/var/empty",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "TMPDIR": "/private/tmp",
    }


def _run_git(
    *,
    git_executable: Path,
    repository_top_level: Path,
    arguments: Sequence[str],
    environment: Mapping[str, str],
    allow_exit_one: bool = False,
) -> bytes:
    command = [str(git_executable), "-C", str(repository_top_level), *arguments]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(environment),
            shell=False,
        )
    except OSError as exc:
        raise FactorV4_1DiscoveryRunnerError(f"Git object read failed: {exc}") from exc
    allowed = {0, 1} if allow_exit_one else {0}
    if completed.returncode not in allowed:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise FactorV4_1DiscoveryRunnerError(
            f"Git object read rejected ({completed.returncode}): {detail}"
        )
    return completed.stdout


def _git_text(
    *,
    git_executable: Path,
    repository_top_level: Path,
    arguments: Sequence[str],
    environment: Mapping[str, str],
    allow_exit_one: bool = False,
) -> str:
    raw = _run_git(
        git_executable=git_executable,
        repository_top_level=repository_top_level,
        arguments=arguments,
        environment=environment,
        allow_exit_one=allow_exit_one,
    )
    try:
        return raw.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise FactorV4_1DiscoveryRunnerError("Git metadata is not UTF-8") from exc


def _expected_git_specs(args: argparse.Namespace) -> tuple[AquantSourceSpec, ...]:
    bound: list[AquantSourceSpec] = []
    by_key = {spec.key: spec for spec in AQUANT_SOURCE_SPECS}
    for key in sorted(by_key):
        authoritative = by_key[key]
        caller_oid = _oid40(
            getattr(args, f"expected_aquant_{key}_blob_oid"),
            f"expected A_quant {key} blob object ID",
        )
        caller_sha = _sha256(
            getattr(args, f"expected_aquant_{key}_sha256"),
            f"expected A_quant {key} raw SHA-256",
        )
        if caller_oid != authoritative.expected_blob_oid:
            raise FactorV4_1DiscoveryRunnerError(
                f"A_quant {key} blob object ID differs from the approved pin"
            )
        if caller_sha != authoritative.expected_raw_sha256:
            raise FactorV4_1DiscoveryRunnerError(
                f"A_quant {key} raw SHA-256 differs from the approved pin"
            )
        bound.append(authoritative)
    return tuple(bound)


def _reject_git_indirection(
    *,
    git_executable: Path,
    top_level: Path,
    git_dir: Path,
    object_dir: Path,
    environment: Mapping[str, str],
) -> None:
    expected_git_dir = top_level / ".git"
    if git_dir != expected_git_dir:
        raise FactorV4_1DiscoveryRunnerError("Git directory override is forbidden")
    if object_dir != git_dir / "objects":
        raise FactorV4_1DiscoveryRunnerError(
            "Git object directory override is forbidden"
        )
    alternates = object_dir / "info" / "alternates"
    if alternates.exists() or alternates.is_symlink():
        raise FactorV4_1DiscoveryRunnerError(
            "Git alternate object database is forbidden"
        )
    worktree_config = git_dir / "config.worktree"
    if worktree_config.exists() or worktree_config.is_symlink():
        raise FactorV4_1DiscoveryRunnerError(
            "Git per-worktree config is forbidden"
        )
    replacements = _git_text(
        git_executable=git_executable,
        repository_top_level=top_level,
        arguments=("for-each-ref", "--format=%(refname)", "refs/replace"),
        environment=environment,
    )
    if replacements:
        raise FactorV4_1DiscoveryRunnerError("Git replacement objects are forbidden")
    local_config = _git_text(
        git_executable=git_executable,
        repository_top_level=top_level,
        arguments=("config", "--local", "--name-only", "--list"),
        environment=environment,
    )
    forbidden_exact = {
        "core.alternaterefscommand",
        "core.alternaterefsprefixes",
        "core.worktree",
        "extensions.partialclone",
    }
    for key in local_config.splitlines():
        normalized = key.strip().lower()
        if (
            normalized in forbidden_exact
            or normalized == "include.path"
            or normalized.startswith("includeif.")
            or normalized.endswith(".promisor")
        ):
            raise FactorV4_1DiscoveryRunnerError(
                f"forbidden Git config override: {normalized}"
            )


def bind_pinned_aquant_git_objects(args: argparse.Namespace) -> BoundGitObjects:
    """Read and bind all six approved A_quant blobs without using its worktree."""

    git_executable = _absolute_path(args.git_executable, "git executable")
    if git_executable != git_executable.resolve(strict=True):
        raise FactorV4_1DiscoveryRunnerError(
            "git executable must be a resolved absolute file"
        )
    if not git_executable.is_file():
        raise FactorV4_1DiscoveryRunnerError("git executable must be a file")

    requested_top = _absolute_path(args.aquant_git_top_level, "A_quant Git top-level")
    try:
        top_level = requested_top.resolve(strict=True)
    except OSError as exc:
        raise FactorV4_1DiscoveryRunnerError(
            f"A_quant Git top-level cannot be resolved: {exc}"
        ) from exc
    expected_top = EXPECTED_AQUANT_GIT_TOP_LEVEL.resolve(strict=True)
    if top_level != expected_top:
        raise FactorV4_1DiscoveryRunnerError(
            "A_quant Git top-level differs from the approved repository"
        )
    pinned_commit = _oid40(args.aquant_pinned_commit, "A_quant pinned commit")
    if pinned_commit != PINNED_AQUANT_COMMIT:
        raise FactorV4_1DiscoveryRunnerError(
            "A_quant commit differs from the approved pin"
        )
    specifications = _expected_git_specs(args)
    environment = sanitized_git_environment()

    observed_top = _git_text(
        git_executable=git_executable,
        repository_top_level=top_level,
        arguments=("rev-parse", "--show-toplevel"),
        environment=environment,
    )
    if observed_top != str(top_level):
        raise FactorV4_1DiscoveryRunnerError("Git top-level identity mismatch")
    git_dir_text = _git_text(
        git_executable=git_executable,
        repository_top_level=top_level,
        arguments=("rev-parse", "--absolute-git-dir"),
        environment=environment,
    )
    object_dir_text = _git_text(
        git_executable=git_executable,
        repository_top_level=top_level,
        arguments=("rev-parse", "--path-format=absolute", "--git-path", "objects"),
        environment=environment,
    )
    try:
        git_dir = Path(git_dir_text).resolve(strict=True)
        object_dir = Path(object_dir_text).resolve(strict=True)
    except OSError as exc:
        raise FactorV4_1DiscoveryRunnerError(
            f"Git object database cannot be resolved: {exc}"
        ) from exc
    _reject_git_indirection(
        git_executable=git_executable,
        top_level=top_level,
        git_dir=git_dir,
        object_dir=object_dir,
        environment=environment,
    )
    commit_type = _git_text(
        git_executable=git_executable,
        repository_top_level=top_level,
        arguments=("cat-file", "-t", pinned_commit),
        environment=environment,
    )
    if commit_type != "commit":
        raise FactorV4_1DiscoveryRunnerError("pinned A_quant object is not a commit")

    sources: list[BoundGitSource] = []
    for spec in sorted(specifications, key=lambda item: item.repository_path):
        tree_row = _run_git(
            git_executable=git_executable,
            repository_top_level=top_level,
            arguments=("ls-tree", "-z", pinned_commit, "--", spec.repository_path),
            environment=environment,
        )
        expected_prefix = f"100644 blob {spec.expected_blob_oid}\t".encode("ascii")
        expected_row = expected_prefix + spec.repository_path.encode("utf-8") + b"\0"
        if tree_row != expected_row:
            raise FactorV4_1DiscoveryRunnerError(
                f"A_quant source tree mode/blob/path mismatch: {spec.repository_path}"
            )
        blob_type = _git_text(
            git_executable=git_executable,
            repository_top_level=top_level,
            arguments=("cat-file", "-t", spec.expected_blob_oid),
            environment=environment,
        )
        if blob_type != "blob":
            raise FactorV4_1DiscoveryRunnerError(
                f"A_quant source object is not a blob: {spec.repository_path}"
            )
        data = _run_git(
            git_executable=git_executable,
            repository_top_level=top_level,
            arguments=("cat-file", "-p", spec.expected_blob_oid),
            environment=environment,
        )
        actual_sha = _sha256_bytes(data)
        if actual_sha != spec.expected_raw_sha256:
            raise FactorV4_1DiscoveryRunnerError(
                f"A_quant source raw SHA-256 mismatch: {spec.repository_path}"
            )
        sources.append(
            BoundGitSource(
                key=spec.key,
                repository_path=spec.repository_path,
                blob_oid=spec.expected_blob_oid,
                raw_sha256=actual_sha,
                size_bytes=len(data),
                data=data,
            )
        )
    return BoundGitObjects(
        repository_top_level=str(top_level),
        git_dir=str(git_dir),
        object_dir=str(object_dir),
        pinned_commit=pinned_commit,
        sources=tuple(sources),
    )


def _runtime_code_bindings(args: argparse.Namespace) -> tuple[BoundFile, ...]:
    paths = {
        Path(__file__).resolve(strict=True),
        Path(cycle_state.__file__).resolve(strict=True),
        Path(discovery.__file__).resolve(strict=True),
        Path(publication.__file__).resolve(strict=True),
        Path(screening.__file__).resolve(strict=True),
        Path(predecessor_readback.__file__).resolve(strict=True),
        Path(predecessor_source.__file__).resolve(strict=True),
        _absolute_path(args.local_evaluator_path, "local evaluator"),
    }
    bindings: list[BoundFile] = []
    for path in sorted(paths, key=str):
        resolved, data = _regular_file_bytes(
            str(path), f"participating code file {path}", require_private=False
        )
        bindings.append(BoundFile(str(resolved), _sha256_bytes(data), len(data)))
    return tuple(bindings)


def _stable_identity_payload(bound: BoundDiscoveryInputs) -> dict[str, Any]:
    return {
        "predecessor_input_binding": {
            "absolute_path": bound.predecessor_input_binding.absolute_path,
            "raw_sha256": bound.predecessor_input_binding.raw_sha256,
            "semantic_sha256": bound.predecessor_input_binding.semantic_sha256,
        },
        "predecessor_design_source": {
            "absolute_path": bound.predecessor_design_source.absolute_path,
            "raw_sha256": bound.predecessor_design_source.raw_sha256,
            "semantic_sha256": bound.predecessor_design_source.semantic_sha256,
        },
        "predecessor_state": {
            "absolute_path": bound.predecessor_state.absolute_path,
            "raw_sha256": bound.predecessor_state.raw_sha256,
            "semantic_sha256": bound.predecessor_state.semantic_sha256,
        },
        "predecessor_source_node": {
            "absolute_path": bound.predecessor_source_node.absolute_path,
            "raw_sha256": bound.predecessor_source_node.raw_sha256,
            "semantic_sha256": bound.predecessor_source_node.semantic_sha256,
        },
        "predecessor_readback_report": {
            "absolute_path": bound.predecessor_readback_report.absolute_path,
            "raw_sha256": bound.predecessor_readback_report.raw_sha256,
            "semantic_sha256": bound.predecessor_readback_report.semantic_sha256,
        },
        "base_ontology": {
            "absolute_path": bound.base_ontology.absolute_path,
            "raw_sha256": bound.base_ontology.raw_sha256,
            "semantic_sha256": bound.base_ontology.semantic_sha256,
        },
        "base_catalog": {
            "absolute_path": bound.base_catalog.absolute_path,
            "raw_sha256": bound.base_catalog.raw_sha256,
            "semantic_sha256": bound.base_catalog.semantic_sha256,
        },
        "local_evaluator": {
            "absolute_path": bound.local_evaluator.absolute_path,
            "raw_sha256": bound.local_evaluator.raw_sha256,
            "size_bytes": bound.local_evaluator.size_bytes,
        },
        "git": {
            "repository_top_level": bound.git_objects.repository_top_level,
            "git_dir": bound.git_objects.git_dir,
            "object_dir": bound.git_objects.object_dir,
            "pinned_commit": bound.git_objects.pinned_commit,
            "sources": [
                {
                    "key": item.key,
                    "repository_path": item.repository_path,
                    "blob_oid": item.blob_oid,
                    "raw_sha256": item.raw_sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in bound.git_objects.sources
            ],
        },
        "code_bindings": [
            {
                "absolute_path": item.absolute_path,
                "raw_sha256": item.raw_sha256,
                "size_bytes": item.size_bytes,
            }
            for item in bound.code_bindings
        ],
    }


def _bind_all_inputs(args: argparse.Namespace) -> BoundDiscoveryInputs:
    cycle_id = _safe_id(args.cycle_id, "cycle_id")
    input_binding = _bind_json_artifact(
        path_value=args.predecessor_input_binding_path,
        expected_raw_sha256=args.expected_predecessor_input_binding_sha256,
        expected_semantic_sha256=(
            args.expected_predecessor_input_binding_semantic_sha256
        ),
        label="predecessor input binding",
        validator=lambda value: _validate_predecessor_input_binding(
            value, cycle_id=cycle_id
        ),
        semantic_field=None,
        trailing_newline=True,
        semantic_calculator=_predecessor_input_binding_semantic_sha256,
    )
    design_source = _bind_json_artifact(
        path_value=args.predecessor_design_source_path,
        expected_raw_sha256=args.expected_predecessor_design_source_sha256,
        expected_semantic_sha256=(
            args.expected_predecessor_design_source_semantic_sha256
        ),
        label="predecessor design source",
        validator=lambda value: _validate_predecessor_design_source(
            value, cycle_id=cycle_id
        ),
        semantic_field="semantic_sha256",
        trailing_newline=True,
    )
    state = _bind_json_artifact(
        path_value=args.precommitted_state_path,
        expected_raw_sha256=args.expected_precommitted_state_sha256,
        expected_semantic_sha256=args.expected_precommitted_state_semantic_sha256,
        label="PRECOMMITTED state",
        validator=lambda value: cycle_state.validate_cycle_state_v4_1(
            value,
            expected_cycle_id=cycle_id,
            expected_state=cycle_state.PRECOMMITTED,
        ),
        semantic_field="state_semantic_sha256",
        trailing_newline=True,
    )
    source_node = _bind_json_artifact(
        path_value=args.predecessor_source_node_path,
        expected_raw_sha256=args.expected_predecessor_source_node_sha256,
        expected_semantic_sha256=(
            args.expected_predecessor_source_node_semantic_sha256
        ),
        label="predecessor source node",
        validator=lambda value: _validate_predecessor_source_node(
            value, cycle_id=cycle_id
        ),
        semantic_field="semantic_sha256",
        trailing_newline=True,
    )
    if state.value["source_chain_node_sha256"] != source_node.semantic_sha256:
        raise FactorV4_1DiscoveryRunnerError(
            "PRECOMMITTED state does not bind the predecessor source node"
        )
    readback_report = _bind_json_artifact(
        path_value=args.predecessor_readback_report_path,
        expected_raw_sha256=args.expected_predecessor_readback_report_sha256,
        expected_semantic_sha256=(
            args.expected_predecessor_readback_report_semantic_sha256
        ),
        label="predecessor readback report",
        validator=lambda value: _validate_predecessor_readback_report(
            value, cycle_id=cycle_id
        ),
        semantic_field=None,
        trailing_newline=True,
        semantic_calculator=_full_canonical_predecessor_semantic_sha256,
    )
    predecessor_artifacts = {
        PREDECESSOR_INPUT_BINDING_FILENAME: input_binding,
        PREDECESSOR_DESIGN_SOURCE_FILENAME: design_source,
        PREDECESSOR_SOURCE_NODE_FILENAME: source_node,
        PREDECESSOR_STATE_FILENAME: state,
        PREDECESSOR_READBACK_FILENAME: readback_report,
    }
    _verify_predecessor_directory(predecessor_artifacts)
    _verify_predecessor_cross_bindings(
        cycle_id=cycle_id,
        input_binding=input_binding,
        design_source=design_source,
        source_node=source_node,
        state=state,
        readback_report=readback_report,
    )
    ontology = _bind_json_artifact(
        path_value=args.base_ontology_path,
        expected_raw_sha256=args.expected_base_ontology_sha256,
        expected_semantic_sha256=args.expected_base_ontology_semantic_sha256,
        label="base ontology",
        validator=screening.validate_primitive_ontology_v4,
        semantic_field="semantic_sha256",
        trailing_newline=False,
    )
    catalog = _bind_json_artifact(
        path_value=args.base_catalog_path,
        expected_raw_sha256=args.expected_base_catalog_sha256,
        expected_semantic_sha256=args.expected_base_catalog_semantic_sha256,
        label="base catalog",
        validator=lambda value: screening.validate_candidate_catalog_v4(
            value, ontology=ontology.value
        ),
        semantic_field="semantic_sha256",
        trailing_newline=False,
    )
    if len(catalog.value["candidates"]) != EXPECTED_BASE_CANDIDATE_COUNT:
        raise FactorV4_1DiscoveryRunnerError(
            "base catalog must contain exactly "
            f"{EXPECTED_BASE_CANDIDATE_COUNT} candidates"
        )
    local_evaluator = _bind_local_file(
        path_value=args.local_evaluator_path,
        expected_raw_sha256=args.expected_local_evaluator_sha256,
        label="local evaluator",
    )
    git_objects = bind_pinned_aquant_git_objects(args)
    code_bindings = _runtime_code_bindings(args)
    temporary = BoundDiscoveryInputs(
        predecessor_input_binding=input_binding,
        predecessor_design_source=design_source,
        predecessor_state=state,
        predecessor_source_node=source_node,
        predecessor_readback_report=readback_report,
        base_ontology=ontology,
        base_catalog=catalog,
        local_evaluator=local_evaluator,
        git_objects=git_objects,
        code_bindings=code_bindings,
        stable_identity_sha256="0" * 64,
    )
    identity = _sha256_bytes(_canonical_json_bytes(_stable_identity_payload(temporary)))
    return BoundDiscoveryInputs(
        predecessor_input_binding=input_binding,
        predecessor_design_source=design_source,
        predecessor_state=state,
        predecessor_source_node=source_node,
        predecessor_readback_report=readback_report,
        base_ontology=ontology,
        base_catalog=catalog,
        local_evaluator=local_evaluator,
        git_objects=git_objects,
        code_bindings=code_bindings,
        stable_identity_sha256=identity,
    )


def _semantic_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _verify_candidate_oracle(candidates: Sequence[Mapping[str, Any]]) -> None:
    if len(candidates) != EXPECTED_AQUANT_CANDIDATE_COUNT:
        raise FactorV4_1DiscoveryRunnerError(
            "pinned A_quant generator must yield exactly "
            f"{EXPECTED_AQUANT_CANDIDATE_COUNT} ideas"
        )
    names = [item.get("name") for item in candidates]
    if any(type(name) is not str for name in names) or len(names) != len(set(names)):
        raise FactorV4_1DiscoveryRunnerError(
            "pinned A_quant generator names must be unique strings"
        )
    if _semantic_sha256(names) != EXPECTED_ORDERED_NAMES_SEMANTIC_SHA256:
        raise FactorV4_1DiscoveryRunnerError(
            "pinned A_quant ordered candidate-name oracle mismatch"
        )


def _source_files_for_receipt(bound: BoundGitObjects) -> list[dict[str, Any]]:
    return [
        {
            "path": item.repository_path,
            "git_mode": "100644",
            "blob_oid": item.blob_oid,
            "raw_sha256": item.raw_sha256,
        }
        for item in bound.sources
    ]


def _code_bindings_for_artifact(bound: BoundDiscoveryInputs) -> list[dict[str, Any]]:
    return [
        {
            "absolute_path": item.absolute_path,
            "raw_sha256": item.raw_sha256,
            "size_bytes": item.size_bytes,
        }
        for item in bound.code_bindings
    ]


def _predecessor_bundle_bindings(
    bound: BoundDiscoveryInputs,
) -> list[dict[str, str]]:
    by_filename = {
        PREDECESSOR_INPUT_BINDING_FILENAME: bound.predecessor_input_binding,
        PREDECESSOR_DESIGN_SOURCE_FILENAME: bound.predecessor_design_source,
        PREDECESSOR_SOURCE_NODE_FILENAME: bound.predecessor_source_node,
        PREDECESSOR_STATE_FILENAME: bound.predecessor_state,
        PREDECESSOR_READBACK_FILENAME: bound.predecessor_readback_report,
    }
    return [
        {
            "filename": filename,
            "byte_sha256": by_filename[filename].raw_sha256,
            "semantic_sha256": by_filename[filename].semantic_sha256,
        }
        for filename in sorted(by_filename)
    ]


def _artifact_by_filename(
    artifacts: Mapping[str, Any], filename: str
) -> dict[str, Any]:
    value = artifacts.get(filename)
    if not isinstance(value, Mapping):
        raise FactorV4_1DiscoveryRunnerError(
            f"core did not return required artifact: {filename}"
        )
    return dict(value)


def _validate_accounting_oracle(artifacts: Mapping[str, Any]) -> None:
    audit = _artifact_by_filename(artifacts, "source_idea_audit.v4_1.json")
    catalog = _artifact_by_filename(artifacts, "discovery_catalog.v4_1.json")
    _artifact_by_filename(
        artifacts, "structural_collision_audit.v4_1.json"
    )
    member_count = catalog.get("member_count")
    selected_count = catalog.get("selected_count")
    accounting = {
        "source_idea_count": audit.get("total_idea_count"),
        "compatible_count": audit.get("compatible_count"),
        "incompatible_count": audit.get("incompatible_count"),
        "new_candidate_count": audit.get("new_candidate_count"),
        "structural_alias_count": audit.get("structural_alias_count"),
        "discovery_member_count": member_count,
        "selected_count": selected_count,
        "unselected_count": (
            member_count - selected_count
            if type(member_count) is int and type(selected_count) is int
            else None
        ),
    }
    for key, expected in EXPECTED_AQUANT_ACCOUNTING.items():
        if accounting.get(key) != expected:
            raise FactorV4_1DiscoveryRunnerError(
                f"pinned A_quant accounting oracle mismatch: {key}"
            )
    if (
        audit.get("compatible_ordered_names_semantic_sha256")
        != EXPECTED_COMPATIBLE_NAMES_SEMANTIC_SHA256
    ):
        raise FactorV4_1DiscoveryRunnerError(
            "pinned A_quant compatible-name oracle mismatch"
        )
    if (
        audit.get("structural_alias_ordered_names_semantic_sha256")
        != EXPECTED_ALIAS_NAMES_SEMANTIC_SHA256
    ):
        raise FactorV4_1DiscoveryRunnerError(
            "pinned A_quant structural-alias-name oracle mismatch"
        )


def _make_revalidator(
    args: argparse.Namespace, initial: BoundDiscoveryInputs
) -> Callable[[], None]:
    initial_identity = initial.stable_identity_sha256

    def revalidate() -> None:
        current = _bind_all_inputs(args)
        if current.stable_identity_sha256 != initial_identity:
            raise FactorV4_1DiscoveryRunnerError(
                "stable input CAS changed before DISCOVERY commit"
            )

    return revalidate


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Build and atomically publish one bounded DISCOVERY artifact bundle."""

    _absolute_path(args.private_root, "private output root")
    _safe_id(args.run_id, "run_id")
    bound = _bind_all_inputs(args)
    generator = next(
        item for item in bound.git_objects.sources if item.key == "generator"
    )
    try:
        candidates = discovery.extract_aquant_candidates_from_source(generator.data)
    except (TypeError, ValueError) as exc:
        raise FactorV4_1DiscoveryRunnerError(
            f"pinned A_quant AST extraction failed: {exc}"
        ) from exc
    _verify_candidate_oracle(candidates)

    source_receipt = discovery.build_aquant_source_receipt_v4_1(
        repository_top_level=bound.git_objects.repository_top_level,
        pinned_commit=bound.git_objects.pinned_commit,
        source_files=_source_files_for_receipt(bound.git_objects),
        candidates=candidates,
    )
    compatibility_contract = discovery.build_local_compatibility_contract_v4_1(
        evaluator_source_byte_sha256=bound.local_evaluator.raw_sha256,
    )
    source_idea_audit = discovery.build_source_idea_audit_v4_1(
        cycle_id=args.cycle_id,
        candidates=candidates,
        source_receipt=source_receipt,
        compatibility_contract=compatibility_contract,
        base_catalog=bound.base_catalog.value,
    )
    discovery_catalog = discovery.build_discovery_catalog_v4_1(
        cycle_id=args.cycle_id,
        base_ontology=bound.base_ontology.value,
        base_catalog=bound.base_catalog.value,
        source_receipt=source_receipt,
        compatibility_contract=compatibility_contract,
        source_idea_audit=source_idea_audit,
    )
    collision_audit = discovery.build_structural_collision_audit_v4_1(
        cycle_id=args.cycle_id,
        discovery_catalog=discovery_catalog,
    )
    source_node = discovery.build_discovery_source_node_v4_1(
        cycle_id=args.cycle_id,
        run_id=args.run_id,
        predecessor_bundle_bindings=_predecessor_bundle_bindings(bound),
        predecessor_source_node=bound.predecessor_source_node.value,
        predecessor_source_node_byte_sha256=(
            bound.predecessor_source_node.raw_sha256
        ),
        predecessor_state=bound.predecessor_state.value,
        predecessor_state_byte_sha256=bound.predecessor_state.raw_sha256,
        base_ontology=bound.base_ontology.value,
        base_ontology_byte_sha256=bound.base_ontology.raw_sha256,
        base_catalog=bound.base_catalog.value,
        base_catalog_byte_sha256=bound.base_catalog.raw_sha256,
        aquant_source_receipt=source_receipt,
        local_compatibility_contract=compatibility_contract,
        source_idea_audit=source_idea_audit,
        discovery_catalog=discovery_catalog,
        structural_collision_audit=collision_audit,
        code_bindings=_code_bindings_for_artifact(bound),
    )
    artifacts = {
        discovery.AQUANT_SOURCE_RECEIPT_FILENAME: source_receipt,
        discovery.SOURCE_IDEA_AUDIT_FILENAME: source_idea_audit,
        discovery.LOCAL_COMPATIBILITY_CONTRACT_FILENAME: compatibility_contract,
        discovery.DISCOVERY_CATALOG_FILENAME: discovery_catalog,
        discovery.STRUCTURAL_COLLISION_AUDIT_FILENAME: collision_audit,
        discovery.DISCOVERY_SOURCE_NODE_FILENAME: source_node,
    }
    _validate_accounting_oracle(artifacts)
    source_node_sha = _sha256(
        source_node.get("semantic_sha256"),
        "DISCOVERY source-node semantic SHA-256",
    )
    discovery_state = discovery.build_discovery_cycle_state_v4_1(
        predecessor_state=bound.predecessor_state.value,
        predecessor_state_byte_sha256=bound.predecessor_state.raw_sha256,
        expected_predecessor_byte_sha256=(
            args.expected_precommitted_state_sha256
        ),
        expected_predecessor_semantic_sha256=(
            args.expected_precommitted_state_semantic_sha256
        ),
        cycle_id=args.cycle_id,
        cycle_root_sha256=bound.predecessor_state.value["cycle_root_sha256"],
        discovery_source_node=source_node,
    )
    artifacts[discovery.DISCOVERY_CYCLE_STATE_FILENAME] = discovery_state

    expected_first_seven = {
        "aquant_source_receipt.v4_1.json",
        "source_idea_audit.v4_1.json",
        "local_compatibility_contract.v4_1.json",
        "discovery_catalog.v4_1.json",
        "structural_collision_audit.v4_1.json",
        "discovery_source_node.v4_1.json",
        "cycle_state.discovery.v4_1.json",
    }
    if set(artifacts) != expected_first_seven:
        raise FactorV4_1DiscoveryRunnerError(
            "core/publication artifact set must be the exact first seven files"
        )
    artifacts = {
        filename: discovery.validate_discovery_artifact_v4_1(
            filename, artifacts[filename]
        )
        for filename in discovery.PRE_READBACK_ARTIFACT_FILENAMES
    }
    readback_context = {
        "cycle_id": args.cycle_id,
        "run_id": args.run_id,
        "readiness": "EXPLORATORY_DISCOVERY",
        "qualification": False,
        "formal_admission_authority": False,
        "production_apply_enabled": False,
        "holdout": "sealed_not_appended",
        "blockers": list(FIXED_BLOCKERS),
        "statuses": dict(FIXED_NOT_RUN_STATUSES),
        "side_effects": dict(FIXED_SIDE_EFFECTS),
        "predecessor_state_byte_sha256": bound.predecessor_state.raw_sha256,
        "predecessor_state_semantic_sha256": (
            bound.predecessor_state.semantic_sha256
        ),
    }
    result = publication.publish_discovery_bundle_v4_1(
        private_root=args.private_root,
        run_id=args.run_id,
        artifacts=artifacts,
        revalidate_inputs=_make_revalidator(args, bound),
    )
    if not isinstance(result, Mapping):
        raise FactorV4_1DiscoveryRunnerError("publisher result must be a mapping")
    published = dict(result)
    if published.get("readiness") != "EXPLORATORY_DISCOVERY":
        raise FactorV4_1DiscoveryRunnerError(
            "publisher readiness contradicts the DISCOVERY contract"
        )
    if published.get("qualification") is not False:
        raise FactorV4_1DiscoveryRunnerError(
            "publisher qualification contradicts the DISCOVERY contract"
        )
    if published.get("side_effects") != FIXED_SIDE_EFFECTS:
        raise FactorV4_1DiscoveryRunnerError(
            "publisher side-effect declaration contradicts the DISCOVERY contract"
        )
    return {
        **published,
        **readback_context,
        "cycle_root_semantic_sha256": bound.predecessor_state.value[
            "cycle_root_sha256"
        ],
        "discovery_source_node_semantic_sha256": source_node_sha,
        "discovery_state_semantic_sha256": discovery_state[
            "state_semantic_sha256"
        ],
        "discovery_state_byte_sha256": cycle_state.byte_sha256(discovery_state),
        "discovery_input_set_semantic_sha256": bound.stable_identity_sha256,
        "predecessor_input_binding_semantic_sha256": (
            bound.predecessor_input_binding.semantic_sha256
        ),
        "predecessor_bundle_bindings": _predecessor_bundle_bindings(bound),
        "base_ontology_binding": {
            "absolute_path": bound.base_ontology.absolute_path,
            "byte_sha256": bound.base_ontology.raw_sha256,
            "semantic_sha256": bound.base_ontology.semantic_sha256,
        },
        "base_catalog_binding": {
            "absolute_path": bound.base_catalog.absolute_path,
            "byte_sha256": bound.base_catalog.raw_sha256,
            "semantic_sha256": bound.base_catalog.semantic_sha256,
        },
        "local_evaluator_binding": {
            "absolute_path": bound.local_evaluator.absolute_path,
            "byte_sha256": bound.local_evaluator.raw_sha256,
        },
        "aquant_accounting": dict(EXPECTED_AQUANT_ACCOUNTING),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one private offline Factor v4.1 DISCOVERY bundle"
    )
    parser.add_argument("--predecessor-input-binding-path", required=True)
    parser.add_argument(
        "--expected-predecessor-input-binding-sha256", required=True
    )
    parser.add_argument(
        "--expected-predecessor-input-binding-semantic-sha256", required=True
    )
    parser.add_argument("--predecessor-design-source-path", required=True)
    parser.add_argument(
        "--expected-predecessor-design-source-sha256", required=True
    )
    parser.add_argument(
        "--expected-predecessor-design-source-semantic-sha256", required=True
    )
    parser.add_argument("--precommitted-state-path", required=True)
    parser.add_argument("--expected-precommitted-state-sha256", required=True)
    parser.add_argument(
        "--expected-precommitted-state-semantic-sha256", required=True
    )
    parser.add_argument("--predecessor-source-node-path", required=True)
    parser.add_argument(
        "--expected-predecessor-source-node-sha256", required=True
    )
    parser.add_argument(
        "--expected-predecessor-source-node-semantic-sha256", required=True
    )
    parser.add_argument("--predecessor-readback-report-path", required=True)
    parser.add_argument(
        "--expected-predecessor-readback-report-sha256", required=True
    )
    parser.add_argument(
        "--expected-predecessor-readback-report-semantic-sha256", required=True
    )
    parser.add_argument("--base-ontology-path", required=True)
    parser.add_argument("--expected-base-ontology-sha256", required=True)
    parser.add_argument(
        "--expected-base-ontology-semantic-sha256", required=True
    )
    parser.add_argument("--base-catalog-path", required=True)
    parser.add_argument("--expected-base-catalog-sha256", required=True)
    parser.add_argument(
        "--expected-base-catalog-semantic-sha256", required=True
    )
    parser.add_argument("--local-evaluator-path", required=True)
    parser.add_argument("--expected-local-evaluator-sha256", required=True)
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cycle-id", required=True)
    parser.add_argument("--aquant-git-top-level", required=True)
    parser.add_argument("--aquant-pinned-commit", required=True)
    parser.add_argument("--git-executable", default=str(DEFAULT_GIT_EXECUTABLE))
    for spec in AQUANT_SOURCE_SPECS:
        flag = spec.key.replace("_", "-")
        parser.add_argument(
            f"--expected-aquant-{flag}-blob-oid", required=True
        )
        parser.add_argument(f"--expected-aquant-{flag}-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        result = run(parse_args(argv))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "readiness": "BLOCKED_FAIL_CLOSED",
                    "qualification": False,
                    "error": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("readiness") == "EXPLORATORY_DISCOVERY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
