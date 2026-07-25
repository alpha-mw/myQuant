#!/usr/bin/env python3
"""Run the unique, non-resumable, offline v17 Phase 0 v2 evidence session.

This runner is intentionally a closed orchestrator.  It accepts only fixed
input and output paths, executes the repository-owned ten-role DAG once, and
has no role, command, environment, resume, repair, or overwrite interface.
It must be launched by the frozen CPython with ``-I -S -B``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import selectors
import stat
import struct
import subprocess
import sys
from types import ModuleType
from typing import Any, Iterator, Mapping, Sequence

PROTOCOL_VERSION = "myquant.v17.v2"
SESSION_VERSION = "myquant.v17.v2.phase0-session.v2"
SESSION_SCHEMA_ID = "myquant.v17.v2.phase0-session.schema.v2"
COMMAND_RECEIPT_VERSION = "myquant.v17.v2.phase0-command-receipt.v2"
COMMAND_RECEIPT_SCHEMA_ID = "myquant.v17.v2.phase0-command-receipt.schema.v2"
MAIN_SUITE_RUNTIME_POLICY_VERSION = "myquant.v17.v2.phase0-main-suite-runtime-policy.v1"
MAIN_SUITE_RUNTIME_POLICY_SCHEMA_ID = "myquant.v17.v2.phase0-main-suite-runtime-policy.schema.v1"
MAIN_SUITE_RECEIPT_VERSION = "myquant.v17.v2.phase0-main-suite-receipt.v1"
MAIN_SUITE_RECEIPT_SCHEMA_ID = "myquant.v17.v2.phase0-main-suite-receipt.schema.v1"
DEPENDENCY_VERSION = "v17_third_party_dependency_environment_evidence.v2"
DEPENDENCY_SCHEMA_ID = "myquant.v17.v2.third-party-dependency-environment-evidence.schema.v2"
SKIP_BASELINE_VERSION = "myquant.v17.v2.phase0-skip-baseline.v2"
SKIP_BASELINE_SCHEMA_ID = "myquant.v17.v2.phase0-skip-baseline.schema.v2"
PACKAGE_VERSION = "myquant.v17.v2.phase0-package-parity-evidence.v2"
PACKAGE_SCHEMA_ID = "myquant.v17.v2.phase0-package-parity-evidence.schema.v2"
HASH_FREEZE_VERSION = "myquant.v17.v2.phase0-hash-freeze.v2"
HASH_FREEZE_SCHEMA_ID = "myquant.v17.v2.phase0-hash-freeze.schema.v2"
GATE_MANIFEST_VERSION = "myquant.v17.v2.phase0-gate-manifest.v2"
GATE_MANIFEST_SCHEMA_ID = "myquant.v17.v2.phase0-gate-manifest.schema.v2"
EVIDENCE_INDEX_VERSION = "myquant.v17.v2.phase0-evidence-index.v2"
EVIDENCE_INDEX_SCHEMA_ID = "myquant.v17.v2.phase0-evidence-index.schema.v2"
CLASSIFICATION_VERSION = "myquant.v17.v2.phase0-pre-existing-classification.v2"
CLASSIFICATION_SCHEMA_ID = "myquant.v17.v2.phase0-pre-existing-classification.schema.v2"
FAILURE_VERSION = "myquant.v17.v2.phase0-unpublished-failure.v2"
FAILURE_SCHEMA_ID = "myquant.v17.v2.phase0-unpublished-failure.schema.v2"

SESSION_PRODUCER_VERSION = "myquant.v17.v2.phase0-evidence-session-runner.v2"
DEPENDENCY_PRODUCER_VERSION = DEPENDENCY_VERSION
PACKAGE_PRODUCER_VERSION = "myquant.v17.v2.phase0-package-evidence-producer.v2"
INDEX_PRODUCER_VERSION = "myquant.v17.v2.phase0-evidence-index-producer.v2"
SEMANTIC_FIELD = "semantic_sha256"

BASE_PYTHON = Path(
    "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/"
    "Python.framework/Versions/3.13/bin/python3.13"
)
BASE_PYTHON_SHA256 = "a708f6e9f4803b806b29146c4e0feecfd9bf2d9eb60f3e15b850cd7cb56f200b"
BASE_PYTHON_SIZE = 52_640
BASE_PYTHON_VERSION = "3.13.7"
UV_BIN = Path("/Users/maxwell/.local/bin/uv")
UV_SHA256 = "bc50ab0e90f24491f0e794f5b8649722f8fd2bf483c53490c012b41b89151ef9"
UV_SIZE = 44_698_848
UV_VERSION = "0.10.9"
UV_VERSION_OUTPUT = "uv 0.10.9 (f675560f3 2026-03-06)"
UV_CACHE = Path("/Users/maxwell/.cache/uv")
AUTHORITY_REPO_ROOT = Path("/Users/maxwell/mySpace/myQuant")

MAX_STREAM_BYTES = 128 * 1024 * 1024
MAX_COMMAND_BYTES = 256 * 1024 * 1024
MAX_FILE_BYTES = 512 * 1024 * 1024
COMMAND_RECEIPT_PREFIX = b"MYQUANT_PHASE0_COMMAND_RECEIPT="
MAIN_SUITE_RECEIPT_PREFIX = b"MYQUANT_PHASE0_MAIN_SUITE_RECEIPT="
COMMAND_FRAMING = "UINT64_BE_STDOUT_THEN_STDOUT_UINT64_BE_STDERR_THEN_STDERR_PER_COMMAND"
MAIN_SUITE_FRAMING = (
    "UINT64_BE_STDOUT_THEN_STDOUT_UINT64_BE_STDERR_THEN_STDERR_"
    "UINT64_BE_ATTESTATION_THEN_ATTESTATION"
)
MAIN_SUITE_POLICY_PATH = (
    "quant_investor/v17_v2_contract/resources/main_suite_runtime_policy.v1.json"
)
MAIN_SUITE_POLICY_SCHEMA_PATH = (
    "quant_investor/v17_v2_contract/schemas/main_suite_runtime_policy.v1.schema.json"
)
MAIN_SUITE_RECEIPT_SCHEMA_PATH = "scripts/schemas/v17_phase0_main_suite_receipt.v1.schema.json"
MAIN_SUITE_HARNESS_PATH = "scripts/v17_phase0_main_suite_harness.py"
MAIN_SUITE_PACKAGE_MANIFEST_PATH = (
    "quant_investor/v17_v2_contract/resources/package_manifest.v1.json"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$", re.ASCII)
SESSION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$", re.ASCII)
FAILURE_PHASES = frozenset({"PRIMARY", "CLEANUP", "EXTERNAL_AFTER"})
MAIN_SUITE_FINALIZATION_NOT_ATTEMPTED = {
    "attempted": False,
    "cleanup": {
        "attempted": False,
        "status": "NOT_ATTEMPTED",
    },
    "external_after": {
        "attempted": False,
        "equal": None,
        "status": "NOT_ATTEMPTED",
    },
}

LIMITATIONS = [
    "PRECOLLECTION_ALLOWED_TREES_ATTESTED_NOT_COMPLETE_MAIN_RUNTIME_FILESET",
    "NO_DIRECT_LAUNCH_REFERENCE_ONLY_NOT_PROCESS_IO_ATTESTED",
    "OFFLINE_FLAGS_ONLY_NOT_OS_EGRESS_ATTESTED",
    "OWNER_DECLARED_PATHS_ONLY_NOT_CONTENT_PROVENANCE",
    "PIP_25_2_PACKAGE_INSTALL_ENV_ONLY_NATIVE_AND_BUILD_ENV_PIP_ABSENT",
]
PIP_SCOPE = {
    "allowed_wrappers": ["pip3", "pip3.13"],
    "build_pip_absent": True,
    "bundled_wheel": {
        "name": "pip-25.2-py3-none-any.whl",
        "sha256": "690972885fc9270380d1bb28212cafdff6a96e0b6e04396b9fa7505253591e11",
        "size_bytes": 1_752_557,
    },
    "ensurepip_argv_suffix": ["-I", "-m", "ensurepip", "--upgrade"],
    "environment_scope": "PACKAGE_INSTALL_ENV_ONLY",
    "native_pip_absent": True,
    "plain_pip_absent": True,
    "version": "25.2",
}

GATE_ROLES = (
    "native_sync_log",
    "native_sync_receipt",
    "v2_evidence_tests",
    "recommended_core_tests",
    "full_offline_suite",
    "mypy",
    "black",
    "diff_check",
    "package_parity",
    "hash_freeze_readback",
)
GATE_FILENAMES = (
    "10_native_sync.log",
    "20_native_dependency.json",
    "30_v2_tests.log",
    "31_recommended_core.log",
    "32_full_suite.log",
    "33_mypy.log",
    "34_black.log",
    "35_diff_check.log",
    "40_package_parity.json",
    "50_hash_freeze.json",
)
FIXED_BUNDLE_FILES = (
    "00_session.json",
    *GATE_FILENAMES,
    "60_gate_manifest.json",
    "70_evidence_index.json",
    "70_evidence_index.json.sha256",
)
FAILURE_FILENAME = "99_unpublished_failure.json"
LOG_ROLES = {
    "native_sync_log",
    "v2_evidence_tests",
    "recommended_core_tests",
    "full_offline_suite",
    "mypy",
    "black",
    "diff_check",
}

V2_EVIDENCE_TESTS = (
    "tests/unit/test_v17_v2_action_matrix.py",
    "tests/unit/test_v17_v2_identities.py",
    "tests/unit/test_v17_v2_limits.py",
    "tests/unit/test_v17_v2_namespace.py",
    "tests/unit/test_v17_v2_package.py",
    "tests/unit/test_v17_v2_policy_resources.py",
    "tests/unit/test_v17_v2_schema_validation.py",
    "tests/unit/test_v17_v2_schemas.py",
    "tests/unit/test_v17_v2_validators.py",
    "tests/unit/test_v17_offline_dependency_evidence.py",
    "tests/unit/test_v17_phase0_diff_check.py",
    "tests/unit/test_v17_phase0_evidence_index.py",
    "tests/unit/test_v17_phase0_main_suite_contract.py",
    "tests/unit/test_v17_phase0_evidence_session.py",
    "tests/unit/test_v17_phase0_package_evidence.py",
    "tests/unit/test_v17_phase0_skip_baseline.py",
)
RECOMMENDED_CORE_TESTS = (
    "tests/unit/test_public_package_smoke.py",
    "tests/unit/test_data_layer.py",
    "tests/unit/test_forecast_snapshot_cache.py",
    "tests/unit/test_llm_env_inventory.py",
    "tests/unit/test_tushare_url_defaults.py",
    "tests/unit/test_fundamental_provider_contract.py",
    "tests/unit/test_fundamental_live_fetch_resilience.py",
    "tests/unit/test_fundamental_generation_promotion.py",
    "tests/integration/test_review_layer_timeout_budget.py",
)
PYTEST_OPTIONS = ("-q", "-p", "no:cacheprovider", "--color=no")
FULL_PYTEST_OPTIONS = (*PYTEST_OPTIONS, "-rs")
MAIN_SUITE_PYTEST_ARGS = (
    "-p",
    "pytest_cov",
    "-p",
    "asyncio",
    "-p",
    "anyio",
    "-p",
    "no:cacheprovider",
    "-q",
    "--color=no",
    "-rs",
)
MAIN_SUITE_PATH_TOPOLOGY = {
    "cache_children": [
        "BLACK_CACHE_DIR",
        "MYPY_CACHE_DIR",
        "PYTHONPYCACHEPREFIX",
    ],
    "closed_root_siblings": ["HOME", "TMPDIR", "XDG_CACHE_HOME"],
    "must_remain_empty": ["PYTHONPYCACHEPREFIX"],
}
MYPY_TARGETS = (
    "quant_investor/v17_v2_contract",
    "scripts/v17_phase0_evidence_index.py",
    "scripts/v17_phase0_main_suite_harness.py",
    "scripts/v17_phase0_main_suite_wrapper.py",
)
BLACK_TARGETS = (
    "quant_investor/v17_v2_contract",
    "scripts/v17_offline_dependency_evidence.py",
    "scripts/v17_phase0_diff_check.py",
    "scripts/v17_phase0_evidence_index.py",
    "scripts/v17_phase0_evidence_session.py",
    "scripts/v17_phase0_main_suite_harness.py",
    "scripts/v17_phase0_main_suite_wrapper.py",
    "scripts/v17_phase0_package_evidence.py",
    "scripts/v17_phase0_skip_baseline.py",
    "tests/unit/test_v17_offline_dependency_evidence.py",
    "tests/unit/test_v17_phase0_diff_check.py",
    "tests/unit/test_v17_phase0_evidence_index.py",
    "tests/unit/test_v17_phase0_main_suite_contract.py",
    "tests/unit/test_v17_phase0_evidence_session.py",
    "tests/unit/test_v17_phase0_package_evidence.py",
    "tests/unit/test_v17_phase0_skip_baseline.py",
    *V2_EVIDENCE_TESTS[:9],
)

SCHEMA_REGISTRY = (
    (
        CLASSIFICATION_VERSION,
        "scripts/schemas/v17_phase0_pre_existing_classification.v2.schema.json",
        CLASSIFICATION_SCHEMA_ID,
    ),
    (
        SESSION_VERSION,
        "scripts/schemas/v17_phase0_session.v2.schema.json",
        SESSION_SCHEMA_ID,
    ),
    (
        COMMAND_RECEIPT_VERSION,
        "scripts/schemas/v17_phase0_command_receipt.v2.schema.json",
        COMMAND_RECEIPT_SCHEMA_ID,
    ),
    (
        MAIN_SUITE_RUNTIME_POLICY_VERSION,
        MAIN_SUITE_POLICY_SCHEMA_PATH,
        MAIN_SUITE_RUNTIME_POLICY_SCHEMA_ID,
    ),
    (
        MAIN_SUITE_RECEIPT_VERSION,
        MAIN_SUITE_RECEIPT_SCHEMA_PATH,
        MAIN_SUITE_RECEIPT_SCHEMA_ID,
    ),
    (
        DEPENDENCY_VERSION,
        "scripts/schemas/v17_offline_dependency_evidence.v2.schema.json",
        DEPENDENCY_SCHEMA_ID,
    ),
    (
        SKIP_BASELINE_VERSION,
        "scripts/schemas/v17_phase0_skip_baseline.v2.schema.json",
        SKIP_BASELINE_SCHEMA_ID,
    ),
    (
        PACKAGE_VERSION,
        "scripts/schemas/v17_phase0_package_evidence.v2.schema.json",
        PACKAGE_SCHEMA_ID,
    ),
    (
        HASH_FREEZE_VERSION,
        "scripts/schemas/v17_phase0_hash_freeze.v2.schema.json",
        HASH_FREEZE_SCHEMA_ID,
    ),
    (
        GATE_MANIFEST_VERSION,
        "scripts/schemas/v17_phase0_gate_manifest.v2.schema.json",
        GATE_MANIFEST_SCHEMA_ID,
    ),
    (
        EVIDENCE_INDEX_VERSION,
        "scripts/schemas/v17_phase0_evidence_index.v2.schema.json",
        EVIDENCE_INDEX_SCHEMA_ID,
    ),
    (
        FAILURE_VERSION,
        "scripts/schemas/v17_phase0_unpublished_failure.v2.schema.json",
        FAILURE_SCHEMA_ID,
    ),
)

BASE_CLOSED_ENVIRONMENT = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
    "PIP_CONFIG_FILE": "/dev/null",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PIP_NO_INDEX": "1",
    "PIP_NO_INPUT": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "UV_CACHE_DIR": str(UV_CACHE),
    "UV_NO_CONFIG": "1",
    "UV_OFFLINE": "1",
    "UV_PYTHON_DOWNLOADS": "never",
}
PIP_STATUS_ENV_KEYS = (
    "PIP_CONFIG_FILE",
    "PIP_DISABLE_PIP_VERSION_CHECK",
    "PIP_NO_INDEX",
    "PIP_NO_INPUT",
    "UV_NO_CONFIG",
    "UV_OFFLINE",
    "UV_PYTHON_DOWNLOADS",
)
PIP_OBSERVATION_SCOPE = "NON_IMPORTING_PARENT_VISIBILITY_AND_FIXED_CHILD_ENVIRONMENT_ONLY"


class Phase0SessionError(RuntimeError):
    """Raised when the closed Phase 0 session must stop unpublished."""

    exit_code = 2

    def __init__(self, message: str, *, stage: str = "preflight") -> None:
        super().__init__(message)
        self.stage = stage


class MainSuiteRejectedError(Phase0SessionError):
    """Carry a rejected harness receipt into the unpublished failure artifact."""

    def __init__(
        self,
        message: str,
        *,
        failures: Sequence[Mapping[str, Any]],
        finalization: Mapping[str, Any],
        stage: str,
    ) -> None:
        super().__init__(message, stage=stage)
        self.main_suite_failures = [dict(failure) for failure in failures]
        self.main_suite_finalization = dict(finalization)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise Phase0SessionError("value is not canonical JSON") from exc


def _canonical_resource_bytes(value: Any) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _index_sidecar_bytes(index_raw: bytes, *, filename: str) -> bytes:
    return f"{_sha256(index_raw)}  {filename}\n".encode("ascii")


def _semantic_sha256(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop(SEMANTIC_FIELD, None)
    return _sha256(_canonical_bytes(body))


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    if SEMANTIC_FIELD in value:
        raise Phase0SessionError("semantic_sha256 must not be supplied")
    sealed = dict(value)
    sealed[SEMANTIC_FIELD] = _semantic_sha256(sealed)
    return sealed


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_file(
    path: Path,
    *,
    label: str,
    max_bytes: int = MAX_FILE_BYTES - 1,
    mode: int | None = None,
    owner: bool = False,
    single_link: bool = False,
) -> tuple[bytes, os.stat_result]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise Phase0SessionError(f"{label} is unavailable") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_size > max_bytes
    ):
        raise Phase0SessionError(f"{label} is not a bounded regular non-symlink file")
    if owner and before.st_uid != os.getuid():
        raise Phase0SessionError(f"{label} is not owner-owned")
    if mode is not None and stat.S_IMODE(before.st_mode) != mode:
        raise Phase0SessionError(f"{label} must be mode {mode:04o}")
    if single_link and before.st_nlink != 1:
        raise Phase0SessionError(f"{label} must have exactly one hard link")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Phase0SessionError(f"{label} cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise Phase0SessionError(f"{label} exceeds its byte limit")
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as exc:
        raise Phase0SessionError(f"{label} disappeared during read") from exc
    if _stat_signature(before) != _stat_signature(opened) or _stat_signature(
        before
    ) != _stat_signature(after):
        raise Phase0SessionError(f"{label} changed during stable read")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise Phase0SessionError(f"{label} size changed during stable read")
    return raw, before


def _file_binding(path: Path, *, label: str, relative_path: str | None = None) -> dict[str, Any]:
    raw, _observed = _stable_file(path, label=label)
    return {
        "path": relative_path if relative_path is not None else str(path),
        "sha256": _sha256(raw),
        "size_bytes": len(raw),
    }


def _main_suite_file_binding(path: Path, *, label: str) -> dict[str, Any]:
    raw, observed = _stable_file(path, label=label, single_link=True)
    return {
        "gid": observed.st_gid,
        "mode": f"{stat.S_IMODE(observed.st_mode):04o}",
        "path": str(path),
        "sha256": _sha256(raw),
        "size_bytes": len(raw),
        "st_dev": observed.st_dev,
        "st_ino": observed.st_ino,
        "st_nlink": observed.st_nlink,
        "uid": observed.st_uid,
    }


def _empty_private_directory_binding(path: Path, *, label: str) -> dict[str, Any]:
    try:
        before = path.lstat()
        with os.scandir(path) as entries:
            if next(entries, None) is not None:
                raise Phase0SessionError(f"{label} is not empty")
        after = path.lstat()
    except Phase0SessionError:
        raise
    except OSError as exc:
        raise Phase0SessionError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISDIR(before.st_mode)
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) != 0o700
        or _stat_signature(before) != _stat_signature(after)
    ):
        raise Phase0SessionError(f"{label} is not a stable owner-private directory")
    return {
        "gid": before.st_gid,
        "mode": "0700",
        "path": str(path),
        "st_ctime_ns": before.st_ctime_ns,
        "st_dev": before.st_dev,
        "st_ino": before.st_ino,
        "st_mtime_ns": before.st_mtime_ns,
        "st_nlink": before.st_nlink,
        "uid": before.st_uid,
    }


def _main_suite_expected_command(
    *,
    repo_root: Path,
    policy: Mapping[str, Any],
    policy_bindings: Mapping[str, Mapping[str, Any]],
    environment: Mapping[str, str],
) -> dict[str, Any]:
    main_runtime = policy.get("main_runtime")
    wrapper = policy.get("wrapper_binding")
    policy_binding = policy_bindings.get("policy_binding")
    if (
        type(main_runtime) is not dict
        or type(wrapper) is not dict
        or type(policy_binding) is not dict
        or type(main_runtime.get("lexical_python")) is not str
        or type(wrapper.get("path")) is not str
        or type(policy_binding.get("sha256")) is not str
        or type(environment.get("PYTHONPYCACHEPREFIX")) is not str
    ):
        raise Phase0SessionError("main-suite command policy is invalid")
    return {
        "argv": [
            main_runtime["lexical_python"],
            "-I",
            "-S",
            "-B",
            "-X",
            f"pycache_prefix={environment['PYTHONPYCACHEPREFIX']}",
            wrapper["path"],
            str(repo_root / MAIN_SUITE_POLICY_PATH),
            policy_binding["sha256"],
            "--",
            *MAIN_SUITE_PYTEST_ARGS,
        ],
        "cwd": str(repo_root),
        "environment": dict(sorted(environment.items())),
    }


def _require_absolute(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise Phase0SessionError(f"{label} must be absolute")
    return path.absolute()


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _paths_nested(left: Path, right: Path) -> bool:
    return _path_within(left, right) or _path_within(right, left)


def _assert_no_symlink_components(path: Path, *, include_leaf: bool, label: str) -> None:
    parts = path.parts
    stop = len(parts) - 1
    current = Path(parts[0])
    for part in parts[1:stop]:
        current /= part
        try:
            observed = current.lstat()
        except OSError as exc:
            raise Phase0SessionError(f"{label} parent component is unavailable: {current}") from exc
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise Phase0SessionError(f"{label} has unsafe path component: {current}")
    if include_leaf:
        try:
            leaf = path.lstat()
        except OSError as exc:
            raise Phase0SessionError(f"{label} is unavailable") from exc
        if stat.S_ISLNK(leaf.st_mode):
            raise Phase0SessionError(f"{label} must not be a symlink")


def _protected_root_specs(repo_root: Path) -> tuple[tuple[str, Path], ...]:
    return (
        ("authority_v16", AUTHORITY_REPO_ROOT / "results" / "v16"),
        (
            "authority_v16_operator_advisory",
            AUTHORITY_REPO_ROOT / "results" / "v16_operator_advisory",
        ),
        ("candidate_v16", repo_root / "results" / "v16"),
        (
            "candidate_v16_operator_advisory",
            repo_root / "results" / "v16_operator_advisory",
        ),
    )


def _validate_external_location(path: Path, *, repo_root: Path, label: str) -> Path:
    absolute = _require_absolute(path, label=label)
    for forbidden in (repo_root, *[item[1] for item in _protected_root_specs(repo_root)]):
        if _path_within(absolute, forbidden):
            raise Phase0SessionError(f"{label} must be outside repository/protected roots")
    _assert_no_symlink_components(absolute, include_leaf=True, label=label)
    try:
        parent = absolute.parent.lstat()
    except OSError as exc:
        raise Phase0SessionError(f"{label} parent is unavailable") from exc
    if (
        not stat.S_ISDIR(parent.st_mode)
        or stat.S_IMODE(parent.st_mode) != 0o700
        or parent.st_uid != os.getuid()
    ):
        raise Phase0SessionError(f"{label} parent must be owner-owned mode 0700")
    return absolute


def _prepare_new_roots(
    bundle_root: Path,
    work_root: Path,
    *,
    repo_root: Path,
) -> tuple[Path, Path]:
    bundle = _require_absolute(bundle_root, label="bundle root")
    work = _require_absolute(work_root, label="work root")
    forbidden = (repo_root, *[item[1] for item in _protected_root_specs(repo_root)])
    if bundle == work or _paths_nested(bundle, work):
        raise Phase0SessionError("bundle and work roots must be distinct and non-nested")
    for label, path in (("bundle root", bundle), ("work root", work)):
        for root in forbidden:
            if _paths_nested(path, root):
                raise Phase0SessionError(f"{label} must be outside repository/protected roots")
        _assert_no_symlink_components(path, include_leaf=False, label=label)
        try:
            path.lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise Phase0SessionError(f"{label} cannot be inspected") from exc
        else:
            raise Phase0SessionError(f"{label} must never have existed")
    old_umask = os.umask(0o077)
    try:
        bundle.mkdir(mode=0o700, parents=False, exist_ok=False)
        work.mkdir(mode=0o700, parents=False, exist_ok=False)
    except OSError as exc:
        raise Phase0SessionError("cannot create fresh Phase 0 roots") from exc
    finally:
        os.umask(old_umask)
    for label, path in (("bundle root", bundle), ("work root", work)):
        observed = path.lstat()
        if (
            stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISDIR(observed.st_mode)
            or stat.S_IMODE(observed.st_mode) != 0o700
            or observed.st_uid != os.getuid()
            or any(path.iterdir())
        ):
            raise Phase0SessionError(f"{label} creation did not yield an empty owner-0700 root")
    return bundle, work


class BundlePublisher:
    """Exact-once, no-repair publisher pinned to one owner-private bundle fd."""

    def __init__(self, root: Path, *, session_id: str) -> None:
        self.root = root
        self.session_id = session_id
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            self.fd = os.open(root, flags)
        except OSError as exc:
            raise Phase0SessionError("cannot pin bundle root") from exc
        self.identity = os.fstat(self.fd)
        self._assert_current()

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def _assert_current(self) -> None:
        try:
            current = self.root.lstat()
            opened = os.fstat(self.fd)
        except OSError as exc:
            raise Phase0SessionError("bundle root identity is unavailable") from exc
        expected = self.identity
        if (
            stat.S_ISLNK(current.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or current.st_uid != os.getuid()
            or stat.S_IMODE(current.st_mode) != 0o700
            or (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino)
            or (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino)
        ):
            raise Phase0SessionError("bundle root identity changed")

    def _read_at(self, name: str) -> tuple[bytes, os.stat_result]:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=self.fd)
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.getuid()
                or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_size >= MAX_FILE_BYTES
            ):
                raise Phase0SessionError(f"published file is unsafe: {name}")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, min(1024 * 1024, MAX_FILE_BYTES - total))
                if not chunk:
                    break
                total += len(chunk)
                if total >= MAX_FILE_BYTES:
                    raise Phase0SessionError(f"published file exceeds limit: {name}")
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        if _stat_signature(before) != _stat_signature(after):
            raise Phase0SessionError(f"published file changed during readback: {name}")
        return b"".join(chunks), before

    def publish(self, name: str, raw: bytes) -> dict[str, Any]:
        if name not in FIXED_BUNDLE_FILES and name != FAILURE_FILENAME:
            raise Phase0SessionError(f"noncanonical bundle filename rejected: {name}")
        if not raw or len(raw) >= MAX_FILE_BYTES:
            raise Phase0SessionError(f"invalid publication size for {name}")
        self._assert_current()
        stage_name = f".{name}.staged-{self.session_id}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(stage_name, flags, 0o600, dir_fd=self.fd)
        except FileExistsError as exc:
            raise Phase0SessionError(f"orphaned staged inode exists for {name}") from exc
        except OSError as exc:
            raise Phase0SessionError(f"cannot create staged inode for {name}") from exc
        try:
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise Phase0SessionError(f"short write while staging {name}")
                view = view[written:]
            os.fsync(descriptor)
            staged_stat = os.fstat(descriptor)
            if (
                not stat.S_ISREG(staged_stat.st_mode)
                or stat.S_IMODE(staged_stat.st_mode) != 0o600
                or staged_stat.st_uid != os.getuid()
                or staged_stat.st_nlink != 1
                or staged_stat.st_size != len(raw)
            ):
                raise Phase0SessionError(f"staged inode is unsafe for {name}")
        finally:
            os.close(descriptor)
        try:
            os.link(
                stage_name,
                name,
                src_dir_fd=self.fd,
                dst_dir_fd=self.fd,
                follow_symlinks=False,
            )
            os.fsync(self.fd)
        except FileExistsError as exc:
            raise Phase0SessionError(f"published file already exists: {name}") from exc
        except OSError as exc:
            raise Phase0SessionError(f"cannot hard-link staged publication: {name}") from exc
        staged_raw, staged_readback = self._read_at(stage_name)
        final_raw, final_readback = self._read_at(name)
        if (
            staged_raw != raw
            or final_raw != raw
            or (staged_readback.st_dev, staged_readback.st_ino)
            != (final_readback.st_dev, final_readback.st_ino)
            or final_readback.st_nlink != 2
        ):
            raise Phase0SessionError(f"same-inode readback failed for {name}")
        try:
            os.unlink(stage_name, dir_fd=self.fd)
            os.fsync(self.fd)
        except OSError as exc:
            raise Phase0SessionError(f"cannot remove proven staged link for {name}") from exc
        self._assert_current()
        final_raw, final_stat = self._read_at(name)
        if final_raw != raw or final_stat.st_nlink != 1:
            raise Phase0SessionError(f"final publication readback failed for {name}")
        return {
            "mode": "0600",
            "path": str(self.root / name),
            "sha256": _sha256(raw),
            "size_bytes": len(raw),
            "st_dev": final_stat.st_dev,
            "st_ino": final_stat.st_ino,
        }


@contextmanager
def _closed_process_environment(
    overrides: Mapping[str, str] | None = None,
) -> Iterator[dict[str, str]]:
    environment = dict(BASE_CLOSED_ENVIRONMENT)
    if overrides:
        environment.update(overrides)
    previous = dict(os.environ)
    os.environ.clear()
    os.environ.update(environment)
    try:
        yield environment
    finally:
        os.environ.clear()
        os.environ.update(previous)


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise Phase0SessionError(f"cannot load repository helper: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(name, None)
        raise Phase0SessionError(f"repository helper import failed: {path.name}") from exc
    finally:
        sys.dont_write_bytecode = previous
    return module


def _checked_schema(
    value: Mapping[str, Any],
    *,
    repo_root: Path,
    relative_path: str,
    schema_id: str,
    artifact_version: str,
) -> None:
    raw, _observed = _stable_file(
        repo_root / relative_path,
        label=f"{artifact_version} schema",
    )
    try:
        schema = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Phase0SessionError(f"{artifact_version} schema is invalid JSON") from exc
    if (
        type(schema) is not dict
        or schema.get("$id") != schema_id
        or not isinstance(schema.get("properties"), dict)
        or schema["properties"].get("version") != {"const": artifact_version}
        and schema["properties"].get("schema_version") != {"const": artifact_version}
    ):
        raise Phase0SessionError(f"{artifact_version} schema identity mismatch")

    package_name = "_myquant_phase0_session_schema"
    contract_root = repo_root / "quant_investor" / "v17_v2_contract"
    loaded: list[str] = []
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        package = ModuleType(package_name)
        package.__path__ = [str(contract_root)]  # type: ignore[attr-defined]
        package.__package__ = package_name
        sys.modules[package_name] = package
        loaded.append(package_name)
        for short_name in ("limits", "identities", "canonical"):
            full_name = f"{package_name}.{short_name}"
            _load_module(full_name, contract_root / f"{short_name}.py")
            loaded.append(full_name)
        resources_name = f"{package_name}.resources"
        resources = ModuleType(resources_name)

        def unavailable_resource(*_args: Any, **_kwargs: Any) -> None:
            raise Phase0SessionError("packaged resources are unavailable during schema validation")

        resources.load_packaged_json = unavailable_resource  # type: ignore[attr-defined]
        sys.modules[resources_name] = resources
        loaded.append(resources_name)
        validation_name = f"{package_name}.schema_validation"
        validation = _load_module(
            validation_name,
            contract_root / "schema_validation.py",
        )
        loaded.append(validation_name)
        validation.preflight_packaged_schema(schema)
        validation.validate_instance_against_schema(dict(value), schema)
    except Phase0SessionError:
        raise
    except Exception as exc:
        raise Phase0SessionError(f"{artifact_version} schema validation failed") from exc
    finally:
        sys.dont_write_bytecode = previous
        for loaded_name in reversed(loaded):
            sys.modules.pop(loaded_name, None)


def _strict_canonical_json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Phase0SessionError(f"{label} is invalid JSON") from exc
    if type(value) is not dict or raw != _canonical_resource_bytes(value):
        raise Phase0SessionError(f"{label} must be canonical JSON plus newline")
    return value


def _external_json_binding(
    path: Path,
    *,
    repo_root: Path,
    label: str,
    artifact_version: str,
    schema_id: str,
    schema_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    absolute = _validate_external_location(path, repo_root=repo_root, label=label)
    raw, _observed = _stable_file(
        absolute,
        label=label,
        mode=0o600,
        owner=True,
        single_link=True,
    )
    value = _strict_canonical_json(raw, label=label)
    version = value.get("version", value.get("schema_version"))
    if version != artifact_version or value.get(SEMANTIC_FIELD) != _semantic_sha256(value):
        raise Phase0SessionError(f"{label} identity or semantic SHA-256 mismatch")
    _checked_schema(
        value,
        repo_root=repo_root,
        relative_path=schema_path,
        schema_id=schema_id,
        artifact_version=artifact_version,
    )
    return (
        {
            "path": str(absolute),
            "semantic_sha256": value[SEMANTIC_FIELD],
            "sha256": _sha256(raw),
            "size_bytes": len(raw),
        },
        value,
    )


def _resolve_repo(repo_root: Path) -> tuple[Path, str]:
    absolute = _require_absolute(repo_root, label="repo root")
    _assert_no_symlink_components(absolute, include_leaf=True, label="repo root")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise Phase0SessionError("repo root is unavailable") from exc
    if resolved != absolute or not (absolute / ".git").exists():
        raise Phase0SessionError("repo root must be the concrete candidate Git worktree")
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel", "HEAD"],
        cwd=absolute,
        env=BASE_CLOSED_ENVIRONMENT,
        shell=False,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise Phase0SessionError("cannot resolve candidate Git HEAD")
    try:
        lines = completed.stdout.decode("utf-8", errors="strict").splitlines()
    except UnicodeError as exc:
        raise Phase0SessionError("Git returned invalid repository identity") from exc
    if len(lines) != 2 or Path(lines[0]).resolve(strict=True) != absolute:
        raise Phase0SessionError("repo root is not the Git top level")
    base_commit = lines[1]
    if COMMIT_RE.fullmatch(base_commit) is None:
        raise Phase0SessionError("Git HEAD is not a full object ID")
    return absolute, base_commit


def _source_snapshot(repo_root: Path, base_commit: str) -> tuple[dict[str, Any], dict[str, str]]:
    name = "_myquant_phase0_session_index_source"
    module = _load_module(name, repo_root / "scripts" / "v17_phase0_evidence_index.py")
    try:
        with _closed_process_environment():
            snapshot = module._git_snapshot(repo_root, base_commit)
        public = module._public_source_state(snapshot)
        binding = module._source_binding_from_state(public)
    except Exception as exc:
        if isinstance(exc, Phase0SessionError):
            raise
        raise Phase0SessionError("cannot capture canonical Phase 0 source state") from exc
    finally:
        sys.modules.pop(name, None)
    if (
        type(public) is not dict
        or type(binding) is not dict
        or binding.get("base_commit") != base_commit
        or set(binding)
        != {
            "base_commit",
            "binary_diff_sha256",
            "porcelain_sha256",
            "source_state_sha256",
            "untracked_inventory_sha256",
        }
        or any(
            key != "base_commit" and (type(value) is not str or SHA256_RE.fullmatch(value) is None)
            for key, value in binding.items()
        )
    ):
        raise Phase0SessionError("canonical Phase 0 source binding is invalid")
    return dict(public), dict(binding)


def _package_source_superset(repo_root: Path) -> dict[str, Any]:
    name = "_myquant_phase0_session_package_source"
    module = _load_module(name, repo_root / "scripts" / "v17_phase0_package_evidence.py")
    try:
        with _closed_process_environment():
            value = module._sample_physical_superset(repo_root)
    except Exception as exc:
        if isinstance(exc, Phase0SessionError):
            raise
        raise Phase0SessionError("cannot capture physical package-source superset") from exc
    finally:
        sys.modules.pop(name, None)
    if (
        type(value) is not dict
        or set(value) != {"row_count", "rows", "sha256"}
        or type(value["row_count"]) is not int
        or value["row_count"] <= 0
        or type(value["rows"]) is not list
        or len(value["rows"]) != value["row_count"]
        or type(value["sha256"]) is not str
        or SHA256_RE.fullmatch(value["sha256"]) is None
    ):
        raise Phase0SessionError("physical package-source superset binding is invalid")
    return {
        "row_count": value["row_count"],
        "sha256": value["sha256"],
    }


def _validate_candidate_source_membership(
    *,
    repo_root: Path,
    base_commit: str,
    policy: Mapping[str, Any],
    source_state: Mapping[str, Any],
    package_binding: Mapping[str, Any],
) -> None:
    package_name = "_myquant_phase0_session_candidate_package"
    index_name = "_myquant_phase0_session_candidate_index"
    package_module = _load_module(
        package_name,
        repo_root / "scripts" / "v17_phase0_package_evidence.py",
    )
    index_module = _load_module(
        index_name,
        repo_root / "scripts" / "v17_phase0_evidence_index.py",
    )
    paths: Any = None
    try:
        with _closed_process_environment():
            package_full = package_module._sample_physical_superset(repo_root)
            reduced = {
                "row_count": package_full.get("row_count"),
                "sha256": package_full.get("sha256"),
            }
            if reduced != dict(package_binding):
                raise Phase0SessionError("candidate package source seal mismatch")
            paths = index_module._validate_v2_candidate_module_source_membership(
                policy,
                repo_root=repo_root,
                base_commit=base_commit,
                source_state=source_state,
                package_source_full=package_full,
            )
    except Exception as exc:
        if isinstance(exc, Phase0SessionError):
            raise
        raise Phase0SessionError("candidate source membership validation failed") from exc
    finally:
        sys.modules.pop(index_name, None)
        sys.modules.pop(package_name, None)
    if type(paths) is not list or not paths:
        raise Phase0SessionError("candidate source membership proof is empty")


def _sample_protected_roots(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for identifier, path in _protected_root_specs(repo_root):
        absolute = path.absolute()
        try:
            observed = absolute.lstat()
        except FileNotFoundError:
            rows.append({"id": identifier, "path": str(absolute), "state": "ABSENT"})
            continue
        except OSError as exc:
            raise Phase0SessionError(f"protected root cannot be sampled: {identifier}") from exc
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise Phase0SessionError(
                f"protected root must be absent or a concrete directory: {identifier}"
            )
        try:
            resolved = absolute.resolve(strict=True)
        except OSError as exc:
            raise Phase0SessionError(f"protected root cannot be resolved: {identifier}") from exc
        if resolved != absolute:
            raise Phase0SessionError(f"protected root has symlink indirection: {identifier}")
        rows.append(
            {
                "ctime_ns": observed.st_ctime_ns,
                "id": identifier,
                "mode": f"{stat.S_IMODE(observed.st_mode):04o}",
                "mtime_ns": observed.st_mtime_ns,
                "path": str(absolute),
                "realpath": str(resolved),
                "st_dev": observed.st_dev,
                "st_ino": observed.st_ino,
                "state": "PRESENT_DIRECTORY",
                "uid": observed.st_uid,
            }
        )
    return rows


def _executable_binding(
    path: Path,
    *,
    label: str,
    expected_sha256: str,
    expected_size: int,
) -> tuple[bytes, os.stat_result]:
    raw, observed = _stable_file(path, label=label)
    if (
        _sha256(raw) != expected_sha256
        or len(raw) != expected_size
        or stat.S_IMODE(observed.st_mode) != 0o755
        or not os.access(path, os.X_OK)
    ):
        raise Phase0SessionError(f"{label} frozen binary identity mismatch")
    return raw, observed


def _simple_closed_probe(argv: Sequence[str], *, cwd: Path) -> tuple[bytes, bytes, int]:
    try:
        completed = subprocess.run(
            list(argv),
            cwd=cwd,
            env=BASE_CLOSED_ENVIRONMENT,
            shell=False,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise Phase0SessionError(f"cannot probe frozen tool: {argv[0]}") from exc
    if (
        len(completed.stdout) > MAX_STREAM_BYTES
        or len(completed.stderr) > MAX_STREAM_BYTES
        or len(completed.stdout) + len(completed.stderr) + 16 > MAX_COMMAND_BYTES
    ):
        raise Phase0SessionError(f"frozen tool probe exceeded limits: {argv[0]}")
    return completed.stdout, completed.stderr, completed.returncode


def _uv_cache_binding() -> dict[str, Any]:
    try:
        observed = UV_CACHE.lstat()
        resolved = UV_CACHE.resolve(strict=True)
    except OSError as exc:
        raise Phase0SessionError("frozen uv cache is unavailable") from exc
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or resolved != UV_CACHE
        or observed.st_uid != os.getuid()
        or stat.S_IMODE(observed.st_mode) & 0o022
    ):
        raise Phase0SessionError("frozen uv cache directory identity is unsafe")
    return {
        "mode": f"{stat.S_IMODE(observed.st_mode):04o}",
        "path": str(UV_CACHE),
        "realpath": str(resolved),
        "st_dev": observed.st_dev,
        "st_ino": observed.st_ino,
        "uid": observed.st_uid,
    }


def _canonical_parent_sys_path() -> list[str]:
    paths: list[str] = []
    for index, raw in enumerate(sys.path):
        if type(raw) is not str or not raw:
            raise Phase0SessionError(f"parent sys.path[{index}] is invalid")
        path = Path(raw)
        if not path.is_absolute() or Path(os.path.abspath(path)) != path:
            raise Phase0SessionError(f"parent sys.path[{index}] is not normalized absolute")
        paths.append(str(path))
    if len(paths) != len(set(paths)):
        raise Phase0SessionError("parent sys.path contains duplicate entries")
    return paths


def _pyvenv_cfg_binding(prefix: Path) -> dict[str, Any]:
    path = prefix / "pyvenv.cfg"
    try:
        path.lstat()
    except FileNotFoundError:
        return {"path": str(path), "state": "ABSENT"}
    except OSError as exc:
        raise Phase0SessionError("parent pyvenv.cfg state is unavailable") from exc
    raw, observed = _stable_file(
        path,
        label="parent pyvenv.cfg",
        max_bytes=1024 * 1024,
    )
    return {
        "mode": f"{stat.S_IMODE(observed.st_mode):04o}",
        "path": str(path),
        "sha256": _sha256(raw),
        "size_bytes": len(raw),
        "state": "PRESENT",
    }


def _parent_runtime_binding() -> dict[str, Any]:
    lexical = Path(sys.executable)
    if not lexical.is_absolute() or Path(os.path.abspath(lexical)) != lexical:
        raise Phase0SessionError("parent executable path is not normalized absolute")
    raw, observed = _executable_binding(
        lexical,
        label="parent frozen CPython",
        expected_sha256=BASE_PYTHON_SHA256,
        expected_size=BASE_PYTHON_SIZE,
    )
    del raw
    try:
        resolved = lexical.resolve(strict=True)
        prefix = Path(sys.prefix)
        base_prefix = Path(sys.base_prefix)
        resolved_prefix = prefix.resolve(strict=True)
        resolved_base_prefix = base_prefix.resolve(strict=True)
    except OSError as exc:
        raise Phase0SessionError("parent runtime path identity is unavailable") from exc
    if (
        lexical != BASE_PYTHON
        or resolved != BASE_PYTHON.resolve(strict=True)
        or sys.implementation.name != "cpython"
        or tuple(sys.version_info[:3]) != (3, 13, 7)
        or not prefix.is_absolute()
        or not base_prefix.is_absolute()
        or Path(os.path.abspath(prefix)) != prefix
        or Path(os.path.abspath(base_prefix)) != base_prefix
    ):
        raise Phase0SessionError("parent runtime identity mismatch")
    flags = {
        "dont_write_bytecode": sys.flags.dont_write_bytecode,
        "ignore_environment": sys.flags.ignore_environment,
        "isolated": sys.flags.isolated,
        "no_site": sys.flags.no_site,
        "no_user_site": sys.flags.no_user_site,
        "safe_path": bool(getattr(sys.flags, "safe_path", False)),
    }
    if flags != {
        "dont_write_bytecode": 1,
        "ignore_environment": 1,
        "isolated": 1,
        "no_site": 1,
        "no_user_site": 1,
        "safe_path": True,
    }:
        raise Phase0SessionError("parent runtime flags do not prove -I -S -B")
    paths = _canonical_parent_sys_path()
    return {
        "executable": True,
        "flags": flags,
        "implementation": "cpython",
        "lexical_executable": str(lexical),
        "mode": f"{stat.S_IMODE(observed.st_mode):04o}",
        "pyvenv_cfg": _pyvenv_cfg_binding(prefix),
        "resolved_executable": str(resolved),
        "sha256": BASE_PYTHON_SHA256,
        "size_bytes": BASE_PYTHON_SIZE,
        "sys_base_prefix": str(base_prefix),
        "sys_base_prefix_realpath": str(resolved_base_prefix),
        "sys_path": paths,
        "sys_path_sha256": _sha256(_canonical_bytes(paths)),
        "sys_prefix": str(prefix),
        "sys_prefix_realpath": str(resolved_prefix),
        "version": BASE_PYTHON_VERSION,
        "version_info": [3, 13, 7],
    }


def _sample_pip_status() -> dict[str, Any]:
    loaded_before = sorted(name for name in sys.modules if name == "pip" or name.startswith("pip."))
    try:
        spec = importlib.util.find_spec("pip")
    except (ImportError, AttributeError, ValueError) as exc:
        raise Phase0SessionError("pip visibility cannot be observed safely") from exc
    loaded_after = sorted(name for name in sys.modules if name == "pip" or name.startswith("pip."))
    if loaded_after != loaded_before:
        raise Phase0SessionError("pip visibility probe imported pip")
    paths = _canonical_parent_sys_path()
    site_roots = [
        path
        for path in paths
        if any(part.casefold() in {"site-packages", "dist-packages"} for part in Path(path).parts)
    ]
    search_locations = (
        []
        if spec is None or spec.submodule_search_locations is None
        else [str(Path(path)) for path in spec.submodule_search_locations]
    )
    value = {
        "child_environment_policy": {
            key: BASE_CLOSED_ENVIRONMENT[key] for key in PIP_STATUS_ENV_KEYS
        },
        "loaded_modules": loaded_after,
        "observation_scope": PIP_OBSERVATION_SCOPE,
        "pip_spec": {
            "origin": None if spec is None else spec.origin,
            "search_locations": search_locations,
            "visible": spec is not None,
        },
        "site_sys_path_entries": site_roots,
    }
    if (
        value["loaded_modules"]
        or value["pip_spec"]
        != {
            "origin": None,
            "search_locations": [],
            "visible": False,
        }
        or value["site_sys_path_entries"]
    ):
        raise Phase0SessionError("pip is visible in the isolated parent runtime")
    return value


def _sample_toolchain(repo_root: Path) -> dict[str, Any]:
    _base_raw, base_stat = _executable_binding(
        BASE_PYTHON,
        label="frozen base CPython",
        expected_sha256=BASE_PYTHON_SHA256,
        expected_size=BASE_PYTHON_SIZE,
    )
    _uv_raw, uv_stat = _executable_binding(
        UV_BIN,
        label="frozen uv",
        expected_sha256=UV_SHA256,
        expected_size=UV_SIZE,
    )
    probe_code = (
        "import json,platform,sys;"
        "print(json.dumps({'implementation':platform.python_implementation().lower(),"
        "'version':platform.python_version(),'version_info':list(sys.version_info[:3])},"
        "sort_keys=True,separators=(',',':')))"
    )
    python_stdout, python_stderr, python_code = _simple_closed_probe(
        [str(BASE_PYTHON), "-I", "-S", "-B", "-c", probe_code],
        cwd=repo_root,
    )
    if python_code != 0 or python_stderr:
        raise Phase0SessionError("frozen base CPython probe failed")
    try:
        python_identity = json.loads(python_stdout.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Phase0SessionError("frozen base CPython probe is invalid") from exc
    if python_identity != {
        "implementation": "cpython",
        "version": BASE_PYTHON_VERSION,
        "version_info": [3, 13, 7],
    }:
        raise Phase0SessionError("frozen base CPython runtime identity mismatch")
    uv_stdout, uv_stderr, uv_code = _simple_closed_probe(
        [str(UV_BIN), "--version"],
        cwd=repo_root,
    )
    try:
        uv_output = uv_stdout.decode("utf-8", errors="strict").strip()
    except UnicodeError as exc:
        raise Phase0SessionError("frozen uv version output is invalid") from exc
    if uv_code != 0 or uv_stderr or uv_output != UV_VERSION_OUTPUT:
        raise Phase0SessionError("frozen uv runtime identity mismatch")
    cache = _uv_cache_binding()
    return {
        "base_python": {
            "executable": True,
            "implementation": "cpython",
            "lexical_path": str(BASE_PYTHON),
            "mode": f"{stat.S_IMODE(base_stat.st_mode):04o}",
            "realpath": str(BASE_PYTHON.resolve(strict=True)),
            "sha256": BASE_PYTHON_SHA256,
            "size_bytes": BASE_PYTHON_SIZE,
            "version": BASE_PYTHON_VERSION,
            "version_info": [3, 13, 7],
        },
        "pip_scope": dict(PIP_SCOPE),
        "uv": {
            "executable": True,
            "lexical_path": str(UV_BIN),
            "mode": f"{stat.S_IMODE(uv_stat.st_mode):04o}",
            "output": UV_VERSION_OUTPUT,
            "realpath": str(UV_BIN.resolve(strict=True)),
            "sha256": UV_SHA256,
            "size_bytes": UV_SIZE,
            "version": UV_VERSION,
        },
        "uv_cache": cache,
    }


def _capture_invariants(
    repo_root: Path,
    base_commit: str,
) -> dict[str, Any]:
    source_state, source_binding = _source_snapshot(repo_root, base_commit)
    return {
        "package_source_superset": _package_source_superset(repo_root),
        "parent_runtime": _parent_runtime_binding(),
        "pip_status": _sample_pip_status(),
        "protected_roots": _sample_protected_roots(repo_root),
        "source_binding": source_binding,
        "source_state": source_state,
        "toolchain": _sample_toolchain(repo_root),
    }


def _assert_invariants_equal(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    session: Mapping[str, Any] | None,
    stage: str,
) -> None:
    if _canonical_bytes(before) != _canonical_bytes(after):
        raise Phase0SessionError("source/toolchain/package/protected-root drift", stage=stage)
    if session is not None:
        expected = {
            "package_source_superset": session["package_source_superset"],
            "parent_runtime": session["parent_runtime_binding"],
            "pip_status": session["pip_status_after"],
            "protected_roots": session["protected_roots"],
            "source_binding": session["source_binding"],
            "source_state": session["source_state"],
            "toolchain": session["toolchain_binding"],
        }
        if _canonical_bytes(after) != _canonical_bytes(expected):
            raise Phase0SessionError("stage invariants differ from session seal", stage=stage)


def _schema_bindings(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact_version, relative, schema_id in SCHEMA_REGISTRY:
        binding = _file_binding(
            repo_root / relative,
            label=f"{artifact_version} schema",
            relative_path=relative,
        )
        rows.append(
            {
                "artifact_version": artifact_version,
                "path": relative,
                "schema_id": schema_id,
                "sha256": binding["sha256"],
                "size_bytes": binding["size_bytes"],
            }
        )
    return rows


def _gate_plan() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for ordinal, (role, filename) in enumerate(
        zip(GATE_ROLES, GATE_FILENAMES, strict=True),
        start=1,
    ):
        if role == "native_sync_receipt":
            artifact_version = DEPENDENCY_VERSION
            schema_id = DEPENDENCY_SCHEMA_ID
            producer_path = "scripts/v17_offline_dependency_evidence.py"
            producer_version = DEPENDENCY_PRODUCER_VERSION
            kind = "artifact"
        elif role == "package_parity":
            artifact_version = PACKAGE_VERSION
            schema_id = PACKAGE_SCHEMA_ID
            producer_path = "scripts/v17_phase0_package_evidence.py"
            producer_version = PACKAGE_PRODUCER_VERSION
            kind = "artifact"
        elif role == "hash_freeze_readback":
            artifact_version = HASH_FREEZE_VERSION
            schema_id = HASH_FREEZE_SCHEMA_ID
            producer_path = "scripts/v17_phase0_evidence_index.py"
            producer_version = INDEX_PRODUCER_VERSION
            kind = "artifact"
        elif role == "full_offline_suite":
            artifact_version = MAIN_SUITE_RECEIPT_VERSION
            schema_id = MAIN_SUITE_RECEIPT_SCHEMA_ID
            producer_path = MAIN_SUITE_HARNESS_PATH
            producer_version = MAIN_SUITE_RECEIPT_VERSION
            kind = "log"
        else:
            artifact_version = COMMAND_RECEIPT_VERSION
            schema_id = COMMAND_RECEIPT_SCHEMA_ID
            producer_path = "scripts/v17_phase0_evidence_session.py"
            producer_version = SESSION_PRODUCER_VERSION
            kind = "log"
        result.append(
            {
                "artifact_version": artifact_version,
                "filename": filename,
                "kind": kind,
                "ordinal": ordinal,
                "producer_path": producer_path,
                "producer_version": producer_version,
                "role": role,
                "schema_id": schema_id,
            }
        )
    return result


def _producer_binding(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "scripts" / "v17_phase0_evidence_session.py"
    binding = _file_binding(path, label="session runner")
    return {
        "path": str(path),
        "sha256": binding["sha256"],
        "size_bytes": binding["size_bytes"],
        "version": SESSION_PRODUCER_VERSION,
    }


def _build_session_manifest(
    *,
    repo_root: Path,
    base_commit: str,
    session_id: str,
    classification_binding: Mapping[str, Any],
    skip_baseline_binding: Mapping[str, Any],
    invariants: Mapping[str, Any],
) -> dict[str, Any]:
    toolchain = dict(invariants["toolchain"])
    parent_runtime_before = dict(invariants["parent_runtime"])
    pip_status_before = dict(invariants["pip_status"])
    parent_runtime_after = _parent_runtime_binding()
    pip_status_after = _sample_pip_status()
    if _canonical_bytes(parent_runtime_before) != _canonical_bytes(
        parent_runtime_after
    ) or _canonical_bytes(pip_status_before) != _canonical_bytes(pip_status_after):
        raise Phase0SessionError("parent runtime or pip status drifted during initialization")
    payload = {
        "authority": False,
        "base_commit": base_commit,
        "classification_binding": dict(classification_binding),
        "gate_plan": _gate_plan(),
        "limitations": list(LIMITATIONS),
        "package_source_superset": dict(invariants["package_source_superset"]),
        "parent_runtime_binding": parent_runtime_after,
        "pip_status_after": pip_status_after,
        "pip_status_before": pip_status_before,
        "producer": _producer_binding(repo_root),
        "protected_roots": list(invariants["protected_roots"]),
        "protocol_version": PROTOCOL_VERSION,
        "repo_root": str(repo_root),
        "schemas": _schema_bindings(repo_root),
        "session_id": session_id,
        "skip_baseline_binding": dict(skip_baseline_binding),
        "source_binding": dict(invariants["source_binding"]),
        "source_state": dict(invariants["source_state"]),
        "status": "INITIALIZED",
        "toolchain_binding": toolchain,
        "uv_cache_binding": dict(toolchain["uv_cache"]),
        "version": SESSION_VERSION,
    }
    manifest = _seal(payload)
    _checked_schema(
        manifest,
        repo_root=repo_root,
        relative_path="scripts/schemas/v17_phase0_session.v2.schema.json",
        schema_id=SESSION_SCHEMA_ID,
        artifact_version=SESSION_VERSION,
    )
    return manifest


def _published_json_binding(
    binding: Mapping[str, Any],
    value: Mapping[str, Any],
    *,
    session_id: str,
) -> dict[str, Any]:
    return {
        "path": binding["path"],
        "semantic_sha256": value[SEMANTIC_FIELD],
        "session_id": session_id,
        "sha256": binding["sha256"],
        "size_bytes": binding["size_bytes"],
    }


def _assert_command_has_no_protected_reference(
    argv: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    *,
    repo_root: Path,
) -> None:
    values = [*argv, str(cwd), *environment.keys(), *environment.values()]
    for identifier, protected in _protected_root_specs(repo_root):
        protected_text = str(protected)
        if any(protected_text in value for value in values):
            raise Phase0SessionError(
                f"direct command reference to protected root rejected: {identifier}"
            )


def _run_bounded_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    tool_version: str,
    repo_root: Path,
) -> dict[str, Any]:
    if (
        not argv
        or any(type(item) is not str or not item for item in argv)
        or set(environment) != set(dict(environment))
        or any(type(key) is not str or type(value) is not str for key, value in environment.items())
    ):
        raise Phase0SessionError("command argv/environment is invalid")
    if environment.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise Phase0SessionError("every subprocess must disable bytecode writes")
    _assert_command_has_no_protected_reference(
        argv,
        cwd,
        environment,
        repo_root=repo_root,
    )
    try:
        process = subprocess.Popen(
            list(argv),
            cwd=cwd,
            env=dict(environment),
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise Phase0SessionError(f"cannot execute fixed command: {argv[0]}") from exc
    if process.stdout is None or process.stderr is None:  # pragma: no cover - Popen contract
        process.kill()
        raise Phase0SessionError("command pipes were not created")

    selector = selectors.DefaultSelector()
    stdout_fd = process.stdout.fileno()
    stderr_fd = process.stderr.fileno()
    streams = {
        stdout_fd: ("stdout", bytearray()),
        stderr_fd: ("stderr", bytearray()),
    }
    for descriptor in streams:
        os.set_blocking(descriptor, False)
        selector.register(descriptor, selectors.EVENT_READ)
    overflow = False
    try:
        while selector.get_map():
            for key, _mask in selector.select():
                descriptor = int(key.fd)
                label, buffer = streams[descriptor]
                try:
                    chunk = os.read(descriptor, 64 * 1024)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(descriptor)
                    continue
                buffer.extend(chunk)
                if (
                    len(buffer) > MAX_STREAM_BYTES
                    or sum(len(item[1]) for item in streams.values()) + 16 > MAX_COMMAND_BYTES
                ):
                    overflow = True
                    process.kill()
                    for registered in list(selector.get_map().values()):
                        selector.unregister(registered.fd)
                    break
            if overflow:
                break
        return_code = process.wait()
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    if overflow:
        raise Phase0SessionError(f"command output exceeded hard limits: {argv[0]}")
    signal_number = -return_code if return_code < 0 else None
    exit_code = 128 + signal_number if signal_number is not None else return_code
    return {
        "argv": list(argv),
        "cwd": str(cwd),
        "environment": dict(sorted(environment.items())),
        "exit_code": exit_code,
        "signal": signal_number,
        "stderr": bytes(streams[stderr_fd][1]),
        "stdout": bytes(streams[stdout_fd][1]),
        "tool_version": tool_version,
    }


def _frame_commands(captures: Sequence[Mapping[str, Any]]) -> tuple[bytes, list[dict[str, Any]]]:
    framed = bytearray()
    commands: list[dict[str, Any]] = []
    for ordinal, capture in enumerate(captures, start=1):
        stdout = capture["stdout"]
        stderr = capture["stderr"]
        if type(stdout) is not bytes or type(stderr) is not bytes:
            raise Phase0SessionError("command capture streams must be bytes")
        if (
            len(stdout) > MAX_STREAM_BYTES
            or len(stderr) > MAX_STREAM_BYTES
            or len(stdout) + len(stderr) + 16 > MAX_COMMAND_BYTES
        ):
            raise Phase0SessionError("command capture exceeds framing limits")
        stdout_offset = len(framed) + 8
        framed.extend(struct.pack(">Q", len(stdout)))
        framed.extend(stdout)
        stderr_offset = len(framed) + 8
        framed.extend(struct.pack(">Q", len(stderr)))
        framed.extend(stderr)
        commands.append(
            {
                "argv": list(capture["argv"]),
                "cwd": capture["cwd"],
                "environment": dict(capture["environment"]),
                "exit_code": capture["exit_code"],
                "ordinal": ordinal,
                "signal": capture["signal"],
                "stderr_offset_bytes": stderr_offset,
                "stderr_sha256": _sha256(stderr),
                "stderr_size_bytes": len(stderr),
                "stdout_offset_bytes": stdout_offset,
                "stdout_sha256": _sha256(stdout),
                "stdout_size_bytes": len(stdout),
                "tool_version": capture["tool_version"],
            }
        )
    if len(framed) >= MAX_FILE_BYTES:
        raise Phase0SessionError("framed command output exceeds file limit")
    return bytes(framed), commands


def _pytest_counts(raw: bytes, *, label: str) -> dict[str, int]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise Phase0SessionError(f"{label} output is not strict UTF-8") from exc
    summary_lines = [
        line
        for line in text.splitlines()
        if re.search(
            r"\b(?:passed|failed|skipped|error|errors|xfailed|xpassed)\b",
            line,
        )
    ]
    if not summary_lines:
        raise Phase0SessionError(f"{label} output lacks a pytest summary")
    summary = summary_lines[-1]
    counts = {
        "errors": 0,
        "failed": 0,
        "passed": 0,
        "skipped": 0,
        "xfail": 0,
        "xpass": 0,
    }
    aliases = {
        "error": "errors",
        "errors": "errors",
        "failed": "failed",
        "passed": "passed",
        "skipped": "skipped",
        "xfailed": "xfail",
        "xpassed": "xpass",
    }
    for amount, word in re.findall(
        r"(?<![A-Za-z0-9_])([0-9]+)\s+" r"(passed|failed|skipped|error|errors|xfailed|xpassed)\b",
        summary,
    ):
        counts[aliases[word]] += int(amount)
    if not any(counts.values()):
        raise Phase0SessionError(f"{label} pytest summary is unparsable")
    return counts


def _pytest_skip_rows(raw: bytes, *, label: str) -> list[dict[str, Any]]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise Phase0SessionError(f"{label} output is not strict UTF-8") from exc
    rows: list[dict[str, Any]] = []
    pattern = re.compile(
        r"^SKIPPED \[(?P<count>[1-9][0-9]*)\] "
        r"(?P<path>[^:\r\n]+):(?P<line>[1-9][0-9]*): (?P<reason>[^\r\n]+)$"
    )
    for line in text.splitlines():
        match = pattern.fullmatch(line)
        if match is None:
            continue
        rows.append(
            {
                "count": int(match.group("count")),
                "line": int(match.group("line")),
                "path": match.group("path"),
                "reason": match.group("reason"),
            }
        )
    rows.sort(
        key=lambda row: (
            row["path"].encode("utf-8"),
            row["line"],
            row["reason"].encode("utf-8"),
            row["count"],
        )
    )
    paths = [f"{row['path']}:{row['line']}:{row['reason']}" for row in rows]
    if len(paths) != len(set(paths)):
        raise Phase0SessionError(f"{label} contains duplicate skip rows")
    return rows


def _log_claims(
    role: str,
    captures: Sequence[Mapping[str, Any]],
    framed: bytes,
    *,
    skip_baseline: Mapping[str, Any],
) -> dict[str, Any]:
    if role == "native_sync_log":
        return {"exit_code": captures[0]["exit_code"]}
    if role in {"v2_evidence_tests", "recommended_core_tests", "full_offline_suite"}:
        pytest_capture = captures[-1]
        raw = pytest_capture["stdout"] + pytest_capture["stderr"]
        counts = _pytest_counts(raw, label=role)
        claims: dict[str, Any] = {
            **counts,
            "exit_code": pytest_capture["exit_code"],
        }
        if role == "recommended_core_tests":
            claims["staged_upgrade_exit_code"] = captures[0]["exit_code"]
        if role == "full_offline_suite":
            skip_rows = _pytest_skip_rows(raw, label=role)
            claims["raw_output_sha256"] = _sha256(raw)
            claims["skip_allowlist"] = skip_rows
            baseline_rows = skip_baseline.get("entries")
            if skip_rows != baseline_rows or sum(row["count"] for row in skip_rows) != 42:
                raise Phase0SessionError("full-suite skip rows differ from frozen baseline")
        return claims
    if role == "mypy":
        return {"exit_code": captures[0]["exit_code"]}
    if role == "black":
        text = (captures[0]["stdout"] + captures[0]["stderr"]).decode(
            "utf-8",
            errors="replace",
        )
        return {
            "exit_code": captures[0]["exit_code"],
            "unchanged": "would reformat" not in text.lower(),
        }
    if role == "diff_check":
        return {
            "exit_code": captures[0]["exit_code"],
            "raw_output_sha256": _sha256(captures[0]["stdout"] + captures[0]["stderr"]),
        }
    raise Phase0SessionError(f"claims requested for non-log role: {role}")


def _receipt(
    *,
    role: str,
    filename: str,
    ordinal: int,
    captures: Sequence[Mapping[str, Any]],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    framed: bytes,
    commands: Sequence[Mapping[str, Any]],
    session_binding: Mapping[str, Any],
    producer: Mapping[str, Any],
    claims: Mapping[str, Any],
) -> dict[str, Any]:
    passed = all(command["exit_code"] == 0 and command["signal"] is None for command in commands)
    payload = {
        "claims": dict(claims),
        "commands": list(commands),
        "failure_codes": [] if passed else ["COMMAND_NONZERO_OR_SIGNALED"],
        "framing": COMMAND_FRAMING,
        "limitations": list(LIMITATIONS),
        "outcome": "PASSED" if passed else "FAILED",
        "output_sha256": _sha256(framed),
        "output_size_bytes": len(framed),
        "package_source_superset_after": dict(after["package_source_superset"]),
        "package_source_superset_before": dict(before["package_source_superset"]),
        "producer": dict(producer),
        "protected_roots_after": list(after["protected_roots"]),
        "protected_roots_before": list(before["protected_roots"]),
        "protocol_version": PROTOCOL_VERSION,
        "session_binding": dict(session_binding),
        "source_after": dict(after["source_binding"]),
        "source_before": dict(before["source_binding"]),
        "step": {
            "filename": filename,
            "kind": "log",
            "ordinal": ordinal,
            "role": role,
        },
        "toolchain_after": dict(after["toolchain"]),
        "toolchain_before": dict(before["toolchain"]),
        "version": COMMAND_RECEIPT_VERSION,
    }
    return _seal(payload)


def _publish_log_gate(
    *,
    repo_root: Path,
    base_commit: str,
    session: Mapping[str, Any],
    session_binding: Mapping[str, Any],
    publisher: BundlePublisher,
    role: str,
    ordinal: int,
    filename: str,
    command_specs: Sequence[tuple[Sequence[str], Mapping[str, str], str]],
    skip_baseline: Mapping[str, Any],
) -> None:
    before = _capture_invariants(repo_root, base_commit)
    captures = [
        _run_bounded_command(
            argv,
            cwd=repo_root,
            environment=environment,
            tool_version=tool_version,
            repo_root=repo_root,
        )
        for argv, environment, tool_version in command_specs
    ]
    after = _capture_invariants(repo_root, base_commit)
    _assert_invariants_equal(before, after, session=session, stage=role)
    framed, commands = _frame_commands(captures)
    claims = _log_claims(
        role,
        captures,
        framed,
        skip_baseline=skip_baseline,
    )
    receipt = _receipt(
        role=role,
        filename=filename,
        ordinal=ordinal,
        captures=captures,
        before=before,
        after=after,
        framed=framed,
        commands=commands,
        session_binding=session_binding,
        producer=session["producer"],
        claims=claims,
    )
    _checked_schema(
        receipt,
        repo_root=repo_root,
        relative_path="scripts/schemas/v17_phase0_command_receipt.v2.schema.json",
        schema_id=COMMAND_RECEIPT_SCHEMA_ID,
        artifact_version=COMMAND_RECEIPT_VERSION,
    )
    raw = COMMAND_RECEIPT_PREFIX + _canonical_bytes(receipt) + b"\n" + framed
    publisher.publish(filename, raw)
    if receipt["outcome"] != "PASSED":
        raise Phase0SessionError(f"{role} command failed", stage=role)


def _publish_captured_log_gate(
    *,
    repo_root: Path,
    session: Mapping[str, Any],
    session_binding: Mapping[str, Any],
    publisher: BundlePublisher,
    role: str,
    ordinal: int,
    filename: str,
    captures: Sequence[Mapping[str, Any]],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    skip_baseline: Mapping[str, Any],
) -> None:
    _assert_invariants_equal(before, after, session=session, stage=role)
    framed, commands = _frame_commands(captures)
    claims = _log_claims(
        role,
        captures,
        framed,
        skip_baseline=skip_baseline,
    )
    receipt = _receipt(
        role=role,
        filename=filename,
        ordinal=ordinal,
        captures=captures,
        before=before,
        after=after,
        framed=framed,
        commands=commands,
        session_binding=session_binding,
        producer=session["producer"],
        claims=claims,
    )
    _checked_schema(
        receipt,
        repo_root=repo_root,
        relative_path="scripts/schemas/v17_phase0_command_receipt.v2.schema.json",
        schema_id=COMMAND_RECEIPT_SCHEMA_ID,
        artifact_version=COMMAND_RECEIPT_VERSION,
    )
    raw = COMMAND_RECEIPT_PREFIX + _canonical_bytes(receipt) + b"\n" + framed
    publisher.publish(filename, raw)
    if receipt["outcome"] != "PASSED":
        raise Phase0SessionError(f"{role} command failed", stage=role)


def _fresh_private_directory(path: Path, *, label: str) -> Path:
    if path.exists() or path.is_symlink():
        raise Phase0SessionError(f"{label} must never have existed")
    old_umask = os.umask(0o077)
    try:
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
    except OSError as exc:
        raise Phase0SessionError(f"cannot create {label}") from exc
    finally:
        os.umask(old_umask)
    observed = path.lstat()
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o700
        or observed.st_uid != os.getuid()
    ):
        raise Phase0SessionError(f"{label} is not owner-owned mode 0700")
    return path


def _write_work_resource(path: Path, value: Mapping[str, Any]) -> None:
    raw = _canonical_resource_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise Phase0SessionError(f"cannot create retained work resource: {path.name}") from exc
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise Phase0SessionError(f"short write for retained work resource: {path.name}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    readback, _observed = _stable_file(
        path,
        label=f"retained work resource {path.name}",
        mode=0o600,
        owner=True,
        single_link=True,
    )
    if readback != raw:
        raise Phase0SessionError(f"retained work resource readback mismatch: {path.name}")


DEPENDENCY_BRIDGE = r"""
import importlib.util,json,pathlib,sys
module_path=pathlib.Path(sys.argv[1])
spec=importlib.util.spec_from_file_location("_phase0_dependency_bridge",module_path)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load dependency producer")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
report,accepted=module.build_evidence(
    repo_root=pathlib.Path(sys.argv[2]),
    frozen_export=pathlib.Path(sys.argv[3]),
    target_venv=pathlib.Path(sys.argv[4]),
    target_python=pathlib.Path(sys.argv[5]),
    wheelhouse=pathlib.Path(sys.argv[6]),
    uv_cache=pathlib.Path(sys.argv[7]),
    uv_binary=sys.argv[8],
    native_sync_log=pathlib.Path(sys.argv[9]),
    work_root=pathlib.Path(sys.argv[10]),
)
if not accepted:
    raise RuntimeError("dependency evidence rejected")
raw=json.dumps(report,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")+b"\n"
sys.stdout.buffer.write(raw)
""".strip()


def _run_dependency_evidence(
    *,
    repo_root: Path,
    base_commit: str,
    work_root: Path,
    frozen_export: Path,
    session: Mapping[str, Any],
    session_binding: Mapping[str, Any],
    publisher: BundlePublisher,
) -> None:
    stage = "native_sync_receipt"
    target_venv = work_root / "native_venv"
    target_python = target_venv / "bin" / "python"
    if not target_python.exists():
        raise Phase0SessionError("native sync did not create the fixed target Python", stage=stage)
    wheelhouse = _fresh_private_directory(work_root / "wheelhouse", label="wheelhouse")
    runtime_tmp = _fresh_private_directory(
        work_root / "tmp_dependency",
        label="dependency runtime directory",
    )
    before = _capture_invariants(repo_root, base_commit)
    environment = {
        **BASE_CLOSED_ENVIRONMENT,
        "TMPDIR": str(runtime_tmp),
    }
    capture = _run_bounded_command(
        [
            str(target_python),
            "-I",
            "-B",
            "-c",
            DEPENDENCY_BRIDGE,
            str(repo_root / "scripts" / "v17_offline_dependency_evidence.py"),
            str(repo_root),
            str(frozen_export),
            str(target_venv),
            str(target_python),
            str(wheelhouse),
            str(UV_CACHE),
            str(UV_BIN),
            str(publisher.root / "10_native_sync.log"),
            str(work_root),
        ],
        cwd=repo_root,
        environment=environment,
        tool_version=BASE_PYTHON_VERSION,
        repo_root=repo_root,
    )
    after = _capture_invariants(repo_root, base_commit)
    _assert_invariants_equal(before, after, session=session, stage=stage)
    if capture["exit_code"] != 0 or capture["signal"] is not None or capture["stderr"]:
        raise Phase0SessionError("native dependency evidence builder failed", stage=stage)
    report = _strict_canonical_json(capture["stdout"], label="native dependency evidence")
    if (
        report.get("schema_version") != DEPENDENCY_VERSION
        or report.get("accepted") is not True
        or report.get(SEMANTIC_FIELD) != _semantic_sha256(report)
        or report.get("step")
        != {
            "filename": "20_native_dependency.json",
            "kind": "artifact",
            "ordinal": 2,
            "role": "native_sync_receipt",
        }
        or report.get("session_binding") != dict(session_binding)
        or type(report.get("producer")) is not dict
        or set(report["producer"]) != {"path", "version", "sha256", "size_bytes"}
    ):
        raise Phase0SessionError("native dependency evidence contract mismatch", stage=stage)
    _checked_schema(
        report,
        repo_root=repo_root,
        relative_path="scripts/schemas/v17_offline_dependency_evidence.v2.schema.json",
        schema_id=DEPENDENCY_SCHEMA_ID,
        artifact_version=DEPENDENCY_VERSION,
    )
    publisher.publish("20_native_dependency.json", capture["stdout"])


def _run_package_evidence(
    *,
    repo_root: Path,
    base_commit: str,
    work_root: Path,
    session: Mapping[str, Any],
    session_binding: Mapping[str, Any],
    publisher: BundlePublisher,
) -> None:
    stage = "package_parity"
    source_binding_path = work_root / "source_binding.json"
    _write_work_resource(source_binding_path, session["source_binding"])
    package_work_root = work_root / "package"
    package_tmp = _fresh_private_directory(
        work_root / "tmp_package",
        label="package runtime directory",
    )
    if package_work_root.exists() or package_work_root.is_symlink():
        raise Phase0SessionError("package work root must never have existed", stage=stage)
    before = _capture_invariants(repo_root, base_commit)
    name = "_myquant_phase0_session_package_builder"
    module = _load_module(name, repo_root / "scripts" / "v17_phase0_package_evidence.py")
    try:
        with _closed_process_environment({"TMPDIR": str(package_tmp)}):
            report = module.build_package_evidence(
                repo_root=repo_root,
                expected_base_commit=base_commit,
                session_manifest=publisher.root / "00_session.json",
                expected_source_binding_json=source_binding_path,
                base_python=BASE_PYTHON,
                uv_bin=UV_BIN,
                uv_cache=UV_CACHE,
                work_root=package_work_root,
            )
    except Exception as exc:
        if isinstance(exc, Phase0SessionError):
            raise
        raise Phase0SessionError("package evidence builder failed", stage=stage) from exc
    finally:
        sys.modules.pop(name, None)
    after = _capture_invariants(repo_root, base_commit)
    _assert_invariants_equal(before, after, session=session, stage=stage)
    if (
        type(report) is not dict
        or report.get("version") != PACKAGE_VERSION
        or report.get("accepted") is not True
        or report.get(SEMANTIC_FIELD) != _semantic_sha256(report)
        or report.get("step")
        != {
            "filename": "40_package_parity.json",
            "kind": "artifact",
            "ordinal": 9,
            "role": "package_parity",
        }
        or report.get("session_binding") != dict(session_binding)
    ):
        raise Phase0SessionError("package evidence contract mismatch", stage=stage)
    _checked_schema(
        report,
        repo_root=repo_root,
        relative_path="scripts/schemas/v17_phase0_package_evidence.v2.schema.json",
        schema_id=PACKAGE_SCHEMA_ID,
        artifact_version=PACKAGE_VERSION,
    )
    publisher.publish("40_package_parity.json", _canonical_resource_bytes(report))


def _run_diff_evidence(
    *,
    repo_root: Path,
    base_commit: str,
    work_root: Path,
    session: Mapping[str, Any],
    session_binding: Mapping[str, Any],
    publisher: BundlePublisher,
    skip_baseline: Mapping[str, Any],
) -> None:
    stage = "diff_check"
    before = _capture_invariants(repo_root, base_commit)
    diff_work_root = _fresh_private_directory(
        work_root / "diff",
        label="diff work root",
    )
    name = "_myquant_phase0_session_diff_check"
    module = _load_module(name, repo_root / "scripts" / "v17_phase0_diff_check.py")
    try:
        with _closed_process_environment(
            {
                "TMPDIR": str(work_root / "tmp_diff"),
            }
        ):
            capture = module.run_isolated_diff_check(
                repo_root=repo_root,
                work_root=diff_work_root,
                expected_source_binding=session["source_binding"],
            )
    except Exception as exc:
        if isinstance(exc, Phase0SessionError):
            raise
        raise Phase0SessionError("isolated diff check failed", stage=stage) from exc
    finally:
        sys.modules.pop(name, None)
    after = _capture_invariants(repo_root, base_commit)
    if type(capture) is not dict or set(capture) != {
        "argv",
        "cwd",
        "environment",
        "exit_code",
        "signal",
        "tool_version",
        "stdout",
        "stderr",
    }:
        raise Phase0SessionError(
            "isolated diff helper returned an invalid command capture", stage=stage
        )
    if capture.get("environment", {}).get("PYTHONDONTWRITEBYTECODE") != "1":
        raise Phase0SessionError("isolated diff subprocess did not disable bytecode", stage=stage)
    _assert_command_has_no_protected_reference(
        capture["argv"],
        Path(capture["cwd"]),
        capture["environment"],
        repo_root=repo_root,
    )
    _publish_captured_log_gate(
        repo_root=repo_root,
        session=session,
        session_binding=session_binding,
        publisher=publisher,
        role=stage,
        ordinal=8,
        filename="35_diff_check.log",
        captures=[capture],
        before=before,
        after=after,
        skip_baseline=skip_baseline,
    )


def _run_index_seal(
    *,
    repo_root: Path,
    base_commit: str,
    bundle_root: Path,
    classification_manifest: Path,
    skip_baseline_path: Path,
    session: Mapping[str, Any],
    session_binding: Mapping[str, Any],
) -> dict[str, Any]:
    stage = "hash_freeze_readback"
    before = _capture_invariants(repo_root, base_commit)
    name = "_myquant_phase0_session_final_index"
    module = _load_module(name, repo_root / "scripts" / "v17_phase0_evidence_index.py")
    try:
        with _closed_process_environment():
            result = module.seal_phase0_session(
                repo_root=repo_root,
                bundle_root=bundle_root,
                classification_manifest=classification_manifest,
                skip_baseline=skip_baseline_path,
                session_manifest=bundle_root / "00_session.json",
                expected_session_sha256=session_binding["sha256"],
            )
    except Exception as exc:
        if isinstance(exc, Phase0SessionError):
            raise
        raise Phase0SessionError("final evidence index seal failed", stage=stage) from exc
    finally:
        sys.modules.pop(name, None)
    after = _capture_invariants(repo_root, base_commit)
    _assert_invariants_equal(before, after, session=session, stage=stage)
    index_path = bundle_root / "70_evidence_index.json"
    sidecar_path = bundle_root / "70_evidence_index.json.sha256"
    index_raw, _index_stat = _stable_file(
        index_path,
        label="final evidence index",
        mode=0o600,
        owner=True,
        single_link=True,
    )
    sidecar_raw, _sidecar_stat = _stable_file(
        sidecar_path,
        label="final evidence index SHA sidecar",
        mode=0o600,
        owner=True,
        single_link=True,
    )
    index = _strict_canonical_json(index_raw, label="final evidence index")
    if (
        index.get("version") != EVIDENCE_INDEX_VERSION
        or index.get("status") != "SEALED"
        or index.get("authority") is not False
        or index.get(SEMANTIC_FIELD) != _semantic_sha256(index)
        or sidecar_raw != _index_sidecar_bytes(index_raw, filename=index_path.name)
        or type(result) is not dict
    ):
        raise Phase0SessionError("final evidence index readback is invalid", stage=stage)
    return index


def _validate_skip_baseline(value: Mapping[str, Any], *, repo_root: Path) -> None:
    entries = value.get("entries")
    claims = value.get("claims")
    if (
        value.get("version") != SKIP_BASELINE_VERSION
        or value.get("status") != "FROZEN"
        or value.get("accepted") is not True
        or value.get("authority") is not False
        or value.get("expected_skip_count") != 42
        or value.get("observed_skip_count") != 42
        or value.get("limitations") != LIMITATIONS
        or type(entries) is not list
        or type(claims) is not dict
        or claims.get("exit_code") != 0
        or claims.get("skipped") != 42
        or any(claims.get(key) != 0 for key in ("errors", "failed", "xfail", "xpass"))
        or sum(
            entry.get("count", 0)
            for entry in entries
            if type(entry) is dict and type(entry.get("count")) is int
        )
        != 42
    ):
        raise Phase0SessionError("skip baseline is not the accepted frozen 42-skip baseline")
    name = "_myquant_phase0_session_skip_validator"
    module = _load_module(name, repo_root / "scripts" / "v17_phase0_skip_baseline.py")
    try:
        with _closed_process_environment():
            validated = module.validate_skip_baseline(dict(value), repo_root=repo_root)
    except Exception as exc:
        if isinstance(exc, Phase0SessionError):
            raise
        raise Phase0SessionError("skip baseline semantic revalidation failed") from exc
    finally:
        sys.modules.pop(name, None)
    if validated != dict(value):
        raise Phase0SessionError("skip baseline validator changed the frozen artifact")


def _validate_runtime_invocation() -> None:
    _parent_runtime_binding()
    _sample_pip_status()
    sys.dont_write_bytecode = True


def _validate_frozen_export(path: Path, *, repo_root: Path) -> Path:
    absolute = _validate_external_location(
        path,
        repo_root=repo_root,
        label="frozen uv export",
    )
    raw, _observed = _stable_file(
        absolute,
        label="frozen uv export",
        mode=0o600,
        owner=True,
        single_link=True,
    )
    try:
        header = raw.decode("utf-8", errors="strict").splitlines()[:8]
    except UnicodeError as exc:
        raise Phase0SessionError("frozen uv export is not strict UTF-8") from exc
    joined = "\n".join(header)
    for token in ("uv export", "--frozen", "--no-hashes", "--no-emit-project"):
        if token not in joined:
            raise Phase0SessionError(f"frozen uv export header lacks {token}")
    return absolute


def _stage_environment(
    work_root: Path,
    stage: str,
    *,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    root = _fresh_private_directory(
        work_root / f"runtime_{stage}",
        label=f"{stage} runtime directory",
    )
    home = _fresh_private_directory(root / "home", label=f"{stage} HOME")
    tmp = _fresh_private_directory(root / "tmp", label=f"{stage} TMPDIR")
    cache = _fresh_private_directory(root / "cache", label=f"{stage} cache")
    environment = {
        **BASE_CLOSED_ENVIRONMENT,
        "BLACK_CACHE_DIR": str(cache / "black"),
        "HOME": str(home),
        "MYPY_CACHE_DIR": str(cache / "mypy"),
        "TMPDIR": str(tmp),
        "XDG_CACHE_HOME": str(cache),
    }
    if extra:
        environment.update(extra)
    return environment


def _main_suite_contract(
    repo_root: Path,
) -> tuple[ModuleType, dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    harness_path = repo_root / MAIN_SUITE_HARNESS_PATH
    harness_before = _file_binding(
        harness_path,
        label="main-suite harness",
        relative_path=MAIN_SUITE_HARNESS_PATH,
    )
    harness = _load_module("_myquant_phase0_main_suite_harness", harness_path)
    if (
        getattr(harness, "RECEIPT_PREFIX", None) != MAIN_SUITE_RECEIPT_PREFIX
        or getattr(harness, "RECEIPT_VERSION", None) != MAIN_SUITE_RECEIPT_VERSION
        or getattr(harness, "RECEIPT_SCHEMA_ID", None) != MAIN_SUITE_RECEIPT_SCHEMA_ID
        or getattr(harness, "POLICY_VERSION", None) != MAIN_SUITE_RUNTIME_POLICY_VERSION
        or getattr(harness, "POLICY_SCHEMA_ID", None) != MAIN_SUITE_RUNTIME_POLICY_SCHEMA_ID
        or not callable(getattr(harness, "validate_policy_bytes", None))
        or not callable(getattr(harness, "run_main_suite", None))
    ):
        raise Phase0SessionError("main-suite harness public contract mismatch")

    policy_path = repo_root / MAIN_SUITE_POLICY_PATH
    policy_raw, _observed = _stable_file(
        policy_path,
        label="main-suite runtime policy",
    )
    policy_document = _strict_canonical_json(
        policy_raw,
        label="main-suite runtime policy",
    )
    _checked_schema(
        policy_document,
        repo_root=repo_root,
        relative_path=MAIN_SUITE_POLICY_SCHEMA_PATH,
        schema_id=MAIN_SUITE_RUNTIME_POLICY_SCHEMA_ID,
        artifact_version=MAIN_SUITE_RUNTIME_POLICY_VERSION,
    )
    try:
        policy = harness.validate_policy_bytes(policy_raw)
    except Exception as exc:
        raise Phase0SessionError("main-suite runtime policy validation failed") from exc
    if (
        type(policy) is not dict
        or policy_raw != _canonical_resource_bytes(policy)
        or policy.get("version") != MAIN_SUITE_RUNTIME_POLICY_VERSION
        or policy.get("schema_id") != MAIN_SUITE_RUNTIME_POLICY_SCHEMA_ID
        or policy.get("semantic_sha256") != _semantic_sha256(policy)
        or policy.get("discovery_mode") is not False
        or policy.get("candidate_root") != str(repo_root)
        or policy.get("limitations") != LIMITATIONS
        or policy.get("pytest_args") != list(MAIN_SUITE_PYTEST_ARGS)
    ):
        raise Phase0SessionError("main-suite runtime policy identity mismatch")
    policy_bindings = _main_suite_policy_bindings(repo_root)
    if policy_bindings["policy_binding"]["sha256"] != _sha256(policy_raw) or policy_bindings[
        "policy_binding"
    ]["size_bytes"] != len(policy_raw):
        raise Phase0SessionError("main-suite runtime policy changed during validation")
    return harness, policy, policy_bindings, harness_before


def _main_suite_policy_bindings(
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    policy_path = repo_root / MAIN_SUITE_POLICY_PATH
    manifest_path = repo_root / MAIN_SUITE_PACKAGE_MANIFEST_PATH
    schema_path = repo_root / MAIN_SUITE_POLICY_SCHEMA_PATH
    policy_binding = _main_suite_file_binding(
        policy_path,
        label="main-suite runtime policy binding",
    )
    manifest_binding = _main_suite_file_binding(
        manifest_path,
        label="main-suite package manifest binding",
    )
    schema_binding = _main_suite_file_binding(
        schema_path,
        label="main-suite runtime policy schema binding",
    )
    manifest_raw, _observed = _stable_file(
        manifest_path,
        label="main-suite package manifest",
        single_link=True,
    )
    manifest = _strict_canonical_json(
        manifest_raw,
        label="main-suite package manifest",
    )
    resource_rows = [
        row
        for row in manifest.get("resources", [])
        if type(row) is dict
        and row.get("relative_path") == "resources/main_suite_runtime_policy.v1.json"
    ]
    schema_rows = [
        row
        for row in manifest.get("schemas", [])
        if type(row) is dict
        and row.get("relative_path") == "schemas/main_suite_runtime_policy.v1.schema.json"
    ]
    if (
        manifest.get("version") != "myquant.v17.v2.package-manifest.v1"
        or manifest.get("authority") is not False
        or len(resource_rows) != 1
        or resource_rows[0].get("resource_version") != MAIN_SUITE_RUNTIME_POLICY_VERSION
        or resource_rows[0].get("byte_sha256") != policy_binding["sha256"]
        or len(schema_rows) != 1
        or schema_rows[0].get("schema_id") != MAIN_SUITE_RUNTIME_POLICY_SCHEMA_ID
        or schema_rows[0].get("byte_sha256") != schema_binding["sha256"]
    ):
        raise Phase0SessionError("main-suite package manifest binding mismatch")
    return {
        "policy_binding": policy_binding,
        "policy_manifest_binding": manifest_binding,
        "policy_schema_binding": schema_binding,
    }


def _main_suite_environment(
    work_root: Path,
    *,
    stage: str,
    policy: Mapping[str, Any],
) -> dict[str, str]:
    environment_policy = policy.get("pytest_environment")
    if type(environment_policy) is not dict or set(environment_policy) != {
        "allowed_keys",
        "dynamic_path_keys",
        "forbidden",
        "path_topology",
        "required",
    }:
        raise Phase0SessionError("main-suite environment policy shape mismatch")
    required = environment_policy["required"]
    allowed = environment_policy["allowed_keys"]
    dynamic = environment_policy["dynamic_path_keys"]
    forbidden = environment_policy["forbidden"]
    topology = environment_policy["path_topology"]
    if (
        type(required) is not dict
        or type(allowed) is not list
        or type(dynamic) is not list
        or type(forbidden) is not list
        or topology != MAIN_SUITE_PATH_TOPOLOGY
        or any(type(key) is not str or type(value) is not str for key, value in required.items())
        or any(type(key) is not str for key in [*allowed, *dynamic, *forbidden])
        or len(allowed) != len(set(allowed))
        or len(dynamic) != len(set(dynamic))
        or set(allowed) != set(required) | set(dynamic)
        or set(forbidden) & set(allowed)
        or set(dynamic)
        != set(MAIN_SUITE_PATH_TOPOLOGY["closed_root_siblings"])
        | set(MAIN_SUITE_PATH_TOPOLOGY["cache_children"])
    ):
        raise Phase0SessionError("main-suite environment policy is invalid")
    runtime = _fresh_private_directory(
        work_root / f"runtime_{stage}",
        label=f"{stage} runtime directory",
    )
    environment = dict(required)
    for key in dynamic:
        if re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", key, re.ASCII) is None:
            raise Phase0SessionError("main-suite dynamic environment key is invalid")
    siblings: dict[str, Path] = {}
    for key in MAIN_SUITE_PATH_TOPOLOGY["closed_root_siblings"]:
        siblings[key] = _fresh_private_directory(
            runtime / f"path_{key.casefold()}",
            label=f"{stage} {key}",
        )
        environment[key] = str(siblings[key])
    cache = siblings["XDG_CACHE_HOME"]
    for key in MAIN_SUITE_PATH_TOPOLOGY["cache_children"]:
        child = _fresh_private_directory(
            cache / f"path_{key.casefold()}",
            label=f"{stage} {key}",
        )
        environment[key] = str(child)
    if set(environment) != set(allowed):
        raise Phase0SessionError("main-suite environment closure mismatch")
    return dict(sorted(environment.items()))


def _validate_main_suite_result(
    value: Any,
    *,
    repo_root: Path,
    policy: Mapping[str, Any],
    policy_bindings: Mapping[str, Mapping[str, Any]],
    expected_environment: Mapping[str, str],
    expected_pycache_binding: Mapping[str, Any],
    challenge_binding_kind: str,
    challenge_binding_sha256: str,
) -> dict[str, Any]:
    if challenge_binding_kind not in {"SKIP_SOURCE_STATE", "PHASE0_SESSION_FILE"}:
        raise Phase0SessionError("main-suite challenge binding kind is invalid")
    if SHA256_RE.fullmatch(challenge_binding_sha256) is None:
        raise Phase0SessionError("main-suite challenge binding SHA-256 is invalid")
    if type(value) is not dict or set(value) != {
        "attestation",
        "raw",
        "receipt",
        "stderr",
        "stdout",
    }:
        raise Phase0SessionError("main-suite harness result shape mismatch")
    raw = value["raw"]
    receipt = value["receipt"]
    stdout = value["stdout"]
    stderr = value["stderr"]
    attestation = value["attestation"]
    if (
        type(raw) is not bytes
        or type(receipt) is not dict
        or type(stdout) is not bytes
        or type(stderr) is not bytes
        or type(attestation) is not bytes
    ):
        raise Phase0SessionError("main-suite harness result types mismatch")
    if (
        len(stdout) > MAX_STREAM_BYTES
        or len(stderr) > MAX_STREAM_BYTES
        or len(raw) >= MAX_FILE_BYTES
    ):
        raise Phase0SessionError("main-suite harness result exceeds byte limits")
    framed_tail = (
        struct.pack(">Q", len(stdout))
        + stdout
        + struct.pack(">Q", len(stderr))
        + stderr
        + struct.pack(">Q", len(attestation))
        + attestation
    )
    receipt_raw = _canonical_bytes(receipt)
    expected_raw = MAIN_SUITE_RECEIPT_PREFIX + receipt_raw + b"\n" + framed_tail
    if raw != expected_raw or not raw.startswith(MAIN_SUITE_RECEIPT_PREFIX):
        raise Phase0SessionError("main-suite raw receipt framing mismatch")
    if (
        receipt.get("version") != MAIN_SUITE_RECEIPT_VERSION
        or receipt.get("schema_id") != MAIN_SUITE_RECEIPT_SCHEMA_ID
        or receipt.get("protocol_version") != PROTOCOL_VERSION
        or receipt.get("authority") is not False
        or receipt.get("limitations") != LIMITATIONS
        or receipt.get("framing") != MAIN_SUITE_FRAMING
        or receipt.get("semantic_sha256") != _semantic_sha256(receipt)
        or receipt.get("challenge_binding")
        != {
            "kind": challenge_binding_kind,
            "sha256": challenge_binding_sha256,
        }
    ):
        raise Phase0SessionError("main-suite receipt identity mismatch")
    expected_command = _main_suite_expected_command(
        repo_root=repo_root,
        policy=policy,
        policy_bindings=policy_bindings,
        environment=expected_environment,
    )
    if receipt.get("command") != expected_command:
        raise Phase0SessionError("main-suite command binding mismatch")
    claims = receipt.get("claims")
    if type(claims) is not dict:
        raise Phase0SessionError("main-suite receipt claims shape mismatch")
    if any(
        receipt.get(name) != policy_bindings.get(name)
        for name in (
            "policy_binding",
            "policy_manifest_binding",
            "policy_schema_binding",
        )
    ):
        raise Phase0SessionError("main-suite policy receipt binding mismatch")
    if (
        type(receipt.get("failures")) is not list
        or type(receipt.get("failure_codes")) is not list
        or type(receipt.get("finalization")) is not dict
    ):
        raise Phase0SessionError("main-suite rejection/finalization shape mismatch")
    streams = receipt.get("streams")
    expected_streams = {
        "attestation": {
            "offset_bytes": 8 + len(stdout) + 8 + len(stderr) + 8,
            "sha256": _sha256(attestation),
            "size_bytes": len(attestation),
        },
        "stderr": {
            "offset_bytes": 8 + len(stdout) + 8,
            "sha256": _sha256(stderr),
            "size_bytes": len(stderr),
        },
        "stdout": {
            "offset_bytes": 8,
            "sha256": _sha256(stdout),
            "size_bytes": len(stdout),
        },
        "tail_sha256": _sha256(framed_tail),
        "tail_size_bytes": len(framed_tail),
    }
    if streams != expected_streams:
        raise Phase0SessionError("main-suite stream binding mismatch")
    frames = receipt.get("attestations")
    if type(frames) is not list or len(frames) > 3:
        raise Phase0SessionError("main-suite attestation frame count mismatch")
    expected_bytecode_policy = {
        "dont_write_bytecode": True,
        "pycache_prefix": expected_environment["PYTHONPYCACHEPREFIX"],
    }
    for ordinal, frame in enumerate(frames, start=1):
        payload = frame.get("payload") if type(frame) is dict else None
        if (
            type(payload) is not dict
            or frame.get("phase") != ordinal
            or payload.get("challenge_binding_sha256") != challenge_binding_sha256
        ):
            raise Phase0SessionError("main-suite attestation challenge binding mismatch")
        if ordinal in {1, 2}:
            runtime = payload.get("runtime")
            if (
                type(runtime) is not dict
                or runtime.get("bytecode_policy") != expected_bytecode_policy
            ):
                raise Phase0SessionError("main-suite bytecode policy binding mismatch")
        if ordinal == 1 and payload.get("environment") != dict(
            sorted(expected_environment.items())
        ):
            raise Phase0SessionError("main-suite environment attestation mismatch")
    for snapshot_name in ("external_before", "external_after"):
        snapshot = receipt.get(snapshot_name)
        if snapshot is not None and (
            type(snapshot) is not dict or snapshot.get("pycache_prefix") != expected_pycache_binding
        ):
            raise Phase0SessionError(f"main-suite {snapshot_name} pycache binding mismatch")
    if len(frames) == 3:
        terminal_payload = frames[2]["payload"]
        if (
            terminal_payload.get("frame") != "terminal_complete"
            or type(terminal_payload.get("pytest_exit_code")) is not int
            or type(terminal_payload.get("final_loaded_modules")) is not dict
        ):
            raise Phase0SessionError("main-suite terminal attestation mismatch")
    return dict(value)


def _parse_main_suite_result(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "attestation",
        "raw",
        "receipt",
        "stderr",
        "stdout",
    }:
        raise Phase0SessionError("main-suite harness result shape mismatch")
    raw = value["raw"]
    receipt = value["receipt"]
    stdout = value["stdout"]
    stderr = value["stderr"]
    attestation = value["attestation"]
    if (
        type(raw) is not bytes
        or type(receipt) is not dict
        or type(stdout) is not bytes
        or type(stderr) is not bytes
        or type(attestation) is not bytes
    ):
        raise Phase0SessionError("main-suite harness result types mismatch")
    if (
        len(stdout) > MAX_STREAM_BYTES
        or len(stderr) > MAX_STREAM_BYTES
        or len(raw) >= MAX_FILE_BYTES
    ):
        raise Phase0SessionError("main-suite harness result exceeds byte limits")
    framed_tail = (
        struct.pack(">Q", len(stdout))
        + stdout
        + struct.pack(">Q", len(stderr))
        + stderr
        + struct.pack(">Q", len(attestation))
        + attestation
    )
    expected_raw = MAIN_SUITE_RECEIPT_PREFIX + _canonical_bytes(receipt) + b"\n" + framed_tail
    if raw != expected_raw or not raw.startswith(MAIN_SUITE_RECEIPT_PREFIX):
        raise Phase0SessionError("main-suite raw receipt framing mismatch")
    return dict(value)


def _require_main_suite_accepted(
    value: Mapping[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    receipt = value["receipt"]
    claims = receipt["claims"]
    frames = receipt["attestations"]
    successful = (
        receipt.get("accepted") is True
        and receipt.get("outcome") == "PASSED"
        and receipt.get("failure_codes") == []
        and receipt.get("failures") == []
        and claims.get("exit_code") == 0
        and claims.get("signal") is None
        and claims.get("final_audit_completed") is True
        and claims.get("final_audit_enforced") is True
        and claims.get("offline_policy_enforced") is True
        and claims.get("kernel_egress_attested") is False
        and claims.get("network_unreachability_proven") is False
        and type(receipt.get("external_before")) is dict
        and receipt.get("external_before") == receipt.get("external_after")
        and receipt.get("finalization")
        == {
            "cleanup": {"attempted": True, "status": "PASSED"},
            "external_after": {
                "attempted": True,
                "equal": True,
                "status": "PASSED",
            },
        }
        and len(frames) == 3
        and frames[2]["payload"].get("pytest_exit_code") == 0
    )
    if successful:
        return dict(value)
    failure_codes = receipt.get("failure_codes", [])
    detail = ", ".join(str(code) for code in failure_codes) or "UNSPECIFIED_REJECTION"
    raise MainSuiteRejectedError(
        f"main-suite receipt rejected: {detail}",
        failures=receipt.get("failures", []),
        finalization=receipt.get("finalization", {}),
        stage=stage,
    )


def _validate_main_suite_contract_result(
    value: Any,
    *,
    repo_root: Path,
    policy: Mapping[str, Any],
    policy_bindings: Mapping[str, Mapping[str, Any]],
    expected_environment: Mapping[str, str],
    expected_pycache_binding: Mapping[str, Any],
    challenge_binding_kind: str,
    challenge_binding_sha256: str,
    stage: str,
) -> dict[str, Any]:
    parsed = _parse_main_suite_result(value)
    _checked_schema(
        parsed["receipt"],
        repo_root=repo_root,
        relative_path=MAIN_SUITE_RECEIPT_SCHEMA_PATH,
        schema_id=MAIN_SUITE_RECEIPT_SCHEMA_ID,
        artifact_version=MAIN_SUITE_RECEIPT_VERSION,
    )
    validated = _validate_main_suite_result(
        parsed,
        repo_root=repo_root,
        policy=policy,
        policy_bindings=policy_bindings,
        expected_environment=expected_environment,
        expected_pycache_binding=expected_pycache_binding,
        challenge_binding_kind=challenge_binding_kind,
        challenge_binding_sha256=challenge_binding_sha256,
    )
    return _require_main_suite_accepted(validated, stage=stage)


def _run_main_suite_contract(
    *,
    repo_root: Path,
    base_commit: str,
    work_root: Path,
    stage: str,
    source_state: Mapping[str, Any],
    package_binding: Mapping[str, Any],
    challenge_binding_kind: str,
    challenge_binding_sha256: str,
) -> dict[str, Any]:
    harness, policy, policy_bindings, harness_before = _main_suite_contract(repo_root)
    _validate_candidate_source_membership(
        repo_root=repo_root,
        base_commit=base_commit,
        policy=policy,
        source_state=source_state,
        package_binding=package_binding,
    )
    environment = _main_suite_environment(
        work_root,
        stage=stage,
        policy=policy,
    )
    pycache_prefix = Path(environment["PYTHONPYCACHEPREFIX"])
    pycache_before = _empty_private_directory_binding(
        pycache_prefix,
        label=f"{stage} main-suite pycache prefix",
    )
    try:
        result = harness.run_main_suite(
            repo_root=repo_root,
            policy_path=repo_root / MAIN_SUITE_POLICY_PATH,
            challenge_binding_kind=challenge_binding_kind,
            challenge_binding_sha256=challenge_binding_sha256,
            environment=environment,
            pytest_args=list(MAIN_SUITE_PYTEST_ARGS),
        )
    except Exception as exc:
        raise Phase0SessionError("main-suite harness execution failed", stage=stage) from exc
    pycache_after = _empty_private_directory_binding(
        pycache_prefix,
        label=f"{stage} main-suite pycache prefix",
    )
    if pycache_after != pycache_before:
        raise Phase0SessionError(
            "main-suite pycache prefix changed during execution",
            stage=stage,
        )
    harness_after = _file_binding(
        repo_root / MAIN_SUITE_HARNESS_PATH,
        label="main-suite harness readback",
        relative_path=MAIN_SUITE_HARNESS_PATH,
    )
    if harness_after != harness_before:
        raise Phase0SessionError("main-suite harness changed during execution", stage=stage)
    if _main_suite_policy_bindings(repo_root) != policy_bindings:
        raise Phase0SessionError(
            "main-suite policy binding files changed during execution",
            stage=stage,
        )
    return _validate_main_suite_contract_result(
        result,
        repo_root=repo_root,
        policy=policy,
        policy_bindings=policy_bindings,
        expected_environment=environment,
        expected_pycache_binding=pycache_before,
        challenge_binding_kind=challenge_binding_kind,
        challenge_binding_sha256=challenge_binding_sha256,
        stage=stage,
    )


def _publish_main_suite_gate(
    *,
    repo_root: Path,
    base_commit: str,
    work_root: Path,
    session: Mapping[str, Any],
    session_binding: Mapping[str, Any],
    publisher: BundlePublisher,
    skip_baseline: Mapping[str, Any],
) -> None:
    before = _capture_invariants(repo_root, base_commit)
    session_sha256 = session_binding.get("sha256")
    if type(session_sha256) is not str:
        raise Phase0SessionError("session artifact SHA-256 is missing", stage="full_offline_suite")
    result = _run_main_suite_contract(
        repo_root=repo_root,
        base_commit=base_commit,
        work_root=work_root,
        stage="full_suite",
        source_state=before["source_state"],
        package_binding=before["package_source_superset"],
        challenge_binding_kind="PHASE0_SESSION_FILE",
        challenge_binding_sha256=session_sha256,
    )
    after = _capture_invariants(repo_root, base_commit)
    _assert_invariants_equal(
        before,
        after,
        session=session,
        stage="full_offline_suite",
    )
    transcript = result["stdout"] + result["stderr"]
    counts = _pytest_counts(transcript, label="full_offline_suite")
    skip_rows = _pytest_skip_rows(transcript, label="full_offline_suite")
    if (
        counts["errors"] != 0
        or counts["failed"] != 0
        or counts["xfail"] != 0
        or counts["xpass"] != 0
        or counts["skipped"] != 42
        or skip_rows != skip_baseline.get("entries")
        or sum(row["count"] for row in skip_rows) != 42
    ):
        raise Phase0SessionError(
            "full-suite transcript differs from frozen baseline",
            stage="full_offline_suite",
        )
    publisher.publish("32_full_suite.log", result["raw"])


def _native_environment(work_root: Path) -> dict[str, str]:
    runtime = _fresh_private_directory(
        work_root / "tmp",
        label="native sync runtime directory",
    )
    tmp = _fresh_private_directory(runtime / "native_sync", label="native sync TMPDIR")
    return {
        **BASE_CLOSED_ENVIRONMENT,
        "TMPDIR": str(tmp),
        "UV_PROJECT_ENVIRONMENT": str(work_root / "native_venv"),
    }


def _native_tool_versions(
    target_python: Path,
    *,
    repo_root: Path,
    work_root: Path,
) -> dict[str, str]:
    environment = _stage_environment(work_root, "tool_versions")
    code = (
        "import json;"
        "from importlib.metadata import version;"
        "print(json.dumps({name:version(name) for name in ('pytest','mypy','black')},"
        "sort_keys=True,separators=(',',':')))"
    )
    capture = _run_bounded_command(
        [str(target_python), "-I", "-B", "-c", code],
        cwd=repo_root,
        environment=environment,
        tool_version=BASE_PYTHON_VERSION,
        repo_root=repo_root,
    )
    if capture["exit_code"] != 0 or capture["signal"] is not None or capture["stderr"]:
        raise Phase0SessionError("cannot bind native test-tool versions")
    try:
        versions = json.loads(capture["stdout"].decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Phase0SessionError("native test-tool version probe is invalid") from exc
    if (
        type(versions) is not dict
        or set(versions) != {"black", "mypy", "pytest"}
        or not all(
            type(value) is str and re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", value, re.ASCII)
            for value in versions.values()
        )
    ):
        raise Phase0SessionError("native test-tool version inventory is invalid")
    bash_capture = _run_bounded_command(
        ["/bin/bash", "--version"],
        cwd=repo_root,
        environment=environment,
        tool_version="bash version probe",
        repo_root=repo_root,
    )
    try:
        bash_first = bash_capture["stdout"].decode("utf-8", errors="strict").splitlines()[0]
    except (UnicodeError, IndexError) as exc:
        raise Phase0SessionError("bash version probe is invalid") from exc
    match = re.search(r"version ([0-9]+(?:\.[0-9]+){1,3})", bash_first)
    if bash_capture["exit_code"] != 0 or bash_capture["stderr"] or match is None:
        raise Phase0SessionError("bash version probe failed")
    return {
        "bash": f"bash {match.group(1)}",
        "black": f"black {versions['black']}",
        "mypy": f"mypy {versions['mypy']}",
        "pytest": f"pytest {versions['pytest']}",
    }


def _run_command_log_gates(
    *,
    repo_root: Path,
    base_commit: str,
    work_root: Path,
    session: Mapping[str, Any],
    session_binding: Mapping[str, Any],
    publisher: BundlePublisher,
    skip_baseline: Mapping[str, Any],
) -> None:
    target_python = work_root / "native_venv" / "bin" / "python"
    versions = _native_tool_versions(
        target_python,
        repo_root=repo_root,
        work_root=work_root,
    )
    v2_env = _stage_environment(work_root, "v2_tests")
    _publish_log_gate(
        repo_root=repo_root,
        base_commit=base_commit,
        session=session,
        session_binding=session_binding,
        publisher=publisher,
        role="v2_evidence_tests",
        ordinal=3,
        filename="30_v2_tests.log",
        command_specs=[
            (
                [
                    str(target_python),
                    "-m",
                    "pytest",
                    *V2_EVIDENCE_TESTS,
                    *PYTEST_OPTIONS,
                ],
                v2_env,
                versions["pytest"],
            )
        ],
        skip_baseline=skip_baseline,
    )
    recommended_env = _stage_environment(
        work_root,
        "recommended_core",
        extra={"PYTHON": str(target_python)},
    )
    _publish_log_gate(
        repo_root=repo_root,
        base_commit=base_commit,
        session=session,
        session_binding=session_binding,
        publisher=publisher,
        role="recommended_core_tests",
        ordinal=4,
        filename="31_recommended_core.log",
        command_specs=[
            (
                ["scripts/staged_upgrade_quality_gate.sh"],
                recommended_env,
                versions["bash"],
            ),
            (
                [
                    str(target_python),
                    "-m",
                    "pytest",
                    *RECOMMENDED_CORE_TESTS,
                    *PYTEST_OPTIONS,
                ],
                recommended_env,
                versions["pytest"],
            ),
        ],
        skip_baseline=skip_baseline,
    )
    _publish_main_suite_gate(
        repo_root=repo_root,
        base_commit=base_commit,
        work_root=work_root,
        session=session,
        session_binding=session_binding,
        publisher=publisher,
        skip_baseline=skip_baseline,
    )
    mypy_env = _stage_environment(work_root, "mypy")
    _publish_log_gate(
        repo_root=repo_root,
        base_commit=base_commit,
        session=session,
        session_binding=session_binding,
        publisher=publisher,
        role="mypy",
        ordinal=6,
        filename="33_mypy.log",
        command_specs=[
            (
                [
                    str(target_python),
                    "-m",
                    "mypy",
                    *MYPY_TARGETS,
                ],
                mypy_env,
                versions["mypy"],
            )
        ],
        skip_baseline=skip_baseline,
    )
    black_env = _stage_environment(work_root, "black")
    _publish_log_gate(
        repo_root=repo_root,
        base_commit=base_commit,
        session=session,
        session_binding=session_binding,
        publisher=publisher,
        role="black",
        ordinal=7,
        filename="34_black.log",
        command_specs=[
            (
                [
                    str(target_python),
                    "-m",
                    "black",
                    "--check",
                    *BLACK_TARGETS,
                ],
                black_env,
                versions["black"],
            )
        ],
        skip_baseline=skip_baseline,
    )


def _bundle_visible_bindings(bundle_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        entries = sorted(
            bundle_root.iterdir(),
            key=lambda path: path.name.encode("utf-8"),
        )
    except OSError:
        return rows
    for path in entries:
        if path.name.startswith(".") or path.name == FAILURE_FILENAME:
            continue
        try:
            raw, observed = _stable_file(
                path,
                label=f"partial bundle file {path.name}",
                mode=0o600,
                owner=True,
                single_link=True,
            )
        except Phase0SessionError:
            continue
        rows.append(
            {
                "mode": f"{stat.S_IMODE(observed.st_mode):04o}",
                "path": str(path),
                "sha256": _sha256(raw),
                "size_bytes": len(raw),
            }
        )
    return rows


def _failure_event(
    error: BaseException,
    *,
    ordinal: int,
    phase: str,
) -> dict[str, Any]:
    stage_value = getattr(error, "stage", None)
    stage = (
        stage_value
        if type(stage_value) is str and stage_value
        else ("internal" if phase == "PRIMARY" else phase.casefold())
    )
    message = str(error).strip() or type(error).__name__
    return {
        "exception_type": type(error).__name__[:256],
        "message": message[:2048],
        "ordinal": ordinal,
        "phase": phase,
        "stage": stage[:128],
    }


def _main_suite_failure_events(
    error: MainSuiteRejectedError,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[str, str]]] = {}
    for failure in error.main_suite_failures:
        phase = failure.get("phase")
        code = failure.get("code")
        detail = failure.get("detail")
        if (
            phase not in FAILURE_PHASES
            or type(code) is not str
            or not code
            or type(detail) is not str
            or not detail
        ):
            raise Phase0SessionError("main-suite failure event is invalid")
        grouped.setdefault(phase, []).append((code, detail))
    events: list[dict[str, Any]] = []
    if "PRIMARY" not in grouped:
        events.append(_failure_event(error, ordinal=1, phase="PRIMARY"))
    for phase in ("PRIMARY", "CLEANUP", "EXTERNAL_AFTER"):
        rows = grouped.get(phase)
        if not rows:
            continue
        message = " | ".join(f"{code}: {detail}" for code, detail in rows)
        events.append(
            {
                "exception_type": type(error).__name__,
                "message": message[:2048],
                "ordinal": len(events) + 1,
                "phase": phase,
                "stage": f"main_suite:{rows[0][0]}"[:128],
            }
        )
    return events


def _main_suite_failure_finalization(error: BaseException) -> dict[str, Any]:
    if not isinstance(error, MainSuiteRejectedError):
        return {
            "attempted": False,
            "cleanup": {
                "attempted": False,
                "status": "NOT_ATTEMPTED",
            },
            "external_after": {
                "attempted": False,
                "equal": None,
                "status": "NOT_ATTEMPTED",
            },
        }
    finalization = error.main_suite_finalization
    cleanup = finalization.get("cleanup")
    external_after = finalization.get("external_after")
    if type(cleanup) is not dict or type(external_after) is not dict:
        raise Phase0SessionError("main-suite finalization is invalid")
    return {
        "attempted": True,
        "cleanup": dict(cleanup),
        "external_after": dict(external_after),
    }


def _validate_failure_report(value: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "authority",
        "bundle_entries",
        "bundle_root",
        "failures",
        "limitations",
        "main_suite_finalization",
        "producer",
        "protocol_version",
        "semantic_sha256",
        "session_id",
        "status",
        "version",
        "work_root",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise Phase0SessionError("unpublished failure receipt root shape is invalid")
    if (
        value["authority"] is not False
        or value["status"] != "UNPUBLISHED"
        or value["protocol_version"] != PROTOCOL_VERSION
        or value["version"] != FAILURE_VERSION
        or value["limitations"] != LIMITATIONS
        or value["semantic_sha256"] != _semantic_sha256(value)
    ):
        raise Phase0SessionError("unpublished failure receipt identity is invalid")
    failures = value["failures"]
    if type(failures) is not list or not 1 <= len(failures) <= 3:
        raise Phase0SessionError("unpublished failure receipt must bind one to three failures")
    phases: list[str] = []
    for index, event in enumerate(failures, start=1):
        if type(event) is not dict or set(event) != {
            "exception_type",
            "message",
            "ordinal",
            "phase",
            "stage",
        }:
            raise Phase0SessionError("unpublished failure event shape is invalid")
        phase = event["phase"]
        if (
            event["ordinal"] != index
            or phase not in FAILURE_PHASES
            or not all(
                type(event[key]) is str and bool(event[key])
                for key in ("exception_type", "message", "stage")
            )
            or len(event["exception_type"]) > 256
            or len(event["message"]) > 2048
            or len(event["stage"]) > 128
        ):
            raise Phase0SessionError("unpublished failure event is invalid")
        phases.append(phase)
    if phases[0] != "PRIMARY" or phases.count("PRIMARY") != 1 or len(phases) != len(set(phases)):
        raise Phase0SessionError("unpublished failure phase order/uniqueness is invalid")
    finalization = value["main_suite_finalization"]
    if type(finalization) is not dict or set(finalization) != {
        "attempted",
        "cleanup",
        "external_after",
    }:
        raise Phase0SessionError("unpublished main-suite finalization shape is invalid")
    cleanup = finalization["cleanup"]
    external_after = finalization["external_after"]
    if (
        type(cleanup) is not dict
        or set(cleanup) != {"attempted", "status"}
        or type(external_after) is not dict
        or set(external_after) != {"attempted", "equal", "status"}
    ):
        raise Phase0SessionError("unpublished main-suite finalization rows are invalid")
    if finalization["attempted"] is False:
        if finalization != MAIN_SUITE_FINALIZATION_NOT_ATTEMPTED:
            raise Phase0SessionError("unpublished pre-harness finalization is invalid")
    elif (
        finalization["attempted"] is not True
        or cleanup["attempted"] is not True
        or cleanup["status"] not in {"PASSED", "FAILED"}
        or external_after["attempted"] is not True
        or external_after["status"] not in {"PASSED", "FAILED"}
        or (external_after["equal"] is not None and type(external_after["equal"]) is not bool)
    ):
        raise Phase0SessionError("unpublished attempted finalization is invalid")
    return dict(value)


def _failure_report(
    *,
    error: BaseException,
    session_id: str,
    repo_root: Path | None,
    bundle_root: Path,
    work_root: Path,
    additional_failures: Sequence[tuple[str, BaseException]] = (),
) -> dict[str, Any]:
    failures = (
        _main_suite_failure_events(error)
        if isinstance(error, MainSuiteRejectedError)
        else [_failure_event(error, ordinal=1, phase="PRIMARY")]
    )
    observed_phases = {event["phase"] for event in failures}
    for phase, additional in additional_failures:
        if phase not in {"CLEANUP", "EXTERNAL_AFTER"} or phase in observed_phases:
            raise Phase0SessionError("additional failure phase is invalid or duplicated")
        if not isinstance(additional, BaseException):
            raise Phase0SessionError("additional failure must be an exception")
        observed_phases.add(phase)
        failures.append(
            _failure_event(
                additional,
                ordinal=len(failures) + 1,
                phase=phase,
            )
        )
    report = _seal(
        {
            "authority": False,
            "bundle_entries": _bundle_visible_bindings(bundle_root),
            "bundle_root": str(bundle_root),
            "failures": failures,
            "limitations": list(LIMITATIONS),
            "main_suite_finalization": _main_suite_failure_finalization(error),
            "producer": (
                _producer_binding(repo_root)
                if repo_root is not None
                else {
                    "path": str(Path(__file__).resolve(strict=False)),
                    "sha256": "0" * 64,
                    "size_bytes": 0,
                    "version": SESSION_PRODUCER_VERSION,
                }
            ),
            "protocol_version": PROTOCOL_VERSION,
            "session_id": session_id,
            "status": "UNPUBLISHED",
            "version": FAILURE_VERSION,
            "work_root": str(work_root),
        }
    )
    return _validate_failure_report(report)


def _assert_sealed_bundle(bundle_root: Path) -> None:
    names = sorted(path.name for path in bundle_root.iterdir())
    if names != sorted(FIXED_BUNDLE_FILES):
        raise Phase0SessionError("sealed bundle file closure mismatch", stage="seal_readback")
    for name in FIXED_BUNDLE_FILES:
        _raw, _observed = _stable_file(
            bundle_root / name,
            label=f"sealed bundle {name}",
            mode=0o600,
            owner=True,
            single_link=True,
        )


def run_session(
    *,
    repo_root: Path,
    classification_manifest: Path,
    skip_baseline: Path,
    frozen_export: Path,
    bundle_root: Path,
    work_root: Path,
) -> dict[str, Any]:
    """Execute the complete closed Phase 0 DAG exactly once."""

    _validate_runtime_invocation()
    repo, base_commit = _resolve_repo(repo_root)
    classification_binding, _classification = _external_json_binding(
        classification_manifest,
        repo_root=repo,
        label="pre-existing classification manifest",
        artifact_version=CLASSIFICATION_VERSION,
        schema_id=CLASSIFICATION_SCHEMA_ID,
        schema_path="scripts/schemas/v17_phase0_pre_existing_classification.v2.schema.json",
    )
    skip_binding, skip_value = _external_json_binding(
        skip_baseline,
        repo_root=repo,
        label="skip baseline",
        artifact_version=SKIP_BASELINE_VERSION,
        schema_id=SKIP_BASELINE_SCHEMA_ID,
        schema_path="scripts/schemas/v17_phase0_skip_baseline.v2.schema.json",
    )
    _validate_skip_baseline(skip_value, repo_root=repo)
    frozen = _validate_frozen_export(frozen_export, repo_root=repo)
    bundle, work = _prepare_new_roots(
        bundle_root,
        work_root,
        repo_root=repo,
    )
    session_id = f"phase0-{secrets.token_hex(16)}"
    publisher = BundlePublisher(bundle, session_id=session_id)
    sealed = False
    try:
        invariants = _capture_invariants(repo, base_commit)
        session = _build_session_manifest(
            repo_root=repo,
            base_commit=base_commit,
            session_id=session_id,
            classification_binding=classification_binding,
            skip_baseline_binding=skip_binding,
            invariants=invariants,
        )
        session_raw = _canonical_resource_bytes(session)
        session_published = publisher.publish("00_session.json", session_raw)
        session_binding = _published_json_binding(
            session_published,
            session,
            session_id=session_id,
        )
        after_session = _capture_invariants(repo, base_commit)
        _assert_invariants_equal(
            invariants,
            after_session,
            session=session,
            stage="session",
        )

        native_env = _native_environment(work)
        _publish_log_gate(
            repo_root=repo,
            base_commit=base_commit,
            session=session,
            session_binding=session_binding,
            publisher=publisher,
            role="native_sync_log",
            ordinal=1,
            filename="10_native_sync.log",
            command_specs=[
                (
                    [
                        str(UV_BIN),
                        "sync",
                        "--python",
                        str(BASE_PYTHON),
                        "--locked",
                        "--all-extras",
                        "--offline",
                    ],
                    native_env,
                    UV_VERSION_OUTPUT,
                )
            ],
            skip_baseline=skip_value,
        )
        _run_dependency_evidence(
            repo_root=repo,
            base_commit=base_commit,
            work_root=work,
            frozen_export=frozen,
            session=session,
            session_binding=session_binding,
            publisher=publisher,
        )
        _run_command_log_gates(
            repo_root=repo,
            base_commit=base_commit,
            work_root=work,
            session=session,
            session_binding=session_binding,
            publisher=publisher,
            skip_baseline=skip_value,
        )
        _run_diff_evidence(
            repo_root=repo,
            base_commit=base_commit,
            work_root=work,
            session=session,
            session_binding=session_binding,
            publisher=publisher,
            skip_baseline=skip_value,
        )
        _run_package_evidence(
            repo_root=repo,
            base_commit=base_commit,
            work_root=work,
            session=session,
            session_binding=session_binding,
            publisher=publisher,
        )
        index = _run_index_seal(
            repo_root=repo,
            base_commit=base_commit,
            bundle_root=bundle,
            classification_manifest=classification_manifest,
            skip_baseline_path=skip_baseline,
            session=session,
            session_binding=session_binding,
        )
        _assert_sealed_bundle(bundle)
        sealed = True
        return index
    except BaseException as exc:
        if not sealed:
            try:
                failure = _failure_report(
                    error=exc,
                    session_id=session_id,
                    repo_root=repo,
                    bundle_root=bundle,
                    work_root=work,
                )
                _checked_schema(
                    failure,
                    repo_root=repo,
                    relative_path="scripts/schemas/v17_phase0_unpublished_failure.v2.schema.json",
                    schema_id=FAILURE_SCHEMA_ID,
                    artifact_version=FAILURE_VERSION,
                )
                publisher.publish(FAILURE_FILENAME, _canonical_resource_bytes(failure))
            except BaseException as failure_exc:
                print(
                    f"v17 Phase 0 failure receipt could not be published: {failure_exc}",
                    file=sys.stderr,
                )
        raise
    finally:
        publisher.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--classification-manifest", type=Path, required=True)
    parser.add_argument("--skip-baseline", type=Path, required=True)
    parser.add_argument("--frozen-export", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        index = run_session(
            repo_root=args.repo_root,
            classification_manifest=args.classification_manifest,
            skip_baseline=args.skip_baseline,
            frozen_export=args.frozen_export,
            bundle_root=args.bundle_root,
            work_root=args.work_root,
        )
    except (Phase0SessionError, OSError, ValueError) as exc:
        print(f"v17 Phase 0 session failed UNPUBLISHED: {exc}", file=sys.stderr)
        return Phase0SessionError.exit_code
    print(
        json.dumps(
            {
                "authority": False,
                "semantic_sha256": index["semantic_sha256"],
                "status": "SEALED",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
