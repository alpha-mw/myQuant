#!/usr/bin/env python3
"""Freeze the exact offline pytest skip baseline for v17 Phase 0.

This producer is intentionally standalone and stdlib-only.  It accepts only
fresh owner-private roots outside the repository and protected v16 roots, runs
one native locked offline sync followed by the complete pytest suite, derives
the skip rows from the captured pytest bytes, and publishes one canonical JSON
resource exact-once.  A failed attempt is not resumable and never publishes a
partial or rejected baseline.
"""

from __future__ import annotations

import argparse
import base64
from email.parser import BytesParser
from email.policy import default as email_policy
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import subprocess
import sys
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

PROTOCOL_VERSION = "myquant.v17.v2"
SKIP_BASELINE_VERSION = "myquant.v17.v2.phase0-skip-baseline.v2"
SKIP_BASELINE_SCHEMA_ID = "myquant.v17.v2.phase0-skip-baseline.schema.v2"
PRODUCER_VERSION = "myquant.v17.v2.phase0-skip-baseline-producer.v2"
PRODUCER_PATH = "scripts/v17_phase0_skip_baseline.py"
SEMANTIC_FIELD = "semantic_sha256"
STATUS = "FROZEN"
EXPECTED_SKIP_COUNT = 42
MAIN_SUITE_RUNTIME_POLICY_VERSION = "myquant.v17.v2.phase0-main-suite-runtime-policy.v1"
MAIN_SUITE_RUNTIME_POLICY_SCHEMA_ID = "myquant.v17.v2.phase0-main-suite-runtime-policy.schema.v1"
MAIN_SUITE_RECEIPT_VERSION = "myquant.v17.v2.phase0-main-suite-receipt.v1"
MAIN_SUITE_RECEIPT_SCHEMA_ID = "myquant.v17.v2.phase0-main-suite-receipt.schema.v1"
MAIN_SUITE_RECEIPT_PREFIX = b"MYQUANT_PHASE0_MAIN_SUITE_RECEIPT="
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
MAIN_SUITE_PACKAGE_MANIFEST_PATH = (
    "quant_investor/v17_v2_contract/resources/package_manifest.v1.json"
)
MAIN_SUITE_HARNESS_PATH = "scripts/v17_phase0_main_suite_harness.py"
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

BASE_PYTHON_PATH = (
    "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/"
    "Python.framework/Versions/3.13/bin/python3.13"
)
BASE_PYTHON_SHA256 = "a708f6e9f4803b806b29146c4e0feecfd9bf2d9eb60f3e15b850cd7cb56f200b"
BASE_PYTHON_SIZE = 52_640
BASE_PYTHON_VERSION = "3.13.7"
BASE_PYTHON_VERSION_INFO = [3, 13, 7]
UV_PATH = "/Users/maxwell/.local/bin/uv"
UV_SHA256 = "bc50ab0e90f24491f0e794f5b8649722f8fd2bf483c53490c012b41b89151ef9"
UV_SIZE = 44_698_848
UV_VERSION = "0.10.9"
UV_VERSION_OUTPUT = "uv 0.10.9 (f675560f3 2026-03-06)"
UV_CACHE_PATH = "/Users/maxwell/.cache/uv"
NATIVE_VENV_NAME = "native-venv"

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
NORMATIVE_LIMITATIONS = [
    "PRECOLLECTION_ALLOWED_TREES_ATTESTED_NOT_COMPLETE_MAIN_RUNTIME_FILESET",
    "NO_DIRECT_LAUNCH_REFERENCE_ONLY_NOT_PROCESS_IO_ATTESTED",
    "OFFLINE_FLAGS_ONLY_NOT_OS_EGRESS_ATTESTED",
    "OWNER_DECLARED_PATHS_ONLY_NOT_CONTENT_PROVENANCE",
    "PIP_25_2_PACKAGE_INSTALL_ENV_ONLY_NATIVE_AND_BUILD_ENV_PIP_ABSENT",
]
AUTHORITY_REPO_ROOT = Path("/Users/maxwell/mySpace/myQuant")
PACKAGE_EXTRA_PATHS = ("README.md", "pyproject.toml", "requirements.txt")
SAFE_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
RAW_OUTPUT_FRAMING = "UINT64_BE_STDOUT_THEN_STDOUT_UINT64_BE_STDERR_THEN_STDERR"
MAX_CAPTURE_BYTES = 256 * 1024 * 1024
PIP_OBSERVATION_SCOPE = "NON_IMPORTING_PARENT_VISIBILITY_AND_FIXED_CHILD_ENVIRONMENT_ONLY"
PIP_CHILD_ENVIRONMENT_POLICY = {
    "PIP_CONFIG_FILE": "/dev/null",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PIP_NO_INDEX": "1",
    "PIP_NO_INPUT": "1",
    "UV_NO_CONFIG": "1",
    "UV_OFFLINE": "1",
    "UV_PYTHON_DOWNLOADS": "never",
}

SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$", re.ASCII)
SKIP_LINE_RE = re.compile(
    r"^SKIPPED \[(?P<count>[1-9][0-9]*)\] "
    r"(?P<path>[^:\r\n]+):(?P<line>[1-9][0-9]*): (?P<reason>[^\r\n]+)$",
    re.ASCII,
)
SUMMARY_TOKEN_RE = re.compile(
    r"(?P<count>[0-9]+) " r"(?P<kind>passed|skipped|failed|errors?|xfailed|xpassed)(?:,| in |$)",
    re.ASCII,
)

ROOT_KEYS = {
    "accepted",
    "authority",
    "challenge_binding",
    "claims",
    "commands",
    "entries",
    "expected_skip_count",
    "limitations",
    "main_suite_attestation",
    "main_suite_raw_binding",
    "main_suite_receipt",
    "observed_skip_count",
    "package_source_superset",
    "parent_runtime_binding",
    "pip_status_after",
    "pip_status_before",
    "producer",
    "protected_roots_after",
    "protected_roots_before",
    "protocol_version",
    "semantic_sha256",
    "source_binding",
    "source_state",
    "status",
    "toolchain_binding",
    "version",
}
SOURCE_BINDING_KEYS = {
    "base_commit",
    "binary_diff_sha256",
    "porcelain_sha256",
    "source_state_sha256",
    "untracked_inventory_sha256",
}
COMMAND_KEYS = {
    "argv",
    "cwd",
    "environment",
    "exit_code",
    "ordinal",
    "signal",
    "stderr",
    "stdout",
    "tool_version",
}
CLAIMS_KEYS = {
    "errors",
    "exit_code",
    "failed",
    "passed",
    "raw_output_sha256",
    "skip_allowlist_sha256",
    "skipped",
    "xfail",
    "xpass",
}


class SkipBaselineError(RuntimeError):
    """Raised when a baseline attempt is unsafe, ambiguous, or unsuccessful."""

    exit_code = 2


CommandRunner = Callable[
    [Sequence[str], Path, Mapping[str, str]],
    tuple[int, bytes, bytes],
]
MainSuiteRunner = Callable[..., dict[str, Any]]


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SkipBaselineError("value is not canonical JSON") from exc


def _canonical_resource_bytes(value: Any) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _strict_canonical_json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SkipBaselineError(f"{label} is invalid JSON") from exc
    if type(value) is not dict or raw != _canonical_resource_bytes(value):
        raise SkipBaselineError(f"{label} must be canonical JSON plus newline")
    return value


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _semantic_sha256(value: Mapping[str, Any]) -> str:
    unsealed = dict(value)
    unsealed.pop(SEMANTIC_FIELD, None)
    return _sha256(_canonical_bytes(unsealed))


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    if SEMANTIC_FIELD in value:
        raise SkipBaselineError("semantic_sha256 must not be supplied")
    sealed = dict(value)
    sealed[SEMANTIC_FIELD] = _semantic_sha256(sealed)
    return sealed


def _require_exact_keys(value: Any, keys: set[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise SkipBaselineError(f"{label} must have exact keys {sorted(keys)!r}")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise SkipBaselineError(f"{label} must be a lowercase SHA-256")
    return value


def _mode_string(mode: int) -> str:
    return f"{stat.S_IMODE(mode):04o}"


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _assert_no_symlink_components(path: Path, *, label: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            observed = current.lstat()
        except OSError as exc:
            raise SkipBaselineError(f"{label} component is unavailable: {current}") from exc
        if stat.S_ISLNK(observed.st_mode):
            raise SkipBaselineError(f"{label} cannot contain symlink components")


def _resolve_existing_directory(
    path: Path,
    *,
    label: str,
    owner_private: bool,
) -> Path:
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise SkipBaselineError(f"{label} must be a normalized absolute path")
    _assert_no_symlink_components(path, label=label)
    try:
        observed = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise SkipBaselineError(f"{label} is unavailable") from exc
    if not stat.S_ISDIR(observed.st_mode) or resolved != path:
        raise SkipBaselineError(f"{label} must be a concrete directory")
    if observed.st_uid != os.getuid():
        raise SkipBaselineError(f"{label} must be owned by the current user")
    if owner_private and stat.S_IMODE(observed.st_mode) != 0o700:
        raise SkipBaselineError(f"{label} must have mode 0700")
    return resolved


def _require_fresh_root(path: Path, *, label: str) -> Path:
    resolved = _resolve_existing_directory(path, label=label, owner_private=True)
    try:
        names = tuple(entry.name for entry in os.scandir(resolved))
    except OSError as exc:
        raise SkipBaselineError(f"{label} cannot be listed") from exc
    if names:
        raise SkipBaselineError(f"{label} must be fresh and empty")
    return resolved


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


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


def _validate_isolated_roots(
    *,
    repo_root: Path,
    bundle_root: Path,
    work_root: Path,
) -> tuple[Path, Path]:
    bundle = _require_fresh_root(bundle_root, label="bundle root")
    work = _require_fresh_root(work_root, label="work root")
    if _path_within(bundle, work) or _path_within(work, bundle):
        raise SkipBaselineError("bundle root and work root cannot be nested")
    forbidden = (
        repo_root,
        *(path.absolute() for _identifier, path in _protected_root_specs(repo_root)),
    )
    for label, candidate in (("bundle root", bundle), ("work root", work)):
        for root in forbidden:
            if _path_within(candidate, root) or _path_within(root, candidate):
                raise SkipBaselineError(f"{label} cannot overlap repository/protected roots")
    return bundle, work


def _stable_regular_bytes(
    path: Path,
    *,
    label: str,
    require_nlink_one: bool = True,
) -> tuple[bytes, os.stat_result]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise SkipBaselineError(f"{label} is unavailable") from exc
    if not stat.S_ISREG(before.st_mode) or (require_nlink_one and before.st_nlink != 1):
        raise SkipBaselineError(f"{label} must be a regular non-hardlinked file")
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_CAPTURE_BYTES:
                raise SkipBaselineError(f"{label} exceeds the capture limit")
            chunks.append(chunk)
        raw = b"".join(chunks)
        after_fd = os.fstat(descriptor)
    except OSError as exc:
        raise SkipBaselineError(f"{label} cannot be read safely") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as exc:
        raise SkipBaselineError(f"{label} disappeared during read") from exc
    signature = _stat_signature(before)
    if (
        signature != _stat_signature(opened)
        or signature != _stat_signature(after_fd)
        or signature != _stat_signature(after)
        or len(raw) != before.st_size
    ):
        raise SkipBaselineError(f"{label} changed during read")
    return raw, before


def _raw_binding(raw: bytes) -> dict[str, Any]:
    return {
        "bytes_base64": base64.b64encode(raw).decode("ascii"),
        "encoding": "base64",
        "sha256": _sha256(raw),
        "size_bytes": len(raw),
    }


def _file_binding(path: Path, *, label: str) -> dict[str, Any]:
    raw, observed = _stable_regular_bytes(path, label=label)
    return {
        "executable": bool(observed.st_mode & 0o111),
        "mode": _mode_string(observed.st_mode),
        "realpath": str(path.resolve(strict=True)),
        "sha256": _sha256(raw),
        "size_bytes": len(raw),
    }


def _main_suite_file_binding(path: Path, *, label: str) -> dict[str, Any]:
    raw, observed = _stable_regular_bytes(path, label=label)
    return {
        "gid": observed.st_gid,
        "mode": _mode_string(observed.st_mode),
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
                raise SkipBaselineError(f"{label} is not empty")
        after = path.lstat()
    except SkipBaselineError:
        raise
    except OSError as exc:
        raise SkipBaselineError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISDIR(before.st_mode)
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) != 0o700
        or _stat_signature(before) != _stat_signature(after)
    ):
        raise SkipBaselineError(f"{label} is not a stable owner-private directory")
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
        raise SkipBaselineError("main-suite command policy is invalid")
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


def _checked_schema(
    value: Mapping[str, Any],
    *,
    repo_root: Path,
    relative_path: str,
    schema_id: str,
    artifact_version: str,
) -> None:
    raw, _observed = _stable_regular_bytes(
        repo_root / relative_path,
        label=f"{artifact_version} schema",
    )
    schema = _strict_canonical_json(
        raw,
        label=f"{artifact_version} schema",
    )
    properties = schema.get("properties")
    if (
        schema.get("$id") != schema_id
        or type(properties) is not dict
        or properties.get("version") != {"const": artifact_version}
        and properties.get("schema_version") != {"const": artifact_version}
    ):
        raise SkipBaselineError(f"{artifact_version} schema identity mismatch")

    package_name = "_myquant_phase0_skip_schema"
    contract_root = repo_root / "quant_investor" / "v17_v2_contract"
    loaded_names: list[str] = []
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        package = ModuleType(package_name)
        package.__path__ = [str(contract_root)]  # type: ignore[attr-defined]
        package.__package__ = package_name
        sys.modules[package_name] = package
        loaded_names.append(package_name)
        canonical_name = f"{package_name}.canonical"
        canonical_spec = importlib.util.spec_from_file_location(
            canonical_name,
            contract_root / "canonical.py",
        )
        if canonical_spec is None or canonical_spec.loader is None:
            raise SkipBaselineError("cannot load bound schema canonical helper")
        canonical = importlib.util.module_from_spec(canonical_spec)
        sys.modules[canonical_name] = canonical
        loaded_names.append(canonical_name)
        canonical_spec.loader.exec_module(canonical)

        resources_name = f"{package_name}.resources"
        resources = ModuleType(resources_name)

        def unavailable_resource(*_args: Any, **_kwargs: Any) -> None:
            raise SkipBaselineError("packaged resources are unavailable during schema validation")

        resources.load_packaged_json = unavailable_resource  # type: ignore[attr-defined]
        sys.modules[resources_name] = resources
        loaded_names.append(resources_name)
        validation_name = f"{package_name}.schema_validation"
        validation_spec = importlib.util.spec_from_file_location(
            validation_name,
            contract_root / "schema_validation.py",
        )
        if validation_spec is None or validation_spec.loader is None:
            raise SkipBaselineError("cannot load bound schema validator")
        validation = importlib.util.module_from_spec(validation_spec)
        sys.modules[validation_name] = validation
        loaded_names.append(validation_name)
        validation_spec.loader.exec_module(validation)
        validation.preflight_packaged_schema(schema)
        validation.validate_instance_against_schema(dict(value), schema)
    except SkipBaselineError:
        raise
    except Exception as exc:
        raise SkipBaselineError(f"{artifact_version} schema validation failed") from exc
    finally:
        sys.dont_write_bytecode = previous
        for name in reversed(loaded_names):
            sys.modules.pop(name, None)


def _git_environment() -> dict[str, str]:
    return {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/var/empty",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": SAFE_PATH,
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": "/private/tmp",
    }


def _run_bytes(
    argv: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> bytes:
    try:
        completed = subprocess.run(
            list(argv),
            cwd=cwd,
            env=dict(environment),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise SkipBaselineError(f"command could not start: {argv[0]}") from exc
    if completed.returncode != 0:
        raise SkipBaselineError(f"command failed during preflight: {list(argv)!r}")
    return completed.stdout


def _resolve_repo_root(repo_root: Path) -> Path:
    repo = _resolve_existing_directory(repo_root, label="repo root", owner_private=False)
    top_raw = _run_bytes(
        ("git", "rev-parse", "--show-toplevel"),
        cwd=repo,
        environment=_git_environment(),
    )
    try:
        top = Path(top_raw.decode("utf-8", errors="strict").strip()).resolve(strict=True)
    except (UnicodeError, OSError) as exc:
        raise SkipBaselineError("git returned an invalid top-level path") from exc
    if top != repo:
        raise SkipBaselineError("repo root must be the exact git worktree top level")
    return repo


def _decode_nul_paths(raw: bytes, *, label: str) -> list[str]:
    if raw and not raw.endswith(b"\0"):
        raise SkipBaselineError(f"{label} is not NUL terminated")
    values: list[str] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            value = item.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise SkipBaselineError(f"{label} contains a non-UTF-8 path") from exc
        pure = PurePosixPath(value)
        if (
            pure.is_absolute()
            or pure.as_posix() != value
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise SkipBaselineError(f"{label} contains an unsafe path")
        values.append(value)
    if len(values) != len(set(values)) or len({value.casefold() for value in values}) != len(
        values
    ):
        raise SkipBaselineError(f"{label} contains duplicate/colliding paths")
    return values


def _safe_repo_path(repo_root: Path, relative: str) -> Path:
    candidate = repo_root / Path(*PurePosixPath(relative).parts)
    try:
        candidate.relative_to(repo_root)
    except ValueError as exc:
        raise SkipBaselineError(f"repository path escapes root: {relative}") from exc
    return candidate


def _stable_untracked(repo_root: Path, relative: str) -> dict[str, Any]:
    path = _safe_repo_path(repo_root, relative)
    try:
        observed = path.lstat()
    except OSError as exc:
        raise SkipBaselineError(f"untracked path disappeared: {relative}") from exc
    base = {
        "mode": _mode_string(observed.st_mode),
        "path": relative,
        "size_bytes": observed.st_size,
    }
    if stat.S_ISREG(observed.st_mode):
        raw, after = _stable_regular_bytes(path, label=f"untracked file {relative}")
        if _stat_signature(observed) != _stat_signature(after):
            raise SkipBaselineError(f"untracked file drift: {relative}")
        return {
            **base,
            "sha256": _sha256(raw),
            "symlink_target": None,
            "type": "file",
        }
    if stat.S_ISLNK(observed.st_mode):
        try:
            before_target = os.readlink(path)
            after = path.lstat()
            after_target = os.readlink(path)
            target_raw = before_target.encode("utf-8", errors="strict")
        except (OSError, UnicodeError) as exc:
            raise SkipBaselineError(f"untracked symlink is unstable: {relative}") from exc
        if _stat_signature(observed) != _stat_signature(after) or before_target != after_target:
            raise SkipBaselineError(f"untracked symlink drift: {relative}")
        return {
            **base,
            "sha256": _sha256(target_raw),
            "symlink_target": before_target,
            "type": "symlink",
        }
    raise SkipBaselineError(f"unsupported untracked node: {relative}")


def _git_snapshot(repo_root: Path) -> dict[str, Any]:
    environment = _git_environment()
    head_raw = _run_bytes(
        ("git", "rev-parse", "--verify", "HEAD"),
        cwd=repo_root,
        environment=environment,
    )
    try:
        base_commit = head_raw.decode("ascii", errors="strict").strip()
    except UnicodeError as exc:
        raise SkipBaselineError("git HEAD is not ASCII") from exc
    if COMMIT_RE.fullmatch(base_commit) is None:
        raise SkipBaselineError("git HEAD is not a full object id")
    porcelain = _run_bytes(
        ("git", "status", "--porcelain=v1", "-z", "--untracked-files=all"),
        cwd=repo_root,
        environment=environment,
    )
    binary_diff = _run_bytes(
        (
            "git",
            "diff",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            base_commit,
            "--",
        ),
        cwd=repo_root,
        environment=environment,
    )
    tracked_raw = _run_bytes(
        (
            "git",
            "diff",
            "--name-only",
            "-z",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            base_commit,
            "--",
        ),
        cwd=repo_root,
        environment=environment,
    )
    untracked_raw = _run_bytes(
        ("git", "ls-files", "--others", "--exclude-standard", "-z"),
        cwd=repo_root,
        environment=environment,
    )
    tracked = _decode_nul_paths(tracked_raw, label="tracked path inventory")
    untracked_paths = _decode_nul_paths(untracked_raw, label="untracked path inventory")
    dirty_paths = sorted(
        set(tracked).union(untracked_paths),
        key=lambda value: value.encode("utf-8"),
    )
    if len({value.casefold() for value in dirty_paths}) != len(dirty_paths):
        raise SkipBaselineError("dirty path inventory contains casefold collisions")
    untracked = [
        _stable_untracked(repo_root, relative)
        for relative in sorted(untracked_paths, key=lambda value: value.encode("utf-8"))
    ]
    public = {
        "base_commit": base_commit,
        "binary_diff_from_base": _raw_binding(binary_diff),
        "dirty_paths": dirty_paths,
        "porcelain_v1_z": _raw_binding(porcelain),
        "untracked": untracked,
    }
    return {
        **public,
        "source_state_sha256": _sha256(_canonical_bytes(public)),
        "_guards": {
            "head": head_raw,
            "porcelain": porcelain,
            "binary_diff": binary_diff,
            "tracked": tracked_raw,
            "untracked": untracked_raw,
        },
    }


def _public_source_state(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in snapshot.items() if key != "_guards"}


def _source_binding_from_state(source_state: Mapping[str, Any]) -> dict[str, str]:
    binary = _require_exact_keys(
        source_state.get("binary_diff_from_base"),
        {"bytes_base64", "encoding", "sha256", "size_bytes"},
        label="source_state.binary_diff_from_base",
    )
    porcelain = _require_exact_keys(
        source_state.get("porcelain_v1_z"),
        {"bytes_base64", "encoding", "sha256", "size_bytes"},
        label="source_state.porcelain_v1_z",
    )
    untracked = source_state.get("untracked")
    if type(untracked) is not list:
        raise SkipBaselineError("source_state.untracked must be an array")
    binding = {
        "base_commit": str(source_state.get("base_commit")),
        "binary_diff_sha256": str(binary.get("sha256")),
        "porcelain_sha256": str(porcelain.get("sha256")),
        "source_state_sha256": str(source_state.get("source_state_sha256")),
        "untracked_inventory_sha256": _sha256(_canonical_bytes(untracked)),
    }
    _validate_source_binding(binding)
    return binding


def _validate_source_binding(value: Any) -> dict[str, str]:
    binding = _require_exact_keys(value, SOURCE_BINDING_KEYS, label="source_binding")
    for key, item in binding.items():
        if type(item) is not str:
            raise SkipBaselineError(f"source_binding.{key} must be a string")
        if key == "base_commit":
            if COMMIT_RE.fullmatch(item) is None:
                raise SkipBaselineError("source_binding.base_commit is invalid")
        else:
            _require_sha256(item, label=f"source_binding.{key}")
    return dict(binding)


def _assert_snapshot_equal(first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    if _canonical_bytes(_public_source_state(first)) != _canonical_bytes(
        _public_source_state(second)
    ) or first.get("_guards") != second.get("_guards"):
        raise SkipBaselineError("repository source state changed during baseline run")


def _scan_package_source_superset(
    repo_root: Path,
) -> tuple[dict[str, Any], list[Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    guards: list[Any] = []
    identities: dict[tuple[int, int], str] = {}

    def record_file(path: Path, relative: str) -> None:
        raw, observed = _stable_regular_bytes(
            path,
            label=f"package source {relative}",
        )
        identity = (observed.st_dev, observed.st_ino)
        if identity in identities:
            raise SkipBaselineError(
                f"package source contains hardlink collision: {identities[identity]} / {relative}"
            )
        identities[identity] = relative
        rows.append(
            {
                "kind": "file",
                "mode": stat.S_IMODE(observed.st_mode),
                "path": relative,
                "sha256": _sha256(raw),
                "size_bytes": len(raw),
            }
        )
        guards.append((relative, _stat_signature(observed)))

    def scan_directory(path: Path, relative: str) -> None:
        try:
            observed = path.lstat()
            names_before = tuple(sorted(entry.name for entry in os.scandir(path)))
        except OSError as exc:
            raise SkipBaselineError(f"package directory is unstable: {relative}") from exc
        if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
            raise SkipBaselineError(f"package directory must be concrete: {relative}")
        rows.append(
            {
                "kind": "directory",
                "mode": stat.S_IMODE(observed.st_mode),
                "path": relative,
                "sha256": None,
                "size_bytes": 0,
            }
        )
        guards.append((relative, _stat_signature(observed), names_before))
        for name in names_before:
            child = path / name
            child_relative = f"{relative}/{name}"
            try:
                child_stat = child.lstat()
            except OSError as exc:
                raise SkipBaselineError(
                    f"package namespace changed while scanning: {child_relative}"
                ) from exc
            if stat.S_ISDIR(child_stat.st_mode) and not stat.S_ISLNK(child_stat.st_mode):
                scan_directory(child, child_relative)
            elif stat.S_ISREG(child_stat.st_mode):
                record_file(child, child_relative)
            elif stat.S_ISLNK(child_stat.st_mode):
                raise SkipBaselineError(f"package source contains symlink: {child_relative}")
            else:
                raise SkipBaselineError(
                    f"package source contains unsupported node: {child_relative}"
                )
        try:
            after = path.lstat()
            names_after = tuple(sorted(entry.name for entry in os.scandir(path)))
        except OSError as exc:
            raise SkipBaselineError(f"package directory drift: {relative}") from exc
        if _stat_signature(observed) != _stat_signature(after) or names_before != names_after:
            raise SkipBaselineError(f"package directory drift: {relative}")

    package_root = repo_root / "quant_investor"
    scan_directory(package_root, "quant_investor")
    if not (package_root / "__init__.py").is_file():
        raise SkipBaselineError("quant_investor package root is invalid")
    for relative in sorted(PACKAGE_EXTRA_PATHS):
        pure = PurePosixPath(relative)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise SkipBaselineError("package extra path registry is invalid")
        record_file(repo_root / Path(*pure.parts), relative)
    rows.sort(key=lambda item: item["path"])
    binding = {
        "row_count": len(rows),
        "sha256": _sha256(_canonical_bytes(rows)),
    }
    return binding, guards, rows


def _sample_package_source_superset(repo_root: Path) -> dict[str, Any]:
    first, first_guards, first_rows = _scan_package_source_superset(repo_root)
    second, second_guards, second_rows = _scan_package_source_superset(repo_root)
    if first != second or first_guards != second_guards or first_rows != second_rows:
        raise SkipBaselineError("package source superset changed during sampling")
    return first


def _validate_candidate_source_membership(
    *,
    repo_root: Path,
    policy: Mapping[str, Any],
    source_state: Mapping[str, Any],
    package_binding: Mapping[str, Any],
) -> None:
    module_policy = policy.get("module_policy")
    if (
        type(module_policy) is not dict
        or module_policy.get("candidate_content_binding") != "OUTER_SOURCE_STATE"
    ):
        raise SkipBaselineError("main-suite candidate content binding is invalid")
    candidates = module_policy.get("candidate_module_source_paths")
    if type(candidates) is not list or not candidates:
        raise SkipBaselineError("main-suite candidate source inventory is empty")
    normalized: list[str] = []
    casefolded: set[str] = set()
    for relative in candidates:
        if type(relative) is not str:
            raise SkipBaselineError("main-suite candidate source path is invalid")
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or not pure.parts
            or pure.suffix != ".py"
            or pure.as_posix() != relative
            or any(part in {"", ".", ".."} for part in pure.parts)
            or relative.casefold() in casefolded
        ):
            raise SkipBaselineError("main-suite candidate source path semantics are invalid")
        normalized.append(relative)
        casefolded.add(relative.casefold())
    if normalized != sorted(normalized, key=lambda item: item.encode("utf-8")):
        raise SkipBaselineError("main-suite candidate source inventory is not canonical")

    base_commit = source_state.get("base_commit")
    if type(base_commit) is not str or COMMIT_RE.fullmatch(base_commit) is None:
        raise SkipBaselineError("candidate source base commit is invalid")
    tracked_raw = _run_bytes(
        ("git", "ls-tree", "-r", "--name-only", "-z", base_commit, "--"),
        cwd=repo_root,
        environment=_git_environment(),
    )
    tracked = set(_decode_nul_paths(tracked_raw, label="base tracked source inventory"))
    untracked_rows = source_state.get("untracked")
    if type(untracked_rows) is not list:
        raise SkipBaselineError("candidate source untracked inventory is invalid")
    untracked_files = {
        row["path"]
        for row in untracked_rows
        if type(row) is dict and row.get("type") == "file" and type(row.get("path")) is str
    }
    current_package_binding, _guards, package_rows = _scan_package_source_superset(repo_root)
    if current_package_binding != dict(package_binding):
        raise SkipBaselineError("candidate package source seal mismatch")
    package_files = {
        row["path"]
        for row in package_rows
        if row.get("kind") == "file" and type(row.get("path")) is str
    }
    for relative in normalized:
        path = repo_root / Path(*PurePosixPath(relative).parts)
        try:
            observed = path.lstat()
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise SkipBaselineError(f"candidate source is unavailable: {relative}") from exc
        if (
            not stat.S_ISREG(observed.st_mode)
            or stat.S_ISLNK(observed.st_mode)
            or resolved != path
            or path.parent.resolve(strict=True) != resolved.parent
        ):
            raise SkipBaselineError(f"candidate source is not concrete: {relative}")
        admitted = relative in tracked or relative in untracked_files
        if relative.startswith("quant_investor/") and relative in package_files:
            admitted = True
        if not admitted:
            raise SkipBaselineError(f"candidate source is outside sealed source state: {relative}")


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
            raise SkipBaselineError(f"protected root is unavailable: {identifier}") from exc
        try:
            resolved = absolute.resolve(strict=True)
        except OSError as exc:
            raise SkipBaselineError(f"protected root cannot resolve: {identifier}") from exc
        if (
            stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISDIR(observed.st_mode)
            or resolved != absolute
        ):
            raise SkipBaselineError(f"protected root must be a concrete directory: {identifier}")
        rows.append(
            {
                "ctime_ns": observed.st_ctime_ns,
                "id": identifier,
                "mode": _mode_string(observed.st_mode),
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


def _canonical_parent_sys_path() -> list[str]:
    paths: list[str] = []
    for index, raw in enumerate(sys.path):
        if type(raw) is not str or not raw:
            raise SkipBaselineError(f"parent sys.path[{index}] is invalid")
        path = Path(raw)
        if not path.is_absolute() or Path(os.path.abspath(path)) != path:
            raise SkipBaselineError(f"parent sys.path[{index}] is not normalized absolute")
        paths.append(str(path))
    if len(paths) != len(set(paths)):
        raise SkipBaselineError("parent sys.path contains duplicate entries")
    return paths


def _pyvenv_cfg_binding(prefix: Path) -> dict[str, Any]:
    path = prefix / "pyvenv.cfg"
    try:
        path.lstat()
    except FileNotFoundError:
        return {"path": str(path), "state": "ABSENT"}
    except OSError as exc:
        raise SkipBaselineError("parent pyvenv.cfg state is unavailable") from exc
    raw, observed = _stable_regular_bytes(
        path,
        label="parent pyvenv.cfg",
        require_nlink_one=False,
    )
    return {
        "mode": _mode_string(observed.st_mode),
        "path": str(path),
        "sha256": _sha256(raw),
        "size_bytes": len(raw),
        "state": "PRESENT",
    }


def _parent_runtime_binding() -> dict[str, Any]:
    lexical = Path(sys.executable)
    if not lexical.is_absolute() or Path(os.path.abspath(lexical)) != lexical:
        raise SkipBaselineError("parent executable path is not normalized absolute")
    executable = _file_binding(lexical, label="parent frozen CPython")
    try:
        resolved = lexical.resolve(strict=True)
        prefix = Path(sys.prefix)
        base_prefix = Path(sys.base_prefix)
        resolved_prefix = prefix.resolve(strict=True)
        resolved_base_prefix = base_prefix.resolve(strict=True)
    except OSError as exc:
        raise SkipBaselineError("parent runtime path identity is unavailable") from exc
    if (
        lexical != Path(BASE_PYTHON_PATH)
        or resolved != Path(BASE_PYTHON_PATH).resolve(strict=True)
        or executable["sha256"] != BASE_PYTHON_SHA256
        or executable["size_bytes"] != BASE_PYTHON_SIZE
        or executable["mode"] != "0755"
        or executable["executable"] is not True
        or sys.implementation.name != "cpython"
        or tuple(sys.version_info[:3]) != tuple(BASE_PYTHON_VERSION_INFO)
        or not prefix.is_absolute()
        or not base_prefix.is_absolute()
        or Path(os.path.abspath(prefix)) != prefix
        or Path(os.path.abspath(base_prefix)) != base_prefix
    ):
        raise SkipBaselineError("parent runtime identity mismatch")
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
        raise SkipBaselineError("parent runtime flags do not prove -I -S -B")
    paths = _canonical_parent_sys_path()
    return {
        "executable": True,
        "flags": flags,
        "implementation": "cpython",
        "lexical_executable": str(lexical),
        "mode": executable["mode"],
        "pyvenv_cfg": _pyvenv_cfg_binding(prefix),
        "resolved_executable": str(resolved),
        "sha256": executable["sha256"],
        "size_bytes": executable["size_bytes"],
        "sys_base_prefix": str(base_prefix),
        "sys_base_prefix_realpath": str(resolved_base_prefix),
        "sys_path": paths,
        "sys_path_sha256": _sha256(_canonical_bytes(paths)),
        "sys_prefix": str(prefix),
        "sys_prefix_realpath": str(resolved_prefix),
        "version": BASE_PYTHON_VERSION,
        "version_info": list(BASE_PYTHON_VERSION_INFO),
    }


def _sample_pip_status() -> dict[str, Any]:
    loaded_before = sorted(name for name in sys.modules if name == "pip" or name.startswith("pip."))
    try:
        spec = importlib.util.find_spec("pip")
    except (ImportError, AttributeError, ValueError) as exc:
        raise SkipBaselineError("pip visibility cannot be observed safely") from exc
    loaded_after = sorted(name for name in sys.modules if name == "pip" or name.startswith("pip."))
    if loaded_after != loaded_before:
        raise SkipBaselineError("pip visibility probe imported pip")
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
        "child_environment_policy": dict(PIP_CHILD_ENVIRONMENT_POLICY),
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
        raise SkipBaselineError("pip is visible in the isolated parent runtime")
    return value


def _validate_runtime_invocation() -> None:
    _parent_runtime_binding()
    _sample_pip_status()
    sys.dont_write_bytecode = True


def _sample_toolchain() -> dict[str, Any]:
    base_path = Path(BASE_PYTHON_PATH)
    base = _file_binding(base_path, label="frozen base Python")
    if (
        base["realpath"] != BASE_PYTHON_PATH
        or base["sha256"] != BASE_PYTHON_SHA256
        or base["size_bytes"] != BASE_PYTHON_SIZE
        or base["mode"] != "0755"
        or base["executable"] is not True
    ):
        raise SkipBaselineError("frozen base Python binary identity mismatch")
    uv_path = Path(UV_PATH)
    uv = _file_binding(uv_path, label="frozen uv")
    if (
        uv["realpath"] != UV_PATH
        or uv["sha256"] != UV_SHA256
        or uv["size_bytes"] != UV_SIZE
        or uv["mode"] != "0755"
        or uv["executable"] is not True
    ):
        raise SkipBaselineError("frozen uv binary identity mismatch")
    cache = _resolve_existing_directory(
        Path(UV_CACHE_PATH),
        label="frozen uv cache",
        owner_private=False,
    )
    cache_stat = cache.lstat()
    return {
        "base_python": {
            "executable": True,
            "implementation": "cpython",
            "lexical_path": BASE_PYTHON_PATH,
            "mode": "0755",
            "realpath": BASE_PYTHON_PATH,
            "sha256": BASE_PYTHON_SHA256,
            "size_bytes": BASE_PYTHON_SIZE,
            "version": BASE_PYTHON_VERSION,
            "version_info": list(BASE_PYTHON_VERSION_INFO),
        },
        "pip_scope": dict(PIP_SCOPE),
        "uv": {
            "executable": True,
            "lexical_path": UV_PATH,
            "mode": "0755",
            "output": UV_VERSION_OUTPUT,
            "realpath": UV_PATH,
            "sha256": UV_SHA256,
            "size_bytes": UV_SIZE,
            "version": UV_VERSION,
        },
        "uv_cache": {
            "mode": _mode_string(cache_stat.st_mode),
            "path": UV_CACHE_PATH,
            "realpath": str(cache),
            "st_dev": cache_stat.st_dev,
            "st_ino": cache_stat.st_ino,
            "uid": cache_stat.st_uid,
        },
    }


def _producer_binding(repo_root: Path) -> dict[str, Any]:
    path = repo_root / PRODUCER_PATH
    raw, _observed = _stable_regular_bytes(path, label="skip baseline producer")
    return {
        "path": str(path),
        "sha256": _sha256(raw),
        "size_bytes": len(raw),
        "version": PRODUCER_VERSION,
    }


def _make_private_directory(path: Path, *, parent: Path, label: str) -> Path:
    if path.parent != parent or path.exists():
        raise SkipBaselineError(f"{label} must be a new direct child")
    try:
        path.mkdir(mode=0o700)
        observed = path.lstat()
    except OSError as exc:
        raise SkipBaselineError(f"{label} cannot be created") from exc
    if (
        not stat.S_ISDIR(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o700
        or observed.st_uid != os.getuid()
    ):
        raise SkipBaselineError(f"{label} is not owner-private 0700")
    return path


def _sync_environment(*, native_venv: Path, home: Path, tmp: Path) -> dict[str, str]:
    return {
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": SAFE_PATH,
        "PIP_CONFIG_FILE": "/dev/null",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INDEX": "1",
        "PIP_NO_INPUT": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(tmp),
        "UV_CACHE_DIR": UV_CACHE_PATH,
        "UV_NO_CONFIG": "1",
        "UV_OFFLINE": "1",
        "UV_PROJECT_ENVIRONMENT": str(native_venv),
        "UV_PYTHON_DOWNLOADS": "never",
    }


def _pytest_environment(*, home: Path, tmp: Path) -> dict[str, str]:
    return {
        "COLUMNS": "100000",
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": SAFE_PATH,
        "PIP_CONFIG_FILE": "/dev/null",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INDEX": "1",
        "PIP_NO_INPUT": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "TERM": "dumb",
        "TMPDIR": str(tmp),
        "UV_NO_CONFIG": "1",
        "UV_OFFLINE": "1",
        "UV_PYTHON_DOWNLOADS": "never",
    }


def _load_main_suite_module(repo_root: Path) -> tuple[Any, dict[str, Any]]:
    path = repo_root / MAIN_SUITE_HARNESS_PATH
    raw, _observed = _stable_regular_bytes(path, label="main-suite harness")
    spec = importlib.util.spec_from_file_location("_myquant_phase0_skip_main_suite", path)
    if spec is None or spec.loader is None:
        raise SkipBaselineError("main-suite harness cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(spec.name, None)
        raise SkipBaselineError("main-suite harness import failed") from exc
    finally:
        sys.dont_write_bytecode = previous
    if (
        getattr(module, "RECEIPT_PREFIX", None) != MAIN_SUITE_RECEIPT_PREFIX
        or getattr(module, "RECEIPT_VERSION", None) != MAIN_SUITE_RECEIPT_VERSION
        or getattr(module, "RECEIPT_SCHEMA_ID", None) != MAIN_SUITE_RECEIPT_SCHEMA_ID
        or getattr(module, "POLICY_VERSION", None) != MAIN_SUITE_RUNTIME_POLICY_VERSION
        or getattr(module, "POLICY_SCHEMA_ID", None) != MAIN_SUITE_RUNTIME_POLICY_SCHEMA_ID
        or not callable(getattr(module, "validate_policy_bytes", None))
        or not callable(getattr(module, "run_main_suite", None))
    ):
        raise SkipBaselineError("main-suite harness public contract mismatch")
    return (
        module,
        {
            "path": MAIN_SUITE_HARNESS_PATH,
            "sha256": _sha256(raw),
            "size_bytes": len(raw),
        },
    )


def _main_suite_policy(
    repo_root: Path,
    harness: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = repo_root / MAIN_SUITE_POLICY_PATH
    raw, _observed = _stable_regular_bytes(path, label="main-suite runtime policy")
    policy_document = _strict_canonical_json(
        raw,
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
        policy = harness.validate_policy_bytes(raw)
    except Exception as exc:
        raise SkipBaselineError("main-suite runtime policy validation failed") from exc
    if (
        type(policy) is not dict
        or raw != _canonical_resource_bytes(policy)
        or policy.get("version") != MAIN_SUITE_RUNTIME_POLICY_VERSION
        or policy.get("schema_id") != MAIN_SUITE_RUNTIME_POLICY_SCHEMA_ID
        or policy.get("semantic_sha256") != _semantic_sha256(policy)
        or policy.get("discovery_mode") is not False
        or policy.get("candidate_root") != str(repo_root)
        or policy.get("limitations") != NORMATIVE_LIMITATIONS
        or policy.get("pytest_args") != list(MAIN_SUITE_PYTEST_ARGS)
    ):
        raise SkipBaselineError("main-suite runtime policy identity mismatch")
    policy_bindings = _main_suite_policy_bindings(repo_root)
    if policy_bindings["policy_binding"]["sha256"] != _sha256(raw) or policy_bindings[
        "policy_binding"
    ]["size_bytes"] != len(raw):
        raise SkipBaselineError("main-suite runtime policy changed during validation")
    return (
        policy,
        policy_bindings,
    )


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
    manifest_raw, _observed = _stable_regular_bytes(
        manifest_path,
        label="main-suite package manifest",
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
        raise SkipBaselineError("main-suite package manifest binding mismatch")
    return {
        "policy_binding": policy_binding,
        "policy_manifest_binding": manifest_binding,
        "policy_schema_binding": schema_binding,
    }


def _main_suite_environment(
    work_root: Path,
    *,
    policy: Mapping[str, Any],
) -> dict[str, str]:
    environment_policy = _main_suite_environment_policy(policy)
    required = environment_policy["required"]
    allowed = environment_policy["allowed_keys"]
    dynamic = environment_policy["dynamic_path_keys"]
    topology = environment_policy["path_topology"]
    runtime = _make_private_directory(
        work_root / "main-suite-runtime",
        parent=work_root,
        label="main-suite runtime",
    )
    environment = dict(required)
    siblings: dict[str, Path] = {}
    for key in topology["closed_root_siblings"]:
        siblings[key] = _make_private_directory(
            runtime / f"path_{key.casefold()}",
            parent=runtime,
            label=f"main-suite {key}",
        )
        environment[key] = str(siblings[key])
    cache = siblings["XDG_CACHE_HOME"]
    for key in topology["cache_children"]:
        child = _make_private_directory(
            cache / f"path_{key.casefold()}",
            parent=cache,
            label=f"main-suite {key}",
        )
        environment[key] = str(child)
    if set(environment) != set(allowed):
        raise SkipBaselineError("main-suite environment closure mismatch")
    return dict(sorted(environment.items()))


def _main_suite_environment_policy(
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    environment_policy = policy.get("pytest_environment")
    if type(environment_policy) is not dict or set(environment_policy) != {
        "allowed_keys",
        "dynamic_path_keys",
        "forbidden",
        "path_topology",
        "required",
    }:
        raise SkipBaselineError("main-suite environment policy shape mismatch")
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
        raise SkipBaselineError("main-suite environment policy is invalid")
    for key in dynamic:
        if re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", key, re.ASCII) is None:
            raise SkipBaselineError("main-suite dynamic environment key is invalid")
    return dict(environment_policy)


def _validate_recorded_main_suite_environment(
    environment: Any,
    *,
    policy: Mapping[str, Any] | None,
    live: bool,
    external_before: Any,
) -> tuple[dict[str, str], dict[str, Any]]:
    if type(environment) is not dict or any(
        type(key) is not str or type(value) is not str for key, value in environment.items()
    ):
        raise SkipBaselineError("main-suite recorded environment is invalid")
    recorded = dict(sorted(environment.items()))
    if policy is None:
        required = {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
        allowed = {
            "BLACK_CACHE_DIR",
            "HOME",
            "MYPY_CACHE_DIR",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
            "PYTHONPYCACHEPREFIX",
            "TMPDIR",
            "XDG_CACHE_HOME",
        }
        dynamic = allowed - set(required)
    else:
        environment_policy = _main_suite_environment_policy(policy)
        required = environment_policy["required"]
        allowed = set(environment_policy["allowed_keys"])
        dynamic = set(environment_policy["dynamic_path_keys"])
    if set(recorded) != allowed or any(
        recorded.get(key) != value for key, value in required.items()
    ):
        raise SkipBaselineError("main-suite recorded environment closure mismatch")
    for key in dynamic:
        value = recorded.get(key)
        if (
            type(value) is not str
            or "\0" in value
            or not Path(value).is_absolute()
            or os.path.normpath(value) != value
        ):
            raise SkipBaselineError(f"main-suite recorded path is invalid: {key}")
    home = Path(recorded["HOME"])
    tmpdir = Path(recorded["TMPDIR"])
    cache = Path(recorded["XDG_CACHE_HOME"])
    if (
        home.parent != tmpdir.parent
        or home.parent != cache.parent
        or Path(recorded["BLACK_CACHE_DIR"]).parent != cache
        or Path(recorded["MYPY_CACHE_DIR"]).parent != cache
        or Path(recorded["PYTHONPYCACHEPREFIX"]).parent != cache
    ):
        raise SkipBaselineError("main-suite recorded path topology mismatch")
    pycache_prefix = Path(recorded["PYTHONPYCACHEPREFIX"])
    if live:
        for label, path in (
            ("HOME", home),
            ("TMPDIR", tmpdir),
            ("XDG_CACHE_HOME", cache),
            ("BLACK_CACHE_DIR", Path(recorded["BLACK_CACHE_DIR"])),
            ("MYPY_CACHE_DIR", Path(recorded["MYPY_CACHE_DIR"])),
        ):
            try:
                observed = path.lstat()
            except OSError as exc:
                raise SkipBaselineError(f"main-suite recorded {label} is unavailable") from exc
            if (
                not stat.S_ISDIR(observed.st_mode)
                or stat.S_ISLNK(observed.st_mode)
                or observed.st_uid != os.getuid()
                or stat.S_IMODE(observed.st_mode) != 0o700
            ):
                raise SkipBaselineError(f"main-suite recorded {label} is not owner-private")
        pycache_binding = _empty_private_directory_binding(
            pycache_prefix,
            label="main-suite recorded PYTHONPYCACHEPREFIX",
        )
    else:
        if type(external_before) is not dict:
            raise SkipBaselineError("main-suite external_before is unavailable")
        pycache_binding = external_before.get("pycache_prefix")
        if (
            type(pycache_binding) is not dict
            or pycache_binding.get("path") != str(pycache_prefix)
            or pycache_binding.get("mode") != "0700"
        ):
            raise SkipBaselineError("main-suite recorded pycache binding mismatch")
    return recorded, dict(pycache_binding)


def _recorded_main_suite_contract(
    receipt: Mapping[str, Any],
    *,
    repo_root: Path,
    policy: Mapping[str, Any] | None,
    live: bool,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    command = receipt.get("command")
    if type(command) is not dict or type(command.get("argv")) is not list:
        raise SkipBaselineError("main-suite recorded command is invalid")
    argv = command["argv"]
    if len(argv) != 10 + len(MAIN_SUITE_PYTEST_ARGS):
        raise SkipBaselineError("main-suite recorded argv length mismatch")
    recorded_policy = (
        dict(policy)
        if policy is not None
        else {
            "main_runtime": {"lexical_python": argv[0]},
            "wrapper_binding": {"path": argv[6]},
        }
    )
    environment, pycache_binding = _validate_recorded_main_suite_environment(
        command.get("environment"),
        policy=policy,
        live=live,
        external_before=receipt.get("external_before"),
    )
    return recorded_policy, environment, pycache_binding


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
        raise SkipBaselineError("main-suite challenge binding kind is invalid")
    _require_sha256(
        challenge_binding_sha256,
        label="main-suite challenge binding SHA-256",
    )
    if type(value) is not dict or set(value) != {
        "attestation",
        "raw",
        "receipt",
        "stderr",
        "stdout",
    }:
        raise SkipBaselineError("main-suite harness result shape mismatch")
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
        or len(stdout) + len(stderr) + len(attestation) > MAX_CAPTURE_BYTES
    ):
        raise SkipBaselineError("main-suite harness result types or limits mismatch")
    tail = (
        struct.pack(">Q", len(stdout))
        + stdout
        + struct.pack(">Q", len(stderr))
        + stderr
        + struct.pack(">Q", len(attestation))
        + attestation
    )
    expected_raw = MAIN_SUITE_RECEIPT_PREFIX + _canonical_bytes(receipt) + b"\n" + tail
    if raw != expected_raw or not raw.startswith(MAIN_SUITE_RECEIPT_PREFIX):
        raise SkipBaselineError("main-suite raw receipt framing mismatch")
    if (
        receipt.get("version") != MAIN_SUITE_RECEIPT_VERSION
        or receipt.get("schema_id") != MAIN_SUITE_RECEIPT_SCHEMA_ID
        or receipt.get("protocol_version") != PROTOCOL_VERSION
        or receipt.get("authority") is not False
        or receipt.get("limitations") != NORMATIVE_LIMITATIONS
        or receipt.get("framing") != MAIN_SUITE_FRAMING
        or receipt.get("semantic_sha256") != _semantic_sha256(receipt)
        or receipt.get("challenge_binding")
        != {
            "kind": challenge_binding_kind,
            "sha256": challenge_binding_sha256,
        }
    ):
        raise SkipBaselineError("main-suite receipt identity mismatch")
    expected_command = _main_suite_expected_command(
        repo_root=repo_root,
        policy=policy,
        policy_bindings=policy_bindings,
        environment=expected_environment,
    )
    if receipt.get("command") != expected_command:
        raise SkipBaselineError("main-suite command binding mismatch")
    claims = receipt.get("claims")
    if type(claims) is not dict:
        raise SkipBaselineError("main-suite receipt claims shape mismatch")
    if any(
        receipt.get(name) != policy_bindings.get(name)
        for name in (
            "policy_binding",
            "policy_manifest_binding",
            "policy_schema_binding",
        )
    ):
        raise SkipBaselineError("main-suite policy receipt binding mismatch")
    if (
        type(receipt.get("failures")) is not list
        or type(receipt.get("failure_codes")) is not list
        or type(receipt.get("finalization")) is not dict
    ):
        raise SkipBaselineError("main-suite rejection/finalization shape mismatch")
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
        "tail_sha256": _sha256(tail),
        "tail_size_bytes": len(tail),
    }
    if receipt.get("streams") != expected_streams:
        raise SkipBaselineError("main-suite stream binding mismatch")
    frames = receipt.get("attestations")
    if type(frames) is not list or len(frames) > 3:
        raise SkipBaselineError("main-suite attestation frame count mismatch")
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
            raise SkipBaselineError("main-suite attestation challenge binding mismatch")
        if ordinal in {1, 2}:
            runtime = payload.get("runtime")
            if (
                type(runtime) is not dict
                or runtime.get("bytecode_policy") != expected_bytecode_policy
            ):
                raise SkipBaselineError("main-suite bytecode policy binding mismatch")
        if ordinal == 1 and payload.get("environment") != dict(
            sorted(expected_environment.items())
        ):
            raise SkipBaselineError("main-suite environment attestation mismatch")
    for snapshot_name in ("external_before", "external_after"):
        snapshot = receipt.get(snapshot_name)
        if snapshot is not None and (
            type(snapshot) is not dict or snapshot.get("pycache_prefix") != expected_pycache_binding
        ):
            raise SkipBaselineError(f"main-suite {snapshot_name} pycache binding mismatch")
    if len(frames) == 3:
        terminal = frames[2]["payload"]
        if (
            terminal.get("frame") != "terminal_complete"
            or type(terminal.get("pytest_exit_code")) is not int
            or type(terminal.get("final_loaded_modules")) is not dict
        ):
            raise SkipBaselineError("main-suite terminal attestation mismatch")
    return dict(value)


def _parse_main_suite_result(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "attestation",
        "raw",
        "receipt",
        "stderr",
        "stdout",
    }:
        raise SkipBaselineError("main-suite harness result shape mismatch")
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
        or len(stdout) + len(stderr) + len(attestation) > MAX_CAPTURE_BYTES
    ):
        raise SkipBaselineError("main-suite harness result types or limits mismatch")
    tail = (
        struct.pack(">Q", len(stdout))
        + stdout
        + struct.pack(">Q", len(stderr))
        + stderr
        + struct.pack(">Q", len(attestation))
        + attestation
    )
    expected_raw = MAIN_SUITE_RECEIPT_PREFIX + _canonical_bytes(receipt) + b"\n" + tail
    if raw != expected_raw or not raw.startswith(MAIN_SUITE_RECEIPT_PREFIX):
        raise SkipBaselineError("main-suite raw receipt framing mismatch")
    return dict(value)


def _require_main_suite_accepted(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = value["receipt"]
    claims = receipt["claims"]
    frames = receipt["attestations"]
    if (
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
    ):
        return dict(value)
    raise SkipBaselineError("main-suite receipt was rejected")


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
    return _require_main_suite_accepted(validated)


def _run_main_suite_contract(
    *,
    repo_root: Path,
    work_root: Path,
    challenge_binding_kind: str,
    challenge_binding_sha256: str,
) -> dict[str, Any]:
    harness, harness_before = _load_main_suite_module(repo_root)
    policy, policy_bindings = _main_suite_policy(repo_root, harness)
    environment = _main_suite_environment(work_root, policy=policy)
    pycache_prefix = Path(environment["PYTHONPYCACHEPREFIX"])
    pycache_before = _empty_private_directory_binding(
        pycache_prefix,
        label="main-suite pycache prefix",
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
        raise SkipBaselineError("main-suite harness execution failed") from exc
    pycache_after = _empty_private_directory_binding(
        pycache_prefix,
        label="main-suite pycache prefix",
    )
    if pycache_after != pycache_before:
        raise SkipBaselineError("main-suite pycache prefix changed during execution")
    harness_raw, _observed = _stable_regular_bytes(
        repo_root / MAIN_SUITE_HARNESS_PATH,
        label="main-suite harness readback",
    )
    harness_after = {
        "path": MAIN_SUITE_HARNESS_PATH,
        "sha256": _sha256(harness_raw),
        "size_bytes": len(harness_raw),
    }
    if harness_after != harness_before:
        raise SkipBaselineError("main-suite harness changed during execution")
    if _main_suite_policy_bindings(repo_root) != policy_bindings:
        raise SkipBaselineError("main-suite policy binding files changed during execution")
    return _validate_main_suite_contract_result(
        result,
        repo_root=repo_root,
        policy=policy,
        policy_bindings=policy_bindings,
        expected_environment=environment,
        expected_pycache_binding=pycache_before,
        challenge_binding_kind=challenge_binding_kind,
        challenge_binding_sha256=challenge_binding_sha256,
    )


def _run_command(
    argv: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
) -> tuple[int, bytes, bytes]:
    try:
        completed = subprocess.run(
            list(argv),
            cwd=cwd,
            env=dict(environment),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise SkipBaselineError(f"command could not start: {argv[0]}") from exc
    if len(completed.stdout) + len(completed.stderr) > MAX_CAPTURE_BYTES:
        raise SkipBaselineError("command transcript exceeds the capture limit")
    return completed.returncode, completed.stdout, completed.stderr


def _stream_binding(raw: bytes) -> dict[str, Any]:
    return {
        "bytes_base64": base64.b64encode(raw).decode("ascii"),
        "encoding": "base64",
        "sha256": _sha256(raw),
        "size_bytes": len(raw),
    }


def _command_record(
    *,
    ordinal: int,
    argv: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    returncode: int,
    tool_version: str,
    stdout: bytes,
    stderr: bytes,
) -> dict[str, Any]:
    return {
        "argv": list(argv),
        "cwd": str(cwd),
        "environment": dict(environment),
        "exit_code": returncode if returncode >= 0 else None,
        "ordinal": ordinal,
        "signal": -returncode if returncode < 0 else None,
        "stderr": _stream_binding(stderr),
        "stdout": _stream_binding(stdout),
        "tool_version": tool_version,
    }


def _native_python_path(native_venv: Path) -> Path:
    return native_venv / "bin" / "python"


def _validate_native_python(native_python: Path) -> None:
    try:
        observed = native_python.lstat()
        resolved = native_python.resolve(strict=True)
    except OSError as exc:
        raise SkipBaselineError(
            "native sync did not create the expected Python entrypoint"
        ) from exc
    if not (stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode)) or resolved != Path(
        BASE_PYTHON_PATH
    ):
        raise SkipBaselineError("native Python entrypoint is not bound to frozen base Python")


def _installed_pytest_version(native_venv: Path) -> str:
    candidates = sorted(
        native_venv.glob("lib/python*/site-packages/pytest-*.dist-info/METADATA"),
        key=lambda path: str(path).encode("utf-8"),
    )
    if len(candidates) != 1:
        raise SkipBaselineError("native environment must contain exactly one pytest METADATA")
    raw, _observed = _stable_regular_bytes(
        candidates[0],
        label="native pytest METADATA",
    )
    message = BytesParser(policy=email_policy).parsebytes(raw)
    if message.get("Name", "").strip().casefold() != "pytest":
        raise SkipBaselineError("native pytest METADATA name mismatch")
    version = message.get("Version", "").strip()
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", version, re.ASCII) is None:
        raise SkipBaselineError("native pytest version is invalid")
    return version


def _framed_output(stdout: bytes, stderr: bytes) -> bytes:
    return struct.pack(">Q", len(stdout)) + stdout + struct.pack(">Q", len(stderr)) + stderr


def _parse_pytest_transcript(
    *,
    stdout: bytes,
    stderr: bytes,
    returncode: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decoded: list[str] = []
    for label, raw in (("pytest stdout", stdout), ("pytest stderr", stderr)):
        try:
            decoded.append(raw.decode("utf-8", errors="strict"))
        except UnicodeError as exc:
            raise SkipBaselineError(f"{label} is not strict UTF-8") from exc
    entries: list[dict[str, Any]] = []
    summary_candidates: list[dict[str, int]] = []
    for text in decoded:
        for line in text.splitlines():
            skip_match = SKIP_LINE_RE.fullmatch(line)
            if skip_match is not None:
                path = skip_match.group("path")
                pure = PurePosixPath(path)
                if (
                    pure.is_absolute()
                    or pure.as_posix() != path
                    or any(part in {"", ".", ".."} for part in pure.parts)
                ):
                    raise SkipBaselineError("pytest skip path is not safe repository-relative")
                entries.append(
                    {
                        "count": int(skip_match.group("count")),
                        "line": int(skip_match.group("line")),
                        "path": path,
                        "reason": skip_match.group("reason"),
                    }
                )
            tokens = {
                match.group("kind"): int(match.group("count"))
                for match in SUMMARY_TOKEN_RE.finditer(line)
            }
            if "passed" in tokens and " in " in line:
                summary_candidates.append(tokens)
    if len(summary_candidates) != 1:
        raise SkipBaselineError("pytest transcript must contain exactly one terminal summary")
    keys = [(entry["path"], entry["line"], entry["reason"], entry["count"]) for entry in entries]
    if len(keys) != len(set(keys)):
        raise SkipBaselineError("pytest skip rows contain duplicates")
    canonical = sorted(
        entries,
        key=lambda entry: (
            entry["path"].encode("utf-8"),
            entry["line"],
            entry["reason"].encode("utf-8"),
            entry["count"],
        ),
    )
    if entries != canonical:
        entries = canonical
    summary = summary_candidates[0]
    claims: dict[str, Any] = {
        "errors": summary.get("error", 0) + summary.get("errors", 0),
        "exit_code": returncode,
        "failed": summary.get("failed", 0),
        "passed": summary.get("passed", 0),
        "raw_output_sha256": _sha256(_framed_output(stdout, stderr)),
        "skip_allowlist_sha256": _sha256(_canonical_bytes(entries)),
        "skipped": summary.get("skipped", 0),
        "xfail": summary.get("xfailed", 0),
        "xpass": summary.get("xpassed", 0),
    }
    observed = sum(entry["count"] for entry in entries)
    if (
        returncode != 0
        or claims["passed"] <= 0
        or claims["skipped"] != EXPECTED_SKIP_COUNT
        or observed != EXPECTED_SKIP_COUNT
        or any(claims[key] != 0 for key in ("failed", "errors", "xfail", "xpass"))
    ):
        raise SkipBaselineError("pytest transcript does not satisfy the frozen skip baseline")
    return entries, claims


def _decode_stream(value: Any, *, label: str) -> bytes:
    stream = _require_exact_keys(
        value,
        {"bytes_base64", "encoding", "sha256", "size_bytes"},
        label=label,
    )
    if stream["encoding"] != "base64":
        raise SkipBaselineError(f"{label}.encoding must be base64")
    try:
        raw = base64.b64decode(stream["bytes_base64"], validate=True)
    except (ValueError, TypeError) as exc:
        raise SkipBaselineError(f"{label}.bytes_base64 is invalid") from exc
    if stream["size_bytes"] != len(raw) or stream["sha256"] != _sha256(raw):
        raise SkipBaselineError(f"{label} binding mismatch")
    return raw


def _validate_parent_runtime_binding(value: Any) -> dict[str, Any]:
    binding = _require_exact_keys(
        value,
        {
            "executable",
            "flags",
            "implementation",
            "lexical_executable",
            "mode",
            "pyvenv_cfg",
            "resolved_executable",
            "sha256",
            "size_bytes",
            "sys_base_prefix",
            "sys_base_prefix_realpath",
            "sys_path",
            "sys_path_sha256",
            "sys_prefix",
            "sys_prefix_realpath",
            "version",
            "version_info",
        },
        label="parent_runtime_binding",
    )
    expected_flags = {
        "dont_write_bytecode": 1,
        "ignore_environment": 1,
        "isolated": 1,
        "no_site": 1,
        "no_user_site": 1,
        "safe_path": True,
    }
    if (
        binding["executable"] is not True
        or binding["flags"] != expected_flags
        or binding["implementation"] != "cpython"
        or binding["lexical_executable"] != BASE_PYTHON_PATH
        or binding["resolved_executable"] != BASE_PYTHON_PATH
        or binding["mode"] != "0755"
        or binding["sha256"] != BASE_PYTHON_SHA256
        or binding["size_bytes"] != BASE_PYTHON_SIZE
        or binding["version"] != BASE_PYTHON_VERSION
        or binding["version_info"] != BASE_PYTHON_VERSION_INFO
    ):
        raise SkipBaselineError("parent runtime fixed identity mismatch")
    for key in (
        "sys_base_prefix",
        "sys_base_prefix_realpath",
        "sys_prefix",
        "sys_prefix_realpath",
    ):
        raw = binding[key]
        if type(raw) is not str or not Path(raw).is_absolute():
            raise SkipBaselineError(f"parent runtime {key} is invalid")
    paths = binding["sys_path"]
    if (
        type(paths) is not list
        or not paths
        or len(paths) != len(set(paths))
        or any(type(path) is not str or not Path(path).is_absolute() for path in paths)
        or binding["sys_path_sha256"] != _sha256(_canonical_bytes(paths))
    ):
        raise SkipBaselineError("parent runtime sys.path binding mismatch")
    pyvenv = binding["pyvenv_cfg"]
    if type(pyvenv) is not dict or pyvenv.get("state") not in {"ABSENT", "PRESENT"}:
        raise SkipBaselineError("parent runtime pyvenv.cfg binding is invalid")
    if pyvenv["state"] == "ABSENT":
        if set(pyvenv) != {"path", "state"}:
            raise SkipBaselineError("absent parent pyvenv.cfg binding is invalid")
    else:
        if set(pyvenv) != {"mode", "path", "sha256", "size_bytes", "state"}:
            raise SkipBaselineError("present parent pyvenv.cfg binding is invalid")
        _require_sha256(pyvenv["sha256"], label="parent pyvenv.cfg sha256")
        if type(pyvenv["size_bytes"]) is not int or pyvenv["size_bytes"] < 0:
            raise SkipBaselineError("parent pyvenv.cfg size is invalid")
    if type(pyvenv.get("path")) is not str or not Path(pyvenv["path"]).is_absolute():
        raise SkipBaselineError("parent pyvenv.cfg path is invalid")
    return binding


def _validate_pip_status(value: Any, *, label: str) -> dict[str, Any]:
    status = _require_exact_keys(
        value,
        {
            "child_environment_policy",
            "loaded_modules",
            "observation_scope",
            "pip_spec",
            "site_sys_path_entries",
        },
        label=label,
    )
    if (
        status["child_environment_policy"] != PIP_CHILD_ENVIRONMENT_POLICY
        or status["loaded_modules"] != []
        or status["observation_scope"] != PIP_OBSERVATION_SCOPE
        or status["pip_spec"] != {"origin": None, "search_locations": [], "visible": False}
        or status["site_sys_path_entries"] != []
    ):
        raise SkipBaselineError(f"{label} does not prove closed parent pip visibility")
    return status


def validate_skip_baseline(
    value: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    root = _require_exact_keys(value, ROOT_KEYS, label="skip baseline")
    if (
        root["version"] != SKIP_BASELINE_VERSION
        or root["protocol_version"] != PROTOCOL_VERSION
        or root["status"] != STATUS
        or root["accepted"] is not True
        or root["authority"] is not False
        or root["limitations"] != NORMATIVE_LIMITATIONS
        or root["expected_skip_count"] != EXPECTED_SKIP_COUNT
        or root["observed_skip_count"] != EXPECTED_SKIP_COUNT
    ):
        raise SkipBaselineError("skip baseline root identity/acceptance mismatch")
    _require_sha256(root["semantic_sha256"], label="skip baseline semantic_sha256")
    if root["semantic_sha256"] != _semantic_sha256(root):
        raise SkipBaselineError("skip baseline semantic SHA-256 mismatch")
    producer = _require_exact_keys(
        root["producer"],
        {"path", "sha256", "size_bytes", "version"},
        label="skip baseline producer",
    )
    if (
        type(producer["path"]) is not str
        or not Path(producer["path"]).is_absolute()
        or producer["version"] != PRODUCER_VERSION
    ):
        raise SkipBaselineError("skip baseline producer identity mismatch")
    _require_sha256(producer["sha256"], label="skip baseline producer.sha256")
    if type(producer["size_bytes"]) is not int or producer["size_bytes"] <= 0:
        raise SkipBaselineError("skip baseline producer.size_bytes is invalid")
    source_state = _require_exact_keys(
        root["source_state"],
        {
            "base_commit",
            "binary_diff_from_base",
            "dirty_paths",
            "porcelain_v1_z",
            "source_state_sha256",
            "untracked",
        },
        label="skip baseline source_state",
    )
    declared_source_sha = source_state["source_state_sha256"]
    _require_sha256(declared_source_sha, label="source_state.source_state_sha256")
    source_unsealed = dict(source_state)
    source_unsealed.pop("source_state_sha256")
    if declared_source_sha != _sha256(_canonical_bytes(source_unsealed)):
        raise SkipBaselineError("source_state semantic binding mismatch")
    binding = _validate_source_binding(root["source_binding"])
    if binding != _source_binding_from_state(source_state):
        raise SkipBaselineError("skip baseline source_binding mismatch")
    challenge_binding = _require_exact_keys(
        root["challenge_binding"],
        {"kind", "sha256"},
        label="challenge_binding",
    )
    if challenge_binding != {
        "kind": "SKIP_SOURCE_STATE",
        "sha256": declared_source_sha,
    }:
        raise SkipBaselineError("skip baseline challenge binding mismatch")
    package_binding = _require_exact_keys(
        root["package_source_superset"],
        {"row_count", "sha256"},
        label="skip baseline package_source_superset",
    )
    if type(package_binding["row_count"]) is not int or package_binding["row_count"] <= 0:
        raise SkipBaselineError("package_source_superset.row_count is invalid")
    _require_sha256(package_binding["sha256"], label="package_source_superset.sha256")
    parent_runtime = _validate_parent_runtime_binding(root["parent_runtime_binding"])
    pip_before = _validate_pip_status(root["pip_status_before"], label="pip_status_before")
    pip_after = _validate_pip_status(root["pip_status_after"], label="pip_status_after")
    if pip_before != pip_after:
        raise SkipBaselineError("pip status changed during baseline run")
    if root["protected_roots_before"] != root["protected_roots_after"]:
        raise SkipBaselineError("protected roots changed during baseline run")
    if root["toolchain_binding"] != _sample_toolchain():
        raise SkipBaselineError("skip baseline toolchain binding mismatch")
    commands = root["commands"]
    if type(commands) is not list or len(commands) != 2:
        raise SkipBaselineError("skip baseline must bind exactly two commands")
    normalized_commands = [
        _require_exact_keys(command, COMMAND_KEYS, label=f"commands[{index}]")
        for index, command in enumerate(commands)
    ]
    if [command["ordinal"] for command in normalized_commands] != [1, 2]:
        raise SkipBaselineError("skip baseline command ordinals mismatch")
    sync, pytest_command = normalized_commands
    if sync["argv"] != [
        UV_PATH,
        "sync",
        "--python",
        BASE_PYTHON_PATH,
        "--locked",
        "--all-extras",
        "--offline",
    ]:
        raise SkipBaselineError("skip baseline uv sync argv mismatch")
    for index, command in enumerate(normalized_commands):
        if (
            command["exit_code"] != 0
            or command["signal"] is not None
            or type(command["cwd"]) is not str
            or not Path(command["cwd"]).is_absolute()
            or type(command["environment"]) is not dict
            or type(command["tool_version"]) is not str
            or not command["tool_version"]
        ):
            raise SkipBaselineError(f"commands[{index}] is not an accepted command")
    if sync["environment"].get("PYTHONDONTWRITEBYTECODE") != "1":
        raise SkipBaselineError("skip baseline uv sync bytecode policy mismatch")
    sync_stdout = _decode_stream(sync["stdout"], label="commands[0].stdout")
    sync_stderr = _decode_stream(sync["stderr"], label="commands[0].stderr")
    del sync_stdout, sync_stderr
    pytest_stdout = _decode_stream(pytest_command["stdout"], label="commands[1].stdout")
    pytest_stderr = _decode_stream(pytest_command["stderr"], label="commands[1].stderr")
    main_suite_receipt = root["main_suite_receipt"]
    if type(main_suite_receipt) is not dict:
        raise SkipBaselineError("main_suite_receipt must be an object")
    receipt_command = main_suite_receipt.get("command")
    receipt_claims = main_suite_receipt.get("claims")
    if (
        type(receipt_command) is not dict
        or type(receipt_claims) is not dict
        or pytest_command["argv"] != receipt_command.get("argv")
        or pytest_command["cwd"] != receipt_command.get("cwd")
        or pytest_command["environment"] != receipt_command.get("environment")
        or pytest_command["exit_code"] != receipt_claims.get("exit_code")
        or pytest_command["signal"] != receipt_claims.get("signal")
        or pytest_command["tool_version"] != MAIN_SUITE_RECEIPT_VERSION
    ):
        raise SkipBaselineError("skip baseline main-suite command projection mismatch")
    main_attestation = _decode_stream(
        root["main_suite_attestation"],
        label="main_suite_attestation",
    )
    main_tail = (
        struct.pack(">Q", len(pytest_stdout))
        + pytest_stdout
        + struct.pack(">Q", len(pytest_stderr))
        + pytest_stderr
        + struct.pack(">Q", len(main_attestation))
        + main_attestation
    )
    main_raw = MAIN_SUITE_RECEIPT_PREFIX + _canonical_bytes(main_suite_receipt) + b"\n" + main_tail
    raw_binding = _require_exact_keys(
        root["main_suite_raw_binding"],
        {"sha256", "size_bytes"},
        label="main_suite_raw_binding",
    )
    if raw_binding != {
        "sha256": _sha256(main_raw),
        "size_bytes": len(main_raw),
    }:
        raise SkipBaselineError("main-suite raw binding mismatch")
    main_suite_value = {
        "attestation": main_attestation,
        "raw": main_raw,
        "receipt": main_suite_receipt,
        "stderr": pytest_stderr,
        "stdout": pytest_stdout,
    }
    receipt_policy_bindings = {
        name: main_suite_receipt.get(name, {})
        for name in (
            "policy_binding",
            "policy_manifest_binding",
            "policy_schema_binding",
        )
    }
    if repo_root is None:
        recorded_policy, recorded_environment, recorded_pycache_binding = (
            _recorded_main_suite_contract(
                main_suite_receipt,
                repo_root=Path(pytest_command["cwd"]),
                policy=None,
                live=False,
            )
        )
        _require_main_suite_accepted(
            _validate_main_suite_result(
                _parse_main_suite_result(main_suite_value),
                repo_root=Path(pytest_command["cwd"]),
                policy=recorded_policy,
                policy_bindings=receipt_policy_bindings,
                expected_environment=recorded_environment,
                expected_pycache_binding=recorded_pycache_binding,
                challenge_binding_kind="SKIP_SOURCE_STATE",
                challenge_binding_sha256=declared_source_sha,
            )
        )
    else:
        contract_repo = _resolve_repo_root(Path(pytest_command["cwd"]))
        harness, _harness_binding = _load_main_suite_module(contract_repo)
        current_policy, current_policy_bindings = _main_suite_policy(
            contract_repo,
            harness,
        )
        _validate_candidate_source_membership(
            repo_root=contract_repo,
            policy=current_policy,
            source_state=source_state,
            package_binding=package_binding,
        )
        recorded_policy, recorded_environment, recorded_pycache_binding = (
            _recorded_main_suite_contract(
                main_suite_receipt,
                repo_root=contract_repo,
                policy=current_policy,
                live=True,
            )
        )
        _validate_main_suite_contract_result(
            main_suite_value,
            repo_root=contract_repo,
            policy=recorded_policy,
            policy_bindings=current_policy_bindings,
            expected_environment=recorded_environment,
            expected_pycache_binding=recorded_pycache_binding,
            challenge_binding_kind="SKIP_SOURCE_STATE",
            challenge_binding_sha256=declared_source_sha,
        )
    parsed_entries, parsed_claims = _parse_pytest_transcript(
        stdout=pytest_stdout,
        stderr=pytest_stderr,
        returncode=pytest_command["exit_code"],
    )
    entries = root["entries"]
    if type(entries) is not list or entries != parsed_entries:
        raise SkipBaselineError("skip baseline entries do not equal parsed pytest rows")
    claims = _require_exact_keys(root["claims"], CLAIMS_KEYS, label="skip baseline claims")
    if claims != parsed_claims:
        raise SkipBaselineError("skip baseline claims do not equal parsed pytest claims")
    if repo_root is not None:
        repo = _resolve_repo_root(repo_root)
        if producer != _producer_binding(repo):
            raise SkipBaselineError("skip baseline producer does not match repository bytes")
        current = _git_snapshot(repo)
        if _public_source_state(current) != source_state:
            raise SkipBaselineError("skip baseline source state is no longer current")
        if _sample_package_source_superset(repo) != package_binding:
            raise SkipBaselineError("skip baseline package source binding is no longer current")
        if _sample_protected_roots(repo) != root["protected_roots_after"]:
            raise SkipBaselineError("skip baseline protected roots are no longer current")
        if _parent_runtime_binding() != parent_runtime:
            raise SkipBaselineError("skip baseline parent runtime binding is no longer current")
        if _sample_pip_status() != pip_after:
            raise SkipBaselineError("skip baseline pip status is no longer current")
    return dict(root)


def _validate_output_target(output_json: Path, *, bundle_root: Path) -> Path:
    if not output_json.is_absolute() or Path(os.path.abspath(output_json)) != output_json:
        raise SkipBaselineError("output JSON must be a normalized absolute path")
    if output_json.parent != bundle_root or not output_json.name:
        raise SkipBaselineError("output JSON must be a direct child of bundle root")
    try:
        output_json.lstat()
    except FileNotFoundError:
        return output_json
    except OSError as exc:
        raise SkipBaselineError("output JSON cannot be preflighted") from exc
    raise SkipBaselineError("output JSON already exists; overwrite/repair is forbidden")


def write_skip_baseline_exact_once(
    output_json: Path,
    value: Mapping[str, Any],
    *,
    bundle_root: Path,
) -> None:
    target = _validate_output_target(output_json, bundle_root=bundle_root)
    raw = _canonical_resource_bytes(value)
    parent_fd = -1
    descriptor = -1
    try:
        parent_fd = os.open(
            bundle_root,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        parent_stat = os.fstat(parent_fd)
        if (
            not stat.S_ISDIR(parent_stat.st_mode)
            or stat.S_IMODE(parent_stat.st_mode) != 0o700
            or parent_stat.st_uid != os.getuid()
        ):
            raise SkipBaselineError("bundle root changed before publication")
        descriptor = os.open(
            target.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        created = os.fstat(descriptor)
        if (
            not stat.S_ISREG(created.st_mode)
            or stat.S_IMODE(created.st_mode) != 0o600
            or created.st_uid != os.getuid()
            or created.st_nlink != 1
        ):
            raise SkipBaselineError("created baseline output is unsafe")
        view = memoryview(raw)
        written = 0
        while written < len(raw):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise SkipBaselineError("short write while publishing skip baseline")
            written += count
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.fsync(parent_fd)
    except FileExistsError as exc:
        raise SkipBaselineError("output JSON already exists; overwrite is forbidden") from exc
    except OSError as exc:
        raise SkipBaselineError("exact-once skip baseline publication failed") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_fd >= 0:
            os.close(parent_fd)
    observed_raw, observed = _stable_regular_bytes(target, label="published skip baseline")
    if (
        observed_raw != raw
        or stat.S_IMODE(observed.st_mode) != 0o600
        or observed.st_uid != os.getuid()
    ):
        raise SkipBaselineError("published skip baseline readback mismatch")


def build_skip_baseline(
    *,
    repo_root: Path,
    bundle_root: Path,
    work_root: Path,
    output_json: Path,
    command_runner: CommandRunner = _run_command,
    main_suite_runner: MainSuiteRunner = _run_main_suite_contract,
) -> dict[str, Any]:
    _validate_runtime_invocation()
    parent_runtime_before = _parent_runtime_binding()
    pip_status_before = _sample_pip_status()
    repo = _resolve_repo_root(repo_root)
    bundle, work = _validate_isolated_roots(
        repo_root=repo,
        bundle_root=bundle_root,
        work_root=work_root,
    )
    _validate_output_target(output_json, bundle_root=bundle)
    source_before = _git_snapshot(repo)
    source_state = _public_source_state(source_before)
    source_binding = _source_binding_from_state(source_state)
    package_before = _sample_package_source_superset(repo)
    protected_before = _sample_protected_roots(repo)
    toolchain_before = _sample_toolchain()
    producer = _producer_binding(repo)

    home = _make_private_directory(work / "home", parent=work, label="work HOME")
    tmp = _make_private_directory(work / "tmp", parent=work, label="work TMPDIR")
    native_venv = work / NATIVE_VENV_NAME
    sync_argv = [
        UV_PATH,
        "sync",
        "--python",
        BASE_PYTHON_PATH,
        "--locked",
        "--all-extras",
        "--offline",
    ]
    sync_env = _sync_environment(native_venv=native_venv, home=home, tmp=tmp)
    sync_code, sync_stdout, sync_stderr = command_runner(sync_argv, repo, sync_env)
    sync_command = _command_record(
        ordinal=1,
        argv=sync_argv,
        cwd=repo,
        environment=sync_env,
        returncode=sync_code,
        tool_version=UV_VERSION_OUTPUT,
        stdout=sync_stdout,
        stderr=sync_stderr,
    )
    if sync_code != 0:
        raise SkipBaselineError("native locked offline uv sync failed")
    _validate_native_python(_native_python_path(native_venv))
    _installed_pytest_version(native_venv)
    harness, _harness_binding = _load_main_suite_module(repo)
    policy, policy_bindings = _main_suite_policy(repo, harness)
    _validate_candidate_source_membership(
        repo_root=repo,
        policy=policy,
        source_state=source_state,
        package_binding=package_before,
    )
    challenge_binding = {
        "kind": "SKIP_SOURCE_STATE",
        "sha256": source_state["source_state_sha256"],
    }
    try:
        main_suite = main_suite_runner(
            repo_root=repo,
            work_root=work,
            challenge_binding_kind=challenge_binding["kind"],
            challenge_binding_sha256=challenge_binding["sha256"],
        )
    except SkipBaselineError:
        raise
    except Exception as exc:
        raise SkipBaselineError("main-suite collection failed") from exc
    recorded_policy, recorded_environment, recorded_pycache_binding = _recorded_main_suite_contract(
        main_suite["receipt"],
        repo_root=repo,
        policy=policy,
        live=True,
    )
    main_suite = _validate_main_suite_contract_result(
        main_suite,
        repo_root=repo,
        policy=recorded_policy,
        policy_bindings=policy_bindings,
        expected_environment=recorded_environment,
        expected_pycache_binding=recorded_pycache_binding,
        challenge_binding_kind=challenge_binding["kind"],
        challenge_binding_sha256=challenge_binding["sha256"],
    )
    main_receipt = main_suite["receipt"]
    main_command = main_receipt["command"]
    main_claims = main_receipt["claims"]
    pytest_command = _command_record(
        ordinal=2,
        argv=main_command["argv"],
        cwd=repo,
        environment=main_command["environment"],
        returncode=main_claims["exit_code"],
        tool_version=MAIN_SUITE_RECEIPT_VERSION,
        stdout=main_suite["stdout"],
        stderr=main_suite["stderr"],
    )
    entries, claims = _parse_pytest_transcript(
        stdout=main_suite["stdout"],
        stderr=main_suite["stderr"],
        returncode=main_claims["exit_code"],
    )

    source_after = _git_snapshot(repo)
    _assert_snapshot_equal(source_before, source_after)
    package_after = _sample_package_source_superset(repo)
    if package_after != package_before:
        raise SkipBaselineError("package source superset changed during baseline run")
    toolchain_after = _sample_toolchain()
    if toolchain_after != toolchain_before:
        raise SkipBaselineError("toolchain identity changed during baseline run")
    protected_after = _sample_protected_roots(repo)
    if protected_after != protected_before:
        raise SkipBaselineError("protected roots changed during baseline run")
    parent_runtime_after = _parent_runtime_binding()
    pip_status_after = _sample_pip_status()
    if parent_runtime_after != parent_runtime_before:
        raise SkipBaselineError("parent runtime binding changed during baseline run")
    if pip_status_after != pip_status_before:
        raise SkipBaselineError("pip status changed during baseline run")
    _require_fresh_root(bundle, label="bundle root before publication")

    value = _seal(
        {
            "accepted": True,
            "authority": False,
            "challenge_binding": challenge_binding,
            "claims": claims,
            "commands": [sync_command, pytest_command],
            "entries": entries,
            "expected_skip_count": EXPECTED_SKIP_COUNT,
            "limitations": list(NORMATIVE_LIMITATIONS),
            "main_suite_attestation": _stream_binding(main_suite["attestation"]),
            "main_suite_raw_binding": {
                "sha256": _sha256(main_suite["raw"]),
                "size_bytes": len(main_suite["raw"]),
            },
            "main_suite_receipt": main_receipt,
            "observed_skip_count": sum(entry["count"] for entry in entries),
            "package_source_superset": package_before,
            "parent_runtime_binding": parent_runtime_before,
            "pip_status_after": pip_status_after,
            "pip_status_before": pip_status_before,
            "producer": producer,
            "protected_roots_after": protected_after,
            "protected_roots_before": protected_before,
            "protocol_version": PROTOCOL_VERSION,
            "source_binding": source_binding,
            "source_state": source_state,
            "status": STATUS,
            "toolchain_binding": toolchain_before,
            "version": SKIP_BASELINE_VERSION,
        }
    )
    validate_skip_baseline(value, repo_root=repo)
    write_skip_baseline_exact_once(output_json, value, bundle_root=bundle)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        value = build_skip_baseline(
            repo_root=args.repo_root,
            bundle_root=args.bundle_root,
            work_root=args.work_root,
            output_json=args.output_json,
        )
    except SkipBaselineError as exc:
        print(f"v17 Phase 0 skip baseline failed: {exc}", file=sys.stderr)
        return exc.exit_code
    print(
        _canonical_bytes(
            {
                "accepted": value["accepted"],
                "semantic_sha256": value["semantic_sha256"],
                "status": value["status"],
            }
        ).decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
