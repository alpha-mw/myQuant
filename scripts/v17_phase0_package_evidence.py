#!/usr/bin/env python3
"""Build owner-private offline package-parity evidence for v17 Phase 0."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence

PROTOCOL_VERSION = "myquant.v17.v2"
PACKAGE_EVIDENCE_VERSION = "myquant.v17.v2.phase0-package-parity-evidence.v2"
PACKAGE_EVIDENCE_SCHEMA_ID = "myquant.v17.v2.phase0-package-parity-evidence.schema.v2"
PACKAGE_EVIDENCE_SCHEMA_PATH = "scripts/schemas/v17_phase0_package_evidence.v2.schema.json"
PACKAGE_PRODUCER_VERSION = "myquant.v17.v2.phase0-package-evidence-producer.v2"
SESSION_VERSION = "myquant.v17.v2.phase0-session.v2"
SESSION_SCHEMA_ID = "myquant.v17.v2.phase0-session.schema.v2"
SESSION_SCHEMA_PATH = "scripts/schemas/v17_phase0_session.v2.schema.json"
SEMANTIC_FIELD = "semantic_sha256"
EXPECTED_GATE_ROLES = (
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
COMMAND_ROLES = (
    "base_python_probe",
    "uv_version",
    "create_build_venv",
    "install_build_backend",
    "build_backend_probe",
    "hatch_selector_before",
    "build_sdist",
    "build_wheel_from_sdist",
    "create_install_venv",
    "ensurepip_bundle_before",
    "ensurepip",
    "ensurepip_bundle_after",
    "pip_version",
    "install_inventory_before_project",
    "install_wheel_no_compile",
    "installed_paths_probe",
    "hatch_selector_parity",
    "package_parity",
    "hatch_selector_after",
)
STEP: dict[str, Any] = {
    "filename": "40_package_parity.json",
    "kind": "artifact",
    "ordinal": 9,
    "role": "package_parity",
}
EXPECTED_UV_VERSION = "0.10.9"
EXPECTED_UV_VERSION_OUTPUT = "uv 0.10.9 (f675560f3 2026-03-06)"
EXPECTED_PIP_VERSION = "25.2"
EXPECTED_PYTHON_VERSION = "3.13.7"
EXPECTED_PYTHON_VERSION_INFO = [3, 13, 7]
EXPECTED_PIP_WHEEL_NAME = "pip-25.2-py3-none-any.whl"
EXPECTED_PIP_WHEEL_SIZE = 1_752_557
EXPECTED_PIP_WHEEL_SHA256 = "690972885fc9270380d1bb28212cafdff6a96e0b6e04396b9fa7505253591e11"
EXPECTED_PIP_WRAPPERS = ["pip3", "pip3.13"]
BASE_RUNTIME_PIP_OBSERVATION_SCOPE = "NON_IMPORTING_ISOLATED_NO_SITE_RUNTIME_VISIBILITY_ONLY"
EXPECTED_BASE_RUNTIME_FLAGS = {
    "dont_write_bytecode": 1,
    "ignore_environment": 1,
    "isolated": 1,
    "no_site": 1,
    "no_user_site": 1,
    "safe_path": True,
}
EXPECTED_BUILD_BACKEND_PACKAGES = {
    "hatchling": "1.31.0",
    "packaging": "26.2",
    "pathspec": "1.1.1",
    "pluggy": "1.6.0",
    "trove-classifiers": "2026.6.1.19",
}
EXPECTED_BUILD_BACKEND_INVENTORY = sorted(
    (
        {"name": name, "version": version}
        for name, version in EXPECTED_BUILD_BACKEND_PACKAGES.items()
    ),
    key=lambda item: item["name"].casefold(),
)
BUILD_BACKEND_REQUIREMENTS = tuple(
    f"{name}=={version}" for name, version in EXPECTED_BUILD_BACKEND_PACKAGES.items()
)
HATCH_EXTRA_PATHS = ("README.md", "pyproject.toml", "requirements.txt")
LIMITATIONS = [
    "PRECOLLECTION_ALLOWED_TREES_ATTESTED_NOT_COMPLETE_MAIN_RUNTIME_FILESET",
    "NO_DIRECT_LAUNCH_REFERENCE_ONLY_NOT_PROCESS_IO_ATTESTED",
    "OFFLINE_FLAGS_ONLY_NOT_OS_EGRESS_ATTESTED",
    "OWNER_DECLARED_PATHS_ONLY_NOT_CONTENT_PROVENANCE",
    "PIP_25_2_PACKAGE_INSTALL_ENV_ONLY_NATIVE_AND_BUILD_ENV_PIP_ABSENT",
]
AUTHORITY_REPO_ROOT = Path("/Users/maxwell/mySpace/myQuant")
PROTECTED_ROOT_IDS = (
    "authority_v16",
    "authority_v16_operator_advisory",
    "candidate_v16",
    "candidate_v16_operator_advisory",
)
SESSION_KEYS = {
    "authority",
    "base_commit",
    "classification_binding",
    "gate_plan",
    "limitations",
    "package_source_superset",
    "parent_runtime_binding",
    "pip_status_after",
    "pip_status_before",
    "producer",
    "protected_roots",
    "protocol_version",
    "repo_root",
    "schemas",
    "semantic_sha256",
    "session_id",
    "skip_baseline_binding",
    "source_binding",
    "source_state",
    "status",
    "toolchain_binding",
    "uv_cache_binding",
    "version",
}
SAFE_EXECUTION_ENVIRONMENT = {
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
    "PYTHONDONTWRITEBYTECODE": "1",
}
FIXED_ENVIRONMENT_OVERRIDES = {
    "PIP_CONFIG_FILE": "/dev/null",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PIP_NO_CACHE_DIR": "1",
    "PIP_NO_COMPILE": "1",
    "PIP_NO_INDEX": "1",
    "UV_NO_CACHE": "1",
    "UV_NO_CONFIG": "1",
    "UV_OFFLINE": "1",
    "UV_PYTHON_DOWNLOADS": "never",
}
PATH_ENVIRONMENT_OVERRIDES = frozenset({"UV_CACHE_DIR"})
SOURCE_BINDING_KEYS = frozenset(
    {
        "base_commit",
        "binary_diff_sha256",
        "porcelain_sha256",
        "source_state_sha256",
        "untracked_inventory_sha256",
    }
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$", re.ASCII)
SESSION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$", re.ASCII)
MODE_RE = re.compile(r"^0[0-7]{3}$", re.ASCII)
PIP_VERSION_RE = re.compile(
    r"^pip (?P<version>[0-9]+(?:\.[0-9]+){1,2}) from "
    r"(?P<location>.+) \(python (?P<python>[0-9]+\.[0-9]+)\)$",
    re.ASCII,
)


class PackageEvidenceError(RuntimeError):
    """Raised when package evidence cannot be generated safely."""

    exit_code = 2


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
        raise PackageEvidenceError("value is not canonical JSON") from exc


def _canonical_resource_bytes(value: Any) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _semantic_sha256(payload: Mapping[str, Any]) -> str:
    unsealed = dict(payload)
    unsealed.pop(SEMANTIC_FIELD, None)
    return _sha256(_canonical_bytes(unsealed))


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    if SEMANTIC_FIELD in payload:
        raise PackageEvidenceError("semantic_sha256 must not be supplied")
    sealed = dict(payload)
    sealed[SEMANTIC_FIELD] = _semantic_sha256(sealed)
    return sealed


def _mode_string(mode: int) -> str:
    return f"0{stat.S_IMODE(mode):03o}"


def _require_exact_keys(value: Any, keys: set[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise PackageEvidenceError(f"{label} must have exact keys {sorted(keys)!r}")
    return value


def _require_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PackageEvidenceError(f"{label} must be an integer >= {minimum}")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise PackageEvidenceError(f"{label} must be lowercase SHA-256")
    return value


def _load_strict_json(raw: bytes, *, label: str, canonical_resource: bool) -> Any:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise PackageEvidenceError(f"{label} BOM is forbidden")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except PackageEvidenceError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise PackageEvidenceError(f"{label} is invalid JSON") from exc
    if canonical_resource and raw != _canonical_resource_bytes(value):
        raise PackageEvidenceError(f"{label} must be canonical JSON plus newline")
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PackageEvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(token: str) -> None:
    raise PackageEvidenceError(f"non-finite JSON constant rejected: {token}")


def _require_absolute_path(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise PackageEvidenceError(f"{label} must be an absolute path")
    return path


def _load_expected_source_binding(raw: bytes) -> dict[str, str]:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise PackageEvidenceError("expected source binding BOM is forbidden")
    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except PackageEvidenceError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise PackageEvidenceError("expected source binding is invalid JSON") from exc
    if raw != _canonical_resource_bytes(payload):
        raise PackageEvidenceError("expected source binding must be canonical JSON plus newline")
    return _validate_source_binding(payload, label="expected source binding")


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        return False
    return True


def _stable_stat_identity(observed: os.stat_result) -> tuple[int, ...]:
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_uid,
        observed.st_nlink,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


def _directory_identity(observed: os.stat_result) -> tuple[int, int, int, int]:
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_uid,
    )


def _required_open_flag(name: str) -> int:
    value = getattr(os, name, None)
    if type(value) is not int or value == 0:
        raise PackageEvidenceError(f"required OS open flag is unavailable: {name}")
    return value


def _file_object_identity(observed: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_uid,
        observed.st_nlink,
    )


def _read_stable_regular_file(
    path: Path,
    *,
    label: str,
    require_executable: bool = False,
    required_file_mode: int | None = None,
    required_parent_mode: int | None = None,
) -> tuple[Path, bytes]:
    _require_absolute_path(path, label=label)
    absolute = path.absolute()
    parent_descriptor = -1
    file_descriptor = -1
    try:
        parent_before = absolute.parent.lstat()
        parent_resolved = absolute.parent.resolve(strict=True)
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise PackageEvidenceError(f"{label} is unavailable") from exc
    if parent_resolved != absolute.parent or resolved != absolute:
        raise PackageEvidenceError(f"{label} path must not contain symlink indirection")
    if (
        not stat.S_ISDIR(parent_before.st_mode)
        or parent_before.st_uid != os.getuid()
        or stat.S_IMODE(parent_before.st_mode) & 0o022
        or (
            required_parent_mode is not None
            and stat.S_IMODE(parent_before.st_mode) != required_parent_mode
        )
    ):
        raise PackageEvidenceError(f"{label} parent permissions are unsafe")
    try:
        parent_descriptor = os.open(
            absolute.parent,
            os.O_RDONLY
            | _required_open_flag("O_CLOEXEC")
            | _required_open_flag("O_DIRECTORY")
            | _required_open_flag("O_NOFOLLOW"),
        )
        parent_fd_before = os.fstat(parent_descriptor)
        file_before = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(file_before.st_mode)
            or file_before.st_uid != os.getuid()
            or file_before.st_nlink != 1
            or (require_executable and not file_before.st_mode & 0o111)
            or (
                required_file_mode is not None
                and stat.S_IMODE(file_before.st_mode) != required_file_mode
            )
        ):
            raise PackageEvidenceError(f"{label} file permissions or link count are unsafe")
        file_descriptor = os.open(
            absolute.name,
            os.O_RDONLY | _required_open_flag("O_CLOEXEC") | _required_open_flag("O_NOFOLLOW"),
            dir_fd=parent_descriptor,
        )
        file_fd_before = os.fstat(file_descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        file_fd_after = os.fstat(file_descriptor)
        file_after = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        parent_fd_after = os.fstat(parent_descriptor)
        parent_after = absolute.parent.lstat()
    except OSError as exc:
        raise PackageEvidenceError(f"{label} cannot be read stably") from exc
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)
    parent_identities = {
        _stable_stat_identity(parent_before),
        _stable_stat_identity(parent_fd_before),
        _stable_stat_identity(parent_fd_after),
        _stable_stat_identity(parent_after),
    }
    file_identities = {
        _stable_stat_identity(file_before),
        _stable_stat_identity(file_fd_before),
        _stable_stat_identity(file_fd_after),
        _stable_stat_identity(file_after),
    }
    if len(parent_identities) != 1:
        raise PackageEvidenceError(f"{label} parent changed during stable read")
    if len(file_identities) != 1:
        raise PackageEvidenceError(f"{label} changed during stable read")
    return resolved, b"".join(chunks)


def _read_private_external_file(path: Path, *, repo_root: Path, label: str) -> tuple[Path, bytes]:
    _require_absolute_path(path, label=label)
    absolute = path.absolute()
    if _path_within(absolute, repo_root):
        raise PackageEvidenceError(f"{label} must be outside the repository")
    resolved, raw = _read_stable_regular_file(
        absolute,
        label=label,
        required_file_mode=0o600,
        required_parent_mode=0o700,
    )
    return resolved, raw


def _load_module_from_path(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PackageEvidenceError(f"cannot load checked module: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _validate_checked_schema(
    value: Mapping[str, Any],
    *,
    repo_root: Path,
    schema_relative_path: str,
    schema_id: str,
    artifact_version: str,
) -> dict[str, Any]:
    schema_path = repo_root / schema_relative_path
    _resolved, raw = _read_stable_regular_file(schema_path, label=f"{artifact_version} schema")
    schema = _load_strict_json(
        raw,
        label=f"{artifact_version} schema",
        canonical_resource=False,
    )
    if type(schema) is not dict:
        raise PackageEvidenceError(f"{artifact_version} schema must be an object")
    properties = schema.get("properties")
    if (
        schema.get("$id") != schema_id
        or type(properties) is not dict
        or properties.get("version") != {"const": artifact_version}
    ):
        raise PackageEvidenceError(f"{artifact_version} schema identity mismatch")

    package_name = "_myquant_phase0_checked_schema"
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
        for short_name in ("limits", "identities", "canonical"):
            full_name = f"{package_name}.{short_name}"
            _load_module_from_path(full_name, contract_root / f"{short_name}.py")
            loaded_names.append(full_name)
        resources_name = f"{package_name}.resources"
        resources = ModuleType(resources_name)

        class PackageResourceError(Exception):
            pass

        def unavailable_resource(*_args: Any, **_kwargs: Any) -> None:
            raise PackageEvidenceError("packaged resource access is outside schema validation")

        resources.PackageResourceError = PackageResourceError  # type: ignore[attr-defined]
        resources.load_packaged_json = unavailable_resource  # type: ignore[attr-defined]
        sys.modules[resources_name] = resources
        loaded_names.append(resources_name)
        validation_name = f"{package_name}.schema_validation"
        validation = _load_module_from_path(
            validation_name,
            contract_root / "schema_validation.py",
        )
        loaded_names.append(validation_name)
        validation.preflight_packaged_schema(schema)
        validation.validate_instance_against_schema(dict(value), schema)
    except PackageEvidenceError:
        raise
    except Exception as exc:
        raise PackageEvidenceError(f"{artifact_version} schema validation failed") from exc
    finally:
        sys.dont_write_bytecode = previous
        for name in reversed(loaded_names):
            sys.modules.pop(name, None)
    return schema


def _validate_source_binding(value: Any, *, label: str) -> dict[str, str]:
    if type(value) is not dict or set(value) != SOURCE_BINDING_KEYS:
        raise PackageEvidenceError(f"{label} must have the exact Phase 0 source binding keys")
    result: dict[str, str] = {}
    for key in sorted(SOURCE_BINDING_KEYS):
        item = value[key]
        if type(item) is not str:
            raise PackageEvidenceError(f"{label}.{key} must be a string")
        if key == "base_commit":
            if COMMIT_RE.fullmatch(item) is None:
                raise PackageEvidenceError(f"{label}.base_commit is invalid")
        elif SHA256_RE.fullmatch(item) is None:
            raise PackageEvidenceError(f"{label}.{key} is invalid")
        result[key] = item
    return result


def _protected_root_specs(repo_root: Path) -> list[tuple[str, Path]]:
    return [
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
    ]


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
            raise PackageEvidenceError(f"protected root cannot be sampled: {identifier}") from exc
        try:
            resolved = absolute.resolve(strict=True)
        except OSError as exc:
            raise PackageEvidenceError(f"protected root cannot be resolved: {identifier}") from exc
        if (
            stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISDIR(observed.st_mode)
            or resolved != absolute
        ):
            raise PackageEvidenceError(
                f"protected root must be a concrete lexical directory: {identifier}"
            )
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


def _validate_protected_roots(
    value: Any,
    *,
    repo_root: Path,
    label: str,
) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) != len(PROTECTED_ROOT_IDS):
        raise PackageEvidenceError(f"{label} must contain the exact protected-root rows")
    expected_specs = _protected_root_specs(repo_root)
    result: list[dict[str, Any]] = []
    for index, ((expected_id, expected_path), raw_row) in enumerate(
        zip(expected_specs, value, strict=True)
    ):
        row_label = f"{label}[{index}]"
        if type(raw_row) is not dict:
            raise PackageEvidenceError(f"{row_label} must be an object")
        state = raw_row.get("state")
        if state == "ABSENT":
            row = _require_exact_keys(
                raw_row,
                {"id", "path", "state"},
                label=row_label,
            )
        elif state == "PRESENT_DIRECTORY":
            row = _require_exact_keys(
                raw_row,
                {
                    "ctime_ns",
                    "id",
                    "mode",
                    "mtime_ns",
                    "path",
                    "realpath",
                    "st_dev",
                    "st_ino",
                    "state",
                    "uid",
                },
                label=row_label,
            )
            if (
                type(row["realpath"]) is not str
                or row["realpath"] != row["path"]
                or type(row["mode"]) is not str
                or MODE_RE.fullmatch(row["mode"]) is None
            ):
                raise PackageEvidenceError(f"{row_label} present directory binding is invalid")
            for key in ("ctime_ns", "mtime_ns", "st_dev", "st_ino", "uid"):
                _require_int(row[key], label=f"{row_label}.{key}")
        else:
            raise PackageEvidenceError(f"{row_label}.state is invalid")
        if row["id"] != expected_id or row["path"] != str(expected_path.absolute()):
            raise PackageEvidenceError(f"{row_label} identity/path mismatch")
        result.append(dict(row))
    return result


def _validate_binding(value: Any, *, label: str) -> dict[str, Any]:
    binding = _require_exact_keys(value, {"row_count", "sha256"}, label=label)
    _require_int(binding["row_count"], label=f"{label}.row_count", minimum=1)
    _require_sha256(binding["sha256"], label=f"{label}.sha256")
    return dict(binding)


def _load_package_parity_module(repo_root: Path) -> ModuleType:
    path = repo_root / "quant_investor" / "v17_v2_contract" / "package_parity.py"
    name = "_v17_phase0_package_parity_for_evidence"
    previous = sys.dont_write_bytecode
    inserted_tomllib = False
    try:
        sys.dont_write_bytecode = True
        if importlib.util.find_spec("tomllib") is None:
            tomllib_stub = ModuleType("tomllib")

            def unavailable_toml(*_args: Any, **_kwargs: Any) -> None:
                raise PackageEvidenceError("TOML parsing is outside physical source sampling")

            tomllib_stub.loads = unavailable_toml  # type: ignore[attr-defined]
            sys.modules["tomllib"] = tomllib_stub
            inserted_tomllib = True
        return _load_module_from_path(name, path)
    except Exception as exc:
        raise PackageEvidenceError("cannot import package-parity source helpers") from exc
    finally:
        sys.dont_write_bytecode = previous
        sys.modules.pop(name, None)
        if inserted_tomllib:
            sys.modules.pop("tomllib", None)


def _validate_physical_superset(value: Any, *, label: str) -> dict[str, Any]:
    session = _require_exact_keys(
        value,
        {"row_count", "rows", "sha256"},
        label=label,
    )
    rows = session["rows"]
    if type(rows) is not list or not rows:
        raise PackageEvidenceError(f"{label}.rows must be a nonempty array")
    normalized: list[dict[str, Any]] = []
    collision_keys: set[str] = set()
    previous_key: tuple[str, str] | None = None
    for index, raw_row in enumerate(rows):
        row_label = f"{label}.rows[{index}]"
        row = _require_exact_keys(
            raw_row,
            {"kind", "mode", "path", "sha256", "size_bytes"},
            label=row_label,
        )
        path = row["path"]
        if type(path) is not str:
            raise PackageEvidenceError(f"{row_label}.path must be a string")
        pure = PurePosixPath(path)
        if (
            not path
            or pure.is_absolute()
            or path != pure.as_posix()
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise PackageEvidenceError(f"{row_label}.path must be safe POSIX-relative")
        collision = path.casefold()
        if collision in collision_keys:
            raise PackageEvidenceError(f"{label} contains a case-colliding path")
        collision_keys.add(collision)
        if type(row["mode"]) is not int or not 0 <= row["mode"] <= 0o7777:
            raise PackageEvidenceError(f"{row_label}.mode is invalid")
        size = _require_int(row["size_bytes"], label=f"{row_label}.size_bytes")
        if row["kind"] == "directory":
            if row["sha256"] is not None or size != 0:
                raise PackageEvidenceError(f"{row_label} directory binding is invalid")
        elif row["kind"] == "file":
            _require_sha256(row["sha256"], label=f"{row_label}.sha256")
        else:
            raise PackageEvidenceError(f"{row_label}.kind is invalid")
        sort_key = (path, path)
        if previous_key is not None and sort_key <= previous_key:
            raise PackageEvidenceError(f"{label}.rows are not canonical")
        previous_key = sort_key
        normalized.append(dict(row))
    if session["row_count"] != len(normalized):
        raise PackageEvidenceError(f"{label}.row_count mismatch")
    digest = _sha256(_canonical_bytes(normalized))
    if session["sha256"] != digest:
        raise PackageEvidenceError(f"{label}.sha256 mismatch")
    return {"row_count": len(normalized), "rows": normalized, "sha256": digest}


def _sample_physical_superset(repo_root: Path) -> dict[str, Any]:
    module = _load_package_parity_module(repo_root)
    collector = getattr(module, "collect_physical_source_superset", None)
    if not callable(collector):
        raise PackageEvidenceError("package-parity physical superset helper is unavailable")
    try:
        value = collector(repo_root, extra_paths=HATCH_EXTRA_PATHS)
    except Exception as exc:
        raise PackageEvidenceError("cannot sample package physical source superset") from exc
    return _validate_physical_superset(value, label="package physical source superset")


def _resolve_repo_root(repo_root: Path) -> Path:
    _require_absolute_path(repo_root, label="repo root")
    try:
        resolved = repo_root.resolve(strict=True)
    except OSError as exc:
        raise PackageEvidenceError("repo root is unavailable") from exc
    if not (resolved / ".git").exists():
        raise PackageEvidenceError("repo root must be a git worktree")
    return resolved


def _schema_row(
    rows: Any,
    *,
    artifact_version: str,
    schema_id: str,
    relative_path: str,
    repo_root: Path,
) -> dict[str, Any]:
    if type(rows) is not list:
        raise PackageEvidenceError("session schemas must be an array")
    matches = [
        row for row in rows if type(row) is dict and row.get("artifact_version") == artifact_version
    ]
    if len(matches) != 1:
        raise PackageEvidenceError(f"session must bind exactly one {artifact_version} schema")
    row = _require_exact_keys(
        matches[0],
        {"artifact_version", "path", "schema_id", "sha256", "size_bytes"},
        label=f"session schema {artifact_version}",
    )
    if row["schema_id"] != schema_id or row["path"] != relative_path:
        raise PackageEvidenceError(f"session schema identity mismatch: {artifact_version}")
    actual = _file_binding(
        repo_root / relative_path,
        label=f"session schema file {artifact_version}",
    )
    if row["sha256"] != actual["sha256"] or row["size_bytes"] != actual["size_bytes"]:
        raise PackageEvidenceError(f"session schema bytes drifted: {artifact_version}")
    return dict(row)


def _load_session_manifest(
    path: Path,
    *,
    repo_root: Path,
    expected_base_commit: str,
) -> tuple[dict[str, Any], dict[str, Any], bytes, Path]:
    resolved, raw = _read_private_external_file(
        path,
        repo_root=repo_root,
        label="session manifest",
    )
    value = _load_strict_json(raw, label="session manifest", canonical_resource=True)
    manifest = _require_exact_keys(value, SESSION_KEYS, label="session manifest")
    _validate_checked_schema(
        manifest,
        repo_root=repo_root,
        schema_relative_path=SESSION_SCHEMA_PATH,
        schema_id=SESSION_SCHEMA_ID,
        artifact_version=SESSION_VERSION,
    )
    if (
        manifest["version"] != SESSION_VERSION
        or manifest["protocol_version"] != PROTOCOL_VERSION
        or manifest["status"] != "INITIALIZED"
        or manifest["authority"] is not False
        or manifest["repo_root"] != str(repo_root)
        or manifest["base_commit"] != expected_base_commit
    ):
        raise PackageEvidenceError("session manifest envelope does not match package step")
    session_id = manifest["session_id"]
    if type(session_id) is not str or SESSION_ID_RE.fullmatch(session_id) is None:
        raise PackageEvidenceError("session manifest session_id is invalid")
    _require_sha256(manifest[SEMANTIC_FIELD], label="session manifest semantic_sha256")
    if manifest[SEMANTIC_FIELD] != _semantic_sha256(manifest):
        raise PackageEvidenceError("session manifest semantic SHA-256 mismatch")
    source_binding = _validate_source_binding(
        manifest["source_binding"],
        label="session source_binding",
    )
    if source_binding["base_commit"] != expected_base_commit:
        raise PackageEvidenceError("session source binding base commit mismatch")
    if manifest["limitations"] != LIMITATIONS:
        raise PackageEvidenceError("session limitations mismatch")
    _validate_binding(
        manifest["package_source_superset"],
        label="session package_source_superset",
    )
    _validate_protected_roots(
        manifest["protected_roots"],
        repo_root=repo_root,
        label="session protected_roots",
    )
    _schema_row(
        manifest["schemas"],
        artifact_version=SESSION_VERSION,
        schema_id=SESSION_SCHEMA_ID,
        relative_path=SESSION_SCHEMA_PATH,
        repo_root=repo_root,
    )
    _schema_row(
        manifest["schemas"],
        artifact_version=PACKAGE_EVIDENCE_VERSION,
        schema_id=PACKAGE_EVIDENCE_SCHEMA_ID,
        relative_path=PACKAGE_EVIDENCE_SCHEMA_PATH,
        repo_root=repo_root,
    )
    expected_gate = {
        "artifact_version": PACKAGE_EVIDENCE_VERSION,
        "filename": STEP["filename"],
        "kind": STEP["kind"],
        "ordinal": STEP["ordinal"],
        "producer_path": "scripts/v17_phase0_package_evidence.py",
        "producer_version": PACKAGE_PRODUCER_VERSION,
        "role": STEP["role"],
        "schema_id": PACKAGE_EVIDENCE_SCHEMA_ID,
    }
    gate_rows = manifest["gate_plan"]
    if (
        type(gate_rows) is not list
        or len(gate_rows) != len(EXPECTED_GATE_ROLES)
        or gate_rows[STEP["ordinal"] - 1] != expected_gate
    ):
        raise PackageEvidenceError("session gate_plan package step mismatch")
    binding = {
        "path": str(resolved),
        "semantic_sha256": manifest[SEMANTIC_FIELD],
        "session_id": session_id,
        "sha256": _sha256(raw),
        "size_bytes": len(raw),
    }
    return dict(manifest), binding, raw, resolved


def _load_phase0_index_module(repo_root: Path) -> Any:
    path = repo_root / "scripts" / "v17_phase0_evidence_index.py"
    spec = importlib.util.spec_from_file_location("v17_phase0_evidence_index_for_package", path)
    if spec is None or spec.loader is None:
        raise PackageEvidenceError("cannot load Phase 0 evidence index helpers")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - fail-closed import guard
        raise PackageEvidenceError("cannot import Phase 0 evidence index helpers") from exc
    finally:
        sys.dont_write_bytecode = previous
    return module


def _sample_source_binding(repo_root: Path, base_commit: str) -> dict[str, str]:
    module = _load_phase0_index_module(repo_root)
    try:
        snapshot = module._git_snapshot(repo_root, base_commit)
        public = module._public_source_state(snapshot)
        binding = module._source_binding_from_state(public)
    except Exception as exc:
        raise PackageEvidenceError("cannot sample Phase 0 source binding") from exc
    return _validate_source_binding(binding, label="sampled source binding")


def _private_fresh_work_root(path: Path, *, repo_root: Path) -> Path:
    _require_absolute_path(path, label="work root")
    if _path_within(path, repo_root):
        raise PackageEvidenceError("work root must be outside the repository")
    try:
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
    except OSError as exc:
        raise PackageEvidenceError("work root must be fresh and nonexisting") from exc
    try:
        resolved = path.resolve(strict=True)
        observed = resolved.lstat()
    except OSError as exc:
        raise PackageEvidenceError("work root cannot be resolved after creation") from exc
    if (
        stat.S_IMODE(observed.st_mode) != 0o700
        or not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != os.getuid()
        or resolved.is_symlink()
    ):
        raise PackageEvidenceError("work root must be owner-private 0700")
    return resolved


def _assert_existing_executable(path: Path, *, label: str) -> Path:
    _require_absolute_path(path, label=label)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise PackageEvidenceError(f"{label} is unavailable") from exc
    stable, _raw = _read_stable_regular_file(
        resolved,
        label=label,
        require_executable=True,
    )
    return stable


def _assert_existing_uv_cache(path: Path, *, label: str) -> Path:
    _require_absolute_path(path, label=label)
    absolute = path.absolute()
    try:
        resolved = absolute.resolve(strict=True)
        observed = resolved.lstat()
    except OSError as exc:
        raise PackageEvidenceError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISDIR(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) & 0o022
        or observed.st_uid != os.getuid()
        or resolved.is_symlink()
        or resolved != absolute
    ):
        raise PackageEvidenceError(
            f"{label} must be a concrete owner-owned directory and not group/other writable"
        )
    return resolved


def _executable_binding(
    lexical_path: Path,
    resolved_path: Path,
    file_binding: Mapping[str, Any],
    *,
    implementation: str | None = None,
    version: str,
    output: str | None = None,
    version_info: list[int] | None = None,
) -> dict[str, Any]:
    try:
        observed = resolved_path.lstat()
    except OSError as exc:
        raise PackageEvidenceError("tool executable disappeared during binding") from exc
    binding: dict[str, Any] = {
        "executable": bool(observed.st_mode & 0o111),
        "lexical_path": str(lexical_path.absolute()),
        "mode": _mode_string(observed.st_mode),
        "realpath": str(resolved_path),
        "sha256": file_binding["sha256"],
        "size_bytes": file_binding["size_bytes"],
        "version": version,
    }
    if implementation is not None:
        binding["implementation"] = implementation
    if output is not None:
        binding["output"] = output
    if version_info is not None:
        binding["version_info"] = version_info
    return binding


def _uv_cache_binding(path: Path) -> dict[str, Any]:
    absolute = path.absolute()
    try:
        resolved = absolute.resolve(strict=True)
        observed = absolute.lstat()
    except OSError as exc:
        raise PackageEvidenceError("uv cache disappeared during binding") from exc
    if (
        resolved != absolute
        or not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != os.getuid()
        or stat.S_IMODE(observed.st_mode) & 0o022
    ):
        raise PackageEvidenceError("uv cache binding is unsafe")
    return {
        "mode": _mode_string(observed.st_mode),
        "path": str(absolute),
        "realpath": str(resolved),
        "st_dev": observed.st_dev,
        "st_ino": observed.st_ino,
        "uid": observed.st_uid,
    }


def _validate_output_target(path: Path, *, repo_root: Path) -> Path:
    _require_absolute_path(path, label="output JSON")
    absolute = path.absolute()
    if _path_within(absolute, repo_root):
        raise PackageEvidenceError("output JSON must be outside the repository")
    try:
        parent_stat = absolute.parent.lstat()
        parent = absolute.parent.resolve(strict=True)
    except OSError as exc:
        raise PackageEvidenceError("output JSON parent is unavailable") from exc
    if parent != absolute.parent:
        raise PackageEvidenceError("output JSON parent must not be a symlink")
    if (
        not stat.S_ISDIR(parent_stat.st_mode)
        or stat.S_IMODE(parent_stat.st_mode) != 0o700
        or parent_stat.st_uid != os.getuid()
        or stat.S_ISLNK(parent_stat.st_mode)
    ):
        raise PackageEvidenceError("output JSON parent must be owner-private 0700")
    return parent / absolute.name


def _file_binding(path: Path, *, label: str) -> dict[str, Any]:
    resolved, raw = _read_stable_regular_file(path, label=label)
    return {
        "path": str(resolved),
        "sha256": _sha256(raw),
        "size_bytes": len(raw),
    }


def _bytes_binding(raw: bytes) -> dict[str, Any]:
    return {
        "bytes_base64": base64.b64encode(raw).decode("ascii"),
        "encoding": "base64",
        "sha256": _sha256(raw),
        "size_bytes": len(raw),
    }


def _selected_env(**values: str) -> dict[str, str]:
    return {key: value for key, value in values.items() if value}


def _environment_proof(
    env_overrides: Mapping[str, str] | None,
) -> tuple[dict[str, str], dict[str, Any]]:
    overrides = dict(env_overrides or {})
    if any(type(key) is not str or not key for key in overrides):
        raise PackageEvidenceError("environment override names must be nonempty strings")
    if any(type(value) is not str or "\x00" in value for value in overrides.values()):
        raise PackageEvidenceError("environment override values must be safe strings")
    unknown_keys = set(overrides) - set(FIXED_ENVIRONMENT_OVERRIDES) - PATH_ENVIRONMENT_OVERRIDES
    if unknown_keys:
        raise PackageEvidenceError("environment overrides contain a non-allowlisted name")
    if any(
        overrides.get(key, expected) != expected
        for key, expected in FIXED_ENVIRONMENT_OVERRIDES.items()
    ):
        raise PackageEvidenceError("fixed environment override value mismatch")
    for key in PATH_ENVIRONMENT_OVERRIDES.intersection(overrides):
        candidate = Path(overrides[key])
        if not candidate.is_absolute() or "\n" in overrides[key] or "\r" in overrides[key]:
            raise PackageEvidenceError("path environment override value is unsafe")
    effective = {**SAFE_EXECUTION_ENVIRONMENT, **overrides}
    host_names = sorted(os.environ)
    proof = {
        "base_environment": dict(sorted(SAFE_EXECUTION_ENVIRONMENT.items())),
        "effective_environment": dict(sorted(effective.items())),
        "host_environment": {
            "inherited_value_count": 0,
            "secret_values_recorded": False,
            "stripped_variable_name_count": len(host_names),
            "stripped_variable_names_sha256": _sha256(_canonical_bytes(host_names)),
        },
        "overrides": dict(sorted(overrides.items())),
    }
    return effective, proof


def _run_command(
    argv: Sequence[str],
    *,
    role: str,
    cwd: Path,
    env_overrides: Mapping[str, str] | None,
    tool_version: str,
    commands: list[dict[str, Any]],
) -> subprocess.CompletedProcess[bytes]:
    if not argv or any(type(item) is not str or not item for item in argv):
        raise PackageEvidenceError("command argv must be a nonempty string array")
    if any(item in {"|", "&&", "||", ";"} for item in argv):
        raise PackageEvidenceError("shell control operators are forbidden")
    env, environment_proof = _environment_proof(env_overrides)
    execution_env = dict(env)
    try:
        completed = subprocess.run(
            list(argv),
            cwd=cwd,
            env=execution_env,
            shell=False,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise PackageEvidenceError(f"cannot execute command: {argv[0]}") from exc
    if execution_env != env:
        raise PackageEvidenceError("command environment changed during execution")
    if environment_proof["effective_environment"] != dict(sorted(execution_env.items())):
        raise PackageEvidenceError("recorded command environment does not match execution")
    commands.append(
        {
            "argv": list(argv),
            "cwd": str(cwd),
            "env": dict(sorted((env_overrides or {}).items())),
            "sanitized_environment": environment_proof,
            "exit_code": completed.returncode,
            "role": role,
            "stderr": _bytes_binding(completed.stderr),
            "stdout": _bytes_binding(completed.stdout),
            "tool_version": tool_version,
        }
    )
    if completed.returncode != 0:
        raise PackageEvidenceError(f"command failed: {argv[0]}")
    return completed


def _parse_json_stdout(
    completed: subprocess.CompletedProcess[bytes], *, label: str
) -> dict[str, Any]:
    try:
        payload = json.loads(
            completed.stdout.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except PackageEvidenceError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise PackageEvidenceError(f"{label} did not emit JSON") from exc
    if type(payload) is not dict:
        raise PackageEvidenceError(f"{label} JSON must be an object")
    return payload


def _python_probe_code() -> str:
    return (
        "import hashlib,importlib.util,json,os,pathlib,platform,sys;"
        "p=os.path.realpath(sys.executable);"
        "raw=open(p,'rb').read();"
        "loaded_before=sorted(n for n in sys.modules if n=='pip' or n.startswith('pip.'));"
        "spec=importlib.util.find_spec('pip');"
        "loaded_after=sorted(n for n in sys.modules if n=='pip' or n.startswith('pip.'));"
        "loaded=sorted(set(loaded_before+loaded_after));"
        "site_paths=sorted(x for x in sys.path if any("
        "part.casefold() in {'site-packages','dist-packages'} for part in pathlib.Path(x).parts));"
        "print(json.dumps({"
        "'executable':sys.executable,"
        "'implementation':platform.python_implementation().lower(),"
        "'pip_absence':{"
        "'loaded_modules':loaded,"
        "'observation_scope':'NON_IMPORTING_ISOLATED_NO_SITE_RUNTIME_VISIBILITY_ONLY',"
        "'find_spec_present':spec is not None,"
        "'site_sys_path_entries':site_paths"
        "},"
        "'realpath':p,"
        "'runtime_flags':{"
        "'dont_write_bytecode':sys.flags.dont_write_bytecode,"
        "'ignore_environment':sys.flags.ignore_environment,"
        "'isolated':sys.flags.isolated,"
        "'no_site':sys.flags.no_site,"
        "'no_user_site':sys.flags.no_user_site,"
        "'safe_path':bool(getattr(sys.flags,'safe_path',False))"
        "},"
        "'sha256':hashlib.sha256(raw).hexdigest(),"
        "'version':platform.python_version(),"
        "'version_info':list(sys.version_info[:3])"
        "},sort_keys=True,separators=(',',':')))"
    )


def _backend_probe_code() -> str:
    return (
        "import importlib,importlib.metadata as m,importlib.util,json,pathlib,sys,sysconfig;"
        "backend=importlib.import_module('hatchling.build');"
        "dists=list(m.distributions());"
        "inventory=sorted("
        "[{'name':d.metadata['Name'],'version':d.version} "
        "for d in dists if d.metadata.get('Name')],"
        "key=lambda item:item['name'].casefold());"
        "versions={item['name']:item['version'] for item in inventory};"
        "pure=pathlib.Path(sysconfig.get_paths()['purelib']);"
        "plat=pathlib.Path(sysconfig.get_paths()['platlib']);"
        "pip_names=sorted({d.metadata.get('Name','') for d in dists "
        "if d.metadata.get('Name','').lower()=='pip'});"
        "pip_paths=sorted({str(x.resolve()) for root in {pure,plat} if root.exists() "
        "for x in root.glob('pip*')});"
        "pip_wrappers=sorted("
        "str(x.resolve()) for x in pathlib.Path(sys.executable).parent.glob('pip*'));"
        "print(json.dumps({"
        "'backend_module':backend.__name__,"
        "'backend_file':getattr(backend,'__file__',None),"
        "'hatchling_version':m.version('hatchling'),"
        "'package_inventory':inventory,"
        "'package_versions':dict(sorted(versions.items(),key=lambda kv:kv[0].casefold())),"
        "'pip_absence':{"
        "'distribution_names':pip_names,"
        "'find_spec_present':importlib.util.find_spec('pip') is not None,"
        "'package_paths':pip_paths,"
        "'wrapper_paths':pip_wrappers"
        "},"
        "'unnamed_distribution_count':sum("
        "1 for d in dists if not d.metadata.get('Name'))"
        "},sort_keys=True,separators=(',',':')))"
    )


def _pip_wrapper_probe_expression() -> str:
    return (
        "[{'is_symlink':p.is_symlink(),'mode':f'0{(p.lstat().st_mode & 0o777):03o}',"
        "'name':p.name,'path':str(p.absolute()),"
        "'sha256':None if p.is_symlink() else hashlib.sha256(p.read_bytes()).hexdigest(),"
        "'size_bytes':p.lstat().st_size} "
        "for p in sorted(pathlib.Path(sys.executable).parent.glob('pip*'),key=lambda p:p.name)]"
    )


def _install_inventory_probe_code() -> str:
    return (
        "import hashlib,importlib.metadata as m,importlib.util,json,pathlib,sys,sysconfig;"
        "dists=sorted([{'name':d.metadata['Name'],'version':d.version} "
        "for d in m.distributions() if d.metadata.get('Name')],"
        "key=lambda item:item['name'].casefold());"
        "pure=pathlib.Path(sysconfig.get_paths()['purelib']).resolve();"
        "pip_paths=sorted(str(p.resolve()) for p in pure.glob('pip*'));"
        f"wrappers={_pip_wrapper_probe_expression()};"
        "print(json.dumps({"
        "'distribution_inventory':dists,"
        "'pip_find_spec_present':importlib.util.find_spec('pip') is not None,"
        "'pip_package_paths':pip_paths,"
        "'pip_wrappers':wrappers,"
        "'plain_pip_absent':not (pathlib.Path(sys.executable).parent/'pip').exists(),"
        "'site_packages_root':str(pure)"
        "},sort_keys=True,separators=(',',':')))"
    )


def _installed_paths_probe_code() -> str:
    return (
        "import hashlib,importlib.metadata as m,importlib.util,json,pathlib,sys,sysconfig;"
        "spec=importlib.util.find_spec('quant_investor');"
        "pkg=pathlib.Path(spec.submodule_search_locations[0]).resolve();"
        "site=pkg.parent;"
        "matches=sorted(site.glob('quant_investor-*.dist-info'));"
        "dists=sorted([{'name':d.metadata['Name'],'version':d.version} "
        "for d in m.distributions() if d.metadata.get('Name')],"
        "key=lambda item:item['name'].casefold());"
        "pip_paths=sorted(str(p.resolve()) for p in site.glob('pip*'));"
        f"wrappers={_pip_wrapper_probe_expression()};"
        "print(json.dumps({"
        "'distribution_inventory':dists,"
        "'installed_package_root':str(pkg),"
        "'installed_dist_info':str(matches[0]) if len(matches)==1 else None,"
        "'pip_find_spec_present':importlib.util.find_spec('pip') is not None,"
        "'pip_package_paths':pip_paths,"
        "'pip_wrappers':wrappers,"
        "'plain_pip_absent':not (pathlib.Path(sys.executable).parent/'pip').exists(),"
        "'site_packages_root':str(site)"
        "},sort_keys=True,separators=(',',':')))"
    )


def _ensurepip_bundle_probe_code() -> str:
    return (
        "import hashlib,json,os,pathlib,stat,ensurepip;"
        "root=pathlib.Path(ensurepip.__file__).resolve().parent/'_bundled';"
        "matches=sorted(root.glob('*.whl'),key=lambda p:p.name);"
        "p=matches[0] if len(matches)==1 else None;"
        "before=p.lstat() if p is not None else None;"
        "flags=os.O_RDONLY|getattr(os,'O_CLOEXEC',0)|getattr(os,'O_NOFOLLOW',0);"
        "fd=os.open(p,flags) if p is not None else -1;"
        "raw=b'';"
        "raw=b''.join(iter(lambda:os.read(fd,1048576),b'')) if fd>=0 else b'';"
        "opened=os.fstat(fd) if fd>=0 else None;"
        "os.close(fd) if fd>=0 else None;"
        "after=p.lstat() if p is not None else None;"
        "print(json.dumps({"
        "'ensurepip_version':ensurepip.version(),"
        "'match_count':len(matches),"
        "'wheel':None if p is None else {"
        "'is_symlink':p.is_symlink(),"
        "'mode':f'0{(before.st_mode & 0o777):03o}',"
        "'name':p.name,"
        "'nlink':before.st_nlink,"
        "'path':str(p.absolute()),"
        "'realpath':str(p.resolve()),"
        "'sha256':hashlib.sha256(raw).hexdigest(),"
        "'size_bytes':len(raw),"
        "'stable':(before.st_dev,before.st_ino,before.st_mode,before.st_size,"
        "before.st_mtime_ns,before.st_ctime_ns)=="
        "(opened.st_dev,opened.st_ino,opened.st_mode,opened.st_size,"
        "opened.st_mtime_ns,opened.st_ctime_ns)=="
        "(after.st_dev,after.st_ino,after.st_mode,after.st_size,"
        "after.st_mtime_ns,after.st_ctime_ns)"
        "}"
        "},sort_keys=True,separators=(',',':')))"
    )


def _hatch_selector_probe_code() -> str:
    return (
        "import hashlib,importlib.metadata as md,importlib.util,json,pathlib,sys,hatchling.build;"
        "sys.dont_write_bytecode=True;"
        "root=pathlib.Path(sys.argv[1]).resolve();"
        "module_path=root/'quant_investor/v17_v2_contract/package_parity.py';"
        "spec=importlib.util.spec_from_file_location('_phase0_parity_selector',module_path);"
        "mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;"
        "spec.loader.exec_module(mod);"
        "from hatchling.builders.sdist import SdistBuilder;"
        "from hatchling.builders.wheel import WheelBuilder;"
        "superset=mod.collect_physical_source_superset("
        "root,extra_paths=('README.md','pyproject.toml','requirements.txt'));"
        "physical={row['path']:row for row in superset['rows'] if row['kind']=='file'};"
        "rows=[];"
        "builders=(('sdist',SdistBuilder(str(root))),('wheel',WheelBuilder(str(root))));"
        "[(lambda source,row,target,item:rows.append({"
        "'distribution_path':pathlib.PurePosixPath(item.distribution_path).as_posix(),"
        "'mode':row['mode'],'sha256':row['sha256'],'size_bytes':row['size_bytes'],"
        "'source_path':source,'target':target"
        "}))(pathlib.Path(item.path).resolve().relative_to(root).as_posix(),"
        "physical[pathlib.Path(item.path).resolve().relative_to(root).as_posix()],"
        "target,item) for target,builder in builders for item in builder.recurse_included_files()];"
        "rows.sort(key=lambda row:(row['target'],row['distribution_path'],row['source_path']));"
        "namespace=mod.validate_hatch_namespace_rows(rows,physical_superset=superset);"
        "modules={};"
        "[(lambda p,name,raw:modules.__setitem__(name,{"
        "'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)"
        "}))(pathlib.Path(sys.modules[name].__file__).resolve(),name,"
        "pathlib.Path(sys.modules[name].__file__).resolve().read_bytes()) "
        "for name in ('hatchling.build','hatchling.builders.sdist','hatchling.builders.wheel')];"
        "raw=module_path.read_bytes();"
        "modules['package_parity']={'path':str(module_path),'sha256':hashlib.sha256(raw).hexdigest(),"
        "'size_bytes':len(raw)};"
        "print(json.dumps({"
        "'hatch_source_namespace':namespace,"
        "'hatchling_version':md.version('hatchling'),"
        "'package_source_superset':superset,"
        "'selector_modules':modules"
        "},sort_keys=True,separators=(',',':')))"
    )


def _require_cpython313(binding: Mapping[str, Any]) -> None:
    if (
        binding.get("implementation") != "cpython"
        or not isinstance(binding.get("version_info"), list)
        or binding["version_info"] != EXPECTED_PYTHON_VERSION_INFO
        or binding.get("version") != EXPECTED_PYTHON_VERSION
    ):
        raise PackageEvidenceError(
            f"base interpreter must be exact CPython {EXPECTED_PYTHON_VERSION}"
        )
    if not isinstance(binding.get("realpath"), str) or not isinstance(binding.get("sha256"), str):
        raise PackageEvidenceError("base interpreter binding is incomplete")
    if binding.get("runtime_flags") != EXPECTED_BASE_RUNTIME_FLAGS:
        raise PackageEvidenceError("base interpreter does not prove -I -S -B")
    _validate_runtime_pip_absence(binding.get("pip_absence"), label="base interpreter pip absence")


def _validate_runtime_pip_absence(value: Any, *, label: str) -> dict[str, Any]:
    proof = _require_exact_keys(
        value,
        {
            "find_spec_present",
            "loaded_modules",
            "observation_scope",
            "site_sys_path_entries",
        },
        label=label,
    )
    if (
        proof["find_spec_present"] is not False
        or proof["loaded_modules"] != []
        or proof["observation_scope"] != BASE_RUNTIME_PIP_OBSERVATION_SCOPE
        or proof["site_sys_path_entries"] != []
    ):
        raise PackageEvidenceError(f"{label} is not exact")
    return dict(proof)


def _validate_pip_absence(value: Any, *, label: str) -> dict[str, Any]:
    proof = _require_exact_keys(
        value,
        {"distribution_names", "find_spec_present", "package_paths", "wrapper_paths"},
        label=label,
    )
    if (
        proof["distribution_names"] != []
        or proof["find_spec_present"] is not False
        or proof["package_paths"] != []
        or proof["wrapper_paths"] != []
    ):
        raise PackageEvidenceError(f"{label} is not exact")
    return dict(proof)


def _parse_uv_version(raw: bytes) -> dict[str, str]:
    try:
        decoded = raw.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise PackageEvidenceError("uv version probe returned invalid UTF-8") from exc
    if decoded != f"{EXPECTED_UV_VERSION_OUTPUT}\n":
        raise PackageEvidenceError(f"uv version must be the frozen {EXPECTED_UV_VERSION}")
    output = decoded[:-1]
    return {"output": output, "version": EXPECTED_UV_VERSION}


def _parse_pip_version(raw: bytes, *, install_venv: Path) -> dict[str, str]:
    try:
        decoded = raw.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise PackageEvidenceError("pip version probe returned invalid UTF-8") from exc
    if not decoded.endswith("\n") or decoded.endswith("\n\n"):
        raise PackageEvidenceError("pip version probe must emit exactly one line")
    output = decoded[:-1]
    if "\n" in output or "\r" in output:
        raise PackageEvidenceError("pip version probe must emit exactly one line")
    matched = PIP_VERSION_RE.fullmatch(output)
    if matched is None:
        raise PackageEvidenceError("pip version probe returned an invalid version")
    version = matched.group("version")
    if version != EXPECTED_PIP_VERSION or matched.group("python") != "3.13":
        raise PackageEvidenceError(
            f"pip version must be the frozen {EXPECTED_PIP_VERSION} for Python 3.13"
        )
    location = Path(matched.group("location"))
    if not location.is_absolute() or not _path_within(location, install_venv):
        raise PackageEvidenceError("pip version probe is outside the fresh install environment")
    return {
        "location": str(location),
        "output": output,
        "python_version": "3.13",
        "version": version,
    }


def _validate_backend_probe(value: Mapping[str, Any], *, build_venv: Path) -> dict[str, Any]:
    if (
        value.get("backend_module") != "hatchling.build"
        or value.get("hatchling_version") != EXPECTED_BUILD_BACKEND_PACKAGES["hatchling"]
    ):
        raise PackageEvidenceError("build backend identity/version mismatch")
    backend_file = value.get("backend_file")
    if type(backend_file) is not str:
        raise PackageEvidenceError("build backend file binding is missing")
    backend_path = Path(backend_file)
    if not backend_path.is_absolute() or not _path_within(backend_path, build_venv):
        raise PackageEvidenceError("build backend file is outside the fresh build environment")
    package_versions = value.get("package_versions")
    if type(package_versions) is not dict or package_versions != EXPECTED_BUILD_BACKEND_PACKAGES:
        raise PackageEvidenceError("build backend package inventory mismatch")
    unnamed_count = value.get("unnamed_distribution_count")
    if (
        type(value.get("package_inventory")) is not list
        or value.get("package_inventory") != EXPECTED_BUILD_BACKEND_INVENTORY
        or type(unnamed_count) is not int
        or unnamed_count != 0
    ):
        raise PackageEvidenceError("build backend distribution inventory mismatch")
    _validate_pip_absence(value.get("pip_absence"), label="build environment pip absence")
    return dict(value)


def _validate_bundle_probe(value: Any, *, install_venv: Path, label: str) -> dict[str, Any]:
    proof = _require_exact_keys(
        value,
        {"ensurepip_version", "match_count", "wheel"},
        label=label,
    )
    if proof["ensurepip_version"] != EXPECTED_PIP_VERSION or proof["match_count"] != 1:
        raise PackageEvidenceError(f"{label} must expose exact bundled pip {EXPECTED_PIP_VERSION}")
    wheel = _require_exact_keys(
        proof["wheel"],
        {
            "is_symlink",
            "mode",
            "name",
            "nlink",
            "path",
            "realpath",
            "sha256",
            "size_bytes",
            "stable",
        },
        label=f"{label}.wheel",
    )
    if (
        wheel["name"] != EXPECTED_PIP_WHEEL_NAME
        or wheel["size_bytes"] != EXPECTED_PIP_WHEEL_SIZE
        or wheel["sha256"] != EXPECTED_PIP_WHEEL_SHA256
        or wheel["is_symlink"] is not False
        or wheel["stable"] is not True
        or wheel["nlink"] != 1
        or wheel["path"] != wheel["realpath"]
        or type(wheel["mode"]) is not str
        or MODE_RE.fullmatch(wheel["mode"]) is None
    ):
        raise PackageEvidenceError(f"{label} bundled pip wheel mismatch")
    path = Path(str(wheel["path"]))
    if not path.is_absolute() or _path_within(path, install_venv):
        raise PackageEvidenceError(f"{label} bundled wheel must belong to the base interpreter")
    return {"ensurepip_version": EXPECTED_PIP_VERSION, "match_count": 1, "wheel": dict(wheel)}


def _validate_pip_wrappers(
    value: Any,
    *,
    install_venv: Path,
    label: str,
) -> list[dict[str, Any]]:
    if type(value) is not list or [item.get("name") for item in value if type(item) is dict] != (
        EXPECTED_PIP_WRAPPERS
    ):
        raise PackageEvidenceError(f"{label} must contain only {EXPECTED_PIP_WRAPPERS!r}")
    wrappers: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        item = _require_exact_keys(
            raw,
            {"is_symlink", "mode", "name", "path", "sha256", "size_bytes"},
            label=f"{label}[{index}]",
        )
        path = Path(str(item["path"]))
        if (
            item["is_symlink"] is not False
            or type(item["mode"]) is not str
            or MODE_RE.fullmatch(item["mode"]) is None
            or not path.is_absolute()
            or path.parent != install_venv / "bin"
        ):
            raise PackageEvidenceError(f"{label}[{index}] wrapper binding is unsafe")
        _require_sha256(item["sha256"], label=f"{label}[{index}].sha256")
        _require_int(item["size_bytes"], label=f"{label}[{index}].size_bytes", minimum=1)
        wrappers.append(dict(item))
    return wrappers


def _validate_install_inventory(
    value: Any,
    *,
    install_venv: Path,
    include_project: bool,
    label: str,
) -> dict[str, Any]:
    required = {
        "distribution_inventory",
        "pip_find_spec_present",
        "pip_package_paths",
        "pip_wrappers",
        "plain_pip_absent",
        "site_packages_root",
    }
    if include_project:
        required |= {"installed_dist_info", "installed_package_root"}
    proof = _require_exact_keys(value, required, label=label)
    expected_inventory = [{"name": "pip", "version": EXPECTED_PIP_VERSION}]
    if include_project:
        expected_inventory.append({"name": "quant-investor", "version": "17.0.0"})
    expected_inventory.sort(key=lambda item: item["name"].casefold())
    if proof["distribution_inventory"] != expected_inventory:
        raise PackageEvidenceError(f"{label} distribution inventory mismatch")
    site = Path(str(proof["site_packages_root"]))
    if not site.is_absolute() or not _path_within(site, install_venv):
        raise PackageEvidenceError(f"{label} site-packages root is outside install environment")
    package_paths = proof["pip_package_paths"]
    if type(package_paths) is not list or {Path(str(item)).name for item in package_paths} != {
        "pip",
        f"pip-{EXPECTED_PIP_VERSION}.dist-info",
    }:
        raise PackageEvidenceError(f"{label} pip package paths mismatch")
    if any(
        not Path(str(item)).is_absolute() or Path(str(item)).parent != site
        for item in package_paths
    ):
        raise PackageEvidenceError(f"{label} pip package path escapes site-packages")
    wrappers = _validate_pip_wrappers(
        proof["pip_wrappers"],
        install_venv=install_venv,
        label=f"{label}.pip_wrappers",
    )
    if proof["pip_find_spec_present"] is not True or proof["plain_pip_absent"] is not True:
        raise PackageEvidenceError(f"{label} pip module/wrapper scope mismatch")
    result = dict(proof)
    result["pip_wrappers"] = wrappers
    if include_project:
        package_root = Path(str(proof["installed_package_root"]))
        dist_info = Path(str(proof["installed_dist_info"]))
        if (
            not package_root.is_absolute()
            or package_root.parent != site
            or not dist_info.is_absolute()
            or dist_info.parent != site
        ):
            raise PackageEvidenceError(f"{label} installed project paths mismatch")
    return result


def _validate_selector_modules(value: Any, *, label: str) -> dict[str, Any]:
    expected = {
        "hatchling.build",
        "hatchling.builders.sdist",
        "hatchling.builders.wheel",
        "package_parity",
    }
    modules = _require_exact_keys(value, expected, label=label)
    result: dict[str, Any] = {}
    for name in sorted(expected):
        binding = _require_exact_keys(
            modules[name],
            {"path", "sha256", "size_bytes"},
            label=f"{label}.{name}",
        )
        if type(binding["path"]) is not str or not Path(binding["path"]).is_absolute():
            raise PackageEvidenceError(f"{label}.{name}.path must be absolute")
        _require_sha256(binding["sha256"], label=f"{label}.{name}.sha256")
        _require_int(binding["size_bytes"], label=f"{label}.{name}.size_bytes", minimum=1)
        result[name] = dict(binding)
    return result


def _validate_hatch_namespace(value: Any, *, label: str) -> dict[str, Any]:
    namespace = _require_exact_keys(
        value,
        {"row_count", "rows", "sha256", "wheel_projection_sha256"},
        label=label,
    )
    rows = namespace["rows"]
    if type(rows) is not list or not rows:
        raise PackageEvidenceError(f"{label}.rows must be nonempty")
    normalized: list[dict[str, Any]] = []
    previous: tuple[str, str, str] | None = None
    collision_keys: set[tuple[str, str]] = set()
    for index, raw in enumerate(rows):
        row = _require_exact_keys(
            raw,
            {"distribution_path", "mode", "sha256", "size_bytes", "source_path", "target"},
            label=f"{label}.rows[{index}]",
        )
        if row["target"] not in {"sdist", "wheel"}:
            raise PackageEvidenceError(f"{label}.rows[{index}].target is invalid")
        for key in ("source_path", "distribution_path"):
            path = row[key]
            if type(path) is not str:
                raise PackageEvidenceError(f"{label}.rows[{index}].{key} must be a string")
            pure = PurePosixPath(path)
            if (
                not path
                or pure.is_absolute()
                or path != pure.as_posix()
                or any(part in {"", ".", ".."} for part in pure.parts)
            ):
                raise PackageEvidenceError(f"{label}.rows[{index}].{key} is unsafe")
        if type(row["mode"]) is not int or not 0 <= row["mode"] <= 0o7777:
            raise PackageEvidenceError(f"{label}.rows[{index}].mode is invalid")
        _require_sha256(row["sha256"], label=f"{label}.rows[{index}].sha256")
        _require_int(row["size_bytes"], label=f"{label}.rows[{index}].size_bytes")
        collision = (row["target"], row["distribution_path"].casefold())
        if collision in collision_keys:
            raise PackageEvidenceError(f"{label} contains a distribution-path collision")
        collision_keys.add(collision)
        sort_key = (row["target"], row["distribution_path"], row["source_path"])
        if previous is not None and sort_key <= previous:
            raise PackageEvidenceError(f"{label}.rows are not canonical")
        previous = sort_key
        normalized.append(dict(row))
    if namespace["row_count"] != len(normalized):
        raise PackageEvidenceError(f"{label}.row_count mismatch")
    digest = _sha256(_canonical_bytes(normalized))
    if namespace["sha256"] != digest:
        raise PackageEvidenceError(f"{label}.sha256 mismatch")
    _require_sha256(
        namespace["wheel_projection_sha256"],
        label=f"{label}.wheel_projection_sha256",
    )
    wheel_sources = {row["source_path"] for row in normalized if row["target"] == "wheel"}
    sdist_sources = {row["source_path"] for row in normalized if row["target"] == "sdist"}
    if not wheel_sources or sdist_sources != wheel_sources | set(HATCH_EXTRA_PATHS):
        raise PackageEvidenceError(f"{label} target source sets mismatch")
    if any(
        row["distribution_path"] != row["source_path"]
        for row in normalized
        if row["target"] in {"sdist", "wheel"}
    ):
        raise PackageEvidenceError(f"{label} unexpected Hatch path rewrite")
    return {
        "row_count": len(normalized),
        "rows": normalized,
        "sha256": digest,
        "wheel_projection_sha256": namespace["wheel_projection_sha256"],
    }


def _validate_selector_probe(value: Any, *, label: str) -> dict[str, Any]:
    probe = _require_exact_keys(
        value,
        {
            "hatch_source_namespace",
            "hatchling_version",
            "package_source_superset",
            "selector_modules",
        },
        label=label,
    )
    if probe["hatchling_version"] != EXPECTED_BUILD_BACKEND_PACKAGES["hatchling"]:
        raise PackageEvidenceError(f"{label} Hatchling version mismatch")
    superset = _validate_physical_superset(
        probe["package_source_superset"],
        label=f"{label}.package_source_superset",
    )
    namespace = _validate_hatch_namespace(
        probe["hatch_source_namespace"],
        label=f"{label}.hatch_source_namespace",
    )
    physical = {row["path"]: row for row in superset["rows"] if row["kind"] == "file"}
    for index, row in enumerate(namespace["rows"]):
        source = physical.get(row["source_path"])
        if (
            source is None
            or row["sha256"] != source["sha256"]
            or row["size_bytes"] != source["size_bytes"]
            or row["mode"] != source["mode"]
        ):
            raise PackageEvidenceError(
                f"{label}.hatch_source_namespace.rows[{index}] physical projection mismatch"
            )
    return {
        "hatch_source_namespace": namespace,
        "hatchling_version": probe["hatchling_version"],
        "package_source_superset": superset,
        "selector_modules": _validate_selector_modules(
            probe["selector_modules"],
            label=f"{label}.selector_modules",
        ),
    }


def _single_artifact(directory: Path, suffix: str, *, label: str) -> Path:
    matches = sorted(directory.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise PackageEvidenceError(f"{label} must produce exactly one {suffix} artifact")
    return matches[0].resolve(strict=True)


def _validate_parity_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("accepted") is not True:
        raise PackageEvidenceError("package parity command did not accept the artifact")
    required = {
        "package_file_count",
        "package_inventory",
        "package_inventory_sha256",
        "sdist_sha256",
        "source_equals_sdist_equals_wheel_equals_installed",
        "wheel_sha256",
    }
    if any(key not in value for key in required):
        raise PackageEvidenceError("package parity output is missing required fields")
    if value.get("source_equals_sdist_equals_wheel_equals_installed") is not True:
        raise PackageEvidenceError("package payload parity failed")
    return dict(value)


def build_package_evidence(
    *,
    repo_root: Path,
    expected_base_commit: str,
    session_manifest: Path,
    expected_source_binding_json: Path,
    base_python: Path,
    uv_bin: Path,
    uv_cache: Path,
    work_root: Path,
) -> dict[str, Any]:
    if COMMIT_RE.fullmatch(expected_base_commit) is None:
        raise PackageEvidenceError("expected base commit is invalid")
    repo = _resolve_repo_root(repo_root)
    _require_absolute_path(session_manifest, label="session manifest")
    _require_absolute_path(expected_source_binding_json, label="expected source binding")
    _require_absolute_path(base_python, label="base Python")
    _require_absolute_path(uv_bin, label="uv binary")
    _require_absolute_path(uv_cache, label="uv cache")
    _require_absolute_path(work_root, label="work root")
    session, session_binding, session_raw, session_path = _load_session_manifest(
        session_manifest,
        repo_root=repo,
        expected_base_commit=expected_base_commit,
    )
    expected_binding_path, expected_binding_raw = _read_private_external_file(
        expected_source_binding_json,
        repo_root=repo,
        label="expected source binding",
    )
    expected_binding_artifact = {
        "path": str(expected_binding_path),
        "sha256": _sha256(expected_binding_raw),
        "size_bytes": len(expected_binding_raw),
    }
    expected_binding = _load_expected_source_binding(expected_binding_raw)
    if expected_binding["base_commit"] != expected_base_commit:
        raise PackageEvidenceError("expected base commit and source binding disagree")
    if session["source_binding"] != expected_binding:
        raise PackageEvidenceError("session and expected source binding disagree")
    source_binding_before = _sample_source_binding(repo, expected_base_commit)
    if source_binding_before != expected_binding:
        raise PackageEvidenceError("repository source binding differs before package build")
    protected_roots_before = _sample_protected_roots(repo)
    if protected_roots_before != session["protected_roots"]:
        raise PackageEvidenceError("protected roots differ before package build")
    physical_superset_initial = _sample_physical_superset(repo)
    physical_binding_initial = {
        "row_count": physical_superset_initial["row_count"],
        "sha256": physical_superset_initial["sha256"],
    }
    if physical_binding_initial != session["package_source_superset"]:
        raise PackageEvidenceError("package source superset differs before package build")
    producer_before = {
        **_file_binding(Path(__file__).resolve(), label="package evidence producer"),
        "version": PACKAGE_PRODUCER_VERSION,
    }

    python_lexical = base_python.absolute()
    uv_lexical = uv_bin.absolute()
    python = _assert_existing_executable(base_python, label="base Python")
    uv = _assert_existing_executable(uv_bin, label="uv binary")
    cache = _assert_existing_uv_cache(uv_cache, label="uv cache")
    cache_binding_before = _uv_cache_binding(cache)
    if session["uv_cache_binding"] != cache_binding_before:
        raise PackageEvidenceError("session uv cache binding mismatch")
    root = _private_fresh_work_root(work_root, repo_root=repo)
    build_venv = root / "build-venv"
    install_venv = root / "install-venv"
    sdist_dir = root / "sdist"
    wheel_dir = root / "wheel"
    sdist_dir.mkdir(mode=0o700)
    wheel_dir.mkdir(mode=0o700)

    commands: list[dict[str, Any]] = []
    python_binding_before = _file_binding(python, label="base Python")
    uv_binding_before = _file_binding(uv, label="uv binary")
    python_probe = _parse_json_stdout(
        _run_command(
            [str(python), "-I", "-S", "-B", "-c", _python_probe_code()],
            role="base_python_probe",
            cwd=repo,
            env_overrides={},
            tool_version="python probe",
            commands=commands,
        ),
        label="base Python probe",
    )
    _require_cpython313(python_probe)
    python_binding_after_probe = _file_binding(python, label="base Python")
    if python_binding_after_probe != python_binding_before:
        raise PackageEvidenceError("base Python binary drifted during probe")
    if (
        python_probe.get("realpath") != str(python)
        or not isinstance(python_probe.get("executable"), str)
        or not Path(python_probe["executable"]).is_absolute()
        or Path(python_probe["executable"]).resolve(strict=True) != python
        or python_probe.get("sha256") != python_binding_before["sha256"]
    ):
        raise PackageEvidenceError("base Python probe does not match the executed binary")
    python_tool_version = f"CPython {python_probe['version']}"
    uv_version_completed = _run_command(
        [str(uv), "--version"],
        role="uv_version",
        cwd=repo,
        env_overrides={"UV_NO_CONFIG": "1", "UV_OFFLINE": "1"},
        tool_version="uv version probe",
        commands=commands,
    )
    uv_runtime = _parse_uv_version(uv_version_completed.stdout)
    uv_tool_version = uv_runtime["output"]
    commands[-1]["tool_version"] = uv_tool_version
    uv_binding_after_version = _file_binding(uv, label="uv binary")
    if uv_binding_after_version != uv_binding_before:
        raise PackageEvidenceError("uv binary drifted during version probe")
    offline_env = _selected_env(
        UV_CACHE_DIR=str(cache),
        UV_NO_CONFIG="1",
        UV_OFFLINE="1",
        UV_PYTHON_DOWNLOADS="never",
    )
    no_cache_offline_env = _selected_env(
        UV_CACHE_DIR=str(cache),
        UV_NO_CONFIG="1",
        UV_NO_CACHE="1",
        UV_OFFLINE="1",
        UV_PYTHON_DOWNLOADS="never",
    )
    pip_env = _selected_env(
        PIP_CONFIG_FILE="/dev/null",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        PIP_NO_CACHE_DIR="1",
        PIP_NO_INDEX="1",
    )

    _run_command(
        [
            str(uv),
            "venv",
            "--python",
            str(python),
            "--offline",
            "--no-python-downloads",
            "--no-project",
            "--no-index",
            "--no-config",
            "--color",
            "never",
            "--no-progress",
            str(build_venv),
        ],
        role="create_build_venv",
        cwd=repo,
        env_overrides=offline_env,
        tool_version=uv_tool_version,
        commands=commands,
    )
    build_python = build_venv / "bin" / "python"
    _run_command(
        [
            str(uv),
            "pip",
            "install",
            "--python",
            str(build_python),
            "--offline",
            "--no-python-downloads",
            "--no-config",
            "--color",
            "never",
            "--no-progress",
            *BUILD_BACKEND_REQUIREMENTS,
        ],
        role="install_build_backend",
        cwd=repo,
        env_overrides=offline_env,
        tool_version=uv_tool_version,
        commands=commands,
    )
    backend_probe = _validate_backend_probe(
        _parse_json_stdout(
            _run_command(
                [str(build_python), "-I", "-c", _backend_probe_code()],
                role="build_backend_probe",
                cwd=repo,
                env_overrides={},
                tool_version=python_tool_version,
                commands=commands,
            ),
            label="build backend probe",
        ),
        build_venv=build_venv,
    )
    selector_before = _validate_selector_probe(
        _parse_json_stdout(
            _run_command(
                [
                    str(build_python),
                    "-I",
                    "-c",
                    _hatch_selector_probe_code(),
                    str(repo),
                ],
                role="hatch_selector_before",
                cwd=repo,
                env_overrides={},
                tool_version=f"Hatchling {EXPECTED_BUILD_BACKEND_PACKAGES['hatchling']}",
                commands=commands,
            ),
            label="Hatch selector before build",
        ),
        label="Hatch selector before build",
    )
    if selector_before["package_source_superset"] != physical_superset_initial:
        raise PackageEvidenceError("Hatch selector physical superset differs before build")

    _run_command(
        [
            str(uv),
            "build",
            "--sdist",
            "--python",
            str(build_python),
            "--no-build-isolation",
            "--offline",
            "--no-python-downloads",
            "--no-sources",
            "--no-index",
            "--no-cache",
            "--no-config",
            "--color",
            "never",
            "--no-progress",
            "--out-dir",
            str(sdist_dir),
            str(repo),
        ],
        role="build_sdist",
        cwd=repo,
        env_overrides=no_cache_offline_env,
        tool_version=uv_tool_version,
        commands=commands,
    )
    sdist = _single_artifact(sdist_dir, ".tar.gz", label="sdist build")
    _run_command(
        [
            str(uv),
            "build",
            "--wheel",
            "--python",
            str(build_python),
            "--no-build-isolation",
            "--offline",
            "--no-python-downloads",
            "--no-sources",
            "--no-index",
            "--no-cache",
            "--no-config",
            "--color",
            "never",
            "--no-progress",
            "--out-dir",
            str(wheel_dir),
            str(sdist),
        ],
        role="build_wheel_from_sdist",
        cwd=repo,
        env_overrides=no_cache_offline_env,
        tool_version=uv_tool_version,
        commands=commands,
    )
    wheel = _single_artifact(wheel_dir, ".whl", label="wheel build")
    _run_command(
        [
            str(uv),
            "venv",
            "--python",
            str(python),
            "--offline",
            "--no-python-downloads",
            "--no-project",
            "--no-index",
            "--no-config",
            "--color",
            "never",
            "--no-progress",
            str(install_venv),
        ],
        role="create_install_venv",
        cwd=repo,
        env_overrides=offline_env,
        tool_version=uv_tool_version,
        commands=commands,
    )
    install_python = install_venv / "bin" / "python"
    bundle_before = _validate_bundle_probe(
        _parse_json_stdout(
            _run_command(
                [str(install_python), "-I", "-c", _ensurepip_bundle_probe_code()],
                role="ensurepip_bundle_before",
                cwd=repo,
                env_overrides=pip_env,
                tool_version=python_tool_version,
                commands=commands,
            ),
            label="ensurepip bundled wheel before",
        ),
        install_venv=install_venv,
        label="ensurepip bundled wheel before",
    )
    _run_command(
        [str(install_python), "-I", "-m", "ensurepip", "--upgrade"],
        role="ensurepip",
        cwd=repo,
        env_overrides=pip_env,
        tool_version=python_tool_version,
        commands=commands,
    )
    bundle_after = _validate_bundle_probe(
        _parse_json_stdout(
            _run_command(
                [str(install_python), "-I", "-c", _ensurepip_bundle_probe_code()],
                role="ensurepip_bundle_after",
                cwd=repo,
                env_overrides=pip_env,
                tool_version=python_tool_version,
                commands=commands,
            ),
            label="ensurepip bundled wheel after",
        ),
        install_venv=install_venv,
        label="ensurepip bundled wheel after",
    )
    if bundle_after != bundle_before:
        raise PackageEvidenceError("bundled pip wheel drifted across ensurepip")
    pip_version_completed = _run_command(
        [str(install_python), "-I", "-m", "pip", "--version"],
        role="pip_version",
        cwd=repo,
        env_overrides=pip_env,
        tool_version=python_tool_version,
        commands=commands,
    )
    pip_runtime = _parse_pip_version(
        pip_version_completed.stdout,
        install_venv=install_venv,
    )
    pip_tool_version = pip_runtime["output"]
    commands[-1]["tool_version"] = pip_tool_version
    install_inventory_before = _validate_install_inventory(
        _parse_json_stdout(
            _run_command(
                [str(install_python), "-I", "-c", _install_inventory_probe_code()],
                role="install_inventory_before_project",
                cwd=repo,
                env_overrides=pip_env,
                tool_version=pip_tool_version,
                commands=commands,
            ),
            label="install inventory before project",
        ),
        install_venv=install_venv,
        include_project=False,
        label="install inventory before project",
    )
    _run_command(
        [
            str(install_python),
            "-I",
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--no-compile",
            str(wheel),
        ],
        role="install_wheel_no_compile",
        cwd=repo,
        env_overrides={**pip_env, "PIP_NO_COMPILE": "1"},
        tool_version=pip_tool_version,
        commands=commands,
    )
    installed_paths = _validate_install_inventory(
        _parse_json_stdout(
            _run_command(
                [str(install_python), "-I", "-c", _installed_paths_probe_code()],
                role="installed_paths_probe",
                cwd=repo,
                env_overrides={},
                tool_version=python_tool_version,
                commands=commands,
            ),
            label="installed path probe",
        ),
        install_venv=install_venv,
        include_project=True,
        label="installed path probe",
    )
    if installed_paths["pip_wrappers"] != install_inventory_before["pip_wrappers"]:
        raise PackageEvidenceError("pip wrappers drifted after project installation")
    installed_package = Path(str(installed_paths.get("installed_package_root")))
    installed_dist_info = Path(str(installed_paths.get("installed_dist_info")))
    if str(installed_dist_info) in {"", "None"}:
        raise PackageEvidenceError("installed dist-info could not be located")
    selector_parity = _validate_selector_probe(
        _parse_json_stdout(
            _run_command(
                [
                    str(build_python),
                    "-I",
                    "-c",
                    _hatch_selector_probe_code(),
                    str(repo),
                ],
                role="hatch_selector_parity",
                cwd=repo,
                env_overrides={},
                tool_version=f"Hatchling {EXPECTED_BUILD_BACKEND_PACKAGES['hatchling']}",
                commands=commands,
            ),
            label="Hatch selector at parity",
        ),
        label="Hatch selector at parity",
    )
    parity_script = installed_package / "v17_v2_contract" / "package_parity.py"
    parity_completed = _run_command(
        [
            str(install_python),
            str(parity_script),
            "--source-package-root",
            str(repo / "quant_investor"),
            "--sdist",
            str(sdist),
            "--wheel",
            str(wheel),
            "--installed-package-root",
            str(installed_package),
            "--installed-dist-info",
            str(installed_dist_info),
            "--installed-environment-root",
            str(install_venv),
            "--expected-name",
            "quant-investor",
            "--expected-version",
            "17.0.0",
        ],
        role="package_parity",
        cwd=repo,
        env_overrides=pip_env,
        tool_version=python_tool_version,
        commands=commands,
    )
    parity = _validate_parity_payload(_parse_json_stdout(parity_completed, label="package parity"))
    selector_after = _validate_selector_probe(
        _parse_json_stdout(
            _run_command(
                [
                    str(build_python),
                    "-I",
                    "-c",
                    _hatch_selector_probe_code(),
                    str(repo),
                ],
                role="hatch_selector_after",
                cwd=repo,
                env_overrides={},
                tool_version=f"Hatchling {EXPECTED_BUILD_BACKEND_PACKAGES['hatchling']}",
                commands=commands,
            ),
            label="Hatch selector after parity",
        ),
        label="Hatch selector after parity",
    )
    if not (selector_before == selector_parity == selector_after):
        raise PackageEvidenceError("Hatch selector or physical source rows drifted")
    sdist_binding = _file_binding(sdist, label="sdist")
    wheel_binding = _file_binding(wheel, label="wheel")
    if (
        parity["sdist_sha256"] != sdist_binding["sha256"]
        or parity["wheel_sha256"] != wheel_binding["sha256"]
    ):
        raise PackageEvidenceError("package parity artifact hash mismatch")
    namespace = selector_after["hatch_source_namespace"]
    if (
        namespace["wheel_projection_sha256"] != parity["package_inventory_sha256"]
        or namespace["wheel_projection_sha256"] != parity["package_inventory"]["sha256"]
    ):
        raise PackageEvidenceError("Hatch wheel projection differs from package parity")

    source_binding_after = _sample_source_binding(repo, expected_base_commit)
    if source_binding_after != expected_binding:
        raise PackageEvidenceError("repository source binding drifted during package build")
    expected_binding_path_after, expected_binding_raw_after = _read_private_external_file(
        expected_source_binding_json,
        repo_root=repo,
        label="expected source binding",
    )
    if (
        expected_binding_path_after != expected_binding_path
        or expected_binding_raw_after != expected_binding_raw
    ):
        raise PackageEvidenceError("expected source binding drifted during package build")
    uv_binding_after = _file_binding(uv, label="uv binary")
    if uv_binding_after != uv_binding_before:
        raise PackageEvidenceError("uv binary drifted during package build")
    python_binding_after = _file_binding(python, label="base Python")
    if python_binding_after != python_binding_before:
        raise PackageEvidenceError("base Python binary drifted during package build")
    cache_binding_after = _uv_cache_binding(cache)
    if cache_binding_after != cache_binding_before:
        raise PackageEvidenceError("uv cache identity drifted during package build")
    protected_roots_after = _sample_protected_roots(repo)
    if (
        protected_roots_after != protected_roots_before
        or protected_roots_after != session["protected_roots"]
    ):
        raise PackageEvidenceError("protected roots drifted during package build")
    physical_superset_final = _sample_physical_superset(repo)
    if (
        physical_superset_final != physical_superset_initial
        or physical_superset_final != selector_after["package_source_superset"]
    ):
        raise PackageEvidenceError("package physical source superset drifted")
    session_path_after, session_raw_after = _read_private_external_file(
        session_manifest,
        repo_root=repo,
        label="session manifest",
    )
    if session_path_after != session_path or session_raw_after != session_raw:
        raise PackageEvidenceError("session manifest drifted during package build")
    producer_after = {
        **_file_binding(Path(__file__).resolve(), label="package evidence producer"),
        "version": PACKAGE_PRODUCER_VERSION,
    }
    if producer_after != producer_before:
        raise PackageEvidenceError("package evidence producer drifted during package build")

    command_output_hash = _sha256(_canonical_bytes(commands))
    observed_roles = tuple(command["role"] for command in commands)
    if observed_roles != COMMAND_ROLES:
        raise PackageEvidenceError("package evidence command role order mismatch")
    base_python_toolchain = _executable_binding(
        python_lexical,
        python,
        python_binding_after,
        implementation="cpython",
        version=EXPECTED_PYTHON_VERSION,
        version_info=list(EXPECTED_PYTHON_VERSION_INFO),
    )
    uv_toolchain = _executable_binding(
        uv_lexical,
        uv,
        uv_binding_after,
        version=EXPECTED_UV_VERSION,
        output=EXPECTED_UV_VERSION_OUTPUT,
    )
    pip_scope = {
        "allowed_wrappers": list(EXPECTED_PIP_WRAPPERS),
        "build_pip_absent": True,
        "bundled_wheel": {
            "name": EXPECTED_PIP_WHEEL_NAME,
            "sha256": EXPECTED_PIP_WHEEL_SHA256,
            "size_bytes": EXPECTED_PIP_WHEEL_SIZE,
        },
        "ensurepip_argv_suffix": ["-I", "-m", "ensurepip", "--upgrade"],
        "environment_scope": "PACKAGE_INSTALL_ENV_ONLY",
        "native_pip_absent": True,
        "plain_pip_absent": True,
        "version": EXPECTED_PIP_VERSION,
    }
    toolchain_binding = {
        "base_python": base_python_toolchain,
        "pip_scope": pip_scope,
        "uv": uv_toolchain,
        "uv_cache": cache_binding_after,
    }
    if session["toolchain_binding"] != toolchain_binding:
        raise PackageEvidenceError("live package toolchain differs from session binding")
    if session["uv_cache_binding"] != toolchain_binding["uv_cache"]:
        raise PackageEvidenceError("session uv cache cross-binding mismatch")
    build_python_resolved = build_python.resolve(strict=True)
    build_python_file = _file_binding(build_python_resolved, label="retained build Python")
    selector_binding = {
        "app_target_importability_verified": False,
        "build_environment": str(build_venv),
        "build_python": _executable_binding(
            build_python,
            build_python_resolved,
            build_python_file,
            implementation="cpython",
            version=EXPECTED_PYTHON_VERSION,
            version_info=list(EXPECTED_PYTHON_VERSION_INFO),
        ),
        "hatchling_version": EXPECTED_BUILD_BACKEND_PACKAGES["hatchling"],
        "probe_code_sha256": _sha256(_hatch_selector_probe_code().encode("utf-8")),
        "repo_root": str(repo),
        "retained_for_external_revalidation": True,
        "selector_modules": selector_after["selector_modules"],
        "targets": ["sdist", "wheel"],
    }
    physical_session = {
        "after_rows": selector_after["package_source_superset"]["rows"],
        "before_rows": selector_before["package_source_superset"]["rows"],
        "parity_rows": selector_parity["package_source_superset"]["rows"],
        "row_count": selector_after["package_source_superset"]["row_count"],
        "sha256": selector_after["package_source_superset"]["sha256"],
    }
    hatch_session = {
        "after_rows": selector_after["hatch_source_namespace"]["rows"],
        "before_rows": selector_before["hatch_source_namespace"]["rows"],
        "parity_rows": selector_parity["hatch_source_namespace"]["rows"],
        "row_count": namespace["row_count"],
        "selector_binding": selector_binding,
        "sha256": namespace["sha256"],
        "wheel_projection_sha256": namespace["wheel_projection_sha256"],
    }
    provenance = {
        "artifact_bindings": {
            "sdist": sdist_binding,
            "wheel": wheel_binding,
        },
        "artifact_install_projection": {
            "hatch_wheel_projection_sha256": namespace["wheel_projection_sha256"],
            "installed_record_sha256": parity["installed_provenance"]["record"]["record_sha256"],
            "package_inventory_sha256": parity["package_inventory_sha256"],
            "sdist_sha256": parity["sdist_sha256"],
            "source_equals_sdist_equals_wheel_equals_installed": True,
            "wheel_sha256": parity["wheel_sha256"],
        },
        "base_interpreter": python_probe,
        "base_interpreter_binary": {
            "binary_after": python_binding_after,
            "binary_after_probe": python_binding_after_probe,
            "binary_before": python_binding_before,
        },
        "build_backend": backend_probe,
        "command_count": len(commands),
        "command_roles": list(COMMAND_ROLES),
        "commands": commands,
        "combined_output_sha256": command_output_hash,
        "environment": {
            "build_venv": str(build_venv),
            "install_venv": str(install_venv),
            "uv_binary": str(uv),
            "uv_cache": str(cache),
            "work_root": str(root),
        },
        "hatch_source_namespace_session": hatch_session,
        "install_environment": {
            "bundled_pip_after": bundle_after,
            "bundled_pip_before": bundle_before,
            "inventory_after_project": installed_paths,
            "inventory_before_project": install_inventory_before,
        },
        "offline_only": True,
        "network_actions_performed": False,
        "package_source_superset_session": physical_session,
        "role": "package_parity",
        "source_binding_artifact": expected_binding_artifact,
        "pip_runtime": pip_runtime,
        "source_binding_after": source_binding_after,
        "source_binding_before": source_binding_before,
        "uv_runtime": {
            **uv_runtime,
            "binary_after": uv_binding_after,
            "binary_after_version": uv_binding_after_version,
            "binary_before": uv_binding_before,
        },
    }
    payload = {
        **parity,
        "accepted": True,
        "authority": False,
        "build_install_provenance": provenance,
        "hatch_source_namespace": {
            "row_count": namespace["row_count"],
            "sha256": namespace["sha256"],
        },
        "limitations": list(LIMITATIONS),
        "network_actions_performed": False,
        "offline_only": True,
        "package_source_superset": physical_binding_initial,
        "phase0_gate_roles": list(EXPECTED_GATE_ROLES),
        "producer": producer_after,
        "protected_roots_after": protected_roots_after,
        "protected_roots_before": protected_roots_before,
        "protocol_version": PROTOCOL_VERSION,
        "session_binding": session_binding,
        "source_binding": expected_binding,
        "status": "SEALED",
        "step": dict(STEP),
        "toolchain_binding": toolchain_binding,
        "version": PACKAGE_EVIDENCE_VERSION,
    }
    sealed = _seal(payload)
    _validate_checked_schema(
        sealed,
        repo_root=repo,
        schema_relative_path=PACKAGE_EVIDENCE_SCHEMA_PATH,
        schema_id=PACKAGE_EVIDENCE_SCHEMA_ID,
        artifact_version=PACKAGE_EVIDENCE_VERSION,
    )
    return sealed


def _write_exact_once(
    path: Path,
    raw: bytes,
    *,
    repo_root: Path,
    expected_parent_identity: tuple[int, int, int, int] | None = None,
) -> None:
    target = _validate_output_target(path, repo_root=repo_root)
    try:
        parent_before = target.parent.lstat()
    except OSError as exc:
        raise PackageEvidenceError("output JSON parent is unavailable before write") from exc
    if (
        not stat.S_ISDIR(parent_before.st_mode)
        or stat.S_IMODE(parent_before.st_mode) != 0o700
        or parent_before.st_uid != os.getuid()
    ):
        raise PackageEvidenceError("output JSON parent must remain owner-private 0700")
    if (
        expected_parent_identity is not None
        and _directory_identity(parent_before) != expected_parent_identity
    ):
        raise PackageEvidenceError("output JSON parent changed since initial authorization")
    parent_descriptor = -1
    descriptor = -1
    created_identity: tuple[int, int] | None = None

    def cleanup_created_file() -> None:
        if parent_descriptor < 0 or created_identity is None:
            return
        try:
            current = os.stat(
                target.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (current.st_dev, current.st_ino) != created_identity:
                return
            os.unlink(target.name, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise PackageEvidenceError("cannot safely clean failed exact-once output") from exc

    try:
        parent_descriptor = os.open(
            target.parent,
            os.O_RDONLY
            | _required_open_flag("O_CLOEXEC")
            | _required_open_flag("O_DIRECTORY")
            | _required_open_flag("O_NOFOLLOW"),
        )
        parent_fd_before = os.fstat(parent_descriptor)
        parent_path_after_open = target.parent.lstat()
        if {
            _directory_identity(parent_before),
            _directory_identity(parent_fd_before),
            _directory_identity(parent_path_after_open),
        } != {_directory_identity(parent_before)}:
            raise PackageEvidenceError("output JSON parent changed before create")
        descriptor = os.open(
            target.name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | _required_open_flag("O_CLOEXEC")
            | _required_open_flag("O_NOFOLLOW"),
            0o600,
            dir_fd=parent_descriptor,
        )
        created = os.fstat(descriptor)
        created_identity = (created.st_dev, created.st_ino)
        if (
            not stat.S_ISREG(created.st_mode)
            or stat.S_IMODE(created.st_mode) != 0o600
            or created.st_uid != os.getuid()
            or created.st_nlink != 1
        ):
            raise PackageEvidenceError("created output JSON identity is unsafe")
        view = memoryview(raw)
        total = 0
        while total < len(raw):
            written = os.write(descriptor, view[total:])
            if written <= 0:
                raise PackageEvidenceError("short write while writing output JSON")
            total += written
        os.fsync(descriptor)
        created_after = os.fstat(descriptor)
        path_after = os.stat(
            target.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        parent_after = os.fstat(parent_descriptor)
        parent_path_after_write = target.parent.lstat()
        if {
            _file_object_identity(created),
            _file_object_identity(created_after),
            _file_object_identity(path_after),
        } != {_file_object_identity(created)}:
            raise PackageEvidenceError("output JSON changed during exact-once write")
        if created_after.st_size != len(raw) or path_after.st_size != len(raw):
            raise PackageEvidenceError("output JSON size differs after exact-once write")
        os.lseek(descriptor, 0, os.SEEK_SET)
        readback_chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            readback_chunks.append(chunk)
        if b"".join(readback_chunks) != raw:
            raise PackageEvidenceError("output JSON readback differs after exact-once write")
        if {
            _directory_identity(parent_before),
            _directory_identity(parent_after),
            _directory_identity(parent_path_after_write),
        } != {_directory_identity(parent_before)}:
            raise PackageEvidenceError("output JSON parent changed during exact-once write")
        os.fsync(parent_descriptor)
        parent_path_after_fsync = target.parent.lstat()
        if _directory_identity(parent_path_after_fsync) != _directory_identity(parent_before):
            raise PackageEvidenceError("output JSON parent changed after exact-once fsync")
    except FileExistsError as exc:
        raise PackageEvidenceError("output JSON already exists") from exc
    except PackageEvidenceError:
        cleanup_created_file()
        raise
    except OSError as exc:
        cleanup_created_file()
        raise PackageEvidenceError("cannot write output JSON exact-once") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-base-commit", required=True)
    parser.add_argument("--session-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-binding-json", type=Path, required=True)
    parser.add_argument("--base-python", type=Path, required=True)
    parser.add_argument("--uv-bin", type=Path, required=True)
    parser.add_argument("--uv-cache", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        repo = _resolve_repo_root(args.repo_root)
        output = _validate_output_target(args.output_json, repo_root=repo)
        try:
            output_parent_identity = _directory_identity(output.parent.lstat())
        except OSError as exc:
            raise PackageEvidenceError("output JSON parent disappeared after validation") from exc
        if output.exists():
            raise PackageEvidenceError("output JSON already exists")
        evidence = build_package_evidence(
            repo_root=repo,
            expected_base_commit=args.expected_base_commit,
            session_manifest=args.session_manifest,
            expected_source_binding_json=args.expected_source_binding_json,
            base_python=args.base_python,
            uv_bin=args.uv_bin,
            uv_cache=args.uv_cache,
            work_root=args.work_root,
        )
        _write_exact_once(
            output,
            _canonical_resource_bytes(evidence),
            repo_root=repo,
            expected_parent_identity=output_parent_identity,
        )
    except PackageEvidenceError as exc:
        print(f"v17_phase0_package_evidence: {exc}", file=sys.stderr)
        return PackageEvidenceError.exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
