#!/usr/bin/env python3
"""Build v17 dependency evidence from one authoritative native-sync v2 log.

The script never installs, resolves, downloads, bootstraps pip, copies cache
entries, or repackages a distribution. It validates the canonical command-v2
receipt and binary frames in ``10_native_sync.log`` before reconciling the exact
lock, frozen export, local project, installed environment, and retained
wheelhouse. A wheelhouse can be incomplete; it can never replace the successful
native offline sync.

This remains a third-party environment gate. It does not prove local
``quant-investor`` wheel/sdist provenance or grant runtime, CLI, release,
cutover, broker, order, trade, purge, or v16-mutation authority.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlsplit

try:
    import tomllib
except ModuleNotFoundError as exc:  # pragma: no cover - release Python is 3.13
    raise RuntimeError("Python 3.11+ is required to read frozen TOML inputs") from exc

try:
    from packaging.requirements import Requirement
    from packaging.tags import Tag
    from packaging.utils import canonicalize_name, parse_wheel_filename
    from packaging.version import Version
except ModuleNotFoundError as exc:  # pragma: no cover - packaging is lock-bound
    raise RuntimeError("the lock-bound 'packaging' distribution is required") from exc


SCHEMA_VERSION = "v17_third_party_dependency_environment_evidence.v2"
COMMAND_RECEIPT_VERSION = "myquant.v17.v2.phase0-command-receipt.v2"
PROTOCOL_VERSION = "myquant.v17.v2"
COMMAND_RECEIPT_PREFIX = b"MYQUANT_PHASE0_COMMAND_RECEIPT="
COMMAND_FRAMING = "UINT64_BE_STDOUT_THEN_STDOUT_UINT64_BE_STDERR_THEN_STDERR_PER_COMMAND"
BASE_PYTHON_PATH = (
    "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/"
    "Python.framework/Versions/3.13/bin/python3.13"
)
BASE_PYTHON_SHA256 = "a708f6e9f4803b806b29146c4e0feecfd9bf2d9eb60f3e15b850cd7cb56f200b"
BASE_PYTHON_SIZE = 52_640
UV_VERSION_OUTPUT = "uv 0.10.9 (f675560f3 2026-03-06)"
UV_SHA256 = "bc50ab0e90f24491f0e794f5b8649722f8fd2bf483c53490c012b41b89151ef9"
UV_SIZE = 44_698_848
UV_PATH = "/Users/maxwell/.local/bin/uv"
UV_CACHE_PATH = "/Users/maxwell/.cache/uv"
PIP_SCOPE = {
    "version": "25.2",
    "environment_scope": "PACKAGE_INSTALL_ENV_ONLY",
    "bundled_wheel": {
        "name": "pip-25.2-py3-none-any.whl",
        "size_bytes": 1_752_557,
        "sha256": "690972885fc9270380d1bb28212cafdff6a96e0b6e04396b9fa7505253591e11",
    },
    "ensurepip_argv_suffix": ["-I", "-m", "ensurepip", "--upgrade"],
    "allowed_wrappers": ["pip3", "pip3.13"],
    "plain_pip_absent": True,
    "native_pip_absent": True,
    "build_pip_absent": True,
}
NORMATIVE_LIMITATIONS = [
    "PRECOLLECTION_ALLOWED_TREES_ATTESTED_NOT_COMPLETE_MAIN_RUNTIME_FILESET",
    "NO_DIRECT_LAUNCH_REFERENCE_ONLY_NOT_PROCESS_IO_ATTESTED",
    "OFFLINE_FLAGS_ONLY_NOT_OS_EGRESS_ATTESTED",
    "OWNER_DECLARED_PATHS_ONLY_NOT_CONTENT_PROVENANCE",
    "PIP_25_2_PACKAGE_INSTALL_ENV_ONLY_NATIVE_AND_BUILD_ENV_PIP_ABSENT",
]
PROTECTED_ROOT_SPECS = [
    ("authority_v16", "/Users/maxwell/mySpace/myQuant/results/v16"),
    (
        "authority_v16_operator_advisory",
        "/Users/maxwell/mySpace/myQuant/results/v16_operator_advisory",
    ),
    (
        "candidate_v16",
        "/private/tmp/myquant-v17-neutral-baseline-20260722/results/v16",
    ),
    (
        "candidate_v16_operator_advisory",
        ("/private/tmp/myquant-v17-neutral-baseline-20260722/" "results/v16_operator_advisory"),
    ),
]
RECEIPT_ROOT_KEYS = {
    "version",
    "protocol_version",
    "session_binding",
    "step",
    "producer",
    "source_before",
    "source_after",
    "toolchain_before",
    "toolchain_after",
    "package_source_superset_before",
    "package_source_superset_after",
    "protected_roots_before",
    "protected_roots_after",
    "limitations",
    "outcome",
    "failure_codes",
    "commands",
    "claims",
    "framing",
    "output_sha256",
    "output_size_bytes",
    "semantic_sha256",
}
COMMAND_KEYS = {
    "ordinal",
    "argv",
    "cwd",
    "environment",
    "exit_code",
    "signal",
    "tool_version",
    "stdout_offset_bytes",
    "stdout_size_bytes",
    "stdout_sha256",
    "stderr_offset_bytes",
    "stderr_size_bytes",
    "stderr_sha256",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPORT_PIN_RE = re.compile(r"^[A-Za-z0-9_.-]+(?:\[[^\]]+\])?==")
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class EvidenceError(RuntimeError):
    """Raised when evidence inputs are malformed, ambiguous, or drifting."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _expect_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise EvidenceError(f"{label} keys mismatch; missing={missing}, extra={extra}")


def _require_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{label} must be a non-empty string")
    return value


def _require_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvidenceError(f"{label} must be an integer >= {minimum}")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise EvidenceError(f"{label} must be a lowercase SHA-256")
    return value


def _stable_lstat(path: Path, *, label: str) -> os.stat_result:
    try:
        first = path.lstat()
        second = path.lstat()
    except OSError as exc:
        raise EvidenceError(f"cannot stat {label}: {path}") from exc
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
    )
    if identity(first) != identity(second):
        raise EvidenceError(f"{label} changed during stat: {path}")
    return first


def _stable_file_bytes(path: Path) -> tuple[bytes, os.stat_result]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise EvidenceError(f"cannot stat file: {path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise EvidenceError(f"expected regular file: {path}")
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            raw = handle.read()
            descriptor_stat = os.fstat(handle.fileno())
    except OSError as exc:
        raise EvidenceError(f"cannot read file: {path}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as exc:
        raise EvidenceError(f"file disappeared during read: {path}") from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_descriptor = (
        descriptor_stat.st_dev,
        descriptor_stat.st_ino,
        descriptor_stat.st_mode,
        descriptor_stat.st_size,
        descriptor_stat.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_descriptor or identity_before != identity_after:
        raise EvidenceError(f"file changed during read: {path}")
    if len(raw) != before.st_size:
        raise EvidenceError(f"file size drift during read: {path}")
    return raw, before


def _file_binding(path: Path) -> dict[str, Any]:
    raw, file_stat = _stable_file_bytes(path)
    return _file_binding_from_bytes(path, raw, file_stat)


def _file_binding_from_bytes(path: Path, raw: bytes, file_stat: os.stat_result) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256_bytes(raw),
        "size": len(raw),
        "mode": f"{stat.S_IMODE(file_stat.st_mode):04o}",
    }


def _executable_file_binding(path: Path, *, label: str) -> dict[str, Any]:
    raw, file_stat = _stable_file_bytes(path)
    mode = stat.S_IMODE(file_stat.st_mode)
    if mode != 0o755 or not os.access(path, os.X_OK):
        raise EvidenceError(f"{label} must be mode 0755 and executable")
    return {
        "path": str(path),
        "sha256": _sha256_bytes(raw),
        "size": len(raw),
        "mode": f"{mode:04o}",
        "executable": True,
    }


def _directory_identity(
    path: Path,
    *,
    label: str,
    owner_private: bool,
) -> dict[str, Any]:
    path = path.absolute()
    value = _stable_lstat(path, label=label)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise EvidenceError(f"cannot resolve {label}: {path}") from exc
    if resolved != path or stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise EvidenceError(f"{label} must be a real directory without symlink indirection")
    mode = stat.S_IMODE(value.st_mode)
    if value.st_uid != os.getuid() or (owner_private and mode != 0o700):
        qualifier = "owner-private 0700" if owner_private else "owner-owned"
        raise EvidenceError(f"{label} must be an {qualifier} directory")
    return {
        "path": str(path),
        "mode": f"{mode:04o}",
        "st_uid": value.st_uid,
        "st_dev": value.st_dev,
        "st_ino": value.st_ino,
    }


def _run_bytes(command: Sequence[str], *, cwd: Path) -> bytes:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except OSError as exc:
        raise EvidenceError(f"cannot execute command: {command[0]}") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise EvidenceError(
            f"command failed ({completed.returncode}): {' '.join(command)}: {stderr}"
        )
    return completed.stdout


def _decode_nul_paths(raw: bytes, *, label: str) -> list[str]:
    if raw and not raw.endswith(b"\0"):
        raise EvidenceError(f"{label} is not NUL terminated")
    values: list[str] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            value = item.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise EvidenceError(f"{label} contains a non-UTF-8 path") from exc
        if any("\ud800" <= character <= "\udfff" for character in value):
            raise EvidenceError(f"{label} contains a surrogate path")
        values.append(value)
    return values


def _safe_repo_path(repo_root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if not relative or pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise EvidenceError(f"unsafe repository path: {relative!r}")
    candidate = repo_root.joinpath(*pure.parts)
    try:
        resolved_parent = candidate.parent.resolve(strict=True)
        resolved_parent.relative_to(repo_root)
    except (OSError, ValueError) as exc:
        raise EvidenceError(f"repository path escapes root: {relative!r}") from exc
    return candidate


def _stable_untracked_entry(repo_root: Path, relative: str) -> dict[str, Any]:
    path = _safe_repo_path(repo_root, relative)
    try:
        before = path.lstat()
    except OSError as exc:
        raise EvidenceError(f"untracked path disappeared: {relative}") from exc
    base: dict[str, Any] = {
        "path": relative,
        "mode": f"{stat.S_IMODE(before.st_mode):04o}",
        "size": before.st_size,
    }
    if stat.S_ISREG(before.st_mode):
        raw, observed = _stable_file_bytes(path)
        if (before.st_dev, before.st_ino, before.st_mode) != (
            observed.st_dev,
            observed.st_ino,
            observed.st_mode,
        ):
            raise EvidenceError(f"untracked file identity drift: {relative}")
        base.update({"type": "file", "sha256": _sha256_bytes(raw)})
    elif stat.S_ISLNK(before.st_mode):
        try:
            target_before = os.readlink(path)
            after = path.lstat()
            target_after = os.readlink(path)
        except OSError as exc:
            raise EvidenceError(f"cannot read untracked symlink: {relative}") from exc
        if target_before != target_after or (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_mode, after.st_mtime_ns):
            raise EvidenceError(f"untracked symlink drift: {relative}")
        try:
            target_before.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise EvidenceError(f"untracked symlink target is not UTF-8: {relative}") from exc
        base.update({"type": "symlink", "target": target_before})
    else:
        raise EvidenceError(f"unsupported untracked path type: {relative}")
    return base


def _git_snapshot(repo_root: Path) -> dict[str, Any]:
    top = _run_bytes(("git", "rev-parse", "--show-toplevel"), cwd=repo_root)
    try:
        top_path = Path(top.decode("utf-8", errors="strict").strip()).resolve(strict=True)
    except (UnicodeDecodeError, OSError) as exc:
        raise EvidenceError("git returned an invalid top-level path") from exc
    if top_path != repo_root:
        raise EvidenceError(f"explicit repo root is not the git top level: {repo_root}")
    head_raw = _run_bytes(("git", "rev-parse", "--verify", "HEAD"), cwd=repo_root)
    try:
        head = head_raw.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise EvidenceError("git HEAD is not ASCII") from exc
    if not re.fullmatch(r"[0-9a-f]{40,64}", head):
        raise EvidenceError("git HEAD is not a full object id")
    status_raw = _run_bytes(
        ("git", "status", "--porcelain=v1", "-z", "--untracked-files=all"),
        cwd=repo_root,
    )
    diff_raw = _run_bytes(
        (
            "git",
            "diff",
            "--binary",
            "--no-ext-diff",
            "--no-textconv",
            "HEAD",
        ),
        cwd=repo_root,
    )
    untracked_raw = _run_bytes(
        ("git", "ls-files", "--others", "--exclude-standard", "-z"),
        cwd=repo_root,
    )
    untracked_paths = _decode_nul_paths(untracked_raw, label="git untracked inventory")
    if len(untracked_paths) != len(set(untracked_paths)):
        raise EvidenceError("git untracked inventory contains duplicate paths")
    untracked = [
        _stable_untracked_entry(repo_root, path)
        for path in sorted(untracked_paths, key=lambda value: value.encode("utf-8"))
    ]
    manifest = {
        "head": head,
        "porcelain_v1_z": {
            "sha256": _sha256_bytes(status_raw),
            "size": len(status_raw),
        },
        "binary_diff_from_head": {
            "sha256": _sha256_bytes(diff_raw),
            "size": len(diff_raw),
        },
        "untracked": untracked,
    }
    return {
        **manifest,
        "source_state_sha256": _sha256_bytes(_canonical_bytes(manifest)),
        "_raw_guard": {
            "head": head_raw,
            "status": status_raw,
            "diff": diff_raw,
            "untracked": untracked_raw,
        },
    }


def _public_source_state(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in snapshot.items() if key != "_raw_guard"}


def _command_source_binding(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    public = _public_source_state(snapshot)
    return {
        "base_commit": public["head"],
        "source_state_sha256": public["source_state_sha256"],
        "porcelain_sha256": public["porcelain_v1_z"]["sha256"],
        "binary_diff_sha256": public["binary_diff_from_head"]["sha256"],
        "untracked_inventory_sha256": _sha256_bytes(_canonical_bytes(public["untracked"])),
    }


def _canonical_command_source_binding(repo_root: Path) -> dict[str, Any]:
    module_path = repo_root / "scripts" / "v17_phase0_evidence_index.py"
    module_name = "_myquant_v17_dependency_canonical_source"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise EvidenceError("cannot load canonical Phase 0 source helper")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.modules[module_name] = module
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
        head_raw = _run_bytes(("git", "rev-parse", "--verify", "HEAD"), cwd=repo_root)
        try:
            base_commit = head_raw.decode("ascii", errors="strict").strip()
        except UnicodeError as exc:
            raise EvidenceError("canonical source HEAD is not ASCII") from exc
        snapshot = module._git_snapshot(repo_root, base_commit)
        public = module._public_source_state(snapshot)
        binding = module._source_binding_from_state(public)
    except EvidenceError:
        raise
    except Exception as exc:
        raise EvidenceError("cannot capture canonical Phase 0 source binding") from exc
    finally:
        sys.dont_write_bytecode = previous
        sys.modules.pop(module_name, None)
    return _validate_source_binding(binding, label="canonical Phase 0 source binding")


def _assert_source_state_stable(first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    if _canonical_bytes(_public_source_state(first)) != _canonical_bytes(
        _public_source_state(second)
    ):
        raise EvidenceError("repository source state changed during evidence collection")
    if first.get("_raw_guard") != second.get("_raw_guard"):
        raise EvidenceError("raw git evidence changed during evidence collection")


def _target_probe(target_python: Path) -> dict[str, Any]:
    probe = r"""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import sysconfig
from packaging.markers import default_environment
from packaging.tags import sys_tags

items = []
for dist in importlib.metadata.distributions():
    name = dist.metadata.get("Name")
    version = dist.metadata.get("Version")
    if not name or not version:
        continue
    raw = (dist.read_text("METADATA") or "").encode("utf-8")
    items.append({
        "name": name,
        "version": version,
        "metadata_sha256": hashlib.sha256(raw).hexdigest(),
        "metadata_size": len(raw),
    })
site_package_paths = sorted({
    value
    for key, value in sysconfig.get_paths().items()
    if key in {"purelib", "platlib"} and value
})
site_package_pip_matches = []
for root in site_package_paths:
    if not os.path.isdir(root):
        continue
    site_package_pip_matches.extend(
        os.path.join(root, name)
        for name in sorted(os.listdir(root))
        if name.casefold().startswith("pip")
    )
bin_root = os.path.dirname(sys.executable)
bin_pip_matches = [
    os.path.join(bin_root, name)
    for name in sorted(os.listdir(bin_root))
    if name.casefold().startswith("pip")
]
payload = {
    "python": {
        "executable": sys.executable,
        "version": sys.version,
        "version_info": list(sys.version_info[:3]),
        "implementation": sys.implementation.name,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
    },
    "platform": {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
    },
    "marker_environment": default_environment(),
    "supported_tags": [str(tag) for tag in sys_tags()],
    "installed": items,
    "pip_probe": {
        "find_spec_is_none": importlib.util.find_spec("pip") is None,
        "site_package_paths": site_package_paths,
        "site_package_matches": site_package_pip_matches,
        "bin_matches": bin_pip_matches,
    },
}
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
"""
    try:
        completed = subprocess.run(
            (str(target_python), "-I", "-c", probe),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
        )
    except OSError as exc:
        raise EvidenceError(f"cannot execute target Python: {target_python}") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise EvidenceError(f"target Python probe failed: {stderr}")
    try:
        payload = json.loads(completed.stdout.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("target Python returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise EvidenceError("target Python probe returned a non-object")
    return payload


def _validate_target_venv(
    target_venv: Path, target_python: Path, probe: Mapping[str, Any]
) -> dict[str, Any]:
    target_venv = target_venv.absolute()
    target_python = target_python.absolute()
    _directory_identity(target_venv, label="target venv", owner_private=False)
    expected_entrypoint = target_venv / "bin" / "python"
    if target_python != expected_entrypoint:
        raise EvidenceError("target Python must be the lexical fresh venv bin/python")
    entrypoint_stat = _stable_lstat(target_python, label="target Python entrypoint")
    if not (stat.S_ISREG(entrypoint_stat.st_mode) or stat.S_ISLNK(entrypoint_stat.st_mode)):
        raise EvidenceError("target Python entrypoint must be a regular file or symlink")
    try:
        resolved_python = target_python.resolve(strict=True)
    except OSError as exc:
        raise EvidenceError("target Python entrypoint cannot be resolved") from exc
    if str(resolved_python) != BASE_PYTHON_PATH:
        raise EvidenceError("target Python does not resolve to the frozen base CPython")
    base_binding = _executable_file_binding(resolved_python, label="frozen base CPython")
    expected_base = {
        "path": BASE_PYTHON_PATH,
        "sha256": BASE_PYTHON_SHA256,
        "size": BASE_PYTHON_SIZE,
        "mode": "0755",
        "executable": True,
    }
    if base_binding != expected_base:
        raise EvidenceError("frozen base CPython binary identity mismatch")
    python_info = probe.get("python")
    if not isinstance(python_info, dict):
        raise EvidenceError("target Python probe omitted Python identity")
    _expect_keys(
        python_info,
        {
            "executable",
            "version",
            "version_info",
            "implementation",
            "prefix",
            "base_prefix",
        },
        label="target Python probe identity",
    )
    if (
        python_info["implementation"] != "cpython"
        or python_info["version_info"] != [3, 13, 7]
        or Path(str(python_info["executable"])).absolute() != target_python
    ):
        raise EvidenceError("target Python must be exact CPython 3.13.7")
    try:
        prefix = Path(str(python_info["prefix"])).resolve(strict=True)
    except (KeyError, OSError) as exc:
        raise EvidenceError("target Python prefix is invalid") from exc
    if prefix != target_venv.resolve(strict=True):
        raise EvidenceError("target Python sys.prefix does not match explicit target venv")
    expected_base_prefix = Path(BASE_PYTHON_PATH).parents[1]
    try:
        base_prefix = Path(str(python_info["base_prefix"])).resolve(strict=True)
    except OSError as exc:
        raise EvidenceError("target Python base_prefix is invalid") from exc
    if base_prefix != expected_base_prefix:
        raise EvidenceError("target Python base_prefix does not match frozen CPython")
    pyvenv = target_venv / "pyvenv.cfg"
    binding = _file_binding(pyvenv)
    executable_target = os.readlink(target_python) if target_python.is_symlink() else None
    return {
        "path": str(target_venv),
        "pyvenv_cfg": binding,
        "python_entrypoint": str(target_python),
        "python_entrypoint_symlink_target": executable_target,
        "python_entrypoint_lstat": {
            "mode": f"{stat.S_IMODE(entrypoint_stat.st_mode):04o}",
            "st_dev": entrypoint_stat.st_dev,
            "st_ino": entrypoint_stat.st_ino,
        },
        "resolved_base_python": base_binding,
    }


def _uv_binding(uv_binary: str, *, cwd: Path) -> dict[str, Any]:
    resolved_text = shutil.which(uv_binary)
    if resolved_text is None:
        raise EvidenceError(f"uv binary not found: {uv_binary}")
    resolved = Path(resolved_text).resolve(strict=True)
    raw, file_stat = _stable_file_bytes(resolved)
    try:
        completed = subprocess.run(
            (str(resolved), "--version"),
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "UV_OFFLINE": "1"},
        )
    except OSError as exc:
        raise EvidenceError("cannot execute uv --version") from exc
    if completed.returncode != 0:
        raise EvidenceError("uv --version failed")
    try:
        version = completed.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise EvidenceError("uv --version output is not UTF-8") from exc
    binding = {
        "path": str(resolved),
        "version_output": version,
        "sha256": _sha256_bytes(raw),
        "size": len(raw),
        "mode": f"{stat.S_IMODE(file_stat.st_mode):04o}",
        "executable": bool(os.access(resolved, os.X_OK)),
    }
    if binding != {
        "path": UV_PATH,
        "version_output": UV_VERSION_OUTPUT,
        "sha256": UV_SHA256,
        "size": UV_SIZE,
        "mode": "0755",
        "executable": True,
    }:
        raise EvidenceError("uv binary identity does not match frozen uv 0.10.9")
    return binding


def _parse_frozen_export(
    raw: bytes, marker_environment: Mapping[str, str]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise EvidenceError("frozen export is not UTF-8") from exc
    header = "\n".join(text.splitlines()[:8])
    if "uv export" not in header or "--frozen" not in header or "--no-hashes" not in header:
        raise EvidenceError("export header must bind 'uv export --frozen --no-hashes'")
    if "--no-emit-project" not in header:
        raise EvidenceError("export must use --no-emit-project")
    if "--hash=" in text:
        raise EvidenceError("frozen no-hash export unexpectedly contains hashes")
    active: dict[str, dict[str, str]] = {}
    inactive: list[dict[str, str]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            raise EvidenceError("frozen no-hash export must not contain continuations")
        if line.startswith(("-", ".")) or not EXPORT_PIN_RE.match(line):
            raise EvidenceError(f"unsupported export line {line_number}: {line!r}")
        try:
            requirement = Requirement(line)
        except Exception as exc:
            raise EvidenceError(f"invalid export requirement on line {line_number}") from exc
        if requirement.url is not None:
            raise EvidenceError(f"URL requirement is not allowed on line {line_number}")
        specifiers = list(requirement.specifier)
        if (
            len(specifiers) != 1
            or specifiers[0].operator != "=="
            or specifiers[0].version.endswith(".*")
        ):
            raise EvidenceError(f"requirement is not an exact pin on line {line_number}")
        normalized = canonicalize_name(requirement.name)
        try:
            version = str(Version(specifiers[0].version))
        except Exception as exc:
            raise EvidenceError(f"invalid version on line {line_number}") from exc
        marker_text = str(requirement.marker) if requirement.marker is not None else ""
        applies = (
            requirement.marker.evaluate(environment=dict(marker_environment))
            if requirement.marker is not None
            else True
        )
        record = {"name": normalized, "version": version, "marker": marker_text}
        if not applies:
            inactive.append(record)
            continue
        previous = active.get(normalized)
        if previous is not None and previous["version"] != version:
            raise EvidenceError(f"multiple active versions in export for {normalized}")
        if previous is not None:
            raise EvidenceError(f"duplicate active requirement in export for {normalized}")
        active[normalized] = record
    if not active:
        raise EvidenceError("frozen export has no active requirements")
    return (
        sorted(active.values(), key=lambda item: item["name"]),
        sorted(inactive, key=lambda item: (item["name"], item["version"], item["marker"])),
    )


def _load_project_identity(pyproject_raw: bytes) -> dict[str, str]:
    try:
        payload = tomllib.loads(pyproject_raw.decode("utf-8", errors="strict"))
        project = payload["project"]
        name = canonicalize_name(str(project["name"]))
        version = str(Version(str(project["version"])))
    except (UnicodeDecodeError, KeyError, TypeError, ValueError) as exc:
        raise EvidenceError("pyproject project identity is invalid") from exc
    return {"name": name, "version": version}


def _reconcile_installed(
    expected_dependencies: Sequence[Mapping[str, str]],
    project_identity: Mapping[str, str],
    installed: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = {
        str(item["name"]): str(Version(str(item["version"]))) for item in expected_dependencies
    }
    project_name = str(project_identity["name"])
    if project_name in expected:
        raise EvidenceError("frozen export unexpectedly includes the local project")
    expected[project_name] = str(Version(str(project_identity["version"])))
    observed: dict[str, dict[str, Any]] = {}
    for item in installed:
        try:
            display_name = str(item["name"])
            normalized = canonicalize_name(display_name)
            version = str(Version(str(item["version"])))
            metadata_sha = str(item["metadata_sha256"])
            metadata_size = int(item["metadata_size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidenceError("target installed-distribution record is invalid") from exc
        if not SHA256_RE.fullmatch(metadata_sha) or metadata_size < 0:
            raise EvidenceError(f"installed metadata binding is invalid for {normalized}")
        if normalized in observed:
            raise EvidenceError(f"duplicate installed distribution: {normalized}")
        observed[normalized] = {
            "name": normalized,
            "display_name": display_name,
            "version": version,
            "metadata_sha256": metadata_sha,
            "metadata_size": metadata_size,
        }
    missing = [
        {"name": name, "expected_version": version}
        for name, version in sorted(expected.items())
        if name not in observed
    ]
    extra = [
        {"name": name, "installed_version": observed[name]["version"]}
        for name in sorted(set(observed) - set(expected))
    ]
    version_mismatch = [
        {
            "name": name,
            "expected_version": expected[name],
            "installed_version": observed[name]["version"],
        }
        for name in sorted(set(expected) & set(observed))
        if Version(expected[name]) != Version(observed[name]["version"])
    ]
    return {
        "expected_count": len(expected),
        "third_party_expected_count": len(expected_dependencies),
        "installed_count": len(observed),
        "local_project_identity_only": {
            "name": project_name,
            "version": expected[project_name],
            "artifact_provenance_verified": False,
        },
        "installed": [observed[name] for name in sorted(observed)],
        "missing": missing,
        "extra": extra,
        "version_mismatch": version_mismatch,
        "exact_match": not missing and not extra and not version_mismatch,
    }


def _lock_packages(lock_raw: bytes) -> list[Mapping[str, Any]]:
    try:
        payload = tomllib.loads(lock_raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise EvidenceError("uv.lock is invalid TOML") from exc
    packages = payload.get("package")
    if not isinstance(packages, list):
        raise EvidenceError("uv.lock package table is missing")
    return [item for item in packages if isinstance(item, dict)]


def _reconcile_lock(
    export_dependencies: Sequence[Mapping[str, str]],
    project_identity: Mapping[str, str],
    packages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_pairs = {
        (str(item["name"]), str(Version(str(item["version"])))) for item in export_dependencies
    }
    project_pair = (
        str(project_identity["name"]),
        str(Version(str(project_identity["version"]))),
    )
    expected_pairs.add(project_pair)
    observed_counts: dict[tuple[str, str], int] = {}
    malformed_count = 0
    for package in packages:
        try:
            pair = (
                canonicalize_name(str(package["name"])),
                str(Version(str(package["version"]))),
            )
        except (KeyError, TypeError, ValueError):
            malformed_count += 1
            continue
        observed_counts[pair] = observed_counts.get(pair, 0) + 1
    observed_pairs = set(observed_counts)
    missing = [
        {"name": name, "version": version}
        for name, version in sorted(expected_pairs - observed_pairs)
    ]
    extra = [
        {"name": name, "version": version}
        for name, version in sorted(observed_pairs - expected_pairs)
    ]
    duplicates = [
        {"name": name, "version": version, "count": count}
        for (name, version), count in sorted(observed_counts.items())
        if count != 1
    ]
    local_count = observed_counts.get(project_pair, 0)
    return {
        "expected_count": len(expected_pairs),
        "observed_count": len(observed_pairs),
        "malformed_count": malformed_count,
        "local_project": {
            "name": project_pair[0],
            "version": project_pair[1],
            "count": local_count,
        },
        "missing": missing,
        "extra": extra,
        "duplicates": duplicates,
        "exact_match": (
            malformed_count == 0
            and not missing
            and not extra
            and not duplicates
            and local_count == 1
        ),
    }


def _pip_absence(
    *,
    active: Sequence[Mapping[str, str]],
    inactive: Sequence[Mapping[str, str]],
    lock_packages: Sequence[Mapping[str, Any]],
    installed: Sequence[Mapping[str, Any]],
    probe: Mapping[str, Any],
) -> dict[str, Any]:
    export_names = {canonicalize_name(str(item["name"])) for item in [*active, *inactive]}
    lock_names: set[str] = set()
    for package in lock_packages:
        name = package.get("name")
        if isinstance(name, str):
            lock_names.add(canonicalize_name(name))
    installed_names: set[str] = set()
    for item in installed:
        name = item.get("name")
        if isinstance(name, str):
            installed_names.add(canonicalize_name(name))
    pip_probe = probe.get("pip_probe")
    if not isinstance(pip_probe, dict):
        raise EvidenceError("target Python probe omitted pip absence data")
    _expect_keys(
        pip_probe,
        {
            "find_spec_is_none",
            "site_package_paths",
            "site_package_matches",
            "bin_matches",
        },
        label="target pip probe",
    )
    for key in ("site_package_paths", "site_package_matches", "bin_matches"):
        if not isinstance(pip_probe[key], list) or not all(
            isinstance(item, str) for item in pip_probe[key]
        ):
            raise EvidenceError(f"target pip probe {key} must be a string array")
    if not isinstance(pip_probe["find_spec_is_none"], bool):
        raise EvidenceError("target pip find_spec result must be a boolean")
    result = {
        "export_absent": "pip" not in export_names,
        "lock_absent": "pip" not in lock_names,
        "installed_distribution_absent": "pip" not in installed_names,
        "find_spec_none": pip_probe["find_spec_is_none"],
        "site_package_paths": sorted(pip_probe["site_package_paths"]),
        "site_package_matches": sorted(pip_probe["site_package_matches"]),
        "bin_matches": sorted(pip_probe["bin_matches"]),
    }
    result["accepted"] = bool(
        result["export_absent"]
        and result["lock_absent"]
        and result["installed_distribution_absent"]
        and result["find_spec_none"]
        and not result["site_package_matches"]
        and not result["bin_matches"]
    )
    return result


def _artifact_record(value: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    try:
        url = str(value["url"])
        hash_value = str(value["hash"])
        size = int(value["size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceError(f"locked {kind} metadata is incomplete") from exc
    if not hash_value.startswith("sha256:"):
        raise EvidenceError(f"locked {kind} hash is not SHA256")
    digest = hash_value.removeprefix("sha256:")
    if not SHA256_RE.fullmatch(digest) or size <= 0:
        raise EvidenceError(f"locked {kind} hash or size is invalid")
    filename = unquote(Path(urlsplit(url).path).name)
    if not filename:
        raise EvidenceError(f"locked {kind} URL has no filename")
    return {
        "kind": kind,
        "filename": filename,
        "url": url,
        "sha256": digest,
        "size": size,
    }


def _select_lock_artifact(
    packages: Sequence[Mapping[str, Any]],
    *,
    name: str,
    version: str,
    supported_tags: Sequence[str],
) -> dict[str, Any]:
    matches = []
    for package in packages:
        try:
            package_name = canonicalize_name(str(package["name"]))
            package_version = str(Version(str(package["version"])))
        except (KeyError, TypeError, ValueError):
            continue
        if package_name == name and Version(package_version) == Version(version):
            matches.append(package)
    if len(matches) != 1:
        raise EvidenceError(
            f"uv.lock must contain exactly one {name}=={version} package; found {len(matches)}"
        )
    package = matches[0]
    source = package.get("source")
    if not isinstance(source, dict) or "registry" not in source:
        raise EvidenceError(f"locked package is not registry-backed: {name}=={version}")
    tag_rank = {value: index for index, value in enumerate(supported_tags)}
    candidates: list[tuple[int, str, dict[str, Any]]] = []
    wheels = package.get("wheels", [])
    if not isinstance(wheels, list):
        raise EvidenceError(f"locked wheels metadata is invalid: {name}=={version}")
    for value in wheels:
        if not isinstance(value, dict):
            raise EvidenceError(f"locked wheel record is invalid: {name}=={version}")
        record = _artifact_record(value, kind="wheel")
        try:
            wheel_name, wheel_version, _build, tags = parse_wheel_filename(record["filename"])
        except Exception as exc:
            raise EvidenceError(f"locked wheel filename is invalid: {record['filename']}") from exc
        if canonicalize_name(wheel_name) != name or Version(str(wheel_version)) != Version(version):
            raise EvidenceError(f"locked wheel identity mismatch: {record['filename']}")
        record["tags"] = sorted(str(tag) for tag in tags)
        record["requires_source_build"] = False
        ranks = [tag_rank[tag] for tag in record["tags"] if tag in tag_rank]
        if ranks:
            candidates.append((min(ranks), record["filename"], record))
    if candidates:
        return sorted(candidates, key=lambda item: (item[0], item[1]))[0][2]
    sdist = package.get("sdist")
    if not isinstance(sdist, dict):
        raise EvidenceError(f"no target-compatible wheel or sdist: {name}=={version}")
    record = _artifact_record(sdist, kind="sdist")
    record["tags"] = []
    record["requires_source_build"] = True
    return record


def _verify_raw_artifact(path: Path, selected: Mapping[str, Any]) -> dict[str, Any]:
    raw, observed = _stable_file_bytes(path)
    actual_sha = _sha256_bytes(raw)
    actual_size = len(raw)
    expected_sha = str(selected["sha256"])
    expected_size = int(selected["size"])
    valid = actual_sha == expected_sha and actual_size == expected_size
    return {
        "source": "wheelhouse_raw_artifact",
        "path": str(path),
        "raw_artifact_retained": True,
        "sha256": actual_sha,
        "size": actual_size,
        "mode": f"{stat.S_IMODE(observed.st_mode):04o}",
        "valid": valid,
        "errors": (
            []
            if valid
            else [
                {
                    "reason": "raw_artifact_hash_or_size_mismatch",
                    "expected_sha256": expected_sha,
                    "expected_size": expected_size,
                    "actual_sha256": actual_sha,
                    "actual_size": actual_size,
                }
            ]
        ),
    }


def _scan_pip_http_cache(
    pip_http_cache: Path,
    selected_artifacts: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[int, str], list[Path]], dict[str, int]]:
    raise EvidenceError("pip HTTP cache evidence is forbidden by the v2 contract")
    expected_by_size: dict[int, set[str]] = {}
    for selected in selected_artifacts:
        expected_by_size.setdefault(int(selected["size"]), set()).add(str(selected["sha256"]))
    matches: dict[tuple[int, str], list[Path]] = {}
    body_count = 0
    size_candidate_count = 0

    def fail_walk(exc: OSError) -> None:
        raise EvidenceError(f"cannot scan pip HTTP cache: {exc}") from exc

    try:
        iterator = os.walk(pip_http_cache, topdown=True, followlinks=False, onerror=fail_walk)
        for directory, child_directories, filenames in iterator:
            child_directories.sort()
            filenames.sort()
            base = Path(directory)
            for filename in filenames:
                if not filename.endswith(".body"):
                    continue
                body_count += 1
                candidate = base / filename
                try:
                    candidate_stat = candidate.lstat()
                except OSError as exc:
                    raise EvidenceError(f"pip HTTP cache path disappeared: {candidate}") from exc
                if not stat.S_ISREG(candidate_stat.st_mode):
                    continue
                expected_hashes = expected_by_size.get(candidate_stat.st_size)
                if not expected_hashes:
                    continue
                size_candidate_count += 1
                raw, _ = _stable_file_bytes(candidate)
                digest = _sha256_bytes(raw)
                if digest in expected_hashes:
                    matches.setdefault((len(raw), digest), []).append(candidate)
    except EvidenceError:
        raise
    except OSError as exc:
        raise EvidenceError(f"cannot scan pip HTTP cache: {pip_http_cache}") from exc
    for paths in matches.values():
        paths.sort(key=lambda path: str(path).encode("utf-8"))
    return matches, {
        "body_file_count": body_count,
        "size_candidate_count": size_candidate_count,
        "exact_match_count": sum(len(paths) for paths in matches.values()),
    }


def _materialize_raw_artifact_exact_once(
    *,
    source: Path,
    destination: Path,
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    raise EvidenceError("wheelhouse materialization is forbidden by the v2 contract")
    parent = destination.parent
    parent_stat = parent.lstat()
    if (
        not stat.S_ISDIR(parent_stat.st_mode)
        or stat.S_IMODE(parent_stat.st_mode) != 0o700
        or parent_stat.st_uid != os.getuid()
    ):
        raise EvidenceError("materialized wheelhouse must be owner-private mode 0700")
    source_raw, _ = _stable_file_bytes(source)
    if len(source_raw) != int(selected["size"]) or _sha256_bytes(source_raw) != str(
        selected["sha256"]
    ):
        raise EvidenceError("pip HTTP cache artifact drifted before materialization")
    temporary = parent / (
        f".{destination.name}.{os.getpid()}.{os.urandom(8).hex()}.materialize.tmp"
    )
    created = False
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(source_raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
            created = True
        except FileExistsError:
            existing = _verify_raw_artifact(destination, selected)
            if not existing["valid"]:
                raise EvidenceError("materialized wheelhouse filename exists with different bytes")
        os.chmod(destination, 0o600, follow_symlinks=False)
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    verified = _verify_raw_artifact(destination, selected)
    if not verified["valid"] or stat.S_IMODE(destination.lstat().st_mode) != 0o600:
        raise EvidenceError("materialized artifact readback failed")
    return {
        "created": created,
        "destination": str(destination),
        "sha256": verified["sha256"],
        "size": verified["size"],
        "mode": verified["mode"],
    }


def _pip_http_cache_evidence(
    *,
    selected: Mapping[str, Any],
    matches: Mapping[tuple[int, str], Sequence[Path]],
    wheelhouse: Path,
    materialize_wheelhouse: bool,
) -> dict[str, Any] | None:
    raise EvidenceError("pip HTTP cache evidence is forbidden by the v2 contract")
    key = (int(selected["size"]), str(selected["sha256"]))
    candidates = list(matches.get(key, ()))
    if not candidates:
        return None
    source = candidates[0]
    source_evidence = _verify_raw_artifact(source, selected)
    if not source_evidence["valid"]:
        raise EvidenceError("pip HTTP cache artifact failed readback")
    evidence: dict[str, Any] = {
        **source_evidence,
        "source": "pip_http_cache_raw_artifact",
        "matching_body_count": len(candidates),
    }
    if materialize_wheelhouse:
        destination = wheelhouse / str(selected["filename"])
        materialization = _materialize_raw_artifact_exact_once(
            source=source,
            destination=destination,
            selected=selected,
        )
        evidence = {
            **_verify_raw_artifact(destination, selected),
            "source": "wheelhouse_materialized_from_pip_http_cache",
            "pip_http_cache_source": {
                "path": str(source),
                "sha256": source_evidence["sha256"],
                "size": source_evidence["size"],
                "matching_body_count": len(candidates),
            },
            "materialization": materialization,
        }
    return evidence


def _metadata_identity(path: Path) -> tuple[str, str]:
    raw, _ = _stable_file_bytes(path)
    message = BytesParser(policy=email_policy).parsebytes(raw)
    name = message.get("Name")
    version = message.get("Version")
    if not name or not version:
        raise EvidenceError(f"cached METADATA omits Name or Version: {path}")
    return canonicalize_name(str(name)), str(Version(str(version)))


def _wheel_metadata_tags(path: Path) -> tuple[list[str], dict[str, Any]]:
    raw, observed = _stable_file_bytes(path)
    message = BytesParser(policy=email_policy).parsebytes(raw)
    tags = [str(value) for value in (message.get_all("Tag") or [])]
    if not tags:
        raise EvidenceError(f"cached WHEEL omits Tag: {path}")
    for value in tags:
        try:
            Tag(*value.split("-", 2))
        except Exception as exc:
            raise EvidenceError(f"cached WHEEL has invalid Tag: {path}") from exc
    return tags, {
        "path": str(path),
        "sha256": _sha256_bytes(raw),
        "size": len(raw),
        "mode": f"{stat.S_IMODE(observed.st_mode):04o}",
    }


def _find_uv_cache_fallback(
    uv_cache: Path,
    *,
    name: str,
    version: str,
    selected: Mapping[str, Any],
    supported_tags: Sequence[str],
) -> dict[str, Any] | None:
    archive_root = uv_cache / "archive-v0"
    if not archive_root.exists():
        return None
    if not archive_root.is_dir():
        raise EvidenceError("uv archive-v0 is not a directory")
    supported = set(supported_tags)
    selected_tags = set(str(value) for value in selected.get("tags", []))
    matches: list[tuple[int, str, dict[str, Any]]] = []
    try:
        archive_dirs = sorted(
            (path for path in archive_root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
    except OSError as exc:
        raise EvidenceError("cannot enumerate uv archive-v0") from exc
    for archive_dir in archive_dirs:
        try:
            metadata_paths = sorted(archive_dir.glob("*.dist-info/METADATA"))
        except OSError as exc:
            raise EvidenceError(f"cannot inspect uv archive: {archive_dir}") from exc
        for metadata_path in metadata_paths:
            cached_name, cached_version = _metadata_identity(metadata_path)
            if cached_name != name or Version(cached_version) != Version(version):
                continue
            wheel_path = metadata_path.parent / "WHEEL"
            if not wheel_path.is_file():
                continue
            tags, wheel_binding = _wheel_metadata_tags(wheel_path)
            compatible = supported.intersection(tags)
            if not compatible:
                continue
            if selected["kind"] == "wheel" and not selected_tags.intersection(tags):
                continue
            metadata_raw, metadata_stat = _stable_file_bytes(metadata_path)
            archive_stat_before = archive_dir.stat()
            archive_stat_after = archive_dir.stat()
            if (
                archive_stat_before.st_dev,
                archive_stat_before.st_ino,
                archive_stat_before.st_mode,
                archive_stat_before.st_mtime_ns,
            ) != (
                archive_stat_after.st_dev,
                archive_stat_after.st_ino,
                archive_stat_after.st_mode,
                archive_stat_after.st_mtime_ns,
            ):
                raise EvidenceError(f"uv archive identity drift: {archive_dir}")
            rank = min(supported_tags.index(tag) for tag in compatible if tag in supported_tags)
            record = {
                "source": "uv_extracted_cache",
                "raw_artifact_retained": False,
                "valid": True,
                "metadata": {
                    "path": str(metadata_path),
                    "sha256": _sha256_bytes(metadata_raw),
                    "size": len(metadata_raw),
                    "mode": f"{stat.S_IMODE(metadata_stat.st_mode):04o}",
                },
                "wheel_metadata": {**wheel_binding, "tags": sorted(tags)},
                "archive_directory_identity": {
                    "path": str(archive_dir),
                    "name": archive_dir.name,
                    "st_dev": archive_stat_before.st_dev,
                    "st_ino": archive_stat_before.st_ino,
                    "mode": f"{stat.S_IMODE(archive_stat_before.st_mode):04o}",
                    "mtime_ns": archive_stat_before.st_mtime_ns,
                },
                "errors": [],
            }
            matches.append((rank, archive_dir.name, record))
    if not matches:
        return None
    return sorted(matches, key=lambda item: (item[0], item[1]))[0][2]


def _artifact_evidence(
    *,
    expected_dependencies: Sequence[Mapping[str, str]],
    lock_packages: Sequence[Mapping[str, Any]],
    supported_tags: Sequence[str],
    wheelhouse: Path,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    retained_count = 0
    for dependency in expected_dependencies:
        name = str(dependency["name"])
        version = str(dependency["version"])
        try:
            selected = _select_lock_artifact(
                lock_packages,
                name=name,
                version=version,
                supported_tags=supported_tags,
            )
        except EvidenceError as exc:
            invalid.append(
                {
                    "name": name,
                    "version": version,
                    "reason": "locked_artifact_invalid",
                    "detail": str(exc),
                }
            )
            continue
        raw_path = wheelhouse / str(selected["filename"])
        if raw_path.exists() or raw_path.is_symlink():
            try:
                evidence = _verify_raw_artifact(raw_path, selected)
            except EvidenceError as exc:
                invalid.append(
                    {
                        "name": name,
                        "version": version,
                        "reason": "raw_artifact_unreadable",
                        "detail": str(exc),
                    }
                )
                continue
            if evidence["valid"]:
                if evidence["mode"] != "0600":
                    invalid.append(
                        {
                            "name": name,
                            "version": version,
                            "reason": "wheelhouse_artifact_mode_invalid",
                            "detail": {
                                "path": evidence["path"],
                                "expected_mode": "0600",
                                "actual_mode": evidence["mode"],
                            },
                        }
                    )
                    continue
                retained_count += 1
            else:
                invalid.append(
                    {
                        "name": name,
                        "version": version,
                        "reason": "raw_artifact_invalid",
                        "detail": evidence["errors"],
                    }
                )
                continue
        else:
            missing.append(
                {
                    "name": name,
                    "version": version,
                    "reason": "retained_wheelhouse_artifact_missing",
                    "selected_filename": selected["filename"],
                }
            )
            continue
        records.append(
            {
                "name": name,
                "version": version,
                "selected_locked_artifact": selected,
                "requires_source_build": bool(selected["requires_source_build"]),
                "raw_lock_artifact_retained": bool(evidence["raw_artifact_retained"]),
                "evidence": evidence,
            }
        )
    expected_count = len(expected_dependencies)
    return {
        "expected_artifact_count": expected_count,
        "evidenced_artifact_count": len(records),
        "complete_raw_artifact_count": retained_count,
        "wheelhouse_raw_artifact_count": retained_count,
        "complete_raw_artifacts": retained_count == expected_count,
        "complete_wheelhouse": retained_count == expected_count,
        "pip_http_cache_used": False,
        "materialization_performed": False,
        "records": records,
        "missing": missing,
        "invalid": invalid,
    }


def _validate_source_binding(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must be an object")
    _expect_keys(
        value,
        {
            "base_commit",
            "source_state_sha256",
            "porcelain_sha256",
            "binary_diff_sha256",
            "untracked_inventory_sha256",
        },
        label=label,
    )
    base_commit = _require_string(value["base_commit"], label=f"{label}.base_commit")
    if re.fullmatch(r"[0-9a-f]{40,64}", base_commit) is None:
        raise EvidenceError(f"{label}.base_commit must be a full Git object id")
    for key in (
        "source_state_sha256",
        "porcelain_sha256",
        "binary_diff_sha256",
        "untracked_inventory_sha256",
    ):
        _require_sha256(value[key], label=f"{label}.{key}")
    return dict(value)


def _validate_toolchain(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must be an object")
    _expect_keys(value, {"base_python", "uv", "uv_cache", "pip_scope"}, label=label)
    base = value["base_python"]
    if not isinstance(base, dict):
        raise EvidenceError(f"{label}.base_python must be an object")
    _expect_keys(
        base,
        {
            "lexical_path",
            "realpath",
            "sha256",
            "size_bytes",
            "mode",
            "executable",
            "implementation",
            "version",
            "version_info",
        },
        label=f"{label}.base_python",
    )
    expected_base = {
        "lexical_path": BASE_PYTHON_PATH,
        "realpath": BASE_PYTHON_PATH,
        "sha256": BASE_PYTHON_SHA256,
        "size_bytes": BASE_PYTHON_SIZE,
        "mode": "0755",
        "executable": True,
        "implementation": "cpython",
        "version": "3.13.7",
        "version_info": [3, 13, 7],
    }
    if base != expected_base:
        raise EvidenceError(f"{label}.base_python identity mismatch")
    uv = value["uv"]
    if not isinstance(uv, dict):
        raise EvidenceError(f"{label}.uv must be an object")
    _expect_keys(
        uv,
        {
            "lexical_path",
            "realpath",
            "sha256",
            "size_bytes",
            "mode",
            "executable",
            "version",
            "output",
        },
        label=f"{label}.uv",
    )
    expected_uv = {
        "lexical_path": UV_PATH,
        "realpath": UV_PATH,
        "sha256": UV_SHA256,
        "size_bytes": UV_SIZE,
        "mode": "0755",
        "executable": True,
        "version": "0.10.9",
        "output": UV_VERSION_OUTPUT,
    }
    if uv != expected_uv:
        raise EvidenceError(f"{label}.uv identity mismatch")
    cache = value["uv_cache"]
    if not isinstance(cache, dict):
        raise EvidenceError(f"{label}.uv_cache must be an object")
    _expect_keys(
        cache,
        {"path", "realpath", "st_dev", "st_ino", "uid", "mode"},
        label=f"{label}.uv_cache",
    )
    if cache["path"] != UV_CACHE_PATH or cache["realpath"] != UV_CACHE_PATH:
        raise EvidenceError(f"{label}.uv_cache path mismatch")
    for key in ("st_dev", "st_ino", "uid"):
        _require_int(cache[key], label=f"{label}.uv_cache.{key}")
    if not isinstance(cache["mode"], str) or re.fullmatch(r"[0-7]{4}", cache["mode"]) is None:
        raise EvidenceError(f"{label}.uv_cache.mode is invalid")
    if value["pip_scope"] != PIP_SCOPE:
        raise EvidenceError(f"{label}.pip_scope mismatch")
    return dict(value)


def _observed_toolchain(uv_binary: str, uv_cache: Path, *, repo_root: Path) -> dict[str, Any]:
    base = _executable_file_binding(Path(BASE_PYTHON_PATH), label="frozen base CPython")
    if base != {
        "path": BASE_PYTHON_PATH,
        "sha256": BASE_PYTHON_SHA256,
        "size": BASE_PYTHON_SIZE,
        "mode": "0755",
        "executable": True,
    }:
        raise EvidenceError("frozen base CPython binary identity mismatch")
    uv_binding = _uv_binding(uv_binary, cwd=repo_root)
    cache = _directory_identity(uv_cache, label="uv cache", owner_private=False)
    return {
        "base_python": {
            "lexical_path": BASE_PYTHON_PATH,
            "realpath": BASE_PYTHON_PATH,
            "sha256": BASE_PYTHON_SHA256,
            "size_bytes": BASE_PYTHON_SIZE,
            "mode": "0755",
            "executable": True,
            "implementation": "cpython",
            "version": "3.13.7",
            "version_info": [3, 13, 7],
        },
        "uv": {
            "lexical_path": UV_PATH,
            "realpath": uv_binding["path"],
            "sha256": uv_binding["sha256"],
            "size_bytes": uv_binding["size"],
            "mode": uv_binding["mode"],
            "executable": uv_binding["executable"],
            "version": "0.10.9",
            "output": uv_binding["version_output"],
        },
        "uv_cache": {
            "path": cache["path"],
            "realpath": cache["path"],
            "st_dev": cache["st_dev"],
            "st_ino": cache["st_ino"],
            "uid": cache["st_uid"],
            "mode": cache["mode"],
        },
        "pip_scope": PIP_SCOPE,
    }


def _validate_package_source_superset(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must be an object")
    _expect_keys(value, {"row_count", "sha256"}, label=label)
    _require_int(value["row_count"], label=f"{label}.row_count")
    _require_sha256(value["sha256"], label=f"{label}.sha256")
    return dict(value)


def _sample_protected_roots() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for identifier, text_path in PROTECTED_ROOT_SPECS:
        path = Path(text_path)
        try:
            value = path.lstat()
        except FileNotFoundError:
            rows.append({"id": identifier, "path": text_path, "state": "ABSENT"})
            continue
        except OSError as exc:
            raise EvidenceError(f"cannot inspect protected root: {text_path}") from exc
        if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
            raise EvidenceError(f"protected root must be absent or a real directory: {text_path}")
        try:
            realpath = str(path.resolve(strict=True))
        except OSError as exc:
            raise EvidenceError(f"cannot resolve protected root: {text_path}") from exc
        rows.append(
            {
                "id": identifier,
                "path": text_path,
                "state": "PRESENT_DIRECTORY",
                "realpath": realpath,
                "st_dev": value.st_dev,
                "st_ino": value.st_ino,
                "uid": value.st_uid,
                "mode": f"{stat.S_IMODE(value.st_mode):04o}",
                "mtime_ns": value.st_mtime_ns,
                "ctime_ns": value.st_ctime_ns,
            }
        )
    return rows


def _validate_protected_roots(value: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(PROTECTED_ROOT_SPECS):
        raise EvidenceError(f"{label} must contain exactly four ordered rows")
    rows: list[dict[str, Any]] = []
    for index, ((expected_id, expected_path), row) in enumerate(
        zip(PROTECTED_ROOT_SPECS, value, strict=True)
    ):
        row_label = f"{label}[{index}]"
        if not isinstance(row, dict):
            raise EvidenceError(f"{row_label} must be an object")
        state = row.get("state")
        if state == "ABSENT":
            _expect_keys(row, {"id", "path", "state"}, label=row_label)
        elif state == "PRESENT_DIRECTORY":
            _expect_keys(
                row,
                {
                    "id",
                    "path",
                    "state",
                    "realpath",
                    "st_dev",
                    "st_ino",
                    "uid",
                    "mode",
                    "mtime_ns",
                    "ctime_ns",
                },
                label=row_label,
            )
            for key in ("st_dev", "st_ino", "uid", "mtime_ns", "ctime_ns"):
                _require_int(row[key], label=f"{row_label}.{key}")
            if not isinstance(row["realpath"], str) or not isinstance(row["mode"], str):
                raise EvidenceError(f"{row_label} realpath/mode is invalid")
        else:
            raise EvidenceError(f"{row_label}.state is invalid")
        if row["id"] != expected_id or row["path"] != expected_path:
            raise EvidenceError(f"{row_label} identity/order mismatch")
        rows.append(dict(row))
    return rows


def _verify_bound_file(
    binding: Mapping[str, Any],
    *,
    label: str,
    required_mode: str | None = None,
) -> dict[str, Any]:
    path = Path(_require_string(binding["path"], label=f"{label}.path"))
    if not path.is_absolute():
        raise EvidenceError(f"{label}.path must be absolute")
    raw, observed = _stable_file_bytes(path)
    _require_sha256(binding["sha256"], label=f"{label}.sha256")
    _require_int(binding["size_bytes"], label=f"{label}.size_bytes")
    if binding["sha256"] != _sha256_bytes(raw) or binding["size_bytes"] != len(raw):
        raise EvidenceError(f"{label} live byte binding mismatch")
    mode = f"{stat.S_IMODE(observed.st_mode):04o}"
    if required_mode is not None and mode != required_mode:
        raise EvidenceError(f"{label} must be mode {required_mode}")
    if "semantic_sha256" in binding:
        _require_sha256(binding["semantic_sha256"], label=f"{label}.semantic_sha256")
    return {
        "path": str(path),
        "sha256": _sha256_bytes(raw),
        "size_bytes": len(raw),
        "mode": mode,
    }


def _parse_native_sync_log(
    path: Path,
    *,
    repo_root: Path,
    work_root: Path,
    target_venv: Path,
    source: Mapping[str, Any],
    toolchain: Mapping[str, Any],
    protected_roots: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, log_stat = _stable_file_bytes(path)
    if len(raw) >= 512 * 1024 * 1024:
        raise EvidenceError("native sync log exceeds the <512 MiB file limit")
    newline = raw.find(b"\n")
    if newline < 0 or not raw.startswith(COMMAND_RECEIPT_PREFIX):
        raise EvidenceError("native sync log lacks the canonical command receipt line")
    receipt_raw = raw[len(COMMAND_RECEIPT_PREFIX) : newline]
    try:
        receipt = json.loads(receipt_raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("native sync command receipt is invalid JSON") from exc
    if not isinstance(receipt, dict):
        raise EvidenceError("native sync command receipt must be an object")
    if receipt_raw != _canonical_bytes(receipt):
        raise EvidenceError("native sync command receipt must be compact canonical JSON")
    _expect_keys(receipt, RECEIPT_ROOT_KEYS, label="native sync command receipt")
    if receipt["version"] != COMMAND_RECEIPT_VERSION:
        raise EvidenceError("native sync command receipt v1/downgrade is rejected")
    if receipt["protocol_version"] != PROTOCOL_VERSION:
        raise EvidenceError("native sync command receipt protocol mismatch")
    semantic = _require_sha256(
        receipt["semantic_sha256"], label="native sync command receipt.semantic_sha256"
    )
    semantic_body = dict(receipt)
    semantic_body.pop("semantic_sha256")
    if semantic != _sha256_bytes(_canonical_bytes(semantic_body)):
        raise EvidenceError("native sync command receipt semantic SHA-256 mismatch")

    session = receipt["session_binding"]
    if not isinstance(session, dict):
        raise EvidenceError("native sync session_binding must be an object")
    _expect_keys(
        session,
        {"session_id", "path", "sha256", "size_bytes", "semantic_sha256"},
        label="native sync session_binding",
    )
    session_id = _require_string(
        session["session_id"], label="native sync session_binding.session_id"
    )
    if SESSION_ID_RE.fullmatch(session_id) is None:
        raise EvidenceError("native sync session id is invalid")
    _verify_bound_file(session, label="native sync session_binding", required_mode="0600")

    step = receipt["step"]
    if step != {
        "ordinal": 1,
        "role": "native_sync_log",
        "kind": "log",
        "filename": "10_native_sync.log",
    }:
        raise EvidenceError("native sync command receipt step mismatch")
    if path.name != "10_native_sync.log":
        raise EvidenceError("native sync log filename must be 10_native_sync.log")

    producer = receipt["producer"]
    if not isinstance(producer, dict):
        raise EvidenceError("native sync producer must be an object")
    _expect_keys(
        producer,
        {"path", "version", "sha256", "size_bytes"},
        label="native sync producer",
    )
    _require_string(producer["version"], label="native sync producer.version")
    _verify_bound_file(producer, label="native sync producer")

    source_before = _validate_source_binding(
        receipt["source_before"], label="native sync source_before"
    )
    source_after = _validate_source_binding(
        receipt["source_after"], label="native sync source_after"
    )
    if source_before != source_after or source_after != source:
        raise EvidenceError("native sync source binding changed or is stale")

    toolchain_before = _validate_toolchain(
        receipt["toolchain_before"], label="native sync toolchain_before"
    )
    toolchain_after = _validate_toolchain(
        receipt["toolchain_after"], label="native sync toolchain_after"
    )
    if toolchain_before != toolchain_after or toolchain_after != toolchain:
        raise EvidenceError("native sync toolchain changed or is stale")

    package_before = _validate_package_source_superset(
        receipt["package_source_superset_before"],
        label="native sync package_source_superset_before",
    )
    package_after = _validate_package_source_superset(
        receipt["package_source_superset_after"],
        label="native sync package_source_superset_after",
    )
    if package_before != package_after:
        raise EvidenceError("native sync package source superset changed")

    roots_before = _validate_protected_roots(
        receipt["protected_roots_before"], label="native sync protected_roots_before"
    )
    roots_after = _validate_protected_roots(
        receipt["protected_roots_after"], label="native sync protected_roots_after"
    )
    if roots_before != roots_after or roots_after != list(protected_roots):
        raise EvidenceError("native sync protected roots changed or are stale")
    if receipt["limitations"] != NORMATIVE_LIMITATIONS:
        raise EvidenceError("native sync limitations mismatch")
    if receipt["outcome"] != "PASSED" or receipt["failure_codes"] != []:
        raise EvidenceError("native sync command receipt is not successful")
    if receipt["claims"] != {"exit_code": 0} or isinstance(
        receipt["claims"].get("exit_code"), bool
    ):
        raise EvidenceError("native sync claims mismatch")
    if receipt["framing"] != COMMAND_FRAMING:
        raise EvidenceError("native sync framing mismatch")

    commands = receipt["commands"]
    if not isinstance(commands, list) or len(commands) != 1:
        raise EvidenceError("native sync receipt must contain exactly one command")
    command = commands[0]
    if not isinstance(command, dict):
        raise EvidenceError("native sync command must be an object")
    _expect_keys(command, COMMAND_KEYS, label="native sync command")
    for key in (
        "ordinal",
        "exit_code",
        "stdout_offset_bytes",
        "stdout_size_bytes",
        "stderr_offset_bytes",
        "stderr_size_bytes",
    ):
        _require_int(command[key], label=f"native sync command.{key}")
    if command["signal"] is not None:
        raise EvidenceError("native sync command.signal must be null")
    expected_argv = [
        UV_PATH,
        "sync",
        "--python",
        BASE_PYTHON_PATH,
        "--locked",
        "--all-extras",
        "--offline",
    ]
    expected_environment = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PIP_CONFIG_FILE": "/dev/null",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INDEX": "1",
        "PIP_NO_INPUT": "1",
        "UV_CACHE_DIR": UV_CACHE_PATH,
        "UV_NO_CONFIG": "1",
        "UV_OFFLINE": "1",
        "UV_PYTHON_DOWNLOADS": "never",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "TMPDIR": str(work_root / "tmp" / "native_sync"),
        "UV_PROJECT_ENVIRONMENT": str(target_venv),
    }
    if (
        command["ordinal"] != 1
        or command["argv"] != expected_argv
        or command["cwd"] != str(repo_root)
        or command["environment"] != expected_environment
        or command["exit_code"] != 0
        or command["tool_version"] != UV_VERSION_OUTPUT
    ):
        raise EvidenceError("native sync command semantics mismatch")

    framed = raw[newline + 1 :]
    _require_sha256(receipt["output_sha256"], label="native sync output_sha256")
    _require_int(receipt["output_size_bytes"], label="native sync output_size_bytes")
    if receipt["output_sha256"] != _sha256_bytes(framed) or receipt["output_size_bytes"] != len(
        framed
    ):
        raise EvidenceError("native sync framed output binding mismatch")
    if len(framed) < 16:
        raise EvidenceError("native sync framed output is truncated")
    stdout_size = struct.unpack(">Q", framed[:8])[0]
    if stdout_size > 128 * 1024 * 1024:
        raise EvidenceError("native sync stdout exceeds the 128 MiB stream limit")
    stdout_start = 8
    stdout_end = stdout_start + stdout_size
    if stdout_end + 8 > len(framed):
        raise EvidenceError("native sync stdout frame is truncated")
    stderr_size = struct.unpack(">Q", framed[stdout_end : stdout_end + 8])[0]
    if stderr_size > 128 * 1024 * 1024:
        raise EvidenceError("native sync stderr exceeds the 128 MiB stream limit")
    stderr_start = stdout_end + 8
    stderr_end = stderr_start + stderr_size
    if stderr_end != len(framed) or len(framed) > 256 * 1024 * 1024:
        raise EvidenceError("native sync command frame size is invalid")
    stdout = framed[stdout_start:stdout_end]
    stderr = framed[stderr_start:stderr_end]
    if (
        command["stdout_offset_bytes"] != stdout_start
        or command["stdout_size_bytes"] != stdout_size
        or command["stdout_sha256"] != _sha256_bytes(stdout)
        or command["stderr_offset_bytes"] != stderr_start
        or command["stderr_size_bytes"] != stderr_size
        or command["stderr_sha256"] != _sha256_bytes(stderr)
    ):
        raise EvidenceError("native sync command stream binding mismatch")
    for key in ("stdout_sha256", "stderr_sha256"):
        _require_sha256(command[key], label=f"native sync command.{key}")
    mode = f"{stat.S_IMODE(log_stat.st_mode):04o}"
    if mode != "0600":
        raise EvidenceError("native sync log must be mode 0600")
    projection = {
        "path": str(path),
        "sha256": _sha256_bytes(raw),
        "size_bytes": len(raw),
        "mode": mode,
        "receipt_version": COMMAND_RECEIPT_VERSION,
        "receipt_semantic_sha256": semantic,
        "framed_output_sha256": receipt["output_sha256"],
        "framed_output_size_bytes": receipt["output_size_bytes"],
        "outcome": "PASSED",
    }
    return receipt, projection


def _assess_dependency_environment(
    *,
    native_sync_receipt_valid: bool,
    lock_reconciliation: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
    pip_absence: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    lock_exact_match = bool(lock_reconciliation.get("exact_match"))
    installed_exact_match = bool(reconciliation.get("exact_match"))
    pip_absent = bool(pip_absence.get("accepted"))
    invalid_artifact_evidence = bool(artifacts.get("invalid"))
    native_dependency_environment_accepted = bool(
        native_sync_receipt_valid
        and lock_exact_match
        and installed_exact_match
        and pip_absent
        and not invalid_artifact_evidence
    )
    complete_wheelhouse = bool(artifacts.get("complete_wheelhouse"))
    status = (
        "THIRD_PARTY_NATIVE_ENVIRONMENT_ACCEPTED"
        if native_dependency_environment_accepted
        else "THIRD_PARTY_NATIVE_ENVIRONMENT_REJECTED"
    )
    failures = {
        "native_sync_receipt_invalid": not native_sync_receipt_valid,
        "lock_reconciliation_mismatch": not lock_exact_match,
        "installed_environment_mismatch": not installed_exact_match,
        "native_pip_present": not pip_absent,
        "invalid_artifact_evidence": invalid_artifact_evidence,
    }
    return {
        "accepted": native_dependency_environment_accepted,
        "status": status,
        "native_dependency_environment_accepted": native_dependency_environment_accepted,
        "complete_wheelhouse": complete_wheelhouse,
        "failure_reasons": failures,
    }


def _validate_directory(path: Path, *, label: str) -> None:
    try:
        value = path.stat()
    except OSError as exc:
        raise EvidenceError(f"{label} does not exist: {path}") from exc
    if not stat.S_ISDIR(value.st_mode):
        raise EvidenceError(f"{label} is not a directory: {path}")


def _private_directory_binding(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = path.lstat()
    except OSError as exc:
        raise EvidenceError(f"{label} does not exist: {path}") from exc
    if (
        not stat.S_ISDIR(value.st_mode)
        or stat.S_IMODE(value.st_mode) != 0o700
        or value.st_uid != os.getuid()
    ):
        raise EvidenceError(f"{label} must be an owner-private 0700 directory")
    return {
        "path": str(path),
        "mode": f"{stat.S_IMODE(value.st_mode):04o}",
        "st_uid": value.st_uid,
        "st_dev": value.st_dev,
        "st_ino": value.st_ino,
        "owner_private": True,
    }


def _write_private_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.absolute()
    parent = path.parent
    parent_stat = _stable_lstat(parent, label="output parent")
    if (
        not stat.S_ISDIR(parent_stat.st_mode)
        or stat.S_IMODE(parent_stat.st_mode) != 0o700
        or parent_stat.st_uid != os.getuid()
        or stat.S_ISLNK(parent_stat.st_mode)
    ):
        raise EvidenceError("output parent must be owner-private mode 0700")
    try:
        if parent.resolve(strict=True) != parent:
            raise EvidenceError("output parent must not contain symlink indirection")
    except OSError as exc:
        raise EvidenceError("cannot resolve output parent") from exc
    raw = _canonical_bytes(value) + b"\n"
    directory_descriptor = -1
    staged_descriptor = -1
    read_descriptor = -1
    temporary_name = f".{path.name}.{os.getpid()}.{os.urandom(8).hex()}.tmp"
    staged_identity: tuple[int, int] | None = None
    linked = False
    successful_readback = False

    def parent_matches() -> bool:
        try:
            current = parent.lstat()
        except OSError:
            return False
        return (
            current.st_dev,
            current.st_ino,
            current.st_mode,
            current.st_uid,
        ) == (
            parent_stat.st_dev,
            parent_stat.st_ino,
            parent_stat.st_mode,
            parent_stat.st_uid,
        )

    try:
        directory_descriptor = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        pinned = os.fstat(directory_descriptor)
        if (pinned.st_dev, pinned.st_ino, pinned.st_mode, pinned.st_uid) != (
            parent_stat.st_dev,
            parent_stat.st_ino,
            parent_stat.st_mode,
            parent_stat.st_uid,
        ):
            raise EvidenceError("output parent changed while it was pinned")
        if not parent_matches():
            raise EvidenceError("output parent path changed after pinning")
        try:
            os.stat(path.name, dir_fd=directory_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise EvidenceError("output already exists; exact-once publication refuses overwrite")
        orphan_prefix = f".{path.name}."
        if any(
            name.startswith(orphan_prefix) and name.endswith(".tmp")
            for name in os.listdir(directory_descriptor)
        ):
            raise EvidenceError("orphaned staged output exists; refusing automatic reuse")
        staged_descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_descriptor,
        )
        staged_stat = os.fstat(staged_descriptor)
        if (
            not stat.S_ISREG(staged_stat.st_mode)
            or stat.S_IMODE(staged_stat.st_mode) != 0o600
            or staged_stat.st_nlink != 1
        ):
            raise EvidenceError("staged output inode is not a private regular file")
        staged_identity = (staged_stat.st_dev, staged_stat.st_ino)
        offset = 0
        while offset < len(raw):
            written = os.write(staged_descriptor, raw[offset:])
            if written <= 0:
                raise EvidenceError("short write while staging output")
            offset += written
        os.fsync(staged_descriptor)
        os.close(staged_descriptor)
        staged_descriptor = -1
        if not parent_matches():
            raise EvidenceError("output parent path changed before link commit")
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise EvidenceError("output appeared before exact-once link commit") from exc
        linked = True
        os.fsync(directory_descriptor)
        if not parent_matches():
            raise EvidenceError("output parent path changed after link commit")
        read_descriptor = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
        final_stat = os.fstat(read_descriptor)
        if (
            staged_identity != (final_stat.st_dev, final_stat.st_ino)
            or stat.S_IMODE(final_stat.st_mode) != 0o600
            or final_stat.st_nlink != 2
            or final_stat.st_size != len(raw)
        ):
            raise EvidenceError("published output inode readback mismatch")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(read_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        if b"".join(chunks) != raw:
            raise EvidenceError("published output byte readback mismatch")
        os.close(read_descriptor)
        read_descriptor = -1
        if not parent_matches():
            raise EvidenceError("output parent path changed during readback")
        temporary_stat = os.stat(
            temporary_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if staged_identity != (temporary_stat.st_dev, temporary_stat.st_ino):
            raise EvidenceError("staged output inode changed before cleanup")
        successful_readback = True
        os.unlink(temporary_name, dir_fd=directory_descriptor)
        os.fsync(directory_descriptor)
    finally:
        if read_descriptor >= 0:
            os.close(read_descriptor)
        if staged_descriptor >= 0:
            os.close(staged_descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)
    if not linked or not successful_readback:
        raise EvidenceError("exact-once publication did not complete")


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.absolute().relative_to(root.absolute())
    except ValueError:
        return False
    return True


def _prepare_private_output_path(path: Path, repo_root: Path) -> Path:
    path = path.absolute()
    parent = path.parent
    try:
        prospective_parent = parent.resolve(strict=True)
    except OSError as exc:
        raise EvidenceError(f"cannot resolve output parent: {parent}") from exc
    if _path_is_within(prospective_parent, repo_root):
        raise EvidenceError("output must be outside the bound repository")
    try:
        parent_stat = parent.lstat()
        resolved_parent = parent.resolve(strict=True)
    except OSError as exc:
        raise EvidenceError(f"cannot validate output parent: {parent}") from exc
    if stat.S_ISLNK(parent_stat.st_mode) or resolved_parent != parent:
        raise EvidenceError("output parent path must not contain symlink indirection")
    if _path_is_within(resolved_parent, repo_root):
        raise EvidenceError("resolved output parent is inside the bound repository")
    if (
        not stat.S_ISDIR(parent_stat.st_mode)
        or stat.S_IMODE(parent_stat.st_mode) != 0o700
        or parent_stat.st_uid != os.getuid()
    ):
        raise EvidenceError("output parent must be owner-private mode 0700")
    return resolved_parent / path.name


def _validate_evidence_v2(value: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "status",
        "accepted",
        "dependency_environment_accepted",
        "native_dependency_environment_accepted",
        "complete_wheelhouse",
        "offline_only",
        "network_actions_performed",
        "repackaged_artifacts",
        "session_binding",
        "step",
        "producer",
        "native_sync_log",
        "source",
        "toolchain",
        "package_source_superset",
        "protected_roots",
        "limitations",
        "scope",
        "inputs",
        "runtime",
        "native_sync",
        "local_project_identity_for_environment_reconciliation",
        "expected_dependencies",
        "inactive_marker_dependencies",
        "lock_reconciliation",
        "installed_reconciliation",
        "pip_absence",
        "artifact_evidence",
        "expected_raw_artifact_count",
        "complete_raw_artifact_count",
        "wheelhouse_raw_artifact_count",
        "missing",
        "invalid",
        "failure_reasons",
        "semantic_sha256",
    }
    _expect_keys(value, expected_keys, label="dependency evidence v2")
    if value["schema_version"] != SCHEMA_VERSION:
        raise EvidenceError("dependency evidence v1/downgrade is rejected")
    for key in (
        "accepted",
        "dependency_environment_accepted",
        "native_dependency_environment_accepted",
        "complete_wheelhouse",
        "offline_only",
        "network_actions_performed",
        "repackaged_artifacts",
    ):
        if not isinstance(value[key], bool):
            raise EvidenceError(f"dependency evidence {key} must be a boolean")
    if (
        value["offline_only"] is not True
        or value["network_actions_performed"] is not False
        or value["repackaged_artifacts"] is not False
    ):
        raise EvidenceError("dependency evidence offline/no-materialization flags mismatch")
    session = value["session_binding"]
    if not isinstance(session, dict):
        raise EvidenceError("dependency evidence session_binding must be an object")
    _expect_keys(
        session,
        {"session_id", "path", "sha256", "size_bytes", "semantic_sha256"},
        label="dependency evidence session_binding",
    )
    _require_string(session["session_id"], label="dependency evidence session_id")
    for key in ("sha256", "semantic_sha256"):
        _require_sha256(session[key], label=f"dependency evidence session_binding.{key}")
    _require_int(session["size_bytes"], label="dependency evidence session_binding.size_bytes")
    if value["step"] != {
        "ordinal": 2,
        "role": "native_sync_receipt",
        "kind": "artifact",
        "filename": "20_native_dependency.json",
    }:
        raise EvidenceError("dependency evidence step mismatch")
    producer = value["producer"]
    if not isinstance(producer, dict):
        raise EvidenceError("dependency evidence producer must be an object")
    _expect_keys(
        producer,
        {"path", "version", "sha256", "size_bytes"},
        label="dependency evidence producer",
    )
    if producer["version"] != SCHEMA_VERSION:
        raise EvidenceError("dependency evidence producer version mismatch")
    _require_sha256(producer["sha256"], label="dependency evidence producer.sha256")
    _require_int(producer["size_bytes"], label="dependency evidence producer.size_bytes", minimum=1)
    native_log = value["native_sync_log"]
    if not isinstance(native_log, dict):
        raise EvidenceError("dependency evidence native_sync_log must be an object")
    _expect_keys(
        native_log,
        {
            "path",
            "sha256",
            "size_bytes",
            "mode",
            "receipt_version",
            "receipt_semantic_sha256",
            "framed_output_sha256",
            "framed_output_size_bytes",
            "outcome",
        },
        label="dependency evidence native_sync_log",
    )
    if (
        native_log["receipt_version"] != COMMAND_RECEIPT_VERSION
        or native_log["outcome"] != "PASSED"
        or native_log["mode"] != "0600"
    ):
        raise EvidenceError("dependency evidence native sync log authority mismatch")
    for key in ("sha256", "receipt_semantic_sha256", "framed_output_sha256"):
        _require_sha256(native_log[key], label=f"dependency evidence native_sync_log.{key}")
    for key in ("size_bytes", "framed_output_size_bytes"):
        _require_int(native_log[key], label=f"dependency evidence native_sync_log.{key}")
    _validate_source_binding(value["source"], label="dependency evidence source")
    _validate_toolchain(value["toolchain"], label="dependency evidence toolchain")
    _validate_package_source_superset(
        value["package_source_superset"],
        label="dependency evidence package_source_superset",
    )
    _validate_protected_roots(value["protected_roots"], label="dependency evidence protected_roots")
    if value["limitations"] != NORMATIVE_LIMITATIONS:
        raise EvidenceError("dependency evidence limitations mismatch")
    expected_scope = {
        "kind": "third_party_dependency_environment_only",
        "local_project_identity_checked_for_environment_exactness": True,
        "local_project_artifact_provenance_verified": False,
        "local_project_wheel_sdist_source_state_release_binding": (
            "OUTSIDE_SCOPE_SEPARATE_LOCAL_ARTIFACT_GATE_REQUIRED"
        ),
        "release_readiness_proven_by_this_report": False,
    }
    if value["scope"] != expected_scope:
        raise EvidenceError("dependency evidence scope mismatch")
    native_sync = value["native_sync"]
    if not isinstance(native_sync, dict):
        raise EvidenceError("dependency evidence native_sync must be an object")
    _expect_keys(
        native_sync,
        {
            "receipt_version",
            "receipt_semantic_sha256",
            "receipt_step",
            "command",
            "outcome",
            "claims",
            "validated",
        },
        label="dependency evidence native_sync",
    )
    if (
        native_sync["receipt_version"] != COMMAND_RECEIPT_VERSION
        or native_sync["receipt_step"]
        != {
            "ordinal": 1,
            "role": "native_sync_log",
            "kind": "log",
            "filename": "10_native_sync.log",
        }
        or native_sync["outcome"] != "PASSED"
        or native_sync["claims"] != {"exit_code": 0}
        or native_sync["validated"] is not True
    ):
        raise EvidenceError("dependency evidence native sync projection mismatch")
    if not isinstance(native_sync["command"], dict):
        raise EvidenceError("dependency evidence native sync command must be an object")
    _expect_keys(
        native_sync["command"], COMMAND_KEYS, label="dependency evidence native sync command"
    )
    lock = value["lock_reconciliation"]
    installed = value["installed_reconciliation"]
    pip_absence = value["pip_absence"]
    artifacts = value["artifact_evidence"]
    if not all(isinstance(item, dict) for item in (lock, installed, pip_absence, artifacts)):
        raise EvidenceError("dependency evidence reconciliation objects are invalid")
    _expect_keys(
        lock,
        {
            "expected_count",
            "observed_count",
            "malformed_count",
            "local_project",
            "missing",
            "extra",
            "duplicates",
            "exact_match",
        },
        label="dependency evidence lock_reconciliation",
    )
    for key in ("expected_count", "observed_count", "malformed_count"):
        _require_int(lock[key], label=f"dependency evidence lock_reconciliation.{key}")
    if not isinstance(lock["local_project"], dict):
        raise EvidenceError("dependency evidence lock local_project must be an object")
    _expect_keys(
        lock["local_project"],
        {"name", "version", "count"},
        label="dependency evidence lock local_project",
    )
    _require_int(
        lock["local_project"]["count"],
        label="dependency evidence lock local_project.count",
    )
    if not isinstance(lock["exact_match"], bool):
        raise EvidenceError("dependency evidence lock exact_match must be a boolean")
    _expect_keys(
        installed,
        {
            "expected_count",
            "third_party_expected_count",
            "installed_count",
            "local_project_identity_only",
            "installed",
            "missing",
            "extra",
            "version_mismatch",
            "exact_match",
        },
        label="dependency evidence installed_reconciliation",
    )
    for key in ("expected_count", "third_party_expected_count", "installed_count"):
        _require_int(
            installed[key],
            label=f"dependency evidence installed_reconciliation.{key}",
        )
    if not isinstance(installed["exact_match"], bool):
        raise EvidenceError("dependency evidence installed exact_match must be a boolean")
    for index, row in enumerate(installed["installed"]):
        if not isinstance(row, dict):
            raise EvidenceError("dependency evidence installed row must be an object")
        _require_int(
            row.get("metadata_size"),
            label=f"dependency evidence installed[{index}].metadata_size",
        )
    _expect_keys(
        pip_absence,
        {
            "export_absent",
            "lock_absent",
            "installed_distribution_absent",
            "find_spec_none",
            "site_package_paths",
            "site_package_matches",
            "bin_matches",
            "accepted",
        },
        label="dependency evidence pip_absence",
    )
    for key in (
        "export_absent",
        "lock_absent",
        "installed_distribution_absent",
        "find_spec_none",
        "accepted",
    ):
        if not isinstance(pip_absence[key], bool):
            raise EvidenceError(f"dependency evidence pip_absence.{key} must be a boolean")
    _expect_keys(
        artifacts,
        {
            "expected_artifact_count",
            "evidenced_artifact_count",
            "complete_raw_artifact_count",
            "wheelhouse_raw_artifact_count",
            "complete_raw_artifacts",
            "complete_wheelhouse",
            "pip_http_cache_used",
            "materialization_performed",
            "records",
            "missing",
            "invalid",
        },
        label="dependency evidence artifact_evidence",
    )
    for key in (
        "expected_artifact_count",
        "evidenced_artifact_count",
        "complete_raw_artifact_count",
        "wheelhouse_raw_artifact_count",
    ):
        _require_int(
            artifacts[key],
            label=f"dependency evidence artifact_evidence.{key}",
        )
    for key in (
        "expected_raw_artifact_count",
        "complete_raw_artifact_count",
        "wheelhouse_raw_artifact_count",
    ):
        _require_int(value[key], label=f"dependency evidence {key}")
    if (
        value["expected_raw_artifact_count"] != artifacts.get("expected_artifact_count")
        or value["complete_raw_artifact_count"] != artifacts.get("complete_raw_artifact_count")
        or value["wheelhouse_raw_artifact_count"] != artifacts.get("wheelhouse_raw_artifact_count")
        or value["complete_wheelhouse"] is not artifacts.get("complete_wheelhouse")
        or value["missing"] != artifacts.get("missing")
        or value["invalid"] != artifacts.get("invalid")
        or artifacts.get("pip_http_cache_used") is not False
        or artifacts.get("materialization_performed") is not False
    ):
        raise EvidenceError("dependency evidence artifact projection mismatch")
    expected_accepted = bool(
        lock.get("exact_match") is True
        and installed.get("exact_match") is True
        and pip_absence.get("accepted") is True
        and not value["invalid"]
    )
    if (
        value["accepted"] is not expected_accepted
        or value["dependency_environment_accepted"] is not expected_accepted
        or value["native_dependency_environment_accepted"] is not expected_accepted
        or value["status"]
        != (
            "THIRD_PARTY_NATIVE_ENVIRONMENT_ACCEPTED"
            if expected_accepted
            else "THIRD_PARTY_NATIVE_ENVIRONMENT_REJECTED"
        )
    ):
        raise EvidenceError("dependency evidence acceptance semantics mismatch")
    expected_failures = {
        "native_sync_receipt_invalid": False,
        "lock_reconciliation_mismatch": lock.get("exact_match") is not True,
        "installed_environment_mismatch": installed.get("exact_match") is not True,
        "native_pip_present": pip_absence.get("accepted") is not True,
        "invalid_artifact_evidence": bool(value["invalid"]),
    }
    if value["failure_reasons"] != expected_failures:
        raise EvidenceError("dependency evidence failure reasons mismatch")
    semantic = _require_sha256(
        value["semantic_sha256"], label="dependency evidence semantic_sha256"
    )
    semantic_body = dict(value)
    semantic_body.pop("semantic_sha256")
    if semantic != _sha256_bytes(_canonical_bytes(semantic_body)):
        raise EvidenceError("dependency evidence semantic SHA-256 mismatch")


def build_evidence(
    *,
    repo_root: Path,
    frozen_export: Path,
    target_venv: Path,
    target_python: Path,
    wheelhouse: Path,
    uv_cache: Path,
    uv_binary: str,
    native_sync_log: Path,
    work_root: Path,
) -> tuple[dict[str, Any], bool]:
    repo_root = repo_root.resolve(strict=True)
    if str(repo_root) != "/private/tmp/myquant-v17-neutral-baseline-20260722":
        raise EvidenceError("repo root does not match the approved candidate worktree")
    frozen_export = frozen_export.resolve(strict=True)
    work_root = work_root.absolute()
    work_root_identity = _directory_identity(
        work_root, label="Phase 0 work root", owner_private=True
    )
    target_venv = target_venv.absolute()
    target_python = target_python.absolute()
    if target_venv != work_root / "native_venv":
        raise EvidenceError("target venv must be <work-root>/native_venv")
    wheelhouse = wheelhouse.absolute()
    wheelhouse_security = _private_directory_binding(wheelhouse, label="wheelhouse")
    resolved_wheelhouse = wheelhouse.resolve(strict=True)
    if resolved_wheelhouse != wheelhouse:
        raise EvidenceError("wheelhouse path must not contain symlink indirection")
    wheelhouse = resolved_wheelhouse
    try:
        wheelhouse.relative_to(work_root)
    except ValueError as exc:
        raise EvidenceError("wheelhouse must be inside the Phase 0 work root") from exc
    uv_cache = uv_cache.resolve(strict=True)
    if str(uv_cache) != UV_CACHE_PATH:
        raise EvidenceError("uv cache path does not match the frozen global cache")
    native_sync_log = native_sync_log.absolute()
    bundle_root = native_sync_log.parent
    _directory_identity(bundle_root, label="Phase 0 bundle root", owner_private=True)
    if _path_is_within(bundle_root, repo_root) or _path_is_within(work_root, repo_root):
        raise EvidenceError("Phase 0 bundle/work roots must be outside the repository")
    if _path_is_within(bundle_root, work_root) or _path_is_within(work_root, bundle_root):
        raise EvidenceError("Phase 0 bundle and work roots must be distinct and non-nested")

    source_state_before = _git_snapshot(repo_root)
    source_binding = _canonical_command_source_binding(repo_root)
    protected_roots_before = _sample_protected_roots()
    toolchain_before = _observed_toolchain(uv_binary, uv_cache, repo_root=repo_root)
    receipt, native_log_binding = _parse_native_sync_log(
        native_sync_log,
        repo_root=repo_root,
        work_root=work_root,
        target_venv=target_venv,
        source=source_binding,
        toolchain=toolchain_before,
        protected_roots=protected_roots_before,
    )
    session_path = Path(str(receipt["session_binding"]["path"]))
    if session_path.parent != bundle_root or session_path.name != "00_session.json":
        raise EvidenceError("native receipt session must bind bundle/00_session.json")
    pyproject_path = repo_root / "pyproject.toml"
    lock_path = repo_root / "uv.lock"
    pyproject_raw, pyproject_stat = _stable_file_bytes(pyproject_path)
    lock_raw, lock_stat = _stable_file_bytes(lock_path)
    export_raw, export_stat = _stable_file_bytes(frozen_export)
    probe = _target_probe(target_python)
    venv_binding = _validate_target_venv(target_venv, target_python, probe)

    marker_environment = probe.get("marker_environment")
    supported_tags = probe.get("supported_tags")
    installed = probe.get("installed")
    if not isinstance(marker_environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in marker_environment.items()
    ):
        raise EvidenceError("target marker environment is invalid")
    if not isinstance(supported_tags, list) or not all(
        isinstance(item, str) for item in supported_tags
    ):
        raise EvidenceError("target supported tags are invalid")
    if not supported_tags:
        raise EvidenceError("target supported tags are empty")
    if not isinstance(installed, list):
        raise EvidenceError("target installed inventory is invalid")

    expected, inactive = _parse_frozen_export(export_raw, marker_environment)
    project_identity = _load_project_identity(pyproject_raw)
    reconciliation = _reconcile_installed(expected, project_identity, installed)
    lock_packages = _lock_packages(lock_raw)
    lock_reconciliation = _reconcile_lock(
        [*expected, *inactive],
        project_identity,
        lock_packages,
    )
    pip_absence = _pip_absence(
        active=expected,
        inactive=inactive,
        lock_packages=lock_packages,
        installed=installed,
        probe=probe,
    )
    artifacts = _artifact_evidence(
        expected_dependencies=expected,
        lock_packages=lock_packages,
        supported_tags=supported_tags,
        wheelhouse=wheelhouse,
    )
    wheelhouse_security_after = _private_directory_binding(wheelhouse, label="wheelhouse")
    if wheelhouse_security_after != wheelhouse_security:
        raise EvidenceError("wheelhouse identity or permissions changed during collection")
    probe_after = _target_probe(target_python)
    if _canonical_bytes(probe_after) != _canonical_bytes(probe):
        raise EvidenceError("target Python environment changed during collection")
    venv_binding_after = _validate_target_venv(target_venv, target_python, probe_after)
    if venv_binding_after != venv_binding:
        raise EvidenceError("target Python identity changed during collection")
    work_root_after = _directory_identity(work_root, label="Phase 0 work root", owner_private=True)
    if work_root_after != work_root_identity:
        raise EvidenceError("Phase 0 work root changed during collection")
    toolchain_after = _observed_toolchain(uv_binary, uv_cache, repo_root=repo_root)
    if toolchain_after != toolchain_before:
        raise EvidenceError("toolchain changed during evidence collection")
    protected_roots_after = _sample_protected_roots()
    if protected_roots_after != protected_roots_before:
        raise EvidenceError("protected roots changed during evidence collection")
    source_state_after = _git_snapshot(repo_root)
    _assert_source_state_stable(source_state_before, source_state_after)
    if _canonical_command_source_binding(repo_root) != source_binding:
        raise EvidenceError("canonical Phase 0 source binding changed during collection")

    gate_assessment = _assess_dependency_environment(
        native_sync_receipt_valid=True,
        lock_reconciliation=lock_reconciliation,
        reconciliation=reconciliation,
        pip_absence=pip_absence,
        artifacts=artifacts,
    )
    accepted = bool(gate_assessment["accepted"])
    status = str(gate_assessment["status"])
    producer_raw, _producer_stat = _stable_file_bytes(Path(__file__).resolve(strict=True))
    package_source_superset = receipt["package_source_superset_after"]
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "accepted": accepted,
        "dependency_environment_accepted": accepted,
        "native_dependency_environment_accepted": gate_assessment[
            "native_dependency_environment_accepted"
        ],
        "complete_wheelhouse": gate_assessment["complete_wheelhouse"],
        "offline_only": True,
        "network_actions_performed": False,
        "repackaged_artifacts": False,
        "session_binding": dict(receipt["session_binding"]),
        "step": {
            "ordinal": 2,
            "role": "native_sync_receipt",
            "kind": "artifact",
            "filename": "20_native_dependency.json",
        },
        "producer": {
            "path": str(Path(__file__).resolve(strict=True)),
            "version": SCHEMA_VERSION,
            "sha256": _sha256_bytes(producer_raw),
            "size_bytes": len(producer_raw),
        },
        "native_sync_log": native_log_binding,
        "source": source_binding,
        "toolchain": toolchain_before,
        "package_source_superset": package_source_superset,
        "protected_roots": protected_roots_before,
        "limitations": NORMATIVE_LIMITATIONS,
        "scope": {
            "kind": "third_party_dependency_environment_only",
            "local_project_identity_checked_for_environment_exactness": True,
            "local_project_artifact_provenance_verified": False,
            "local_project_wheel_sdist_source_state_release_binding": (
                "OUTSIDE_SCOPE_SEPARATE_LOCAL_ARTIFACT_GATE_REQUIRED"
            ),
            "release_readiness_proven_by_this_report": False,
        },
        "inputs": {
            "pyproject": _file_binding_from_bytes(pyproject_path, pyproject_raw, pyproject_stat),
            "uv_lock": _file_binding_from_bytes(lock_path, lock_raw, lock_stat),
            "frozen_no_hash_export": _file_binding_from_bytes(
                frozen_export, export_raw, export_stat
            ),
            "wheelhouse": wheelhouse_security,
            "work_root": work_root_identity,
            "bundle_root": _directory_identity(
                bundle_root, label="Phase 0 bundle root", owner_private=True
            ),
        },
        "runtime": {
            "target_venv": venv_binding,
            "python": probe["python"],
            "platform": probe["platform"],
            "marker_environment": marker_environment,
            "supported_tags_sha256": _sha256_bytes(_canonical_bytes(supported_tags)),
            "supported_tag_count": len(supported_tags),
        },
        "native_sync": {
            "receipt_version": receipt["version"],
            "receipt_semantic_sha256": receipt["semantic_sha256"],
            "receipt_step": dict(receipt["step"]),
            "command": dict(receipt["commands"][0]),
            "outcome": receipt["outcome"],
            "claims": dict(receipt["claims"]),
            "validated": True,
        },
        "local_project_identity_for_environment_reconciliation": project_identity,
        "expected_dependencies": expected,
        "inactive_marker_dependencies": inactive,
        "lock_reconciliation": lock_reconciliation,
        "installed_reconciliation": reconciliation,
        "pip_absence": pip_absence,
        "artifact_evidence": artifacts,
        "expected_raw_artifact_count": artifacts["expected_artifact_count"],
        "complete_raw_artifact_count": artifacts["complete_raw_artifact_count"],
        "wheelhouse_raw_artifact_count": artifacts["wheelhouse_raw_artifact_count"],
        "missing": artifacts["missing"],
        "invalid": artifacts["invalid"],
        "failure_reasons": gate_assessment["failure_reasons"],
    }
    body["semantic_sha256"] = _sha256_bytes(_canonical_bytes(body))
    _validate_evidence_v2(body)
    return body, accepted


def publish_evidence(path: Path, report: Mapping[str, Any]) -> None:
    """Validate and exact-once publish one dependency evidence v2 artifact."""

    _validate_evidence_v2(report)
    _write_private_json(path, report)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--frozen-export", required=True)
    parser.add_argument("--target-venv", required=True)
    parser.add_argument("--target-python", required=True)
    parser.add_argument(
        "--work-root",
        required=True,
        help="Never-reused owner-private 0700 Phase 0 work root.",
    )
    parser.add_argument(
        "--wheelhouse",
        required=True,
        help=(
            "Owner-private 0700 retained-artifact directory inside work root; "
            "present artifacts must be mode 0600. Incompleteness is non-authorizing "
            "and non-blocking only after the bound native sync succeeds."
        ),
    )
    parser.add_argument("--uv-cache", required=True)
    parser.add_argument("--uv-binary", required=True)
    parser.add_argument(
        "--native-sync-log",
        required=True,
        help=(
            "Authoritative 10_native_sync.log containing the canonical "
            "phase0-command-receipt.v2 line and binary stdout/stderr frame."
        ),
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        repo_root = Path(args.repo_root).resolve(strict=True)
        output_path = _prepare_private_output_path(Path(args.output_json), repo_root)
        native_sync_log = Path(args.native_sync_log).absolute()
        if output_path.name != "20_native_dependency.json":
            raise EvidenceError("output filename must be 20_native_dependency.json")
        if output_path.parent != native_sync_log.parent:
            raise EvidenceError("native sync log and dependency output must share bundle root")
        report, accepted = build_evidence(
            repo_root=repo_root,
            frozen_export=Path(args.frozen_export),
            target_venv=Path(args.target_venv),
            target_python=Path(args.target_python),
            work_root=Path(args.work_root),
            wheelhouse=Path(args.wheelhouse),
            uv_cache=Path(args.uv_cache),
            uv_binary=args.uv_binary,
            native_sync_log=native_sync_log,
        )
        publish_evidence(output_path, report)
    except (EvidenceError, OSError, ValueError) as exc:
        print(f"v17 dependency evidence failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"accepted": accepted, "status": report["status"]}, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
