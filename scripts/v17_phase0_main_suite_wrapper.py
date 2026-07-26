#!/usr/bin/env python3
"""Run the one main-runtime Phase 0 pytest suite with pipe attestations.

This file is executed only by the fixed main-worktree virtual-environment
interpreter.  It starts with ``-I -S -B -X pycache_prefix=...`` so no ``.pth``
code runs before the parent challenge is consumed and bytecode stays isolated.
After the inherited descriptors are marked close-on-exec, it explicitly
restores the hash-bound virtual-environment site startup, replaces only the
authority-repository source path with the candidate root in memory, and emits
three bounded attestations outside stdout/stderr.
"""

from __future__ import annotations

import fcntl
import os
import struct
import sys
from types import ModuleType
from typing import Any, Literal, Sequence, overload

CHALLENGE_ENV = "MYQUANT_PHASE0_CHALLENGE_FD"
ATTEST_ENV = "MYQUANT_PHASE0_ATTEST_FD"
CHALLENGE_MAGIC = b"MQP0CH01"
ATTEST_MAGIC = b"MQP0AT01"
PROTOCOL_VERSION = 1
CHALLENGE_STRUCT = struct.Struct(">8sB3s32s32s")
ATTEST_HEADER = struct.Struct(">8sBBHI32s32s")
MAX_FRAME_BYTES = 1024 * 1024
MAX_TERMINAL_FRAME_BYTES = 16 * 1024
POLICY_VERSION = "myquant.v17.v2.phase0-main-suite-runtime-policy.v1"
DISCOVERY_FINAL_PREFIX = "MYQUANT_PHASE0_DISCOVERY_FINAL="
WRAPPER_FAILURE_EXIT = 86
CANDIDATE_CONTENT_BINDING = "OUTER_SOURCE_STATE"
PATH_TOPOLOGY = {
    "cache_children": [
        "BLACK_CACHE_DIR",
        "MYPY_CACHE_DIR",
        "PYTHONPYCACHEPREFIX",
    ],
    "closed_root_siblings": ["HOME", "TMPDIR", "XDG_CACHE_HOME"],
    "must_remain_empty": ["PYTHONPYCACHEPREFIX"],
}


class MainSuiteWrapperError(RuntimeError):
    """Raised when the fixed main-suite runtime cannot be attested."""


def _fd_from_environment(name: str) -> int:
    raw = os.environ.pop(name, None)
    if raw is None or not raw.isascii() or not raw.isdecimal():
        raise MainSuiteWrapperError(f"{name} is missing or invalid")
    descriptor = int(raw)
    if descriptor < 3:
        raise MainSuiteWrapperError(f"{name} is not a private descriptor")
    try:
        flags = fcntl.fcntl(descriptor, fcntl.F_GETFD)
        fcntl.fcntl(descriptor, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)
    except OSError as exc:
        raise MainSuiteWrapperError(f"{name} is unavailable") from exc
    return descriptor


def _read_exact(descriptor: int, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            raise MainSuiteWrapperError("challenge is truncated")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_challenge(descriptor: int) -> tuple[bytes, bytes]:
    raw = _read_exact(descriptor, CHALLENGE_STRUCT.size)
    if os.read(descriptor, 1):
        raise MainSuiteWrapperError("challenge has trailing bytes")
    magic, version, reserved, nonce, challenge_sha = CHALLENGE_STRUCT.unpack(raw)
    if magic != CHALLENGE_MAGIC or version != PROTOCOL_VERSION or reserved != b"\0\0\0":
        raise MainSuiteWrapperError("challenge header is invalid")
    return nonce, challenge_sha


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise MainSuiteWrapperError("attestation write failed")
        view = view[written:]


def _canonical_bytes(value: object) -> bytes:
    import json

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise MainSuiteWrapperError("attestation is not canonical JSON") from exc


def _sha256(raw: bytes) -> str:
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def _semantic_sha256(value: dict[str, object]) -> str:
    body = dict(value)
    body.pop("semantic_sha256", None)
    return _sha256(_canonical_bytes(body))


def _strict_policy(path: str, expected_sha256: str) -> tuple[dict[str, object], bytes]:
    import json

    binding, raw = _stable_file_binding(path, max_bytes=4 * 1024 * 1024, include_raw=True)
    if binding["uid"] != os.getuid() or binding["sha256"] != expected_sha256:
        raise MainSuiteWrapperError("runtime policy file binding mismatch")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MainSuiteWrapperError("runtime policy is invalid JSON") from exc
    if (
        type(value) is not dict
        or raw != _canonical_bytes(value) + b"\n"
        or value.get("version") != POLICY_VERSION
        or value.get("protocol_version") != "myquant.v17.v2"
        or value.get("status") != "FROZEN"
        or value.get("authority") is not False
        or value.get("semantic_sha256") != _semantic_sha256(value)
    ):
        raise MainSuiteWrapperError("runtime policy contract mismatch")
    return value, raw


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise MainSuiteWrapperError("JSON contains duplicate keys")
        result[key] = value
    return result


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


@overload
def _stable_file_binding(
    path: str,
    *,
    max_bytes: int = 512 * 1024 * 1024,
    include_raw: Literal[False] = False,
) -> dict[str, object]: ...


@overload
def _stable_file_binding(
    path: str,
    *,
    max_bytes: int = 512 * 1024 * 1024,
    include_raw: Literal[True],
) -> tuple[dict[str, object], bytes]: ...


def _stable_file_binding(
    path: str,
    *,
    max_bytes: int = 512 * 1024 * 1024,
    include_raw: bool = False,
) -> dict[str, object] | tuple[dict[str, object], bytes]:
    import hashlib
    import stat

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise MainSuiteWrapperError(f"cannot open bound file: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise MainSuiteWrapperError(f"bound file identity is unsafe: {path}")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise MainSuiteWrapperError(f"bound file exceeds limit: {path}")
            digest.update(chunk)
            if include_raw:
                chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise MainSuiteWrapperError(f"bound file path disappeared: {path}") from exc
    if (
        _stat_signature(before) != _stat_signature(after)
        or _stat_signature(after) != _stat_signature(path_after)
        or size != before.st_size
    ):
        raise MainSuiteWrapperError(f"bound file drifted: {path}")
    binding = {
        "gid": before.st_gid,
        "mode": f"{before.st_mode & 0o7777:04o}",
        "path": path,
        "sha256": digest.hexdigest(),
        "size_bytes": size,
        "st_dev": before.st_dev,
        "st_ino": before.st_ino,
        "st_nlink": before.st_nlink,
        "uid": before.st_uid,
    }
    if include_raw:
        return binding, b"".join(chunks)
    return binding


def _stable_symlink_binding(path: str) -> dict[str, object]:
    import stat

    try:
        before = os.lstat(path)
        target = os.readlink(path)
        after = os.lstat(path)
    except OSError as exc:
        raise MainSuiteWrapperError(f"cannot bind symlink: {path}") from exc
    if (
        not stat.S_ISLNK(before.st_mode)
        or _stat_signature(before) != _stat_signature(after)
        or before.st_uid != os.getuid()
    ):
        raise MainSuiteWrapperError(f"symlink identity drifted: {path}")
    return {
        "gid": before.st_gid,
        "link_text": target,
        "mode": f"{before.st_mode & 0o7777:04o}",
        "path": path,
        "size_bytes": before.st_size,
        "st_dev": before.st_dev,
        "st_ino": before.st_ino,
        "st_nlink": before.st_nlink,
        "uid": before.st_uid,
    }


def _assert_binding(
    observed: dict[str, object],
    expected: object,
    *,
    label: str,
) -> None:
    if type(expected) is not dict:
        raise MainSuiteWrapperError(f"{label} policy binding is invalid")
    for key in (
        "path",
        "sha256",
        "size_bytes",
        "mode",
        "st_dev",
        "st_ino",
        "st_nlink",
        "uid",
        "gid",
    ):
        if key in expected and observed.get(key) != expected.get(key):
            raise MainSuiteWrapperError(f"{label} binding mismatch: {key}")


def _assert_symlink_binding(
    observed: dict[str, object],
    expected: object,
    *,
    label: str,
) -> None:
    if type(expected) is not dict:
        raise MainSuiteWrapperError(f"{label} policy binding is invalid")
    for key in (
        "path",
        "link_text",
        "size_bytes",
        "mode",
        "st_dev",
        "st_ino",
        "st_nlink",
        "uid",
        "gid",
    ):
        if observed.get(key) != expected.get(key):
            raise MainSuiteWrapperError(f"{label} binding mismatch: {key}")


def _runtime_state() -> dict[str, object]:
    return {
        "sys_base_prefix": sys.base_prefix,
        "sys_executable": sys.executable,
        "sys_exec_prefix": sys.exec_prefix,
        "sys_path": list(sys.path),
        "sys_prefix": sys.prefix,
        "version_info": list(sys.version_info[:3]),
    }


def _bytecode_policy() -> dict[str, object]:
    from pathlib import Path
    import stat

    expected = os.environ.get("PYTHONPYCACHEPREFIX")
    if (
        type(expected) is not str
        or not Path(expected).is_absolute()
        or os.path.normpath(expected) != expected
        or sys.pycache_prefix != expected
        or sys.dont_write_bytecode is not True
    ):
        raise MainSuiteWrapperError("isolated bytecode policy is not active")
    path = Path(expected)
    try:
        observed = path.lstat()
        with os.scandir(path) as entries:
            if next(entries, None) is not None:
                raise MainSuiteWrapperError("isolated pycache directory is not empty")
    except MainSuiteWrapperError:
        raise
    except OSError as exc:
        raise MainSuiteWrapperError("isolated pycache directory is unavailable") from exc
    if (
        not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != os.getuid()
        or stat.S_IMODE(observed.st_mode) != 0o700
    ):
        raise MainSuiteWrapperError("isolated pycache directory is not owner-private")
    return {
        "dont_write_bytecode": True,
        "pycache_prefix": expected,
    }


def _assert_startup_surfaces(policy: dict[str, object]) -> dict[str, object]:
    from pathlib import Path

    main_runtime = policy.get("main_runtime")
    if type(main_runtime) is not dict:
        raise MainSuiteWrapperError("policy main_runtime is invalid")
    if sys.flags.isolated != 1 or sys.flags.no_site != 1 or sys.flags.dont_write_bytecode != 1:
        raise MainSuiteWrapperError("main wrapper must use -I -S -B")
    expected_flags = main_runtime.get("interpreter_flags")
    observed_flags = {
        "dont_write_bytecode": sys.flags.dont_write_bytecode,
        "isolated": sys.flags.isolated,
        "no_site": sys.flags.no_site,
    }
    if observed_flags != expected_flags:
        raise MainSuiteWrapperError("isolated interpreter flags drift")
    lexical_python = main_runtime.get("lexical_python")
    if type(lexical_python) is not str or sys.executable != lexical_python:
        raise MainSuiteWrapperError("main lexical interpreter path drift")
    lexical = _stable_symlink_binding(lexical_python)
    _assert_symlink_binding(
        lexical,
        main_runtime.get("lexical_python_binding"),
        label="main lexical interpreter",
    )
    resolved_path = str(Path(lexical_python).resolve(strict=True))
    resolved = _stable_file_binding(resolved_path)
    if type(resolved) is not dict:  # pragma: no cover - typing guard
        raise MainSuiteWrapperError("resolved interpreter binding is invalid")
    _assert_binding(
        resolved,
        main_runtime.get("resolved_python_binding"),
        label="main resolved interpreter",
    )
    wrapper = _stable_file_binding(str(Path(__file__).resolve(strict=True)))
    if type(wrapper) is not dict:  # pragma: no cover - typing guard
        raise MainSuiteWrapperError("wrapper binding is invalid")
    _assert_binding(wrapper, policy.get("wrapper_binding"), label="wrapper")
    startup_rows = main_runtime.get("startup_files")
    if type(startup_rows) is not list or not startup_rows:
        raise MainSuiteWrapperError("startup file policy is empty")
    observed_rows: list[dict[str, object]] = []
    for index, row in enumerate(startup_rows):
        if type(row) is not dict or type(row.get("path")) is not str:
            raise MainSuiteWrapperError(f"startup file row {index} is invalid")
        path = str(row["path"])
        present = row.get("present")
        if present is True:
            observed = _stable_file_binding(path)
            if type(observed) is not dict:  # pragma: no cover - typing guard
                raise MainSuiteWrapperError("startup file binding is invalid")
            _assert_binding(observed, row, label=f"startup file {index}")
            observed_rows.append({"path": path, "present": True, **observed})
        elif present is False:
            try:
                os.lstat(path)
            except FileNotFoundError:
                observed_rows.append({"path": path, "present": False})
            except OSError as exc:
                raise MainSuiteWrapperError(f"cannot verify absent startup file: {path}") from exc
            else:
                raise MainSuiteWrapperError(f"startup file unexpectedly exists: {path}")
        else:
            raise MainSuiteWrapperError(f"startup file row {index} lacks presence")
    before = _runtime_state()
    if before != main_runtime.get("pre_site_state"):
        raise MainSuiteWrapperError("pre-site interpreter state drift")
    return {
        "lexical_python": lexical,
        "resolved_python": resolved,
        "startup_files": observed_rows,
        "wrapper": wrapper,
    }


def _restore_site(policy: dict[str, object]) -> dict[str, object]:
    startup = _assert_startup_surfaces(policy)

    import importlib
    import importlib.util
    import site
    from pathlib import Path

    main_runtime = policy["main_runtime"]
    if type(main_runtime) is not dict:  # pragma: no cover - validated above
        raise MainSuiteWrapperError("policy main_runtime is invalid")
    site.main()
    site_started = _runtime_state()
    if site_started != main_runtime.get("post_site_state"):
        raise MainSuiteWrapperError("site-started interpreter state drift")
    expected_modules = main_runtime.get("startup_modules")
    if type(expected_modules) is not list:
        raise MainSuiteWrapperError("startup module policy is invalid")
    startup_modules: list[dict[str, object]] = []
    for index, row in enumerate(expected_modules):
        if type(row) is not dict or type(row.get("module")) is not str:
            raise MainSuiteWrapperError(f"startup module row {index} is invalid")
        module = sys.modules.get(str(row["module"]))
        origin = None if module is None else getattr(module, "__file__", None)
        if not isinstance(origin, str):
            raise MainSuiteWrapperError(f"startup module was not loaded: {row['module']}")
        binding = _stable_file_binding(str(Path(origin).resolve(strict=True)))
        if type(binding) is not dict:  # pragma: no cover - typing guard
            raise MainSuiteWrapperError("startup module binding is invalid")
        _assert_binding(binding, row, label=f"startup module {row['module']}")
        startup_modules.append({"module": row["module"], **binding})

    module_policy = policy.get("module_policy")
    if type(module_policy) is not dict or type(module_policy.get("authority_root")) is not str:
        raise MainSuiteWrapperError("authority module root policy is invalid")
    authority = Path(str(module_policy["authority_root"])).resolve(strict=True)
    candidate = Path(str(policy["candidate_root"])).resolve(strict=True)
    site_packages = Path(str(main_runtime["site_packages_root"])).resolve(strict=True)
    wrapper_candidate = Path(__file__).resolve(strict=True).parent.parent
    if candidate != wrapper_candidate or candidate == authority:
        raise MainSuiteWrapperError("candidate root derivation mismatch")
    sanitized: list[str] = []
    removed: list[str] = []
    for item in sys.path:
        if not item:
            sanitized.append(item)
            continue
        try:
            resolved = Path(item).resolve(strict=False)
        except OSError:
            sanitized.append(item)
            continue
        if _within(resolved, authority) and not _within(resolved, site_packages):
            removed.append(item)
        else:
            sanitized.append(item)
    routing_policy = policy.get("routing")
    if type(routing_policy) is not dict:
        raise MainSuiteWrapperError("routing policy is invalid")
    if removed != routing_policy.get("removed_authority_entries"):
        raise MainSuiteWrapperError("authority path removal drift")
    candidate_text = str(candidate)
    sanitized = [
        item
        for item in sanitized
        if not item or not _within(Path(item).resolve(strict=False), candidate)
    ]
    sys.path[:] = [candidate_text, *sanitized]
    if list(sys.path) != routing_policy.get("sanitized_sys_path"):
        raise MainSuiteWrapperError("sanitized sys.path drift")
    importlib.invalidate_caches()
    spec = importlib.util.find_spec("quant_investor")
    origin = None if spec is None else spec.origin
    if origin is None or not _within(Path(origin), candidate):
        raise MainSuiteWrapperError("candidate quant_investor route is not authoritative")
    expected_origin = routing_policy.get("quant_investor_origin")
    resolved_origin = str(Path(origin).resolve(strict=True))
    if resolved_origin != expected_origin:
        raise MainSuiteWrapperError("candidate quant_investor origin drift")
    return {
        "candidate_root": candidate_text,
        "quant_investor_origin": resolved_origin,
        "removed_authority_entries": removed,
        "runtime_state": _runtime_state(),
        "startup": startup,
        "startup_modules": startup_modules,
    }


def _within(path: object, root: object) -> bool:
    from pathlib import Path

    try:
        Path(str(path)).resolve(strict=False).relative_to(Path(str(root)).resolve(strict=False))
    except (OSError, ValueError):
        return False
    return True


def _validate_pytest_environment(policy: dict[str, object]) -> dict[str, str]:
    expected = policy.get("pytest_environment")
    if type(expected) is not dict:
        raise MainSuiteWrapperError("pytest environment policy is invalid")
    required = expected.get("required")
    forbidden = expected.get("forbidden")
    allowed_keys = expected.get("allowed_keys")
    dynamic_path_keys = expected.get("dynamic_path_keys")
    if (
        type(required) is not dict
        or type(forbidden) is not list
        or type(allowed_keys) is not list
        or type(dynamic_path_keys) is not list
        or expected.get("path_topology") != PATH_TOPOLOGY
    ):
        raise MainSuiteWrapperError("pytest environment policy is incomplete")
    if set(os.environ) != set(allowed_keys):
        missing = sorted(set(allowed_keys) - set(os.environ))
        extra = sorted(set(os.environ) - set(allowed_keys))
        raise MainSuiteWrapperError(
            f"pytest environment key set drift: missing={missing!r} extra={extra!r}"
        )
    for key, value in required.items():
        if type(key) is not str or type(value) is not str or os.environ.get(key) != value:
            raise MainSuiteWrapperError(f"required pytest environment drift: {key}")
    for key in forbidden:
        if type(key) is not str or key in os.environ:
            raise MainSuiteWrapperError(f"forbidden pytest environment is set: {key}")
    from pathlib import Path

    for key in dynamic_path_keys:
        value = os.environ.get(str(key))
        if (
            not isinstance(value, str)
            or not Path(value).is_absolute()
            or os.path.normpath(value) != value
        ):
            raise MainSuiteWrapperError(f"dynamic pytest path is invalid: {key}")
    home = Path(os.environ["HOME"])
    tmpdir = Path(os.environ["TMPDIR"])
    cache = Path(os.environ["XDG_CACHE_HOME"])
    if (
        home.parent != tmpdir.parent
        or home.parent != cache.parent
        or Path(os.environ["BLACK_CACHE_DIR"]).parent != cache
        or Path(os.environ["MYPY_CACHE_DIR"]).parent != cache
        or Path(os.environ["PYTHONPYCACHEPREFIX"]).parent != cache
    ):
        raise MainSuiteWrapperError("dynamic pytest path topology drift")
    if os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") != "1":
        raise MainSuiteWrapperError("pytest plugin autoload must be disabled")
    return {key: os.environ[key] for key in sorted(os.environ)}


def _distribution_inventory() -> dict[str, object]:
    import importlib.metadata
    import re
    from pathlib import Path

    rows: list[list[str]] = []
    seen: set[str] = set()
    for distribution in importlib.metadata.distributions():
        metadata: Any = distribution.metadata
        name = metadata.get("Name")
        if not name:
            continue
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if normalized in seen:
            raise MainSuiteWrapperError("duplicate normalized distribution")
        seen.add(normalized)
        rows.append([normalized, distribution.version])
    rows.sort(key=lambda row: (row[0].encode("utf-8"), row[1].encode("utf-8")))
    site_packages = next(
        (
            Path(item)
            for item in sys.path
            if item.endswith("/site-packages") and Path(item).is_dir()
        ),
        None,
    )
    if site_packages is None:
        raise MainSuiteWrapperError("main site-packages path is unavailable")
    physical_names = sorted(
        (
            child.name
            for child in site_packages.iterdir()
            if child.name.endswith(".dist-info") and child.is_dir()
        ),
        key=lambda value: value.encode("utf-8"),
    )
    return {
        "count": len(rows),
        "physical_dist_info_count": len(physical_names),
        "physical_dist_info_names_sha256": _sha256(_canonical_bytes(physical_names)),
        "rows_sha256": _sha256(_canonical_bytes(rows)),
    }


def _invalid_dist_info_inventory(policy: dict[str, object]) -> list[dict[str, object]]:
    import stat
    from pathlib import Path

    main_runtime = policy.get("main_runtime")
    rows = None if type(main_runtime) is not dict else main_runtime.get("invalid_dist_info")
    if type(rows) is not list:
        raise MainSuiteWrapperError("invalid dist-info policy is invalid")
    observed_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if (
            type(row) is not dict
            or type(row.get("path")) is not str
            or type(row.get("child_names")) is not list
            or type(row.get("files")) is not list
        ):
            raise MainSuiteWrapperError(f"invalid dist-info row {index} is invalid")
        root = Path(str(row["path"]))
        try:
            before = root.lstat()
            names = sorted(
                (child.name for child in root.iterdir()),
                key=lambda value: value.encode("utf-8"),
            )
            after = root.lstat()
        except OSError as exc:
            raise MainSuiteWrapperError("invalid dist-info stub is unavailable") from exc
        if (
            not stat.S_ISDIR(before.st_mode)
            or _stat_signature(before) != _stat_signature(after)
            or names != row["child_names"]
        ):
            raise MainSuiteWrapperError("invalid dist-info stub drift")
        files: list[dict[str, object]] = []
        for expected_file in row["files"]:
            if type(expected_file) is not dict or type(expected_file.get("path")) is not str:
                raise MainSuiteWrapperError("invalid dist-info file row is invalid")
            binding = _stable_file_binding(str(expected_file["path"]))
            if type(binding) is not dict:  # pragma: no cover - typing guard
                raise MainSuiteWrapperError("invalid dist-info binding is invalid")
            _assert_binding(
                binding,
                expected_file,
                label=f"invalid dist-info file {expected_file['path']}",
            )
            files.append(binding)
        observed_rows.append(
            {
                "child_names": names,
                "files": files,
                "path": str(root),
            }
        )
    return observed_rows


def _tree_inventory(roots: object) -> dict[str, object]:
    import stat
    from pathlib import Path

    if type(roots) is not list or not roots:
        raise MainSuiteWrapperError("tree roots are invalid")
    rows: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    seen_casefold: set[str] = set()
    seen_file_identities: set[tuple[int, int]] = set()

    def add_row(relative: str, observed: os.stat_result, *, parent_fd: int, name: str) -> None:
        folded = relative.casefold()
        if relative in seen_paths or folded in seen_casefold:
            raise MainSuiteWrapperError("tree path/casefold collision")
        seen_paths.add(relative)
        seen_casefold.add(folded)
        if stat.S_ISDIR(observed.st_mode):
            rows.append(
                {
                    "kind": "directory",
                    "mode": stat.S_IMODE(observed.st_mode),
                    "path": relative,
                    "sha256": None,
                    "size_bytes": 0,
                }
            )
            return
        if not stat.S_ISREG(observed.st_mode):
            raise MainSuiteWrapperError("tree contains symlink or special entry")
        identity = (observed.st_dev, observed.st_ino)
        if observed.st_nlink != 1 or identity in seen_file_identities:
            raise MainSuiteWrapperError("tree contains a hardlinked file")
        seen_file_identities.add(identity)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(name, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise MainSuiteWrapperError(f"cannot open tree file: {relative}") from exc
        try:
            before = os.fstat(descriptor)
            if _stat_signature(before) != _stat_signature(observed):
                raise MainSuiteWrapperError(f"tree file identity drift: {relative}")
            import hashlib

            digest = hashlib.sha256()
            size = 0
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > 512 * 1024 * 1024:
                    raise MainSuiteWrapperError(f"tree file exceeds cap: {relative}")
                digest.update(chunk)
            after = os.fstat(descriptor)
            path_after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        finally:
            os.close(descriptor)
        if (
            _stat_signature(before) != _stat_signature(after)
            or _stat_signature(after) != _stat_signature(path_after)
            or size != before.st_size
        ):
            raise MainSuiteWrapperError(f"tree file drifted: {relative}")
        rows.append(
            {
                "kind": "file",
                "mode": stat.S_IMODE(before.st_mode),
                "path": relative,
                "sha256": digest.hexdigest(),
                "size_bytes": size,
            }
        )

    def scan_directory(descriptor: int, relative: str) -> None:
        try:
            names = sorted(os.listdir(descriptor), key=lambda value: value.encode("utf-8"))
        except OSError as exc:
            raise MainSuiteWrapperError(f"cannot enumerate tree directory: {relative}") from exc
        if len(names) != len(set(names)) or len(names) != len({name.casefold() for name in names}):
            raise MainSuiteWrapperError("tree directory contains name collision")
        for name in names:
            if not name or name in {".", ".."} or "/" in name or "\0" in name:
                raise MainSuiteWrapperError("tree directory contains invalid name")
            child_relative = f"{relative}/{name}"
            try:
                observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError as exc:
                raise MainSuiteWrapperError(f"cannot stat tree entry: {child_relative}") from exc
            add_row(child_relative, observed, parent_fd=descriptor, name=name)
            if stat.S_ISDIR(observed.st_mode):
                flags = (
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                try:
                    child_fd = os.open(name, flags, dir_fd=descriptor)
                except OSError as exc:
                    raise MainSuiteWrapperError(
                        f"cannot open tree directory: {child_relative}"
                    ) from exc
                try:
                    opened = os.fstat(child_fd)
                    if _stat_signature(opened) != _stat_signature(observed):
                        raise MainSuiteWrapperError(
                            f"tree directory identity drift: {child_relative}"
                        )
                    scan_directory(child_fd, child_relative)
                    after = os.fstat(child_fd)
                    path_after = os.stat(
                        name,
                        dir_fd=descriptor,
                        follow_symlinks=False,
                    )
                    if _stat_signature(opened) != _stat_signature(after) or _stat_signature(
                        after
                    ) != _stat_signature(path_after):
                        raise MainSuiteWrapperError(f"tree directory drifted: {child_relative}")
                finally:
                    os.close(child_fd)
        try:
            after_names = sorted(
                os.listdir(descriptor),
                key=lambda value: value.encode("utf-8"),
            )
        except OSError as exc:
            raise MainSuiteWrapperError(f"cannot re-enumerate tree directory: {relative}") from exc
        if names != after_names:
            raise MainSuiteWrapperError(f"tree directory names drifted: {relative}")

    for raw_root in roots:
        if type(raw_root) is not str:
            raise MainSuiteWrapperError("tree root path is invalid")
        root = Path(raw_root)
        try:
            root_stat = root.lstat()
        except OSError as exc:
            raise MainSuiteWrapperError(f"tree root is unavailable: {root}") from exc
        parent = root.parent
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            parent_fd = os.open(str(parent), flags)
        except OSError as exc:
            raise MainSuiteWrapperError(f"cannot open tree parent: {parent}") from exc
        try:
            add_row(root.name, root_stat, parent_fd=parent_fd, name=root.name)
            if stat.S_ISDIR(root_stat.st_mode):
                try:
                    root_fd = os.open(root.name, flags, dir_fd=parent_fd)
                except OSError as exc:
                    raise MainSuiteWrapperError(f"cannot open tree root: {root}") from exc
                try:
                    opened = os.fstat(root_fd)
                    if _stat_signature(opened) != _stat_signature(root_stat):
                        raise MainSuiteWrapperError(f"tree root identity drift: {root}")
                    scan_directory(root_fd, root.name)
                    after = os.fstat(root_fd)
                    path_after = os.stat(
                        root.name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                    if _stat_signature(opened) != _stat_signature(after) or _stat_signature(
                        after
                    ) != _stat_signature(path_after):
                        raise MainSuiteWrapperError(f"tree root drifted: {root}")
                finally:
                    os.close(root_fd)
        finally:
            os.close(parent_fd)
    rows.sort(key=lambda row: str(row["path"]).encode("utf-8"))
    file_rows = [row for row in rows if row["kind"] == "file"]
    byte_rows = [
        {
            "byte_sha256": row["sha256"],
            "relative_path": row["path"],
            "size_bytes": row["size_bytes"],
        }
        for row in file_rows
    ]
    return {
        "byte_inventory_sha256": _sha256(_canonical_bytes(byte_rows)),
        "directory_count": len(rows) - len(file_rows),
        "entry_count": len(rows),
        "file_count": len(file_rows),
        "total_file_bytes": sum(int(str(row["size_bytes"])) for row in file_rows),
        "tree_inventory_sha256": _sha256(_canonical_bytes(rows)),
    }


def _plugin_closure(
    policy: dict[str, object],
    manager: Any,
) -> list[dict[str, object]]:
    import importlib.metadata
    from pathlib import Path

    expected_rows = policy.get("pytest_plugins")
    if type(expected_rows) is not list:
        raise MainSuiteWrapperError("pytest plugin policy is invalid")
    listed = [
        (name, plugin)
        for name, plugin in manager.list_name_plugin()
        if plugin is not None
        and name in {str(row.get("entry_point_name")) for row in expected_rows}
    ]
    if [name for name, _plugin in listed] != [row.get("entry_point_name") for row in expected_rows]:
        raise MainSuiteWrapperError("pytest plugin registration order drift")
    observed_rows: list[dict[str, object]] = []
    for expected, (name, plugin) in zip(expected_rows, listed, strict=True):
        if type(expected) is not dict:
            raise MainSuiteWrapperError("pytest plugin row is invalid")
        module_name = getattr(plugin, "__name__", None)
        origin = getattr(plugin, "__file__", None)
        if module_name != expected.get("module") or not isinstance(origin, str):
            raise MainSuiteWrapperError(f"pytest plugin module drift: {name}")
        binding = _stable_file_binding(str(Path(origin).resolve(strict=True)))
        if type(binding) is not dict:  # pragma: no cover - typing guard
            raise MainSuiteWrapperError("pytest plugin binding is invalid")
        _assert_binding(binding, expected.get("module_file_binding"), label=f"plugin {name}")
        entry_binding = _stable_file_binding(str(expected.get("entry_points_path")))
        if type(entry_binding) is not dict:  # pragma: no cover - typing guard
            raise MainSuiteWrapperError("entry-points binding is invalid")
        _assert_binding(
            entry_binding,
            expected.get("entry_points_binding"),
            label=f"plugin entry-points {name}",
        )
        distribution = importlib.metadata.distribution(str(expected.get("distribution")))
        if distribution.version != expected.get("version"):
            raise MainSuiteWrapperError(f"pytest plugin distribution drift: {name}")
        entry_points = [
            entry
            for entry in distribution.entry_points
            if entry.group == "pytest11" and entry.name == name
        ]
        if len(entry_points) != 1 or entry_points[0].value != expected.get("value"):
            raise MainSuiteWrapperError(f"pytest plugin entry point drift: {name}")
        hook_trace: list[dict[str, object]] = []
        hook_callers = manager.get_hookcallers(plugin) or []
        for hook in sorted(
            hook_callers,
            key=lambda value: str(value.name).encode("utf-8"),
        ):
            for execution_index, implementation in enumerate(hook.get_hookimpls()):
                if implementation.plugin is not plugin:
                    continue
                function = implementation.function
                hook_trace.append(
                    {
                        "function_module": getattr(function, "__module__", None),
                        "function_qualname": getattr(function, "__qualname__", None),
                        "hook": hook.name,
                        "hookwrapper": implementation.hookwrapper,
                        "tryfirst": implementation.tryfirst,
                        "trylast": implementation.trylast,
                        "wrapper": implementation.wrapper,
                        "execution_index": execution_index,
                    }
                )
        hook_trace.sort(
            key=lambda row: (
                str(row["hook"]).encode("utf-8"),
                int(str(row["execution_index"])),
                str(row["function_qualname"]).encode("utf-8"),
            )
        )
        if policy.get("discovery_mode") is not True and hook_trace != expected.get("hook_trace"):
            raise MainSuiteWrapperError(f"pytest plugin hook trace drift: {name}")
        tree = _tree_inventory(expected.get("physical_tree_roots"))
        for key in (
            "byte_inventory_sha256",
            "directory_count",
            "entry_count",
            "file_count",
            "total_file_bytes",
            "tree_inventory_sha256",
        ):
            if tree.get(key) != expected.get("physical_tree", {}).get(key):
                raise MainSuiteWrapperError(f"pytest plugin tree drift: {name}:{key}")
        observed_rows.append(
            {
                "distribution": expected["distribution"],
                "entry_point_name": name,
                "hook_trace": hook_trace,
                "module": module_name,
                "module_file_binding": binding,
                "physical_tree": tree,
                "version": distribution.version,
            }
        )
    return observed_rows


def _support_tree_closure(policy: dict[str, object]) -> list[dict[str, object]]:
    expected_rows = policy.get("pytest_support_trees")
    if type(expected_rows) is not list:
        raise MainSuiteWrapperError("pytest support-tree policy is invalid")
    observed_rows: list[dict[str, object]] = []
    for index, expected in enumerate(expected_rows):
        if type(expected) is not dict or type(expected.get("name")) is not str:
            raise MainSuiteWrapperError(f"pytest support-tree row {index} is invalid")
        tree = _tree_inventory(expected.get("roots"))
        descriptor = expected.get("descriptor")
        if type(descriptor) is not dict:
            raise MainSuiteWrapperError("pytest support-tree descriptor is invalid")
        for key in (
            "byte_inventory_sha256",
            "directory_count",
            "entry_count",
            "file_count",
            "total_file_bytes",
            "tree_inventory_sha256",
        ):
            if tree.get(key) != descriptor.get(key):
                raise MainSuiteWrapperError(f"pytest support tree drift: {expected['name']}:{key}")
        observed_rows.append({"name": expected["name"], **tree})
    return observed_rows


def _loaded_conftest_binding(
    policy: dict[str, object],
    manager: Any,
) -> dict[str, object]:
    from pathlib import Path

    expected = policy.get("candidate_conftest")
    if type(expected) is not dict or type(expected.get("path")) is not str:
        raise MainSuiteWrapperError("candidate conftest policy is invalid")
    expected_path = Path(str(expected["path"])).resolve(strict=True)
    matches: list[object] = []
    for _name, plugin in manager.list_name_plugin():
        origin = getattr(plugin, "__file__", None)
        if not isinstance(origin, str):
            continue
        try:
            resolved = Path(origin).resolve(strict=True)
        except OSError:
            continue
        if resolved == expected_path:
            matches.append(plugin)
    if len(matches) != 1:
        raise MainSuiteWrapperError("candidate initial conftest was not loaded exactly once")
    binding = _stable_file_binding(str(expected_path))
    if type(binding) is not dict:  # pragma: no cover - typing guard
        raise MainSuiteWrapperError("candidate conftest binding is invalid")
    _assert_binding(binding, expected, label="candidate conftest")
    return binding


def _assert_phase_module_closure(
    policy: dict[str, object],
    *,
    phase: str,
    summary: object,
) -> None:
    if policy.get("discovery_mode") is True:
        return
    closures = policy.get("module_closures")
    if type(closures) is not dict or type(summary) is not dict:
        raise MainSuiteWrapperError("module closure policy is invalid")
    expected = closures.get(phase)
    if type(expected) is not dict or summary != expected:
        raise MainSuiteWrapperError(f"{phase} module closure drift")


def _module_path_class(
    path: str,
    *,
    candidate_root: object,
    authority_root: object,
    site_packages_root: object,
    runtime_roots: object,
) -> str:
    from pathlib import Path

    if (
        not all(
            isinstance(value, str) for value in (candidate_root, authority_root, site_packages_root)
        )
        or type(runtime_roots) is not list
    ):
        raise MainSuiteWrapperError("module path policy is invalid")
    candidate = Path(str(candidate_root))
    authority = Path(str(authority_root))
    site_packages = Path(str(site_packages_root))
    observed = Path(path)
    if _within(observed, candidate):
        return "candidate"
    if _within(observed, site_packages):
        return "site_packages"
    if _within(observed, authority):
        raise MainSuiteWrapperError(f"module escaped through authority source: {path}")
    for root in runtime_roots:
        if type(root) is not str:
            raise MainSuiteWrapperError("runtime module root is invalid")
        if path.startswith(f"{root}/") and root.endswith(".zip"):
            return "runtime_archive_member"
        if _within(observed, Path(root)):
            return "runtime"
    raise MainSuiteWrapperError(f"module origin is outside the frozen roots: {path}")


def _normalized_distribution_name(value: str) -> str:
    import re

    return re.sub(r"[-_.]+", "-", value).lower()


def _site_package_owner_index(
    policy: dict[str, object],
) -> tuple[dict[str, list[dict[str, object]]], dict[str, dict[str, object]]]:
    import importlib.metadata
    from pathlib import Path

    module_policy = policy.get("module_policy")
    if type(module_policy) is not dict:
        raise MainSuiteWrapperError("module policy is invalid")
    expected_rows = module_policy.get("distribution_ownership")
    if type(expected_rows) is not list:
        raise MainSuiteWrapperError("distribution ownership policy is invalid")
    expected_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for row in expected_rows:
        if (
            type(row) is not dict
            or type(row.get("name")) is not str
            or type(row.get("version")) is not str
        ):
            raise MainSuiteWrapperError("distribution ownership row is invalid")
        key = (str(row["name"]), str(row["version"]))
        if key in expected_by_key:
            raise MainSuiteWrapperError("duplicate distribution ownership row")
        expected_by_key[key] = row

    owners: dict[str, list[dict[str, object]]] = {}
    descriptors: dict[str, dict[str, object]] = {}
    for distribution in importlib.metadata.distributions():
        metadata: Any = distribution.metadata
        raw_name = metadata.get("Name")
        if not raw_name:
            continue
        name = _normalized_distribution_name(raw_name)
        version = distribution.version
        dist_path = getattr(distribution, "_path", None)
        if not isinstance(dist_path, Path):
            dist_path = Path(str(dist_path))
        metadata_path = dist_path / "METADATA"
        record_path = dist_path / "RECORD"
        if not metadata_path.is_file() or not record_path.is_file():
            continue
        metadata_binding = _stable_file_binding(str(metadata_path.resolve(strict=True)))
        record_binding = _stable_file_binding(str(record_path.resolve(strict=True)))
        if type(metadata_binding) is not dict or type(record_binding) is not dict:
            raise MainSuiteWrapperError("distribution metadata binding is invalid")
        descriptor: dict[str, object] = {
            "metadata_binding": metadata_binding,
            "name": name,
            "record_binding": record_binding,
            "version": version,
        }
        expected = expected_by_key.get((name, version))
        if policy.get("discovery_mode") is not True:
            if expected is None:
                raise MainSuiteWrapperError(
                    f"distribution ownership is not frozen: {name}=={version}"
                )
            _assert_binding(
                metadata_binding,
                expected.get("metadata_binding"),
                label=f"distribution METADATA {name}",
            )
            _assert_binding(
                record_binding,
                expected.get("record_binding"),
                label=f"distribution RECORD {name}",
            )
        descriptor_key = f"{name}=={version}"
        descriptors[descriptor_key] = descriptor
        files = distribution.files
        if files is None:
            continue
        for relative in files:
            try:
                located = Path(str(distribution.locate_file(relative))).resolve(strict=False)
            except (OSError, TypeError, ValueError):
                continue
            owners.setdefault(str(located), []).append(descriptor)
    if policy.get("discovery_mode") is not True:
        observed_keys = {(str(row["name"]), str(row["version"])) for row in descriptors.values()}
        if observed_keys != set(expected_by_key):
            raise MainSuiteWrapperError("distribution ownership descriptor set drift")
    return owners, descriptors


def _loaded_module_inventory(
    policy: dict[str, object],
    *,
    strict_all: bool,
) -> list[dict[str, object]]:
    from pathlib import Path

    module_policy = policy.get("module_policy")
    if type(module_policy) is not dict:
        raise MainSuiteWrapperError("module policy is invalid")
    rows: list[dict[str, object]] = []
    owners, _descriptors = _site_package_owner_index(policy)
    unowned_rows = module_policy.get("unowned_site_package_files")
    candidate_source_rows = module_policy.get("candidate_module_source_paths")
    if (
        type(unowned_rows) is not list
        or type(candidate_source_rows) is not list
        or module_policy.get("candidate_content_binding") != CANDIDATE_CONTENT_BINDING
    ):
        raise MainSuiteWrapperError("unowned site-package policy is invalid")
    candidate_source_paths = {str(value) for value in candidate_source_rows if type(value) is str}
    if len(candidate_source_paths) != len(candidate_source_rows):
        raise MainSuiteWrapperError("candidate module source policy is invalid")
    candidate_root = Path(str(module_policy.get("candidate_root"))).resolve(strict=True)
    unowned_by_path = {str(row.get("path")): row for row in unowned_rows if type(row) is dict}
    for name, module in sorted(sys.modules.items(), key=lambda item: item[0].encode("utf-8")):
        spec = getattr(module, "__spec__", None)
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str) and spec is not None:
            spec_origin = getattr(spec, "origin", None)
            if isinstance(spec_origin, str):
                origin = spec_origin
        locations = None if spec is None else getattr(spec, "submodule_search_locations", None)
        if origin in {"built-in", "frozen"}:
            rows.append({"classification": origin, "name": name})
            continue
        if not isinstance(origin, str):
            location_rows: list[dict[str, str]] = []
            if locations is not None:
                for location in locations:
                    location_text = str(Path(str(location)).resolve(strict=False))
                    classification = _module_path_class(
                        location_text,
                        candidate_root=module_policy.get("candidate_root"),
                        authority_root=module_policy.get("authority_root"),
                        site_packages_root=module_policy.get("site_packages_root"),
                        runtime_roots=module_policy.get("runtime_roots"),
                    )
                    location_rows.append({"classification": classification, "path": location_text})
            discovery_mode = policy.get("discovery_mode") is True
            if location_rows:
                allowed_namespaces = module_policy.get("allowed_namespace_modules")
                if (
                    strict_all
                    and not discovery_mode
                    and (type(allowed_namespaces) is not list or name not in allowed_namespaces)
                ):
                    raise MainSuiteWrapperError(f"namespace module is not frozen in policy: {name}")
            else:
                allowed_no_origin = module_policy.get("allowed_no_origin_modules")
                if (
                    strict_all
                    and not discovery_mode
                    and (type(allowed_no_origin) is not list or name not in allowed_no_origin)
                ):
                    raise MainSuiteWrapperError(f"module has no frozen classified origin: {name}")
            rows.append(
                {
                    "classification": "namespace" if location_rows else "no_origin",
                    "locations": location_rows,
                    "name": name,
                }
            )
            continue
        origin_path = str(Path(origin).resolve(strict=False))
        classification = _module_path_class(
            origin_path,
            candidate_root=module_policy.get("candidate_root"),
            authority_root=module_policy.get("authority_root"),
            site_packages_root=module_policy.get("site_packages_root"),
            runtime_roots=module_policy.get("runtime_roots"),
        )
        if classification == "runtime_archive_member":
            rows.append(
                {
                    "classification": classification,
                    "name": name,
                    "path": origin_path,
                }
            )
            continue
        binding = _stable_file_binding(str(Path(origin_path).resolve(strict=True)))
        if type(binding) is not dict:  # pragma: no cover - typing guard
            raise MainSuiteWrapperError("module binding is invalid")
        if (
            name == "quant_investor" or name.startswith("quant_investor.")
        ) and classification != "candidate":
            raise MainSuiteWrapperError(f"project module escaped candidate root: {name}")
        owner: dict[str, object] | None = None
        if classification == "site_packages":
            matching_owners = owners.get(str(binding["path"]), [])
            unique = {(str(row["name"]), str(row["version"])): row for row in matching_owners}
            if len(unique) == 1:
                descriptor = next(iter(unique.values()))
                metadata_binding = descriptor.get("metadata_binding")
                record_binding = descriptor.get("record_binding")
                if type(metadata_binding) is not dict or type(record_binding) is not dict:
                    raise MainSuiteWrapperError("distribution ownership binding is invalid")
                owner = {
                    "metadata_sha256": metadata_binding["sha256"],
                    "name": descriptor["name"],
                    "record_sha256": record_binding["sha256"],
                    "version": descriptor["version"],
                }
            elif len(unique) == 0:
                expected_unowned = unowned_by_path.get(str(binding["path"]))
                if expected_unowned is None and policy.get("discovery_mode") is not True:
                    raise MainSuiteWrapperError(
                        f"site-package module has no frozen RECORD owner: {name}"
                    )
                if expected_unowned is not None:
                    _assert_binding(
                        binding,
                        expected_unowned,
                        label=f"unowned site-package module {name}",
                    )
            else:
                raise MainSuiteWrapperError(
                    f"site-package module has multiple RECORD owners: {name}"
                )
        if classification == "candidate":
            if owner is not None:
                raise MainSuiteWrapperError(
                    f"candidate module unexpectedly has package owner: {name}"
                )
            try:
                candidate_relative = (
                    Path(str(binding["path"]))
                    .resolve(strict=True)
                    .relative_to(candidate_root)
                    .as_posix()
                )
            except ValueError as exc:
                raise MainSuiteWrapperError(
                    f"candidate module path escaped candidate root: {name}"
                ) from exc
            if (
                policy.get("discovery_mode") is not True
                and candidate_relative not in candidate_source_paths
            ):
                raise MainSuiteWrapperError(f"candidate module source is not frozen: {name}")
            rows.append(
                {
                    "classification": classification,
                    "content_binding": CANDIDATE_CONTENT_BINDING,
                    "name": name,
                    "owner": None,
                    "path": binding["path"],
                }
            )
        else:
            rows.append(
                {
                    "classification": classification,
                    "name": name,
                    "owner": owner,
                    "path": binding["path"],
                    "sha256": binding["sha256"],
                }
            )
    return rows


def _project_modules(candidate_root: str) -> list[dict[str, str]]:
    from pathlib import Path

    candidate = Path(candidate_root)
    rows: list[dict[str, str]] = []
    for name, module in sorted(sys.modules.items(), key=lambda item: item[0].encode("utf-8")):
        if not (name == "quant_investor" or name.startswith("quant_investor.")):
            continue
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str) or not _within(Path(origin), candidate):
            raise MainSuiteWrapperError(f"project module escaped candidate root: {name}")
        binding = _stable_file_binding(str(Path(origin).resolve(strict=True)))
        if type(binding) is not dict:  # pragma: no cover - typing guard
            raise MainSuiteWrapperError("project module binding is invalid")
        rows.append(
            {
                "name": name,
                "path": str(binding["path"]),
                "sha256": str(binding["sha256"]),
            }
        )
    return rows


def _runtime_snapshot(
    policy: dict[str, object],
    policy_raw: bytes,
    routing: dict[str, object],
) -> dict[str, object]:
    from pathlib import Path

    main_runtime = policy.get("main_runtime")
    if type(main_runtime) is not dict:
        raise MainSuiteWrapperError("policy main_runtime is invalid")
    lexical_python = main_runtime.get("lexical_python")
    expected_inventory = main_runtime.get("valid_inventory")
    if not isinstance(lexical_python, str) or type(expected_inventory) is not dict:
        raise MainSuiteWrapperError("policy main runtime bindings are incomplete")
    interpreter = _stable_file_binding(str(Path(lexical_python).resolve(strict=True)))
    if type(interpreter) is not dict:  # pragma: no cover - typing guard
        raise MainSuiteWrapperError("interpreter binding is invalid")
    inventory = _distribution_inventory()
    for key in (
        "count",
        "physical_dist_info_count",
        "physical_dist_info_names_sha256",
        "rows_sha256",
    ):
        if inventory.get(key) != expected_inventory.get(key):
            raise MainSuiteWrapperError(f"main distribution inventory drift: {key}")
    invalid_dist_info = _invalid_dist_info_inventory(policy)
    factor_rows = policy.get("factor_authority_sources")
    if type(factor_rows) is not list:
        raise MainSuiteWrapperError("policy factor source inventory is invalid")
    module_policy = policy.get("module_policy")
    if type(module_policy) is not dict:
        raise MainSuiteWrapperError("policy module root inventory is invalid")
    factor_bindings: list[dict[str, object]] = []
    for row in factor_rows:
        if type(row) is not dict:
            raise MainSuiteWrapperError("policy factor source row is invalid")
        relative = row.get("relative_path")
        sha = row.get("sha256")
        size = row.get("size_bytes")
        if not isinstance(relative, str) or not isinstance(sha, str) or type(size) is not int:
            raise MainSuiteWrapperError("policy factor source row is incomplete")
        from pathlib import PurePosixPath

        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or not pure.parts
            or pure.as_posix() != relative
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise MainSuiteWrapperError("factor source path is unsafe")
        for root_key in ("candidate_root", "authority_root"):
            root = (
                policy.get("candidate_root")
                if root_key == "candidate_root"
                else module_policy.get("authority_root")
            )
            if not isinstance(root, str):
                raise MainSuiteWrapperError("policy repository root is invalid")
            root_path = Path(root).resolve(strict=True)
            target = (root_path / relative).resolve(strict=True)
            try:
                target.relative_to(root_path)
            except ValueError as exc:
                raise MainSuiteWrapperError("factor source escaped repository root") from exc
            binding = _stable_file_binding(str(target))
            if type(binding) is not dict:  # pragma: no cover - typing guard
                raise MainSuiteWrapperError("factor source binding is invalid")
            if binding["sha256"] != sha or binding["size_bytes"] != size:
                raise MainSuiteWrapperError(f"factor authority source drift: {relative}")
        factor_bindings.append({"relative_path": relative, "sha256": sha, "size_bytes": size})
    module_rows = _loaded_module_inventory(policy, strict_all=True)
    classification_counts: dict[str, int] = {}
    for row in module_rows:
        classification = str(row["classification"])
        classification_counts[classification] = classification_counts.get(classification, 0) + 1
    result: dict[str, object] = {
        "bytecode_policy": _bytecode_policy(),
        "factor_authority_sha256": _sha256(_canonical_bytes(factor_bindings)),
        "interpreter": interpreter,
        "invalid_dist_info_sha256": _sha256(_canonical_bytes(invalid_dist_info)),
        "inventory": inventory,
        "loaded_modules": {
            "classification_counts": dict(sorted(classification_counts.items())),
            "count": len(module_rows),
            "rows_sha256": _sha256(_canonical_bytes(module_rows)),
        },
        "policy_sha256": _sha256(policy_raw),
        "project_modules": _project_modules(str(policy["candidate_root"])),
        "routing": routing,
    }
    if policy.get("discovery_mode") is True:
        result["loaded_module_discovery_rows"] = module_rows
    return result


def _emit_frame(
    descriptor: int,
    *,
    phase: int,
    nonce: bytes,
    payload: dict[str, object],
) -> dict[str, object]:
    raw = _canonical_bytes(payload)
    frame_cap = MAX_TERMINAL_FRAME_BYTES if phase == 3 else MAX_FRAME_BYTES
    if not raw or len(raw) > frame_cap:
        raise MainSuiteWrapperError("attestation frame exceeds bounds")
    digest = bytes.fromhex(_sha256(raw))
    header = ATTEST_HEADER.pack(
        ATTEST_MAGIC,
        PROTOCOL_VERSION,
        phase,
        0,
        len(raw),
        nonce,
        digest,
    )
    _write_all(descriptor, header + raw)
    return {"payload_sha256": digest.hex(), "payload_size_bytes": len(raw)}


class _CandidateImportGuard:
    def __init__(self, policy: dict[str, object]) -> None:
        self.policy = policy
        self._myquant_phase0_candidate_guard = True
        self.owners, _descriptors = _site_package_owner_index(policy)
        module_policy = policy.get("module_policy")
        if type(module_policy) is not dict:
            raise MainSuiteWrapperError("module policy is invalid")
        rows = module_policy.get("unowned_site_package_files")
        if type(rows) is not list:
            raise MainSuiteWrapperError("unowned site-package policy is invalid")
        self.unowned = {str(row.get("path")): row for row in rows if type(row) is dict}

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> Any:
        import importlib.machinery
        from pathlib import Path

        is_project = fullname == "quant_investor" or fullname.startswith("quant_investor.")
        if is_project:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        else:
            spec = None
            for finder in tuple(sys.meta_path):
                if finder is self:
                    continue
                method = getattr(finder, "find_spec", None)
                if method is None:
                    continue
                spec = method(fullname, path, target)
                if spec is not None:
                    break
        if spec is None:
            return None
        module_policy = self.policy.get("module_policy")
        if type(module_policy) is not dict:
            raise ImportError("module policy is invalid")
        origin = getattr(spec, "origin", None)
        if is_project:
            loader = getattr(spec, "loader", None)
            if type(loader) is not importlib.machinery.SourceFileLoader:
                raise ImportError(f"project import loader rejected: {fullname}")
            if not isinstance(origin, str):
                raise ImportError(f"project import origin rejected: {fullname}")
            loader_name = getattr(loader, "name", None)
            loader_path = getattr(loader, "path", None)
            if (
                loader_name != fullname
                or not isinstance(loader_path, str)
                or Path(loader_path).resolve(strict=False) != Path(origin).resolve(strict=False)
            ):
                raise ImportError(f"project import loader drift: {fullname}")
        if origin not in {"built-in", "frozen", None}:
            origin_path = str(Path(str(origin)).resolve(strict=False))
            try:
                classification = _module_path_class(
                    origin_path,
                    candidate_root=module_policy.get("candidate_root"),
                    authority_root=module_policy.get("authority_root"),
                    site_packages_root=module_policy.get("site_packages_root"),
                    runtime_roots=module_policy.get("runtime_roots"),
                )
            except MainSuiteWrapperError as exc:
                raise ImportError(f"import origin rejected: {fullname}") from exc
            if is_project and classification != "candidate":
                raise ImportError(f"project import escaped candidate root: {fullname}")
            if classification == "site_packages":
                matching = {
                    (str(row["name"]), str(row["version"]))
                    for row in self.owners.get(origin_path, [])
                }
                if len(matching) > 1:
                    raise ImportError(f"site-package import has multiple RECORD owners: {fullname}")
                if len(matching) == 0:
                    expected = self.unowned.get(origin_path)
                    if expected is None and self.policy.get("discovery_mode") is not True:
                        raise ImportError(f"site-package import has no RECORD owner: {fullname}")
                    if expected is not None:
                        binding = _stable_file_binding(origin_path)
                        if type(binding) is not dict:
                            raise ImportError(f"unowned site-package binding failed: {fullname}")
                        try:
                            _assert_binding(
                                binding,
                                expected,
                                label=f"unowned site-package import {fullname}",
                            )
                        except MainSuiteWrapperError as exc:
                            raise ImportError(
                                f"unowned site-package import drift: {fullname}"
                            ) from exc
        locations = getattr(spec, "submodule_search_locations", None)
        if locations is not None:
            for location in locations:
                try:
                    _module_path_class(
                        str(Path(str(location)).resolve(strict=False)),
                        candidate_root=module_policy.get("candidate_root"),
                        authority_root=module_policy.get("authority_root"),
                        site_packages_root=module_policy.get("site_packages_root"),
                        runtime_roots=module_policy.get("runtime_roots"),
                    )
                except MainSuiteWrapperError as exc:
                    raise ImportError(f"namespace import rejected: {fullname}") from exc
            if origin is None:
                allowed = module_policy.get("allowed_namespace_modules")
                if (
                    self.policy.get("discovery_mode") is not True
                    and type(allowed) is list
                    and fullname not in allowed
                ):
                    raise ImportError(f"namespace import is not frozen: {fullname}")
        elif origin is None:
            allowed = module_policy.get("allowed_no_origin_modules")
            if (
                self.policy.get("discovery_mode") is not True
                and type(allowed) is list
                and fullname not in allowed
            ):
                raise ImportError(f"originless import is not frozen: {fullname}")
        return spec


def _assert_guard_head(guard: _CandidateImportGuard) -> None:
    if not sys.meta_path or sys.meta_path[0] is not guard:
        prefix = [
            f"{type(finder).__module__}.{type(finder).__name__}" for finder in sys.meta_path[:4]
        ]
        raise MainSuiteWrapperError(
            f"candidate import guard is not first on sys.meta_path: {prefix}"
        )


def _restore_guard_head_after_pytest(
    guard: _CandidateImportGuard,
) -> None:
    positions = [index for index, finder in enumerate(sys.meta_path) if finder is guard]
    if len(positions) != 1:
        raise MainSuiteWrapperError("candidate import guard multiplicity changed")
    index = positions[0]
    displaced_by = sys.meta_path[:index]
    if any(
        type(finder).__module__ != "_pytest.assertion.rewrite"
        or type(finder).__name__ != "AssertionRewritingHook"
        for finder in displaced_by
    ):
        raise MainSuiteWrapperError("candidate import guard was displaced by an unknown finder")
    if index:
        sys.meta_path.pop(index)
        sys.meta_path.insert(0, guard)
    _assert_guard_head(guard)


def _pytest_main_with_guard(
    pytest_module: Any,
    pytest_args: list[str],
    plugin: object,
    guard: _CandidateImportGuard,
) -> int:
    import importlib

    assertion_module = importlib.import_module("_pytest.assertion")
    original = getattr(assertion_module, "install_importhook", None)
    if not callable(original):
        raise MainSuiteWrapperError("pytest assertion import-hook installer is unavailable")

    def install_importhook(config: object) -> object:
        hook = original(config)
        _restore_guard_head_after_pytest(guard)
        return hook

    setattr(assertion_module, "install_importhook", install_importhook)
    try:
        return int(pytest_module.main(pytest_args, plugins=[plugin]))
    finally:
        setattr(assertion_module, "install_importhook", original)


class _AttestationPlugin:
    def __init__(
        self,
        *,
        descriptor: int,
        nonce: bytes,
        challenge_sha: bytes,
        guard: _CandidateImportGuard,
        policy: dict[str, object],
        policy_raw: bytes,
        routing: dict[str, object],
    ) -> None:
        self.descriptor = descriptor
        self.nonce = nonce
        self.challenge_sha = challenge_sha
        self.guard = guard
        self.policy = policy
        self.policy_raw = policy_raw
        self.routing = routing
        self.emitted = False
        self.final_audit_error: str | None = None
        self.sessionfinish_audit_completed = False

    def pytest_load_initial_conftests(
        self,
        early_config: object,
        parser: object,
        args: object,
    ) -> None:
        del early_config, parser, args
        _restore_guard_head_after_pytest(self.guard)

    def pytest_sessionstart(self, session: Any) -> None:
        import pytest

        _assert_guard_head(self.guard)
        manager = session.config.pluginmanager
        plugins = _plugin_closure(self.policy, manager)
        support_trees = _support_tree_closure(self.policy)
        conftest = _loaded_conftest_binding(self.policy, manager)
        snapshot = _runtime_snapshot(self.policy, self.policy_raw, self.routing)
        _assert_phase_module_closure(
            self.policy,
            phase="pre_collection",
            summary=snapshot.get("loaded_modules"),
        )
        payload = {
            "candidate_conftest": conftest,
            "frame": "pre_collection",
            "initial_conftest_loaded": True,
            "pid": os.getpid(),
            "plugins": plugins,
            "ppid": os.getppid(),
            "project_modules": _project_modules(str(self.policy["candidate_root"])),
            "pytest_version": pytest.__version__,
            "runtime": snapshot,
            "challenge_binding_sha256": self.challenge_sha.hex(),
            "support_trees": support_trees,
        }
        _emit_frame(self.descriptor, phase=2, nonce=self.nonce, payload=payload)
        self.emitted = True

    def pytest_sessionfinish(self, session: object, exitstatus: object) -> None:
        try:
            _assert_guard_head(self.guard)
            _loaded_module_inventory(self.policy, strict_all=True)
            _project_modules(str(self.policy["candidate_root"]))
            snapshot = _runtime_snapshot(self.policy, self.policy_raw, self.routing)
            if snapshot.get("loaded_modules") is None:
                raise MainSuiteWrapperError("sessionfinish module audit is empty")
            self.sessionfinish_audit_completed = True
        except MainSuiteWrapperError as exc:
            self.final_audit_error = str(exc)
            raise


def _parse_args(argv: list[str]) -> tuple[str, str, str, list[str]]:
    if len(argv) < 5 or argv[3] != "--":
        raise MainSuiteWrapperError("usage: wrapper POLICY EXPECTED_POLICY_SHA -- PYTEST_ARGS...")
    policy_path, policy_sha = argv[1], argv[2]
    pytest_args = argv[4:]
    if not pytest_args:
        raise MainSuiteWrapperError("pytest argument plan is empty")
    return policy_path, policy_sha, argv[3], pytest_args


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv if argv is None else argv)
    _bytecode_policy()
    challenge_fd = _fd_from_environment(CHALLENGE_ENV)
    attest_fd = _fd_from_environment(ATTEST_ENV)
    if challenge_fd == attest_fd:
        raise MainSuiteWrapperError("challenge and attestation descriptors must differ")
    try:
        nonce, challenge_sha = _read_challenge(challenge_fd)
    finally:
        os.close(challenge_fd)
    policy_path, policy_sha, _separator, pytest_args = _parse_args(values)
    policy, policy_raw = _strict_policy(policy_path, policy_sha)
    environment = _validate_pytest_environment(policy)
    pycache_prefix = environment.get("PYTHONPYCACHEPREFIX")
    if not isinstance(pycache_prefix, str):
        raise MainSuiteWrapperError("PYTHONPYCACHEPREFIX is missing")
    _bytecode_policy()
    routing = _restore_site(policy)
    guard = _CandidateImportGuard(policy)
    sys.meta_path.insert(0, guard)
    _assert_guard_head(guard)
    phase1_runtime = _runtime_snapshot(policy, policy_raw, routing)
    _assert_phase_module_closure(
        policy,
        phase="pre_import",
        summary=phase1_runtime.get("loaded_modules"),
    )
    phase1 = {
        "environment": environment,
        "frame": "pre_import",
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "runtime": phase1_runtime,
        "challenge_binding_sha256": challenge_sha.hex(),
    }
    _emit_frame(attest_fd, phase=1, nonce=nonce, payload=phase1)
    import pytest

    _restore_guard_head_after_pytest(guard)
    if pytest_args != policy.get("pytest_args"):
        raise MainSuiteWrapperError("pytest argv differs from policy")
    plugin = _AttestationPlugin(
        descriptor=attest_fd,
        nonce=nonce,
        challenge_sha=challenge_sha,
        guard=guard,
        policy=policy,
        policy_raw=policy_raw,
        routing=routing,
    )
    code = _pytest_main_with_guard(pytest, pytest_args, plugin, guard)
    _assert_guard_head(guard)
    if not plugin.emitted:
        raise MainSuiteWrapperError("pre-collection attestation was not emitted")
    if plugin.final_audit_error is not None:
        raise MainSuiteWrapperError(f"final loaded-module audit failed: {plugin.final_audit_error}")
    if not plugin.sessionfinish_audit_completed:
        raise MainSuiteWrapperError("pytest sessionfinish audit was not completed")
    final_snapshot = _runtime_snapshot(policy, policy_raw, routing)
    final_summary = final_snapshot.get("loaded_modules")
    _assert_phase_module_closure(
        policy,
        phase="final",
        summary=final_summary,
    )
    if policy.get("discovery_mode") is True:
        print(
            DISCOVERY_FINAL_PREFIX
            + _canonical_bytes(
                {
                    "rows": final_snapshot.get("loaded_module_discovery_rows"),
                    "summary": final_summary,
                }
            ).decode("utf-8", errors="strict"),
            file=sys.stderr,
        )
    terminal_payload = {
        "challenge_binding_sha256": challenge_sha.hex(),
        "final_loaded_modules": final_summary,
        "frame": "terminal_complete",
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "pytest_exit_code": code,
    }
    _emit_frame(
        plugin.descriptor,
        phase=3,
        nonce=nonce,
        payload=terminal_payload,
    )
    os.close(plugin.descriptor)
    plugin.descriptor = -1
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MainSuiteWrapperError as exc:
        print(f"v17 main-suite wrapper failed: {exc}", file=sys.stderr)
        raise SystemExit(WRAPPER_FAILURE_EXIT) from exc
    except Exception as exc:
        detail = str(exc).replace("\r", " ").replace("\n", " ")[:512]
        print(
            f"v17 main-suite wrapper unexpected failure: {type(exc).__name__}: {detail}",
            file=sys.stderr,
        )
        raise SystemExit(WRAPPER_FAILURE_EXIT) from exc
