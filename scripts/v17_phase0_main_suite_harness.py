#!/usr/bin/env python3
"""Parent-side fail-closed harness for the Phase 0 main-runtime pytest suite."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import selectors
import signal
import stat
import struct
import subprocess
import time
from typing import Any, Mapping, Sequence
from pathlib import PurePosixPath

POLICY_VERSION = "myquant.v17.v2.phase0-main-suite-runtime-policy.v1"
POLICY_SCHEMA_ID = "myquant.v17.v2.phase0-main-suite-runtime-policy.schema.v1"
RECEIPT_VERSION = "myquant.v17.v2.phase0-main-suite-receipt.v1"
RECEIPT_SCHEMA_ID = "myquant.v17.v2.phase0-main-suite-receipt.schema.v1"
RECEIPT_PREFIX = b"MYQUANT_PHASE0_MAIN_SUITE_RECEIPT="
CHALLENGE_ENV = "MYQUANT_PHASE0_CHALLENGE_FD"
ATTEST_ENV = "MYQUANT_PHASE0_ATTEST_FD"
CHALLENGE_MAGIC = b"MQP0CH01"
ATTEST_MAGIC = b"MQP0AT01"
PROTOCOL_VERSION = 1
CHALLENGE_STRUCT = struct.Struct(">8sB3s32s32s")
ATTEST_HEADER = struct.Struct(">8sBBHI32s32s")
MAX_POLICY_BYTES = 4 * 1024 * 1024
MAX_FRAME_BYTES = 1024 * 1024
MAX_TERMINAL_FRAME_BYTES = 16 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
LIMITATIONS = [
    "PRECOLLECTION_ALLOWED_TREES_ATTESTED_NOT_COMPLETE_MAIN_RUNTIME_FILESET",
    "NO_DIRECT_LAUNCH_REFERENCE_ONLY_NOT_PROCESS_IO_ATTESTED",
    "OFFLINE_FLAGS_ONLY_NOT_OS_EGRESS_ATTESTED",
    "OWNER_DECLARED_PATHS_ONLY_NOT_CONTENT_PROVENANCE",
    "PIP_25_2_PACKAGE_INSTALL_ENV_ONLY_NATIVE_AND_BUILD_ENV_PIP_ABSENT",
]
PYTEST_ARGS = [
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
]
TIMEOUTS = {
    "cleanup_seconds": 5,
    "max_attestation_bytes": 2_113_776,
    "max_stream_bytes": 128 * 1024 * 1024,
    "phase1_seconds": 60,
    "phase2_seconds": 60,
    "suite_seconds": 7_200,
}
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

POLICY_KEYS = {
    "authority",
    "candidate_conftest",
    "candidate_root",
    "claims",
    "discovery_mode",
    "factor_authority_sources",
    "harness_binding",
    "limitations",
    "main_runtime",
    "module_closures",
    "module_policy",
    "protected_roots",
    "protocol_version",
    "pytest_args",
    "pytest_environment",
    "pytest_plugins",
    "pytest_support_trees",
    "routing",
    "schema_id",
    "semantic_sha256",
    "status",
    "timeouts",
    "version",
    "wrapper_binding",
}


class MainSuiteHarnessError(RuntimeError):
    """Raised when the parent cannot safely launch or attest the main suite."""


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise MainSuiteHarnessError("JSON contains duplicate keys")
        value[key] = item
    return value


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise MainSuiteHarnessError("value is not canonical JSON") from exc


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _semantic_sha256(value: Mapping[str, object]) -> str:
    body = dict(value)
    body.pop("semantic_sha256", None)
    return _sha256(_canonical_bytes(body))


def _seal(value: Mapping[str, object]) -> dict[str, object]:
    if "semantic_sha256" in value:
        raise MainSuiteHarnessError("semantic_sha256 must not be supplied")
    sealed = dict(value)
    sealed["semantic_sha256"] = _semantic_sha256(sealed)
    return sealed


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


def _stable_file(path: Path, *, max_bytes: int) -> tuple[dict[str, object], bytes]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise MainSuiteHarnessError(f"cannot open bound file: {path}") from exc
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise MainSuiteHarnessError(f"unsafe bound file: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise MainSuiteHarnessError(f"bound file exceeds cap: {path}")
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise MainSuiteHarnessError(f"bound file path disappeared: {path}") from exc
    if (
        _stat_signature(before) != _stat_signature(after)
        or _stat_signature(after) != _stat_signature(path_after)
        or size != before.st_size
    ):
        raise MainSuiteHarnessError(f"bound file drifted: {path}")
    return (
        {
            "gid": before.st_gid,
            "mode": f"{stat.S_IMODE(before.st_mode):04o}",
            "path": str(path),
            "sha256": digest.hexdigest(),
            "size_bytes": size,
            "st_dev": before.st_dev,
            "st_ino": before.st_ino,
            "st_nlink": before.st_nlink,
            "uid": before.st_uid,
        },
        b"".join(chunks),
    )


def _stable_symlink(path: Path) -> dict[str, object]:
    try:
        before = path.lstat()
        link_text = os.readlink(path)
        after = path.lstat()
    except OSError as exc:
        raise MainSuiteHarnessError(f"cannot bind symlink: {path}") from exc
    if (
        not stat.S_ISLNK(before.st_mode)
        or _stat_signature(before) != _stat_signature(after)
        or before.st_uid != os.getuid()
    ):
        raise MainSuiteHarnessError(f"symlink binding drifted: {path}")
    return {
        "gid": before.st_gid,
        "link_text": link_text,
        "mode": f"{stat.S_IMODE(before.st_mode):04o}",
        "path": str(path),
        "size_bytes": before.st_size,
        "st_dev": before.st_dev,
        "st_ino": before.st_ino,
        "st_nlink": before.st_nlink,
        "uid": before.st_uid,
    }


def _assert_binding(
    observed: Mapping[str, object],
    expected: object,
    *,
    label: str,
) -> None:
    if type(expected) is not dict:
        raise MainSuiteHarnessError(f"{label} policy binding is invalid")
    for key, value in expected.items():
        if key == "present":
            continue
        if observed.get(key) != value:
            raise MainSuiteHarnessError(f"{label} binding mismatch: {key}")


def _parse_policy_bytes(raw: bytes) -> dict[str, object]:
    if type(raw) is not bytes or not raw or len(raw) > MAX_POLICY_BYTES:
        raise MainSuiteHarnessError("runtime policy bytes are invalid")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MainSuiteHarnessError("runtime policy is invalid JSON") from exc
    if type(value) is not dict:
        raise MainSuiteHarnessError("runtime policy root is not an object")
    if raw != _canonical_bytes(value) + b"\n":
        raise MainSuiteHarnessError("runtime policy is not canonical JSON")
    return value


def validate_policy_bytes(raw: bytes) -> dict[str, object]:
    """Validate and return one canonical policy resource."""

    value = _parse_policy_bytes(raw)
    if set(value) != POLICY_KEYS:
        raise MainSuiteHarnessError("runtime policy has unexpected keys")
    if (
        value.get("version") != POLICY_VERSION
        or value.get("protocol_version") != "myquant.v17.v2"
        or value.get("schema_id") != POLICY_SCHEMA_ID
        or value.get("status") != "FROZEN"
        or value.get("authority") is not False
        or type(value.get("discovery_mode")) is not bool
        or value.get("semantic_sha256") != _semantic_sha256(value)
    ):
        raise MainSuiteHarnessError("runtime policy identity is invalid")
    if value.get("discovery_mode") is True:
        raise MainSuiteHarnessError("packaged runtime policy cannot enable discovery mode")
    for key in ("candidate_root",):
        path = value.get(key)
        if type(path) is not str or not Path(path).is_absolute():
            raise MainSuiteHarnessError(f"runtime policy {key} is invalid")
    candidate = Path(str(value["candidate_root"])).resolve(strict=False)
    expected_paths = {
        "candidate_conftest": candidate / "tests/conftest.py",
        "harness_binding": candidate / "scripts/v17_phase0_main_suite_harness.py",
        "wrapper_binding": candidate / "scripts/v17_phase0_main_suite_wrapper.py",
    }
    for label, expected_path in expected_paths.items():
        binding = value.get(label)
        if (
            type(binding) is not dict
            or type(binding.get("path")) is not str
            or Path(str(binding["path"])).resolve(strict=False) != expected_path
        ):
            raise MainSuiteHarnessError(f"runtime policy {label} path is invalid")
    if type(value.get("pytest_args")) is not list or not value["pytest_args"]:
        raise MainSuiteHarnessError("runtime policy pytest_args are invalid")
    if value["pytest_args"] != PYTEST_ARGS:
        raise MainSuiteHarnessError("runtime policy pytest_args are not frozen")
    if value.get("claims") != {
        "kernel_egress_attested": False,
        "network_unreachability_proven": False,
        "offline_policy_enforced": True,
    }:
        raise MainSuiteHarnessError("runtime policy claims are invalid")
    if value.get("limitations") != LIMITATIONS:
        raise MainSuiteHarnessError("runtime policy limitations are invalid")
    if value.get("timeouts") != TIMEOUTS:
        raise MainSuiteHarnessError("runtime policy timeouts are invalid")
    main_runtime = value.get("main_runtime")
    if type(main_runtime) is not dict or set(main_runtime) != {
        "interpreter_flags",
        "invalid_dist_info",
        "lexical_python",
        "lexical_python_binding",
        "post_site_state",
        "pre_site_state",
        "resolved_python_binding",
        "site_packages_root",
        "startup_files",
        "startup_modules",
        "valid_inventory",
    }:
        raise MainSuiteHarnessError("runtime policy main_runtime shape is invalid")
    if (
        type(main_runtime.get("lexical_python")) is not str
        or type(main_runtime.get("startup_files")) is not list
        or type(main_runtime.get("startup_modules")) is not list
        or type(main_runtime.get("invalid_dist_info")) is not list
    ):
        raise MainSuiteHarnessError("runtime policy main_runtime values are invalid")
    if main_runtime.get("interpreter_flags") != {
        "dont_write_bytecode": 1,
        "isolated": 1,
        "no_site": 1,
    }:
        raise MainSuiteHarnessError("runtime policy interpreter flags are invalid")
    module_policy = value.get("module_policy")
    if type(module_policy) is not dict or set(module_policy) != {
        "allowed_namespace_modules",
        "allowed_no_origin_modules",
        "authority_root",
        "candidate_root",
        "candidate_content_binding",
        "candidate_module_source_paths",
        "distribution_ownership",
        "runtime_roots",
        "site_packages_root",
        "unowned_site_package_files",
    }:
        raise MainSuiteHarnessError("runtime policy module_policy shape is invalid")
    for key in (
        "allowed_namespace_modules",
        "allowed_no_origin_modules",
        "candidate_module_source_paths",
        "distribution_ownership",
        "runtime_roots",
        "unowned_site_package_files",
    ):
        if type(module_policy.get(key)) is not list:
            raise MainSuiteHarnessError(f"runtime policy module_policy.{key} is invalid")
    if module_policy.get("candidate_content_binding") != CANDIDATE_CONTENT_BINDING:
        raise MainSuiteHarnessError("runtime policy candidate content binding is invalid")
    candidate_module_paths = module_policy.get("candidate_module_source_paths")
    if type(candidate_module_paths) is not list or not candidate_module_paths:
        raise MainSuiteHarnessError("runtime policy candidate module source paths are invalid")
    normalized_candidate_paths: list[str] = []
    casefolded_candidate_paths: set[str] = set()
    for relative in candidate_module_paths:
        if type(relative) is not str:
            raise MainSuiteHarnessError("runtime policy candidate module path is invalid")
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or not pure.parts
            or pure.suffix != ".py"
            or any(part in {"", ".", ".."} for part in pure.parts)
            or pure.as_posix() != relative
            or relative.casefold() in casefolded_candidate_paths
        ):
            raise MainSuiteHarnessError(
                "runtime policy candidate module path semantics are invalid"
            )
        normalized_candidate_paths.append(relative)
        casefolded_candidate_paths.add(relative.casefold())
    if normalized_candidate_paths != sorted(
        normalized_candidate_paths,
        key=lambda item: item.encode("utf-8"),
    ):
        raise MainSuiteHarnessError("runtime policy candidate module paths are not canonical")
    environment = value.get("pytest_environment")
    if type(environment) is not dict or set(environment) != {
        "allowed_keys",
        "dynamic_path_keys",
        "forbidden",
        "path_topology",
        "required",
    }:
        raise MainSuiteHarnessError("runtime policy pytest_environment shape is invalid")
    if (
        type(environment.get("allowed_keys")) is not list
        or type(environment.get("dynamic_path_keys")) is not list
        or type(environment.get("forbidden")) is not list
        or environment.get("path_topology") != PATH_TOPOLOGY
        or type(environment.get("required")) is not dict
    ):
        raise MainSuiteHarnessError("runtime policy pytest_environment values are invalid")
    module_closures = value.get("module_closures")
    factor_sources = value.get("factor_authority_sources")
    pytest_plugins = value.get("pytest_plugins")
    support_trees = value.get("pytest_support_trees")
    protected_roots = value.get("protected_roots")
    if type(module_closures) is not dict or set(module_closures) != {
        "final",
        "pre_collection",
        "pre_import",
    }:
        raise MainSuiteHarnessError("runtime policy module closures are invalid")
    if (
        type(factor_sources) is not list
        or len(factor_sources) != 19
        or type(pytest_plugins) is not list
        or len(pytest_plugins) != 3
        or type(support_trees) is not list
        or not support_trees
        or type(protected_roots) is not list
        or len(protected_roots) != 4
    ):
        raise MainSuiteHarnessError("runtime policy fixed inventories are invalid")
    if (
        module_policy.get("candidate_root") != value["candidate_root"]
        or module_policy.get("site_packages_root") != main_runtime.get("site_packages_root")
        or type(module_policy.get("authority_root")) is not str
        or module_policy.get("authority_root") == value["candidate_root"]
        or not Path(str(module_policy.get("authority_root"))).is_absolute()
    ):
        raise MainSuiteHarnessError("runtime policy root semantics are invalid")
    protected_classification_roots = [
        Path(str(value["candidate_root"])).resolve(strict=False),
        Path(str(module_policy["authority_root"])).resolve(strict=False),
        Path(str(module_policy["site_packages_root"])).resolve(strict=False),
    ]
    raw_runtime_roots = module_policy["runtime_roots"]
    if any(
        type(item) is not str or not Path(item).is_absolute() or os.path.normpath(item) != item
        for item in raw_runtime_roots
    ):
        raise MainSuiteHarnessError("runtime module roots are invalid")
    runtime_roots = [Path(item).resolve(strict=False) for item in raw_runtime_roots]
    if len(runtime_roots) != len(set(runtime_roots)):
        raise MainSuiteHarnessError("runtime module roots are invalid")
    for runtime_root in runtime_roots:
        if not runtime_root.is_absolute():
            raise MainSuiteHarnessError("runtime module root is not absolute")
        for protected_root in protected_classification_roots:
            try:
                runtime_root.relative_to(protected_root)
            except ValueError:
                pass
            else:
                raise MainSuiteHarnessError("runtime root overlaps protected root")
            try:
                protected_root.relative_to(runtime_root)
            except ValueError:
                pass
            else:
                raise MainSuiteHarnessError("runtime root contains protected root")
    seen_factor_paths: set[str] = set()
    seen_factor_paths_casefolded: set[str] = set()
    for row in factor_sources:
        if type(row) is not dict or set(row) != {
            "relative_path",
            "sha256",
            "size_bytes",
        }:
            raise MainSuiteHarnessError("factor authority source row is invalid")
        relative = row.get("relative_path")
        if type(relative) is not str:
            raise MainSuiteHarnessError("factor authority relative path is invalid")
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or not pure.parts
            or any(part in {"", ".", ".."} for part in pure.parts)
            or pure.as_posix() != relative
            or relative in seen_factor_paths
            or relative.casefold() in seen_factor_paths_casefolded
            or SHA256_RE.fullmatch(str(row.get("sha256"))) is None
            or type(row.get("size_bytes")) is not int
            or row["size_bytes"] < 0
        ):
            raise MainSuiteHarnessError("factor authority source semantics are invalid")
        seen_factor_paths.add(relative)
        seen_factor_paths_casefolded.add(relative.casefold())
        for root in (
            Path(str(value["candidate_root"])),
            Path(str(module_policy["authority_root"])),
        ):
            resolved = (root / relative).resolve(strict=False)
            try:
                resolved.relative_to(root.resolve(strict=False))
            except ValueError as exc:
                raise MainSuiteHarnessError("factor authority source escapes root") from exc
    routing = value.get("routing")
    if (
        type(routing) is not dict
        or set(routing)
        != {
            "quant_investor_origin",
            "removed_authority_entries",
            "sanitized_sys_path",
        }
        or routing.get("quant_investor_origin") != str(candidate / "quant_investor/__init__.py")
        or routing.get("removed_authority_entries") != [module_policy["authority_root"]]
        or type(routing.get("sanitized_sys_path")) is not list
        or not routing["sanitized_sys_path"]
        or routing["sanitized_sys_path"][0] != value["candidate_root"]
    ):
        raise MainSuiteHarnessError("runtime policy routing semantics are invalid")
    expected_protected = [
        (
            "authority_results_v16",
            str(Path(str(module_policy["authority_root"])) / "results/v16"),
            "PRESENT_DIRECTORY",
        ),
        (
            "authority_results_v16_operator_advisory",
            str(Path(str(module_policy["authority_root"])) / "results/v16_operator_advisory"),
            "PRESENT_DIRECTORY",
        ),
        (
            "candidate_results_v16",
            str(candidate / "results/v16"),
            "ABSENT",
        ),
        (
            "candidate_results_v16_operator_advisory",
            str(candidate / "results/v16_operator_advisory"),
            "ABSENT",
        ),
    ]
    for row, (label, path, state) in zip(
        protected_roots,
        expected_protected,
        strict=True,
    ):
        if (
            type(row) is not dict
            or set(row) != {"identity", "label", "path", "state"}
            or row.get("label") != label
            or row.get("path") != path
            or row.get("state") != state
            or (state == "ABSENT" and row.get("identity") is not None)
            or (state == "PRESENT_DIRECTORY" and type(row.get("identity")) is not dict)
        ):
            raise MainSuiteHarnessError("runtime policy protected roots are invalid")
    return value


def _read_policy(
    path: Path,
    *,
    repo_root: Path,
) -> tuple[
    dict[str, object],
    bytes,
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    expected_path = (
        repo_root / "quant_investor/v17_v2_contract/resources/main_suite_runtime_policy.v1.json"
    ).resolve(strict=False)
    if path.resolve(strict=False) != expected_path:
        raise MainSuiteHarnessError("runtime policy must use the canonical package path")
    binding, raw = _stable_file(path, max_bytes=MAX_POLICY_BYTES)
    manifest_path = repo_root / "quant_investor/v17_v2_contract/resources/package_manifest.v1.json"
    manifest_binding, manifest_raw = _stable_file(
        manifest_path,
        max_bytes=MAX_POLICY_BYTES,
    )
    try:
        manifest = json.loads(
            manifest_raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MainSuiteHarnessError("package manifest is invalid JSON") from exc
    if (
        type(manifest) is not dict
        or manifest_raw != _canonical_bytes(manifest) + b"\n"
        or manifest.get("version") != "myquant.v17.v2.package-manifest.v1"
        or manifest.get("authority") is not False
        or type(manifest.get("resources")) is not list
    ):
        raise MainSuiteHarnessError("package manifest identity is invalid")
    schema_path = (
        repo_root
        / "quant_investor/v17_v2_contract/schemas/main_suite_runtime_policy.v1.schema.json"
    )
    schema_binding, schema_raw = _stable_file(
        schema_path,
        max_bytes=MAX_POLICY_BYTES,
    )
    try:
        schema = json.loads(
            schema_raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MainSuiteHarnessError("runtime policy schema is invalid JSON") from exc
    if (
        type(schema) is not dict
        or schema_raw != _canonical_bytes(schema) + b"\n"
        or schema.get("$id") != POLICY_SCHEMA_ID
        or schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema"
    ):
        raise MainSuiteHarnessError("runtime policy schema identity is invalid")
    matches = [
        row
        for row in manifest["resources"]
        if type(row) is dict
        and row.get("relative_path") == "resources/main_suite_runtime_policy.v1.json"
    ]
    if (
        len(matches) != 1
        or matches[0].get("byte_sha256") != binding["sha256"]
        or matches[0].get("resource_version") != POLICY_VERSION
    ):
        raise MainSuiteHarnessError("package manifest does not bind runtime policy")
    schema_matches = [
        row
        for row in manifest.get("schemas", [])
        if type(row) is dict
        and row.get("relative_path") == "schemas/main_suite_runtime_policy.v1.schema.json"
    ]
    if (
        len(schema_matches) != 1
        or schema_matches[0].get("byte_sha256") != schema_binding["sha256"]
        or schema_matches[0].get("schema_id") != POLICY_SCHEMA_ID
    ):
        raise MainSuiteHarnessError("package manifest does not bind runtime policy schema")
    parsed_policy = _parse_policy_bytes(raw)
    _execute_bound_schema_validator(
        parsed_policy,
        schema,
        repo_root=repo_root,
    )
    validated_policy = validate_policy_bytes(raw)
    return (
        validated_policy,
        raw,
        binding,
        manifest_binding,
        schema_binding,
    )


def _execute_bound_schema_validator(
    instance: Mapping[str, object],
    schema: Mapping[str, object],
    *,
    repo_root: Path,
) -> None:
    import importlib.machinery
    import importlib.util
    from types import ModuleType
    import sys

    package_name = "_myquant_phase0_bound_schema"
    package_root = repo_root / "quant_investor/v17_v2_contract"
    bindings = (
        ("canonical", package_root / "canonical.py"),
        ("resources", package_root / "resources.py"),
        ("schema_validation", package_root / "schema_validation.py"),
    )
    package = ModuleType(package_name)
    package.__package__ = package_name
    package.__path__ = [str(package_root)]
    package.__spec__ = importlib.machinery.ModuleSpec(
        package_name,
        loader=None,
        is_package=True,
    )
    package.__spec__.submodule_search_locations = [str(package_root)]
    loaded_names = [package_name]
    sys.modules[package_name] = package
    try:
        loaded: dict[str, ModuleType] = {}
        for short_name, path in bindings:
            observed, _raw = _stable_file(path, max_bytes=MAX_POLICY_BYTES)
            if observed["path"] != str(path):
                raise MainSuiteHarnessError("schema validator binding drift")
            full_name = f"{package_name}.{short_name}"
            spec = importlib.util.spec_from_file_location(full_name, path)
            if spec is None or spec.loader is None:
                raise MainSuiteHarnessError("cannot load bound schema validator")
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            loaded_names.append(full_name)
            spec.loader.exec_module(module)
            loaded[short_name] = module
        validator = loaded["schema_validation"]
        validator.preflight_packaged_schema(schema)
        validator.validate_instance_against_schema(instance, schema)
    except MainSuiteHarnessError:
        raise
    except Exception as exc:
        raise MainSuiteHarnessError("runtime policy schema validation failed") from exc
    finally:
        for name in reversed(loaded_names):
            sys.modules.pop(name, None)


def _protected_root_snapshot(rows: object) -> list[dict[str, object]]:
    if type(rows) is not list:
        raise MainSuiteHarnessError("protected-root policy is invalid")
    observed_rows: list[dict[str, object]] = []
    for row in rows:
        if (
            type(row) is not dict
            or set(row) != {"identity", "label", "path", "state"}
            or type(row.get("path")) is not str
            or type(row.get("label")) is not str
            or row.get("state") not in {"ABSENT", "PRESENT_DIRECTORY"}
        ):
            raise MainSuiteHarnessError("protected-root row is invalid")
        path = Path(str(row["path"]))
        try:
            observed = path.lstat()
        except FileNotFoundError:
            exists = False
            identity = None
        except OSError as exc:
            raise MainSuiteHarnessError(f"cannot stat protected root: {path}") from exc
        else:
            exists = True
            identity = {
                "mode": f"{stat.S_IMODE(observed.st_mode):04o}",
                "st_dev": observed.st_dev,
                "st_ino": observed.st_ino,
                "uid": observed.st_uid,
            }
        expected_exists = row["state"] == "PRESENT_DIRECTORY"
        if exists is not expected_exists:
            raise MainSuiteHarnessError(f"protected-root state drift: {path}")
        if exists and (
            identity is None
            or not stat.S_ISDIR(observed.st_mode)
            or type(row.get("identity")) is not dict
        ):
            raise MainSuiteHarnessError(f"protected root is not a directory: {path}")
        if exists and type(row.get("identity")) is dict:
            for key, value in row["identity"].items():
                if identity is None or identity.get(key) != value:
                    raise MainSuiteHarnessError(f"protected-root identity drift: {path}")
        if not exists and row.get("identity") is not None:
            raise MainSuiteHarnessError(f"absent protected root has identity: {path}")
        observed_rows.append(
            {
                "identity": identity,
                "label": row["label"],
                "path": str(path),
                "state": row["state"],
            }
        )
    return observed_rows


def _empty_private_directory_binding(
    path: Path,
    *,
    label: str,
) -> dict[str, object]:
    try:
        before = path.lstat()
        with os.scandir(path) as entries:
            if next(entries, None) is not None:
                raise MainSuiteHarnessError(f"{label} is not empty")
        after = path.lstat()
    except MainSuiteHarnessError:
        raise
    except OSError as exc:
        raise MainSuiteHarnessError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISDIR(before.st_mode)
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) != 0o700
        or _stat_signature(before) != _stat_signature(after)
    ):
        raise MainSuiteHarnessError(f"{label} is not a stable owner-private directory")
    return {
        "gid": before.st_gid,
        "mode": f"{stat.S_IMODE(before.st_mode):04o}",
        "path": str(path),
        "st_ctime_ns": before.st_ctime_ns,
        "st_dev": before.st_dev,
        "st_ino": before.st_ino,
        "st_mtime_ns": before.st_mtime_ns,
        "st_nlink": before.st_nlink,
        "uid": before.st_uid,
    }


def _attach_pycache_binding(
    snapshot: Mapping[str, object],
    binding: Mapping[str, object],
) -> dict[str, object]:
    value = dict(snapshot)
    value.pop("snapshot_sha256", None)
    value["pycache_prefix"] = dict(binding)
    value["snapshot_sha256"] = _sha256(_canonical_bytes(value))
    return value


def capture_external_runtime(
    policy: Mapping[str, object],
    repo_root: Path,
) -> dict[str, object]:
    """Capture hash-bound state that the child cannot authoritatively self-report."""

    if type(policy) is not dict:
        raise MainSuiteHarnessError("runtime policy is invalid")
    candidate = Path(str(policy.get("candidate_root"))).resolve(strict=True)
    if Path(repo_root).resolve(strict=True) != candidate:
        raise MainSuiteHarnessError("candidate repository root mismatch")
    binding_rows: list[dict[str, object]] = []
    for label in ("wrapper_binding", "harness_binding", "candidate_conftest"):
        expected = policy.get(label)
        if type(expected) is not dict or type(expected.get("path")) is not str:
            raise MainSuiteHarnessError(f"{label} is invalid")
        observed, _raw = _stable_file(
            Path(str(expected["path"])),
            max_bytes=MAX_POLICY_BYTES,
        )
        _assert_binding(observed, expected, label=label)
        binding_rows.append({"label": label, **observed})
    for label, relative in (
        (
            "package_manifest",
            "quant_investor/v17_v2_contract/resources/package_manifest.v1.json",
        ),
        (
            "runtime_policy",
            "quant_investor/v17_v2_contract/resources/main_suite_runtime_policy.v1.json",
        ),
        (
            "runtime_policy_schema",
            "quant_investor/v17_v2_contract/schemas/main_suite_runtime_policy.v1.schema.json",
        ),
        (
            "schema_validator_canonical",
            "quant_investor/v17_v2_contract/canonical.py",
        ),
        (
            "schema_validator_resources",
            "quant_investor/v17_v2_contract/resources.py",
        ),
        (
            "schema_validator_runtime",
            "quant_investor/v17_v2_contract/schema_validation.py",
        ),
    ):
        observed, _raw = _stable_file(
            candidate / relative,
            max_bytes=MAX_POLICY_BYTES,
        )
        binding_rows.append({"label": label, **observed})
    main_runtime = policy.get("main_runtime")
    if type(main_runtime) is not dict:
        raise MainSuiteHarnessError("main runtime policy is invalid")
    lexical = _stable_symlink(Path(str(main_runtime.get("lexical_python"))))
    _assert_binding(
        lexical,
        main_runtime.get("lexical_python_binding"),
        label="main lexical interpreter",
    )
    resolved, _raw = _stable_file(
        Path(str(main_runtime.get("resolved_python_binding", {}).get("path"))),
        max_bytes=512 * 1024 * 1024,
    )
    _assert_binding(
        resolved,
        main_runtime.get("resolved_python_binding"),
        label="main resolved interpreter",
    )
    startup_rows: list[dict[str, object]] = []
    for row in main_runtime.get("startup_files", []):
        if type(row) is not dict or type(row.get("path")) is not str:
            raise MainSuiteHarnessError("startup-file policy is invalid")
        if row.get("present") is False:
            try:
                Path(str(row["path"])).lstat()
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise MainSuiteHarnessError("cannot inspect absent startup file") from exc
            else:
                raise MainSuiteHarnessError("absent startup file appeared")
            startup_rows.append({"path": row["path"], "present": False})
            continue
        observed, _raw = _stable_file(
            Path(str(row["path"])),
            max_bytes=512 * 1024 * 1024,
        )
        _assert_binding(observed, row, label=f"startup file {row['path']}")
        startup_rows.append({"present": True, **observed})
    site_packages = Path(str(main_runtime.get("site_packages_root")))
    try:
        physical_dist_info_names = sorted(
            (
                child.name
                for child in site_packages.iterdir()
                if child.name.endswith(".dist-info") and child.is_dir()
            ),
            key=lambda value: value.encode("utf-8"),
        )
    except OSError as exc:
        raise MainSuiteHarnessError("cannot scan main dist-info inventory") from exc
    valid_inventory = main_runtime.get("valid_inventory")
    if (
        type(valid_inventory) is not dict
        or len(physical_dist_info_names) != valid_inventory.get("physical_dist_info_count")
        or _sha256(_canonical_bytes(physical_dist_info_names))
        != valid_inventory.get("physical_dist_info_names_sha256")
    ):
        raise MainSuiteHarnessError("main physical dist-info inventory drift")
    invalid_dist_info: list[dict[str, object]] = []
    for row in main_runtime.get("invalid_dist_info", []):
        if type(row) is not dict or type(row.get("path")) is not str:
            raise MainSuiteHarnessError("invalid dist-info policy is invalid")
        root = Path(str(row["path"]))
        try:
            names = sorted(
                (child.name for child in root.iterdir()),
                key=lambda value: value.encode("utf-8"),
            )
        except OSError as exc:
            raise MainSuiteHarnessError("invalid dist-info stub is unavailable") from exc
        if names != row.get("child_names"):
            raise MainSuiteHarnessError("invalid dist-info stub names drift")
        files: list[dict[str, object]] = []
        for expected in row.get("files", []):
            if type(expected) is not dict or type(expected.get("path")) is not str:
                raise MainSuiteHarnessError("invalid dist-info file row is invalid")
            observed, _raw = _stable_file(
                Path(str(expected["path"])),
                max_bytes=MAX_POLICY_BYTES,
            )
            _assert_binding(observed, expected, label="invalid dist-info file")
            files.append(observed)
        invalid_dist_info.append({"child_names": names, "files": files, "path": str(root)})
    factor_rows: list[dict[str, object]] = []
    module_policy = policy.get("module_policy")
    if type(module_policy) is not dict:
        raise MainSuiteHarnessError("module policy is invalid")
    for row in policy.get("factor_authority_sources", []):
        if type(row) is not dict or type(row.get("relative_path")) is not str:
            raise MainSuiteHarnessError("factor source policy is invalid")
        for root_key in ("candidate_root", "authority_root"):
            root = policy.get(root_key)
            if root_key == "authority_root":
                root = module_policy.get("authority_root")
            if type(root) is not str:
                raise MainSuiteHarnessError("repository root policy is invalid")
            observed, _raw = _stable_file(
                Path(root) / str(row["relative_path"]),
                max_bytes=512 * 1024 * 1024,
            )
            if observed["sha256"] != row.get("sha256") or observed["size_bytes"] != row.get(
                "size_bytes"
            ):
                raise MainSuiteHarnessError("factor authority source drift")
        factor_rows.append(dict(row))
    ownership_rows: list[dict[str, object]] = []
    for row in module_policy.get("distribution_ownership", []):
        if type(row) is not dict:
            raise MainSuiteHarnessError("distribution ownership row is invalid")
        metadata, _raw = _stable_file(
            Path(str(row.get("metadata_binding", {}).get("path"))),
            max_bytes=MAX_POLICY_BYTES,
        )
        record, _raw = _stable_file(
            Path(str(row.get("record_binding", {}).get("path"))),
            max_bytes=MAX_POLICY_BYTES,
        )
        _assert_binding(metadata, row.get("metadata_binding"), label="distribution METADATA")
        _assert_binding(record, row.get("record_binding"), label="distribution RECORD")
        ownership_rows.append(
            {
                "metadata_sha256": metadata["sha256"],
                "name": row.get("name"),
                "record_sha256": record["sha256"],
                "version": row.get("version"),
            }
        )
    wrapper_path = Path(str(policy.get("wrapper_binding", {}).get("path")))
    wrapper_binding, wrapper_raw = _stable_file(
        wrapper_path,
        max_bytes=MAX_POLICY_BYTES,
    )
    _assert_binding(
        wrapper_binding,
        policy.get("wrapper_binding"),
        label="external wrapper",
    )
    namespace: dict[str, object] = {
        "__file__": str(wrapper_path),
        "__name__": "_myquant_phase0_bound_wrapper",
    }
    try:
        exec(compile(wrapper_raw, str(wrapper_path), "exec"), namespace)
        tree_inventory = namespace["_tree_inventory"]
    except Exception as exc:
        raise MainSuiteHarnessError("cannot load bound tree scanner") from exc
    tree_rows: list[dict[str, object]] = []
    tree_policies: list[tuple[str, object, object]] = []
    for row in policy.get("pytest_support_trees", []):
        if type(row) is not dict:
            raise MainSuiteHarnessError("pytest support-tree row is invalid")
        tree_policies.append((str(row.get("name")), row.get("roots"), row.get("descriptor")))
    for row in policy.get("pytest_plugins", []):
        if type(row) is not dict:
            raise MainSuiteHarnessError("pytest plugin row is invalid")
        tree_policies.append(
            (
                f"plugin:{row.get('entry_point_name')}",
                row.get("physical_tree_roots"),
                row.get("physical_tree"),
            )
        )
    for name, roots, expected in tree_policies:
        if not callable(tree_inventory):
            raise MainSuiteHarnessError("bound tree scanner is not callable")
        observed = tree_inventory(roots)
        if type(expected) is not dict or observed != expected:
            raise MainSuiteHarnessError(f"bound physical tree drift: {name}")
        tree_rows.append({"descriptor": observed, "name": name})
    protected = _protected_root_snapshot(policy.get("protected_roots"))
    payload: dict[str, object] = {
        "bindings": binding_rows,
        "factor_authority_sha256": _sha256(_canonical_bytes(factor_rows)),
        "invalid_dist_info_sha256": _sha256(_canonical_bytes(invalid_dist_info)),
        "lexical_python": lexical,
        "protected_roots": protected,
        "resolved_python": resolved,
        "startup_files": startup_rows,
        "distribution_ownership_sha256": _sha256(_canonical_bytes(ownership_rows)),
        "physical_trees": tree_rows,
    }
    payload["snapshot_sha256"] = _sha256(_canonical_bytes(payload))
    return payload


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise MainSuiteHarnessError("challenge write failed")
        view = view[written:]


def _decode_frame(
    raw: bytes,
    *,
    expected_phase: int,
    nonce: bytes,
    challenge_sha: bytes,
    child_pid: int,
    parent_pid: int,
) -> dict[str, object]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MainSuiteHarnessError("attestation payload is invalid JSON") from exc
    if type(value) is not dict or raw != _canonical_bytes(value):
        raise MainSuiteHarnessError("attestation payload is not canonical JSON")
    expected_name = {
        1: "pre_import",
        2: "pre_collection",
        3: "terminal_complete",
    }.get(expected_phase)
    if (
        value.get("frame") != expected_name
        or value.get("challenge_binding_sha256") != challenge_sha.hex()
        or value.get("pid") != child_pid
        or value.get("ppid") != parent_pid
    ):
        raise MainSuiteHarnessError("attestation payload identity mismatch")
    if expected_phase == 3 and (
        type(value.get("pytest_exit_code")) is not int
        or value["pytest_exit_code"] not in {0, 1, 2, 3, 4, 5}
        or type(value.get("final_loaded_modules")) is not dict
    ):
        raise MainSuiteHarnessError("terminal attestation payload is invalid")
    return value


def _consume_frames(
    buffer: bytearray,
    offset: int,
    frames: list[dict[str, object]],
    *,
    nonce: bytes,
    challenge_sha: bytes,
    child_pid: int,
    parent_pid: int,
) -> int:
    while len(frames) < 3 and len(buffer) - offset >= ATTEST_HEADER.size:
        header = bytes(buffer[offset : offset + ATTEST_HEADER.size])
        magic, version, phase, reserved, size, frame_nonce, digest = ATTEST_HEADER.unpack(header)
        expected_phase = len(frames) + 1
        if (
            magic != ATTEST_MAGIC
            or version != PROTOCOL_VERSION
            or phase != expected_phase
            or reserved != 0
            or size < 2
            or size > MAX_FRAME_BYTES
            or (expected_phase == 3 and size > MAX_TERMINAL_FRAME_BYTES)
            or frame_nonce != nonce
        ):
            raise MainSuiteHarnessError("attestation frame header mismatch")
        end = offset + ATTEST_HEADER.size + size
        if len(buffer) < end:
            break
        payload_raw = bytes(buffer[offset + ATTEST_HEADER.size : end])
        if hashlib.sha256(payload_raw).digest() != digest:
            raise MainSuiteHarnessError("attestation frame digest mismatch")
        payload = _decode_frame(
            payload_raw,
            expected_phase=expected_phase,
            nonce=nonce,
            challenge_sha=challenge_sha,
            child_pid=child_pid,
            parent_pid=parent_pid,
        )
        frames.append(
            {
                "payload": payload,
                "payload_sha256": digest.hex(),
                "payload_size_bytes": size,
                "phase": expected_phase,
            }
        )
        offset = end
    if len(frames) == 3 and len(buffer) != offset:
        raise MainSuiteHarnessError("attestation stream has trailing bytes")
    return offset


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _timeout_value(policy: Mapping[str, object], name: str) -> int:
    timeouts = policy.get("timeouts")
    value = None if type(timeouts) is not dict else timeouts.get(name)
    if type(value) is not int or not 1 <= value <= 21_600:
        raise MainSuiteHarnessError(f"runtime timeout is invalid: {name}")
    return value


def run_main_suite(
    *,
    repo_root: Path,
    policy_path: Path,
    challenge_binding_kind: str,
    challenge_binding_sha256: str,
    environment: Mapping[str, str],
    pytest_args: Sequence[str],
) -> dict[str, object]:
    """Run and attest the sole main-worktree-interpreter full pytest suite."""

    if challenge_binding_kind not in {"SKIP_SOURCE_STATE", "PHASE0_SESSION_FILE"}:
        raise MainSuiteHarnessError("challenge binding kind is invalid")
    if SHA256_RE.fullmatch(challenge_binding_sha256) is None:
        raise MainSuiteHarnessError("challenge binding SHA-256 is invalid")
    if (
        not environment
        or any(type(key) is not str or type(value) is not str for key, value in environment.items())
        or CHALLENGE_ENV in environment
        or ATTEST_ENV in environment
    ):
        raise MainSuiteHarnessError("child environment is invalid")
    candidate = Path(repo_root).resolve(strict=True)
    (
        policy,
        policy_raw,
        policy_binding,
        manifest_binding,
        policy_schema_binding,
    ) = _read_policy(
        Path(policy_path),
        repo_root=candidate,
    )
    runtime_module_closures = policy.get("module_closures")
    limitations = policy.get("limitations")
    if type(runtime_module_closures) is not dict or type(limitations) is not list:
        raise MainSuiteHarnessError("runtime policy module closures are invalid")
    if candidate != Path(str(policy["candidate_root"])).resolve(strict=True):
        raise MainSuiteHarnessError("candidate repository does not match policy")
    if list(pytest_args) != policy.get("pytest_args"):
        raise MainSuiteHarnessError("pytest arguments do not match policy")
    pytest_environment = policy.get("pytest_environment")
    if type(pytest_environment) is not dict:
        raise MainSuiteHarnessError("pytest environment policy is invalid")
    required = pytest_environment.get("required")
    forbidden = pytest_environment.get("forbidden")
    allowed_keys = pytest_environment.get("allowed_keys")
    dynamic_path_keys = pytest_environment.get("dynamic_path_keys")
    if (
        type(required) is not dict
        or type(forbidden) is not list
        or type(allowed_keys) is not list
        or type(dynamic_path_keys) is not list
    ):
        raise MainSuiteHarnessError("pytest environment policy is invalid")
    if set(environment) != set(allowed_keys):
        raise MainSuiteHarnessError("child environment key set does not match policy")
    for key, value in required.items():
        if environment.get(key) != value:
            raise MainSuiteHarnessError(f"required environment mismatch: {key}")
    for key in forbidden:
        if key in environment:
            raise MainSuiteHarnessError(f"forbidden environment is set: {key}")
    for key in dynamic_path_keys:
        value = environment.get(str(key))
        if (
            type(value) is not str
            or "\0" in value
            or not Path(value).is_absolute()
            or os.path.normpath(value) != value
        ):
            raise MainSuiteHarnessError(f"dynamic environment path is invalid: {key}")
    home = Path(environment["HOME"])
    tmpdir = Path(environment["TMPDIR"])
    cache = Path(environment["XDG_CACHE_HOME"])
    if (
        home.parent != tmpdir.parent
        or home.parent != cache.parent
        or Path(environment["BLACK_CACHE_DIR"]).parent != cache
        or Path(environment["MYPY_CACHE_DIR"]).parent != cache
        or Path(environment["PYTHONPYCACHEPREFIX"]).parent != cache
    ):
        raise MainSuiteHarnessError("dynamic environment paths do not share closed root")
    for label, path in (("HOME", home), ("TMPDIR", tmpdir), ("XDG_CACHE_HOME", cache)):
        try:
            observed = path.lstat()
        except OSError as exc:
            raise MainSuiteHarnessError(f"{label} is unavailable") from exc
        if (
            not stat.S_ISDIR(observed.st_mode)
            or observed.st_uid != os.getuid()
            or stat.S_IMODE(observed.st_mode) != 0o700
        ):
            raise MainSuiteHarnessError(f"{label} is not owner-private")
    pycache_prefix = Path(environment["PYTHONPYCACHEPREFIX"])
    pycache_before = _empty_private_directory_binding(
        pycache_prefix,
        label="PYTHONPYCACHEPREFIX",
    )

    nonce = secrets.token_bytes(32)
    challenge_sha = bytes.fromhex(challenge_binding_sha256)
    challenge = CHALLENGE_STRUCT.pack(
        CHALLENGE_MAGIC,
        PROTOCOL_VERSION,
        b"\0\0\0",
        nonce,
        challenge_sha,
    )
    main_runtime = policy["main_runtime"]
    wrapper = policy["wrapper_binding"]
    if type(main_runtime) is not dict or type(wrapper) is not dict:
        raise MainSuiteHarnessError("main launch policy is invalid")
    argv = [
        str(main_runtime["lexical_python"]),
        "-I",
        "-S",
        "-B",
        "-X",
        f"pycache_prefix={environment['PYTHONPYCACHEPREFIX']}",
        str(wrapper["path"]),
        str(policy_path),
        str(policy_binding["sha256"]),
        "--",
        *list(pytest_args),
    ]
    child_environment = dict(environment)
    process: subprocess.Popen[bytes] | None = None
    selector: selectors.BaseSelector | None = None
    challenge_read = -1
    challenge_write = -1
    attest_read = -1
    attest_write = -1
    stdout = bytearray()
    stderr = bytearray()
    attest = bytearray()
    frames: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    before: dict[str, object] | None = None
    after: dict[str, object] | None = None
    signal_number: int | None = None
    launch_ns: int | None = None
    phase1_ns: int | None = None
    phase2_ns: int | None = None
    raw_return_code: int | None = None
    normalized_exit_code = 2
    pipes_drained = False
    attest_offset = 0

    def fail(phase: str, code: str, detail: object) -> None:
        if phase not in {"PRIMARY", "CLEANUP", "EXTERNAL_AFTER"}:
            raise MainSuiteHarnessError("invalid failure phase")
        text = str(detail).replace("\r", " ").replace("\n", " ")[:1024]
        failures.append({"code": code, "detail": text, "phase": phase})

    try:
        try:
            before = _attach_pycache_binding(
                capture_external_runtime(policy, candidate),
                pycache_before,
            )
        except Exception as exc:  # fail-closed evidence capture
            fail("PRIMARY", "EXTERNAL_BEFORE_FAILED", exc)
        if not failures:
            try:
                challenge_read, challenge_write = os.pipe()
                attest_read, attest_write = os.pipe()
                for descriptor in (
                    challenge_read,
                    challenge_write,
                    attest_read,
                    attest_write,
                ):
                    flags = fcntl.fcntl(descriptor, fcntl.F_GETFD)
                    fcntl.fcntl(
                        descriptor,
                        fcntl.F_SETFD,
                        flags | fcntl.FD_CLOEXEC,
                    )
                child_environment[CHALLENGE_ENV] = str(challenge_read)
                child_environment[ATTEST_ENV] = str(attest_write)
                popen_started = time.monotonic()
                launch_ns = time.monotonic_ns()
                process = subprocess.Popen(
                    argv,
                    cwd=candidate,
                    env=child_environment,
                    shell=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    pass_fds=(challenge_read, attest_write),
                    start_new_session=True,
                )
                phase1_deadline = popen_started + _timeout_value(
                    policy,
                    "phase1_seconds",
                )
                os.close(challenge_read)
                challenge_read = -1
                os.close(attest_write)
                attest_write = -1
                _write_all(challenge_write, challenge)
                os.close(challenge_write)
                challenge_write = -1
                if process.stdout is None or process.stderr is None:  # pragma: no cover
                    raise MainSuiteHarnessError("child capture pipes were not created")
                selector = selectors.DefaultSelector()
                streams = {
                    process.stdout.fileno(): ("stdout", stdout),
                    process.stderr.fileno(): ("stderr", stderr),
                    attest_read: ("attest", attest),
                }
                for descriptor in streams:
                    os.set_blocking(descriptor, False)
                    selector.register(descriptor, selectors.EVENT_READ)
                phase2_deadline: float | None = None
                suite_deadline: float | None = None
                cleanup_deadline: float | None = None
                leader_exit_seen: float | None = None
                overflowed_labels: set[str] = set()
                timeouts = policy.get("timeouts")
                if type(timeouts) is not dict:
                    raise MainSuiteHarnessError("runtime timeout policy is invalid")
                max_stream = int(str(timeouts["max_stream_bytes"]))
                max_attest = int(str(timeouts["max_attestation_bytes"]))
                while selector.get_map():
                    now = time.monotonic()
                    if process.poll() is not None and leader_exit_seen is None:
                        leader_exit_seen = now
                    active_deadline = (
                        phase1_deadline
                        if len(frames) == 0
                        else phase2_deadline if len(frames) == 1 else suite_deadline
                    )
                    if active_deadline is None:
                        raise MainSuiteHarnessError("phase deadline is uninitialized")
                    drain_deadline = (
                        None
                        if leader_exit_seen is None
                        else leader_exit_seen + _timeout_value(policy, "cleanup_seconds")
                    )
                    deadline_candidates = (
                        (cleanup_deadline, drain_deadline)
                        if cleanup_deadline is not None
                        else (active_deadline, drain_deadline)
                    )
                    deadline = min(value for value in deadline_candidates if value is not None)
                    if now >= deadline:
                        if drain_deadline is not None and deadline == drain_deadline:
                            fail(
                                "CLEANUP",
                                "PIPE_DRAIN_TIMEOUT",
                                "leader exited while inherited pipes remained open",
                            )
                        elif cleanup_deadline is None:
                            code = (
                                "PHASE1_TIMEOUT"
                                if len(frames) == 0
                                else "PHASE2_TIMEOUT" if len(frames) == 1 else "SUITE_TIMEOUT"
                            )
                            fail("PRIMARY", code, "phase deadline expired")
                            cleanup_deadline = now + _timeout_value(policy, "cleanup_seconds")
                        else:
                            fail(
                                "CLEANUP",
                                "PIPE_DRAIN_TIMEOUT",
                                "pipes remained open after process-group termination",
                            )
                        _kill_process_group(process)
                        if any(failure["code"] == "PIPE_DRAIN_TIMEOUT" for failure in failures):
                            break
                        continue
                    events = selector.select(min(1.0, max(0.0, deadline - now)))
                    for key, _mask in events:
                        descriptor = int(key.fd)
                        label, buffer = streams[descriptor]
                        try:
                            chunk = os.read(descriptor, 64 * 1024)
                        except BlockingIOError:
                            continue
                        if not chunk:
                            selector.unregister(descriptor)
                            if label == "attest" and len(frames) != 3:
                                fail(
                                    "PRIMARY",
                                    "ATTESTATION_INCOMPLETE",
                                    "attestation pipe closed before three frames",
                                )
                                _kill_process_group(process)
                                cleanup_deadline = time.monotonic() + _timeout_value(
                                    policy, "cleanup_seconds"
                                )
                            continue
                        if label in overflowed_labels:
                            continue
                        cap = max_attest if label == "attest" else max_stream
                        remaining_capacity = max(0, cap - len(buffer))
                        buffer.extend(chunk[:remaining_capacity])
                        if len(chunk) > remaining_capacity:
                            overflowed_labels.add(label)
                            fail(
                                "PRIMARY",
                                f"{label.upper()}_OVERFLOW",
                                "captured stream exceeded policy cap",
                            )
                            _kill_process_group(process)
                            cleanup_deadline = time.monotonic() + _timeout_value(
                                policy, "cleanup_seconds"
                            )
                            continue
                        if label == "attest":
                            try:
                                old_count = len(frames)
                                attest_offset = _consume_frames(
                                    attest,
                                    attest_offset,
                                    frames,
                                    nonce=nonce,
                                    challenge_sha=challenge_sha,
                                    child_pid=process.pid,
                                    parent_pid=os.getpid(),
                                )
                                if old_count == 0 and len(frames) >= 1:
                                    phase1_ns = time.monotonic_ns()
                                    phase2_deadline = time.monotonic() + _timeout_value(
                                        policy, "phase2_seconds"
                                    )
                                if old_count < 2 and len(frames) >= 2:
                                    phase2_ns = time.monotonic_ns()
                                    suite_deadline = time.monotonic() + _timeout_value(
                                        policy, "suite_seconds"
                                    )
                            except MainSuiteHarnessError as exc:
                                fail(
                                    "PRIMARY",
                                    "ATTESTATION_PROTOCOL_ERROR",
                                    exc,
                                )
                                _kill_process_group(process)
                                cleanup_deadline = time.monotonic() + _timeout_value(
                                    policy, "cleanup_seconds"
                                )
                pipes_drained = not selector.get_map()
                if process.poll() is None:
                    active_deadline = (
                        phase1_deadline
                        if len(frames) == 0
                        else phase2_deadline if len(frames) == 1 else suite_deadline
                    )
                    remaining = (
                        0.0
                        if active_deadline is None
                        else max(
                            0.0,
                            active_deadline - time.monotonic(),
                        )
                    )
                    try:
                        raw_return_code = process.wait(timeout=remaining)
                    except subprocess.TimeoutExpired:
                        fail("PRIMARY", "SUITE_TIMEOUT", "child did not exit")
                        _kill_process_group(process)
                else:
                    raw_return_code = process.returncode
            except Exception as exc:
                fail("PRIMARY", "HARNESS_EXECUTION_FAILURE", exc)
    finally:
        if selector is not None:
            try:
                selector.close()
            except Exception as exc:
                fail("CLEANUP", "SELECTOR_CLOSE_FAILED", exc)
        for descriptor in (challenge_read, challenge_write, attest_read, attest_write):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError as exc:
                    fail("CLEANUP", "FD_CLOSE_FAILED", exc)
        if process is not None:
            if process.stdout is not None:
                try:
                    process.stdout.close()
                except Exception as exc:
                    fail("CLEANUP", "STDOUT_CLOSE_FAILED", exc)
            if process.stderr is not None:
                try:
                    process.stderr.close()
                except Exception as exc:
                    fail("CLEANUP", "STDERR_CLOSE_FAILED", exc)
            if process.poll() is None:
                try:
                    _kill_process_group(process)
                except OSError as exc:
                    fail("CLEANUP", "PROCESS_GROUP_KILL_FAILED", exc)
            try:
                raw_return_code = process.wait(timeout=_timeout_value(policy, "cleanup_seconds"))
            except subprocess.TimeoutExpired as exc:
                fail("CLEANUP", "PROCESS_REAP_TIMEOUT", exc)
                try:
                    _kill_process_group(process)
                except OSError as kill_exc:
                    fail("CLEANUP", "PROCESS_GROUP_KILL_FAILED", kill_exc)
                try:
                    raw_return_code = process.wait(
                        timeout=_timeout_value(policy, "cleanup_seconds")
                    )
                except subprocess.TimeoutExpired as reap_exc:
                    fail("CLEANUP", "FINAL_PROCESS_REAP_TIMEOUT", reap_exc)
                except Exception as reap_exc:
                    fail("CLEANUP", "FINAL_PROCESS_REAP_FAILED", reap_exc)
            except Exception as exc:
                fail("CLEANUP", "PROCESS_REAP_FAILED", exc)
        try:
            pycache_after = _empty_private_directory_binding(
                pycache_prefix,
                label="PYTHONPYCACHEPREFIX",
            )
            after = _attach_pycache_binding(
                capture_external_runtime(policy, candidate),
                pycache_after,
            )
        except Exception as exc:
            fail("EXTERNAL_AFTER", "EXTERNAL_AFTER_FAILED", exc)
    if before is not None and after is not None and before != after:
        fail("EXTERNAL_AFTER", "EXTERNAL_RUNTIME_DRIFT", "before/after differ")
    if raw_return_code is not None:
        if raw_return_code < 0:
            signal_number = -raw_return_code
            normalized_exit_code = 128 + signal_number
        else:
            normalized_exit_code = raw_return_code
    terminal_payload = frames[2].get("payload") if len(frames) == 3 else None
    phase3_verified = (
        len(frames) == 3
        and type(terminal_payload) is dict
        and terminal_payload.get("pytest_exit_code") == normalized_exit_code
        and terminal_payload.get("final_loaded_modules") == runtime_module_closures["final"]
    )
    if len(frames) == 3 and not phase3_verified:
        fail(
            "PRIMARY",
            "TERMINAL_ATTESTATION_MISMATCH",
            "terminal exit or final module closure differs from policy",
        )
    if (
        process is not None
        and normalized_exit_code != 0
        and not any(
            failure["code"]
            in {
                "HARNESS_EXECUTION_FAILURE",
                "PHASE1_TIMEOUT",
                "PHASE2_TIMEOUT",
                "SUITE_TIMEOUT",
            }
            for failure in failures
        )
    ):
        fail(
            "PRIMARY",
            "WRAPPER_FAILURE" if normalized_exit_code == 86 else "PYTEST_NONZERO",
            f"child exit code {normalized_exit_code}",
        )
    if process is not None and (attest_offset != len(attest) or len(frames) != 3):
        if not any(failure["code"] == "ATTESTATION_INCOMPLETE" for failure in failures):
            fail(
                "PRIMARY",
                "ATTESTATION_INCOMPLETE",
                "attestation stream did not contain exactly three frames",
            )
    indexed_failures = list(enumerate(failures))
    phase_order = {"PRIMARY": 0, "CLEANUP": 1, "EXTERNAL_AFTER": 2}
    failures = [
        failure
        for _index, failure in sorted(
            indexed_failures,
            key=lambda item: (phase_order[item[1]["phase"]], item[0]),
        )
    ]
    failure_codes: list[str] = []
    for failure in failures:
        if failure["code"] not in failure_codes:
            failure_codes.append(failure["code"])
    accepted = (
        not failures
        and normalized_exit_code == 0
        and len(frames) == 3
        and attest_offset == len(attest)
        and pipes_drained
        and before == after
    )
    final_audit_completed = phase3_verified and attest_offset == len(attest)
    cleanup_failed = any(failure["phase"] == "CLEANUP" for failure in failures)
    external_after_failed = any(failure["phase"] == "EXTERNAL_AFTER" for failure in failures)
    external_equal = None if before is None or after is None else before == after
    stdout_bytes = bytes(stdout)
    stderr_bytes = bytes(stderr)
    attest_bytes = bytes(attest)
    stdout_offset = 8
    stderr_offset = stdout_offset + len(stdout_bytes) + 8
    attest_offset_in_tail = stderr_offset + len(stderr_bytes) + 8
    framed_tail = (
        struct.pack(">Q", len(stdout_bytes))
        + stdout_bytes
        + struct.pack(">Q", len(stderr_bytes))
        + stderr_bytes
        + struct.pack(">Q", len(attest_bytes))
        + attest_bytes
    )
    receipt = _seal(
        {
            "accepted": accepted,
            "attestations": frames,
            "authority": False,
            "claims": {
                "exit_code": normalized_exit_code,
                "final_audit_completed": final_audit_completed,
                "final_audit_enforced": process is not None,
                "kernel_egress_attested": False,
                "network_unreachability_proven": False,
                "offline_policy_enforced": True,
                "signal": signal_number,
            },
            "command": {
                "argv": argv,
                "cwd": str(candidate),
                "environment": dict(sorted(environment.items())),
            },
            "external_after": after,
            "external_before": before,
            "failures": failures,
            "failure_codes": failure_codes,
            "finalization": {
                "cleanup": {
                    "attempted": True,
                    "status": "FAILED" if cleanup_failed else "PASSED",
                },
                "external_after": {
                    "attempted": True,
                    "equal": external_equal,
                    "status": "FAILED" if external_after_failed else "PASSED",
                },
            },
            "framing": "UINT64_BE_STDOUT_THEN_STDOUT_UINT64_BE_STDERR_THEN_STDERR_UINT64_BE_ATTESTATION_THEN_ATTESTATION",
            "limitations": list(limitations),
            "outcome": "PASSED" if accepted else "FAILED",
            "policy_binding": policy_binding,
            "policy_manifest_binding": manifest_binding,
            "policy_schema_binding": policy_schema_binding,
            "protocol_version": "myquant.v17.v2",
            "schema_id": RECEIPT_SCHEMA_ID,
            "challenge_binding": {
                "kind": challenge_binding_kind,
                "sha256": challenge_binding_sha256,
            },
            "timing": {
                "phase1_elapsed_ms": (
                    None if phase1_ns is None else (phase1_ns - int(launch_ns)) // 1_000_000
                ),
                "phase2_elapsed_ms": (
                    None if phase2_ns is None else (phase2_ns - int(launch_ns)) // 1_000_000
                ),
            },
            "streams": {
                "attestation": {
                    "offset_bytes": attest_offset_in_tail,
                    "sha256": _sha256(attest_bytes),
                    "size_bytes": len(attest_bytes),
                },
                "stderr": {
                    "offset_bytes": stderr_offset,
                    "sha256": _sha256(stderr_bytes),
                    "size_bytes": len(stderr_bytes),
                },
                "stdout": {
                    "offset_bytes": stdout_offset,
                    "sha256": _sha256(stdout_bytes),
                    "size_bytes": len(stdout_bytes),
                },
                "tail_sha256": _sha256(framed_tail),
                "tail_size_bytes": len(framed_tail),
            },
            "version": RECEIPT_VERSION,
        }
    )
    raw = RECEIPT_PREFIX + _canonical_bytes(receipt) + b"\n" + framed_tail
    return {
        "attestation": attest_bytes,
        "raw": raw,
        "receipt": receipt,
        "stderr": stderr_bytes,
        "stdout": stdout_bytes,
    }


__all__ = [
    "MainSuiteHarnessError",
    "RECEIPT_PREFIX",
    "capture_external_runtime",
    "run_main_suite",
    "validate_policy_bytes",
]
