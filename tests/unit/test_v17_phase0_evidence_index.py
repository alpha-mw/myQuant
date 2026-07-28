from __future__ import annotations

from collections.abc import Callable
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).parents[2]


def _load_subject() -> ModuleType:
    path = REPO_ROOT / "scripts/v17_phase0_evidence_index.py"
    spec = importlib.util.spec_from_file_location("v17_phase0_evidence_index", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


subject = _load_subject()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _main_suite_attestation_frame(
    phase: int,
    payload: bytes,
    *,
    nonce: bytes = b"n" * 32,
) -> bytes:
    return (
        subject.MAIN_SUITE_ATTEST_HEADER.pack(
            subject.MAIN_SUITE_ATTEST_MAGIC,
            subject.MAIN_SUITE_ATTEST_PROTOCOL_VERSION,
            phase,
            0,
            len(payload),
            nonce,
            hashlib.sha256(payload).digest(),
        )
        + payload
    )


def _main_suite_attestation(
    payloads: tuple[bytes, bytes, bytes] = (b"{}", b"{}", b"{}"),
) -> bytes:
    return b"".join(
        _main_suite_attestation_frame(phase, payload)
        for phase, payload in enumerate(payloads, start=1)
    )


def _minimal_command_schema() -> dict[str, object]:
    return {
        "$id": subject.COMMAND_RECEIPT_SCHEMA_ID,
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": True,
        "properties": {
            "version": {"const": subject.COMMAND_RECEIPT_VERSION},
        },
        "required": ["version"],
        "type": "object",
    }


def _schemas_for(
    artifact_version: str,
    schema: dict[str, object],
) -> dict[str, dict[str, object]]:
    return {
        "__repo_root__": {"path": str(REPO_ROOT)},
        artifact_version: schema,
    }


def _source_binding() -> dict[str, str]:
    digest = "1" * 64
    return {
        "base_commit": "2" * 40,
        "binary_diff_sha256": digest,
        "porcelain_sha256": digest,
        "source_state_sha256": digest,
        "untracked_inventory_sha256": digest,
    }


def _toolchain() -> dict[str, Any]:
    return {
        "base_python": {
            "executable": True,
            "implementation": "cpython",
            "lexical_path": (
                "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/"
                "Python.framework/Versions/3.13/bin/python3.13"
            ),
            "mode": "0755",
            "realpath": (
                "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/"
                "Python.framework/Versions/3.13/bin/python3.13"
            ),
            "sha256": ("a708f6e9f4803b806b29146c4e0feecfd9bf2d9eb60f3e15b850cd7cb56f200b"),
            "size_bytes": 52_640,
            "version": "3.13.7",
            "version_info": [3, 13, 7],
        },
        "pip_scope": copy.deepcopy(subject.EXPECTED_PIP_SCOPE),
        "uv": {
            "executable": True,
            "lexical_path": "/Users/maxwell/.local/bin/uv",
            "mode": "0755",
            "output": subject.PACKAGE_EXPECTED_UV_OUTPUT,
            "realpath": "/Users/maxwell/.local/bin/uv",
            "sha256": ("bc50ab0e90f24491f0e794f5b8649722f8fd2bf483c53490c012b41b89151ef9"),
            "size_bytes": 44_698_848,
            "version": "0.10.9",
        },
        "uv_cache": {
            "mode": "0700",
            "path": "/Users/maxwell/.cache/uv",
            "realpath": "/Users/maxwell/.cache/uv",
            "st_dev": 1,
            "st_ino": 2,
            "uid": 501,
        },
    }


def _protected_roots_absent() -> list[dict[str, object]]:
    return [
        {"id": root_id, "path": str(path), "state": "ABSENT"}
        for root_id, path in subject.PROTECTED_ROOT_SPECS
    ]


def _command_receipt(
    *,
    stdout: bytes = b"12 passed in 0.01s\n",
    stderr: bytes = b"",
) -> tuple[dict[str, object], bytes]:
    framed = struct.pack(">Q", len(stdout)) + stdout + struct.pack(">Q", len(stderr)) + stderr
    command = {
        "argv": ["/private/tmp/python", "-I", "-B", "-m", "pytest"],
        "cwd": str(REPO_ROOT),
        "environment": {
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "exit_code": 0,
        "ordinal": 1,
        "signal": None,
        "stderr_offset_bytes": 16 + len(stdout),
        "stderr_sha256": _sha(stderr),
        "stderr_size_bytes": len(stderr),
        "stdout_offset_bytes": 8,
        "stdout_sha256": _sha(stdout),
        "stdout_size_bytes": len(stdout),
        "tool_version": "pytest 9.0.2",
    }
    unsealed = {
        "claims": {
            "errors": 0,
            "exit_code": 0,
            "failed": 0,
            "passed": 12,
            "skipped": 0,
            "xfail": 0,
            "xpass": 0,
        },
        "commands": [command],
        "failure_codes": [],
        "framing": subject.COMMAND_FRAMING,
        "limitations": list(subject.NORMATIVE_LIMITATIONS),
        "outcome": "PASSED",
        "output_sha256": _sha(framed),
        "output_size_bytes": len(framed),
        "package_source_superset_after": {"row_count": 1, "sha256": "3" * 64},
        "package_source_superset_before": {"row_count": 1, "sha256": "3" * 64},
        "producer": {
            "path": str(REPO_ROOT / "scripts/v17_phase0_evidence_session.py"),
            "sha256": "4" * 64,
            "size_bytes": 1,
            "version": subject.SESSION_PRODUCER_VERSION,
        },
        "protected_roots_after": _protected_roots_absent(),
        "protected_roots_before": _protected_roots_absent(),
        "protocol_version": subject.PROTOCOL_VERSION,
        "session_binding": {
            "path": "/private/tmp/00_session.json",
            "semantic_sha256": "5" * 64,
            "session_id": "session-1",
            "sha256": "6" * 64,
            "size_bytes": 1,
        },
        "source_after": _source_binding(),
        "source_before": _source_binding(),
        "step": {
            "filename": "30_v2_tests.log",
            "kind": "log",
            "ordinal": 3,
            "role": "v2_evidence_tests",
        },
        "toolchain_after": _toolchain(),
        "toolchain_before": _toolchain(),
        "version": subject.COMMAND_RECEIPT_VERSION,
    }
    receipt = subject._seal(unsealed)
    raw = subject.COMMAND_RECEIPT_PREFIX + subject._canonical_bytes(receipt) + b"\n" + framed
    return receipt, raw


def _main_suite_receipt(
    *,
    stdout: bytes = b"1 passed in 0.01s\n",
    stderr: bytes = b"",
    attestation: bytes | None = None,
) -> tuple[dict[str, object], bytes]:
    if attestation is None:
        attestation = _main_suite_attestation()
    stdout_offset = 8
    stderr_offset = stdout_offset + len(stdout) + 8
    attestation_offset = stderr_offset + len(stderr) + 8
    framed = (
        struct.pack(">Q", len(stdout))
        + stdout
        + struct.pack(">Q", len(stderr))
        + stderr
        + struct.pack(">Q", len(attestation))
        + attestation
    )
    receipt = subject._seal(
        {
            "framing": subject.MAIN_SUITE_FRAMING,
            "schema_id": subject.MAIN_SUITE_RECEIPT_SCHEMA_ID,
            "streams": {
                "attestation": {
                    "offset_bytes": attestation_offset,
                    "sha256": _sha(attestation),
                    "size_bytes": len(attestation),
                },
                "stderr": {
                    "offset_bytes": stderr_offset,
                    "sha256": _sha(stderr),
                    "size_bytes": len(stderr),
                },
                "stdout": {
                    "offset_bytes": stdout_offset,
                    "sha256": _sha(stdout),
                    "size_bytes": len(stdout),
                },
                "tail_sha256": _sha(framed),
                "tail_size_bytes": len(framed),
            },
            "version": subject.MAIN_SUITE_RECEIPT_VERSION,
        }
    )
    return receipt, _main_suite_raw(receipt, framed)


def _main_suite_raw(receipt: dict[str, object], framed: bytes) -> bytes:
    return subject.MAIN_SUITE_RECEIPT_PREFIX + subject._canonical_bytes(receipt) + b"\n" + framed


def _reseal_main_suite(
    receipt: dict[str, object],
    framed: bytes,
    *,
    bind_tail: bool,
) -> tuple[dict[str, object], bytes]:
    unsealed = copy.deepcopy(receipt)
    unsealed.pop("semantic_sha256")
    if bind_tail:
        streams = unsealed["streams"]
        assert isinstance(streams, dict)
        streams["tail_sha256"] = _sha(framed)
        streams["tail_size_bytes"] = len(framed)
    sealed = subject._seal(unsealed)
    return sealed, _main_suite_raw(sealed, framed)


def _main_suite_full_binding(path: str, fill: str) -> dict[str, object]:
    return {
        "gid": 20,
        "mode": "0644",
        "path": path,
        "sha256": fill * 64,
        "size_bytes": 1,
        "st_dev": 1,
        "st_ino": ord(fill),
        "st_nlink": 1,
        "uid": 501,
    }


def _main_suite_semantic_fixture(tmp_path: Path) -> dict[str, Any]:
    policy_binding = _main_suite_full_binding(
        str(
            REPO_ROOT / "quant_investor/v17_v2_contract/resources/"
            "main_suite_runtime_policy.v1.json"
        ),
        "a",
    )
    manifest_binding = _main_suite_full_binding(
        str(REPO_ROOT / "quant_investor/v17_v2_contract/resources/package_manifest.v1.json"),
        "b",
    )
    schema_binding = _main_suite_full_binding(
        str(
            REPO_ROOT / "quant_investor/v17_v2_contract/schemas/"
            "main_suite_runtime_policy.v1.schema.json"
        ),
        "c",
    )
    wrapper_binding = _main_suite_full_binding(
        str(REPO_ROOT / subject.MAIN_SUITE_WRAPPER_PATH),
        "d",
    )
    harness_binding = _main_suite_full_binding(
        str(REPO_ROOT / subject.MAIN_SUITE_HARNESS_PATH),
        "e",
    )
    conftest_binding = _main_suite_full_binding(
        str(REPO_ROOT / "tests/conftest.py"),
        "f",
    )
    lexical_binding = _main_suite_full_binding(
        "/private/tmp/main-worktree-python",
        "1",
    )
    resolved_binding = _main_suite_full_binding(
        "/private/tmp/main-worktree-resolved-python",
        "2",
    )
    metadata_binding = _main_suite_full_binding(
        "/private/tmp/pytest.dist-info/METADATA",
        "3",
    )
    record_binding = _main_suite_full_binding(
        "/private/tmp/pytest.dist-info/RECORD",
        "4",
    )

    runtime_root = tmp_path / "runtime"
    home = runtime_root / "home"
    tmpdir = runtime_root / "tmp"
    xdg_cache = runtime_root / "xdg"
    black_cache = xdg_cache / "black"
    mypy_cache = xdg_cache / "mypy"
    pycache = xdg_cache / "pycache"
    for path in (
        runtime_root,
        home,
        tmpdir,
        xdg_cache,
        black_cache,
        mypy_cache,
        pycache,
    ):
        path.mkdir(mode=0o700)
        path.chmod(0o700)

    environment = {
        "BLACK_CACHE_DIR": str(black_cache),
        "HOME": str(home),
        "MYPY_CACHE_DIR": str(mypy_cache),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPYCACHEPREFIX": str(pycache),
        "TMPDIR": str(tmpdir),
        "XDG_CACHE_HOME": str(xdg_cache),
    }
    dynamic_path_keys = [
        "BLACK_CACHE_DIR",
        "HOME",
        "MYPY_CACHE_DIR",
        "PYTHONPYCACHEPREFIX",
        "TMPDIR",
        "XDG_CACHE_HOME",
    ]
    closures = {
        "final": {
            "classification_counts": {"candidate": 1},
            "count": 1,
            "rows_sha256": "1" * 64,
        },
        "pre_collection": {
            "classification_counts": {"candidate": 1},
            "count": 1,
            "rows_sha256": "2" * 64,
        },
        "pre_import": {
            "classification_counts": {"candidate": 1},
            "count": 1,
            "rows_sha256": "3" * 64,
        },
    }
    routing_policy = {
        "quant_investor_origin": str(REPO_ROOT / "quant_investor/__init__.py"),
        "removed_authority_entries": [],
        "sanitized_sys_path": [str(REPO_ROOT)],
    }
    main_runtime = {
        "invalid_dist_info": [],
        "lexical_python": lexical_binding["path"],
        "lexical_python_binding": lexical_binding,
        "post_site_state": {"sys_path": ["/unrouted"]},
        "resolved_python_binding": resolved_binding,
        "startup_files": [],
        "startup_modules": [],
        "valid_inventory": {"rows": []},
    }
    ownership_rows = [
        {
            "metadata_binding": metadata_binding,
            "name": "pytest",
            "record_binding": record_binding,
            "version": "8.4.2",
        }
    ]
    policy = {
        "candidate_conftest": conftest_binding,
        "candidate_root": str(REPO_ROOT),
        "claims": {
            "kernel_egress_attested": False,
            "network_unreachability_proven": False,
            "offline_policy_enforced": True,
        },
        "factor_authority_sources": [],
        "harness_binding": harness_binding,
        "limitations": list(subject.NORMATIVE_LIMITATIONS),
        "main_runtime": main_runtime,
        "module_closures": closures,
        "module_policy": {
            "authority_root": str(REPO_ROOT),
            "candidate_content_binding": "OUTER_SOURCE_STATE",
            "candidate_module_source_paths": ["quant_investor/__init__.py"],
            "distribution_ownership": ownership_rows,
        },
        "protected_roots": [],
        "pytest_args": [
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
        ],
        "pytest_environment": {
            "allowed_keys": sorted(environment),
            "dynamic_path_keys": dynamic_path_keys,
            "forbidden": [],
            "path_topology": {
                "cache_children": [
                    "BLACK_CACHE_DIR",
                    "MYPY_CACHE_DIR",
                    "PYTHONPYCACHEPREFIX",
                ],
                "closed_root_siblings": [
                    "HOME",
                    "TMPDIR",
                    "XDG_CACHE_HOME",
                ],
                "must_remain_empty": ["PYTHONPYCACHEPREFIX"],
            },
            "required": {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        },
        "pytest_plugins": [],
        "pytest_support_trees": [],
        "routing": routing_policy,
        "wrapper_binding": wrapper_binding,
    }
    session_binding = {
        "path": "/private/tmp/phase0/00_session.json",
        "semantic_sha256": "4" * 64,
        "session_id": "session-1",
        "sha256": "5" * 64,
        "size_bytes": 1,
    }

    module_path = REPO_ROOT / "quant_investor/__init__.py"
    project_modules = [
        {
            "name": "quant_investor",
            "path": str(module_path),
            "sha256": _sha(module_path.read_bytes()),
        }
    ]
    startup_projection: list[dict[str, Any]] = []
    routed_state = dict(main_runtime["post_site_state"])
    routed_state["sys_path"] = routing_policy["sanitized_sys_path"]
    runtime_routing = {
        "candidate_root": str(REPO_ROOT),
        "quant_investor_origin": routing_policy["quant_investor_origin"],
        "removed_authority_entries": [],
        "runtime_state": routed_state,
        "startup": {
            "lexical_python": lexical_binding,
            "resolved_python": resolved_binding,
            "startup_files": startup_projection,
            "wrapper": wrapper_binding,
        },
        "startup_modules": [],
    }

    def runtime(
        closure: Mapping[str, Any],
        modules: list[dict[str, str]],
    ) -> dict[str, Any]:
        return {
            "bytecode_policy": {
                "dont_write_bytecode": True,
                "pycache_prefix": str(pycache),
            },
            "factor_authority_sha256": _sha(subject._canonical_bytes([])),
            "interpreter": resolved_binding,
            "invalid_dist_info_sha256": _sha(subject._canonical_bytes([])),
            "inventory": main_runtime["valid_inventory"],
            "loaded_modules": closure,
            "policy_sha256": policy_binding["sha256"],
            "project_modules": modules,
            "routing": runtime_routing,
        }

    pre_import_runtime = runtime(closures["pre_import"], [])
    pre_collection_runtime = runtime(
        closures["pre_collection"],
        project_modules,
    )
    frames = [
        {
            "payload": {
                "challenge_binding_sha256": session_binding["sha256"],
                "environment": environment,
                "frame": "pre_import",
                "pid": 123,
                "ppid": 45,
                "runtime": pre_import_runtime,
            },
            "payload_sha256": "6" * 64,
            "payload_size_bytes": 2,
            "phase": 1,
        },
        {
            "payload": {
                "challenge_binding_sha256": session_binding["sha256"],
                "frame": "pre_collection",
                "pid": 123,
                "ppid": 45,
                "candidate_conftest": conftest_binding,
                "initial_conftest_loaded": True,
                "plugins": [],
                "project_modules": project_modules,
                "pytest_version": "8.4.2",
                "runtime": pre_collection_runtime,
                "support_trees": [],
            },
            "payload_sha256": "7" * 64,
            "payload_size_bytes": 2,
            "phase": 2,
        },
        {
            "payload": {
                "challenge_binding_sha256": session_binding["sha256"],
                "final_loaded_modules": closures["final"],
                "frame": "terminal_complete",
                "pid": 123,
                "ppid": 45,
                "pytest_exit_code": 0,
            },
            "payload_sha256": "8" * 64,
            "payload_size_bytes": 2,
            "phase": 3,
        },
    ]
    validator_bindings = []
    for label, relative in (
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
        binding, _raw = subject._main_suite_live_file_binding(
            REPO_ROOT / relative,
            label=label,
        )
        validator_bindings.append({"label": label, **binding})
    pycache_stat = pycache.lstat()
    pycache_binding = {
        "gid": pycache_stat.st_gid,
        "mode": "0700",
        "path": str(pycache),
        "st_ctime_ns": pycache_stat.st_ctime_ns,
        "st_dev": pycache_stat.st_dev,
        "st_ino": pycache_stat.st_ino,
        "st_mtime_ns": pycache_stat.st_mtime_ns,
        "st_nlink": pycache_stat.st_nlink,
        "uid": pycache_stat.st_uid,
    }
    ownership_projection = [
        {
            "metadata_sha256": metadata_binding["sha256"],
            "name": "pytest",
            "record_sha256": record_binding["sha256"],
            "version": "8.4.2",
        }
    ]
    live_projection = {
        "distribution_ownership_sha256": _sha(subject._canonical_bytes(ownership_projection)),
        "factor_authority_sha256": _sha(subject._canonical_bytes([])),
        "invalid_dist_info_sha256": _sha(subject._canonical_bytes([])),
        "lexical_python": lexical_binding,
        "physical_trees": [],
        "protected_roots": [],
        "resolved_python": resolved_binding,
        "startup_files": startup_projection,
    }
    snapshot_without_sha = {
        "bindings": [
            {"label": "wrapper_binding", **wrapper_binding},
            {"label": "harness_binding", **harness_binding},
            {"label": "candidate_conftest", **conftest_binding},
            {"label": "package_manifest", **manifest_binding},
            {"label": "runtime_policy", **policy_binding},
            {"label": "runtime_policy_schema", **schema_binding},
            *validator_bindings,
        ],
        **live_projection,
        "pycache_prefix": pycache_binding,
    }
    external = {
        **snapshot_without_sha,
        "snapshot_sha256": _sha(subject._canonical_bytes(snapshot_without_sha)),
    }
    output = (
        b"SKIPPED [42] tests/unit/test_optional_platform.py:12: "
        b"optional platform dependency unavailable\n"
        b"1 passed, 42 skipped in 0.01s\n"
    )
    skip_entries = [
        {
            "count": 42,
            "line": 12,
            "path": "tests/unit/test_optional_platform.py",
            "reason": "optional platform dependency unavailable",
        }
    ]
    receipt = subject._seal(
        {
            "accepted": True,
            "attestations": frames,
            "authority": False,
            "challenge_binding": {
                "kind": "PHASE0_SESSION_FILE",
                "sha256": session_binding["sha256"],
            },
            "claims": {
                "exit_code": 0,
                "final_audit_completed": True,
                "final_audit_enforced": True,
                "kernel_egress_attested": False,
                "network_unreachability_proven": False,
                "offline_policy_enforced": True,
                "signal": None,
            },
            "command": {
                "argv": [
                    policy["main_runtime"]["lexical_python"],
                    "-I",
                    "-S",
                    "-B",
                    "-X",
                    f"pycache_prefix={pycache}",
                    wrapper_binding["path"],
                    policy_binding["path"],
                    policy_binding["sha256"],
                    "--",
                    *policy["pytest_args"],
                ],
                "cwd": str(REPO_ROOT),
                "environment": environment,
            },
            "external_after": external,
            "external_before": external,
            "failure_codes": [],
            "failures": [],
            "finalization": {
                "cleanup": {"attempted": True, "status": "PASSED"},
                "external_after": {
                    "attempted": True,
                    "equal": True,
                    "status": "PASSED",
                },
            },
            "framing": subject.MAIN_SUITE_FRAMING,
            "limitations": list(subject.NORMATIVE_LIMITATIONS),
            "outcome": "PASSED",
            "policy_binding": policy_binding,
            "policy_manifest_binding": manifest_binding,
            "policy_schema_binding": schema_binding,
            "protocol_version": subject.PROTOCOL_VERSION,
            "schema_id": subject.MAIN_SUITE_RECEIPT_SCHEMA_ID,
            "streams": {},
            "timing": {"phase1_elapsed_ms": 1, "phase2_elapsed_ms": 2},
            "version": subject.MAIN_SUITE_RECEIPT_VERSION,
        }
    )
    base_commit = (
        subject._git_bytes(
            ("git", "rev-parse", "HEAD"),
            repo_root=REPO_ROOT,
        )
        .decode("ascii")
        .strip()
    )
    return {
        "base_commit": base_commit,
        "frames": frames,
        "live_projection": live_projection,
        "package_source_full": {"rows": []},
        "policy": policy,
        "policy_bindings": {
            "policy_binding": policy_binding,
            "policy_manifest_binding": manifest_binding,
            "policy_schema_binding": schema_binding,
        },
        "receipt": receipt,
        "session_binding": session_binding,
        "skip_baseline": {"entries": skip_entries},
        "skip_baseline_raw": b"frozen\n",
        "source_state": {"untracked": []},
        "streams": {"attestation": b"", "stderr": b"", "stdout": output},
    }


def _reseal_semantic_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    unsealed = copy.deepcopy(receipt)
    unsealed.pop("semantic_sha256")
    return subject._seal(unsealed)


def test_v2_identities_are_distinct_and_old_ids_are_not_registered() -> None:
    pairs = [(artifact, schema_id) for artifact, _path, schema_id in subject.SCHEMA_REGISTRY]
    assert len(pairs) == 12
    assert len({item for pair in pairs for item in pair}) == 24
    assert all(artifact != schema_id for artifact, schema_id in pairs)
    assert all(
        not item.endswith(".v1")
        for item in (
            subject.EVIDENCE_INDEX_VERSION,
            subject.GATE_MANIFEST_VERSION,
            subject.HASH_FREEZE_VERSION,
            subject.CLASSIFICATION_VERSION,
            subject.COMMAND_RECEIPT_VERSION,
            subject.PACKAGE_EVIDENCE_VERSION,
        )
    )


def test_owned_v2_schemas_preflight_and_bind_artifact_versions() -> None:
    schema_module = subject._schema_validation_module(REPO_ROOT)
    expected = {
        "v17_phase0_evidence_index.v2.schema.json": (
            subject.EVIDENCE_INDEX_SCHEMA_ID,
            subject.EVIDENCE_INDEX_VERSION,
        ),
        "v17_phase0_gate_manifest.v2.schema.json": (
            subject.GATE_MANIFEST_SCHEMA_ID,
            subject.GATE_MANIFEST_VERSION,
        ),
        "v17_phase0_hash_freeze.v2.schema.json": (
            subject.HASH_FREEZE_SCHEMA_ID,
            subject.HASH_FREEZE_VERSION,
        ),
        "v17_phase0_pre_existing_classification.v2.schema.json": (
            subject.CLASSIFICATION_SCHEMA_ID,
            subject.CLASSIFICATION_VERSION,
        ),
    }
    for filename, (schema_id, artifact_version) in expected.items():
        schema = json.loads((REPO_ROOT / "scripts/schemas" / filename).read_text())
        schema_module.preflight_packaged_schema(schema)
        assert schema["$id"] == schema_id
        assert schema["properties"]["version"] == {"const": artifact_version}
        assert schema_id != artifact_version
        assert schema["additionalProperties"] is False


def test_complete_v2_schema_registry_and_real_command_schema_load() -> None:
    schemas, bindings, raw_by_id = subject._load_v2_schema_registry(REPO_ROOT)
    assert len(bindings) == 12
    assert len(raw_by_id) == 12
    assert [row["schema_id"] for row in bindings] == [
        schema_id for _artifact, _path, schema_id in subject.SCHEMA_REGISTRY
    ]
    receipt, raw = _command_receipt()
    parsed, streams, framed = subject._parse_framed_command_receipt_v2(
        raw,
        schemas=schemas,
        label="v2_evidence_tests",
    )
    assert parsed == receipt
    assert streams == [(b"12 passed in 0.01s\n", b"")]
    assert _sha(framed) == receipt["output_sha256"]


def test_classification_v2_requires_provenance_and_rejects_v1_shape() -> None:
    schema = json.loads(
        (
            REPO_ROOT / "scripts/schemas/v17_phase0_pre_existing_classification.v2.schema.json"
        ).read_text()
    )
    base_commit = "a" * 40
    valid = subject._seal(
        {
            "base_commit": base_commit,
            "entries": [
                {
                    "classification": subject.PRE_EXISTING_CLASSIFICATION,
                    "path": "README.md",
                }
            ],
            "protocol_version": subject.PROTOCOL_VERSION,
            "provenance": subject.CLASSIFICATION_PROVENANCE,
            "version": subject.CLASSIFICATION_VERSION,
        }
    )
    raw = subject._canonical_resource_bytes(valid)
    assert subject._parse_classification_manifest_v2(
        raw,
        base_commit=base_commit,
        schemas=_schemas_for(subject.CLASSIFICATION_VERSION, schema),
    ) == ["README.md"]

    old = copy.deepcopy(valid)
    old["version"] = "myquant.v17.v2.phase0-pre-existing-classification.v1"
    old.pop("provenance")
    old["semantic_sha256"] = subject._semantic_sha256(old)
    with pytest.raises(subject.Phase0EvidenceError):
        subject._parse_classification_manifest_v2(
            subject._canonical_resource_bytes(old),
            base_commit=base_commit,
            schemas=_schemas_for(subject.CLASSIFICATION_VERSION, schema),
        )


def test_classification_v2_rejects_wrong_provenance_and_noncanonical_order() -> None:
    schema = json.loads(
        (
            REPO_ROOT / "scripts/schemas/v17_phase0_pre_existing_classification.v2.schema.json"
        ).read_text()
    )
    base = {
        "base_commit": "a" * 40,
        "entries": [
            {
                "classification": subject.PRE_EXISTING_CLASSIFICATION,
                "path": "z.txt",
            },
            {
                "classification": subject.PRE_EXISTING_CLASSIFICATION,
                "path": "a.txt",
            },
        ],
        "protocol_version": subject.PROTOCOL_VERSION,
        "provenance": subject.CLASSIFICATION_PROVENANCE,
        "version": subject.CLASSIFICATION_VERSION,
    }
    with pytest.raises(subject.Phase0EvidenceError, match="canonically ordered"):
        subject._parse_classification_manifest_v2(
            subject._canonical_resource_bytes(subject._seal(base)),
            base_commit="a" * 40,
            schemas=_schemas_for(subject.CLASSIFICATION_VERSION, schema),
        )
    wrong = copy.deepcopy(base)
    wrong["entries"].reverse()
    wrong["provenance"] = "UNTRUSTED"
    with pytest.raises(subject.Phase0EvidenceError):
        subject._parse_classification_manifest_v2(
            subject._canonical_resource_bytes(subject._seal(wrong)),
            base_commit="a" * 40,
            schemas=_schemas_for(subject.CLASSIFICATION_VERSION, schema),
        )


def test_collection_classification_parser_enforces_v2_provenance() -> None:
    base_commit = "a" * 40
    valid = subject._seal(
        {
            "base_commit": base_commit,
            "entries": [
                {
                    "classification": subject.PRE_EXISTING_CLASSIFICATION,
                    "path": "README.md",
                }
            ],
            "protocol_version": subject.PROTOCOL_VERSION,
            "provenance": subject.CLASSIFICATION_PROVENANCE,
            "version": subject.CLASSIFICATION_VERSION,
        }
    )
    raw = subject._canonical_resource_bytes(valid)
    assert subject._parse_classification_manifest(
        raw,
        base_commit=base_commit,
    ) == ["README.md"]

    missing = copy.deepcopy(valid)
    missing.pop("provenance")
    missing["semantic_sha256"] = subject._semantic_sha256(missing)
    with pytest.raises(subject.Phase0EvidenceError, match="shape mismatch"):
        subject._parse_classification_manifest(
            subject._canonical_resource_bytes(missing),
            base_commit=base_commit,
        )

    wrong = copy.deepcopy(valid)
    wrong["provenance"] = "UNTRUSTED"
    wrong["semantic_sha256"] = subject._semantic_sha256(wrong)
    with pytest.raises(subject.Phase0EvidenceError, match="identity mismatch"):
        subject._parse_classification_manifest(
            subject._canonical_resource_bytes(wrong),
            base_commit=base_commit,
        )


def test_index_skip_validation_delegates_to_transcript_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RejectingSkipProducer:
        @staticmethod
        def validate_skip_baseline(
            value: object,
            *,
            repo_root: Path,
        ) -> dict[str, object]:
            assert value == {"commands": ["forged"]}
            assert repo_root == REPO_ROOT
            raise ValueError("transcript rows do not match")

    monkeypatch.setattr(
        subject,
        "_load_local_module",
        lambda *_args, **_kwargs: RejectingSkipProducer,
    )
    with pytest.raises(
        subject.Phase0EvidenceError,
        match="producer semantic validation failed",
    ):
        subject._validate_v2_skip_baseline(
            {"commands": ["forged"]},
            source_state={},
            source_binding={},
            toolchain={},
            package_source_superset={},
            protected_roots=[],
            repo_root=REPO_ROOT,
        )


def test_v2_command_identity_rejects_argv_environment_and_tool_tamper() -> None:
    toolchain = _toolchain()
    work_root = Path("/private/tmp/phase0-command-identity")
    environment = {
        **subject._v2_base_environment(toolchain),
        "TMPDIR": str(work_root / "tmp/native_sync"),
        "UV_PROJECT_ENVIRONMENT": str(work_root / "native_venv"),
    }
    command = {
        "argv": [
            toolchain["uv"]["lexical_path"],
            "sync",
            "--python",
            toolchain["base_python"]["lexical_path"],
            "--locked",
            "--all-extras",
            "--offline",
        ],
        "environment": environment,
        "tool_version": toolchain["uv"]["output"],
    }
    receipt = {"commands": [command]}
    assert (
        subject._validate_v2_command_identity(
            "native_sync_log",
            receipt,
            repo_root=REPO_ROOT,
            toolchain=toolchain,
            work_root=None,
            tool_versions=None,
        )
        == work_root
    )
    for field, replacement in (
        ("argv", [*command["argv"][:-1], "--online"]),
        ("environment", {**environment, "UV_OFFLINE": "0"}),
        ("tool_version", "uv 0.10.8"),
    ):
        changed = copy.deepcopy(receipt)
        changed["commands"][0][field] = replacement
        with pytest.raises(subject.Phase0EvidenceError, match="command identity"):
            subject._validate_v2_command_identity(
                "native_sync_log",
                changed,
                repo_root=REPO_ROOT,
                toolchain=toolchain,
                work_root=None,
                tool_versions=None,
            )


def test_pytest_skip_entries_use_producer_canonical_order_and_safe_paths() -> None:
    raw = (
        b"SKIPPED [15] tests/unit/example.py:85: later reason\n"
        b"SKIPPED [1] tests/unit/example.py:54: earlier reason\n"
    )
    assert [row["line"] for row in subject._pytest_skip_entries(raw, label="skip")] == [
        54,
        85,
    ]
    with pytest.raises(subject.Phase0EvidenceError, match="safe repository-relative"):
        subject._pytest_skip_entries(
            b"SKIPPED [1] ../outside.py:1: unsafe\n",
            label="skip",
        )


def test_command_v2_parser_accepts_exact_binary_frames() -> None:
    receipt, raw = _command_receipt()
    parsed, streams, framed = subject._parse_framed_command_receipt_v2(
        raw,
        schemas=_schemas_for(subject.COMMAND_RECEIPT_VERSION, _minimal_command_schema()),
        label="v2_evidence_tests",
    )
    assert parsed == receipt
    assert streams == [(b"12 passed in 0.01s\n", b"")]
    assert _sha(framed) == receipt["output_sha256"]


@pytest.mark.parametrize(
    "mutator,match",
    [
        (
            lambda receipt, framed: (
                {**receipt, "version": "myquant.v17.v2.phase0-command-receipt.v1"},
                framed,
            ),
            "identity",
        ),
        (
            lambda receipt, framed: (
                receipt,
                struct.pack(">Q", 999) + framed[8:],
            ),
            "frame",
        ),
        (
            lambda receipt, framed: (
                {
                    **receipt,
                    "commands": [
                        {
                            **receipt["commands"][0],
                            "stdout_offset_bytes": 9,
                        }
                    ],
                },
                framed,
            ),
            "frame",
        ),
    ],
)
def test_command_v2_parser_rejects_downgrade_length_and_offset_tamper(
    mutator: Callable[
        [dict[str, object], bytes],
        tuple[dict[str, object], bytes],
    ],
    match: str,
) -> None:
    receipt, raw = _command_receipt()
    prefix_line, framed = raw.split(b"\n", 1)
    assert prefix_line.startswith(subject.COMMAND_RECEIPT_PREFIX)
    changed, changed_framed = mutator(copy.deepcopy(receipt), framed)
    changed["output_sha256"] = _sha(changed_framed)
    changed["output_size_bytes"] = len(changed_framed)
    changed["semantic_sha256"] = subject._semantic_sha256(changed)
    changed_raw = (
        subject.COMMAND_RECEIPT_PREFIX + subject._canonical_bytes(changed) + b"\n" + changed_framed
    )
    with pytest.raises(subject.Phase0EvidenceError, match=match):
        subject._parse_framed_command_receipt_v2(
            changed_raw,
            schemas=_schemas_for(
                subject.COMMAND_RECEIPT_VERSION,
                _minimal_command_schema(),
            ),
            label="v2_evidence_tests",
        )


def test_main_suite_parser_accepts_exact_three_stream_binary_frames() -> None:
    receipt, raw = _main_suite_receipt()
    parsed, streams, framed = subject._parse_framed_main_suite_receipt_v1(
        raw,
        label="full_offline_suite",
    )
    assert parsed == receipt
    assert streams == {
        "attestation": _main_suite_attestation(),
        "stderr": b"",
        "stdout": b"1 passed in 0.01s\n",
    }
    assert _sha(framed) == receipt["streams"]["tail_sha256"]


def test_main_suite_parser_rejects_duplicate_and_noncanonical_json() -> None:
    receipt, raw = _main_suite_receipt()
    _line, framed = raw.split(b"\n", 1)
    duplicate = (
        subject.MAIN_SUITE_RECEIPT_PREFIX
        + (
            b'{"version":"myquant.v17.v2.phase0-main-suite-receipt.v1",'
            b'"version":"myquant.v17.v2.phase0-main-suite-receipt.v1"}'
        )
        + b"\n"
        + framed
    )
    with pytest.raises(subject.Phase0EvidenceError, match="duplicate"):
        subject._parse_framed_main_suite_receipt_v1(
            duplicate,
            label="full_offline_suite",
        )

    spaced = json.dumps(receipt, allow_nan=False, sort_keys=True).encode("utf-8")
    noncanonical = subject.MAIN_SUITE_RECEIPT_PREFIX + spaced + b"\n" + framed
    with pytest.raises(subject.Phase0EvidenceError, match="canonical"):
        subject._parse_framed_main_suite_receipt_v1(
            noncanonical,
            label="full_offline_suite",
        )


@pytest.mark.parametrize(
    "dispatch_site",
    [
        "external_collection",
        "external_readback",
        "sealed_bundle",
    ],
)
def test_full_suite_dispatch_sites_reject_generic_command_envelope(
    dispatch_site: str,
) -> None:
    _receipt, generic_raw = _command_receipt()
    assert dispatch_site
    assert "full_offline_suite" not in subject.LOG_ROLES
    with pytest.raises(subject.Phase0EvidenceError, match="main-suite receipt envelope"):
        subject._parse_framed_main_suite_receipt_v1(
            generic_raw,
            label=f"{dispatch_site}:full_offline_suite",
        )


def test_main_suite_parser_rejects_truncation_and_extra_tail_bytes() -> None:
    receipt, raw = _main_suite_receipt()
    _line, framed = raw.split(b"\n", 1)

    _truncated_receipt, truncated_raw = _reseal_main_suite(
        receipt,
        framed[:-1],
        bind_tail=True,
    )
    with pytest.raises(subject.Phase0EvidenceError, match="truncated"):
        subject._parse_framed_main_suite_receipt_v1(
            truncated_raw,
            label="full_offline_suite",
        )

    _extra_receipt, extra_raw = _reseal_main_suite(
        receipt,
        framed + b"x",
        bind_tail=True,
    )
    with pytest.raises(subject.Phase0EvidenceError, match="trailing"):
        subject._parse_framed_main_suite_receipt_v1(
            extra_raw,
            label="full_offline_suite",
        )


@pytest.mark.parametrize("stream_name", ["stdout", "attestation"])
def test_main_suite_parser_rejects_stream_size_caps(stream_name: str) -> None:
    receipt, _raw = _main_suite_receipt(stdout=b"", stderr=b"", attestation=b"")
    changed = copy.deepcopy(receipt)
    streams = changed["streams"]
    assert isinstance(streams, dict)
    binding = streams[stream_name]
    assert isinstance(binding, dict)
    maximum = (
        subject.MAX_COMMAND_STREAM_BYTES
        if stream_name == "stdout"
        else subject.MAX_MAIN_SUITE_ATTESTATION_BYTES
    )
    binding["size_bytes"] = maximum + 1
    framed = (
        struct.pack(">Q", maximum + 1)
        if stream_name == "stdout"
        else struct.pack(">Q", 0) + struct.pack(">Q", 0) + struct.pack(">Q", maximum + 1)
    )
    _sealed, changed_raw = _reseal_main_suite(changed, framed, bind_tail=True)
    with pytest.raises(subject.Phase0EvidenceError, match="limit exceeded"):
        subject._parse_framed_main_suite_receipt_v1(
            changed_raw,
            label="full_offline_suite",
        )


def test_main_suite_parser_accepts_exact_three_frame_attestation_cap() -> None:
    attestation = _main_suite_attestation(
        (
            b"x" * subject.MAX_MAIN_SUITE_FRAME_BYTES,
            b"y" * subject.MAX_MAIN_SUITE_FRAME_BYTES,
            b"z" * subject.MAX_MAIN_SUITE_TERMINAL_FRAME_BYTES,
        )
    )
    assert len(attestation) == subject.MAX_MAIN_SUITE_ATTESTATION_BYTES
    _receipt, raw = _main_suite_receipt(
        stdout=b"",
        stderr=b"",
        attestation=attestation,
    )
    _parsed, streams, _framed = subject._parse_framed_main_suite_receipt_v1(
        raw,
        label="full_offline_suite",
    )
    assert streams["attestation"] == attestation


def test_main_suite_parser_rejects_missing_or_out_of_order_attestation_phase() -> None:
    phase1 = _main_suite_attestation_frame(1, b"{}")
    phase2 = _main_suite_attestation_frame(2, b"{}")
    _receipt, missing_raw = _main_suite_receipt(attestation=phase1 + phase2)
    with pytest.raises(subject.Phase0EvidenceError, match="phase 3 header is truncated"):
        subject._parse_framed_main_suite_receipt_v1(
            missing_raw,
            label="full_offline_suite",
        )

    out_of_order = (
        _main_suite_attestation_frame(2, b"{}") + phase2 + _main_suite_attestation_frame(3, b"{}")
    )
    _receipt, out_of_order_raw = _main_suite_receipt(attestation=out_of_order)
    with pytest.raises(subject.Phase0EvidenceError, match="phase 1 header mismatch"):
        subject._parse_framed_main_suite_receipt_v1(
            out_of_order_raw,
            label="full_offline_suite",
        )


def test_main_suite_parser_rejects_terminal_cap_nonce_and_digest_tamper() -> None:
    oversized_terminal = _main_suite_attestation(
        (
            b"{}",
            b"{}",
            b"x" * (subject.MAX_MAIN_SUITE_TERMINAL_FRAME_BYTES + 1),
        )
    )
    _receipt, oversized_raw = _main_suite_receipt(attestation=oversized_terminal)
    with pytest.raises(subject.Phase0EvidenceError, match="phase 3 header mismatch"):
        subject._parse_framed_main_suite_receipt_v1(
            oversized_raw,
            label="full_offline_suite",
        )

    nonce_tamper = (
        _main_suite_attestation_frame(1, b"{}")
        + _main_suite_attestation_frame(2, b"{}", nonce=b"m" * 32)
        + _main_suite_attestation_frame(3, b"{}")
    )
    _receipt, nonce_raw = _main_suite_receipt(attestation=nonce_tamper)
    with pytest.raises(subject.Phase0EvidenceError, match="nonce mismatch"):
        subject._parse_framed_main_suite_receipt_v1(
            nonce_raw,
            label="full_offline_suite",
        )

    digest_tamper = bytearray(_main_suite_attestation())
    digest_tamper[subject.MAIN_SUITE_ATTEST_HEADER.size] ^= 1
    _receipt, digest_raw = _main_suite_receipt(attestation=bytes(digest_tamper))
    with pytest.raises(subject.Phase0EvidenceError, match="digest mismatch"):
        subject._parse_framed_main_suite_receipt_v1(
            digest_raw,
            label="full_offline_suite",
        )


@pytest.mark.parametrize(
    "mutator,match",
    [
        (
            lambda streams: streams["stdout"].__setitem__("offset_bytes", 9),
            "stdout frame binding",
        ),
        (
            lambda streams: streams["stdout"].__setitem__(
                "sha256",
                "0" * 64,
            ),
            "stdout frame binding",
        ),
        (
            lambda streams: streams["stdout"].__setitem__("size_bytes", 19),
            "stdout frame binding",
        ),
        (
            lambda streams: streams.__setitem__("tail_sha256", "0" * 64),
            "tail binding",
        ),
        (
            lambda streams: streams.__setitem__(
                "tail_size_bytes",
                streams["tail_size_bytes"] + 1,
            ),
            "tail binding",
        ),
    ],
)
def test_main_suite_parser_rejects_receipt_stream_binding_mismatch(
    mutator: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    receipt, raw = _main_suite_receipt()
    _line, framed = raw.split(b"\n", 1)
    changed = copy.deepcopy(receipt)
    streams = changed["streams"]
    assert isinstance(streams, dict)
    mutator(streams)
    _sealed, changed_raw = _reseal_main_suite(changed, framed, bind_tail=False)
    with pytest.raises(subject.Phase0EvidenceError, match=match):
        subject._parse_framed_main_suite_receipt_v1(
            changed_raw,
            label="full_offline_suite",
        )


def test_full_suite_claims_require_exact_frozen_42_skip_rows() -> None:
    output = (
        b"SKIPPED [42] tests/unit/test_optional_platform.py:12: "
        b"optional platform dependency unavailable\n"
        b"1 passed, 42 skipped in 0.01s\n"
    )
    entries = [
        {
            "count": 42,
            "line": 12,
            "path": "tests/unit/test_optional_platform.py",
            "reason": "optional platform dependency unavailable",
        }
    ]
    baseline_raw = b"frozen-baseline\n"
    baseline = {
        "entries": entries,
        "semantic_sha256": "7" * 64,
    }
    claims = {
        "errors": 0,
        "exit_code": 0,
        "failed": 0,
        "passed": 1,
        "raw_output_sha256": _sha(output),
        "skip_allowlist": entries,
        "skipped": 42,
        "xfail": 0,
        "xpass": 0,
    }
    assert (
        subject._validate_v2_gate_claims(
            "full_offline_suite",
            claims,
            output=output,
            skip_baseline=baseline,
            skip_baseline_raw=baseline_raw,
        )
        == claims
    )

    changed = output.replace(b"optional platform dependency unavailable", b"changed reason")
    with pytest.raises(subject.Phase0EvidenceError, match="differ"):
        subject._validate_v2_gate_claims(
            "full_offline_suite",
            {**claims, "raw_output_sha256": _sha(changed)},
            output=changed,
            skip_baseline=baseline,
            skip_baseline_raw=baseline_raw,
        )


def _validate_main_suite_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    return subject._validate_v2_main_suite_semantics(
        fixture["receipt"],
        streams=fixture["streams"],
        frames=fixture["frames"],
        repo_root=REPO_ROOT,
        policy=fixture["policy"],
        policy_bindings=fixture["policy_bindings"],
        session_binding=fixture["session_binding"],
        skip_baseline=fixture["skip_baseline"],
        skip_baseline_raw=fixture["skip_baseline_raw"],
        base_commit=fixture["base_commit"],
        source_state=fixture["source_state"],
        package_source_full=fixture["package_source_full"],
    )


def test_main_suite_gate_semantics_accepts_exact_dedicated_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _main_suite_semantic_fixture(tmp_path)
    monkeypatch.setattr(
        subject,
        "_live_v2_main_suite_policy_projection",
        lambda *_args, **_kwargs: copy.deepcopy(fixture["live_projection"]),
    )
    claims = _validate_main_suite_fixture(fixture)
    assert claims["exit_code"] == 0
    assert claims["skipped"] == 42
    assert claims["skip_allowlist"] == fixture["skip_baseline"]["entries"]


@pytest.mark.parametrize(
    "tamper,match",
    [
        ("challenge_kind", "session challenge"),
        ("challenge_sha", "session challenge"),
        ("policy_binding", "policy binding"),
        ("manifest_binding", "policy binding"),
        ("schema_binding", "policy binding"),
        ("rejected", "not accepted"),
        ("binary_attestation", "binary attestation"),
        ("terminal_closure", "attested closure"),
        ("candidate_membership", "outer-source-bound"),
    ],
)
def test_main_suite_gate_semantics_rejects_tamper_before_pytest_claims(
    tamper: str,
    match: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _main_suite_semantic_fixture(tmp_path)
    monkeypatch.setattr(
        subject,
        "_live_v2_main_suite_policy_projection",
        lambda *_args, **_kwargs: copy.deepcopy(fixture["live_projection"]),
    )
    receipt = copy.deepcopy(fixture["receipt"])
    frames = copy.deepcopy(fixture["frames"])
    policy = copy.deepcopy(fixture["policy"])
    if tamper == "challenge_kind":
        receipt["challenge_binding"]["kind"] = "SKIP_SOURCE_STATE"
    elif tamper == "challenge_sha":
        receipt["challenge_binding"]["sha256"] = "0" * 64
    elif tamper == "policy_binding":
        receipt["policy_binding"]["sha256"] = "0" * 64
    elif tamper == "manifest_binding":
        receipt["policy_manifest_binding"]["sha256"] = "0" * 64
    elif tamper == "schema_binding":
        receipt["policy_schema_binding"]["sha256"] = "0" * 64
    elif tamper == "rejected":
        receipt["accepted"] = False
        receipt["outcome"] = "FAILED"
        receipt["failure_codes"] = ["PYTEST_NONZERO"]
        receipt["failures"] = [
            {
                "code": "PYTEST_NONZERO",
                "detail": "child exit code 1",
                "phase": "PRIMARY",
            }
        ]
    elif tamper == "binary_attestation":
        receipt["attestations"][0]["payload_sha256"] = "0" * 64
    elif tamper == "terminal_closure":
        frames[2]["payload"]["final_loaded_modules"] = {
            "classification_counts": {},
            "count": 0,
            "rows_sha256": "0" * 64,
        }
        receipt["attestations"] = copy.deepcopy(frames)
    elif tamper == "candidate_membership":
        policy["module_policy"]["candidate_module_source_paths"] = [
            "quant_investor/not_sealed_missing.py"
        ]
    else:  # pragma: no cover - parametrization guard
        raise AssertionError(tamper)
    fixture["receipt"] = _reseal_semantic_receipt(receipt)
    fixture["frames"] = frames
    fixture["policy"] = policy

    def _claims_must_not_run(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise AssertionError("pytest claims validation ran before gate semantics")

    monkeypatch.setattr(subject, "_validate_v2_gate_claims", _claims_must_not_run)
    with pytest.raises(subject.Phase0EvidenceError, match=match):
        _validate_main_suite_fixture(fixture)


def test_gate_roles_use_log_first_order_fixed_filenames_and_ordinals() -> None:
    assert subject.GATE_ROLES == (
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
    assert subject.GATE_ORDINALS == tuple(range(1, 11))
    assert subject.GATE_FILENAMES == (
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
    assert subject.GATE_ARTIFACT_VERSIONS[4] == subject.MAIN_SUITE_RECEIPT_VERSION
    assert subject.GATE_SCHEMA_IDS[4] == subject.MAIN_SUITE_RECEIPT_SCHEMA_ID
    assert subject.GATE_PRODUCER_SPECS[4] == (
        subject.MAIN_SUITE_HARNESS_PATH,
        subject.MAIN_SUITE_RECEIPT_VERSION,
    )
    assert subject._expected_v2_step(8) == {
        "filename": "40_package_parity.json",
        "kind": "artifact",
        "ordinal": 9,
        "role": "package_parity",
    }


def test_hash_freeze_and_allowlist_cover_all_v2_authority_files() -> None:
    expected_schemas = {path for _artifact, path, _schema in subject.SCHEMA_REGISTRY}
    assert expected_schemas <= set(subject.HASH_FREEZE_PATHS)
    assert expected_schemas <= set(subject.PHASE0_ALLOWED_PATTERN_REGISTRY)
    expected_producers = {
        "scripts/v17_offline_dependency_evidence.py",
        "scripts/v17_phase0_diff_check.py",
        "scripts/v17_phase0_evidence_index.py",
        "scripts/v17_phase0_evidence_session.py",
        "scripts/v17_phase0_package_evidence.py",
        "scripts/v17_phase0_skip_baseline.py",
    }
    assert expected_producers <= set(subject.HASH_FREEZE_PATHS)
    assert expected_producers <= set(subject.PHASE0_ALLOWED_PATTERN_REGISTRY)
    assert not any(
        ".v1.schema.json" in path
        for path in expected_schemas
        if "session" not in path
        and "skip" not in path
        and "failure" not in path
        and "main_suite" not in path
    )


def test_main_suite_frozen_inventory_is_registered_without_runtime_assumptions() -> None:
    frozen_paths = {
        "scripts/v17_phase0_main_suite_wrapper.py",
        "scripts/v17_phase0_main_suite_harness.py",
        "quant_investor/v17_v2_contract/resources/main_suite_runtime_policy.v1.json",
        "quant_investor/v17_v2_contract/schemas/main_suite_runtime_policy.v1.schema.json",
        "scripts/schemas/v17_phase0_main_suite_receipt.v1.schema.json",
    }
    assert frozen_paths <= set(subject.PHASE0_ALLOWED_PATTERN_REGISTRY)
    assert frozen_paths <= set(subject.HASH_FREEZE_PATHS)
    assert {
        "scripts/v17_phase0_main_suite_wrapper.py",
        "scripts/v17_phase0_main_suite_harness.py",
    } <= set(subject.BLACK_TARGETS)
    assert subject.MYPY_TARGETS == (
        "quant_investor/v17_v2_contract",
        "scripts/v17_phase0_evidence_index.py",
        "scripts/v17_phase0_main_suite_harness.py",
        "scripts/v17_phase0_main_suite_wrapper.py",
    )
    assert (
        subject.MAIN_SUITE_RUNTIME_POLICY_VERSION,
        "quant_investor/v17_v2_contract/schemas/main_suite_runtime_policy.v1.schema.json",
        subject.MAIN_SUITE_RUNTIME_POLICY_SCHEMA_ID,
    ) in subject.SCHEMA_REGISTRY
    assert (
        subject.MAIN_SUITE_RECEIPT_VERSION,
        "scripts/schemas/v17_phase0_main_suite_receipt.v1.schema.json",
        subject.MAIN_SUITE_RECEIPT_SCHEMA_ID,
    ) in subject.SCHEMA_REGISTRY
    assert subject.MAIN_SUITE_RECEIPT_PREFIX == b"MYQUANT_PHASE0_MAIN_SUITE_RECEIPT="


def test_session_producer_and_index_fixed_gate_inventories_match() -> None:
    path = REPO_ROOT / "scripts/v17_phase0_evidence_session.py"
    spec = importlib.util.spec_from_file_location("v17_phase0_session_inventory", path)
    assert spec is not None and spec.loader is not None
    session = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(session)
    for name in (
        "V2_EVIDENCE_TESTS",
        "RECOMMENDED_CORE_TESTS",
        "MYPY_TARGETS",
        "BLACK_TARGETS",
    ):
        assert getattr(session, name) == getattr(subject, name)


def test_legacy_builder_publisher_and_index_shape_fail_closed() -> None:
    with pytest.raises(subject.Phase0EvidenceError, match="v1 builder"):
        subject.build_evidence_index(base_commit="a" * 40)
    with pytest.raises(subject.Phase0EvidenceError, match="v1 publisher"):
        subject.write_evidence_index_exact_once(output_json=Path("/tmp/old.json"))
    with pytest.raises(subject.Phase0EvidenceError, match="v1/downgrade"):
        subject.validate_evidence_index(
            {
                "repo_root": str(REPO_ROOT),
                "version": "myquant.v17.v2.phase0-evidence-index.v1",
            },
            verify_external=False,
        )


def test_production_cli_is_closed_and_legacy_flags_are_rejected() -> None:
    parsed = subject.parse_args(
        [
            "--repo-root",
            str(REPO_ROOT),
            "--bundle-root",
            "/private/tmp/phase0-bundle",
            "--classification-manifest",
            "/private/tmp/classification.json",
            "--skip-baseline",
            "/private/tmp/skip.json",
            "--session-manifest",
            "/private/tmp/phase0-bundle/00_session.json",
            "--expected-session-sha256",
            "a" * 64,
        ]
    )
    assert parsed.expected_session_sha256 == "a" * 64
    with pytest.raises(SystemExit):
        subject.parse_args(
            [
                "--repo-root",
                str(REPO_ROOT),
                "--base-commit",
                "b" * 40,
                "--gate-manifest",
                "/private/tmp/gate.json",
                "--output-json",
                "/private/tmp/index.json",
            ]
        )


def test_hash_schema_closes_all_authority_hashes() -> None:
    schema = json.loads(
        (REPO_ROOT / "scripts/schemas/v17_phase0_hash_freeze.v2.schema.json").read_text()
    )
    hashes = schema["properties"]["hashes"]
    assert hashes["additionalProperties"] is False
    assert set(hashes["properties"]) == set(subject.HASH_FREEZE_PATHS)
    assert set(hashes["required"]) == set(subject.HASH_FREEZE_PATHS)


def test_index_schema_closes_exact_phase0_path_pattern_registry() -> None:
    schema = json.loads(
        (REPO_ROOT / "scripts/schemas/v17_phase0_evidence_index.v2.schema.json").read_text()
    )
    enum = schema["properties"]["allowlist"]["properties"]["allowed_phase0_path_patterns"]["items"][
        "enum"
    ]
    assert len(enum) == len(set(enum))
    assert set(enum) == set(subject.PHASE0_ALLOWED_PATTERN_REGISTRY)
    schema_rows = schema["properties"]["schemas"]
    assert schema_rows["minItems"] == len(subject.SCHEMA_REGISTRY)
    assert schema_rows["maxItems"] == len(subject.SCHEMA_REGISTRY)


def test_v2_session_keys_equal_closed_session_schema_required_keys() -> None:
    schema = json.loads(
        (REPO_ROOT / "scripts/schemas/v17_phase0_session.v2.schema.json").read_text()
    )
    assert subject.V2_SESSION_KEYS == set(schema["required"])


def test_v2_exact_once_publishers_require_two_links_before_staged_unlink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    monkeypatch.setattr(
        subject,
        "_validate_v2_schema",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        subject,
        "_validate_v2_seal",
        lambda *_args, **_kwargs: None,
    )

    resource_path = tmp_path / "50_hash_freeze.json"
    resource = {"accepted": True}
    binding, raw = subject._publish_v2_resource_exact_once(
        resource_path,
        resource,
        artifact_version=subject.HASH_FREEZE_VERSION,
        schemas={},
    )
    assert resource_path.read_bytes() == raw
    assert resource_path.stat().st_nlink == 1
    assert binding["sha256"] == _sha(raw)
    assert not any(path.name.startswith(".50_hash_freeze.json.") for path in tmp_path.iterdir())

    index_path = tmp_path / "70_evidence_index.json"
    sidecar_path = tmp_path / "70_evidence_index.json.sha256"
    index_binding, index_raw = subject._publish_v2_index_pair_exact_once(
        index_path=index_path,
        sidecar_path=sidecar_path,
        report={"accepted": True},
        schemas={},
    )
    assert index_path.read_bytes() == index_raw
    assert index_path.stat().st_nlink == 1
    assert sidecar_path.stat().st_nlink == 1
    assert sidecar_path.read_bytes() == (f"{_sha(index_raw)}  {index_path.name}\n".encode("ascii"))
    assert index_binding["sha256"] == _sha(index_raw)
    assert not any(path.name.startswith(".70_evidence_index") for path in tmp_path.iterdir())


def test_protected_root_validator_rejects_bool_as_integer() -> None:
    roots = _protected_roots_absent()
    roots[0] = {
        "ctime_ns": 1,
        "id": subject.PROTECTED_ROOT_SPECS[0][0],
        "mode": "0755",
        "mtime_ns": 1,
        "path": str(subject.PROTECTED_ROOT_SPECS[0][1]),
        "realpath": str(subject.PROTECTED_ROOT_SPECS[0][1]),
        "st_dev": True,
        "st_ino": 1,
        "state": "PRESENT_DIRECTORY",
        "uid": 501,
    }
    with pytest.raises(subject.Phase0EvidenceError, match="integer"):
        subject._validate_v2_protected_roots(roots, label="roots")
