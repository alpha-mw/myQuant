from __future__ import annotations

import hashlib
import copy
import json
import os
from pathlib import Path
import stat
import struct
import sys

import pytest

from quant_investor.v17_v2_contract.schema_validation import (
    SchemaValidationError,
    preflight_packaged_schema,
    validate_instance_against_schema,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import v17_phase0_evidence_session as session  # noqa: E402


def _private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path


def _capture(
    *,
    stdout: bytes,
    stderr: bytes,
    argv: list[str] | None = None,
    exit_code: int = 0,
) -> dict[str, object]:
    return {
        "argv": argv or ["/bin/echo", "fixed"],
        "cwd": "/private/tmp/repo",
        "environment": {"PYTHONDONTWRITEBYTECODE": "1"},
        "exit_code": exit_code,
        "signal": None,
        "stderr": stderr,
        "stdout": stdout,
        "tool_version": "fixed 1.0",
    }


def _main_suite_binding(path: Path, *, fill: str) -> dict[str, object]:
    return {
        "gid": os.getgid(),
        "mode": "0644",
        "path": str(path),
        "sha256": fill * 64,
        "size_bytes": 123,
        "st_dev": 1,
        "st_ino": 2,
        "st_nlink": 1,
        "uid": os.getuid(),
    }


def _policy_bindings(repo_root: Path) -> dict[str, dict[str, object]]:
    return {
        "policy_binding": _main_suite_binding(
            repo_root / session.MAIN_SUITE_POLICY_PATH,
            fill="5",
        ),
        "policy_manifest_binding": _main_suite_binding(
            repo_root / session.MAIN_SUITE_PACKAGE_MANIFEST_PATH,
            fill="6",
        ),
        "policy_schema_binding": _main_suite_binding(
            repo_root / session.MAIN_SUITE_POLICY_SCHEMA_PATH,
            fill="7",
        ),
    }


def _main_suite_contract_inputs(
    repo_root: Path,
) -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
    cache = repo_root / "runtime" / "cache"
    environment = {
        "BLACK_CACHE_DIR": str(cache / "black"),
        "HOME": str(repo_root / "runtime" / "home"),
        "MYPY_CACHE_DIR": str(cache / "mypy"),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPYCACHEPREFIX": str(cache / "pycache"),
        "TMPDIR": str(repo_root / "runtime" / "tmp"),
        "XDG_CACHE_HOME": str(cache),
    }
    policy = {
        "main_runtime": {"lexical_python": "/bound/python"},
        "wrapper_binding": {"path": "/bound/wrapper"},
    }
    pycache_binding = {
        "gid": os.getgid(),
        "mode": "0700",
        "path": environment["PYTHONPYCACHEPREFIX"],
        "st_ctime_ns": 11,
        "st_dev": 1,
        "st_ino": 4,
        "st_mtime_ns": 12,
        "st_nlink": 2,
        "uid": os.getuid(),
    }
    return policy, dict(sorted(environment.items())), pycache_binding


def _main_suite_result(
    *,
    challenge_kind: str,
    challenge_sha256: str,
    repo_root: Path,
    stdout: bytes,
    stderr: bytes = b"",
) -> dict[str, object]:
    attestation = b"three-bound-attestation-frames"
    policy, environment, pycache_binding = _main_suite_contract_inputs(repo_root)
    bytecode_policy = {
        "dont_write_bytecode": True,
        "pycache_prefix": environment["PYTHONPYCACHEPREFIX"],
    }
    frames = [
        {
            "payload": {
                "challenge_binding_sha256": challenge_sha256,
                "environment": environment,
                "frame": "pre_import",
                "pid": 101,
                "ppid": 100,
                "runtime": {"bytecode_policy": bytecode_policy},
            },
            "payload_sha256": "1" * 64,
            "payload_size_bytes": 1,
            "phase": 1,
        },
        {
            "payload": {
                "challenge_binding_sha256": challenge_sha256,
                "frame": "pre_collection",
                "pid": 101,
                "ppid": 100,
                "runtime": {"bytecode_policy": bytecode_policy},
            },
            "payload_sha256": "2" * 64,
            "payload_size_bytes": 1,
            "phase": 2,
        },
        {
            "payload": {
                "challenge_binding_sha256": challenge_sha256,
                "final_loaded_modules": {},
                "frame": "terminal_complete",
                "pid": 101,
                "ppid": 100,
                "pytest_exit_code": 0,
            },
            "payload_sha256": "3" * 64,
            "payload_size_bytes": 1,
            "phase": 3,
        },
    ]
    tail = (
        struct.pack(">Q", len(stdout))
        + stdout
        + struct.pack(">Q", len(stderr))
        + stderr
        + struct.pack(">Q", len(attestation))
        + attestation
    )
    receipt = session._seal(
        {
            "accepted": True,
            "attestations": frames,
            "authority": False,
            "challenge_binding": {
                "kind": challenge_kind,
                "sha256": challenge_sha256,
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
                    "/bound/python",
                    "-I",
                    "-S",
                    "-B",
                    "-X",
                    f"pycache_prefix={environment['PYTHONPYCACHEPREFIX']}",
                    "/bound/wrapper",
                    str(repo_root / session.MAIN_SUITE_POLICY_PATH),
                    _policy_bindings(repo_root)["policy_binding"]["sha256"],
                    "--",
                    *session.MAIN_SUITE_PYTEST_ARGS,
                ],
                "cwd": str(repo_root),
                "environment": environment,
            },
            "external_after": {
                "pycache_prefix": pycache_binding,
                "snapshot_sha256": "4" * 64,
            },
            "external_before": {
                "pycache_prefix": pycache_binding,
                "snapshot_sha256": "4" * 64,
            },
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
            "framing": session.MAIN_SUITE_FRAMING,
            "limitations": list(session.LIMITATIONS),
            "outcome": "PASSED",
            **_policy_bindings(repo_root),
            "protocol_version": session.PROTOCOL_VERSION,
            "schema_id": session.MAIN_SUITE_RECEIPT_SCHEMA_ID,
            "streams": {
                "attestation": {
                    "offset_bytes": 8 + len(stdout) + 8 + len(stderr) + 8,
                    "sha256": hashlib.sha256(attestation).hexdigest(),
                    "size_bytes": len(attestation),
                },
                "stderr": {
                    "offset_bytes": 8 + len(stdout) + 8,
                    "sha256": hashlib.sha256(stderr).hexdigest(),
                    "size_bytes": len(stderr),
                },
                "stdout": {
                    "offset_bytes": 8,
                    "sha256": hashlib.sha256(stdout).hexdigest(),
                    "size_bytes": len(stdout),
                },
                "tail_sha256": hashlib.sha256(tail).hexdigest(),
                "tail_size_bytes": len(tail),
            },
            "timing": {"phase1_elapsed_ms": 1, "phase2_elapsed_ms": 2},
            "version": session.MAIN_SUITE_RECEIPT_VERSION,
        }
    )
    raw = session.MAIN_SUITE_RECEIPT_PREFIX + session._canonical_bytes(receipt) + b"\n" + tail
    return {
        "attestation": attestation,
        "raw": raw,
        "receipt": receipt,
        "stderr": stderr,
        "stdout": stdout,
    }


def _replace_main_suite_receipt(
    result: dict[str, object],
    receipt: dict[str, object],
) -> dict[str, object]:
    candidate = dict(receipt)
    candidate.pop("semantic_sha256", None)
    sealed = session._seal(candidate)
    raw = result["raw"]
    assert type(raw) is bytes
    tail = raw.split(b"\n", 1)[1]
    changed = dict(result)
    changed["receipt"] = sealed
    changed["raw"] = (
        session.MAIN_SUITE_RECEIPT_PREFIX + session._canonical_bytes(sealed) + b"\n" + tail
    )
    return changed


def _schema_tree_descriptor(*, fill: str) -> dict[str, object]:
    return {
        "byte_inventory_sha256": fill * 64,
        "directory_count": 1,
        "entry_count": 2,
        "file_count": 1,
        "total_file_bytes": 10,
        "tree_inventory_sha256": fill * 64,
    }


def _schema_symlink_binding(path: Path) -> dict[str, object]:
    return {
        "gid": os.getgid(),
        "link_text": "/bound/python",
        "mode": "0777",
        "path": str(path),
        "size_bytes": 13,
        "st_dev": 1,
        "st_ino": 3,
        "st_nlink": 1,
        "uid": os.getuid(),
    }


def _schema_pycache_binding(path: Path) -> dict[str, object]:
    return {
        "gid": os.getgid(),
        "mode": "0700",
        "path": str(path),
        "st_ctime_ns": 11,
        "st_dev": 1,
        "st_ino": 4,
        "st_mtime_ns": 12,
        "st_nlink": 2,
        "uid": os.getuid(),
    }


def _schema_runtime_snapshot(repo_root: Path) -> dict[str, object]:
    prefix = repo_root / "runtime" / "cache" / "pycache"
    resolved = _main_suite_binding(repo_root / "runtime" / "python", fill="8")
    lexical = _schema_symlink_binding(repo_root / "runtime" / "python-link")
    startup_file = {"path": str(repo_root / "runtime" / "sitecustomize.py"), "present": False}
    runtime_state = {
        "sys_base_prefix": "/bound/base",
        "sys_executable": str(repo_root / "runtime" / "python-link"),
        "sys_exec_prefix": "/bound/base",
        "sys_path": ["/bound/lib"],
        "sys_prefix": "/bound/base",
        "version_info": [3, 13, 7],
    }
    routing = {
        "candidate_root": str(repo_root),
        "quant_investor_origin": str(repo_root / "quant_investor" / "__init__.py"),
        "removed_authority_entries": [],
        "runtime_state": runtime_state,
        "startup": {
            "lexical_python": lexical,
            "resolved_python": resolved,
            "startup_files": [startup_file],
            "wrapper": _main_suite_binding(
                repo_root / "scripts/v17_phase0_main_suite_wrapper.py",
                fill="9",
            ),
        },
        "startup_modules": [
            {
                **_main_suite_binding(
                    repo_root / "runtime" / "site.py",
                    fill="a",
                ),
                "module": "site",
            }
        ],
    }
    return {
        "bytecode_policy": {
            "dont_write_bytecode": True,
            "pycache_prefix": str(prefix),
        },
        "factor_authority_sha256": "b" * 64,
        "interpreter": resolved,
        "invalid_dist_info_sha256": "c" * 64,
        "inventory": {
            "count": 3,
            "physical_dist_info_count": 3,
            "physical_dist_info_names_sha256": "d" * 64,
            "rows_sha256": "e" * 64,
        },
        "loaded_modules": {
            "classification_counts": {"runtime": 3},
            "count": 3,
            "rows_sha256": "f" * 64,
        },
        "policy_sha256": "1" * 64,
        "project_modules": [],
        "routing": routing,
    }


def _schema_external_snapshot(repo_root: Path) -> dict[str, object]:
    labels = [
        "wrapper_binding",
        "harness_binding",
        "candidate_conftest",
        "package_manifest",
        "runtime_policy",
        "runtime_policy_schema",
        "schema_validator_canonical",
        "schema_validator_resources",
        "schema_validator_runtime",
    ]
    return {
        "bindings": [
            {
                **_main_suite_binding(
                    repo_root / "bound" / f"{index}.bin",
                    fill=hex(index + 1)[2:],
                ),
                "label": label,
            }
            for index, label in enumerate(labels)
        ],
        "distribution_ownership_sha256": "a" * 64,
        "factor_authority_sha256": "b" * 64,
        "invalid_dist_info_sha256": "c" * 64,
        "lexical_python": _schema_symlink_binding(repo_root / "runtime" / "python-link"),
        "physical_trees": [
            {
                "descriptor": _schema_tree_descriptor(fill="d"),
                "name": "pytest-support",
            }
        ],
        "protected_roots": [
            {
                "identity": None,
                "label": f"protected-{index}",
                "path": str(repo_root / "protected" / str(index)),
                "state": "ABSENT",
            }
            for index in range(4)
        ],
        "pycache_prefix": _schema_pycache_binding(repo_root / "runtime" / "cache" / "pycache"),
        "resolved_python": _main_suite_binding(
            repo_root / "runtime" / "python",
            fill="e",
        ),
        "snapshot_sha256": "f" * 64,
        "startup_files": [
            {
                "path": str(repo_root / "runtime" / "sitecustomize.py"),
                "present": False,
            }
        ],
    }


def _schema_attestations(repo_root: Path, challenge_sha256: str) -> list[dict[str, object]]:
    runtime = _schema_runtime_snapshot(repo_root)
    common = {
        "challenge_binding_sha256": challenge_sha256,
        "pid": 101,
        "ppid": 100,
    }
    plugins = [
        {
            "distribution": f"plugin-{index}",
            "entry_point_name": f"plugin_{index}",
            "hook_trace": [],
            "module": f"plugin_{index}",
            "module_file_binding": _main_suite_binding(
                repo_root / "plugins" / f"{index}.py",
                fill=str(index + 1),
            ),
            "physical_tree": _schema_tree_descriptor(fill=str(index + 1)),
            "version": "1.0",
        }
        for index in range(3)
    ]
    return [
        {
            "payload": {
                **common,
                "environment": {
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPYCACHEPREFIX": str(repo_root / "runtime" / "cache" / "pycache"),
                },
                "frame": "pre_import",
                "runtime": runtime,
            },
            "payload_sha256": "1" * 64,
            "payload_size_bytes": 100,
            "phase": 1,
        },
        {
            "payload": {
                **common,
                "candidate_conftest": _main_suite_binding(
                    repo_root / "tests" / "conftest.py",
                    fill="2",
                ),
                "frame": "pre_collection",
                "initial_conftest_loaded": True,
                "plugins": plugins,
                "project_modules": [],
                "pytest_version": "9.0.2",
                "runtime": runtime,
                "support_trees": [
                    {
                        **_schema_tree_descriptor(fill="3"),
                        "name": "pytest-support",
                    }
                ],
            },
            "payload_sha256": "2" * 64,
            "payload_size_bytes": 200,
            "phase": 2,
        },
        {
            "payload": {
                **common,
                "final_loaded_modules": runtime["loaded_modules"],
                "frame": "terminal_complete",
                "pytest_exit_code": 0,
            },
            "payload_sha256": "3" * 64,
            "payload_size_bytes": 100,
            "phase": 3,
        },
    ]


def _structural_main_suite_receipt(
    repo_root: Path,
    *,
    frame_count: int = 3,
    rejected: bool = False,
) -> dict[str, object]:
    challenge_sha256 = "7" * 64
    frames = _schema_attestations(repo_root, challenge_sha256)[:frame_count]
    if rejected and frame_count == 3:
        frames[2]["payload"]["pytest_exit_code"] = 2  # type: ignore[index]
    environment = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(repo_root / "runtime" / "cache" / "pycache"),
    }
    external = None if rejected else _schema_external_snapshot(repo_root)
    return session._seal(
        {
            "accepted": not rejected,
            "attestations": frames,
            "authority": False,
            "challenge_binding": {
                "kind": "PHASE0_SESSION_FILE",
                "sha256": challenge_sha256,
            },
            "claims": {
                "exit_code": 2 if rejected else 0,
                "final_audit_completed": not rejected,
                "final_audit_enforced": not rejected,
                "kernel_egress_attested": False,
                "network_unreachability_proven": False,
                "offline_policy_enforced": True,
                "signal": None,
            },
            "command": {
                "argv": [
                    "/bound/python",
                    "-I",
                    "-S",
                    "-B",
                    "-X",
                    f"pycache_prefix={environment['PYTHONPYCACHEPREFIX']}",
                    "/bound/wrapper",
                ],
                "cwd": str(repo_root),
                "environment": environment,
            },
            "external_after": external,
            "external_before": external,
            "failure_codes": ["HARNESS_REJECTED"] if rejected else [],
            "failures": (
                [
                    {
                        "code": "HARNESS_REJECTED",
                        "detail": "synthetic structural rejection",
                        "phase": "PRIMARY",
                    }
                ]
                if rejected
                else []
            ),
            "finalization": {
                "cleanup": {"attempted": True, "status": "PASSED"},
                "external_after": {
                    "attempted": True,
                    "equal": None if rejected else True,
                    "status": "FAILED" if rejected else "PASSED",
                },
            },
            "framing": session.MAIN_SUITE_FRAMING,
            "limitations": list(session.LIMITATIONS),
            "outcome": "FAILED" if rejected else "PASSED",
            **_policy_bindings(repo_root),
            "protocol_version": session.PROTOCOL_VERSION,
            "schema_id": session.MAIN_SUITE_RECEIPT_SCHEMA_ID,
            "streams": {
                "attestation": {
                    "offset_bytes": 24,
                    "sha256": "4" * 64,
                    "size_bytes": 0,
                },
                "stderr": {
                    "offset_bytes": 16,
                    "sha256": "5" * 64,
                    "size_bytes": 0,
                },
                "stdout": {
                    "offset_bytes": 8,
                    "sha256": "6" * 64,
                    "size_bytes": 0,
                },
                "tail_sha256": "8" * 64,
                "tail_size_bytes": 24,
            },
            "timing": {
                "phase1_elapsed_ms": None if rejected else 1,
                "phase2_elapsed_ms": None if rejected else 2,
            },
            "version": session.MAIN_SUITE_RECEIPT_VERSION,
        }
    )


def test_protocol_constants_and_gate_plan_are_log_first() -> None:
    assert session.FAILURE_FILENAME == "99_unpublished_failure.json"
    assert session.SESSION_VERSION.endswith("phase0-session.v2")
    assert session.SKIP_BASELINE_VERSION.endswith("phase0-skip-baseline.v2")
    assert session.FAILURE_VERSION.endswith("phase0-unpublished-failure.v2")
    assert session.GATE_ROLES == (
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
    assert session.GATE_FILENAMES == (
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
    plan = session._gate_plan()
    assert [row["ordinal"] for row in plan] == list(range(1, 11))
    assert [row["role"] for row in plan] == list(session.GATE_ROLES)
    assert [row["filename"] for row in plan] == list(session.GATE_FILENAMES)
    assert all(not Path(row["producer_path"]).is_absolute() for row in plan)
    assert plan[1] == {
        "artifact_version": session.DEPENDENCY_VERSION,
        "filename": "20_native_dependency.json",
        "kind": "artifact",
        "ordinal": 2,
        "producer_path": "scripts/v17_offline_dependency_evidence.py",
        "producer_version": session.DEPENDENCY_VERSION,
        "role": "native_sync_receipt",
        "schema_id": session.DEPENDENCY_SCHEMA_ID,
    }
    assert plan[4] == {
        "artifact_version": session.MAIN_SUITE_RECEIPT_VERSION,
        "filename": "32_full_suite.log",
        "kind": "log",
        "ordinal": 5,
        "producer_path": session.MAIN_SUITE_HARNESS_PATH,
        "producer_version": session.MAIN_SUITE_RECEIPT_VERSION,
        "role": "full_offline_suite",
        "schema_id": session.MAIN_SUITE_RECEIPT_SCHEMA_ID,
    }
    assert plan[8]["producer_version"] == session.PACKAGE_PRODUCER_VERSION


def test_cli_has_only_fixed_path_inputs_and_rejects_role_resume_or_env() -> None:
    required = [
        "--repo-root",
        "/private/tmp/repo",
        "--classification-manifest",
        "/private/tmp/classification.json",
        "--skip-baseline",
        "/private/tmp/skip.json",
        "--frozen-export",
        "/private/tmp/export.txt",
        "--bundle-root",
        "/private/tmp/bundle",
        "--work-root",
        "/private/tmp/work",
    ]
    parsed = session.parse_args(required)
    assert set(vars(parsed)) == {
        "bundle_root",
        "classification_manifest",
        "frozen_export",
        "repo_root",
        "skip_baseline",
        "work_root",
    }
    for forbidden in ("--role", "--resume", "--repair", "--command", "--env"):
        with pytest.raises(SystemExit):
            session.parse_args([*required, forbidden, "forbidden"])


def test_every_closed_environment_disables_bytecode() -> None:
    assert session.BASE_CLOSED_ENVIRONMENT["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "PYTHONPATH" not in session.BASE_CLOSED_ENVIRONMENT
    assert "VIRTUAL_ENV" not in session.BASE_CLOSED_ENVIRONMENT
    assert session.BASE_CLOSED_ENVIRONMENT["UV_OFFLINE"] == "1"
    assert session.BASE_CLOSED_ENVIRONMENT["PIP_NO_INDEX"] == "1"


def test_package_source_superset_projects_current_helper_full_shape() -> None:
    binding = session._package_source_superset(REPO_ROOT)
    assert set(binding) == {"row_count", "sha256"}
    assert binding["row_count"] > 0
    assert len(binding["sha256"]) == 64


def test_final_index_sidecar_uses_sha256sum_filename_format() -> None:
    raw = b'{"status":"SEALED"}\n'
    assert session._index_sidecar_bytes(
        raw,
        filename="70_evidence_index.json",
    ) == (f"{hashlib.sha256(raw).hexdigest()}  70_evidence_index.json\n".encode("ascii"))


def test_prepare_new_roots_creates_empty_owner_0700_nonnested_roots(
    tmp_path: Path,
) -> None:
    repo = _private_directory(tmp_path / "repo")
    roots_parent = _private_directory(tmp_path / "outside")
    bundle, work = session._prepare_new_roots(
        roots_parent / "bundle",
        roots_parent / "work",
        repo_root=repo,
    )
    assert list(bundle.iterdir()) == []
    assert list(work.iterdir()) == []
    assert stat.S_IMODE(bundle.stat().st_mode) == 0o700
    assert stat.S_IMODE(work.stat().st_mode) == 0o700
    assert bundle.stat().st_uid == os.getuid()
    assert work.stat().st_uid == os.getuid()


def test_external_location_accepts_regular_file_and_rejects_symlink_component(
    tmp_path: Path,
) -> None:
    repo = _private_directory(tmp_path / "repo")
    outside = _private_directory(tmp_path / "outside")
    external = outside / "classification.json"
    external.write_bytes(b"{}\n")
    external.chmod(0o600)
    assert (
        session._validate_external_location(
            external,
            repo_root=repo,
            label="classification",
        )
        == external
    )

    real = _private_directory(tmp_path / "real")
    link = tmp_path / "outside-link"
    link.symlink_to(real, target_is_directory=True)
    linked_external = real / "classification.json"
    linked_external.write_bytes(b"{}\n")
    linked_external.chmod(0o600)
    with pytest.raises(session.Phase0SessionError, match="unsafe path component"):
        session._validate_external_location(
            link / linked_external.name,
            repo_root=repo,
            label="classification",
        )


def test_native_environment_matches_dependency_evidence_tmp_contract(
    tmp_path: Path,
) -> None:
    work = _private_directory(tmp_path / "work")
    environment = session._native_environment(work)
    assert environment["TMPDIR"] == str(work / "tmp" / "native_sync")
    assert environment["UV_PROJECT_ENVIRONMENT"] == str(work / "native_venv")
    assert stat.S_IMODE((work / "tmp").stat().st_mode) == 0o700
    assert stat.S_IMODE((work / "tmp" / "native_sync").stat().st_mode) == 0o700


def test_prepare_new_roots_rejects_reuse_nesting_and_symlink_parent(
    tmp_path: Path,
) -> None:
    repo = _private_directory(tmp_path / "repo")
    outside = _private_directory(tmp_path / "outside")
    existing = _private_directory(outside / "existing")
    with pytest.raises(session.Phase0SessionError, match="never have existed"):
        session._prepare_new_roots(
            existing,
            outside / "work",
            repo_root=repo,
        )
    with pytest.raises(session.Phase0SessionError, match="non-nested"):
        session._prepare_new_roots(
            outside / "parent",
            outside / "parent" / "child",
            repo_root=repo,
        )
    real = _private_directory(tmp_path / "real")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(session.Phase0SessionError, match="unsafe path component"):
        session._prepare_new_roots(
            link / "bundle",
            outside / "other-work",
            repo_root=repo,
        )


def test_exact_once_publisher_hardlinks_reads_back_and_never_overwrites(
    tmp_path: Path,
) -> None:
    bundle = _private_directory(tmp_path / "bundle")
    publisher = session.BundlePublisher(bundle, session_id="phase0-test")
    raw = b'{"status":"INITIALIZED"}\n'
    try:
        binding = publisher.publish("00_session.json", raw)
        target = bundle / "00_session.json"
        assert target.read_bytes() == raw
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
        assert target.stat().st_nlink == 1
        assert binding["sha256"] == hashlib.sha256(raw).hexdigest()
        assert not any(path.name.startswith(".") for path in bundle.iterdir())
        with pytest.raises(session.Phase0SessionError, match="already exists"):
            publisher.publish("00_session.json", b"replacement\n")
        assert target.read_bytes() == raw
        assert any(path.name.startswith(".") for path in bundle.iterdir())
    finally:
        publisher.close()


def test_publication_failure_retains_orphan_and_never_creates_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _private_directory(tmp_path / "bundle")
    publisher = session.BundlePublisher(bundle, session_id="phase0-orphan")

    def fail_link(*_args: object, **_kwargs: object) -> None:
        raise OSError("synthetic hardlink failure")

    monkeypatch.setattr(session.os, "link", fail_link)
    try:
        with pytest.raises(session.Phase0SessionError, match="hard-link"):
            publisher.publish("00_session.json", b"immutable\n")
    finally:
        publisher.close()
    assert not (bundle / "00_session.json").exists()
    staged = list(bundle.iterdir())
    assert [path.name for path in staged] == [".00_session.json.staged-phase0-orphan"]
    assert staged[0].read_bytes() == b"immutable\n"
    assert stat.S_IMODE(staged[0].stat().st_mode) == 0o600


def test_command_framing_offsets_are_relative_to_framed_remainder() -> None:
    captures = [
        _capture(stdout=b"abc", stderr=b"de"),
        _capture(stdout=b"", stderr=b"f"),
    ]
    framed, commands = session._frame_commands(captures)
    expected = (
        struct.pack(">Q", 3)
        + b"abc"
        + struct.pack(">Q", 2)
        + b"de"
        + struct.pack(">Q", 0)
        + b""
        + struct.pack(">Q", 1)
        + b"f"
    )
    assert framed == expected
    assert commands[0]["stdout_offset_bytes"] == 8
    assert commands[0]["stderr_offset_bytes"] == 19
    assert commands[1]["stdout_offset_bytes"] == 29
    assert commands[1]["stderr_offset_bytes"] == 37
    assert commands[0]["signal"] is None
    assert commands[1]["signal"] is None
    assert commands[0]["stdout_sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert commands[1]["stderr_sha256"] == hashlib.sha256(b"f").hexdigest()


def test_full_suite_claims_require_exact_frozen_skip_rows() -> None:
    raw = (
        b"SKIPPED [42] tests/unit/test_offline.py:17: requires optional offline asset\n"
        b"8 passed, 42 skipped in 1.00s\n"
    )
    captures = [_capture(stdout=raw, stderr=b"")]
    framed, _commands = session._frame_commands(captures)
    entries = [
        {
            "count": 42,
            "line": 17,
            "path": "tests/unit/test_offline.py",
            "reason": "requires optional offline asset",
        }
    ]
    claims = session._log_claims(
        "full_offline_suite",
        captures,
        framed,
        skip_baseline={"entries": entries},
    )
    assert claims == {
        "errors": 0,
        "exit_code": 0,
        "failed": 0,
        "passed": 8,
        "raw_output_sha256": hashlib.sha256(raw).hexdigest(),
        "skip_allowlist": entries,
        "skipped": 42,
        "xfail": 0,
        "xpass": 0,
    }
    with pytest.raises(session.Phase0SessionError, match="frozen baseline"):
        session._log_claims(
            "full_offline_suite",
            captures,
            framed,
            skip_baseline={"entries": [{**entries[0], "line": 18}]},
        )


def test_main_suite_result_requires_exact_raw_challenge_and_terminal_frame(
    tmp_path: Path,
) -> None:
    challenge_sha256 = "a" * 64
    result = _main_suite_result(
        challenge_kind="PHASE0_SESSION_FILE",
        challenge_sha256=challenge_sha256,
        repo_root=tmp_path,
        stdout=b"8 passed, 42 skipped in 1.00s\n",
    )
    policy_bindings = _policy_bindings(tmp_path)
    policy, environment, pycache_binding = _main_suite_contract_inputs(tmp_path)
    assert (
        session._validate_main_suite_result(
            result,
            repo_root=tmp_path,
            policy=policy,
            policy_bindings=policy_bindings,
            expected_environment=environment,
            expected_pycache_binding=pycache_binding,
            challenge_binding_kind="PHASE0_SESSION_FILE",
            challenge_binding_sha256=challenge_sha256,
        )
        == result
    )

    wrong_raw = dict(result)
    wrong_raw["raw"] = result["raw"] + b"x"
    with pytest.raises(session.Phase0SessionError, match="framing"):
        session._validate_main_suite_result(
            wrong_raw,
            repo_root=tmp_path,
            policy=policy,
            policy_bindings=policy_bindings,
            expected_environment=environment,
            expected_pycache_binding=pycache_binding,
            challenge_binding_kind="PHASE0_SESSION_FILE",
            challenge_binding_sha256=challenge_sha256,
        )

    rejected = copy.deepcopy(result)
    rejected["receipt"]["accepted"] = False
    rejected["receipt"]["semantic_sha256"] = session._semantic_sha256(rejected["receipt"])
    rejected["raw"] = (
        session.MAIN_SUITE_RECEIPT_PREFIX
        + session._canonical_bytes(rejected["receipt"])
        + b"\n"
        + result["raw"].split(b"\n", 1)[1]
    )
    structurally_valid = session._validate_main_suite_result(
        rejected,
        repo_root=tmp_path,
        policy=policy,
        policy_bindings=policy_bindings,
        expected_environment=environment,
        expected_pycache_binding=pycache_binding,
        challenge_binding_kind="PHASE0_SESSION_FILE",
        challenge_binding_sha256=challenge_sha256,
    )
    with pytest.raises(session.MainSuiteRejectedError, match="rejected"):
        session._require_main_suite_accepted(
            structurally_valid,
            stage="full_offline_suite",
        )


def test_main_suite_receipt_schema_runs_before_semantic_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    challenge_sha256 = "d" * 64
    result = _main_suite_result(
        challenge_kind="PHASE0_SESSION_FILE",
        challenge_sha256=challenge_sha256,
        repo_root=tmp_path,
        stdout=b"8 passed, 42 skipped in 1.00s\n",
    )
    policy, environment, pycache_binding = _main_suite_contract_inputs(tmp_path)
    order: list[str] = []

    def checked_schema(value: object, **_kwargs: object) -> None:
        assert value == result["receipt"]
        order.append("schema")

    monkeypatch.setattr(session, "_checked_schema", checked_schema)
    assert (
        session._validate_main_suite_contract_result(
            result,
            repo_root=tmp_path,
            policy=policy,
            policy_bindings=_policy_bindings(tmp_path),
            expected_environment=environment,
            expected_pycache_binding=pycache_binding,
            challenge_binding_kind="PHASE0_SESSION_FILE",
            challenge_binding_sha256=challenge_sha256,
            stage="full_offline_suite",
        )
        == result
    )
    assert order == ["schema"]
    assert result["receipt"]["finalization"] == {
        "cleanup": {"attempted": True, "status": "PASSED"},
        "external_after": {
            "attempted": True,
            "equal": True,
            "status": "PASSED",
        },
    }

    tampered_receipt = copy.deepcopy(result["receipt"])
    tampered_receipt["policy_manifest_binding"]["sha256"] = "0" * 64
    tampered = _replace_main_suite_receipt(result, tampered_receipt)
    order.clear()

    def checked_tampered(value: object, **_kwargs: object) -> None:
        assert value == tampered["receipt"]
        order.append("schema")

    monkeypatch.setattr(session, "_checked_schema", checked_tampered)
    with pytest.raises(session.Phase0SessionError, match="policy receipt binding"):
        session._validate_main_suite_contract_result(
            tampered,
            repo_root=tmp_path,
            policy=policy,
            policy_bindings=_policy_bindings(tmp_path),
            expected_environment=environment,
            expected_pycache_binding=pycache_binding,
            challenge_binding_kind="PHASE0_SESSION_FILE",
            challenge_binding_sha256=challenge_sha256,
            stage="full_offline_suite",
        )
    assert order == ["schema"]


@pytest.mark.parametrize(
    ("secondary_phase", "secondary_code", "expected_finalization"),
    [
        (
            "CLEANUP",
            "CLEANUP_FAILED",
            {
                "cleanup": {"attempted": True, "status": "FAILED"},
                "external_after": {
                    "attempted": True,
                    "equal": True,
                    "status": "PASSED",
                },
            },
        ),
        (
            "EXTERNAL_AFTER",
            "EXTERNAL_AFTER_MISMATCH",
            {
                "cleanup": {"attempted": True, "status": "PASSED"},
                "external_after": {
                    "attempted": True,
                    "equal": False,
                    "status": "FAILED",
                },
            },
        ),
    ],
)
def test_rejected_main_suite_preserves_secondary_failure_and_finalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    secondary_phase: str,
    secondary_code: str,
    expected_finalization: dict[str, object],
) -> None:
    challenge_sha256 = "e" * 64
    result = _main_suite_result(
        challenge_kind="PHASE0_SESSION_FILE",
        challenge_sha256=challenge_sha256,
        repo_root=tmp_path,
        stdout=b"8 passed, 42 skipped in 1.00s\n",
    )
    policy, environment, pycache_binding = _main_suite_contract_inputs(tmp_path)
    receipt = copy.deepcopy(result["receipt"])
    receipt["accepted"] = False
    receipt["outcome"] = "FAILED"
    receipt["failure_codes"] = ["PRIMARY_FAILURE", secondary_code]
    receipt["failures"] = [
        {
            "code": "PRIMARY_FAILURE",
            "detail": "primary harness rejection",
            "phase": "PRIMARY",
        },
        {
            "code": secondary_code,
            "detail": f"{secondary_phase.casefold()} audit failed",
            "phase": secondary_phase,
        },
    ]
    receipt["finalization"] = expected_finalization
    if secondary_phase == "EXTERNAL_AFTER":
        receipt["external_after"] = None
    rejected = _replace_main_suite_receipt(result, receipt)
    monkeypatch.setattr(session, "_checked_schema", lambda *_args, **_kwargs: None)

    with pytest.raises(session.MainSuiteRejectedError) as captured:
        session._validate_main_suite_contract_result(
            rejected,
            repo_root=tmp_path,
            policy=policy,
            policy_bindings=_policy_bindings(tmp_path),
            expected_environment=environment,
            expected_pycache_binding=pycache_binding,
            challenge_binding_kind="PHASE0_SESSION_FILE",
            challenge_binding_sha256=challenge_sha256,
            stage="full_offline_suite",
        )
    error = captured.value
    assert error.main_suite_failures == receipt["failures"]
    assert error.main_suite_finalization == expected_finalization

    bundle = _private_directory(tmp_path / f"bundle-{secondary_phase.casefold()}")
    work = _private_directory(tmp_path / f"work-{secondary_phase.casefold()}")
    report = session._failure_report(
        error=error,
        session_id=f"phase0-{secondary_phase.casefold()}",
        repo_root=REPO_ROOT,
        bundle_root=bundle,
        work_root=work,
    )
    assert [event["phase"] for event in report["failures"]] == [
        "PRIMARY",
        secondary_phase,
    ]
    assert secondary_code in report["failures"][1]["message"]
    assert report["main_suite_finalization"] == {
        "attempted": True,
        **expected_finalization,
    }


def test_final_full_suite_publishes_exact_harness_raw_with_session_file_challenge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    challenge_sha256 = "b" * 64
    transcript = (
        b"SKIPPED [42] tests/unit/test_offline.py:17: requires optional offline asset\n"
        b"8 passed, 42 skipped in 1.00s\n"
    )
    result = _main_suite_result(
        challenge_kind="PHASE0_SESSION_FILE",
        challenge_sha256=challenge_sha256,
        repo_root=tmp_path,
        stdout=transcript,
    )
    calls: list[dict[str, object]] = []

    def fake_run(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return result

    class Publisher:
        def __init__(self) -> None:
            self.rows: list[tuple[str, bytes]] = []

        def publish(self, name: str, raw: bytes) -> dict[str, object]:
            self.rows.append((name, raw))
            return {"sha256": hashlib.sha256(raw).hexdigest()}

    monkeypatch.setattr(session, "_run_main_suite_contract", fake_run)
    invariants = {
        "package_source_superset": {"row_count": 1, "sha256": "d" * 64},
        "source_state": {"base_commit": "c" * 40},
    }
    monkeypatch.setattr(session, "_capture_invariants", lambda *_args: invariants)
    monkeypatch.setattr(session, "_assert_invariants_equal", lambda *_args, **_kwargs: None)
    publisher = Publisher()
    session._publish_main_suite_gate(
        repo_root=tmp_path,
        base_commit="c" * 40,
        work_root=tmp_path / "work",
        session={},
        session_binding={"sha256": challenge_sha256},
        publisher=publisher,  # type: ignore[arg-type]
        skip_baseline={
            "entries": [
                {
                    "count": 42,
                    "line": 17,
                    "path": "tests/unit/test_offline.py",
                    "reason": "requires optional offline asset",
                }
            ]
        },
    )
    assert calls == [
        {
            "repo_root": tmp_path,
            "base_commit": "c" * 40,
            "work_root": tmp_path / "work",
            "stage": "full_suite",
            "source_state": invariants["source_state"],
            "package_binding": invariants["package_source_superset"],
            "challenge_binding_kind": "PHASE0_SESSION_FILE",
            "challenge_binding_sha256": challenge_sha256,
        }
    ]
    assert publisher.rows == [("32_full_suite.log", result["raw"])]


def test_failure_receipt_is_unpublished_non_authorizing_and_semantically_bound(
    tmp_path: Path,
) -> None:
    bundle = _private_directory(tmp_path / "bundle")
    work = _private_directory(tmp_path / "work")
    report = session._failure_report(
        error=session.Phase0SessionError("fixed failure", stage="mypy"),
        session_id="phase0-test",
        repo_root=REPO_ROOT,
        bundle_root=bundle,
        work_root=work,
        additional_failures=[
            ("EXTERNAL_AFTER", RuntimeError("external-after failed")),
            ("CLEANUP", OSError("cleanup failed")),
        ],
    )
    assert report["status"] == "UNPUBLISHED"
    assert report["authority"] is False
    assert report["main_suite_finalization"] == session.MAIN_SUITE_FINALIZATION_NOT_ATTEMPTED
    assert report["failures"] == [
        {
            "exception_type": "Phase0SessionError",
            "message": "fixed failure",
            "ordinal": 1,
            "phase": "PRIMARY",
            "stage": "mypy",
        },
        {
            "exception_type": "RuntimeError",
            "message": "external-after failed",
            "ordinal": 2,
            "phase": "EXTERNAL_AFTER",
            "stage": "external_after",
        },
        {
            "exception_type": "OSError",
            "message": "cleanup failed",
            "ordinal": 3,
            "phase": "CLEANUP",
            "stage": "cleanup",
        },
    ]
    assert report["semantic_sha256"] == session._semantic_sha256(report)
    assert session._validate_failure_report(report) == report
    schema = json.loads(
        (
            REPO_ROOT / "scripts" / "schemas" / "v17_phase0_unpublished_failure.v2.schema.json"
        ).read_text(encoding="utf-8")
    )
    preflight_packaged_schema(schema)
    validate_instance_against_schema(report, schema)


def test_failure_receipt_rejects_noncontiguous_or_duplicate_phases(tmp_path: Path) -> None:
    bundle = _private_directory(tmp_path / "bundle")
    work = _private_directory(tmp_path / "work")
    report = session._failure_report(
        error=RuntimeError("primary"),
        session_id="phase0-test",
        repo_root=REPO_ROOT,
        bundle_root=bundle,
        work_root=work,
    )
    changed = json.loads(json.dumps(report))
    changed["failures"][0]["ordinal"] = 2
    changed["semantic_sha256"] = session._semantic_sha256(changed)
    with pytest.raises(session.Phase0SessionError, match="event is invalid"):
        session._validate_failure_report(changed)
    with pytest.raises(session.Phase0SessionError, match="duplicated"):
        session._failure_report(
            error=RuntimeError("primary"),
            session_id="phase0-test",
            repo_root=REPO_ROOT,
            bundle_root=bundle,
            work_root=work,
            additional_failures=[
                ("CLEANUP", OSError("first")),
                ("CLEANUP", OSError("second")),
            ],
        )


def test_normative_limitations_match_the_authoritative_five_string_order() -> None:
    assert session.LIMITATIONS == [
        "PRECOLLECTION_ALLOWED_TREES_ATTESTED_NOT_COMPLETE_MAIN_RUNTIME_FILESET",
        "NO_DIRECT_LAUNCH_REFERENCE_ONLY_NOT_PROCESS_IO_ATTESTED",
        "OFFLINE_FLAGS_ONLY_NOT_OS_EGRESS_ATTESTED",
        "OWNER_DECLARED_PATHS_ONLY_NOT_CONTENT_PROVENANCE",
        "PIP_25_2_PACKAGE_INSTALL_ENV_ONLY_NATIVE_AND_BUILD_ENV_PIP_ABSENT",
    ]


@pytest.mark.parametrize(
    ("relative", "schema_id", "version_field", "version"),
    (
        (
            "scripts/schemas/v17_phase0_session.v2.schema.json",
            session.SESSION_SCHEMA_ID,
            "version",
            session.SESSION_VERSION,
        ),
        (
            "scripts/schemas/v17_phase0_command_receipt.v2.schema.json",
            session.COMMAND_RECEIPT_SCHEMA_ID,
            "version",
            session.COMMAND_RECEIPT_VERSION,
        ),
        (
            session.MAIN_SUITE_RECEIPT_SCHEMA_PATH,
            session.MAIN_SUITE_RECEIPT_SCHEMA_ID,
            "version",
            session.MAIN_SUITE_RECEIPT_VERSION,
        ),
        (
            "scripts/schemas/v17_phase0_unpublished_failure.v2.schema.json",
            session.FAILURE_SCHEMA_ID,
            "version",
            session.FAILURE_VERSION,
        ),
    ),
)
def test_owned_schemas_are_self_contained_closed_draft_2020_12(
    relative: str,
    schema_id: str,
    version_field: str,
    version: str,
) -> None:
    schema = json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == schema_id
    assert schema["additionalProperties"] is False
    assert schema["properties"][version_field] == {"const": version}
    serialized = json.dumps(schema, sort_keys=True)
    assert "http://" not in serialized
    assert "https://" not in serialized.replace(
        "https://json-schema.org/draft/2020-12/schema",
        "",
    )


def test_main_suite_receipt_schema_accepts_passed_and_rejected_prefixes() -> None:
    schema = json.loads(
        (REPO_ROOT / session.MAIN_SUITE_RECEIPT_SCHEMA_PATH).read_text(encoding="utf-8")
    )
    preflight_packaged_schema(schema)
    validate_instance_against_schema(
        _structural_main_suite_receipt(REPO_ROOT),
        schema,
    )
    for frame_count in range(4):
        validate_instance_against_schema(
            _structural_main_suite_receipt(
                REPO_ROOT,
                frame_count=frame_count,
                rejected=True,
            ),
            schema,
        )


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda receipt: receipt["command"]["environment"].update({"BROKEN": 1}),
            "invalid JSON type",
        ),
        (
            lambda receipt: receipt["attestations"][0]["payload"].update({"unexpected": True}),
            "oneOf",
        ),
        (
            lambda receipt: receipt["attestations"][1].update({"phase": 1}),
            "oneOf",
        ),
        (
            lambda receipt: receipt["attestations"][2]["payload"].update({"pytest_exit_code": 6}),
            "oneOf",
        ),
        (
            lambda receipt: receipt["external_before"]["pycache_prefix"].pop("st_ctime_ns"),
            "oneOf",
        ),
        (
            lambda receipt: receipt["attestations"][0]["payload"]["runtime"][
                "bytecode_policy"
            ].update({"pycache_prefix": "relative"}),
            "oneOf",
        ),
    ],
)
def test_main_suite_receipt_schema_rejects_nested_forgery(
    mutator: object,
    match: str,
) -> None:
    schema = json.loads(
        (REPO_ROOT / session.MAIN_SUITE_RECEIPT_SCHEMA_PATH).read_text(encoding="utf-8")
    )
    receipt = _structural_main_suite_receipt(REPO_ROOT)
    assert callable(mutator)
    mutator(receipt)
    with pytest.raises(SchemaValidationError, match=match):
        validate_instance_against_schema(receipt, schema)


def test_session_and_command_schemas_reject_mismatched_gate_tuples() -> None:
    session_schema = json.loads(
        (REPO_ROOT / "scripts/schemas/v17_phase0_session.v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    command_schema = json.loads(
        (REPO_ROOT / "scripts/schemas/v17_phase0_command_receipt.v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    gate_schema = {
        "$id": "myquant.v17.v2.test.phase0-gate-plan-row.schema.v1",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **session_schema["$defs"]["gate_plan"],
    }
    step_schema = {
        "$id": "myquant.v17.v2.test.phase0-command-step.schema.v1",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **command_schema["$defs"]["step"],
    }
    preflight_packaged_schema(gate_schema)
    preflight_packaged_schema(step_schema)

    plan = session._gate_plan()
    for row in plan:
        validate_instance_against_schema(row, gate_schema)
    log_rows = [row for row in plan if row["artifact_version"] == session.COMMAND_RECEIPT_VERSION]
    for row in log_rows:
        validate_instance_against_schema(
            {
                "filename": row["filename"],
                "kind": row["kind"],
                "ordinal": row["ordinal"],
                "role": row["role"],
            },
            step_schema,
        )

    mismatched_gate = dict(plan[4])
    mismatched_gate["filename"] = "33_mypy.log"
    with pytest.raises(SchemaValidationError, match="oneOf"):
        validate_instance_against_schema(mismatched_gate, gate_schema)

    mismatched_step = {
        "filename": "32_full_suite.log",
        "kind": "log",
        "ordinal": 6,
        "role": "mypy",
    }
    with pytest.raises(SchemaValidationError, match="oneOf"):
        validate_instance_against_schema(mismatched_step, step_schema)


def test_bounded_command_requires_no_bytecode_and_captures_split_streams(
    tmp_path: Path,
) -> None:
    environment = {
        **session.BASE_CLOSED_ENVIRONMENT,
        "TMPDIR": str(tmp_path),
    }
    capture = session._run_bounded_command(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import sys;sys.stdout.buffer.write(b'out');sys.stderr.buffer.write(b'err')",
        ],
        cwd=tmp_path,
        environment=environment,
        tool_version="cpython test",
        repo_root=tmp_path,
    )
    assert capture["exit_code"] == 0
    assert capture["signal"] is None
    assert capture["stdout"] == b"out"
    assert capture["stderr"] == b"err"
    assert capture["environment"]["PYTHONDONTWRITEBYTECODE"] == "1"
    with pytest.raises(session.Phase0SessionError, match="disable bytecode"):
        session._run_bounded_command(
            [sys.executable, "-I", "-B", "-c", "pass"],
            cwd=tmp_path,
            environment={},
            tool_version="cpython test",
            repo_root=tmp_path,
        )
