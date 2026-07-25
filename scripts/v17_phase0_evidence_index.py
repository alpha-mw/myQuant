#!/usr/bin/env python3
"""Seal one fail-closed, offline v17 Phase 0 evidence session.

The tool performs no dependency resolution, network access, legacy-v17 runtime
import, or repository write.  It accepts only the fixed repository-owned
session layout, validates every v2 producer artifact against its separately
identified schema, writes the hash-freeze and gate-manifest artifacts, and
publishes the final evidence index plus byte-SHA sidecar exact-once.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import subprocess
import sys
import types
from typing import Any, Mapping, Sequence

PROTOCOL_VERSION = "myquant.v17.v2"
EVIDENCE_INDEX_VERSION = "myquant.v17.v2.phase0-evidence-index.v2"
EVIDENCE_INDEX_SCHEMA_ID = "myquant.v17.v2.phase0-evidence-index.schema.v2"
CLASSIFICATION_VERSION = "myquant.v17.v2.phase0-pre-existing-classification.v2"
CLASSIFICATION_SCHEMA_ID = "myquant.v17.v2.phase0-pre-existing-classification.schema.v2"
SESSION_VERSION = "myquant.v17.v2.phase0-session.v2"
SESSION_SCHEMA_ID = "myquant.v17.v2.phase0-session.schema.v2"
COMMAND_RECEIPT_VERSION = "myquant.v17.v2.phase0-command-receipt.v2"
COMMAND_RECEIPT_SCHEMA_ID = "myquant.v17.v2.phase0-command-receipt.schema.v2"
MAIN_SUITE_RUNTIME_POLICY_VERSION = "myquant.v17.v2.phase0-main-suite-runtime-policy.v1"
MAIN_SUITE_RUNTIME_POLICY_SCHEMA_ID = "myquant.v17.v2.phase0-main-suite-runtime-policy.schema.v1"
MAIN_SUITE_RECEIPT_VERSION = "myquant.v17.v2.phase0-main-suite-receipt.v1"
MAIN_SUITE_RECEIPT_SCHEMA_ID = "myquant.v17.v2.phase0-main-suite-receipt.schema.v1"
DEPENDENCY_RECEIPT_VERSION = "v17_third_party_dependency_environment_evidence.v2"
DEPENDENCY_RECEIPT_SCHEMA_ID = (
    "myquant.v17.v2.third-party-dependency-environment-evidence.schema.v2"
)
SKIP_BASELINE_VERSION = "myquant.v17.v2.phase0-skip-baseline.v2"
SKIP_BASELINE_SCHEMA_ID = "myquant.v17.v2.phase0-skip-baseline.schema.v2"
SKIP_BASELINE_PRODUCER_VERSION = "myquant.v17.v2.phase0-skip-baseline-producer.v2"
PACKAGE_EVIDENCE_VERSION = "myquant.v17.v2.phase0-package-parity-evidence.v2"
PACKAGE_EVIDENCE_SCHEMA_ID = "myquant.v17.v2.phase0-package-parity-evidence.schema.v2"
PACKAGE_PRODUCER_VERSION = "myquant.v17.v2.phase0-package-evidence-producer.v2"
HASH_FREEZE_VERSION = "myquant.v17.v2.phase0-hash-freeze.v2"
HASH_FREEZE_SCHEMA_ID = "myquant.v17.v2.phase0-hash-freeze.schema.v2"
GATE_MANIFEST_VERSION = "myquant.v17.v2.phase0-gate-manifest.v2"
GATE_MANIFEST_SCHEMA_ID = "myquant.v17.v2.phase0-gate-manifest.schema.v2"
FAILURE_VERSION = "myquant.v17.v2.phase0-unpublished-failure.v2"
FAILURE_SCHEMA_ID = "myquant.v17.v2.phase0-unpublished-failure.schema.v2"
INDEX_PRODUCER_VERSION = "myquant.v17.v2.phase0-evidence-index-producer.v2"
SESSION_PRODUCER_VERSION = "myquant.v17.v2.phase0-evidence-session-runner.v2"
CLASSIFICATION_PROVENANCE = "OWNER_DECLARED_PATHS_ONLY_NOT_CONTENT_PROVENANCE"
PRE_EXISTING_CLASSIFICATION = "PRE_EXISTING_NON_PHASE0"
PHASE0_CLASSIFICATION = "PHASE0_ALLOWED"
SEMANTIC_FIELD = "semantic_sha256"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$", re.ASCII)
TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$", re.ASCII)
MAX_EXTERNAL_BYTES = 512 * 1024 * 1024
MAX_GIT_CAPTURE_BYTES = 512 * 1024 * 1024
MAX_COMMAND_STREAM_BYTES = 128 * 1024 * 1024
MAX_COMMAND_BYTES = 256 * 1024 * 1024
MAIN_SUITE_ATTEST_MAGIC = b"MQP0AT01"
MAIN_SUITE_ATTEST_PROTOCOL_VERSION = 1
MAIN_SUITE_ATTEST_HEADER = struct.Struct(">8sBBHI32s32s")
MAX_MAIN_SUITE_FRAME_BYTES = 1024 * 1024
MAX_MAIN_SUITE_TERMINAL_FRAME_BYTES = 16 * 1024
MAX_MAIN_SUITE_ATTESTATION_BYTES = (
    2 * (MAIN_SUITE_ATTEST_HEADER.size + MAX_MAIN_SUITE_FRAME_BYTES)
    + MAIN_SUITE_ATTEST_HEADER.size
    + MAX_MAIN_SUITE_TERMINAL_FRAME_BYTES
)
MAX_MAIN_SUITE_TAIL_BYTES = 2 * MAX_COMMAND_STREAM_BYTES + MAX_MAIN_SUITE_ATTESTATION_BYTES + 24
COMMAND_RECEIPT_PREFIX = b"MYQUANT_PHASE0_COMMAND_RECEIPT="
MAIN_SUITE_RECEIPT_PREFIX = b"MYQUANT_PHASE0_MAIN_SUITE_RECEIPT="
MAIN_SUITE_POLICY_PATH = (
    "quant_investor/v17_v2_contract/resources/main_suite_runtime_policy.v1.json"
)
MAIN_SUITE_POLICY_SCHEMA_PATH = (
    "quant_investor/v17_v2_contract/schemas/main_suite_runtime_policy.v1.schema.json"
)
MAIN_SUITE_PACKAGE_MANIFEST_PATH = (
    "quant_investor/v17_v2_contract/resources/package_manifest.v1.json"
)
MAIN_SUITE_PACKAGE_MANIFEST_VERSION = "myquant.v17.v2.package-manifest.v1"
MAIN_SUITE_HARNESS_PATH = "scripts/v17_phase0_main_suite_harness.py"
MAIN_SUITE_WRAPPER_PATH = "scripts/v17_phase0_main_suite_wrapper.py"
COMMAND_FRAMING = "UINT64_BE_STDOUT_THEN_STDOUT_UINT64_BE_STDERR_THEN_STDERR_PER_COMMAND"
MAIN_SUITE_FRAMING = (
    "UINT64_BE_STDOUT_THEN_STDOUT_UINT64_BE_STDERR_THEN_STDERR_"
    "UINT64_BE_ATTESTATION_THEN_ATTESTATION"
)
NORMATIVE_LIMITATIONS = [
    "PRECOLLECTION_ALLOWED_TREES_ATTESTED_NOT_COMPLETE_MAIN_RUNTIME_FILESET",
    "NO_DIRECT_LAUNCH_REFERENCE_ONLY_NOT_PROCESS_IO_ATTESTED",
    "OFFLINE_FLAGS_ONLY_NOT_OS_EGRESS_ATTESTED",
    "OWNER_DECLARED_PATHS_ONLY_NOT_CONTENT_PROVENANCE",
    "PIP_25_2_PACKAGE_INSTALL_ENV_ONLY_NATIVE_AND_BUILD_ENV_PIP_ABSENT",
]
PROTECTED_ROOT_SPECS = (
    ("authority_v16", Path("/Users/maxwell/mySpace/myQuant/results/v16")),
    (
        "authority_v16_operator_advisory",
        Path("/Users/maxwell/mySpace/myQuant/results/v16_operator_advisory"),
    ),
    ("candidate_v16", Path("/private/tmp/myquant-v17-neutral-baseline-20260722/results/v16")),
    (
        "candidate_v16_operator_advisory",
        Path("/private/tmp/myquant-v17-neutral-baseline-20260722/" "results/v16_operator_advisory"),
    ),
)
EXPECTED_PIP_SCOPE = {
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
V2_SESSION_KEYS = {
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

PHASE0_ALLOWED_PATTERN_REGISTRY = frozenset(
    {
        "docs/architecture/v17_v2_contract.md",
        "docs/runbooks/v17_shadow_operations.md",
        "quant_investor/v17_v2_contract/**",
        "quant_investor/v17_v2_contract/resources/main_suite_runtime_policy.v1.json",
        "quant_investor/v17_v2_contract/schemas/main_suite_runtime_policy.v1.schema.json",
        "scripts/schemas/v17_offline_dependency_evidence.v2.schema.json",
        "scripts/schemas/v17_phase0_command_receipt.v2.schema.json",
        "scripts/schemas/v17_phase0_evidence_index.v2.schema.json",
        "scripts/schemas/v17_phase0_gate_manifest.v2.schema.json",
        "scripts/schemas/v17_phase0_hash_freeze.v2.schema.json",
        "scripts/schemas/v17_phase0_main_suite_receipt.v1.schema.json",
        "scripts/schemas/v17_phase0_package_evidence.v2.schema.json",
        "scripts/schemas/v17_phase0_pre_existing_classification.v2.schema.json",
        "scripts/schemas/v17_phase0_session.v2.schema.json",
        "scripts/schemas/v17_phase0_skip_baseline.v2.schema.json",
        "scripts/schemas/v17_phase0_unpublished_failure.v2.schema.json",
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
        "tests/unit/test_v17_v2_*.py",
    }
)
FORBIDDEN_V16_ROOTS = (
    PurePosixPath("results/v16"),
    PurePosixPath("results/v16_operator_advisory"),
)
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
GATE_ORDINALS = tuple(range(1, 11))
GATE_ARTIFACT_VERSIONS = (
    COMMAND_RECEIPT_VERSION,
    DEPENDENCY_RECEIPT_VERSION,
    COMMAND_RECEIPT_VERSION,
    COMMAND_RECEIPT_VERSION,
    MAIN_SUITE_RECEIPT_VERSION,
    COMMAND_RECEIPT_VERSION,
    COMMAND_RECEIPT_VERSION,
    COMMAND_RECEIPT_VERSION,
    PACKAGE_EVIDENCE_VERSION,
    HASH_FREEZE_VERSION,
)
GATE_SCHEMA_IDS = (
    COMMAND_RECEIPT_SCHEMA_ID,
    DEPENDENCY_RECEIPT_SCHEMA_ID,
    COMMAND_RECEIPT_SCHEMA_ID,
    COMMAND_RECEIPT_SCHEMA_ID,
    MAIN_SUITE_RECEIPT_SCHEMA_ID,
    COMMAND_RECEIPT_SCHEMA_ID,
    COMMAND_RECEIPT_SCHEMA_ID,
    COMMAND_RECEIPT_SCHEMA_ID,
    PACKAGE_EVIDENCE_SCHEMA_ID,
    HASH_FREEZE_SCHEMA_ID,
)
GATE_PRODUCER_SPECS = (
    ("scripts/v17_phase0_evidence_session.py", SESSION_PRODUCER_VERSION),
    ("scripts/v17_offline_dependency_evidence.py", DEPENDENCY_RECEIPT_VERSION),
    ("scripts/v17_phase0_evidence_session.py", SESSION_PRODUCER_VERSION),
    ("scripts/v17_phase0_evidence_session.py", SESSION_PRODUCER_VERSION),
    ("scripts/v17_phase0_main_suite_harness.py", MAIN_SUITE_RECEIPT_VERSION),
    ("scripts/v17_phase0_evidence_session.py", SESSION_PRODUCER_VERSION),
    ("scripts/v17_phase0_evidence_session.py", SESSION_PRODUCER_VERSION),
    ("scripts/v17_phase0_evidence_session.py", SESSION_PRODUCER_VERSION),
    ("scripts/v17_phase0_package_evidence.py", PACKAGE_PRODUCER_VERSION),
    ("scripts/v17_phase0_evidence_index.py", INDEX_PRODUCER_VERSION),
)
NATIVE_ACCEPTED_STATUS = "THIRD_PARTY_NATIVE_ENVIRONMENT_ACCEPTED"
PACKAGE_COMMAND_ROLES = (
    "base_python_probe",
    "uv_version",
    "create_build_venv",
    "install_build_backend",
    "build_backend_probe",
    "build_sdist",
    "build_wheel_from_sdist",
    "create_install_venv",
    "ensurepip",
    "pip_version",
    "install_wheel_no_compile",
    "installed_paths_probe",
    "package_parity",
)
PACKAGE_EXPECTED_UV_VERSION = "0.10.9"
PACKAGE_EXPECTED_UV_OUTPUT = "uv 0.10.9 (f675560f3 2026-03-06)"
V2_BASE_CLOSED_ENVIRONMENT = {
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
    "UV_NO_CONFIG": "1",
    "UV_OFFLINE": "1",
    "UV_PYTHON_DOWNLOADS": "never",
}
PACKAGE_EXPECTED_PIP_VERSION = "25.2"
PACKAGE_EXPECTED_BACKEND_PACKAGES = {
    "hatchling": "1.31.0",
    "packaging": "26.2",
    "pathspec": "1.1.1",
    "pluggy": "1.6.0",
    "trove-classifiers": "2026.6.1.19",
}
PACKAGE_EXPECTED_BACKEND_INVENTORY = sorted(
    (
        {"name": name, "version": version}
        for name, version in PACKAGE_EXPECTED_BACKEND_PACKAGES.items()
    ),
    key=lambda item: item["name"].casefold(),
)
PACKAGE_BUILD_BACKEND_REQUIREMENTS = tuple(
    f"{name}=={version}" for name, version in PACKAGE_EXPECTED_BACKEND_PACKAGES.items()
)
PACKAGE_SAFE_EXECUTION_ENVIRONMENT = {
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
}
PACKAGE_FIXED_ENVIRONMENT_OVERRIDES = {
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
PACKAGE_PATH_ENVIRONMENT_OVERRIDES = frozenset({"UV_CACHE_DIR"})
PACKAGE_PROBE_CODE_SHA256 = {
    "base_python_probe": "83d50fd5e46c76965ecb09772e1b4dcf5734cf1c949840013c3520eaee37427d",
    "build_backend_probe": "85ecd59fdbfbc714338273d7c3d23773896457b40d8a8a4d71d672914611a952",
    "installed_paths_probe": "3e3f5815d13826d284d5a4d8673ff95a0aefb1d92c964e0d3027bcc8bb6302e3",
}
DEPENDENCY_FAILURE_KEYS = frozenset(
    {
        "native_uv_sync_not_asserted_passed",
        "native_environment_installed_mismatch",
        "native_environment_invalid_artifact_evidence",
        "installed_environment_mismatch",
        "missing_artifact_evidence",
        "invalid_artifact_evidence",
        "hermetic_environment_incomplete_wheelhouse",
        "hermetic_environment_source_build_validation_missing",
        "strict_complete_wheelhouse_requirement_unmet",
    }
)
LOG_ROLES = frozenset(
    {
        "native_sync_log",
        "v2_evidence_tests",
        "recommended_core_tests",
        "mypy",
        "black",
        "diff_check",
    }
)
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
GATE_KINDS = {
    "native_sync_receipt": "artifact",
    "native_sync_log": "log",
    "v2_evidence_tests": "log",
    "recommended_core_tests": "log",
    "full_offline_suite": "log",
    "mypy": "log",
    "black": "log",
    "diff_check": "log",
    "package_parity": "artifact",
    "hash_freeze_readback": "artifact",
}
HASH_FREEZE_PATHS = (
    "pyproject.toml",
    "uv.lock",
    "quant_investor/v17_v2_contract/validators.py",
    "quant_investor/v17_v2_contract/resources.py",
    "quant_investor/v17_v2_contract/package_parity.py",
    "quant_investor/v17_v2_contract/schemas/generation_catalog.v1.schema.json",
    "quant_investor/v17_v2_contract/resources/package_manifest.v1.json",
    "quant_investor/v17_v2_contract/resources/main_suite_runtime_policy.v1.json",
    "quant_investor/v17_v2_contract/schemas/main_suite_runtime_policy.v1.schema.json",
    "scripts/v17_offline_dependency_evidence.py",
    "scripts/v17_phase0_diff_check.py",
    "scripts/v17_phase0_evidence_index.py",
    "scripts/v17_phase0_evidence_session.py",
    "scripts/v17_phase0_main_suite_harness.py",
    "scripts/v17_phase0_main_suite_wrapper.py",
    "scripts/v17_phase0_package_evidence.py",
    "scripts/v17_phase0_skip_baseline.py",
    "scripts/schemas/v17_offline_dependency_evidence.v2.schema.json",
    "scripts/schemas/v17_phase0_command_receipt.v2.schema.json",
    "scripts/schemas/v17_phase0_evidence_index.v2.schema.json",
    "scripts/schemas/v17_phase0_gate_manifest.v2.schema.json",
    "scripts/schemas/v17_phase0_hash_freeze.v2.schema.json",
    "scripts/schemas/v17_phase0_main_suite_receipt.v1.schema.json",
    "scripts/schemas/v17_phase0_package_evidence.v2.schema.json",
    "scripts/schemas/v17_phase0_pre_existing_classification.v2.schema.json",
    "scripts/schemas/v17_phase0_session.v2.schema.json",
    "scripts/schemas/v17_phase0_skip_baseline.v2.schema.json",
    "scripts/schemas/v17_phase0_unpublished_failure.v2.schema.json",
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
        "quant_investor/v17_v2_contract/schemas/main_suite_runtime_policy.v1.schema.json",
        MAIN_SUITE_RUNTIME_POLICY_SCHEMA_ID,
    ),
    (
        MAIN_SUITE_RECEIPT_VERSION,
        "scripts/schemas/v17_phase0_main_suite_receipt.v1.schema.json",
        MAIN_SUITE_RECEIPT_SCHEMA_ID,
    ),
    (
        DEPENDENCY_RECEIPT_VERSION,
        "scripts/schemas/v17_offline_dependency_evidence.v2.schema.json",
        DEPENDENCY_RECEIPT_SCHEMA_ID,
    ),
    (
        SKIP_BASELINE_VERSION,
        "scripts/schemas/v17_phase0_skip_baseline.v2.schema.json",
        SKIP_BASELINE_SCHEMA_ID,
    ),
    (
        PACKAGE_EVIDENCE_VERSION,
        "scripts/schemas/v17_phase0_package_evidence.v2.schema.json",
        PACKAGE_EVIDENCE_SCHEMA_ID,
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
SCHEMA_BY_ARTIFACT = {
    artifact_version: (path, schema_id) for artifact_version, path, schema_id in SCHEMA_REGISTRY
}


class Phase0EvidenceError(RuntimeError):
    """Raised for ambiguous, unsafe, drifting, or noncanonical evidence."""

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
        raise Phase0EvidenceError("value is not canonical JSON") from exc


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
        raise Phase0EvidenceError("semantic_sha256 must not be supplied")
    sealed = dict(payload)
    sealed[SEMANTIC_FIELD] = _semantic_sha256(sealed)
    return sealed


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_regular_file(
    path: Path,
    *,
    require_private: bool,
    max_bytes: int = MAX_EXTERNAL_BYTES,
) -> tuple[bytes, os.stat_result]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise Phase0EvidenceError(f"cannot stat input: {path}") from exc
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise Phase0EvidenceError(f"input must be a regular non-symlink file: {path}")
    if before.st_size > max_bytes:
        raise Phase0EvidenceError(f"input exceeds byte limit: {path}")
    if require_private and (
        before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) != 0o600
        or before.st_nlink != 1
    ):
        raise Phase0EvidenceError(f"external input must be owner-private 0600: {path}")
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        descriptor_stat = os.fstat(descriptor)
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            if len(chunk) > max_bytes - total:
                raise Phase0EvidenceError(f"input exceeds byte limit while reading: {path}")
            chunks.append(chunk)
            total += len(chunk)
        after_descriptor = os.fstat(descriptor)
    except OSError as exc:
        raise Phase0EvidenceError(f"cannot read input: {path}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as exc:
        raise Phase0EvidenceError(f"input disappeared during read: {path}") from exc
    signature = _stat_signature(before)
    if (
        signature != _stat_signature(descriptor_stat)
        or signature != _stat_signature(after_descriptor)
        or signature != _stat_signature(after)
    ):
        raise Phase0EvidenceError(f"input changed during stable read: {path}")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise Phase0EvidenceError(f"input size changed during stable read: {path}")
    return raw, before


def _private_parent(path: Path, *, label: str) -> Path:
    absolute = path.absolute()
    parent = absolute.parent
    try:
        parent_lstat = parent.lstat()
        resolved = parent.resolve(strict=True)
    except OSError as exc:
        raise Phase0EvidenceError(f"{label} parent is unavailable: {parent}") from exc
    if (
        resolved != parent
        or stat.S_ISLNK(parent_lstat.st_mode)
        or not stat.S_ISDIR(parent_lstat.st_mode)
        or stat.S_IMODE(parent_lstat.st_mode) != 0o700
        or parent_lstat.st_uid != os.getuid()
    ):
        raise Phase0EvidenceError(f"{label} parent must be owner-private 0700")
    return resolved


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.absolute().relative_to(root.absolute())
    except ValueError:
        return False
    return True


def _external_file_binding(
    path: Path,
    *,
    repo_root: Path,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    parent = _private_parent(path, label=label)
    parent_before = parent.lstat()
    canonical_path = parent / path.name
    if _path_within(canonical_path, repo_root):
        raise Phase0EvidenceError(f"{label} must be outside the repository")
    raw, observed = _stable_regular_file(canonical_path, require_private=True)
    try:
        parent_after = parent.lstat()
        resolved_after = parent.resolve(strict=True)
    except OSError as exc:
        raise Phase0EvidenceError(f"{label} parent changed during stable read") from exc
    parent_identity_before = (
        parent_before.st_dev,
        parent_before.st_ino,
        parent_before.st_mode,
        parent_before.st_uid,
    )
    parent_identity_after = (
        parent_after.st_dev,
        parent_after.st_ino,
        parent_after.st_mode,
        parent_after.st_uid,
    )
    if parent_identity_before != parent_identity_after or resolved_after != parent:
        raise Phase0EvidenceError(f"{label} parent changed during stable read")
    return (
        {
            "mode": f"{stat.S_IMODE(observed.st_mode):04o}",
            "path": str(canonical_path),
            "sha256": _sha256(raw),
            "size_bytes": len(raw),
        },
        raw,
    )


def _require_exact_keys(
    value: Any,
    expected: set[str],
    *,
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise Phase0EvidenceError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise Phase0EvidenceError(
            f"{label} shape mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _require_bool(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise Phase0EvidenceError(f"{label} must be a boolean")
    return value


def _require_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise Phase0EvidenceError(f"{label} must be an integer >= {minimum}")
    return value


def _require_string(value: Any, *, label: str) -> str:
    if type(value) is not str or not value:
        raise Phase0EvidenceError(f"{label} must be a nonempty string")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise Phase0EvidenceError(f"{label} must be a lowercase SHA-256")
    return value


def _require_absolute_path(value: Any, *, label: str) -> str:
    path = _require_string(value, label=label)
    if not Path(path).is_absolute():
        raise Phase0EvidenceError(f"{label} must be an absolute path")
    return path


def _require_string_array(
    value: Any,
    *,
    label: str,
    nonempty: bool = False,
) -> list[str]:
    if (
        type(value) is not list
        or (nonempty and not value)
        or not all(type(item) is str and item for item in value)
    ):
        qualifier = "nonempty " if nonempty else ""
        raise Phase0EvidenceError(f"{label} must be a {qualifier}string array")
    return value


def _casefold_ascii(value: str) -> str:
    return "".join(
        chr(ord(character) + 32) if "A" <= character <= "Z" else character for character in value
    )


def _require_unique_casefold(values: Sequence[str], *, label: str) -> None:
    exact: set[str] = set()
    folded: dict[str, str] = {}
    for value in values:
        if value in exact:
            raise Phase0EvidenceError(f"{label} contains duplicate value: {value}")
        exact.add(value)
        key = _casefold_ascii(value)
        if key in folded:
            raise Phase0EvidenceError(
                f"{label} contains ASCII-casefold collision: {folded[key]} and {value}"
            )
        folded[key] = value


def _repo_relative_path(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise Phase0EvidenceError(f"{label} must be a canonical repository path")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise Phase0EvidenceError(f"{label} is not strict UTF-8") from exc
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or str(pure) != value
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.parts[0] == ".git"
    ):
        raise Phase0EvidenceError(f"{label} must be a canonical repository path")
    for root in FORBIDDEN_V16_ROOTS:
        if pure == root or root in pure.parents:
            raise Phase0EvidenceError(f"{label} enters forbidden v16 results: {value}")
    return value


def _require_token(value: Any, *, label: str) -> str:
    if type(value) is not str or TOKEN_RE.fullmatch(value) is None:
        raise Phase0EvidenceError(f"{label} must be a lowercase ASCII token")
    return value


def _validate_base_commit(value: Any) -> str:
    if type(value) is not str or COMMIT_RE.fullmatch(value) is None:
        raise Phase0EvidenceError("base commit must be a full lowercase object ID")
    return value


def _git_bytes(command: Sequence[str], *, repo_root: Path) -> bytes:
    try:
        completed = subprocess.run(
            list(command),
            cwd=repo_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={
                **os.environ,
                "GIT_OPTIONAL_LOCKS": "0",
                "LC_ALL": "C",
            },
        )
    except OSError as exc:
        raise Phase0EvidenceError(f"cannot execute {command[0]}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise Phase0EvidenceError(
            f"command failed ({completed.returncode}): {' '.join(command)}: {detail}"
        )
    if len(completed.stdout) > MAX_GIT_CAPTURE_BYTES:
        raise Phase0EvidenceError(f"git capture exceeds byte limit: {' '.join(command)}")
    return completed.stdout


def _resolve_repo(repo_root: Path, base_commit: str) -> tuple[Path, str]:
    try:
        root = repo_root.resolve(strict=True)
    except OSError as exc:
        raise Phase0EvidenceError(f"repository root unavailable: {repo_root}") from exc
    top_raw = _git_bytes(("git", "rev-parse", "--show-toplevel"), repo_root=root)
    try:
        top = Path(top_raw.decode("utf-8", errors="strict").strip()).resolve(strict=True)
    except (UnicodeError, OSError) as exc:
        raise Phase0EvidenceError("git top level is invalid") from exc
    if top != root:
        raise Phase0EvidenceError("explicit repository root is not the git top level")
    resolved_raw = _git_bytes(
        ("git", "rev-parse", "--verify", f"{base_commit}^{{commit}}"),
        repo_root=root,
    )
    try:
        resolved = resolved_raw.decode("ascii", errors="strict").strip()
    except UnicodeError as exc:
        raise Phase0EvidenceError("resolved base commit is not ASCII") from exc
    if resolved != base_commit:
        raise Phase0EvidenceError("base commit is abbreviated, ambiguous, or not canonical")
    return root, resolved


def _decode_nul_paths(raw: bytes, *, label: str) -> list[str]:
    if raw and not raw.endswith(b"\0"):
        raise Phase0EvidenceError(f"{label} is not NUL terminated")
    paths: list[str] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            value = item.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise Phase0EvidenceError(f"{label} contains a non-UTF-8 path") from exc
        paths.append(_repo_relative_path(value, label=label))
    _require_unique_casefold(paths, label=label)
    return paths


def _safe_repo_entry(repo_root: Path, relative: str) -> Path:
    pure = PurePosixPath(_repo_relative_path(relative, label="untracked path"))
    candidate = repo_root.joinpath(*pure.parts)
    try:
        resolved_parent = candidate.parent.resolve(strict=True)
        resolved_parent.relative_to(repo_root)
    except (OSError, ValueError) as exc:
        raise Phase0EvidenceError(f"untracked path escapes repository: {relative}") from exc
    return candidate


def _stable_untracked(repo_root: Path, relative: str) -> dict[str, Any]:
    path = _safe_repo_entry(repo_root, relative)
    try:
        before = path.lstat()
    except OSError as exc:
        raise Phase0EvidenceError(f"untracked path disappeared: {relative}") from exc
    base = {
        "mode": f"{stat.S_IMODE(before.st_mode):04o}",
        "path": relative,
        "size_bytes": before.st_size,
    }
    if stat.S_ISREG(before.st_mode):
        raw, observed = _stable_regular_file(
            path,
            require_private=False,
            max_bytes=MAX_EXTERNAL_BYTES,
        )
        if _stat_signature(before) != _stat_signature(observed):
            raise Phase0EvidenceError(f"untracked file identity drift: {relative}")
        return {
            **base,
            "sha256": _sha256(raw),
            "symlink_target": None,
            "type": "file",
        }
    if stat.S_ISLNK(before.st_mode):
        try:
            target_before = os.readlink(path)
            after = path.lstat()
            target_after = os.readlink(path)
        except OSError as exc:
            raise Phase0EvidenceError(f"cannot read untracked symlink: {relative}") from exc
        if _stat_signature(before) != _stat_signature(after) or target_before != target_after:
            raise Phase0EvidenceError(f"untracked symlink changed during read: {relative}")
        try:
            target_bytes = target_before.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise Phase0EvidenceError(
                f"untracked symlink target is not strict UTF-8: {relative}"
            ) from exc
        return {
            **base,
            "sha256": _sha256(target_bytes),
            "symlink_target": target_before,
            "type": "symlink",
        }
    raise Phase0EvidenceError(f"unsupported untracked path type: {relative}")


def _raw_binding(raw: bytes) -> dict[str, Any]:
    return {
        "bytes_base64": base64.b64encode(raw).decode("ascii"),
        "encoding": "base64",
        "sha256": _sha256(raw),
        "size_bytes": len(raw),
    }


def _git_snapshot(repo_root: Path, base_commit: str) -> dict[str, Any]:
    porcelain = _git_bytes(
        ("git", "status", "--porcelain=v1", "-z", "--untracked-files=all"),
        repo_root=repo_root,
    )
    binary_diff = _git_bytes(
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
        repo_root=repo_root,
    )
    tracked_names_raw = _git_bytes(
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
        repo_root=repo_root,
    )
    untracked_names_raw = _git_bytes(
        ("git", "ls-files", "--others", "--exclude-standard", "-z"),
        repo_root=repo_root,
    )
    tracked = _decode_nul_paths(tracked_names_raw, label="tracked dirty paths")
    untracked_paths = _decode_nul_paths(
        untracked_names_raw,
        label="untracked inventory",
    )
    dirty_paths = sorted(
        set(tracked).union(untracked_paths),
        key=lambda value: value.encode("utf-8"),
    )
    _require_unique_casefold(dirty_paths, label="dirty path inventory")
    untracked = [
        _stable_untracked(repo_root, path)
        for path in sorted(untracked_paths, key=lambda value: value.encode("utf-8"))
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
            "binary_diff": binary_diff,
            "porcelain": porcelain,
            "tracked_names": tracked_names_raw,
            "untracked_names": untracked_names_raw,
        },
    }


def _public_source_state(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in snapshot.items() if key != "_guards"}


def _assert_snapshot_equal(first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    if _canonical_bytes(_public_source_state(first)) != _canonical_bytes(
        _public_source_state(second)
    ) or first.get("_guards") != second.get("_guards"):
        raise Phase0EvidenceError("repository source state changed during collection")


def _glob_regex(pattern: str) -> re.Pattern[str]:
    fragments = ["^"]
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == "*":
            if index + 1 < len(pattern) and pattern[index + 1] == "*":
                fragments.append(".*")
                index += 2
                continue
            fragments.append("[^/]*")
        elif character == "?":
            fragments.append("[^/]")
        else:
            fragments.append(re.escape(character))
        index += 1
    fragments.append("$")
    return re.compile("".join(fragments), re.ASCII)


def _allowed_patterns(raw_patterns: Sequence[str]) -> list[str]:
    if not raw_patterns:
        raise Phase0EvidenceError("at least one explicit Phase 0 path pattern is required")
    patterns: list[str] = []
    for value in raw_patterns:
        if value not in PHASE0_ALLOWED_PATTERN_REGISTRY:
            raise Phase0EvidenceError(f"unknown or over-broad Phase 0 path pattern: {value}")
        patterns.append(value)
    _require_unique_casefold(patterns, label="Phase 0 path patterns")
    return sorted(patterns, key=lambda value: value.encode("utf-8"))


def _parse_classification_manifest(
    raw: bytes,
    *,
    base_commit: str,
) -> list[str]:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise Phase0EvidenceError("classification manifest BOM is forbidden")
    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except Phase0EvidenceError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise Phase0EvidenceError("classification manifest is invalid JSON") from exc
    if raw != _canonical_resource_bytes(payload):
        raise Phase0EvidenceError("classification manifest is not canonical JSON plus newline")
    manifest = _require_exact_keys(
        payload,
        {
            "base_commit",
            "entries",
            "protocol_version",
            "provenance",
            "semantic_sha256",
            "version",
        },
        label="classification manifest",
    )
    if (
        manifest["protocol_version"] != PROTOCOL_VERSION
        or manifest["version"] != CLASSIFICATION_VERSION
        or manifest["base_commit"] != base_commit
        or manifest["provenance"] != CLASSIFICATION_PROVENANCE
    ):
        raise Phase0EvidenceError("classification manifest identity mismatch")
    declared = manifest["semantic_sha256"]
    if type(declared) is not str or SHA256_RE.fullmatch(declared) is None:
        raise Phase0EvidenceError("classification manifest semantic SHA-256 is invalid")
    if declared != _semantic_sha256(manifest):
        raise Phase0EvidenceError("classification manifest semantic SHA-256 mismatch")
    entries = manifest["entries"]
    if type(entries) is not list:
        raise Phase0EvidenceError("classification entries must be an array")
    paths: list[str] = []
    for index, raw_entry in enumerate(entries):
        entry = _require_exact_keys(
            raw_entry,
            {"classification", "path"},
            label=f"classification entries[{index}]",
        )
        if entry["classification"] != PRE_EXISTING_CLASSIFICATION:
            raise Phase0EvidenceError("classification must be PRE_EXISTING_NON_PHASE0")
        paths.append(
            _repo_relative_path(
                entry["path"],
                label=f"classification entries[{index}].path",
            )
        )
    _require_unique_casefold(paths, label="pre-existing classification paths")
    canonical_order = sorted(paths, key=lambda value: value.encode("utf-8"))
    if paths != canonical_order:
        raise Phase0EvidenceError("classification entries are not canonically ordered")
    return paths


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Phase0EvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(token: str) -> None:
    raise Phase0EvidenceError(f"non-finite JSON constant rejected: {token}")


def _load_canonical_json_resource(raw: bytes, *, label: str) -> dict[str, Any]:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise Phase0EvidenceError(f"{label} BOM is forbidden")
    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except Phase0EvidenceError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise Phase0EvidenceError(f"{label} is invalid JSON") from exc
    if type(payload) is not dict:
        raise Phase0EvidenceError(f"{label} must be a JSON object")
    if raw != _canonical_resource_bytes(payload):
        raise Phase0EvidenceError(f"{label} is not canonical JSON plus newline")
    return payload


def _source_binding_from_state(source_state: Mapping[str, Any]) -> dict[str, str]:
    binary = _require_exact_keys(
        source_state.get("binary_diff_from_base"),
        {"bytes_base64", "encoding", "sha256", "size_bytes"},
        label="source binding binary diff",
    )
    porcelain = _require_exact_keys(
        source_state.get("porcelain_v1_z"),
        {"bytes_base64", "encoding", "sha256", "size_bytes"},
        label="source binding porcelain",
    )
    untracked = source_state.get("untracked")
    if type(untracked) is not list:
        raise Phase0EvidenceError("source binding untracked inventory must be an array")
    result = {
        "base_commit": str(source_state.get("base_commit")),
        "binary_diff_sha256": str(binary.get("sha256")),
        "porcelain_sha256": str(porcelain.get("sha256")),
        "source_state_sha256": str(source_state.get("source_state_sha256")),
        "untracked_inventory_sha256": _sha256(_canonical_bytes(untracked)),
    }
    _validate_source_binding(result, label="source binding")
    return result


def _normalize_receipt_source_state(
    value: Any,
    *,
    current_source: Mapping[str, Any],
    repo_root: Path,
) -> None:
    index_porcelain = _require_exact_keys(
        current_source.get("porcelain_v1_z"),
        {"bytes_base64", "encoding", "sha256", "size_bytes"},
        label="index porcelain_v1_z",
    )
    index_diff = _require_exact_keys(
        current_source.get("binary_diff_from_base"),
        {"bytes_base64", "encoding", "sha256", "size_bytes"},
        label="index binary_diff_from_base",
    )
    expected_untracked = []
    for index, raw_entry in enumerate(current_source.get("untracked", [])):
        entry = _require_exact_keys(
            raw_entry,
            {"mode", "path", "sha256", "size_bytes", "symlink_target", "type"},
            label=f"index untracked[{index}]",
        )
        if entry["type"] == "file":
            expected_untracked.append(
                {
                    "mode": entry["mode"],
                    "path": entry["path"],
                    "sha256": entry["sha256"],
                    "size": entry["size_bytes"],
                    "type": "file",
                }
            )
        elif entry["type"] == "symlink":
            expected_untracked.append(
                {
                    "mode": entry["mode"],
                    "path": entry["path"],
                    "size": entry["size_bytes"],
                    "target": entry["symlink_target"],
                    "type": "symlink",
                }
            )
        else:
            raise Phase0EvidenceError("index untracked entry type is invalid")
    dependency_head_raw = _git_bytes(
        ("git", "rev-parse", "--verify", "HEAD"),
        repo_root=repo_root,
    )
    dependency_diff = _git_bytes(
        (
            "git",
            "diff",
            "--binary",
            "--no-ext-diff",
            "--no-textconv",
            "HEAD",
        ),
        repo_root=repo_root,
    )
    try:
        dependency_head = dependency_head_raw.decode("ascii", errors="strict").strip()
    except UnicodeError as exc:
        raise Phase0EvidenceError("native receipt source HEAD is not ASCII") from exc
    if dependency_head != current_source.get("base_commit"):
        raise Phase0EvidenceError("native receipt source HEAD does not match index base commit")
    expected_public_receipt_source = {
        "binary_diff_from_head": {
            "sha256": _sha256(dependency_diff),
            "size": len(dependency_diff),
        },
        "head": current_source.get("base_commit"),
        "porcelain_v1_z": {
            "sha256": index_porcelain["sha256"],
            "size": index_porcelain["size_bytes"],
        },
        "untracked": expected_untracked,
    }
    expected_receipt_source = {
        **expected_public_receipt_source,
        "source_state_sha256": _sha256(_canonical_bytes(expected_public_receipt_source)),
    }
    receipt_source = _require_exact_keys(
        value,
        {
            "binary_diff_from_head",
            "head",
            "porcelain_v1_z",
            "source_state_sha256",
            "untracked",
        },
        label="native receipt source_state",
    )
    if receipt_source != expected_receipt_source:
        raise Phase0EvidenceError("native receipt source_state does not match index source")


def _pytest_summary_counts(raw: bytes, *, label: str) -> dict[str, int]:
    text = _decode_text(raw, label=label)
    summary_line = ""
    for line in reversed([line.strip() for line in text.splitlines()]):
        if " in " in line and re.search(
            r"\b(passed|failed|error|errors|skipped|xfailed|xpassed)\b", line
        ):
            summary_line = line
            break
    if not summary_line:
        raise Phase0EvidenceError(f"{label} raw evidence has no final pytest summary")
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
        r"(?<![\w.])([0-9]+)\s+(passed|failed|errors?|skipped|xfailed|xpassed)\b",
        summary_line,
    ):
        key = aliases[word]
        counts[key] += int(amount)
    return counts


def _pytest_skip_entries(raw: bytes, *, label: str) -> list[dict[str, Any]]:
    text = _decode_text(raw, label=label)
    entries: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = re.fullmatch(r"SKIPPED \[([0-9]+)\] ([^:]+):([0-9]+): (.+)", line.strip())
        if match is None:
            continue
        path = match.group(2)
        pure = PurePosixPath(path)
        if (
            pure.is_absolute()
            or pure.as_posix() != path
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise Phase0EvidenceError(f"{label} skip path is not safe repository-relative")
        entries.append(
            {
                "count": int(match.group(1)),
                "line": int(match.group(3)),
                "path": path,
                "reason": match.group(4),
            }
        )
    keys = [(entry["path"], entry["line"], entry["reason"], entry["count"]) for entry in entries]
    if len(keys) != len(set(keys)):
        raise Phase0EvidenceError(f"{label} skip rows contain duplicates")
    return sorted(
        entries,
        key=lambda entry: (
            entry["path"].encode("utf-8"),
            entry["line"],
            entry["reason"].encode("utf-8"),
            entry["count"],
        ),
    )


def _parse_command_receipt_log(
    raw: bytes,
    *,
    role: str,
    repo_root: Path,
    current_source_binding: Mapping[str, str],
) -> tuple[dict[str, Any], bytes]:
    prefix = b"MYQUANT_PHASE0_COMMAND_RECEIPT="
    first_line, separator, remainder = raw.partition(b"\n")
    if not separator or not first_line.startswith(prefix):
        raise Phase0EvidenceError(f"{role} log missing command receipt envelope")
    receipt_raw = first_line[len(prefix) :]
    try:
        receipt = json.loads(
            receipt_raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except Phase0EvidenceError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise Phase0EvidenceError(f"{role} command receipt is invalid JSON") from exc
    if type(receipt) is not dict or receipt_raw != _canonical_bytes(receipt):
        raise Phase0EvidenceError(f"{role} command receipt is not canonical compact JSON")
    parsed = _require_exact_keys(
        receipt,
        {
            "commands",
            "cwd",
            "output_sha256",
            "output_size_bytes",
            "role",
            "semantic_sha256",
            "source_binding",
            "version",
        },
        label=f"{role} command receipt",
    )
    if (
        parsed["version"] != COMMAND_RECEIPT_VERSION
        or parsed["role"] != role
        or parsed["cwd"] != str(repo_root)
        or parsed["source_binding"] != dict(current_source_binding)
        or parsed["output_sha256"] != _sha256(remainder)
        or type(parsed["output_size_bytes"]) is not int
        or parsed["output_size_bytes"] < 0
        or parsed["output_size_bytes"] != len(remainder)
        or parsed["semantic_sha256"] != _semantic_sha256(parsed)
    ):
        raise Phase0EvidenceError(f"{role} command receipt binding mismatch")
    commands = parsed["commands"]
    if type(commands) is not list or not commands:
        raise Phase0EvidenceError(f"{role} command receipt commands must be nonempty")
    for index, raw_command in enumerate(commands):
        command = _require_exact_keys(
            raw_command,
            {"argv", "env", "exit_code", "tool_version"},
            label=f"{role} command receipt commands[{index}]",
        )
        argv = command["argv"]
        env = command["env"]
        if (
            type(argv) is not list
            or not argv
            or not all(type(item) is str and item for item in argv)
            or type(env) is not dict
            or not all(type(key) is str and type(value) is str for key, value in env.items())
            or type(command["exit_code"]) is not int
            or command["exit_code"] != 0
            or type(command["tool_version"]) is not str
            or not command["tool_version"]
        ):
            raise Phase0EvidenceError(f"{role} command receipt command is invalid")
        _validate_tool_version_shape(role, index, command["tool_version"])
        executable = argv[0]
        if (
            executable != "git"
            and not Path(executable).is_absolute()
            and not executable.startswith("scripts/")
        ):
            raise Phase0EvidenceError(f"{role} command executable is not absolute or repo script")
    return dict(parsed), remainder


def _validate_tool_version_shape(role: str, index: int, version: str) -> None:
    patterns = {
        "native_sync_log": (r"^uv \S+(?: .*)?$",),
        "v2_evidence_tests": (r"^pytest \S+(?: .*)?$",),
        "recommended_core_tests": (r"^(?:bash \S+(?: .*)?|pytest \S+(?: .*)?)$",),
        "full_offline_suite": (r"^pytest \S+(?: .*)?$",),
        "mypy": (r"^mypy \S+(?: .*)?$",),
        "black": (r"^black \S+(?: .*)?$",),
        "diff_check": (r"^git version \S+(?: .*)?$",),
    }
    if not any(re.fullmatch(pattern, version) for pattern in patterns.get(role, ())):
        raise Phase0EvidenceError(
            f"{role} command receipt commands[{index}] tool_version is invalid"
        )


def _tool_versions_from_receipt(role: str, receipt: Mapping[str, Any]) -> dict[str, str]:
    commands = receipt["commands"]
    if role == "native_sync_log":
        return {"uv": commands[0]["tool_version"]}
    if role == "v2_evidence_tests":
        return {"pytest": commands[0]["tool_version"]}
    if role == "recommended_core_tests":
        return {
            "staged_upgrade_quality_gate": commands[0]["tool_version"],
            "pytest": commands[1]["tool_version"],
        }
    if role == "full_offline_suite":
        return {"pytest": commands[0]["tool_version"]}
    if role == "mypy":
        return {"mypy": commands[0]["tool_version"]}
    if role == "black":
        return {"black": commands[0]["tool_version"]}
    if role == "diff_check":
        return {"git": commands[0]["tool_version"]}
    return {}


def _validate_dependency_digest_binding(value: Any, *, label: str) -> dict[str, Any]:
    binding = _require_exact_keys(value, {"sha256", "size"}, label=label)
    _require_sha256(binding["sha256"], label=f"{label}.sha256")
    _require_int(binding["size"], label=f"{label}.size")
    return binding


def _validate_dependency_file_binding(value: Any, *, label: str) -> dict[str, Any]:
    binding = _require_exact_keys(
        value,
        {"mode", "path", "sha256", "size"},
        label=label,
    )
    _require_absolute_path(binding["path"], label=f"{label}.path")
    _require_sha256(binding["sha256"], label=f"{label}.sha256")
    _require_int(binding["size"], label=f"{label}.size")
    if type(binding["mode"]) is not str or re.fullmatch(r"[0-7]{4}", binding["mode"]) is None:
        raise Phase0EvidenceError(f"{label}.mode is invalid")
    return binding


def _validate_dependency_source_state_shape(value: Any) -> dict[str, Any]:
    source = _require_exact_keys(
        value,
        {
            "binary_diff_from_head",
            "head",
            "porcelain_v1_z",
            "source_state_sha256",
            "untracked",
        },
        label="native_sync_receipt source_state",
    )
    _validate_base_commit(source["head"])
    _validate_dependency_digest_binding(
        source["binary_diff_from_head"],
        label="native_sync_receipt source_state.binary_diff_from_head",
    )
    _validate_dependency_digest_binding(
        source["porcelain_v1_z"],
        label="native_sync_receipt source_state.porcelain_v1_z",
    )
    untracked = source["untracked"]
    if type(untracked) is not list:
        raise Phase0EvidenceError("native_sync_receipt source_state.untracked must be an array")
    paths: list[str] = []
    for index, raw_entry in enumerate(untracked):
        label = f"native_sync_receipt source_state.untracked[{index}]"
        if type(raw_entry) is not dict:
            raise Phase0EvidenceError(f"{label} must be an object")
        entry_type = raw_entry.get("type")
        expected = (
            {"mode", "path", "sha256", "size", "type"}
            if entry_type == "file"
            else {"mode", "path", "size", "target", "type"} if entry_type == "symlink" else set()
        )
        if not expected:
            raise Phase0EvidenceError(f"{label}.type is invalid")
        entry = _require_exact_keys(raw_entry, expected, label=label)
        path = _repo_relative_path(entry["path"], label=f"{label}.path")
        paths.append(path)
        if type(entry["mode"]) is not str or re.fullmatch(r"[0-7]{4}", entry["mode"]) is None:
            raise Phase0EvidenceError(f"{label}.mode is invalid")
        _require_int(entry["size"], label=f"{label}.size")
        if entry_type == "file":
            _require_sha256(entry["sha256"], label=f"{label}.sha256")
        else:
            _require_string(entry["target"], label=f"{label}.target")
    _require_unique_casefold(paths, label="native_sync_receipt source_state untracked paths")
    if paths != sorted(paths, key=lambda item: item.encode("utf-8")):
        raise Phase0EvidenceError("native_sync_receipt source_state.untracked is not sorted")
    _require_sha256(
        source["source_state_sha256"],
        label="native_sync_receipt source_state.source_state_sha256",
    )
    unsealed = dict(source)
    declared = unsealed.pop("source_state_sha256")
    if declared != _sha256(_canonical_bytes(unsealed)):
        raise Phase0EvidenceError("native_sync_receipt source_state semantic SHA-256 mismatch")
    return source


def _validate_dependency_inputs_shape(value: Any) -> dict[str, Any]:
    inputs = _require_exact_keys(
        value,
        {
            "frozen_no_hash_export",
            "pip_http_cache",
            "pyproject",
            "uv_cache",
            "uv_lock",
            "wheelhouse",
        },
        label="native_sync_receipt inputs",
    )
    for key in ("frozen_no_hash_export", "pyproject", "uv_lock"):
        _validate_dependency_file_binding(
            inputs[key],
            label=f"native_sync_receipt inputs.{key}",
        )
    wheelhouse = _require_exact_keys(
        inputs["wheelhouse"],
        {"mode", "owner_private", "path", "st_dev", "st_ino", "st_uid"},
        label="native_sync_receipt inputs.wheelhouse",
    )
    _require_absolute_path(wheelhouse["path"], label="native_sync_receipt inputs.wheelhouse.path")
    if wheelhouse["mode"] != "0700" or wheelhouse["owner_private"] is not True:
        raise Phase0EvidenceError("native_sync_receipt wheelhouse is not owner-private 0700")
    for key in ("st_dev", "st_ino", "st_uid"):
        _require_int(wheelhouse[key], label=f"native_sync_receipt inputs.wheelhouse.{key}")
    uv_cache = _require_exact_keys(
        inputs["uv_cache"],
        {"path"},
        label="native_sync_receipt inputs.uv_cache",
    )
    _require_absolute_path(uv_cache["path"], label="native_sync_receipt inputs.uv_cache.path")
    if inputs["pip_http_cache"] is not None:
        pip_cache = _require_exact_keys(
            inputs["pip_http_cache"],
            {"path"},
            label="native_sync_receipt inputs.pip_http_cache",
        )
        _require_absolute_path(
            pip_cache["path"],
            label="native_sync_receipt inputs.pip_http_cache.path",
        )
    return inputs


def _validate_dependency_runtime_shape(value: Any) -> dict[str, Any]:
    runtime = _require_exact_keys(
        value,
        {
            "marker_environment",
            "platform",
            "python",
            "supported_tag_count",
            "supported_tags_sha256",
            "target_venv",
            "uv",
        },
        label="native_sync_receipt runtime",
    )
    uv = _require_exact_keys(
        runtime["uv"],
        {"mode", "path", "sha256", "size", "version"},
        label="native_sync_receipt runtime.uv",
    )
    uv_path_text = _require_absolute_path(
        uv["path"],
        label="native_sync_receipt runtime.uv.path",
    )
    _require_sha256(uv["sha256"], label="native_sync_receipt runtime.uv.sha256")
    _require_int(uv["size"], label="native_sync_receipt runtime.uv.size", minimum=1)
    _require_string(uv["version"], label="native_sync_receipt runtime.uv.version")
    if (
        type(uv["mode"]) is not str
        or re.fullmatch(r"[0-7]{4}", uv["mode"]) is None
        or not uv["version"].startswith("uv ")
    ):
        raise Phase0EvidenceError("native_sync_receipt runtime.uv identity is invalid")
    uv_path = Path(uv_path_text)
    try:
        resolved_uv_path = uv_path.resolve(strict=True)
    except OSError as exc:
        raise Phase0EvidenceError("native_sync_receipt uv executable is unavailable") from exc
    if resolved_uv_path != uv_path:
        raise Phase0EvidenceError("native_sync_receipt uv path is not resolved")
    uv_raw, uv_stat = _stable_regular_file(uv_path, require_private=False)
    if (
        not os.access(uv_path, os.X_OK)
        or uv["sha256"] != _sha256(uv_raw)
        or uv["size"] != len(uv_raw)
        or uv["mode"] != f"{stat.S_IMODE(uv_stat.st_mode):04o}"
    ):
        raise Phase0EvidenceError("native_sync_receipt uv executable binding mismatch")

    target_venv = _require_exact_keys(
        runtime["target_venv"],
        {
            "path",
            "python_entrypoint",
            "python_entrypoint_symlink_target",
            "pyvenv_cfg",
        },
        label="native_sync_receipt runtime.target_venv",
    )
    _require_absolute_path(
        target_venv["path"],
        label="native_sync_receipt runtime.target_venv.path",
    )
    _require_absolute_path(
        target_venv["python_entrypoint"],
        label="native_sync_receipt runtime.target_venv.python_entrypoint",
    )
    if (
        target_venv["python_entrypoint_symlink_target"] is not None
        and type(target_venv["python_entrypoint_symlink_target"]) is not str
    ):
        raise Phase0EvidenceError(
            "native_sync_receipt runtime.target_venv.python_entrypoint_symlink_target "
            "must be a string or null"
        )
    _validate_dependency_file_binding(
        target_venv["pyvenv_cfg"],
        label="native_sync_receipt runtime.target_venv.pyvenv_cfg",
    )
    if Path(target_venv["pyvenv_cfg"]["path"]) != Path(target_venv["path"]) / "pyvenv.cfg":
        raise Phase0EvidenceError("native_sync_receipt target pyvenv.cfg binding mismatch")

    python = _require_exact_keys(
        runtime["python"],
        {
            "base_prefix",
            "executable",
            "implementation",
            "prefix",
            "version",
            "version_info",
        },
        label="native_sync_receipt runtime.python",
    )
    for key in ("base_prefix", "executable", "prefix"):
        _require_absolute_path(
            python[key],
            label=f"native_sync_receipt runtime.python.{key}",
        )
    _require_string(
        python["implementation"], label="native_sync_receipt runtime.python.implementation"
    )
    _require_string(python["version"], label="native_sync_receipt runtime.python.version")
    version_info = python["version_info"]
    if type(version_info) is not list or len(version_info) != 3:
        raise Phase0EvidenceError(
            "native_sync_receipt python version_info must be a list of exactly three integers"
        )
    for index, item in enumerate(version_info):
        _require_int(
            item,
            label=f"native_sync_receipt runtime.python.version_info[{index}]",
        )

    platform = _require_exact_keys(
        runtime["platform"],
        {"machine", "platform", "release", "system"},
        label="native_sync_receipt runtime.platform",
    )
    for key in platform:
        _require_string(platform[key], label=f"native_sync_receipt runtime.platform.{key}")
    marker_environment = runtime["marker_environment"]
    if (
        type(marker_environment) is not dict
        or not marker_environment
        or not all(
            type(key) is str and key and type(item) is str
            for key, item in marker_environment.items()
        )
    ):
        raise Phase0EvidenceError("native_sync_receipt runtime.marker_environment is invalid")
    _require_sha256(
        runtime["supported_tags_sha256"],
        label="native_sync_receipt runtime.supported_tags_sha256",
    )
    _require_int(
        runtime["supported_tag_count"],
        label="native_sync_receipt runtime.supported_tag_count",
        minimum=1,
    )
    return runtime


def _validate_dependency_requirement_list(value: Any, *, label: str) -> list[dict[str, Any]]:
    if type(value) is not list:
        raise Phase0EvidenceError(f"{label} must be an array")
    records: list[dict[str, Any]] = []
    for index, raw_record in enumerate(value):
        record = _require_exact_keys(
            raw_record,
            {"marker", "name", "version"},
            label=f"{label}[{index}]",
        )
        _require_token(record["name"], label=f"{label}[{index}].name")
        _require_string(record["version"], label=f"{label}[{index}].version")
        if type(record["marker"]) is not str:
            raise Phase0EvidenceError(f"{label}[{index}].marker must be a string")
        records.append(record)
    return records


def _validate_dependency_project_identity(value: Any, *, label: str) -> dict[str, Any]:
    identity = _require_exact_keys(value, {"name", "version"}, label=label)
    _require_token(identity["name"], label=f"{label}.name")
    _require_string(identity["version"], label=f"{label}.version")
    return identity


def _validate_dependency_reconciliation(
    value: Any,
    *,
    expected_dependencies: Sequence[Mapping[str, Any]],
    project_identity: Mapping[str, Any],
) -> dict[str, Any]:
    reconciliation = _require_exact_keys(
        value,
        {
            "exact_match",
            "expected_count",
            "extra",
            "installed",
            "installed_count",
            "local_project_identity_only",
            "missing",
            "third_party_expected_count",
            "version_mismatch",
        },
        label="native_sync_receipt installed_reconciliation",
    )
    for key in ("expected_count", "installed_count", "third_party_expected_count"):
        _require_int(
            reconciliation[key],
            label=f"native_sync_receipt installed_reconciliation.{key}",
        )
    _require_bool(
        reconciliation["exact_match"],
        label="native_sync_receipt installed_reconciliation.exact_match",
    )
    local = _require_exact_keys(
        reconciliation["local_project_identity_only"],
        {"artifact_provenance_verified", "name", "version"},
        label="native_sync_receipt installed_reconciliation.local_project_identity_only",
    )
    if (
        local["name"] != project_identity["name"]
        or local["version"] != project_identity["version"]
        or local["artifact_provenance_verified"] is not False
    ):
        raise Phase0EvidenceError("native_sync_receipt local project reconciliation mismatch")

    installed = reconciliation["installed"]
    if type(installed) is not list:
        raise Phase0EvidenceError(
            "native_sync_receipt installed_reconciliation.installed must be an array"
        )
    installed_by_name: dict[str, str] = {}
    for index, raw_record in enumerate(installed):
        label = f"native_sync_receipt installed_reconciliation.installed[{index}]"
        record = _require_exact_keys(
            raw_record,
            {"display_name", "metadata_sha256", "metadata_size", "name", "version"},
            label=label,
        )
        name = _require_token(record["name"], label=f"{label}.name")
        _require_string(record["display_name"], label=f"{label}.display_name")
        version = _require_string(record["version"], label=f"{label}.version")
        _require_sha256(record["metadata_sha256"], label=f"{label}.metadata_sha256")
        _require_int(record["metadata_size"], label=f"{label}.metadata_size")
        if name in installed_by_name:
            raise Phase0EvidenceError(
                f"native_sync_receipt duplicate installed distribution: {name}"
            )
        installed_by_name[name] = version
    if [record["name"] for record in installed] != sorted(installed_by_name):
        raise Phase0EvidenceError("native_sync_receipt installed inventory is not sorted")

    list_shapes = {
        "missing": {"expected_version", "name"},
        "extra": {"installed_version", "name"},
        "version_mismatch": {"expected_version", "installed_version", "name"},
    }
    for key, expected_keys in list_shapes.items():
        records = reconciliation[key]
        if type(records) is not list:
            raise Phase0EvidenceError(
                f"native_sync_receipt installed_reconciliation.{key} must be an array"
            )
        for index, raw_record in enumerate(records):
            label = f"native_sync_receipt installed_reconciliation.{key}[{index}]"
            record = _require_exact_keys(raw_record, expected_keys, label=label)
            _require_token(record["name"], label=f"{label}.name")
            for version_key in expected_keys - {"name"}:
                _require_string(record[version_key], label=f"{label}.{version_key}")

    expected_versions = {
        str(record["name"]): str(record["version"]) for record in expected_dependencies
    }
    expected_versions[str(project_identity["name"])] = str(project_identity["version"])
    if (
        reconciliation["third_party_expected_count"] != len(expected_dependencies)
        or reconciliation["expected_count"] != len(expected_versions)
        or reconciliation["installed_count"] != len(installed)
        or reconciliation["exact_match"] is not True
        or reconciliation["missing"]
        or reconciliation["extra"]
        or reconciliation["version_mismatch"]
        or installed_by_name != expected_versions
    ):
        raise Phase0EvidenceError("native_sync_receipt installed reconciliation mismatch")
    return reconciliation


def _validate_dependency_selected_artifact(value: Any, *, label: str) -> dict[str, Any]:
    selected = _require_exact_keys(
        value,
        {"filename", "kind", "requires_source_build", "sha256", "size", "tags", "url"},
        label=label,
    )
    if selected["kind"] not in {"wheel", "sdist"}:
        raise Phase0EvidenceError(f"{label}.kind is invalid")
    for key in ("filename", "url"):
        _require_string(selected[key], label=f"{label}.{key}")
    _require_sha256(selected["sha256"], label=f"{label}.sha256")
    _require_int(selected["size"], label=f"{label}.size", minimum=1)
    tags = _require_string_array(selected["tags"], label=f"{label}.tags")
    _require_bool(selected["requires_source_build"], label=f"{label}.requires_source_build")
    if selected["requires_source_build"] is not (selected["kind"] == "sdist") or (
        selected["kind"] == "sdist" and tags
    ):
        raise Phase0EvidenceError(f"{label} source-build fields mismatch")
    return selected


def _validate_dependency_raw_evidence_fields(
    evidence: Mapping[str, Any],
    *,
    label: str,
) -> None:
    _require_absolute_path(evidence["path"], label=f"{label}.path")
    _require_sha256(evidence["sha256"], label=f"{label}.sha256")
    _require_int(evidence["size"], label=f"{label}.size")
    if type(evidence["mode"]) is not str or re.fullmatch(r"[0-7]{4}", evidence["mode"]) is None:
        raise Phase0EvidenceError(f"{label}.mode is invalid")
    if evidence["raw_artifact_retained"] is not True or evidence["valid"] is not True:
        raise Phase0EvidenceError(f"{label} is not valid retained raw evidence")
    if evidence["errors"] != []:
        raise Phase0EvidenceError(f"{label}.errors must be empty")


def _validate_dependency_artifact_record(value: Any, *, index: int) -> dict[str, Any]:
    label = f"native_sync_receipt artifact_evidence.records[{index}]"
    record = _require_exact_keys(
        value,
        {
            "evidence",
            "name",
            "raw_lock_artifact_retained",
            "requires_source_build",
            "selected_locked_artifact",
            "version",
        },
        label=label,
    )
    _require_token(record["name"], label=f"{label}.name")
    _require_string(record["version"], label=f"{label}.version")
    selected = _validate_dependency_selected_artifact(
        record["selected_locked_artifact"],
        label=f"{label}.selected_locked_artifact",
    )
    _require_bool(record["requires_source_build"], label=f"{label}.requires_source_build")
    _require_bool(
        record["raw_lock_artifact_retained"],
        label=f"{label}.raw_lock_artifact_retained",
    )
    if record["requires_source_build"] is not selected["requires_source_build"]:
        raise Phase0EvidenceError(f"{label} source-build flag mismatch")

    raw_evidence = record["evidence"]
    if type(raw_evidence) is not dict:
        raise Phase0EvidenceError(f"{label}.evidence must be an object")
    source = raw_evidence.get("source")
    raw_keys = {
        "errors",
        "mode",
        "path",
        "raw_artifact_retained",
        "sha256",
        "size",
        "source",
        "valid",
    }
    if source == "wheelhouse_raw_artifact":
        evidence = _require_exact_keys(raw_evidence, raw_keys, label=f"{label}.evidence")
        _validate_dependency_raw_evidence_fields(evidence, label=f"{label}.evidence")
    elif source == "pip_http_cache_raw_artifact":
        evidence = _require_exact_keys(
            raw_evidence,
            raw_keys | {"matching_body_count"},
            label=f"{label}.evidence",
        )
        _validate_dependency_raw_evidence_fields(evidence, label=f"{label}.evidence")
        _require_int(
            evidence["matching_body_count"],
            label=f"{label}.evidence.matching_body_count",
            minimum=1,
        )
    elif source == "wheelhouse_materialized_from_pip_http_cache":
        evidence = _require_exact_keys(
            raw_evidence,
            raw_keys | {"materialization", "pip_http_cache_source"},
            label=f"{label}.evidence",
        )
        _validate_dependency_raw_evidence_fields(evidence, label=f"{label}.evidence")
        pip_source = _require_exact_keys(
            evidence["pip_http_cache_source"],
            {"matching_body_count", "path", "sha256", "size"},
            label=f"{label}.evidence.pip_http_cache_source",
        )
        _require_absolute_path(
            pip_source["path"],
            label=f"{label}.evidence.pip_http_cache_source.path",
        )
        _require_sha256(
            pip_source["sha256"],
            label=f"{label}.evidence.pip_http_cache_source.sha256",
        )
        _require_int(
            pip_source["size"],
            label=f"{label}.evidence.pip_http_cache_source.size",
        )
        _require_int(
            pip_source["matching_body_count"],
            label=f"{label}.evidence.pip_http_cache_source.matching_body_count",
            minimum=1,
        )
        materialization = _require_exact_keys(
            evidence["materialization"],
            {"created", "destination", "mode", "sha256", "size"},
            label=f"{label}.evidence.materialization",
        )
        _require_bool(
            materialization["created"],
            label=f"{label}.evidence.materialization.created",
        )
        _require_absolute_path(
            materialization["destination"],
            label=f"{label}.evidence.materialization.destination",
        )
        _require_sha256(
            materialization["sha256"],
            label=f"{label}.evidence.materialization.sha256",
        )
        _require_int(
            materialization["size"],
            label=f"{label}.evidence.materialization.size",
        )
        if materialization["mode"] != "0600":
            raise Phase0EvidenceError(f"{label}.evidence.materialization.mode is invalid")
    elif source == "uv_extracted_cache":
        evidence = _require_exact_keys(
            raw_evidence,
            {
                "archive_directory_identity",
                "errors",
                "metadata",
                "raw_artifact_retained",
                "source",
                "valid",
                "wheel_metadata",
            },
            label=f"{label}.evidence",
        )
        if (
            evidence["raw_artifact_retained"] is not False
            or evidence["valid"] is not True
            or evidence["errors"] != []
        ):
            raise Phase0EvidenceError(f"{label}.evidence uv cache flags are invalid")
        _validate_dependency_file_binding(
            evidence["metadata"],
            label=f"{label}.evidence.metadata",
        )
        wheel_metadata = _require_exact_keys(
            evidence["wheel_metadata"],
            {"mode", "path", "sha256", "size", "tags"},
            label=f"{label}.evidence.wheel_metadata",
        )
        _validate_dependency_file_binding(
            {key: wheel_metadata[key] for key in ("mode", "path", "sha256", "size")},
            label=f"{label}.evidence.wheel_metadata",
        )
        _require_string_array(
            wheel_metadata["tags"],
            label=f"{label}.evidence.wheel_metadata.tags",
            nonempty=True,
        )
        archive = _require_exact_keys(
            evidence["archive_directory_identity"],
            {"mode", "mtime_ns", "name", "path", "st_dev", "st_ino"},
            label=f"{label}.evidence.archive_directory_identity",
        )
        _require_absolute_path(
            archive["path"],
            label=f"{label}.evidence.archive_directory_identity.path",
        )
        _require_string(
            archive["name"],
            label=f"{label}.evidence.archive_directory_identity.name",
        )
        if type(archive["mode"]) is not str or re.fullmatch(r"[0-7]{4}", archive["mode"]) is None:
            raise Phase0EvidenceError(
                f"{label}.evidence.archive_directory_identity.mode is invalid"
            )
        for key in ("mtime_ns", "st_dev", "st_ino"):
            _require_int(
                archive[key],
                label=f"{label}.evidence.archive_directory_identity.{key}",
            )
    else:
        raise Phase0EvidenceError(f"{label}.evidence.source is invalid")
    if record["raw_lock_artifact_retained"] is not bool(raw_evidence["raw_artifact_retained"]):
        raise Phase0EvidenceError(f"{label} raw artifact retention mismatch")
    return record


def _validate_dependency_artifacts(
    value: Any,
    *,
    expected_dependencies: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    artifacts = _require_exact_keys(
        value,
        {
            "complete_raw_artifact_count",
            "complete_raw_artifacts",
            "complete_wheelhouse",
            "evidenced_artifact_count",
            "expected_artifact_count",
            "invalid",
            "missing",
            "pip_http_cache_scan",
            "records",
            "source_build_validation",
            "wheelhouse_raw_artifact_count",
        },
        label="native_sync_receipt artifact_evidence",
    )
    for key in (
        "complete_raw_artifact_count",
        "evidenced_artifact_count",
        "expected_artifact_count",
        "wheelhouse_raw_artifact_count",
    ):
        _require_int(
            artifacts[key],
            label=f"native_sync_receipt artifact_evidence.{key}",
        )
    for key in ("complete_raw_artifacts", "complete_wheelhouse"):
        _require_bool(
            artifacts[key],
            label=f"native_sync_receipt artifact_evidence.{key}",
        )
    pip_scan = _require_exact_keys(
        artifacts["pip_http_cache_scan"],
        {"body_file_count", "exact_match_count", "provided", "size_candidate_count"},
        label="native_sync_receipt artifact_evidence.pip_http_cache_scan",
    )
    _require_bool(
        pip_scan["provided"],
        label="native_sync_receipt artifact_evidence.pip_http_cache_scan.provided",
    )
    for key in ("body_file_count", "exact_match_count", "size_candidate_count"):
        _require_int(
            pip_scan[key],
            label=f"native_sync_receipt artifact_evidence.pip_http_cache_scan.{key}",
        )

    source_build = _require_exact_keys(
        artifacts["source_build_validation"],
        {"missing", "status"},
        label="native_sync_receipt artifact_evidence.source_build_validation",
    )
    if source_build["status"] not in {"MISSING", "NOT_REQUIRED"}:
        raise Phase0EvidenceError(
            "native_sync_receipt artifact_evidence.source_build_validation.status is invalid"
        )
    if type(source_build["missing"]) is not list:
        raise Phase0EvidenceError(
            "native_sync_receipt artifact_evidence.source_build_validation.missing "
            "must be an array"
        )
    for index, raw_record in enumerate(source_build["missing"]):
        label = "native_sync_receipt artifact_evidence.source_build_validation." f"missing[{index}]"
        record = _require_exact_keys(
            raw_record,
            {"name", "reason", "selected_filename", "version"},
            label=label,
        )
        _require_token(record["name"], label=f"{label}.name")
        for key in ("reason", "selected_filename", "version"):
            _require_string(record[key], label=f"{label}.{key}")
    if (source_build["status"] == "MISSING") is not bool(source_build["missing"]):
        raise Phase0EvidenceError("native_sync_receipt source-build validation status mismatch")

    records = artifacts["records"]
    if type(records) is not list:
        raise Phase0EvidenceError("native_sync_receipt artifact_evidence.records must be an array")
    validated_records = [
        _validate_dependency_artifact_record(record, index=index)
        for index, record in enumerate(records)
    ]
    record_names = [record["name"] for record in validated_records]
    if record_names != sorted(record_names) or len(record_names) != len(set(record_names)):
        raise Phase0EvidenceError("native_sync_receipt artifact records are not unique and sorted")

    missing = artifacts["missing"]
    if type(missing) is not list:
        raise Phase0EvidenceError("native_sync_receipt artifact_evidence.missing must be an array")
    for index, raw_record in enumerate(missing):
        label = f"native_sync_receipt artifact_evidence.missing[{index}]"
        record = _require_exact_keys(
            raw_record,
            {"name", "reason", "selected_filename", "version"},
            label=label,
        )
        _require_token(record["name"], label=f"{label}.name")
        for key in ("reason", "selected_filename", "version"):
            _require_string(record[key], label=f"{label}.{key}")
    invalid = artifacts["invalid"]
    if type(invalid) is not list:
        raise Phase0EvidenceError("native_sync_receipt artifact_evidence.invalid must be an array")
    for index, raw_record in enumerate(invalid):
        label = f"native_sync_receipt artifact_evidence.invalid[{index}]"
        record = _require_exact_keys(
            raw_record,
            {"detail", "name", "reason", "version"},
            label=label,
        )
        _require_token(record["name"], label=f"{label}.name")
        _require_string(record["reason"], label=f"{label}.reason")
        _require_string(record["version"], label=f"{label}.version")
        _canonical_bytes(record["detail"])

    complete_raw_count = sum(
        1 for record in validated_records if record["raw_lock_artifact_retained"] is True
    )
    wheelhouse_count = sum(
        1
        for record in validated_records
        if record["evidence"]["source"]
        in {"wheelhouse_raw_artifact", "wheelhouse_materialized_from_pip_http_cache"}
    )
    if (
        artifacts["expected_artifact_count"] != len(expected_dependencies)
        or artifacts["evidenced_artifact_count"] != len(validated_records)
        or artifacts["complete_raw_artifact_count"] != complete_raw_count
        or artifacts["wheelhouse_raw_artifact_count"] != wheelhouse_count
        or artifacts["complete_raw_artifacts"]
        is not (complete_raw_count == len(expected_dependencies))
        or artifacts["complete_wheelhouse"] is not (wheelhouse_count == len(expected_dependencies))
    ):
        raise Phase0EvidenceError("native_sync_receipt artifact counts mismatch")
    return artifacts


def _validate_native_dependency_receipt(
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    receipt = _require_exact_keys(
        payload,
        {
            "accepted",
            "artifact_evidence",
            "complete_raw_artifact_count",
            "complete_wheelhouse",
            "dependency_environment_accepted",
            "expected_dependencies",
            "expected_raw_artifact_count",
            "failure_reasons",
            "hermetic_dependency_environment_accepted",
            "hermetic_environment_verified",
            "inactive_marker_dependencies",
            "inputs",
            "installed_reconciliation",
            "invalid",
            "local_project_identity_for_environment_reconciliation",
            "materialize_wheelhouse",
            "missing",
            "native_dependency_environment_accepted",
            "native_uv_sync_status",
            "network_actions_performed",
            "offline_only",
            "repackaged_artifacts",
            "require_complete_wheelhouse",
            "runtime",
            "schema_version",
            "scope",
            "semantic_sha256",
            "source_build_validation_complete",
            "source_state",
            "status",
            "wheelhouse_raw_artifact_count",
        },
        label="native_sync_receipt",
    )
    if receipt["schema_version"] != DEPENDENCY_RECEIPT_VERSION:
        raise Phase0EvidenceError("native_sync_receipt schema_version mismatch")
    _require_string(receipt["status"], label="native_sync_receipt status")
    for key in (
        "accepted",
        "complete_wheelhouse",
        "dependency_environment_accepted",
        "hermetic_dependency_environment_accepted",
        "hermetic_environment_verified",
        "materialize_wheelhouse",
        "native_dependency_environment_accepted",
        "network_actions_performed",
        "offline_only",
        "repackaged_artifacts",
        "require_complete_wheelhouse",
        "source_build_validation_complete",
    ):
        _require_bool(receipt[key], label=f"native_sync_receipt {key}")
    for key in (
        "complete_raw_artifact_count",
        "expected_raw_artifact_count",
        "wheelhouse_raw_artifact_count",
    ):
        _require_int(receipt[key], label=f"native_sync_receipt {key}")
    _require_sha256(
        receipt["semantic_sha256"],
        label="native_sync_receipt semantic_sha256",
    )
    if receipt["semantic_sha256"] != _semantic_sha256(receipt):
        raise Phase0EvidenceError("native_sync_receipt semantic SHA-256 mismatch")

    scope = _require_exact_keys(
        receipt["scope"],
        {
            "kind",
            "local_project_artifact_provenance_verified",
            "local_project_identity_checked_for_environment_exactness",
            "local_project_wheel_sdist_source_state_release_binding",
            "release_readiness_proven_by_this_report",
        },
        label="native_sync_receipt scope",
    )
    if scope != {
        "kind": "third_party_dependency_environment_only",
        "local_project_artifact_provenance_verified": False,
        "local_project_identity_checked_for_environment_exactness": True,
        "local_project_wheel_sdist_source_state_release_binding": (
            "OUTSIDE_SCOPE_SEPARATE_LOCAL_ARTIFACT_GATE_REQUIRED"
        ),
        "release_readiness_proven_by_this_report": False,
    }:
        raise Phase0EvidenceError("native_sync_receipt scope mismatch")
    _validate_dependency_source_state_shape(receipt["source_state"])
    inputs = _validate_dependency_inputs_shape(receipt["inputs"])
    runtime = _validate_dependency_runtime_shape(receipt["runtime"])

    native_status = _require_exact_keys(
        receipt["native_uv_sync_status"],
        {
            "environment_acceptance_also_requires",
            "independently_verified_by_this_tool",
            "operator_asserted_passed",
            "source",
            "status",
        },
        label="native_sync_receipt native_uv_sync_status",
    )
    if native_status != {
        "environment_acceptance_also_requires": [
            "installed_exact_match",
            "no_invalid_artifact_evidence",
        ],
        "independently_verified_by_this_tool": False,
        "operator_asserted_passed": True,
        "source": "explicit_cli_assertion",
        "status": "PASSED",
    }:
        raise Phase0EvidenceError("native_sync_receipt native uv sync status mismatch")

    project_identity = _validate_dependency_project_identity(
        receipt["local_project_identity_for_environment_reconciliation"],
        label="native_sync_receipt local_project_identity_for_environment_reconciliation",
    )
    expected_dependencies = _validate_dependency_requirement_list(
        receipt["expected_dependencies"],
        label="native_sync_receipt expected_dependencies",
    )
    inactive_dependencies = _validate_dependency_requirement_list(
        receipt["inactive_marker_dependencies"],
        label="native_sync_receipt inactive_marker_dependencies",
    )
    if expected_dependencies != sorted(expected_dependencies, key=lambda item: item["name"]):
        raise Phase0EvidenceError("native_sync_receipt expected_dependencies is not sorted")
    if inactive_dependencies != sorted(
        inactive_dependencies,
        key=lambda item: (item["name"], item["version"], item["marker"]),
    ):
        raise Phase0EvidenceError("native_sync_receipt inactive_marker_dependencies is not sorted")
    reconciliation = _validate_dependency_reconciliation(
        receipt["installed_reconciliation"],
        expected_dependencies=expected_dependencies,
        project_identity=project_identity,
    )
    artifacts = _validate_dependency_artifacts(
        receipt["artifact_evidence"],
        expected_dependencies=expected_dependencies,
    )
    if artifacts["pip_http_cache_scan"]["provided"] is not (
        inputs["pip_http_cache"] is not None
    ) or (receipt["materialize_wheelhouse"] and inputs["pip_http_cache"] is None):
        raise Phase0EvidenceError("native_sync_receipt pip cache/materialization mismatch")
    if receipt["missing"] != artifacts["missing"] or receipt["invalid"] != artifacts["invalid"]:
        raise Phase0EvidenceError("native_sync_receipt artifact failure projection mismatch")
    if artifacts["invalid"]:
        raise Phase0EvidenceError("native_sync_receipt contains invalid artifact evidence")
    if (
        receipt["expected_raw_artifact_count"] != artifacts["expected_artifact_count"]
        or receipt["complete_raw_artifact_count"] != artifacts["complete_raw_artifact_count"]
        or receipt["wheelhouse_raw_artifact_count"] != artifacts["wheelhouse_raw_artifact_count"]
        or receipt["complete_wheelhouse"] is not artifacts["complete_wheelhouse"]
    ):
        raise Phase0EvidenceError("native_sync_receipt top-level artifact counts mismatch")

    source_build_complete = artifacts["source_build_validation"]["status"] in {
        "NOT_REQUIRED",
        "VERIFIED",
    }
    hermetic_verified = bool(
        reconciliation["exact_match"] and not artifacts["missing"] and not artifacts["invalid"]
    )
    hermetic_accepted = bool(
        hermetic_verified and artifacts["complete_wheelhouse"] and source_build_complete
    )
    native_accepted = bool(reconciliation["exact_match"] and not artifacts["invalid"])
    dependency_accepted = bool(native_accepted or hermetic_accepted)
    accepted = bool(
        dependency_accepted
        and (artifacts["complete_wheelhouse"] if receipt["require_complete_wheelhouse"] else True)
    )
    if (
        receipt["accepted"] is not accepted
        or receipt["dependency_environment_accepted"] is not dependency_accepted
        or receipt["native_dependency_environment_accepted"] is not native_accepted
        or receipt["hermetic_environment_verified"] is not hermetic_verified
        or receipt["hermetic_dependency_environment_accepted"] is not hermetic_accepted
        or receipt["source_build_validation_complete"] is not source_build_complete
        or receipt["status"] != NATIVE_ACCEPTED_STATUS
        or receipt["offline_only"] is not True
        or receipt["network_actions_performed"] is not False
        or receipt["repackaged_artifacts"] is not False
    ):
        raise Phase0EvidenceError("native_sync_receipt semantic gate failed")

    failures = _require_exact_keys(
        receipt["failure_reasons"],
        set(DEPENDENCY_FAILURE_KEYS),
        label="native_sync_receipt failure_reasons",
    )
    for key in DEPENDENCY_FAILURE_KEYS:
        _require_bool(failures[key], label=f"native_sync_receipt failure_reasons.{key}")
    expected_failures = {
        "native_uv_sync_not_asserted_passed": False,
        "native_environment_installed_mismatch": not reconciliation["exact_match"],
        "native_environment_invalid_artifact_evidence": bool(artifacts["invalid"]),
        "installed_environment_mismatch": not reconciliation["exact_match"],
        "missing_artifact_evidence": bool(artifacts["missing"]),
        "invalid_artifact_evidence": bool(artifacts["invalid"]),
        "hermetic_environment_incomplete_wheelhouse": not artifacts["complete_wheelhouse"],
        "hermetic_environment_source_build_validation_missing": not source_build_complete,
        "strict_complete_wheelhouse_requirement_unmet": bool(
            receipt["require_complete_wheelhouse"] and not artifacts["complete_wheelhouse"]
        ),
    }
    if failures != expected_failures:
        raise Phase0EvidenceError("native_sync_receipt failure reasons mismatch")
    if runtime["python"]["implementation"] != "cpython" or runtime["python"]["version_info"][
        :2
    ] != [3, 13]:
        raise Phase0EvidenceError("native_sync_receipt semantic gate failed")
    _fresh_python_binding_from_native_receipt(receipt, repo_root=repo_root)
    return receipt


def _validate_native_tool_bindings(
    native_receipt: Mapping[str, Any],
    native_log_receipt: Mapping[str, Any],
    tool_versions: Mapping[str, set[str]],
) -> None:
    runtime = native_receipt["runtime"]
    uv = runtime["uv"]
    uv_command = native_log_receipt["commands"][0]
    command_path = Path(uv_command["argv"][0])
    try:
        resolved_command_path = command_path.resolve(strict=True)
    except OSError as exc:
        raise Phase0EvidenceError("native_sync_log uv executable is unavailable") from exc
    if resolved_command_path != Path(uv["path"]) or uv_command["tool_version"] != uv["version"]:
        raise Phase0EvidenceError("native_sync_log uv receipt binding mismatch")
    if uv_command["env"]["UV_CACHE_DIR"] != native_receipt["inputs"]["uv_cache"]["path"]:
        raise Phase0EvidenceError("native_sync_log uv cache binding mismatch")
    if tool_versions.get("uv") != {uv["version"]}:
        raise Phase0EvidenceError("native_sync_log uv tool version mismatch")

    installed_versions = {
        record["name"]: record["version"]
        for record in native_receipt["installed_reconciliation"]["installed"]
    }
    python_version = ".".join(str(item) for item in runtime["python"]["version_info"])
    for tool in ("pytest", "mypy", "black"):
        installed_version = installed_versions.get(tool)
        if installed_version is None:
            raise Phase0EvidenceError(f"native_sync_receipt installed inventory omits {tool}")
        expected = f"{tool} {installed_version} python {python_version}"
        if tool_versions.get(tool) != {expected}:
            raise Phase0EvidenceError(f"log receipts do not match installed {tool} version")


def _fresh_python_binding_from_native_receipt(
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[str, str]:
    runtime = payload.get("runtime")
    if type(runtime) is not dict:
        raise Phase0EvidenceError("native_sync_receipt runtime missing")
    target_venv = runtime.get("target_venv")
    python = runtime.get("python")
    if type(target_venv) is not dict or type(python) is not dict:
        raise Phase0EvidenceError("native_sync_receipt target runtime binding missing")
    target_venv_path = target_venv.get("path")
    target_python = target_venv.get("python_entrypoint")
    python_executable = python.get("executable")
    python_prefix = python.get("prefix")
    if not all(
        type(value) is str
        for value in (target_venv_path, target_python, python_executable, python_prefix)
    ):
        raise Phase0EvidenceError("native_sync_receipt target Python paths missing")
    venv_path = Path(target_venv_path)
    python_path = Path(target_python)
    executable_path = Path(python_executable)
    prefix_path = Path(python_prefix)
    if (
        not venv_path.is_absolute()
        or not python_path.is_absolute()
        or not executable_path.is_absolute()
        or not prefix_path.is_absolute()
        or _path_within(venv_path, repo_root)
        or _path_within(python_path, repo_root)
        or _path_within(executable_path, repo_root)
        or _path_within(prefix_path, repo_root)
        or python_path != venv_path / "bin" / "python"
        or executable_path != python_path
        or prefix_path != venv_path
    ):
        raise Phase0EvidenceError("native_sync_receipt fresh Python binding mismatch")
    _assert_absolute_executable_path(str(python_path), label="native receipt target Python")
    return str(venv_path), str(python_path)


def _python_executable_from_receipt(role: str, receipt: Mapping[str, Any]) -> str | None:
    commands = receipt["commands"]
    if role == "native_sync_log":
        return None
    if role == "recommended_core_tests":
        first_env = commands[0]["env"]
        return first_env["PYTHON"]
    if role in {"v2_evidence_tests", "full_offline_suite", "mypy", "black"}:
        return commands[0]["argv"][0]
    return None


def _native_sync_python_from_receipt(role: str, receipt: Mapping[str, Any]) -> str | None:
    if role != "native_sync_log":
        return None
    return receipt["commands"][0]["argv"][3]


def _same_resolved_path(left: str, right: str) -> bool:
    return Path(left).resolve(strict=False) == Path(right).resolve(strict=False)


def _assert_absolute_executable_path(value: str, *, label: str) -> None:
    if not Path(value).is_absolute():
        raise Phase0EvidenceError(f"{label} must be an absolute executable path")


def _validate_command_identity(
    role: str,
    receipt: Mapping[str, Any],
    *,
    repo_root: Path,
) -> None:
    commands = receipt["commands"]
    if role == "native_sync_log":
        if len(commands) != 1:
            raise Phase0EvidenceError("native_sync_log must bind exactly one command")
        command = commands[0]
        argv = command["argv"]
        if (
            len(argv) != 7
            or argv[1:3] != ["sync", "--python"]
            or argv[4:]
            != [
                "--locked",
                "--all-extras",
                "--offline",
            ]
        ):
            raise Phase0EvidenceError("native_sync_log command identity mismatch")
        _assert_absolute_executable_path(argv[3], label="native sync --python")
        env = command["env"]
        if set(env) != {"UV_CACHE_DIR", "UV_PROJECT_ENVIRONMENT", "UV_PYTHON_DOWNLOADS"}:
            raise Phase0EvidenceError("native_sync_log environment keys mismatch")
        if env["UV_PYTHON_DOWNLOADS"] != "never":
            raise Phase0EvidenceError("native_sync_log must disable Python downloads")
        project_environment = Path(env["UV_PROJECT_ENVIRONMENT"])
        if (
            not project_environment.is_absolute()
            or _path_within(project_environment, repo_root)
            or project_environment.parent == project_environment
        ):
            raise Phase0EvidenceError("native_sync_log fresh environment is invalid")
        return
    if role == "v2_evidence_tests":
        _validate_python_pytest_command(
            commands,
            expected_tests=V2_EVIDENCE_TESTS,
            label=role,
        )
        return
    if role == "recommended_core_tests":
        if len(commands) != 2:
            raise Phase0EvidenceError("recommended_core_tests must bind two commands")
        first, second = commands
        if first["argv"] != ["scripts/staged_upgrade_quality_gate.sh"] or set(first["env"]) != {
            "PYTHON"
        }:
            raise Phase0EvidenceError("recommended_core_tests staged command mismatch")
        _assert_absolute_executable_path(first["env"]["PYTHON"], label="recommended staged PYTHON")
        _validate_python_pytest_command(
            [second],
            expected_tests=RECOMMENDED_CORE_TESTS,
            label=role,
        )
        if second["argv"][0] != first["env"]["PYTHON"]:
            raise Phase0EvidenceError("recommended_core_tests Python executable mismatch")
        return
    if role == "full_offline_suite":
        _validate_exact_command(
            commands,
            argv_prefix=None,
            expected_argv=["<python>", "-m", "pytest", *FULL_PYTEST_OPTIONS],
            label=role,
        )
        _assert_absolute_executable_path(commands[0]["argv"][0], label=role)
        return
    if role == "mypy":
        _validate_exact_command(
            commands,
            argv_prefix=None,
            expected_argv=[
                "<python>",
                "-m",
                "mypy",
                *MYPY_TARGETS,
            ],
            label=role,
        )
        _assert_absolute_executable_path(commands[0]["argv"][0], label=role)
        return
    if role == "black":
        _validate_exact_command(
            commands,
            argv_prefix=None,
            expected_argv=["<python>", "-m", "black", "--check", *BLACK_TARGETS],
            label=role,
        )
        _assert_absolute_executable_path(commands[0]["argv"][0], label=role)
        return
    if role == "diff_check":
        _validate_exact_command(
            commands,
            argv_prefix=None,
            expected_argv=["git", "diff", "--check"],
            label=role,
        )
        return


def _validate_python_pytest_command(
    commands: Sequence[Mapping[str, Any]],
    *,
    expected_tests: Sequence[str],
    label: str,
) -> None:
    _validate_exact_command(
        commands,
        argv_prefix=None,
        expected_argv=["<python>", "-m", "pytest", *expected_tests, *PYTEST_OPTIONS],
        label=label,
    )
    _assert_absolute_executable_path(commands[0]["argv"][0], label=label)


def _validate_exact_command(
    commands: Sequence[Mapping[str, Any]],
    *,
    argv_prefix: Sequence[str] | None,
    expected_argv: Sequence[str],
    label: str,
) -> None:
    if len(commands) != 1:
        raise Phase0EvidenceError(f"{label} must bind exactly one command")
    command = commands[0]
    if command["env"] != {}:
        raise Phase0EvidenceError(f"{label} command environment must be empty")
    argv = list(command["argv"])
    expected = list(expected_argv)
    if expected and expected[0] == "<python>":
        expected[0] = argv[0] if argv else ""
    if argv_prefix is not None and argv[: len(argv_prefix)] != list(argv_prefix):
        raise Phase0EvidenceError(f"{label} command identity mismatch")
    if argv != expected:
        raise Phase0EvidenceError(f"{label} command identity mismatch")


def _validate_source_binding(value: Any, *, label: str) -> dict[str, str]:
    binding = _require_exact_keys(
        value,
        {
            "base_commit",
            "binary_diff_sha256",
            "porcelain_sha256",
            "source_state_sha256",
            "untracked_inventory_sha256",
        },
        label=label,
    )
    _validate_base_commit(binding["base_commit"])
    for key in (
        "binary_diff_sha256",
        "porcelain_sha256",
        "source_state_sha256",
        "untracked_inventory_sha256",
    ):
        if type(binding[key]) is not str or SHA256_RE.fullmatch(binding[key]) is None:
            raise Phase0EvidenceError(f"{label}.{key} must be a SHA-256")
    return {key: str(binding[key]) for key in sorted(binding)}


def _coerce_source_binding(value: Any, *, label: str) -> dict[str, str]:
    if type(value) is dict and set(value) == {
        "base_commit",
        "binary_diff_sha256",
        "porcelain_sha256",
        "source_state_sha256",
        "untracked_inventory_sha256",
    }:
        return _validate_source_binding(value, label=label)
    if type(value) is dict and "source_state_sha256" in value and "porcelain_v1_z" in value:
        return _source_binding_from_state(value)
    raise Phase0EvidenceError(f"{label} does not contain a recognizable source binding")


def _current_hash_freeze(repo_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in HASH_FREEZE_PATHS:
        path = _safe_repo_entry(repo_root, relative)
        raw, _observed = _stable_regular_file(path, require_private=False)
        hashes[relative] = _sha256(raw)
    return hashes


def _decode_text(raw: bytes, *, label: str) -> str:
    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise Phase0EvidenceError(f"{label} is not strict UTF-8") from exc


def _require_claims(value: Any, expected: set[str], *, label: str) -> dict[str, Any]:
    return _require_exact_keys(value, expected, label=label)


def _require_exit_zero_claims(claims: Mapping[str, Any], *, label: str) -> None:
    if type(claims.get("exit_code")) is not int or claims["exit_code"] != 0:
        raise Phase0EvidenceError(f"{label} did not bind exit_code=0")


def _package_canonical_path(value: Any, *, label: str) -> Path:
    raw = _require_absolute_path(value, label=label)
    path = Path(raw)
    try:
        resolved = path.resolve(strict=False)
    except OSError as exc:
        raise Phase0EvidenceError(f"{label} cannot be resolved") from exc
    if str(path) != raw or resolved != path:
        raise Phase0EvidenceError(f"{label} must be a canonical absolute path")
    return path


def _package_path_outside_repo(value: Any, *, label: str, repo_root: Path) -> Path:
    path = _package_canonical_path(value, label=label)
    if _path_within(path, repo_root):
        raise Phase0EvidenceError(f"{label} must be outside the repository")
    return path


def _validate_package_file_binding(
    value: Any,
    *,
    label: str,
    readback: bool,
) -> dict[str, Any]:
    binding = _require_exact_keys(
        value,
        {"path", "sha256", "size_bytes"},
        label=label,
    )
    path = _package_canonical_path(binding["path"], label=f"{label}.path")
    _require_sha256(binding["sha256"], label=f"{label}.sha256")
    _require_int(binding["size_bytes"], label=f"{label}.size_bytes")
    if readback:
        raw, _observed = _stable_regular_file(path, require_private=False)
        if binding["sha256"] != _sha256(raw) or binding["size_bytes"] != len(raw):
            raise Phase0EvidenceError(f"{label} does not match file readback")
    return binding


def _validate_package_bytes_binding(value: Any, *, label: str) -> bytes:
    binding = _require_exact_keys(
        value,
        {"bytes_base64", "encoding", "sha256", "size_bytes"},
        label=label,
    )
    if binding["encoding"] != "base64" or type(binding["bytes_base64"]) is not str:
        raise Phase0EvidenceError(f"{label} must use base64")
    try:
        raw = base64.b64decode(
            binding["bytes_base64"].encode("ascii", errors="strict"),
            validate=True,
        )
    except (binascii.Error, UnicodeError, ValueError) as exc:
        raise Phase0EvidenceError(f"{label} contains invalid base64") from exc
    if base64.b64encode(raw).decode("ascii") != binding["bytes_base64"]:
        raise Phase0EvidenceError(f"{label} base64 is not canonical")
    _require_sha256(binding["sha256"], label=f"{label}.sha256")
    _require_int(binding["size_bytes"], label=f"{label}.size_bytes")
    if binding["sha256"] != _sha256(raw) or binding["size_bytes"] != len(raw):
        raise Phase0EvidenceError(f"{label} byte binding mismatch")
    return raw


def _load_package_command_json(raw: bytes, *, label: str) -> dict[str, Any]:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise Phase0EvidenceError(f"{label} BOM is forbidden")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except Phase0EvidenceError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise Phase0EvidenceError(f"{label} is invalid JSON") from exc
    if type(value) is not dict or raw != _canonical_resource_bytes(value):
        raise Phase0EvidenceError(f"{label} must be canonical JSON plus newline")
    return value


def _validate_package_evidence_schema(
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
) -> None:
    try:
        from quant_investor.v17_v2_contract.schema_validation import (
            SchemaValidationError,
            preflight_packaged_schema,
            validate_instance_against_schema,
        )
    except ImportError as exc:
        raise Phase0EvidenceError("package_parity closed schema executor is unavailable") from exc
    schema_path = _safe_repo_entry(
        repo_root,
        "scripts/schemas/v17_phase0_package_evidence.v2.schema.json",
    )
    schema_raw, _observed = _stable_regular_file(
        schema_path,
        require_private=False,
    )
    if schema_raw.startswith(b"\xef\xbb\xbf"):
        raise Phase0EvidenceError("package_parity schema BOM is forbidden")
    try:
        schema = json.loads(
            schema_raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except Phase0EvidenceError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise Phase0EvidenceError("package_parity schema is invalid JSON") from exc
    properties = schema.get("properties") if type(schema) is dict else None
    if (
        type(schema) is not dict
        or type(properties) is not dict
        or schema.get("$id") != PACKAGE_EVIDENCE_SCHEMA_ID
        or properties.get("version") != {"const": PACKAGE_EVIDENCE_VERSION}
    ):
        raise Phase0EvidenceError("package_parity schema identity mismatch")
    try:
        preflight_packaged_schema(schema)
        validate_instance_against_schema(dict(payload), schema)
    except SchemaValidationError as exc:
        raise Phase0EvidenceError(f"package_parity schema validation failed: {exc}") from exc


def _validate_package_sha_map(
    value: Any,
    *,
    label: str,
    required_names: frozenset[str],
) -> dict[str, str]:
    if type(value) is not dict or not required_names.issubset(value):
        raise Phase0EvidenceError(f"{label} is missing required dist-info hashes")
    result: dict[str, str] = {}
    for raw_name, raw_sha256 in value.items():
        if type(raw_name) is not str or not raw_name or "\x00" in raw_name or "\\" in raw_name:
            raise Phase0EvidenceError(f"{label} contains an unsafe name")
        name = PurePosixPath(raw_name)
        if (
            name.is_absolute()
            or name.as_posix() != raw_name
            or any(part in {"", ".", ".."} for part in name.parts)
        ):
            raise Phase0EvidenceError(f"{label} contains an unsafe name")
        result[raw_name] = _require_sha256(
            raw_sha256,
            label=f"{label}.{raw_name}",
        )
    return result


def _validate_package_parity_result(
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    parity = _require_exact_keys(
        payload,
        {
            "accepted",
            "installed_provenance",
            "package_file_count",
            "package_inventory",
            "package_inventory_sha256",
            "sdist_sha256",
            "source_equals_sdist_equals_wheel_equals_installed",
            "wheel_provenance",
            "wheel_sha256",
        },
        label="package_parity parity result",
    )
    if (
        parity["accepted"] is not True
        or parity["source_equals_sdist_equals_wheel_equals_installed"] is not True
    ):
        raise Phase0EvidenceError("package_parity did not accept four-surface byte parity")
    file_count = _require_int(
        parity["package_file_count"],
        label="package_parity package_file_count",
        minimum=1,
    )
    package_inventory = _require_exact_keys(
        parity["package_inventory"],
        {"file_count", "sha256"},
        label="package_parity package_inventory",
    )
    if (
        _require_int(
            package_inventory["file_count"],
            label="package_parity package_inventory.file_count",
            minimum=1,
        )
        != file_count
    ):
        raise Phase0EvidenceError("package_parity package inventory count mismatch")
    inventory_sha256 = _require_sha256(
        parity["package_inventory_sha256"],
        label="package_parity package_inventory_sha256",
    )
    if (
        _require_sha256(
            package_inventory["sha256"],
            label="package_parity package_inventory.sha256",
        )
        != inventory_sha256
    ):
        raise Phase0EvidenceError("package_parity package inventory SHA mismatch")
    sdist_sha256 = _require_sha256(
        parity["sdist_sha256"],
        label="package_parity sdist_sha256",
    )
    wheel_sha256 = _require_sha256(
        parity["wheel_sha256"],
        label="package_parity wheel_sha256",
    )

    installed = _require_exact_keys(
        parity["installed_provenance"],
        {
            "direct_url",
            "dist_info_file_sha256s",
            "dist_info_path",
            "environment_root",
            "installed_package_root",
            "metadata",
            "non_editable_verified",
            "record",
            "site_packages_root",
        },
        label="package_parity installed_provenance",
    )
    metadata = _require_exact_keys(
        installed["metadata"],
        {"name", "version"},
        label="package_parity installed metadata",
    )
    if metadata != {"name": "quant-investor", "version": "17.0.0"}:
        raise Phase0EvidenceError("package_parity installed metadata mismatch")
    if installed["non_editable_verified"] is not True:
        raise Phase0EvidenceError("package_parity installed artifact is not non-editable")
    installed_hashes = _validate_package_sha_map(
        installed["dist_info_file_sha256s"],
        label="package_parity installed dist-info hashes",
        required_names=frozenset(
            {"METADATA", "WHEEL", "RECORD", "entry_points.txt", "direct_url.json"}
        ),
    )
    installed_record = _require_exact_keys(
        installed["record"],
        {
            "dist_info_file_count",
            "file_count",
            "package_file_count",
            "record_sha256",
        },
        label="package_parity installed RECORD",
    )
    installed_dist_info_count = _require_int(
        installed_record["dist_info_file_count"],
        label="package_parity installed RECORD dist_info_file_count",
        minimum=5,
    )
    installed_total_count = _require_int(
        installed_record["file_count"],
        label="package_parity installed RECORD file_count",
        minimum=1,
    )
    if (
        installed_record["package_file_count"] != file_count
        or installed_dist_info_count != len(installed_hashes)
        or installed_total_count != file_count + installed_dist_info_count + 2
    ):
        raise Phase0EvidenceError("package_parity installed RECORD count mismatch")
    _require_sha256(
        installed_record["record_sha256"],
        label="package_parity installed RECORD sha256",
    )
    if installed_hashes["RECORD"] != installed_record["record_sha256"]:
        raise Phase0EvidenceError("package_parity installed RECORD SHA mismatch")

    paths = {
        key: _package_path_outside_repo(
            installed[key],
            label=f"package_parity installed_provenance.{key}",
            repo_root=repo_root,
        )
        for key in (
            "environment_root",
            "site_packages_root",
            "installed_package_root",
            "dist_info_path",
        )
    }
    if (
        paths["installed_package_root"].parent != paths["site_packages_root"]
        or paths["dist_info_path"].parent != paths["site_packages_root"]
        or paths["installed_package_root"].name != "quant_investor"
        or paths["dist_info_path"].name != "quant_investor-17.0.0.dist-info"
        or not _path_within(paths["site_packages_root"], paths["environment_root"])
    ):
        raise Phase0EvidenceError("package_parity installed path hierarchy mismatch")

    direct_url = _require_exact_keys(
        installed["direct_url"],
        {"archive_info_sha256", "editable", "present", "sha256", "url"},
        label="package_parity installed direct_url",
    )
    if (
        direct_url["present"] is not True
        or direct_url["editable"] is not False
        or direct_url["archive_info_sha256"] != wheel_sha256
        or installed_hashes["direct_url.json"]
        != _require_sha256(
            direct_url["sha256"],
            label="package_parity installed direct_url.sha256",
        )
    ):
        raise Phase0EvidenceError("package_parity installed direct_url mismatch")
    _require_string(direct_url["url"], label="package_parity installed direct_url.url")

    wheel = _require_exact_keys(
        parity["wheel_provenance"],
        {
            "dist_info_file_sha256s",
            "dist_info_root",
            "metadata",
            "record",
        },
        label="package_parity wheel_provenance",
    )
    if wheel["dist_info_root"] != "quant_investor-17.0.0.dist-info":
        raise Phase0EvidenceError("package_parity wheel dist-info root mismatch")
    wheel_metadata = _require_exact_keys(
        wheel["metadata"],
        {"name", "version"},
        label="package_parity wheel metadata",
    )
    if wheel_metadata != {"name": "quant-investor", "version": "17.0.0"}:
        raise Phase0EvidenceError("package_parity wheel metadata mismatch")
    wheel_hashes = _validate_package_sha_map(
        wheel["dist_info_file_sha256s"],
        label="package_parity wheel dist-info hashes",
        required_names=frozenset({"METADATA", "WHEEL", "RECORD", "entry_points.txt"}),
    )
    wheel_record = _require_exact_keys(
        wheel["record"],
        {"file_count", "record_sha256"},
        label="package_parity wheel RECORD",
    )
    if _require_int(
        wheel_record["file_count"],
        label="package_parity wheel RECORD file_count",
        minimum=1,
    ) != file_count + len(wheel_hashes) or wheel_hashes["RECORD"] != _require_sha256(
        wheel_record["record_sha256"],
        label="package_parity wheel RECORD sha256",
    ):
        raise Phase0EvidenceError("package_parity wheel RECORD mismatch")
    for name in set(wheel_hashes) - {"RECORD"}:
        if installed_hashes.get(name) != wheel_hashes[name]:
            raise Phase0EvidenceError("package_parity immutable dist-info SHA mismatch")

    return dict(parity), paths


def _validate_package_environment_proof(
    value: Any,
    *,
    command_env: Mapping[str, str],
    label: str,
) -> None:
    proof = _require_exact_keys(
        value,
        {
            "base_environment",
            "effective_environment",
            "host_environment",
            "overrides",
        },
        label=label,
    )
    if proof["base_environment"] != PACKAGE_SAFE_EXECUTION_ENVIRONMENT:
        raise Phase0EvidenceError(f"{label} base environment mismatch")
    if proof["overrides"] != dict(command_env):
        raise Phase0EvidenceError(f"{label} overrides mismatch")
    if (
        type(command_env) is not dict
        or any(type(key) is not str or not key for key in command_env)
        or any(type(item) is not str for item in command_env.values())
        or set(command_env)
        - set(PACKAGE_FIXED_ENVIRONMENT_OVERRIDES)
        - PACKAGE_PATH_ENVIRONMENT_OVERRIDES
    ):
        raise Phase0EvidenceError(f"{label} command overrides are invalid")
    for key, expected in PACKAGE_FIXED_ENVIRONMENT_OVERRIDES.items():
        if key in command_env and command_env[key] != expected:
            raise Phase0EvidenceError(f"{label} fixed override mismatch")
    expected_effective = {**PACKAGE_SAFE_EXECUTION_ENVIRONMENT, **command_env}
    if proof["effective_environment"] != expected_effective:
        raise Phase0EvidenceError(f"{label} effective environment mismatch")
    host = _require_exact_keys(
        proof["host_environment"],
        {
            "inherited_value_count",
            "secret_values_recorded",
            "stripped_variable_name_count",
            "stripped_variable_names_sha256",
        },
        label=f"{label}.host_environment",
    )
    if (
        _require_int(
            host["inherited_value_count"],
            label=f"{label}.host_environment.inherited_value_count",
        )
        != 0
        or host["secret_values_recorded"] is not False
    ):
        raise Phase0EvidenceError(f"{label} inherited host state")
    _require_int(
        host["stripped_variable_name_count"],
        label=f"{label}.host_environment.stripped_variable_name_count",
    )
    _require_sha256(
        host["stripped_variable_names_sha256"],
        label=f"{label}.host_environment.stripped_variable_names_sha256",
    )


def _validate_package_commands(
    provenance: Mapping[str, Any],
    *,
    parity: Mapping[str, Any],
    installed_paths: Mapping[str, Path],
    environment: Mapping[str, Path],
    artifact_bindings: Mapping[str, Mapping[str, Any]],
    base_interpreter: Mapping[str, Any],
    pip_runtime: Mapping[str, Any],
    repo_root: Path,
) -> None:
    commands = provenance["commands"]
    if type(commands) is not list or len(commands) != len(PACKAGE_COMMAND_ROLES):
        raise Phase0EvidenceError("package_parity command count mismatch")
    if provenance["command_roles"] != list(PACKAGE_COMMAND_ROLES):
        raise Phase0EvidenceError("package_parity command role declaration mismatch")
    if type(provenance["command_count"]) is not int or provenance["command_count"] != len(
        PACKAGE_COMMAND_ROLES
    ):
        raise Phase0EvidenceError("package_parity command_count mismatch")

    base_python = str(base_interpreter["realpath"])
    uv_binary = str(environment["uv_binary"])
    uv_cache = str(environment["uv_cache"])
    build_venv = environment["build_venv"]
    install_venv = environment["install_venv"]
    build_python = str(build_venv / "bin" / "python")
    install_python = str(install_venv / "bin" / "python")
    sdist = str(artifact_bindings["sdist"]["path"])
    wheel = str(artifact_bindings["wheel"]["path"])
    python_tool_version = f"CPython {base_interpreter['version']}"
    uv_tool_version = PACKAGE_EXPECTED_UV_OUTPUT
    pip_tool_version = str(pip_runtime["output"])
    offline_env = {
        "UV_CACHE_DIR": uv_cache,
        "UV_NO_CONFIG": "1",
        "UV_OFFLINE": "1",
        "UV_PYTHON_DOWNLOADS": "never",
    }
    no_cache_offline_env = {**offline_env, "UV_NO_CACHE": "1"}
    pip_env = {
        "PIP_CONFIG_FILE": "/dev/null",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_CACHE_DIR": "1",
        "PIP_NO_INDEX": "1",
    }
    probe_code: dict[int, str] = {}
    for index, role in (
        (0, "base_python_probe"),
        (4, "build_backend_probe"),
        (11, "installed_paths_probe"),
    ):
        raw_command = commands[index]
        if (
            type(raw_command) is not dict
            or type(raw_command.get("argv")) is not list
            or len(raw_command["argv"]) != 4
            or type(raw_command["argv"][3]) is not str
            or _sha256(raw_command["argv"][3].encode("utf-8", errors="strict"))
            != PACKAGE_PROBE_CODE_SHA256[role]
        ):
            raise Phase0EvidenceError(f"package_parity {role} probe code mismatch")
        probe_code[index] = raw_command["argv"][3]

    expected_argv = [
        [base_python, "-I", "-c", probe_code[0]],
        [uv_binary, "--version"],
        [
            uv_binary,
            "venv",
            "--python",
            base_python,
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
        [
            uv_binary,
            "pip",
            "install",
            "--python",
            build_python,
            "--offline",
            "--no-python-downloads",
            "--no-config",
            "--color",
            "never",
            "--no-progress",
            *PACKAGE_BUILD_BACKEND_REQUIREMENTS,
        ],
        [build_python, "-I", "-c", probe_code[4]],
        [
            uv_binary,
            "build",
            "--sdist",
            "--python",
            build_python,
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
            str(environment["work_root"] / "sdist"),
            str(repo_root),
        ],
        [
            uv_binary,
            "build",
            "--wheel",
            "--python",
            build_python,
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
            str(environment["work_root"] / "wheel"),
            sdist,
        ],
        [
            uv_binary,
            "venv",
            "--python",
            base_python,
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
        [install_python, "-m", "ensurepip", "--upgrade"],
        [install_python, "-I", "-m", "pip", "--version"],
        [
            install_python,
            "-I",
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--no-compile",
            wheel,
        ],
        [install_python, "-I", "-c", probe_code[11]],
        [
            install_python,
            str(installed_paths["installed_package_root"] / "v17_v2_contract/package_parity.py"),
            "--source-package-root",
            str(repo_root / "quant_investor"),
            "--sdist",
            sdist,
            "--wheel",
            wheel,
            "--installed-package-root",
            str(installed_paths["installed_package_root"]),
            "--installed-dist-info",
            str(installed_paths["dist_info_path"]),
            "--installed-environment-root",
            str(install_venv),
            "--expected-name",
            "quant-investor",
            "--expected-version",
            "17.0.0",
        ],
    ]
    expected_env = [
        {},
        {"UV_NO_CONFIG": "1", "UV_OFFLINE": "1"},
        offline_env,
        offline_env,
        {},
        no_cache_offline_env,
        no_cache_offline_env,
        offline_env,
        pip_env,
        pip_env,
        {**pip_env, "PIP_NO_COMPILE": "1"},
        {},
        pip_env,
    ]
    expected_tool_versions = [
        "python probe",
        uv_tool_version,
        uv_tool_version,
        uv_tool_version,
        python_tool_version,
        uv_tool_version,
        uv_tool_version,
        uv_tool_version,
        python_tool_version,
        pip_tool_version,
        pip_tool_version,
        python_tool_version,
        python_tool_version,
    ]
    stdout_values: list[bytes] = []
    for index, raw_command in enumerate(commands):
        label = f"package_parity commands[{index}]"
        command = _require_exact_keys(
            raw_command,
            {
                "argv",
                "cwd",
                "env",
                "exit_code",
                "role",
                "sanitized_environment",
                "stderr",
                "stdout",
                "tool_version",
            },
            label=label,
        )
        if (
            command["role"] != PACKAGE_COMMAND_ROLES[index]
            or command["argv"] != expected_argv[index]
            or command["cwd"] != str(repo_root)
            or command["env"] != expected_env[index]
            or type(command["exit_code"]) is not int
            or command["exit_code"] != 0
            or command["tool_version"] != expected_tool_versions[index]
        ):
            raise Phase0EvidenceError(f"{label} identity mismatch")
        _validate_package_environment_proof(
            command["sanitized_environment"],
            command_env=command["env"],
            label=f"{label}.sanitized_environment",
        )
        stdout_values.append(
            _validate_package_bytes_binding(
                command["stdout"],
                label=f"{label}.stdout",
            )
        )
        _validate_package_bytes_binding(
            command["stderr"],
            label=f"{label}.stderr",
        )

    if _load_package_command_json(
        stdout_values[0],
        label="package_parity base Python probe output",
    ) != dict(base_interpreter):
        raise Phase0EvidenceError("package_parity base Python probe output mismatch")
    if stdout_values[1] != f"{PACKAGE_EXPECTED_UV_OUTPUT}\n".encode("ascii"):
        raise Phase0EvidenceError("package_parity uv version output mismatch")
    if (
        _load_package_command_json(
            stdout_values[4],
            label="package_parity backend probe output",
        )
        != provenance["build_backend"]
    ):
        raise Phase0EvidenceError("package_parity backend probe output mismatch")
    if stdout_values[9] != f"{pip_tool_version}\n".encode("utf-8"):
        raise Phase0EvidenceError("package_parity pip version output mismatch")
    installed_probe = _load_package_command_json(
        stdout_values[11],
        label="package_parity installed paths probe output",
    )
    if installed_probe != {
        "installed_dist_info": str(installed_paths["dist_info_path"]),
        "installed_package_root": str(installed_paths["installed_package_root"]),
        "site_packages_root": str(installed_paths["site_packages_root"]),
    }:
        raise Phase0EvidenceError("package_parity installed paths probe mismatch")
    if _load_package_command_json(
        stdout_values[12],
        label="package_parity parity command output",
    ) != dict(parity):
        raise Phase0EvidenceError("package_parity command output does not match parity result")
    if provenance["combined_output_sha256"] != _sha256(_canonical_bytes(commands)):
        raise Phase0EvidenceError("package_parity combined command SHA mismatch")


def _validate_package_evidence(
    payload: Mapping[str, Any],
    *,
    current_source_binding: Mapping[str, str],
    repo_root: Path,
) -> None:
    root = _require_exact_keys(
        payload,
        {
            "accepted",
            "authority",
            "build_install_provenance",
            "installed_provenance",
            "network_actions_performed",
            "offline_only",
            "package_file_count",
            "package_inventory",
            "package_inventory_sha256",
            "phase0_gate_roles",
            "protocol_version",
            "sdist_sha256",
            "semantic_sha256",
            "source_binding",
            "source_equals_sdist_equals_wheel_equals_installed",
            "status",
            "version",
            "wheel_provenance",
            "wheel_sha256",
        },
        label="package_parity",
    )
    if (
        root["version"] != PACKAGE_EVIDENCE_VERSION
        or root["status"] != "SEALED"
        or root["authority"] is not False
        or root["offline_only"] is not True
        or root["network_actions_performed"] is not False
        or root["protocol_version"] != PROTOCOL_VERSION
        or root["phase0_gate_roles"] != list(GATE_ROLES)
    ):
        raise Phase0EvidenceError("package_parity envelope identity mismatch")
    declared_semantic = _require_sha256(
        root["semantic_sha256"],
        label="package_parity semantic_sha256",
    )
    if declared_semantic != _semantic_sha256(root):
        raise Phase0EvidenceError("package_parity semantic SHA-256 mismatch")
    source_binding = _coerce_source_binding(
        root["source_binding"],
        label="package_parity source_binding",
    )
    if source_binding != dict(current_source_binding):
        raise Phase0EvidenceError("package_parity current source binding mismatch")

    parity_keys = {
        "accepted",
        "installed_provenance",
        "package_file_count",
        "package_inventory",
        "package_inventory_sha256",
        "sdist_sha256",
        "source_equals_sdist_equals_wheel_equals_installed",
        "wheel_provenance",
        "wheel_sha256",
    }
    parity, installed_paths = _validate_package_parity_result(
        {key: root[key] for key in parity_keys},
        repo_root=repo_root,
    )
    provenance = _require_exact_keys(
        root["build_install_provenance"],
        {
            "artifact_bindings",
            "base_interpreter",
            "base_interpreter_binary",
            "build_backend",
            "command_count",
            "command_roles",
            "commands",
            "combined_output_sha256",
            "environment",
            "network_actions_performed",
            "offline_only",
            "pip_runtime",
            "role",
            "source_binding_after",
            "source_binding_artifact",
            "source_binding_before",
            "uv_runtime",
        },
        label="package_parity build_install_provenance",
    )
    if (
        provenance["role"] != "package_parity"
        or provenance["offline_only"] is not True
        or provenance["network_actions_performed"] is not False
        or _coerce_source_binding(
            provenance["source_binding_before"],
            label="package_parity source_binding_before",
        )
        != source_binding
        or _coerce_source_binding(
            provenance["source_binding_after"],
            label="package_parity source_binding_after",
        )
        != source_binding
    ):
        raise Phase0EvidenceError("package_parity provenance source or authority mismatch")

    environment_raw = _require_exact_keys(
        provenance["environment"],
        {"build_venv", "install_venv", "uv_binary", "uv_cache", "work_root"},
        label="package_parity environment",
    )
    environment = {
        "work_root": _package_path_outside_repo(
            environment_raw["work_root"],
            label="package_parity environment.work_root",
            repo_root=repo_root,
        ),
        "build_venv": _package_path_outside_repo(
            environment_raw["build_venv"],
            label="package_parity environment.build_venv",
            repo_root=repo_root,
        ),
        "install_venv": _package_path_outside_repo(
            environment_raw["install_venv"],
            label="package_parity environment.install_venv",
            repo_root=repo_root,
        ),
        "uv_binary": _package_canonical_path(
            environment_raw["uv_binary"],
            label="package_parity environment.uv_binary",
        ),
        "uv_cache": _package_canonical_path(
            environment_raw["uv_cache"],
            label="package_parity environment.uv_cache",
        ),
    }
    if (
        environment["build_venv"] != environment["work_root"] / "build-venv"
        or environment["install_venv"] != environment["work_root"] / "install-venv"
        or environment["build_venv"] == environment["install_venv"]
        or installed_paths["environment_root"] != environment["install_venv"]
    ):
        raise Phase0EvidenceError("package_parity fresh environment topology mismatch")
    for key in ("work_root", "build_venv", "install_venv", "uv_cache"):
        path = environment[key]
        try:
            observed = path.lstat()
        except OSError as exc:
            raise Phase0EvidenceError(f"package_parity environment.{key} is unavailable") from exc
        if (
            not stat.S_ISDIR(observed.st_mode)
            or stat.S_ISLNK(observed.st_mode)
            or observed.st_uid != os.getuid()
            or stat.S_IMODE(observed.st_mode) & 0o022
            or (key == "work_root" and stat.S_IMODE(observed.st_mode) != 0o700)
        ):
            raise Phase0EvidenceError(f"package_parity environment.{key} permissions are unsafe")
    for key in (
        "site_packages_root",
        "installed_package_root",
        "dist_info_path",
    ):
        try:
            observed = installed_paths[key].lstat()
        except OSError as exc:
            raise Phase0EvidenceError(f"package_parity installed {key} is unavailable") from exc
        if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
            raise Phase0EvidenceError(f"package_parity installed {key} is not a concrete directory")

    artifact_bindings = _require_exact_keys(
        provenance["artifact_bindings"],
        {"sdist", "wheel"},
        label="package_parity artifact_bindings",
    )
    for role, suffix in (("sdist", ".tar.gz"), ("wheel", ".whl")):
        binding = _validate_package_file_binding(
            artifact_bindings[role],
            label=f"package_parity artifact_bindings.{role}",
            readback=True,
        )
        path = Path(binding["path"])
        if (
            not _path_within(path, environment["work_root"])
            or path.parent != environment["work_root"] / role
            or not path.name.endswith(suffix)
            or binding["sha256"] != root[f"{role}_sha256"]
        ):
            raise Phase0EvidenceError(f"package_parity {role} artifact binding mismatch")
    wheel_path = Path(artifact_bindings["wheel"]["path"])
    direct_url = parity["installed_provenance"]["direct_url"]
    if direct_url["url"] != wheel_path.as_uri():
        raise Phase0EvidenceError("package_parity direct_url path does not bind the wheel")

    base_interpreter = _require_exact_keys(
        provenance["base_interpreter"],
        {"executable", "implementation", "realpath", "sha256", "version", "version_info"},
        label="package_parity base_interpreter",
    )
    version_info = base_interpreter["version_info"]
    if (
        base_interpreter["implementation"] != "cpython"
        or type(version_info) is not list
        or len(version_info) != 3
        or any(type(item) is not int or item < 0 for item in version_info)
        or version_info[:2] != [3, 13]
        or base_interpreter["version"] != ".".join(str(item) for item in version_info)
        or base_interpreter["executable"] != base_interpreter["realpath"]
    ):
        raise Phase0EvidenceError("package_parity base interpreter must be exact CPython 3.13")
    _require_sha256(
        base_interpreter["sha256"],
        label="package_parity base_interpreter.sha256",
    )
    base_binary = _require_exact_keys(
        provenance["base_interpreter_binary"],
        {"binary_after", "binary_after_probe", "binary_before"},
        label="package_parity base_interpreter_binary",
    )
    base_bindings = [
        _validate_package_file_binding(
            base_binary[key],
            label=f"package_parity base_interpreter_binary.{key}",
            readback=True,
        )
        for key in ("binary_before", "binary_after_probe", "binary_after")
    ]
    if (
        not all(binding == base_bindings[0] for binding in base_bindings[1:])
        or base_bindings[0]["path"] != base_interpreter["realpath"]
        or base_bindings[0]["sha256"] != base_interpreter["sha256"]
    ):
        raise Phase0EvidenceError("package_parity base interpreter binary drift")
    try:
        base_mode = Path(base_bindings[0]["path"]).stat().st_mode
    except OSError as exc:
        raise Phase0EvidenceError("package_parity base interpreter is unavailable") from exc
    if not base_mode & 0o111:
        raise Phase0EvidenceError("package_parity base interpreter is not executable")

    build_backend = _require_exact_keys(
        provenance["build_backend"],
        {
            "backend_file",
            "backend_module",
            "hatchling_version",
            "package_inventory",
            "package_versions",
            "unnamed_distribution_count",
        },
        label="package_parity build_backend",
    )
    backend_file = _package_canonical_path(
        build_backend["backend_file"],
        label="package_parity build_backend.backend_file",
    )
    if (
        build_backend["backend_module"] != "hatchling.build"
        or build_backend["hatchling_version"] != PACKAGE_EXPECTED_BACKEND_PACKAGES["hatchling"]
        or build_backend["package_versions"] != PACKAGE_EXPECTED_BACKEND_PACKAGES
        or build_backend["package_inventory"] != PACKAGE_EXPECTED_BACKEND_INVENTORY
        or type(build_backend["unnamed_distribution_count"]) is not int
        or build_backend["unnamed_distribution_count"] != 0
        or not _path_within(backend_file, environment["build_venv"])
    ):
        raise Phase0EvidenceError("package_parity build backend inventory mismatch")
    _stable_regular_file(backend_file, require_private=False)

    pip_runtime = _require_exact_keys(
        provenance["pip_runtime"],
        {"location", "output", "python_version", "version"},
        label="package_parity pip_runtime",
    )
    pip_location = _package_canonical_path(
        pip_runtime["location"],
        label="package_parity pip_runtime.location",
    )
    expected_pip_output = f"pip {PACKAGE_EXPECTED_PIP_VERSION} from {pip_location} (python 3.13)"
    if (
        pip_runtime["version"] != PACKAGE_EXPECTED_PIP_VERSION
        or pip_runtime["python_version"] != "3.13"
        or pip_runtime["output"] != expected_pip_output
        or not _path_within(pip_location, environment["install_venv"])
    ):
        raise Phase0EvidenceError("package_parity pip runtime mismatch")
    try:
        pip_stat = pip_location.lstat()
    except OSError as exc:
        raise Phase0EvidenceError("package_parity pip location is unavailable") from exc
    if not stat.S_ISDIR(pip_stat.st_mode) or stat.S_ISLNK(pip_stat.st_mode):
        raise Phase0EvidenceError("package_parity pip location is not a concrete directory")

    uv_runtime = _require_exact_keys(
        provenance["uv_runtime"],
        {"binary_after", "binary_after_version", "binary_before", "output", "version"},
        label="package_parity uv_runtime",
    )
    uv_bindings = [
        _validate_package_file_binding(
            uv_runtime[key],
            label=f"package_parity uv_runtime.{key}",
            readback=True,
        )
        for key in ("binary_before", "binary_after_version", "binary_after")
    ]
    if (
        uv_runtime["version"] != PACKAGE_EXPECTED_UV_VERSION
        or uv_runtime["output"] != PACKAGE_EXPECTED_UV_OUTPUT
        or not all(binding == uv_bindings[0] for binding in uv_bindings[1:])
        or uv_bindings[0]["path"] != str(environment["uv_binary"])
    ):
        raise Phase0EvidenceError("package_parity uv runtime mismatch")
    try:
        uv_mode = environment["uv_binary"].stat().st_mode
    except OSError as exc:
        raise Phase0EvidenceError("package_parity uv binary is unavailable") from exc
    if not uv_mode & 0o111:
        raise Phase0EvidenceError("package_parity uv binary is not executable")

    source_artifact = _validate_package_file_binding(
        provenance["source_binding_artifact"],
        label="package_parity source_binding_artifact",
        readback=False,
    )
    artifact_binding, artifact_raw = _external_file_binding(
        Path(source_artifact["path"]),
        repo_root=repo_root,
        label="package_parity source_binding_artifact",
    )
    if (
        source_artifact["sha256"] != artifact_binding["sha256"]
        or source_artifact["size_bytes"] != artifact_binding["size_bytes"]
        or artifact_raw != _canonical_resource_bytes(source_binding)
    ):
        raise Phase0EvidenceError("package_parity source binding artifact mismatch")

    _validate_package_commands(
        provenance,
        parity=parity,
        installed_paths=installed_paths,
        environment=environment,
        artifact_bindings=artifact_bindings,
        base_interpreter=base_interpreter,
        pip_runtime=pip_runtime,
        repo_root=repo_root,
    )


def _validate_pytest_gate(role: str, claims: Mapping[str, Any], raw: bytes) -> None:
    expected = {"errors", "exit_code", "failed", "passed", "xfail", "xpass"}
    if role == "full_offline_suite":
        expected = expected.union({"raw_output_sha256", "skip_allowlist", "skipped"})
    elif role == "recommended_core_tests":
        expected = expected.union({"skipped", "staged_upgrade_exit_code"})
    else:
        expected = expected.union({"skipped"})
    parsed = _require_claims(claims, expected, label=f"{role} claims")
    _require_exit_zero_claims(parsed, label=role)
    if type(parsed["passed"]) is not int or parsed["passed"] <= 0:
        raise Phase0EvidenceError(f"{role} must bind passed > 0")
    for key in ("failed", "errors", "xfail", "xpass"):
        if type(parsed[key]) is not int or parsed[key] != 0:
            raise Phase0EvidenceError(f"{role} must bind zero {key}")
    if type(parsed["skipped"]) is not int or parsed["skipped"] < 0:
        raise Phase0EvidenceError(f"{role} skipped count is invalid")
    if role != "full_offline_suite" and parsed["skipped"] != 0:
        raise Phase0EvidenceError(f"{role} must not bind skipped tests")
    if role == "full_offline_suite":
        skip_allowlist = parsed["skip_allowlist"]
        if type(skip_allowlist) is not list:
            raise Phase0EvidenceError("full_offline_suite skip_allowlist must be an array")
        normalized_allowlist: list[dict[str, Any]] = []
        for index, raw_entry in enumerate(skip_allowlist):
            entry = _require_exact_keys(
                raw_entry,
                {"count", "line", "path", "reason"},
                label=f"full_offline_suite skip_allowlist[{index}]",
            )
            if (
                type(entry["count"]) is not int
                or entry["count"] <= 0
                or type(entry["line"]) is not int
                or entry["line"] <= 0
                or type(entry["path"]) is not str
                or not entry["path"]
                or type(entry["reason"]) is not str
                or not entry["reason"]
            ):
                raise Phase0EvidenceError("full_offline_suite skip_allowlist entry invalid")
            normalized_allowlist.append(dict(entry))
        if (
            parsed["raw_output_sha256"] != _sha256(raw)
            or normalized_allowlist != _pytest_skip_entries(raw, label=role)
            or sum(entry["count"] for entry in normalized_allowlist) != parsed["skipped"]
        ):
            raise Phase0EvidenceError("full_offline_suite skip allowlist mismatch")
    raw_counts = _pytest_summary_counts(raw, label=role)
    for key in ("passed", "skipped", "failed", "errors", "xfail", "xpass"):
        if parsed[key] != raw_counts[key]:
            raise Phase0EvidenceError(f"{role} pytest summary count mismatch for {key}")
    if role == "recommended_core_tests":
        if (
            type(parsed["staged_upgrade_exit_code"]) is not int
            or parsed["staged_upgrade_exit_code"] != 0
        ):
            raise Phase0EvidenceError("recommended_core_tests staged gate did not exit 0")
        text = _decode_text(raw, label=role)
        if (
            text.count("Running staged upgrade focused tests...") != 1
            or text.count("Running staged upgrade focused mypy...") != 1
            or text.count("staged_upgrade_exit_code=0") != 1
            or "Success: no issues found" not in text
        ):
            raise Phase0EvidenceError("recommended_core_tests staged gate evidence missing")


def _validate_gate_semantics(
    *,
    role: str,
    claims: Mapping[str, Any],
    raw: bytes,
    current_source_binding: Mapping[str, str],
    current_source_state: Mapping[str, Any],
    repo_root: Path,
) -> None:
    if role == "native_sync_receipt":
        payload = _load_canonical_json_resource(raw, label=role)
        declared = payload.get("semantic_sha256")
        if type(declared) is not str or declared != _semantic_sha256(payload):
            raise Phase0EvidenceError("native_sync_receipt semantic SHA-256 mismatch")
        _validate_native_dependency_receipt(payload, repo_root=repo_root)
        _normalize_receipt_source_state(
            payload.get("source_state"),
            current_source=current_source_state,
            repo_root=repo_root,
        )
        if _source_binding_from_state(current_source_state) != dict(current_source_binding):
            raise Phase0EvidenceError("native_sync_receipt index source binding mismatch")
        if claims != {"accepted": True}:
            raise Phase0EvidenceError("native_sync_receipt claims must bind accepted=true")
        return
    if role == "native_sync_log":
        parsed = _require_claims(claims, {"exit_code"}, label=f"{role} claims")
        _require_exit_zero_claims(parsed, label=role)
        text = _decode_text(raw, label=role)
        if "uv sync --locked --all-extras --offline" not in text:
            raise Phase0EvidenceError("native_sync_log does not contain the exact offline sync")
        if not re.search(r"\b(exit0|exit 0|exit_code=0)\b", text):
            raise Phase0EvidenceError("native_sync_log does not prove exit0")
        return
    if role in {"v2_evidence_tests", "recommended_core_tests", "full_offline_suite"}:
        _validate_pytest_gate(role, claims, raw)
        return
    if role == "mypy":
        parsed = _require_claims(claims, {"exit_code"}, label=f"{role} claims")
        _require_exit_zero_claims(parsed, label=role)
        if "Success: no issues found" not in _decode_text(raw, label=role):
            raise Phase0EvidenceError("mypy raw evidence does not prove success")
        return
    if role == "black":
        parsed = _require_claims(
            claims,
            {"exit_code", "unchanged"},
            label=f"{role} claims",
        )
        _require_exit_zero_claims(parsed, label=role)
        if parsed["unchanged"] is not True:
            raise Phase0EvidenceError("black did not bind unchanged=true")
        text = _decode_text(raw, label=role)
        if "would reformat" in text.lower() or "left unchanged" not in text.lower():
            raise Phase0EvidenceError("black raw evidence does not prove unchanged")
        return
    if role == "diff_check":
        parsed = _require_claims(
            claims,
            {"exit_code", "raw_output_sha256"},
            label=f"{role} claims",
        )
        _require_exit_zero_claims(parsed, label=role)
        if raw != b"" or parsed["raw_output_sha256"] != _sha256(b""):
            raise Phase0EvidenceError("diff_check must bind exit0 and empty raw output")
        return
    if role == "package_parity":
        payload = _load_canonical_json_resource(raw, label=role)
        _validate_package_evidence_schema(payload, repo_root=repo_root)
        _validate_package_evidence(
            payload,
            current_source_binding=current_source_binding,
            repo_root=repo_root,
        )
        if claims != {"accepted": True}:
            raise Phase0EvidenceError("package_parity claims must bind accepted=true")
        return
    if role == "hash_freeze_readback":
        payload = _load_canonical_json_resource(raw, label=role)
        readback = _require_exact_keys(
            payload,
            {"accepted", "hashes", "source_binding"},
            label=role,
        )
        if readback["accepted"] is not True:
            raise Phase0EvidenceError("hash_freeze_readback accepted must be true")
        if _coerce_source_binding(readback["source_binding"], label=role) != dict(
            current_source_binding
        ):
            raise Phase0EvidenceError("hash_freeze_readback source binding mismatch")
        if readback["hashes"] != _current_hash_freeze(repo_root):
            raise Phase0EvidenceError("hash_freeze_readback hashes do not match current files")
        if claims != {"accepted": True}:
            raise Phase0EvidenceError("hash_freeze_readback claims must bind accepted=true")
        return
    raise Phase0EvidenceError(f"unknown Phase 0 gate role: {role}")


def _parse_gate_manifest(
    raw: bytes,
    *,
    base_commit: str,
    current_source_binding: Mapping[str, str],
) -> list[tuple[str, str, str, Path, Mapping[str, Any]]]:
    payload = _load_canonical_json_resource(raw, label="gate manifest")
    manifest = _require_exact_keys(
        payload,
        {
            "base_commit",
            "gates",
            "protocol_version",
            "semantic_sha256",
            "source_binding",
            "version",
        },
        label="gate manifest",
    )
    if (
        manifest["protocol_version"] != PROTOCOL_VERSION
        or manifest["version"] != GATE_MANIFEST_VERSION
        or manifest["base_commit"] != base_commit
    ):
        raise Phase0EvidenceError("gate manifest identity mismatch")
    if manifest["semantic_sha256"] != _semantic_sha256(manifest):
        raise Phase0EvidenceError("gate manifest semantic SHA-256 mismatch")
    source_binding = _validate_source_binding(
        manifest["source_binding"],
        label="gate manifest source_binding",
    )
    if source_binding != dict(current_source_binding):
        raise Phase0EvidenceError("gate manifest source binding is stale")
    gates = manifest["gates"]
    if type(gates) is not list:
        raise Phase0EvidenceError("gate manifest gates must be an array")
    if len(gates) != len(GATE_ROLES):
        raise Phase0EvidenceError("gate manifest must bind each closed gate exactly once")
    specs: list[tuple[str, str, str, Path, Mapping[str, Any]]] = []
    observed_roles: list[str] = []
    for index, raw_gate in enumerate(gates):
        gate = _require_exact_keys(
            raw_gate,
            {"claims", "id", "kind", "path", "role", "source_binding"},
            label=f"gate manifest gates[{index}]",
        )
        role = gate["role"]
        if role not in GATE_KINDS:
            raise Phase0EvidenceError(f"unknown Phase 0 gate role: {role}")
        observed_roles.append(role)
        if gate["id"] != role or gate["kind"] != GATE_KINDS[role]:
            raise Phase0EvidenceError(f"gate {role} has noncanonical id or kind")
        gate_source = _validate_source_binding(
            gate["source_binding"],
            label=f"gate {role} source_binding",
        )
        if gate_source != dict(current_source_binding):
            raise Phase0EvidenceError(f"gate {role} source binding is stale")
        if type(gate["path"]) is not str or not Path(gate["path"]).is_absolute():
            raise Phase0EvidenceError(f"gate {role} path must be absolute")
        if type(gate["claims"]) is not dict:
            raise Phase0EvidenceError(f"gate {role} claims must be an object")
        specs.append((gate["kind"], gate["id"], role, Path(gate["path"]), gate["claims"]))
    if tuple(observed_roles) != GATE_ROLES:
        missing = sorted(set(GATE_ROLES) - set(observed_roles))
        extra = sorted(set(observed_roles) - set(GATE_ROLES))
        if len(set(observed_roles)) != len(observed_roles):
            raise Phase0EvidenceError("gate manifest contains duplicate gate role")
        raise Phase0EvidenceError(
            f"gate manifest role order/closure mismatch; missing={missing} extra={extra}"
        )
    return specs


def _classify_dirty_paths(
    dirty_paths: Sequence[str],
    *,
    allowed_patterns: Sequence[str],
    pre_existing_paths: Sequence[str],
) -> list[dict[str, str]]:
    compiled = [(pattern, _glob_regex(pattern)) for pattern in allowed_patterns]
    pre_existing = set(pre_existing_paths)
    dirty = set(dirty_paths)
    stale = sorted(pre_existing - dirty)
    if stale:
        raise Phase0EvidenceError(f"stale pre-existing classifications: {stale}")
    classified: list[dict[str, str]] = []
    for path in dirty_paths:
        phase0_matches = [pattern for pattern, regex in compiled if regex.fullmatch(path)]
        is_pre_existing = path in pre_existing
        if phase0_matches and is_pre_existing:
            raise Phase0EvidenceError(f"mixed Phase 0/pre-existing classification: {path}")
        if phase0_matches:
            classification = PHASE0_CLASSIFICATION
        elif is_pre_existing:
            classification = PRE_EXISTING_CLASSIFICATION
        else:
            raise Phase0EvidenceError(f"unknown dirty path: {path}")
        classified.append({"classification": classification, "path": path})
    return classified


def _parse_external_spec(value: str, *, kind: str) -> tuple[str, str, Path]:
    if "=" not in value:
        raise Phase0EvidenceError(f"{kind} must use id=/absolute/path")
    raw_id, raw_path = value.split("=", 1)
    evidence_id = _require_token(raw_id, label=f"{kind} ID")
    path = Path(raw_path)
    if not path.is_absolute():
        raise Phase0EvidenceError(f"{kind} path must be absolute")
    return kind, evidence_id, path


def _external_bindings(
    specs: Sequence[tuple[str, str, str, Path, Mapping[str, Any]]],
    *,
    repo_root: Path,
    current_source_binding: Mapping[str, str],
    current_source_state: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], bytes]]:
    if len(specs) != len(GATE_ROLES):
        raise Phase0EvidenceError("gate evidence closure must bind all required gates")
    identifiers = [identifier for _kind, identifier, _role, _path, _claims in specs]
    paths = [str(path) for _kind, _identifier, _role, path, _claims in specs]
    _require_unique_casefold(identifiers, label="external evidence IDs")
    _require_unique_casefold(paths, label="external evidence paths")
    records: list[dict[str, Any]] = []
    raw_by_identity: dict[tuple[str, str], bytes] = {}
    python_executables: set[str] = set()
    native_sync_python: str | None = None
    tool_versions: dict[str, set[str]] = {}
    fresh_venv: str | None = None
    fresh_python: str | None = None
    native_receipt_payload: dict[str, Any] | None = None
    native_log_receipt: dict[str, Any] | None = None
    for kind, identifier, role, path, claims in sorted(
        specs,
        key=lambda item: (item[0], item[1], str(item[3]).encode("utf-8")),
    ):
        binding, raw = _external_file_binding(
            path,
            repo_root=repo_root,
            label=f"external {kind} {identifier}",
        )
        semantic_raw = raw
        if role == "full_offline_suite":
            receipt, parsed_streams, _framed = _parse_framed_main_suite_receipt_v1(
                raw,
                label=role,
            )
            schemas, _schema_bindings, _schema_raw = _load_v2_schema_registry(repo_root)
            _validate_v2_schema(
                receipt,
                artifact_version=MAIN_SUITE_RECEIPT_VERSION,
                schemas=schemas,
                label=f"{role} main-suite receipt",
            )
            _validate_v2_seal(receipt, label=f"{role} main-suite receipt")
            legacy_context = _legacy_v2_main_suite_context(
                evidence_path=path,
                repo_root=repo_root,
                schemas=schemas,
                current_source_state=current_source_state,
            )
            binary_frames = _validate_main_suite_attestation_frames(
                parsed_streams["attestation"],
                label=role,
            )
            decoded_frames = _decode_main_suite_attestation_frames(
                binary_frames,
                label=role,
            )
            policy, policy_bindings, _producer_binding = _load_v2_main_suite_policy(
                repo_root=repo_root,
                schemas=schemas,
            )
            derived_claims = _validate_v2_main_suite_semantics(
                receipt,
                streams=parsed_streams,
                frames=decoded_frames,
                repo_root=repo_root,
                policy=policy,
                policy_bindings=policy_bindings,
                session_binding=legacy_context["session_binding"],
                skip_baseline=legacy_context["skip"],
                skip_baseline_raw=legacy_context["skip_raw"],
                base_commit=legacy_context["base_commit"],
                source_state=legacy_context["source_state"],
                package_source_full=legacy_context["package_source_full"],
            )
            if dict(claims) != derived_claims:
                raise Phase0EvidenceError(
                    "full_offline_suite gate claims differ from dedicated receipt"
                )
            semantic_raw = parsed_streams["stdout"] + parsed_streams["stderr"]
        elif role in LOG_ROLES:
            receipt, semantic_raw = _parse_command_receipt_log(
                raw,
                role=role,
                repo_root=repo_root,
                current_source_binding=current_source_binding,
            )
            _validate_command_identity(role, receipt, repo_root=repo_root)
            if role == "native_sync_log":
                native_log_receipt = receipt
            python_executable = _python_executable_from_receipt(role, receipt)
            if python_executable is not None:
                python_executables.add(python_executable)
            native_python = _native_sync_python_from_receipt(role, receipt)
            if native_python is not None:
                native_sync_python = native_python
            for tool, version in _tool_versions_from_receipt(role, receipt).items():
                tool_versions.setdefault(tool, set()).add(version)
        _validate_gate_semantics(
            role=role,
            claims=claims,
            raw=semantic_raw,
            current_source_binding=current_source_binding,
            current_source_state=current_source_state,
            repo_root=repo_root,
        )
        if role == "native_sync_receipt":
            payload = _load_canonical_json_resource(raw, label=role)
            native_receipt_payload = payload
            fresh_venv, fresh_python = _fresh_python_binding_from_native_receipt(
                payload,
                repo_root=repo_root,
            )
        if role == "native_sync_log" and fresh_venv is not None:
            receipt, _output = _parse_command_receipt_log(
                raw,
                role=role,
                repo_root=repo_root,
                current_source_binding=current_source_binding,
            )
            if receipt["commands"][0]["env"]["UV_PROJECT_ENVIRONMENT"] != fresh_venv:
                raise Phase0EvidenceError("native_sync_log fresh environment mismatch")
        records.append(
            {"claims": dict(claims), "id": identifier, "kind": kind, "role": role, **binding}
        )
        raw_by_identity[(kind, identifier)] = raw
    if fresh_python is not None and any(
        python_executable != fresh_python for python_executable in python_executables
    ):
        raise Phase0EvidenceError("log receipts do not use dependency fresh Python")
    if len(python_executables) != 1:
        raise Phase0EvidenceError("log receipts must bind one shared CPython 3.13 executable")
    if (
        fresh_python is not None
        and native_sync_python is not None
        and not _same_resolved_path(native_sync_python, fresh_python)
    ):
        raise Phase0EvidenceError("native_sync_log native Python binding mismatch")
    for versions in tool_versions.values():
        if len(versions) != 1:
            raise Phase0EvidenceError("log receipts bind inconsistent tool versions")
    if native_receipt_payload is None or native_log_receipt is None:
        raise Phase0EvidenceError("native dependency receipt/log binding is incomplete")
    _validate_native_tool_bindings(
        native_receipt_payload,
        native_log_receipt,
        tool_versions,
    )
    return records, raw_by_identity


def _assert_external_stable(
    before_records: Sequence[Mapping[str, Any]],
    before_raw: Mapping[tuple[str, str], bytes],
    after_records: Sequence[Mapping[str, Any]],
    after_raw: Mapping[tuple[str, str], bytes],
) -> None:
    if _canonical_bytes(list(before_records)) != _canonical_bytes(list(after_records)):
        raise Phase0EvidenceError("external evidence bindings changed during collection")
    if dict(before_raw) != dict(after_raw):
        raise Phase0EvidenceError("external evidence bytes changed during collection")


def build_evidence_index(
    *,
    repo_root: Path,
    base_commit: str,
    allowed_path_patterns: Sequence[str],
    classification_manifest: Path,
    gate_manifest: Path,
) -> dict[str, Any]:
    """Build and validate one sealed Phase 0 evidence index in memory."""

    canonical_commit = _validate_base_commit(base_commit)
    root, canonical_commit = _resolve_repo(repo_root, canonical_commit)
    patterns = _allowed_patterns(allowed_path_patterns)
    classification_binding_before, classification_raw_before = _external_file_binding(
        classification_manifest,
        repo_root=root,
        label="pre-existing classification manifest",
    )
    pre_existing_paths = _parse_classification_manifest(
        classification_raw_before,
        base_commit=canonical_commit,
    )
    source_before = _git_snapshot(root, canonical_commit)
    source_state_before = _public_source_state(source_before)
    source_binding = _source_binding_from_state(source_state_before)
    gate_manifest_binding_before, gate_manifest_raw_before = _external_file_binding(
        gate_manifest,
        repo_root=root,
        label="Phase 0 gate manifest",
    )
    gate_specs = _parse_gate_manifest(
        gate_manifest_raw_before,
        base_commit=canonical_commit,
        current_source_binding=source_binding,
    )
    external_before, external_raw_before = _external_bindings(
        gate_specs,
        repo_root=root,
        current_source_binding=source_binding,
        current_source_state=source_state_before,
    )

    classification_binding_after, classification_raw_after = _external_file_binding(
        classification_manifest,
        repo_root=root,
        label="pre-existing classification manifest",
    )
    gate_manifest_binding_after, gate_manifest_raw_after = _external_file_binding(
        gate_manifest,
        repo_root=root,
        label="Phase 0 gate manifest",
    )
    gate_specs_after = _parse_gate_manifest(
        gate_manifest_raw_after,
        base_commit=canonical_commit,
        current_source_binding=source_binding,
    )
    external_after, external_raw_after = _external_bindings(
        gate_specs_after,
        repo_root=root,
        current_source_binding=source_binding,
        current_source_state=source_state_before,
    )
    source_after = _git_snapshot(root, canonical_commit)
    if (
        classification_binding_before != classification_binding_after
        or classification_raw_before != classification_raw_after
    ):
        raise Phase0EvidenceError("classification manifest changed during collection")
    if (
        gate_manifest_binding_before != gate_manifest_binding_after
        or gate_manifest_raw_before != gate_manifest_raw_after
        or gate_specs != gate_specs_after
    ):
        raise Phase0EvidenceError("gate manifest changed during collection")
    _assert_external_stable(
        external_before,
        external_raw_before,
        external_after,
        external_raw_after,
    )
    _assert_snapshot_equal(source_before, source_after)
    classified = _classify_dirty_paths(
        source_state_before["dirty_paths"],
        allowed_patterns=patterns,
        pre_existing_paths=pre_existing_paths,
    )
    report = _seal(
        {
            "accepted": True,
            "allowlist": {
                "allowed_phase0_path_patterns": patterns,
                "classified_paths": classified,
                "pre_existing_classification_manifest": classification_binding_before,
            },
            "authority": False,
            "base_commit": canonical_commit,
            "external_evidence": external_before,
            "gate_manifest": gate_manifest_binding_before,
            "network_actions_performed": False,
            "offline_only": True,
            "protocol_version": PROTOCOL_VERSION,
            "repo_root": str(root),
            "source_binding": source_binding,
            "source_state": source_state_before,
            "status": "SEALED",
            "version": EVIDENCE_INDEX_VERSION,
        }
    )
    validate_evidence_index(report, verify_external=True)
    return report


def _raw_binding_valid(
    value: Any,
    *,
    label: str,
    require_nul_termination: bool = False,
) -> None:
    binding = _require_exact_keys(
        value,
        {"bytes_base64", "encoding", "sha256", "size_bytes"},
        label=label,
    )
    if binding["encoding"] != "base64":
        raise Phase0EvidenceError(f"{label} encoding must be base64")
    try:
        raw = base64.b64decode(binding["bytes_base64"], validate=True)
    except (TypeError, ValueError) as exc:
        raise Phase0EvidenceError(f"{label} bytes are invalid base64") from exc
    if (
        type(binding["size_bytes"]) is not int
        or binding["size_bytes"] < 0
        or len(raw) != binding["size_bytes"]
        or type(binding["sha256"]) is not str
        or SHA256_RE.fullmatch(binding["sha256"]) is None
        or _sha256(raw) != binding["sha256"]
    ):
        raise Phase0EvidenceError(f"{label} byte binding mismatch")
    if require_nul_termination and raw and not raw.endswith(b"\0"):
        raise Phase0EvidenceError(f"{label} must be NUL terminated")


def _validate_file_binding(
    value: Any,
    *,
    label: str,
    expected_extra_keys: set[str] | None = None,
) -> dict[str, Any]:
    extra = expected_extra_keys or set()
    binding = _require_exact_keys(
        value,
        {"mode", "path", "sha256", "size_bytes"}.union(extra),
        label=label,
    )
    if (
        binding["mode"] != "0600"
        or type(binding["path"]) is not str
        or type(binding["size_bytes"]) is not int
        or binding["size_bytes"] < 0
        or type(binding["sha256"]) is not str
        or SHA256_RE.fullmatch(binding["sha256"]) is None
    ):
        raise Phase0EvidenceError(f"{label} binding is invalid")
    return binding


def validate_evidence_index(
    value: Any,
    *,
    verify_external: bool,
) -> dict[str, Any]:
    """Validate exact shape, seals, raw bytes, and optionally external readback."""

    report = _require_exact_keys(
        value,
        {
            "accepted",
            "allowlist",
            "authority",
            "base_commit",
            "external_evidence",
            "gate_manifest",
            "network_actions_performed",
            "offline_only",
            "protocol_version",
            "repo_root",
            "semantic_sha256",
            "source_binding",
            "source_state",
            "status",
            "version",
        },
        label="evidence index",
    )
    if (
        report["protocol_version"] != PROTOCOL_VERSION
        or report["version"] != EVIDENCE_INDEX_VERSION
        or report["accepted"] is not True
        or report["authority"] is not False
        or report["offline_only"] is not True
        or report["network_actions_performed"] is not False
        or report["status"] != "SEALED"
    ):
        raise Phase0EvidenceError("evidence index identity or authority mismatch")
    _validate_base_commit(report["base_commit"])
    if (
        type(report["semantic_sha256"]) is not str
        or SHA256_RE.fullmatch(report["semantic_sha256"]) is None
        or report["semantic_sha256"] != _semantic_sha256(report)
    ):
        raise Phase0EvidenceError("evidence index semantic SHA-256 mismatch")
    allowlist = _require_exact_keys(
        report["allowlist"],
        {
            "allowed_phase0_path_patterns",
            "classified_paths",
            "pre_existing_classification_manifest",
        },
        label="allowlist",
    )
    patterns = _allowed_patterns(allowlist["allowed_phase0_path_patterns"])
    if patterns != allowlist["allowed_phase0_path_patterns"]:
        raise Phase0EvidenceError("allowed Phase 0 patterns are not canonically ordered")
    classification_binding = _validate_file_binding(
        allowlist["pre_existing_classification_manifest"],
        label="classification manifest binding",
    )
    classified = allowlist["classified_paths"]
    if type(classified) is not list:
        raise Phase0EvidenceError("classified paths must be an array")
    classified_paths: list[str] = []
    for index, raw_entry in enumerate(classified):
        entry = _require_exact_keys(
            raw_entry,
            {"classification", "path"},
            label=f"classified_paths[{index}]",
        )
        if entry["classification"] not in {
            PHASE0_CLASSIFICATION,
            PRE_EXISTING_CLASSIFICATION,
        }:
            raise Phase0EvidenceError("classified path has unknown classification")
        classified_paths.append(
            _repo_relative_path(entry["path"], label=f"classified_paths[{index}].path")
        )
    _require_unique_casefold(classified_paths, label="classified paths")
    if classified_paths != sorted(
        classified_paths,
        key=lambda item: item.encode("utf-8"),
    ):
        raise Phase0EvidenceError("classified paths are not canonically ordered")

    source = _require_exact_keys(
        report["source_state"],
        {
            "base_commit",
            "binary_diff_from_base",
            "dirty_paths",
            "porcelain_v1_z",
            "source_state_sha256",
            "untracked",
        },
        label="source_state",
    )
    if source["base_commit"] != report["base_commit"]:
        raise Phase0EvidenceError("source state base commit mismatch")
    _raw_binding_valid(source["binary_diff_from_base"], label="binary diff")
    _raw_binding_valid(
        source["porcelain_v1_z"],
        label="NUL porcelain",
        require_nul_termination=True,
    )
    dirty_paths = source["dirty_paths"]
    if type(dirty_paths) is not list:
        raise Phase0EvidenceError("dirty paths must be an array")
    normalized_dirty = [_repo_relative_path(item, label="dirty path") for item in dirty_paths]
    _require_unique_casefold(normalized_dirty, label="dirty paths")
    if normalized_dirty != sorted(normalized_dirty, key=lambda item: item.encode("utf-8")):
        raise Phase0EvidenceError("dirty paths are not canonically ordered")
    if classified_paths != normalized_dirty:
        raise Phase0EvidenceError("classified paths do not exactly cover dirty paths")
    untracked = source["untracked"]
    if type(untracked) is not list:
        raise Phase0EvidenceError("untracked inventory must be an array")
    untracked_paths: list[str] = []
    for index, raw_entry in enumerate(untracked):
        entry = _require_exact_keys(
            raw_entry,
            {
                "mode",
                "path",
                "sha256",
                "size_bytes",
                "symlink_target",
                "type",
            },
            label=f"untracked[{index}]",
        )
        path = _repo_relative_path(entry["path"], label=f"untracked[{index}].path")
        if (
            entry["type"] not in {"file", "symlink"}
            or type(entry["mode"]) is not str
            or re.fullmatch(r"[0-7]{4}", entry["mode"]) is None
            or type(entry["size_bytes"]) is not int
            or entry["size_bytes"] < 0
            or type(entry["sha256"]) is not str
            or SHA256_RE.fullmatch(entry["sha256"]) is None
            or (entry["type"] == "file" and entry["symlink_target"] is not None)
            or (entry["type"] == "symlink" and type(entry["symlink_target"]) is not str)
        ):
            raise Phase0EvidenceError(f"untracked[{index}] binding is invalid")
        if entry["type"] == "symlink":
            target_raw = entry["symlink_target"].encode("utf-8", errors="strict")
            if _sha256(target_raw) != entry["sha256"]:
                raise Phase0EvidenceError(f"untracked[{index}] symlink target SHA-256 mismatch")
        untracked_paths.append(path)
    _require_unique_casefold(untracked_paths, label="untracked paths")
    if untracked_paths != sorted(
        untracked_paths,
        key=lambda item: item.encode("utf-8"),
    ):
        raise Phase0EvidenceError("untracked paths are not canonically ordered")
    if not set(untracked_paths).issubset(normalized_dirty):
        raise Phase0EvidenceError("untracked paths are not covered by dirty paths")
    source_unsealed = dict(source)
    declared_source_sha = source_unsealed.pop("source_state_sha256", None)
    if (
        type(declared_source_sha) is not str
        or SHA256_RE.fullmatch(declared_source_sha) is None
        or declared_source_sha != _sha256(_canonical_bytes(source_unsealed))
    ):
        raise Phase0EvidenceError("source_state_sha256 mismatch")
    source_binding = _validate_source_binding(
        report["source_binding"],
        label="evidence index source_binding",
    )
    if source_binding != _source_binding_from_state(source):
        raise Phase0EvidenceError("evidence index source_binding mismatch")
    gate_manifest_binding = _validate_file_binding(
        report["gate_manifest"],
        label="gate manifest binding",
    )

    external = report["external_evidence"]
    if type(external) is not list:
        raise Phase0EvidenceError("external evidence must be an array")
    external_ids: list[str] = []
    external_paths: list[str] = []
    external_kinds: set[str] = set()
    external_roles: list[str] = []
    for index, raw_entry in enumerate(external):
        entry = _validate_file_binding(
            raw_entry,
            label=f"external_evidence[{index}]",
            expected_extra_keys={"claims", "id", "kind", "role"},
        )
        identifier = _require_token(entry["id"], label="external evidence ID")
        role = entry["role"]
        if role not in GATE_KINDS:
            raise Phase0EvidenceError("external evidence role is invalid")
        if entry["id"] != role or entry["kind"] != GATE_KINDS[role]:
            raise Phase0EvidenceError("external evidence has noncanonical id or kind")
        if type(entry["claims"]) is not dict:
            raise Phase0EvidenceError("external evidence claims must be an object")
        if entry["kind"] not in {"artifact", "log"}:
            raise Phase0EvidenceError("external evidence kind is invalid")
        external_ids.append(identifier)
        external_paths.append(entry["path"])
        external_kinds.add(entry["kind"])
        external_roles.append(role)
    _require_unique_casefold(external_ids, label="external evidence IDs")
    _require_unique_casefold(external_paths, label="external evidence paths")
    if external_kinds != {"artifact", "log"}:
        raise Phase0EvidenceError("external evidence must include artifacts and logs")
    if sorted(external_roles) != sorted(GATE_ROLES) or len(set(external_roles)) != len(GATE_ROLES):
        raise Phase0EvidenceError("external evidence does not bind closed gate roles exactly once")
    expected_external_order = sorted(
        external,
        key=lambda item: (
            item["kind"],
            item["id"],
            item["path"].encode("utf-8"),
        ),
    )
    if external != expected_external_order:
        raise Phase0EvidenceError("external evidence is not canonically ordered")

    if verify_external:
        if type(report["repo_root"]) is not str:
            raise Phase0EvidenceError("repo_root must be an absolute canonical path")
        raw_repo_root = Path(report["repo_root"])
        repo_root = raw_repo_root.resolve(strict=True)
        if not raw_repo_root.is_absolute() or repo_root != raw_repo_root:
            raise Phase0EvidenceError("repo_root must be an absolute canonical path")
        classification_current, _ = _external_file_binding(
            Path(classification_binding["path"]),
            repo_root=repo_root,
            label="classification manifest readback",
        )
        if classification_current != classification_binding:
            raise Phase0EvidenceError("classification manifest readback mismatch")
        gate_manifest_current, gate_manifest_raw = _external_file_binding(
            Path(gate_manifest_binding["path"]),
            repo_root=repo_root,
            label="gate manifest readback",
        )
        if gate_manifest_current != gate_manifest_binding:
            raise Phase0EvidenceError("gate manifest readback mismatch")
        gate_specs = _parse_gate_manifest(
            gate_manifest_raw,
            base_commit=report["base_commit"],
            current_source_binding=source_binding,
        )
        gate_claims = {role: dict(claims) for _kind, _identifier, role, _path, claims in gate_specs}
        python_executables: set[str] = set()
        native_sync_python: str | None = None
        tool_versions: dict[str, set[str]] = {}
        fresh_venv: str | None = None
        fresh_python: str | None = None
        native_receipt_payload: dict[str, Any] | None = None
        native_log_receipt: dict[str, Any] | None = None
        for entry in external:
            current, raw = _external_file_binding(
                Path(entry["path"]),
                repo_root=repo_root,
                label=f"external evidence readback {entry['id']}",
            )
            expected = {key: entry[key] for key in ("mode", "path", "sha256", "size_bytes")}
            if current != expected:
                raise Phase0EvidenceError(f"external evidence readback mismatch: {entry['id']}")
            if gate_claims.get(entry["role"]) != entry["claims"]:
                raise Phase0EvidenceError(f"gate manifest claims mismatch: {entry['role']}")
            semantic_raw = raw
            if entry["role"] == "full_offline_suite":
                receipt, parsed_streams, _framed = _parse_framed_main_suite_receipt_v1(
                    raw,
                    label=entry["role"],
                )
                schemas, _schema_bindings, _schema_raw = _load_v2_schema_registry(repo_root)
                _validate_v2_schema(
                    receipt,
                    artifact_version=MAIN_SUITE_RECEIPT_VERSION,
                    schemas=schemas,
                    label=f"{entry['role']} main-suite receipt",
                )
                _validate_v2_seal(
                    receipt,
                    label=f"{entry['role']} main-suite receipt",
                )
                legacy_context = _legacy_v2_main_suite_context(
                    evidence_path=Path(entry["path"]),
                    repo_root=repo_root,
                    schemas=schemas,
                    current_source_state=source,
                )
                binary_frames = _validate_main_suite_attestation_frames(
                    parsed_streams["attestation"],
                    label=entry["role"],
                )
                decoded_frames = _decode_main_suite_attestation_frames(
                    binary_frames,
                    label=entry["role"],
                )
                policy, policy_bindings, _producer_binding = _load_v2_main_suite_policy(
                    repo_root=repo_root,
                    schemas=schemas,
                )
                derived_claims = _validate_v2_main_suite_semantics(
                    receipt,
                    streams=parsed_streams,
                    frames=decoded_frames,
                    repo_root=repo_root,
                    policy=policy,
                    policy_bindings=policy_bindings,
                    session_binding=legacy_context["session_binding"],
                    skip_baseline=legacy_context["skip"],
                    skip_baseline_raw=legacy_context["skip_raw"],
                    base_commit=legacy_context["base_commit"],
                    source_state=legacy_context["source_state"],
                    package_source_full=legacy_context["package_source_full"],
                )
                if entry["claims"] != derived_claims:
                    raise Phase0EvidenceError(
                        "full_offline_suite readback claims differ from dedicated receipt"
                    )
                semantic_raw = parsed_streams["stdout"] + parsed_streams["stderr"]
            elif entry["role"] in LOG_ROLES:
                receipt, semantic_raw = _parse_command_receipt_log(
                    raw,
                    role=entry["role"],
                    repo_root=repo_root,
                    current_source_binding=source_binding,
                )
                _validate_command_identity(entry["role"], receipt, repo_root=repo_root)
                if entry["role"] == "native_sync_log":
                    native_log_receipt = receipt
                python_executable = _python_executable_from_receipt(entry["role"], receipt)
                if python_executable is not None:
                    python_executables.add(python_executable)
                native_python = _native_sync_python_from_receipt(entry["role"], receipt)
                if native_python is not None:
                    native_sync_python = native_python
                for tool, version in _tool_versions_from_receipt(entry["role"], receipt).items():
                    tool_versions.setdefault(tool, set()).add(version)
            _validate_gate_semantics(
                role=entry["role"],
                claims=entry["claims"],
                raw=semantic_raw,
                current_source_binding=source_binding,
                current_source_state=source,
                repo_root=repo_root,
            )
            if entry["role"] == "native_sync_receipt":
                payload = _load_canonical_json_resource(raw, label=entry["role"])
                native_receipt_payload = payload
                fresh_venv, fresh_python = _fresh_python_binding_from_native_receipt(
                    payload,
                    repo_root=repo_root,
                )
            if entry["role"] == "native_sync_log" and fresh_venv is not None:
                receipt, _output = _parse_command_receipt_log(
                    raw,
                    role=entry["role"],
                    repo_root=repo_root,
                    current_source_binding=source_binding,
                )
                if receipt["commands"][0]["env"]["UV_PROJECT_ENVIRONMENT"] != fresh_venv:
                    raise Phase0EvidenceError("native_sync_log fresh environment mismatch")
        if fresh_python is not None and any(
            python_executable != fresh_python for python_executable in python_executables
        ):
            raise Phase0EvidenceError("log receipts do not use dependency fresh Python")
        if len(python_executables) != 1:
            raise Phase0EvidenceError("log receipts must bind one shared CPython 3.13 executable")
        if (
            fresh_python is not None
            and native_sync_python is not None
            and not _same_resolved_path(native_sync_python, fresh_python)
        ):
            raise Phase0EvidenceError("native_sync_log native Python binding mismatch")
        for versions in tool_versions.values():
            if len(versions) != 1:
                raise Phase0EvidenceError("log receipts bind inconsistent tool versions")
        if native_receipt_payload is None or native_log_receipt is None:
            raise Phase0EvidenceError("native dependency receipt/log binding is incomplete")
        _validate_native_tool_bindings(
            native_receipt_payload,
            native_log_receipt,
            tool_versions,
        )
    return dict(report)


def _open_private_parent_fd(parent: Path, *, label: str) -> int:
    checked = _private_parent(parent / "placeholder", label=label)
    descriptor = -1
    try:
        descriptor = os.open(
            checked,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        _assert_private_parent_fd_current(checked, descriptor, label=label)
        return descriptor
    except (OSError, Phase0EvidenceError) as exc:
        if descriptor >= 0:
            os.close(descriptor)
        if isinstance(exc, Phase0EvidenceError):
            raise
        raise Phase0EvidenceError(f"{label} parent cannot be opened safely") from exc


def _assert_private_parent_fd_current(parent: Path, descriptor: int, *, label: str) -> None:
    try:
        path_stat = parent.lstat()
        descriptor_stat = os.fstat(descriptor)
        resolved = parent.resolve(strict=True)
    except OSError as exc:
        raise Phase0EvidenceError(f"{label} parent changed during publication") from exc
    path_identity = (
        path_stat.st_dev,
        path_stat.st_ino,
        path_stat.st_mode,
        path_stat.st_uid,
    )
    descriptor_identity = (
        descriptor_stat.st_dev,
        descriptor_stat.st_ino,
        descriptor_stat.st_mode,
        descriptor_stat.st_uid,
    )
    if (
        resolved != parent
        or path_identity != descriptor_identity
        or not stat.S_ISDIR(descriptor_stat.st_mode)
        or stat.S_IMODE(descriptor_stat.st_mode) != 0o700
        or descriptor_stat.st_uid != os.getuid()
    ):
        raise Phase0EvidenceError(f"{label} parent changed during publication")


def _entry_exists_at(parent_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise Phase0EvidenceError(f"cannot inspect output entry: {name}") from exc
    return True


def _stage_private_file_at(parent_fd: int, name: str, raw: bytes) -> str:
    temporary = f".{name}.{os.getpid()}.{os.urandom(12).hex()}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        os.fchmod(descriptor, 0o600)
        written = 0
        while written < len(raw):
            count = os.write(descriptor, raw[written:])
            if count <= 0:
                raise Phase0EvidenceError("staged output write made no progress")
            written += count
        os.fsync(descriptor)
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or stat.S_IMODE(observed.st_mode) != 0o600
            or observed.st_size != len(raw)
        ):
            raise Phase0EvidenceError("staged output identity mismatch")
    except (OSError, Phase0EvidenceError) as exc:
        if descriptor >= 0:
            os.close(descriptor)
            descriptor = -1
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        except OSError:
            pass
        if isinstance(exc, Phase0EvidenceError):
            raise
        raise Phase0EvidenceError("cannot stage private output") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return temporary


def _unlink_entry_if_same(
    parent_fd: int,
    *,
    installed_name: str,
    staged_name: str,
) -> None:
    try:
        installed = os.stat(
            installed_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    try:
        staged = os.stat(
            staged_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError as exc:
        raise Phase0EvidenceError("staged publication identity disappeared") from exc
    if (installed.st_dev, installed.st_ino) != (staged.st_dev, staged.st_ino):
        raise Phase0EvidenceError("installed publication identity drifted")
    try:
        os.unlink(installed_name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except OSError as exc:
        raise Phase0EvidenceError("cannot roll back uncommitted evidence publication") from exc


def _stable_private_file_at(
    parent_fd: int,
    name: str,
    *,
    expected_nlink: int = 1,
) -> tuple[bytes, os.stat_result]:
    if type(expected_nlink) is not int or expected_nlink < 1:
        raise Phase0EvidenceError("published output expected link count is invalid")
    descriptor = -1
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after_open = os.fstat(descriptor)
        after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise Phase0EvidenceError(f"cannot read published output: {name}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    signature = _stat_signature(before)
    if (
        signature != _stat_signature(opened)
        or signature != _stat_signature(after_open)
        or signature != _stat_signature(after)
        or not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o600
        or before.st_uid != os.getuid()
        or before.st_nlink != expected_nlink
    ):
        raise Phase0EvidenceError(f"published output identity mismatch: {name}")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise Phase0EvidenceError(f"published output size mismatch: {name}")
    return raw, before


def _assert_report_source_state_current(
    report: Mapping[str, Any],
    *,
    repo_root: Path,
) -> None:
    current = _public_source_state(_git_snapshot(repo_root, str(report["base_commit"])))
    if _canonical_bytes(current) != _canonical_bytes(report["source_state"]):
        raise Phase0EvidenceError("repository source state drifted before publication")


def _prepare_output_pair(
    output_json: Path,
    output_sha256: Path,
    *,
    repo_root: Path,
) -> tuple[Path, Path, Path]:
    output = output_json.absolute()
    sidecar = output_sha256.absolute()
    if output == sidecar or _casefold_ascii(str(output)) == _casefold_ascii(str(sidecar)):
        raise Phase0EvidenceError("output and sidecar paths collide")
    output_parent = _private_parent(output, label="output")
    sidecar_parent = _private_parent(sidecar, label="sidecar")
    if output_parent != sidecar_parent:
        raise Phase0EvidenceError("output and sidecar must share one private parent")
    if _path_within(output, repo_root) or _path_within(sidecar, repo_root):
        raise Phase0EvidenceError("evidence outputs must be outside the repository")
    if os.path.lexists(output) or os.path.lexists(sidecar):
        raise Phase0EvidenceError("evidence output and sidecar must both be absent")
    return output, sidecar, output_parent


def write_evidence_index_exact_once(
    *,
    output_json: Path,
    output_sha256: Path,
    repo_root: Path,
    report: Mapping[str, Any],
) -> str:
    """Publish the 0600 canonical report and byte-SHA sidecar exact-once."""

    validate_evidence_index(report, verify_external=True)
    output, sidecar, parent = _prepare_output_pair(
        output_json,
        output_sha256,
        repo_root=repo_root,
    )
    report_raw = _canonical_resource_bytes(report)
    report_sha = _sha256(report_raw)
    sidecar_raw = f"{report_sha}  {output.name}\n".encode("ascii")
    parent_fd = _open_private_parent_fd(parent, label="output")
    report_temp: str | None = None
    sidecar_temp: str | None = None
    sidecar_linked = False
    report_linked = False
    publication_error: BaseException | None = None
    try:
        _assert_private_parent_fd_current(parent, parent_fd, label="output")
        if _entry_exists_at(parent_fd, output.name) or _entry_exists_at(
            parent_fd,
            sidecar.name,
        ):
            raise Phase0EvidenceError("evidence output and sidecar must both be absent")
        report_temp = _stage_private_file_at(parent_fd, output.name, report_raw)
        _assert_private_parent_fd_current(parent, parent_fd, label="output")
        sidecar_temp = _stage_private_file_at(parent_fd, sidecar.name, sidecar_raw)
        _assert_private_parent_fd_current(parent, parent_fd, label="output")
        _assert_report_source_state_current(report, repo_root=repo_root)
        validate_evidence_index(report, verify_external=True)
        _assert_private_parent_fd_current(parent, parent_fd, label="output")
        try:
            os.link(
                sidecar_temp,
                sidecar.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            sidecar_linked = True
            os.fsync(parent_fd)
            _assert_private_parent_fd_current(parent, parent_fd, label="output")
            os.link(
                report_temp,
                output.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            report_linked = True
            os.fsync(parent_fd)
            _assert_private_parent_fd_current(parent, parent_fd, label="output")
            _assert_report_source_state_current(report, repo_root=repo_root)
            validate_evidence_index(report, verify_external=True)
        except (OSError, Phase0EvidenceError) as exc:
            if report_linked and report_temp is not None:
                _unlink_entry_if_same(
                    parent_fd,
                    installed_name=output.name,
                    staged_name=report_temp,
                )
                report_linked = False
            if sidecar_linked and sidecar_temp is not None:
                _unlink_entry_if_same(
                    parent_fd,
                    installed_name=sidecar.name,
                    staged_name=sidecar_temp,
                )
                sidecar_linked = False
            if isinstance(exc, Phase0EvidenceError):
                raise
            raise Phase0EvidenceError("evidence output exact-once publication failed") from exc
    except BaseException as exc:
        publication_error = exc
        raise
    finally:
        try:
            for temporary in (report_temp, sidecar_temp):
                if temporary is None:
                    continue
                try:
                    os.unlink(temporary, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
            os.fsync(parent_fd)
        except BaseException:
            os.close(parent_fd)
            raise
        if publication_error is not None:
            os.close(parent_fd)
    try:
        _assert_private_parent_fd_current(parent, parent_fd, label="output")
        for name, expected_raw in (
            (output.name, report_raw),
            (sidecar.name, sidecar_raw),
        ):
            raw, _observed = _stable_private_file_at(parent_fd, name)
            if raw != expected_raw:
                raise Phase0EvidenceError(f"published output readback mismatch: {name}")
    finally:
        os.close(parent_fd)
    return report_sha


_LOCAL_MODULE_CACHE: dict[tuple[str, str], Any] = {}


def _load_local_module(path: Path, *, module_name: str) -> Any:
    key = (str(path), module_name)
    cached = _LOCAL_MODULE_CACHE.get(key)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise Phase0EvidenceError(f"cannot load local module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    prior_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.dont_write_bytecode = prior_dont_write_bytecode
    _LOCAL_MODULE_CACHE[key] = module
    return module


def _schema_validation_module(repo_root: Path) -> Any:
    package_name = "_phase0_v17_v2_contract"
    package_root = repo_root / "quant_investor/v17_v2_contract"
    cached = sys.modules.get(f"{package_name}.schema_validation")
    if cached is not None:
        return cached
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__package__ = package_name
        package.__path__ = [str(package_root)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package
    try:
        return importlib.import_module(f"{package_name}.schema_validation")
    except (ImportError, OSError) as exc:
        raise Phase0EvidenceError("closed schema executor is unavailable") from exc


def _load_v2_schema_registry(
    repo_root: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, bytes]]:
    """Stable-read and preflight the complete, separately identified schema set."""

    try:
        schema_module = _schema_validation_module(repo_root)
        schema_error = schema_module.SchemaValidationError
        preflight_packaged_schema = schema_module.preflight_packaged_schema
    except (AttributeError, ImportError, OSError) as exc:
        raise Phase0EvidenceError("closed schema executor is unavailable") from exc
    schemas: dict[str, dict[str, Any]] = {}
    bindings: list[dict[str, Any]] = []
    raw_by_id: dict[str, bytes] = {}
    seen_ids: list[str] = []
    for artifact_version, relative_path, schema_id in SCHEMA_REGISTRY:
        schema_path = _safe_repo_entry(repo_root, relative_path)
        raw, _observed = _stable_regular_file(schema_path, require_private=False)
        if raw.startswith(b"\xef\xbb\xbf"):
            raise Phase0EvidenceError(f"schema BOM is forbidden: {relative_path}")
        try:
            schema = json.loads(
                raw.decode("utf-8", errors="strict"),
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except Phase0EvidenceError:
            raise
        except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise Phase0EvidenceError(f"schema is invalid JSON: {relative_path}") from exc
        properties = schema.get("properties") if type(schema) is dict else None
        artifact_property = (
            "schema_version" if artifact_version == DEPENDENCY_RECEIPT_VERSION else "version"
        )
        if (
            type(schema) is not dict
            or type(properties) is not dict
            or schema.get("$id") != schema_id
            or properties.get(artifact_property) != {"const": artifact_version}
            or artifact_version == schema_id
        ):
            raise Phase0EvidenceError(f"schema identity mismatch: {relative_path}")
        try:
            preflight_packaged_schema(schema)
        except schema_error as exc:
            raise Phase0EvidenceError(f"schema preflight failed: {relative_path}: {exc}") from exc
        schemas[artifact_version] = schema
        raw_by_id[schema_id] = raw
        bindings.append(
            {
                "artifact_version": artifact_version,
                "path": relative_path,
                "schema_id": schema_id,
                "sha256": _sha256(raw),
                "size_bytes": len(raw),
            }
        )
        seen_ids.extend((artifact_version, schema_id))
    _require_unique_casefold(seen_ids, label="artifact and schema identifiers")
    schemas["__repo_root__"] = {"path": str(repo_root)}
    return schemas, bindings, raw_by_id


def _assert_v2_schema_registry_stable(
    repo_root: Path,
    expected_bindings: Sequence[Mapping[str, Any]],
    expected_raw: Mapping[str, bytes],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, bytes]]:
    schemas, bindings, raw_by_id = _load_v2_schema_registry(repo_root)
    if list(expected_bindings) != bindings or dict(expected_raw) != raw_by_id:
        raise Phase0EvidenceError("Phase 0 schema registry changed during sealing")
    return schemas, bindings, raw_by_id


def _validate_v2_schema(
    payload: Mapping[str, Any],
    *,
    artifact_version: str,
    schemas: Mapping[str, Mapping[str, Any]],
    label: str,
) -> None:
    try:
        schema_meta = schemas.get("__repo_root__")
        if type(schema_meta) is not dict or type(schema_meta.get("path")) is not str:
            raise AttributeError
        schema_module = _schema_validation_module(Path(schema_meta["path"]))
        schema_error = schema_module.SchemaValidationError
        validate_instance_against_schema = schema_module.validate_instance_against_schema
    except (AttributeError, ImportError, OSError) as exc:
        raise Phase0EvidenceError("closed schema executor is unavailable") from exc
    schema = schemas.get(artifact_version)
    if schema is None:
        raise Phase0EvidenceError(f"{label} uses an unregistered artifact version")
    try:
        validate_instance_against_schema(dict(payload), dict(schema))
    except schema_error as exc:
        raise Phase0EvidenceError(f"{label} schema validation failed: {exc}") from exc


def _validate_v2_seal(payload: Mapping[str, Any], *, label: str) -> None:
    semantic_sha256 = payload.get(SEMANTIC_FIELD)
    _require_sha256(semantic_sha256, label=f"{label}.semantic_sha256")
    if semantic_sha256 != _semantic_sha256(payload):
        raise Phase0EvidenceError(f"{label} semantic SHA-256 mismatch")


def _v2_file_binding(
    path: Path,
    *,
    repo_root: Path,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    return _external_file_binding(path, repo_root=repo_root, label=label)


def _load_v2_resource(
    path: Path,
    *,
    artifact_version: str,
    schemas: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    binding, raw = _v2_file_binding(path, repo_root=repo_root, label=label)
    payload = _load_canonical_json_resource(raw, label=label)
    identity_field = (
        "schema_version" if artifact_version == DEPENDENCY_RECEIPT_VERSION else "version"
    )
    if payload.get(identity_field) != artifact_version:
        raise Phase0EvidenceError(f"{label} artifact identity mismatch")
    _validate_v2_schema(
        payload,
        artifact_version=artifact_version,
        schemas=schemas,
        label=label,
    )
    _validate_v2_seal(payload, label=label)
    return payload, binding, raw


def _v2_session_binding(
    payload: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    session_id = _require_string(payload.get("session_id"), label="session.session_id")
    semantic_sha256 = _require_sha256(
        payload.get("semantic_sha256"),
        label="session.semantic_sha256",
    )
    return {
        "path": binding["path"],
        "semantic_sha256": semantic_sha256,
        "session_id": session_id,
        "sha256": binding["sha256"],
        "size_bytes": binding["size_bytes"],
    }


def _validate_v2_session_binding(
    value: Any,
    *,
    expected: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    binding = _require_exact_keys(
        value,
        {"path", "semantic_sha256", "session_id", "sha256", "size_bytes"},
        label=label,
    )
    _require_absolute_path(binding["path"], label=f"{label}.path")
    _require_sha256(binding["semantic_sha256"], label=f"{label}.semantic_sha256")
    _require_string(binding["session_id"], label=f"{label}.session_id")
    _require_sha256(binding["sha256"], label=f"{label}.sha256")
    _require_int(binding["size_bytes"], label=f"{label}.size_bytes", minimum=1)
    if binding != dict(expected):
        raise Phase0EvidenceError(f"{label} mismatch")
    return binding


def _validate_v2_namespace_binding(value: Any, *, label: str) -> dict[str, Any]:
    binding = _require_exact_keys(value, {"row_count", "sha256"}, label=label)
    _require_int(binding["row_count"], label=f"{label}.row_count", minimum=1)
    _require_sha256(binding["sha256"], label=f"{label}.sha256")
    return binding


def _v2_live_producer_binding(path: Path, *, version: str) -> dict[str, Any]:
    raw, _observed = _stable_regular_file(path, require_private=False)
    return {
        "path": str(path),
        "sha256": _sha256(raw),
        "size_bytes": len(raw),
        "version": version,
    }


def _validate_v2_producer_binding(
    value: Any,
    *,
    expected_path: Path | None,
    expected_version: str | None,
    label: str,
) -> dict[str, Any]:
    binding = _require_exact_keys(
        value,
        {"path", "sha256", "size_bytes", "version"},
        label=label,
    )
    path = Path(_require_absolute_path(binding["path"], label=f"{label}.path"))
    _require_sha256(binding["sha256"], label=f"{label}.sha256")
    _require_int(binding["size_bytes"], label=f"{label}.size_bytes", minimum=1)
    _require_string(binding["version"], label=f"{label}.version")
    if expected_path is not None and path != expected_path:
        raise Phase0EvidenceError(f"{label} path mismatch")
    if expected_version is not None and binding["version"] != expected_version:
        raise Phase0EvidenceError(f"{label} version mismatch")
    current = _v2_live_producer_binding(path, version=str(binding["version"]))
    if current != binding:
        raise Phase0EvidenceError(f"{label} live readback mismatch")
    return binding


def _sample_v2_protected_roots() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root_id, path in PROTECTED_ROOT_SPECS:
        try:
            first = path.lstat()
        except FileNotFoundError:
            rows.append({"id": root_id, "path": str(path), "state": "ABSENT"})
            continue
        except OSError as exc:
            raise Phase0EvidenceError(f"cannot sample protected root: {path}") from exc
        try:
            second = path.lstat()
            realpath = path.resolve(strict=True)
        except OSError as exc:
            raise Phase0EvidenceError(f"protected root changed during sample: {path}") from exc
        if (
            _stat_signature(first) != _stat_signature(second)
            or stat.S_ISLNK(first.st_mode)
            or not stat.S_ISDIR(first.st_mode)
        ):
            raise Phase0EvidenceError(
                f"protected root must be absent or a concrete directory: {path}"
            )
        rows.append(
            {
                "ctime_ns": first.st_ctime_ns,
                "id": root_id,
                "mode": f"{stat.S_IMODE(first.st_mode):04o}",
                "mtime_ns": first.st_mtime_ns,
                "path": str(path),
                "realpath": str(realpath),
                "st_dev": first.st_dev,
                "st_ino": first.st_ino,
                "state": "PRESENT_DIRECTORY",
                "uid": first.st_uid,
            }
        )
    return rows


def _validate_v2_protected_roots(value: Any, *, label: str) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) != len(PROTECTED_ROOT_SPECS):
        raise Phase0EvidenceError(f"{label} must contain the exact four protected roots")
    rows: list[dict[str, Any]] = []
    for index, ((expected_id, expected_path), raw_row) in enumerate(
        zip(PROTECTED_ROOT_SPECS, value, strict=True)
    ):
        row_label = f"{label}[{index}]"
        if type(raw_row) is not dict:
            raise Phase0EvidenceError(f"{row_label} must be an object")
        state = raw_row.get("state")
        expected_keys = (
            {"id", "path", "state"}
            if state == "ABSENT"
            else {
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
            }
        )
        row = _require_exact_keys(raw_row, expected_keys, label=row_label)
        if (
            row["id"] != expected_id
            or row["path"] != str(expected_path)
            or state not in {"ABSENT", "PRESENT_DIRECTORY"}
        ):
            raise Phase0EvidenceError(f"{row_label} identity mismatch")
        if state == "PRESENT_DIRECTORY":
            _require_absolute_path(row["realpath"], label=f"{row_label}.realpath")
            if (
                type(row["mode"]) is not str
                or re.fullmatch(r"[0-7]{4}", row["mode"], re.ASCII) is None
            ):
                raise Phase0EvidenceError(f"{row_label}.mode is invalid")
            for key, minimum in (
                ("ctime_ns", 0),
                ("mtime_ns", 0),
                ("st_dev", 0),
                ("st_ino", 1),
                ("uid", 0),
            ):
                _require_int(row[key], label=f"{row_label}.{key}", minimum=minimum)
        rows.append(dict(row))
    return rows


def _validate_v2_toolchain(value: Any, *, label: str, live: bool) -> dict[str, Any]:
    toolchain = _require_exact_keys(
        value,
        {"base_python", "pip_scope", "uv", "uv_cache"},
        label=label,
    )
    base = _require_exact_keys(
        toolchain["base_python"],
        {
            "executable",
            "implementation",
            "lexical_path",
            "mode",
            "realpath",
            "sha256",
            "size_bytes",
            "version",
            "version_info",
        },
        label=f"{label}.base_python",
    )
    if (
        base["lexical_path"]
        != (
            "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/"
            "Python.framework/Versions/3.13/bin/python3.13"
        )
        or base["sha256"] != "a708f6e9f4803b806b29146c4e0feecfd9bf2d9eb60f3e15b850cd7cb56f200b"
        or base["size_bytes"] != 52_640
        or base["mode"] != "0755"
        or base["executable"] is not True
        or base["implementation"] != "cpython"
        or base["version"] != "3.13.7"
        or base["version_info"] != [3, 13, 7]
    ):
        raise Phase0EvidenceError(f"{label}.base_python frozen identity mismatch")
    _require_absolute_path(base["realpath"], label=f"{label}.base_python.realpath")
    uv = _require_exact_keys(
        toolchain["uv"],
        {
            "executable",
            "lexical_path",
            "mode",
            "output",
            "realpath",
            "sha256",
            "size_bytes",
            "version",
        },
        label=f"{label}.uv",
    )
    if (
        uv["sha256"] != "bc50ab0e90f24491f0e794f5b8649722f8fd2bf483c53490c012b41b89151ef9"
        or uv["size_bytes"] != 44_698_848
        or uv["mode"] != "0755"
        or uv["executable"] is not True
        or uv["version"] != "0.10.9"
        or uv["output"] != PACKAGE_EXPECTED_UV_OUTPUT
    ):
        raise Phase0EvidenceError(f"{label}.uv frozen identity mismatch")
    _require_absolute_path(uv["lexical_path"], label=f"{label}.uv.lexical_path")
    _require_absolute_path(uv["realpath"], label=f"{label}.uv.realpath")
    cache = _require_exact_keys(
        toolchain["uv_cache"],
        {"mode", "path", "realpath", "st_dev", "st_ino", "uid"},
        label=f"{label}.uv_cache",
    )
    if (
        type(cache["mode"]) is not str
        or re.fullmatch(r"0[0-7]{3}", cache["mode"], re.ASCII) is None
        or int(cache["mode"], 8) & 0o022
    ):
        raise Phase0EvidenceError(f"{label}.uv_cache mode is unsafe")
    _require_absolute_path(cache["path"], label=f"{label}.uv_cache.path")
    _require_absolute_path(cache["realpath"], label=f"{label}.uv_cache.realpath")
    for key, minimum in (("st_dev", 0), ("st_ino", 1), ("uid", 0)):
        _require_int(cache[key], label=f"{label}.uv_cache.{key}", minimum=minimum)
    if toolchain["pip_scope"] != EXPECTED_PIP_SCOPE:
        raise Phase0EvidenceError(f"{label}.pip_scope mismatch")
    if live:
        for item_label, item in (("base_python", base), ("uv", uv)):
            path = Path(str(item["lexical_path"]))
            raw, observed = _stable_regular_file(path, require_private=False)
            if (
                str(path.resolve(strict=True)) != item["realpath"]
                or _sha256(raw) != item["sha256"]
                or len(raw) != item["size_bytes"]
                or f"{stat.S_IMODE(observed.st_mode):04o}" != item["mode"]
                or not observed.st_mode & 0o111
            ):
                raise Phase0EvidenceError(f"{label}.{item_label} live readback mismatch")
        cache_path = Path(str(cache["path"]))
        try:
            observed_cache = cache_path.lstat()
            resolved_cache = cache_path.resolve(strict=True)
        except OSError as exc:
            raise Phase0EvidenceError(f"{label}.uv_cache live readback failed") from exc
        if (
            not stat.S_ISDIR(observed_cache.st_mode)
            or stat.S_ISLNK(observed_cache.st_mode)
            or str(resolved_cache) != cache["realpath"]
            or observed_cache.st_dev != cache["st_dev"]
            or observed_cache.st_ino != cache["st_ino"]
            or observed_cache.st_uid != cache["uid"]
            or f"{stat.S_IMODE(observed_cache.st_mode):04o}" != cache["mode"]
        ):
            raise Phase0EvidenceError(f"{label}.uv_cache live readback mismatch")
    return dict(toolchain)


def _parse_classification_manifest_v2(
    raw: bytes,
    *,
    base_commit: str,
    schemas: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    payload = _load_canonical_json_resource(raw, label="classification manifest")
    _validate_v2_schema(
        payload,
        artifact_version=CLASSIFICATION_VERSION,
        schemas=schemas,
        label="classification manifest",
    )
    _validate_v2_seal(payload, label="classification manifest")
    if (
        payload.get("version") != CLASSIFICATION_VERSION
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("base_commit") != base_commit
        or payload.get("provenance") != CLASSIFICATION_PROVENANCE
    ):
        raise Phase0EvidenceError("classification manifest identity mismatch")
    entries = payload.get("entries")
    if type(entries) is not list:
        raise Phase0EvidenceError("classification entries must be an array")
    paths: list[str] = []
    for index, value in enumerate(entries):
        entry = _require_exact_keys(
            value,
            {"classification", "path"},
            label=f"classification entries[{index}]",
        )
        if entry["classification"] != PRE_EXISTING_CLASSIFICATION:
            raise Phase0EvidenceError("classification must be PRE_EXISTING_NON_PHASE0")
        paths.append(
            _repo_relative_path(entry["path"], label=f"classification entries[{index}].path")
        )
    _require_unique_casefold(paths, label="pre-existing classification paths")
    if paths != sorted(paths, key=lambda item: item.encode("utf-8")):
        raise Phase0EvidenceError("classification entries are not canonically ordered")
    return paths


def _parse_framed_command_receipt_v2(
    raw: bytes,
    *,
    schemas: Mapping[str, Mapping[str, Any]],
    label: str,
) -> tuple[dict[str, Any], list[tuple[bytes, bytes]], bytes]:
    line, separator, framed = raw.partition(b"\n")
    if not separator or not line.startswith(COMMAND_RECEIPT_PREFIX):
        raise Phase0EvidenceError(f"{label} missing command receipt envelope")
    receipt_raw = line[len(COMMAND_RECEIPT_PREFIX) :]
    receipt = _load_canonical_json_resource(receipt_raw + b"\n", label=f"{label} receipt")
    if _canonical_bytes(receipt) != receipt_raw:
        raise Phase0EvidenceError(f"{label} receipt line is not canonical compact JSON")
    if receipt.get("version") != COMMAND_RECEIPT_VERSION:
        raise Phase0EvidenceError(f"{label} command receipt identity mismatch")
    _validate_v2_schema(
        receipt,
        artifact_version=COMMAND_RECEIPT_VERSION,
        schemas=schemas,
        label=f"{label} command receipt",
    )
    _validate_v2_seal(receipt, label=f"{label} command receipt")
    if (
        receipt.get("protocol_version") != PROTOCOL_VERSION
        or receipt.get("framing") != COMMAND_FRAMING
        or receipt.get("outcome") != "PASSED"
        or receipt.get("failure_codes") != []
        or receipt.get("limitations") != NORMATIVE_LIMITATIONS
        or receipt.get("output_sha256") != _sha256(framed)
        or receipt.get("output_size_bytes") != len(framed)
        or len(raw) >= MAX_EXTERNAL_BYTES
    ):
        raise Phase0EvidenceError(f"{label} command receipt binding mismatch")
    commands = receipt.get("commands")
    if type(commands) is not list or not commands:
        raise Phase0EvidenceError(f"{label} commands must be nonempty")
    streams: list[tuple[bytes, bytes]] = []
    offset = 0
    for index, command_value in enumerate(commands, start=1):
        command = _require_exact_keys(
            command_value,
            {
                "argv",
                "cwd",
                "environment",
                "exit_code",
                "ordinal",
                "signal",
                "stderr_offset_bytes",
                "stderr_sha256",
                "stderr_size_bytes",
                "stdout_offset_bytes",
                "stdout_sha256",
                "stdout_size_bytes",
                "tool_version",
            },
            label=f"{label}.commands[{index - 1}]",
        )
        if command["ordinal"] != index:
            raise Phase0EvidenceError(f"{label} command ordinals are not canonical")
        stdout_size = _require_int(
            command["stdout_size_bytes"],
            label=f"{label}.commands[{index - 1}].stdout_size_bytes",
        )
        stderr_size = _require_int(
            command["stderr_size_bytes"],
            label=f"{label}.commands[{index - 1}].stderr_size_bytes",
        )
        if (
            stdout_size > MAX_COMMAND_STREAM_BYTES
            or stderr_size > MAX_COMMAND_STREAM_BYTES
            or stdout_size + stderr_size > MAX_COMMAND_BYTES
        ):
            raise Phase0EvidenceError(f"{label} command stream limit exceeded")
        if len(framed) - offset < 8:
            raise Phase0EvidenceError(f"{label} stdout frame is truncated")
        framed_stdout_size = struct.unpack(">Q", framed[offset : offset + 8])[0]
        stdout_offset = offset + 8
        if framed_stdout_size != stdout_size or command["stdout_offset_bytes"] != stdout_offset:
            raise Phase0EvidenceError(f"{label} stdout frame binding mismatch")
        stdout_end = stdout_offset + stdout_size
        if stdout_end > len(framed):
            raise Phase0EvidenceError(f"{label} stdout frame is truncated")
        stdout = framed[stdout_offset:stdout_end]
        offset = stdout_end
        if len(framed) - offset < 8:
            raise Phase0EvidenceError(f"{label} stderr frame is truncated")
        framed_stderr_size = struct.unpack(">Q", framed[offset : offset + 8])[0]
        stderr_offset = offset + 8
        if framed_stderr_size != stderr_size or command["stderr_offset_bytes"] != stderr_offset:
            raise Phase0EvidenceError(f"{label} stderr frame binding mismatch")
        stderr_end = stderr_offset + stderr_size
        if stderr_end > len(framed):
            raise Phase0EvidenceError(f"{label} stderr frame is truncated")
        stderr = framed[stderr_offset:stderr_end]
        offset = stderr_end
        if (
            command["stdout_sha256"] != _sha256(stdout)
            or command["stderr_sha256"] != _sha256(stderr)
            or command["exit_code"] != 0
            or command["signal"] is not None
        ):
            raise Phase0EvidenceError(f"{label} command outcome binding mismatch")
        streams.append((stdout, stderr))
    if offset != len(framed):
        raise Phase0EvidenceError(f"{label} has trailing framed bytes")
    return receipt, streams, framed


def _parse_framed_main_suite_receipt_v1(
    raw: bytes,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, bytes], bytes]:
    if type(raw) is not bytes or len(raw) >= MAX_EXTERNAL_BYTES:
        raise Phase0EvidenceError(f"{label} main-suite receipt exceeds the input limit")
    line, separator, framed = raw.partition(b"\n")
    if not separator or not line.startswith(MAIN_SUITE_RECEIPT_PREFIX):
        raise Phase0EvidenceError(f"{label} missing main-suite receipt envelope")
    receipt_raw = line[len(MAIN_SUITE_RECEIPT_PREFIX) :]
    receipt = _load_canonical_json_resource(
        receipt_raw + b"\n",
        label=f"{label} main-suite receipt",
    )
    if _canonical_bytes(receipt) != receipt_raw:
        raise Phase0EvidenceError(f"{label} main-suite receipt line is not canonical compact JSON")
    if (
        receipt.get("version") != MAIN_SUITE_RECEIPT_VERSION
        or receipt.get("schema_id") != MAIN_SUITE_RECEIPT_SCHEMA_ID
        or receipt.get("framing") != MAIN_SUITE_FRAMING
    ):
        raise Phase0EvidenceError(f"{label} main-suite receipt identity mismatch")
    streams = _require_exact_keys(
        receipt.get("streams"),
        {
            "attestation",
            "stderr",
            "stdout",
            "tail_sha256",
            "tail_size_bytes",
        },
        label=f"{label}.streams",
    )
    tail_size = _require_int(
        streams["tail_size_bytes"],
        label=f"{label}.streams.tail_size_bytes",
    )
    tail_sha256 = _require_sha256(
        streams["tail_sha256"],
        label=f"{label}.streams.tail_sha256",
    )
    if (
        tail_size != len(framed)
        or tail_sha256 != _sha256(framed)
        or tail_size > MAX_MAIN_SUITE_TAIL_BYTES
    ):
        raise Phase0EvidenceError(f"{label} main-suite tail binding mismatch")

    parsed_streams: dict[str, bytes] = {}
    offset = 0
    for name, maximum in (
        ("stdout", MAX_COMMAND_STREAM_BYTES),
        ("stderr", MAX_COMMAND_STREAM_BYTES),
        ("attestation", MAX_MAIN_SUITE_ATTESTATION_BYTES),
    ):
        binding = _require_exact_keys(
            streams[name],
            {"offset_bytes", "sha256", "size_bytes"},
            label=f"{label}.streams.{name}",
        )
        declared_size = _require_int(
            binding["size_bytes"],
            label=f"{label}.streams.{name}.size_bytes",
        )
        declared_offset = _require_int(
            binding["offset_bytes"],
            label=f"{label}.streams.{name}.offset_bytes",
        )
        declared_sha256 = _require_sha256(
            binding["sha256"],
            label=f"{label}.streams.{name}.sha256",
        )
        if declared_size > maximum:
            raise Phase0EvidenceError(f"{label} main-suite {name} limit exceeded")
        if len(framed) - offset < 8:
            raise Phase0EvidenceError(f"{label} main-suite {name} frame is truncated")
        framed_size = struct.unpack(">Q", framed[offset : offset + 8])[0]
        data_offset = offset + 8
        if framed_size != declared_size or declared_offset != data_offset:
            raise Phase0EvidenceError(f"{label} main-suite {name} frame binding mismatch")
        data_end = data_offset + declared_size
        if data_end > len(framed):
            raise Phase0EvidenceError(f"{label} main-suite {name} frame is truncated")
        stream = framed[data_offset:data_end]
        if declared_sha256 != _sha256(stream):
            raise Phase0EvidenceError(f"{label} main-suite {name} frame binding mismatch")
        parsed_streams[name] = stream
        offset = data_end
    if offset != len(framed):
        raise Phase0EvidenceError(f"{label} main-suite receipt has trailing framed bytes")
    _validate_main_suite_attestation_frames(
        parsed_streams["attestation"],
        label=label,
    )
    return receipt, parsed_streams, framed


def _validate_main_suite_attestation_frames(
    raw: bytes,
    *,
    label: str,
) -> list[dict[str, Any]]:
    offset = 0
    nonce: bytes | None = None
    frames: list[dict[str, Any]] = []
    for expected_phase, maximum in (
        (1, MAX_MAIN_SUITE_FRAME_BYTES),
        (2, MAX_MAIN_SUITE_FRAME_BYTES),
        (3, MAX_MAIN_SUITE_TERMINAL_FRAME_BYTES),
    ):
        if len(raw) - offset < MAIN_SUITE_ATTEST_HEADER.size:
            raise Phase0EvidenceError(
                f"{label} main-suite attestation phase {expected_phase} header is truncated"
            )
        (
            magic,
            version,
            phase,
            reserved,
            payload_size,
            frame_nonce,
            digest,
        ) = MAIN_SUITE_ATTEST_HEADER.unpack(raw[offset : offset + MAIN_SUITE_ATTEST_HEADER.size])
        if (
            magic != MAIN_SUITE_ATTEST_MAGIC
            or version != MAIN_SUITE_ATTEST_PROTOCOL_VERSION
            or phase != expected_phase
            or reserved != 0
            or payload_size < 2
            or payload_size > maximum
        ):
            raise Phase0EvidenceError(
                f"{label} main-suite attestation phase {expected_phase} header mismatch"
            )
        if nonce is None:
            nonce = frame_nonce
        elif frame_nonce != nonce:
            raise Phase0EvidenceError(f"{label} main-suite attestation nonce mismatch")
        payload_offset = offset + MAIN_SUITE_ATTEST_HEADER.size
        payload_end = payload_offset + payload_size
        if payload_end > len(raw):
            raise Phase0EvidenceError(
                f"{label} main-suite attestation phase {expected_phase} is truncated"
            )
        payload = raw[payload_offset:payload_end]
        if hashlib.sha256(payload).digest() != digest:
            raise Phase0EvidenceError(
                f"{label} main-suite attestation phase {expected_phase} digest mismatch"
            )
        frames.append(
            {
                "payload_raw": payload,
                "payload_sha256": digest.hex(),
                "payload_size_bytes": payload_size,
                "phase": expected_phase,
            }
        )
        offset = payload_end
    if offset != len(raw):
        raise Phase0EvidenceError(f"{label} main-suite attestation has trailing bytes")
    return frames


def _decode_main_suite_attestation_frames(
    frames: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> list[dict[str, Any]]:
    decoded: list[dict[str, Any]] = []
    for index, frame in enumerate(frames, start=1):
        raw = frame.get("payload_raw")
        if type(raw) is not bytes:
            raise Phase0EvidenceError(
                f"{label} main-suite attestation phase {index} payload is invalid"
            )
        payload = _load_canonical_json_resource(
            raw + b"\n",
            label=f"{label} main-suite attestation phase {index} payload",
        )
        decoded.append(
            {
                "payload": payload,
                "payload_sha256": frame["payload_sha256"],
                "payload_size_bytes": frame["payload_size_bytes"],
                "phase": frame["phase"],
            }
        )
    return decoded


def _main_suite_live_file_binding(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    raw, observed = _stable_regular_file(path, require_private=False)
    if observed.st_nlink != 1:
        raise Phase0EvidenceError(f"{label} must have exactly one hard link")
    return (
        {
            "gid": observed.st_gid,
            "mode": f"{stat.S_IMODE(observed.st_mode):04o}",
            "path": str(path),
            "sha256": _sha256(raw),
            "size_bytes": len(raw),
            "st_dev": observed.st_dev,
            "st_ino": observed.st_ino,
            "st_nlink": observed.st_nlink,
            "uid": observed.st_uid,
        },
        raw,
    )


def _load_v2_main_suite_policy(
    *,
    repo_root: Path,
    schemas: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    policy_path = _safe_repo_entry(repo_root, MAIN_SUITE_POLICY_PATH)
    schema_path = _safe_repo_entry(repo_root, MAIN_SUITE_POLICY_SCHEMA_PATH)
    manifest_path = _safe_repo_entry(repo_root, MAIN_SUITE_PACKAGE_MANIFEST_PATH)
    harness_path = _safe_repo_entry(repo_root, MAIN_SUITE_HARNESS_PATH)
    wrapper_path = _safe_repo_entry(repo_root, MAIN_SUITE_WRAPPER_PATH)
    conftest_path = _safe_repo_entry(repo_root, "tests/conftest.py")

    policy_binding, policy_raw = _main_suite_live_file_binding(
        policy_path,
        label="main-suite runtime policy",
    )
    policy = _load_canonical_json_resource(
        policy_raw,
        label="main-suite runtime policy",
    )
    _validate_v2_schema(
        policy,
        artifact_version=MAIN_SUITE_RUNTIME_POLICY_VERSION,
        schemas=schemas,
        label="main-suite runtime policy",
    )
    _validate_v2_seal(policy, label="main-suite runtime policy")
    if (
        policy.get("version") != MAIN_SUITE_RUNTIME_POLICY_VERSION
        or policy.get("schema_id") != MAIN_SUITE_RUNTIME_POLICY_SCHEMA_ID
        or policy.get("protocol_version", PROTOCOL_VERSION) != PROTOCOL_VERSION
        or policy.get("authority") is not False
        or policy.get("status") != "FROZEN"
        or policy.get("discovery_mode") is not False
        or policy.get("candidate_root") != str(repo_root)
        or policy.get("limitations") != NORMATIVE_LIMITATIONS
    ):
        raise Phase0EvidenceError("main-suite runtime policy identity mismatch")

    schema_binding, _schema_raw = _main_suite_live_file_binding(
        schema_path,
        label="main-suite runtime policy schema",
    )
    manifest_binding, manifest_raw = _main_suite_live_file_binding(
        manifest_path,
        label="main-suite package manifest",
    )
    harness_binding, _harness_raw = _main_suite_live_file_binding(
        harness_path,
        label="main-suite harness",
    )
    wrapper_binding, _wrapper_raw = _main_suite_live_file_binding(
        wrapper_path,
        label="main-suite wrapper",
    )
    conftest_binding, _conftest_raw = _main_suite_live_file_binding(
        conftest_path,
        label="main-suite candidate conftest",
    )
    if (
        policy.get("harness_binding") != harness_binding
        or policy.get("wrapper_binding") != wrapper_binding
        or policy.get("candidate_conftest") != conftest_binding
    ):
        raise Phase0EvidenceError("main-suite bound producer files mismatch")

    manifest = _load_canonical_json_resource(
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
        manifest.get("version") != MAIN_SUITE_PACKAGE_MANIFEST_VERSION
        or manifest.get("authority") is not False
        or len(resource_rows) != 1
        or resource_rows[0].get("resource_version") != MAIN_SUITE_RUNTIME_POLICY_VERSION
        or resource_rows[0].get("byte_sha256") != policy_binding["sha256"]
        or len(schema_rows) != 1
        or schema_rows[0].get("schema_id") != MAIN_SUITE_RUNTIME_POLICY_SCHEMA_ID
        or schema_rows[0].get("byte_sha256") != schema_binding["sha256"]
    ):
        raise Phase0EvidenceError("main-suite package manifest binding mismatch")

    policy_bindings = {
        "policy_binding": policy_binding,
        "policy_manifest_binding": manifest_binding,
        "policy_schema_binding": schema_binding,
    }
    producer_binding = {
        "path": harness_binding["path"],
        "sha256": harness_binding["sha256"],
        "size_bytes": harness_binding["size_bytes"],
        "version": MAIN_SUITE_RECEIPT_VERSION,
    }
    return policy, policy_bindings, producer_binding


def _base_git_tree_paths(repo_root: Path, base_commit: str) -> set[str]:
    raw = _git_bytes(
        ("git", "ls-tree", "-r", "--name-only", "-z", base_commit, "--"),
        repo_root=repo_root,
    )
    return set(_decode_nul_paths(raw, label="base Git tree paths"))


def _validate_v2_candidate_module_source_membership(
    policy: Mapping[str, Any],
    *,
    repo_root: Path,
    base_commit: str,
    source_state: Mapping[str, Any],
    package_source_full: Mapping[str, Any],
) -> list[str]:
    module_policy = policy.get("module_policy")
    if (
        type(module_policy) is not dict
        or module_policy.get("candidate_content_binding") != "OUTER_SOURCE_STATE"
    ):
        raise Phase0EvidenceError("main-suite candidate content binding mismatch")
    raw_paths = module_policy.get("candidate_module_source_paths")
    if type(raw_paths) is not list or not raw_paths:
        raise Phase0EvidenceError("main-suite candidate module source paths are missing")

    paths: list[str] = []
    for index, value in enumerate(raw_paths):
        relative = _repo_relative_path(
            value,
            label=f"candidate_module_source_paths[{index}]",
        )
        pure = PurePosixPath(relative)
        if pure.suffix != ".py" or pure.as_posix() != relative:
            raise Phase0EvidenceError(
                "main-suite candidate module source path must be a safe .py path"
            )
        paths.append(relative)
    _require_unique_casefold(paths, label="candidate module source paths")
    if paths != sorted(paths, key=lambda value: value.encode("utf-8")):
        raise Phase0EvidenceError("main-suite candidate module source paths are not canonical")

    untracked_rows = source_state.get("untracked")
    physical_rows = package_source_full.get("rows")
    if type(untracked_rows) is not list or type(physical_rows) is not list:
        raise Phase0EvidenceError("candidate source membership inputs are invalid")
    untracked_files: set[str] = set()
    untracked_symlinks: set[str] = set()
    for row in untracked_rows:
        if type(row) is not dict or type(row.get("path")) is not str:
            raise Phase0EvidenceError("sealed untracked source row is invalid")
        if row.get("type") == "file":
            untracked_files.add(str(row["path"]))
        elif row.get("type") == "symlink":
            untracked_symlinks.add(str(row["path"]))
        else:
            raise Phase0EvidenceError("sealed untracked source type is invalid")
    physical_files = {
        str(row["path"])
        for row in physical_rows
        if type(row) is dict and row.get("kind") == "file" and type(row.get("path")) is str
    }
    allowed = _base_git_tree_paths(repo_root, base_commit) | untracked_files | physical_files
    for relative in paths:
        if relative in untracked_symlinks or relative not in allowed:
            raise Phase0EvidenceError(
                f"main-suite candidate module source is not outer-source-bound: {relative}"
            )
        path = _safe_repo_entry(repo_root, relative)
        try:
            observed = path.lstat()
        except OSError as exc:
            raise Phase0EvidenceError(
                f"main-suite candidate module source is unavailable: {relative}"
            ) from exc
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
            raise Phase0EvidenceError(
                f"main-suite candidate module source is not a regular file: {relative}"
            )
        _stable_regular_file(path, require_private=False)
    return paths


def _validate_v2_main_suite_environment(
    value: Any,
    *,
    policy: Mapping[str, Any],
) -> dict[str, str]:
    if type(value) is not dict or not all(
        type(key) is str and type(item) is str for key, item in value.items()
    ):
        raise Phase0EvidenceError("main-suite command environment is invalid")
    environment_policy = policy.get("pytest_environment")
    expected_topology = {
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
    }
    if (
        type(environment_policy) is not dict
        or set(environment_policy)
        != {
            "allowed_keys",
            "dynamic_path_keys",
            "forbidden",
            "path_topology",
            "required",
        }
        or environment_policy.get("path_topology") != expected_topology
    ):
        raise Phase0EvidenceError("main-suite environment policy is invalid")
    required = environment_policy.get("required")
    allowed = environment_policy.get("allowed_keys")
    dynamic = environment_policy.get("dynamic_path_keys")
    forbidden = environment_policy.get("forbidden")
    if (
        type(required) is not dict
        or type(allowed) is not list
        or type(dynamic) is not list
        or type(forbidden) is not list
        or any(type(key) is not str or type(item) is not str for key, item in required.items())
        or any(type(item) is not str for item in [*allowed, *dynamic, *forbidden])
        or len(allowed) != len(set(allowed))
        or len(dynamic) != len(set(dynamic))
        or set(allowed) != set(required) | set(dynamic)
        or set(forbidden) & set(allowed)
        or set(value) != set(allowed)
    ):
        raise Phase0EvidenceError("main-suite environment closure mismatch")
    for key, expected in required.items():
        if value.get(key) != expected:
            raise Phase0EvidenceError(f"main-suite required environment mismatch: {key}")
    for key in dynamic:
        path = Path(value[key])
        if (
            not path.is_absolute()
            or "\0" in value[key]
            or os.path.normpath(value[key]) != value[key]
        ):
            raise Phase0EvidenceError(f"main-suite dynamic environment path is invalid: {key}")
    topology_keys = {
        "BLACK_CACHE_DIR",
        "HOME",
        "MYPY_CACHE_DIR",
        "PYTHONPYCACHEPREFIX",
        "TMPDIR",
        "XDG_CACHE_HOME",
    }
    canonical_dynamic_keys = sorted(
        topology_keys,
        key=lambda item: item.encode("utf-8"),
    )
    if dynamic != canonical_dynamic_keys:
        raise Phase0EvidenceError("main-suite environment dynamic path keys mismatch")
    home = Path(value["HOME"])
    tmpdir = Path(value["TMPDIR"])
    cache = Path(value["XDG_CACHE_HOME"])
    black_cache = Path(value["BLACK_CACHE_DIR"])
    mypy_cache = Path(value["MYPY_CACHE_DIR"])
    pycache = Path(value["PYTHONPYCACHEPREFIX"])
    runtime_root = home.parent
    if (
        tmpdir.parent != runtime_root
        or cache.parent != runtime_root
        or black_cache.parent != cache
        or mypy_cache.parent != cache
        or pycache.parent != cache
        or len({home, tmpdir, cache, black_cache, mypy_cache, pycache}) != 6
    ):
        raise Phase0EvidenceError("main-suite environment topology mismatch")
    for label, path in (
        ("runtime root", runtime_root),
        ("HOME", home),
        ("TMPDIR", tmpdir),
        ("XDG_CACHE_HOME", cache),
        ("BLACK_CACHE_DIR", black_cache),
        ("MYPY_CACHE_DIR", mypy_cache),
        ("PYTHONPYCACHEPREFIX", pycache),
    ):
        try:
            observed = path.lstat()
        except OSError as exc:
            raise Phase0EvidenceError(
                f"main-suite environment path is unavailable: {label}"
            ) from exc
        if (
            not stat.S_ISDIR(observed.st_mode)
            or observed.st_uid != os.getuid()
            or stat.S_IMODE(observed.st_mode) != 0o700
        ):
            raise Phase0EvidenceError(
                f"main-suite environment path is not owner-private 0700: {label}"
            )
    return dict(value)


def _main_suite_live_symlink_binding(
    path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    try:
        before = path.lstat()
        link_text = os.readlink(path)
        after = path.lstat()
    except OSError as exc:
        raise Phase0EvidenceError(f"{label} symlink is unavailable") from exc
    if (
        not stat.S_ISLNK(before.st_mode)
        or _stat_signature(before) != _stat_signature(after)
        or before.st_uid != os.getuid()
    ):
        raise Phase0EvidenceError(f"{label} symlink binding drifted")
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


def _require_main_suite_live_binding(
    expected: Any,
    observed: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if type(expected) is not dict:
        raise Phase0EvidenceError(f"{label} policy binding is invalid")
    if any(expected.get(key) != value for key, value in observed.items()):
        raise Phase0EvidenceError(f"{label} live binding mismatch")


def _live_main_suite_protected_roots(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list:
        raise Phase0EvidenceError("main-suite protected-root policy is invalid")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if (
            type(row) is not dict
            or type(row.get("label")) is not str
            or type(row.get("path")) is not str
            or row.get("state") not in {"ABSENT", "PRESENT_DIRECTORY"}
        ):
            raise Phase0EvidenceError(f"main-suite protected-root policy row {index} is invalid")
        path = Path(str(row["path"]))
        try:
            observed = path.lstat()
        except FileNotFoundError:
            identity = None
            state = "ABSENT"
        except OSError as exc:
            raise Phase0EvidenceError(f"main-suite protected root is unavailable: {path}") from exc
        else:
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
                raise Phase0EvidenceError(
                    f"main-suite protected root is not a concrete directory: {path}"
                )
            identity = {
                "mode": f"{stat.S_IMODE(observed.st_mode):04o}",
                "st_dev": observed.st_dev,
                "st_ino": observed.st_ino,
                "uid": observed.st_uid,
            }
            state = "PRESENT_DIRECTORY"
        projected = {
            "identity": identity,
            "label": row["label"],
            "path": str(path),
            "state": state,
        }
        if projected != row:
            raise Phase0EvidenceError(f"main-suite protected-root live binding mismatch: {path}")
        rows.append(projected)
    return rows


def _phase0_index_tree_inventory(roots: Any) -> dict[str, Any]:
    if type(roots) is not list or not roots:
        raise Phase0EvidenceError("main-suite tree roots are invalid")
    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_casefold: set[str] = set()
    seen_file_identities: set[tuple[int, int]] = set()

    def add_row(
        relative: str,
        observed: os.stat_result,
        *,
        parent_fd: int,
        name: str,
    ) -> None:
        folded = relative.casefold()
        if relative in seen_paths or folded in seen_casefold:
            raise Phase0EvidenceError("main-suite tree path/casefold collision")
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
            raise Phase0EvidenceError("main-suite tree contains a symlink or special entry")
        identity = (observed.st_dev, observed.st_ino)
        if observed.st_nlink != 1 or identity in seen_file_identities:
            raise Phase0EvidenceError("main-suite tree contains a hardlinked file")
        seen_file_identities.add(identity)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = -1
        try:
            descriptor = os.open(name, flags, dir_fd=parent_fd)
            before = os.fstat(descriptor)
            if _stat_signature(before) != _stat_signature(observed):
                raise Phase0EvidenceError(f"main-suite tree file identity drift: {relative}")
            digest = hashlib.sha256()
            size = 0
            while chunk := os.read(descriptor, 1024 * 1024):
                size += len(chunk)
                if size > MAX_EXTERNAL_BYTES:
                    raise Phase0EvidenceError(f"main-suite tree file exceeds cap: {relative}")
                digest.update(chunk)
            after = os.fstat(descriptor)
            path_after = os.stat(
                name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise Phase0EvidenceError(f"cannot read main-suite tree file: {relative}") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if (
            _stat_signature(before) != _stat_signature(after)
            or _stat_signature(after) != _stat_signature(path_after)
            or size != before.st_size
        ):
            raise Phase0EvidenceError(f"main-suite tree file drifted: {relative}")
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
            names = sorted(
                os.listdir(descriptor),
                key=lambda item: item.encode("utf-8"),
            )
        except OSError as exc:
            raise Phase0EvidenceError(
                f"cannot enumerate main-suite tree directory: {relative}"
            ) from exc
        if len(names) != len(set(names)) or len(names) != len({name.casefold() for name in names}):
            raise Phase0EvidenceError("main-suite tree directory contains a name collision")
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        for name in names:
            if not name or name in {".", ".."} or "/" in name or "\0" in name:
                raise Phase0EvidenceError("main-suite tree directory contains an invalid name")
            child_relative = f"{relative}/{name}"
            try:
                observed = os.stat(
                    name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise Phase0EvidenceError(
                    f"cannot stat main-suite tree entry: {child_relative}"
                ) from exc
            add_row(
                child_relative,
                observed,
                parent_fd=descriptor,
                name=name,
            )
            if stat.S_ISDIR(observed.st_mode):
                child_fd = -1
                try:
                    child_fd = os.open(
                        name,
                        directory_flags,
                        dir_fd=descriptor,
                    )
                    opened = os.fstat(child_fd)
                    if _stat_signature(opened) != _stat_signature(observed):
                        raise Phase0EvidenceError(
                            f"main-suite tree directory identity drift: {child_relative}"
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
                        raise Phase0EvidenceError(
                            f"main-suite tree directory drifted: {child_relative}"
                        )
                except OSError as exc:
                    raise Phase0EvidenceError(
                        f"cannot open main-suite tree directory: {child_relative}"
                    ) from exc
                finally:
                    if child_fd >= 0:
                        os.close(child_fd)
        try:
            after_names = sorted(
                os.listdir(descriptor),
                key=lambda item: item.encode("utf-8"),
            )
        except OSError as exc:
            raise Phase0EvidenceError(
                f"cannot re-enumerate main-suite tree directory: {relative}"
            ) from exc
        if names != after_names:
            raise Phase0EvidenceError(f"main-suite tree directory names drifted: {relative}")

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for raw_root in roots:
        if type(raw_root) is not str:
            raise Phase0EvidenceError("main-suite tree root path is invalid")
        root = Path(raw_root)
        try:
            root_stat = root.lstat()
            parent_fd = os.open(str(root.parent), directory_flags)
        except OSError as exc:
            raise Phase0EvidenceError(f"main-suite tree root is unavailable: {root}") from exc
        try:
            add_row(
                root.name,
                root_stat,
                parent_fd=parent_fd,
                name=root.name,
            )
            if stat.S_ISDIR(root_stat.st_mode):
                root_fd = -1
                try:
                    root_fd = os.open(
                        root.name,
                        directory_flags,
                        dir_fd=parent_fd,
                    )
                    opened = os.fstat(root_fd)
                    if _stat_signature(opened) != _stat_signature(root_stat):
                        raise Phase0EvidenceError(f"main-suite tree root identity drift: {root}")
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
                        raise Phase0EvidenceError(f"main-suite tree root drifted: {root}")
                except OSError as exc:
                    raise Phase0EvidenceError(f"cannot open main-suite tree root: {root}") from exc
                finally:
                    if root_fd >= 0:
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
        "total_file_bytes": sum(int(row["size_bytes"]) for row in file_rows),
        "tree_inventory_sha256": _sha256(_canonical_bytes(rows)),
    }


def _live_v2_main_suite_policy_projection(
    policy: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    main_runtime = policy.get("main_runtime")
    module_policy = policy.get("module_policy")
    factor_sources = policy.get("factor_authority_sources")
    plugins = policy.get("pytest_plugins")
    support_trees = policy.get("pytest_support_trees")
    if (
        type(main_runtime) is not dict
        or type(module_policy) is not dict
        or type(factor_sources) is not list
        or type(plugins) is not list
        or type(support_trees) is not list
    ):
        raise Phase0EvidenceError("main-suite live policy inventories are invalid")

    lexical_path = Path(str(main_runtime.get("lexical_python")))
    lexical = _main_suite_live_symlink_binding(
        lexical_path,
        label="main-suite lexical interpreter",
    )
    _require_main_suite_live_binding(
        main_runtime.get("lexical_python_binding"),
        lexical,
        label="main-suite lexical interpreter",
    )
    resolved_path = lexical_path.resolve(strict=True)
    resolved, _resolved_raw = _main_suite_live_file_binding(
        resolved_path,
        label="main-suite resolved interpreter",
    )
    _require_main_suite_live_binding(
        main_runtime.get("resolved_python_binding"),
        resolved,
        label="main-suite resolved interpreter",
    )

    startup_rows = main_runtime.get("startup_files")
    startup_modules = main_runtime.get("startup_modules")
    if type(startup_rows) is not list or type(startup_modules) is not list:
        raise Phase0EvidenceError("main-suite startup policy is invalid")
    startup_projection: list[dict[str, Any]] = []
    for index, row in enumerate(startup_rows):
        if type(row) is not dict or type(row.get("path")) is not str:
            raise Phase0EvidenceError(f"main-suite startup file policy row {index} is invalid")
        path = Path(str(row["path"]))
        if row.get("present") is False:
            try:
                path.lstat()
            except FileNotFoundError:
                startup_projection.append({"path": str(path), "present": False})
                continue
            except OSError as exc:
                raise Phase0EvidenceError(
                    f"main-suite absent startup file is unreadable: {path}"
                ) from exc
            raise Phase0EvidenceError(f"main-suite absent startup file appeared: {path}")
        observed, _raw = _main_suite_live_file_binding(
            path,
            label=f"main-suite startup file {index}",
        )
        _require_main_suite_live_binding(
            row,
            observed,
            label=f"main-suite startup file {index}",
        )
        startup_projection.append({"present": True, **observed})
    for index, row in enumerate(startup_modules):
        if type(row) is not dict or type(row.get("path")) is not str:
            raise Phase0EvidenceError(f"main-suite startup module policy row {index} is invalid")
        observed, _raw = _main_suite_live_file_binding(
            Path(str(row["path"])),
            label=f"main-suite startup module {index}",
        )
        _require_main_suite_live_binding(
            row,
            observed,
            label=f"main-suite startup module {index}",
        )

    invalid_rows = main_runtime.get("invalid_dist_info")
    if type(invalid_rows) is not list:
        raise Phase0EvidenceError("main-suite invalid dist-info policy is invalid")
    invalid_projection: list[dict[str, Any]] = []
    for index, row in enumerate(invalid_rows):
        if (
            type(row) is not dict
            or type(row.get("path")) is not str
            or type(row.get("files")) is not list
        ):
            raise Phase0EvidenceError(f"main-suite invalid dist-info row {index} is invalid")
        root = Path(str(row["path"]))
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        root_fd = -1
        try:
            try:
                root_before = root.lstat()
                if stat.S_ISLNK(root_before.st_mode) or not stat.S_ISDIR(root_before.st_mode):
                    raise Phase0EvidenceError(
                        "main-suite invalid dist-info root is not a concrete " f"directory: {root}"
                    )
                root_fd = os.open(str(root), directory_flags)
                root_opened = os.fstat(root_fd)
                if _stat_signature(root_before) != _stat_signature(root_opened):
                    raise Phase0EvidenceError(
                        f"main-suite invalid dist-info root identity drift: {root}"
                    )
                children = sorted(
                    os.listdir(root_fd),
                    key=lambda item: item.encode("utf-8"),
                )
            except OSError as exc:
                raise Phase0EvidenceError(
                    f"main-suite invalid dist-info root is unavailable: {root}"
                ) from exc
            files: list[dict[str, Any]] = []
            for file_index, expected in enumerate(row["files"]):
                if type(expected) is not dict or type(expected.get("path")) is not str:
                    raise Phase0EvidenceError(
                        f"main-suite invalid dist-info file {index}:{file_index} " "is invalid"
                    )
                expected_path = Path(str(expected["path"]))
                if expected_path.parent != root or expected_path.name not in children:
                    raise Phase0EvidenceError(
                        f"main-suite invalid dist-info file {index}:{file_index} "
                        "escaped its root"
                    )
                observed, _raw = _main_suite_live_file_binding(
                    expected_path,
                    label=f"main-suite invalid dist-info file {index}:{file_index}",
                )
                _require_main_suite_live_binding(
                    expected,
                    observed,
                    label=f"main-suite invalid dist-info file {index}:{file_index}",
                )
                files.append(observed)
            try:
                after_children = sorted(
                    os.listdir(root_fd),
                    key=lambda item: item.encode("utf-8"),
                )
                root_after = os.fstat(root_fd)
                path_after = root.lstat()
            except OSError as exc:
                raise Phase0EvidenceError(
                    f"main-suite invalid dist-info root cannot be revalidated: {root}"
                ) from exc
            if (
                children != after_children
                or _stat_signature(root_opened) != _stat_signature(root_after)
                or _stat_signature(root_after) != _stat_signature(path_after)
            ):
                raise Phase0EvidenceError(f"main-suite invalid dist-info root drifted: {root}")
            projected = {"child_names": children, "files": files, "path": str(root)}
            if projected != row:
                raise Phase0EvidenceError(
                    f"main-suite invalid dist-info live projection mismatch: {root}"
                )
            invalid_projection.append(projected)
        finally:
            if root_fd >= 0:
                os.close(root_fd)

    factor_projection: list[dict[str, Any]] = []
    authority_root = module_policy.get("authority_root")
    if type(authority_root) is not str:
        raise Phase0EvidenceError("main-suite factor authority root is invalid")
    for index, row in enumerate(factor_sources):
        if (
            type(row) is not dict
            or type(row.get("relative_path")) is not str
            or type(row.get("sha256")) is not str
            or type(row.get("size_bytes")) is not int
        ):
            raise Phase0EvidenceError(f"main-suite factor source row {index} is invalid")
        for root in (repo_root, Path(authority_root)):
            target = _safe_repo_entry(root, str(row["relative_path"]))
            raw, _observed = _stable_regular_file(
                target,
                require_private=False,
            )
            if _sha256(raw) != row["sha256"] or len(raw) != row["size_bytes"]:
                raise Phase0EvidenceError(
                    f"main-suite factor source live mismatch: {row['relative_path']}"
                )
        factor_projection.append(dict(row))

    ownership_rows = module_policy.get("distribution_ownership")
    if type(ownership_rows) is not list:
        raise Phase0EvidenceError("main-suite distribution ownership policy is invalid")
    ownership_projection: list[dict[str, Any]] = []
    for index, row in enumerate(ownership_rows):
        if type(row) is not dict:
            raise Phase0EvidenceError(f"main-suite distribution ownership row {index} is invalid")
        metadata = row.get("metadata_binding")
        record = row.get("record_binding")
        if (
            type(metadata) is not dict
            or type(metadata.get("path")) is not str
            or type(record) is not dict
            or type(record.get("path")) is not str
        ):
            raise Phase0EvidenceError(
                f"main-suite distribution ownership bindings {index} are invalid"
            )
        live_metadata, _raw = _main_suite_live_file_binding(
            Path(str(metadata["path"])),
            label=f"main-suite distribution METADATA {index}",
        )
        live_record, _raw = _main_suite_live_file_binding(
            Path(str(record["path"])),
            label=f"main-suite distribution RECORD {index}",
        )
        _require_main_suite_live_binding(
            metadata,
            live_metadata,
            label=f"main-suite distribution METADATA {index}",
        )
        _require_main_suite_live_binding(
            record,
            live_record,
            label=f"main-suite distribution RECORD {index}",
        )
        ownership_projection.append(
            {
                "metadata_sha256": live_metadata["sha256"],
                "name": row.get("name"),
                "record_sha256": live_record["sha256"],
                "version": row.get("version"),
            }
        )

    tree_projection: list[dict[str, Any]] = []
    tree_policies = [
        (row.get("name"), row.get("roots"), row.get("descriptor"))
        for row in support_trees
        if type(row) is dict
    ] + [
        (
            f"plugin:{row.get('entry_point_name')}",
            row.get("physical_tree_roots"),
            row.get("physical_tree"),
        )
        for row in plugins
        if type(row) is dict
    ]
    if len(tree_policies) != len(support_trees) + len(plugins):
        raise Phase0EvidenceError("main-suite physical tree policy is invalid")
    for name, roots, expected in tree_policies:
        observed = _phase0_index_tree_inventory(roots)
        if type(expected) is not dict or observed != expected:
            raise Phase0EvidenceError(f"main-suite physical tree live mismatch: {name}")
        tree_projection.append({"descriptor": observed, "name": name})

    return {
        "distribution_ownership_sha256": _sha256(_canonical_bytes(ownership_projection)),
        "factor_authority_sha256": _sha256(_canonical_bytes(factor_projection)),
        "invalid_dist_info_sha256": _sha256(_canonical_bytes(invalid_projection)),
        "lexical_python": lexical,
        "physical_trees": tree_projection,
        "protected_roots": _live_main_suite_protected_roots(policy.get("protected_roots")),
        "resolved_python": resolved,
        "startup_files": startup_projection,
    }


def _validate_v2_main_suite_external_snapshot(
    value: Any,
    *,
    repo_root: Path,
    policy: Mapping[str, Any],
    policy_bindings: Mapping[str, Mapping[str, Any]],
    environment: Mapping[str, str],
) -> dict[str, Any]:
    expected_keys = {
        "bindings",
        "distribution_ownership_sha256",
        "factor_authority_sha256",
        "invalid_dist_info_sha256",
        "lexical_python",
        "physical_trees",
        "protected_roots",
        "pycache_prefix",
        "resolved_python",
        "snapshot_sha256",
        "startup_files",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise Phase0EvidenceError("main-suite external snapshot is invalid")
    declared_sha256 = value.get("snapshot_sha256")
    _require_sha256(
        declared_sha256,
        label="main-suite external snapshot.snapshot_sha256",
    )
    unsealed = dict(value)
    unsealed.pop("snapshot_sha256")
    if declared_sha256 != _sha256(_canonical_bytes(unsealed)):
        raise Phase0EvidenceError("main-suite external snapshot SHA-256 mismatch")
    rows = value.get("bindings")
    if type(rows) is not list:
        raise Phase0EvidenceError("main-suite external snapshot bindings are invalid")
    validator_bindings: list[tuple[str, dict[str, Any]]] = []
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
        binding, _raw = _main_suite_live_file_binding(
            _safe_repo_entry(repo_root, relative),
            label=f"main-suite {label}",
        )
        validator_bindings.append((label, binding))
    expected_binding_pairs = [
        ("wrapper_binding", policy.get("wrapper_binding")),
        ("harness_binding", policy.get("harness_binding")),
        ("candidate_conftest", policy.get("candidate_conftest")),
        ("package_manifest", policy_bindings["policy_manifest_binding"]),
        ("runtime_policy", policy_bindings["policy_binding"]),
        ("runtime_policy_schema", policy_bindings["policy_schema_binding"]),
        *validator_bindings,
    ]
    if any(type(binding) is not dict for _label, binding in expected_binding_pairs):
        raise Phase0EvidenceError("main-suite external binding policy is invalid")
    expected_rows = [{"label": label, **binding} for label, binding in expected_binding_pairs]
    if rows != expected_rows:
        raise Phase0EvidenceError("main-suite external binding rows mismatch")

    live_projection = _live_v2_main_suite_policy_projection(
        policy,
        repo_root=repo_root,
    )
    if any(value.get(key) != projected for key, projected in live_projection.items()):
        raise Phase0EvidenceError("main-suite external live projection mismatch")

    pycache_binding = _require_exact_keys(
        value["pycache_prefix"],
        {
            "gid",
            "mode",
            "path",
            "st_ctime_ns",
            "st_dev",
            "st_ino",
            "st_mtime_ns",
            "st_nlink",
            "uid",
        },
        label="main-suite external pycache_prefix",
    )
    pycache_path = Path(environment["PYTHONPYCACHEPREFIX"])
    try:
        before = pycache_path.lstat()
        with os.scandir(pycache_path) as entries:
            if next(entries, None) is not None:
                raise Phase0EvidenceError("main-suite external pycache prefix is not empty")
        after = pycache_path.lstat()
    except Phase0EvidenceError:
        raise
    except OSError as exc:
        raise Phase0EvidenceError("main-suite external pycache prefix is unavailable") from exc
    expected_pycache = {
        "gid": before.st_gid,
        "mode": f"{stat.S_IMODE(before.st_mode):04o}",
        "path": str(pycache_path),
        "st_ctime_ns": before.st_ctime_ns,
        "st_dev": before.st_dev,
        "st_ino": before.st_ino,
        "st_mtime_ns": before.st_mtime_ns,
        "st_nlink": before.st_nlink,
        "uid": before.st_uid,
    }
    if (
        _stat_signature(before) != _stat_signature(after)
        or not stat.S_ISDIR(before.st_mode)
        or pycache_binding != expected_pycache
        or pycache_binding["mode"] != "0700"
        or pycache_binding["uid"] != os.getuid()
    ):
        raise Phase0EvidenceError("main-suite external pycache binding mismatch")
    return dict(value)


def _validate_v2_main_suite_project_modules(
    value: Any,
    *,
    repo_root: Path,
    candidate_paths: Sequence[str],
    label: str,
) -> list[dict[str, str]]:
    if type(value) is not list:
        raise Phase0EvidenceError(f"{label} project_modules are invalid")
    rows: list[dict[str, str]] = []
    names: list[str] = []
    allowed = set(candidate_paths)
    for index, raw_row in enumerate(value):
        row = _require_exact_keys(
            raw_row,
            {"name", "path", "sha256"},
            label=f"{label}.project_modules[{index}]",
        )
        name = _require_string(
            row["name"],
            label=f"{label}.project_modules[{index}].name",
        )
        path = Path(
            _require_absolute_path(
                row["path"],
                label=f"{label}.project_modules[{index}].path",
            )
        )
        sha256 = _require_sha256(
            row["sha256"],
            label=f"{label}.project_modules[{index}].sha256",
        )
        try:
            relative = path.relative_to(repo_root).as_posix()
        except ValueError as exc:
            raise Phase0EvidenceError(f"{label} project module escaped the candidate root") from exc
        if relative not in allowed or not (
            name == "quant_investor" or name.startswith("quant_investor.")
        ):
            raise Phase0EvidenceError(f"{label} project module is not outer-source-bound")
        raw, _observed = _stable_regular_file(path, require_private=False)
        if _sha256(raw) != sha256:
            raise Phase0EvidenceError(f"{label} project module byte binding mismatch")
        names.append(name)
        rows.append({"name": name, "path": str(path), "sha256": sha256})
    _require_unique_casefold(names, label=f"{label} project module names")
    if names != sorted(names, key=lambda item: item.encode("utf-8")):
        raise Phase0EvidenceError(f"{label} project modules are not canonical")
    return rows


def _validate_v2_main_suite_runtime(
    value: Any,
    *,
    label: str,
    repo_root: Path,
    policy: Mapping[str, Any],
    policy_sha256: str,
    expected_closure: Any,
    environment: Mapping[str, str],
    candidate_paths: Sequence[str],
) -> dict[str, Any]:
    runtime = _require_exact_keys(
        value,
        {
            "bytecode_policy",
            "factor_authority_sha256",
            "interpreter",
            "invalid_dist_info_sha256",
            "inventory",
            "loaded_modules",
            "policy_sha256",
            "project_modules",
            "routing",
        },
        label=label,
    )
    main_runtime = policy.get("main_runtime")
    factor_sources = policy.get("factor_authority_sources")
    routing_policy = policy.get("routing")
    startup_files = None if type(main_runtime) is not dict else main_runtime.get("startup_files")
    startup_modules = (
        None if type(main_runtime) is not dict else main_runtime.get("startup_modules")
    )
    if (
        type(main_runtime) is not dict
        or type(factor_sources) is not list
        or type(routing_policy) is not dict
        or type(startup_files) is not list
        or type(startup_modules) is not list
    ):
        raise Phase0EvidenceError(f"{label} runtime policy is invalid")
    startup_projection = [
        (
            dict(row)
            if type(row) is dict and row.get("present") is False
            else {"path": row.get("path"), "present": True, **row}
        )
        for row in startup_files
        if type(row) is dict
    ]
    if len(startup_projection) != len(startup_files):
        raise Phase0EvidenceError(f"{label} startup policy is invalid")
    runtime_state = main_runtime.get("post_site_state")
    if type(runtime_state) is not dict:
        raise Phase0EvidenceError(f"{label} post-site runtime policy is invalid")
    routed_state = dict(runtime_state)
    routed_state["sys_path"] = routing_policy.get("sanitized_sys_path")
    expected_routing = {
        "candidate_root": policy.get("candidate_root"),
        "quant_investor_origin": routing_policy.get("quant_investor_origin"),
        "removed_authority_entries": routing_policy.get("removed_authority_entries"),
        "runtime_state": routed_state,
        "startup": {
            "lexical_python": main_runtime.get("lexical_python_binding"),
            "resolved_python": main_runtime.get("resolved_python_binding"),
            "startup_files": startup_projection,
            "wrapper": policy.get("wrapper_binding"),
        },
        "startup_modules": startup_modules,
    }
    if (
        runtime["bytecode_policy"]
        != {
            "dont_write_bytecode": True,
            "pycache_prefix": environment["PYTHONPYCACHEPREFIX"],
        }
        or runtime["factor_authority_sha256"] != _sha256(_canonical_bytes(factor_sources))
        or runtime["interpreter"] != main_runtime.get("resolved_python_binding")
        or runtime["inventory"] != main_runtime.get("valid_inventory")
        or runtime["invalid_dist_info_sha256"]
        != _sha256(_canonical_bytes(main_runtime.get("invalid_dist_info")))
        or runtime["loaded_modules"] != expected_closure
        or runtime["policy_sha256"] != policy_sha256
        or runtime["routing"] != expected_routing
    ):
        raise Phase0EvidenceError(f"{label} runtime policy projection mismatch")
    _validate_v2_main_suite_project_modules(
        runtime["project_modules"],
        repo_root=repo_root,
        candidate_paths=candidate_paths,
        label=label,
    )
    return runtime


def _main_suite_expected_plugin_rows(policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = policy.get("pytest_plugins")
    if type(rows) is not list:
        raise Phase0EvidenceError("main-suite pytest plugin policy is invalid")
    projected: list[dict[str, Any]] = []
    for row in rows:
        if type(row) is not dict:
            raise Phase0EvidenceError("main-suite pytest plugin row is invalid")
        projected.append(
            {
                "distribution": row.get("distribution"),
                "entry_point_name": row.get("entry_point_name"),
                "hook_trace": row.get("hook_trace"),
                "module": row.get("module"),
                "module_file_binding": row.get("module_file_binding"),
                "physical_tree": row.get("physical_tree"),
                "version": row.get("version"),
            }
        )
    return projected


def _main_suite_expected_support_tree_rows(
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = policy.get("pytest_support_trees")
    if type(rows) is not list:
        raise Phase0EvidenceError("main-suite pytest support-tree policy is invalid")
    projected: list[dict[str, Any]] = []
    for row in rows:
        if type(row) is not dict or type(row.get("descriptor")) is not dict:
            raise Phase0EvidenceError("main-suite pytest support-tree row is invalid")
        projected.append({"name": row.get("name"), **row["descriptor"]})
    return projected


def _validate_v2_main_suite_semantics(
    receipt: Mapping[str, Any],
    *,
    streams: Mapping[str, bytes],
    frames: Sequence[Mapping[str, Any]],
    repo_root: Path,
    policy: Mapping[str, Any],
    policy_bindings: Mapping[str, Mapping[str, Any]],
    session_binding: Mapping[str, Any],
    skip_baseline: Mapping[str, Any],
    skip_baseline_raw: bytes,
    base_commit: str,
    source_state: Mapping[str, Any],
    package_source_full: Mapping[str, Any],
) -> dict[str, Any]:
    _require_exact_keys(
        receipt,
        {
            "accepted",
            "attestations",
            "authority",
            "challenge_binding",
            "claims",
            "command",
            "external_after",
            "external_before",
            "failure_codes",
            "failures",
            "finalization",
            "framing",
            "limitations",
            "outcome",
            "policy_binding",
            "policy_manifest_binding",
            "policy_schema_binding",
            "protocol_version",
            "schema_id",
            "semantic_sha256",
            "streams",
            "timing",
            "version",
        },
        label="full_offline_suite main-suite receipt",
    )
    if (
        receipt.get("version") != MAIN_SUITE_RECEIPT_VERSION
        or receipt.get("schema_id") != MAIN_SUITE_RECEIPT_SCHEMA_ID
        or receipt.get("protocol_version") != PROTOCOL_VERSION
        or receipt.get("authority") is not False
        or receipt.get("limitations") != NORMATIVE_LIMITATIONS
        or receipt.get("limitations") != policy.get("limitations")
        or receipt.get("framing") != MAIN_SUITE_FRAMING
        or receipt.get("accepted") is not True
        or receipt.get("outcome") != "PASSED"
        or receipt.get("failure_codes") != []
        or receipt.get("failures") != []
    ):
        raise Phase0EvidenceError("full_offline_suite main-suite receipt was not accepted")
    if any(
        receipt.get(name) != policy_bindings.get(name)
        for name in (
            "policy_binding",
            "policy_manifest_binding",
            "policy_schema_binding",
        )
    ):
        raise Phase0EvidenceError("full_offline_suite policy binding mismatch")

    session_sha256 = _require_sha256(
        session_binding.get("sha256"),
        label="full_offline_suite session byte SHA-256",
    )
    if receipt.get("challenge_binding") != {
        "kind": "PHASE0_SESSION_FILE",
        "sha256": session_sha256,
    }:
        raise Phase0EvidenceError("full_offline_suite session challenge binding mismatch")
    candidate_paths = _validate_v2_candidate_module_source_membership(
        policy,
        repo_root=repo_root,
        base_commit=base_commit,
        source_state=source_state,
        package_source_full=package_source_full,
    )

    command = _require_exact_keys(
        receipt.get("command"),
        {"argv", "cwd", "environment"},
        label="full_offline_suite command",
    )
    main_runtime = policy.get("main_runtime")
    wrapper = policy.get("wrapper_binding")
    pytest_args = policy.get("pytest_args")
    if (
        type(main_runtime) is not dict
        or type(wrapper) is not dict
        or type(pytest_args) is not list
        or not all(type(item) is str for item in pytest_args)
    ):
        raise Phase0EvidenceError("full_offline_suite policy command is invalid")
    environment = _validate_v2_main_suite_environment(
        command["environment"],
        policy=policy,
    )
    expected_argv = [
        main_runtime.get("lexical_python"),
        "-I",
        "-S",
        "-B",
        "-X",
        f"pycache_prefix={environment['PYTHONPYCACHEPREFIX']}",
        wrapper.get("path"),
        policy_bindings["policy_binding"]["path"],
        policy_bindings["policy_binding"]["sha256"],
        "--",
        *pytest_args,
    ]
    if command["argv"] != expected_argv or command["cwd"] != str(repo_root):
        raise Phase0EvidenceError("full_offline_suite main-bound command mismatch")

    claims = _require_exact_keys(
        receipt.get("claims"),
        {
            "exit_code",
            "final_audit_completed",
            "final_audit_enforced",
            "kernel_egress_attested",
            "network_unreachability_proven",
            "offline_policy_enforced",
            "signal",
        },
        label="full_offline_suite receipt claims",
    )
    policy_claims = policy.get("claims")
    if (
        type(policy_claims) is not dict
        or claims["exit_code"] != 0
        or claims["signal"] is not None
        or claims["final_audit_completed"] is not True
        or claims["final_audit_enforced"] is not True
        or any(claims.get(key) != expected for key, expected in policy_claims.items())
    ):
        raise Phase0EvidenceError("full_offline_suite acceptance claims mismatch")
    if receipt.get("finalization") != {
        "cleanup": {"attempted": True, "status": "PASSED"},
        "external_after": {
            "attempted": True,
            "equal": True,
            "status": "PASSED",
        },
    }:
        raise Phase0EvidenceError("full_offline_suite finalization did not pass")
    if receipt.get("external_before") != receipt.get("external_after"):
        raise Phase0EvidenceError("full_offline_suite external runtime drifted")
    _validate_v2_main_suite_external_snapshot(
        receipt["external_before"],
        repo_root=repo_root,
        policy=policy,
        policy_bindings=policy_bindings,
        environment=environment,
    )

    if receipt.get("attestations") != list(frames) or len(frames) != 3:
        raise Phase0EvidenceError("full_offline_suite binary attestation mismatch")
    closures = policy.get("module_closures")
    if type(closures) is not dict or set(closures) != {
        "final",
        "pre_collection",
        "pre_import",
    }:
        raise Phase0EvidenceError("full_offline_suite module closure policy is invalid")
    expected_names = ("pre_import", "pre_collection", "terminal_complete")
    pids: set[tuple[Any, Any]] = set()
    for index, frame in enumerate(frames):
        payload = frame.get("payload") if type(frame) is dict else None
        if (
            type(payload) is not dict
            or frame.get("phase") != index + 1
            or payload.get("frame") != expected_names[index]
            or payload.get("challenge_binding_sha256") != session_sha256
            or type(payload.get("pid")) is not int
            or type(payload.get("ppid")) is not int
        ):
            raise Phase0EvidenceError(f"full_offline_suite attestation phase {index + 1} mismatch")
        pids.add((payload["pid"], payload["ppid"]))
    if len(pids) != 1:
        raise Phase0EvidenceError("full_offline_suite attestation process identity mismatch")
    first_payload = frames[0]["payload"]
    second_payload = frames[1]["payload"]
    terminal_payload = frames[2]["payload"]
    first_runtime = first_payload.get("runtime")
    second_runtime = second_payload.get("runtime")
    _validate_v2_main_suite_runtime(
        first_runtime,
        label="full_offline_suite pre_import",
        repo_root=repo_root,
        policy=policy,
        policy_sha256=str(policy_bindings["policy_binding"]["sha256"]),
        expected_closure=closures["pre_import"],
        environment=environment,
        candidate_paths=candidate_paths,
    )
    _validate_v2_main_suite_runtime(
        second_runtime,
        label="full_offline_suite pre_collection",
        repo_root=repo_root,
        policy=policy,
        policy_sha256=str(policy_bindings["policy_binding"]["sha256"]),
        expected_closure=closures["pre_collection"],
        environment=environment,
        candidate_paths=candidate_paths,
    )
    module_policy = policy.get("module_policy")
    ownership_rows = (
        None if type(module_policy) is not dict else module_policy.get("distribution_ownership")
    )
    pytest_versions = (
        [
            row.get("version")
            for row in ownership_rows
            if type(row) is dict
            and type(row.get("name")) is str
            and re.sub(r"[-_.]+", "-", str(row["name"])).casefold() == "pytest"
        ]
        if type(ownership_rows) is list
        else []
    )
    if (
        first_payload.get("environment") != environment
        or second_payload.get("candidate_conftest") != policy.get("candidate_conftest")
        or second_payload.get("initial_conftest_loaded") is not True
        or second_payload.get("plugins") != _main_suite_expected_plugin_rows(policy)
        or second_payload.get("support_trees") != _main_suite_expected_support_tree_rows(policy)
        or second_payload.get("project_modules") != second_runtime.get("project_modules")
        or len(pytest_versions) != 1
        or second_payload.get("pytest_version") != pytest_versions[0]
        or terminal_payload.get("final_loaded_modules") != closures["final"]
        or terminal_payload.get("pytest_exit_code") != 0
    ):
        raise Phase0EvidenceError("full_offline_suite attested closure mismatch")

    output = streams["stdout"] + streams["stderr"]
    counts = _pytest_summary_counts(output, label="full_offline_suite")
    return _validate_v2_gate_claims(
        "full_offline_suite",
        {
            **counts,
            "exit_code": 0,
            "raw_output_sha256": _sha256(output),
            "skip_allowlist": skip_baseline["entries"],
        },
        output=output,
        skip_baseline=skip_baseline,
        skip_baseline_raw=skip_baseline_raw,
    )


def _legacy_v2_main_suite_context(
    *,
    evidence_path: Path,
    repo_root: Path,
    schemas: Mapping[str, Mapping[str, Any]],
    current_source_state: Mapping[str, Any],
) -> dict[str, Any]:
    session_path = evidence_path.parent / "00_session.json"
    session, session_file_binding, _session_raw = _load_v2_resource(
        session_path,
        artifact_version=SESSION_VERSION,
        schemas=schemas,
        repo_root=repo_root,
        label="full_offline_suite sibling session",
    )
    session_binding = _v2_session_binding(session, session_file_binding)
    base_commit = _validate_base_commit(session.get("base_commit"))
    if session.get("repo_root") != str(repo_root) or session.get("source_state") != dict(
        current_source_state
    ):
        raise Phase0EvidenceError("full_offline_suite sibling session source binding mismatch")
    physical_full, package_source_superset = _live_v2_package_source_superset(repo_root)
    if session.get("package_source_superset") != package_source_superset:
        raise Phase0EvidenceError("full_offline_suite sibling session package source mismatch")
    skip_reference = session.get("skip_baseline_binding")
    if type(skip_reference) is not dict or type(skip_reference.get("path")) is not str:
        raise Phase0EvidenceError("full_offline_suite sibling session skip binding is missing")
    skip_path = Path(str(skip_reference["path"]))
    skip, skip_binding, skip_raw = _load_v2_resource(
        skip_path,
        artifact_version=SKIP_BASELINE_VERSION,
        schemas=schemas,
        repo_root=repo_root,
        label="full_offline_suite frozen skip baseline",
    )
    _validate_session_external_binding(
        skip_reference,
        actual_binding=skip_binding,
        semantic_sha256=skip["semantic_sha256"],
        label="full_offline_suite sibling session skip_baseline_binding",
    )
    return {
        "base_commit": base_commit,
        "package_source_full": physical_full,
        "session_binding": session_binding,
        "skip": skip,
        "skip_raw": skip_raw,
        "source_state": dict(current_source_state),
    }


def _expected_v2_step(index: int) -> dict[str, Any]:
    return {
        "filename": GATE_FILENAMES[index],
        "kind": GATE_KINDS[GATE_ROLES[index]],
        "ordinal": GATE_ORDINALS[index],
        "role": GATE_ROLES[index],
    }


def _project_v2_producer_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": value["path"],
        "sha256": value["sha256"],
        "size_bytes": value["size_bytes"],
        "version": value["version"],
    }


def _validate_v2_gate_plan(
    value: Any,
    *,
    repo_root: Path,
) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) != len(GATE_ROLES):
        raise Phase0EvidenceError("session gate_plan must contain ten exact rows")
    rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(value):
        row = _require_exact_keys(
            raw_row,
            {
                "artifact_version",
                "filename",
                "kind",
                "ordinal",
                "producer_path",
                "producer_version",
                "role",
                "schema_id",
            },
            label=f"session gate_plan[{index}]",
        )
        producer_relative, producer_version = GATE_PRODUCER_SPECS[index]
        expected = {
            "artifact_version": GATE_ARTIFACT_VERSIONS[index],
            "filename": GATE_FILENAMES[index],
            "kind": GATE_KINDS[GATE_ROLES[index]],
            "ordinal": GATE_ORDINALS[index],
            "producer_path": producer_relative,
            "producer_version": producer_version,
            "role": GATE_ROLES[index],
            "schema_id": GATE_SCHEMA_IDS[index],
        }
        if row != expected:
            raise Phase0EvidenceError(f"session gate_plan[{index}] mismatch")
        rows.append(dict(row))
    return rows


def _command_output_bytes(streams: Sequence[tuple[bytes, bytes]]) -> bytes:
    return b"".join(stdout + stderr for stdout, stderr in streams)


def _expected_gate_claim_keys(role: str) -> set[str]:
    pytest_keys = {"errors", "exit_code", "failed", "passed", "skipped", "xfail", "xpass"}
    if role in {"native_sync_receipt", "package_parity", "hash_freeze_readback"}:
        return {"accepted"}
    if role == "native_sync_log" or role == "mypy":
        return {"exit_code"}
    if role == "v2_evidence_tests":
        return pytest_keys
    if role == "recommended_core_tests":
        return pytest_keys | {"staged_upgrade_exit_code"}
    if role == "full_offline_suite":
        return pytest_keys | {
            "raw_output_sha256",
            "skip_allowlist",
        }
    if role == "black":
        return {"exit_code", "unchanged"}
    if role == "diff_check":
        return {"exit_code", "raw_output_sha256"}
    raise Phase0EvidenceError(f"unknown Phase 0 gate role: {role}")


def _validate_v2_gate_claims(
    role: str,
    value: Any,
    *,
    output: bytes | None,
    skip_baseline: Mapping[str, Any] | None,
    skip_baseline_raw: bytes | None,
) -> dict[str, Any]:
    claims = _require_exact_keys(
        value,
        _expected_gate_claim_keys(role),
        label=f"{role} claims",
    )
    if role in {"native_sync_receipt", "package_parity", "hash_freeze_readback"}:
        if claims["accepted"] is not True:
            raise Phase0EvidenceError(f"{role} is not accepted")
        return claims
    if claims["exit_code"] != 0 or type(claims["exit_code"]) is not int:
        raise Phase0EvidenceError(f"{role} exit_code is not integer zero")
    raw = b"" if output is None else output
    if role in {"v2_evidence_tests", "recommended_core_tests", "full_offline_suite"}:
        observed = _pytest_summary_counts(raw, label=role)
        expected_counts = {
            key: claims[key] for key in ("errors", "failed", "passed", "skipped", "xfail", "xpass")
        }
        for key, count in expected_counts.items():
            _require_int(count, label=f"{role} claims.{key}")
        if observed != expected_counts:
            raise Phase0EvidenceError(f"{role} pytest summary mismatch")
        if (
            claims["failed"] != 0
            or claims["errors"] != 0
            or claims["xfail"] != 0
            or claims["xpass"] != 0
        ):
            raise Phase0EvidenceError(f"{role} carries a non-passing pytest outcome")
        if role != "full_offline_suite" and claims["skipped"] != 0:
            raise Phase0EvidenceError(f"{role} may not carry skips")
        if role == "recommended_core_tests":
            if (
                claims["staged_upgrade_exit_code"] != 0
                or type(claims["staged_upgrade_exit_code"]) is not int
            ):
                raise Phase0EvidenceError("recommended_core_tests staged gate did not pass")
        if role == "full_offline_suite":
            if skip_baseline is None or skip_baseline_raw is None:
                raise Phase0EvidenceError("full suite is missing its frozen skip baseline")
            entries = skip_baseline["entries"]
            observed_entries = _pytest_skip_entries(raw, label=role)
            if (
                claims["skipped"] != 42
                or observed_entries != entries
                or claims["skip_allowlist"] != entries
            ):
                raise Phase0EvidenceError("full suite skip rows differ from frozen baseline")
            if claims["raw_output_sha256"] != _sha256(raw):
                raise Phase0EvidenceError("full suite raw_output_sha256 mismatch")
        return claims
    text = _decode_text(raw, label=role)
    if role == "mypy" and "Success: no issues found" not in text:
        raise Phase0EvidenceError("mypy evidence lacks the exact success marker")
    if role == "black":
        if (
            claims["unchanged"] is not True
            or "would reformat" in text
            or ("left unchanged" not in text and "files would be left unchanged" not in text)
        ):
            raise Phase0EvidenceError("Black evidence is not unchanged")
    if role == "diff_check":
        if raw or claims["raw_output_sha256"] != _sha256(b""):
            raise Phase0EvidenceError("diff-check output is not empty")
    return claims


def _validate_v2_command_context(
    receipt: Mapping[str, Any],
    *,
    gate_index: int,
    repo_root: Path,
    session_binding: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    toolchain: Mapping[str, Any],
    package_source_superset: Mapping[str, Any],
    protected_roots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    role = GATE_ROLES[gate_index]
    if receipt.get("step") != _expected_v2_step(gate_index):
        raise Phase0EvidenceError(f"{role} step mismatch")
    _validate_v2_session_binding(
        receipt.get("session_binding"),
        expected=session_binding,
        label=f"{role}.session_binding",
    )
    producer_relative, producer_version = GATE_PRODUCER_SPECS[gate_index]
    _validate_v2_producer_binding(
        receipt.get("producer"),
        expected_path=repo_root / producer_relative,
        expected_version=producer_version,
        label=f"{role}.producer",
    )
    for key in ("source_before", "source_after"):
        if _validate_source_binding(receipt.get(key), label=f"{role}.{key}") != source_binding:
            raise Phase0EvidenceError(f"{role} source binding mismatch")
    for key in ("toolchain_before", "toolchain_after"):
        observed_toolchain = _validate_v2_toolchain(
            receipt.get(key),
            label=f"{role}.{key}",
            live=False,
        )
        if observed_toolchain != toolchain:
            raise Phase0EvidenceError(f"{role} toolchain binding mismatch")
    for key in ("package_source_superset_before", "package_source_superset_after"):
        observed_source = _validate_v2_namespace_binding(
            receipt.get(key),
            label=f"{role}.{key}",
        )
        if observed_source != package_source_superset:
            raise Phase0EvidenceError(f"{role} package source superset mismatch")
    for key in ("protected_roots_before", "protected_roots_after"):
        observed_roots = _validate_v2_protected_roots(
            receipt.get(key),
            label=f"{role}.{key}",
        )
        if observed_roots != list(protected_roots):
            raise Phase0EvidenceError(f"{role} protected roots mismatch")
    if receipt.get("limitations") != NORMATIVE_LIMITATIONS:
        raise Phase0EvidenceError(f"{role} limitations mismatch")
    return dict(receipt)


def _v2_command_work_root(value: Any, *, repo_root: Path) -> Path:
    raw = _require_absolute_path(value, label="Phase 0 command work root")
    root = Path(raw)
    if (
        Path(os.path.abspath(root)) != root
        or _path_within(root, repo_root)
        or _path_within(repo_root, root)
        or any(
            _path_within(root, protected) or _path_within(protected, root)
            for _identifier, protected in PROTECTED_ROOT_SPECS
        )
    ):
        raise Phase0EvidenceError("Phase 0 command work root is unsafe")
    return root


def _v2_base_environment(toolchain: Mapping[str, Any]) -> dict[str, str]:
    return {
        **V2_BASE_CLOSED_ENVIRONMENT,
        "UV_CACHE_DIR": str(toolchain["uv_cache"]["path"]),
    }


def _v2_stage_environment(
    work_root: Path,
    stage: str,
    *,
    toolchain: Mapping[str, Any],
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    runtime = work_root / f"runtime_{stage}"
    cache = runtime / "cache"
    value = {
        **_v2_base_environment(toolchain),
        "BLACK_CACHE_DIR": str(cache / "black"),
        "HOME": str(runtime / "home"),
        "MYPY_CACHE_DIR": str(cache / "mypy"),
        "TMPDIR": str(runtime / "tmp"),
        "XDG_CACHE_HOME": str(cache),
    }
    if extra is not None:
        value.update(extra)
    return value


def _v2_live_tool_versions(
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, str]:
    installed = payload.get("installed_reconciliation")
    rows = installed.get("installed") if type(installed) is dict else None
    if type(rows) is not list:
        raise Phase0EvidenceError("native dependency installed tool inventory is missing")
    versions: dict[str, str] = {}
    for row in rows:
        if type(row) is not dict:
            continue
        name = row.get("name")
        version = row.get("version")
        if name in {"black", "mypy", "pytest"} and type(version) is str:
            versions[str(name)] = f"{name} {version}"
    if set(versions) != {"black", "mypy", "pytest"}:
        raise Phase0EvidenceError("native dependency installed tool inventory is incomplete")
    try:
        bash = subprocess.run(
            ["/bin/bash", "--version"],
            cwd=repo_root,
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise Phase0EvidenceError("cannot read live bash version") from exc
    try:
        first_line = bash.stdout.decode("ascii", errors="strict").splitlines()[0]
    except (UnicodeError, IndexError) as exc:
        raise Phase0EvidenceError("live bash version output is invalid") from exc
    match = re.search(r"version ([0-9]+(?:\.[0-9]+){1,3})", first_line)
    if bash.returncode != 0 or bash.stderr or match is None:
        raise Phase0EvidenceError("live bash version probe failed")
    git_output = _git_bytes(("git", "--version"), repo_root=repo_root)
    try:
        git_version = git_output.decode("ascii", errors="strict").strip()
    except UnicodeError as exc:
        raise Phase0EvidenceError("live git version output is invalid") from exc
    if re.fullmatch(r"git version \S+(?: .*)?", git_version, re.ASCII) is None:
        raise Phase0EvidenceError("live git version probe failed")
    versions["bash"] = f"bash {match.group(1)}"
    versions["git"] = git_version
    return versions


def _v2_git_objects_path(repo_root: Path) -> str:
    raw = _git_bytes(("git", "rev-parse", "--git-path", "objects"), repo_root=repo_root)
    try:
        lexical = Path(raw.decode("utf-8", errors="strict").strip())
    except UnicodeError as exc:
        raise Phase0EvidenceError("Git object path is invalid") from exc
    if not lexical.is_absolute():
        lexical = repo_root / lexical
    try:
        return str(lexical.resolve(strict=True))
    except OSError as exc:
        raise Phase0EvidenceError("Git object path is unavailable") from exc


def _validate_v2_command_identity(
    role: str,
    receipt: Mapping[str, Any],
    *,
    repo_root: Path,
    toolchain: Mapping[str, Any],
    work_root: Path | None,
    tool_versions: Mapping[str, str] | None,
) -> Path:
    commands = receipt["commands"]
    uv_path = str(toolchain["uv"]["lexical_path"])
    base_python = str(toolchain["base_python"]["lexical_path"])
    if role == "native_sync_log":
        if len(commands) != 1:
            raise Phase0EvidenceError("native_sync_log must bind one exact command")
        command = commands[0]
        environment = command["environment"]
        if type(environment) is not dict:
            raise Phase0EvidenceError("native_sync_log environment is invalid")
        project_environment = environment.get("UV_PROJECT_ENVIRONMENT")
        if type(project_environment) is not str:
            raise Phase0EvidenceError("native_sync_log environment root is missing")
        root = _v2_command_work_root(
            str(Path(project_environment).parent),
            repo_root=repo_root,
        )
        expected_environment = {
            **_v2_base_environment(toolchain),
            "TMPDIR": str(root / "tmp/native_sync"),
            "UV_PROJECT_ENVIRONMENT": str(root / "native_venv"),
        }
        if (
            command["argv"]
            != [
                uv_path,
                "sync",
                "--python",
                base_python,
                "--locked",
                "--all-extras",
                "--offline",
            ]
            or environment != expected_environment
            or command["tool_version"] != toolchain["uv"]["output"]
        ):
            raise Phase0EvidenceError("native_sync_log command identity mismatch")
        return root
    if work_root is None or tool_versions is None:
        raise Phase0EvidenceError(f"{role} lacks native command/tool predecessors")
    target_python = str(work_root / "native_venv/bin/python")
    expected: list[tuple[list[str], dict[str, str], str]]
    if role == "v2_evidence_tests":
        expected = [
            (
                [target_python, "-m", "pytest", *V2_EVIDENCE_TESTS, *PYTEST_OPTIONS],
                _v2_stage_environment(work_root, "v2_tests", toolchain=toolchain),
                tool_versions["pytest"],
            )
        ]
    elif role == "recommended_core_tests":
        environment = _v2_stage_environment(
            work_root,
            "recommended_core",
            toolchain=toolchain,
            extra={"PYTHON": target_python},
        )
        expected = [
            (
                ["scripts/staged_upgrade_quality_gate.sh"],
                environment,
                tool_versions["bash"],
            ),
            (
                [target_python, "-m", "pytest", *RECOMMENDED_CORE_TESTS, *PYTEST_OPTIONS],
                environment,
                tool_versions["pytest"],
            ),
        ]
    elif role == "mypy":
        expected = [
            (
                [
                    target_python,
                    "-m",
                    "mypy",
                    *MYPY_TARGETS,
                ],
                _v2_stage_environment(work_root, "mypy", toolchain=toolchain),
                tool_versions["mypy"],
            )
        ]
    elif role == "black":
        expected = [
            (
                [target_python, "-m", "black", "--check", *BLACK_TARGETS],
                _v2_stage_environment(work_root, "black", toolchain=toolchain),
                tool_versions["black"],
            )
        ]
    elif role == "diff_check":
        diff_root = work_root / "diff"
        expected = [
            (
                ["git", "diff", "--check"],
                {
                    "GIT_ALTERNATE_OBJECT_DIRECTORIES": _v2_git_objects_path(repo_root),
                    "GIT_CONFIG_GLOBAL": "/dev/null",
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_INDEX_FILE": str(diff_root / "phase0-alternate.index"),
                    "GIT_OBJECT_DIRECTORY": str(diff_root / "objects"),
                    "GIT_OPTIONAL_LOCKS": "0",
                    "HOME": str(diff_root / "home"),
                    "LANG": "C",
                    "LC_ALL": "C",
                    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "TMPDIR": str(diff_root / "tmp"),
                },
                tool_versions["git"],
            )
        ]
    else:
        raise Phase0EvidenceError(f"unsupported log command role: {role}")
    observed = [
        (command["argv"], command["environment"], command["tool_version"]) for command in commands
    ]
    if observed != expected:
        raise Phase0EvidenceError(f"{role} command identity mismatch")
    return work_root


def _validate_v2_skip_baseline(
    payload: Mapping[str, Any],
    *,
    source_state: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    toolchain: Mapping[str, Any],
    package_source_superset: Mapping[str, Any],
    protected_roots: Sequence[Mapping[str, Any]],
    repo_root: Path,
) -> list[dict[str, Any]]:
    skip_module = _load_local_module(
        repo_root / "scripts/v17_phase0_skip_baseline.py",
        module_name="_phase0_v17_skip_baseline",
    )
    producer_validator = getattr(skip_module, "validate_skip_baseline", None)
    if not callable(producer_validator):
        raise Phase0EvidenceError("skip baseline producer validator is unavailable")
    try:
        producer_validated = producer_validator(payload, repo_root=repo_root)
    except Exception as exc:
        raise Phase0EvidenceError("skip baseline producer semantic validation failed") from exc
    if producer_validated != dict(payload):
        raise Phase0EvidenceError("skip baseline producer semantic readback mismatch")
    if (
        payload.get("version") != SKIP_BASELINE_VERSION
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("status") != "FROZEN"
        or payload.get("accepted") is not True
        or payload.get("authority") is not False
        or payload.get("limitations") != NORMATIVE_LIMITATIONS
    ):
        raise Phase0EvidenceError("skip baseline identity or authority mismatch")
    producer = payload.get("producer")
    producer_row = _require_exact_keys(
        producer,
        {"path", "sha256", "size_bytes", "version"},
        label="skip baseline producer",
    )
    _validate_v2_producer_binding(
        producer_row,
        expected_path=repo_root / "scripts/v17_phase0_skip_baseline.py",
        expected_version=SKIP_BASELINE_PRODUCER_VERSION,
        label="skip baseline producer",
    )
    if (
        _validate_source_binding(
            payload.get("source_binding"),
            label="skip baseline source_binding",
        )
        != source_binding
    ):
        raise Phase0EvidenceError("skip baseline source binding mismatch")
    if payload.get("source_state") != source_state:
        raise Phase0EvidenceError("skip baseline source state mismatch")
    observed_toolchain = _validate_v2_toolchain(
        payload.get("toolchain_binding"),
        label="skip baseline toolchain_binding",
        live=False,
    )
    if observed_toolchain != toolchain:
        raise Phase0EvidenceError("skip baseline toolchain mismatch")
    observed_source = _validate_v2_namespace_binding(
        payload.get("package_source_superset"),
        label="skip baseline package_source_superset",
    )
    if observed_source != package_source_superset:
        raise Phase0EvidenceError("skip baseline package source mismatch")
    for key in ("protected_roots_before", "protected_roots_after"):
        observed_roots = _validate_v2_protected_roots(
            payload.get(key),
            label=f"skip baseline {key}",
        )
        if observed_roots != list(protected_roots):
            raise Phase0EvidenceError("skip baseline protected roots mismatch")
    entries_value = payload.get("entries")
    if type(entries_value) is not list:
        raise Phase0EvidenceError("skip baseline entries must be an array")
    entries: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(entries_value):
        entry = _require_exact_keys(
            raw_entry,
            {"count", "line", "path", "reason"},
            label=f"skip baseline entries[{index}]",
        )
        normalized = {
            "count": _require_int(
                entry["count"],
                label=f"skip baseline entries[{index}].count",
                minimum=1,
            ),
            "line": _require_int(
                entry["line"],
                label=f"skip baseline entries[{index}].line",
                minimum=1,
            ),
            "path": _repo_relative_path(
                entry["path"],
                label=f"skip baseline entries[{index}].path",
            ),
            "reason": _require_string(
                entry["reason"],
                label=f"skip baseline entries[{index}].reason",
            ),
        }
        entries.append(normalized)
    canonical_entries = sorted(
        entries,
        key=lambda row: (
            str(row["path"]).encode("utf-8"),
            int(row["line"]),
            str(row["reason"]).encode("utf-8"),
            int(row["count"]),
        ),
    )
    if entries != canonical_entries or len({_canonical_bytes(row) for row in entries}) != len(
        entries
    ):
        raise Phase0EvidenceError("skip baseline entries are not canonical and unique")
    total = sum(int(row["count"]) for row in entries)
    if (
        payload.get("expected_skip_count") != 42
        or payload.get("observed_skip_count") != 42
        or total != 42
    ):
        raise Phase0EvidenceError("skip baseline count must sum exactly to 42")
    claims = _require_exact_keys(
        payload.get("claims"),
        {
            "errors",
            "exit_code",
            "failed",
            "passed",
            "raw_output_sha256",
            "skip_allowlist_sha256",
            "skipped",
            "xfail",
            "xpass",
        },
        label="skip baseline claims",
    )
    for key in ("exit_code", "failed", "errors", "xfail", "xpass"):
        if claims[key] != 0 or type(claims[key]) is not int:
            raise Phase0EvidenceError(f"skip baseline claims.{key} must be integer zero")
    _require_int(claims["passed"], label="skip baseline claims.passed")
    if claims["skipped"] != 42 or type(claims["skipped"]) is not int:
        raise Phase0EvidenceError("skip baseline claims.skipped must equal 42")
    _require_sha256(
        claims["raw_output_sha256"],
        label="skip baseline claims.raw_output_sha256",
    )
    if claims["skip_allowlist_sha256"] != _sha256(_canonical_bytes(entries)):
        raise Phase0EvidenceError("skip baseline row digest mismatch")
    return entries


def _validate_v2_dependency_evidence(
    payload: Mapping[str, Any],
    *,
    binding: Mapping[str, Any],
    native_log_binding: Mapping[str, Any],
    native_log_receipt: Mapping[str, Any],
    native_framed: bytes,
    repo_root: Path,
    session_binding: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    toolchain: Mapping[str, Any],
    package_source_superset: Mapping[str, Any],
    protected_roots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if (
        payload.get("schema_version") != DEPENDENCY_RECEIPT_VERSION
        or payload.get("status") != NATIVE_ACCEPTED_STATUS
        or payload.get("accepted") is not True
        or payload.get("dependency_environment_accepted") is not True
        or payload.get("native_dependency_environment_accepted") is not True
        or payload.get("offline_only") is not True
        or payload.get("network_actions_performed") is not False
        or payload.get("repackaged_artifacts") is not False
        or payload.get("step") != _expected_v2_step(1)
        or payload.get("limitations") != NORMATIVE_LIMITATIONS
    ):
        raise Phase0EvidenceError("native dependency v2 receipt is not accepted")
    _validate_v2_session_binding(
        payload.get("session_binding"),
        expected=session_binding,
        label="native dependency session_binding",
    )
    producer = _require_exact_keys(
        payload.get("producer"),
        {"path", "sha256", "size_bytes", "version"},
        label="native dependency producer",
    )
    _validate_v2_producer_binding(
        producer,
        expected_path=repo_root / "scripts/v17_offline_dependency_evidence.py",
        expected_version=DEPENDENCY_RECEIPT_VERSION,
        label="native dependency producer",
    )
    if (
        _validate_source_binding(
            payload.get("source"),
            label="native dependency source",
        )
        != source_binding
    ):
        raise Phase0EvidenceError("native dependency source mismatch")
    if (
        _validate_v2_toolchain(
            payload.get("toolchain"),
            label="native dependency toolchain",
            live=False,
        )
        != toolchain
    ):
        raise Phase0EvidenceError("native dependency toolchain mismatch")
    if (
        _validate_v2_namespace_binding(
            payload.get("package_source_superset"),
            label="native dependency package_source_superset",
        )
        != package_source_superset
    ):
        raise Phase0EvidenceError("native dependency package source mismatch")
    if _validate_v2_protected_roots(
        payload.get("protected_roots"),
        label="native dependency protected_roots",
    ) != list(protected_roots):
        raise Phase0EvidenceError("native dependency protected roots mismatch")
    native_log = _require_exact_keys(
        payload.get("native_sync_log"),
        {
            "framed_output_sha256",
            "framed_output_size_bytes",
            "mode",
            "outcome",
            "path",
            "receipt_semantic_sha256",
            "receipt_version",
            "sha256",
            "size_bytes",
        },
        label="native dependency native_sync_log",
    )
    if (
        native_log["path"] != native_log_binding["path"]
        or native_log["sha256"] != native_log_binding["sha256"]
        or native_log["size_bytes"] != native_log_binding["size_bytes"]
        or native_log["mode"] != "0600"
        or native_log["receipt_version"] != COMMAND_RECEIPT_VERSION
        or native_log["receipt_semantic_sha256"] != native_log_receipt["semantic_sha256"]
        or native_log["framed_output_sha256"] != _sha256(native_framed)
        or native_log["framed_output_size_bytes"] != len(native_framed)
        or native_log["outcome"] != "PASSED"
    ):
        raise Phase0EvidenceError("native dependency log projection mismatch")
    native_sync = payload.get("native_sync")
    if type(native_sync) is not dict or (
        native_sync.get("receipt_version") != COMMAND_RECEIPT_VERSION
        or native_sync.get("receipt_semantic_sha256") != native_log_receipt["semantic_sha256"]
        or native_sync.get("receipt_step") != _expected_v2_step(0)
        or native_sync.get("outcome") != "PASSED"
        or native_sync.get("claims") != {"exit_code": 0}
        or native_sync.get("validated") is not True
        or native_sync.get("command") != native_log_receipt["commands"][0]
    ):
        raise Phase0EvidenceError("native dependency sync projection mismatch")
    lock = payload.get("lock_reconciliation")
    installed = payload.get("installed_reconciliation")
    pip_absence = payload.get("pip_absence")
    if (
        type(lock) is not dict
        or lock.get("exact_match") is not True
        or type(installed) is not dict
        or installed.get("exact_match") is not True
        or type(pip_absence) is not dict
        or pip_absence.get("accepted") is not True
        or payload.get("invalid") != []
    ):
        raise Phase0EvidenceError("native dependency reconciliation is not exact")
    if binding["size_bytes"] <= 0:
        raise Phase0EvidenceError("native dependency receipt is empty")
    return dict(payload)


def _validate_v2_executable_binding(
    value: Any,
    *,
    label: str,
    live: bool,
) -> dict[str, Any]:
    binding = _require_exact_keys(
        value,
        {
            "executable",
            "implementation",
            "lexical_path",
            "mode",
            "realpath",
            "sha256",
            "size_bytes",
            "version",
            "version_info",
        },
        label=label,
    )
    lexical = Path(_require_absolute_path(binding["lexical_path"], label=f"{label}.lexical_path"))
    _require_absolute_path(binding["realpath"], label=f"{label}.realpath")
    _require_sha256(binding["sha256"], label=f"{label}.sha256")
    _require_int(binding["size_bytes"], label=f"{label}.size_bytes", minimum=1)
    if (
        binding["mode"] != "0755"
        or binding["executable"] is not True
        or binding["implementation"] != "cpython"
        or binding["version"] != "3.13.7"
        or binding["version_info"] != [3, 13, 7]
    ):
        raise Phase0EvidenceError(f"{label} interpreter identity mismatch")
    if live:
        raw, observed = _stable_regular_file(Path(str(binding["realpath"])), require_private=False)
        if (
            str(lexical.resolve(strict=True)) != binding["realpath"]
            or _sha256(raw) != binding["sha256"]
            or len(raw) != binding["size_bytes"]
            or f"{stat.S_IMODE(observed.st_mode):04o}" != binding["mode"]
            or not observed.st_mode & 0o111
        ):
            raise Phase0EvidenceError(f"{label} live readback mismatch")
    return dict(binding)


def _live_v2_hatch_revalidation(
    package_payload: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    provenance = package_payload.get("build_install_provenance")
    if type(provenance) is not dict:
        raise Phase0EvidenceError("package v2 provenance must be an object")
    physical_session = _require_exact_keys(
        provenance.get("package_source_superset_session"),
        {"after_rows", "before_rows", "parity_rows", "row_count", "sha256"},
        label="package source superset session",
    )
    hatch_session = _require_exact_keys(
        provenance.get("hatch_source_namespace_session"),
        {
            "after_rows",
            "before_rows",
            "parity_rows",
            "row_count",
            "selector_binding",
            "sha256",
            "wheel_projection_sha256",
        },
        label="Hatch source namespace session",
    )
    selector = _require_exact_keys(
        hatch_session["selector_binding"],
        {
            "app_target_importability_verified",
            "build_environment",
            "build_python",
            "hatchling_version",
            "probe_code_sha256",
            "repo_root",
            "retained_for_external_revalidation",
            "selector_modules",
            "targets",
        },
        label="Hatch selector binding",
    )
    if (
        selector["app_target_importability_verified"] is not False
        or selector["hatchling_version"] != "1.31.0"
        or selector["repo_root"] != str(repo_root)
        or selector["retained_for_external_revalidation"] is not True
        or selector["targets"] != ["sdist", "wheel"]
    ):
        raise Phase0EvidenceError("Hatch selector authority mismatch")
    build_environment = Path(
        _require_absolute_path(
            selector["build_environment"],
            label="Hatch selector build_environment",
        )
    )
    if _path_within(build_environment, repo_root):
        raise Phase0EvidenceError("retained Hatch build environment is inside repository")
    build_python = _validate_v2_executable_binding(
        selector["build_python"],
        label="Hatch selector build_python",
        live=True,
    )
    package_module_path = repo_root / "scripts/v17_phase0_package_evidence.py"
    package_module = _load_local_module(
        package_module_path,
        module_name="_phase0_package_evidence_revalidation",
    )
    try:
        probe_code = package_module._hatch_selector_probe_code()
    except AttributeError as exc:
        raise Phase0EvidenceError("package producer lacks the Hatch selector probe") from exc
    if selector["probe_code_sha256"] != _sha256(probe_code.encode("utf-8")):
        raise Phase0EvidenceError("Hatch selector probe code binding mismatch")
    try:
        completed = subprocess.run(
            [
                str(build_python["lexical_path"]),
                "-I",
                "-B",
                "-c",
                probe_code,
                str(repo_root),
            ],
            cwd=repo_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
            },
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Phase0EvidenceError("retained Hatch selector revalidation failed") from exc
    if completed.returncode != 0 or completed.stderr or len(completed.stdout) > 64 * 1024 * 1024:
        raise Phase0EvidenceError("retained Hatch selector revalidation did not pass")
    live_probe = _load_canonical_json_resource(
        completed.stdout,
        label="retained Hatch selector output",
    )
    try:
        live = package_module._validate_selector_probe(
            live_probe,
            label="retained Hatch selector",
        )
    except Exception as exc:
        raise Phase0EvidenceError("retained Hatch selector output is invalid") from exc
    live_physical = live["package_source_superset"]
    live_hatch = live["hatch_source_namespace"]
    for key in ("before_rows", "parity_rows", "after_rows"):
        if physical_session[key] != live_physical["rows"]:
            raise Phase0EvidenceError("package physical source rows changed after build")
        if hatch_session[key] != live_hatch["rows"]:
            raise Phase0EvidenceError("Hatch source rows changed after build")
    if (
        physical_session["row_count"] != live_physical["row_count"]
        or physical_session["sha256"] != live_physical["sha256"]
        or hatch_session["row_count"] != live_hatch["row_count"]
        or hatch_session["sha256"] != live_hatch["sha256"]
        or hatch_session["wheel_projection_sha256"] != live_hatch["wheel_projection_sha256"]
        or selector["selector_modules"] != live["selector_modules"]
        or package_payload.get("package_source_superset")
        != {
            "row_count": live_physical["row_count"],
            "sha256": live_physical["sha256"],
        }
        or package_payload.get("hatch_source_namespace")
        != {
            "row_count": live_hatch["row_count"],
            "sha256": live_hatch["sha256"],
        }
        or package_payload.get("package_inventory_sha256") != live_hatch["wheel_projection_sha256"]
    ):
        raise Phase0EvidenceError("retained Hatch selector projection mismatch")
    return (
        {
            "row_count": live_physical["row_count"],
            "sha256": live_physical["sha256"],
        },
        {
            "row_count": live_hatch["row_count"],
            "sha256": live_hatch["sha256"],
        },
    )


def _validate_v2_package_evidence(
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
    session_binding: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    toolchain: Mapping[str, Any],
    package_source_superset: Mapping[str, Any],
    protected_roots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if (
        payload.get("version") != PACKAGE_EVIDENCE_VERSION
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("status") != "SEALED"
        or payload.get("accepted") is not True
        or payload.get("authority") is not False
        or payload.get("offline_only") is not True
        or payload.get("network_actions_performed") is not False
        or payload.get("phase0_gate_roles") != list(GATE_ROLES)
        or payload.get("step") != _expected_v2_step(8)
        or payload.get("limitations") != NORMATIVE_LIMITATIONS
    ):
        raise Phase0EvidenceError("package v2 evidence identity mismatch")
    _validate_v2_session_binding(
        payload.get("session_binding"),
        expected=session_binding,
        label="package v2 session_binding",
    )
    producer = _validate_v2_producer_binding(
        payload.get("producer"),
        expected_path=repo_root / "scripts/v17_phase0_package_evidence.py",
        expected_version=PACKAGE_PRODUCER_VERSION,
        label="package v2 producer",
    )
    if (
        _validate_source_binding(
            payload.get("source_binding"),
            label="package v2 source_binding",
        )
        != source_binding
    ):
        raise Phase0EvidenceError("package v2 source binding mismatch")
    if (
        _validate_v2_toolchain(
            payload.get("toolchain_binding"),
            label="package v2 toolchain_binding",
            live=True,
        )
        != toolchain
    ):
        raise Phase0EvidenceError("package v2 toolchain mismatch")
    if (
        _validate_v2_namespace_binding(
            payload.get("package_source_superset"),
            label="package v2 package_source_superset",
        )
        != package_source_superset
    ):
        raise Phase0EvidenceError("package v2 source superset mismatch")
    for key in ("protected_roots_before", "protected_roots_after"):
        if _validate_v2_protected_roots(
            payload.get(key),
            label=f"package v2 {key}",
        ) != list(protected_roots):
            raise Phase0EvidenceError("package v2 protected roots mismatch")
    _live_v2_hatch_revalidation(payload, repo_root=repo_root)
    return producer


def _resolve_v2_bundle_root(bundle_root: Path, *, repo_root: Path) -> Path:
    absolute = bundle_root.absolute()
    try:
        observed = absolute.lstat()
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise Phase0EvidenceError("Phase 0 bundle root is unavailable") from exc
    if (
        resolved != absolute
        or stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o700
        or observed.st_uid != os.getuid()
        or _path_within(resolved, repo_root)
        or any(
            _path_within(resolved, path) or _path_within(path, resolved)
            for _id, path in PROTECTED_ROOT_SPECS
        )
    ):
        raise Phase0EvidenceError("Phase 0 bundle root must be owner-private 0700 and isolated")
    return resolved


def _publish_v2_resource_exact_once(
    path: Path,
    payload: Mapping[str, Any],
    *,
    artifact_version: str,
    schemas: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], bytes]:
    _validate_v2_schema(
        payload,
        artifact_version=artifact_version,
        schemas=schemas,
        label=path.name,
    )
    _validate_v2_seal(payload, label=path.name)
    raw = _canonical_resource_bytes(payload)
    parent = _private_parent(path, label=path.name)
    parent_fd = _open_private_parent_fd(parent, label=path.name)
    staged_name: str | None = None
    linked = False
    try:
        _assert_private_parent_fd_current(parent, parent_fd, label=path.name)
        if _entry_exists_at(parent_fd, path.name):
            raise Phase0EvidenceError(f"Phase 0 output already exists: {path.name}")
        staged_name = _stage_private_file_at(parent_fd, path.name, raw)
        _assert_private_parent_fd_current(parent, parent_fd, label=path.name)
        try:
            os.link(
                staged_name,
                path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            linked = True
            os.fsync(parent_fd)
            _assert_private_parent_fd_current(parent, parent_fd, label=path.name)
            readback, readback_stat = _stable_private_file_at(
                parent_fd,
                path.name,
                expected_nlink=2,
            )
            staged_stat = os.stat(
                staged_name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if readback != raw or (readback_stat.st_dev, readback_stat.st_ino) != (
                staged_stat.st_dev,
                staged_stat.st_ino,
            ):
                raise Phase0EvidenceError(f"Phase 0 output readback mismatch: {path.name}")
            parsed = _load_canonical_json_resource(readback, label=f"{path.name} readback")
            _validate_v2_schema(
                parsed,
                artifact_version=artifact_version,
                schemas=schemas,
                label=f"{path.name} readback",
            )
            _validate_v2_seal(parsed, label=f"{path.name} readback")
            os.unlink(staged_name, dir_fd=parent_fd)
            os.fsync(parent_fd)
            staged_name = None
            final_stat = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            if final_stat.st_nlink != 1 or stat.S_IMODE(final_stat.st_mode) != 0o600:
                raise Phase0EvidenceError(f"Phase 0 output final identity mismatch: {path.name}")
        except OSError as exc:
            raise Phase0EvidenceError(
                f"Phase 0 exact-once publication failed: {path.name}"
            ) from exc
    finally:
        os.close(parent_fd)
    if not linked:
        raise Phase0EvidenceError(f"Phase 0 output was not linked: {path.name}")
    return (
        {
            "mode": "0600",
            "path": str(path),
            "sha256": _sha256(raw),
            "size_bytes": len(raw),
        },
        raw,
    )


def _publish_v2_index_pair_exact_once(
    *,
    index_path: Path,
    sidecar_path: Path,
    report: Mapping[str, Any],
    schemas: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], bytes]:
    _validate_v2_schema(
        report,
        artifact_version=EVIDENCE_INDEX_VERSION,
        schemas=schemas,
        label="evidence index",
    )
    _validate_v2_seal(report, label="evidence index")
    report_raw = _canonical_resource_bytes(report)
    report_sha256 = _sha256(report_raw)
    sidecar_raw = f"{report_sha256}  {index_path.name}\n".encode("ascii")
    parent = _private_parent(index_path, label="evidence index")
    if sidecar_path.parent != parent:
        raise Phase0EvidenceError("evidence index sidecar parent mismatch")
    parent_fd = _open_private_parent_fd(parent, label="evidence index")
    report_stage: str | None = None
    sidecar_stage: str | None = None
    try:
        for name in (index_path.name, sidecar_path.name):
            if _entry_exists_at(parent_fd, name):
                raise Phase0EvidenceError(f"Phase 0 output already exists: {name}")
        report_stage = _stage_private_file_at(parent_fd, index_path.name, report_raw)
        sidecar_stage = _stage_private_file_at(parent_fd, sidecar_path.name, sidecar_raw)
        os.link(
            sidecar_stage,
            sidecar_path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        os.fsync(parent_fd)
        os.link(
            report_stage,
            index_path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        os.fsync(parent_fd)
        linked_report, report_stat = _stable_private_file_at(
            parent_fd,
            index_path.name,
            expected_nlink=2,
        )
        linked_sidecar, sidecar_stat = _stable_private_file_at(
            parent_fd,
            sidecar_path.name,
            expected_nlink=2,
        )
        if linked_report != report_raw or linked_sidecar != sidecar_raw:
            raise Phase0EvidenceError("evidence index pair linked readback mismatch")
        parsed = _load_canonical_json_resource(linked_report, label="evidence index readback")
        _validate_v2_schema(
            parsed,
            artifact_version=EVIDENCE_INDEX_VERSION,
            schemas=schemas,
            label="evidence index readback",
        )
        _validate_v2_seal(parsed, label="evidence index readback")
        for staged, installed_stat in (
            (report_stage, report_stat),
            (sidecar_stage, sidecar_stat),
        ):
            staged_stat = os.stat(staged, dir_fd=parent_fd, follow_symlinks=False)
            if (staged_stat.st_dev, staged_stat.st_ino) != (
                installed_stat.st_dev,
                installed_stat.st_ino,
            ):
                raise Phase0EvidenceError("evidence index staged inode mismatch")
        os.unlink(report_stage, dir_fd=parent_fd)
        report_stage = None
        os.unlink(sidecar_stage, dir_fd=parent_fd)
        sidecar_stage = None
        os.fsync(parent_fd)
        for name in (index_path.name, sidecar_path.name):
            final_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if final_stat.st_nlink != 1 or stat.S_IMODE(final_stat.st_mode) != 0o600:
                raise Phase0EvidenceError("evidence index pair final identity mismatch")
    except OSError as exc:
        raise Phase0EvidenceError("evidence index pair exact-once publication failed") from exc
    finally:
        os.close(parent_fd)
    return (
        {
            "mode": "0600",
            "path": str(index_path),
            "sha256": report_sha256,
            "size_bytes": len(report_raw),
        },
        report_raw,
    )


def _validate_session_external_binding(
    value: Any,
    *,
    actual_binding: Mapping[str, Any],
    semantic_sha256: str,
    label: str,
) -> None:
    if type(value) is not dict:
        raise Phase0EvidenceError(f"{label} must be an object")
    for key in ("path", "sha256", "size_bytes", "semantic_sha256"):
        if key not in value:
            raise Phase0EvidenceError(f"{label} is missing {key}")
    if (
        value["path"] != actual_binding["path"]
        or value["sha256"] != actual_binding["sha256"]
        or value["size_bytes"] != actual_binding["size_bytes"]
        or value["semantic_sha256"] != semantic_sha256
    ):
        raise Phase0EvidenceError(f"{label} mismatch")


def _live_v2_package_source_superset(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    parity_module = _load_local_module(
        repo_root / "quant_investor/v17_v2_contract/package_parity.py",
        module_name="_phase0_package_parity_source",
    )
    try:
        full = parity_module.collect_physical_source_superset(
            repo_root,
            extra_paths=("README.md", "pyproject.toml", "requirements.txt"),
        )
    except Exception as exc:
        raise Phase0EvidenceError("physical package source collection failed") from exc
    if (
        type(full) is not dict
        or set(full) != {"row_count", "rows", "sha256"}
        or type(full["rows"]) is not list
        or type(full["row_count"]) is not int
        or full["row_count"] != len(full["rows"])
    ):
        raise Phase0EvidenceError("physical package source binding is invalid")
    binding = {
        "row_count": full["row_count"],
        "sha256": full["sha256"],
    }
    _validate_v2_namespace_binding(binding, label="physical package source binding")
    return dict(full), binding


def _prepare_v2_session_context(
    *,
    repo_root: Path,
    bundle_root: Path,
    classification_manifest: Path,
    skip_baseline: Path,
    session_manifest: Path,
    expected_session_sha256: str,
    expected_schema_bindings: Sequence[Mapping[str, Any]] | None = None,
    expected_schema_raw: Mapping[str, bytes] | None = None,
) -> dict[str, Any]:
    expected_session_sha256 = _require_sha256(
        expected_session_sha256,
        label="expected session SHA-256",
    )
    root_candidate = repo_root.resolve(strict=True)
    schemas, schema_bindings, schema_raw = _load_v2_schema_registry(root_candidate)
    if expected_schema_bindings is not None and (
        list(expected_schema_bindings) != schema_bindings
        or dict(expected_schema_raw or {}) != schema_raw
    ):
        raise Phase0EvidenceError("Phase 0 schema registry drifted")
    bundle = _resolve_v2_bundle_root(bundle_root, repo_root=root_candidate)
    if session_manifest.absolute() != bundle / "00_session.json":
        raise Phase0EvidenceError("session manifest must be bundle/00_session.json")
    session, session_file_binding, session_raw = _load_v2_resource(
        session_manifest,
        artifact_version=SESSION_VERSION,
        schemas=schemas,
        repo_root=root_candidate,
        label="Phase 0 session",
    )
    if session_file_binding["sha256"] != expected_session_sha256:
        raise Phase0EvidenceError("expected session SHA-256 mismatch")
    base_commit = _validate_base_commit(session.get("base_commit"))
    root, base_commit = _resolve_repo(root_candidate, base_commit)
    if root != root_candidate:
        raise Phase0EvidenceError("session repository root mismatch")
    _require_exact_keys(session, V2_SESSION_KEYS, label="Phase 0 session")
    if (
        session["version"] != SESSION_VERSION
        or session["protocol_version"] != PROTOCOL_VERSION
        or session["status"] != "INITIALIZED"
        or session["authority"] is not False
        or session["repo_root"] != str(root)
        or session["base_commit"] != base_commit
        or session["limitations"] != NORMATIVE_LIMITATIONS
        or session["schemas"] != schema_bindings
    ):
        raise Phase0EvidenceError("Phase 0 session identity mismatch")
    session_binding = _v2_session_binding(session, session_file_binding)
    session_producer = _require_exact_keys(
        session["producer"],
        {"path", "sha256", "size_bytes", "version"},
        label="Phase 0 session producer",
    )
    expected_session_producer = _v2_live_producer_binding(
        root / "scripts/v17_phase0_evidence_session.py",
        version=SESSION_PRODUCER_VERSION,
    )
    if session_producer != expected_session_producer:
        raise Phase0EvidenceError("Phase 0 session producer mismatch")
    _validate_v2_gate_plan(session["gate_plan"], repo_root=root)
    source_before = _git_snapshot(root, base_commit)
    source_state = _public_source_state(source_before)
    source_binding = _source_binding_from_state(source_state)
    if session["source_state"] != source_state or session["source_binding"] != source_binding:
        raise Phase0EvidenceError("Phase 0 session source seal is stale")
    physical_full, package_source_superset = _live_v2_package_source_superset(root)
    if session["package_source_superset"] != package_source_superset:
        raise Phase0EvidenceError("Phase 0 session physical package source is stale")
    protected_roots = _sample_v2_protected_roots()
    if (
        _validate_v2_protected_roots(
            session["protected_roots"],
            label="Phase 0 session protected_roots",
        )
        != protected_roots
    ):
        raise Phase0EvidenceError("Phase 0 session protected roots are stale")
    toolchain = _validate_v2_toolchain(
        session["toolchain_binding"],
        label="Phase 0 session toolchain_binding",
        live=True,
    )
    if session["uv_cache_binding"] != toolchain["uv_cache"]:
        raise Phase0EvidenceError("Phase 0 session uv cache binding mismatch")
    classification, classification_binding, classification_raw = _load_v2_resource(
        classification_manifest,
        artifact_version=CLASSIFICATION_VERSION,
        schemas=schemas,
        repo_root=root,
        label="pre-existing classification",
    )
    pre_existing_paths = _parse_classification_manifest_v2(
        classification_raw,
        base_commit=base_commit,
        schemas=schemas,
    )
    _validate_session_external_binding(
        session["classification_binding"],
        actual_binding=classification_binding,
        semantic_sha256=classification["semantic_sha256"],
        label="session classification_binding",
    )
    skip, skip_binding, skip_raw = _load_v2_resource(
        skip_baseline,
        artifact_version=SKIP_BASELINE_VERSION,
        schemas=schemas,
        repo_root=root,
        label="frozen skip baseline",
    )
    _validate_v2_skip_baseline(
        skip,
        source_state=source_state,
        source_binding=source_binding,
        toolchain=toolchain,
        package_source_superset=package_source_superset,
        protected_roots=protected_roots,
        repo_root=root,
    )
    _validate_session_external_binding(
        session["skip_baseline_binding"],
        actual_binding=skip_binding,
        semantic_sha256=skip["semantic_sha256"],
        label="session skip_baseline_binding",
    )
    patterns = _allowed_patterns(tuple(PHASE0_ALLOWED_PATTERN_REGISTRY))
    classified_paths = _classify_dirty_paths(
        source_state["dirty_paths"],
        allowed_patterns=patterns,
        pre_existing_paths=pre_existing_paths,
    )
    source_after = _git_snapshot(root, base_commit)
    _assert_snapshot_equal(source_before, source_after)
    physical_after, package_after = _live_v2_package_source_superset(root)
    if physical_full != physical_after or package_source_superset != package_after:
        raise Phase0EvidenceError("physical package source changed during session readback")
    if _sample_v2_protected_roots() != protected_roots:
        raise Phase0EvidenceError("protected roots changed during session readback")
    return {
        "base_commit": base_commit,
        "bundle_root": bundle,
        "classification": classification,
        "classification_binding": classification_binding,
        "classification_raw": classification_raw,
        "classified_paths": classified_paths,
        "package_source_full": physical_full,
        "package_source_superset": package_source_superset,
        "patterns": patterns,
        "protected_roots": protected_roots,
        "repo_root": root,
        "schema_bindings": schema_bindings,
        "schema_raw": schema_raw,
        "schemas": schemas,
        "session": session,
        "session_binding": session_binding,
        "session_raw": session_raw,
        "skip": skip,
        "skip_binding": skip_binding,
        "skip_raw": skip_raw,
        "source_binding": source_binding,
        "source_snapshot": source_before,
        "source_state": source_state,
        "toolchain": toolchain,
    }


def _validate_v2_hash_evidence(
    payload: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    hatch_source_namespace: Mapping[str, Any],
) -> dict[str, Any]:
    repo_root = Path(str(context["repo_root"]))
    if (
        payload.get("version") != HASH_FREEZE_VERSION
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("accepted") is not True
        or payload.get("authority") is not False
        or payload.get("step") != _expected_v2_step(9)
    ):
        raise Phase0EvidenceError("hash-freeze v2 identity mismatch")
    _validate_v2_session_binding(
        payload.get("session_binding"),
        expected=context["session_binding"],
        label="hash-freeze session_binding",
    )
    _validate_v2_producer_binding(
        payload.get("producer"),
        expected_path=repo_root / "scripts/v17_phase0_evidence_index.py",
        expected_version=INDEX_PRODUCER_VERSION,
        label="hash-freeze producer",
    )
    if (
        _validate_source_binding(
            payload.get("source_binding"),
            label="hash-freeze source_binding",
        )
        != context["source_binding"]
    ):
        raise Phase0EvidenceError("hash-freeze source binding mismatch")
    if (
        _validate_v2_toolchain(
            payload.get("toolchain_binding"),
            label="hash-freeze toolchain_binding",
            live=True,
        )
        != context["toolchain"]
    ):
        raise Phase0EvidenceError("hash-freeze toolchain mismatch")
    if (
        _validate_v2_namespace_binding(
            payload.get("package_source_superset"),
            label="hash-freeze package_source_superset",
        )
        != context["package_source_superset"]
        or _validate_v2_namespace_binding(
            payload.get("hatch_source_namespace"),
            label="hash-freeze hatch_source_namespace",
        )
        != hatch_source_namespace
    ):
        raise Phase0EvidenceError("hash-freeze package namespace mismatch")
    for key in ("protected_roots_before", "protected_roots_after"):
        if (
            _validate_v2_protected_roots(
                payload.get(key),
                label=f"hash-freeze {key}",
            )
            != context["protected_roots"]
        ):
            raise Phase0EvidenceError("hash-freeze protected roots mismatch")
    hashes = payload.get("hashes")
    current_hashes = _current_hash_freeze(repo_root)
    if type(hashes) is not dict or hashes != current_hashes:
        raise Phase0EvidenceError("hash-freeze bytes differ from current authority files")
    return dict(payload)


def _read_v2_gate_evidence(
    context: Mapping[str, Any],
    *,
    include_hash_freeze: bool,
) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, Any]]:
    repo_root = Path(str(context["repo_root"]))
    bundle_root = Path(str(context["bundle_root"]))
    schemas = context["schemas"]
    skip = context["skip"]
    skip_raw = context["skip_raw"]
    records: list[dict[str, Any]] = []
    raw_by_role: dict[str, bytes] = {}
    payload_by_role: dict[str, Any] = {}
    receipt_by_role: dict[str, Any] = {}
    framed_by_role: dict[str, bytes] = {}
    package_payload: dict[str, Any] | None = None
    hatch_source_namespace: dict[str, Any] | None = None
    command_work_root: Path | None = None
    command_tool_versions: dict[str, str] | None = None
    limit = len(GATE_ROLES) if include_hash_freeze else len(GATE_ROLES) - 1
    for index in range(limit):
        role = GATE_ROLES[index]
        path = bundle_root / GATE_FILENAMES[index]
        binding, raw = _v2_file_binding(
            path,
            repo_root=repo_root,
            label=f"gate {role}",
        )
        if Path(str(binding["path"])).parent != bundle_root:
            raise Phase0EvidenceError(f"{role} is outside the fixed bundle root")
        claims: dict[str, Any]
        producer_binding: dict[str, Any]
        semantic_sha256: str
        if role == "full_offline_suite":
            receipt, parsed_streams, framed = _parse_framed_main_suite_receipt_v1(
                raw,
                label=role,
            )
            _validate_v2_schema(
                receipt,
                artifact_version=MAIN_SUITE_RECEIPT_VERSION,
                schemas=schemas,
                label=f"{role} main-suite receipt",
            )
            _validate_v2_seal(receipt, label=f"{role} main-suite receipt")
            binary_frames = _validate_main_suite_attestation_frames(
                parsed_streams["attestation"],
                label=role,
            )
            decoded_frames = _decode_main_suite_attestation_frames(
                binary_frames,
                label=role,
            )
            policy, policy_bindings, producer_binding = _load_v2_main_suite_policy(
                repo_root=repo_root,
                schemas=schemas,
            )
            claims = _validate_v2_main_suite_semantics(
                receipt,
                streams=parsed_streams,
                frames=decoded_frames,
                repo_root=repo_root,
                policy=policy,
                policy_bindings=policy_bindings,
                session_binding=context["session_binding"],
                skip_baseline=skip,
                skip_baseline_raw=skip_raw,
                base_commit=context["base_commit"],
                source_state=context["source_state"],
                package_source_full=context["package_source_full"],
            )
            semantic_sha256 = receipt["semantic_sha256"]
            receipt_by_role[role] = receipt
            framed_by_role[role] = framed
            payload_by_role[role] = receipt
        elif role in LOG_ROLES:
            receipt, streams, framed = _parse_framed_command_receipt_v2(
                raw,
                schemas=schemas,
                label=role,
            )
            _validate_v2_command_context(
                receipt,
                gate_index=index,
                repo_root=repo_root,
                session_binding=context["session_binding"],
                source_binding=context["source_binding"],
                toolchain=context["toolchain"],
                package_source_superset=context["package_source_superset"],
                protected_roots=context["protected_roots"],
            )
            command_work_root = _validate_v2_command_identity(
                role,
                receipt,
                repo_root=repo_root,
                toolchain=context["toolchain"],
                work_root=command_work_root,
                tool_versions=command_tool_versions,
            )
            output = _command_output_bytes(streams)
            claims = _validate_v2_gate_claims(
                role,
                receipt.get("claims"),
                output=output,
                skip_baseline=skip,
                skip_baseline_raw=skip_raw,
            )
            for command_index, command in enumerate(receipt["commands"]):
                if command["cwd"] != str(repo_root):
                    raise Phase0EvidenceError(f"{role} command cwd mismatch")
                serialized_command = _canonical_bytes(
                    {
                        "argv": command["argv"],
                        "cwd": command["cwd"],
                        "environment": command["environment"],
                    }
                )
                for _root_id, protected_path in PROTECTED_ROOT_SPECS:
                    if str(protected_path).encode("utf-8") in serialized_command:
                        raise Phase0EvidenceError(
                            f"{role} command[{command_index}] directly references protected root"
                        )
            producer_binding = dict(receipt["producer"])
            semantic_sha256 = receipt["semantic_sha256"]
            receipt_by_role[role] = receipt
            framed_by_role[role] = framed
            payload_by_role[role] = receipt
        elif role == "native_sync_receipt":
            payload, _loaded_binding, loaded_raw = _load_v2_resource(
                path,
                artifact_version=DEPENDENCY_RECEIPT_VERSION,
                schemas=schemas,
                repo_root=repo_root,
                label=role,
            )
            if loaded_raw != raw:
                raise Phase0EvidenceError("native dependency bytes changed during read")
            native_receipt = receipt_by_role.get("native_sync_log")
            native_framed = framed_by_role.get("native_sync_log")
            native_raw_binding = next(
                (prior for prior in records if prior["role"] == "native_sync_log"),
                None,
            )
            if native_receipt is None or native_framed is None or native_raw_binding is None:
                raise Phase0EvidenceError("log-first native dependency binding is incomplete")
            _validate_v2_dependency_evidence(
                payload,
                binding=binding,
                native_log_binding=native_raw_binding,
                native_log_receipt=native_receipt,
                native_framed=native_framed,
                repo_root=repo_root,
                session_binding=context["session_binding"],
                source_binding=context["source_binding"],
                toolchain=context["toolchain"],
                package_source_superset=context["package_source_superset"],
                protected_roots=context["protected_roots"],
            )
            command_tool_versions = _v2_live_tool_versions(payload, repo_root=repo_root)
            claims = _validate_v2_gate_claims(
                role,
                {"accepted": True},
                output=None,
                skip_baseline=None,
                skip_baseline_raw=None,
            )
            producer_binding = _project_v2_producer_binding(payload["producer"])
            semantic_sha256 = payload["semantic_sha256"]
            payload_by_role[role] = payload
        elif role == "package_parity":
            payload, _loaded_binding, loaded_raw = _load_v2_resource(
                path,
                artifact_version=PACKAGE_EVIDENCE_VERSION,
                schemas=schemas,
                repo_root=repo_root,
                label=role,
            )
            if loaded_raw != raw:
                raise Phase0EvidenceError("package evidence bytes changed during read")
            producer_binding = _validate_v2_package_evidence(
                payload,
                repo_root=repo_root,
                session_binding=context["session_binding"],
                source_binding=context["source_binding"],
                toolchain=context["toolchain"],
                package_source_superset=context["package_source_superset"],
                protected_roots=context["protected_roots"],
            )
            claims = _validate_v2_gate_claims(
                role,
                {"accepted": True},
                output=None,
                skip_baseline=None,
                skip_baseline_raw=None,
            )
            semantic_sha256 = payload["semantic_sha256"]
            package_payload = payload
            hatch_source_namespace = dict(payload["hatch_source_namespace"])
            payload_by_role[role] = payload
        elif role == "hash_freeze_readback":
            if package_payload is None or hatch_source_namespace is None:
                raise Phase0EvidenceError("hash-freeze lacks package predecessor")
            payload, _loaded_binding, loaded_raw = _load_v2_resource(
                path,
                artifact_version=HASH_FREEZE_VERSION,
                schemas=schemas,
                repo_root=repo_root,
                label=role,
            )
            if loaded_raw != raw:
                raise Phase0EvidenceError("hash-freeze bytes changed during read")
            _validate_v2_hash_evidence(
                payload,
                context=context,
                hatch_source_namespace=hatch_source_namespace,
            )
            claims = _validate_v2_gate_claims(
                role,
                {"accepted": True},
                output=None,
                skip_baseline=None,
                skip_baseline_raw=None,
            )
            producer_binding = dict(payload["producer"])
            semantic_sha256 = payload["semantic_sha256"]
            payload_by_role[role] = payload
        else:
            raise Phase0EvidenceError(f"unknown artifact role: {role}")
        record = {
            "artifact_version": GATE_ARTIFACT_VERSIONS[index],
            "claims": claims,
            "filename": GATE_FILENAMES[index],
            "id": role,
            "kind": GATE_KINDS[role],
            "mode": binding["mode"],
            "ordinal": GATE_ORDINALS[index],
            "path": binding["path"],
            "producer_binding": producer_binding,
            "role": role,
            "schema_id": GATE_SCHEMA_IDS[index],
            "semantic_sha256": semantic_sha256,
            "session_id": context["session_binding"]["session_id"],
            "sha256": binding["sha256"],
            "size_bytes": binding["size_bytes"],
            "source_binding": dict(context["source_binding"]),
        }
        records.append(record)
        raw_by_role[role] = raw
    if package_payload is None or hatch_source_namespace is None:
        raise Phase0EvidenceError("package parity evidence is missing")
    return (
        records,
        raw_by_role,
        {
            "hatch_source_namespace": hatch_source_namespace,
            "package_payload": package_payload,
            "payload_by_role": payload_by_role,
            "receipt_by_role": receipt_by_role,
        },
    )


def _assert_v2_context_matches(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> None:
    keys = (
        "base_commit",
        "classification_binding",
        "classification_raw",
        "classified_paths",
        "package_source_full",
        "package_source_superset",
        "patterns",
        "protected_roots",
        "schema_bindings",
        "schema_raw",
        "session_binding",
        "session_raw",
        "skip_binding",
        "skip_raw",
        "source_binding",
        "source_state",
        "toolchain",
    )
    for key in keys:
        if expected[key] != observed[key]:
            raise Phase0EvidenceError(f"Phase 0 session context drifted: {key}")


def _refresh_v2_context(context: Mapping[str, Any]) -> dict[str, Any]:
    refreshed = _prepare_v2_session_context(
        repo_root=Path(str(context["repo_root"])),
        bundle_root=Path(str(context["bundle_root"])),
        classification_manifest=Path(str(context["classification_binding"]["path"])),
        skip_baseline=Path(str(context["skip_binding"]["path"])),
        session_manifest=Path(str(context["session_binding"]["path"])),
        expected_session_sha256=str(context["session_binding"]["sha256"]),
        expected_schema_bindings=context["schema_bindings"],
        expected_schema_raw=context["schema_raw"],
    )
    _assert_v2_context_matches(context, refreshed)
    return refreshed


def _build_v2_hash_freeze(
    context: Mapping[str, Any],
    *,
    hatch_source_namespace: Mapping[str, Any],
) -> dict[str, Any]:
    repo_root = Path(str(context["repo_root"]))
    protected_before = _sample_v2_protected_roots()
    hashes = _current_hash_freeze(repo_root)
    protected_after = _sample_v2_protected_roots()
    if protected_before != context["protected_roots"] or protected_after != protected_before:
        raise Phase0EvidenceError("protected roots changed while building hash-freeze")
    payload = _seal(
        {
            "accepted": True,
            "authority": False,
            "hatch_source_namespace": dict(hatch_source_namespace),
            "hashes": hashes,
            "package_source_superset": dict(context["package_source_superset"]),
            "producer": _v2_live_producer_binding(
                repo_root / "scripts/v17_phase0_evidence_index.py",
                version=INDEX_PRODUCER_VERSION,
            ),
            "protected_roots_after": protected_after,
            "protected_roots_before": protected_before,
            "protocol_version": PROTOCOL_VERSION,
            "session_binding": dict(context["session_binding"]),
            "source_binding": dict(context["source_binding"]),
            "step": _expected_v2_step(9),
            "toolchain_binding": dict(context["toolchain"]),
            "version": HASH_FREEZE_VERSION,
        }
    )
    _validate_v2_schema(
        payload,
        artifact_version=HASH_FREEZE_VERSION,
        schemas=context["schemas"],
        label="hash-freeze",
    )
    _validate_v2_hash_evidence(
        payload,
        context=context,
        hatch_source_namespace=hatch_source_namespace,
    )
    return payload


def _project_v2_gate_manifest_rows(
    external_evidence: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len(external_evidence) != len(GATE_ROLES):
        raise Phase0EvidenceError("gate manifest projection lacks ten evidence rows")
    for index, record in enumerate(external_evidence):
        row = {
            "artifact_version": record["artifact_version"],
            "claims": dict(record["claims"]),
            "filename": record["filename"],
            "kind": record["kind"],
            "ordinal": record["ordinal"],
            "producer_binding": dict(record["producer_binding"]),
            "role": record["role"],
            "schema_id": record["schema_id"],
            "semantic_sha256": record["semantic_sha256"],
            "session_id": record["session_id"],
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
        }
        expected = {
            "artifact_version": GATE_ARTIFACT_VERSIONS[index],
            "filename": GATE_FILENAMES[index],
            "kind": GATE_KINDS[GATE_ROLES[index]],
            "ordinal": GATE_ORDINALS[index],
            "role": GATE_ROLES[index],
            "schema_id": GATE_SCHEMA_IDS[index],
            "session_id": record["session_id"],
        }
        for key, expected_value in expected.items():
            if row[key] != expected_value:
                raise Phase0EvidenceError(f"gate manifest projection mismatch: {key}")
        rows.append(row)
    return rows


def _validate_v2_gate_manifest(
    payload: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    expected_rows: Sequence[Mapping[str, Any]],
    hatch_source_namespace: Mapping[str, Any],
) -> dict[str, Any]:
    repo_root = Path(str(context["repo_root"]))
    _require_exact_keys(
        payload,
        {
            "base_commit",
            "gates",
            "hatch_source_namespace",
            "limitations",
            "package_source_superset",
            "producer",
            "protocol_version",
            "semantic_sha256",
            "session_binding",
            "source_binding",
            "version",
        },
        label="gate manifest v2",
    )
    if (
        payload["version"] != GATE_MANIFEST_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["base_commit"] != context["base_commit"]
        or payload["source_binding"] != context["source_binding"]
        or payload["package_source_superset"] != context["package_source_superset"]
        or payload["hatch_source_namespace"] != hatch_source_namespace
        or payload["limitations"] != NORMATIVE_LIMITATIONS
        or payload["gates"] != list(expected_rows)
    ):
        raise Phase0EvidenceError("gate manifest v2 binding mismatch")
    _validate_v2_session_binding(
        payload["session_binding"],
        expected=context["session_binding"],
        label="gate manifest session_binding",
    )
    _validate_v2_producer_binding(
        payload["producer"],
        expected_path=repo_root / "scripts/v17_phase0_evidence_index.py",
        expected_version=INDEX_PRODUCER_VERSION,
        label="gate manifest producer",
    )
    _validate_v2_seal(payload, label="gate manifest v2")
    return dict(payload)


def _build_v2_gate_manifest(
    context: Mapping[str, Any],
    *,
    external_evidence: Sequence[Mapping[str, Any]],
    hatch_source_namespace: Mapping[str, Any],
) -> dict[str, Any]:
    rows = _project_v2_gate_manifest_rows(external_evidence)
    repo_root = Path(str(context["repo_root"]))
    payload = _seal(
        {
            "base_commit": context["base_commit"],
            "gates": rows,
            "hatch_source_namespace": dict(hatch_source_namespace),
            "limitations": list(NORMATIVE_LIMITATIONS),
            "package_source_superset": dict(context["package_source_superset"]),
            "producer": _v2_live_producer_binding(
                repo_root / "scripts/v17_phase0_evidence_index.py",
                version=INDEX_PRODUCER_VERSION,
            ),
            "protocol_version": PROTOCOL_VERSION,
            "session_binding": dict(context["session_binding"]),
            "source_binding": dict(context["source_binding"]),
            "version": GATE_MANIFEST_VERSION,
        }
    )
    _validate_v2_schema(
        payload,
        artifact_version=GATE_MANIFEST_VERSION,
        schemas=context["schemas"],
        label="gate manifest v2",
    )
    return _validate_v2_gate_manifest(
        payload,
        context=context,
        expected_rows=rows,
        hatch_source_namespace=hatch_source_namespace,
    )


def _validate_v2_evidence_index_report(
    payload: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    external_evidence: Sequence[Mapping[str, Any]],
    gate_manifest_binding: Mapping[str, Any],
    hatch_source_namespace: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = {
        "accepted",
        "allowlist",
        "authority",
        "base_commit",
        "classification_provenance",
        "external_evidence",
        "gate_manifest",
        "hatch_source_namespace",
        "limitations",
        "network_actions_performed",
        "offline_only",
        "package_source_superset",
        "protected_roots",
        "protocol_version",
        "repo_root",
        "schemas",
        "semantic_sha256",
        "session_binding",
        "source_binding",
        "source_state",
        "status",
        "version",
    }
    _require_exact_keys(payload, expected_keys, label="evidence index v2")
    if (
        payload["version"] != EVIDENCE_INDEX_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["status"] != "SEALED"
        or payload["accepted"] is not True
        or payload["authority"] is not False
        or payload["offline_only"] is not True
        or payload["network_actions_performed"] is not False
        or payload["base_commit"] != context["base_commit"]
        or payload["classification_provenance"] != CLASSIFICATION_PROVENANCE
        or payload["external_evidence"] != list(external_evidence)
        or payload["gate_manifest"] != dict(gate_manifest_binding)
        or payload["hatch_source_namespace"] != hatch_source_namespace
        or payload["limitations"] != NORMATIVE_LIMITATIONS
        or payload["package_source_superset"] != context["package_source_superset"]
        or payload["protected_roots"] != context["protected_roots"]
        or payload["repo_root"] != str(context["repo_root"])
        or payload["schemas"] != context["schema_bindings"]
        or payload["source_binding"] != context["source_binding"]
        or payload["source_state"] != context["source_state"]
    ):
        raise Phase0EvidenceError("evidence index v2 cross-binding mismatch")
    _validate_v2_session_binding(
        payload["session_binding"],
        expected=context["session_binding"],
        label="evidence index session_binding",
    )
    expected_allowlist = {
        "allowed_phase0_path_patterns": context["patterns"],
        "classified_paths": context["classified_paths"],
        "pre_existing_classification_manifest": dict(context["classification_binding"]),
    }
    if payload["allowlist"] != expected_allowlist:
        raise Phase0EvidenceError("evidence index allowlist mismatch")
    _validate_v2_seal(payload, label="evidence index v2")
    return dict(payload)


def _build_v2_evidence_index_report(
    context: Mapping[str, Any],
    *,
    external_evidence: Sequence[Mapping[str, Any]],
    gate_manifest_binding: Mapping[str, Any],
    hatch_source_namespace: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _seal(
        {
            "accepted": True,
            "allowlist": {
                "allowed_phase0_path_patterns": list(context["patterns"]),
                "classified_paths": list(context["classified_paths"]),
                "pre_existing_classification_manifest": dict(context["classification_binding"]),
            },
            "authority": False,
            "base_commit": context["base_commit"],
            "classification_provenance": CLASSIFICATION_PROVENANCE,
            "external_evidence": [dict(row) for row in external_evidence],
            "gate_manifest": dict(gate_manifest_binding),
            "hatch_source_namespace": dict(hatch_source_namespace),
            "limitations": list(NORMATIVE_LIMITATIONS),
            "network_actions_performed": False,
            "offline_only": True,
            "package_source_superset": dict(context["package_source_superset"]),
            "protected_roots": [dict(row) for row in context["protected_roots"]],
            "protocol_version": PROTOCOL_VERSION,
            "repo_root": str(context["repo_root"]),
            "schemas": [dict(row) for row in context["schema_bindings"]],
            "session_binding": dict(context["session_binding"]),
            "source_binding": dict(context["source_binding"]),
            "source_state": dict(context["source_state"]),
            "status": "SEALED",
            "version": EVIDENCE_INDEX_VERSION,
        }
    )
    _validate_v2_schema(
        payload,
        artifact_version=EVIDENCE_INDEX_VERSION,
        schemas=context["schemas"],
        label="evidence index v2",
    )
    return _validate_v2_evidence_index_report(
        payload,
        context=context,
        external_evidence=external_evidence,
        gate_manifest_binding=gate_manifest_binding,
        hatch_source_namespace=hatch_source_namespace,
    )


def seal_phase0_session(
    repo_root: Path,
    bundle_root: Path,
    classification_manifest: Path,
    skip_baseline: Path,
    session_manifest: Path,
    expected_session_sha256: str,
) -> dict[str, Any]:
    """Validate one fixed Phase 0 session and publish 50/60/70 exact-once."""

    context = _prepare_v2_session_context(
        repo_root=repo_root,
        bundle_root=bundle_root,
        classification_manifest=classification_manifest,
        skip_baseline=skip_baseline,
        session_manifest=session_manifest,
        expected_session_sha256=expected_session_sha256,
    )
    initial_records, _initial_raw, initial_details = _read_v2_gate_evidence(
        context,
        include_hash_freeze=False,
    )
    hatch_source_namespace = initial_details["hatch_source_namespace"]
    context = _refresh_v2_context(context)
    repeat_records, _repeat_raw, repeat_details = _read_v2_gate_evidence(
        context,
        include_hash_freeze=False,
    )
    if initial_records != repeat_records or (
        hatch_source_namespace != repeat_details["hatch_source_namespace"]
    ):
        raise Phase0EvidenceError("pre-hash gate evidence changed during validation")
    hash_payload = _build_v2_hash_freeze(
        context,
        hatch_source_namespace=hatch_source_namespace,
    )
    hash_path = Path(str(context["bundle_root"])) / "50_hash_freeze.json"
    _publish_v2_resource_exact_once(
        hash_path,
        hash_payload,
        artifact_version=HASH_FREEZE_VERSION,
        schemas=context["schemas"],
    )
    context = _refresh_v2_context(context)
    external_records, _external_raw, details = _read_v2_gate_evidence(
        context,
        include_hash_freeze=True,
    )
    if details["hatch_source_namespace"] != hatch_source_namespace:
        raise Phase0EvidenceError("Hatch namespace changed after hash-freeze")
    gate_payload = _build_v2_gate_manifest(
        context,
        external_evidence=external_records,
        hatch_source_namespace=hatch_source_namespace,
    )
    gate_path = Path(str(context["bundle_root"])) / "60_gate_manifest.json"
    gate_binding, _gate_raw = _publish_v2_resource_exact_once(
        gate_path,
        gate_payload,
        artifact_version=GATE_MANIFEST_VERSION,
        schemas=context["schemas"],
    )
    context = _refresh_v2_context(context)
    final_records, _final_raw, final_details = _read_v2_gate_evidence(
        context,
        include_hash_freeze=True,
    )
    if final_records != external_records or (
        final_details["hatch_source_namespace"] != hatch_source_namespace
    ):
        raise Phase0EvidenceError("gate evidence changed after manifest publication")
    loaded_gate, loaded_gate_binding, _loaded_gate_raw = _load_v2_resource(
        gate_path,
        artifact_version=GATE_MANIFEST_VERSION,
        schemas=context["schemas"],
        repo_root=Path(str(context["repo_root"])),
        label="gate manifest linked readback",
    )
    if loaded_gate_binding != gate_binding:
        raise Phase0EvidenceError("gate manifest binding changed after publication")
    _validate_v2_gate_manifest(
        loaded_gate,
        context=context,
        expected_rows=_project_v2_gate_manifest_rows(final_records),
        hatch_source_namespace=hatch_source_namespace,
    )
    report = _build_v2_evidence_index_report(
        context,
        external_evidence=final_records,
        gate_manifest_binding=gate_binding,
        hatch_source_namespace=hatch_source_namespace,
    )
    context = _refresh_v2_context(context)
    prepublish_records, _prepublish_raw, prepublish_details = _read_v2_gate_evidence(
        context,
        include_hash_freeze=True,
    )
    if prepublish_records != final_records or (
        prepublish_details["hatch_source_namespace"] != hatch_source_namespace
    ):
        raise Phase0EvidenceError("gate evidence changed before index publication")
    _validate_v2_schema(
        report,
        artifact_version=EVIDENCE_INDEX_VERSION,
        schemas=context["schemas"],
        label="evidence index prepublication",
    )
    _validate_v2_evidence_index_report(
        report,
        context=context,
        external_evidence=prepublish_records,
        gate_manifest_binding=gate_binding,
        hatch_source_namespace=hatch_source_namespace,
    )
    index_path = Path(str(context["bundle_root"])) / "70_evidence_index.json"
    sidecar_path = Path(str(context["bundle_root"])) / "70_evidence_index.json.sha256"
    _publish_v2_index_pair_exact_once(
        index_path=index_path,
        sidecar_path=sidecar_path,
        report=report,
        schemas=context["schemas"],
    )
    linked_raw, _linked_stat = _stable_regular_file(
        index_path,
        require_private=True,
    )
    linked = _load_canonical_json_resource(linked_raw, label="evidence index final readback")
    context = _refresh_v2_context(context)
    linked_records, _linked_evidence_raw, linked_details = _read_v2_gate_evidence(
        context,
        include_hash_freeze=True,
    )
    if linked != report or linked_records != final_records:
        raise Phase0EvidenceError("final evidence index linked readback mismatch")
    _validate_v2_schema(
        linked,
        artifact_version=EVIDENCE_INDEX_VERSION,
        schemas=context["schemas"],
        label="evidence index final linked readback",
    )
    _validate_v2_evidence_index_report(
        linked,
        context=context,
        external_evidence=linked_records,
        gate_manifest_binding=gate_binding,
        hatch_source_namespace=linked_details["hatch_source_namespace"],
    )
    return dict(linked)


def build_evidence_index(**_legacy_arguments: Any) -> dict[str, Any]:  # type: ignore[no-redef]
    """Reject the retired arbitrary-input v1 builder without compatibility fallback."""

    raise Phase0EvidenceError(
        "Phase 0 evidence-index v1 builder is retired; use seal_phase0_session"
    )


def write_evidence_index_exact_once(**_legacy_arguments: Any) -> str:  # type: ignore[no-redef]
    """Reject the retired arbitrary-output v1 publisher."""

    raise Phase0EvidenceError(
        "Phase 0 evidence-index v1 publisher is retired; use seal_phase0_session"
    )


def validate_evidence_index(  # type: ignore[no-redef]
    value: Any,
    *,
    verify_external: bool,
) -> dict[str, Any]:
    """Validate the v2 index envelope; production readback uses the seal orchestrator."""

    if type(value) is not dict or value.get("version") != EVIDENCE_INDEX_VERSION:
        raise Phase0EvidenceError("Phase 0 evidence-index v1/downgrade is rejected")
    repo_root = Path(_require_absolute_path(value.get("repo_root"), label="repo_root"))
    schemas, _bindings, _raw = _load_v2_schema_registry(repo_root)
    _validate_v2_schema(
        value,
        artifact_version=EVIDENCE_INDEX_VERSION,
        schemas=schemas,
        label="evidence index v2",
    )
    _validate_v2_seal(value, label="evidence index v2")
    if verify_external:
        for row in value.get("external_evidence", []):
            if type(row) is not dict:
                raise Phase0EvidenceError("external evidence row is invalid")
            binding, _raw_value = _external_file_binding(
                Path(str(row.get("path"))),
                repo_root=repo_root,
                label=f"external evidence {row.get('role')}",
            )
            if any(
                binding[key] != row.get(key) for key in ("mode", "path", "sha256", "size_bytes")
            ):
                raise Phase0EvidenceError("external evidence linked readback mismatch")
    return dict(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--classification-manifest", required=True)
    parser.add_argument("--skip-baseline", required=True)
    parser.add_argument("--session-manifest", required=True)
    parser.add_argument("--expected-session-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = seal_phase0_session(
            repo_root=Path(args.repo_root),
            bundle_root=Path(args.bundle_root),
            classification_manifest=Path(args.classification_manifest),
            skip_baseline=Path(args.skip_baseline),
            session_manifest=Path(args.session_manifest),
            expected_session_sha256=args.expected_session_sha256,
        )
    except (Phase0EvidenceError, OSError, ValueError) as exc:
        print(f"v17 Phase 0 evidence index failed: {exc}", file=sys.stderr)
        return 2
    index_path = Path(args.bundle_root).absolute() / "70_evidence_index.json"
    raw, _observed = _stable_regular_file(index_path, require_private=True)
    print(
        json.dumps(
            {
                "byte_sha256": _sha256(raw),
                "semantic_sha256": report["semantic_sha256"],
                "status": "SEALED",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
