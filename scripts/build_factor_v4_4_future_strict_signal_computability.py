#!/usr/bin/env python3
"""Publish future-only strict-full-A exact-five computability evidence.

The publisher is deliberately manifest-first after its fixed-interpreter,
private-shadow, and native-loader bootstrap.  Until an owner-private canonical
input manifest has passed two fresh anchored reads, it does not import project
code or touch a governed publication/data root.  Historical readback is
sealed-bundle-graph-only.

An uncatchable parent SIGKILL or power loss can leave one exact-prefix runtime
shadow behind.  A later invocation removes only dead-PID, owner-private,
non-symlink shadows in that namespace; it never treats this as crash cleanup.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping, Sequence
import contextlib
import copy
from dataclasses import dataclass
from datetime import date, datetime
import errno
import gc
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import time
import types
from typing import Any, BinaryIO, Callable


PROJECT_ROOT = Path("/Users/maxwell/mySpace/myQuant")
PRODUCTION_PRIVATE_ROOT = PROJECT_ROOT.joinpath(
    "reports",
    "factor_governance",
    "private",
    "v4_4_signal_computability_strict",
)
ROOT_SUFFIX = (
    "reports",
    "factor_governance",
    "private",
    "v4_4_signal_computability_strict",
)
STRICT_INPUT_FILENAMES = (
    "strict_computability_input_manifest.v4_4.json",
    "strict_computability_input_receipt.v4_4.json",
    "strict_data_field_receipt.v4_4.json",
    "strict_two_pass_equivalence_receipt.v4_4.json",
    "strict_exact_five_signal_computability_proof.v4_4.json",
)
STRICT_READBACK_FILENAME = "strict_signal_computability_readback.v4_4.json"
PREREGISTRATION_ROOT_SUFFIX = (
    "reports",
    "factor_governance",
    "private",
    "v4_4_candidate_preregistration",
)
PREREGISTRATION_INPUT_FILENAMES = (
    "v4_2_predecessor.aquant_idea_source_receipt.v4_2.json",
    "v4_2_predecessor.myquant_alpha158_source_receipt.v4_2.json",
    "v4_2_predecessor.operator_semantics.v4_2.json",
    "v4_2_predecessor.comparison_catalog_receipt.v4_2.json",
    "v4_2_predecessor.candidate_selection_spec.v4_2.json",
    "v4_2_predecessor.strict_full_a_source_binding.v4_2.json",
    "v4_2_predecessor.code_binding_set.v4_2.json",
    "v4_2_predecessor.future_source_envelope.v4_2.json",
    "v4_2_predecessor.cycle_root.v4_2.json",
    "v4_2_predecessor.definition_identity_collision_audit.v4_2.json",
    "v4_2_predecessor.cycle_state.precommitted.v4_1.json",
    "v4_2_predecessor.discovery_source_node.v4_2.json",
    "v4_2_predecessor.cycle_state.discovery.v4_1.json",
    "v4_2_predecessor.prereg_discovery_orchestration.v4_2.json",
    "code_binding_set.v4_4.json",
    "prior_diagnostic_runtime_binding.v4_3.json",
    "prior_diagnostic_nomination.v4_3.json",
    "prior_diagnostic_nomination_readback.v4_3.json",
    "expanded_candidate_selection.v4_4.json",
    "definition_identity_collision_audit.v4_4.json",
    "cycle_root.v4_4.json",
    "future_source_envelope.v4_4.json",
    "cycle_state.precommitted.v4_1.json",
    "discovery_source_node.v4_4.json",
    "cycle_state.discovery.v4_1.json",
    "prereg_discovery_orchestration.v4_4.json",
)
PREREGISTRATION_READBACK_FILENAME = (
    "candidate_preregistration_readback.v4_4.json"
)
PREREGISTRATION_CODE_BINDING_PATHS = (
    "scripts/build_factor_v4_4_candidate_preregistration.py",
    "quant_investor/factors/governance_candidate_preregistration_v4_4.py",
    "quant_investor/factors/governance_candidate_preregistration_bundle_v4_4.py",
    "quant_investor/factors/governance_candidate_preregistration_v4_2.py",
    "quant_investor/factors/governance_candidate_preregistration_bundle_v4_2.py",
    "scripts/build_factor_v4_2_candidate_preregistration.py",
    "quant_investor/factors/governance_prior_diagnostic_nomination_v4_3.py",
    "quant_investor/factors/governance_prior_diagnostic_nomination_bundle_v4_3.py",
    "quant_investor/factors/governance_cycle_state_v4_1.py",
    "quant_investor/factors/governance_private_bundle_io.py",
    "quant_investor/factors/governance_source_readback_v4_1.py",
    "quant_investor/factors/governance_screening_v4.py",
    "quant_investor/codex_review/storage.py",
    "quant_investor/market/pit_universe.py",
    "quant_investor/factors/governance_source_v4_1.py",
)
PREDECESSOR_CODE_BINDING_PATHS = (
    "scripts/build_factor_v4_2_candidate_preregistration.py",
    "quant_investor/factors/governance_candidate_preregistration_v4_2.py",
    "quant_investor/factors/governance_candidate_preregistration_bundle_v4_2.py",
    "quant_investor/factors/governance_cycle_state_v4_1.py",
    "quant_investor/factors/governance_private_bundle_io.py",
    "quant_investor/factors/governance_source_readback_v4_1.py",
    "quant_investor/factors/governance_screening_v4.py",
    "quant_investor/codex_review/storage.py",
    "quant_investor/market/pit_universe.py",
    "quant_investor/factors/governance_source_v4_1.py",
)
RUNTIME_DISTRIBUTION_TOP_LEVEL = (
    ("numpy", "numpy", "numpy/__init__.py"),
    ("pandas", "pandas", "pandas/__init__.py"),
    ("pyarrow", "pyarrow", "pyarrow/__init__.py"),
    ("python-dateutil", "dateutil", "dateutil/__init__.py"),
    ("pytz", "pytz", "pytz/__init__.py"),
    ("six", "six", "six.py"),
)
RUNTIME_SHADOW_DIRECTORY_ROOTS = (
    "numpy",
    "numpy-2.4.3.dist-info",
    "pandas",
    "pandas-3.0.1.dist-info",
    "pyarrow",
    "pyarrow-24.0.0.dist-info",
    "dateutil",
    "python_dateutil-2.9.0.post0.dist-info",
    "pytz",
    "pytz-2026.1.post1.dist-info",
    "six-1.17.0.dist-info",
)
RUNTIME_SHADOW_FILE_ROOTS = ("six.py",)
RUNTIME_DISTRIBUTION_SHADOW_ROOTS = {
    "numpy": ("numpy", "numpy-2.4.3.dist-info"),
    "pandas": ("pandas", "pandas-3.0.1.dist-info"),
    "pyarrow": ("pyarrow", "pyarrow-24.0.0.dist-info"),
    "python-dateutil": ("dateutil", "python_dateutil-2.9.0.post0.dist-info"),
    "pytz": ("pytz", "pytz-2026.1.post1.dist-info"),
    "six": ("six.py", "six-1.17.0.dist-info"),
}
RUNTIME_SHADOW_MAX_FILES = 5_000
RUNTIME_SHADOW_MAX_DIRECTORIES = 1_000
RUNTIME_SHADOW_MAX_TOTAL_BYTES = 256 * 1024 * 1024
RUNTIME_SHADOW_MAX_FILE_BYTES = 64 * 1024 * 1024
RUNTIME_SHADOW_MIN_FREE_BYTES = 512 * 1024 * 1024
RUNTIME_SHADOW_WALL_SECONDS_MAX = 120
RUNTIME_SHADOW_PARENT = Path("/private/tmp")
RUNTIME_SHADOW_PREFIX = "myquant-v44-runtime-shadow-"
RUNTIME_SHADOW_SITE_DIRECTORY = "site-packages"
FIXED_INTERPRETER = PROJECT_ROOT / ".venv" / "bin" / "python"
FIXED_INTERPRETER_RESOLVED = Path(
    "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/"
    "Python.framework/Versions/3.13/bin/python3.13"
)
FIXED_INTERPRETER_SIZE = 52_640
FIXED_INTERPRETER_SHA256 = (
    "a708f6e9f4803b806b29146c4e0feecfd9bf2d9eb60f3e15b850cd7cb56f200b"
)
FIXED_SITE_PACKAGES = (
    PROJECT_ROOT / ".venv" / "lib" / "python3.13" / "site-packages"
)
FIXED_OTOOL = Path("/usr/bin/otool")
ISOLATED_CHILD_ENV = "MYQUANT_V44_ISOLATED_CHILD"
ISOLATED_CHILD_SHADOW_FD_ENV = "MYQUANT_V44_SHADOW_FD"
ISOLATED_CHILD_SHADOW_PATH_ENV = "MYQUANT_V44_SHADOW_PATH"
ISOLATED_CHILD_NONCE_ENV = "MYQUANT_V44_SHADOW_NONCE"
ISOLATED_CHILD_TEST_PROBE_ENV = "MYQUANT_V44_TEST_PARQUET_PROBE"
_ISOLATED_CHILD_ENV_KEYS = frozenset(
    {
        "PATH",
        "HOME",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "TZ",
        "__CF_USER_TEXT_ENCODING",
        ISOLATED_CHILD_ENV,
        ISOLATED_CHILD_SHADOW_FD_ENV,
        ISOLATED_CHILD_SHADOW_PATH_ENV,
        ISOLATED_CHILD_NONCE_ENV,
        ISOLATED_CHILD_TEST_PROBE_ENV,
    }
)
MANIFEST_SCHEMA_VERSION = (
    "factor-governance-future-strict-computability-input.v4.4"
)
PROTOCOL_VERSION = "v4"
EVIDENCE_CONTRACT_VERSION = "v4.4"
FROZEN_PREVIOUS_CUTOFF = "2026-07-19"
MAX_MANIFEST_BYTES = 65_536

FIXED_CODE_BINDING_PATHS = (
    "quant_investor/factors/governance_future_strict_exact_five_eval_v4_4.py",
    "quant_investor/factors/governance_future_strict_signal_computability_v4_4.py",
    "scripts/build_factor_v4_4_future_strict_signal_computability.py",
    "quant_investor/factors/governance_private_bundle_io.py",
    "quant_investor/factors/governance_candidate_preregistration_v4_4.py",
    "quant_investor/factors/governance_candidate_preregistration_bundle_v4_4.py",
    "quant_investor/factors/governance_candidate_preregistration_v4_2.py",
    "quant_investor/factors/governance_candidate_preregistration_bundle_v4_2.py",
    "quant_investor/factors/governance_prior_diagnostic_nomination_v4_3.py",
    "quant_investor/factors/governance_prior_diagnostic_nomination_bundle_v4_3.py",
    "quant_investor/factors/governance_source_readback_v4_1.py",
    "quant_investor/alpha158.py",
)
PROTECTED_CONTROL_RELATIVE_PATHS = (
    ("registry", "quant_investor/factor_registry/mined_factors.json"),
    ("latest_pointer", "data/parquet/cn/_latest.json"),
    ("catalog", "data/parquet/cn/_catalog.json"),
    ("fundamental_pointer", "data/parquet/cn/_fundamental_latest.json"),
    ("latest_manifest", "data/parquet/cn/latest_manifest.json"),
)

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "evidence_contract_version",
        "cycle_id",
        "cutoff",
        "snapshot_id",
        "proof_output_start",
        "preregistration",
        "strict_source_expected",
        "source_definition_bindings",
        "code_binding_set",
        "runtime_binding_expected_semantic_sha256",
        "protected_control_expected_sha256",
        "resource_contract",
        "selection_disclosures",
        "negative_claims",
    }
)
_PREREGISTRATION_FIELDS = frozenset(
    {
        "bundle_path",
        "readback_byte_sha256",
        "readback_semantic_sha256",
        "artifact_count",
        "cycle_id",
        "candidate_rows_semantic_sha256",
    }
)
_STRICT_SOURCE_EXPECTED_FIELDS = frozenset(
    {
        "strict_source_binding_semantic_sha256",
        "snapshot_manifest_byte_sha256",
        "pit_generation_manifest_byte_sha256",
        "pit_membership_byte_sha256",
        "table_inventory_semantic_sha256",
        "full_a_scope_count",
        "full_a_scope_sha256",
        "source_calendar_semantic_sha256",
        "recorded_latest_pointer_byte_sha256",
        "recorded_components_byte_sha256",
    }
)
_SOURCE_DEFINITION_FIELDS = frozenset(
    {
        "order",
        "name",
        "definition_identity_sha256",
        "direction",
        "source_repository",
        "source_commit",
        "source_tree_oid",
        "source_relative_path",
        "source_blob_oid",
        "source_raw_sha256",
        "source_ast_sha256",
        "field_semantics_sha256",
        "operator_program_sha256",
        "operator_program_set_sha256",
    }
)
_CODE_BINDING_FIELDS = frozenset({"relative_path", "byte_sha256"})
_PROTECTED_CONTROL_FIELDS = frozenset(
    name for name, _relative in PROTECTED_CONTROL_RELATIVE_PATHS
)
_RESOURCE_FIELDS = frozenset(
    {
        "manifest_max_bytes",
        "prereg_artifact_max_bytes",
        "prereg_bundle_max_bytes",
        "strict_artifact_max_bytes",
        "strict_bundle_max_bytes",
        "table_member_count_max",
        "table_member_max_bytes",
        "table_total_max_bytes",
        "pit_max_bytes",
        "pit_row_count_max",
        "source_session_count_max",
        "historical_symbol_count_max",
        "projected_row_count_per_pass_max",
        "dense_cell_count_per_block_max",
        "rss_max_bytes",
        "pass_wall_seconds_max",
        "total_wall_seconds_max",
        "halo_session_count",
        "output_block_session_count",
    }
)
_SELECTION_DISCLOSURE_FIELDS = frozenset(
    {
        "outcome_informed_selection",
        "external_label_independence",
        "prior_statistics_inherited_as_formal_evidence",
    }
)
_NEGATIVE_CLAIM_SECTIONS = frozenset(
    {"measurement", "authority", "side_effects"}
)
RESOURCE_CONTRACT = {
    "manifest_max_bytes": 65_536,
    "prereg_artifact_max_bytes": 67_108_864,
    "prereg_bundle_max_bytes": 268_435_456,
    "strict_artifact_max_bytes": 16_777_216,
    "strict_bundle_max_bytes": 67_108_864,
    "table_member_count_max": 256,
    "table_member_max_bytes": 67_108_864,
    "table_total_max_bytes": 1_073_741_824,
    "pit_max_bytes": 134_217_728,
    "pit_row_count_max": 16_384,
    "source_session_count_max": 4_096,
    "historical_symbol_count_max": 8_192,
    "projected_row_count_per_pass_max": 16_777_216,
    "dense_cell_count_per_block_max": 1_540_096,
    "rss_max_bytes": 3_221_225_472,
    "pass_wall_seconds_max": 1_800,
    "total_wall_seconds_max": 3_900,
    "halo_session_count": 60,
    "output_block_session_count": 128,
}
SELECTION_DISCLOSURES = {
    "outcome_informed_selection": True,
    "external_label_independence": False,
    "prior_statistics_inherited_as_formal_evidence": False,
}
NEGATIVE_CLAIMS = {
    "measurement": {
        "statistics": "not_run",
        "ic": "not_run",
        "fdr": "not_run",
        "family_bh": "not_run",
        "maturity": "not_run",
        "walk_forward": "not_run",
        "cost": "not_run",
        "neutralization": "not_run",
        "stability": "not_run",
        "structural_dedup": "not_run",
        "formal_dedup": "not_run",
        "high_correlation_dedup": "not_run",
        "qualification": "not_run",
        "admission": "not_run",
        "verified_v4_replay": "not_run",
        "transaction_plan": "not_run",
    },
    "authority": {
        "healthy_source_receipt": False,
        "healthy_factor_authorized": False,
        "measurement_authorized": False,
        "screening_authorized": False,
        "statistics_authorized": False,
        "ic_authorized": False,
        "fdr_authorized": False,
        "family_bh_authorized": False,
        "maturity_authorized": False,
        "walk_forward_authorized": False,
        "cost_authorized": False,
        "neutralization_authorized": False,
        "stability_authorized": False,
        "dedup_authorized": False,
        "qualification_authorized": False,
        "admission_authorized": False,
        "replay_authorized": False,
        "transaction_authorized": False,
        "proposal_authorized": False,
        "registry_write_authorized": False,
        "apply_authorized": False,
        "activation_authorized": False,
        "production_candidate_authorized": False,
        "production_new_risk_authorized": False,
        "new_risk_authorized": False,
    },
    "side_effects": {
        "registry": False,
        "wal": False,
        "budget": False,
        "production_receipt": False,
        "production_pointer": False,
        "proposal": False,
        "apply": False,
        "activation": False,
        "provider": False,
        "network": False,
        "portfolio": False,
        "broker": False,
        "order": False,
        "trade": False,
        "transaction": False,
    },
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_OID_RE = re.compile(r"[0-9a-f]{40}")
_SNAPSHOT_RE = re.compile(r"\d{8}T\d{6}Z")
_CN_SYMBOL_RE = re.compile(r"\d{6}\.(?:SZ|SH|BJ)")
_RUNTIME_SHADOW_NAME_RE = re.compile(
    rf"{re.escape(RUNTIME_SHADOW_PREFIX)}([1-9][0-9]*)-([0-9a-f]{{32}})"
)
_SANDBOX_ARROW_SYSCTL_WARNINGS = tuple(
    (
        "/Users/runner/work/crossbow/crossbow/arrow/cpp/src/arrow/util/"
        "cpu_info.cc:242: IOError: sysctlbyname failed for "
        f"'{name}'. Detail: [errno 1] Operation not permitted"
    )
    for name in (
        "hw.l1dcachesize",
        "hw.l2cachesize",
        "hw.l3cachesize",
        "hw.optional.neon",
    )
)

_ISOLATED_CHILD_ACTIVE = False
_ACTIVE_RUNTIME_SHADOW_ROOT: Path | None = None
_ACTIVE_RUNTIME_SHADOW_SITE: Path | None = None
_ACTIVE_RUNTIME_SHADOW_INVENTORY: dict[str, Any] | None = None


class FactorV4_4FutureStrictRunnerError(ValueError):
    """A future-strict publisher/readback invariant failed closed."""


@dataclass(frozen=True)
class StableBytes:
    """Bytes collected from one stable anchored regular-file identity."""

    path: Path
    raw: bytes
    byte_sha256: str
    signature: tuple[int, ...]


@dataclass(frozen=True)
class RuntimeTreeFile:
    """One owner-regular runtime file observed through anchored descriptors."""

    relative_path: str
    size_bytes: int
    byte_sha256: str
    signature: tuple[int, ...]


@dataclass(frozen=True)
class RuntimeTreeScan:
    """Exact bounded source/shadow inventory, excluding bytecode caches."""

    directories: tuple[str, ...]
    directory_signatures: tuple[tuple[str, tuple[int, ...]], ...]
    files: tuple[RuntimeTreeFile, ...]
    descriptor: dict[str, Any]


@dataclass(frozen=True)
class LoadedModules:
    """Project modules imported only after stage-0 manifest acceptance."""

    contract: Any
    prereg_core: Any
    prereg_bundle: Any
    predecessor_bundle: Any
    private_io: Any
    prebound_runtime: dict[str, Any] | None = None
    import_guard: Any = None
    preregistration_snapshot: Any = None


@dataclass(frozen=True)
class RawPrivateBundleSnapshot:
    """Strict stdlib-only intake of a manifest-pinned private bundle."""

    path: Path
    values: dict[str, dict[str, Any]]
    files: dict[str, StableBytes]


@dataclass(frozen=True)
class VerifiedImportEnvironment:
    """Closed verified project-source loader kept alive for later imports."""

    finder: Any
    code_files: tuple[StableBytes, ...]
    runtime: dict[str, Any]


@dataclass(frozen=True)
class AcceptedPreregistration:
    """The fully validated 26+1 future preregistration graph."""

    readback: dict[str, Any]
    artifacts: dict[str, dict[str, Any]]
    strict_source: dict[str, Any]
    backend_binding: dict[str, Any]
    calendar_sessions: tuple[str, ...]
    candidate_rows: tuple[dict[str, Any], ...]
    snapshot_manifest_path: Path
    pit_manifest_path: Path
    pit_membership_path: Path
    table_root: Path


@dataclass(frozen=True)
class PublicationPreflight:
    """An anchored observation of the fixed private publication root."""

    root: Path
    root_signature: tuple[int, ...]
    cycle_id: str


@dataclass(frozen=True)
class FrozenBindings:
    """Initial fixed code, runtime, and protected-control observations."""

    code: tuple[dict[str, Any], ...]
    code_files: tuple[StableBytes, ...]
    protected: tuple[tuple[str, StableBytes], ...]
    runtime: dict[str, Any]


@dataclass(frozen=True)
class DataStack:
    """Heavy data modules loaded only after every pre-data gate passes."""

    np: Any
    pd: Any
    pa: Any
    pq: Any
    pc: Any
    evaluator: Any


@dataclass(frozen=True)
class PITProjection:
    """One-row-per-symbol inclusive/exclusive PIT interval projection."""

    records: tuple[dict[str, str], ...]
    historical_symbols: tuple[str, ...]
    cutoff_symbols: tuple[str, ...]
    eligibility_mask: Any
    row_count: int
    membership_byte_sha256: str


@dataclass(frozen=True)
class TableCollection:
    """Hash-bound projected Arrow tables for one fresh pass."""

    table: Any
    inventory_rows: tuple[dict[str, Any], ...]
    projected_row_count: int
    outside_pit_bar_count: int
    ignored_pre_analysis_row_count: int
    collection_sha256: str


@dataclass(frozen=True)
class PassEvidence:
    """JSON-only evidence retained after one fresh pass is discarded."""

    pass_id: str
    accepted_preregistration_semantic_sha256: str
    source_observation_bindings: dict[str, Any]
    source_identity_signatures: dict[str, Any]
    block_manifest: dict[str, Any]
    pit_mask_descriptor: dict[str, Any]
    proof_pit_mask_descriptor: dict[str, Any]
    candidate_non_null_mask_descriptors: tuple[dict[str, Any], ...]
    historical_symbol_axis: dict[str, Any]
    pit_membership_contract: dict[str, Any]
    input_matrix_descriptors: tuple[dict[str, Any], ...]
    engine_matrix_descriptors: tuple[dict[str, Any], ...]
    field_missing_counts: dict[str, int]
    outside_pit_non_null_counts: dict[str, int]
    bars_outside_pit_interval_count: int
    ignored_pre_analysis_row_count: int
    dense_projected_row_count: int
    table_content_binding_sha256: str
    elapsed_seconds: float
    peak_rss_bytes: int


def _error(message: str) -> FactorV4_4FutureStrictRunnerError:
    return FactorV4_4FutureStrictRunnerError(message)


def _sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise _error(f"{label} must be lowercase SHA-256")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise _error(f"{label} must be a positive integer")
    return value


def _canonical_date(value: Any, label: str) -> str:
    if type(value) is not str:
        raise _error(f"{label} must be a canonical ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise _error(f"{label} must be a canonical ISO date") from exc
    if parsed.isoformat() != value:
        raise _error(f"{label} must be a canonical ISO date")
    return value


def _absolute_normalized_path(value: Any, label: str) -> Path:
    if (
        type(value) is not str
        or not value.startswith("/")
        or value == "/"
        or "\x00" in value
        or os.path.abspath(value) != value
    ):
        raise _error(f"{label} must be an absolute normalized path")
    components = value.split("/")[1:]
    if not components or any(part in {"", ".", ".."} for part in components):
        raise _error(f"{label} must be an absolute normalized path")
    return Path(value)


def _exact_mapping(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise _error(f"{label} must be an object with string fields")
    payload = copy.deepcopy(dict(value))
    if set(payload) != set(fields):
        missing = sorted(set(fields) - set(payload))
        extra = sorted(set(payload) - set(fields))
        raise _error(
            f"{label} fields are not exact: "
            f"missing={','.join(missing) or '-'};extra={','.join(extra) or '-'}"
        )
    return payload


def _reject_json_constant(value: str) -> None:
    raise _error(f"input manifest contains non-finite JSON constant: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _error(f"input manifest contains duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise _error(f"{label} must be strict UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except FactorV4_4FutureStrictRunnerError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise _error(f"{label} must be strict JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise _error(f"{label} must contain one JSON object")
    try:
        canonical = (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise _error(f"{label} is not finite canonical JSON") from exc
    if raw != canonical:
        raise _error(f"{label} bytes must be compact sorted canonical JSON plus newline")
    return copy.deepcopy(dict(value))


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _file_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


def _signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_uid),
        int(metadata.st_gid),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _same_object(left: os.stat_result, right: os.stat_result) -> bool:
    return (int(left.st_dev), int(left.st_ino)) == (
        int(right.st_dev),
        int(right.st_ino),
    )


def _runtime_tree_descriptor(
    directories: Sequence[str], files: Sequence[RuntimeTreeFile]
) -> dict[str, Any]:
    rows = [
        {
            "relative_path": item.relative_path,
            "size_bytes": item.size_bytes,
            "byte_sha256": item.byte_sha256,
        }
        for item in files
    ]
    body = {
        "schema_version": "factor-governance-isolated-runtime-tree.v4.4",
        "directories": list(directories),
        "files": rows,
        "directory_count": len(directories),
        "file_count": len(rows),
        "total_bytes": sum(row["size_bytes"] for row in rows),
    }
    body["tree_semantic_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return body


def _open_relative_directory(root_fd: int, parts: Sequence[str]) -> int:
    descriptor = os.dup(root_fd)
    os.set_inheritable(descriptor, False)
    try:
        for part in parts:
            if type(part) is not str or part in {"", ".", ".."} or "/" in part:
                raise _error("runtime tree relative directory is unsafe")
            child = os.open(part, _directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _read_runtime_file_at(
    parent_fd: int,
    leaf: str,
    *,
    label: str,
    required_mode: int | None,
) -> RuntimeTreeFile:
    descriptor = -1
    try:
        path_metadata = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(path_metadata.st_mode):
            raise _error(f"{label} must not be a symlink")
        descriptor = os.open(leaf, _file_flags(), dir_fd=parent_fd)
        before = os.fstat(descriptor)
        if not _same_object(path_metadata, before) or not stat.S_ISREG(before.st_mode):
            raise _error(f"{label} must be one anchored regular file")
        if int(before.st_uid) != os.getuid() or int(before.st_nlink) != 1:
            raise _error(f"{label} must be owner-regular with one link")
        if required_mode is not None and stat.S_IMODE(before.st_mode) != required_mode:
            raise _error(f"{label} mode must be {required_mode:04o}")
        if int(before.st_size) > RUNTIME_SHADOW_MAX_FILE_BYTES:
            raise _error(f"{label} exceeds the per-file byte cap")
        digest = hashlib.sha256()
        observed = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, RUNTIME_SHADOW_MAX_FILE_BYTES + 1))
            if not chunk:
                break
            observed += len(chunk)
            if observed > RUNTIME_SHADOW_MAX_FILE_BYTES:
                raise _error(f"{label} exceeds the per-file byte cap")
            digest.update(chunk)
        after = os.fstat(descriptor)
        current = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if (
            _signature(before) != _signature(after)
            or _signature(before) != _signature(current)
            or observed != int(before.st_size)
        ):
            raise _error(f"{label} drifted while hashing")
        return RuntimeTreeFile(
            relative_path="",
            size_bytes=observed,
            byte_sha256=digest.hexdigest(),
            signature=_signature(before),
        )
    except OSError as exc:
        raise _error(f"{label} is unavailable: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _runtime_tree_root_entries(root_fd: int) -> tuple[os.DirEntry[str], ...]:
    try:
        with os.scandir(root_fd) as iterator:
            entries = tuple(sorted(iterator, key=lambda item: item.name))
    except OSError as exc:
        raise _error(f"runtime tree root cannot be scanned: {exc}") from exc
    seen: set[str] = set()
    for entry in entries:
        folded = entry.name.casefold()
        if folded in seen:
            raise _error("runtime tree has a case-colliding root entry")
        seen.add(folded)
    return entries


def _scan_runtime_tree_fd(
    root_fd: int,
    *,
    exact_root_entries: bool,
    require_sealed: bool,
) -> RuntimeTreeScan:
    """Hash the exact frozen roots without following any link or RECORD path."""

    started = time.monotonic()
    expected_roots = set(RUNTIME_SHADOW_DIRECTORY_ROOTS) | set(
        RUNTIME_SHADOW_FILE_ROOTS
    )
    root_entries = _runtime_tree_root_entries(root_fd)
    observed_names = {entry.name for entry in root_entries}
    if exact_root_entries and observed_names != expected_roots:
        raise _error("runtime shadow has missing or extra top-level roots")
    folded_expected = {name.casefold(): name for name in expected_roots}
    for entry in root_entries:
        canonical = folded_expected.get(entry.name.casefold())
        if canonical is not None and canonical != entry.name:
            raise _error("runtime source root has a case/path collision")
        if entry.name.endswith(".libs") and entry.name.removesuffix(".libs") in {
            "numpy",
            "pandas",
            "pyarrow",
            "dateutil",
            "pytz",
            "six",
        }:
            raise _error("runtime source has an unbound top-level .libs root")
    if not expected_roots.issubset(observed_names):
        raise _error("runtime source is missing a frozen package root")

    directories: list[str] = []
    directory_signatures: list[tuple[str, tuple[int, ...]]] = []
    files: list[RuntimeTreeFile] = []
    case_paths: set[str] = set()

    def add_case_path(relative: str) -> None:
        folded = relative.casefold()
        if folded in case_paths:
            raise _error("runtime tree has a case/path collision")
        case_paths.add(folded)

    def walk(directory_fd: int, relative: str) -> None:
        if time.monotonic() - started > RUNTIME_SHADOW_WALL_SECONDS_MAX:
            raise _error("runtime tree scan exceeded the wall-time cap")
        metadata = os.fstat(directory_fd)
        if not stat.S_ISDIR(metadata.st_mode) or int(metadata.st_uid) != os.getuid():
            raise _error("runtime directory must be an owner real directory")
        if require_sealed and stat.S_IMODE(metadata.st_mode) != 0o500:
            raise _error("runtime shadow directory mode must be 0500")
        add_case_path(relative)
        directories.append(relative)
        directory_signatures.append((relative, _signature(metadata)))
        if len(directories) > RUNTIME_SHADOW_MAX_DIRECTORIES:
            raise _error("runtime tree exceeds the directory-count cap")
        try:
            with os.scandir(directory_fd) as iterator:
                entries = tuple(sorted(iterator, key=lambda item: item.name))
        except OSError as exc:
            raise _error(f"runtime directory scan failed: {relative}: {exc}") from exc
        sibling_names: set[str] = set()
        for entry in entries:
            name = entry.name
            folded = name.casefold()
            if folded in sibling_names or name in {"", ".", ".."} or "/" in name:
                raise _error("runtime tree contains an unsafe/colliding name")
            sibling_names.add(folded)
            child_relative = f"{relative}/{name}"
            try:
                child_metadata = os.stat(
                    name, dir_fd=directory_fd, follow_symlinks=False
                )
            except OSError as exc:
                raise _error(f"runtime member unavailable: {child_relative}") from exc
            if stat.S_ISLNK(child_metadata.st_mode):
                raise _error(f"runtime member is a symlink: {child_relative}")
            if stat.S_ISDIR(child_metadata.st_mode):
                if name == "__pycache__":
                    if int(child_metadata.st_uid) != os.getuid():
                        raise _error("excluded runtime bytecode cache owner mismatch")
                    continue
                child_fd = os.open(name, _directory_flags(), dir_fd=directory_fd)
                try:
                    opened = os.fstat(child_fd)
                    if _signature(opened) != _signature(child_metadata):
                        raise _error(f"runtime directory swapped: {child_relative}")
                    walk(child_fd, child_relative)
                finally:
                    os.close(child_fd)
                continue
            if name.endswith((".pyc", ".pyo")):
                if not stat.S_ISREG(child_metadata.st_mode):
                    raise _error("excluded runtime bytecode is not regular")
                continue
            if not stat.S_ISREG(child_metadata.st_mode):
                raise _error(f"runtime member is special: {child_relative}")
            add_case_path(child_relative)
            item = _read_runtime_file_at(
                directory_fd,
                name,
                label=f"runtime member {child_relative}",
                required_mode=0o400 if require_sealed else None,
            )
            files.append(
                RuntimeTreeFile(
                    relative_path=child_relative,
                    size_bytes=item.size_bytes,
                    byte_sha256=item.byte_sha256,
                    signature=item.signature,
                )
            )
            if len(files) > RUNTIME_SHADOW_MAX_FILES:
                raise _error("runtime tree exceeds the file-count cap")
            if sum(row.size_bytes for row in files) > RUNTIME_SHADOW_MAX_TOTAL_BYTES:
                raise _error("runtime tree exceeds the total-byte cap")

    for name in RUNTIME_SHADOW_DIRECTORY_ROOTS:
        try:
            metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise _error(f"runtime root must be a real directory: {name}")
            child_fd = os.open(name, _directory_flags(), dir_fd=root_fd)
        except OSError as exc:
            raise _error(f"runtime root is unavailable: {name}") from exc
        try:
            if _signature(os.fstat(child_fd)) != _signature(metadata):
                raise _error(f"runtime root swapped while opening: {name}")
            walk(child_fd, name)
        finally:
            os.close(child_fd)
    for name in RUNTIME_SHADOW_FILE_ROOTS:
        add_case_path(name)
        item = _read_runtime_file_at(
            root_fd,
            name,
            label=f"runtime root file {name}",
            required_mode=0o400 if require_sealed else None,
        )
        files.append(
            RuntimeTreeFile(
                relative_path=name,
                size_bytes=item.size_bytes,
                byte_sha256=item.byte_sha256,
                signature=item.signature,
            )
        )
    ordered_directories = tuple(sorted(directories))
    ordered_signatures = tuple(sorted(directory_signatures))
    ordered_files = tuple(sorted(files, key=lambda item: item.relative_path))
    descriptor = _runtime_tree_descriptor(ordered_directories, ordered_files)
    if (
        descriptor["file_count"] > RUNTIME_SHADOW_MAX_FILES
        or descriptor["directory_count"] > RUNTIME_SHADOW_MAX_DIRECTORIES
        or descriptor["total_bytes"] > RUNTIME_SHADOW_MAX_TOTAL_BYTES
    ):
        raise _error("runtime tree exceeds a fixed resource cap")
    return RuntimeTreeScan(
        directories=ordered_directories,
        directory_signatures=ordered_signatures,
        files=ordered_files,
        descriptor=descriptor,
    )


def _open_runtime_root(path: Path, label: str) -> int:
    parent_fd, leaf = _open_parent_anchored(path)
    descriptor = -1
    try:
        path_metadata = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(leaf, _directory_flags(), dir_fd=parent_fd)
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(path_metadata.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or _signature(path_metadata) != _signature(opened)
            or int(opened.st_uid) != os.getuid()
        ):
            raise _error(f"{label} must be an owner real directory")
        return descriptor
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    finally:
        os.close(parent_fd)


def _open_shadow_parent() -> int:
    parent_fd, leaf = _open_parent_anchored(RUNTIME_SHADOW_PARENT)
    descriptor = -1
    try:
        path_metadata = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(leaf, _directory_flags(), dir_fd=parent_fd)
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(path_metadata.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or _signature(path_metadata) != _signature(opened)
            or int(opened.st_uid) != 0
            or stat.S_IMODE(opened.st_mode) != 0o1777
        ):
            raise _error("runtime shadow parent must be the root-owned sticky 1777 root")
        return descriptor
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    finally:
        os.close(parent_fd)


def _copy_runtime_file(
    *,
    source_root_fd: int,
    destination_root_fd: int,
    expected: RuntimeTreeFile,
) -> None:
    parts = expected.relative_path.split("/")
    source_parent = _open_relative_directory(source_root_fd, parts[:-1])
    destination_parent = _open_relative_directory(destination_root_fd, parts[:-1])
    source_fd = -1
    destination_fd = -1
    try:
        source_fd = os.open(parts[-1], _file_flags(), dir_fd=source_parent)
        before = os.fstat(source_fd)
        if _signature(before) != expected.signature:
            raise _error(f"runtime source swapped before copy: {expected.relative_path}")
        destination_fd = os.open(
            parts[-1],
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=destination_parent,
        )
        digest = hashlib.sha256()
        observed = 0
        while True:
            chunk = os.read(source_fd, min(1024 * 1024, expected.size_bytes + 1))
            if not chunk:
                break
            observed += len(chunk)
            if observed > expected.size_bytes:
                raise _error(f"runtime source grew during copy: {expected.relative_path}")
            digest.update(chunk)
            pending = memoryview(chunk)
            while pending:
                written = os.write(destination_fd, pending)
                if written <= 0:
                    raise _error("runtime shadow copy made no progress")
                pending = pending[written:]
        after = os.fstat(source_fd)
        destination = os.fstat(destination_fd)
        if (
            _signature(before) != _signature(after)
            or observed != expected.size_bytes
            or digest.hexdigest() != expected.byte_sha256
            or not stat.S_ISREG(destination.st_mode)
            or int(destination.st_uid) != os.getuid()
            or int(destination.st_nlink) != 1
            or int(destination.st_size) != expected.size_bytes
            or stat.S_IMODE(destination.st_mode) != 0o600
        ):
            raise _error(f"runtime source/copy drifted: {expected.relative_path}")
    except OSError as exc:
        raise _error(
            f"runtime shadow copy failed: {expected.relative_path}: {exc}"
        ) from exc
    finally:
        if destination_fd >= 0:
            os.close(destination_fd)
        if source_fd >= 0:
            os.close(source_fd)
        os.close(destination_parent)
        os.close(source_parent)


def _seal_runtime_shadow(site_fd: int, scan: RuntimeTreeScan) -> None:
    for item in scan.files:
        parts = item.relative_path.split("/")
        parent_fd = _open_relative_directory(site_fd, parts[:-1])
        descriptor = -1
        try:
            descriptor = os.open(parts[-1], _file_flags(), dir_fd=parent_fd)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or int(metadata.st_uid) != os.getuid()
                or int(metadata.st_nlink) != 1
            ):
                raise _error("runtime shadow file changed before sealing")
            os.fchmod(descriptor, 0o400)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent_fd)
    for relative in sorted(
        scan.directories, key=lambda value: (value.count("/"), value), reverse=True
    ):
        parts = relative.split("/")
        parent_fd = _open_relative_directory(site_fd, parts[:-1])
        descriptor = -1
        try:
            descriptor = os.open(parts[-1], _directory_flags(), dir_fd=parent_fd)
            os.fchmod(descriptor, 0o500)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent_fd)
    os.fchmod(site_fd, 0o500)


def _build_runtime_shadow(
    *,
    shadow_root_fd: int,
    shadow_root_path: Path,
    source_root: Path = FIXED_SITE_PACKAGES,
    _test_fault_hook: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Copy and seal only the frozen six-distribution import closure."""

    started = time.monotonic()
    root_metadata = os.fstat(shadow_root_fd)
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or int(root_metadata.st_uid) != os.getuid()
        or stat.S_IMODE(root_metadata.st_mode) != 0o700
    ):
        raise _error("runtime shadow root must be an owner-private 0700 directory")
    if _runtime_tree_root_entries(shadow_root_fd):
        raise _error("runtime shadow root must be empty before construction")
    if shutil.disk_usage(shadow_root_path).free < RUNTIME_SHADOW_MIN_FREE_BYTES:
        raise _error("runtime shadow filesystem has less than 512 MiB free")
    source_fd = _open_runtime_root(source_root, "fixed runtime site-packages")
    site_fd = -1
    try:
        source_scan = _scan_runtime_tree_fd(
            source_fd, exact_root_entries=False, require_sealed=False
        )
        if _test_fault_hook is not None:
            _test_fault_hook("after_source_scan")
        os.mkdir(RUNTIME_SHADOW_SITE_DIRECTORY, 0o700, dir_fd=shadow_root_fd)
        site_fd = os.open(
            RUNTIME_SHADOW_SITE_DIRECTORY,
            _directory_flags(),
            dir_fd=shadow_root_fd,
        )
        for relative in sorted(
            source_scan.directories,
            key=lambda value: (value.count("/"), value),
        ):
            try:
                os.mkdir(relative, 0o700, dir_fd=site_fd)
            except OSError as exc:
                raise _error(
                    f"cannot create runtime shadow directory: {relative}: {exc}"
                ) from exc
        for item in source_scan.files:
            if time.monotonic() - started > RUNTIME_SHADOW_WALL_SECONDS_MAX:
                raise _error("runtime shadow construction exceeded 120 seconds")
            _copy_runtime_file(
                source_root_fd=source_fd,
                destination_root_fd=site_fd,
                expected=item,
            )
        if _test_fault_hook is not None:
            _test_fault_hook("after_copy")
        source_after = _scan_runtime_tree_fd(
            source_fd, exact_root_entries=False, require_sealed=False
        )
        if source_after != source_scan:
            raise _error("runtime source tree drifted across scan/copy")
        copied = _scan_runtime_tree_fd(
            site_fd, exact_root_entries=True, require_sealed=False
        )
        if copied.descriptor != source_scan.descriptor:
            raise _error("runtime shadow bytes differ from the frozen source inventory")
        _seal_runtime_shadow(site_fd, copied)
        os.fchmod(shadow_root_fd, 0o500)
        sealed = _scan_runtime_tree_fd(
            site_fd, exact_root_entries=True, require_sealed=True
        )
        if sealed.descriptor != source_scan.descriptor:
            raise _error("sealed runtime shadow inventory differs after rebuild")
        if time.monotonic() - started > RUNTIME_SHADOW_WALL_SECONDS_MAX:
            raise _error("runtime shadow construction exceeded 120 seconds")
        return shadow_root_path / RUNTIME_SHADOW_SITE_DIRECTORY, copy.deepcopy(
            sealed.descriptor
        )
    finally:
        if site_fd >= 0:
            os.close(site_fd)
        os.close(source_fd)


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_native_shadow_path(
    token: str,
    *,
    loader: Path,
    shadow_site: Path,
    require_directory: bool = False,
) -> Path:
    if token == "@loader_path":
        suffix = ""
    elif token.startswith("@loader_path/"):
        suffix = token[len("@loader_path/") :]
    else:
        raise _error("native loader token is not @loader_path-relative")
    try:
        resolved = (loader.parent / suffix).resolve(strict=True)
    except OSError as exc:
        raise _error(f"native dependency cannot resolve in shadow: {token}") from exc
    if not _path_is_within(resolved, shadow_site):
        raise _error("native loader dependency escaped the runtime shadow")
    metadata = os.lstat(resolved)
    expected_type = (
        stat.S_ISDIR(metadata.st_mode)
        if require_directory
        else stat.S_ISREG(metadata.st_mode)
    )
    if stat.S_ISLNK(metadata.st_mode) or not expected_type:
        raise _error("native loader dependency has the wrong shadow type")
    return resolved


def _otool_output(option: str, path: Path) -> str:
    if option not in {"-L", "-l"}:
        raise _error("otool operation is not allowlisted")
    try:
        completed = subprocess.run(
            [str(FIXED_OTOOL), option, str(path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/",
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": "/var/empty",
                "TMPDIR": "/private/tmp",
                "LANG": "C",
                "LC_ALL": "C",
            },
            timeout=10,
            check=False,
            shell=False,
            close_fds=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _error(f"native dependency preflight failed: {path.name}: {exc}") from exc
    if (
        completed.returncode != 0
        or completed.stderr
        or len(completed.stdout) > 2 * 1024 * 1024
    ):
        raise _error(f"native dependency preflight rejected: {path.name}")
    try:
        return completed.stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise _error("otool output is not UTF-8") from exc


def _native_rpaths(loader: Path, shadow_site: Path) -> tuple[Path, ...]:
    lines = _otool_output("-l", loader).splitlines()
    tokens: list[str] = []
    for index, line in enumerate(lines):
        if line.strip() != "cmd LC_RPATH":
            continue
        for candidate in lines[index + 1 : index + 5]:
            stripped = candidate.strip()
            if stripped.startswith("path "):
                tokens.append(stripped[5:].split(" (offset ", 1)[0])
                break
        else:
            raise _error("native LC_RPATH lacks one exact path")
    resolved: list[Path] = []
    for token in tokens:
        if token.startswith("@loader_path"):
            path = _resolve_native_shadow_path(
                token,
                loader=loader,
                shadow_site=shadow_site,
                require_directory=True,
            )
        elif token.startswith("/"):
            try:
                path = Path(token).resolve(strict=True)
            except OSError as exc:
                raise _error("native absolute rpath is unavailable") from exc
            if not _path_is_within(path, shadow_site):
                raise _error("native absolute rpath escaped the runtime shadow")
        else:
            raise _error("native rpath uses a forbidden expansion")
        if not path.is_dir():
            raise _error("native rpath does not resolve to a shadow directory")
        resolved.append(path)
    if len(set(resolved)) != len(resolved):
        raise _error("native loader has duplicate resolved rpaths")
    return tuple(resolved)


def _preflight_native_shadow(
    *, shadow_site: Path, inventory: Mapping[str, Any]
) -> None:
    metadata = os.lstat(FIXED_OTOOL)
    if (
        not FIXED_OTOOL.is_absolute()
        or FIXED_OTOOL.resolve(strict=True) != FIXED_OTOOL
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or int(metadata.st_uid) != 0
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise _error("fixed /usr/bin/otool identity is unsafe")
    for row in inventory["files"]:
        relative = row["relative_path"]
        if not relative.endswith((".so", ".dylib", ".pyd", ".dll")):
            continue
        loader = shadow_site.joinpath(*relative.split("/"))
        if loader.resolve(strict=True) != loader:
            raise _error("native loader path is not exact inside the shadow")
        rpaths = _native_rpaths(loader, shadow_site)
        lines = _otool_output("-L", loader).splitlines()
        if not lines:
            raise _error("native loader dependency inventory is empty")
        dependencies = [
            line.strip().split(" (", 1)[0]
            for line in lines[1:]
            if line.strip()
        ]
        for dependency in dependencies:
            if dependency.startswith("/usr/lib/") or dependency.startswith(
                "/System/Library/"
            ):
                continue
            if dependency.startswith("@loader_path"):
                _resolve_native_shadow_path(
                    dependency, loader=loader, shadow_site=shadow_site
                )
                continue
            if dependency.startswith("@rpath/"):
                suffix = dependency[len("@rpath/") :]
                matches: list[Path] = []
                for rpath in rpaths:
                    try:
                        candidate = (rpath / suffix).resolve(strict=True)
                    except OSError:
                        continue
                    if _path_is_within(candidate, shadow_site) and candidate.is_file():
                        matches.append(candidate)
                if len(set(matches)) != 1:
                    raise _error("native @rpath dependency is missing or ambiguous")
                continue
            raise _error(f"native dependency is outside the allowlist: {dependency}")


def _rehash_runtime_shadow() -> dict[str, Any]:
    if (
        not _ISOLATED_CHILD_ACTIVE
        or _ACTIVE_RUNTIME_SHADOW_SITE is None
        or _ACTIVE_RUNTIME_SHADOW_INVENTORY is None
    ):
        raise _error("isolated runtime shadow is not active")
    descriptor = _open_runtime_root(
        _ACTIVE_RUNTIME_SHADOW_SITE, "active runtime shadow site-packages"
    )
    try:
        observed = _scan_runtime_tree_fd(
            descriptor, exact_root_entries=True, require_sealed=True
        ).descriptor
    finally:
        os.close(descriptor)
    if observed != _ACTIVE_RUNTIME_SHADOW_INVENTORY:
        raise _error("active runtime shadow drifted after import")
    return copy.deepcopy(observed)


def _open_parent_anchored(path: Path) -> tuple[int, str]:
    """Open an absolute parent one component at a time from the ``/`` dirfd."""

    descriptor = os.open("/", _directory_flags())
    try:
        for component in path.parts[1:-1]:
            child = os.open(component, _directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor, path.name
    except Exception:
        os.close(descriptor)
        raise


def _path_identity_at(parent_fd: int, leaf: str, opened: os.stat_result, label: str) -> None:
    try:
        current = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise _error(f"{label} path identity is unavailable") from exc
    if stat.S_ISLNK(current.st_mode) or not _same_object(current, opened):
        raise _error(f"{label} path identity differs from opened inode")


def _read_owner_private_once(path: Path, *, label: str, max_bytes: int) -> StableBytes:
    parent_fd, leaf = _open_parent_anchored(path)
    descriptor = -1
    try:
        descriptor = os.open(leaf, _file_flags(), dir_fd=parent_fd)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise _error(f"{label} must be a regular non-symlink file")
        if int(before.st_uid) != os.getuid():
            raise _error(f"{label} owner must be the current uid")
        if stat.S_IMODE(before.st_mode) != 0o600:
            raise _error(f"{label} mode must be 0600")
        if int(before.st_nlink) != 1:
            raise _error(f"{label} hard-link count must be one")
        if int(before.st_size) > max_bytes:
            raise _error(f"{label} exceeds the fixed byte limit")
        _path_identity_at(parent_fd, leaf, before, label)
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1_048_576, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise _error(f"{label} exceeds the fixed byte limit")
        after = os.fstat(descriptor)
        _path_identity_at(parent_fd, leaf, after, label)
        if _signature(before) != _signature(after) or total != int(after.st_size):
            raise _error(f"{label} changed during stable read")
        raw = b"".join(chunks)
        return StableBytes(path, raw, digest.hexdigest(), _signature(after))
    except OSError as exc:
        raise _error(f"{label} anchored read failed: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def _read_bound_regular_once(
    path: Path,
    *,
    label: str,
    max_bytes: int,
    expected_sha256: str | None = None,
    expected_nlink: int | None = 1,
) -> StableBytes:
    """Read one fixed code/control file through an anchored stable inode."""

    parent_fd, leaf = _open_parent_anchored(path)
    descriptor = -1
    try:
        descriptor = os.open(leaf, _file_flags(), dir_fd=parent_fd)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or int(before.st_uid) != os.getuid():
            raise _error(f"{label} must be a current-owner regular file")
        if expected_nlink is not None and int(before.st_nlink) != expected_nlink:
            raise _error(f"{label} hard-link count mismatch")
        if int(before.st_size) > max_bytes:
            raise _error(f"{label} exceeds its fixed byte cap")
        _path_identity_at(parent_fd, leaf, before, label)
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1_048_576, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise _error(f"{label} exceeds its fixed byte cap")
        after = os.fstat(descriptor)
        _path_identity_at(parent_fd, leaf, after, label)
        if _signature(before) != _signature(after) or total != int(after.st_size):
            raise _error(f"{label} changed during anchored read")
        actual_sha = digest.hexdigest()
        if expected_sha256 is not None and actual_sha != expected_sha256:
            raise _error(f"{label} byte SHA mismatch")
        return StableBytes(path, b"".join(chunks), actual_sha, _signature(after))
    except OSError as exc:
        raise _error(f"{label} anchored read failed: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def _read_manifest_two_fresh(
    path_value: Any, *, expected_byte_sha256: Any
) -> tuple[dict[str, Any], StableBytes]:
    path = _absolute_normalized_path(path_value, "input manifest")
    expected = _sha256(expected_byte_sha256, "expected input manifest byte SHA")
    first = _read_owner_private_once(
        path, label="input manifest first observation", max_bytes=MAX_MANIFEST_BYTES
    )
    if first.byte_sha256 != expected:
        raise _error("input manifest byte SHA mismatch")
    first_value = _strict_json_object(first.raw, "input manifest")
    second = _read_owner_private_once(
        path, label="input manifest second observation", max_bytes=MAX_MANIFEST_BYTES
    )
    if (
        second.byte_sha256 != expected
        or second.signature != first.signature
        or second.raw != first.raw
    ):
        raise _error("input manifest differs across two fresh stable opens")
    second_value = _strict_json_object(second.raw, "input manifest")
    if second_value != first_value:
        raise _error("input manifest parse differs across two fresh opens")
    return first_value, second


def deterministic_cycle_id(*, cutoff: str, snapshot_id: str) -> str:
    return (
        "cn_full_a_v4_4_strict_computability_"
        f"{cutoff.replace('-', '')}_{snapshot_id}"
    )


def _stage0_manifest_validate(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate all intake-safe shape/identity rules without project imports."""

    payload = _exact_mapping(value, _TOP_LEVEL_FIELDS, "input manifest")
    if (
        payload["schema_version"] != MANIFEST_SCHEMA_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["evidence_contract_version"] != EVIDENCE_CONTRACT_VERSION
    ):
        raise _error("input manifest schema/protocol identity mismatch")
    cutoff = _canonical_date(payload["cutoff"], "input manifest cutoff")
    if date.fromisoformat(cutoff) <= date.fromisoformat(FROZEN_PREVIOUS_CUTOFF):
        raise _error("input manifest cutoff must be strictly later than 2026-07-19")
    snapshot_id = payload["snapshot_id"]
    if type(snapshot_id) is not str or _SNAPSHOT_RE.fullmatch(snapshot_id) is None:
        raise _error("input manifest snapshot_id must be YYYYMMDDTHHMMSSZ")
    try:
        parsed_snapshot = datetime.strptime(snapshot_id, "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise _error("input manifest snapshot_id must be a real UTC timestamp") from exc
    if parsed_snapshot.strftime("%Y%m%dT%H%M%SZ") != snapshot_id:
        raise _error("input manifest snapshot_id must be canonical")
    expected_cycle = deterministic_cycle_id(cutoff=cutoff, snapshot_id=snapshot_id)
    if payload["cycle_id"] != expected_cycle:
        raise _error("input manifest cycle_id is not deterministic")
    proof_start = _canonical_date(
        payload["proof_output_start"], "input manifest proof_output_start"
    )
    if proof_start > cutoff:
        raise _error("proof_output_start must not be after cutoff")

    preregistration = _exact_mapping(
        payload["preregistration"], _PREREGISTRATION_FIELDS, "preregistration"
    )
    _absolute_normalized_path(preregistration["bundle_path"], "preregistration bundle")
    _sha256(preregistration["readback_byte_sha256"], "preregistration readback byte SHA")
    _sha256(
        preregistration["readback_semantic_sha256"],
        "preregistration readback semantic SHA",
    )
    _sha256(
        preregistration["candidate_rows_semantic_sha256"],
        "preregistration candidate rows semantic SHA",
    )
    if preregistration["artifact_count"] != 27:
        raise _error("preregistration artifact_count must be exact 27")
    if type(preregistration["cycle_id"]) is not str or not preregistration["cycle_id"]:
        raise _error("preregistration cycle_id must be non-empty")

    strict_source = _exact_mapping(
        payload["strict_source_expected"],
        _STRICT_SOURCE_EXPECTED_FIELDS,
        "strict_source_expected",
    )
    for key, item in strict_source.items():
        if key == "full_a_scope_count":
            _positive_int(item, f"strict_source_expected.{key}")
        else:
            _sha256(item, f"strict_source_expected.{key}")

    definitions = payload["source_definition_bindings"]
    if not isinstance(definitions, list) or len(definitions) != 5:
        raise _error("source_definition_bindings must contain exact five rows")
    seen_names: set[str] = set()
    seen_identities: set[str] = set()
    normalized_definitions: list[dict[str, Any]] = []
    for index, item in enumerate(definitions, start=1):
        row = _exact_mapping(
            item, _SOURCE_DEFINITION_FIELDS, f"source_definition_bindings[{index - 1}]"
        )
        if row["order"] != index:
            raise _error("source definition order must be exact 1..5")
        if type(row["name"]) is not str or not row["name"] or row["name"] in seen_names:
            raise _error("source definition names must be distinct non-empty strings")
        seen_names.add(row["name"])
        identity = _sha256(
            row["definition_identity_sha256"], "definition identity SHA"
        )
        if identity in seen_identities:
            raise _error("source definition identities must be distinct")
        seen_identities.add(identity)
        if type(row["direction"]) is not int or row["direction"] not in {-1, 1}:
            raise _error("source definition direction must be exact integer -1 or 1")
        for key in ("source_repository", "source_relative_path"):
            if type(row[key]) is not str or not row[key]:
                raise _error(f"source definition {key} must be non-empty")
        if (
            row["source_relative_path"].startswith("/")
            or any(
                part in {"", ".", ".."}
                for part in Path(row["source_relative_path"]).parts
            )
        ):
            raise _error("source definition relative path is unsafe")
        for key in ("source_commit", "source_tree_oid", "source_blob_oid"):
            if type(row[key]) is not str or _OID_RE.fullmatch(row[key]) is None:
                raise _error(
                    f"source definition {key} must be a 40-hex Git identity"
                )
        for key in (
            "source_raw_sha256",
            "source_ast_sha256",
            "field_semantics_sha256",
            "operator_program_sha256",
            "operator_program_set_sha256",
        ):
            _sha256(row[key], f"source definition {key}")
        normalized_definitions.append(row)

    code_rows = payload["code_binding_set"]
    if not isinstance(code_rows, list) or len(code_rows) != len(FIXED_CODE_BINDING_PATHS):
        raise _error("code_binding_set must contain the exact fixed code inventory")
    normalized_code: list[dict[str, str]] = []
    for expected_path, item in zip(FIXED_CODE_BINDING_PATHS, code_rows, strict=True):
        row = _exact_mapping(item, _CODE_BINDING_FIELDS, "code binding row")
        if row["relative_path"] != expected_path:
            raise _error("code_binding_set path inventory/order mismatch")
        normalized_code.append(
            {
                "relative_path": expected_path,
                "byte_sha256": _sha256(row["byte_sha256"], f"code SHA {expected_path}"),
            }
        )
    _sha256(
        payload["runtime_binding_expected_semantic_sha256"],
        "runtime binding expected semantic SHA",
    )
    controls = _exact_mapping(
        payload["protected_control_expected_sha256"],
        _PROTECTED_CONTROL_FIELDS,
        "protected_control_expected_sha256",
    )
    for key, item in controls.items():
        _sha256(item, f"protected control SHA {key}")

    resources = _exact_mapping(
        payload["resource_contract"], _RESOURCE_FIELDS, "resource_contract"
    )
    for key, item in resources.items():
        _positive_int(item, f"resource_contract.{key}")
    if resources != RESOURCE_CONTRACT:
        raise _error("resource_contract must equal the fixed non-adaptive limits")

    disclosures = _exact_mapping(
        payload["selection_disclosures"],
        _SELECTION_DISCLOSURE_FIELDS,
        "selection_disclosures",
    )
    if disclosures != SELECTION_DISCLOSURES:
        raise _error("selection disclosures must preserve exact outcome-informed claims")
    negative = _exact_mapping(
        payload["negative_claims"], _NEGATIVE_CLAIM_SECTIONS, "negative_claims"
    )
    if negative != NEGATIVE_CLAIMS:
        raise _error("negative_claims must equal the fixed non-authorizing claims")

    payload["preregistration"] = preregistration
    payload["strict_source_expected"] = strict_source
    payload["source_definition_bindings"] = normalized_definitions
    payload["code_binding_set"] = normalized_code
    payload["protected_control_expected_sha256"] = controls
    payload["resource_contract"] = resources
    payload["selection_disclosures"] = disclosures
    payload["negative_claims"] = copy.deepcopy(negative)
    return payload


def _read_private_bundle_raw_snapshot(
    path_value: Any,
    *,
    root_suffix: Sequence[str],
    input_filenames: Sequence[str],
    readback_filename: str,
    expected_readback_byte_sha256: str,
    expected_readback_semantic_sha256: str,
    artifact_max_bytes: int,
    bundle_max_bytes: int,
    label: str,
) -> RawPrivateBundleSnapshot:
    """Authenticate one private bundle without importing project code."""

    path = _absolute_normalized_path(path_value, f"{label} path")
    if tuple(path.parent.parts[-len(root_suffix) :]) != tuple(root_suffix):
        raise _error(f"{label} parent root suffix mismatch")
    directory_fd, directory_metadata = _open_anchored_directory(path, label=label)
    try:
        if (
            int(directory_metadata.st_uid) != os.getuid()
            or stat.S_IMODE(directory_metadata.st_mode) != 0o700
            or int(directory_metadata.st_nlink) < 2
        ):
            raise _error(f"{label} must be a current-owner mode-0700 directory")
        observed_names = tuple(sorted(os.listdir(directory_fd)))
    finally:
        os.close(directory_fd)
    expected_names = (*tuple(input_filenames), readback_filename)
    if observed_names != tuple(sorted(expected_names)):
        raise _error(f"{label} exact private artifact inventory mismatch")

    files: dict[str, StableBytes] = {}
    values: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for filename in expected_names:
        stable = _read_bound_regular_once(
            path / filename,
            label=f"{label} {filename}",
            max_bytes=artifact_max_bytes,
            expected_nlink=1,
        )
        if stat.S_IMODE(stable.signature[2]) != 0o600:
            raise _error(f"{label} artifact must be mode 0600: {filename}")
        total_bytes += len(stable.raw)
        if total_bytes > bundle_max_bytes:
            raise _error(f"{label} exceeds its fixed aggregate byte cap")
        files[filename] = stable
        values[filename] = _strict_json_object(
            stable.raw, f"{label} {filename}"
        )

    report_file = files[readback_filename]
    report = values[readback_filename]
    if report_file.byte_sha256 != _sha256(
        expected_readback_byte_sha256, f"{label} expected report byte SHA"
    ):
        raise _error(f"{label} readback byte SHA mismatch")
    supplied_semantic = _sha256(
        report.get("artifact_semantic_sha256"),
        f"{label} report semantic SHA",
    )
    if (
        supplied_semantic != expected_readback_semantic_sha256
        or supplied_semantic
        != _semantic_sha(
            {
                key: copy.deepcopy(item)
                for key, item in report.items()
                if key != "artifact_semantic_sha256"
            }
        )
    ):
        raise _error(f"{label} readback semantic SHA mismatch")
    bindings = report.get("artifact_bindings")
    if not isinstance(bindings, list) or len(bindings) != len(input_filenames):
        raise _error(f"{label} readback input binding inventory mismatch")
    for filename, row_value in zip(input_filenames, bindings, strict=True):
        if not isinstance(row_value, Mapping):
            raise _error(f"{label} readback binding must be an object")
        row = dict(row_value)
        stable = files[filename]
        if (
            row.get("filename") != filename
            or row.get("byte_sha256") != stable.byte_sha256
            or row.get("size_bytes") != len(stable.raw)
            or row.get("mode") != 0o600
            or row.get("uid") != os.getuid()
            or row.get("nlink") != 1
        ):
            raise _error(f"{label} readback byte/file binding mismatch: {filename}")
    return RawPrivateBundleSnapshot(path=path, values=values, files=files)


def _code_binding_rows_from_artifact(
    value: Mapping[str, Any],
    *,
    label: str,
    expected_schema_version: str,
    expected_paths: Sequence[str],
    expect_evidence_contract_version: bool,
) -> tuple[dict[str, Any], ...]:
    payload = copy.deepcopy(dict(value))
    expected_fields = {
        "schema_version",
        "protocol_version",
        "path_count",
        "ordered_bindings",
        "artifact_semantic_sha256",
    }
    if expect_evidence_contract_version:
        expected_fields.add("evidence_contract_version")
    if set(payload) != expected_fields:
        raise _error(f"{label} exact schema fields mismatch")
    if (
        payload["schema_version"] != expected_schema_version
        or payload["protocol_version"] != PROTOCOL_VERSION
        or (
            expect_evidence_contract_version
            and payload["evidence_contract_version"]
            != EVIDENCE_CONTRACT_VERSION
        )
    ):
        raise _error(f"{label} schema/protocol identity mismatch")
    supplied = _sha256(
        payload.get("artifact_semantic_sha256"), f"{label} self SHA"
    )
    if supplied != _semantic_sha(
        {
            key: copy.deepcopy(item)
            for key, item in payload.items()
            if key != "artifact_semantic_sha256"
        }
    ):
        raise _error(f"{label} self SHA mismatch")
    rows = payload.get("ordered_bindings")
    if (
        type(rows) is not list
        or len(rows) != len(expected_paths)
        or payload.get("path_count") != len(expected_paths)
    ):
        raise _error(f"{label} ordered binding inventory mismatch")
    normalized: list[dict[str, Any]] = []
    for index, (row_value, expected_path) in enumerate(
        zip(rows, expected_paths, strict=True), start=1
    ):
        if not isinstance(row_value, Mapping):
            raise _error(f"{label} binding[{index}] must be an object")
        row = dict(row_value)
        if set(row) != {"order", "relative_path", "byte_sha256", "size_bytes"}:
            raise _error(f"{label} binding[{index}] exact fields mismatch")
        relative = row.get("relative_path")
        if (
            row.get("order") != index
            or relative != expected_path
            or type(row.get("size_bytes")) is not int
            or row["size_bytes"] <= 0
        ):
            raise _error(f"{label} binding[{index}] identity is malformed")
        normalized.append(
            {
                "relative_path": relative,
                "byte_sha256": _sha256(
                    row.get("byte_sha256"), f"{label} binding[{index}] SHA"
                ),
                "size_bytes": row["size_bytes"],
            }
        )
    return tuple(normalized)


def _verified_code_union(
    *,
    manifest: Mapping[str, Any],
    preregistration: RawPrivateBundleSnapshot | None,
) -> tuple[StableBytes, ...]:
    expected: dict[str, tuple[str, int | None]] = {}

    def add(relative: str, digest: str, size: int | None, label: str) -> None:
        prior = expected.get(relative)
        candidate = (digest, size)
        if prior is not None and (
            prior[0] != digest
            or (prior[1] is not None and size is not None and prior[1] != size)
        ):
            raise _error(f"conflicting authenticated code binding: {relative} ({label})")
        expected[relative] = (
            digest,
            prior[1] if prior is not None and prior[1] is not None else size,
        )

    for row in manifest["code_binding_set"]:
        add(row["relative_path"], row["byte_sha256"], None, "strict manifest")
    if preregistration is not None:
        for (
            filename,
            label,
            schema_version,
            paths,
            expect_evidence_contract_version,
        ) in (
            (
                "code_binding_set.v4_4.json",
                "v4.4 preregistration code",
                "factor-governance-code-binding-set.v4.4",
                PREREGISTRATION_CODE_BINDING_PATHS,
                True,
            ),
            (
                "v4_2_predecessor.code_binding_set.v4_2.json",
                "v4.2 predecessor code",
                "factor-governance-code-binding-set.v4.2",
                PREDECESSOR_CODE_BINDING_PATHS,
                False,
            ),
        ):
            for row in _code_binding_rows_from_artifact(
                preregistration.values[filename],
                label=label,
                expected_schema_version=schema_version,
                expected_paths=paths,
                expect_evidence_contract_version=(
                    expect_evidence_contract_version
                ),
            ):
                add(
                    row["relative_path"],
                    row["byte_sha256"],
                    row["size_bytes"],
                    label,
                )
    observations: list[StableBytes] = []
    for relative, (digest, size) in expected.items():
        stable = _read_bound_regular_once(
            PROJECT_ROOT / relative,
            label=f"pre-import authenticated code {relative}",
            max_bytes=max(size or 16 * 1024 * 1024, 1),
            expected_sha256=digest,
            expected_nlink=1,
        )
        if size is not None and len(stable.raw) != size:
            raise _error(f"pre-import authenticated code size mismatch: {relative}")
        observations.append(stable)
    return tuple(observations)


class _VerifiedSourceLoader(importlib.abc.Loader):
    def __init__(self, stable: StableBytes) -> None:
        self.stable = stable

    def create_module(self, spec: Any) -> Any:
        return None

    def exec_module(self, module: Any) -> None:
        module.__file__ = str(self.stable.path)
        code = compile(
            self.stable.raw,
            str(self.stable.path),
            "exec",
            dont_inherit=True,
        )
        exec(code, module.__dict__)


class _ClosedVerifiedFinder(importlib.abc.MetaPathFinder):
    def __init__(self, code_files: Sequence[StableBytes]) -> None:
        self.sources: dict[str, StableBytes] = {}
        self.verified_paths = {stable.path.resolve() for stable in code_files}
        for stable in code_files:
            try:
                relative = stable.path.relative_to(PROJECT_ROOT)
            except ValueError:
                continue
            if relative.suffix != ".py" or not relative.parts or relative.parts[0] != "quant_investor":
                continue
            parts = list(relative.with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            module_name = ".".join(parts)
            if module_name in self.sources:
                raise _error(f"duplicate verified module source: {module_name}")
            self.sources[module_name] = stable
        self.namespaces = {
            ".".join(name.split(".")[:index])
            for name in self.sources
            for index in range(1, len(name.split(".")))
        }
        self.distribution_roots: dict[str, Path] = {}
        self.stdlib_top_levels = frozenset(sys.stdlib_module_names) | {
            "__future__",
        }
        os_source = getattr(os, "__file__", None)
        if os_source is None:
            raise _error("stdlib origin root is unavailable")
        stdlib = Path(os_source).resolve().parent
        self.stdlib_roots = (stdlib, stdlib / "lib-dynload")

    def _is_stdlib_location(self, location: Path) -> bool:
        if "site-packages" in location.parts or _path_is_within(
            location, PROJECT_ROOT
        ):
            return False
        for root in self.stdlib_roots:
            try:
                relative = location.relative_to(root)
            except ValueError:
                continue
            if relative.parts and relative.parts[0] == "site-packages":
                return False
            return True
        return False

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname in self.sources:
            stable = self.sources[fullname]
            return importlib.util.spec_from_loader(
                fullname,
                _VerifiedSourceLoader(stable),
                origin=str(stable.path),
            )
        if fullname.startswith("quant_investor"):
            raise ImportError(f"unbound project module import rejected: {fullname}")
        top = fullname.split(".", 1)[0]
        if top not in self.distribution_roots:
            if top in sys.builtin_module_names:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None:
                if top in self.stdlib_top_levels:
                    return None
                raise ImportError(f"unbound external module import rejected: {fullname}")
            locations: list[Path] = []
            if spec.origin not in {None, "built-in", "frozen"}:
                locations.append(Path(spec.origin).resolve())
            if spec.submodule_search_locations is not None:
                locations.extend(
                    Path(item).resolve()
                    for item in spec.submodule_search_locations
                )
            if not locations or not all(
                self._is_stdlib_location(location) for location in locations
            ):
                raise ImportError(
                    f"shadowed or unbound external module rejected: {fullname}"
                )
            return spec
        if top == "six" and fullname.startswith("six.moves"):
            six_module = sys.modules.get("six")
            six_origin = getattr(six_module, "__file__", None)
            if six_origin is None or Path(six_origin).resolve() != (
                self.distribution_roots["six"]
            ):
                raise ImportError("unbound six.moves virtual import rejected")
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None:
            raise ImportError(f"bound distribution module unavailable: {fullname}")
        anchor = self.distribution_roots[top]
        locations: list[Path] = []
        if spec.origin not in {None, "built-in", "frozen"}:
            locations.append(Path(spec.origin).resolve())
        if spec.submodule_search_locations is not None:
            locations.extend(
                Path(item).resolve() for item in spec.submodule_search_locations
            )
        for location in locations:
            if anchor.is_file():
                permitted = location == anchor
            else:
                try:
                    location.relative_to(anchor)
                except ValueError:
                    permitted = False
                else:
                    permitted = True
            if not permitted:
                raise ImportError(
                    f"shadowed distribution module origin rejected: {fullname}"
                )
        return spec


def _reject_preloaded_execution_modules() -> None:
    bound_top_levels = {row[1] for row in RUNTIME_DISTRIBUTION_TOP_LEVEL}
    prohibited = sorted(
        name
        for name in sys.modules
        if name == "quant_investor"
        or name.startswith("quant_investor.")
        or name.split(".", 1)[0] in bound_top_levels
    )
    if prohibited:
        raise _error(
            "preloaded project/data modules violate the closed import boundary: "
            + ",".join(prohibited[:8])
        )


def _install_verified_finder(
    code_files: Sequence[StableBytes],
) -> _ClosedVerifiedFinder:
    _reject_preloaded_execution_modules()
    finder = _ClosedVerifiedFinder(code_files)
    for namespace in sorted(finder.namespaces, key=lambda item: item.count(".")):
        module = types.ModuleType(namespace)
        module.__package__ = namespace
        relative = Path(*namespace.split("."))
        module.__path__ = [str(PROJECT_ROOT / relative)]
        spec = importlib.machinery.ModuleSpec(namespace, loader=None, is_package=True)
        spec.submodule_search_locations = list(module.__path__)
        module.__spec__ = spec
        sys.modules[namespace] = module
        parent_name, _, child_name = namespace.rpartition(".")
        if parent_name:
            setattr(sys.modules[parent_name], child_name, module)
    sys.meta_path.insert(0, finder)
    return finder


def _trusted_site_packages_root() -> Path:
    if not _ISOLATED_CHILD_ACTIVE or _ACTIVE_RUNTIME_SHADOW_SITE is None:
        raise _error("trusted runtime site-packages requires the isolated shadow")
    root = _ACTIVE_RUNTIME_SHADOW_SITE
    _real_owned_directory(root, "trusted runtime site-packages root")
    return root


def _require_trusted_distribution_base(distribution: Any, name: str) -> None:
    try:
        base = Path(distribution.locate_file("")).resolve(strict=True)
    except OSError as exc:
        raise _error(f"runtime distribution base is unavailable: {name}") from exc
    if base != _trusted_site_packages_root():
        raise _error(f"runtime distribution base is untrusted: {name}")


def _distribution_package_roots() -> dict[str, Path]:
    metadata_module = importlib.import_module("importlib.metadata")
    roots: dict[str, Path] = {}
    for distribution_name, top_level, anchor_relative in (
        RUNTIME_DISTRIBUTION_TOP_LEVEL
    ):
        distribution = metadata_module.distribution(distribution_name)
        _require_trusted_distribution_base(distribution, distribution_name)
        anchor = _trusted_site_packages_root() / anchor_relative
        if not anchor.is_file():
            raise _error(
                f"runtime distribution anchor is unavailable: {distribution_name}"
            )
        roots[top_level] = anchor if anchor_relative == "six.py" else anchor.parent
    return roots


def _purge_verified_project_modules(finder: _ClosedVerifiedFinder) -> None:
    names = sorted(
        set(finder.sources) | set(finder.namespaces),
        key=lambda item: item.count("."),
        reverse=True,
    )
    for name in names:
        module = sys.modules.pop(name, None)
        if module is None:
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None and getattr(parent, child_name, None) is module:
            delattr(parent, child_name)


def _audit_closed_project_imports(finder: _ClosedVerifiedFinder) -> None:
    for name, module in tuple(sys.modules.items()):
        if not (name == "quant_investor" or name.startswith("quant_investor.")):
            continue
        if name in finder.namespaces:
            continue
        stable = finder.sources.get(name)
        origin = getattr(module, "__file__", None)
        if stable is None or origin is None or Path(origin) != stable.path:
            raise _error(f"project import escaped the verified closed set: {name}")


def _audit_closed_runtime_imports(
    finder: _ClosedVerifiedFinder,
    *,
    _module_items: Sequence[tuple[str, Any]] | None = None,
) -> None:
    """Classify every loaded module into stdlib, verified project, or shadow."""

    bound_tops = set(finder.distribution_roots)
    pandas_injected_runtime = {"cython_runtime", "_cython_3_2_4"}
    cyutility = sys.modules.get("_cyutility")
    cyutility_origin = getattr(cyutility, "__file__", None)
    cyutility_spec = getattr(cyutility, "__spec__", None)
    cyutility_is_bound = (
        cyutility_origin is not None
        and getattr(cyutility_spec, "name", None) == "pandas._libs._cyutility"
        and "pandas" in finder.distribution_roots
    )
    if cyutility_is_bound:
        try:
            Path(cyutility_origin).resolve().relative_to(
                finder.distribution_roots["pandas"]
            )
        except ValueError:
            cyutility_is_bound = False
    module_items = (
        tuple(sys.modules.items())
        if _module_items is None
        else tuple(_module_items)
    )
    for name, module in module_items:
        if module is None:
            raise _error(f"loaded module has no module object: {name}")
        top = name.split(".", 1)[0]
        if name == "quant_investor" or name.startswith("quant_investor."):
            continue
        origin = getattr(module, "__file__", None)
        spec = getattr(module, "__spec__", None)
        spec_origin = getattr(spec, "origin", None)
        if top in bound_tops:
            if origin is None:
                if top == "six" and name.startswith("six.moves"):
                    continue
                raise _error(f"bound runtime module has no auditable origin: {name}")
            observed = Path(origin).resolve()
            anchor = finder.distribution_roots[top]
            if anchor.is_file():
                permitted = observed == anchor
            else:
                try:
                    observed.relative_to(anchor)
                except ValueError:
                    permitted = False
                else:
                    permitted = True
            if not permitted:
                raise _error(f"bound runtime module escaped its distribution: {name}")
            continue
        if (
            origin is not None
            and getattr(spec, "name", None) == "pandas._libs._cyutility"
            and cyutility_is_bound
        ):
            continue
        if (
            name in pandas_injected_runtime
            and cyutility_is_bound
            and origin is None
            and spec is None
            and getattr(module, "__loader__", None) is None
        ):
            continue
        if origin is not None:
            observed = Path(origin).resolve()
            if observed in finder.verified_paths:
                continue
            if finder._is_stdlib_location(observed):
                continue
            raise _error(f"unbound module origin escaped the closed roots: {name}")
        if spec_origin in {"built-in", "frozen"}:
            continue
        if top == "six" and name.startswith("six.moves"):
            continue
        raise _error(f"unbound/no-origin runtime module loaded: {name}")


def _load_project_modules_after_stage0(
    *,
    manifest: Mapping[str, Any],
    preregistration_snapshot: RawPrivateBundleSnapshot,
) -> LoadedModules:
    """Load only authenticated code without executing package initializers."""

    code_files = _verified_code_union(
        manifest=manifest, preregistration=preregistration_snapshot
    )
    finder = _install_verified_finder(code_files)
    try:
        contract = importlib.import_module(
            "quant_investor.factors.governance_future_strict_signal_computability_v4_4"
        )
        private_io = importlib.import_module(
            "quant_investor.factors.governance_private_bundle_io"
        )
        runtime = _runtime_binding(contract=contract)
        if runtime["artifact_semantic_sha256"] != manifest[
            "runtime_binding_expected_semantic_sha256"
        ]:
            raise _error("pre-import runtime binding semantic SHA mismatch")
        finder.distribution_roots = _distribution_package_roots()
        prereg_core = importlib.import_module(
            "quant_investor.factors.governance_candidate_preregistration_v4_4"
        )
        prereg_bundle = importlib.import_module(
            "quant_investor.factors.governance_candidate_preregistration_bundle_v4_4"
        )
        predecessor_bundle = importlib.import_module(
            "quant_investor.factors.governance_candidate_preregistration_bundle_v4_2"
        )
        _audit_closed_project_imports(finder)
        _audit_closed_runtime_imports(finder)
        _rehash_runtime_shadow()
    except Exception:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        _purge_verified_project_modules(finder)
        raise
    return LoadedModules(
        contract=contract,
        prereg_core=prereg_core,
        prereg_bundle=prereg_bundle,
        predecessor_bundle=predecessor_bundle,
        private_io=private_io,
        prebound_runtime=runtime,
        import_guard=finder,
        preregistration_snapshot=preregistration_snapshot,
    )


def _load_readback_modules_after_stage0(
    *, manifest: Mapping[str, Any]
) -> LoadedModules:
    """Load only authenticated contract/private-I/O code for sealed readback."""

    code_files = _verified_code_union(manifest=manifest, preregistration=None)
    finder = _install_verified_finder(code_files)
    try:
        contract = importlib.import_module(
            "quant_investor.factors.governance_future_strict_signal_computability_v4_4"
        )
        private_io = importlib.import_module(
            "quant_investor.factors.governance_private_bundle_io"
        )
        _audit_closed_project_imports(finder)
        _audit_closed_runtime_imports(finder)
        _rehash_runtime_shadow()
    except Exception:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        _purge_verified_project_modules(finder)
        raise
    return LoadedModules(
        contract=contract,
        prereg_core=None,
        prereg_bundle=None,
        predecessor_bundle=None,
        private_io=private_io,
        import_guard=finder,
    )


def _teardown_loaded_modules(modules: LoadedModules) -> None:
    finder = modules.import_guard
    if not isinstance(finder, _ClosedVerifiedFinder):
        return
    if finder in sys.meta_path:
        sys.meta_path.remove(finder)
    _purge_verified_project_modules(finder)


def _canonical_bytes_local(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise _error(f"value is not finite canonical JSON: {exc}") from exc


def _assert_immutable_source_path(
    value: Any, *, expected: Path, label: str
) -> Path:
    path = _absolute_normalized_path(value, label)
    if path != expected:
        raise _error(f"{label} is not the exact recorded immutable repo path")
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise _error(f"{label} escapes the fixed repository root") from exc
    return path


def _rebuild_candidate_rows_from_prereg_graph(
    artifacts: Mapping[str, Mapping[str, Any]], modules: LoadedModules
) -> tuple[dict[str, Any], ...]:
    """Rebuild the exact-five selection from its v4.2 and v4.3 source nodes."""

    core = modules.prereg_core
    bundle = modules.prereg_bundle
    predecessor = modules.predecessor_bundle
    prefix = core.V4_2_PREDECESSOR_PREFIX
    expanded = artifacts[bundle.EXPANDED_CANDIDATE_SELECTION_FILENAME_V4_4]
    diagnostic_files = tuple(bundle.PRIOR_DIAGNOSTIC_FILENAMES_V4_4)
    diagnostic_bindings = [
        core.build_artifact_binding_v4_4(
            filename=filename, artifact=artifacts[filename]
        )
        for filename in diagnostic_files
    ]
    rebuilt = core.build_expanded_candidate_selection_v4_4(
        v4_2_selection_spec=artifacts[
            prefix + predecessor.CANDIDATE_SELECTION_SPEC_FILENAME_V4_2
        ],
        v4_2_aquant_receipt=artifacts[
            prefix + predecessor.AQUANT_IDEA_SOURCE_RECEIPT_FILENAME_V4_2
        ],
        v4_2_myquant_receipt=artifacts[
            prefix + predecessor.MYQUANT_ALPHA158_SOURCE_RECEIPT_FILENAME_V4_2
        ],
        v4_2_operator_semantics=artifacts[
            prefix + predecessor.OPERATOR_SEMANTICS_FILENAME_V4_2
        ],
        v4_2_comparison_catalog_receipt=artifacts[
            prefix + predecessor.COMPARISON_CATALOG_RECEIPT_FILENAME_V4_2
        ],
        prior_diagnostic_nomination=artifacts[diagnostic_files[1]],
        diagnostic_artifact_bindings=diagnostic_bindings,
    )
    if core.canonical_file_bytes_v4_4(rebuilt) != core.canonical_file_bytes_v4_4(
        expanded
    ):
        raise _error("preregistration expanded selection did not rebuild from graph")
    rows = rebuilt.get("candidates")
    if not isinstance(rows, list) or len(rows) != 5:
        raise _error("rebuilt preregistration must contain exact five candidates")
    return tuple(copy.deepcopy(dict(row)) for row in rows)


def _accept_preregistration(
    *, manifest: Mapping[str, Any], modules: LoadedModules
) -> AcceptedPreregistration:
    """Accept one explicit future v4.4 26+1 bundle before any root/data probe."""

    try:
        normalized_manifest = modules.contract.validate_input_manifest_v4_4(manifest)
    except Exception as exc:
        raise _error(f"full input manifest validation failed: {exc}") from exc
    if _canonical_bytes_local(normalized_manifest) != _canonical_bytes_local(manifest):
        raise _error("full input manifest validation changed canonical semantics")
    prereg = normalized_manifest["preregistration"]
    bundle_path = _absolute_normalized_path(
        prereg["bundle_path"], "preregistration bundle_path"
    )
    try:
        result = modules.prereg_bundle.readback_candidate_preregistration_bundle_files_v4_4(
            bundle_path
        )
    except Exception as exc:
        raise _error(f"future v4.4 preregistration is missing or rejected: {exc}") from exc
    if result.get("accepted") is not True:
        raise _error("future v4.4 preregistration readback was not accepted")
    report_filename = modules.prereg_bundle.READBACK_REPORT_FILENAME_V4_4
    descriptors = result.get("artifact_descriptors")
    artifacts = result.get("artifacts")
    report = result.get("readback_report")
    exact_files = (
        *modules.prereg_bundle.INPUT_FILENAMES_V4_4,
        report_filename,
    )
    if (
        not isinstance(descriptors, Mapping)
        or tuple(descriptors) != exact_files
        or not isinstance(artifacts, Mapping)
        or tuple(artifacts) != exact_files
        or not isinstance(report, Mapping)
        or len(exact_files) != 27
        or prereg["artifact_count"] != len(exact_files)
    ):
        raise _error("preregistration must be the exact validated 26+1 inventory")
    raw_snapshot = modules.preregistration_snapshot
    if raw_snapshot is not None:
        if raw_snapshot.path != bundle_path or tuple(raw_snapshot.values) != exact_files:
            raise _error("validated preregistration differs from pre-import intake")
        for filename in exact_files:
            if _canonical_bytes_local(raw_snapshot.values[filename]) != (
                _canonical_bytes_local(artifacts[filename])
            ):
                raise _error(
                    f"preregistration changed across pre-import validation: {filename}"
                )
    total_bytes = 0
    for filename in exact_files:
        row = descriptors[filename]
        size = row.get("size_bytes") if isinstance(row, Mapping) else None
        if type(size) is not int or size <= 0:
            raise _error(f"preregistration descriptor size is invalid: {filename}")
        if size > RESOURCE_CONTRACT["prereg_artifact_max_bytes"]:
            raise _error(f"preregistration artifact resource cap exceeded: {filename}")
        total_bytes += size
    if total_bytes > RESOURCE_CONTRACT["prereg_bundle_max_bytes"]:
        raise _error("preregistration bundle resource cap exceeded")
    if descriptors[report_filename].get("byte_sha256") != prereg[
        "readback_byte_sha256"
    ]:
        raise _error("preregistration readback byte SHA mismatch")
    if report.get("artifact_semantic_sha256") != prereg[
        "readback_semantic_sha256"
    ]:
        raise _error("preregistration readback semantic SHA mismatch")
    root = artifacts[modules.prereg_bundle.CYCLE_ROOT_FILENAME_V4_4]
    if root.get("cycle_id") != prereg["cycle_id"] or report.get(
        "cycle_id"
    ) != prereg["cycle_id"]:
        raise _error("preregistration cycle identity mismatch")

    candidate_rows = _rebuild_candidate_rows_from_prereg_graph(artifacts, modules)
    candidate_rows_sha = modules.contract.semantic_sha256_v4_4(
        list(candidate_rows)
    )
    if candidate_rows_sha != prereg["candidate_rows_semantic_sha256"]:
        raise _error("preregistration candidate rows semantic SHA mismatch")

    strict_filename = (
        modules.prereg_core.V4_2_PREDECESSOR_PREFIX
        + modules.predecessor_bundle.STRICT_FULL_A_SOURCE_BINDING_FILENAME_V4_2
    )
    strict_source = copy.deepcopy(dict(artifacts[strict_filename]))
    backend = copy.deepcopy(dict(strict_source["backend_binding"]))
    expected_source = normalized_manifest["strict_source_expected"]
    expected_values = {
        "strict_source_binding_semantic_sha256": strict_source[
            "artifact_semantic_sha256"
        ],
        "snapshot_manifest_byte_sha256": backend["snapshot_manifest"]["sha256"],
        "pit_generation_manifest_byte_sha256": backend["pit_generation"][
            "manifest"
        ]["sha256"],
        "pit_membership_byte_sha256": backend["pit_generation"]["membership"][
            "sha256"
        ],
        "table_inventory_semantic_sha256": backend["table"]["inventory_sha256"],
        "full_a_scope_count": strict_source["expected_scope_count"],
        "full_a_scope_sha256": strict_source["full_a_scope_sha256"],
        "source_calendar_semantic_sha256": backend["calendar"]["semantic_sha256"],
        "recorded_latest_pointer_byte_sha256": strict_source[
            "latest_pointer_raw_evidence"
        ]["byte_sha256"],
        "recorded_components_byte_sha256": strict_source[
            "components_raw_evidence"
        ]["byte_sha256"],
    }
    if expected_source != expected_values:
        raise _error("manifest strict_source_expected differs from accepted graph")
    if (
        strict_source["cutoff"] != normalized_manifest["cutoff"]
        or strict_source["snapshot_id"] != normalized_manifest["snapshot_id"]
    ):
        raise _error("manifest cutoff/snapshot differs from accepted strict source")
    sessions = backend["calendar"]["open_sessions"]
    if (
        not isinstance(sessions, list)
        or len(sessions) < RESOURCE_CONTRACT["halo_session_count"] + 1
        or normalized_manifest["proof_output_start"]
        != sessions[RESOURCE_CONTRACT["halo_session_count"]]
    ):
        raise _error("proof_output_start must equal exact source calendar session[60]")
    if len(sessions) > RESOURCE_CONTRACT["source_session_count_max"]:
        raise _error("source session resource cap exceeded")

    snapshot_id = strict_source["snapshot_id"]
    snapshot_manifest_path = _assert_immutable_source_path(
        backend["snapshot_manifest"]["absolute_path"],
        expected=PROJECT_ROOT
        / "data"
        / "parquet"
        / "cn"
        / "_snapshots"
        / f"{snapshot_id}.json",
        label="snapshot manifest",
    )
    table_root = _assert_immutable_source_path(
        backend["table"]["absolute_root"],
        expected=PROJECT_ROOT
        / "data"
        / "parquet"
        / "cn"
        / "_snapshots"
        / snapshot_id
        / "table"
        / "bars",
        label="snapshot table root",
    )
    generation_id = backend["pit_generation"]["generation_id"]
    pit_parent = (
        PROJECT_ROOT
        / "data"
        / "parquet"
        / "cn"
        / "reference"
        / "_generations"
        / generation_id
    )
    pit_manifest_path = _assert_immutable_source_path(
        backend["pit_generation"]["manifest"]["absolute_path"],
        expected=pit_parent / "manifest.json",
        label="PIT generation manifest",
    )
    pit_membership_path = _assert_immutable_source_path(
        backend["pit_generation"]["membership"]["absolute_path"],
        expected=pit_parent / "stock_basic_membership.parquet",
        label="PIT membership",
    )
    return AcceptedPreregistration(
        readback=copy.deepcopy(dict(result)),
        artifacts={name: copy.deepcopy(dict(artifacts[name])) for name in exact_files},
        strict_source=strict_source,
        backend_binding=backend,
        calendar_sessions=tuple(sessions),
        candidate_rows=candidate_rows,
        snapshot_manifest_path=snapshot_manifest_path,
        pit_manifest_path=pit_manifest_path,
        pit_membership_path=pit_membership_path,
        table_root=table_root,
    )


def _open_fixed_private_root(root: Path) -> tuple[int, os.stat_result]:
    if tuple(root.parts[-len(ROOT_SUFFIX) :]) != ROOT_SUFFIX:
        raise _error("fixed publication root suffix mismatch")
    descriptor = os.open("/", _directory_flags())
    try:
        for component in root.parts[1:]:
            child = os.open(component, _directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or int(metadata.st_uid) != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise _error("fixed publication root must be current-owner mode 0700")
        return descriptor, metadata
    except OSError as exc:
        os.close(descriptor)
        raise _error(f"fixed publication root preflight failed: {exc}") from exc
    except Exception:
        os.close(descriptor)
        raise


def _require_exclusive_publication_capability(modules: LoadedModules) -> None:
    """Require Darwin's exclusive directory rename with no fallback."""

    if sys.platform != "darwin":
        raise _error("future strict publication requires Darwin renameatx_np")
    try:
        modules.private_io._require_exclusive_rename_support()
    except Exception as exc:
        raise _error(f"Darwin exclusive rename capability is unavailable: {exc}") from exc


def _publication_preflight(
    *,
    manifest: Mapping[str, Any],
    accepted: AcceptedPreregistration,
    modules: LoadedModules,
) -> PublicationPreflight:
    _require_exclusive_publication_capability(modules)
    root = PRODUCTION_PRIVATE_ROOT
    if not root.is_absolute():
        raise _error("fixed publication root must be absolute")
    prereg_path = Path(manifest["preregistration"]["bundle_path"])
    if root == prereg_path or root == prereg_path.parent:
        raise _error("strict publication root must be distinct from preregistration")
    root_fd, metadata = _open_fixed_private_root(root)
    try:
        cycle_id = manifest["cycle_id"]
        try:
            target = os.stat(cycle_id, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            target = None
        if target is not None:
            raise _error("fixed strict publication target already exists")
        current = os.fstat(root_fd)
        if _signature(metadata) != _signature(current):
            raise _error("fixed publication root changed during preflight")
        return PublicationPreflight(root, _signature(current), cycle_id)
    finally:
        os.close(root_fd)


def _semantic_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes_local(value)).hexdigest()


def _distribution_descriptor(distribution_name: str) -> dict[str, Any]:
    """Bind one frozen shadow inventory without interpreting RECORD paths."""

    metadata_module = importlib.import_module("importlib.metadata")
    try:
        distribution = metadata_module.distribution(distribution_name)
    except Exception as exc:
        raise _error(f"runtime distribution unavailable: {distribution_name}") from exc
    _require_trusted_distribution_base(distribution, distribution_name)
    if _ACTIVE_RUNTIME_SHADOW_INVENTORY is None:
        raise _error("runtime shadow inventory is unavailable")
    roots = RUNTIME_DISTRIBUTION_SHADOW_ROOTS.get(distribution_name)
    if roots is None:
        raise _error(f"runtime distribution is outside the exact six: {distribution_name}")
    rows = [
        copy.deepcopy(row)
        for row in _ACTIVE_RUNTIME_SHADOW_INVENTORY["files"]
        if any(
            row["relative_path"] == root
            or row["relative_path"].startswith(root + "/")
            for root in roots
        )
    ]
    native_rows = [
        copy.deepcopy(row)
        for row in rows
        if row["relative_path"].endswith((".so", ".dylib", ".pyd", ".dll"))
    ]
    if not rows or not any(
        row["relative_path"].endswith(".dist-info/RECORD") for row in rows
    ):
        raise _error(f"runtime distribution RECORD is missing: {distribution_name}")
    return {
        "name": distribution_name,
        "version": distribution.version,
        "distribution_file_count": len(rows),
        "distribution_inventory_sha256": _semantic_sha(rows),
        "native_binary_count": len(native_rows),
        "native_binary_inventory_sha256": _semantic_sha(native_rows),
    }


def _runtime_binding(
    *, contract: Any
) -> dict[str, Any]:
    """Collect the fixed CPython/data-stack/platform runtime identity."""

    platform_module = importlib.import_module("platform")
    executable = Path(os.path.realpath(sys.executable))
    executable_stable = _read_bound_regular_once(
        executable,
        label="CPython executable",
        max_bytes=256 * 1024 * 1024,
        expected_nlink=None,
    )
    if sys.byteorder != "little":
        raise _error("runtime byte order must be little endian")
    return contract.build_runtime_binding_v4_4(
        python_implementation=platform_module.python_implementation(),
        python_version=platform_module.python_version(),
        python_executable_byte_sha256=executable_stable.byte_sha256,
        platform_system=platform_module.system(),
        platform_release=platform_module.release(),
        machine=platform_module.machine(),
        byteorder=sys.byteorder,
        distributions=[
            _distribution_descriptor(name)
            for name in contract.RUNTIME_DISTRIBUTION_NAMES
        ],
        import_root_tree_sha256=_rehash_runtime_shadow()[
            "tree_semantic_sha256"
        ],
    )


_GIT_EXECUTABLE = Path("/usr/bin/git")
_GIT_FORBIDDEN_LOCAL_CONFIG = (
    r"^(include|includeif)\.|^core\.worktree$|^extensions\.partialclone$|"
    r"^core\.alternaterefs|^remote\..*\.promisor$"
)


def _git_environment() -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": "/var/empty",
        "TMPDIR": "/private/tmp",
        "XDG_CONFIG_HOME": "/var/empty",
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "http_proxy": "",
        "https_proxy": "",
        "all_proxy": "",
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
        "ALL_PROXY": "",
        "no_proxy": "*",
        "NO_PROXY": "*",
    }


def _invoke_git(
    repository: Path,
    operation: str,
    *values: str,
) -> bytes:
    allowed_returncodes = (0,)
    if operation == "show_toplevel" and not values:
        arguments = ("rev-parse", "--show-toplevel")
    elif operation == "absolute_git_dir" and not values:
        arguments = ("rev-parse", "--absolute-git-dir")
    elif operation == "object_dir" and not values:
        arguments = (
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            "objects",
        )
    elif operation == "object_format" and not values:
        arguments = ("rev-parse", "--show-object-format")
    elif operation == "replace_refs" and not values:
        arguments = ("for-each-ref", "--format=%(refname)", "refs/replace/")
    elif operation == "forbidden_config" and not values:
        arguments = (
            "config",
            "--local",
            "--name-only",
            "--get-regexp",
            _GIT_FORBIDDEN_LOCAL_CONFIG,
        )
        allowed_returncodes = (0, 1)
    elif operation in {"resolve_commit", "resolve_tree"} and len(values) == 1:
        suffix = "^{commit}" if operation == "resolve_commit" else "^{tree}"
        arguments = ("rev-parse", "--verify", f"{values[0]}{suffix}")
    elif operation in {"object_type", "object_size", "cat_blob"} and len(values) == 1:
        mode = {
            "object_type": "-t",
            "object_size": "-s",
            "cat_blob": "blob",
        }[operation]
        arguments = ("cat-file", mode, values[0])
    elif operation == "ls_tree" and len(values) == 2:
        arguments = (
            "ls-tree",
            "-z",
            "--full-name",
            values[0],
            "--",
            values[1],
        )
    else:
        raise _error("internal Git operation is not allowlisted")
    try:
        git_metadata = os.lstat(_GIT_EXECUTABLE)
    except OSError as exc:
        raise _error("fixed /usr/bin/git executable is unavailable") from exc
    if (
        stat.S_ISLNK(git_metadata.st_mode)
        or not stat.S_ISREG(git_metadata.st_mode)
        or _GIT_EXECUTABLE.resolve(strict=True) != _GIT_EXECUTABLE
        or stat.S_IMODE(git_metadata.st_mode) & 0o022
    ):
        raise _error("fixed /usr/bin/git executable identity is unsafe")
    command = [
        str(_GIT_EXECUTABLE),
        "--no-replace-objects",
        "-c",
        "core.useReplaceRefs=false",
        "-c",
        "extensions.partialClone=",
        "-c",
        "protocol.file.allow=never",
        "-c",
        "protocol.ext.allow=never",
        "-C",
        str(repository),
        *arguments,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_git_environment(),
            cwd="/",
            timeout=30,
            shell=False,
            close_fds=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _error(f"pinned Git object read failed: {repository}: {exc}") from exc
    if completed.returncode not in allowed_returncodes:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise _error(f"pinned Git object read failed: {repository}: {detail}")
    if completed.stderr:
        raise _error("pinned Git emitted unexpected stderr")
    return completed.stdout


def _git_read(repository: Path, operation: str, *values: str) -> bytes:
    return _invoke_git(repository, operation, *values)


def _real_owned_directory(path: Path, label: str) -> tuple[int, ...]:
    try:
        metadata = os.lstat(path)
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise _error(f"{label} is unavailable: {path}") from exc
    if (
        not path.is_absolute()
        or resolved != path
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or int(metadata.st_uid) != os.getuid()
    ):
        raise _error(f"{label} must be an owner real directory: {path}")
    return _signature(metadata)


def _reject_git_indirection_file(path: Path, label: str) -> tuple[int, ...] | None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _error(f"cannot inspect {label}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or int(metadata.st_uid) != os.getuid()
        or int(metadata.st_nlink) != 1
        or int(metadata.st_size) != 0
    ):
        raise _error(f"{label} must be absent or an empty owner regular file")
    stable = _read_bound_regular_once(
        path, label=label, max_bytes=1, expected_nlink=1
    )
    if stable.raw:
        raise _error(f"{label} must be empty")
    return stable.signature


def _validate_git_repository(repository: Path) -> dict[str, Any]:
    root_signature = _real_owned_directory(repository, "pinned Git root")
    git_dir = repository / ".git"
    git_signature = _real_owned_directory(git_dir, "pinned Git directory")
    config_before = _read_bound_regular_once(
        git_dir / "config",
        label="pinned Git local config",
        max_bytes=16 * 1024 * 1024,
        expected_nlink=1,
    )
    if int(config_before.signature[3]) != os.getuid():
        raise _error("pinned Git local config must be current-owner")
    if _git_read(repository, "show_toplevel") != (
        str(repository).encode("utf-8") + b"\n"
    ):
        raise _error("pinned Git top-level root mismatch")
    if _git_read(repository, "absolute_git_dir") != (
        str(git_dir).encode("utf-8") + b"\n"
    ):
        raise _error("pinned Git directory resolution mismatch")
    object_dir = git_dir / "objects"
    object_signature = _real_owned_directory(
        object_dir, "pinned Git object directory"
    )
    if _git_read(repository, "object_dir") != (
        str(object_dir).encode("utf-8") + b"\n"
    ):
        raise _error("pinned Git object directory resolution mismatch")
    if _git_read(repository, "object_format") != b"sha1\n":
        raise _error("pinned Git object format must be exact sha1")
    if _git_read(repository, "replace_refs") != b"":
        raise _error("pinned Git replace refs are forbidden")
    if _git_read(repository, "forbidden_config") != b"":
        raise _error("pinned Git local indirection/promisor config is forbidden")
    try:
        os.lstat(git_dir / "config.worktree")
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise _error("cannot inspect pinned Git config.worktree") from exc
    else:
        raise _error("pinned Git config.worktree is forbidden")
    alternates = _reject_git_indirection_file(
        object_dir / "info" / "alternates", "Git alternates"
    )
    http_alternates = _reject_git_indirection_file(
        object_dir / "info" / "http-alternates", "Git HTTP alternates"
    )
    pack_dir = object_dir / "pack"
    try:
        with os.scandir(pack_dir) as entries:
            pack_entries = tuple(entries)
    except FileNotFoundError:
        pack_entries = ()
    except OSError as exc:
        raise _error("cannot inspect pinned Git pack directory") from exc
    if any(entry.name.endswith(".promisor") for entry in pack_entries):
        raise _error("pinned Git promisor pack metadata is forbidden")
    config_after = _read_bound_regular_once(
        git_dir / "config",
        label="pinned Git local config after checks",
        max_bytes=16 * 1024 * 1024,
        expected_nlink=1,
    )
    if (
        config_after.signature != config_before.signature
        or config_after.raw != config_before.raw
    ):
        raise _error("pinned Git local config drifted during validation")
    return {
        "root": root_signature,
        "git_dir": git_signature,
        "object_dir": object_signature,
        "alternates": alternates,
        "http_alternates": http_alternates,
        "config_signature": config_before.signature,
        "config_byte_sha256": config_before.byte_sha256,
    }


def _verify_git_blob_binding(
    *,
    repository: Path,
    commit: str,
    tree_path: str,
    blob_oid: str,
    raw_sha256: str,
    root_tree_oid: str,
    expected_size: int,
) -> bytes:
    for oid_value, label in (
        (commit, "commit"),
        (blob_oid, "blob"),
        (root_tree_oid, "root tree"),
    ):
        if type(oid_value) is not str or _OID_RE.fullmatch(oid_value) is None:
            raise _error(f"pinned Git {label} OID must be exact lowercase SHA-1")
    if (
        type(tree_path) is not str
        or tree_path.startswith("/")
        or any(part in {"", ".", ".."} for part in Path(tree_path).parts)
        or type(expected_size) is not int
        or expected_size <= 0
    ):
        raise _error("pinned Git path/size binding is malformed")
    before = _validate_git_repository(repository)
    if _git_read(repository, "resolve_commit", commit) != (
        commit.encode("ascii") + b"\n"
    ):
        raise _error("pinned Git commit resolution mismatch")
    if _git_read(repository, "object_type", commit) != b"commit\n":
        raise _error("pinned Git commit object type mismatch")
    if _git_read(repository, "resolve_tree", commit) != (
        root_tree_oid.encode("ascii") + b"\n"
    ) or _git_read(repository, "object_type", root_tree_oid) != b"tree\n":
        raise _error("pinned Git root tree binding mismatch")
    expected_tree_row = (
        b"100644 blob "
        + blob_oid.encode("ascii")
        + b"\t"
        + tree_path.encode("utf-8")
        + b"\0"
    )
    if _git_read(repository, "ls_tree", commit, tree_path) != expected_tree_row:
        raise _error("pinned Git tree/blob/mode binding mismatch")
    if _git_read(repository, "object_type", blob_oid) != b"blob\n":
        raise _error("pinned Git source object is not a blob")
    if _git_read(repository, "object_size", blob_oid) != (
        str(expected_size).encode("ascii") + b"\n"
    ):
        raise _error("pinned Git blob size mismatch")
    raw = _git_read(repository, "cat_blob", blob_oid)
    if (
        len(raw) != expected_size
        or hashlib.sha256(raw).hexdigest() != raw_sha256
        or hashlib.sha1(
            b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
        ).hexdigest()
        != blob_oid
        or raw.startswith(b"version https://git-lfs.github.com/spec/v1")
    ):
        raise _error("pinned Git blob raw SHA-256 mismatch")
    if _validate_git_repository(repository) != before:
        raise _error("pinned Git repository identity drifted during object reads")
    return raw


def _aquant_definition_identity_sha256(
    tree: ast.Module, *, pinned_commit: str
) -> str:
    template = (
        "(close - ts_min(close, {w})) / "
        "(ts_max(close, {w}) - ts_min(close, {w}))"
    )
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "generate_default_candidates"
    ]
    if len(functions) != 1 or not isinstance(functions[0], ast.FunctionDef):
        raise _error("pinned A_quant generator function is not unique")
    function = functions[0]
    add_definitions = [
        node
        for node in function.body
        if isinstance(node, ast.FunctionDef) and node.name == "add"
    ]
    if len(add_definitions) != 1:
        raise _error("pinned A_quant local add helper is not unique")
    add_definition = add_definitions[0]
    if (
        [item.arg for item in add_definition.args.args]
        != ["name", "expression", "family", "rationale", "factor_type"]
        or len(add_definition.args.defaults) != 1
        or not isinstance(add_definition.args.defaults[0], ast.Constant)
        or add_definition.args.defaults[0].value != "alpha"
    ):
        raise _error("pinned A_quant add helper/default factor type drifted")
    all_pos_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "pos"
    ]
    if (
        len(all_pos_assignments) != 1
        or all_pos_assignments[0] not in function.body
        or not isinstance(all_pos_assignments[0].value, ast.Constant)
        or all_pos_assignments[0].value.value != template
    ):
        raise _error("pinned A_quant range template AST is not unique")
    expected_call = ast.parse(
        "add(f'alpha_range_position_momentum_{window}d', "
        "f'cs_rank({pos.format(w=window)})', 'price_momentum', "
        "'Buys stocks high in their recent range.')",
        mode="exec",
    ).body[0]
    assert isinstance(expected_call, ast.Expr)
    expected_dump = ast.dump(
        expected_call.value, annotate_fields=True, include_attributes=False
    )
    all_momentum_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "add"
        and node.args
        and isinstance(node.args[0], ast.JoinedStr)
        and any(
            isinstance(item, ast.Constant)
            and item.value == "alpha_range_position_momentum_"
            for item in node.args[0].values
        )
    ]
    if len(all_momentum_calls) != 1:
        raise _error("pinned A_quant momentum add call is not unique module-wide")
    matching_loops: list[ast.For] = []
    for node in function.body:
        if not isinstance(node, ast.For):
            continue
        if (
            not isinstance(node.target, ast.Name)
            or node.target.id != "window"
            or not isinstance(node.iter, ast.List)
            or [
                item.value
                if isinstance(item, ast.Constant) and type(item.value) is int
                else None
                for item in node.iter.elts
            ]
            != [20, 60, 120]
        ):
            continue
        matches = [
            child
            for child in node.body
            if isinstance(child, ast.Expr)
            and isinstance(child.value, ast.Call)
            and ast.dump(
                child.value,
                annotate_fields=True,
                include_attributes=False,
            )
            == expected_dump
        ]
        if len(matches) == 1:
            matching_loops.append(node)
    if (
        len(matching_loops) != 1
        or all_momentum_calls[0]
        not in [
            item.value
            for item in matching_loops[0].body
            if isinstance(item, ast.Expr) and isinstance(item.value, ast.Call)
        ]
    ):
        raise _error("pinned A_quant momentum generator AST is not unique")
    expression = f"cs_rank({template.format(w=20)})"
    expected_expression = (
        "cs_rank((close - ts_min(close, 20)) / "
        "(ts_max(close, 20) - ts_min(close, 20)))"
    )
    if expression != expected_expression:
        raise _error("pinned A_quant extracted expression mismatch")
    payload = {
        "version": "aquant-source-definition.v1",
        "pinned_commit": pinned_commit,
        "name": f"alpha_range_position_momentum_{20}d",
        "expression": expression,
        "factor_type": add_definition.args.defaults[0].value,
        "source_family": all_momentum_calls[0].args[2].value,
        "rationale": all_momentum_calls[0].args[3].value,
        "direction": 1.0 if not expression.startswith("cs_rank(-") else -1.0,
        "direction_origin": "expression_signed_ast",
    }
    return _semantic_sha(payload)


def _myquant_lambda_node(
    tree: ast.Module, source_factor: str, expected_method: str
) -> ast.Lambda:
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Alpha158"
    ]
    if len(classes) != 1:
        raise _error("pinned myQuant Alpha158 class is not unique")
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == expected_method
    ]
    if len(methods) != 1:
        raise _error(f"pinned myQuant method is not unique: {expected_method}")
    stored_targets = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Store)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == source_factor
    ]
    assignments: list[ast.Assign] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == source_factor
            ):
                assignments.append(node)
    if (
        len(stored_targets) != 1
        or len(assignments) != 1
        or assignments[0] not in methods[0].body
    ):
        raise _error(f"pinned myQuant factor lambda is not unique: {source_factor}")
    assignment = assignments[0]
    if len(assignment.targets) != 1:
        raise _error(f"pinned myQuant factor target is not exact: {source_factor}")
    target = assignment.targets[0]
    if not (
        isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Attribute)
        and isinstance(target.value.value, ast.Name)
        and target.value.value.id == "self"
        and target.value.attr == "factor_functions"
        and isinstance(assignment.value, ast.Lambda)
    ):
        raise _error(f"pinned myQuant factor assignment drifted: {source_factor}")
    return assignment.value


def _myquant_lambda_ast_sha256(
    tree: ast.Module, source_factor: str, expected_method: str
) -> str:
    lambda_node = _myquant_lambda_node(tree, source_factor, expected_method)
    dumped = ast.dump(
        lambda_node, annotate_fields=True, include_attributes=False
    ).encode("utf-8")
    return hashlib.sha256(dumped).hexdigest()


class _CanonicalProgramBuilder:
    """Emit sequential nodes with only exact structural common subexpressions."""

    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self._by_structure: dict[bytes, str] = {}

    def add(
        self,
        opcode: str,
        inputs: Sequence[str] = (),
        parameters: Mapping[str, Any] | None = None,
    ) -> str:
        structure = {
            "opcode": opcode,
            "inputs": list(inputs),
            "parameters": copy.deepcopy(dict(parameters or {})),
        }
        key = json.dumps(
            structure,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        existing = self._by_structure.get(key)
        if existing is not None:
            return existing
        node_id = f"n{len(self.nodes):03d}"
        node = {"node_id": node_id, **structure}
        self.nodes.append(node)
        self._by_structure[key] = node_id
        return node_id


def _aquant_expression_from_authenticated_generator(
    tree: ast.Module, *, pinned_commit: str
) -> tuple[str, str]:
    identity = _aquant_definition_identity_sha256(
        tree, pinned_commit=pinned_commit
    )
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "pos"
        and isinstance(node.value, ast.Constant)
        and type(node.value.value) is str
    ]
    if len(assignments) != 1:
        raise _error("authenticated A_quant template cannot be rederived")
    try:
        instantiated = assignments[0].value.value.format(w=20)
    except (KeyError, IndexError, ValueError) as exc:
        raise _error("authenticated A_quant template instantiation failed") from exc
    expression = f"cs_rank({instantiated})"
    return identity, expression


def _lower_aquant_expression(
    expression: str, *, adapter: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise _error("authenticated A_quant expression is not closed Python syntax") from exc
    mapping = dict(
        zip(
            adapter["source_facing_fields"],
            adapter["canonical_inputs"],
            strict=True,
        )
    )
    builder = _CanonicalProgramBuilder()

    def lower(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            if node.id not in mapping:
                raise _error("A_quant expression references an unbound source field")
            return builder.add(
                "source",
                parameters={
                    "source_field": node.id,
                    "canonical_input": mapping[node.id],
                },
            )
        if isinstance(node, ast.BinOp) and type(node.op) in {
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
        }:
            opcode = {
                ast.Add: "add",
                ast.Sub: "subtract",
                ast.Mult: "multiply",
                ast.Div: "divide",
            }[type(node.op)]
            return builder.add(opcode, (lower(node.left), lower(node.right)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.keywords:
                raise _error("A_quant expression calls cannot use keyword arguments")
            if node.func.id in {"ts_min", "ts_max"}:
                if (
                    len(node.args) != 2
                    or not isinstance(node.args[1], ast.Constant)
                    or type(node.args[1].value) is not int
                    or node.args[1].value != 20
                ):
                    raise _error("A_quant rolling window must be instantiated at 20")
                return builder.add(
                    "rolling_min" if node.func.id == "ts_min" else "rolling_max",
                    (lower(node.args[0]),),
                    {"window": 20, "min_periods": 1},
                )
            if node.func.id == "cs_rank":
                if len(node.args) != 1:
                    raise _error("A_quant cs_rank arity drifted")
                return builder.add(
                    "cross_section_rank",
                    (lower(node.args[0]),),
                    {
                        "axis": "symbols",
                        "method": "average",
                        "na_option": "keep",
                        "pct": True,
                        "ascending": True,
                    },
                )
        raise _error(
            "A_quant expression contains an opcode outside the closed translator"
        )

    output = lower(parsed.body)
    return copy.deepcopy(builder.nodes), output


def _validate_exact_lambda_signature(lambda_node: ast.Lambda, source_factor: str) -> None:
    arguments = lambda_node.args
    if (
        [argument.arg for argument in arguments.posonlyargs] != []
        or [argument.arg for argument in arguments.args] != ["df"]
        or arguments.vararg is not None
        or arguments.kwonlyargs
        or arguments.kw_defaults
        or arguments.kwarg is not None
        or arguments.defaults
    ):
        raise _error(f"pinned myQuant lambda signature drifted: {source_factor}")


def _lower_myquant_lambda(
    lambda_node: ast.Lambda, *, adapter: Mapping[str, Any], source_factor: str
) -> tuple[list[dict[str, Any]], str]:
    _validate_exact_lambda_signature(lambda_node, source_factor)
    source_mapping = dict(
        zip(
            adapter["source_facing_fields"],
            adapter["canonical_inputs"],
            strict=True,
        )
    )
    builder = _CanonicalProgramBuilder()

    def source(node: ast.AST) -> str | None:
        if not (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "df"
            and isinstance(node.slice, ast.Constant)
            and type(node.slice.value) is str
            and isinstance(node.ctx, ast.Load)
        ):
            return None
        field = node.slice.value
        if field not in source_mapping:
            raise _error(
                f"pinned myQuant lambda uses an unbound source field: {source_factor}"
            )
        return builder.add(
            "source",
            parameters={
                "source_field": field,
                "canonical_input": source_mapping[field],
            },
        )

    def exact_rolling_call(node: ast.AST) -> tuple[ast.AST, int] | None:
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "rolling"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Constant)
            and type(node.args[0].value) is int
            and node.args[0].value > 0
        ):
            return None
        return node.func.value, node.args[0].value

    def lower(node: ast.AST) -> str:
        source_id = source(node)
        if source_id is not None:
            return source_id
        if isinstance(node, ast.Constant):
            if type(node.value) is float and node.value == 1e-9:
                return builder.add(
                    "constant", parameters={"float64_be_hex": "3e112e0be826d695"}
                )
            raise _error("pinned myQuant lambda contains a forbidden constant")
        if isinstance(node, ast.BinOp) and type(node.op) in {
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
        }:
            opcode = {
                ast.Add: "add",
                ast.Sub: "subtract",
                ast.Mult: "multiply",
                ast.Div: "divide",
            }[type(node.op)]
            return builder.add(opcode, (lower(node.left), lower(node.right)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            method = node.func.attr
            receiver = node.func.value
            if method == "shift":
                if (
                    len(node.args) != 1
                    or node.keywords
                    or not isinstance(node.args[0], ast.Constant)
                    or type(node.args[0].value) is not int
                    or node.args[0].value != 1
                ):
                    raise _error("myQuant shift must be exact shift(1)")
                return builder.add("shift", (lower(receiver),), {"periods": 1})
            if method == "diff":
                if node.args or node.keywords:
                    raise _error("myQuant diff must use the exact omitted period=1")
                base = lower(receiver)
                shifted = builder.add("shift", (base,), {"periods": 1})
                return builder.add("subtract", (base, shifted))
            if method == "pct_change":
                if node.args or node.keywords:
                    raise _error(
                        "myQuant pct_change must omit only period=1/fill_method=None"
                    )
                base = lower(receiver)
                shifted = builder.add("shift", (base,), {"periods": 1})
                divided = builder.add("divide", (base, shifted))
                one = builder.add(
                    "constant", parameters={"float64_be_hex": "3ff0000000000000"}
                )
                return builder.add("subtract", (divided, one))
            if method == "apply":
                if (
                    len(node.args) != 1
                    or node.keywords
                    or not isinstance(node.args[0], ast.Attribute)
                    or not isinstance(node.args[0].value, ast.Name)
                    or node.args[0].value.id != "np"
                    or node.args[0].attr != "sign"
                ):
                    raise _error("myQuant apply is restricted to exact np.sign")
                return builder.add("sign", (lower(receiver),))
            if method == "abs":
                if node.args or node.keywords:
                    raise _error("myQuant abs must not receive arguments")
                return builder.add("absolute", (lower(receiver),))
            if method in {"min", "max", "mean", "std"}:
                rolling = exact_rolling_call(receiver)
                if rolling is None or node.args or node.keywords:
                    raise _error(
                        "myQuant rolling aggregation must use exact omitted defaults"
                    )
                base_node, window = rolling
                parameters: dict[str, Any] = {
                    "window": window,
                    "min_periods": window,
                }
                if method == "std":
                    parameters["ddof"] = 1
                return builder.add(
                    {
                        "min": "rolling_min",
                        "max": "rolling_max",
                        "mean": "rolling_mean",
                        "std": "rolling_std",
                    }[method],
                    (lower(base_node),),
                    parameters,
                )
        raise _error(
            f"pinned myQuant lambda contains a forbidden AST/API node: {source_factor}"
        )

    output = lower(lambda_node.body)
    return copy.deepcopy(builder.nodes), output


def _seal_derived_operator_program_set(
    *,
    contract: Any,
    rows: Sequence[Mapping[str, Any]],
    adapters: Sequence[Mapping[str, Any]],
    derived_nodes: Sequence[tuple[Sequence[Mapping[str, Any]], str]],
) -> dict[str, Any]:
    programs: list[dict[str, Any]] = []
    for row, adapter, (nodes, output_node_id) in zip(
        rows, adapters, derived_nodes, strict=True
    ):
        program = {
            "order": row["order"],
            "name": row["name"],
            "direction": float(row["direction"]),
            "definition_identity_sha256": row["definition_identity_sha256"],
            "source_repository": row["source_repository"],
            "source_commit": row["source_commit"],
            "source_tree_oid": row["source_tree_oid"],
            "source_relative_path": row["source_relative_path"],
            "source_blob_oid": row["source_blob_oid"],
            "source_raw_sha256": row["source_raw_sha256"],
            "source_ast_sha256": row["source_ast_sha256"],
            "field_semantics_sha256": row["field_semantics_sha256"],
            "field_adapter": copy.deepcopy(dict(adapter)),
            "nodes": [copy.deepcopy(dict(node)) for node in nodes],
            "output_node_id": output_node_id,
        }
        program["program_semantic_sha256"] = contract.semantic_sha256_v4_4(
            program
        )
        if program["program_semantic_sha256"] != row["operator_program_sha256"]:
            raise _error(f"derived operator program SHA mismatch: {row['name']}")
        programs.append(program)
    result = {
        "schema_version": contract.OPERATOR_PROGRAM_SET_SCHEMA_VERSION,
        "protocol_version": contract.PROTOCOL_VERSION,
        "evidence_contract_version": contract.EVIDENCE_CONTRACT_VERSION,
        "execution_semantics": copy.deepcopy(contract.OPERATOR_EXECUTION_SEMANTICS),
        "candidate_count": len(programs),
        "candidates": programs,
    }
    result["artifact_semantic_sha256"] = contract.semantic_sha256_v4_4(result)
    if any(
        row["operator_program_set_sha256"]
        != result["artifact_semantic_sha256"]
        for row in rows
    ):
        raise _error("derived operator program-set SHA mismatch")
    return contract.validate_operator_program_set_v4_4(result)


def _verify_pinned_source_definitions(
    *, manifest: Mapping[str, Any], contract: Any
) -> dict[str, Any]:
    """Recompute source bindings and mechanically lower them to canonical IR."""

    rows = tuple(manifest["source_definition_bindings"])
    if rows != tuple(contract.SOURCE_DEFINITION_BINDINGS):
        raise _error("source definition bindings differ from the loaded fixed oracle")
    for index, (row, semantics) in enumerate(
        zip(rows, contract.FIELD_SEMANTICS, strict=True), start=1
    ):
        if (
            contract.semantic_sha256_v4_4(semantics)
            != row["field_semantics_sha256"]
        ):
            raise _error(f"source definition {index} field semantics SHA mismatch")

    aquant = rows[0]
    aquant_raw = _verify_git_blob_binding(
        repository=PROJECT_ROOT.parent,
        commit=aquant["source_commit"],
        tree_path=aquant["source_relative_path"],
        blob_oid=aquant["source_blob_oid"],
        raw_sha256=aquant["source_raw_sha256"],
        root_tree_oid=aquant["source_tree_oid"],
        expected_size=26_431,
    )
    try:
        aquant_tree = ast.parse(aquant_raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise _error("pinned A_quant blob is not valid UTF-8 Python") from exc
    aquant_identity, aquant_expression = (
        _aquant_expression_from_authenticated_generator(
            aquant_tree, pinned_commit=aquant["source_commit"]
        )
    )
    if (
        aquant_identity != aquant["source_ast_sha256"]
        or aquant_identity != aquant["definition_identity_sha256"]
    ):
        raise _error("A_quant legacy AST/definition identity binding mismatch")

    myquant_rows = rows[1:]
    first = myquant_rows[0]
    if any(
        row[key] != first[key]
        for row in myquant_rows[1:]
        for key in (
            "source_repository",
            "source_commit",
            "source_relative_path",
            "source_blob_oid",
            "source_raw_sha256",
        )
    ):
        raise _error("myQuant exact-four pinned source identity is inconsistent")
    myquant_raw = _verify_git_blob_binding(
        repository=PROJECT_ROOT,
        commit=first["source_commit"],
        tree_path=first["source_relative_path"],
        blob_oid=first["source_blob_oid"],
        raw_sha256=first["source_raw_sha256"],
        root_tree_oid=first["source_tree_oid"],
        expected_size=22_768,
    )
    try:
        myquant_tree = ast.parse(myquant_raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise _error("pinned myQuant blob is not valid UTF-8 Python") from exc
    source_factors = (
        ("OVERNIGHT_GAP_20D", "_register_volatility_regime_factors"),
        ("VOL_RATIO_10_60", "_register_volatility_regime_factors"),
        ("PRICE_VOL_CONSISTENCY_20D", "_register_quality_factors"),
        ("VOL_OF_VOL_20D", "_register_volatility_regime_factors"),
    )
    derived_nodes: list[tuple[list[dict[str, Any]], str]] = [
        _lower_aquant_expression(
            aquant_expression, adapter=contract.FIELD_SEMANTICS[0]
        )
    ]
    for row, adapter, (source_factor, expected_method) in zip(
        myquant_rows, contract.FIELD_SEMANTICS[1:], source_factors, strict=True
    ):
        lambda_node = _myquant_lambda_node(
            myquant_tree, source_factor, expected_method
        )
        dumped = ast.dump(
            lambda_node, annotate_fields=True, include_attributes=False
        ).encode("utf-8")
        if hashlib.sha256(dumped).hexdigest() != row["source_ast_sha256"]:
            raise _error(f"pinned myQuant source AST SHA mismatch: {source_factor}")
        derived_nodes.append(
            _lower_myquant_lambda(
                lambda_node, adapter=adapter, source_factor=source_factor
            )
        )
    return _seal_derived_operator_program_set(
        contract=contract,
        rows=rows,
        adapters=contract.FIELD_SEMANTICS,
        derived_nodes=derived_nodes,
    )


def _collect_fixed_bindings(
    *, manifest: Mapping[str, Any], stable_manifest: StableBytes, modules: LoadedModules
) -> FrozenBindings:
    code_rows: list[dict[str, Any]] = []
    code_files: list[StableBytes] = []
    for row in manifest["code_binding_set"]:
        relative = row["relative_path"]
        stable = _read_bound_regular_once(
            PROJECT_ROOT / relative,
            label=f"fixed code binding {relative}",
            max_bytes=16 * 1024 * 1024,
            expected_sha256=row["byte_sha256"],
        )
        code_files.append(stable)
        code_rows.append(
            {
                "relative_path": relative,
                "byte_sha256": stable.byte_sha256,
            }
        )
    controls: list[tuple[str, StableBytes]] = []
    expected_controls = manifest["protected_control_expected_sha256"]
    for name, relative in PROTECTED_CONTROL_RELATIVE_PATHS:
        controls.append(
            (
                name,
                _read_bound_regular_once(
                    PROJECT_ROOT / relative,
                    label=f"protected control {name}",
                    max_bytes=64 * 1024 * 1024,
                    expected_sha256=expected_controls[name],
                ),
            )
        )
    runtime = _runtime_binding(contract=modules.contract)
    if runtime["artifact_semantic_sha256"] != manifest[
        "runtime_binding_expected_semantic_sha256"
    ]:
        raise _error("runtime binding semantic SHA mismatch")
    return FrozenBindings(
        code=tuple(code_rows),
        code_files=tuple(code_files),
        protected=tuple(controls),
        runtime=runtime,
    )


def _revalidate_fixed_bindings(
    *, frozen: FrozenBindings, manifest: Mapping[str, Any], modules: LoadedModules
) -> None:
    for expected in frozen.code_files:
        current = _read_bound_regular_once(
            expected.path,
            label=f"locked code revalidation {expected.path.name}",
            max_bytes=max(len(expected.raw), 1),
            expected_sha256=expected.byte_sha256,
        )
        if current.signature != expected.signature or current.raw != expected.raw:
            raise _error(f"fixed code changed before commit: {expected.path}")
    for name, expected in frozen.protected:
        current = _read_bound_regular_once(
            expected.path,
            label=f"locked protected-control revalidation {name}",
            max_bytes=max(len(expected.raw), 1),
            expected_sha256=expected.byte_sha256,
        )
        if current.signature != expected.signature or current.raw != expected.raw:
            raise _error(f"protected control changed before commit: {name}")
    runtime = _runtime_binding(contract=modules.contract)
    if _canonical_bytes_local(runtime) != _canonical_bytes_local(frozen.runtime):
        raise _error("runtime binding changed before commit")


def _postcommit_protected_diagnostics(
    frozen: FrozenBindings,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, expected in frozen.protected:
        try:
            current = _read_bound_regular_once(
                expected.path,
                label=f"postcommit protected-control diagnostic {name}",
                max_bytes=max(len(expected.raw), 1),
            )
            unchanged = (
                current.byte_sha256 == expected.byte_sha256
                and current.signature == expected.signature
            )
            actual_sha = current.byte_sha256
        except Exception as exc:  # diagnostic only after successful commit
            unchanged = False
            actual_sha = None
            rows.append(
                {
                    "name": name,
                    "expected_byte_sha256": expected.byte_sha256,
                    "actual_byte_sha256": actual_sha,
                    "unchanged": unchanged,
                    "diagnostic_error": str(exc),
                }
            )
            continue
        rows.append(
            {
                "name": name,
                "expected_byte_sha256": expected.byte_sha256,
                "actual_byte_sha256": actual_sha,
                "unchanged": unchanged,
                "diagnostic_error": None,
            }
        )
    return {
        "scope": "POSTCOMMIT_DIAGNOSTIC_ONLY",
        "external_maintenance_serialized": False,
        "rows": rows,
    }


def _lazy_data_stack() -> DataStack:
    """Import NumPy/Pandas/PyArrow/evaluator only at the data-pass stage."""

    np = importlib.import_module("numpy")
    pd = importlib.import_module("pandas")
    pa = importlib.import_module("pyarrow")
    pq = importlib.import_module("pyarrow.parquet")
    pc = importlib.import_module("pyarrow.compute")
    evaluator = importlib.import_module(
        "quant_investor.factors.governance_future_strict_exact_five_eval_v4_4"
    )
    return DataStack(np=np, pd=pd, pa=pa, pq=pq, pc=pc, evaluator=evaluator)


def _strict_json_stream(file_object: BinaryIO, label: str) -> dict[str, Any]:
    raw = file_object.read()
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise _error(f"{label} must be strict UTF-8 JSON") from exc
    duplicate = False

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        result: dict[str, Any] = {}
        for key, item in values:
            if key in result:
                duplicate = True
            result[key] = item
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise _error(f"{label} is invalid strict JSON: {exc}") from exc
    if duplicate or not isinstance(value, Mapping):
        raise _error(f"{label} must be one object without duplicate fields")
    return copy.deepcopy(dict(value))


def _stream_hash_then_parse(
    path: Path,
    *,
    label: str,
    expected_sha256: str,
    expected_size: int,
    max_bytes: int,
    expected_nlink: int | None,
    parser: Callable[[BinaryIO], Any],
) -> tuple[Any, StableBytes]:
    """Hash, seek, and parse one exact opened file object, then re-stat it."""

    if expected_size < 0 or expected_size > max_bytes:
        raise _error(f"{label} expected size violates the resource cap")
    parent_fd, leaf = _open_parent_anchored(path)
    descriptor = -1
    file_object: BinaryIO | None = None
    try:
        descriptor = os.open(leaf, _file_flags(), dir_fd=parent_fd)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or int(before.st_uid) != os.getuid():
            raise _error(f"{label} must be a current-owner regular file")
        if expected_nlink is not None and int(before.st_nlink) != expected_nlink:
            raise _error(f"{label} hard-link count mismatch")
        if int(before.st_size) != expected_size or int(before.st_size) > max_bytes:
            raise _error(f"{label} byte size mismatch")
        _path_identity_at(parent_fd, leaf, before, label)
        file_object = os.fdopen(descriptor, "rb", closefd=True)
        descriptor = -1
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = file_object.read(1_048_576)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise _error(f"{label} exceeded its byte cap while hashing")
            digest.update(chunk)
        if total != expected_size or digest.hexdigest() != expected_sha256:
            raise _error(f"{label} byte SHA/size mismatch")
        file_object.seek(0, os.SEEK_SET)
        parsed = parser(file_object)
        after = os.fstat(file_object.fileno())
        _path_identity_at(parent_fd, leaf, after, label)
        if _signature(before) != _signature(after):
            raise _error(f"{label} changed between hash and parse")
        return (
            parsed,
            StableBytes(path, b"", expected_sha256, _signature(after)),
        )
    except OSError as exc:
        raise _error(f"{label} anchored hash/parse failed: {exc}") from exc
    finally:
        if file_object is not None:
            file_object.close()
        elif descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def _open_anchored_directory(path: Path, *, label: str) -> tuple[int, os.stat_result]:
    parent_fd, leaf = _open_parent_anchored(path)
    descriptor = -1
    try:
        descriptor = os.open(leaf, _directory_flags(), dir_fd=parent_fd)
        metadata = os.fstat(descriptor)
        current = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or not _same_object(metadata, current)
        ):
            raise _error(f"{label} must be an anchored non-symlink directory")
        return descriptor, metadata
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        raise _error(f"{label} anchored directory open failed: {exc}") from exc
    finally:
        os.close(parent_fd)


def _scan_regular_tree(path: Path, *, label: str) -> tuple[list[dict[str, Any]], tuple[int, ...]]:
    root_fd, root_metadata = _open_anchored_directory(path, label=label)
    rows: list[dict[str, Any]] = []

    def walk(directory_fd: int, prefix: str) -> None:
        try:
            names = sorted(os.listdir(directory_fd))
        except OSError as exc:
            raise _error(f"{label} directory enumeration failed") from exc
        for name in names:
            if name in {"", ".", ".."} or "/" in name:
                raise _error(f"{label} contains an unsafe member name")
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            relative = f"{prefix}/{name}" if prefix else name
            if stat.S_ISDIR(metadata.st_mode):
                child = os.open(name, _directory_flags(), dir_fd=directory_fd)
                try:
                    if not _same_object(
                        metadata,
                        os.stat(name, dir_fd=directory_fd, follow_symlinks=False),
                    ):
                        raise _error(f"{label} directory identity changed: {relative}")
                    walk(child, relative)
                finally:
                    os.close(child)
            elif stat.S_ISREG(metadata.st_mode):
                rows.append(
                    {
                        "relative_path": relative,
                        "size_bytes": int(metadata.st_size),
                        "hard_link_count": int(metadata.st_nlink),
                    }
                )
            else:
                raise _error(f"{label} contains non-regular member: {relative}")

    try:
        walk(root_fd, "")
        after = os.fstat(root_fd)
        if _signature(root_metadata) != _signature(after):
            raise _error(f"{label} root changed during inventory")
        return rows, _signature(after)
    finally:
        os.close(root_fd)


def _validated_table_inventory(
    *, accepted: AcceptedPreregistration
) -> tuple[dict[str, Any], ...]:
    expected = accepted.backend_binding["table"]["parquet_inventory"]
    if (
        not isinstance(expected, list)
        or len(expected) == 0
        or len(expected) > RESOURCE_CONTRACT["table_member_count_max"]
    ):
        raise _error("recorded table member count violates the resource contract")
    actual, _root_signature = _scan_regular_tree(
        accepted.table_root, label="strict snapshot table"
    )
    if [row["relative_path"] for row in actual] != [
        row["relative_path"] for row in expected
    ]:
        raise _error("strict table inventory path set/order mismatch")
    total_bytes = 0
    normalized: list[dict[str, Any]] = []
    for actual_row, expected_row in zip(actual, expected, strict=True):
        if set(expected_row) != {
            "relative_path",
            "size_bytes",
            "sha256",
            "hard_link_count",
            "dataset_member",
        }:
            raise _error("recorded strict table inventory fields are not exact")
        if (
            actual_row["size_bytes"] != expected_row["size_bytes"]
            or actual_row["hard_link_count"] != expected_row["hard_link_count"]
            or type(expected_row["dataset_member"]) is not bool
        ):
            raise _error(
                f"strict table inventory metadata mismatch: {expected_row['relative_path']}"
            )
        _sha256(expected_row["sha256"], "recorded table member SHA")
        if expected_row["size_bytes"] > RESOURCE_CONTRACT["table_member_max_bytes"]:
            raise _error("strict table member byte cap exceeded")
        total_bytes += expected_row["size_bytes"]
        normalized.append(copy.deepcopy(dict(expected_row)))
    if total_bytes > RESOURCE_CONTRACT["table_total_max_bytes"]:
        raise _error("strict table total byte cap exceeded")
    return tuple(normalized)


def _parse_physical_table(file_object: BinaryIO, *, stack: DataStack) -> Any:
    parquet = stack.pq.ParquetFile(file_object)
    schema = parquet.schema_arrow
    projection = ("trade_date", "ts_code", "open", "close", "vol", "adj_close")
    identifier_types = (stack.pa.string(), stack.pa.large_string())
    for name in projection:
        index = schema.get_field_index(name)
        if index < 0:
            raise _error(f"strict table physical field {name} is missing")
        actual_type = schema.field(index).type
        valid_type = (
            actual_type in identifier_types
            if name in {"trade_date", "ts_code"}
            else actual_type == stack.pa.float64()
        )
        if not valid_type:
            raise _error(
                f"strict table physical field {name} has a prohibited Arrow type"
            )
    return parquet.read(columns=list(projection))


def _date_from_source(value: Any, label: str) -> str:
    if type(value) is not str:
        raise _error(f"{label} must be a string date")
    if re.fullmatch(r"\d{8}", value):
        try:
            parsed = datetime.strptime(value, "%Y%m%d").date()
        except ValueError as exc:
            raise _error(f"{label} is not a calendar date") from exc
        return parsed.isoformat()
    return _canonical_date(value, label)


def _validate_pit_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    calendar_sessions: Sequence[str],
    cutoff_symbols: Sequence[str],
    expected_cutoff_sha256: str,
    expected_cutoff_count: int,
    stack: DataStack,
    membership_byte_sha256: str,
) -> PITProjection:
    """Build full historical PIT axes from one canonical row per symbol."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise _error("PIT membership rows must be a sequence")
    if not rows or len(rows) > RESOURCE_CONTRACT["pit_row_count_max"]:
        raise _error("PIT membership row count violates the resource contract")
    calendar = tuple(calendar_sessions)
    if not calendar or calendar != tuple(sorted(set(calendar))):
        raise _error("source calendar must be sorted and unique")
    start = calendar[0]
    cutoff = calendar[-1]
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping):
            raise _error(f"PIT membership row {index} must be an object")
        if "ts_code" in raw_row:
            raise _error("PIT membership ts_code alias is prohibited; exact symbol required")
        symbol = raw_row.get("symbol")
        if type(symbol) is not str or _CN_SYMBOL_RE.fullmatch(symbol) is None:
            raise _error("PIT membership contains a noncanonical symbol")
        if symbol in seen:
            raise _error("PIT membership must contain exactly one row per symbol")
        seen.add(symbol)
        effective_from = _date_from_source(
            raw_row.get("effective_from"), "PIT effective_from"
        )
        effective_to_raw = raw_row.get("effective_to")
        if effective_to_raw == "":
            effective_to = ""
        elif type(effective_to_raw) is str:
            effective_to = _date_from_source(effective_to_raw, "PIT effective_to")
        else:
            raise _error("PIT effective_to must be a date or exact blank infinity")
        if effective_to and effective_to <= effective_from:
            raise _error("PIT effective interval must be inclusive/exclusive and nonempty")
        list_date = raw_row.get("list_date", "")
        if list_date not in {"", None} and _date_from_source(
            list_date, "PIT list_date"
        ) != effective_from:
            raise _error("PIT list_date/effective_from conflict")
        delist_date = raw_row.get("delist_date", "")
        if delist_date not in {"", None} and (
            not effective_to
            or _date_from_source(delist_date, "PIT delist_date") != effective_to
        ):
            raise _error("PIT delist_date/effective_to conflict")
        if raw_row.get("source_list_status") == "D" and not effective_to:
            raise _error("delisted PIT row must have a finite effective_to")
        normalized.append(
            {
                "symbol": symbol,
                "effective_from": effective_from,
                "effective_to": effective_to,
            }
        )
    normalized.sort(key=lambda row: row["symbol"])
    active_cutoff = tuple(
        row["symbol"]
        for row in normalized
        if row["effective_from"] <= cutoff
        and (not row["effective_to"] or cutoff < row["effective_to"])
    )
    expected_symbols = tuple(sorted(cutoff_symbols))
    if active_cutoff != expected_symbols or len(active_cutoff) != expected_cutoff_count:
        raise _error("cutoff PIT set differs from the exact recorded full_a scope")
    actual_cutoff_sha = hashlib.sha256(
        "\n".join(active_cutoff).encode("ascii")
    ).hexdigest()
    if actual_cutoff_sha != expected_cutoff_sha256:
        raise _error("cutoff PIT set SHA differs from recorded full_a scope")
    historical = tuple(
        row["symbol"]
        for row in normalized
        if row["effective_from"] <= cutoff
        and (not row["effective_to"] or start < row["effective_to"])
    )
    if not historical or len(historical) > RESOURCE_CONTRACT[
        "historical_symbol_count_max"
    ]:
        raise _error("historical PIT union violates the resource contract")
    row_by_symbol = {row["symbol"]: row for row in normalized}
    mask = stack.np.zeros((len(calendar), len(historical)), dtype=bool)
    for column, symbol in enumerate(historical):
        row = row_by_symbol[symbol]
        mask[:, column] = [
            row["effective_from"] <= session
            and (not row["effective_to"] or session < row["effective_to"])
            for session in calendar
        ]
    mask.setflags(write=False)
    return PITProjection(
        records=tuple(normalized),
        historical_symbols=historical,
        cutoff_symbols=active_cutoff,
        eligibility_mask=mask,
        row_count=len(normalized),
        membership_byte_sha256=membership_byte_sha256,
    )


def _read_pit_membership(
    *,
    accepted: AcceptedPreregistration,
    stack: DataStack,
) -> tuple[Any, StableBytes]:
    record = accepted.backend_binding["pit_generation"]["membership"]

    def parser(file_object: BinaryIO) -> Any:
        parquet = stack.pq.ParquetFile(file_object)
        table = parquet.read()
        required_columns = {
            "symbol",
            "effective_from",
            "effective_to",
        }
        if not required_columns.issubset(set(table.column_names)):
            raise _error("PIT membership exact interval fields are missing")
        if "ts_code" in table.column_names:
            raise _error("PIT membership ts_code alias is prohibited")
        return table

    return _stream_hash_then_parse(
        accepted.pit_membership_path,
        label="immutable PIT membership",
        expected_sha256=record["sha256"],
        expected_size=record["size_bytes"],
        max_bytes=RESOURCE_CONTRACT["pit_max_bytes"],
        expected_nlink=1,
        parser=parser,
    )


def _validate_pit_manifest(
    value: Mapping[str, Any], *, accepted: AcceptedPreregistration
) -> None:
    backend = accepted.backend_binding["pit_generation"]
    if (
        value.get("generation_id") != backend["generation_id"]
        or value.get("canonical_path") != str(accepted.pit_membership_path)
        or value.get("canonical_sha256") != backend["membership"]["sha256"]
        or value.get("row_count") != backend["row_count"]
    ):
        raise _error("immutable PIT generation manifest binding mismatch")


def _read_source_json(
    path: Path,
    *,
    label: str,
    record: Mapping[str, Any],
    max_bytes: int,
) -> tuple[dict[str, Any], StableBytes]:
    return _stream_hash_then_parse(
        path,
        label=label,
        expected_sha256=record["sha256"],
        expected_size=record["size_bytes"],
        max_bytes=max_bytes,
        expected_nlink=1,
        parser=lambda file_object: _strict_json_stream(file_object, label),
    )


def _read_table_members(
    *,
    accepted: AcceptedPreregistration,
    inventory: Sequence[Mapping[str, Any]],
    stack: DataStack,
    parse_dataset: bool,
) -> tuple[list[Any], dict[str, tuple[int, ...]]]:
    tables: list[Any] = []
    signatures: dict[str, tuple[int, ...]] = {}
    for row in inventory:
        relative = row["relative_path"]
        member_path = accepted.table_root.joinpath(*relative.split("/"))
        if parse_dataset and row["dataset_member"]:
            parser = lambda file_object, _stack=stack: _parse_physical_table(
                file_object, stack=_stack
            )
        else:
            parser = lambda file_object: file_object.read()
        value, stable = _stream_hash_then_parse(
            member_path,
            label=f"strict table member {relative}",
            expected_sha256=row["sha256"],
            expected_size=row["size_bytes"],
            max_bytes=RESOURCE_CONTRACT["table_member_max_bytes"],
            expected_nlink=row["hard_link_count"],
            parser=parser,
        )
        signatures[relative] = stable.signature
        if parse_dataset and row["dataset_member"]:
            tables.append(value)
    if parse_dataset and not tables:
        raise _error("strict table inventory contains no dataset members")
    return tables, signatures


def _validate_and_project_table(
    *,
    tables: Sequence[Any],
    accepted: AcceptedPreregistration,
    pit: PITProjection,
    stack: DataStack,
    inventory: Sequence[Mapping[str, Any]],
    source_observation_bindings: Mapping[str, Any],
) -> TableCollection:
    try:
        combined = stack.pa.concat_tables(list(tables), promote_options="none")
    except TypeError:  # PyArrow versions before promote_options
        combined = stack.pa.concat_tables(list(tables), promote=False)
    projected_count = int(combined.num_rows)
    if projected_count > RESOURCE_CONTRACT["projected_row_count_per_pass_max"]:
        raise _error("projected table row resource cap exceeded")
    frame = combined.to_pandas()
    if list(frame.columns) != [
        "trade_date",
        "ts_code",
        "open",
        "close",
        "vol",
        "adj_close",
    ]:
        raise _error("strict table projection columns/order mismatch")
    frame["trade_date"] = [
        _date_from_source(value, "strict table trade_date")
        for value in frame["trade_date"].tolist()
    ]
    symbols = frame["ts_code"].tolist()
    if any(type(value) is not str or _CN_SYMBOL_RE.fullmatch(value) is None for value in symbols):
        raise _error("strict table contains a noncanonical ts_code")
    pit_symbols = {row["symbol"] for row in pit.records}
    missing_pit = sorted(set(symbols) - pit_symbols)
    if missing_pit:
        raise _error("strict table contains a symbol missing from PIT membership")
    for field in ("open", "close", "vol", "adj_close"):
        values = frame[field].to_numpy(dtype=stack.np.float64, copy=True)
        if stack.np.isinf(values).any():
            raise _error(f"strict table physical field {field} contains infinity")
        frame[field] = values
    calendar = tuple(accepted.calendar_sessions)
    calendar_set = set(calendar)
    analysis_start = calendar[0]
    cutoff = calendar[-1]
    in_interval = frame["trade_date"].between(analysis_start, cutoff)
    off_calendar = frame.loc[
        in_interval & ~frame["trade_date"].isin(calendar_set),
        ["trade_date", "ts_code"],
    ]
    if not off_calendar.empty:
        raise _error("strict table has calendar-off rows inside analysis interval")
    relevant = frame.loc[in_interval & frame["trade_date"].isin(calendar_set)].copy()
    if relevant.duplicated(subset=["trade_date", "ts_code"], keep=False).any():
        raise _error("strict table has duplicate trade_date/ts_code rows")
    ignored_pre_analysis = int((frame["trade_date"] < analysis_start).sum())
    interval_by_symbol = {
        row["symbol"]: (row["effective_from"], row["effective_to"])
        for row in pit.records
    }
    outside_pit = 0
    for trade_date, symbol in relevant[["trade_date", "ts_code"]].itertuples(
        index=False, name=None
    ):
        effective_from, effective_to = interval_by_symbol[symbol]
        if trade_date < effective_from or (effective_to and trade_date >= effective_to):
            outside_pit += 1
    collection_payload = {
        "preregistration_readback_byte_sha256": source_observation_bindings[
            "preregistration_readback_byte_sha256"
        ],
        "snapshot_manifest_byte_sha256": source_observation_bindings[
            "snapshot_manifest_byte_sha256"
        ],
        "pit_generation_manifest_byte_sha256": source_observation_bindings[
            "pit_generation_manifest_byte_sha256"
        ],
        "pit_membership_byte_sha256": pit.membership_byte_sha256,
        "table_inventory": [copy.deepcopy(dict(row)) for row in inventory],
        "source_calendar": list(calendar),
        "historical_symbols": list(pit.historical_symbols),
        "cutoff_symbols": list(pit.cutoff_symbols),
        "pit_row_count": pit.row_count,
        "projected_row_count": int(len(relevant)),
        "ignored_pre_analysis_row_count": ignored_pre_analysis,
        "outside_pit_bar_count": outside_pit,
    }
    return TableCollection(
        table=relevant,
        inventory_rows=tuple(copy.deepcopy(dict(row)) for row in inventory),
        projected_row_count=int(len(relevant)),
        outside_pit_bar_count=outside_pit,
        ignored_pre_analysis_row_count=ignored_pre_analysis,
        collection_sha256=_semantic_sha(collection_payload),
    )


def _block_input(
    *,
    table: Any,
    pit: PITProjection,
    calendar_sessions: Sequence[str],
    block_row: Mapping[str, Any],
    stack: DataStack,
) -> Any:
    start = block_row["input_start_offset"]
    end = block_row["input_end_offset"]
    dates = tuple(calendar_sessions[start:end])
    symbols = pit.historical_symbols
    dense_cells = len(dates) * len(symbols)
    if dense_cells > RESOURCE_CONTRACT["dense_cell_count_per_block_max"]:
        raise _error("dense input block cell cap exceeded")
    selected = table.loc[table["trade_date"].isin(dates)]
    matrices: dict[str, Any] = {}
    mapping = {
        "raw_close": "close",
        "raw_open": "open",
        "vol": "vol",
        "adj_close": "adj_close",
    }
    for input_name, physical_name in mapping.items():
        pivot = selected.pivot(
            index="trade_date", columns="ts_code", values=physical_name
        ).reindex(index=dates, columns=symbols)
        matrix = stack.np.array(
            pivot.to_numpy(dtype=stack.np.float64),
            dtype=stack.np.float64,
            order="C",
            copy=True,
        )
        matrices[input_name] = matrix
    mask = stack.np.array(
        pit.eligibility_mask[start:end], dtype=bool, order="C", copy=True
    )
    for matrix in matrices.values():
        matrix[~mask] = stack.np.nan
    return stack.evaluator.build_input_block_v4_4(
        dates=dates,
        symbols=symbols,
        raw_close=matrices["raw_close"],
        raw_open=matrices["raw_open"],
        vol=matrices["vol"],
        adj_close=matrices["adj_close"],
        pit_mask=mask,
    )


def _axis_descriptor_local(values: Sequence[str]) -> dict[str, Any]:
    items = list(values)
    return {
        "count": len(items),
        "sha256": hashlib.sha256(
            b"".join(item.encode("utf-8") + b"\n" for item in items)
        ).hexdigest(),
        "first": items[0] if items else None,
        "last": items[-1] if items else None,
    }


class _StreamingPackedBinaryMask:
    """Pack chronological boolean row blocks without retaining a dense mask."""

    def __init__(self, *, symbols: Sequence[str], stack: DataStack) -> None:
        self._symbols = tuple(symbols)
        self._stack = stack
        self._dates: list[str] = []
        self._packed = bytearray()
        self._pending_value = 0
        self._pending_count = 0
        self._bit_count = 0
        self._finalized = False

    def update(self, dates: Sequence[str], values: Any) -> None:
        if self._finalized:
            raise _error("streaming binary mask is already finalized")
        chunk_dates = tuple(dates)
        if not chunk_dates:
            raise _error("streaming binary mask chunk must not be empty")
        if (
            tuple(sorted(chunk_dates)) != chunk_dates
            or len(set(chunk_dates)) != len(chunk_dates)
            or (self._dates and chunk_dates[0] <= self._dates[-1])
        ):
            raise _error("streaming binary mask dates must be chronological")
        np = self._stack.np
        matrix = np.asarray(values)
        if (
            type(matrix) is not np.ndarray
            or matrix.dtype != np.dtype(bool)
            or matrix.ndim != 2
            or matrix.shape != (len(chunk_dates), len(self._symbols))
        ):
            raise _error("streaming binary mask chunk shape/dtype mismatch")
        flat = np.array(matrix, dtype=bool, order="C", copy=False).reshape(-1)
        offset = 0
        if self._pending_count:
            needed = min(8 - self._pending_count, len(flat))
            for bit_offset in range(needed):
                if bool(flat[bit_offset]):
                    self._pending_value |= 1 << (
                        self._pending_count + bit_offset
                    )
            self._pending_count += needed
            offset += needed
            if self._pending_count == 8:
                self._packed.append(self._pending_value)
                self._pending_value = 0
                self._pending_count = 0
            elif offset == len(flat):
                self._bit_count += len(flat)
                self._dates.extend(chunk_dates)
                return
        remaining = len(flat) - offset
        full_bit_count = (remaining // 8) * 8
        if full_bit_count:
            packed = np.packbits(
                flat[offset : offset + full_bit_count], bitorder="little"
            )
            self._packed.extend(packed.tobytes(order="C"))
            offset += full_bit_count
        for bit_offset in range(len(flat) - offset):
            if bool(flat[offset + bit_offset]):
                self._pending_value |= 1 << bit_offset
        self._pending_count = len(flat) - offset
        self._bit_count += len(flat)
        self._dates.extend(chunk_dates)

    def finalize(self, *, contract: Any) -> dict[str, Any]:
        if self._finalized or not self._dates:
            raise _error("streaming binary mask cannot be finalized")
        self._finalized = True
        if self._pending_count:
            self._packed.append(self._pending_value)
        if self._bit_count != len(self._dates) * len(self._symbols):
            raise _error("streaming binary mask cell accounting mismatch")
        return contract.build_packed_binary_mask_descriptor_v4_4(
            packed_bits=bytes(self._packed),
            bit_count=self._bit_count,
            dates=self._dates,
            symbols=self._symbols,
        )


def _peak_rss_bytes() -> int:
    resource_module = importlib.import_module("resource")
    value = int(
        resource_module.getrusage(resource_module.RUSAGE_SELF).ru_maxrss
    )
    return value if sys.platform == "darwin" else value * 1024


def _evaluate_exact_five_blocks(
    *,
    pass_id: str,
    table: Any,
    pit: PITProjection,
    calendar_sessions: Sequence[str],
    stack: DataStack,
    contract: Any,
    operator_program_set: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    dict[str, Any],
    tuple[dict[str, Any], ...],
    dict[str, int],
]:
    evaluator = stack.evaluator
    validated_program_set = contract.validate_operator_program_set_v4_4(
        operator_program_set
    )
    block_manifest = evaluator.build_block_manifest_v4_4(
        calendar_sessions, pit.historical_symbols
    )
    proof_dates = tuple(block_manifest["proof_output_calendar"])
    raw_states = {
        engine_id: {
            name: evaluator.StreamingMatrixDescriptorV4_4(pit.historical_symbols)
            for name in evaluator.FACTOR_NAMES
        }
        for engine_id in (evaluator.PANDAS_ENGINE_ID, evaluator.NUMPY_ENGINE_ID)
    }
    adjusted_states = {
        engine_id: {
            name: evaluator.StreamingMatrixDescriptorV4_4(pit.historical_symbols)
            for name in evaluator.FACTOR_NAMES
        }
        for engine_id in (evaluator.PANDAS_ENGINE_ID, evaluator.NUMPY_ENGINE_ID)
    }
    proof_pit_state = _StreamingPackedBinaryMask(
        symbols=pit.historical_symbols, stack=stack
    )
    full_pit_state = _StreamingPackedBinaryMask(
        symbols=pit.historical_symbols, stack=stack
    )
    candidate_mask_states = {
        name: _StreamingPackedBinaryMask(
            symbols=pit.historical_symbols, stack=stack
        )
        for name in evaluator.FACTOR_NAMES
    }
    input_states = {
        field: evaluator.StreamingMatrixDescriptorV4_4(pit.historical_symbols)
        for field in evaluator.INPUT_FIELDS
    }
    outside_pit_non_null_counts = {
        field: 0 for field in evaluator.INPUT_FIELDS
    }
    futures = importlib.import_module("concurrent.futures")
    observed_dates: list[str] = []
    with futures.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix=f"{pass_id}-strict-engine"
    ) as executor:
        for block_index, block_row in enumerate(block_manifest["blocks"]):
            input_block = _block_input(
                table=table,
                pit=pit,
                calendar_sessions=calendar_sessions,
                block_row=block_row,
                stack=stack,
            )
            if block_index == 0:
                descriptor_start = 0
            else:
                descriptor_start = block_row["local_output_start_offset"]
            descriptor_end = block_row["local_output_end_offset"]
            descriptor_dates = input_block.dates[descriptor_start:descriptor_end]
            descriptor_mask = input_block.pit_mask[descriptor_start:descriptor_end]
            full_pit_state.update(descriptor_dates, descriptor_mask)
            for field in evaluator.INPUT_FIELDS:
                descriptor_matrix = getattr(input_block, field)[
                    descriptor_start:descriptor_end
                ]
                input_states[field].update(descriptor_dates, descriptor_matrix)
                outside_pit_non_null_counts[field] += int(
                    stack.np.count_nonzero(
                        ~stack.np.isnan(descriptor_matrix[~descriptor_mask])
                    )
                )
            pandas_future = executor.submit(
                evaluator.evaluate_pandas_engine_v4_4,
                input_block,
                operator_program_set=validated_program_set,
            )
            numpy_future = executor.submit(
                evaluator.evaluate_numpy_engine_v4_4,
                input_block,
                operator_program_set=validated_program_set,
            )
            pandas_outputs = pandas_future.result()
            numpy_outputs = numpy_future.result()
            evaluator.compare_exact_engine_outputs_v4_4(
                pandas_outputs,
                numpy_outputs,
                input_block,
                require_positive_proof=False,
            )
            selected_pandas = evaluator.slice_non_halo_outputs_v4_4(
                pandas_outputs, block_row, source_block=input_block
            )
            selected_numpy = evaluator.slice_non_halo_outputs_v4_4(
                numpy_outputs, block_row, source_block=input_block
            )
            local_start = block_row["local_output_start_offset"]
            local_end = block_row["local_output_end_offset"]
            chunk_dates = input_block.dates[local_start:local_end]
            observed_dates.extend(chunk_dates)
            proof_pit = stack.np.array(
                input_block.pit_mask[local_start:local_end],
                dtype=bool,
                order="C",
                copy=True,
            )
            proof_pit_state.update(chunk_dates, proof_pit)
            for name in evaluator.FACTOR_NAMES:
                candidate_mask_states[name].update(
                    chunk_dates, stack.np.isfinite(selected_pandas[name])
                )
            for engine_id, outputs in (
                (evaluator.PANDAS_ENGINE_ID, selected_pandas),
                (evaluator.NUMPY_ENGINE_ID, selected_numpy),
            ):
                for name in evaluator.FACTOR_NAMES:
                    raw_states[engine_id][name].update(chunk_dates, outputs[name])
                    direction = evaluator.FACTOR_DIRECTIONS[name]
                    adjusted = stack.np.array(
                        outputs[name] * direction,
                        dtype=stack.np.float64,
                        order="C",
                        copy=True,
                    )
                    adjusted_states[engine_id][name].update(chunk_dates, adjusted)
            del input_block, pandas_outputs, numpy_outputs, selected_pandas, selected_numpy
    if tuple(observed_dates) != proof_dates:
        raise _error("evaluator blocks did not cover proof output calendar exactly once")
    proof_pit_descriptor = proof_pit_state.finalize(contract=contract)
    full_pit_descriptor = full_pit_state.finalize(contract=contract)
    candidate_mask_descriptors = {
        name: candidate_mask_states[name].finalize(contract=contract)
        for name in evaluator.FACTOR_NAMES
    }
    input_descriptors = tuple(
        {
            "field": field,
            "descriptor": input_states[field].finalize(),
        }
        for field in evaluator.INPUT_FIELDS
    )
    engine_descriptors: list[dict[str, Any]] = []
    for engine_id in (evaluator.PANDAS_ENGINE_ID, evaluator.NUMPY_ENGINE_ID):
        raw_descriptors = {
            name: raw_states[engine_id][name].finalize()
            for name in evaluator.FACTOR_NAMES
        }
        adjusted_descriptors = {
            name: adjusted_states[engine_id][name].finalize()
            for name in evaluator.FACTOR_NAMES
        }
        for name in evaluator.FACTOR_NAMES:
            if raw_descriptors[name]["finite_count"] <= 0:
                raise _error(f"{name} has no finite eligible observation")
        engine_descriptors.append(
            {
                "engine_id": engine_id,
                "raw_matrix_descriptors": copy.deepcopy(raw_descriptors),
                "adjusted_matrix_descriptors": copy.deepcopy(
                    adjusted_descriptors
                ),
            }
        )
    return (
        block_manifest,
        proof_pit_descriptor,
        tuple(
            candidate_mask_descriptors[name]
            for name in evaluator.FACTOR_NAMES
        ),
        tuple(engine_descriptors),
        full_pit_descriptor,
        input_descriptors,
        outside_pit_non_null_counts,
    )


def _validate_snapshot_manifest(
    value: Mapping[str, Any], *, accepted: AcceptedPreregistration
) -> None:
    cutoff_compact = accepted.strict_source["cutoff"].replace("-", "")
    if (
        value.get("snapshot_id") != accepted.strict_source["snapshot_id"]
        or value.get("market") != "CN"
        or value.get("status") != "OK"
        or value.get("blockers") != []
        or value.get("table_root") != str(accepted.table_root)
        or value.get("latest_available_trade_date") != cutoff_compact
        or value.get("latest_complete_trade_date") != cutoff_compact
    ):
        raise _error("immutable snapshot manifest is not the exact healthy cutoff")


def _accepted_prereg_semantic(accepted: AcceptedPreregistration) -> str:
    return _semantic_sha(
        {
            "artifacts": accepted.artifacts,
            "calendar_sessions": list(accepted.calendar_sessions),
            "candidate_rows": list(accepted.candidate_rows),
        }
    )


def _execute_data_pass(
    *,
    pass_id: str,
    manifest: Mapping[str, Any],
    modules: LoadedModules,
    stack: DataStack,
    operator_program_set: Mapping[str, Any],
) -> PassEvidence:
    if pass_id not in {"fresh_pass_1", "fresh_pass_2"}:
        raise _error("data pass id must be fresh_pass_1 or fresh_pass_2")
    started = time.monotonic()
    accepted = _accept_preregistration(manifest=manifest, modules=modules)
    accepted_semantic = _accepted_prereg_semantic(accepted)

    snapshot_value, snapshot_before = _read_source_json(
        accepted.snapshot_manifest_path,
        label=f"{pass_id} immutable snapshot manifest before",
        record=accepted.backend_binding["snapshot_manifest"],
        max_bytes=RESOURCE_CONTRACT["strict_artifact_max_bytes"],
    )
    _validate_snapshot_manifest(snapshot_value, accepted=accepted)
    pit_manifest_value, pit_manifest_before = _read_source_json(
        accepted.pit_manifest_path,
        label=f"{pass_id} immutable PIT manifest before",
        record=accepted.backend_binding["pit_generation"]["manifest"],
        max_bytes=RESOURCE_CONTRACT["strict_artifact_max_bytes"],
    )
    _validate_pit_manifest(pit_manifest_value, accepted=accepted)
    pit_table, pit_before = _read_pit_membership(accepted=accepted, stack=stack)
    if int(pit_table.num_rows) != accepted.backend_binding["pit_generation"][
        "row_count"
    ]:
        raise _error("PIT membership row count differs from accepted source binding")
    pit = _validate_pit_rows(
        pit_table.to_pylist(),
        calendar_sessions=accepted.calendar_sessions,
        cutoff_symbols=accepted.strict_source["component_symbols"],
        expected_cutoff_sha256=accepted.strict_source["full_a_scope_sha256"],
        expected_cutoff_count=accepted.strict_source["expected_scope_count"],
        stack=stack,
        membership_byte_sha256=pit_before.byte_sha256,
    )
    inventory = _validated_table_inventory(accepted=accepted)
    tables, table_signatures_before = _read_table_members(
        accepted=accepted,
        inventory=inventory,
        stack=stack,
        parse_dataset=True,
    )
    report_filename = modules.prereg_bundle.READBACK_REPORT_FILENAME_V4_4
    source_observations = {
        "preregistration_readback_byte_sha256": accepted.readback[
            "artifact_descriptors"
        ][report_filename]["byte_sha256"],
        "snapshot_manifest_byte_sha256": snapshot_before.byte_sha256,
        "pit_generation_manifest_byte_sha256": pit_manifest_before.byte_sha256,
        "pit_membership_byte_sha256": pit_before.byte_sha256,
        "table_inventory_semantic_sha256": accepted.backend_binding["table"][
            "inventory_sha256"
        ],
    }
    table_collection = _validate_and_project_table(
        tables=tables,
        accepted=accepted,
        pit=pit,
        stack=stack,
        inventory=inventory,
        source_observation_bindings=source_observations,
    )
    del tables, pit_table
    dense_projected_row_count = len(accepted.calendar_sessions) * len(
        pit.historical_symbols
    )
    if dense_projected_row_count > RESOURCE_CONTRACT[
        "projected_row_count_per_pass_max"
    ]:
        raise _error("dense projected row count per pass exceeds fixed cap")
    (
        block_manifest,
        proof_pit_mask_descriptor,
        candidate_non_null_mask_descriptors,
        engine_bodies,
        pit_mask_descriptor,
        input_descriptors,
        outside_pit_non_null_counts,
    ) = _evaluate_exact_five_blocks(
        pass_id=pass_id,
        table=table_collection.table,
        pit=pit,
        calendar_sessions=accepted.calendar_sessions,
        stack=stack,
        contract=modules.contract,
        operator_program_set=operator_program_set,
    )
    field_missing_counts: dict[str, int] = {}
    outside_cells = pit_mask_descriptor["zero_count"]
    for row in input_descriptors:
        field = row["field"]
        descriptor = row["descriptor"]
        missing = descriptor["nan_count"] - outside_cells
        if missing < 0:
            raise _error(f"{pass_id} {field} missing accounting underflow")
        field_missing_counts[field] = missing
        if outside_pit_non_null_counts[field] != 0:
            raise _error(f"{pass_id} {field} leaked a non-null value outside PIT")

    # Every immutable source is fully reopened after computation in this pass.
    snapshot_after_value, snapshot_after = _read_source_json(
        accepted.snapshot_manifest_path,
        label=f"{pass_id} immutable snapshot manifest after",
        record=accepted.backend_binding["snapshot_manifest"],
        max_bytes=RESOURCE_CONTRACT["strict_artifact_max_bytes"],
    )
    _validate_snapshot_manifest(snapshot_after_value, accepted=accepted)
    pit_manifest_after_value, pit_manifest_after = _read_source_json(
        accepted.pit_manifest_path,
        label=f"{pass_id} immutable PIT manifest after",
        record=accepted.backend_binding["pit_generation"]["manifest"],
        max_bytes=RESOURCE_CONTRACT["strict_artifact_max_bytes"],
    )
    _validate_pit_manifest(pit_manifest_after_value, accepted=accepted)
    pit_after_table, pit_after = _read_pit_membership(
        accepted=accepted, stack=stack
    )
    inventory_after = _validated_table_inventory(accepted=accepted)
    _unused, table_signatures_after = _read_table_members(
        accepted=accepted,
        inventory=inventory_after,
        stack=stack,
        parse_dataset=False,
    )
    if (
        snapshot_after.signature != snapshot_before.signature
        or pit_manifest_after.signature != pit_manifest_before.signature
        or pit_after.signature != pit_before.signature
        or int(pit_after_table.num_rows) != pit.row_count
        or inventory_after != inventory
        or table_signatures_after != table_signatures_before
    ):
        raise _error(f"{pass_id} immutable source drifted across before/after reads")
    accepted_after = _accept_preregistration(manifest=manifest, modules=modules)
    if _accepted_prereg_semantic(accepted_after) != accepted_semantic:
        raise _error(f"{pass_id} predecessor27 drifted during computation")

    elapsed = time.monotonic() - started
    peak_rss = _peak_rss_bytes()
    if elapsed > RESOURCE_CONTRACT["pass_wall_seconds_max"]:
        raise _error(f"{pass_id} exceeded fixed wall-time resource cap")
    if peak_rss > RESOURCE_CONTRACT["rss_max_bytes"]:
        raise _error(f"{pass_id} exceeded fixed RSS resource cap")
    historical_only = len(pit.historical_symbols) - len(pit.cutoff_symbols)
    if historical_only <= 0:
        raise _error("historical PIT axis must not collapse to cutoff-only symbols")
    historical_symbol_axis = {
        "scope": "all_historical_pit_symbols",
        "cutoff_only": False,
        "contains_all_cutoff_full_a": True,
        "historical_only_symbol_count": historical_only,
        "hash_algorithm": modules.contract.AXIS_HASH_ALGORITHM,
        "descriptor": _axis_descriptor_local(pit.historical_symbols),
    }
    pit_contract = {
        "row_count": pit.row_count,
        "distinct_symbol_count": len(pit.records),
        "historical_union_symbol_count": len(pit.historical_symbols),
        "duplicate_symbol_count": 0,
        "one_row_per_symbol": True,
        "effective_from_semantics": "inclusive",
        "effective_to_semantics": "exclusive",
        "blank_effective_to_semantics": "positive_infinity",
        "membership_byte_sha256": pit.membership_byte_sha256,
    }
    return PassEvidence(
        pass_id=pass_id,
        accepted_preregistration_semantic_sha256=accepted_semantic,
        source_observation_bindings=source_observations,
        source_identity_signatures={
            "snapshot_manifest": list(snapshot_before.signature),
            "pit_generation_manifest": list(pit_manifest_before.signature),
            "pit_membership": list(pit_before.signature),
            "table_members": {
                relative: list(signature)
                for relative, signature in table_signatures_before.items()
            },
        },
        block_manifest=copy.deepcopy(block_manifest),
        pit_mask_descriptor=copy.deepcopy(pit_mask_descriptor),
        proof_pit_mask_descriptor=copy.deepcopy(
            proof_pit_mask_descriptor
        ),
        candidate_non_null_mask_descriptors=tuple(
            copy.deepcopy(candidate_non_null_mask_descriptors)
        ),
        historical_symbol_axis=historical_symbol_axis,
        pit_membership_contract=pit_contract,
        input_matrix_descriptors=tuple(copy.deepcopy(input_descriptors)),
        engine_matrix_descriptors=tuple(copy.deepcopy(engine_bodies)),
        field_missing_counts=field_missing_counts,
        outside_pit_non_null_counts=outside_pit_non_null_counts,
        bars_outside_pit_interval_count=table_collection.outside_pit_bar_count,
        ignored_pre_analysis_row_count=table_collection.ignored_pre_analysis_row_count,
        dense_projected_row_count=dense_projected_row_count,
        table_content_binding_sha256=table_collection.collection_sha256,
        elapsed_seconds=elapsed,
        peak_rss_bytes=peak_rss,
    )


def _verify_loaded_stack_identity(
    *, stack: DataStack, frozen: FrozenBindings, manifest: Mapping[str, Any]
) -> None:
    expected_runtime_names = tuple(
        row[0] for row in RUNTIME_DISTRIBUTION_TOP_LEVEL
    )
    if tuple(
        row["name"] for row in frozen.runtime["distributions"]
    ) != expected_runtime_names:
        raise _error("frozen runtime is not the exact six-distribution closure")
    runtime_versions = {
        row["name"]: row["version"] for row in frozen.runtime["distributions"]
    }
    observed_versions = {
        "numpy": stack.np.__version__,
        "pandas": stack.pd.__version__,
        "pyarrow": stack.pa.__version__,
    }
    if observed_versions != {
        name: runtime_versions[name] for name in observed_versions
    }:
        raise _error("loaded data stack differs from the frozen runtime inventory")
    evaluator = stack.evaluator
    if (
        evaluator.HALO != RESOURCE_CONTRACT["halo_session_count"]
        or evaluator.OUTPUT_BLOCK
        != RESOURCE_CONTRACT["output_block_session_count"]
        or tuple(evaluator.INPUT_FIELDS)
        != ("raw_close", "raw_open", "vol", "adj_close")
        or tuple(evaluator.FACTOR_NAMES)
        != tuple(row["name"] for row in manifest["source_definition_bindings"])
        or tuple(evaluator.FACTOR_DIRECTIONS[name] for name in evaluator.FACTOR_NAMES)
        != tuple(row["direction"] for row in manifest["source_definition_bindings"])
        or evaluator.PANDAS_ENGINE_ID
        != "closed_pandas_source_dag.future_strictexact.v4.4"
        or evaluator.NUMPY_ENGINE_ID
        != "independent_numpy_local_formulas.future_strictexact.v4.4"
    ):
        raise _error("loaded exact-five evaluator identity differs from manifest contract")


def _pass_equivalence_payload(evidence: PassEvidence) -> dict[str, Any]:
    return {
        "accepted_preregistration_semantic_sha256": (
            evidence.accepted_preregistration_semantic_sha256
        ),
        "source_observation_bindings": evidence.source_observation_bindings,
        "source_identity_signatures": evidence.source_identity_signatures,
        "block_manifest": evidence.block_manifest,
        "pit_mask_descriptor": evidence.pit_mask_descriptor,
        "proof_pit_mask_descriptor": evidence.proof_pit_mask_descriptor,
        "candidate_non_null_mask_descriptors": list(
            evidence.candidate_non_null_mask_descriptors
        ),
        "historical_symbol_axis": evidence.historical_symbol_axis,
        "pit_membership_contract": evidence.pit_membership_contract,
        "input_matrix_descriptors": list(evidence.input_matrix_descriptors),
        "engine_matrix_descriptors": list(evidence.engine_matrix_descriptors),
        "field_missing_counts": evidence.field_missing_counts,
        "outside_pit_non_null_counts": evidence.outside_pit_non_null_counts,
        "bars_outside_pit_interval_count": (
            evidence.bars_outside_pit_interval_count
        ),
        "ignored_pre_analysis_row_count": (
            evidence.ignored_pre_analysis_row_count
        ),
        "dense_projected_row_count": evidence.dense_projected_row_count,
        "table_content_binding_sha256": evidence.table_content_binding_sha256,
    }


def _assert_fresh_pass_equivalence(
    first: PassEvidence, second: PassEvidence
) -> None:
    if first.pass_id != "fresh_pass_1" or second.pass_id != "fresh_pass_2":
        raise _error("fresh pass order/identity mismatch")
    if _canonical_bytes_local(_pass_equivalence_payload(first)) != (
        _canonical_bytes_local(_pass_equivalence_payload(second))
    ):
        raise _error("two fresh source/evaluator passes are not bitwise equivalent")


def _assert_preflight_still_current(preflight: PublicationPreflight) -> None:
    root_fd, metadata = _open_fixed_private_root(preflight.root)
    try:
        if _signature(metadata) != preflight.root_signature:
            raise _error("fixed publication root identity changed before commit")
        try:
            os.stat(preflight.cycle_id, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise _error("fixed strict publication target appeared before commit")
    finally:
        os.close(root_fd)


def _locked_precommit_revalidate(
    *,
    manifest: Mapping[str, Any],
    stable_manifest: StableBytes,
    modules: LoadedModules,
    stack: DataStack,
    frozen: FrozenBindings,
    preflight: PublicationPreflight,
    reference_pass: PassEvidence,
    reference_operator_program_set: Mapping[str, Any],
    total_started: float,
) -> None:
    """Reopen every external input under the private publication lock."""

    current_manifest, current_stable = _read_manifest_two_fresh(
        stable_manifest.path,
        expected_byte_sha256=stable_manifest.byte_sha256,
    )
    current_manifest = _stage0_manifest_validate(current_manifest)
    if (
        current_stable.signature != stable_manifest.signature
        or current_stable.raw != stable_manifest.raw
        or _canonical_bytes_local(current_manifest) != _canonical_bytes_local(manifest)
    ):
        raise _error("input manifest identity changed before locked commit")
    accepted = _accept_preregistration(manifest=manifest, modules=modules)
    if _accepted_prereg_semantic(accepted) != (
        reference_pass.accepted_preregistration_semantic_sha256
    ):
        raise _error("predecessor27 changed before locked commit")

    snapshot_value, snapshot = _read_source_json(
        accepted.snapshot_manifest_path,
        label="locked immutable snapshot manifest",
        record=accepted.backend_binding["snapshot_manifest"],
        max_bytes=RESOURCE_CONTRACT["strict_artifact_max_bytes"],
    )
    _validate_snapshot_manifest(snapshot_value, accepted=accepted)
    pit_manifest_value, pit_manifest = _read_source_json(
        accepted.pit_manifest_path,
        label="locked immutable PIT generation manifest",
        record=accepted.backend_binding["pit_generation"]["manifest"],
        max_bytes=RESOURCE_CONTRACT["strict_artifact_max_bytes"],
    )
    _validate_pit_manifest(pit_manifest_value, accepted=accepted)
    pit_table, pit_stable = _read_pit_membership(accepted=accepted, stack=stack)
    locked_pit = _validate_pit_rows(
        pit_table.to_pylist(),
        calendar_sessions=accepted.calendar_sessions,
        cutoff_symbols=accepted.strict_source["component_symbols"],
        expected_cutoff_sha256=accepted.strict_source["full_a_scope_sha256"],
        expected_cutoff_count=accepted.strict_source["expected_scope_count"],
        stack=stack,
        membership_byte_sha256=pit_stable.byte_sha256,
    )
    inventory = _validated_table_inventory(accepted=accepted)
    _unused, table_signatures = _read_table_members(
        accepted=accepted,
        inventory=inventory,
        stack=stack,
        parse_dataset=False,
    )
    locked_identities = {
        "snapshot_manifest": list(snapshot.signature),
        "pit_generation_manifest": list(pit_manifest.signature),
        "pit_membership": list(pit_stable.signature),
        "table_members": {
            relative: list(signature)
            for relative, signature in table_signatures.items()
        },
    }
    if locked_identities != reference_pass.source_identity_signatures:
        raise _error("immutable source identity changed before locked commit")
    if reference_pass.historical_symbol_axis["descriptor"] != (
        _axis_descriptor_local(locked_pit.historical_symbols)
    ):
        raise _error("locked PIT historical symbol axis changed")
    accepted_after = _accept_preregistration(manifest=manifest, modules=modules)
    if _accepted_prereg_semantic(accepted_after) != (
        reference_pass.accepted_preregistration_semantic_sha256
    ):
        raise _error("predecessor27 changed during locked source revalidation")
    _revalidate_fixed_bindings(
        frozen=frozen,
        manifest=manifest,
        modules=modules,
    )
    locked_program_set = _verify_pinned_source_definitions(
        manifest=manifest, contract=modules.contract
    )
    if modules.contract.canonical_json_bytes_v4_4(locked_program_set) != (
        modules.contract.canonical_json_bytes_v4_4(
            reference_operator_program_set
        )
    ):
        raise _error("canonical operator program set changed before locked commit")
    _audit_closed_project_imports(modules.import_guard)
    _audit_closed_runtime_imports(modules.import_guard)
    _rehash_runtime_shadow()
    _assert_preflight_still_current(preflight)
    final_manifest, final_stable = _read_manifest_two_fresh(
        stable_manifest.path,
        expected_byte_sha256=stable_manifest.byte_sha256,
    )
    if (
        final_stable.signature != stable_manifest.signature
        or final_stable.raw != stable_manifest.raw
        or _canonical_bytes_local(_stage0_manifest_validate(final_manifest))
        != _canonical_bytes_local(manifest)
    ):
        raise _error("input manifest changed during locked revalidation")
    if time.monotonic() - total_started > RESOURCE_CONTRACT[
        "total_wall_seconds_max"
    ]:
        raise _error("publisher exceeded fixed total wall-time cap before commit")


def _validate_shadow_container(
    path: Path, descriptor: int, *, expected_mode: frozenset[int]
) -> tuple[int, ...]:
    if (
        path.parent != RUNTIME_SHADOW_PARENT
        or _RUNTIME_SHADOW_NAME_RE.fullmatch(path.name) is None
        or _path_is_within(path, PRODUCTION_PRIVATE_ROOT)
        or _path_is_within(PRODUCTION_PRIVATE_ROOT, path)
    ):
        raise _error("runtime shadow path is outside the exact private namespace")
    try:
        path_metadata = os.lstat(path)
        opened = os.fstat(descriptor)
    except OSError as exc:
        raise _error("runtime shadow identity is unavailable") from exc
    if (
        stat.S_ISLNK(path_metadata.st_mode)
        or not stat.S_ISDIR(path_metadata.st_mode)
        or not _same_object(path_metadata, opened)
        or int(opened.st_uid) != os.getuid()
        or stat.S_IMODE(opened.st_mode) not in expected_mode
    ):
        raise _error("runtime shadow must be one owner-private real directory")
    return _signature(opened)


def _remove_shadow_contents_fd(descriptor: int) -> None:
    os.fchmod(descriptor, 0o700)
    with os.scandir(descriptor) as iterator:
        names = tuple(entry.name for entry in iterator)
    for name in names:
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
            child = os.open(name, _directory_flags(), dir_fd=descriptor)
            try:
                opened = os.fstat(child)
                if not _same_object(metadata, opened):
                    raise _error("runtime shadow directory swapped during cleanup")
                _remove_shadow_contents_fd(child)
            finally:
                os.close(child)
            os.rmdir(name, dir_fd=descriptor)
        else:
            os.unlink(name, dir_fd=descriptor)


def _anchored_cleanup_runtime_shadow(
    path: Path, *, expected_identity: tuple[int, int] | None = None
) -> None:
    parent_fd = _open_shadow_parent()
    descriptor = -1
    try:
        match = _RUNTIME_SHADOW_NAME_RE.fullmatch(path.name)
        if path.parent != RUNTIME_SHADOW_PARENT or match is None:
            raise _error("refusing to clean a non-shadow path")
        metadata = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise _error("refusing to clean a linked/non-directory shadow")
        descriptor = os.open(path.name, _directory_flags(), dir_fd=parent_fd)
        opened = os.fstat(descriptor)
        _validate_shadow_container(
            path, descriptor, expected_mode=frozenset({0o700, 0o500})
        )
        identity = (int(opened.st_dev), int(opened.st_ino))
        if expected_identity is not None and identity != expected_identity:
            raise _error("runtime shadow identity changed before cleanup")
        _remove_shadow_contents_fd(descriptor)
        current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if not _same_object(opened, current):
            raise _error("runtime shadow path swapped before removal")
        os.rmdir(path.name, dir_fd=parent_fd)
    except FileNotFoundError:
        if expected_identity is not None:
            raise _error("runtime shadow disappeared before anchored cleanup")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def _process_is_absent(pid: int) -> bool:
    if pid <= 0 or pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
        return False
    return False


def _cleanup_stale_runtime_shadows() -> None:
    parent_fd = _open_shadow_parent()
    try:
        with os.scandir(parent_fd) as iterator:
            names = tuple(entry.name for entry in iterator)
    finally:
        os.close(parent_fd)
    for name in names:
        match = _RUNTIME_SHADOW_NAME_RE.fullmatch(name)
        if match is None or not _process_is_absent(int(match.group(1))):
            continue
        path = RUNTIME_SHADOW_PARENT / name
        try:
            metadata = os.lstat(path)
        except FileNotFoundError:
            continue
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or int(metadata.st_uid) != os.getuid()
            or stat.S_IMODE(metadata.st_mode) not in {0o700, 0o500}
            or _path_is_within(path, PRODUCTION_PRIVATE_ROOT)
            or _path_is_within(PRODUCTION_PRIVATE_ROOT, path)
        ):
            continue
        _anchored_cleanup_runtime_shadow(
            path,
            expected_identity=(int(metadata.st_dev), int(metadata.st_ino)),
        )


def _create_runtime_shadow() -> tuple[Path, int, str, tuple[int, int]]:
    parent_fd = _open_shadow_parent()
    try:
        for _attempt in range(32):
            nonce = secrets.token_hex(16)
            name = f"{RUNTIME_SHADOW_PREFIX}{os.getpid()}-{nonce}"
            try:
                os.mkdir(name, 0o700, dir_fd=parent_fd)
            except FileExistsError:
                continue
            path = RUNTIME_SHADOW_PARENT / name
            descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
            identity = _validate_shadow_container(
                path, descriptor, expected_mode=frozenset({0o700})
            )
            return path, descriptor, nonce, (identity[0], identity[1])
    finally:
        os.close(parent_fd)
    raise _error("cannot allocate a unique isolated runtime shadow")


def _fixed_stdlib_sys_path() -> tuple[str, ...]:
    if sys.version_info[:2] != (3, 13):
        raise _error("isolated runtime requires the fixed CPython 3.13 interpreter")
    base = Path(sys.base_prefix)
    version = "python3.13"
    return (
        str(base / "lib" / "python313.zip"),
        str(base / "lib" / version),
        str(base / "lib" / version / "lib-dynload"),
    )


def _validate_fixed_interpreter() -> None:
    if Path(sys.executable) != FIXED_INTERPRETER:
        raise _error("execution must use the fixed repository virtualenv interpreter")
    try:
        resolved = FIXED_INTERPRETER.resolve(strict=True)
        metadata = os.lstat(resolved)
    except OSError as exc:
        raise _error("fixed interpreter is unavailable") from exc
    if (
        resolved != FIXED_INTERPRETER_RESOLVED
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or int(metadata.st_size) != FIXED_INTERPRETER_SIZE
    ):
        raise _error("fixed interpreter target is not a protected regular file")
    _read_bound_regular_once(
        resolved,
        label="fixed isolated CPython executable",
        max_bytes=FIXED_INTERPRETER_SIZE,
        expected_sha256=FIXED_INTERPRETER_SHA256,
        expected_nlink=1,
    )


def _child_environment(
    *,
    shadow_path: Path,
    shadow_fd: int,
    nonce: str,
    test_probe_path: Path | None = None,
) -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": "/var/empty",
        "TMPDIR": "/private/tmp",
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "__CF_USER_TEXT_ENCODING": f"0x{os.getuid():X}:0x0:0x0",
        ISOLATED_CHILD_ENV: "1",
        ISOLATED_CHILD_SHADOW_FD_ENV: str(shadow_fd),
        ISOLATED_CHILD_SHADOW_PATH_ENV: str(shadow_path),
        ISOLATED_CHILD_NONCE_ENV: nonce,
        ISOLATED_CHILD_TEST_PROBE_ENV: (
            "" if test_probe_path is None else str(test_probe_path)
        ),
    }


def _activate_isolated_child() -> None:
    global _ISOLATED_CHILD_ACTIVE
    global _ACTIVE_RUNTIME_SHADOW_INVENTORY
    global _ACTIVE_RUNTIME_SHADOW_ROOT
    global _ACTIVE_RUNTIME_SHADOW_SITE

    if _ISOLATED_CHILD_ACTIVE:
        raise _error("isolated child bootstrap is one-shot")
    if set(os.environ) != set(_ISOLATED_CHILD_ENV_KEYS):
        raise _error("isolated child environment is not the sanitized exact set")
    if os.environ.get("__CF_USER_TEXT_ENCODING") != (
        f"0x{os.getuid():X}:0x0:0x0"
    ):
        raise _error("isolated child CoreFoundation encoding identity drifted")
    if os.environ.get(ISOLATED_CHILD_ENV) != "1":
        raise _error("isolated child authorization marker is absent")
    _validate_fixed_interpreter()
    if tuple(sys.path) != _fixed_stdlib_sys_path():
        raise _error("isolated child initial sys.path is not the exact stdlib closure")
    nonce = os.environ.get(ISOLATED_CHILD_NONCE_ENV, "")
    raw_fd = os.environ.get(ISOLATED_CHILD_SHADOW_FD_ENV, "")
    raw_path = os.environ.get(ISOLATED_CHILD_SHADOW_PATH_ENV, "")
    if (
        re.fullmatch(r"[0-9]+", raw_fd) is None
        or re.fullmatch(r"[0-9a-f]{32}", nonce) is None
    ):
        raise _error("isolated child shadow capability is malformed")
    shadow_fd = int(raw_fd)
    shadow_path = _absolute_normalized_path(raw_path, "isolated shadow path")
    match = _RUNTIME_SHADOW_NAME_RE.fullmatch(shadow_path.name)
    if (
        match is None
        or match.group(2) != nonce
        or int(match.group(1)) != os.getppid()
    ):
        raise _error("isolated child shadow capability identity mismatch")
    _validate_shadow_container(
        shadow_path, shadow_fd, expected_mode=frozenset({0o700})
    )
    site_path, inventory = _build_runtime_shadow(
        shadow_root_fd=shadow_fd,
        shadow_root_path=shadow_path,
    )
    _preflight_native_shadow(shadow_site=site_path, inventory=inventory)
    sys.path.append(str(site_path))
    if tuple(sys.path) != (*_fixed_stdlib_sys_path(), str(site_path)):
        raise _error("isolated child shadow is not the sole external import root")
    _ACTIVE_RUNTIME_SHADOW_ROOT = shadow_path
    _ACTIVE_RUNTIME_SHADOW_SITE = site_path
    _ACTIVE_RUNTIME_SHADOW_INVENTORY = copy.deepcopy(inventory)
    _ISOLATED_CHILD_ACTIVE = True


class _HandledParentSignal(Exception):
    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"handled parent signal {signum}")


def _run_isolated_parent(
    argv: Sequence[str], *, _test_probe_path: Path | None = None
) -> int:
    _validate_fixed_interpreter()
    _cleanup_stale_runtime_shadows()
    shadow_path, shadow_fd, nonce, identity = _create_runtime_shadow()
    child: subprocess.Popen[bytes] | None = None
    previous_handlers: dict[int, Any] = {}

    def handle_signal(signum: int, _frame: Any) -> None:
        if child is not None and child.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                child.send_signal(signum)
        raise _HandledParentSignal(signum)

    try:
        child = subprocess.Popen(
            [
                str(FIXED_INTERPRETER),
                "-B",
                "-I",
                "-S",
                str(Path(__file__).resolve()),
                *tuple(argv),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/",
            env=_child_environment(
                shadow_path=shadow_path,
                shadow_fd=shadow_fd,
                nonce=nonce,
                test_probe_path=_test_probe_path,
            ),
            pass_fds=(shadow_fd,),
            shell=False,
            close_fds=True,
        )
        if threading_is_main := (
            importlib.import_module("threading").current_thread()
            is importlib.import_module("threading").main_thread()
        ):
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, handle_signal)
        try:
            stdout, stderr = child.communicate(
                timeout=RESOURCE_CONTRACT["total_wall_seconds_max"]
                + RUNTIME_SHADOW_WALL_SECONDS_MAX
                + 30
            )
        except subprocess.TimeoutExpired as exc:
            child.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                child.communicate(timeout=10)
            if child.poll() is None:
                child.kill()
                child.communicate()
            raise _error("isolated child exceeded the total one-shot timeout") from exc
        finally:
            if threading_is_main:
                for signum, previous in previous_handlers.items():
                    signal.signal(signum, previous)
        if len(stdout) > 4 * 1024 * 1024 or len(stderr) > 256 * 1024:
            raise _error("isolated child output exceeded the fixed IPC cap")
        stderr_lines = tuple(stderr.decode("utf-8", errors="replace").splitlines())
        if stderr and stderr_lines != _SANDBOX_ARROW_SYSCTL_WARNINGS:
            raise _error("isolated child emitted unexpected stderr")
        try:
            text = stdout.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise _error("isolated child output is not UTF-8") from exc
        sys.stdout.write(text)
        return int(child.returncode)
    except _HandledParentSignal as exc:
        if child is not None and child.poll() is None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                child.wait(timeout=10)
            if child.poll() is None:
                child.kill()
                child.wait()
        return 128 + exc.signum
    finally:
        os.close(shadow_fd)
        _anchored_cleanup_runtime_shadow(
            shadow_path, expected_identity=identity
        )


def _require_isolated_child(operation: str) -> None:
    if not _ISOLATED_CHILD_ACTIVE:
        raise _error(f"direct nonisolated {operation} is forbidden")


def _run_private_isolated_parquet_probe(path_value: str) -> dict[str, Any]:
    """Private test-only proof that the sealed shadow can read one temp Parquet."""

    _require_isolated_child("test parquet probe")
    path = _absolute_normalized_path(path_value, "test parquet probe path")
    if (
        not path.name.startswith("v44-isolated-probe-")
        or _path_is_within(path, PROJECT_ROOT)
        or _path_is_within(path, PRODUCTION_PRIVATE_ROOT)
    ):
        raise _error("test parquet probe path is outside the private harness scope")
    stable = _read_owner_private_once(
        path, label="test parquet probe", max_bytes=1024 * 1024
    )
    runner_stable = _read_bound_regular_once(
        Path(__file__).resolve(),
        label="isolated runner probe binding",
        max_bytes=16 * 1024 * 1024,
        expected_nlink=1,
    )
    finder = _install_verified_finder((runner_stable,))
    try:
        finder.distribution_roots = _distribution_package_roots()
        parquet = importlib.import_module("pyarrow.parquet")
        table = parquet.read_table(str(stable.path))
        _audit_closed_project_imports(finder)
        _audit_closed_runtime_imports(finder)
        _rehash_runtime_shadow()
        return {
            "mode": "private_isolated_parquet_probe",
            "accepted": True,
            "row_count": int(table.num_rows),
            "column_count": int(table.num_columns),
        }
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        _purge_verified_project_modules(finder)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser(
        "publish", help="publish one future-only manifest-bound strict proof"
    )
    publish.add_argument("--input-manifest", required=True)
    publish.add_argument("--expected-input-manifest-byte-sha256", required=True)
    readback = commands.add_parser(
        "readback", help="read one explicit sealed historical strict bundle"
    )
    readback.add_argument("--bundle-path", required=True)
    readback.add_argument("--expected-readback-report-byte-sha256", required=True)
    readback.add_argument("--expected-readback-report-semantic-sha256", required=True)
    return parser


def run_publish(
    *,
    input_manifest: str | Path,
    expected_input_manifest_byte_sha256: str,
    _test_fault_hook: Callable[[str], None] | None = None,
    _test_race_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Execute the manifest-first publisher state machine."""

    _require_isolated_child("publish")
    manifest_value, stable_manifest = _read_manifest_two_fresh(
        os.fspath(input_manifest),
        expected_byte_sha256=expected_input_manifest_byte_sha256,
    )
    manifest = _stage0_manifest_validate(manifest_value)
    prereg = manifest["preregistration"]
    preregistration_snapshot = _read_private_bundle_raw_snapshot(
        prereg["bundle_path"],
        root_suffix=PREREGISTRATION_ROOT_SUFFIX,
        input_filenames=PREREGISTRATION_INPUT_FILENAMES,
        readback_filename=PREREGISTRATION_READBACK_FILENAME,
        expected_readback_byte_sha256=prereg["readback_byte_sha256"],
        expected_readback_semantic_sha256=prereg["readback_semantic_sha256"],
        artifact_max_bytes=RESOURCE_CONTRACT["prereg_artifact_max_bytes"],
        bundle_max_bytes=RESOURCE_CONTRACT["prereg_bundle_max_bytes"],
        label="future v4.4 preregistration",
    )
    if preregistration_snapshot.values[
        PREREGISTRATION_READBACK_FILENAME
    ].get("cycle_id") != prereg["cycle_id"]:
        raise _error("raw preregistration cycle identity mismatch")
    modules = _load_project_modules_after_stage0(
        manifest=manifest,
        preregistration_snapshot=preregistration_snapshot,
    )
    try:
        # The remaining stages are deliberately below the intake/import boundary.
        return _run_publish_after_stage0(
            manifest=manifest,
            stable_manifest=stable_manifest,
            modules=modules,
            test_fault_hook=_test_fault_hook,
            test_race_hook=_test_race_hook,
        )
    finally:
        _teardown_loaded_modules(modules)


def _run_publish_after_stage0(
    *,
    manifest: Mapping[str, Any],
    stable_manifest: StableBytes,
    modules: LoadedModules,
    test_fault_hook: Callable[[str], None] | None,
    test_race_hook: Callable[[], None] | None,
) -> dict[str, Any]:
    total_started = time.monotonic()

    # Nothing under the fixed publication root is inspected until the explicit
    # future preregistration has been accepted in full.
    accepted = _accept_preregistration(manifest=manifest, modules=modules)
    preflight = _publication_preflight(
        manifest=manifest,
        accepted=accepted,
        modules=modules,
    )
    frozen = _collect_fixed_bindings(
        manifest=manifest,
        stable_manifest=stable_manifest,
        modules=modules,
    )
    if (
        modules.prebound_runtime is None
        or modules.import_guard is None
        or _canonical_bytes_local(modules.prebound_runtime)
        != _canonical_bytes_local(frozen.runtime)
    ):
        raise _error("pre-import runtime/import guard binding is unavailable or drifted")
    _audit_closed_project_imports(modules.import_guard)
    operator_program_set = _verify_pinned_source_definitions(
        manifest=manifest, contract=modules.contract
    )
    input_receipt = modules.contract.build_input_receipt_v4_4(
        manifest=manifest,
        observed_preregistration=manifest["preregistration"],
        observed_code_binding_set=frozen.code,
        runtime_binding=frozen.runtime,
        observed_protected_control_sha256={
            name: stable.byte_sha256 for name, stable in frozen.protected
        },
    )

    stack = _lazy_data_stack()
    _verify_loaded_stack_identity(stack=stack, frozen=frozen, manifest=manifest)
    _audit_closed_project_imports(modules.import_guard)
    _audit_closed_runtime_imports(modules.import_guard)
    _rehash_runtime_shadow()
    first = _execute_data_pass(
        pass_id="fresh_pass_1",
        manifest=manifest,
        modules=modules,
        stack=stack,
        operator_program_set=operator_program_set,
    )
    gc.collect()
    second = _execute_data_pass(
        pass_id="fresh_pass_2",
        manifest=manifest,
        modules=modules,
        stack=stack,
        operator_program_set=operator_program_set,
    )
    gc.collect()
    _assert_fresh_pass_equivalence(first, second)

    data_receipt = modules.contract.build_data_field_receipt_v4_4(
        manifest=manifest,
        input_receipt=input_receipt,
        source_calendar_open_sessions=first.block_manifest["source_calendar"],
        historical_symbol_axis=first.historical_symbol_axis,
        pit_membership_contract=first.pit_membership_contract,
        pit_mask_descriptor=first.pit_mask_descriptor,
        block_manifest=first.block_manifest,
        field_missing_counts=first.field_missing_counts,
        bars_outside_pit_interval_count=(
            first.bars_outside_pit_interval_count
        ),
        ignored_pre_analysis_row_count=first.ignored_pre_analysis_row_count,
        outside_pit_non_null_counts=first.outside_pit_non_null_counts,
        projected_row_count_per_pass=first.dense_projected_row_count,
    )
    raw_matrix_descriptors = first.engine_matrix_descriptors[0][
        "raw_matrix_descriptors"
    ]
    candidate_non_null_masks = (
        modules.contract.build_candidate_non_null_mask_set_v4_4(
            proof_pit_mask=first.proof_pit_mask_descriptor,
            candidate_masks=first.candidate_non_null_mask_descriptors,
            raw_matrix_descriptors=raw_matrix_descriptors,
            data_field_receipt=data_receipt,
        )
    )
    collections = tuple(
        modules.contract.build_collection_descriptor_v4_4(
            pass_id=evidence.pass_id,
            data_field_receipt=data_receipt,
            input_matrix_descriptors=evidence.input_matrix_descriptors,
        )
        for evidence in (first, second)
    )
    engine_results = tuple(
        tuple(
            modules.contract.build_engine_pass_result_v4_4(
                pass_id=evidence.pass_id,
                engine_id=descriptor["engine_id"],
                collection_sha256=collection["collection_sha256"],
                data_field_receipt=data_receipt,
                operator_program_set=operator_program_set,
                proof_pit_mask=first.proof_pit_mask_descriptor,
                candidate_non_null_masks=candidate_non_null_masks,
                raw_matrix_descriptors=descriptor[
                    "raw_matrix_descriptors"
                ],
                adjusted_matrix_descriptors=descriptor[
                    "adjusted_matrix_descriptors"
                ],
            )
            for descriptor in evidence.engine_matrix_descriptors
        )
        for evidence, collection in zip(
            (first, second), collections, strict=True
        )
    )
    two_pass_receipt = (
        modules.contract.build_two_pass_equivalence_receipt_v4_4(
            manifest=manifest,
            input_receipt=input_receipt,
            data_field_receipt=data_receipt,
            proof_pit_mask=first.proof_pit_mask_descriptor,
            candidate_non_null_masks=candidate_non_null_masks,
            collections=collections,
            engine_results=engine_results,
        )
    )
    proof = modules.contract.build_proof_v4_4(
        manifest=manifest,
        input_receipt=input_receipt,
        data_field_receipt=data_receipt,
        two_pass_equivalence_receipt=two_pass_receipt,
    )
    artifacts = {
        modules.contract.INPUT_MANIFEST_FILENAME: copy.deepcopy(dict(manifest)),
        modules.contract.INPUT_RECEIPT_FILENAME: input_receipt,
        modules.contract.DATA_FIELD_RECEIPT_FILENAME: data_receipt,
        modules.contract.TWO_PASS_EQUIVALENCE_RECEIPT_FILENAME: two_pass_receipt,
        modules.contract.PROOF_FILENAME: proof,
    }
    bundle_contract = modules.contract.private_bundle_contract_v4_4()

    def locked_revalidation() -> None:
        _locked_precommit_revalidate(
            manifest=manifest,
            stable_manifest=stable_manifest,
            modules=modules,
            stack=stack,
            frozen=frozen,
            preflight=preflight,
            reference_pass=second,
            reference_operator_program_set=operator_program_set,
            total_started=total_started,
        )

    published = modules.private_io.publish_private_bundle(
        private_root=preflight.root,
        run_id=preflight.cycle_id,
        artifacts=artifacts,
        contract=bundle_contract,
        revalidate_inputs=locked_revalidation,
        _test_fault_hook=test_fault_hook,
        _test_race_hook=test_race_hook,
    )
    independently_read = modules.private_io.readback_private_bundle(
        Path(published["bundle_path"]),
        contract=bundle_contract,
    )
    if (
        independently_read.get("accepted") is not True
        or independently_read["artifacts"] != published["artifacts"]
        or independently_read["readback_report"]
        != published["readback_report"]
    ):
        raise _error("independent post-publication sealed readback differed")
    report_filename = modules.contract.READBACK_FILENAME
    report_descriptor = independently_read["artifact_descriptors"][
        report_filename
    ]
    report = independently_read["readback_report"]
    diagnostics = _postcommit_protected_diagnostics(frozen)
    return {
        "mode": "publish",
        "accepted": True,
        "cycle_id": manifest["cycle_id"],
        "cutoff": manifest["cutoff"],
        "snapshot_id": manifest["snapshot_id"],
        "bundle_path": independently_read["bundle_path"],
        "artifact_count": len(modules.contract.BUNDLE_FILENAMES),
        "readback_report_byte_sha256": report_descriptor["byte_sha256"],
        "readback_report_semantic_sha256": report[
            "artifact_semantic_sha256"
        ],
        "readback_scope": report["readback_scope"],
        "double_fresh_read_reproducibility": True,
        "independent_engine_equivalence": True,
        "signal_computability_proven": True,
        "pass_elapsed_seconds": {
            first.pass_id: first.elapsed_seconds,
            second.pass_id: second.elapsed_seconds,
        },
        "peak_rss_bytes": max(first.peak_rss_bytes, second.peak_rss_bytes),
        "total_elapsed_seconds": time.monotonic() - total_started,
        "postcommit_protected_control_diagnostics": diagnostics,
    }


def run_readback(
    *,
    bundle_path: str | Path,
    expected_readback_report_byte_sha256: str,
    expected_readback_report_semantic_sha256: str,
) -> dict[str, Any]:
    _require_isolated_child("readback")
    path = _absolute_normalized_path(os.fspath(bundle_path), "bundle_path")
    expected_byte = _sha256(
        expected_readback_report_byte_sha256, "expected readback report byte SHA"
    )
    expected_semantic = _sha256(
        expected_readback_report_semantic_sha256,
        "expected readback report semantic SHA",
    )
    raw_before = _read_private_bundle_raw_snapshot(
        str(path),
        root_suffix=ROOT_SUFFIX,
        input_filenames=STRICT_INPUT_FILENAMES,
        readback_filename=STRICT_READBACK_FILENAME,
        expected_readback_byte_sha256=expected_byte,
        expected_readback_semantic_sha256=expected_semantic,
        artifact_max_bytes=RESOURCE_CONTRACT["strict_artifact_max_bytes"],
        bundle_max_bytes=RESOURCE_CONTRACT["strict_bundle_max_bytes"],
        label="historical strict sealed bundle",
    )
    manifest = _stage0_manifest_validate(
        raw_before.values[STRICT_INPUT_FILENAMES[0]]
    )
    modules = _load_readback_modules_after_stage0(manifest=manifest)
    try:
        if (
            tuple(modules.contract.INPUT_FILENAMES) != STRICT_INPUT_FILENAMES
            or modules.contract.READBACK_FILENAME != STRICT_READBACK_FILENAME
            or tuple(modules.contract.ROOT_SUFFIX) != ROOT_SUFFIX
        ):
            raise _error("loaded readback contract inventory differs from fixed runner")
        _audit_closed_project_imports(modules.import_guard)
        result = modules.private_io.readback_private_bundle(
            path,
            contract=modules.contract.private_bundle_contract_v4_4(),
        )
        report = result["readback_report"]
        actual_byte = result["artifact_descriptors"][
            STRICT_READBACK_FILENAME
        ]["byte_sha256"]
        if actual_byte != expected_byte:
            raise _error("historical readback report byte SHA mismatch")
        if report.get("artifact_semantic_sha256") != expected_semantic:
            raise _error("historical readback report semantic SHA mismatch")
        if report.get("readback_scope") != "SEALED_BUNDLE_GRAPH_ONLY":
            raise _error("historical readback scope must be SEALED_BUNDLE_GRAPH_ONLY")
        for field in (
            "external_predecessor_revalidated",
            "immutable_source_revalidated",
            "protected_controls_revalidated",
            "external_state_claimed",
        ):
            if report.get(field) is not False:
                raise _error(f"historical readback must keep {field}=false")
        raw_after = _read_private_bundle_raw_snapshot(
            str(path),
            root_suffix=ROOT_SUFFIX,
            input_filenames=STRICT_INPUT_FILENAMES,
            readback_filename=STRICT_READBACK_FILENAME,
            expected_readback_byte_sha256=expected_byte,
            expected_readback_semantic_sha256=expected_semantic,
            artifact_max_bytes=RESOURCE_CONTRACT["strict_artifact_max_bytes"],
            bundle_max_bytes=RESOURCE_CONTRACT["strict_bundle_max_bytes"],
            label="historical strict sealed bundle after readback",
        )
        for filename in (*STRICT_INPUT_FILENAMES, STRICT_READBACK_FILENAME):
            before = raw_before.files[filename]
            after = raw_after.files[filename]
            if before.signature != after.signature or before.raw != after.raw:
                raise _error("historical strict bundle drifted during readback")
        _audit_closed_project_imports(modules.import_guard)
        return {
            "mode": "readback",
            "accepted": True,
            "bundle_path": result["bundle_path"],
            "readback_report_byte_sha256": expected_byte,
            "readback_report_semantic_sha256": expected_semantic,
            "readback_scope": "SEALED_BUNDLE_GRAPH_ONLY",
            "external_predecessor_revalidated": False,
            "immutable_source_revalidated": False,
            "protected_controls_revalidated": False,
            "external_state_claimed": False,
        }
    finally:
        _teardown_loaded_modules(modules)


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    if os.environ.get(ISOLATED_CHILD_ENV) != "1":
        try:
            return _run_isolated_parent(raw_argv)
        except Exception as exc:
            print(json.dumps({"accepted": False, "error": str(exc)}, sort_keys=True))
            return 2
    try:
        _activate_isolated_child()
        probe_path = os.environ.get(ISOLATED_CHILD_TEST_PROBE_ENV, "")
        if probe_path:
            result = _run_private_isolated_parquet_probe(probe_path)
            _rehash_runtime_shadow()
            print(json.dumps(result, sort_keys=True))
            return 0
        parser = build_parser()
        args = parser.parse_args(raw_argv)
        if args.command == "publish":
            result = run_publish(
                input_manifest=args.input_manifest,
                expected_input_manifest_byte_sha256=(
                    args.expected_input_manifest_byte_sha256
                ),
            )
        else:
            result = run_readback(
                bundle_path=args.bundle_path,
                expected_readback_report_byte_sha256=(
                    args.expected_readback_report_byte_sha256
                ),
                expected_readback_report_semantic_sha256=(
                    args.expected_readback_report_semantic_sha256
                ),
            )
        _rehash_runtime_shadow()
    except Exception as exc:
        if _ISOLATED_CHILD_ACTIVE:
            with contextlib.suppress(Exception):
                _rehash_runtime_shadow()
        print(json.dumps({"accepted": False, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
