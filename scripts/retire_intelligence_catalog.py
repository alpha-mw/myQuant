"""Fail-closed activation gate for retiring the Intelligence catalog entry.

The production CLI intentionally exposes no path or digest overrides.  Tests may
inject an isolated repository and fixture digests through
``retire_intelligence_catalog``; production always uses the canonical constants
defined in this module.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_investor.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.versioning import (
    ARCHITECTURE_VERSION,
    BRANCH_SCHEMA_VERSION,
    LIKELIHOOD_SCHEMA_VERSION,
    REPORT_PROTOCOL_VERSION,
)

CONFIRM_TOKEN = "RETIRE_INTELLIGENCE_CATALOG_V14"
ACTIVATION_RECEIPT_SCHEMA_VERSION = "myquant.intelligence-retirement-activation-receipt.v14"
APPROVAL_ATTESTATION_SCHEMA_VERSION = "myquant.maxwell-intelligence-retirement-approval.v14"
CANONICAL_LIKELIHOOD_ORDER: tuple[str, ...] = ("quant", "fundamental")

PRODUCTION_REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CATALOG_PATH = Path("data/parquet/cn/_catalog.json")
CANONICAL_LATEST_PATH = Path("data/parquet/cn/_latest.json")
CANONICAL_MART_PATH = Path("data/parquet/cn/intelligence_daily/part.parquet")
CANONICAL_CATALOG_LOCK_PATH = Path("data/parquet/cn/._catalog.json.intelligence-retirement.lock")
CANONICAL_ACTIVATION_RECEIPT_PATH = Path(
    "reports/intelligence_retirement/v14_activation_receipt.json"
)
CANONICAL_APPROVAL_ATTESTATION_PATH = (
    Path.home() / ".config" / "myquant" / "approvals" / "v14_intelligence_catalog_retirement.json"
)

EXPECTED_PRE_CUTOVER_CATALOG_SHA256 = (
    "f3b45a0bf9d141c13c03b1f19fe6a46ed42937a0979098a2b7b8190fb5fee2de"
)
EXPECTED_LATEST_SHA256 = "1c9e7af7c764d5c90d5d660c20572d91183a3c40de54cd84e1ae90e1d764a17d"
EXPECTED_MART_SHA256 = "aa81955db7c7692d2f908e50fa18a850efc33c49d11f048cdea862e253043b07"
EXPECTED_HISTORICAL_REPORT_TREES: dict[str, dict[str, Any]] = {
    "reports/daily": {
        "file_count": 8,
        "sha256": "e5b7459a38234225daaf51522c7df885ac7e9496fc6dada1717453adb6adf4de",
    },
    "reports/branch_readiness": {
        "file_count": 2_016,
        "sha256": "b256e1c7eb5b1c0abbc5e926ef208d6e01b5484ae8ca1a05016a4154b26e39a2",
    },
    "reports/branch_readiness_clean_parquet_smoke": {
        "file_count": 3,
        "sha256": "f4b5ae8bec66aafcd3de8d2a55b6ad7f99f4a059a040e879c3edb626a8671cc6",
    },
    "reports/holdings_dag_review": {
        "file_count": 6,
        "sha256": "df00f19ee8c2bfa1cb4770ebc30d07b5ad01cbfa6a347ed221603beebae4269f",
    },
}

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_NONCE_RE = re.compile(r"^[0-9a-f]{64,128}$")
_RUNTIME_ATTESTATION_MAX_AGE_SECONDS = 300
_RUNTIME_QUIESCENCE_SCOPE = (
    "daily_daemon",
    "web_runtime",
    "market_runtime",
    "research_runtime",
    "retired_source_and_pyc",
)
_RETIRED_SOURCE_PATHS = (
    "quant_investor/agents/intelligence_agent.py",
    "quant_investor/agents/subagents/intelligence_agent.py",
    "quant_investor/ensemble_judge.py",
    "quant_investor/market/intelligence_mart.py",
    "quant_investor/monitoring/intelligence_monitor.py",
)
_RUNTIME_PROCESS_PATTERNS = {
    "daily_daemon": re.compile(
        r"(?:daily_runner(?:\.py)?|quant_investor\.automation\.daily_runner).*--daemon",
        re.IGNORECASE,
    ),
    "web_runtime": re.compile(
        r"(?:uvicorn|gunicorn|python(?:3)?\s+-m\s+web).*(?:web\.main|app:app|web)",
        re.IGNORECASE,
    ),
    "market_runtime": re.compile(
        r"(?:quant-investor|quant_investor\.cli\.main).*\bmarket\s+"
        r"(?:run|maintain|analyze|backtest|data-governance)\b",
        re.IGNORECASE,
    ),
    "research_runtime": re.compile(
        r"(?:quant-investor|quant_investor\.cli\.main).*\bresearch\s+run\b",
        re.IGNORECASE,
    ),
}
_REQUIRED_GATES = (
    "code_gate_passed",
    "replay_gate_passed",
    "no_new_buy",
    "risk_guard_not_weakened",
    "parallel_work_integrated",
    "runtime_quiesced",
    "retired_pyc_removed",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_with_sha256(path: Path) -> tuple[bytes, str]:
    payload = path.read_bytes()
    return payload, _sha256_bytes(payload)


class CatalogLockBusy(RuntimeError):
    """Raised when another catalog activation transaction owns the lock."""


def _acquire_catalog_lock(lock_path: Path) -> int:
    """Acquire the repository's sole catalog-writer lock without waiting."""

    flags = os.O_CREAT | os.O_RDWR
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise CatalogLockBusy(f"catalog lock open failed: {exc}") from exc
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise CatalogLockBusy("catalog lock is not a regular file")
        os.fchmod(fd, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError, CatalogLockBusy) as exc:
        os.close(fd)
        raise CatalogLockBusy(f"catalog lock unavailable: {exc}") from exc
    return fd


def _release_catalog_lock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _repo_relative(repo_root: Path, path: Path) -> str:
    root = repo_root.resolve(strict=True)
    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"path must stay inside repository: {path}") from exc


def _validate_repo_paths(
    *,
    repo_root: Path,
    paths: Mapping[str, Path],
    report_tree_paths: Mapping[str, Path],
    activation_receipt_path: Path,
    approval_attestation_path: Path,
) -> list[str]:
    blockers: list[str] = []
    try:
        root = repo_root.resolve(strict=True)
    except OSError as exc:
        return [f"repository_root_unreadable:{exc}"]

    for name, path in {
        **paths,
        **report_tree_paths,
        "activation_receipt": activation_receipt_path,
    }.items():
        try:
            path.resolve(strict=False).relative_to(root)
        except (OSError, RuntimeError, ValueError):
            blockers.append(f"{name}_path_outside_repository")
        if path.is_symlink():
            blockers.append(f"{name}_symlink_not_allowed")

    try:
        approval_attestation_path.resolve(strict=False).relative_to(root)
    except (OSError, RuntimeError, ValueError):
        pass
    else:
        blockers.append("approval_attestation_must_be_repo_external")
    if approval_attestation_path.is_symlink():
        blockers.append("approval_attestation_symlink_not_allowed")
    return blockers


def _tree_digest(repo_root: Path, tree_root: Path) -> dict[str, Any]:
    """Match ``find ... -type f -print0 | sort -z | xargs shasum``.

    The final digest is over lines of ``<file-sha>  <repo-relative-path>\n`` in
    bytewise path order. Symlinks are not followed or counted, matching
    ``find`` without ``-L``.
    """

    root = repo_root.resolve(strict=True)
    tree = tree_root.resolve(strict=True)
    tree.relative_to(root)
    files: list[tuple[bytes, Path, str]] = []
    for current_root, dirnames, filenames in os.walk(tree, followlinks=False):
        current = Path(current_root)
        dirnames[:] = [name for name in dirnames if not (current / name).is_symlink()]
        for filename in filenames:
            path = current / filename
            mode = path.lstat().st_mode
            if not stat.S_ISREG(mode):
                continue
            relative = path.relative_to(root).as_posix()
            files.append((relative.encode("utf-8"), path, relative))
    files.sort(key=lambda item: item[0])

    digest_input = bytearray()
    for _sort_key, path, relative in files:
        file_sha = _sha256_bytes(path.read_bytes())
        digest_input.extend(f"{file_sha}  {relative}\n".encode("utf-8"))
    return {
        "file_count": len(files),
        "sha256": _sha256_bytes(bytes(digest_input)),
    }


def _capture_immutable_evidence(
    *,
    repo_root: Path,
    paths: Mapping[str, Path],
    report_tree_paths: Mapping[str, Path],
) -> tuple[dict[str, Any], list[str]]:
    snapshot: dict[str, Any] = {"files": {}, "report_trees": {}}
    blockers: list[str] = []
    for name, path in paths.items():
        try:
            _payload, digest = _read_with_sha256(path)
            snapshot["files"][name] = {"sha256": digest}
        except OSError as exc:
            blockers.append(f"{name}_unreadable:{exc}")
    for label, path in report_tree_paths.items():
        try:
            snapshot["report_trees"][label] = _tree_digest(repo_root, path)
        except (OSError, RuntimeError, ValueError) as exc:
            blockers.append(f"report_tree_unreadable:{label}:{exc}")
    return snapshot, blockers


def _expected_evidence(
    *,
    expected_catalog_sha256: str,
    expected_latest_sha256: str,
    expected_mart_sha256: str,
    expected_report_trees: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "files": {
            "catalog": {"sha256": str(expected_catalog_sha256)},
            "latest": {"sha256": str(expected_latest_sha256)},
            "mart": {"sha256": str(expected_mart_sha256)},
        },
        "report_trees": {
            str(label): {
                "file_count": int(values["file_count"]),
                "sha256": str(values["sha256"]),
            }
            for label, values in expected_report_trees.items()
        },
    }


def _evidence_mismatch_blockers(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    phase: str,
) -> list[str]:
    blockers: list[str] = []
    for name, expected_entry in expected["files"].items():
        if actual.get("files", {}).get(name) != expected_entry:
            blockers.append(f"{phase}_{name}_sha256_mismatch")
    actual_trees = actual.get("report_trees", {})
    expected_trees = expected["report_trees"]
    if set(actual_trees) != set(expected_trees):
        blockers.append(f"{phase}_report_tree_set_mismatch")
    for label, expected_entry in expected_trees.items():
        if actual_trees.get(label) != expected_entry:
            blockers.append(f"{phase}_report_tree_mismatch:{label}")
    return blockers


def _retired_catalog_payload(catalog: dict[str, Any]) -> tuple[dict[str, Any], bool, bool]:
    required_tables = catalog.get("required_tables")
    tables = catalog.get("tables")
    if not isinstance(required_tables, list):
        raise ValueError("catalog required_tables must be a list")
    if not isinstance(tables, dict):
        raise ValueError("catalog tables must be an object")
    if required_tables.count("intelligence_daily") != 1:
        raise ValueError("catalog must contain exactly one required intelligence_daily entry")
    if "intelligence_daily" not in tables:
        raise ValueError("catalog tables must contain intelligence_daily before retirement")

    retired = dict(catalog)
    retired["required_tables"] = [item for item in required_tables if item != "intelligence_daily"]
    retired_tables = dict(tables)
    del retired_tables["intelligence_daily"]
    retired["tables"] = retired_tables
    return retired, True, True


def _atomic_write(path: Path, payload: bytes) -> None:
    target_mode = stat.S_IMODE(path.stat().st_mode)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.v14-intelligence-retirement.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, target_mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _scan_retired_residue(repo_root: Path) -> list[str]:
    residue: set[str] = set()
    root = repo_root.resolve(strict=True)
    for relative_text in _RETIRED_SOURCE_PATHS:
        source = root / relative_text
        if os.path.lexists(source):
            residue.add(relative_text)
        direct_bytecode = source.with_suffix(".pyc")
        if os.path.lexists(direct_bytecode):
            residue.add(direct_bytecode.relative_to(root).as_posix())
        cache_root = source.parent / "__pycache__"
        if cache_root.is_dir():
            for bytecode in cache_root.glob(f"{source.stem}.*.pyc"):
                residue.add(bytecode.relative_to(root).as_posix())
    return sorted(residue)


def _scan_active_runtime_processes() -> tuple[list[dict[str, Any]], list[str]]:
    try:
        output = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,command="],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return [], [f"runtime_process_scan_failed:{exc}"]

    active: list[dict[str, Any]] = []
    own_pid = os.getpid()
    for raw_line in output.splitlines():
        parts = raw_line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
            parent_pid = int(parts[1])
        except ValueError:
            continue
        if pid == own_pid:
            continue
        command = parts[2]
        for scope, pattern in _RUNTIME_PROCESS_PATTERNS.items():
            if pattern.search(command):
                active.append(
                    {
                        "scope": scope,
                        "pid": pid,
                        "parent_pid": parent_pid,
                        "command": command,
                    }
                )
                break
    active.sort(key=lambda item: (item["scope"], item["pid"]))
    return active, []


def _capture_runtime_state(repo_root: Path) -> tuple[dict[str, Any], list[str]]:
    active_processes, scan_errors = _scan_active_runtime_processes()
    try:
        residue = _scan_retired_residue(repo_root)
    except (OSError, RuntimeError, ValueError) as exc:
        residue = []
        scan_errors.append(f"retired_residue_scan_failed:{exc}")
    return {
        "active_processes": active_processes,
        "residue": residue,
    }, scan_errors


def _runtime_state_blockers(state: Mapping[str, Any], *, phase: str) -> list[str]:
    blockers: list[str] = []
    if state.get("active_processes"):
        blockers.append(f"{phase}_runtime_processes_active")
    if state.get("residue"):
        blockers.append(f"{phase}_retired_source_or_pyc_residue")
    return blockers


def _parse_utc_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text.endswith("Z"):
        return None
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        return None
    return parsed.astimezone(timezone.utc)


def _load_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    payload, digest = _read_with_sha256(path)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, digest


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> list[str]:
    actual = set(value)
    if actual == expected:
        return []
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    return [f"{label}_keys_mismatch:missing={missing}:unexpected={unexpected}"]


def _validate_activation_receipt(
    receipt: Mapping[str, Any],
    *,
    repo_root: Path,
    paths: Mapping[str, Path],
    candidate_commit: str,
    expected: Mapping[str, Any],
) -> list[str]:
    blockers = _require_exact_keys(
        receipt,
        {
            "schema_version",
            "architecture_version",
            "branch_schema_version",
            "likelihood_schema_version",
            "report_protocol_version",
            "canonical_branch_order",
            "canonical_likelihood_order",
            "candidate_commit",
            "gates",
            "evidence",
        },
        label="activation_receipt",
    )
    exact_values = {
        "schema_version": ACTIVATION_RECEIPT_SCHEMA_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "branch_schema_version": BRANCH_SCHEMA_VERSION,
        "likelihood_schema_version": LIKELIHOOD_SCHEMA_VERSION,
        "report_protocol_version": REPORT_PROTOCOL_VERSION,
        "candidate_commit": candidate_commit,
    }
    for field, required in exact_values.items():
        if receipt.get(field) != required:
            blockers.append(f"activation_receipt_{field}_mismatch")
    if receipt.get("canonical_branch_order") != list(CANONICAL_BRANCH_ORDER):
        blockers.append("activation_receipt_branch_order_mismatch")
    if receipt.get("canonical_likelihood_order") != list(CANONICAL_LIKELIHOOD_ORDER):
        blockers.append("activation_receipt_likelihood_order_mismatch")

    gates = receipt.get("gates")
    if not isinstance(gates, Mapping):
        blockers.append("activation_receipt_gates_not_object")
    else:
        blockers.extend(
            _require_exact_keys(gates, set(_REQUIRED_GATES), label="activation_receipt_gates")
        )
        for gate in _REQUIRED_GATES:
            if gates.get(gate) is not True:
                blockers.append(f"activation_receipt_gate_not_passed:{gate}")

    evidence = receipt.get("evidence")
    if not isinstance(evidence, Mapping):
        blockers.append("activation_receipt_evidence_not_object")
        return blockers
    blockers.extend(
        _require_exact_keys(
            evidence,
            {"catalog", "latest", "mart", "report_trees"},
            label="activation_receipt_evidence",
        )
    )
    for name in ("catalog", "latest", "mart"):
        entry = evidence.get(name)
        if not isinstance(entry, Mapping):
            blockers.append(f"activation_receipt_evidence_{name}_not_object")
            continue
        blockers.extend(
            _require_exact_keys(
                entry, {"path", "sha256"}, label=f"activation_receipt_evidence_{name}"
            )
        )
        try:
            expected_path = _repo_relative(repo_root, paths[name])
        except ValueError:
            expected_path = ""
        if entry.get("path") != expected_path:
            blockers.append(f"activation_receipt_evidence_{name}_path_mismatch")
        if entry.get("sha256") != expected["files"][name]["sha256"]:
            blockers.append(f"activation_receipt_evidence_{name}_sha256_mismatch")

    report_trees = evidence.get("report_trees")
    if not isinstance(report_trees, Mapping):
        blockers.append("activation_receipt_report_trees_not_object")
    elif dict(report_trees) != dict(expected["report_trees"]):
        blockers.append("activation_receipt_report_trees_mismatch")
    return blockers


def _validate_approval_attestation(
    approval: Mapping[str, Any],
    *,
    receipt_sha256: str,
    candidate_commit: str,
    expected_catalog_sha256: str,
) -> list[str]:
    blockers = _require_exact_keys(
        approval,
        {
            "schema_version",
            "approved_by",
            "approval_scope",
            "nonce",
            "explicit_confirmation",
            "activation_receipt_sha256",
            "candidate_commit",
            "catalog_sha256",
            "runtime_quiescence",
        },
        label="approval_attestation",
    )
    required = {
        "schema_version": APPROVAL_ATTESTATION_SCHEMA_VERSION,
        "approved_by": "Maxwell",
        "approval_scope": CONFIRM_TOKEN,
        "explicit_confirmation": True,
        "activation_receipt_sha256": receipt_sha256,
        "candidate_commit": candidate_commit,
        "catalog_sha256": expected_catalog_sha256,
    }
    for field, expected_value in required.items():
        if approval.get(field) != expected_value:
            blockers.append(f"approval_attestation_{field}_mismatch")
    nonce = approval.get("nonce")
    if not isinstance(nonce, str) or _NONCE_RE.fullmatch(nonce) is None:
        blockers.append("approval_attestation_nonce_invalid")

    runtime = approval.get("runtime_quiescence")
    if not isinstance(runtime, Mapping):
        blockers.append("approval_attestation_runtime_quiescence_not_object")
        return blockers
    blockers.extend(
        _require_exact_keys(
            runtime,
            {
                "observed_at_utc",
                "max_age_seconds",
                "candidate_commit",
                "catalog_sha256",
                "scope",
                "active_processes",
                "residue",
            },
            label="approval_attestation_runtime_quiescence",
        )
    )
    if runtime.get("max_age_seconds") != _RUNTIME_ATTESTATION_MAX_AGE_SECONDS:
        blockers.append("approval_attestation_runtime_max_age_mismatch")
    if runtime.get("candidate_commit") != candidate_commit:
        blockers.append("approval_attestation_runtime_candidate_commit_mismatch")
    if runtime.get("catalog_sha256") != expected_catalog_sha256:
        blockers.append("approval_attestation_runtime_catalog_sha256_mismatch")
    if runtime.get("scope") != list(_RUNTIME_QUIESCENCE_SCOPE):
        blockers.append("approval_attestation_runtime_scope_mismatch")
    if runtime.get("active_processes") != []:
        blockers.append("approval_attestation_runtime_active_processes_not_empty")
    if runtime.get("residue") != []:
        blockers.append("approval_attestation_runtime_residue_not_empty")
    observed_at = _parse_utc_timestamp(runtime.get("observed_at_utc"))
    if observed_at is None:
        blockers.append("approval_attestation_runtime_observed_at_invalid")
    else:
        age_seconds = (datetime.now(timezone.utc) - observed_at).total_seconds()
        if age_seconds < -30 or age_seconds > _RUNTIME_ATTESTATION_MAX_AGE_SECONDS:
            blockers.append("approval_attestation_runtime_observation_stale")
    return blockers


def _load_and_validate_activation_evidence(
    *,
    activation_receipt_path: Path,
    approval_attestation_path: Path,
    repo_root: Path,
    paths: Mapping[str, Path],
    candidate_commit: str,
    expected: Mapping[str, Any],
) -> tuple[dict[str, str], list[str]]:
    digests: dict[str, str] = {}
    blockers: list[str] = []
    try:
        receipt, receipt_sha = _load_json_object(
            activation_receipt_path, label="activation receipt"
        )
        digests["activation_receipt"] = receipt_sha
    except (OSError, ValueError) as exc:
        return digests, [f"activation_receipt_unreadable:{exc}"]
    blockers.extend(
        _validate_activation_receipt(
            receipt,
            repo_root=repo_root,
            paths=paths,
            candidate_commit=candidate_commit,
            expected=expected,
        )
    )

    try:
        mode = stat.S_IMODE(approval_attestation_path.stat().st_mode)
        if mode != 0o600:
            blockers.append("approval_attestation_permissions_must_be_0600")
        approval, approval_sha = _load_json_object(
            approval_attestation_path, label="approval attestation"
        )
        digests["approval_attestation"] = approval_sha
    except (OSError, ValueError) as exc:
        blockers.append(f"approval_attestation_unreadable:{exc}")
        return digests, blockers
    blockers.extend(
        _validate_approval_attestation(
            approval,
            receipt_sha256=receipt_sha,
            candidate_commit=candidate_commit,
            expected_catalog_sha256=expected["files"]["catalog"]["sha256"],
        )
    )
    return digests, blockers


def _restore_catalog(
    *, path: Path, original_payload: bytes, expected_sha256: str
) -> tuple[bool, str]:
    try:
        _atomic_write(path, original_payload)
        restored_sha = _read_with_sha256(path)[1]
    except OSError as exc:
        return False, f"rollback_io_failed:{exc}"
    if restored_sha != expected_sha256:
        return False, f"rollback_sha256_mismatch:{restored_sha}"
    return True, restored_sha


def _fresh_reader_retirement_probe(repo_root: Path) -> tuple[bool, str]:
    """Prove a newly constructed strict reader cannot resolve the retired table."""

    from quant_investor.market.market_data_reader import (
        MarketDataReader,
        MarketDataUnavailableError,
    )

    reader = MarketDataReader(market="CN", data_root=repo_root / "data", mode_policy="strict")
    try:
        reader.read_table("intelligence_daily")
    except MarketDataUnavailableError as exc:
        detail = str(exc)
        expected = "Parquet logical table not found in catalog: intelligence_daily"
        return detail == expected, detail
    except Exception as exc:  # fail closed on an unrelated snapshot/read failure
        return False, f"unexpected_error:{type(exc).__name__}:{exc}"
    return False, "retired logical table remained readable"


def _retire_intelligence_catalog_transaction(
    *,
    catalog_path: Path,
    latest_path: Path,
    mart_path: Path,
    report_tree_paths: Mapping[str, Path],
    activation_receipt_path: Path,
    approval_attestation_path: Path,
    repo_root: Path,
    candidate_commit: str,
    candidate_worktree_clean: bool,
    expected_catalog_sha256: str = EXPECTED_PRE_CUTOVER_CATALOG_SHA256,
    expected_latest_sha256: str = EXPECTED_LATEST_SHA256,
    expected_mart_sha256: str = EXPECTED_MART_SHA256,
    expected_report_trees: Mapping[str, Mapping[str, Any]] = EXPECTED_HISTORICAL_REPORT_TREES,
    apply: bool = False,
    confirm_token: str | None = None,
) -> dict[str, Any]:
    """Remove only the catalog declaration after every activation gate passes."""

    paths = {
        "catalog": Path(catalog_path),
        "latest": Path(latest_path),
        "mart": Path(mart_path),
    }
    report_tree_paths = {str(label): Path(path) for label, path in report_tree_paths.items()}
    expected = _expected_evidence(
        expected_catalog_sha256=expected_catalog_sha256,
        expected_latest_sha256=expected_latest_sha256,
        expected_mart_sha256=expected_mart_sha256,
        expected_report_trees=expected_report_trees,
    )
    blockers = _validate_repo_paths(
        repo_root=Path(repo_root),
        paths=paths,
        report_tree_paths=report_tree_paths,
        activation_receipt_path=Path(activation_receipt_path),
        approval_attestation_path=Path(approval_attestation_path),
    )
    if set(report_tree_paths) != set(expected["report_trees"]):
        blockers.append("report_tree_path_set_mismatch")
    for digest in (
        expected_catalog_sha256,
        expected_latest_sha256,
        expected_mart_sha256,
        *(entry["sha256"] for entry in expected["report_trees"].values()),
    ):
        if _DIGEST_RE.fullmatch(str(digest)) is None:
            blockers.append("invalid_expected_sha256")
            break

    report: dict[str, Any] = {
        "schema_version": "myquant.intelligence-catalog-retirement.v14",
        "apply_requested": bool(apply),
        "candidate_commit": str(candidate_commit),
        "candidate_worktree_clean": bool(candidate_worktree_clean),
        "catalog_path": str(paths["catalog"]),
        "latest_path": str(paths["latest"]),
        "mart_path": str(paths["mart"]),
        "activation_receipt_path": str(activation_receipt_path),
        "approval_attestation_path": str(approval_attestation_path),
        "expected_evidence": expected,
        "blockers": blockers,
        "removed_required_table": False,
        "removed_table_entry": False,
        "output_catalog_sha256": "",
    }
    if blockers:
        report["status"] = "blocked_path_or_configuration"
        return report

    initial, initial_errors = _capture_immutable_evidence(
        repo_root=Path(repo_root),
        paths=paths,
        report_tree_paths=report_tree_paths,
    )
    report["initial_evidence"] = initial
    report["blockers"].extend(initial_errors)
    report["blockers"].extend(_evidence_mismatch_blockers(initial, expected, phase="initial"))
    if report["blockers"]:
        report["status"] = "blocked_initial_evidence_mismatch"
        return report
    if apply:
        initial_runtime, initial_runtime_errors = _capture_runtime_state(Path(repo_root))
        report["initial_runtime_state"] = initial_runtime
        report["blockers"].extend(initial_runtime_errors)
        report["blockers"].extend(_runtime_state_blockers(initial_runtime, phase="initial"))
        if report["blockers"]:
            report["status"] = "blocked_initial_runtime_not_quiescent"
            return report

    try:
        original_catalog_payload, catalog_digest = _read_with_sha256(paths["catalog"])
        if catalog_digest != expected_catalog_sha256:
            raise ValueError("catalog changed between initial evidence and contract read")
        catalog = json.loads(original_catalog_payload.decode("utf-8"))
        if not isinstance(catalog, dict):
            raise ValueError("catalog must be a JSON object")
        retired, removed_required, removed_table = _retired_catalog_payload(catalog)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        report["blockers"].append(f"catalog_contract_mismatch:{exc}")
        report["status"] = "blocked_catalog_contract_mismatch"
        return report

    output = (json.dumps(retired, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    report["removed_required_table"] = removed_required
    report["removed_table_entry"] = removed_table
    report["output_catalog_sha256"] = _sha256_bytes(output)

    if not apply:
        report["status"] = "would_retire"
        return report
    if confirm_token != CONFIRM_TOKEN:
        report["blockers"].append("static_confirmation_token_required")
        report["status"] = "blocked_confirmation_required"
        return report
    if _COMMIT_RE.fullmatch(str(candidate_commit)) is None:
        report["blockers"].append("candidate_commit_invalid")
    if not candidate_worktree_clean:
        report["blockers"].append("candidate_worktree_not_clean")
    if report["blockers"]:
        report["status"] = "blocked_candidate_state"
        return report

    activation_digests, activation_blockers = _load_and_validate_activation_evidence(
        activation_receipt_path=Path(activation_receipt_path),
        approval_attestation_path=Path(approval_attestation_path),
        repo_root=Path(repo_root),
        paths=paths,
        candidate_commit=str(candidate_commit),
        expected=expected,
    )
    report["activation_evidence_sha256"] = activation_digests
    report["blockers"].extend(activation_blockers)
    if report["blockers"]:
        report["status"] = "blocked_activation_evidence"
        return report

    # Re-read every immutable input immediately before replacement. The
    # receipt and external approval are also rebound to their initial hashes.
    pre_replace, pre_replace_errors = _capture_immutable_evidence(
        repo_root=Path(repo_root),
        paths=paths,
        report_tree_paths=report_tree_paths,
    )
    report["pre_replace_evidence"] = pre_replace
    report["blockers"].extend(pre_replace_errors)
    report["blockers"].extend(
        _evidence_mismatch_blockers(pre_replace, expected, phase="pre_replace")
    )
    rebound_digests, rebound_blockers = _load_and_validate_activation_evidence(
        activation_receipt_path=Path(activation_receipt_path),
        approval_attestation_path=Path(approval_attestation_path),
        repo_root=Path(repo_root),
        paths=paths,
        candidate_commit=str(candidate_commit),
        expected=expected,
    )
    report["blockers"].extend(rebound_blockers)
    if rebound_digests != activation_digests:
        report["blockers"].append("activation_evidence_changed_before_replace")
    pre_replace_runtime, pre_replace_runtime_errors = _capture_runtime_state(Path(repo_root))
    report["pre_replace_runtime_state"] = pre_replace_runtime
    report["blockers"].extend(pre_replace_runtime_errors)
    report["blockers"].extend(_runtime_state_blockers(pre_replace_runtime, phase="pre_replace"))
    if report["blockers"]:
        report["status"] = "blocked_pre_replace_evidence_changed"
        return report

    # This script is the repository's only production _catalog.json writer.
    # Future writers must acquire the same lock. Recheck the CAS hash inside
    # that lock immediately adjacent to os.replace so a competing activation
    # can never overwrite a newer catalog.
    try:
        _immediate_payload, immediate_catalog_sha = _read_with_sha256(paths["catalog"])
    except OSError as exc:
        report["blockers"].append(f"catalog_immediate_cas_read_failed:{exc}")
        report["status"] = "blocked_catalog_cas_failed"
        return report
    report["immediate_pre_replace_catalog_sha256"] = immediate_catalog_sha
    if immediate_catalog_sha != expected_catalog_sha256:
        report["blockers"].append("catalog_changed_immediately_before_replace")
        report["status"] = "blocked_catalog_cas_failed"
        return report

    try:
        _atomic_write(paths["catalog"], output)
    except OSError as exc:
        report["blockers"].append(f"catalog_atomic_write_failed:{exc}")
        rollback_ok, rollback_detail = _restore_catalog(
            path=paths["catalog"],
            original_payload=original_catalog_payload,
            expected_sha256=expected_catalog_sha256,
        )
        report["rollback_verified"] = rollback_ok
        report["rollback_detail"] = rollback_detail
        report["status"] = (
            "rolled_back_catalog_write_failed"
            if rollback_ok
            else "critical_catalog_rollback_failed"
        )
        return report

    post_expected = {
        "files": {
            **expected["files"],
            "catalog": {"sha256": report["output_catalog_sha256"]},
        },
        "report_trees": expected["report_trees"],
    }
    post_runtime, post_runtime_errors = _capture_runtime_state(Path(repo_root))
    report["post_apply_runtime_state"] = post_runtime
    fresh_reader_ok, fresh_reader_detail = _fresh_reader_retirement_probe(Path(repo_root))
    report["fresh_reader_retirement_probe"] = {
        "passed": fresh_reader_ok,
        "detail": fresh_reader_detail,
    }
    post_apply, post_errors = _capture_immutable_evidence(
        repo_root=Path(repo_root),
        paths=paths,
        report_tree_paths=report_tree_paths,
    )
    report["post_apply_evidence"] = post_apply
    post_blockers = list(post_errors)
    post_blockers.extend(post_runtime_errors)
    post_blockers.extend(_runtime_state_blockers(post_runtime, phase="post_apply"))
    if not fresh_reader_ok:
        post_blockers.append("fresh_reader_retirement_probe_failed")
    post_blockers.extend(_evidence_mismatch_blockers(post_apply, post_expected, phase="post_apply"))
    if post_blockers:
        report["blockers"].extend(post_blockers)
        rollback_ok, rollback_detail = _restore_catalog(
            path=paths["catalog"],
            original_payload=original_catalog_payload,
            expected_sha256=expected_catalog_sha256,
        )
        report["rollback_verified"] = rollback_ok
        report["rollback_detail"] = rollback_detail
        report["status"] = (
            "rolled_back_post_apply_readback_failed"
            if rollback_ok
            else "critical_catalog_rollback_failed"
        )
        return report

    report["rollback_verified"] = False
    report["status"] = "retired"
    return report


def retire_intelligence_catalog(
    *,
    catalog_path: Path,
    latest_path: Path,
    mart_path: Path,
    report_tree_paths: Mapping[str, Path],
    activation_receipt_path: Path,
    approval_attestation_path: Path,
    repo_root: Path,
    candidate_commit: str,
    candidate_worktree_clean: bool,
    expected_catalog_sha256: str = EXPECTED_PRE_CUTOVER_CATALOG_SHA256,
    expected_latest_sha256: str = EXPECTED_LATEST_SHA256,
    expected_mart_sha256: str = EXPECTED_MART_SHA256,
    expected_report_trees: Mapping[str, Mapping[str, Any]] = EXPECTED_HISTORICAL_REPORT_TREES,
    apply: bool = False,
    confirm_token: str | None = None,
) -> dict[str, Any]:
    """Run the full catalog transaction under a non-blocking exclusive lock."""

    kwargs = {
        "catalog_path": Path(catalog_path),
        "latest_path": Path(latest_path),
        "mart_path": Path(mart_path),
        "report_tree_paths": report_tree_paths,
        "activation_receipt_path": Path(activation_receipt_path),
        "approval_attestation_path": Path(approval_attestation_path),
        "repo_root": Path(repo_root),
        "candidate_commit": candidate_commit,
        "candidate_worktree_clean": candidate_worktree_clean,
        "expected_catalog_sha256": expected_catalog_sha256,
        "expected_latest_sha256": expected_latest_sha256,
        "expected_mart_sha256": expected_mart_sha256,
        "expected_report_trees": expected_report_trees,
        "apply": apply,
        "confirm_token": confirm_token,
    }
    lock_path = Path(catalog_path).parent / CANONICAL_CATALOG_LOCK_PATH.name
    if not apply:
        result = _retire_intelligence_catalog_transaction(**kwargs)
        result["catalog_lock"] = {
            "path": str(lock_path),
            "acquired": False,
            "reason": "dry_run",
        }
        return result

    try:
        lock_fd = _acquire_catalog_lock(lock_path)
    except CatalogLockBusy as exc:
        return {
            "schema_version": "myquant.intelligence-catalog-retirement.v14",
            "apply_requested": True,
            "catalog_path": str(catalog_path),
            "catalog_lock": {
                "path": str(lock_path),
                "acquired": False,
                "non_blocking": True,
            },
            "blockers": [f"catalog_activation_lock_busy:{exc}"],
            "status": "blocked_catalog_lock_busy",
        }
    try:
        result = _retire_intelligence_catalog_transaction(**kwargs)
        result["catalog_lock"] = {
            "path": str(lock_path),
            "acquired": True,
            "non_blocking": True,
            "held_through_transaction": True,
        }
        return result
    finally:
        _release_catalog_lock(lock_fd)


def _git_candidate_state(repo_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status_output = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "", False
    return commit, not bool(status_output.strip())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed removal of intelligence_daily from the canonical CN catalog. "
            "Paths and evidence digests are compiled constants."
        )
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-token", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    candidate_commit, candidate_worktree_clean = _git_candidate_state(PRODUCTION_REPO_ROOT)
    report = retire_intelligence_catalog(
        repo_root=PRODUCTION_REPO_ROOT,
        catalog_path=PRODUCTION_REPO_ROOT / CANONICAL_CATALOG_PATH,
        latest_path=PRODUCTION_REPO_ROOT / CANONICAL_LATEST_PATH,
        mart_path=PRODUCTION_REPO_ROOT / CANONICAL_MART_PATH,
        report_tree_paths={
            label: PRODUCTION_REPO_ROOT / label for label in EXPECTED_HISTORICAL_REPORT_TREES
        },
        activation_receipt_path=(PRODUCTION_REPO_ROOT / CANONICAL_ACTIVATION_RECEIPT_PATH),
        approval_attestation_path=CANONICAL_APPROVAL_ATTESTATION_PATH,
        candidate_commit=candidate_commit,
        candidate_worktree_clean=candidate_worktree_clean,
        apply=args.apply,
        confirm_token=args.confirm_token,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"would_retire", "retired"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
