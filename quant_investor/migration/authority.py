"""Detached exact-byte authority evidence for the unified cutover.

These artifacts live outside the release tree.  They record observations and
authorization, but never write the System active pointer and never grant any
trading-side authority.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
import base64
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import stat
import subprocess
from typing import Any, Final, Literal

from quant_investor.contracts import (
    ContractError,
    canonical_json_bytes,
    get_contract,
    parse_canonical_json_bytes,
    seal_artifact,
    validate_artifact,
)
from quant_investor.system.errors import (
    SystemContractError,
    SystemPreconditionError,
    SystemSecurityError,
    SystemStorageError,
)
from quant_investor.system.release_install import (
    RELEASE_INSTALL_EVIDENCE_KIND,
    validate_release_install_evidence,
)
from quant_investor.system.store import object_ref_for_artifact, validate_object_ref

CONCURRENT_HANDOFF_KIND: Final = "system.concurrent_task_handoff"
LEGACY_DISPOSITION_KIND: Final = "system.legacy_source_disposition"
FINAL_AUTHORIZATION_KIND: Final = "system.final_cutover_authorization"
GATE_EVIDENCE_KIND: Final = "system.cutover_gate_evidence"
AUTHORITY_KINDS: Final = frozenset(
    {
        CONCURRENT_HANDOFF_KIND,
        LEGACY_DISPOSITION_KIND,
        FINAL_AUTHORIZATION_KIND,
        GATE_EVIDENCE_KIND,
        RELEASE_INSTALL_EVIDENCE_KIND,
    }
)
_CONTRACT_SHA256S: Final = {kind: get_contract(kind).contract_sha256 for kind in AUTHORITY_KINDS}
_GIT_OID_RE: Final = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_THREAD_RE: Final = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_PATH_ROW_FIELDS: Final = frozenset(
    {"path", "status", "mode", "size", "git_blob_oid", "byte_sha256"}
)
_TEST_ROW_FIELDS: Final = frozenset({"command", "exit_code", "stdout_sha256", "status"})
_READBACK_ROW_FIELDS: Final = frozenset(
    {
        "commit",
        "tree",
        "status_porcelain_sha256",
        "path_inventory_sha256",
        "observed_at",
    }
)
_DISPOSITION_ROW_FIELDS: Final = frozenset(
    {
        "source_path",
        "source_blob_oid",
        "classification",
        "stable_target_path",
        "stable_target_blob_oid",
        "behavior_test_selector",
        "reason",
    }
)
_ANCESTRY_ROW_FIELDS: Final = frozenset({"ancestor", "descendant", "proved"})
_PREFLIGHT_ROW_FIELDS: Final = frozenset({"gate_id", "evidence_ref"})
_GATE_BATCH_FIELDS: Final = frozenset(
    {
        "argv",
        "exit_code",
        "stdout_base64",
        "stdout_sha256",
        "stderr_base64",
        "stderr_sha256",
        "executable_path",
        "executable_sha256",
        "stdin_sha256",
    }
)
_EXCLUDED_ROW_FIELDS: Final = frozenset({"commit", "descendant", "proved_not_ancestor"})
REQUIRED_FINAL_PREFLIGHT_GATES: Final = frozenset(
    {
        "clean_detached_clone",
        "contract_catalog",
        "flake8",
        "full_pytest",
        "legacy_zero_call",
        "mypy",
        "projection",
        "release_install_origin",
        "replacement_selectors",
    }
)
FinalAuthorizationValidationMode = Literal["PRE_CAS_CURRENT", "HISTORICAL"]
_GATE_RUNNER_ID: Final = "myquant.system.final-cutover-gate-runner"
_GATE_SPECS: Final[dict[str, tuple[tuple[str, ...], ...]]] = {
    "clean_detached_clone": (("git", "status", "--porcelain=v1", "--untracked-files=all"),),
    "contract_catalog": (("uv", "run", "pytest", "tests/unit/test_unified_contracts.py", "-q"),),
    "flake8": (
        (
            "uv",
            "run",
            "flake8",
            "quant_investor",
            "--count",
            "--select=E9,F63,F7,F82",
            "--show-source",
            "--statistics",
        ),
        (
            "uv",
            "run",
            "flake8",
            "quant_investor/contracts",
            "quant_investor/system",
            "quant_investor/factors/governance",
            "quant_investor/intelligence",
            "quant_investor/mainline",
            "quant_investor/cli",
            "--max-complexity=10",
            "--max-line-length=100",
        ),
    ),
    "full_pytest": (("uv", "run", "pytest", "tests/unit", "-q", "-ra"),),
    "legacy_zero_call": (
        (
            "uv",
            "run",
            "pytest",
            "tests/unit/test_unified_migration_resolver.py",
            "tests/unit/test_unified_cli_commands.py",
            "tests/unit/test_unified_cli_input.py",
            "tests/unit/test_unified_cli_output.py",
            "-q",
        ),
    ),
    "mypy": (
        (
            "uv",
            "run",
            "mypy",
            "quant_investor/contracts",
            "quant_investor/system",
            "quant_investor/factors/governance",
            "quant_investor/intelligence",
            "quant_investor/mainline",
            "quant_investor/cli",
            "--ignore-missing-imports",
        ),
    ),
    "projection": (("uv", "run", "python", "operations/codex/verify_projection.py"),),
    "release_install_origin": (
        (
            "uv",
            "run",
            "python",
            "-m",
            "quant_investor.system.release_install",
        ),
    ),
    "replacement_selectors": (
        (
            "uv",
            "run",
            "python",
            "scripts/run_unified_replacement_selectors.py",
        ),
    ),
}
_GATE_OUTPUT_MAX_BYTES: Final = 64 * 1024 * 1024


def _gate_runner_spec_sha256(gate_id: str, batches: Sequence[Sequence[str]]) -> str:
    return _sha256(
        canonical_json_bytes(
            {
                "runner_id": _GATE_RUNNER_ID,
                "gate_id": gate_id,
                "batches": [list(argv) for argv in batches],
            }
        )
    )


def _gate_specs_from_runner_source(  # noqa: C901
    raw: bytes,
) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Recover the fixed spec literal from the independently authorized blob."""

    try:
        module = ast.parse(raw.decode("utf-8"))
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise SystemPreconditionError("historical gate runner source is invalid") from exc
    literal: Any = None
    for node in module.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "_GATE_SPECS" and node.value is not None:
                try:
                    literal = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError) as exc:
                    raise SystemPreconditionError(
                        "historical gate runner spec is not a literal"
                    ) from exc
                break
    if type(literal) is not dict or set(literal) != REQUIRED_FINAL_PREFLIGHT_GATES:
        raise SystemPreconditionError("historical gate runner set differs")
    result: dict[str, tuple[tuple[str, ...], ...]] = {}
    for gate, batches in literal.items():
        if (
            type(gate) is not str
            or type(batches) is not tuple
            or not batches
            or any(
                type(argv) is not tuple
                or not argv
                or any(type(argument) is not str or not argument for argument in argv)
                for argv in batches
            )
        ):
            raise SystemPreconditionError("historical gate runner spec is malformed")
        result[gate] = batches
    return result


_ALLOWED_DISPOSITIONS: Final = frozenset(
    {
        "PORTED_TO_STABLE",
        "PACKAGING_ONLY_NOT_REQUIRED",
        "LEGACY_CUSTODY_ONLY",
        "BLOCKED_UNRESOLVED",
    }
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SystemContractError(f"{label} is not lowercase SHA-256")
    return value


def _git_oid(value: Any, *, label: str) -> str:
    if type(value) is not str or _GIT_OID_RE.fullmatch(value) is None:
        raise SystemContractError(f"{label} is not a canonical git object id")
    return value


def _text(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise SystemContractError(f"{label} is not canonical text")
    return value


def _path(value: Any, *, label: str, allow_empty: bool = False) -> str:
    if allow_empty and value == "":
        return ""
    text = _text(value, label=label)
    parsed = PurePosixPath(text)
    if (
        parsed.is_absolute()
        or str(parsed) != text
        or "\\" in text
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise SystemContractError(f"{label} is not a canonical relative path")
    return text


def _artifact(document: Mapping[str, Any] | bytes, kind: str) -> dict[str, Any]:
    try:
        return validate_artifact(
            document,
            expected_kind=kind,
            expected_contract_sha256=_CONTRACT_SHA256S[kind],
        )
    except ContractError as exc:
        raise SystemContractError(f"{kind} contract failed") from exc


def _validate_path_rows(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list or not value:
        raise SystemContractError("handoff path rows are absent")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if type(row) is not dict or set(row) != _PATH_ROW_FIELDS:
            raise SystemContractError("handoff path row fields are not exact")
        status = row["status"]
        if status != "PRESENT":
            raise SystemContractError("handoff only accepts final PRESENT paths")
        mode = row["mode"]
        size = row["size"]
        if mode not in {"100600", "100644", "100755"} or type(size) is not int or size < 0:
            raise SystemContractError("handoff path mode/size is invalid")
        result.append(
            {
                "path": _path(row["path"], label=f"path_rows[{index}].path"),
                "status": status,
                "mode": mode,
                "size": size,
                "git_blob_oid": _git_oid(
                    row["git_blob_oid"], label=f"path_rows[{index}].git_blob_oid"
                ),
                "byte_sha256": _sha(row["byte_sha256"], label=f"path_rows[{index}].byte_sha256"),
            }
        )
    paths = [row["path"] for row in result]
    if paths != sorted(set(paths)):
        raise SystemContractError("handoff paths are not sorted unique")
    return result


def _validate_test_rows(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list or not value:
        raise SystemContractError("focused test evidence is absent")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if type(row) is not dict or set(row) != _TEST_ROW_FIELDS:
            raise SystemContractError("focused test row fields are not exact")
        if row["exit_code"] != 0 or row["status"] != "PASS":
            raise SystemPreconditionError("focused task test did not pass")
        result.append(
            {
                "command": _text(row["command"], label=f"test_rows[{index}].command"),
                "exit_code": 0,
                "stdout_sha256": _sha(
                    row["stdout_sha256"], label=f"test_rows[{index}].stdout_sha256"
                ),
                "status": "PASS",
            }
        )
    commands = [row["command"] for row in result]
    if commands != sorted(set(commands)):
        raise SystemContractError("focused test commands are not sorted unique")
    return result


def _validate_readback_rows(value: Any, *, label: str) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) != 2:
        raise SystemContractError(f"{label} requires exactly two readbacks")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if type(row) is not dict or set(row) != _READBACK_ROW_FIELDS:
            raise SystemContractError(f"{label} fields are not exact")
        normalized.append(
            {
                "commit": _git_oid(row["commit"], label=f"{label}[{index}].commit"),
                "tree": _git_oid(row["tree"], label=f"{label}[{index}].tree"),
                "status_porcelain_sha256": _sha(
                    row["status_porcelain_sha256"],
                    label=f"{label}[{index}].status_porcelain_sha256",
                ),
                "path_inventory_sha256": _sha(
                    row["path_inventory_sha256"],
                    label=f"{label}[{index}].path_inventory_sha256",
                ),
                "observed_at": _text(row["observed_at"], label=f"{label}[{index}].observed_at"),
            }
        )
    first = {key: value for key, value in normalized[0].items() if key != "observed_at"}
    second = {key: value for key, value in normalized[1].items() if key != "observed_at"}
    if first != second or normalized[0]["observed_at"] == normalized[1]["observed_at"]:
        raise SystemPreconditionError(f"{label} is not a stable double-read")
    return normalized


def build_concurrent_task_handoff(
    *,
    handoff_id: str,
    task_name: str,
    thread_id: str,
    accepted_baseline_commit: str,
    task_commit: str,
    task_tree: str,
    path_rows: Sequence[Mapping[str, Any]],
    focused_test_rows: Sequence[Mapping[str, Any]],
    readback_rows: Sequence[Mapping[str, Any]],
    writer_ended: bool,
    main_clean: bool,
    created_at: str,
) -> dict[str, Any]:
    if writer_ended is not True or main_clean is not True:
        raise SystemPreconditionError("concurrent task writer/main cleanliness is unresolved")
    if type(thread_id) is not str or _THREAD_RE.fullmatch(thread_id) is None:
        raise SystemContractError("concurrent task thread id is invalid")
    rows = _validate_readback_rows(list(readback_rows), label="handoff readback_rows")
    if rows[0]["commit"] != task_commit or rows[0]["tree"] != task_tree:
        raise SystemPreconditionError("handoff readback commit/tree differs")
    return seal_artifact(
        CONCURRENT_HANDOFF_KIND,
        {
            "handoff_id": _text(handoff_id, label="handoff_id"),
            "state": "IMMUTABLE",
            "task_name": _text(task_name, label="task_name"),
            "thread_id": thread_id,
            "accepted_baseline_commit": _git_oid(
                accepted_baseline_commit, label="accepted_baseline_commit"
            ),
            "handoff_type": "COMMIT",
            "task_commit": _git_oid(task_commit, label="task_commit"),
            "task_tree": _git_oid(task_tree, label="task_tree"),
            "path_rows": _validate_path_rows(list(path_rows)),
            "focused_test_rows": _validate_test_rows(list(focused_test_rows)),
            "writer_ended": True,
            "main_clean": True,
            "readback_rows": rows,
        },
        created_at=created_at,
    )


def validate_concurrent_task_handoff(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    artifact = _artifact(document, CONCURRENT_HANDOFF_KIND)
    payload = artifact["payload"]
    rebuilt = build_concurrent_task_handoff(
        handoff_id=payload["handoff_id"],
        task_name=payload["task_name"],
        thread_id=payload["thread_id"],
        accepted_baseline_commit=payload["accepted_baseline_commit"],
        task_commit=payload["task_commit"],
        task_tree=payload["task_tree"],
        path_rows=payload["path_rows"],
        focused_test_rows=payload["focused_test_rows"],
        readback_rows=payload["readback_rows"],
        writer_ended=payload["writer_ended"],
        main_clean=payload["main_clean"],
        created_at=artifact["created_at"],
    )
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(artifact):
        raise SystemContractError("concurrent task handoff semantic replay differs")
    return artifact


def build_legacy_source_disposition(
    *,
    disposition_id: str,
    source_commit: str,
    rows: Sequence[Mapping[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    if not rows:
        raise SystemContractError("legacy disposition rows are absent")
    normalized: list[dict[str, Any]] = []
    for index, value in enumerate(rows):
        row = dict(value)
        if set(row) != _DISPOSITION_ROW_FIELDS:
            raise SystemContractError("legacy disposition row fields are not exact")
        classification = row["classification"]
        if classification not in _ALLOWED_DISPOSITIONS:
            raise SystemContractError("legacy disposition classification is invalid")
        stable_path = _path(
            row["stable_target_path"],
            label=f"disposition[{index}].stable_target_path",
            allow_empty=True,
        )
        stable_oid = row["stable_target_blob_oid"]
        if stable_path:
            _git_oid(stable_oid, label=f"disposition[{index}].stable_target_blob_oid")
        elif stable_oid != "":
            raise SystemContractError("legacy disposition empty target has a blob")
        normalized.append(
            {
                "source_path": _path(row["source_path"], label=f"disposition[{index}].source_path"),
                "source_blob_oid": _git_oid(
                    row["source_blob_oid"],
                    label=f"disposition[{index}].source_blob_oid",
                ),
                "classification": classification,
                "stable_target_path": stable_path,
                "stable_target_blob_oid": stable_oid,
                "behavior_test_selector": _text(
                    row["behavior_test_selector"],
                    label=f"disposition[{index}].behavior_test_selector",
                ),
                "reason": _text(row["reason"], label=f"disposition[{index}].reason"),
            }
        )
    paths = [row["source_path"] for row in normalized]
    if paths != sorted(set(paths)):
        raise SystemContractError("legacy disposition paths are not sorted unique")
    blocked = sum(row["classification"] == "BLOCKED_UNRESOLVED" for row in normalized)
    if blocked:
        raise SystemPreconditionError("legacy disposition contains BLOCKED_UNRESOLVED")
    return seal_artifact(
        LEGACY_DISPOSITION_KIND,
        {
            "disposition_id": _text(disposition_id, label="disposition_id"),
            "state": "IMMUTABLE",
            "source_commit": _git_oid(source_commit, label="source_commit"),
            "rows": normalized,
            "blocked_unresolved_count": 0,
        },
        created_at=created_at,
    )


def validate_legacy_source_disposition(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    artifact = _artifact(document, LEGACY_DISPOSITION_KIND)
    payload = artifact["payload"]
    rebuilt = build_legacy_source_disposition(
        disposition_id=payload["disposition_id"],
        source_commit=payload["source_commit"],
        rows=payload["rows"],
        created_at=artifact["created_at"],
    )
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(artifact):
        raise SystemContractError("legacy disposition semantic replay differs")
    return artifact


def _canonical_timestamp(value: Any, *, label: str) -> str:
    text = _text(value, label=label)
    try:
        parsed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise SystemContractError(f"{label} is not canonical UTC") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != text:
        raise SystemContractError(f"{label} is not canonical UTC")
    return text


def _raw_result(value: Any, *, label: str) -> tuple[str, str]:
    if type(value) is not str:
        raise SystemContractError(f"{label} is not base64 text")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise SystemContractError(f"{label} is not canonical base64") from exc
    if len(raw) > _GATE_OUTPUT_MAX_BYTES or base64.b64encode(raw).decode("ascii") != value:
        raise SystemContractError(f"{label} is not bounded canonical base64")
    return value, _sha256(raw)


def _seal_cutover_gate_evidence(  # noqa: C901
    *,
    gate_id: str,
    final_commit: str,
    final_tree: str,
    runner_code_sha256: str,
    environment_sha256: str,
    batch_results: Sequence[Mapping[str, Any]],
    subject_ref: Mapping[str, Any],
    started_at: str,
    finished_at: str,
    _expected_batches: Sequence[Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Private sealing primitive used by the fixed runner and unit fixtures."""

    gate = _text(gate_id, label="gate_id")
    if gate not in REQUIRED_FINAL_PREFLIGHT_GATES:
        raise SystemContractError("cutover gate id is not in the fixed allowlist")
    expected_batches = (
        tuple(tuple(argument for argument in argv) for argv in _expected_batches)
        if _expected_batches is not None
        else _GATE_SPECS[gate]
    )
    if len(batch_results) != len(expected_batches):
        raise SystemContractError("cutover gate batch count differs from fixed runner")
    normalized_results: list[dict[str, Any]] = []
    all_passed = True
    for index, (raw_row, expected_argv) in enumerate(zip(batch_results, expected_batches)):
        row = dict(raw_row)
        if set(row) != _GATE_BATCH_FIELDS:
            raise SystemContractError("cutover gate batch fields are not exact")
        argv = row["argv"]
        if argv != list(expected_argv):
            raise SystemContractError("cutover gate argv differs from fixed runner")
        exit_code = row["exit_code"]
        if type(exit_code) is not int or exit_code < 0 or exit_code > 255:
            raise SystemContractError("cutover gate exit code is invalid")
        stdout_base64, stdout_sha = _raw_result(
            row["stdout_base64"], label=f"batch_results[{index}].stdout_base64"
        )
        stderr_base64, stderr_sha = _raw_result(
            row["stderr_base64"], label=f"batch_results[{index}].stderr_base64"
        )
        executable_path = row["executable_path"]
        if type(executable_path) is not str or not Path(executable_path).is_absolute():
            raise SystemContractError("cutover gate executable path is not absolute")
        if row["stdout_sha256"] != stdout_sha or row["stderr_sha256"] != stderr_sha:
            raise SystemContractError("cutover gate exact output hash differs")
        normalized_results.append(
            {
                "argv": list(expected_argv),
                "exit_code": exit_code,
                "stdout_base64": stdout_base64,
                "stdout_sha256": stdout_sha,
                "stderr_base64": stderr_base64,
                "stderr_sha256": stderr_sha,
                "executable_path": executable_path,
                "executable_sha256": _sha(
                    row["executable_sha256"],
                    label=f"batch_results[{index}].executable_sha256",
                ),
                "stdin_sha256": _sha(
                    row["stdin_sha256"], label=f"batch_results[{index}].stdin_sha256"
                ),
            }
        )
        if gate != "release_install_origin" and row["stdin_sha256"] != _sha256(b""):
            raise SystemContractError("non-release cutover gate stdin is not empty")
        all_passed = all_passed and exit_code == 0
    started = _canonical_timestamp(started_at, label="started_at")
    finished = _canonical_timestamp(finished_at, label="finished_at")
    if finished < started:
        raise SystemContractError("cutover gate finish precedes start")
    body = {
        "state": "PASS" if all_passed else "FAIL",
        "gate_id": gate,
        "runner_id": _GATE_RUNNER_ID,
        "runner_spec_sha256": _gate_runner_spec_sha256(gate, expected_batches),
        "runner_code_sha256": _sha(runner_code_sha256, label="runner_code_sha256"),
        "final_commit": _git_oid(final_commit, label="final_commit"),
        "final_tree": _git_oid(final_tree, label="final_tree"),
        "environment_sha256": _sha(environment_sha256, label="environment_sha256"),
        "batch_results": normalized_results,
        "subject_ref": validate_object_ref(subject_ref, label="subject_ref"),
        "started_at": started,
        "finished_at": finished,
    }
    evidence_id = "cutover-gate-" + _sha256(canonical_json_bytes(body))
    return seal_artifact(
        GATE_EVIDENCE_KIND,
        {**body, "evidence_id": evidence_id},
        created_at=finished,
    )


def run_cutover_gate(  # noqa: C901
    *,
    repository_root: str | os.PathLike[str],
    gate_id: str,
    final_commit: str,
    final_tree: str,
    subject_ref: Mapping[str, Any],
    release_install_evidence: Mapping[str, Any] | bytes | None = None,
    deployed_release: Mapping[str, Any] | bytes | None = None,
    timeout_seconds: int = 7200,
) -> dict[str, Any]:
    """Execute one repository-owned final gate without caller-supplied argv."""

    gate = _text(gate_id, label="gate_id")
    if gate not in REQUIRED_FINAL_PREFLIGHT_GATES:
        raise SystemContractError("cutover gate id is not in the fixed allowlist")
    if type(timeout_seconds) is not int or timeout_seconds <= 0 or timeout_seconds > 14400:
        raise SystemContractError("cutover gate timeout is invalid")
    root = Path(repository_root).resolve(strict=True)
    commit = _git_oid(final_commit, label="final_commit")
    tree = _git_oid(final_tree, label="final_tree")
    if _git_scalar(root, "rev-parse", "HEAD^{commit}") != commit:
        raise SystemPreconditionError("cutover gate checkout is not frozen HEAD")
    if _git_scalar(root, "rev-parse", "HEAD^{tree}") != tree:
        raise SystemPreconditionError("cutover gate checkout tree differs")
    if _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all"):
        raise SystemPreconditionError("cutover gate checkout is not clean")
    _mode, _oid, runner_raw = _git_blob(root, commit, "quant_investor/migration/authority.py")
    if Path(__file__).resolve(strict=True).read_bytes() != runner_raw:
        raise SystemPreconditionError("executing gate runner differs from frozen source")
    environment = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ.get("PATH", ""),
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": "",
        "UV_CACHE_DIR": "/tmp/myquant-cutover-uv-cache",
    }
    environment_sha = _sha256(canonical_json_bytes(environment))
    normalized_subject_ref = validate_object_ref(subject_ref, label="subject_ref")
    if gate == "release_install_origin":
        if release_install_evidence is None or deployed_release is None:
            raise SystemPreconditionError("release install gate inputs are absent")
        evidence = validate_release_install_evidence(release_install_evidence)
        try:
            release = validate_artifact(deployed_release, expected_kind="system.release")
        except ContractError as exc:
            raise SystemContractError("release install gate release contract failed") from exc
        if object_ref_for_artifact(evidence) != normalized_subject_ref:
            raise SystemPreconditionError("release install gate subject differs")
        if object_ref_for_artifact(release) != evidence["payload"]["release_ref"]:
            raise SystemPreconditionError("release install gate release differs")
        gate_stdin = canonical_json_bytes(
            {
                "release_install_evidence": evidence,
                "deployed_release": release,
            }
        )
    else:
        if release_install_evidence is not None or deployed_release is not None:
            raise SystemContractError("non-release cutover gate rejects release inputs")
        gate_stdin = b""
    results: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for argv in _GATE_SPECS[gate]:
        executable = shutil.which(argv[0], path=environment["PATH"])
        if executable is None:
            raise SystemPreconditionError("cutover gate executable is unavailable")
        executable_path = Path(executable).resolve(strict=True)
        executable_raw = executable_path.read_bytes()
        try:
            completed = subprocess.run(
                [str(executable_path), *argv[1:]],
                cwd=root,
                env=environment,
                input=gate_stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise SystemPreconditionError("fixed cutover gate execution failed") from exc
        results.append(
            {
                "argv": list(argv),
                "exit_code": completed.returncode,
                "stdout_base64": base64.b64encode(completed.stdout).decode("ascii"),
                "stdout_sha256": _sha256(completed.stdout),
                "stderr_base64": base64.b64encode(completed.stderr).decode("ascii"),
                "stderr_sha256": _sha256(completed.stderr),
                "executable_path": str(executable_path),
                "executable_sha256": _sha256(executable_raw),
                "stdin_sha256": _sha256(gate_stdin),
            }
        )
    if (
        _git_scalar(root, "rev-parse", "HEAD^{commit}") != commit
        or _git_scalar(root, "rev-parse", "HEAD^{tree}") != tree
    ):
        raise SystemPreconditionError("cutover gate changed frozen Git identity")
    if _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all"):
        raise SystemPreconditionError("cutover gate changed frozen checkout")
    finished = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return _seal_cutover_gate_evidence(
        gate_id=gate,
        final_commit=commit,
        final_tree=tree,
        runner_code_sha256=_sha256(runner_raw),
        environment_sha256=environment_sha,
        batch_results=results,
        subject_ref=normalized_subject_ref,
        started_at=started,
        finished_at=finished,
    )


def validate_cutover_gate_evidence(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    artifact = _artifact(document, GATE_EVIDENCE_KIND)
    payload = artifact["payload"]
    rebuilt = _seal_cutover_gate_evidence(
        gate_id=payload["gate_id"],
        final_commit=payload["final_commit"],
        final_tree=payload["final_tree"],
        runner_code_sha256=payload["runner_code_sha256"],
        environment_sha256=payload["environment_sha256"],
        batch_results=payload["batch_results"],
        subject_ref=payload["subject_ref"],
        started_at=payload["started_at"],
        finished_at=payload["finished_at"],
    )
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(artifact):
        raise SystemContractError("cutover gate evidence semantic replay differs")
    return artifact


def _validate_historical_cutover_gate_evidence(
    document: Mapping[str, Any] | bytes,
    *,
    authorized_runner_source: bytes,
) -> dict[str, Any]:
    """Replay a receipt from the exact initial runner blob, never current HEAD."""

    artifact = _artifact(document, GATE_EVIDENCE_KIND)
    payload = artifact["payload"]
    gate = payload.get("gate_id")
    specs = _gate_specs_from_runner_source(authorized_runner_source)
    if gate not in specs:
        raise SystemPreconditionError("historical gate is absent from authorized runner")
    rebuilt = _seal_cutover_gate_evidence(
        gate_id=gate,
        final_commit=payload["final_commit"],
        final_tree=payload["final_tree"],
        runner_code_sha256=payload["runner_code_sha256"],
        environment_sha256=payload["environment_sha256"],
        batch_results=payload["batch_results"],
        subject_ref=payload["subject_ref"],
        started_at=payload["started_at"],
        finished_at=payload["finished_at"],
        _expected_batches=specs[gate],
    )
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(artifact):
        raise SystemContractError("historical cutover gate semantic replay differs")
    return artifact


def _validate_preflight_rows(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list or not value:
        raise SystemPreconditionError("final preflight rows are absent")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if type(raw) is not dict or set(raw) != _PREFLIGHT_ROW_FIELDS:
            raise SystemContractError("final preflight row fields are not exact")
        rows.append(
            {
                "gate_id": _text(raw["gate_id"], label=f"preflight[{index}].gate_id"),
                "evidence_ref": validate_object_ref(
                    raw["evidence_ref"],
                    label=f"preflight[{index}].evidence_ref",
                ),
            }
        )
    gate_ids = [row["gate_id"] for row in rows]
    if gate_ids != sorted(REQUIRED_FINAL_PREFLIGHT_GATES):
        raise SystemPreconditionError("final preflight gate set is not exact")
    return rows


def _git(repository_root: Path, *arguments: str, allow_not_ancestor: bool = False) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemPreconditionError("git authority verification could not run") from exc
    if completed.returncode == 0:
        return completed.stdout
    if allow_not_ancestor and completed.returncode == 1:
        return b""
    raise SystemPreconditionError("git authority verification failed")


def _git_scalar(repository_root: Path, *arguments: str) -> str:
    raw = _git(repository_root, *arguments)
    try:
        value = raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise SystemPreconditionError("git returned non-ASCII identity") from exc
    return _git_oid(value, label="git identity")


def _git_inventory(repository_root: Path, commit: str) -> tuple[list[dict[str, Any]], str]:
    raw = _git(repository_root, "ls-tree", "-rz", "--full-tree", commit)
    rows: list[dict[str, Any]] = []
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        try:
            header, path_raw = entry.split(b"\t", 1)
            mode_raw, type_raw, oid_raw = header.split(b" ", 2)
            path = path_raw.decode("utf-8")
            mode = mode_raw.decode("ascii")
            object_type = type_raw.decode("ascii")
            oid = oid_raw.decode("ascii")
        except (UnicodeDecodeError, ValueError) as exc:
            raise SystemPreconditionError("git tree inventory is malformed") from exc
        if object_type != "blob":
            continue
        rows.append(
            {
                "path": _path(path, label="git inventory path"),
                "mode": mode,
                "git_blob_oid": _git_oid(oid, label="git inventory blob"),
            }
        )
    if rows != sorted(rows, key=lambda row: row["path"]):
        raise SystemPreconditionError("git tree inventory order is unstable")
    return rows, _sha256(canonical_json_bytes(rows))


def _git_blob(repository_root: Path, commit: str, path: str) -> tuple[str, str, bytes]:
    raw = _git(repository_root, "ls-tree", "-z", commit, "--", path)
    entries = [entry for entry in raw.split(b"\0") if entry]
    if len(entries) != 1:
        raise SystemPreconditionError("authority-bound path is absent or ambiguous")
    try:
        header, observed_path = entries[0].split(b"\t", 1)
        mode_raw, type_raw, oid_raw = header.split(b" ", 2)
    except ValueError as exc:
        raise SystemPreconditionError("authority-bound tree row is malformed") from exc
    if observed_path.decode("utf-8") != path or type_raw != b"blob":
        raise SystemPreconditionError("authority-bound path is not an exact blob")
    mode = mode_raw.decode("ascii")
    oid = _git_oid(oid_raw.decode("ascii"), label="authority-bound blob")
    blob = _git(repository_root, "cat-file", "blob", oid)
    return mode, oid, blob


def _is_ancestor(repository_root: Path, ancestor: str, descendant: str) -> bool:
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemPreconditionError("git ancestry verification could not run") from exc
    if completed.returncode not in {0, 1}:
        raise SystemPreconditionError("git ancestry verification failed")
    return completed.returncode == 0


def _seal_final_cutover_authorization(  # noqa: C901
    *,
    final_authorization_id: str,
    accepted_baseline_commit: str,
    historical_integration_commit: str,
    historical_dirty_evidence_ref: Mapping[str, Any],
    concurrent_task_handoff_ref: Mapping[str, Any],
    legacy_disposition_ref: Mapping[str, Any],
    deployed_release_ref: Mapping[str, Any],
    release_commit: str,
    release_tree: str,
    final_integration_commit: str,
    final_integration_tree: str,
    ancestry_rows: Sequence[Mapping[str, Any]],
    excluded_commit_rows: Sequence[Mapping[str, Any]],
    final_worktree_inventory_sha256: str,
    clean_checkout_readback_rows: Sequence[Mapping[str, Any]],
    user_authorization_basis: str,
    preflight_rows: Sequence[Mapping[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    ancestry: list[dict[str, Any]] = []
    for index, value in enumerate(ancestry_rows):
        row = dict(value)
        if set(row) != _ANCESTRY_ROW_FIELDS or row["proved"] is not True:
            raise SystemPreconditionError("final integration ancestry is not proved")
        ancestry.append(
            {
                "ancestor": _git_oid(row["ancestor"], label=f"ancestry[{index}].ancestor"),
                "descendant": _git_oid(row["descendant"], label=f"ancestry[{index}].descendant"),
                "proved": True,
            }
        )
    if not ancestry or ancestry != sorted(
        ancestry, key=lambda row: (row["ancestor"], row["descendant"])
    ):
        raise SystemContractError("final ancestry rows are not exact sorted")
    excluded: list[dict[str, Any]] = []
    for index, value in enumerate(excluded_commit_rows):
        row = dict(value)
        if set(row) != _EXCLUDED_ROW_FIELDS or row["proved_not_ancestor"] is not True:
            raise SystemPreconditionError("excluded commit non-ancestry is not proved")
        excluded.append(
            {
                "commit": _git_oid(row["commit"], label=f"excluded[{index}].commit"),
                "descendant": _git_oid(row["descendant"], label=f"excluded[{index}].descendant"),
                "proved_not_ancestor": True,
            }
        )
    if excluded != sorted(excluded, key=lambda row: (row["commit"], row["descendant"])):
        raise SystemContractError("excluded commit rows are not exact sorted")
    readbacks = _validate_readback_rows(
        list(clean_checkout_readback_rows), label="clean_checkout_readback_rows"
    )
    if (
        readbacks[0]["commit"] != final_integration_commit
        or readbacks[0]["tree"] != final_integration_tree
    ):
        raise SystemPreconditionError("final clean readback differs from frozen tree")
    preflights = _validate_preflight_rows(list(preflight_rows))
    return seal_artifact(
        FINAL_AUTHORIZATION_KIND,
        {
            "final_authorization_id": _text(final_authorization_id, label="final_authorization_id"),
            "state": "AUTHORIZED",
            "accepted_baseline_commit": _git_oid(
                accepted_baseline_commit, label="accepted_baseline_commit"
            ),
            "historical_integration_commit": _git_oid(
                historical_integration_commit, label="historical_integration_commit"
            ),
            "historical_dirty_evidence_ref": validate_object_ref(
                historical_dirty_evidence_ref, label="historical_dirty_evidence_ref"
            ),
            "concurrent_task_handoff_ref": validate_object_ref(
                concurrent_task_handoff_ref, label="concurrent_task_handoff_ref"
            ),
            "legacy_disposition_ref": validate_object_ref(
                legacy_disposition_ref, label="legacy_disposition_ref"
            ),
            "deployed_release_ref": validate_object_ref(
                deployed_release_ref, label="deployed_release_ref"
            ),
            "release_commit": _git_oid(release_commit, label="release_commit"),
            "release_tree": _git_oid(release_tree, label="release_tree"),
            "final_integration_commit": _git_oid(
                final_integration_commit, label="final_integration_commit"
            ),
            "final_integration_tree": _git_oid(
                final_integration_tree, label="final_integration_tree"
            ),
            "ancestry_rows": ancestry,
            "excluded_commit_rows": excluded,
            "final_worktree_inventory_sha256": _sha(
                final_worktree_inventory_sha256,
                label="final_worktree_inventory_sha256",
            ),
            "clean_checkout_readback_rows": readbacks,
            "user_authorization_basis": _text(
                user_authorization_basis, label="user_authorization_basis"
            ),
            "preflight_rows": preflights,
            "final_build_authorized": True,
            "cas_authorized": True,
        },
        created_at=created_at,
    )


def build_final_cutover_authorization(  # noqa: C901
    *,
    final_authorization_id: str,
    accepted_baseline_commit: str,
    historical_integration_commit: str,
    historical_dirty_evidence_ref: Mapping[str, Any],
    concurrent_task_handoff_ref: Mapping[str, Any],
    legacy_disposition_ref: Mapping[str, Any],
    deployed_release_ref: Mapping[str, Any],
    release_commit: str,
    release_tree: str,
    final_integration_commit: str,
    final_integration_tree: str,
    ancestry_rows: Sequence[Mapping[str, Any]],
    excluded_commit_rows: Sequence[Mapping[str, Any]],
    final_worktree_inventory_sha256: str,
    clean_checkout_readback_rows: Sequence[Mapping[str, Any]],
    user_authorization_basis: str,
    preflight_evidence: Sequence[Mapping[str, Any] | bytes],
    created_at: str,
) -> dict[str, Any]:
    """Derive final build/CAS authority from the exact fixed-runner receipts."""

    commit = _git_oid(final_integration_commit, label="final_integration_commit")
    tree = _git_oid(final_integration_tree, label="final_integration_tree")
    rows: list[dict[str, Any]] = []
    for raw in preflight_evidence:
        evidence = validate_cutover_gate_evidence(raw)
        payload = evidence["payload"]
        if (
            payload["state"] != "PASS"
            or payload["final_commit"] != commit
            or payload["final_tree"] != tree
            or any(batch["exit_code"] != 0 for batch in payload["batch_results"])
        ):
            raise SystemPreconditionError("final preflight fixed-runner receipt did not pass")
        rows.append(
            {
                "gate_id": payload["gate_id"],
                "evidence_ref": object_ref_for_artifact(evidence),
            }
        )
    rows.sort(key=lambda row: row["gate_id"])
    _validate_preflight_rows(rows)
    return _seal_final_cutover_authorization(
        final_authorization_id=final_authorization_id,
        accepted_baseline_commit=accepted_baseline_commit,
        historical_integration_commit=historical_integration_commit,
        historical_dirty_evidence_ref=historical_dirty_evidence_ref,
        concurrent_task_handoff_ref=concurrent_task_handoff_ref,
        legacy_disposition_ref=legacy_disposition_ref,
        deployed_release_ref=deployed_release_ref,
        release_commit=release_commit,
        release_tree=release_tree,
        final_integration_commit=commit,
        final_integration_tree=tree,
        ancestry_rows=ancestry_rows,
        excluded_commit_rows=excluded_commit_rows,
        final_worktree_inventory_sha256=final_worktree_inventory_sha256,
        clean_checkout_readback_rows=clean_checkout_readback_rows,
        user_authorization_basis=user_authorization_basis,
        preflight_rows=rows,
        created_at=created_at,
    )


def validate_final_cutover_authorization(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    artifact = _artifact(document, FINAL_AUTHORIZATION_KIND)
    payload = artifact["payload"]
    if (
        payload.get("final_build_authorized") is not True
        or payload.get("cas_authorized") is not True
    ):
        raise SystemPreconditionError("final cutover authorization is not machine-authorized")
    rebuilt = _seal_final_cutover_authorization(
        final_authorization_id=payload["final_authorization_id"],
        accepted_baseline_commit=payload["accepted_baseline_commit"],
        historical_integration_commit=payload["historical_integration_commit"],
        historical_dirty_evidence_ref=payload["historical_dirty_evidence_ref"],
        concurrent_task_handoff_ref=payload["concurrent_task_handoff_ref"],
        legacy_disposition_ref=payload["legacy_disposition_ref"],
        deployed_release_ref=payload["deployed_release_ref"],
        release_commit=payload["release_commit"],
        release_tree=payload["release_tree"],
        final_integration_commit=payload["final_integration_commit"],
        final_integration_tree=payload["final_integration_tree"],
        ancestry_rows=payload["ancestry_rows"],
        excluded_commit_rows=payload["excluded_commit_rows"],
        final_worktree_inventory_sha256=payload["final_worktree_inventory_sha256"],
        clean_checkout_readback_rows=payload["clean_checkout_readback_rows"],
        user_authorization_basis=payload["user_authorization_basis"],
        preflight_rows=payload["preflight_rows"],
        created_at=artifact["created_at"],
    )
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(artifact):
        raise SystemContractError("final cutover authorization semantic replay differs")
    return artifact


def validate_final_cutover_authorization_closure(  # noqa: C901
    document: Mapping[str, Any] | bytes,
    *,
    repository_root: str | os.PathLike[str],
    object_resolver: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    deployed_release_ref: Mapping[str, Any],
    validation_mode: FinalAuthorizationValidationMode,
) -> dict[str, Any]:
    """Replay current pre-CAS or immutable historical cutover authority.

    ``PRE_CAS_CURRENT`` proves that the checkout about to perform the first
    pointer CAS is still the exact clean frozen tree. ``HISTORICAL`` proves the
    anchored Git objects and evidence after activation without requiring HEAD
    to remain pinned forever to the initial cutover commit.
    """

    artifact = validate_final_cutover_authorization(document)
    payload = artifact["payload"]
    root = Path(repository_root).resolve(strict=True)
    top = Path(_git(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()).resolve()
    if top != root:
        raise SystemPreconditionError("final authorization repository root differs")
    final_commit = payload["final_integration_commit"]
    final_tree = payload["final_integration_tree"]
    if validation_mode not in {"PRE_CAS_CURRENT", "HISTORICAL"}:
        raise SystemContractError("final authorization validation mode is invalid")
    if _git_scalar(root, "rev-parse", f"{final_commit}^{{commit}}") != final_commit:
        raise SystemPreconditionError("authorized final commit object is absent")
    if _git_scalar(root, "rev-parse", f"{final_commit}^{{tree}}") != final_tree:
        raise SystemPreconditionError("authorized final tree object differs")
    if validation_mode == "PRE_CAS_CURRENT":
        if _git_scalar(root, "rev-parse", "HEAD^{commit}") != final_commit:
            raise SystemPreconditionError("authorized final commit is not current HEAD")
        if _git_scalar(root, "rev-parse", "HEAD^{tree}") != final_tree:
            raise SystemPreconditionError("authorized final tree is not current HEAD tree")
    if payload["release_commit"] != final_commit or payload["release_tree"] != final_tree:
        raise SystemPreconditionError("release identity is not the frozen final tree")
    normalized_release = validate_object_ref(deployed_release_ref, label="deployed_release_ref")
    if normalized_release != payload["deployed_release_ref"]:
        raise SystemPreconditionError("deployed release differs from final authorization")
    release = dict(object_resolver(normalized_release))
    if release.get("kind") != "system.release":
        raise SystemPreconditionError("authorized deployed release kind is invalid")
    if object_ref_for_artifact(release) != normalized_release:
        raise SystemPreconditionError("deployed release exact object differs")

    empty_status_sha = _sha256(b"")
    if validation_mode == "PRE_CAS_CURRENT":
        status_raw = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
        if status_raw:
            raise SystemPreconditionError("authorized final checkout is not clean")
        status_sha = _sha256(status_raw)
    else:
        status_sha = empty_status_sha
    inventory, inventory_sha = _git_inventory(root, final_commit)
    if inventory_sha != payload["final_worktree_inventory_sha256"]:
        raise SystemPreconditionError("authorized final tree inventory differs")
    for row in payload["clean_checkout_readback_rows"]:
        if (
            row["commit"] != final_commit
            or row["tree"] != final_tree
            or row["status_porcelain_sha256"] != status_sha
            or row["path_inventory_sha256"] != inventory_sha
        ):
            raise SystemPreconditionError("clean checkout readback is not reproducible")

    handoff = validate_concurrent_task_handoff(
        object_resolver(payload["concurrent_task_handoff_ref"])
    )
    disposition = validate_legacy_source_disposition(
        object_resolver(payload["legacy_disposition_ref"])
    )
    object_resolver(payload["historical_dirty_evidence_ref"])
    handoff_payload = handoff["payload"]
    if handoff_payload["accepted_baseline_commit"] != payload["accepted_baseline_commit"]:
        raise SystemPreconditionError("handoff baseline differs from final authorization")
    task_commit = handoff_payload["task_commit"]
    if _git_scalar(root, "rev-parse", f"{task_commit}^{{tree}}") != handoff_payload["task_tree"]:
        raise SystemPreconditionError("handoff task tree differs from Git")

    required_ancestors = {
        payload["accepted_baseline_commit"],
        payload["historical_integration_commit"],
        task_commit,
    }
    claimed_pairs = {(row["ancestor"], row["descendant"]) for row in payload["ancestry_rows"]}
    for ancestor in required_ancestors:
        if (ancestor, final_commit) not in claimed_pairs or not _is_ancestor(
            root, ancestor, final_commit
        ):
            raise SystemPreconditionError("required final ancestry is not proved by Git")
    for row in payload["ancestry_rows"]:
        if not _is_ancestor(root, row["ancestor"], row["descendant"]):
            raise SystemPreconditionError("claimed final ancestry differs from Git")
    for row in payload["excluded_commit_rows"]:
        if _is_ancestor(root, row["commit"], row["descendant"]):
            raise SystemPreconditionError("excluded commit is an ancestor of final release")

    disposition_by_source = {row["source_path"]: row for row in disposition["payload"]["rows"]}
    for row in handoff_payload["path_rows"]:
        task_mode, task_oid, task_raw = _git_blob(root, task_commit, row["path"])
        if (
            task_mode != row["mode"]
            or task_oid != row["git_blob_oid"]
            or len(task_raw) != row["size"]
            or _sha256(task_raw) != row["byte_sha256"]
        ):
            raise SystemPreconditionError("handoff path does not match task commit")
        try:
            final_mode, final_oid, final_raw = _git_blob(root, final_commit, row["path"])
        except SystemPreconditionError:
            final_mode = final_oid = ""
            final_raw = b""
        if (final_mode, final_oid, final_raw) == (task_mode, task_oid, task_raw):
            continue
        disposition_row = disposition_by_source.get(row["path"])
        if disposition_row is None or disposition_row["classification"] != "PORTED_TO_STABLE":
            raise SystemPreconditionError("task-owned path is not preserved or ported")
        stable_mode, stable_oid, _ = _git_blob(
            root, final_commit, disposition_row["stable_target_path"]
        )
        if stable_mode not in {"100600", "100644", "100755"} or stable_oid != (
            disposition_row["stable_target_blob_oid"]
        ):
            raise SystemPreconditionError("ported task path target differs from disposition")

    for row in disposition["payload"]["rows"]:
        _mode, source_oid, _raw = _git_blob(
            root, disposition["payload"]["source_commit"], row["source_path"]
        )
        if source_oid != row["source_blob_oid"]:
            raise SystemPreconditionError("legacy source disposition blob differs")
        if row["classification"] == "PORTED_TO_STABLE":
            _mode, target_oid, _raw = _git_blob(root, final_commit, row["stable_target_path"])
            if target_oid != row["stable_target_blob_oid"]:
                raise SystemPreconditionError("stable port disposition target differs")

    if any(
        row["path"].startswith(
            ("quant_investor/v17_v4_runtime/", "quant_investor/v17_v4_contract/")
        )
        for row in inventory
    ):
        raise SystemPreconditionError("active V17 package remains in final tree")

    observed_gates: set[str] = set()
    _mode, _runner_oid, runner_code = _git_blob(
        root, final_commit, "quant_investor/migration/authority.py"
    )
    runner_code_sha = _sha256(runner_code)
    for row in payload["preflight_rows"]:
        resolved_evidence = object_resolver(row["evidence_ref"])
        if validation_mode == "PRE_CAS_CURRENT":
            evidence = validate_cutover_gate_evidence(resolved_evidence)
        else:
            evidence = _validate_historical_cutover_gate_evidence(
                resolved_evidence,
                authorized_runner_source=runner_code,
            )
        evidence_payload = evidence["payload"]
        if (
            object_ref_for_artifact(evidence) != row["evidence_ref"]
            or evidence_payload["gate_id"] != row["gate_id"]
            or evidence_payload["final_commit"] != final_commit
            or evidence_payload["final_tree"] != final_tree
            or evidence_payload["state"] != "PASS"
            or evidence_payload["runner_id"] != _GATE_RUNNER_ID
            or evidence_payload["runner_code_sha256"] != runner_code_sha
            or any(batch["exit_code"] != 0 for batch in evidence_payload["batch_results"])
        ):
            raise SystemPreconditionError("final preflight evidence binding differs")
        if validation_mode == "PRE_CAS_CURRENT" and evidence_payload[
            "runner_spec_sha256"
        ] != _gate_runner_spec_sha256(row["gate_id"], _GATE_SPECS[row["gate_id"]]):
            raise SystemPreconditionError("current cutover gate runner spec differs")
        if validation_mode == "PRE_CAS_CURRENT":
            for batch in evidence_payload["batch_results"]:
                try:
                    executable_raw = (
                        Path(batch["executable_path"]).resolve(strict=True).read_bytes()
                    )
                except OSError as exc:
                    raise SystemPreconditionError(
                        "cutover gate executable readback failed"
                    ) from exc
                if _sha256(executable_raw) != batch["executable_sha256"]:
                    raise SystemPreconditionError("cutover gate executable identity drifted")
        subject = object_resolver(evidence_payload["subject_ref"])
        if object_ref_for_artifact(subject) != evidence_payload["subject_ref"]:
            raise SystemPreconditionError("cutover gate subject exact object differs")
        if row["gate_id"] == "release_install_origin":
            install_evidence = validate_release_install_evidence(subject)
            if install_evidence["payload"]["release_ref"] != normalized_release:
                raise SystemPreconditionError("release install gate release differs")
            if (
                install_evidence["payload"]["code_tree_sha256"] != release["payload"]["code_sha256"]
                or install_evidence["payload"]["wheel"]["byte_sha256"]
                != release["payload"]["wheel_sha256"]
                or install_evidence["payload"]["installed_code_manifest_sha256"]
                != release["payload"]["code_manifest_sha256"]
            ):
                raise SystemPreconditionError("release install evidence identity differs")
            exact_input = canonical_json_bytes(
                {
                    "release_install_evidence": install_evidence,
                    "deployed_release": release,
                }
            )
            batches = evidence_payload["batch_results"]
            if len(batches) != 1 or batches[0]["stdin_sha256"] != _sha256(exact_input):
                raise SystemPreconditionError("release install gate exact input differs")
            try:
                output = parse_canonical_json_bytes(
                    base64.b64decode(batches[0]["stdout_base64"], validate=True)
                )
            except (ContractError, ValueError, TypeError) as exc:
                raise SystemPreconditionError("release install gate output is invalid") from exc
            if (
                type(output) is not dict
                or set(output)
                != {
                    "state",
                    "release_ref",
                    "source_archive_sha256",
                    "wheel_sha256",
                    "code_tree_sha256",
                    "installed_code_manifest_sha256",
                    "contract_catalog_sha256",
                    "import_origin",
                }
                or output["state"] != "PASS"
                or output["release_ref"] != normalized_release
                or output["source_archive_sha256"]
                != install_evidence["payload"]["source_archive"]["byte_sha256"]
                or output["wheel_sha256"] != install_evidence["payload"]["wheel"]["byte_sha256"]
                or output["code_tree_sha256"] != install_evidence["payload"]["code_tree_sha256"]
                or output["installed_code_manifest_sha256"]
                != install_evidence["payload"]["installed_code_manifest_sha256"]
                or output["contract_catalog_sha256"]
                != install_evidence["payload"]["contract_catalog_sha256"]
                or output["import_origin"] != install_evidence["payload"]["import_origin"]
            ):
                raise SystemPreconditionError("release install gate output binding differs")
        observed_gates.add(row["gate_id"])
    if observed_gates != REQUIRED_FINAL_PREFLIGHT_GATES:
        raise SystemPreconditionError("final preflight evidence set is incomplete")
    return artifact


def publish_authority_artifact(  # noqa: C901
    authority_root: str | os.PathLike[str], document: Mapping[str, Any]
) -> Path:
    kind = document.get("kind")
    if kind not in AUTHORITY_KINDS:
        raise SystemContractError("authority artifact kind is not permitted")
    validators = {
        CONCURRENT_HANDOFF_KIND: validate_concurrent_task_handoff,
        LEGACY_DISPOSITION_KIND: validate_legacy_source_disposition,
        FINAL_AUTHORIZATION_KIND: validate_final_cutover_authorization,
        GATE_EVIDENCE_KIND: validate_cutover_gate_evidence,
        RELEASE_INSTALL_EVIDENCE_KIND: validate_release_install_evidence,
    }
    artifact = validators[kind](document)
    raw = canonical_json_bytes(artifact)
    byte_sha = _sha256(raw)
    root = Path(authority_root)
    current = root
    if current.exists():
        metadata = os.lstat(current)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise SystemSecurityError("authority evidence root is not owner-only")
    else:
        current.mkdir(parents=True, mode=0o700)
        current.chmod(0o700)
    kind_root = current / kind
    if not kind_root.exists():
        kind_root.mkdir(mode=0o700)
    metadata = os.lstat(kind_root)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise SystemSecurityError("authority evidence kind root is not owner-only")
    target = kind_root / f"{byte_sha}.json"
    if target.exists():
        observed = target.read_bytes()
        if observed != raw:
            raise SystemStorageError("authority evidence exact-once conflict")
        return target
    temporary = kind_root / f".{byte_sha}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, target, follow_symlinks=False)
    except FileExistsError as exc:
        raise SystemStorageError("authority evidence exact-once conflict") from exc
    finally:
        temporary.unlink(missing_ok=True)
    os.chmod(target, 0o600, follow_symlinks=False)
    metadata = os.lstat(target)
    if (
        metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or target.read_bytes() != raw
    ):
        raise SystemStorageError("authority evidence readback failed")
    return target


__all__ = [
    "AUTHORITY_KINDS",
    "CONCURRENT_HANDOFF_KIND",
    "FINAL_AUTHORIZATION_KIND",
    "LEGACY_DISPOSITION_KIND",
    "REQUIRED_FINAL_PREFLIGHT_GATES",
    "build_concurrent_task_handoff",
    "build_final_cutover_authorization",
    "build_legacy_source_disposition",
    "publish_authority_artifact",
    "run_cutover_gate",
    "validate_concurrent_task_handoff",
    "validate_cutover_gate_evidence",
    "validate_final_cutover_authorization",
    "validate_final_cutover_authorization_closure",
    "validate_legacy_source_disposition",
]
