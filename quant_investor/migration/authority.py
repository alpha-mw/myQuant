"""Detached exact-byte authority evidence for the unified cutover.

These artifacts live outside the release tree.  They record observations and
authorization, but never write the System active pointer and never grant any
trading-side authority.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import subprocess
from typing import Any, Final

from quant_investor.contracts import (
    ContractError,
    canonical_json_bytes,
    get_contract,
    seal_artifact,
    validate_artifact,
)
from quant_investor.system.errors import (
    SystemContractError,
    SystemPreconditionError,
    SystemSecurityError,
    SystemStorageError,
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


def build_cutover_gate_evidence(
    *,
    gate_id: str,
    final_commit: str,
    final_tree: str,
    command: str,
    exit_code: int,
    stdout_sha256: str,
    subject_ref: Mapping[str, Any],
    observed_at: str,
) -> dict[str, Any]:
    """Seal one exact, commit-bound machine gate result.

    The final authorization accepts only the complete fixed gate set and
    resolves every subject reference again.  A non-zero result cannot be
    represented as an authorized gate.
    """

    gate = _text(gate_id, label="gate_id")
    if gate not in REQUIRED_FINAL_PREFLIGHT_GATES:
        raise SystemContractError("cutover gate id is not in the fixed allowlist")
    if type(exit_code) is not int or exit_code != 0:
        raise SystemPreconditionError("cutover gate command did not pass")
    body = {
        "state": "PASS",
        "gate_id": gate,
        "final_commit": _git_oid(final_commit, label="final_commit"),
        "final_tree": _git_oid(final_tree, label="final_tree"),
        "command": _text(command, label="command"),
        "exit_code": 0,
        "stdout_sha256": _sha(stdout_sha256, label="stdout_sha256"),
        "subject_ref": validate_object_ref(subject_ref, label="subject_ref"),
        "observed_at": _text(observed_at, label="observed_at"),
    }
    evidence_id = "cutover-gate-" + _sha256(canonical_json_bytes(body))
    return seal_artifact(
        GATE_EVIDENCE_KIND,
        {**body, "evidence_id": evidence_id},
        created_at=observed_at,
    )


def validate_cutover_gate_evidence(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    artifact = _artifact(document, GATE_EVIDENCE_KIND)
    payload = artifact["payload"]
    rebuilt = build_cutover_gate_evidence(
        gate_id=payload["gate_id"],
        final_commit=payload["final_commit"],
        final_tree=payload["final_tree"],
        command=payload["command"],
        exit_code=payload["exit_code"],
        stdout_sha256=payload["stdout_sha256"],
        subject_ref=payload["subject_ref"],
        observed_at=payload["observed_at"],
    )
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(artifact):
        raise SystemContractError("cutover gate evidence semantic replay differs")
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
    rebuilt = build_final_cutover_authorization(
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
) -> dict[str, Any]:
    """Independently replay the final Git/evidence/release authority closure."""

    artifact = validate_final_cutover_authorization(document)
    payload = artifact["payload"]
    root = Path(repository_root).resolve(strict=True)
    top = Path(_git(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()).resolve()
    if top != root:
        raise SystemPreconditionError("final authorization repository root differs")
    final_commit = payload["final_integration_commit"]
    final_tree = payload["final_integration_tree"]
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

    status_raw = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if status_raw:
        raise SystemPreconditionError("authorized final checkout is not clean")
    inventory, inventory_sha = _git_inventory(root, final_commit)
    if inventory_sha != payload["final_worktree_inventory_sha256"]:
        raise SystemPreconditionError("authorized final tree inventory differs")
    for row in payload["clean_checkout_readback_rows"]:
        if (
            row["commit"] != final_commit
            or row["tree"] != final_tree
            or row["status_porcelain_sha256"] != _sha256(status_raw)
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
    for row in payload["preflight_rows"]:
        evidence = validate_cutover_gate_evidence(object_resolver(row["evidence_ref"]))
        evidence_payload = evidence["payload"]
        if (
            evidence_payload["gate_id"] != row["gate_id"]
            or evidence_payload["final_commit"] != final_commit
            or evidence_payload["final_tree"] != final_tree
            or evidence_payload["state"] != "PASS"
            or evidence_payload["exit_code"] != 0
        ):
            raise SystemPreconditionError("final preflight evidence binding differs")
        object_resolver(evidence_payload["subject_ref"])
        if row["gate_id"] == "release_install_origin" and (
            evidence_payload["subject_ref"] != normalized_release
        ):
            raise SystemPreconditionError("release install gate subject differs")
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
    "build_cutover_gate_evidence",
    "build_final_cutover_authorization",
    "build_legacy_source_disposition",
    "publish_authority_artifact",
    "validate_concurrent_task_handoff",
    "validate_cutover_gate_evidence",
    "validate_final_cutover_authorization",
    "validate_final_cutover_authorization_closure",
    "validate_legacy_source_disposition",
]
