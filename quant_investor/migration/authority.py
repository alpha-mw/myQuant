"""Detached exact-byte authority evidence for the unified cutover.

These artifacts live outside the release tree.  They record observations and
authorization, but never write the System active pointer and never grant any
trading-side authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
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
from quant_investor.system.store import validate_object_ref


CONCURRENT_HANDOFF_KIND: Final = "system.concurrent_task_handoff"
LEGACY_DISPOSITION_KIND: Final = "system.legacy_source_disposition"
FINAL_AUTHORIZATION_KIND: Final = "system.final_cutover_authorization"
AUTHORITY_KINDS: Final = frozenset(
    {CONCURRENT_HANDOFF_KIND, LEGACY_DISPOSITION_KIND, FINAL_AUTHORIZATION_KIND}
)
_CONTRACT_SHA256S: Final = {
    kind: get_contract(kind).contract_sha256 for kind in AUTHORITY_KINDS
}
_GIT_OID_RE: Final = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_THREAD_RE: Final = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_PATH_ROW_FIELDS: Final = frozenset(
    {"path", "status", "mode", "size", "git_blob_oid", "byte_sha256"}
)
_TEST_ROW_FIELDS: Final = frozenset(
    {"command", "exit_code", "stdout_sha256", "status"}
)
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
_PREFLIGHT_ROW_FIELDS: Final = frozenset(
    {"gate_id", "status", "evidence_sha256"}
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
                "byte_sha256": _sha(
                    row["byte_sha256"], label=f"path_rows[{index}].byte_sha256"
                ),
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
                "observed_at": _text(
                    row["observed_at"], label=f"{label}[{index}].observed_at"
                ),
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
                "source_path": _path(
                    row["source_path"], label=f"disposition[{index}].source_path"
                ),
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


def _validate_preflight_rows(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list or not value:
        raise SystemContractError("final preflight rows are absent")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if type(raw) is not dict or set(raw) != _PREFLIGHT_ROW_FIELDS:
            raise SystemContractError("final preflight row fields are not exact")
        if raw["status"] != "PASS":
            raise SystemPreconditionError("final cutover preflight did not pass")
        rows.append(
            {
                "gate_id": _text(raw["gate_id"], label=f"preflight[{index}].gate_id"),
                "status": "PASS",
                "evidence_sha256": _sha(
                    raw["evidence_sha256"],
                    label=f"preflight[{index}].evidence_sha256",
                ),
            }
        )
    gate_ids = [row["gate_id"] for row in rows]
    if gate_ids != sorted(set(gate_ids)):
        raise SystemContractError("final preflight gate ids are not sorted unique")
    return rows


def build_final_cutover_authorization(  # noqa: C901
    *,
    final_authorization_id: str,
    accepted_baseline_commit: str,
    historical_integration_commit: str,
    historical_dirty_evidence_ref: Mapping[str, Any],
    concurrent_task_handoff_ref: Mapping[str, Any],
    legacy_disposition_ref: Mapping[str, Any],
    final_integration_commit: str,
    final_integration_tree: str,
    ancestry_rows: Sequence[Mapping[str, Any]],
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
                "ancestor": _git_oid(
                    row["ancestor"], label=f"ancestry[{index}].ancestor"
                ),
                "descendant": _git_oid(
                    row["descendant"], label=f"ancestry[{index}].descendant"
                ),
                "proved": True,
            }
        )
    if not ancestry or ancestry != sorted(
        ancestry, key=lambda row: (row["ancestor"], row["descendant"])
    ):
        raise SystemContractError("final ancestry rows are not exact sorted")
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
            "final_authorization_id": _text(
                final_authorization_id, label="final_authorization_id"
            ),
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
            "final_integration_commit": _git_oid(
                final_integration_commit, label="final_integration_commit"
            ),
            "final_integration_tree": _git_oid(
                final_integration_tree, label="final_integration_tree"
            ),
            "ancestry_rows": ancestry,
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
    if payload.get("final_build_authorized") is not True or payload.get(
        "cas_authorized"
    ) is not True:
        raise SystemPreconditionError("final cutover authorization is not machine-authorized")
    rebuilt = build_final_cutover_authorization(
        final_authorization_id=payload["final_authorization_id"],
        accepted_baseline_commit=payload["accepted_baseline_commit"],
        historical_integration_commit=payload["historical_integration_commit"],
        historical_dirty_evidence_ref=payload["historical_dirty_evidence_ref"],
        concurrent_task_handoff_ref=payload["concurrent_task_handoff_ref"],
        legacy_disposition_ref=payload["legacy_disposition_ref"],
        final_integration_commit=payload["final_integration_commit"],
        final_integration_tree=payload["final_integration_tree"],
        ancestry_rows=payload["ancestry_rows"],
        final_worktree_inventory_sha256=payload["final_worktree_inventory_sha256"],
        clean_checkout_readback_rows=payload["clean_checkout_readback_rows"],
        user_authorization_basis=payload["user_authorization_basis"],
        preflight_rows=payload["preflight_rows"],
        created_at=artifact["created_at"],
    )
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(artifact):
        raise SystemContractError("final cutover authorization semantic replay differs")
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
    "build_concurrent_task_handoff",
    "build_final_cutover_authorization",
    "build_legacy_source_disposition",
    "publish_authority_artifact",
    "validate_concurrent_task_handoff",
    "validate_final_cutover_authorization",
    "validate_legacy_source_disposition",
]
